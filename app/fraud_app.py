"""
FraudLens: our fraud detection prototype.

Run from the repository folder:
    pip install -r app/requirements.txt
    streamlit run app/fraud_app.py

Steps in the app:
  1. Load a transaction table (CSV upload, a demo file or a file path).
  2. Tell the app which column plays which role.
  3. Check the columns for leakage, train a model with our leak-free protocol,
     and explore the results: review budget, threshold, features, time split.
  4. Review the alerts: the riskiest test transactions, each with its card's history.
  5. Score new transactions with the trained model.

The model logic lives in prototype.py; the look and the charts live in ui.py.
"""

import glob
import os
import sys

import numpy as np
import pandas as pd
import streamlit as st

APP_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(APP_DIR)
from prototype import (ROLES, FEATURES, guess_columns, available_features, prepare,  # noqa: E402
                       sample_cards, time_split, leak_check, train_and_evaluate,
                       alert_budgets, score, explain, point_metrics, to_bytes, from_bytes)
import ui  # noqa: E402

st.set_page_config(page_title="FraudLens", page_icon="🛡️", layout="wide", initial_sidebar_state="auto")
ui.inject_css()
ss = st.session_state

STEPS = ["Load data", "Match columns", "Train and evaluate", "Review alerts", "Score new data"]
FEATURE_GROUP = {f: g for f, (g, _, _) in FEATURES.items()}
ROLE_GROUPS = {
    "Needed to train": ["label", "time", "card"],
    "Transaction": ["amount", "category"],
    "Customer": ["age", "dob", "gender"],
    "Location": ["home_lat", "home_long", "merch_lat", "merch_long", "home_state", "merchant_state"],
}
RESULT_KEYS = ["bundle", "test", "pr_curve", "importance", "split_info", "leaks", "t_range"]


@st.cache_data(show_spinner=False)
def read_csv(source):
    return pd.read_csv(source, low_memory=False)


def go(step):
    ss["step"] = step


def demo_files():
    return sorted(glob.glob(os.path.join(APP_DIR, "demo_data", "*.csv")))


def load_table(df, label):
    ss["raw"] = df
    ss["source_name"] = label
    ss["mapping"] = guess_columns(df)
    for role in ROLES:
        ss.pop(f"role_{role}", None)
    for key in RESULT_KEYS:
        ss.pop(key, None)


def card_label(card):
    """Show long card numbers the way a statement does: only the last four digits."""
    c = str(card)
    return f"Card ending {c[-4:]}" if c.isdigit() and len(c) > 8 else f"Card {c}"


def fmt_money(x):
    return f"${x:,.0f}" if abs(x) >= 1000 else f"${x:,.2f}"


# ---------------------------------------------------------------------------
# Sidebar: brand, the five steps, and what is ready
# ---------------------------------------------------------------------------
ss.setdefault("step", STEPS[0])
done = {
    STEPS[0]: "raw" in ss,
    STEPS[1]: "raw" in ss and bool(ss.get("mapping", {}).get("label")) and bool(ss.get("mapping", {}).get("time")),
    STEPS[2]: "bundle" in ss and "test" in ss,
    STEPS[3]: False,
    STEPS[4]: False,
}

with st.sidebar:
    st.markdown('<p class="fl-brand">FraudLens<span></span></p>'
                '<p class="fl-tag">Card fraud detection prototype<br>COMP8851 Major Project, Group 2</p>',
                unsafe_allow_html=True)
    st.radio("Steps", STEPS, key="step", label_visibility="collapsed",
             format_func=lambda s: f"{STEPS.index(s) + 1}   {s}" + ("   ✓" if done.get(s) else ""))
    st.divider()
    lines = []
    if "raw" in ss:
        lines.append(f"<b>Data</b> {ui.esc(ss.get('source_name', 'table'))}, {len(ss['raw']):,} rows")
    if "bundle" in ss:
        b = ss["bundle"]
        lines.append(f"<b>Model</b> {len(b['features'])} features, threshold {b['threshold']:.2f}")
        if "metrics" in b and not np.isnan(b["metrics"].get("PR-AUC", np.nan)):
            lines.append(f"<b>Test PR-AUC</b> {b['metrics']['PR-AUC']:.3f}")
    if not lines:
        lines.append("Nothing loaded yet. Start with step 1.")
    st.markdown('<div class="fl-status">' + "<br>".join(lines) + "</div>", unsafe_allow_html=True)
    if "bundle" in ss:
        st.download_button("Download the trained model", to_bytes(ss["bundle"]), file_name="fraudlens_model.pkl",
                           width="stretch")
    with st.expander("How FraudLens works"):
        st.markdown(
            "- Splits the table **by time**: train, then validation, then test\n"
            "- Learns the category fraud rate from the **training part only**\n"
            "- Random Forest with our tuned settings\n"
            "- Picks the alert threshold with the best **F2** on validation\n"
            "- Reports PR-AUC, F2, MCC and alert budgets on the **test part**")

step = ss["step"]


# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
if step == STEPS[0]:
    if "raw" not in ss:
        left, right = st.columns([1.15, 1], gap="large")
        with left:
            st.markdown(
                '<div class="fl-hero"><h1>Find the fraud in a card transaction table.</h1>'
                "<p>Load your transactions and FraudLens learns what fraud looks like from your own history: "
                "bursts of purchases minutes apart, unusual amounts, risky merchants, spending far from home. "
                "Then it ranks new transactions so analysts review the riskiest first.</p></div>",
                unsafe_allow_html=True)
        with right:
            st.markdown(ui.LEDGER, unsafe_allow_html=True)
        st.markdown(
            '<div class="fl-steps">'
            "<div><b>Load data</b>One row per card transaction, with a fraud label.</div>"
            "<div><b>Match columns</b>Tell FraudLens which column is the time, card, amount and so on.</div>"
            "<div><b>Train and evaluate</b>A leak-free split by time, then results on unseen months.</div>"
            "<div><b>Review alerts</b>The riskiest transactions, each with its card's history and reasons.</div>"
            "<div><b>Score new data</b>Use the trained model on a fresh table.</div>"
            "</div>", unsafe_allow_html=True)
        st.write("")

    st.subheader("Load a transaction table" if "raw" not in ss else "Your table")
    if "raw" not in ss or st.toggle("Load a different table", value=False):
        c1, c2 = st.columns([1.3, 1], gap="large")
        with c1:
            upload = st.file_uploader("Upload a CSV file", type=["csv"],
                                      help="Each row is one card transaction. Training needs a fraud label column.")
            if upload is not None and st.button("Load this file", type="primary"):
                try:
                    load_table(read_csv(upload), upload.name)
                    st.rerun()
                except Exception as e:
                    st.error(f"FraudLens could not read this file: {e}")
        with c2:
            demos = demo_files()
            if demos:
                pick = st.selectbox("Or use a demo file", demos, format_func=os.path.basename)
                if st.button("Load demo file"):
                    load_table(read_csv(pick), os.path.basename(pick))
                    st.rerun()
            else:
                st.caption("Demo files appear here after you run `app/prototype_demo.ipynb`.")
            path = st.text_input("Or type the path of a CSV file", placeholder="app/demo_data/sparkov_sample.csv")
            if path and st.button("Load from path"):
                try:
                    load_table(read_csv(path), os.path.basename(path))
                    st.rerun()
                except Exception as e:
                    st.error(f"FraudLens could not read {path}: {e}")

    if "raw" in ss:
        raw, m = ss["raw"], ss["mapping"]
        c = st.columns(4)
        c[0].metric("Transactions", f"{len(raw):,}")
        c[1].metric("Columns", raw.shape[1])
        if m.get("card"):
            c[2].metric("Cards", f"{raw[m['card']].nunique():,}")
        y = None
        if m.get("label"):
            from prototype import _to_label
            y = _to_label(raw[m["label"]])
            c[3].metric("Fraud rate", f"{y.mean():.2%}", help=f"{int(y.sum()):,} fraud transactions, using the column '{m['label']}'.")
        if y is not None and m.get("time"):
            from prototype import _to_time
            t = _to_time(raw[m["time"]])
            ok = t.notna()
            if ok.any():
                with st.container(border=True, key="panel_weekly"):
                    ui.show(ui.weekly_fraud_chart(t[ok], y[ok]))
        with st.expander("Preview the first 50 rows", expanded=False):
            st.dataframe(raw.head(50), width="stretch")
        st.button("Continue to match columns", type="primary", on_click=go, args=(STEPS[1],))


# ---------------------------------------------------------------------------
# 2. Match columns
# ---------------------------------------------------------------------------
elif step == STEPS[1]:
    st.title("Match columns")
    if "raw" not in ss:
        ui.empty_state("No table loaded yet.", "Load a transaction table in step 1, then come back here.")
        st.button("Go to step 1", on_click=go, args=(STEPS[0],))
    else:
        raw = ss["raw"]
        st.markdown('<p class="fl-sub">FraudLens guessed these from the column names. Fix any that are wrong, '
                    "and choose <i>(none)</i> for anything your table does not have.</p>", unsafe_allow_html=True)
        options = ["(none)"] + list(raw.columns)
        new_map = dict(ss["mapping"])
        left, right = st.columns([1.25, 1], gap="large")
        with left:
            for gi, (group, roles) in enumerate(ROLE_GROUPS.items()):
                with st.container(border=True, key=f"panel_roles_{gi}"):
                    st.markdown(f"**{group}**")
                    cols = st.columns(2)
                    for i, role in enumerate(roles):
                        desc = ROLES[role][0]
                        current = ss["mapping"].get(role)
                        choice = cols[i % 2].selectbox(desc, options, key=f"role_{role}",
                                                       index=options.index(current) if current in options else 0)
                        new_map[role] = None if choice == "(none)" else choice
        ss["mapping"] = new_map
        feats = available_features(new_map)
        with right:
            st.subheader("Features FraudLens can build")
            st.markdown(f'<p class="fl-sub">{len(feats)} of {len(FEATURES)} with these columns. '
                        "Dashed ones need a column you have not matched.</p>", unsafe_allow_html=True)
            for group in ["Temporal", "Spatial", "Amount", "Customer"]:
                items = [(f, "on" if f in feats else "off") for f, (g, _, _) in FEATURES.items() if g == group]
                st.markdown(f'<div class="fl-group">{group}</div>' + ui.chips(items), unsafe_allow_html=True)
            with st.expander("What each feature means"):
                st.dataframe(pd.DataFrame([{"Feature": f, "Meaning": d} for f, (_, _, d) in FEATURES.items()]),
                             hide_index=True, width="stretch")
            if not new_map.get("label"):
                st.warning("Without a fraud label you can only score with a model you trained earlier (step 5).")
            elif not new_map.get("time"):
                st.warning("Training needs a time column, to split by time and to build the time-gap feature.")
            else:
                st.button("Continue to train and evaluate", type="primary", on_click=go, args=(STEPS[2],))


# ---------------------------------------------------------------------------
# 3. Train and evaluate
# ---------------------------------------------------------------------------
elif step == STEPS[2]:
    st.title("Train and evaluate")
    m = ss.get("mapping", {})
    if "raw" not in ss or not m.get("label") or not m.get("time"):
        ui.empty_state("Training needs a table with a fraud label and a time column.",
                       "Load a table in step 1 and match at least those two columns in step 2.")
        st.button("Go to step 2", on_click=go, args=(STEPS[1],))
    else:
        feats = available_features(m)
        with st.container(border=True, key="panel_2"):
            c1, c2, c3 = st.columns([2.2, 1, 1], gap="medium")
            chosen = c1.multiselect("Features for the model", feats, default=feats)
            trees = c2.select_slider("Trees", [50, 100, 200], value=100,
                                     help="200 is the setting from our experiments; 100 trains faster.")
            max_rows = c3.select_slider("Largest table to use", [100_000, 250_000, 500_000, 1_000_000, 2_000_000],
                                        value=500_000, format_func=lambda v: f"{v // 1000:,}k rows",
                                        help="Larger tables are cut to a random sample of whole cards.")
            run = st.button("Train the model", type="primary", disabled=not chosen)

        if run:
            with st.status("Training", expanded=True) as status:
                st.write("Building features from your columns")
                d = sample_cards(prepare(ss["raw"], m), max_rows)
                train_sub, _, test_part, _, _ = time_split(d)
                st.write("Checking each raw column for leakage")
                skip = {m[r] for r in ROLES if m.get(r)} | {"_time", "_y", "_card", "_amount", "_gap_h"} | set(FEATURES)
                raw_cols = [c for c in ss["raw"].columns if c not in skip and c in d.columns]
                ss["leaks"] = leak_check(train_sub, test_part, raw_cols)
                st.write(f"Training a Random Forest with {trees} trees on {len(train_sub):,} transactions")
                try:
                    bundle, test, pr_curve, importance, split_info = train_and_evaluate(d, m, chosen, n_estimators=trees)
                    ss.update(bundle=bundle, test=test, pr_curve=pr_curve, importance=importance,
                              split_info=split_info, t_range=(d["_time"].min(), d["_time"].max()))
                    ss.pop("budget", None)
                    ss.pop("thr", None)
                    status.update(label="Model trained", state="complete", expanded=False)
                    st.rerun()
                except ValueError as e:
                    status.update(label="Training stopped", state="error")
                    st.error(f"FraudLens could not train: {e}")

        if "bundle" in ss and "test" in ss:
            bundle, test, met = ss["bundle"], ss["test"], ss["bundle"]["metrics"]
            y, s = test["_y"].values, test["fraud_score"].values
            amount = test["_amount"].values if "_amount" in test else None

            # ---- the review budget: the main readout ----
            st.write("")
            with st.container(border=True, key="panel_3"):
                st.subheader("How much fraud does a review budget catch?")
                left, right = st.columns([1, 1.35], gap="large")
                with left:
                    ss.setdefault("budget", 1.0)
                    budget = st.slider("Share of test transactions your analysts review", 0.1, 5.0,
                                       step=0.1, format="%.1f%%", key="budget") / 100
                    r = ui.budget_readout(y, s, amount, budget)
                    money = ""
                    if amount is not None and r.get("dollars_total"):
                        money = (f" and <strong>{fmt_money(r['dollars_caught'])}</strong> of the "
                                 f"{fmt_money(r['dollars_total'])} in fraud")
                    st.markdown(
                        f'<p class="fl-budget">Review the top <em>{r["alerts"]:,}</em> transactions and catch '
                        f'<strong>{r["caught"]:,} of {r["frauds"]:,}</strong> frauds{money}.</p>'
                        f'<p class="fl-sub">That is {r["recall"]:.0%} of fraud cases. {r["precision"]:.0%} of the '
                        f'reviewed transactions are fraud, against {met["Test fraud rate"]:.2%} if analysts picked '
                        "at random.</p>", unsafe_allow_html=True)
                with right:
                    ui.show(ui.capture_chart(ui.capture_curve(y, s, amount, max_share=max(0.06, budget * 1.6)), budget))

            # ---- headline metrics ----
            lift = met["PR-AUC"] / max(met["Test fraud rate"], 1e-9)
            c = st.columns(4)
            c[0].metric("PR-AUC", f"{met['PR-AUC']:.3f}", f"{lift:,.0f}× random", delta_color="off",
                        help=f"Random ranking scores about {met['Test fraud rate']:.4f}, the test fraud rate.")
            c[1].metric("F2", f"{met['F2']:.3f}", help="Weights recall above precision, at the alert threshold.")
            c[2].metric("MCC", f"{met['MCC']:.3f}", help="Balanced score from -1 to 1 at the alert threshold.")
            c[3].metric("ROC-AUC", f"{met['ROC-AUC']:.3f}", help="Can look high even for weak fraud models; read with PR-AUC.")

            # ---- threshold explorer ----
            with st.container(border=True, key="panel_4"):
                st.subheader("Where should alerts start?")
                left, right = st.columns([1, 1.35], gap="large")
                with left:
                    ss.setdefault("thr", float(bundle["threshold"]))
                    thr = st.slider("Alert threshold (fraud score)", 0.01, 0.99,
                                    step=0.01, key="thr",
                                    help=f"FraudLens chose {bundle['threshold']:.2f}: the best F2 on the validation months.")
                    pm = point_metrics(y, s, thr)
                    pred = s >= thr
                    caught, missed, false_alarm = int((pred & (y == 1)).sum()), int((~pred & (y == 1)).sum()), int((pred & (y == 0)).sum())
                    st.markdown(
                        '<div class="fl-tally">'
                        f'<div><b class="hit">{caught:,}</b>fraud caught</div>'
                        f'<div><b>{missed:,}</b>fraud missed</div>'
                        f'<div><b class="alarm">{false_alarm:,}</b>false alarms</div></div>', unsafe_allow_html=True)
                    st.markdown(f'<p class="fl-sub">Recall {pm["Recall"]:.1%}, precision {pm["Precision"]:.1%}, '
                                f'F2 {pm["F2"]:.3f}. The open circle on the curve is the threshold chosen on validation.</p>',
                                unsafe_allow_html=True)
                    if abs(thr - bundle["threshold"]) > 1e-9:
                        if st.button(f"Use {thr:.2f} for alerts and scoring"):
                            bundle["threshold"] = thr
                            ss["test"] = test.assign(alert=(test["fraud_score"] >= thr).astype(int))
                            st.toast(f"Alerts now start at a fraud score of {thr:.2f}")
                            st.rerun()
                with right:
                    vm = point_metrics(y, s, met["Threshold"])
                    ui.show(ui.pr_chart(ss["pr_curve"], pm["Recall"], pm["Precision"], vm["Recall"], vm["Precision"]))

            # ---- features and time split ----
            left, right = st.columns(2, gap="large")
            with left:
                with st.container(border=True, key="panel_5"):
                    st.subheader("What the model relies on")
                    ui.show(ui.importance_chart(ss["importance"], FEATURE_GROUP))
            with right:
                with st.container(border=True, key="panel_6"):
                    st.subheader("How the data was split")
                    st.markdown('<p class="fl-sub">The model never sees the test months while training, '
                                "the same way a bank would use it on next month's transactions.</p>", unsafe_allow_html=True)
                    t_min, t_max = ss["t_range"]
                    info = ss["split_info"]
                    ui.show(ui.split_chart(info, t_min, t_max))
                    st.markdown(
                        '<dl class="fl-facts fl-split">'
                        f'<dt>Train</dt><dd>{info["train rows"]:,} transactions, {info["train fraud"]:,} fraud</dd>'
                        f'<dt>Validation</dt><dd>{info["validation rows"]:,} transactions, {info["validation fraud"]:,} fraud</dd>'
                        f'<dt>Test</dt><dd>{info["test rows"]:,} transactions, {info["test fraud"]:,} fraud</dd></dl>',
                        unsafe_allow_html=True)
                    b = alert_budgets(y, s, amount)
                    st.dataframe(b, hide_index=True, width="stretch",
                                 column_config={"Recall": st.column_config.ProgressColumn("Recall", format="percent", min_value=0, max_value=1),
                                                "Dollar recall": st.column_config.ProgressColumn("Dollar recall", format="percent", min_value=0, max_value=1),
                                                "Precision": st.column_config.NumberColumn("Precision", format="percent")})

            # ---- leakage ----
            leaks = ss.get("leaks")
            with st.container(border=True, key="panel_7"):
                st.subheader("Leakage check")
                st.markdown('<p class="fl-sub">Each raw column on its own, scored on the test months. '
                            "A column near 1.0 almost gives the answer away and should be checked before you trust "
                            "the model.</p>", unsafe_allow_html=True)
                if leaks is None or not len(leaks):
                    st.caption("No other raw columns to check.")
                else:
                    if (leaks["ROC-AUC"] >= 0.95).any():
                        flagged = ", ".join(leaks.loc[leaks["ROC-AUC"] >= 0.95, "column"].astype(str))
                        st.warning(f"Check how these columns were created: {flagged}.")
                    st.dataframe(leaks, hide_index=True, width="stretch",
                                 column_config={"ROC-AUC": st.column_config.ProgressColumn("ROC-AUC on its own", format="%.3f", min_value=0.5, max_value=1)})
            st.button("Continue to review alerts", type="primary", on_click=go, args=(STEPS[3],))


# ---------------------------------------------------------------------------
# 4. Review alerts
# ---------------------------------------------------------------------------
elif step == STEPS[3]:
    st.title("Review alerts")
    if "test" not in ss:
        ui.empty_state("No alerts yet.", "Train a model in step 3. Its riskiest test transactions will appear here.")
        st.button("Go to step 3", on_click=go, args=(STEPS[2],))
    else:
        test, bundle = ss["test"], ss["bundle"]
        m = bundle["mapping"]
        n_alert = int(test["alert"].sum())
        st.markdown(f'<p class="fl-sub">{n_alert:,} of {len(test):,} test transactions score above the alert '
                    f'threshold of {bundle["threshold"]:.2f}. Click a row to open its card.</p>', unsafe_allow_html=True)

        f1, f2, f3 = st.columns([1, 1, 1.4])
        n = f1.select_slider("Show", [25, 50, 100, 250, 500], value=50, format_func=lambda v: f"top {v}")
        only = f2.toggle("Only above the threshold", value=True)
        cats = sorted(test[m["category"]].dropna().astype(str).unique()) if m.get("category") else []
        cat_pick = f3.multiselect("Merchant category", cats, placeholder="All categories") if cats else []

        view = test[test["alert"] == 1] if only else test
        if cat_pick:
            view = view[view[m["category"]].astype(str).isin(cat_pick)]
        top = view.sort_values("fraud_score", ascending=False).head(n).copy()

        if top.empty:
            ui.empty_state("No transactions match these filters.", "Turn off 'Only above the threshold' or pick other categories.")
        else:
            top["Reasons"] = top.apply(lambda r: explain(r, bundle["reference"]), axis=1)
            cols = list(dict.fromkeys([c for c in [m.get("time"), m.get("card"), m.get("amount"), m.get("category")] if c]))
            table = top[cols + ["fraud_score", "Reasons"] + (["_y"] if "_y" in top else [])].rename(
                columns={"_y": "Actual fraud", "fraud_score": "Fraud score"})
            if m.get("time") in table:
                table[m["time"]] = pd.to_datetime(top["_time"])
            if m.get("card") in table:
                table[m["card"]] = table[m["card"]].astype(str).map(lambda c: "…" + c[-4:] if c.isdigit() and len(c) > 8 else c)
            if "Actual fraud" in table:
                table["Actual fraud"] = table["Actual fraud"].map({1: "Fraud", 0: "Genuine"})

            left, right = st.columns([1.55, 1], gap="large")
            with left:
                event = st.dataframe(
                    table, hide_index=True, width="stretch", height=520,
                    on_select="rerun", selection_mode="single-row", key="queue",
                    column_config={"Fraud score": st.column_config.ProgressColumn("Fraud score", format="%.3f", min_value=0, max_value=1),
                                   "Reasons": st.column_config.TextColumn("Reasons", width="large"),
                                   **({m["time"]: st.column_config.DatetimeColumn(m["time"], format="D MMM YYYY, HH:mm")}
                                      if pd.api.types.is_datetime64_any_dtype(table[m["time"]]) else {})})
                st.download_button("Download these alerts as CSV", table.to_csv(index=False), file_name="fraudlens_alerts.csv")
            rows = event.selection.rows if event and event.selection else []
            pick = top.iloc[rows[0] if rows else 0]
            with right:
                with st.container(border=True, key="panel_8"):
                    sc = float(pick["fraud_score"])
                    card = pick["_card"]
                    st.markdown(
                        f'<div class="fl-case"><h3>{ui.esc(card_label(card))}</h3>'
                        f'<div class="fl-score {"low" if sc < bundle["threshold"] else ""}">{sc:.2f}</div>'
                        f'<p class="fl-sub">fraud score, alert threshold {bundle["threshold"]:.2f}</p></div>',
                        unsafe_allow_html=True)
                    reasons = [r for r in explain(pick, bundle["reference"]).split("; ") if r]
                    st.markdown(ui.chips([(r, "risk") for r in reasons]), unsafe_allow_html=True)
                    facts = [("When", pd.Timestamp(pick["_time"]).strftime("%a %d %b %Y, %H:%M"))]
                    if "_amount" in pick:
                        facts.append(("Amount", fmt_money(float(pick["_amount"]))))
                    if m.get("category"):
                        facts.append(("Category", pick[m["category"]]))
                    if "_gap_h" in pick and pd.notna(pick["_gap_h"]):
                        g = float(pick["_gap_h"])
                        facts.append(("Since previous", f"{g * 60:.0f} min" if g < 2 else f"{g:.1f} h"))
                    if "dist_home_km" in pick and pick["dist_home_km"] >= 0:
                        facts.append(("From home", f"{pick['dist_home_km']:,.0f} km"))
                    if "_y" in pick:
                        facts.append(("Actual label", "Fraud" if pick["_y"] == 1 else "Genuine"))
                    st.markdown('<dl class="fl-facts">' + "".join(f"<dt>{ui.esc(a)}</dt><dd>{ui.esc(b)}</dd>" for a, b in facts)
                                + "</dl>", unsafe_allow_html=True)
                    hist = test[test["_card"] == card]
                    t0 = pd.Timestamp(pick["_time"])
                    near = hist[(hist["_time"] >= t0 - pd.Timedelta(days=14)) & (hist["_time"] <= t0 + pd.Timedelta(days=14))]
                    if "_amount" in near and len(near) > 1:
                        st.markdown("**This card, two weeks either side**")
                        ui.show(ui.card_history_chart(near[near["_amount"] > 0], pick.name, bundle["threshold"]))


# ---------------------------------------------------------------------------
# 5. Score new data
# ---------------------------------------------------------------------------
elif step == STEPS[4]:
    st.title("Score new data")
    st.markdown('<p class="fl-sub">Use the model from step 3 or a model you saved earlier. The new table needs the same '
                "column names as the training table. Include each card's earlier transactions so the time gaps are right.</p>",
                unsafe_allow_html=True)
    c1, c2 = st.columns(2, gap="large")
    with c1:
        new_file = st.file_uploader("Transactions to score (CSV)", type=["csv"], key="score_csv")
    with c2:
        model_file = st.file_uploader("A saved FraudLens model (optional)", type=["pkl"],
                                      help="Only load model files saved by this app: model files can run code when opened.")
        if model_file is not None:
            ss["bundle"] = from_bytes(model_file.getvalue())
            st.success("Model loaded")
    if "bundle" not in ss:
        ui.empty_state("No model yet.", "Train one in step 3, or upload a model file you downloaded earlier.")
    elif new_file is not None and st.button("Score these transactions", type="primary"):
        try:
            scored = score(ss["bundle"], pd.read_csv(new_file, low_memory=False))
            scored["Reasons"] = ""
            idx = scored["alert"] == 1
            scored.loc[idx, "Reasons"] = scored[idx].apply(lambda r: explain(r, ss["bundle"]["reference"]), axis=1)
            ss["scored"] = scored
        except Exception as e:
            st.error(f"FraudLens could not score this table: {e}")
    if "scored" in ss and "bundle" in ss:
        scored, bm = ss["scored"], ss["bundle"]["mapping"]
        thr = ss["bundle"]["threshold"]
        c = st.columns(3)
        c[0].metric("Transactions scored", f"{len(scored):,}")
        c[1].metric("Alerts", f"{int(scored['alert'].sum()):,}")
        c[2].metric("Alert rate", f"{scored['alert'].mean():.2%}")
        with st.container(border=True, key="panel_9"):
            st.subheader("Fraud scores")
            ui.show(ui.score_histogram(scored["fraud_score"], thr))
        cols = list(dict.fromkeys([c for c in [bm.get("time"), bm.get("card"), bm.get("amount"), bm.get("category")] if c in scored]))
        out = scored.sort_values("fraud_score", ascending=False)[cols + ["fraud_score", "alert", "Reasons"]].rename(
            columns={"fraud_score": "Fraud score", "alert": "Alert"})
        out["Alert"] = out["Alert"].map({1: "Alert", 0: ""})
        st.dataframe(out.head(500), hide_index=True, width="stretch",
                     column_config={"Fraud score": st.column_config.ProgressColumn("Fraud score", format="%.3f", min_value=0, max_value=1)})
        st.download_button("Download all scores as CSV", out.to_csv(index=False), file_name="fraudlens_scores.csv")
