"""
FraudLens: our fraud detection prototype.

Run from the repository folder:
    pip install -r app/requirements.txt
    streamlit run app/fraud_app.py

Steps in the app:
  1. Load a transaction table (CSV upload or a file path).
  2. Tell the app which column plays which role.
  3. Check the columns for leakage and train a model with our leak-free protocol.
  4. Review the alerts: the riskiest test transactions with short reasons.
  5. Score new transactions with the trained model.
"""

import os
import sys

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from prototype import (ROLES, FEATURES, guess_columns, available_features, prepare,
                           sample_cards, time_split, leak_check, train_and_evaluate,
                           alert_budgets, score, explain, to_bytes, from_bytes)

st.set_page_config(page_title="FraudLens", page_icon="🛡️", layout="wide")
ss = st.session_state


@st.cache_data(show_spinner=False)
def read_csv(source):
    return pd.read_csv(source, low_memory=False)


# ---------------------------------------------------------------------------
# Sidebar: what the app does
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("🛡️ FraudLens")
    st.caption("Card fraud detection prototype, Group 2, COMP8851 Major Project")
    st.markdown(
        "**Features the app can build**\n\n"
        "- **Temporal:** time since the card's previous transaction, night-time\n"
        "- **Spatial:** distance from home, merchant outside the home state, merchant abroad\n"
        "- **Amount and category:** log amount, category fraud rate\n"
        "- **Customer:** age, gender\n\n"
        "**Protocol**\n\n"
        "- Split by time: train, then validation, then test\n"
        "- Category risk fitted on the training part only\n"
        "- Random Forest with our settings (3 features per split, 5 rows per leaf; 200 trees in our experiments)\n"
        "- Alert threshold chosen for the best F2 on validation\n"
        "- Results: PR-AUC, F2, MCC and alert budgets on the test part"
    )
    if "bundle" in ss:
        st.success("A trained model is ready")
        st.download_button("Download the trained model", to_bytes(ss["bundle"]),
                           file_name="fraudlens_model.pkl")

st.title("FraudLens: card fraud detection")
tab_data, tab_map, tab_train, tab_alerts, tab_score = st.tabs(
    ["1. Data", "2. Columns", "3. Train and evaluate", "4. Alerts", "5. Score new data"])


# ---------------------------------------------------------------------------
# 1. Data
# ---------------------------------------------------------------------------
with tab_data:
    st.subheader("Load a transaction table")
    st.write("Each row should be one card transaction. To train a model, the table needs a fraud label.")
    upload = st.file_uploader("Upload a CSV file", type=["csv"])
    path = st.text_input("Or type the path of a CSV file", value=ss.get("path", ""),
                         placeholder="app/demo_data/sparkov_sample.csv")
    if st.button("Load data", type="primary"):
        try:
            source = upload if upload is not None else path
            if not source:
                st.warning("Upload a file or type a path first.")
            else:
                ss["raw"] = read_csv(source)
                ss["path"] = path
                ss["mapping"] = guess_columns(ss["raw"])
                for role in ROLES:
                    ss.pop(f"role_{role}", None)
                for key in ["bundle", "test", "pr_curve", "importance", "split_info", "leaks"]:
                    ss.pop(key, None)
        except Exception as e:
            st.error(f"Could not read the file: {e}")

    if "raw" in ss:
        raw = ss["raw"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Rows", f"{len(raw):,}")
        c2.metric("Columns", raw.shape[1])
        label_guess = ss["mapping"].get("label")
        if label_guess:
            c3.metric("Fraud rate (guessed label)", f"{pd.to_numeric(raw[label_guess], errors='coerce').mean():.3%}"
                      if pd.api.types.is_numeric_dtype(raw[label_guess]) else "see step 2")
        st.dataframe(raw.head(20), width="stretch")
        st.info("Go to **2. Columns** to check the column roles.")


# ---------------------------------------------------------------------------
# 2. Columns
# ---------------------------------------------------------------------------
with tab_map:
    if "raw" not in ss:
        st.info("Load a table in **1. Data** first.")
    else:
        raw = ss["raw"]
        st.subheader("Which column plays which role?")
        st.write("We filled in our best guesses from common column names. Change any that are wrong, "
                 "and choose *(none)* for roles the table does not have.")
        options = ["(none)"] + list(raw.columns)
        new_map = {}
        cols = st.columns(2)
        for i, (role, (desc, _)) in enumerate(ROLES.items()):
            current = ss["mapping"].get(role)
            choice = cols[i % 2].selectbox(desc, options, index=options.index(current) if current in options else 0,
                                           key=f"role_{role}")
            new_map[role] = None if choice == "(none)" else choice
        ss["mapping"] = new_map

        feats = available_features(new_map)
        st.subheader("Features we can build")
        rows = [{"Feature": f, "Group": g, "Description": desc, "Available": "✅" if f in feats else "—"}
                for f, (g, _, desc) in FEATURES.items()]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        if not new_map.get("label"):
            st.warning("Without a fraud label you can only score with a model trained earlier (step 5).")
        if not new_map.get("time"):
            st.warning("A time column is needed to split by time and to build the temporal features.")


# ---------------------------------------------------------------------------
# 3. Train and evaluate
# ---------------------------------------------------------------------------
with tab_train:
    m = ss.get("mapping", {})
    if "raw" not in ss or not m.get("label") or not m.get("time"):
        st.info("Load a table and choose at least the **fraud label** and **time** columns in step 2.")
    else:
        feats = available_features(m)
        st.subheader("Choose the features")
        chosen = st.multiselect("Features for the model", feats, default=feats)
        c1, c2 = st.columns(2)
        trees = c1.select_slider("Number of trees", [50, 100, 200], value=100,
                                 help="200 is the setting from our experiments; 100 is faster.")
        max_rows = c2.select_slider("Largest number of rows to use", [100_000, 250_000, 500_000, 1_000_000, 2_000_000],
                                    value=500_000, help="Larger tables are reduced to a random sample of whole cards.")

        if st.button("Train and evaluate", type="primary", disabled=not chosen):
            with st.spinner("Preparing the data and training the model..."):
                d = sample_cards(prepare(ss["raw"], m), max_rows)
                train_sub, _, test_part, _, _ = time_split(d)
                skip = {m[r] for r in ROLES if m.get(r)} | {"_time", "_y", "_card", "_amount", "_gap_h"} | set(FEATURES)
                raw_cols = [c for c in ss["raw"].columns if c not in skip and c in d.columns]
                ss["leaks"] = leak_check(train_sub, test_part, raw_cols)
                try:
                    bundle, test, pr_curve, importance, split_info = train_and_evaluate(d, m, chosen, n_estimators=trees)
                    ss.update(bundle=bundle, test=test, pr_curve=pr_curve, importance=importance, split_info=split_info)
                except ValueError as e:
                    st.error(f"Could not train: {e}")

        if "bundle" in ss:
            met, info = ss["bundle"]["metrics"], ss["split_info"]
            st.subheader("Results on the test part")
            c = st.columns(5)
            c[0].metric("PR-AUC", f"{met['PR-AUC']:.4f}",
                        help=f"Guessing would give about {met['Test fraud rate']:.4f} (the test fraud rate).")
            c[1].metric("F2", f"{met['F2']:.4f}")
            c[2].metric("MCC", f"{met['MCC']:.4f}")
            c[3].metric("Recall", f"{met['Recall']:.3f}")
            c[4].metric("Precision", f"{met['Precision']:.3f}")
            st.caption(f"Alert threshold {met['Threshold']:.2f} (best F2 on validation) · "
                       f"validation PR-AUC {met['Val PR-AUC']:.4f} · test ROC-AUC {met['ROC-AUC']:.4f} · "
                       f"test fraud rate {met['Test fraud rate']:.3%}")

            with st.expander("How the data was split"):
                st.json({k: str(v) if not isinstance(v, (int, float)) else v for k, v in info.items()})

            left, right = st.columns(2)
            with left:
                st.markdown("**Precision-recall curve**")
                st.altair_chart(alt.Chart(ss["pr_curve"]).mark_line().encode(
                    x=alt.X("Recall", scale=alt.Scale(domain=[0, 1])),
                    y=alt.Y("Precision", scale=alt.Scale(domain=[0, 1]))), width="stretch")
            with right:
                st.markdown("**Which features the model relies on**")
                st.altair_chart(alt.Chart(ss["importance"]).mark_bar().encode(
                    x="importance", y=alt.Y("feature", sort="-x")), width="stretch")

            st.markdown("**Alert budgets:** analysts review only the top share of transactions")
            amount = ss["test"]["_amount"] if "_amount" in ss["test"] else None
            budgets = alert_budgets(ss["test"]["_y"], ss["test"]["fraud_score"], amount)
            st.dataframe(budgets.style.format({"Recall": "{:.3f}", "Precision": "{:.3f}", "Dollar recall": "{:.3f}"}),
                         hide_index=True, width="stretch")

            st.markdown("**Leakage check:** each raw column on its own (ROC-AUC on the test part)")
            leaks = ss["leaks"]
            if len(leaks) and (leaks["ROC-AUC"] >= 0.95).any():
                st.warning("Some columns almost give the answer on their own. Check how they were created "
                           "before using them in a model.")
            st.dataframe(leaks, hide_index=True, width="stretch")


# ---------------------------------------------------------------------------
# 4. Alerts
# ---------------------------------------------------------------------------
with tab_alerts:
    if "test" not in ss:
        st.info("Train a model in step 3 first.")
    else:
        test, bundle, m = ss["test"], ss["bundle"], ss["bundle"]["mapping"]
        st.subheader("Riskiest transactions in the test part")
        n = st.slider("How many to show", 10, 500, 50)
        only_alerts = st.checkbox("Only transactions above the alert threshold", value=False)
        view = test[test["alert"] == 1] if only_alerts else test
        top = view.sort_values("fraud_score", ascending=False).head(n).copy()
        top["reasons"] = top.apply(lambda r: explain(r, bundle["reference"]), axis=1)
        show = [c for c in [m.get("time"), m.get("card"), m.get("amount"), m.get("category")] if c]
        show = list(dict.fromkeys(show)) + ["fraud_score", "reasons"] + (["_y"] if "_y" in top else [])
        st.dataframe(top[show].rename(columns={"_y": "actual fraud"}), hide_index=True, width="stretch",
                     column_config={"fraud_score": st.column_config.NumberColumn("fraud score", format="%.3f")})
        st.download_button("Download these alerts as CSV", top[show].to_csv(index=False),
                           file_name="fraudlens_alerts.csv")


# ---------------------------------------------------------------------------
# 5. Score new data
# ---------------------------------------------------------------------------
with tab_score:
    st.subheader("Score new transactions")
    st.write("Use the model trained in step 3, or upload a model saved earlier. The new table needs the same "
             "column names as the training table. Include each card's earlier transactions, so the time gaps are right.")
    model_file = st.file_uploader("Upload a saved model (optional, only files saved by this app)", type=["pkl"])
    if model_file is not None:
        ss["bundle"] = from_bytes(model_file.getvalue())
        st.success("Model loaded")
    new_file = st.file_uploader("Upload the transactions to score", type=["csv"], key="score_csv")
    if "bundle" not in ss:
        st.info("Train a model in step 3 or upload a saved model.")
    elif new_file is not None and st.button("Score", type="primary"):
        try:
            scored = score(ss["bundle"], pd.read_csv(new_file, low_memory=False))
            scored["reasons"] = ""
            idx = scored["alert"] == 1
            scored.loc[idx, "reasons"] = scored[idx].apply(lambda r: explain(r, ss["bundle"]["reference"]), axis=1)
            st.metric("Alerts", f"{int(scored['alert'].sum()):,} of {len(scored):,} transactions")
            m = ss["bundle"]["mapping"]
            show = [c for c in [m.get("time"), m.get("card"), m.get("amount"), m.get("category")] if c in scored]
            show = list(dict.fromkeys(show)) + ["fraud_score", "alert", "reasons"]
            st.dataframe(scored.sort_values("fraud_score", ascending=False)[show].head(500),
                         hide_index=True, width="stretch",
                         column_config={"fraud_score": st.column_config.NumberColumn("fraud score", format="%.3f")})
            st.download_button("Download all scores as CSV", scored[show].to_csv(index=False),
                               file_name="fraudlens_scores.csv")
        except Exception as e:
            st.error(f"Could not score this table: {e}")
