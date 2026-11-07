"""
Look and feel for FraudLens: colours, fonts, page styles and chart helpers.

Kept apart from fraud_app.py so the app code reads as the five steps and nothing else.
"""

import html

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Design tokens
# ---------------------------------------------------------------------------
INK = "#161B33"        # text, sidebar
PAPER = "#F4F5F9"      # page background (cool grey, not cream)
PANEL = "#FFFFFF"      # raised surfaces
LINE = "#DCE0EA"       # hairlines and borders
MUTED = "#5D6480"      # secondary text
INDIGO = "#3D44C9"     # actions, normal transactions, the model
CRIMSON = "#C2185B"    # fraud and only fraud
CRIMSON_SOFT = "#F7DCE7"
INDIGO_SOFT = "#E3E5FA"
GRID = "#E7EAF1"

HEAD_FONT = "Bricolage Grotesque"
BODY_FONT = "Public Sans"


# ---------------------------------------------------------------------------
# Page styles
# ---------------------------------------------------------------------------
CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,700;12..96,800&family=Public+Sans:wght@400;500;600;700&display=swap');

html, body, .stApp, .stApp p, .stApp label, .stApp input, .stApp textarea, .stApp button, .stApp li, .stApp td, .stApp th {{
  font-family: '{BODY_FONT}', system-ui, sans-serif;
}}
/* keep Streamlit's own icon font */
[data-testid="stIconMaterial"] {{ font-family: 'Material Symbols Rounded' !important; }}
.stApp {{ background: {PAPER}; color: {INK}; }}
[data-testid="stHeader"] {{ background: transparent; }}
.block-container {{ padding-top: 2.2rem; max-width: 1240px; }}

h1, h2, h3, h4 {{ font-family: '{HEAD_FONT}', '{BODY_FONT}', sans-serif !important; color: {INK}; letter-spacing: -0.02em; }}
h1 {{ font-weight: 800 !important; }}
h2, h3 {{ font-weight: 700 !important; }}
p, li {{ line-height: 1.55; }}
[data-testid="stMetricValue"], .fl-num, td, th {{ font-variant-numeric: tabular-nums; }}

/* ---------- sidebar: the analyst's desk ---------- */
[data-testid="stSidebar"] {{ background: {INK}; }}
[data-testid="stSidebar"] * {{ color: #E8EAF4; }}
[data-testid="stSidebar"] .fl-brand {{ font-family: '{HEAD_FONT}', sans-serif; font-weight: 800; font-size: 1.65rem; letter-spacing: -0.03em; color: #fff; margin: 0; }}
[data-testid="stSidebar"] .fl-brand span {{ display:inline-block; width: .62em; height: .62em; border-radius: 50%; background: {CRIMSON}; margin-left: .12em; box-shadow: 0 0 0 4px rgba(194,24,91,.25); }}
[data-testid="stSidebar"] .fl-tag {{ color: #A9AECB; font-size: .86rem; margin: .2rem 0 1.4rem; }}
[data-testid="stSidebar"] [role="radiogroup"] {{ gap: .15rem; }}
[data-testid="stSidebar"] [role="radiogroup"] label {{
  padding: .55rem .7rem; border-radius: 10px; width: 100%; transition: background .15s;
}}
[data-testid="stSidebar"] [role="radiogroup"] label:hover {{ background: rgba(255,255,255,.06); }}
[data-testid="stSidebar"] [data-testid="stRadioOption"][data-selected="true"] {{ background: rgba(61,68,201,.6); }}
[data-testid="stSidebar"] [data-testid="stRadioOption"] > div > div:first-child {{ display: none; }}
[data-testid="stSidebar"] [role="radiogroup"] p {{ font-size: .98rem; font-weight: 500; }}
[data-testid="stSidebar"] hr {{ border-color: rgba(255,255,255,.12); }}
[data-testid="stSidebar"] .fl-status {{ font-size: .86rem; color: #C5C9E0; line-height: 1.7; }}
[data-testid="stSidebar"] .fl-status b {{ color: #fff; font-weight: 600; }}
[data-testid="stSidebar"] [data-testid="stDownloadButton"] button {{ background: transparent; border: 1px solid rgba(255,255,255,.35); }}
[data-testid="stSidebar"] [data-testid="stExpander"] {{ border-color: rgba(255,255,255,.15); background: transparent; }}

/* ---------- buttons ---------- */
.stButton > button, [data-testid="stDownloadButton"] button, [data-testid="stFormSubmitButton"] button {{
  border-radius: 10px; font-weight: 600; padding: .5rem 1.1rem;
}}
.stButton > button[kind="primary"] {{ background: {INDIGO}; border-color: {INDIGO}; }}
.stButton > button[kind="primary"]:hover {{ background: #2F35A8; border-color: #2F35A8; }}
button:focus-visible {{ outline: 3px solid {CRIMSON} !important; outline-offset: 2px; }}

/* ---------- surfaces ---------- */
[data-testid="stMetric"] {{ background: {PANEL}; border: 1px solid {LINE}; border-radius: 14px; padding: .9rem 1rem; }}
[data-testid="stMetricLabel"] p {{ color: {MUTED}; font-weight: 500; }}
[data-testid="stMetricValue"] {{ font-family: '{HEAD_FONT}', sans-serif; font-weight: 700; }}
[class*="st-key-panel_"] {{ background: {PANEL}; border-radius: 16px !important; padding: 1.1rem 1.25rem !important; }}

/* threshold tally */
.fl-tally {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: .5rem; margin: .8rem 0 .4rem; }}
.fl-tally div {{ border-left: 3px solid {LINE}; padding: .1rem 0 .1rem .7rem; color: {MUTED}; font-size: .9rem; }}
.fl-tally b {{ display: block; font-family: '{HEAD_FONT}', sans-serif; font-weight: 800; font-size: 1.9rem; color: {INK}; line-height: 1.1; font-variant-numeric: tabular-nums; }}
.fl-tally b.hit {{ color: {CRIMSON}; }}
.fl-tally div:has(b.hit) {{ border-color: {CRIMSON}; }}
.fl-tally div:has(b.alarm) {{ border-color: {INDIGO}; }}
.fl-split {{ margin-top: .2rem; }}
[data-testid="stExpander"] {{ background: {PANEL}; border-radius: 12px; }}
[data-testid="stFileUploaderDropzone"] {{ background: {PANEL}; border: 1.5px dashed #B9BFD3; border-radius: 14px; }}

/* ---------- hero ---------- */
.fl-hero {{ padding: 1.2rem 0 1.6rem; max-width: 760px; }}
.fl-hero h1 {{ font-size: clamp(2.3rem, 4.6vw, 3.6rem); line-height: 1.02; margin: 0 0 1rem; }}
.fl-hero p {{ font-size: 1.12rem; color: {MUTED}; max-width: 62ch; }}

/* the ledger strip: a few rows of a card statement with the fraud burst flagged */
.fl-ledger {{ background: {PANEL}; border: 1px solid {LINE}; border-radius: 16px; padding: .5rem 0; font-size: .92rem; }}
.fl-ledger .row {{ display: grid; grid-template-columns: 6.4rem 1fr 5.4rem 7.2rem; gap: .6rem; padding: .46rem 1.1rem; align-items: center; }}
.fl-ledger .row + .row {{ border-top: 1px solid {GRID}; }}
.fl-ledger .t {{ color: {MUTED}; font-variant-numeric: tabular-nums; }}
.fl-ledger .a {{ text-align: right; font-variant-numeric: tabular-nums; font-weight: 600; }}
.fl-ledger .bar {{ height: 6px; border-radius: 3px; background: {GRID}; overflow: hidden; }}
.fl-ledger .bar i {{ display: block; height: 100%; background: {INDIGO}; border-radius: 3px; }}
.fl-ledger .row.hit {{ background: linear-gradient(90deg, {CRIMSON_SOFT}, rgba(247,220,231,0)); }}
.fl-ledger .row.hit .bar i {{ background: {CRIMSON}; }}
.fl-ledger .row.hit .a {{ color: {CRIMSON}; }}
@media (max-width: 640px) {{
  .fl-ledger .row {{ grid-template-columns: 4.4rem 1fr 4.6rem 2.6rem; gap: .4rem; padding: .42rem .8rem; font-size: .86rem; }}
}}
.fl-ledger .cap {{ padding: .55rem 1.1rem .3rem; color: {MUTED}; font-size: .84rem; border-top: 1px solid {GRID}; }}
@media (prefers-reduced-motion: no-preference) {{
  .fl-ledger .row.hit {{ animation: fl-flag 1.1s ease-out both; }}
  .fl-ledger .row.hit:nth-of-type(5) {{ animation-delay: .25s; }}
  .fl-ledger .row.hit:nth-of-type(6) {{ animation-delay: .5s; }}
  @keyframes fl-flag {{ from {{ background-color: transparent; background-size: 0 100%; }} }}
}}

/* step list on the welcome screen: a real sequence, so it is numbered */
.fl-steps {{ display: grid; grid-template-columns: repeat(5, 1fr); gap: .8rem; margin: 1.6rem 0 .6rem; counter-reset: s; }}
.fl-steps div {{ border-top: 2px solid {INK}; padding-top: .6rem; font-size: .92rem; color: {MUTED}; }}
.fl-steps div b {{ display: block; color: {INK}; font-weight: 600; margin-bottom: .15rem; }}
.fl-steps div b::before {{ counter-increment: s; content: counter(s) "  "; color: {CRIMSON}; font-family: '{HEAD_FONT}', sans-serif; }}
@media (max-width: 800px) {{ .fl-steps {{ grid-template-columns: 1fr 1fr; }} }}

/* the review-budget sentence: the main readout of the results */
.fl-budget {{ font-family: '{HEAD_FONT}', sans-serif; font-size: clamp(1.45rem, 2.6vw, 2.05rem); line-height: 1.3; font-weight: 500; letter-spacing: -0.015em; max-width: 30ch; margin: .2rem 0 .4rem; }}
.fl-budget em {{ font-style: normal; font-weight: 800; color: {INDIGO}; }}
.fl-budget strong {{ font-weight: 800; color: {CRIMSON}; }}
.fl-sub {{ color: {MUTED}; font-size: .95rem; }}

/* chips for features and reasons */
.fl-chips {{ display: flex; flex-wrap: wrap; gap: .4rem; margin: .3rem 0 .6rem; }}
.fl-chip {{ display: inline-flex; align-items: center; gap: .35rem; padding: .28rem .7rem; border-radius: 999px; font-size: .86rem; border: 1px solid {LINE}; background: {PANEL}; color: {INK}; }}
.fl-chip.on {{ border-color: {INDIGO}; background: {INDIGO_SOFT}; }}
.fl-chip.off {{ color: #9097AE; background: transparent; border-style: dashed; }}
.fl-chip.risk {{ border-color: {CRIMSON}; background: {CRIMSON_SOFT}; color: #8E0F41; }}
.fl-group {{ font-weight: 600; font-size: .9rem; margin-top: .6rem; color: {INK}; }}

/* case panel */
.fl-case h3 {{ margin: 0 0 .2rem; }}
.fl-score {{ font-family: '{HEAD_FONT}', sans-serif; font-weight: 800; font-size: 3rem; line-height: 1; color: {CRIMSON}; }}
.fl-score.low {{ color: {INDIGO}; }}
.fl-facts {{ display: grid; grid-template-columns: auto 1fr; gap: .25rem 1rem; font-size: .93rem; margin-top: .8rem; }}
.fl-facts dt {{ color: {MUTED}; }}
.fl-facts dd {{ margin: 0; font-weight: 600; font-variant-numeric: tabular-nums; }}

.fl-empty {{ background: {PANEL}; border: 1px dashed #B9BFD3; border-radius: 14px; padding: 1.6rem 1.4rem; color: {MUTED}; }}
.fl-empty b {{ color: {INK}; }}
</style>
"""


def inject_css():
    st.markdown(CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Small HTML pieces
# ---------------------------------------------------------------------------
def esc(x):
    return html.escape(str(x))


def chips(items, kind=""):
    """items: list of (text, kind) or plain strings."""
    out = []
    for it in items:
        text, k = (it if isinstance(it, tuple) else (it, kind))
        out.append(f'<span class="fl-chip {k}">{esc(text)}</span>')
    return f'<div class="fl-chips">{"".join(out)}</div>'


def empty_state(title, body):
    st.markdown(f'<div class="fl-empty"><b>{esc(title)}</b><br>{esc(body)}</div>', unsafe_allow_html=True)


LEDGER = """
<div class="fl-ledger" aria-label="Example card statement with three flagged transactions">
  <div class="row"><span class="t">Mon 09:14</span><span>Grocery, in store</span><span class="a">$42.10</span><span class="bar"><i style="width:4%"></i></span></div>
  <div class="row"><span class="t">Mon 18:02</span><span>Fuel</span><span class="a">$61.75</span><span class="bar"><i style="width:3%"></i></span></div>
  <div class="row"><span class="t">Wed 12:40</span><span>Restaurant</span><span class="a">$28.00</span><span class="bar"><i style="width:2%"></i></span></div>
  <div class="row hit"><span class="t">Thu 23:51</span><span>Online shopping</span><span class="a">$912.40</span><span class="bar"><i style="width:91%"></i></span></div>
  <div class="row hit"><span class="t">Thu 23:58</span><span>Online shopping</span><span class="a">$874.99</span><span class="bar"><i style="width:94%"></i></span></div>
  <div class="row hit"><span class="t">Fri 00:06</span><span>Grocery, in store</span><span class="a">$356.20</span><span class="bar"><i style="width:88%"></i></span></div>
  <div class="cap">Three large purchases seven minutes apart, late at night. The bar is the fraud score.</div>
</div>
"""


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------
@alt.theme.register("fraudlens", enable=True)
def _theme():
    return alt.theme.ThemeConfig({
        "config": {
            "background": PANEL,
            "font": f"{BODY_FONT}, system-ui, Helvetica, Arial, sans-serif",
            "padding": 8,
            "view": {"stroke": None},
            "text": {"font": f"{BODY_FONT}, system-ui, Helvetica, Arial, sans-serif"},
            "axis": {"labelColor": MUTED, "titleColor": MUTED, "titleFontWeight": 500, "labelFontSize": 12, "labelFont": f"{BODY_FONT}, system-ui, sans-serif",
                     "titleFont": f"{BODY_FONT}, system-ui, sans-serif", "titleFontSize": 12, "gridColor": GRID, "domainColor": LINE, "tickColor": LINE},
            "legend": {"labelFont": f"{BODY_FONT}, system-ui, sans-serif", "labelColor": INK, "titleColor": MUTED, "orient": "top", "labelFontSize": 12},
            "title": {"font": f"{HEAD_FONT}, system-ui, sans-serif", "color": INK, "fontSize": 15, "anchor": "start", "fontWeight": 700},
        }
    })


def show(chart):
    """Render with our theme rather than Streamlit's."""
    st.altair_chart(chart, theme=None, width="stretch")


def capture_curve(y, score, amount=None, max_share=0.10, points=400):
    """Share of fraud (and fraud dollars) caught when reviewing the top x% of transactions."""
    y = np.asarray(y, dtype=float)
    order = np.argsort(-np.asarray(score), kind="stable")
    ys = y[order]
    n = len(y)
    cum = np.cumsum(ys) / max(ys.sum(), 1)
    k = np.unique(np.clip(np.linspace(1, max(1, int(n * max_share)), points).astype(int), 1, n))
    df = pd.DataFrame({"share": k / n, "Fraud caught": cum[k - 1]})
    if amount is not None:
        a = np.nan_to_num(np.asarray(amount, dtype=float)[order]) * ys
        df["Fraud dollars caught"] = (np.cumsum(a) / max(a.sum(), 1e-9))[k - 1]
    return df


def budget_readout(y, score, amount, budget):
    y = np.asarray(y)
    order = np.argsort(-np.asarray(score), kind="stable")
    k = max(1, int(round(budget * len(y))))
    top = order[:k]
    out = {"alerts": k, "caught": int(y[top].sum()), "frauds": int(y.sum()),
           "recall": y[top].sum() / max(y.sum(), 1), "precision": y[top].mean()}
    if amount is not None:
        a = np.nan_to_num(np.asarray(amount, dtype=float))
        out["dollars_caught"] = float((a[top] * y[top]).sum())
        out["dollars_total"] = float((a * y).sum())
    return out


def capture_chart(curve, budget):
    long = curve.melt("share", var_name="measure", value_name="caught")
    colors = alt.Scale(domain=["Fraud caught", "Fraud dollars caught"], range=[CRIMSON, INDIGO])
    base = alt.Chart(long).encode(
        x=alt.X("share:Q", title="Share of transactions reviewed", axis=alt.Axis(format=".1~%", tickCount=6, labelOverlap=True),
                scale=alt.Scale(domain=[0, float(curve["share"].max())])),
        y=alt.Y("caught:Q", title="Share caught", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1])),
        color=alt.Color("measure:N", scale=colors, title=None),
    )
    lines = base.mark_line(strokeWidth=2.4)
    hover = alt.selection_point(fields=["share"], nearest=True, on="pointerover", empty=False)
    pts = base.mark_point(size=70, filled=True, stroke=PANEL, strokeWidth=2).encode(
        opacity=alt.condition(hover, alt.value(1), alt.value(0)),
        tooltip=[alt.Tooltip("share:Q", title="Reviewed", format=".2%"),
                 alt.Tooltip("measure:N", title="Measure"), alt.Tooltip("caught:Q", title="Caught", format=".1%")],
    ).add_params(hover)
    rand = alt.Chart(pd.DataFrame({"share": [0, curve["share"].max()]})).mark_line(
        strokeDash=[4, 4], color="#A3A9BE", strokeWidth=1.5).encode(x="share:Q", y=alt.Y("share:Q"))
    rule = alt.Chart(pd.DataFrame({"share": [budget]})).mark_rule(color=INK, strokeWidth=1.5).encode(x="share:Q")
    label = alt.Chart(pd.DataFrame({"share": [budget], "t": ["your budget"]})).mark_text(
        align="left", dx=6, dy=-6, color=INK, fontSize=12, fontWeight=600).encode(x="share:Q", y=alt.value(12), text="t:N")
    rl = alt.Chart(pd.DataFrame({"share": [curve["share"].max()], "y": [curve["share"].max()], "t": ["random review"]})).mark_text(
        align="right", dy=-8, color="#8A90A6", fontSize=11).encode(x="share:Q", y="y:Q", text="t:N")
    return (rand + rl + lines + pts + rule + label).properties(height=320)


def pr_chart(pr, op_recall, op_precision, val_recall=None, val_precision=None):
    pr = pr.iloc[::max(1, len(pr) // 1500)]
    line = alt.Chart(pr).mark_line(color=INDIGO, strokeWidth=2.2, interpolate="step-after").encode(
        x=alt.X("Recall:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%")),
        y=alt.Y("Precision:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%")),
        tooltip=[alt.Tooltip("Recall:Q", format=".1%"), alt.Tooltip("Precision:Q", format=".1%")])
    layers = [line]
    if val_recall is not None:
        layers.append(alt.Chart(pd.DataFrame({"Recall": [val_recall], "Precision": [val_precision]})).mark_point(
            size=140, shape="circle", filled=False, color=INK, strokeWidth=2).encode(x="Recall:Q", y="Precision:Q"))
    op = pd.DataFrame({"Recall": [op_recall], "Precision": [op_precision], "t": ["your threshold"]})
    layers.append(alt.Chart(op).mark_point(size=150, filled=True, color=CRIMSON, stroke=PANEL, strokeWidth=2).encode(
        x="Recall:Q", y="Precision:Q",
        tooltip=[alt.Tooltip("Recall:Q", format=".1%"), alt.Tooltip("Precision:Q", format=".1%")]))
    layers.append(alt.Chart(op).mark_text(align="left", dx=10, dy=-10, color=CRIMSON, fontWeight=600, fontSize=12).encode(
        x="Recall:Q", y="Precision:Q", text="t:N"))
    return alt.layer(*layers).properties(height=300)


def importance_chart(imp, groups):
    imp = imp.assign(group=imp["feature"].map(groups).fillna(""))
    return alt.Chart(imp).mark_bar(color=INDIGO, cornerRadiusEnd=4, height=18).encode(
        x=alt.X("importance:Q", title="Share of the model's decisions", axis=alt.Axis(format="%")),
        y=alt.Y("feature:N", sort="-x", title=None, axis=alt.Axis(labelColor=INK, labelFontSize=13)),
        tooltip=[alt.Tooltip("feature:N", title="Feature"), alt.Tooltip("group:N", title="Group"),
                 alt.Tooltip("importance:Q", title="Importance", format=".1%")],
    ).properties(height=max(140, 34 * len(imp)))


def split_chart(info, t_min, t_max):
    df = pd.DataFrame([
        {"part": "Train", "start": t_min, "end": info["validation starts"], "rows": info["train rows"], "fraud": info["train fraud"]},
        {"part": "Validation", "start": info["validation starts"], "end": info["test starts"], "rows": info["validation rows"], "fraud": info["validation fraud"]},
        {"part": "Test", "start": info["test starts"], "end": t_max, "rows": info["test rows"], "fraud": info["test fraud"]},
    ])
    df["mid"] = df["start"] + (df["end"] - df["start"]) / 2
    scale = alt.Scale(domain=["Train", "Validation", "Test"], range=["#C9CDF2", "#8F95E3", INDIGO])
    x = alt.X("start:T", title=None, scale=alt.Scale(domain=[t_min, t_max]),
              axis=alt.Axis(format="%b %Y", grid=False, tickCount=6, labelOverlap=True))
    bars = alt.Chart(df).mark_bar(height=40, cornerRadius=4, stroke=PANEL, strokeWidth=2).encode(
        x=x, x2="end:T", color=alt.Color("part:N", scale=scale, legend=None),
        tooltip=[alt.Tooltip("part:N", title="Part"), alt.Tooltip("start:T", title="From", format="%d %b %Y"),
                 alt.Tooltip("end:T", title="To", format="%d %b %Y"), alt.Tooltip("rows:Q", title="Transactions", format=","),
                 alt.Tooltip("fraud:Q", title="Fraud", format=",")])
    text = alt.Chart(df).mark_text(fontSize=12, fontWeight=600).encode(
        x="mid:T", text="part:N",
        color=alt.condition(alt.datum.part == "Train", alt.value(INK), alt.value("#FFFFFF")))
    return (bars + text).properties(height=70)


def weekly_fraud_chart(t, y):
    df = pd.DataFrame({"week": pd.to_datetime(t).dt.to_period("W").dt.start_time, "y": y})
    g = df.groupby("week").agg(transactions=("y", "size"), fraud=("y", "sum")).reset_index()
    return alt.Chart(g).mark_bar(color=CRIMSON, cornerRadiusEnd=2).encode(
        x=alt.X("week:T", title=None, axis=alt.Axis(format="%b %Y", grid=False)),
        y=alt.Y("fraud:Q", title="Fraud cases per week"),
        tooltip=[alt.Tooltip("week:T", title="Week of", format="%d %b %Y"),
                 alt.Tooltip("fraud:Q", title="Fraud cases", format=","),
                 alt.Tooltip("transactions:Q", title="Transactions", format=",")],
    ).properties(height=170)


def score_histogram(scores, threshold):
    df = pd.DataFrame({"score": np.asarray(scores)})
    df["side"] = np.where(df["score"] >= threshold, "Alert", "No alert")
    bars = alt.Chart(df).mark_bar(cornerRadiusEnd=2).encode(
        x=alt.X("score:Q", bin=alt.Bin(step=0.025, extent=[0, 1]), title="Fraud score"),
        y=alt.Y("count():Q", title="Transactions", scale=alt.Scale(type="symlog")),
        color=alt.Color("side:N", scale=alt.Scale(domain=["No alert", "Alert"], range=[INDIGO, CRIMSON]), title=None),
        tooltip=[alt.Tooltip("count():Q", title="Transactions", format=",")])
    rule = alt.Chart(pd.DataFrame({"t": [threshold]})).mark_rule(color=INK, strokeDash=[4, 3]).encode(x="t:Q")
    return (bars + rule).properties(height=220)


def card_history_chart(hist, selected_index, threshold):
    h = hist.copy()
    h["status"] = np.where(h["fraud_score"] >= threshold, "Alert", "No alert")
    h["selected"] = h.index == selected_index
    tips = [alt.Tooltip("_time:T", title="When", format="%d %b %Y %H:%M"),
            alt.Tooltip("_amount:Q", title="Amount", format=",.2f"),
            alt.Tooltip("fraud_score:Q", title="Fraud score", format=".3f")]
    if "_y" in h:
        tips.append(alt.Tooltip("_y:N", title="Actual fraud (1 = yes)"))
    base = alt.Chart(h).encode(
        x=alt.X("_time:T", title=None, axis=alt.Axis(format="%d %b", grid=False)),
        y=alt.Y("_amount:Q", title="Amount", scale=alt.Scale(type="log"), axis=alt.Axis(tickCount=4, format="$,.0f", grid=False)))
    dots = base.mark_circle(size=70, opacity=0.9, stroke=PANEL, strokeWidth=1.5).encode(
        color=alt.Color("status:N", scale=alt.Scale(domain=["No alert", "Alert"], range=[INDIGO, CRIMSON]), title=None),
        tooltip=tips)
    ring = alt.Chart(h[h["selected"]]).mark_point(size=360, color=INK, strokeWidth=2).encode(
        x="_time:T", y="_amount:Q")
    return (dots + ring).properties(height=250)
