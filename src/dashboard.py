import json
import sys
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

# Lets this file import from the same folder, wherever it's run from.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cascade import load_answers, route
from sweep import SCORES, curve, fixed_points, with_oof_scores

# T32: the cascade dashboard. Run with:
#   .venv/bin/streamlit run src/dashboard.py
#
# Move the threshold and watch cost and recall trade off. Everything is computed offline from the
# shadow-mode log (every layer already ran), so the slider costs nothing.
# Default view: TEST at the operating point picked on train (T30).

BASE_DIR = Path(__file__).resolve().parent.parent
LAYER_NAMES = {'L1': 'L1 rules', 'L2': 'L2 classifier', 'L3': 'Judge'}

st.set_page_config(page_title="Hallucination cascade", layout="wide")


@st.cache_data
def load():
    rows = with_oof_scores(load_answers())
    point = json.loads((BASE_DIR / 'data' / 'operating_point.json').read_text())
    return rows, point


@st.cache_data
def curves_for(split):
    part = rows[rows['split'] == split].reset_index(drop=True)
    lines = pd.concat([curve(part, column).assign(score=name) for name, column in SCORES.items()])
    points = pd.DataFrame([{**values, 'system': name} for name, values in fixed_points(part).items()])
    return lines, points


rows, point = load()

# --- Sidebar: what to look at ---------------------------------------------------------------
with st.sidebar:
    st.header("Settings")
    split = st.radio("Answers", ['test', 'train'], format_func=lambda s: {
        'test': "Test: 40, reference labels", 'train': "Train: 280, judge labels"}[s])
    score_name = st.radio("Clearing score for answers L1 can't settle", list(SCORES))
    column = SCORES[score_name]
    part = rows[rows['split'] == split].reset_index(drop=True)
    values = part.loc[part['l1_verdict'] == 'uncertain', column]
    default = point[score_name]['threshold']
    low, high = float(values.min()), float(values.max())
    threshold = st.slider("Escalate to the judge when the score is at least", low, high,
                          float(np.clip(default, low, high)), step=(high - low) / 200, format="%.3f")
    st.caption(f"Operating point picked on train: **{default:.3f}** (fewest judge calls with ≥ 90% recall).")
    st.markdown("""
**How the cascade decides**
- **L1** fails answers with hard evidence and passes clean refusals.
- The rest get a score. Below the threshold, the answer is **cleared**; at or above it, the **judge** decides.
- L2 never blocks an answer on its own.
""")

routed = route(part, column, threshold)
truth, flagged = part['truth'], routed['flagged']
judge_cost = part['cost_l3_usd'].mean() * 1000

# --- Headline -----------------------------------------------------------------------------
st.title("Hallucination cascade: cost vs recall")
st.caption(f"{len(part)} {split} answers, {int(truth.sum())} hallucinations. "
           + ("Small sample: read every number with its interval (see reports/)." if split == 'test' else
              "Train scores for L2 are out-of-fold: each answer scored by a model that never saw it."))
cols = st.columns(5)
cols[0].metric("Caught", f"{flagged[truth == 1].mean():.0%}", help="Share of hallucinations flagged")
cols[0].caption(f"{int((flagged & truth).sum())} of {int(truth.sum())}")
cols[1].metric("To judge", f"{(routed['terminating_layer'] == 'L3').mean():.0%}", help="Share of answers sent to the LLM judge")
cols[1].caption("judge everything: 100%")
cols[2].metric("$ per 1,000", f"${routed['cost_usd'].mean() * 1000:.2f}", help="Judge cost per 1,000 answers; L1 and L2 are free")
cols[2].caption(f"judge everything: ${judge_cost:.2f}")
cols[3].metric("Median time", f"{routed['latency_ms'].median() / 1000:.2f} s", help="The evaluator's added time per answer")
cols[3].caption(f"judge everything: {part['latency_l3_ms'].median() / 1000:.2f} s")
cols[4].metric("Real flags", f"{truth[flagged == 1].mean():.0%}" if flagged.sum() else "–", help="Precision: flagged answers that are hallucinations")
cols[4].caption(f"{int(flagged.sum())} flagged")

# --- The curve ----------------------------------------------------------------------------
lines, points = curves_for(split)
current = pd.DataFrame([{'judge_share': (routed['terminating_layer'] == 'L3').mean(),
                         'recall': flagged[truth == 1].mean(), 'label': 'current setting'}])
chart = (
    alt.Chart(lines).mark_line(interpolate='step-after').encode(
        x=alt.X('judge_share:Q', title='Share of answers sent to the judge (cost)', axis=alt.Axis(format='%')),
        y=alt.Y('recall:Q', title='Hallucinations caught (recall)', axis=alt.Axis(format='%'), scale=alt.Scale(domain=[0, 1])),
        color=alt.Color('score:N', title='Clearing score'))
    + alt.Chart(points).mark_point(size=90, filled=True, color='#2f855a').encode(x='judge_share:Q', y='recall:Q', tooltip=['system', 'recall', 'judge_share'])
    + alt.Chart(points).mark_text(dx=-8, dy=14, align='right', color='#2f855a').encode(
        x='judge_share:Q', y='recall:Q', text='system')    # left of and below the dot: stays inside the chart
    + alt.Chart(current).mark_point(size=200, shape='diamond', filled=True, color='black').encode(x='judge_share:Q', y='recall:Q')
    + alt.Chart(pd.DataFrame({'y': [0.9]})).mark_rule(strokeDash=[4, 4], color='#999').encode(y='y:Q')
).properties(height=380)
st.altair_chart(chart, use_container_width=True)
st.caption("◆ = the current slider setting. Dashed line = 90% recall. Each line steps through every threshold the score takes.")

# --- Details ------------------------------------------------------------------------------
left, middle, right = st.columns(3)
with left:
    st.subheader("Where answers stop")
    stops = routed['terminating_layer'].map(LAYER_NAMES).value_counts().reindex(LAYER_NAMES.values(), fill_value=0)
    st.bar_chart(stops)
with middle:
    st.subheader("Confusion matrix")
    st.dataframe(pd.crosstab(truth.map({1: 'Hallucination', 0: 'Fine'}).rename('Truth'),
                             flagged.map({1: 'Flagged', 0: 'Let through'}).rename('Cascade')), use_container_width=True)
    misses = routed[(truth == 1) & (flagged == 0)]['terminating_layer'].map(LAYER_NAMES).value_counts()
    alarms = routed[(truth == 0) & (flagged == 1)]['terminating_layer'].map(LAYER_NAMES).value_counts()
    st.markdown("**Misses let through by:** " + (", ".join(f"{n} {layer}" for layer, n in misses.items()) or "none"))
    st.markdown("**False alarms raised by:** " + (", ".join(f"{n} {layer}" for layer, n in alarms.items()) or "none"))
with right:
    st.subheader("By failure mode")
    modes = pd.DataFrame({'Answers': part.groupby('induction_mode').size(),
                          'Hallucinations': truth.groupby(part['induction_mode']).sum(),
                          'Caught': (truth & flagged).groupby(part['induction_mode']).sum(),
                          'To judge': (routed['terminating_layer'] == 'L3').groupby(part['induction_mode']).sum()})
    st.dataframe(modes, use_container_width=True)

st.caption("Judging everything scores 100% against the test reference partly by construction: the reference uses "
           "the judge's label wherever the human labeller agreed with it.")
