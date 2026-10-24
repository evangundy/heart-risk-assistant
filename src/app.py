"""Streamlit chat app: ask about a patient in plain English, get a model-backed risk estimate.

Run from the project root:
    streamlit run src/app.py
"""
from __future__ import annotations
from src.predictor import HeartRiskPredictor
from src.llm_interface import FEATURE_SPEC, REQUIRED, HeartRiskAssistant, NebiusLLM
from dotenv import load_dotenv
import streamlit as st
import pandas as pd

import sys
from pathlib import Path

# allow `from src...` imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


load_dotenv()
st.set_page_config(page_title="Heart Risk Assistant",
                   page_icon="🫀", layout="wide")

EXAMPLES = {
    "Complete patient": (
        "58-year-old man with asymptomatic chest pain, resting BP 140, cholesterol 289, fasting blood "
        "sugar normal, normal resting ECG, max heart rate 125, angina on exercise, oldpeak 1.5, flat ST slope."
    ),
    "Missing details": "I'm a 45 year old woman with a blood pressure of 130/85. What's my heart disease risk?",
    "Out of scope": "What medication should I take for my high blood pressure?",
}


@st.cache_resource
def load_predictor() -> HeartRiskPredictor:
    return HeartRiskPredictor()


def make_llm():
    try:
        return NebiusLLM(), None
    except Exception as exc:
        return None, str(exc)


def render_prediction(pred) -> None:
    c1, c2 = st.columns([1, 2])
    c1.metric("Estimated probability", f"{pred.probability:.0%}", pred.risk_band.capitalize() + " risk",
              delta_color="off")
    if pred.contributions:
        impacts = pd.DataFrame(
            {"impact (percentage points)": [
                c["impact"] * 100 for c in pred.contributions]},
            index=[
                f"{FEATURE_SPEC[c['feature']]['label']} = {c['patient_value']}" for c in pred.contributions],
        )
        c2.caption("What moved this estimate (vs. a typical patient)")
        c2.bar_chart(impacts, horizontal=True, height=180)


# ---------------------------------------------------------------- state
try:
    predictor = load_predictor()
except FileNotFoundError as exc:
    st.error(str(exc))
    st.stop()

if "assistant" not in st.session_state:
    llm, llm_error = make_llm()
    st.session_state.assistant = HeartRiskAssistant(predictor, llm)
    st.session_state.llm_error = llm_error
    st.session_state.messages = []
assistant: HeartRiskAssistant = st.session_state.assistant

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Patient values")
    for f in REQUIRED:
        icon = "✅" if f in assistant.known else "⬜"
        shown = assistant.known.get(f, "—")
        shown = "not measured" if f in assistant.known and shown is None else shown
        st.write(
            f"{icon} **{FEATURE_SPEC[f]['label'][0].upper() + FEATURE_SPEC[f]['label'][1:]}**: {shown}")
    if st.button("New patient / reset", use_container_width=True):
        assistant.reset()
        st.session_state.messages = []
        st.rerun()

    with st.expander("Manual entry (no LLM needed)"):
        with st.form("manual"):
            vals = {
                "Age": st.number_input("Age", 18, 100, 55),
                "Sex": st.selectbox("Sex", ["M", "F"]),
                "ChestPainType": st.selectbox("Chest pain type", ["ASY", "NAP", "ATA", "TA"],
                                              help="ASY asymptomatic, NAP non-anginal, ATA atypical, TA typical"),
                "RestingBP": st.number_input("Resting BP (mm Hg)", 70, 220, 130),
                "Cholesterol": st.number_input("Cholesterol (mg/dL)", 85, 650, 220),
                "FastingBS": st.selectbox("Fasting blood sugar > 120 mg/dL", [0, 1]),
                "RestingECG": st.selectbox("Resting ECG", ["Normal", "ST", "LVH"]),
                "MaxHR": st.number_input("Max heart rate", 60, 220, 140),
                "ExerciseAngina": st.selectbox("Exercise angina", ["N", "Y"]),
                "Oldpeak": st.number_input("Oldpeak (mm)", -3.0, 7.0, 1.0, step=0.1),
                "ST_Slope": st.selectbox("ST slope", ["Up", "Flat", "Down"]),
            }
            if st.form_submit_button("Predict"):
                assistant.set_features(vals)
                pred = predictor.predict(assistant.known)
                text = assistant.explain(pred, [])
                st.session_state.messages.append(
                    {"role": "user", "content": "_(Manual entry submitted)_"})
                st.session_state.messages.append(
                    {"role": "assistant", "content": text, "prediction": pred})
                st.rerun()

    st.divider()
    meta = predictor.metadata
    st.caption(
        f"Model: **{meta['run_name']}** ({meta['model_type']}) · MLflow run `{meta['run_id'][:8]}`")
    m = meta["metrics"]
    st.caption(f"Test AUC {m['test_roc_auc']:.2f} · Recall {m['test_recall']:.2f} · "
               f"Precision {m['test_precision']:.2f} · F1 {m['test_f1']:.2f}")

# ---------------------------------------------------------------- main
st.title("🫀 Heart Risk Assistant")
st.caption("Describe a patient in plain English. A model trained on 918 patients estimates the probability "
           "of heart disease and explains what drove it. Educational tool, not a diagnosis.")

if st.session_state.llm_error:
    st.warning(
        f"LLM unavailable: {st.session_state.llm_error} Chat is disabled; use Manual entry in the sidebar.")

if not st.session_state.messages:
    st.write("Try an example:")
    cols = st.columns(len(EXAMPLES))
    for col, (label, text) in zip(cols, EXAMPLES.items()):
        if col.button(label, use_container_width=True):
            st.session_state.pending = text
            st.rerun()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("prediction") is not None:
            render_prediction(msg["prediction"])
        if msg.get("extracted"):
            with st.expander("What the system parsed"):
                st.json(msg["extracted"])

prompt = st.chat_input("e.g. 62-year-old woman, typical angina, BP 150 ...",
                       disabled=assistant.llm is None) or st.session_state.pop("pending", None)
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.spinner("Reading your message and running the model..."):
        reply = assistant.handle(prompt)
    st.session_state.messages.append({"role": "assistant", "content": reply.text,
                                      "prediction": reply.prediction, "extracted": reply.extracted})
    st.rerun()
