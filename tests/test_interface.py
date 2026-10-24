"""Tests for the natural-language layer.

The LLM is replaced by a deterministic fake (see conftest.FakeLLM) so these
run offline and fast. The fake returns what the extraction prompt asks the
real LLM to return; the tests then verify everything *our* code does with it:
normalization to model codes, validation, clarifying questions, scope
handling, and that the model is only invoked with complete inputs.
An optional live test at the bottom calls the real Nebius API when a key is set.
"""
import os

import pytest

from src.llm_interface import REQUIRED, SCOPE_MESSAGE, normalize_features, normalize_value, parse_json

COMPLETE_MSG = ("58-year-old man, asymptomatic, BP 140/90, cholesterol 289, fasting glucose 135, "
                "normal ECG, max HR 125, angina on exercise, oldpeak 1.5, flat ST slope")
# What the LLM extracts from COMPLETE_MSG: natural wording, not yet model codes.
COMPLETE_EXTRACTION = {
    "intent": "predict",
    "features": {"Age": 58, "Sex": "male", "ChestPainType": "asymptomatic", "RestingBP": "140/90",
                 "Cholesterol": "289", "FastingBS": 135, "RestingECG": "normal", "MaxHR": 125,
                 "ExerciseAngina": "yes", "Oldpeak": 1.5, "ST_Slope": "flat"},
    "ambiguous": [], "unsupported": [],
}
EXPECTED_FEATURES = {"Age": 58, "Sex": "M", "ChestPainType": "ASY", "RestingBP": 140, "Cholesterol": 289,
                     "FastingBS": 1, "RestingECG": "Normal", "MaxHR": 125, "ExerciseAngina": "Y",
                     "Oldpeak": 1.5, "ST_Slope": "Flat"}


# ---------------------------------------------------------------- parsing
def test_parse_json_tolerates_code_fences_and_prose():
    assert parse_json('```json\n{"intent": "predict"}\n```') == {"intent": "predict"}
    assert parse_json('Sure! {"intent": "reset", "features": {}} Hope that helps.')["intent"] == "reset"


def test_natural_wording_is_normalized_to_model_codes():
    clean, errors = normalize_features(COMPLETE_EXTRACTION["features"])
    assert errors == {}
    assert clean == EXPECTED_FEATURES


@pytest.mark.parametrize("feature,raw,expected", [
    ("Sex", "Female", "F"),
    ("ChestPainType", "non-anginal pain", "NAP"),
    ("ChestPainType", "typical angina", "TA"),
    ("RestingECG", "left ventricular hypertrophy", "LVH"),
    ("ST_Slope", "downsloping", "Down"),
    ("FastingBS", 98, 0),            # mg/dL below 120
    ("FastingBS", "yes", 1),
    ("FastingBS", 7.5, 1),           # mmol/L (7.5 * 18 = 135 mg/dL)
    ("Cholesterol", 6.2, 240),       # mmol/L converted to mg/dL
    ("Cholesterol", "not measured", None),
    ("Oldpeak", "none", 0.0),
    ("Age", "45 years", 45),
])
def test_unit_and_synonym_handling(feature, raw, expected):
    value, error = normalize_value(feature, raw)
    assert error is None and value == expected


@pytest.mark.parametrize("feature,raw", [
    ("Age", 250), ("RestingBP", 20), ("MaxHR", "very high"), ("ChestPainType", "sharp"), ("Sex", "unknown"),
])
def test_invalid_values_are_rejected(feature, raw):
    value, error = normalize_value(feature, raw)
    assert value is None and error


def test_complete_message_runs_the_real_pipeline_with_parsed_features(make_assistant):
    assistant, predictor = make_assistant({COMPLETE_MSG: COMPLETE_EXTRACTION})
    reply = assistant.handle(COMPLETE_MSG)
    assert reply.kind == "prediction"
    assert predictor.calls == [EXPECTED_FEATURES]
    assert reply.text.startswith("EXPLANATION")  # response came from the LLM step


# ---------------------------------------------------------------- edge cases
def test_incomplete_input_asks_for_missing_features_and_does_not_predict(make_assistant):
    msg = "I'm a 45 year old woman with BP 130/85. What's my risk?"
    assistant, predictor = make_assistant({msg: {"intent": "predict",
                                                 "features": {"Age": 45, "Sex": "female", "RestingBP": 130}}})
    reply = assistant.handle(msg)
    assert reply.kind == "clarify"
    assert predictor.calls == []
    assert "max" in reply.text.lower() and "st segment" in reply.text.lower()
    assert assistant.known == {"Age": 45, "Sex": "F", "RestingBP": 130}


def test_details_can_be_completed_over_several_turns(make_assistant):
    first = {k: v for k, v in COMPLETE_EXTRACTION["features"].items() if k not in {"MaxHR", "ST_Slope"}}
    msgs = {"turn1": {"intent": "predict", "features": first},
            "turn2": {"intent": "provide_info", "features": {"MaxHR": "125 bpm", "ST_Slope": "flat"}}}
    assistant, predictor = make_assistant(msgs)
    assert assistant.handle("turn1").kind == "clarify"
    assert assistant.handle("turn2").kind == "prediction"
    assert predictor.calls == [EXPECTED_FEATURES]


def test_ambiguous_values_trigger_a_clarifying_question(make_assistant):
    msg = "He's in his 50s with high cholesterol"
    extraction = {"intent": "predict", "features": {"Sex": "M"},
                  "ambiguous": [{"feature": "Age", "reason": "only a decade was given"},
                                {"feature": "Cholesterol", "reason": "'high' is not a number"}]}
    assistant, predictor = make_assistant({msg: extraction})
    reply = assistant.handle(msg)
    assert reply.kind == "clarify" and predictor.calls == []
    assert "only a decade was given" in reply.text and "Age" not in assistant.known


def test_implausible_value_is_questioned_not_predicted(make_assistant):
    msg = "patient is 250 years old"
    assistant, predictor = make_assistant({msg: {"intent": "predict", "features": {"Age": 250}}})
    reply = assistant.handle(msg)
    assert reply.kind == "invalid" and "plausible range" in reply.text
    assert predictor.calls == []


def test_out_of_scope_question_is_declined(make_assistant):
    msg = "What medication should I take for blood pressure?"
    assistant, predictor = make_assistant({msg: {"intent": "out_of_scope", "features": {}}})
    reply = assistant.handle(msg)
    assert reply.kind == "out_of_scope" and reply.text == SCOPE_MESSAGE
    assert predictor.calls == []


def test_unsupported_facts_are_acknowledged(make_assistant):
    msg = "55 year old male smoker"
    assistant, _ = make_assistant({msg: {"intent": "predict", "features": {"Age": 55, "Sex": "M"},
                                         "unsupported": ["smoking status"]}})
    assert "smoking status" in assistant.handle(msg).text


def test_llm_outage_is_reported_gracefully(make_assistant):
    assistant, predictor = make_assistant({}, fail=True)
    reply = assistant.handle("anything")
    assert reply.kind == "error" and "manual entry" in reply.text.lower()
    assert predictor.calls == []


def test_reset_clears_collected_values(make_assistant):
    assistant, _ = make_assistant({"a": {"intent": "predict", "features": {"Age": 50}},
                                   "new patient": {"intent": "reset", "features": {}}})
    assistant.handle("a")
    assert assistant.handle("new patient").kind == "reset" and assistant.known == {}


# ---------------------------------------------------------------- live LLM (optional)
@pytest.mark.skipif(not os.getenv("NEBIUS_API_KEY"), reason="set NEBIUS_API_KEY to run the live LLM test")
def test_live_llm_extracts_features_from_natural_language(make_assistant):
    from src.llm_interface import HeartRiskAssistant, NebiusLLM
    from conftest import SpyPredictor

    assistant = HeartRiskAssistant(SpyPredictor(), NebiusLLM())
    extraction = assistant.extract(COMPLETE_MSG)
    clean, errors = normalize_features(extraction["features"])
    assert not errors
    assert set(clean) == set(REQUIRED)
    assert clean == EXPECTED_FEATURES
