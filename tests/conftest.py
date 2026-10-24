import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.preprocess import load_config  # noqa: E402


@pytest.fixture(scope="session")
def cfg():
    return load_config()


@pytest.fixture
def raw_df():
    """Small hand-made sample in the exact format of heart.csv, including the
    dataset's real quirks: Cholesterol = 0 and RestingBP = 0 meaning 'not recorded'."""
    return pd.DataFrame({
        "Age": [40, 49, 37, 48, 54, 39, 45, 58],
        "Sex": ["M", "F", "M", "F", "M", "M", "F", "M"],
        "ChestPainType": ["ATA", "NAP", "ATA", "ASY", "NAP", "NAP", "ATA", "ASY"],
        "RestingBP": [140, 160, 130, 138, 150, 0, 130, 136],
        "Cholesterol": [289, 180, 283, 214, 0, 339, 237, 164],
        "FastingBS": [0, 0, 0, 0, 0, 0, 0, 1],
        "RestingECG": ["Normal", "Normal", "ST", "Normal", "Normal", "Normal", "Normal", "ST"],
        "MaxHR": [172, 156, 98, 108, 122, 170, 170, 99],
        "ExerciseAngina": ["N", "N", "N", "Y", "N", "N", "N", "Y"],
        "Oldpeak": [0.0, 1.0, 0.0, 1.5, 0.0, 0.0, 0.0, 2.0],
        "ST_Slope": ["Up", "Flat", "Up", "Flat", "Up", "Up", "Up", "Flat"],
        "HeartDisease": [0, 1, 0, 1, 0, 0, 0, 1],
    })


HIGH_RISK = {"Age": 65, "Sex": "M", "ChestPainType": "ASY", "RestingBP": 150, "Cholesterol": 280,
             "FastingBS": 1, "RestingECG": "ST", "MaxHR": 105, "ExerciseAngina": "Y",
             "Oldpeak": 3.0, "ST_Slope": "Flat"}
LOW_RISK = {"Age": 35, "Sex": "F", "ChestPainType": "ATA", "RestingBP": 118, "Cholesterol": 190,
            "FastingBS": 0, "RestingECG": "Normal", "MaxHR": 180, "ExerciseAngina": "N",
            "Oldpeak": 0.0, "ST_Slope": "Up"}


@pytest.fixture(scope="session")
def trained_predictor():
    """The real exported model. Tests using it are skipped (not failed) until
    `python -m src.train && python -m src.compare_runs` has been run."""
    if not (ROOT / "models" / "best_model.joblib").exists():
        pytest.skip("No trained model yet: run `python -m src.train` then `python -m src.compare_runs`.")
    from src.predictor import HeartRiskPredictor
    return HeartRiskPredictor()


class FakeLLM:
    """Deterministic stand-in for the Nebius API so interface tests run offline.

    `extractions` maps a user message to the JSON the LLM would return.
    """

    def __init__(self, extractions: dict[str, dict], fail: bool = False):
        self.extractions = extractions
        self.fail = fail
        self.calls: list[list[dict]] = []

    def chat(self, messages, json_mode=False, temperature=0.0):
        self.calls.append(messages)
        if self.fail:
            raise ConnectionError("simulated outage")
        if json_mode:
            return json.dumps(self.extractions[messages[-1]["content"]])
        return "EXPLANATION: " + messages[-1]["content"][:60]


class SpyPredictor:
    """Records whether the model was invoked; returns a fixed result."""

    metadata = {"model_type": "test", "metrics": {"test_roc_auc": 0.9}}

    def __init__(self):
        self.calls = []

    def predict(self, features):
        from src.predictor import Prediction
        self.calls.append(dict(features))
        return Prediction(probability=0.73, label=1, risk_band="high",
                          contributions=[{"feature": "ST_Slope", "patient_value": "Flat",
                                          "typical_value": "Up", "impact": 0.21}],
                          features=dict(features))


@pytest.fixture
def make_assistant():
    from src.llm_interface import HeartRiskAssistant

    def _make(extractions, fail=False):
        predictor = SpyPredictor()
        return HeartRiskAssistant(predictor, FakeLLM(extractions, fail=fail)), predictor
    return _make


@pytest.fixture
def rng():
    return np.random.default_rng(0)
