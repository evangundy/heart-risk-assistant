"""Load the exported best model and run inference with per-feature explanations."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import pandas as pd

from src.preprocess import PROJECT_ROOT

MODEL_DIR = PROJECT_ROOT / "models"

# Communication bands for the predicted probability. The binary decision
# threshold lives in config/best_model.json; these only shape the wording.
RISK_BANDS = [(0.30, "low"), (0.60, "moderate"), (1.01, "high")]


@dataclass
class Prediction:
    probability: float
    label: int
    risk_band: str
    contributions: list[dict] = field(default_factory=list)
    features: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "probability": round(self.probability, 4),
            "predicted_label": "heart disease likely" if self.label else "heart disease unlikely",
            "risk_band": self.risk_band,
            "top_factors": self.contributions,
            "inputs": self.features,
        }


def risk_band(p: float) -> str:
    return next(name for upper, name in RISK_BANDS if p < upper)


class HeartRiskPredictor:
    """Wraps the trained sklearn pipeline (preprocessing + classifier)."""

    def __init__(self, model_dir: str | Path = MODEL_DIR):
        model_dir = Path(model_dir)
        model_path, meta_path = model_dir / "best_model.joblib", model_dir / "best_model.json"
        if not model_path.exists():
            raise FileNotFoundError(
                f"No trained model at {model_path}. Run `python -m src.train` and then "
                "`python -m src.compare_runs` first."
            )
        self.pipeline = joblib.load(model_path)
        self.metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        self.feature_columns: list[str] = self.metadata["feature_columns"]
        self.reference: dict = self.metadata["reference_profile"]
        self.threshold: float = self.metadata.get("decision_threshold", 0.5)

    def _frame(self, rows: list[dict]) -> pd.DataFrame:
        df = pd.DataFrame(rows)[self.feature_columns]
        numeric = df.columns[df.columns.map(lambda c: not isinstance(self.reference[c], str))]
        df[numeric] = df[numeric].astype("float64")
        return df

    def predict_proba(self, features: dict) -> float:
        return float(self.pipeline.predict_proba(self._frame([features]))[0, 1])

    def explain(self, features: dict, top_k: int = 4) -> list[dict]:
        """Model-agnostic local explanation.

        For each feature, swap the patient's value for the typical training
        value and measure how much the predicted probability changes. A
        positive impact means this patient's value *raises* their risk
        relative to a typical patient. Simple, fast (one batch of 11 rows),
        and easy to explain to a non-technical user.
        """
        base = self.predict_proba(features)
        rows = []
        for col in self.feature_columns:
            swapped = dict(features)
            swapped[col] = self.reference[col]
            rows.append(swapped)
        probs = self.pipeline.predict_proba(self._frame(rows))[:, 1]
        impacts = [
            {
                "feature": col,
                "patient_value": features[col],
                "typical_value": self.reference[col],
                "impact": round(base - float(p), 4),
            }
            for col, p in zip(self.feature_columns, probs)
            if features[col] != self.reference[col]
        ]
        impacts.sort(key=lambda d: abs(d["impact"]), reverse=True)
        return [d for d in impacts if abs(d["impact"]) >= 0.005][:top_k]

    def predict(self, features: dict) -> Prediction:
        missing = [c for c in self.feature_columns if c not in features]
        if missing:
            raise ValueError(f"Missing features: {missing}")
        clean = {c: features[c] for c in self.feature_columns}
        p = self.predict_proba(clean)
        return Prediction(
            probability=p,
            label=int(p >= self.threshold),
            risk_band=risk_band(p),
            contributions=self.explain(clean),
            features=clean,
        )
