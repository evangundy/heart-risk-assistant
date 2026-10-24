"""Data loading, cleaning, and feature preprocessing.

Design notes
------------
* ``clean_data`` handles dataset-specific quirks *before* splitting. It is
  stateless (no statistics are learned), so it cannot leak test information.
* Everything that learns from data (imputation medians, scaling, one-hot
  categories) lives in the sklearn ``ColumnTransformer`` returned by
  ``build_preprocessor``. It is embedded in the model pipeline and fit on the
  training split only, so the saved model carries its own preprocessing and
  the app can pass raw feature values straight in.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "config.yaml"

# Valid category codes in the source data (used by the app's validator, too).
CATEGORY_LEVELS: dict[str, list[str]] = {
    "Sex": ["M", "F"],
    "ChestPainType": ["TA", "ATA", "NAP", "ASY"],
    "RestingECG": ["Normal", "ST", "LVH"],
    "ExerciseAngina": ["Y", "N"],
    "ST_Slope": ["Up", "Flat", "Down"],
}


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(path: str | Path) -> Path:
    """Resolve config paths relative to the project root."""
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def load_data(path: str | Path) -> pd.DataFrame:
    path = resolve_path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Download heart.csv from "
            "https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction "
            "and place it in the data/ folder (see README)."
        )
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame, zero_as_missing: list[str] | None = None) -> pd.DataFrame:
    """Return a cleaned copy of ``df``; the input is never modified.

    * Strips whitespace from text columns.
    * Converts physiologically impossible zeros (e.g. Cholesterol = 0, which
      the source data uses for "not measured") to NaN so they are imputed
      rather than treated as real measurements.
    * Drops exact duplicate rows.
    """
    out = df.copy()
    for col in out.select_dtypes(include=["object", "string"]).columns:
        out[col] = out[col].str.strip()
    for col in zero_as_missing or []:
        if col in out.columns:
            out[col] = out[col].replace(0, np.nan)
    return out.drop_duplicates().reset_index(drop=True)


def split_features_target(df: pd.DataFrame, target: str) -> tuple[pd.DataFrame, pd.Series]:
    return df.drop(columns=[target]), df[target].astype(int)


def make_splits(df: pd.DataFrame, cfg: dict):
    """Stratified train/test split driven by the config."""
    X, y = split_features_target(df, cfg["data"]["target"])
    return train_test_split(
        X, y,
        test_size=cfg["data"]["test_size"],
        random_state=cfg["data"]["random_state"],
        stratify=y,
    )


def build_preprocessor(features: dict) -> ColumnTransformer:
    """Build the (unfitted) preprocessing transformer.

    numeric     -> median imputation (+ missing-indicator flags) -> standard scaling
    binary      -> most-frequent imputation, kept as 0/1
    categorical -> most-frequent imputation -> one-hot encoding

    The missing-indicator matters clinically: in this dataset, whether
    cholesterol was recorded at all differs between hospital cohorts and
    carries signal, so we keep that information instead of hiding it.
    """
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
    ])
    binary = Pipeline([("impute", SimpleImputer(strategy="most_frequent"))])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(
            categories=[CATEGORY_LEVELS[c] for c in features["categorical"]],
            handle_unknown="ignore",
            sparse_output=False,
        )),
    ])
    return ColumnTransformer(
        [
            ("num", numeric, features["numeric"]),
            ("bin", binary, features["binary"]),
            ("cat", categorical, features["categorical"]),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )


def feature_columns(features: dict) -> list[str]:
    """The raw input columns the model expects, in order."""
    return features["numeric"] + features["binary"] + features["categorical"]
