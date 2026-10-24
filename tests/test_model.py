"""Validation of the exported best model (models/best_model.joblib)."""
import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from conftest import HIGH_RISK, LOW_RISK
from src.preprocess import clean_data, feature_columns, load_data, make_splits

# Minimum test-set ROC AUC. Published models on this dataset typically reach ~0.90;
# anything below this indicates a broken pipeline or a bad model export.
MIN_TEST_AUC = 0.75


@pytest.fixture(scope="module")
def test_split(cfg):
    df = clean_data(load_data(cfg["data"]["path"]), cfg["data"]["zero_as_missing"])
    df = df[feature_columns(cfg["features"]) + [cfg["data"]["target"]]]
    _, X_test, y_train, y_test = make_splits(df, cfg)  # same seed as training -> same held-out rows
    return X_test, y_train, y_test


def test_prediction_type_and_shape(trained_predictor, test_split):
    X_test, _, _ = test_split
    proba = trained_predictor.pipeline.predict_proba(X_test)
    labels = trained_predictor.pipeline.predict(X_test)
    assert proba.shape == (len(X_test), 2)
    assert np.all((proba >= 0) & (proba <= 1))
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)
    assert labels.shape == (len(X_test),)
    assert set(np.unique(labels)) <= {0, 1}


def test_meets_minimum_performance_on_held_out_data(trained_predictor, test_split):
    X_test, y_train, y_test = test_split
    proba = trained_predictor.pipeline.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, proba)
    accuracy = np.mean((proba >= 0.5) == y_test)
    majority = max(y_train.mean(), 1 - y_train.mean())
    assert auc >= MIN_TEST_AUC, f"AUC {auc:.3f} below {MIN_TEST_AUC}"
    assert accuracy > majority, "model must beat always predicting the majority class"


def test_single_patient_prediction_with_explanation(trained_predictor):
    pred = trained_predictor.predict(HIGH_RISK)
    assert isinstance(pred.probability, float) and 0 <= pred.probability <= 1
    assert pred.label in (0, 1)
    assert pred.risk_band in {"low", "moderate", "high"}
    assert all(c["feature"] in HIGH_RISK for c in pred.contributions)


def test_risk_ordering_is_clinically_sensible(trained_predictor):
    assert trained_predictor.predict(HIGH_RISK).probability > trained_predictor.predict(LOW_RISK).probability


def test_missing_cholesterol_is_handled(trained_predictor):
    patient = {**LOW_RISK, "Cholesterol": None}
    assert 0 <= trained_predictor.predict(patient).probability <= 1


def test_incomplete_input_is_rejected(trained_predictor):
    partial = {k: v for k, v in HIGH_RISK.items() if k != "ST_Slope"}
    with pytest.raises(ValueError, match="ST_Slope"):
        trained_predictor.predict(partial)
