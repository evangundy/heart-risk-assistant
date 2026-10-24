import numpy as np
import pandas as pd
import pytest

from src.preprocess import (
    CATEGORY_LEVELS,
    build_preprocessor,
    clean_data,
    feature_columns,
    make_splits,
    split_features_target,
)

ZEROS = ["RestingBP", "Cholesterol"]


def fitted(raw_df, cfg):
    X, _ = split_features_target(clean_data(raw_df, ZEROS), "HeartDisease")
    pre = build_preprocessor(cfg["features"])
    out = pre.fit_transform(X[feature_columns(cfg["features"])])
    return pd.DataFrame(out, columns=pre.get_feature_names_out()), pre


def test_zero_placeholders_become_missing(raw_df):
    cleaned = clean_data(raw_df, ZEROS)
    assert cleaned.loc[4, "Cholesterol"] != cleaned.loc[4, "Cholesterol"]  # NaN
    assert np.isnan(cleaned.loc[5, "RestingBP"])
    # genuine zeros in other columns (Oldpeak, FastingBS) must be preserved
    assert (cleaned["Oldpeak"] == 0).sum() == (raw_df["Oldpeak"] == 0).sum()
    assert (cleaned["FastingBS"] == 0).sum() == (raw_df["FastingBS"] == 0).sum()


def test_missing_values_are_imputed_with_indicator(raw_df, cfg):
    out, _ = fitted(raw_df, cfg)
    assert not out.isna().any().any()
    indicator = [c for c in out.columns if "missingindicator_Cholesterol" in c]
    assert indicator, "a missing-indicator column for Cholesterol should exist"
    flag = out[indicator[0]]  # standardized along with the numerics, so test relative values
    assert flag.idxmax() == 4 and (flag.drop(4) == flag.min()).all()


def test_categoricals_are_one_hot_encoded(raw_df, cfg):
    out, _ = fitted(raw_df, cfg)
    for col, levels in CATEGORY_LEVELS.items():
        cols = [f"cat__{col}_{lvl}" for lvl in levels]
        assert all(c in out.columns for c in cols)
        assert (out[cols].sum(axis=1) == 1).all(), f"{col} rows should have exactly one active level"
    assert out.loc[0, "cat__ChestPainType_ATA"] == 1
    assert out.loc[0, "cat__ChestPainType_ASY"] == 0


def test_unseen_category_does_not_crash(raw_df, cfg):
    _, pre = fitted(raw_df, cfg)
    row = clean_data(raw_df, ZEROS).drop(columns="HeartDisease").head(1).copy()
    row["RestingECG"] = "Unknown"
    out = pd.DataFrame(pre.transform(row), columns=pre.get_feature_names_out())
    assert out.filter(like="cat__RestingECG_").sum(axis=1).iloc[0] == 0


def test_numeric_features_are_standardized(raw_df, cfg):
    out, _ = fitted(raw_df, cfg)
    for col in cfg["features"]["numeric"]:
        values = out[f"num__{col}"]
        assert values.mean() == pytest.approx(0, abs=1e-9)
        assert values.std(ddof=0) == pytest.approx(1, abs=1e-9)
    assert set(out["bin__FastingBS"].unique()) <= {0, 1}  # binary flag left as 0/1


def test_original_dataframe_is_not_modified(raw_df, cfg):
    snapshot = raw_df.copy(deep=True)
    cleaned = clean_data(raw_df, ZEROS)
    fitted(raw_df, cfg)
    pd.testing.assert_frame_equal(raw_df, snapshot)
    assert cleaned is not raw_df


def test_split_is_stratified_and_disjoint(cfg, rng):
    n = 200
    df = pd.DataFrame({c: rng.normal(size=n) for c in ["a", "b"]})
    df["HeartDisease"] = (rng.random(n) < 0.3).astype(int)
    local = {"data": {**cfg["data"], "target": "HeartDisease"}}
    X_train, X_test, y_train, y_test = make_splits(df, local)
    assert set(X_train.index).isdisjoint(X_test.index)
    assert abs(y_train.mean() - y_test.mean()) < 0.05
