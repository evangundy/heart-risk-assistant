"""Compare MLflow runs with mlflow.search_runs(), pick the best one, and export it.

Selection rule (from configs/config.yaml -> selection):
  1. Disqualify runs whose test recall is below `min_recall`. For a screening
     tool, missing a sick patient (false negative) is the costliest error.
  2. Among the remaining runs, choose the highest `primary_metric` (test ROC AUC),
     breaking ties with the smaller gap between CV and test AUC (more stable).

Outputs:
  models/best_model.joblib       the winning sklearn pipeline (preprocessing + model)
  models/best_model.json         run id, metrics, and reference values used for explanations
  reports/model_comparison.csv   every run, ranked
  reports/model_comparison.md    the same table in Markdown, ready for the README

Usage:
    python -m src.compare_runs
"""
from __future__ import annotations

import argparse
import json

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from src.preprocess import (
    DEFAULT_CONFIG,
    PROJECT_ROOT,
    clean_data,
    feature_columns,
    load_config,
    load_data,
    make_splits,
    resolve_path,
)
from src.train import tracking_uri

REPORT_COLUMNS = {
    "tags.mlflow.runName": "run",
    "params.model_type": "model",
    "metrics.test_roc_auc": "AUC",
    "metrics.test_recall": "recall",
    "metrics.test_precision": "precision",
    "metrics.test_f1": "F1",
    "metrics.test_accuracy": "accuracy",
    "metrics.test_specificity": "specificity",
    "metrics.cv_roc_auc_mean": "CV AUC",
    "metrics.train_roc_auc": "train AUC",
}


def fetch_runs(cfg: dict) -> pd.DataFrame:
    mlflow.set_tracking_uri(tracking_uri(cfg))
    runs = mlflow.search_runs(
        experiment_names=[cfg["mlflow"]["experiment_name"]],
        filter_string="attributes.status = 'FINISHED'",
        order_by=[f"metrics.{cfg['selection']['primary_metric']} DESC"],
    )
    if runs.empty:
        raise SystemExit("No finished runs found. Run `python -m src.train` first.")
    # If an experiment was re-run, keep only its most recent run.
    runs = runs.sort_values("start_time", ascending=False).drop_duplicates("tags.mlflow.runName")
    return runs


def rank_runs(runs: pd.DataFrame, primary_metric: str, min_recall: float) -> pd.DataFrame:
    ranked = runs.copy()
    ranked["eligible"] = ranked["metrics.test_recall"] >= min_recall
    ranked["auc_gap"] = (ranked["metrics.cv_roc_auc_mean"] - ranked["metrics.test_roc_auc"]).abs()
    return ranked.sort_values(
        ["eligible", f"metrics.{primary_metric}", "auc_gap"],
        ascending=[False, False, True],
    ).reset_index(drop=True)


def reference_profile(cfg: dict) -> dict:
    """Typical training-set values (median / most common), used as the
    'average patient' baseline when explaining individual predictions."""
    df = clean_data(load_data(cfg["data"]["path"]), cfg["data"]["zero_as_missing"])
    X_train = make_splits(df, cfg)[0]
    ref = {}
    for col in cfg["features"]["numeric"]:
        ref[col] = float(X_train[col].median())
    for col in cfg["features"]["binary"] + cfg["features"]["categorical"]:
        ref[col] = X_train[col].mode().iloc[0]
        ref[col] = ref[col].item() if hasattr(ref[col], "item") else ref[col]
    return ref


def main(argv=None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    sel = cfg["selection"]

    ranked = rank_runs(fetch_runs(cfg), sel["primary_metric"], sel["min_recall"])
    table = ranked[list(REPORT_COLUMNS) + ["eligible"]].rename(columns=REPORT_COLUMNS)
    for col in table.select_dtypes("number").columns:
        table[col] = table[col].round(3)

    reports = PROJECT_ROOT / "reports"
    reports.mkdir(exist_ok=True)
    table.to_csv(reports / "model_comparison.csv", index=False)
    (reports / "model_comparison.md").write_text(table.to_markdown(index=False), encoding="utf-8")

    pd.set_option("display.width", 200)
    print(table.to_string(index=False))

    best = ranked.iloc[0]
    if not best["eligible"]:
        print(f"\nWARNING: no run reached recall >= {sel['min_recall']}; using the best AUC anyway.")
    print(f"\nBest run: {best['tags.mlflow.runName']} (run_id={best['run_id']})")

    model = mlflow.sklearn.load_model(f"runs:/{best['run_id']}/model")
    out_dir = resolve_path(sel["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_dir / "best_model.joblib")

    metadata = {
        "run_id": best["run_id"],
        "run_name": best["tags.mlflow.runName"],
        "model_type": best["params.model_type"],
        "selection_rule": f"max {sel['primary_metric']} with test_recall >= {sel['min_recall']}",
        "decision_threshold": cfg["training"]["decision_threshold"],
        "metrics": {k: round(float(best[f"metrics.{k}"]), 4) for k in [
            "test_roc_auc", "test_recall", "test_precision", "test_f1",
            "test_accuracy", "test_specificity", "cv_roc_auc_mean"]},
        "feature_columns": feature_columns(cfg["features"]),
        "reference_profile": reference_profile(cfg),
        "data_md5": best["params.data_md5"],
    }
    (out_dir / "best_model.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Exported to {out_dir / 'best_model.joblib'}")
    return metadata


if __name__ == "__main__":
    main()
