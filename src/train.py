"""Train every experiment in configs/config.yaml and log each one to MLflow.

Usage:
    python -m src.train                       # run all experiments
    python -m src.train --only xgboost_shallow
    python -m src.train --config configs/config.yaml
"""
from __future__ import annotations

import argparse
import hashlib
import logging
import tempfile

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.evaluate import compute_metrics, majority_baseline, save_diagnostic_plots
from src.preprocess import (
    DEFAULT_CONFIG,
    PROJECT_ROOT,
    build_preprocessor,
    clean_data,
    feature_columns,
    load_config,
    load_data,
    make_splits,
    resolve_path,
)

# We knowingly use cloudpickle for our own models; keep the console readable.
logging.getLogger("mlflow.sklearn").setLevel(logging.ERROR)

MODEL_REGISTRY = {
    "logistic_regression": LogisticRegression,
    "random_forest": RandomForestClassifier,
    "gradient_boosting": GradientBoostingClassifier,
    "xgboost": XGBClassifier,
}


def build_model(model_type: str, params: dict, random_state: int):
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model type '{model_type}'. Options: {list(MODEL_REGISTRY)}")
    params = dict(params)
    params.setdefault("random_state", random_state)
    if model_type == "xgboost":
        params.setdefault("eval_metric", "logloss")
    return MODEL_REGISTRY[model_type](**params)


def build_pipeline(cfg: dict, exp: dict) -> Pipeline:
    return Pipeline([
        ("preprocess", build_preprocessor(cfg["features"])),
        ("model", build_model(exp["model"], exp.get("params", {}), cfg["data"]["random_state"])),
    ])


def file_md5(path) -> str:
    h = hashlib.md5()
    with open(resolve_path(path), "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def tracking_uri(cfg: dict) -> str:
    uri = cfg["mlflow"]["tracking_uri"]
    if uri.startswith("sqlite:///") and not uri.startswith("sqlite:////"):
        uri = f"sqlite:///{PROJECT_ROOT / uri.removeprefix('sqlite:///')}"
    elif "://" not in uri:
        uri = str(resolve_path(uri))
    return uri


def run_experiment(cfg: dict, exp: dict, splits, data_info: dict) -> str:
    X_train, X_test, y_train, y_test = splits
    threshold = cfg["training"]["decision_threshold"]
    pipeline = build_pipeline(cfg, exp)

    with mlflow.start_run(run_name=exp["name"]) as run:
        # ---- parameters: model hyperparameters + data/version description ----
        mlflow.log_params({f"model__{k}": v for k, v in exp.get("params", {}).items()})
        mlflow.log_params({
            "model_type": exp["model"],
            "decision_threshold": threshold,
            "cv_folds": cfg["training"]["cv_folds"],
            **data_info,
        })
        mlflow.set_tags({"experiment_name": exp["name"], "task": "binary_classification"})

        # ---- cross-validated AUC on the training split (model stability) ----
        cv = StratifiedKFold(cfg["training"]["cv_folds"], shuffle=True,
                             random_state=cfg["data"]["random_state"])
        cv_auc = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="roc_auc")

        # ---- fit on train, evaluate on the untouched test split ----
        pipeline.fit(X_train, y_train)
        y_proba = pipeline.predict_proba(X_test)[:, 1]
        metrics = compute_metrics(y_test, y_proba, threshold)
        metrics.update(compute_metrics(y_train, pipeline.predict_proba(X_train)[:, 1],
                                       threshold, prefix="train_"))
        metrics.update(majority_baseline(y_train, y_test))
        metrics["cv_roc_auc_mean"] = float(cv_auc.mean())
        metrics["cv_roc_auc_std"] = float(cv_auc.std())
        mlflow.log_metrics(metrics)

        # ---- artifacts: diagnostic plots + the full pipeline as a model ----
        with tempfile.TemporaryDirectory() as tmp:
            for p in save_diagnostic_plots(y_test, y_proba, tmp, threshold):
                mlflow.log_artifact(str(p), artifact_path="plots")
        signature = infer_signature(X_test, pipeline.predict_proba(X_test))
        # cloudpickle keeps the whole pipeline (incl. XGBoost) loadable with one call;
        # only load models you trained yourself.
        mlflow.sklearn.log_model(
            pipeline, name="model", signature=signature, input_example=X_test.head(3),
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
        )

        print(f"{exp['name']:<28} AUC={metrics['test_roc_auc']:.3f}  "
              f"recall={metrics['test_recall']:.3f}  F1={metrics['test_f1']:.3f}  "
              f"acc={metrics['test_accuracy']:.3f}  (CV AUC {cv_auc.mean():.3f}±{cv_auc.std():.3f})")
        return run.info.run_id


def main(argv=None) -> list[str]:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--only", nargs="*", help="Run only the named experiments")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    raw = load_data(cfg["data"]["path"])
    df = clean_data(raw, cfg["data"]["zero_as_missing"])
    df = df[feature_columns(cfg["features"]) + [cfg["data"]["target"]]]
    # Float columns let the logged MLflow signature accept missing values at inference time.
    num_cols = cfg["features"]["numeric"] + cfg["features"]["binary"]
    df[num_cols] = df[num_cols].astype("float64")
    splits = make_splits(df, cfg)

    data_info = {
        "data_path": cfg["data"]["path"],
        "data_md5": file_md5(cfg["data"]["path"]),
        "data_description": cfg["data"]["description"][:250],
        "n_rows": len(df),
        "n_train": len(splits[0]),
        "n_test": len(splits[1]),
        "test_size": cfg["data"]["test_size"],
        "split_random_state": cfg["data"]["random_state"],
        "positive_rate": round(float(df[cfg["data"]["target"]].mean()), 4),
        "features": ",".join(feature_columns(cfg["features"])),
    }

    mlflow.set_tracking_uri(tracking_uri(cfg))
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])

    experiments = cfg["experiments"]
    if args.only:
        experiments = [e for e in experiments if e["name"] in args.only]
    print(f"Training {len(experiments)} configurations on {len(df)} rows "
          f"(train={data_info['n_train']}, test={data_info['n_test']})\n")
    run_ids = [run_experiment(cfg, exp, splits, data_info) for exp in experiments]
    print("\nDone. Next: python -m src.compare_runs   (or open the MLflow UI)")
    return run_ids


if __name__ == "__main__":
    main()
