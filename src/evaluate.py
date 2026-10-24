"""Model evaluation utilities: metrics and diagnostic plots."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: no display needed
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(y_true, y_proba, threshold: float = 0.5, prefix: str = "test_") -> dict[str, float]:
    """Classification metrics from predicted probabilities of the positive class.

    Recall (sensitivity) and specificity are reported separately because in a
    screening context a missed case (false negative) costs far more than a
    false alarm.
    """
    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba, dtype=float)
    y_pred = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "specificity": tn / (tn + fp) if (tn + fp) else 0.0,
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "false_negatives": float(fn),
    }
    return {f"{prefix}{k}": float(v) for k, v in metrics.items()}


def majority_baseline(y_train, y_test) -> dict[str, float]:
    """Accuracy of always predicting the most common training class.

    Any useful model must clearly beat this.
    """
    majority = int(np.bincount(np.asarray(y_train).astype(int)).argmax())
    return {"baseline_accuracy": float(np.mean(np.asarray(y_test) == majority))}


def save_diagnostic_plots(y_true, y_proba, out_dir: str | Path, threshold: float = 0.5) -> list[Path]:
    """Write a confusion matrix and ROC curve PNG; return their paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    y_pred = (np.asarray(y_proba) >= threshold).astype(int)

    cm_path = out_dir / "confusion_matrix.png"
    fig, ax = plt.subplots(figsize=(4, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, display_labels=["No disease", "Disease"], cmap="Blues", ax=ax, colorbar=False
    )
    ax.set_title("Confusion matrix (test set)")
    fig.tight_layout()
    fig.savefig(cm_path, dpi=120)
    plt.close(fig)

    roc_path = out_dir / "roc_curve.png"
    fig, ax = plt.subplots(figsize=(4, 4))
    RocCurveDisplay.from_predictions(y_true, y_proba, ax=ax)
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", linewidth=1)
    ax.set_title("ROC curve (test set)")
    fig.tight_layout()
    fig.savefig(roc_path, dpi=120)
    plt.close(fig)

    return [cm_path, roc_path]
