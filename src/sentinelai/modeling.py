"""Supervised anomaly-classification models for SentinelAI.

Each model is a full sklearn Pipeline: feature engineering (event selector +
derived features, scaler only where the algorithm needs it) followed by the
classifier. The pipeline is fitted on the training split only and can be
persisted and reused as-is for inference on raw event vectors.

Class imbalance handling:

* LogisticRegression / RandomForest: ``class_weight="balanced"``
* HistGradientBoosting / XGBoost: balanced ``sample_weight`` at fit time
  (see :func:`fit_params`)

Metric convention: the ANOMALY class (label 1) is the positive class.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight

from sentinelai.preprocessing import build_feature_pipeline, build_scaled_pipeline

DEFAULT_RANDOM_STATE = 42
POSITIVE_LABEL = 1  # anomaly

METRIC_COLUMNS = (
    "model",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "pr_auc",
    "fpr",
    "fnr",
    "accuracy",
    "tp",
    "fp",
    "fn",
    "tn",
    "train_seconds",
)

MODEL_LABELS = {
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "hist_gradient_boosting": "HistGradientBoosting",
    "xgboost": "XGBoost",
}


def available_model_names() -> list[str]:
    """Base models plus xgboost when it is importable."""
    names = [
        "logistic_regression",
        "random_forest",
        "hist_gradient_boosting",
    ]
    try:
        import xgboost  # noqa: F401

        names.append("xgboost")
    except ImportError:
        pass
    return names


def build_classifier_pipeline(
    model_name: str, random_state: int = DEFAULT_RANDOM_STATE
) -> Pipeline:
    """Feature pipeline + classifier. Fit on the training split only."""
    if model_name == "logistic_regression":
        steps = list(build_scaled_pipeline().steps) + [
            (
                "model",
                LogisticRegression(
                    max_iter=3000,
                    class_weight="balanced",
                    random_state=random_state,
                ),
            )
        ]
    elif model_name == "random_forest":
        steps = list(build_feature_pipeline().steps) + [
            (
                "model",
                RandomForestClassifier(
                    n_estimators=300,
                    n_jobs=-1,
                    class_weight="balanced",
                    random_state=random_state,
                ),
            )
        ]
    elif model_name == "hist_gradient_boosting":
        steps = list(build_feature_pipeline().steps) + [
            (
                "model",
                HistGradientBoostingClassifier(
                    max_iter=300,
                    learning_rate=0.1,
                    early_stopping=True,
                    validation_fraction=0.1,
                    n_iter_no_change=20,
                    random_state=random_state,
                ),
            )
        ]
    elif model_name == "xgboost":
        import xgboost as xgb

        steps = list(build_feature_pipeline().steps) + [
            (
                "model",
                xgb.XGBClassifier(
                    n_estimators=300,
                    learning_rate=0.1,
                    max_depth=6,
                    n_jobs=-1,
                    tree_method="hist",
                    random_state=random_state,
                    eval_metric="logloss",
                ),
            )
        ]
    else:
        raise ValueError(f"unknown model {model_name!r}; choose from {MODEL_LABELS}")
    return Pipeline(steps=steps)


def fit_params(model_name: str, y: Any) -> dict[str, Any]:
    """Balanced sample weights for boosting models; empty otherwise."""
    if model_name in {"hist_gradient_boosting", "xgboost"}:
        weights = compute_sample_weight("balanced", y)
        return {"model__sample_weight": weights}
    return {}


def evaluate_model(
    y_true: Any, y_pred: Any, y_proba: Any, pos_label: int = POSITIVE_LABEL
) -> dict[str, float]:
    """Anomaly-class metrics computed on a held-out split only."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_proba = np.asarray(y_proba)
    if len(y_true) != len(y_pred) or len(y_true) != len(y_proba):
        raise ValueError("y_true, y_pred and y_proba must have the same length")
    if not ((y_true == pos_label).any() and (y_true != pos_label).any()):
        raise ValueError(
            "evaluation requires both classes present in y_true; "
            "a held-out split without anomalies cannot be evaluated"
        )
    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    ).ravel()
    precision = precision_score(y_true, y_pred, pos_label=pos_label, zero_division=0)
    recall = recall_score(y_true, y_pred, pos_label=pos_label, zero_division=0)
    f1 = f1_score(y_true, y_pred, pos_label=pos_label, zero_division=0)
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "fpr": float(fp / (fp + tn)),
        "fnr": float(fn / (fn + tp)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }


def select_best_model(
    comparison: pd.DataFrame, max_fpr: float = 0.05
) -> dict[str, Any]:
    """Pick the final classifier.

    Rule: keep only models with acceptable false-alert behaviour
    (FPR <= max_fpr); among them take the best PR-AUC, breaking ties by F1
    then Recall. If no model meets the FPR bound, take the lowest-FPR model.
    """
    if comparison.empty:
        raise ValueError("comparison table is empty")
    required = {"model", "fpr", "pr_auc", "f1", "recall"}
    if not required.issubset(comparison.columns):
        raise ValueError(
            f"comparison table needs columns {sorted(required)}"
        )
    pool = comparison[comparison["fpr"] <= max_fpr]
    if pool.empty:
        pool = comparison.nsmallest(1, "fpr")
    ranked = pool.sort_values(["pr_auc", "f1", "recall"], ascending=False)
    return ranked.iloc[0].to_dict()


def save_model(pipeline: Pipeline, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)


def load_model(path: Path) -> Pipeline:
    return joblib.load(path)
