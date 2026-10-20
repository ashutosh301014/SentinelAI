"""Train and compare SentinelAI supervised anomaly-classification models.

Flow (leakage-resistant by construction):

1. grouped train/test split on the real HDFS_v1 data: every row is
   assigned a deterministic SHA-256 group ID of its complete feature
   vector and StratifiedGroupKFold assigns WHOLE groups to folds, so no
   exact feature vector ever appears in both splits (hard-checked,
   overlap = 0)
2. each model pipeline is fitted on the training split only
3. all metrics are computed on the STRICT unseen-pattern test split only

Artifacts:

* reports/model_comparison.csv       one row per model, anomaly-class metrics
* reports/figures/confusion_matrices.png
* reports/figures/pr_curves.png
* reports/figures/roc_curves.png
* models/best_model_pipeline.joblib  persisted winner (features + classifier)
* models/model_metadata.json         winner metrics, feature names, provenance
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from sentinelai.config import get_settings
from sentinelai.data_loader import load_anomaly_labels, load_occurrence_matrix
from sentinelai.modeling import (
    METRIC_COLUMNS,
    MODEL_LABELS,
    available_model_names,
    build_classifier_pipeline,
    evaluate_model,
    fit_params,
    save_model,
    select_best_model,
)
from sentinelai.preprocessing import (
    feature_vector_overlap,
    grouped_split_dataset,
)

TEST_SIZE = 0.2
N_SPLITS = 5
MAX_FPR = 0.05
THRESHOLD = 0.5


def _save_confusion_matrix_grid(
    y_true: np.ndarray,
    predictions: dict[str, np.ndarray],
    path: str,
) -> None:
    n = len(predictions)
    fig, axes = plt.subplots(1, n, figsize=(4.6 * n, 4.2), squeeze=False)
    for ax, (name, y_pred) in zip(axes[0], predictions.items()):
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        cm = np.array([[tn, fp], [fn, tp]])
        im = ax.imshow(cm, cmap="Blues")
        ax.set_title(
            f"{MODEL_LABELS[name]}\nF1={f1_score(y_true, y_pred):.3f}  "
            f"Recall={recall_score(y_true, y_pred):.3f}",
            fontsize=10,
        )
        ax.set_xticks([0, 1], ["Normal", "Anomaly"])
        ax.set_yticks([0, 1], ["Normal", "Anomaly"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        for i in range(2):
            for j in range(2):
                ax.text(
                    j,
                    i,
                    f"{cm[i, j]:,}",
                    ha="center",
                    va="center",
                    fontsize=11,
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                )
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _save_pr_curves(
    y_true: np.ndarray,
    probas: dict[str, np.ndarray],
    path: str,
) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 6))
    baseline = float((y_true == 1).mean())
    for name, y_proba in probas.items():
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        pr_auc = average_precision_score(y_true, y_proba)
        ax.plot(
            recall,
            precision,
            label=f"{MODEL_LABELS[name]} (PR-AUC={pr_auc:.3f})",
        )
    ax.axhline(baseline, color="gray", linestyle="--", linewidth=1)
    ax.text(0.98, baseline + 0.01, f"chance {baseline:.3f}", ha="right", color="gray")
    ax.set_xlabel("Recall (anomaly)")
    ax.set_ylabel("Precision (anomaly)")
    ax.set_title("Precision-Recall curves — held-out test set")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _save_roc_curves(
    y_true: np.ndarray,
    probas: dict[str, np.ndarray],
    path: str,
) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 6))
    for name, y_proba in probas.items():
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        auc = roc_auc_score(y_true, y_proba)
        ax.plot(fpr, tpr, label=f"{MODEL_LABELS[name]} (ROC-AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate (recall)")
    ax.set_title("ROC curves — held-out test set")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    settings = get_settings()
    settings.ensure_dirs()

    matrix = load_occurrence_matrix()
    labels = load_anomaly_labels()
    X_train, X_test, y_train, y_test = grouped_split_dataset(
        matrix,
        labels,
        n_splits=N_SPLITS,
        random_state=settings.random_state,
    )
    overlap = feature_vector_overlap(X_train, X_test)
    print(
        f"split: train={len(X_train):,} test={len(X_test):,} "
        f"(anomalies: train={int(y_train.sum()):,} test={int(y_test.sum()):,}, "
        f"grouped StratifiedGroupKFold(n_splits={N_SPLITS}), "
        f"random_state={settings.random_state}, exact-vector overlap={overlap})"
    )
    if overlap != 0:
        raise RuntimeError(
            f"grouped split is invalid: {overlap} test rows share an exact "
            "feature vector with the training split"
        )

    names = available_model_names()
    predictions: dict[str, np.ndarray] = {}
    probas: dict[str, np.ndarray] = {}
    rows: list[dict] = []
    fitted: dict[str, object] = {}

    for name in names:
        print(f"\ntraining {MODEL_LABELS[name]} ...", flush=True)
        started = time.time()
        pipeline = build_classifier_pipeline(name, random_state=settings.random_state)
        pipeline.fit(X_train, y_train, **fit_params(name, y_train))
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]
        metrics = evaluate_model(y_test, y_pred, y_proba)
        metrics["model"] = name
        metrics["train_seconds"] = round(time.time() - started, 2)
        rows.append(metrics)
        predictions[name] = y_pred
        probas[name] = y_proba
        fitted[name] = pipeline
        print(
            f"  P={metrics['precision']:.4f} R={metrics['recall']:.4f} "
            f"F1={metrics['f1']:.4f} ROC-AUC={metrics['roc_auc']:.4f} "
            f"PR-AUC={metrics['pr_auc']:.4f} FPR={metrics['fpr']:.4f} "
            f"FNR={metrics['fnr']:.4f} acc={metrics['accuracy']:.4f} "
            f"({metrics['train_seconds']}s)"
        )

    comparison = pd.DataFrame(rows, columns=METRIC_COLUMNS)
    comparison_path = settings.reports_dir / "model_comparison.csv"
    comparison.to_csv(comparison_path, index=False)
    print(f"\ncomparison table: {comparison_path}")

    _save_confusion_matrix_grid(
        y_test.to_numpy(), predictions, str(settings.figures_dir / "confusion_matrices.png")
    )
    _save_pr_curves(y_test.to_numpy(), probas, str(settings.figures_dir / "pr_curves.png"))
    _save_roc_curves(y_test.to_numpy(), probas, str(settings.figures_dir / "roc_curves.png"))
    print("figures saved: confusion_matrices.png, pr_curves.png, roc_curves.png")

    best = select_best_model(comparison, max_fpr=MAX_FPR)
    best_name = best["model"]
    best_pipeline = fitted[best_name]
    model_path = settings.model_path("best_model_pipeline")
    save_model(best_pipeline, model_path)

    metadata = {
        "model": best_name,
        "model_label": MODEL_LABELS[best_name],
        "selection_rule": f"FPR <= {MAX_FPR}, then best PR-AUC (tie-break F1, Recall)",
        "threshold": THRESHOLD,
        "metrics": {k: best[k] for k in METRIC_COLUMNS if k != "model"},
        "feature_names": list(best_pipeline[:-1].get_feature_names_out()),
        "random_state": settings.random_state,
        "test_size": TEST_SIZE,
        "split": {
            "strategy": (
                "grouped: StratifiedGroupKFold(n_splits=5, shuffle=True, "
                "random_state=42) over deterministic SHA-256 group IDs of "
                "the complete feature vector; test partition = fold 0"
            ),
            "guarantee": (
                "no exact feature vector appears in both train and test "
                "(overlap hard-checked to 0)"
            ),
            "vector_overlap_train_test": int(overlap),
        },
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "test_anomalies": int(y_test.sum()),
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    metadata_path = settings.models_dir / "model_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"best model: {MODEL_LABELS[best_name]} -> {model_path}")
    print(f"metadata: {metadata_path}")

    print("\n" + comparison.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
