"""Train and evaluate SentinelAI's Isolation-Forest anomaly detector.

Protocol (leakage-resistant):

1. grouped train/test split on the real HDFS_v1 data (whole feature-vector
   groups confined to one partition; exact-vector overlap hard-checked = 0)
2. the detector is fitted on NORMAL rows of the training split only
3. test labels are used exclusively for evaluation metrics

Artifacts:

* models/anomaly_detector.joblib                  persisted detector
* reports/figures/isolation_score_distribution.png  normal vs anomalous
  normalized-score distributions on the held-out test set
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from sentinelai.anomaly_detector import AnomalyDetector, save_detector
from sentinelai.config import get_settings
from sentinelai.data_loader import load_anomaly_labels, load_occurrence_matrix
from sentinelai.preprocessing import (
    feature_vector_overlap,
    grouped_split_dataset,
)


def _distribution_stats(name: str, scores: np.ndarray) -> dict[str, float]:
    return {
        "group": name,
        "count": int(len(scores)),
        "mean": float(scores.mean()),
        "median": float(np.median(scores)),
        "p5": float(np.percentile(scores, 5)),
        "p95": float(np.percentile(scores, 95)),
    }


def _save_distribution_figure(
    y_test: np.ndarray,
    scores: np.ndarray,
    path: str,
) -> None:
    normal = scores[y_test == 0]
    anomalous = scores[y_test == 1]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    bins = np.linspace(0, 1, 51)
    ax.hist(
        normal,
        bins=bins,
        density=True,
        alpha=0.6,
        color="#2e7d32",
        label=f"Normal (n={len(normal):,})",
    )
    ax.hist(
        anomalous,
        bins=bins,
        density=True,
        alpha=0.6,
        color="#c62828",
        label=f"Anomalous (n={len(anomalous):,})",
    )
    for scores_subset, color in ((normal, "#1b5e20"), (anomalous, "#8e0000")):
        ax.axvline(
            np.median(scores_subset),
            color=color,
            linestyle="--",
            linewidth=1.5,
        )
    ax.set_xlabel("Normalized anomaly score (0 = normal, 1 = anomalous)")
    ax.set_ylabel("Density")
    ax.set_title("Isolation Forest score distributions — held-out test set")
    ax.legend(loc="upper center")
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
        n_splits=5,
        random_state=settings.random_state,
    )
    overlap = feature_vector_overlap(X_train, X_test)
    print(
        f"split: train={len(X_train):,} (normal {int((y_train == 0).sum()):,}) "
        f"test={len(X_test):,} (anomalies {int(y_test.sum()):,}), "
        f"grouped, exact-vector overlap={overlap}"
    )
    if overlap != 0:
        raise RuntimeError(
            "grouped split is invalid: test rows share exact feature "
            "vectors with the training split"
        )

    detector = AnomalyDetector(random_state=settings.random_state)
    detector.fit(X_train, y_train)
    print(
        f"fitted on {detector.normal_train_rows_:,} normal training rows "
        f"({detector.n_features_out_} features, raw_mean={detector.raw_mean_:.4f}, "
        f"raw_scale={detector.raw_scale_:.4f})"
    )

    train_scores = detector.score(X_train)
    test_scores = detector.score(X_test)
    print(
        f"training-normal score sanity: mean={train_scores[y_train == 0].mean():.4f} "
        f"max={train_scores[y_train == 0].max():.4f}"
    )

    roc_auc = roc_auc_score(y_test, test_scores)
    pr_auc = average_precision_score(y_test, test_scores)
    print(f"\nheld-out evaluation:")
    print(f"  ROC-AUC = {roc_auc:.4f}")
    print(f"  PR-AUC  = {pr_auc:.4f}")

    print("\nnormalized score distributions (test):")
    for stats in (
        _distribution_stats("normal", test_scores[y_test == 0]),
        _distribution_stats("anomalous", test_scores[y_test == 1]),
    ):
        print(
            f"  {stats['group']:>9}: n={stats['count']:>7,}  mean={stats['mean']:.4f}  "
            f"median={stats['median']:.4f}  p5={stats['p5']:.4f}  p95={stats['p95']:.4f}"
        )

    model_path = settings.model_path("anomaly_detector")
    save_detector(detector, model_path)
    print(f"\nmodel saved: {model_path}")

    figure_path = settings.figures_dir / "isolation_score_distribution.png"
    _save_distribution_figure(
        y_test.to_numpy(), test_scores, str(figure_path)
    )
    print(f"figure saved: {figure_path}")


if __name__ == "__main__":
    main()
