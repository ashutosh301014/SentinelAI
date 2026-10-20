"""Persist every production inference artifact under models/.

Loads the already-trained artifacts (supervised classifier pipeline,
Isolation Forest detector, evaluation metadata) and derives everything the
production inference pipeline needs at load time:

* ``preprocessing_metadata.json``  fitted select/derive state
* ``feature_order.json``           expected model feature order
* ``event_templates.json``         E1..E29 -> HDFS event templates
* ``training_profile.joblib``      training-normal evidence profile
* ``metadata.json``                consolidated production metadata

This script NEVER fits anything: the classifier and detector are only
loaded, and the held-out test split is used exclusively for evaluation.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import joblib
from sklearn.metrics import average_precision_score, roc_auc_score

from sentinelai.anomaly_detector import DEFAULT_THRESHOLD, load_detector
from sentinelai.config import get_settings
from sentinelai.data_loader import (
    EVENT_COLUMNS,
    load_anomaly_labels,
    load_occurrence_matrix,
    load_templates,
)
from sentinelai.explainability import build_training_profile, pipeline_feature_names
from sentinelai.features import DERIVED_FEATURE_NAMES
from sentinelai.modeling import load_model
from sentinelai.preprocessing import (
    feature_vector_overlap,
    grouped_split_dataset,
    group_statistics,
)

MODEL_VERSION = "2.0.0"
DATASET_NAME = "HDFS_v1"
N_SPLITS = 5

METRIC_KEYS = (
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


def _write_json(path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
        fh.write("\n")


def main() -> None:
    settings = get_settings()
    models_dir = settings.models_dir
    classifier = load_model(settings.model_path("best_model_pipeline"))
    detector = load_detector(settings.model_path("anomaly_detector"))
    with open(models_dir / "model_metadata.json", encoding="utf-8") as fh:
        trained_metadata = json.load(fh)

    select = classifier.named_steps["select"]
    derive = classifier.named_steps["derive"]
    feature_names = pipeline_feature_names(classifier)
    if feature_names != trained_metadata["feature_names"]:
        raise ValueError(
            "fitted pipeline feature names differ from model_metadata.json"
        )

    # ---- preprocessing metadata + expected feature order ----
    preprocessing_metadata = {
        "description": (
            "Fitted state of the feature-engineering steps of the "
            "production classifier pipeline. Inference must reproduce "
            "exactly this schema."
        ),
        "input_event_columns": list(select.input_event_columns_),
        "kept_event_columns": list(select.kept_event_columns_),
        "constant_columns": list(select.constant_columns_),
        "duplicate_columns": list(select.duplicate_columns_),
        "derived_feature_names": list(DERIVED_FEATURE_NAMES),
        "rare_event_columns": list(derive.rare_event_columns_),
        "feature_count": len(feature_names),
    }
    _write_json(models_dir / "preprocessing_metadata.json", preprocessing_metadata)
    _write_json(
        models_dir / "feature_order.json",
        {
            "description": (
                "Expected model feature order at inference time. The "
                "production pipeline checks every engineered feature "
                "vector against this order."
            ),
            "features": feature_names,
        },
    )

    # ---- event-template mapping ----
    templates = load_templates()
    template_map = {
        str(row.EventId): str(row.EventTemplate)
        for row in templates.itertuples(index=False)
    }
    _write_json(
        models_dir / "event_templates.json",
        {
            "description": (
                "HDFS_v1 event-id -> event-template mapping used by "
                "explainability and incident intelligence."
            ),
            "templates": template_map,
        },
    )

    # ---- training-normal evidence profile ----
    labels = load_anomaly_labels()
    matrix = load_occurrence_matrix()
    X_train, X_test, y_train, y_test = grouped_split_dataset(
        matrix, labels, n_splits=N_SPLITS, random_state=settings.random_state
    )
    overlap = feature_vector_overlap(X_train, X_test)
    if overlap != 0:
        raise RuntimeError(
            "grouped split is invalid: test rows share exact feature "
            "vectors with the training split"
        )
    profile = build_training_profile(classifier, X_train[y_train == 0])
    joblib.dump(profile, models_dir / "training_profile.joblib")

    # ---- held-out detector evaluation (evaluation only, never fitting) ----
    detector_scores = detector.score(X_test)
    detector_metrics = {
        "roc_auc": float(roc_auc_score(y_test, detector_scores)),
        "pr_auc": float(average_precision_score(y_test, detector_scores)),
    }

    # ---- detector feature-space documentation (from its own fitted state) ----
    detector_pipeline = detector.feature_pipeline
    detector_select = detector_pipeline.named_steps["select"]
    detector_feature_count = int(detector.n_features_out_)
    detector_feature_space_note = (
        "detector fitted its own feature pipeline on training-normal rows "
        "only; events constant on normal rows "
        f"({', '.join(sorted(detector_select.constant_columns_))}) and exact "
        "duplicates "
        f"({', '.join(sorted(detector_select.duplicate_columns_))}) are "
        "dropped, yielding "
        f"{len(detector_select.kept_event_columns_)} event columns + "
        f"{len(DERIVED_FEATURE_NAMES)} derived features"
    )

    # ---- consolidated production metadata ----
    metadata = {
        "dataset": DATASET_NAME,
        "trained_at": trained_metadata["trained_at"],
        "selected_model": trained_metadata["model"],
        "model_version": MODEL_VERSION,
        "classifier_threshold": float(trained_metadata["threshold"]),
        "feature_count": len(feature_names),
        "feature_names": feature_names,
        "evaluation_metrics": {
            key: trained_metadata["metrics"][key] for key in METRIC_KEYS
        },
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
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
            "train_anomalies": int((y_train == 1).sum()),
            "test_anomalies": int((y_test == 1).sum()),
            "dataset_group_statistics": group_statistics(
                matrix.set_index("BlockId")[EVENT_COLUMNS],
                matrix["BlockId"].map(labels.set_index("BlockId")["Label"]),
            ),
        },
        "anomaly_detector": {
            "n_estimators": int(detector.n_estimators),
            "contamination": str(detector.contamination),
            "threshold": float(DEFAULT_THRESHOLD),
            "trained_on_normal_rows": int(detector.normal_train_rows_),
            "feature_count": detector_feature_count,
            "feature_space_note": detector_feature_space_note,
            "evaluation_metrics": detector_metrics,
        },
        "artifacts": {
            "classifier": "best_model_pipeline.joblib",
            "anomaly_detector": "anomaly_detector.joblib",
            "preprocessing_metadata": "preprocessing_metadata.json",
            "feature_order": "feature_order.json",
            "event_templates": "event_templates.json",
            "training_profile": "training_profile.joblib",
        },
        "metadata_built_at": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(models_dir / "metadata.json", metadata)

    print("Production inference artifacts persisted under models/:")
    for name in metadata["artifacts"]:
        print(f"  {metadata['artifacts'][name]}")
    print("  metadata.json")
    print(f"dataset={DATASET_NAME} model={metadata['selected_model']} "
          f"version={MODEL_VERSION} features={metadata['feature_count']}")
    print(f"supervised metrics: f1={metadata['evaluation_metrics']['f1']:.4f} "
          f"pr_auc={metadata['evaluation_metrics']['pr_auc']:.4f}")
    print(f"detector held-out metrics: roc_auc={detector_metrics['roc_auc']:.4f} "
          f"pr_auc={detector_metrics['pr_auc']:.4f}")


if __name__ == "__main__":
    main()
