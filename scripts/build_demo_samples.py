"""Select and persist two representative HELD-OUT demo samples.

Picks one representative NORMAL trace and one representative ANOMALOUS
trace from the STRICT grouped test partition (StratifiedGroupKFold over
deterministic feature-vector group IDs — no exact feature vector ever
appears in both train and test) and stores them as two one-row event
matrices under ``data/demo/`` together with a metadata file recording
the ground-truth (intended) label of each sample.

Selection NEVER retrains: the trained artifacts are loaded read-only and
used only to score candidate held-out rows so the samples demonstrate the
system well. Only the two one-row matrices and the metadata file are
written to ``data/demo/`` — the full dataset is never copied there.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np

from sentinelai.config import get_settings
from sentinelai.data_loader import load_anomaly_labels, load_occurrence_matrix
from sentinelai.explainability import NORMAL_EVIDENCE, pipeline_feature_names
from sentinelai.inference import InferencePipeline
from sentinelai.preprocessing import (
    feature_vector_overlap,
    grouped_split_dataset,
)
from sentinelai.risk_engine import compute_risk_score, severity_for_score

NORMAL_CANDIDATES = 500
ANOMALY_CANDIDATES = 200

NORMAL_FILE = "demo_normal_block.csv"
ANOMALOUS_FILE = "demo_anomalous_block.csv"
METADATA_FILE = "demo_samples_metadata.json"


def _risk_scores(
    pipeline: InferencePipeline, rows
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Composite risk per candidate row via the loaded artifacts only."""
    names = pipeline_feature_names(pipeline.classifier)
    percentiles = pipeline.profile.percentiles.loc[names]
    values = np.asarray(pipeline.classifier[:-1].transform(rows))
    p5 = percentiles["p5"].to_numpy(dtype="float64")
    p95 = percentiles["p95"].to_numpy(dtype="float64")
    irregularity = ((values < p5) | (values > p95)).sum(axis=1) / len(names)
    total_index = names.index("total_event_count")
    intensity = np.minimum(
        1.0, values[:, total_index] / max(float(p95[total_index]), 1.0)
    )
    probabilities = pipeline.classifier.predict_proba(rows)[:, 1]
    anomaly_scores = pipeline.detector.score(rows)
    risks = np.array(
        [
            compute_risk_score(
                float(probabilities[i]),
                float(anomaly_scores[i]),
                float(irregularity[i]),
                float(intensity[i]),
                settings=pipeline.settings,
            ).risk_score
            for i in range(len(rows))
        ]
    )
    return probabilities, anomaly_scores, risks


def main() -> None:
    settings = get_settings()
    settings.ensure_dirs()
    pipeline = InferencePipeline()

    matrix = load_occurrence_matrix()
    labels = load_anomaly_labels()
    X_train, X_test, y_train, y_test = grouped_split_dataset(
        matrix, labels, n_splits=5, random_state=42
    )
    overlap = feature_vector_overlap(X_train, X_test)
    if overlap != 0:
        raise RuntimeError(
            "grouped split is invalid: test rows share exact feature "
            "vectors with the training split"
        )

    # ---- representative NORMAL: lowest composite risk among held-out
    # rows the classifier predicts normal ----
    normal_pool = X_test[y_test == 0].iloc[:NORMAL_CANDIDATES]
    normal_candidates = normal_pool[pipeline.classifier.predict(normal_pool) == 0]
    normal_probs, normal_scores, normal_risks = _risk_scores(
        pipeline, normal_candidates
    )
    normal_position = int(normal_risks.argmin())
    normal_row = normal_candidates.iloc[[normal_position]]
    normal_risk = int(normal_risks[normal_position])
    if severity_for_score(normal_risk, settings.risk_band_thresholds).value != "LOW":
        raise RuntimeError(
            f"selected normal sample scores {normal_risk} — not LOW; "
            "the held-out pool does not demonstrate a normal trace well"
        )

    # ---- representative ANOMALOUS: highest composite risk among
    # held-out rows the classifier predicts anomalous, additionally
    # required to show evidence phrases and a non-empty blast radius ----
    anomaly_pool = X_test[y_test == 1].iloc[:ANOMALY_CANDIDATES]
    anomaly_candidates = anomaly_pool[pipeline.classifier.predict(anomaly_pool) == 1]
    anomaly_probs, anomaly_scores, anomaly_risks = _risk_scores(
        pipeline, anomaly_candidates
    )
    order = np.argsort(-anomaly_risks)
    anomaly_row = None
    anomaly_detail: dict = {}
    for position in order:
        row = anomaly_candidates.iloc[[int(position)]]
        result = pipeline.run(row)
        blast = result.blast_radius_prediction
        report = result.incident_intelligence
        suitable = (
            blast["status"] == "computed"
            and len(blast["services_at_risk"]) > 0
            and result.composite_risk_score["severity"] in {"HIGH", "CRITICAL"}
            and list(report["top_evidence"]) != list(NORMAL_EVIDENCE)
            and not str(report["probable_cause"]).startswith(
                "Probable Cause: unknown"
            )
        )
        if suitable:
            anomaly_row = row
            anomaly_detail = {
                "classifier_probability": float(anomaly_probs[position]),
                "anomaly_score": float(anomaly_scores[position]),
                "risk_score": int(anomaly_risks[position]),
                "severity": result.composite_risk_score["severity"],
                "mapped_service": result.dependency_graph_mapping["service_id"],
                "blast_radius_services_at_risk": len(
                    blast["services_at_risk"]
                ),
                "top_evidence_count": len(report["top_evidence"]),
            }
            break
    if anomaly_row is None:
        raise RuntimeError(
            "no suitable held-out anomalous sample found in the first "
            f"{ANOMALY_CANDIDATES} candidate rows"
        )

    # ---- held-out guarantee: neither sample may be in the training split ----
    train_ids = set(X_train.index)
    for row, role in ((normal_row, "normal"), (anomaly_row, "anomalous")):
        block_id = str(row.index[0])
        if block_id in train_ids:
            raise RuntimeError(
                f"{role} sample {block_id} is in the training split"
            )

    normal_id = str(normal_row.index[0])
    anomaly_id = str(anomaly_row.index[0])

    # ---- persist the two one-row matrices + metadata ----
    demo_dir = settings.data_demo
    normal_row.to_csv(demo_dir / NORMAL_FILE)
    anomaly_row.to_csv(demo_dir / ANOMALOUS_FILE)
    metadata = {
        "description": (
            "Production-safe demo inputs for SentinelAI: two one-row event "
            "matrices selected from the HELD-OUT test split (never used for "
            "training). Intended labels are ground-truth labels "
            "(anomaly_label.csv) recorded for evaluation only — the "
            "inference pipeline never sees them and all predictions are "
            "computed live from the trained models."
        ),
        "dataset": pipeline.metadata["dataset"],
        "dataset_source": "Loghub HDFS_v1 (Zenodo record 8196385)",
        "split": {
            "name": "test",
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
        },
        "selection": {
            "method": (
                "candidates scored with the already-trained artifacts "
                "(loaded read-only, NO retraining); normal = lowest "
                "composite risk among held-out rows the classifier "
                "predicts normal; anomalous = highest composite risk "
                "among held-out rows the classifier predicts anomalous, "
                "additionally required to show HIGH/CRITICAL risk, "
                "evidence phrases and a non-empty blast radius"
            ),
            "model_version": pipeline.metadata["model_version"],
            "normal_candidates_scanned": NORMAL_CANDIDATES,
            "anomaly_candidates_scanned": ANOMALY_CANDIDATES,
        },
        "samples": [
            {
                "role": "representative_normal",
                "file": NORMAL_FILE,
                "block_id": normal_id,
                "intended_label": 0,
                "intended_label_name": "Normal",
                "ground_truth_source": "anomaly_label.csv (Label column)",
                "held_out": True,
                "selection_stats": {
                    "classifier_probability": float(
                        normal_probs[normal_position]
                    ),
                    "anomaly_score": float(normal_scores[normal_position]),
                    "risk_score": normal_risk,
                    "severity": "LOW",
                },
            },
            {
                "role": "representative_anomalous",
                "file": ANOMALOUS_FILE,
                "block_id": anomaly_id,
                "intended_label": 1,
                "intended_label_name": "Anomaly",
                "ground_truth_source": "anomaly_label.csv (Label column)",
                "held_out": True,
                "selection_stats": anomaly_detail,
            },
        ],
        "selected_at": datetime.now(timezone.utc).isoformat(),
    }
    metadata_path = demo_dir / METADATA_FILE
    with open(metadata_path, "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)
        fh.write("\n")

    print("Demo samples persisted under data/demo/ (no full dataset copied):")
    for sample in metadata["samples"]:
        stats = sample["selection_stats"]
        print(
            f"  {sample['file']}: block {sample['block_id']} "
            f"intended={sample['intended_label_name']} "
            f"proba={stats['classifier_probability']:.4g} "
            f"if={stats['anomaly_score']:.3f} "
            f"risk={stats['risk_score']} ({stats['severity']})"
        )
    print(f"  {METADATA_FILE}")
    print("No model was retrained during selection.")


if __name__ == "__main__":
    main()
