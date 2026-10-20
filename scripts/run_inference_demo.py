"""Run the persisted demo samples through the full production pipeline.

Loads the two one-row demo event matrices and their metadata from
``data/demo/``, executes the complete 10-stage inference pipeline on each
sample, verifies the expected behaviour (normal sample stays low risk;
anomalous sample shows detection, incident risk, probable cause evidence
and a blast radius), and prints the actual inference outputs. Inference
loads artifacts only — it never retrains.
"""
from __future__ import annotations

import json
import sys

import pandas as pd

from sentinelai.config import get_settings
from sentinelai.inference import InferencePipeline

CHECKS: list[tuple[str, bool]] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    passed = bool(condition)
    CHECKS.append((label, passed))
    suffix = f" - {detail}" if detail else ""
    print(f"[{'PASS' if passed else 'FAIL'}] {label}{suffix}")


def main() -> int:
    settings = get_settings()
    metadata_path = settings.data_demo / "demo_samples_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    pipeline = InferencePipeline()
    print(
        "Loaded production pipeline: "
        f"dataset={pipeline.metadata['dataset']} "
        f"model={pipeline.metadata['selected_model']} "
        f"version={pipeline.metadata['model_version']} "
        "(load-only, no retraining)"
    )
    print(f"Demo metadata: {metadata_path}")

    for record in metadata["samples"]:
        frame = pd.read_csv(settings.data_demo / record["file"], index_col=0)
        block_id = str(frame.index[0])
        check(
            f"{record['role']}: CSV block matches metadata",
            block_id == str(record["block_id"]),
            f"{block_id}",
        )
        result = pipeline.run(frame, block_id=block_id)
        payload = result.to_dict()
        print(
            f"\n===== INFERENCE OUTPUT - {record['role'].upper()} "
            f"(intended label: {record['intended_label_name']}) ====="
        )
        print(json.dumps(payload, indent=2, default=str))

        print(f"\n===== VERIFICATION - {record['role'].upper()} =====")
        if record["role"] == "representative_normal":
            prediction = payload["supervised_classifier"]
            risk = payload["composite_risk_score"]
            check(
                "classifier predicts normal",
                prediction["prediction"] == 0,
                f"anomaly_probability={prediction['anomaly_probability']:.4g}",
            )
            check(
                "composite risk is LOW",
                risk["severity"] == "LOW",
                f"risk_score={risk['risk_score']}",
            )
            check(
                "blast radius skipped (no incident origin)",
                payload["blast_radius_prediction"]["status"] == "skipped",
            )
        else:
            incident = payload["incident_intelligence"]
            blast = payload["blast_radius_prediction"]
            check(
                "anomaly detected",
                incident["anomaly_detected"] is True,
                f"anomaly_probability={incident['classifier_probability']:.4g}",
            )
            check(
                "elevated incident risk",
                payload["composite_risk_score"]["severity"] in {"HIGH", "CRITICAL"},
                f"risk_score={payload['composite_risk_score']['risk_score']} "
                f"({payload['composite_risk_score']['severity']})",
            )
            cause = str(incident["probable_cause"])
            check(
                "probable cause evidence present",
                not cause.startswith("Probable Cause: unknown")
                and len(incident["top_evidence"]) > 0,
                cause,
            )
            check(
                "blast radius computed with services at risk",
                blast["status"] == "computed"
                and len(blast["services_at_risk"]) > 0,
                f"origin={blast['probable_origin']['name']} "
                f"({blast['probable_origin']['incident_risk']:.2f}), "
                f"{len(blast['services_at_risk'])} service(s) at risk, "
                f"highest={blast['highest_risk_service']}",
            )

    failures = [label for label, passed in CHECKS if not passed]
    print(
        f"\nVerification: {len(CHECKS) - len(failures)}/{len(CHECKS)} "
        "checks passed."
    )
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
