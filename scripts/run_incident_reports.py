"""Generate IncidentReports for one held-out normal and one held-out
anomalous trace, using the trained models.

Reports are printed as JSON and saved under reports/. All statements are
evidence-based; no causal claim is made and no system state is invented
(HDFS_v1 provides no block timestamps, so ``timestamp`` is null).
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from sentinelai.anomaly_detector import load_detector
from sentinelai.config import get_settings
from sentinelai.data_loader import (
    load_anomaly_labels,
    load_occurrence_matrix,
    load_templates,
)
from sentinelai.explainability import (
    build_training_profile,
    pipeline_feature_names,
)
from sentinelai.incident_engine import build_incident_report
from sentinelai.modeling import load_model
from sentinelai.preprocessing import grouped_split_dataset


def main() -> None:
    settings = get_settings()
    settings.ensure_dirs()

    matrix = load_occurrence_matrix()
    labels = load_anomaly_labels()
    templates = load_templates(required=True)
    X_train, X_test, y_train, y_test = grouped_split_dataset(
        matrix, labels, n_splits=5, random_state=settings.random_state
    )
    pipeline = load_model(settings.model_path("best_model_pipeline"))
    detector = load_detector(settings.model_path("anomaly_detector"))
    profile = build_training_profile(
        pipeline, X_train[y_train.to_numpy() == 0]
    )

    # most typical normal trace: fewest profile deviations, tie-broken by
    # lowest anomaly probability; most confident anomalous trace by proba
    probabilities = pipeline.predict_proba(X_test)[:, 1]
    normal_mask = y_test.to_numpy() == 0
    anomalous_mask = y_test.to_numpy() == 1
    transformed_test = pd.DataFrame(
        np.asarray(pipeline[:-1].transform(X_test)),
        columns=pipeline_feature_names(pipeline),
    )
    p5 = profile.percentiles["p5"]
    p95 = profile.percentiles["p95"]
    deviation_counts = (
        (transformed_test > p95).sum(axis=1)
        + ((transformed_test < p5) & (p5 > 0)).sum(axis=1)
    ).to_numpy()
    normal_candidates = np.where(normal_mask)[0]
    order = np.lexsort(
        (probabilities[normal_candidates], deviation_counts[normal_candidates])
    )
    normal_pos = int(normal_candidates[order[0]])
    anomalous_pos = int(np.argmax(probabilities[anomalous_mask]))

    normal_sample = X_test.iloc[[normal_pos]]
    anomalous_sample = X_test.iloc[anomalous_mask].iloc[[anomalous_pos]]

    normal_report = build_incident_report(
        str(normal_sample.index[0]),
        normal_sample,
        pipeline,
        detector,
        profile,
        templates=templates,
    )
    anomalous_report = build_incident_report(
        str(anomalous_sample.index[0]),
        anomalous_sample,
        pipeline,
        detector,
        profile,
        templates=templates,
    )

    for name, report in (
        ("normal", normal_report),
        ("anomalous", anomalous_report),
    ):
        print(f"===== {name.upper()} TRACE =====")
        print(json.dumps(report.to_dict(), indent=2))
        print()
        path = settings.reports_dir / f"incident_report_{name}.json"
        path.write_text(
            json.dumps(report.to_dict(), indent=2), encoding="utf-8"
        )
        print(f"saved: {path}\n")

    print(
        "Note: HDFS_v1 provides no experimentally verified causal "
        "root-cause labels; statements above are probable causes and "
        "evidence-based recommendations, not confirmed root causes."
    )


if __name__ == "__main__":
    main()
