"""Strict grouped-split audit (Prompt 24).

1. deterministic SHA-256 group ID per complete feature vector
2. group statistics: unique vectors, duplicated vectors, duplicate rows,
   and feature vectors with CONFLICTING Normal/Anomaly labels (reported,
   never silently resolved)
3. StratifiedGroupKFold split on whole vector groups — hard-verified that
   the exact feature-vector overlap between train and test is ZERO
4. class distribution of both partitions

Artifact: reports/grouped_split_audit.json
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

from sentinelai.config import get_settings
from sentinelai.data_loader import (
    EVENT_COLUMNS,
    load_anomaly_labels,
    load_occurrence_matrix,
)
from sentinelai.preprocessing import (
    feature_vector_overlap,
    group_statistics,
    grouped_split_dataset,
    vector_group_ids,
)

N_SPLITS = 5
TEST_FOLD = 0


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
        test_fold=TEST_FOLD,
    )

    # full-dataset group statistics (reported, conflicts NOT resolved)
    label_map = labels.set_index("BlockId")["Label"]
    y_all = matrix["BlockId"].map(label_map)
    X_all = matrix.set_index("BlockId")[EVENT_COLUMNS]
    stats = group_statistics(X_all, y_all)

    train_groups = len(set(vector_group_ids(X_train)))
    test_groups = len(set(vector_group_ids(X_test)))
    overlap = feature_vector_overlap(X_train, X_test)

    train_anomalies = int((y_train == 1).sum())
    test_anomalies = int((y_test == 1).sum())

    audit = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": "HDFS_v1",
        "split": {
            "strategy": (
                "StratifiedGroupKFold(n_splits=5, shuffle=True, "
                "random_state=42) over deterministic SHA-256 group IDs of "
                "the complete feature vector; test partition = fold 0"
            ),
            "guarantee": (
                "all rows sharing one exact feature vector belong to "
                "exactly one partition; train/test exact-vector overlap "
                "is hard-checked to be 0"
            ),
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
            "train_anomalies": train_anomalies,
            "test_anomalies": test_anomalies,
            "train_anomaly_rate": round(train_anomalies / len(X_train), 6),
            "test_anomaly_rate": round(test_anomalies / len(X_test), 6),
            "train_vector_groups": train_groups,
            "test_vector_groups": test_groups,
            "exact_vector_overlap_train_test": int(overlap),
        },
        "group_statistics": stats,
        "conflicting_labels": {
            "policy": (
                "reported, never silently resolved: whole vector groups "
                "are kept intact inside one partition; StratifiedGroupKFold "
                "assigns mixed-label groups first and accounts for their "
                "label distribution"
            ),
        },
    }

    out = settings.reports_dir / "grouped_split_audit.json"
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(audit, fh, indent=2)
        fh.write("\n")

    print("=== Grouped-Split Audit ===")
    print(f"total rows              : {stats['total_rows']:,}")
    print(f"unique feature vectors  : {stats['unique_vectors']:,}")
    print(f"duplicated vectors      : {stats['duplicated_vectors']:,}")
    print(f"duplicate rows          : {stats['duplicate_rows']:,}")
    print(f"conflicting-label vecs  : {stats['conflicting_label_vectors']:,}")
    print(f"  (rows in those groups : {stats['conflicting_label_rows']:,})")
    print(f"train                   : {len(X_train):,} rows "
          f"(anomalies {train_anomalies:,})")
    print(f"test                    : {len(X_test):,} rows "
          f"(anomalies {test_anomalies:,})")
    print(f"exact vector overlap    : {overlap} "
          f"({'PASS' if overlap == 0 else 'FAIL'})")
    print(f"audit saved             : {out}")


if __name__ == "__main__":
    main()
