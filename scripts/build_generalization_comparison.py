"""Build reports/generalization_comparison.csv (Prompt 24).

Merges the two evaluation tables side by side:

* reports/model_comparison_random_split.csv — OLD metrics, plain random
  stratified split (train 460,048 / test 115,013), where 114,968 of
  115,013 test rows shared an exact feature vector with the training set
* reports/model_comparison.csv — NEW metrics on the STRICT grouped
  unseen-pattern split (StratifiedGroupKFold over deterministic
  feature-vector group IDs; exact-vector overlap between train and test
  is hard-checked to 0)

The drop between the two columns is the honest cost of removing the
train/test vector overlap — it is reported, never hidden.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import pandas as pd

from sentinelai.config import get_settings

OLD_SPLIT_TEST_ROWS = 115_013
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


def main() -> None:
    settings = get_settings()
    old = pd.read_csv(settings.reports_dir / "model_comparison_random_split.csv")
    new = pd.read_csv(settings.reports_dir / "model_comparison.csv")

    audit = json.loads(
        (settings.reports_dir / "grouped_split_audit.json").read_text(
            encoding="utf-8"
        )
    )
    new_test_rows = audit["split"]["test_rows"]

    old_rows = []
    for row in old.itertuples(index=False):
        entry = {
            "model": row.model,
            "split_type": "random_stratified",
            "test_rows": OLD_SPLIT_TEST_ROWS,
        }
        entry.update({key: getattr(row, key) for key in METRIC_KEYS})
        old_rows.append(entry)

    new_rows = []
    for row in new.itertuples(index=False):
        entry = {
            "model": row.model,
            "split_type": "grouped_unseen_pattern",
            "test_rows": new_test_rows,
        }
        entry.update({key: getattr(row, key) for key in METRIC_KEYS})
        new_rows.append(entry)

    frame = pd.DataFrame(
        old_rows + new_rows,
        columns=["model", "split_type", "test_rows", *METRIC_KEYS],
    )
    out = settings.reports_dir / "generalization_comparison.csv"
    frame.to_csv(out, index=False)
    print(f"generalization comparison saved: {out}")
    print()
    print(frame.round(4).to_string(index=False))

    provenance = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "old_split": {
            "file": "model_comparison_random_split.csv",
            "strategy": (
                "random stratified train_test_split(test_size=0.2, "
                "random_state=42)"
            ),
            "train_rows": 460_048,
            "test_rows": OLD_SPLIT_TEST_ROWS,
            "known_overlap_issue": (
                "114,968 of 115,013 test rows shared an exact feature "
                "vector with the training set"
            ),
        },
        "new_split": {
            "file": "model_comparison.csv",
            "strategy": (
                "grouped StratifiedGroupKFold(n_splits=5, shuffle=True, "
                "random_state=42) over deterministic SHA-256 feature-vector "
                "group IDs; test partition = fold 0"
            ),
            "train_rows": audit["split"]["train_rows"],
            "test_rows": new_test_rows,
            "exact_vector_overlap_train_test": audit["split"][
                "exact_vector_overlap_train_test"
            ],
        },
    }
    provenance_path = settings.reports_dir / "generalization_comparison.json"
    with open(provenance_path, "w", encoding="utf-8") as fh:
        json.dump(provenance, fh, indent=2)
        fh.write("\n")
    print(f"provenance saved: {provenance_path}")


if __name__ == "__main__":
    main()
