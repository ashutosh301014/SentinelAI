"""Inspect the Loghub HDFS_v1 dataset under data/raw/hdfs/.

Reads the raw CSVs (so malformed labels are counted, not rejected) and
cross-checks them with the validated loaders from sentinelai.data_loader.
Exit code 0 = dataset healthy.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sentinelai.config import get_settings  # noqa: E402
from sentinelai.data_loader import (  # noqa: E402
    EVENT_COLUMNS,
    FILENAME_LABELS,
    FILENAME_MATRIX,
    FILENAME_TEMPLATES,
    FILENAME_TRACES,
    GROUND_TRUTH_LABELS,
    OUTCOME_LABELS,
    convert_label,
    convert_outcome_label,
    load_anomaly_labels,
    load_event_traces,
    load_occurrence_matrix,
    load_templates,
    parse_event_sequence,
)

SECTION = "-" * 72


def fmt_size(path: Path) -> str:
    return f"{path.stat().st_size / 1e6:,.1f} MB"


def inspect_labels(directory: Path) -> list[str]:
    problems: list[str] = []
    raw = pd.read_csv(directory / FILENAME_LABELS)
    print(f"[{FILENAME_LABELS}] {fmt_size(directory / FILENAME_LABELS)}")
    print(f"  shape: {raw.shape[0]:,} rows x {raw.shape[1]} cols | columns: {list(raw.columns)}")

    counts = raw["Label"].value_counts(dropna=False)
    for value, count in counts.items():
        print(f"  raw label {value!r}: {count:,}")
    malformed = raw["Label"].map(
        lambda v: str(v).strip().lower() not in GROUND_TRUTH_LABELS
    )
    n_malformed = int(malformed.sum())
    print(f"  malformed labels (not Normal/Anomaly): {n_malformed}")
    if n_malformed:
        problems.append(f"{n_malformed} malformed label values")
        print(f"    examples: {sorted(raw['Label'][malformed].astype(str).unique())[:10]}")

    print(f"  missing values: {int(raw.isna().sum().sum())}")
    print(f"  duplicate BlockIds: {int(raw['BlockId'].duplicated().sum())}")

    loaded = load_anomaly_labels(directory / FILENAME_LABELS)
    normal = int((loaded["Label"] == 0).sum())
    anomaly = int((loaded["Label"] == 1).sum())
    print(f"  converted: Normal=0 -> {normal:,} | Anomaly=1 -> {anomaly:,}")
    print(f"  converted dtype: {loaded['Label'].dtype}")
    return problems


def inspect_matrix(directory: Path) -> list[str]:
    problems: list[str] = []
    raw = pd.read_csv(directory / FILENAME_MATRIX)
    print(f"[{FILENAME_MATRIX}] {fmt_size(directory / FILENAME_MATRIX)}")
    print(f"  shape: {raw.shape[0]:,} rows x {raw.shape[1]} cols")

    n_events = sum(1 for c in raw.columns if str(c).startswith("E"))
    print(f"  event features: {n_events} ({EVENT_COLUMNS[0]}..{EVENT_COLUMNS[-1]})")
    print(f"  missing values in event columns: {int(raw[EVENT_COLUMNS].isna().sum().sum())}")
    print(f"  duplicate BlockIds: {int(raw['BlockId'].duplicated().sum())}")

    outcomes = raw["Label"].value_counts(dropna=False)
    malformed_outcome = raw["Label"].map(
        lambda v: str(v).strip().lower() not in OUTCOME_LABELS
    )
    print(f"  outcome Label: {outcomes.to_dict()} | malformed: {int(malformed_outcome.sum())}")
    typed = raw["Type"].notna().sum()
    type_values = [int(x) for x in sorted(raw["Type"].dropna().unique())]
    print(
        f"  Type populated: {int(typed):,} rows (only anomalous) | "
        f"distinct: {type_values}"
    )

    counts = raw[EVENT_COLUMNS]
    print(
        f"  event counts: min={int(counts.min().min())} "
        f"max={int(counts.max().max())} "
        f"mean={counts.mean().mean():.3f}"
    )
    loaded = load_occurrence_matrix(directory / FILENAME_MATRIX)
    print(f"  loader validation: OK (converted Label dtype {loaded['Label'].dtype})")
    return problems


def inspect_traces(directory: Path) -> list[str]:
    problems: list[str] = []
    raw = pd.read_csv(directory / FILENAME_TRACES)
    print(f"[{FILENAME_TRACES}] {fmt_size(directory / FILENAME_TRACES)}")
    print(f"  shape: {raw.shape[0]:,} rows x {raw.shape[1]} cols | columns: {list(raw.columns)}")
    print(
        f"  missing values (Features/Latency): "
        f"{int(raw[['Features', 'Latency']].isna().sum().sum())}"
    )
    print(f"  Type populated: {int(raw['Type'].notna().sum()):,} rows (only anomalous)")
    print(f"  duplicate BlockIds: {int(raw['BlockId'].duplicated().sum())}")

    outcomes = raw["Label"].value_counts(dropna=False)
    malformed_outcome = raw["Label"].map(
        lambda v: str(v).strip().lower() not in OUTCOME_LABELS
    )
    print(f"  outcome Label: {outcomes.to_dict()} | malformed: {int(malformed_outcome.sum())}")

    lengths, bad_sequences = [], 0
    for value in raw["Features"]:
        try:
            lengths.append(len(parse_event_sequence(value)))
        except ValueError:
            bad_sequences += 1
    lengths = pd.Series(lengths)
    print(
        f"  Features event-sequence length: min={int(lengths.min())} "
        f"mean={lengths.mean():.2f} max={int(lengths.max())}"
    )
    print(f"  malformed event sequences: {bad_sequences}")
    print(f"  TimeInterval dtype: {raw['TimeInterval'].dtype} (string lists)")
    print(
        f"  Latency: min={int(raw['Latency'].min())} "
        f"mean={raw['Latency'].mean():.1f} max={int(raw['Latency'].max())}"
    )
    loaded = load_event_traces(directory / FILENAME_TRACES)
    print(f"  loader validation: OK (converted Label dtype {loaded['Label'].dtype})")
    return problems


def inspect_templates(directory: Path) -> list[str]:
    problems: list[str] = []
    path = directory / FILENAME_TEMPLATES
    raw = pd.read_csv(path)
    print(f"[{FILENAME_TEMPLATES}] {fmt_size(path)}")
    print(f"  shape: {raw.shape[0]:,} rows x {raw.shape[1]} cols")
    ids = raw["EventId"].astype(str).tolist()
    print(f"  EventIds: {ids[0]}..{ids[-1]} ({len(ids)} templates)")
    print(f"  sample: {raw.iloc[0]['EventId']} -> {raw.iloc[0]['EventTemplate'][:60]}...")
    load_templates(path)
    print("  loader validation: OK")
    return problems


def inspect_consistency(directory: Path) -> list[str]:
    """Cross-file checks on raw CSVs (labels converted for comparison)."""
    problems: list[str] = []
    print("[cross-file consistency]")
    try:
        labels = pd.read_csv(directory / FILENAME_LABELS)
        matrix = pd.read_csv(directory / FILENAME_MATRIX)
        traces = pd.read_csv(directory / FILENAME_TRACES)
        labels["_v"] = labels["Label"].map(convert_label)
        matrix["_v"] = matrix["Label"].map(convert_outcome_label)
        traces["_v"] = traces["Label"].map(convert_outcome_label)
    except Exception as exc:
        problems.append(f"cross-file validation failed: {exc}")
        print(f"  VALIDATION FAILED: {exc}")
        return problems

    print(
        f"  rows: labels={len(labels):,} matrix={len(matrix):,} "
        f"traces={len(traces):,}"
    )
    same_sets = (
        set(labels["BlockId"]) == set(matrix["BlockId"]) == set(traces["BlockId"])
    )
    print(f"  BlockId sets identical: {same_sets}")

    n_disagree_matrix = 0
    n_disagree_traces = 0
    for other, counter_name in ((matrix, "n_disagree_matrix"), (traces, "n_disagree_traces")):
        merged = labels[["BlockId", "_v"]].merge(
            other[["BlockId", "_v"]], on="BlockId", suffixes=("_gt", "_out")
        )
        n = int((merged["_v_gt"] != merged["_v_out"]).sum())
        if counter_name == "n_disagree_matrix":
            n_disagree_matrix = n
        else:
            n_disagree_traces = n

    print(f"  label disagreements vs matrix: {n_disagree_matrix}")
    print(f"  label disagreements vs traces: {n_disagree_traces}")
    if not same_sets:
        problems.append("BlockId sets differ")
    if n_disagree_matrix or n_disagree_traces:
        problems.append("labels disagree across files")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dir",
        type=Path,
        default=get_settings().data_raw / "hdfs",
        help="directory containing the HDFS_v1 CSV files",
    )
    args = parser.parse_args()
    directory = args.dir

    missing = [
        name
        for name in (FILENAME_LABELS, FILENAME_MATRIX, FILENAME_TRACES, FILENAME_TEMPLATES)
        if not (directory / name).is_file()
    ]
    if missing:
        print(f"FATAL: missing files in {directory}: {missing}")
        print(
            "Download the official Loghub HDFS_v1 archive "
            "(https://zenodo.org/records/8196385) and place the preprocessed "
            "CSVs in that directory."
        )
        return 1

    print(f"HDFS_v1 dataset inspection — {directory}")
    print(SECTION)
    problems: list[str] = []
    problems += inspect_labels(directory)
    print(SECTION)
    problems += inspect_matrix(directory)
    print(SECTION)
    problems += inspect_traces(directory)
    print(SECTION)
    problems += inspect_templates(directory)
    print(SECTION)
    problems += inspect_consistency(directory)
    print(SECTION)

    if problems:
        print(f"DATASET CHECK FAILED ({len(problems)} problem(s))")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print("DATASET CHECK PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
