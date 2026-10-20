"""Loaders for the Loghub HDFS_v1 preprocessed dataset.

Expected files under ``data/raw/hdfs/`` (official Loghub HDFS_v1 archive,
Zenodo record 8196385, https://zenodo.org/records/8196385):

* anomaly_label.csv            BlockId, Label(Normal|Anomaly)  — ground truth
* Event_occurrence_matrix.csv  BlockId, Label(Success|Fail), Type, E1..E29
* Event_traces.csv             BlockId, Label(Success|Fail), Type, Features,
                               TimeInterval, Latency
* HDFS_templates.csv           EventId, EventTemplate

Labels are normalised to ``0 = Normal``, ``1 = Anomaly``. Values outside the
known vocabularies raise ValueError — data is never silently dropped,
imputed, or synthesised.
"""
from __future__ import annotations

import numbers
import re
from pathlib import Path
from typing import Any

import pandas as pd

from sentinelai.config import get_settings

HDFS_SUBDIR = "hdfs"
FILENAME_LABELS = "anomaly_label.csv"
FILENAME_MATRIX = "Event_occurrence_matrix.csv"
FILENAME_TRACES = "Event_traces.csv"
FILENAME_TEMPLATES = "HDFS_templates.csv"

GROUND_TRUTH_LABELS = {"normal": 0, "anomaly": 1}
OUTCOME_LABELS = {"success": 0, "fail": 1}

N_EVENTS = 29
EVENT_COLUMNS = [f"E{i}" for i in range(1, N_EVENTS + 1)]

_EVENT_TOKEN_RE = re.compile(r"E\d+")

_CANONICAL_COLUMNS = {
    "blockid": "BlockId",
    "label": "Label",
    "type": "Type",
    "features": "Features",
    "timeinterval": "TimeInterval",
    "latency": "Latency",
    "eventid": "EventId",
    "eventtemplate": "EventTemplate",
}


def hdfs_data_dir() -> Path:
    return get_settings().data_raw / HDFS_SUBDIR


def convert_label(value: Any) -> int:
    """Normal/Anomaly -> 0/1. Raises ValueError on anything else."""
    if isinstance(value, str):
        key = value.strip().lower()
        if key in GROUND_TRUTH_LABELS:
            return GROUND_TRUTH_LABELS[key]
    elif isinstance(value, numbers.Integral) and not isinstance(value, bool):
        if int(value) in (0, 1):
            return int(value)
    raise ValueError(
        f"unknown label {value!r}; expected Normal or Anomaly"
    )


def convert_outcome_label(value: Any) -> int:
    """Success/Fail -> 0/1 (matrix/traces Label column). Raises otherwise."""
    if isinstance(value, str):
        key = value.strip().lower()
        if key in OUTCOME_LABELS:
            return OUTCOME_LABELS[key]
    raise ValueError(
        f"unknown outcome label {value!r}; expected Success or Fail"
    )


def parse_event_sequence(text: Any) -> list[int]:
    """'[E5,E22,E11]' -> [5, 22, 11]. Raises ValueError on malformed input."""
    if not isinstance(text, str):
        raise ValueError(
            f"expected string, got {type(text).__name__}: {text!r}"
        )
    cleaned = text.strip()
    if not (cleaned.startswith("[") and cleaned.endswith("]")):
        raise ValueError(f"malformed event sequence: {text!r}")
    tokens = [t.strip() for t in cleaned[1:-1].split(",")] if cleaned[1:-1].strip() else []
    for token in tokens:
        if not _EVENT_TOKEN_RE.fullmatch(token):
            raise ValueError(f"malformed event sequence: {text!r}")
    return [int(token[1:]) for token in tokens]


def _read_csv_required(path: Path, name: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(
            f"{name}: file not found at {path}. Download the official Loghub "
            f"HDFS_v1 archive (Zenodo record 8196385) and place the "
            f"preprocessed files under {hdfs_data_dir()}."
        )
    try:
        return pd.read_csv(path)
    except Exception as exc:
        raise ValueError(f"{name}: failed to parse {path} as CSV: {exc}") from exc


def _canonicalize_columns(df: pd.DataFrame, name: str) -> pd.DataFrame:
    rename = {
        col: _CANONICAL_COLUMNS[str(col).strip().lower()]
        for col in df.columns
        if str(col).strip().lower() in _CANONICAL_COLUMNS
    }
    return df.rename(columns=rename)


def _require_columns(
    df: pd.DataFrame, name: str, required: set[str]
) -> None:
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"{name}: missing required columns {sorted(missing)}; "
            f"got {list(df.columns)}"
        )


def _require_no_null_block_ids(df: pd.DataFrame, name: str) -> None:
    if df["BlockId"].isna().any():
        raise ValueError(f"{name}: BlockId contains missing values")


def _validate_event_counts(df: pd.DataFrame, name: str) -> pd.DataFrame:
    for col in EVENT_COLUMNS:
        series = df[col]
        if series.isna().any():
            raise ValueError(f"{name}: column {col} contains missing values")
        numeric = pd.to_numeric(series, errors="coerce")
        bad = numeric.isna()
        if bad.any():
            examples = sorted({str(v) for v in series[bad]})[:5]
            raise ValueError(
                f"{name}: non-numeric values in column {col}: {examples}"
            )
        if (numeric < 0).any():
            raise ValueError(f"{name}: negative counts in column {col}")
        df[col] = numeric.astype("int64")
    return df


def _validate_outcome_label(df: pd.DataFrame, name: str) -> pd.DataFrame:
    if "Label" not in df.columns:
        return df
    try:
        converted = df["Label"].map(convert_outcome_label)
    except ValueError as exc:
        raise ValueError(f"{name}: {exc}") from exc
    if converted.isna().any():
        raise ValueError(
            f"{name}: unknown outcome label: missing value (null)"
        )
    df["Label"] = converted.astype("int8")
    return df


def _validate_type_column(df: pd.DataFrame, name: str) -> pd.DataFrame:
    if "Type" not in df.columns:
        return df
    numeric = pd.to_numeric(df["Type"], errors="coerce")
    bad = df["Type"].notna() & numeric.isna()
    if bad.any():
        examples = sorted({str(v) for v in df["Type"][bad]})[:5]
        raise ValueError(f"{name}: non-numeric values in column Type: {examples}")
    df["Type"] = numeric
    return df


def load_anomaly_labels(path: Path | None = None) -> pd.DataFrame:
    """Ground-truth labels: BlockId + Label, converted to 0/1."""
    path = path or hdfs_data_dir() / FILENAME_LABELS
    df = _read_csv_required(path, FILENAME_LABELS)
    df = _canonicalize_columns(df, FILENAME_LABELS)
    _require_columns(df, FILENAME_LABELS, {"BlockId", "Label"})
    _require_no_null_block_ids(df, FILENAME_LABELS)
    try:
        labels = df["Label"].map(convert_label)
    except ValueError as exc:
        raise ValueError(f"{FILENAME_LABELS}: {exc}") from exc
    if labels.isna().any():
        raise ValueError(f"{FILENAME_LABELS}: unknown label: missing value (null)")
    df["Label"] = labels.astype("int8")
    return df


def load_occurrence_matrix(path: Path | None = None) -> pd.DataFrame:
    """BlockId x 29 event-id occurrence counts (int64, no nulls)."""
    path = path or hdfs_data_dir() / FILENAME_MATRIX
    df = _read_csv_required(path, FILENAME_MATRIX)
    df = _canonicalize_columns(df, FILENAME_MATRIX)
    _require_columns(df, FILENAME_MATRIX, {"BlockId", *EVENT_COLUMNS})
    _require_no_null_block_ids(df, FILENAME_MATRIX)
    df = _validate_event_counts(df, FILENAME_MATRIX)
    df = _validate_outcome_label(df, FILENAME_MATRIX)
    df = _validate_type_column(df, FILENAME_MATRIX)
    return df


def load_event_traces(path: Path | None = None) -> pd.DataFrame:
    """BlockId + raw Features (event-sequence strings) and metadata."""
    path = path or hdfs_data_dir() / FILENAME_TRACES
    df = _read_csv_required(path, FILENAME_TRACES)
    df = _canonicalize_columns(df, FILENAME_TRACES)
    _require_columns(df, FILENAME_TRACES, {"BlockId", "Features"})
    _require_no_null_block_ids(df, FILENAME_TRACES)
    if df["Features"].isna().any():
        raise ValueError(f"{FILENAME_TRACES}: Features contains missing values")
    try:
        df["Features"].map(parse_event_sequence)
    except ValueError as exc:
        raise ValueError(f"{FILENAME_TRACES}: {exc}") from exc
    df = _validate_outcome_label(df, FILENAME_TRACES)
    df = _validate_type_column(df, FILENAME_TRACES)
    if "Latency" in df.columns:
        latency = pd.to_numeric(df["Latency"], errors="coerce")
        if latency.isna().any():
            raise ValueError(f"{FILENAME_TRACES}: non-numeric Latency values")
        df["Latency"] = latency
    return df


def load_templates(
    path: Path | None = None, required: bool = True
) -> pd.DataFrame | None:
    """EventId -> EventTemplate mapping (E1..E29)."""
    path = path or hdfs_data_dir() / FILENAME_TEMPLATES
    if not path.is_file():
        if required:
            raise FileNotFoundError(
                f"{FILENAME_TEMPLATES}: file not found at {path}"
            )
        return None
    df = _read_csv_required(path, FILENAME_TEMPLATES)
    df = _canonicalize_columns(df, FILENAME_TEMPLATES)
    _require_columns(df, FILENAME_TEMPLATES, {"EventId", "EventTemplate"})
    bad_ids = df["EventId"][~df["EventId"].astype(str).str.fullmatch(r"E\d+")]
    if len(bad_ids):
        raise ValueError(
            f"{FILENAME_TEMPLATES}: malformed EventId values: "
            f"{sorted(bad_ids.astype(str).unique())[:5]}"
        )
    if df["EventId"].duplicated().any():
        raise ValueError(f"{FILENAME_TEMPLATES}: duplicate EventId values")
    return df


def load_hdfs(require_templates: bool = True) -> dict[str, pd.DataFrame]:
    """Load all HDFS_v1 files and cross-validate them against ground truth."""
    labels = load_anomaly_labels()
    matrix = load_occurrence_matrix()
    traces = load_event_traces()
    templates = load_templates(required=require_templates)

    for name, other in (("occurrence matrix", matrix), ("traces", traces)):
        if len(other) != len(labels):
            raise ValueError(
                f"{name}: {len(other)} rows vs {len(labels)} ground-truth rows"
            )
        if not set(other["BlockId"]) == set(labels["BlockId"]):
            raise ValueError(
                f"{name}: BlockId set differs from ground truth"
            )
        merged = labels[["BlockId", "Label"]].merge(
            other[["BlockId", "Label"]], on="BlockId", suffixes=("_gt", "_out")
        )
        disagree = merged[merged["Label_gt"] != merged["Label_out"]]
        if len(disagree):
            raise ValueError(
                f"{name}: labels disagree with ground truth on "
                f"{len(disagree)} block(s)"
            )

    result: dict[str, pd.DataFrame] = {
        "labels": labels,
        "occurrence_matrix": matrix,
        "traces": traces,
    }
    if templates is not None:
        result["templates"] = templates
    return result
