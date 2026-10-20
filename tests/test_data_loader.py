"""Tests for sentinelai.data_loader.

All fixture CSVs here are minimal hand-written structural examples used only
to exercise error handling and schema validation. They are NOT dataset
substitutes — the real HDFS_v1 data lives exclusively in data/raw/hdfs/.
"""
from __future__ import annotations

import pandas as pd
import pytest

from sentinelai.data_loader import (
    EVENT_COLUMNS,
    FILENAME_LABELS,
    convert_label,
    convert_outcome_label,
    load_anomaly_labels,
    load_event_traces,
    load_hdfs,
    load_occurrence_matrix,
    load_templates,
    parse_event_sequence,
)


# ---------------------------------------------------------------- labels


class TestConvertLabel:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ("Normal", 0),
            ("Anomaly", 1),
            (" normal ", 0),
            ("ANOMALY", 1),
            (0, 0),
            (1, 1),
        ],
    )
    def test_valid(self, value, expected):
        assert convert_label(value) == expected

    @pytest.mark.parametrize(
        "value",
        ["Suspicious", "", "normall", None, 2, -1, 0.5, float("nan"), True],
    )
    def test_invalid_raises(self, value):
        with pytest.raises(ValueError, match="unknown label"):
            convert_label(value)

    def test_outcome_labels(self):
        assert convert_outcome_label("Success") == 0
        assert convert_outcome_label("Fail") == 1
        assert convert_outcome_label(" fail ") == 1
        with pytest.raises(ValueError, match="unknown outcome label"):
            convert_outcome_label("Normal")


# ------------------------------------------------------ event sequences


class TestParseEventSequence:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("[E5,E22,E11]", [5, 22, 11]),
            ("[]", []),
            (" [E3, E5] ", [3, 5]),
            ("[E29]", [29]),
        ],
    )
    def test_valid(self, text, expected):
        assert parse_event_sequence(text) == expected

    @pytest.mark.parametrize(
        "text",
        ["E5,E22", "[E5,X]", "[5,22]", "garbage", "[E5,E22", "", None, 42],
    )
    def test_malformed_raises(self, text):
        with pytest.raises(ValueError, match="malformed event sequence|expected string"):
            parse_event_sequence(text)


# ---------------------------------------------------------------- loading


LABELS_CSV = "BlockId,Label\nblk_1,Normal\nblk_2,Anomaly\nblk_3,Normal\n"
MATRIX_CSV = (
    "BlockId,Label,Type,"
    + ",".join(EVENT_COLUMNS)
    + "\n"
    + "blk_1,Success,,"
    + ",".join(["1"] + ["0"] * 28)
    + "\n"
    + "blk_2,Fail,5.0,"
    + ",".join(["0", "3"] + ["0"] * 27)
    + "\n"
    + "blk_3,Success,,"
    + ",".join(["2"] + ["0"] * 28)
    + "\n"
)
TRACES_CSV = (
    "BlockId,Label,Type,Features,TimeInterval,Latency\n"
    'blk_1,Success,,"[E5,E22]",0,120\n'
    'blk_2,Fail,5.0,"[E1,E1,E2]",0,5400\n'
    'blk_3,Success,,"[E29]",0,300\n'
)
TEMPLATES_CSV = (
    "EventId,EventTemplate\n"
    "E1,[*]Adding an already existing block[*]\n"
    "E2,[*]Verification succeeded for[*]\n"
)


def matrix_csv(
    label: str = "Success",
    row_type: str = "",
    overrides: dict[str, object] | None = None,
) -> str:
    """One-row matrix CSV with all 29 event columns."""
    values: dict[str, object] = {col: 0 for col in EVENT_COLUMNS}
    if overrides:
        values.update(overrides)
    row = ",".join(str(values[col]) for col in EVENT_COLUMNS)
    return (
        f"BlockId,Label,Type,{','.join(EVENT_COLUMNS)}\n"
        f"blk_1,{label},{row_type},{row}\n"
    )


class TestLoadAnomalyLabels:
    def test_valid_schema(self, tmp_path):
        path = tmp_path / FILENAME_LABELS
        path.write_text(LABELS_CSV)
        df = load_anomaly_labels(path)
        assert list(df.columns) == ["BlockId", "Label"]
        assert df["Label"].tolist() == [0, 1, 0]
        assert df["Label"].dtype == "int8"

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match=FILENAME_LABELS):
            load_anomaly_labels(tmp_path / FILENAME_LABELS)

    def test_malformed_csv_raises(self, tmp_path):
        path = tmp_path / FILENAME_LABELS
        path.write_text('BlockId,Label\n"unclosed,1\n')
        with pytest.raises(ValueError, match="failed to parse"):
            load_anomaly_labels(path)

    def test_invalid_label_raises(self, tmp_path):
        path = tmp_path / FILENAME_LABELS
        path.write_text("BlockId,Label\nblk_1,Normal\nblk_2,Suspicious\n")
        with pytest.raises(ValueError, match="Suspicious"):
            load_anomaly_labels(path)

    def test_missing_label_column_raises(self, tmp_path):
        path = tmp_path / FILENAME_LABELS
        path.write_text("BlockId,Other\nblk_1,Normal\n")
        with pytest.raises(ValueError, match="missing required columns"):
            load_anomaly_labels(path)

    def test_null_label_raises(self, tmp_path):
        path = tmp_path / FILENAME_LABELS
        path.write_text("BlockId,Label\nblk_1,Normal\nblk_2,\n")
        with pytest.raises(ValueError, match="unknown label"):
            load_anomaly_labels(path)


class TestLoadOccurrenceMatrix:
    def test_expected_schema(self, tmp_path):
        path = tmp_path / "Event_occurrence_matrix.csv"
        path.write_text(MATRIX_CSV)
        df = load_occurrence_matrix(path)
        assert list(df.columns) == ["BlockId", "Label", "Type", *EVENT_COLUMNS]
        assert len(df) == 3
        assert df[EVENT_COLUMNS].dtypes.eq("int64").all()
        assert df["Label"].tolist() == [0, 1, 0]

    def test_non_numeric_event_column_raises(self, tmp_path):
        path = tmp_path / "Event_occurrence_matrix.csv"
        path.write_text(matrix_csv(overrides={"E1": "oops"}))
        with pytest.raises(ValueError, match="non-numeric values in column E1"):
            load_occurrence_matrix(path)

    def test_negative_count_raises(self, tmp_path):
        path = tmp_path / "Event_occurrence_matrix.csv"
        path.write_text(matrix_csv(overrides={"E1": -1}))
        with pytest.raises(ValueError, match="negative counts"):
            load_occurrence_matrix(path)

    def test_missing_event_column_raises(self, tmp_path):
        path = tmp_path / "Event_occurrence_matrix.csv"
        path.write_text("BlockId,E1\nblk_1,1\n")
        with pytest.raises(ValueError, match="missing required columns"):
            load_occurrence_matrix(path)

    def test_invalid_outcome_label_raises(self, tmp_path):
        path = tmp_path / "Event_occurrence_matrix.csv"
        path.write_text(matrix_csv(label="Normal"))
        with pytest.raises(ValueError, match="unknown outcome label"):
            load_occurrence_matrix(path)


class TestLoadEventTraces:
    def test_expected_schema(self, tmp_path):
        path = tmp_path / "Event_traces.csv"
        path.write_text(TRACES_CSV)
        df = load_event_traces(path)
        assert list(df.columns) == [
            "BlockId", "Label", "Type", "Features", "TimeInterval", "Latency",
        ]
        assert df["Label"].tolist() == [0, 1, 0]
        assert df["Latency"].tolist() == [120, 5400, 300]

    def test_malformed_features_raises(self, tmp_path):
        path = tmp_path / "Event_traces.csv"
        path.write_text('BlockId,Label,Features\nblk_1,Success,"[E5,E22]"\nblk_2,Fail,"not-a-sequence"\n')
        with pytest.raises(ValueError, match="malformed event sequence"):
            load_event_traces(path)


class TestLoadTemplates:
    def test_valid(self, tmp_path):
        path = tmp_path / "HDFS_templates.csv"
        path.write_text(TEMPLATES_CSV)
        df = load_templates(path)
        assert list(df.columns) == ["EventId", "EventTemplate"]
        assert df["EventId"].tolist() == ["E1", "E2"]

    def test_optional_missing_returns_none(self, tmp_path):
        assert load_templates(tmp_path / "HDFS_templates.csv", required=False) is None

    def test_required_missing_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_templates(tmp_path / "HDFS_templates.csv", required=True)

    def test_duplicate_event_ids_raise(self, tmp_path):
        path = tmp_path / "HDFS_templates.csv"
        path.write_text("EventId,EventTemplate\nE1,a\nE1,b\n")
        with pytest.raises(ValueError, match="duplicate EventId"):
            load_templates(path)

    def test_malformed_event_id_raises(self, tmp_path):
        path = tmp_path / "HDFS_templates.csv"
        path.write_text("EventId,EventTemplate\nX7,a\n")
        with pytest.raises(ValueError, match="malformed EventId"):
            load_templates(path)


class TestLoadHdfs:
    def _write(self, tmp_path: Path) -> None:
        (tmp_path / "anomaly_label.csv").write_text(LABELS_CSV)
        (tmp_path / "Event_occurrence_matrix.csv").write_text(MATRIX_CSV)
        (tmp_path / "Event_traces.csv").write_text(TRACES_CSV)
        (tmp_path / "HDFS_templates.csv").write_text(TEMPLATES_CSV)

    def test_consistent_dataset(self, tmp_path, monkeypatch):
        self._write(tmp_path)
        monkeypatch.setattr(
            "sentinelai.data_loader.hdfs_data_dir", lambda: tmp_path
        )
        data = load_hdfs()
        assert set(data) == {"labels", "occurrence_matrix", "traces", "templates"}
        assert len(data["labels"]) == 3

    def test_mismatched_block_ids_raise(self, tmp_path, monkeypatch):
        self._write(tmp_path)
        (tmp_path / "Event_occurrence_matrix.csv").write_text(
            MATRIX_CSV.replace("blk_3", "blk_999")
        )
        monkeypatch.setattr(
            "sentinelai.data_loader.hdfs_data_dir", lambda: tmp_path
        )
        with pytest.raises(ValueError, match="BlockId set differs"):
            load_hdfs()

    def test_disagreeing_labels_raise(self, tmp_path, monkeypatch):
        self._write(tmp_path)
        (tmp_path / "Event_traces.csv").write_text(
            TRACES_CSV.replace("blk_1,Success", "blk_1,Fail")
        )
        monkeypatch.setattr(
            "sentinelai.data_loader.hdfs_data_dir", lambda: tmp_path
        )
        with pytest.raises(ValueError, match="disagree with ground truth"):
            load_hdfs()

    def test_row_count_mismatch_raises(self, tmp_path, monkeypatch):
        self._write(tmp_path)
        (tmp_path / "Event_traces.csv").write_text(
            TRACES_CSV.rsplit("\n", 2)[0] + "\n"
        )
        monkeypatch.setattr(
            "sentinelai.data_loader.hdfs_data_dir", lambda: tmp_path
        )
        with pytest.raises(ValueError, match="rows vs"):
            load_hdfs()


# ------------------------------------------------------------- pandas 3.x


def test_loaded_frames_are_plain_pandas(tmp_path):
    path = tmp_path / FILENAME_LABELS
    path.write_text(LABELS_CSV)
    df = load_anomaly_labels(path)
    assert isinstance(df, pd.DataFrame)
