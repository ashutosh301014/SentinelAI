"""Unit tests for sentinelai.incident_engine (toy pipelines only)."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from sentinelai.anomaly_detector import AnomalyDetector
from sentinelai.explainability import build_training_profile
from sentinelai.incident_engine import (
    IncidentReport,
    build_incident_report,
    component_for_template,
    event_intensity,
    event_irregularity,
)
from sentinelai.modeling import build_classifier_pipeline

REPORT_FIELDS = {
    "incident_id",
    "timestamp",
    "anomaly_detected",
    "classifier_probability",
    "anomaly_score",
    "risk_score",
    "severity",
    "top_evidence",
    "probable_contributing_events",
    "probable_component",
    "probable_cause",
    "investigation_recommendations",
}

SEVERITIES = {"LOW", "MEDIUM", "HIGH", "CRITICAL"}

TOY_TEMPLATES = pd.DataFrame(
    {
        "EventId": ["E1", "E2", "E3"],
        "EventTemplate": [
            "PacketResponder <*> for block <*> terminating",
            "BLOCK* NameSystem.delete: <*> is added to invalidSet of <*>",
            "Verification succeeded for <*>",
        ],
    }
)


def make_toy_frame(
    n: int = 300, n_anomaly: int = 60, seed: int = 0
) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(
        rng.integers(0, 4, size=(n, 6)),
        columns=[f"E{i}" for i in range(1, 7)],
    )
    anomaly_idx = rng.choice(n, size=n_anomaly, replace=False)
    frame.loc[anomaly_idx, "E1"] = rng.integers(20, 50, size=n_anomaly)
    frame.loc[anomaly_idx, "E3"] = rng.integers(10, 30, size=n_anomaly)
    y = pd.Series(np.zeros(n, dtype="int8"), name="Label")
    y.iloc[anomaly_idx] = 1
    return frame, y


@pytest.fixture(scope="module")
def engine():
    frame, y = make_toy_frame()
    pipeline = build_classifier_pipeline(
        "hist_gradient_boosting", random_state=0
    ).fit(frame, y)
    detector = AnomalyDetector(random_state=0).fit(frame, y)
    profile = build_training_profile(pipeline, frame[y.to_numpy() == 0])
    return frame, y, pipeline, detector, profile


def _report(engine, sample, **kwargs):
    frame, _, pipeline, detector, profile = engine
    return build_incident_report(
        str(sample.index[0]),
        sample,
        pipeline,
        detector,
        profile,
        templates=TOY_TEMPLATES,
        **kwargs,
    )


class TestReportStructure:
    def test_all_fields_present(self, engine):
        frame, y, _, _, _ = engine
        report = _report(engine, frame.iloc[[0]])
        assert set(report.to_dict()) == REPORT_FIELDS

    def test_field_types(self, engine):
        frame, _, _, _, _ = engine
        report = _report(engine, frame.iloc[[0]])
        assert isinstance(report.incident_id, str)
        assert report.incident_id.startswith("INC-")
        assert report.timestamp is None
        assert isinstance(report.anomaly_detected, bool)
        assert 0.0 <= report.classifier_probability <= 1.0
        assert 0.0 <= report.anomaly_score <= 1.0
        assert 0 <= report.risk_score <= 100
        assert report.severity in SEVERITIES
        assert isinstance(report.top_evidence, list)
        assert isinstance(report.probable_cause, str)
        assert isinstance(report.investigation_recommendations, list)

    def test_timestamp_preserved(self, engine):
        frame, _, _, _, _ = engine
        report = _report(
            engine, frame.iloc[[0]], timestamp="2026-10-19T12:00:00Z"
        )
        assert report.timestamp == "2026-10-19T12:00:00Z"


class TestNormalReport:
    def test_typical_normal_trace(self, engine):
        frame, y, pipeline, _, profile = engine
        normal_rows = np.where(y.to_numpy() == 0)[0]
        irregularities = [
            event_irregularity(profile, pipeline, frame.iloc[[pos]])
            for pos in normal_rows
        ]
        typical_pos = normal_rows[int(np.argmin(irregularities))]
        report = _report(engine, frame.iloc[[typical_pos]])
        assert report.anomaly_detected is False
        assert report.probable_contributing_events == []
        assert report.probable_component == "none (no anomaly detected)"
        assert "Probable Cause: none identified" in report.probable_cause
        assert any(
            "archive this trace" in rec
            for rec in report.investigation_recommendations
        )


class TestAnomalousReport:
    def test_anomalous_trace(self, engine):
        frame, y, _, _, _ = engine
        anomaly_pos = int(np.where(y.to_numpy() == 1)[0][0])
        report = _report(engine, frame.iloc[[anomaly_pos]])
        assert report.anomaly_detected is True
        assert report.probable_cause.startswith("Probable Cause: ")
        assert report.probable_contributing_events
        for event in report.probable_contributing_events:
            assert event["event_id"].startswith("E")
            assert event["event_id"][1:].isdigit()
            assert "contribution" in event
            assert event["contribution"] > 0
        assert report.probable_component != "none (no anomaly detected)"
        assert report.investigation_recommendations

    def test_cause_includes_root_cause_evidence(self, engine):
        frame, y, _, _, _ = engine
        anomaly_pos = int(np.where(y.to_numpy() == 1)[0][0])
        report = _report(engine, frame.iloc[[anomaly_pos]])
        assert "Root-Cause Evidence" in report.probable_cause


class TestTerminology:
    def test_never_claims_confirmed_root_cause(self, engine):
        frame, y, _, _, _ = engine
        for pos in (0, int(np.where(y.to_numpy() == 1)[0][0])):
            report = _report(engine, frame.iloc[[pos]])
            serialized = json.dumps(report.to_dict()).lower()
            assert "confirmed root cause" not in serialized
            assert "root cause confirmed" not in serialized
            assert "confirmed" not in serialized


class TestComponentMapping:
    def test_component_keyword_rules(self):
        assert (
            component_for_template(
                "PacketResponder <*> for block <*> terminating"
            )
            == "DataNode (block pipeline)"
        )
        assert (
            component_for_template(
                "BLOCK* NameSystem.delete: <*> is added to invalidSet of <*>"
            )
            == "NameNode (FSNamesystem/BlockManager)"
        )
        assert (
            component_for_template("Verification succeeded for <*>")
            == "DataNode (BlockScanner)"
        )

    def test_unknown_template_falls_back(self):
        assert component_for_template("totally unrecognizable text") == (
            "unknown component (insufficient evidence)"
        )
        assert component_for_template(None) == (
            "unknown component (insufficient evidence)"
        )


class TestSignalHelpers:
    def test_irregularity_in_unit_range(self, engine):
        frame, _, pipeline, _, profile = engine
        for pos in (0, 100, 200):
            value = event_irregularity(
                profile, pipeline, frame.iloc[[pos]]
            )
            assert 0.0 <= value <= 1.0

    def test_irregularity_extreme_higher_than_typical(self, engine):
        frame, y, pipeline, _, profile = engine
        typical = frame.iloc[[int(np.where(y.to_numpy() == 0)[0][0])]]
        extreme = frame.iloc[[0]].copy()
        for col in frame.columns:
            extreme[col] = 500
        assert event_irregularity(
            profile, pipeline, extreme
        ) > event_irregularity(profile, pipeline, typical)

    def test_intensity_in_unit_range_and_capped(self, engine):
        frame, _, pipeline, _, profile = engine
        typical = event_intensity(profile, pipeline, frame.iloc[[0]])
        extreme = frame.iloc[[0]].copy()
        for col in frame.columns:
            extreme[col] = 500
        capped = event_intensity(profile, pipeline, extreme)
        assert 0.0 <= typical <= 1.0
        assert capped == 1.0


class TestDeterminism:
    def test_repeated_calls_identical(self, engine):
        frame, y, _, _, _ = engine
        anomaly_pos = int(np.where(y.to_numpy() == 1)[0][0])
        first = _report(engine, frame.iloc[[anomaly_pos]]).to_dict()
        second = _report(engine, frame.iloc[[anomaly_pos]]).to_dict()
        assert first == second
