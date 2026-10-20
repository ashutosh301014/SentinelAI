"""Tests for the production inference pipeline (sentinelai.inference).

Covers artifact consistency, the no-retrain contract, input validation,
stage structure, determinism, and the end-to-end execution on held-out
normal and anomalous samples (random_state=42 split).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sentinelai.config import get_settings
from sentinelai.data_loader import load_anomaly_labels, load_occurrence_matrix
from sentinelai.dependency_graph import service_criticality
from sentinelai.explainability import pipeline_feature_names
from sentinelai.inference import PIPELINE_STAGES, InferencePipeline, InferenceResult
from sentinelai.preprocessing import grouped_split_dataset


@pytest.fixture(scope="module")
def settings():
    return get_settings()


@pytest.fixture(scope="module")
def pipeline(settings) -> InferencePipeline:
    return InferencePipeline()


@pytest.fixture(scope="module")
def held_out():
    matrix = load_occurrence_matrix()
    labels = load_anomaly_labels()
    X_train, X_test, y_train, y_test = grouped_split_dataset(matrix, labels)
    return {"X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test}


def _raw_row(base: pd.DataFrame, **changes) -> pd.DataFrame:
    row = base.iloc[[0]].copy()
    for column, value in changes.items():
        row[column] = value
    return row


def _pick_clearly_normal(pipeline: InferencePipeline, X_normal: pd.DataFrame) -> pd.DataFrame:
    candidates = X_normal.iloc[:200]
    predicted_normal = pipeline.classifier.predict(candidates) == 0
    candidates = candidates[predicted_normal]
    scores = pipeline.detector.score(candidates)
    return candidates.iloc[[int(scores.argmin())]]


def _pick_confident_anomaly(pipeline: InferencePipeline, X_anomaly: pd.DataFrame) -> pd.DataFrame:
    candidates = X_anomaly.iloc[:20]
    predicted_anomaly = pipeline.classifier.predict(candidates) == 1
    return candidates[predicted_anomaly].iloc[[0]]


# ---- artifact consistency ----

def test_metadata_has_required_fields(pipeline):
    metadata = pipeline.metadata
    for key in (
        "dataset",
        "trained_at",
        "selected_model",
        "model_version",
        "feature_count",
        "feature_names",
        "evaluation_metrics",
    ):
        assert key in metadata
    metrics = metadata["evaluation_metrics"]
    for key in ("precision", "recall", "f1", "roc_auc", "pr_auc", "fpr"):
        assert key in metrics


def test_feature_order_matches_pipeline_and_artifacts(pipeline, settings):
    fitted = pipeline_feature_names(pipeline.classifier)
    assert fitted == pipeline.feature_order
    assert fitted == pipeline.metadata["feature_names"]
    assert len(fitted) == pipeline.metadata["feature_count"]
    on_disk = json.loads(
        (settings.models_dir / "feature_order.json").read_text(encoding="utf-8")
    )
    assert on_disk["features"] == list(fitted)


def test_preprocessing_metadata_matches_fitted_pipeline(pipeline):
    metadata = pipeline.preprocessing_metadata
    select = pipeline.classifier.named_steps["select"]
    derive = pipeline.classifier.named_steps["derive"]
    assert metadata["input_event_columns"] == list(select.input_event_columns_)
    assert metadata["kept_event_columns"] == list(select.kept_event_columns_)
    assert metadata["constant_columns"] == list(select.constant_columns_)
    assert metadata["duplicate_columns"] == list(select.duplicate_columns_)
    assert metadata["rare_event_columns"] == list(derive.rare_event_columns_)
    assert metadata["constant_columns"] == ["E22"]
    assert metadata["duplicate_columns"] == ["E25"]
    assert metadata["feature_count"] == 35


def test_event_templates_cover_all_event_ids(pipeline):
    ids = set(pipeline.templates["EventId"])
    assert ids == {f"E{i}" for i in range(1, 30)}
    assert pipeline.templates["EventTemplate"].notna().all()


def test_pipeline_exposes_no_fit_path(pipeline):
    assert not hasattr(InferencePipeline, "fit")
    assert not hasattr(InferencePipeline, "fit_transform")
    assert not hasattr(InferencePipeline, "partial_fit")


def test_component_mapping_is_data_driven(pipeline, settings):
    path = settings.component_service_mapping
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload["mapping"], list) and payload["mapping"]
    for rule in payload["mapping"]:
        assert rule["service_id"] in pipeline.graph
        assert rule["keywords"]


# ---- stage 8: dependency graph mapping ----

def test_mapping_of_known_components(pipeline):
    expected = {
        "NameNode (FSNamesystem/BlockManager)": "database",
        "NameNode (client request serving)": "api_gateway",
        "DataNode (block pipeline)": "inventory",
        "DataNode (BlockScanner)": "inventory",
    }
    for component, service in expected.items():
        mapped = pipeline._map_component(component)
        assert mapped["status"] == "mapped"
        assert mapped["service_id"] == service
        assert mapped["service_name"] == pipeline.graph.nodes[service]["name"]


def test_mapping_no_anomaly_skips(pipeline):
    mapped = pipeline._map_component("none (no anomaly detected)")
    assert mapped["status"] == "skipped_no_anomaly"
    assert mapped["service_id"] is None


def test_mapping_unknown_component_falls_back_to_highest_criticality(pipeline):
    mapped = pipeline._map_component("Probable Cause: unknown — no evidence.")
    assert mapped["status"] == "fallback_highest_criticality"
    criticalities = {
        str(node): service_criticality(pipeline.graph, str(node))
        for node in pipeline.graph.nodes
    }
    assert service_criticality(pipeline.graph, mapped["service_id"]) == max(
        criticalities.values()
    )


# ---- stage 1: input validation ----

def test_validation_rejects_non_dataframe(pipeline):
    with pytest.raises(TypeError, match="DataFrame"):
        pipeline._validate_raw_sample([[1] * 29])


def test_validation_rejects_multiple_rows(pipeline, held_out):
    sample = held_out["X_test"].iloc[:2]
    with pytest.raises(ValueError, match="exactly one row"):
        pipeline._validate_raw_sample(sample)


def test_validation_rejects_missing_column(pipeline, held_out):
    sample = held_out["X_test"].iloc[[0]].drop(columns=["E1"])
    with pytest.raises(ValueError, match="missing"):
        pipeline._validate_raw_sample(sample)


def test_validation_rejects_unseen_column(pipeline, held_out):
    sample = _raw_row(held_out["X_test"], Label=1)
    with pytest.raises(ValueError, match="unseen"):
        pipeline._validate_raw_sample(sample)


def test_validation_rejects_non_numeric(pipeline, held_out):
    sample = _raw_row(held_out["X_test"], E1="abc")
    with pytest.raises(ValueError, match="non-numeric"):
        pipeline._validate_raw_sample(sample)


def test_validation_rejects_negative_counts(pipeline, held_out):
    sample = _raw_row(held_out["X_test"], E1=-1)
    with pytest.raises(ValueError, match="negative counts"):
        pipeline._validate_raw_sample(sample)


def test_validation_rejects_infinite(pipeline, held_out):
    sample = _raw_row(held_out["X_test"], E1=np.inf)
    with pytest.raises(ValueError, match="infinite"):
        pipeline._validate_raw_sample(sample)


def test_validation_accepts_permuted_column_order(pipeline, held_out):
    base = held_out["X_test"].iloc[[0]]
    shuffled = base.reindex(columns=list(reversed(base.columns)))
    validated = pipeline._validate_raw_sample(shuffled)
    assert validated["input_event_columns"] == validated["expected_event_columns"]


# ---- stage structure and determinism ----

def test_result_covers_all_ten_stages(pipeline, held_out):
    sample = _pick_clearly_normal(pipeline, held_out["X_test"][held_out["y_test"] == 0])
    result = pipeline.run(sample)
    assert isinstance(result, InferenceResult)
    assert result.stages == PIPELINE_STAGES
    payload = result.to_dict()
    assert set(payload["stages"]) == {
        "input_validation",
        "feature_engineering",
        "supervised_classifier",
        "isolation_forest",
        "composite_risk_score",
        "explainability",
        "incident_intelligence",
        "dependency_graph_mapping",
        "blast_radius_prediction",
        "what_if_simulation",
    }
    assert len(payload["feature_engineering"]["features"]) == 35
    contributions = payload["composite_risk_score"]["contributions"]
    assert sum(contributions.values()) == payload["composite_risk_score"]["risk_score"]
    assert "NOT a calibrated probability" in payload["composite_risk_score"]["caveat"]


def test_repeated_runs_are_idempotent(pipeline, held_out):
    sample = _pick_clearly_normal(pipeline, held_out["X_test"][held_out["y_test"] == 0])
    first = pipeline.run(sample).to_dict()
    second = pipeline.run(sample).to_dict()
    assert first == second


def test_module_declares_no_retraining():
    import sentinelai.inference as module

    doc = " ".join(module.__doc__.lower().split())
    assert "never retrains" in doc
    assert "estimated propagation risk" in doc


# ---- no-retrain proof: artifacts are byte-identical after inference ----

def test_inference_does_not_modify_any_artifact(pipeline, settings, held_out):
    files = sorted(settings.models_dir.iterdir())
    snapshots = {path.name: path.read_bytes() for path in files}
    sample = _pick_confident_anomaly(
        pipeline, held_out["X_test"][held_out["y_test"] == 1]
    )
    pipeline.run(sample, what_if=[{"action": "restart_service", "affected_service": "database"}])
    for path in files:
        assert path.read_bytes() == snapshots[path.name], (
            f"artifact {path.name} changed during inference — "
            "production inference must never write to models/"
        )


# ---- what-if parsing ----

def test_what_if_rejects_unknown_action(pipeline):
    with pytest.raises(ValueError, match="unknown recovery action"):
        pipeline._parse_what_if({"action": "reboot_everything", "affected_service": "database"})


def test_what_if_rejects_unknown_service(pipeline):
    with pytest.raises(ValueError, match="unknown service"):
        pipeline._parse_what_if({"action": "restart_service", "affected_service": "mainframe"})


def test_what_if_rejects_non_dict(pipeline):
    with pytest.raises(TypeError, match="dicts"):
        pipeline._parse_what_if(["restart_service"])


# ---- end-to-end on held-out samples ----

def test_end_to_end_normal_held_out_sample(pipeline, held_out):
    sample = _pick_clearly_normal(pipeline, held_out["X_test"][held_out["y_test"] == 0])
    result = pipeline.run(sample)
    payload = result.to_dict()
    assert payload["input_validation"]["status"] == "passed"
    assert payload["supervised_classifier"]["prediction"] == 0
    assert payload["supervised_classifier"]["anomaly_probability"] < 0.5
    assert payload["incident_intelligence"]["anomaly_detected"] is False
    assert payload["dependency_graph_mapping"]["status"] == "skipped_no_anomaly"
    assert payload["blast_radius_prediction"]["status"] == "skipped"
    assert payload["composite_risk_score"]["severity"] == "LOW"
    assert payload["what_if_simulation"] == []


def test_end_to_end_anomalous_held_out_sample(pipeline, held_out):
    sample = _pick_confident_anomaly(
        pipeline, held_out["X_test"][held_out["y_test"] == 1]
    )
    result = pipeline.run(sample, timestamp="2026-10-19T14:30:00Z")
    payload = result.to_dict()
    assert payload["supervised_classifier"]["prediction"] == 1
    assert payload["supervised_classifier"]["anomaly_probability"] > 0.9
    assert payload["incident_intelligence"]["anomaly_detected"] is True
    assert payload["incident_intelligence"]["timestamp"] == "2026-10-19T14:30:00Z"

    mapping = payload["dependency_graph_mapping"]
    assert mapping["status"] in {"mapped", "fallback_highest_criticality"}
    assert mapping["service_id"] in pipeline.graph

    blast = payload["blast_radius_prediction"]
    assert blast["status"] == "computed"
    assert blast["probable_origin"]["id"] == mapping["service_id"]
    expected_origin_risk = payload["composite_risk_score"]["risk_score"] / 100.0
    assert blast["probable_origin"]["incident_risk"] == pytest.approx(
        expected_origin_risk
    )
    path = blast["most_likely_propagation_path"]
    services_at_risk = blast["services_at_risk"]

    if services_at_risk:
        assert path
        assert path[0] == mapping["service_id"]
    else:
        assert path == []
        assert blast["highest_risk_service"] is None




def test_end_to_end_what_if_simulation(pipeline, held_out):
    sample = _pick_confident_anomaly(
        pipeline, held_out["X_test"][held_out["y_test"] == 1]
    )
    first = pipeline.run(sample)
    origin = first.dependency_graph_mapping["service_id"]
    result = pipeline.run(
        sample,
        what_if=[
            {"action": "restart_service", "affected_service": origin},
            {"action": "increase_database_capacity", "affected_service": origin},
        ],
    )
    simulations = result.what_if_simulation
    assert len(simulations) == 2
    for simulation in simulations:
        assert simulation["status"] == "estimated counterfactual impact"
        assert simulation["original_incident_risk"] == first.composite_risk_score["risk_score"]
        assert 0 <= simulation["estimated_risk_reduction"] <= 100
    assert simulations[0]["simulated_incident_risk"] == 0
