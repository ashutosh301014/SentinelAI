"""FastAPI backend tests: every endpoint, payload validation, structured
error envelope, body-size limit and no-stack-trace guarantees.

The client fixture is module-scoped: the lifespan handler loads the
trained artifacts exactly once (production inference never retrains).
"""
from __future__ import annotations

import json

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.main import MAX_BODY_BYTES, app
from sentinelai.config import get_settings

SETTINGS = get_settings()
DEMO_DIR = SETTINGS.data_demo
API = SETTINGS.api_prefix

_DEMO_FILES = {
    "normal": "demo_normal_block.csv",
    "anomalous": "demo_anomalous_block.csv",
}


def _demo_events(role: str) -> dict[str, int]:
    frame = pd.read_csv(DEMO_DIR / _DEMO_FILES[role], index_col=0)
    return {str(col): int(frame.iloc[0][col]) for col in frame.columns}


@pytest.fixture(scope="module")
def client():
    # Starlette's ServerErrorMiddleware always re-raises after producing the
    # 500 response (so servers can log it); disable the TestClient re-raise
    # and assert on status codes/envelopes explicitly instead.
    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client


# ---- root / api index / static assets ----


def test_root_serves_dashboard(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    body = response.text
    assert "SentinelAI" in body
    assert "/static/css/dashboard.css" in body
    assert "/static/js/dashboard.js" in body


def test_static_assets_served(client: TestClient) -> None:
    for path, content_type in (
        ("/static/css/dashboard.css", "text/css"),
        ("/static/js/dashboard.js", "text/javascript"),
    ):
        response = client.get(path)
        assert response.status_code == 200
        assert content_type in response.headers["content-type"]
        assert response.text.strip()


def test_api_index_lists_all_endpoints(client: TestClient) -> None:
    response = client.get(API)
    assert response.status_code == 200
    body = response.json()
    assert body["name"] == SETTINGS.app_name
    assert body["api"] == API
    listed = body["endpoints"]
    for suffix in (
        "health",
        "model-info",
        "analyze",
        "demo/normal",
        "demo/anomaly",
        "demo/analyze",
        "dependency-graph",
        "blast-radius",
        "simulate-recovery",
    ):
        assert f"{API}/{suffix}" in listed


def test_health_reports_loaded_model(client: TestClient) -> None:
    response = client.get(f"{API}/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["dataset"] == "HDFS_v1"
    assert body["model_version"]


def test_model_info_matches_metadata(client: TestClient) -> None:
    response = client.get(f"{API}/model-info")
    assert response.status_code == 200
    body = response.json()
    stored = json.loads(
        (SETTINGS.models_dir / "metadata.json").read_text(encoding="utf-8")
    )
    assert body["dataset"] == "HDFS_v1"
    assert body["model_version"] == stored["model_version"] == "2.0.0"
    assert body["feature_count"] == stored["feature_count"] == 35
    assert len(body["feature_names"]) == 35
    assert body["feature_names"] == stored["feature_names"]
    assert body["trained_at"]
    assert body["selected_model"] == stored["selected_model"] == "xgboost"
    metrics = body["evaluation_metrics"]
    assert metrics == pytest.approx(stored["evaluation_metrics"])
    assert "anomaly_detector" in body


# ---- POST /api/analyze ----


def test_analyze_normal_events_are_low_risk(client: TestClient) -> None:
    payload = {"block_id": "api-test-normal", "events": _demo_events("normal")}
    response = client.post(f"{API}/analyze", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["block_id"] == "api-test-normal"
    assert body["dataset"] == "HDFS_v1"
    assert len(body["stages"]) == 10
    assert body["incident_intelligence"]["anomaly_detected"] is False
    assert body["composite_risk_score"]["severity"] == "LOW"
    assert body["blast_radius_prediction"]["status"] == "skipped"
    assert body["dependency_graph_mapping"]["status"] == "skipped_no_anomaly"


def test_analyze_anomalous_events_show_full_incident(client: TestClient) -> None:
    payload = {
        "block_id": "api-test-anomalous",
        "events": _demo_events("anomalous"),
    }
    response = client.post(f"{API}/analyze", json=payload)
    assert response.status_code == 200
    body = response.json()
    incident = body["incident_intelligence"]
    assert incident["anomaly_detected"] is True
    assert body["composite_risk_score"]["severity"] in {"HIGH", "CRITICAL"}
    assert incident["top_evidence"]
    assert incident["probable_component"] != "none (no anomaly detected)"
    mapping = body["dependency_graph_mapping"]
    assert mapping["status"] == "mapped"
    blast = body["blast_radius_prediction"]
    assert blast is not None
    assert blast["probable_origin"]["id"] == mapping["service_id"]
    assert blast["services_at_risk"]
    assert blast["most_likely_propagation_path"]
    assert body["what_if_simulation"] == []


def test_analyze_generates_block_id_when_absent(client: TestClient) -> None:
    payload = {"events": _demo_events("normal")}
    response = client.post(f"{API}/analyze", json=payload)
    assert response.status_code == 200
    assert response.json()["block_id"].startswith("api-")


def test_analyze_with_what_if_simulation(client: TestClient) -> None:
    payload = {
        "block_id": "api-test-whatif",
        "events": _demo_events("anomalous"),
    }
    probe = client.post(f"{API}/analyze", json=payload).json()
    service = probe["dependency_graph_mapping"]["service_id"]
    payload["what_if"] = [
        {"action": "restart_service", "affected_service": service}
    ]
    response = client.post(f"{API}/analyze", json=payload)
    assert response.status_code == 200
    simulations = response.json()["what_if_simulation"]
    assert len(simulations) == 1
    sim = simulations[0]
    assert sim["action"] == "restart_service"
    assert sim["affected_service"] == service
    assert sim["simulated_incident_risk"] < sim["original_incident_risk"]
    assert "estimated" in sim["status"]


# ---- validation errors (structured 422 envelope) ----


def _assert_validation_envelope(response, field: str) -> dict:
    assert response.status_code == 422
    body = response.json()
    assert body["error"]["type"] == "validation_error"
    assert body["error"]["message"] == "request validation failed"
    locations = [d["loc"] for d in body["error"]["details"]]
    assert any(field in loc for loc in locations)
    assert all(set(d) >= {"loc", "msg", "type"} for d in body["error"]["details"])
    return body


def test_analyze_missing_events_is_422(client: TestClient) -> None:
    response = client.post(f"{API}/analyze", json={"block_id": "x"})
    _assert_validation_envelope(response, "events")


def test_analyze_empty_events_is_422(client: TestClient) -> None:
    response = client.post(f"{API}/analyze", json={"events": {}})
    _assert_validation_envelope(response, "events")


def test_analyze_bad_event_key_is_422(client: TestClient) -> None:
    response = client.post(f"{API}/analyze", json={"events": {"X1": 3}})
    _assert_validation_envelope(response, "events")


def test_analyze_negative_count_is_422(client: TestClient) -> None:
    response = client.post(f"{API}/analyze", json={"events": {"E1": -1}})
    _assert_validation_envelope(response, "events")


def test_analyze_non_integer_count_is_422(client: TestClient) -> None:
    response = client.post(f"{API}/analyze", json={"events": {"E1": "abc"}})
    _assert_validation_envelope(response, "events")


def test_analyze_unseen_event_column_is_422(client: TestClient) -> None:
    response = client.post(f"{API}/analyze", json={"events": {"E99": 1}})
    assert response.status_code == 422
    assert response.json()["error"]["type"] == "unprocessable_entity"


def test_analyze_unknown_what_if_service_is_422(client: TestClient) -> None:
    payload = {
        "events": _demo_events("normal"),
        "what_if": [
            {"action": "restart_service", "affected_service": "no_such_service"}
        ],
    }
    response = client.post(f"{API}/analyze", json=payload)
    assert response.status_code == 422
    assert response.json()["error"]["type"] == "unprocessable_entity"


def test_analyze_top_k_out_of_range_is_422(client: TestClient) -> None:
    response = client.post(
        f"{API}/analyze", json={"events": _demo_events("normal"), "top_k": 11}
    )
    _assert_validation_envelope(response, "top_k")


def test_analyze_too_many_what_ifs_is_422(client: TestClient) -> None:
    payload = {
        "events": _demo_events("normal"),
        "what_if": [
            {"action": "restart_service", "affected_service": "inventory"}
        ]
        * 6,
    }
    response = client.post(f"{API}/analyze", json=payload)
    _assert_validation_envelope(response, "what_if")


# ---- payload limit, routing and internal errors ----


def test_oversized_body_is_413(client: TestClient) -> None:
    oversized = b'{"events": "' + b"x" * (MAX_BODY_BYTES + 1) + b'"}'
    response = client.post(
        f"{API}/analyze",
        content=oversized,
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 413
    body = response.json()
    assert body["error"]["type"] == "payload_too_large"
    assert "KB" in body["error"]["message"]


def test_unknown_route_is_structured_404(client: TestClient) -> None:
    response = client.get(f"{API}/does-not-exist")
    assert response.status_code == 404
    body = response.json()
    assert body["error"]["type"] == "not_found"
    assert body["error"]["message"]


def test_internal_error_never_leaks_stack_trace(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(*args, **kwargs):
        raise RuntimeError("secret internal detail XYZ")

    monkeypatch.setattr(client.app.state.pipeline, "run", boom)
    response = client.post(
        f"{API}/analyze", json={"events": _demo_events("normal")}
    )
    assert response.status_code == 500
    body = response.json()
    assert body["error"]["type"] == "internal_error"
    assert body["error"]["message"] == "an internal error occurred"
    assert "XYZ" not in response.text
    assert "Traceback" not in response.text


# ---- demo endpoints ----


def test_demo_normal_get(client: TestClient) -> None:
    response = client.get(f"{API}/demo/normal")
    assert response.status_code == 200
    body = response.json()
    assert body["incident_intelligence"]["anomaly_detected"] is False
    assert body["composite_risk_score"]["severity"] == "LOW"
    assert body["blast_radius_prediction"]["status"] == "skipped"


def test_demo_anomaly_get(client: TestClient) -> None:
    response = client.get(f"{API}/demo/anomaly")
    assert response.status_code == 200
    body = response.json()
    incident = body["incident_intelligence"]
    assert incident["anomaly_detected"] is True
    assert body["composite_risk_score"]["severity"] in {"HIGH", "CRITICAL"}
    assert incident["top_evidence"]
    blast = body["blast_radius_prediction"]
    assert blast and blast["services_at_risk"]


def test_demo_analyze_post(client: TestClient) -> None:
    for sample, is_anomaly in (("normal", False), ("anomalous", True)):
        response = client.post(
            f"{API}/demo/analyze", json={"sample": sample}
        )
        assert response.status_code == 200
        incident = response.json()["incident_intelligence"]
        assert incident["anomaly_detected"] is is_anomaly


def test_demo_analyze_invalid_sample_is_422(client: TestClient) -> None:
    response = client.post(
        f"{API}/demo/analyze", json={"sample": "suspicious"}
    )
    _assert_validation_envelope(response, "sample")


# ---- dependency graph ----


def test_dependency_graph_structure(client: TestClient) -> None:
    response = client.get(f"{API}/dependency-graph")
    assert response.status_code == 200
    body = response.json()
    assert body["service_count"] == 7
    assert body["dependency_count"] == 7
    assert len(body["services"]) == 7
    assert len(body["dependencies"]) == 7
    node = body["services"][0]
    for attr in (
        "anomaly_score",
        "incident_probability",
        "error_rate",
        "latency_change",
        "service_criticality",
        "current_health",
    ):
        assert attr in node
    edge = body["dependencies"][0]
    for attr in (
        "dependency_strength",
        "historical_propagation_factor",
        "latency_sensitivity",
    ):
        assert attr in edge
    ids = {service["id"] for service in body["services"]}
    assert {edge["service_id"] for edge in body["dependencies"]} <= ids
    assert {edge["depends_on"] for edge in body["dependencies"]} <= ids


# ---- blast radius ----


def test_blast_radius_valid_origin(client: TestClient) -> None:
    payload = {"origin": "inventory", "origin_risk": 0.85}
    response = client.post(f"{API}/blast-radius", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["probable_origin"]["id"] == "inventory"
    assert body["probable_origin"]["incident_risk"] == pytest.approx(0.85)
    assert body["services_at_risk"]
    assert body["most_likely_propagation_path"][0] == "inventory"
    assert body["highest_risk_service"]


def test_blast_radius_defaults_risk(client: TestClient) -> None:
    response = client.post(
        f"{API}/blast-radius", json={"origin": "inventory"}
    )
    assert response.status_code == 200
    assert response.json()["probable_origin"]["id"] == "inventory"


def test_blast_radius_unknown_origin_is_404(client: TestClient) -> None:
    response = client.post(
        f"{API}/blast-radius", json={"origin": "no_such_service"}
    )
    assert response.status_code == 404
    assert response.json()["error"]["type"] == "not_found"


def test_blast_radius_risk_out_of_range_is_422(client: TestClient) -> None:
    response = client.post(
        f"{API}/blast-radius", json={"origin": "inventory", "origin_risk": 1.5}
    )
    _assert_validation_envelope(response, "origin_risk")


# ---- recovery simulation ----


def test_simulate_recovery_restart(client: TestClient) -> None:
    payload = {
        "action": "restart_service",
        "affected_service": "api_gateway",
        "origin": "api_gateway",
        "origin_risk": 0.85,
    }
    response = client.post(f"{API}/simulate-recovery", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["action"] == "restart_service"
    assert body["affected_service"] == "api_gateway"
    assert body["original_incident_risk"] == 85
    assert body["simulated_incident_risk"] == 0
    assert body["estimated_risk_reduction"] == 85
    assert body["simulated_services_at_risk"] >= 0


def test_simulate_recovery_unknown_service_is_404(client: TestClient) -> None:
    payload = {
        "action": "restart_service",
        "affected_service": "no_such_service",
    }
    response = client.post(f"{API}/simulate-recovery", json=payload)
    assert response.status_code == 404
    assert response.json()["error"]["type"] == "not_found"


def test_simulate_recovery_isolate_critical_is_422(client: TestClient) -> None:
    payload = {
        "action": "isolate_noncritical_service",
        "affected_service": "database",
    }
    response = client.post(f"{API}/simulate-recovery", json=payload)
    assert response.status_code == 422
    assert response.json()["error"]["type"] == "unprocessable_entity"


def test_simulate_recovery_invalid_action_is_422(client: TestClient) -> None:
    payload = {
        "action": "sacrifice_server_to_the_old_gods",
        "affected_service": "database",
    }
    response = client.post(f"{API}/simulate-recovery", json=payload)
    _assert_validation_envelope(response, "action")
