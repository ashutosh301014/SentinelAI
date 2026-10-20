"""Tests for the What-If Recovery Simulator."""
from __future__ import annotations

import json

import networkx as nx
import pytest

import sentinelai.recovery_simulator as recovery_simulator_module
from sentinelai.config import get_settings
from sentinelai.dependency_graph import load_topology
from sentinelai.recovery_simulator import (
    RECOVERY_ACTIONS,
    apply_action,
    simulate_recovery,
)

REQUIRED_OUTPUT_FIELDS = {
    "action",
    "affected_service",
    "original_incident_risk",
    "simulated_incident_risk",
    "original_services_at_risk",
    "simulated_services_at_risk",
    "estimated_risk_reduction",
    "changed_propagation_paths",
    "before_state",
    "simulated_after_state",
}


def _node(
    health: float = 100.0,
    criticality: float = 5.0,
    incident_probability: float = 0.1,
    error_rate: float = 0.1,
    latency_change: float = 1.4,
) -> dict:
    return {
        "anomaly_score": 0.1,
        "incident_probability": incident_probability,
        "error_rate": error_rate,
        "latency_change": latency_change,
        "service_criticality": criticality,
        "current_health": health,
    }


def _make_graph(
    edges: list[tuple[str, str, float, float]],
    nodes: dict[str, dict] | None = None,
) -> nx.DiGraph:
    graph = nx.DiGraph()
    for node_id, attrs in (nodes or {}).items():
        graph.add_node(node_id, name=node_id.upper(), **attrs)
    for source, target, strength, propagation in edges:
        graph.add_edge(
            source,
            target,
            dependency_strength=strength,
            historical_propagation_factor=propagation,
            latency_sensitivity=0.5,
        )
    return graph


def _chain() -> nx.DiGraph:
    return _make_graph(
        [("a", "b", 0.8, 0.9), ("b", "c", 0.8, 0.9)],
        {"a": _node(), "b": _node(), "c": _node()},
    )


def _snapshot(graph: nx.DiGraph) -> tuple:
    return (
        {node: dict(attrs) for node, attrs in graph.nodes(data=True)},
        {(u, v): dict(attrs) for u, v, attrs in graph.edges(data=True)},
    )


# ---- terminology and output shape ----


def test_module_uses_estimated_counterfactual_impact() -> None:
    docstring = " ".join(recovery_simulator_module.__doc__.lower().split())
    assert "estimated counterfactual impact" in docstring
    assert "guaranteed recovery" not in docstring


def test_to_dict_has_all_output_fields() -> None:
    result = simulate_recovery(_chain(), "restart_service", "a")
    assert set(result.to_dict()) == REQUIRED_OUTPUT_FIELDS
    json.dumps(result.to_dict())  # serializable


def test_format_summary_matches_example_shape() -> None:
    result = simulate_recovery(_chain(), "restart_service", "a", origin_risk=0.85)
    summary = result.format_summary()
    assert summary.splitlines()[0] == "BEFORE"
    assert "Incident Risk = 85" in summary
    assert "Services At Risk = 2" in summary
    assert "ACTION" in summary
    assert "Restart Service (a)" in summary
    assert "SIMULATED AFTER" in summary
    assert "Incident Risk = 0" in summary
    assert "Estimated Risk Reduction = 85 points" in summary


def test_format_summary_display_name_for_capacity_action() -> None:
    result = simulate_recovery(
        _chain(), "increase_database_capacity", "b", origin="a"
    )
    assert "Increase Database Capacity (b)" in result.format_summary()


# ---- never mutate the original graph ----


def test_apply_action_returns_new_graph() -> None:
    graph = _chain()
    copied = apply_action(graph, "restart_service", "a")
    assert copied is not graph
    assert isinstance(copied, nx.DiGraph)


@pytest.mark.parametrize(
    ("action", "service"),
    [
        ("restart_service", "a"),
        ("reduce_error_rate", "a"),
        ("reduce_latency", "a"),
        ("improve_service_health", "a"),
        ("increase_database_capacity", "a"),
        ("isolate_noncritical_service", "c"),
    ],
)
def test_apply_action_never_mutates_input_graph(action: str, service: str) -> None:
    graph = _make_graph(
        [("a", "b", 0.8, 0.9), ("b", "c", 0.8, 0.9)],
        {"a": _node(), "b": _node(), "c": _node(criticality=2.0)},
    )
    before = _snapshot(graph)
    apply_action(graph, action, service)
    assert _snapshot(graph) == before


@pytest.mark.parametrize(
    ("action", "affected", "origin"),
    [
        ("restart_service", "a", "a"),
        ("improve_service_health", "b", "a"),
        ("isolate_noncritical_service", "c", "a"),
    ],
)
def test_simulate_recovery_never_mutates_input_graph(
    action: str, affected: str, origin: str
) -> None:
    graph = _make_graph(
        [("a", "b", 0.8, 0.9), ("b", "c", 0.8, 0.9)],
        {"a": _node(), "b": _node(), "c": _node(criticality=2.0)},
    )
    before = _snapshot(graph)
    simulate_recovery(graph, action, affected, origin=origin, origin_risk=0.8)
    assert _snapshot(graph) == before


def test_simulate_recovery_never_mutates_real_topology() -> None:
    graph = load_topology(get_settings().service_topology)
    before = _snapshot(graph)
    simulate_recovery(
        graph,
        "isolate_noncritical_service",
        "notification",
        origin="api_gateway",
        origin_risk=0.85,
    )
    simulate_recovery(
        graph, "restart_service", "api_gateway", origin_risk=0.85
    )
    assert _snapshot(graph) == before


# ---- action effects on the copied state ----


def test_restart_service_resets_service_state() -> None:
    graph = _make_graph([], {"a": _node(health=55.0, error_rate=0.4)})
    copied = apply_action(graph, "restart_service", "a")
    node = copied.nodes["a"]
    assert node["current_health"] == 100.0
    assert node["error_rate"] == 0.0
    assert node["latency_change"] == 1.0
    assert node["anomaly_score"] == 0.0
    assert node["incident_probability"] == 0.0


def test_reduce_error_rate_halves_error_and_improves_health() -> None:
    graph = _make_graph([], {"a": _node(health=70.0, error_rate=0.4)})
    copied = apply_action(graph, "reduce_error_rate", "a")
    assert copied.nodes["a"]["error_rate"] == pytest.approx(0.2)
    assert copied.nodes["a"]["current_health"] == pytest.approx(80.0)


def test_reduce_latency_moves_halfway_to_baseline() -> None:
    graph = _make_graph([], {"a": _node(health=70.0, latency_change=2.0)})
    copied = apply_action(graph, "reduce_latency", "a")
    assert copied.nodes["a"]["latency_change"] == pytest.approx(1.5)
    assert copied.nodes["a"]["current_health"] == pytest.approx(80.0)


def test_improve_service_health_raises_health_capped() -> None:
    graph = _make_graph([], {"a": _node(health=95.0)})
    copied = apply_action(graph, "improve_service_health", "a")
    assert copied.nodes["a"]["current_health"] == 100.0


def test_increase_database_capacity_effects() -> None:
    graph = _make_graph(
        [], {"a": _node(health=70.0, incident_probability=0.4)}
    )
    copied = apply_action(graph, "increase_database_capacity", "a")
    assert copied.nodes["a"]["current_health"] == 100.0
    assert copied.nodes["a"]["incident_probability"] == pytest.approx(0.2)


def test_isolate_removes_all_edges_of_service() -> None:
    graph = load_topology(get_settings().service_topology)
    copied = apply_action(graph, "isolate_noncritical_service", "notification")
    assert copied.number_of_edges() == graph.number_of_edges() - 1
    assert copied.degree("notification") == 0


# ---- action validation ----


def test_unknown_action_raises() -> None:
    with pytest.raises(ValueError, match="unknown recovery action"):
        apply_action(_chain(), "scale_up_everything", "a")


def test_unknown_service_raises() -> None:
    with pytest.raises(KeyError, match="unknown service"):
        apply_action(_chain(), "restart_service", "ghost")


def test_isolate_critical_service_raises() -> None:
    graph = _chain()  # all criticality 5.0
    with pytest.raises(ValueError, match="non-critical"):
        apply_action(graph, "isolate_noncritical_service", "a")


# ---- simulation semantics ----


def test_restart_origin_clears_incident_risk() -> None:
    result = simulate_recovery(_chain(), "restart_service", "a", origin_risk=1.0)
    assert result.original_incident_risk == 100
    assert result.simulated_incident_risk == 0
    assert result.estimated_risk_reduction == 100
    # healthy downstream services carry no residual risk -> none at risk
    assert result.original_services_at_risk == 2
    assert result.simulated_services_at_risk == 0


def test_improve_origin_health_halves_incident_risk() -> None:
    result = simulate_recovery(
        _chain(), "improve_service_health", "a", origin_risk=0.9
    )
    assert result.original_incident_risk == 90
    assert result.simulated_incident_risk == 45
    assert result.estimated_risk_reduction == 45


def test_downstream_action_keeps_origin_risk_but_reduces_service_risk() -> None:
    graph = _make_graph(
        [("a", "b", 0.8, 0.9), ("b", "c", 0.8, 0.9)],
        {"a": _node(), "b": _node(health=70.0), "c": _node()},
    )
    result = simulate_recovery(
        graph, "improve_service_health", "b", origin="a", origin_risk=0.85
    )
    assert result.original_incident_risk == 85
    assert result.simulated_incident_risk == 85
    assert result.estimated_risk_reduction == 0
    before_b = next(
        s
        for s in result.before_state["services_at_risk"]
        if s["service_id"] == "b"
    )
    after_b = next(
        s
        for s in result.simulated_after_state["services_at_risk"]
        if s["service_id"] == "b"
    )
    assert after_b["estimated_propagation_risk"] < before_b["estimated_propagation_risk"]


def test_services_at_risk_counts_only_positive_risks() -> None:
    healthy = simulate_recovery(_chain(), "restart_service", "b", origin="a", origin_risk=0.0)
    assert healthy.original_services_at_risk == 0
    risky = simulate_recovery(_chain(), "restart_service", "b", origin="a", origin_risk=1.0)
    assert risky.original_services_at_risk == 2


def test_origin_defaults_to_affected_service() -> None:
    graph = _make_graph(
        [("a", "b", 0.8, 0.9)],
        {"a": _node(), "b": _node(incident_probability=0.4)},
    )
    result = simulate_recovery(graph, "restart_service", "b")
    assert result.original_incident_risk == 40
    assert result.simulated_incident_risk == 0


def test_changed_propagation_paths_detect_dropout() -> None:
    graph = load_topology(get_settings().service_topology)
    result = simulate_recovery(
        graph,
        "isolate_noncritical_service",
        "notification",
        origin="api_gateway",
        origin_risk=0.85,
    )
    assert result.original_services_at_risk == 6
    assert result.simulated_services_at_risk == 5
    assert result.changed_propagation_paths == [
        {
            "service": "notification",
            "before_path": ["api_gateway", "orders", "notification"],
            "after_path": None,
        }
    ]


def test_changed_propagation_paths_empty_when_topology_unchanged() -> None:
    graph = load_topology(get_settings().service_topology)
    result = simulate_recovery(
        graph, "restart_service", "api_gateway", origin_risk=0.85
    )
    assert result.changed_propagation_paths == []


def test_residual_risk_of_degraded_services_is_reported() -> None:
    # restarting a healthy origin clears propagation, but already-degraded
    # downstream services keep residual risk from their own low health
    graph = load_topology(get_settings().service_topology)
    graph_copy = graph.copy()
    graph_copy.nodes["payments"]["current_health"] = 70.0
    result = simulate_recovery(
        graph_copy, "restart_service", "api_gateway", origin_risk=0.85
    )
    assert result.simulated_incident_risk == 0
    payments_after = next(
        s
        for s in result.simulated_after_state["services_at_risk"]
        if s["service_id"] == "payments"
    )
    assert payments_after["estimated_propagation_risk"] > 0.0
    assert result.simulated_services_at_risk >= 1


def test_simulation_is_deterministic() -> None:
    graph = load_topology(get_settings().service_topology)
    first = simulate_recovery(
        graph, "improve_service_health", "database", origin="api_gateway",
        origin_risk=0.85,
    )
    second = simulate_recovery(
        graph, "improve_service_health", "database", origin="api_gateway",
        origin_risk=0.85,
    )
    assert first.to_dict() == second.to_dict()


def test_all_actions_are_simulatable_on_real_topology() -> None:
    graph = load_topology(get_settings().service_topology)
    for action in sorted(RECOVERY_ACTIONS):
        service = (
            "notification"
            if action == "isolate_noncritical_service"
            else "api_gateway"
        )
        result = simulate_recovery(
            graph, action, service, origin="api_gateway", origin_risk=0.85
        )
        assert result.estimated_risk_reduction >= 0
        assert result.estimated_risk_reduction <= 100
