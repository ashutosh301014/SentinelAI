"""Tests for the Predictive Blast-Radius Engine."""
from __future__ import annotations

import json

import networkx as nx
import pytest

import sentinelai.blast_radius as blast_radius_module
from sentinelai.blast_radius import (
    AffectedService,
    BlastRadiusResult,
    estimate_blast_radius,
)
from sentinelai.config import get_settings
from sentinelai.dependency_graph import load_topology

REQUIRED_SERVICE_FIELDS = {
    "service_id",
    "service_name",
    "estimated_propagation_risk",
    "dependency_distance",
    "risk_level",
    "probable_propagation_path",
    "relative_impact_order",
}


def _node(
    health: float = 100.0,
    criticality: float = 5.0,
    incident_probability: float = 0.1,
) -> dict:
    return {
        "anomaly_score": 0.1,
        "incident_probability": incident_probability,
        "error_rate": 0.01,
        "latency_change": 1.0,
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


def _chain(strength: float = 0.8, propagation: float = 0.9) -> nx.DiGraph:
    return _make_graph(
        [("a", "b", strength, propagation), ("b", "c", strength, propagation)],
        {"a": _node(), "b": _node(), "c": _node()},
    )


# ---- result shape ----


def test_result_dataclass_fields() -> None:
    result = estimate_blast_radius(_chain(), "a", 1.0)
    assert isinstance(result, BlastRadiusResult)
    assert result.origin == "a"
    assert result.origin_risk == pytest.approx(1.0)
    assert result.services_at_risk
    for service in result.services_at_risk:
        assert isinstance(service, AffectedService)
        assert set(service.to_dict()) == REQUIRED_SERVICE_FIELDS


def test_result_to_dict_is_json_serializable() -> None:
    result = estimate_blast_radius(_chain(), "a", 0.7)
    payload = json.loads(json.dumps(result.to_dict()))
    assert set(payload) == {
        "probable_origin",
        "services_at_risk",
        "most_likely_propagation_path",
        "highest_risk_service",
    }


# ---- direct dependencies ----


def test_direct_dependency_risk_formula() -> None:
    graph = _make_graph(
        [("a", "b", 0.5, 0.8)], {"a": _node(), "b": _node()}
    )
    result = estimate_blast_radius(graph, "a", 0.8)
    (service,) = result.services_at_risk
    # healthy (vulnerability 0), criticality 5 (factor 1.0):
    # risk = 0.8 * 0.5 * 0.8
    assert service.estimated_propagation_risk == pytest.approx(0.32)
    assert service.dependency_distance == 1
    assert service.probable_propagation_path == ["a", "b"]
    assert service.risk_level == "MEDIUM"


def test_direct_dependency_weak_edge_low_risk() -> None:
    graph = _make_graph(
        [("a", "b", 0.1, 0.1)], {"a": _node(), "b": _node()}
    )
    result = estimate_blast_radius(graph, "a", 1.0)
    (service,) = result.services_at_risk
    assert service.estimated_propagation_risk == pytest.approx(0.01)
    assert service.risk_level == "LOW"


# ---- multi-hop propagation and decay ----


def test_multi_hop_risk_decays_with_distance() -> None:
    graph = _chain(0.8, 0.9)
    result = estimate_blast_radius(graph, "a", 1.0)
    by_id = {service.service_id: service for service in result.services_at_risk}
    risk_b = by_id["b"].estimated_propagation_risk  # 1 * .8 * .9 = .72
    risk_c = by_id["c"].estimated_propagation_risk  # .72 * .8 * .9 = .5184
    assert risk_b == pytest.approx(0.72)
    assert risk_c == pytest.approx(0.5184)
    assert risk_c < risk_b < 1.0
    assert by_id["b"].dependency_distance == 1
    assert by_id["c"].dependency_distance == 2
    assert by_id["c"].probable_propagation_path == ["a", "b", "c"]
    # most likely path leads to the highest-risk service (b, not the deepest)
    assert result.most_likely_propagation_path == ["a", "b"]
    assert result.highest_risk_service == "b"


# ---- disconnected services ----


def test_disconnected_service_is_not_affected() -> None:
    graph = _make_graph(
        [("a", "b", 0.8, 0.9)],
        {"a": _node(), "b": _node(), "x": _node()},
    )
    result = estimate_blast_radius(graph, "a", 0.5)
    assert {s.service_id for s in result.services_at_risk} == {"b"}


# ---- healthy vs high-risk origin ----


def test_healthy_origin_propagates_zero_risk() -> None:
    result = estimate_blast_radius(_chain(), "a", 0.0)
    assert len(result.services_at_risk) == 2
    assert all(
        service.estimated_propagation_risk == 0.0
        for service in result.services_at_risk
    )
    assert all(
        service.risk_level == "LOW" for service in result.services_at_risk
    )
    # zero-risk paths still trace back to the origin
    assert all(
        service.probable_propagation_path[0] == "a"
        for service in result.services_at_risk
    )


def test_high_risk_origin_strong_edge_reaches_critical() -> None:
    graph = _make_graph(
        [("a", "b", 1.0, 1.0)], {"a": _node(), "b": _node()}
    )
    result = estimate_blast_radius(graph, "a", 1.0)
    (service,) = result.services_at_risk
    assert service.estimated_propagation_risk == pytest.approx(1.0)
    assert service.risk_level == "CRITICAL"


# ---- cyclic topology protection ----


def test_cyclic_topology_terminates_and_ignores_back_edges() -> None:
    graph = _make_graph(
        [
            ("a", "b", 0.8, 0.9),
            ("b", "c", 0.8, 0.9),
            ("c", "b", 0.5, 0.5),  # back edge -> must be ignored
        ],
        {"a": _node(), "b": _node(), "c": _node()},
    )
    result = estimate_blast_radius(graph, "a", 1.0)
    by_id = {s.service_id: s for s in result.services_at_risk}
    assert by_id["b"].estimated_propagation_risk == pytest.approx(0.72)
    assert by_id["c"].estimated_propagation_risk == pytest.approx(0.5184)
    assert by_id["b"].dependency_distance == 1
    assert by_id["c"].dependency_distance == 2


def test_self_loop_rejected_by_topology_validation() -> None:
    graph = _make_graph(
        [("a", "a", 1.0, 1.0), ("a", "b", 0.5, 0.5)],
        {"a": _node(), "b": _node()},
    )
    with pytest.raises(ValueError, match="self-loop"):
        estimate_blast_radius(graph, "a", 1.0)


def test_cyclic_topology_is_deterministic() -> None:
    graph = _make_graph(
        [
            ("a", "b", 0.8, 0.9),
            ("b", "c", 0.8, 0.9),
            ("c", "b", 0.5, 0.5),
        ],
        {"a": _node(), "b": _node(), "c": _node()},
    )
    first = estimate_blast_radius(graph, "a", 1.0)
    second = estimate_blast_radius(graph, "a", 1.0)
    assert first.to_dict() == second.to_dict()


# ---- health and criticality modulation ----


def test_degraded_health_raises_propagation_risk() -> None:
    healthy = _make_graph(
        [("a", "b", 0.5, 0.5)], {"a": _node(), "b": _node(health=100.0)}
    )
    degraded = _make_graph(
        [("a", "b", 0.5, 0.5)], {"a": _node(), "b": _node(health=40.0)}
    )
    risk_healthy = estimate_blast_radius(healthy, "a", 1.0).services_at_risk[0]
    risk_degraded = estimate_blast_radius(degraded, "a", 1.0).services_at_risk[0]
    assert risk_healthy.estimated_propagation_risk == pytest.approx(0.25)
    # 0.25 + (1 - 0.25) * 0.6 = 0.70
    assert risk_degraded.estimated_propagation_risk == pytest.approx(0.70)
    assert risk_degraded.estimated_propagation_risk > risk_healthy.estimated_propagation_risk


def test_criticality_scales_propagation_risk() -> None:
    high_crit = _make_graph(
        [("a", "b", 0.5, 0.5)],
        {"a": _node(), "b": _node(criticality=5.0)},
    )
    low_crit = _make_graph(
        [("a", "b", 0.5, 0.5)],
        {"a": _node(), "b": _node(criticality=1.0)},
    )
    risk_high = estimate_blast_radius(high_crit, "a", 1.0).services_at_risk[0]
    risk_low = estimate_blast_radius(low_crit, "a", 1.0).services_at_risk[0]
    assert risk_high.estimated_propagation_risk == pytest.approx(0.25)
    # criticality_factor(1.0) = 0.6 + 0.4 * 0.2 = 0.68 -> 0.25 * 0.68
    assert risk_low.estimated_propagation_risk == pytest.approx(0.17)
    assert risk_high.estimated_propagation_risk > risk_low.estimated_propagation_risk


# ---- multiple paths ----


def test_multiple_paths_strongest_wins() -> None:
    graph = _make_graph(
        [
            ("a", "b", 0.5, 1.0),
            ("b", "d", 0.9, 1.0),
            ("a", "c", 0.9, 1.0),
            ("c", "d", 0.2, 1.0),
        ],
        {"a": _node(), "b": _node(), "c": _node(), "d": _node()},
    )
    result = estimate_blast_radius(graph, "a", 1.0)
    d = next(s for s in result.services_at_risk if s.service_id == "d")
    # via b: 0.5 * 0.9 = 0.45 ; via c: 0.9 * 0.2 = 0.18 -> strongest wins
    assert d.estimated_propagation_risk == pytest.approx(0.45)
    assert d.probable_propagation_path == ["a", "b", "d"]
    assert d.dependency_distance == 2


# ---- ranking / impact order ----


def test_relative_impact_order_follows_risk_ranking() -> None:
    graph = _make_graph(
        [
            ("a", "b", 0.9, 1.0),
            ("a", "c", 0.2, 1.0),
        ],
        {"a": _node(), "b": _node(), "c": _node()},
    )
    result = estimate_blast_radius(graph, "a", 1.0)
    assert [s.service_id for s in result.services_at_risk] == ["b", "c"]
    assert [s.relative_impact_order for s in result.services_at_risk] == [1, 2]
    assert result.highest_risk_service == "b"


# ---- risk levels ----


@pytest.mark.parametrize(
    ("strength", "expected_level"),
    [
        (0.1, "LOW"),
        (0.3, "MEDIUM"),
        (0.6, "HIGH"),
        (0.85, "CRITICAL"),
    ],
)
def test_risk_level_bands(strength: float, expected_level: str) -> None:
    graph = _make_graph(
        [("a", "b", strength, 1.0)], {"a": _node(), "b": _node()}
    )
    result = estimate_blast_radius(graph, "a", 1.0)
    assert result.services_at_risk[0].risk_level == expected_level


# ---- edge cases ----


def test_leaf_origin_has_empty_blast_radius() -> None:
    graph = _make_graph([], {"a": _node(), "b": _node()})
    result = estimate_blast_radius(graph, "a", 0.9)
    assert result.services_at_risk == []
    assert result.highest_risk_service is None
    assert result.most_likely_propagation_path == []


def test_origin_risk_defaults_to_incident_probability() -> None:
    graph = _make_graph(
        [("a", "b", 0.5, 1.0)],
        {"a": _node(incident_probability=0.4), "b": _node()},
    )
    result = estimate_blast_radius(graph, "a")
    assert result.origin_risk == pytest.approx(0.4)
    assert result.services_at_risk[0].estimated_propagation_risk == pytest.approx(0.2)


@pytest.mark.parametrize("bad", [-0.1, 1.1, "0.5", True])
def test_invalid_origin_risk_raises(bad) -> None:
    with pytest.raises(ValueError, match="origin_risk"):
        estimate_blast_radius(_chain(), "a", bad)


def test_unknown_origin_raises() -> None:
    with pytest.raises(KeyError, match="unknown service"):
        estimate_blast_radius(_chain(), "ghost", 0.5)


def test_invalid_graph_raises() -> None:
    graph = nx.DiGraph()
    graph.add_node("a")
    with pytest.raises(ValueError, match="invalid dependency graph"):
        estimate_blast_radius(graph, "a", 0.5)


# ---- terminology guard ----


def test_module_disclaims_calibrated_probability() -> None:
    docstring = " ".join(blast_radius_module.__doc__.lower().split())
    assert "not a guaranteed or calibrated failure probability" in docstring


# ---- real topology integration ----


@pytest.fixture(scope="module")
def real_topology() -> nx.DiGraph:
    return load_topology(get_settings().service_topology)


def test_real_topology_distances(real_topology: nx.DiGraph) -> None:
    result = estimate_blast_radius(real_topology, "api_gateway", 0.85)
    by_id = {s.service_id: s for s in result.services_at_risk}
    assert set(by_id) == {
        "authentication",
        "orders",
        "payments",
        "inventory",
        "notification",
        "database",
    }
    assert by_id["authentication"].dependency_distance == 1
    assert by_id["orders"].dependency_distance == 1
    assert by_id["payments"].dependency_distance == 2
    assert by_id["inventory"].dependency_distance == 2
    assert by_id["notification"].dependency_distance == 2
    assert by_id["database"].dependency_distance == 3
    for service in result.services_at_risk:
        assert service.probable_propagation_path[0] == "api_gateway"
        assert (
            service.probable_propagation_path[-1] == service.service_id
        )


def test_real_topology_sink_origin_has_no_downstream(
    real_topology: nx.DiGraph,
) -> None:
    result = estimate_blast_radius(real_topology, "database", 0.9)
    assert result.services_at_risk == []
    assert result.highest_risk_service is None


def test_real_topology_risk_decays_with_distance(
    real_topology: nx.DiGraph,
) -> None:
    result = estimate_blast_radius(real_topology, "api_gateway", 0.85)
    orders = next(
        s for s in result.services_at_risk if s.service_id == "orders"
    )
    notification = next(
        s for s in result.services_at_risk if s.service_id == "notification"
    )
    # same-origin pressure decays across the 2-hop route orders -> notification
    assert notification.estimated_propagation_risk < orders.estimated_propagation_risk
