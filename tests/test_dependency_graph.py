"""Tests for the service-dependency intelligence layer."""
from __future__ import annotations

import json
from pathlib import Path

import networkx as nx
import pytest

import sentinelai.dependency_graph as dependency_graph_module
from sentinelai.config import get_settings
from sentinelai.dependency_graph import (
    EDGE_ATTRIBUTES,
    NODE_ATTRIBUTES,
    criticality_ranking,
    dependency_traversal,
    downstream_dependents,
    load_topology,
    service_criticality,
    shortest_path,
    upstream_dependencies,
    validate_graph,
)

SERVICE_IDS = {
    "api_gateway",
    "authentication",
    "orders",
    "payments",
    "inventory",
    "database",
    "notification",
}

EXPECTED_EDGES = {
    ("api_gateway", "authentication"),
    ("api_gateway", "orders"),
    ("orders", "payments"),
    ("orders", "inventory"),
    ("payments", "database"),
    ("inventory", "database"),
    ("orders", "notification"),
}


@pytest.fixture(scope="module")
def topology() -> nx.DiGraph:
    return load_topology(get_settings().service_topology)


def _toy_graph() -> nx.DiGraph:
    graph = nx.DiGraph()
    for node in ("a", "b"):
        graph.add_node(
            node,
            anomaly_score=0.1,
            incident_probability=0.1,
            error_rate=0.01,
            latency_change=1.0,
            service_criticality=3.0,
            current_health=90.0,
        )
    graph.add_edge(
        "a",
        "b",
        dependency_strength=0.5,
        historical_propagation_factor=0.4,
        latency_sensitivity=0.6,
    )
    return graph


def _service(service_id: str, **overrides: float) -> dict:
    attrs: dict = {
        "anomaly_score": 0.1,
        "incident_probability": 0.1,
        "error_rate": 0.01,
        "latency_change": 1.0,
        "service_criticality": 3.0,
        "current_health": 90.0,
    }
    attrs.update(overrides)
    return {"id": service_id, **attrs}


def _dependency(source: str, target: str, **overrides: float) -> dict:
    attrs: dict = {
        "dependency_strength": 0.5,
        "historical_propagation_factor": 0.4,
        "latency_sensitivity": 0.6,
    }
    attrs.update(overrides)
    return {"source": source, "target": target, **attrs}


def _write_topology(
    tmp_path: Path, services: list[dict], dependencies: list[dict]
) -> Path:
    path = tmp_path / "topology.json"
    path.write_text(
        json.dumps({"services": services, "dependencies": dependencies}),
        encoding="utf-8",
    )
    return path


# ---- structure of the real data-driven topology ----


def test_settings_point_to_existing_topology_file() -> None:
    assert get_settings().service_topology.is_file()


def test_load_topology_builds_expected_graph(topology: nx.DiGraph) -> None:
    assert isinstance(topology, nx.DiGraph)
    assert set(topology.nodes) == SERVICE_IDS
    assert set(topology.edges) == EXPECTED_EDGES


def test_every_node_has_required_attributes(topology: nx.DiGraph) -> None:
    for node in topology.nodes:
        assert set(NODE_ATTRIBUTES) <= set(topology.nodes[node]), node


def test_every_edge_has_required_attributes(topology: nx.DiGraph) -> None:
    for edge in topology.edges:
        assert set(EDGE_ATTRIBUTES) <= set(topology.edges[edge]), edge


def test_demo_topology_validates_cleanly(topology: nx.DiGraph) -> None:
    assert validate_graph(topology) == []


def test_module_source_contains_no_hardcoded_topology() -> None:
    source = Path(dependency_graph_module.__file__).read_text(
        encoding="utf-8"
    )
    for service_id in SERVICE_IDS:
        assert service_id not in source, (
            f"topology must be data-driven: '{service_id}' hardcoded "
            "in dependency_graph.py"
        )


# ---- validation ----


def test_validate_rejects_non_digraph() -> None:
    assert validate_graph(nx.Graph()) == ["graph must be a networkx.DiGraph"]


def test_validate_flags_empty_graph() -> None:
    assert "graph has no services" in validate_graph(nx.DiGraph())


@pytest.mark.parametrize("attr", NODE_ATTRIBUTES)
def test_validate_flags_missing_node_attribute(attr: str) -> None:
    graph = _toy_graph()
    del graph.nodes["a"][attr]
    issues = validate_graph(graph)
    assert any(attr in issue and "missing" in issue for issue in issues)


@pytest.mark.parametrize("attr", EDGE_ATTRIBUTES)
def test_validate_flags_missing_edge_attribute(attr: str) -> None:
    graph = _toy_graph()
    del graph.edges["a", "b"][attr]
    issues = validate_graph(graph)
    assert any(attr in issue and "missing" in issue for issue in issues)


@pytest.mark.parametrize(
    ("attr", "value"),
    [
        ("anomaly_score", 1.5),
        ("incident_probability", -0.1),
        ("error_rate", 2.0),
        ("latency_change", -1.0),
        ("service_criticality", 6.0),
        ("current_health", 150.0),
    ],
)
def test_validate_flags_out_of_range_node_attributes(
    attr: str, value: float
) -> None:
    graph = _toy_graph()
    graph.nodes["a"][attr] = value
    issues = validate_graph(graph)
    assert any(attr in issue and "outside" in issue for issue in issues)


@pytest.mark.parametrize(
    ("attr", "value"),
    [
        ("dependency_strength", 1.2),
        ("historical_propagation_factor", -0.5),
        ("latency_sensitivity", 2.0),
    ],
)
def test_validate_flags_out_of_range_edge_attributes(
    attr: str, value: float
) -> None:
    graph = _toy_graph()
    graph.edges["a", "b"][attr] = value
    issues = validate_graph(graph)
    assert any(attr in issue and "outside" in issue for issue in issues)


@pytest.mark.parametrize("bad", [True, "0.5", None, [0.5]])
def test_validate_flags_non_numeric_node_attribute(bad) -> None:
    graph = _toy_graph()
    graph.nodes["a"]["anomaly_score"] = bad
    issues = validate_graph(graph)
    assert any("must be a number" in issue for issue in issues)


def test_validate_flags_self_loop() -> None:
    graph = _toy_graph()
    graph.add_edge(
        "a",
        "a",
        dependency_strength=0.5,
        historical_propagation_factor=0.4,
        latency_sensitivity=0.6,
    )
    issues = validate_graph(graph)
    assert any("self-loop" in issue for issue in issues)


# ---- malformed topology files ----


def test_load_raises_on_duplicate_service_id(tmp_path: Path) -> None:
    path = _write_topology(
        tmp_path, [_service("a"), _service("a")], [_dependency("a", "b")]
    )
    with pytest.raises(ValueError, match="duplicate service id"):
        load_topology(path)


def test_load_raises_on_unknown_endpoint(tmp_path: Path) -> None:
    path = _write_topology(
        tmp_path, [_service("a"), _service("b")], [_dependency("a", "ghost")]
    )
    with pytest.raises(ValueError, match="unknown service"):
        load_topology(path)


def test_load_raises_on_duplicate_dependency(tmp_path: Path) -> None:
    path = _write_topology(
        tmp_path,
        [_service("a"), _service("b")],
        [_dependency("a", "b"), _dependency("a", "b")],
    )
    with pytest.raises(ValueError, match="duplicate dependency"):
        load_topology(path)


def test_load_raises_on_invalid_attribute(tmp_path: Path) -> None:
    path = _write_topology(
        tmp_path,
        [_service("a", anomaly_score=7.0), _service("b")],
        [_dependency("a", "b")],
    )
    with pytest.raises(ValueError, match="invalid service topology"):
        load_topology(path)


def test_load_raises_on_missing_attribute(tmp_path: Path) -> None:
    service = _service("a")
    del service["error_rate"]
    path = _write_topology(tmp_path, [service, _service("b")], [])
    with pytest.raises(ValueError, match="invalid service topology"):
        load_topology(path)


def test_load_raises_on_non_object_file(tmp_path: Path) -> None:
    path = tmp_path / "topology.json"
    path.write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object"):
        load_topology(path)


def test_load_raises_on_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_topology(tmp_path / "nope.json")


# ---- upstream / downstream retrieval ----


def test_upstream_dependencies_direct(topology: nx.DiGraph) -> None:
    assert upstream_dependencies(topology, "orders") == {"api_gateway"}
    assert upstream_dependencies(topology, "database") == {
        "payments",
        "inventory",
    }
    assert upstream_dependencies(topology, "api_gateway") == set()


def test_upstream_dependencies_transitive(topology: nx.DiGraph) -> None:
    assert upstream_dependencies(topology, "database", transitive=True) == {
        "api_gateway",
        "orders",
        "payments",
        "inventory",
    }
    assert upstream_dependencies(
        topology, "authentication", transitive=True
    ) == {"api_gateway"}
    assert (
        upstream_dependencies(topology, "api_gateway", transitive=True)
        == set()
    )


def test_downstream_dependents_direct(topology: nx.DiGraph) -> None:
    assert downstream_dependents(topology, "api_gateway") == {
        "authentication",
        "orders",
    }
    assert downstream_dependents(topology, "orders") == {
        "payments",
        "inventory",
        "notification",
    }
    assert downstream_dependents(topology, "notification") == set()


def test_downstream_dependents_transitive(topology: nx.DiGraph) -> None:
    assert downstream_dependents(
        topology, "api_gateway", transitive=True
    ) == SERVICE_IDS - {"api_gateway"}
    assert (
        downstream_dependents(topology, "database", transitive=True) == set()
    )


@pytest.mark.parametrize(
    "function",
    [
        upstream_dependencies,
        downstream_dependents,
        service_criticality,
        dependency_traversal,
    ],
)
def test_unknown_service_raises(function) -> None:
    graph = _toy_graph()
    with pytest.raises(KeyError, match="unknown service"):
        function(graph, "ghost")


def test_shortest_path_unknown_service_raises(topology: nx.DiGraph) -> None:
    with pytest.raises(KeyError, match="unknown service"):
        shortest_path(topology, "ghost", "orders")


# ---- shortest paths ----


def test_shortest_path_gateway_to_database(topology: nx.DiGraph) -> None:
    path = shortest_path(topology, "api_gateway", "database")
    assert path is not None
    assert path[0] == "api_gateway"
    assert path[-1] == "database"
    assert len(path) == 4  # two equal-length 3-hop routes
    assert all(
        (path[i], path[i + 1]) in topology.edges
        for i in range(len(path) - 1)
    )


def test_shortest_path_single_hop(topology: nx.DiGraph) -> None:
    assert shortest_path(topology, "orders", "notification") == [
        "orders",
        "notification",
    ]


def test_shortest_path_no_route_returns_none(topology: nx.DiGraph) -> None:
    assert shortest_path(topology, "authentication", "notification") is None


def test_shortest_path_to_self(topology: nx.DiGraph) -> None:
    assert shortest_path(topology, "payments", "payments") == ["payments"]


# ---- service criticality ----


def test_service_criticality_reads_attribute(topology: nx.DiGraph) -> None:
    assert service_criticality(topology, "api_gateway") == pytest.approx(5.0)
    assert service_criticality(topology, "notification") == pytest.approx(2.0)


def test_criticality_ranking_sorted_descending(topology: nx.DiGraph) -> None:
    ranking = criticality_ranking(topology)
    assert len(ranking) == len(SERVICE_IDS)
    assert {service for service, _ in ranking} == SERVICE_IDS
    scores = [score for _, score in ranking]
    assert scores == sorted(scores, reverse=True)
    assert ranking[0][1] == pytest.approx(5.0)
    assert ranking[-1] == ("notification", 2.0)


# ---- dependency traversal ----


def test_traversal_bfs_downstream_covers_all(topology: nx.DiGraph) -> None:
    order = dependency_traversal(topology, "api_gateway")
    assert order[0] == "api_gateway"
    assert set(order) == SERVICE_IDS
    assert len(order) == len(SERVICE_IDS)


def test_traversal_bfs_downstream_exact_order(topology: nx.DiGraph) -> None:
    assert dependency_traversal(topology, "api_gateway") == [
        "api_gateway",
        "authentication",
        "orders",
        "payments",
        "inventory",
        "notification",
        "database",
    ]


def test_traversal_dfs_downstream_exact_order(topology: nx.DiGraph) -> None:
    assert dependency_traversal(
        topology, "api_gateway", order="dfs"
    ) == [
        "api_gateway",
        "orders",
        "notification",
        "inventory",
        "database",
        "payments",
        "authentication",
    ]


def test_traversal_bfs_upstream_from_database(topology: nx.DiGraph) -> None:
    assert dependency_traversal(
        topology, "database", direction="upstream"
    ) == ["database", "payments", "inventory", "orders", "api_gateway"]


def test_traversal_from_leaf(topology: nx.DiGraph) -> None:
    assert dependency_traversal(topology, "notification") == ["notification"]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"direction": "sideways"}, "direction"),
        ({"order": "random"}, "order"),
    ],
)
def test_traversal_invalid_arguments_raise(
    topology: nx.DiGraph, kwargs: dict, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        dependency_traversal(topology, "api_gateway", **kwargs)
