"""Service dependency intelligence for SentinelAI, built on NetworkX.

The topology is fully data-driven: services and dependency edges are loaded
from a JSON topology file (see ``data/config/service_topology.json``) rather
than hardcoded in this module.

Edge direction semantics: ``u -> v`` means "u depends on v" — ``v`` is a
dependency (callee) of ``u`` and ``u`` is a dependent (caller) of ``v``.

Traversal convention (used by :func:`dependency_traversal` and the
blast-radius engine): following edges forward (successors, i.e. moving
from a service toward the services it depends on) is "downstream";
following edges backward (predecessors, i.e. moving toward the callers
that depend on a service) is "upstream". Accordingly,
:func:`upstream_dependencies` returns the direct callers of a service
and :func:`downstream_dependents` returns the services a service
directly depends on.

Attribute scales (enforced by :func:`validate_graph`):
  node: anomaly_score [0, 1], incident_probability [0, 1], error_rate [0, 1],
        latency_change >= 0 (multiplicative, 1.0 = no change),
        service_criticality [0, 5], current_health [0, 100]
  edge: dependency_strength [0, 1], historical_propagation_factor [0, 1],
        latency_sensitivity [0, 1]
"""
from __future__ import annotations

import json
from collections import deque
from pathlib import Path
from typing import Any

import networkx as nx

NODE_ATTRIBUTES: tuple[str, ...] = (
    "anomaly_score",
    "incident_probability",
    "error_rate",
    "latency_change",
    "service_criticality",
    "current_health",
)

EDGE_ATTRIBUTES: tuple[str, ...] = (
    "dependency_strength",
    "historical_propagation_factor",
    "latency_sensitivity",
)

_NODE_RANGES: dict[str, tuple[float, float | None]] = {
    "anomaly_score": (0.0, 1.0),
    "incident_probability": (0.0, 1.0),
    "error_rate": (0.0, 1.0),
    "latency_change": (0.0, None),
    "service_criticality": (0.0, 5.0),
    "current_health": (0.0, 100.0),
}

_EDGE_RANGES: dict[str, tuple[float, float]] = {
    "dependency_strength": (0.0, 1.0),
    "historical_propagation_factor": (0.0, 1.0),
    "latency_sensitivity": (0.0, 1.0),
}


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _require_service(graph: nx.DiGraph, service: str) -> None:
    if service not in graph.nodes:
        raise KeyError(f"unknown service: {service}")


def validate_graph(graph: nx.DiGraph) -> list[str]:
    """Validate a dependency graph; returns a list of human-readable issues.

    An empty list means the graph is valid.
    """
    issues: list[str] = []
    if not isinstance(graph, nx.DiGraph):
        return ["graph must be a networkx.DiGraph"]
    if graph.number_of_nodes() == 0:
        issues.append("graph has no services")

    for node in graph.nodes:
        for attr in NODE_ATTRIBUTES:
            if attr not in graph.nodes[node]:
                issues.append(
                    f"service '{node}' is missing attribute '{attr}'"
                )
                continue
            value = graph.nodes[node][attr]
            if not _is_number(value):
                issues.append(
                    f"service '{node}' attribute '{attr}' must be a number, "
                    f"got {value!r}"
                )
                continue
            low, high = _NODE_RANGES[attr]
            bound = f"[{low}, {high}]" if high is not None else f">= {low}"
            if value < low or (high is not None and value > high):
                issues.append(
                    f"service '{node}' attribute '{attr}' = {value} "
                    f"is outside {bound}"
                )

    for source, target in graph.edges:
        if source == target:
            issues.append(f"self-loop on service '{source}'")
        for attr in EDGE_ATTRIBUTES:
            if attr not in graph.edges[source, target]:
                issues.append(
                    f"dependency {source} -> {target} "
                    f"is missing attribute '{attr}'"
                )
                continue
            value = graph.edges[source, target][attr]
            if not _is_number(value):
                issues.append(
                    f"dependency {source} -> {target} attribute '{attr}' "
                    f"must be a number, got {value!r}"
                )
                continue
            low, high = _EDGE_RANGES[attr]
            if not low <= value <= high:
                issues.append(
                    f"dependency {source} -> {target} attribute '{attr}' "
                    f"= {value} is outside [{low}, {high}]"
                )
    return issues


def load_topology(path: str | Path) -> nx.DiGraph:
    """Load a service topology from a JSON file and validate it.

    Raises ``ValueError`` for structural problems (duplicate services or
    dependencies, unknown endpoints) or attribute violations; raises
    ``FileNotFoundError`` if the file does not exist.
    """
    topology = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(topology, dict):
        raise ValueError("topology file must contain a JSON object")
    services = topology.get("services")
    dependencies = topology.get("dependencies")
    if not isinstance(services, list) or not isinstance(dependencies, list):
        raise ValueError(
            "topology must define 'services' and 'dependencies' lists"
        )

    graph = nx.DiGraph()
    service_ids: set[str] = set()
    for service in services:
        if not isinstance(service, dict) or "id" not in service:
            raise ValueError("every service must be an object with an 'id'")
        service_id = service["id"]
        if service_id in service_ids:
            raise ValueError(f"duplicate service id '{service_id}'")
        service_ids.add(service_id)
        graph.add_node(
            service_id,
            **{key: value for key, value in service.items() if key != "id"},
        )

    seen_edges: set[tuple[str, str]] = set()
    for dependency in dependencies:
        if not isinstance(dependency, dict):
            raise ValueError("every dependency must be an object")
        source = dependency.get("source")
        target = dependency.get("target")
        if source not in service_ids or target not in service_ids:
            raise ValueError(
                f"dependency {source} -> {target} references an unknown service"
            )
        if (source, target) in seen_edges:
            raise ValueError(f"duplicate dependency {source} -> {target}")
        seen_edges.add((source, target))
        graph.add_edge(
            source,
            target,
            **{
                key: value
                for key, value in dependency.items()
                if key not in ("source", "target")
            },
        )

    issues = validate_graph(graph)
    if issues:
        raise ValueError("invalid service topology: " + "; ".join(issues))
    return graph


def upstream_dependencies(
    graph: nx.DiGraph, service: str, *, transitive: bool = False
) -> set[str]:
    """Services that directly depend on ``service`` (its direct callers).

    With ``transitive=True`` returns all ancestors (everything reachable by
    walking dependency edges backward).
    """
    _require_service(graph, service)
    if transitive:
        return set(nx.ancestors(graph, service))
    return set(graph.predecessors(service))


def downstream_dependents(
    graph: nx.DiGraph, service: str, *, transitive: bool = False
) -> set[str]:
    """Services ``service`` directly depends on (its direct callees).

    With ``transitive=True`` returns all descendants (everything reachable by
    walking dependency edges forwards).
    """
    _require_service(graph, service)
    if transitive:
        return set(nx.descendants(graph, service))
    return set(graph.successors(service))


def shortest_path(
    graph: nx.DiGraph, source: str, target: str
) -> list[str] | None:
    """Shortest dependency path from ``source`` to ``target``.

    Returns the list of services along the path (inclusive of both ends), or
    ``None`` when the two services are not connected in the given direction.
    """
    _require_service(graph, source)
    _require_service(graph, target)
    try:
        return nx.shortest_path(graph, source=source, target=target)
    except nx.NetworkXNoPath:
        return None


def service_criticality(graph: nx.DiGraph, service: str) -> float:
    """Configured criticality of a single service (0-5)."""
    _require_service(graph, service)
    return float(graph.nodes[service]["service_criticality"])


def criticality_ranking(graph: nx.DiGraph) -> list[tuple[str, float]]:
    """All services ranked by criticality, most critical first."""
    return sorted(
        (
            (str(node), service_criticality(graph, str(node)))
            for node in graph.nodes
        ),
        key=lambda item: item[1],
        reverse=True,
    )


def dependency_traversal(
    graph: nx.DiGraph,
    start: str,
    *,
    direction: str = "downstream",
    order: str = "bfs",
) -> list[str]:
    """Traverse the dependency graph from ``start`` in visit order.

    ``direction``: "downstream" follows dependents (successors), "upstream"
    follows dependencies (predecessors). ``order``: "bfs" or "dfs". The start
    service is always the first element.
    """
    _require_service(graph, start)
    if direction not in ("downstream", "upstream"):
        raise ValueError(
            f"direction must be 'downstream' or 'upstream', got {direction!r}"
        )
    if order not in ("bfs", "dfs"):
        raise ValueError(f"order must be 'bfs' or 'dfs', got {order!r}")

    neighbors = (
        graph.successors if direction == "downstream" else graph.predecessors
    )
    visited: set[str] = {start}
    result: list[str] = []
    if order == "bfs":
        queue: deque[str] = deque([start])
        while queue:
            node = queue.popleft()
            result.append(node)
            for neighbor in neighbors(node):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
    else:
        stack: list[str] = [start]
        while stack:
            node = stack.pop()
            result.append(node)
            for neighbor in neighbors(node):
                if neighbor not in visited:
                    visited.add(neighbor)
                    stack.append(neighbor)
    return result
