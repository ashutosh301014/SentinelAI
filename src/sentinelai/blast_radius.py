"""Predictive Blast-Radius Engine for SentinelAI.

Given a probable incident origin service and the current per-service health
signals carried on the data-driven dependency graph (anomaly score, incident
probability, error rate, latency change, criticality, health),
:func:`estimate_blast_radius` walks downstream along dependency edges and
computes, for every reachable service, an **Estimated Propagation Risk**.

Transparent per-hop model::

    transmitted(p -> v) = risk(p) * dependency_strength * historical_propagation_factor
    risk(v) = min(1.0, [transmitted + (1 - transmitted) * health_vulnerability(v)]
                      * criticality_factor(v))

where ``health_vulnerability(v) = (100 - current_health(v)) / 100`` and
``criticality_factor(v) = CRITICALITY_FLOOR + (1 - CRITICALITY_FLOOR)
* service_criticality(v) / 5``. Every per-hop multiplier is at most 1.0, so
the estimate generally decays with dependency distance. When several paths
reach a service, the strongest one (highest transmitted risk) wins. Each
service is evaluated exactly once, at its shortest distance from the origin;
edges coming from services at equal or greater distance (back edges,
including self-loops) are ignored, which makes the computation safe on
cyclic topologies.

IMPORTANT: Estimated Propagation Risk is a relative risk estimate for
prioritization and triage. It is NOT a guaranteed or calibrated failure
probability.
"""
from __future__ import annotations

import math
import numbers
from collections import deque
from dataclasses import dataclass, replace
from typing import Any

import networkx as nx

from sentinelai.config import Settings, get_settings
from sentinelai.dependency_graph import validate_graph
from sentinelai.risk_engine import severity_for_score

CRITICALITY_FLOOR = 0.6  # criticality_factor at service_criticality = 0


def _is_number(value: Any) -> bool:
    return isinstance(value, numbers.Real) and not isinstance(value, bool)


def _criticality_factor(criticality: float) -> float:
    return CRITICALITY_FLOOR + (1.0 - CRITICALITY_FLOOR) * (criticality / 5.0)


def _health_vulnerability(health: float) -> float:
    return (100.0 - health) / 100.0


@dataclass(frozen=True)
class AffectedService:
    """One downstream service's Estimated Propagation Risk."""

    service_id: str
    service_name: str
    estimated_propagation_risk: float
    dependency_distance: int
    risk_level: str
    probable_propagation_path: list[str]
    relative_impact_order: int

    def to_dict(self) -> dict:
        return {
            "service_id": self.service_id,
            "service_name": self.service_name,
            "estimated_propagation_risk": self.estimated_propagation_risk,
            "dependency_distance": self.dependency_distance,
            "risk_level": self.risk_level,
            "probable_propagation_path": self.probable_propagation_path,
            "relative_impact_order": self.relative_impact_order,
        }


@dataclass(frozen=True)
class BlastRadiusResult:
    """Blast-radius assessment for one probable incident origin."""

    origin: str
    origin_name: str
    origin_risk: float
    services_at_risk: list[AffectedService]
    most_likely_propagation_path: list[str]
    highest_risk_service: str | None

    def to_dict(self) -> dict:
        return {
            "probable_origin": {
                "id": self.origin,
                "name": self.origin_name,
                "incident_risk": self.origin_risk,
            },
            "services_at_risk": [
                service.to_dict() for service in self.services_at_risk
            ],
            "most_likely_propagation_path": self.most_likely_propagation_path,
            "highest_risk_service": self.highest_risk_service,
        }


def estimate_blast_radius(
    graph: nx.DiGraph,
    origin: str,
    origin_risk: float | None = None,
    *,
    settings: Settings | None = None,
) -> BlastRadiusResult:
    """Estimate the blast radius of an incident at ``origin``.

    ``origin_risk`` is the incident risk at the origin service in [0, 1]
    (e.g. the Composite Incident Risk Score normalized); it defaults to the
    origin's configured ``incident_probability`` node attribute.
    """
    issues = validate_graph(graph)
    if issues:
        raise ValueError("invalid dependency graph: " + "; ".join(issues))
    if origin not in graph.nodes:
        raise KeyError(f"unknown service: {origin}")
    if origin_risk is None:
        origin_risk = float(graph.nodes[origin]["incident_probability"])
    if not _is_number(origin_risk):
        raise ValueError(
            f"origin_risk must be a number in [0, 1], got {origin_risk!r}"
        )
    origin_risk = float(origin_risk)
    if not 0.0 <= origin_risk <= 1.0:
        raise ValueError(
            f"origin_risk must be within [0, 1], got {origin_risk}"
        )
    settings = settings or get_settings()

    distances: dict[str, int] = {origin: 0}
    layers: dict[int, list[str]] = {0: [origin]}
    queue: deque[str] = deque([origin])
    while queue:
        node = queue.popleft()
        for neighbor in graph.successors(node):
            if neighbor not in distances:
                distances[neighbor] = distances[node] + 1
                layers.setdefault(distances[neighbor], []).append(neighbor)
                queue.append(neighbor)

    risks: dict[str, float] = {origin: origin_risk}
    best_predecessor: dict[str, str | None] = {}
    for depth in range(1, max(layers) + 1):
        for node in layers.get(depth, []):
            best_transmitted = 0.0
            best_pred: str | None = None
            for predecessor in graph.predecessors(node):
                if distances.get(predecessor, math.inf) >= depth:
                    continue  # back edge: cycle protection
                transmitted = (
                    risks[predecessor]
                    * float(
                        graph.edges[predecessor, node][
                            "dependency_strength"
                        ]
                    )
                    * float(
                        graph.edges[predecessor, node][
                            "historical_propagation_factor"
                        ]
                    )
                )
                if best_pred is None or transmitted > best_transmitted:
                    best_transmitted = transmitted
                    best_pred = predecessor
            vulnerability = _health_vulnerability(
                float(graph.nodes[node]["current_health"])
            )
            base_risk = best_transmitted + (1.0 - best_transmitted) * vulnerability
            criticality = _criticality_factor(
                float(graph.nodes[node]["service_criticality"])
            )
            risks[node] = min(1.0, base_risk * criticality)
            best_predecessor[node] = best_pred

    def _path_to(node: str) -> list[str]:
        path = [node]
        while path[-1] != origin:
            parent = best_predecessor.get(path[-1])
            if parent is None:
                break
            path.append(parent)
        path.reverse()
        return path

    def _level_for(risk: float) -> str:
        score = min(100, int(math.floor(100.0 * risk + 0.5)))
        return severity_for_score(
            score, thresholds=settings.risk_band_thresholds
        ).value

    affected: list[AffectedService] = []
    for node, distance in distances.items():
        if node == origin:
            continue
        affected.append(
            AffectedService(
                service_id=node,
                service_name=str(graph.nodes[node].get("name", node)),
                estimated_propagation_risk=round(risks[node], 6),
                dependency_distance=distance,
                risk_level=_level_for(risks[node]),
                probable_propagation_path=_path_to(node),
                relative_impact_order=0,
            )
        )
    affected.sort(
        key=lambda service: (
            -service.estimated_propagation_risk,
            service.dependency_distance,
            service.service_id,
        )
    )
    ranked = [
        replace(service, relative_impact_order=index)
        for index, service in enumerate(affected, start=1)
    ]

    return BlastRadiusResult(
        origin=origin,
        origin_name=str(graph.nodes[origin].get("name", origin)),
        origin_risk=origin_risk,
        services_at_risk=ranked,
        most_likely_propagation_path=(
            ranked[0].probable_propagation_path if ranked else []
        ),
        highest_risk_service=(ranked[0].service_id if ranked else None),
    )
