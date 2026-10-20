"""What-If Recovery Simulator for SentinelAI.

Simulates hypothetical remediation actions against a copy of the current
service-dependency state and recalculates the blast radius, so operators can
compare "before" and "after" without touching the live graph. Results are
labeled **Estimated Counterfactual Impact**.

Workflow::

    1. copy the current dependency graph state (deep copy)
    2. NEVER mutate the original graph
    3. apply the hypothetical remediation to the copied state
    4. recalculate propagation / blast radius on the copy
    5. compare before and after
    6. return the Estimated Counterfactual Impact

Supported actions and their transparent effects on the copied state::

    restart_service               health=100, error_rate=0, latency_change=1,
                                  anomaly_score=0, incident_probability=0
    reduce_error_rate             error_rate halved, health +10 (capped at 100)
    reduce_latency                latency_change moved halfway back to 1.0,
                                  health +10 (capped at 100)
    improve_service_health        health +20 (capped at 100)
    increase_database_capacity    health +30 (capped at 100),
                                  incident_probability halved
    isolate_noncritical_service   all dependency edges into/out of the service
                                  removed (only for services with
                                  service_criticality < 4.0)

When the remediation targets the incident origin itself, the origin's
incident risk is scaled by an action-specific factor (restart fully clears
it; the other actions halve it, reduce_latency to 0.7x). Actions on
non-origin services leave the origin risk unchanged.

IMPORTANT: these are what-if estimates computed from the current model and
topology. A simulated improvement does not guarantee recovery in the real
system; results must always be presented as Estimated Counterfactual Impact.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass

import networkx as nx

from sentinelai.blast_radius import estimate_blast_radius
from sentinelai.config import Settings, get_settings
from sentinelai.dependency_graph import validate_graph

RECOVERY_ACTIONS: frozenset[str] = frozenset(
    {
        "restart_service",
        "reduce_error_rate",
        "reduce_latency",
        "improve_service_health",
        "increase_database_capacity",
        "isolate_noncritical_service",
    }
)

NONCRITICAL_CRITICALITY_THRESHOLD = 4.0

_ORIGIN_RISK_FACTORS: dict[str, float] = {
    "restart_service": 0.0,
    "reduce_error_rate": 0.5,
    "reduce_latency": 0.7,
    "improve_service_health": 0.5,
    "increase_database_capacity": 0.5,
    "isolate_noncritical_service": 0.5,
}


def _display_action(action: str) -> str:
    return " ".join(word.capitalize() for word in action.split("_"))


def _to_points(risk: float) -> int:
    return max(0, min(100, int(math.floor(risk * 100.0 + 0.5))))


def apply_action(
    graph: nx.DiGraph, action: str, service: str
) -> nx.DiGraph:
    """Return a NEW graph with ``action`` applied to ``service``.

    The input graph is deep-copied first and is never mutated.
    """
    if action not in RECOVERY_ACTIONS:
        raise ValueError(
            f"unknown recovery action {action!r}; supported actions: "
            f"{sorted(RECOVERY_ACTIONS)}"
        )
    if service not in graph.nodes:
        raise KeyError(f"unknown service: {service}")
    if action == "isolate_noncritical_service":
        criticality = float(graph.nodes[service]["service_criticality"])
        if criticality >= NONCRITICAL_CRITICALITY_THRESHOLD:
            raise ValueError(
                f"service '{service}' is critical (service_criticality = "
                f"{criticality} >= {NONCRITICAL_CRITICALITY_THRESHOLD}); "
                "only non-critical services can be isolated"
            )

    copied = copy.deepcopy(graph)
    node = copied.nodes[service]
    if action == "restart_service":
        node.update(
            {
                "current_health": 100.0,
                "error_rate": 0.0,
                "latency_change": 1.0,
                "anomaly_score": 0.0,
                "incident_probability": 0.0,
            }
        )
    elif action == "reduce_error_rate":
        node["error_rate"] = max(0.0, float(node["error_rate"]) / 2.0)
        node["current_health"] = min(
            100.0, float(node["current_health"]) + 10.0
        )
    elif action == "reduce_latency":
        node["latency_change"] = 1.0 + (
            float(node["latency_change"]) - 1.0
        ) / 2.0
        node["current_health"] = min(
            100.0, float(node["current_health"]) + 10.0
        )
    elif action == "improve_service_health":
        node["current_health"] = min(
            100.0, float(node["current_health"]) + 20.0
        )
    elif action == "increase_database_capacity":
        node["current_health"] = min(
            100.0, float(node["current_health"]) + 30.0
        )
        node["incident_probability"] = (
            float(node["incident_probability"]) / 2.0
        )
    else:  # isolate_noncritical_service
        copied.remove_edges_from(
            list(copied.in_edges(service)) + list(copied.out_edges(service))
        )
    return copied


@dataclass(frozen=True)
class RecoverySimulation:
    """Estimated Counterfactual Impact of one hypothetical action."""

    action: str
    affected_service: str
    original_incident_risk: int
    simulated_incident_risk: int
    original_services_at_risk: int
    simulated_services_at_risk: int
    estimated_risk_reduction: int
    changed_propagation_paths: list[dict]
    before_state: dict
    simulated_after_state: dict

    def to_dict(self) -> dict:
        return {
            "action": self.action,
            "affected_service": self.affected_service,
            "original_incident_risk": self.original_incident_risk,
            "simulated_incident_risk": self.simulated_incident_risk,
            "original_services_at_risk": self.original_services_at_risk,
            "simulated_services_at_risk": self.simulated_services_at_risk,
            "estimated_risk_reduction": self.estimated_risk_reduction,
            "changed_propagation_paths": self.changed_propagation_paths,
            "before_state": self.before_state,
            "simulated_after_state": self.simulated_after_state,
        }

    def format_summary(self) -> str:
        """Human-readable before/after summary in the canonical format."""
        return (
            "BEFORE\n"
            f"Incident Risk = {self.original_incident_risk}\n"
            f"Services At Risk = {self.original_services_at_risk}\n"
            "\n"
            "ACTION\n"
            f"{_display_action(self.action)} ({self.affected_service})\n"
            "\n"
            "SIMULATED AFTER\n"
            f"Incident Risk = {self.simulated_incident_risk}\n"
            f"Services At Risk = {self.simulated_services_at_risk}\n"
            "\n"
            "Estimated Risk Reduction = "
            f"{self.estimated_risk_reduction} points"
        )


def simulate_recovery(
    graph: nx.DiGraph,
    action: str,
    affected_service: str,
    *,
    origin: str | None = None,
    origin_risk: float | None = None,
    settings: Settings | None = None,
) -> RecoverySimulation:
    """Simulate one hypothetical recovery action and compare before/after.

    The original ``graph`` is never mutated: the remediation is applied to a
    deep copy. ``origin`` defaults to the affected service; ``origin_risk``
    defaults to the origin's ``incident_probability`` attribute (same as
    :func:`sentinelai.blast_radius.estimate_blast_radius`).
    """
    issues = validate_graph(graph)
    if issues:
        raise ValueError("invalid dependency graph: " + "; ".join(issues))
    origin = affected_service if origin is None else origin

    before = estimate_blast_radius(
        graph, origin, origin_risk, settings=settings
    )
    simulated_graph = apply_action(graph, action, affected_service)
    factor = (
        _ORIGIN_RISK_FACTORS[action]
        if affected_service == origin
        else 1.0
    )
    after = estimate_blast_radius(
        simulated_graph, origin, before.origin_risk * factor,
        settings=settings,
    )

    before_paths = {
        service.service_id: service.probable_propagation_path
        for service in before.services_at_risk
    }
    after_paths = {
        service.service_id: service.probable_propagation_path
        for service in after.services_at_risk
    }
    changed_propagation_paths = [
        {
            "service": service_id,
            "before_path": before_paths.get(service_id),
            "after_path": after_paths.get(service_id),
        }
        for service_id in sorted(set(before_paths) | set(after_paths))
        if before_paths.get(service_id) != after_paths.get(service_id)
    ]

    original_points = _to_points(before.origin_risk)
    simulated_points = _to_points(after.origin_risk)
    return RecoverySimulation(
        action=action,
        affected_service=affected_service,
        original_incident_risk=original_points,
        simulated_incident_risk=simulated_points,
        original_services_at_risk=sum(
            1
            for service in before.services_at_risk
            if service.estimated_propagation_risk > 0.0
        ),
        simulated_services_at_risk=sum(
            1
            for service in after.services_at_risk
            if service.estimated_propagation_risk > 0.0
        ),
        estimated_risk_reduction=original_points - simulated_points,
        changed_propagation_paths=changed_propagation_paths,
        before_state=before.to_dict(),
        simulated_after_state=after.to_dict(),
    )
