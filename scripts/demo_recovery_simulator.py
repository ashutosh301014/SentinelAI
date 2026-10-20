"""Working what-if simulation: a degraded API Gateway incident and two
hypothetical recovery actions evaluated against a copy of the topology.

Scenario telemetry: Payments (health 70) and Database (health 75) are already
degraded when the gateway incident (risk 0.85) occurs.
"""
from __future__ import annotations

import json

from sentinelai.config import get_settings
from sentinelai.dependency_graph import load_topology
from sentinelai.recovery_simulator import simulate_recovery


def main() -> None:
    settings = get_settings()
    graph = load_topology(settings.service_topology)
    graph.nodes["payments"]["current_health"] = 70.0
    graph.nodes["database"]["current_health"] = 75.0

    print("===== WORKING WHAT-IF SIMULATION 1 =====")
    print("action: restart the API Gateway (the incident origin)")
    restart = simulate_recovery(
        graph, "restart_service", "api_gateway", origin_risk=0.85
    )
    print(restart.format_summary())
    print()
    print(json.dumps(restart.to_dict(), indent=2))
    print()

    print("===== WORKING WHAT-IF SIMULATION 2 =====")
    print("action: isolate the non-critical Notification service")
    isolate = simulate_recovery(
        graph,
        "isolate_noncritical_service",
        "notification",
        origin="api_gateway",
        origin_risk=0.85,
    )
    print(isolate.format_summary())
    print()
    print(
        "changed propagation paths: "
        + json.dumps(isolate.changed_propagation_paths, indent=2)
    )
    print()

    # prove the live graph was never mutated
    unchanged = (
        graph.number_of_edges() == 7
        and graph.nodes["payments"]["current_health"] == 70.0
        and graph.nodes["database"]["current_health"] == 75.0
        and graph.nodes["api_gateway"]["current_health"] == 98.0
    )
    print(f"original graph untouched: {unchanged}")
    print(
        "Note: Estimated Counterfactual Impact is a what-if estimate; "
        "it does not guarantee recovery."
    )


if __name__ == "__main__":
    main()
