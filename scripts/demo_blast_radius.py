"""Working demo incident: a probable incident at the API Gateway and the
resulting estimated blast radius over the demo service topology.

Scenario telemetry: Payments (health 70) and Database (health 75) are already
degraded before the gateway incident; every other service is at its
configured baseline health.
"""
from __future__ import annotations

import json

from sentinelai.blast_radius import estimate_blast_radius
from sentinelai.config import get_settings
from sentinelai.dependency_graph import load_topology


def main() -> None:
    settings = get_settings()
    graph = load_topology(settings.service_topology)

    graph.nodes["payments"]["current_health"] = 70.0
    graph.nodes["database"]["current_health"] = 75.0

    origin_risk = 0.85
    result = estimate_blast_radius(graph, "api_gateway", origin_risk)

    print("===== DEMO INCIDENT: API GATEWAY DEGRADATION =====")
    print(
        f"probable origin: {result.origin_name} "
        f"(incident risk {result.origin_risk})"
    )
    print()
    print(json.dumps(result.to_dict(), indent=2))
    print()
    print(
        "Estimated Propagation Risk is a relative estimate for "
        "prioritization; it is NOT a guaranteed or calibrated failure "
        "probability."
    )


if __name__ == "__main__":
    main()
