"""Pydantic schemas for the SentinelAI FastAPI backend.

Request models enforce payload shape and limits before anything reaches
the inference pipeline; response models give the API a stable, documented
JSON contract. Every response is one structured object and every error
uses the same envelope (see ``ErrorEnvelope`` in ``app.main``).
"""
from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

EVENT_KEY_RE = re.compile(r"^E\d+$")

RecoveryAction = Literal[
    "restart_service",
    "reduce_error_rate",
    "reduce_latency",
    "improve_service_health",
    "increase_database_capacity",
    "isolate_noncritical_service",
]

DemoSampleName = Literal["normal", "anomalous"]


class WhatIfSpec(BaseModel):
    """One hypothetical recovery action for the what-if stage."""

    action: RecoveryAction
    affected_service: str = Field(min_length=1, max_length=100)


class AnalyzeRequest(BaseModel):
    """One raw event-count vector for the full inference pipeline."""

    block_id: str | None = Field(
        default=None, min_length=1, max_length=200
    )
    events: dict[str, int] = Field(min_length=1, max_length=100)
    timestamp: str | None = Field(default=None, max_length=100)
    what_if: list[WhatIfSpec] | None = Field(default=None, max_length=5)
    top_k: int = Field(default=5, ge=1, le=10)

    @field_validator("events")
    @classmethod
    def _validate_event_counts(cls, value: dict[str, int]) -> dict[str, int]:
        for key, count in value.items():
            if not EVENT_KEY_RE.fullmatch(key):
                raise ValueError(
                    f"event keys must look like 'E<number>', got {key!r}"
                )
            if count < 0:
                raise ValueError(
                    f"event count for {key!r} must be non-negative, got {count}"
                )
        return value


class DemoAnalyzeRequest(BaseModel):
    """Run one of the persisted held-out demo samples through the pipeline."""

    sample: DemoSampleName
    timestamp: str | None = Field(default=None, max_length=100)
    what_if: list[WhatIfSpec] | None = Field(default=None, max_length=5)


class BlastRadiusRequest(BaseModel):
    """Predictive blast radius from one probable incident origin."""

    origin: str = Field(min_length=1, max_length=100)
    origin_risk: float | None = Field(default=None, ge=0.0, le=1.0)


class SimulateRecoveryRequest(BaseModel):
    """What-if recovery simulation on the current topology state."""

    action: RecoveryAction
    affected_service: str = Field(min_length=1, max_length=100)
    origin: str | None = Field(default=None, min_length=1, max_length=100)
    origin_risk: float | None = Field(default=None, ge=0.0, le=1.0)


# ---- responses ----


class HealthResponse(BaseModel):
    status: str
    app_name: str
    version: str
    model_loaded: bool
    dataset: str
    model_version: str


class ModelInfoResponse(BaseModel):
    dataset: str
    trained_at: str
    selected_model: str
    model_version: str
    classifier_threshold: float
    feature_count: int
    feature_names: list[str]
    evaluation_metrics: dict[str, float | int]
    anomaly_detector: dict[str, Any]


class RiskStageModel(BaseModel):
    risk_score: int
    severity: str
    contributions: dict[str, int]
    caveat: str


class IncidentModel(BaseModel):
    incident_id: str
    timestamp: str | None
    anomaly_detected: bool
    classifier_probability: float
    anomaly_score: float
    risk_score: int
    severity: str
    top_evidence: list[str]
    probable_contributing_events: list[dict[str, Any]]
    probable_component: str
    probable_cause: str
    investigation_recommendations: list[str]


class AnalyzeResponse(BaseModel):
    """The single structured object returned by the inference pipeline."""

    request_id: str
    block_id: str
    dataset: str
    model_version: str
    stages: list[str]
    input_validation: dict[str, Any]
    feature_engineering: dict[str, Any]
    supervised_classifier: dict[str, Any]
    isolation_forest: dict[str, Any]
    composite_risk_score: RiskStageModel
    explainability: dict[str, Any]
    incident_intelligence: IncidentModel
    dependency_graph_mapping: dict[str, Any]
    blast_radius_prediction: dict[str, Any] | None
    what_if_simulation: list[dict[str, Any]]


class ServiceNodeModel(BaseModel):
    id: str
    name: str
    anomaly_score: float
    incident_probability: float
    error_rate: float
    latency_change: float
    service_criticality: float
    current_health: float


class DependencyEdgeModel(BaseModel):
    service_id: str
    depends_on: str
    dependency_strength: float
    historical_propagation_factor: float
    latency_sensitivity: float


class DependencyGraphResponse(BaseModel):
    service_count: int
    dependency_count: int
    services: list[ServiceNodeModel]
    dependencies: list[DependencyEdgeModel]


class AffectedServiceModel(BaseModel):
    service_id: str
    service_name: str
    estimated_propagation_risk: float
    dependency_distance: int
    risk_level: str
    probable_propagation_path: list[str]
    relative_impact_order: int


class ProbableOriginModel(BaseModel):
    id: str
    name: str
    incident_risk: float


class BlastRadiusResponse(BaseModel):
    """Estimated Propagation Risk — not a calibrated failure probability."""

    probable_origin: ProbableOriginModel
    services_at_risk: list[AffectedServiceModel]
    most_likely_propagation_path: list[str]
    highest_risk_service: str | None


class SimulateRecoveryResponse(BaseModel):
    """Estimated Counterfactual Impact — never a guaranteed recovery."""

    action: str
    affected_service: str
    original_incident_risk: int
    simulated_incident_risk: int
    original_services_at_risk: int
    simulated_services_at_risk: int
    estimated_risk_reduction: int
    changed_propagation_paths: list[dict[str, Any]]
    before_state: dict[str, Any]
    simulated_after_state: dict[str, Any]
