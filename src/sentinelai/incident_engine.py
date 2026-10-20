"""Incident Intelligence Engine for SentinelAI.

Builds an :class:`IncidentReport` for one observed trace (block) by fusing
every signal SentinelAI has:

* supervised classifier probability and prediction
* Isolation Forest anomaly score
* Composite Incident Risk Score (severity + band)
* Top Evidence phrases against the training-normal profile
* probable contributing events (SHAP, mapped to HDFS templates)
* probable contributing component and probable cause
* evidence-based investigation recommendations

Causality disclaimer: the HDFS_v1 dataset provides NO experimentally
verified causal root-cause labels. This module therefore never claims a
confirmed root cause. It uses "Probable Cause", "Root-Cause Evidence",
"Probable Contributing Event" and "Probable Contributing Component", and
all statements are derived from observed evidence only — no system state
is invented (e.g. timestamps are only reported when actually provided).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from sentinelai.config import Settings, get_settings
from sentinelai.explainability import (
    EVENT_FEATURE_RE,
    NORMAL_EVIDENCE,
    EvidenceProfile,
    local_feature_contributions,
    map_event_features,
    pipeline_feature_names,
    top_evidence,
)
from sentinelai.risk_engine import compute_risk_score

NORMAL_CAUSE = (
    "Probable Cause: none identified — no anomalous evidence observed "
    "(normal operation)."
)
NO_DETECTION_CAUSE_TEMPLATE = (
    "Probable Cause: none identified — no anomaly detected, although "
    "minor deviations were observed ({deviations})."
)
UNKNOWN_CAUSE = (
    "Probable Cause: unknown — anomaly detected without clear evidence."
)
ARCHIVE_RECOMMENDATION = (
    "No investigation required — archive this trace to extend the "
    "normal-behavior baseline."
)
UNKNOWN_COMPONENT = "unknown component (insufficient evidence)"
NO_ANOMALY_COMPONENT = "none (no anomaly detected)"

# template keyword -> probable HDFS component (grounded in the 29 real
# HDFS_v1 event templates; first matching rule wins)
COMPONENT_RULES = (
    (
        "NameNode (FSNamesystem/BlockManager)",
        (
            "namesystem",
            "volumemap",
            "neededreplications",
            "pendingreplicationmonitor",
            "allocateblock",
            "invalidset",
            "adding an already existing block",
        ),
    ),
    (
        "NameNode (client request serving)",
        ("served block", "got exception while serving"),
    ),
    (
        "DataNode (block pipeline)",
        (
            "packetresponder",
            "receiving block",
            "received block",
            "receiveblock",
            "writeblock",
            "writing block",
            "transmitted block",
            "transfer",
            "reopen block",
            "changing block file offset",
            "deleting block",
        ),
    ),
    (
        "DataNode (BlockScanner)",
        ("verification",),
    ),
)


@dataclass
class IncidentReport:
    """Structured incident report for one observed trace."""

    incident_id: str
    timestamp: str | None
    anomaly_detected: bool
    classifier_probability: float
    anomaly_score: float
    risk_score: int
    severity: str
    top_evidence: list[str]
    probable_contributing_events: list[dict[str, Any]] = field(
        default_factory=list
    )
    probable_component: str = NO_ANOMALY_COMPONENT
    probable_cause: str = NORMAL_CAUSE
    investigation_recommendations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "incident_id": self.incident_id,
            "timestamp": self.timestamp,
            "anomaly_detected": self.anomaly_detected,
            "classifier_probability": self.classifier_probability,
            "anomaly_score": self.anomaly_score,
            "risk_score": self.risk_score,
            "severity": self.severity,
            "top_evidence": list(self.top_evidence),
            "probable_contributing_events": [
                dict(event) for event in self.probable_contributing_events
            ],
            "probable_component": self.probable_component,
            "probable_cause": self.probable_cause,
            "investigation_recommendations": list(
                self.investigation_recommendations
            ),
        }


def component_for_template(template: str | None) -> str:
    """Probable HDFS component for an event template (keyword rules).

    Returns the fallback ``unknown component`` phrase when the template
    is missing or matches no rule.
    """
    if not template:
        return UNKNOWN_COMPONENT
    lowered = template.lower()
    for component, keywords in COMPONENT_RULES:
        if any(keyword in lowered for keyword in keywords):
            return component
    return UNKNOWN_COMPONENT


def event_irregularity(
    profile: EvidenceProfile, pipeline: Any, sample: pd.DataFrame
) -> float:
    """Share of model features outside the training-normal [p5, p95]."""
    names = pipeline_feature_names(pipeline)
    values = pd.Series(
        np.asarray(pipeline[:-1].transform(sample)).ravel(), index=names
    )
    p5 = profile.percentiles["p5"]
    p95 = profile.percentiles["p95"]
    outside = int(((values < p5) | (values > p95)).sum())
    return outside / len(names)


def event_intensity(
    profile: EvidenceProfile, pipeline: Any, sample: pd.DataFrame
) -> float:
    """Total event count relative to the training-normal p95, capped at 1."""
    names = pipeline_feature_names(pipeline)
    values = pd.Series(
        np.asarray(pipeline[:-1].transform(sample)).ravel(), index=names
    )
    total = float(values["total_event_count"])
    p95 = float(profile.percentiles.loc["total_event_count", "p95"])
    return min(1.0, total / max(p95, 1.0))


def _contributing_events(
    pipeline: Any,
    templates: pd.DataFrame | None,
    sample: pd.DataFrame,
    max_events: int = 3,
) -> list[dict[str, Any]]:
    _, contributions = local_feature_contributions(pipeline, sample)
    row = contributions.iloc[0]
    event_features = [
        name for name in row.index if EVENT_FEATURE_RE.match(str(name))
    ]
    positive = row[event_features][row[event_features] > 0].sort_values(
        ascending=False
    )
    mapping = map_event_features(templates, list(positive.index))
    return [
        {
            "event_id": str(name),
            "template": mapping.get(str(name)),
            "contribution": float(positive[name]),
        }
        for name in positive.index[:max_events]
    ]


def _recommendations(
    evidence: list[str],
    contributing_events: list[dict[str, Any]],
    anomaly_detected: bool,
    max_recommendations: int = 5,
) -> list[str]:
    recommendations: list[str] = []
    if any("rare event" in phrase for phrase in evidence):
        recommendations.append(
            "Review why the rare event(s) occurred — their presence is "
            "outside the training-normal baseline."
        )
    if any("abnormal total event count" in phrase for phrase in evidence):
        recommendations.append(
            "Compare the block's event volume against its historical "
            "baseline."
        )
    if any("low event count" in phrase for phrase in evidence):
        recommendations.append(
            "Investigate the unusually low event count (possible stalled "
            "or partial block operation)."
        )
    if any("repeated" in phrase for phrase in evidence):
        recommendations.append(
            "Inspect retry/repetition patterns in the event sequence "
            "(possible loop or failing operation)."
        )
    if any(
        "no dominant event type" in phrase or "diversity" in phrase
        for phrase in evidence
    ):
        recommendations.append(
            "Review the event-type diversity — an unusually mixed pattern "
            "deviates from typical block operations."
        )
    if any("absent" in phrase for phrase in evidence):
        recommendations.append(
            "Check why an event type that is usually present did not occur."
        )
    seen: set[tuple[str, str]] = set()
    for event in contributing_events:
        component = component_for_template(event.get("template"))
        key = (component, str(event["event_id"]))
        if key in seen:
            continue
        seen.add(key)
        recommendations.append(
            f"Inspect {component} logs for '{event['template']}' "
            f"occurrences in the incident window (probable contributing "
            f"event {event['event_id']})."
        )
    if not anomaly_detected:
        recommendations.append(ARCHIVE_RECOMMENDATION)
    return recommendations[:max_recommendations]


def _probable_component(
    contributing_events: list[dict[str, Any]], anomaly_detected: bool
) -> str:
    if not anomaly_detected:
        return NO_ANOMALY_COMPONENT
    if not contributing_events:
        return UNKNOWN_COMPONENT
    return component_for_template(contributing_events[0].get("template"))


def _probable_cause(
    evidence: list[str],
    contributing_events: list[dict[str, Any]],
    anomaly_detected: bool,
) -> str:
    if not anomaly_detected:
        if evidence and evidence != list(NORMAL_EVIDENCE):
            return NO_DETECTION_CAUSE_TEMPLATE.format(
                deviations="; ".join(evidence[:2])
            )
        return NORMAL_CAUSE
    if not evidence:
        return UNKNOWN_CAUSE
    cause = "Probable Cause: " + "; ".join(evidence[:3]) + "."
    if contributing_events:
        top_event = contributing_events[0]
        template = top_event.get("template") or "unknown template"
        cause += (
            " Root-Cause Evidence: probable contributing event "
            f"{top_event['event_id']} ({template})."
        )
    return cause


def build_incident_report(
    block_id: str,
    sample: pd.DataFrame,
    classifier_pipeline: Any,
    anomaly_detector: Any,
    profile: EvidenceProfile,
    templates: pd.DataFrame | None = None,
    *,
    timestamp: str | None = None,
    settings: Settings | None = None,
    top_k: int = 5,
) -> IncidentReport:
    """Build an IncidentReport for one observed trace.

    All inputs (``sample``) must be held-out traces. The report fuses the
    classifier verdict, the Isolation Forest score, the Composite
    Incident Risk Score and the evidence pipeline. No causal claim is
    made — see the module docstring.
    """
    settings = settings or get_settings()
    classifier_probability = float(
        classifier_pipeline.predict_proba(sample)[0, 1]
    )
    predicted = int(classifier_pipeline.predict(sample)[0])
    anomaly_score = float(anomaly_detector.score(sample)[0])
    irregularity = event_irregularity(profile, classifier_pipeline, sample)
    intensity = event_intensity(profile, classifier_pipeline, sample)
    risk = compute_risk_score(
        classifier_probability,
        anomaly_score,
        irregularity,
        intensity,
        settings=settings,
    )
    evidence = top_evidence(
        profile, classifier_pipeline, sample, top_k=top_k
    )
    anomaly_detected = predicted == 1
    contributing_events = (
        _contributing_events(classifier_pipeline, templates, sample)
        if anomaly_detected
        else []
    )
    return IncidentReport(
        incident_id=f"INC-{block_id}",
        timestamp=timestamp,
        anomaly_detected=anomaly_detected,
        classifier_probability=classifier_probability,
        anomaly_score=anomaly_score,
        risk_score=risk.risk_score,
        severity=risk.severity.value,
        top_evidence=evidence,
        probable_contributing_events=contributing_events,
        probable_component=_probable_component(
            contributing_events, anomaly_detected
        ),
        probable_cause=_probable_cause(
            evidence, contributing_events, anomaly_detected
        ),
        investigation_recommendations=_recommendations(
            evidence, contributing_events, anomaly_detected
        ),
    )
