"""Composite Incident Risk Score for SentinelAI.

The Composite Incident Risk Score fuses four independent signals into a
single operational severity number on a 0-100 scale:

* classifier_probability  — supervised anomaly probability
* anomaly_score           — Isolation Forest anomaly score
* event_irregularity      — how unusual the event pattern is
* event_intensity         — event volume relative to normal behaviour

Every signal must be normalized to [0, 1]. The score is a transparent
weighted sum: each signal contributes ``round_half_up(100 * weight *
signal)`` points of the 100-point scale, and the risk score is the sum of
the four contributions, clamped to [0, 100] so the result always stays on
the 0-100 scale. Weights and severity-band thresholds live in
``sentinelai.config.Settings``.

The Composite Incident Risk Score is an operational risk indicator for
triage and prioritization. It is NOT a calibrated probability.

Severity bands (lower bounds inclusive, from ``risk_band_thresholds``):

    0-29    LOW
    30-59   MEDIUM
    60-79   HIGH
    80-100  CRITICAL
"""
from __future__ import annotations

import math
import numbers
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from sentinelai.config import Settings, get_settings

RISK_SCORE_MIN = 0
RISK_SCORE_MAX = 100

SIGNAL_WEIGHT_ATTRS = {
    "classifier": "weight_classifier",
    "anomaly_detector": "weight_anomaly_detector",
    "event_irregularity": "weight_event_irregularity",
    "event_intensity": "weight_event_intensity",
}

SIGNAL_DESCRIPTIONS = {
    "classifier": "supervised anomaly probability",
    "anomaly_detector": "Isolation Forest anomaly score",
    "event_irregularity": "event-pattern irregularity",
    "event_intensity": "event volume intensity",
}


class Severity(StrEnum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


@dataclass(frozen=True)
class RiskAssessment:
    """Composite Incident Risk Score result for one incident.

    ``contributions`` maps each signal name to its point contribution
    (integer, 0-100 scale); the contributions always sum to
    ``risk_score``.
    """

    risk_score: int
    severity: Severity
    contributions: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "risk_score": self.risk_score,
            "severity": self.severity.value,
            "contributions": dict(self.contributions),
        }


def severity_for_score(
    score: int, thresholds: tuple[int, int, int] = (30, 60, 80)
) -> Severity:
    """Map a 0-100 risk score to its severity band."""
    if score < thresholds[0]:
        return Severity.LOW
    if score < thresholds[1]:
        return Severity.MEDIUM
    if score < thresholds[2]:
        return Severity.HIGH
    return Severity.CRITICAL


def _validate_signal(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(
            f"{name} must be a real number in [0, 1], got {value!r}"
        )
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite, got {value!r}")
    if not 0.0 <= number <= 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {number}")
    return number


def _round_half_up(points: float) -> int:
    """Round to the nearest integer, ties away from zero."""
    return int(math.floor(points + 0.5))


def compute_risk_score(
    classifier_probability: Any,
    anomaly_score: Any,
    event_irregularity: Any,
    event_intensity: Any,
    *,
    settings: Settings | None = None,
) -> RiskAssessment:
    """Compute the Composite Incident Risk Score (0-100) for one incident.

    Formula (fully transparent, weights from ``Settings``):

        contribution_i  = round_half_up(100 * weight_i * signal_i)
        risk_score      = sum of the four contributions, clamped to [0, 100]

    Raises ValueError if any input is not a finite real number in [0, 1].
    """
    settings = settings or get_settings()
    signals = {
        "classifier": _validate_signal(
            "classifier_probability", classifier_probability
        ),
        "anomaly_detector": _validate_signal("anomaly_score", anomaly_score),
        "event_irregularity": _validate_signal(
            "event_irregularity", event_irregularity
        ),
        "event_intensity": _validate_signal(
            "event_intensity", event_intensity
        ),
    }
    contributions = {
        name: _round_half_up(
            100.0 * getattr(settings, SIGNAL_WEIGHT_ATTRS[name]) * signal
        )
        for name, signal in signals.items()
    }
    risk_score = min(
        RISK_SCORE_MAX, sum(contributions.values())
    )
    severity = severity_for_score(risk_score, settings.risk_band_thresholds)
    return RiskAssessment(
        risk_score=int(risk_score),
        severity=severity,
        contributions=contributions,
    )
