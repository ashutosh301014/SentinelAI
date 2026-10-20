"""Unit tests for the Composite Incident Risk Score (risk_engine)."""
from __future__ import annotations

import math

import numpy as np
import pytest
from pydantic import ValidationError

from sentinelai.config import Settings
from sentinelai.risk_engine import (
    RISK_SCORE_MAX,
    RISK_SCORE_MIN,
    Severity,
    compute_risk_score,
    severity_for_score,
)

DEFAULT_WEIGHTS = Settings()


def _compute(*signals: float, settings: Settings | None = None):
    return compute_risk_score(*signals, settings=settings)


class TestSeverityBands:
    @pytest.mark.parametrize(
        ("score", "expected"),
        [
            (0, Severity.LOW),
            (29, Severity.LOW),
            (30, Severity.MEDIUM),
            (59, Severity.MEDIUM),
            (60, Severity.HIGH),
            (79, Severity.HIGH),
            (80, Severity.CRITICAL),
            (100, Severity.CRITICAL),
        ],
    )
    def test_severity_for_score(self, score: int, expected: Severity) -> None:
        assert severity_for_score(score) is expected


class TestBoundaries:
    """Exact boundary scores under the default weights 0.45/0.30/0.15/0.10."""

    @pytest.mark.parametrize(
        ("signals", "score", "severity"),
        [
            # 0 boundary
            ((0.0, 0.0, 0.0, 0.0), 0, Severity.LOW),
            # 29/30 boundary
            ((0.4, 0.2, 0.2, 0.2), 29, Severity.LOW),
            ((0.4, 0.2, 0.2, 0.3), 30, Severity.MEDIUM),
            # 59/60 boundary
            ((0.6, 0.6, 0.6, 0.5), 59, Severity.MEDIUM),
            ((0.6, 0.6, 0.6, 0.6), 60, Severity.HIGH),
            # 79/80 boundary
            ((0.8, 0.8, 0.8, 0.7), 79, Severity.HIGH),
            ((0.8, 0.8, 0.8, 0.8), 80, Severity.CRITICAL),
            # 100 boundary
            ((1.0, 1.0, 1.0, 1.0), 100, Severity.CRITICAL),
        ],
    )
    def test_exact_boundary_scores(
        self,
        signals: tuple[float, float, float, float],
        score: int,
        severity: Severity,
    ) -> None:
        assessment = _compute(*signals)
        assert assessment.risk_score == score
        assert assessment.severity is severity

    def test_score_always_within_scale(self) -> None:
        for classifier in np.linspace(0, 1, 11):
            for anomaly in np.linspace(0, 1, 11):
                for irregularity in (0.0, 0.5, 1.0):
                    for intensity in (0.0, 0.5, 1.0):
                        assessment = _compute(
                            classifier, anomaly, irregularity, intensity
                        )
                        assert (
                            RISK_SCORE_MIN
                            <= assessment.risk_score
                            <= RISK_SCORE_MAX
                        )


class TestFormula:
    def test_documented_example(self) -> None:
        assessment = _compute(0.85, 0.90, 0.93, 0.80)
        assert assessment.risk_score == 87
        assert assessment.severity is Severity.CRITICAL
        assert assessment.contributions == {
            "classifier": 38,
            "anomaly_detector": 27,
            "event_irregularity": 14,
            "event_intensity": 8,
        }

    def test_contributions_sum_to_score(self) -> None:
        for signals in (
            (0.85, 0.90, 0.93, 0.80),
            (0.4, 0.2, 0.2, 0.2),
            (1.0, 1.0, 1.0, 1.0),
            (0.0, 0.0, 0.0, 0.0),
            (0.33, 0.47, 0.61, 0.09),
        ):
            assessment = _compute(*signals)
            assert assessment.risk_score == sum(
                assessment.contributions.values()
            )

    def test_to_dict_shape(self) -> None:
        assessment = _compute(0.85, 0.90, 0.93, 0.80)
        assert assessment.to_dict() == {
            "risk_score": 87,
            "severity": "CRITICAL",
            "contributions": {
                "classifier": 38,
                "anomaly_detector": 27,
                "event_irregularity": 14,
                "event_intensity": 8,
            },
        }

    def test_weights_come_from_settings(self) -> None:
        equal = Settings(
            weight_classifier=0.25,
            weight_anomaly_detector=0.25,
            weight_event_irregularity=0.25,
            weight_event_intensity=0.25,
        )
        default = _compute(0.5, 0.5, 0.5, 0.5)
        customized = _compute(0.5, 0.5, 0.5, 0.5, settings=equal)
        assert default.risk_score != customized.risk_score
        # 12.5 points each, half-up rounded: 13 + 13 + 13 + 13
        assert customized.risk_score == 52
        assert customized.contributions == {
            "classifier": 13,
            "anomaly_detector": 13,
            "event_irregularity": 13,
            "event_intensity": 13,
        }

    def test_thresholds_come_from_settings(self) -> None:
        relaxed = Settings(risk_band_thresholds=(20, 50, 80))
        # score 29 is LOW under defaults, MEDIUM under (20, 50, 80)
        assert _compute(0.4, 0.2, 0.2, 0.2).severity is Severity.LOW
        assert (
            _compute(0.4, 0.2, 0.2, 0.2, settings=relaxed).severity
            is Severity.MEDIUM
        )

    def test_accepts_numpy_scalars(self) -> None:
        assessment = _compute(
            np.float64(0.85), np.float32(0.90), np.float64(0.93), np.int64(0)
        )
        assert assessment.risk_score == 79


class TestInvalidInputs:
    @pytest.mark.parametrize(
        "bad",
        [
            float("nan"),
            float("inf"),
            float("-inf"),
            -0.01,
            1.01,
            "0.5",
            None,
            True,
            [0.5],
            complex(0.5, 0.0),
        ],
    )
    def test_invalid_classifier_probability(self, bad) -> None:
        with pytest.raises(ValueError, match="classifier_probability"):
            _compute(bad, 0.5, 0.5, 0.5)

    def test_invalid_value_raises_per_position(self) -> None:
        for position in range(4):
            signals: list = [0.5, 0.5, 0.5, 0.5]
            signals[position] = 1.5
            with pytest.raises(ValueError):
                _compute(*signals)

    def test_nan_rejected_in_any_position(self) -> None:
        for position in range(4):
            signals: list = [0.5, 0.5, 0.5, 0.5]
            signals[position] = float("nan")
            with pytest.raises(ValueError, match="finite"):
                _compute(*signals)


class TestSettingsValidation:
    def test_weights_must_sum_to_one(self) -> None:
        with pytest.raises(ValidationError):
            Settings(
                weight_classifier=0.45,
                weight_anomaly_detector=0.30,
                weight_event_irregularity=0.15,
                weight_event_intensity=0.05,
            )

    def test_thresholds_must_be_in_range(self) -> None:
        with pytest.raises(ValidationError):
            Settings(risk_band_thresholds=(0, 60, 80))
        with pytest.raises(ValidationError):
            Settings(risk_band_thresholds=(30, 60, 100))

    def test_thresholds_must_increase(self) -> None:
        with pytest.raises(ValidationError):
            Settings(risk_band_thresholds=(60, 30, 80))

    def test_thresholds_must_be_three_integers(self) -> None:
        with pytest.raises(ValidationError):
            Settings(risk_band_thresholds=(30, 60))

    def test_default_settings_are_valid(self) -> None:
        total = (
            DEFAULT_WEIGHTS.weight_classifier
            + DEFAULT_WEIGHTS.weight_anomaly_detector
            + DEFAULT_WEIGHTS.weight_event_irregularity
            + DEFAULT_WEIGHTS.weight_event_intensity
        )
        assert math.isclose(total, 1.0)
        assert DEFAULT_WEIGHTS.risk_band_thresholds == (30, 60, 80)
