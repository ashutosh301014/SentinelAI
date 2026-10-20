"""Unit tests for sentinelai.anomaly_detector (toy data only)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from sentinelai.anomaly_detector import (
    CALIBRATION_OFFSET,
    AnomalyDetector,
    load_detector,
    save_detector,
)


def make_toy_frame(n: int = 60, n_anomaly: int = 8, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(
        rng.integers(0, 5, size=(n, 6)),
        columns=[f"E{i}" for i in range(1, 7)],
    )
    y = pd.Series(
        np.array([1] * n_anomaly + [0] * (n - n_anomaly), dtype="int8"),
        name="Label",
    )
    return frame, y


class TestFitProtocol:
    def test_fit_uses_normal_rows_only(self):
        frame, y = make_toy_frame()
        detector = AnomalyDetector(random_state=0).fit(frame, y)
        assert detector.normal_train_rows_ == int((y == 0).sum())
        assert detector.total_train_rows_ == len(y)

    def test_fit_without_labels_uses_all_rows(self):
        frame, _ = make_toy_frame()
        detector = AnomalyDetector(random_state=0).fit(frame)
        assert detector.normal_train_rows_ == len(frame)

    def test_fit_requires_normal_samples(self):
        frame, y = make_toy_frame()
        with pytest.raises(ValueError, match="no normal samples"):
            AnomalyDetector(random_state=0).fit(frame, np.ones(len(frame), dtype=int))

    def test_fit_rejects_mismatched_labels(self):
        frame, _ = make_toy_frame()
        with pytest.raises(ValueError, match="has .* rows"):
            AnomalyDetector(random_state=0).fit(frame, np.zeros(3, dtype=int))

    def test_deterministic(self):
        frame, y = make_toy_frame()
        a = AnomalyDetector(random_state=0).fit(frame, y)
        b = AnomalyDetector(random_state=0).fit(frame, y)
        np.testing.assert_array_equal(a.score(frame), b.score(frame))


class TestScoreOutput:
    def test_output_range_0_1(self):
        frame, y = make_toy_frame()
        detector = AnomalyDetector(random_state=0).fit(frame, y)
        scores = detector.score(frame)
        assert scores.shape == (len(frame),)
        assert scores.min() >= 0.0 and scores.max() <= 1.0
        assert np.isfinite(scores).all()

    def test_mapping_consistency(self):
        frame, y = make_toy_frame()
        detector = AnomalyDetector(random_state=0).fit(frame, y)
        raw = detector.decision_function(frame)
        expected = 1.0 / (
            1.0
            + np.exp(
                (raw - detector.raw_mean_) / detector.raw_scale_
                + CALIBRATION_OFFSET
            )
        )
        np.testing.assert_allclose(detector.score(frame), expected)

    def test_calibration_anchor_mean_normal_at_015(self, monkeypatch):
        frame, y = make_toy_frame()
        detector = AnomalyDetector(random_state=0).fit(frame, y)
        # raw score equal to the training-normal mean -> z = 0 -> 1/(1+e^k)
        anchor = 1.0 / (1.0 + np.exp(CALIBRATION_OFFSET))
        assert anchor == pytest.approx(0.15, abs=1e-3)
        monkeypatch.setattr(
            detector,
            "decision_function",
            lambda X: np.full(len(X), detector.raw_mean_),
        )
        np.testing.assert_allclose(
            detector.score(frame.iloc[:3]), [anchor, anchor, anchor]
        )

    def test_outlier_scores_higher_than_typical(self):
        frame, y = make_toy_frame()
        normal = frame[y.to_numpy() == 0]
        detector = AnomalyDetector(random_state=0).fit(normal)
        typical = normal.iloc[[0]]
        extreme = pd.DataFrame(
            {col: [50] for col in frame.columns}
        )
        assert detector.score(extreme)[0] > detector.score(typical)[0]

    def test_inference_single_row(self):
        frame, y = make_toy_frame()
        detector = AnomalyDetector(random_state=0).fit(frame, y)
        row = frame.iloc[[5]]
        assert detector.score(row).shape == (1,)
        np.testing.assert_allclose(
            detector.score(row), detector.score(frame)[[5]]
        )

    def test_predict_threshold(self):
        frame, y = make_toy_frame()
        detector = AnomalyDetector(random_state=0).fit(frame, y)
        pred = detector.predict(frame)
        assert set(np.unique(pred)) <= {0, 1}
        np.testing.assert_array_equal(
            pred, (detector.score(frame) >= 0.5).astype(int)
        )
        strict = detector.predict(frame, threshold=0.99)
        assert (strict == 1).sum() <= (pred == 1).sum()

    def test_unfitted_raises(self):
        frame, _ = make_toy_frame()
        with pytest.raises(NotFittedError):
            AnomalyDetector(random_state=0).score(frame)


class TestFeatureCompatibility:
    def test_missing_event_column_raises(self):
        frame, y = make_toy_frame()
        detector = AnomalyDetector(random_state=0).fit(frame, y)
        with pytest.raises(ValueError, match="missing"):
            detector.score(frame.drop(columns=["E3"]))

    def test_non_event_column_raises(self):
        frame, y = make_toy_frame()
        detector = AnomalyDetector(random_state=0).fit(frame, y)
        with pytest.raises(ValueError, match="only event columns"):
            detector.score(frame.assign(Label=0))

    def test_column_order_independent(self):
        frame, y = make_toy_frame()
        detector = AnomalyDetector(random_state=0).fit(frame, y)
        shuffled = frame[["E6", "E3", "E1", "E5", "E4", "E2"]]
        np.testing.assert_allclose(
            detector.score(shuffled), detector.score(frame)
        )


class TestSerialization:
    def test_roundtrip(self, tmp_path):
        frame, y = make_toy_frame()
        detector = AnomalyDetector(random_state=0).fit(frame, y)
        path = tmp_path / "anomaly_detector.joblib"
        save_detector(detector, path)
        loaded = load_detector(path)
        np.testing.assert_array_equal(
            detector.score(frame), loaded.score(frame)
        )
        assert loaded.raw_mean_ == detector.raw_mean_
        assert loaded.raw_scale_ == detector.raw_scale_
        assert loaded.normal_train_rows_ == detector.normal_train_rows_
