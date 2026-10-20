"""Unsupervised anomaly detection with Isolation Forest.

Semi-supervised protocol: the detector is fitted on NORMAL samples from the
training split only. Test data and test labels are never used for fitting
(labels are only used for evaluation afterwards).

Normalized anomaly score: raw IsolationForest ``score_samples`` (higher =
more normal, bounded in [-1, 0]) are mapped to [0, 1] with a sigmoid
calibrated on the training-normal score distribution:

    z = (raw - mu) / sigma
    anomaly_score(x) = 1 / (1 + exp(z + k))

where ``mu``/``sigma`` are the mean/std of raw scores on the training
normal set, and ``k`` anchors a mean training-normal sample at 0.15.
0 = strongly normal, 1 = strongly anomalous. ``contamination`` is "auto"
because the training set contains no anomalies, so no in-training anomaly
rate is assumed.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.ensemble import IsolationForest
from sklearn.utils.validation import check_is_fitted

from sentinelai.preprocessing import build_feature_pipeline

DEFAULT_N_ESTIMATORS = 200
DEFAULT_THRESHOLD = 0.5
# sigmoid offset: a mean training-normal raw score (z = 0) maps to
# 1 / (1 + exp(k)) = 0.15, so typical normal samples score near 0
CALIBRATION_OFFSET = 1.73


class AnomalyDetector(BaseEstimator):
    """Isolation-Forest anomaly detector with a normalized [0, 1] score.

    Parameters
    ----------
    n_estimators, contamination, max_samples, random_state
        Passed to :class:`sklearn.ensemble.IsolationForest`.
    feature_pipeline
        Feature engineering pipeline applied to raw event vectors; defaults
        to :func:`sentinelai.preprocessing.build_feature_pipeline`.
    """

    def __init__(
        self,
        n_estimators: int = DEFAULT_N_ESTIMATORS,
        contamination: Any = "auto",
        max_samples: Any = "auto",
        random_state: int = 42,
        feature_pipeline: Any = None,
    ):
        self.n_estimators = n_estimators
        self.contamination = contamination
        self.max_samples = max_samples
        self.random_state = random_state
        self.feature_pipeline = (
            feature_pipeline
            if feature_pipeline is not None
            else build_feature_pipeline()
        )
        self._iso = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            max_samples=max_samples,
            n_jobs=-1,
            random_state=random_state,
        )

    def fit(self, X: Any, y: Any = None) -> "AnomalyDetector":
        """Fit on normal samples.

        ``y`` (optional, training labels only) is used solely to select the
        normal rows; when omitted, all rows are treated as normal. Test
        labels must never be passed here.
        """
        if y is not None:
            y_array = np.asarray(y)
            if len(y_array) != len(X):
                raise ValueError(
                    f"y has {len(y_array)} rows but X has {len(X)}"
                )
            normal_mask = y_array == 0
            if not normal_mask.any():
                raise ValueError(
                    "no normal samples (label 0) in the training set; "
                    "the detector trains on normal samples only"
                )
            X_train = X.loc[normal_mask]
            self.normal_train_rows_ = int(normal_mask.sum())
            self.total_train_rows_ = len(X)
        else:
            X_train = X
            self.normal_train_rows_ = len(X)
            self.total_train_rows_ = len(X)

        features = self.feature_pipeline.fit_transform(X_train)
        self._iso.fit(features)
        raw_train = self._iso.score_samples(features)
        self.raw_mean_ = float(np.mean(raw_train))
        scale = float(np.std(raw_train))
        self.raw_scale_ = scale if np.isfinite(scale) and scale > 0 else 1e-6
        self.n_features_out_ = features.shape[1]
        return self

    def decision_function(self, X: Any) -> np.ndarray:
        """Raw IsolationForest score_samples: higher = more normal."""
        check_is_fitted(self, attributes=["raw_mean_"])
        features = self.feature_pipeline.transform(X)
        return self._iso.score_samples(features)

    def score(self, X: Any) -> np.ndarray:
        """Normalized anomaly score in [0, 1]: 0 normal, 1 anomalous."""
        raw = self.decision_function(X)
        z = (raw - self.raw_mean_) / self.raw_scale_
        return 1.0 / (1.0 + np.exp(z + CALIBRATION_OFFSET))

    def predict(self, X: Any, threshold: float = DEFAULT_THRESHOLD) -> np.ndarray:
        """Binary labels: 1 (anomalous) where score >= threshold."""
        return (self.score(X) >= threshold).astype(int)


def save_detector(detector: AnomalyDetector, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(detector, path)


def load_detector(path: Path) -> AnomalyDetector:
    return joblib.load(path)
