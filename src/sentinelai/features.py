"""Feature engineering for HDFS event occurrence vectors.

Production contract:

* All transformers are per-row and unsupervised: ``fit`` inspects only the
  event-count columns and never sees labels, so nothing is learned from y
  and nothing leaks across the train/test split.
* The output feature order is fixed at fit time (canonical E-number order
  for the kept original events, then the 8 derived features) and enforced on
  every transform - including single unseen inference rows.
* Input must be a DataFrame containing only event columns matching ``E\\d+``.
  Labels, identifiers and metadata are rejected outright.

Derived features (computed per row from the event counts):

* ``total_event_count``              sum of all event counts
* ``unique_event_count``             number of event types present (count > 0)
* ``dominant_event_ratio``           most frequent event count / total
* ``repeated_event_ratio``           (total - unique) / total
* ``event_entropy``                  Shannon entropy of the count
                                     distribution, normalised by
                                     log2(unique) so it lies in [0, 1];
                                     0 when total == 0 or unique <= 1
* ``rare_event_count``               number of rare event types present in
                                     the row; the rare set is learned at fit
                                     (presence rate in the training split
                                     below ``rare_presence_threshold``)
* ``zero_event_ratio``               event types with count == 0 / total
                                     event types
* ``maximum_single_event_frequency`` max single event count / total; equals
                                     ``dominant_event_ratio`` by construction
                                     and is kept as a separate named column
                                     for spec parity
"""
from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

EVENT_COLUMN_RE = re.compile(r"^E(\d+)$")

DERIVED_FEATURE_NAMES: tuple[str, ...] = (
    "total_event_count",
    "unique_event_count",
    "dominant_event_ratio",
    "repeated_event_ratio",
    "event_entropy",
    "rare_event_count",
    "zero_event_ratio",
    "maximum_single_event_frequency",
)


def _event_number(column: Any) -> int:
    match = EVENT_COLUMN_RE.fullmatch(str(column))
    if not match:
        raise ValueError(f"not an event column: {column!r}")
    return int(match.group(1))


def _validate_event_frame(
    X: Any,
    *,
    expected: set[str] | None,
    context: str,
) -> pd.DataFrame:
    """Enforce the input contract and return a sanitised copy.

    * DataFrame only
    * columns must all match ``E\\d+`` (no labels/identifiers/metadata)
    * no duplicate column names
    * when ``expected`` is given (transform), the column set must match the
      fitted schema exactly - missing and unseen columns are errors
    * numeric, finite, non-null values only
    """
    if not isinstance(X, pd.DataFrame):
        raise TypeError(
            f"{context}: expected a pandas DataFrame of event columns, "
            f"got {type(X).__name__}"
        )
    columns = [str(c) for c in X.columns]
    if len(columns) != len(set(columns)):
        raise ValueError(f"{context}: duplicate column names")
    non_event = [c for c in columns if not EVENT_COLUMN_RE.fullmatch(c)]
    if non_event:
        raise ValueError(
            f"{context}: only event columns (E<number>) are allowed in the "
            f"feature matrix; found {non_event}. Labels, identifiers and "
            f"metadata must never enter X."
        )
    if expected is not None and set(columns) != expected:
        missing = sorted(expected - set(columns))
        unseen = sorted(set(columns) - expected)
        parts = []
        if missing:
            parts.append(f"missing {missing}")
        if unseen:
            parts.append(f"unseen {unseen}")
        raise ValueError(
            f"{context}: feature schema differs from the fitted one: "
            f"{', '.join(parts)}"
        )
    sanitized = X.copy()
    for col in columns:
        series = pd.to_numeric(sanitized[col], errors="coerce")
        if series.isna().any():
            raise ValueError(
                f"{context}: event column {col!r} contains missing or "
                f"non-numeric values"
            )
        if not np.isfinite(series.to_numpy(dtype="float64")).all():
            raise ValueError(
                f"{context}: event column {col!r} contains infinite values"
            )
        sanitized[col] = series
    return sanitized


class EventFeatureSelector(BaseEstimator, TransformerMixin):
    """Keep only useful original event columns.

    Learned at fit (from the training split only):

    * drops constant columns (zero variance, e.g. HDFS E22 which is always 1)
    * drops exact-duplicate columns (e.g. HDFS E25, identical to E18)

    Output columns are always the fitted kept columns in canonical E-number
    order, regardless of the input column order, so training and inference
    share exactly the same feature ordering.
    """

    def __init__(self, drop_constant: bool = True, drop_duplicate: bool = True):
        self.drop_constant = drop_constant
        self.drop_duplicate = drop_duplicate

    def fit(self, X: pd.DataFrame, y: Any = None) -> "EventFeatureSelector":
        del y
        X = _validate_event_frame(X, expected=None, context="EventFeatureSelector.fit")
        ordered = sorted(X.columns, key=_event_number)
        matrix = X[ordered].to_numpy()
        kept: list[str] = []
        constant: list[str] = []
        duplicate: list[str] = []
        for i, col in enumerate(ordered):
            vector = matrix[:, i]
            if self.drop_constant and float(np.ptp(vector)) == 0.0:
                constant.append(col)
                continue
            if self.drop_duplicate and any(
                np.array_equal(vector, matrix[:, j]) for j in range(i) if ordered[j] in kept
            ):
                duplicate.append(col)
                continue
            kept.append(col)
        self.kept_event_columns_: list[str] = kept
        self.constant_columns_: list[str] = constant
        self.duplicate_columns_: list[str] = duplicate
        self.input_event_columns_: list[str] = ordered
        self.n_features_in_ = len(ordered)
        self.feature_names_in_ = np.asarray(ordered, dtype=object)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = _validate_event_frame(
            X,
            expected=set(self.input_event_columns_),
            context="EventFeatureSelector.transform",
        )
        return X.reindex(columns=self.kept_event_columns_)

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        del input_features
        return np.asarray(self.kept_event_columns_, dtype=object)


class DerivedFeatureExtractor(BaseEstimator, TransformerMixin):
    """Append the 8 derived per-row features to the event columns.

    The rare-event set is learned at fit from the training split alone:
    event types whose presence rate (fraction of rows with count > 0) is
    below ``rare_presence_threshold``. It is unsupervised - labels are never
    consulted - so nothing is fitted on the full dataset before splitting
    and nothing leaks from the test set.
    """

    def __init__(self, rare_presence_threshold: float = 0.01):
        if not 0.0 <= rare_presence_threshold < 1.0:
            raise ValueError(
                f"rare_presence_threshold must be in [0, 1), "
                f"got {rare_presence_threshold!r}"
            )
        self.rare_presence_threshold = rare_presence_threshold

    def fit(self, X: pd.DataFrame, y: Any = None) -> "DerivedFeatureExtractor":
        del y
        X = _validate_event_frame(X, expected=None, context="DerivedFeatureExtractor.fit")
        ordered = sorted(X.columns, key=_event_number)
        presence = (X[ordered] > 0).mean(axis=0)
        rare = [c for c, p in zip(ordered, presence) if p < self.rare_presence_threshold]
        self.event_columns_: list[str] = ordered
        self.rare_event_columns_: list[str] = rare
        self.n_features_in_ = len(ordered)
        self.feature_names_in_ = np.asarray(ordered, dtype=object)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = _validate_event_frame(
            X,
            expected=set(self.event_columns_),
            context="DerivedFeatureExtractor.transform",
        )
        ordered = X.reindex(columns=self.event_columns_)
        counts = ordered.to_numpy(dtype="float64")
        n_types = counts.shape[1]
        total = counts.sum(axis=1)
        unique = (counts > 0).sum(axis=1)
        max_count = counts.max(axis=1)
        safe_total = np.where(total > 0, total, 1.0)

        dominant = np.where(total > 0, max_count / safe_total, 0.0)
        repeated = np.where(total > 0, (total - unique) / safe_total, 0.0)

        with np.errstate(divide="ignore", invalid="ignore"):
            proportions = counts / total[:, None]
            positive = np.where(proportions > 0, proportions, 0.0)
            log_terms = np.where(proportions > 0, np.log2(proportions), 0.0)
            entropy_raw = -(positive * log_terms).sum(axis=1)
            entropy_normalised = entropy_raw / np.log2(
                np.where(unique > 1, unique, 2)
            )
        entropy = np.where(unique > 1, entropy_normalised, 0.0)
        entropy = np.where(total > 0, entropy, 0.0)

        rare_count = np.zeros(counts.shape[0])
        if self.rare_event_columns_:
            rare_idx = [self.event_columns_.index(c) for c in self.rare_event_columns_]
            rare_count = (counts[:, rare_idx] > 0).sum(axis=1)

        zero_ratio = (n_types - unique) / n_types

        derived = pd.DataFrame(
            {
                "total_event_count": total,
                "unique_event_count": unique,
                "dominant_event_ratio": dominant,
                "repeated_event_ratio": repeated,
                "event_entropy": entropy,
                "rare_event_count": rare_count,
                "zero_event_ratio": zero_ratio,
                "maximum_single_event_frequency": dominant,
            },
            index=X.index,
        )
        return pd.concat([ordered, derived], axis=1)

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        del input_features
        return np.asarray(
            list(self.event_columns_) + list(DERIVED_FEATURE_NAMES), dtype=object
        )
