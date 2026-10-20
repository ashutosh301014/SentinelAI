"""Production feature-engineering orchestration for SentinelAI.

Golden rule: split first, then fit the pipeline on the training split only,
then transform train and test (and any inference data) with the same fitted
pipeline. Nothing here ever fits on the complete dataset before splitting,
and labels never enter the feature matrix.

Recommended flow (wrapped by :func:`prepare_features`)::

    X_train, X_test, y_train, y_test = split_dataset(matrix, labels)
    pipeline = build_feature_pipeline()
    X_train_feat = pipeline.fit_transform(X_train)
    X_test_feat = pipeline.transform(X_test)

Scaling is opt-in via :func:`build_scaled_pipeline` - only for algorithms
that require it (e.g. LogisticRegression). Tree ensembles and
HistGradientBoosting should use the unscaled pipeline.
"""
from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sentinelai.data_loader import EVENT_COLUMNS
from sentinelai.features import DerivedFeatureExtractor, EventFeatureSelector

FORBIDDEN_COLUMN_TOKENS: tuple[str, ...] = (
    "label",
    "type",
    "target",
    "class",
    "blockid",
)


def build_feature_pipeline(
    *,
    drop_constant: bool = True,
    drop_duplicate: bool = True,
    rare_presence_threshold: float = 0.01,
) -> Pipeline:
    """Unscaled feature pipeline: select useful events -> derived features.

    Fit on X_train only; transform train and all inference data with the
    same fitted object so the feature ordering is always identical.
    Suitable as-is for tree-based models.
    """
    return Pipeline(
        steps=[
            (
                "select",
                EventFeatureSelector(
                    drop_constant=drop_constant, drop_duplicate=drop_duplicate
                ),
            ),
            (
                "derive",
                DerivedFeatureExtractor(
                    rare_presence_threshold=rare_presence_threshold
                ),
            ),
        ]
    )


def build_scaled_pipeline(scaler: str = "standard", **kwargs: Any) -> Pipeline:
    """Same feature pipeline plus a trailing scaler step.

    Use only for algorithms that need scaled inputs (e.g. LogisticRegression
    with L2 regularisation). Tree-based models must use the unscaled
    :func:`build_feature_pipeline`. The scaler is fitted on the training
    split together with the rest of the pipeline.
    """
    if scaler != "standard":
        raise ValueError(
            f"unsupported scaler {scaler!r}; only 'standard' is available"
        )
    steps = list(build_feature_pipeline(**kwargs).steps)
    steps.append(("scaler", StandardScaler()))
    return Pipeline(steps=steps)


def _prepare_split_inputs(
    occurrence_matrix: pd.DataFrame, labels: pd.DataFrame
) -> tuple[pd.DataFrame, np.ndarray]:
    if "BlockId" not in occurrence_matrix.columns:
        raise ValueError("occurrence_matrix: missing BlockId column")
    if "BlockId" not in labels.columns or "Label" not in labels.columns:
        raise ValueError("labels: expected BlockId and Label columns")
    if occurrence_matrix["BlockId"].duplicated().any():
        raise ValueError("occurrence_matrix: duplicate BlockId values")
    if labels["BlockId"].duplicated().any():
        raise ValueError("labels: duplicate BlockId values")
    missing_events = sorted(set(EVENT_COLUMNS) - set(occurrence_matrix.columns))
    if missing_events:
        raise ValueError(
            f"occurrence_matrix: missing event columns {missing_events}"
        )

    label_map = labels.set_index("BlockId")["Label"]
    y = occurrence_matrix["BlockId"].map(label_map)
    if y.isna().any():
        raise ValueError(
            "labels: missing ground-truth label for "
            f"{int(y.isna().sum())} block(s)"
        )

    X = occurrence_matrix[EVENT_COLUMNS].copy()
    X.index = occurrence_matrix["BlockId"]
    X.index.name = "BlockId"
    return X, y.to_numpy(dtype="int8")


def split_dataset(
    occurrence_matrix: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified train/test split; X contains only the event columns.

    ``occurrence_matrix`` is the loader output (BlockId + E1..E29, plus
    optional Label/Type which are stripped). ``labels`` is the ground-truth
    table (BlockId + Label 0/1). Rows are aligned on BlockId regardless of
    row order; BlockId becomes the index of X and never a feature.

    Nothing is fitted here - fit the pipeline on X_train only.

    Note: this is a plain random stratified split. Rows with identical
    feature vectors MAY appear in both splits, which can inflate held-out
    metrics on datasets with massively repeated block patterns (HDFS_v1).
    Use :func:`grouped_split_dataset` for the leakage-resistant evaluation
    split, where every exact feature vector is confined to one partition.
    """
    X, y_array = _prepare_split_inputs(occurrence_matrix, labels)
    X_train, X_test, y_train_array, y_test_array = train_test_split(
        X,
        y_array,
        test_size=test_size,
        random_state=random_state,
        stratify=y_array if stratify else None,
    )
    y_train = pd.Series(y_train_array, index=X_train.index, name="Label")
    y_test = pd.Series(y_test_array, index=X_test.index, name="Label")
    return X_train, X_test, y_train, y_test


def vector_group_ids(X: pd.DataFrame) -> pd.Series:
    """Deterministic group ID for each complete feature vector.

    Rows with byte-identical event counts share one group ID (SHA-256 of
    the row's values), so a grouped split can never place copies of the
    same feature vector in both train and test. The digest is stable
    across processes, platforms and Python runs.
    """
    values = X.to_numpy(dtype="float64")
    digests = [
        hashlib.sha256(np.ascontiguousarray(row).tobytes()).hexdigest()
        for row in values
    ]
    return pd.Series(digests, index=X.index, name="vector_group_id")


def group_statistics(X: pd.DataFrame, y: Any) -> dict[str, int]:
    """Group-level statistics of the exact feature vectors in ``X``.

    Reports unique vectors, duplicated vectors (appearing more than once),
    duplicate rows (extra copies beyond the first), and feature vectors
    whose rows carry CONFLICTING labels (both 0 and 1). Conflicts are
    reported here, never silently resolved — the grouped split keeps each
    whole vector group in one partition.
    """
    labels = np.asarray(y)
    if len(labels) != len(X):
        raise ValueError(f"y has {len(labels)} rows but X has {len(X)}")
    groups = vector_group_ids(X)
    counts = (
        pd.DataFrame({"group": groups.to_numpy(), "label": labels})
        .groupby("group")["label"]
        .agg(["count", "nunique"])
    )
    conflicting = counts[counts["nunique"] > 1]
    return {
        "total_rows": int(len(X)),
        "unique_vectors": int(len(counts)),
        "duplicated_vectors": int((counts["count"] > 1).sum()),
        "duplicate_rows": int((counts["count"] - 1).clip(lower=0).sum()),
        "conflicting_label_vectors": int(len(conflicting)),
        "conflicting_label_rows": int(conflicting["count"].sum()),
    }


def feature_vector_overlap(X_reference: pd.DataFrame, X_candidate: pd.DataFrame) -> int:
    """Rows in ``X_candidate`` whose exact feature vector appears in
    ``X_reference``."""
    reference_groups = set(vector_group_ids(X_reference))
    return int(vector_group_ids(X_candidate).isin(reference_groups).sum())


def grouped_split_dataset(
    occurrence_matrix: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    n_splits: int = 5,
    random_state: int = 42,
    test_fold: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Leakage-resistant grouped train/test split.

    Every row is assigned a deterministic group ID equal to the hash of
    its complete feature vector (:func:`vector_group_ids`);
    :class:`~sklearn.model_selection.StratifiedGroupKFold` then assigns
    WHOLE groups to folds, so no exact feature vector ever appears in
    both train and test — verified here with a hard check. Class
    proportions are preserved as far as group granularity allows.
    Groups with conflicting labels are kept intact in one partition and
    reported by :func:`group_statistics`; they are never split apart.

    Returns the same 4-tuple as :func:`split_dataset`. ``n_splits``
    controls the group granularity of the partition (``test_fold`` is
    the held-out fold, ~1/``n_splits`` of the rows).
    """
    X, y_array = _prepare_split_inputs(occurrence_matrix, labels)
    groups = vector_group_ids(X)
    splitter = StratifiedGroupKFold(
        n_splits=n_splits, shuffle=True, random_state=random_state
    )
    train_idx, test_idx = list(
        splitter.split(X, y_array, groups.to_numpy())
    )[test_fold]
    shared = set(groups.iloc[train_idx]) & set(groups.iloc[test_idx])
    if shared:
        raise RuntimeError(
            f"grouped split produced {len(shared)} shared exact feature "
            "vector(s) between train and test — split is invalid"
        )
    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]
    y_train = pd.Series(y_array[train_idx], index=X_train.index, name="Label")
    y_test = pd.Series(y_array[test_idx], index=X_test.index, name="Label")
    return X_train, X_test, y_train, y_test


def assert_no_label_columns(X: pd.DataFrame) -> None:
    """Raise if any column name resembles a label/identifier column."""
    if not isinstance(X, pd.DataFrame):
        raise TypeError(
            f"expected a pandas DataFrame, got {type(X).__name__}"
        )
    for col in X.columns:
        folded = (
            str(col).casefold().replace("_", "").replace("-", "").replace(" ", "")
        )
        for token in FORBIDDEN_COLUMN_TOKENS:
            if token in folded:
                raise ValueError(
                    f"forbidden column {col!r} in feature matrix (matches "
                    f"token {token!r}): labels and identifiers must never "
                    f"enter X"
                )


def validate_feature_matrix(X: pd.DataFrame) -> None:
    """Raise on label-like columns, missing, non-numeric, or infinite values."""
    assert_no_label_columns(X)
    for col in X.columns:
        series = pd.to_numeric(X[col], errors="coerce")
        if series.isna().any():
            raise ValueError(
                f"feature column {col!r} contains missing or non-numeric values"
            )
        if not np.isfinite(series.to_numpy(dtype="float64")).all():
            raise ValueError(f"feature column {col!r} contains infinite values")


def prepare_features(
    occurrence_matrix: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
    scale: bool = False,
    **pipeline_kwargs: Any,
) -> dict[str, Any]:
    """One-shot recommended flow: split -> fit on train only -> transform.

    Returns X_train/X_test as transformed feature matrices (DataFrame, or
    ndarray when ``scale=True``), the aligned y_train/y_test Series, the
    fitted pipeline, and the ordered feature names. The pipeline is fit on
    the training split only; labels never enter the pipeline.
    """
    X_train, X_test, y_train, y_test = split_dataset(
        occurrence_matrix,
        labels,
        test_size=test_size,
        random_state=random_state,
    )
    if scale:
        pipeline = build_scaled_pipeline(**pipeline_kwargs)
    else:
        pipeline = build_feature_pipeline(**pipeline_kwargs)
    X_train_feat = pipeline.fit_transform(X_train, y=None)
    X_test_feat = pipeline.transform(X_test)
    for name, matrix in (("X_train", X_train_feat), ("X_test", X_test_feat)):
        if isinstance(matrix, pd.DataFrame):
            validate_feature_matrix(matrix)
        elif np.isnan(matrix).any() or not np.isfinite(matrix).all():
            raise ValueError(f"{name}: transformed matrix contains NaN/inf")
    return {
        "X_train": X_train_feat,
        "X_test": X_test_feat,
        "y_train": y_train,
        "y_test": y_test,
        "pipeline": pipeline,
        "feature_names": list(pipeline.get_feature_names_out()),
    }
