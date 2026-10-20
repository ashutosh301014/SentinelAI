"""Integration tests: feature pipeline contract on the real HDFS_v1 data.

These tests load the actual occurrence matrix and ground-truth labels and
verify the production contract end to end. They fail loudly if the raw data
files are missing - real data is never substituted.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sentinelai.data_loader import load_anomaly_labels, load_occurrence_matrix
from sentinelai.features import DERIVED_FEATURE_NAMES
from sentinelai.preprocessing import (
    build_feature_pipeline,
    prepare_features,
    split_dataset,
    validate_feature_matrix,
)

N_BLOCKS = 575_061
N_FEATURES = 27 + len(DERIVED_FEATURE_NAMES)  # 29 events - E22 - E25 + 8


@pytest.fixture(scope="module")
def hdfs() -> tuple[pd.DataFrame, pd.DataFrame]:
    return load_occurrence_matrix(), load_anomaly_labels()


def test_real_pipeline_contract(hdfs):
    matrix, labels = hdfs
    X_train, X_test, y_train, y_test = split_dataset(matrix, labels)
    assert len(X_train) + len(X_test) == N_BLOCKS
    assert len(y_test) == 115_013  # sklearn ceil(0.2 * N_BLOCKS)
    assert y_test.mean() == pytest.approx(y_train.mean(), abs=0.002)

    pipeline = build_feature_pipeline()
    X_train_feat = pipeline.fit_transform(X_train)
    X_test_feat = pipeline.transform(X_test)
    select = pipeline.named_steps["select"]
    derive = pipeline.named_steps["derive"]

    # EDA-verified drops: E22 constant, E25 exact duplicate of E18
    assert select.constant_columns_ == ["E22"]
    assert select.duplicate_columns_ == ["E25"]

    # expected dimensions and identical feature ordering for train/inference
    assert X_train_feat.shape == (len(X_train), N_FEATURES)
    assert X_test_feat.shape == (len(X_test), N_FEATURES)
    assert list(X_train_feat.columns) == list(X_test_feat.columns)
    assert list(X_train_feat.columns) == pipeline.get_feature_names_out().tolist()

    # no NaN, no inf, labels never in features
    assert X_train_feat.notna().all().all()
    assert X_test_feat.notna().all().all()
    assert np.isfinite(X_train_feat.to_numpy()).all()
    assert np.isfinite(X_test_feat.to_numpy()).all()
    validate_feature_matrix(X_train_feat)
    validate_feature_matrix(X_test_feat)
    for col in X_train_feat.columns:
        assert col not in {"Label", "Type", "BlockId"}

    # deterministic: transforming train again reproduces the fit output
    pd.testing.assert_frame_equal(pipeline.transform(X_train), X_train_feat)

    # rare set is learned from the training split only, never the full data
    presence = (X_train[select.kept_event_columns_] > 0).mean()
    manual_rare = set(
        presence[presence < derive.rare_presence_threshold].index
    )
    assert set(derive.rare_event_columns_) == manual_rare
    assert "E1" in derive.rare_event_columns_
    assert "E5" not in derive.rare_event_columns_


def test_real_unseen_inference_row(hdfs):
    matrix, labels = hdfs
    X_train, X_test, _, _ = split_dataset(matrix, labels)
    pipeline = build_feature_pipeline().fit(X_train)
    X_test_feat = pipeline.transform(X_test)

    row = X_test.iloc[[0]]
    row_feat = pipeline.transform(row)
    assert list(row_feat.columns) == list(X_test_feat.columns)
    pd.testing.assert_frame_equal(row_feat, X_test_feat.loc[[row.index[0]]])

    zero_row = pd.DataFrame(
        {col: [0] for col in X_train.columns}, index=["blk_unseen"]
    )
    zero_feat = pipeline.transform(zero_row)
    assert zero_feat.notna().all().all()
    assert zero_feat["total_event_count"].iloc[0] == 0
    assert zero_feat["event_entropy"].iloc[0] == 0


def test_real_prepare_features_bundle(hdfs):
    matrix, labels = hdfs
    out = prepare_features(matrix, labels, random_state=42)
    assert out["X_train"].shape[1] == N_FEATURES
    assert out["X_test"].shape[1] == N_FEATURES
    assert len(out["feature_names"]) == N_FEATURES
    assert len(out["X_train"]) == len(out["y_train"])
    assert len(out["X_test"]) == len(out["y_test"])
    assert out["X_train"].notna().all().all()
    assert out["X_test"].notna().all().all()

    scaled = prepare_features(matrix, labels, random_state=42, scale=True)
    assert scaled["X_train"].shape == (len(out["X_train"]), N_FEATURES)
    assert np.isfinite(scaled["X_train"]).all()
    assert np.isfinite(scaled["X_test"]).all()
    assert scaled["feature_names"] == out["feature_names"]
