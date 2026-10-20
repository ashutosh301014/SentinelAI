"""Unit tests for sentinelai.preprocessing orchestration."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sentinelai.data_loader import EVENT_COLUMNS
from sentinelai.features import DERIVED_FEATURE_NAMES
from sentinelai.preprocessing import (
    assert_no_label_columns,
    build_feature_pipeline,
    feature_vector_overlap,
    group_statistics,
    grouped_split_dataset,
    prepare_features,
    split_dataset,
    validate_feature_matrix,
    vector_group_ids,
)


def make_matrix_labels(
    n: int = 40, n_anomaly: int = 8, seed: int = 0
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    matrix = pd.DataFrame(
        rng.integers(0, 4, size=(n, len(EVENT_COLUMNS))),
        columns=EVENT_COLUMNS,
    )
    matrix.insert(0, "BlockId", [f"blk_{i}" for i in range(n)])
    labels = pd.DataFrame(
        {
            "BlockId": [f"blk_{i}" for i in range(n)],
            "Label": np.array(
                [1] * n_anomaly + [0] * (n - n_anomaly), dtype="int8"
            ),
        }
    )
    return matrix, labels


class TestSplitDataset:
    def test_shapes_and_stratification(self):
        matrix, labels = make_matrix_labels()
        X_train, X_test, y_train, y_test = split_dataset(matrix, labels)
        assert X_train.shape == (32, 29)
        assert X_test.shape == (8, 29)
        assert list(X_train.columns) == EVENT_COLUMNS
        assert y_train.sum() == 6
        assert y_test.sum() == 2
        assert y_train.index.equals(X_train.index)
        assert y_test.index.equals(X_test.index)
        assert X_train.index.name == "BlockId"
        assert set(X_train.index).isdisjoint(set(X_test.index))

    def test_aligns_labels_by_blockid(self):
        matrix, labels = make_matrix_labels()
        shuffled_labels = labels.sample(frac=1, random_state=1).reset_index(
            drop=True
        )
        _, _, y_a, _ = split_dataset(matrix, labels, random_state=42)
        _, _, y_b, _ = split_dataset(matrix, shuffled_labels, random_state=42)
        pd.testing.assert_series_equal(y_a, y_b)

    def test_rejects_missing_labels(self):
        matrix, labels = make_matrix_labels()
        with pytest.raises(ValueError, match="missing ground-truth label"):
            split_dataset(matrix, labels.iloc[:-1])

    def test_rejects_missing_blockid_or_event_column(self):
        matrix, labels = make_matrix_labels()
        with pytest.raises(ValueError, match="missing BlockId"):
            split_dataset(matrix.drop(columns=["BlockId"]), labels)
        with pytest.raises(ValueError, match="missing event columns"):
            split_dataset(matrix.drop(columns=["E29"]), labels)

    def test_rejects_duplicate_blockids(self):
        matrix, labels = make_matrix_labels()
        dup = pd.concat([matrix.iloc[[0]], matrix], ignore_index=True)
        with pytest.raises(ValueError, match="duplicate BlockId"):
            split_dataset(dup, labels)

    def test_no_stratify_option(self):
        matrix, labels = make_matrix_labels()
        X_train, X_test, y_train, y_test = split_dataset(
            matrix, labels, stratify=False
        )
        assert len(X_train) + len(X_test) == 40
        assert len(y_train) + len(y_test) == 40


def make_grouped_matrix_labels(
    n_unique: int = 10,
    copies: int = 4,
    n_anomaly_groups: int = 3,
    conflicting_groups: int = 1,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Synthetic matrix with EXACT duplicate feature vectors.

    ``n_unique`` distinct 29-event vectors, each repeated ``copies``
    times. The first ``n_anomaly_groups`` groups are labelled anomalous,
    and in the first ``conflicting_groups`` of those, one copy is
    relabelled normal — producing groups whose rows carry CONFLICTING
    labels.
    """
    rng = np.random.default_rng(seed)
    vectors = rng.integers(0, 4, size=(n_unique, len(EVENT_COLUMNS)))
    rows = np.repeat(vectors, copies, axis=0)
    n = n_unique * copies
    matrix = pd.DataFrame(rows, columns=EVENT_COLUMNS)
    matrix.insert(0, "BlockId", [f"blk_{i}" for i in range(n)])
    labels = np.zeros(n, dtype="int8")
    labels[: n_anomaly_groups * copies] = 1
    for group in range(conflicting_groups):
        labels[group * copies] = 0
    label_frame = pd.DataFrame(
        {"BlockId": [f"blk_{i}" for i in range(n)], "Label": labels}
    )
    return matrix, label_frame

class TestGroupedSplit:
    def test_group_ids_deterministic_and_equal_for_identical_vectors(self):
        copies = 4
        matrix, labels = make_grouped_matrix_labels(copies=copies)

        X = matrix.set_index("BlockId")[EVENT_COLUMNS]

        ids_a = vector_group_ids(X)
        ids_b = vector_group_ids(X)

        pd.testing.assert_series_equal(ids_a, ids_b)

        # Copies of the same vector must have identical group IDs.
        assert ids_a.iloc[0] == ids_a.iloc[1]

        # First row of the next unique vector must have another group ID.
        assert ids_a.iloc[0] != ids_a.iloc[copies]

    def test_group_statistics_reports_duplicates_and_conflicts(self):
        matrix, labels = make_grouped_matrix_labels()

        X = matrix.set_index("BlockId")[EVENT_COLUMNS]
        y = matrix["BlockId"].map(labels.set_index("BlockId")["Label"])

        stats = group_statistics(X, y)

        assert stats["total_rows"] == 40
        assert stats["unique_vectors"] == 10
        assert stats["duplicated_vectors"] == 10
        assert stats["duplicate_rows"] == 30
        assert stats["conflicting_label_vectors"] == 1
        assert stats["conflicting_label_rows"] == 4

    def test_grouped_split_has_zero_exact_vector_overlap(self):
        matrix, labels = make_grouped_matrix_labels()

        X_train, X_test, y_train, y_test = grouped_split_dataset(
            matrix,
            labels,
            n_splits=5,
            random_state=42,
        )

        assert feature_vector_overlap(X_train, X_test) == 0

        assert len(X_train) + len(X_test) == 40
        assert len(y_train) + len(y_test) == 40

        assert set(X_train.index).isdisjoint(set(X_test.index))

        assert y_train.index.equals(X_train.index)
        assert y_test.index.equals(X_test.index)

    def test_grouped_split_is_deterministic(self):
        matrix, labels = make_grouped_matrix_labels()

        a = grouped_split_dataset(
            matrix,
            labels,
            n_splits=5,
            random_state=42,
        )

        b = grouped_split_dataset(
            matrix,
            labels,
            n_splits=5,
            random_state=42,
        )

        # X_train and X_test
        pd.testing.assert_frame_equal(a[0], b[0])
        pd.testing.assert_frame_equal(a[1], b[1])

        # y_train and y_test
        pd.testing.assert_series_equal(a[2], b[2])
        pd.testing.assert_series_equal(a[3], b[3])

    def test_grouped_split_keeps_each_vector_group_in_one_partition(self):
        matrix, labels = make_grouped_matrix_labels()

        X_train, X_test, _, _ = grouped_split_dataset(
            matrix,
            labels,
            n_splits=5,
            random_state=42,
        )

        train_groups = set(vector_group_ids(X_train))
        test_groups = set(vector_group_ids(X_test))

        assert train_groups.isdisjoint(test_groups)

        # All 10 unique vectors must exist entirely in one partition.
        assert len(train_groups) + len(test_groups) == 10

    def test_grouped_split_preserves_both_classes_in_train(self):
        matrix, labels = make_grouped_matrix_labels()

        _, _, y_train, _ = grouped_split_dataset(
            matrix,
            labels,
            n_splits=5,
            random_state=42,
        )

        assert (y_train == 1).any()
        assert (y_train == 0).any()

    def test_grouped_split_rejects_missing_labels(self):
        matrix, labels = make_grouped_matrix_labels()

        with pytest.raises(ValueError, match="missing ground-truth label"):
            grouped_split_dataset(
                matrix,
                labels.iloc[:-1],
                n_splits=5,
            )

    def test_grouped_split_fails_on_too_few_groups(self):
        matrix, labels = make_grouped_matrix_labels(n_unique=3)

        with pytest.raises(ValueError):
            grouped_split_dataset(
                matrix,
                labels,
                n_splits=5,
            )


class TestValidationGuards:
    def test_assert_no_label_columns_passes_on_events(self):
        frame = pd.DataFrame(np.zeros((3, 3)), columns=["E1", "E2", "E3"])
        assert_no_label_columns(frame)

    @pytest.mark.parametrize("bad", ["Label", "label", "Type", "BlockId", "block_id"])
    def test_assert_no_label_columns_rejects(self, bad):
        frame = pd.DataFrame(np.zeros((3, 3)), columns=["E1", bad, "E3"])
        with pytest.raises(ValueError, match="forbidden column"):
            assert_no_label_columns(frame)

    def test_validate_feature_matrix_passes_on_clean(self):
        frame = build_feature_pipeline().fit_transform(
            pd.DataFrame(
                {
                    "E1": [5, 1, 3],
                    "E2": [1, 1, 1],
                    "E3": [5, 1, 3],
                    "E4": [1, 0, 0],
                }
            )
        )
        validate_feature_matrix(frame)

    def test_validate_feature_matrix_rejects_nan_inf_strings(self):
        with pytest.raises(ValueError, match="missing or non-numeric"):
            validate_feature_matrix(pd.DataFrame({"E1": [1.0, np.nan]}))
        with pytest.raises(ValueError, match="infinite"):
            validate_feature_matrix(pd.DataFrame({"E1": [1.0, np.inf]}))
        with pytest.raises(ValueError, match="missing or non-numeric"):
            validate_feature_matrix(pd.DataFrame({"E1": [1, "bad"]}))

    def test_validate_feature_matrix_rejects_label_column(self):
        with pytest.raises(ValueError, match="forbidden column"):
            validate_feature_matrix(pd.DataFrame({"E1": [1.0], "Label": [0]}))


class TestPrepareFeatures:
    def test_bundle_contract(self):
        matrix, labels = make_matrix_labels()
        out = prepare_features(matrix, labels, random_state=42)
        assert set(out) == {
            "X_train",
            "X_test",
            "y_train",
            "y_test",
            "pipeline",
            "feature_names",
        }
        assert isinstance(out["X_train"], pd.DataFrame)
        assert out["X_train"].shape == (32, 29 + len(DERIVED_FEATURE_NAMES))
        assert out["X_test"].shape == (8, 29 + len(DERIVED_FEATURE_NAMES))
        assert list(out["X_train"].columns) == out["feature_names"]
        assert list(out["X_test"].columns) == out["feature_names"]
        assert out["X_train"].notna().all().all()
        assert out["X_test"].notna().all().all()
        assert len(out["y_train"]) == 32
        assert len(out["y_test"]) == 8

    def test_deterministic(self):
        matrix, labels = make_matrix_labels()
        out1 = prepare_features(matrix, labels, random_state=42)
        out2 = prepare_features(matrix, labels, random_state=42)
        pd.testing.assert_frame_equal(out1["X_train"], out2["X_train"])
        pd.testing.assert_frame_equal(out1["X_test"], out2["X_test"])
        pd.testing.assert_series_equal(out1["y_train"], out2["y_train"])

    def test_scale_variant(self):
        matrix, labels = make_matrix_labels()
        out = prepare_features(matrix, labels, random_state=42, scale=True)
        assert isinstance(out["X_train"], np.ndarray)
        assert out["X_train"].shape == (32, 29 + len(DERIVED_FEATURE_NAMES))
        assert np.isfinite(out["X_train"]).all()
        assert np.isfinite(out["X_test"]).all()
        np.testing.assert_allclose(out["X_train"].mean(axis=0), 0.0, atol=1e-9)
        assert out["feature_names"] == list(
            out["pipeline"].get_feature_names_out()
        )
