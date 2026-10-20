"""Unit tests for sentinelai.features and the feature pipeline."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_is_fitted

from sentinelai.features import (
    DERIVED_FEATURE_NAMES,
    DerivedFeatureExtractor,
    EventFeatureSelector,
)
from sentinelai.preprocessing import (
    assert_no_label_columns,
    build_feature_pipeline,
    build_scaled_pipeline,
    validate_feature_matrix,
)


def make_event_frame(n: int = 40, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    e1 = rng.integers(0, 6, size=n)
    return pd.DataFrame(
        {
            "E1": e1,
            "E2": np.ones(n, dtype="int64"),  # constant -> dropped
            "E3": e1.copy(),  # exact duplicate of E1 -> dropped
            "E4": np.where(np.arange(n) == 0, 1, 0),  # sparse
            "E5": rng.integers(0, 4, size=n),
            "E6": rng.integers(0, 3, size=n),
        }
    )


EXPECTED_KEPT = ["E1", "E4", "E5", "E6"]
EXPECTED_COLUMNS = EXPECTED_KEPT + list(DERIVED_FEATURE_NAMES)


class TestEventFeatureSelector:
    def test_drops_constant_and_exact_duplicates(self):
        selector = EventFeatureSelector().fit(make_event_frame())
        assert selector.kept_event_columns_ == EXPECTED_KEPT
        assert selector.constant_columns_ == ["E2"]
        assert selector.duplicate_columns_ == ["E3"]

    def test_keeps_near_duplicate(self):
        frame = make_event_frame()
        frame.loc[0, "E3"] += 1
        selector = EventFeatureSelector().fit(frame)
        assert "E3" in selector.kept_event_columns_
        assert selector.duplicate_columns_ == []

    def test_rejects_non_event_columns(self):
        frame = make_event_frame().assign(Label=0)
        with pytest.raises(ValueError, match="only event columns"):
            EventFeatureSelector().fit(frame)

    def test_rejects_duplicate_column_names(self):
        frame = make_event_frame()
        frame.columns = ["E1", "E2", "E3", "E4", "E5", "E1"]
        with pytest.raises(ValueError, match="duplicate column"):
            EventFeatureSelector().fit(frame)

    def test_rejects_missing_column_at_transform(self):
        selector = EventFeatureSelector().fit(make_event_frame())
        with pytest.raises(ValueError, match="missing"):
            selector.transform(make_event_frame().drop(columns=["E2"]))

    def test_rejects_unseen_column_at_transform(self):
        selector = EventFeatureSelector().fit(make_event_frame())
        frame = make_event_frame().assign(E7=0)
        with pytest.raises(ValueError, match="unseen"):
            selector.transform(frame)

    def test_rejects_nan_input(self):
        frame = make_event_frame()
        frame.loc[3, "E5"] = np.nan
        with pytest.raises(ValueError, match="missing or non-numeric"):
            EventFeatureSelector().fit(frame)

    def test_output_order_independent_of_input_order(self):
        frame = make_event_frame()
        selector = EventFeatureSelector().fit(frame)
        shuffled = frame[["E6", "E3", "E1", "E5", "E4", "E2"]]
        out = selector.transform(shuffled)
        assert list(out.columns) == EXPECTED_KEPT
        pd.testing.assert_frame_equal(out, selector.transform(frame))

    def test_rejects_non_dataframe(self):
        with pytest.raises(TypeError, match="DataFrame"):
            EventFeatureSelector().fit(make_event_frame().to_numpy())


class TestDerivedFeatureExtractor:
    def test_known_values(self):
        frame = pd.DataFrame(
            {"E1": [3, 0, 0], "E2": [2, 4, 0]}
        )
        extractor = DerivedFeatureExtractor(rare_presence_threshold=0.4)
        out = extractor.fit_transform(frame)
        row0, row1, row2 = out.iloc[0], out.iloc[1], out.iloc[2]

        assert row0["total_event_count"] == 5
        assert row0["unique_event_count"] == 2
        assert row0["dominant_event_ratio"] == pytest.approx(0.6)
        assert row0["repeated_event_ratio"] == pytest.approx(0.6)
        assert row0["event_entropy"] == pytest.approx(0.9709506, abs=1e-6)
        assert row0["zero_event_ratio"] == pytest.approx(0.0)
        assert row0["maximum_single_event_frequency"] == pytest.approx(0.6)
        assert row0["rare_event_count"] == 1  # E1 present in 1/3 rows (< 0.4)

        assert row1["total_event_count"] == 4
        assert row1["unique_event_count"] == 1
        assert row1["dominant_event_ratio"] == pytest.approx(1.0)
        assert row1["repeated_event_ratio"] == pytest.approx(0.75)
        assert row1["event_entropy"] == pytest.approx(0.0)
        assert row1["zero_event_ratio"] == pytest.approx(0.5)
        assert row1["rare_event_count"] == 0

        assert row2["total_event_count"] == 0
        assert row2["dominant_event_ratio"] == pytest.approx(0.0)
        assert row2["repeated_event_ratio"] == pytest.approx(0.0)
        assert row2["event_entropy"] == pytest.approx(0.0)
        assert row2["zero_event_ratio"] == pytest.approx(1.0)
        assert row2["rare_event_count"] == 0

    def test_rare_set_learned_at_fit_not_global(self):
        common = pd.DataFrame({"E1": [1] * 30, "E2": [0] * 30})
        sparse = pd.DataFrame(
            {"E1": np.where(np.arange(30) == 0, 1, 0), "E2": np.ones(30, dtype="int64")}
        )
        row = pd.DataFrame({"E1": [1], "E2": [0]})
        fitted_on_common = DerivedFeatureExtractor(
            rare_presence_threshold=0.05
        ).fit(common)
        fitted_on_sparse = DerivedFeatureExtractor(
            rare_presence_threshold=0.05
        ).fit(sparse)
        assert fitted_on_common.transform(row)["rare_event_count"].iloc[0] == 0
        assert fitted_on_sparse.transform(row)["rare_event_count"].iloc[0] == 1

    def test_rejects_invalid_threshold(self):
        for bad in (-0.1, 1.0, 1.5):
            with pytest.raises(ValueError, match="rare_presence_threshold"):
                DerivedFeatureExtractor(rare_presence_threshold=bad)

    def test_feature_names_out(self):
        extractor = DerivedFeatureExtractor().fit(make_event_frame())
        names = extractor.get_feature_names_out().tolist()
        assert names == ["E1", "E2", "E3", "E4", "E5", "E6"] + list(
            DERIVED_FEATURE_NAMES
        )


class TestFeaturePipeline:
    def test_expected_output_dimensions(self):
        pipeline = build_feature_pipeline()
        out = pipeline.fit_transform(make_event_frame())
        assert out.shape == (40, len(EXPECTED_COLUMNS))
        assert list(out.columns) == EXPECTED_COLUMNS
        assert pipeline.get_feature_names_out().tolist() == EXPECTED_COLUMNS

    def test_deterministic_transformation(self):
        frame = make_event_frame()
        pipeline = build_feature_pipeline()
        out1 = pipeline.fit_transform(frame)
        out2 = pipeline.transform(frame)
        pd.testing.assert_frame_equal(out1, out2)

        refit = build_feature_pipeline().fit_transform(frame)
        pd.testing.assert_frame_equal(out1, refit)

    def test_stable_feature_order_train_and_test(self):
        frame = make_event_frame()
        y = np.tile([0, 1], len(frame) // 2)
        X_train, X_test = train_test_split(
            frame, test_size=0.25, random_state=7, stratify=y
        )
        pipeline = build_feature_pipeline()
        train_out = pipeline.fit_transform(X_train)
        test_out = pipeline.transform(X_test)
        assert list(train_out.columns) == list(test_out.columns) == EXPECTED_COLUMNS

    def test_no_nan_no_inf(self):
        pipeline = build_feature_pipeline()
        out = pipeline.fit_transform(make_event_frame())
        assert out.notna().all().all()
        assert np.isfinite(out.to_numpy(dtype="float64")).all()

    def test_known_row_through_pipeline(self):
        frame = pd.DataFrame(
            {
                "E1": [5, 1, 3],
                "E2": [1, 1, 1],  # constant -> dropped
                "E3": [5, 1, 3],  # duplicate -> dropped
                "E4": [1, 0, 0],  # sparse -> rare at threshold 0.5
            }
        )
        pipeline = build_feature_pipeline(rare_presence_threshold=0.5)
        out = pipeline.fit_transform(frame)
        assert list(out.columns) == ["E1", "E4"] + list(DERIVED_FEATURE_NAMES)

        row = out.iloc[0]
        assert row["E1"] == 5
        assert row["E4"] == 1
        assert row["total_event_count"] == 6
        assert row["unique_event_count"] == 2
        assert row["dominant_event_ratio"] == pytest.approx(5 / 6)
        assert row["repeated_event_ratio"] == pytest.approx(4 / 6)
        assert row["event_entropy"] == pytest.approx(0.6500224, abs=1e-5)
        assert row["rare_event_count"] == 1
        assert row["zero_event_ratio"] == pytest.approx(0.0)  # 0 of 2 kept types
        assert row["maximum_single_event_frequency"] == pytest.approx(5 / 6)

        row2 = out.iloc[2]
        assert row2["total_event_count"] == 3
        assert row2["dominant_event_ratio"] == pytest.approx(1.0)
        assert row2["event_entropy"] == pytest.approx(0.0)
        assert row2["zero_event_ratio"] == pytest.approx(0.5)
        assert row2["rare_event_count"] == 0

    def test_unseen_inference_sample(self):
        frame = make_event_frame()
        pipeline = build_feature_pipeline().fit(frame)
        batch = pipeline.transform(frame)

        existing = frame.iloc[[5]]
        assert pipeline.transform(existing).columns.tolist() == EXPECTED_COLUMNS
        pd.testing.assert_frame_equal(pipeline.transform(existing), batch.loc[[5]])

        zero_row = pd.DataFrame({f"E{i}": [0] for i in range(1, 7)})
        transformed = pipeline.transform(zero_row)
        assert transformed.columns.tolist() == EXPECTED_COLUMNS
        assert transformed.notna().all().all()
        assert transformed["total_event_count"].iloc[0] == 0
        assert transformed["event_entropy"].iloc[0] == 0
        assert transformed["dominant_event_ratio"].iloc[0] == 0
        assert transformed["zero_event_ratio"].iloc[0] == 1.0

    def test_labels_never_enter_features(self):
        pipeline = build_feature_pipeline()
        out = pipeline.fit_transform(make_event_frame())
        assert_no_label_columns(out)
        with pytest.raises(ValueError, match="only event columns"):
            pipeline.fit(make_event_frame().assign(Label=0))
        with pytest.raises(ValueError, match="only event columns"):
            pipeline.fit(make_event_frame().assign(BlockId="blk_1"))

    def test_unfitted_until_fit(self):
        with pytest.raises(NotFittedError):
            check_is_fitted(build_feature_pipeline())

    def test_fit_ignores_labels(self):
        frame = make_event_frame()
        y = np.random.default_rng(3).integers(0, 2, size=len(frame))
        plain = build_feature_pipeline().fit(frame)
        with_labels = build_feature_pipeline().fit(frame, y=y)
        pd.testing.assert_frame_equal(
            plain.transform(frame), with_labels.transform(frame)
        )
        assert plain["select"].kept_event_columns_ == with_labels[
            "select"
        ].kept_event_columns_
        assert plain["derive"].rare_event_columns_ == with_labels[
            "derive"
        ].rare_event_columns_

    def test_default_pipeline_is_unscaled(self):
        frame = make_event_frame()
        out = build_feature_pipeline().fit_transform(frame)
        assert isinstance(out, pd.DataFrame)
        expected_total = frame[EXPECTED_KEPT].sum(axis=1).to_numpy()
        np.testing.assert_allclose(out["total_event_count"], expected_total)

    def test_scaled_pipeline_only_on_request(self):
        frame = make_event_frame()
        out = build_scaled_pipeline().fit_transform(frame)
        assert isinstance(out, np.ndarray)
        assert out.shape == (40, len(EXPECTED_COLUMNS))
        assert np.isfinite(out).all()
        unscaled = build_feature_pipeline().fit_transform(frame)
        constant = unscaled.columns[unscaled.nunique(dropna=False) <= 1]
        means = out.mean(axis=0)
        stds = out.std(axis=0)
        np.testing.assert_allclose(means, 0.0, atol=1e-9)
        np.testing.assert_allclose(stds[~unscaled.columns.isin(constant)], 1.0, atol=1e-6)
        np.testing.assert_allclose(stds[unscaled.columns.isin(constant)], 0.0)

    def test_scaled_pipeline_rejects_unknown_scaler(self):
        with pytest.raises(ValueError, match="unsupported scaler"):
            build_scaled_pipeline(scaler="minmax")
