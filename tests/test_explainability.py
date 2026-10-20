"""Unit tests for sentinelai.explainability (toy pipelines only)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import sentinelai.explainability as explainability
from sentinelai.explainability import (
    EvidenceProfile,
    build_training_profile,
    explain_prediction,
    global_feature_importance,
    global_shap_importance,
    local_feature_contributions,
    map_event_features,
    pipeline_feature_names,
    top_evidence,
)
from sentinelai.modeling import build_classifier_pipeline


def make_toy_frame(
    n: int = 300, n_anomaly: int = 60, seed: int = 0
) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(
        rng.integers(0, 4, size=(n, 6)),
        columns=[f"E{i}" for i in range(1, 7)],
    )
    anomaly_idx = rng.choice(n, size=n_anomaly, replace=False)
    frame.loc[anomaly_idx, "E1"] = rng.integers(20, 50, size=n_anomaly)
    frame.loc[anomaly_idx, "E3"] = rng.integers(10, 30, size=n_anomaly)
    y = pd.Series(np.zeros(n, dtype="int8"), name="Label")
    y.iloc[anomaly_idx] = 1
    return frame, y


@pytest.fixture(scope="module")
def toy_pipeline():
    frame, y = make_toy_frame()
    pipeline = build_classifier_pipeline(
        "hist_gradient_boosting", random_state=0
    )
    pipeline.fit(frame, y)
    return pipeline


@pytest.fixture(scope="module")
def toy_data():
    frame, y = make_toy_frame()
    anomaly_idx = np.where(y.to_numpy() == 1)[0]
    normal_idx = np.where(y.to_numpy() == 0)[0]
    held_idx = np.concatenate([anomaly_idx[::2], normal_idx[::3]])
    return frame, y, held_idx


TOY_TEMPLATES = pd.DataFrame(
    {
        "EventId": ["E1", "E2", "E3"],
        "EventTemplate": [
            "Receiving block <*> of size <*>",
            "Verification succeeded for <*>",
            "PacketResponder <*> for block <*> terminating",
        ],
    }
)


class TestGlobalImportance:
    def test_permutation_importance_on_raw_columns(
        self, toy_pipeline, toy_data
    ):
        frame, y, held_idx = toy_data
        held_x = frame.iloc[held_idx]
        importance = global_feature_importance(
            toy_pipeline,
            held_x,
            y.iloc[held_idx],
            n_repeats=2,
            random_state=0,
        )
        assert list(importance.columns) == [
            "feature",
            "importance_mean",
            "importance_std",
        ]
        assert list(importance["feature"]) == list(held_x.columns)
        assert np.isfinite(importance["importance_mean"]).all()
        assert np.isfinite(importance["importance_std"]).all()
        assert importance["importance_mean"].is_monotonic_decreasing

    def test_shap_importance_covers_model_features(
        self, toy_pipeline, toy_data
    ):
        frame, _, held_idx = toy_data
        importance = global_shap_importance(
            toy_pipeline, frame.iloc[held_idx[:40]]
        )
        assert set(importance["feature"]) == set(
            pipeline_feature_names(toy_pipeline)
        )
        assert (importance["importance_mean"] >= 0).all()
        assert importance["importance_mean"].is_monotonic_decreasing

    def test_rejects_label_columns(self, toy_pipeline, toy_data):
        frame, y, held_idx = toy_data
        polluted = frame.iloc[held_idx].copy()
        polluted["Label"] = y.iloc[held_idx].to_numpy()
        with pytest.raises(ValueError, match="[Ll]abel"):
            global_feature_importance(
                toy_pipeline, polluted, y.iloc[held_idx], n_repeats=1
            )


class TestLocalContributions:
    def test_reconstructs_model_output(self, toy_pipeline, toy_data):
        frame, _, held_idx = toy_data
        sample = frame.iloc[held_idx[:8]]
        expected, contributions = local_feature_contributions(
            toy_pipeline, sample
        )
        assert contributions.shape == (len(sample), len(
            pipeline_feature_names(toy_pipeline)
        ))
        assert list(contributions.columns) == pipeline_feature_names(
            toy_pipeline
        )
        probabilities = toy_pipeline.predict_proba(sample)[:, 1]
        margins = expected + contributions.sum(axis=1).to_numpy()
        reconstructed = 1.0 / (1.0 + np.exp(-margins))
        np.testing.assert_allclose(reconstructed, probabilities, atol=1e-6)

    def test_requires_tree_model(self, toy_data):
        frame, y, held_idx = toy_data
        logistic = build_classifier_pipeline(
            "logistic_regression", random_state=0
        ).fit(frame.iloc[:200], y.iloc[:200])
        with pytest.raises(ValueError, match="tree-based"):
            local_feature_contributions(
                logistic, frame.iloc[held_idx[:2]]
            )


class TestEventMapping:
    def test_maps_event_features_to_templates(self):
        mapping = map_event_features(
            TOY_TEMPLATES,
            ["E1", "E2", "total_event_count", "E99"],
        )
        assert mapping == {
            "E1": "Receiving block <*> of size <*>",
            "E2": "Verification succeeded for <*>",
            "total_event_count": None,
            "E99": None,
        }

    def test_handles_missing_templates(self):
        mapping = map_event_features(None, ["E1", "total_event_count"])
        assert mapping == {"E1": None, "total_event_count": None}


class TestTrainingProfile:
    def test_percentile_order_and_index(self, toy_pipeline, toy_data):
        frame, y, _ = toy_data
        normal = frame[y.to_numpy() == 0]
        profile = build_training_profile(toy_pipeline, normal)
        assert list(profile.percentiles.columns) == ["p5", "median", "p95"]
        assert list(profile.percentiles.index) == pipeline_feature_names(
            toy_pipeline
        )
        assert (
            profile.percentiles["p5"] <= profile.percentiles["median"]
        ).all()
        assert (
            profile.percentiles["median"] <= profile.percentiles["p95"]
        ).all()

    def test_rare_event_columns_from_extractor(
        self, toy_pipeline, toy_data
    ):
        frame, y, _ = toy_data
        profile = build_training_profile(
            toy_pipeline, frame[y.to_numpy() == 0]
        )
        assert profile.rare_event_columns_ == tuple(
            toy_pipeline.named_steps["derive"].rare_event_columns_
        )


class TestTopEvidence:
    def _profile(self, index, p5=0.0, p95=5.0, rare=()):
        return EvidenceProfile(
            percentiles=pd.DataFrame(
                {"p5": [p5], "median": [0.0], "p95": [p95]},
                index=[index],
            ),
            rare_event_columns_=tuple(rare),
        )

    def test_abnormal_frequency_phrase(self, toy_pipeline, toy_data):
        frame, _, _ = toy_data
        sample = frame.iloc[[0]].copy()
        sample["E1"] = 8
        profile = self._profile("E1", p95=5.0)
        evidence = top_evidence(profile, toy_pipeline, sample)
        assert evidence == ["E1 abnormal frequency"]

    def test_unusually_repeated_phrase(self, toy_pipeline, toy_data):
        frame, _, _ = toy_data
        sample = frame.iloc[[0]].copy()
        sample["E1"] = 50
        profile = self._profile("E1", p95=5.0)
        evidence = top_evidence(profile, toy_pipeline, sample)
        assert evidence == ["E1 unusually repeated"]

    def test_rare_event_phrase_takes_precedence(
        self, toy_pipeline, toy_data
    ):
        frame, _, _ = toy_data
        sample = frame.iloc[[0]].copy()
        sample["E1"] = 50
        profile = self._profile("E1", p95=5.0, rare=("E1",))
        evidence = top_evidence(profile, toy_pipeline, sample)
        assert evidence == ["E1 rare event occurred"]

    def test_normal_sample_gets_typical_phrases(
        self, toy_pipeline, toy_data
    ):
        frame, _, _ = toy_data
        sample = frame.iloc[[0]]
        profile = self._profile("E1", p95=1e9)
        evidence = top_evidence(profile, toy_pipeline, sample)
        assert evidence == [
            "all event frequencies within normal range",
            "typical event pattern",
        ]

    def test_top_k_limits_evidence(self, toy_pipeline, toy_data):
        frame, _, _ = toy_data
        sample = frame.iloc[[0]].copy()
        for col in ("E1", "E2", "E3", "E4"):
            sample[col] = 100
        profile = EvidenceProfile(
            percentiles=pd.DataFrame(
                {"p5": [0.0] * 4, "median": [0.0] * 4, "p95": [1.0] * 4},
                index=["E1", "E2", "E3", "E4"],
            ),
            rare_event_columns_=(),
        )
        evidence = top_evidence(profile, toy_pipeline, sample, top_k=2)
        assert len(evidence) == 2

    def test_derived_feature_phrases(self, toy_pipeline, toy_data):
        frame, y, _ = toy_data
        normal = frame[y.to_numpy() == 0]
        profile = build_training_profile(toy_pipeline, normal)
        sample = frame.iloc[[0]].copy()
        for col in frame.columns:
            sample[col] = 500
        evidence = top_evidence(profile, toy_pipeline, sample, top_k=12)
        assert any("abnormal total event count" in e for e in evidence)
        assert any("high repeated-event ratio" in e for e in evidence)

    def test_deterministic(self, toy_pipeline, toy_data):
        frame, _, _ = toy_data
        sample = frame.iloc[[0]].copy()
        sample["E1"] = 50
        profile = self._profile("E1", p95=5.0)
        first = top_evidence(profile, toy_pipeline, sample)
        second = top_evidence(profile, toy_pipeline, sample)
        assert first == second


class TestExplainPrediction:
    def test_bundle_shape(self, toy_pipeline, toy_data):
        frame, y, _ = toy_data
        normal = frame[y.to_numpy() == 0]
        profile = build_training_profile(toy_pipeline, normal)
        sample = frame.iloc[[0]]
        bundle = explain_prediction(
            toy_pipeline, profile, TOY_TEMPLATES, sample
        )
        assert set(bundle) == {
            "block_id",
            "predicted_label",
            "anomaly_probability",
            "top_evidence",
            "top_contributions",
        }
        assert bundle["predicted_label"] in (0, 1)
        assert 0.0 <= bundle["anomaly_probability"] <= 1.0
        assert isinstance(bundle["top_evidence"], list)
        assert bundle["top_contributions"]
        for entry in bundle["top_contributions"].values():
            assert isinstance(entry["contribution"], float)
            assert "template" in entry

    def test_module_disclaims_causation(self):
        doc = explainability.__doc__.lower()
        assert "causation" in doc and "not" in doc
