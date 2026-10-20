"""Unit tests for sentinelai.modeling (toy data only, no real dataset)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sentinelai.modeling import (
    available_model_names,
    build_classifier_pipeline,
    evaluate_model,
    fit_params,
    load_model,
    save_model,
    select_best_model,
)
from sentinelai.preprocessing import assert_no_label_columns


class TestEvaluateModel:
    def test_known_values_hand_computed(self):
        # y_true: 0 0 0 0 1 1 ; y_pred: 0 0 1 0 1 0
        y_true = np.array([0, 0, 0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 0, 1, 0])
        y_proba = np.array([0.1, 0.2, 0.6, 0.3, 0.9, 0.4])
        m = evaluate_model(y_true, y_pred, y_proba)
        # tn=3, fp=1, fn=1, tp=1
        assert (m["tn"], m["fp"], m["fn"], m["tp"]) == (3, 1, 1, 1)
        assert m["precision"] == pytest.approx(0.5)
        assert m["recall"] == pytest.approx(0.5)
        assert m["f1"] == pytest.approx(0.5)
        assert m["accuracy"] == pytest.approx(4 / 6)
        assert m["fpr"] == pytest.approx(0.25)
        assert m["fnr"] == pytest.approx(0.5)
        # ROC-AUC: positive 0.9 beats all 4 negatives; 0.4 beats 0.1/0.2/0.3
        # -> 7 wins of 8 pairs = 0.875
        assert m["roc_auc"] == pytest.approx(0.875)
        # PR-AUC: (0.9,tp): P=1,R=0.5 -> 0.5; (0.4,tp): P=2/3,R=1 -> +1/3
        assert m["pr_auc"] == pytest.approx(5 / 6)

    def test_perfect_prediction(self):
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_pred = y_true.copy()
        y_proba = y_true.astype(float)
        m = evaluate_model(y_true, y_pred, y_proba)
        assert m["precision"] == 1.0
        assert m["recall"] == 1.0
        assert m["f1"] == 1.0
        assert m["roc_auc"] == 1.0
        assert m["pr_auc"] == 1.0
        assert m["fpr"] == 0.0
        assert m["fnr"] == 0.0
        assert m["accuracy"] == 1.0

    def test_requires_both_classes(self):
        with pytest.raises(ValueError, match="both classes"):
            evaluate_model(
                np.zeros(5, dtype=int),
                np.zeros(5, dtype=int),
                np.zeros(5),
            )

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="same length"):
            evaluate_model(np.zeros(3), np.zeros(4), np.zeros(3))


class TestSelectBestModel:
    def make_table(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "model": ["a", "b", "c"],
                "pr_auc": [0.99, 0.80, 0.90],
                "f1": [0.5, 0.5, 0.5],
                "recall": [0.9, 0.7, 0.8],
                "fpr": [0.50, 0.02, 0.03],
            }
        )

    def test_fpr_filter_excludes_high_false_alerts(self):
        best = select_best_model(self.make_table(), max_fpr=0.05)
        assert best["model"] == "c"  # a excluded (fpr 0.5); c beats b on pr_auc

    def test_fallback_to_lowest_fpr_when_all_unacceptable(self):
        best = select_best_model(self.make_table(), max_fpr=0.005)
        assert best["model"] == "b"  # lowest fpr

    def test_tiebreak_order(self):
        table = pd.DataFrame(
            {
                "model": ["a", "b"],
                "pr_auc": [0.9, 0.9],
                "f1": [0.55, 0.60],
                "recall": [0.9, 0.8],
                "fpr": [0.01, 0.01],
            }
        )
        assert select_best_model(table)["model"] == "b"  # higher f1 wins

    def test_empty_table_raises(self):
        with pytest.raises(ValueError, match="empty"):
            select_best_model(pd.DataFrame())


def make_toy_event_frame(n: int = 60, n_pos: int = 12, seed: int = 0):
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(
        rng.integers(0, 5, size=(n, 6)),
        columns=[f"E{i}" for i in range(1, 7)],
    )
    y = pd.Series(
        np.array([1] * n_pos + [0] * (n - n_pos), dtype="int8"),
        name="Label",
    )
    return frame, y


class TestPipelines:
    @pytest.mark.parametrize(
        "model_name",
        [
            "logistic_regression",
            "random_forest",
            "hist_gradient_boosting",
            "xgboost",
        ],
    )
    def test_smoke_fit_predict(self, model_name):
        if model_name == "xgboost":
            pytest.importorskip("xgboost")
        frame, y = make_toy_event_frame()
        pipeline = build_classifier_pipeline(model_name, random_state=0)
        pipeline.fit(frame, y, **fit_params(model_name, y))
        proba = pipeline.predict_proba(frame)
        pred = pipeline.predict(frame)
        assert proba.shape == (len(frame), 2)
        assert set(np.unique(pred)) <= {0, 1}
        assert_no_label_columns(frame)

    def test_fit_params_balanced_weights(self):
        frame, y = make_toy_event_frame()
        params = fit_params("hist_gradient_boosting", y)
        weights = params["model__sample_weight"]
        assert len(weights) == len(y)
        minority_weight = weights[y.to_numpy() == 1][0]
        majority_weight = weights[y.to_numpy() == 0][0]
        assert minority_weight > majority_weight
        assert fit_params("random_forest", y) == {}

    def test_unknown_model_raises(self):
        with pytest.raises(ValueError, match="unknown model"):
            build_classifier_pipeline("svm")

    def test_save_load_roundtrip(self, tmp_path):
        frame, y = make_toy_event_frame()
        pipeline = build_classifier_pipeline("logistic_regression", random_state=0)
        pipeline.fit(frame, y)
        path = tmp_path / "model.joblib"
        save_model(pipeline, path)
        loaded = load_model(path)
        np.testing.assert_array_equal(
            pipeline.predict(frame), loaded.predict(frame)
        )

    def test_available_model_names(self):
        names = available_model_names()
        assert names[:3] == [
            "logistic_regression",
            "random_forest",
            "hist_gradient_boosting",
        ]
        try:
            import xgboost  # noqa: F401

            has_xgboost = True
        except ImportError:
            has_xgboost = False
        assert ("xgboost" in names) == has_xgboost
