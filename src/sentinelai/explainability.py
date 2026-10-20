"""Explainability for SentinelAI.

Global and local explanations for the trained incident-prediction
pipeline, plus evidence phrasing that maps model features back to HDFS
event IDs and templates.

Feature importance and evidence statements describe ASSOCIATION between
features and model predictions. They do NOT prove causation.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd
import shap
from sklearn.inspection import permutation_importance

from sentinelai.preprocessing import assert_no_label_columns

EVENT_FEATURE_RE = re.compile(r"^E(\d+)$")

_TREE_MODEL_NAMES = {
    "HistGradientBoostingClassifier",
    "RandomForestClassifier",
    "XGBClassifier",
}

# derived feature -> (high-tail phrase, low-tail phrase); None = not phrased
DERIVED_EVIDENCE = {
    "total_event_count": (
        "abnormal total event count",
        "abnormally low event count",
    ),
    "unique_event_count": ("unusually many distinct event types", None),
    "dominant_event_ratio": (None, "no dominant event type"),
    "repeated_event_ratio": ("high repeated-event ratio", None),
    "event_entropy": ("unusual event-type diversity", None),
    "rare_event_count": ("rare events occurred", None),
    "zero_event_ratio": (None, None),
    "maximum_single_event_frequency": (None, None),
}

RARE_EVENT_SEVERITY = 2.5
NORMAL_EVIDENCE = [
    "all event frequencies within normal range",
    "typical event pattern",
]


def pipeline_feature_names(pipeline: Any) -> list[str]:
    """Feature names of the model's input space (all steps but the last)."""
    return list(pipeline[:-1].get_feature_names_out())


def global_feature_importance(
    pipeline: Any,
    X: pd.DataFrame,
    y: Any,
    *,
    n_repeats: int = 5,
    random_state: int = 42,
    scoring: str = "average_precision",
) -> pd.DataFrame:
    """Permutation-based global feature importance on held-out data.

    ``X``/``y`` must be held-out rows only — never data used to fit the
    pipeline. ``scoring`` should be a metric that needs both classes
    (e.g. "average_precision"), so pass a sample containing anomalies.

    Permutation operates on the pipeline's RAW input columns (the event
    counts); derived features are computed from those columns and are
    covered by :func:`global_shap_importance`.

    Importance reflects model association, not causation.
    """
    assert_no_label_columns(X)
    result = permutation_importance(
        pipeline,
        X,
        y,
        scoring=scoring,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=-1,
    )
    frame = pd.DataFrame(
        {
            "feature": list(X.columns),
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    )
    return (
        frame.sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )


def global_shap_importance(
    pipeline: Any, X: pd.DataFrame
) -> pd.DataFrame:
    """Global importance as mean absolute SHAP contribution per feature.

    Covers the full model feature space (raw event features AND derived
    features) on the held-out rows in ``X``. Reflects model association,
    not causation.
    """
    assert_no_label_columns(X)
    _, contributions = local_feature_contributions(pipeline, X)
    magnitudes = contributions.abs()
    frame = pd.DataFrame(
        {
            "feature": list(magnitudes.columns),
            "importance_mean": magnitudes.mean().to_numpy(),
            "importance_std": magnitudes.std().to_numpy(),
        }
    )
    return (
        frame.sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )


def local_feature_contributions(
    pipeline: Any, X: pd.DataFrame
) -> tuple[float, pd.DataFrame]:
    """SHAP margin contributions per feature for each sample in ``X``.

    Returns ``(expected_value, contributions)`` where the row sums of
    ``contributions`` plus ``expected_value`` reconstruct the model's
    decision margin (log-odds) for every sample:

        sigmoid(expected_value + sum(contributions)) == predict_proba

    Contributions are on the model's decision scale — they are
    association-based attributions, not causal effects. Only tree-based
    classifiers (HistGradientBoosting / RandomForest / XGBoost) are
    supported.
    """
    assert_no_label_columns(X)
    model = pipeline[-1]
    if model.__class__.__name__ not in _TREE_MODEL_NAMES:
        raise ValueError(
            "local_feature_contributions requires a tree-based classifier, "
            f"got {model.__class__.__name__}"
        )
    features = np.asarray(pipeline[:-1].transform(X))
    explainer = shap.TreeExplainer(model)
    values = np.asarray(explainer.shap_values(features))
    expected = float(np.asarray(explainer.expected_value).ravel()[0])
    contributions = pd.DataFrame(
        values,
        columns=pipeline_feature_names(pipeline),
        index=X.index,
    )
    return expected, contributions


def map_event_features(
    templates: pd.DataFrame | None, feature_names: Sequence[str]
) -> dict[str, str | None]:
    """Map event features (E1..E29) to HDFS event templates where available.

    Returns one entry per feature name: the template text for event
    features that exist in ``templates``, ``None`` otherwise (derived
    features and unknown event IDs).
    """
    template_lookup: dict[str, str] = {}
    if templates is not None:
        template_lookup = {
            str(row.EventId): str(row.EventTemplate)
            for row in templates.itertuples(index=False)
        }
    mapping: dict[str, str | None] = {}
    for name in feature_names:
        match = EVENT_FEATURE_RE.match(str(name))
        if match:
            mapping[name] = template_lookup.get(f"E{match.group(1)}")
        else:
            mapping[name] = None
    return mapping


@dataclass(frozen=True)
class EvidenceProfile:
    """Reference distribution of model features on training-normal rows."""

    percentiles: pd.DataFrame  # index=feature, columns [p5, median, p95]
    rare_event_columns_: tuple[str, ...]


def build_training_profile(
    pipeline: Any, X_train: pd.DataFrame
) -> EvidenceProfile:
    """Reference percentiles of each model feature on training-normal rows.

    ``X_train`` must be training-normal rows (label 0) — the reference
    distribution the model was fitted against. Test data must never be
    passed here.
    """
    assert_no_label_columns(X_train)
    transformed = np.asarray(pipeline[:-1].transform(X_train))
    features = pd.DataFrame(
        transformed, columns=pipeline_feature_names(pipeline)
    )
    percentiles = pd.DataFrame(
        {
            "p5": features.quantile(0.05),
            "median": features.quantile(0.50),
            "p95": features.quantile(0.95),
        }
    )
    rare = tuple(pipeline.named_steps["derive"].rare_event_columns_)
    return EvidenceProfile(percentiles=percentiles, rare_event_columns_=rare)


def _exceedance_ratio(value: float, reference: float, floor: float) -> float:
    return min(value / max(reference, floor), 100.0)


def top_evidence(
    profile: EvidenceProfile,
    pipeline: Any,
    sample: pd.DataFrame,
    *,
    top_k: int = 5,
) -> list[str]:
    """Human-readable Top Evidence for one sample.

    Compares the sample's model features against the training-normal
    profile and phrases the most extreme deviations, e.g. "E17 abnormal
    frequency" or "abnormal total event count". Returns the normal
    summary phrases when nothing deviates. Evidence states association
    with abnormal patterns — it does not prove causation.
    """
    names = pipeline_feature_names(pipeline)
    transformed = np.asarray(pipeline[:-1].transform(sample))
    values = pd.Series(transformed.ravel(), index=names)
    candidates: list[tuple[float, str]] = []
    percentiles = profile.percentiles

    for name in percentiles.index:
        if not EVENT_FEATURE_RE.match(str(name)):
            continue
        value = float(values[name])
        p5 = float(percentiles.loc[name, "p5"])
        p95 = float(percentiles.loc[name, "p95"])
        if (
            name in profile.rare_event_columns_
            and name in sample.columns
            and float(sample[name].iloc[0]) > 0
        ):
            candidates.append(
                (RARE_EVENT_SEVERITY, f"{name} rare event occurred")
            )
        elif value > p95:
            ratio = _exceedance_ratio(value, p95, floor=1.0)
            phrase = (
                f"{name} unusually repeated"
                if ratio >= 3.0
                else f"{name} abnormal frequency"
            )
            candidates.append((ratio, phrase))
        elif p5 > 0 and value < p5:
            ratio = _exceedance_ratio(p5, value, floor=1.0)
            candidates.append((ratio, f"{name} unusually absent"))

    for name, (high_phrase, low_phrase) in DERIVED_EVIDENCE.items():
        if name not in percentiles.index:
            continue
        value = float(values[name])
        p5 = float(percentiles.loc[name, "p5"])
        p95 = float(percentiles.loc[name, "p95"])
        if high_phrase and value > p95:
            candidates.append(
                (_exceedance_ratio(value, p95, floor=1e-6), high_phrase)
            )
        elif low_phrase and p5 > 0 and value < p5:
            candidates.append(
                (_exceedance_ratio(p5, value, floor=1e-6), low_phrase)
            )

    if not candidates:
        return list(NORMAL_EVIDENCE)
    candidates.sort(key=lambda pair: pair[0], reverse=True)
    return [phrase for _, phrase in candidates[:top_k]]


def explain_prediction(
    pipeline: Any,
    profile: EvidenceProfile,
    templates: pd.DataFrame | None,
    sample: pd.DataFrame,
    *,
    top_k: int = 5,
) -> dict[str, Any]:
    """Full explanation bundle for one sample.

    Combines the predicted label, the anomaly probability, the Top
    Evidence phrases and the largest local feature contributions (with
    event templates where available). Describes association, not
    causation.
    """
    probability = float(pipeline.predict_proba(sample)[0, 1])
    predicted = int(pipeline.predict(sample)[0])
    evidence = top_evidence(profile, pipeline, sample, top_k=top_k)
    _, contributions = local_feature_contributions(pipeline, sample)
    top = contributions.abs().iloc[0].sort_values(ascending=False).head(8)
    mapping = map_event_features(templates, list(top.index))
    return {
        "block_id": str(sample.index[0]),
        "predicted_label": predicted,
        "anomaly_probability": probability,
        "top_evidence": evidence,
        "top_contributions": {
            str(name): {
                "contribution": float(contributions.iloc[0][name]),
                "template": mapping.get(str(name)),
            }
            for name in top.index
        },
    }
