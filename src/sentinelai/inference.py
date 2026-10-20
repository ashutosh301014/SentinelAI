"""Production inference pipeline for SentinelAI.

One inference request executes, in order:

1.  Input Validation            raw event-count row schema and values
2.  Feature Engineering         fitted select -> derive transform
3.  Supervised Classifier       HistGradientBoosting pipeline
4.  Isolation Forest            normalized anomaly score in [0, 1]
5.  Composite Incident Risk Score   4-signal weighted fusion (0-100)
6.  Explainability              Top Evidence + local SHAP contributions
7.  Incident Intelligence       fused IncidentReport (probable cause)
8.  Dependency Graph Mapping    probable component -> topology service
9.  Blast-Radius Prediction     Estimated Propagation Risk from the origin
10. What-If Simulation          optional Estimated Counterfactual Impact

The output is one structured object (:class:`InferenceResult`).

Production inference NEVER retrains: :class:`InferencePipeline` only loads
the artifacts persisted under ``models/`` (classifier, Isolation Forest,
preprocessing metadata, expected feature order, event-template mapping,
training-normal evidence profile, consolidated evaluation metadata) and
has no fit path. The blast-radius and what-if stages carry the same
disclaimers as their modules: propagation risks are estimates, not
calibrated failure probabilities, and simulations are estimated
counterfactual impacts, never guaranteed recoveries.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.utils.validation import check_is_fitted

from sentinelai.anomaly_detector import load_detector
from sentinelai.blast_radius import estimate_blast_radius
from sentinelai.config import Settings, get_settings
from sentinelai.dependency_graph import criticality_ranking, load_topology
from sentinelai.explainability import (
    EvidenceProfile,
    local_feature_contributions,
    map_event_features,
    pipeline_feature_names,
    top_evidence,
)
from sentinelai.incident_engine import (
    NO_ANOMALY_COMPONENT,
    build_incident_report,
    event_intensity,
    event_irregularity,
)
from sentinelai.modeling import MODEL_LABELS, load_model
from sentinelai.recovery_simulator import RECOVERY_ACTIONS, simulate_recovery
from sentinelai.risk_engine import compute_risk_score

PIPELINE_STAGES: tuple[str, ...] = (
    "input_validation",
    "feature_engineering",
    "supervised_classifier",
    "isolation_forest",
    "composite_risk_score",
    "explainability",
    "incident_intelligence",
    "dependency_graph_mapping",
    "blast_radius_prediction",
    "what_if_simulation",
)

POSITIVE_LABEL_NAME = "anomaly"
RISK_CAVEAT = "operational risk indicator, NOT a calibrated probability"
EXPLAINABILITY_CAVEAT = "association-based attribution, not causation"
WHAT_IF_STATUS = "estimated counterfactual impact"


@dataclass(frozen=True)
class InferenceResult:
    """One structured object: the complete production inference output."""

    request_id: str
    block_id: str
    dataset: str
    model_version: str
    stages: tuple[str, ...] = PIPELINE_STAGES
    input_validation: dict[str, Any] = field(default_factory=dict)
    feature_engineering: dict[str, Any] = field(default_factory=dict)
    supervised_classifier: dict[str, Any] = field(default_factory=dict)
    isolation_forest: dict[str, Any] = field(default_factory=dict)
    composite_risk_score: dict[str, Any] = field(default_factory=dict)
    explainability: dict[str, Any] = field(default_factory=dict)
    incident_intelligence: dict[str, Any] = field(default_factory=dict)
    dependency_graph_mapping: dict[str, Any] = field(default_factory=dict)
    blast_radius_prediction: dict[str, Any] = field(default_factory=dict)
    what_if_simulation: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "block_id": self.block_id,
            "dataset": self.dataset,
            "model_version": self.model_version,
            "stages": list(self.stages),
            "input_validation": dict(self.input_validation),
            "feature_engineering": dict(self.feature_engineering),
            "supervised_classifier": dict(self.supervised_classifier),
            "isolation_forest": dict(self.isolation_forest),
            "composite_risk_score": dict(self.composite_risk_score),
            "explainability": dict(self.explainability),
            "incident_intelligence": dict(self.incident_intelligence),
            "dependency_graph_mapping": dict(self.dependency_graph_mapping),
            "blast_radius_prediction": dict(self.blast_radius_prediction),
            "what_if_simulation": [dict(s) for s in self.what_if_simulation],
        }


def _read_json(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        payload = json.load(fh)
    if not isinstance(payload, dict):
        raise RuntimeError(f"{path}: expected a JSON object")
    return payload


def _templates_frame(payload: dict[str, str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "EventId": list(payload.keys()),
            "EventTemplate": list(payload.values()),
        }
    )


class InferencePipeline:
    """Production inference pipeline. Loads artifacts only — never fits.

    Parameters
    ----------
    settings
        SentinelAI settings; defaults to the cached singleton.
    models_dir
        Override for the artifact directory (defaults to
        ``settings.models_dir``).
    """

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        models_dir: Path | None = None,
    ):
        self.settings = settings or get_settings()
        models_dir = Path(models_dir or self.settings.models_dir)
        self.classifier = load_model(models_dir / "best_model_pipeline.joblib")
        self.detector = load_detector(models_dir / "anomaly_detector.joblib")
        self.profile = joblib.load(models_dir / "training_profile.joblib")
        self.metadata = _read_json(models_dir / "metadata.json")
        self.feature_order = list(
            _read_json(models_dir / "feature_order.json")["features"]
        )
        self.preprocessing_metadata = _read_json(
            models_dir / "preprocessing_metadata.json"
        )
        self.templates = _templates_frame(
            _read_json(models_dir / "event_templates.json")["templates"]
        )
        self.graph = load_topology(self.settings.service_topology)
        self.component_mapping = _read_json(
            self.settings.component_service_mapping
        )
        self._verify_artifacts()

    def _verify_artifacts(self) -> None:
        fitted = pipeline_feature_names(self.classifier)
        expected = list(self.metadata["feature_names"])
        if fitted != expected or fitted != self.feature_order:
            raise RuntimeError(
                "classifier feature names disagree with the models/ "
                "artifacts (feature_order.json, metadata.json)"
            )
        if int(self.metadata["feature_count"]) != len(expected):
            raise RuntimeError(
                "metadata.json feature_count disagrees with feature_names"
            )
        if self.metadata["selected_model"] not in MODEL_LABELS:
            raise RuntimeError(
                f"unknown selected_model {self.metadata['selected_model']!r}"
            )
        if not isinstance(self.profile, EvidenceProfile):
            raise RuntimeError(
                "training_profile.joblib is not an EvidenceProfile"
            )
        check_is_fitted(self.detector, attributes=["raw_mean_"])

    # ---- stage 1: input validation ----
    def _validate_raw_sample(self, sample: Any) -> dict[str, Any]:
        if not isinstance(sample, pd.DataFrame):
            raise TypeError(
                "sample must be a pandas DataFrame of raw event counts"
            )
        if len(sample) != 1:
            raise ValueError(
                f"exactly one row per request is supported, got {len(sample)}"
            )
        expected = set(self.preprocessing_metadata["input_event_columns"])
        columns = [str(col) for col in sample.columns]
        if len(columns) != len(set(columns)):
            raise ValueError("sample contains duplicate column names")
        missing = sorted(expected - set(columns))
        unseen = sorted(set(columns) - expected)
        if missing or unseen:
            parts = []
            if missing:
                parts.append(f"missing {missing}")
            if unseen:
                parts.append(f"unseen {unseen}")
            raise ValueError(
                "sample schema differs from the production feature schema: "
                + ", ".join(parts)
            )
        for col in sorted(expected):
            series = pd.to_numeric(sample[col], errors="coerce")
            if series.isna().any():
                raise ValueError(
                    f"event column {col!r} contains missing or "
                    "non-numeric values"
                )
            values = series.to_numpy(dtype="float64")
            if not np.isfinite(values).all():
                raise ValueError(
                    f"event column {col!r} contains infinite values"
                )
            if (values < 0).any():
                raise ValueError(
                    f"event column {col!r} contains negative counts"
                )
        return {
            "block_id": str(sample.index[0]),
            "input_event_columns": sorted(columns),
            "expected_event_columns": sorted(expected),
        }

    # ---- stage 8: dependency graph mapping ----
    def _map_component(self, component: str) -> dict[str, Any]:
        if component == NO_ANOMALY_COMPONENT:
            return {
                "probable_component": component,
                "service_id": None,
                "service_name": None,
                "status": "skipped_no_anomaly",
                "rationale": (
                    "No anomaly detected — no incident origin to map "
                    "onto the service topology."
                ),
            }
        lowered = component.lower()
        for rule in self.component_mapping.get("mapping", []):
            keywords = [str(k).lower() for k in rule.get("keywords", [])]
            if keywords and any(k in lowered for k in keywords):
                service = str(rule["service_id"])
                if service not in self.graph:
                    raise RuntimeError(
                        "component mapping references unknown service "
                        f"{service!r}"
                    )
                return {
                    "probable_component": component,
                    "service_id": service,
                    "service_name": str(self.graph.nodes[service]["name"]),
                    "status": "mapped",
                    "matched_rule": rule.get("component"),
                    "rationale": rule.get("rationale"),
                }
        fallback = str(
            self.component_mapping.get("unmapped_fallback", "highest_criticality")
        )
        if fallback != "highest_criticality":
            raise RuntimeError(
                f"unsupported unmapped_fallback {fallback!r} in "
                "component_service_mapping.json"
            )
        service = str(criticality_ranking(self.graph)[0][0])
        return {
            "probable_component": component,
            "service_id": service,
            "service_name": str(self.graph.nodes[service]["name"]),
            "status": "fallback_highest_criticality",
            "matched_rule": None,
            "rationale": self.component_mapping.get(
                "unmapped_fallback_rationale"
            ),
        }

    # ---- stage 10: what-if parsing ----
    def _parse_what_if(self, spec: Any) -> tuple[str, str]:
        if not isinstance(spec, dict):
            raise TypeError(
                f"what_if entries must be dicts, got {type(spec).__name__}"
            )
        action = spec.get("action")
        service = spec.get("affected_service")
        if action not in RECOVERY_ACTIONS:
            raise ValueError(
                f"unknown recovery action {action!r}; choose from "
                f"{sorted(RECOVERY_ACTIONS)}"
            )
        if service not in self.graph:
            raise ValueError(
                f"what_if simulation references unknown service {service!r}"
            )
        return str(action), str(service)

    def run(
        self,
        sample: pd.DataFrame,
        *,
        block_id: str | None = None,
        request_id: str | None = None,
        timestamp: str | None = None,
        what_if: list[dict[str, str]] | None = None,
        top_k: int = 5,
    ) -> InferenceResult:
        """Execute the full 10-stage pipeline for one raw event-count row.

        ``sample`` must be a one-row DataFrame with exactly the production
        event columns (E1..E29 for HDFS_v1) — raw counts, not engineered
        features. ``what_if`` is an optional list of
        ``{"action": ..., "affected_service": ...}`` specs.
        """
        # 1. Input Validation
        validated = self._validate_raw_sample(sample)
        block_id = str(block_id) if block_id is not None else validated["block_id"]
        request_id = request_id or f"REQ-{block_id}"

        # 2. Feature Engineering
        features = np.asarray(self.classifier[:-1].transform(sample))
        produced = list(self.classifier[:-1].get_feature_names_out())
        if produced != self.feature_order:
            raise RuntimeError(
                "engineered feature order differs from feature_order.json"
            )
        engineered = {
            name: float(features[0, i])
            for i, name in enumerate(self.feature_order)
        }

        # 3. Supervised Classifier
        probability = float(self.classifier.predict_proba(sample)[0, 1])
        predicted = int(self.classifier.predict(sample)[0])
        threshold = float(self.metadata["classifier_threshold"])

        # 4. Isolation Forest
        anomaly_score = float(self.detector.score(sample)[0])

        # 5. Composite Incident Risk Score
        irregularity = event_irregularity(
            self.profile, self.classifier, sample
        )
        intensity = event_intensity(self.profile, self.classifier, sample)
        risk = compute_risk_score(
            probability,
            anomaly_score,
            irregularity,
            intensity,
            settings=self.settings,
        )

        # 6. Explainability
        evidence = top_evidence(
            self.profile, self.classifier, sample, top_k=top_k
        )
        _, contributions = local_feature_contributions(
            self.classifier, sample
        )
        top = contributions.abs().iloc[0].sort_values(ascending=False).head(8)
        template_mapping = map_event_features(
            self.templates, list(top.index)
        )

        # 7. Incident Intelligence
        report = build_incident_report(
            block_id,
            sample,
            self.classifier,
            self.detector,
            self.profile,
            self.templates,
            timestamp=timestamp,
            settings=self.settings,
            top_k=top_k,
        )

        # 8. Dependency Graph Mapping
        mapped = self._map_component(report.probable_component)

        # 9. Blast-Radius Prediction
        origin_id = mapped.get("service_id")
        if origin_id is None:
            blast: dict[str, Any] = {
                "status": "skipped",
                "reason": (
                    "no anomaly detected — no incident origin to seed "
                    "the blast radius"
                ),
            }
        else:
            blast = estimate_blast_radius(
                self.graph,
                origin_id,
                origin_risk=risk.risk_score / 100.0,
                settings=self.settings,
            ).to_dict()
            blast["status"] = "computed"
            blast["origin_mapped_from"] = mapped

        # 10. optional What-If Simulation
        simulations: list[dict[str, Any]] = []
        for spec in what_if or []:
            action, service = self._parse_what_if(spec)
            simulation = simulate_recovery(
                self.graph,
                action,
                service,
                origin=origin_id if origin_id is not None else service,
                origin_risk=(
                    risk.risk_score / 100.0 if origin_id is not None else None
                ),
                settings=self.settings,
            )
            record = simulation.to_dict()
            record["status"] = WHAT_IF_STATUS
            simulations.append(record)

        return InferenceResult(
            request_id=request_id,
            block_id=block_id,
            dataset=str(self.metadata["dataset"]),
            model_version=str(self.metadata["model_version"]),
            input_validation={
                "status": "passed",
                "block_id": block_id,
                "row_count": 1,
                "input_event_columns": validated["input_event_columns"],
                "expected_event_columns": validated["expected_event_columns"],
                "checks": [
                    "exactly one row",
                    "exact production event schema",
                    "numeric and finite event counts",
                    "non-negative event counts",
                ],
            },
            feature_engineering={
                "status": "ok",
                "feature_order_matched": True,
                "n_features": len(self.feature_order),
                "features": engineered,
            },
            supervised_classifier={
                "model": self.metadata["selected_model"],
                "model_label": MODEL_LABELS[self.metadata["selected_model"]],
                "prediction": predicted,
                "prediction_label": (
                    POSITIVE_LABEL_NAME if predicted == 1 else "normal"
                ),
                "anomaly_probability": probability,
                "threshold": threshold,
                "positive_class": POSITIVE_LABEL_NAME,
            },
            isolation_forest={
                "anomaly_score": anomaly_score,
                "threshold": float(
                    self.metadata["anomaly_detector"]["threshold"]
                ),
                "scale": "0 = normal, 1 = anomalous",
            },
            composite_risk_score={
                **risk.to_dict(),
                "caveat": RISK_CAVEAT,
            },
            explainability={
                "top_evidence": evidence,
                "top_contributions": {
                    str(name): {
                        "contribution": float(
                            contributions.iloc[0][name]
                        ),
                        "template": template_mapping.get(str(name)),
                    }
                    for name in top.index
                },
                "caveat": EXPLAINABILITY_CAVEAT,
            },
            incident_intelligence=report.to_dict(),
            dependency_graph_mapping=mapped,
            blast_radius_prediction=blast,
            what_if_simulation=simulations,
        )
