"""Generate and persist model explanations on held-out test samples.

Uses ONLY held-out test rows for explanation subjects and for global
importance evaluation. The training split is used solely for the
training-normal reference profile (the distribution the model was fitted
against).

Artifacts:

* reports/figures/global_feature_importance.png   two-panel global view
* reports/figures/local_explanations.png          one normal + one
                                                  anomalous sample

All importance and evidence output reflects model ASSOCIATION, not
causation.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sentinelai.config import get_settings
from sentinelai.data_loader import (
    load_anomaly_labels,
    load_occurrence_matrix,
    load_templates,
)
from sentinelai.explainability import (
    build_training_profile,
    explain_prediction,
    global_feature_importance,
    global_shap_importance,
    local_feature_contributions,
    map_event_features,
    pipeline_feature_names,
)
from sentinelai.modeling import load_model
from sentinelai.preprocessing import grouped_split_dataset

N_PERMUTATION_ROWS = 4_000
N_SHAP_ROWS = 2_000
PERMUTATION_REPEATS = 5


def _held_out_subsample(
    X_test: pd.DataFrame,
    y_test: pd.Series,
    n_rows: int,
    positive_share: float = 0.3,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    anomaly_idx = np.where(y_test.to_numpy() == 1)[0]
    normal_idx = np.where(y_test.to_numpy() == 0)[0]
    n_positive = int(n_rows * positive_share)
    picked = np.concatenate(
        [
            rng.choice(anomaly_idx, size=n_positive, replace=False),
            rng.choice(normal_idx, size=n_rows - n_positive, replace=False),
        ]
    )
    return X_test.iloc[picked], y_test.iloc[picked]


def _save_global_figure(
    permutation: pd.DataFrame, shap_global: pd.DataFrame, path: str
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    top = permutation.head(12).iloc[::-1]
    axes[0].barh(
        top["feature"],
        top["importance_mean"],
        xerr=top["importance_std"],
        color="#1565c0",
        alpha=0.85,
    )
    axes[0].set_title(
        "Permutation importance — raw event counts\n"
        "(held-out, average precision)"
    )
    axes[0].set_xlabel("mean importance (score drop when permuted)")
    top_shap = shap_global.head(15).iloc[::-1]
    axes[1].barh(
        top_shap["feature"],
        top_shap["importance_mean"],
        color="#6a1b9a",
        alpha=0.85,
    )
    axes[1].set_title(
        "Global SHAP — mean |contribution| per model feature\n"
        "(held-out, decision margin)"
    )
    axes[1].set_xlabel("mean absolute SHAP contribution")
    fig.suptitle(
        "Global feature importance (association, not causation)",
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _save_local_figure(
    normal_bundle: dict,
    anomalous_bundle: dict,
    normal_contributions: pd.DataFrame,
    anomalous_contributions: pd.DataFrame,
    path: str,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    for ax, bundle, contributions, title_color in (
        (
            axes[0],
            normal_bundle,
            normal_contributions,
            "#2e7d32",
        ),
        (
            axes[1],
            anomalous_bundle,
            anomalous_contributions,
            "#c62828",
        ),
    ):
        top = contributions.iloc[0].abs().sort_values(
            ascending=False
        ).head(10)
        values = contributions.iloc[0][top.index]
        colors = ["#c62828" if v > 0 else "#2e7d32" for v in values]
        ax.barh(top.index[::-1], values[::-1], color=colors[::-1], alpha=0.85)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title(
            f"{bundle['predicted_label'] and 'ANOMALOUS' or 'NORMAL'} "
            f"sample — block {bundle['block_id']}\n"
            f"anomaly probability = {bundle['anomaly_probability']:.4f}",
            color=title_color,
            fontweight="bold",
        )
        ax.set_xlabel("SHAP contribution (decision margin)")
    fig.suptitle(
        "Local explanations on held-out test samples "
        "(association, not causation)",
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _print_bundle(title: str, bundle: dict) -> None:
    print(f"\n=== {title} ===")
    print(f"block_id: {bundle['block_id']}")
    print(f"predicted_label: {bundle['predicted_label']}")
    print(f"anomaly_probability: {bundle['anomaly_probability']:.6f}")
    print("top evidence:")
    for phrase in bundle["top_evidence"]:
        print(f"  - {phrase}")
    print("top contributions:")
    for feature, entry in bundle["top_contributions"].items():
        template = entry["template"] or ""
        suffix = f"  [{template}]" if template else ""
        print(
            f"  - {feature}: {entry['contribution']:+.3f}{suffix}"
        )


def main() -> None:
    settings = get_settings()
    settings.ensure_dirs()

    matrix = load_occurrence_matrix()
    labels = load_anomaly_labels()
    templates = load_templates(required=True)
    X_train, X_test, y_train, y_test = grouped_split_dataset(
        matrix, labels, n_splits=5, random_state=settings.random_state
    )
    pipeline = load_model(settings.model_path("best_model_pipeline"))

    profile = build_training_profile(
        pipeline, X_train[y_train.to_numpy() == 0]
    )

    print("computing global importance on held-out test subsamples ...")
    X_perm, y_perm = _held_out_subsample(X_test, y_test, N_PERMUTATION_ROWS)
    permutation = global_feature_importance(
        pipeline,
        X_perm,
        y_perm,
        n_repeats=PERMUTATION_REPEATS,
        random_state=settings.random_state,
    )
    X_shap, _ = _held_out_subsample(X_test, y_test, N_SHAP_ROWS)
    shap_global = global_shap_importance(pipeline, X_shap)

    print("\nglobal permutation importance (raw event counts):")
    for row in permutation.head(10).itertuples(index=False):
        print(
            f"  {row.feature:>6}  "
            f"{row.importance_mean:+.4f} +/- {row.importance_std:.4f}"
        )
    print("\nglobal SHAP importance (model features, mean |contribution|):")
    for row in shap_global.head(10).itertuples(index=False):
        print(
            f"  {row.feature:>28}  "
            f"{row.importance_mean:.4f} +/- {row.importance_std:.4f}"
        )

    figure_path = settings.figures_dir / "global_feature_importance.png"
    _save_global_figure(permutation, shap_global, str(figure_path))
    print(f"\nglobal importance figure saved: {figure_path}")

    probabilities = pipeline.predict_proba(X_test)[:, 1]
    normal_mask = y_test.to_numpy() == 0
    anomalous_mask = y_test.to_numpy() == 1

    # pick the most typical normal sample: fewest deviations from the
    # training-normal profile, tie-broken by lowest anomaly probability
    transformed_test = pd.DataFrame(
        np.asarray(pipeline[:-1].transform(X_test)),
        columns=pipeline_feature_names(pipeline),
    )
    p5 = profile.percentiles["p5"]
    p95 = profile.percentiles["p95"]
    deviation_counts = (
        (transformed_test > p95).sum(axis=1)
        + ((transformed_test < p5) & (p5 > 0)).sum(axis=1)
    ).to_numpy()
    normal_candidates = np.where(normal_mask)[0]
    order = np.lexsort(
        (probabilities[normal_candidates], deviation_counts[normal_candidates])
    )
    normal_pos = int(normal_candidates[order[0]])

    anomalous_pos = int(np.argmax(probabilities[anomalous_mask]))
    normal_sample = X_test.iloc[[normal_pos]]
    anomalous_sample = X_test.iloc[anomalous_mask].iloc[[anomalous_pos]]

    normal_bundle = explain_prediction(
        pipeline, profile, templates, normal_sample
    )
    anomalous_bundle = explain_prediction(
        pipeline, profile, templates, anomalous_sample
    )
    _, normal_contributions = local_feature_contributions(
        pipeline, normal_sample
    )
    _, anomalous_contributions = local_feature_contributions(
        pipeline, anomalous_sample
    )

    _print_bundle("normal explanation (held-out)", normal_bundle)
    _print_bundle("anomalous explanation (held-out)", anomalous_bundle)

    for name, bundle in (
        ("normal", normal_bundle),
        ("anomalous", anomalous_bundle),
    ):
        expected, contributions = local_feature_contributions(
            pipeline,
            X_test.loc[[bundle["block_id"]]],
        )
        margin = expected + contributions.sum(axis=1).to_numpy()[0]
        reconstructed = 1.0 / (1.0 + np.exp(-margin))
        print(
            f"{name} reconstruction check: |reconstructed - proba| = "
            f"{abs(reconstructed - bundle['anomaly_probability']):.2e}"
        )

    local_path = settings.figures_dir / "local_explanations.png"
    _save_local_figure(
        normal_bundle,
        anomalous_bundle,
        normal_contributions,
        anomalous_contributions,
        str(local_path),
    )
    print(f"\nlocal explanation figure saved: {local_path}")

    evidence_events = sorted(
        {
            token
            for bundle in (normal_bundle, anomalous_bundle)
            for phrase in bundle["top_evidence"]
            for token in phrase.split()
            if token.startswith("E") and token[1:].isdigit()
        }
    )
    if evidence_events:
        mapping = map_event_features(templates, evidence_events)
        print("\nevent templates referenced by evidence:")
        for event, template in mapping.items():
            print(f"  {event}: {template}")

    print(
        "\nNOTE: importance and evidence reflect model association on "
        "held-out data; they do not prove causation."
    )


if __name__ == "__main__":
    main()
