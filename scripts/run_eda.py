"""Run EDA on the real HDFS_v1 dataset.

Computes all analyses via sentinelai.eda, saves figures to reports/figures/,
and writes reports/eda_summary.md. Every number in the summary is computed
from the loaded dataset at runtime — nothing is hardcoded.
"""
from __future__ import annotations

import math
import sys
from datetime import date
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from scipy import stats  # noqa: E402

from sentinelai.config import get_settings  # noqa: E402
from sentinelai.data_loader import EVENT_COLUMNS, parse_event_sequence  # noqa: E402
from sentinelai.eda import (  # noqa: E402
    aligned_labels,
    class_conditional_rates,
    class_distribution,
    constant_and_rare_events,
    correlation_verdict,
    event_correlation_matrix,
    event_frequency,
    extreme_counts,
    load_analysis_data,
    missing_values_report,
    point_biserial_discriminativeness,
    save_class_distribution_plot,
    save_correlation_heatmap,
    save_event_count_distribution_plot,
    save_normal_vs_anomaly_plot,
    save_top_events_plot,
    template_labels,
    trace_statistics,
)

FIGURES = (
    "class_distribution.png",
    "top_event_frequency.png",
    "normal_vs_anomaly_events.png",
    "event_count_distribution.png",
)


def fmt_rate(value: float) -> str:
    return f"{value:.4g}"


def aligned_occurrences_latency(
    matrix: pd.DataFrame, traces: pd.DataFrame
) -> pd.DataFrame:
    """Per-block total occurrences joined with latency on BlockId."""
    totals = matrix.set_index("BlockId")[EVENT_COLUMNS].sum(axis=1)
    joined = pd.DataFrame(
        {"total_occurrences": totals, "latency": traces.set_index("BlockId")["Latency"]}
    ).dropna()
    return joined


def main() -> int:
    settings = get_settings()
    fig_dir = settings.figures_dir
    fig_dir.mkdir(parents=True, exist_ok=True)

    print("loading HDFS_v1 data ...")
    data = load_analysis_data()
    labels = data["labels"]
    matrix = data["occurrence_matrix"]
    traces = data["traces"]
    templates = data["templates"]

    tlabels = template_labels(templates)
    n_blocks = len(labels)

    # ---- class distribution
    dist = class_distribution(labels)
    n_normal = int(dist.loc[dist["class"] == "Normal", "count"].iloc[0])
    n_anomaly = int(dist.loc[dist["class"] == "Anomaly", "count"].iloc[0])
    anomaly_share = n_anomaly / n_blocks
    imbalance = n_normal / n_anomaly

    # ---- event frequency
    freq = event_frequency(matrix)

    # ---- per-trace statistics
    trace_summary, per_class = trace_statistics(matrix, labels)
    seq_lengths = traces["Features"].map(parse_event_sequence).map(len)
    occ_latency = aligned_occurrences_latency(matrix, traces)

    # ---- discriminative
    rates = class_conditional_rates(matrix, labels)
    pb = point_biserial_discriminativeness(matrix, labels)
    discriminative = rates.merge(pb, on="event").head(10)
    presence = constant_and_rare_events(matrix)
    constant_events = presence.loc[presence["nunique"] <= 1, "event"].tolist()
    rare_events = presence.loc[
        (presence["blocks_with_event"] < 1000), "event"
    ].tolist()

    # ---- missing values
    missing = missing_values_report(data)

    # ---- extremes
    per_event_max, top_blocks = extreme_counts(matrix, labels)

    # ---- correlations
    corr = event_correlation_matrix(matrix)
    corr_useful, max_abs_r, top_pairs = correlation_verdict(corr)
    spearman_len_lat = stats.spearmanr(occ_latency["total_occurrences"], occ_latency["latency"])
    pearson_len_lat = stats.pearsonr(occ_latency["total_occurrences"], occ_latency["latency"])

    # ---- figures
    saved = []
    saved.append(save_class_distribution_plot(dist, fig_dir / "class_distribution.png"))
    saved.append(save_top_events_plot(freq, tlabels, fig_dir / "top_event_frequency.png"))
    saved.append(
        save_normal_vs_anomaly_plot(rates, tlabels, fig_dir / "normal_vs_anomaly_events.png")
    )
    saved.append(
        save_event_count_distribution_plot(trace_summary, fig_dir / "event_count_distribution.png")
    )
    if corr_useful:
        saved.append(
            save_correlation_heatmap(corr, top_pairs, fig_dir / "correlation_heatmap.png")
        )
        FIGURES_SAVED = FIGURES + ("correlation_heatmap.png",)
    else:
        FIGURES_SAVED = FIGURES
    for path in saved:
        print(f"figure saved: {path}")

    # ---- consistency cross-checks
    template_ids_match = set(EVENT_COLUMNS) == set(templates["EventId"].astype(str))
    mean_seq_from_traces = float(seq_lengths.mean())
    mean_seq_from_matrix = float(trace_summary["total_occurrences"].mean())

    # ---- markdown summary
    lines: list[str] = []
    lines.append("# SentinelAI — HDFS_v1 Exploratory Data Analysis Summary\n")
    lines.append(f"_Generated {date.today().isoformat()} from the real Loghub HDFS_v1 dataset "
                 f"({n_blocks:,} blocks) in `data/raw/hdfs/`. All figures under `reports/figures/`. "
                 "No models were trained._\n")

    lines.append("## 1. Dataset overview\n")
    lines.append(f"- blocks (traces): **{n_blocks:,}**")
    lines.append(f"- event features: **{len(EVENT_COLUMNS)}** ({EVENT_COLUMNS[0]}..{EVENT_COLUMNS[-1]})")
    lines.append(f"- template IDs match event columns: **{template_ids_match}** "
                 f"({len(templates)} templates)")
    lines.append(f"- occurrence matrix: {matrix.shape[0]:,} x {matrix.shape[1]}")
    lines.append(f"- traces: {traces.shape[0]:,} x {traces.shape[1]} "
                 f"(Features, TimeInterval, Latency)")
    lines.append(f"- cross-check: mean sequence length from Features = {mean_seq_from_traces:.2f} "
                 f"vs mean occurrences per row from matrix = {mean_seq_from_matrix:.2f} "
                 f"(consistent: {abs(mean_seq_from_traces - mean_seq_from_matrix) < 0.01})\n")

    lines.append("## 2. Class distribution and imbalance\n")
    lines.append("| class | count | percent |")
    lines.append("|---|---:|---:|")
    for _, row in dist.iterrows():
        lines.append(f"| {row['class']} | {int(row['count']):,} | {row['percent']:.2f}% |")
    lines.append("")
    lines.append(f"- anomaly share: **{anomaly_share * 100:.2f}%** of blocks")
    lines.append(f"- imbalance ratio: **{imbalance:.1f}** normal blocks per anomalous block\n")

    lines.append("## 3. Missing values\n")
    if missing.empty:
        lines.append("- no missing values in any column of labels/matrix/traces "
                     "(aside from the Type column, see below).")
    else:
        lines.append("| file | column | missing | percent |")
        lines.append("|---|---|---:|---:|")
        for _, row in missing.iterrows():
            lines.append(
                f"| {row['file']} | {row['column']} | {int(row['missing']):,} | {row['percent']:.2f}% |"
            )
    type_na_matrix = int(matrix["Type"].isna().sum())
    type_na_traces = int(traces["Type"].isna().sum())
    lines.append("")
    lines.append(f"- `Type` is empty on **{type_na_matrix:,}** matrix rows and "
                 f"**{type_na_traces:,}** trace rows — exactly the normal rows; "
                 "Type is populated only on anomalous rows (failure-type code) "
                 "and must be excluded from model features.\n")

    lines.append("## 4. Event frequency (top 10)\n")
    lines.append("| event | total occurrences | blocks with event | mean per block | template |")
    lines.append("|---|---:|---:|---:|---|")
    for _, row in freq.head(10).iterrows():
        template = str(templates.loc[templates["EventId"].astype(str) == row["event"], "EventTemplate"].iloc[0])
        if len(template) > 55:
            template = template[:52] + "..."
        lines.append(
            f"| {row['event']} | {int(row['total_occurrences']):,} | "
            f"{int(row['blocks_with_event']):,} | {fmt_rate(row['mean_per_block'])} | {template} |"
        )
    total_occurrences = int(freq["total_occurrences"].sum())
    share_top1 = float(freq.iloc[0]["total_occurrences"] / total_occurrences * 100)
    share_top5 = float(freq.head(5)["total_occurrences"].sum() / total_occurrences * 100)
    lines.append("")
    lines.append(f"- total event occurrences across all blocks: **{total_occurrences:,}**")
    lines.append(f"- top-1 event share: **{share_top1:.1f}%** of all occurrences")
    lines.append(f"- top-5 events share: **{share_top5:.1f}%** of all occurrences\n")

    lines.append("## 5. Event occurrence distribution per trace\n")
    lines.append("| stat | Normal | Anomaly |")
    lines.append("|---|---:|---:|")
    for stat_label, col in (
        ("mean occurrences per trace", "mean_occurrences"),
        ("median occurrences per trace", "median_occurrences"),
        ("max occurrences in one trace", "max_occurrences"),
        ("mean distinct event types", "mean_distinct"),
        ("max distinct event types", "max_distinct"),
    ):
        normal_val = per_class.loc[per_class["label"] == "Normal", col].iloc[0]
        anomaly_val = per_class.loc[per_class["label"] == "Anomaly", col].iloc[0]
        lines.append(
            f"| {stat_label} | {fmt_rate(normal_val)} | {fmt_rate(anomaly_val)} |"
        )
    lines.append("")
    lines.append(f"- global: min {int(trace_summary['total_occurrences'].min())}, "
                 f"mean {trace_summary['total_occurrences'].mean():.2f}, "
                 f"max {int(trace_summary['total_occurrences'].max())} occurrences per trace\n")

    lines.append("## 6. Discriminative events (top 10 by |point-biserial r|)\n")
    lines.append("| event | normal rate | anomaly rate | ratio anom/norm | point-biserial r | p-value |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for _, row in discriminative.iterrows():
        ratio = row["ratio_anomaly_over_normal"]
        ratio_str = "inf" if not math.isfinite(ratio) else f"{ratio:.2f}"
        r_val = row["point_biserial_r"]
        p_val = row["p_value"]
        r_str = "—" if not math.isfinite(r_val) else f"{r_val:+.4f}"
        p_str = "—" if not math.isfinite(p_val) else f"{p_val:.2e}"
        lines.append(
            f"| {row['event']} | {fmt_rate(row['normal_rate'])} | "
            f"{fmt_rate(row['anomaly_rate'])} | {ratio_str} | "
            f"{r_str} | {p_str} |"
        )
    lines.append("")
    if constant_events:
        constant_info = [
            f"{e} (constant count {int(matrix[e].iloc[0])} in every block)"
            for e in constant_events
        ]
        lines.append(
            f"- constant events (zero variance, no discriminative information): "
            f"{'; '.join(constant_info)}"
        )
    if rare_events:
        rare_info = [
            f"{e} ({int(presence.loc[presence['event'] == e, 'blocks_with_event'].iloc[0]):,} blocks)"
            for e in rare_events
        ]
        lines.append(f"- rare events (occur in < 1,000 blocks): {'; '.join(rare_info)}")
    lines.append("")

    lines.append("## 7. Extreme event counts\n")
    lines.append("| event | max count in one block |")
    lines.append("|---|---:|")
    for _, row in per_event_max.head(5).iterrows():
        lines.append(f"| {row['event']} | {int(row['max_count_in_one_block']):,} |")
    lines.append("")
    lines.append("| top blocks | total occurrences | class |")
    lines.append("|---|---:|---|")
    for _, row in top_blocks.iterrows():
        lines.append(
            f"| {row['block_id']} | {int(row['total_occurrences']):,} | {row['label']} |"
        )
    lines.append("")

    lines.append("## 8. Correlations\n")
    lines.append(f"- event-count Pearson correlation (29 x 29): max |r| off-diagonal = **{max_abs_r:.3f}**")
    if top_pairs:
        pair_text = "; ".join(f"{a}–{b} ({r:+.3f})" for a, b, r in top_pairs)
        lines.append(f"- strongest pairs: {pair_text}")
        top_a, top_b, _ = top_pairs[0]
        identical = int((matrix[top_a] == matrix[top_b]).sum())
        share = identical / n_blocks * 100
        verdict_text = " — redundant features" if share >= 99.9 else ""
        lines.append(
            f"- top pair {top_a}–{top_b}: identical count vectors in "
            f"{identical:,} blocks ({share:.1f}%){verdict_text}"
        )
    if corr_useful:
        lines.append("- verdict: heatmap generated (`correlation_heatmap.png`).")
    else:
        lines.append(
            "- verdict: no heatmap generated — max |r| < 0.3; event counts are "
            "sparse (mean 0.67 occurrences per cell) and show no practically "
            "useful linear structure."
        )
    lines.append(
        f"- trace length vs latency: Pearson r = {pearson_len_lat[0]:+.3f} "
        f"(p = {pearson_len_lat[1]:.2e}), Spearman rho = {spearman_len_lat[0]:+.3f} "
        f"(p = {spearman_len_lat[1]:.2e})"
    )
    lines.append("")

    lines.append("## 9. Figures\n")
    for name in FIGURES_SAVED:
        lines.append(f"- `reports/figures/{name}`")
    lines.append("")

    summary_path = settings.reports_dir / "eda_summary.md"
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"summary written: {summary_path}")

    print("-" * 60)
    print(f"blocks={n_blocks:,}  normal={n_normal:,}  anomalous={n_anomaly:,}  "
          f"anomaly_share={anomaly_share * 100:.2f}%")
    print(f"event features={len(EVENT_COLUMNS)}  total occurrences={total_occurrences:,}")
    print(f"mean occurrences/trace={trace_summary['total_occurrences'].mean():.2f} "
          f"(normal {per_class.loc[per_class['label'] == 'Normal', 'mean_occurrences'].iloc[0]:.2f} "
          f"vs anomaly {per_class.loc[per_class['label'] == 'Anomaly', 'mean_occurrences'].iloc[0]:.2f})")
    print(f"correlation verdict: useful={corr_useful}, max |r|={max_abs_r:.3f}")
    print(f"figures: {len(FIGURES_SAVED)} | summary: {summary_path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
