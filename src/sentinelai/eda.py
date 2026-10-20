"""Reusable exploratory-data-analysis toolkit for the HDFS_v1 dataset.

Every function operates on data loaded from the real files via
``sentinelai.data_loader.load_hdfs`` and returns computed frames/numbers —
nothing is synthesised, and no model is trained here.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

from sentinelai.data_loader import EVENT_COLUMNS, load_hdfs  # noqa: E402

NORMAL = 0
ANOMALY = 1

CLASS_NAMES = {NORMAL: "Normal", ANOMALY: "Anomaly"}

COLOR_NORMAL = "#2e7d32"
COLOR_ANOMALY = "#c62828"
COLOR_NEUTRAL = "#455a64"


def load_analysis_data() -> dict[str, pd.DataFrame]:
    return load_hdfs(require_templates=True)


def aligned_labels(matrix: pd.DataFrame, labels: pd.DataFrame) -> np.ndarray:
    """Ground-truth label per matrix row, joined on BlockId (order-agnostic)."""
    label_map = labels.set_index("BlockId")["Label"]
    return matrix["BlockId"].map(label_map).to_numpy()


def template_labels(templates: pd.DataFrame, max_len: int = 45) -> dict[str, str]:
    """Short plot labels like 'E5 · Receiving block[*]src:[*]dest:[*]'."""
    out: dict[str, str] = {}
    for _, row in templates.iterrows():
        event_id = str(row["EventId"])
        text = str(row["EventTemplate"])
        if len(text) > max_len:
            text = text[: max_len - 3] + "..."
        out[event_id] = f"{event_id} · {text}"
    return out


def class_distribution(labels: pd.DataFrame) -> pd.DataFrame:
    total = len(labels)
    counts = labels["Label"].value_counts().sort_index()
    out = pd.DataFrame(
        {
            "class": [CLASS_NAMES[NORMAL], CLASS_NAMES[ANOMALY]],
            "count": [int(counts.get(NORMAL, 0)), int(counts.get(ANOMALY, 0))],
        }
    )
    out["percent"] = out["count"] / total * 100
    return out


def event_frequency(matrix: pd.DataFrame) -> pd.DataFrame:
    """Total occurrences, presence, and mean-per-block for each event."""
    totals = matrix[EVENT_COLUMNS].sum(axis=0)
    presence = (matrix[EVENT_COLUMNS] > 0).sum(axis=0)
    out = pd.DataFrame(
        {
            "event": EVENT_COLUMNS,
            "total_occurrences": totals.to_numpy(),
            "blocks_with_event": presence.to_numpy(),
            "mean_per_block": totals.to_numpy() / len(matrix),
        }
    )
    return out.sort_values("total_occurrences", ascending=False).reset_index(
        drop=True
    )


def trace_statistics(
    matrix: pd.DataFrame, labels: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-trace totals/distinct events, overall and per class."""
    occurrences = matrix[EVENT_COLUMNS].sum(axis=1)
    distinct = (matrix[EVENT_COLUMNS] > 0).sum(axis=1)
    label = aligned_labels(matrix, labels)
    summary = pd.DataFrame(
        {
            "total_occurrences": occurrences,
            "distinct_events": distinct,
            "label": label,
        }
    )
    per_class = (
        summary.groupby("label")
        .agg(
            blocks=("total_occurrences", "size"),
            mean_occurrences=("total_occurrences", "mean"),
            median_occurrences=("total_occurrences", "median"),
            max_occurrences=("total_occurrences", "max"),
            mean_distinct=("distinct_events", "mean"),
            max_distinct=("distinct_events", "max"),
        )
        .reset_index()
    )
    per_class["label"] = per_class["label"].map(CLASS_NAMES)
    return summary, per_class


def class_conditional_rates(
    matrix: pd.DataFrame, labels: pd.DataFrame
) -> pd.DataFrame:
    """Mean occurrence rate per event for each class, with ratio/difference."""
    label = aligned_labels(matrix, labels)
    normal_mask = pd.Series(label == NORMAL, index=matrix.index)
    normal_rates = matrix.loc[normal_mask, EVENT_COLUMNS].mean(axis=0)
    anomaly_rates = matrix.loc[~normal_mask, EVENT_COLUMNS].mean(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = anomaly_rates / normal_rates
    out = pd.DataFrame(
        {
            "event": EVENT_COLUMNS,
            "normal_rate": normal_rates.to_numpy(),
            "anomaly_rate": anomaly_rates.to_numpy(),
            "ratio_anomaly_over_normal": ratio.to_numpy(),
            "abs_diff": (anomaly_rates - normal_rates).abs().to_numpy(),
        }
    )
    return out.sort_values("abs_diff", ascending=False).reset_index(drop=True)


def point_biserial_discriminativeness(
    matrix: pd.DataFrame, labels: pd.DataFrame
) -> pd.DataFrame:
    """Point-biserial correlation of each event count with the anomaly label.

    Constant columns (zero variance) yield NaN — the correlation is
    undefined there, and the event carries no discriminative information.
    """
    import warnings

    y = aligned_labels(matrix, labels)
    rows = []
    for col in EVENT_COLUMNS:
        x = matrix[col].to_numpy()
        if np.ptp(x) == 0:
            rows.append(
                {
                    "event": col,
                    "point_biserial_r": np.nan,
                    "p_value": np.nan,
                    "constant": True,
                }
            )
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r, p = stats.pointbiserialr(x, y)
        rows.append(
            {
                "event": col,
                "point_biserial_r": r,
                "p_value": p,
                "constant": False,
            }
        )
    out = pd.DataFrame(rows)
    out["abs_r"] = out["point_biserial_r"].abs()
    return (
        out.sort_values("abs_r", ascending=False, na_position="last")
        .drop(columns=["abs_r"])
        .reset_index(drop=True)
    )


def constant_and_rare_events(matrix: pd.DataFrame) -> pd.DataFrame:
    """Variance and presence profile of every event column."""
    rows = []
    for col in EVENT_COLUMNS:
        x = matrix[col]
        rows.append(
            {
                "event": col,
                "nunique": int(x.nunique()),
                "blocks_with_event": int((x > 0).sum()),
            }
        )
    return pd.DataFrame(rows)


def missing_values_report(
    data: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Columns with any missing value, per file (zero-missing columns omitted)."""
    rows = []
    for name, df in (
        ("labels", data["labels"]),
        ("occurrence_matrix", data["occurrence_matrix"]),
        ("traces", data["traces"]),
    ):
        for col, n in df.isna().sum().items():
            if n:
                rows.append(
                    {
                        "file": name,
                        "column": col,
                        "missing": int(n),
                        "percent": n / len(df) * 100,
                    }
                )
    return pd.DataFrame(rows).reset_index(drop=True)


def extreme_counts(
    matrix: pd.DataFrame, labels: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Events with the largest single-block counts; blocks with most events."""
    maxima = matrix[EVENT_COLUMNS].max(axis=0)
    per_event = pd.DataFrame(
        {"event": EVENT_COLUMNS, "max_count_in_one_block": maxima.to_numpy()}
    ).sort_values("max_count_in_one_block", ascending=False)

    totals = matrix[EVENT_COLUMNS].sum(axis=1)
    label = aligned_labels(matrix, labels)
    idx = totals.nlargest(5).index
    top_blocks = pd.DataFrame(
        {
            "block_id": matrix.loc[idx, "BlockId"].to_numpy(),
            "total_occurrences": totals.loc[idx].to_numpy(),
            "label": [CLASS_NAMES[int(v)] for v in label[idx]],
        }
    ).reset_index(drop=True)
    return per_event.reset_index(drop=True), top_blocks


def event_correlation_matrix(matrix: pd.DataFrame) -> pd.DataFrame:
    return matrix[EVENT_COLUMNS].corr(method="pearson")


def correlation_verdict(
    corr: pd.DataFrame, min_r: float = 0.3
) -> tuple[bool, float, list[tuple[str, str, float]]]:
    """(is_plot_useful, max_abs_r, top_pairs) over the off-diagonal."""
    upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
    stacked = upper.stack().dropna()
    top = stacked.reindex(stacked.abs().sort_values(ascending=False).index).head(5)
    pairs = [(str(i), str(j), float(v)) for (i, j), v in top.items()]
    max_r = float(top.abs().iloc[0]) if len(top) else 0.0
    return max_r >= min_r, max_r, pairs


# ---------------------------------------------------------------- plotting


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def save_class_distribution_plot(dist: pd.DataFrame, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = [COLOR_NORMAL, COLOR_ANOMALY]
    bars = ax.bar(dist["class"], dist["count"], color=colors)
    for bar, pct in zip(bars, dist["percent"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + dist["count"].max() * 0.01,
            f"{int(bar.get_height()):,}\n({pct:.2f}%)",
            ha="center",
            va="bottom",
            fontsize=11,
        )
    ax.set_ylabel("blocks")
    ax.set_ylim(0, dist["count"].max() * 1.18)
    ax.set_title(
        f"HDFS_v1 — class distribution ({int(dist['count'].sum()):,} blocks)"
    )
    return _save(fig, path)


def save_top_events_plot(
    freq: pd.DataFrame, tlabels: dict[str, str], path: Path, top_n: int = 15
) -> Path:
    top = freq.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(11, 8))
    bars = ax.barh(
        [tlabels.get(e, e) for e in top["event"]],
        top["total_occurrences"],
        color=COLOR_NEUTRAL,
    )
    for bar, value in zip(bars, top["total_occurrences"]):
        ax.text(
            bar.get_width() * 1.04,
            bar.get_y() + bar.get_height() / 2,
            f"{int(value):,}",
            va="center",
            fontsize=9,
        )
    ax.set_xscale("log")
    ax.set_xlabel("total occurrences (log scale)")
    ax.set_title(f"HDFS_v1 — top {top_n} events by total occurrences")
    return _save(fig, path)


def save_normal_vs_anomaly_plot(
    rates: pd.DataFrame, tlabels: dict[str, str], path: Path, top_n: int = 15
) -> Path:
    top = rates.head(top_n).iloc[::-1]
    y = np.arange(len(top))
    fig, ax = plt.subplots(figsize=(11, 8))
    height = 0.35
    ax.barh(
        y + height / 2,
        top["normal_rate"],
        height=height,
        label="Normal",
        color=COLOR_NORMAL,
    )
    ax.barh(
        y - height / 2,
        top["anomaly_rate"],
        height=height,
        label="Anomaly",
        color=COLOR_ANOMALY,
    )
    ax.set_yticks(y)
    ax.set_yticklabels([tlabels.get(e, e) for e in top["event"]], fontsize=9)
    ax.set_xlabel("mean occurrences per block")
    ax.legend()
    ax.set_title(
        f"HDFS_v1 — mean event rate per class (top {top_n} by |difference|)"
    )
    return _save(fig, path)


def save_event_count_distribution_plot(summary: pd.DataFrame, path: Path) -> Path:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    distinct_max = int(summary["distinct_events"].max())
    ax1.hist(
        summary["distinct_events"],
        bins=range(1, distinct_max + 2),
        color=COLOR_NEUTRAL,
    )
    ax1.set_title("distinct event types per trace")
    ax1.set_xlabel("distinct events")
    ax1.set_ylabel("blocks")
    ax2.hist(
        summary["total_occurrences"], bins=60, color=COLOR_NEUTRAL
    )
    ax2.set_yscale("log")
    ax2.set_title("total event occurrences per trace (log y)")
    ax2.set_xlabel("occurrences")
    ax2.set_ylabel("blocks")
    fig.suptitle("HDFS_v1 — event occurrence distribution per trace")
    return _save(fig, path)


def save_correlation_heatmap(
    corr: pd.DataFrame, top_pairs: list[tuple[str, str, float]], path: Path
) -> Path:
    fig, ax = plt.subplots(figsize=(10, 9))
    image = ax.imshow(corr.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(EVENT_COLUMNS)))
    ax.set_xticklabels(EVENT_COLUMNS, rotation=90, fontsize=8)
    ax.set_yticks(range(len(EVENT_COLUMNS)))
    ax.set_yticklabels(EVENT_COLUMNS, fontsize=8)
    fig.colorbar(image, ax=ax, label="Pearson r")
    pairs_text = "; ".join(f"{a}–{b}: {r:+.2f}" for a, b, r in top_pairs)
    ax.set_title(f"HDFS_v1 — event-count Pearson correlation\nstrongest: {pairs_text}")
    return _save(fig, path)
