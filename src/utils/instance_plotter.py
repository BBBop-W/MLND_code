"""Static study-window plots for canonical unified-MLND instances."""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from src.model_contract import FORMULATION_ID as RESULT_FORMULATION_ID
from src.model_contract import RESULT_SCHEMA_VERSION


def _require_current_summary_contract(summary: object) -> None:
    """Reject foreign or numerically coerced result-schema declarations."""

    if not isinstance(summary, dict):
        raise ValueError("solution layout summary must contain a JSON object")
    schema_version = summary.get("schema_version")
    formulation_id = summary.get("formulation_id")
    if (
        type(schema_version) is not int
        or schema_version != RESULT_SCHEMA_VERSION
        or not isinstance(formulation_id, str)
        or formulation_id != RESULT_FORMULATION_ID
    ):
        raise ValueError(
            f"solution layout requires summary schema_version={RESULT_SCHEMA_VERSION} and "
            f"formulation_id={RESULT_FORMULATION_ID}; legacy results cannot be mixed"
        )


@dataclass(frozen=True)
class StudyLayout:
    """Plot-ready data read from one canonical instance directory."""

    metadata: dict[str, object]
    nodes: tuple[dict[str, object], ...]
    sites: tuple[dict[str, object], ...]
    endpoint_rates: dict[str, float]

    @property
    def half_size_km(self) -> float:
        config = self.metadata["data_config"]
        return float(config["map_half_size_km"])

    @property
    def od_grid_km(self) -> float:
        config = self.metadata["data_config"]
        return float(config["od_grid_km"])


def load_study_layout(instance_dir: str | Path) -> StudyLayout:
    """Load only the canonical files needed for an input-layout figure."""

    root = Path(instance_dir)
    with (root / "metadata.json").open(encoding="utf-8") as handle:
        metadata = json.load(handle)

    nodes: list[dict[str, object]] = []
    with (root / "od_nodes.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            nodes.append(
                {
                    "id": row["node_id"],
                    "role": row["role"],
                    "x": float(row["x_km"]),
                    "y": float(row["y_km"]),
                }
            )

    sites: list[dict[str, object]] = []
    with (root / "sites.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            sites.append(
                {
                    "id": row["site_id"],
                    "type": row["type"],
                    "x": float(row["x_km"]),
                    "y": float(row["y_km"]),
                    "activity_rate": float(row["activity_rate"]),
                }
            )

    endpoint_rates: dict[str, float] = defaultdict(float)
    with (root / "od_demand.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rate = float(row["lambda_per_hour"])
            endpoint_rates[row["origin_node"]] += rate
            endpoint_rates[row["destination_node"]] += rate

    return StudyLayout(
        metadata=metadata,
        nodes=tuple(nodes),
        sites=tuple(sites),
        endpoint_rates=dict(endpoint_rates),
    )


def plot_study_window(
    instance_dir: str | Path,
    output_stem: str | Path | None = None,
) -> tuple[Path, Path]:
    """Render the study window in the repository's historical Nanshan style."""

    root = Path(instance_dir)
    layout = load_study_layout(root)
    stem = Path(output_stem) if output_stem else root / "study_window_layout"
    stem.parent.mkdir(parents=True, exist_ok=True)

    half = layout.half_size_km
    grid = layout.od_grid_km
    edges = np.arange(-half, half + grid * 1.01, grid)
    node_x = np.asarray([float(node["x"]) for node in layout.nodes])
    node_y = np.asarray([float(node["y"]) for node in layout.nodes])
    node_rate = np.asarray(
        [layout.endpoint_rates.get(str(node["id"]), 0.0) for node in layout.nodes]
    )
    heat, x_edges, y_edges = np.histogram2d(
        node_x,
        node_y,
        bins=(edges, edges),
        weights=node_rate,
    )
    x_centres = (x_edges[:-1] + x_edges[1:]) / 2.0
    y_centres = (y_edges[:-1] + y_edges[1:]) / 2.0
    activity_x, activity_y = np.meshgrid(x_centres, y_centres, indexing="ij")
    positive = heat > 0.0
    activity_rate = heat[positive]
    max_rate = max(float(activity_rate.max()), 1.0)
    activity_size = 10.0 + 80.0 * np.sqrt(activity_rate / max_rate)

    fig, ax = plt.subplots(figsize=(10.0, 8.6))
    ax.scatter(
        activity_x[positive],
        activity_y[positive],
        s=activity_size,
        c="#777777",
        alpha=0.18,
        linewidths=0,
        zorder=1,
    )

    # These colors, markers, and relative sizes reproduce the historical
    # Nanshan candidate-point figures.  H is drawn last so a co-located H/L
    # pair remains visible as a red circle nested in a blue triangle.
    site_styles = {
        "H": {"label": "H candidates", "marker": "o", "color": "#d73a31", "size": 42},
        "L": {"label": "L candidates", "marker": "^", "color": "#1f6feb", "size": 78},
        "E": {"label": "E candidates", "marker": "s", "color": "#9a6700", "size": 58},
    }
    site_counts: dict[str, int] = defaultdict(int)
    for site_type in ("L", "E", "H"):
        style = site_styles[site_type]
        sites = [site for site in layout.sites if site["type"] == site_type]
        site_counts[site_type] = len(sites)
        ax.scatter(
            [float(site["x"]) for site in sites],
            [float(site["y"]) for site in sites],
            marker=str(style["marker"]),
            s=float(style["size"]),
            c=str(style["color"]),
            edgecolors="white",
            linewidths=0.7,
            zorder=6 if site_type == "H" else 5,
        )

    layer_handles = [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            markerfacecolor="#777777",
            markeredgecolor="none",
            alpha=0.28,
            markersize=6.5,
            label="OD endpoint activity",
        )
    ]
    layer_handles.extend(
        Line2D(
            [],
            [],
            marker=str(style["marker"]),
            linestyle="none",
            markerfacecolor=str(style["color"]),
            markeredgecolor="white",
            markeredgewidth=0.7,
            markersize=7.5,
            label=f"{style['label']} (n={site_counts[site_type]})",
        )
        for site_type, style in site_styles.items()
    )
    layer_legend = ax.legend(
        handles=layer_handles,
        loc="upper left",
        frameon=True,
        framealpha=0.92,
        facecolor="white",
        edgecolor="#cccccc",
        fontsize=9,
    )
    layer_legend.set_zorder(10)

    reference_rates = (10.0, 100.0, 400.0)
    size_handles = [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            markerfacecolor="#777777",
            markeredgecolor="none",
            alpha=0.28,
            markersize=np.sqrt(10.0 + 80.0 * np.sqrt(rate / max_rate)),
            label=f"{rate:g}",
        )
        for rate in reference_rates
    ]
    size_legend = ax.legend(
        handles=size_handles,
        title="Endpoint rate (orders/hour)",
        loc="upper right",
        frameon=True,
        framealpha=0.92,
        facecolor="white",
        edgecolor="#cccccc",
        fontsize=8,
        title_fontsize=8.5,
        labelspacing=0.8,
    )
    size_legend.set_zorder(10)
    ax.add_artist(layer_legend)

    metadata = layout.metadata
    config = metadata["data_config"]
    demand_start = int(config["demand_hour_start"])
    demand_end = int(config["demand_hour_end"])
    source_days = len(metadata.get("selected_days", ()))
    demand = float(metadata["total_demand_per_hour"])
    candidate_count = site_counts["H"]
    source = str(metadata.get("source", "instance"))
    source_label = {
        "nanshan": "Meituan Shenzhen Nanshan",
        "tsl": "TSL",
    }.get(source.casefold(), source.replace("_", " ").title())
    window_size = 2.0 * half
    fig.suptitle(
        f"{source_label} candidate sites (H{candidate_count})",
        y=0.975,
        fontsize=15,
        fontweight="normal",
        color="#111111",
    )
    ax.set_title(
        f"{window_size:g} × {window_size:g} km local window · "
        f"{demand_start:02d}:00–{demand_end:02d}:00 · "
        f"averaged over {source_days} source days · total demand {demand:,.1f} orders/hour",
        fontsize=9.5,
        color="#555555",
        pad=9,
    )
    ax.set_xlim(-half, half)
    ax.set_ylim(-half, half)
    ax.set_aspect("equal", adjustable="box")
    ticks = np.arange(-half, half + 0.1, 2.0)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xlabel("Local x coordinate (km)")
    ax.set_ylabel("Local y coordinate (km)")
    ax.grid(True, color="#777777", linewidth=0.45, alpha=0.22, zorder=0)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color("#222222")
        spine.set_linewidth(0.8)

    fig.text(
        0.10,
        0.018,
        f"OD endpoint bubbles aggregate each order once at origin and once at destination; "
        f"rates are orders/hour per {grid:g} km cell.",
        ha="left",
        fontsize=8,
        color="#666666",
    )
    fig.subplots_adjust(left=0.10, right=0.97, bottom=0.10, top=0.90)

    png_path = stem.with_suffix(".png")
    svg_path = stem.with_suffix(".svg")
    fig.savefig(png_path, dpi=180, bbox_inches="tight", facecolor="white")
    fig.savefig(svg_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return png_path, svg_path


def plot_solution_window(
    instance_dir: str | Path,
    mode_result_dir: str | Path,
    output_stem: str | Path | None = None,
) -> tuple[Path, Path]:
    """Rebuild a clean station-layout figure from saved solution artifacts."""

    root = Path(instance_dir)
    result_root = Path(mode_result_dir)
    layout = load_study_layout(root)
    stem = Path(output_stem) if output_stem else result_root / "solution_layout"
    stem.parent.mkdir(parents=True, exist_ok=True)

    with (result_root / "summary.json").open(encoding="utf-8") as handle:
        summary = json.load(handle)
    _require_current_summary_contract(summary)
    with (result_root / "open_facilities.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        open_facilities = tuple(csv.DictReader(handle))
    with (result_root / "corridor_flows.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        corridor_flows = tuple(csv.DictReader(handle))

    half = layout.half_size_km
    grid = layout.od_grid_km
    edges = np.arange(-half, half + grid * 1.01, grid)
    node_x = np.asarray([float(node["x"]) for node in layout.nodes])
    node_y = np.asarray([float(node["y"]) for node in layout.nodes])
    node_rate = np.asarray(
        [layout.endpoint_rates.get(str(node["id"]), 0.0) for node in layout.nodes]
    )
    heat, x_edges, y_edges = np.histogram2d(
        node_x,
        node_y,
        bins=(edges, edges),
        weights=node_rate,
    )
    x_centres = (x_edges[:-1] + x_edges[1:]) / 2.0
    y_centres = (y_edges[:-1] + y_edges[1:]) / 2.0
    activity_x, activity_y = np.meshgrid(x_centres, y_centres, indexing="ij")
    positive_activity = heat > 0.0
    activity_rate = heat[positive_activity]
    max_activity = max(float(activity_rate.max()), 1.0)
    activity_size = 9.0 + 72.0 * np.sqrt(activity_rate / max_activity)

    site_by_id = {str(site["id"]): site for site in layout.sites}
    loaded_by_pair: dict[tuple[str, str], float] = defaultdict(float)
    for row in corridor_flows:
        loaded = float(row["loaded_psi_per_hour"])
        if loaded <= 1e-7:
            continue
        origin = row["origin_locker"]
        destination = row["destination_locker"]
        pair = tuple(sorted((origin, destination)))
        loaded_by_pair[pair] += loaded
    max_loaded = max(loaded_by_pair.values(), default=1.0)

    fig, ax = plt.subplots(figsize=(10.2, 8.9))
    ax.scatter(
        activity_x[positive_activity],
        activity_y[positive_activity],
        s=activity_size,
        c="#777777",
        alpha=0.17,
        linewidths=0,
        zorder=1,
    )

    for (origin, destination), loaded in sorted(
        loaded_by_pair.items(), key=lambda item: item[1]
    ):
        origin_site = site_by_id[origin]
        destination_site = site_by_id[destination]
        width = 0.45 + 3.0 * np.sqrt(loaded / max_loaded)
        ax.plot(
            [float(origin_site["x"]), float(destination_site["x"])],
            [float(origin_site["y"]), float(destination_site["y"])],
            color="#374151",
            linewidth=width,
            alpha=0.52,
            solid_capstyle="round",
            zorder=2,
        )

    candidate_styles = {
        "H": {"marker": "o", "size": 58},
        "L": {"marker": "s", "size": 74},
        "E": {"marker": "D", "size": 68},
    }
    for site_type in ("L", "E", "H"):
        sites = [site for site in layout.sites if site["type"] == site_type]
        style = candidate_styles[site_type]
        ax.scatter(
            [float(site["x"]) for site in sites],
            [float(site["y"]) for site in sites],
            marker=str(style["marker"]),
            s=float(style["size"]),
            facecolors="none",
            edgecolors="#8b8f94",
            linewidths=0.9,
            zorder=3,
        )

    open_styles = {
        "base": {
            "code": "L",
            "label": "Open L (base)",
            "marker": "s",
            "color": "#d62728",
            "size": 118,
        },
        "locker": {
            "code": "H",
            "label": "Open H (locker)",
            "marker": "o",
            "color": "#1f77b4",
            "size": 72,
        },
        "swap": {
            "code": "E",
            "label": "Open E (swap)",
            "marker": "D",
            "color": "#2ca02c",
            "size": 104,
        },
    }
    open_counts = {facility_type: 0 for facility_type in open_styles}
    for facility_type in ("base", "swap", "locker"):
        rows = [row for row in open_facilities if row["type"] == facility_type]
        open_counts[facility_type] = len(rows)
        if not rows:
            continue
        style = open_styles[facility_type]
        ax.scatter(
            [float(row["x_km"]) for row in rows],
            [float(row["y_km"]) for row in rows],
            marker=str(style["marker"]),
            s=float(style["size"]),
            c=str(style["color"]),
            edgecolors="white",
            linewidths=0.9,
            zorder=6 if facility_type == "locker" else 5,
        )

    rho_rows = []
    for row in open_facilities:
        rho_text = row["rho"].strip()
        if row["type"] not in {"base", "locker"} or not rho_text:
            continue
        rho = float(rho_text)
        if np.isfinite(rho):
            rho_rows.append((row, rho))
    label_offsets = ((4, 5), (4, -10), (-4, 5), (-4, -10))
    for label_index, (row, rho) in enumerate(rho_rows):
        rho_digits = f"{rho:.2f}".removeprefix("0")
        rho_symbol = "rho" if row["type"] == "base" else "rho_v"
        rho_symbol = "ρ" if rho_symbol == "rho" else "ρv"
        x_offset, y_offset = label_offsets[label_index % len(label_offsets)]
        ax.annotate(
            f"{row['facility']}  {rho_symbol}={rho_digits}",
            (float(row["x_km"]), float(row["y_km"])),
            xytext=(x_offset, y_offset),
            textcoords="offset points",
            ha="left" if x_offset > 0 else "right",
            va="bottom" if y_offset > 0 else "top",
            fontsize=7.2,
            color="#202124",
            bbox={
                "boxstyle": "round,pad=0.18",
                "facecolor": "white",
                "edgecolor": "#d1d5db",
                "linewidth": 0.45,
                "alpha": 0.82,
            },
            zorder=8,
        )

    legend_handles: list[Line2D] = []
    for site_type in ("H", "L", "E"):
        style = candidate_styles[site_type]
        legend_handles.append(
            Line2D(
                [],
                [],
                marker=str(style["marker"]),
                linestyle="none",
                markerfacecolor="none",
                markeredgecolor="#8b8f94",
                markeredgewidth=0.9,
                markersize=7,
                label=f"Candidate {site_type}",
            )
        )
    for facility_type in ("locker", "base", "swap"):
        style = open_styles[facility_type]
        legend_handles.append(
            Line2D(
                [],
                [],
                marker=str(style["marker"]),
                linestyle="none",
                markerfacecolor=str(style["color"]),
                markeredgecolor="white",
                markeredgewidth=0.8,
                markersize=8,
                label=f"{style['label']} (n={open_counts[facility_type]})",
            )
        )
    if loaded_by_pair:
        legend_handles.append(
            Line2D(
                [],
                [],
                color="#374151",
                linewidth=2.0,
                alpha=0.65,
                label="Loaded H–H flow",
            )
        )
    layer_legend = ax.legend(
        handles=legend_handles,
        loc="upper left",
        ncol=2,
        frameon=True,
        framealpha=0.92,
        facecolor="white",
        edgecolor="#cccccc",
        fontsize=8.2,
        columnspacing=1.0,
        handletextpad=0.55,
    )
    layer_legend.set_zorder(10)

    reference_rates = (100.0, 400.0)
    size_handles = [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            markerfacecolor="#777777",
            markeredgecolor="none",
            alpha=0.25,
            markersize=np.sqrt(9.0 + 72.0 * np.sqrt(rate / max_activity)),
            label=f"{rate:g}",
        )
        for rate in reference_rates
    ]
    size_legend = ax.legend(
        handles=size_handles,
        title="Endpoint demand (orders/hour)",
        loc="upper right",
        frameon=True,
        framealpha=0.92,
        facecolor="white",
        edgecolor="#cccccc",
        fontsize=8,
        title_fontsize=8.3,
        labelspacing=0.75,
    )
    size_legend.set_zorder(10)
    ax.add_artist(layer_legend)

    mode = str(summary["mode"])
    funnel = summary["demand_funnel_orders_per_hour"]
    realized = float(funnel["total_drone_realized"])
    baseline = summary["all_rider_baseline"]
    saving = float(baseline["savings_per_hour"])
    saving_fraction = baseline["savings_fraction"]
    saving_percent = (
        100.0 * float(saving_fraction) if saving_fraction is not None else None
    )
    gap = summary["mip_gap"]
    saving_text = (
        f"{saving:,.1f} CNY/hour ({saving_percent:.2f}%)"
        if saving_percent is not None
        else f"{saving:,.1f} CNY/hour"
    )
    gap_text = f"{float(gap):.2%}" if gap is not None else "N/A"
    fig.suptitle(
        f"Meituan Shenzhen Nanshan opened-station layout — MLND {mode}",
        y=0.976,
        fontsize=15,
        fontweight="normal",
        color="#111111",
    )
    ax.set_title(
        f"Open L/H/E = {open_counts['base']}/{open_counts['locker']}/{open_counts['swap']}"
        f"  ·  realized = {realized:,.1f} orders/hour"
        f"  ·  saving = {saving_text}"
        f"  ·  MIP gap = {gap_text}",
        fontsize=9.4,
        color="#555555",
        pad=9,
    )
    ax.set_xlim(-half, half)
    ax.set_ylim(-half, half)
    ax.set_aspect("equal", adjustable="box")
    ticks = np.arange(-half, half + 0.1, 2.0)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xlabel("Local x coordinate (km)")
    ax.set_ylabel("Local y coordinate (km)")
    ax.grid(True, color="#777777", linewidth=0.45, alpha=0.20, zorder=0)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color("#222222")
        spine.set_linewidth(0.8)

    fig.text(
        0.10,
        0.018,
        "Gray bubbles show OD endpoint demand; corridor width scales with loaded physical "
        "flow aggregated over both H–H directions. Both are measured in orders/hour.",
        ha="left",
        fontsize=8,
        color="#666666",
    )
    fig.subplots_adjust(left=0.10, right=0.97, bottom=0.10, top=0.90)

    png_path = stem.with_suffix(".png")
    svg_path = stem.with_suffix(".svg")
    fig.savefig(png_path, dpi=180, bbox_inches="tight", facecolor="white")
    fig.savefig(svg_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return png_path, svg_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot a canonical MLND OD study window"
    )
    parser.add_argument("instance_dir")
    parser.add_argument("--output-stem", default=None)
    parser.add_argument(
        "--solution-dir",
        default=None,
        help="Mode result directory containing summary/open-facility/flow artifacts",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = _parse_args()
    if arguments.solution_dir:
        paths = plot_solution_window(
            arguments.instance_dir,
            arguments.solution_dir,
            arguments.output_stem,
        )
    else:
        paths = plot_study_window(arguments.instance_dir, arguments.output_stem)
    print("\n".join(str(path.resolve()) for path in paths))
