"""Figures for prepared scale ladders and independently validated saved runs.

No function starts a solver, reads an unvalidated incumbent, or substitutes a
number for missing evidence. Paths returned by each function include PNG and PDF.
The caller owns run selection and certificate verification; numerical run rows
must have ``state == 'validated'``. Phase/progress rows must already be restricted
to those runs (an explicit non-validated state is rejected here as well).
"""

from __future__ import annotations

from collections import defaultdict
import csv
import json
import math
from pathlib import Path

import numpy as np

from .style import _grid, _save, _style, plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle


METHODS = {
    "full_cplex": ("CPLEX full", "#31688e"),
    "full_gurobi": ("Gurobi full", "#bd7b27"),
    "benders_cplex": ("CPLEX Benders", "#7d729a"),
}
PHASES = {
    "input_support": ("Input / physical support", "#b9c8d0"),
    "initialization": ("Initial columns / cleanup", "#8d9c85"),
    "master": ("Master / root", "#31688e"),
    "restricted_mip": ("Restricted MISOCP", "#d0a251"),
    "capacity": ("Capacity cuts", "#af7f9b"),
    "subproblem": ("Subproblem primal / certificate", "#4f7570"),
    "dual": ("Ordinary dual", "#729ba8"),
    "pricing": ("Pricing", "#b6ac8c"),
    "add_columns": ("Add columns", "#bd7b27"),
    "pareto": ("Pareto auxiliary", "#7d729a"),
    "full_solve": ("Full model build / solve / read", "#6b889a"),
    "other": ("Other measured time", "#737e87"),
}


def _number(value):
    try:
        result = float(value)
    except (ValueError, TypeError):
        return None
    return result if math.isfinite(result) else None


def _read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _xy(rows):
    return ([float(r["x_km"]) for r in rows], [float(r["y_km"]) for r in rows])


def _export(fig, output, name):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    _save(fig, output, name)
    return [output / f"{name}.{ext}" for ext in ("png", "pdf")]


def _validated(rows):
    return [r for r in rows if r.get("state") == "validated"]


def _eligible_detail(rows):
    return [r for r in rows if r.get("state", "validated") == "validated"]


def _level(row):
    value = str(row["level"])
    return int(value[1:] if value.startswith("L") else value)


def _k(row):
    return int(str(row["K"]).removeprefix("K"))


def _run_label(row):
    return f"L{_level(row):02d} · K={_k(row)} · {METHODS.get(row['method'], (row['method'],))[0]}"


def plot_preparation(batch: Path, output: Path) -> list[Path]:
    """Plot candidate points and exact crops on a shared local-kilometre scale."""
    batch = Path(batch)
    manifest = json.loads((batch / "preparation_manifest.json").read_text())
    levels = sorted(manifest.get("levels", []), key=_level)
    if not levels:
        return []
    _style()
    ncols = 5 if len(levels) > 4 else len(levels)
    nrows = math.ceil(len(levels) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.1 * ncols, 4.1 * nrows),
                             squeeze=False, sharex=True, sharey=True)
    bounds = [r["crop_bounds_local_km"] for r in levels]
    xlo, xhi = min(b["x_min"] for b in bounds), max(b["x_max"] for b in bounds)
    ylo, yhi = min(b["y_min"] for b in bounds), max(b["y_max"] for b in bounds)
    span = max(xhi - xlo, yhi - ylo)
    cx, cy = (xhi + xlo) / 2, (yhi + ylo) / 2
    for ax, level in zip(axes.flat, levels):
        instance = batch / level["instance_path"]
        sites = _read_csv(instance / "sites.csv")
        nodes = _read_csv(instance / "od_nodes.csv")
        for role, marker, color in (("origin", ".", "#a8b3bc"),
                                     ("destination", "x", "#c5a985")):
            points = [r for r in nodes if r["role"] == role]
            if points:
                ax.scatter(*_xy(points), s=10, marker=marker, c=color, alpha=.65,
                           linewidths=.6, zorder=1)
        for kind, marker, color, size in (("H", "o", "#31688e", 16),
                                         ("L", "^", "#bd7b27", 55)):
            points = [r for r in sites if r["type"] == kind]
            ax.scatter(*_xy(points), s=size, marker=marker, c=color,
                       edgecolors="white", linewidths=.35, zorder=3)
        b = level["crop_bounds_local_km"]
        ax.add_patch(Rectangle((b["x_min"], b["y_min"]), b["x_max"] - b["x_min"],
                               b["y_max"] - b["y_min"], fill=False, ec="#263943",
                               ls="--", lw=1, zorder=2))
        title = f"L{_level(level):02d}: {level['n_l']}L / {level['n_h']}H / {level['n_od']} OD"
        demand = _number(level.get("hourly_demand"))
        ax.set_title(title + (f"\nDemand {demand:,.1f}/h" if demand is not None else ""), fontsize=12)
        ax.set(xlim=(cx - .52 * span, cx + .52 * span),
               ylim=(cy - .52 * span, cy + .52 * span))
        ax.set_aspect("equal", adjustable="box")
        _grid(ax, "both")
        ax.tick_params(labelsize=9)
    for ax in list(axes.flat)[len(levels):]:
        ax.set_visible(False)
    handles = [Line2D([], [], marker=m, ls="none", color=c, label=label, markersize=s)
               for m, c, label, s in (("^", "#bd7b27", "Base candidates", 8),
                                   ("o", "#31688e", "Locker candidates", 6),
                                   (".", "#a8b3bc", "Origin nodes", 6),
                                   ("x", "#c5a985", "Destination nodes", 5))]
    handles.append(Line2D([], [], ls="--", color="#263943", label="Exact crop boundary"))
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False,
               bbox_to_anchor=(.5, .015))
    fig.suptitle("Prepared real-data instances · identical spatial scale in every panel", fontsize=17)
    fig.supxlabel("Local x (km); no external basemap; candidate sets need not be nested", y=.065, fontsize=11)
    fig.supylabel("Local y (km)")
    fig.tight_layout(rect=(.02, .10, 1, .95), h_pad=2)
    paths = _export(fig, output, "preparation_candidates")

    fig, (ax, table_ax) = plt.subplots(1, 2, figsize=(14, 8),
                                      gridspec_kw={"width_ratios": [1.35, 1]})
    colors = plt.get_cmap("viridis")(np.linspace(.12, .88, len(levels)))
    table_rows = []
    centers = []
    for index, (level, color) in enumerate(zip(levels, colors)):
        b = level["crop_bounds_local_km"]
        width, height = b["x_max"] - b["x_min"], b["y_max"] - b["y_min"]
        centers.append(((b["x_max"] + b["x_min"]) / 2, (b["y_max"] + b["y_min"]) / 2))
        ax.add_patch(Rectangle((b["x_min"], b["y_min"]), width, height, fill=False,
                               ec=color, lw=1.8, ls="-" if index % 2 == 0 else "--"))
        table_rows.append([f"L{_level(level):02d}",
                           f"{level['n_l']} / {level['n_h']} / {level['n_od']}",
                           f"{width:.2f} × {height:.2f}", f"{width * height:.2f}"])
    if all(math.isclose(x, centers[0][0], abs_tol=1e-8) and
           math.isclose(y, centers[0][1], abs_tol=1e-8) for x, y in centers):
        ax.plot(*centers[0], marker="+", color="#263943", ms=10, mew=1.5)
        ax.annotate("Common crop center", centers[0], xytext=(8, 8),
                    textcoords="offset points", fontsize=10)
    ax.set(xlim=(cx - .54 * span, cx + .54 * span),
           ylim=(cy - .54 * span, cy + .54 * span),
           xlabel="Local x (km)", ylabel="Local y (km)", title="All exact crop boundaries")
    ax.set_aspect("equal", adjustable="box")
    _grid(ax, "both")
    table_ax.axis("off")
    table = table_ax.table(cellText=table_rows,
                           colLabels=["Level", "L / H / OD", "Width × height\n(km)", "Area\n(km²)"],
                           colWidths=[.15, .28, .35, .22], loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.1)
    for (row_index, column), cell in table.get_celld().items():
        cell.set_edgecolor("#e0e4e7")
        if row_index == 0:
            cell.set_facecolor("#edf1f4")
            cell.set_text_props(weight="bold")
        elif column == 0:
            cell.set_facecolor(colors[row_index - 1])
            cell.set_text_props(color="white" if row_index <= len(levels) / 2 else "#22313d", weight="bold")
    fig.suptitle("Scale-ladder crop overview", fontsize=17)
    fig.text(.5, .025, "Local coordinates; no external basemap. Boundaries show sampling areas, not selected facilities.\n"
             "Per-level candidate points are shown separately; candidate sets need not be nested.",
             ha="center", fontsize=11)
    fig.tight_layout(rect=(0, .09, 1, .94))
    paths += _export(fig, output, "preparation_nested_crops")
    return paths


def _comparison_axes(rows, title, ylabel):
    ks = sorted({_k(r) for r in rows})
    levels = sorted({_level(r) for r in rows})
    fig, axes = plt.subplots(1, len(ks), figsize=(7.2 * len(ks), 5.3), squeeze=False)
    for ax, k in zip(axes.flat, ks):
        ax.set(title=f"K={k}", xlabel="Instance level", ylabel=ylabel,
               xticks=levels, xticklabels=[f"L{n:02d}" for n in levels])
        ax.tick_params(axis="x", labelsize=9)
        _grid(ax, "y")
    fig.suptitle(title, fontsize=16)
    return fig, list(axes.flat), ks


def plot_comparisons(rows: list[dict], output: Path) -> list[Path]:
    """Objective bounds, relative gaps and timing, only for validated run rows."""
    rows = [r for r in _validated(rows) if r.get("method") in METHODS]
    if not rows:
        return []
    _style()
    paths = []
    objective = [r for r in rows if _number(r.get("upper_bound")) is not None
                 or _number(r.get("lower_bound")) is not None]
    if objective:
        fig, axes, ks = _comparison_axes(objective, "Validated feasible costs and certified lower bounds", "Cost (CNY/h)")
        for ax, k in zip(axes, ks):
            for index, (method, (name, color)) in enumerate(METHODS.items()):
                selected = sorted([r for r in objective if _k(r) == k and r["method"] == method], key=_level)
                for r in selected:
                    x = _level(r) + (index - 1) * .19
                    ub, lb = _number(r.get("upper_bound")), _number(r.get("lower_bound"))
                    if ub is not None and lb is not None:
                        ax.plot([x, x], [lb, ub], color=color, lw=1.5, alpha=.65)
                    if ub is not None:
                        ax.scatter(x, ub, color=color, marker="o", s=35)
                    if lb is not None:
                        ax.scatter(x, lb, edgecolors=color, marker="v", s=35, facecolors="none")
        handles = [Line2D([], [], color=c, marker="o", ls="none", label=n) for n, c in METHODS.values()]
        handles += [Line2D([], [], color="#555555", marker="o", ls="none", label="Feasible UB"),
                    Line2D([], [], color="#555555", marker="v", markerfacecolor="none", ls="none", label="Certified LB")]
        fig.legend(handles=handles, ncol=5, loc="lower center", frameon=False)
        fig.tight_layout(rect=(0, .09, 1, .94))
        paths += _export(fig, output, "objective_bounds")
    gaps = [r for r in rows if _number(r.get("gap")) is not None]
    if gaps:
        fig, axes, ks = _comparison_axes(gaps, "Validated saved optimality gaps", "Relative gap (%)")
        for ax, k in zip(axes, ks):
            for index, (method, (name, color)) in enumerate(METHODS.items()):
                selected = [r for r in gaps if _k(r) == k and r["method"] == method]
                ax.scatter([_level(r) + (index - 1) * .19 for r in selected],
                           [100 * float(r["gap"]) for r in selected], s=40, color=color, label=name)
            ax.set_ylim(bottom=0)
        axes[0].legend(frameon=False)
        fig.tight_layout(rect=(0, 0, 1, .94))
        paths += _export(fig, output, "optimality_gaps")
    timed = [r for r in rows if any(_number(r.get(c)) is not None for c in ("solve_seconds", "external_wall_seconds"))]
    if timed:
        fig, axes, ks = _comparison_axes(timed, "Recorded run times · solver budget and external process wall", "Seconds")
        for ax, k in zip(axes, ks):
            for index, (method, (name, color)) in enumerate(METHODS.items()):
                for r in [r for r in timed if _k(r) == k and r["method"] == method]:
                    x = _level(r) + (index - 1) * .19
                    for field, marker, filled in (("solve_seconds", "o", True), ("external_wall_seconds", "s", False)):
                        value = _number(r.get(field))
                        if value is not None:
                            ax.scatter(x, value, marker=marker, edgecolors=color,
                                       facecolors=color if filled else "none", s=38)
            ax.set_ylim(bottom=0)
        handles = [Line2D([], [], color=c, marker="o", ls="none", label=n) for n, c in METHODS.values()]
        handles += [Line2D([], [], color="#555555", marker="o", ls="none", label="Inclusive solve time"),
                    Line2D([], [], color="#555555", marker="s", markerfacecolor="none", ls="none", label="External wall time")]
        fig.legend(handles=handles, ncol=3, loc="lower center", frameon=False)
        fig.tight_layout(rect=(0, .13, 1, .94))
        paths += _export(fig, output, "run_times")
    return paths


def _phase_group(phase):
    subproblem = phase.startswith(("subproblem.", "subproblem_"))
    phase = phase.removeprefix("subproblem.")
    if "pareto" in phase:
        return "pareto"
    if "add_columns" in phase:
        return "add_columns"
    if "pricing" in phase:
        return "pricing"
    if "dual" in phase:
        return "dual"
    if "capacity" in phase:
        return "capacity"
    if "restricted_mip" in phase:
        return "restricted_mip"
    if "master" in phase:
        return "master"
    if phase in ("read", "enumeration", "support_preprocessing", "io_enumeration"):
        return "input_support"
    if "initial" in phase or "cleanup" in phase:
        return "initialization"
    if subproblem or phase in ("primal", "certificate", "validate"):
        return "subproblem"
    if phase in ("build", "optimize", "read_solution", "dispose"):
        return "full_solve"
    return "other"


def plot_phases(phase_rows: list[dict], output: Path) -> list[Path]:
    """Aggregate disjoint phase seconds without adding parent or overlap timers."""
    groups = {}
    for row in _eligible_detail(phase_rows):
        if row["phase"] in ("total", "wall", "subproblem_wall", "subproblem.wall"):
            raise ValueError(f"Parent/overlapping timer is not a disjoint phase: {row['phase']}")
        seconds = _number(row.get("seconds"))
        if seconds is None:
            continue
        if seconds < 0:
            raise ValueError(f"Negative disjoint phase time: {row}")
        key = (str(row["id"]), _level(row), _k(row), row["method"])
        phases = groups.setdefault(key, defaultdict(float))
        phases[_phase_group(row["phase"])] += seconds
    if not groups:
        return []
    _style()
    paths = []
    for k in sorted({key[2] for key in groups}):
        keys = sorted([key for key in groups if key[2] == k], key=lambda key: (key[1], key[3], key[0]))
        active = [p for p in PHASES if any(groups[key].get(p, 0) > 0 for key in keys)]
        if not active:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(16, max(5, .34 * len(keys) + 2.4)), sharey=True)
        y = np.arange(len(keys))
        total = np.array([sum(groups[key].values()) for key in keys])
        for ax, normalized in zip(axes, (False, True)):
            left = np.zeros(len(keys))
            for phase in active:
                values = np.array([groups[key].get(phase, 0) for key in keys])
                if normalized:
                    values = np.divide(100 * values, total, out=np.zeros_like(values), where=total > 0)
                label, color = PHASES[phase]
                ax.barh(y, values, left=left, color=color, label=label, height=.75)
                left += values
            ax.set(xlabel="Share of recorded phases (%)" if normalized else "Seconds",
                   yticks=y, yticklabels=[f"L{key[1]:02d} · {METHODS.get(key[3], (key[3],))[0]}" for key in keys])
            _grid(ax)
        axes[0].invert_yaxis()
        axes[1].set_xlim(0, 100)
        fig.suptitle(f"K={k} · disjoint measured phase composition", fontsize=16)
        fig.legend(*axes[0].get_legend_handles_labels(), ncol=4, loc="lower center", frameon=False, fontsize=9)
        fig.tight_layout(rect=(0, min(.22, 1.3 / fig.get_figheight()), 1, .96))
        paths += _export(fig, output, f"phase_composition_K{k}")
    return paths


def plot_operations(record: dict, output: Path) -> list[Path]:
    """Map validated open/closed assets and show availability, flow and fleet."""
    row = record["row"]
    if row.get("state") != "validated":
        return []
    operations = record.get("operations", {})
    if operations.get("summary", {}).get("checks", {}).get("passed") is False:
        return []
    tables = operations.get("tables", {})
    bases, lockers = tables.get("bases", []), tables.get("lockers", [])
    if not bases and not lockers:
        return []
    _style()
    paths = []
    sites = _read_csv(Path(record["instance"]) / "sites.csv")
    fig, ax = plt.subplots(figsize=(11, 9))
    opened = {"L": {int(r["base"]) for r in bases if r.get("opened")},
              "H": {int(r["locker"]) for r in lockers if r.get("opened")}}
    counters = defaultdict(int)
    closed_sites = []
    for site in sites:
        kind = site["type"]
        if counters[kind] not in opened.get(kind, set()):
            closed_sites.append(site)
        counters[kind] += 1
    ax.scatter(*_xy(closed_sites), marker="x", s=24, color="#b8bec3", linewidths=.8,
               label="Closed candidate sites", zorder=1)
    opened_h = [r for r in lockers if r.get("opened")]
    throughput = [max(0., float(r.get("handling_events_per_hour",
                              float(r.get("pickup_orders_per_hour", 0)) + float(r.get("delivery_orders_per_hour", 0)))))
                  for r in opened_h]
    scale = max(throughput, default=1.) or 1.
    if opened_h:
        ax.scatter(*_xy(opened_h), s=[25 + 210 * v / scale for v in throughput],
                   color="#31688e", alpha=.65, edgecolors="white", linewidths=.4,
                   label="Open lockers (size: handoffs/h)", zorder=3)
    opened_l = [r for r in bases if r.get("opened")]
    if opened_l:
        colors = [float(r["availability"]) for r in opened_l]
        scatter = ax.scatter(*_xy(opened_l), c=colors, cmap="viridis", vmin=0, vmax=1,
                             marker="^", s=155, edgecolors="#263943", linewidths=.9,
                             zorder=5, label="Open bases")
        fig.colorbar(scatter, ax=ax, fraction=.045, pad=.035, label="Base availability ρ (not utilization)")
        for r in opened_l:
            ax.annotate(f"L{r['base']}", (r["x_km"], r["y_km"]), xytext=(5, 7),
                        textcoords="offset points", fontsize=10, weight="bold", zorder=6)
    for r in (opened_h if len(opened_h) <= 25 else sorted(opened_h, key=lambda r: float(r.get("handling_events_per_hour", 0)), reverse=True)[:8]):
        ax.annotate(f"H{r['locker']}", (r["x_km"], r["y_km"]), xytext=(4, 3),
                    textcoords="offset points", fontsize=8, color="#354a5a")
    map_handles, map_labels = ax.get_legend_handles_labels()
    if throughput:
        examples = sorted(set(round(scale * f, 1) for f in (.25, .5, 1)))
        handles = [ax.scatter([], [], s=25 + 210 * value / scale, color="#31688e", alpha=.65,
                              label=f"{value:g} handoffs/h") for value in examples]
        map_handles.extend(handles)
        map_labels.extend(f"{value:g} handoffs/h" for value in examples)
    fig.legend(map_handles, map_labels, loc="lower center", bbox_to_anchor=(.5, .075),
               ncol=3, frameon=False, fontsize=9, labelspacing=1.2)
    ax.set(xlabel="Local x (km)", ylabel="Local y (km)", title=_run_label(row))
    ax.set_aspect("equal", adjustable="box")
    _grid(ax, "both")
    summary = operations.get("summary", {})
    pieces = []
    for field, label, unit in (("served", "Drone service", "/h"), ("fleet", "Continuous fleet", ""),
                               ("busy_drones", "Mean busy drones", "")):
        value = _number(summary.get(field))
        if value is not None:
            pieces.append(f"{label}: {value:,.2f}{unit}")
    caption = " · ".join(pieces) + "\nLocal coordinates; no external basemap. Locker handoffs = pickups + deliveries."
    if len(opened_h) > 25:
        caption += " Only the eight busiest lockers are labelled."
    fig.text(.5, .015, caption, ha="center", va="bottom", fontsize=10)
    fig.tight_layout(rect=(0, .16, 1, 1))
    paths += _export(fig, output, "operations_map")
    if bases:
        x = np.arange(len(bases))
        fig, axes = plt.subplots(1, 3, figsize=(17, 5))
        axes[0].bar(x, [r["availability"] for r in bases], color="#31688e")
        axes[0].set(ylabel="Availability ρ", ylim=(0, 1.05), title="Immediate drone availability")
        for shift, key, label, color in ((-.2, "outflow", "Departures", "#31688e"),
                                        (.2, "inflow", "Returns", "#bd7b27")):
            axes[1].bar(x + shift, [r[key] for r in bases], width=.4, label=label, color=color)
        axes[1].set(ylabel="Drone tasks/h", title="Base task flow")
        axes[1].legend(frameon=False)
        idle = np.array([float(r["idle"]) for r in bases])
        busy = np.array([float(r["busy_drones"]) for r in bases])
        axes[2].bar(x, idle, color="#7d729a", label="Idle")
        axes[2].bar(x, busy, bottom=idle, color="#31688e", label="Busy, by departure base")
        axes[2].set(ylabel="Mean drones", title="Idle and attributed busy inventory")
        axes[2].legend(frameon=False, fontsize=9)
        for ax in axes:
            ax.set(xticks=x, xticklabels=[f"L{r['base']}" for r in bases])
            _grid(ax, "y")
        fig.suptitle(_run_label(row))
        fig.text(.5, .01, "Busy inventory = task departures/h × mean task duration (h); not an integer fleet allocation.", ha="center", fontsize=10)
        fig.tight_layout(rect=(0, .045, 1, .94))
        paths += _export(fig, output, "base_operations")
    return paths


def plot_progress(rows: list[dict], output: Path) -> list[Path]:
    """Plot recorded incumbent/bound events; never extend beyond saved events."""
    rows = [r for r in _eligible_detail(rows) if r.get("method") in METHODS
            and _number(r.get("elapsed_seconds")) is not None
            and float(r["elapsed_seconds"]) >= 0
            and any(_number(r.get(field)) is not None for field in ("upper_bound", "lower_bound"))]
    if not rows:
        return []
    _style()
    paths = []
    for k in sorted({_k(r) for r in rows}):
        selected = [r for r in rows if _k(r) == k]
        levels = sorted({_level(r) for r in selected})
        cols = min(2, len(levels))
        fig, axes = plt.subplots(math.ceil(len(levels) / cols), cols,
                                 figsize=(7 * cols, 3.8 * math.ceil(len(levels) / cols)), squeeze=False)
        for ax, level in zip(axes.flat, levels):
            runs = defaultdict(list)
            for row in selected:
                if _level(row) == level:
                    runs[(row["method"], str(row["id"]))].append(row)
            for (method, _), events in runs.items():
                events.sort(key=lambda r: float(r["elapsed_seconds"]))
                for field, style in (("upper_bound", "-"), ("lower_bound", "--")):
                    # NaN retains explicit gaps rather than inventing a bound.
                    values = [_number(r.get(field)) for r in events]
                    if any(v is not None for v in values):
                        ax.step([float(r["elapsed_seconds"]) for r in events],
                                [np.nan if v is None else v for v in values], where="post",
                                ls=style, color=METHODS[method][1], lw=1.5, marker=".", ms=3)
            ax.set(title=f"L{level:02d}", xlabel="Recorded elapsed time (s)", ylabel="Cost (CNY/h)")
            _grid(ax, "both")
        for ax in list(axes.flat)[len(levels):]:
            ax.set_visible(False)
        handles = [Line2D([], [], color=c, label=n) for n, c in METHODS.values()]
        handles += [Line2D([], [], color="#555555", ls="-", label="Feasible UB"),
                    Line2D([], [], color="#555555", ls="--", label="Certified LB")]
        fig.legend(handles=handles, ncol=5, loc="lower center", frameon=False)
        fig.suptitle(f"K={k} · saved bound progression for validated runs", fontsize=16)
        fig.tight_layout(rect=(0, .6 / fig.get_figheight(), 1, .96))
        paths += _export(fig, output, f"bound_progress_K{k}")
    return paths
