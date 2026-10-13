"""Persist verified sensitivity metrics and geographic solution maps."""

from __future__ import annotations

from collections import defaultdict
import math
from pathlib import Path

import numpy as np

from src.experiments.storage import write_json
from .style import plt, _style
from .basemap import _basemap
from .io import _csv, _sha
from .loader import load_case
from .map_data import _prepare
from .maps import _map
from .metrics import audit
from .projection import _mercator, local_km_to_plot_lonlat


COLORS = {"L": "#163e63", "H": "#087b86", "E": "#8b2e16"}
SHAPES = {"L": "s", "H": "o", "E": "D"}


def _view(case, cache):
    view = _prepare(case, cache)
    meta = view["source_meta"]
    bounds = meta.get("region_bounds_local_km")
    if bounds:
        x0, x1 = bounds["x_min"], bounds["x_max"]
        y0, y1 = bounds["y_min"], bounds["y_max"]
        square = np.array(
            [[x, y0] for x in np.linspace(x0, x1, 40)]
            + [[x1, y] for y in np.linspace(y0, y1, 40)]
            + [[x, y1] for x in np.linspace(x1, x0, 40)]
            + [[x0, y] for y in np.linspace(y1, y0, 40)]
        )
        ring = _mercator([local_km_to_plot_lonlat(*v, meta) for v in square])
        lo, hi = ring.min(axis=0) - 450, ring.max(axis=0) + 450
        view["bounds"] = (lo[0], hi[0], lo[1], hi[1])
        view["image"], view["extent"], view["tile_sources"] = _basemap(
            view["bounds"], cache
        )
        view["region_ring"] = ring
        view["width_km"], view["height_km"] = x1 - x0, y1 - y0
        view["area_km2"] = (x1 - x0) * (y1 - y0)
    return view


def _nodes(view, ax):
    L, H = view["case"]["row"]["L"], view["case"]["row"]["H"]
    counts = {r["locker"]: r["count"] for r in view["case"]["tables"]["lockers"]}
    grouped = defaultdict(list)
    for idx, point in enumerate(view["xy"]):
        grouped[tuple(np.round(point, 2))].append(idx)
    for idx, point in enumerate(view["xy"]):
        kind = "L" if idx < L else "H" if idx < L + H else "E"
        opened = view["opened"][idx]
        colocated = len(grouped[tuple(np.round(point, 2))]) > 1
        size = {"L": 215, "H": 87, "E": 115}[kind]
        if kind == "H" and opened and not colocated:
            size += 10 * counts[idx - L]
        if colocated and kind == "H":
            size = 66
        ax.scatter(
            *point, marker=SHAPES[kind], s=size,
            facecolors=COLORS[kind] if opened else "white",
            edgecolors="white" if opened else COLORS[kind],
            linewidths=1.5 if opened else 1.9, zorder=12 if kind == "H" else 11,
        )
    for point, indices in grouped.items():
        ax.annotate(
            " / ".join(view["labels"][idx] for idx in indices), point,
            xytext=(7, 9), textcoords="offset points", fontsize=9, weight="bold",
            zorder=25, color="#223f52",
            bbox=dict(facecolor="white", alpha=.87, edgecolor="none", pad=1.4),
        )


def _flows(view, ax):
    weights = defaultdict(float)
    for edge in view["edges"]:
        phase = "empty" if edge["kind"] in ("empty", "reposition") else (
            "loaded" if edge["phase"] == "loaded" else "connection"
        )
        weights[(phase, edge["u"], edge["v"])] += edge["flow_per_hour"]
    selected = [(key, rate) for key, rate in weights.items() if rate >= 1e-4]
    maximum = max((rate for _, rate in selected), default=1.0)
    config = {
        "connection": ("#647e91", .37, 3),
        "loaded": ("#057783", .72, 5),
        "empty": ("#b25118", .95, 7),
    }
    for phase, (color, alpha, zorder) in config.items():
        for (kind, u, v), rate in sorted(selected, key=lambda e: e[1]):
            if kind != phase:
                continue
            a, b = view["xy"][[u, v]]
            if np.linalg.norm(b - a) < 1e-6:
                continue
            width = .45 + 2.7 * math.sqrt(rate / maximum)
            if phase == "empty":
                width = max(width, 1.5)
            ax.plot(
                [a[0], b[0]], [a[1], b[1]], lw=width, color=color,
                alpha=alpha, ls="--" if phase == "empty" else "-", zorder=zorder,
            )
            ax.annotate(
                "", xy=a + .62 * (b - a), xytext=a + .46 * (b - a),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=width * .65,
                                alpha=alpha, mutation_scale=8 + width * 1.2),
                zorder=zorder + 1,
            )
    return dict(plotted_phase_edges=len(selected), threshold_per_hour=1e-4,
                omitted_numerical_flow=sum(r for r in weights.values() if r < 1e-4))


def _legend(fig, routes):
    from matplotlib.lines import Line2D

    handles = [
        Line2D([], [], marker=SHAPES[k], ls="", ms=9, mfc=COLORS[k], mec="white",
               label=label)
        for k, label in (("L", "Base L"), ("H", "Locker H"), ("E", "Swap E"))
    ]
    handles += [
        Line2D([], [], marker="o", ls="", ms=9, mfc="white", mec="#435565",
               label="Not opened"),
        Line2D([], [], color="#243746", ls="--", lw=1.5, label="Study boundary"),
    ]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(.045, .128),
               ncol=5, frameon=False, fontsize=10, handletextpad=.5, columnspacing=1.2)
    if routes:
        handles = [
            Line2D([], [], color="#647e91", lw=2, label="Pickup / return"),
            Line2D([], [], color="#057783", lw=3, label="Loaded flight"),
            Line2D([], [], color="#b25118", lw=2, ls="--", label="Repositioning"),
        ]
        fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(.045, .09),
                   ncol=3, frameon=False, fontsize=10, columnspacing=2)
        note = "Arrows show direction; line widths increase with flight flow."
    else:
        note = "Filled markers: opened sites. Locker marker size reflects installed units."
    fig.text(.052, .049, note, fontsize=10, color="#566c79")


def _draw_maps(case, output, cache):
    _style()
    view = _view(case, cache)
    row, result = case["row"], case["result"]
    provenance = case["metadata"].get("provenance", {})
    scenario = provenance.get("sensitivity", provenance.get("scenario_settings", {}))
    label = scenario.get("label", scenario.get("id", case["path"].name))
    if scenario.get("factor") and scenario.get("factor") != "baseline":
        label = f"{scenario['factor']} × {float(scenario.get('multiplier', 1)):g}"
    if scenario.get("factor") == "max_cabinets":
        label = f"Cabinet limit: {int(case['metadata']['max_cabinets'])} units per locker site"
    elif scenario.get("factor") == "walk_sensitivity":
        label = f"Walk-distance sensitivity × {float(scenario['multiplier']):g}"
    evidence = {}
    for routes in (False, True):
        fig = plt.figure(figsize=(10, 11.6))
        ax = fig.add_axes([.035, .15, .93, .735])
        _map(ax, view)
        for line in ax.lines:
            if line.get_zorder() == -2:
                line.set_alpha(.2)
        if "region_ring" in view:
            ring = view["region_ring"]
            ax.plot(ring[:, 0], ring[:, 1], color="#243746", lw=1.65,
                    ls=(0, (6, 4)), zorder=2)
        title = "Flight flows" if routes else "Facility locations"
        fig.text(.04, .97, f"Nanshan region 2 · H{row['H']} · {title}",
                 fontsize=21, weight="bold")
        fig.text(.04, .942, str(label), fontsize=11, color="#495d69")
        region = (f"{view['width_km']:.2f} × {view['height_km']:.2f} km "
                  f"({view['area_km2']:.2f} km²) | " if "area_km2" in view else "")
        fig.text(.04, .916, f"{region}{row['OD']} OD pairs | "
                 f"{row['demand']:,.0f} orders/hour", fontsize=12, color="#495d69")
        fig.text(.04, .890,
                 f"Opened / candidate: bases {row['open_bases']}/{row['L']} · "
                 f"lockers {row['open_lockers']}/{row['H']} · swaps "
                 f"{row['open_swaps']}/{row['E']} | {row['cabinets']:.0f} locker units",
                 fontsize=12, color="#263f50")
        flow_info = _flows(view, ax) if routes else None
        _nodes(view, ax)
        _legend(fig, routes)
        gap = "n/a" if row["gap_pct"] is None else f"{row['gap_pct']:.3f}%"
        seconds = float(row.get("solver_seconds") or 0)
        reason = str(result.get("termination_reason", result.get("status", "unknown")))
        fig.text(.04, .019, f"Arc-based | {seconds / 60:.2f} min | gap {gap} | "
                 f"{reason} | cost {row['cost']:,.2f} CNY/hour",
                 fontsize=9, color="#536570")
        fig.text(.04, .001, "Basemap © OpenStreetMap contributors; saved Shenzhen "
                 "coordinate conversion.", fontsize=9, color="#6a7b85")
        name = "network" if routes else "facilities"
        for extension in ("png", "pdf"):
            fig.savefig(output / f"{name}.{extension}", dpi=180,
                        bbox_inches="tight", pad_inches=.16, facecolor="white")
        plt.close(fig)
        evidence[name] = dict(png=str(output / f"{name}.png"),
                              pdf=str(output / f"{name}.pdf"), flows=flow_info)
    evidence.update(area_km2=view.get("area_km2"),
                    width_km=view.get("width_km"), height_km=view.get("height_km"),
                    tile_sources=view["tile_sources"], od_lines=False)
    _csv(output / "site_coordinates.csv", [
        dict(site=label, opened=bool(opened), longitude=float(point[0]),
             latitude=float(point[1]))
        for label, opened, point in zip(view["labels"], view["opened"], view["lonlat"])
    ])
    _csv(output / "physical_segment_flows.csv", view["edges"])
    return evidence


def export_case(run_path, *, map_cache, batch_path=None, output=None):
    """Export one finished run without altering any existing run when output is set.

    The same-scenario all-courier baseline supplies the savings denominator.
    No solver is started; saved paths are independently reconstructed by audit.
    """
    run_path = Path(run_path).resolve()
    output = Path(output).resolve() if output else run_path / "sensitivity"
    output.mkdir(parents=True, exist_ok=True)
    case = load_case(run_path)
    if case.get("pending") or case["row"]["validation"] != "通过":
        raise ValueError(f"Saved case has not passed operational checks: {run_path}")
    detail = audit(case)
    row, s = case["row"], case["summary"]
    baseline = float(row["baseline_cost"])
    saving = baseline - row["cost"]
    summary = dict(
        case=run_path.name, method="arc", OD=row["OD"],
        candidate_bases=row["L"], candidate_lockers=row["H"], candidate_swaps=row["E"],
        demand_per_hour=row["demand"], total_cost_per_hour=row["cost"],
        all_courier_cost_per_hour=baseline, saving_per_hour=saving,
        saving_pct=100 * saving / baseline if baseline else 0.0,
        drone_served_per_hour=row["served"], drone_served_share_pct=row["served_pct"],
        pickup_per_hour=row["pickup"], home_per_hour=row["home"],
        direct_courier_per_hour=row["direct"], fleet=row["fleet"],
        opened_bases=row["open_bases"], opened_lockers=row["open_lockers"],
        opened_swaps=row["open_swaps"], locker_units=row["cabinets"],
        opened_airlines=s.get("open_links"), used_airlines=s.get("used_links"),
        reposition_flow_per_hour=row["empty_flow"],
        cross_base_service_pct=row["cross_base_pct"],
        swap_service_visits_per_hour=detail["swapping"]["service_visits_per_hour"],
        swap_reposition_visits_per_hour=detail["swapping"]["reposition_visits_per_hour"],
        service_using_swap_pct=100 * detail["swapping"]["service_flow_with_swap_fraction"],
        locker_capacity_utilization_pct=100 * detail["locker_capacity"]["utilization"],
        mean_delivery_minutes=detail["times"].get("mean_delivery"),
        direct_same_served_OD_minutes=detail["times"].get("direct_same_served_OD"),
        gap_pct=row["gap_pct"], bound=row["bound"], solver_seconds=row["solver_seconds"],
        termination_reason=case["result"].get("termination_reason"),
        validation_passed=True,
    )
    for component, value in case["fleet"].items():
        summary[f"fleet_{component}"] = value
    for component, value in case["costs"].items():
        summary[f"cost_{component}_per_hour"] = value
    sensitivity = case["metadata"].get("provenance", {}).get("sensitivity", {})
    summary.update(
        round=sensitivity.get("round", 1),
        max_cabinets=case["metadata"].get("max_cabinets"),
        walk_distance_coefficient=(case["metadata"].get("provenance", {})
                                   .get("resolved_config", {}).get("choice", {})
                                   .get("walk_distance_coefficient")),
        scenario=sensitivity.get("id", "baseline"),
        factor=sensitivity.get("factor", "baseline"),
        multiplier=float(sensitivity.get("multiplier", 1.0)),
        swap_visits_per_hour=(summary["swap_service_visits_per_hour"]
                             + summary["swap_reposition_visits_per_hour"]),
    )
    for name, rows in case["tables"].items():
        if isinstance(rows, list) and rows:
            _csv(output / f"{name}.csv", rows)
    _csv(output / "swapping_stations.csv", detail["swapping"]["stations"])
    _csv(output / "repositioning_pairs.csv", detail["repositioning"]["pairs"],
         fields=["origin", "destination", "flow_per_hour"])
    _csv(output / "summary.csv", [summary])
    report = dict(
        summary=summary, details=detail,
        scenario=case["metadata"].get("provenance", {}).get("sensitivity", {}),
        run_path=str(run_path), batch_path=str(batch_path) if batch_path else None,
        output=str(output),
        sources={name: _sha(run_path / name) for name in
                 ("input_metadata.json", "input_arrays.npz", "result.json",
                  "solution.npz", "operations.json")},
    )
    # Persist numerical results before rendering so a map failure cannot lose them.
    write_json(output / "sensitivity_metrics.json", report)
    report["maps"] = _draw_maps(case, output, Path(map_cache))
    write_json(output / "sensitivity_metrics.json", report)
    return report
