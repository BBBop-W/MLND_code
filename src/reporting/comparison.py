"""Reproducible comparisons of saved consecutive runs, without optimization.

Every plotted incumbent is independently reconstructed. Intermediate MIP
incumbents are explicitly solver-reported; a time axis is never invented from
iteration numbers. Bounds are optimization bounds, not confidence intervals.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from src.data.instances import pack_planning_data, to_planning_data
from src.experiments.storage import json_payload, load_seed, sha256, write_json
from src.model.types import MODEL_VERSION
from src.operations.validation import build_operations
from .report import _coordinates, _projection_metadata
from .style import _grid, _save, _style, plt


METHODS = {
    "full_path": "Full MISOCP",
    "benders_cg": "Benders + CG",
    "benders_cg_plain": "Benders + CG (plain)",
}
COLORS = {
    "full_path": "#31688e",
    "benders_cg": "#bd7b27",
    "benders_cg_plain": "#7d729a",
}
COSTS = {
    "direct_courier": ("Courier", "#a8b3bc"),
    "service": ("Service flights / riders", "#31688e"),
    "fleet": ("Fleet", "#bd7b27"),
    "bases": ("Bases", "#4f7570"),
    "lockers": ("Lockers", "#7d729a"),
    "links": ("Links", "#b6ac8c"),
    "repositioning": ("Repositioning", "#d5b27c"),
}


def _finite(value):
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) and abs(number) < 1e90 else None


def _label(row):
    return (
        f"K={row['max_orders']} | {row['flow_policy']} | "
        f"{METHODS.get(row['method'], row['method'])}"
    )


def _csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _discover(run_dirs):
    if isinstance(run_dirs, (str, Path)):
        parent = Path(run_dirs)
        run_dirs = (
            [parent]
            if (parent / "result.json").is_file()
            else [
                p.parent
                for p in parent.rglob("result.json")
                if "checkpoints" not in p.relative_to(parent).parts
            ]
        )
    paths = sorted({Path(path).resolve() for path in run_dirs})
    if not paths:
        raise ValueError("No saved result.json runs found")
    return paths


def _scenario_fingerprint(inputs):
    values = pack_planning_data(to_planning_data(inputs))
    # K and flow_policy define the compared operating policies. Remaining
    # physical input must agree before interpreting these policy comparisons.
    values.pop("max_orders")
    values.pop("flow_policy")
    values.pop("metadata")
    for key in (
        "bases",
        "hubs",
        "origins",
        "destinations",
        "od_origin",
        "od_destination",
    ):
        if key in inputs:
            values[key] = inputs[key]
    return hashlib.sha256(
        json.dumps(json_payload(values), sort_keys=True).encode()
    ).hexdigest()


def _history(path, result):
    rows = []
    source = path / "progress.jsonl"
    if source.exists():
        for number, text in enumerate(source.read_text().splitlines(), 1):
            if not text.strip():
                continue
            event = json.loads(text)
            elapsed = _finite(event.get("elapsed"))
            if elapsed is None or elapsed < 0:
                raise ValueError(f"Invalid elapsed time in {source}:{number}")
            raw_solver = "solver_incumbent" in event or "solver_bound" in event
            incumbent = _finite(
                event.get("solver_incumbent" if raw_solver else "objective")
            )
            bound = _finite(event.get("solver_bound" if raw_solver else "bound"))
            if incumbent is not None or bound is not None:
                rows.append(
                    dict(
                        elapsed=elapsed,
                        incumbent=incumbent,
                        bound=bound,
                        kind="solver_reported" if raw_solver else "validated_iteration",
                        iteration=event.get("iteration"),
                        stage=event.get("stage"),
                        pricing_complete=event.get("pricing_complete"),
                    )
                )
    # The endpoint is measured, and is valid even without intermediate history.
    elapsed = _finite(result.get("wall"))
    if elapsed is not None:
        rows.append(
            dict(
                elapsed=elapsed,
                incumbent=_finite(result.get("objective")),
                bound=_finite(result.get("bound")),
                kind="validated_final",
                iteration=None,
                stage="final",
                pricing_complete=result.get("pricing_complete"),
            )
        )
    return sorted(rows, key=lambda row: row["elapsed"])


def _read_run(path):
    inputs, result = load_seed(path)
    if result.get("model_version") != MODEL_VERSION:
        raise ValueError(f"Run {path} is not model {MODEL_VERSION}")
    payload = build_operations(inputs, result)
    summary = payload["summary"]
    if not summary["checks"]["passed"]:
        raise ValueError(
            f"Independent validation failed for {path}: {summary['checks']}"
        )
    data = to_planning_data(inputs)
    manifest_path = path / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    if manifest and manifest.get("status") != "completed":
        raise ValueError(f"Run manifest is not completed: {path}")
    options = manifest.get("options", {})
    objective = summary["objective"]
    bound = _finite(result.get("bound"))
    row = dict(
        case=path.name,
        method=result.get("method", "unknown"),
        max_orders=data.max_orders,
        flow_policy=data.flow_policy,
        model_version=MODEL_VERSION,
        n_l=data.n_l,
        n_h=data.n_h,
        n_od=data.n_od,
        path_scope=result.get("path_scope"),
        scenario_sha256=_scenario_fingerprint(inputs),
        objective=objective,
        bound=bound,
        gap=(max(0, objective - bound) / max(1, abs(objective)))
        if bound is not None
        else None,
        wall_seconds=result.get("wall"),
        budget_seconds=result.get("budget", options.get("time_limit")),
        target_gap=options.get("mip_gap"),
        threads=options.get("threads"),
        seed=options.get("seed"),
        termination_reason=result.get("termination_reason"),
        incumbent_kind=result.get("incumbent_kind"),
        independently_validated=True,
        pricing_complete=result.get("pricing_complete"),
        catalogue_complete=result.get("catalogue_complete"),
        construction_complete=result.get("construction_complete"),
        pricing_implementation=result.get("pricing_implementation"),
        demand_per_hour=summary["demand"],
        served_per_hour=summary["served"],
        served_share=summary["served"] / summary["demand"] if summary["demand"] else 0,
        direct_per_hour=summary["direct"],
        task_flights_per_hour=summary["task_flights"],
        single_tasks_per_hour=summary["single_tasks"],
        double_tasks_per_hour=summary["double_tasks"],
        empty_tasks_per_hour=summary["empty_tasks"],
        fleet=summary["fleet"],
        slots=summary["slot_total"],
        open_bases=summary["opened"]["bases"],
        open_lockers=summary["opened"]["lockers"],
        iterations=len(result.get("iterations", [])),
        certified_cuts=result.get("certified_cuts"),
        catalogue_service=result.get("catalogue", {}).get("service"),
        catalogue_segments=result.get("catalogue", {}).get("segments"),
        catalogue_empty=result.get("catalogue", {}).get("empty"),
        run_directory=str(path),
        enumeration_seconds=result.get("enumeration_seconds"),
        solve_seconds=result.get("solve_seconds"),
    )
    row.update(
        {
            f"cost_{item['component']}": item["cny_per_hour"]
            for item in payload["tables"]["costs"]
        }
    )
    row.update(
        {
            f"timing_{key}": value
            for key, value in result.get("phase_timings", {}).items()
        }
    )
    weighted = [
        (item["drone_per_hour"], item["mean_receipt_minutes"])
        for item in payload["tables"]["orders"]
        if item["mean_receipt_minutes"] is not None
    ]
    row["mean_receipt_minutes"] = (
        sum(flow * mean for flow, mean in weighted) / summary["served"]
        if summary["served"] > 0
        else None
    )
    return dict(
        path=path,
        inputs=inputs,
        result=result,
        payload=payload,
        row=row,
        history=_history(path, result),
    )


def _performance(runs, output):
    rows = [run["row"] for run in runs]
    positions = np.arange(len(rows))
    labels = [_label(row) for row in rows]
    fig, axes = plt.subplots(
        1, 3, figsize=(16, max(4.5, 0.58 * len(rows) + 1.7)), sharey=True
    )
    for index, row in enumerate(rows):
        color = COLORS.get(row["method"], "#31688e")
        if row["bound"] is not None:
            axes[0].plot(
                [row["bound"], row["objective"]],
                [index, index],
                color=color,
                linewidth=3,
            )
            axes[0].plot(row["bound"], index, "|", color=color, markersize=13)
        axes[0].plot(row["objective"], index, "o", color=color)
        if row["gap"] is not None:
            axes[1].barh(index, row["gap"] * 100, color=color)
            axes[1].annotate(
                f"{row['gap'] * 100:.3g}%",
                (row["gap"] * 100, index),
                xytext=(5, 0),
                textcoords="offset points",
                va="center",
                fontsize=10,
            )
        if row["wall_seconds"] is not None:
            axes[2].barh(index, row["wall_seconds"], color=color)
            axes[2].annotate(
                f"{row['wall_seconds']:.1f}s",
                (row["wall_seconds"], index),
                xytext=(5, 0),
                textcoords="offset points",
                va="center",
                fontsize=10,
            )
    axes[0].set(
        yticks=positions,
        yticklabels=labels,
        xlabel="CNY / hour (focused scale)",
        title="Certified bound — feasible cost",
    )
    axes[0].invert_yaxis()
    axes[1].set(
        xlabel="Relative optimality gap (%)",
        title="Remaining gap",
        xlim=(
            0,
            max([row["gap"] * 100 for row in rows if row["gap"] is not None] + [0.01])
            * 1.25,
        ),
    )
    axes[2].set(
        xlabel="Elapsed seconds",
        title="Measured wall time",
        xlim=(0, max([row["wall_seconds"] or 0 for row in rows] + [1]) * 1.25),
    )
    for axis in axes:
        _grid(axis)
    fig.suptitle("Method comparisons require the same K and flow policy", fontsize=14)
    fig.tight_layout()
    _save(fig, output, "performance")


def _convergence(runs, output):
    policies = sorted(
        {(run["row"]["max_orders"], run["row"]["flow_policy"]) for run in runs}
    )
    fig, axes = plt.subplots(
        1,
        len(policies),
        figsize=(7 * len(policies), 5),
        squeeze=False,
        sharex=True,
        sharey=True,
    )
    any_history = False
    for ax, policy in zip(axes[0], policies):
        for run in runs:
            row = run["row"]
            if (row["max_orders"], row["flow_policy"]) != policy:
                continue
            history = run["history"]
            any_history |= any(event["kind"] != "validated_final" for event in history)
            color = COLORS.get(row["method"], "#31688e")
            method = METHODS.get(row["method"], row["method"])
            for key, style, suffix in (
                ("incumbent", "-", "incumbent"),
                ("bound", "--", "bound"),
            ):
                points = [
                    (event["elapsed"], event[key])
                    for event in history
                    if event[key] is not None
                ]
                if points:
                    x, y = zip(*points)
                    ax.step(
                        x,
                        y,
                        where="post",
                        linestyle=style,
                        color=color,
                        label=f"{method}: {suffix}",
                        linewidth=1.5,
                        marker="." if len(points) == 1 else None,
                    )
            ax.scatter(
                [row["wall_seconds"]],
                [row["objective"]],
                marker="o",
                facecolor="white",
                edgecolor=color,
                s=40,
                zorder=4,
            )
        ax.set(
            xlabel="Measured elapsed seconds",
            ylabel="CNY / hour",
            title=f"K={policy[0]} | {policy[1]}",
        )
        ax.legend(fontsize=9, loc="best")
        ax.set_xlim(left=0)
        _grid(ax, "both")
    fig.suptitle(
        "Recorded progress; open circles are independently validated final incumbents",
        fontsize=12,
    )
    fig.text(
        0.5,
        0.005,
        "Intermediate MIP incumbents are solver-reported. Missing histories are shown only as endpoints.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.035, 1, 0.96))
    _save(fig, output, "convergence")
    return any_history


def _operating(runs, output):
    rows = [run["row"] for run in runs]
    positions = np.arange(len(rows))
    labels = [_label(row) for row in rows]
    fig, ax = plt.subplots(figsize=(12, max(4.5, 0.58 * len(rows) + 1.7)))
    left = np.zeros(len(rows))
    for key, (label, color) in COSTS.items():
        values = np.array([row.get(f"cost_{key}", 0) for row in rows])
        if np.any(values > 1e-7):
            ax.barh(positions, values, left=left, color=color, label=label)
        left += values
    ax.set(
        yticks=positions,
        yticklabels=labels,
        xlabel="CNY / hour",
        title="Cost composition of independently validated incumbents",
        xlim=(0, max(left) * 1.05),
    )
    ax.invert_yaxis()
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False)
    _grid(ax)
    fig.tight_layout()
    _save(fig, output, "operating_costs")
    fig, axes = plt.subplots(
        1, 3, figsize=(16, max(4.5, 0.58 * len(rows) + 1.7)), sharey=True
    )
    axes[0].barh(
        positions,
        [row["single_tasks_per_hour"] for row in rows],
        color="#31688e",
        label="Single-task orders",
    )
    axes[0].barh(
        positions,
        [2 * row["double_tasks_per_hour"] for row in rows],
        left=[row["single_tasks_per_hour"] for row in rows],
        color="#bd7b27",
        label="Double-task orders (2 per task)",
    )
    axes[1].barh(
        positions, [100 * row["served_share"] for row in rows], color="#31688e"
    )
    axes[2].barh(positions, [row["fleet"] for row in rows], color="#4f7570")
    axes[0].set(
        yticks=positions,
        yticklabels=labels,
        xlabel="Drone-served orders / hour",
        title="Served volume",
    )
    axes[0].invert_yaxis()
    axes[0].legend(
        loc="upper left", bbox_to_anchor=(0, -0.18), frameon=False, fontsize=9
    )
    axes[1].set(
        xlabel="Share of all orders (%)", xlim=(0, 100), title="Drone service share"
    )
    axes[2].set(xlabel="Drones (continuous)", title="Fleet")
    for ax in axes:
        _grid(ax)
    fig.tight_layout()
    _save(fig, output, "service_fleet")


def _nested_metadata(inputs, key):
    if not isinstance(inputs, dict):
        return None
    if key in inputs:
        return inputs
    for value in inputs.values():
        found = _nested_metadata(value, key)
        if found is not None:
            return found
    return None


def _map_context(inputs, *, cached_basemap=False):
    xy = _coordinates(inputs)
    if xy is None:
        return None
    projection = _projection_metadata(inputs)
    crop = _nested_metadata(inputs, "crop_half_size_km")
    corners = None
    if (
        crop is not None
        and crop.get("crop_half_size_km") is not None
        and crop.get("crop_center_local_km") is not None
    ):
        center, radius = (
            np.asarray(crop["crop_center_local_km"], float),
            float(crop["crop_half_size_km"]),
        )
        corners = center + radius * np.array(
            [[-1, -1], [1, -1], [1, 1], [-1, 1], [-1, -1]]
        )

    def transform(values):
        return np.asarray(values, float)

    coordinate_label = "Local projected distance (km)"
    sources = []
    image, extent = None, None
    if projection is not None:
        from .projection import local_km_to_plot_lonlat, _mercator

        source_crs = projection.get("geographic_crs")
        if cached_basemap and source_crs not in ("WGS84", "GCJ02"):
            raise ValueError(
                "Cached map comparison requires a documented WGS84 or GCJ02 source CRS"
            )

        def lonlat(values):
            return np.asarray(
                [
                    local_km_to_plot_lonlat(
                        x,
                        y,
                        projection,
                        assume_gcj02=source_crs == "GCJ02" and cached_basemap,
                    )
                    for x, y in values
                ]
            )

        transform = (
            (lambda values: _mercator(lonlat(values))) if cached_basemap else lonlat
        )
        coordinate_label = (
            "EPSG:3857 (m)"
            if cached_basemap
            else f"Source longitude / latitude ({source_crs or 'CRS unspecified'})"
        )
    bounds_xy = transform(corners if corners is not None else xy)
    span = np.maximum(np.ptp(bounds_xy, axis=0), 1e-6)
    low, high = bounds_xy.min(axis=0) - 0.08 * span, bounds_xy.max(axis=0) + 0.08 * span
    if cached_basemap:
        from .basemap import _basemap, CACHE

        image, extent, sources = _basemap((low[0], high[0], low[1], high[1]), CACHE)
    return dict(
        xy=transform(xy),
        transform=transform,
        corners=None if corners is None else transform(corners),
        low=low,
        high=high,
        label=coordinate_label,
        projection=projection,
        image=image,
        extent=extent,
        sources=sources,
        cached_basemap=cached_basemap,
    )


def _map_background(ax, context):
    if context["image"] is not None:
        ax.imshow(context["image"], extent=context["extent"], origin="upper", zorder=0)
        ax.text(
            0.01,
            0.01,
            "© OpenStreetMap contributors",
            transform=ax.transAxes,
            fontsize=7,
        )
    if context["corners"] is not None:
        p = context["corners"]
        ax.plot(
            p[:, 0],
            p[:, 1],
            "--",
            color="#555d65",
            linewidth=1.5,
            label="Study-area crop",
            zorder=2,
        )
    ax.set(
        xlim=(context["low"][0], context["high"][0]),
        ylim=(context["low"][1], context["high"][1]),
    )
    if context["projection"] is not None and not context["cached_basemap"]:
        ax.set_aspect(
            1 / np.cos(np.deg2rad(context["projection"]["projection_center_lat"]))
        )
        ax.set(xlabel="Source longitude (°)", ylabel="Source latitude (°)")
    else:
        ax.set_aspect("equal")
        unit = "m" if context["cached_basemap"] else "km"
        ax.set(xlabel=f"Easting ({unit})", ylabel=f"Northing ({unit})")
    ax.ticklabel_format(useOffset=False, style="plain")
    if context["projection"] is not None and not context["cached_basemap"]:
        from matplotlib.ticker import FormatStrFormatter, MaxNLocator

        for axis in (ax.xaxis, ax.yaxis):
            axis.set_major_locator(MaxNLocator(nbins=4))
            axis.set_major_formatter(FormatStrFormatter("%.3f"))
    _grid(ax, "both")


def _sites(ax, xy, n_l, opened=None, labels=False):
    for indices, marker, name, color in (
        (np.arange(n_l), "s", "Base", "#243a53"),
        (np.arange(n_l, len(xy)), "o", "Locker", "#bd7b27"),
    ):
        ax.scatter(
            xy[indices, 0],
            xy[indices, 1],
            s=65,
            marker=marker,
            facecolors="white",
            edgecolors=color,
            linewidths=1.5,
            label=f"{name} candidate",
            zorder=5,
        )
        if opened is not None:
            active = indices[opened[indices]]
            ax.scatter(
                xy[active, 0],
                xy[active, 1],
                s=65,
                marker=marker,
                color=color,
                label=f"{name} open",
                zorder=6,
            )
    if labels:
        for i, point in enumerate(xy):
            ax.annotate(
                f"L{i}" if i < n_l else f"H{i - n_l}",
                point,
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=9,
                zorder=7,
            )


def _maps(runs, output, *, cached_basemap=False):
    context = _map_context(runs[0]["inputs"], cached_basemap=cached_basemap)
    if context is None:
        return dict(generated=False, reason="No saved candidate coordinates")
    inputs = runs[0]["inputs"]
    fig, ax = plt.subplots(figsize=(10, 8))
    _map_background(ax, context)
    if "origins" in inputs and "destinations" in inputs:
        origins = context["transform"](inputs["origins"])
        destinations = context["transform"](inputs["destinations"])
        origin_index = inputs.get("od_origin", np.arange(len(origins)))
        destination_index = inputs.get("od_destination", np.arange(len(destinations)))
        for oi, di in zip(origin_index, destination_index):
            points = np.array([origins[int(oi)], destinations[int(di)]])
            ax.plot(
                points[:, 0],
                points[:, 1],
                color="#99a5b0",
                alpha=0.28,
                linewidth=0.7,
                zorder=1,
            )
        ax.scatter(
            origins[:, 0],
            origins[:, 1],
            s=15,
            marker="+",
            color="#31688e",
            label="OD origins",
            zorder=3,
        )
        ax.scatter(
            destinations[:, 0],
            destinations[:, 1],
            s=15,
            marker="x",
            color="#6b8884",
            label="OD destinations",
            zorder=3,
        )
    _sites(ax, context["xy"], runs[0]["row"]["n_l"], labels=True)
    row = runs[0]["row"]
    ax.set_title(
        f"Study area: {row['n_l']} bases, {row['n_h']} lockers, {row['n_od']} OD pairs\n{context['label']}",
        fontsize=14,
    )
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False)
    fig.tight_layout()
    _save(fig, output, "study_area")
    cols = min(3, len(runs))
    fig, axes = plt.subplots(
        int(np.ceil(len(runs) / cols)),
        cols,
        figsize=(6 * cols, 5.5 * np.ceil(len(runs) / cols)),
        squeeze=False,
    )
    for ax, run in zip(axes.flat, runs):
        _map_background(ax, context)
        _sites(
            ax,
            context["xy"],
            row["n_l"],
            np.asarray(run["result"]["_solution"]["facility"]) > 0.5,
            labels=True,
        )
        ax.set_title(
            f"{_label(run['row'])}\n{run['row']['termination_reason']}; gap={100 * run['row']['gap']:.3g}%"
            if run["row"]["gap"] is not None
            else _label(run["row"]),
            fontsize=12,
        )
    for ax in list(axes.flat)[len(runs) :]:
        ax.set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False)
    fig.suptitle(
        "Open and candidate facilities of returned feasible solutions", fontsize=15
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    _save(fig, output, "selected_facilities")
    return dict(
        generated=True,
        coordinates=context["label"],
        cached_basemap=cached_basemap,
        sources=context["sources"],
    )


def _fmt(value, digits=4):
    return "—" if value is None else f"{value:.{digits}f}"


def _markdown(rows, output, history_available, map_info):
    lines = [
        "# Consecutive 算法与求解器实验比较",
        "",
        f"模型：`{MODEL_VERSION}`。所有最终解均从保存的路径与输入独立重算并通过可行性核验。",
        "",
        "| 实验 | K | 流量约束 | 方法 | 成本/小时 | 下界 | gap | 时间/s | 停止原因 | 定价完整 | 无人机订单/h |",
        "|---|---:|---|---|---:|---:|---:|---:|---|---|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['case']} | {row['max_orders']} | {row['flow_policy']} | {row['method']} | {_fmt(row['objective'])} | {_fmt(row['bound'])} | {_fmt(None if row['gap'] is None else 100 * row['gap'], 3)}% | {_fmt(row['wall_seconds'], 2)} | {row['termination_reason']} | {row['pricing_complete']} | {_fmt(row['served_per_hour'], 2)} |"
        )
    lines += [
        "",
        "算法比较只在相同 K、流量约束、路径范围和物理输入内进行；不同运营策略的界不能直接合并为同一模型的最优性证明。",
        "prefix 要求每个首段遵守相同续单比例；aggregate 仅要求同基地对、同首单 OD 的各类完整任务汇总量符合重构需求，允许任务使用不同首段。",
        "gap = max(0, incumbent − bound) / max(1, |incumbent|)。图中的区间是最优化上下界，不是统计置信区间。",
        "限时结果保留终止原因、下界和定价完整标志；较低的当前可行成本不自动证明算法总体更好。每项只运行一次时不构成运行时间统计结论。",
        "成本、服务量和机队图比较的是返回的可行方案，未收敛时可能同时反映算法进度差异。",
        "K=2 的正窗口和规定任务组合改变运营约束；K=1 设置窗口为零。因此不预设 K=2 必然降低成本。固定其他输入时 aggregate 松弛 prefix，理论最优成本不会增加；限时可行费用仍可能受优化进度影响。",
        "路径范围及目录完整性仅针对保存配置；有限目录完整扫描不代表任意路径的隐式全局定价。",
        "",
    ]
    if not history_available:
        lines.append(
            "没有带时间戳的中间迭代记录；收敛图仅显示实测终点，不插值或推测过程。"
        )
    lines += [
        "中间完整模型曲线使用求解器报告的 incumbent/bound；最终空心点已经独立核验。所有时间来自运行记录。",
        "",
        "| 实验 | 需求/h | 服务占比 | 单单任务/h | 双单任务/h | 机队 | 开基地/柜 | 平均送达/min |",
        "|---|---:|---:|---:|---:|---:|---|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['case']} | {_fmt(row['demand_per_hour'], 2)} | {100 * row['served_share']:.2f}% | {_fmt(row['single_tasks_per_hour'], 2)} | {_fmt(row['double_tasks_per_hour'], 2)} | {_fmt(row['fleet'], 2)} | {row['open_bases']}/{row['open_lockers']} | {_fmt(row['mean_receipt_minutes'], 2)} |"
        )
    lines += [
        "",
        "一条双单任务服务两个订单；无人机服务量 = 单单任务数 + 2 × 双单任务数。平均送达时间仅以实际无人机服务订单加权。",
        "",
        "![求解表现](performance.png)",
        "",
        "![收敛过程](convergence.png)",
        "",
        "![成本构成](operating_costs.png)",
        "",
        "![服务与机队](service_fleet.png)",
    ]
    if map_info["generated"]:
        lines += [
            "",
            f"地图坐标：{map_info['coordinates']}。底图仅使用已有缓存。"
            if map_info["cached_basemap"]
            else f"地图坐标：{map_info['coordinates']}。未使用外部底图或假定未知坐标基准。",
            "",
            "![研究切片与候选设施](study_area.png)",
            "",
            "![所选设施比较](selected_facilities.png)",
        ]
    lines += [
        "",
        "完整数值见 `comparison.csv`；逐条实测历史见 `history.csv`；输入与结果文件指纹见 `source_manifest.json`。",
        "",
    ]
    (output / "COMPARISON.md").write_text("\n".join(lines), encoding="utf-8")


def produce_comparison(run_dirs, output, *, cached_basemap=False):
    """Compare explicit run directories, or discover saved runs under a parent.

    K and flow_policy can differ, but market, costs, compatibility, paths and
    physical parameters must match. No optimizer is invoked or source altered.
    """
    runs = [_read_run(path) for path in _discover(run_dirs)]
    runs.sort(
        key=lambda run: (
            run["row"]["max_orders"],
            run["row"]["flow_policy"],
            run["row"]["method"],
            run["row"]["case"],
        )
    )
    if (
        len({run["row"]["scenario_sha256"] for run in runs}) != 1
        or len({run["row"]["path_scope"] for run in runs}) != 1
    ):
        raise ValueError(
            "Comparison requires identical physical inputs and path scope, apart from max_orders and flow_policy"
        )
    output = Path(output).resolve()
    if any(output == run["path"] for run in runs):
        raise ValueError(
            "Comparison output must be separate from source run directories"
        )
    output.mkdir(parents=True, exist_ok=True)
    rows = [run["row"] for run in runs]
    _csv(output / "comparison.csv", rows)
    history = [
        dict(
            case=run["row"]["case"],
            method=run["row"]["method"],
            max_orders=run["row"]["max_orders"],
            flow_policy=run["row"]["flow_policy"],
            **event,
        )
        for run in runs
        for event in run["history"]
    ]
    _csv(output / "history.csv", history)
    _style()
    _performance(runs, output)
    history_available = _convergence(runs, output)
    _operating(runs, output)
    map_info = _maps(runs, output, cached_basemap=cached_basemap)
    _markdown(rows, output, history_available, map_info)
    sources = {
        str(run["path"]): {
            name: sha256(run["path"] / name)
            for name in (
                "input_arrays.npz",
                "input_metadata.json",
                "solution.npz",
                "path_flows.json.gz",
                "result.json",
                "manifest.json",
                "progress.jsonl",
            )
            if (run["path"] / name).exists()
        }
        for run in runs
    }
    manifest = dict(
        model_version=MODEL_VERSION,
        flow_policies=sorted({row["flow_policy"] for row in rows}),
        independently_validated=True,
        optimization_rerun=False,
        scenario_sha256=rows[0]["scenario_sha256"],
        sources=sources,
        map=map_info,
        measured_intermediate_history=history_available,
        report_source_hashes={
            name: sha256(Path(__file__).with_name(name))
            for name in ("comparison.py", "style.py", "projection.py", "report.py")
        },
        figures=sorted(path.name for path in output.glob("*.png")),
    )
    write_json(output / "source_manifest.json", manifest)
    write_json(output / "comparison.json", rows)
    return dict(
        output=str(output),
        cases=len(rows),
        figures=manifest["figures"],
        independent_validation_passed=True,
        rows=rows,
    )
