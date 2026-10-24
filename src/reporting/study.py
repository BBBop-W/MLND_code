"""Detailed, independently checked K=1/K=2 studies across real datasets.

This module does not optimize. It reads saved cases, verifies each incumbent,
checks the within-dataset physical controls, and exports every operating table
alongside scientific figures and a Chinese interpretation with defined units.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from src.experiments.storage import load_seed, sha256, source_hashes, write_json
from src.operations.export import export_operations
from .comparison import COSTS, _csv, _scenario_fingerprint
from .report import _coordinates, network_figure
from .style import _grid, _save, _style, plt


DATASET_NAMES = {"nanshan": "南山", "tsl": "美团 TSL"}
DATASET_LABELS = {"nanshan": "Nanshan", "tsl": "Meituan TSL"}
POLICY_COLORS = {1: "#356c91", 2: "#bd7b27"}
BUSY_LABELS = {
    "service_flight": "Service flight",
    "first_pickup_wait": "First pickup wait",
    "second_pickup_wait": "Second pickup wait",
    "empty_flight": "Empty flight",
}
BUSY_COLORS = ["#356c91", "#bd7b27", "#73978c", "#a8b3bc"]
BUSY_CN = {
    "service_flight": "服务任务全部飞行",
    "first_pickup_wait": "首单取餐等待",
    "second_pickup_wait": "第二单取餐等待",
    "empty_flight": "基地间空载调度飞行",
}
STAGE_CN = {
    "first_dispatch": "基地→首单取餐柜",
    "first_loaded": "首单取餐柜→首单投递柜",
    "continuation_transfer": "首单投递柜→第二单取餐柜",
    "second_loaded": "第二单取餐柜→第二单投递柜",
    "return": "最后投递柜→返回基地",
    "empty": "基地→基地空载调度",
}
DETAIL_COST_LABELS = {
    "service_flight": "Service flight",
    "drone_wait": "Drone pickup wait",
    "first_rider": "Feeder rider",
    "terminal_rider": "Terminal rider",
    "bases": "Bases",
    "lockers": "Lockers",
    "links": "Links",
    "fleet": "Fleet",
    "direct_courier": "Direct courier",
    "empty_flight": "Empty flight",
}
DETAIL_COST_CN = {
    "service_flight": "服务飞行",
    "drone_wait": "无人机等待",
    "first_rider": "首端骑手",
    "terminal_rider": "末端骑手",
    "bases": "基地",
    "lockers": "柜格",
    "links": "航线",
    "fleet": "机队",
    "direct_courier": "直送骑手",
    "empty_flight": "空载飞行",
    "net_total": "总成本",
}


def _source(inputs):
    metadata = inputs.get("metadata", {})
    value = metadata.get("source", {})
    return value if isinstance(value, dict) else metadata.get("source_metadata", {})


def _ratio(a, b):
    return float(a / b) if b else None


def _fmt(value, precision=3):
    if value is None:
        return "—"
    if isinstance(value, (bool, np.bool_)):
        return "是" if value else "否"
    if isinstance(value, (float, np.floating)):
        return f"{value:,.{precision}f}"
    return str(value)


def _table(rows, columns, precision=3):
    """Columns are (source field, display header); preserve meaningful zeros."""
    lines = [
        "| " + " | ".join(label for _, label in columns) + " |",
        "|" + "---|" * len(columns),
    ]
    lines.extend(
        "| " + " | ".join(_fmt(row.get(key), precision) for key, _ in columns) + " |"
        for row in rows
    )
    return lines


def _headline(case):
    s, r, t = (
        case["operations"]["summary"],
        case["result"],
        case["operations"]["tables"],
    )
    costs = {row["component"]: row["cny_per_hour"] for row in t["costs"]}
    fleet = {row["component"]: row["drones"] for row in t["fleet"]}
    busy = fleet["service"] + fleet["repositioning"]
    baseline = float(
        np.asarray(case["inputs"]["demand"]) @ np.asarray(case["inputs"]["ground_cost"])
    )
    rows = t["orders"]
    receipt_sum = sum(
        row["drone_per_hour"] * (row["mean_receipt_minutes"] or 0) for row in rows
    )
    row = dict(
        case=case["name"],
        dataset=case["dataset"],
        max_orders=case["k"],
        flow_policy=s["flow_policy"],
        method=r.get("method"),
        scenario_sha256=case["fingerprint"],
        objective=s["objective"],
        bound=r.get("bound"),
        gap=r.get("gap"),
        gap_percent=100 * r["gap"] if r.get("gap") is not None else None,
        status=r.get("status"),
        termination_reason=r.get("termination_reason"),
        wall_seconds=r.get("wall"),
        solve_seconds=r.get("solve_seconds"),
        budget_seconds=r.get("budget"),
        catalogue_complete=r.get("catalogue_complete"),
        pricing_complete=r.get("pricing_complete"),
        path_scope=r.get("path_scope"),
        all_courier_cost=baseline,
        savings_per_hour=baseline - s["objective"],
        savings_percent=100 * (baseline - s["objective"]) / baseline
        if baseline
        else None,
        cost_per_order=_ratio(s["objective"], s["demand"]),
        demand_per_hour=s["demand"],
        drone_orders_per_hour=s["served"],
        direct_orders_per_hour=s["direct"],
        drone_service_percent=100 * s["served"] / s["demand"],
        single_tasks_per_hour=s["single_tasks"],
        double_tasks_per_hour=s["double_tasks"],
        service_tasks_per_hour=s["task_flights"],
        empty_tasks_per_hour=s["empty_tasks"],
        orders_per_service_task=_ratio(s["served"], s["task_flights"]),
        mean_task_minutes=_ratio(60 * fleet["service"], s["task_flights"]),
        mean_drone_receipt_minutes=_ratio(receipt_sum, s["served"]),
        fleet=s["fleet"],
        idle_drones=fleet["idle"],
        busy_drones=busy,
        fleet_per_100_drone_orders=_ratio(100 * s["fleet"], s["served"]),
        optimal_savings_lower=baseline - s["objective"],
        optimal_savings_upper=baseline - r["bound"]
        if r.get("bound") is not None
        else None,
        busy_percent=100 * busy / s["fleet"] if s["fleet"] else 0.0,
        busy_minutes_per_hour=60 * busy,
        fleet_slack=fleet["slack"],
        opened_bases=s["opened"]["bases"],
        opened_lockers=s["opened"]["lockers"],
        compartments=s["slot_total"],
        mean_reservations=sum(v["mean_occupancy"] for v in t["lockers"]),
        validated=s["checks"]["passed"],
    )
    row.update({"cost_" + key: value for key, value in costs.items()})
    row.update(
        {
            "detail_cost_" + v["component"]: v["cny_per_hour"]
            for v in t.get("cost_detail", [])
        }
    )
    row.update(
        {"busy_" + v["component"]: v["drones"] for v in t.get("busy_components", [])}
    )
    row.update(
        {"catalogue_" + key: value for key, value in r.get("catalogue", {}).items()}
    )
    return row


def _policy_changes(rows):
    grouped = defaultdict(dict)
    for row in rows:
        grouped[row["dataset"]][row["max_orders"]] = row
    changes = []
    for dataset, values in grouped.items():
        if set(values) != {1, 2}:
            raise ValueError(f"Dataset {dataset} requires exactly K=1 and K=2")
        first, second = values[1], values[2]
        lower = (
            second["bound"] - first["objective"]
            if second["bound"] is not None
            else None
        )
        upper = (
            second["objective"] - first["bound"] if first["bound"] is not None else None
        )
        certificate = (
            "K2_lower_optimal_cost"
            if upper is not None and upper < 0
            else "K2_higher_optimal_cost"
            if lower is not None and lower > 0
            else "optimal_cost_order_unresolved"
        )
        item = dict(
            dataset=dataset,
            optimal_cost_difference_lower=lower,
            optimal_cost_difference_upper=upper,
            certificate=certificate,
        )
        for key in (
            "objective",
            "drone_orders_per_hour",
            "direct_orders_per_hour",
            "fleet",
            "idle_drones",
            "busy_drones",
            "double_tasks_per_hour",
            "service_tasks_per_hour",
            "mean_drone_receipt_minutes",
            "compartments",
            "cost_per_order",
        ):
            a, b = first[key], second[key]
            item["delta_" + key] = b - a if a is not None and b is not None else None
        item["cost_reduction_percent"] = (
            100 * (first["objective"] - second["objective"]) / first["objective"]
        )
        changes.append(item)
    return changes


def _labels(cases):
    return [
        f"{DATASET_LABELS.get(c['dataset'], c['dataset'])}\nK={c['k']}" for c in cases
    ]


def _headline_figures(cases, rows, out):
    names = _labels(cases)
    x = np.arange(len(cases))
    colors = [POLICY_COLORS[c["k"]] for c in cases]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    panels = [
        ("cost_per_order", "Total cost per original order", "CNY / order"),
        (
            "savings_percent",
            "Savings against all-courier baseline",
            "% of baseline cost",
        ),
        ("drone_service_percent", "Original orders served by drone", "% of all orders"),
        ("fleet", "Continuous planned fleet", "Drones"),
        ("mean_task_minutes", "Mean service task duration", "Minutes / service task"),
        (
            "mean_drone_receipt_minutes",
            "Mean drone-order receipt time",
            "Minutes from own request",
        ),
    ]
    for ax, (key, title, unit) in zip(axes.flat, panels):
        values = [float(row[key]) if row[key] is not None else np.nan for row in rows]
        ax.bar(x, values, color=colors, width=0.65)
        for i, v in enumerate(values):
            if not np.isfinite(v):
                ax.annotate(
                    "N/A",
                    (i, 0),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha="center",
                    fontsize=10,
                )
                continue
            ax.annotate(
                f"{v:.2f}",
                (i, v),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                fontsize=10,
            )
        ax.set(xticks=x, xticklabels=names, ylabel=unit, title=title)
        ax.margins(y=0.15)
        _grid(ax, "y")
    fig.tight_layout()
    _save(fig, out, "headline_comparison")

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    for ax, normalized in zip(axes, (False, True)):
        bottom = np.zeros(len(cases))
        for component, (label, color) in COSTS.items():
            values = np.array([row["cost_" + component] for row in rows])
            if normalized:
                values /= np.array([row["demand_per_hour"] for row in rows])
            ax.bar(x, values, bottom=bottom, label=label, color=color, width=0.65)
            bottom += values
        ax.set(
            xticks=x,
            xticklabels=names,
            title="Cost composition" if not normalized else "Cost per original order",
            ylabel="CNY / hour" if not normalized else "CNY / order",
        )
        _grid(ax, "y")
    axes[1].legend(loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False)
    fig.tight_layout()
    _save(fig, out, "cost_components")

    fig, ax = plt.subplots(figsize=(11, 6))
    bottom = np.zeros(len(cases))
    components = list(BUSY_LABELS)
    if not all(c["operations"]["tables"].get("busy_components") for c in cases):
        components = []
        values = np.array([row["busy_drones"] for row in rows])
        ax.bar(x, values, color=BUSY_COLORS[0], label="Busy drones")
        bottom += values
    for component, color in zip(components, BUSY_COLORS):
        values = np.array([row.get("busy_" + component, 0) for row in rows])
        ax.bar(x, values, bottom=bottom, label=BUSY_LABELS[component], color=color)
        bottom += values
    for key, label, color in (
        ("idle_drones", "Idle inventory", "#7d729a"),
        ("fleet_slack", "Fleet slack", "#e4e8ec"),
    ):
        values = np.maximum(0, np.array([row[key] for row in rows]))
        ax.bar(x, values, bottom=bottom, label=label, color=color)
        bottom += values
    ax.set(
        xticks=x,
        xticklabels=names,
        ylabel="Average drones / continuous fleet",
        title="Fleet and busy-time decomposition",
    )
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False)
    _grid(ax, "y")
    fig.tight_layout()
    _save(fig, out, "fleet_components")


def _cost_delta_rows(rows):
    """Component deltas add to the feasible-incumbent total difference."""
    grouped = defaultdict(dict)
    for row in rows:
        grouped[row["dataset"]][row["max_orders"]] = row
    output = []
    for dataset, pair in grouped.items():
        if set(pair) != {1, 2}:
            raise ValueError(f"Dataset {dataset} requires exactly K=1 and K=2")
        first, second = pair[1], pair[2]
        detailed = all(
            "detail_cost_" + component in first and "detail_cost_" + component in second
            for component in DETAIL_COST_LABELS
        )
        components = DETAIL_COST_LABELS if detailed else COSTS
        for component in (*components, "net_total"):
            field = (
                "objective"
                if component == "net_total"
                else ("detail_cost_" if detailed else "cost_") + component
            )
            output.append(
                dict(
                    dataset=dataset,
                    component=component,
                    k1_cny_per_hour=first[field],
                    k2_cny_per_hour=second[field],
                    delta_cny_per_hour=second[field] - first[field],
                )
            )
    return output


def _normalized_fleet_rows(rows):
    """Fleet capacity per 100 drone orders/hour; undefined with zero service."""
    output = []
    fields = {
        **{"busy_" + name: name for name in BUSY_LABELS},
        "idle_drones": "idle",
        "fleet_slack": "slack",
        "fleet": "total",
    }
    for row in rows:
        for field, component in fields.items():
            value = row.get(field)
            output.append(
                dict(
                    case=row["case"],
                    dataset=row["dataset"],
                    max_orders=row["max_orders"],
                    component=component,
                    drones=value,
                    drone_orders_per_hour=row["drone_orders_per_hour"],
                    drones_per_100_drone_orders_per_hour=(
                        _ratio(100 * value, row["drone_orders_per_hour"])
                        if value is not None
                        else None
                    ),
                )
            )
    return output


def _diagnostic_figures(cases, rows, out):
    deltas = _cost_delta_rows(rows)
    normalized = _normalized_fleet_rows(rows)
    _csv(out / "cost_policy_deltas.csv", deltas)
    _csv(out / "normalized_fleet_components.csv", normalized)
    datasets = list(dict.fromkeys(row["dataset"] for row in rows))
    fig, axes = plt.subplots(
        1, len(datasets), figsize=(8 * len(datasets), 7.5), squeeze=False
    )
    for ax, dataset in zip(axes.flat, datasets):
        selected = [row for row in deltas if row["dataset"] == dataset]
        values = np.array([row["delta_cny_per_hour"] for row in selected])
        colors = [
            "#22313d"
            if row["component"] == "net_total"
            else "#bd7b27"
            if row["delta_cny_per_hour"] > 0
            else "#356c91"
            for row in selected
        ]
        y = np.arange(len(selected))
        ax.barh(y, values, color=colors, height=0.65)
        labels = [
            "Net total"
            if row["component"] == "net_total"
            else DETAIL_COST_LABELS[row["component"]]
            if row["component"] in DETAIL_COST_LABELS
            else COSTS[row["component"]][0]
            for row in selected
        ]
        ax.set(
            yticks=y,
            yticklabels=labels,
            xlabel="K=2 minus K=1 cost (CNY / hour)",
            title=DATASET_LABELS.get(dataset, dataset),
        )
        ax.invert_yaxis()
        ax.axvline(0, color="#22313d", lw=0.8)
        ax.axhline(len(selected) - 1.5, color="#c6cdd3", lw=0.8)
        maxabs = max(float(np.max(np.abs(values))), 1.0)
        ax.set_xlim(
            min(float(values.min()), 0) - 0.28 * maxabs,
            max(float(values.max()), 0) + 0.28 * maxabs,
        )
        for i, value in enumerate(values):
            ax.annotate(
                f"{value:+.2f}",
                (value, i),
                xytext=(5 if value >= 0 else -5, 0),
                textcoords="offset points",
                ha="left" if value >= 0 else "right",
                va="center",
                fontsize=10,
            )
        _grid(ax, "x")
    fig.suptitle("Cost changes between validated feasible incumbents", fontsize=15)
    fig.tight_layout()
    _save(fig, out, "cost_policy_deltas")

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(cases))
    bottom = np.zeros(len(cases))
    by_key = {
        (row["case"], row["component"]): row["drones_per_100_drone_orders_per_hour"]
        for row in normalized
    }
    components = [*BUSY_LABELS, "idle", "slack"]
    colors = [*BUSY_COLORS, "#7d729a", "#e4e8ec"]
    names = {**BUSY_LABELS, "idle": "Idle inventory", "slack": "Fleet slack"}
    for component, color in zip(components, colors):
        values = np.array(
            [
                by_key[c["name"], component]
                if by_key[c["name"], component] is not None
                else np.nan
                for c in cases
            ]
        )
        ax.bar(
            x,
            np.maximum(values, 0),
            bottom=bottom,
            label=names[component],
            color=color,
            width=0.65,
        )
        bottom += np.nan_to_num(np.maximum(values, 0))
    for i, case in enumerate(cases):
        value = by_key[case["name"], "total"]
        ax.annotate(
            "N/A" if value is None else f"{value:.3f}",
            (i, 0 if value is None else value),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            fontsize=10,
        )
    ax.set(
        xticks=x,
        xticklabels=_labels(cases),
        ylabel="Planned drones per 100 drone orders/hour",
        title="Fleet capacity normalized by drone-service throughput",
    )
    upper = max(
        [
            value
            for (case_name, component), value in by_key.items()
            if component == "total" and value is not None
        ]
        + [1.0]
    )
    ax.set_ylim(0, 1.16 * upper)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False)
    _grid(ax, "y")
    fig.tight_layout()
    _save(fig, out, "normalized_fleet_components")


def _study_area(cases, out):
    datasets = list(dict.fromkeys(c["dataset"] for c in cases))
    fig, axes = plt.subplots(
        1, len(datasets), figsize=(7 * len(datasets), 6), squeeze=False
    )
    for ax, dataset in zip(axes.flat, datasets):
        case = next(c for c in cases if c["dataset"] == dataset)
        data = case["inputs"]
        xy = _coordinates(data)
        origins = np.asarray(data.get("origins", []), float)
        destinations = np.asarray(data.get("destinations", []), float)
        if origins.ndim == 2 and destinations.ndim == 2 and "od_origin" in data:
            origin = origins[np.asarray(data["od_origin"], int)]
            destination = destinations[np.asarray(data["od_destination"], int)]
            weights = np.asarray(data["demand"])
            for o, d, w in zip(origin, destination, weights):
                ax.plot(
                    [o[0], d[0]],
                    [o[1], d[1]],
                    color="#a8b3bc",
                    alpha=0.3,
                    lw=0.25 + 1.5 * w / weights.max(),
                    zorder=1,
                )
            ax.scatter(
                origin[:, 0],
                origin[:, 1],
                s=12 + 75 * weights / weights.max(),
                marker="^",
                c="#bd7b27",
                alpha=0.7,
                label="OD origins",
                zorder=2,
            )
            ax.scatter(
                destination[:, 0],
                destination[:, 1],
                s=12 + 75 * weights / weights.max(),
                marker="v",
                c="#73978c",
                alpha=0.7,
                label="OD destinations",
                zorder=2,
            )
        if xy is not None:
            n_l = int(data["n_l"])
            ax.scatter(
                xy[:n_l, 0],
                xy[:n_l, 1],
                s=100,
                marker="s",
                c="#22313d",
                label="Base candidates",
                zorder=4,
            )
            ax.scatter(
                xy[n_l:, 0],
                xy[n_l:, 1],
                s=60,
                marker="o",
                facecolors="white",
                edgecolors="#356c91",
                linewidths=1.5,
                label="Locker candidates",
                zorder=3,
            )
            for i, (px, py) in enumerate(xy):
                ax.annotate(
                    f"B{i}" if i < n_l else f"H{i - n_l}",
                    (px, py),
                    xytext=(4, 4),
                    textcoords="offset points",
                    fontsize=9,
                )
        bounds = _source(data).get("crop_bounds_local_km")
        if bounds:
            x0, x1, y0, y1 = (bounds[k] for k in ("x_min", "x_max", "y_min", "y_max"))
            ax.plot(
                [x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], "--", color="#22313d", lw=1
            )
        ax.set(
            xlabel="Local easting (km)",
            ylabel="Local northing (km)",
            title=DATASET_LABELS.get(dataset, dataset),
        )
        ax.set_aspect("equal", adjustable="datalim")
        _grid(ax, "both")
    axes[0, 0].legend(
        loc="upper left", bbox_to_anchor=(0, -0.15), ncol=2, frameon=False
    )
    fig.suptitle(
        "Matched square crops and all aggregated OD demand; source CRS unspecified",
        fontsize=15,
    )
    fig.tight_layout()
    _save(fig, out, "study_areas")


def _case_figures(case, out):
    t = case["operations"]["tables"]
    label = f"{DATASET_LABELS.get(case['dataset'], case['dataset'])}, K={case['k']}"
    network_figure(case["inputs"], case["result"], case["operations"], out)
    bases = t["bases"]
    x = np.arange(len(bases))
    names = [f"B{r['base']}" for r in bases]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    axes[0].bar(x, [r["availability"] for r in bases], color="#356c91")
    axes[0].set(
        ylim=(0, 1.05),
        ylabel="Availability probability rho",
        title="Base availability (not utilization)",
    )
    for i, r in enumerate(bases):
        axes[0].annotate(
            f"{r['availability']:.4f}",
            (i, r["availability"]),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
        )
    for shift, key, name, color in (
        (-0.2, "outflow", "All departures", "#356c91"),
        (0.2, "inflow", "All returns", "#bd7b27"),
    ):
        axes[1].bar(
            x + shift, [r[key] for r in bases], width=0.4, label=name, color=color
        )
    axes[1].set(ylabel="Drone tasks / hour", title="Base task-flow balance")
    axes[1].legend(frameon=False)
    axes[2].bar(x, [r["idle"] for r in bases], color="#7d729a", label="Idle inventory")
    axes[2].bar(
        x,
        [r.get("busy_drones", 0) for r in bases],
        bottom=[r["idle"] for r in bases],
        color="#356c91",
        label="Busy, attributed to departure base",
    )
    axes[2].set(ylabel="Average drones", title="Idle and attributed busy drones")
    axes[2].legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), frameon=False)
    for ax in axes:
        ax.set(xticks=x, xticklabels=names)
        _grid(ax, "y")
    fig.suptitle(label)
    fig.tight_layout()
    _save(fig, out, "base_operations")

    lockers = t["lockers"]
    x = np.arange(len(lockers))
    fig, axes = plt.subplots(2, 1, figsize=(13, 9))
    axes[0].bar(
        x,
        [r["compartments"] for r in lockers],
        color="#e4e8ec",
        label="Installed physical compartments",
    )
    axes[0].bar(
        x,
        [r["mean_occupancy"] for r in lockers],
        color="#356c91",
        width=0.5,
        label="Mean reservation load",
    )
    axes[0].plot(
        x,
        [r["mean_threshold"] for r in lockers],
        "o--",
        color="#bd7b27",
        label="Allowed Poisson mean",
    )
    axes[0].set(
        ylabel="Compartments / mean reservations",
        title="Locker reservations and chance-constraint threshold",
    )
    axes[0].legend(frameon=False, ncol=3)
    for shift, key, name, color in (
        (-0.2, "pickup_orders_per_hour", "Picked up by drone", "#356c91"),
        (0.2, "delivery_orders_per_hour", "Delivered by drone", "#bd7b27"),
    ):
        axes[1].bar(
            x + shift,
            [r.get(key, 0) for r in lockers],
            width=0.4,
            label=name,
            color=color,
        )
    axes[1].set(ylabel="Order handoffs / hour", title="Locker order throughput")
    axes[1].legend(frameon=False)
    for ax in axes:
        ax.set(xticks=x, xticklabels=[f"H{r['locker']}" for r in lockers])
        _grid(ax, "y")
    fig.suptitle(label)
    fig.tight_layout()
    _save(fig, out, "locker_operations")

    orders = t["orders"]
    x = np.array([r["od"] for r in orders])
    fig, axes = plt.subplots(3, 1, figsize=(15, 12))
    direct = np.array([r["direct_per_hour"] for r in orders])
    drone = np.array([r["drone_per_hour"] for r in orders])
    axes[0].bar(x, drone, color="#356c91", label="Drone orders")
    axes[0].bar(x, direct, bottom=drone, color="#c6cdd3", label="Direct courier orders")
    axes[0].set(ylabel="Orders / hour", title="All OD demand is retained")
    axes[0].legend(frameon=False, ncol=2)
    axes[1].bar(
        x,
        [r["first_per_hour"] for r in orders],
        color="#356c91",
        label="First position",
    )
    axes[1].bar(
        x,
        [r["second_per_hour"] for r in orders],
        bottom=[r["first_per_hour"] for r in orders],
        color="#bd7b27",
        label="Second position",
    )
    axes[1].set(
        ylabel="Drone orders / hour", title="Position within the executed drone task"
    )
    axes[1].legend(frameon=False, ncol=2)
    axes[2].plot(
        x,
        [r["deadline_minutes"] for r in orders],
        "--",
        color="#8a939e",
        label="OD mean receipt limit",
    )
    axes[2].plot(
        x,
        [
            np.nan if r["mean_receipt_minutes"] is None else r["mean_receipt_minutes"]
            for r in orders
        ],
        "o",
        color="#356c91",
        markersize=4,
        label="Flow-weighted mean receipt",
    )
    axes[2].set(
        ylabel="Minutes from own request", title="Unserved OD receipt times are omitted"
    )
    axes[2].legend(frameon=False, ncol=2)
    for ax in axes:
        ax.set(xlabel="OD ID (zero based)", xticks=x[:: max(1, len(x) // 25)])
        _grid(ax, "y")
    fig.suptitle(label)
    fig.tight_layout()
    _save(fig, out, "od_operations")

    n = int(case["inputs"]["n_l"] + case["inputs"]["n_h"])
    matrix = np.zeros((n, n))
    for row in t["links"]:
        matrix[int(row["u"]), int(row["v"])] += row["traversals_per_hour"]
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(matrix, cmap="Blues", vmin=0)
    names = [
        f"B{i}" if i < case["inputs"]["n_l"] else f"H{i - case['inputs']['n_l']}"
        for i in range(n)
    ]
    ax.set(
        xticks=np.arange(n),
        xticklabels=names,
        yticks=np.arange(n),
        yticklabels=names,
        xlabel="Arrival node",
        ylabel="Departure node",
        title=label + ": directed flight traversals",
    )
    for u, v in zip(*np.where(matrix > 1e-5)):
        ax.text(
            v,
            u,
            f"{matrix[u, v]:.0f}",
            ha="center",
            va="center",
            fontsize=9,
            color="white" if matrix[u, v] > 0.5 * matrix.max() else "#22313d",
        )
    fig.colorbar(im, ax=ax, label="Flight traversals / hour")
    fig.tight_layout()
    _save(fig, out, "arc_flows")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, table, key, weight, title, unit in (
        (
            axes[0],
            t["paths"],
            "mean_operating_minutes",
            "flow_per_hour",
            "Service task mean duration",
            "Mean minutes / task",
        ),
        (
            axes[1],
            t["order_paths"],
            "receipt_hours",
            "flow_per_hour",
            "Order-path mean receipt time",
            "Mean minutes from own request",
        ),
    ):
        selected = [
            r
            for r in table
            if r.get("kind", "service") == "service" and r[weight] > 1e-10
        ]
        if selected:
            values = np.array(
                [r[key] * (60 if key == "receipt_hours" else 1) for r in selected]
            )
            weights = np.array([r[weight] for r in selected])
            ordered = np.argsort(values)
            ax.step(
                values[ordered],
                np.cumsum(weights[ordered]) / weights.sum(),
                where="post",
                color="#356c91",
            )
        else:
            ax.text(
                0.5,
                0.5,
                "No positive drone flow",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
        ax.set(xlabel=unit, ylabel="Cumulative flow share", ylim=(0, 1.03), title=title)
        _grid(ax, "both")
    fig.suptitle(
        label + ": flow-weighted distribution of path means (not realized delays)",
        fontsize=14,
    )
    fig.tight_layout()
    _save(fig, out, "path_mean_distributions")
    if case["k"] == 2:
        n_od = len(t["orders"])
        pair_matrix = np.zeros((n_od, n_od))
        for row in t.get("task_pairs", []):
            pair_matrix[row["first_od"], row["second_od"]] += row[
                "actual_tasks_per_hour"
            ]
        fig, ax = plt.subplots(figsize=(9, 8))
        im = ax.imshow(pair_matrix, cmap="Blues", vmin=0)
        ax.set(
            xlabel="Second OD ID",
            ylabel="First OD ID",
            title=label + ": executed ordered pairs",
        )
        fig.colorbar(im, ax=ax, label="Double-order tasks / hour")
        fig.tight_layout()
        _save(fig, out, "ordered_pair_flows")


def _od_changes(cases, out):
    datasets = list(dict.fromkeys(c["dataset"] for c in cases))
    fig, axes = plt.subplots(
        len(datasets), 1, figsize=(15, 5 * len(datasets)), squeeze=False
    )
    rows = []
    for ax, dataset in zip(axes.flat, datasets):
        pair = {c["k"]: c for c in cases if c["dataset"] == dataset}
        first = pair[1]["operations"]["tables"]["orders"]
        second = pair[2]["operations"]["tables"]["orders"]
        values = []
        for a, b in zip(first, second):
            assert a["od"] == b["od"]
            delta = b["drone_per_hour"] - a["drone_per_hour"]
            values.append(delta)
            rows.append(
                dict(
                    dataset=dataset,
                    od=a["od"],
                    demand_per_hour=a["demand_per_hour"],
                    k1_drone_per_hour=a["drone_per_hour"],
                    k2_drone_per_hour=b["drone_per_hour"],
                    delta_drone_per_hour=delta,
                    k1_mean_receipt_minutes=a["mean_receipt_minutes"],
                    k2_mean_receipt_minutes=b["mean_receipt_minutes"],
                )
            )
        ax.bar(
            [r["od"] for r in first],
            values,
            color=["#356c91" if v >= 0 else "#bd7b27" for v in values],
        )
        ax.axhline(0, color="#22313d", lw=0.8)
        ax.set(
            xlabel="OD ID (zero based)",
            ylabel="K=2 minus K=1 drone orders / hour",
            title=DATASET_LABELS.get(dataset, dataset),
        )
        _grid(ax, "y")
    fig.tight_layout()
    _save(fig, out, "od_policy_changes")
    _csv(out / "od_policy_changes.csv", rows)
    return rows


def _interpretation(rows, changes):
    result = []
    for change in changes:
        name = DATASET_NAMES.get(change["dataset"], change["dataset"])
        pair = {r["max_orders"]: r for r in rows if r["dataset"] == change["dataset"]}
        a, b = pair[1], pair[2]
        movement = "降低" if change["delta_objective"] < 0 else "增加"
        result += [
            f"**{name}。** 已找到的 K=2 可行方案相对 K=1，总成本{movement} {abs(change['delta_objective']):,.3f} 元/小时；"
            f"无人机服务量变化 {change['delta_drone_orders_per_hour']:+,.3f} 单/小时，"
            f"连续机队变化 {change['delta_fleet']:+,.3f} 架，忙碌无人机均值变化 {change['delta_busy_drones']:+,.3f} 架。"
            f"K=2 执行 {b['double_tasks_per_hour']:,.3f} 个双单任务/小时，其第二单贡献相同数量的订单；"
            f"每个服务任务平均承载 {b['orders_per_service_task'] or 0:.3f} 单。",
            "",
        ]
        component_deltas = sorted(
            ((key, b["cost_" + key] - a["cost_" + key]) for key in COSTS),
            key=lambda kv: abs(kv[1]),
            reverse=True,
        )
        cn = {
            "direct_courier": "直送骑手",
            "service": "无人机服务及衔接骑手",
            "fleet": "机队",
            "bases": "基地",
            "lockers": "柜格",
            "links": "航线",
            "repositioning": "空载调位",
        }
        result += [
            "成本差额按相同成本口径精确分解为："
            + "；".join(f"{cn[k]} {v:+,.3f} 元/小时" for k, v in component_deltas)
            + "。这是一组优化方案之间的会计分解，不能单独解释为某一运营机制的因果效应。",
            "",
        ]
        lo, hi = (
            change["optimal_cost_difference_lower"],
            change["optimal_cost_difference_upper"],
        )
        if lo is not None and hi is not None:
            result += [
                f"令 C₁*、C₂* 分别为两个离散设施选址模型的未知最优值。已认证上下界给出 "
                f"C₂*−C₁* ∈ [{lo:,.3f}, {hi:,.3f}] 元/小时。"
                + (
                    "该区间完全小于零，因此当前界限足以证明 K=2 最优成本较低。"
                    if hi < 0
                    else "该区间完全大于零，因此当前界限足以证明 K=2 最优成本较高。"
                    if lo > 0
                    else "该区间跨越零，因此目前只能比较已找到的可行方案，不能认证两者最优成本的严格排序。"
                ),
                "",
            ]
    return result


def _markdown(cases, rows, changes, out, study_root):
    lines = [
        "# 南山与美团 TSL：同规模 K=1 / K=2 真实数据实验",
        "",
        "本报告比较同一数据集上相同的候选基地、柜点、50 个聚合 OD、需求、费用和物理参数；"
        "K 是一次无人机服务任务最多处理的订单数。K=1 只允许单单任务，K=2 允许单单与双单任务。"
        "当前采用 `aggregate` 流量口径，双单需求在基地池内跨首程路径汇总分配。",
        "",
        "## 结论与求解证据",
        "",
    ]
    lines += _interpretation(rows, changes)
    display = [
        dict(
            r,
            label=f"{DATASET_NAMES.get(r['dataset'], r['dataset'])} K={r['max_orders']}",
        )
        for r in rows
    ]
    lines += _table(
        display,
        [
            ("label", "方案"),
            ("objective", "可行总成本 元/h"),
            ("bound", "认证下界 元/h"),
            ("gap_percent", "gap %"),
            ("termination_reason", "停止原因"),
            ("wall_seconds", "墙钟 s"),
            ("catalogue_complete", "路径目录完整"),
        ],
    )
    lines += [
        "",
        "gap=(可行总成本−认证下界)/|可行总成本|。认证下界是优化下界，不是统计置信区间。"
        "`time_limit` 表示达到时间限制；即使全部有限候选路径已枚举，也不等于已证明全局最优。"
        "所有指标均由保存的物理路径和流量重新计算，并通过独立守恒、资源与目标值校验。",
        "",
        "求解方法："
        + ", ".join(sorted({str(c["result"].get("method")) for c in cases}))
        + "；候选路径范围："
        + ", ".join(sorted({str(c["result"].get("path_scope")) for c in cases}))
        + "。`direct` 表示每个飞行阶段使用直连航段；目录完整不表示枚举了该范围之外的多跳路线。",
        "",
        "## 数据、地图与可比性",
        "",
    ]
    source_rows = []
    for dataset in dict.fromkeys(c["dataset"] for c in cases):
        case = next(c for c in cases if c["dataset"] == dataset)
        data = case["inputs"]
        s = _source(data)
        row = dict(
            dataset=DATASET_NAMES.get(dataset, dataset),
            days=", ".join(map(str, s.get("selected_days", []))),
            orders=s.get("demand_orders_in_area"),
            exposure=s.get("demand_exposure_hours"),
            demand=float(np.sum(data["demand"])),
            candidate_orders=s.get("candidate_orders_in_area"),
            area=4 * s["crop_half_size_km"] ** 2
            if s.get("crop_half_size_km") is not None
            else None,
            center=str(s.get("crop_center_local_km")),
            source=s.get("raw_path"),
            sha256=s.get("raw_file_sha256"),
            positive_kappa=int(np.count_nonzero(data["kappa"])),
        )
        source_rows.append(row)
    lines += _table(
        source_rows,
        [
            ("dataset", "数据"),
            ("orders", "午间原始单数"),
            ("exposure", "累计午间观测 h"),
            ("demand", "订单/h"),
            ("candidate_orders", "候选时段原始单数"),
            ("area", "面积 km²"),
            ("center", "局部坐标切片中心"),
            ("positive_kappa", "正 κ 有序 OD 对数"),
        ],
    )
    lines += [
        "",
        "两端均落入正方形切片的订单才进入实例。需求使用 12:00–13:00，候选点使用 08:00–24:00，"
        "每个数据源使用其全部可用日期；不人为缩放需求。50 个 OD 是对保留午间订单进行四维起终点联合聚类的结果，"
        "不是抽取 50 条原始订单。两组数据的日期数和每小时订单量可能不同，因此总成本和机队还需结合每单成本与服务比例比较。",
        "",
        "骑手时间参数由各数据集自己的有效取餐后样本估计，采用相同估计程序，数值不要求一致。"
        "数据只观测到订单、取餐和送达等时间；模型的指数等待、均值资源和 Poisson 预约负荷是规划近似，"
        "不是从这些实验中验证得到的随机分布。源数据坐标基准未确认，地图显示原始经纬度投影后的局部公里坐标，不叠加未经核实的公共底图。",
        "",
        "![研究区域](study_areas.png)",
        "",
        "### 来源日期与文件",
        "",
    ]
    for row in source_rows:
        lines += [
            f"- {row['dataset']}：日期/匿名日标识 `{row['days']}`；原始文件 `{row['source']}`；SHA-256 `{row['sha256']}`。"
        ]
    audit_document = study_root / "INPUTS.md"
    if audit_document.exists():
        lines += [
            "",
            f"切片选择、数据清洗、日期口径、逐日需求、骑手时长估计与独立输入核验见 [输入审计报告]({audit_document})。",
            "",
        ]
    lines += [
        "",
        "## 指标先定义，再读数",
        "",
        "| 指标 | 定义、单位和解释 |",
        "|---|---|",
        "| 原始需求 | 每小时聚合前原始订单数，经聚合后总量不变；单位 单/h。 |",
        "| 订单流与任务流 | 一个双单任务贡献 1 个任务和 2 单；无人机订单流=单单任务流+2×双单任务流。 |",
        "| 基地可用率 ρ | 订单请求到来时基地能提供无人机的规划概率；不是无人机忙碌比例。 |",
        "| 空闲库存 I | 基地平均空闲无人机数，受 I≥ρ/(1−ρ) 的锥约束控制；它是平均库存而非停机位数量。 |",
        "| 忙碌无人机均值 | Σ(路径任务/h × 路径平均时长 h)，即 drone-hours/hour，量纲等于平均同时忙碌的无人机数量。 |",
        "| 忙碌分钟/h | 忙碌无人机均值×60；是所有无人机合计工作量，允许大于 60 分钟/h。 |",
        "| 平均任务分钟 | 任务流加权的飞行与等待时间，含返回；不同于每单送达时间。 |",
        "| 连续机队 Γ | 模型允许实数的规划机队数量，覆盖忙碌、空闲和数值松弛；不是部署好的整数机队。 |",
        "| 机队忙碌比例 | 平均忙碌无人机/Γ；与基地 ρ 是不同指标。 |",
        "| 基地归属忙碌 | 将服务与空载路径整段资源记在出发基地，便于对账；不表示这些无人机始终位于该基地。 |",
        "| 柜点取送流 | 无人机在柜点取走/投递订单的次数/h；同一柜点可兼任两种角色，不能当作唯一订单数相加。 |",
        "| 柜格平均预约负荷 | 路径流×预约时长形成的平均占用，含预约期，不等于仅包裹实际在柜内的数量。 |",
        "| Poisson 均值上限 | 在安装柜格与溢出容忍概率下允许的最大平均预约负荷；不是物理柜格数量。 |",
        "| 平均送达分钟 | 仅针对实际无人机服务订单，按订单流加权；每一单从自己的请求时刻计时。未服务 OD 留空。 |",
        "| 基地起降流 | 单单/双单服务任务与空载任务分别按一次出发、一次返回计数；长期基地总出入流应平衡。 |",
        "| 航段流 | 有向边上的实际飞行遍历次数/h；同一任务经过多条边会在每条边各计一次。 |",
        "| 总成本/原始单 | 总小时成本/全部原始订单量，含直送兜底和固定设施；不等于无人机单单边际成本。 |",
        "",
        "完整字段定义及单位见 `tables/metric_dictionary.csv`；零流点仍保留，未定义均值以空值表示。",
        "",
        "## 成本、服务与机队对比",
        "",
    ]
    lines += _table(
        display,
        [
            ("label", "方案"),
            ("demand_per_hour", "原始 单/h"),
            ("drone_orders_per_hour", "无人机 单/h"),
            ("direct_orders_per_hour", "直送 单/h"),
            ("drone_service_percent", "无人机占比 %"),
            ("cost_per_order", "元/原始单"),
            ("savings_percent", "比全直送节省 %"),
        ],
    )
    lines += [
        "",
        "![核心指标](headline_comparison.png)",
        "",
        "![成本分解](cost_components.png)",
        "",
        "![K2减K1成本分项差额](cost_policy_deltas.png)",
        "",
        "差额图以零为基准，正值表示 K=2 成本增加，负值表示减少；十个成本分项之和等于净总成本差。"
        "这些是已验证可行方案的差异，不代表已认证的最优差异。每个数据集独立选择横轴范围以看清本组差额，"
        "跨数据集比较大小应读取数值。",
        "",
    ]
    cost_deltas = _cost_delta_rows(rows)
    for dataset in dict.fromkeys(row["dataset"] for row in rows):
        detail = [
            dict(row, label=DETAIL_COST_CN.get(row["component"], row["component"]))
            for row in cost_deltas
            if row["dataset"] == dataset
        ]
        lines += [f"### {DATASET_NAMES.get(dataset, dataset)}：成本完整明细", ""]
        lines += _table(
            detail,
            [
                ("label", "成本项"),
                ("k1_cny_per_hour", "K=1 元/h"),
                ("k2_cny_per_hour", "K=2 元/h"),
                ("delta_cny_per_hour", "K=2−K=1 元/h"),
            ],
        )
        by_component = {row["component"]: row["delta_cny_per_hour"] for row in detail}
        if all(
            key in by_component
            for key in ("service_flight", "drone_wait", "direct_courier")
        ):
            lines += [
                "",
                f"服务飞行成本变化 {by_component['service_flight']:+,.3f} 元/h，"
                f"无人机等待成本变化 {by_component['drone_wait']:+,.3f} 元/h，"
                f"直送成本变化 {by_component['direct_courier']:+,.3f} 元/h。"
                "飞行、等待与两端骑手费用分列后，可以看出总成本变化是多项增减的净结果。",
                "",
            ]
    lines += _table(
        display,
        [
            ("label", "方案"),
            ("single_tasks_per_hour", "单单 任务/h"),
            ("double_tasks_per_hour", "双单 任务/h"),
            ("empty_tasks_per_hour", "空载 任务/h"),
            ("fleet", "机队 Γ"),
            ("idle_drones", "空闲均值"),
            ("busy_drones", "忙碌均值"),
            ("busy_percent", "忙碌 %"),
            ("mean_task_minutes", "分钟/服务任务"),
        ],
    )
    lines += [
        "",
        "![机队工作量](fleet_components.png)",
        "",
        "![按无人机服务量标准化的机队](normalized_fleet_components.png)",
        "",
        "标准化图将每个资源分项除以无人机服务订单/h，再乘 100，表示支撑每 100 单/h 无人机服务对应的机队配置。"
        "它把服务规模差异与资源强度区分开；分母包含双单任务的两个订单。空闲库存也按同一服务量分摊，"
        "因此该比率不是单架无人机的忙碌率；零无人机服务时比率不定义，标为 N/A。",
        "",
        "![OD 政策变化](od_policy_changes.png)",
        "",
        "K=2 的改变同时影响任务重构、路径选择、基地可用率、设施安装和直送比例。"
        "允许跨首程路径调配任务流后，不再要求每一条具体首程路径都独自承受相同的双单比例；"
        "但 κ、任务重构、基地池流量、订单守恒、柜格、续航和各单均值时限仍然保留。"
        "因此不能把差异简单归结为双单路径更短，也不能预设两个 K 模型具有严格包含关系。",
        "",
    ]
    for case in cases:
        t = case["operations"]["tables"]
        title = f"{DATASET_NAMES.get(case['dataset'], case['dataset'])} K={case['k']}"
        relative = f"cases/{case['name']}"
        lines += [
            f"## {title}：逐点明细",
            "",
            f"所有明细文件位于 [{relative}]({relative}/OPERATIONS.md)。",
            "",
        ]
        lines += ["### 基地：可用率、库存与任务流", ""]
        lines += _table(
            t["bases"],
            [
                ("base", "基地 ID"),
                ("opened", "开设"),
                ("availability", "ρ"),
                ("idle", "空闲 I"),
                ("offered_orders_per_hour", "分配原始单/h"),
                ("served_orders_depart_per_hour", "服务订单/h"),
                ("outflow", "总出发任务/h"),
                ("inflow", "总返回任务/h"),
                ("busy_drones", "归属忙碌均值"),
            ],
            4,
        )
        lines += [
            "",
            f"![{title} 基地]({relative}/base_operations.png)",
            "",
            "### 柜点：订单吞吐、预约负荷与容量",
            "",
        ]
        lines += _table(
            t["lockers"],
            [
                ("locker", "柜点 ID"),
                ("opened", "开设"),
                ("compartments", "柜格"),
                ("pickup_orders_per_hour", "取单/h"),
                ("delivery_orders_per_hour", "投单/h"),
                ("mean_occupancy", "预约均值"),
                ("mean_threshold", "均值上限"),
                ("potential_overflow_probability", "溢出概率"),
            ],
            4,
        )
        lines += [
            "",
            f"![{title} 柜点]({relative}/locker_operations.png)",
            "",
            "### 订单与物理网络",
            "",
            f"完整 {len(t['orders'])} 个 OD 的需求、分配基地、ρ、无人机/直送流、首单/第二单流和均值送达指标见 "
            f"[`orders.csv`]({relative}/orders.csv)；路径与柜点衔接明细见 [`order_paths.csv`]({relative}/order_paths.csv)。",
            "",
            f"![{title} OD]({relative}/od_operations.png)",
            "",
            f"![{title} 路径均值分布]({relative}/path_mean_distributions.png)",
            "",
            "上图展示路径平均时长按任务流或订单流加权后的分布。它刻画不同路径之间的均值差异，"
            "不是单次配送随机送达时间的经验分布，因此其分位数不能解释为实际配送时间分位数。",
            "",
            f"![{title} 网络]({relative}/network.png)",
            "",
            "网络图中 B 表示基地、H 表示柜点，编号从 0 开始；实心为启用点、空心为未启用候选点。"
            "first（蓝）包含基地到首单取餐柜及首单载货飞行，transfer（金）是首单投递柜到第二单取餐柜，"
            "second（绿）是第二单载货飞行，return（紫）是返回基地，empty（灰）是基地间空载调度。"
            "箭头表示方向；线宽在每张图内随该阶段流量增加，跨图精确比较请读航段 CSV。",
            "",
            f"![{title} 航段]({relative}/arc_flows.png)",
            "",
        ]
        if t.get("busy_components"):
            lines += ["### 无人机忙碌资源明细", ""]
            lines += _table(
                [
                    dict(row, label=BUSY_CN.get(row["component"], row["component"]))
                    for row in t["busy_components"]
                ],
                [
                    ("label", "资源"),
                    ("drones", "平均忙碌架数"),
                    ("busy_minutes_per_hour", "合计分钟/h"),
                ],
            )
            lines += [""]
        if t.get("flight_stages"):
            lines += _table(
                [
                    dict(row, label=STAGE_CN.get(row["stage"], row["stage"]))
                    for row in t["flight_stages"]
                ],
                [
                    ("label", "飞行阶段"),
                    ("task_legs_per_hour", "阶段执行/h"),
                    ("traversals_per_hour", "边遍历/h"),
                    ("mean_minutes", "分钟/阶段"),
                    ("busy_drones", "平均忙碌架数"),
                ],
            )
            lines += [
                "",
                "阶段执行次数包含同一柜点之间的零距离转移；边遍历次数只计实际飞行。"
                "续单窗口内的 δ/2 进入第二单自己的送达时钟，不额外加计无人机忙碌时间。",
                "",
            ]
        if case["k"] == 2:
            lines += [
                f"![{title} 双单任务对]({relative}/ordered_pair_flows.png)",
                "",
                "热图的行是首单 OD，列是第二单 OD，颜色表示每小时双单任务数。对角线允许同一聚合 OD 的两个不同订单组成任务。"
                f"完整的 κ、重构需求、目标服务流和实际流见 [`task_pairs.csv`]({relative}/task_pairs.csv)。",
                "",
            ]
    lines += [
        "## 可复核交付物",
        "",
        "- `comparison.csv/json`：每个方案的求解质量、成本、服务量、任务流、容量和机队指标。",
        "- `policy_changes.csv/json`：每个数据集的 K=2−K=1 变化及最优成本差的认证区间。",
        "- `tables/`：全部运营表合并导出，附数据集、K 与方案 ID；保留逐基地、逐柜点、逐 OD、逐航段、逐任务对。",
        "- `cases/`：每个方案的独立验证、全部原始精度表格与图。",
        "- `source_manifest.json`：输入、解、路径流文件的 SHA-256；本报告不重新优化。",
        "- 每张图均同时提供 `.png` 与矢量 `.pdf`；图中舍入只用于显示，CSV 保留数值精度。",
        "",
        "以上是规划模型下的稳态均值结果。没有额外运行离散事件仿真，因此不能据此声称实测等待分布、实际溢柜概率或整数无人机排班已验证。",
        "",
    ]
    (out / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def produce_study(study_root, runs=None, output=None):
    """Produce the detailed study from ``{dataset}_K{1,2}`` saved runs.

    ``runs`` optionally maps ``(dataset, k)`` to a saved run directory. The
    default discovers immediate children of ``study_root / 'runs'``. No source
    files or saved run files are modified; artifacts default to ``root/report``.
    """
    root = Path(study_root).resolve()
    out = Path(output).resolve() if output is not None else root / "report"
    if runs is None:
        runs = {}
        for path in sorted((root / "runs").iterdir()):
            if not (path / "result.json").exists() or "_K" not in path.name:
                continue
            dataset, number = path.name.rsplit("_K", 1)
            if number in ("1", "2"):
                runs[dataset, int(number)] = path
    if not runs:
        raise ValueError("No saved dataset_K1 / dataset_K2 runs found")
    cases = []
    fingerprints = defaultdict(set)
    for (dataset, k), path in sorted(runs.items()):
        path = Path(path).resolve()
        inputs, result = load_seed(path)
        if int(inputs["max_orders"]) != k:
            raise ValueError(f"Run label K={k} disagrees with saved inputs: {path}")
        if inputs.get("flow_policy") != "aggregate":
            raise ValueError(
                "Detailed real-data study requires current aggregate flow policy"
            )
        if float(np.sum(inputs["demand"])) <= 0 or float(result["objective"]) <= 0:
            raise ValueError(
                "Real-data study requires positive total demand and positive incumbent cost"
            )
        fingerprint = _scenario_fingerprint(inputs)
        fingerprints[dataset].add(fingerprint)
        cases.append(
            dict(
                name=f"{dataset}_K{k}",
                dataset=dataset,
                k=k,
                path=path,
                inputs=inputs,
                result=result,
                fingerprint=fingerprint,
            )
        )
    for dataset, values in fingerprints.items():
        if len(values) != 1:
            raise ValueError(
                f"K comparisons require identical physical inputs within dataset {dataset}"
            )
        if {c["k"] for c in cases if c["dataset"] == dataset} != {1, 2}:
            raise ValueError(f"Dataset {dataset} requires exactly K=1 and K=2")
    for case in cases:
        if out == case["path"] or out in case["path"].parents:
            raise ValueError(
                "Report output must not overwrite a source run or its parent"
            )
    out.mkdir(parents=True, exist_ok=True)
    all_tables = defaultdict(list)
    for case in cases:
        case_out = out / "cases" / case["name"]
        case["operations"] = export_operations(case["inputs"], case["result"], case_out)
        for name, table in case["operations"]["tables"].items():
            all_tables[name].extend(
                dict(
                    dataset=case["dataset"],
                    max_orders=case["k"],
                    case=case["name"],
                    **row,
                )
                for row in table
            )
    rows = [_headline(c) for c in cases]
    changes = _policy_changes(rows)
    (out / "tables").mkdir(exist_ok=True)
    for name, table in all_tables.items():
        _csv(out / "tables" / (name + ".csv"), table)
    _csv(out / "comparison.csv", rows)
    write_json(out / "comparison.json", rows)
    _csv(out / "policy_changes.csv", changes)
    write_json(out / "policy_changes.json", changes)
    _style()
    _headline_figures(cases, rows, out)
    _diagnostic_figures(cases, rows, out)
    _study_area(cases, out)
    _od_changes(cases, out)
    for case in cases:
        _case_figures(case, out / "cases" / case["name"])
    _markdown(cases, rows, changes, out, root)
    postprocessing_hashes = source_hashes()
    solver_hashes = {}
    changed_files = {}
    for case in cases:
        saved_manifest = case["path"] / "manifest.json"
        recorded = (
            json.loads(saved_manifest.read_text()).get("source_hashes", {})
            if saved_manifest.exists()
            else {}
        )
        solver_hashes[case["name"]] = recorded
        changed_files[case["name"]] = (
            sorted(
                name
                for name in set(recorded) | set(postprocessing_hashes)
                if recorded.get(name) != postprocessing_hashes.get(name)
            )
            if recorded
            else None
        )
    manifest = dict(
        independent_validation_passed=True,
        optimization_rerun=False,
        within_dataset_physics_identical=True,
        source_crs="unspecified; local kilometre maps",
        postprocessing_source_hashes=postprocessing_hashes,
        solver_source_hashes=solver_hashes,
        all_solver_runs_use_identical_source_hashes=len(
            {json.dumps(value, sort_keys=True) for value in solver_hashes.values()}
        )
        == 1
        and all(solver_hashes.values()),
        files_changed_after_optimization=changed_files,
        changes_after_optimization_confined_to_reporting=all(
            names is not None
            and all(name.startswith("src/reporting/") for name in names)
            for names in changed_files.values()
        ),
        sources={
            c["name"]: dict(
                directory=str(c["path"]),
                scenario_sha256=c["fingerprint"],
                files={
                    name: sha256(c["path"] / name)
                    for name in (
                        "input_metadata.json",
                        "input_arrays.npz",
                        "result.json",
                        "solution.npz",
                        "path_flows.json.gz",
                    )
                },
            )
            for c in cases
        },
        tables={name: len(table) for name, table in all_tables.items()},
        figures=[str(p.relative_to(out)) for p in sorted(out.rglob("*.png"))],
    )
    write_json(out / "source_manifest.json", manifest)
    return dict(
        output=str(out),
        cases=len(cases),
        figures=len(manifest["figures"]),
        independent_validation_passed=True,
        comparison=rows,
        changes=changes,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study_root", type=Path)
    parser.add_argument("--output", type=Path)
    options = parser.parse_args()
    result = produce_study(options.study_root, output=options.output)
    print(
        f"Validated {result['cases']} cases; {result['figures']} figures: {result['output']}"
    )
