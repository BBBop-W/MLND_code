"""Source-reconciled figures for the 15 new sensitivity runs in round three."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from .complete_figures import COSTS, FLEET
from .sensitivity_figures import _save, _style


GROUPS = ("demand", "speed", "max_cabinets", "walk_sensitivity")
TITLES = {"demand": "Order demand", "speed": "Flight speed",
          "max_cabinets": "Cabinet limit per site", "walk_sensitivity": "Walking-distance sensitivity"}
EXPECTED = {"demand": [.2, .3, .4], "speed": [.6, .7, .8, .9],
            "max_cabinets": [3, 4, 6, 7], "walk_sensitivity": [.5, .75, 1.25, 1.5]}
BLUE, RUST, GREY = "#163e63", "#b25118", "#7c8890"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(report: Path) -> tuple[dict, dict, list[dict]]:
    rows = json.loads((report / "all_results.json").read_text())
    baseline = json.loads((report / "baseline_reference.json").read_text())
    if len(rows) != 15 or len({r["case"] for r in rows}) != 15:
        raise ValueError("Expected exactly 15 unique new runs.")
    groups = {g: [] for g in GROUPS}
    records = []
    for row in rows:
        if row["round"] != 3 or not row["validation_passed"]:
            raise ValueError(f"Unexpected or unvalidated run: {row['case']}")
        group = row["factor"]
        row["x"] = row["max_cabinets"] if group == "max_cabinets" else row["multiplier"]
        path = Path(row["run_path"]) / "sensitivity" / "sensitivity_metrics.json"
        detail = json.loads(path.read_text())["details"]
        raw_cost, raw_fleet = detail["costs"]["components"], detail["fleet"]["components"]
        if {k for _, keys, _, _ in COSTS for k in keys} != set(raw_cost):
            raise ValueError(f"Unmapped cost component: {row['case']}")
        fleet_keys = {k for _, keys, _, _ in FLEET for k in keys}
        if set(raw_fleet) - fleet_keys:
            raise ValueError(f"Unmapped fleet component: {row['case']}")
        if ({"empty_flight", "empty_swap"} - set(raw_fleet)
                and abs(detail["fleet"]["empty_occupancy"]) > 1e-7):
            raise ValueError(f"Missing nonzero empty occupancy: {row['case']}")
        row["cost_components"] = [sum(raw_cost[k] for k in keys) / row["demand_per_hour"]
                                  for _, keys, _, _ in COSTS]
        row["fleet_components"] = [sum(raw_fleet.get(k, 0.) for k in keys)
                                   for _, keys, _, _ in FLEET]
        cost_error = sum(row["cost_components"]) - row["cost_per_order"]
        fleet_error = sum(row["fleet_components"]) - row["fleet"]
        if max(abs(cost_error), abs(fleet_error)) > 1e-6:
            raise ValueError(f"Component mismatch: {row['case']}")
        if min(row["cost_components"] + row["fleet_components"]) < -1e-7:
            raise ValueError(f"Negative chart component: {row['case']}")
        for source, target in (("mean_delivery", "mean_delivery_minutes"),
                               ("direct_same_served_OD", "direct_same_served_OD_minutes")):
            if abs(detail["times"][source] - row[target]) > 1e-7:
                raise ValueError(f"Delivery-time mismatch: {row['case']}")
        if row["saving_upper_bound_pct"] < row["saving_pct"] - 1e-7:
            raise ValueError(f"Inverted savings bound: {row['case']}")
        record = {key: row[key] for key in (
            "case", "round", "factor", "multiplier", "x", "max_cabinets",
            "walk_distance_coefficient", "demand_per_hour", "gap_pct", "above_gap_target",
            "saving_pct", "saving_upper_bound_pct", "drone_served_share_pct", "fleet",
            "opened_bases", "opened_lockers", "opened_swaps", "cost_per_order",
            "mean_delivery_minutes", "direct_same_served_OD_minutes")}
        record.update(source=str(path), source_sha256=_sha256(path),
                      cost_sum_error=cost_error, fleet_sum_error=fleet_error,
                      fleet_slack_in_idle=raw_fleet["fleet_slack"],
                      empty_occupancy=detail["fleet"]["empty_occupancy"])
        record.update({"cost_" + name.lower().replace(" ", "_"): value
                       for (name, _, _, _), value in zip(COSTS, row["cost_components"])})
        record.update({"fleet_" + name.lower().replace(" ", "_"): value
                       for (name, _, _, _), value in zip(FLEET, row["fleet_components"])})
        records.append(record)
        groups[group].append(row)
    for group, points in groups.items():
        points.sort(key=lambda r: r["x"])
        if [r["x"] for r in points] != EXPECTED[group]:
            raise ValueError(f"Unexpected levels for {group}")
    return groups, baseline, records


def _axis(ax, group: str, rows: list[dict], categorical: bool = False) -> list[float]:
    x = list(range(len(rows))) if categorical else [r["x"] for r in rows]
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r['x']:g}" for r in rows])
    margin = .55 if categorical else (x[-1] - x[0]) * .14
    ax.set_xlim(x[0] - margin, x[-1] + margin)
    ax.set_xlabel("Maximum cabinets/site" if group == "max_cabinets" else "Multiplier")
    ax.grid(axis="y")
    ax.tick_params(length=3, width=.6, pad=4)
    return x


def _line(ax, x: list[float], rows: list[dict], key: str, *, color: str = BLUE,
          marker: str = "o", style: str = "-") -> None:
    ax.plot(x, [r[key] for r in rows], color=color, lw=1.5, ls=style)
    for value, row in zip(x, rows):
        ax.plot(value, row[key], color=color, marker=marker, ls="none", ms=5.2,
                mfc="white" if row["above_gap_target"] else color, mew=1.1)


def performance(groups: dict, baseline: dict, output: Path) -> None:
    fig, axes = plt.subplots(4, 3, figsize=(8.8, 10.7))
    fig.subplots_adjust(left=.085, right=.985, top=.905, bottom=.105,
                        hspace=.80, wspace=.39)
    fig.suptitle("Round 3: cost, service share and fleet", x=.085, y=.986,
                 ha="left", fontsize=14)
    fig.legend(handles=[
        Line2D([], [], color=BLUE, marker="o", lw=1.4, label="New result"),
        Line2D([], [], color=BLUE, marker="o", mfc="white", lw=0, label="Gap > 0.5%"),
        Line2D([], [], color=GREY, lw=1, ls="--", label="Original baseline"),
    ], loc="center", bbox_to_anchor=(.53, .946), ncol=3, fontsize=8.8)
    metrics = (("saving_pct", "Saving (%)", (0, 7)),
               ("drone_served_share_pct", "Drone-served share (%)", (0, 45)),
               ("fleet", "Fleet (drones)", (0, 160)))
    for i, group in enumerate(GROUPS):
        rows = groups[group]
        for j, (key, label, limits) in enumerate(metrics):
            ax = axes[i, j]
            x = _axis(ax, group, rows)
            ax.set_title(f"({chr(97 + 3 * i + j)}) {TITLES[group]}", loc="left", fontsize=9.5)
            _line(ax, x, rows, key)
            ax.axhline(baseline[key], color=GREY, lw=.8, ls=(0, (4, 3)), zorder=0)
            if key == "saving_pct":
                cap = (x[-1] - x[0]) * .025
                ax.vlines(x, [r[key] for r in rows], [r["saving_upper_bound_pct"] for r in rows],
                          color=BLUE, lw=.9, alpha=.7)
                ax.hlines([r["saving_upper_bound_pct"] for r in rows],
                          [v - cap for v in x], [v + cap for v in x], color=BLUE, lw=.9)
            ax.set_ylim(*limits)
            ax.set_ylabel(label, fontsize=9)
    fig.text(.085, .025,
             "Only the 15 new runs are plotted. Vertical caps show upper bounds on savings.\n"
             "Walking-distance coefficient = 2.4 x multiplier; cutoff = 800 m. Fleet size is continuous.",
             fontsize=8.5, color="#58646d", linespacing=1.55)
    _save(fig, output, "performance_summary")


def composition(groups: dict, output: Path) -> None:
    fig, axes = plt.subplots(4, 2, figsize=(8.8, 11.6))
    fig.subplots_adjust(left=.085, right=.985, bottom=.09, top=.87,
                        hspace=.79, wspace=.30)
    fig.suptitle("Round 3: cost and fleet composition", x=.085, y=.988,
                 ha="left", fontsize=14)
    cost_handles = [Patch(facecolor=color, hatch=hatch, label=name)
                    for name, _, color, hatch in COSTS]
    fleet_handles = [Patch(facecolor=color, hatch=hatch, label=name)
                     for name, _, color, hatch in FLEET]
    fig.legend(handles=cost_handles, title="Cost components", loc="upper left",
               bbox_to_anchor=(.07, .966), ncol=3, fontsize=8, title_fontsize=9,
               columnspacing=1.3, handlelength=1.2)
    fig.legend(handles=fleet_handles, title="Fleet components", loc="upper right",
               bbox_to_anchor=(.985, .966), ncol=3, fontsize=8, title_fontsize=9,
               columnspacing=1.3, handlelength=1.2)
    for i, group in enumerate(GROUPS):
        rows = groups[group]
        for j, (specs, key) in enumerate(((COSTS, "cost_components"), (FLEET, "fleet_components"))):
            ax = axes[i, j]
            x = _axis(ax, group, rows, categorical=True)
            ax.set_title(f"({chr(97 + 2 * i + j)}) {TITLES[group]}", loc="left", fontsize=10)
            bottom = np.zeros(len(rows))
            for k, (_, _, color, hatch) in enumerate(specs):
                values = np.array([r[key][k] for r in rows])
                ax.bar(x, values, bottom=bottom, width=.62, facecolor=color,
                       edgecolor="white", linewidth=.4, hatch=hatch, zorder=2)
                bottom += values
            for x_i, total, row in zip(x, bottom, rows):
                label = f"{total:.2f}" if j == 0 else f"{total:.1f}"
                ax.annotate(label + ("*" if row["above_gap_target"] else ""), (x_i, total),
                            xytext=(0, 4), textcoords="offset points", ha="center", fontsize=8.5)
            ax.set_ylabel("Cost (CNY/order)" if j == 0 else "Drones")
            ax.set_ylim((0, 7) if j == 0 else (0, 160))
    fig.text(.085, .018,
             "Cost denominator: all orders in each case. Flight and swap include empty repositioning.\n"
             "Idle includes numerical fleet slack. * Gap > 0.5%. Only new runs are included.",
             fontsize=8.5, color="#58646d", linespacing=1.55)
    _save(fig, output, "cost_fleet_composition")


def delivery(groups: dict, output: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(8.8, 6.8), sharey=True)
    fig.subplots_adjust(left=.08, right=.985, top=.79, bottom=.20,
                        wspace=.23, hspace=.55)
    fig.suptitle("Round 3: mean delivery time", x=.08, y=.98,
                 ha="left", fontsize=14)
    series = (("mean_delivery_minutes", "Drone service", BLUE, "o", "-"),
              ("direct_same_served_OD_minutes", "Direct courier: same OD weights", RUST, "s", "--"))
    fig.legend(handles=[
        *[Line2D([], [], color=color, marker=marker, ls=style, lw=1.5, label=label)
          for _, label, color, marker, style in series],
        Line2D([], [], color=GREY, marker="o", mfc="white", lw=0, label="Gap > 0.5%")],
        loc="center", bbox_to_anchor=(.52, .9), ncol=2, fontsize=9)
    for i, (group, ax) in enumerate(zip(GROUPS, axes.flat)):
        rows = groups[group]
        x = _axis(ax, group, rows)
        ax.set_title(f"({chr(97 + i)}) {TITLES[group]}", loc="left", fontsize=10)
        for key, _, color, marker, style in series:
            _line(ax, x, rows, key, color=color, marker=marker, style=style)
        ax.set_ylim(0, 27)
        ax.set_yticks([0, 5, 10, 15, 20, 25])
        if i % 2 == 0:
            ax.set_ylabel("Delivery time (minutes)")
    fig.text(.08, .045,
             "Both means use each case's drone-served OD flow weights and start at merchant pickup.\n"
             "Drone time includes the common final home-delivery leg and excludes the return flight.\n"
             "Only the 15 new runs are included; no baseline point is repeated.",
             fontsize=8.5, color="#58646d", linespacing=1.55)
    _save(fig, output, "delivery_times")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    report = parser.parse_args().report.resolve()
    output = report / "figures"
    output.mkdir(exist_ok=True)
    groups, baseline, records = prepare(report)
    _style()
    performance(groups, baseline, output)
    composition(groups, output)
    delivery(groups, output)
    with (output / "figure_source_data.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    (output / "figure_validation.json").write_text(json.dumps({
        "source": str(report / "all_results.json"),
        "source_sha256": _sha256(report / "all_results.json"),
        "baseline_reference_sha256": _sha256(report / "baseline_reference.json"),
        "unique_new_experiments": len(records), "baseline_plotted_as_point": False,
        "maximum_cost_sum_error": max(abs(r["cost_sum_error"]) for r in records),
        "maximum_fleet_sum_error": max(abs(r["fleet_sum_error"]) for r in records),
        "maximum_absolute_fleet_slack": max(abs(r["fleet_slack_in_idle"]) for r in records),
        "validated": True,
    }, indent=2) + "\n")
    print(f"Saved three PNG/PDF figures and source data to {output}")


if __name__ == "__main__":
    main()
