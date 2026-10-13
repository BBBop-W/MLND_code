"""Additional, source-reconciled figures for sensitivity rounds one and two."""

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

from .sensitivity_figures import GROUPS, TITLES, _save, _style, _validate


COSTS = (
    ("Direct courier", ("direct_rider",), "#98a3ad", ""),
    ("Feeder couriers", ("first_rider", "terminal_rider"), "#315d80", ""),
    ("Facilities", ("base_fixed", "locker_fixed", "swap_fixed"), "#4d9897", ""),
    ("Fleet", ("fleet",), "#ce9143", ""),
    ("Flights and hover", ("service_flight", "empty_flight", "aerial_wait"), "#9e667f", ""),
    ("Routes", ("link_fixed",), "#33404d", "///"),
)
FLEET = (
    ("Idle", ("idle", "fleet_slack"), "#98a3ad", ""),
    ("Flight", ("pre_pickup_flight", "loaded_flight", "return_flight", "empty_flight"), "#315d80", ""),
    ("Swap", ("pre_pickup_swap", "loaded_swap", "return_swap", "empty_swap"), "#4d9897", ""),
    ("Handling", ("pickup_handling", "delivery_handling"), "#ce9143", ""),
    ("Pickup wait", ("sync_wait",), "#9e667f", ""),
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(report: Path) -> tuple[dict, list[dict]]:
    """Validate every plotted component against the saved per-case audit."""
    source = report / "plot_data.json"
    groups = _validate(json.loads(source.read_text()))
    records = []
    loaded = {}
    for group, rows in groups.items():
        for row in rows:
            path = Path(row["run_path"]) / "sensitivity" / "sensitivity_metrics.json"
            if str(path) not in loaded:
                loaded[str(path)] = json.loads(path.read_text())["details"]
            detail = loaded[str(path)]
            raw_cost = detail["costs"]["components"]
            raw_fleet = detail["fleet"]["components"]
            # These dictionaries record all nonzero components; empty movement
            # components may be absent only when their audited occupancy is zero.
            missing = {"empty_flight", "empty_swap"} - set(raw_fleet)
            if missing and abs(detail["fleet"]["empty_occupancy"]) > 1e-7:
                raise ValueError(f"Missing nonzero empty component: {row['case']}")
            cost_keys = {key for _, keys, _, _ in COSTS for key in keys}
            fleet_keys = {key for _, keys, _, _ in FLEET for key in keys}
            if set(raw_cost) != cost_keys or set(raw_fleet) - fleet_keys:
                raise ValueError(f"Unmapped components: {row['case']}")
            rate = row["demand_per_hour"]
            cost = [sum(raw_cost[key] for key in keys) / rate
                    for _, keys, _, _ in COSTS]
            fleet = [sum(raw_fleet.get(key, 0.0) for key in keys)
                     for _, keys, _, _ in FLEET]
            cost_error = sum(cost) - row["cost_per_order"]
            fleet_error = sum(fleet) - row["fleet"]
            if abs(cost_error) > 1e-6 or abs(fleet_error) > 1e-6:
                raise ValueError(f"Component total mismatch: {row['case']}")
            if min(cost + fleet) < -1e-7:
                raise ValueError(f"Negative chart component: {row['case']}")
            times = detail["times"]
            for field, target in (("mean_delivery", "mean_delivery_minutes"),
                                  ("direct_same_served_OD", "direct_same_served_OD_minutes")):
                if abs(times[field] - row[target]) > 1e-7:
                    raise ValueError(f"Delivery-time mismatch: {row['case']}")
            row["cost_components"] = cost
            row["fleet_components"] = fleet
            record = {
                "group": group, "multiplier_or_H": row["x"], "case": row["case"],
                "round": row["round"], "shared_baseline": row["shared_baseline"],
                "gap_pct": row["gap_pct"], "source": str(path), "source_sha256": _sha256(path),
                "cost_per_order": row["cost_per_order"], "fleet": row["fleet"],
                "cost_sum_error": cost_error, "fleet_sum_error": fleet_error,
                "fleet_slack_in_idle": raw_fleet["fleet_slack"],
                "mean_delivery_minutes": row["mean_delivery_minutes"],
                "direct_same_served_OD_minutes": row["direct_same_served_OD_minutes"],
            }
            record.update({"cost_" + name.lower().replace(" ", "_"): value
                           for (name, _, _, _), value in zip(COSTS, cost)})
            record.update({"fleet_" + name.lower().replace(" ", "_"): value
                           for (name, _, _, _), value in zip(FLEET, fleet)})
            records.append(record)
    if len(loaded) != 25:
        raise ValueError("Expected 25 unique experiments.")
    return groups, records


def _layout(title: str):
    fig, axes = plt.subplots(2, 3, figsize=(11.8, 7.6), sharey=True)
    fig.subplots_adjust(left=.07, right=.985, bottom=.19, top=.82,
                        wspace=.19, hspace=.48)
    fig.suptitle(title, x=.07, y=.98, ha="left", fontsize=15, fontweight="medium")
    return fig, axes


def _panel(ax, index: int, group: str, rows: list[dict]) -> None:
    ax.set_title(f"({chr(97 + index)}) {TITLES[group]}", loc="left", pad=10)
    ax.set_xticks(np.arange(5))
    ax.set_xticklabels([f"{r['x']:g}" for r in rows])
    ax.set_xlim(-.55, 4.55)
    ax.set_xlabel("Candidate lockers, H" if group == "density" else "Multiplier")
    ax.grid(axis="y")
    ax.tick_params(length=3, width=.6, pad=5)


def composition(groups: dict, output: Path, *, fleet: bool) -> None:
    specs = FLEET if fleet else COSTS
    key = "fleet_components" if fleet else "cost_components"
    fig, axes = _layout("Fleet composition" if fleet else "Cost composition per order")
    handles = [Patch(facecolor=color, edgecolor="white", hatch=hatch, label=name)
               for name, _, color, hatch in specs]
    fig.legend(handles=handles, loc="center", bbox_to_anchor=(.5, .922),
               ncol=len(specs), handlelength=1.5, columnspacing=1.55, handletextpad=.55)
    for idx, (group, ax) in enumerate(zip(GROUPS, axes.flat)):
        rows = groups[group]
        _panel(ax, idx, group, rows)
        x = np.arange(5)
        bottom = np.zeros(5)
        for k, (_, _, color, hatch) in enumerate(specs):
            values = np.array([r[key][k] for r in rows])
            ax.bar(x, values, bottom=bottom, width=.66, facecolor=color,
                   edgecolor="white", linewidth=.45, hatch=hatch, zorder=2)
            bottom += values
        for x_i, total, row in zip(x, bottom, rows):
            label = f"{total:.1f}" if fleet else f"{total:.2f}"
            if row["above_gap_target"]:
                label += "*"
            ax.annotate(label, (x_i, total), xytext=(0, 4), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8.6,
                        weight="bold" if row["shared_baseline"] else "normal")
        ax.set_ylim((0, 185) if fleet else (0, 9.3))
        ax.set_yticks([0, 40, 80, 120, 160] if fleet else [0, 2, 4, 6, 8])
        if idx % 3 == 0:
            ax.set_ylabel("Drones" if fleet else "Cost (CNY/order)")
    if fleet:
        note = ("Flight and swap include service and empty repositioning. Handling includes pickup and drop-off.\n"
                "Idle includes numerical fleet slack (absolute value < 1.2e-8 drones). * Gap > 0.5%; bold total = shared baseline.")
    else:
        note = ("Each cost component is divided by total order demand. Feeder couriers include first and final delivery legs.\n"
                "Facilities include bases, lockers and swaps; flights include empty repositioning. * Gap > 0.5%; bold total = shared baseline.")
    fig.text(.07, .045, note, fontsize=8.6, color="#58646d", linespacing=1.6)
    _save(fig, output, "fleet_composition" if fleet else "cost_composition")


def delivery(groups: dict, output: Path) -> None:
    fig, axes = _layout("Mean delivery time for drone-served orders")
    series = (("mean_delivery_minutes", "Drone service", "#163e63", "o", "-"),
              ("direct_same_served_OD_minutes", "Direct courier: same OD weights", "#b25118", "s", "--"))
    handles = [Line2D([], [], color=color, marker=marker, lw=1.6, ls=style, label=label)
               for _, label, color, marker, style in series]
    handles += [Line2D([], [], color="#4a5055", marker="o", mfc="white", lw=0,
                       label="Gap > 0.5%"),
                Line2D([], [], color="#262d33", marker="D", lw=0,
                       label="Shared baseline")]
    fig.legend(handles=handles, loc="center", bbox_to_anchor=(.5, .922),
               ncol=4, columnspacing=1.4, handletextpad=.55)
    for idx, (group, ax) in enumerate(zip(GROUPS, axes.flat)):
        rows = groups[group]
        _panel(ax, idx, group, rows)
        for key, _, color, marker, style in series:
            ax.plot(np.arange(5), [r[key] for r in rows], color=color, lw=1.7, ls=style)
            for x_i, row in enumerate(rows):
                shared = row["shared_baseline"]
                ax.plot(x_i, row[key], color=color, ls="none", marker="D" if shared else marker,
                        mec="#262d33" if shared else color,
                        mfc="#262d33" if shared else "white" if row["above_gap_target"] else color,
                        ms=5.5, mew=1.1)
        ax.set_ylim(0, 27)
        ax.set_yticks([0, 5, 10, 15, 20, 25])
        if idx % 3 == 0:
            ax.set_ylabel("Delivery time (minutes)")
    fig.text(.07, .045,
             "Both means use each case's drone-served OD flow weights and start at merchant pickup.\n"
             "Drone time includes the common final home-delivery leg and excludes the return flight; the shared baseline is repeated.",
             fontsize=8.7, color="#58646d", linespacing=1.6)
    _save(fig, output, "delivery_times")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report = args.report.resolve()
    output = report / "figures"
    output.mkdir(exist_ok=True)
    groups, records = prepare(report)
    _style()
    composition(groups, output, fleet=False)
    composition(groups, output, fleet=True)
    delivery(groups, output)
    with (output / "composition_delivery_source_data.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    (output / "composition_delivery_validation.json").write_text(json.dumps({
        "source": str(report / "plot_data.json"),
        "source_sha256": _sha256(report / "plot_data.json"),
        "unique_experiments": 25, "plotted_rows_including_shared_baseline": 30,
        "maximum_cost_sum_error": max(abs(r["cost_sum_error"]) for r in records),
        "maximum_fleet_sum_error": max(abs(r["fleet_sum_error"]) for r in records),
        "maximum_absolute_fleet_slack": max(abs(r["fleet_slack_in_idle"]) for r in records),
        "validated": True,
    }, indent=2) + "\n")
    print(f"Saved three figures (PNG and PDF) and their source data to {output}")


if __name__ == "__main__":
    main()
