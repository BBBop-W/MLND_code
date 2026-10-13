"""Draw demand and speed sensitivity curves, including supplemental runs.

Run with ``python -m src.reporting.demand_speed_figures``. The curves are
shape-preserving interpolants; all observations remain visible as markers.
"""

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
from scipy.interpolate import PchipInterpolator

from .sensitivity_figures import BLUE, TEAL, _style


ROOT = Path(__file__).resolve().parents[2]
BATCHES = (
    "h20_sensitivity_20260920",
    "sensitivity_round2_20260920",
    "sensitivity_round3_20260920",
)
LEVELS = {
    "demand": [.2, .3, .4, .5, .75, 1., 1.25, 1.5],
    "speed": [.5, .6, .7, .75, .8, .9, 1., 1.25, 1.5],
}
METRICS = (
    ("saving_pct", "Cost savings", "Savings vs. all-courier delivery (%)", (0, 9), [0, 2, 4, 6, 8]),
    ("drone_served_share_pct", "Drone-served share", "Share of all orders (%)", (0, 42), [0, 10, 20, 30, 40]),
    ("fleet", "Fleet size", "Number of drones", (0, 140), [0, 35, 70, 105, 140]),
    ("mean_delivery_minutes", "Mean delivery time", "Minutes from merchant pickup", (11.5, 17.5), [12, 13, 14, 15, 16, 17]),
)


def read_results() -> dict[str, list[dict]]:
    """Read saved, validated solutions; share the same baseline in both groups."""
    observations = {}
    for batch in BATCHES:
        folder = ROOT / "outputs/experiments" / batch / "batch"
        status = json.loads((folder / "status.json").read_text())
        assert status["status"] == "finished", batch
        for job in status["jobs"]:
            factor = job["factor"]
            if factor not in ("baseline", "demand", "speed"):
                continue
            assert job["state"] == "finished" and job["passed"], job["id"]
            assert job["postprocess_status"] == "complete", job["id"]
            run = Path(job["output"])
            source = run / "sensitivity/sensitivity_metrics.json"
            payload = json.loads(source.read_text())
            result_file = run / "result.json"
            result = json.loads(result_file.read_text())
            row = dict(payload["summary"])
            assert row["validation_passed"], job["id"]
            assert (row["OD"], row["candidate_lockers"], row["candidate_bases"],
                    row["candidate_swaps"]) == (400, 20, 4, 8)
            assert np.isclose(row["total_cost_per_hour"], result["objective"], atol=1e-5)
            for metric, *_ in METRICS:
                assert np.isfinite(row[metric]), (job["id"], metric)
            row.update(
                multiplier=float(job["multiplier"]),
                speed_kmh=payload["details"]["parameters"]["speed"],
                battery_endurance_minutes=60 * payload["details"]["parameters"]["battery_time"],
                gap_pct=100 * result["gap"],
                saving_upper_bound_pct=100 * (1 - result["bound"] / row["all_courier_cost_per_hour"]),
                shared_baseline=factor == "baseline",
                source=str(source.relative_to(ROOT)),
                source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                result_source=str(result_file.relative_to(ROOT)),
                result_sha256=hashlib.sha256(result_file.read_bytes()).hexdigest(),
            )
            assert row["saving_upper_bound_pct"] >= row["saving_pct"] - 1e-6
            key = (factor, row["multiplier"])
            assert key not in observations, key
            observations[key] = row
    baseline = observations[("baseline", 1.)]
    groups = {}
    for factor, levels in LEVELS.items():
        groups[factor] = [dict(baseline if level == 1 else observations[(factor, level)],
                               plot_factor=factor) for level in levels]
    assert len(observations) == 16
    return groups


def smooth_values(rows: list[dict], metric: str) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray([row["multiplier"] for row in rows])
    y = np.asarray([row[metric] for row in rows])
    curve = PchipInterpolator(x, y, extrapolate=False)
    assert np.allclose(curve(x), y, rtol=0, atol=1e-10)
    for left, right, a, b in zip(x[:-1], x[1:], y[:-1], y[1:]):
        values = curve(np.linspace(left, right, 101))
        assert values.min() >= min(a, b) - 1e-9
        assert values.max() <= max(a, b) + 1e-9
    grid = np.unique(np.concatenate((np.linspace(x[0], x[-1], 600), x)))
    return grid, curve(grid)


def draw_group(factor: str, rows: list[dict], output: Path) -> None:
    color = BLUE if factor == "demand" else TEAL
    fig, axes = plt.subplots(2, 2, figsize=(11.4, 8.9))
    fig.subplots_adjust(left=.087, right=.974, bottom=.20, top=.86,
                        wspace=.29, hspace=.47)
    title = "Order demand" if factor == "demand" else "Drone flight speed"
    fig.suptitle(f"Sensitivity to {title.lower()}", x=.087, y=.974, ha="left",
                 fontsize=19, fontweight="medium")
    fig.text(.087, .930,
             f"{len(rows)} tested levels  |  Nanshan, region 2  |  400 OD pairs  |  Candidates: 20 lockers, 4 bases, 8 swaps",
             fontsize=10, color="#58646d")
    ticks = LEVELS["demand"] if factor == "demand" else [.5, .6, .7, .8, .9, 1., 1.25, 1.5]
    x = [r["multiplier"] for r in rows]
    for idx, (ax, (metric, panel, ylabel, ylim, yticks)) in enumerate(zip(axes.flat, METRICS)):
        xx, yy = smooth_values(rows, metric)
        ax.plot(xx, yy, color=color, lw=1.8, zorder=2)
        for row in rows:
            ax.plot(row["multiplier"], row[metric], linestyle="none",
                    marker="o", ms=5.7, mec=color, mfc=color,
                    mew=1.35, zorder=4)
        ax.set_title(f"({chr(97 + idx)}) {panel}", loc="left", pad=11, fontsize=12)
        ax.set_xlabel("Demand multiplier" if factor == "demand" else "Speed multiplier", labelpad=7)
        ax.set_ylabel(ylabel, labelpad=7)
        ax.set_xticks(ticks, [f"{tick:g}" for tick in ticks])
        if factor == "speed":
            ax.set_xticks([.75], minor=True)
        ax.set_xlim(x[0] - .045, x[-1] + .045)
        ax.set_ylim(*ylim)
        ax.set_yticks(yticks)
        ax.grid(axis="y")
        ax.tick_params(length=3, width=.6, pad=5)
    base_note = ("Baseline demand: 3,583.57 orders/hour. Geography and all 400 OD pairs are held fixed."
                 if factor == "demand" else
                 "Baseline speed: 72 km/h. Flight endurance is fixed at 8.33 min; reachable distance varies with speed.")
    fig.text(.087, .118, base_note, fontsize=9.2, color="#58646d")
    fig.text(.087, .079,
             "Lines: shape-preserving PCHIP interpolation. Markers: actual experiment results.\n"
             "Delivery time is weighted by drone-served orders, from merchant pickup through the modeled home-delivery leg.",
             fontsize=9.0, color="#58646d", linespacing=1.65, va="top")
    for extension in ("png", "svg"):
        fig.savefig(output / f"{factor}_sensitivity_smooth.{extension}",
                    dpi=240, bbox_inches="tight", pad_inches=.16)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "outputs/reports/demand_speed_smooth_20260922")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    groups = read_results()
    _style()
    rows = [row for group in groups.values() for row in group]
    with (args.output / "plot_data.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(key for row in rows for key in row)))
        writer.writeheader()
        writer.writerows(rows)
    (args.output / "plot_data.json").write_text(json.dumps(groups, indent=2) + "\n")
    for factor, points in groups.items():
        draw_group(factor, points, args.output)
    validation = {
        "unique_runs": 16, "plotted_points": 17,
        "levels": LEVELS, "all_sources_validated": True,
        "smoothing": "PCHIP, exact at observations, verified without overshoot between adjacent observations",
        "delivery_time": "Drone-order-weighted minutes from merchant pickup, with the common modeled home-delivery terminal leg",
        "markers": "Uniform filled circles for all observations; gap and baseline status are not encoded",
        "uncertainty": "Solver gaps and objective bounds retained in source data, not displayed on figures",
        "model": "Cross-base service allowed; same-base runs excluded",
        "above_gap_target": [{"case": r["case"], "gap_pct": r["gap_pct"]}
                             for r in rows if r["gap_pct"] > .5],
    }
    (args.output / "validation.json").write_text(json.dumps(validation, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "points": {k: len(v) for k, v in groups.items()}}))


if __name__ == "__main__":
    main()
