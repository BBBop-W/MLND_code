"""Collect and visualize the Nanshan candidate-scale benchmark."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


def collect_scale_results(results_root: str | Path) -> list[dict[str, object]]:
    root = Path(results_root)
    rows: list[dict[str, object]] = []
    for run_dir in sorted(root.glob("H*"), key=lambda path: int(path.name[1:]), reverse=True):
        summary_path = run_dir / "MX" / "summary.json"
        if not summary_path.exists():
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        run_config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
        solver = summary["solver"]
        target_gap = float(run_config["solver"]["mip_gap"])
        time_limit = float(run_config["solver"]["time_limit"])
        gap = solver["mip_gap"]
        facilities = summary["facilities"]["open"]
        funnel = summary["demand_funnel_orders_per_hour"]
        rho_by_facility: dict[str, float] = {}
        facilities_path = run_dir / "MX" / "open_facilities.csv"
        if facilities_path.exists():
            with facilities_path.open(newline="", encoding="utf-8") as handle:
                for facility in csv.DictReader(handle):
                    if facility["rho"]:
                        rho_by_facility[facility["facility"]] = float(facility["rho"])
        runtime = float(solver["runtime_seconds"])
        rows.append(
            {
                "candidate_lockers": int(summary["instance"]["candidate_lockers"]),
                "status": solver["status_name"],
                "runtime_seconds": runtime,
                "configured_time_limit_seconds": time_limit,
                "runtime_over_limit_ratio": runtime / time_limit,
                "objective_cny_per_hour": float(solver["objective_cost_per_hour"]),
                "best_bound_cny_per_hour": solver["objective_bound"],
                "mip_gap": gap,
                "target_mip_gap": target_gap,
                "meets_target_gap": gap is not None and float(gap) <= target_gap + 1e-9,
                "open_bases": int(facilities["bases"]),
                "open_lockers": int(facilities["lockers"]),
                "open_swap_stations": int(facilities["swap_stations"]),
                "selected_rho_by_facility": rho_by_facility,
                "active_virtual_states": int(
                    summary["discrete_availability"]["active_virtual_state_count"]
                ),
                "drone_offered_per_hour": float(funnel["total_drone_offered"]),
                "drone_served_per_hour": float(funnel["total_drone_served"]),
                "network_is_nontrivial": float(funnel["total_drone_served"]) > 1e-7,
                "model_variables": int(summary["model_size"]["variables"]),
                "linear_constraints": int(summary["model_size"]["linear_constraints"]),
                "quadratic_constraints": int(summary["model_size"]["quadratic_constraints"]),
            }
        )
    return rows


def write_scale_report(results_root: str | Path) -> list[dict[str, object]]:
    root = Path(results_root)
    rows = collect_scale_results(root)
    fields = list(rows[0])
    with (root / "benchmark_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            payload = dict(row)
            payload["selected_rho_by_facility"] = json.dumps(
                payload["selected_rho_by_facility"], sort_keys=True
            )
            writer.writerow(payload)

    labels = [f"H{row['candidate_lockers']}" for row in rows]
    runtime = [float(row["runtime_seconds"]) for row in rows]
    limit = float(rows[0]["configured_time_limit_seconds"])
    solved_color = "#2563eb"
    unsolved_color = "#d97706"
    colors = [solved_color if row["meets_target_gap"] else unsolved_color for row in rows]
    fig, ax = plt.subplots(figsize=(max(8.6, 1.35 * len(rows) + 4.8), 6.4))
    bars = ax.bar(labels, runtime, color=colors, edgecolor="#334155", linewidth=0.8)
    for bar, row in zip(bars, rows):
        if not row["meets_target_gap"]:
            bar.set_hatch("//")
        gap = row["mip_gap"]
        gap_text = "gap N/A" if gap is None else f"gap {100.0 * float(gap):.1f}%"
        rho_map = row["selected_rho_by_facility"]
        rho_text = "rho none" if not rho_map else "rho " + ", ".join(
            f"{facility}={rho:.2f}" for facility, rho in sorted(rho_map.items())
        )
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(runtime) * 0.018,
            f"{bar.get_height():.1f}s\n{gap_text}\n"
            f"open L/H/E={row['open_bases']}/{row['open_lockers']}/{row['open_swap_stations']}\n"
            f"{rho_text}",
            ha="center",
            va="bottom",
            fontsize=8,
            color="#334155",
        )
    ax.axhline(limit, color="#475569", linestyle="--", linewidth=1.3)
    ax.text(
        len(rows) - 0.55,
        limit + max(runtime) * 0.015,
        f"configured limit = {limit:.0f}s",
        ha="right",
        va="bottom",
        fontsize=8.5,
        color="#475569",
    )
    ax.set_ylabel("Actual Gurobi runtime (seconds)")
    ax.set_xlabel("Number of candidate H lockers")
    fig.suptitle(
        "Nanshan MX candidate-scale benchmark",
        x=0.105,
        y=0.975,
        ha="left",
        fontweight="semibold",
    )
    fig.text(
        0.105,
        0.925,
        "Default screened formulation; target gap 1%; bars show actual runtime, including solver overrun",
        ha="left",
        color="#475569",
        fontsize=9,
    )
    ax.legend(
        handles=[
            Patch(facecolor=solved_color, edgecolor="#334155", label="Reached target gap"),
            Patch(
                facecolor=unsolved_color,
                edgecolor="#334155",
                hatch="//",
                label="Did not reach target gap",
            ),
        ],
        loc="upper right",
        frameon=False,
    )
    ax.set_ylim(0.0, max(runtime) * 1.25)
    ax.grid(axis="y", color="#e2e8f0", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    if all(not bool(row["network_is_nontrivial"]) for row in rows):
        fig.text(
            0.5,
            0.018,
            "Every incumbent shown is courier-only: zero opened facilities, no selected rho, and zero drone service.",
            ha="center",
            color="#9a3412",
            fontsize=8.5,
        )
    fig.tight_layout(rect=(0.0, 0.06, 1.0, 0.88))
    fig.savefig(root / "benchmark_runtime.png", dpi=220, bbox_inches="tight")
    fig.savefig(root / "benchmark_runtime.svg", bbox_inches="tight")
    plt.close(fig)
    return rows


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("results_root")
    write_scale_report(parser.parse_args().results_root)
