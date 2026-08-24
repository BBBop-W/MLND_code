"""Collect and visualize OD-aware parameter-sweep results.

Each scenario directory is expected to contain ``run_config.json`` and
``MX/summary.json``.  The command-line entry point writes a consolidated CSV
and a two-panel savings/service figure into the supplied results root.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")


_COST_FIELDS = (
    "base_cost_per_hour",
    "locker_fixed_cost_per_hour",
    "swap_cost_per_hour",
    "corridor_cost_per_km_hour",
    "connection_cost_per_km_hour",
    "flight_cost_per_km",
    "drone_cost_per_hour",
    "rider_base_cost",
    "rider_cost_per_km",
    "road_distance_factor",
    "locker_service_cost",
)

_CSV_FIELDS = (
    "scenario",
    "data_source",
    "instance_dir",
    "candidate_lockers",
    "demand_scale",
    "availability_levels",
    "locker_service_rates",
    "locker_capacity_costs",
    *_COST_FIELDS,
    "solver_time_limit_seconds",
    "solver_target_mip_gap",
    "solver_threads",
    "solver_seed",
    "status",
    "objective_cny_per_hour",
    "best_bound_cny_per_hour",
    "mip_gap_fraction",
    "mip_gap_percent",
    "runtime_seconds",
    "courier_only_baseline_cny_per_hour",
    "cost_saving_cny_per_hour",
    "cost_saving_percent",
    "drone_offered_per_hour",
    "drone_served_per_hour",
    "drone_served_rate",
    "locker_offered_per_hour",
    "locker_served_per_hour",
    "home_offered_per_hour",
    "home_served_per_hour",
    "open_bases",
    "open_lockers",
    "open_swap_stations",
    "active_virtual_states",
    "selected_rho_by_facility",
    "selected_rho_by_idle_state",
    "network_is_nontrivial",
    "run_config_path",
    "summary_path",
)

_JSON_FIELDS = {
    "availability_levels",
    "locker_service_rates",
    "locker_capacity_costs",
    "selected_rho_by_facility",
    "selected_rho_by_idle_state",
}


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _finite_number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _first_number(*values: Any) -> float | None:
    for value in values:
        number = _finite_number(value)
        if number is not None:
            return number
    return None


def _integer(value: Any) -> int | None:
    number = _finite_number(value)
    return int(round(number)) if number is not None else None


def _load_json_object(path: Path) -> Mapping[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError("top-level JSON value is not an object")
    return payload


def _scenario_directories(root: Path) -> list[Path]:
    candidates = {path.parent for path in root.rglob("run_config.json")}
    candidates.update(
        path.parent.parent
        for path in root.rglob("summary.json")
        if path.parent.name == "MX"
    )
    return sorted(
        candidates,
        key=lambda path: path.relative_to(root).as_posix().casefold(),
    )


def _numeric_mapping(value: Any) -> dict[str, float]:
    result: dict[str, float] = {}
    for key, raw_value in _mapping(value).items():
        number = _finite_number(raw_value)
        if number is not None:
            result[str(key)] = number
    return dict(sorted(result.items()))


def _rho_mappings(summary: Mapping[str, Any]) -> tuple[dict[str, float], dict[str, float]]:
    discrete = _mapping(summary.get("discrete_availability"))
    idle = _numeric_mapping(
        summary.get("selected_rho_by_idle_state")
        or discrete.get("selected_rho_by_idle_state")
    )
    facility: dict[str, float] = {}
    facility.update(_numeric_mapping(discrete.get("selected_base_rho")))
    facility.update(_numeric_mapping(discrete.get("selected_virtual_rho_by_locker")))
    if not facility:
        facility.update(idle)
    return dict(sorted(facility.items())), idle


def _scenario_name(root: Path, scenario_dir: Path) -> str:
    relative = scenario_dir.relative_to(root).as_posix()
    return root.name if relative == "." else relative


def _collect_scenario(
    root: Path,
    scenario_dir: Path,
    run_config: Mapping[str, Any],
    summary: Mapping[str, Any],
) -> dict[str, object]:
    model = _mapping(run_config.get("model"))
    solver_config = _mapping(run_config.get("solver"))
    solver_result = _mapping(summary.get("solver"))
    instance_config = _mapping(run_config.get("instance"))
    metadata = _mapping(instance_config.get("metadata"))
    data_config = _mapping(metadata.get("data_config"))
    summary_instance = _mapping(summary.get("instance"))
    facilities = _mapping(summary.get("facilities"))
    open_facilities = _mapping(facilities.get("open"))
    funnel = _mapping(summary.get("demand_funnel_orders_per_hour"))
    baseline_block = _mapping(summary.get("all_rider_baseline"))
    discrete = _mapping(summary.get("discrete_availability"))

    objective = _first_number(
        summary.get("objective"),
        solver_result.get("objective_cost_per_hour"),
        _mapping(summary.get("costs_cny_per_hour")).get("optimized_total"),
    )
    if objective is None:
        raise ValueError("summary has no finite objective")

    bound = _first_number(summary.get("objective_bound"), solver_result.get("objective_bound"))
    gap = _first_number(summary.get("mip_gap"), solver_result.get("mip_gap"))
    runtime = _first_number(summary.get("runtime_seconds"), solver_result.get("runtime_seconds"))
    baseline = _first_number(
        summary.get("courier_only_baseline"),
        baseline_block.get("cost_per_hour"),
    )
    saving = _first_number(summary.get("cost_saving"), baseline_block.get("savings_per_hour"))
    if saving is None and baseline is not None:
        saving = baseline - objective
    saving_percent = _first_number(summary.get("cost_saving_percent"))
    if saving_percent is None and saving is not None and baseline not in (None, 0.0):
        saving_percent = 100.0 * saving / baseline

    offered = _first_number(funnel.get("total_drone_offered"))
    served = _first_number(funnel.get("total_drone_served"))
    total_demand = _first_number(funnel.get("total_demand"))
    served_rate = _first_number(
        summary.get("drone_served_rate"),
        funnel.get("drone_service_fraction_of_demand"),
    )
    if served_rate is None and served is not None and total_demand not in (None, 0.0):
        served_rate = served / total_demand

    rho_by_facility, rho_by_idle = _rho_mappings(summary)
    open_bases = _integer(summary.get("open_bases"))
    open_lockers = _integer(summary.get("open_lockers"))
    open_swaps = _integer(summary.get("open_swap_stations"))
    if open_bases is None:
        open_bases = _integer(open_facilities.get("bases"))
    if open_lockers is None:
        open_lockers = _integer(open_facilities.get("lockers"))
    if open_swaps is None:
        open_swaps = _integer(open_facilities.get("swap_stations"))

    row: dict[str, object] = {
        "scenario": _scenario_name(root, scenario_dir),
        "data_source": metadata.get("source", summary_instance.get("data_source", "")),
        "instance_dir": instance_config.get("directory", ""),
        "candidate_lockers": _integer(
            summary_instance.get("candidate_lockers", data_config.get("candidate_count"))
        ),
        "demand_scale": _first_number(run_config.get("demand_scale")),
        "availability_levels": model.get("availability_levels", []),
        "locker_service_rates": model.get("locker_service_rates", []),
        "locker_capacity_costs": model.get("locker_capacity_costs", []),
        "solver_time_limit_seconds": _first_number(solver_config.get("time_limit")),
        "solver_target_mip_gap": _first_number(solver_config.get("mip_gap")),
        "solver_threads": _integer(solver_config.get("threads")),
        "solver_seed": _integer(solver_config.get("seed")),
        "status": summary.get("status_name", solver_result.get("status_name", "")),
        "objective_cny_per_hour": objective,
        "best_bound_cny_per_hour": bound,
        "mip_gap_fraction": gap,
        "mip_gap_percent": 100.0 * gap if gap is not None else None,
        "runtime_seconds": runtime,
        "courier_only_baseline_cny_per_hour": baseline,
        "cost_saving_cny_per_hour": saving,
        "cost_saving_percent": saving_percent,
        "drone_offered_per_hour": offered,
        "drone_served_per_hour": served,
        "drone_served_rate": served_rate,
        "locker_offered_per_hour": _first_number(funnel.get("locker_offered")),
        "locker_served_per_hour": _first_number(funnel.get("locker_served")),
        "home_offered_per_hour": _first_number(funnel.get("home_offered")),
        "home_served_per_hour": _first_number(funnel.get("home_served")),
        "open_bases": open_bases,
        "open_lockers": open_lockers,
        "open_swap_stations": open_swaps,
        "active_virtual_states": _integer(
            summary.get("active_virtual_state_count", discrete.get("active_virtual_state_count"))
        ),
        "selected_rho_by_facility": rho_by_facility,
        "selected_rho_by_idle_state": rho_by_idle,
        "network_is_nontrivial": bool(
            summary.get("network_is_nontrivial", served is not None and served > 1e-7)
        ),
        "run_config_path": str(scenario_dir / "run_config.json"),
        "summary_path": str(scenario_dir / "MX" / "summary.json"),
    }
    for field in _COST_FIELDS:
        row[field] = _finite_number(model.get(field))
    return row


def collect_parameter_results(results_root: str | Path) -> list[dict[str, object]]:
    """Return valid parameter-sweep scenarios below ``results_root``.

    Direct and nested scenario directories are supported.  A malformed or
    incomplete scenario is reported on stderr and skipped without preventing
    other scenarios from being collected.
    """

    root = Path(results_root).expanduser().resolve()
    if not root.exists():
        raise ValueError(f"results root does not exist: {root}")
    if not root.is_dir():
        raise ValueError(f"results root is not a directory: {root}")

    rows: list[dict[str, object]] = []
    for scenario_dir in _scenario_directories(root):
        run_config_path = scenario_dir / "run_config.json"
        summary_path = scenario_dir / "MX" / "summary.json"
        scenario = _scenario_name(root, scenario_dir)
        missing = [
            str(path.relative_to(scenario_dir))
            for path in (run_config_path, summary_path)
            if not path.is_file()
        ]
        if missing:
            print(f"[skip] {scenario}: missing {', '.join(missing)}", file=sys.stderr)
            continue
        try:
            run_config = _load_json_object(run_config_path)
            summary = _load_json_object(summary_path)
            rows.append(_collect_scenario(root, scenario_dir, run_config, summary))
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
            print(f"[skip] {scenario}: {exc}", file=sys.stderr)
    return rows


def _write_csv(root: Path, rows: Sequence[Mapping[str, object]]) -> Path:
    output_path = root / "parameter_sweep_summary.csv"
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=_CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            payload = dict(row)
            for field in _JSON_FIELDS:
                payload[field] = json.dumps(
                    payload.get(field, {}),
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            writer.writerow(payload)
    return output_path


def _rho_lines(row: Mapping[str, object], chunk_size: int = 4) -> list[str]:
    rho = _numeric_mapping(row.get("selected_rho_by_facility"))
    counts = (
        f"open L/H/E={row.get('open_bases') or 0}/"
        f"{row.get('open_lockers') or 0}/{row.get('open_swap_stations') or 0}"
    )
    if not rho:
        return [counts, "rho: none"]
    entries = [f"{facility}={value:.2f}" for facility, value in rho.items()]
    chunks = [entries[index : index + chunk_size] for index in range(0, len(entries), chunk_size)]
    return [counts] + [
        ("rho: " if index == 0 else "     ") + ", ".join(chunk)
        for index, chunk in enumerate(chunks)
    ]


def _plot_results(root: Path, rows: Sequence[Mapping[str, object]]) -> tuple[Path, Path]:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    notes = [_rho_lines(row) for row in rows]
    positions: list[float] = []
    cursor = 0.0
    for lines in notes:
        height = max(1.25, 0.42 * len(lines) + 0.55)
        positions.append(cursor + height / 2.0)
        cursor += height

    figure_height = max(5.4, 2.2 + 0.52 * cursor)
    fig = plt.figure(figsize=(16.0, figure_height))
    grid = fig.add_gridspec(1, 3, width_ratios=(1.05, 1.0, 1.65), wspace=0.25)
    savings_ax = fig.add_subplot(grid[0, 0])
    served_ax = fig.add_subplot(grid[0, 1])
    notes_ax = fig.add_subplot(grid[0, 2])

    savings_raw = [_finite_number(row.get("cost_saving_percent")) for row in rows]
    savings = [value if value is not None else 0.0 for value in savings_raw]
    served_raw = [_finite_number(row.get("drone_served_rate")) for row in rows]
    served_percent = [100.0 * value if value is not None else 0.0 for value in served_raw]

    saving_colors = [
        "#2563eb" if value > 1e-9 else "#d97706" if value < -1e-9 else "#94a3b8"
        for value in savings
    ]
    savings_ax.barh(positions, savings, height=0.62, color=saving_colors, edgecolor="#334155")
    served_ax.barh(
        positions,
        served_percent,
        height=0.62,
        color="#0f766e",
        edgecolor="#334155",
    )

    labels = [str(row["scenario"]) for row in rows]
    savings_ax.set_yticks(positions, labels=labels)
    served_ax.set_yticks(positions, labels=[])
    bounds = (cursor, 0.0)
    savings_ax.set_ylim(*bounds)
    served_ax.set_ylim(*bounds)
    notes_ax.set_ylim(*bounds)

    minimum = min([0.0, *savings])
    maximum = max([0.0, *savings])
    span = max(1.0, maximum - minimum)
    savings_ax.set_xlim(minimum - 0.10 * span, maximum + 0.22 * span)
    savings_ax.axvline(0.0, color="#64748b", linewidth=0.9)
    for position, raw_value, value in zip(positions, savings_raw, savings):
        label = "N/A" if raw_value is None else f"{value:.2f}%"
        direction = 1.0 if value >= 0.0 else -1.0
        savings_ax.text(
            value + direction * 0.025 * span,
            position,
            label,
            ha="left" if direction > 0 else "right",
            va="center",
            fontsize=8,
        )

    served_limit = max(100.0, max(served_percent, default=0.0) * 1.15, 1.0)
    served_ax.set_xlim(0.0, served_limit)
    for position, raw_value, value in zip(positions, served_raw, served_percent):
        label = "N/A" if raw_value is None else f"{value:.2f}%"
        served_ax.text(
            value + 0.018 * served_limit,
            position,
            label,
            ha="left",
            va="center",
            fontsize=8,
        )

    savings_ax.set_xlabel("Cost saving vs. all-rider baseline (%)")
    served_ax.set_xlabel("Demand served by drone network (%)")
    savings_ax.set_title("Modeled cost saving", loc="left", fontweight="semibold")
    served_ax.set_title("Realized drone service", loc="left", fontweight="semibold")
    notes_ax.set_title("Facility decisions and discrete rho", loc="left", fontweight="semibold")
    notes_ax.set_xlim(0.0, 1.0)
    notes_ax.axis("off")
    for position, lines in zip(positions, notes):
        notes_ax.text(
            0.0,
            position,
            "\n".join(lines),
            ha="left",
            va="center",
            fontsize=7.8,
            color="#334155",
        )

    for axis in (savings_ax, served_ax):
        axis.grid(axis="x", color="#e2e8f0", linewidth=0.7)
        axis.set_axisbelow(True)
        axis.spines[["top", "right", "left"]].set_visible(False)
        axis.tick_params(axis="y", length=0)

    fig.suptitle(
        "Parameter sweep: savings and drone-network service",
        x=0.06,
        y=0.975,
        ha="left",
        fontweight="semibold",
    )
    fig.text(
        0.06,
        0.94,
        f"{root} | MX summaries; savings are relative to each scenario's all-rider baseline",
        ha="left",
        va="top",
        fontsize=8.5,
        color="#475569",
    )
    fig.subplots_adjust(left=0.13, right=0.985, top=0.88, bottom=0.10)

    png_path = root / "parameter_sweep_summary.png"
    svg_path = root / "parameter_sweep_summary.svg"
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, svg_path


def write_parameter_report(results_root: str | Path) -> list[dict[str, object]]:
    """Write the consolidated CSV and comparison figure for a sweep root."""

    root = Path(results_root).expanduser().resolve()
    rows = collect_parameter_results(root)
    if not rows:
        raise ValueError(
            f"no valid MX parameter scenarios found under {root}; expected each scenario "
            "to contain run_config.json and MX/summary.json"
        )
    csv_path = _write_csv(root, rows)
    png_path, svg_path = _plot_results(root, rows)
    print(f"Collected {len(rows)} valid scenario(s).")
    print(f"CSV: {csv_path}")
    print(f"Figure: {png_path}")
    print(f"Figure: {svg_path}")
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Summarize OD-aware MX parameter-sweep scenarios."
    )
    parser.add_argument(
        "results_root",
        metavar="ROOT",
        help="root containing scenario directories with run_config.json and MX/summary.json",
    )
    args = parser.parse_args(argv)
    try:
        write_parameter_report(args.results_root)
    except (OSError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
