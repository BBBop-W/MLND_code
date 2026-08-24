"""Command-line entry point for OD preparation and the new MLND formulations."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mlnd-matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/mlnd-xdg-cache")

from src.config import (
    DEFAULT_NANSHAN_PATH,
    DEFAULT_TSL_PATH,
    REPO_ROOT,
    ChoiceConfig,
    DataConfig,
    ModelConfig,
    SolverConfig,
    to_dict,
)


VALID_MODES = {"MX", "HD", "LO"}


def _float_tuple(raw: str) -> tuple[float, ...]:
    values = tuple(float(value.strip()) for value in raw.split(",") if value.strip())
    if not values:
        raise argparse.ArgumentTypeError("expected a comma-separated list of numbers")
    return values


def _string_tuple(raw: str) -> tuple[str, ...]:
    return tuple(value.strip() for value in raw.split(",") if value.strip())


def _positive_int_or_all(raw: str) -> int | None:
    if raw.lower() == "all":
        return None
    value = int(raw)
    if value < 1:
        raise argparse.ArgumentTypeError("value must be positive or 'all'")
    return value


def _positive_float_or_all(raw: str) -> float | None:
    if raw.lower() == "all":
        return None
    value = float(raw)
    if value <= 0:
        raise argparse.ArgumentTypeError("value must be positive or 'all'")
    return value


def _add_data_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--source", choices=("tsl", "nanshan"), required=True)
    parser.add_argument("--raw-data", default=None)
    parser.add_argument("--instance-dir", default=None)
    parser.add_argument("--candidate-count", "--num-lockers", dest="candidate_count", type=int, default=35)
    parser.add_argument("--num-bases", type=int, default=5)
    parser.add_argument("--num-swaps", type=int, default=8)
    parser.add_argument("--candidate-hours", nargs=2, type=int, metavar=("START", "END"), default=(8, 24))
    parser.add_argument("--demand-hours", nargs=2, type=int, metavar=("START", "END"), default=(12, 13))
    parser.add_argument("--map-half-size-km", type=float, default=8.0)
    parser.add_argument("--od-grid-km", type=float, default=0.5)
    parser.add_argument("--site-grid-km", type=float, default=0.25)
    parser.add_argument("--dates", type=_string_tuple, default=())
    parser.add_argument("--seed", type=int, default=42)


def _add_solve_arguments(parser: argparse.ArgumentParser, include_instance: bool = True) -> None:
    defaults = ModelConfig()
    if include_instance:
        parser.add_argument("--instance-dir", required=True)
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--modes", type=_string_tuple, default=("MX",))
    parser.add_argument("--demand-scale", type=float, default=1.0)
    parser.add_argument("--rho-levels", type=_float_tuple, default=(0.50, 0.65, 0.80, 0.90, 0.95))
    parser.add_argument("--max-walk-km", type=_positive_float_or_all, default=1.5)
    parser.add_argument("--locker-asc", type=float, default=-1.0)
    parser.add_argument("--choice-time-coef", type=float, default=0.06)
    parser.add_argument("--choice-price-coef", type=float, default=0.35)
    parser.add_argument("--choice-walk-coef", type=float, default=1.2)
    parser.add_argument("--home-eta-minutes", type=float, default=30.0)
    parser.add_argument("--locker-eta-minutes", type=float, default=20.0)
    parser.add_argument("--locker-discount", type=float, default=1.0)
    parser.add_argument("--origin-locker-choices", type=_positive_int_or_all, default=3)
    parser.add_argument("--home-corridor-choices", type=_positive_int_or_all, default=4)
    parser.add_argument("--direct-range-km", type=float, default=6.0)
    parser.add_argument("--swap-leg-km", type=float, default=6.0)
    parser.add_argument(
        "--locker-service-rates",
        type=_float_tuple,
        default=defaults.locker_service_rates,
        help="comma-separated locker service-rate levels in orders/hour",
    )
    parser.add_argument(
        "--locker-capacity-costs",
        type=_float_tuple,
        default=defaults.locker_capacity_costs,
        help="comma-separated hourly costs corresponding to locker service-rate levels",
    )
    parser.add_argument("--drone-speed-kmph", type=float, default=defaults.drone_speed_kmph)
    parser.add_argument(
        "--swap-service-minutes",
        type=float,
        default=60.0 * defaults.swap_service_time_hours,
    )
    parser.add_argument(
        "--max-locker-time-minutes",
        type=float,
        default=60.0 * defaults.max_locker_time_hours,
    )
    parser.add_argument(
        "--max-virtual-wait-minutes",
        type=float,
        default=60.0 * defaults.max_virtual_wait_hours,
    )
    parser.add_argument("--base-cost-per-hour", type=float, default=defaults.base_cost_per_hour)
    parser.add_argument(
        "--locker-fixed-cost-per-hour",
        type=float,
        default=defaults.locker_fixed_cost_per_hour,
    )
    parser.add_argument("--swap-cost-per-hour", type=float, default=defaults.swap_cost_per_hour)
    parser.add_argument(
        "--corridor-cost-per-km-hour",
        type=float,
        default=defaults.corridor_cost_per_km_hour,
    )
    parser.add_argument(
        "--connection-cost-per-km-hour",
        type=float,
        default=defaults.connection_cost_per_km_hour,
    )
    parser.add_argument("--flight-cost-per-km", type=float, default=defaults.flight_cost_per_km)
    parser.add_argument("--drone-cost-per-hour", type=float, default=defaults.drone_cost_per_hour)
    parser.add_argument("--rider-base-cost", type=float, default=defaults.rider_base_cost)
    parser.add_argument("--rider-cost-per-km", type=float, default=defaults.rider_cost_per_km)
    parser.add_argument(
        "--road-distance-factor",
        type=float,
        default=defaults.road_distance_factor,
    )
    parser.add_argument("--locker-service-cost", type=float, default=defaults.locker_service_cost)
    parser.add_argument("--time-limit", type=float, default=300.0)
    parser.add_argument("--mip-gap", type=float, default=0.01)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument(
        "--network-start-seconds",
        type=float,
        default=SolverConfig().network_start_seconds,
        help=(
            "time within the total solve budget used to construct a feasible network start; "
            "the temporary seed constraint is removed before the reported solve"
        ),
    )
    static_group = parser.add_mutually_exclusive_group()
    static_group.add_argument(
        "--valid-inequalities",
        dest="use_valid_inequalities",
        action="store_true",
        help="enable the full static strengthening pool (disabled by default after ablation)",
    )
    static_group.add_argument(
        "--no-valid-inequalities",
        dest="use_valid_inequalities",
        action="store_false",
        help=argparse.SUPPRESS,
    )
    parser.set_defaults(use_valid_inequalities=False)
    parser.add_argument(
        "--no-user-cuts",
        action="store_true",
        help="disable root-node OD, queue, capacity-cover, and return-connectivity cuts",
    )
    parser.add_argument("--max-root-cut-rounds", type=int, default=3)
    parser.add_argument("--max-cuts-per-round", type=int, default=250)
    parser.add_argument("--cut-violation-tolerance", type=float, default=1e-6)
    parser.add_argument(
        "--no-local-branching",
        action="store_true",
        help="disable exact-conic local branching in the network-start phase",
    )
    parser.add_argument("--local-branching-radius", type=int, default=12)
    parser.add_argument(
        "--root-method",
        type=int,
        choices=(-1, 0, 1, 2, 3, 4, 5),
        default=2,
        help="Gurobi Method for the root relaxation; 2=barrier (default), -1=automatic",
    )
    if include_instance:
        parser.add_argument("--solver-seed", "--seed", dest="solver_seed", type=int, default=42)
    else:
        parser.add_argument("--solver-seed", type=int, default=42)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--write-lp", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare Meituan OD data and solve the OD-aware drone network model."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare", help="convert raw TSL/Nanshan data to the canonical schema")
    _add_data_arguments(prepare)
    solve = subparsers.add_parser("solve", help="solve one or more formulations on a canonical instance")
    _add_solve_arguments(solve)
    run = subparsers.add_parser("run", help="prepare raw data and solve in one command")
    _add_data_arguments(run)
    _add_solve_arguments(run, include_instance=False)
    return parser


def _data_config(args: argparse.Namespace) -> DataConfig:
    raw_path = str(
        Path(
            args.raw_data
            or (DEFAULT_TSL_PATH if args.source == "tsl" else DEFAULT_NANSHAN_PATH)
        ).expanduser().resolve()
    )
    normalized_dates = tuple(sorted(set(args.dates)))
    preparation_signature = {
        "source": args.source,
        "raw_path": raw_path,
        "candidate_count": args.candidate_count,
        "num_bases": args.num_bases,
        "num_swaps": args.num_swaps,
        "candidate_hours": tuple(args.candidate_hours),
        "demand_hours": tuple(args.demand_hours),
        "map_half_size_km": args.map_half_size_km,
        "od_grid_km": args.od_grid_km,
        "site_grid_km": args.site_grid_km,
        "dates": normalized_dates,
        "seed": args.seed,
    }
    default_name = (
        f"{args.source}_od_h{args.candidate_count:02d}_"
        f"{_short_hash(preparation_signature)}"
    )
    instance_dir = str(
        Path(args.instance_dir).expanduser().resolve()
        if args.instance_dir
        else REPO_ROOT / "data" / "instances" / default_name
    )
    return DataConfig(
        source=args.source,
        raw_path=raw_path,
        output_dir=instance_dir,
        candidate_count=args.candidate_count,
        num_bases=args.num_bases,
        num_swaps=args.num_swaps,
        candidate_hour_start=args.candidate_hours[0],
        candidate_hour_end=args.candidate_hours[1],
        demand_hour_start=args.demand_hours[0],
        demand_hour_end=args.demand_hours[1],
        map_half_size_km=args.map_half_size_km,
        od_grid_km=args.od_grid_km,
        site_grid_km=args.site_grid_km,
        dates=normalized_dates,
        seed=args.seed,
    )


def _model_config(args: argparse.Namespace) -> ModelConfig:
    choice = ChoiceConfig(
        locker_asc=args.locker_asc,
        delivery_time_coefficient=args.choice_time_coef,
        price_coefficient=args.choice_price_coef,
        walk_distance_coefficient=args.choice_walk_coef,
        home_eta_minutes=args.home_eta_minutes,
        locker_eta_minutes=args.locker_eta_minutes,
        locker_discount_cny=args.locker_discount,
        max_walk_km=args.max_walk_km,
    )
    return ModelConfig(
        choice=choice,
        availability_levels=tuple(args.rho_levels),
        locker_service_rates=tuple(args.locker_service_rates),
        locker_capacity_costs=tuple(args.locker_capacity_costs),
        max_direct_range_km=args.direct_range_km,
        max_swap_leg_km=args.swap_leg_km,
        drone_speed_kmph=args.drone_speed_kmph,
        swap_service_time_hours=args.swap_service_minutes / 60.0,
        max_locker_time_hours=args.max_locker_time_minutes / 60.0,
        max_virtual_wait_hours=args.max_virtual_wait_minutes / 60.0,
        base_cost_per_hour=args.base_cost_per_hour,
        locker_fixed_cost_per_hour=args.locker_fixed_cost_per_hour,
        swap_cost_per_hour=args.swap_cost_per_hour,
        corridor_cost_per_km_hour=args.corridor_cost_per_km_hour,
        connection_cost_per_km_hour=args.connection_cost_per_km_hour,
        flight_cost_per_km=args.flight_cost_per_km,
        drone_cost_per_hour=args.drone_cost_per_hour,
        rider_base_cost=args.rider_base_cost,
        rider_cost_per_km=args.rider_cost_per_km,
        road_distance_factor=args.road_distance_factor,
        locker_service_cost=args.locker_service_cost,
        origin_locker_choices=args.origin_locker_choices,
        home_corridor_choices=args.home_corridor_choices,
    )


def _solve(args: argparse.Namespace, instance_dir: str) -> list[dict[str, object]]:
    from src.data_processor.loader import DataLoader
    from src.model.gurobi_model import MLNDModel
    from src.utils.output_processor import ResultProcessor

    modes = tuple(dict.fromkeys(raw_mode.upper() for raw_mode in args.modes))
    invalid_modes = sorted(set(modes) - VALID_MODES)
    if not modes or invalid_modes:
        raise ValueError(
            "modes must contain one or more of MX, HD, LO"
            + (f"; invalid={','.join(invalid_modes)}" if invalid_modes else "")
        )
    if not math.isfinite(args.demand_scale) or args.demand_scale <= 0:
        raise ValueError("demand_scale must be positive and finite")
    model_config = _model_config(args)
    model_config.validate()
    solver_config = SolverConfig(
        time_limit=args.time_limit,
        mip_gap=args.mip_gap,
        threads=args.threads,
        output_flag=0 if args.quiet else 1,
        seed=args.solver_seed,
        network_start_seconds=args.network_start_seconds,
        use_valid_inequalities=args.use_valid_inequalities,
        use_user_cuts=not args.no_user_cuts,
        max_root_cut_rounds=args.max_root_cut_rounds,
        max_cuts_per_round=args.max_cuts_per_round,
        cut_violation_tolerance=args.cut_violation_tolerance,
        use_local_branching=not args.no_local_branching,
        local_branching_radius=args.local_branching_radius,
        root_relaxation_method=args.root_method,
    )
    solver_config.validate()
    instance_path = Path(instance_dir).expanduser().resolve()
    with open(instance_path / "metadata.json", encoding="utf-8") as handle:
        instance_metadata = json.load(handle)
    canonical_hashes = {
        name: _sha256_file(instance_path / name)
        for name in ("metadata.json", "sites.csv", "od_nodes.csv", "od_demand.csv", "periods.csv")
        if (instance_path / name).exists()
    }
    run_signature = {
        "instance_dir": str(instance_path),
        "canonical_file_sha256": canonical_hashes,
        "demand_scale": args.demand_scale,
        "modes": modes,
        "model": to_dict(model_config),
        "solver": to_dict(solver_config),
    }
    default_results = REPO_ROOT / "results" / f"{instance_path.name}_{_short_hash(run_signature)}"
    results_root = Path(args.results_dir).expanduser().resolve() if args.results_dir else default_results
    results_root.mkdir(parents=True, exist_ok=True)
    with open(results_root / "run_config.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "instance": {
                    "directory": str(instance_path),
                    "metadata": instance_metadata,
                    "canonical_file_sha256": canonical_hashes,
                },
                "demand_scale": args.demand_scale,
                "modes": list(modes),
                "screening_is_exact": _screening_is_exact(model_config),
                "model": to_dict(model_config),
                "solver": to_dict(solver_config),
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )

    summaries: list[dict[str, object]] = []
    for mode in modes:
        instance = DataLoader().load_instance_dir(
            str(instance_path), model_config=model_config, demand_scale=args.demand_scale
        )
        model = MLNDModel(instance, mode=mode).prepare_algorithm(solver_config)
        mode_dir = results_root / mode
        mode_dir.mkdir(parents=True, exist_ok=True)
        if args.write_lp:
            model.write_model(str(mode_dir / "model.lp"))
        if not model.solve(solver_config):
            raise RuntimeError(f"{mode} finished without a feasible incumbent (status={model.model.Status})")
        processor = ResultProcessor(model, instance)
        processor.print_summary()
        summary = processor.write_all(str(mode_dir))
        summaries.append(summary)

    _write_mode_comparison(results_root, summaries)
    print(f"Results: {results_root.resolve()}")
    return summaries


def _write_mode_comparison(output_dir: Path, summaries: list[dict[str, object]]) -> None:
    if not summaries:
        return
    fields = [
        "mode",
        "status_name",
        "objective",
        "courier_only_baseline",
        "cost_saving",
        "cost_saving_percent",
        "drone_served_rate",
        "fleet_size",
        "network_is_nontrivial",
        "courier_only_incumbent",
        "open_bases",
        "open_lockers",
        "open_swap_stations",
        "selected_rho_state_count",
        "selected_rho_by_idle_state",
        "runtime_seconds",
        "mip_gap",
        "requested_mip_gap",
        "screening_is_exact",
    ]
    with open(output_dir / "mode_comparison.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for summary in summaries:
            row = {}
            for field in fields:
                value = summary.get(field)
                row[field] = (
                    json.dumps(value, ensure_ascii=False, sort_keys=True)
                    if isinstance(value, (dict, list))
                    else value
                )
            writer.writerow(row)

    import matplotlib.pyplot as plt

    labels = ["Courier only"] + [str(summary["mode"]) for summary in summaries]
    baseline = float(summaries[0]["courier_only_baseline"])
    costs = [baseline] + [float(summary["objective"]) for summary in summaries]
    savings = [baseline - float(summary["objective"]) for summary in summaries]
    palette = ["#93c5fd", "#3b82f6", "#1d4ed8"][: len(summaries)]
    fig, (cost_ax, saving_ax) = plt.subplots(
        1,
        2,
        figsize=(max(11.8, 2.1 * len(labels) + 7.0), 6.1),
        gridspec_kw={"width_ratios": (1.05, 1.0)},
    )
    bars = cost_ax.bar(
        labels,
        costs,
        color=["#9ca3af"] + palette,
        edgecolor="#334155",
        linewidth=0.7,
    )
    cost_ax.bar_label(bars, labels=[f"{value:,.2f}" for value in costs], padding=3, fontsize=8.5)
    cost_ax.set_ylabel("Modeled fulfillment cost (CNY/hour)")
    cost_ax.set_ylim(bottom=0)
    cost_ax.grid(axis="y", color="#e2e8f0", linewidth=0.7)
    cost_ax.set_axisbelow(True)
    cost_ax.spines[["top", "right"]].set_visible(False)
    cost_ax.set_title("Absolute incumbent cost", loc="left", fontsize=11, fontweight="semibold")

    saving_bars = saving_ax.bar(
        labels[1:],
        savings,
        color=palette,
        edgecolor="#334155",
        linewidth=0.7,
    )
    saving_ax.axhline(0.0, color="#64748b", linewidth=0.9)
    saving_ax.set_ylabel("Saving versus all-courier (CNY/hour)")
    saving_ax.grid(axis="y", color="#e2e8f0", linewidth=0.7)
    saving_ax.set_axisbelow(True)
    saving_ax.spines[["top", "right"]].set_visible(False)
    saving_ax.set_title(
        "Difference view (resolves near-equal cost bars)",
        loc="left",
        fontsize=11,
        fontweight="semibold",
    )
    saving_span = max((abs(value) for value in savings), default=1.0)
    saving_ax.set_ylim(
        min(0.0, min(savings, default=0.0) - 0.28 * saving_span),
        max(0.0, max(savings, default=0.0) + 0.32 * saving_span),
    )
    for index, summary in enumerate(summaries, start=1):
        gap = summary.get("mip_gap")
        gap_text = "gap N/A" if gap is None else f"gap {100.0 * float(gap):.1f}%"
        open_text = (
            f"open L/H/E={summary.get('open_bases', 0)}/"
            f"{summary.get('open_lockers', 0)}/{summary.get('open_swap_stations', 0)}"
        )
        rho_counts = summary.get("discrete_availability", {}).get(
            "selected_state_count_by_rho", {}
        )
        rho_text = "rho none" if not rho_counts else "rho " + ", ".join(
            f"{rho}:{count}" for rho, count in sorted(rho_counts.items())
        )
        value = savings[index - 1]
        percent = 100.0 * value / baseline if baseline else 0.0
        saving_ax.annotate(
            f"{value:,.2f} ({percent:.3f}%)\n{gap_text}\n{open_text}\n{rho_text}",
            (saving_bars[index - 1].get_x() + saving_bars[index - 1].get_width() / 2.0, value),
            xytext=(0, 6 if value >= 0 else -6),
            textcoords="offset points",
            ha="center",
            va="bottom" if value >= 0 else "top",
            fontsize=8,
            color="#334155",
        )
    fig.suptitle(
        "Incumbent fulfillment cost and savings by formulation",
        x=0.06,
        y=0.985,
        ha="left",
        fontweight="semibold",
    )
    fig.text(
        0.06,
        0.935,
        "Annotations report actual MIP gap, open L/H/E, and selected discrete rho; savings use the all-courier counterfactual.",
        color="#475569",
        fontsize=9,
        ha="left",
    )
    if all(bool(summary.get("courier_only_incumbent")) for summary in summaries):
        fig.text(
            0.5,
            0.018,
            "All formulation bars are courier-only incumbents (zero realized drone service).\n"
            "Equal bars do not establish equal optimized network costs.",
            color="#9a3412",
            fontsize=8.5,
            ha="center",
        )
    fig.tight_layout(rect=(0.03, 0.07, 0.99, 0.90), w_pad=3.0)
    fig.savefig(output_dir / "mode_cost_comparison.png", dpi=180)
    fig.savefig(output_dir / "mode_cost_comparison.svg")
    plt.close(fig)


def _screening_is_exact(config: ModelConfig) -> bool:
    return (
        config.choice.max_walk_km is None
        and config.origin_locker_choices is None
        and config.home_corridor_choices is None
    )


def _short_hash(payload: object) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:10]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "prepare":
        from src.data_processor.generator import DataGenerator

        DataGenerator().generate(_data_config(args))
        return
    if args.command == "run":
        from src.data_processor.generator import DataGenerator

        instance_dir = DataGenerator().generate(_data_config(args))
        _solve(args, instance_dir)
        return
    _solve(args, args.instance_dir)


if __name__ == "__main__":
    main()
