"""Command-line entry point for OD preparation and the new MLND formulations."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
import platform
from pathlib import Path
import subprocess
import sys

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
from src.model_contract import FORMULATION_ID, RESULT_SCHEMA_VERSION


MODE_RUN_ORDER = ("HD", "MX", "LO")
VALID_MODES = frozenset(MODE_RUN_ORDER)
CLI_DEFAULT_DISPLAY_ORIGIN_CHOICES = 8
CLI_DEFAULT_GENERAL_CORRIDOR_CHOICES = 10
FORMULATION_SOURCE = REPO_ROOT / "MLND" / "main.tex"
FORMULATION_IMPLEMENTATION_FILES = (
    REPO_ROOT / "main.py",
    REPO_ROOT / "src" / "model_contract.py",
    REPO_ROOT / "src" / "config.py",
    REPO_ROOT / "src" / "data_processor" / "loader.py",
    REPO_ROOT / "src" / "data_processor" / "instance.py",
    REPO_ROOT / "src" / "model" / "gurobi_model.py",
    REPO_ROOT / "src" / "model" / "primal_heuristic.py",
    REPO_ROOT / "src" / "model" / "valid_inequalities.py",
    REPO_ROOT / "src" / "utils" / "output_processor.py",
)


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


def _validate_instance_demand_scale(metadata: object, demand_scale: float) -> None:
    """Enforce an optional instance-level solve-time demand-scale contract."""

    if not isinstance(metadata, dict):
        raise ValueError("metadata.json must contain a JSON object")

    declarations: list[tuple[str, object]] = []
    if metadata.get("required_solve_demand_scale") is not None:
        declarations.append(
            (
                "required_solve_demand_scale",
                metadata["required_solve_demand_scale"],
            )
        )

    nested_scale = metadata.get("nested_scale")
    if nested_scale is not None:
        if not isinstance(nested_scale, dict):
            raise ValueError("metadata.json nested_scale must contain a JSON object")
        if nested_scale.get("required_solve_demand_scale") is not None:
            declarations.append(
                (
                    "nested_scale.required_solve_demand_scale",
                    nested_scale["required_solve_demand_scale"],
                )
            )

    if not declarations:
        return

    requirements: list[tuple[str, float]] = []
    for field, raw_value in declarations:
        if isinstance(raw_value, bool):
            raise ValueError(f"metadata.json {field} must be positive and finite")
        try:
            required = float(raw_value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"metadata.json {field} must be positive and finite"
            ) from exc
        if not math.isfinite(required) or required <= 0:
            raise ValueError(f"metadata.json {field} must be positive and finite")
        requirements.append((field, required))

    required = requirements[0][1]
    if any(
        not math.isclose(value, required, rel_tol=0.0, abs_tol=1e-12)
        for _field, value in requirements[1:]
    ):
        fields = ", ".join(field for field, _value in requirements)
        raise ValueError(
            f"metadata.json has conflicting demand-scale contracts: {fields}"
        )
    if not math.isclose(demand_scale, required, rel_tol=0.0, abs_tol=1e-12):
        fields = ", ".join(field for field, _value in requirements)
        raise ValueError(
            f"instance metadata requires demand_scale={required:g} via {fields}; "
            f"got {demand_scale:g}"
        )


def _add_data_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--source", choices=("tsl", "nanshan"), required=True)
    parser.add_argument("--raw-data", default=None)
    parser.add_argument("--instance-dir", default=None)
    parser.add_argument(
        "--candidate-count",
        "--num-lockers",
        dest="candidate_count",
        type=int,
        default=34,
    )
    parser.add_argument("--num-bases", type=int, default=5)
    parser.add_argument("--num-swaps", type=int, default=8)
    parser.add_argument(
        "--candidate-hours",
        nargs=2,
        type=int,
        metavar=("START", "END"),
        default=(8, 24),
    )
    parser.add_argument(
        "--demand-hours", nargs=2, type=int, metavar=("START", "END"), default=(12, 13)
    )
    parser.add_argument("--map-half-size-km", type=float, default=8.0)
    parser.add_argument("--od-grid-km", type=float, default=0.5)
    parser.add_argument("--site-grid-km", type=float, default=0.25)
    parser.add_argument("--dates", type=_string_tuple, default=())
    parser.add_argument("--seed", type=int, default=42)


def _add_solve_arguments(
    parser: argparse.ArgumentParser, include_instance: bool = True
) -> None:
    defaults = ModelConfig()
    if include_instance:
        parser.add_argument("--instance-dir", required=True)
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--modes", type=_string_tuple, default=("MX",))
    parser.add_argument("--demand-scale", type=float, default=1.0)
    parser.add_argument(
        "--rho-levels", type=_float_tuple, default=(0.50, 0.65, 0.80, 0.90, 0.95)
    )
    parser.add_argument("--max-walk-km", type=_positive_float_or_all, default=1.5)
    parser.add_argument("--locker-asc", type=float, default=-1.0)
    parser.add_argument("--choice-time-coef", type=float, default=0.06)
    parser.add_argument("--choice-price-coef", type=float, default=0.35)
    parser.add_argument("--choice-walk-coef", type=float, default=1.2)
    parser.add_argument("--home-eta-minutes", type=float, default=30.0)
    parser.add_argument("--locker-eta-minutes", type=float, default=20.0)
    parser.add_argument("--locker-discount", type=float, default=1.0)
    parser.add_argument(
        "--display-origin-choices",
        type=_positive_int_or_all,
        default=CLI_DEFAULT_DISPLAY_ORIGIN_CHOICES,
        help=(
            "top-k backend origins per displayed locker; CLI uses 8 as a "
            "large-instance screening preset, or pass 'all' for the paper index"
        ),
    )
    parser.add_argument(
        "--general-corridor-choices",
        type=_positive_int_or_all,
        default=CLI_DEFAULT_GENERAL_CORRIDOR_CHOICES,
        help=(
            "top-k general corridors per OD; CLI uses 10 as a large-instance "
            "screening preset, or pass 'all' for the paper index"
        ),
    )
    parser.add_argument(
        "--battery-range-km",
        type=float,
        default=defaults.battery_range_km,
        help=(
            "usable range of one battery in km; applies to a direct movement "
            "and to each leg of a one-swap movement"
        ),
    )
    parser.add_argument(
        "--locker-service-rates",
        type=_float_tuple,
        default=defaults.locker_service_rates,
        help="comma-separated locker service-rate levels in drone handlings/hour",
    )
    parser.add_argument(
        "--locker-capacity-costs",
        type=_float_tuple,
        default=defaults.locker_capacity_costs,
        help="comma-separated hourly costs corresponding to locker service-rate levels",
    )
    parser.add_argument(
        "--drone-speed-mps",
        type=float,
        default=defaults.drone_speed_mps,
        help="drone cruise speed in metres/second",
    )
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
    parser.add_argument(
        "--base-cost-per-hour", type=float, default=defaults.base_cost_per_hour
    )
    parser.add_argument(
        "--locker-fixed-cost-per-hour",
        type=float,
        default=defaults.locker_fixed_cost_per_hour,
    )
    parser.add_argument(
        "--swap-cost-per-hour", type=float, default=defaults.swap_cost_per_hour
    )
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
    parser.add_argument(
        "--flight-cost-per-km", type=float, default=defaults.flight_cost_per_km
    )
    parser.add_argument(
        "--drone-cost-per-hour", type=float, default=defaults.drone_cost_per_hour
    )
    parser.add_argument(
        "--rider-initial-charge",
        type=float,
        default=defaults.rider_initial_charge_cny,
        help="courier charge covering the included road-distance tier",
    )
    parser.add_argument(
        "--rider-included-distance-km",
        type=float,
        default=defaults.rider_included_distance_km,
    )
    parser.add_argument(
        "--rider-extra-cost-per-km",
        type=float,
        default=defaults.rider_incremental_cost_per_km,
    )
    parser.add_argument(
        "--road-distance-factor",
        type=float,
        default=defaults.road_distance_factor,
    )
    parser.add_argument(
        "--feeder-rider-cost-factor",
        type=float,
        default=defaults.feeder_rider_cost_factor,
        help=(
            "factor applied only to first/last-mile rider movements; direct "
            "courier fallback retains the stated rider tariff"
        ),
    )
    parser.add_argument(
        "--locker-pickup-cost",
        type=float,
        default=defaults.locker_pickup_cost_per_order,
        help=(
            "pickup-fulfillment cost per successful locker order; distinct from "
            "--locker-discount, which is a customer-choice utility input"
        ),
    )
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
    parser.add_argument(
        "--no-rel-heur-seconds",
        type=float,
        default=SolverConfig().no_rel_heur_seconds,
        help="maximum main-solve time used by Gurobi's pre-root primal heuristic",
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
    parser.add_argument(
        "--root-cut-passes",
        type=int,
        default=SolverConfig().root_cut_passes,
        help=(
            "maximum native root cut passes (default: 2); -1 keeps Gurobi "
            "automatic, while a nonnegative value is an explicit benchmark override"
        ),
    )
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
    parser.add_argument(
        "--miqcp-method",
        type=int,
        choices=(-1, 0, 1),
        default=SolverConfig().miqcp_method,
        help=(
            "Gurobi MIQCPMethod: -1=automatic, 0=continuous QCP node "
            "relaxations, 1=outer approximation"
        ),
    )
    if include_instance:
        parser.add_argument(
            "--solver-seed", "--seed", dest="solver_seed", type=int, default=42
        )
    else:
        parser.add_argument("--solver-seed", type=int, default=42)
    parser.add_argument(
        "--warm-start-from",
        default=None,
        metavar="INCUMBENT.sol",
        help=(
            "load a named Gurobi solution into the first requested mode; "
            "later MX still prefers an incumbent produced by HD in this run"
        ),
    )
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--write-lp", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare Meituan OD data and solve the OD-aware drone network model."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser(
        "prepare", help="convert raw TSL/Nanshan data to the canonical schema"
    )
    _add_data_arguments(prepare)
    solve = subparsers.add_parser(
        "solve", help="solve one or more formulations on a canonical instance"
    )
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
        )
        .expanduser()
        .resolve()
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
        battery_range_km=args.battery_range_km,
        drone_speed_mps=args.drone_speed_mps,
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
        rider_initial_charge_cny=args.rider_initial_charge,
        rider_included_distance_km=args.rider_included_distance_km,
        rider_incremental_cost_per_km=args.rider_extra_cost_per_km,
        road_distance_factor=args.road_distance_factor,
        feeder_rider_cost_factor=args.feeder_rider_cost_factor,
        locker_pickup_cost_per_order=args.locker_pickup_cost,
        display_origin_choices=args.display_origin_choices,
        general_corridor_choices=args.general_corridor_choices,
    )


def _load_gurobi_solution(
    raw_path: str | os.PathLike[str],
) -> tuple[Path, str, dict[str, float]]:
    """Strictly load a plain-text Gurobi ``.sol`` file and hash its bytes."""

    requested_path = Path(raw_path).expanduser()
    if requested_path.suffix.lower() != ".sol":
        raise ValueError("warm-start file must use the .sol extension")
    try:
        path = requested_path.resolve(strict=True)
    except OSError as exc:
        raise ValueError(f"warm-start file does not exist: {requested_path}") from exc
    if not path.is_file():
        raise ValueError(f"warm-start path is not a regular file: {path}")

    try:
        payload = path.read_bytes()
        text = payload.decode("utf-8-sig")
    except (OSError, UnicodeError) as exc:
        raise ValueError(f"cannot read warm-start file as UTF-8: {path}") from exc

    values: dict[str, float] = {}
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) != 2:
            raise ValueError(
                f"{path}:{line_number}: expected '<variable-name> <finite-value>'"
            )
        name, raw_value = fields
        if name in values:
            raise ValueError(
                f"{path}:{line_number}: duplicate warm-start variable {name!r}"
            )
        try:
            value = float(raw_value)
        except ValueError as exc:
            raise ValueError(
                f"{path}:{line_number}: invalid warm-start value for {name!r}"
            ) from exc
        if not math.isfinite(value):
            raise ValueError(
                f"{path}:{line_number}: nonfinite warm-start value for {name!r}"
            )
        values[name] = value

    if not values:
        raise ValueError(f"warm-start file contains no variable assignments: {path}")
    return path, hashlib.sha256(payload).hexdigest(), values


def _validated_modes(raw_modes: tuple[str, ...]) -> tuple[str, ...]:
    """Return a duplicate-free legal subsequence of HD -> MX -> LO."""

    modes = tuple(raw_mode.upper() for raw_mode in raw_modes)
    invalid_modes = sorted(set(modes) - VALID_MODES)
    if not modes or invalid_modes:
        raise ValueError(
            "modes must contain one or more of HD, MX, LO"
            + (f"; invalid={','.join(invalid_modes)}" if invalid_modes else "")
        )
    if len(set(modes)) != len(modes):
        raise ValueError("modes must not contain duplicates")
    positions = tuple(MODE_RUN_ORDER.index(mode) for mode in modes)
    if positions != tuple(sorted(positions)):
        raise ValueError("modes must preserve the legal HD -> MX -> LO order")
    return modes


def _cross_mode_start_policy(modes: tuple[str, ...]) -> str | None:
    """Describe the installed HD-to-MX start only when that transfer can occur."""

    if "HD" in modes and "MX" in modes and modes.index("HD") < modes.index("MX"):
        return "HD_to_later_MX_by_variable_name"
    return None


def _prepare_empty_results_root(results_root: Path) -> None:
    """Create an output directory or reject any pre-existing contents."""

    if results_root.exists():
        if not results_root.is_dir():
            raise ValueError(f"results path is not a directory: {results_root}")
        if next(results_root.iterdir(), None) is not None:
            raise FileExistsError(
                "results directory must be empty; refusing to overwrite or mix "
                f"artifacts: {results_root}"
            )
        return
    results_root.mkdir(parents=True)


def _solve(args: argparse.Namespace, instance_dir: str) -> list[dict[str, object]]:
    started_at = datetime.now(timezone.utc).isoformat()
    git_provenance = _git_provenance()
    execution_environment = _execution_environment(started_at)
    modes = _validated_modes(args.modes)
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
        no_rel_heur_seconds=args.no_rel_heur_seconds,
        use_valid_inequalities=args.use_valid_inequalities,
        use_user_cuts=not args.no_user_cuts,
        max_root_cut_rounds=args.max_root_cut_rounds,
        root_cut_passes=args.root_cut_passes,
        max_cuts_per_round=args.max_cuts_per_round,
        cut_violation_tolerance=args.cut_violation_tolerance,
        use_local_branching=not args.no_local_branching,
        local_branching_radius=args.local_branching_radius,
        root_relaxation_method=args.root_method,
        miqcp_method=args.miqcp_method,
    )
    solver_config.validate()
    instance_path = Path(instance_dir).expanduser().resolve()
    with open(instance_path / "metadata.json", encoding="utf-8") as handle:
        instance_metadata = json.load(handle)
    _validate_instance_demand_scale(instance_metadata, args.demand_scale)

    warm_start_path: Path | None = None
    warm_start_sha256: str | None = None
    warm_start_values: dict[str, float] | None = None
    if args.warm_start_from is not None:
        warm_start_path, warm_start_sha256, warm_start_values = _load_gurobi_solution(
            args.warm_start_from
        )
    warm_start_record = (
        {
            "absolute_path": str(warm_start_path),
            "sha256": warm_start_sha256,
            "variable_count": len(warm_start_values),
        }
        if warm_start_values is not None
        else None
    )

    from src.data_processor.loader import DataLoader
    from src.model.gurobi_model import MLNDModel
    from src.utils.output_processor import ResultProcessor

    canonical_hashes = {
        name: _sha256_file(instance_path / name)
        for name in (
            "metadata.json",
            "sites.csv",
            "od_nodes.csv",
            "od_demand.csv",
            "periods.csv",
        )
        if (instance_path / name).exists()
    }
    formulation_source = {
        "path": str(FORMULATION_SOURCE.resolve()),
        "sha256": _sha256_file(FORMULATION_SOURCE),
    }
    implementation = _implementation_provenance()
    model_derived = {
        "drone_speed_kmph": model_config.drone_speed_kmph,
        "usable_battery_time_hours": model_config.usable_battery_time_hours,
        "usable_battery_time_minutes": 60.0 * model_config.usable_battery_time_hours,
    }
    instance = DataLoader().load_instance_dir(
        str(instance_path),
        model_config=model_config,
        demand_scale=args.demand_scale,
    )
    screening_by_mode = _screening_is_exact(instance, modes)
    cross_mode_start_policy = _cross_mode_start_policy(modes)
    run_signature = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "formulation_id": FORMULATION_ID,
        "formulation_source": formulation_source,
        "implementation": implementation,
        "instance_dir": str(instance_path),
        "canonical_file_sha256": canonical_hashes,
        "demand_scale": args.demand_scale,
        "modes": modes,
        "cross_mode_start_policy": cross_mode_start_policy,
        "warm_start_from": warm_start_record,
        "file_warm_start_policy": "first_mode_only",
        "screening_is_exact": screening_by_mode,
        "model": to_dict(model_config),
        "model_derived": model_derived,
        "solver": to_dict(solver_config),
    }
    run_signature_sha256 = _payload_sha256(run_signature)
    default_results = (
        REPO_ROOT / "results" / f"{instance_path.name}_{run_signature_sha256[:10]}"
    )
    results_root = (
        Path(args.results_dir).expanduser().resolve()
        if args.results_dir
        else default_results
    )
    _prepare_empty_results_root(results_root)
    with open(results_root / "run_config.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "schema_version": RESULT_SCHEMA_VERSION,
                "formulation_id": FORMULATION_ID,
                "run_signature_sha256": run_signature_sha256,
                "formulation_source": formulation_source,
                "implementation": implementation,
                "instance": {
                    "directory": str(instance_path),
                    "metadata": instance_metadata,
                    "canonical_file_sha256": canonical_hashes,
                },
                "demand_scale": args.demand_scale,
                "modes": list(modes),
                "cross_mode_start_policy": cross_mode_start_policy,
                "warm_start_from": warm_start_record,
                "file_warm_start_policy": "first_mode_only",
                "screening_is_exact": screening_by_mode,
                "model": to_dict(model_config),
                "model_derived": model_derived,
                "solver": to_dict(solver_config),
                "git": git_provenance,
                "execution_environment": execution_environment,
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )

    summaries: list[dict[str, object]] = []
    hd_start: dict[str, float] | None = None
    for mode_index, mode in enumerate(modes):
        model = MLNDModel(instance, mode=mode).prepare_algorithm(solver_config)
        mode_dir = results_root / mode
        mode_dir.mkdir(parents=True, exist_ok=True)
        # Keep the complete solver trace next to the structured result.  The
        # trace is needed to audit incumbent/bound progress on time-limited
        # benchmark runs; summary.json alone contains only the final point.
        model.model.Params.LogFile = str(mode_dir / "solver.log")
        if args.write_lp:
            model.write_model(str(mode_dir / "model.lp"))
        external_start: dict[str, float] | None = None
        external_start_source: str | None = None
        if mode_index == 0 and warm_start_values is not None:
            external_start = warm_start_values
            external_start_source = f"file:{warm_start_path}"
        elif mode == "MX" and hd_start is not None:
            external_start = hd_start
            external_start_source = "HD"
        if not model.solve(
            solver_config,
            external_start=external_start,
            external_start_source=external_start_source,
        ):
            raise RuntimeError(
                f"{mode} finished without a feasible incumbent (status={model.model.Status})"
            )
        incumbent_path = mode_dir / "incumbent.sol"
        model.model.write(str(incumbent_path))
        if not incumbent_path.is_file() or incumbent_path.stat().st_size == 0:
            raise RuntimeError(
                f"{mode} incumbent solution was not written: {incumbent_path}"
            )
        processor = ResultProcessor(
            model,
            instance,
            run_signature_sha256=run_signature_sha256,
        )
        processor.print_summary()
        summary = processor.write_all(str(mode_dir))
        summaries.append(summary)
        if mode == "HD":
            hd_start = model.export_incumbent_start()
        dispose = getattr(model.model, "dispose", None)
        if callable(dispose):
            dispose()

    _write_mode_comparison(results_root, summaries)
    completion_path = _write_run_complete(
        results_root,
        started_at=started_at,
        summaries=summaries,
    )
    print(f"Run completion manifest: {completion_path.resolve()}")
    print(f"Results: {results_root.resolve()}")
    return summaries


def _write_mode_comparison(
    output_dir: Path, summaries: list[dict[str, object]]
) -> None:
    if not summaries:
        return
    fields = [
        "mode",
        "run_signature_sha256",
        "status_name",
        "objective",
        "courier_only_baseline",
        "cost_saving",
        "cost_saving_percent",
        "drone_served_rate",
        "fleet_size",
        "network_is_nontrivial",
        "network_opened",
        "courier_only_incumbent",
        "open_bases",
        "open_lockers",
        "active_lockers",
        "idle_open_lockers",
        "open_swap_stations",
        "selected_rho_state_count",
        "selected_rho_by_idle_state",
        "runtime_seconds",
        "mip_gap",
        "requested_mip_gap",
        "screening_is_exact",
    ]
    with open(
        output_dir / "mode_comparison.csv", "w", newline="", encoding="utf-8"
    ) as handle:
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
    savings = [
        0.0
        if abs(
            float(summary.get("cost_saving", baseline - float(summary["objective"])))
        )
        <= 1e-7
        else float(summary.get("cost_saving", baseline - float(summary["objective"])))
        for summary in summaries
    ]
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
    cost_ax.bar_label(
        bars, labels=[f"{value:,.2f}" for value in costs], padding=3, fontsize=8.5
    )
    cost_ax.set_ylabel("Modeled fulfillment cost (CNY/hour)")
    cost_ax.set_ylim(bottom=0)
    cost_ax.grid(axis="y", color="#e2e8f0", linewidth=0.7)
    cost_ax.set_axisbelow(True)
    cost_ax.spines[["top", "right"]].set_visible(False)
    cost_ax.set_title(
        "Absolute incumbent cost", loc="left", fontsize=11, fontweight="semibold"
    )

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
    saving_span = max((abs(value) for value in savings), default=0.0)
    if saving_span <= 1e-7:
        saving_ax.set_ylim(-1.0, 1.0)
        saving_ax.text(
            0.5,
            0.62,
            "No saving in any incumbent",
            transform=saving_ax.transAxes,
            ha="center",
            va="center",
            color="#475569",
            fontsize=10,
        )
    else:
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
        rho_text = (
            "rho none"
            if not rho_counts
            else "rho "
            + ", ".join(f"{rho}:{count}" for rho, count in sorted(rho_counts.items()))
        )
        value = savings[index - 1]
        percent = 100.0 * value / baseline if baseline else 0.0
        saving_ax.annotate(
            f"{value:,.2f} ({percent:.3f}%)\n{gap_text}\n{open_text}\n{rho_text}",
            (
                saving_bars[index - 1].get_x()
                + saving_bars[index - 1].get_width() / 2.0,
                value,
            ),
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


def _screening_is_exact(instance: object, modes: tuple[str, ...]) -> dict[str, bool]:
    screening_method = getattr(instance, "screening_is_exact", None)
    if not callable(screening_method):
        raise TypeError("instance must provide screening_is_exact(mode)")
    return {mode: bool(screening_method(mode)) for mode in modes}


def _payload_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _short_hash(payload: object) -> str:
    return _payload_sha256(payload)[:10]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_text(*args: str) -> str | None:
    """Return one short Git value without making the experiment depend on Git."""

    try:
        completed = subprocess.run(
            ["git", "-C", str(REPO_ROOT), *args],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip()


def _git_provenance() -> dict[str, object]:
    """Record the exact tracked source state used by a computational run."""

    tracked_status = _git_text("status", "--porcelain=v1", "--untracked-files=no")
    diff_payload = _git_text("diff", "--binary", "HEAD")
    return {
        "branch": _git_text("branch", "--show-current"),
        "head": _git_text("rev-parse", "HEAD"),
        "tracked_worktree_dirty": bool(tracked_status),
        "tracked_status": tracked_status or "",
        "tracked_diff_sha256": (
            hashlib.sha256(diff_payload.encode("utf-8")).hexdigest()
            if diff_payload
            else None
        ),
    }


def _sysctl_value(name: str) -> str | None:
    try:
        completed = subprocess.run(
            ["sysctl", "-n", name],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    value = completed.stdout.strip()
    return value or None


def _execution_environment(started_at: str) -> dict[str, object]:
    """Return non-sensitive hardware and software fields needed for replication."""

    try:
        import gurobipy as gp

        gurobi_version: list[int] | None = list(gp.gurobi.version())
    except Exception:
        gurobi_version = None
    memory_raw = _sysctl_value("hw.memsize")
    return {
        "started_at_utc": started_at,
        "python_version": platform.python_version(),
        "python_executable": str(Path(sys.executable).resolve()),
        "gurobi_version": gurobi_version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_model": _sysctl_value("machdep.cpu.brand_string") or platform.processor(),
        "logical_cpu_count": os.cpu_count(),
        "memory_bytes": int(memory_raw) if memory_raw and memory_raw.isdigit() else None,
    }


def _write_run_complete(
    results_root: Path,
    *,
    started_at: str,
    summaries: list[dict[str, object]],
) -> Path:
    """Seal one complete run with hashes for every solver-produced artifact."""

    artifacts: list[dict[str, object]] = []
    for path in sorted(results_root.rglob("*")):
        if not path.is_file() or path.name == "run_complete.json":
            continue
        artifacts.append(
            {
                "path": path.relative_to(results_root).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
    payload = {
        "schema_version": 1,
        "completed": True,
        "started_at_utc": started_at,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "modes": [str(summary["mode"]) for summary in summaries],
        "mode_status": {
            str(summary["mode"]): {
                "status": summary.get("status_name"),
                "objective": summary.get("objective"),
                "objective_bound": summary.get("objective_bound"),
                "mip_gap": summary.get("mip_gap"),
                "runtime_seconds": summary.get("runtime_seconds"),
                "screening_is_exact": summary.get("screening_is_exact"),
            }
            for summary in summaries
        },
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
    }
    output = results_root / "run_complete.json"
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return output


def _implementation_provenance() -> dict[str, object]:
    """Hash the executable model surface so different builds cannot be pooled."""

    files = {
        path.relative_to(REPO_ROOT).as_posix(): _sha256_file(path)
        for path in FORMULATION_IMPLEMENTATION_FILES
    }
    encoded = json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "files": files,
    }


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
