"""Command-line entry for the single manuscript-aligned model."""

from __future__ import annotations
import argparse
from dataclasses import replace
import json
import math
from pathlib import Path

from src.config import load_config
from src.experiments.storage import json_payload
from src.solvers.methods import ACCELERATIONS, resolve_accelerations


def emit(value):
    print(
        json.dumps(json_payload(value), ensure_ascii=False, indent=2, allow_nan=False)
    )


def _seconds(value):
    if value.lower() in ("none", "unlimited"):
        return None
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise argparse.ArgumentTypeError("seconds must be nonnegative or unlimited")
    return result


def parser():
    root = argparse.ArgumentParser(
        description="Consecutive delivery network design (consecutive)"
    )
    commands = root.add_subparsers(dest="command", required=True)
    commands.add_parser("methods", help="Describe supported methods without solving")
    for name in ("inspect", "plan", "run", "compare"):
        p = commands.add_parser(name)
        p.add_argument(
            "--input",
            type=Path,
            help="Saved input directory; raw market conversion is recorded",
        )
        p.add_argument("--config", type=Path, help="Explicit scenario overrides")
        p.add_argument(
            "--hubs",
            type=int,
            help="Read canonical CSV candidates instead of a saved market",
        )
        if name != "compare":
            p.add_argument("--max-orders", type=int, choices=(1, 2))
        p.add_argument("--flow-policy", choices=("prefix", "aggregate"))
        if name in ("plan", "run", "compare"):
            p.add_argument(
                "--method",
                choices=("full_path", "benders_cg", "benders_cg_plain"),
                default="benders_cg",
            )
            p.add_argument("--seconds", type=_seconds, default=argparse.SUPPRESS)
            p.add_argument("--threads", type=int, default=argparse.SUPPRESS)
            p.add_argument("--seed", type=int, default=argparse.SUPPRESS)
            p.add_argument("--gap", type=float, default=argparse.SUPPRESS)
            p.add_argument("--column-limit", type=int, default=argparse.SUPPRESS)
            p.add_argument("--max-iterations", type=int, default=argparse.SUPPRESS)
            p.add_argument("--out", type=Path, required=name in ("run", "compare"))
            p.add_argument("--no-report", action="store_true")
            p.add_argument(
                "--enable-acceleration",
                choices=ACCELERATIONS,
                action="append",
                default=[],
                help="Enable one Benders strategy; repeat for multiple strategies",
            )
            p.add_argument(
                "--disable-acceleration",
                choices=ACCELERATIONS,
                action="append",
                default=[],
                help="Disable one Benders strategy; repeat for an ablation",
            )
    p = commands.add_parser(
        "prepare", help="Build an L/H instance from observed orders"
    )
    p.add_argument("--source", choices=("nanshan", "tsl"), required=True)
    p.add_argument("--raw-path", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--bases", type=int, default=2)
    p.add_argument("--lockers", type=int, default=10)
    p.add_argument("--ods", type=int, default=200)
    p.add_argument("--seed", type=int, default=43)
    p.add_argument("--dates", nargs="+")
    p.add_argument("--demand-hours", type=float, nargs=2, default=(12, 13))
    p.add_argument("--candidate-hours", type=float, nargs=2, default=(8, 24))
    p.add_argument("--center", type=float, nargs=2)
    p.add_argument("--half-size-km", type=float)
    p.add_argument("--config", type=Path)
    p = commands.add_parser(
        "report", help="Revalidate a saved result and produce figures"
    )
    p.add_argument("path", type=Path)
    p.add_argument("--out", type=Path)
    p.add_argument("--no-maps", action="store_true")
    p.add_argument("--cached-basemap", action="store_true")
    p = commands.add_parser(
        "compare-report", help="Revalidate and compare saved algorithm runs"
    )
    p.add_argument("path", type=Path, help="Parent directory of saved runs")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--cached-basemap", action="store_true")
    p = commands.add_parser(
        "study-report", help="Compare detailed K=1/K=2 operations across real datasets"
    )
    p.add_argument(
        "path", type=Path, help="Study directory containing instances/ and runs/"
    )
    p.add_argument("--out", type=Path)
    for name in ("validate", "status", "stop"):
        p = commands.add_parser(name)
        p.add_argument("path", type=Path)
    return root


def _data(args):
    from src.data.instances import (
        load_saved,
        load_instance,
        to_planning_data,
        pack_planning_data,
    )

    if args.input and args.hubs is not None:
        raise ValueError("Choose either --input or --hubs")
    data = (
        load_saved(args.input, args.config)
        if args.input
        else load_instance(args.hubs, args.config)
    )
    k = getattr(args, "max_orders", None)
    policy = getattr(args, "flow_policy", None)
    overrides = {}
    if k is not None:
        overrides["max_orders"] = k
    if policy is not None:
        overrides["flow_policy"] = policy
    if overrides:
        p = replace(to_planning_data(data), **overrides)
        metadata = data.get("metadata", {})
        if "resolved_config" in metadata:
            metadata["resolved_config"].update(overrides)
        metadata.setdefault("run_overrides", {}).update(overrides)
        data.update(pack_planning_data(p, metadata=metadata))
    return data


def _accelerations(args):
    enabled = set(getattr(args, "enable_acceleration", []))
    disabled = set(getattr(args, "disable_acceleration", []))
    conflicts = enabled & disabled
    if conflicts:
        raise ValueError(
            "Cannot both enable and disable acceleration: "
            + ", ".join(sorted(conflicts))
        )
    if not enabled and not disabled:
        return None
    return {**{name: True for name in enabled}, **{name: False for name in disabled}}


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.command == "methods":
            from src.solvers.methods import METHODS

            emit(METHODS)
        elif args.command == "prepare":
            from src.data.prepare import prepare_orders

            emit(
                prepare_orders(
                    args.source,
                    args.out,
                    n_l=args.bases,
                    n_h=args.lockers,
                    n_od=args.ods,
                    raw_path=args.raw_path,
                    dates=args.dates,
                    demand_hours=args.demand_hours,
                    candidate_hours=args.candidate_hours,
                    center=args.center,
                    half_size_km=args.half_size_km,
                    seed=args.seed,
                    config=args.config,
                )
            )
        elif args.command in ("inspect", "plan", "run", "compare"):
            from src.data.instances import to_planning_data

            accelerations = _accelerations(args)
            if args.command != "inspect":
                resolved_accelerations = resolve_accelerations(
                    args.method, accelerations
                )
            data = _data(args)
            p = to_planning_data(data)
            info = dict(
                model_version="consecutive",
                n_l=p.n_l,
                n_h=p.n_h,
                n_od=p.n_od,
                demand_per_hour=float(p.demand.sum()),
                max_orders=p.max_orders,
                flow_policy=p.flow_policy,
                window_minutes=60 * p.effective_delta,
                route_mode=p.route_mode,
                supplied_routes=p.candidate_routes is not None,
                parameter_sources=data.get("metadata", {}),
            )
            if args.command == "inspect":
                emit(info)
                return 0
            cfg = load_config(args.config)
            solver = cfg["solver"]
            options = dict(
                method=args.method,
                time_limit=getattr(args, "seconds", solver["time_limit"]),
                threads=getattr(args, "threads", solver["threads"]),
                seed=getattr(args, "seed", solver["seed"]),
                mip_gap=getattr(args, "gap", solver["mip_gap"]),
                column_limit=getattr(args, "column_limit", solver["column_limit"]),
                max_iterations=getattr(
                    args, "max_iterations", solver["max_iterations"]
                ),
                make_report=not args.no_report,
                accelerations=accelerations,
            )
            if args.command == "plan":
                emit(
                    dict(
                        inputs=info,
                        options=options,
                        output=args.out,
                        resolved_accelerations=resolved_accelerations,
                    )
                )
                return 0
            from src.experiments.runner import run_case, compare

            emit(
                (compare if args.command == "compare" else run_case)(
                    data, args.out, **options
                )
            )
        elif args.command == "report":
            from src.reporting.report import run

            emit(
                run(
                    args.path,
                    args.out,
                    draw_maps=not args.no_maps,
                    cached_basemap=args.cached_basemap,
                )
            )
        elif args.command == "compare-report":
            from src.reporting.comparison import produce_comparison

            emit(
                produce_comparison(
                    args.path, args.out, cached_basemap=args.cached_basemap
                )
            )
        elif args.command == "study-report":
            from src.reporting.study import produce_study

            emit(produce_study(args.path, output=args.out))
        elif args.command == "validate":
            from src.experiments.storage import load_seed
            from src.operations.validation import build_operations

            data, result = load_seed(args.path)
            s = build_operations(data, result)["summary"]
            emit(s)
            return 0 if s["checks"]["passed"] else 1
        elif args.command == "status":
            from src.experiments.status import read_status

            emit(read_status(args.path))
        elif args.command == "stop":
            if not (args.path / "manifest.json").is_file():
                raise ValueError("STOP target is not a saved run")
            (args.path / "STOP").touch()
            emit(dict(stop_requested=str(args.path.resolve())))
        return 0
    except (ValueError, FileNotFoundError, FileExistsError) as exc:
        emit(dict(error=str(exc)))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
