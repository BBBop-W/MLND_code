"""Inspect inputs, run optimization, check solutions and generate reports."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path

from .config import load_config, ROOT
from .experiments.storage import json_payload


def emit(value):
    print(
        json.dumps(json_payload(value), ensure_ascii=False, indent=2, allow_nan=False),
        flush=True,
    )


def time_limit(value):
    if value.lower() in ("none", "unlimited"):
        return None
    return float(value)


def parser():
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser(
        "inspect", help="Show candidate counts and complete parameter configuration"
    )
    inspect.add_argument("--hubs", type=int, default=120)
    inspect.add_argument("--config", type=Path)
    run = commands.add_parser(
        "run", help="Run one or more instances in a new experiment directory"
    )
    run.add_argument("--hubs", type=int, nargs="+", default=[11])
    run.add_argument("--config", type=Path)
    run.add_argument(
        "--method", choices=("benders", "hybrid", "direct"), default="benders"
    )
    run.add_argument("--seconds", type=time_limit, default=argparse.SUPPRESS)
    run.add_argument("--threads", type=int)
    run.add_argument("--seed", type=int)
    run.add_argument("--gap", type=float)
    run.add_argument("--max-iterations", type=int)
    run.add_argument("--out", type=Path)
    run.add_argument(
        "--input", type=Path, help="Replay saved instance inputs as a single new case"
    )
    report = commands.add_parser(
        "report", help="Generate Chinese/English reports and figures"
    )
    report.add_argument("path", type=Path)
    report.add_argument("--out", type=Path)
    report.add_argument("--no-maps", action="store_true")
    for name, help_text in (
        ("validate", "Independently reconstruct and check a saved solution"),
        ("status", "Read a run's summary and current progress"),
        ("stop", "Request graceful stop using the run's STOP file"),
    ):
        command = commands.add_parser(name, help=help_text)
        command.add_argument("path", type=Path)
    return root


def _inspect(args):
    from .data.instances import load_instance

    config = load_config(args.config)
    data = load_instance(args.hubs, config)
    emit(
        dict(
            model=data["model_version"],
            candidates={
                "bases": data["n_l"],
                "lockers": data["n_h"],
                "swaps": data["n_k"],
            },
            od_pairs=data["n_od"],
            hourly_demand=float(data["demand"].sum()),
            config=config,
        )
    )
    return 0


def _run(args):
    from .experiments.runner import run_case, run_batch

    if args.input and args.config:
        raise ValueError(
            "--input reuses saved model parameters; use --config with canonical instances"
        )
    config = load_config(args.config)
    mapping = {
        "threads": "threads",
        "seed": "seed",
        "gap": "mip_gap",
        "max_iterations": "max_iterations",
    }
    options = {
        key: getattr(args, flag)
        for flag, key in mapping.items()
        if getattr(args, flag) is not None
    }
    if hasattr(args, "seconds"):
        options["time_limit"] = args.seconds
    load_config(config, overrides={"solver": options})
    output = args.out or ROOT / "outputs/experiments" / datetime.now().strftime(
        "%Y%m%d_%H%M%S_%f"
    )
    if args.input:
        result = run_case(
            None,
            output,
            config=config,
            method=args.method,
            options=options,
            input_path=args.input,
        )
        emit(result)
    else:
        hubs = list(dict.fromkeys(args.hubs))
        result = run_batch(
            hubs,
            output,
            config=config,
            method=args.method,
            options=options,
            progress=emit,
        )
    return 0 if result["passed"] else 1


def _report(args):
    if (args.path / "input_metadata.json").exists():
        from .reporting.case import run
        from .reporting.composition import run as composition

        output = args.out or args.path / "report"
        run(args.path, output=output, draw_maps=not args.no_maps)
        composition(output)
    else:
        from .reporting.scale import build_report

        build_report(args.path, output=args.out)
    return 0


def main(argv=None):
    arguments = parser()
    args = arguments.parse_args(argv)
    try:
        if args.command == "inspect":
            return _inspect(args)
        if args.command == "run":
            return _run(args)
        if args.command == "report":
            return _report(args)
        if args.command == "validate":
            from .experiments.storage import load_seed
            from .operations.validation import build_operations

            data, result = load_seed(args.path)
            summary = build_operations(data, result)["summary"]
            emit(summary)
            return 0 if summary["checks"]["passed"] else 1
        if args.command == "stop":
            if not any(
                (args.path / name).is_file()
                for name in ("manifest.json", "batch_manifest.json")
            ):
                raise ValueError("The target is not an experiment directory")
            (args.path / "STOP").touch()
            emit({"stop_requested": str(args.path.resolve())})
            return 0
        records = {}
        for name in (
            "summary.json",
            "progress.json",
            "checkpoint.json",
            "batch_manifest.json",
        ):
            if (args.path / name).exists():
                records[name.removesuffix(".json")] = json.loads(
                    (args.path / name).read_text()
                )
        if not records:
            raise FileNotFoundError(f"No experiment status found: {args.path}")
        emit(records)
        return 0
    except (ValueError, FileNotFoundError, FileExistsError) as error:
        arguments.error(str(error))


if __name__ == "__main__":
    raise SystemExit(main())
