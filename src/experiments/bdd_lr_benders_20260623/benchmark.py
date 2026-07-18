"""Unified Gurobi/LR/BD/LR+BD benchmark for the 2026-06-23 framework."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from typing import Iterable

from gurobipy import GRB


PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src import config
from src.data_processor.loader import DataLoader
from src.experiments.benchmark_regime_ii_algorithms import (
    _status_name,
    infer_instance_dir,
    override_config,
    save_design,
    validate_design,
)
from src.model.bdd_lr_benders_20260623 import (
    ConicBendersSolver20260623,
    LROnlySolver20260623,
)
from src.model.gurobi_model import MLNDModel


METHODS = ("gurobi", "lr", "bd", "lr_bd")


def _parse_methods(value: str) -> list[str]:
    methods = [item.strip() for item in value.split(",") if item.strip()]
    unknown = [item for item in methods if item not in METHODS]
    if unknown:
        raise argparse.ArgumentTypeError(
            f"Unknown methods: {unknown}; available={METHODS}"
        )
    return methods


def _write_csv(path: str, rows: Iterable[dict]) -> None:
    rows = list(rows)
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with open(path, "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--series-root",
        default=os.path.join(
            PROJECT_ROOT, "data", "instances", "Meituan_H_series"
        ),
    )
    parser.add_argument("--hub-count", type=int, default=5)
    parser.add_argument("--instance-dir", default=None)
    parser.add_argument("--period-id", default="h12")
    parser.add_argument(
        "--methods",
        type=_parse_methods,
        default=_parse_methods("gurobi,lr,bd,lr_bd"),
    )
    parser.add_argument(
        "--results-root",
        default=os.path.join(
            PROJECT_ROOT, "results", "bdd_lr_benders_20260623"
        ),
    )
    parser.add_argument("--variant", default="full")
    parser.add_argument("--time-limit", type=float, default=900.0)
    parser.add_argument("--validation-time-limit", type=float, default=60.0)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--mip-gap", type=float, default=0.001)
    parser.add_argument("--max-iterations", type=int, default=10000)
    parser.add_argument("--benders-tolerance", type=float, default=0.0001)
    parser.add_argument("--lr-time-limit", type=float, default=120.0)
    parser.add_argument("--lr-iterations", type=int, default=50)
    parser.add_argument("--lr-initial-step", type=float, default=20.0)
    parser.add_argument("--lr-max-designs", type=int, default=12)
    parser.add_argument("--lr-restricted-solution-pool-size", type=int, default=1)
    parser.add_argument("--lr-restricted-time-limit", type=float, default=None)
    parser.add_argument("--lr-anchor-top-pairs", type=int, default=0)
    parser.add_argument(
        "--lr-queue-tangents",
        dest="lr_queue_tangents",
        action="store_true",
    )
    parser.add_argument(
        "--no-lr-queue-tangents",
        dest="lr_queue_tangents",
        action="store_false",
    )
    parser.add_argument(
        "--local-branching",
        dest="local_branching",
        action="store_true",
    )
    parser.add_argument(
        "--no-local-branching",
        dest="local_branching",
        action="store_false",
    )
    parser.set_defaults(
        lr_queue_tangents=True,
        local_branching=True,
    )
    parser.add_argument("--no-pareto-cuts", action="store_true")
    parser.add_argument("--no-bdd-strengthened", action="store_true")
    parser.add_argument("--no-bdd-lagrangian", action="store_true")
    parser.add_argument("--bdd-root-rounds", type=int, default=3)
    parser.add_argument("--bdd-inner-time-limit", type=float, default=10.0)
    parser.add_argument("--bdd-full-integrality", action="store_true")
    parser.add_argument("--bdd-epsilon-abs", type=float, default=None)
    parser.add_argument("--no-global-linearized-bound", action="store_true")
    parser.add_argument("--no-master-proxy", action="store_true")
    parser.add_argument(
        "--global-bound-time-fraction", type=float, default=0.10
    )
    parser.add_argument("--global-bound-time-limit", type=float, default=60.0)
    parser.add_argument("--local-branching-radius", type=int, default=6)
    parser.add_argument("--local-branching-gamma-radius", type=int, default=8)
    parser.add_argument("--local-branching-max-calls", type=int, default=32)
    parser.add_argument("--local-branching-max-radius", type=int, default=160)
    parser.add_argument(
        "--local-branching-max-gamma-radius", type=int, default=240
    )
    parser.add_argument(
        "--local-branching-distance-scope",
        choices=("all", "facility", "layered"),
        default="all",
    )
    parser.add_argument("--local-branching-start-pool-size", type=int, default=1)
    parser.add_argument(
        "--local-branching-max-calls-per-center", type=int, default=0
    )
    parser.add_argument("--local-branching-arc-radius", type=int, default=64)
    parser.add_argument(
        "--local-branching-max-arc-radius", type=int, default=260
    )
    parser.add_argument(
        "--local-branching-solution-pool-size", type=int, default=1
    )
    parser.add_argument(
        "--local-branching-benders-envelope-cuts", type=int, default=0
    )
    parser.add_argument(
        "--local-branching-benders-envelope-weight", type=float, default=1.0
    )
    parser.add_argument(
        "--local-branching-benders-envelope-cutoff",
        action="store_true",
    )
    parser.add_argument("--local-branching-max-removals", type=int, default=-1)
    parser.add_argument("--local-branching-protect-top-pairs", type=int, default=0)
    parser.add_argument("--local-branching-time-limit", type=float, default=5.0)
    parser.add_argument("--gamma-sweep-max-calls", type=int, default=0)
    parser.add_argument("--gamma-sweep-radius", type=int, default=40)
    parser.add_argument("--gamma-sweep-step", type=int, default=8)
    parser.add_argument("--gamma-sweep-time-reserve", type=float, default=0.0)
    parser.add_argument("--skip-validation", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser


def _extract_design(model: MLNDModel) -> dict[str, object]:
    return {
        "x": {key: float(var.X) for key, var in model.vars["x"].items()},
        "y": {key: float(var.X) for key, var in model.vars["y"].items()},
        "z": {key: float(var.X) for key, var in model.vars["z"].items()},
        "Gamma": float(model.vars["Gamma"].X),
    }


def run_gurobi(instance, args):
    model = MLNDModel(instance)
    model.build()
    model.model.Params.OutputFlag = int(args.verbose)
    model.model.Params.Threads = int(args.threads)
    model.model.Params.TimeLimit = float(args.time_limit)
    model.model.Params.MIPGap = float(args.mip_gap)
    # The queueing MISOCP is numerically delicate.  A default run has
    # repeatedly declared optimality below independently validated feasible
    # designs, so the formal Gurobi baseline uses conservative numerics.
    model.model.Params.NumericFocus = 3
    model.model.Params.ScaleFlag = 2
    model.model.Params.FeasibilityTol = 1e-7
    model.model.Params.OptimalityTol = 1e-7
    model.model.Params.IntFeasTol = 1e-7
    model.model.Params.BarQCPConvTol = 1e-8
    model.model._best_callback_obj = -float("inf")
    model.model._time_to_best = None
    start = time.perf_counter()

    def callback(gurobi_model, where):
        if where != GRB.Callback.MIPSOL:
            return
        value = float(
            gurobi_model.cbGet(GRB.Callback.MIPSOL_OBJ)
        )
        if value > gurobi_model._best_callback_obj + 1e-7:
            gurobi_model._best_callback_obj = value
            gurobi_model._time_to_best = time.perf_counter() - start

    model.model.optimize(callback)
    wall_time = time.perf_counter() - start
    sol_count = int(model.model.SolCount)
    summary = {
        "method": "gurobi",
        "status": int(model.model.Status),
        "status_name": _status_name(int(model.model.Status)),
        "sol_count": sol_count,
        "obj": float(model.model.ObjVal) if sol_count else None,
        "bound": float(model.model.ObjBound),
        "gap": float(model.model.MIPGap) if sol_count else None,
        "wall_time": wall_time,
        "time_to_best": model.model._time_to_best,
        "master_is_milp": False,
        "master_qconstrs": None,
        "gurobi_threads_effective": int(model.model.Params.Threads),
        "gurobi_numeric_focus": int(model.model.Params.NumericFocus),
        "gurobi_scale_flag": int(model.model.Params.ScaleFlag),
    }
    return summary, _extract_design(model) if sol_count else None, None


def run_decomposition(instance, method: str, args):
    common = {
        "lr_iterations": args.lr_iterations,
        "lr_initial_step": args.lr_initial_step,
        "lr_max_designs": args.lr_max_designs,
        "lr_use_queue_tangents": args.lr_queue_tangents,
        "lr_restricted_solution_pool_size": (
            args.lr_restricted_solution_pool_size
        ),
        "lr_restricted_time_limit": args.lr_restricted_time_limit,
        "lr_anchor_top_pairs": args.lr_anchor_top_pairs,
        "gamma_sweep_max_calls": args.gamma_sweep_max_calls,
        "gamma_sweep_radius": args.gamma_sweep_radius,
        "gamma_sweep_step": args.gamma_sweep_step,
        "gamma_sweep_time_reserve": args.gamma_sweep_time_reserve,
    }
    if method == "lr":
        solver = LROnlySolver20260623(
            instance,
            use_global_linearized_bound=(
                not args.no_global_linearized_bound
            ),
            global_bound_time_fraction=args.global_bound_time_fraction,
            global_bound_time_limit=args.global_bound_time_limit,
            **common,
        )
        result = solver.solve(time_limit=args.time_limit)
    else:
        solver = ConicBendersSolver20260623(
            instance,
            use_lr=(method == "lr_bd"),
            use_pareto_cuts=not args.no_pareto_cuts,
            use_bdd_strengthened=not args.no_bdd_strengthened,
            use_bdd_lagrangian=not args.no_bdd_lagrangian,
            use_local_branching=args.local_branching,
            local_branching_radius=args.local_branching_radius,
            local_branching_gamma_radius=args.local_branching_gamma_radius,
            local_branching_max_calls=args.local_branching_max_calls,
            local_branching_max_radius=args.local_branching_max_radius,
            local_branching_max_gamma_radius=(
                args.local_branching_max_gamma_radius
            ),
            local_branching_distance_scope=(
                args.local_branching_distance_scope
            ),
            local_branching_start_pool_size=(
                args.local_branching_start_pool_size
            ),
            local_branching_max_calls_per_center=(
                args.local_branching_max_calls_per_center
            ),
            local_branching_arc_radius=args.local_branching_arc_radius,
            local_branching_max_arc_radius=(
                args.local_branching_max_arc_radius
            ),
            local_branching_solution_pool_size=(
                args.local_branching_solution_pool_size
            ),
            local_branching_benders_envelope_cuts=(
                args.local_branching_benders_envelope_cuts
            ),
            local_branching_benders_envelope_weight=(
                args.local_branching_benders_envelope_weight
            ),
            local_branching_benders_envelope_cutoff=(
                args.local_branching_benders_envelope_cutoff
            ),
            local_branching_max_removals=args.local_branching_max_removals,
            local_branching_protect_top_pairs=(
                args.local_branching_protect_top_pairs
            ),
            bdd_root_rounds=args.bdd_root_rounds,
            bdd_inner_time_limit=args.bdd_inner_time_limit,
            bdd_full_integrality=args.bdd_full_integrality,
            bdd_epsilon_abs=args.bdd_epsilon_abs,
            use_global_linearized_bound=(
                not args.no_global_linearized_bound
            ),
            use_master_proxy=not args.no_master_proxy,
            global_bound_time_fraction=args.global_bound_time_fraction,
            global_bound_time_limit=args.global_bound_time_limit,
            verbose=args.verbose,
            **common,
        )
        result = solver.solve(
            time_limit=args.time_limit,
            lr_time_limit=args.lr_time_limit if method == "lr_bd" else 0.0,
            max_iterations=args.max_iterations,
            tolerance=args.benders_tolerance,
            local_branching_time_limit=args.local_branching_time_limit,
        )
    return result.summary(), result.best_design, result


def _flat_summary(summary: dict) -> dict:
    return {
        key: value
        for key, value in summary.items()
        if not isinstance(value, (dict, list, tuple))
    }


def main() -> None:
    args = build_parser().parse_args()
    if args.time_limit <= 0.0:
        raise ValueError("--time-limit must be positive")
    if args.threads <= 0:
        raise ValueError("--threads must be positive")

    instance_dir = args.instance_dir or infer_instance_dir(
        args.series_root, args.hub_count
    )
    scale_dir = os.path.join(
        args.results_root, f"H{args.hub_count:03d}"
    )
    os.makedirs(scale_dir, exist_ok=True)
    summaries = []

    overrides = {
        "MODEL_TIME_LIMIT": float(args.time_limit),
        "MODEL_MIP_GAP": float(args.mip_gap),
        "BENDERS_TIME_LIMIT": float(args.time_limit),
        "BENDERS_MASTER_THREADS": int(args.threads),
        "BENDERS_SUB_THREADS": 1,
        "BENDERS_SUB_TIME_LIMIT": min(120.0, float(args.time_limit)),
        "BENDERS_MASTER_MIP_GAP": float(args.mip_gap),
    }
    with override_config(**overrides):
        instance = DataLoader().load_instance_dir(
            instance_dir, period_id=args.period_id
        )
        for method in args.methods:
            result_name = (
                method
                if args.variant == "full"
                else f"{method}__{args.variant}"
            )
            result_dir = os.path.join(scale_dir, result_name)
            os.makedirs(result_dir, exist_ok=True)
            print(
                f"=== 20260623 {method}: H={len(instance.H)}, "
                f"budget={args.time_limit:.0f}s ===",
                flush=True,
            )
            if method == "gurobi":
                summary, design, result = run_gurobi(instance, args)
            else:
                summary, design, result = run_decomposition(
                    instance, method, args
                )

            summary.update(
                {
                    "instance_dir": instance_dir,
                    "period_id": args.period_id,
                    "H": len(instance.H),
                    "L": len(instance.L),
                    "E": len(instance.E),
                    "OD": len(instance.h),
                    "A": len(instance.A),
                    "time_limit": float(args.time_limit),
                    "threads": int(args.threads),
                    "version": "20260623",
                    "variant": args.variant,
                }
            )
            if design is not None and not args.skip_validation:
                validation = validate_design(
                    instance,
                    design,
                    time_limit=args.validation_time_limit,
                )
                summary.update(validation)
                if (
                    validation.get("validation_obj") is not None
                    and summary.get("obj") is not None
                ):
                    summary["validation_obj_diff"] = (
                        float(validation["validation_obj"])
                        - float(summary["obj"])
                    )

            if design is not None:
                save_design(os.path.join(result_dir, "design.json"), design)
            if result is not None:
                _write_csv(
                    os.path.join(result_dir, "history.csv"),
                    result.history,
                )
                _write_csv(
                    os.path.join(result_dir, "initial_design_history.csv"),
                    result.initial_design_history,
                )
                _write_csv(
                    os.path.join(result_dir, "cut_history.csv"),
                    [record.as_dict() for record in result.cut_history],
                )
                _write_csv(
                    os.path.join(result_dir, "local_branching_history.csv"),
                    [
                        record.as_dict()
                        for record in result.local_branching_history
                    ],
                )
                _write_csv(
                    os.path.join(result_dir, "gamma_sweep_history.csv"),
                    result.gamma_sweep_history,
                )
                if result.lr_result is not None:
                    _write_csv(
                        os.path.join(result_dir, "lr_history.csv"),
                        [
                            record.as_dict()
                            for record in result.lr_result.history
                        ],
                    )
                    _write_csv(
                        os.path.join(result_dir, "lr_candidates.csv"),
                        result.lr_result.candidate_history,
                    )
                timing_rows = result.metadata.get("timing_rows", [])
                _write_csv(
                    os.path.join(result_dir, "timing.csv"), timing_rows
                )
                with open(
                    os.path.join(result_dir, "cpp_decision.json"),
                    "w",
                    encoding="utf-8",
                ) as stream:
                    json.dump(
                        result.metadata.get("cpp_acceleration_decision"),
                        stream,
                        indent=2,
                        sort_keys=True,
                    )

            with open(
                os.path.join(result_dir, "summary.json"),
                "w",
                encoding="utf-8",
            ) as stream:
                json.dump(summary, stream, indent=2, sort_keys=True)
            _write_csv(
                os.path.join(result_dir, "summary.csv"),
                [_flat_summary(summary)],
            )
            summaries.append(summary)
            print(
                f"{method}: obj={summary.get('obj')} "
                f"bound={summary.get('bound')} gap={summary.get('gap')} "
                f"wall={summary.get('wall_time'):.3f}s",
                flush=True,
            )

    baseline = next(
        (row for row in summaries if row["method"] == "gurobi"), None
    )
    if baseline is not None:
        baseline_obj = baseline.get("validation_obj") or baseline.get("obj")
        if baseline_obj is not None:
            for row in summaries:
                value = row.get("validation_obj") or row.get("obj")
                if value is None:
                    continue
                row["obj_vs_gurobi_abs"] = float(value) - float(baseline_obj)
                row["obj_vs_gurobi_pct"] = row[
                    "obj_vs_gurobi_abs"
                ] / max(1.0, abs(float(baseline_obj)))

    _write_csv(
        os.path.join(scale_dir, "comparison.csv"),
        [_flat_summary(row) for row in summaries],
    )
    with open(
        os.path.join(scale_dir, "comparison.json"),
        "w",
        encoding="utf-8",
    ) as stream:
        json.dump(summaries, stream, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
