"""Benchmark strict Lagrangian/Benders variants against the original model.

All search phases of a decomposition method, including Lagrangian warm start,
profit estimators, flow-cut separation, MILP master solves and exact SOCP
recourse solves, share ``--time-limit`` (900 seconds by default).

The fresh-original-model validation is deliberately outside that search
budget: it certifies the returned first-stage design and never improves it.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict

from gurobipy import GRB


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src import config
from src.data_processor.loader import DataLoader
from src.experiments.benchmark_regime_ii_algorithms import (
    _status_name,
    _write_csv,
    infer_instance_dir,
    override_config,
    save_design,
    validate_design,
)
from src.model.gurobi_model import MLNDModel
from src.model.lagrangian_benders import StrictLagrangianBendersSolver


METHODS = (
    "gurobi",
    "gurobi_numeric",
    "strict_lagrangian_benders",
    "strict_no_lr",
    "strict_no_flow_cuts",
    "strict_topology_only",
    "strict_no_network_milp",
    "strict_no_global_ub",
    "strict_no_swaps",
    "strict_continuous_gamma",
)


def parse_methods(raw: str) -> list[str]:
    methods = [part.strip() for part in raw.split(",") if part.strip()]
    unknown = [method for method in methods if method not in METHODS]
    if unknown:
        raise argparse.ArgumentTypeError(
            f"Unknown methods: {', '.join(unknown)}. Available: {', '.join(METHODS)}"
        )
    if not methods:
        raise argparse.ArgumentTypeError("At least one method is required.")
    return methods


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark strict MILP-master Lagrangian/Benders algorithms."
    )
    parser.add_argument(
        "--series-root",
        default=os.path.join(PROJECT_ROOT, "data", "instances", "Meituan_H_series"),
    )
    parser.add_argument("--hub-count", "--scale", dest="hub_count", type=int, default=20)
    parser.add_argument("--instance-dir", default=None)
    parser.add_argument("--period-id", default="h12")
    parser.add_argument(
        "--methods",
        type=parse_methods,
        default=parse_methods("gurobi,strict_lagrangian_benders"),
    )
    parser.add_argument(
        "--results-root",
        default=os.path.join(
            PROJECT_ROOT, "results", "strict_lagrangian_benders_benchmark"
        ),
    )
    parser.add_argument("--time-limit", type=float, default=900.0)
    parser.add_argument("--validation-time-limit", type=float, default=60.0)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--mip-gap", type=float, default=0.001)
    parser.add_argument("--max-iterations", type=int, default=10000)
    parser.add_argument("--benders-tolerance", type=float, default=0.0001)
    parser.add_argument("--lr-time-limit", type=float, default=120.0)
    parser.add_argument("--lr-iterations", type=int, default=50)
    parser.add_argument("--lr-initial-step", type=float, default=20.0)
    parser.add_argument("--lr-max-designs", type=int, default=8)
    parser.add_argument("--global-ub-time-limit", type=float, default=60.0)
    parser.add_argument("--global-ub-time-fraction", type=float, default=0.10)
    parser.add_argument("--max-consecutive-estimator-screens", type=int, default=5)
    parser.add_argument("--max-flow-cuts-per-iteration", type=int, default=30)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--skip-incumbent-validation", action="store_true")
    return parser


def _extract_design(model: MLNDModel) -> dict[str, object]:
    return {
        "x": {key: float(var.X) for key, var in model.vars["x"].items()},
        "y": {key: float(var.X) for key, var in model.vars["y"].items()},
        "z": {key: float(var.X) for key, var in model.vars["z"].items()},
        "Gamma": float(model.vars["Gamma"].X),
    }


def run_gurobi(instance, method: str, args):
    model = MLNDModel(instance)
    model.build()
    model.model.Params.OutputFlag = int(args.verbose)
    model.model.Params.Threads = int(args.threads)
    model.model.Params.TimeLimit = float(args.time_limit)
    model.model.Params.MIPGap = float(args.mip_gap)
    if method == "gurobi_numeric":
        model.model.Params.NumericFocus = 3
        model.model.Params.ScaleFlag = 2
        model.model.Params.FeasibilityTol = 1e-7
        model.model.Params.OptimalityTol = 1e-7
        model.model.Params.IntFeasTol = 1e-7
        model.model.Params.BarQCPConvTol = 1e-8
    start = time.perf_counter()
    model.model.optimize()
    wall_time = time.perf_counter() - start
    sol_count = int(model.model.SolCount)
    return (
        {
            "method": method,
            "status": int(model.model.Status),
            "status_name": _status_name(int(model.model.Status)),
            "sol_count": sol_count,
            "obj": float(model.model.ObjVal) if sol_count else None,
            "bound": float(model.model.ObjBound),
            "gap": float(model.model.MIPGap) if sol_count else None,
            "wall_time": wall_time,
            "lr_bound": None,
            "iterations": None,
            "master_is_milp": False,
            "master_qconstrs": None,
        },
        _extract_design(model) if sol_count else None,
        [],
    )


def run_strict(instance, method: str, args):
    use_lr = method != "strict_no_lr"
    use_topology = method != "strict_no_flow_cuts"
    use_swap = method not in {"strict_no_flow_cuts", "strict_topology_only"}
    use_network_milp = method != "strict_no_network_milp"
    use_global_ub = method != "strict_no_global_ub"
    disable_swaps = method == "strict_no_swaps"
    relax_gamma = method == "strict_continuous_gamma"
    solver = StrictLagrangianBendersSolver(
        instance,
        use_topology_cuts=use_topology,
        use_swap_reachability_cuts=use_swap,
        use_topology_estimator=use_topology,
        use_swap_estimator=use_swap,
        use_lagrangian_warmstart=use_lr,
        use_restricted_network_milp=use_network_milp,
        use_global_network_upper_bound=use_global_ub,
        disable_swapping_stations=disable_swaps,
        relax_gamma_integrality=relax_gamma,
        lr_iterations=args.lr_iterations,
        lr_initial_step=args.lr_initial_step,
        lr_max_designs=args.lr_max_designs,
        max_consecutive_estimator_screens=args.max_consecutive_estimator_screens,
        max_combinatorial_cuts_per_iteration=args.max_flow_cuts_per_iteration,
        verbose=args.verbose,
    )
    start = time.perf_counter()
    success = solver.solve(
        max_iterations=args.max_iterations,
        tolerance=args.benders_tolerance,
        time_limit=args.time_limit,
        lr_time_limit=args.lr_time_limit if use_lr else 0.0,
    )
    wall_time = time.perf_counter() - start
    last = solver.history[-1] if solver.history else {}
    lr_result = solver.lagrangian_result
    obj = float(solver.best_total_obj) if success else None
    bound = last.get("UB")
    gap = last.get("rel_gap")
    if gap is None and obj is not None and bound is not None:
        gap = (float(bound) - obj) / max(1.0, abs(obj))
    summary = {
        "method": method,
        "status": None,
        "status_name": "FEASIBLE" if success else "NO_INCUMBENT",
        "sol_count": 1 if success else 0,
        "obj": obj,
        "bound": bound,
        "gap": gap,
        "wall_time": wall_time,
        "lr_bound": (
            float(lr_result.best_dual_bound) if lr_result is not None else None
        ),
        "lr_wall_time": (
            float(lr_result.wall_time) if lr_result is not None else 0.0
        ),
        "global_network_bound": solver.global_network_bound,
        "global_network_incumbent": solver.global_network_incumbent,
        "global_network_gap": solver.global_network_gap,
        "global_network_status": solver.global_network_status,
        "global_network_wall_time": solver.global_network_wall_time,
        "iterations": len(solver.history),
        "cuts": int(solver.cut_count),
        "master_is_milp": True,
        "master_qconstrs": int(solver.master.NumQConstrs),
        **asdict(solver.flow_cut_stats),
    }
    return summary, solver.best_design if success else None, solver.history, solver


def main() -> None:
    args = build_parser().parse_args()
    if args.time_limit <= 0:
        raise ValueError("--time-limit must be positive.")
    if args.threads <= 0:
        raise ValueError("--threads must be positive.")

    instance_dir = args.instance_dir or infer_instance_dir(
        args.series_root, args.hub_count
    )
    scale_dir = os.path.join(args.results_root, f"H{args.hub_count:03d}")
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
        "BENDERS_GLOBAL_UB_TIME_LIMIT": float(
            args.global_ub_time_limit
        ),
        "BENDERS_GLOBAL_UB_TIME_FRACTION": float(
            args.global_ub_time_fraction
        ),
    }
    with override_config(**overrides):
        instance = DataLoader().load_instance_dir(
            instance_dir, period_id=args.period_id
        )
        for method in args.methods:
            result_dir = os.path.join(scale_dir, method)
            os.makedirs(result_dir, exist_ok=True)
            print(
                f"\n=== {method}: H={len(instance.H)}, L={len(instance.L)}, "
                f"E={len(instance.E)}, budget={args.time_limit:.0f}s ===",
                flush=True,
            )
            solver = None
            if method in {"gurobi", "gurobi_numeric"}:
                summary, design, history = run_gurobi(instance, method, args)
            else:
                summary, design, history, solver = run_strict(
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
                }
            )
            if design is not None and not args.skip_incumbent_validation:
                validation = validate_design(
                    instance,
                    design,
                    time_limit=args.validation_time_limit,
                    relax_gamma=(method == "strict_continuous_gamma"),
                )
                summary.update(validation)
                if validation.get("validation_obj") is not None:
                    summary["validation_obj_diff"] = (
                        float(validation["validation_obj"])
                        - float(summary["obj"])
                    )

            if design is not None:
                save_design(os.path.join(result_dir, "design.json"), design)
            if history:
                _write_csv(os.path.join(result_dir, "history.csv"), history)
            if solver is not None and solver.lagrangian_result is not None:
                _write_csv(
                    os.path.join(result_dir, "lagrangian_history.csv"),
                    solver.lagrangian_result.history,
                )
                if solver.lagrangian_result.tabu_history:
                    _write_csv(
                        os.path.join(
                            result_dir, "lagrangian_tabu_history.csv"
                        ),
                        solver.lagrangian_result.tabu_history,
                    )
            if solver is not None and solver.initial_design_history:
                _write_csv(
                    os.path.join(result_dir, "initial_design_history.csv"),
                    solver.initial_design_history,
                )
            with open(
                os.path.join(result_dir, "summary.json"), "w", encoding="utf-8"
            ) as f:
                json.dump(summary, f, indent=2, sort_keys=True)
            _write_csv(os.path.join(result_dir, "summary.csv"), [summary])
            summaries.append(summary)
            print(
                f"{method}: status={summary['status_name']} "
                f"obj={summary.get('obj')} bound={summary.get('bound')} "
                f"gap={summary.get('gap')} wall={summary['wall_time']:.2f}s",
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
                diff = float(value) - float(baseline_obj)
                row["obj_vs_gurobi_abs"] = diff
                row["obj_vs_gurobi_pct"] = diff / max(
                    1.0, abs(float(baseline_obj))
                )
                row["beats_gurobi_incumbent"] = diff > 1e-5

    _write_csv(os.path.join(scale_dir, "comparison.csv"), summaries)
    with open(
        os.path.join(scale_dir, "comparison.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(summaries, f, indent=2, sort_keys=True)
    print(f"\nComparison: {os.path.join(scale_dir, 'comparison.csv')}")


if __name__ == "__main__":
    main()
