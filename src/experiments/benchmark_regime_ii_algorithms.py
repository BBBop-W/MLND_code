"""Compare exact/model-based Regime-II algorithms under one wall-clock budget.

Every method receives the same total budget (900 seconds by default).  For the
max-flow variant, root-cut separation time is included in that budget.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Dict, Iterable, Mapping, Optional

from gurobipy import GRB


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src import config
from src.data_processor.loader import DataLoader
try:
    # This checkout currently contains both Benders.py and benders.py.  On the
    # default macOS case-insensitive filesystem, the lowercase module is not
    # always discoverable even though the path exists.
    from src.model.Benders import BendersRegimeIISolver
except ImportError:
    from src.model.benders import BendersRegimeIISolver
from src.model.benders_experimental import (
    ExperimentalBendersRegimeIISolver,
    SinglePeriodPartialBendersRegimeIISolver,
)
from src.model.gurobi_model import MLNDModel
from src.model.max_flow_cuts import add_root_max_flow_cuts


METHODS = (
    "gurobi",
    "gurobi_numeric",
    "gurobi_maxflow",
    "benders",
    "experimental_benders",
    "partial_benders",
)
MODEL_VARIANTS = (
    "full",
    "no_battery_endurance",
    "no_swapping_unlimited_direct",
)


@contextmanager
def override_config(**kwargs):
    old_values = {name: getattr(config, name, None) for name in kwargs}
    existed = {name: hasattr(config, name) for name in kwargs}
    try:
        for name, value in kwargs.items():
            setattr(config, name, value)
        yield
    finally:
        for name, value in old_values.items():
            if existed[name]:
                setattr(config, name, value)
            elif hasattr(config, name):
                delattr(config, name)


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


def infer_instance_dir(series_root: str, hub_count: int) -> str:
    summary_path = os.path.join(series_root, "series_summary.csv")
    with open(summary_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    matches = [row for row in rows if int(row["H"]) == int(hub_count)]
    if not matches:
        available = ", ".join(str(int(row["H"])) for row in rows)
        raise ValueError(
            f"H={hub_count} is unavailable under {series_root}. Available: {available}"
        )
    return matches[0]["instance_dir"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark model-based Regime-II algorithms with a shared time limit."
    )
    parser.add_argument(
        "--series-root",
        default=os.path.join(PROJECT_ROOT, "data", "instances", "Meituan_H_series"),
    )
    parser.add_argument("--hub-count", "--scale", dest="hub_count", type=int, default=20)
    parser.add_argument("--instance-dir", default=None)
    parser.add_argument("--period-id", default="h12")
    parser.add_argument(
        "--model-variant",
        choices=MODEL_VARIANTS,
        default="full",
        help=(
            "full keeps the published model; simplified variants are written "
            "to separate result folders and must not be compared as equivalent models."
        ),
    )
    parser.add_argument(
        "--methods",
        type=parse_methods,
        default=parse_methods(
            "gurobi,gurobi_numeric,gurobi_maxflow,partial_benders"
        ),
        help=f"Comma-separated subset of: {', '.join(METHODS)}.",
    )
    parser.add_argument(
        "--results-root",
        default=os.path.join(PROJECT_ROOT, "results", "regime_ii_algorithm_benchmark"),
    )
    parser.add_argument("--time-limit", type=float, default=900.0)
    parser.add_argument("--mip-gap", type=float, default=0.001)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--benders-max-iterations", type=int, default=10000)
    parser.add_argument("--benders-tolerance", type=float, default=0.0001)
    parser.add_argument("--partial-master-time-limit", type=float, default=30.0)
    parser.add_argument(
        "--partial-keep-route-feasibility",
        dest="partial_keep_route_feasibility",
        action="store_true",
        help="Keep route feasibility variables in the partial master (default).",
    )
    parser.add_argument(
        "--partial-drop-route-feasibility",
        dest="partial_keep_route_feasibility",
        action="store_false",
        help="Use pure route recourse; currently weaker and mainly for ablation.",
    )
    parser.set_defaults(partial_keep_route_feasibility=True)
    parser.add_argument(
        "--no-partial-maxflow-cuts",
        action="store_true",
        help="Disable root max-flow cuts in the partial-Benders master.",
    )
    parser.add_argument(
        "--partial-embed-route-cost",
        action="store_true",
        help=(
            "Add a primal route-cost strengthening using route variables "
            "already kept in the partial master."
        ),
    )
    parser.add_argument(
        "--partial-warmstart-time-limit",
        type=float,
        default=None,
        help=(
            "Maximum time for the selected model-based primal warm start. "
            "Defaults to 60s for maxflow and 240s for numeric; included in --time-limit."
        ),
    )
    parser.add_argument(
        "--partial-warmstart-mode",
        choices=("auto", "maxflow", "numeric"),
        default="auto",
        help="Model-based primal warm-start strategy used inside partial Benders.",
    )
    parser.add_argument("--maxflow-time-limit", type=float, default=90.0)
    parser.add_argument("--maxflow-rounds", type=int, default=5)
    parser.add_argument("--maxflow-cuts-per-round", type=int, default=50)
    parser.add_argument("--maxflow-od-pairs", type=int, default=500)
    parser.add_argument("--validation-time-limit", type=float, default=60.0)
    parser.add_argument(
        "--initial-design",
        action="append",
        default=[],
        help="Optional design.json from another run; repeat to provide multiple starts.",
    )
    parser.add_argument(
        "--skip-incumbent-validation",
        action="store_true",
        help="Skip fixing the returned integer design in a fresh original model.",
    )
    return parser


def _status_name(status: int) -> str:
    for name in (
        "OPTIMAL",
        "INFEASIBLE",
        "INF_OR_UNBD",
        "UNBOUNDED",
        "TIME_LIMIT",
        "SUBOPTIMAL",
        "INTERRUPTED",
        "NUMERIC",
    ):
        if int(getattr(GRB, name, -999999)) == int(status):
            return name
    return str(status)


def _extract_model_design(model: MLNDModel) -> Dict[str, object]:
    return {
        "x": {key: float(var.X) for key, var in model.vars["x"].items()},
        "y": {key: float(var.X) for key, var in model.vars["y"].items()},
        "z": {key: float(var.X) for key, var in model.vars["z"].items()},
        "Gamma": float(model.vars["Gamma"].X),
    }


def _normalized_design(design: Mapping[str, object]) -> Dict[str, object]:
    return {
        "x": {key: float(value) for key, value in design["x"].items()},
        "y": {key: float(value) for key, value in design["y"].items()},
        "z": {key: float(value) for key, value in design["z"].items()},
        "Gamma": float(design["Gamma"]),
    }


def save_design(path: str, design: Mapping[str, object]) -> None:
    normalized = _normalized_design(design)
    payload = {
        "x": normalized["x"],
        "y": [
            {"u": u, "v": v, "value": value}
            for (u, v), value in normalized["y"].items()
        ],
        "z": [
            {"hub": hub, "level": level, "value": value}
            for (hub, level), value in normalized["z"].items()
        ],
        "Gamma": normalized["Gamma"],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def load_design(path: str) -> Dict[str, object]:
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    return {
        "x": {str(key): float(value) for key, value in payload.get("x", {}).items()},
        "y": {
            (row["u"], row["v"]): float(row["value"])
            for row in payload.get("y", [])
        },
        "z": {
            (row["hub"], int(row["level"])): float(row["value"])
            for row in payload.get("z", [])
        },
        "Gamma": float(payload.get("Gamma", 0.0)),
    }


def validate_design(
    instance,
    design: Mapping[str, object],
    *,
    time_limit: float,
    relax_gamma: bool = False,
) -> Dict[str, object]:
    """Fix first-stage integer variables in a fresh uncut model."""
    start = time.perf_counter()
    model = MLNDModel(instance)
    model.build()
    model.model.Params.OutputFlag = 0
    model.model.Params.Threads = 1
    model.model.Params.TimeLimit = max(0.01, float(time_limit))

    normalized = _normalized_design(design)
    for key, var in model.vars["x"].items():
        value = float(round(normalized["x"].get(key, 0.0)))
        var.LB = value
        var.UB = value
    for key, var in model.vars["y"].items():
        value = float(round(normalized["y"].get(key, 0.0)))
        var.LB = value
        var.UB = value
    for key, var in model.vars["z"].items():
        value = float(round(normalized["z"].get(key, 0.0)))
        var.LB = value
        var.UB = value
    if relax_gamma:
        model.vars["Gamma"].VType = GRB.CONTINUOUS
        model.model.update()
        gamma = float(normalized["Gamma"])
    else:
        gamma = float(round(normalized["Gamma"]))
    model.vars["Gamma"].LB = gamma
    model.vars["Gamma"].UB = gamma

    model.model.optimize()
    return {
        "validation_status": int(model.model.Status),
        "validation_status_name": _status_name(int(model.model.Status)),
        "validation_sol_count": int(model.model.SolCount),
        "validation_obj": (
            float(model.model.ObjVal) if int(model.model.SolCount) > 0 else None
        ),
        "validation_runtime": float(model.model.Runtime),
        "validation_wall_time": time.perf_counter() - start,
    }


def _base_summary(instance, instance_dir: str, method: str, args) -> Dict[str, object]:
    return {
        "method": method,
        "model_variant": args.model_variant,
        "instance_dir": instance_dir,
        "period_id": args.period_id,
        "H": len(instance.H),
        "L": len(instance.L),
        "S": len(instance.S),
        "E": len(instance.E),
        "OD": len(instance.h),
        "A": len(instance.A),
        "A_F": len(instance.A_F),
        "demand": sum(float(value) for value in instance.h.values()),
        "time_limit": float(args.time_limit),
        "mip_gap_target": float(args.mip_gap),
        "threads": int(args.threads),
    }


def run_gurobi(instance, instance_dir: str, method: str, args, result_dir: str):
    summary = _base_summary(instance, instance_dir, method, args)
    model = MLNDModel(instance)
    model.build()
    model.model.Params.OutputFlag = 1
    model.model.Params.Threads = int(args.threads)
    model.model.Params.MIPGap = float(args.mip_gap)
    if method == "gurobi_numeric":
        model.model.Params.NumericFocus = 3
        model.model.Params.ScaleFlag = 2
        model.model.Params.FeasibilityTol = 1e-7
        model.model.Params.OptimalityTol = 1e-7
        model.model.Params.IntFeasTol = 1e-7
        model.model.Params.BarQCPConvTol = 1e-8

    cuts = []
    cut_stats = None
    method_start = time.perf_counter()
    if method == "gurobi_maxflow":
        cut_budget = min(float(args.maxflow_time_limit), float(args.time_limit))
        cuts, cut_stats = add_root_max_flow_cuts(
            model,
            time_limit=cut_budget,
            max_rounds=args.maxflow_rounds,
            max_cuts_per_round=args.maxflow_cuts_per_round,
            max_od_pairs=args.maxflow_od_pairs,
            output_flag=0,
            threads=1,
        )
        with open(os.path.join(result_dir, "maxflow_cuts.json"), "w", encoding="utf-8") as f:
            json.dump(
                [
                    {
                        "origin": cut.origin,
                        "destination": cut.destination,
                        "demand": cut.demand,
                        "crossing_arcs": list(cut.crossing_arcs),
                        "violation": cut.violation,
                        "round_index": cut.round_index,
                    }
                    for cut in cuts
                ],
                f,
                indent=2,
            )

    elapsed = time.perf_counter() - method_start
    solve_budget = max(0.01, float(args.time_limit) - elapsed)
    model.model.Params.TimeLimit = solve_budget
    solve_start = time.perf_counter()
    model.model.optimize()
    solve_wall = time.perf_counter() - solve_start
    total_wall = time.perf_counter() - method_start

    sol_count = int(model.model.SolCount)
    obj = float(model.model.ObjVal) if sol_count > 0 else None
    bound = float(model.model.ObjBound)
    gap = float(model.model.MIPGap) if sol_count > 0 else None
    design = _extract_model_design(model) if sol_count > 0 else None
    summary.update(
        {
            "status": int(model.model.Status),
            "status_name": _status_name(int(model.model.Status)),
            "sol_count": sol_count,
            "obj": obj,
            "bound": bound,
            "gap": gap,
            "preprocess_time": elapsed,
            "solver_runtime": float(model.model.Runtime),
            "solve_wall_time": solve_wall,
            "wall_time": total_wall,
            "iterations": None,
            "cuts": len(cuts),
        }
    )
    if cut_stats is not None:
        summary.update({f"maxflow_{key}": value for key, value in cut_stats.to_dict().items()})
    return summary, design


def run_benders(instance, instance_dir: str, method: str, args):
    summary = _base_summary(instance, instance_dir, method, args)
    if method == "benders":
        solver = BendersRegimeIISolver(instance, use_pareto_cuts=True)
    elif method == "experimental_benders":
        solver = ExperimentalBendersRegimeIISolver(
            instance,
            use_pareto_cuts=True,
            use_eta_origin_split=True,
            use_persistent_subproblem=True,
            use_local_branching=True,
        )
    elif method == "partial_benders":
        solver = SinglePeriodPartialBendersRegimeIISolver(
            instance,
            max_iter=args.benders_max_iterations,
            gap_tol=args.benders_tolerance,
            master_time_limit=args.partial_master_time_limit,
            master_mip_gap=args.mip_gap,
            keep_route_feasibility=args.partial_keep_route_feasibility,
            embed_route_cost=args.partial_embed_route_cost,
            validate_incumbents=True,
            verbose=True,
        )
    else:
        raise ValueError(method)

    for design_path in args.initial_design:
        solver.add_initial_design(load_design(design_path))

    start = time.perf_counter()
    preprocess_time = 0.0
    cut_stats = None
    warmstart_obj = None
    warmstart_status = None
    warmstart_mode = args.partial_warmstart_mode
    if warmstart_mode == "auto":
        warmstart_mode = "numeric" if len(instance.H) >= 60 else "maxflow"
    warmstart_limit = args.partial_warmstart_time_limit
    if warmstart_limit is None:
        warmstart_limit = 240.0 if warmstart_mode == "numeric" else 60.0

    if method == "partial_benders" and warmstart_limit > 0.0:
        warm_budget = min(
            float(warmstart_limit),
            float(args.time_limit),
        )
        warm_model = MLNDModel(instance)
        warm_model.build()
        if warmstart_mode == "maxflow":
            warm_cut_budget = min(
                float(args.maxflow_time_limit),
                0.20 * warm_budget,
            )
            add_root_max_flow_cuts(
                warm_model,
                time_limit=warm_cut_budget,
                max_rounds=args.maxflow_rounds,
                max_cuts_per_round=args.maxflow_cuts_per_round,
                max_od_pairs=args.maxflow_od_pairs,
                output_flag=0,
                threads=1,
            )
        warm_elapsed = time.perf_counter() - start
        warm_model.model.Params.OutputFlag = 0
        warm_model.model.Params.Threads = int(args.threads)
        warm_model.model.Params.MIPFocus = 1
        warm_model.model.Params.MIPGap = float(args.mip_gap)
        if warmstart_mode == "numeric":
            warm_model.model.Params.NumericFocus = 3
            warm_model.model.Params.ScaleFlag = 2
            warm_model.model.Params.FeasibilityTol = 1e-7
            warm_model.model.Params.OptimalityTol = 1e-7
            warm_model.model.Params.IntFeasTol = 1e-7
            warm_model.model.Params.BarQCPConvTol = 1e-8
        warm_model.model.Params.TimeLimit = max(0.01, warm_budget - warm_elapsed)
        warm_model.model.optimize()
        warmstart_status = _status_name(int(warm_model.model.Status))
        if int(warm_model.model.SolCount) > 0:
            warmstart_obj = float(warm_model.model.ObjVal)
            solver.add_initial_design(_extract_model_design(warm_model))
        preprocess_time = time.perf_counter() - start

    if method == "partial_benders" and not args.no_partial_maxflow_cuts:
        solver.build_partial_master()
        adapter = SimpleNamespace(
            _built=True,
            model=solver.partial_master,
            vars=solver.partial_vars,
            inst=solver.inst,
            HLS=solver.HLS,
            A_hat=solver.A_hat,
            A_HS=solver.A_HS,
            A_SH=solver.A_SH,
            A_HH=solver.A_HH,
        )
        _, cut_stats = add_root_max_flow_cuts(
            adapter,
            time_limit=max(
                0.0,
                min(
                    args.maxflow_time_limit,
                    args.time_limit - (time.perf_counter() - start),
                ),
            ),
            max_rounds=args.maxflow_rounds,
            max_cuts_per_round=args.maxflow_cuts_per_round,
            max_od_pairs=args.maxflow_od_pairs,
            output_flag=0,
            threads=1,
        )
        preprocess_time = time.perf_counter() - start

    remaining_time = max(0.01, float(args.time_limit) - preprocess_time)
    success = solver.solve(
        max_iterations=args.benders_max_iterations,
        tolerance=args.benders_tolerance,
        time_limit=remaining_time,
    )
    wall_time = time.perf_counter() - start
    history = list(solver.history)
    last = history[-1] if history else {}
    obj = float(solver.best_total_obj) if success else None
    bound = last.get("UB")
    if bound is not None:
        bound = float(bound)
    gap = last.get("rel_gap")
    if gap is not None:
        gap = float(gap)
    summary.update(
        {
            "status": None,
            "status_name": "FEASIBLE" if success else "NO_INCUMBENT",
            "sol_count": 1 if success else 0,
            "obj": obj,
            "bound": bound,
            "gap": gap,
            "preprocess_time": preprocess_time,
            "solver_runtime": wall_time - preprocess_time,
            "solve_wall_time": wall_time - preprocess_time,
            "wall_time": wall_time,
            "iterations": len(history),
            "cuts": int(last.get("cuts", getattr(solver, "cut_count", 0))),
        }
    )
    if cut_stats is not None:
        summary.update({f"maxflow_{key}": value for key, value in cut_stats.to_dict().items()})
    summary["warmstart_obj"] = warmstart_obj
    summary["warmstart_status"] = warmstart_status
    summary["warmstart_mode"] = (
        warmstart_mode if method == "partial_benders" else None
    )
    design = _normalized_design(solver.best_design) if success else None
    return summary, design, history


def _write_csv(path: str, rows: Iterable[Mapping[str, object]]) -> None:
    rows = list(rows)
    if not rows:
        return
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: (
                        json.dumps(value, sort_keys=True)
                        if isinstance(value, (dict, list, tuple))
                        else value
                    )
                    for key, value in row.items()
                }
            )


def main() -> None:
    args = build_parser().parse_args()
    if args.time_limit <= 0:
        raise ValueError("--time-limit must be positive.")
    if args.threads <= 0:
        raise ValueError("--threads must be positive.")

    config_overrides = {
        "MODEL_TIME_LIMIT": float(args.time_limit),
        "MODEL_MIP_GAP": float(args.mip_gap),
        "BENDERS_TIME_LIMIT": float(args.time_limit),
        "BENDERS_MAX_ITER": int(args.benders_max_iterations),
        "BENDERS_TOL": float(args.benders_tolerance),
        "BENDERS_MASTER_THREADS": int(args.threads),
        "BENDERS_SUB_THREADS": 1,
        # The partial master already contains exact route feasibility.  Route
        # recourse slacks around 1e-5 are numerical noise from the LP/QCP
        # boundary; every accepted incumbent is still checked by the full
        # continuous subproblem and by a fresh original-model validation.
        "BENDERS_PARTIAL_SLACK_TOL": 1e-4,
        "BENDERS_PARTIAL_EXACT_SLACK_TOL": 1e-4,
        "BENDERS_PARTIAL_ACCEPT_SLACK_TOL": 1e-4,
        "BENDERS_PARTIAL_RECOURSE_FEAS_TOL": 1e-6,
        "BENDERS_PARTIAL_RECOURSE_NUMERIC_FOCUS": 2,
    }
    if args.model_variant == "no_battery_endurance":
        config_overrides["ENERGY_RATE_OMEGA"] = 0.0
    elif args.model_variant == "no_swapping_unlimited_direct":
        config_overrides["ENERGY_RATE_OMEGA"] = 0.0
        config_overrides["MAX_DIRECT_DISTANCE"] = 1e9

    all_summaries = []
    with override_config(**config_overrides):
        instance_dir = args.instance_dir or infer_instance_dir(
            args.series_root, args.hub_count
        )
        instance = DataLoader().load_instance_dir(
            instance_dir, period_id=args.period_id
        )
        if args.model_variant == "no_swapping_unlimited_direct":
            instance.E = []
            instance.build_network()

        scale_name = f"H{len(instance.H):03d}"
        if args.model_variant != "full":
            scale_name = f"{scale_name}_{args.model_variant}"
        scale_dir = os.path.join(args.results_root, scale_name)
        os.makedirs(scale_dir, exist_ok=True)

        for method in args.methods:
            result_dir = os.path.join(scale_dir, method)
            os.makedirs(result_dir, exist_ok=True)
            print(
                f"\n=== {method}: H={len(instance.H)}, L={len(instance.L)}, "
                f"E={len(instance.E)}, budget={args.time_limit:.0f}s ===",
                flush=True,
            )
            history = []
            if method in {"gurobi", "gurobi_numeric", "gurobi_maxflow"}:
                summary, design = run_gurobi(
                    instance, instance_dir, method, args, result_dir
                )
            else:
                summary, design, history = run_benders(
                    instance, instance_dir, method, args
                )

            if design is not None and not args.skip_incumbent_validation:
                validation = validate_design(
                    instance,
                    design,
                    time_limit=args.validation_time_limit,
                )
                summary.update(validation)
                if summary.get("obj") is not None and validation.get("validation_obj") is not None:
                    summary["validation_obj_diff"] = (
                        float(validation["validation_obj"]) - float(summary["obj"])
                    )

            if design is not None:
                save_design(os.path.join(result_dir, "design.json"), design)
            with open(os.path.join(result_dir, "summary.json"), "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, sort_keys=True)
            _write_csv(os.path.join(result_dir, "summary.csv"), [summary])
            if history:
                _write_csv(os.path.join(result_dir, "history.csv"), history)
            all_summaries.append(summary)
            print(
                f"{method}: status={summary['status_name']} obj={summary.get('obj')} "
                f"bound={summary.get('bound')} gap={summary.get('gap')} "
                f"wall={summary.get('wall_time'):.2f}s",
                flush=True,
            )

    gurobi_row = next(
        (row for row in all_summaries if row.get("method") == "gurobi"),
        None,
    )
    if gurobi_row is not None:
        gurobi_obj = gurobi_row.get("validation_obj")
        if gurobi_obj is None:
            gurobi_obj = gurobi_row.get("obj")
        if gurobi_obj is not None:
            gurobi_obj = float(gurobi_obj)
            for row in all_summaries:
                row_obj = row.get("validation_obj")
                if row_obj is None:
                    row_obj = row.get("obj")
                if row_obj is None:
                    continue
                improvement = float(row_obj) - gurobi_obj
                row["obj_vs_gurobi_abs"] = improvement
                row["obj_vs_gurobi_pct"] = improvement / max(1.0, abs(gurobi_obj))
                row["beats_gurobi_incumbent"] = improvement > 1e-5

    gurobi_candidates = []
    for row in all_summaries:
        if row.get("method") not in {"gurobi", "gurobi_numeric", "gurobi_maxflow"}:
            continue
        value = row.get("validation_obj")
        if value is None:
            value = row.get("obj")
        if value is not None:
            gurobi_candidates.append(float(value))
    if gurobi_candidates:
        best_gurobi_obj = max(gurobi_candidates)
        for row in all_summaries:
            row_obj = row.get("validation_obj")
            if row_obj is None:
                row_obj = row.get("obj")
            if row_obj is None:
                continue
            improvement = float(row_obj) - best_gurobi_obj
            row["obj_vs_best_gurobi_abs"] = improvement
            row["obj_vs_best_gurobi_pct"] = improvement / max(
                1.0, abs(best_gurobi_obj)
            )
            row["beats_best_gurobi_incumbent"] = improvement > 1e-5

    _write_csv(os.path.join(scale_dir, "comparison.csv"), all_summaries)
    with open(os.path.join(scale_dir, "comparison.json"), "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, indent=2, sort_keys=True)
    print(f"\nComparison: {os.path.join(scale_dir, 'comparison.csv')}")


if __name__ == "__main__":
    main()
