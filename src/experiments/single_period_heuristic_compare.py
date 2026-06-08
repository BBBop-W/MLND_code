import argparse
import csv
import os
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, Iterable, Optional

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src import config
from src.data_processor.generator import DataGenerator
from src.data_processor.instance import Instance
from src.data_processor.loader import DataLoader
from src.model.gurobi_model import MLNDModel
from src.model.lagrangian_heuristic import (
    STANDARD_HEURISTIC_METHODS,
    HeuristicRun,
    run_standard_heuristics,
)
from src.model.local_search_heuristic import LocalSearchSolution, RegimeIILocalSearch


@dataclass
class ExactResult:
    status: int
    obj: Optional[float]
    bound: Optional[float]
    gap: Optional[float]
    runtime: float
    wall_time: float
    sol_count: int
    warm_start: bool = False


@contextmanager
def override_config(**kwargs):
    old_values = {name: getattr(config, name) for name in kwargs}
    try:
        for name, value in kwargs.items():
            setattr(config, name, value)
        yield
    finally:
        for name, value in old_values.items():
            setattr(config, name, value)


def parse_size(raw: str) -> tuple[int, int, int]:
    parts = raw.lower().replace("h", "").replace("l", "").replace("e", "").split(":")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("size must be H:L:E, for example 8:2:2")
    h, l, e = (int(part) for part in parts)
    if min(h, l, e) <= 0:
        raise argparse.ArgumentTypeError("H, L, and E must be positive")
    if e < l:
        raise argparse.ArgumentTypeError("E must be >= L for the synthetic generator")
    return h, l, e


def scale_demands(inst: Instance, factor: float) -> None:
    if abs(factor - 1.0) <= 1e-12:
        return
    inst.h = {pair: float(value) * factor for pair, value in inst.h.items()}


def load_synthetic_instance(
    *,
    h: int,
    l: int,
    e: int,
    seed: int,
    output_dir: str,
    demand_scale: float,
) -> tuple[str, Instance]:
    name = f"synthetic_H{h}_L{l}_E{e}_seed{seed}"
    instance_dir = os.path.join(output_dir, "instances", name)
    with override_config(NUM_HUBS=h, NUM_BASES=l, NUM_SWAPPING=e):
        DataGenerator(seed=seed).generate_synthetic(instance_dir)
    inst = DataLoader().load_instance_dir(instance_dir)
    scale_demands(inst, demand_scale)
    return name, inst


def load_meituan_instance(period_id: str) -> tuple[str, Instance]:
    instance_dir = os.path.join(PROJECT_ROOT, "data", "instances", "Meituan")
    inst = DataLoader().load_instance_dir(instance_dir, period_id=period_id)
    return f"meituan_{period_id}", inst


def solve_exact(
    inst: Instance,
    *,
    time_limit: float,
    mip_gap: float,
    warm_start: Optional[LocalSearchSolution] = None,
) -> ExactResult:
    model = MLNDModel(inst)
    model.build()
    if warm_start is not None:
        for u, var in model.vars["x"].items():
            var.Start = warm_start.x.get(u, 0.0)
        for key, var in model.vars["y"].items():
            var.Start = warm_start.y.get(key, 0.0)
        for key, var in model.vars["z"].items():
            var.Start = warm_start.z.get(key, 0.0)
        if warm_start.gamma is not None:
            model.vars["Gamma"].Start = warm_start.gamma
    model.model.setParam("OutputFlag", 0)
    model.model.setParam("Threads", 1)
    model.model.setParam("TimeLimit", float(time_limit))
    model.model.setParam("MIPGap", float(mip_gap))
    start = time.perf_counter()
    model.model.optimize()
    wall = time.perf_counter() - start
    sol_count = int(model.model.SolCount)
    obj = float(model.model.ObjVal) if sol_count > 0 else None
    bound = None
    gap = None
    try:
        bound = float(model.model.ObjBound)
    except Exception:
        bound = None
    if sol_count > 0:
        try:
            gap = float(model.model.MIPGap)
        except Exception:
            gap = None
    return ExactResult(
        status=int(model.model.Status),
        obj=obj,
        bound=bound,
        gap=gap,
        runtime=float(model.model.Runtime),
        wall_time=wall,
        sol_count=sol_count,
        warm_start=warm_start is not None,
    )


def serialize_run(
    *,
    instance_name: str,
    inst: Instance,
    method: str,
    obj: Optional[float],
    runtime: float,
    evaluations: Optional[int],
    cache_hits: Optional[int],
    candidate_hubs: Optional[int],
    candidate_bases: Optional[int],
    candidate_swaps: Optional[int],
    gamma: Optional[float],
    status: Optional[int],
    exact: ExactResult,
    plain_exact: Optional[ExactResult] = None,
    warm_exact: Optional[ExactResult] = None,
) -> Dict[str, object]:
    gap_to_exact = None
    if exact.obj is not None and obj is not None:
        gap_to_exact = (exact.obj - obj) / max(1.0, abs(exact.obj))
    return {
        "instance": instance_name,
        "H": len(inst.H),
        "L": len(inst.L),
        "S": len(inst.S),
        "E": len(inst.E),
        "OD": len(inst.h),
        "demand": sum(float(v) for v in inst.h.values()),
        "method": method,
        "obj": obj,
        "gap_to_gurobi_incumbent": gap_to_exact,
        "runtime": runtime,
        "evaluations": evaluations,
        "cache_hits": cache_hits,
        "open_hubs": candidate_hubs,
        "open_bases": candidate_bases,
        "open_swaps": candidate_swaps,
        "open_hub_ids": "",
        "open_base_ids": "",
        "open_swap_ids": "",
        "gamma": gamma,
        "status": status,
        "gurobi_obj": exact.obj,
        "gurobi_bound": exact.bound,
        "gurobi_gap": exact.gap,
        "gurobi_status": exact.status,
        "gurobi_runtime": exact.runtime,
        "gurobi_wall_time": exact.wall_time,
        "gurobi_sol_count": exact.sol_count,
        "gurobi_warm_start": exact.warm_start,
        "gurobi_plain_obj": plain_exact.obj if plain_exact is not None else None,
        "gurobi_plain_bound": plain_exact.bound if plain_exact is not None else None,
        "gurobi_plain_gap": plain_exact.gap if plain_exact is not None else None,
        "gurobi_plain_status": plain_exact.status if plain_exact is not None else None,
        "gurobi_plain_wall_time": plain_exact.wall_time if plain_exact is not None else None,
        "gurobi_warm_obj": warm_exact.obj if warm_exact is not None else None,
        "gurobi_warm_bound": warm_exact.bound if warm_exact is not None else None,
        "gurobi_warm_gap": warm_exact.gap if warm_exact is not None else None,
        "gurobi_warm_status": warm_exact.status if warm_exact is not None else None,
        "gurobi_warm_wall_time": warm_exact.wall_time if warm_exact is not None else None,
    }


def run_heuristics(inst: Instance, args) -> list[HeuristicRun]:
    methods = list(args.methods)
    standard_methods = [method for method in methods if method in STANDARD_HEURISTIC_METHODS]
    runs = []
    if standard_methods:
        runs = run_standard_heuristics(
            inst,
            methods=standard_methods,
            eval_time_limit=args.eval_time_limit,
            eval_mip_gap=args.eval_mip_gap,
            constructive_candidate_hubs=args.constructive_candidate_hubs,
            constructive_candidate_swaps=args.constructive_candidate_swaps,
            constructive_eval_top=args.constructive_eval_top,
            lr_iterations=args.lr_iterations,
            lr_eval_top=args.lr_eval_top,
            lr_decomposition=args.lr_decomposition,
            tabu_iterations=args.tabu_iterations,
            tabu_tenure=args.tabu_tenure,
            tabu_candidate_hubs=args.tabu_candidate_hubs,
            tabu_candidate_swaps=args.tabu_candidate_swaps,
            tabu_max_neighbors=args.tabu_max_neighbors,
            max_hubs=args.max_hubs,
            max_swaps=args.max_swaps,
            verbose=args.verbose,
        )

    if "local_search" in methods:
        local = RegimeIILocalSearch(
            inst,
            eval_time_limit=args.eval_time_limit,
            eval_mip_gap=args.eval_mip_gap,
            max_seed_hubs=args.local_seed_hubs,
            max_seed_candidates=args.local_seed_candidates,
            max_iter=args.local_iterations,
            verbose=args.verbose,
        ).solve()
        runs.append(
            HeuristicRun(
                name="greedy_local_search",
                solution=local,
                wall_time=local.wall_time,
                evaluations=local.evaluations,
                cache_hits=local.cache_hits,
                history=[],
            )
        )
    return runs


def heuristic_rows(
    *,
    instance_name: str,
    inst: Instance,
    exact: ExactResult,
    plain_exact: Optional[ExactResult],
    warm_exact: Optional[ExactResult],
    runs: list[HeuristicRun],
) -> list[Dict[str, object]]:
    rows: list[Dict[str, object]] = []
    for run in runs:
        sol = run.solution
        row = serialize_run(
            instance_name=instance_name,
            inst=inst,
            method=run.name,
            obj=sol.obj,
            runtime=run.wall_time,
            evaluations=run.evaluations,
            cache_hits=run.cache_hits,
            candidate_hubs=len(sol.candidate.hubs),
            candidate_bases=len(sol.candidate.bases),
            candidate_swaps=len(sol.candidate.swaps),
            gamma=sol.gamma,
            status=sol.status,
            exact=exact,
            plain_exact=plain_exact,
            warm_exact=warm_exact,
        )
        row["open_hub_ids"] = ";".join(sorted(sol.candidate.hubs))
        row["open_base_ids"] = ";".join(sorted(sol.candidate.bases))
        row["open_swap_ids"] = ";".join(sorted(sol.candidate.swaps))
        rows.append(row)
    return rows


def write_rows(path: str, rows: Iterable[Dict[str, object]]) -> None:
    rows = list(rows)
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run(args) -> list[Dict[str, object]]:
    all_rows: list[Dict[str, object]] = []
    instances: list[tuple[str, Instance]] = []
    for h, l, e in args.sizes:
        instances.append(
            load_synthetic_instance(
                h=h,
                l=l,
                e=e,
                seed=args.seed,
                output_dir=args.output_dir,
                demand_scale=args.synthetic_demand_scale,
            )
        )
    if args.include_meituan:
        instances.append(load_meituan_instance(args.meituan_period))

    for instance_name, inst in instances:
        print(
            f"\n=== {instance_name}: H={len(inst.H)} L={len(inst.L)} "
            f"E={len(inst.E)} OD={len(inst.h)} demand={sum(inst.h.values()):.2f} ==="
        )
        plain_exact = solve_exact(inst, time_limit=args.gurobi_time_limit, mip_gap=args.gurobi_mip_gap)
        print(
            f"Gurobi cold status={plain_exact.status} obj={plain_exact.obj} "
            f"bound={plain_exact.bound} gap={plain_exact.gap} wall={plain_exact.wall_time:.2f}s"
        )
        runs = run_heuristics(inst, args)
        best_start = max(
            (run.solution for run in runs),
            key=lambda sol: -float("inf") if sol.obj is None else float(sol.obj),
        )
        warm_exact = solve_exact(
            inst,
            time_limit=args.gurobi_time_limit,
            mip_gap=args.gurobi_mip_gap,
            warm_start=best_start,
        )
        exact = warm_exact
        if plain_exact.obj is not None and (warm_exact.obj is None or plain_exact.obj > warm_exact.obj):
            exact = plain_exact
        print(
            f"Gurobi warm status={warm_exact.status} obj={warm_exact.obj} "
            f"bound={warm_exact.bound} gap={warm_exact.gap} wall={warm_exact.wall_time:.2f}s"
        )
        all_rows.extend(
            heuristic_rows(
                instance_name=instance_name,
                inst=inst,
                exact=exact,
                plain_exact=plain_exact,
                warm_exact=warm_exact,
                runs=runs,
            )
        )
        write_rows(args.output_csv, all_rows)
        for row in all_rows:
            if row["instance"] != instance_name:
                continue
            print(
                f"{row['method']}: obj={row['obj']} gap={row['gap_to_gurobi_incumbent']} "
                f"time={float(row['runtime']):.2f}s evals={row['evaluations']}"
            )
    return all_rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare single-period Regime II heuristics against monolithic Gurobi."
    )
    parser.add_argument("--sizes", nargs="*", type=parse_size, default=[(5, 1, 1), (8, 2, 2)])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default=os.path.join(PROJECT_ROOT, "results", "heuristic_compare"))
    parser.add_argument(
        "--output-csv",
        default=os.path.join(PROJECT_ROOT, "results", "heuristic_compare", "summary.csv"),
    )
    parser.add_argument("--synthetic-demand-scale", type=float, default=20.0)
    parser.add_argument("--include-meituan", action="store_true")
    parser.add_argument("--meituan-period", default=f"h{config.MEITUAN_DEMAND_HOUR:02d}")
    parser.add_argument("--gurobi-time-limit", type=float, default=30.0)
    parser.add_argument("--gurobi-mip-gap", type=float, default=0.01)
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=list(STANDARD_HEURISTIC_METHODS) + ["local_search"],
        default=list(STANDARD_HEURISTIC_METHODS),
        help="Heuristic methods to run. Defaults to the three standardized methods.",
    )
    parser.add_argument("--eval-time-limit", type=float, default=3.0)
    parser.add_argument("--eval-mip-gap", type=float, default=0.05)
    parser.add_argument("--constructive-candidate-hubs", type=int, default=15)
    parser.add_argument("--constructive-candidate-swaps", type=int, default=10)
    parser.add_argument("--constructive-eval-top", type=int, default=25)
    parser.add_argument("--lr-iterations", type=int, default=35)
    parser.add_argument("--lr-eval-top", type=int, default=25)
    parser.add_argument("--lr-decomposition", choices=["coverage", "route_swap"], default="route_swap")
    parser.add_argument("--tabu-iterations", type=int, default=15)
    parser.add_argument("--tabu-tenure", type=int, default=5)
    parser.add_argument("--tabu-candidate-hubs", type=int, default=15)
    parser.add_argument("--tabu-candidate-swaps", type=int, default=10)
    parser.add_argument("--tabu-max-neighbors", type=int, default=80)
    parser.add_argument("--max-hubs", type=int, default=None)
    parser.add_argument("--max-swaps", type=int, default=None)
    parser.add_argument("--local-seed-hubs", type=int, default=4)
    parser.add_argument("--local-seed-candidates", type=int, default=150)
    parser.add_argument("--local-iterations", type=int, default=5)
    parser.add_argument("--verbose", action="store_true")
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
