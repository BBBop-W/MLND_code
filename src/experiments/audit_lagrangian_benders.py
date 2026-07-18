"""Reproducible completion audit for the strict Lagrangian/Benders workflow."""

from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Dict


PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

from src.data_processor.loader import DataLoader
from src.experiments.benchmark_regime_ii_algorithms import load_design
from src.model.lagrangian_benders import StrictLagrangianBendersSolver


def parse_int_list(raw: str) -> list[int]:
    values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("At least one hub count is required.")
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit strict formulation and stored benchmark evidence."
    )
    parser.add_argument(
        "--series-root",
        default=os.path.join(
            PROJECT_ROOT, "data", "instances", "Meituan_H_series"
        ),
    )
    parser.add_argument(
        "--results-root",
        default=os.path.join(
            PROJECT_ROOT,
            "results",
            "strict_lagrangian_benders_benchmark_900s",
        ),
    )
    parser.add_argument(
        "--hub-counts",
        type=parse_int_list,
        default=parse_int_list("10,20,40"),
    )
    parser.add_argument("--period-id", default="h12")
    parser.add_argument("--wall-time-slack", type=float, default=2.0)
    parser.add_argument("--output", default=None)
    return parser


def instance_dir(root: str, hub_count: int) -> str:
    prefix = f"Meituan_H{hub_count:03d}_"
    matches = sorted(
        name
        for name in os.listdir(root)
        if name.startswith(prefix)
        and os.path.isdir(os.path.join(root, name))
    )
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one instance for H={hub_count}; found {matches}"
        )
    return os.path.join(root, matches[0])


def read_json(path: str) -> Dict[str, object]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def initial_sources(path: str) -> list[str]:
    with open(path, newline="", encoding="utf-8") as f:
        return [str(row["source"]) for row in csv.DictReader(f)]


def audit_scale(args, hub_count: int) -> Dict[str, object]:
    result_dir = os.path.join(
        args.results_root,
        f"H{hub_count:03d}",
        "strict_lagrangian_benders",
    )
    summary = read_json(os.path.join(result_dir, "summary.json"))
    design = load_design(os.path.join(result_dir, "design.json"))
    instance = DataLoader().load_instance_dir(
        instance_dir(args.series_root, hub_count),
        period_id=args.period_id,
    )
    solver = StrictLagrangianBendersSolver(instance, verbose=False)
    solver.build_master()
    names = [var.VarName.lower() for var in solver.master.getVars()]
    exact = solver.solve_subproblem(
        design, label=f"audit_exact_{hub_count}"
    )
    topology = solver.solve_topology_profit_estimator(
        design, label=f"audit_topology_{hub_count}"
    )
    swapping = solver.solve_swap_profit_estimator(
        design, label=f"audit_swapping_{hub_count}"
    )

    validation_obj = float(summary["validation_obj"])
    global_bound = float(summary["global_network_bound"])
    checks = {
        "master_has_no_qconstrs": int(solver.master.NumQConstrs) == 0,
        "master_has_no_psi": not any(
            name.startswith("psi") for name in names
        ),
        "master_has_no_queue_vars": not any(
            "queue" in name for name in names
        ),
        "master_gamma_is_integer": solver.master_vars["Gamma"].VType == "I",
        "algorithm_time_limit_is_900": float(summary["time_limit"]) == 900.0,
        "algorithm_wall_time_within_slack": (
            float(summary["wall_time"])
            <= 900.0 + float(args.wall_time_slack)
        ),
        "validation_matches_incumbent": abs(
            validation_obj - float(summary["obj"])
        )
        <= 1e-2,
        "global_bound_dominates_incumbent": global_bound + 1e-6
        >= validation_obj,
        "topology_lp_dominates_exact": (
            topology.obj_value is not None
            and exact.obj_value is not None
            and float(topology.obj_value) + 1e-5
            >= float(exact.obj_value)
        ),
        "swap_lp_dominates_exact": (
            swapping.obj_value is not None
            and exact.obj_value is not None
            and float(swapping.obj_value) + 1e-5
            >= float(exact.obj_value)
        ),
        "initial_designs_are_lagrangian": all(
            source == "lagrangian"
            for source in initial_sources(
                os.path.join(result_dir, "initial_design_history.csv")
            )
        ),
    }
    return {
        "H": hub_count,
        "checks": checks,
        "passed": all(checks.values()),
        "exact_recourse": exact.obj_value,
        "topology_estimator": topology.obj_value,
        "swap_estimator": swapping.obj_value,
        "strict_obj": summary["obj"],
        "validation_obj": validation_obj,
        "strict_bound": summary["bound"],
        "wall_time": summary["wall_time"],
    }


def main() -> None:
    args = build_parser().parse_args()
    strict_source = os.path.join(
        PROJECT_ROOT, "src", "model", "lagrangian_benders.py"
    )
    with open(strict_source, encoding="utf-8") as f:
        source = f.read()
    source_checks = {
        "strict_solver_does_not_import_original_model": "MLNDModel"
        not in source,
        "global_relaxation_incumbent_not_added_as_start": (
            "add_initial_design(self.global_network_incumbent)" not in source
        ),
    }
    scales = [audit_scale(args, value) for value in args.hub_counts]
    payload = {
        "source_checks": source_checks,
        "scales": scales,
        "passed": all(source_checks.values())
        and all(row["passed"] for row in scales),
    }
    output = args.output or os.path.join(
        args.results_root, "verification_audit.json"
    )
    with open(output, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not payload["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
