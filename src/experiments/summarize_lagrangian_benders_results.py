"""Summarize strict Lagrangian/Benders benchmarks and convergence checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Dict, Iterable, Optional


PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)


def parse_int_list(raw: str) -> list[int]:
    values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("At least one hub count is required.")
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build paper-facing strict-vs-Gurobi benchmark tables."
    )
    parser.add_argument(
        "--strict-root",
        default=os.path.join(
            PROJECT_ROOT,
            "results",
            "strict_lagrangian_benders_benchmark_900s",
        ),
    )
    parser.add_argument(
        "--baseline-root",
        default=os.path.join(
            PROJECT_ROOT,
            "results",
            "regime_ii_algorithm_benchmark_900s",
        ),
    )
    parser.add_argument(
        "--hub-counts",
        type=parse_int_list,
        default=parse_int_list("20,40,80"),
    )
    parser.add_argument(
        "--checkpoints",
        type=parse_int_list,
        default=parse_int_list("180,300,600,900"),
    )
    parser.add_argument("--output-dir", default=None)
    return parser


def read_json(path: str) -> Optional[Dict[str, object]]:
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def objective_value(summary: Optional[Dict[str, object]]) -> Optional[float]:
    if not summary:
        return None
    value = summary.get("validation_obj")
    if value is None:
        value = summary.get("obj")
    return None if value is None else float(value)


def relative_difference(
    value: Optional[float], reference: Optional[float]
) -> Optional[float]:
    if value is None or reference is None:
        return None
    return (value - reference) / max(1.0, abs(reference))


def write_csv(path: str, rows: Iterable[Dict[str, object]]) -> None:
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
        writer.writerows(rows)


def load_method_summary(
    root: str, hub_count: int, method: str
) -> Optional[Dict[str, object]]:
    return read_json(
        os.path.join(
            root,
            f"H{hub_count:03d}",
            method,
            "summary.json",
        )
    )


def benchmark_rows(args) -> list[Dict[str, object]]:
    rows = []
    for hub_count in args.hub_counts:
        strict = load_method_summary(
            args.strict_root, hub_count, "strict_lagrangian_benders"
        )
        if strict is None:
            continue
        gurobi = load_method_summary(args.baseline_root, hub_count, "gurobi")
        if gurobi is None:
            gurobi = load_method_summary(
                args.strict_root, hub_count, "gurobi"
            )
        numeric = load_method_summary(
            args.baseline_root, hub_count, "gurobi_numeric"
        )
        if numeric is None:
            numeric = load_method_summary(
                args.strict_root, hub_count, "gurobi_numeric"
            )
        strict_obj = objective_value(strict)
        gurobi_obj = objective_value(gurobi)
        numeric_obj = objective_value(numeric)
        baseline_values = [
            value for value in (gurobi_obj, numeric_obj) if value is not None
        ]
        best_original = max(baseline_values) if baseline_values else None

        strict_bound = strict.get("bound")
        strict_bound = (
            None if strict_bound is None else float(strict_bound)
        )
        best_known_gap = (
            None
            if strict_bound is None or best_original is None
            else (strict_bound - best_original)
            / max(1.0, abs(best_original))
        )
        rows.append(
            {
                "H": hub_count,
                "strict_obj": strict_obj,
                "strict_bound": strict_bound,
                "strict_gap": strict.get("gap"),
                "strict_wall_time": strict.get("wall_time"),
                "strict_global_bound": strict.get("global_network_bound"),
                "strict_lr_bound": strict.get("lr_bound"),
                "gurobi_obj": gurobi_obj,
                "gurobi_bound": None if gurobi is None else gurobi.get("bound"),
                "gurobi_gap": None if gurobi is None else gurobi.get("gap"),
                "gurobi_numeric_obj": numeric_obj,
                "gurobi_numeric_bound": (
                    None if numeric is None else numeric.get("bound")
                ),
                "gurobi_numeric_gap": (
                    None if numeric is None else numeric.get("gap")
                ),
                "best_original_obj": best_original,
                "strict_vs_gurobi_pct": relative_difference(
                    strict_obj, gurobi_obj
                ),
                "strict_vs_best_original_pct": relative_difference(
                    strict_obj, best_original
                ),
                "strict_bound_vs_best_original_pct": best_known_gap,
                "master_qconstrs": strict.get("master_qconstrs"),
            }
        )
    return rows


def convergence_rows(args) -> list[Dict[str, object]]:
    rows = []
    for hub_count in args.hub_counts:
        path = os.path.join(
            args.strict_root,
            f"H{hub_count:03d}",
            "strict_lagrangian_benders",
            "history.csv",
        )
        if not os.path.exists(path):
            continue
        with open(path, newline="", encoding="utf-8") as f:
            history = list(csv.DictReader(f))
        if not history:
            continue
        for checkpoint in args.checkpoints:
            eligible = [
                row
                for row in history
                if row.get("time")
                and float(row["time"]) <= float(checkpoint)
            ]
            row = eligible[-1] if eligible else history[0]
            ub = float(row["UB"])
            lb = float(row["LB"])
            rows.append(
                {
                    "H": hub_count,
                    "checkpoint": checkpoint,
                    "iteration": int(row["iter"]),
                    "time": float(row["time"]),
                    "UB": ub,
                    "LB": lb,
                    "abs_gap": ub - lb,
                    "rel_gap": (ub - lb) / max(1.0, abs(lb)),
                    "cuts": int(row.get("cuts") or 0),
                }
            )
    return rows


def ablation_rows(args) -> list[Dict[str, object]]:
    methods = (
        "strict_lagrangian_benders",
        "strict_no_lr",
        "strict_no_network_milp",
        "strict_no_global_ub",
        "strict_no_flow_cuts",
        "strict_topology_only",
    )
    rows = []
    for hub_count in args.hub_counts:
        full = load_method_summary(
            args.strict_root, hub_count, "strict_lagrangian_benders"
        )
        full_obj = objective_value(full)
        for method in methods:
            summary = load_method_summary(
                args.strict_root, hub_count, method
            )
            if summary is None:
                continue
            value = objective_value(summary)
            rows.append(
                {
                    "H": hub_count,
                    "method": method,
                    "obj": value,
                    "bound": summary.get("bound"),
                    "gap": summary.get("gap"),
                    "wall_time": summary.get("wall_time"),
                    "obj_vs_full_pct": relative_difference(
                        value, full_obj
                    ),
                    "global_network_bound": summary.get(
                        "global_network_bound"
                    ),
                    "lr_bound": summary.get("lr_bound"),
                    "iterations": summary.get("iterations"),
                    "cuts": summary.get("cuts"),
                    "topology_cuts": summary.get("topology_cuts"),
                    "swap_reachability_cuts": summary.get(
                        "swap_reachability_cuts"
                    ),
                    "screened_iterations": summary.get(
                        "screened_iterations"
                    ),
                    "master_qconstrs": summary.get("master_qconstrs"),
                }
            )
    return rows


def main() -> None:
    args = build_parser().parse_args()
    output_dir = args.output_dir or args.strict_root
    os.makedirs(output_dir, exist_ok=True)
    benchmark_path = os.path.join(output_dir, "paper_comparison.csv")
    convergence_path = os.path.join(
        output_dir, "convergence_checkpoints.csv"
    )
    ablation_path = os.path.join(output_dir, "ablation_comparison.csv")
    write_csv(benchmark_path, benchmark_rows(args))
    write_csv(convergence_path, convergence_rows(args))
    write_csv(ablation_path, ablation_rows(args))
    print(f"Comparison: {benchmark_path}")
    print(f"Convergence: {convergence_path}")
    print(f"Ablation: {ablation_path}")


if __name__ == "__main__":
    main()
