"""Run the Regime-II algorithm benchmark over multiple H scales."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.experiments.benchmark_regime_ii_algorithms import (
    METHODS,
    MODEL_VARIANTS,
    parse_methods,
)


def parse_int_list(raw: str) -> list[int]:
    values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not values or min(values) <= 0:
        raise argparse.ArgumentTypeError("Provide a comma-separated list of positive H values.")
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run model-based Regime-II algorithms on an H-scale grid."
    )
    parser.add_argument(
        "--series-root",
        default=os.path.join(PROJECT_ROOT, "data", "instances", "Meituan_H_series"),
    )
    parser.add_argument(
        "--hub-counts",
        type=parse_int_list,
        default=parse_int_list("5,10,20,40,60,80,100"),
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
    parser.add_argument("--logs-root", default=None)
    parser.add_argument("--period-id", default="h12")
    parser.add_argument(
        "--model-variant",
        choices=MODEL_VARIANTS,
        default="full",
    )
    parser.add_argument("--time-limit", type=float, default=900.0)
    parser.add_argument("--mip-gap", type=float, default=0.001)
    parser.add_argument("--benders-tolerance", type=float, default=0.0001)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--parallel-runs", type=int, default=1)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--maxflow-time-limit", type=float, default=90.0)
    parser.add_argument("--partial-master-time-limit", type=float, default=30.0)
    parser.add_argument("--partial-warmstart-time-limit", type=float, default=None)
    parser.add_argument(
        "--partial-warmstart-mode",
        choices=("auto", "maxflow", "numeric"),
        default="auto",
    )
    parser.add_argument("--partial-embed-route-cost", action="store_true")
    parser.add_argument("--validation-time-limit", type=float, default=60.0)
    parser.add_argument("--skip-incumbent-validation", action="store_true")
    return parser


def run_one(args, hub_count: int, method: str):
    logs_root = args.logs_root or os.path.join(args.results_root, "logs")
    os.makedirs(logs_root, exist_ok=True)
    command = [
        args.python,
        "-m",
        "src.experiments.benchmark_regime_ii_algorithms",
        "--series-root",
        args.series_root,
        "--hub-count",
        str(hub_count),
        "--period-id",
        args.period_id,
        "--model-variant",
        args.model_variant,
        "--methods",
        method,
        "--results-root",
        args.results_root,
        "--time-limit",
        str(args.time_limit),
        "--mip-gap",
        str(args.mip_gap),
        "--benders-tolerance",
        str(args.benders_tolerance),
        "--threads",
        str(args.threads),
        "--maxflow-time-limit",
        str(args.maxflow_time_limit),
        "--partial-master-time-limit",
        str(args.partial_master_time_limit),
        "--partial-warmstart-mode",
        args.partial_warmstart_mode,
        "--validation-time-limit",
        str(args.validation_time_limit),
    ]
    if args.skip_incumbent_validation:
        command.append("--skip-incumbent-validation")
    if args.partial_warmstart_time_limit is not None:
        command.extend(
            [
                "--partial-warmstart-time-limit",
                str(args.partial_warmstart_time_limit),
            ]
        )
    if args.partial_embed_route_cost:
        command.append("--partial-embed-route-cost")

    variant_suffix = "" if args.model_variant == "full" else f"_{args.model_variant}"
    log_path = os.path.join(
        logs_root,
        f"H{hub_count:03d}{variant_suffix}_{method}.log",
    )
    with open(log_path, "w", encoding="utf-8") as log:
        log.write(" ".join(command) + "\n")
        log.flush()
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    return hub_count, method, int(completed.returncode), log_path


def aggregate_results(results_root: str) -> str:
    rows = []
    for scale_name in sorted(os.listdir(results_root)):
        if not scale_name.startswith("H"):
            continue
        scale_dir = os.path.join(results_root, scale_name)
        if not os.path.isdir(scale_dir):
            continue
        for method in METHODS:
            summary_path = os.path.join(scale_dir, method, "summary.json")
            if not os.path.exists(summary_path):
                continue
            with open(summary_path, encoding="utf-8") as f:
                rows.append(json.load(f))

    output_path = os.path.join(results_root, "comparison_all_scales.csv")
    if not rows:
        return output_path
    baselines = {}
    best_gurobi_baselines = {}
    for row in rows:
        key = (row.get("H"), row.get("model_variant"))
        value = row.get("validation_obj")
        if value is None:
            value = row.get("obj")
        if value is None:
            continue
        if row.get("method") == "gurobi":
            baselines[key] = float(value)
        if row.get("method") in {"gurobi", "gurobi_numeric", "gurobi_maxflow"}:
            best_gurobi_baselines[key] = max(
                float(value),
                best_gurobi_baselines.get(key, -float("inf")),
            )
    for row in rows:
        key = (row.get("H"), row.get("model_variant"))
        if key not in baselines:
            continue
        value = row.get("validation_obj")
        if value is None:
            value = row.get("obj")
        if value is None:
            continue
        improvement = float(value) - baselines[key]
        row["obj_vs_gurobi_abs"] = improvement
        row["obj_vs_gurobi_pct"] = improvement / max(1.0, abs(baselines[key]))
        row["beats_gurobi_incumbent"] = improvement > 1e-5
        if key in best_gurobi_baselines:
            best_improvement = float(value) - best_gurobi_baselines[key]
            row["obj_vs_best_gurobi_abs"] = best_improvement
            row["obj_vs_best_gurobi_pct"] = best_improvement / max(
                1.0, abs(best_gurobi_baselines[key])
            )
            row["beats_best_gurobi_incumbent"] = best_improvement > 1e-5

    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return output_path


def main() -> None:
    args = build_parser().parse_args()
    if args.parallel_runs <= 0:
        raise ValueError("--parallel-runs must be positive.")
    jobs = [(hub_count, method) for hub_count in args.hub_counts for method in args.methods]
    failures = []
    with ThreadPoolExecutor(max_workers=args.parallel_runs) as executor:
        futures = {
            executor.submit(run_one, args, hub_count, method): (hub_count, method)
            for hub_count, method in jobs
        }
        for future in as_completed(futures):
            hub_count, method, returncode, log_path = future.result()
            print(
                f"H={hub_count} method={method} returncode={returncode} log={log_path}",
                flush=True,
            )
            if returncode != 0:
                failures.append((hub_count, method, returncode))

    comparison_path = aggregate_results(args.results_root)
    print(f"Combined comparison: {comparison_path}")
    if failures:
        details = ", ".join(
            f"H={hub_count}/{method}: {code}"
            for hub_count, method, code in failures
        )
        raise SystemExit(f"Failed runs: {details}")


if __name__ == "__main__":
    main()
