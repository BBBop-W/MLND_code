"""Run the strict Lagrangian/Benders benchmark over multiple H scales."""

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

from src.experiments.benchmark_lagrangian_benders import METHODS, parse_methods


def parse_int_list(raw: str) -> list[int]:
    values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not values or min(values) <= 0:
        raise argparse.ArgumentTypeError(
            "Provide a comma-separated list of positive H values."
        )
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run strict Lagrangian/Benders on an H-scale grid."
    )
    parser.add_argument(
        "--series-root",
        default=os.path.join(PROJECT_ROOT, "data", "instances", "Meituan_H_series"),
    )
    parser.add_argument(
        "--hub-counts",
        type=parse_int_list,
        default=parse_int_list("20,40,60,80,100"),
    )
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
    parser.add_argument("--logs-root", default=None)
    parser.add_argument("--period-id", default="h12")
    parser.add_argument("--time-limit", type=float, default=900.0)
    parser.add_argument("--validation-time-limit", type=float, default=60.0)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--parallel-runs", type=int, default=1)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--mip-gap", type=float, default=0.001)
    parser.add_argument("--lr-time-limit", type=float, default=120.0)
    parser.add_argument("--lr-iterations", type=int, default=50)
    parser.add_argument("--lr-max-designs", type=int, default=8)
    parser.add_argument("--global-ub-time-limit", type=float, default=60.0)
    parser.add_argument("--global-ub-time-fraction", type=float, default=0.10)
    parser.add_argument("--skip-incumbent-validation", action="store_true")
    return parser


def run_one(args, hub_count: int, method: str):
    logs_root = args.logs_root or os.path.join(args.results_root, "logs")
    os.makedirs(logs_root, exist_ok=True)
    command = [
        args.python,
        "-m",
        "src.experiments.benchmark_lagrangian_benders",
        "--series-root",
        args.series_root,
        "--hub-count",
        str(hub_count),
        "--period-id",
        args.period_id,
        "--methods",
        method,
        "--results-root",
        args.results_root,
        "--time-limit",
        str(args.time_limit),
        "--validation-time-limit",
        str(args.validation_time_limit),
        "--threads",
        str(args.threads),
        "--mip-gap",
        str(args.mip_gap),
        "--lr-time-limit",
        str(args.lr_time_limit),
        "--lr-iterations",
        str(args.lr_iterations),
        "--lr-max-designs",
        str(args.lr_max_designs),
        "--global-ub-time-limit",
        str(args.global_ub_time_limit),
        "--global-ub-time-fraction",
        str(args.global_ub_time_fraction),
    ]
    if args.skip_incumbent_validation:
        command.append("--skip-incumbent-validation")

    log_path = os.path.join(logs_root, f"H{hub_count:03d}_{method}.log")
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
            path = os.path.join(scale_dir, method, "summary.json")
            if not os.path.exists(path):
                continue
            with open(path, encoding="utf-8") as f:
                rows.append(json.load(f))

    output = os.path.join(results_root, "comparison_all_scales.csv")
    if not rows:
        return output
    baselines = {}
    for row in rows:
        if row.get("method") != "gurobi":
            continue
        value = row.get("validation_obj")
        if value is None:
            value = row.get("obj")
        if value is not None:
            baselines[int(row["H"])] = float(value)
    for row in rows:
        hub_count = int(row["H"])
        if hub_count not in baselines:
            continue
        value = row.get("validation_obj")
        if value is None:
            value = row.get("obj")
        if value is None:
            continue
        diff = float(value) - baselines[hub_count]
        row["obj_vs_gurobi_abs"] = diff
        row["obj_vs_gurobi_pct"] = diff / max(1.0, abs(baselines[hub_count]))
        row["beats_gurobi_incumbent"] = diff > 1e-5

    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with open(output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return output


def main() -> None:
    args = build_parser().parse_args()
    if args.parallel_runs <= 0:
        raise ValueError("--parallel-runs must be positive.")
    jobs = [
        (hub_count, method)
        for hub_count in args.hub_counts
        for method in args.methods
    ]
    failures = []
    with ThreadPoolExecutor(max_workers=args.parallel_runs) as executor:
        futures = {
            executor.submit(run_one, args, hub_count, method): (
                hub_count,
                method,
            )
            for hub_count, method in jobs
        }
        for future in as_completed(futures):
            hub_count, method, returncode, log_path = future.result()
            print(
                f"H={hub_count} method={method} returncode={returncode} "
                f"log={log_path}",
                flush=True,
            )
            if returncode != 0:
                failures.append((hub_count, method, returncode))
    print(f"Combined comparison: {aggregate_results(args.results_root)}")
    if failures:
        raise SystemExit(
            "Failed runs: "
            + ", ".join(
                f"H={hub_count}/{method}:{code}"
                for hub_count, method, code in failures
            )
        )


if __name__ == "__main__":
    main()
