import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def parse_int_list(raw: str) -> list[int]:
    values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("list must contain at least one integer")
    if min(values) <= 0:
        raise argparse.ArgumentTypeError("all values must be positive")
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Gurobi on a Meituan H-scale grid.")
    parser.add_argument(
        "--series-root",
        default=os.path.join(PROJECT_ROOT, "data", "instances", "Meituan_H_series"),
    )
    parser.add_argument(
        "--hub-counts",
        type=parse_int_list,
        default=list(range(5, 101, 5)),
        help="Comma-separated H scales to solve. Default: 5,10,...,100.",
    )
    parser.add_argument(
        "--results-root",
        default=os.path.join(PROJECT_ROOT, "results", "Meituan_H_series"),
    )
    parser.add_argument(
        "--logs-root",
        default=os.path.join(PROJECT_ROOT, "results", "Meituan_H_series", "logs"),
    )
    parser.add_argument("--time-limit", type=float, default=900.0)
    parser.add_argument(
        "--mip-gap",
        type=float,
        default=None,
        help="Use one fixed MIP gap for every scale. Overrides the tiered gap options.",
    )
    parser.add_argument("--small-gap", type=float, default=0.01)
    parser.add_argument("--large-gap", type=float, default=0.05)
    parser.add_argument("--gap-threshold", type=int, default=30)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--parallel-runs", type=int, default=2)
    parser.add_argument("--period-id", default="h12")
    parser.add_argument("--python", default=sys.executable)
    return parser


def mip_gap_for_scale(args: argparse.Namespace, hub_count: int) -> float:
    if args.mip_gap is not None:
        return float(args.mip_gap)
    return float(args.small_gap if hub_count <= args.gap_threshold else args.large_gap)


def solve_one(args: argparse.Namespace, hub_count: int) -> tuple[int, int, float]:
    os.makedirs(args.results_root, exist_ok=True)
    os.makedirs(args.logs_root, exist_ok=True)
    results_dir = os.path.join(args.results_root, f"H{hub_count:03d}_gurobi")
    log_path = os.path.join(args.logs_root, f"H{hub_count:03d}.log")
    mip_gap = mip_gap_for_scale(args, hub_count)
    command = [
        args.python,
        os.path.join(PROJECT_ROOT, "src", "experiments", "solve_meituan_scale_gurobi.py"),
        "--series-root",
        args.series_root,
        "--hub-count",
        str(hub_count),
        "--period-id",
        args.period_id,
        "--results-dir",
        results_dir,
        "--time-limit",
        str(args.time_limit),
        "--mip-gap",
        str(mip_gap),
        "--threads",
        str(args.threads),
    ]
    with open(log_path, "w", encoding="utf-8") as log_file:
        log_file.write(" ".join(command) + "\n")
        log_file.flush()
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            check=False,
        )
    return hub_count, int(completed.returncode), mip_gap


def main() -> None:
    args = build_parser().parse_args()
    counts = list(args.hub_counts)
    if args.parallel_runs <= 0:
        raise ValueError("--parallel-runs must be positive")

    failures = []
    with ThreadPoolExecutor(max_workers=args.parallel_runs) as executor:
        futures = [executor.submit(solve_one, args, h) for h in counts]
        for future in as_completed(futures):
            hub_count, returncode, mip_gap = future.result()
            print(f"H={hub_count}: gap={mip_gap:g} returncode={returncode}", flush=True)
            if returncode != 0:
                failures.append((hub_count, returncode))

    if failures:
        detail = ", ".join(f"H={h}: {code}" for h, code in failures)
        raise SystemExit(f"Failed runs: {detail}")


if __name__ == "__main__":
    main()
