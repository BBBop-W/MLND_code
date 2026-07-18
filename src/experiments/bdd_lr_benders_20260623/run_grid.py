"""Sequential multi-scale runner for the isolated 2026-06-23 benchmark."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys


PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scales", default="5,10,20,40,60,80,100")
    parser.add_argument("--methods", default="gurobi,lr,bd,lr_bd")
    parser.add_argument("--time-limit", type=float, default=900.0)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument(
        "--results-root",
        default=os.path.join(
            PROJECT_ROOT, "results", "bdd_lr_benders_20260623"
        ),
    )
    parser.add_argument("--local-branching", action="store_true")
    args, extra = parser.parse_known_args()

    scales = [int(value) for value in args.scales.split(",") if value.strip()]
    for scale in scales:
        command = [
            sys.executable,
            "-m",
            "src.experiments.bdd_lr_benders_20260623.benchmark",
            "--hub-count",
            str(scale),
            "--methods",
            args.methods,
            "--time-limit",
            str(args.time_limit),
            "--threads",
            str(args.threads),
            "--results-root",
            args.results_root,
            *extra,
        ]
        if args.local_branching:
            command.append("--local-branching")
        print(" ".join(command), flush=True)
        completed = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
        if completed.returncode != 0:
            raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
