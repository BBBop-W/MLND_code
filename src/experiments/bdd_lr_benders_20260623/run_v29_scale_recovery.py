"""Reproduce the v28/v29 restricted-refinement + Gamma-sweep scale runs."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys


PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)


def _run(scale: int, args: argparse.Namespace, extra: list[str]) -> None:
    if scale == 50:
        variant = "qfix_recovery_v28_h50_restricted24_gamma300"
        restricted_time = "24"
    else:
        variant = "qfix_recovery_v29_scale_restricted30_gamma300"
        restricted_time = "30"

    command = [
        sys.executable,
        "-m",
        "src.experiments.bdd_lr_benders_20260623.benchmark",
        "--hub-count",
        str(scale),
        "--methods",
        args.methods,
        "--variant",
        variant,
        "--time-limit",
        str(args.time_limit),
        "--validation-time-limit",
        str(args.validation_time_limit),
        "--threads",
        str(args.threads),
        "--results-root",
        args.results_root,
        "--lr-time-limit",
        str(args.lr_time_limit),
        "--lr-iterations",
        str(args.lr_iterations),
        "--lr-max-designs",
        str(args.lr_max_designs),
        "--lr-restricted-time-limit",
        restricted_time,
        "--bdd-root-rounds",
        "0",
        "--no-global-linearized-bound",
        "--local-branching",
        "--local-branching-start-pool-size",
        "4",
        "--local-branching-max-calls-per-center",
        "8",
        "--local-branching-protect-top-pairs",
        "50",
        "--local-branching-solution-pool-size",
        "4",
        "--local-branching-time-limit",
        "8",
        "--local-branching-radius",
        "24",
        "--local-branching-max-radius",
        "192",
        "--local-branching-gamma-radius",
        "32",
        "--local-branching-max-gamma-radius",
        "240",
        "--gamma-sweep-max-calls",
        "12",
        "--gamma-sweep-radius",
        "48",
        "--gamma-sweep-step",
        "8",
        "--gamma-sweep-time-reserve",
        "10",
        *extra,
    ]
    print(" ".join(command), flush=True)
    completed = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scales", default="50,60,70,80,100")
    parser.add_argument("--methods", default="lr_bd")
    parser.add_argument("--time-limit", type=float, default=300.0)
    parser.add_argument("--validation-time-limit", type=float, default=60.0)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--lr-time-limit", type=float, default=165.0)
    parser.add_argument("--lr-iterations", type=int, default=50)
    parser.add_argument("--lr-max-designs", type=int, default=50)
    parser.add_argument(
        "--results-root",
        default=os.path.join(
            PROJECT_ROOT, "results", "bdd_lr_benders_20260623"
        ),
    )
    args, extra = parser.parse_known_args()
    for scale in [
        int(value) for value in args.scales.split(",") if value.strip()
    ]:
        _run(scale, args, extra)


if __name__ == "__main__":
    main()
