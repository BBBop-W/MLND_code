"""Run the required cut/local-branching ablations with isolated result names."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys


PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)

VARIANTS = {
    "classic": (
        "--no-local-branching",
        "--no-pareto-cuts",
        "--no-bdd-strengthened",
        "--no-bdd-lagrangian",
    ),
    "pareto": (
        "--no-local-branching",
        "--no-bdd-strengthened",
        "--no-bdd-lagrangian",
    ),
    "bdd_strengthened": (
        "--no-local-branching",
        "--no-bdd-lagrangian",
    ),
    "bdd_lagrangian": ("--no-local-branching",),
    "bdd_lagrangian_local_branching": ("--local-branching",),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scales", default="5,10,20")
    parser.add_argument("--methods", default="bd,lr_bd")
    parser.add_argument("--variants", default=",".join(VARIANTS))
    parser.add_argument("--variant-prefix", default="qfix_ablation_")
    parser.add_argument("--time-limit", type=float, default=900.0)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument(
        "--results-root",
        default=os.path.join(
            PROJECT_ROOT, "results", "bdd_lr_benders_20260623"
        ),
    )
    args, extra = parser.parse_known_args()

    variants = [
        value.strip() for value in args.variants.split(",") if value.strip()
    ]
    unknown = [value for value in variants if value not in VARIANTS]
    if unknown:
        raise SystemExit(f"Unknown variants: {unknown}")

    for scale in [
        int(value) for value in args.scales.split(",") if value.strip()
    ]:
        for variant in variants:
            command = [
                sys.executable,
                "-m",
                "src.experiments.bdd_lr_benders_20260623.benchmark",
                "--hub-count",
                str(scale),
                "--methods",
                args.methods,
                "--variant",
                f"{args.variant_prefix}{variant}",
                "--time-limit",
                str(args.time_limit),
                "--threads",
                str(args.threads),
                "--results-root",
                args.results_root,
                *VARIANTS[variant],
                *extra,
            ]
            print(" ".join(command), flush=True)
            completed = subprocess.run(
                command, cwd=PROJECT_ROOT, check=False
            )
            if completed.returncode != 0:
                raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
