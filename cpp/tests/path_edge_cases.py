"""Native/Python path parity for blocked arcs, zero means and timing boundaries."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import subprocess
import tempfile

import numpy as np

from path_reference import compare
from src.model.paths import enumerate_paths
from src.model.types import PlanningData


def randomized(seed):
    rng = np.random.default_rng(seed)
    n_l, n_h, n_od = 2, 3, 4
    n = n_l + n_h
    flight = rng.uniform(0, 0.16, (n, n))
    np.fill_diagonal(flight, 0)
    allowed = rng.random((n, n)) > 0.15
    np.fill_diagonal(allowed, False)
    if seed % 2:
        flight[~allowed] = np.inf
        np.fill_diagonal(flight, 0)
    rider = rng.uniform(0, 0.2, (n_od, n_h))
    rider[0, 0] = 0
    receipt = rng.uniform(0, 0.1, (n_od, n_h))
    kappa = rng.random((n_od, n_od))
    kappa[kappa < 0.25] = 0
    return PlanningData(
        n_l=n_l,
        n_h=n_h,
        demand=rng.uniform(1, 10, n_od),
        flight_times=flight,
        allowed_links=allowed,
        rider_mean=rider,
        first_cost=rng.random((n_od, n_h)),
        terminal_cost=rng.random((n_od, n_h)),
        receipt_mean=receipt,
        removal_mean=receipt / 2,
        ground_cost=5,
        deadline=rng.uniform(0.15, 0.4, n_od),
        base_cost=0.5,
        slot_counts=[10, 20],
        slot_cost=[0.1, 0.2],
        slot_epsilon=0.05,
        link_cost=0,
        kappa=kappa,
        delta=0.01 if seed < 8 else 0,
        operating_limit=float(rng.uniform(0.15, 0.6)),
        fleet_cost=1,
        flight_cost=0.2,
        waiting_cost=0.3,
    )


def encoded(value):
    if isinstance(value, np.ndarray):
        return encoded(value.tolist())
    if isinstance(value, (list, tuple)):
        return [encoded(item) for item in value]
    if isinstance(value, dict):
        return {key: encoded(item) for key, item in value.items()}
    if isinstance(value, float) and np.isinf(value):
        return None
    return value


def cases():
    for seed in range(10):
        data = randomized(seed)
        for k in (1, 2):
            yield f"random_{seed}_K{k}", replace(data, max_orders=k)
    base = randomized(0)
    paths = enumerate_paths(base)
    double = next(p for p in paths.service_paths if p.second_od is not None)
    for shift in (-2e-12, -1e-12, -1e-15, 0, 1e-15, 1e-12, 2e-12):
        yield (
            f"mission_boundary_{shift}",
            replace(base, operating_limit=double.time + shift),
        )
        deadline = base.deadline.copy()
        deadline[double.second_od] = double.delivery_times[1] + shift
        yield f"receipt_boundary_{shift}", replace(base, deadline=deadline)
    yield "zero_window", replace(base, delta=0)
    yield (
        "all_arcs_blocked",
        replace(base, allowed_links=np.zeros_like(base.allowed_links)),
    )
    yield "all_zero_courier", replace(base, rider_mean=np.zeros_like(base.rider_mean))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    results = []
    with tempfile.TemporaryDirectory(prefix="mlnd_path_boundaries_") as temporary:
        fixture, output = Path(temporary) / "input.json", Path(temporary) / "paths.json"
        for name, data in cases():
            fixture.write_text(json.dumps(encoded(asdict(data)), allow_nan=False))
            subprocess.run(
                [str(args.binary.resolve()), "--fixture", str(fixture), str(output)],
                check=True,
                capture_output=True,
                text=True,
            )
            native = json.loads(output.read_text())["paths"]
            reference = asdict(enumerate_paths(data))
            differences = {
                "float_values_compared": 0,
                "float_values_with_roundoff": 0,
                "max_absolute_difference": 0,
                "largest_difference_at": None,
            }
            compare(native, reference, name, differences)
            results.append(
                {
                    "name": name,
                    "passed": True,
                    "service_paths": len(native["service_paths"]),
                    **differences,
                }
            )
    report = {"passed": True, "cases": len(results), "results": results}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "passed": True,
                "cases": len(results),
                "max_absolute_difference": max(
                    r["max_absolute_difference"] for r in results
                ),
            }
        )
    )


if __name__ == "__main__":
    main()
