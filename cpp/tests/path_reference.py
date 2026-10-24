"""Compare every native path and resource coefficient with Python enumeration.

Run with the CMake-built mlnd_path_catalogue target. No optimizer is invoked.
"""

# ruff: noqa: E402 -- standalone script adds the repository root before imports.

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.instances import to_planning_data
from src.data.standard import load_standard
from src.model.paths import enumerate_paths


def compare(native, reference, location, differences):
    if isinstance(reference, dict):
        assert isinstance(native, dict), location
        assert native.keys() == reference.keys(), location
        for key in reference:
            compare(native[key], reference[key], f"{location}.{key}", differences)
    elif isinstance(reference, (tuple, list)):
        assert isinstance(native, list) and len(native) == len(reference), location
        for index, (left, right) in enumerate(zip(native, reference)):
            compare(left, right, f"{location}[{index}]", differences)
    elif isinstance(reference, float):
        delta = abs(native - reference)
        differences["float_values_compared"] += 1
        if delta > differences["max_absolute_difference"]:
            differences["max_absolute_difference"] = delta
            differences["largest_difference_at"] = location
        if delta:
            differences["float_values_with_roundoff"] += 1
        assert delta <= 5e-14 * max(1, abs(reference)), (
            location,
            native,
            reference,
            delta,
        )
    else:
        assert native == reference, (location, native, reference)


def signature(paths):
    physical = [
        [
            p["id"],
            p["segment_id"],
            p["b"],
            p["first_od"],
            p["second_od"],
            p["pickups"],
            p["deliveries"],
            p["legs"],
            p["served_counts"],
            p["arc_counts"],
        ]
        for p in paths["service_paths"]
    ]
    return hashlib.sha256(
        json.dumps(physical, separators=(",", ":")).encode()
    ).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument(
        "--instance", type=Path, default=ROOT / "examples/tsl_2l20h50od"
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-orders", type=int, nargs="+", default=[1, 2])
    args = parser.parse_args()
    rows = []
    with tempfile.TemporaryDirectory(prefix="mlnd_native_paths_") as temporary:
        for k in args.max_orders:
            target = Path(temporary) / f"K{k}.json"
            subprocess.run(
                [
                    str(args.binary.resolve()),
                    str(args.instance.resolve()),
                    str(k),
                    str(target),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            native = json.loads(target.read_text())
            data = to_planning_data(load_standard(args.instance, case=f"K{k}"))
            started = time.perf_counter()
            reference = enumerate_paths(data)
            elapsed = time.perf_counter() - started
            reference = asdict(reference)
            differences = {
                "float_values_compared": 0,
                "float_values_with_roundoff": 0,
                "max_absolute_difference": 0.0,
                "largest_difference_at": None,
            }
            compare(native["paths"], reference, "paths", differences)
            native_signature = signature(native["paths"])
            assert native_signature == signature(reference)
            rows.append(
                {
                    **native["summary"],
                    "passed": True,
                    "python_enumeration_seconds": elapsed,
                    "physical_signature_sha256": native_signature,
                    **differences,
                }
            )
    result = {
        "passed": True,
        "instance": str(args.instance.resolve()),
        "cases": rows,
        "comparison": "every path, ordering, integer field and physical resource coefficient",
        "numeric_tolerance": "5e-14 * max(1, abs(reference))",
        "solver_run": False,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
