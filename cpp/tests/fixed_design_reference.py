"""Check fixed-design rejection and assignment-with-zero-flow semantics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import tempfile

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="mlnd_fixed_design_") as temp:
        records_file = Path(temp) / "results.json"
        result = subprocess.run(
            [str(args.binary.resolve()), str(records_file)],
            text=True,
            capture_output=True,
        )
        if result.returncode:
            raise AssertionError(result.stderr)
        records = json.loads(records_file.read_text())
    assert len(records) == 8
    for row in records:
        if row["case"] in {"closed_assigned_base", "no_solve_budget"}:
            assert row["rejected"]
        else:
            assert row["has_solver_solution"] and not row["rejected"]
            expected = 10.0 if row["case"] == "legal_zero" else 10.5
            assert abs(row["objective"] - expected) <= 1e-6, row
            assert abs(row["served"]) <= 1e-7, row
            np.testing.assert_array_equal(row["design"], row["expected_design"])
        row["passed"] = True
    report = {"passed": True, "cases": records}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
