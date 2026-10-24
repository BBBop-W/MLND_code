"""Validate saved label-CG runs against same-design full-path SOCP runs.

Expects RUNS/fixed_cg/K{1,2} and RUNS/full_cplex/K{1,2}. This checker
does not optimize: experiment budgets and runs remain explicit CLI actions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.experiments.native import validate_native  # noqa: E402
from benders_reference import check_cg_trace  # noqa: E402


def check(runs: Path, instance: Path, cg_label: str = "fixed_cg"):
    comparisons = []
    for k in (1, 2):
        cg_dir, full_dir = runs / cg_label / f"K{k}", runs / "full_cplex" / f"K{k}"
        cg = json.loads((cg_dir / "result.json").read_text())
        full = json.loads((full_dir / "result.json").read_text())
        checked = [validate_native(directory, instance, max_orders=k)
                   for directory in (cg_dir, full_dir)]
        assert all(item["passed"] for item in checked), checked
        assert cg["binary_sha256"] == full["binary_sha256"], "Compare runs from the same executable"
        assert cg["input_manifest_sha256"] == full["input_manifest_sha256"]
        assert cg["pricing_complete"] and cg["termination_reason"] == "pricing_complete"
        assert cg["pricing_method"] == "exact_resource_labeling"
        assert cg["cg_trace"] and cg["cg_trace"][-1]["certified"]
        check_cg_trace(cg["cg_trace"], max_columns=cg["cg_max_columns"])
        first = cg["cg_trace"][0]
        assert first["service_columns"] == cg["initial_service_columns"]
        assert first["empty_columns"] == cg["initial_empty_columns"]
        assert cg["support_complete"] and not cg["catalogue_complete"]
        assert cg["path_storage"] == "generated_columns_only"
        assert cg["enumeration_seconds"] == 0
        assert first["service_columns"] > 0, "Greedy route seeds were not supplied"
        assert first["service_columns"] < full["catalogue"]["service"]
        assert cg["catalogue"]["service"] < full["catalogue"]["service"]
        assert full["has_solver_solution"]
        tolerance = cg["cg_duality_tolerance"] * (1 + abs(cg["objective"]))
        assert abs(cg["objective"] - full["objective"]) <= tolerance
        assert cg["bound"] <= full["objective"] + tolerance
        left = json.loads((cg_dir / "solution.json").read_text())
        right = json.loads((full_dir / "solution.json").read_text())
        for name in ("facility", "capacity_level", "assignment", "airline"):
            np.testing.assert_array_equal(left[name], right[name], err_msg=name)
        # Last complete pass is the optimality certificate; earlier pass batch
        # limits do not imply convergence or full-master lower bounds.
        final = cg["cg_trace"][-1]
        assert final["pricing"]["complete"]
        assert final["pricing"]["minimum_reduced_cost"] >= -cg["pricing_tolerance"]
        comparisons.append({
            "K": k, "passed": True, "label_cg_objective": cg["objective"],
            "full_cplex_objective": full["objective"],
            "objective_difference": cg["objective"] - full["objective"],
            "certified_bound": cg["bound"], "relative_gap": cg["gap"],
            "service_columns": cg["service_columns"], "empty_columns": cg["empty_columns"],
            "initial_service_columns": first["service_columns"],
            "registered_service_columns": cg["catalogue"]["service"],
            "complete_service_catalogue": full["catalogue"]["service"],
            "support_preprocessing_seconds": cg["support_preprocessing_seconds"],
            "cg_iterations": cg["cg_iterations"], "final_min_rc": final["pricing"]["minimum_reduced_cost"],
            "final_primal_dual_gap": final["full_duality_gap"],
            "labels_created": sum(row.get("pricing", {}).get("labels_created", 0) for row in cg["cg_trace"]),
            "labels_dominated": sum(row.get("pricing", {}).get("labels_dominated", 0) for row in cg["cg_trace"]),
            "primal_recoveries": sum(len(row.get("primal_attempts", [])) > 1 for row in cg["cg_trace"]),
            "timings": cg["timings"], "wall": cg["wall"],
            "served": cg["served"], "fleet": cg["fleet"],
            "independent_validation": True,
            "evidence": {str(directory.relative_to(runs)): {
                name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
                for name in ("result.json", "solution.json", "path_flows.json")}
                for directory in (cg_dir, full_dir)},
        })
    report = {"passed": True, "scope": "fixed_design_SOCP_not_global_MIP_optimality",
              "instance": str(instance.resolve()), "comparisons": comparisons,
              "source_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                                for path in sorted((ROOT / "cpp/src").glob("*.cpp"))}}
    (runs / "cg_verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"passed": True, "comparisons": [
        {key: row[key] for key in ("K", "label_cg_objective", "full_cplex_objective",
                                   "objective_difference", "cg_iterations", "service_columns",
                                   "final_min_rc", "labels_dominated", "wall")}
        for row in comparisons]}, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", required=True, type=Path)
    parser.add_argument("--instance", type=Path, default=ROOT / "examples/tsl_2l20h50od")
    parser.add_argument("--cg-label", default="fixed_cg", help="CG run subdirectory; preserve failed attempts")
    args = parser.parse_args()
    check(args.runs, args.instance, args.cg_label)
