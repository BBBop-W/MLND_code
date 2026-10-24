"""Check strengthened native CG cuts against the original Gurobi model.

This does not implement the new relaxation in Python. It independently checks
integer-equivalence and validity against unchanged full-path operating values,
including conditional cuts evaluated outside their anchor assignment pattern.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests" / "unit"))
from test_solver import data  # noqa: E402
from src.model.paths import enumerate_paths  # noqa: E402
from src.solvers.algorithm import solve_full_path, solve_subproblem  # noqa: E402
from src.solvers.formulation import shapes  # noqa: E402


def unflatten(d, values):
    result, offset = {}, 0
    for name, shape in shapes(d).items():
        count = int(np.prod(shape))
        result[name] = np.asarray(values[offset:offset + count]).reshape(shape)
        offset += count
    assert offset == len(values)
    return result


def check(native):
    assert native["passed"] and len(native["cases"]) == 4
    checks, changed_assignment_checks, worst = 0, 0, 0.0
    cases = []
    for case in native["cases"]:
        d = data(n_l=2, max_orders=case["K"], flow_policy=case["policy"],
                 flight_times=np.asarray([[0, .03, .05, .07], [.03, 0, .06, .04],
                                          [.05, .06, 0, .1], [.07, .04, .1, 0]]))
        paths = enumerate_paths(d)
        full = solve_full_path(d, paths, time_limit=15, mip_gap=1e-8)
        assert full["termination_reason"] == "gap_limit"
        assert abs(full["objective"] - case["reference_integer_objective"]) < 3e-5
        references = []
        for point in case["references"]:
            x = np.asarray(point["design"])
            fixed = solve_subproblem(d, paths, unflatten(d, x),
                                     column_generation=False, time_limit=15)
            assert fixed["solution"] is not None and fixed["cut"] is not None
            q = fixed["solution"]["continuous_cost"]
            assert abs(q - point["operating_cost"]) < 3e-5
            references.append((x, q))
        for result in case["cg_certificates"]:
            anchor = references[result["anchor_index"]][0]
            assert abs(result["objective"] - references[result["anchor_index"]][1]) < 3e-5
            for cut in result["cuts"]:
                for x, q in references:
                    violation = cut["intercept"] + np.dot(cut["coefficients"], x) - q
                    worst = max(worst, float(violation))
                    checks += 1
                    if result["conditional"] and not np.array_equal(anchor[8:16], x[8:16]):
                        changed_assignment_checks += 1
                    assert violation <= 3e-6 * (1 + abs(q))
        for root in case["continuous_roots"]:
            assert root["result"]["bound_valid"]
            assert root["result"]["certified_lower_bound"] <= full["objective"] + 3e-5
        cases.append({"K": case["K"], "policy": case["policy"],
                      "independent_integer_objective": full["objective"],
                      "fixed_design_count": len(references)})
    end_to_end = []
    for k in (1, 2):
        d = data(max_orders=k)
        full = solve_full_path(d, enumerate_paths(d), time_limit=15, mip_gap=1e-8)
        assert full["termination_reason"] == "gap_limit"
        for run in native.get("end_to_end_benders", []):
            if run["K"] != k:
                continue
            result = run["result"]
            assert abs(result["objective"] - full["objective"]) < 3e-5
            assert result["bound"] <= full["objective"] + 3e-5
            assert result["gap"] <= 2e-6
            end_to_end.append({"K": k, "conditional": run["conditional"],
                               "independent_integer_objective": full["objective"],
                               "algorithm_objective": result["objective"],
                               "algorithm_lower_bound": result["bound"]})
    return {"passed": True, "independent_cross_design_cut_checks": checks,
            "conditional_changed_assignment_checks": changed_assignment_checks,
            "maximum_independent_cut_violation": worst, "cases": cases,
            "end_to_end_benders": end_to_end}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Refusing to overwrite existing independent report")
    result = check(json.loads(args.native.read_text()))
    result["native_report"] = str(args.native.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "cases"}, indent=2))


if __name__ == "__main__":
    main()
