"""Independent Gurobi checks for full-domain BDD cuts on the physical fixtures.

The continuous/partially integral C++ inner values are not equated to a separate
Python relaxation with potentially different valid big-M coefficients. Every
returned cut is checked on integer strategies with exact fixed-design recourse;
the fully integral Lagrangian value and objective-identity control are also
checked independently. No timing/performance claim is made by this regression.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import gurobipy as gp
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests" / "unit"))

from test_solver import data  # noqa: E402
from src.model.paths import enumerate_paths  # noqa: E402
from src.solvers.algorithm import solve_full_path, solve_subproblem  # noqa: E402
from src.solvers.formulation import DESIGN_NAMES, build_model, shapes  # noqa: E402


def unflatten(d, values):
    offset = 0
    result = {}
    for name, shape in shapes(d).items():
        count = int(np.prod(shape))
        result[name] = np.asarray(values[offset:offset + count]).reshape(shape)
        offset += count
    assert offset == len(values)
    return result


def check(native):
    checks = 0
    inner_comparisons = []
    worst = 0.0
    assert native["passed"] and len(native["cases"]) == 4
    for case in native["cases"]:
        d = data(max_orders=case["K"], flow_policy=case["policy"])
        paths = enumerate_paths(d)
        full = solve_full_path(d, paths, time_limit=20, mip_gap=1e-8)
        assert full["termination_reason"] == "gap_limit"
        assert abs(full["objective"] - case["full_reference_objective"]) < 2e-5
        for key in ("root_master_comparison", "root_master_comparison_with_continuous_floor"):
            root = case[key]
            assert root["fixed_costs_retained"]
            for value in (root["baseline"], root["all_accepted_cuts"]):
                assert value["certified"] and value["lower_bound"] <= full["objective"] + 2e-5
            assert root["all_accepted_cuts"]["root_objective_change"] >= -2e-5
        floored = case["root_master_comparison_with_continuous_floor"]
        assert floored["total_cost_floor_includes_fixed_costs"]
        assert floored["baseline"]["objective"] >= floored["certified_total_cost_floor"] - 2e-5
        identity = case["objective_identity_control"]
        assert identity["valid_global_lower_bound"]
        assert abs(identity["inner_upper_bound"] - full["objective"]) < 2e-5
        assert identity["inner_lower_bound"] <= full["objective"] + 2e-6
        references = []
        for point in case["cross_design_recourse"]:
            fixed = solve_subproblem(d, paths, unflatten(d, point["design"]),
                                    column_generation=False, time_limit=20)
            assert fixed["solution"] is not None and fixed["cut"] is not None
            value = fixed["solution"]["continuous_cost"]
            assert abs(value - point["operating_objective"]) < 3e-5
            references.append((np.asarray(point["design"]), value))
        for anchor in case["anchors"]:
            assert anchor["pricing_complete"]
            beta = np.asarray(anchor["coefficients"])
            built = build_model(d, paths, threads=1, mip_gap=1e-8)
            try:
                design_vars = [variable for name in DESIGN_NAMES
                               for variable in built.variables[name].values()]
                assert len(design_vars) == len(beta)
                built.model.setObjective(built.continuous_cost -
                    gp.quicksum(float(b) * variable for b, variable in zip(beta, design_vars)))
                built.model.Params.TimeLimit = 20
                built.model.optimize()
                assert built.model.Status == gp.GRB.OPTIMAL
                full_inner_value = built.model.ObjVal
            finally:
                built.model.dispose()
            for variant in anchor["variants"]:
                assert variant["valid_global_lower_bound"]
                assert not variant["restricted_column_bound_used"]
                if variant["mode"] != "all_design":
                    assert variant["validated_original_integer_upper_bound"] is None
                else:
                    assert abs(variant["inner_upper_bound"] - full_inner_value) < 5e-5
                    assert variant["inner_lower_bound"] <= full_inner_value + 2e-6
                    inner_comparisons.append({"K": case["K"], "policy": case["policy"],
                        "anchor": anchor["name"], "gurobi_full_inner": full_inner_value,
                        "cplex_full_inner": variant["inner_upper_bound"]})
                for design, value in references:
                    residual = variant["safe_intercept"] + beta @ design - value
                    checks += 1
                    worst = max(worst, float(residual))
                    assert residual <= 3e-6 * (1 + abs(value))
                if variant["accepted"]:
                    assert variant["intercept_gain"] > variant["improvement_tolerance"]
                    assert variant["safe_cut"]["validity_scope"] == "all_original_integer_designs"
                else:
                    assert variant["safe_cut"] is None
            if anchor["name"] == "integer_tight_control":
                for comparison in anchor["comparison"]["modes"]:
                    assert comparison["structural_gain_upper_bound"] <= \
                        5e-5 * (1 + abs(anchor["operating_objective"]))
    return {"passed": True, "independent_cross_design_checks": checks,
            "max_independent_cross_design_violation": worst,
            "full_integer_inner_comparisons": inner_comparisons}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path)
    parser.add_argument("--native", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Refusing to overwrite a reference report")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    native_path = args.native or args.output.with_name(args.output.stem + "_native.json")
    if args.native is None:
        if args.binary is None:
            parser.error("Supply --binary or --native")
        subprocess.run([str(args.binary.resolve()), "--self-test", str(native_path.resolve())],
                       cwd=ROOT, check=True)
    result = check(json.loads(native_path.read_text()))
    result["native_result"] = str(native_path.resolve())
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items()
                      if key != "full_integer_inner_comparisons"}, indent=2))


if __name__ == "__main__":
    main()
