"""Compare native Benders, CPLEX full models, and Gurobi on identical fixtures.

Run after building cpp/build/mlnd_benders_probe. Native certificates are also
checked across assignments, capacity levels, closed links, and all-closed designs.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from itertools import product
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests" / "unit"))

from test_solver import data, design  # noqa: E402
from src.model.paths import enumerate_paths  # noqa: E402
from src.operations.validation import build_operations  # noqa: E402
from src.solvers.algorithm import solve_full_path, solve_subproblem  # noqa: E402


def check_cg_trace(trace, *, max_columns):
    """Audit the algorithmic contract, not just the final objective."""
    previous = float("inf")
    for iteration in trace:
        value = iteration.get("rmp_objective")
        if value is not None and iteration.get("rmp_optimal") and not iteration.get("primal_rejected_due_to_duality"):
            assert value <= previous + 2e-5 * (1 + abs(value))
            previous = value
        priced = iteration.get("pricing")
        if priced is None:
            assert not iteration["certified"]
            continue
        assert iteration["rmp_optimal"] and iteration["rmp_feasible"]
        assert iteration["dual_optimal"]
        assert abs(iteration["rmp_duality_gap"]) <= iteration["duality_tolerance"]
        assert priced["generated_service"] + priced["generated_empty"] <= max_columns
        if iteration["certified"]:
            assert priced["complete"]
            assert priced["generated_service"] == priced["generated_empty"] == 0
            assert abs(iteration["full_duality_gap"]) <= iteration["duality_tolerance"]
            assert iteration["termination_reason"] == "pricing_complete"
        if iteration["termination_reason"] == "columns_added":
            assert priced["complete"]
            assert priced["generated_service"] + priced["generated_empty"] > 0


def check_cut_schedule(result, *, mode):
    """Check that separating capacity cannot skip a candidate's CG call."""
    assert result["cut_schedule"] == {
        "policy": "same_candidate_capacity_and_cg",
        "root_round_unit": "root_master_and_cg",
        "frequency_unit": "integer_cg_calls",
    }
    statistics = result["acceleration_statistics"]
    roots = [record for record in result["acceleration_records"]
             if record["stage"] == "root_relaxation"]
    integers = result["iterations"]
    assert statistics["root_cg_calls"] == len(roots)
    assert statistics["integer_cg_calls"] == len(integers)
    assert len(roots) <= 2 * statistics["root_master_solves"]
    assert statistics["root_master_solves"] <= result["strengthening_parameters"]["root_cut_rounds"]
    assert [record["cg_call"] for record in integers] == list(range(1, len(integers) + 1))
    fresh = [record for record in integers if not record["resumed_design"]]
    # A final master solve can establish the requested gap without another CG
    # call. Resuming an unfinished design must not reoptimize the master.
    assert len(fresh) <= statistics["integer_master_solves"] <= len(fresh) + 1
    if result["accelerations"]["warm_start"]:
        assert 0 <= statistics["master_starts"] <= statistics["integer_master_solves"]
    else:
        assert statistics["master_starts"] == 0

    grouped = defaultdict(list)
    for record in roots + integers:
        candidate = record["candidate_id"]
        assert candidate
        grouped[candidate].append(record)
        assert record["capacity_cuts_added"] in (0, 1)
        if record["pricing_complete"]:
            assert record["benders_cuts_added"] >= 1
        else:
            assert record["benders_cuts_added"] == 0, "Incomplete CG emitted a Benders cut"
        if record["cg_iterations"] == 0:
            assert not record["pricing_complete"]
    previous = {}
    for record in integers:
        candidate = record["candidate_id"]
        assert candidate.startswith("integer:")
        if record["resumed_design"]:
            assert candidate in previous and not previous[candidate]["pricing_complete"]
            assert record["capacity_cuts_added"] == 0
        else:
            assert candidate not in previous, "A fresh master solve reused a candidate ID"
        previous[candidate] = record

    capacity_by_candidate = Counter(cut["candidate_id"] for cut in result["capacity_cuts"])
    benders_by_candidate = Counter(cut["candidate_id"] for cut in result["benders_cuts"])
    for candidate, records in grouped.items():
        assert sum(record["capacity_cuts_added"] for record in records) == capacity_by_candidate[candidate]
        assert sum(record["benders_cuts_added"] for record in records) == benders_by_candidate[candidate]
    for cut in result["capacity_cuts"]:
        if cut["stage"] == "integer_master" or result["accelerations"]["root_cuts"]:
            assert cut["candidate_id"] in grouped, "Capacity separation skipped the candidate's CG call"
    root_pairs = sum(bool(record["capacity_cuts_added"] and record["benders_cuts_added"])
                     for record in roots)
    integer_pairs = sum(bool(record["capacity_cuts_added"] and record["benders_cuts_added"])
                        for record in integers)
    if mode == "capacity_joint_root":
        assert len(roots) == result["strengthening_parameters"]["root_cut_rounds"] == 3
        assert roots[0]["capacity_cuts_added"] == 1 and roots[0]["benders_cuts_added"] == 1
        assert root_pairs > 0
    if mode == "capacity_joint_integer":
        assert statistics["root_master_solves"] == statistics["root_cg_calls"] == 0
        assert integers[0]["iteration"] == integers[0]["cg_call"] == 1
        assert integers[0]["capacity_cuts_added"] == 1 and integers[0]["benders_cuts_added"] == 1
        assert integer_pairs > 0
    if mode == "capacity_only":
        assert not result["accelerations"]["root_cuts"] and not roots
        assert statistics["root_cg_calls"] == 0
        assert sum(cut["stage"] == "root_relaxation" for cut in result["capacity_cuts"]) >= 2
    return dict(verified=True, certified_root_pairs=root_pairs,
                certified_integer_pairs=integer_pairs,
                root_master_solves=statistics["root_master_solves"], root_cg_calls=len(roots),
                integer_master_solves=statistics["integer_master_solves"], integer_cg_calls=len(integers))


def check(binary: Path, output: Path, supplied_native: Path | None = None):
    output.parent.mkdir(parents=True, exist_ok=True)
    native_path = output.with_name(output.stem + "_native.json")
    if supplied_native is None:
        subprocess.run([str(binary.resolve()), str(native_path.resolve())], cwd=ROOT, check=True)
    else:
        native_path.write_bytes(supplied_native.read_bytes())
    native = json.loads(native_path.read_text())
    comparisons = []
    total_cut_checks = 0
    total_capacity_cut_checks = 0
    max_capacity_cut_violation = 0.0
    for case in native:
        d = data(max_orders=case["K"], flow_policy=case["policy"])
        paths = enumerate_paths(d)
        assert case["implicit_preprocessing_verified"]
        assert case["matrix_invariance"], "Fixed designs changed the shared matrix"
        assert case["fractional_start_rejected"], "Fractional Benders start was accepted"
        reference = solve_full_path(d, paths, time_limit=20, mip_gap=1e-8)
        assert reference["termination_reason"] == "gap_limit"
        assert abs(case["full"]["objective"] - reference["objective"]) < 2e-5
        fixed = case["fixed_one_column_cg"]
        assert fixed["pricing_complete"] and fixed["cg_trace"][-1]["certified"]
        assert abs(fixed["objective"] - reference["objective"]) < 2e-5
        check_cg_trace(fixed["cg_trace"], max_columns=1)
        stopped = case["zero_budget_cg"]
        assert not stopped["pricing_complete"] and not stopped["benders_cuts"]
        assert stopped["termination_reason"] == "time_limit"
        designs = [design(d, assignment=a, level=level)
                   for a in product((0, 1), repeat=2) for level in range(2)]
        designs += [design(d, closed_link=(0, 1)), design(d, closed_link=(1, 2)), design(d, all_closed=True)]
        recourse = []
        for target in designs:
            fixed = solve_subproblem(d, paths, target, column_generation=False, time_limit=20)
            assert fixed["solution"] is not None
            assert fixed["cut"] is not None
            flattened = np.concatenate([target[n].ravel() for n in ("facility", "capacity_level", "assignment", "airline")])
            recourse.append((flattened, fixed["solution"]["continuous_cost"]))
        for mode, result in case["variants"].items():
            schedule = check_cut_schedule(result, mode=mode)
            design_count = len(recourse[0][0])
            structure = result["master_structure"]
            assert structure["variables"] == design_count + 1
            assert structure["design_binaries"] == design_count
            assert structure["operating_cost_variables"] == 1
            assert structure["operational_auxiliary_variables"] == structure["cones"] == 0
            base_rows = (d.n_h + d.n_od + 2 * d.n_od * d.n_l**2
                         + 2 * (d.n_l + d.n_h)**2 + int(np.count_nonzero(d.allowed_links == 0)))
            bound_metadata = result["master_bound_metadata"]
            analytical_rows = (1 + int(bound_metadata["locker_cut_enabled"])) if bound_metadata["enabled"] else 0
            assert structure["initial_rows"] == base_rows + analytical_rows + int(result["accelerations"]["incumbent_cutoff"])
            capacity_cuts = result["capacity_cuts"]
            assert structure["final_rows"] == (structure["initial_rows"] + len(result["benders_cuts"])
                                                + len(capacity_cuts) + len(result["resource_lp_cuts"])
                                                + result["acceleration_statistics"]["continuous_bound_rows"])
            continuous = result["continuous_root"]
            if continuous["enabled"]:
                assert continuous["bound_valid"] and continuous["primal_valid"]
                assert continuous["certified_lower_bound"] <= case["full_continuous_objective"] + 2e-5
                assert continuous["continuous_objective"] >= case["full_continuous_objective"] - 2e-5
                assert result["bound"] >= continuous["applied_total_cost_lower_bound"] - 2e-5
                assert not continuous["is_integer_solution"]
                check_cg_trace(continuous["cg_trace"], max_columns=continuous["cg_max_columns"])
            if not result["accelerations"]["capacity_cuts"]:
                assert not capacity_cuts and not result["capacity_cut_metadata"]["enabled"]
            for record in result["iterations"] + result["acceleration_records"]:
                check_cg_trace(record["cg_trace"], max_columns=result["cg_max_columns"])
            physical = dict(result)
            physical["_solution"] = {name: np.asarray(value) for name, value in result["_solution"].items()}
            checked = build_operations(d, physical)["summary"]["checks"]
            assert checked["passed"], checked
            assert result["termination_reason"] == "gap_limit", (case["K"], case["policy"], mode, result["termination_reason"])
            assert result["gap"] <= 1e-7
            assert result["pricing_complete"]
            assert abs(result["objective"] - reference["objective"]) < 2e-5
            assert result["bound"] <= reference["objective"] + 2e-6
            worst = -float("inf")
            for cut in result["benders_cuts"]:
                assert cut["pricing_complete"] and cut["support_complete"]
                assert cut["catalogue_complete"] == (mode == "full_recourse")
                for target, cost in recourse:
                    value = cut["intercept"] + np.asarray(cut["coefficients"]) @ target
                    worst = max(worst, value - cost)
                    total_cut_checks += 1
                    assert value <= cost + 2e-6, (case["K"], case["policy"], mode, value, cost)
            capacity_worst = None
            for cut in capacity_cuts:
                assert cut["certificate"] == "original_crossing_affine_capacities"
                theta_coefficient = sum(coefficient for index, coefficient in cut["coefficients"] if index == design_count)
                assert theta_coefficient == 1.0
                affine = np.zeros(design_count)
                for index, coefficient in cut["coefficients"]:
                    if index != design_count:
                        assert 0 <= index < design_count and coefficient >= 0
                        affine[index] += coefficient
                for target, cost in recourse:
                    # The emitted row is theta + affine(X) >= rhs. Compare
                    # its implied cost bound to independently solved Gurobi
                    # recourse at every design, not only its separation point.
                    value = cut["rhs"] - affine @ target
                    residual = float(value - cost)
                    capacity_worst = residual if capacity_worst is None else max(capacity_worst, residual)
                    max_capacity_cut_violation = max(max_capacity_cut_violation, residual)
                    total_capacity_cut_checks += 1
                    assert residual <= 2e-6, (case["K"], case["policy"], mode, "capacity", value, cost)
            for cut in result["resource_lp_cuts"]:
                assert cut["certificate"] == "full_occurrence_domain_dual_feasibility"
                assert cut["valid"] and cut["max_dual_violation_after"] <= 0
                affine = np.zeros(design_count)
                for index, coefficient in cut["coefficients"]:
                    if index != design_count:
                        affine[index] += coefficient
                for target, cost in recourse:
                    assert cut["rhs"] - affine @ target <= cost + 2e-6
            for record in result["master_solves"]:
                assert record["certified_bound"] <= reference["objective"] + 2e-6
            for record in result["iterations"]:
                pareto = record["pareto"]
                if pareto.get("attempted"):
                    assert pareto["requested_seconds"] >= pareto["allocated_seconds"] >= 0
                    assert pareto["seconds"] >= 0
                    if pareto["accepted"]:
                        assert pareto["pricing_complete"]
                        assert pareto["auxiliary_solves"][-1]["optimal"]
            comparisons.append(dict(
                K=case["K"], policy=case["policy"], mode=mode,
                python_gurobi_full=reference["objective"], native_cplex_full=case["full"]["objective"],
                native_benders=result["objective"], difference=result["objective"] - reference["objective"],
                bound=result["bound"], gap=result["gap"], physical_validation=True,
                max_physical_residual=max(checked["residuals"].values()),
                matrix_invariance=True, fractional_start_rejected=True, cuts=len(result["benders_cuts"]),
                cross_design_cut_checks=len(result["benders_cuts"]) * len(recourse),
                max_cut_minus_recourse=worst, accelerations=result["accelerations"],
                lean_master_verified=True, master_structure=structure,
                cut_schedule=schedule,
                capacity_cuts=len(capacity_cuts),
                cross_design_capacity_cut_checks=len(capacity_cuts) * len(recourse),
                max_capacity_cut_minus_recourse=capacity_worst,
                max_capacity_cut_violation=max(0.0, capacity_worst or 0.0),
                statistics=result["acceleration_statistics"], algorithm_seconds=result["algorithm_wall"],
            ))
    assert total_capacity_cut_checks > 0, "No emitted capacity cut was independently verified"
    assert sum(row["cut_schedule"]["certified_root_pairs"] for row in comparisons) > 0
    assert sum(row["cut_schedule"]["certified_integer_pairs"] for row in comparisons) > 0
    report = dict(
        passed=True, comparisons=comparisons, variants_tested=len(comparisons),
        cross_design_cut_checks=total_cut_checks,
        cross_design_capacity_cut_checks=total_capacity_cut_checks,
        max_capacity_cut_violation=max_capacity_cut_violation,
        max_objective_difference=max(abs(row["difference"]) for row in comparisons),
        native_evidence=str(native_path.resolve()),
        source_sha256={str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                       for path in (ROOT / "cpp/src/benders.cpp", ROOT / "cpp/src/formulation.cpp",
                                    ROOT / "cpp/src/strengthening.cpp", ROOT / "cpp/tests/benders_probe.cpp",
                                    ROOT / "cpp/tests/benders_reference.py")},
    )
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("passed", "variants_tested", "cross_design_cut_checks",
                                                  "cross_design_capacity_cut_checks", "max_capacity_cut_violation",
                                                  "max_objective_difference")}, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=ROOT / "cpp/build/mlnd_benders_probe")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/cpp_cplex_20261016/benders_verification.json")
    parser.add_argument("--native", type=Path, help="Use an explicitly saved probe run instead of solving again")
    args = parser.parse_args()
    check(args.binary, args.output, args.native)
