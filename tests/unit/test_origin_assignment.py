"""K=1 departure assignment pools all physical returns without weakening balance."""

from copy import deepcopy

import numpy as np
import pytest
from gurobipy import GRB

from test_solver import data, design
from test_operations import instance, single_solution
from src.model.assignment import assignment_shape, saved_assignment_scope
from src.model.paths import enumerate_paths
from src.operations.validation import build_operations
from src.solvers.formulation import build_model, read_solution, validate_design_arrays
from src.solvers.algorithm import solve_subproblem
from src.solvers.pareto import CorePoint, validate_core_point
from src.solvers.strengthening import master_bounds


def two_base_data(max_orders=1):
    return data(
        n_l=2,
        demand=[4.0],
        rider_mean=0.08,
        kappa=[[1.0]],
        flight_times=[
            [0, 0.1, 0.04, 0.12],
            [0.1, 0, 0.12, 0.04],
            [0.04, 0.12, 0, 0.08],
            [0.12, 0.04, 0.08, 0],
        ],
        max_orders=max_orders,
        slot_counts=[100],
        slot_cost=[0.1],
    )


def fixed_split_model(*, close_return=False, remove_relocation=False):
    d = two_base_data()
    paths = enumerate_paths(d)
    chosen = design(d, assignment=[1])
    if close_return:
        chosen["facility"][1] = 0
        chosen["airline"][1, :] = 0
        chosen["airline"][:, 1] = 0
    built = build_model(d, paths, fixed_design=chosen)
    selected = [
        next(
            i
            for i, p in enumerate(paths.service_paths)
            if p.b == (0, end) and p.pickups == (0,) and p.deliveries == (1,)
        )
        for end in (0, 1)
    ]
    reverse = next(i for i, p in enumerate(paths.empty_paths) if p.b == (1, 0))
    for i, variable in built.variables["flow"].items():
        built.model.addConstr(variable == float(i in selected))
    for i, variable in built.variables["empty_flow"].items():
        built.model.addConstr(variable == float(i == reverse and not remove_relocation))
    built.model.optimize()
    return d, paths, chosen, built


def test_k1_one_origin_can_serve_the_same_od_with_two_return_bases():
    d, paths, chosen, built = fixed_split_model()
    try:
        assert built.model.Status == GRB.OPTIMAL
        result = read_solution(built)
        assert result["assignment_scope"] == "origin"
        assert result["_solution"]["assignment"].shape == (1, 2)
        assert len(built.variables["v"]) == 2
        order_rows = [
            r for r in built.model.getConstrs() if r.ConstrName.startswith("orders_")
        ]
        assert len(order_rows) == 2
        result_ops = build_operations(d, result)
        assert result_ops["summary"]["checks"]["passed"]
        assert result_ops["summary"]["served"] == pytest.approx(2)
        assert result["_solution"]["rho"][0] == pytest.approx(0.5)
        assert len(result_ops["tables"]["pools"]) == 2
        assert result_ops["tables"]["orders"][0]["return_base"] is None
        assert result_ops["summary"][
            "cross_base_service_tasks_per_hour"
        ] == pytest.approx(1)
        flows = {
            tuple(row["path"]["b"]): row["flow"]
            for row in result["_missions"]
            if row["flow"] > 1e-8
        }
        assert flows == pytest.approx({(0, 0): 1, (0, 1): 1})
        # The complete SOCP and CG use the same origin-specific rows and matrix.
        full = solve_subproblem(
            d, paths, chosen, column_generation=False, time_limit=10
        )
        cg = solve_subproblem(d, paths, chosen, time_limit=10)
        assert cg["pricing_complete"]
        assert cg["solution"]["objective"] == pytest.approx(
            full["solution"]["objective"], abs=2e-5
        )
    finally:
        built.model.dispose()


@pytest.mark.parametrize(
    "options", [{"close_return": True}, {"remove_relocation": True}]
)
def test_origin_assignment_still_requires_open_returns_and_base_balance(options):
    _, _, _, built = fixed_split_model(**options)
    try:
        assert built.model.Status in (GRB.INFEASIBLE, GRB.INF_OR_UNBD)
    finally:
        built.model.dispose()


def test_independent_audit_rejects_closed_return_and_missing_relocation():
    d, _, _, built = fixed_split_model()
    try:
        result = read_solution(built)
    finally:
        built.model.dispose()
    closed = deepcopy(result)
    closed["_solution"]["facility"][1] = 0
    closed["_solution"]["airline"][1, :] = 0
    closed["_solution"]["airline"][:, 1] = 0
    checks = build_operations(d, closed)["summary"]["checks"]
    assert not checks["passed"] and checks["residuals"]["closed_link_use"] > 0
    missing = deepcopy(result)
    missing["_solution"]["empty_flow"][:] = 0
    for row in missing["_relocations"]:
        row["flow"] = 0
    checks = build_operations(d, missing)["summary"]["checks"]
    assert not checks["passed"] and checks["residuals"]["base_balance"] > 0


def test_k2_even_with_zero_window_retains_base_pair_assignment():
    from dataclasses import replace

    d = replace(two_base_data(max_orders=2), delta=0)
    assert assignment_shape(d) == (1, 2, 2)
    built = build_model(d, enumerate_paths(d), fixed_design=design(d, assignment=[1]))
    try:
        assert len(built.variables["v"]) == 4
        built.model.optimize()
        result = read_solution(built)
        assert result["assignment_scope"] == "base_pair"
        assert build_operations(d, result)["summary"]["checks"]["passed"]
    finally:
        built.model.dispose()


def test_historical_k1_pair_solution_is_readable_but_not_a_new_fixed_design():
    d = instance()
    historical = single_solution(d)
    assert "assignment_scope" not in historical
    assert saved_assignment_scope(d, historical) == "base_pair"
    operations = build_operations(d, historical)
    assert operations["summary"]["checks"]["passed"]
    assert operations["summary"]["assignment_scope"] == "base_pair"
    assert operations["tables"]["orders"][0]["return_base"] == 0
    fixed = {
        key: historical["_solution"][key]
        for key in ("facility", "capacity_level", "assignment", "airline")
    }
    with pytest.raises(ValueError, match="fixed_design.assignment.*shape"):
        build_model(d, enumerate_paths(d), fixed_design=fixed)
    with pytest.raises(ValueError, match="warm_start.assignment.*shape"):
        validate_design_arrays(d, fixed, source="warm_start")
    historical["assignment_scope"] = "origin"
    with pytest.raises(ValueError, match="assignment_scope"):
        build_operations(d, historical)


def test_origin_strengthening_takes_best_return_and_core_uses_only_departure():
    d = two_base_data()
    paths = enumerate_paths(d)
    bounds = master_bounds(d, paths)
    assert bounds["perassignment_savings"].shape == (1, 2)
    for departure in range(2):
        lower = min(
            p.cost + d.fleet_cost * p.time
            for p in paths.service_paths
            if p.b[0] == departure
        )
        assert bounds["perassignment_savings"][0, departure] == pytest.approx(
            max(0, d.ground_cost[0] - lower)
        )
    core = CorePoint(d)
    assert validate_core_point(d, core.point)
    np.testing.assert_allclose(core.point["assignment"], 0.5 / 3)


def test_native_json_assignment_scope_is_metadata_not_a_numeric_array(tmp_path):
    import json
    from src.experiments.native import load_native_result, load_solution_file

    d = two_base_data()
    chosen = design(d, assignment=[1])
    content = {key: value.tolist() for key, value in chosen.items()}
    content["assignment_scope"] = "origin"
    (tmp_path / "solution.json").write_text(json.dumps(content))
    (tmp_path / "result.json").write_text(json.dumps({"assignment_scope": "origin"}))
    (tmp_path / "path_flows.json").write_text(
        json.dumps(dict(segments=[], missions=[], relocations=[]))
    )
    loaded = load_native_result(tmp_path)
    assert loaded["assignment_scope"] == "origin"
    assert "assignment_scope" not in loaded["_solution"]
    assert loaded["_solution"]["assignment"].shape == (1, 2)
    initial = load_solution_file(tmp_path)
    assert initial["assignment_scope"] == "origin"
    validate_design_arrays(d, initial, source="warm_start")
    (tmp_path / "result.json").write_text(json.dumps({"assignment_scope": "base_pair"}))
    with pytest.raises(ValueError, match="assignment_scope disagree"):
        load_native_result(tmp_path)


def test_structural_diagnostics_read_origin_and_historical_points():
    from src.reporting.structural_comparison import structural_point_violations

    dimensions = dict(n_l=2, n_h=1, n_od=1, slot_counts=np.array([[10]]))
    result = dict(
        assignment_scope="origin",
        primal_valid=True,
        design=[0.5, 0.5, 0.5, 0.5, 0.2, 0.1],
        operating_point=dict(
            v_order="r_then_departure", rho=[0.2, 0.2], idle=[1.0, 1.0], v=[0.1, 0.05]
        ),
    )
    summary, rows = structural_point_violations(result, dimensions)
    assert summary["available"]
    assert len(rows) == 8
    assert not any(row["exceeds_tolerance"] for row in rows)
    result["assignment_scope"] = "base_pair"
    assert not structural_point_violations(result, dimensions)[0]["available"]
