"""Solver equivalence and cross-design certificate checks on finite instances."""

from dataclasses import replace
from itertools import product

import numpy as np
import pytest

from src.model.types import PlanningData
from src.model.paths import enumerate_paths
from src.solvers import solve
from src.solvers.algorithm import solve_subproblem
from src.solvers.formulation import build_model, shapes
from src.solvers.dual import cut_value, certified_cut, price_catalogue


def data(**changes):
    values = dict(
        n_l=1,
        n_h=2,
        demand=np.array([4.0, 7.0]),
        flight_times=np.array([[0.0, 0.05, 0.07], [0.05, 0.0, 0.1], [0.07, 0.1, 0.0]]),
        rider_mean=np.array([[0.08, 0.12], [0.1, 0.06]]),
        first_cost=1.0,
        terminal_cost=0.5,
        receipt_mean=0.08,
        removal_mean=0.03,
        ground_cost=5.0,
        deadline=1.0,
        base_cost=0.5,
        slot_counts=[10, 20],
        slot_cost=[0.1, 0.2],
        slot_epsilon=0.05,
        link_cost=0.0,
        kappa=np.array([[0.3, 0.9], [0.2, 0.7]]),
        delta=0.04,
        operating_limit=1.0,
        fleet_cost=1.0,
        flight_cost=0.2,
        waiting_cost=0.3,
    )
    values.update(changes)
    return PlanningData(**values)


def design(d, assignment=(1, 1), level=0, closed_link=None, all_closed=False):
    result = {name: np.zeros(shape) for name, shape in shapes(d).items()}
    if all_closed:
        return result
    result["facility"][:] = 1
    result["capacity_level"][:, level] = 1
    result["assignment"][:, 0, 0] = assignment
    result["airline"][:] = d.allowed_links
    if closed_link:
        result["airline"][closed_link] = 0
    return result


@pytest.mark.parametrize("max_orders", [1, 2])
def test_full_path_and_both_benders_methods_agree(max_orders):
    d = data(max_orders=max_orders)
    full = solve(d, method="full_path", time_limit=20, mip_gap=1e-7)
    assert full["termination_reason"] == "gap_limit"
    assert full["objective"] < d.demand @ d.ground_cost
    for method in ("benders_cg", "benders_cg_plain"):
        result = solve(d, method=method, time_limit=20, mip_gap=1e-6)
        assert result["pricing_complete"]
        assert result["catalogue_complete"]
        assert result["gap"] < 1e-6
        assert result["objective"] == pytest.approx(full["objective"], abs=2e-5)
        assert result["bound"] <= full["objective"] + 1e-6


def test_empty_restricted_problem_generates_coupled_branches():
    d = data()
    paths = enumerate_paths(d)
    chosen = design(d)
    full = solve_subproblem(d, paths, chosen, column_generation=False, time_limit=10)
    cg = solve_subproblem(d, paths, chosen, time_limit=10)
    assert cg["iterations"] >= 2
    assert cg["pricing_complete"]
    assert cg["solution"]["objective"] == pytest.approx(
        full["solution"]["objective"], abs=1e-5
    )
    assert cg["cut"]["anchor"] == pytest.approx(
        full["solution"]["continuous_cost"], abs=2e-5
    )


def test_so_cp_cuts_are_global_across_assignments_capacities_and_closed_links():
    d = data()
    paths = enumerate_paths(d)
    designs = [
        design(d, assignment=a, level=level)
        for a in product((0, 1), repeat=2)
        for level in range(2)
    ]
    designs += [
        design(d, closed_link=(0, 1)),
        design(d, closed_link=(1, 2)),
        design(d, all_closed=True),
    ]
    solved = [
        solve_subproblem(d, paths, a, column_generation=False, time_limit=10)
        for a in designs
    ]
    assert all(row["cut"] is not None for row in solved)
    for row in solved:
        for target, other in zip(designs, solved):
            assert (
                cut_value(row["cut"], target)
                <= other["solution"]["continuous_cost"] + 2e-6
            )


def test_fixed_design_uses_identical_matrix_and_bounds():
    d = data()
    paths = enumerate_paths(d)
    built = [
        build_model(d, paths, fixed_design=design(d, assignment=a))
        for a in [(1, 0), (0, 1)]
    ]
    try:
        first, second = [b.model for b in built]
        assert (first.getA() != second.getA()).nnz == 0
        for attribute in ("LB", "UB", "Obj"):
            np.testing.assert_array_equal(
                first.getAttr(attribute), second.getAttr(attribute)
            )
        changed = np.flatnonzero(
            np.array(first.getAttr("RHS")) != np.array(second.getAttr("RHS"))
        )
        assert len(changed) == 2
        assert all(
            first.getConstrs()[i].ConstrName.startswith("fix_assignment")
            for i in changed
        )
    finally:
        for b in built:
            b.model.dispose()


def test_missing_double_branch_forces_zero_initiation():
    d = data(demand=[4.0], rider_mean=0.08, kappa=[[1.0]])
    paths = enumerate_paths(d)
    paths = replace(
        paths, service_paths=tuple(p for p in paths.service_paths if p.mode == "single")
    )
    result = solve_subproblem(
        d, paths, design(d, assignment=[1]), column_generation=False, time_limit=10
    )
    assert result["solution"]["continuous_cost"] == pytest.approx(
        float(d.demand @ d.ground_cost), abs=1e-6
    )
    assert np.max(result["solution"]["_solution"]["head"]) < 1e-7


def test_pricing_visits_paths_on_closed_network_and_rejects_interrupted_certificate():
    d = data()
    paths = enumerate_paths(d)
    built = build_model(
        d,
        paths,
        fixed_design=design(d, all_closed=True),
        service_indices=(),
        empty_indices=(),
    )
    try:
        prices = np.zeros(built.model.NumConstrs)
        for row in built.constraints["order"].values():
            prices[row.index] = 100.0
        priced = price_catalogue(built, {"_pi": prices})
        assert priced["scanned"] == len(paths.service_paths) + len(paths.empty_paths)
        assert len(priced["service"]) == len(paths.service_paths)
        assert certified_cut({"intercept": 0}, priced) is None
        interrupted = price_catalogue(
            built, {"_pi": prices}, stop_requested=lambda: True
        )
        assert not interrupted["complete"]
        assert certified_cut({"intercept": 0}, interrupted) is None
    finally:
        built.model.dispose()


@pytest.mark.parametrize(
    "options, reason",
    [
        ({"column_limit": 0}, "column_limit"),
        ({"time_limit": 0}, "time_limit"),
        ({"stop_requested": lambda: True}, "user_stop"),
    ],
)
def test_incomplete_catalogue_returns_only_all_courier_baseline(options, reason):
    d = data()
    result = solve(d, **options)
    assert result["termination_reason"] == reason
    assert result["objective"] == float(d.demand @ d.ground_cost)
    assert result["bound"] == 0
    assert not result["pricing_complete"]
    assert not result["catalogue_complete"]
    assert result["_solution"]["fleet"] == 0
    assert not result["_solution"]["facility"].any()


@pytest.mark.parametrize("method", ["full_path", "benders_cg"])
def test_zero_courier_cost_has_zero_optimum(method):
    result = solve(data(ground_cost=0), method=method, time_limit=10)
    assert result["objective"] == result["bound"] == 0


def test_two_base_repositioning_is_generated_and_balances_service():
    d = data(
        n_l=2,
        flight_times=np.array(
            [
                [0.0, 0.1, 0.04, 0.12],
                [0.1, 0.0, 0.12, 0.04],
                [0.04, 0.12, 0.0, 0.08],
                [0.12, 0.04, 0.08, 0.0],
            ]
        ),
        delta=0.02,
    )
    paths = enumerate_paths(d)
    chosen = design(d)
    chosen["assignment"][:] = 0
    chosen["assignment"][:, 0, 1] = 1
    full = solve_subproblem(d, paths, chosen, column_generation=False, time_limit=10)
    cg = solve_subproblem(d, paths, chosen, time_limit=10)
    assert cg["pricing_complete"]
    assert cg["solution"]["objective"] == pytest.approx(
        full["solution"]["objective"], abs=1e-5
    )
    service = sum(row["flow"] for row in cg["solution"]["_missions"])
    reverse = sum(
        row["flow"]
        for row in cg["solution"]["_relocations"]
        if tuple(row["path"]["b"]) == (1, 0)
    )
    assert service > 1
    assert reverse == pytest.approx(service, abs=1e-7)


def test_interrupted_actual_cg_iteration_never_returns_a_cut(monkeypatch):
    from src.solvers import algorithm

    monkeypatch.setattr(
        algorithm,
        "price_catalogue",
        lambda *args, **kwargs: dict(
            complete=False, reason="time_limit", service=[], empty=[]
        ),
    )
    d = data()
    result = solve_subproblem(d, enumerate_paths(d), design(d), time_limit=10)
    assert result["solution"] is not None
    assert result["cut"] is None
    assert result["pricing_complete"] is False
    assert result["termination_reason"] == "time_limit"


def test_forged_physical_coefficients_are_rejected_before_optimization():
    d = data()
    paths = enumerate_paths(d)
    forged = replace(paths.service_paths[0], cost=-100.0)
    paths = replace(paths, service_paths=(forged, *paths.service_paths[1:]))
    with pytest.raises(ValueError, match="coefficients"):
        build_model(d, paths)


@pytest.mark.parametrize("method", ["full_path", "benders_cg"])
def test_explicit_model_size_guard_returns_feasible_baseline(monkeypatch, method):
    from src.solvers import algorithm

    original_build_model = algorithm.build_model

    def limited_build_model(*args, **kwargs):
        # build_model captures its default at definition time. Changing the
        # module constant only affects later incremental column insertion;
        # pass the explicit construction limit for both solver frontends.
        kwargs["max_lifted_variables"] = 1
        return original_build_model(*args, **kwargs)

    monkeypatch.setattr(algorithm, "build_model", limited_build_model)
    d = data()
    result = solve(d, method=method, time_limit=10)
    assert result["termination_reason"] == "model_size_limit"
    assert result["objective"] == float(d.demand @ d.ground_cost)
    assert result["construction_complete"] is False
    assert result["pricing_complete"] is False


def test_material_bound_contradiction_is_rejected():
    from src.solvers.algorithm import _checked_bound

    lower, inconsistent, correction = _checked_bound(120.0, 100.0, 40.0)
    assert lower == 40.0 and inconsistent and correction == 0


def test_unvalidated_primal_never_replaces_the_baseline(monkeypatch):
    from src.solvers import algorithm

    monkeypatch.setattr(algorithm, "_valid_primal", lambda *args: False)
    d = data()
    result = solve(d, method="full_path", time_limit=10)
    assert result["termination_reason"] == "primal_validation_failed"
    assert result["objective"] == float(d.demand @ d.ground_cost)
    assert not result["_solution"]["facility"].any()
