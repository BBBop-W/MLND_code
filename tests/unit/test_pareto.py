"""Global-cut and fallback contracts for the lifted MW auxiliary."""

from itertools import product
import numpy as np
import pytest

from test_solver import data, design
from src.model.paths import enumerate_paths
from src.solvers.algorithm import solve_subproblem
from src.solvers.formulation import build_model
from src.solvers.dual import cut_value
from src.solvers.pareto import CorePoint, solve_pareto_cut, validate_core_point


def test_core_respects_design_face_and_retains_interior_under_updates():
    d = data()
    core = CorePoint(d)
    validate_core_point(d, core.point)
    np.testing.assert_array_equal(core.point["airline"][~d.allowed_links], 0)
    for _ in range(100):
        core.update(design(d, all_closed=True))
    point = core.point
    validate_core_point(d, point)
    assert point["facility"].min() >= 0.01 * 0.5
    assert np.allclose(point["capacity_level"].sum(axis=1), point["facility"][d.n_l :])
    exposed = core.point
    exposed["facility"][:] = 1
    assert np.all(core.point["facility"] < 1)
    invalid = design(d)
    invalid["capacity_level"][:] = 1
    with pytest.raises(ValueError, match="Capacity levels"):
        core.update(invalid)


def test_two_base_core_counts_shared_endpoints_only_once():
    d = data(n_l=2, n_h=1, rider_mean=0.1)
    point = CorePoint(d).point
    np.testing.assert_allclose(np.diagonal(point["assignment"][0]), 0.1)
    assert point["assignment"][0, 0, 1] == pytest.approx(0.05)
    assert point["airline"][0, 1] == pytest.approx(0.125)


@pytest.mark.parametrize("max_orders", [1, 2])
@pytest.mark.parametrize("policy", ["aggregate", "prefix"])
def test_complete_auxiliary_cuts_are_valid_across_designs(max_orders, policy):
    d = data(max_orders=max_orders, flow_policy=policy)
    paths = enumerate_paths(d)
    anchor = design(d)
    ordinary = solve_subproblem(d, paths, anchor, time_limit=10)
    # No path variables: the auxiliary must restore every omitted dual row.
    built = build_model(
        d, paths, fixed_design=anchor, service_indices=(), empty_indices=()
    )
    try:
        core = CorePoint(d).point
        answer = solve_pareto_cut(
            built,
            ordinary["cut"],
            anchor,
            core,
            ordinary["solution"]["continuous_cost"],
            time_limit=10,
        )
        assert answer["accepted"], answer["reason"]
        assert answer["pricing_complete"]
        assert answer["scanned"] == len(paths.service_paths) + len(paths.empty_paths)
        assert answer["cut"]["auxiliary_missing_columns"] == answer["scanned"]
        assert answer["cut"]["dual_cones"] == d.n_l
        assert answer["core_gain"] >= -answer["verification_tolerance"]
        assert (
            abs(answer["anchor_gap"])
            <= answer["face_tolerance"] + answer["verification_tolerance"]
        )
        targets = [
            design(d, assignment=a, level=level)
            for a in product((0, 1), repeat=2)
            for level in range(2)
        ]
        targets += [design(d, closed_link=(0, 1)), design(d, all_closed=True)]
        for target in targets:
            solved = solve_subproblem(
                d, paths, target, column_generation=False, time_limit=10
            )
            assert (
                cut_value(answer["cut"], target)
                <= solved["solution"]["continuous_cost"] + 2e-6
            )
    finally:
        built.model.dispose()


def test_auxiliary_requires_its_own_complete_pricing_and_preserves_fallback(
    monkeypatch,
):
    from src.solvers import pareto

    d = data(flow_policy="aggregate")
    paths = enumerate_paths(d)
    anchor = design(d)
    ordinary = solve_subproblem(d, paths, anchor, time_limit=10)
    built = build_model(d, paths, fixed_design=anchor)
    try:
        monkeypatch.setattr(
            pareto,
            "price_catalogue",
            lambda *args, **kwargs: dict(
                complete=False, reason="time_limit", service=[], empty=[]
            ),
        )
        answer = solve_pareto_cut(
            built,
            ordinary["cut"],
            anchor,
            CorePoint(d).point,
            ordinary["solution"]["continuous_cost"],
            time_limit=10,
        )
        assert not answer["accepted"]
        assert answer["reason"] == "incomplete_pricing"
        assert answer["cut"] is ordinary["cut"]
        assert not answer["pricing_complete"]
    finally:
        built.model.dispose()


def test_stop_and_loose_anchor_return_existing_cut_without_auxiliary():
    d = data()
    paths = enumerate_paths(d)
    anchor = design(d)
    ordinary = solve_subproblem(d, paths, anchor, time_limit=10)
    built = build_model(d, paths, fixed_design=anchor)
    try:
        for kwargs, value, reason in [
            ({"time_limit": 0}, ordinary["solution"]["continuous_cost"], "time_limit"),
            (
                {"stop_requested": lambda: True},
                ordinary["solution"]["continuous_cost"],
                "user_stop",
            ),
            (
                {},
                ordinary["solution"]["continuous_cost"] + 1,
                "ordinary_anchor_not_tight",
            ),
        ]:
            answer = solve_pareto_cut(
                built, ordinary["cut"], anchor, CorePoint(d).point, value, **kwargs
            )
            assert not answer["accepted"]
            assert answer["reason"] == reason
            assert answer["cut"] is ordinary["cut"]
    finally:
        built.model.dispose()


@pytest.mark.parametrize(
    "mode, reason",
    [("status", "auxiliary_not_optimal"), ("pricing", "pricing_violation")],
)
def test_auxiliary_status_and_pricing_violations_cannot_replace_cut(
    monkeypatch, mode, reason
):
    from src.solvers import pareto
    from gurobipy import GRB

    d = data()
    paths = enumerate_paths(d)
    anchor = design(d)
    ordinary = solve_subproblem(d, paths, anchor, time_limit=10)
    built = build_model(d, paths, fixed_design=anchor)
    try:
        if mode == "status":
            monkeypatch.setattr(
                pareto,
                "solve_dual",
                lambda *args, **kwargs: dict(status=GRB.TIME_LIMIT),
            )
        else:
            monkeypatch.setattr(
                pareto,
                "price_catalogue",
                lambda *args, **kwargs: dict(
                    complete=True, service=[(-1.0, 0)], empty=[]
                ),
            )
        answer = solve_pareto_cut(
            built,
            ordinary["cut"],
            anchor,
            CorePoint(d).point,
            ordinary["solution"]["continuous_cost"],
            time_limit=10,
        )
        assert not answer["accepted"]
        assert not answer["certified"]
        assert answer["reason"] == reason
        assert answer["cut"] is ordinary["cut"]
    finally:
        built.model.dispose()


def test_dual_timing_accumulates_exclusive_phases_and_keeps_cut_valid():
    from src.solvers.dual import (
        DUAL_TIMING_PHASES,
        solve_dual,
        price_catalogue,
        certified_cut,
    )

    d = data()
    paths, anchor = enumerate_paths(d), design(d)
    built = build_model(d, paths, fixed_design=anchor)
    timings = {}
    try:
        built.model.optimize()
        for _ in range(2):
            previous = sum(timings.get(key, 0.0) for key in DUAL_TIMING_PHASES)
            dual = solve_dual(built, time_limit=10, timings=timings)
            cut = certified_cut(dual, price_catalogue(built, dual))
            assert cut is not None
            assert cut_value(cut, anchor) <= built.model.ObjVal + 1e-6
            assert sum(timings[key] for key in DUAL_TIMING_PHASES) > previous
        assert all(timings[key] > 0 for key in DUAL_TIMING_PHASES)
        assert timings["solver_runtime"] >= 0
    finally:
        built.model.dispose()


def test_dual_early_timeout_records_build_without_entering_optimizer():
    from src.solvers.dual import solve_dual
    from src.solvers.formulation import BuildStopped

    timings = {}
    with pytest.raises(BuildStopped, match="time_limit"):
        solve_dual(None, time_limit=0, timings=timings)
    assert timings["build"] > 0
    assert all(timings[key] == 0 for key in ("optimize", "postprocess", "dispose"))


def test_dual_optimizer_exception_records_optimize_and_disposal(monkeypatch):
    from src.solvers import dual as dual_module

    d = data()
    built = build_model(d, enumerate_paths(d), fixed_design=design(d))
    factory = dual_module.gp.Model

    class FailingOptimizer:
        def __init__(self, *args, **kwargs):
            self.model = factory(*args, **kwargs)

        def __getattr__(self, name):
            return getattr(self.model, name)

        def optimize(self, *args):
            raise RuntimeError("diagnostic optimizer failure")

    monkeypatch.setattr(dual_module.gp, "Model", FailingOptimizer)
    timings = {}
    try:
        with pytest.raises(RuntimeError, match="diagnostic optimizer failure"):
            dual_module.solve_dual(built, timings=timings)
        assert all(timings[key] > 0 for key in dual_module.DUAL_TIMING_PHASES)
    finally:
        built.model.dispose()


def test_pareto_timing_leaves_reconcile_and_timeout_keeps_ordinary_cut():
    from src.solvers.dual import DUAL_TIMING_PHASES

    d = data()
    paths, anchor = enumerate_paths(d), design(d)
    ordinary = solve_subproblem(d, paths, anchor, time_limit=10)
    built = build_model(d, paths, fixed_design=anchor)
    timings = {}
    try:
        result = solve_pareto_cut(
            built,
            ordinary["cut"],
            anchor,
            CorePoint(d).point,
            ordinary["solution"]["continuous_cost"],
            time_limit=10,
            timings=timings,
        )
        assert result["accepted"]
        assert timings["wall"] == result["wall"]
        assert timings["pricing"] > 0
        before = timings["wall"]
        result = solve_pareto_cut(
            built,
            ordinary["cut"],
            anchor,
            CorePoint(d).point,
            ordinary["solution"]["continuous_cost"],
            time_limit=0,
            timings=timings,
        )
        assert result["reason"] == "time_limit"
        assert result["cut"] is ordinary["cut"]
        assert timings["wall"] == pytest.approx(before + result["wall"])
        leaves = sum(timings["dual"][key] for key in DUAL_TIMING_PHASES)
        leaves += timings["pricing"] + timings["validation"]
        assert leaves == pytest.approx(timings["wall"], abs=1e-10)
    finally:
        built.model.dispose()
