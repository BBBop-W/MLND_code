"""Independent full-column checks for Magnanti--Wong conic Benders cuts."""

import importlib.util
from pathlib import Path

import gurobipy as gp
import numpy as np
import pytest

import src.solvers.benders.pareto as pareto
import src.solvers.benders.subproblem as recourse
from src.model.network import network
from src.solvers.direct.path_model import build_full_column_model
from src.solvers.pricing.backend import PricingStopped
from src.solvers.pricing.label import enumerate_all_columns


_spec = importlib.util.spec_from_file_location(
    "pareto_fixture", Path(__file__).with_name("test_optimization.py")
)
_fixtures = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_fixtures)


def _cut_value(cut, point):
    x, z, assignment, y = point
    return float(
        cut["cut_intercept"]
        + cut["cut_facility"] @ x
        + np.sum(cut["cut_level"] * z)
        + np.sum(cut["cut_B"] * assignment)
        + cut["cut_airline"] @ y
    )


def _reference(data, columns, point):
    """Fix a full-column SOCP at binary or fractional strategic right sides."""
    bundle = build_full_column_model(data, columns, time_limit=30)
    model = bundle.model
    try:
        for variable, value in zip(
            (bundle.x, bundle.z, bundle.alpha, bundle.y), point
        ):
            variable.VType = gp.GRB.CONTINUOUS
            variable.LB = value
            variable.UB = value
        model.optimize()
        assert model.Status == gp.GRB.OPTIMAL
        return float(bundle.recourse.getValue())
    finally:
        model.dispose()


@pytest.fixture(scope="module")
def evidence():
    data = _fixtures._data()
    designs = _fixtures._designs(data)
    core = pareto.CorePoint(data).point
    fractional = [core]
    fractional.extend(
        tuple(0.37 * a + 0.63 * b for a, b in zip(designs[index], core))
        for index in (3, 9, 13)
    )
    points = designs + fractional
    columns = enumerate_all_columns(data)
    values = [_reference(data, columns, point) for point in points]
    return data, designs, core, points, values


def _check_core(data, point, *, fixed_x=None, fixed_z=None):
    """Check the actual master polytope rather than a construction recipe."""
    x, z, assignment, y = point
    count = data["n_l"]
    hubs = data["n_h"]
    net = network(data)
    assert np.all((x >= 0) & (x <= 1))
    assert np.all((z >= 0) & (z <= 1))
    assert np.all(assignment >= 0)
    assert np.all((y >= 0) & (y <= 1))
    np.testing.assert_allclose(z.sum(axis=1), x[count : count + hubs], atol=1e-12)
    assert np.all(assignment.sum(axis=(1, 2)) < 1)
    for axis in (1, 2):
        mass = assignment.sum(axis=axis)
        assert np.all(mass <= x[:count] + 1e-12)
        assert np.all(mass[:, x[:count] > 0] < x[:count][x[:count] > 0])
    for edge, (origin, destination) in enumerate(net["uv"]):
        cap = min(x[origin], x[destination])
        if cap:
            assert 0 < y[edge] < cap
        else:
            assert y[edge] == 0
    for origin in range(count):
        for destination in range(count):
            if min(x[origin], x[destination]):
                assert np.all(assignment[:, origin, destination] > 0)
            else:
                assert np.all(assignment[:, origin, destination] == 0)
    if fixed_x is None:
        assert np.all((x > 0) & (x < 1))
    else:
        np.testing.assert_array_equal(x, fixed_x)
    if fixed_z is None:
        assert np.all(z[x[count : count + hubs] > 0] > 0)
    else:
        np.testing.assert_array_equal(z, fixed_z)


def test_core_is_in_master_relative_interior_and_survives_updates():
    data = _fixtures._data()
    state = pareto.CorePoint(data)
    _check_core(data, state.point)
    for design in _fixtures._designs(data)[::3]:
        before = tuple(value.copy() for value in state.point)
        state.update(*design)
        _check_core(data, state.point)
        assert any(not np.array_equal(a, b) for a, b in zip(before, state.point))


def test_core_honors_closed_facilities_and_fixed_capacity_face():
    data = _fixtures._data()
    net = network(data)
    x = np.ones(net["N"])
    x[0] = 0
    x[data["n_l"] + 1] = 0
    x[-1] = 0
    z = np.zeros_like(data["level_capacity"])
    z[:, 1] = x[data["n_l"] : data["n_l"] + data["n_h"]]
    state = pareto.CorePoint(data, fixed_facility=x, fixed_capacity_level=z)
    _check_core(data, state.point, fixed_x=x, fixed_z=z)
    assignment = _fixtures._assignment([(1, 1), None, (1, 1)])
    links = x[net["uv"][:, 0]] * x[net["uv"][:, 1]]
    state.update(x, z, assignment, links)
    _check_core(data, state.point, fixed_x=x, fixed_z=z)


@pytest.mark.parametrize("source", [0, 3, 4, 9, 13])
def test_cut_is_tight_at_anchor_and_valid_on_binary_and_fractional_designs(
    evidence, source
):
    data, designs, core, points, values = evidence
    design = designs[source]
    ordinary = recourse.solve(data, *design, time_limit=30)
    assert ordinary["cg_complete"]
    strengthened = pareto.strengthen(
        data, design, core, ordinary, time_limit=30, threads=1, seed=43,
        pricing="label", stop_requested=None,
    )
    assert strengthened["cut_valid"]
    assert strengthened["accepted"], strengthened["reason"]
    assert strengthened["certified"]
    anchor = _cut_value(strengthened, design)
    assert anchor == pytest.approx(
        values[source], abs=strengthened["anchor_tolerance"] + 2e-5
    )
    assert _cut_value(strengthened, core) >= _cut_value(ordinary, core) - 2e-5
    auxiliary_value = strengthened["auxiliary_objective"]
    assert strengthened["core_optimality_loss"] == pytest.approx(
        auxiliary_value - _cut_value(strengthened, core), abs=1e-10
    )
    assert _cut_value(strengthened, core) >= auxiliary_value - 2e-6 * (
        1 + abs(auxiliary_value)
    )
    if source in (4, 9, 13):
        assert _cut_value(strengthened, core) > _cut_value(ordinary, core) + 1e-3
    for target, (point, value) in enumerate(zip(points, values)):
        assert _cut_value(strengthened, point) <= value + 2e-5, (source, target)


@pytest.mark.parametrize("pricing", ["label", "solver"])
def test_strengthening_prices_unselected_groups_and_closed_facilities(
    evidence, monkeypatch, pricing
):
    data, designs, core, points, values = evidence
    design = designs[9]  # One locker is closed at the incumbent.
    ordinary = recourse.solve(data, *design, time_limit=30)
    calls = []
    original = pareto.price_design

    def logged(*args, **kwargs):
        calls.append(kwargs.copy())
        return original(*args, **kwargs)

    monkeypatch.setattr(pareto, "price_design", logged)
    cut = pareto.strengthen(
        data, design, core, ordinary, time_limit=30, threads=1, seed=43,
        pricing=pricing, stop_requested=None,
    )
    assert cut["accepted"], cut["reason"]
    assert cut["certified"]
    assert calls
    assert calls[-1].get("groups") is None
    assert calls[-1].get("facility") is None
    assert all(call["backend"] == pricing for call in calls)
    for point, value in zip(points, values):
        assert _cut_value(cut, point) <= value + 2e-5


@pytest.mark.parametrize("stop", [False, True])
def test_unfinished_auxiliary_solve_preserves_ordinary_cut(evidence, stop):
    data, designs, core, points, values = evidence
    ordinary = recourse.solve(data, *designs[3], time_limit=30)
    result = pareto.strengthen(
        data, designs[3], core, ordinary, time_limit=30 if stop else 0,
        threads=1, seed=43, pricing="label",
        stop_requested=(lambda: True) if stop else None,
    )
    assert not result["accepted"]
    assert not result["certified"]
    for name in (
        "cut_intercept", "cut_facility", "cut_level", "cut_B", "cut_airline"
    ):
        np.testing.assert_array_equal(result[name], ordinary[name])
    assert result["cut_valid"]
    for point, value in zip(points, values):
        assert _cut_value(result, point) <= value + 2e-5


def test_interrupted_pricing_cannot_certify_restricted_auxiliary_dual(
    evidence, monkeypatch
):
    data, designs, core, points, values = evidence
    ordinary = recourse.solve(data, *designs[0], time_limit=30)
    calls = []

    def interrupted(*args, **kwargs):
        calls.append(kwargs)
        raise PricingStopped("time_limit")

    monkeypatch.setattr(pareto, "price_design", interrupted)
    result = pareto.strengthen(
        data, designs[0], core, ordinary, time_limit=30, threads=1, seed=43,
        pricing="label", stop_requested=None,
    )
    assert not result["accepted"]
    assert not result["certified"]
    assert calls
    for name in (
        "cut_intercept", "cut_facility", "cut_level", "cut_B", "cut_airline"
    ):
        np.testing.assert_array_equal(result[name], ordinary[name])
    for point, value in zip(points, values):
        assert _cut_value(result, point) <= value + 2e-5


def test_uncertified_anchor_does_not_claim_pareto_optimality(evidence):
    data, designs, core, _, _ = evidence
    ordinary = recourse.solve(data, *designs[0], max_iterations=0, time_limit=30)
    assert ordinary["cut_valid"]
    assert not ordinary["cg_complete"]
    result = pareto.strengthen(
        data, designs[0], core, ordinary, time_limit=30, threads=1, seed=43,
        pricing="label", stop_requested=None,
    )
    assert not result["accepted"]
    assert not result["certified"]
    for name in (
        "cut_intercept", "cut_facility", "cut_level", "cut_B", "cut_airline"
    ):
        np.testing.assert_array_equal(result[name], ordinary[name])
