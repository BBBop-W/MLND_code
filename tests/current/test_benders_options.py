"""Benders interface regression definitions; no experiments run at import."""

import importlib.util
from pathlib import Path
import numpy as np
import pytest

import src.solvers.benders.engine as benders
import src.solvers.benders.subproblem as recourse
from src.solvers.benders.master import build_master
from src.solvers.pricing.backend import PricingStopped
from src.solvers.pricing.label import price_design

_spec = importlib.util.spec_from_file_location(
    "options_fixture", Path(__file__).with_name("test_optimization.py")
)
_fixtures = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_fixtures)


def test_optional_master_bound_is_absent_when_disabled():
    data = _fixtures._tiny()
    prices = price_design(data)
    for enabled in (False, True):
        model, *_ = build_master(data, prices, initial_bound=enabled)
        try:
            assert (
                model.getConstrByName("initial_operating_bound") is not None
            ) == enabled
        finally:
            model.dispose()


def test_plain_has_no_initial_universal_pricing(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Bare setup must not price every OD/base pair")

    monkeypatch.setattr(benders, "price_design", forbidden)
    result = benders.solve(
        _fixtures._tiny(),
        max_iterations=0,
        initial_bound=False,
        initial_design=False,
        reuse_columns=False,
        warm_start=False,
        incumbent_cleanup=False,
        pareto_cuts=False,
    )
    assert result["stats"]["pricing_calls"] == 0
    assert result["stats"]["recourse_calls"] == 0
    assert result["accelerations"]["extra_valid_inequalities"] == []
    assert result["stats"]["pareto_attempts"] == 0


def test_no_unimplemented_cut_switch_silently_accepted():
    with pytest.raises(NotImplementedError, match="valid inequalities"):
        benders.solve({}, valid_inequalities=True)


def test_interrupted_initial_pricing_keeps_feasible_baseline(monkeypatch):
    def interrupted(*args, **kwargs):
        raise PricingStopped("time_limit")

    monkeypatch.setattr(benders, "price_design", interrupted)
    result = benders.solve(_fixtures._tiny())
    assert result["termination_reason"] == "time_limit"
    assert result["objective"] == pytest.approx(result["all_ground_cost"])
    assert result["stats"]["benders_cuts"] == 0


def test_recourse_seeds_only_selected_groups(monkeypatch):
    data = _fixtures._data()
    design = _fixtures._designs(data)[3]
    calls = []
    original = recourse.price_design

    def logged(*args, **kwargs):
        calls.append(kwargs.get("groups"))
        return original(*args, **kwargs)

    monkeypatch.setattr(recourse, "price_design", logged)
    result = recourse.solve(data, *design, max_iterations=0, time_limit=30)
    assert np.array_equal(calls[0], np.argwhere(design[2] > 0.5))
    assert calls[-1] is None
    assert result["cut_valid"]


@pytest.mark.parametrize("pricing", ["label", "solver"])
def test_benders_backends_have_the_same_tiny_optimum(pricing):
    data = _fixtures._tiny()
    reference = _fixtures.direct_solver.solve(data, time_limit=60)
    result = benders.solve(data, pricing=pricing, time_limit=120, mip_gap=1e-6)
    assert abs(result["objective"] - reference["objective"]) < 2e-4
    assert result["pricing_backend"] == pricing
