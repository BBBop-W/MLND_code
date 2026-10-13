"""Auxiliary-cut integration preserves certification, budgets and stop handling."""

from functools import lru_cache
import importlib.util
from pathlib import Path

import numpy as np
import pytest

import src.solvers.benders.engine as engine
from src.model.network import network
from src.solvers.benders.subproblem import RecourseStopped

_spec = importlib.util.spec_from_file_location(
    "pareto_integration_fixture", Path(__file__).with_name("test_optimization.py")
)
_fixtures = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_fixtures)


@lru_cache(maxsize=1)
def evaluated_design():
    data = _fixtures._tiny()
    net = network(data)
    capacity = np.zeros((data["n_h"], 2))
    capacity[:, 1] = 1
    assignment = np.zeros((data["n_od"], data["n_l"], data["n_l"]))
    assignment[:, 0, 0] = 1
    design = (np.ones(net["N"]), capacity, assignment, np.ones(net["count"]))
    answer = engine.recourse(data, *design, time_limit=30)
    assert answer["cut_valid"] and answer["cg_complete"]
    return data, design, answer


def one_evaluation(monkeypatch, auxiliary, *, enabled=True, complete=True):
    data, design, answer = evaluated_design()
    calls = dict(core=0, updates=0, auxiliary=0)

    class Core:
        def __init__(self, *args, **kwargs):
            assert enabled, "Disabled Pareto cuts must not construct a core point"
            calls["core"] += 1
            self.point = tuple(part * 0.5 for part in design)

        def update(self, *parts):
            calls["updates"] += 1
            for actual, expected in zip(parts, design):
                np.testing.assert_array_equal(actual, expected)

    def strengthen(*args, **kwargs):
        assert enabled
        calls["auxiliary"] += 1
        calls["budget"] = kwargs["time_limit"]
        assert 0 < kwargs["time_limit"] <= 30
        assert kwargs["pricing"] == "label"
        return auxiliary(answer)

    monkeypatch.setattr(engine.pareto, "CorePoint", Core)
    monkeypatch.setattr(engine.pareto, "strengthen", strengthen)
    monkeypatch.setattr(engine, "initial_designs", lambda *args: [design])
    monkeypatch.setattr(
        engine, "recourse", lambda *args, **kwargs: dict(answer, cg_complete=complete)
    )
    result = engine.solve(
        data,
        time_limit=60,
        max_iterations=0,
        initial_bound=False,
        initial_design=True,
        pareto_cuts=enabled,
    )
    return result, calls


def test_accepted_pareto_cut_is_additional_to_ordinary_cut(monkeypatch):
    result, calls = one_evaluation(
        monkeypatch,
        lambda answer: dict(
            answer,
            accepted=True,
            cg_iterations=2,
            pricing_calls=3,
            pricing_wall=0.5,
            core_gain=1.25,
            core_optimality_loss=1e-7,
            tightness_error=1e-8,
            anchor_tolerance=1e-6,
            certified=True,
            reason="accepted",
        ),
    )
    assert calls["updates"] == calls["auxiliary"] == calls["core"] == 1
    stats = result["stats"]
    assert stats["benders_cuts"] == 2
    assert stats["pareto_cuts"] == stats["pareto_attempts"] == 1
    assert stats["pareto_cg_iterations"] == 2
    assert stats["pareto_pricing_calls"] == 3
    assert stats["pareto_pricing_wall"] == 0.5
    assert stats["pareto_rejected"] == 0
    trace = next(entry for entry in result["trace"] if entry["stage"] == "pareto")
    assert trace["core_gain"] == 1.25
    assert trace["core_optimality_loss"] == 1e-7
    assert trace["tightness_error"] == 1e-8
    assert trace["anchor_tolerance"] == 1e-6
    assert trace["certified"]


@pytest.mark.parametrize(
    "accepted,valid,certified",
    [(False, True, True), (True, False, True), (True, True, False)],
)
def test_rejected_or_uncertified_auxiliary_cut_keeps_ordinary_cut(
    monkeypatch, accepted, valid, certified
):
    result, calls = one_evaluation(
        monkeypatch,
        lambda answer: dict(
            answer, accepted=accepted, cut_valid=valid, certified=certified
        ),
    )
    assert calls["updates"] == 1
    assert result["stats"]["benders_cuts"] == 1
    assert result["stats"]["pareto_cuts"] == 0
    assert result["stats"]["pareto_rejected"] == 1


def test_local_pareto_timeout_preserves_certified_result(monkeypatch):
    def timeout(answer):
        raise RecourseStopped("time_limit")

    result, calls = one_evaluation(monkeypatch, timeout)
    assert result["termination_reason"] == "iteration_limit"
    assert result["stats"]["benders_cuts"] == 1
    assert result["stats"]["pareto_interrupted"] == 1
    assert calls["updates"] == 1
    assert not result["errors"]


def test_pareto_user_stop_is_honored(monkeypatch):
    def stopped(answer):
        raise RecourseStopped("user_stop")

    result, calls = one_evaluation(monkeypatch, stopped)
    assert result["termination_reason"] == "user_stop"
    assert result["stats"]["benders_cuts"] == 1
    assert result["stats"]["pareto_interrupted"] == 1
    assert calls["updates"] == 1


@pytest.mark.parametrize("reason", ["time_limit", "user_stop", "incomplete_pricing"])
def test_returned_interruption_is_recorded_and_user_stop_propagates(
    monkeypatch, reason
):
    result, calls = one_evaluation(
        monkeypatch,
        lambda answer: dict(answer, accepted=False, certified=False, reason=reason),
    )
    assert result["termination_reason"] == (
        "user_stop" if reason == "user_stop" else "iteration_limit"
    )
    assert result["stats"]["pareto_interrupted"] == 1
    assert result["stats"]["pareto_rejected"] == 0
    assert result["stats"]["benders_cuts"] == 1
    assert calls["updates"] == 1


@pytest.mark.parametrize("enabled,complete", [(False, True), (True, False)])
def test_disabled_or_incomplete_recourse_does_not_run_pareto(
    monkeypatch, enabled, complete
):
    def forbidden(answer):
        raise AssertionError("Unexpected Pareto solve")

    result, calls = one_evaluation(
        monkeypatch, forbidden, enabled=enabled, complete=complete
    )
    assert calls["auxiliary"] == result["stats"]["pareto_attempts"] == 0
    assert calls["core"] == calls["updates"] == int(enabled)
    assert result["stats"]["benders_cuts"] == 1
