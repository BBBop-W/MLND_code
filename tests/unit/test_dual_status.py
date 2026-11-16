"""A stopped dual keeps its actual cause and any previously valid evidence."""

import pytest
from gurobipy import GRB

from src.model.paths import enumerate_paths
from src.solvers import algorithm, dual, pareto
from src.solvers.formulation import BuildStopped, build_model
from test_solver import data, design


@pytest.mark.parametrize(
    "status, reason", [(GRB.TIME_LIMIT, "time_limit"), (GRB.INTERRUPTED, "user_stop")]
)
def test_dual_with_no_solution_preserves_stop_status(monkeypatch, status, reason):
    d = data()
    built = build_model(d, enumerate_paths(d), fixed_design=design(d))
    real_model = dual.gp.Model

    class StoppedModel:
        def __init__(self, *args, **kwargs):
            self.model = real_model(*args, **kwargs)

        def __getattr__(self, name):
            if name == "Status":
                return status
            if name == "SolCount":
                return 0
            return getattr(self.model, name)

        def optimize(self, *args):
            pass

    monkeypatch.setattr(dual.gp, "Model", StoppedModel)
    try:
        with pytest.raises(BuildStopped) as error:
            dual.solve_dual(built)
        assert error.value.reason == reason
    finally:
        built.model.dispose()


@pytest.mark.parametrize("reason", ["time_limit", "user_stop"])
def test_stopped_ordinary_dual_keeps_the_verified_primal(monkeypatch, reason):
    def stopped(*args, **kwargs):
        raise BuildStopped(reason)

    monkeypatch.setattr(algorithm, "solve_dual", stopped)
    d = data()
    answer = algorithm.solve_subproblem(d, enumerate_paths(d), design(d), time_limit=10)
    assert answer["termination_reason"] == reason
    assert answer["solution"]["independently_validated"]
    assert answer["cut"] is None
    assert not answer["pricing_complete"]


@pytest.mark.parametrize("reason", ["time_limit", "user_stop"])
def test_stopped_pareto_dual_keeps_the_ordinary_certified_cut(monkeypatch, reason):
    d = data()
    paths = enumerate_paths(d)
    chosen = design(d)
    ordinary = algorithm.solve_subproblem(d, paths, chosen, time_limit=10)
    assert ordinary["pricing_complete"]
    built = build_model(d, paths, fixed_design=chosen)

    def stopped(*args, **kwargs):
        raise BuildStopped(reason)

    monkeypatch.setattr(pareto, "solve_dual", stopped)
    try:
        answer = pareto.solve_pareto_cut(
            built,
            ordinary["cut"],
            chosen,
            pareto.CorePoint(d).point,
            ordinary["solution"]["continuous_cost"],
        )
        assert answer["reason"] == reason
        assert not answer["accepted"]
        assert answer["cut"] is ordinary["cut"]
        assert answer["cut"]["pricing_complete"]
    finally:
        built.model.dispose()
