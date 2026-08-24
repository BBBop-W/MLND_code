from __future__ import annotations

import gurobipy as gp
import pytest
from gurobipy import GRB

from src.model.primal_heuristic import (
    LocalBranchingConfig,
    PrimalSeed,
    build_hamming_expression,
    improve_mip_start,
    temporary_local_branching_neighborhood,
)


def _empty_structure_groups() -> dict[str, gp.tupledict]:
    return {name: gp.tupledict() for name in ("z", "delta", "alpha")}


def _synthetic_conic_model() -> tuple[gp.Model, dict[str, object], dict[gp.Var, float]]:
    model = gp.Model("exact_conic_local_branching_test")
    model.Params.OutputFlag = 0
    x = model.addVars(("a", "b"), vtype=GRB.BINARY, name="x")
    u = model.addVar(lb=0.0, name="u")
    t = model.addVar(lb=0.0, name="t")
    model.addConstr(u == 1.0, name="fixed_soc_coordinate")
    model.addQConstr(u * u <= t * t, name="soc")
    model.setObjective(10.0 * x["a"] + 2.0 * x["b"] + t, GRB.MINIMIZE)
    model.update()
    variables: dict[str, object] = {"x": x, **_empty_structure_groups()}
    seed_values = {x["a"]: 1.0, x["b"]: 0.0, u: 1.0, t: 1.0}
    return model, variables, seed_values


def test_hamming_expression_has_the_exact_binary_distance_coefficients() -> None:
    model = gp.Model()
    model.Params.OutputFlag = 0
    x = model.addVars(("zero", "one"), vtype=GRB.BINARY, name="x")
    model.update()
    variables: dict[str, object] = {"x": x, **_empty_structure_groups()}
    expression, count = build_hamming_expression(
        variables,
        {"x": {"zero": 0, "one": 1}, "z": {}, "delta": {}, "alpha": {}},
    )

    assert count == 2
    assert expression.getConstant() == pytest.approx(1.0)
    assert expression.getCoeff(0) == pytest.approx(1.0)
    assert expression.getCoeff(1) == pytest.approx(-1.0)


def test_exact_conic_local_branching_improves_start_and_restores_model() -> None:
    model, variables, seed_values = _synthetic_conic_model()
    x = variables["x"]
    seed = PrimalSeed(
        objective=11.0,
        structure={"x": {"a": 1, "b": 0}, "z": {}, "delta": {}, "alpha": {}},
        values=seed_values,
    )

    model.Params.TimeLimit = 77.0
    model.Params.MIPGap = 0.123
    model.Params.MIPFocus = 2
    model.Params.Heuristics = 0.17
    model.Params.NoRelHeurTime = 0.25
    original_parameters = {
        name: getattr(model.Params, name)
        for name in ("TimeLimit", "MIPGap", "MIPFocus", "Heuristics", "NoRelHeurTime")
    }
    original_bounds = {variable: (variable.LB, variable.UB) for variable in model.getVars()}
    original_linear_constraints = model.NumConstrs
    original_quadratic_constraints = model.NumQConstrs

    result = improve_mip_start(
        model,
        variables,
        seed,
        LocalBranchingConfig(radius=1, time_limit=5.0, mip_gap=0.0, output_flag=0),
    )

    assert result.improved is True
    assert result.candidate_objective == pytest.approx(1.0)
    assert result.installed_objective == pytest.approx(1.0)
    assert result.hamming_distance == 1
    assert result.selected_binary_count == 2
    assert model.NumConstrs == original_linear_constraints
    assert model.NumQConstrs == original_quadratic_constraints
    for name, value in original_parameters.items():
        assert getattr(model.Params, name) == pytest.approx(value)
    for variable, bounds in original_bounds.items():
        assert (variable.LB, variable.UB) == pytest.approx(bounds)
    assert model.NumStart == 1
    model.Params.StartNumber = 0
    assert x["a"].Start == pytest.approx(0.0)
    assert x["b"].Start == pytest.approx(0.0)


def test_temporary_neighborhood_restores_after_exception() -> None:
    model, variables, _ = _synthetic_conic_model()
    x = variables["x"]
    expression, _ = build_hamming_expression(
        variables,
        {"x": {"a": 1, "b": 0}, "z": {}, "delta": {}, "alpha": {}},
    )
    model.Params.TimeLimit = 91.0
    original_constraint_count = model.NumConstrs
    original_bounds = {variable: (variable.LB, variable.UB) for variable in model.getVars()}

    with pytest.raises(RuntimeError, match="synthetic failure"):
        with temporary_local_branching_neighborhood(
            model,
            expression,
            1,
            objective_cutoff=10.0,
            parameter_overrides={"TimeLimit": 2.0},
        ):
            x["a"].LB = 1.0
            x["b"].UB = 0.0
            model.Params.TimeLimit = 3.0
            raise RuntimeError("synthetic failure")

    assert model.NumConstrs == original_constraint_count
    assert model.Params.TimeLimit == pytest.approx(91.0)
    for variable, bounds in original_bounds.items():
        assert (variable.LB, variable.UB) == pytest.approx(bounds)

