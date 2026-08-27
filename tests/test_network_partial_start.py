from __future__ import annotations

import gurobipy as gp
import pytest
from gurobipy import GRB

from src.model.gurobi_model import MLNDModel
from src.model.primal_heuristic import DEFAULT_STRUCTURE_GROUPS


def _start_values(model: gp.Model, index: int) -> dict[str, float]:
    model.Params.StartNumber = index
    model.update()
    return {variable.VarName: float(variable.Start) for variable in model.getVars()}


def test_network_fallback_pool_preserves_complete_and_adds_partial_start() -> None:
    model = gp.Model("network_partial_start_test")
    model.Params.OutputFlag = 0
    x = model.addVars(("L1", "H1"), vtype=GRB.BINARY, name="x")
    z = model.addVars(("H1",), vtype=GRB.BINARY, name="z")
    delta = model.addVars(("S1",), vtype=GRB.BINARY, name="delta")
    alpha = model.addVars(("D1",), vtype=GRB.BINARY, name="alpha")
    y = model.addVars(("A1",), vtype=GRB.BINARY, name="y")
    gamma = model.addVar(vtype=GRB.INTEGER, name="gamma")
    chi = model.addVars(("OD1",), lb=0.0, name="chi")
    queue_v2 = model.addVars(("Q1",), lb=0.0, name="queue_v2")
    model.update()

    wrapper = MLNDModel.__new__(MLNDModel)
    wrapper.model = model
    wrapper.vars = {
        "x": x,
        "z": z,
        "delta": delta,
        "alpha": alpha,
        "y": y,
        "gamma": gamma,
        "chi": chi,
        "queue_v2": queue_v2,
    }
    wrapper.network_partial_start_groups = DEFAULT_STRUCTURE_GROUPS
    wrapper.network_partial_start_installed = False
    wrapper.network_partial_start_index = None
    wrapper.network_partial_start_variable_count = 0

    complete_values = {
        x["L1"]: 1.0,
        x["H1"]: 0.0,
        z["H1"]: 1.0,
        delta["S1"]: 0.0,
        alpha["D1"]: 1.0,
        y["A1"]: 1.0,
        gamma: 2.0,
        chi["OD1"]: 0.999999,
        queue_v2["Q1"]: 1.000001,
    }
    model.NumStart = 1
    model.update()
    model.Params.StartNumber = 0
    for variable, value in complete_values.items():
        variable.Start = value
    model.update()

    wrapper._append_network_fallback_starts()

    assert model.NumStart == 3
    assert wrapper.network_partial_start_installed is True
    assert wrapper.network_partial_start_index == 1
    assert wrapper.network_partial_start_variable_count == 5

    full = _start_values(model, 0)
    for variable, expected in complete_values.items():
        assert full[variable.VarName] == pytest.approx(expected)

    partial = _start_values(model, 1)
    structural_variables = {
        variable.VarName
        for group in DEFAULT_STRUCTURE_GROUPS
        for variable in wrapper.vars[group].values()
    }
    for variable in model.getVars():
        value = partial[variable.VarName]
        if variable.VarName in structural_variables:
            assert value == pytest.approx(float(complete_values[variable] >= 0.5))
        else:
            assert value == GRB.UNDEFINED

    courier = _start_values(model, 2)
    for variable in model.getVars():
        expected = 1.0 if variable is queue_v2["Q1"] else 0.0
        assert courier[variable.VarName] == pytest.approx(expected)
