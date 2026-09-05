"""Focused regressions for transactional MLND network-start construction."""

from __future__ import annotations

from pathlib import Path

import gurobipy as gp
import pytest
from gurobipy import GRB

from src.config import SolverConfig
from src.data_processor.loader import DataLoader
from src.model.gurobi_model import MLNDModel
from src.model.primal_heuristic import (
    FixedIntegerRepairConfig,
    PrimalSeed,
    repair_fixed_integer_design_start,
)
from tests.test_new_pipeline import _exact_model_config, _write_canonical_instance


def _wrapper(tmp_path: Path, mode: str) -> MLNDModel:
    instance = DataLoader().load_instance_dir(
        str(_write_canonical_instance(tmp_path / f"warm_start_{mode.lower()}")),
        model_config=_exact_model_config(),
    )
    wrapper = MLNDModel(instance, mode)
    wrapper.build()
    wrapper.model.Params.OutputFlag = 0
    wrapper.model.Params.Threads = 1
    return wrapper


def _network_config(*, start_mode: str = "network") -> SolverConfig:
    return SolverConfig(
        time_limit=10.0,
        mip_gap=0.0,
        threads=1,
        output_flag=0,
        start_mode=start_mode,
        network_start_seconds=2.0,
        network_start_method=2,
        use_valid_inequalities=False,
        use_user_cuts=False,
        use_local_branching=False,
        local_branching_seconds=0.0,
        root_relaxation_method=-1,
        miqcp_method=1,
    )


def _construct_without_local_branching(
    wrapper: MLNDModel,
    config: SolverConfig,
    *,
    include_courier_start: bool,
) -> None:
    wrapper._construct_network_start_with_budgets(
        config.network_start_seconds,
        0.0,
        0.0,
        0.20 * config.time_limit,
        config,
        include_courier_start=include_courier_start,
    )


def test_restricted_chi_restores_every_original_finite_upper_bound(
    tmp_path: Path,
) -> None:
    """The auxiliary restriction must be a transaction, not a bound reset."""

    wrapper = _wrapper(tmp_path, "MX")
    chi = wrapper.vars["chi"]
    for index, variable in enumerate(chi.values()):
        variable.UB = 0.40 + 0.01 * (index % 5)
    wrapper.model.update()
    original_upper_bounds = {key: float(variable.UB) for key, variable in chi.items()}

    _construct_without_local_branching(
        wrapper,
        _network_config(),
        include_courier_start=False,
    )

    assert {
        key: float(variable.UB) for key, variable in chi.items()
    } == original_upper_bounds


def test_lo_network_and_courier_preserves_f884_three_start_pool(
    tmp_path: Path,
) -> None:
    wrapper = _wrapper(tmp_path, "LO")
    config = _network_config(start_mode="network_and_courier")

    solved = wrapper.solve(config)

    assert solved
    assert wrapper.network_start_found
    assert wrapper.network_start_used
    assert wrapper.model.NumStart == 3

    model_variables = wrapper.model.getVars()
    wrapper.model.Params.StartNumber = 0
    network_values = wrapper.model.getAttr(GRB.Attr.Start, model_variables)
    assert all(abs(float(value)) < 0.99 * GRB.INFINITY for value in network_values)
    assert sum(float(variable.Start) for variable in wrapper.vars["x"].values()) > 0.5
    assert (
        sum(float(variable.Start) for variable in wrapper.vars["chi"].values()) > 1e-8
    )

    wrapper.model.Params.StartNumber = 1
    partial_values = wrapper.model.getAttr(GRB.Attr.Start, model_variables)
    assert any(abs(float(value)) >= 0.99 * GRB.UNDEFINED for value in partial_values)

    wrapper.model.Params.StartNumber = 2
    courier_values = wrapper.model.getAttr(GRB.Attr.Start, model_variables)
    assert all(abs(float(value)) < 0.99 * GRB.INFINITY for value in courier_values)
    assert all(
        variable.Start == pytest.approx(0.0) for variable in wrapper.vars["x"].values()
    )
    assert all(
        variable.Start == pytest.approx(0.0)
        for variable in wrapper.vars["chi"].values()
    )
    assert all(
        variable.Start == pytest.approx(1.0)
        for variable in wrapper.vars["queue_v2"].values()
    )
    assert all(
        variable.Start == pytest.approx(1.0)
        for variable in wrapper.vars["queue_t"].values()
    )
    assert network_values != courier_values
    wrapper.model.Params.StartNumber = 0


def test_auxiliary_optimize_exception_restores_temporary_model_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An auxiliary failure may propagate, but may not leave model mutations."""

    wrapper = _wrapper(tmp_path, "MX")
    x = wrapper.vars["x"]
    alpha = wrapper.vars["alpha"]
    chi = wrapper.vars["chi"]
    original_x_bounds = {
        key: (float(variable.LB), float(variable.UB)) for key, variable in x.items()
    }
    original_alpha_bounds = {
        key: (float(variable.LB), float(variable.UB)) for key, variable in alpha.items()
    }
    original_chi_bounds = {
        key: (float(variable.LB), float(variable.UB)) for key, variable in chi.items()
    }
    original_constraint_count = wrapper.model.NumConstrs
    wrapper.model.Params.Method = 0
    wrapper.model.Params.MIQCPMethod = 0

    def fail_auxiliary_optimize(
        failed_model: gp.Model, *_args: object, **_kwargs: object
    ) -> None:
        assert failed_model.Params.Method == 2
        assert failed_model.Params.MIQCPMethod == 1
        raise RuntimeError("synthetic auxiliary optimize failure")

    monkeypatch.setattr(gp.Model, "optimize", fail_auxiliary_optimize)
    with pytest.raises(RuntimeError, match="synthetic auxiliary optimize failure"):
        _construct_without_local_branching(
            wrapper,
            _network_config(),
            include_courier_start=False,
        )

    assert {
        key: (float(variable.LB), float(variable.UB)) for key, variable in x.items()
    } == original_x_bounds
    assert {
        key: (float(variable.LB), float(variable.UB)) for key, variable in alpha.items()
    } == original_alpha_bounds
    assert {
        key: (float(variable.LB), float(variable.UB)) for key, variable in chi.items()
    } == original_chi_bounds
    assert wrapper.model.NumConstrs == original_constraint_count
    assert wrapper.model.Params.Method == 0
    assert wrapper.model.Params.MIQCPMethod == 0
    assert not any(
        constraint.ConstrName == "__network_start_positive_chi"
        for constraint in wrapper.model.getConstrs()
    )


def _fixed_integer_repair_model() -> tuple[gp.Model, gp.Var, gp.Var, gp.Var, gp.Var]:
    model = gp.Model("fixed_integer_design_repair")
    model.Params.OutputFlag = 0
    binary = model.addVar(vtype=GRB.BINARY, name="design_binary")
    integer = model.addVar(lb=0.0, ub=3.0, vtype=GRB.INTEGER, name="design_integer")
    continuous = model.addVar(lb=0.0, ub=20.0, name="continuous_repair")
    epigraph = model.addVar(lb=0.0, name="conic_epigraph")
    model.addConstr(
        continuous == 2.0 * binary + integer,
        name="continuous_balance",
    )
    model.addQConstr(continuous * continuous <= epigraph, name="convex_conic")
    model.setObjective(epigraph + 0.1 * binary, GRB.MINIMIZE)
    model.update()
    return model, binary, integer, continuous, epigraph


def _repair_seed(
    binary: gp.Var,
    integer: gp.Var,
    continuous: gp.Var,
    epigraph: gp.Var,
) -> PrimalSeed:
    return PrimalSeed(
        objective=0.1,
        structure={},
        values={
            binary: 1.0,
            integer: 1.0,
            continuous: 0.0,
            epigraph: 0.0,
        },
    )


def test_fixed_integer_design_repairs_continuous_start_transactionally() -> None:
    model, binary, integer, continuous, epigraph = _fixed_integer_repair_model()
    seed = _repair_seed(binary, integer, continuous, epigraph)
    model.Params.TimeLimit = 77.0
    model.Params.MIPGap = 0.123
    model.Params.Threads = 3
    model.Params.Method = 0
    model.Params.MIQCPMethod = 0
    original_parameters = {
        name: getattr(model.Params, name)
        for name in (
            "TimeLimit",
            "MIPGap",
            "Threads",
            "Method",
            "MIQCPMethod",
            "OutputFlag",
        )
    }
    model_variables = model.getVars()
    original_bounds = {
        variable: (float(variable.LB), float(variable.UB))
        for variable in model_variables
    }
    original_objective = model.getAttr(GRB.Attr.Obj, model_variables)
    original_linear_constraints = model.NumConstrs
    original_quadratic_constraints = model.NumQConstrs

    result = repair_fixed_integer_design_start(
        model,
        seed,
        FixedIntegerRepairConfig(
            time_limit=5.0,
            mip_gap=0.0,
            method=2,
            miqcp_method=1,
            output_flag=0,
        ),
    )

    assert result.found_incumbent
    assert result.installed_source == "repaired"
    assert result.fixed_integer_count == 2
    assert result.candidate_objective == pytest.approx(9.1)
    assert result.installed_objective == pytest.approx(9.1)
    assert model.NumConstrs == original_linear_constraints
    assert model.NumQConstrs == original_quadratic_constraints
    assert model.getAttr(GRB.Attr.Obj, model_variables) == original_objective
    for name, value in original_parameters.items():
        assert getattr(model.Params, name) == pytest.approx(value)
    for variable, bounds in original_bounds.items():
        assert (float(variable.LB), float(variable.UB)) == pytest.approx(bounds)

    assert model.NumStart == 1
    model.Params.StartNumber = 0
    installed_values = model.getAttr(GRB.Attr.Start, model_variables)
    assert installed_values == pytest.approx([1.0, 1.0, 3.0, 9.0])


def test_fixed_integer_design_keeps_input_seed_when_no_incumbent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, binary, integer, continuous, epigraph = _fixed_integer_repair_model()
    seed = _repair_seed(binary, integer, continuous, epigraph)
    model_variables = model.getVars()
    original_bounds = {
        variable: (float(variable.LB), float(variable.UB))
        for variable in model_variables
    }

    def return_without_incumbent(
        _model: gp.Model, *_args: object, **_kwargs: object
    ) -> None:
        return None

    monkeypatch.setattr(gp.Model, "optimize", return_without_incumbent)
    result = repair_fixed_integer_design_start(
        model,
        seed,
        FixedIntegerRepairConfig(time_limit=1.0, mip_gap=0.0, output_flag=0),
    )

    assert not result.found_incumbent
    assert result.installed_source == "seed"
    assert result.candidate_objective is None
    assert result.installed_objective == pytest.approx(seed.objective)
    assert model.NumStart == 1
    model.Params.StartNumber = 0
    assert model.getAttr(GRB.Attr.Start, model_variables) == pytest.approx(
        [seed.values[variable] for variable in model_variables]
    )
    for variable, bounds in original_bounds.items():
        assert (float(variable.LB), float(variable.UB)) == pytest.approx(bounds)


def test_fixed_integer_design_exception_restores_bounds_parameters_and_starts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, binary, integer, continuous, epigraph = _fixed_integer_repair_model()
    seed = _repair_seed(binary, integer, continuous, epigraph)
    model_variables = model.getVars()
    previous_start = [0.0, 0.0, 0.0, 0.0]
    model.NumStart = 1
    model.update()
    model.Params.StartNumber = 0
    model.setAttr(GRB.Attr.Start, model_variables, previous_start)
    model.Params.TimeLimit = 83.0
    model.Params.MIPGap = 0.234
    model.Params.Threads = 2
    model.Params.Method = 0
    model.Params.MIQCPMethod = 0
    original_parameters = {
        name: getattr(model.Params, name)
        for name in (
            "TimeLimit",
            "MIPGap",
            "Threads",
            "Method",
            "MIQCPMethod",
            "OutputFlag",
        )
    }
    original_bounds = {
        variable: (float(variable.LB), float(variable.UB))
        for variable in model_variables
    }

    def fail_after_mutation(
        failed_model: gp.Model, *_args: object, **_kwargs: object
    ) -> None:
        binary.LB = 0.0
        integer.UB = 3.0
        continuous.LB = 7.0
        epigraph.UB = 11.0
        failed_model.Params.MIPGap = 0.9
        assert failed_model.Params.Method == 2
        assert failed_model.Params.MIQCPMethod == 1
        raise RuntimeError("synthetic fixed-design repair failure")

    monkeypatch.setattr(gp.Model, "optimize", fail_after_mutation)
    with pytest.raises(RuntimeError, match="synthetic fixed-design repair failure"):
        repair_fixed_integer_design_start(
            model,
            seed,
            FixedIntegerRepairConfig(
                time_limit=1.0,
                mip_gap=0.0,
                method=2,
                miqcp_method=1,
                output_flag=0,
            ),
        )

    for name, value in original_parameters.items():
        assert getattr(model.Params, name) == pytest.approx(value)
    for variable, bounds in original_bounds.items():
        assert (float(variable.LB), float(variable.UB)) == pytest.approx(bounds)
    assert model.NumStart == 1
    model.Params.StartNumber = 0
    assert model.getAttr(GRB.Attr.Start, model_variables) == pytest.approx(
        previous_start
    )
