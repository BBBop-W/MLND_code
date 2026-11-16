"""Incremental columns must reproduce the full model and its certificates."""

import numpy as np
import pytest

from src.data.instances import to_planning_data
from src.model.paths import enumerate_paths
from src.solvers import solve
from src.solvers.algorithm import optimize
from src.solvers.formulation import (
    build_model,
    add_path_columns,
    read_solution,
    shapes,
    BuildStopped,
)


def opened(data):
    design = {name: np.zeros(shape) for name, shape in shapes(data).items()}
    design["facility"][:] = 1
    design["capacity_level"][:, -1] = 1
    design["assignment"][:, 0, 0] = 1
    design["airline"][:] = data.allowed_links
    return design


def test_incremental_objective_and_resources_equal_full_model(small_input):
    data = to_planning_data(small_input)
    paths = enumerate_paths(data)
    full = build_model(data, paths, fixed_design=opened(data))
    rmp = build_model(
        data, paths, fixed_design=opened(data), service_indices=(), empty_indices=()
    )
    try:
        optimize(rmp.model, seconds=5)
        before = rmp.model.NumVars
        count = add_path_columns(
            rmp, range(len(paths.service_paths)), range(len(paths.empty_paths))
        )
        assert rmp.model.NumVars == before + count
        assert add_path_columns(rmp, range(len(paths.service_paths))) == 0
        for built in (rmp, full):
            optimize(built.model, seconds=5)
        assert read_solution(rmp)["objective"] == pytest.approx(
            read_solution(full)["objective"], abs=1e-6
        )

        # Compare by variable names because the insertion order intentionally differs.
        def columns(built):
            m = built.model
            matrix = m.getA().tocsc()
            rows = [row.ConstrName for row in m.getConstrs()]
            return {
                v.VarName: (
                    v.Obj,
                    v.LB,
                    v.UB,
                    {
                        rows[i]: float(matrix[i, v.index])
                        for i in matrix[:, v.index].nonzero()[0]
                    },
                )
                for v in m.getVars()
            }

        assert columns(rmp) == columns(full)
    finally:
        rmp.model.dispose()
        full.model.dispose()


def test_incremental_column_limit_precedes_mutation(small_input, monkeypatch):
    from src.solvers import formulation

    data = to_planning_data(small_input)
    paths = enumerate_paths(data)
    rmp = build_model(
        data, paths, fixed_design=opened(data), service_indices=(), empty_indices=()
    )
    try:
        before = rmp.model.NumVars
        monkeypatch.setattr(formulation, "MAX_LIFTED_VARIABLES", before)
        with pytest.raises(BuildStopped, match="model_size_limit"):
            add_path_columns(rmp, [0])
        assert rmp.model.NumVars == before
        assert not rmp.variables["flow"]
    finally:
        rmp.model.dispose()


def test_pareto_switch_reaches_actual_solver_and_preserves_optimum(small_input):
    data = to_planning_data(small_input)
    with_pareto = solve(data, method="benders_cg", time_limit=10, mip_gap=1e-6)
    without = solve(
        data,
        method="benders_cg",
        accelerations={"pareto_cuts": False},
        time_limit=10,
        mip_gap=1e-6,
    )
    assert with_pareto["acceleration_statistics"]["pareto_cuts"] > 0
    assert without["acceleration_statistics"]["pareto_attempts"] == 0
    assert with_pareto["objective"] == pytest.approx(without["objective"], abs=1e-5)
    assert with_pareto["gap"] <= 1e-6 and without["gap"] <= 1e-6


def test_plain_counts_generated_columns_without_reusing(small_input):
    result = solve(
        to_planning_data(small_input),
        method="benders_cg_plain",
        time_limit=10,
        mip_gap=1e-6,
    )
    stats = result["acceleration_statistics"]
    assert stats["unique_service_columns"] > 0
    assert stats["reused_service_columns"] == 0
    assert stats["pareto_attempts"] == 0
