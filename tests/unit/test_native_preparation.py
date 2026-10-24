"""Long-run preparation contract; all optimize calls are Python mocks only."""

from pathlib import Path
from types import SimpleNamespace
import json

import pytest

from src.experiments import native
from src.experiments.storage import sha256
from src.model.types import PathCollection
from src.reporting.native import _read_run, native_phase_times

EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "tsl_2l20h50od"


@pytest.fixture(autouse=True)
def forbid_real_optimize(monkeypatch):
    import gurobipy as gp

    def forbidden(*args, **kwargs):
        raise AssertionError("Preparation tests must never invoke a real optimizer")

    monkeypatch.setattr(gp.Model, "optimize", forbidden)


@pytest.fixture
def fake_reference(monkeypatch):
    from gurobipy import GRB
    from src.model import paths
    from src.solvers import formulation

    calls = {}

    class Model:
        Params = SimpleNamespace()
        SolCount = 0
        Status = GRB.TIME_LIMIT
        ObjBound = 0.0
        NodeCount = 0.0
        NumVars = 12
        NumConstrs = 17
        NumQConstrs = 2
        disposed = False

        def optimize(self, callback):
            calls["callback"] = callback
            calls["optimization_limit"] = self.Params.TimeLimit
            callback(self, GRB.Callback.MIP)

        def cbGet(self, code):
            return {
                GRB.Callback.MIP_OBJBST: GRB.INFINITY,
                GRB.Callback.MIP_OBJBND: 0.0,
                GRB.Callback.MIP_NODCNT: 0.0,
            }[code]

        def terminate(self):
            calls["terminated"] = True

        def dispose(self):
            self.disposed = True

    model = Model()
    catalogue = PathCollection((), (), (), complete=True, scope="direct")

    def enumerate_mock(data, **kwargs):
        calls["enumeration"] = kwargs
        return catalogue

    def build_mock(data, path_set, **kwargs):
        assert path_set is catalogue
        calls["build"] = kwargs
        return SimpleNamespace(model=model, data=data, paths=path_set, variables={})

    monkeypatch.setattr(paths, "enumerate_paths", enumerate_mock)
    monkeypatch.setattr(formulation, "build_model", build_mock)
    monkeypatch.setattr(
        native, "align_native_arc_bounds", lambda *args, **kwargs: {"mock": True}
    )
    return calls, model


def test_mock_full_run_keeps_exhaustive_budget_and_complete_output(
    tmp_path, fake_reference
):
    calls, model = fake_reference
    result = native.gurobi_reference(
        EXAMPLE, tmp_path, max_orders=1, seconds=60, gap=0, max_model_variables=0
    )
    assert "column_limit" not in calls["enumeration"]
    assert "service_indices" not in calls["build"]
    assert calls["build"]["max_lifted_variables"] is None
    assert (
        0
        < calls["optimization_limit"]
        <= calls["build"]["time_limit"]
        <= calls["enumeration"]["time_limit"]
        <= 60
    )
    assert model.disposed
    assert result["path_source"] == "exhaustive_enumeration"
    assert result["catalogue_complete"] and result["construction_complete"]
    assert not result["has_solver_solution"]
    assert result["incumbent_kind"] == "all_courier_baseline"
    assert result["independently_validated"]
    for name in [
        "costs",
        "orders",
        "lockers",
        "bases",
        "fleet",
        "links",
        "paths",
        "od_assignments",
        "busy_components",
    ]:
        assert (tmp_path / "operations" / (name + ".csv")).is_file()
    validation = json.loads((tmp_path / "native_validation.json").read_text())
    assert validation["checks"]["passed"]
    assert validation["operations_sha256"] == sha256(
        tmp_path / "operations" / "operations.json"
    )
    assert {"bases.csv", "lockers.csv", "operations.json", "OPERATIONS.md"} <= set(
        validation["operations_files_sha256"]
    )
    for name, digest in validation["operations_files_sha256"].items():
        assert sha256(tmp_path / "operations" / name) == digest
    assert validation["result_sha256"] == sha256(tmp_path / "result.json")
    checkpoint = json.loads((tmp_path / "checkpoint.json").read_text())
    assert checkpoint["validated"] and checkpoint["fallback"]
    for file in checkpoint["files"].values():
        assert sha256(tmp_path / file["path"]) == file["sha256"]
    events = [
        json.loads(row)
        for row in (tmp_path / "progress.jsonl").read_text().splitlines()
    ]
    assert events[-1] == json.loads((tmp_path / "progress.json").read_text())
    assert events[-1]["validated_incumbent"] is True
    assert all(
        "timestamp_utc" in event and "elapsed_seconds" in event for event in events
    )
    assert [event["elapsed_seconds"] for event in events] == sorted(
        event["elapsed_seconds"] for event in events
    )
    solver_event = next(
        event for event in events if event["event"] == "solver_progress"
    )
    assert solver_event["objective"] is None and solver_event["upper_bound"] is None
    assert solver_event["validated_incumbent"] is False
    assert solver_event["bound_scope"] == "global"
    timing = native_phase_times(result)
    assert sum(timing["phases"].values()) == pytest.approx(
        result["solve_wall_before_validation"]
    )
    assert "validation_and_export" not in timing["phases"]
    assert (
        _read_run(tmp_path, "mock", "mock", "black", 1)["row"]["state"] == "validated"
    )
    locker_table = tmp_path / "operations" / "lockers.csv"
    locker_table.write_text(locker_table.read_text() + "\n")
    with pytest.raises(ValueError, match="stale for operations tables"):
        _read_run(tmp_path, "mock", "mock", "black", 1)


@pytest.mark.parametrize("reason", ["time_limit", "user_stop"])
def test_enumeration_stop_returns_valid_fallback_without_claiming_full_solve(
    tmp_path, monkeypatch, fake_reference, reason
):
    from src.model import paths

    def stopped(*args, **kwargs):
        raise paths.EnumerationStopped(reason, columns=17, initial_segments=9)

    monkeypatch.setattr(paths, "enumerate_paths", stopped)
    calls, _ = fake_reference
    result = native.gurobi_reference(
        EXAMPLE, tmp_path, max_orders=2, seconds=60, max_model_variables=0
    )
    assert "build" not in calls and "callback" not in calls
    assert result["termination_reason"] == reason
    assert result["stopped_stage"] == "enumeration"
    assert (
        not result["catalogue_complete"]
        and not result["construction_complete"]
        and not result["pricing_complete"]
    )
    assert result["preprocessing_stop"]["columns_constructed"] == 17
    assert result["bound"] == 0 and result["gap"] == 1
    assert result["independently_validated"]


def test_build_limit_keeps_full_catalogue_but_unbuilt_model(
    tmp_path, monkeypatch, fake_reference
):
    from src.solvers import formulation

    def stopped(*args, **kwargs):
        raise formulation.BuildStopped("model_size_limit")

    monkeypatch.setattr(formulation, "build_model", stopped)
    result = native.gurobi_reference(EXAMPLE, tmp_path, max_orders=1, seconds=60)
    assert result["catalogue_complete"] and not result["construction_complete"]
    assert not result["pricing_complete"]
    assert result["termination_reason"] == "model_size_limit"
    assert result["independently_validated"]


def test_unexpected_build_failure_retains_validated_fallback_and_failure_record(
    tmp_path, monkeypatch, fake_reference
):
    from src.solvers import formulation

    def broken(*args, **kwargs):
        raise MemoryError("mock allocation failure")

    monkeypatch.setattr(formulation, "build_model", broken)
    with pytest.raises(RuntimeError, match="validated fallback"):
        native.gurobi_reference(EXAMPLE, tmp_path, max_orders=1, seconds=60)
    result = json.loads((tmp_path / "result.json").read_text())
    failure = json.loads((tmp_path / "failure.json").read_text())
    assert failure["stage"] == "build" and failure["error_type"] == "MemoryError"
    assert result["execution_failed"] and result["termination_reason"] == "error"
    assert result["independently_validated"] and not result["has_solver_solution"]


def test_validation_failure_preserves_raw_solution_and_previous_valid_checkpoint(
    tmp_path, monkeypatch, fake_reference
):
    def failed(*args, **kwargs):
        raise ValueError("mock physical residual")

    monkeypatch.setattr(native, "export_operations", failed)
    with pytest.raises(ValueError, match="physical residual"):
        native.gurobi_reference(EXAMPLE, tmp_path, max_orders=1, seconds=60)
    result = json.loads((tmp_path / "result.json").read_text())
    assert not result["independently_validated"]
    assert (tmp_path / "solution.json").exists() and (
        tmp_path / "path_flows.json"
    ).exists()
    assert not (tmp_path / "native_validation.json").exists()
    assert json.loads((tmp_path / "failure.json").read_text())["raw_solution_saved"]
    checkpoint = json.loads((tmp_path / "checkpoint.json").read_text())
    assert checkpoint["validated"] and checkpoint["fallback"]
    assert checkpoint["directory"] == "recovery/all_courier"


def test_callback_terminates_at_inclusive_deadline_and_rejects_infinite_bounds(
    tmp_path,
):
    from gurobipy import GRB

    monitor = native._ReferenceProgress(tmp_path, native.time.perf_counter() - 10, 1)

    class Model:
        terminated = False

        def terminate(self):
            self.terminated = True

        def cbGet(self, code):
            return GRB.INFINITY

    model = Model()
    native._gurobi_progress_callback(monitor, GRB)(model, GRB.Callback.MIP)
    assert model.terminated
    point = json.loads((tmp_path / "progress.json").read_text())
    assert (
        point["objective"] is None and point["bound"] is None and point["gap"] is None
    )


def test_disjoint_benders_timings_include_root_heuristic_capacity_and_subproblem_remainder():
    result = {
        "solver": "CPLEX",
        "wall": 101.5,
        "wall_before_output": 100.0,
        "input_read_seconds": 1.0,
        "support_preprocessing_seconds": 2.0,
        "enumeration_seconds": 0,
        "timings": {
            "greedy_initial_columns": 3.0,
            "master_build": 4.0,
            "root_master": 5.0,
            "master_optimize": 40.0,
            "master_update": 1.0,
            "capacity_cut_build": 0.1,
            "capacity_cut_separation": 0.2,
            "restricted_mip": 10.0,
            "incumbent_cleanup": 0.3,
            "subproblem_wall": 20.0,
            "subproblem": {"build": 4.0, "primal": 2.0, "dual": 7.0, "pricing": 3.0},
        },
    }
    phases = native_phase_times(result)["phases"]
    assert sum(phases.values()) == pytest.approx(100)
    assert phases["subproblem_other"] == 4
    assert phases["other"] == pytest.approx(13.4)
    assert "subproblem_wall" not in phases


def test_callback_value_accessor_reconstructs_without_an_optimized_model(small_input):
    from src.data.instances import to_planning_data
    from src.model.paths import enumerate_paths
    from src.solvers.formulation import build_model, read_solution, BuildStopped
    from src.operations.validation import build_operations

    data = to_planning_data(small_input)
    paths = enumerate_paths(data)
    with pytest.raises(BuildStopped, match="model_size_limit"):
        build_model(data, paths, max_lifted_variables=1)
    built = build_model(data, paths, max_lifted_variables=None)
    try:

        def value(var):
            name = var.VarName
            return (
                float(data.demand[int(name[8:-1])])
                if name.startswith("courier[")
                else 0.0
            )

        result = read_solution(built, value=value)
        assert result["objective"] == pytest.approx(
            float(data.demand @ data.ground_cost)
        )
        assert build_operations(small_input, result)["summary"]["checks"]["passed"]
    finally:
        built.model.dispose()


def test_periodic_callback_checkpoint_uses_physical_validation(
    tmp_path, monkeypatch, fake_reference
):
    from gurobipy import GRB
    from src.solvers import formulation, algorithm

    calls, model = fake_reference
    monkeypatch.setattr(
        formulation,
        "read_solution",
        lambda built, **kwargs: algorithm.baseline(built.data),
    )
    model.getVars = lambda: []
    model.cbGetSolution = lambda variables: []
    model.cbGet = lambda code: 20000.0 if code == GRB.Callback.MIPSOL_OBJ else 0.0
    model.optimize = lambda callback: callback(model, GRB.Callback.MIPSOL)
    result = native.gurobi_reference(
        EXAMPLE, tmp_path, max_orders=1, seconds=60, checkpoint_seconds=1e-12
    )
    assert result["checkpoint_statistics"]["attempts"] == 1
    assert result["checkpoint_statistics"]["accepted"] == 1
    assert (
        result["checkpoint_statistics"]["seconds_in_optimize"]
        <= result["timings"]["optimize"]
    )
    directory = tmp_path / "recovery" / "incumbent_000001"
    assert (directory / "solution.json").is_file() and (
        directory / "path_flows.json"
    ).is_file()


def test_rejected_callback_checkpoint_cannot_replace_valid_checkpoint(
    tmp_path, monkeypatch, fake_reference
):
    from gurobipy import GRB
    from src.solvers import formulation, algorithm

    calls, model = fake_reference

    def invalid(built, **kwargs):
        result = algorithm.baseline(built.data)
        result["_solution"]["rho"][0] = 2.0
        return result

    monkeypatch.setattr(formulation, "read_solution", invalid)
    model.getVars = lambda: []
    model.cbGetSolution = lambda variables: []
    model.cbGet = lambda code: 20000.0 if code == GRB.Callback.MIPSOL_OBJ else 0.0
    model.optimize = lambda callback: callback(model, GRB.Callback.MIPSOL)
    result = native.gurobi_reference(
        EXAMPLE, tmp_path, max_orders=1, seconds=60, checkpoint_seconds=1e-12
    )
    assert result["checkpoint_statistics"]["attempts"] == 1
    assert result["checkpoint_statistics"]["accepted"] == 0
    assert not (tmp_path / "recovery" / "incumbent_000001").exists()
    checkpoint = json.loads((tmp_path / "checkpoint.json").read_text())
    assert checkpoint["validated"] and checkpoint["fallback"]
