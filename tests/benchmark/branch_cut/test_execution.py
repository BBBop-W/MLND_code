from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from main import build_parser
from src.benchmark.branch_cut.artifacts import claim_attempt_directory
from src.benchmark.branch_cut.execution import (
    build_core_solver_config,
    execute_profile,
)
from src.benchmark.branch_cut.hashing import build_logical_run_id
from src.benchmark.branch_cut.formal_solution_evidence import (
    FormalQualificationTrace,
    bind_qualification_trace,
    no_incumbent_solution_structure,
)
from src.benchmark.branch_cut.profiles import build_paper_profile_suite
from src.benchmark.branch_cut.qualification import (
    QualificationEvidence,
    assess_qualification,
)
from src.benchmark.branch_cut.schema import (
    BenchmarkMode,
    PairSpec,
    SolverConfig,
    StartConfig,
    StartMode,
    TelemetryConfig,
)
from src.config import SolverConfig as CoreSolverConfig


def _pair(solver: SolverConfig | None = None) -> PairSpec:
    return PairSpec(
        manifest_id="manifest-paper-v1",
        case_id="nanshan-large-b00-s11",
        mode=BenchmarkMode.LO,
        solver_seed=731,
        time_budget_seconds=47.5,
        solver=solver or SolverConfig(),
    )


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_adapter_maps_every_common_and_profile_specific_setting() -> None:
    common = SolverConfig(
        threads=1,
        method=1,
        miqcp_method=1,
        mip_focus=3,
        heuristics=0.17,
        no_rel_heur_time=2.5,
        presolve=0,
        cuts=2,
        root_cut_passes=7,
        numeric_focus=2,
        feasibility_tol=2e-6,
        optimality_tol=3e-6,
        integer_feasibility_tol=4e-5,
        target_mip_gap=0.012,
        output_flag=0,
    )
    heuristic = StartConfig(
        start_mode=StartMode.NETWORK_AND_COURIER,
        network_start_seconds=7.0,
        network_start_method=2,
        local_branching=True,
        local_branching_seconds=2.0,
        local_branching_radius=9,
        local_branching_include_topology=True,
    )
    suite = build_paper_profile_suite(solver=common, heuristic_start=heuristic)
    pair = _pair(common)
    telemetry = TelemetryConfig(progress_interval_seconds=0.75)

    native = build_core_solver_config(pair, suite.profile("A"), telemetry)
    assert native.time_limit == pytest.approx(47.5)
    assert native.mip_gap == pytest.approx(0.012)
    assert native.threads == 1
    assert native.seed == 731
    assert native.root_relaxation_method == 1
    assert native.miqcp_method == 1
    assert native.mip_focus == 3
    assert native.heuristics == pytest.approx(0.17)
    assert native.no_rel_heur_time == pytest.approx(2.5)
    assert native.no_rel_heur_seconds == pytest.approx(2.5)
    assert native.root_cut_passes == 7
    assert native.presolve == 0
    assert native.cuts == 2
    assert native.numeric_focus == 2
    assert native.feasibility_tol == pytest.approx(2e-6)
    assert native.optimality_tol == pytest.approx(3e-6)
    assert native.integer_feasibility_tol == pytest.approx(4e-5)
    assert native.start_mode == "none"
    assert native.network_start_seconds == pytest.approx(0.0)
    assert native.network_start_method == 2
    assert not native.use_local_branching
    assert not native.local_branching_include_y
    assert not native.use_valid_inequalities
    assert not native.use_user_cuts
    assert not native.static_nearest_display_suffix
    assert native.static_chi_vub_top_k_per_corridor == 0
    assert not native.cut_chi_facility_vub
    assert not native.cut_route_swap_station_vub
    assert not native.cut_route_swap_leg_vub
    assert not native.cut_od_disaggregated_return
    assert not native.use_objective_partial_start
    assert not any(
        (
            native.static_source_level,
            native.static_virtual_level_lower,
            native.static_lo_choice_corridor,
            native.static_lo_choice_capacity,
            native.static_queue_perspective,
            native.cut_chi_vub,
            native.cut_queue_perspective,
            native.cut_lo_choice_corridor,
            native.cut_lo_choice_capacity,
            native.cut_directed_return,
        )
    )
    assert native.collect_progress
    assert native.progress_interval_seconds == pytest.approx(0.75)

    full = build_core_solver_config(pair, suite.profile("E"), telemetry)
    assert full.start_mode == "network_and_courier"
    assert full.network_start_seconds == pytest.approx(7.0)
    assert full.network_start_method == 2
    assert full.use_local_branching
    assert full.local_branching_radius == 9
    assert full.local_branching_include_y
    assert full.use_valid_inequalities
    assert full.use_user_cuts
    # This development family is intentionally outside formal B/C/E.
    assert not full.static_nearest_display_suffix
    assert full.static_chi_vub_top_k_per_corridor == 0
    assert not full.cut_chi_facility_vub
    assert not full.cut_route_swap_station_vub
    assert not full.cut_route_swap_leg_vub
    assert not full.cut_od_disaggregated_return
    assert not full.use_objective_partial_start
    assert all(
        (
            full.static_source_level,
            full.static_virtual_level_lower,
            full.static_lo_choice_corridor,
            full.static_lo_choice_capacity,
            full.static_queue_perspective,
            full.cut_chi_vub,
            full.cut_queue_perspective,
            full.cut_lo_choice_corridor,
            full.cut_lo_choice_capacity,
        )
    )
    assert not full.cut_directed_return


def test_core_defaults_preserve_f884_and_keep_dev_families_off() -> None:
    config = CoreSolverConfig()

    assert config.no_rel_heur_seconds == pytest.approx(10.0)
    assert config.root_cut_passes == 2
    assert config.start_mode == "auto"
    assert not config.use_objective_partial_start
    assert not config.static_nearest_display_suffix
    assert config.static_chi_vub_top_k_per_corridor == 0
    assert not config.cut_chi_facility_vub
    assert not config.cut_route_swap_station_vub
    assert not config.cut_route_swap_leg_vub
    assert not config.cut_od_disaggregated_return
    config.validate()

    with pytest.raises(ValueError, match="cut_od_disaggregated_return"):
        CoreSolverConfig(cut_od_disaggregated_return=1).validate()  # type: ignore[arg-type]


def test_objective_partial_start_rejects_every_legacy_start_budget() -> None:
    compatible = {
        "use_objective_partial_start": True,
        "start_mode": "none",
        "network_start_seconds": 0.0,
        "fixed_integer_repair_seconds": 0.0,
        "use_local_branching": False,
        "local_branching_seconds": 0.0,
    }
    CoreSolverConfig(**compatible).validate()

    conflicts = (
        {"start_mode": "auto"},
        {"start_mode": "courier"},
        {"network_start_seconds": 0.1},
        {"fixed_integer_repair_seconds": 0.1},
        {"use_local_branching": True},
        {"local_branching_seconds": 0.1},
    )
    for conflict in conflicts:
        with pytest.raises(ValueError, match="objective partial starts require"):
            CoreSolverConfig(**(compatible | conflict)).validate()


def test_solve_cli_exposes_bnc_controls_without_changing_f884_defaults() -> None:
    parser = build_parser()
    default = parser.parse_args(["solve", "--instance-dir", "/tmp/example"])

    assert default.no_rel_heur_seconds == pytest.approx(10.0)
    assert default.root_cut_passes == 2
    assert default.start_mode == "auto"
    assert default.dev_objective_partial_start is False
    assert default.nearest_display_suffix_rows is False
    assert default.route_swap_station_vub_cuts is False
    assert default.route_swap_leg_vub_cuts is False
    assert default.od_disaggregated_return_cuts is False

    tuned = parser.parse_args(
        [
            "solve",
            "--instance-dir",
            "/tmp/example",
            "--start-mode",
            "none",
            "--network-start-seconds",
            "0",
            "--fixed-integer-repair-seconds",
            "0",
            "--no-local-branching",
            "--local-branching-seconds",
            "0",
            "--dev-objective-partial-start",
            "--objective-partial-start-seconds",
            "0.25",
            "--objective-partial-start-max-candidates",
            "3",
            "--static-od-vub-per-corridor",
            "8",
            "--nearest-display-suffix-rows",
            "--od-facility-vub-cuts",
            "--route-swap-station-vub-cuts",
            "--route-swap-leg-vub-cuts",
            "--od-disaggregated-return-cuts",
            "--miqcp-method",
            "1",
            "--mip-focus",
            "2",
            "--heuristics",
            "0.2",
            "--no-rel-heur-time",
            "0",
            "--collect-progress",
        ]
    )

    assert tuned.dev_objective_partial_start is True
    assert tuned.objective_partial_start_seconds == pytest.approx(0.25)
    assert tuned.objective_partial_start_max_candidates == 3
    assert tuned.static_od_vub_per_corridor == 8
    assert tuned.nearest_display_suffix_rows is True
    assert tuned.od_facility_vub_cuts is True
    assert tuned.route_swap_station_vub_cuts is True
    assert tuned.route_swap_leg_vub_cuts is True
    assert tuned.od_disaggregated_return_cuts is True
    assert tuned.miqcp_method == 1
    assert tuned.mip_focus == 2
    assert tuned.heuristics == pytest.approx(0.2)
    assert tuned.no_rel_heur_time == pytest.approx(0.0)
    assert tuned.collect_progress is True


class _FakeBackend:
    _READ_BEFORE_DISPOSE = {
        "Params",
        "Status",
        "SolCount",
        "ObjVal",
        "ObjBound",
        "NodeCount",
        "Runtime",
    }

    def __init__(self) -> None:
        self.Params = SimpleNamespace()
        self.Status = 9
        self.SolCount = 0
        self.ObjVal = 0.0
        self.ObjBound = 120.5
        self.NodeCount = 84.0
        self.Runtime = 4.25
        self.disposed = False

    def __getattribute__(self, name: str) -> Any:
        if name in object.__getattribute__(self, "_READ_BEFORE_DISPOSE"):
            if object.__getattribute__(self, "disposed"):
                raise RuntimeError(f"{name} read after backend disposal")
        return object.__getattribute__(self, name)

    def dispose(self) -> None:
        self.disposed = True

    def getParamInfo(self, name: str) -> tuple[object, ...]:
        defaults: dict[str, object] = {
            "Cuts": -1,
            "FeasibilityTol": 1e-6,
            "Heuristics": 0.05,
            "IntFeasTol": 1e-5,
            "LogFile": "",
            "Method": -1,
            "MIPFocus": 0,
            "MIPGap": 1e-4,
            "MIQCPMethod": -1,
            "NoRelHeurTime": 0.0,
            "NumericFocus": 0,
            "OptimalityTol": 1e-6,
            "OutputFlag": 1,
            "PreCrush": 0,
            "Presolve": -1,
            "Seed": 0,
            "Threads": 0,
            "TimeLimit": 1e100,
        }
        default = defaults.get(name, 0)
        current = getattr(self.Params, name, default)
        return name, type(default), current, None, None, default


class _FakeEnvironment:
    def __init__(self) -> None:
        self.disposed = False

    def dispose(self) -> None:
        self.disposed = True


class _FakeModel:
    def __init__(
        self,
        *,
        fail: bool = False,
        fatal_error: BaseException | None = None,
    ) -> None:
        self.model = _FakeBackend()
        self.env = _FakeEnvironment()
        self.fail = fail
        self.fatal_error = fatal_error
        self.received_solver = None
        self.main_solve_started = False
        self.optimize_called = False
        self.budget_exhausted_before_main_solve = False
        self.main_solve_runtime = 4.25
        self.total_runtime = 5.5
        self.algorithm_prepare_runtime = 1.0
        self.network_start_runtime = 0.25
        self.local_branching_runtime = 0.0
        self.callback_runtime = 0.04
        self.first_incumbent_time = None
        self.first_incumbent_objective = None
        self.root_bound_before_cuts = 118.0
        self.root_bound_after_cuts = 120.5
        self.network_start_found = False
        self.network_start_used = False
        self.local_branching_improved = False
        self.progress_events = [
            {
                "event": "mip_progress",
                "runtime_seconds": 1.0,
                "incumbent": None,
                "bound": 118.0,
                "relative_gap": None,
                "node_count": 0.0,
                "solution_count": 0,
            },
            {
                "event": "final",
                "runtime_seconds": 4.25,
                "incumbent": None,
                "bound": 120.5,
                "relative_gap": None,
                "node_count": 84.0,
                "solution_count": 0,
                "status": 9,
            },
        ]
        self.root_round_events = [
            {
                "round": 1,
                "runtime_seconds": 0.5,
                "root_bound_before_round": 118.0,
                "cuts_added": 3,
                "cuts_added_by_family": {"chi_od_corridor_vub": 3},
            }
        ]
        self.static_cut_counts = {"source_level_upper": 2}
        self.user_cut_counts = {"chi_od_corridor_vub": 3}
        self.user_cut_candidate_counts = {"chi_od_corridor_vub": 5}
        self.user_cut_duplicate_counts = {"chi_od_corridor_vub": 1}
        self.user_cut_separation_seconds = {"chi_od_corridor_vub": 0.03}
        self.route_bundle_index_counts = {
            "swap_route_options": 19,
            "swap_leg_bundles": 8,
            "swap_station_bundles": 2,
            "strengthening_swap_leg_bundles": 5,
            "strengthening_swap_station_bundles": 2,
        }
        self.root_cut_rounds = 1

    def solve(self, solver: Any) -> bool:
        self.received_solver = solver
        parameter_values = {
            "TimeLimit": solver.time_limit,
            "MIPGap": solver.mip_gap,
            "Threads": solver.threads,
            "OutputFlag": solver.output_flag,
            "Seed": solver.seed,
            "Method": solver.root_relaxation_method,
            "MIQCPMethod": solver.miqcp_method,
            "MIPFocus": solver.mip_focus,
            "Heuristics": solver.heuristics,
            "NoRelHeurTime": solver.no_rel_heur_time,
            "Presolve": solver.presolve,
            "Cuts": solver.cuts,
            "NumericFocus": solver.numeric_focus,
            "FeasibilityTol": solver.feasibility_tol,
            "OptimalityTol": solver.optimality_tol,
            "IntFeasTol": solver.integer_feasibility_tol,
            "PreCrush": int(solver.use_user_cuts),
        }
        for name, value in parameter_values.items():
            setattr(self.model.Params, name, value)
        log_path = getattr(self.model.Params, "LogFile", None)
        if log_path:
            Path(log_path).write_text("fake solver log\n", encoding="utf-8")
        self.main_solve_started = True
        self.optimize_called = True
        if self.fatal_error is not None:
            raise self.fatal_error
        if self.fail:
            raise RuntimeError("synthetic optimizer failure")
        return False


def _claim(tmp_path: Path, pair: PairSpec, profile: Any):
    logical_run_id = build_logical_run_id(pair, profile)
    return claim_attempt_directory(tmp_path, logical_run_id)


def test_mlnd_model_passes_explicit_environment_to_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.model.gurobi_model as gurobi_model_module

    environment = object()
    backend = object()
    calls: list[tuple[str, object]] = []

    def fake_backend_factory(name: str, *, env: object) -> object:
        calls.append((name, env))
        return backend

    monkeypatch.setattr(gurobi_model_module.gp, "Model", fake_backend_factory)
    monkeypatch.setattr(
        gurobi_model_module.MLNDModel,
        "_prepare_indices",
        lambda _self: None,
    )

    wrapper = gurobi_model_module.MLNDModel(
        object(),
        "MX",
        environment=environment,
    )

    assert wrapper.model is backend
    assert calls == [("MLND_MX_CHI_EPSILON_ENERGY", environment)]


def test_private_environment_creation_is_serialized_and_distinct(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import gurobipy as gp

    import src.benchmark.branch_cut.execution as execution_module

    state_lock = threading.Lock()
    start_barrier = threading.Barrier(2)
    active = 0
    maximum_active = 0
    created: list[object] = []
    started: list[object] = []
    empty_arguments: list[bool] = []

    class Environment:
        def start(self) -> None:
            started.append(self)

        def dispose(self) -> None:
            raise AssertionError("successful creation must not dispose the environment")

    def fake_environment_factory(*, empty: bool) -> object:
        nonlocal active, maximum_active
        environment = Environment()
        empty_arguments.append(empty)
        with state_lock:
            active += 1
            maximum_active = max(maximum_active, active)
        time.sleep(0.02)
        with state_lock:
            active -= 1
            created.append(environment)
        return environment

    def create_after_barrier() -> object:
        start_barrier.wait(timeout=2.0)
        return execution_module._create_private_gurobi_environment()

    monkeypatch.setattr(gp, "Env", fake_environment_factory)
    with ThreadPoolExecutor(max_workers=2) as executor:
        environments = tuple(
            executor.map(lambda _index: create_after_barrier(), range(2))
        )

    assert maximum_active == 1
    assert len({id(environment) for environment in environments}) == 2
    assert empty_arguments == [True, True]
    assert {id(environment) for environment in started} == {
        id(environment) for environment in environments
    }
    assert {id(environment) for environment in environments} == {
        id(environment) for environment in created
    }


def test_private_environment_failed_start_is_disposed_without_default_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import gurobipy as gp

    import src.benchmark.branch_cut.execution as execution_module

    events: list[object] = []

    class Environment:
        def start(self) -> None:
            events.append("start")
            raise RuntimeError("synthetic private environment start failure")

        def dispose(self) -> None:
            events.append("dispose")

    def environment_factory(*, empty: bool) -> Environment:
        events.append(("create", empty))
        return Environment()

    monkeypatch.setattr(gp, "Env", environment_factory)
    with pytest.raises(RuntimeError, match="start failure"):
        execution_module._create_private_gurobi_environment()

    assert events == [("create", True), "start", "dispose"]


def test_private_environment_disposal_is_serialized_after_each_model() -> None:
    import src.benchmark.branch_cut.execution as execution_module

    state_lock = threading.Lock()
    start_barrier = threading.Barrier(2)
    active_environment_disposals = 0
    maximum_active_environment_disposals = 0
    events: list[tuple[str, int]] = []

    class Backend:
        def __init__(self, identifier: int) -> None:
            self.identifier = identifier

        def dispose(self) -> None:
            with state_lock:
                events.append(("model", self.identifier))

    class Environment:
        def __init__(self, identifier: int) -> None:
            self.identifier = identifier

        def dispose(self) -> None:
            nonlocal active_environment_disposals
            nonlocal maximum_active_environment_disposals
            with state_lock:
                active_environment_disposals += 1
                maximum_active_environment_disposals = max(
                    maximum_active_environment_disposals,
                    active_environment_disposals,
                )
            time.sleep(0.02)
            with state_lock:
                events.append(("environment", self.identifier))
                active_environment_disposals -= 1

    wrappers = tuple(
        SimpleNamespace(
            model=Backend(identifier),
            _owned_environment=Environment(identifier),
        )
        for identifier in range(2)
    )

    def dispose_after_barrier(wrapper: object) -> None:
        start_barrier.wait(timeout=2.0)
        execution_module._dispose_model_resources(wrapper)

    with ThreadPoolExecutor(max_workers=2) as executor:
        tuple(executor.map(dispose_after_barrier, wrappers))

    assert maximum_active_environment_disposals == 1
    for identifier in range(2):
        assert events.count(("model", identifier)) == 1
        assert events.count(("environment", identifier)) == 1
        assert events.index(("model", identifier)) < events.index(
            ("environment", identifier)
        )


def test_default_factory_owns_one_distinct_environment_per_concurrent_attempt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.benchmark.branch_cut.execution as execution_module
    import src.model.gurobi_model as gurobi_model_module

    creation_lock = threading.Lock()
    created_environments: list[_FakeEnvironment] = []
    received_environments: list[object] = []

    def create_environment() -> _FakeEnvironment:
        environment = _FakeEnvironment()
        with creation_lock:
            created_environments.append(environment)
        return environment

    class FakeMLNDModel:
        def __init__(
            self,
            _instance: object,
            _mode: str,
            *,
            environment: object,
        ) -> None:
            assert self._owned_environment is environment
            self.model = _FakeBackend()
            with creation_lock:
                received_environments.append(environment)

    monkeypatch.setattr(gurobi_model_module, "MLNDModel", FakeMLNDModel)
    monkeypatch.setattr(
        execution_module,
        "_create_private_gurobi_environment",
        create_environment,
    )

    with ThreadPoolExecutor(max_workers=2) as executor:
        wrappers = tuple(
            executor.map(
                lambda identifier: execution_module._default_model_factory(
                    identifier,
                    "MX",
                ),
                range(2),
            )
        )

    assert len({id(item) for item in created_environments}) == 2
    assert {id(item) for item in received_environments} == {
        id(item) for item in created_environments
    }
    assert {id(wrapper._owned_environment) for wrapper in wrappers} == {
        id(item) for item in created_environments
    }
    for wrapper in wrappers:
        execution_module._dispose_model_resources(wrapper)
        assert wrapper.model.disposed
        assert wrapper._owned_environment.disposed


def test_default_factory_disposes_partially_initialized_native_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.benchmark.branch_cut.execution as execution_module
    import src.model.gurobi_model as gurobi_model_module

    events: list[str] = []

    class RecordingBackend(_FakeBackend):
        def dispose(self) -> None:
            events.append("model")
            super().dispose()

    class RecordingEnvironment(_FakeEnvironment):
        def dispose(self) -> None:
            events.append("environment")
            super().dispose()

    environment = RecordingEnvironment()
    created: list[RecordingBackend] = []

    class FailingMLNDModel:
        def __init__(
            self,
            _instance: object,
            _mode: str,
            *,
            environment: object,
        ) -> None:
            assert self._owned_environment is environment
            self.model = RecordingBackend()
            created.append(self.model)
            raise RuntimeError("synthetic model construction failure")

    monkeypatch.setattr(gurobi_model_module, "MLNDModel", FailingMLNDModel)
    monkeypatch.setattr(
        execution_module,
        "_create_private_gurobi_environment",
        lambda: environment,
    )

    with pytest.raises(RuntimeError, match="synthetic model construction failure"):
        execution_module._default_model_factory(object(), "MX")

    assert created[0].disposed
    assert environment.disposed
    assert events == ["model", "environment"]


def test_execute_profile_preserves_not_created_metrics_when_factory_fails(
    tmp_path: Path,
) -> None:
    suite = build_paper_profile_suite()
    pair = _pair(suite.solver)
    profile = suite.profile("A")
    claim = _claim(tmp_path, pair, profile)

    def factory(_instance: object, _mode: str) -> _FakeModel:
        raise RuntimeError("synthetic model construction failure")

    result = execute_profile(
        object(),
        pair,
        profile,
        suite.telemetry,
        claim,
        model_factory=factory,
        gurobi_version_provider=lambda: "13.0.1",
    )

    assert not result.completed_without_exception
    assert result.outcome == "failed"
    assert result.failure is not None
    assert result.failure["stage"] == "solve"
    metrics = result.run_record["solver_metrics"]
    assert metrics["status"] is None
    assert metrics["status_name"] == "NOT_CREATED"
    assert metrics["has_incumbent"] is False
    assert metrics["budget_exhausted_before_main_solve"] is None
    assert metrics["main_solve_started"] is None
    assert metrics["optimize_called"] is None
    assert result.run_record["budget_exhausted_before_main_solve"] is None
    assert result.run_record["main_solve_started"] is None
    assert result.run_record["optimize_called"] is None

    stored = _json(claim.directory / "run_record.json")
    assert stored["solver_metrics"]["status_name"] == "NOT_CREATED"
    assert stored["solver_metrics"]["optimize_called"] is None


def test_execute_profile_preserves_no_incumbent_outcome_and_artifacts(
    tmp_path: Path,
) -> None:
    suite = build_paper_profile_suite()
    pair = _pair(suite.solver)
    profile = suite.profile("C")
    claim = _claim(tmp_path, pair, profile)
    created: list[_FakeModel] = []

    def factory(instance: Any, mode: str) -> _FakeModel:
        assert instance == "instance-object"
        assert mode == "LO"
        model = _FakeModel()
        created.append(model)
        return model

    formal_context = {
        "campaign_id": "formal-campaign-" + "1" * 64,
        "formal_plan_id": "formal-plan-" + "2" * 64,
        "block_id": "block-" + "3" * 64,
        "slot_id": "slot-" + "4" * 64,
    }

    result = execute_profile(
        "instance-object",
        pair,
        profile,
        suite.telemetry,
        claim,
        model_factory=factory,
        gurobi_version_provider=lambda: (13, 0, 1),
        formal_context=formal_context,
    )

    assert result.completed_without_exception
    assert not result.has_incumbent
    assert result.outcome == "completed_no_incumbent"
    assert created[0].received_solver.collect_progress
    assert created[0].model.disposed
    assert created[0].env.disposed
    expected_files = {
        "claim.json",
        "solver.log",
        "run_spec.json",
        "run_record.json",
        "progress.jsonl",
        "root_rounds.jsonl",
        "cut_stats.json",
    }
    assert {path.name for path in claim.directory.iterdir()} == expected_files
    assert claim.directory.joinpath("solver.log").read_text() == "fake solver log\n"

    spec = _json(claim.directory / "run_spec.json")
    record = _json(claim.directory / "run_record.json")
    cuts = _json(claim.directory / "cut_stats.json")
    assert spec["gurobi_version"] == "13.0.1"
    assert spec["formal_context"] == formal_context
    assert spec["core_solver_config"]["threads"] == 1
    assert spec["core_solver_config"]["collect_progress"] is True
    assert record["solver_metrics"]["status_name"] == "TIME_LIMIT"
    assert record["formal_context"] == formal_context
    assert record["solver_metrics"]["objective"] is None
    assert record["solver_metrics"]["best_bound"] == pytest.approx(120.5)
    assert record["solver_metrics"]["node_count"] == pytest.approx(84.0)
    assert record["solver_metrics"]["total_algorithm_runtime_seconds"] == pytest.approx(
        5.5
    )
    assert record["applied_gurobi_parameters"]["threads"] == 1
    assert record["applied_gurobi_parameters"]["miqcp_method"] == -1
    assert record["gurobi_parameter_audit"]["enumeration_complete"] is True
    assert record["gurobi_parameter_audit"]["enumerated_parameter_count"] > 100
    assert record["gurobi_execution_context"]["execution_mode"] == "local"


    assert record["unallowed_nondefault_parameters"] == []
    assert record["effective_nondefault_parameters"]["Threads"]["current"] == 1
    assert record["effective_nondefault_parameters"]["LogFile"]["current"][
        "redacted"
    ] is True
    assert str(claim.directory) not in json.dumps(record, sort_keys=True)
    assert len(_jsonl(claim.directory / "progress.jsonl")) == 2
    assert len(_jsonl(claim.directory / "root_rounds.jsonl")) == 1
    assert cuts["user_cuts_added_total"] == 3
    assert cuts["route_bundle_index_counts"] == created[0].route_bundle_index_counts

    original_spec = (claim.directory / "run_spec.json").read_bytes()
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        execute_profile(
            "instance-object",
            pair,
            profile,
            suite.telemetry,
            claim,
            model_factory=factory,
            gurobi_version_provider=lambda: "13.0.1",
        )
    assert (claim.directory / "run_spec.json").read_bytes() == original_spec


def test_execute_profile_writes_formal_no_incumbent_solution_structure(
    tmp_path: Path,
) -> None:
    suite = build_paper_profile_suite()
    pair = _pair(suite.solver)
    profile = suite.profile("A")
    claim = _claim(tmp_path, pair, profile)
    decision = assess_qualification(
        QualificationEvidence(
            case_id=pair.case_id,
            scale="small",
            courier_only_objective=1_000.0,
            witness_objective=900.0,
            total_demand_per_hour=100.0,
            witness_service_per_hour=20.0,
            open_facilities=3,
            positive_cost_open_facilities=3,
            candidate_h=4,
            open_h=2,
            witness_feasible=True,
            qualification_run_id="qualification-test",
            instance_sha256="1" * 64,
            active_h=2,
            active_h_threshold_per_hour=0.001,
        )
    )
    trace = FormalQualificationTrace.from_decision(
        decision,
        manifest_id=pair.manifest_id,
        benchmark_manifest_sha256="2" * 64,
        manifest_artifact_sha256="3" * 64,
    )
    formal_context = bind_qualification_trace(
        {"slot_id": "slot-" + "4" * 64},
        trace,
    )

    result = execute_profile(
        "instance-object",
        pair,
        profile,
        suite.telemetry,
        claim,
        model_factory=lambda _instance, _mode: _FakeModel(),
        gurobi_version_provider=lambda: "13.0.1",
        formal_context=formal_context,
    )

    expected = no_incumbent_solution_structure(trace).to_payload()
    assert result.run_record["solution_structure"] == expected
    assert result.run_record["formal_context"] == formal_context


def test_execute_profile_discards_stale_auxiliary_solution_when_main_is_skipped(
    tmp_path: Path,
) -> None:
    suite = build_paper_profile_suite()
    pair = _pair(suite.solver)
    profile = suite.profile("D")
    claim = _claim(tmp_path, pair, profile)

    class StaleAuxiliaryModel(_FakeModel):
        def __init__(self) -> None:
            super().__init__()
            self.model.Status = 2
            self.model.SolCount = 1
            self.model.ObjVal = 80.0
            self.model.ObjBound = 79.0
            self.model.NodeCount = 999.0
            self.model.Runtime = 7.5
            self.main_solve_runtime = 7.5
            self.total_runtime = 10.0
            self.algorithm_prepare_runtime = 1.0
            self.start_preparation_runtime = 9.0
            self.network_start_runtime = 7.5
            self.budget_exhausted_before_main_solve = True

        def solve(self, solver: Any) -> bool:
            self.received_solver = solver
            self.model.Params.TimeLimit = 0.0
            return False

    created: list[StaleAuxiliaryModel] = []

    def factory(_instance: object, _mode: str) -> StaleAuxiliaryModel:
        model = StaleAuxiliaryModel()
        created.append(model)
        return model

    result = execute_profile(
        object(),
        pair,
        profile,
        suite.telemetry,
        claim,
        model_factory=factory,
        gurobi_version_provider=lambda: "13.0.1",
    )

    assert result.completed_without_exception
    assert not result.has_incumbent
    assert result.outcome == "completed_no_incumbent"
    metrics = result.run_record["solver_metrics"]
    assert metrics["status"] == 1
    assert metrics["status_name"] == "LOADED"
    assert metrics["solution_count"] == 0
    assert metrics["has_incumbent"] is False
    for name in ("objective", "best_bound", "relative_gap", "node_count"):
        assert metrics[name] is None
    assert metrics["main_solve_runtime_seconds"] == pytest.approx(0.0)
    assert metrics["algorithm_prepare_runtime_seconds"] == pytest.approx(1.0)
    assert metrics["start_preparation_runtime_seconds"] == pytest.approx(9.0)
    assert metrics["network_start_runtime_seconds"] == pytest.approx(7.5)
    assert metrics["total_algorithm_runtime_seconds"] == pytest.approx(10.0)
    assert metrics["budget_exhausted_before_main_solve"] is True
    assert metrics["main_solve_started"] is False
    assert metrics["optimize_called"] is False
    assert result.run_record["budget_exhausted_before_main_solve"] is True
    assert result.run_record["main_solve_started"] is False
    assert result.run_record["optimize_called"] is False
    assert created[0].model.disposed
    assert created[0].env.disposed

    stored = _json(claim.directory / "run_record.json")
    assert stored["outcome"] == "completed_no_incumbent"
    assert stored["solver_metrics"] == metrics


def test_execute_profile_records_exception_and_partial_solver_state(
    tmp_path: Path,
) -> None:
    suite = build_paper_profile_suite()
    pair = _pair(suite.solver)
    profile = suite.profile("E")
    claim = _claim(tmp_path, pair, profile)

    created: list[_FakeModel] = []

    def factory(_instance: object, _mode: str) -> _FakeModel:
        model = _FakeModel(fail=True)
        created.append(model)
        return model

    formal_context = {
        "campaign_id": "formal-campaign-" + "5" * 64,
        "formal_plan_id": "formal-plan-" + "6" * 64,
        "block_id": "block-" + "7" * 64,
        "slot_id": "slot-" + "8" * 64,
    }

    result = execute_profile(
        object(),
        pair,
        profile,
        suite.telemetry,
        claim,
        model_factory=factory,
        gurobi_version_provider=lambda: "13.0.1",
        formal_context=formal_context,
    )

    assert not result.completed_without_exception
    assert not result.has_incumbent
    assert result.outcome == "failed"
    assert result.failure is not None
    assert result.failure["message"] == "synthetic optimizer failure"
    failure = _json(claim.directory / "failure.json")
    record = _json(claim.directory / "run_record.json")
    assert failure["exception_type"].endswith("RuntimeError")
    assert "synthetic optimizer failure" in failure["traceback"]
    assert failure["solver_metrics_at_failure"]["status"] == 9
    assert failure["solver_metrics_at_failure"]["best_bound"] == pytest.approx(120.5)
    assert failure["formal_context"] == formal_context
    assert record["outcome"] == "failed"
    assert record["formal_context"] == formal_context
    assert record["solver_metrics"]["node_count"] == pytest.approx(84.0)
    assert created[0].model.disposed
    assert created[0].env.disposed
    assert (claim.directory / "progress.jsonl").is_file()
    assert (claim.directory / "root_rounds.jsonl").is_file()
    assert (claim.directory / "cut_stats.json").is_file()


@pytest.mark.parametrize(
    "fatal_error",
    [
        MemoryError("synthetic memory exhaustion"),
        KeyboardInterrupt("synthetic interrupt"),
        SystemExit("synthetic exit"),
    ],
)
def test_execute_profile_rethrows_fatal_exceptions_after_disposal(
    tmp_path: Path,
    fatal_error: BaseException,
) -> None:
    suite = build_paper_profile_suite()
    pair = _pair(suite.solver)
    profile = suite.profile("A")
    claim = _claim(tmp_path, pair, profile)
    created: list[_FakeModel] = []

    def factory(_instance: object, _mode: str) -> _FakeModel:
        model = _FakeModel(fatal_error=fatal_error)
        created.append(model)
        return model

    with pytest.raises(type(fatal_error), match="synthetic"):
        execute_profile(
            object(),
            pair,
            profile,
            suite.telemetry,
            claim,
            model_factory=factory,
            gurobi_version_provider=lambda: "13.0.1",
        )

    assert created[0].model.disposed
    assert created[0].env.disposed
    assert not (claim.directory / "failure.json").exists()
    assert not (claim.directory / "run_record.json").exists()


def test_telemetry_error_is_secondary_to_original_solver_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.benchmark.branch_cut.execution as execution_module

    suite = build_paper_profile_suite()
    pair = _pair(suite.solver)
    profile = suite.profile("E")
    claim = _claim(tmp_path, pair, profile)
    created: list[_FakeModel] = []

    def factory(_instance: object, _mode: str) -> _FakeModel:
        model = _FakeModel(fail=True)
        created.append(model)
        return model

    def fail_telemetry(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("synthetic telemetry failure")

    monkeypatch.setattr(
        execution_module,
        "write_telemetry_artifacts",
        fail_telemetry,
    )
    result = execute_profile(
        object(),
        pair,
        profile,
        suite.telemetry,
        claim,
        model_factory=factory,
        gurobi_version_provider=lambda: "13.0.1",
    )

    assert result.failure is not None
    assert result.failure["stage"] == "solve"
    assert result.failure["message"] == "synthetic optimizer failure"
    assert result.failure["secondary_errors"][0]["stage"] == "telemetry_artifacts"
    assert (
        result.failure["secondary_errors"][0]["message"]
        == "synthetic telemetry failure"
    )
    assert created[0].model.disposed
    assert created[0].env.disposed
