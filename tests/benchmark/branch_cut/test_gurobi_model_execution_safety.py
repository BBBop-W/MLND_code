"""Pure unit tests for formal-solve timing and MIP-start lifecycle safety."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from gurobipy import GRB

import src.model.gurobi_model as gurobi_model_module
from src.benchmark.branch_cut.telemetry import collect_solver_metrics
from src.config import SolverConfig
from src.model.gurobi_model import MLNDModel


class _Clock:
    def __init__(self, start: float = 100.0) -> None:
        self.now = start

    def perf_counter(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class _FakeVariable:
    def __init__(self) -> None:
        self.Start = 0.0


class _FakeBackend:
    _POST_OPTIMIZE_ATTRIBUTES = {
        "Status",
        "SolCount",
        "ObjVal",
        "ObjBound",
        "NodeCount",
        "Runtime",
    }

    def __init__(
        self,
        clock: _Clock,
        *,
        main_runtime: float = 0.0,
        optimize_error: Exception | None = None,
    ) -> None:
        self.clock = clock
        self.main_runtime = main_runtime
        self.optimize_error = optimize_error
        self.Params = SimpleNamespace()
        self.NumStart = 0
        self.Runtime = 0.0
        self.SolCount = 0
        self.ObjVal = 0.0
        self.ObjBound = 0.0
        self.NodeCount = 0.0
        self.Status = int(GRB.LOADED)
        self.optimize_calls = 0
        self.get_vars_calls = 0
        self.update_calls = 0
        self.variables: list[_FakeVariable] = []
        self.raise_on_get_vars = False
        self.guard_unoptimized_reads = False

    def __getattribute__(self, name: str) -> object:
        if name in object.__getattribute__(self, "_POST_OPTIMIZE_ATTRIBUTES"):
            guarded = object.__getattribute__(self, "guard_unoptimized_reads")
            optimize_calls = object.__getattribute__(self, "optimize_calls")
            if guarded and optimize_calls == 0:
                raise AssertionError(f"{name} is unavailable before optimize")
        return object.__getattribute__(self, name)

    def update(self) -> None:
        self.update_calls += 1

    def getVars(self) -> list[_FakeVariable]:
        self.get_vars_calls += 1
        if self.raise_on_get_vars:
            raise AssertionError("fresh no-start model must not enumerate variables")
        return self.variables

    def optimize(self, *_args: object) -> None:
        self.optimize_calls += 1
        self.clock.advance(self.main_runtime)
        self.Runtime = self.main_runtime
        if self.optimize_error is not None:
            raise self.optimize_error


class _TimedMLNDModel(MLNDModel):
    def __init__(
        self,
        clock: _Clock,
        *,
        prepare_runtime: float,
        main_runtime: float = 0.0,
        prepare_error: Exception | None = None,
        optimize_error: Exception | None = None,
    ) -> None:
        # Deliberately avoid MLNDModel.__init__: this is a pure control-flow
        # test and must not allocate a real Gurobi model or environment.
        self.clock = clock
        self.prepare_runtime_for_test = prepare_runtime
        self.prepare_error_for_test = prepare_error
        self.model = _FakeBackend(
            clock,
            main_runtime=main_runtime,
            optimize_error=optimize_error,
        )
        self.chi_keys: tuple[object, ...] = ()
        self._user_cut_signatures: set[tuple[object, ...]] = set()

    def build(self) -> _TimedMLNDModel:
        return self

    def prepare_algorithm(
        self,
        _solver: SolverConfig | None = None,
    ) -> _TimedMLNDModel:
        self.clock.advance(self.prepare_runtime_for_test)
        if self.prepare_error_for_test is not None:
            raise self.prepare_error_for_test
        return self


def _formal_no_start_config(
    *,
    time_limit: float = 10.0,
    collect_progress: bool = False,
) -> SolverConfig:
    return SolverConfig(
        time_limit=time_limit,
        threads=1,
        output_flag=0,
        network_start_seconds=0.0,
        use_valid_inequalities=False,
        use_user_cuts=False,
        use_local_branching=False,
        local_branching_seconds=0.0,
        fixed_integer_repair_seconds=0.0,
        start_mode="none",
        mip_focus=0,
        heuristics=0.05,
        no_rel_heur_time=0.0,
        collect_progress=collect_progress,
    )


def test_clear_mip_starts_is_constant_time_when_pool_is_empty() -> None:
    clock = _Clock()
    wrapper = _TimedMLNDModel(clock, prepare_runtime=0.0)
    backend = wrapper.model
    backend.NumStart = 0
    backend.raise_on_get_vars = True

    wrapper._clear_mip_starts()

    assert backend.get_vars_calls == 0
    assert backend.update_calls == 0
    assert backend.NumStart == 0


def test_clear_mip_starts_enumerates_variables_only_for_existing_pool() -> None:
    clock = _Clock()
    wrapper = _TimedMLNDModel(clock, prepare_runtime=0.0)
    backend = wrapper.model
    backend.NumStart = 1
    backend.variables = [_FakeVariable(), _FakeVariable()]

    wrapper._clear_mip_starts()

    assert backend.get_vars_calls == 1
    assert backend.update_calls == 3
    assert backend.NumStart == 0
    assert [variable.Start for variable in backend.variables] == [
        GRB.UNDEFINED,
        GRB.UNDEFINED,
    ]


def test_absolute_deadline_skips_main_solve_when_prepare_uses_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = _Clock()
    monkeypatch.setattr(
        gurobi_model_module,
        "time",
        SimpleNamespace(perf_counter=clock.perf_counter),
    )
    wrapper = _TimedMLNDModel(clock, prepare_runtime=10.0)
    wrapper.model.guard_unoptimized_reads = True

    solved = wrapper.solve(
        _formal_no_start_config(time_limit=10.0, collect_progress=True)
    )

    assert not solved
    assert wrapper.model.optimize_calls == 0
    assert wrapper.model.Params.TimeLimit == pytest.approx(0.0)
    assert wrapper.budget_exhausted_before_main_solve
    assert wrapper.start_preparation_runtime == pytest.approx(0.0)
    assert wrapper.main_solve_runtime == pytest.approx(0.0)
    assert wrapper.total_runtime == pytest.approx(10.0)
    assert wrapper.main_solve_started is False
    assert wrapper.optimize_called is False
    assert wrapper.progress_events[-1] == {
        "event": "final",
        "runtime_seconds": 0.0,
        "total_algorithm_runtime_seconds": 10.0,
        "budget_exhausted_before_main_solve": True,
        "main_solve_started": False,
        "optimize_called": False,
        "incumbent": None,
        "bound": None,
        "relative_gap": None,
        "node_count": 0.0,
        "solution_count": 0,
        "status": int(GRB.LOADED),
    }

    metrics = collect_solver_metrics(wrapper)
    assert metrics["has_incumbent"] is False
    assert metrics["objective"] is None
    assert metrics["best_bound"] is None
    assert metrics["node_count"] is None
    assert metrics["main_solve_runtime_seconds"] == pytest.approx(0.0)
    assert metrics["total_algorithm_runtime_seconds"] == pytest.approx(10.0)


def test_main_solve_receives_only_absolute_deadline_remainder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = _Clock()
    monkeypatch.setattr(
        gurobi_model_module,
        "time",
        SimpleNamespace(perf_counter=clock.perf_counter),
    )
    wrapper = _TimedMLNDModel(
        clock,
        prepare_runtime=2.0,
        main_runtime=8.0,
    )

    wrapper.solve(_formal_no_start_config(time_limit=10.0))

    assert wrapper.model.optimize_calls == 1
    assert wrapper.model.Params.TimeLimit == pytest.approx(8.0)
    assert not wrapper.budget_exhausted_before_main_solve
    assert wrapper.main_solve_started
    assert wrapper.optimize_called
    assert wrapper.start_preparation_runtime == pytest.approx(0.0)
    assert wrapper.main_solve_runtime == pytest.approx(8.0)
    assert wrapper.total_runtime == pytest.approx(10.0)


def test_total_runtime_is_preserved_when_main_solve_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = _Clock()
    monkeypatch.setattr(
        gurobi_model_module,
        "time",
        SimpleNamespace(perf_counter=clock.perf_counter),
    )
    wrapper = _TimedMLNDModel(
        clock,
        prepare_runtime=2.0,
        main_runtime=3.0,
        optimize_error=RuntimeError("synthetic main solve failure"),
    )

    with pytest.raises(RuntimeError, match="synthetic main solve failure"):
        wrapper.solve(_formal_no_start_config(time_limit=10.0))

    assert wrapper.model.optimize_calls == 1
    assert wrapper.main_solve_runtime == pytest.approx(3.0)
    assert wrapper.total_runtime == pytest.approx(5.0)


def test_total_runtime_is_preserved_when_prepare_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = _Clock()
    monkeypatch.setattr(
        gurobi_model_module,
        "time",
        SimpleNamespace(perf_counter=clock.perf_counter),
    )
    wrapper = _TimedMLNDModel(
        clock,
        prepare_runtime=3.0,
        prepare_error=RuntimeError("synthetic prepare failure"),
    )

    with pytest.raises(RuntimeError, match="synthetic prepare failure"):
        wrapper.solve(_formal_no_start_config(time_limit=10.0))

    assert wrapper.model.optimize_calls == 0
    assert wrapper.total_runtime == pytest.approx(3.0)


def test_lo_network_start_retains_f884_five_candidate_attempt_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = _Clock()
    monkeypatch.setattr(
        gurobi_model_module,
        "time",
        SimpleNamespace(perf_counter=clock.perf_counter),
    )
    wrapper = MLNDModel.__new__(MLNDModel)
    wrapper.mode = "LO"
    wrapper._choice_cover_data = tuple(range(7))
    wrapper.chi_by_destination_locker = {
        (f"d{index}", f"h{index}"): () for index in range(7)
    }
    wrapper.network_start_candidate_selection_seconds = 0.0
    wrapper.network_start_candidate_count = 0
    wrapper.network_start_capacity_rejected_count = 0
    wrapper.network_start_attempted_keys = ()
    wrapper.network_start_key = None
    wrapper.network_start_found = False
    wrapper.network_start_used = False
    wrapper.network_start_runtime = 0.0
    wrapper.network_construction_runtime = 0.0
    attempted: list[tuple[str, str]] = []
    candidates = tuple(
        gurobi_model_module._LOStartCandidate(
            alpha_key=(f"d{index}", f"h{index}"),
            offered_demand=1.0,
            effective_min_arrival=0.5,
            optimistic_saving=float(7 - index),
        )
        for index in range(7)
    )
    wrapper._rank_lo_network_start_candidates = lambda _solver: (candidates, 0)

    def reject_candidate(**kwargs: object) -> None:
        attempted.append(kwargs["alpha_target"])  # type: ignore[arg-type]
        return None

    wrapper._solve_network_start_attempt = reject_candidate
    fallback_calls: list[str] = []
    wrapper._clear_mip_starts = lambda: fallback_calls.append("clear")
    wrapper._set_courier_only_start = lambda: fallback_calls.append("courier")

    wrapper._construct_network_start_with_budgets(
        1.0,
        0.0,
        0.0,
        2.0,
        SolverConfig(output_flag=0),
        include_courier_start=False,
    )

    assert attempted == [candidate.alpha_key for candidate in candidates[:5]]
    assert wrapper.network_start_attempted_keys == tuple(attempted)
    assert fallback_calls == ["clear"]
