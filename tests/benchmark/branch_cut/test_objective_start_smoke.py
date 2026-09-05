from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from gurobipy import GRB
import pytest

from src.benchmark.branch_cut import dev_micro_ce as dev
from src.benchmark.branch_cut import objective_start_smoke as module


class RecordingLock:
    calls: list[tuple[Path, str]] = []

    def __init__(self, path: Path, command: str) -> None:
        self.path = Path(path)
        self.command = command

    def __enter__(self) -> RecordingLock:
        self.calls.append((self.path, self.command))
        return self

    def __exit__(self, *_: object) -> None:
        return None


def _write_canonical_instance(root: Path) -> str:
    root.mkdir(parents=True)
    payloads = {
        "metadata.json": '{"coordinate_system":"local_km"}\n',
        "sites.csv": "site_id,type,x_km,y_km,activity_rate\nH001,H,0,0,1\n",
        "od_nodes.csv": "node_id,role,x_km,y_km\no,origin,0,0\nd,destination,1,1\n",
        "od_demand.csv": (
            "od_id,origin_node,destination_node,order_count,lambda_per_hour\n"
            "od,o,d,100,100\n"
        ),
        "periods.csv": "period,hours\np,1\n",
    }
    for name, text in payloads.items():
        (root / name).write_text(text, encoding="utf-8")
    aggregate, _files = dev.canonical_instance_hash(root)
    return aggregate


def _fake_git(_repo_root: Path, *arguments: str) -> str:
    if arguments == ("rev-parse", "--is-inside-work-tree"):
        return "true"
    if arguments == ("status", "--porcelain=v1", "--untracked-files=all"):
        return ""
    if arguments == ("rev-parse", "--verify", "HEAD"):
        return "a" * 40
    if arguments == ("branch", "--show-current"):
        return "codex/test"
    if arguments == (
        "merge-base",
        "--is-ancestor",
        dev.REQUIRED_BASELINE_COMMIT,
        "a" * 40,
    ):
        return ""
    raise AssertionError(arguments)


def _repository(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path, Path]:
    repo = tmp_path / "repo"
    repo.mkdir()
    for relative in dev.IMPLEMENTATION_RELATIVE_PATHS:
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"{relative.as_posix()}\n", encoding="utf-8")
    harness = repo / module.HARNESS_RELATIVE_PATH
    harness.parent.mkdir(parents=True, exist_ok=True)
    harness.write_text("frozen harness\n", encoding="utf-8")

    specs: dict[str, dev.InstanceSpec] = {}
    for stage, name, candidate_h in (
        ("H4", "H004_L02_E02_R02p00", 4),
        ("H6", "H006_L02_E02_R02p20", 6),
        ("H9", "H009_L02_E02_R02p50", 9),
        ("H10", "H010_L02_E02_R02p60", 10),
    ):
        relative = Path("instances") / name
        aggregate = _write_canonical_instance(repo / relative)
        specs[stage] = dev.InstanceSpec(stage, relative, aggregate, candidate_h)
    monkeypatch.setattr(dev, "INSTANCE_SPECS", specs)
    monkeypatch.setattr(dev, "_git", _fake_git)
    monkeypatch.setattr(dev, "_process_table", lambda: "harmless-process\n")

    source = repo / "MLND" / "main.tex"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text("frozen formulation\n", encoding="utf-8")
    shared_lock = tmp_path / "benchmark.lock"
    monkeypatch.setattr(dev, "DEFAULT_BENCHMARK_RESOURCE_LOCK", shared_lock)
    monkeypatch.setattr(module, "BenchmarkResourceLock", RecordingLock)
    RecordingLock.calls.clear()
    return repo, source, shared_lock


def _config(
    tmp_path: Path,
    repo: Path,
    source: Path,
    lock: Path,
    *,
    requested_stages: tuple[str, ...] | None = None,
    include_h10: bool = False,
    confirm_h10: bool = False,
) -> module.SmokeConfig:
    return module.SmokeConfig(
        repo_root=repo,
        output_root=tmp_path / "output",
        formulation_source=source,
        expected_formulation_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        requested_stages=requested_stages,
        include_h10=include_h10,
        confirm_h10=confirm_h10,
        resource_lock=lock,
    )


def _good_record(
    stage: str,
    candidate_h: int,
    repetition: int,
    *,
    first_signature: str = "a",
) -> dict[str, Any]:
    network_assigned = 12
    return {
        "stage": stage,
        "repetition": repetition,
        "construction_entrypoint": "_construct_objective_partial_starts",
        "backend_optimize_guard_installed": True,
        "timed_solver_run_started": False,
        "optimization_api_call_count": 0,
        "optimize_called_before": False,
        "optimize_called_after": False,
        "main_solve_started_before": False,
        "main_solve_started_after": False,
        "candidate_h": candidate_h,
        "total_demand_per_hour": 100.0,
        "model_variable_count": 100,
        "constructor_elapsed_seconds": 0.10,
        "backend_num_start": 2,
        "start_pool": [
            {
                "start_index": 0,
                "assigned_variable_count": network_assigned,
                "assignment_sha256": first_signature * 64,
                "first_assigned_variable_names": ["x[H001]"],
            },
            {
                "start_index": 1,
                "assigned_variable_count": 8,
                "assignment_sha256": "b" * 64,
                "first_assigned_variable_names": ["x[L001]"],
            },
        ],
        "telemetry": {
            "policy_version": module.POLICY_VERSION,
            "proxy_only": True,
            "proxy_feasibility_tolerances": {
                "qos_arrival_absolute": (
                    module.OBJECTIVE_START_PROXY_QOS_ABS_TOLERANCE
                ),
                "system_energy_relative": (
                    module.OBJECTIVE_START_PROXY_ENERGY_REL_TOLERANCE
                ),
                "construction_search_relative": (
                    module.OBJECTIVE_START_PROXY_SEARCH_REL_TOLERANCE
                ),
                "solver_feasibility_tolerance_linked": False,
            },
            "enabled": True,
            "used": True,
            "budget_seconds": module.START_BUDGET_SECONDS,
            "preparation_seconds": 0.10 + repetition * 0.001,
            "opportunity_count": 4,
            "singleton_support_count": 2,
            "evaluated_support_count": 3,
            "network_candidate_count": 1,
            "heuristic_candidate_count": 1,
            "safety_start_count": 1,
            "installed_start_count": 2,
            "candidate_supports": [
                {
                    "candidate_id": "objective_partial_base_01",
                    "move": "base",
                    "selected_supports": [["o", "d", "H001", "H002", 0]],
                    "open_h": ["H001", "H002"],
                    "open_facilities": ["L001", "H001", "H002"],
                    "active_connections": [
                        ["L001", "H001"],
                        ["H001", "H002"],
                        ["H002", "L001"],
                    ],
                    "capacity_levels": [["H001", 0], ["H002", 0]],
                    "forced_closed_h": [],
                    "forced_off_connections": [],
                    "estimated_service": 20.0,
                    "estimated_service_fraction": 0.5,
                    "fraction_basis_service": 40.0,
                    "proxy_only": True,
                    "score": 10.0,
                    "objective_breakdown": {"net_savings": 10.0},
                    "energy_proxy": {
                        "system_energy_use": 0.5,
                        "system_energy_supply": (
                            module._model_config().usable_battery_time_hours * 4.0
                        ),
                        "base_departure_population": 4.0,
                        "swap_event_population": 0.0,
                    },
                    "assigned_variable_count": network_assigned,
                }
            ],
            "rejection_reasons": [],
            "fallback_reason": None,
            "timed_out": False,
        },
    }


def test_configs_fix_exact_no_optimize_construction_contract() -> None:
    model = module._model_config()
    solver = module._solver_config()

    assert model.feeder_rider_cost_factor == pytest.approx(0.30)
    assert model.display_origin_choices is None
    assert model.general_corridor_choices is None
    assert solver.time_limit == pytest.approx(5.0)
    assert solver.threads == 1
    assert solver.start_mode == "none"
    assert solver.use_objective_partial_start is True
    assert solver.objective_partial_start_seconds == pytest.approx(0.25)
    assert solver.network_start_seconds == 0.0
    assert solver.fixed_integer_repair_seconds == 0.0
    assert solver.use_local_branching is False
    assert solver.local_branching_seconds == 0.0
    assert solver.use_valid_inequalities is True
    assert solver.use_user_cuts is True
    assert solver.root_cut_passes == 2
    assert solver.root_relaxation_method == -1
    assert solver.miqcp_method == -1
    assert solver.mip_focus == 1
    assert solver.heuristics == pytest.approx(0.2)
    assert solver.cut_chi_facility_vub is False
    assert solver.cut_route_swap_station_vub is False
    assert solver.cut_route_swap_leg_vub is False


def test_h9_default_runs_twice_and_persists_exclusive_machine_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, source, lock = _repository(tmp_path, monkeypatch)
    config = _config(tmp_path, repo, source, lock)
    calls: list[tuple[str, int]] = []

    def runner(stage: str, _path: Path, candidate_h: int, repetition: int):
        calls.append((stage, repetition))
        return _good_record(stage, candidate_h, repetition)

    summary = module.run_smoke(config, construction_runner=runner)

    assert summary["status"] == "passed"
    assert summary["timed_solver_run_started"] is False
    assert summary["optimization_api_call_count"] == 0
    assert calls == [("H9", 1), ("H9", 2)]
    assert summary["stage_audits"] == [
        {
            "stage": "H9",
            "completed_repetitions": 2,
            "complete": True,
            "individually_valid": True,
            "deterministic": True,
            "determinism_sha256": summary["stage_audits"][0][
                "determinism_sha256"
            ],
            "passed": True,
        }
    ]
    assert RecordingLock.calls == [
        (lock, "objective-start-construction-smoke")
    ]

    manifest_path = config.output_root / "manifest.json"
    summary_path = config.output_root / "summary.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    persisted = json.loads(summary_path.read_text(encoding="utf-8"))
    assert persisted == summary
    assert manifest["repository"]["required_baseline_ancestor"] == (
        dev.REQUIRED_BASELINE_COMMIT
    )
    assert manifest["formulation_source"]["path"] == str(source)
    assert manifest["protocol"]["optimization_allowed"] is False
    assert manifest["protocol"]["backend_optimize_guard_required"] is True
    assert manifest["protocol"]["timed_solver_runs_planned"] == 0
    assert manifest["protocol"]["stages"] == ["H9"]
    assert manifest["protocol"]["repetitions_per_stage"] == 2

    with pytest.raises(module.ObjectiveStartSmokeError, match="output root"):
        module.run_smoke(config, construction_runner=runner)


def test_h10_requires_both_flags_before_any_artifact_or_construction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, source, lock = _repository(tmp_path, monkeypatch)
    config = _config(
        tmp_path,
        repo,
        source,
        lock,
        include_h10=True,
        confirm_h10=False,
    )
    called = False

    def runner(*_args: Any) -> Mapping[str, Any]:
        nonlocal called
        called = True
        raise AssertionError("must not construct")

    with pytest.raises(module.ObjectiveStartSmokeError, match="H10 requires both"):
        module.run_smoke(config, construction_runner=runner)
    assert called is False
    assert not config.output_root.exists()


def test_h10_dual_confirmation_runs_only_after_passing_h9(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, source, lock = _repository(tmp_path, monkeypatch)
    config = _config(
        tmp_path,
        repo,
        source,
        lock,
        include_h10=True,
        confirm_h10=True,
    )
    calls: list[tuple[str, int]] = []

    def runner(stage: str, _path: Path, candidate_h: int, repetition: int):
        calls.append((stage, repetition))
        return _good_record(stage, candidate_h, repetition)

    summary = module.run_smoke(config, construction_runner=runner)

    assert summary["status"] == "passed"
    assert calls == [("H9", 1), ("H9", 2), ("H10", 1), ("H10", 2)]
    assert [audit["stage"] for audit in summary["stage_audits"]] == ["H9", "H10"]


def test_explicit_micro_stages_preserve_order_and_freeze_selected_hashes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, source, lock = _repository(tmp_path, monkeypatch)
    config = _config(
        tmp_path,
        repo,
        source,
        lock,
        requested_stages=("H6", "H4"),
    )
    calls: list[tuple[str, int]] = []

    def runner(stage: str, _path: Path, candidate_h: int, repetition: int):
        calls.append((stage, repetition))
        return _good_record(stage, candidate_h, repetition)

    summary = module.run_smoke(config, construction_runner=runner)

    assert summary["status"] == "passed"
    assert calls == [("H6", 1), ("H6", 2), ("H4", 1), ("H4", 2)]
    manifest = json.loads((config.output_root / "manifest.json").read_text())
    assert manifest["protocol"]["stages"] == ["H6", "H4"]
    assert set(manifest["instances"]) == {"H4", "H6"}
    for stage in ("H6", "H4"):
        assert manifest["instances"][stage]["sha256"] == (
            dev.INSTANCE_SPECS[stage].expected_sha256
        )


def test_repeated_explicit_stage_is_refused_before_construction_or_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, source, lock = _repository(tmp_path, monkeypatch)
    config = _config(
        tmp_path,
        repo,
        source,
        lock,
        requested_stages=("H4", "H4"),
    )
    called = False

    def runner(*_args: Any) -> Mapping[str, Any]:
        nonlocal called
        called = True
        raise AssertionError("must not construct")

    with pytest.raises(module.ObjectiveStartSmokeError, match="must be unique"):
        module.run_smoke(config, construction_runner=runner)
    assert called is False
    assert not config.output_root.exists()


def test_explicit_h10_selection_requires_second_confirmation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, source, lock = _repository(tmp_path, monkeypatch)
    config = _config(
        tmp_path,
        repo,
        source,
        lock,
        requested_stages=("H4", "H10"),
        confirm_h10=False,
    )

    with pytest.raises(module.ObjectiveStartSmokeError, match="H10 requires both"):
        module.run_smoke(config, construction_runner=lambda *_args: {})
    assert not config.output_root.exists()


def test_nondeterminism_fails_closed_and_stops_before_h10(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, source, lock = _repository(tmp_path, monkeypatch)
    config = _config(
        tmp_path,
        repo,
        source,
        lock,
        include_h10=True,
        confirm_h10=True,
    )
    calls: list[tuple[str, int]] = []

    def runner(stage: str, _path: Path, candidate_h: int, repetition: int):
        calls.append((stage, repetition))
        signature = "c" if repetition == 2 else "a"
        return _good_record(
            stage, candidate_h, repetition, first_signature=signature
        )

    summary = module.run_smoke(config, construction_runner=runner)

    assert summary["status"] == "failed_closed"
    assert summary["stop_reason"] == "H9_nondeterministic_construction"
    assert summary["stage_audits"][0]["deterministic"] is False
    assert calls == [("H9", 1), ("H9", 2)]
    assert (config.output_root / "manifest.json").is_file()
    assert (config.output_root / "summary.json").is_file()
    assert summary["timed_solver_run_started"] is False


@pytest.mark.parametrize(
    ("mutate", "expected_error"),
    [
        (
            lambda record: record["start_pool"][0].update(
                assigned_variable_count=100
            ),
            "start_0_not_sparse",
        ),
        (
            lambda record: record["telemetry"]["candidate_supports"][0].update(
                open_h=[]
            ),
            "candidate_0_open_h_not_interior",
        ),
        (
            lambda record: record["telemetry"]["candidate_supports"][0].update(
                estimated_service=0.0
            ),
            "candidate_0_nonpositive_estimated_service",
        ),
        (
            lambda record: record["telemetry"]["candidate_supports"][0][
                "energy_proxy"
            ].update(system_energy_use=2.0),
            "candidate_0_system_energy_proxy_exceeds_supply",
        ),
        (
            lambda record: record["telemetry"].update(
                preparation_seconds=0.251
            ),
            "preparation_exceeds_hard_budget",
        ),
        (
            lambda record: record["telemetry"].update(used=False),
            "not_used",
        ),
    ],
)
def test_construction_audit_enforces_required_gates(
    mutate: Any, expected_error: str
) -> None:
    record = _good_record("H9", 9, 1)
    mutate(record)

    audit = module._audit_construction(record, candidate_h=9)

    assert audit["valid"] is False
    assert expected_error in audit["errors"]


class FakeVar:
    def __init__(self, name: str) -> None:
        self.VarName = name


class NeverOptimizeBackend:
    def __init__(self) -> None:
        self.variables = [FakeVar(f"v{index}") for index in range(6)]
        self.NumStart = 0
        self.Params = SimpleNamespace(StartNumber=0)
        self.starts: list[dict[FakeVar, float]] = []
        self.optimize_calls = 0
        self.disposed = False

    def update(self) -> None:
        return None

    def getVars(self) -> list[FakeVar]:
        return self.variables

    def getAttr(self, attribute: Any, variables: list[FakeVar]) -> list[float]:
        assert attribute == GRB.Attr.Start
        start = self.starts[self.Params.StartNumber]
        return [start.get(variable, float(GRB.UNDEFINED)) for variable in variables]

    def optimize(self, *_args: Any) -> None:
        self.optimize_calls += 1
        raise AssertionError("optimize must never be called")

    def dispose(self) -> None:
        self.disposed = True


class NeverSolveWrapper:
    def __init__(self, backend: NeverOptimizeBackend) -> None:
        self.model = backend
        self.optimize_called = False
        self.main_solve_started = False
        self.solve_calls = 0
        self.objective_partial_start_telemetry: dict[str, Any] = {}

    def solve(self, *_args: Any, **_kwargs: Any) -> None:
        self.solve_calls += 1
        raise AssertionError("solve must never be called")

    def _construct_objective_partial_starts(
        self, budget: float, _solver: Any
    ) -> None:
        assert budget == pytest.approx(0.25)
        first = {self.model.variables[0]: 1.0, self.model.variables[1]: 0.0}
        safety = {self.model.variables[2]: 0.0}
        self.model.starts = [first, safety]
        self.model.NumStart = 2
        self.objective_partial_start_telemetry = _good_record(
            "H9", 9, 1
        )["telemetry"]
        self.objective_partial_start_telemetry["candidate_supports"][0][
            "assigned_variable_count"
        ] = 2


def test_construct_once_uses_no_optimize_entrypoint_with_pure_mocks() -> None:
    instance = SimpleNamespace(H=[f"H{i}" for i in range(9)], total_demand=100.0)
    backend = NeverOptimizeBackend()
    wrapper = NeverSolveWrapper(backend)
    clock_values = iter((10.0, 10.1))
    hooks = module.ConstructionHooks(
        load_instance=lambda _path, _config: instance,
        create_built_model=lambda _instance: wrapper,
        clock=lambda: next(clock_values),
    )

    record = module._construct_once(
        "H9", Path("unused"), 9, 1, hooks=hooks
    )
    audit = module._audit_construction(record, candidate_h=9)

    assert audit["valid"] is True
    assert wrapper.solve_calls == 0
    assert backend.optimize_calls == 0
    assert backend.disposed is True
    assert record["optimization_api_call_count"] == 0
    assert record["backend_optimize_guard_installed"] is True
    assert record["timed_solver_run_started"] is False
    assert [item["assigned_variable_count"] for item in record["start_pool"]] == [
        2,
        1,
    ]


def test_backend_optimize_guard_blocks_an_accidental_call() -> None:
    instance = SimpleNamespace(H=[f"H{i}" for i in range(9)], total_demand=100.0)
    backend = NeverOptimizeBackend()
    wrapper = NeverSolveWrapper(backend)

    def accidental_optimize(_budget: float, _solver: Any) -> None:
        wrapper.model.optimize()

    wrapper._construct_objective_partial_starts = accidental_optimize  # type: ignore[method-assign]
    hooks = module.ConstructionHooks(
        load_instance=lambda _path, _config: instance,
        create_built_model=lambda _instance: wrapper,
        clock=lambda: 0.0,
    )

    with pytest.raises(module.ObjectiveStartSmokeError, match="forbidden"):
        module._construct_once("H9", Path("unused"), 9, 1, hooks=hooks)

    assert backend.optimize_calls == 0
    assert backend.disposed is True
    assert wrapper.model is backend


def test_process_conflict_after_manifest_fails_closed_without_construction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, source, lock = _repository(tmp_path, monkeypatch)
    config = _config(tmp_path, repo, source, lock)
    process_tables = iter(
        (
            "harmless-process\n",
            "123 1 /opt/anaconda3/bin/python main.py solve --instance-dir x\n",
        )
    )
    monkeypatch.setattr(dev, "_process_table", lambda: next(process_tables))
    called = False

    def runner(*_args: Any) -> Mapping[str, Any]:
        nonlocal called
        called = True
        raise AssertionError("must not construct")

    summary = module.run_smoke(config, construction_runner=runner)

    assert summary["status"] == "failed_closed"
    assert summary["stop_reason"] == "H9_repeat1_process_conflict"
    assert called is False
    assert summary["construction_records"] == []
    assert summary["timed_solver_run_started"] is False
    assert (config.output_root / "summary.json").is_file()


def test_real_instance_symlink_is_refused_before_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, source, lock = _repository(tmp_path, monkeypatch)
    real = repo / dev.INSTANCE_SPECS["H9"].relative_path
    link = repo / "instances" / "H9-link"
    link.symlink_to(real, target_is_directory=True)
    original = dev.INSTANCE_SPECS["H9"]
    monkeypatch.setitem(
        dev.INSTANCE_SPECS,
        "H9",
        dev.InstanceSpec("H9", link.relative_to(repo), original.expected_sha256, 9),
    )
    config = _config(tmp_path, repo, source, lock)

    with pytest.raises(module.ObjectiveStartSmokeError, match="non-symlink"):
        module.run_smoke(config, construction_runner=lambda *_args: {})
    assert not config.output_root.exists()


def test_cli_has_no_option_that_can_enable_optimization() -> None:
    parser = module.build_parser()
    option_strings = {
        option
        for action in parser._actions
        for option in action.option_strings
    }

    assert "--run-timed" not in option_strings
    assert "--optimize" not in option_strings
    assert "--time-limit" not in option_strings
    assert "--stages" in option_strings
    assert "--include-h10" in option_strings
    assert "--confirm-h10" in option_strings

    args = parser.parse_args(
        [
            "--output-root",
            "unused",
            "--stages",
            "H6",
            "H4",
            "--stages",
            "H9",
        ]
    )
    assert args.stages == [["H6", "H4"], ["H9"]]


def test_main_flattens_repeated_stage_option_groups(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[module.SmokeConfig] = []

    def run(config: module.SmokeConfig) -> dict[str, str]:
        captured.append(config)
        return {"status": "passed"}

    monkeypatch.setattr(module, "run_smoke", run)

    assert module.main(
        [
            "--output-root",
            "unused",
            "--stages",
            "H6",
            "H4",
            "--stages",
            "H9",
        ]
    ) == 0
    assert captured[0].requested_stages == ("H6", "H4", "H9")
