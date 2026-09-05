from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import threading
from typing import Any

import pytest

import main as solve_main
from src.benchmark.branch_cut import dev_micro_ce as module
from src.benchmark.branch_cut import telemetry as telemetry_module


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


def _write_canonical_instance(root: Path) -> tuple[str, dict[str, str]]:
    root.mkdir(parents=True)
    payloads = {
        "metadata.json": '{"source":"nanshan"}\n',
        "sites.csv": "site,type,x_km,y_km,fixed_cost_per_hour\nH001,H,0,0,1\n",
        "od_nodes.csv": "node,x_km,y_km\no,0,0\nd,1,1\n",
        "od_demand.csv": "origin,destination,demand_per_hour\no,d,100\n",
        "periods.csv": "period,hours\np,1\n",
    }
    for name, text in payloads.items():
        (root / name).write_text(text, encoding="utf-8")
    return module.canonical_instance_hash(root)


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
        module.REQUIRED_BASELINE_COMMIT,
        "a" * 40,
    ):
        return ""
    raise AssertionError(arguments)


def _repository(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    repo = tmp_path / "repo"
    repo.mkdir()
    for relative in module.IMPLEMENTATION_RELATIVE_PATHS:
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"{relative.as_posix()}\n", encoding="utf-8")

    specs: dict[str, module.InstanceSpec] = {}
    for stage, name, candidate_h in (
        ("H4", "H004_L02_E02_R02p00", 4),
        ("H6", "H006_L02_E02_R02p20", 6),
        ("H9", "H009_L02_E02_R02p50", 9),
        ("H10", "H010_L02_E02_R02p60", 10),
    ):
        relative = Path("instances") / name
        aggregate, _ = _write_canonical_instance(repo / relative)
        specs[stage] = module.InstanceSpec(stage, relative, aggregate, candidate_h)
    monkeypatch.setattr(module, "INSTANCE_SPECS", specs)
    monkeypatch.setattr(module, "_git", _fake_git)
    monkeypatch.setattr(module, "_process_table", lambda: "harmless-process\n")
    RecordingLock.calls.clear()
    monkeypatch.setattr(module, "BenchmarkResourceLock", RecordingLock)

    shared_lock = tmp_path / "benchmark.lock"
    monkeypatch.setattr(module, "DEFAULT_BENCHMARK_RESOURCE_LOCK", shared_lock)

    source = repo / "MLND" / "main.tex"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text("frozen formulation\n", encoding="utf-8")
    return repo, source


def _config(
    tmp_path: Path,
    repo: Path,
    source: Path,
    *,
    requested_stages: tuple[str, ...] | None = None,
    include_h10: bool = False,
    confirm_h10: bool = False,
) -> module.BatchConfig:
    return module.BatchConfig(
        repo_root=repo,
        output_root=tmp_path / "output",
        formulation_source=source,
        expected_formulation_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        requested_stages=requested_stages,
        include_h10=include_h10,
        confirm_h10=confirm_h10,
        jobs=2,
        resource_lock=Path(module.DEFAULT_BENCHMARK_RESOURCE_LOCK),
    )


def _argument(command: tuple[str, ...], flag: str) -> str:
    return command[command.index(flag) + 1]


def _mock_runner(
    config: module.BatchConfig,
    *,
    unreliable_seeds: set[int] | None = None,
    tamper_e: bool = False,
    worse_seeds: set[int] | None = None,
    better_bound_seeds: set[int] | None = None,
    missing_bound_seeds: set[int] | None = None,
    barrier: threading.Barrier | None = None,
) -> tuple[Any, list[dict[str, Any]]]:
    calls: list[dict[str, Any]] = []
    call_lock = threading.Lock()
    implementation_sha, implementation_files = module._implementation_provenance(
        config.repo_root
    )
    source_sha = hashlib.sha256(config.formulation_source.read_bytes()).hexdigest()

    def run(command: tuple[str, ...], **kwargs: Any) -> SimpleNamespace:
        if barrier is not None:
            barrier.wait(timeout=2)
        output = Path(_argument(command, "--results-dir"))
        instance = Path(_argument(command, "--instance-dir"))
        seed = int(_argument(command, "--solver-seed"))
        profile = "E" if "--dev-objective-partial-start" in command else "C"
        stage = next(
            stage
            for stage, spec in module.INSTANCE_SPECS.items()
            if instance == (config.repo_root / spec.relative_path).resolve()
        )
        with call_lock:
            calls.append(
                {
                    "stage": stage,
                    "profile": profile,
                    "seed": seed,
                    "command": command,
                    "env": kwargs["env"],
                }
            )
        kwargs["stdout"].write(b"mock driver output\n")
        kwargs["stdout"].flush()

        aggregate, instance_files = module.canonical_instance_hash(instance)
        assert aggregate == module.INSTANCE_SPECS[stage].expected_sha256
        solver = {
            "time_limit": 5.0,
            "mip_gap": 0.0,
            "threads": 1,
            "output_flag": 1,
            "seed": seed,
            "network_start_seconds": 0.0,
            "network_start_method": 2,
            "use_valid_inequalities": True,
            "use_user_cuts": True,
            "max_root_cut_rounds": 3,
            "root_cut_passes": module.ROOT_CUT_PASSES,
            "max_cuts_per_round": 250,
            "cut_violation_tolerance": 1e-6,
            "cut_efficacy_tolerance": 1e-9,
            "fixed_integer_repair_seconds": 0.0,
            "use_local_branching": False,
            "local_branching_seconds": 0.0,
            "root_relaxation_method": -1,
            "miqcp_method": -1,
            "start_mode": "none",
            "use_objective_partial_start": profile == "E",
            "objective_partial_start_seconds": 0.25,
            "objective_partial_start_max_candidates": 3,
            "mip_focus": 1,
            "heuristics": 0.3 if tamper_e and profile == "E" else 0.2,
            "no_rel_heur_seconds": 0.0,
            "no_rel_heur_time": 0.0,
            "presolve": -1,
            "cuts": -1,
            "numeric_focus": 0,
            "static_nearest_display_suffix": False,
            "static_chi_vub_top_k_per_corridor": 0,
            "cut_chi_facility_vub": False,
            "cut_route_swap_station_vub": False,
            "cut_route_swap_leg_vub": False,
            "collect_progress": True,
            "progress_interval_seconds": 0.25,
        }
        run_config = {
            "schema_version": module.RESULT_SCHEMA_VERSION,
            "formulation_id": module.FORMULATION_ID,
            "formulation_source": {
                "path": str(config.formulation_source.resolve()),
                "sha256": source_sha,
            },
            "implementation": {
                "sha256": implementation_sha,
                "files": implementation_files,
            },
            "instance": {
                "directory": str(instance.resolve()),
                "metadata": {"source": "nanshan"},
                "canonical_file_sha256": instance_files,
            },
            "demand_scale": 1.0,
            "modes": ["MX"],
            "cross_mode_start_policy": None,
            "warm_start_from": None,
            "file_warm_start_policy": "first_mode_only",
            "screening_is_exact": {"MX": True},
            "model": {
                "display_origin_choices": None,
                "general_corridor_choices": None,
                "feeder_rider_cost_factor": module.FEEDER_RIDER_COST_FACTOR,
            },
            "model_derived": {},
            "solver": solver,
        }
        run_config["run_signature_sha256"] = module._computed_run_signature_sha256(
            run_config
        )
        output.mkdir(parents=True)
        (output / "run_config.json").write_text(
            json.dumps(run_config), encoding="utf-8"
        )

        worse = profile == "E" and seed in (worse_seeds or set())
        gap = 0.2 if worse else (0.08 if profile == "E" else 0.1)
        reliable = profile == "E" and seed not in (unreliable_seeds or set())
        telemetry = {
            "policy_version": module.OBJECTIVE_PARTIAL_POLICY_VERSION,
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
            "enabled": profile == "E",
            "used": reliable,
            "budget_seconds": 0.25 if profile == "E" else 0.0,
            "preparation_seconds": 0.1 if profile == "E" else 0.0,
            "network_candidate_count": 1 if reliable else 0,
            "safety_start_count": 1 if reliable else 0,
            "installed_start_count": 2 if reliable else 0,
            "candidate_supports": (
                [
                    {
                        "candidate_id": "base-1",
                        "open_h": ["H001", "H002"],
                        "estimated_service": 20.0,
                        "estimated_service_fraction": 0.5,
                        "fraction_basis_service": 40.0,
                        "proxy_only": True,
                        "score": 10.0,
                        "energy_proxy": {
                            "system_energy_use": 1.0,
                            "system_energy_supply": 1.1,
                            "base_departure_population": 4.0,
                            "swap_event_population": 0.0,
                        },
                    }
                ]
                if reliable
                else []
            ),
            "fallback_reason": None if reliable or profile == "C" else "no_candidate",
            "timed_out": False,
        }
        better_bound_pair = seed in (better_bound_seeds or set())
        baseline = 1500.0 if better_bound_pair else 1000.0
        objective = 1100.0 if profile == "E" and better_bound_pair else 900.0
        bound = objective * (1.0 - gap)
        if profile == "E" and seed in (missing_bound_seeds or set()):
            bound = None
            gap = None
        summary = {
            "schema_version": module.RESULT_SCHEMA_VERSION,
            "formulation_id": module.FORMULATION_ID,
            "run_signature_sha256": run_config["run_signature_sha256"],
            "mode": "MX",
            "status": 9,
            "status_name": "TIME_LIMIT",
            "objective": objective,
            "objective_bound": bound,
            "mip_gap": gap,
            "runtime_seconds": 5.0,
            "courier_only_baseline": baseline,
            "network_is_nontrivial": True,
            "network_opened": True,
            "courier_only_incumbent": False,
            "open_bases": 1,
            "open_lockers": 3,
            "active_lockers": 3,
            "idle_open_lockers": 0,
            "open_swap_stations": 0,
            "screening_is_exact": True,
            "instance": {"screening_is_exact": True},
            "facilities": {
                "candidates": {"lockers": module.INSTANCE_SPECS[stage].candidate_h},
                "open": {"bases": 1, "lockers": 3, "swap_stations": 0},
                "active_lockers": 3,
                "idle_open_lockers": 0,
            },
            "all_rider_baseline": {"cost_per_hour": baseline},
            "demand_funnel_orders_per_hour": {
                "total_demand": 100.0,
                "total_drone_realized": 20.0,
            },
            "energy_balance": {
                "balance_scope": "system_wide_base_departure_plus_swap",
                "total_energy_use_equivalent_drones": 1.0,
                "total_energy_supply_equivalent_drones": 1.1,
                "maximum_constraint_violation": 0.0,
            },
            "solver": {
                "status": 9,
                "status_name": "TIME_LIMIT",
                "solution_count": 2,
                "objective_cost_per_hour": objective,
                "objective_bound": bound,
                "mip_gap": gap,
                "runtime_seconds": 5.0,
                "main_solve_runtime_seconds": 4.9,
                "algorithm_prepare_runtime_seconds": 0.0,
                "start_preparation_runtime_seconds": 0.1 if profile == "E" else 0.0,
                "first_incumbent_time_seconds": 1.0,
                "node_count": 1.0,
                "external_start_installed": False,
                "external_start_source": None,
                "external_start_matched_variables": 0,
                "algorithm": {
                    "objective_partial_start": telemetry,
                    "installed_mip_start_count": 2 if reliable else 0,
                    "all_courier_start_installed": reliable
                    if profile == "E"
                    else False,
                    "native_root_cut_passes": module.ROOT_CUT_PASSES,
                    "applied_native_root_cut_passes": module.ROOT_CUT_PASSES,
                    "no_rel_heur_seconds": 0.0,
                    "no_rel_heur_time": 0.0,
                    "applied_no_rel_heur_time": 0.0,
                    "external_start_installed": False,
                    "external_start_source": None,
                    "external_start_matched_variables": 0,
                },
            },
        }
        mode = output / "MX"
        mode.mkdir()
        (mode / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
        (mode / "solver.log").write_text("mock solver log\n", encoding="utf-8")
        with (mode / "open_facilities.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=(
                    "facility",
                    "type",
                    "fixed_cost_per_hour",
                    "incident_physical_flow_per_hour",
                ),
            )
            writer.writeheader()
            writer.writerow(
                {
                    "facility": "L001",
                    "type": "base",
                    "fixed_cost_per_hour": 14.0,
                    "incident_physical_flow_per_hour": 20.0,
                }
            )
            for index in range(3):
                writer.writerow(
                    {
                        "facility": f"H{index + 1:03d}",
                        "type": "locker",
                        "fixed_cost_per_hour": 1.0,
                        "incident_physical_flow_per_hour": 10.0,
                    }
                )
        return SimpleNamespace(returncode=0)

    return run, calls


def test_default_h9_batch_is_locked_paired_and_audited(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, source = _repository(tmp_path, monkeypatch)
    config = _config(tmp_path, repo, source)
    runner, calls = _mock_runner(config, barrier=threading.Barrier(2))

    summary = module.run_batch(config, process_runner=runner)

    assert summary["status"] == "passed"
    assert len(summary["job_records"]) == 6
    assert len(summary["pair_audits"]) == 3
    assert not summary["skipped_planned_job_ids"]
    assert RecordingLock.calls == [(config.resource_lock, "dev-micro-ce")]
    assert {(call["stage"], call["seed"]) for call in calls} == {
        ("H9", 11),
        ("H9", 29),
        ("H9", 47),
    }
    for call in calls:
        assert call["env"]["OMP_NUM_THREADS"] == "1"
        assert call["env"]["MKL_NUM_THREADS"] == "1"
        assert "--quiet" not in call["command"]
        assert call["command"].count("--threads") == 1
        assert _argument(call["command"], "--threads") == "1"
        parsed = solve_main.build_parser().parse_args(call["command"][2:])
        assert parsed.command == "solve"
        assert parsed.threads == 1
        assert parsed.feeder_rider_cost_factor == pytest.approx(0.30)
        assert parsed.root_cut_passes == module.ROOT_CUT_PASSES
        assert parsed.no_rel_heur_seconds == pytest.approx(0.0)
        assert parsed.no_rel_heur_time == pytest.approx(0.0)
        assert parsed.local_branching_seconds == pytest.approx(0.0)
        assert parsed.start_mode == "none"
    manifest = json.loads((config.output_root / "manifest.json").read_text())
    assert manifest["development_only"] is True
    assert manifest["formal_evidence_eligible"] is False
    assert manifest["protocol"]["stages"] == ["H9"]
    assert manifest["repository"]["required_baseline_ancestor"] == (
        module.REQUIRED_BASELINE_COMMIT
    )
    assert manifest["formulation_source"]["path"] == str(
        (repo / "MLND" / "main.tex").resolve()
    )
    assert (config.output_root / "summary.json").is_file()


def test_real_micro_instance_specs_pin_h4_and_h6() -> None:
    h4 = module.INSTANCE_SPECS["H4"]
    h6 = module.INSTANCE_SPECS["H6"]

    assert h4.relative_path == Path(
        "results/branch_cut_paper/dev/micro_crop_matrix_v2/"
        "nanshan_d04_05_noon_12_13/H004_L02_E02_R02p00"
    )
    assert h4.expected_sha256 == (
        "7a6f524f9a77d395e8e6bd455e752a1d81494137b4dca1d0af105af0ae8d5f11"
    )
    assert h4.candidate_h == 4
    assert h6.relative_path == Path(
        "results/branch_cut_paper/dev/micro_crop_matrix_v2/"
        "nanshan_d04_05_noon_12_13/H006_L02_E02_R02p20"
    )
    assert h6.expected_sha256 == (
        "d6d70495f240c12b35ad3b1fc6d817c7f7b3948a8c62e5662f21fb77790e7617"
    )
    assert h6.candidate_h == 6


def test_explicit_micro_stages_preserve_order_and_frozen_hashes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, source = _repository(tmp_path, monkeypatch)
    config = _config(
        tmp_path,
        repo,
        source,
        requested_stages=("H6", "H4"),
    )
    runner, calls = _mock_runner(config)

    summary = module.run_batch(config, process_runner=runner)

    assert summary["status"] == "passed"
    assert [audit["stage"] for audit in summary["stage_audits"]] == ["H6", "H4"]
    assert [call["stage"] for call in calls[:6]] == ["H6"] * 6
    assert [call["stage"] for call in calls[6:]] == ["H4"] * 6
    manifest = json.loads((config.output_root / "manifest.json").read_text())
    assert manifest["protocol"]["stages"] == ["H6", "H4"]
    assert set(manifest["instances"]) == {"H4", "H6"}
    for stage in ("H6", "H4"):
        assert manifest["instances"][stage]["sha256"] == (
            module.INSTANCE_SPECS[stage].expected_sha256
        )


def test_repeated_explicit_stage_is_refused_before_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, source = _repository(tmp_path, monkeypatch)
    config = _config(
        tmp_path,
        repo,
        source,
        requested_stages=("H4", "H4"),
    )

    with pytest.raises(module.DevMicroCEError, match="must be unique"):
        module.run_batch(config, process_runner=lambda *_args, **_kwargs: None)
    assert not config.output_root.exists()


def test_explicit_h10_requires_confirmation_before_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, source = _repository(tmp_path, monkeypatch)
    config = _config(
        tmp_path,
        repo,
        source,
        requested_stages=("H4", "H10"),
        confirm_h10=False,
    )

    with pytest.raises(module.DevMicroCEError, match="H10 requires both"):
        module.run_batch(config, process_runner=lambda *_args, **_kwargs: None)
    assert not config.output_root.exists()


def test_cli_accepts_ordered_and_repeated_stage_option_groups() -> None:
    parser = module.build_parser()
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
    assert args.include_h10 is False


def test_main_flattens_repeated_stage_option_groups(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[module.BatchConfig] = []

    def run(config: module.BatchConfig) -> dict[str, str]:
        captured.append(config)
        return {"status": "passed"}

    monkeypatch.setattr(module, "run_batch", run)

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


def test_h10_requires_two_explicit_flags(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, source = _repository(tmp_path, monkeypatch)
    config = _config(tmp_path, repo, source, include_h10=True, confirm_h10=False)
    with pytest.raises(module.DevMicroCEError, match="requires both"):
        module.run_batch(config, process_runner=lambda *_args, **_kwargs: None)
    assert not config.output_root.exists()


def test_confirmed_h10_runs_only_after_successful_complete_h9(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, source = _repository(tmp_path, monkeypatch)
    config = _config(
        tmp_path,
        repo,
        source,
        include_h10=True,
        confirm_h10=True,
    )
    runner, calls = _mock_runner(config)

    summary = module.run_batch(config, process_runner=runner)

    assert summary["status"] == "passed"
    assert len(summary["job_records"]) == 12
    assert len(summary["pair_audits"]) == 6
    assert [audit["stage"] for audit in summary["stage_audits"]] == ["H9", "H10"]
    assert all(audit["passed"] for audit in summary["stage_audits"])
    assert {call["stage"] for call in calls} == {"H9", "H10"}
    assert len(summary["pair_process_audits"]) == 6


def test_run_config_drift_fails_closed_and_preserves_first_pair(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, source = _repository(tmp_path, monkeypatch)
    config = _config(tmp_path, repo, source)
    runner, _ = _mock_runner(config, tamper_e=True)

    summary = module.run_batch(config, process_runner=runner)

    assert summary["status"] == "failed_closed"
    assert summary["stop_reason"] == "H9_seed11_infrastructure_failure"
    assert len(summary["job_records"]) == 2
    assert len(summary["skipped_planned_job_ids"]) == 4
    assert Path(summary["job_records"][1]["results_dir"]).is_dir()
    assert any(
        "run_config:solver.heuristics" in error
        for error in summary["job_records"][1]["infrastructure_errors"]
    )


def test_symlinked_result_artifact_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, source = _repository(tmp_path, monkeypatch)
    config = _config(tmp_path, repo, source)
    base_runner, _ = _mock_runner(config)

    def runner(command: tuple[str, ...], **kwargs: Any) -> SimpleNamespace:
        completed = base_runner(command, **kwargs)
        if (
            "--dev-objective-partial-start" in command
            and _argument(command, "--solver-seed") == "11"
        ):
            summary_path = (
                Path(_argument(command, "--results-dir")) / "MX" / "summary.json"
            )
            moved = summary_path.with_name("summary-real.json")
            summary_path.rename(moved)
            summary_path.symlink_to(moved)
        return completed

    summary = module.run_batch(config, process_runner=runner)

    assert summary["status"] == "failed_closed"
    assert summary["stop_reason"] == "H9_seed11_infrastructure_failure"
    e_record = next(
        record for record in summary["job_records"] if record["profile"] == "E"
    )
    assert e_record["infrastructure_valid"] is False
    assert any("symlink" in error for error in e_record["infrastructure_errors"])


def test_exact_f884_no_incumbent_error_is_structured_algorithmic_outcome(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, source = _repository(tmp_path, monkeypatch)
    config = _config(tmp_path, repo, source)
    base_runner, _ = _mock_runner(config)

    def runner(command: tuple[str, ...], **kwargs: Any) -> SimpleNamespace:
        completed = base_runner(command, **kwargs)
        if _argument(command, "--solver-seed") == "11":
            summary_path = (
                Path(_argument(command, "--results-dir")) / "MX" / "summary.json"
            )
            summary_path.unlink()
            kwargs["stdout"].write(
                b"RuntimeError: MX finished without a feasible incumbent (status=9)\n"
            )
            kwargs["stdout"].flush()
            return SimpleNamespace(returncode=1)
        return completed

    summary = module.run_batch(config, process_runner=runner)

    assert summary["status"] == "passed"
    no_incumbents = [
        record for record in summary["job_records"] if record["seed"] == 11
    ]
    assert len(no_incumbents) == 2
    assert all(record["infrastructure_valid"] for record in no_incumbents)
    assert all(record["outcome"] == "no_incumbent" for record in no_incumbents)
    assert all(record["no_incumbent_structured_by_runner"] for record in no_incumbents)
    assert all(not record["qualification_valid"] for record in no_incumbents)
    assert summary["stage_audits"][0]["effective_pair_count"] == 2


def test_incumbent_without_finite_bound_is_not_infrastructure_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, source = _repository(tmp_path, monkeypatch)
    config = _config(tmp_path, repo, source)
    runner, _ = _mock_runner(config, missing_bound_seeds={11})

    summary = module.run_batch(config, process_runner=runner)

    assert summary["status"] == "passed"
    e_record = next(
        record
        for record in summary["job_records"]
        if record["profile"] == "E" and record["seed"] == 11
    )
    assert e_record["outcome"] == "incumbent"
    assert e_record["infrastructure_valid"] is True
    assert e_record["qualification_valid"] is True
    assert e_record["qualification"]["objective_bound"] is None
    assert e_record["qualification"]["mip_gap"] is None
    pair = next(item for item in summary["pair_audits"] if item["seed"] == 11)
    assert pair["infrastructure_valid"] is True
    assert pair["effective_pair"] is True
    assert pair["e_bound"] is None
    assert pair["e_gap"] is None
    assert pair["e_adverse_gap"] is None


def test_scale_consistency_and_nonnegative_time_fail_qualification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, source = _repository(tmp_path, monkeypatch)
    config = _config(tmp_path, repo, source)
    base_runner, _ = _mock_runner(config)

    def runner(command: tuple[str, ...], **kwargs: Any) -> SimpleNamespace:
        completed = base_runner(command, **kwargs)
        if "--dev-objective-partial-start" not in command:
            return completed
        seed = int(_argument(command, "--solver-seed"))
        summary_path = Path(_argument(command, "--results-dir")) / "MX" / "summary.json"
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        if seed == 11:
            payload["mip_gap"] = 0.5
            payload["solver"]["mip_gap"] = 0.5
        elif seed == 29:
            payload["solver"]["algorithm_prepare_runtime_seconds"] = -0.1
        summary_path.write_text(json.dumps(payload), encoding="utf-8")
        return completed

    summary = module.run_batch(config, process_runner=runner)

    assert summary["status"] == "failed_closed"
    assert summary["stop_reason"] == "H9_effective_pair_on_1_of_3_seeds"
    e_records = {
        record["seed"]: record
        for record in summary["job_records"]
        if record["profile"] == "E"
    }
    assert (
        "reported_gap_inconsistent_with_objective_and_bound"
        in e_records[11]["qualification"]["errors"]
    )
    assert "negative_runtime_or_node_field" in e_records[29]["qualification"]["errors"]


def test_one_unreliable_e_start_is_retained_and_other_seeds_continue(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, source = _repository(tmp_path, monkeypatch)
    config = _config(tmp_path, repo, source)
    runner, _ = _mock_runner(config, unreliable_seeds={11})

    summary = module.run_batch(config, process_runner=runner)

    assert summary["status"] == "passed"
    assert len(summary["pair_audits"]) == 3
    assert len(summary["job_records"]) == 6
    e_record = next(
        record for record in summary["job_records"] if record["profile"] == "E"
    )
    assert e_record["treatment_reliable"] is False
    assert "e_partial_start_not_used" in e_record["telemetry_audit"]["errors"]
    assert summary["stage_audits"][0]["e_reliable_start_count"] == 2
    assert not summary["skipped_planned_job_ids"]


def test_two_unreliable_e_starts_fail_only_after_all_h9_seeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, source = _repository(tmp_path, monkeypatch)
    config = _config(tmp_path, repo, source)
    runner, calls = _mock_runner(config, unreliable_seeds={11, 29})

    summary = module.run_batch(config, process_runner=runner)

    assert summary["status"] == "failed_closed"
    assert summary["stop_reason"] == "H9_e_reliable_start_on_1_of_3_seeds"
    assert len(summary["pair_audits"]) == 3
    assert len(summary["job_records"]) == 6
    assert len(calls) == 6


def test_two_of_three_worse_h9_seeds_prevent_confirmed_h10(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, source = _repository(tmp_path, monkeypatch)
    config = _config(
        tmp_path,
        repo,
        source,
        include_h10=True,
        confirm_h10=True,
    )
    runner, calls = _mock_runner(config, worse_seeds={11, 29})

    summary = module.run_batch(config, process_runner=runner)

    assert summary["status"] == "failed_closed"
    assert summary["stop_reason"] == "H9_e_adverse_gap_on_2_of_3_seeds"
    assert len(summary["job_records"]) == 6
    assert len(summary["skipped_planned_job_ids"]) == 6
    assert {call["stage"] for call in calls} == {"H9"}
    assert summary["stage_audits"] == [
        {
            "stage": "H9",
            "completed_seed_count": 3,
            "complete_seed_set": True,
            "e_reliable_start_count": 3,
            "effective_pair_count": 3,
            "e_adverse_gap_count": 2,
            "passed": False,
        }
    ]


def test_terminal_bound_improvement_never_compensates_two_worse_gaps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, source = _repository(tmp_path, monkeypatch)
    config = _config(tmp_path, repo, source)
    runner, _ = _mock_runner(
        config,
        worse_seeds={11, 29},
        better_bound_seeds={11, 29},
    )

    summary = module.run_batch(config, process_runner=runner)

    assert summary["status"] == "failed_closed"
    assert summary["stop_reason"] == "H9_e_adverse_gap_on_2_of_3_seeds"
    adverse_with_better_bound = [
        pair
        for pair in summary["pair_audits"]
        if pair["e_gap_worse"] is True and pair["e_bound_improved"] is True
    ]
    assert len(adverse_with_better_bound) == 2
    assert all(pair["e_adverse_gap"] is True for pair in adverse_with_better_bound)
    assert summary["stage_audits"][0]["e_adverse_gap_count"] == 2


def test_preflight_rejects_dirty_tree_existing_output_and_symlinked_instance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, source = _repository(tmp_path, monkeypatch)
    config = _config(tmp_path, repo, source)

    def dirty_git(root: Path, *arguments: str) -> str:
        if arguments == ("status", "--porcelain=v1", "--untracked-files=all"):
            return " M main.py"
        return _fake_git(root, *arguments)

    monkeypatch.setattr(module, "_git", dirty_git)
    with pytest.raises(module.DevMicroCEError, match="clean committed"):
        module.run_batch(config, process_runner=lambda *_args, **_kwargs: None)

    monkeypatch.setattr(module, "_git", _fake_git)
    config.output_root.mkdir()
    with pytest.raises(module.DevMicroCEError, match="must not exist"):
        module.run_batch(config, process_runner=lambda *_args, **_kwargs: None)

    config.output_root.rmdir()
    original = repo / module.INSTANCE_SPECS["H9"].relative_path
    moved = repo / "real-h9"
    original.rename(moved)
    original.symlink_to(moved, target_is_directory=True)
    with pytest.raises(module.DevMicroCEError, match="non-symlink"):
        module.run_batch(config, process_runner=lambda *_args, **_kwargs: None)


def test_preflight_requires_f884_ancestor_and_worktree_formulation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, source = _repository(tmp_path, monkeypatch)
    config = _config(tmp_path, repo, source)

    def unrelated_git(root: Path, *arguments: str) -> str:
        if arguments[:2] == ("merge-base", "--is-ancestor"):
            raise module.DevMicroCEError("not an ancestor")
        return _fake_git(root, *arguments)

    monkeypatch.setattr(module, "_git", unrelated_git)
    with pytest.raises(module.DevMicroCEError, match="f884 baseline"):
        module.run_batch(config, process_runner=lambda *_args, **_kwargs: None)

    monkeypatch.setattr(module, "_git", _fake_git)
    external = tmp_path / "external-main.tex"
    external.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    external_config = module.BatchConfig(
        repo_root=repo,
        output_root=config.output_root,
        formulation_source=external,
        expected_formulation_sha256=hashlib.sha256(external.read_bytes()).hexdigest(),
        resource_lock=config.resource_lock,
    )
    with pytest.raises(module.DevMicroCEError, match="this worktree's MLND/main.tex"):
        module.run_batch(external_config, process_runner=lambda *_args, **_kwargs: None)


def test_preflight_rejects_nonshared_and_symlinked_resource_locks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, source = _repository(tmp_path, monkeypatch)
    config = _config(tmp_path, repo, source)
    nonshared = tmp_path / "other.lock"
    wrong = module.BatchConfig(
        repo_root=config.repo_root,
        output_root=config.output_root,
        formulation_source=config.formulation_source,
        expected_formulation_sha256=config.expected_formulation_sha256,
        resource_lock=nonshared,
    )
    with pytest.raises(module.DevMicroCEError, match="single shared"):
        module.run_batch(wrong, process_runner=lambda *_args, **_kwargs: None)

    actual = tmp_path / "actual.lock"
    actual.write_text("lock\n", encoding="utf-8")
    shared_link = tmp_path / "shared-link.lock"
    shared_link.symlink_to(actual)
    monkeypatch.setattr(module, "DEFAULT_BENCHMARK_RESOURCE_LOCK", shared_link)
    linked = module.BatchConfig(
        repo_root=config.repo_root,
        output_root=config.output_root,
        formulation_source=config.formulation_source,
        expected_formulation_sha256=config.expected_formulation_sha256,
        resource_lock=shared_link,
    )
    with pytest.raises(module.DevMicroCEError, match="must not be a symlink"):
        module.run_batch(linked, process_runner=lambda *_args, **_kwargs: None)
    assert RecordingLock.calls == []


def test_each_pair_process_audit_blocks_main_run_processes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, source = _repository(tmp_path, monkeypatch)
    config = _config(tmp_path, repo, source)
    process_calls = 0

    def process_table() -> str:
        nonlocal process_calls
        process_calls += 1
        if process_calls == 1:
            return "harmless-process\n"
        return "123 1 /opt/anaconda3/bin/python main.py run --source nanshan\n"

    monkeypatch.setattr(module, "_process_table", process_table)
    summary = module.run_batch(
        config,
        process_runner=lambda *_args, **_kwargs: pytest.fail(
            "solver job must not start after a per-pair process conflict"
        ),
    )

    assert summary["status"] == "failed_closed"
    assert summary["stop_reason"] == "H9_seed11_process_conflict"
    assert summary["pair_process_audits"][0]["valid"] is False
    assert len(summary["skipped_planned_job_ids"]) == 6


def test_preparation_share_uses_fixed_five_second_limit() -> None:
    job = module.JobSpec(
        stage="H9",
        profile="E",
        seed=11,
        instance_dir=Path("instance"),
        results_dir=Path("results"),
        driver_log=Path("driver.log"),
        cache_dir=Path("cache"),
    )
    qualification = {
        "candidate_h": 9,
        "total_demand_per_hour": 100.0,
        "total_algorithm_runtime_seconds": 2.0,
        "start_preparation_runtime_seconds": 0.24,
    }
    telemetry = {
        "policy_version": module.OBJECTIVE_PARTIAL_POLICY_VERSION,
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
        "timed_out": False,
        "fallback_reason": None,
        "budget_seconds": 0.25,
        "preparation_seconds": 0.24,
        "network_candidate_count": 1,
        "safety_start_count": 1,
        "installed_start_count": 2,
        "candidate_supports": [
            {
                "open_h": ["H001", "H002"],
                "estimated_service": 2.0,
                "estimated_service_fraction": 0.5,
                "fraction_basis_service": 4.0,
                "proxy_only": True,
                "score": 1.0,
                "energy_proxy": {
                    "system_energy_use": 1.0,
                    "system_energy_supply": 1.0,
                    "base_departure_population": 3.0,
                    "swap_event_population": 0.0,
                },
            }
        ],
    }
    summary = {
        "solver": {
            "algorithm": {
                "objective_partial_start": telemetry,
                "installed_mip_start_count": 2,
                "all_courier_start_installed": True,
                "native_root_cut_passes": module.ROOT_CUT_PASSES,
                "applied_native_root_cut_passes": module.ROOT_CUT_PASSES,
                "no_rel_heur_seconds": 0.0,
                "no_rel_heur_time": 0.0,
                "applied_no_rel_heur_time": 0.0,
                "external_start_installed": False,
                "external_start_source": None,
                "external_start_matched_variables": 0,
            },
            "external_start_installed": False,
            "external_start_source": None,
            "external_start_matched_variables": 0,
        }
    }
    audit = module._telemetry_audit(job, summary, qualification)
    assert audit["valid"] is True

    qualification["start_preparation_runtime_seconds"] = 0.2405
    audit = module._telemetry_audit(job, summary, qualification)
    assert audit["valid"] is True

    qualification["start_preparation_runtime_seconds"] = 0.2411
    audit = module._telemetry_audit(job, summary, qualification)
    assert audit["valid"] is False
    assert audit["errors"] == ["e_preparation_telemetry_mismatch"]


def test_fatal_base_exceptions_from_job_runner_propagate(tmp_path: Path) -> None:
    config = module.BatchConfig(
        repo_root=tmp_path,
        output_root=tmp_path / "output",
    )
    job = module.JobSpec(
        stage="H9",
        profile="C",
        seed=11,
        instance_dir=tmp_path / "instance",
        results_dir=tmp_path / "output" / "runs" / "H9" / "C" / "seed011",
        driver_log=tmp_path / "output" / "driver_logs" / "job.log",
        cache_dir=tmp_path / "output" / "cache" / "job",
    )
    config.output_root.mkdir()

    def interrupt(*_args: object, **_kwargs: object) -> None:
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        module._run_one_job(config, job, interrupt)


def test_telemetry_audits_cutpasses_and_external_start_state() -> None:
    backend = SimpleNamespace(
        Status=9,
        SolCount=1,
        ObjVal=10.0,
        ObjBound=9.0,
        MIPGap=0.1,
        Runtime=1.0,
        NodeCount=2.0,
        NumStart=3,
    )
    wrapper = SimpleNamespace(
        model=backend,
        total_runtime=1.2,
        main_solve_runtime=1.0,
        objective_partial_start_telemetry={"enabled": True},
        external_start_installed=True,
        external_start_source="file:test.sol",
        external_start_matched_variables=17,
    )

    metrics = telemetry_module.collect_solver_metrics(wrapper)

    assert ("root_cut_passes", "CutPasses") in (
        telemetry_module.APPLIED_GUROBI_PARAMETER_ATTRIBUTES
    )
    assert "CutPasses" in telemetry_module.CORE_SOLVER_ALLOWED_GUROBI_PARAMETERS
    assert metrics["installed_mip_start_count"] == 3
    assert metrics["external_start_installed"] is True
    assert metrics["external_start_source"] == "file:test.sol"
    assert metrics["external_start_matched_variables"] == 17
