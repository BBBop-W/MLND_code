from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from src.benchmark.branch_cut import h6_facility_incident_star_ablation as module
from src.benchmark.branch_cut.hashing import canonical_sha256
from src.benchmark.branch_cut.qualification import (
    QualificationEvidence,
    assess_qualification,
    service_tolerance,
)
from src.benchmark.branch_cut.qualification_runner import (
    QualificationModelAudit,
    QualificationRunnerOutcome,
)


class RecordingLock:
    calls: list[tuple[Path, str]] = []

    def __init__(self, path: Path, command: str) -> None:
        self.path = Path(path)
        self.command = command

    def __enter__(self) -> RecordingLock:
        self.calls.append((self.path, self.command))
        return self

    def __exit__(self, *_args: object) -> None:
        return None


def _frozen(tmp_path: Path) -> tuple[module.FrozenInputs, Path, Path]:
    repo = tmp_path / "repo"
    development = repo / module.DEVELOPMENT_ROOT_RELATIVE_PATH
    development.mkdir(parents=True)
    formulation = repo / module.FORMULATION_RELATIVE_PATH
    formulation.parent.mkdir(parents=True, exist_ok=True)
    formulation.write_text("frozen formulation\n", encoding="utf-8")
    instance = repo / module.INSTANCE_RELATIVE_PATH
    instance.mkdir(parents=True)
    (instance / "metadata.json").write_text("{}\n", encoding="utf-8")
    implementation_files = {
        relative.as_posix(): f"{index + 1:064x}"
        for index, relative in enumerate(module.IMPLEMENTATION_RELATIVE_PATHS)
    }
    model_sha256, model_files = module._subset_provenance(
        implementation_files, module._MODEL_IMPLEMENTATION_RELATIVE_PATHS
    )
    qualification_sha256, qualification_files = module._subset_provenance(
        implementation_files, module._QUALIFICATION_IMPLEMENTATION_RELATIVE_PATHS
    )
    frozen = module.FrozenInputs(
        repo_root=repo,
        commit="a" * 40,
        branch="codex/test-h6-facility-star",
        implementation_sha256=canonical_sha256(implementation_files),
        implementation_files=implementation_files,
        model_implementation_sha256=model_sha256,
        model_implementation_files=model_files,
        qualification_implementation_sha256=qualification_sha256,
        qualification_implementation_files=qualification_files,
        formulation_path=formulation,
        formulation_sha256=module.EXPECTED_FORMULATION_SHA256,
        instance_dir=instance,
        instance_sha256=module.EXPECTED_INSTANCE_SHA256,
        instance_files={"metadata.json": "b" * 64},
        screening_by_fixed_cost={"fc24": False, "fc26": False},
        python_executable=Path(module.sys.executable).resolve(),
    )
    return (
        frozen,
        development / "h6-facility-star-plan.json",
        development / "h6-facility-star-run",
    )


def _accepted_outcome(
    frozen: module.FrozenInputs,
    fixed_cost: float,
    qualification_id: str,
) -> QualificationRunnerOutcome:
    profile, contract = module._qualification_contract(frozen, fixed_cost)
    case_id = f"h6-fc{int(fixed_cost):02d}-current-model"
    threshold = service_tolerance(1000.0)
    evidence = QualificationEvidence(
        case_id=case_id,
        scale="small",
        courier_only_objective=7026.0,
        witness_objective=7000.0,
        total_demand_per_hour=1000.0,
        witness_service_per_hour=50.0,
        open_facilities=2,
        positive_cost_open_facilities=2,
        candidate_h=6,
        open_h=2,
        active_h=2,
        active_h_threshold_per_hour=threshold,
        witness_feasible=True,
        qualification_run_id=qualification_id,
        instance_sha256=frozen.instance_sha256,
        qualification_case_sha256=contract.qualification_case_sha256,
        effective_model_config_sha256=contract.effective_model_config_sha256,
        screening_is_exact=False,
    )
    decision = assess_qualification(evidence, module._qualification_policy())
    assert decision.accepted is True
    audit = QualificationModelAudit(
        effective_model_config_sha256=contract.effective_model_config_sha256,
        demand_scale=profile.demand_scale,
        locker_fixed_cost_per_hour=fixed_cost,
        display_origin_choices=profile.display_origin_choices,
        general_corridor_choices=profile.general_corridor_choices,
        screening_is_exact=False,
        model_fingerprint=12345,
    )
    return QualificationRunnerOutcome(
        accepted=True,
        rejection_reasons=(),
        evidence=evidence,
        decision=decision,
        required_service_per_hour=1.0,
        required_objective_upper_bound=(
            evidence.courier_only_objective
            - max(
                float(decision.objective_margin or 0.0),
                module._qualification_policy().minimum_savings_fraction
                * evidence.courier_only_objective,
            )
        ),
        temporary_constraint_used=True,
        temporary_constraint_removed=True,
        fresh_model_validation_performed=True,
        witness_status="OPTIMAL",
        validation_status="TIME_LIMIT",
        captured_variable_count=123,
        exception_messages=(),
        feasibility_objective_used=True,
        expected_model_contract=contract,
        witness_model_audit=audit,
        fresh_model_audit=audit,
        geometry_start_policy=module._qualification_geometry_policy(),
    )


def _write_accepted_qualification_artifact(
    frozen: module.FrozenInputs, root: Path
) -> Path:
    root.mkdir()
    (root / "checkpoints").mkdir()
    identity = module._qualification_identity_payload(frozen)
    qualification_id = module._qualification_id(identity)
    records: list[dict[str, Any]] = []
    snapshot = module._expected_source_snapshot(frozen)
    for fixed_cost in module.FIXED_COSTS:
        profile, contract = module._qualification_contract(frozen, fixed_cost)
        case_id = f"h6-fc{int(fixed_cost):02d}-current-model"
        outcome = _accepted_outcome(frozen, fixed_cost, qualification_id)
        checkpoint = {
            "artifact_type": module.QUALIFICATION_CHECKPOINT_ARTIFACT_TYPE,
            "schema_version": module.QUALIFICATION_SCHEMA_VERSION,
            "development_only": True,
            "formal_evidence_eligible": False,
            "qualification_id": qualification_id,
            "case": {
                "case_id": case_id,
                "scale_label": "small",
                "fixed_cost": fixed_cost,
                "profile": profile.to_payload(),
                "contract": contract.to_payload(),
            },
            "source_before": snapshot,
            "source_after": snapshot,
            "status": "completed",
            "infrastructure_error": None,
            "outcome": outcome.to_payload(),
        }
        checkpoint_path = module._qualification_checkpoint_path(root, fixed_cost)
        module.write_canonical_json_exclusive(checkpoint_path, checkpoint)
        records.append(
            {
                "case_id": case_id,
                "fixed_cost": fixed_cost,
                "status": "completed",
                "accepted": True,
                "fresh_accepted": True,
                "checkpoint_path": str(checkpoint_path),
                "checkpoint_sha256": module._sha256_file(checkpoint_path),
            }
        )
    summary = {
        "artifact_type": module.QUALIFICATION_ARTIFACT_TYPE,
        "schema_version": module.QUALIFICATION_SCHEMA_VERSION,
        "development_only": True,
        "formal_evidence_eligible": False,
        "eligible_only_as_ablation_prerequisite": True,
        "qualification_id": qualification_id,
        "identity": identity,
        "started_at_utc": "2026-08-31T11:00:00+00:00",
        "completed_at_utc": "2026-08-31T11:01:00+00:00",
        "status": "qualified",
        "both_fresh_accepted": True,
        "stop_reason": None,
        "case_records": records,
        "process_audits": [
            {
                "fixed_cost": fixed_cost,
                "process_table_sha256": f"{index + 10:064x}",
                "conflicting_process_count": 0,
                "valid": True,
            }
            for index, fixed_cost in enumerate(module.FIXED_COSTS)
        ],
    }
    summary_path = root / "summary.json"
    module.write_canonical_json_exclusive(summary_path, summary)
    return summary_path


def _build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[module.FrozenInputs, Path, Path, dict[str, Any]]:
    frozen, plan_path, run_root = _frozen(tmp_path)
    monkeypatch.setattr(module, "_freeze_inputs", lambda _repo: frozen)
    qualification_summary = _write_accepted_qualification_artifact(
        frozen, plan_path.parent / "h6-qualification"
    )
    payload = module.build_plan(
        repo_root=frozen.repo_root,
        qualification_summary_path=qualification_summary,
        plan_path=plan_path,
        run_root=run_root,
        now=lambda: datetime(2026, 8, 31, 12, 0, tzinfo=timezone.utc),
    )
    return frozen, plan_path, run_root, payload


def _mock_process_runner(
    plan: dict[str, Any],
    *,
    contaminate_baseline_with_incident_star: bool = False,
    outcome_mode: str = "incumbent",
    tamper_instance_hash: bool = False,
    malformed_solver: bool = False,
) -> tuple[Any, list[str]]:
    calls: list[str] = []
    by_results = {str(job["results_dir"]): job for job in plan["planned_jobs"]}
    model_paths = {
        path.as_posix() for path in module._MODEL_IMPLEMENTATION_RELATIVE_PATHS
    }
    main_files = {
        key: value
        for key, value in plan["repository"]["implementation_files"].items()
        if key in model_paths
    }

    def argument(command: tuple[str, ...], flag: str) -> str:
        return command[command.index(flag) + 1]

    def run(command: tuple[str, ...], **kwargs: Any) -> SimpleNamespace:
        results_dir = Path(argument(command, "--results-dir"))
        job = by_results[str(results_dir)]
        calls.append(str(job["job_id"]))
        kwargs["stdout"].write(b"mock solver completed\n")
        kwargs["stdout"].flush()
        run_config: dict[str, Any] = {
            "schema_version": module.RESULT_SCHEMA_VERSION,
            "formulation_id": module.FORMULATION_ID,
            "formulation_source": {
                "path": plan["formulation"]["path"],
                "sha256": plan["formulation"]["sha256"],
            },
            "implementation": {
                "sha256": canonical_sha256(main_files),
                "files": main_files,
            },
            "instance": {
                "directory": plan["instance"]["path"],
                "metadata": {},
                "canonical_file_sha256": plan["instance"]["canonical_file_sha256"],
            },
            "demand_scale": 1.0,
            "modes": ["MX"],
            "cross_mode_start_policy": None,
            "warm_start_from": None,
            "file_warm_start_policy": "first_mode_only",
            "screening_is_exact": {"MX": False},
            "model": job["model_config"],
            "model_derived": {
                "drone_speed_kmph": 72.0,
                "usable_battery_time_hours": 10.0 / 72.0,
                "usable_battery_time_minutes": 10.0 / 72.0 * 60.0,
            },
            "solver": job["solver_config"],
        }
        if tamper_instance_hash:
            run_config["instance"]["canonical_file_sha256"] = {
                "metadata.json": "0" * 64
            }
        if malformed_solver:
            run_config["solver"] = "not-an-object"
        run_config["run_signature_sha256"] = canonical_sha256(
            module._run_signature_payload(run_config)
        )
        results_dir.mkdir(parents=True)
        (results_dir / "run_config.json").write_text(
            json.dumps(run_config), encoding="utf-8"
        )
        if outcome_mode == "generic_failure":
            kwargs["stdout"].write(b"mock process crashed\n")
            kwargs["stdout"].flush()
            return SimpleNamespace(returncode=2)
        if outcome_mode == "no_incumbent":
            kwargs["stdout"].write(
                b"RuntimeError: MX finished without a feasible incumbent (status=9)\n"
            )
            kwargs["stdout"].flush()
            return SimpleNamespace(returncode=1)
        separation = {"chi_od_corridor_vub": 0.01, "queue_perspective": 0.001}
        if job["cut_chi_facility_vub"] or contaminate_baseline_with_incident_star:
            separation["chi_od_incident_star_vub"] = 0.007
        algorithm = {
            "root_user_cut_separation_seconds": separation,
            "root_user_cut_candidate_counts": {"chi_od_corridor_vub": 110},
            "root_user_cut_counts": {"chi_od_corridor_vub": 105},
            "root_user_cut_duplicate_counts": {},
            "root_user_cut_rejection_counts": {},
        }
        summary = {
            "schema_version": module.RESULT_SCHEMA_VERSION,
            "formulation_id": module.FORMULATION_ID,
            "run_signature_sha256": run_config["run_signature_sha256"],
            "mode": "MX",
            "status": 9,
            "status_name": "TIME_LIMIT",
            "objective": 7000.0,
            "objective_bound": 6980.0,
            "mip_gap": 20.0 / 7000.0,
            "runtime_seconds": 5.0,
            "screening_is_exact": False,
            "instance": {"screening_is_exact": False},
            "open_lockers": 6 if job["slot_index"] == 1 else 4,
            "active_lockers": 6 if job["slot_index"] == 1 else 4,
            "courier_only_baseline": 7026.0,
            "demand_funnel_orders_per_hour": {
                "total_demand": 1000.0,
                "total_drone_realized": 50.0,
            },
            "network_is_nontrivial": True,
            "solver": {"algorithm": algorithm},
        }
        mode_dir = results_dir / "MX"
        mode_dir.mkdir()
        (mode_dir / "solver.log").write_text("Gurobi mock log\n", encoding="utf-8")
        (mode_dir / "incumbent.sol").write_text("x_H0 1\n", encoding="utf-8")
        (mode_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
        return SimpleNamespace(returncode=0)

    return run, calls


def test_clean_repository_accepts_full_sha1_commit_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()

    def fake_git(_repo: Path, *arguments: str) -> str:
        responses = {
            ("rev-parse", "--is-inside-work-tree"): "true",
            ("status", "--porcelain=v1", "--untracked-files=all"): "",
            ("rev-parse", "--verify", "HEAD"): "a" * 40,
            ("rev-parse", "--show-object-format"): "sha1",
            ("branch", "--show-current"): "codex/test-h6-facility-star",
            (
                "merge-base",
                "--is-ancestor",
                module.REQUIRED_BASELINE_COMMIT,
                "a" * 40,
            ): "",
        }
        return responses[arguments]

    monkeypatch.setattr(module, "_git", fake_git)

    assert module._clean_repository(repo) == (
        "a" * 40,
        "codex/test-h6-facility-star",
    )


def test_python_executable_symlink_is_frozen_to_real_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "python-real"
    target.write_text("#!/bin/sh\n", encoding="utf-8")
    link = tmp_path / "python-link"
    link.symlink_to(target)
    monkeypatch.setattr(module.sys, "executable", str(link))

    assert module._resolved_python_executable() == target.resolve()


def test_h6_qualification_profiles_have_frozen_model_and_case_hashes(
    tmp_path: Path,
) -> None:
    frozen, _plan_path, _run_root = _frozen(tmp_path)
    observed = []
    for fixed_cost in module.FIXED_COSTS:
        _profile, contract = module._qualification_contract(frozen, fixed_cost)
        observed.append(
            (
                contract.effective_model_config_sha256,
                contract.qualification_case_sha256,
            )
        )
    assert observed == [
        (
            "74063b03b6e2c46fd2cdaf25e79305f993e11361bd7fcf53c53f4bc766bca7c7",
            "a4c8b182808ad2f91de07e67bcbbb0817898d63f6b1bc7c8dc60096ea9d53762",
        ),
        (
            "04291e5be29920cff010c465123118706a4b4304455b928439ba99daebc0261b",
            "b7875d3270dd1a0f07d7df1430c558934c777d2ae9d0bc119308234b2d02bf8d",
        ),
    ]


def test_mock_qualification_wrapper_publishes_two_fresh_accepted_cases(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frozen, plan_path, _run_root = _frozen(tmp_path)
    qualification_root = plan_path.parent / "qualification-run"
    expected_snapshot = module._expected_source_snapshot(frozen)
    monkeypatch.setattr(module, "_freeze_inputs", lambda _repo: frozen)
    monkeypatch.setattr(module, "_source_snapshot", lambda _frozen: expected_snapshot)
    RecordingLock.calls.clear()
    monkeypatch.setattr(module, "BenchmarkResourceLock", RecordingLock)
    qualifier_calls: list[dict[str, Any]] = []

    def load_instance(
        _instance_dir: Path, model_config: Any, demand_scale: float
    ) -> SimpleNamespace:
        return SimpleNamespace(
            config=model_config,
            metadata={"demand_scale": demand_scale},
            screening_is_exact=lambda mode: mode != "MX",
        )

    def qualify(_instance: object, _mode: str, **kwargs: Any) -> Any:
        qualifier_calls.append(kwargs)
        fixed_cost = kwargs["expected_model_contract"].locker_fixed_cost_per_hour
        return _accepted_outcome(
            frozen,
            fixed_cost,
            kwargs["qualification_run_id"],
        )

    summary = module.run_h6_qualification(
        repo_root=frozen.repo_root,
        qualification_root=qualification_root,
        qualifier=qualify,
        instance_loader=load_instance,
        process_table_provider=lambda: "harmless\n",
        now=lambda: datetime(2026, 8, 31, 11, 0, tzinfo=timezone.utc),
    )

    assert summary["status"] == "qualified"
    assert summary["both_fresh_accepted"] is True
    assert [record["fixed_cost"] for record in summary["case_records"]] == [
        24.0,
        26.0,
    ]
    assert len(qualifier_calls) == 2
    for call in qualifier_calls:
        settings = call["settings"]
        assert settings.time_limit_seconds == 15.0
        assert settings.threads == 1
        assert settings.solver_seed == 42
        assert settings.method == -1
        assert settings.miqcp_method == -1
        assert settings.native_cuts == -1
        assert settings.root_cut_passes == 2
        assert settings.mip_focus == 0
        assert settings.heuristics == pytest.approx(0.05)
        assert call["geometry_start_policy"].enabled is False
        assert call["fail_fast_infrastructure"] is True
    assert RecordingLock.calls == [
        (
            Path(module.DEFAULT_BENCHMARK_RESOURCE_LOCK).resolve(),
            "h6-current-model-qualification",
        )
    ]
    verified = module.verify_qualification_artifact(
        qualification_root / "summary.json", frozen=frozen
    )
    assert verified.qualification_id == summary["qualification_id"]


def test_qualification_verifier_recomputes_policy_and_rejects_evidence_tamper(
    tmp_path: Path,
) -> None:
    frozen, plan_path, _run_root = _frozen(tmp_path)
    summary_path = _write_accepted_qualification_artifact(
        frozen, plan_path.parent / "qualification-tamper"
    )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    checkpoint_path = Path(summary["case_records"][0]["checkpoint_path"])
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    checkpoint["outcome"]["evidence"]["witness_service_per_hour"] = 0.0
    checkpoint_path.write_text(json.dumps(checkpoint), encoding="utf-8")
    summary["case_records"][0]["checkpoint_sha256"] = module._sha256_file(
        checkpoint_path
    )
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    with pytest.raises(
        module.H6FacilityIncidentStarAblationError,
        match="lacks fresh accepted evidence",
    ):
        module.verify_qualification_artifact(summary_path, frozen=frozen)


def test_qualification_verifier_rejects_canonical_source_after_tamper(
    tmp_path: Path,
) -> None:
    frozen, plan_path, _run_root = _frozen(tmp_path)
    summary_path = _write_accepted_qualification_artifact(
        frozen, plan_path.parent / "qualification-source-tamper"
    )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    checkpoint_path = Path(summary["case_records"][1]["checkpoint_path"])
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    checkpoint["source_after"]["instance_sha256"] = "0" * 64
    checkpoint_path.write_text(json.dumps(checkpoint), encoding="utf-8")
    summary["case_records"][1]["checkpoint_sha256"] = module._sha256_file(
        checkpoint_path
    )
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    with pytest.raises(
        module.H6FacilityIncidentStarAblationError,
        match="lacks fresh accepted evidence",
    ):
        module.verify_qualification_artifact(summary_path, frozen=frozen)


def test_qualification_verifier_rejects_nonancestor_commit_identity(
    tmp_path: Path,
) -> None:
    frozen, plan_path, _run_root = _frozen(tmp_path)
    summary_path = _write_accepted_qualification_artifact(
        frozen, plan_path.parent / "qualification-old-commit"
    )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["identity"]["repository"]["commit"] = "b" * 40
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    with pytest.raises(
        module.H6FacilityIncidentStarAblationError,
        match="not an ancestor of the current commit",
    ):
        module.verify_qualification_artifact(summary_path, frozen=frozen)


def test_qualification_verifier_reuses_ancestor_when_model_files_are_identical(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frozen, plan_path, _run_root = _frozen(tmp_path)
    summary_path = _write_accepted_qualification_artifact(
        frozen, plan_path.parent / "qualification-ancestor"
    )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    ancestor_commit = "b" * 40
    summary["identity"]["repository"]["commit"] = ancestor_commit
    qualification_id = module._qualification_id(summary["identity"])
    summary["qualification_id"] = qualification_id
    for record in summary["case_records"]:
        checkpoint_path = Path(record["checkpoint_path"])
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        checkpoint["qualification_id"] = qualification_id
        checkpoint["source_before"]["repository_commit"] = ancestor_commit
        checkpoint["source_after"]["repository_commit"] = ancestor_commit
        checkpoint["outcome"]["evidence"]["qualification_run_id"] = qualification_id
        recomputed_evidence = QualificationEvidence(**checkpoint["outcome"]["evidence"])
        checkpoint["outcome"]["decision"] = assess_qualification(
            recomputed_evidence, module._qualification_policy()
        ).to_payload()
        checkpoint_path.write_text(json.dumps(checkpoint), encoding="utf-8")
        record["checkpoint_sha256"] = module._sha256_file(checkpoint_path)
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    observed: list[str] = []
    monkeypatch.setattr(
        module,
        "_verify_qualification_source_ancestry",
        lambda _frozen, source_commit: observed.append(source_commit),
    )

    verified = module.verify_qualification_artifact(summary_path, frozen=frozen)

    assert verified.qualification_id == qualification_id
    assert observed == [ancestor_commit]


def test_plan_freezes_exact_twelve_slots_and_balanced_rotation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _frozen_inputs, plan_path, run_root, payload = _build(tmp_path, monkeypatch)

    assert plan_path.is_file()
    assert not run_root.exists()
    assert payload["development_only"] is True
    assert payload["formal_evidence_eligible"] is False
    assert payload["qualification_evidence_eligible"] is False
    assert payload["screening_is_exact_by_fixed_cost"] == {
        "fc24": False,
        "fc26": False,
    }
    assert payload["qualification_gate"]["reuse_policy"] == (
        "ancestor_commit_identical_model_implementation_v1"
    )
    assert payload["qualification_gate"]["model_implementation_identical"] is True
    assert payload["qualification_gate"]["wrapper_identity_reused"] is False
    jobs = payload["planned_jobs"]
    assert len(jobs) == 12
    assert [
        (jobs[index]["seed"], jobs[index]["fixed_cost"]) for index in range(0, 12, 2)
    ] == [
        (11, 24.0),
        (11, 26.0),
        (29, 24.0),
        (29, 26.0),
        (47, 24.0),
        (47, 26.0),
    ]
    orders = [
        tuple(job["variant"] for job in jobs[index : index + 2])
        for index in range(0, 12, 2)
    ]
    assert orders == [
        module.VARIANTS,
        tuple(reversed(module.VARIANTS)),
        tuple(reversed(module.VARIANTS)),
        module.VARIANTS,
        module.VARIANTS,
        tuple(reversed(module.VARIANTS)),
    ]
    for index in range(0, 12, 2):
        left, right = jobs[index : index + 2]
        normalized_left = dict(left["solver_config"])
        normalized_right = dict(right["solver_config"])
        normalized_left.pop("cut_chi_facility_vub")
        normalized_right.pop("cut_chi_facility_vub")
        assert normalized_left == normalized_right
        assert left["model_config"] == right["model_config"]
        assert {left["cut_chi_facility_vub"], right["cut_chi_facility_vub"]} == {
            False,
            True,
        }
        assert left["solver_config"]["time_limit"] == 5.0
        assert left["solver_config"]["threads"] == 1
        assert left["solver_config"]["cuts"] == -1
        assert left["solver_config"]["root_cut_passes"] == 2
        assert left["solver_config"]["mip_focus"] == 1
        assert left["solver_config"]["heuristics"] == pytest.approx(0.2)
        assert left["solver_config"]["cut_directed_return"] is False
        assert right["solver_config"]["cut_directed_return"] is False
        assert left["solver_config"]["cut_route_swap_station_vub"] is False
        assert left["solver_config"]["cut_route_swap_leg_vub"] is False


def test_pair_contract_mismatch_cannot_be_comparison_eligible() -> None:
    common = {
        "pair_id": "h6-facility-star-pair-test",
        "fixed_cost": 24.0,
        "seed": 11,
        "infrastructure_valid": True,
        "development_comparison_eligible": True,
        "all_h_open": False,
        "outcome": "incumbent",
    }
    audit = module._pair_audit(
        [
            {
                **common,
                "variant": "C-current",
                "normalized_run_config_sha256": "a" * 64,
            },
            {
                **common,
                "variant": "C-facility-incident-star",
                "normalized_run_config_sha256": "b" * 64,
            },
        ]
    )

    assert audit["only_difference_is_cut_chi_facility_vub"] is False
    assert audit["infrastructure_valid"] is False
    assert audit["development_comparison_eligible"] is False


def test_verifier_rejects_consistently_rehashed_treatment_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _frozen_inputs, plan_path, _run_root, payload = _build(tmp_path, monkeypatch)
    tampered = json.loads(json.dumps(payload))
    job = tampered["planned_jobs"][0]
    job["solver_config"]["heuristics"] = 0.3
    job["solver_config_sha256"] = canonical_sha256(job["solver_config"])
    tampered["plan_id"] = module._plan_id(tampered)
    tampered_path = plan_path.with_name("tampered-plan.json")
    tampered_path.write_text(json.dumps(tampered), encoding="utf-8")

    with pytest.raises(
        module.H6FacilityIncidentStarAblationError,
        match="differs beyond|reconstructed|command violates|shared solver config",
    ):
        module.verify_plan(
            tampered_path,
            expected_plan_id=tampered["plan_id"],
        )


def test_mock_runner_is_serial_and_retains_all_open_outcome(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _frozen_inputs, plan_path, run_root, payload = _build(tmp_path, monkeypatch)
    RecordingLock.calls.clear()
    monkeypatch.setattr(module, "BenchmarkResourceLock", RecordingLock)
    runner, calls = _mock_process_runner(payload)

    summary = module.run_plan(
        plan_path,
        expected_plan_id=payload["plan_id"],
        max_slots=12,
        process_runner=runner,
        process_table_provider=lambda: "harmless\n",
    )

    assert summary["status"] == "completed_full"
    assert summary["complete_twelve_slot_plan_requested"] is True
    assert len(summary["job_records"]) == 12
    assert len(summary["pair_audits"]) == 6
    assert len(summary["process_audits"]) == 12
    assert all(
        pair["only_difference_is_cut_chi_facility_vub"]
        for pair in summary["pair_audits"]
    )
    assert calls == [job["job_id"] for job in payload["planned_jobs"]]
    assert summary["all_open_job_ids_retained"] == [
        payload["planned_jobs"][0]["job_id"]
    ]
    assert not summary["skipped_planned_job_ids"]
    assert (run_root / "summary.json").is_file()
    assert RecordingLock.calls == [
        (
            Path(module.DEFAULT_BENCHMARK_RESOURCE_LOCK).resolve(),
            "h6-facility-incident-star-ablation",
        )
    ]
    assert all(record["infrastructure_valid"] for record in summary["job_records"])
    for record in summary["job_records"]:
        telemetry = record["facility_incident_star_cut_telemetry"]
        assert isinstance(telemetry, dict)
        if record["cut_chi_facility_vub"]:
            assert telemetry["root_user_cut_separation_seconds"] == pytest.approx(0.007)
        else:
            assert telemetry["root_user_cut_separation_seconds"] is None
    assert summary["development_comparison_eligible_pair_count"] == 5
    assert summary["all_completed_pairs_development_comparison_eligible"] is False
    assert summary["pair_audits"][0]["development_comparison_eligible"] is False
    assert all(
        pair["development_comparison_eligible"] for pair in summary["pair_audits"][1:]
    )
    assert all(
        set(record["artifact_file_sha256"])
        == {
            "run_config.json",
            "MX/summary.json",
            "driver_log",
            "MX/solver.log",
            "MX/incumbent.sol",
        }
        for record in summary["job_records"]
    )


def test_runner_refuses_process_conflict_before_creating_run_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _frozen_inputs, plan_path, run_root, payload = _build(tmp_path, monkeypatch)
    RecordingLock.calls.clear()
    monkeypatch.setattr(module, "BenchmarkResourceLock", RecordingLock)

    with pytest.raises(
        module.H6FacilityIncidentStarAblationError,
        match="conflicting solver/test processes",
    ):
        module.run_plan(
            plan_path,
            expected_plan_id=payload["plan_id"],
            max_slots=2,
            process_runner=lambda *_args, **_kwargs: None,
            process_table_provider=lambda: "999 1 /opt/python main.py solve --threads 4\n",
        )
    assert not run_root.exists()


def test_process_conflict_scan_excludes_self_and_all_ancestors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(module.os, "getpid", lambda: 300)
    table = "\n".join(
        (
            (
                "100 1 /bin/zsh -lc python -m "
                "src.benchmark.branch_cut.h6_facility_incident_star_ablation run"
            ),
            "200 100 /usr/bin/env benchmark-wrapper",
            "300 200 /opt/python -c child-process",
            "400 1 /opt/python harmless-worker.py",
        )
    )

    assert module._conflicting_processes(table) == ()


@pytest.mark.parametrize(
    "command",
    (
        "/opt/python /repo/main.py solve --threads 1",
        "/opt/python -m main run --source nanshan",
        "/opt/python -m src.benchmark.branch_cut run-formal --plan plan.json",
        "/opt/python -m src.benchmark.branch_cut.cli qualify --pool pool.json",
        "/usr/local/bin/branch_cut run --manifest manifest.json",
        (
            "/opt/python -m src.benchmark.branch_cut.calibration_runner "
            "--manifest manifest.json"
        ),
        "/opt/gurobi/bin/gurobi_cl model.lp",
        "/opt/python -m pytest -q",
    ),
)
def test_known_external_process_entrypoints_are_detected(
    command: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(module.os, "getpid", lambda: 900)
    external_line = f"401 1 {command}"
    table = f"900 1 /opt/python audit.py\n{external_line}\n"

    assert module._conflicting_processes(table) == (external_line,)


def test_process_scan_does_not_claim_to_detect_arbitrary_python_solvers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(module.os, "getpid", lambda: 900)
    table = (
        "900 1 /opt/python audit.py\n"
        "401 1 /opt/python /tmp/custom_solver.py --backend gurobi\n"
    )

    assert module._conflicting_processes(table) == ()


def test_baseline_incident_star_telemetry_contamination_fails_closed_and_is_retained(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _frozen_inputs, plan_path, _run_root, payload = _build(tmp_path, monkeypatch)
    monkeypatch.setattr(module, "BenchmarkResourceLock", RecordingLock)
    runner, _calls = _mock_process_runner(
        payload, contaminate_baseline_with_incident_star=True
    )

    summary = module.run_plan(
        plan_path,
        expected_plan_id=payload["plan_id"],
        max_slots=2,
        process_runner=runner,
        process_table_provider=lambda: "harmless\n",
    )

    assert summary["status"] == "failed_closed"
    contaminated = next(
        record for record in summary["job_records"] if record["variant"] == "C-current"
    )
    assert contaminated["infrastructure_valid"] is False
    assert any(
        "treatment.baseline_contains_incident_star_root_user_cut_separation_seconds"
        in error
        for error in contaminated["infrastructure_errors"]
    )
    assert contaminated["job_id"] in {
        record["job_id"] for record in summary["job_records"]
    }
    assert summary["pair_audits"] == []
    assert summary["development_comparison_eligible_pair_count"] == 0
    assert summary["all_completed_pairs_development_comparison_eligible"] is False


def test_recognized_no_incumbent_is_retained_without_infrastructure_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _frozen_inputs, plan_path, _run_root, payload = _build(tmp_path, monkeypatch)
    monkeypatch.setattr(module, "BenchmarkResourceLock", RecordingLock)
    runner, _calls = _mock_process_runner(payload, outcome_mode="no_incumbent")

    summary = module.run_plan(
        plan_path,
        expected_plan_id=payload["plan_id"],
        max_slots=2,
        process_runner=runner,
        process_table_provider=lambda: "harmless\n",
    )

    assert summary["status"] == "completed_partial"
    assert len(summary["job_records"]) == 2
    assert all(record["outcome"] == "no_incumbent" for record in summary["job_records"])
    assert all(record["infrastructure_valid"] for record in summary["job_records"])
    assert all(
        record["development_comparison_eligible"] is False
        for record in summary["job_records"]
    )
    assert set(summary["failed_job_ids_retained"]) == {
        job["job_id"] for job in payload["planned_jobs"][:2]
    }


def test_generic_nonzero_process_failure_fails_closed_and_is_retained(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _frozen_inputs, plan_path, _run_root, payload = _build(tmp_path, monkeypatch)
    monkeypatch.setattr(module, "BenchmarkResourceLock", RecordingLock)
    runner, _calls = _mock_process_runner(payload, outcome_mode="generic_failure")

    summary = module.run_plan(
        plan_path,
        expected_plan_id=payload["plan_id"],
        max_slots=2,
        process_runner=runner,
        process_table_provider=lambda: "harmless\n",
    )

    assert summary["status"] == "failed_closed"
    assert len(summary["job_records"]) == 1
    record = summary["job_records"][0]
    assert record["outcome"] == "process_failure"
    assert record["infrastructure_valid"] is False
    assert (
        "subprocess_failed_without_recognized_no_incumbent"
        in record["infrastructure_errors"]
    )
    assert summary["failed_job_ids_retained"] == [record["job_id"]]


def test_instance_file_hash_tamper_is_rejected_even_with_rehashed_signature(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _frozen_inputs, plan_path, _run_root, payload = _build(tmp_path, monkeypatch)
    monkeypatch.setattr(module, "BenchmarkResourceLock", RecordingLock)
    runner, _calls = _mock_process_runner(payload, tamper_instance_hash=True)

    summary = module.run_plan(
        plan_path,
        expected_plan_id=payload["plan_id"],
        max_slots=2,
        process_runner=runner,
        process_table_provider=lambda: "harmless\n",
    )

    assert summary["status"] == "failed_closed"
    assert summary["job_records"][0]["infrastructure_valid"] is False
    assert (
        "run_config.instance.files"
        in summary["job_records"][0]["infrastructure_errors"]
    )


def test_malformed_solver_config_is_checkpointed_instead_of_crashing_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _frozen_inputs, plan_path, run_root, payload = _build(tmp_path, monkeypatch)
    monkeypatch.setattr(module, "BenchmarkResourceLock", RecordingLock)
    runner, _calls = _mock_process_runner(payload, malformed_solver=True)

    summary = module.run_plan(
        plan_path,
        expected_plan_id=payload["plan_id"],
        max_slots=2,
        process_runner=runner,
        process_table_provider=lambda: "harmless\n",
    )

    assert summary["status"] == "failed_closed"
    assert (run_root / "summary.json").is_file()
    record = summary["job_records"][0]
    assert record["infrastructure_valid"] is False
    assert any(
        error.startswith("run_config.normalization:")
        for error in record["infrastructure_errors"]
    )


def test_cli_requires_expected_plan_identity_and_explicit_slot_cap() -> None:
    parser = module.build_cli_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["run", "--plan-path", "plan.json", "--max-slots", "12"])
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "run",
                "--plan-path",
                "plan.json",
                "--expected-plan-id",
                module.PLAN_ID_PREFIX + "a" * 64,
            ]
        )
    args = parser.parse_args(
        [
            "run",
            "--plan-path",
            "plan.json",
            "--expected-plan-id",
            module.PLAN_ID_PREFIX + "a" * 64,
            "--max-slots",
            "12",
        ]
    )
    assert args.max_slots == 12
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "plan",
                "--plan-path",
                "plan.json",
                "--run-root",
                "run",
            ]
        )


def test_main_cli_facility_incident_star_switch_preserves_default_and_allows_ablation() -> (
    None
):
    parser = module.build_main_parser()
    default = parser.parse_args(["solve", "--instance-dir", "/tmp/example"])
    disabled = parser.parse_args(
        [
            "solve",
            "--instance-dir",
            "/tmp/example",
            "--no-od-facility-vub-cuts",
        ]
    )
    explicit = parser.parse_args(
        [
            "solve",
            "--instance-dir",
            "/tmp/example",
            "--od-facility-vub-cuts",
        ]
    )
    assert default.od_facility_vub_cuts is False
    assert disabled.od_facility_vub_cuts is False
    assert explicit.od_facility_vub_cuts is True
