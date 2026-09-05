from __future__ import annotations

from contextlib import ExitStack
import copy
import json
import math
from pathlib import Path
import shutil
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from src.benchmark.branch_cut.calibration_builder import (
    development_calibration_profiles,
)
from src.benchmark.branch_cut.calibration_runner import CalibrationJob
from src.benchmark.branch_cut.case_builder import qualification_case_sha256
from src.benchmark.branch_cut.h23_qualification_diagnostic_contract import (
    FROZEN_MANIFEST_FILE_SHA256,
    FROZEN_MANIFEST_SHA256,
    FROZEN_SOURCE_CHECKPOINTS,
    H23_DIAGNOSTIC_MAX_JOBS,
    H23_DIAGNOSTIC_PLAN_ID,
    IMPLEMENTATION_FILES,
    LEGACY_IMPLEMENTATION_FILES,
)
from src.benchmark.branch_cut.hashing import canonical_json, canonical_sha256
from src.benchmark.branch_cut.qualification import (
    QualificationEvidence,
    assess_qualification,
    service_tolerance,
)
from src.benchmark.branch_cut.qualification_geometry_start import (
    GeometryStartAttemptAudit,
    GeometryStartPlanAudit,
)
from src.benchmark.branch_cut.qualification_runner import (
    QualificationModelAudit,
    QualificationPhaseTimings,
    QualificationRunnerOutcome,
)
import src.benchmark.branch_cut.h23_qualification_diagnostic as diagnostic
import src.benchmark.branch_cut.h23_qualification_diagnostic_verifier as verifier


_COMMIT = "a" * 40
_INSTANCE_SHA = "1" * 64
_FIXED_NOW = "2026-08-31T00:00:00Z"


class _FakeInstance:
    def __init__(self, config: object, demand_scale: float) -> None:
        self.config = config
        self.metadata = {"demand_scale": demand_scale}

    @staticmethod
    def screening_is_exact(mode: str) -> bool:
        assert mode == "MX"
        return False


def _fake_implementation() -> dict[str, object]:
    ordered = {name: f"{index + 1:064x}" for index, name in enumerate(IMPLEMENTATION_FILES)}
    return {
        "ordered_file_sha256": ordered,
        "implementation_sha256": canonical_sha256(ordered),
    }


def _fake_manifests(root: Path) -> tuple[object, object]:
    profiles = development_calibration_profiles()
    physical_id = "fake-h23-small-seed1729"
    runner_jobs: list[CalibrationJob] = []
    verifier_jobs: list[dict[str, object]] = []
    for index, (profile, frozen) in enumerate(
        zip(profiles, FROZEN_SOURCE_CHECKPOINTS, strict=True)
    ):
        solve = profile.solve_profile()
        job = CalibrationJob(
            job_order_index=index,
            job_id=frozen[0],
            profile_id=solve.profile_id,
            scale="small",
            scale_order_index=0,
            layout_seed=1729,
            development_seed_stage=0,
            profile_order_index=index,
            physical_instance_id=physical_id,
            instance_ref=f"instances/{physical_id}",
            instance_sha256=_INSTANCE_SHA,
            effective_model_config_sha256=profile.effective_model_config_sha256(),
            qualification_case_sha256=qualification_case_sha256(
                _INSTANCE_SHA, solve
            ),
        )
        runner_jobs.append(job)
        verifier_jobs.append(job.to_payload())
    manifest_path = root / "fake-manifest.json"
    raw_path = root / "fake-raw.csv"
    instance_dir = root / "fake-instance"
    runner_manifest = SimpleNamespace(
        path=manifest_path,
        root=root / "manifest-root",
        manifest_sha256=FROZEN_MANIFEST_SHA256,
        manifest_file_sha256=FROZEN_MANIFEST_FILE_SHA256,
        raw_path=raw_path,
        raw_sha256="2" * 64,
        raw_size_bytes=7,
        physical_instances=(),
        physical_directories={physical_id: instance_dir},
        profiles=profiles,
        jobs=tuple(runner_jobs),
    )
    verifier_manifest = SimpleNamespace(
        path=manifest_path,
        root=root / "manifest-root",
        payload={"raw_input": {"size_bytes": 7}},
        manifest_sha256=FROZEN_MANIFEST_SHA256,
        manifest_file_sha256=FROZEN_MANIFEST_FILE_SHA256,
        raw_path=raw_path,
        raw_sha256="2" * 64,
        physical_directories={physical_id: instance_dir},
        profiles=profiles,
        jobs=tuple(verifier_jobs),
    )
    return runner_manifest, verifier_manifest


def _fake_source(root: Path) -> tuple[dict[str, object], tuple[dict[str, str], ...]]:
    bindings = tuple(
        {
            "source_job_id": job_id,
            "relative_path": f"checkpoints/{filename}",
            "file_sha256": file_sha256,
            "checkpoint_sha256": checkpoint_sha256,
        }
        for job_id, filename, file_sha256, checkpoint_sha256 in FROZEN_SOURCE_CHECKPOINTS
    )
    payload: dict[str, object] = {
        "root": str(root / "source-run"),
        "calibration_run_id": diagnostic.FROZEN_SOURCE_RUN_ID,
        "git_commit": diagnostic.FROZEN_BASE_COMMIT,
        "implementation_sha256": diagnostic.FROZEN_SOURCE_IMPLEMENTATION_SHA256,
        "run_identity": {
            "relative_path": "run_identity.json",
            "run_identity_sha256": diagnostic.FROZEN_SOURCE_RUN_IDENTITY_SHA256,
            "file_sha256": diagnostic.FROZEN_SOURCE_RUN_IDENTITY_FILE_SHA256,
        },
        "batch_summary": {
            "relative_path": "batch_summary.json",
            "summary_sha256": diagnostic.FROZEN_SOURCE_SUMMARY_SHA256,
            "file_sha256": diagnostic.FROZEN_SOURCE_SUMMARY_FILE_SHA256,
        },
        "checkpoint_count": H23_DIAGNOSTIC_MAX_JOBS,
        "checkpoints": list(bindings),
    }
    return payload, bindings


def _environment(implementation: dict[str, object]) -> dict[str, object]:
    return {
        "git": {
            "repository_root": str(diagnostic._REPOSITORY_ROOT),
            "commit": _COMMIT,
            "worktree_dirty": False,
            "status_includes_untracked": True,
            "status_lines": [],
            "status_sha256": canonical_sha256(()),
        },
        "implementation": implementation,
        "python": {
            "implementation": "CPython",
            "version": "test",
            "executable": "test-python",
        },
        "platform": {
            "system": "test",
            "release": "test",
            "machine": "test",
            "processor": "test",
        },
        "gurobi_version": "test-version",
    }


def _audit(contract: object) -> QualificationModelAudit:
    return QualificationModelAudit(
        effective_model_config_sha256=contract.effective_model_config_sha256,
        demand_scale=contract.demand_scale,
        locker_fixed_cost_per_hour=contract.locker_fixed_cost_per_hour,
        display_origin_choices=contract.display_origin_choices,
        general_corridor_choices=contract.general_corridor_choices,
        screening_is_exact=False,
        model_fingerprint=7,
    )


def _rejected_outcome(**kwargs: object) -> QualificationRunnerOutcome:
    contract = kwargs["expected_model_contract"]
    policy = kwargs["policy"]
    geometry = kwargs["geometry_start_policy"]
    settings = kwargs["settings"]
    assert settings.time_limit_seconds == 60.0
    assert settings.threads == 1
    assert geometry.enabled is True
    assert geometry.small_support_schedule == (12, 8, 15)
    evidence = QualificationEvidence(
        case_id=str(kwargs["case_id"]),
        scale="small",
        courier_only_objective=1000.0,
        witness_objective=math.nan,
        total_demand_per_hour=100.0,
        witness_service_per_hour=math.nan,
        open_facilities=0,
        positive_cost_open_facilities=0,
        candidate_h=23,
        open_h=0,
        witness_feasible=False,
        qualification_run_id=str(kwargs["qualification_run_id"]),
        instance_sha256=str(kwargs["instance_sha256"]),
        active_h=0,
        active_h_threshold_per_hour=service_tolerance(100.0),
        qualification_case_sha256=contract.qualification_case_sha256,
        effective_model_config_sha256=contract.effective_model_config_sha256,
        screening_is_exact=False,
    )
    decision = assess_qualification(evidence, policy)
    attempts = tuple(
        GeometryStartAttemptAudit(
            plan=GeometryStartPlanAudit.planning_failure(
                plan_version=geometry.plan_version,
                target_support_size=support,
            ),
            per_attempt_seconds=geometry.per_attempt_seconds,
            start_node_limit=geometry.start_node_limit,
            error=(
                "geometry_start_planning: GeometryStartInfeasibilityError: "
                "fake topology precheck"
            ),
        )
        for support in geometry.small_support_schedule
    )
    return QualificationRunnerOutcome(
        accepted=False,
        rejection_reasons=(
            "no_positive_service_incumbent",
            "no_objective_improving_incumbent",
            *decision.rejection_reasons,
        ),
        evidence=evidence,
        decision=decision,
        required_service_per_hour=0.1,
        required_objective_upper_bound=999.0,
        temporary_constraint_used=True,
        temporary_constraint_removed=True,
        fresh_model_validation_performed=False,
        witness_status="TIME_LIMIT",
        validation_status="NOT_RUN",
        captured_variable_count=0,
        exception_messages=(),
        feasibility_objective_used=True,
        expected_model_contract=contract,
        witness_model_audit=_audit(contract),
        fresh_model_audit=None,
        geometry_start_policy=geometry,
        geometry_start_attempts=attempts,
        geometry_start_selected_plan_sha256=None,
        phase_timings=QualificationPhaseTimings(),
    )


def _accepted_outcome(**kwargs: object) -> QualificationRunnerOutcome:
    contract = kwargs["expected_model_contract"]
    policy = kwargs["policy"]
    geometry = kwargs["geometry_start_policy"]
    settings = kwargs["settings"]
    assert settings.time_limit_seconds == 60.0
    evidence = QualificationEvidence(
        case_id=str(kwargs["case_id"]),
        scale="small",
        courier_only_objective=1000.0,
        witness_objective=900.0,
        total_demand_per_hour=100.0,
        witness_service_per_hour=50.0,
        open_facilities=5,
        positive_cost_open_facilities=5,
        candidate_h=23,
        open_h=5,
        witness_feasible=True,
        qualification_run_id=str(kwargs["qualification_run_id"]),
        instance_sha256=str(kwargs["instance_sha256"]),
        active_h=5,
        active_h_threshold_per_hour=service_tolerance(100.0),
        qualification_case_sha256=contract.qualification_case_sha256,
        effective_model_config_sha256=contract.effective_model_config_sha256,
        screening_is_exact=False,
    )
    decision = assess_qualification(evidence, policy)
    assert decision.accepted
    audit = _audit(contract)
    return QualificationRunnerOutcome(
        accepted=True,
        rejection_reasons=(),
        evidence=evidence,
        decision=decision,
        required_service_per_hour=0.1,
        required_objective_upper_bound=999.0,
        temporary_constraint_used=False,
        temporary_constraint_removed=None,
        fresh_model_validation_performed=True,
        witness_status="TIME_LIMIT",
        validation_status="OPTIMAL",
        captured_variable_count=10,
        exception_messages=(),
        feasibility_objective_used=False,
        expected_model_contract=contract,
        witness_model_audit=audit,
        fresh_model_audit=audit,
        geometry_start_policy=geometry,
        geometry_start_attempts=(),
        geometry_start_selected_plan_sha256=None,
        phase_timings=QualificationPhaseTimings(),
    )


def _dependency_patches(root: Path) -> tuple[ExitStack, object, object, dict[str, object], tuple[dict[str, str], ...]]:
    runner_manifest, verifier_manifest = _fake_manifests(root)
    source_payload, bindings = _fake_source(root)
    stack = ExitStack()
    stack.enter_context(
        patch.object(
            diagnostic,
            "load_development_calibration_manifest",
            return_value=runner_manifest,
        )
    )
    stack.enter_context(
        patch.object(
            diagnostic,
            "_validate_source_run",
            return_value=(source_payload, bindings),
        )
    )
    stack.enter_context(
        patch.object(diagnostic, "_revalidate_job_inputs", return_value=None)
    )
    stack.enter_context(
        patch.object(
            verifier.source_verifier,
            "_verify_manifest",
            return_value=verifier_manifest,
        )
    )
    stack.enter_context(
        patch.object(
            verifier,
            "_verify_source_run",
            return_value=(source_payload, bindings),
        )
    )
    return stack, runner_manifest, verifier_manifest, source_payload, bindings


def _build_artifacts(
    root: Path,
    *,
    qualifier: object,
) -> tuple[object, Path, dict[str, object]]:
    implementation = _fake_implementation()
    with _dependency_patches(root)[0]:
        state = {
            "repository_root": str(diagnostic._REPOSITORY_ROOT),
            "git_commit": _COMMIT,
            "status_includes_untracked": True,
            "status_lines": [],
            "worktree_dirty": False,
            "implementation": implementation,
        }
        plan = diagnostic.build_h23_diagnostic_plan(
            diagnostic.H23PlanBuildConfig(
                manifest_path=root / "fake-manifest.json",
                source_run_root=root / "source-run",
                output_path=root / "plan.json",
                plan_id=H23_DIAGNOSTIC_PLAN_ID,
                max_jobs=H23_DIAGNOSTIC_MAX_JOBS,
                expected_git_commit=_COMMIT,
            ),
            hooks=diagnostic.H23PlanBuildHooks(
                capture_repository_state=lambda _root: state
            ),
        )
        hooks = diagnostic.H23DiagnosticHooks(
            load_instance=lambda _path, config, demand_scale: _FakeInstance(
                config, demand_scale
            ),
            qualify_instance=qualifier,
            capture_environment=lambda _root: _environment(implementation),
            process_table=lambda: "1 0 launchd\n",
            now_utc=lambda: _FIXED_NOW,
        )
        result_root = root / "result"
        result = diagnostic.run_h23_diagnostic(
            diagnostic.H23DiagnosticRunConfig(
                plan_path=plan.path,
                results_root=result_root,
                plan_id=H23_DIAGNOSTIC_PLAN_ID,
                max_jobs=H23_DIAGNOSTIC_MAX_JOBS,
                resource_lock=root / "resource.lock",
            ),
            hooks=hooks,
        )
    return plan, result.results_root, implementation


def _verify(
    root: Path,
    *,
    plan_path: Path,
    results_root: Path,
    run_id: str,
    implementation: dict[str, object],
) -> verifier.H23DiagnosticVerificationReport:
    with _dependency_patches(root)[0]:
        return verifier.verify_h23_diagnostic_results(
            verifier.H23DiagnosticVerificationRequest(
                plan_path=plan_path,
                results_root=results_root,
                expected_plan_id=H23_DIAGNOSTIC_PLAN_ID,
                expected_max_jobs=H23_DIAGNOSTIC_MAX_JOBS,
                expected_run_id=run_id,
                expected_git_commit=_COMMIT,
                expected_implementation_sha256=str(
                    implementation["implementation_sha256"]
                ),
            )
        )


def _run_id(results_root: Path) -> str:
    return json.loads((results_root / "run_identity.json").read_text())[
        "diagnostic_run_id"
    ]


def _write_canonical(path: Path, payload: object) -> None:
    path.write_text(canonical_json(payload) + "\n", encoding="utf-8")


def _rehash_plan(payload: dict[str, object]) -> None:
    payload["plan_sha256"] = canonical_sha256(
        {
            key: value
            for key, value in payload.items()
            if key
            not in {"artifact_type", "artifact_schema_version", "plan_sha256"}
        }
    )


def _rehash_summary(payload: dict[str, object]) -> None:
    payload["summary_sha256"] = canonical_sha256(
        {
            key: value
            for key, value in payload.items()
            if key
            not in {"artifact_type", "artifact_schema_version", "summary_sha256"}
        }
    )


def _relocate_result(source: Path, target: Path) -> Path:
    shutil.copytree(source, target)
    summary_path = target / "batch_summary.json"
    if not summary_path.exists():
        summary_path = target / "batch_aborted.json"
    summary = json.loads(summary_path.read_text())
    for record in summary["checkpoints"]:
        record["path"] = str(target / "checkpoints" / Path(record["path"]).name)
    for record in summary["diagnostic_job_ledger"]:
        if record["checkpoint_path"] is not None:
            record["checkpoint_path"] = str(
                target / "checkpoints" / Path(record["checkpoint_path"]).name
            )
    _rehash_summary(summary)
    _write_canonical(summary_path, summary)
    return target


def test_full_24_normal_rejections_continue_and_verify(tmp_path: Path) -> None:
    plan, result_root, implementation = _build_artifacts(
        tmp_path, qualifier=lambda _instance, _mode, **kwargs: _rejected_outcome(**kwargs)
    )
    report = _verify(
        tmp_path,
        plan_path=plan.path,
        results_root=result_root,
        run_id=_run_id(result_root),
        implementation=implementation,
    )
    assert report.status == "completed"
    assert report.attempted_job_count == 24
    assert report.accepted_job_count == 0
    assert report.rejected_attempted_job_count == 24


@pytest.mark.parametrize("registered", (True, False))
def test_historical_plan_source_set_requires_a_registered_complete_set(
    tmp_path: Path, registered: bool
) -> None:
    source_files = (
        LEGACY_IMPLEMENTATION_FILES
        if registered
        else LEGACY_IMPLEMENTATION_FILES[:-1]
    )
    files = {name: f"{index + 1:064x}" for index, name in enumerate(source_files)}
    implementation = {
        "ordered_file_sha256": files,
        "implementation_sha256": canonical_sha256(files),
    }
    # Simulate an artifact produced by its preserved historical source set.
    # Verification runs with the current verifier and its unchanged explicit pins.
    with (
        patch.object(diagnostic, "IMPLEMENTATION_FILES", source_files),
        patch(f"{__name__}._fake_implementation", return_value=implementation),
    ):
        plan, result_root, _ = _build_artifacts(
            tmp_path,
            qualifier=lambda _instance, _mode, **kwargs: _rejected_outcome(**kwargs),
        )
    if registered:
        report = _verify(
            tmp_path,
            plan_path=plan.path,
            results_root=result_root,
            run_id=_run_id(result_root),
            implementation=implementation,
        )
        assert report.status == "completed"
    else:
        with pytest.raises(verifier.H23DiagnosticVerificationError, match="file set"):
            _verify(
                tmp_path,
                plan_path=plan.path,
                results_root=result_root,
                run_id=_run_id(result_root),
                implementation=implementation,
            )


def test_full_24_fresh_validated_acceptances_verify(tmp_path: Path) -> None:
    plan, result_root, implementation = _build_artifacts(
        tmp_path, qualifier=lambda _instance, _mode, **kwargs: _accepted_outcome(**kwargs)
    )
    report = _verify(
        tmp_path,
        plan_path=plan.path,
        results_root=result_root,
        run_id=_run_id(result_root),
        implementation=implementation,
    )
    assert report.status == "completed"
    assert report.attempted_job_count == 24
    assert report.accepted_job_count == 24
    summary = json.loads((result_root / "batch_summary.json").read_text())
    assert summary["may_lock_profiles"] is False
    assert summary["locked_profile_ids"] == []
    assert summary["accepted_profile_order_indices"] == list(range(24))


def test_infrastructure_failure_aborts_and_verifies_prefix(tmp_path: Path) -> None:
    implementation = _fake_implementation()
    with _dependency_patches(tmp_path)[0]:
        state = {
            "repository_root": str(diagnostic._REPOSITORY_ROOT),
            "git_commit": _COMMIT,
            "status_includes_untracked": True,
            "status_lines": [],
            "worktree_dirty": False,
            "implementation": implementation,
        }
        plan = diagnostic.build_h23_diagnostic_plan(
            diagnostic.H23PlanBuildConfig(
                tmp_path / "fake-manifest.json",
                tmp_path / "source-run",
                tmp_path / "plan.json",
                H23_DIAGNOSTIC_PLAN_ID,
                24,
                _COMMIT,
            ),
            hooks=diagnostic.H23PlanBuildHooks(
                capture_repository_state=lambda _root: state
            ),
        )

        def fail(_instance: object, _mode: str, **_kwargs: object) -> object:
            raise RuntimeError("synthetic infrastructure failure")

        with pytest.raises(diagnostic.H23DiagnosticBatchInfrastructureError):
            diagnostic.run_h23_diagnostic(
                diagnostic.H23DiagnosticRunConfig(
                    plan.path,
                    tmp_path / "result",
                    H23_DIAGNOSTIC_PLAN_ID,
                    24,
                    tmp_path / "resource.lock",
                ),
                hooks=diagnostic.H23DiagnosticHooks(
                    load_instance=lambda _path, config, scale: _FakeInstance(
                        config, scale
                    ),
                    qualify_instance=fail,
                    capture_environment=lambda _root: _environment(implementation),
                    process_table=lambda: "1 0 launchd\n",
                    now_utc=lambda: _FIXED_NOW,
                ),
            )
    result_root = tmp_path / "result"
    report = _verify(
        tmp_path,
        plan_path=plan.path,
        results_root=result_root,
        run_id=_run_id(result_root),
        implementation=implementation,
    )
    assert report.status == "aborted_on_infrastructure_failure"
    assert report.attempted_job_count == 1
    assert report.infrastructure_failure_count == 1
    assert report.unattempted_job_count == 23


def test_initial_process_conflict_fails_before_claiming_results(tmp_path: Path) -> None:
    implementation = _fake_implementation()
    with _dependency_patches(tmp_path)[0]:
        state = {
            "repository_root": str(diagnostic._REPOSITORY_ROOT),
            "git_commit": _COMMIT,
            "status_includes_untracked": True,
            "status_lines": [],
            "worktree_dirty": False,
            "implementation": implementation,
        }
        plan = diagnostic.build_h23_diagnostic_plan(
            diagnostic.H23PlanBuildConfig(
                tmp_path / "fake-manifest.json",
                tmp_path / "source-run",
                tmp_path / "plan.json",
                H23_DIAGNOSTIC_PLAN_ID,
                24,
                _COMMIT,
            ),
            hooks=diagnostic.H23PlanBuildHooks(
                capture_repository_state=lambda _root: state
            ),
        )
        with pytest.raises(diagnostic.H23DiagnosticError, match="conflicting"):
            diagnostic.run_h23_diagnostic(
                diagnostic.H23DiagnosticRunConfig(
                    plan.path,
                    tmp_path / "result",
                    H23_DIAGNOSTIC_PLAN_ID,
                    24,
                    tmp_path / "resource.lock",
                ),
                hooks=diagnostic.H23DiagnosticHooks(
                    capture_environment=lambda _root: _environment(implementation),
                    process_table=lambda: "999 1 main.py solve --mode MX\n",
                ),
            )
    assert not (tmp_path / "result").exists()


@pytest.mark.parametrize("drift", ["native", "geometry", "profile", "source_hash"])
def test_plan_drift_is_rejected(tmp_path: Path, drift: str) -> None:
    base = tmp_path / "base"
    plan, result_root, implementation = _build_artifacts(
        base, qualifier=lambda _instance, _mode, **kwargs: _rejected_outcome(**kwargs)
    )
    tampered_path = tmp_path / f"tampered-{drift}.json"
    payload = json.loads(plan.path.read_text())
    if drift == "native":
        payload["native_settings"]["time_limit_seconds"] = 31.0
    elif drift == "geometry":
        payload["geometry_start_policy"]["small_support_schedule"] = [8, 12, 15]
    elif drift == "profile":
        payload["jobs"][0]["source_job"]["profile_id"] = "tampered-profile"
    else:
        payload["source_5s_run"]["checkpoints"][0]["file_sha256"] = "f" * 64
    _rehash_plan(payload)
    _write_canonical(tampered_path, payload)
    with pytest.raises(verifier.H23DiagnosticVerificationError):
        _verify(
            base,
            plan_path=tampered_path,
            results_root=result_root,
            run_id=_run_id(result_root),
            implementation=implementation,
        )


@pytest.mark.parametrize("tamper", ["missing", "extra", "duplicate", "out_of_order"])
def test_checkpoint_set_and_order_tampering_is_rejected(
    tmp_path: Path, tamper: str
) -> None:
    base = tmp_path / "base"
    plan, result_root, implementation = _build_artifacts(
        base, qualifier=lambda _instance, _mode, **kwargs: _rejected_outcome(**kwargs)
    )
    target = _relocate_result(result_root, tmp_path / "tampered-result")
    summary_path = target / "batch_summary.json"
    summary = json.loads(summary_path.read_text())
    if tamper == "missing":
        Path(summary["checkpoints"][0]["path"]).unlink()
    elif tamper == "extra":
        _write_canonical(target / "checkpoints" / "extra.json", {"extra": True})
    elif tamper == "duplicate":
        summary["checkpoints"][1] = copy.deepcopy(summary["checkpoints"][0])
        _rehash_summary(summary)
        _write_canonical(summary_path, summary)
    else:
        summary["checkpoints"][0], summary["checkpoints"][1] = (
            summary["checkpoints"][1],
            summary["checkpoints"][0],
        )
        _rehash_summary(summary)
        _write_canonical(summary_path, summary)
    with pytest.raises(verifier.H23DiagnosticVerificationError):
        _verify(
            base,
            plan_path=plan.path,
            results_root=target,
            run_id=_run_id(target),
            implementation=implementation,
        )


def test_symlinked_checkpoint_is_rejected(tmp_path: Path) -> None:
    base = tmp_path / "base"
    plan, result_root, implementation = _build_artifacts(
        base, qualifier=lambda _instance, _mode, **kwargs: _rejected_outcome(**kwargs)
    )
    target = _relocate_result(result_root, tmp_path / "tampered-result")
    summary = json.loads((target / "batch_summary.json").read_text())
    checkpoint = Path(summary["checkpoints"][0]["path"])
    outside = tmp_path / "outside-checkpoint.json"
    shutil.copy2(checkpoint, outside)
    checkpoint.unlink()
    checkpoint.symlink_to(outside)
    with pytest.raises(verifier.H23DiagnosticVerificationError, match="symbolic-link"):
        _verify(
            base,
            plan_path=plan.path,
            results_root=target,
            run_id=_run_id(target),
            implementation=implementation,
        )


def test_fresh_validation_inconsistency_is_rejected(tmp_path: Path) -> None:
    base = tmp_path / "base"
    plan, result_root, implementation = _build_artifacts(
        base, qualifier=lambda _instance, _mode, **kwargs: _rejected_outcome(**kwargs)
    )
    target = _relocate_result(result_root, tmp_path / "tampered-result")
    summary_path = target / "batch_summary.json"
    summary = json.loads(summary_path.read_text())
    checkpoint_path = Path(summary["checkpoints"][0]["path"])
    checkpoint = json.loads(checkpoint_path.read_text())
    checkpoint["result"]["outcome"]["fresh_model_validation_performed"] = True
    checkpoint["checkpoint_sha256"] = canonical_sha256(
        {"identity": checkpoint["identity"], "result": checkpoint["result"]}
    )
    _write_canonical(checkpoint_path, checkpoint)
    summary["checkpoints"][0]["checkpoint_sha256"] = checkpoint[
        "checkpoint_sha256"
    ]
    summary["diagnostic_job_ledger"][0]["checkpoint_sha256"] = checkpoint[
        "checkpoint_sha256"
    ]
    summary["diagnostic_job_ledger"][0]["outcome"] = checkpoint["result"][
        "outcome"
    ]
    _rehash_summary(summary)
    _write_canonical(summary_path, summary)
    with pytest.raises(verifier.H23DiagnosticVerificationError, match="fresh"):
        _verify(
            base,
            plan_path=plan.path,
            results_root=target,
            run_id=_run_id(target),
            implementation=implementation,
        )


def test_cli_requires_explicit_plan_id_and_fixed_24() -> None:
    parser = diagnostic.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["plan"])
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "plan",
                "--plan-id",
                H23_DIAGNOSTIC_PLAN_ID,
                "--max-jobs",
                "23",
                "--manifest",
                "m",
                "--source-run-root",
                "s",
                "--output",
                "o",
                "--expected-git-commit",
                _COMMIT,
            ]
        )
    verify_parser = verifier.build_parser()
    with pytest.raises(SystemExit):
        verify_parser.parse_args([])
