"""Frozen H23 qualification diagnostic plan and serial execution runner.

The diagnostic is development evidence only.  It re-runs all 24 profiles from
the frozen small/seed-stage-0 calibration slice in manifest order.  Relative to
the completed 30-second diagnostic, version 2 changes only the per-phase native
qualification budget to sixty seconds and each registered cost-blind
geometry-start attempt to thirty seconds.  It cannot select profiles, lock
profiles, create a formal pool, or run an A--E comparison.

The ``plan`` subcommand is solver-free.  The ``run`` subcommand imports Gurobi
only through the existing qualification adapter after every frozen input,
process, resource-lock, clean-worktree, commit, and implementation check passes.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any

from src.config import ModelConfig, to_dict

from .artifacts import write_canonical_json_exclusive
from .calibration_runner import (
    CalibrationJob,
    LoadedDevelopmentCalibrationManifest,
    load_development_calibration_manifest,
)
from .calibration_verifier import (
    CalibrationVerificationRequest,
    verify_development_calibration_results,
)
from .case_builder import canonical_instance_hash
from .h23_qualification_diagnostic_contract import (
    FROZEN_BASE_COMMIT,
    FROZEN_MANIFEST_FILE_SHA256,
    FROZEN_MANIFEST_SHA256,
    FROZEN_SOURCE_CHECKPOINTS,
    FROZEN_SOURCE_IMPLEMENTATION_SHA256,
    FROZEN_SOURCE_RUN_ID,
    FROZEN_SOURCE_RUN_IDENTITY_FILE_SHA256,
    FROZEN_SOURCE_RUN_IDENTITY_SHA256,
    FROZEN_SOURCE_SUMMARY_FILE_SHA256,
    FROZEN_SOURCE_SUMMARY_SHA256,
    H23_DIAGNOSTIC_CHECKPOINT_SCHEMA,
    H23_DIAGNOSTIC_MAX_JOBS,
    H23_DIAGNOSTIC_PLAN_ID,
    H23_DIAGNOSTIC_PLAN_SCHEMA,
    H23_DIAGNOSTIC_RUN_SCHEMA,
    H23_DIAGNOSTIC_SUMMARY_SCHEMA,
    IMPLEMENTATION_FILES,
    REGISTERED_GEOMETRY_POLICY_PAYLOAD,
    REGISTERED_NATIVE_SETTINGS_PAYLOAD,
    REGISTERED_POLICY_PAYLOAD,
)
from .hashing import canonical_json, canonical_sha256
from .qualification import QualificationPolicy, assess_qualification
from .qualification_geometry_start import GeometryStartPolicy
from .qualification_runner import (
    NORMAL_QUALIFICATION_RUNNER_REASONS,
    NORMAL_QUALIFICATION_SOLVER_STATUSES,
    NativeQualificationSettings,
    QualificationInfrastructureError,
    QualificationModelAudit,
    QualificationModelContract,
    QualificationRunnerOutcome,
    qualify_canonical_instance,
)
from .resource_lock import (
    DEFAULT_BENCHMARK_RESOURCE_LOCK,
    BenchmarkResourceLock,
    BenchmarkResourceLockError,
)


_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
_RUN_ID_PREFIX = "h23-qualification-diagnostic-run-"
_PLAN_ARTIFACT_TYPE = "mlnd_branch_cut_h23_qualification_diagnostic_plan"
_RUN_ARTIFACT_TYPE = "mlnd_branch_cut_h23_qualification_diagnostic_run_identity"
_CHECKPOINT_ARTIFACT_TYPE = (
    "mlnd_branch_cut_h23_qualification_diagnostic_checkpoint"
)
_SUMMARY_ARTIFACT_TYPE = (
    "mlnd_branch_cut_h23_qualification_diagnostic_batch_summary"
)


class H23DiagnosticError(RuntimeError):
    """A frozen plan, input, environment, or request failed validation."""


class H23DiagnosticBatchInfrastructureError(RuntimeError):
    """One infrastructure failure stopped the serial diagnostic batch."""


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_exact_keys(
    payload: Mapping[str, Any], expected: set[str], *, field_name: str
) -> None:
    if set(payload) != expected:
        missing = sorted(expected.difference(payload))
        extra = sorted(set(payload).difference(expected))
        raise H23DiagnosticError(
            f"{field_name} fields differ; missing={missing}, extra={extra}"
        )


def _require_mapping(value: object, *, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise H23DiagnosticError(f"{field_name} must be an object with string keys")
    return value


def _require_sha256(value: object, *, field_name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise H23DiagnosticError(f"{field_name} must be a lowercase SHA256 digest")
    return value


def _require_real_path(path: str | Path, *, field_name: str, kind: str) -> Path:
    absolute = Path(path).expanduser().absolute()
    try:
        resolved = absolute.resolve(strict=True)
    except OSError as error:
        raise H23DiagnosticError(
            f"{field_name} does not resolve: {absolute}: {error}"
        ) from error
    if absolute != resolved or absolute.is_symlink():
        raise H23DiagnosticError(f"{field_name} must be a real non-symlink path")
    if kind == "file" and not resolved.is_file():
        raise H23DiagnosticError(f"{field_name} must be a regular file")
    if kind == "directory" and not resolved.is_dir():
        raise H23DiagnosticError(f"{field_name} must be a directory")
    return resolved


def _read_canonical_object(path: Path, *, field_name: str) -> dict[str, Any]:
    try:
        raw = path.read_text(encoding="utf-8")
        payload = json.loads(raw)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise H23DiagnosticError(f"cannot read {field_name}: {error}") from error
    if not isinstance(payload, dict):
        raise H23DiagnosticError(f"{field_name} must contain one JSON object")
    try:
        expected = canonical_json(payload) + "\n"
    except (TypeError, ValueError) as error:
        raise H23DiagnosticError(f"{field_name} is not JSON-safe: {error}") from error
    if raw != expected:
        raise H23DiagnosticError(f"{field_name} is not canonical JSON")
    return payload


def _git(repository_root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ("git", *arguments),
        cwd=repository_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        message = (completed.stderr or completed.stdout).strip()
        raise H23DiagnosticError(f"git {' '.join(arguments)} failed: {message}")
    return completed.stdout.strip()


def _implementation_payload(repository_root: Path) -> dict[str, Any]:
    ordered = {
        relative: _sha256_file(repository_root / relative)
        for relative in IMPLEMENTATION_FILES
    }
    return {
        "ordered_file_sha256": ordered,
        "implementation_sha256": canonical_sha256(ordered),
    }


def _capture_plan_repository_state(repository_root: Path) -> Mapping[str, Any]:
    status_lines = tuple(
        line
        for line in _git(
            repository_root, "status", "--porcelain", "--untracked-files=all"
        ).splitlines()
        if line
    )
    commit = _git(repository_root, "rev-parse", "HEAD")
    try:
        _git(
            repository_root,
            "merge-base",
            "--is-ancestor",
            FROZEN_BASE_COMMIT,
            commit,
        )
    except H23DiagnosticError as error:
        raise H23DiagnosticError(
            f"diagnostic implementation must descend from {FROZEN_BASE_COMMIT}"
        ) from error
    return {
        "repository_root": str(repository_root),
        "git_commit": commit,
        "status_includes_untracked": True,
        "status_lines": list(status_lines),
        "worktree_dirty": bool(status_lines),
        "implementation": _implementation_payload(repository_root),
    }


@dataclass(frozen=True, slots=True)
class H23PlanBuildHooks:
    capture_repository_state: Callable[[Path], Mapping[str, Any]] = (
        _capture_plan_repository_state
    )


DEFAULT_PLAN_HOOKS = H23PlanBuildHooks()


@dataclass(frozen=True, slots=True)
class H23PlanBuildConfig:
    manifest_path: Path
    source_run_root: Path
    output_path: Path
    plan_id: str
    max_jobs: int
    expected_git_commit: str

    def __post_init__(self) -> None:
        if self.plan_id != H23_DIAGNOSTIC_PLAN_ID:
            raise ValueError(f"plan_id must be {H23_DIAGNOSTIC_PLAN_ID}")
        if self.max_jobs != H23_DIAGNOSTIC_MAX_JOBS or isinstance(
            self.max_jobs, bool
        ):
            raise ValueError("max_jobs is frozen at 24")
        if (
            not isinstance(self.expected_git_commit, str)
            or len(self.expected_git_commit) != 40
            or any(
                character not in "0123456789abcdef"
                for character in self.expected_git_commit
            )
        ):
            raise ValueError("expected_git_commit must be a lowercase Git digest")


@dataclass(frozen=True, slots=True)
class H23PlanBuildResult:
    path: Path
    plan_id: str
    plan_sha256: str
    plan_file_sha256: str
    job_count: int
    git_commit: str
    implementation_sha256: str

    def to_payload(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "plan_id": self.plan_id,
            "plan_sha256": self.plan_sha256,
            "plan_file_sha256": self.plan_file_sha256,
            "job_count": self.job_count,
            "git_commit": self.git_commit,
            "implementation_sha256": self.implementation_sha256,
        }


def _validate_source_run(
    manifest_path: Path, source_run_root: Path
) -> tuple[dict[str, Any], tuple[dict[str, Any], ...]]:
    manifest = _require_real_path(
        manifest_path, field_name="manifest_path", kind="file"
    )
    root = _require_real_path(
        source_run_root, field_name="source_run_root", kind="directory"
    )
    report = verify_development_calibration_results(
        CalibrationVerificationRequest(
            manifest_path=manifest,
            results_root=root,
            expected_run_id=FROZEN_SOURCE_RUN_ID,
            expected_manifest_sha256=FROZEN_MANIFEST_SHA256,
            expected_git_commit=FROZEN_BASE_COMMIT,
            expected_implementation_sha256=FROZEN_SOURCE_IMPLEMENTATION_SHA256,
        )
    )
    if (
        report.status != "completed"
        or report.selected_job_count != H23_DIAGNOSTIC_MAX_JOBS
        or report.attempted_job_count != H23_DIAGNOSTIC_MAX_JOBS
        or report.accepted_job_count != 0
        or report.infrastructure_failure_count != 0
    ):
        raise H23DiagnosticError("frozen 5-second source run report drifted")
    identity_path = _require_real_path(
        root / "run_identity.json", field_name="source run identity", kind="file"
    )
    summary_path = _require_real_path(
        root / "batch_summary.json", field_name="source batch summary", kind="file"
    )
    if _sha256_file(identity_path) != FROZEN_SOURCE_RUN_IDENTITY_FILE_SHA256:
        raise H23DiagnosticError("source run_identity.json file hash drifted")
    if _sha256_file(summary_path) != FROZEN_SOURCE_SUMMARY_FILE_SHA256:
        raise H23DiagnosticError("source batch_summary.json file hash drifted")
    identity = _read_canonical_object(identity_path, field_name="source run identity")
    summary = _read_canonical_object(summary_path, field_name="source batch summary")
    if (
        identity.get("calibration_run_id") != FROZEN_SOURCE_RUN_ID
        or identity.get("run_identity_sha256")
        != FROZEN_SOURCE_RUN_IDENTITY_SHA256
        or summary.get("summary_sha256") != FROZEN_SOURCE_SUMMARY_SHA256
    ):
        raise H23DiagnosticError("source run canonical identity drifted")
    raw_records = summary.get("checkpoints")
    if not isinstance(raw_records, list) or len(raw_records) != H23_DIAGNOSTIC_MAX_JOBS:
        raise H23DiagnosticError("source summary must bind exactly 24 checkpoints")
    bindings: list[dict[str, Any]] = []
    checkpoints_root = _require_real_path(
        root / "checkpoints", field_name="source checkpoints", kind="directory"
    )
    observed: set[Path] = set()
    for index, (frozen, record) in enumerate(
        zip(FROZEN_SOURCE_CHECKPOINTS, raw_records, strict=True)
    ):
        job_id, filename, file_sha256, checkpoint_sha256 = frozen
        if not isinstance(record, Mapping):
            raise H23DiagnosticError("source checkpoint record must be an object")
        path = _require_real_path(
            checkpoints_root / filename,
            field_name=f"source checkpoint {index}",
            kind="file",
        )
        if path.parent != checkpoints_root or path in observed:
            raise H23DiagnosticError("source checkpoint path escaped or duplicated")
        observed.add(path)
        if (
            record.get("job_id") != job_id
            or record.get("job_order_index") != index
            or record.get("execution_attempt_index") != index
            or record.get("checkpoint_sha256") != checkpoint_sha256
            or record.get("status") != "completed"
            or record.get("accepted") is not False
            or record.get("path") != str(path)
            or _sha256_file(path) != file_sha256
        ):
            raise H23DiagnosticError(f"source checkpoint binding drifted at {index}")
        checkpoint = _read_canonical_object(
            path, field_name=f"source checkpoint {index}"
        )
        if checkpoint.get("checkpoint_sha256") != checkpoint_sha256:
            raise H23DiagnosticError(f"source checkpoint self-hash drifted at {index}")
        bindings.append(
            {
                "source_job_id": job_id,
                "relative_path": f"checkpoints/{filename}",
                "file_sha256": file_sha256,
                "checkpoint_sha256": checkpoint_sha256,
            }
        )
    directory_files = {
        path.resolve(strict=True) for path in checkpoints_root.iterdir() if path.is_file()
    }
    if directory_files != observed:
        raise H23DiagnosticError("source checkpoint directory has missing/extra files")
    source_payload = {
        "root": str(root),
        "calibration_run_id": FROZEN_SOURCE_RUN_ID,
        "git_commit": FROZEN_BASE_COMMIT,
        "implementation_sha256": FROZEN_SOURCE_IMPLEMENTATION_SHA256,
        "run_identity": {
            "relative_path": "run_identity.json",
            "run_identity_sha256": FROZEN_SOURCE_RUN_IDENTITY_SHA256,
            "file_sha256": FROZEN_SOURCE_RUN_IDENTITY_FILE_SHA256,
        },
        "batch_summary": {
            "relative_path": "batch_summary.json",
            "summary_sha256": FROZEN_SOURCE_SUMMARY_SHA256,
            "file_sha256": FROZEN_SOURCE_SUMMARY_FILE_SHA256,
        },
        "checkpoint_count": H23_DIAGNOSTIC_MAX_JOBS,
        "checkpoints": bindings,
    }
    return source_payload, tuple(bindings)


def _registered_policy() -> QualificationPolicy:
    payload = REGISTERED_POLICY_PAYLOAD
    return QualificationPolicy(
        optimality_tolerance=payload["optimality_tolerance"],
        minimum_savings_fraction=payload["minimum_savings_fraction"],
        minimum_service_fraction=payload["minimum_service_fraction"],
        minimum_h_open_fraction=payload["minimum_h_open_fraction"],
        maximum_h_open_fraction=payload["maximum_h_open_fraction"],
        large_scale_open_h_target=tuple(payload["large_scale_open_h_target"]),
    )


def _registered_native_settings() -> NativeQualificationSettings:
    return NativeQualificationSettings(**REGISTERED_NATIVE_SETTINGS_PAYLOAD)


def _registered_geometry_policy() -> GeometryStartPolicy:
    payload = REGISTERED_GEOMETRY_POLICY_PAYLOAD
    return GeometryStartPolicy(
        enabled=payload["enabled"],
        per_attempt_seconds=payload["per_attempt_seconds"],
        start_node_limit=payload["start_node_limit"],
        plan_version=payload["plan_version"],
        active_tolerance_multiplier=payload["active_tolerance_multiplier"],
        required_service_multiplier=payload["required_service_multiplier"],
        small_support_schedule=tuple(payload["small_support_schedule"]),
        medium_support_schedule=tuple(payload["medium_support_schedule"]),
        large_support_schedule=tuple(payload["large_support_schedule"]),
    )


def _source_jobs(
    manifest: LoadedDevelopmentCalibrationManifest,
) -> tuple[CalibrationJob, ...]:
    jobs = tuple(
        job
        for job in manifest.jobs
        if job.scale == "small" and job.development_seed_stage == 0
    )
    if (
        len(jobs) != H23_DIAGNOSTIC_MAX_JOBS
        or tuple(job.job_order_index for job in jobs)
        != tuple(range(H23_DIAGNOSTIC_MAX_JOBS))
        or tuple(job.profile_order_index for job in jobs)
        != tuple(range(H23_DIAGNOSTIC_MAX_JOBS))
        or tuple(job.job_id for job in jobs)
        != tuple(item[0] for item in FROZEN_SOURCE_CHECKPOINTS)
    ):
        raise H23DiagnosticError(
            "manifest small/seed-stage-0 slice is not the frozen 24-profile order"
        )
    return jobs


def _diagnostic_jobs(
    manifest: LoadedDevelopmentCalibrationManifest,
    source_bindings: Sequence[Mapping[str, Any]],
    *,
    plan_id: str = H23_DIAGNOSTIC_PLAN_ID,
) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    for index, (source_job, binding) in enumerate(
        zip(_source_jobs(manifest), source_bindings, strict=True)
    ):
        profile = manifest.profiles[source_job.profile_order_index].solve_profile()
        model_config = profile.effective_model_config()
        contract = {
            "qualification_case_sha256": source_job.qualification_case_sha256,
            "effective_model_config_sha256": (
                source_job.effective_model_config_sha256
            ),
            "demand_scale": profile.demand_scale,
            "locker_fixed_cost_per_hour": profile.locker_fixed_cost_per_hour,
            "display_origin_choices": profile.display_origin_choices,
            "general_corridor_choices": profile.general_corridor_choices,
            "screening_is_exact": False,
        }
        identity = {
            "plan_id": plan_id,
            "diagnostic_job_order_index": index,
            "source_job_id": source_job.job_id,
            "source_checkpoint_file_sha256": binding["file_sha256"],
            "source_checkpoint_sha256": binding["checkpoint_sha256"],
            "qualification_case_sha256": source_job.qualification_case_sha256,
        }
        jobs.append(
            {
                "diagnostic_job_order_index": index,
                "diagnostic_job_id": (
                    "h23-qualification-diagnostic-job-"
                    + canonical_sha256(identity)
                ),
                "source_job": source_job.to_payload(),
                "source_checkpoint": dict(binding),
                "physical_instance_directory": str(
                    manifest.physical_directories[source_job.physical_instance_id]
                ),
                "qualification_solve_profile": profile.to_payload(),
                "effective_model_config": to_dict(model_config),
                "qualification_model_contract": contract,
            }
        )
    return jobs


def build_h23_diagnostic_plan(
    config: H23PlanBuildConfig,
    *,
    hooks: H23PlanBuildHooks = DEFAULT_PLAN_HOOKS,
) -> H23PlanBuildResult:
    """Create one deterministic, exclusive, solver-free diagnostic plan."""

    if not isinstance(config, H23PlanBuildConfig):
        raise TypeError("config must be H23PlanBuildConfig")
    repository_state = dict(hooks.capture_repository_state(_REPOSITORY_ROOT))
    if (
        repository_state.get("status_includes_untracked") is not True
        or repository_state.get("worktree_dirty") is not False
        or repository_state.get("status_lines") != []
    ):
        raise H23DiagnosticError(
            "plan generation requires a clean worktree including untracked files"
        )
    if repository_state.get("git_commit") != config.expected_git_commit:
        raise H23DiagnosticError("plan expected commit does not match exact HEAD")
    implementation = _require_mapping(
        repository_state.get("implementation"), field_name="implementation"
    )
    if set(
        _require_mapping(
            implementation.get("ordered_file_sha256"),
            field_name="implementation files",
        )
    ) != set(IMPLEMENTATION_FILES):
        raise H23DiagnosticError("implementation file set drifted")

    manifest = load_development_calibration_manifest(config.manifest_path)
    if (
        manifest.manifest_sha256 != FROZEN_MANIFEST_SHA256
        or manifest.manifest_file_sha256 != FROZEN_MANIFEST_FILE_SHA256
    ):
        raise H23DiagnosticError("frozen development manifest drifted")
    source_payload, bindings = _validate_source_run(
        manifest.path, config.source_run_root
    )
    jobs = _diagnostic_jobs(manifest, bindings)
    entry = {
        "plan_schema": H23_DIAGNOSTIC_PLAN_SCHEMA,
        "plan_id": H23_DIAGNOSTIC_PLAN_ID,
        "development_only": True,
        "eligible_for_formal_pool": False,
        "formal_pool": False,
        "may_lock_profiles": False,
        "algorithm_comparison_performed": False,
        "mode": "MX",
        "max_jobs": H23_DIAGNOSTIC_MAX_JOBS,
        "scope": {
            "scale": "small",
            "development_seed_stage": 0,
            "all_manifest_profiles": True,
            "profile_subset_allowed": False,
            "job_order": "frozen_manifest_order",
        },
        "source_manifest": {
            "path": str(manifest.path),
            "manifest_sha256": manifest.manifest_sha256,
            "manifest_file_sha256": manifest.manifest_file_sha256,
            "raw_path": str(manifest.raw_path),
            "raw_sha256": manifest.raw_sha256,
            "raw_size_bytes": manifest.raw_size_bytes,
        },
        "source_5s_run": source_payload,
        "qualification_policy": dict(REGISTERED_POLICY_PAYLOAD),
        "native_settings": dict(REGISTERED_NATIVE_SETTINGS_PAYLOAD),
        "geometry_start_policy": dict(REGISTERED_GEOMETRY_POLICY_PAYLOAD),
        "execution_contract": {
            "concurrency": 1,
            "threads": 1,
            "exclusive_results_root": True,
            "shared_resource_lock": True,
            "initial_and_per_job_process_conflict_checks": True,
            "normal_no_witness_rejections_continue": True,
            "infrastructure_fail_fast": True,
            "fresh_original_model_validation_required_for_witness": True,
            "adds_forced_opening_constraint": False,
            "screening_is_exact": False,
        },
        "required_execution_environment": {
            "repository_root": repository_state["repository_root"],
            "git_commit": config.expected_git_commit,
            "clean_status_includes_untracked": True,
            "implementation": dict(implementation),
        },
        "jobs": jobs,
    }
    plan_sha256 = canonical_sha256(entry)
    payload = {
        "artifact_type": _PLAN_ARTIFACT_TYPE,
        "artifact_schema_version": 1,
        **entry,
        "plan_sha256": plan_sha256,
    }
    output = config.output_path.expanduser().absolute()
    output.parent.mkdir(parents=True, exist_ok=True)
    published = write_canonical_json_exclusive(output, payload)
    return H23PlanBuildResult(
        path=published,
        plan_id=H23_DIAGNOSTIC_PLAN_ID,
        plan_sha256=plan_sha256,
        plan_file_sha256=_sha256_file(published),
        job_count=len(jobs),
        git_commit=config.expected_git_commit,
        implementation_sha256=str(implementation["implementation_sha256"]),
    )


@dataclass(frozen=True, slots=True)
class LoadedH23DiagnosticPlan:
    path: Path
    file_sha256: str
    plan_sha256: str
    payload: Mapping[str, Any]
    manifest: LoadedDevelopmentCalibrationManifest
    jobs: tuple[Mapping[str, Any], ...]


def load_h23_diagnostic_plan(
    plan_path: str | Path,
    *,
    expected_plan_id: str,
    expected_max_jobs: int,
) -> LoadedH23DiagnosticPlan:
    """Fully reconstruct and validate one frozen plan before execution."""

    # V1 is accepted only by explicit identity for preserved evidence loading.
    # Builders and execution configs retain their registered V2-only contract.
    if expected_plan_id == H23_DIAGNOSTIC_PLAN_ID:
        plan_schema = H23_DIAGNOSTIC_PLAN_SCHEMA
        native_settings = REGISTERED_NATIVE_SETTINGS_PAYLOAD
        geometry_policy = REGISTERED_GEOMETRY_POLICY_PAYLOAD
    else:
        from . import h23_incumbent_reuse_contract as legacy_reuse

        if expected_plan_id != legacy_reuse.H23_DIAGNOSTIC_PLAN_ID:
            raise H23DiagnosticError("explicit plan ID is not a supported diagnostic")
        plan_schema = "mlnd-h23-qualification-diagnostic-plan-v1"
        native_settings = legacy_reuse.REGISTERED_NATIVE_SETTINGS_PAYLOAD
        geometry_policy = legacy_reuse.REGISTERED_GEOMETRY_POLICY_PAYLOAD
    if expected_max_jobs != H23_DIAGNOSTIC_MAX_JOBS or isinstance(
        expected_max_jobs, bool
    ):
        raise H23DiagnosticError("explicit max-jobs must equal 24")
    path = _require_real_path(plan_path, field_name="plan", kind="file")
    payload = _read_canonical_object(path, field_name="diagnostic plan")
    expected_keys = {
        "artifact_type",
        "artifact_schema_version",
        "plan_schema",
        "plan_id",
        "development_only",
        "eligible_for_formal_pool",
        "formal_pool",
        "may_lock_profiles",
        "algorithm_comparison_performed",
        "mode",
        "max_jobs",
        "scope",
        "source_manifest",
        "source_5s_run",
        "qualification_policy",
        "native_settings",
        "geometry_start_policy",
        "execution_contract",
        "required_execution_environment",
        "jobs",
        "plan_sha256",
    }
    _require_exact_keys(payload, expected_keys, field_name="diagnostic plan")
    fixed = {
        "artifact_type": _PLAN_ARTIFACT_TYPE,
        "artifact_schema_version": 1,
        "plan_schema": plan_schema,
        "plan_id": expected_plan_id,
        "development_only": True,
        "eligible_for_formal_pool": False,
        "formal_pool": False,
        "may_lock_profiles": False,
        "algorithm_comparison_performed": False,
        "mode": "MX",
        "max_jobs": H23_DIAGNOSTIC_MAX_JOBS,
    }
    if any(payload.get(key) != value for key, value in fixed.items()):
        raise H23DiagnosticError("diagnostic plan fixed flags or identity drifted")
    if payload["scope"] != {
        "scale": "small",
        "development_seed_stage": 0,
        "all_manifest_profiles": True,
        "profile_subset_allowed": False,
        "job_order": "frozen_manifest_order",
    }:
        raise H23DiagnosticError("diagnostic scope drifted")
    if payload["qualification_policy"] != REGISTERED_POLICY_PAYLOAD:
        raise H23DiagnosticError("qualification policy drifted")
    if payload["native_settings"] != native_settings:
        raise H23DiagnosticError("30-second native settings drifted")
    if payload["geometry_start_policy"] != geometry_policy:
        raise H23DiagnosticError("geometry-start v2 settings drifted")
    if payload["execution_contract"] != {
        "concurrency": 1,
        "threads": 1,
        "exclusive_results_root": True,
        "shared_resource_lock": True,
        "initial_and_per_job_process_conflict_checks": True,
        "normal_no_witness_rejections_continue": True,
        "infrastructure_fail_fast": True,
        "fresh_original_model_validation_required_for_witness": True,
        "adds_forced_opening_constraint": False,
        "screening_is_exact": False,
    }:
        raise H23DiagnosticError("diagnostic execution contract drifted")
    plan_sha256 = _require_sha256(payload["plan_sha256"], field_name="plan_sha256")
    unhashed = {
        key: value
        for key, value in payload.items()
        if key not in {"artifact_type", "artifact_schema_version", "plan_sha256"}
    }
    if canonical_sha256(unhashed) != plan_sha256:
        raise H23DiagnosticError("diagnostic plan self-hash mismatch")

    source_manifest = _require_mapping(
        payload["source_manifest"], field_name="source_manifest"
    )
    _require_exact_keys(
        source_manifest,
        {
            "path",
            "manifest_sha256",
            "manifest_file_sha256",
            "raw_path",
            "raw_sha256",
            "raw_size_bytes",
        },
        field_name="source_manifest",
    )
    if (
        source_manifest["manifest_sha256"] != FROZEN_MANIFEST_SHA256
        or source_manifest["manifest_file_sha256"]
        != FROZEN_MANIFEST_FILE_SHA256
    ):
        raise H23DiagnosticError("plan is not bound to the frozen manifest")
    manifest = load_development_calibration_manifest(source_manifest["path"])
    expected_manifest_payload = {
        "path": str(manifest.path),
        "manifest_sha256": manifest.manifest_sha256,
        "manifest_file_sha256": manifest.manifest_file_sha256,
        "raw_path": str(manifest.raw_path),
        "raw_sha256": manifest.raw_sha256,
        "raw_size_bytes": manifest.raw_size_bytes,
    }
    if dict(source_manifest) != expected_manifest_payload:
        raise H23DiagnosticError("plan source manifest or raw binding drifted")
    source_run = _require_mapping(payload["source_5s_run"], field_name="source_5s_run")
    observed_source, bindings = _validate_source_run(
        manifest.path, Path(str(source_run.get("root", "")))
    )
    if canonical_json(source_run) != canonical_json(observed_source):
        raise H23DiagnosticError("plan source 5-second run binding drifted")
    expected_jobs = _diagnostic_jobs(manifest, bindings, plan_id=expected_plan_id)
    if canonical_json(payload["jobs"]) != canonical_json(expected_jobs):
        raise H23DiagnosticError("diagnostic 24-job order, profile, or hash drifted")
    required_environment = _require_mapping(
        payload["required_execution_environment"],
        field_name="required_execution_environment",
    )
    _require_exact_keys(
        required_environment,
        {
            "repository_root",
            "git_commit",
            "clean_status_includes_untracked",
            "implementation",
        },
        field_name="required_execution_environment",
    )
    commit = required_environment["git_commit"]
    if (
        not isinstance(commit, str)
        or len(commit) != 40
        or any(character not in "0123456789abcdef" for character in commit)
        or required_environment["clean_status_includes_untracked"] is not True
    ):
        raise H23DiagnosticError("required execution commit/status contract is invalid")
    implementation = _require_mapping(
        required_environment["implementation"], field_name="required implementation"
    )
    _require_exact_keys(
        implementation,
        {"ordered_file_sha256", "implementation_sha256"},
        field_name="required implementation",
    )
    files = _require_mapping(
        implementation["ordered_file_sha256"], field_name="required files"
    )
    from .h23_qualification_diagnostic_contract import LEGACY_IMPLEMENTATION_FILES

    if set(files) not in (set(IMPLEMENTATION_FILES), set(LEGACY_IMPLEMENTATION_FILES)):
        raise H23DiagnosticError("required implementation file set drifted")
    if implementation["implementation_sha256"] != canonical_sha256(files):
        raise H23DiagnosticError("required implementation aggregate hash drifted")
    return LoadedH23DiagnosticPlan(
        path=path,
        file_sha256=_sha256_file(path),
        plan_sha256=plan_sha256,
        payload=payload,
        manifest=manifest,
        jobs=tuple(expected_jobs),
    )


def _default_instance_loader(
    instance_dir: Path, model_config: ModelConfig, demand_scale: float
) -> object:
    from src.data_processor.loader import DataLoader

    return DataLoader().load_instance_dir(
        str(instance_dir), model_config=model_config, demand_scale=demand_scale
    )


def _default_runtime_environment(repository_root: Path) -> Mapping[str, Any]:
    import gurobipy as gp

    status_lines = tuple(
        line
        for line in _git(
            repository_root, "status", "--porcelain", "--untracked-files=all"
        ).splitlines()
        if line
    )
    return {
        "git": {
            "repository_root": str(repository_root),
            "commit": _git(repository_root, "rev-parse", "HEAD"),
            "worktree_dirty": bool(status_lines),
            "status_includes_untracked": True,
            "status_lines": list(status_lines),
            "status_sha256": canonical_sha256(status_lines),
        },
        "implementation": _implementation_payload(repository_root),
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
            "executable": sys.executable,
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "gurobi_version": ".".join(str(value) for value in gp.gurobi.version()),
    }


def _process_table() -> str:
    completed = subprocess.run(
        ("ps", "-axo", "pid=,ppid=,command="),
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise H23DiagnosticError(
            "cannot audit concurrent solver processes: "
            + (completed.stderr or completed.stdout).strip()
        )
    return completed.stdout


def _conflicting_processes(process_table: str) -> tuple[str, ...]:
    markers = (
        "main.py solve",
        "main.py run",
        "calibration_runner",
        "h6_directed_return_ablation",
        "branch_cut qualify",
        "branch_cut run ",
        "branch_cut run-formal",
        "mlnd_qualification",
        "gurobi_cl",
        "pytest",
        "h23_qualification_diagnostic run",
    )
    rows: dict[int, tuple[int, str]] = {}
    for raw_line in process_table.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        fields = line.split(maxsplit=2)
        try:
            pid = int(fields[0])
            ppid = int(fields[1])
        except (IndexError, ValueError):
            continue
        rows[pid] = (ppid, line)
    excluded = {os.getpid()}
    cursor = os.getppid()
    while cursor > 0 and cursor not in excluded:
        excluded.add(cursor)
        cursor = rows.get(cursor, (0, ""))[0]
    return tuple(
        line
        for pid, (_ppid, line) in sorted(rows.items())
        if pid not in excluded and any(marker in line for marker in markers)
    )


def _process_audit(table: str) -> dict[str, Any]:
    conflicts = _conflicting_processes(table)
    return {
        "process_table_sha256": hashlib.sha256(table.encode("utf-8")).hexdigest(),
        "conflicting_processes": list(conflicts),
        "valid": not conflicts,
    }


def _stable_environment(environment: Mapping[str, Any]) -> dict[str, Any]:
    git_payload = dict(_require_mapping(environment.get("git"), field_name="git"))
    for key in (
        "worktree_dirty",
        "status_includes_untracked",
        "status_lines",
        "status_sha256",
    ):
        git_payload.pop(key, None)
    return {**dict(environment), "git": git_payload}


@dataclass(frozen=True, slots=True)
class H23DiagnosticHooks:
    load_instance: Callable[[Path, ModelConfig, float], object] = (
        _default_instance_loader
    )
    qualify_instance: Callable[..., QualificationRunnerOutcome] = (
        qualify_canonical_instance
    )
    capture_environment: Callable[[Path], Mapping[str, Any]] = (
        _default_runtime_environment
    )
    process_table: Callable[[], str] = _process_table
    now_utc: Callable[[], str] = _utc_now


DEFAULT_RUN_HOOKS = H23DiagnosticHooks()


@dataclass(frozen=True, slots=True)
class H23DiagnosticRunConfig:
    plan_path: Path
    results_root: Path
    plan_id: str
    max_jobs: int
    resource_lock: Path = field(default_factory=lambda: DEFAULT_BENCHMARK_RESOURCE_LOCK)

    def __post_init__(self) -> None:
        if self.plan_id != H23_DIAGNOSTIC_PLAN_ID:
            raise ValueError(f"plan_id must be {H23_DIAGNOSTIC_PLAN_ID}")
        if self.max_jobs != H23_DIAGNOSTIC_MAX_JOBS or isinstance(
            self.max_jobs, bool
        ):
            raise ValueError("max_jobs is frozen at 24")


@dataclass(frozen=True, slots=True)
class H23DiagnosticRunResult:
    results_root: Path
    run_identity_path: Path
    summary_path: Path
    diagnostic_run_id: str
    attempted_job_count: int
    accepted_job_count: int
    checkpoint_paths: tuple[Path, ...]

    def to_payload(self) -> dict[str, Any]:
        return {
            "results_root": str(self.results_root),
            "run_identity_path": str(self.run_identity_path),
            "summary_path": str(self.summary_path),
            "diagnostic_run_id": self.diagnostic_run_id,
            "attempted_job_count": self.attempted_job_count,
            "accepted_job_count": self.accepted_job_count,
            "checkpoint_paths": [str(path) for path in self.checkpoint_paths],
        }


def _validate_runtime_environment(
    environment: Mapping[str, Any], plan: LoadedH23DiagnosticPlan
) -> None:
    git_payload = _require_mapping(environment.get("git"), field_name="environment.git")
    if (
        git_payload.get("status_includes_untracked") is not True
        or git_payload.get("worktree_dirty") is not False
        or git_payload.get("status_lines") != []
    ):
        raise H23DiagnosticError(
            "diagnostic run requires a clean worktree including untracked files"
        )
    required = _require_mapping(
        plan.payload["required_execution_environment"], field_name="required environment"
    )
    if (
        git_payload.get("commit") != required["git_commit"]
        or git_payload.get("repository_root") != required["repository_root"]
        or environment.get("implementation") != required["implementation"]
    ):
        raise H23DiagnosticError(
            "runtime commit, repository, or implementation differs from the plan"
        )
    if not isinstance(environment.get("gurobi_version"), str) or not environment[
        "gurobi_version"
    ]:
        raise H23DiagnosticError("runtime environment lacks a Gurobi version")


def _validate_loaded_instance(
    instance: object, *, model_config: ModelConfig, demand_scale: float
) -> None:
    if getattr(instance, "config", None) != model_config:
        raise QualificationInfrastructureError(
            "loaded diagnostic instance did not retain its full ModelConfig"
        )
    metadata = getattr(instance, "metadata", None)
    observed = metadata.get("demand_scale") if isinstance(metadata, Mapping) else None
    if (
        isinstance(observed, bool)
        or not isinstance(observed, (int, float))
        or float(observed) != demand_scale
    ):
        raise QualificationInfrastructureError(
            "loaded diagnostic instance did not retain demand_scale"
        )
    screening = getattr(instance, "screening_is_exact", None)
    if not callable(screening) or screening("MX") is not False:
        raise QualificationInfrastructureError(
            "H23 diagnostic requires screening_is_exact=false"
        )


def _audit_matches(
    audit: QualificationModelAudit, contract: QualificationModelContract
) -> bool:
    return audit.contract_values() == (
        contract.effective_model_config_sha256,
        contract.demand_scale,
        contract.locker_fixed_cost_per_hour,
        contract.display_origin_choices,
        contract.general_corridor_choices,
        False,
    )


def _validate_outcome(
    outcome: QualificationRunnerOutcome,
    *,
    job: Mapping[str, Any],
    run_id: str,
    policy: QualificationPolicy,
    geometry_policy: GeometryStartPolicy,
) -> None:
    if not isinstance(outcome, QualificationRunnerOutcome):
        raise QualificationInfrastructureError(
            "qualifier did not return QualificationRunnerOutcome"
        )
    contract = QualificationModelContract(**job["qualification_model_contract"])
    if outcome.expected_model_contract != contract:
        raise QualificationInfrastructureError("qualification model contract drifted")
    if outcome.geometry_start_policy != geometry_policy:
        raise QualificationInfrastructureError("geometry-start policy drifted")
    independently_assessed = assess_qualification(outcome.evidence, policy)
    if outcome.decision != independently_assessed:
        raise QualificationInfrastructureError(
            "qualification decision differs from independent assessment"
        )
    if outcome.accepted != (not outcome.rejection_reasons):
        raise QualificationInfrastructureError("accepted/rejection state is inconsistent")
    decision_cursor = iter(outcome.rejection_reasons)
    if not all(
        any(candidate == reason for candidate in decision_cursor)
        for reason in outcome.decision.rejection_reasons
    ):
        raise QualificationInfrastructureError(
            "decision reasons are not an ordered outcome subset"
        )
    extras = set(outcome.rejection_reasons).difference(
        outcome.decision.rejection_reasons
    )
    if not extras.issubset(NORMAL_QUALIFICATION_RUNNER_REASONS):
        raise QualificationInfrastructureError(
            "normal outcome contains infrastructure-level runner reasons"
        )
    expected_identity = (
        job["diagnostic_job_id"],
        "small",
        run_id,
        job["source_job"]["instance_sha256"],
        job["source_job"]["qualification_case_sha256"],
        job["source_job"]["effective_model_config_sha256"],
        False,
    )
    for label, record in (("evidence", outcome.evidence), ("decision", outcome.decision)):
        observed = (
            record.case_id,
            record.scale,
            record.qualification_run_id,
            record.instance_sha256,
            record.qualification_case_sha256,
            record.effective_model_config_sha256,
            record.screening_is_exact,
        )
        if observed != expected_identity:
            raise QualificationInfrastructureError(
                f"qualification {label} identity differs from the diagnostic job"
            )
    if outcome.decision.policy != policy:
        raise QualificationInfrastructureError("qualification policy drifted")
    if outcome.witness_status not in NORMAL_QUALIFICATION_SOLVER_STATUSES:
        raise QualificationInfrastructureError("abnormal witness solver status")
    if outcome.fresh_model_validation_performed:
        if outcome.validation_status not in NORMAL_QUALIFICATION_SOLVER_STATUSES:
            raise QualificationInfrastructureError("abnormal fresh validation status")
    elif outcome.validation_status != "NOT_RUN":
        raise QualificationInfrastructureError(
            "validation status exists without fresh validation"
        )
    if outcome.exception_messages:
        raise QualificationInfrastructureError(
            "completed diagnostic outcome contains mechanism exceptions"
        )
    if outcome.witness_model_audit is None or not _audit_matches(
        outcome.witness_model_audit, contract
    ):
        raise QualificationInfrastructureError("witness model audit is missing/drifted")
    fresh = outcome.fresh_model_audit
    if outcome.fresh_model_validation_performed:
        if fresh is None or not _audit_matches(fresh, contract):
            raise QualificationInfrastructureError("fresh model audit is missing/drifted")
        if (
            outcome.witness_model_audit.model_fingerprint is None
            or fresh.model_fingerprint is None
            or outcome.witness_model_audit.model_fingerprint
            != fresh.model_fingerprint
        ):
            raise QualificationInfrastructureError(
                "fresh and witness model fingerprints differ"
            )
    elif fresh is not None:
        raise QualificationInfrastructureError(
            "fresh audit exists without fresh validation"
        )
    if outcome.evidence.witness_feasible and not outcome.fresh_model_validation_performed:
        raise QualificationInfrastructureError(
            "feasible witness lacks fresh original-model validation"
        )
    if outcome.accepted and (
        not outcome.fresh_model_validation_performed
        or not outcome.evidence.witness_feasible
        or not outcome.decision.accepted
        or outcome.captured_variable_count <= 0
    ):
        raise QualificationInfrastructureError(
            "accepted outcome lacks a fresh validated complete witness"
        )
    attempts = outcome.geometry_start_attempts
    schedule = geometry_policy.schedule_for("small", outcome.evidence.candidate_h)
    if len(attempts) > len(schedule) or tuple(
        attempt.plan.target_support_size for attempt in attempts
    ) != schedule[: len(attempts)]:
        raise QualificationInfrastructureError("geometry attempt order drifted")
    full = [attempt for attempt in attempts if attempt.full_witness_found]
    if len(full) > 1 or (full and full[0] is not attempts[-1]):
        raise QualificationInfrastructureError("geometry witness sequence is invalid")
    selected = outcome.geometry_start_selected_plan_sha256
    if full:
        if selected != full[0].plan.plan_sha256:
            raise QualificationInfrastructureError("selected geometry plan hash drifted")
    elif selected is not None:
        raise QualificationInfrastructureError("geometry plan selected without witness")
    for attempt in attempts:
        if (
            attempt.per_attempt_seconds != geometry_policy.per_attempt_seconds
            or attempt.start_node_limit != geometry_policy.start_node_limit
        ):
            raise QualificationInfrastructureError("geometry attempt budget drifted")


def _claim_results_root(
    target: Path, *, plan: LoadedH23DiagnosticPlan
) -> Path:
    absolute = target.expanduser().absolute()
    if absolute.exists() or absolute.is_symlink():
        raise H23DiagnosticError(f"refusing to reuse results root: {absolute}")
    forbidden = (
        plan.manifest.root,
        Path(str(plan.payload["source_5s_run"]["root"])),
    )
    resolved_parent = absolute.parent.resolve()
    for root in forbidden:
        try:
            resolved_parent.relative_to(root.resolve())
        except ValueError:
            continue
        raise H23DiagnosticError("results root may not be inside a frozen input root")
    if "formal" in {part.lower() for part in absolute.parts}:
        raise H23DiagnosticError("development diagnostic may not write under formal")
    absolute.parent.mkdir(parents=True, exist_ok=True)
    try:
        absolute.mkdir(exist_ok=False)
    except FileExistsError as error:
        raise H23DiagnosticError(f"refusing to reuse results root: {absolute}") from error
    return absolute.resolve()


def _checkpoint_path(root: Path, job: Mapping[str, Any]) -> Path:
    job_id = str(job["diagnostic_job_id"])
    digest = hashlib.sha256(job_id.encode("utf-8")).hexdigest()
    return root / f"job-{int(job['diagnostic_job_order_index']):03d}-{digest}.json"


def _write_checkpoint(
    path: Path,
    *,
    identity: Mapping[str, Any],
    result: Mapping[str, Any],
) -> tuple[Path, str]:
    entry = {"identity": dict(identity), "result": dict(result)}
    payload = {
        "artifact_type": _CHECKPOINT_ARTIFACT_TYPE,
        "artifact_schema_version": 1,
        "checkpoint_schema": H23_DIAGNOSTIC_CHECKPOINT_SCHEMA,
        "development_only": True,
        "eligible_for_formal_pool": False,
        "formal_pool": False,
        "may_lock_profiles": False,
        "contains_algorithm_comparison_results": False,
        **entry,
        "checkpoint_sha256": canonical_sha256(entry),
    }
    return write_canonical_json_exclusive(path, payload), payload["checkpoint_sha256"]


def _infrastructure_payload(stage: str, error: BaseException) -> dict[str, Any]:
    return {
        "stage": stage,
        "type": type(error).__name__,
        "message": str(error),
        "qualification_infrastructure_error": isinstance(
            error, QualificationInfrastructureError
        ),
    }


def _ledger(
    jobs: Sequence[Mapping[str, Any]],
    records: Mapping[str, Mapping[str, Any]],
    *,
    aborted: bool,
) -> list[dict[str, Any]]:
    ledger: list[dict[str, Any]] = []
    for job in jobs:
        job_id = str(job["diagnostic_job_id"])
        if job_id in records:
            ledger.append(dict(records[job_id]))
        else:
            ledger.append(
                {
                    "diagnostic_job_order_index": job[
                        "diagnostic_job_order_index"
                    ],
                    "diagnostic_job_id": job_id,
                    "source_job_id": job["source_job"]["job_id"],
                    "profile_id": job["source_job"]["profile_id"],
                    "attempted": False,
                    "accepted": False,
                    "not_attempted_reason": (
                        "infrastructure_abort" if aborted else None
                    ),
                    "checkpoint_path": None,
                    "checkpoint_sha256": None,
                    "infrastructure_failure": None,
                    "outcome": None,
                }
            )
    if len(ledger) != H23_DIAGNOSTIC_MAX_JOBS:
        raise H23DiagnosticError("diagnostic ledger lost its 24-row order")
    return ledger


def _revalidate_job_inputs(
    plan: LoadedH23DiagnosticPlan, job: Mapping[str, Any]
) -> None:
    if _sha256_file(plan.path) != plan.file_sha256:
        raise QualificationInfrastructureError("diagnostic plan changed during run")
    manifest = plan.manifest
    if (
        _sha256_file(manifest.path) != manifest.manifest_file_sha256
        or _sha256_file(manifest.raw_path) != manifest.raw_sha256
        or manifest.raw_path.stat().st_size != manifest.raw_size_bytes
    ):
        raise QualificationInfrastructureError("manifest or raw input changed")
    source_job = job["source_job"]
    instance_dir = manifest.physical_directories[source_job["physical_instance_id"]]
    observed_instance, _ = canonical_instance_hash(instance_dir)
    if observed_instance != source_job["instance_sha256"]:
        raise QualificationInfrastructureError("canonical H23 instance changed")
    source_root = Path(str(plan.payload["source_5s_run"]["root"]))
    source_checkpoint = job["source_checkpoint"]
    source_path = source_root / str(source_checkpoint["relative_path"])
    if _sha256_file(source_path) != source_checkpoint["file_sha256"]:
        raise QualificationInfrastructureError("source 5-second checkpoint changed")
    for relative, expected in (
        ("run_identity.json", FROZEN_SOURCE_RUN_IDENTITY_FILE_SHA256),
        ("batch_summary.json", FROZEN_SOURCE_SUMMARY_FILE_SHA256),
    ):
        if _sha256_file(source_root / relative) != expected:
            raise QualificationInfrastructureError(
                f"source 5-second {relative} changed"
            )


def run_h23_diagnostic(
    config: H23DiagnosticRunConfig,
    *,
    hooks: H23DiagnosticHooks = DEFAULT_RUN_HOOKS,
) -> H23DiagnosticRunResult:
    """Run all 24 jobs serially; normal rejections continue, infra aborts."""

    if not isinstance(config, H23DiagnosticRunConfig):
        raise TypeError("config must be H23DiagnosticRunConfig")
    plan = load_h23_diagnostic_plan(
        config.plan_path,
        expected_plan_id=config.plan_id,
        expected_max_jobs=config.max_jobs,
    )
    settings = _registered_native_settings()
    policy = _registered_policy()
    geometry_policy = _registered_geometry_policy()

    with BenchmarkResourceLock(
        config.resource_lock, command="h23-qualification-diagnostic"
    ):
        environment = dict(hooks.capture_environment(_REPOSITORY_ROOT))
        _validate_runtime_environment(environment, plan)
        environment_sha256 = canonical_sha256(environment)
        stable_environment_sha256 = canonical_sha256(
            _stable_environment(environment)
        )
        initial_process_audit = _process_audit(hooks.process_table())
        if not initial_process_audit["valid"]:
            raise H23DiagnosticError(
                "conflicting solver/test processes are active: "
                + " | ".join(initial_process_audit["conflicting_processes"])
            )
        required_environment = plan.payload["required_execution_environment"]
        run_identity = {
            "run_schema": H23_DIAGNOSTIC_RUN_SCHEMA,
            "plan_id": H23_DIAGNOSTIC_PLAN_ID,
            "plan_path": str(plan.path),
            "plan_sha256": plan.plan_sha256,
            "plan_file_sha256": plan.file_sha256,
            "development_only": True,
            "eligible_for_formal_pool": False,
            "formal_pool": False,
            "may_lock_profiles": False,
            "algorithm_comparison_performed": False,
            "mode": "MX",
            "max_jobs": H23_DIAGNOSTIC_MAX_JOBS,
            "concurrency": 1,
            "qualification_policy": dict(REGISTERED_POLICY_PAYLOAD),
            "native_settings": dict(REGISTERED_NATIVE_SETTINGS_PAYLOAD),
            "geometry_start_policy": dict(REGISTERED_GEOMETRY_POLICY_PAYLOAD),
            "jobs": [dict(job) for job in plan.jobs],
            "source_manifest_sha256": FROZEN_MANIFEST_SHA256,
            "source_calibration_run_id": FROZEN_SOURCE_RUN_ID,
            "environment": environment,
            "environment_sha256": environment_sha256,
            "stable_environment_sha256": stable_environment_sha256,
            "git_commit": required_environment["git_commit"],
            "implementation_sha256": required_environment["implementation"][
                "implementation_sha256"
            ],
            "initial_process_audit": initial_process_audit,
        }
        run_sha256 = canonical_sha256(run_identity)
        run_id = _RUN_ID_PREFIX + run_sha256
        results_root = _claim_results_root(config.results_root, plan=plan)
        checkpoints_root = results_root / "checkpoints"
        checkpoints_root.mkdir(exist_ok=False)
        run_identity_path = write_canonical_json_exclusive(
            results_root / "run_identity.json",
            {
                "artifact_type": _RUN_ARTIFACT_TYPE,
                "artifact_schema_version": 1,
                "diagnostic_run_id": run_id,
                "run_identity": run_identity,
                "run_identity_sha256": run_sha256,
            },
        )
        checkpoint_paths: list[Path] = []
        checkpoint_records: list[dict[str, Any]] = []
        ledger_records: dict[str, dict[str, Any]] = {}
        accepted_count = 0

        def revalidate_environment() -> None:
            observed = dict(hooks.capture_environment(_REPOSITORY_ROOT))
            _validate_runtime_environment(observed, plan)
            if canonical_sha256(_stable_environment(observed)) != (
                stable_environment_sha256
            ):
                raise QualificationInfrastructureError(
                    "runtime environment changed during the diagnostic"
                )

        for attempt_index, job in enumerate(plan.jobs):
            job_id = str(job["diagnostic_job_id"])
            checkpoint_path = _checkpoint_path(checkpoints_root, job)
            stage = "process_conflict_check"
            try:
                job_process_audit = _process_audit(hooks.process_table())
            except BaseException as error:
                job_process_audit = {
                    "process_table_sha256": None,
                    "conflicting_processes": [],
                    "valid": False,
                    "audit_error": f"{type(error).__name__}: {error}",
                }
                pending_error: BaseException | None = error
            else:
                pending_error = None
                if not job_process_audit["valid"]:
                    pending_error = QualificationInfrastructureError(
                        "conflicting processes appeared before job: "
                        + " | ".join(job_process_audit["conflicting_processes"])
                    )
            source_job = job["source_job"]
            profile_payload = job["qualification_solve_profile"]
            profile = plan.manifest.profiles[
                source_job["profile_order_index"]
            ].solve_profile()
            model_config = profile.effective_model_config()
            contract = QualificationModelContract(
                **job["qualification_model_contract"]
            )
            identity = {
                "diagnostic_run_id": run_id,
                "run_identity_sha256": run_sha256,
                "plan_id": H23_DIAGNOSTIC_PLAN_ID,
                "plan_sha256": plan.plan_sha256,
                "plan_file_sha256": plan.file_sha256,
                "environment_sha256": environment_sha256,
                "git_commit": required_environment["git_commit"],
                "execution_attempt_index": attempt_index,
                "job": dict(job),
                "process_audit": job_process_audit,
                "physical_instance_directory": job[
                    "physical_instance_directory"
                ],
                "qualification_solve_profile": profile_payload,
                "effective_model_config": to_dict(model_config),
                "qualification_model_contract": contract.to_payload(),
            }
            failure_error: BaseException | None = None
            try:
                if pending_error is not None:
                    raise pending_error
                stage = "pre_solve_input_revalidation"
                revalidate_environment()
                _revalidate_job_inputs(plan, job)
                stage = "instance_load"
                instance_dir = Path(str(job["physical_instance_directory"]))
                instance = hooks.load_instance(
                    instance_dir, model_config, profile.demand_scale
                )
                _validate_loaded_instance(
                    instance,
                    model_config=model_config,
                    demand_scale=profile.demand_scale,
                )
                stage = "qualification"
                outcome = hooks.qualify_instance(
                    instance,
                    "MX",
                    case_id=job_id,
                    scale="small",
                    qualification_run_id=run_id,
                    instance_sha256=source_job["instance_sha256"],
                    policy=policy,
                    settings=settings,
                    geometry_start_policy=geometry_policy,
                    expected_model_contract=contract,
                    fail_fast_infrastructure=True,
                )
                _validate_outcome(
                    outcome,
                    job=job,
                    run_id=run_id,
                    policy=policy,
                    geometry_policy=geometry_policy,
                )
                stage = "post_solve_input_revalidation"
                _revalidate_job_inputs(plan, job)
                revalidate_environment()
                result_payload = {
                    "status": "completed",
                    "completed_at_utc": hooks.now_utc(),
                    "infrastructure_error": None,
                    "outcome": outcome.to_payload(),
                }
                published, checkpoint_sha256 = _write_checkpoint(
                    checkpoint_path, identity=identity, result=result_payload
                )
                accepted = outcome.accepted
                infrastructure_error = None
                outcome_payload: Mapping[str, Any] | None = outcome.to_payload()
            except BaseException as error:
                failure_error = error
                infrastructure_error = _infrastructure_payload(stage, error)
                result_payload = {
                    "status": "infrastructure_failure",
                    "completed_at_utc": hooks.now_utc(),
                    "infrastructure_error": infrastructure_error,
                    "outcome": None,
                }
                published, checkpoint_sha256 = _write_checkpoint(
                    checkpoint_path, identity=identity, result=result_payload
                )
                accepted = False
                outcome_payload = None
            checkpoint_paths.append(published)
            checkpoint_record = {
                "diagnostic_job_id": job_id,
                "diagnostic_job_order_index": job[
                    "diagnostic_job_order_index"
                ],
                "execution_attempt_index": attempt_index,
                "path": str(published),
                "checkpoint_sha256": checkpoint_sha256,
                "status": result_payload["status"],
                "accepted": accepted,
            }
            checkpoint_records.append(checkpoint_record)
            ledger_records[job_id] = {
                "diagnostic_job_order_index": job[
                    "diagnostic_job_order_index"
                ],
                "diagnostic_job_id": job_id,
                "source_job_id": source_job["job_id"],
                "profile_id": source_job["profile_id"],
                "attempted": True,
                "accepted": accepted,
                "not_attempted_reason": None,
                "checkpoint_path": str(published),
                "checkpoint_sha256": checkpoint_sha256,
                "infrastructure_failure": infrastructure_error,
                "outcome": outcome_payload,
            }
            if infrastructure_error is not None:
                assert failure_error is not None
                summary_entry = {
                    "summary_schema": H23_DIAGNOSTIC_SUMMARY_SCHEMA,
                    "plan_id": H23_DIAGNOSTIC_PLAN_ID,
                    "development_only": True,
                    "eligible_for_formal_pool": False,
                    "formal_pool": False,
                    "may_lock_profiles": False,
                    "diagnostic_run_id": run_id,
                    "run_identity_sha256": run_sha256,
                    "plan_sha256": plan.plan_sha256,
                    "status": "aborted_on_infrastructure_failure",
                    "max_jobs": H23_DIAGNOSTIC_MAX_JOBS,
                    "attempted_job_count": len(checkpoint_records),
                    "completed_job_count": sum(
                        item["status"] == "completed" for item in checkpoint_records
                    ),
                    "accepted_job_count": accepted_count,
                    "rejected_attempted_job_count": sum(
                        item["status"] == "completed" and not item["accepted"]
                        for item in checkpoint_records
                    ),
                    "infrastructure_failure_count": 1,
                    "unattempted_job_count": (
                        H23_DIAGNOSTIC_MAX_JOBS - len(checkpoint_records)
                    ),
                    "failed_job_id": job_id,
                    "formal_pool_created": False,
                    "locked_profile_ids": [],
                    "checkpoints": checkpoint_records,
                    "diagnostic_job_ledger": _ledger(
                        plan.jobs, ledger_records, aborted=True
                    ),
                    "completed_at_utc": hooks.now_utc(),
                }
                summary_path = write_canonical_json_exclusive(
                    results_root / "batch_aborted.json",
                    {
                        "artifact_type": _SUMMARY_ARTIFACT_TYPE,
                        "artifact_schema_version": 1,
                        **summary_entry,
                        "summary_sha256": canonical_sha256(summary_entry),
                    },
                )
                if isinstance(
                    failure_error, (MemoryError, KeyboardInterrupt, SystemExit)
                ):
                    raise failure_error
                raise H23DiagnosticBatchInfrastructureError(
                    f"H23 diagnostic stopped at {job_id} during {stage}; "
                    f"artifacts: {results_root}"
                ) from failure_error
            accepted_count += int(accepted)

        summary_entry = {
            "summary_schema": H23_DIAGNOSTIC_SUMMARY_SCHEMA,
            "plan_id": H23_DIAGNOSTIC_PLAN_ID,
            "development_only": True,
            "eligible_for_formal_pool": False,
            "formal_pool": False,
            "may_lock_profiles": False,
            "diagnostic_run_id": run_id,
            "run_identity_sha256": run_sha256,
            "plan_sha256": plan.plan_sha256,
            "status": "completed",
            "diagnostic_conclusion": "development_diagnostic_completed_24_profiles",
            "max_jobs": H23_DIAGNOSTIC_MAX_JOBS,
            "attempted_job_count": H23_DIAGNOSTIC_MAX_JOBS,
            "completed_job_count": H23_DIAGNOSTIC_MAX_JOBS,
            "accepted_job_count": accepted_count,
            "rejected_attempted_job_count": (
                H23_DIAGNOSTIC_MAX_JOBS - accepted_count
            ),
            "infrastructure_failure_count": 0,
            "unattempted_job_count": 0,
            "accepted_profile_order_indices": [
                job["source_job"]["profile_order_index"]
                for job, record in zip(plan.jobs, checkpoint_records, strict=True)
                if record["accepted"]
            ],
            "formal_pool_created": False,
            "locked_profile_ids": [],
            "checkpoints": checkpoint_records,
            "diagnostic_job_ledger": _ledger(
                plan.jobs, ledger_records, aborted=False
            ),
            "completed_at_utc": hooks.now_utc(),
        }
        summary_path = write_canonical_json_exclusive(
            results_root / "batch_summary.json",
            {
                "artifact_type": _SUMMARY_ARTIFACT_TYPE,
                "artifact_schema_version": 1,
                **summary_entry,
                "summary_sha256": canonical_sha256(summary_entry),
            },
        )
        return H23DiagnosticRunResult(
            results_root=results_root,
            run_identity_path=run_identity_path,
            summary_path=summary_path,
            diagnostic_run_id=run_id,
            attempted_job_count=H23_DIAGNOSTIC_MAX_JOBS,
            accepted_job_count=accepted_count,
            checkpoint_paths=tuple(checkpoint_paths),
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.benchmark.branch_cut.h23_qualification_diagnostic",
        description=(
            "Build or execute the frozen development-only H23 qualification "
            "diagnostic.  The plan command never imports Gurobi."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    plan = subparsers.add_parser("plan")
    plan.add_argument("--plan-id", required=True, choices=(H23_DIAGNOSTIC_PLAN_ID,))
    plan.add_argument(
        "--max-jobs", type=int, required=True, choices=(H23_DIAGNOSTIC_MAX_JOBS,)
    )
    plan.add_argument("--manifest", required=True)
    plan.add_argument("--source-run-root", required=True)
    plan.add_argument("--output", required=True)
    plan.add_argument("--expected-git-commit", required=True)

    run = subparsers.add_parser("run")
    run.add_argument("--plan-id", required=True, choices=(H23_DIAGNOSTIC_PLAN_ID,))
    run.add_argument(
        "--max-jobs", type=int, required=True, choices=(H23_DIAGNOSTIC_MAX_JOBS,)
    )
    run.add_argument("--plan", required=True)
    run.add_argument("--results-root", required=True)
    run.add_argument("--resource-lock", default=str(DEFAULT_BENCHMARK_RESOURCE_LOCK))
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "plan":
            result: H23PlanBuildResult | H23DiagnosticRunResult = (
                build_h23_diagnostic_plan(
                    H23PlanBuildConfig(
                        manifest_path=Path(args.manifest),
                        source_run_root=Path(args.source_run_root),
                        output_path=Path(args.output),
                        plan_id=args.plan_id,
                        max_jobs=args.max_jobs,
                        expected_git_commit=args.expected_git_commit,
                    )
                )
            )
        else:
            result = run_h23_diagnostic(
                H23DiagnosticRunConfig(
                    plan_path=Path(args.plan),
                    results_root=Path(args.results_root),
                    plan_id=args.plan_id,
                    max_jobs=args.max_jobs,
                    resource_lock=Path(args.resource_lock),
                )
            )
    except (
        BenchmarkResourceLockError,
        H23DiagnosticError,
        H23DiagnosticBatchInfrastructureError,
        OSError,
        ValueError,
    ) as error:
        parser.exit(2, f"H23 diagnostic failed: {error}\n")
    print(canonical_json(result.to_payload()))
    return 0


__all__ = [
    "DEFAULT_PLAN_HOOKS",
    "DEFAULT_RUN_HOOKS",
    "H23DiagnosticBatchInfrastructureError",
    "H23DiagnosticError",
    "H23DiagnosticHooks",
    "H23DiagnosticRunConfig",
    "H23DiagnosticRunResult",
    "H23PlanBuildConfig",
    "H23PlanBuildHooks",
    "H23PlanBuildResult",
    "LoadedH23DiagnosticPlan",
    "build_h23_diagnostic_plan",
    "build_parser",
    "load_h23_diagnostic_plan",
    "main",
    "run_h23_diagnostic",
]


if __name__ == "__main__":
    raise SystemExit(main())
