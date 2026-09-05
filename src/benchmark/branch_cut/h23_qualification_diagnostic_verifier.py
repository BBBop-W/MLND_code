"""Independent, read-only verifier for the H23 qualification diagnostic.

This module never imports the diagnostic runner, the calibration runner, the
qualification runner, or Gurobi.  It reconstructs the frozen 24-job plan from
the independently verified calibration manifest and 5-second source run, then
recomputes run identity, checkpoint semantics, qualification decisions,
geometry attempts, fresh-model consistency, the 24-row ledger, and every
artifact self-hash.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
from typing import Any, Mapping, Sequence

from src.config import to_dict

from . import calibration_verifier as source_verifier
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
    LEGACY_IMPLEMENTATION_FILES,
    REGISTERED_GEOMETRY_POLICY_PAYLOAD,
    REGISTERED_NATIVE_SETTINGS_PAYLOAD,
    REGISTERED_POLICY_PAYLOAD,
)
from .hashing import canonical_json, canonical_sha256
from .qualification import (
    QualificationEvidence,
    QualificationPolicy,
    assess_qualification,
    objective_margin,
    service_tolerance,
)
from .qualification_geometry_start import (
    GeometryStartAttemptAudit,
    GeometryStartPolicy,
)


_PLAN_ARTIFACT_TYPE = "mlnd_branch_cut_h23_qualification_diagnostic_plan"
_RUN_ARTIFACT_TYPE = "mlnd_branch_cut_h23_qualification_diagnostic_run_identity"
_CHECKPOINT_ARTIFACT_TYPE = (
    "mlnd_branch_cut_h23_qualification_diagnostic_checkpoint"
)
_SUMMARY_ARTIFACT_TYPE = (
    "mlnd_branch_cut_h23_qualification_diagnostic_batch_summary"
)
_RUN_ID_RE = re.compile(r"^h23-qualification-diagnostic-run-([0-9a-f]{64})$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_UTC_RE = re.compile(
    r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$"
)
_NORMAL_SOLVER_STATUSES = frozenset({"OPTIMAL", "TIME_LIMIT", "INFEASIBLE"})
_NORMAL_RUNNER_REASONS = frozenset(
    {
        "geometry_start_fallback_failed",
        "no_objective_improving_incumbent",
        "no_positive_service_incumbent",
    }
)
_PHASE_TIMING_KEYS = {
    "witness_adapter_initialization_seconds",
    "baseline_extraction_seconds",
    "native_witness_solve_seconds",
    "temporary_constraint_add_seconds",
    "constrained_witness_solve_seconds",
    "geometry_start_planning_seconds",
    "geometry_start_installation_seconds",
    "geometry_start_solve_seconds",
    "geometry_start_cleanup_seconds",
    "complete_incumbent_capture_seconds",
    "temporary_constraint_removal_seconds",
    "witness_model_disposal_seconds",
    "fresh_adapter_initialization_seconds",
    "fresh_baseline_extraction_seconds",
    "complete_start_installation_seconds",
    "fresh_validation_solve_seconds",
    "fresh_model_disposal_seconds",
    "total_seconds",
}


class H23DiagnosticVerificationError(RuntimeError):
    """A plan or diagnostic result failed independent verification."""


def _fail(message: str) -> None:
    raise H23DiagnosticVerificationError(message)


def _require_mapping(value: object, *, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        _fail(f"{field_name} must be an object with string keys")
    return value


def _require_list(value: object, *, field_name: str) -> list[Any]:
    if not isinstance(value, list):
        _fail(f"{field_name} must be a list")
    return value


def _require_exact_keys(
    payload: Mapping[str, Any], expected: set[str], *, field_name: str
) -> None:
    if set(payload) != expected:
        missing = sorted(expected.difference(payload))
        extra = sorted(set(payload).difference(expected))
        _fail(f"{field_name} fields differ; missing={missing}, extra={extra}")


def _require_bool(value: object, *, field_name: str) -> bool:
    if type(value) is not bool:
        _fail(f"{field_name} must be boolean")
    return value


def _require_int(value: object, *, field_name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        _fail(f"{field_name} must be an integer >= {minimum}")
    return value


def _require_number(
    value: object, *, field_name: str, minimum: float | None = None
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(f"{field_name} must be numeric")
    number = float(value)
    if not math.isfinite(number) or (minimum is not None and number < minimum):
        _fail(f"{field_name} must be finite and >= {minimum}")
    return number


def _require_sha256(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        _fail(f"{field_name} must be a lowercase SHA256 digest")
    return value


def _require_utc(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or _UTC_RE.fullmatch(value) is None:
        _fail(f"{field_name} must be a second-resolution UTC timestamp")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _real_path(raw_path: str | Path, *, field_name: str, kind: str) -> Path:
    lexical = Path(raw_path).expanduser()
    if not lexical.is_absolute():
        lexical = Path.cwd() / lexical
    lexical = Path(os.path.abspath(lexical))
    current = Path(lexical.anchor)
    for component in lexical.parts[1:]:
        current /= component
        if current.is_symlink():
            _fail(f"{field_name} contains symbolic-link component {current}")
    try:
        resolved = lexical.resolve(strict=True)
    except OSError as error:
        _fail(f"{field_name} does not resolve: {error}")
    if resolved != lexical:
        _fail(f"{field_name} is not an already-resolved path")
    if kind == "file" and not resolved.is_file():
        _fail(f"{field_name} is not a regular file")
    if kind == "directory" and not resolved.is_dir():
        _fail(f"{field_name} is not a directory")
    return resolved


def _require_inside(path: Path, root: Path, *, field_name: str) -> None:
    try:
        path.relative_to(root)
    except ValueError:
        _fail(f"{field_name} escapes {root}")


def _read_canonical(path: Path, *, field_name: str) -> dict[str, Any]:
    try:
        raw = path.read_text(encoding="utf-8")
        payload = json.loads(raw)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        _fail(f"cannot read {field_name}: {error}")
    if not isinstance(payload, dict):
        _fail(f"{field_name} must contain one JSON object")
    try:
        expected = canonical_json(payload) + "\n"
    except (TypeError, ValueError) as error:
        _fail(f"{field_name} is not canonical-JSON-safe: {error}")
    if raw != expected:
        _fail(f"{field_name} is not canonical JSON")
    return payload


@dataclass(frozen=True, slots=True)
class H23DiagnosticVerificationRequest:
    plan_path: Path
    results_root: Path
    expected_plan_id: str
    expected_max_jobs: int
    expected_run_id: str
    expected_git_commit: str
    expected_implementation_sha256: str

    def __post_init__(self) -> None:
        if self.expected_plan_id != H23_DIAGNOSTIC_PLAN_ID:
            raise ValueError(f"expected_plan_id must be {H23_DIAGNOSTIC_PLAN_ID}")
        if self.expected_max_jobs != H23_DIAGNOSTIC_MAX_JOBS or isinstance(
            self.expected_max_jobs, bool
        ):
            raise ValueError("expected_max_jobs must be 24")
        if _RUN_ID_RE.fullmatch(self.expected_run_id) is None:
            raise ValueError("expected_run_id is not an H23 diagnostic run ID")
        if _COMMIT_RE.fullmatch(self.expected_git_commit) is None:
            raise ValueError("expected_git_commit must be a lowercase Git digest")
        if _SHA256_RE.fullmatch(self.expected_implementation_sha256) is None:
            raise ValueError("expected_implementation_sha256 must be SHA256")


@dataclass(frozen=True, slots=True)
class H23DiagnosticVerificationReport:
    status: str
    summary_path: Path
    plan_id: str
    plan_sha256: str
    diagnostic_run_id: str
    git_commit: str
    implementation_sha256: str
    attempted_job_count: int
    accepted_job_count: int
    rejected_attempted_job_count: int
    infrastructure_failure_count: int
    unattempted_job_count: int

    def to_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["summary_path"] = str(self.summary_path)
        return payload


@dataclass(frozen=True, slots=True)
class _VerifiedPlan:
    path: Path
    file_sha256: str
    plan_sha256: str
    payload: Mapping[str, Any]
    manifest: Any
    jobs: tuple[Mapping[str, Any], ...]


@dataclass(frozen=True, slots=True)
class _VerifiedCheckpoint:
    path: Path
    job: Mapping[str, Any]
    checkpoint_sha256: str
    status: str
    accepted: bool
    outcome: Mapping[str, Any] | None
    infrastructure_error: Mapping[str, Any] | None


def _verify_source_run(
    manifest_path: Path, source_root_value: object
) -> tuple[dict[str, Any], tuple[dict[str, Any], ...]]:
    if not isinstance(source_root_value, str):
        _fail("source run root must be a string")
    source_root = _real_path(
        source_root_value, field_name="source run root", kind="directory"
    )
    try:
        report = source_verifier.verify_development_calibration_results(
            source_verifier.CalibrationVerificationRequest(
                manifest_path=manifest_path,
                results_root=source_root,
                expected_run_id=FROZEN_SOURCE_RUN_ID,
                expected_manifest_sha256=FROZEN_MANIFEST_SHA256,
                expected_git_commit=FROZEN_BASE_COMMIT,
                expected_implementation_sha256=FROZEN_SOURCE_IMPLEMENTATION_SHA256,
            )
        )
    except (source_verifier.CalibrationVerificationError, OSError, ValueError) as error:
        _fail(f"source 5-second run failed independent verification: {error}")
    if (
        report.status != "completed"
        or report.selected_job_count != H23_DIAGNOSTIC_MAX_JOBS
        or report.attempted_job_count != H23_DIAGNOSTIC_MAX_JOBS
        or report.accepted_job_count != 0
        or report.infrastructure_failure_count != 0
    ):
        _fail("source 5-second run report differs from the frozen negative result")
    identity_path = _real_path(
        source_root / "run_identity.json", field_name="source identity", kind="file"
    )
    summary_path = _real_path(
        source_root / "batch_summary.json", field_name="source summary", kind="file"
    )
    if _sha256_file(identity_path) != FROZEN_SOURCE_RUN_IDENTITY_FILE_SHA256:
        _fail("source run identity file hash drifted")
    if _sha256_file(summary_path) != FROZEN_SOURCE_SUMMARY_FILE_SHA256:
        _fail("source summary file hash drifted")
    identity = _read_canonical(identity_path, field_name="source identity")
    summary = _read_canonical(summary_path, field_name="source summary")
    if (
        identity.get("calibration_run_id") != FROZEN_SOURCE_RUN_ID
        or identity.get("run_identity_sha256")
        != FROZEN_SOURCE_RUN_IDENTITY_SHA256
        or summary.get("summary_sha256") != FROZEN_SOURCE_SUMMARY_SHA256
    ):
        _fail("source run identity or summary self-hash drifted")
    records = _require_list(summary.get("checkpoints"), field_name="source checkpoints")
    if len(records) != H23_DIAGNOSTIC_MAX_JOBS:
        _fail("source run must contain exactly 24 checkpoints")
    checkpoints_root = _real_path(
        source_root / "checkpoints", field_name="source checkpoints root", kind="directory"
    )
    bindings: list[dict[str, Any]] = []
    observed: set[Path] = set()
    for index, (frozen, raw_record) in enumerate(
        zip(FROZEN_SOURCE_CHECKPOINTS, records, strict=True)
    ):
        job_id, filename, file_sha256, self_sha256 = frozen
        record = _require_mapping(
            raw_record, field_name=f"source checkpoint record {index}"
        )
        path = _real_path(
            checkpoints_root / filename,
            field_name=f"source checkpoint {index}",
            kind="file",
        )
        _require_inside(path, checkpoints_root, field_name=f"source checkpoint {index}")
        if path in observed:
            _fail("source checkpoint path duplicated")
        observed.add(path)
        if (
            record.get("job_id") != job_id
            or record.get("job_order_index") != index
            or record.get("execution_attempt_index") != index
            or record.get("path") != str(path)
            or record.get("checkpoint_sha256") != self_sha256
            or record.get("status") != "completed"
            or record.get("accepted") is not False
            or _sha256_file(path) != file_sha256
        ):
            _fail(f"source checkpoint {index} binding drifted")
        checkpoint = _read_canonical(path, field_name=f"source checkpoint {index}")
        if checkpoint.get("checkpoint_sha256") != self_sha256:
            _fail(f"source checkpoint {index} self-hash drifted")
        bindings.append(
            {
                "source_job_id": job_id,
                "relative_path": f"checkpoints/{filename}",
                "file_sha256": file_sha256,
                "checkpoint_sha256": self_sha256,
            }
        )
    directory_files = {
        _real_path(
            path, field_name=f"source checkpoint entry {path.name}", kind="file"
        )
        for path in checkpoints_root.iterdir()
    }
    if directory_files != observed:
        _fail("source checkpoint directory contains missing or extra files")
    return (
        {
            "root": str(source_root),
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
        },
        tuple(bindings),
    )


def _recompute_jobs(
    manifest: Any, bindings: Sequence[Mapping[str, Any]]
) -> tuple[Mapping[str, Any], ...]:
    source_jobs = tuple(
        job
        for job in manifest.jobs
        if job["scale"] == "small" and job["development_seed_stage"] == 0
    )
    if (
        len(source_jobs) != H23_DIAGNOSTIC_MAX_JOBS
        or tuple(job["job_order_index"] for job in source_jobs)
        != tuple(range(H23_DIAGNOSTIC_MAX_JOBS))
        or tuple(job["profile_order_index"] for job in source_jobs)
        != tuple(range(H23_DIAGNOSTIC_MAX_JOBS))
        or tuple(job["job_id"] for job in source_jobs)
        != tuple(item[0] for item in FROZEN_SOURCE_CHECKPOINTS)
    ):
        _fail("manifest does not reconstruct the frozen 24-job order")
    jobs: list[Mapping[str, Any]] = []
    for index, (source_job, binding) in enumerate(
        zip(source_jobs, bindings, strict=True)
    ):
        profile = manifest.profiles[source_job["profile_order_index"]].solve_profile()
        model_config = profile.effective_model_config()
        contract = {
            "qualification_case_sha256": source_job["qualification_case_sha256"],
            "effective_model_config_sha256": source_job[
                "effective_model_config_sha256"
            ],
            "demand_scale": profile.demand_scale,
            "locker_fixed_cost_per_hour": profile.locker_fixed_cost_per_hour,
            "display_origin_choices": profile.display_origin_choices,
            "general_corridor_choices": profile.general_corridor_choices,
            "screening_is_exact": False,
        }
        diagnostic_identity = {
            "plan_id": H23_DIAGNOSTIC_PLAN_ID,
            "diagnostic_job_order_index": index,
            "source_job_id": source_job["job_id"],
            "source_checkpoint_file_sha256": binding["file_sha256"],
            "source_checkpoint_sha256": binding["checkpoint_sha256"],
            "qualification_case_sha256": source_job["qualification_case_sha256"],
        }
        jobs.append(
            {
                "diagnostic_job_order_index": index,
                "diagnostic_job_id": (
                    "h23-qualification-diagnostic-job-"
                    + canonical_sha256(diagnostic_identity)
                ),
                "source_job": dict(source_job),
                "source_checkpoint": dict(binding),
                "physical_instance_directory": str(
                    manifest.physical_directories[
                        source_job["physical_instance_id"]
                    ]
                ),
                "qualification_solve_profile": profile.to_payload(),
                "effective_model_config": to_dict(model_config),
                "qualification_model_contract": contract,
            }
        )
    return tuple(jobs)


def _verify_plan(
    plan_path: Path, *, expected_plan_id: str, expected_max_jobs: int
) -> _VerifiedPlan:
    if expected_plan_id != H23_DIAGNOSTIC_PLAN_ID:
        _fail("explicit plan ID is not the frozen H23 diagnostic")
    if expected_max_jobs != H23_DIAGNOSTIC_MAX_JOBS:
        _fail("explicit max-jobs is not 24")
    path = _real_path(plan_path, field_name="plan", kind="file")
    payload = _read_canonical(path, field_name="diagnostic plan")
    _require_exact_keys(
        payload,
        {
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
        },
        field_name="diagnostic plan",
    )
    fixed = {
        "artifact_type": _PLAN_ARTIFACT_TYPE,
        "artifact_schema_version": 1,
        "plan_schema": H23_DIAGNOSTIC_PLAN_SCHEMA,
        "plan_id": H23_DIAGNOSTIC_PLAN_ID,
        "development_only": True,
        "eligible_for_formal_pool": False,
        "formal_pool": False,
        "may_lock_profiles": False,
        "algorithm_comparison_performed": False,
        "mode": "MX",
        "max_jobs": H23_DIAGNOSTIC_MAX_JOBS,
    }
    if any(payload.get(key) != value for key, value in fixed.items()):
        _fail("plan identity or development-only flags drifted")
    if payload["scope"] != {
        "scale": "small",
        "development_seed_stage": 0,
        "all_manifest_profiles": True,
        "profile_subset_allowed": False,
        "job_order": "frozen_manifest_order",
    }:
        _fail("plan scope drifted")
    if payload["qualification_policy"] != REGISTERED_POLICY_PAYLOAD:
        _fail("plan qualification policy drifted")
    if payload["native_settings"] != REGISTERED_NATIVE_SETTINGS_PAYLOAD:
        _fail("plan native settings drifted")
    if payload["geometry_start_policy"] != REGISTERED_GEOMETRY_POLICY_PAYLOAD:
        _fail("plan geometry policy drifted")
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
        _fail("plan execution contract drifted")
    plan_sha = _require_sha256(payload["plan_sha256"], field_name="plan_sha256")
    if plan_sha != canonical_sha256(
        {
            key: value
            for key, value in payload.items()
            if key
            not in {"artifact_type", "artifact_schema_version", "plan_sha256"}
        }
    ):
        _fail("plan self-hash mismatch")

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
        _fail("plan does not bind the frozen manifest hashes")
    try:
        manifest = source_verifier._verify_manifest(  # noqa: SLF001
            Path(str(source_manifest["path"])), FROZEN_MANIFEST_SHA256
        )
    except (source_verifier.CalibrationVerificationError, OSError, ValueError) as error:
        _fail(f"manifest failed independent reconstruction: {error}")
    observed_manifest = {
        "path": str(manifest.path),
        "manifest_sha256": manifest.manifest_sha256,
        "manifest_file_sha256": manifest.manifest_file_sha256,
        "raw_path": str(manifest.raw_path),
        "raw_sha256": manifest.raw_sha256,
        "raw_size_bytes": manifest.payload["raw_input"]["size_bytes"],
    }
    if dict(source_manifest) != observed_manifest:
        _fail("manifest/raw path, size, or hash binding drifted")
    source_run = _require_mapping(payload["source_5s_run"], field_name="source_5s_run")
    observed_source, bindings = _verify_source_run(
        manifest.path, source_run.get("root")
    )
    if canonical_json(source_run) != canonical_json(observed_source):
        _fail("source 5-second run binding drifted")
    jobs = _recompute_jobs(manifest, bindings)
    if canonical_json(payload["jobs"]) != canonical_json(jobs):
        _fail("24-job order, profile, config, or checkpoint binding drifted")
    required = _require_mapping(
        payload["required_execution_environment"], field_name="required environment"
    )
    _require_exact_keys(
        required,
        {
            "repository_root",
            "git_commit",
            "clean_status_includes_untracked",
            "implementation",
        },
        field_name="required environment",
    )
    if (
        not isinstance(required["repository_root"], str)
        or not required["repository_root"]
        or not isinstance(required["git_commit"], str)
        or _COMMIT_RE.fullmatch(required["git_commit"]) is None
        or required["clean_status_includes_untracked"] is not True
    ):
        _fail("required execution repository/commit/status is invalid")
    implementation = _require_mapping(
        required["implementation"], field_name="required implementation"
    )
    _require_exact_keys(
        implementation,
        {"ordered_file_sha256", "implementation_sha256"},
        field_name="required implementation",
    )
    files = _require_mapping(
        implementation["ordered_file_sha256"], field_name="implementation files"
    )
    if not any(
        set(files) == set(registered)
        for registered in (IMPLEMENTATION_FILES, LEGACY_IMPLEMENTATION_FILES)
    ):
        _fail("implementation file set drifted")
    for name, digest in files.items():
        _require_sha256(digest, field_name=f"implementation hash {name}")
    if implementation["implementation_sha256"] != canonical_sha256(files):
        _fail("implementation aggregate hash does not recompute")
    return _VerifiedPlan(
        path=path,
        file_sha256=_sha256_file(path),
        plan_sha256=plan_sha,
        payload=payload,
        manifest=manifest,
        jobs=jobs,
    )


def _stable_environment(environment: Mapping[str, Any]) -> dict[str, Any]:
    git_payload = dict(
        _require_mapping(environment.get("git"), field_name="environment.git")
    )
    for key in (
        "worktree_dirty",
        "status_includes_untracked",
        "status_lines",
        "status_sha256",
    ):
        git_payload.pop(key, None)
    return {**dict(environment), "git": git_payload}


def _verify_process_audit(
    value: object, *, field_name: str, completed: bool
) -> Mapping[str, Any]:
    audit = _require_mapping(value, field_name=field_name)
    normal_keys = {"process_table_sha256", "conflicting_processes", "valid"}
    error_keys = normal_keys | {"audit_error"}
    if set(audit) not in (normal_keys, error_keys):
        _fail(f"{field_name} fields are inconsistent")
    digest = audit["process_table_sha256"]
    if digest is not None:
        _require_sha256(digest, field_name=f"{field_name}.process_table_sha256")
    conflicts = _require_list(
        audit["conflicting_processes"], field_name=f"{field_name}.conflicts"
    )
    if any(not isinstance(item, str) or not item for item in conflicts):
        _fail(f"{field_name} conflict entries must be nonempty strings")
    valid = _require_bool(audit["valid"], field_name=f"{field_name}.valid")
    if valid != (not conflicts and "audit_error" not in audit):
        _fail(f"{field_name} validity does not match conflicts/error")
    if completed and (not valid or digest is None):
        _fail(f"{field_name} is invalid for a completed checkpoint")
    if "audit_error" in audit and (
        not isinstance(audit["audit_error"], str) or not audit["audit_error"]
    ):
        _fail(f"{field_name}.audit_error must be nonempty")
    return audit


def _verify_run_identity(
    path: Path,
    *,
    plan: _VerifiedPlan,
    request: H23DiagnosticVerificationRequest,
) -> tuple[Mapping[str, Any], str]:
    payload = _read_canonical(path, field_name="diagnostic run identity")
    _require_exact_keys(
        payload,
        {
            "artifact_type",
            "artifact_schema_version",
            "diagnostic_run_id",
            "run_identity",
            "run_identity_sha256",
        },
        field_name="diagnostic run identity",
    )
    if (
        payload["artifact_type"] != _RUN_ARTIFACT_TYPE
        or payload["artifact_schema_version"] != 1
    ):
        _fail("diagnostic run identity artifact type/schema drifted")
    identity = _require_mapping(payload["run_identity"], field_name="run_identity")
    _require_exact_keys(
        identity,
        {
            "run_schema",
            "plan_id",
            "plan_path",
            "plan_sha256",
            "plan_file_sha256",
            "development_only",
            "eligible_for_formal_pool",
            "formal_pool",
            "may_lock_profiles",
            "algorithm_comparison_performed",
            "mode",
            "max_jobs",
            "concurrency",
            "qualification_policy",
            "native_settings",
            "geometry_start_policy",
            "jobs",
            "source_manifest_sha256",
            "source_calibration_run_id",
            "environment",
            "environment_sha256",
            "stable_environment_sha256",
            "git_commit",
            "implementation_sha256",
            "initial_process_audit",
        },
        field_name="run_identity",
    )
    fixed = {
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
        "qualification_policy": REGISTERED_POLICY_PAYLOAD,
        "native_settings": REGISTERED_NATIVE_SETTINGS_PAYLOAD,
        "geometry_start_policy": REGISTERED_GEOMETRY_POLICY_PAYLOAD,
        "jobs": list(plan.jobs),
        "source_manifest_sha256": FROZEN_MANIFEST_SHA256,
        "source_calibration_run_id": FROZEN_SOURCE_RUN_ID,
    }
    if any(canonical_json(identity[key]) != canonical_json(value) for key, value in fixed.items()):
        _fail("run identity fixed plan/settings/jobs fields drifted")
    required = plan.payload["required_execution_environment"]
    if (
        identity["git_commit"] != required["git_commit"]
        or identity["git_commit"] != request.expected_git_commit
        or identity["implementation_sha256"]
        != required["implementation"]["implementation_sha256"]
        or identity["implementation_sha256"]
        != request.expected_implementation_sha256
    ):
        _fail("run commit or implementation does not match plan and explicit pins")
    environment = _require_mapping(identity["environment"], field_name="environment")
    _require_exact_keys(
        environment,
        {"git", "implementation", "python", "platform", "gurobi_version"},
        field_name="environment",
    )
    git_payload = _require_mapping(environment["git"], field_name="environment.git")
    _require_exact_keys(
        git_payload,
        {
            "repository_root",
            "commit",
            "worktree_dirty",
            "status_includes_untracked",
            "status_lines",
            "status_sha256",
        },
        field_name="environment.git",
    )
    if (
        git_payload["repository_root"] != required["repository_root"]
        or git_payload["commit"] != required["git_commit"]
        or git_payload["worktree_dirty"] is not False
        or git_payload["status_includes_untracked"] is not True
        or git_payload["status_lines"] != []
        or git_payload["status_sha256"] != canonical_sha256(())
        or environment["implementation"] != required["implementation"]
        or not isinstance(environment["gurobi_version"], str)
        or not environment["gurobi_version"]
    ):
        _fail("run environment is not the planned clean exact implementation")
    environment_sha = _require_sha256(
        identity["environment_sha256"], field_name="environment_sha256"
    )
    stable_sha = _require_sha256(
        identity["stable_environment_sha256"],
        field_name="stable_environment_sha256",
    )
    if environment_sha != canonical_sha256(environment):
        _fail("environment_sha256 does not hash environment")
    if stable_sha != canonical_sha256(_stable_environment(environment)):
        _fail("stable environment hash does not recompute")
    _verify_process_audit(
        identity["initial_process_audit"],
        field_name="initial_process_audit",
        completed=True,
    )
    run_sha = _require_sha256(
        payload["run_identity_sha256"], field_name="run_identity_sha256"
    )
    if run_sha != canonical_sha256(identity):
        _fail("run identity self-hash mismatch")
    run_id = "h23-qualification-diagnostic-run-" + run_sha
    if payload["diagnostic_run_id"] != run_id or run_id != request.expected_run_id:
        _fail("run ID does not match its hash and explicit pin")
    return identity, run_sha


def _normalize_nonfinite(value: object) -> object:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: _normalize_nonfinite(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize_nonfinite(item) for item in value]
    return value


def _evidence_from_payload(payload: Mapping[str, Any]) -> QualificationEvidence:
    values = dict(payload)
    for key in (
        "courier_only_objective",
        "witness_objective",
        "total_demand_per_hour",
        "witness_service_per_hour",
    ):
        if values[key] is None:
            values[key] = math.nan
    try:
        return QualificationEvidence(**values)
    except (TypeError, ValueError) as error:
        _fail(f"qualification evidence is invalid: {error}")


def _policy() -> QualificationPolicy:
    payload = REGISTERED_POLICY_PAYLOAD
    return QualificationPolicy(
        optimality_tolerance=payload["optimality_tolerance"],
        minimum_savings_fraction=payload["minimum_savings_fraction"],
        minimum_service_fraction=payload["minimum_service_fraction"],
        minimum_h_open_fraction=payload["minimum_h_open_fraction"],
        maximum_h_open_fraction=payload["maximum_h_open_fraction"],
        large_scale_open_h_target=tuple(payload["large_scale_open_h_target"]),
    )


def _geometry_policy() -> GeometryStartPolicy:
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


def _verify_audit(
    value: object, *, field_name: str, contract: Mapping[str, Any]
) -> Mapping[str, Any]:
    audit = _require_mapping(value, field_name=field_name)
    _require_exact_keys(
        audit,
        {
            "effective_model_config_sha256",
            "demand_scale",
            "locker_fixed_cost_per_hour",
            "display_origin_choices",
            "general_corridor_choices",
            "screening_is_exact",
            "model_fingerprint",
        },
        field_name=field_name,
    )
    for key in (
        "effective_model_config_sha256",
        "demand_scale",
        "locker_fixed_cost_per_hour",
        "display_origin_choices",
        "general_corridor_choices",
        "screening_is_exact",
    ):
        if audit[key] != contract[key]:
            _fail(f"{field_name} differs from qualification contract")
    fingerprint = audit["model_fingerprint"]
    if fingerprint is not None and (
        isinstance(fingerprint, bool) or not isinstance(fingerprint, int)
    ):
        _fail(f"{field_name}.model_fingerprint must be integer or null")
    return audit


def _verify_outcome(
    value: object,
    *,
    job: Mapping[str, Any],
    run_id: str,
) -> tuple[Mapping[str, Any], bool]:
    outcome = _require_mapping(value, field_name="outcome")
    _require_exact_keys(
        outcome,
        {
            "accepted",
            "rejection_reasons",
            "evidence",
            "decision",
            "required_service_per_hour",
            "required_objective_upper_bound",
            "temporary_constraint_used",
            "temporary_constraint_removed",
            "fresh_model_validation_performed",
            "witness_status",
            "validation_status",
            "captured_variable_count",
            "exception_messages",
            "feasibility_objective_used",
            "expected_model_contract",
            "witness_model_audit",
            "fresh_model_audit",
            "geometry_start_policy",
            "geometry_start_attempts",
            "geometry_start_selected_plan_sha256",
            "phase_timings",
        },
        field_name="outcome",
    )
    accepted = _require_bool(outcome["accepted"], field_name="outcome.accepted")
    reasons = _require_list(outcome["rejection_reasons"], field_name="reasons")
    if (
        any(not isinstance(reason, str) or not reason for reason in reasons)
        or len(reasons) != len(set(reasons))
        or accepted != (not reasons)
    ):
        _fail("outcome accepted/rejection state is invalid")
    contract = job["qualification_model_contract"]
    if outcome["expected_model_contract"] != contract:
        _fail("outcome model contract drifted")
    witness_audit = _verify_audit(
        outcome["witness_model_audit"],
        field_name="witness_model_audit",
        contract=contract,
    )
    fresh_performed = _require_bool(
        outcome["fresh_model_validation_performed"],
        field_name="fresh_model_validation_performed",
    )
    fresh_value = outcome["fresh_model_audit"]
    if fresh_performed:
        fresh_audit = _verify_audit(
            fresh_value, field_name="fresh_model_audit", contract=contract
        )
        if (
            witness_audit["model_fingerprint"] is None
            or fresh_audit["model_fingerprint"] is None
            or witness_audit["model_fingerprint"]
            != fresh_audit["model_fingerprint"]
        ):
            _fail("fresh/witness model fingerprints differ")
    elif fresh_value is not None:
        _fail("fresh model audit exists without fresh validation")

    evidence_payload = _require_mapping(outcome["evidence"], field_name="evidence")
    _require_exact_keys(
        evidence_payload,
        {
            "case_id",
            "scale",
            "courier_only_objective",
            "witness_objective",
            "total_demand_per_hour",
            "witness_service_per_hour",
            "open_facilities",
            "positive_cost_open_facilities",
            "candidate_h",
            "open_h",
            "witness_feasible",
            "qualification_run_id",
            "instance_sha256",
            "active_h",
            "active_h_threshold_per_hour",
            "qualification_case_sha256",
            "effective_model_config_sha256",
            "screening_is_exact",
        },
        field_name="evidence",
    )
    source_job = job["source_job"]
    expected_identity = {
        "case_id": job["diagnostic_job_id"],
        "scale": "small",
        "qualification_run_id": run_id,
        "instance_sha256": source_job["instance_sha256"],
        "qualification_case_sha256": source_job["qualification_case_sha256"],
        "effective_model_config_sha256": source_job[
            "effective_model_config_sha256"
        ],
        "screening_is_exact": False,
    }
    if any(evidence_payload[key] != value for key, value in expected_identity.items()):
        _fail("qualification evidence identity differs from diagnostic job")
    evidence = _evidence_from_payload(evidence_payload)
    policy = _policy()
    expected_decision = _normalize_nonfinite(
        assess_qualification(evidence, policy).to_payload()
    )
    decision = _require_mapping(outcome["decision"], field_name="decision")
    if canonical_json(decision) != canonical_json(expected_decision):
        _fail("stored decision differs from independent assessment")
    decision_reasons = _require_list(
        decision["rejection_reasons"], field_name="decision reasons"
    )
    cursor = iter(reasons)
    if not all(any(candidate == reason for candidate in cursor) for reason in decision_reasons):
        _fail("decision reasons are not an ordered outcome subset")
    if not set(reasons).difference(decision_reasons).issubset(
        _NORMAL_RUNNER_REASONS
    ):
        _fail("normal outcome includes infrastructure-level reasons")
    if outcome["witness_status"] not in _NORMAL_SOLVER_STATUSES:
        _fail("outcome witness status is abnormal")
    if fresh_performed:
        if outcome["validation_status"] not in _NORMAL_SOLVER_STATUSES:
            _fail("fresh validation status is abnormal")
    elif outcome["validation_status"] != "NOT_RUN":
        _fail("validation status exists without fresh validation")
    if outcome["exception_messages"] != []:
        _fail("completed outcome contains exception messages")
    captured = _require_int(
        outcome["captured_variable_count"], field_name="captured_variable_count"
    )
    if evidence.witness_feasible and not fresh_performed:
        _fail("feasible witness lacks fresh original-model validation")
    if accepted and (
        not fresh_performed
        or decision["accepted"] is not True
        or evidence_payload["witness_feasible"] is not True
        or captured <= 0
    ):
        _fail("accepted outcome lacks a fresh validated complete witness")
    if fresh_performed and captured <= 0:
        _fail("fresh validation was attempted without a captured complete incumbent")

    required_service = _require_number(
        outcome["required_service_per_hour"],
        field_name="required_service_per_hour",
        minimum=0.0,
    )
    demand = evidence.total_demand_per_hour
    expected_service = (
        max(service_tolerance(demand), policy.minimum_service_fraction * demand)
        if math.isfinite(demand) and demand >= 0.0
        else 1e-3
    )
    if required_service <= 0.0 or required_service != expected_service:
        _fail("required service differs from policy and demand")
    courier = evidence.courier_only_objective
    expected_upper: float | None = None
    if math.isfinite(courier):
        improvement = max(
            objective_margin(courier, policy.optimality_tolerance),
            policy.minimum_savings_fraction * courier if courier > 0.0 else 0.0,
        )
        expected_upper = courier - improvement
    if outcome["required_objective_upper_bound"] != expected_upper:
        _fail("required objective upper bound differs from policy")
    temporary_used = _require_bool(
        outcome["temporary_constraint_used"], field_name="temporary_constraint_used"
    )
    temporary_removed = outcome["temporary_constraint_removed"]
    if (temporary_used and temporary_removed is not True) or (
        not temporary_used and temporary_removed is not None
    ):
        _fail("temporary constraint cleanup state is inconsistent")
    _require_bool(
        outcome["feasibility_objective_used"],
        field_name="feasibility_objective_used",
    )

    geometry_policy = _geometry_policy()
    if outcome["geometry_start_policy"] != REGISTERED_GEOMETRY_POLICY_PAYLOAD:
        _fail("outcome geometry policy drifted")
    raw_attempts = _require_list(
        outcome["geometry_start_attempts"], field_name="geometry attempts"
    )
    try:
        attempts = tuple(
            GeometryStartAttemptAudit.from_payload(item) for item in raw_attempts
        )
    except (TypeError, ValueError) as error:
        _fail(f"geometry attempt audit is invalid: {error}")
    schedule = geometry_policy.schedule_for("small", evidence.candidate_h)
    if len(attempts) > len(schedule) or tuple(
        attempt.plan.target_support_size for attempt in attempts
    ) != schedule[: len(attempts)]:
        _fail("geometry attempts are not the registered schedule prefix")
    full = [attempt for attempt in attempts if attempt.full_witness_found]
    if len(full) > 1 or (full and full[0] is not attempts[-1]):
        _fail("geometry full-witness attempt is not unique and terminal")
    for attempt in attempts:
        if (
            attempt.per_attempt_seconds != geometry_policy.per_attempt_seconds
            or attempt.start_node_limit != geometry_policy.start_node_limit
        ):
            _fail("geometry attempt seconds/StartNodeLimit drifted")
        if attempt.plan.prechecks_passed:
            if (
                attempt.install is None
                or not attempt.install.safety_preserved
                or attempt.solve_status not in _NORMAL_SOLVER_STATUSES
            ):
                _fail("completed geometry attempt lacks safe normal solve evidence")
        elif attempt.solve_status != "NOT_RUN":
            _fail("failed geometry precheck claims a solve")
    selected = outcome["geometry_start_selected_plan_sha256"]
    if full:
        if selected != full[0].plan.plan_sha256:
            _fail("selected geometry plan hash differs from terminal witness")
    elif selected is not None:
        _fail("geometry plan selected without a full witness")
    if (
        captured == 0
        and expected_upper is not None
        and math.isfinite(demand)
        and demand >= 0.0
        and len(attempts) != len(schedule)
    ):
        _fail("no-witness outcome did not exhaust all geometry supports")

    timings = _require_mapping(outcome["phase_timings"], field_name="phase_timings")
    _require_exact_keys(timings, _PHASE_TIMING_KEYS, field_name="phase_timings")
    for key, timing in timings.items():
        _require_number(timing, field_name=f"phase_timings.{key}", minimum=0.0)
    return outcome, accepted


def _verify_checkpoint(
    path: Path,
    *,
    job: Mapping[str, Any],
    run_identity: Mapping[str, Any],
    run_sha: str,
    attempt_index: int,
) -> _VerifiedCheckpoint:
    payload = _read_canonical(path, field_name=f"checkpoint {attempt_index}")
    _require_exact_keys(
        payload,
        {
            "artifact_type",
            "artifact_schema_version",
            "checkpoint_schema",
            "development_only",
            "eligible_for_formal_pool",
            "formal_pool",
            "may_lock_profiles",
            "contains_algorithm_comparison_results",
            "identity",
            "result",
            "checkpoint_sha256",
        },
        field_name=f"checkpoint {attempt_index}",
    )
    fixed = {
        "artifact_type": _CHECKPOINT_ARTIFACT_TYPE,
        "artifact_schema_version": 1,
        "checkpoint_schema": H23_DIAGNOSTIC_CHECKPOINT_SCHEMA,
        "development_only": True,
        "eligible_for_formal_pool": False,
        "formal_pool": False,
        "may_lock_profiles": False,
        "contains_algorithm_comparison_results": False,
    }
    if any(payload[key] != value for key, value in fixed.items()):
        _fail(f"checkpoint {attempt_index} development schema drifted")
    identity = _require_mapping(payload["identity"], field_name="checkpoint identity")
    _require_exact_keys(
        identity,
        {
            "diagnostic_run_id",
            "run_identity_sha256",
            "plan_id",
            "plan_sha256",
            "plan_file_sha256",
            "environment_sha256",
            "git_commit",
            "execution_attempt_index",
            "job",
            "process_audit",
            "physical_instance_directory",
            "qualification_solve_profile",
            "effective_model_config",
            "qualification_model_contract",
        },
        field_name="checkpoint identity",
    )
    result = _require_mapping(payload["result"], field_name="checkpoint result")
    _require_exact_keys(
        result,
        {"status", "completed_at_utc", "infrastructure_error", "outcome"},
        field_name="checkpoint result",
    )
    status = result["status"]
    if status not in {"completed", "infrastructure_failure"}:
        _fail("checkpoint status is invalid")
    process_audit = _verify_process_audit(
        identity["process_audit"],
        field_name="checkpoint process audit",
        completed=status == "completed",
    )
    expected_identity = {
        "diagnostic_run_id": "h23-qualification-diagnostic-run-" + run_sha,
        "run_identity_sha256": run_sha,
        "plan_id": H23_DIAGNOSTIC_PLAN_ID,
        "plan_sha256": run_identity["plan_sha256"],
        "plan_file_sha256": run_identity["plan_file_sha256"],
        "environment_sha256": run_identity["environment_sha256"],
        "git_commit": run_identity["git_commit"],
        "execution_attempt_index": attempt_index,
        "job": dict(job),
        "process_audit": dict(process_audit),
        "physical_instance_directory": job["physical_instance_directory"],
        "qualification_solve_profile": job["qualification_solve_profile"],
        "effective_model_config": job["effective_model_config"],
        "qualification_model_contract": job["qualification_model_contract"],
    }
    if canonical_json(identity) != canonical_json(expected_identity):
        _fail(f"checkpoint {attempt_index} identity drifted")
    checkpoint_sha = _require_sha256(
        payload["checkpoint_sha256"], field_name="checkpoint_sha256"
    )
    if checkpoint_sha != canonical_sha256(
        {"identity": dict(identity), "result": dict(result)}
    ):
        _fail(f"checkpoint {attempt_index} self-hash mismatch")
    _require_utc(result["completed_at_utc"], field_name="checkpoint completed_at_utc")
    if status == "completed":
        if result["infrastructure_error"] is not None:
            _fail("completed checkpoint contains infrastructure error")
        outcome, accepted = _verify_outcome(
            result["outcome"],
            job=job,
            run_id="h23-qualification-diagnostic-run-" + run_sha,
        )
        infrastructure_error = None
    else:
        if result["outcome"] is not None:
            _fail("infrastructure checkpoint contains an outcome")
        infrastructure_error = _require_mapping(
            result["infrastructure_error"], field_name="infrastructure error"
        )
        _require_exact_keys(
            infrastructure_error,
            {"stage", "type", "message", "qualification_infrastructure_error"},
            field_name="infrastructure error",
        )
        if any(
            not isinstance(infrastructure_error[key], str)
            or not infrastructure_error[key]
            for key in ("stage", "type", "message")
        ):
            _fail("infrastructure error text fields must be nonempty")
        _require_bool(
            infrastructure_error["qualification_infrastructure_error"],
            field_name="qualification_infrastructure_error",
        )
        outcome = None
        accepted = False
    return _VerifiedCheckpoint(
        path=path,
        job=job,
        checkpoint_sha256=checkpoint_sha,
        status=str(status),
        accepted=accepted,
        outcome=outcome,
        infrastructure_error=infrastructure_error,
    )


def _checkpoint_record(
    checkpoint: _VerifiedCheckpoint, attempt_index: int
) -> dict[str, Any]:
    job = checkpoint.job
    return {
        "diagnostic_job_id": job["diagnostic_job_id"],
        "diagnostic_job_order_index": job["diagnostic_job_order_index"],
        "execution_attempt_index": attempt_index,
        "path": str(checkpoint.path),
        "checkpoint_sha256": checkpoint.checkpoint_sha256,
        "status": checkpoint.status,
        "accepted": checkpoint.accepted,
    }


def _ledger(
    jobs: Sequence[Mapping[str, Any]], checkpoints: Sequence[_VerifiedCheckpoint]
) -> list[dict[str, Any]]:
    by_id = {
        checkpoint.job["diagnostic_job_id"]: checkpoint
        for checkpoint in checkpoints
    }
    aborted = bool(checkpoints and checkpoints[-1].status == "infrastructure_failure")
    records: list[dict[str, Any]] = []
    for job in jobs:
        job_id = job["diagnostic_job_id"]
        checkpoint = by_id.get(job_id)
        if checkpoint is None:
            records.append(
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
        else:
            records.append(
                {
                    "diagnostic_job_order_index": job[
                        "diagnostic_job_order_index"
                    ],
                    "diagnostic_job_id": job_id,
                    "source_job_id": job["source_job"]["job_id"],
                    "profile_id": job["source_job"]["profile_id"],
                    "attempted": True,
                    "accepted": checkpoint.accepted,
                    "not_attempted_reason": None,
                    "checkpoint_path": str(checkpoint.path),
                    "checkpoint_sha256": checkpoint.checkpoint_sha256,
                    "infrastructure_failure": checkpoint.infrastructure_error,
                    "outcome": checkpoint.outcome,
                }
            )
    return records


def _verify_summary(
    path: Path,
    *,
    plan: _VerifiedPlan,
    run_sha: str,
    checkpoints: Sequence[_VerifiedCheckpoint],
) -> Mapping[str, Any]:
    payload = _read_canonical(path, field_name="diagnostic summary")
    if (
        payload.get("artifact_type") != _SUMMARY_ARTIFACT_TYPE
        or payload.get("artifact_schema_version") != 1
        or payload.get("summary_schema") != H23_DIAGNOSTIC_SUMMARY_SCHEMA
    ):
        _fail("diagnostic summary artifact/schema drifted")
    summary_sha = _require_sha256(
        payload.get("summary_sha256"), field_name="summary_sha256"
    )
    entry = {
        key: value
        for key, value in payload.items()
        if key not in {"artifact_type", "artifact_schema_version", "summary_sha256"}
    }
    if summary_sha != canonical_sha256(entry):
        _fail("diagnostic summary self-hash mismatch")
    checkpoint_records = [
        _checkpoint_record(checkpoint, index)
        for index, checkpoint in enumerate(checkpoints)
    ]
    ledger = _ledger(plan.jobs, checkpoints)
    accepted_count = sum(checkpoint.accepted for checkpoint in checkpoints)
    rejected_count = sum(
        checkpoint.status == "completed" and not checkpoint.accepted
        for checkpoint in checkpoints
    )
    infrastructure_count = sum(
        checkpoint.status == "infrastructure_failure" for checkpoint in checkpoints
    )
    common = {
        "summary_schema": H23_DIAGNOSTIC_SUMMARY_SCHEMA,
        "plan_id": H23_DIAGNOSTIC_PLAN_ID,
        "development_only": True,
        "eligible_for_formal_pool": False,
        "formal_pool": False,
        "may_lock_profiles": False,
        "diagnostic_run_id": "h23-qualification-diagnostic-run-" + run_sha,
        "run_identity_sha256": run_sha,
        "plan_sha256": plan.plan_sha256,
    }
    status = payload.get("status")
    completed_at = _require_utc(
        payload.get("completed_at_utc"), field_name="summary completed_at_utc"
    )
    if status == "completed":
        if (
            len(checkpoints) != H23_DIAGNOSTIC_MAX_JOBS
            or any(checkpoint.status != "completed" for checkpoint in checkpoints)
        ):
            _fail("completed summary does not contain 24 completed checkpoints")
        expected = {
            **common,
            "status": "completed",
            "diagnostic_conclusion": "development_diagnostic_completed_24_profiles",
            "max_jobs": H23_DIAGNOSTIC_MAX_JOBS,
            "attempted_job_count": H23_DIAGNOSTIC_MAX_JOBS,
            "completed_job_count": H23_DIAGNOSTIC_MAX_JOBS,
            "accepted_job_count": accepted_count,
            "rejected_attempted_job_count": rejected_count,
            "infrastructure_failure_count": 0,
            "unattempted_job_count": 0,
            "accepted_profile_order_indices": [
                checkpoint.job["source_job"]["profile_order_index"]
                for checkpoint in checkpoints
                if checkpoint.accepted
            ],
            "formal_pool_created": False,
            "locked_profile_ids": [],
            "checkpoints": checkpoint_records,
            "diagnostic_job_ledger": ledger,
            "completed_at_utc": completed_at,
        }
    elif status == "aborted_on_infrastructure_failure":
        if (
            not checkpoints
            or checkpoints[-1].status != "infrastructure_failure"
            or infrastructure_count != 1
            or any(
                checkpoint.status != "completed" for checkpoint in checkpoints[:-1]
            )
        ):
            _fail("aborted summary is not a completed prefix plus one infra failure")
        expected = {
            **common,
            "status": "aborted_on_infrastructure_failure",
            "max_jobs": H23_DIAGNOSTIC_MAX_JOBS,
            "attempted_job_count": len(checkpoints),
            "completed_job_count": len(checkpoints) - 1,
            "accepted_job_count": accepted_count,
            "rejected_attempted_job_count": rejected_count,
            "infrastructure_failure_count": 1,
            "unattempted_job_count": H23_DIAGNOSTIC_MAX_JOBS - len(checkpoints),
            "failed_job_id": checkpoints[-1].job["diagnostic_job_id"],
            "formal_pool_created": False,
            "locked_profile_ids": [],
            "checkpoints": checkpoint_records,
            "diagnostic_job_ledger": ledger,
            "completed_at_utc": completed_at,
        }
    else:
        _fail("diagnostic summary status is invalid")
    if canonical_json(entry) != canonical_json(expected):
        _fail("diagnostic summary counts, records, or 24-row ledger do not recompute")
    return payload


def verify_h23_diagnostic_results(
    request: H23DiagnosticVerificationRequest,
) -> H23DiagnosticVerificationReport:
    """Verify one immutable plan/result package without importing any runner."""

    if not isinstance(request, H23DiagnosticVerificationRequest):
        raise TypeError("request must be H23DiagnosticVerificationRequest")
    plan = _verify_plan(
        request.plan_path,
        expected_plan_id=request.expected_plan_id,
        expected_max_jobs=request.expected_max_jobs,
    )
    results_root = _real_path(
        request.results_root, field_name="results_root", kind="directory"
    )
    for forbidden in (
        plan.manifest.root,
        Path(str(plan.payload["source_5s_run"]["root"])),
    ):
        try:
            results_root.relative_to(forbidden)
        except ValueError:
            pass
        else:
            _fail("results root is inside a frozen input root")
    if "formal" in {part.lower() for part in results_root.parts}:
        _fail("development diagnostic results may not be stored under formal")
    entries = {path.name for path in results_root.iterdir()}
    summaries = entries & {"batch_summary.json", "batch_aborted.json"}
    if len(summaries) != 1 or entries != {
        "run_identity.json",
        "checkpoints",
        *summaries,
    }:
        _fail("results root contains missing, extra, or ambiguous artifacts")
    run_identity_path = _real_path(
        results_root / "run_identity.json", field_name="run identity", kind="file"
    )
    checkpoints_root = _real_path(
        results_root / "checkpoints", field_name="checkpoints root", kind="directory"
    )
    summary_name = next(iter(summaries))
    summary_path = _real_path(
        results_root / summary_name, field_name=summary_name, kind="file"
    )
    for path in (run_identity_path, checkpoints_root, summary_path):
        _require_inside(path, results_root, field_name=path.name)
    run_identity, run_sha = _verify_run_identity(
        run_identity_path, plan=plan, request=request
    )
    summary_preview = _read_canonical(summary_path, field_name="summary preview")
    raw_records = _require_list(
        summary_preview.get("checkpoints"), field_name="summary checkpoints"
    )
    if len(raw_records) > H23_DIAGNOSTIC_MAX_JOBS:
        _fail("summary contains more than 24 checkpoint records")
    checkpoints: list[_VerifiedCheckpoint] = []
    observed_paths: set[Path] = set()
    for attempt_index, raw_record in enumerate(raw_records):
        record = _require_mapping(
            raw_record, field_name=f"checkpoint record {attempt_index}"
        )
        _require_exact_keys(
            record,
            {
                "diagnostic_job_id",
                "diagnostic_job_order_index",
                "execution_attempt_index",
                "path",
                "checkpoint_sha256",
                "status",
                "accepted",
            },
            field_name=f"checkpoint record {attempt_index}",
        )
        job = plan.jobs[attempt_index]
        if (
            record["diagnostic_job_id"] != job["diagnostic_job_id"]
            or record["diagnostic_job_order_index"] != attempt_index
            or record["execution_attempt_index"] != attempt_index
        ):
            _fail("checkpoint records are duplicate, missing, or out of order")
        job_id = str(job["diagnostic_job_id"])
        expected_name = (
            f"job-{attempt_index:03d}-"
            f"{hashlib.sha256(job_id.encode('utf-8')).hexdigest()}.json"
        )
        claimed_path = record["path"]
        if not isinstance(claimed_path, str):
            _fail("checkpoint path must be a string")
        checkpoint_path = _real_path(
            claimed_path,
            field_name=f"checkpoint path {attempt_index}",
            kind="file",
        )
        if (
            checkpoint_path.parent != checkpoints_root
            or checkpoint_path.name != expected_name
            or str(checkpoint_path) != claimed_path
            or checkpoint_path in observed_paths
        ):
            _fail("checkpoint path escapes, duplicates, or has a noncanonical name")
        observed_paths.add(checkpoint_path)
        checkpoint = _verify_checkpoint(
            checkpoint_path,
            job=job,
            run_identity=run_identity,
            run_sha=run_sha,
            attempt_index=attempt_index,
        )
        if (
            record["checkpoint_sha256"] != checkpoint.checkpoint_sha256
            or record["status"] != checkpoint.status
            or record["accepted"] != checkpoint.accepted
        ):
            _fail("summary checkpoint record differs from checkpoint artifact")
        checkpoints.append(checkpoint)
    directory_files = {
        _real_path(
            path, field_name=f"checkpoint entry {path.name}", kind="file"
        )
        for path in checkpoints_root.iterdir()
    }
    if directory_files != observed_paths:
        _fail("checkpoint directory contains missing or unreferenced files")
    summary = _verify_summary(
        summary_path, plan=plan, run_sha=run_sha, checkpoints=checkpoints
    )
    accepted_count = sum(checkpoint.accepted for checkpoint in checkpoints)
    rejected_count = sum(
        checkpoint.status == "completed" and not checkpoint.accepted
        for checkpoint in checkpoints
    )
    infrastructure_count = sum(
        checkpoint.status == "infrastructure_failure" for checkpoint in checkpoints
    )
    return H23DiagnosticVerificationReport(
        status=str(summary["status"]),
        summary_path=summary_path,
        plan_id=H23_DIAGNOSTIC_PLAN_ID,
        plan_sha256=plan.plan_sha256,
        diagnostic_run_id=request.expected_run_id,
        git_commit=request.expected_git_commit,
        implementation_sha256=request.expected_implementation_sha256,
        attempted_job_count=len(checkpoints),
        accepted_job_count=accepted_count,
        rejected_attempted_job_count=rejected_count,
        infrastructure_failure_count=infrastructure_count,
        unattempted_job_count=H23_DIAGNOSTIC_MAX_JOBS - len(checkpoints),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=(
            "python -m "
            "src.benchmark.branch_cut.h23_qualification_diagnostic_verifier"
        ),
        description=(
            "Read-only independent verification of the frozen H23 diagnostic; "
            "this module never imports a runner or Gurobi."
        ),
    )
    parser.add_argument(
        "--plan-id", required=True, choices=(H23_DIAGNOSTIC_PLAN_ID,)
    )
    parser.add_argument(
        "--max-jobs", type=int, required=True, choices=(H23_DIAGNOSTIC_MAX_JOBS,)
    )
    parser.add_argument("--plan", required=True)
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--expected-run-id", required=True)
    parser.add_argument("--expected-git-commit", required=True)
    parser.add_argument("--expected-implementation-sha256", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        report = verify_h23_diagnostic_results(
            H23DiagnosticVerificationRequest(
                plan_path=Path(args.plan),
                results_root=Path(args.results_root),
                expected_plan_id=args.plan_id,
                expected_max_jobs=args.max_jobs,
                expected_run_id=args.expected_run_id,
                expected_git_commit=args.expected_git_commit,
                expected_implementation_sha256=(
                    args.expected_implementation_sha256
                ),
            )
        )
    except (H23DiagnosticVerificationError, OSError, ValueError) as error:
        print(f"H23 diagnostic verification failed: {error}", file=sys.stderr)
        return 2
    print(canonical_json(report.to_payload()))
    return 0


__all__ = [
    "H23DiagnosticVerificationError",
    "H23DiagnosticVerificationReport",
    "H23DiagnosticVerificationRequest",
    "build_parser",
    "main",
    "verify_h23_diagnostic_results",
]


if __name__ == "__main__":
    raise SystemExit(main())
