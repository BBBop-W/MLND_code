"""Development-only H23 demand-1 exact-incumbent-reuse batch.

The plan command is solver-free.  The run command executes the fixed post-hoc
order ``[4, 0, 1, 2, 3, 5]`` serially.  Index 4 seeds an in-memory pool; every
later action gets a disposed signature-only preflight before it may optimize.
Normal rejected, all-open, all-closed, and no-incumbent outcomes are retained.
Nothing here is eligible for qualification locking or paper benchmark A--E.
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

from src.config import ModelConfig

from .artifacts import write_canonical_json_exclusive
from .case_builder import canonical_instance_hash
from .hashing import canonical_json, canonical_sha256
from .historical_paths import preserved_development_root
from .h23_incumbent_reuse_smoke import (
    H23_REUSE_SMOKE_PHYSICAL_INSTANCE_ID,
    H23_REUSE_SMOKE_PHYSICAL_INSTANCE_SHA256,
    H23IncumbentReuseSmokeError,
    build_signature_comparison,
    load_h23_incumbent_reuse_smoke_plan,
    probe_constraint_signature_only,
)
from .h23_incumbent_reuse_contract import (
    REGISTERED_GEOMETRY_POLICY_PAYLOAD,
    REGISTERED_NATIVE_SETTINGS_PAYLOAD,
    REGISTERED_POLICY_PAYLOAD,
)
from .qualification import QualificationPolicy, assess_qualification
from .qualification_geometry_start import GeometryStartPolicy
from .qualification_incumbent_reuse import (
    DevelopmentQualificationIncumbentPool,
    QUALIFICATION_INCUMBENT_REUSE_POLICY_VERSION,
    QualificationConstraintSignature,
)
from .qualification_runner import (
    MLNDQualificationAdapter,
    NativeQualificationSettings,
    QualificationAdapter,
    QualificationInfrastructureError,
    QualificationModelContract,
    QualificationRunnerOutcome,
    qualify_canonical_instance,
)
from .resource_lock import DEFAULT_BENCHMARK_RESOURCE_LOCK, BenchmarkResourceLock


H23_DEMAND1_REUSE_BATCH_PLAN_ID = "h23-demand1-exact-reuse-batch-v2"
H23_DEMAND1_REUSE_BATCH_PLAN_SCHEMA = "mlnd-h23-demand1-exact-reuse-plan-v2"
H23_DEMAND1_REUSE_BATCH_RUN_SCHEMA = "mlnd-h23-demand1-exact-reuse-run-v2"
H23_DEMAND1_REUSE_BATCH_CHECKPOINT_SCHEMA = "mlnd-h23-demand1-exact-reuse-checkpoint-v2"
H23_DEMAND1_REUSE_BATCH_SUMMARY_SCHEMA = "mlnd-h23-demand1-exact-reuse-summary-v2"
H23_DEMAND1_REUSE_BATCH_ACTION_INDICES = (4, 0, 1, 2, 3, 5)

VERIFIED_SMOKE_PLAN_SHA256 = (
    "38ce31ff307b197b4277682a3aea7d522dc928881232b894cfdc6cb66b01fc11"
)
VERIFIED_SMOKE_PLAN_FILE_SHA256 = (
    "d26652b3896ff85ca92db55ab7856d39969ad163084365749d718192d05d2a63"
)
VERIFIED_SMOKE_RUN_ID = (
    "h23-incumbent-reuse-smoke-run-"
    "d16fd5e3102e01435edc4794a4918c2a133f708fd474eea71d928ab9831ed491"
)
VERIFIED_SMOKE_SUMMARY_SHA256 = (
    "70c748a7d147b145fc981dd5aaeb3c34968b5f826ae7087676da608103c6ad51"
)
VERIFIED_SMOKE_SUMMARY_FILE_SHA256 = (
    "c60684b4f534672a602c1525ff7dae3781584a3c20e39b7a2406ccef2f306731"
)
VERIFIED_SMOKE_CHECKPOINT_FILE_SHA256 = (
    "32c66aee935f6ba39974513cf7dcd6c79aa6c855b52572a77da8c8866c59f25e",
    "e92adb2a9fe234b2252b3bae09b475d35e4674d041d7705235652e77c01bd70b",
    "e479b9275c8ebb8fc327845db6de7dabab7d86b34993b7843f0e19197fff1700",
)

_PLAN_ARTIFACT_TYPE = "mlnd_branch_cut_h23_demand1_exact_reuse_plan"
_RUN_ARTIFACT_TYPE = "mlnd_branch_cut_h23_demand1_exact_reuse_run_identity"
_CHECKPOINT_ARTIFACT_TYPE = "mlnd_branch_cut_h23_demand1_exact_reuse_checkpoint"
_SUMMARY_ARTIFACT_TYPE = "mlnd_branch_cut_h23_demand1_exact_reuse_summary"
_RUN_ID_PREFIX = "h23-demand1-exact-reuse-v2-run-"
_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
_HISTORICAL_DEVELOPMENT_ROOT = preserved_development_root(
    _REPOSITORY_ROOT,
    required_relative_path=Path(
        "h23_demand1_exact_reuse_batch_plan_0c7d6f7_20260831/plan.json"
    ),
)

_SUPERSEDES: Mapping[str, object] = {
    "plan_id": "h23-demand1-exact-reuse-batch-v1",
    "git_commit": "0c7d6f7fd6ecf2acdda21d10cf3cea4d81baea43",
    "plan_path": str(
        _HISTORICAL_DEVELOPMENT_ROOT
        / "h23_demand1_exact_reuse_batch_plan_0c7d6f7_20260831/plan.json"
    ),
    "plan_sha256": (
        "d9536a18bdb8c5e08942f99144550930ca558953a46b8d659b199a9523b06043"
    ),
    "plan_file_sha256": (
        "5ad75e9f3ad786cd842e4d852bf04a9ceb64248ac8068c0d5e35cd59dac4b906"
    ),
    "implementation_sha256": (
        "181040fed6d2ad36ad490d9fbb59d206fa4348305c8d8b5ebd08c525314626cb"
    ),
    "unused_run_root": str(
        _HISTORICAL_DEVELOPMENT_ROOT
        / "h23_demand1_exact_reuse_batch_0c7d6f7_30s_v1"
    ),
    "run_root_was_never_created": True,
    "reason": (
        "independent pre-run audit rejected the v1 preflight-only signature gate "
        "and under-specified result verifier"
    ),
}

_IMPLEMENTATION_FILES = (
    "src/benchmark/branch_cut/historical_paths.py",
    "src/benchmark/branch_cut/h23_incumbent_reuse_contract.py",
    "src/benchmark/branch_cut/qualification_incumbent_reuse.py",
    "src/benchmark/branch_cut/qualification_runner.py",
    "src/benchmark/branch_cut/h23_incumbent_reuse_smoke.py",
    "src/benchmark/branch_cut/h23_demand1_exact_reuse_batch.py",
    "src/benchmark/branch_cut/h23_demand1_exact_reuse_batch_verifier.py",
)

_BOUNDARY_FLAGS: Mapping[str, object] = {
    "development_only": True,
    "qualification_only": True,
    "eligible_for_formal_pool": False,
    "eligible_for_algorithm_comparison": False,
    "algorithm_comparison_performed": False,
    "may_lock_profiles": False,
    "adds_forced_opening_constraint": False,
    "changes_mathematical_model": False,
    "changes_screening": False,
    "screening_is_exact": False,
}

_ORDER_BIAS: Mapping[str, object] = {
    "source_selection_basis": "historical_h23_only_accepted_profile",
    "anchor_original_profile_order_index": 4,
    "post_hoc_anchor_first": True,
    "execution_order_original_indices": list(H23_DEMAND1_REUSE_BATCH_ACTION_INDICES),
    "manifest_order_deviation": True,
    "timing_comparison_allowed": False,
    "causal_acceleration_claim_allowed": False,
    "qualification_pool_promotion_allowed": False,
}

_EXECUTION_CONTRACT: Mapping[str, object] = {
    "concurrency": 1,
    "threads": 1,
    "demand_scale": 1.0,
    "native_time_limit_seconds": 30.0,
    "geometry_per_attempt_seconds": 10.0,
    "fixed_original_profile_order_indices": list(
        H23_DEMAND1_REUSE_BATCH_ACTION_INDICES
    ),
    "preflight_signature_only_for_targets": True,
    "preflight_optimize_allowed": False,
    "preflight_adapter_disposal_required": True,
    "exact_constraint_signature_required_before_target_solve": True,
    "actual_witness_signature_gate_required_before_pool_select": True,
    "actual_witness_signature_gate_required_before_start_install": True,
    "actual_witness_signature_gate_required_before_optimize": True,
    "fresh_validation_required_for_every_captured_incumbent": True,
    "normal_rejections_retained": True,
    "all_open_outcomes_retained": True,
    "all_closed_outcomes_retained": True,
    "no_incumbent_outcomes_retained": True,
    "exclusive_results_root": True,
    "shared_resource_lock": True,
    "shared_resource_lock_is_primary_concurrency_gate": True,
    "process_scan_is_secondary_concurrency_gate": True,
    "initial_and_per_action_process_conflict_checks": True,
    "process_conflict_markers": [
        "main.py solve",
        "main.py run",
        "calibration_runner",
        "dev_micro_ce",
        "h6_directed_return_ablation",
        "h6_facility_incident_star_ablation",
        "h6_od_disaggregated_return_ablation",
        "h23_qualification_diagnostic",
        "h23_incumbent_reuse_smoke",
        "h23_demand1_exact_reuse_batch",
        "branch_cut qualify",
        "branch_cut run ",
        "branch_cut run-formal",
        "mlnd_qualification",
        "gurobi_cl",
        "pytest",
    ],
    "infrastructure_fail_fast": True,
    "infrastructure_failure_writes_aborted_checkpoint_and_summary": True,
    "solver_exception_is_infrastructure_failure": True,
    "normal_time_limit_continues": True,
    "raw_incumbent_values_persisted": False,
}

_PROCESS_CONFLICT_MARKERS = (
    "main.py solve",
    "main.py run",
    "calibration_runner",
    "dev_micro_ce",
    "h6_directed_return_ablation",
    "h6_facility_incident_star_ablation",
    "h6_od_disaggregated_return_ablation",
    "h23_qualification_diagnostic",
    "h23_incumbent_reuse_smoke",
    "h23_demand1_exact_reuse_batch",
    "branch_cut qualify",
    "branch_cut run ",
    "branch_cut run-formal",
    "mlnd_qualification",
    "gurobi_cl",
    "pytest",
)


class H23Demand1ExactReuseBatchError(RuntimeError):
    """A frozen input, plan, or artifact contract failed."""


class H23Demand1ExactReuseBatchInfrastructureError(RuntimeError):
    """The batch stopped after persisting an infrastructure failure."""


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


def _read_canonical(path: Path, *, field_name: str) -> dict[str, Any]:
    try:
        raw = path.read_text(encoding="utf-8")
        payload = json.loads(raw)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise H23Demand1ExactReuseBatchError(
            f"cannot read {field_name}: {error}"
        ) from error
    if not isinstance(payload, dict) or raw != canonical_json(payload) + "\n":
        raise H23Demand1ExactReuseBatchError(f"{field_name} is not canonical JSON")
    return payload


def _self_hashed(payload: Mapping[str, Any], *, hash_field: str) -> dict[str, Any]:
    if hash_field in payload:
        raise ValueError(f"payload already contains {hash_field}")
    entry = dict(payload)
    return {**entry, hash_field: canonical_sha256(entry)}


def _validate_superseded_v1_evidence() -> None:
    """Bind v2 to the preserved NO-GO plan and its never-created run root."""

    plan_path = Path(str(_SUPERSEDES["plan_path"]))
    payload = _read_canonical(plan_path.resolve(strict=True), field_name="superseded plan")
    if (
        payload.get("plan_id") != _SUPERSEDES["plan_id"]
        or payload.get("plan_sha256") != _SUPERSEDES["plan_sha256"]
        or _sha256_file(plan_path) != _SUPERSEDES["plan_file_sha256"]
        or payload.get("required_execution_environment", {}).get("git_commit")
        != _SUPERSEDES["git_commit"]
        or payload.get("required_execution_environment", {})
        .get("implementation", {})
        .get("implementation_sha256")
        != _SUPERSEDES["implementation_sha256"]
    ):
        raise H23Demand1ExactReuseBatchError("superseded v1 plan binding drifted")
    old_root = Path(str(_SUPERSEDES["unused_run_root"]))
    if old_root.exists() or old_root.is_symlink():
        raise H23Demand1ExactReuseBatchError(
            "superseded v1 run root is no longer unused"
        )


def _reject_superseded_root_overlap(target: Path, *, field_name: str) -> None:
    absolute = target.expanduser().resolve(strict=False)
    old_root = Path(str(_SUPERSEDES["unused_run_root"])).resolve(strict=False)
    try:
        absolute.relative_to(old_root)
    except ValueError:
        return
    raise H23Demand1ExactReuseBatchError(
        f"{field_name} overlaps the preserved superseded v1 run root"
    )


def _git(repository_root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ("git", *arguments),
        cwd=repository_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise H23Demand1ExactReuseBatchError(
            f"git {' '.join(arguments)} failed: "
            + (completed.stderr or completed.stdout).strip()
        )
    return completed.stdout.strip()


def _implementation_payload(repository_root: Path) -> dict[str, Any]:
    files = {
        relative: _sha256_file(repository_root / relative)
        for relative in _IMPLEMENTATION_FILES
    }
    return {
        "ordered_file_sha256": files,
        "implementation_sha256": canonical_sha256(files),
    }


def _capture_repository_state(repository_root: Path) -> Mapping[str, Any]:
    status = [
        line
        for line in _git(
            repository_root, "status", "--porcelain", "--untracked-files=all"
        ).splitlines()
        if line
    ]
    return {
        "repository_root": str(repository_root.resolve()),
        "git_commit": _git(repository_root, "rev-parse", "HEAD"),
        "status_includes_untracked": True,
        "status_lines": status,
        "worktree_dirty": bool(status),
        "implementation": _implementation_payload(repository_root),
    }


def _default_runtime_environment(repository_root: Path) -> Mapping[str, Any]:
    import gurobipy as gp

    return {
        "repository": dict(_capture_repository_state(repository_root)),
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
        raise H23Demand1ExactReuseBatchError(
            "cannot audit concurrent solver/test processes: "
            + (completed.stderr or completed.stdout).strip()
        )
    return completed.stdout


def _process_audit(process_table: str) -> dict[str, Any]:
    rows: dict[int, tuple[int, str]] = {}
    for raw_line in process_table.splitlines():
        fields = raw_line.strip().split(maxsplit=2)
        try:
            pid, ppid, line = int(fields[0]), int(fields[1]), fields[2]
        except (IndexError, ValueError):
            continue
        rows[pid] = (ppid, line)
    excluded = {os.getpid()}
    cursor = os.getppid()
    while cursor > 0 and cursor not in excluded:
        excluded.add(cursor)
        cursor = rows.get(cursor, (0, ""))[0]
    conflicts = [
        line
        for pid, (_ppid, line) in sorted(rows.items())
        if pid not in excluded
        and any(marker in line for marker in _PROCESS_CONFLICT_MARKERS)
    ]
    return {
        "process_table_sha256": hashlib.sha256(
            process_table.encode("utf-8")
        ).hexdigest(),
        "conflicting_processes": conflicts,
        "valid": not conflicts,
    }


def _stable_environment(environment: Mapping[str, Any]) -> dict[str, Any]:
    stable = dict(environment)
    repository = dict(stable["repository"])
    repository.pop("status_lines", None)
    repository.pop("worktree_dirty", None)
    stable["repository"] = repository
    return stable


def _registered_policy() -> QualificationPolicy:
    payload = dict(REGISTERED_POLICY_PAYLOAD)
    target = payload.get("large_scale_open_h_target")
    payload["large_scale_open_h_target"] = tuple(target) if target else None
    return QualificationPolicy(**payload)


def _registered_settings() -> NativeQualificationSettings:
    return NativeQualificationSettings(**REGISTERED_NATIVE_SETTINGS_PAYLOAD)


def _registered_geometry_policy() -> GeometryStartPolicy:
    payload = dict(REGISTERED_GEOMETRY_POLICY_PAYLOAD)
    for name in (
        "small_support_schedule",
        "medium_support_schedule",
        "large_support_schedule",
    ):
        payload[name] = tuple(payload[name])
    return GeometryStartPolicy(**payload)


def _default_instance_loader(
    instance_dir: Path, model_config: ModelConfig, demand_scale: float
) -> object:
    from src.data_processor.loader import DataLoader

    return DataLoader().load_instance_dir(
        str(instance_dir), model_config=model_config, demand_scale=demand_scale
    )


def _default_adapter_factory(instance: object, mode: str) -> QualificationAdapter:
    return MLNDQualificationAdapter(instance, mode)


def classify_outcome_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return a compact, lossless-enough triage classification."""

    evidence = payload.get("evidence")
    if not isinstance(evidence, Mapping):
        raise H23Demand1ExactReuseBatchError("outcome evidence is missing")
    candidate_h = evidence.get("candidate_h")
    open_h = evidence.get("open_h")
    active_h = evidence.get("active_h")
    captured = payload.get("captured_variable_count")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in (candidate_h, open_h, captured)
    ):
        raise H23Demand1ExactReuseBatchError("outcome counts are invalid")
    if open_h > candidate_h:
        raise H23Demand1ExactReuseBatchError("open H exceeds candidate H")
    if active_h is not None and (
        isinstance(active_h, bool)
        or not isinstance(active_h, int)
        or active_h < 0
        or active_h > open_h
    ):
        raise H23Demand1ExactReuseBatchError("active H is invalid")
    reasons = payload.get("rejection_reasons")
    if not isinstance(reasons, list) or any(not isinstance(x, str) for x in reasons):
        raise H23Demand1ExactReuseBatchError("rejection reasons are invalid")
    accepted = payload.get("accepted")
    fresh = payload.get("fresh_model_validation_performed")
    if not isinstance(accepted, bool) or not isinstance(fresh, bool):
        raise H23Demand1ExactReuseBatchError("outcome booleans are invalid")
    return {
        "accepted": accepted,
        "has_incumbent": captured > 0,
        "captured_variable_count": captured,
        "fresh_model_validation_performed": fresh,
        "fresh_validation_required": captured > 0,
        "fresh_validation_requirement_satisfied": (captured == 0 or fresh),
        "candidate_h": candidate_h,
        "open_h": open_h,
        "active_h": active_h,
        "all_closed": open_h == 0,
        "all_open": candidate_h > 0 and open_h == candidate_h,
        "interior_open": 0 < open_h < candidate_h,
        "witness_status": payload.get("witness_status"),
        "validation_status": payload.get("validation_status"),
        "rejection_reasons": list(reasons),
    }


def _validate_verified_smoke(
    plan_path: Path, results_root: Path
) -> tuple[object, dict[str, Any]]:
    try:
        smoke_plan = load_h23_incumbent_reuse_smoke_plan(plan_path)
    except H23IncumbentReuseSmokeError as error:
        raise H23Demand1ExactReuseBatchError(str(error)) from error
    root = results_root.expanduser().resolve(strict=True)
    summary_path = root / "batch_summary.json"
    run_path = root / "run_identity.json"
    summary = _read_canonical(summary_path, field_name="verified smoke summary")
    run_identity = _read_canonical(run_path, field_name="verified smoke run identity")
    checkpoint_records = summary.get("checkpoints")
    if not isinstance(checkpoint_records, list) or len(checkpoint_records) != 3:
        raise H23Demand1ExactReuseBatchError("verified smoke checkpoint ledger drifted")
    checkpoint_hashes = tuple(
        _sha256_file(Path(str(record["path"])).resolve(strict=True))
        for record in checkpoint_records
    )
    if (
        smoke_plan.plan_sha256 != VERIFIED_SMOKE_PLAN_SHA256
        or smoke_plan.file_sha256 != VERIFIED_SMOKE_PLAN_FILE_SHA256
        or _sha256_file(summary_path) != VERIFIED_SMOKE_SUMMARY_FILE_SHA256
        or summary.get("summary_sha256") != VERIFIED_SMOKE_SUMMARY_SHA256
        or summary.get("smoke_run_id") != VERIFIED_SMOKE_RUN_ID
        or summary.get("conclusion") != "passed"
        or summary.get("passed") is not True
        or run_identity.get("smoke_run_id") != VERIFIED_SMOKE_RUN_ID
        or checkpoint_hashes != VERIFIED_SMOKE_CHECKPOINT_FILE_SHA256
        or summary_path.parent != root
        or run_path.parent != root
    ):
        raise H23Demand1ExactReuseBatchError("verified v1 smoke evidence drifted")
    binding = {
        "verifier_status": "verified",
        "verifier_conclusion": "passed",
        "plan_path": str(smoke_plan.path),
        "plan_sha256": smoke_plan.plan_sha256,
        "plan_file_sha256": smoke_plan.file_sha256,
        "results_root": str(root),
        "run_identity_path": str(run_path),
        "run_identity_file_sha256": _sha256_file(run_path),
        "smoke_run_id": VERIFIED_SMOKE_RUN_ID,
        "summary_path": str(summary_path),
        "summary_sha256": VERIFIED_SMOKE_SUMMARY_SHA256,
        "summary_file_sha256": VERIFIED_SMOKE_SUMMARY_FILE_SHA256,
        "checkpoint_file_sha256": list(checkpoint_hashes),
        "mechanism_only_no_speedup_claim": True,
    }
    return smoke_plan, binding


def _historical_classification(record: Mapping[str, Any]) -> dict[str, Any]:
    outcome = record.get("outcome")
    if not isinstance(outcome, Mapping):
        raise H23Demand1ExactReuseBatchError("historical outcome is missing")
    return classify_outcome_payload(outcome)


def _batch_actions(smoke_plan: object) -> tuple[dict[str, Any], ...]:
    source_h23_plan = smoke_plan.source_h23_plan
    source_summary_path = Path(
        str(smoke_plan.payload["source_h23"]["summary_path"])
    ).resolve(strict=True)
    source_summary = _read_canonical(
        source_summary_path, field_name="source H23 summary"
    )
    ledger = {
        int(record["diagnostic_job_order_index"]): record
        for record in source_summary["diagnostic_job_ledger"]
    }
    actions: list[dict[str, Any]] = []
    for action_index, original_index in enumerate(
        H23_DEMAND1_REUSE_BATCH_ACTION_INDICES
    ):
        job = dict(source_h23_plan.jobs[original_index])
        source_job = job["source_job"]
        profile = job["qualification_solve_profile"]
        contract = job["qualification_model_contract"]
        if (
            job["diagnostic_job_order_index"] != original_index
            or source_job["profile_order_index"] != original_index
            or source_job["instance_sha256"] != H23_REUSE_SMOKE_PHYSICAL_INSTANCE_SHA256
            or source_job["physical_instance_id"]
            != H23_REUSE_SMOKE_PHYSICAL_INSTANCE_ID
            or source_job["scale"] != "small"
            or float(profile["demand_scale"]) != 1.0
            or float(contract["demand_scale"]) != 1.0
            or contract["screening_is_exact"] is not False
        ):
            raise H23Demand1ExactReuseBatchError(
                f"H23 demand-1 action {original_index} drifted"
            )
        historical = ledger[original_index]
        checkpoint_path = Path(str(historical["checkpoint_path"])).resolve(strict=True)
        role = "anchor_source" if action_index == 0 else "exact_reuse_target"
        case_id = "h23-demand1-exact-reuse-case-" + canonical_sha256(
            {
                "plan_id": H23_DEMAND1_REUSE_BATCH_PLAN_ID,
                "action_index": action_index,
                "original_profile_order_index": original_index,
                "diagnostic_job_id": job["diagnostic_job_id"],
            }
        )
        actions.append(
            {
                "action_index": action_index,
                "role": role,
                "original_profile_order_index": original_index,
                "expected_signature_relation_to_anchor": (
                    "establish" if action_index == 0 else "equal"
                ),
                "batch_case_id": case_id,
                "source_job": job,
                "historical_checkpoint": {
                    "path": str(checkpoint_path),
                    "checkpoint_sha256": historical["checkpoint_sha256"],
                    "file_sha256": _sha256_file(checkpoint_path),
                    "classification": _historical_classification(historical),
                },
                "expected_event_prefix": (
                    ["reuse_cache_miss"]
                    if action_index == 0
                    else ["reuse_candidate_selected", "reuse_start_installed"]
                ),
                "incumbent_offer_expected_iff_accepted": True,
            }
        )
    if actions[0]["historical_checkpoint"]["classification"]["accepted"] is not True:
        raise H23Demand1ExactReuseBatchError("historical anchor is no longer accepted")
    return tuple(actions)


@dataclass(frozen=True, slots=True)
class H23Demand1ExactReusePlanBuildHooks:
    capture_repository_state: Callable[[Path], Mapping[str, Any]] = (
        _capture_repository_state
    )


@dataclass(frozen=True, slots=True)
class H23Demand1ExactReusePlanBuildConfig:
    verified_smoke_plan_path: Path
    verified_smoke_results_root: Path
    output_path: Path
    expected_git_commit: str

    def __post_init__(self) -> None:
        if len(self.expected_git_commit) != 40 or any(
            character not in "0123456789abcdef"
            for character in self.expected_git_commit
        ):
            raise ValueError("expected_git_commit must be a lowercase Git digest")


@dataclass(frozen=True, slots=True)
class H23Demand1ExactReusePlanBuildResult:
    path: Path
    plan_sha256: str
    plan_file_sha256: str
    git_commit: str
    implementation_sha256: str

    def to_payload(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "plan_id": H23_DEMAND1_REUSE_BATCH_PLAN_ID,
            "plan_sha256": self.plan_sha256,
            "plan_file_sha256": self.plan_file_sha256,
            "action_count": len(H23_DEMAND1_REUSE_BATCH_ACTION_INDICES),
            "git_commit": self.git_commit,
            "implementation_sha256": self.implementation_sha256,
        }


def build_h23_demand1_exact_reuse_plan(
    config: H23Demand1ExactReusePlanBuildConfig,
    *,
    hooks: H23Demand1ExactReusePlanBuildHooks = H23Demand1ExactReusePlanBuildHooks(),
) -> H23Demand1ExactReusePlanBuildResult:
    """Build one exclusive canonical plan without importing Gurobi."""

    _reject_superseded_root_overlap(config.output_path, field_name="plan output")
    state = dict(hooks.capture_repository_state(_REPOSITORY_ROOT))
    if (
        state.get("git_commit") != config.expected_git_commit
        or state.get("worktree_dirty") is not False
        or state.get("status_includes_untracked") is not True
        or state.get("status_lines") != []
    ):
        raise H23Demand1ExactReuseBatchError(
            "plan generation requires the exact clean expected commit"
        )
    implementation = dict(state["implementation"])
    if set(implementation["ordered_file_sha256"]) != set(_IMPLEMENTATION_FILES):
        raise H23Demand1ExactReuseBatchError("implementation file set drifted")
    _validate_superseded_v1_evidence()
    smoke_plan, smoke_binding = _validate_verified_smoke(
        config.verified_smoke_plan_path, config.verified_smoke_results_root
    )
    actions = _batch_actions(smoke_plan)
    entry = {
        "artifact_type": _PLAN_ARTIFACT_TYPE,
        "artifact_schema_version": 1,
        "plan_schema": H23_DEMAND1_REUSE_BATCH_PLAN_SCHEMA,
        "plan_id": H23_DEMAND1_REUSE_BATCH_PLAN_ID,
        **dict(_BOUNDARY_FLAGS),
        "mode": "MX",
        "action_count": len(actions),
        "reuse_policy_version": QUALIFICATION_INCUMBENT_REUSE_POLICY_VERSION,
        "supersedes": dict(_SUPERSEDES),
        "order_bias": dict(_ORDER_BIAS),
        "execution_contract": dict(_EXECUTION_CONTRACT),
        "qualification_policy": dict(REGISTERED_POLICY_PAYLOAD),
        "native_settings": dict(REGISTERED_NATIVE_SETTINGS_PAYLOAD),
        "geometry_start_policy": dict(REGISTERED_GEOMETRY_POLICY_PAYLOAD),
        "verified_v1_mechanism_smoke": smoke_binding,
        "source_h23": dict(smoke_plan.payload["source_h23"]),
        "required_execution_environment": {
            "repository_root": state["repository_root"],
            "git_commit": config.expected_git_commit,
            "clean_status_includes_untracked": True,
            "implementation": implementation,
        },
        "actions": list(actions),
    }
    payload = _self_hashed(entry, hash_field="plan_sha256")
    output = config.output_path.expanduser().absolute()
    output.parent.mkdir(parents=True, exist_ok=True)
    published = write_canonical_json_exclusive(output, payload)
    return H23Demand1ExactReusePlanBuildResult(
        path=published,
        plan_sha256=payload["plan_sha256"],
        plan_file_sha256=_sha256_file(published),
        git_commit=config.expected_git_commit,
        implementation_sha256=implementation["implementation_sha256"],
    )


@dataclass(frozen=True, slots=True)
class LoadedH23Demand1ExactReusePlan:
    path: Path
    file_sha256: str
    plan_sha256: str
    payload: Mapping[str, Any]
    source_smoke_plan: object
    actions: tuple[Mapping[str, Any], ...]


def load_h23_demand1_exact_reuse_plan(
    plan_path: str | Path,
) -> LoadedH23Demand1ExactReusePlan:
    path = Path(plan_path).expanduser().resolve(strict=True)
    _validate_superseded_v1_evidence()
    payload = _read_canonical(path, field_name="H23 demand-1 exact-reuse plan")
    plan_sha = payload.get("plan_sha256")
    if (
        not isinstance(plan_sha, str)
        or canonical_sha256(
            {key: value for key, value in payload.items() if key != "plan_sha256"}
        )
        != plan_sha
    ):
        raise H23Demand1ExactReuseBatchError("plan self-hash mismatch")
    fixed = {
        "artifact_type": _PLAN_ARTIFACT_TYPE,
        "artifact_schema_version": 1,
        "plan_schema": H23_DEMAND1_REUSE_BATCH_PLAN_SCHEMA,
        "plan_id": H23_DEMAND1_REUSE_BATCH_PLAN_ID,
        **dict(_BOUNDARY_FLAGS),
        "mode": "MX",
        "action_count": len(H23_DEMAND1_REUSE_BATCH_ACTION_INDICES),
        "reuse_policy_version": QUALIFICATION_INCUMBENT_REUSE_POLICY_VERSION,
        "supersedes": dict(_SUPERSEDES),
        "order_bias": dict(_ORDER_BIAS),
        "execution_contract": dict(_EXECUTION_CONTRACT),
        "qualification_policy": dict(REGISTERED_POLICY_PAYLOAD),
        "native_settings": dict(REGISTERED_NATIVE_SETTINGS_PAYLOAD),
        "geometry_start_policy": dict(REGISTERED_GEOMETRY_POLICY_PAYLOAD),
    }
    if any(payload.get(key) != value for key, value in fixed.items()):
        raise H23Demand1ExactReuseBatchError("fixed plan contract drifted")
    smoke_binding = payload.get("verified_v1_mechanism_smoke")
    if not isinstance(smoke_binding, Mapping):
        raise H23Demand1ExactReuseBatchError("verified smoke binding is missing")
    smoke_plan, expected_binding = _validate_verified_smoke(
        Path(str(smoke_binding["plan_path"])),
        Path(str(smoke_binding["results_root"])),
    )
    if canonical_json(smoke_binding) != canonical_json(expected_binding):
        raise H23Demand1ExactReuseBatchError("verified smoke binding drifted")
    if payload.get("source_h23") != smoke_plan.payload["source_h23"]:
        raise H23Demand1ExactReuseBatchError("source H23 binding drifted")
    expected_actions = _batch_actions(smoke_plan)
    if canonical_json(payload.get("actions")) != canonical_json(expected_actions):
        raise H23Demand1ExactReuseBatchError("batch action contract drifted")
    required = payload.get("required_execution_environment")
    if not isinstance(required, Mapping) or set(
        required.get("implementation", {}).get("ordered_file_sha256", {})
    ) != set(_IMPLEMENTATION_FILES):
        raise H23Demand1ExactReuseBatchError("required implementation drifted")
    return LoadedH23Demand1ExactReusePlan(
        path=path,
        file_sha256=_sha256_file(path),
        plan_sha256=plan_sha,
        payload=payload,
        source_smoke_plan=smoke_plan,
        actions=tuple(expected_actions),
    )


@dataclass(frozen=True, slots=True)
class H23Demand1ExactReuseBatchHooks:
    load_instance: Callable[[Path, ModelConfig, float], object] = (
        _default_instance_loader
    )
    qualify_instance: Callable[..., QualificationRunnerOutcome] = (
        qualify_canonical_instance
    )
    adapter_factory: Callable[[object, str], QualificationAdapter] = (
        _default_adapter_factory
    )
    capture_environment: Callable[[Path], Mapping[str, Any]] = (
        _default_runtime_environment
    )
    process_table: Callable[[], str] = _process_table
    now_utc: Callable[[], str] = _utc_now


@dataclass(frozen=True, slots=True)
class H23Demand1ExactReuseBatchRunConfig:
    plan_path: Path
    results_root: Path
    resource_lock: Path = field(default_factory=lambda: DEFAULT_BENCHMARK_RESOURCE_LOCK)


@dataclass(frozen=True, slots=True)
class H23Demand1ExactReuseBatchRunResult:
    results_root: Path
    run_identity_path: Path
    summary_path: Path
    batch_run_id: str
    conclusion: str
    checkpoint_paths: tuple[Path, ...]

    def to_payload(self) -> dict[str, Any]:
        return {
            "results_root": str(self.results_root),
            "run_identity_path": str(self.run_identity_path),
            "summary_path": str(self.summary_path),
            "batch_run_id": self.batch_run_id,
            "conclusion": self.conclusion,
            "checkpoint_paths": [str(path) for path in self.checkpoint_paths],
        }


def _validate_runtime_environment(
    environment: Mapping[str, Any], plan: LoadedH23Demand1ExactReusePlan
) -> None:
    repository = environment.get("repository")
    required = plan.payload["required_execution_environment"]
    if not isinstance(repository, Mapping) or (
        repository.get("repository_root") != required["repository_root"]
        or repository.get("git_commit") != required["git_commit"]
        or repository.get("status_includes_untracked") is not True
        or repository.get("worktree_dirty") is not False
        or repository.get("status_lines") != []
        or repository.get("implementation") != required["implementation"]
    ):
        raise H23Demand1ExactReuseBatchError("runtime repository/code drifted")
    if not isinstance(environment.get("gurobi_version"), str):
        raise H23Demand1ExactReuseBatchError("runtime Gurobi version is missing")


def _claim_results_root(target: Path, plan: LoadedH23Demand1ExactReusePlan) -> Path:
    absolute = target.expanduser().absolute()
    _reject_superseded_root_overlap(absolute, field_name="results root")
    if absolute.exists() or absolute.is_symlink():
        raise H23Demand1ExactReuseBatchError(f"results root exists: {absolute}")
    if "formal" in {part.lower() for part in absolute.parts}:
        raise H23Demand1ExactReuseBatchError("results cannot be stored under formal")
    forbidden = (
        plan.path.parent,
        Path(str(plan.payload["source_h23"]["summary_path"])).parent,
        Path(str(plan.payload["verified_v1_mechanism_smoke"]["results_root"])),
    )
    parent = absolute.parent.resolve()
    for root in forbidden:
        try:
            parent.relative_to(root.resolve())
        except ValueError:
            continue
        raise H23Demand1ExactReuseBatchError("result root overlaps frozen evidence")
    absolute.parent.mkdir(parents=True, exist_ok=True)
    absolute.mkdir(exist_ok=False)
    return absolute.resolve()


def _revalidate_action_inputs(
    plan: LoadedH23Demand1ExactReusePlan, action: Mapping[str, Any]
) -> None:
    if _sha256_file(plan.path) != plan.file_sha256:
        raise QualificationInfrastructureError("batch plan changed during run")
    for binding in (
        plan.payload["source_h23"],
        plan.payload["verified_v1_mechanism_smoke"],
    ):
        for path_key, hash_key in (
            ("plan_path", "plan_file_sha256"),
            ("summary_path", "summary_file_sha256"),
        ):
            if _sha256_file(Path(str(binding[path_key]))) != binding[hash_key]:
                raise QualificationInfrastructureError("bound evidence changed")
    historical = action["historical_checkpoint"]
    if _sha256_file(Path(str(historical["path"]))) != historical["file_sha256"]:
        raise QualificationInfrastructureError("historical checkpoint changed")
    job = action["source_job"]["source_job"]
    instance_dir = Path(str(action["source_job"]["physical_instance_directory"]))
    observed, _ = canonical_instance_hash(instance_dir)
    if observed != job["instance_sha256"]:
        raise QualificationInfrastructureError("physical instance changed")


def _case_events(
    pool: DevelopmentQualificationIncumbentPool, case_id: str
) -> tuple[str, ...]:
    return tuple(event.action for event in pool.events_for_case(case_id))


def _case_signature(
    pool: DevelopmentQualificationIncumbentPool, case_id: str
) -> QualificationConstraintSignature:
    events = pool.events_for_case(case_id)
    signatures = {event.target_constraint_signature for event in events}
    if not events or len(signatures) != 1:
        raise QualificationInfrastructureError("case signature audit is ambiguous")
    return next(iter(signatures))


def _selected_source(
    pool: DevelopmentQualificationIncumbentPool, case_id: str
) -> dict[str, Any] | None:
    selected = [
        event
        for event in pool.events_for_case(case_id)
        if event.action == "reuse_candidate_selected"
    ]
    if not selected:
        return None
    if len(selected) != 1 or selected[0].source_provenance is None:
        raise QualificationInfrastructureError("selected-source audit is ambiguous")
    return {
        "source_provenance": selected[0].source_provenance.to_payload(),
        "source_reuse_record_sha256": selected[0].source_reuse_record_sha256,
    }


def exact_target_signature_preflight(
    instance: object,
    *,
    contract: QualificationModelContract,
    anchor_signature: QualificationConstraintSignature,
    target_demand_scale: float,
    adapter_factory: Callable[[object, str], QualificationAdapter],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Dispose a no-optimize target probe and require exact anchor equality."""

    probe = probe_constraint_signature_only(
        instance,
        "MX",
        expected_model_contract=contract,
        adapter_factory=adapter_factory,
    )
    comparison = build_signature_comparison(
        role="exact_reuse_target",
        expected_relation="equal",
        source=anchor_signature,
        target=probe.constraint_signature,
        source_demand_scale=1.0,
        target_demand_scale=target_demand_scale,
    )
    if not comparison["comparison_passed"]:
        raise QualificationInfrastructureError(
            "target constraint signature differs from anchor"
        )
    return probe.to_payload(), comparison


def _validate_outcome(
    outcome: QualificationRunnerOutcome,
    *,
    action: Mapping[str, Any],
    run_id: str,
    policy: QualificationPolicy,
) -> dict[str, Any]:
    if not isinstance(outcome, QualificationRunnerOutcome):
        raise QualificationInfrastructureError("qualifier returned invalid outcome")
    job = action["source_job"]
    contract = QualificationModelContract(**job["qualification_model_contract"])
    if (
        outcome.expected_model_contract != contract
        or outcome.decision != assess_qualification(outcome.evidence, policy)
        or outcome.exception_messages
        or outcome.witness_model_audit is None
    ):
        raise QualificationInfrastructureError("qualification semantic audit drifted")
    expected_audit_values = (
        contract.effective_model_config_sha256,
        contract.demand_scale,
        contract.locker_fixed_cost_per_hour,
        contract.display_origin_choices,
        contract.general_corridor_choices,
        contract.screening_is_exact,
    )
    if outcome.witness_model_audit.contract_values() != expected_audit_values:
        raise QualificationInfrastructureError("witness model audit drifted")
    if outcome.witness_model_audit.model_fingerprint is None:
        raise QualificationInfrastructureError("witness model fingerprint is missing")
    evidence = outcome.evidence
    expected_identity = (
        action["batch_case_id"],
        "small",
        run_id,
        H23_REUSE_SMOKE_PHYSICAL_INSTANCE_SHA256,
        contract.qualification_case_sha256,
        contract.effective_model_config_sha256,
        False,
    )
    observed_identity = (
        evidence.case_id,
        evidence.scale,
        evidence.qualification_run_id,
        evidence.instance_sha256,
        evidence.qualification_case_sha256,
        evidence.effective_model_config_sha256,
        evidence.screening_is_exact,
    )
    if observed_identity != expected_identity:
        raise QualificationInfrastructureError("qualification identity drifted")
    classification = classify_outcome_payload(outcome.to_payload())
    if not classification["fresh_validation_requirement_satisfied"]:
        raise QualificationInfrastructureError(
            "captured incumbent lacks independent fresh validation"
        )
    captured = outcome.captured_variable_count
    if outcome.accepted and (
        not outcome.fresh_model_validation_performed
        or outcome.fresh_model_audit is None
        or not outcome.evidence.witness_feasible
        or captured <= 0
    ):
        raise QualificationInfrastructureError(
            "accepted outcome lacks fresh complete witness"
        )
    if captured > 0:
        if (
            not outcome.fresh_model_validation_performed
            or outcome.fresh_model_audit is None
            or outcome.fresh_model_audit != outcome.witness_model_audit
            or outcome.witness_status not in {"OPTIMAL", "TIME_LIMIT"}
            or outcome.validation_status not in {"OPTIMAL", "TIME_LIMIT"}
            or not outcome.evidence.witness_feasible
        ):
            raise QualificationInfrastructureError(
                "captured incumbent fresh validation audit drifted"
            )
    elif (
        outcome.fresh_model_validation_performed
        or outcome.fresh_model_audit is not None
        or outcome.validation_status != "NOT_RUN"
        or outcome.evidence.witness_feasible
    ):
        raise QualificationInfrastructureError(
            "no-incumbent fresh validation audit drifted"
        )
    return classification


def _write_checkpoint(
    root: Path, *, identity: Mapping[str, Any], result: Mapping[str, Any]
) -> tuple[Path, str, str]:
    action = identity["action"]
    path = root / (
        f"action-{int(action['action_index']):02d}-"
        f"profile-{int(action['original_profile_order_index']):02d}.json"
    )
    payload = _self_hashed(
        {
            "artifact_type": _CHECKPOINT_ARTIFACT_TYPE,
            "artifact_schema_version": 1,
            "checkpoint_schema": H23_DEMAND1_REUSE_BATCH_CHECKPOINT_SCHEMA,
            **dict(_BOUNDARY_FLAGS),
            "contains_algorithm_comparison_results": False,
            "identity": dict(identity),
            "result": dict(result),
        },
        hash_field="checkpoint_sha256",
    )
    published = write_canonical_json_exclusive(path, payload)
    return published, payload["checkpoint_sha256"], _sha256_file(published)


def _summary_ledger_entry(
    action: Mapping[str, Any], result: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "action_index": action["action_index"],
        "original_profile_order_index": action["original_profile_order_index"],
        "profile_id": action["source_job"]["source_job"]["profile_id"],
        "status": result["status"],
        "historical_classification": action["historical_checkpoint"]["classification"],
        "runtime_classification": result.get("runtime_classification"),
        "reuse_candidate_selected": result.get("reuse_candidate_selected"),
        "reuse_start_installed": result.get("reuse_start_installed"),
        "fresh_validation_performed": result.get("fresh_validation_performed"),
        "selected_source": result.get("selected_source"),
        "signature_comparison_to_anchor": result.get("signature_comparison_to_anchor"),
    }


def _write_summary(
    root: Path,
    *,
    run_id: str,
    run_sha: str,
    plan: LoadedH23Demand1ExactReusePlan,
    conclusion: str,
    checkpoint_records: Sequence[Mapping[str, Any]],
    ledger: Sequence[Mapping[str, Any]],
    pool: DevelopmentQualificationIncumbentPool,
    completed_at_utc: str,
    infrastructure_failure: Mapping[str, Any] | None = None,
) -> Path:
    classifications = [
        record["runtime_classification"]
        for record in ledger
        if record.get("runtime_classification") is not None
    ]
    payload = _self_hashed(
        {
            "artifact_type": _SUMMARY_ARTIFACT_TYPE,
            "artifact_schema_version": 1,
            "summary_schema": H23_DEMAND1_REUSE_BATCH_SUMMARY_SCHEMA,
            **dict(_BOUNDARY_FLAGS),
            "contains_algorithm_comparison_results": False,
            "batch_run_id": run_id,
            "run_identity_sha256": run_sha,
            "plan_id": H23_DEMAND1_REUSE_BATCH_PLAN_ID,
            "plan_sha256": plan.plan_sha256,
            "order_bias": dict(_ORDER_BIAS),
            "conclusion": conclusion,
            "completed": conclusion == "completed",
            "attempted_action_count": len(checkpoint_records),
            "completed_action_count": sum(
                record["status"] == "completed" for record in checkpoint_records
            ),
            "accepted_count": sum(row["accepted"] for row in classifications),
            "rejected_count": sum(not row["accepted"] for row in classifications),
            "has_incumbent_count": sum(row["has_incumbent"] for row in classifications),
            "no_incumbent_count": sum(
                not row["has_incumbent"] for row in classifications
            ),
            "all_closed_count": sum(row["all_closed"] for row in classifications),
            "all_open_count": sum(row["all_open"] for row in classifications),
            "interior_open_count": sum(row["interior_open"] for row in classifications),
            "fresh_validated_count": sum(
                row["fresh_model_validation_performed"] for row in classifications
            ),
            "infrastructure_failure_count": int(infrastructure_failure is not None),
            "infrastructure_failure": (
                dict(infrastructure_failure)
                if infrastructure_failure is not None
                else None
            ),
            "checkpoints": [dict(record) for record in checkpoint_records],
            "profile_ledger": [dict(record) for record in ledger],
            "reuse_pool_audit": pool.to_payload(),
            "formal_pool_created": False,
            "locked_profile_ids": [],
            "speedup_claimed": False,
            "completed_at_utc": completed_at_utc,
        },
        hash_field="summary_sha256",
    )
    name = (
        "batch_aborted.json"
        if conclusion == "infrastructure_failed"
        else "batch_summary.json"
    )
    return write_canonical_json_exclusive(root / name, payload)


def _failure_payload(
    *, stage: str, action: Mapping[str, Any], error: BaseException
) -> dict[str, Any]:
    return {
        "stage": stage,
        "action_index": action["action_index"],
        "original_profile_order_index": action["original_profile_order_index"],
        "type": type(error).__name__,
        "message": str(error),
    }


def run_h23_demand1_exact_reuse_batch(
    config: H23Demand1ExactReuseBatchRunConfig,
    *,
    hooks: H23Demand1ExactReuseBatchHooks = H23Demand1ExactReuseBatchHooks(),
) -> H23Demand1ExactReuseBatchRunResult:
    """Run the fixed six actions serially with exact-signature preflights."""

    plan = load_h23_demand1_exact_reuse_plan(config.plan_path)
    policy = _registered_policy()
    settings = _registered_settings()
    geometry_policy = _registered_geometry_policy()
    with BenchmarkResourceLock(
        config.resource_lock, command="h23-demand1-exact-reuse-batch"
    ):
        environment = dict(hooks.capture_environment(_REPOSITORY_ROOT))
        _validate_runtime_environment(environment, plan)
        environment_sha = canonical_sha256(environment)
        stable_environment_sha = canonical_sha256(_stable_environment(environment))
        initial_process_audit = _process_audit(hooks.process_table())
        if not initial_process_audit["valid"]:
            raise H23Demand1ExactReuseBatchError(
                "conflicting solver/test processes are active"
            )
        required = plan.payload["required_execution_environment"]
        run_identity = {
            "run_schema": H23_DEMAND1_REUSE_BATCH_RUN_SCHEMA,
            **dict(_BOUNDARY_FLAGS),
            "contains_algorithm_comparison_results": False,
            "plan_id": H23_DEMAND1_REUSE_BATCH_PLAN_ID,
            "plan_path": str(plan.path),
            "plan_sha256": plan.plan_sha256,
            "plan_file_sha256": plan.file_sha256,
            "order_bias": dict(_ORDER_BIAS),
            "execution_contract": dict(_EXECUTION_CONTRACT),
            "actions": [dict(action) for action in plan.actions],
            "environment": environment,
            "environment_sha256": environment_sha,
            "stable_environment_sha256": stable_environment_sha,
            "git_commit": required["git_commit"],
            "implementation_sha256": required["implementation"][
                "implementation_sha256"
            ],
            "initial_process_audit": initial_process_audit,
        }
        run_sha = canonical_sha256(run_identity)
        run_id = _RUN_ID_PREFIX + run_sha
        results_root = _claim_results_root(config.results_root, plan)
        checkpoints_root = results_root / "checkpoints"
        checkpoints_root.mkdir(exist_ok=False)
        run_identity_path = write_canonical_json_exclusive(
            results_root / "run_identity.json",
            {
                "artifact_type": _RUN_ARTIFACT_TYPE,
                "artifact_schema_version": 1,
                "batch_run_id": run_id,
                "run_identity": run_identity,
                "run_identity_sha256": run_sha,
            },
        )
        pool = DevelopmentQualificationIncumbentPool()
        checkpoint_paths: list[Path] = []
        checkpoint_records: list[dict[str, Any]] = []
        ledger: list[dict[str, Any]] = []
        anchor_signature: QualificationConstraintSignature | None = None

        def revalidate_environment() -> None:
            observed = dict(hooks.capture_environment(_REPOSITORY_ROOT))
            _validate_runtime_environment(observed, plan)
            if (
                canonical_sha256(_stable_environment(observed))
                != stable_environment_sha
            ):
                raise QualificationInfrastructureError(
                    "runtime environment changed during batch"
                )

        for action in plan.actions:
            stage = "process_conflict_check"
            process_table_error: Exception | None = None
            try:
                process_audit = _process_audit(hooks.process_table())
            except Exception as error:
                process_table_error = error
                process_message = (
                    f"process_table_unavailable: {type(error).__name__}: {error}"
                )
                process_audit = {
                    "process_table_sha256": hashlib.sha256(
                        process_message.encode("utf-8")
                    ).hexdigest(),
                    "conflicting_processes": [process_message],
                    "valid": False,
                }
            identity = {
                "batch_run_id": run_id,
                "run_identity_sha256": run_sha,
                "plan_id": H23_DEMAND1_REUSE_BATCH_PLAN_ID,
                "plan_sha256": plan.plan_sha256,
                "plan_file_sha256": plan.file_sha256,
                "environment_sha256": environment_sha,
                "git_commit": required["git_commit"],
                "action": dict(action),
                "process_audit": process_audit,
            }
            try:
                if process_table_error is not None:
                    raise QualificationInfrastructureError(
                        str(process_audit["conflicting_processes"][0])
                    ) from process_table_error
                if not process_audit["valid"]:
                    raise QualificationInfrastructureError(
                        "conflicting process appeared before batch action"
                    )
                stage = "pre_action_revalidation"
                revalidate_environment()
                _revalidate_action_inputs(plan, action)
                job = action["source_job"]
                source_job = job["source_job"]
                profile = plan.source_smoke_plan.source_h23_plan.manifest.profiles[
                    source_job["profile_order_index"]
                ].solve_profile()
                model_config = profile.effective_model_config()
                contract = QualificationModelContract(
                    **job["qualification_model_contract"]
                )
                stage = "instance_load"
                instance = hooks.load_instance(
                    Path(str(job["physical_instance_directory"])),
                    model_config,
                    profile.demand_scale,
                )
                if (
                    getattr(instance, "config", None) != model_config
                    or not callable(getattr(instance, "screening_is_exact", None))
                    or instance.screening_is_exact("MX") is not False
                ):
                    raise QualificationInfrastructureError("loaded instance drifted")
                metadata = getattr(instance, "metadata", None)
                observed_demand = (
                    metadata.get("demand_scale")
                    if isinstance(metadata, Mapping)
                    else None
                )
                if (
                    isinstance(observed_demand, bool)
                    or not isinstance(observed_demand, (int, float))
                    or float(observed_demand) != 1.0
                ):
                    raise QualificationInfrastructureError(
                        "loaded demand scale drifted"
                    )
                preflight = None
                comparison = None
                verified_preflight_signature: QualificationConstraintSignature | None = (
                    None
                )
                if action["role"] == "exact_reuse_target":
                    if anchor_signature is None:
                        raise QualificationInfrastructureError(
                            "target lacks anchor signature"
                        )
                    stage = "exact_signature_preflight"
                    preflight, comparison = exact_target_signature_preflight(
                        instance,
                        contract=contract,
                        anchor_signature=anchor_signature,
                        target_demand_scale=profile.demand_scale,
                        adapter_factory=hooks.adapter_factory,
                    )
                    verified_preflight_signature = QualificationConstraintSignature(
                        **preflight["constraint_signature"]
                    )
                stage = "qualification"
                outcome = hooks.qualify_instance(
                    instance,
                    "MX",
                    case_id=action["batch_case_id"],
                    scale="small",
                    qualification_run_id=run_id,
                    instance_sha256=H23_REUSE_SMOKE_PHYSICAL_INSTANCE_SHA256,
                    policy=policy,
                    settings=settings,
                    geometry_start_policy=geometry_policy,
                    expected_model_contract=contract,
                    fail_fast_infrastructure=True,
                    development_incumbent_reuse_pool=pool,
                    development_reuse_profile_id=profile.profile_id,
                    development_expected_target_constraint_signature=(
                        verified_preflight_signature
                    ),
                )
                classification = _validate_outcome(
                    outcome, action=action, run_id=run_id, policy=policy
                )
                case_id = str(action["batch_case_id"])
                signature = _case_signature(pool, case_id)
                events = _case_events(pool, case_id)
                selected_source = _selected_source(pool, case_id)
                if action["role"] == "anchor_source":
                    anchor_signature = signature
                    expected_events = (
                        ("reuse_cache_miss", "incumbent_offered")
                        if outcome.accepted
                        else ("reuse_cache_miss",)
                    )
                else:
                    if (
                        preflight is None
                        or comparison is None
                        or verified_preflight_signature is None
                    ):
                        raise QualificationInfrastructureError(
                            "target preflight audit is missing"
                        )
                    preflight_signature = verified_preflight_signature
                    if signature != preflight_signature or selected_source is None:
                        raise QualificationInfrastructureError(
                            "qualification signature/source differs from preflight"
                        )
                    expected_events = (
                        "reuse_candidate_selected",
                        "reuse_start_installed",
                        *(("incumbent_offered",) if outcome.accepted else ()),
                    )
                if events != expected_events:
                    raise QualificationInfrastructureError(
                        "reuse event sequence drifted"
                    )
                result = {
                    "status": "completed",
                    "completed_at_utc": hooks.now_utc(),
                    "infrastructure_error": None,
                    "signature_preflight": preflight,
                    "model_audit": outcome.witness_model_audit.to_payload(),
                    "constraint_signature": signature.to_payload(),
                    "signature_comparison_to_anchor": comparison,
                    "reuse_case_audit": pool.case_audit_payload(case_id),
                    "reuse_pool_audit_snapshot": pool.to_payload(),
                    "reuse_candidate_selected": selected_source is not None,
                    "reuse_start_installation_attempted": (
                        "reuse_start_installed" in events
                    ),
                    "reuse_start_installed": "reuse_start_installed" in events,
                    "selected_source": selected_source,
                    "solve_started": True,
                    "fresh_validation_performed": (
                        outcome.fresh_model_validation_performed
                    ),
                    "adapter_disposed": True,
                    "runtime_classification": classification,
                    "qualification_outcome": outcome.to_payload(),
                }
                stage = "post_action_revalidation"
                _revalidate_action_inputs(plan, action)
                revalidate_environment()
                checkpoint_path, checkpoint_sha, checkpoint_file_sha = (
                    _write_checkpoint(
                        checkpoints_root, identity=identity, result=result
                    )
                )
            except BaseException as error:
                failure = _failure_payload(stage=stage, action=action, error=error)
                failed_result = {
                    "status": "infrastructure_failure",
                    "completed_at_utc": hooks.now_utc(),
                    "infrastructure_error": failure,
                    "signature_preflight": None,
                    "model_audit": None,
                    "constraint_signature": None,
                    "signature_comparison_to_anchor": None,
                    "reuse_case_audit": pool.case_audit_payload(
                        str(action["batch_case_id"])
                    ),
                    "reuse_pool_audit_snapshot": pool.to_payload(),
                    "reuse_candidate_selected": False,
                    "reuse_start_installation_attempted": False,
                    "reuse_start_installed": False,
                    "selected_source": None,
                    "solve_started": None,
                    "fresh_validation_performed": False,
                    "adapter_disposed": None,
                    "runtime_classification": None,
                    "qualification_outcome": None,
                }
                checkpoint_path, checkpoint_sha, checkpoint_file_sha = (
                    _write_checkpoint(
                        checkpoints_root, identity=identity, result=failed_result
                    )
                )
                checkpoint_paths.append(checkpoint_path)
                record = {
                    "action_index": action["action_index"],
                    "original_profile_order_index": action[
                        "original_profile_order_index"
                    ],
                    "path": str(checkpoint_path),
                    "checkpoint_sha256": checkpoint_sha,
                    "file_sha256": checkpoint_file_sha,
                    "status": "infrastructure_failure",
                }
                checkpoint_records.append(record)
                ledger.append(_summary_ledger_entry(action, failed_result))
                summary_path = _write_summary(
                    results_root,
                    run_id=run_id,
                    run_sha=run_sha,
                    plan=plan,
                    conclusion="infrastructure_failed",
                    checkpoint_records=checkpoint_records,
                    ledger=ledger,
                    pool=pool,
                    completed_at_utc=hooks.now_utc(),
                    infrastructure_failure=failure,
                )
                if isinstance(error, (MemoryError, KeyboardInterrupt, SystemExit)):
                    raise
                raise H23Demand1ExactReuseBatchInfrastructureError(
                    f"batch failed during {stage}; artifacts: {results_root}"
                ) from error
            checkpoint_paths.append(checkpoint_path)
            checkpoint_records.append(
                {
                    "action_index": action["action_index"],
                    "original_profile_order_index": action[
                        "original_profile_order_index"
                    ],
                    "path": str(checkpoint_path),
                    "checkpoint_sha256": checkpoint_sha,
                    "file_sha256": checkpoint_file_sha,
                    "status": "completed",
                }
            )
            ledger.append(_summary_ledger_entry(action, result))
            if action["role"] == "anchor_source" and not outcome.accepted:
                summary_path = _write_summary(
                    results_root,
                    run_id=run_id,
                    run_sha=run_sha,
                    plan=plan,
                    conclusion="source_unavailable",
                    checkpoint_records=checkpoint_records,
                    ledger=ledger,
                    pool=pool,
                    completed_at_utc=hooks.now_utc(),
                )
                return H23Demand1ExactReuseBatchRunResult(
                    results_root=results_root,
                    run_identity_path=run_identity_path,
                    summary_path=summary_path,
                    batch_run_id=run_id,
                    conclusion="source_unavailable",
                    checkpoint_paths=tuple(checkpoint_paths),
                )

        summary_path = _write_summary(
            results_root,
            run_id=run_id,
            run_sha=run_sha,
            plan=plan,
            conclusion="completed",
            checkpoint_records=checkpoint_records,
            ledger=ledger,
            pool=pool,
            completed_at_utc=hooks.now_utc(),
        )
        return H23Demand1ExactReuseBatchRunResult(
            results_root=results_root,
            run_identity_path=run_identity_path,
            summary_path=summary_path,
            batch_run_id=run_id,
            conclusion="completed",
            checkpoint_paths=tuple(checkpoint_paths),
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.benchmark.branch_cut.h23_demand1_exact_reuse_batch",
        description=(
            "Build or run the development-only H23 demand-1 exact-reuse batch. "
            "The plan command is solver-free."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    plan = subparsers.add_parser("plan")
    plan.add_argument("--verified-smoke-plan", required=True)
    plan.add_argument("--verified-smoke-results-root", required=True)
    plan.add_argument("--output", required=True)
    plan.add_argument("--expected-git-commit", required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--plan", required=True)
    run.add_argument("--results-root", required=True)
    run.add_argument("--resource-lock", default=str(DEFAULT_BENCHMARK_RESOURCE_LOCK))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "plan":
            result = build_h23_demand1_exact_reuse_plan(
                H23Demand1ExactReusePlanBuildConfig(
                    verified_smoke_plan_path=Path(args.verified_smoke_plan),
                    verified_smoke_results_root=Path(args.verified_smoke_results_root),
                    output_path=Path(args.output),
                    expected_git_commit=args.expected_git_commit,
                )
            )
        else:
            result = run_h23_demand1_exact_reuse_batch(
                H23Demand1ExactReuseBatchRunConfig(
                    plan_path=Path(args.plan),
                    results_root=Path(args.results_root),
                    resource_lock=Path(args.resource_lock),
                )
            )
    except (
        H23Demand1ExactReuseBatchError,
        H23Demand1ExactReuseBatchInfrastructureError,
        OSError,
        ValueError,
    ) as error:
        print(f"H23 demand-1 exact-reuse batch failed: {error}", file=sys.stderr)
        return 2
    print(canonical_json(result.to_payload()))
    return 0


__all__ = [
    "H23Demand1ExactReuseBatchError",
    "H23Demand1ExactReuseBatchHooks",
    "H23Demand1ExactReuseBatchInfrastructureError",
    "H23Demand1ExactReuseBatchRunConfig",
    "H23Demand1ExactReuseBatchRunResult",
    "H23Demand1ExactReusePlanBuildConfig",
    "H23Demand1ExactReusePlanBuildHooks",
    "H23Demand1ExactReusePlanBuildResult",
    "H23_DEMAND1_REUSE_BATCH_ACTION_INDICES",
    "H23_DEMAND1_REUSE_BATCH_PLAN_ID",
    "build_h23_demand1_exact_reuse_plan",
    "build_parser",
    "classify_outcome_payload",
    "load_h23_demand1_exact_reuse_plan",
    "main",
    "run_h23_demand1_exact_reuse_batch",
]


if __name__ == "__main__":
    raise SystemExit(main())
