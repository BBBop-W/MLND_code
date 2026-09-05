"""Independent solver-free verifier for the H23 demand-1 reuse batch."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Mapping, Sequence

from .hashing import canonical_json, canonical_sha256
from .historical_paths import preserved_development_root
from .h23_incumbent_reuse_smoke import (
    H23_REUSE_SMOKE_PHYSICAL_INSTANCE_ID,
    H23_REUSE_SMOKE_PHYSICAL_INSTANCE_SHA256,
    H23IncumbentReuseSmokeError,
    load_h23_incumbent_reuse_smoke_plan,
)
import src.benchmark.branch_cut.h23_incumbent_reuse_smoke_verifier as smoke_verify
from .h23_incumbent_reuse_contract import (
    REGISTERED_GEOMETRY_POLICY_PAYLOAD,
    REGISTERED_NATIVE_SETTINGS_PAYLOAD,
    REGISTERED_POLICY_PAYLOAD,
)
from .qualification import QualificationEvidence, QualificationPolicy, assess_qualification
from .qualification_incumbent_reuse import (
    QUALIFICATION_INCUMBENT_REUSE_POLICY_VERSION,
    QualificationConstraintSignature,
)


_PLAN_ID = "h23-demand1-exact-reuse-batch-v2"
_PLAN_SCHEMA = "mlnd-h23-demand1-exact-reuse-plan-v2"
_RUN_SCHEMA = "mlnd-h23-demand1-exact-reuse-run-v2"
_CHECKPOINT_SCHEMA = "mlnd-h23-demand1-exact-reuse-checkpoint-v2"
_SUMMARY_SCHEMA = "mlnd-h23-demand1-exact-reuse-summary-v2"
_ACTION_INDICES = (4, 0, 1, 2, 3, 5)
_RUN_ID_PREFIX = "h23-demand1-exact-reuse-v2-run-"
_PLAN_ARTIFACT_TYPE = "mlnd_branch_cut_h23_demand1_exact_reuse_plan"
_RUN_ARTIFACT_TYPE = "mlnd_branch_cut_h23_demand1_exact_reuse_run_identity"
_CHECKPOINT_ARTIFACT_TYPE = "mlnd_branch_cut_h23_demand1_exact_reuse_checkpoint"
_SUMMARY_ARTIFACT_TYPE = "mlnd_branch_cut_h23_demand1_exact_reuse_summary"
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

_VERIFIED_SMOKE_PLAN_SHA = (
    "38ce31ff307b197b4277682a3aea7d522dc928881232b894cfdc6cb66b01fc11"
)
_VERIFIED_SMOKE_PLAN_FILE_SHA = (
    "d26652b3896ff85ca92db55ab7856d39969ad163084365749d718192d05d2a63"
)
_VERIFIED_SMOKE_RUN_ID = (
    "h23-incumbent-reuse-smoke-run-"
    "d16fd5e3102e01435edc4794a4918c2a133f708fd474eea71d928ab9831ed491"
)
_VERIFIED_SMOKE_SUMMARY_SHA = (
    "70c748a7d147b145fc981dd5aaeb3c34968b5f826ae7087676da608103c6ad51"
)
_VERIFIED_SMOKE_SUMMARY_FILE_SHA = (
    "c60684b4f534672a602c1525ff7dae3781584a3c20e39b7a2406ccef2f306731"
)
_VERIFIED_SMOKE_CHECKPOINT_FILE_SHAS = (
    "32c66aee935f6ba39974513cf7dcd6c79aa6c855b52572a77da8c8866c59f25e",
    "e92adb2a9fe234b2252b3bae09b475d35e4674d041d7705235652e77c01bd70b",
    "e479b9275c8ebb8fc327845db6de7dabab7d86b34993b7843f0e19197fff1700",
)

_IMPLEMENTATION_FILES = {
    "src/benchmark/branch_cut/historical_paths.py",
    "src/benchmark/branch_cut/h23_incumbent_reuse_contract.py",
    "src/benchmark/branch_cut/qualification_incumbent_reuse.py",
    "src/benchmark/branch_cut/qualification_runner.py",
    "src/benchmark/branch_cut/h23_incumbent_reuse_smoke.py",
    "src/benchmark/branch_cut/h23_demand1_exact_reuse_batch.py",
    "src/benchmark/branch_cut/h23_demand1_exact_reuse_batch_verifier.py",
}

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
    "execution_order_original_indices": list(_ACTION_INDICES),
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
    "fixed_original_profile_order_indices": list(_ACTION_INDICES),
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

_SHA_RE = re.compile(r"[0-9a-f]{64}")
_COMMIT_RE = re.compile(r"[0-9a-f]{40}")


class H23Demand1ExactReuseVerificationError(RuntimeError):
    """A plan or result assertion did not independently verify."""


def _fail(message: str) -> None:
    raise H23Demand1ExactReuseVerificationError(message)


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
        _fail(f"cannot read {field_name}: {error}")
    if not isinstance(payload, dict):
        _fail(f"{field_name} must contain one object")
    try:
        expected = canonical_json(payload) + "\n"
    except (TypeError, ValueError) as error:
        _fail(f"{field_name} is not canonical-JSON-safe: {error}")
    if raw != expected:
        _fail(f"{field_name} is not canonical JSON")
    return payload


def _mapping(value: object, *, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(k, str) for k in value):
        _fail(f"{field_name} must be an object")
    return value


def _list(value: object, *, field_name: str) -> list[Any]:
    if not isinstance(value, list):
        _fail(f"{field_name} must be a list")
    return value


def _exact_keys(
    value: Mapping[str, Any], expected: set[str], *, field_name: str
) -> None:
    if set(value) != expected:
        _fail(
            f"{field_name} fields differ: missing={sorted(expected - set(value))}, "
            f"extra={sorted(set(value) - expected)}"
        )


def _self_hash(value: Mapping[str, Any], *, hash_field: str, field_name: str) -> str:
    observed = value.get(hash_field)
    if not isinstance(observed, str) or _SHA_RE.fullmatch(observed) is None:
        _fail(f"{field_name} has invalid {hash_field}")
    preimage = {key: item for key, item in value.items() if key != hash_field}
    if canonical_sha256(preimage) != observed:
        _fail(f"{field_name} self-hash mismatch")
    return observed


def _verify_superseded_v1_evidence() -> None:
    path = Path(str(_SUPERSEDES["plan_path"]))
    payload = _read_canonical(path.resolve(strict=True), field_name="superseded v1 plan")
    required = _mapping(
        payload.get("required_execution_environment"),
        field_name="superseded required environment",
    )
    implementation = _mapping(
        required.get("implementation"),
        field_name="superseded implementation",
    )
    if (
        payload.get("plan_id") != _SUPERSEDES["plan_id"]
        or payload.get("plan_sha256") != _SUPERSEDES["plan_sha256"]
        or _sha256_file(path) != _SUPERSEDES["plan_file_sha256"]
        or required.get("git_commit") != _SUPERSEDES["git_commit"]
        or implementation.get("implementation_sha256")
        != _SUPERSEDES["implementation_sha256"]
    ):
        _fail("superseded v1 plan evidence drifted")
    old_root = Path(str(_SUPERSEDES["unused_run_root"]))
    if old_root.exists() or old_root.is_symlink():
        _fail("superseded v1 run root is not preserved as never-created")


def _git(repository_root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ("git", *arguments),
        cwd=repository_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        _fail(
            f"git {' '.join(arguments)} failed: "
            + (completed.stderr or completed.stdout).strip()
        )
    return completed.stdout.strip()


def _verify_boundary(payload: Mapping[str, Any], *, field_name: str) -> None:
    if any(payload.get(key) != value for key, value in _BOUNDARY_FLAGS.items()):
        _fail(f"{field_name} development/formal boundary drifted")


def _classify(payload: Mapping[str, Any]) -> dict[str, Any]:
    evidence = _mapping(payload.get("evidence"), field_name="outcome evidence")
    candidate_h = evidence.get("candidate_h")
    open_h = evidence.get("open_h")
    active_h = evidence.get("active_h")
    captured = payload.get("captured_variable_count")
    if (
        any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in (candidate_h, open_h, captured)
        )
        or open_h > candidate_h
    ):
        _fail("outcome counts are invalid")
    if active_h is not None and (
        isinstance(active_h, bool)
        or not isinstance(active_h, int)
        or active_h < 0
        or active_h > open_h
    ):
        _fail("active H count is invalid")
    reasons = _list(payload.get("rejection_reasons"), field_name="rejection reasons")
    if any(not isinstance(reason, str) for reason in reasons):
        _fail("rejection reasons must be strings")
    accepted = payload.get("accepted")
    fresh = payload.get("fresh_model_validation_performed")
    if not isinstance(accepted, bool) or not isinstance(fresh, bool):
        _fail("outcome booleans are invalid")
    return {
        "accepted": accepted,
        "has_incumbent": captured > 0,
        "captured_variable_count": captured,
        "fresh_model_validation_performed": fresh,
        "fresh_validation_required": captured > 0,
        "fresh_validation_requirement_satisfied": captured == 0 or fresh,
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


def _verify_smoke_binding(binding: Mapping[str, Any]) -> object:
    expected_keys = {
        "verifier_status",
        "verifier_conclusion",
        "plan_path",
        "plan_sha256",
        "plan_file_sha256",
        "results_root",
        "run_identity_path",
        "run_identity_file_sha256",
        "smoke_run_id",
        "summary_path",
        "summary_sha256",
        "summary_file_sha256",
        "checkpoint_file_sha256",
        "mechanism_only_no_speedup_claim",
    }
    _exact_keys(binding, expected_keys, field_name="verified smoke binding")
    smoke_plan = load_h23_incumbent_reuse_smoke_plan(Path(binding["plan_path"]))
    summary_path = Path(binding["summary_path"]).resolve(strict=True)
    run_path = Path(binding["run_identity_path"]).resolve(strict=True)
    root = Path(binding["results_root"]).resolve(strict=True)
    summary = _read_canonical(summary_path, field_name="verified smoke summary")
    run = _read_canonical(run_path, field_name="verified smoke run identity")
    records = _list(summary.get("checkpoints"), field_name="smoke checkpoints")
    checkpoint_shas = tuple(
        _sha256_file(Path(record["path"]).resolve(strict=True)) for record in records
    )
    expected = {
        "verifier_status": "verified",
        "verifier_conclusion": "passed",
        "plan_path": str(smoke_plan.path),
        "plan_sha256": _VERIFIED_SMOKE_PLAN_SHA,
        "plan_file_sha256": _VERIFIED_SMOKE_PLAN_FILE_SHA,
        "results_root": str(root),
        "run_identity_path": str(run_path),
        "run_identity_file_sha256": _sha256_file(run_path),
        "smoke_run_id": _VERIFIED_SMOKE_RUN_ID,
        "summary_path": str(summary_path),
        "summary_sha256": _VERIFIED_SMOKE_SUMMARY_SHA,
        "summary_file_sha256": _VERIFIED_SMOKE_SUMMARY_FILE_SHA,
        "checkpoint_file_sha256": list(_VERIFIED_SMOKE_CHECKPOINT_FILE_SHAS),
        "mechanism_only_no_speedup_claim": True,
    }
    if (
        dict(binding) != expected
        or smoke_plan.file_sha256 != _VERIFIED_SMOKE_PLAN_FILE_SHA
        or _sha256_file(summary_path) != _VERIFIED_SMOKE_SUMMARY_FILE_SHA
        or summary.get("summary_sha256") != _VERIFIED_SMOKE_SUMMARY_SHA
        or summary.get("smoke_run_id") != _VERIFIED_SMOKE_RUN_ID
        or summary.get("conclusion") != "passed"
        or summary.get("passed") is not True
        or run.get("smoke_run_id") != _VERIFIED_SMOKE_RUN_ID
        or checkpoint_shas != _VERIFIED_SMOKE_CHECKPOINT_FILE_SHAS
        or summary_path.parent != root
        or run_path.parent != root
    ):
        _fail("verified v1 smoke binding/evidence drifted")
    return smoke_plan


def _verify_actions(
    plan: Mapping[str, Any], smoke_plan: object
) -> tuple[Mapping[str, Any], ...]:
    actions = _list(plan.get("actions"), field_name="batch actions")
    if len(actions) != 6:
        _fail("batch must contain exactly six actions")
    summary_path = Path(plan["source_h23"]["summary_path"]).resolve(strict=True)
    source_summary = _read_canonical(summary_path, field_name="source H23 summary")
    ledger = {
        int(record["diagnostic_job_order_index"]): record
        for record in source_summary["diagnostic_job_ledger"]
    }
    verified: list[Mapping[str, Any]] = []
    for position, (raw_action, original_index) in enumerate(
        zip(actions, _ACTION_INDICES, strict=True)
    ):
        action = _mapping(raw_action, field_name=f"action {position}")
        expected_keys = {
            "action_index",
            "role",
            "original_profile_order_index",
            "expected_signature_relation_to_anchor",
            "batch_case_id",
            "source_job",
            "historical_checkpoint",
            "expected_event_prefix",
            "incumbent_offer_expected_iff_accepted",
        }
        _exact_keys(action, expected_keys, field_name=f"action {position}")
        role = "anchor_source" if position == 0 else "exact_reuse_target"
        relation = "establish" if position == 0 else "equal"
        expected_case = "h23-demand1-exact-reuse-case-" + canonical_sha256(
            {
                "plan_id": _PLAN_ID,
                "action_index": position,
                "original_profile_order_index": original_index,
                "diagnostic_job_id": smoke_plan.source_h23_plan.jobs[original_index][
                    "diagnostic_job_id"
                ],
            }
        )
        if (
            action["action_index"] != position
            or action["role"] != role
            or action["original_profile_order_index"] != original_index
            or action["expected_signature_relation_to_anchor"] != relation
            or action["batch_case_id"] != expected_case
            or canonical_json(action["source_job"])
            != canonical_json(smoke_plan.source_h23_plan.jobs[original_index])
            or action["incumbent_offer_expected_iff_accepted"] is not True
            or action["expected_event_prefix"]
            != (
                ["reuse_cache_miss"]
                if position == 0
                else ["reuse_candidate_selected", "reuse_start_installed"]
            )
        ):
            _fail(f"action {position} identity/order drifted")
        job = action["source_job"]
        source_job = job["source_job"]
        contract = job["qualification_model_contract"]
        if (
            source_job["instance_sha256"] != H23_REUSE_SMOKE_PHYSICAL_INSTANCE_SHA256
            or source_job["physical_instance_id"]
            != H23_REUSE_SMOKE_PHYSICAL_INSTANCE_ID
            or source_job["profile_order_index"] != original_index
            or float(job["qualification_solve_profile"]["demand_scale"]) != 1.0
            or float(contract["demand_scale"]) != 1.0
            or contract["screening_is_exact"] is not False
        ):
            _fail(f"action {position} is not the exact H23 demand-1 job")
        history = _mapping(
            action["historical_checkpoint"], field_name="historical checkpoint"
        )
        _exact_keys(
            history,
            {"path", "checkpoint_sha256", "file_sha256", "classification"},
            field_name="historical checkpoint",
        )
        historical_record = ledger[original_index]
        historical_outcome = _mapping(
            historical_record["outcome"], field_name="historical outcome"
        )
        if (
            history["path"] != historical_record["checkpoint_path"]
            or history["checkpoint_sha256"] != historical_record["checkpoint_sha256"]
            or _sha256_file(Path(history["path"]).resolve(strict=True))
            != history["file_sha256"]
            or history["classification"] != _classify(historical_outcome)
        ):
            _fail(f"action {position} historical failure/outcome drifted")
        verified.append(action)
    if verified[0]["historical_checkpoint"]["classification"]["accepted"] is not True:
        _fail("anchor historical acceptance drifted")
    return tuple(verified)


def _verify_plan(
    path: Path,
) -> tuple[Mapping[str, Any], object, tuple[Mapping[str, Any], ...]]:
    _verify_superseded_v1_evidence()
    plan = _read_canonical(path.resolve(strict=True), field_name="batch plan")
    expected_keys = {
        "artifact_type",
        "artifact_schema_version",
        "plan_schema",
        "plan_id",
        *_BOUNDARY_FLAGS,
        "mode",
        "action_count",
        "reuse_policy_version",
        "supersedes",
        "order_bias",
        "execution_contract",
        "qualification_policy",
        "native_settings",
        "geometry_start_policy",
        "verified_v1_mechanism_smoke",
        "source_h23",
        "required_execution_environment",
        "actions",
        "plan_sha256",
    }
    _exact_keys(plan, expected_keys, field_name="batch plan")
    _self_hash(plan, hash_field="plan_sha256", field_name="batch plan")
    _verify_boundary(plan, field_name="batch plan")
    fixed = {
        "artifact_type": _PLAN_ARTIFACT_TYPE,
        "artifact_schema_version": 1,
        "plan_schema": _PLAN_SCHEMA,
        "plan_id": _PLAN_ID,
        "mode": "MX",
        "action_count": 6,
        "reuse_policy_version": QUALIFICATION_INCUMBENT_REUSE_POLICY_VERSION,
        "supersedes": dict(_SUPERSEDES),
        "order_bias": dict(_ORDER_BIAS),
        "execution_contract": dict(_EXECUTION_CONTRACT),
        "qualification_policy": dict(REGISTERED_POLICY_PAYLOAD),
        "native_settings": dict(REGISTERED_NATIVE_SETTINGS_PAYLOAD),
        "geometry_start_policy": dict(REGISTERED_GEOMETRY_POLICY_PAYLOAD),
    }
    if any(plan.get(key) != value for key, value in fixed.items()):
        _fail("batch plan fixed contract drifted")
    smoke_plan = _verify_smoke_binding(
        _mapping(
            plan["verified_v1_mechanism_smoke"], field_name="verified smoke binding"
        )
    )
    if plan["source_h23"] != smoke_plan.payload["source_h23"]:
        _fail("source H23 binding differs from verified smoke")
    actions = _verify_actions(plan, smoke_plan)
    return plan, smoke_plan, actions


def _verify_implementation(
    plan: Mapping[str, Any], *, expected_commit: str, expected_sha: str
) -> Mapping[str, Any]:
    required = _mapping(
        plan["required_execution_environment"], field_name="required environment"
    )
    _exact_keys(
        required,
        {
            "repository_root",
            "git_commit",
            "clean_status_includes_untracked",
            "implementation",
        },
        field_name="required environment",
    )
    implementation = _mapping(
        required["implementation"], field_name="required implementation"
    )
    _exact_keys(
        implementation,
        {"ordered_file_sha256", "implementation_sha256"},
        field_name="required implementation",
    )
    files = _mapping(
        implementation["ordered_file_sha256"], field_name="implementation files"
    )
    repository_root = Path(required["repository_root"]).resolve(strict=True)
    observed = {
        relative: _sha256_file(repository_root / relative) for relative in files
    }
    if (
        set(files) != _IMPLEMENTATION_FILES
        or required["git_commit"] != expected_commit
        or required["clean_status_includes_untracked"] is not True
        or _git(repository_root, "rev-parse", "HEAD") != expected_commit
        or _git(repository_root, "status", "--porcelain", "--untracked-files=all") != ""
        or observed != files
        or canonical_sha256(observed) != implementation["implementation_sha256"]
        or implementation["implementation_sha256"] != expected_sha
    ):
        _fail("current clean commit/implementation does not match plan")
    return implementation


@dataclass(frozen=True, slots=True)
class H23Demand1ExactReusePlanVerificationRequest:
    plan_path: Path
    expected_git_commit: str
    expected_implementation_sha256: str

    def __post_init__(self) -> None:
        if _COMMIT_RE.fullmatch(self.expected_git_commit) is None:
            raise ValueError("expected_git_commit must be a Git digest")
        if _SHA_RE.fullmatch(self.expected_implementation_sha256) is None:
            raise ValueError("expected_implementation_sha256 must be SHA256")


@dataclass(frozen=True, slots=True)
class H23Demand1ExactReusePlanVerificationReport:
    status: str
    plan_path: Path
    plan_id: str
    plan_sha256: str
    plan_file_sha256: str
    git_commit: str
    implementation_sha256: str
    execution_order_original_indices: tuple[int, ...]

    def to_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["plan_path"] = str(self.plan_path)
        payload["execution_order_original_indices"] = list(
            self.execution_order_original_indices
        )
        return payload


def verify_h23_demand1_exact_reuse_plan(
    request: H23Demand1ExactReusePlanVerificationRequest,
) -> H23Demand1ExactReusePlanVerificationReport:
    plan_path = request.plan_path.expanduser().resolve(strict=True)
    plan, _smoke_plan, _actions = _verify_plan(plan_path)
    implementation = _verify_implementation(
        plan,
        expected_commit=request.expected_git_commit,
        expected_sha=request.expected_implementation_sha256,
    )
    return H23Demand1ExactReusePlanVerificationReport(
        status="verified_plan",
        plan_path=plan_path,
        plan_id=_PLAN_ID,
        plan_sha256=plan["plan_sha256"],
        plan_file_sha256=_sha256_file(plan_path),
        git_commit=request.expected_git_commit,
        implementation_sha256=implementation["implementation_sha256"],
        execution_order_original_indices=_ACTION_INDICES,
    )


def _call_smoke_helper(function: Any, *args: Any, **kwargs: Any) -> Any:
    try:
        return function(*args, **kwargs)
    except smoke_verify.H23IncumbentReuseSmokeVerificationError as error:
        _fail(str(error))


def _verify_run_identity(
    path: Path,
    *,
    plan: Mapping[str, Any],
    plan_path: Path,
    actions: Sequence[Mapping[str, Any]],
    expected_run_id: str,
) -> tuple[Mapping[str, Any], str]:
    payload = _read_canonical(path, field_name="run identity")
    _exact_keys(
        payload,
        {
            "artifact_type",
            "artifact_schema_version",
            "batch_run_id",
            "run_identity",
            "run_identity_sha256",
        },
        field_name="run identity",
    )
    identity = _mapping(payload["run_identity"], field_name="run identity body")
    run_sha = payload["run_identity_sha256"]
    if not isinstance(run_sha, str) or canonical_sha256(identity) != run_sha:
        _fail("run identity hash does not recompute")
    run_id = _RUN_ID_PREFIX + run_sha
    expected_keys = {
        "run_schema",
        *_BOUNDARY_FLAGS,
        "contains_algorithm_comparison_results",
        "plan_id",
        "plan_path",
        "plan_sha256",
        "plan_file_sha256",
        "order_bias",
        "execution_contract",
        "actions",
        "environment",
        "environment_sha256",
        "stable_environment_sha256",
        "git_commit",
        "implementation_sha256",
        "initial_process_audit",
    }
    _exact_keys(identity, expected_keys, field_name="run identity body")
    _verify_boundary(identity, field_name="run identity")
    environment = _mapping(identity["environment"], field_name="run environment")
    stable = dict(environment)
    repository = dict(_mapping(stable["repository"], field_name="run repository"))
    repository.pop("status_lines", None)
    repository.pop("worktree_dirty", None)
    stable["repository"] = repository
    process = _mapping(
        identity["initial_process_audit"], field_name="initial process audit"
    )
    _exact_keys(
        process,
        {"process_table_sha256", "conflicting_processes", "valid"},
        field_name="initial process audit",
    )
    required = _mapping(
        plan["required_execution_environment"], field_name="required environment"
    )
    environment_repository = _mapping(
        environment["repository"], field_name="runtime repository"
    )
    if (
        payload["artifact_type"] != _RUN_ARTIFACT_TYPE
        or payload["artifact_schema_version"] != 1
        or payload["batch_run_id"] != run_id
        or run_id != expected_run_id
        or identity["run_schema"] != _RUN_SCHEMA
        or identity["contains_algorithm_comparison_results"] is not False
        or identity["plan_id"] != _PLAN_ID
        or identity["plan_path"] != str(plan_path)
        or identity["plan_file_sha256"] != _sha256_file(plan_path)
        or identity["git_commit"] != required["git_commit"]
        or identity["implementation_sha256"]
        != required["implementation"]["implementation_sha256"]
        or environment_repository.get("repository_root") != required["repository_root"]
        or environment_repository.get("git_commit") != required["git_commit"]
        or environment_repository.get("implementation") != required["implementation"]
        or environment_repository.get("status_includes_untracked") is not True
        or environment_repository.get("worktree_dirty") is not False
        or environment_repository.get("status_lines") != []
    ):
        _fail("run identity type/id boundary drifted")
    if (
        identity["plan_sha256"] != plan["plan_sha256"]
        or identity["order_bias"] != _ORDER_BIAS
        or identity["execution_contract"] != _EXECUTION_CONTRACT
        or identity["actions"] != list(actions)
        or canonical_sha256(environment) != identity["environment_sha256"]
        or canonical_sha256(stable) != identity["stable_environment_sha256"]
        or _SHA_RE.fullmatch(str(process.get("process_table_sha256"))) is None
        or process.get("valid") is not True
        or process.get("conflicting_processes") != []
    ):
        _fail("run identity plan/environment/process audit drifted")
    return identity, run_sha


def _expected_comparison(
    anchor: QualificationConstraintSignature,
    target: QualificationConstraintSignature,
) -> Mapping[str, Any]:
    return _call_smoke_helper(
        smoke_verify._recomputed_comparison,
        role="exact_reuse_target",
        expected_relation="equal",
        source=anchor,
        target=target,
        source_demand_scale=1.0,
        target_demand_scale=1.0,
    )


_MODEL_AUDIT_FIELDS = {
    "effective_model_config_sha256",
    "demand_scale",
    "locker_fixed_cost_per_hour",
    "display_origin_choices",
    "general_corridor_choices",
    "screening_is_exact",
    "model_fingerprint",
}
_MODEL_CONTRACT_FIELDS = {
    "qualification_case_sha256",
    "effective_model_config_sha256",
    "demand_scale",
    "locker_fixed_cost_per_hour",
    "display_origin_choices",
    "general_corridor_choices",
    "screening_is_exact",
}
_NORMAL_SOLVER_STATUSES = {"OPTIMAL", "TIME_LIMIT", "INFEASIBLE"}
_VALIDATED_INCUMBENT_STATUSES = {"OPTIMAL", "TIME_LIMIT"}
_NORMAL_RUNNER_REASONS = {
    "geometry_start_fallback_failed",
    "no_objective_improving_incumbent",
    "no_positive_service_incumbent",
}
_OUTCOME_FIELDS = {
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
}


def _verify_model_audit(
    value: object,
    *,
    contract: Mapping[str, Any],
    field_name: str,
) -> Mapping[str, Any]:
    audit = _mapping(value, field_name=field_name)
    _exact_keys(audit, _MODEL_AUDIT_FIELDS, field_name=field_name)
    for key in _MODEL_AUDIT_FIELDS - {"model_fingerprint"}:
        if audit[key] != contract[key]:
            _fail(f"{field_name} differs from frozen model contract at {key}")
    fingerprint = audit["model_fingerprint"]
    if isinstance(fingerprint, bool) or not isinstance(fingerprint, int):
        _fail(f"{field_name} model_fingerprint is invalid")
    return audit


def _reassess_outcome(
    outcome: Mapping[str, Any],
    *,
    action: Mapping[str, Any],
    plan: Mapping[str, Any],
    run_id: str,
) -> tuple[bool, Mapping[str, Any], Mapping[str, Any]]:
    _exact_keys(outcome, _OUTCOME_FIELDS, field_name="qualification outcome")
    contract = _mapping(
        action["source_job"]["qualification_model_contract"],
        field_name="frozen qualification contract",
    )
    expected_contract = _mapping(
        outcome["expected_model_contract"], field_name="outcome expected contract"
    )
    _exact_keys(
        expected_contract,
        _MODEL_CONTRACT_FIELDS,
        field_name="outcome expected contract",
    )
    if expected_contract != contract:
        _fail("outcome expected model contract differs from frozen action contract")

    witness_audit = _verify_model_audit(
        outcome["witness_model_audit"],
        contract=contract,
        field_name="witness model audit",
    )
    evidence_payload = _mapping(
        outcome["evidence"], field_name="qualification evidence"
    )
    try:
        evidence_values = dict(evidence_payload)
        for numeric_name in (
            "courier_only_objective",
            "witness_objective",
            "total_demand_per_hour",
            "witness_service_per_hour",
        ):
            if evidence_values.get(numeric_name) is None:
                evidence_values[numeric_name] = math.nan
        evidence = QualificationEvidence(**evidence_values)
        policy_values = dict(plan["qualification_policy"])
        target = policy_values.get("large_scale_open_h_target")
        policy_values["large_scale_open_h_target"] = tuple(target) if target else None
        policy = QualificationPolicy(**policy_values)
        recomputed = assess_qualification(evidence, policy).to_payload()
    except (TypeError, ValueError) as error:
        _fail(f"qualification outcome cannot be independently reassessed: {error}")
    if outcome["decision"] != recomputed:
        _fail("qualification decision does not independently recompute")
    if (
        evidence.case_id != action["batch_case_id"]
        or evidence.scale != "small"
        or evidence.qualification_run_id != run_id
        or evidence.instance_sha256 != H23_REUSE_SMOKE_PHYSICAL_INSTANCE_SHA256
        or evidence.qualification_case_sha256 != contract["qualification_case_sha256"]
        or evidence.effective_model_config_sha256
        != contract["effective_model_config_sha256"]
        or evidence.screening_is_exact is not False
    ):
        _fail("qualification evidence identity differs from frozen action")

    reasons = _list(outcome["rejection_reasons"], field_name="rejection reasons")
    exceptions = _list(outcome["exception_messages"], field_name="exception messages")
    if (
        any(not isinstance(item, str) or not item for item in reasons)
        or len(set(reasons)) != len(reasons)
        or exceptions != []
    ):
        _fail("completed qualification has invalid reasons or exception messages")
    decision_reasons = _list(
        recomputed["rejection_reasons"], field_name="recomputed rejection reasons"
    )
    if any(reason not in reasons for reason in decision_reasons) or any(
        reason not in decision_reasons and reason not in _NORMAL_RUNNER_REASONS
        for reason in reasons
    ):
        _fail("qualification rejection reasons are inconsistent with fail-fast mode")
    accepted = outcome["accepted"]
    if type(accepted) is not bool or accepted is not (not reasons):
        _fail("qualification accepted flag differs from combined rejection reasons")

    captured = outcome["captured_variable_count"]
    fresh = outcome["fresh_model_validation_performed"]
    if (
        isinstance(captured, bool)
        or not isinstance(captured, int)
        or captured < 0
        or type(fresh) is not bool
        or outcome["witness_status"] not in _NORMAL_SOLVER_STATUSES
    ):
        _fail("qualification captured/fresh/witness status fields are invalid")
    fresh_audit = outcome["fresh_model_audit"]
    if captured > 0:
        if (
            fresh is not True
            or outcome["witness_status"] not in _VALIDATED_INCUMBENT_STATUSES
            or outcome["validation_status"] not in _VALIDATED_INCUMBENT_STATUSES
            or evidence.witness_feasible is not True
        ):
            _fail("captured incumbent lacks successful fresh validation semantics")
        verified_fresh = _verify_model_audit(
            fresh_audit,
            contract=contract,
            field_name="fresh model audit",
        )
        if (
            witness_audit["model_fingerprint"] is None
            or verified_fresh != witness_audit
        ):
            _fail("fresh model identity/fingerprint differs from witness model")
    elif (
        fresh is not False
        or fresh_audit is not None
        or outcome["validation_status"] != "NOT_RUN"
        or evidence.witness_feasible is not False
    ):
        _fail("no-incumbent outcome has inconsistent fresh-validation semantics")
    if accepted and (not recomputed["accepted"] or captured <= 0):
        _fail("accepted outcome lacks a decision-accepted captured incumbent")
    if outcome["geometry_start_policy"] != plan["geometry_start_policy"]:
        _fail("qualification geometry policy differs from frozen plan")
    return accepted, witness_audit, contract


def _verify_infrastructure_error(
    value: object,
    *,
    action: Mapping[str, Any],
) -> Mapping[str, Any]:
    failure = _mapping(value, field_name="infrastructure error")
    _exact_keys(
        failure,
        {"stage", "action_index", "original_profile_order_index", "type", "message"},
        field_name="infrastructure error",
    )
    allowed_stages = {
        "process_conflict_check",
        "pre_action_revalidation",
        "instance_load",
        "exact_signature_preflight",
        "qualification",
        "post_action_revalidation",
    }
    if (
        not isinstance(failure["stage"], str)
        or failure["stage"] not in allowed_stages
        or failure["action_index"] != action["action_index"]
        or failure["original_profile_order_index"]
        != action["original_profile_order_index"]
        or not isinstance(failure["type"], str)
        or not failure["type"]
        or not isinstance(failure["message"], str)
        or not failure["message"]
    ):
        _fail("infrastructure error stage/action/type/message is invalid")
    return failure


def _verify_checkpoint(
    path: Path,
    *,
    action: Mapping[str, Any],
    plan: Mapping[str, Any],
    run_identity: Mapping[str, Any],
    run_id: str,
    run_sha: str,
    anchor_signature: QualificationConstraintSignature | None,
    allowed_source_profile_ids: set[str],
) -> tuple[
    Mapping[str, Any],
    list[Mapping[str, Any]],
    QualificationConstraintSignature | None,
]:
    checkpoint = _read_canonical(path, field_name="checkpoint")
    expected_keys = {
        "artifact_type",
        "artifact_schema_version",
        "checkpoint_schema",
        *_BOUNDARY_FLAGS,
        "contains_algorithm_comparison_results",
        "identity",
        "result",
        "checkpoint_sha256",
    }
    _exact_keys(checkpoint, expected_keys, field_name="checkpoint")
    _self_hash(checkpoint, hash_field="checkpoint_sha256", field_name="checkpoint")
    _verify_boundary(checkpoint, field_name="checkpoint")
    if (
        checkpoint["artifact_type"] != _CHECKPOINT_ARTIFACT_TYPE
        or checkpoint["artifact_schema_version"] != 1
        or checkpoint["checkpoint_schema"] != _CHECKPOINT_SCHEMA
        or checkpoint["contains_algorithm_comparison_results"] is not False
    ):
        _fail("checkpoint type/schema/formal boundary drifted")
    identity = _mapping(checkpoint["identity"], field_name="checkpoint identity")
    _exact_keys(
        identity,
        {
            "batch_run_id",
            "run_identity_sha256",
            "plan_id",
            "plan_sha256",
            "plan_file_sha256",
            "environment_sha256",
            "git_commit",
            "action",
            "process_audit",
        },
        field_name="checkpoint identity",
    )
    process = _mapping(identity["process_audit"], field_name="process audit")
    _exact_keys(
        process,
        {"process_table_sha256", "conflicting_processes", "valid"},
        field_name="process audit",
    )
    raw_result = _mapping(checkpoint["result"], field_name="checkpoint result")
    raw_failure = raw_result.get("infrastructure_error")
    is_process_failure = (
        raw_result.get("status") == "infrastructure_failure"
        and isinstance(raw_failure, Mapping)
        and raw_failure.get("stage") == "process_conflict_check"
    )
    process_state_valid = (
        process.get("valid") is False
        and isinstance(process.get("conflicting_processes"), list)
        and bool(process["conflicting_processes"])
        if is_process_failure
        else process.get("valid") is True
        and process.get("conflicting_processes") == []
    )
    if (
        identity["batch_run_id"] != run_id
        or identity["run_identity_sha256"] != run_sha
        or identity["plan_id"] != _PLAN_ID
        or identity["plan_sha256"] != plan["plan_sha256"]
        or identity["plan_file_sha256"] != run_identity["plan_file_sha256"]
        or identity["environment_sha256"] != run_identity["environment_sha256"]
        or identity["git_commit"] != run_identity["git_commit"]
        or identity["action"] != action
        or _SHA_RE.fullmatch(str(process.get("process_table_sha256"))) is None
        or not process_state_valid
    ):
        _fail("checkpoint identity/process audit drifted")
    result = raw_result
    result_keys = {
        "status",
        "completed_at_utc",
        "infrastructure_error",
        "signature_preflight",
        "model_audit",
        "constraint_signature",
        "signature_comparison_to_anchor",
        "reuse_case_audit",
        "reuse_pool_audit_snapshot",
        "reuse_candidate_selected",
        "reuse_start_installation_attempted",
        "reuse_start_installed",
        "selected_source",
        "solve_started",
        "fresh_validation_performed",
        "adapter_disposed",
        "runtime_classification",
        "qualification_outcome",
    }
    _exact_keys(result, result_keys, field_name="checkpoint result")
    if not isinstance(result["completed_at_utc"], str) or not result["completed_at_utc"]:
        _fail("checkpoint completion timestamp is invalid")
    if result["status"] == "infrastructure_failure":
        _verify_infrastructure_error(result["infrastructure_error"], action=action)
        expected_failed_values = {
            "signature_preflight": None,
            "model_audit": None,
            "constraint_signature": None,
            "signature_comparison_to_anchor": None,
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
        if any(result[key] != value for key, value in expected_failed_values.items()):
            _fail("infrastructure checkpoint result does not match aborted contract")
        events = _call_smoke_helper(
            smoke_verify._verify_case_audit,
            result["reuse_case_audit"],
            case_id=action["batch_case_id"],
            expected_actions=None,
        )
        snapshot = _call_smoke_helper(
            smoke_verify._verify_pool_audit, result["reuse_pool_audit_snapshot"]
        )
        if any(
            snapshot["audit_events"][event["sequence_index"]] != event
            for event in events
        ):
            _fail("failed case audit is not an exact pool-audit view")
        return result, events, anchor_signature
    if result["status"] != "completed" or result["infrastructure_error"] is not None:
        _fail("checkpoint status/failure pairing drifted")
    outcome = _mapping(
        result["qualification_outcome"], field_name="qualification outcome"
    )
    accepted, outcome_audit, contract = _reassess_outcome(
        outcome,
        action=action,
        plan=plan,
        run_id=run_id,
    )
    classification = _classify(outcome)
    if (
        result["runtime_classification"] != classification
        or not classification["fresh_validation_requirement_satisfied"]
        or result["fresh_validation_performed"]
        != classification["fresh_model_validation_performed"]
        or result["solve_started"] is not True
        or result["adapter_disposed"] is not True
    ):
        _fail("classification/fresh-validation/lifecycle audit drifted")
    signature = _call_smoke_helper(
        smoke_verify._signature,
        result["constraint_signature"],
        field_name="qualification signature",
    )
    expected_events = (
        ("reuse_cache_miss", "incumbent_offered")
        if action["role"] == "anchor_source" and accepted
        else ("reuse_cache_miss",)
        if action["role"] == "anchor_source"
        else (
            "reuse_candidate_selected",
            "reuse_start_installed",
            *(("incumbent_offered",) if accepted else ()),
        )
    )
    events = _call_smoke_helper(
        smoke_verify._verify_case_audit,
        result["reuse_case_audit"],
        case_id=action["batch_case_id"],
        expected_actions=expected_events,
    )
    snapshot = _call_smoke_helper(
        smoke_verify._verify_pool_audit, result["reuse_pool_audit_snapshot"]
    )
    if snapshot["audit_events"][-len(events) :] != events:
        _fail("pool snapshot does not end in this case audit")
    audit = _verify_model_audit(
        result["model_audit"], contract=contract, field_name="result model audit"
    )
    if audit != outcome_audit:
        _fail("result model audit differs from outcome witness model audit")
    if action["role"] == "anchor_source":
        if (
            result["signature_preflight"] is not None
            or result["signature_comparison_to_anchor"] is not None
            or result["reuse_candidate_selected"] is not False
            or result["reuse_start_installation_attempted"] is not False
            or result["reuse_start_installed"] is not False
            or result["selected_source"] is not None
        ):
            _fail("anchor claims target preflight/reuse")
        return result, events, signature
    if anchor_signature is None:
        _fail("target lacks verified anchor signature")
    preflight = _mapping(result["signature_preflight"], field_name="preflight")
    _exact_keys(
        preflight,
        {
            "model_audit",
            "constraint_signature",
            "solve_started",
            "start_installation_attempted",
            "adapter_disposed",
        },
        field_name="preflight",
    )
    preflight_signature = _call_smoke_helper(
        smoke_verify._signature,
        preflight["constraint_signature"],
        field_name="preflight signature",
    )
    expected_comparison = _expected_comparison(anchor_signature, preflight_signature)
    preflight_audit = _verify_model_audit(
        preflight["model_audit"],
        contract=contract,
        field_name="preflight model audit",
    )
    if (
        preflight["solve_started"] is not False
        or preflight["start_installation_attempted"] is not False
        or preflight["adapter_disposed"] is not True
        or signature != preflight_signature
        or preflight_signature != anchor_signature
        or result["signature_comparison_to_anchor"] != expected_comparison
        or result["reuse_candidate_selected"] is not True
        or result["reuse_start_installation_attempted"] is not True
        or result["reuse_start_installed"] is not True
        or result["selected_source"] is None
        or preflight_audit != outcome_audit
    ):
        _fail("target exact-signature preflight/reuse transport gate failed")
    selected = _mapping(result["selected_source"], field_name="selected source")
    selected_event = next(
        event for event in events if event["action"] == "reuse_candidate_selected"
    )
    if (
        selected.get("source_provenance") != selected_event["source_provenance"]
        or selected.get("source_reuse_record_sha256")
        != selected_event["source_reuse_record_sha256"]
    ):
        _fail("selected source provenance/hash drifted")
    provenance = _mapping(
        selected["source_provenance"], field_name="selected source provenance"
    )
    if (
        provenance.get("qualification_run_id") != run_id
        or provenance.get("profile_id") not in allowed_source_profile_ids
        or provenance.get("physical_instance_sha256")
        != H23_REUSE_SMOKE_PHYSICAL_INSTANCE_SHA256
        or float(provenance.get("demand_scale", -1.0)) != 1.0
        or provenance.get("screening_is_exact") is not False
    ):
        _fail("selected source is not a prior accepted demand-1 profile")
    return result, events, anchor_signature


def _ledger_entry(
    action: Mapping[str, Any], result: Mapping[str, Any]
) -> Mapping[str, Any]:
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


@dataclass(frozen=True, slots=True)
class H23Demand1ExactReuseVerificationRequest:
    plan_path: Path
    results_root: Path
    expected_run_id: str
    expected_git_commit: str
    expected_implementation_sha256: str

    def __post_init__(self) -> None:
        if (
            not self.expected_run_id.startswith(_RUN_ID_PREFIX)
            or _SHA_RE.fullmatch(self.expected_run_id.removeprefix(_RUN_ID_PREFIX))
            is None
        ):
            raise ValueError("expected_run_id is invalid")
        if _COMMIT_RE.fullmatch(self.expected_git_commit) is None:
            raise ValueError("expected_git_commit must be a Git digest")
        if _SHA_RE.fullmatch(self.expected_implementation_sha256) is None:
            raise ValueError("expected_implementation_sha256 must be SHA256")


@dataclass(frozen=True, slots=True)
class H23Demand1ExactReuseVerificationReport:
    status: str
    conclusion: str
    summary_path: Path
    plan_sha256: str
    batch_run_id: str
    attempted_action_count: int
    accepted_count: int
    no_incumbent_count: int
    all_open_count: int

    def to_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["summary_path"] = str(self.summary_path)
        return payload


def verify_h23_demand1_exact_reuse_results(
    request: H23Demand1ExactReuseVerificationRequest,
) -> H23Demand1ExactReuseVerificationReport:
    plan_path = request.plan_path.expanduser().resolve(strict=True)
    plan, _smoke_plan, actions = _verify_plan(plan_path)
    _verify_implementation(
        plan,
        expected_commit=request.expected_git_commit,
        expected_sha=request.expected_implementation_sha256,
    )
    root = request.results_root.expanduser().resolve(strict=True)
    entries = {path.name for path in root.iterdir()}
    summaries = entries & {"batch_summary.json", "batch_aborted.json"}
    if len(summaries) != 1 or entries != {
        "run_identity.json",
        "checkpoints",
        *summaries,
    }:
        _fail("results root contains missing/extra artifacts")
    identity, run_sha = _verify_run_identity(
        root / "run_identity.json",
        plan=plan,
        plan_path=plan_path,
        actions=actions,
        expected_run_id=request.expected_run_id,
    )
    run_id = _RUN_ID_PREFIX + run_sha
    if identity["plan_file_sha256"] != _sha256_file(plan_path):
        _fail("run identity plan-file binding drifted")
    summary_path = root / next(iter(summaries))
    summary = _read_canonical(summary_path, field_name="batch summary")
    expected_summary_keys = {
        "artifact_type",
        "artifact_schema_version",
        "summary_schema",
        *_BOUNDARY_FLAGS,
        "contains_algorithm_comparison_results",
        "batch_run_id",
        "run_identity_sha256",
        "plan_id",
        "plan_sha256",
        "order_bias",
        "conclusion",
        "completed",
        "attempted_action_count",
        "completed_action_count",
        "accepted_count",
        "rejected_count",
        "has_incumbent_count",
        "no_incumbent_count",
        "all_closed_count",
        "all_open_count",
        "interior_open_count",
        "fresh_validated_count",
        "infrastructure_failure_count",
        "infrastructure_failure",
        "checkpoints",
        "profile_ledger",
        "reuse_pool_audit",
        "formal_pool_created",
        "locked_profile_ids",
        "speedup_claimed",
        "completed_at_utc",
        "summary_sha256",
    }
    _exact_keys(summary, expected_summary_keys, field_name="batch summary")
    _self_hash(summary, hash_field="summary_sha256", field_name="batch summary")
    _verify_boundary(summary, field_name="batch summary")
    conclusion = summary["conclusion"]
    if (
        summary["artifact_type"] != _SUMMARY_ARTIFACT_TYPE
        or summary["artifact_schema_version"] != 1
        or summary["summary_schema"] != _SUMMARY_SCHEMA
        or summary["contains_algorithm_comparison_results"] is not False
        or summary["batch_run_id"] != run_id
        or summary["run_identity_sha256"] != run_sha
        or summary["plan_id"] != _PLAN_ID
        or summary["plan_sha256"] != plan["plan_sha256"]
        or summary["order_bias"] != _ORDER_BIAS
        or summary["formal_pool_created"] is not False
        or summary["locked_profile_ids"] != []
        or summary["speedup_claimed"] is not False
        or conclusion
        not in {"completed", "source_unavailable", "infrastructure_failed"}
        or summary["completed"] is not (conclusion == "completed")
        or not isinstance(summary["completed_at_utc"], str)
        or not summary["completed_at_utc"]
        or summary_path.name
        != (
            "batch_aborted.json"
            if conclusion == "infrastructure_failed"
            else "batch_summary.json"
        )
    ):
        _fail("summary identity/conclusion/formal boundary drifted")
    records = _list(summary["checkpoints"], field_name="checkpoint ledger")
    if summary["attempted_action_count"] != len(records) or len(records) > 6:
        _fail("attempted-action count drifted")
    checkpoint_dir = root / "checkpoints"
    if len(list(checkpoint_dir.iterdir())) != len(records):
        _fail("checkpoint directory has missing/extra files")
    results: list[Mapping[str, Any]] = []
    all_events: list[Mapping[str, Any]] = []
    anchor_signature: QualificationConstraintSignature | None = None
    accepted_profile_ids: set[str] = set()
    expected_ledger: list[Mapping[str, Any]] = []
    for position, record_raw in enumerate(records):
        record = _mapping(record_raw, field_name=f"checkpoint record {position}")
        _exact_keys(
            record,
            {
                "action_index",
                "original_profile_order_index",
                "path",
                "checkpoint_sha256",
                "file_sha256",
                "status",
            },
            field_name=f"checkpoint record {position}",
        )
        action = actions[position]
        path = Path(record["path"]).resolve(strict=True)
        if (
            path.parent != checkpoint_dir
            or record["action_index"] != position
            or record["original_profile_order_index"] != _ACTION_INDICES[position]
            or _sha256_file(path) != record["file_sha256"]
        ):
            _fail("checkpoint ledger path/order/file hash drifted")
        result, events, anchor_signature = _verify_checkpoint(
            path,
            action=action,
            plan=plan,
            run_identity=identity,
            run_id=run_id,
            run_sha=run_sha,
            anchor_signature=anchor_signature,
            allowed_source_profile_ids=accepted_profile_ids,
        )
        checkpoint_payload = _read_canonical(path, field_name="checkpoint")
        if (
            record["checkpoint_sha256"] != checkpoint_payload["checkpoint_sha256"]
            or record["status"] != result["status"]
        ):
            _fail("checkpoint embedded hash/status drifted")
        results.append(result)
        all_events.extend(events)
        expected_ledger.append(_ledger_entry(action, result))
        classification = result.get("runtime_classification")
        if (
            isinstance(classification, Mapping)
            and classification.get("accepted") is True
        ):
            accepted_profile_ids.add(action["source_job"]["source_job"]["profile_id"])
    if summary["profile_ledger"] != expected_ledger:
        _fail("summary profile ledger does not preserve checkpoint outcomes")
    classifications = [
        result["runtime_classification"]
        for result in results
        if result["runtime_classification"] is not None
    ]
    expected_counts = {
        "completed_action_count": sum(
            result["status"] == "completed" for result in results
        ),
        "accepted_count": sum(row["accepted"] for row in classifications),
        "rejected_count": sum(not row["accepted"] for row in classifications),
        "has_incumbent_count": sum(row["has_incumbent"] for row in classifications),
        "no_incumbent_count": sum(not row["has_incumbent"] for row in classifications),
        "all_closed_count": sum(row["all_closed"] for row in classifications),
        "all_open_count": sum(row["all_open"] for row in classifications),
        "interior_open_count": sum(row["interior_open"] for row in classifications),
        "fresh_validated_count": sum(
            row["fresh_model_validation_performed"] for row in classifications
        ),
    }
    if any(summary[key] != value for key, value in expected_counts.items()):
        _fail("summary classification counts do not recompute")
    final_pool = _call_smoke_helper(
        smoke_verify._verify_pool_audit, summary["reuse_pool_audit"]
    )
    if final_pool["audit_events"] != all_events:
        _fail("final pool audit differs from checkpoint event sequence")
    if results and summary["reuse_pool_audit"] != results[-1]["reuse_pool_audit_snapshot"]:
        _fail("summary pool audit differs from the final checkpoint snapshot")
    if conclusion == "completed" and (
        len(results) != 6 or any(result["status"] != "completed" for result in results)
    ):
        _fail("completed batch does not retain six completed profiles")
    if conclusion == "source_unavailable" and (
        len(results) != 1 or results[0]["runtime_classification"]["accepted"]
    ):
        _fail("source-unavailable conclusion is inconsistent")
    if conclusion == "infrastructure_failed":
        infrastructure_results = [
            result for result in results if result["status"] == "infrastructure_failure"
        ]
        checkpoint_failure = (
            infrastructure_results[0]["infrastructure_error"]
            if len(infrastructure_results) == 1
            else None
        )
        if (
            not results
            or results[-1]["status"] != "infrastructure_failure"
            or len(infrastructure_results) != 1
            or summary["infrastructure_failure_count"] != 1
            or summary["infrastructure_failure"] != checkpoint_failure
            or summary["completed_action_count"] != len(results) - 1
        ):
            _fail("infrastructure conclusion/failure evidence drifted")
    elif (
        summary["infrastructure_failure_count"] != 0
        or summary["infrastructure_failure"] is not None
        or any(result["status"] == "infrastructure_failure" for result in results)
    ):
        _fail("non-infrastructure conclusion reports infrastructure failure")
    return H23Demand1ExactReuseVerificationReport(
        status="verified",
        conclusion=conclusion,
        summary_path=summary_path,
        plan_sha256=plan["plan_sha256"],
        batch_run_id=run_id,
        attempted_action_count=len(results),
        accepted_count=summary["accepted_count"],
        no_incumbent_count=summary["no_incumbent_count"],
        all_open_count=summary["all_open_count"],
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=(
            "python -m src.benchmark.branch_cut.h23_demand1_exact_reuse_batch_verifier"
        )
    )
    parser.add_argument("--plan", required=True)
    parser.add_argument("--expected-git-commit", required=True)
    parser.add_argument("--expected-implementation-sha256", required=True)
    parser.add_argument("--results-root")
    parser.add_argument("--expected-run-id")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if (args.results_root is None) != (args.expected_run_id is None):
            raise ValueError(
                "--results-root and --expected-run-id must be supplied together"
            )
        if args.results_root is None:
            report = verify_h23_demand1_exact_reuse_plan(
                H23Demand1ExactReusePlanVerificationRequest(
                    plan_path=Path(args.plan),
                    expected_git_commit=args.expected_git_commit,
                    expected_implementation_sha256=(
                        args.expected_implementation_sha256
                    ),
                )
            )
        else:
            report = verify_h23_demand1_exact_reuse_results(
                H23Demand1ExactReuseVerificationRequest(
                    plan_path=Path(args.plan),
                    results_root=Path(args.results_root),
                    expected_run_id=args.expected_run_id,
                    expected_git_commit=args.expected_git_commit,
                    expected_implementation_sha256=(
                        args.expected_implementation_sha256
                    ),
                )
            )
    except (
        H23Demand1ExactReuseVerificationError,
        H23IncumbentReuseSmokeError,
        smoke_verify.H23IncumbentReuseSmokeVerificationError,
        OSError,
        TypeError,
        ValueError,
    ) as error:
        print(f"H23 demand-1 exact-reuse verification failed: {error}", file=sys.stderr)
        return 2
    print(canonical_json(report.to_payload()))
    return 0


__all__ = [
    "H23Demand1ExactReusePlanVerificationReport",
    "H23Demand1ExactReusePlanVerificationRequest",
    "H23Demand1ExactReuseVerificationError",
    "H23Demand1ExactReuseVerificationReport",
    "H23Demand1ExactReuseVerificationRequest",
    "build_parser",
    "main",
    "verify_h23_demand1_exact_reuse_plan",
    "verify_h23_demand1_exact_reuse_results",
]


if __name__ == "__main__":
    raise SystemExit(main())
