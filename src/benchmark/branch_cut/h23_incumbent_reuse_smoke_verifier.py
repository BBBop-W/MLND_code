"""Independent solver-free verifier for the H23 incumbent-reuse smoke.

This module does not import the smoke runner or Gurobi.  It recomputes every
artifact hash, exact signature relation, reuse pool/case/record hash, event
sequence, post-hoc ordering disclosure, qualification decision, and formal
ineligibility gate from persisted canonical JSON.
"""

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
from .h23_incumbent_reuse_contract import LEGACY_REUSE_SMOKE_PLAN_FILE_SHA256
from .qualification import (
    QualificationEvidence,
    QualificationPolicy,
    assess_qualification,
)
from .qualification_incumbent_reuse import (
    QUALIFICATION_CONSTRAINT_SIGNATURE_SCHEMA,
    QUALIFICATION_INCUMBENT_REUSE_POLICY_VERSION,
    QualificationConstraintSignature,
)


_PLAN_ID = "h23-incumbent-reuse-smoke-v1"
_PLAN_SCHEMA = "mlnd-h23-incumbent-reuse-smoke-plan-v1"
_RUN_SCHEMA = "mlnd-h23-incumbent-reuse-smoke-run-v1"
_CHECKPOINT_SCHEMA = "mlnd-h23-incumbent-reuse-smoke-checkpoint-v1"
_SUMMARY_SCHEMA = "mlnd-h23-incumbent-reuse-smoke-summary-v1"
_PLAN_ARTIFACT_TYPE = "mlnd_branch_cut_h23_incumbent_reuse_smoke_plan"
_RUN_ARTIFACT_TYPE = "mlnd_branch_cut_h23_incumbent_reuse_smoke_run_identity"
_CHECKPOINT_ARTIFACT_TYPE = "mlnd_branch_cut_h23_incumbent_reuse_smoke_checkpoint"
_SUMMARY_ARTIFACT_TYPE = "mlnd_branch_cut_h23_incumbent_reuse_smoke_summary"
_RUN_ID_PREFIX = "h23-incumbent-reuse-smoke-run-"
_IMPLEMENTATION_FILES = {
    "src/benchmark/branch_cut/h23_incumbent_reuse_contract.py",
    "src/benchmark/branch_cut/qualification_incumbent_reuse.py",
    "src/benchmark/branch_cut/qualification_runner.py",
    "src/benchmark/branch_cut/h23_incumbent_reuse_smoke.py",
    "src/benchmark/branch_cut/h23_incumbent_reuse_smoke_verifier.py",
}
_ACTION_INDICES = (4, 3, 10)
_ACTION_ROLES = (
    "accepted_source_rerun",
    "objective_only_positive_target",
    "constraint_changing_negative_probe",
)
_SOURCE_PLAN_SHA256 = "f80bcea97062a521d081c81ff95ca55ae03741e7e3748b92ffdbfc74edd72d59"
_SOURCE_RUN_ID = (
    "h23-qualification-diagnostic-run-"
    "f94dcee8e4cf8fbfb0a60cfbb3c51df5a5850e026e2a7d8da9624a493227f82a"
)
_SOURCE_SUMMARY_SHA256 = (
    "a42a4ae7b9f615bec1e18bd0631359653beb6adefb33fff836d45e59f70f0856"
)
_INSTANCE_SHA256 = "153680bbdcbef0e31021a8257117cc85aa51a02ef0234420174dcb744a28eaef"
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
    "post_hoc_source_selection": True,
    "execution_order_original_indices": [4, 3, 10],
    "manifest_order_deviation": True,
    "positive_target_precedes_source_in_original_manifest": True,
    "timing_comparison_allowed": False,
    "causal_acceleration_claim_allowed": False,
    "qualification_pool_promotion_allowed": False,
}
_EXECUTION_CONTRACT: Mapping[str, object] = {
    "concurrency": 1,
    "threads": 1,
    "exclusive_results_root": True,
    "shared_resource_lock": True,
    "initial_and_per_action_process_conflict_checks": True,
    "infrastructure_fail_fast": True,
    "fresh_original_model_validation_required_for_accepted_witness": True,
    "negative_probe_signature_only": True,
    "negative_probe_optimize_allowed": False,
    "raw_incumbent_values_persisted": False,
}
_EXPECTED_EVENT_ACTIONS = {
    "accepted_source_rerun": ("reuse_cache_miss", "incumbent_offered"),
    "objective_only_positive_target": (
        "reuse_candidate_selected",
        "reuse_start_installed",
        "incumbent_offered",
    ),
    "constraint_changing_negative_probe": ("reuse_cache_miss",),
}
_SHA_RE = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")


class H23IncumbentReuseSmokeVerificationError(RuntimeError):
    """An artifact or semantic smoke claim failed verification."""


def _fail(message: str) -> None:
    raise H23IncumbentReuseSmokeVerificationError(message)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def _require_mapping(value: object, *, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        _fail(f"{field_name} must be an object with string keys")
    return value


def _require_list(value: object, *, field_name: str) -> list[Any]:
    if not isinstance(value, list):
        _fail(f"{field_name} must be a list")
    return value


def _require_exact_keys(
    value: Mapping[str, Any], expected: set[str], *, field_name: str
) -> None:
    if set(value) != expected:
        _fail(
            f"{field_name} fields differ; "
            f"missing={sorted(expected - set(value))}, "
            f"extra={sorted(set(value) - expected)}"
        )


def _require_sha(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or _SHA_RE.fullmatch(value) is None:
        _fail(f"{field_name} must be a lowercase SHA256")
    return value


def _verify_self_hash(
    payload: Mapping[str, Any], *, hash_field: str, field_name: str
) -> str:
    observed = _require_sha(payload.get(hash_field), field_name=hash_field)
    entry = {key: value for key, value in payload.items() if key != hash_field}
    if canonical_sha256(entry) != observed:
        _fail(f"{field_name} self-hash mismatch")
    return observed


def _verify_boundary_flags(payload: Mapping[str, Any], *, field_name: str) -> None:
    if any(payload.get(key) != value for key, value in _BOUNDARY_FLAGS.items()):
        _fail(f"{field_name} development/formal boundary drifted")


def _signature(payload: object, *, field_name: str) -> QualificationConstraintSignature:
    mapping = _require_mapping(payload, field_name=field_name)
    expected = {
        "signature_schema",
        "constraint_signature_sha256",
        "ordered_var_name_sha256",
        "variable_count",
        "linear_constraint_count",
        "quadratic_constraint_count",
        "sos_constraint_count",
        "unsupported_general_constraint_count",
        "unsupported_nonlinear_constraint_count",
        "extraction_complete",
        "objective_excluded",
    }
    _require_exact_keys(mapping, expected, field_name=field_name)
    try:
        signature = QualificationConstraintSignature(**mapping)
    except (TypeError, ValueError) as error:
        _fail(f"{field_name} is invalid: {error}")
    if signature.signature_schema != QUALIFICATION_CONSTRAINT_SIGNATURE_SCHEMA:
        _fail(f"{field_name} schema drifted")
    return signature


def _recomputed_comparison(
    *,
    role: str,
    expected_relation: str,
    source: QualificationConstraintSignature,
    target: QualificationConstraintSignature,
    source_demand_scale: float,
    target_demand_scale: float,
) -> dict[str, Any]:
    equal = source == target
    observed = "equal" if equal else "different"
    return {
        "comparison_role": role,
        "expected_relation": expected_relation,
        "observed_relation": observed,
        "source_constraint_signature": source.to_payload(),
        "target_constraint_signature": target.to_payload(),
        "source_constraint_signature_sha256": source.constraint_signature_sha256,
        "target_constraint_signature_sha256": target.constraint_signature_sha256,
        "constraint_signature_equal": equal,
        "ordered_var_name_sha256_equal": (
            source.ordered_var_name_sha256 == target.ordered_var_name_sha256
        ),
        "variable_count_equal": source.variable_count == target.variable_count,
        "source_demand_scale": float(source_demand_scale),
        "target_demand_scale": float(target_demand_scale),
        "demand_scale_equal": float(source_demand_scale) == float(target_demand_scale),
        "objective_excluded_both": (
            source.objective_excluded and target.objective_excluded
        ),
        "extraction_complete_both": (
            source.extraction_complete and target.extraction_complete
        ),
        "unsupported_counts_zero_both": all(
            value == 0
            for value in (
                source.unsupported_general_constraint_count,
                source.unsupported_nonlinear_constraint_count,
                target.unsupported_general_constraint_count,
                target.unsupported_nonlinear_constraint_count,
            )
        ),
        "comparison_passed": observed == expected_relation,
    }


def _verify_case_audit(
    payload: object,
    *,
    case_id: str,
    expected_actions: Sequence[str] | None,
) -> list[Mapping[str, Any]]:
    audit = _require_mapping(payload, field_name=f"case audit {case_id}")
    expected_keys = {
        "reuse_policy_version",
        "development_only",
        "qualification_only",
        "eligible_for_formal_pool",
        "eligible_for_algorithm_comparison",
        "case_id",
        "audit_events",
        "case_reuse_audit_sha256",
    }
    _require_exact_keys(audit, expected_keys, field_name=f"case audit {case_id}")
    if (
        audit["reuse_policy_version"] != QUALIFICATION_INCUMBENT_REUSE_POLICY_VERSION
        or audit["development_only"] is not True
        or audit["qualification_only"] is not True
        or audit["eligible_for_formal_pool"] is not False
        or audit["eligible_for_algorithm_comparison"] is not False
        or audit["case_id"] != case_id
    ):
        _fail(f"case audit {case_id} boundary or identity drifted")
    entry = {
        key: value for key, value in audit.items() if key != "case_reuse_audit_sha256"
    }
    if canonical_sha256(entry) != audit["case_reuse_audit_sha256"]:
        _fail(f"case audit {case_id} self-hash mismatch")
    events = _require_list(audit["audit_events"], field_name="case audit events")
    actions = tuple(event.get("action") for event in events if isinstance(event, dict))
    if len(actions) != len(events):
        _fail(f"case audit {case_id} contains invalid events")
    if expected_actions is not None and actions != tuple(expected_actions):
        _fail(f"case audit {case_id} event sequence drifted")
    for event in events:
        event_mapping = _require_mapping(event, field_name="reuse audit event")
        _require_exact_keys(
            event_mapping,
            {
                "sequence_index",
                "action",
                "reason",
                "target_provenance",
                "target_constraint_signature",
                "source_provenance",
                "source_reuse_record_sha256",
            },
            field_name="reuse audit event",
        )
        target = _require_mapping(
            event_mapping["target_provenance"], field_name="target provenance"
        )
        if (
            target.get("case_id") != case_id
            or target.get("development_only") is not True
            or target.get("eligible_for_formal_pool") is not False
            or target.get("eligible_for_algorithm_comparison") is not False
            or target.get("screening_is_exact") is not False
        ):
            _fail("reuse target provenance boundary drifted")
        _signature(
            event_mapping["target_constraint_signature"],
            field_name="event target signature",
        )
        source = event_mapping["source_provenance"]
        source_sha = event_mapping["source_reuse_record_sha256"]
        if (source is None) != (source_sha is None):
            _fail("reuse source provenance/hash pairing drifted")
        if source is not None:
            source_mapping = _require_mapping(source, field_name="source provenance")
            if (
                source_mapping.get("development_only") is not True
                or source_mapping.get("eligible_for_formal_pool") is not False
                or source_mapping.get("eligible_for_algorithm_comparison") is not False
                or source_mapping.get("screening_is_exact") is not False
            ):
                _fail("reuse source provenance boundary drifted")
            _require_sha(source_sha, field_name="source_reuse_record_sha256")
    return events


def _verify_reuse_record(entry: Mapping[str, Any]) -> str:
    _require_exact_keys(
        entry,
        {
            "reuse_policy_version",
            "development_only",
            "eligible_for_formal_pool",
            "eligible_for_algorithm_comparison",
            "source_provenance",
            "constraint_signature",
            "source_accepted",
            "source_fresh_validation_performed",
            "incumbent_metadata",
            "reuse_record_sha256",
        },
        field_name="reuse pool entry",
    )
    if (
        entry["reuse_policy_version"] != QUALIFICATION_INCUMBENT_REUSE_POLICY_VERSION
        or entry["development_only"] is not True
        or entry["eligible_for_formal_pool"] is not False
        or entry["eligible_for_algorithm_comparison"] is not False
        or entry["source_accepted"] is not True
        or entry["source_fresh_validation_performed"] is not True
    ):
        _fail("reuse record boundary or source validation drifted")
    provenance = _require_mapping(
        entry["source_provenance"], field_name="reuse record provenance"
    )
    if (
        provenance.get("development_only") is not True
        or provenance.get("eligible_for_formal_pool") is not False
        or provenance.get("eligible_for_algorithm_comparison") is not False
        or provenance.get("screening_is_exact") is not False
    ):
        _fail("reuse record provenance is formal-eligible or exact")
    signature = _signature(entry["constraint_signature"], field_name="reuse signature")
    incumbent = _require_mapping(
        entry["incumbent_metadata"], field_name="incumbent metadata"
    )
    if (
        incumbent.get("values_persisted_to_audit") is not False
        or incumbent.get("variable_count") != signature.variable_count
        or incumbent.get("ordered_var_name_sha256") != signature.ordered_var_name_sha256
    ):
        _fail("reuse incumbent metadata drifted or persists values")
    observed = _require_sha(
        entry["reuse_record_sha256"], field_name="reuse_record_sha256"
    )
    preimage = {
        key: value for key, value in entry.items() if key != "reuse_record_sha256"
    }
    if canonical_sha256(preimage) != observed:
        _fail("reuse record SHA does not recompute")
    return observed


def _verify_pool_audit(payload: object) -> Mapping[str, Any]:
    audit = _require_mapping(payload, field_name="reuse pool audit")
    _require_exact_keys(
        audit,
        {
            "reuse_policy_version",
            "development_only",
            "qualification_only",
            "eligible_for_formal_pool",
            "eligible_for_algorithm_comparison",
            "adds_forced_opening_constraint",
            "changes_mathematical_model",
            "changes_screening",
            "concurrency",
            "entry_count",
            "entries",
            "audit_events",
            "reuse_audit_sha256",
        },
        field_name="reuse pool audit",
    )
    fixed = {
        "reuse_policy_version": QUALIFICATION_INCUMBENT_REUSE_POLICY_VERSION,
        "development_only": True,
        "qualification_only": True,
        "eligible_for_formal_pool": False,
        "eligible_for_algorithm_comparison": False,
        "adds_forced_opening_constraint": False,
        "changes_mathematical_model": False,
        "changes_screening": False,
        "concurrency": 1,
    }
    if any(audit.get(key) != value for key, value in fixed.items()):
        _fail("reuse pool formal/model boundary drifted")
    entries = _require_list(audit["entries"], field_name="reuse pool entries")
    if audit["entry_count"] != len(entries):
        _fail("reuse pool entry count drifted")
    for entry in entries:
        _verify_reuse_record(_require_mapping(entry, field_name="reuse pool entry"))
    events = _require_list(audit["audit_events"], field_name="reuse pool events")
    if [event.get("sequence_index") for event in events] != list(range(len(events))):
        _fail("reuse pool event sequence indices drifted")
    observed = _require_sha(
        audit["reuse_audit_sha256"], field_name="reuse_audit_sha256"
    )
    preimage = {
        key: value for key, value in audit.items() if key != "reuse_audit_sha256"
    }
    if canonical_sha256(preimage) != observed:
        _fail("reuse pool audit SHA does not recompute")
    return audit


def _verify_qualification_outcome(
    payload: object,
    *,
    case_id: str,
    run_id: str,
    policy_payload: Mapping[str, Any],
) -> bool:
    outcome = _require_mapping(payload, field_name="qualification outcome")
    evidence_payload = _require_mapping(
        outcome.get("evidence"), field_name="qualification evidence"
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
        policy_values = dict(policy_payload)
        target = policy_values.get("large_scale_open_h_target")
        policy_values["large_scale_open_h_target"] = tuple(target) if target else None
        policy = QualificationPolicy(**policy_values)
        recomputed = assess_qualification(evidence, policy).to_payload()
    except (TypeError, ValueError) as error:
        _fail(f"qualification evidence cannot be reassessed: {error}")
    if canonical_json(recomputed) != canonical_json(outcome.get("decision")):
        _fail("qualification decision does not independently recompute")
    if (
        evidence.case_id != case_id
        or evidence.qualification_run_id != run_id
        or evidence.instance_sha256 != _INSTANCE_SHA256
        or evidence.screening_is_exact is not False
        or outcome.get("accepted") != recomputed["accepted"]
    ):
        _fail("qualification identity or fresh-validation gate drifted")
    if recomputed["accepted"] and (
        outcome.get("fresh_model_validation_performed") is not True
        or not evidence.witness_feasible
        or outcome.get("captured_variable_count", 0) <= 0
        or outcome.get("fresh_model_audit") is None
    ):
        _fail("accepted qualification lacks a fresh complete witness")
    return bool(recomputed["accepted"])


@dataclass(frozen=True, slots=True)
class H23IncumbentReuseSmokeVerificationRequest:
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
            raise ValueError("expected_run_id is not an incumbent-reuse smoke run")
        if _COMMIT_RE.fullmatch(self.expected_git_commit) is None:
            raise ValueError("expected_git_commit must be a lowercase Git digest")
        if _SHA_RE.fullmatch(self.expected_implementation_sha256) is None:
            raise ValueError("expected_implementation_sha256 must be SHA256")


@dataclass(frozen=True, slots=True)
class H23IncumbentReuseSmokeVerificationReport:
    status: str
    conclusion: str
    summary_path: Path
    plan_sha256: str
    smoke_run_id: str
    git_commit: str
    implementation_sha256: str
    attempted_action_count: int
    positive_signature_equal: bool | None
    negative_signature_different: bool | None

    def to_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["summary_path"] = str(self.summary_path)
        return payload


@dataclass(frozen=True, slots=True)
class H23IncumbentReuseSmokePlanVerificationRequest:
    plan_path: Path
    expected_git_commit: str
    expected_implementation_sha256: str

    def __post_init__(self) -> None:
        if _COMMIT_RE.fullmatch(self.expected_git_commit) is None:
            raise ValueError("expected_git_commit must be a lowercase Git digest")
        if _SHA_RE.fullmatch(self.expected_implementation_sha256) is None:
            raise ValueError("expected_implementation_sha256 must be SHA256")


@dataclass(frozen=True, slots=True)
class H23IncumbentReuseSmokePlanVerificationReport:
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


def _verify_plan(path: Path) -> Mapping[str, Any]:
    plan = _read_canonical(path.resolve(strict=True), field_name="smoke plan")
    expected_keys = {
        "artifact_type",
        "artifact_schema_version",
        "plan_schema",
        "plan_id",
        *_BOUNDARY_FLAGS,
        "mode",
        "action_count",
        "reuse_policy_version",
        "order_bias",
        "execution_contract",
        "qualification_policy",
        "native_settings",
        "geometry_start_policy",
        "source_h23",
        "required_execution_environment",
        "actions",
        "plan_sha256",
    }
    _require_exact_keys(plan, expected_keys, field_name="smoke plan")
    _verify_self_hash(plan, hash_field="plan_sha256", field_name="smoke plan")
    _verify_boundary_flags(plan, field_name="smoke plan")
    if (
        plan["artifact_type"] != _PLAN_ARTIFACT_TYPE
        or plan["artifact_schema_version"] != 1
        or plan["plan_schema"] != _PLAN_SCHEMA
        or plan["plan_id"] != _PLAN_ID
        or plan["mode"] != "MX"
        or plan["action_count"] != 3
        or plan["reuse_policy_version"] != QUALIFICATION_INCUMBENT_REUSE_POLICY_VERSION
        or plan["order_bias"] != _ORDER_BIAS
        or plan["execution_contract"] != _EXECUTION_CONTRACT
    ):
        _fail("smoke plan identity, order bias, or execution contract drifted")
    source = _require_mapping(plan["source_h23"], field_name="source_h23")
    _require_exact_keys(
        source,
        {
            "plan_path",
            "plan_sha256",
            "plan_file_sha256",
            "summary_path",
            "summary_sha256",
            "summary_file_sha256",
            "diagnostic_run_id",
        },
        field_name="source_h23",
    )
    if (
        source["plan_sha256"] != _SOURCE_PLAN_SHA256
        or source["summary_sha256"] != _SOURCE_SUMMARY_SHA256
        or source["diagnostic_run_id"] != _SOURCE_RUN_ID
        or _sha256_file(Path(source["plan_path"])) != source["plan_file_sha256"]
        or _sha256_file(Path(source["summary_path"])) != source["summary_file_sha256"]
    ):
        _fail("source H23 artifacts changed or are not the frozen evidence")
    actions = _require_list(plan["actions"], field_name="smoke actions")
    if len(actions) != 3:
        _fail("smoke plan must contain exactly three actions")
    observed = tuple(
        (
            action.get("original_profile_order_index"),
            action.get("role"),
            action.get("action_index"),
        )
        for action in actions
        if isinstance(action, dict)
    )
    if observed != tuple(
        (index, role, action_index)
        for action_index, (index, role) in enumerate(
            zip(_ACTION_INDICES, _ACTION_ROLES, strict=True)
        )
    ):
        _fail("smoke action order or roles drifted")
    for action in actions:
        source_job = action["source_job"]["source_job"]
        contract = action["source_job"]["qualification_model_contract"]
        if (
            source_job["instance_sha256"] != _INSTANCE_SHA256
            or source_job["profile_order_index"]
            != action["original_profile_order_index"]
            or contract["screening_is_exact"] is not False
            or _sha256_file(Path(action["historical_checkpoint"]["path"]))
            != action["historical_checkpoint"]["file_sha256"]
        ):
            _fail("smoke action input binding drifted")
    return plan


def _verify_required_implementation(
    plan: Mapping[str, Any],
    *,
    expected_git_commit: str,
    expected_implementation_sha256: str,
) -> Mapping[str, Any]:
    required = _require_mapping(
        plan["required_execution_environment"], field_name="required environment"
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
    implementation = _require_mapping(
        required["implementation"], field_name="required implementation"
    )
    _require_exact_keys(
        implementation,
        {"ordered_file_sha256", "implementation_sha256"},
        field_name="required implementation",
    )
    implementation_files = _require_mapping(
        implementation["ordered_file_sha256"],
        field_name="required implementation files",
    )
    repository_root = Path(str(required["repository_root"])).resolve(strict=True)
    legacy_files = _IMPLEMENTATION_FILES - {
        "src/benchmark/branch_cut/h23_incumbent_reuse_contract.py"
    }
    pinned_legacy_source = (
        set(implementation_files) == legacy_files
        and hashlib.sha256((canonical_json(plan) + "\n").encode()).hexdigest()
        == LEGACY_REUSE_SMOKE_PLAN_FILE_SHA256
    )
    if set(implementation_files) != _IMPLEMENTATION_FILES and not pinned_legacy_source:
        _fail("required implementation file set drifted")
    observed_files = {
        relative: _sha256_file(repository_root / relative)
        for relative in implementation_files
    }
    if (
        required.get("git_commit") != expected_git_commit
        or required.get("clean_status_includes_untracked") is not True
        or _git(repository_root, "rev-parse", "HEAD") != expected_git_commit
        or _git(repository_root, "status", "--porcelain", "--untracked-files=all") != ""
        or observed_files != implementation_files
        or canonical_sha256(observed_files) != implementation["implementation_sha256"]
        or implementation["implementation_sha256"] != expected_implementation_sha256
    ):
        _fail("plan does not bind the current expected commit/implementation")
    return implementation


def verify_h23_incumbent_reuse_smoke_plan(
    request: H23IncumbentReuseSmokePlanVerificationRequest,
) -> H23IncumbentReuseSmokePlanVerificationReport:
    """Verify a frozen plan and current implementation without Gurobi."""

    if not isinstance(request, H23IncumbentReuseSmokePlanVerificationRequest):
        raise TypeError("request must be H23IncumbentReuseSmokePlanVerificationRequest")
    plan_path = request.plan_path.expanduser().resolve(strict=True)
    plan = _verify_plan(plan_path)
    implementation = _verify_required_implementation(
        plan,
        expected_git_commit=request.expected_git_commit,
        expected_implementation_sha256=request.expected_implementation_sha256,
    )
    return H23IncumbentReuseSmokePlanVerificationReport(
        status="verified_plan",
        plan_path=plan_path,
        plan_id=_PLAN_ID,
        plan_sha256=plan["plan_sha256"],
        plan_file_sha256=_sha256_file(plan_path),
        git_commit=request.expected_git_commit,
        implementation_sha256=implementation["implementation_sha256"],
        execution_order_original_indices=_ACTION_INDICES,
    )


def _verify_run_identity(
    path: Path,
    *,
    plan: Mapping[str, Any],
    request: H23IncumbentReuseSmokeVerificationRequest,
) -> tuple[Mapping[str, Any], str]:
    payload = _read_canonical(path, field_name="run identity")
    _require_exact_keys(
        payload,
        {
            "artifact_type",
            "artifact_schema_version",
            "smoke_run_id",
            "run_identity",
            "run_identity_sha256",
        },
        field_name="run identity",
    )
    identity = _require_mapping(payload["run_identity"], field_name="run identity body")
    run_sha = _require_sha(
        payload["run_identity_sha256"], field_name="run_identity_sha256"
    )
    if canonical_sha256(identity) != run_sha:
        _fail("run identity SHA does not recompute")
    run_id = _RUN_ID_PREFIX + run_sha
    if (
        payload["artifact_type"] != _RUN_ARTIFACT_TYPE
        or payload["artifact_schema_version"] != 1
        or payload["smoke_run_id"] != run_id
        or run_id != request.expected_run_id
    ):
        _fail("run identity or expected run ID drifted")
    _verify_boundary_flags(identity, field_name="run identity")
    if (
        identity.get("run_schema") != _RUN_SCHEMA
        or identity.get("plan_id") != _PLAN_ID
        or identity.get("plan_sha256") != plan["plan_sha256"]
        or identity.get("plan_file_sha256") != _sha256_file(request.plan_path)
        or identity.get("order_bias") != _ORDER_BIAS
        or identity.get("execution_contract") != _EXECUTION_CONTRACT
        or identity.get("actions") != plan["actions"]
        or identity.get("git_commit") != request.expected_git_commit
        or identity.get("implementation_sha256")
        != request.expected_implementation_sha256
        or identity.get("contains_algorithm_comparison_results") is not False
    ):
        _fail("run identity plan/code/order/formal binding drifted")
    if canonical_sha256(identity["environment"]) != identity["environment_sha256"]:
        _fail("run environment SHA does not recompute")
    process = _require_mapping(
        identity["initial_process_audit"], field_name="initial process audit"
    )
    if process.get("valid") is not True or process.get("conflicting_processes") != []:
        _fail("run began with a process conflict")
    return identity, run_sha


def _verify_checkpoint(
    path: Path,
    *,
    action: Mapping[str, Any],
    plan: Mapping[str, Any],
    run_id: str,
    run_sha: str,
    policy: Mapping[str, Any],
    expected_actions: Sequence[str] | None,
) -> tuple[Mapping[str, Any], list[Mapping[str, Any]]]:
    payload = _read_canonical(path, field_name=f"checkpoint {action['action_index']}")
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
    _require_exact_keys(payload, expected_keys, field_name="smoke checkpoint")
    _verify_self_hash(payload, hash_field="checkpoint_sha256", field_name="checkpoint")
    _verify_boundary_flags(payload, field_name="checkpoint")
    if (
        payload["artifact_type"] != _CHECKPOINT_ARTIFACT_TYPE
        or payload["artifact_schema_version"] != 1
        or payload["checkpoint_schema"] != _CHECKPOINT_SCHEMA
        or payload["contains_algorithm_comparison_results"] is not False
    ):
        _fail("checkpoint type/schema/formal boundary drifted")
    identity = _require_mapping(payload["identity"], field_name="checkpoint identity")
    _require_exact_keys(
        identity,
        {
            "smoke_run_id",
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
    if (
        identity["smoke_run_id"] != run_id
        or identity["run_identity_sha256"] != run_sha
        or identity["plan_id"] != _PLAN_ID
        or identity["plan_sha256"] != plan["plan_sha256"]
        or identity["action"] != action
        or identity["process_audit"].get("valid") is not True
    ):
        _fail("checkpoint identity or process audit drifted")
    result = _require_mapping(payload["result"], field_name="checkpoint result")
    _require_exact_keys(
        result,
        {
            "status",
            "completed_at_utc",
            "infrastructure_error",
            "model_audit",
            "constraint_signature",
            "signature_comparison",
            "reuse_case_audit",
            "reuse_pool_audit_snapshot",
            "reuse_candidate_selected",
            "reuse_start_installation_attempted",
            "reuse_start_installed",
            "selected_source",
            "solve_started",
            "fresh_validation_performed",
            "adapter_disposed",
            "qualification_outcome",
        },
        field_name="checkpoint result",
    )
    events = _verify_case_audit(
        result["reuse_case_audit"],
        case_id=action["smoke_case_id"],
        expected_actions=expected_actions if result["status"] == "completed" else None,
    )
    snapshot = _verify_pool_audit(result["reuse_pool_audit_snapshot"])
    # A checkpoint pool snapshot must end at the same global event recorded by
    # its case audit; filtering by case preserves original sequence indices.
    if events and snapshot["audit_events"][events[-1]["sequence_index"]] != events[-1]:
        _fail("case audit is not an exact view of its pool snapshot")
    if result["status"] == "infrastructure_failure":
        if result["infrastructure_error"] is None:
            _fail("failed checkpoint lacks infrastructure error")
        return result, events
    if result["status"] != "completed" or result["infrastructure_error"] is not None:
        _fail("completed checkpoint status/error drifted")
    signature = _signature(
        result["constraint_signature"], field_name="checkpoint signature"
    )
    model_audit = _require_mapping(
        result["model_audit"], field_name="checkpoint model audit"
    )
    contract = action["source_job"]["qualification_model_contract"]
    if (
        model_audit.get("effective_model_config_sha256")
        != contract["effective_model_config_sha256"]
        or model_audit.get("demand_scale") != contract["demand_scale"]
        or model_audit.get("locker_fixed_cost_per_hour")
        != contract["locker_fixed_cost_per_hour"]
        or model_audit.get("display_origin_choices")
        != contract["display_origin_choices"]
        or model_audit.get("general_corridor_choices")
        != contract["general_corridor_choices"]
        or model_audit.get("screening_is_exact") is not False
    ):
        _fail("checkpoint model audit differs from frozen contract")
    if any(
        _signature(event["target_constraint_signature"], field_name="event signature")
        != signature
        for event in events
    ):
        _fail("checkpoint and reuse-event signatures differ")
    role = action["role"]
    if role == "constraint_changing_negative_probe":
        if (
            result["qualification_outcome"] is not None
            or result["reuse_candidate_selected"] is not False
            or result["reuse_start_installation_attempted"] is not False
            or result["reuse_start_installed"] is not False
            or result["selected_source"] is not None
            or result["solve_started"] is not False
            or result["fresh_validation_performed"] is not False
            or result["adapter_disposed"] is not True
        ):
            _fail("negative probe solved, installed, selected, or leaked its adapter")
    else:
        _verify_qualification_outcome(
            result["qualification_outcome"],
            case_id=action["smoke_case_id"],
            run_id=run_id,
            policy_payload=policy,
        )
        if (
            result["solve_started"] is not True
            or result["adapter_disposed"] is not True
            or result["fresh_validation_performed"]
            is not result["qualification_outcome"]["fresh_model_validation_performed"]
        ):
            _fail("qualification solve/lifecycle/fresh-validation flags drifted")
        if role == "accepted_source_rerun" and (
            result["reuse_candidate_selected"] is not False
            or result["reuse_start_installation_attempted"] is not False
            or result["reuse_start_installed"] is not False
            or result["selected_source"] is not None
        ):
            _fail("source unexpectedly selected or installed a reused start")
        if role == "objective_only_positive_target" and (
            result["reuse_candidate_selected"] is not True
            or result["reuse_start_installation_attempted"] is not True
            or result["reuse_start_installed"] is not True
            or result["selected_source"] is None
        ):
            _fail("positive target lacks selected and installed reused start")
    return result, events


def verify_h23_incumbent_reuse_smoke_results(
    request: H23IncumbentReuseSmokeVerificationRequest,
) -> H23IncumbentReuseSmokeVerificationReport:
    """Verify one immutable smoke package without importing Gurobi or runner."""

    if not isinstance(request, H23IncumbentReuseSmokeVerificationRequest):
        raise TypeError("request must be H23IncumbentReuseSmokeVerificationRequest")
    plan_path = request.plan_path.expanduser().resolve(strict=True)
    plan = _verify_plan(plan_path)
    _verify_required_implementation(
        plan,
        expected_git_commit=request.expected_git_commit,
        expected_implementation_sha256=request.expected_implementation_sha256,
    )
    results_root = request.results_root.expanduser().resolve(strict=True)
    entries = {path.name for path in results_root.iterdir()}
    summaries = entries & {"batch_summary.json", "batch_aborted.json"}
    if len(summaries) != 1 or entries != {
        "run_identity.json",
        "checkpoints",
        *summaries,
    }:
        _fail("results root contains missing, extra, or ambiguous artifacts")
    run_identity, run_sha = _verify_run_identity(
        results_root / "run_identity.json", plan=plan, request=request
    )
    run_id = _RUN_ID_PREFIX + run_sha
    summary_path = results_root / next(iter(summaries))
    summary = _read_canonical(summary_path, field_name="smoke summary")
    expected_summary_keys = {
        "artifact_type",
        "artifact_schema_version",
        "summary_schema",
        *_BOUNDARY_FLAGS,
        "contains_algorithm_comparison_results",
        "smoke_run_id",
        "run_identity_sha256",
        "plan_id",
        "plan_sha256",
        "order_bias",
        "conclusion",
        "passed",
        "attempted_action_count",
        "completed_action_count",
        "infrastructure_failure_count",
        "infrastructure_failure",
        "checkpoints",
        "positive_signature_comparison",
        "negative_signature_comparison",
        "reuse_pool_audit",
        "formal_pool_created",
        "locked_profile_ids",
        "completed_at_utc",
        "summary_sha256",
    }
    _require_exact_keys(summary, expected_summary_keys, field_name="smoke summary")
    _verify_self_hash(summary, hash_field="summary_sha256", field_name="summary")
    _verify_boundary_flags(summary, field_name="summary")
    if (
        summary["artifact_type"] != _SUMMARY_ARTIFACT_TYPE
        or summary["artifact_schema_version"] != 1
        or summary["summary_schema"] != _SUMMARY_SCHEMA
        or summary["contains_algorithm_comparison_results"] is not False
        or summary["smoke_run_id"] != run_id
        or summary["run_identity_sha256"] != run_sha
        or summary["plan_id"] != _PLAN_ID
        or summary["plan_sha256"] != plan["plan_sha256"]
        or summary["order_bias"] != _ORDER_BIAS
        or summary["formal_pool_created"] is not False
        or summary["locked_profile_ids"] != []
    ):
        _fail("summary identity/order/formal boundary drifted")
    records = _require_list(summary["checkpoints"], field_name="summary checkpoints")
    if summary["attempted_action_count"] != len(records):
        _fail("summary attempted-action count drifted")
    checkpoint_results: list[Mapping[str, Any]] = []
    case_events: list[list[Mapping[str, Any]]] = []
    for attempt_index, record in enumerate(records):
        record_mapping = _require_mapping(
            record, field_name=f"checkpoint record {attempt_index}"
        )
        _require_exact_keys(
            record_mapping,
            {
                "action_index",
                "role",
                "path",
                "checkpoint_sha256",
                "file_sha256",
                "status",
            },
            field_name=f"checkpoint record {attempt_index}",
        )
        action = plan["actions"][attempt_index]
        checkpoint_path = Path(record_mapping["path"]).resolve(strict=True)
        if (
            record_mapping["action_index"] != action["action_index"]
            or record_mapping["role"] != action["role"]
            or _sha256_file(checkpoint_path) != record_mapping["file_sha256"]
        ):
            _fail("checkpoint ledger identity/file hash drifted")
        expected_actions: Sequence[str] | None = _EXPECTED_EVENT_ACTIONS[action["role"]]
        if summary["conclusion"] == "source_unavailable":
            expected_actions = ("reuse_cache_miss",)
        elif (
            summary["conclusion"] == "transport_passed_end_to_end_failed"
            and action["role"] == "objective_only_positive_target"
        ):
            expected_actions = (
                "reuse_candidate_selected",
                "reuse_start_installed",
            )
        result, events = _verify_checkpoint(
            checkpoint_path,
            action=action,
            plan=plan,
            run_id=run_id,
            run_sha=run_sha,
            policy=plan["qualification_policy"],
            expected_actions=expected_actions,
        )
        if (
            record_mapping["checkpoint_sha256"]
            != _read_canonical(checkpoint_path, field_name="checkpoint")[
                "checkpoint_sha256"
            ]
            or record_mapping["status"] != result["status"]
        ):
            _fail("checkpoint ledger embedded hash/status drifted")
        checkpoint_results.append(result)
        case_events.append(events)
    if summary["completed_action_count"] != sum(
        result["status"] == "completed" for result in checkpoint_results
    ):
        _fail("summary completed-action count drifted")
    final_pool = _verify_pool_audit(summary["reuse_pool_audit"])
    observed_events = [event for events in case_events for event in events]
    if final_pool["audit_events"] != observed_events:
        _fail("summary pool events differ from checkpoint case audits")
    conclusion = summary["conclusion"]
    allowed = {
        "passed",
        "source_unavailable",
        "transport_passed_end_to_end_failed",
        "infrastructure_failed",
    }
    if conclusion not in allowed or summary["passed"] is not (conclusion == "passed"):
        _fail("summary conclusion/pass flag drifted")
    positive_equal: bool | None = None
    negative_different: bool | None = None
    if conclusion in {"passed", "transport_passed_end_to_end_failed"}:
        if len(checkpoint_results) != 3:
            _fail("completed end-to-end smoke lacks three checkpoints")
        source_signature = _signature(
            checkpoint_results[0]["constraint_signature"], field_name="source signature"
        )
        positive_signature = _signature(
            checkpoint_results[1]["constraint_signature"],
            field_name="positive signature",
        )
        negative_signature = _signature(
            checkpoint_results[2]["constraint_signature"],
            field_name="negative signature",
        )
        positive_expected = _recomputed_comparison(
            role="objective_only_positive_target",
            expected_relation="equal",
            source=source_signature,
            target=positive_signature,
            source_demand_scale=1.0,
            target_demand_scale=1.0,
        )
        negative_expected = _recomputed_comparison(
            role="constraint_changing_negative_probe",
            expected_relation="different",
            source=source_signature,
            target=negative_signature,
            source_demand_scale=1.0,
            target_demand_scale=2.0,
        )
        if (
            checkpoint_results[1]["signature_comparison"] != positive_expected
            or summary["positive_signature_comparison"] != positive_expected
            or checkpoint_results[2]["signature_comparison"] != negative_expected
            or summary["negative_signature_comparison"] != negative_expected
        ):
            _fail("source/target signature comparisons do not recompute")
        positive_equal = bool(positive_expected["constraint_signature_equal"])
        negative_different = not bool(negative_expected["constraint_signature_equal"])
        if not positive_equal or not negative_different:
            _fail("positive/negative exact signature relations failed")
        selected = _require_mapping(
            checkpoint_results[1]["selected_source"], field_name="selected source"
        )
        source_offer = next(
            event for event in case_events[0] if event["action"] == "incumbent_offered"
        )
        if (
            selected["source_provenance"] != source_offer["source_provenance"]
            or selected["source_reuse_record_sha256"]
            != source_offer["source_reuse_record_sha256"]
        ):
            _fail("positive selected-source provenance/hash drifted")
        source_snapshot = checkpoint_results[0]["reuse_pool_audit_snapshot"]
        source_entries = source_snapshot["entries"]
        if len(source_entries) != 1:
            _fail("source checkpoint does not retain one reusable record")
        source_record_sha = _verify_reuse_record(source_entries[0])
        if source_record_sha != selected["source_reuse_record_sha256"]:
            _fail("selected source record SHA cannot be recomputed")
        if (
            checkpoint_results[2]["solve_started"] is not False
            or checkpoint_results[2]["adapter_disposed"] is not True
        ):
            _fail("negative probe optimize/disposal gate failed")
    elif conclusion == "source_unavailable":
        if len(checkpoint_results) != 1:
            _fail("source-unavailable smoke attempted later actions")
        if (
            summary["positive_signature_comparison"] is not None
            or summary["negative_signature_comparison"] is not None
        ):
            _fail("source-unavailable summary claims target comparisons")
    else:
        if summary["infrastructure_failure_count"] != 1:
            _fail("infrastructure-failed summary lacks one failure")
    if conclusion != "infrastructure_failed" and (
        summary["infrastructure_failure_count"] != 0
        or summary["infrastructure_failure"] is not None
    ):
        _fail("non-infrastructure conclusion reports infrastructure failure")
    # Recompute that the final target source remains development-only.  This
    # catches accidental promotion even when every outer artifact flag is safe.
    for entry in final_pool["entries"]:
        provenance = entry["source_provenance"]
        if (
            provenance["eligible_for_formal_pool"] is not False
            or provenance["eligible_for_algorithm_comparison"] is not False
        ):
            _fail("final pool contains formal-eligible provenance")
    return H23IncumbentReuseSmokeVerificationReport(
        status="verified",
        conclusion=conclusion,
        summary_path=summary_path,
        plan_sha256=plan["plan_sha256"],
        smoke_run_id=run_id,
        git_commit=request.expected_git_commit,
        implementation_sha256=request.expected_implementation_sha256,
        attempted_action_count=len(records),
        positive_signature_equal=positive_equal,
        negative_signature_different=negative_different,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=("python -m src.benchmark.branch_cut.h23_incumbent_reuse_smoke_verifier"),
        description="Verify an H23 incumbent-reuse smoke without Gurobi.",
    )
    parser.add_argument("--plan", required=True)
    parser.add_argument("--results-root")
    parser.add_argument("--expected-run-id")
    parser.add_argument("--expected-git-commit", required=True)
    parser.add_argument("--expected-implementation-sha256", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if (args.results_root is None) != (args.expected_run_id is None):
            raise ValueError(
                "--results-root and --expected-run-id must be supplied together"
            )
        if args.results_root is None:
            report = verify_h23_incumbent_reuse_smoke_plan(
                H23IncumbentReuseSmokePlanVerificationRequest(
                    plan_path=Path(args.plan),
                    expected_git_commit=args.expected_git_commit,
                    expected_implementation_sha256=(
                        args.expected_implementation_sha256
                    ),
                )
            )
        else:
            report = verify_h23_incumbent_reuse_smoke_results(
                H23IncumbentReuseSmokeVerificationRequest(
                    plan_path=Path(args.plan),
                    results_root=Path(args.results_root),
                    expected_run_id=args.expected_run_id,
                    expected_git_commit=args.expected_git_commit,
                    expected_implementation_sha256=args.expected_implementation_sha256,
                )
            )
    except (
        H23IncumbentReuseSmokeVerificationError,
        OSError,
        ValueError,
    ) as error:
        print(
            f"H23 incumbent-reuse smoke verification failed: {error}", file=sys.stderr
        )
        return 2
    print(canonical_json(report.to_payload()))
    return 0


__all__ = [
    "H23IncumbentReuseSmokeVerificationError",
    "H23IncumbentReuseSmokePlanVerificationReport",
    "H23IncumbentReuseSmokePlanVerificationRequest",
    "H23IncumbentReuseSmokeVerificationReport",
    "H23IncumbentReuseSmokeVerificationRequest",
    "build_parser",
    "main",
    "verify_h23_incumbent_reuse_smoke_plan",
    "verify_h23_incumbent_reuse_smoke_results",
]


if __name__ == "__main__":
    raise SystemExit(main())
