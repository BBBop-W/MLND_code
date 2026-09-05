"""Frozen development-only H23 incumbent-reuse runtime smoke.

The plan command is solver-free.  The run command executes two qualification
jobs in one serial in-memory reuse pool, then constructs (but never optimizes)
one deliberately constraint-changing negative-control model.  Nothing emitted
by this module is eligible for the paper A--E pool or an algorithm comparison.
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
from .h23_qualification_diagnostic import (
    LoadedH23DiagnosticPlan,
    load_h23_diagnostic_plan,
)
from .h23_incumbent_reuse_contract import (
    LEGACY_REUSE_SMOKE_PLAN_FILE_SHA256,
    H23_DIAGNOSTIC_MAX_JOBS,
    H23_DIAGNOSTIC_PLAN_ID,
    REGISTERED_GEOMETRY_POLICY_PAYLOAD,
    REGISTERED_NATIVE_SETTINGS_PAYLOAD,
    REGISTERED_POLICY_PAYLOAD,
)
from .hashing import canonical_json, canonical_sha256
from .qualification import QualificationPolicy, assess_qualification
from .qualification_geometry_start import GeometryStartPolicy
from .qualification_incumbent_reuse import (
    DevelopmentQualificationIncumbentPool,
    QUALIFICATION_INCUMBENT_REUSE_POLICY_VERSION,
    QualificationConstraintSignature,
    QualificationReuseProvenance,
)
from .qualification_runner import (
    MLNDQualificationAdapter,
    NativeQualificationSettings,
    QualificationAdapter,
    QualificationInfrastructureError,
    QualificationModelAudit,
    QualificationModelContract,
    QualificationRunnerOutcome,
    qualify_canonical_instance,
)
from .resource_lock import (
    DEFAULT_BENCHMARK_RESOURCE_LOCK,
    BenchmarkResourceLock,
)


H23_REUSE_SMOKE_PLAN_ID = "h23-incumbent-reuse-smoke-v1"
H23_REUSE_SMOKE_PLAN_SCHEMA = "mlnd-h23-incumbent-reuse-smoke-plan-v1"
H23_REUSE_SMOKE_RUN_SCHEMA = "mlnd-h23-incumbent-reuse-smoke-run-v1"
H23_REUSE_SMOKE_CHECKPOINT_SCHEMA = "mlnd-h23-incumbent-reuse-smoke-checkpoint-v1"
H23_REUSE_SMOKE_SUMMARY_SCHEMA = "mlnd-h23-incumbent-reuse-smoke-summary-v1"
H23_REUSE_SMOKE_ACTION_INDICES = (4, 3, 10)
H23_REUSE_SMOKE_ACTION_ROLES = (
    "accepted_source_rerun",
    "objective_only_positive_target",
    "constraint_changing_negative_probe",
)
H23_REUSE_SMOKE_SOURCE_PLAN_SHA256 = (
    "f80bcea97062a521d081c81ff95ca55ae03741e7e3748b92ffdbfc74edd72d59"
)
H23_REUSE_SMOKE_SOURCE_RUN_ID = (
    "h23-qualification-diagnostic-run-"
    "f94dcee8e4cf8fbfb0a60cfbb3c51df5a5850e026e2a7d8da9624a493227f82a"
)
H23_REUSE_SMOKE_SOURCE_SUMMARY_SHA256 = (
    "a42a4ae7b9f615bec1e18bd0631359653beb6adefb33fff836d45e59f70f0856"
)
H23_REUSE_SMOKE_PHYSICAL_INSTANCE_SHA256 = (
    "153680bbdcbef0e31021a8257117cc85aa51a02ef0234420174dcb744a28eaef"
)
H23_REUSE_SMOKE_PHYSICAL_INSTANCE_ID = (
    "dev-tsl-20221017-20-noon__small__devseed0000001729"
)

_PLAN_ARTIFACT_TYPE = "mlnd_branch_cut_h23_incumbent_reuse_smoke_plan"
_RUN_ARTIFACT_TYPE = "mlnd_branch_cut_h23_incumbent_reuse_smoke_run_identity"
_CHECKPOINT_ARTIFACT_TYPE = "mlnd_branch_cut_h23_incumbent_reuse_smoke_checkpoint"
_SUMMARY_ARTIFACT_TYPE = "mlnd_branch_cut_h23_incumbent_reuse_smoke_summary"
_RUN_ID_PREFIX = "h23-incumbent-reuse-smoke-run-"
_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]

_IMPLEMENTATION_FILES = (
    "src/benchmark/branch_cut/h23_incumbent_reuse_contract.py",
    "src/benchmark/branch_cut/qualification_incumbent_reuse.py",
    "src/benchmark/branch_cut/qualification_runner.py",
    "src/benchmark/branch_cut/h23_incumbent_reuse_smoke.py",
    "src/benchmark/branch_cut/h23_incumbent_reuse_smoke_verifier.py",
)

_EXPECTED_PROFILE_IDENTITIES: Mapping[int, Mapping[str, object]] = {
    4: {
        "profile_id": (
            "devcal-demand-1x-feeder-035-basefc-024-fc-024-"
            "display-008-general-010-exact-false"
        ),
        "demand_scale": 1.0,
        "locker_fixed_cost_per_hour": 24.0,
        "qualification_case_sha256": (
            "a55c9146cb7f06e615887586a3359a50708b94b117606228a4f105e0c7050ee0"
        ),
        "effective_model_config_sha256": (
            "cf7f18c518e4259a47f8ac21b67e5dff2deb2f02d47be11fe6947279707c04b9"
        ),
        "diagnostic_job_id": (
            "h23-qualification-diagnostic-job-"
            "dacee6e6c0524d3225dd7c561361d9a8228e6be6af9ff3d71d69428cd23ce55e"
        ),
    },
    3: {
        "profile_id": (
            "devcal-demand-1x-feeder-035-basefc-020-fc-020-"
            "display-008-general-010-exact-false"
        ),
        "demand_scale": 1.0,
        "locker_fixed_cost_per_hour": 20.0,
        "qualification_case_sha256": (
            "a3d3b82be0fb27d8d11b04d606516f0ccfbdc5c93cdf3d5f56713f6b5c04b656"
        ),
        "effective_model_config_sha256": (
            "aefc0aa2312efd6dfef9cf79221ac7f9e354b7a72a72ad224438d2de8aafef8a"
        ),
        "diagnostic_job_id": (
            "h23-qualification-diagnostic-job-"
            "b99433a03fbf4709d8b02ac1b189f891cc10e0a04e4be67508cd3e45f5bcef99"
        ),
    },
    10: {
        "profile_id": (
            "devcal-demand-2x-feeder-035-basefc-024-fc-048-"
            "display-008-general-010-exact-false"
        ),
        "demand_scale": 2.0,
        "locker_fixed_cost_per_hour": 48.0,
        "qualification_case_sha256": (
            "1847ecc1112f66837ce11525db309ba796c838320470b94091c2aea889416469"
        ),
        "effective_model_config_sha256": (
            "cca4e068732e6e46119096491cb4299d5e54f4526ba542b29e5fe9036dace54a"
        ),
        "diagnostic_job_id": (
            "h23-qualification-diagnostic-job-"
            "1055c916f7ca20407b54e94113fe567248462b47633f97a9b7bc8c8d423f33fa"
        ),
    },
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
    "post_hoc_source_selection": True,
    "execution_order_original_indices": list(H23_REUSE_SMOKE_ACTION_INDICES),
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

_EXPECTED_EVENT_ACTIONS: Mapping[str, Sequence[str]] = {
    "accepted_source_rerun": ("reuse_cache_miss", "incumbent_offered"),
    "objective_only_positive_target": (
        "reuse_candidate_selected",
        "reuse_start_installed",
        "incumbent_offered",
    ),
    "constraint_changing_negative_probe": ("reuse_cache_miss",),
}


class H23IncumbentReuseSmokeError(RuntimeError):
    """A frozen smoke input, artifact, or semantic contract failed."""


class H23IncumbentReuseSmokeInfrastructureError(RuntimeError):
    """The runtime smoke stopped on an infrastructure failure."""


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


def _require_sha256(value: object, *, field_name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise H23IncumbentReuseSmokeError(
            f"{field_name} must be a lowercase SHA256 digest"
        )
    return value


def _read_canonical(path: Path, *, field_name: str) -> dict[str, Any]:
    try:
        raw = path.read_text(encoding="utf-8")
        payload = json.loads(raw)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise H23IncumbentReuseSmokeError(
            f"cannot read {field_name}: {error}"
        ) from error
    if not isinstance(payload, dict):
        raise H23IncumbentReuseSmokeError(f"{field_name} must be one JSON object")
    if raw != canonical_json(payload) + "\n":
        raise H23IncumbentReuseSmokeError(f"{field_name} is not canonical JSON")
    return payload


def _self_hashed_payload(
    payload: Mapping[str, Any], *, hash_field: str
) -> dict[str, Any]:
    if hash_field in payload:
        raise ValueError(f"payload already contains {hash_field}")
    entry = dict(payload)
    return {**entry, hash_field: canonical_sha256(entry)}


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
        raise H23IncumbentReuseSmokeError(
            f"git {' '.join(arguments)} failed: {message}"
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

    repository = dict(_capture_repository_state(repository_root))
    return {
        "repository": repository,
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
        raise H23IncumbentReuseSmokeError(
            "cannot audit concurrent solver/test processes: "
            + (completed.stderr or completed.stdout).strip()
        )
    return completed.stdout


def _process_audit(process_table: str) -> dict[str, Any]:
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
        "h23_incumbent_reuse_smoke run",
    )
    rows: dict[int, tuple[int, str]] = {}
    for raw_line in process_table.splitlines():
        fields = raw_line.strip().split(maxsplit=2)
        try:
            pid, ppid = int(fields[0]), int(fields[1])
            line = fields[2]
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
        if pid not in excluded and any(marker in line for marker in markers)
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
    for key in ("status_lines", "worktree_dirty"):
        repository.pop(key, None)
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


def _audit_matches_contract(
    audit: QualificationModelAudit, contract: QualificationModelContract
) -> bool:
    return audit.contract_values() == (
        contract.effective_model_config_sha256,
        contract.demand_scale,
        contract.locker_fixed_cost_per_hour,
        contract.display_origin_choices,
        contract.general_corridor_choices,
        contract.screening_is_exact,
    )


def _validate_instance(
    instance: object, *, config: ModelConfig, demand_scale: float
) -> None:
    if getattr(instance, "config", None) != config:
        raise QualificationInfrastructureError("loaded smoke instance config drifted")
    metadata = getattr(instance, "metadata", None)
    observed = metadata.get("demand_scale") if isinstance(metadata, Mapping) else None
    if isinstance(observed, bool) or not isinstance(observed, (int, float)):
        raise QualificationInfrastructureError("loaded smoke demand scale is missing")
    if float(observed) != float(demand_scale):
        raise QualificationInfrastructureError("loaded smoke demand scale drifted")
    screening = getattr(instance, "screening_is_exact", None)
    if not callable(screening) or screening("MX") is not False:
        raise QualificationInfrastructureError(
            "smoke requires screening_is_exact=false"
        )


def _validate_qualification_outcome(
    outcome: QualificationRunnerOutcome,
    *,
    action: Mapping[str, Any],
    run_id: str,
    policy: QualificationPolicy,
    require_accepted: bool,
) -> None:
    if not isinstance(outcome, QualificationRunnerOutcome):
        raise QualificationInfrastructureError("qualifier returned an invalid outcome")
    job = action["source_job"]
    contract = QualificationModelContract(**job["qualification_model_contract"])
    if outcome.expected_model_contract != contract:
        raise QualificationInfrastructureError("qualification contract drifted")
    if outcome.decision != assess_qualification(outcome.evidence, policy):
        raise QualificationInfrastructureError(
            "qualification decision did not recompute"
        )
    expected_identity = (
        action["smoke_case_id"],
        "small",
        run_id,
        H23_REUSE_SMOKE_PHYSICAL_INSTANCE_SHA256,
        contract.qualification_case_sha256,
        contract.effective_model_config_sha256,
        False,
    )
    for record in (outcome.evidence, outcome.decision):
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
            raise QualificationInfrastructureError("qualification identity drifted")
    if outcome.witness_model_audit is None or not _audit_matches_contract(
        outcome.witness_model_audit, contract
    ):
        raise QualificationInfrastructureError("witness model audit drifted")
    if outcome.exception_messages:
        raise QualificationInfrastructureError("qualification recorded exceptions")
    if require_accepted and not outcome.accepted:
        return
    if outcome.accepted and (
        not outcome.fresh_model_validation_performed
        or outcome.fresh_model_audit is None
        or not _audit_matches_contract(outcome.fresh_model_audit, contract)
        or not outcome.evidence.witness_feasible
        or not outcome.decision.accepted
        or outcome.captured_variable_count <= 0
    ):
        raise QualificationInfrastructureError(
            "accepted qualification lacks a fresh validated complete witness"
        )


@dataclass(frozen=True, slots=True)
class SignatureOnlyProbeOutcome:
    """Audit returned by a model-build-only negative-control probe."""

    model_audit: QualificationModelAudit
    constraint_signature: QualificationConstraintSignature
    solve_started: bool = False
    start_installation_attempted: bool = False
    adapter_disposed: bool = True

    def __post_init__(self) -> None:
        if self.solve_started is not False:
            raise ValueError("signature-only probe cannot start a solve")
        if self.start_installation_attempted is not False:
            raise ValueError("signature-only probe cannot install a start")
        if self.adapter_disposed is not True:
            raise ValueError("signature-only probe must dispose its adapter")

    def to_payload(self) -> dict[str, Any]:
        return {
            "model_audit": self.model_audit.to_payload(),
            "constraint_signature": self.constraint_signature.to_payload(),
            "solve_started": self.solve_started,
            "start_installation_attempted": self.start_installation_attempted,
            "adapter_disposed": self.adapter_disposed,
        }


def probe_constraint_signature_only(
    instance: object,
    mode: str,
    *,
    expected_model_contract: QualificationModelContract,
    adapter_factory: Callable[[object, str], QualificationAdapter] = (
        _default_adapter_factory
    ),
) -> SignatureOnlyProbeOutcome:
    """Build, audit, hash, and dispose one model without any optimize call."""

    adapter: QualificationAdapter | None = None
    primary_error: BaseException | None = None
    try:
        adapter = adapter_factory(instance, mode)
        audit = adapter.qualification_model_audit()
        if not isinstance(
            audit, QualificationModelAudit
        ) or not _audit_matches_contract(audit, expected_model_contract):
            raise QualificationInfrastructureError(
                "negative-probe model audit differs from its frozen contract"
            )
        signature = adapter.qualification_constraint_signature()
        if not isinstance(signature, QualificationConstraintSignature):
            raise QualificationInfrastructureError(
                "negative probe returned an invalid constraint signature"
            )
    except BaseException as error:
        primary_error = error
        raise
    finally:
        if adapter is not None:
            try:
                adapter.dispose()
            except BaseException as cleanup_error:
                if primary_error is not None:
                    primary_error.add_note(
                        "secondary negative-probe adapter disposal failure: "
                        f"{type(cleanup_error).__name__}: {cleanup_error}"
                    )
                else:
                    raise QualificationInfrastructureError(
                        "negative-probe adapter disposal failed: "
                        f"{type(cleanup_error).__name__}: {cleanup_error}"
                    ) from cleanup_error
    return SignatureOnlyProbeOutcome(
        model_audit=audit,
        constraint_signature=signature,
    )


def build_signature_comparison(
    *,
    role: str,
    expected_relation: str,
    source: QualificationConstraintSignature,
    target: QualificationConstraintSignature,
    source_demand_scale: float,
    target_demand_scale: float,
) -> dict[str, Any]:
    """Return a complete explicit equality/difference audit."""

    if expected_relation not in {"equal", "different"}:
        raise ValueError("expected_relation must be equal or different")
    equal = source == target
    observed = "equal" if equal else "different"
    payload = {
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
            source.objective_excluded is True and target.objective_excluded is True
        ),
        "extraction_complete_both": (
            source.extraction_complete is True and target.extraction_complete is True
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
    return payload


def _case_signature(
    pool: DevelopmentQualificationIncumbentPool, case_id: str
) -> QualificationConstraintSignature:
    events = pool.events_for_case(case_id)
    if not events:
        raise QualificationInfrastructureError("reuse case has no audit events")
    signatures = {event.target_constraint_signature for event in events}
    if len(signatures) != 1:
        raise QualificationInfrastructureError("reuse case signatures are inconsistent")
    return next(iter(signatures))


def _case_event_actions(
    pool: DevelopmentQualificationIncumbentPool, case_id: str
) -> tuple[str, ...]:
    return tuple(event.action for event in pool.events_for_case(case_id))


def _selected_source_payload(
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


def _validate_source_summary(path: Path) -> tuple[dict[str, Any], str]:
    payload = _read_canonical(path, field_name="source H23 summary")
    if (
        payload.get("diagnostic_run_id") != H23_REUSE_SMOKE_SOURCE_RUN_ID
        or payload.get("plan_sha256") != H23_REUSE_SMOKE_SOURCE_PLAN_SHA256
        or payload.get("summary_sha256") != H23_REUSE_SMOKE_SOURCE_SUMMARY_SHA256
        or payload.get("status") != "completed"
        or payload.get("accepted_profile_order_indices") != [4]
        or payload.get("eligible_for_formal_pool") is not False
    ):
        raise H23IncumbentReuseSmokeError("source H23 summary identity drifted")
    entry = {
        key: value
        for key, value in payload.items()
        if key not in {"artifact_type", "artifact_schema_version", "summary_sha256"}
    }
    if canonical_sha256(entry) != H23_REUSE_SMOKE_SOURCE_SUMMARY_SHA256:
        raise H23IncumbentReuseSmokeError("source H23 summary self-hash drifted")
    return payload, _sha256_file(path)


def _smoke_actions(
    h23_plan: LoadedH23DiagnosticPlan,
    source_summary: Mapping[str, Any],
) -> tuple[dict[str, Any], ...]:
    ledger = {
        int(record["diagnostic_job_order_index"]): record
        for record in source_summary["diagnostic_job_ledger"]
    }
    actions: list[dict[str, Any]] = []
    expected_relations = ("establish", "equal", "different")
    for action_index, (original_index, role, relation) in enumerate(
        zip(
            H23_REUSE_SMOKE_ACTION_INDICES,
            H23_REUSE_SMOKE_ACTION_ROLES,
            expected_relations,
            strict=True,
        )
    ):
        job = dict(h23_plan.jobs[original_index])
        expected = _EXPECTED_PROFILE_IDENTITIES[original_index]
        source_job = job["source_job"]
        profile = job["qualification_solve_profile"]
        contract = job["qualification_model_contract"]
        observed = {
            "profile_id": source_job["profile_id"],
            "demand_scale": profile["demand_scale"],
            "locker_fixed_cost_per_hour": profile["locker_fixed_cost_per_hour"],
            "qualification_case_sha256": contract["qualification_case_sha256"],
            "effective_model_config_sha256": contract["effective_model_config_sha256"],
            "diagnostic_job_id": job["diagnostic_job_id"],
        }
        if observed != expected:
            raise H23IncumbentReuseSmokeError(
                f"source H23 job {original_index} identity drifted"
            )
        if (
            source_job["instance_sha256"] != H23_REUSE_SMOKE_PHYSICAL_INSTANCE_SHA256
            or source_job["physical_instance_id"]
            != H23_REUSE_SMOKE_PHYSICAL_INSTANCE_ID
            or source_job["scale"] != "small"
            or contract["screening_is_exact"] is not False
        ):
            raise H23IncumbentReuseSmokeError("smoke physical instance drifted")
        historical = ledger[original_index]
        checkpoint_path = Path(str(historical["checkpoint_path"]))
        smoke_case_id = "h23-incumbent-reuse-smoke-case-" + canonical_sha256(
            {
                "plan_id": H23_REUSE_SMOKE_PLAN_ID,
                "action_index": action_index,
                "role": role,
                "source_diagnostic_job_id": job["diagnostic_job_id"],
            }
        )
        actions.append(
            {
                "action_index": action_index,
                "role": role,
                "original_profile_order_index": original_index,
                "expected_signature_relation_to_source": relation,
                "smoke_case_id": smoke_case_id,
                "source_job": job,
                "historical_checkpoint": {
                    "path": str(checkpoint_path.resolve()),
                    "file_sha256": _sha256_file(checkpoint_path),
                    "checkpoint_sha256": historical["checkpoint_sha256"],
                    "accepted": historical["accepted"],
                },
                "expected_pool_event_actions": list(_EXPECTED_EVENT_ACTIONS[role]),
            }
        )
    if actions[0]["historical_checkpoint"]["accepted"] is not True or any(
        action["historical_checkpoint"]["accepted"] is not False
        for action in actions[1:]
    ):
        raise H23IncumbentReuseSmokeError("historical action outcomes drifted")
    return tuple(actions)


@dataclass(frozen=True, slots=True)
class H23IncumbentReuseSmokePlanBuildHooks:
    capture_repository_state: Callable[[Path], Mapping[str, Any]] = (
        _capture_repository_state
    )


@dataclass(frozen=True, slots=True)
class H23IncumbentReuseSmokePlanBuildConfig:
    source_h23_plan_path: Path
    source_h23_summary_path: Path
    output_path: Path
    expected_git_commit: str
    plan_id: str = H23_REUSE_SMOKE_PLAN_ID

    def __post_init__(self) -> None:
        if self.plan_id != H23_REUSE_SMOKE_PLAN_ID:
            raise ValueError(f"plan_id must be {H23_REUSE_SMOKE_PLAN_ID}")
        if len(self.expected_git_commit) != 40 or any(
            c not in "0123456789abcdef" for c in self.expected_git_commit
        ):
            raise ValueError("expected_git_commit must be a lowercase Git digest")


@dataclass(frozen=True, slots=True)
class H23IncumbentReuseSmokePlanBuildResult:
    path: Path
    plan_sha256: str
    plan_file_sha256: str
    git_commit: str
    implementation_sha256: str

    def to_payload(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "plan_id": H23_REUSE_SMOKE_PLAN_ID,
            "plan_sha256": self.plan_sha256,
            "plan_file_sha256": self.plan_file_sha256,
            "action_count": 3,
            "git_commit": self.git_commit,
            "implementation_sha256": self.implementation_sha256,
        }


def build_h23_incumbent_reuse_smoke_plan(
    config: H23IncumbentReuseSmokePlanBuildConfig,
    *,
    hooks: H23IncumbentReuseSmokePlanBuildHooks = (
        H23IncumbentReuseSmokePlanBuildHooks()
    ),
) -> H23IncumbentReuseSmokePlanBuildResult:
    """Build one exclusive, canonical, solver-free smoke plan."""

    state = dict(hooks.capture_repository_state(_REPOSITORY_ROOT))
    if (
        state.get("git_commit") != config.expected_git_commit
        or state.get("worktree_dirty") is not False
        or state.get("status_includes_untracked") is not True
        or state.get("status_lines") != []
    ):
        raise H23IncumbentReuseSmokeError(
            "plan generation requires the exact clean expected commit"
        )
    implementation = dict(state["implementation"])
    if set(implementation["ordered_file_sha256"]) != set(_IMPLEMENTATION_FILES):
        raise H23IncumbentReuseSmokeError("smoke implementation file set drifted")
    h23_plan = load_h23_diagnostic_plan(
        config.source_h23_plan_path,
        expected_plan_id=H23_DIAGNOSTIC_PLAN_ID,
        expected_max_jobs=H23_DIAGNOSTIC_MAX_JOBS,
    )
    if h23_plan.plan_sha256 != H23_REUSE_SMOKE_SOURCE_PLAN_SHA256:
        raise H23IncumbentReuseSmokeError("source H23 plan SHA drifted")
    summary_path = config.source_h23_summary_path.expanduser().resolve(strict=True)
    source_summary, source_summary_file_sha = _validate_source_summary(summary_path)
    actions = _smoke_actions(h23_plan, source_summary)
    entry = {
        "artifact_type": _PLAN_ARTIFACT_TYPE,
        "artifact_schema_version": 1,
        "plan_schema": H23_REUSE_SMOKE_PLAN_SCHEMA,
        "plan_id": H23_REUSE_SMOKE_PLAN_ID,
        **dict(_BOUNDARY_FLAGS),
        "mode": "MX",
        "action_count": 3,
        "reuse_policy_version": QUALIFICATION_INCUMBENT_REUSE_POLICY_VERSION,
        "order_bias": dict(_ORDER_BIAS),
        "execution_contract": dict(_EXECUTION_CONTRACT),
        "qualification_policy": dict(REGISTERED_POLICY_PAYLOAD),
        "native_settings": dict(REGISTERED_NATIVE_SETTINGS_PAYLOAD),
        "geometry_start_policy": dict(REGISTERED_GEOMETRY_POLICY_PAYLOAD),
        "source_h23": {
            "plan_path": str(h23_plan.path),
            "plan_sha256": h23_plan.plan_sha256,
            "plan_file_sha256": h23_plan.file_sha256,
            "summary_path": str(summary_path),
            "summary_sha256": H23_REUSE_SMOKE_SOURCE_SUMMARY_SHA256,
            "summary_file_sha256": source_summary_file_sha,
            "diagnostic_run_id": H23_REUSE_SMOKE_SOURCE_RUN_ID,
        },
        "required_execution_environment": {
            "repository_root": state["repository_root"],
            "git_commit": config.expected_git_commit,
            "clean_status_includes_untracked": True,
            "implementation": implementation,
        },
        "actions": list(actions),
    }
    payload = _self_hashed_payload(entry, hash_field="plan_sha256")
    output = config.output_path.expanduser().absolute()
    output.parent.mkdir(parents=True, exist_ok=True)
    published = write_canonical_json_exclusive(output, payload)
    return H23IncumbentReuseSmokePlanBuildResult(
        path=published,
        plan_sha256=payload["plan_sha256"],
        plan_file_sha256=_sha256_file(published),
        git_commit=config.expected_git_commit,
        implementation_sha256=implementation["implementation_sha256"],
    )


@dataclass(frozen=True, slots=True)
class LoadedH23IncumbentReuseSmokePlan:
    path: Path
    file_sha256: str
    plan_sha256: str
    payload: Mapping[str, Any]
    source_h23_plan: LoadedH23DiagnosticPlan
    actions: tuple[Mapping[str, Any], ...]


def load_h23_incumbent_reuse_smoke_plan(
    plan_path: str | Path,
) -> LoadedH23IncumbentReuseSmokePlan:
    """Strictly reload and rebind a frozen smoke plan."""

    path = Path(plan_path).expanduser().resolve(strict=True)
    payload = _read_canonical(path, field_name="incumbent-reuse smoke plan")
    plan_sha = _require_sha256(payload.get("plan_sha256"), field_name="plan_sha256")
    unhashed = {key: value for key, value in payload.items() if key != "plan_sha256"}
    if canonical_sha256(unhashed) != plan_sha:
        raise H23IncumbentReuseSmokeError("smoke plan self-hash mismatch")
    fixed = {
        "artifact_type": _PLAN_ARTIFACT_TYPE,
        "artifact_schema_version": 1,
        "plan_schema": H23_REUSE_SMOKE_PLAN_SCHEMA,
        "plan_id": H23_REUSE_SMOKE_PLAN_ID,
        **dict(_BOUNDARY_FLAGS),
        "mode": "MX",
        "action_count": 3,
        "reuse_policy_version": QUALIFICATION_INCUMBENT_REUSE_POLICY_VERSION,
        "order_bias": dict(_ORDER_BIAS),
        "execution_contract": dict(_EXECUTION_CONTRACT),
        "qualification_policy": dict(REGISTERED_POLICY_PAYLOAD),
        "native_settings": dict(REGISTERED_NATIVE_SETTINGS_PAYLOAD),
        "geometry_start_policy": dict(REGISTERED_GEOMETRY_POLICY_PAYLOAD),
    }
    if any(payload.get(key) != value for key, value in fixed.items()):
        raise H23IncumbentReuseSmokeError("smoke plan fixed contract drifted")
    source = payload.get("source_h23")
    if not isinstance(source, Mapping):
        raise H23IncumbentReuseSmokeError("smoke plan source_h23 is invalid")
    h23_plan = load_h23_diagnostic_plan(
        source["plan_path"],
        expected_plan_id=H23_DIAGNOSTIC_PLAN_ID,
        expected_max_jobs=H23_DIAGNOSTIC_MAX_JOBS,
    )
    if (
        h23_plan.plan_sha256 != H23_REUSE_SMOKE_SOURCE_PLAN_SHA256
        or h23_plan.file_sha256 != source["plan_file_sha256"]
    ):
        raise H23IncumbentReuseSmokeError("bound source H23 plan changed")
    summary_path = Path(str(source["summary_path"])).resolve(strict=True)
    summary, summary_file_sha = _validate_source_summary(summary_path)
    if (
        summary_file_sha != source["summary_file_sha256"]
        or source["summary_sha256"] != H23_REUSE_SMOKE_SOURCE_SUMMARY_SHA256
        or source["diagnostic_run_id"] != H23_REUSE_SMOKE_SOURCE_RUN_ID
    ):
        raise H23IncumbentReuseSmokeError("bound source H23 summary changed")
    expected_actions = _smoke_actions(h23_plan, summary)
    if canonical_json(payload.get("actions")) != canonical_json(expected_actions):
        raise H23IncumbentReuseSmokeError("smoke action identities drifted")
    required = payload.get("required_execution_environment")
    if not isinstance(required, Mapping):
        raise H23IncumbentReuseSmokeError("required environment is invalid")
    implementation = required.get("implementation")
    required_files = (
        set(implementation.get("ordered_file_sha256", {}))
        if isinstance(implementation, Mapping)
        else set()
    )
    legacy_files = set(_IMPLEMENTATION_FILES) - {
        "src/benchmark/branch_cut/h23_incumbent_reuse_contract.py"
    }
    pinned_legacy_source = (
        required_files == legacy_files
        and _sha256_file(path) == LEGACY_REUSE_SMOKE_PLAN_FILE_SHA256
    )
    if required_files != set(_IMPLEMENTATION_FILES) and not pinned_legacy_source:
        raise H23IncumbentReuseSmokeError("required implementation set drifted")
    return LoadedH23IncumbentReuseSmokePlan(
        path=path,
        file_sha256=_sha256_file(path),
        plan_sha256=plan_sha,
        payload=payload,
        source_h23_plan=h23_plan,
        actions=tuple(expected_actions),
    )


@dataclass(frozen=True, slots=True)
class H23IncumbentReuseSmokeHooks:
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
class H23IncumbentReuseSmokeRunConfig:
    plan_path: Path
    results_root: Path
    resource_lock: Path = field(default_factory=lambda: DEFAULT_BENCHMARK_RESOURCE_LOCK)


@dataclass(frozen=True, slots=True)
class H23IncumbentReuseSmokeRunResult:
    results_root: Path
    run_identity_path: Path
    summary_path: Path
    smoke_run_id: str
    conclusion: str
    checkpoint_paths: tuple[Path, ...]

    def to_payload(self) -> dict[str, Any]:
        return {
            "results_root": str(self.results_root),
            "run_identity_path": str(self.run_identity_path),
            "summary_path": str(self.summary_path),
            "smoke_run_id": self.smoke_run_id,
            "conclusion": self.conclusion,
            "checkpoint_paths": [str(path) for path in self.checkpoint_paths],
        }


def _validate_runtime_environment(
    environment: Mapping[str, Any], plan: LoadedH23IncumbentReuseSmokePlan
) -> None:
    repository = environment.get("repository")
    if not isinstance(repository, Mapping):
        raise H23IncumbentReuseSmokeError("runtime repository audit is missing")
    required = plan.payload["required_execution_environment"]
    if (
        repository.get("repository_root") != required["repository_root"]
        or repository.get("git_commit") != required["git_commit"]
        or repository.get("status_includes_untracked") is not True
        or repository.get("worktree_dirty") is not False
        or repository.get("status_lines") != []
        or repository.get("implementation") != required["implementation"]
    ):
        raise H23IncumbentReuseSmokeError("runtime repository or code drifted")
    if (
        not isinstance(environment.get("gurobi_version"), str)
        or not environment["gurobi_version"]
    ):
        raise H23IncumbentReuseSmokeError("runtime Gurobi version is missing")


def _claim_results_root(target: Path, plan: LoadedH23IncumbentReuseSmokePlan) -> Path:
    absolute = target.expanduser().absolute()
    if absolute.exists() or absolute.is_symlink():
        raise H23IncumbentReuseSmokeError(f"results root already exists: {absolute}")
    if "formal" in {part.lower() for part in absolute.parts}:
        raise H23IncumbentReuseSmokeError("smoke results cannot be stored under formal")
    forbidden = (
        plan.source_h23_plan.path.parent,
        Path(str(plan.payload["source_h23"]["summary_path"])).parent,
    )
    resolved_parent = absolute.parent.resolve()
    for root in forbidden:
        try:
            resolved_parent.relative_to(root.resolve())
        except ValueError:
            continue
        raise H23IncumbentReuseSmokeError(
            "smoke results cannot be inside a frozen source artifact root"
        )
    absolute.parent.mkdir(parents=True, exist_ok=True)
    absolute.mkdir(exist_ok=False)
    return absolute.resolve()


def _revalidate_action_inputs(
    plan: LoadedH23IncumbentReuseSmokePlan, action: Mapping[str, Any]
) -> None:
    if _sha256_file(plan.path) != plan.file_sha256:
        raise QualificationInfrastructureError("smoke plan changed during run")
    source = plan.payload["source_h23"]
    for path_key, hash_key in (
        ("plan_path", "plan_file_sha256"),
        ("summary_path", "summary_file_sha256"),
    ):
        if _sha256_file(Path(str(source[path_key]))) != source[hash_key]:
            raise QualificationInfrastructureError("bound H23 source changed")
    historical = action["historical_checkpoint"]
    if _sha256_file(Path(str(historical["path"]))) != historical["file_sha256"]:
        raise QualificationInfrastructureError("historical checkpoint changed")
    job = action["source_job"]["source_job"]
    instance_dir = Path(str(action["source_job"]["physical_instance_directory"]))
    observed, _ = canonical_instance_hash(instance_dir)
    if observed != job["instance_sha256"]:
        raise QualificationInfrastructureError("physical smoke instance changed")


def _write_checkpoint(
    root: Path,
    *,
    identity: Mapping[str, Any],
    result: Mapping[str, Any],
) -> tuple[Path, str, str]:
    action = identity["action"]
    action_index = int(action["action_index"])
    role = str(action["role"])
    name = f"action-{action_index:02d}-{role}.json"
    payload = _self_hashed_payload(
        {
            "artifact_type": _CHECKPOINT_ARTIFACT_TYPE,
            "artifact_schema_version": 1,
            "checkpoint_schema": H23_REUSE_SMOKE_CHECKPOINT_SCHEMA,
            **dict(_BOUNDARY_FLAGS),
            "contains_algorithm_comparison_results": False,
            "identity": dict(identity),
            "result": dict(result),
        },
        hash_field="checkpoint_sha256",
    )
    path = write_canonical_json_exclusive(root / name, payload)
    return path, payload["checkpoint_sha256"], _sha256_file(path)


def _write_summary(
    results_root: Path,
    *,
    run_id: str,
    run_sha: str,
    plan: LoadedH23IncumbentReuseSmokePlan,
    conclusion: str,
    checkpoint_records: Sequence[Mapping[str, Any]],
    pool: DevelopmentQualificationIncumbentPool,
    positive_comparison: Mapping[str, Any] | None,
    negative_comparison: Mapping[str, Any] | None,
    completed_at_utc: str,
    infrastructure_failure: Mapping[str, Any] | None = None,
) -> Path:
    passed = conclusion == "passed"
    payload = _self_hashed_payload(
        {
            "artifact_type": _SUMMARY_ARTIFACT_TYPE,
            "artifact_schema_version": 1,
            "summary_schema": H23_REUSE_SMOKE_SUMMARY_SCHEMA,
            **dict(_BOUNDARY_FLAGS),
            "contains_algorithm_comparison_results": False,
            "smoke_run_id": run_id,
            "run_identity_sha256": run_sha,
            "plan_id": H23_REUSE_SMOKE_PLAN_ID,
            "plan_sha256": plan.plan_sha256,
            "order_bias": dict(_ORDER_BIAS),
            "conclusion": conclusion,
            "passed": passed,
            "attempted_action_count": len(checkpoint_records),
            "completed_action_count": sum(
                record["status"] == "completed" for record in checkpoint_records
            ),
            "infrastructure_failure_count": int(infrastructure_failure is not None),
            "infrastructure_failure": (
                dict(infrastructure_failure)
                if infrastructure_failure is not None
                else None
            ),
            "checkpoints": [dict(record) for record in checkpoint_records],
            "positive_signature_comparison": (
                dict(positive_comparison) if positive_comparison is not None else None
            ),
            "negative_signature_comparison": (
                dict(negative_comparison) if negative_comparison is not None else None
            ),
            "reuse_pool_audit": pool.to_payload(),
            "formal_pool_created": False,
            "locked_profile_ids": [],
            "completed_at_utc": completed_at_utc,
        },
        hash_field="summary_sha256",
    )
    name = (
        "batch_summary.json" if infrastructure_failure is None else "batch_aborted.json"
    )
    return write_canonical_json_exclusive(results_root / name, payload)


def _infrastructure_payload(
    *, stage: str, action: Mapping[str, Any], error: BaseException
) -> dict[str, Any]:
    return {
        "stage": stage,
        "action_index": action["action_index"],
        "role": action["role"],
        "type": type(error).__name__,
        "message": str(error),
    }


def run_h23_incumbent_reuse_smoke(
    config: H23IncumbentReuseSmokeRunConfig,
    *,
    hooks: H23IncumbentReuseSmokeHooks = H23IncumbentReuseSmokeHooks(),
) -> H23IncumbentReuseSmokeRunResult:
    """Execute the frozen three-action smoke in one serial process."""

    plan = load_h23_incumbent_reuse_smoke_plan(config.plan_path)
    policy = _registered_policy()
    settings = _registered_settings()
    geometry_policy = _registered_geometry_policy()
    with BenchmarkResourceLock(
        config.resource_lock, command="h23-incumbent-reuse-smoke"
    ):
        environment = dict(hooks.capture_environment(_REPOSITORY_ROOT))
        _validate_runtime_environment(environment, plan)
        environment_sha = canonical_sha256(environment)
        stable_environment_sha = canonical_sha256(_stable_environment(environment))
        initial_process_audit = _process_audit(hooks.process_table())
        if not initial_process_audit["valid"]:
            raise H23IncumbentReuseSmokeError(
                "conflicting solver/test processes are active"
            )
        required = plan.payload["required_execution_environment"]
        run_identity = {
            "run_schema": H23_REUSE_SMOKE_RUN_SCHEMA,
            **dict(_BOUNDARY_FLAGS),
            "contains_algorithm_comparison_results": False,
            "plan_id": H23_REUSE_SMOKE_PLAN_ID,
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
        run_identity_payload = {
            "artifact_type": _RUN_ARTIFACT_TYPE,
            "artifact_schema_version": 1,
            "smoke_run_id": run_id,
            "run_identity": run_identity,
            "run_identity_sha256": run_sha,
        }
        run_identity_path = write_canonical_json_exclusive(
            results_root / "run_identity.json", run_identity_payload
        )
        pool = DevelopmentQualificationIncumbentPool()
        checkpoint_records: list[dict[str, Any]] = []
        checkpoint_paths: list[Path] = []
        source_signature: QualificationConstraintSignature | None = None
        positive_comparison: dict[str, Any] | None = None
        negative_comparison: dict[str, Any] | None = None
        target_accepted = False

        def revalidate_environment() -> None:
            observed = dict(hooks.capture_environment(_REPOSITORY_ROOT))
            _validate_runtime_environment(observed, plan)
            if canonical_sha256(_stable_environment(observed)) != (
                stable_environment_sha
            ):
                raise QualificationInfrastructureError(
                    "runtime environment changed during smoke"
                )

        for action in plan.actions:
            stage = "process_conflict_check"
            process_audit = _process_audit(hooks.process_table())
            identity = {
                "smoke_run_id": run_id,
                "run_identity_sha256": run_sha,
                "plan_id": H23_REUSE_SMOKE_PLAN_ID,
                "plan_sha256": plan.plan_sha256,
                "plan_file_sha256": plan.file_sha256,
                "environment_sha256": environment_sha,
                "git_commit": required["git_commit"],
                "action": dict(action),
                "process_audit": process_audit,
            }
            try:
                if not process_audit["valid"]:
                    raise QualificationInfrastructureError(
                        "conflicting process appeared before smoke action"
                    )
                stage = "pre_action_revalidation"
                revalidate_environment()
                _revalidate_action_inputs(plan, action)
                job = action["source_job"]
                source_job = job["source_job"]
                profile = plan.source_h23_plan.manifest.profiles[
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
                _validate_instance(
                    instance, config=model_config, demand_scale=profile.demand_scale
                )
                role = str(action["role"])
                case_id = str(action["smoke_case_id"])
                if role != "constraint_changing_negative_probe":
                    stage = "qualification"
                    outcome = hooks.qualify_instance(
                        instance,
                        "MX",
                        case_id=case_id,
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
                    )
                    _validate_qualification_outcome(
                        outcome,
                        action=action,
                        run_id=run_id,
                        policy=policy,
                        require_accepted=True,
                    )
                    signature = _case_signature(pool, case_id)
                    events = _case_event_actions(pool, case_id)
                    selected_source = _selected_source_payload(pool, case_id)
                    if role == "accepted_source_rerun":
                        source_signature = signature
                        expected_events = (
                            tuple(_EXPECTED_EVENT_ACTIONS[role])
                            if outcome.accepted
                            else ("reuse_cache_miss",)
                        )
                        if events != expected_events:
                            raise QualificationInfrastructureError(
                                "source reuse event sequence drifted"
                            )
                    else:
                        if source_signature is None:
                            raise QualificationInfrastructureError(
                                "positive target lacks a source signature"
                            )
                        positive_comparison = build_signature_comparison(
                            role="objective_only_positive_target",
                            expected_relation="equal",
                            source=source_signature,
                            target=signature,
                            source_demand_scale=1.0,
                            target_demand_scale=profile.demand_scale,
                        )
                        if not positive_comparison["comparison_passed"]:
                            raise QualificationInfrastructureError(
                                "objective-only signatures differ"
                            )
                        expected_events = (
                            tuple(_EXPECTED_EVENT_ACTIONS[role])
                            if outcome.accepted
                            else (
                                "reuse_candidate_selected",
                                "reuse_start_installed",
                            )
                        )
                        if events != expected_events:
                            raise QualificationInfrastructureError(
                                "positive-target reuse event sequence drifted"
                            )
                        if (
                            selected_source is None
                            or selected_source["source_provenance"]["profile_id"]
                            != _EXPECTED_PROFILE_IDENTITIES[4]["profile_id"]
                        ):
                            raise QualificationInfrastructureError(
                                "positive target did not retain index-4 source"
                            )
                        target_accepted = outcome.accepted
                    reuse_case_audit = pool.case_audit_payload(case_id)
                    result = {
                        "status": "completed",
                        "completed_at_utc": hooks.now_utc(),
                        "infrastructure_error": None,
                        "model_audit": outcome.witness_model_audit.to_payload(),
                        "constraint_signature": signature.to_payload(),
                        "signature_comparison": (
                            positive_comparison
                            if role == "objective_only_positive_target"
                            else None
                        ),
                        "reuse_case_audit": reuse_case_audit,
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
                        "qualification_outcome": outcome.to_payload(),
                    }
                else:
                    if source_signature is None:
                        raise QualificationInfrastructureError(
                            "negative probe lacks a source signature"
                        )
                    stage = "negative_signature_only_probe"
                    probe = probe_constraint_signature_only(
                        instance,
                        "MX",
                        expected_model_contract=contract,
                        adapter_factory=hooks.adapter_factory,
                    )
                    negative_comparison = build_signature_comparison(
                        role="constraint_changing_negative_probe",
                        expected_relation="different",
                        source=source_signature,
                        target=probe.constraint_signature,
                        source_demand_scale=1.0,
                        target_demand_scale=profile.demand_scale,
                    )
                    if not negative_comparison["comparison_passed"]:
                        raise QualificationInfrastructureError(
                            "constraint-changing signature unexpectedly matched"
                        )
                    provenance = QualificationReuseProvenance(
                        qualification_run_id=run_id,
                        case_id=case_id,
                        profile_id=profile.profile_id,
                        scale="small",
                        mode="MX",
                        physical_instance_sha256=(
                            H23_REUSE_SMOKE_PHYSICAL_INSTANCE_SHA256
                        ),
                        qualification_case_sha256=(contract.qualification_case_sha256),
                        effective_model_config_sha256=(
                            probe.model_audit.effective_model_config_sha256
                        ),
                        demand_scale=probe.model_audit.demand_scale,
                        screening_is_exact=False,
                    )
                    selected = pool.select(
                        target_provenance=provenance,
                        target_constraint_signature=probe.constraint_signature,
                    )
                    events = _case_event_actions(pool, case_id)
                    if selected is not None or events != tuple(
                        _EXPECTED_EVENT_ACTIONS[role]
                    ):
                        raise QualificationInfrastructureError(
                            "negative control did not fail closed"
                        )
                    result = {
                        "status": "completed",
                        "completed_at_utc": hooks.now_utc(),
                        "infrastructure_error": None,
                        "model_audit": probe.model_audit.to_payload(),
                        "constraint_signature": (
                            probe.constraint_signature.to_payload()
                        ),
                        "signature_comparison": negative_comparison,
                        "reuse_case_audit": pool.case_audit_payload(case_id),
                        "reuse_pool_audit_snapshot": pool.to_payload(),
                        "reuse_candidate_selected": False,
                        "reuse_start_installation_attempted": False,
                        "reuse_start_installed": False,
                        "selected_source": None,
                        "solve_started": probe.solve_started,
                        "fresh_validation_performed": False,
                        "adapter_disposed": probe.adapter_disposed,
                        "qualification_outcome": None,
                    }
                stage = "post_action_revalidation"
                _revalidate_action_inputs(plan, action)
                revalidate_environment()
                checkpoint_path, checkpoint_sha, checkpoint_file_sha = (
                    _write_checkpoint(
                        checkpoints_root,
                        identity=identity,
                        result=result,
                    )
                )
            except BaseException as error:
                failure = _infrastructure_payload(
                    stage=stage, action=action, error=error
                )
                failed_result = {
                    "status": "infrastructure_failure",
                    "completed_at_utc": hooks.now_utc(),
                    "infrastructure_error": failure,
                    "model_audit": None,
                    "constraint_signature": None,
                    "signature_comparison": None,
                    "reuse_case_audit": pool.case_audit_payload(
                        str(action["smoke_case_id"])
                    ),
                    "reuse_pool_audit_snapshot": pool.to_payload(),
                    "reuse_candidate_selected": False,
                    "reuse_start_installation_attempted": False,
                    "reuse_start_installed": False,
                    "selected_source": None,
                    "solve_started": None,
                    "fresh_validation_performed": False,
                    "adapter_disposed": None,
                    "qualification_outcome": None,
                }
                checkpoint_path, checkpoint_sha, checkpoint_file_sha = (
                    _write_checkpoint(
                        checkpoints_root,
                        identity=identity,
                        result=failed_result,
                    )
                )
                checkpoint_paths.append(checkpoint_path)
                checkpoint_records.append(
                    {
                        "action_index": action["action_index"],
                        "role": action["role"],
                        "path": str(checkpoint_path),
                        "checkpoint_sha256": checkpoint_sha,
                        "file_sha256": checkpoint_file_sha,
                        "status": "infrastructure_failure",
                    }
                )
                summary_path = _write_summary(
                    results_root,
                    run_id=run_id,
                    run_sha=run_sha,
                    plan=plan,
                    conclusion="infrastructure_failed",
                    checkpoint_records=checkpoint_records,
                    pool=pool,
                    positive_comparison=positive_comparison,
                    negative_comparison=negative_comparison,
                    completed_at_utc=hooks.now_utc(),
                    infrastructure_failure=failure,
                )
                if isinstance(error, (MemoryError, KeyboardInterrupt, SystemExit)):
                    raise
                raise H23IncumbentReuseSmokeInfrastructureError(
                    f"smoke failed during {stage}; artifacts: {results_root}"
                ) from error
            checkpoint_paths.append(checkpoint_path)
            checkpoint_records.append(
                {
                    "action_index": action["action_index"],
                    "role": action["role"],
                    "path": str(checkpoint_path),
                    "checkpoint_sha256": checkpoint_sha,
                    "file_sha256": checkpoint_file_sha,
                    "status": "completed",
                }
            )
            if action["role"] == "accepted_source_rerun" and not outcome.accepted:
                summary_path = _write_summary(
                    results_root,
                    run_id=run_id,
                    run_sha=run_sha,
                    plan=plan,
                    conclusion="source_unavailable",
                    checkpoint_records=checkpoint_records,
                    pool=pool,
                    positive_comparison=None,
                    negative_comparison=None,
                    completed_at_utc=hooks.now_utc(),
                )
                return H23IncumbentReuseSmokeRunResult(
                    results_root=results_root,
                    run_identity_path=run_identity_path,
                    summary_path=summary_path,
                    smoke_run_id=run_id,
                    conclusion="source_unavailable",
                    checkpoint_paths=tuple(checkpoint_paths),
                )

        conclusion = (
            "passed" if target_accepted else "transport_passed_end_to_end_failed"
        )
        summary_path = _write_summary(
            results_root,
            run_id=run_id,
            run_sha=run_sha,
            plan=plan,
            conclusion=conclusion,
            checkpoint_records=checkpoint_records,
            pool=pool,
            positive_comparison=positive_comparison,
            negative_comparison=negative_comparison,
            completed_at_utc=hooks.now_utc(),
        )
        return H23IncumbentReuseSmokeRunResult(
            results_root=results_root,
            run_identity_path=run_identity_path,
            summary_path=summary_path,
            smoke_run_id=run_id,
            conclusion=conclusion,
            checkpoint_paths=tuple(checkpoint_paths),
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.benchmark.branch_cut.h23_incumbent_reuse_smoke",
        description=(
            "Build or run the development-only H23 incumbent-reuse smoke. "
            "The plan command is solver-free."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    plan = subparsers.add_parser("plan")
    plan.add_argument("--source-h23-plan", required=True)
    plan.add_argument("--source-h23-summary", required=True)
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
            result = build_h23_incumbent_reuse_smoke_plan(
                H23IncumbentReuseSmokePlanBuildConfig(
                    source_h23_plan_path=Path(args.source_h23_plan),
                    source_h23_summary_path=Path(args.source_h23_summary),
                    output_path=Path(args.output),
                    expected_git_commit=args.expected_git_commit,
                )
            )
        else:
            result = run_h23_incumbent_reuse_smoke(
                H23IncumbentReuseSmokeRunConfig(
                    plan_path=Path(args.plan),
                    results_root=Path(args.results_root),
                    resource_lock=Path(args.resource_lock),
                )
            )
    except (
        H23IncumbentReuseSmokeError,
        H23IncumbentReuseSmokeInfrastructureError,
        OSError,
        ValueError,
    ) as error:
        print(f"H23 incumbent-reuse smoke failed: {error}", file=sys.stderr)
        return 2
    print(canonical_json(result.to_payload()))
    return 0


__all__ = [
    "H23IncumbentReuseSmokeError",
    "H23IncumbentReuseSmokeHooks",
    "H23IncumbentReuseSmokeInfrastructureError",
    "H23IncumbentReuseSmokePlanBuildConfig",
    "H23IncumbentReuseSmokePlanBuildHooks",
    "H23IncumbentReuseSmokePlanBuildResult",
    "H23IncumbentReuseSmokeRunConfig",
    "H23IncumbentReuseSmokeRunResult",
    "H23_REUSE_SMOKE_ACTION_INDICES",
    "H23_REUSE_SMOKE_PLAN_ID",
    "SignatureOnlyProbeOutcome",
    "build_h23_incumbent_reuse_smoke_plan",
    "build_parser",
    "build_signature_comparison",
    "load_h23_incumbent_reuse_smoke_plan",
    "main",
    "probe_constraint_signature_only",
    "run_h23_incumbent_reuse_smoke",
]


if __name__ == "__main__":
    raise SystemExit(main())
