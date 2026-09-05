"""Crash-auditable journal and scheduler for an immutable formal plan.

The module contains no Gurobi dependency.  A caller injects the operation that
fills one already-claimed attempt directory.  This layer owns attempt-1 claims,
bundle validation, immutable slot checkpoints, resume reconciliation, and the
scale/block concurrency contract.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
import uuid
from collections.abc import Callable, Mapping
from concurrent.futures import (
    FIRST_COMPLETED,
    Executor,
    Future,
    ThreadPoolExecutor,
    wait,
)
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from types import MappingProxyType
from typing import Any, TypeAlias, cast

from .artifacts import (
    AttemptClaim,
    attempt_directory,
    claim_attempt_directory,
    write_canonical_json_exclusive,
)
from .execution import build_core_solver_config
from .formal_environment import (
    FormalEnvironmentContract,
    FormalEnvironmentError,
    SessionSnapshot,
)
from .formal_metrics import (
    FormalMetricValidationError,
    validate_formal_attempt_paper_metrics,
)
from .formal_plan import (
    FORMAL_ATTEMPT_NUMBER,
    FORMAL_SCALE_CONTRACTS,
    FORMAL_SLOT_COUNT,
    FormalBlock,
    FormalExecutionPlan,
    FormalSlot,
)
from .formal_solution_evidence import (
    FormalQualificationTrace,
    bind_qualification_trace,
    validate_solution_structure_payload,
)
from .hashing import canonical_json, canonical_sha256
from .schema import BenchmarkMode, PairSpec, ProfileConfig, TelemetryConfig
from .telemetry import (
    APPLIED_GUROBI_PARAMETER_ATTRIBUTES,
    BENCHMARK_OPERATIONAL_GUROBI_PARAMETERS,
    CORE_SOLVER_ALLOWED_GUROBI_PARAMETERS,
    GUROBI_PARAMETER_AUDIT_SCHEMA_VERSION,
    GUROBI_RUNTIME_CONTEXT_PARAMETERS,
)


FORMAL_CAMPAIGN_SCHEMA_VERSION = 3
FORMAL_CAMPAIGN_ARTIFACT_TYPE = "mlnd_branch_cut_formal_campaign_identity"
FORMAL_SLOT_CHECKPOINT_SCHEMA_VERSION = 4
FORMAL_SLOT_CHECKPOINT_ARTIFACT_TYPE = "mlnd_branch_cut_formal_slot_checkpoint"
FORMAL_ATTEMPT_ATTESTATION_SCHEMA_VERSION = 3
FORMAL_ATTEMPT_ATTESTATION_ARTIFACT_TYPE = "mlnd_branch_cut_formal_attempt_attestation"
FORMAL_SESSION_SCHEMA_VERSION = 1
FORMAL_SESSION_START_ARTIFACT_TYPE = "mlnd_branch_cut_formal_session_start"
FORMAL_SESSION_END_ARTIFACT_TYPE = "mlnd_branch_cut_formal_session_end"
FORMAL_CAMPAIGN_EVIDENCE_SCHEMA_VERSION = 1
FORMAL_CAMPAIGN_EVIDENCE_ARTIFACT_TYPE = (
    "mlnd_branch_cut_formal_campaign_evidence_seal"
)
FORMAL_CAMPAIGN_EVIDENCE_HASH_DOMAIN = (
    "mlnd-branch-cut-formal-campaign-evidence-v1"
)
FORMAL_SLOT_BUNDLE_HASH_DOMAIN = "mlnd-branch-cut-formal-slot-bundle-v1"
FORMAL_CAMPAIGN_EVIDENCE_FILENAME = "campaign_evidence_seal.json"

_PLAN_ID_RE = re.compile(r"^formal-plan-[0-9a-f]{64}$")
_RUN_ID_RE = re.compile(r"^run-[0-9a-f]{64}$")
_ATTEMPT_ID_RE = re.compile(r"^run-[0-9a-f]{64}-a[0-9]{4}$")
_CHECKPOINT_NAME_RE = re.compile(r"^slot-([0-9]{4})\.json$")
_CAMPAIGN_ID_RE = re.compile(r"^formal-campaign-[0-9a-f]{64}$")
_SESSION_ID_RE = re.compile(r"^formal-session-[0-9a-f]{64}$")
_EVIDENCE_ID_RE = re.compile(r"^formal-evidence-[0-9a-f]{64}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

_REQUIRED_ATTEMPT_FILES = frozenset(
    {
        "claim.json",
        "formal_claim.json",
        "solver.log",
        "run_spec.json",
        "run_record.json",
        "progress.jsonl",
        "root_rounds.jsonl",
        "cut_stats.json",
    }
)
_OPTIONAL_ATTEMPT_FILES = frozenset({"failure.json"})
_ALLOWED_CAMPAIGN_ENTRIES = frozenset(
    {
        "campaign_identity.json",
        FORMAL_CAMPAIGN_EVIDENCE_FILENAME,
        "attempts",
        "journal",
        "sessions",
    }
)
_FORMAL_CAMPAIGN_FACTORY_SEAL = object()

_APPLIED_STABLE_PARAMETER_TO_CORE_FIELD = {
    "cuts": "cuts",
    "feasibility_tol": "feasibility_tol",
    "heuristics": "heuristics",
    "integer_feasibility_tol": "integer_feasibility_tol",
    "method": "root_relaxation_method",
    "mip_focus": "mip_focus",
    "mip_gap": "mip_gap",
    "miqcp_method": "miqcp_method",
    "no_rel_heur_time": "no_rel_heur_time",
    "numeric_focus": "numeric_focus",
    "optimality_tol": "optimality_tol",
    "output_flag": "output_flag",
    "presolve": "presolve",
    "root_cut_passes": "root_cut_passes",
    "seed": "seed",
    "threads": "threads",
}
_APPLIED_PARAMETER_NAMES = frozenset(
    {"time_limit", "pre_crush", *_APPLIED_STABLE_PARAMETER_TO_CORE_FIELD}
)
_GUROBI_CANONICAL_TO_APPLIED_PARAMETER = {
    gurobi_name: applied_name
    for applied_name, gurobi_name in APPLIED_GUROBI_PARAMETER_ATTRIBUTES
}
if (
    frozenset(_GUROBI_CANONICAL_TO_APPLIED_PARAMETER)
    != CORE_SOLVER_ALLOWED_GUROBI_PARAMETERS
):
    raise RuntimeError(
        "formal applied-parameter catalog differs from the telemetry allowlist"
    )

_GUROBI_AUDIT_METADATA_FIELDS = frozenset(
    {
        "schema_version",
        "enumeration_complete",
        "reason_code",
        "enumerated_parameter_count",
        "parameter_catalog_sha256",
    }
)
_GUROBI_EFFECTIVE_ENTRY_FIELDS = frozenset(
    {"classification", "current", "default", "value_type"}
)
_GUROBI_EXECUTION_CONTEXT_FIELDS = frozenset(
    {
        "execution_mode",
        "observation_basis",
        "remote_identity_sha256",
        "remote_mechanisms",
    }
)
_GUROBI_AUDIT_REASON_RE = re.compile(r"^[a-z][a-z0-9_]*$")


class FormalExecutionError(RuntimeError):
    """Formal campaign state is unsafe, inconsistent, or incomplete."""


def _path_exists_lstat(path: Path, *, label: str) -> bool:
    """Test for a directory entry without following a final symlink."""

    try:
        path.lstat()
    except FileNotFoundError:
        return False
    except OSError as error:
        raise FormalExecutionError(f"cannot inspect {label} {path}: {error}") from error
    return True


def _require_real_directory(path: Path, *, label: str) -> None:
    """Require one existing directory entry to be a real directory."""

    try:
        metadata = path.lstat()
    except OSError as error:
        raise FormalExecutionError(f"cannot inspect {label} {path}: {error}") from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise FormalExecutionError(
            f"{label} must be a real non-symlink directory: {path}"
        )


def _require_direct_child_directory(
    path: Path,
    parent: Path,
    *,
    label: str,
    parent_label: str,
) -> None:
    """Require a real directory to remain a direct child of a real parent.

    The lexical check protects the frozen layout, while the resolved check
    ensures that an intervening symlink cannot redirect the child outside its
    declared parent.
    """

    _require_real_directory(parent, label=parent_label)
    _require_real_directory(path, label=label)
    if path.parent != parent:
        raise FormalExecutionError(
            f"{label} is not a direct child of {parent_label}: {path}"
        )
    try:
        resolved_parent = parent.resolve(strict=True)
        resolved_path = path.resolve(strict=True)
    except OSError as error:
        raise FormalExecutionError(
            f"cannot resolve {label} containment for {path}: {error}"
        ) from error
    if resolved_path.parent != resolved_parent:
        raise FormalExecutionError(f"{label} escapes {parent_label}: {path}")


def _json_ready(value: object) -> Any:
    return json.loads(canonical_json(value))


def _read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    snapshot = _read_regular_file_once(path, label=label)
    return _json_object_from_snapshot(snapshot, label=label, path=path)


@dataclass(frozen=True, slots=True)
class _ImmutableFileSnapshot:
    """One regular file read exactly once into immutable bytes."""

    name: str
    payload: bytes = field(repr=False)
    sha256: str


def _read_regular_file_once(path: Path, *, label: str) -> _ImmutableFileSnapshot:
    """Open a non-symlink regular file once and retain the exact bytes read.

    Semantic validation and hashing must both consume ``payload``.  In
    particular, callers must never reopen ``path`` to hash data that may differ
    from the bytes they parsed.
    """

    try:
        before = path.lstat()
    except OSError as error:
        raise FormalExecutionError(f"cannot inspect {label} {path}: {error}") from error
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise FormalExecutionError(f"{label} must be a regular non-symlink: {path}")

    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = -1
    try:
        descriptor = os.open(path, flags)
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode):
            raise FormalExecutionError(f"{label} is not a regular file: {path}")
        if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
            raise FormalExecutionError(f"{label} changed while it was opened: {path}")
        with os.fdopen(descriptor, "rb") as handle:
            descriptor = -1
            payload = handle.read()
            after = os.fstat(handle.fileno())
        if (
            after.st_size != opened.st_size
            or after.st_mtime_ns != opened.st_mtime_ns
            or len(payload) != after.st_size
        ):
            raise FormalExecutionError(f"{label} changed while it was read: {path}")
    except OSError as error:
        raise FormalExecutionError(f"cannot read {label} {path}: {error}") from error
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    return _ImmutableFileSnapshot(
        name=path.name,
        payload=payload,
        sha256=hashlib.sha256(payload).hexdigest(),
    )


def _json_object_from_snapshot(
    snapshot: _ImmutableFileSnapshot,
    *,
    label: str,
    path: Path,
) -> dict[str, Any]:
    try:
        payload = json.loads(snapshot.payload.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise FormalExecutionError(f"cannot parse {label} {path}: {error}") from error
    if not isinstance(payload, dict):
        raise FormalExecutionError(f"{label} must contain one JSON object: {path}")
    return payload


def _jsonl_objects_from_snapshot(
    snapshot: _ImmutableFileSnapshot,
    *,
    label: str,
    path: Path,
) -> tuple[dict[str, Any], ...]:
    try:
        lines = snapshot.payload.decode("utf-8").splitlines()
    except UnicodeError as error:
        raise FormalExecutionError(f"cannot decode {label} {path}: {error}") from error
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as error:
            raise FormalExecutionError(
                f"{label} has invalid JSON on line {line_number}: {path}"
            ) from error
        if not isinstance(payload, dict):
            raise FormalExecutionError(
                f"{label} line {line_number} is not an object: {path}"
            )
        rows.append(payload)
    return tuple(rows)


def _freeze_json(value: Any) -> Any:
    """Recursively freeze an already decoded JSON value."""

    if isinstance(value, Mapping):
        return MappingProxyType(
            {key: _freeze_json(item) for key, item in sorted(value.items())}
        )
    if isinstance(value, list | tuple):
        return tuple(_freeze_json(item) for item in value)
    return value


def _freeze_json_object(value: Mapping[str, Any]) -> Mapping[str, Any]:
    return cast(Mapping[str, Any], _freeze_json(value))


@dataclass(frozen=True, slots=True)
class FormalCampaignIdentity:
    """Path-independent scientific identity for one plan on one stable host."""

    formal_plan_id: str
    formal_plan_sha256: str
    benchmark_manifest_sha256: str
    manifest_artifact_sha256: str
    manifest_id: str
    model_id: str
    formal_execution_implementation_sha256: str
    qualification_mode: BenchmarkMode
    gurobi_version: str
    environment_contract: FormalEnvironmentContract
    environment_contract_sha256: str
    host_identity_sha256: str
    attempt_number: int = FORMAL_ATTEMPT_NUMBER
    schema_version: int = FORMAL_CAMPAIGN_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if (
            not isinstance(self.formal_plan_id, str)
            or _PLAN_ID_RE.fullmatch(self.formal_plan_id) is None
        ):
            raise ValueError("formal_plan_id is not canonical")
        for name in (
            "formal_plan_sha256",
            "benchmark_manifest_sha256",
            "manifest_artifact_sha256",
            "formal_execution_implementation_sha256",
            "environment_contract_sha256",
            "host_identity_sha256",
        ):
            value = getattr(self, name)
            if (
                not isinstance(value, str)
                or re.fullmatch(r"[0-9a-f]{64}", value) is None
            ):
                raise ValueError(f"{name} must be a lowercase SHA256 digest")
        for name in (
            "manifest_id",
            "model_id",
            "gurobi_version",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a nonempty string")
        try:
            mode = BenchmarkMode(self.qualification_mode)
        except (TypeError, ValueError) as error:
            raise ValueError("qualification_mode is invalid") from error
        object.__setattr__(self, "qualification_mode", mode)
        if not isinstance(self.environment_contract, FormalEnvironmentContract):
            raise ValueError("environment_contract is invalid")
        if self.environment_contract.sha256 != self.environment_contract_sha256:
            raise ValueError(
                "environment_contract_sha256 differs from environment_contract"
            )
        if self.formal_plan_id != f"formal-plan-{self.formal_plan_sha256}":
            raise ValueError("formal_plan_id and formal_plan_sha256 differ")
        if self.attempt_number != FORMAL_ATTEMPT_NUMBER:
            raise ValueError("formal campaign requires attempt_number=1")
        if self.schema_version != FORMAL_CAMPAIGN_SCHEMA_VERSION:
            raise ValueError("unsupported formal-campaign schema_version")

    def to_payload(self) -> dict[str, Any]:
        return _json_ready(self)

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.to_payload())

    @property
    def campaign_id(self) -> str:
        return f"formal-campaign-{self.sha256}"

    def artifact_payload(self) -> dict[str, Any]:
        return {
            "artifact_schema_version": FORMAL_CAMPAIGN_SCHEMA_VERSION,
            "artifact_type": FORMAL_CAMPAIGN_ARTIFACT_TYPE,
            "campaign_id": self.campaign_id,
            "campaign_identity": self.to_payload(),
            "campaign_identity_sha256": self.sha256,
        }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _session_nonce() -> str:
    return uuid.uuid4().hex


def _parse_utc(value: str, *, field_name: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a nonempty UTC timestamp")
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as error:
        raise ValueError(f"{field_name} must be an ISO-8601 timestamp") from error
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise ValueError(f"{field_name} must explicitly use UTC")
    return parsed


@dataclass(frozen=True, slots=True)
class FormalRunSession:
    """One immutable process/session start bound to a formal campaign."""

    campaign_id: str
    formal_plan_id: str
    environment_contract_sha256: str
    host_identity_sha256: str
    started_at_utc: str
    session_nonce: str
    session_snapshot: SessionSnapshot
    resource_lock_path: str
    schema_version: int = FORMAL_SESSION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if _CAMPAIGN_ID_RE.fullmatch(self.campaign_id) is None:
            raise ValueError("campaign_id is not canonical")
        if _PLAN_ID_RE.fullmatch(self.formal_plan_id) is None:
            raise ValueError("formal_plan_id is not canonical")
        for name in ("environment_contract_sha256", "host_identity_sha256"):
            if re.fullmatch(r"[0-9a-f]{64}", getattr(self, name)) is None:
                raise ValueError(f"{name} must be a lowercase SHA256 digest")
        _parse_utc(self.started_at_utc, field_name="started_at_utc")
        if not isinstance(self.session_nonce, str) or not self.session_nonce.strip():
            raise ValueError("session_nonce must be a nonempty string")
        if not isinstance(self.session_snapshot, SessionSnapshot):
            raise ValueError("session_snapshot is invalid")
        if (
            not isinstance(self.resource_lock_path, str)
            or not self.resource_lock_path.strip()
        ):
            raise ValueError("resource_lock_path must be a nonempty string")
        if self.schema_version != FORMAL_SESSION_SCHEMA_VERSION:
            raise ValueError("unsupported formal-session schema_version")

    def to_payload(self) -> dict[str, Any]:
        return {
            "campaign_id": self.campaign_id,
            "environment_contract_sha256": self.environment_contract_sha256,
            "formal_plan_id": self.formal_plan_id,
            "host_identity_sha256": self.host_identity_sha256,
            "resource_lock_path": self.resource_lock_path,
            "schema_version": self.schema_version,
            "session_nonce": self.session_nonce,
            "session_snapshot": self.session_snapshot.to_payload(),
            "started_at_utc": self.started_at_utc,
        }

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.to_payload())

    @property
    def session_id(self) -> str:
        return f"formal-session-{self.sha256}"

    def start_artifact_payload(self) -> dict[str, Any]:
        return {
            "artifact_schema_version": FORMAL_SESSION_SCHEMA_VERSION,
            "artifact_type": FORMAL_SESSION_START_ARTIFACT_TYPE,
            "session": self.to_payload(),
            "session_id": self.session_id,
            "session_sha256": self.sha256,
        }


@dataclass(frozen=True, slots=True)
class FormalCampaign:
    """Resolved paths and verified identity for one formal campaign."""

    plan: FormalExecutionPlan
    identity: FormalCampaignIdentity
    results_root: Path
    plan_root: Path
    root: Path
    attempts_root: Path
    journal_root: Path
    sessions_root: Path
    identity_path: Path
    evidence_seal_path: Path
    session: FormalRunSession
    session_root: Path
    session_start_path: Path
    session_end_path: Path
    current_binding_provider: CurrentBindingProvider | None = field(
        default=None,
        init=False,
        repr=False,
        compare=False,
    )
    pre_claim_preparer: PreClaimPreparer | None = field(
        default=None,
        init=False,
        repr=False,
        compare=False,
    )
    _claim_lock: Lock = field(
        default_factory=Lock,
        init=False,
        repr=False,
        compare=False,
    )
    _factory_seal: object | None = field(
        default=None,
        init=False,
        repr=False,
        compare=False,
    )


@dataclass(frozen=True, slots=True)
class AuthenticatedFormalSlotSnapshot:
    """Immutable, analyzer-ready values parsed from authenticated file bytes."""

    slot_id: str
    slot_index: int
    block_id: str
    logical_run_id: str
    session_id: str
    outcome: str
    completed_without_exception: bool
    has_incumbent: bool
    run_spec: Mapping[str, Any]
    run_record: Mapping[str, Any]
    progress: tuple[Mapping[str, Any], ...]
    root_rounds: tuple[Mapping[str, Any], ...]
    cut_stats: Mapping[str, Any]
    file_sha256: tuple[tuple[str, str], ...]
    attempt_bundle_sha256: str
    checkpoint_bytes_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.slot_id, str) or not self.slot_id.strip():
            raise ValueError("slot_id must be a nonempty string")
        if isinstance(self.slot_index, bool) or not isinstance(self.slot_index, int):
            raise ValueError("slot_index must be an integer")
        if self.slot_index < 0:
            raise ValueError("slot_index must be nonnegative")
        for name in ("block_id", "logical_run_id", "session_id", "outcome"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a nonempty string")
        if type(self.completed_without_exception) is not bool:
            raise ValueError("completed_without_exception must be boolean")
        if type(self.has_incumbent) is not bool:
            raise ValueError("has_incumbent must be boolean")
        for name in ("run_spec", "run_record", "cut_stats"):
            if not isinstance(getattr(self, name), Mapping):
                raise ValueError(f"{name} must be a mapping")
        for rows_name in ("progress", "root_rounds"):
            rows = getattr(self, rows_name)
            if not isinstance(rows, tuple) or any(
                not isinstance(row, Mapping) for row in rows
            ):
                raise ValueError(f"{rows_name} must be a tuple of mappings")
        names = tuple(name for name, _digest in self.file_sha256)
        if names != tuple(sorted(set(names))):
            raise ValueError("file_sha256 names must be sorted and unique")
        for _name, digest in self.file_sha256:
            if _SHA256_RE.fullmatch(digest) is None:
                raise ValueError("file_sha256 contains an invalid digest")
        for name in ("attempt_bundle_sha256", "checkpoint_bytes_sha256"):
            if _SHA256_RE.fullmatch(getattr(self, name)) is None:
                raise ValueError(f"{name} must be a lowercase SHA256 digest")


@dataclass(frozen=True, slots=True)
class FormalAttemptBundle:
    """Validated immutable evidence produced by one formal attempt."""

    directory: Path
    outcome: str
    completed_without_exception: bool
    has_incumbent: bool
    session_id: str
    prepared_input_attestation: FormalInputAttestation
    prepared_input_attestation_sha256: str
    file_sha256: tuple[tuple[str, str], ...]
    attempt_bundle_sha256: str
    run_spec: Mapping[str, Any]
    run_record: Mapping[str, Any]
    progress: tuple[Mapping[str, Any], ...]
    root_rounds: tuple[Mapping[str, Any], ...]
    cut_stats: Mapping[str, Any]

    def authenticated_snapshot(
        self,
        block: FormalBlock,
        slot: FormalSlot,
        *,
        checkpoint_bytes_sha256: str,
    ) -> AuthenticatedFormalSlotSnapshot:
        return AuthenticatedFormalSlotSnapshot(
            slot_id=slot.slot_id,
            slot_index=slot.slot_index,
            block_id=block.block_id,
            logical_run_id=slot.logical_run_id,
            session_id=self.session_id,
            outcome=self.outcome,
            completed_without_exception=self.completed_without_exception,
            has_incumbent=self.has_incumbent,
            run_spec=self.run_spec,
            run_record=self.run_record,
            progress=self.progress,
            root_rounds=self.root_rounds,
            cut_stats=self.cut_stats,
            file_sha256=self.file_sha256,
            attempt_bundle_sha256=self.attempt_bundle_sha256,
            checkpoint_bytes_sha256=checkpoint_bytes_sha256,
        )


@dataclass(frozen=True, slots=True)
class FormalCampaignState:
    """A reconciled slot-level snapshot."""

    completed_slot_ids: tuple[str, ...]
    pending_slot_ids: tuple[str, ...]
    recovered_slot_ids: tuple[str, ...]
    authenticated_slots: tuple[AuthenticatedFormalSlotSnapshot, ...] = ()
    evidence_id: str | None = None
    evidence_root_sha256: str | None = None

    @property
    def complete(self) -> bool:
        return (
            len(self.completed_slot_ids) == FORMAL_SLOT_COUNT
            and not self.pending_slot_ids
        )


@dataclass(frozen=True, slots=True)
class FormalSlotTask:
    """Solver-free task passed to an injected slot executor."""

    campaign_id: str
    campaign_identity_sha256: str
    benchmark_manifest_sha256: str
    manifest_artifact_sha256: str
    formal_plan_id: str
    formal_plan_sha256: str
    formal_execution_implementation_sha256: str
    model_id: str
    gurobi_version: str
    session_id: str
    block: FormalBlock
    slot: FormalSlot
    pair: PairSpec
    profile: ProfileConfig
    telemetry: TelemetryConfig

    @property
    def formal_context(self) -> dict[str, Any]:
        """Exact execution identity persisted in every formal result artifact."""

        return {
            "block_id": self.block.block_id,
            "block_index": self.block.block_index,
            "campaign_id": self.campaign_id,
            "campaign_identity_sha256": self.campaign_identity_sha256,
            "formal_execution_implementation_sha256": (
                self.formal_execution_implementation_sha256
            ),
            "formal_plan_id": self.formal_plan_id,
            "formal_plan_sha256": self.formal_plan_sha256,
            "slot_id": self.slot.slot_id,
            "slot_index": self.slot.slot_index,
            "session_id": self.session_id,
        }


@dataclass(frozen=True, slots=True)
class FormalInputAttestation:
    """Content proof for the already-loaded input used by one formal slot."""

    case_id: str
    slot_id: str
    expected_instance_sha256: str
    payload_instance_sha256: str
    source_sha256_before: str
    source_sha256_after: str
    qualification_trace: FormalQualificationTrace

    def __post_init__(self) -> None:
        if not isinstance(self.case_id, str) or not self.case_id.strip():
            raise ValueError("case_id must be a nonempty string")
        if not isinstance(self.slot_id, str) or not self.slot_id.strip():
            raise ValueError("slot_id must be a nonempty string")
        for name in (
            "expected_instance_sha256",
            "payload_instance_sha256",
            "source_sha256_before",
            "source_sha256_after",
        ):
            value = getattr(self, name)
            if (
                not isinstance(value, str)
                or re.fullmatch(r"[0-9a-f]{64}", value) is None
            ):
                raise ValueError(f"{name} must be a lowercase SHA256 digest")
        if not isinstance(self.qualification_trace, FormalQualificationTrace):
            raise ValueError(
                "qualification_trace must be a FormalQualificationTrace"
            )

    def to_payload(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "slot_id": self.slot_id,
            "expected_instance_sha256": self.expected_instance_sha256,
            "payload_instance_sha256": self.payload_instance_sha256,
            "source_sha256_before": self.source_sha256_before,
            "source_sha256_after": self.source_sha256_after,
            "qualification_trace": self.qualification_trace.to_payload(),
        }

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.to_payload())

    @classmethod
    def from_payload(cls, payload: object) -> FormalInputAttestation:
        if not isinstance(payload, Mapping):
            raise ValueError("prepared input attestation must be an object")
        expected_fields = {
            "case_id",
            "slot_id",
            "expected_instance_sha256",
            "payload_instance_sha256",
            "source_sha256_before",
            "source_sha256_after",
            "qualification_trace",
        }
        if set(payload) != expected_fields:
            raise ValueError("prepared input attestation fields are not canonical")
        values = {name: payload[name] for name in expected_fields}
        values["qualification_trace"] = FormalQualificationTrace.from_payload(
            values["qualification_trace"]
        )
        return cls(**values)


@dataclass(frozen=True, slots=True)
class PreparedFormalInput:
    """Exclusively owned loaded input prepared before the attempt-1 claim."""

    payload: object = field(repr=False, compare=False)
    attestation: FormalInputAttestation

    def __post_init__(self) -> None:
        if not isinstance(self.attestation, FormalInputAttestation):
            raise ValueError("attestation must be a FormalInputAttestation")


@dataclass(frozen=True, slots=True)
class FormalBlockExecutionResult:
    block_id: str
    executed_slot_ids: tuple[str, ...]
    skipped_slot_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class FormalCampaignRunResult:
    campaign_id: str
    session_id: str
    initial_completed_count: int
    recovered_count: int
    executed_count: int
    skipped_count: int
    final_completed_count: int
    evidence_id: str | None = None
    evidence_root_sha256: str | None = None

    def __post_init__(self) -> None:
        if self.final_completed_count == FORMAL_SLOT_COUNT:
            if (
                not isinstance(self.evidence_id, str)
                or _EVIDENCE_ID_RE.fullmatch(self.evidence_id) is None
            ):
                raise ValueError("a complete formal run requires a canonical evidence_id")
            if (
                not isinstance(self.evidence_root_sha256, str)
                or _SHA256_RE.fullmatch(self.evidence_root_sha256) is None
            ):
                raise ValueError(
                    "a complete formal run requires evidence_root_sha256"
                )


SlotExecutionCallback: TypeAlias = Callable[
    [FormalSlotTask, AttemptClaim, PreparedFormalInput], object
]
BlockCallback: TypeAlias = Callable[[FormalBlock], object]
ExecutorFactory: TypeAlias = Callable[..., Executor]
PreClaimPreparer: TypeAlias = Callable[[FormalSlotTask], PreparedFormalInput]
TimestampProvider: TypeAlias = Callable[[], str]
SessionNonceProvider: TypeAlias = Callable[[], str]


@dataclass(frozen=True, slots=True)
class FormalExecutionBinding:
    """Current manifest/model/solver identity supplied by the outer CLI layer."""

    benchmark_manifest_sha256: str
    manifest_artifact_sha256: str
    manifest_id: str
    model_id: str
    formal_execution_implementation_sha256: str
    qualification_mode: BenchmarkMode
    gurobi_version: str
    environment_contract: FormalEnvironmentContract
    host_identity_sha256: str

    def __post_init__(self) -> None:
        for name in (
            "benchmark_manifest_sha256",
            "manifest_artifact_sha256",
            "formal_execution_implementation_sha256",
            "host_identity_sha256",
        ):
            value = getattr(self, name)
            if (
                not isinstance(value, str)
                or re.fullmatch(r"[0-9a-f]{64}", value) is None
            ):
                raise ValueError(f"{name} must be a lowercase SHA256 digest")
        if (
            not isinstance(self.manifest_id, str)
            or re.fullmatch(r"manifest-[0-9a-f]{64}", self.manifest_id) is None
        ):
            raise ValueError("manifest_id is not canonical")
        if self.manifest_id != f"manifest-{self.manifest_artifact_sha256}":
            raise ValueError("manifest_id and manifest_artifact_sha256 differ")
        if (
            not isinstance(self.model_id, str)
            or re.fullmatch(r"model-[0-9a-f]{64}", self.model_id) is None
        ):
            raise ValueError("model_id is not canonical")
        if not isinstance(self.gurobi_version, str) or not self.gurobi_version.strip():
            raise ValueError("gurobi_version must be a nonempty string")
        if not isinstance(self.environment_contract, FormalEnvironmentContract):
            raise ValueError("environment_contract is invalid")
        try:
            mode = BenchmarkMode(self.qualification_mode)
        except (TypeError, ValueError) as error:
            raise ValueError("qualification_mode is invalid") from error
        object.__setattr__(self, "qualification_mode", mode)

    def validate_plan(self, plan: FormalExecutionPlan) -> None:
        expected = {
            "benchmark_manifest_sha256": plan.benchmark_manifest_sha256,
            "manifest_artifact_sha256": plan.manifest_artifact_sha256,
            "manifest_id": plan.manifest_id,
            "model_id": plan.model_id,
            "formal_execution_implementation_sha256": (
                plan.formal_execution_implementation_sha256
            ),
            "qualification_mode": plan.qualification_mode,
            "gurobi_version": plan.gurobi_version,
            "environment_contract": plan.environment_contract,
        }
        observed = {
            "benchmark_manifest_sha256": self.benchmark_manifest_sha256,
            "manifest_artifact_sha256": self.manifest_artifact_sha256,
            "manifest_id": self.manifest_id,
            "model_id": self.model_id,
            "formal_execution_implementation_sha256": (
                self.formal_execution_implementation_sha256
            ),
            "qualification_mode": self.qualification_mode,
            "gurobi_version": self.gurobi_version,
            "environment_contract": self.environment_contract,
        }
        differing = sorted(
            name for name in expected if observed[name] != expected[name]
        )
        if differing:
            raise FormalExecutionError(
                "formal execution identity differs from the frozen plan: "
                + ", ".join(differing)
            )

    def to_payload(self) -> dict[str, Any]:
        return _json_ready(self)

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.to_payload())


CurrentBindingProvider: TypeAlias = Callable[[], FormalExecutionBinding]
# A production provider must independently re-read and recompute every field on
# every call and must be thread-safe.  A cached binding is suitable only for
# solver-free tests; this layer deliberately calls the provider at each claim.


def _validate_current_binding(
    plan: FormalExecutionPlan,
    provider: CurrentBindingProvider,
    campaign_identity: FormalCampaignIdentity | None = None,
) -> FormalExecutionBinding:
    if not callable(provider):
        raise FormalExecutionError("current_binding_provider must be callable")
    try:
        binding = provider()
    except Exception as error:
        raise FormalExecutionError(
            "cannot observe the current formal execution binding: "
            f"{type(error).__name__}: {error}"
        ) from error
    if not isinstance(binding, FormalExecutionBinding):
        raise FormalExecutionError(
            "current_binding_provider must return a FormalExecutionBinding"
        )
    binding.validate_plan(plan)
    if campaign_identity is not None:
        differing: list[str] = []
        if binding.host_identity_sha256 != campaign_identity.host_identity_sha256:
            differing.append("host_identity_sha256")
        if (
            binding.environment_contract.sha256
            != campaign_identity.environment_contract_sha256
        ):
            differing.append("environment_contract_sha256")
        if differing:
            raise FormalExecutionError(
                "formal execution identity differs from the campaign host: "
                + ", ".join(differing)
            )
    return binding


def _validate_campaign_handle(campaign: FormalCampaign) -> None:
    if (
        not isinstance(campaign, FormalCampaign)
        or campaign._factory_seal is not _FORMAL_CAMPAIGN_FACTORY_SEAL
        or not callable(campaign.current_binding_provider)
        or not callable(campaign.pre_claim_preparer)
    ):
        raise FormalExecutionError(
            "formal campaign was not issued by prepare_formal_campaign"
        )
    _validate_campaign_directory_layout(campaign)


def _validate_campaign_directory_layout(campaign: FormalCampaign) -> None:
    """Revalidate every frozen formal container without following symlinks."""

    expected_paths = {
        "plan root": campaign.results_root / campaign.plan.plan_id,
        "campaign root": campaign.plan_root / campaign.identity.campaign_id,
        "attempts root": campaign.root / "attempts",
        "journal root": campaign.root / "journal",
        "sessions root": campaign.root / "sessions",
        "session root": campaign.sessions_root / campaign.session.session_id,
    }
    observed_paths = {
        "plan root": campaign.plan_root,
        "campaign root": campaign.root,
        "attempts root": campaign.attempts_root,
        "journal root": campaign.journal_root,
        "sessions root": campaign.sessions_root,
        "session root": campaign.session_root,
    }
    differing = sorted(
        label
        for label, expected in expected_paths.items()
        if observed_paths[label] != expected
    )
    if differing:
        raise FormalExecutionError(
            "formal campaign container paths differ from the frozen layout: "
            + ", ".join(differing)
        )

    _require_real_directory(campaign.results_root, label="results root")
    _require_direct_child_directory(
        campaign.plan_root,
        campaign.results_root,
        label="plan root",
        parent_label="results root",
    )
    _require_direct_child_directory(
        campaign.root,
        campaign.plan_root,
        label="campaign root",
        parent_label="plan root",
    )
    for path, label in (
        (campaign.attempts_root, "attempts root"),
        (campaign.journal_root, "journal root"),
        (campaign.sessions_root, "sessions root"),
    ):
        _require_direct_child_directory(
            path,
            campaign.root,
            label=label,
            parent_label="campaign root",
        )
    _require_direct_child_directory(
        campaign.session_root,
        campaign.sessions_root,
        label="session root",
        parent_label="sessions root",
    )


def _campaign_identity(
    plan: FormalExecutionPlan,
    binding: FormalExecutionBinding,
) -> FormalCampaignIdentity:
    return FormalCampaignIdentity(
        formal_plan_id=plan.plan_id,
        formal_plan_sha256=plan.sha256,
        benchmark_manifest_sha256=plan.benchmark_manifest_sha256,
        manifest_artifact_sha256=plan.manifest_artifact_sha256,
        manifest_id=plan.manifest_id,
        model_id=plan.model_id,
        formal_execution_implementation_sha256=(
            plan.formal_execution_implementation_sha256
        ),
        qualification_mode=plan.qualification_mode,
        gurobi_version=plan.gurobi_version,
        environment_contract=binding.environment_contract,
        environment_contract_sha256=binding.environment_contract.sha256,
        host_identity_sha256=binding.host_identity_sha256,
    )


def _validate_campaign_identity_artifact(
    payload: Mapping[str, Any],
    expected: FormalCampaignIdentity,
) -> None:
    expected_payload = expected.artifact_payload()
    if canonical_json(payload) != canonical_json(expected_payload):
        raise FormalExecutionError(
            "campaign identity differs from the formal plan or stable host/environment"
        )


def _validate_campaign_root_entries(campaign: FormalCampaign) -> None:
    _validate_campaign_directory_layout(campaign)
    actual = {path.name for path in campaign.root.iterdir()}
    unexpected = sorted(actual - _ALLOWED_CAMPAIGN_ENTRIES)
    if unexpected:
        raise FormalExecutionError(
            "formal campaign root contains unknown entries: " + ", ".join(unexpected)
        )
    if _path_exists_lstat(
        campaign.evidence_seal_path,
        label="formal campaign evidence seal",
    ) and (
        not campaign.evidence_seal_path.is_file()
        or campaign.evidence_seal_path.is_symlink()
    ):
        raise FormalExecutionError(
            "formal campaign evidence seal must be a regular non-symlink"
        )


def _session_end_payload(
    session: FormalRunSession,
    *,
    ended_at_utc: str,
    status: str,
) -> dict[str, Any]:
    ended = _parse_utc(ended_at_utc, field_name="ended_at_utc")
    started = _parse_utc(session.started_at_utc, field_name="started_at_utc")
    if ended < started:
        raise FormalExecutionError("formal session cannot end before it starts")
    if status not in {"completed", "failed"}:
        raise FormalExecutionError("formal session status must be completed or failed")
    return {
        "artifact_schema_version": FORMAL_SESSION_SCHEMA_VERSION,
        "artifact_type": FORMAL_SESSION_END_ARTIFACT_TYPE,
        "ended_at_utc": ended_at_utc,
        "session_id": session.session_id,
        "session_sha256": session.sha256,
        "started_at_utc": session.started_at_utc,
        "status": status,
    }


def _read_validated_session_start(
    campaign_identity: FormalCampaignIdentity,
    session_root: Path,
) -> FormalRunSession:
    _require_real_directory(session_root, label="session root")
    entries = tuple(session_root.iterdir())
    actual_entries = {path.name for path in entries}
    expected_entries = {"start.json"}
    if "end.json" in actual_entries:
        expected_entries.add("end.json")
    if actual_entries != expected_entries:
        raise FormalExecutionError("formal session directory contains unknown entries")
    payload = _read_json_object(session_root / "start.json", label="session start")
    session_payload = payload.get("session")
    if not isinstance(session_payload, Mapping):
        raise FormalExecutionError("formal session start lacks a session object")
    try:
        snapshot_payload = session_payload["session_snapshot"]
        if not isinstance(snapshot_payload, Mapping):
            raise ValueError("session_snapshot must be an object")
        if snapshot_payload.get("volatile") is not True:
            raise ValueError("session_snapshot must be marked volatile")
        blas_environment = snapshot_payload["blas_environment"]
        if not isinstance(blas_environment, Mapping) or any(
            not isinstance(name, str) or not isinstance(value, str)
            for name, value in blas_environment.items()
        ):
            raise ValueError("BLAS environment must be a string mapping")
        session = FormalRunSession(
            campaign_id=session_payload["campaign_id"],
            formal_plan_id=session_payload["formal_plan_id"],
            environment_contract_sha256=session_payload[
                "environment_contract_sha256"
            ],
            host_identity_sha256=session_payload["host_identity_sha256"],
            started_at_utc=session_payload["started_at_utc"],
            session_nonce=session_payload["session_nonce"],
            session_snapshot=SessionSnapshot(
                pid=snapshot_payload["pid"],
                executable=snapshot_payload["executable"],
                cwd=snapshot_payload["cwd"],
                affinity_cpu_ids=tuple(snapshot_payload["affinity_cpu_ids"]),
                blas_environment=tuple(sorted(blas_environment.items())),
                available_memory_bytes=snapshot_payload[
                    "available_memory_bytes"
                ],
                load_average=tuple(snapshot_payload["load_average"]),
            ),
            resource_lock_path=session_payload["resource_lock_path"],
            schema_version=session_payload["schema_version"],
        )
    except (FormalEnvironmentError, KeyError, TypeError, ValueError) as error:
        raise FormalExecutionError(
            f"formal session start is invalid: {session_root}"
        ) from error
    if (
        session.campaign_id != campaign_identity.campaign_id
        or session.formal_plan_id != campaign_identity.formal_plan_id
        or session.environment_contract_sha256
        != campaign_identity.environment_contract_sha256
        or session.host_identity_sha256 != campaign_identity.host_identity_sha256
    ):
        raise FormalExecutionError("formal session identity differs from its campaign")
    if session_root.name != session.session_id:
        raise FormalExecutionError("formal session directory name differs from session")
    if canonical_json(payload) != canonical_json(session.start_artifact_payload()):
        raise FormalExecutionError("formal session start artifact is not canonical")
    end_path = session_root / "end.json"
    if "end.json" in actual_entries:
        end_payload = _read_json_object(end_path, label="session end")
        status = end_payload.get("status")
        ended_at_utc = end_payload.get("ended_at_utc")
        if not isinstance(status, str) or not isinstance(ended_at_utc, str):
            raise FormalExecutionError("formal session end is invalid")
        if canonical_json(end_payload) != canonical_json(
            _session_end_payload(
                session,
                ended_at_utc=ended_at_utc,
                status=status,
            )
        ):
            raise FormalExecutionError("formal session end artifact is not canonical")
    return session


def _validate_sessions_root(
    campaign_identity: FormalCampaignIdentity,
    sessions_root: Path,
) -> dict[str, FormalRunSession]:
    _require_real_directory(sessions_root, label="sessions root")
    sessions: dict[str, FormalRunSession] = {}
    for session_root in sessions_root.iterdir():
        if _SESSION_ID_RE.fullmatch(session_root.name) is None:
            raise FormalExecutionError(
                f"sessions root contains an unknown entry: {session_root}"
            )
        _require_direct_child_directory(
            session_root,
            sessions_root,
            label="session root",
            parent_label="sessions root",
        )
        session = _read_validated_session_start(campaign_identity, session_root)
        sessions[session.session_id] = session
    return sessions


def finish_formal_session(
    campaign: FormalCampaign,
    *,
    status: str,
    now_utc: TimestampProvider = _utc_now,
) -> Path:
    """Publish the immutable end artifact for the campaign's current session."""

    _validate_campaign_handle(campaign)
    if _path_exists_lstat(campaign.session_end_path, label="formal session end"):
        raise FormalExecutionError("formal session end artifact already exists")
    try:
        ended_at_utc = now_utc()
        payload = _session_end_payload(
            campaign.session,
            ended_at_utc=ended_at_utc,
            status=status,
        )
        return write_canonical_json_exclusive(campaign.session_end_path, payload)
    except (OSError, TypeError, ValueError) as error:
        raise FormalExecutionError(
            f"cannot publish formal session end: {type(error).__name__}: {error}"
        ) from error


def prepare_formal_campaign(
    plan: FormalExecutionPlan,
    results_root: str | Path,
    current_binding_provider: CurrentBindingProvider,
    pre_claim_preparer: PreClaimPreparer,
    *,
    session_snapshot: SessionSnapshot,
    resource_lock_path: str,
    now_utc: TimestampProvider = _utc_now,
    session_nonce_provider: SessionNonceProvider = _session_nonce,
) -> FormalCampaign:
    """Create one new session under a path-independent formal campaign.

    ``current_binding_provider`` must independently observe the manifest,
    implementation digest, model, and Gurobi runtime on every invocation.
    The exact same provider is retained by the campaign for all slot claims.
    """

    if not isinstance(plan, FormalExecutionPlan):
        raise FormalExecutionError("plan must be a FormalExecutionPlan")
    if not callable(pre_claim_preparer):
        raise FormalExecutionError("pre_claim_preparer must be callable")
    if not isinstance(session_snapshot, SessionSnapshot):
        raise FormalExecutionError("session_snapshot must be a SessionSnapshot")
    if not isinstance(resource_lock_path, str) or not resource_lock_path.strip():
        raise FormalExecutionError("resource_lock_path must be a nonempty string")
    if not callable(now_utc) or not callable(session_nonce_provider):
        raise FormalExecutionError("formal session providers must be callable")
    binding = _validate_current_binding(plan, current_binding_provider)
    identity = _campaign_identity(plan, binding)
    expanded_results_root = Path(results_root).expanduser()
    resolved_results_root = Path(os.path.abspath(os.fspath(expanded_results_root)))
    plan_root = resolved_results_root / plan.plan_id
    root = plan_root / identity.campaign_id
    try:
        resolved_results_root.mkdir(parents=True, exist_ok=True)
        _require_real_directory(resolved_results_root, label="results root")
        plan_root.mkdir(parents=True, exist_ok=True)
        _require_direct_child_directory(
            plan_root,
            resolved_results_root,
            label="plan root",
            parent_label="results root",
        )
        for entry in plan_root.iterdir():
            if _CAMPAIGN_ID_RE.fullmatch(entry.name) is None:
                raise FormalExecutionError(
                    f"formal plan root contains an unknown entry: {entry}"
                )
            _require_direct_child_directory(
                entry,
                plan_root,
                label="campaign root",
                parent_label="plan root",
            )
        root.mkdir(parents=True, exist_ok=True)
        _require_direct_child_directory(
            root,
            plan_root,
            label="campaign root",
            parent_label="plan root",
        )
    except OSError as error:
        raise FormalExecutionError(
            f"cannot create formal campaign root: {error}"
        ) from error
    identity_path = root / "campaign_identity.json"
    if _path_exists_lstat(identity_path, label="campaign identity"):
        stored = _read_json_object(identity_path, label="campaign identity")
        _validate_campaign_identity_artifact(stored, identity)
    else:
        existing = tuple(root.iterdir())
        if existing:
            raise FormalExecutionError(
                "campaign identity is missing from a nonempty campaign root"
            )
        try:
            write_canonical_json_exclusive(identity_path, identity.artifact_payload())
        except FileExistsError:
            stored = _read_json_object(identity_path, label="campaign identity")
            _validate_campaign_identity_artifact(stored, identity)
        except OSError as error:
            raise FormalExecutionError(
                f"cannot write campaign identity: {error}"
            ) from error
    attempts_root = root / "attempts"
    journal_root = root / "journal"
    sessions_root = root / "sessions"
    evidence_seal_path = root / FORMAL_CAMPAIGN_EVIDENCE_FILENAME
    try:
        attempts_root.mkdir(exist_ok=True)
        _require_direct_child_directory(
            attempts_root,
            root,
            label="attempts root",
            parent_label="campaign root",
        )
        journal_root.mkdir(exist_ok=True)
        _require_direct_child_directory(
            journal_root,
            root,
            label="journal root",
            parent_label="campaign root",
        )
        sessions_root.mkdir(exist_ok=True)
        _require_direct_child_directory(
            sessions_root,
            root,
            label="sessions root",
            parent_label="campaign root",
        )
    except OSError as error:
        raise FormalExecutionError(
            f"cannot create campaign directories: {error}"
        ) from error
    _validate_sessions_root(identity, sessions_root)
    try:
        session = FormalRunSession(
            campaign_id=identity.campaign_id,
            formal_plan_id=plan.plan_id,
            environment_contract_sha256=identity.environment_contract_sha256,
            host_identity_sha256=identity.host_identity_sha256,
            started_at_utc=now_utc(),
            session_nonce=session_nonce_provider(),
            session_snapshot=session_snapshot,
            resource_lock_path=resource_lock_path,
        )
    except (TypeError, ValueError) as error:
        raise FormalExecutionError(
            f"cannot create formal session identity: {error}"
        ) from error
    session_root = sessions_root / session.session_id
    session_start_path = session_root / "start.json"
    session_end_path = session_root / "end.json"
    try:
        session_root.mkdir(exist_ok=False)
        _require_direct_child_directory(
            session_root,
            sessions_root,
            label="session root",
            parent_label="sessions root",
        )
        write_canonical_json_exclusive(
            session_start_path,
            session.start_artifact_payload(),
        )
    except FileExistsError as error:
        raise FormalExecutionError(
            f"formal session already exists: {session.session_id}"
        ) from error
    except OSError as error:
        try:
            session_start_path.unlink(missing_ok=True)
            session_root.rmdir()
        except OSError:
            pass
        raise FormalExecutionError(f"cannot create formal session: {error}") from error
    campaign = FormalCampaign(
        plan=plan,
        identity=identity,
        results_root=resolved_results_root,
        plan_root=plan_root,
        root=root,
        attempts_root=attempts_root,
        journal_root=journal_root,
        sessions_root=sessions_root,
        identity_path=identity_path,
        evidence_seal_path=evidence_seal_path,
        session=session,
        session_root=session_root,
        session_start_path=session_start_path,
        session_end_path=session_end_path,
    )
    object.__setattr__(campaign, "current_binding_provider", current_binding_provider)
    object.__setattr__(campaign, "pre_claim_preparer", pre_claim_preparer)
    object.__setattr__(campaign, "_factory_seal", _FORMAL_CAMPAIGN_FACTORY_SEAL)
    _validate_campaign_root_entries(campaign)
    _validate_sessions_root(identity, sessions_root)
    return campaign


def _expected_ids(slot: FormalSlot) -> dict[str, Any]:
    return {
        "attempt_id": slot.attempt_id,
        "attempt_number": FORMAL_ATTEMPT_NUMBER,
        "logical_run_id": slot.logical_run_id,
        "pair_id": slot.pair_id,
        "profile_id": slot.profile_id,
    }


def _pair_for_block(plan: FormalExecutionPlan, block: FormalBlock) -> PairSpec:
    return PairSpec(
        manifest_id=plan.manifest_id,
        case_id=block.case_id,
        mode=plan.qualification_mode,
        solver_seed=block.solver_seed,
        time_budget_seconds=block.time_budget_seconds,
        solver=plan.suite.solver,
    )


def _formal_context(
    campaign: FormalCampaign,
    block: FormalBlock,
    slot: FormalSlot,
    *,
    session_id: str | None = None,
) -> dict[str, Any]:
    return {
        "block_id": block.block_id,
        "block_index": block.block_index,
        "campaign_id": campaign.identity.campaign_id,
        "campaign_identity_sha256": campaign.identity.sha256,
        "formal_execution_implementation_sha256": (
            campaign.plan.formal_execution_implementation_sha256
        ),
        "formal_plan_id": campaign.plan.plan_id,
        "formal_plan_sha256": campaign.plan.sha256,
        "slot_id": slot.slot_id,
        "slot_index": slot.slot_index,
        "session_id": session_id or campaign.session.session_id,
    }


def _formal_claim_payload(
    campaign: FormalCampaign,
    block: FormalBlock,
    slot: FormalSlot,
    binding: FormalExecutionBinding,
    input_attestation: FormalInputAttestation,
    *,
    session_id: str | None = None,
) -> dict[str, Any]:
    return {
        "artifact_schema_version": FORMAL_ATTEMPT_ATTESTATION_SCHEMA_VERSION,
        "artifact_type": FORMAL_ATTEMPT_ATTESTATION_ARTIFACT_TYPE,
        "formal_context": _formal_context(
            campaign,
            block,
            slot,
            session_id=session_id,
        ),
        "observed_binding": binding.to_payload(),
        "observed_binding_sha256": binding.sha256,
        "prepared_input_attestation": input_attestation.to_payload(),
        "prepared_input_attestation_sha256": input_attestation.sha256,
    }


def _validate_input_attestation(
    task: FormalSlotTask,
    attestation: FormalInputAttestation,
) -> None:
    if attestation.case_id != task.block.case_id:
        raise FormalExecutionError("prepared input case_id differs from the slot")
    if attestation.slot_id != task.slot.slot_id:
        raise FormalExecutionError("prepared input slot_id differs from the slot")
    expected = attestation.expected_instance_sha256
    if expected != task.block.instance_sha256:
        raise FormalExecutionError(
            "prepared input expected_instance_sha256 differs from the formal plan"
        )
    differing = tuple(
        name
        for name in (
            "payload_instance_sha256",
            "source_sha256_before",
            "source_sha256_after",
        )
        if getattr(attestation, name) != expected
    )
    if differing:
        raise FormalExecutionError(
            "prepared input digest differs from expected_instance_sha256: "
            + ", ".join(differing)
        )
    try:
        attestation.qualification_trace.validate_binding(
            manifest_id=task.pair.manifest_id,
            benchmark_manifest_sha256=task.benchmark_manifest_sha256,
            manifest_artifact_sha256=task.manifest_artifact_sha256,
            case_id=task.block.case_id,
            scale=task.block.scale,
            instance_sha256=task.block.instance_sha256,
            qualification_run_id=task.block.qualification_run_id,
            qualification_decision_sha256=(
                task.block.qualification_decision_sha256
            ),
            qualification_case_sha256=(
                task.block.qualification_case_sha256
            ),
            effective_model_config_sha256=(
                task.block.effective_model_config_sha256
            ),
            screening_is_exact=(
                task.block.qualification_screening_is_exact
            ),
        )
    except ValueError as error:
        raise FormalExecutionError(
            f"prepared input qualification trace differs: {error}"
        ) from error


def _validate_prepared_input(
    task: FormalSlotTask,
    prepared: object,
) -> PreparedFormalInput:
    if not isinstance(prepared, PreparedFormalInput):
        raise FormalExecutionError(
            "pre_claim_preparer must return a PreparedFormalInput"
        )
    _validate_input_attestation(task, prepared.attestation)
    return prepared


def _snapshot_attempt_files(
    directory: Path,
    names: set[str],
) -> dict[str, _ImmutableFileSnapshot]:
    """Read every declared attempt file exactly once."""

    _require_real_directory(directory, label="attempt directory")
    snapshots = {
        name: _read_regular_file_once(
            directory / name,
            label="attempt artifact",
        )
        for name in sorted(names)
    }
    after_names = {path.name for path in directory.iterdir()}
    _require_real_directory(directory, label="attempt directory")
    if after_names != names:
        raise FormalExecutionError(
            "attempt bundle entries changed while immutable bytes were captured"
        )
    return snapshots


def _expected_core_solver_config(
    pair: PairSpec,
    profile: ProfileConfig,
    telemetry: TelemetryConfig,
) -> dict[str, Any]:
    """Rebuild the exact core configuration frozen by the formal plan."""

    return _json_ready(asdict(build_core_solver_config(pair, profile, telemetry)))


def _expected_applied_gurobi_parameters(
    core_solver_config: Mapping[str, Any],
) -> dict[str, Any]:
    expected = {
        parameter_name: core_solver_config[core_field]
        for parameter_name, core_field in (
            _APPLIED_STABLE_PARAMETER_TO_CORE_FIELD.items()
        )
    }
    expected["pre_crush"] = int(
        bool(core_solver_config["use_user_cuts"])
        and int(core_solver_config["max_root_cut_rounds"]) > 0
        and int(core_solver_config["max_cuts_per_round"]) > 0
    )
    return expected


def _validate_applied_gurobi_parameters(
    applied: object,
    expected_core: Mapping[str, Any],
    *,
    completed: bool,
    slot_id: str,
) -> None:
    """Validate stable controls and the remaining main-solve time budget.

    A completed solve must expose every parameter collected by the benchmark.
    A failed solve may expose only the subset reached before its exception, but
    every exposed value is still audited.  ``TimeLimit`` is the sole dynamic
    value: the model replaces the frozen total budget with the wall-clock time
    remaining after start preparation, so its exact admissible interval is
    ``[0, frozen time_limit]``.
    """

    if not isinstance(applied, Mapping):
        raise FormalExecutionError(
            f"run_record applied Gurobi parameters are missing for {slot_id}"
        )
    actual_names = frozenset(applied)
    unknown = sorted(actual_names - _APPLIED_PARAMETER_NAMES)
    if unknown:
        raise FormalExecutionError(
            f"run_record applied Gurobi parameters contain unknown controls for "
            f"{slot_id}: {', '.join(unknown)}"
        )
    if completed:
        missing = sorted(_APPLIED_PARAMETER_NAMES - actual_names)
        if missing:
            raise FormalExecutionError(
                f"run_record applied Gurobi parameters are incomplete for {slot_id}: "
                + ", ".join(missing)
            )

    expected_stable = _expected_applied_gurobi_parameters(expected_core)
    differing = sorted(
        name
        for name in actual_names & expected_stable.keys()
        if canonical_json(applied[name]) != canonical_json(expected_stable[name])
    )
    if differing:
        raise FormalExecutionError(
            f"run_record applied Gurobi parameters differ for {slot_id}: "
            + ", ".join(differing)
        )

    if "time_limit" in applied:
        observed_time_limit = applied["time_limit"]
        if (
            isinstance(observed_time_limit, bool)
            or not isinstance(observed_time_limit, (int, float))
            or not math.isfinite(float(observed_time_limit))
            or not 0.0
            <= float(observed_time_limit)
            <= float(expected_core["time_limit"])
        ):
            raise FormalExecutionError(
                f"run_record applied Gurobi time_limit is outside the frozen "
                f"remaining-budget interval for {slot_id}"
            )


def _expected_gurobi_parameter_classification(parameter_name: str) -> str:
    if parameter_name in CORE_SOLVER_ALLOWED_GUROBI_PARAMETERS:
        return "core_solver_allowed"
    if parameter_name in BENCHMARK_OPERATIONAL_GUROBI_PARAMETERS:
        return "benchmark_operational"
    if parameter_name in GUROBI_RUNTIME_CONTEXT_PARAMETERS:
        return "runtime_context"
    return "unallowed"


def _validate_gurobi_audit_value(
    value: object,
    value_type: str,
    *,
    allow_redacted: bool,
    slot_id: str,
    parameter_name: str,
    field_name: str,
) -> None:
    if isinstance(value, Mapping):
        fields = set(value)
        if value_type == "float" and fields == {"sentinel"}:
            if value.get("sentinel") not in {
                "GRB_INFINITY",
                "GRB_NEGATIVE_INFINITY",
            }:
                raise FormalExecutionError(
                    f"Gurobi parameter audit {field_name} sentinel is invalid for "
                    f"{slot_id}: {parameter_name}"
                )
            return
        redacted_hash = (
            allow_redacted
            and fields == {"redacted", "sha256"}
            and value.get("redacted") is True
            and isinstance(value.get("sha256"), str)
            and _SHA256_RE.fullmatch(cast(str, value.get("sha256"))) is not None
        )
        redacted_presence = (
            allow_redacted
            and fields == {"is_set", "redacted"}
            and value.get("redacted") is True
            and type(value.get("is_set")) is bool
        )
        if redacted_hash or redacted_presence:
            return
        raise FormalExecutionError(
            f"Gurobi parameter audit {field_name} is malformed for {slot_id}: "
            f"{parameter_name}"
        )
    valid = (
        (value_type == "int" and isinstance(value, int) and not isinstance(value, bool))
        or (
            value_type == "float"
            and isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
        )
        or (value_type == "str" and isinstance(value, str))
    )
    if not valid:
        raise FormalExecutionError(
            f"Gurobi parameter audit {field_name} type differs for {slot_id}: "
            f"{parameter_name}"
        )


def _validate_gurobi_parameter_audit(
    run_record: Mapping[str, Any],
    applied: object,
    *,
    completed: bool,
    slot_id: str,
) -> None:
    """Authenticate the full effective Gurobi state before formal recovery."""

    audit = run_record.get("gurobi_parameter_audit")
    if not isinstance(audit, Mapping) or set(audit) != _GUROBI_AUDIT_METADATA_FIELDS:
        raise FormalExecutionError(
            f"run_record Gurobi parameter audit metadata is invalid for {slot_id}"
        )
    if audit.get("schema_version") != GUROBI_PARAMETER_AUDIT_SCHEMA_VERSION or type(
        audit.get("schema_version")
    ) is not int:
        raise FormalExecutionError(
            f"run_record Gurobi parameter audit schema differs for {slot_id}"
        )
    enumeration_complete = audit.get("enumeration_complete")
    if type(enumeration_complete) is not bool:
        raise FormalExecutionError(
            f"run_record Gurobi parameter enumeration flag is invalid for {slot_id}"
        )

    effective = run_record.get("effective_nondefault_parameters")
    if not isinstance(effective, Mapping):
        raise FormalExecutionError(
            f"run_record effective Gurobi parameters are invalid for {slot_id}"
        )
    classifications: dict[str, str] = {}
    for parameter_name, raw_entry in effective.items():
        if not isinstance(parameter_name, str) or not parameter_name:
            raise FormalExecutionError(
                f"run_record effective Gurobi parameter name is invalid for {slot_id}"
            )
        if (
            not isinstance(raw_entry, Mapping)
            or set(raw_entry) != _GUROBI_EFFECTIVE_ENTRY_FIELDS
        ):
            raise FormalExecutionError(
                f"run_record effective Gurobi parameter entry is invalid for "
                f"{slot_id}: {parameter_name}"
            )
        expected_classification = _expected_gurobi_parameter_classification(
            parameter_name
        )
        if raw_entry.get("classification") != expected_classification:
            raise FormalExecutionError(
                f"run_record Gurobi parameter classification differs for "
                f"{slot_id}: {parameter_name}"
            )
        value_type = raw_entry.get("value_type")
        if value_type not in {"int", "float", "str"}:
            raise FormalExecutionError(
                f"run_record Gurobi parameter value_type is invalid for "
                f"{slot_id}: {parameter_name}"
            )
        allow_redacted = expected_classification != "core_solver_allowed"
        for field_name in ("current", "default"):
            _validate_gurobi_audit_value(
                raw_entry.get(field_name),
                cast(str, value_type),
                allow_redacted=allow_redacted,
                slot_id=slot_id,
                parameter_name=parameter_name,
                field_name=field_name,
            )
        classifications[parameter_name] = expected_classification

    raw_unallowed = run_record.get("unallowed_nondefault_parameters")
    if (
        not isinstance(raw_unallowed, list)
        or any(not isinstance(name, str) or not name for name in raw_unallowed)
        or raw_unallowed != sorted(set(raw_unallowed))
    ):
        raise FormalExecutionError(
            f"run_record unallowed Gurobi parameter list is invalid for {slot_id}"
        )
    expected_unallowed = sorted(
        name
        for name, classification in classifications.items()
        if classification == "unallowed"
    )
    if raw_unallowed != expected_unallowed:
        raise FormalExecutionError(
            f"run_record unallowed Gurobi parameters differ from effective state "
            f"for {slot_id}"
        )

    context = run_record.get("gurobi_execution_context")
    if not isinstance(context, Mapping) or set(context) != _GUROBI_EXECUTION_CONTEXT_FIELDS:
        raise FormalExecutionError(
            f"run_record Gurobi execution context is invalid for {slot_id}"
        )
    if context.get("observation_basis") != "gurobi_effective_parameters":
        raise FormalExecutionError(
            f"run_record Gurobi execution observation basis differs for {slot_id}"
        )

    count = audit.get("enumerated_parameter_count")
    catalog_sha256 = audit.get("parameter_catalog_sha256")
    reason_code = audit.get("reason_code")
    if enumeration_complete:
        if reason_code is not None:
            raise FormalExecutionError(
                f"complete Gurobi parameter audit has a reason for {slot_id}"
            )
        if (
            isinstance(count, bool)
            or not isinstance(count, int)
            or count <= 0
            or count < len(effective)
        ):
            raise FormalExecutionError(
                f"complete Gurobi parameter audit count is invalid for {slot_id}"
            )
        if (
            not isinstance(catalog_sha256, str)
            or _SHA256_RE.fullmatch(catalog_sha256) is None
        ):
            raise FormalExecutionError(
                f"complete Gurobi parameter catalog hash is invalid for {slot_id}"
            )
        if raw_unallowed:
            raise FormalExecutionError(
                f"formal Gurobi parameter audit contains unallowed controls for "
                f"{slot_id}: {', '.join(raw_unallowed)}"
            )
        if context != {
            "execution_mode": "local",
            "observation_basis": "gurobi_effective_parameters",
            "remote_identity_sha256": None,
            "remote_mechanisms": [],
        }:
            raise FormalExecutionError(
                f"formal Gurobi execution context must be local for {slot_id}"
            )
        if not isinstance(applied, Mapping):
            raise FormalExecutionError(
                f"run_record applied Gurobi parameters are missing for {slot_id}"
            )
        controlled_drift: list[str] = []
        for parameter_name, classification in classifications.items():
            if classification != "core_solver_allowed":
                continue
            applied_name = _GUROBI_CANONICAL_TO_APPLIED_PARAMETER[parameter_name]
            entry = cast(Mapping[str, Any], effective[parameter_name])
            if (
                applied_name not in applied
                or canonical_json(entry["current"])
                != canonical_json(applied[applied_name])
            ):
                controlled_drift.append(parameter_name)
        if controlled_drift:
            raise FormalExecutionError(
                f"effective and applied Gurobi parameters differ for {slot_id}: "
                + ", ".join(sorted(controlled_drift))
            )
        return

    if completed:
        raise FormalExecutionError(
            f"completed formal run requires a complete Gurobi parameter audit for "
            f"{slot_id}"
        )
    if (
        not isinstance(reason_code, str)
        or _GUROBI_AUDIT_REASON_RE.fullmatch(reason_code) is None
        or type(count) is not int
        or count != 0
        or catalog_sha256 is not None
        or effective
        or raw_unallowed
        or context
        != {
            "execution_mode": "unobserved",
            "observation_basis": "gurobi_effective_parameters",
            "remote_identity_sha256": None,
            "remote_mechanisms": [],
        }
    ):
        raise FormalExecutionError(
            f"failed formal run has a noncanonical incomplete Gurobi audit for "
            f"{slot_id}"
        )


def _validate_attempt_bundle(
    campaign: FormalCampaign,
    block: FormalBlock,
    slot: FormalSlot,
) -> FormalAttemptBundle:
    directory = attempt_directory(
        campaign.attempts_root,
        slot.logical_run_id,
        FORMAL_ATTEMPT_NUMBER,
    )
    logical_root = directory.parent
    _require_direct_child_directory(
        logical_root,
        campaign.attempts_root,
        label="logical-run directory",
        parent_label="attempts root",
    )
    _require_direct_child_directory(
        directory,
        logical_root,
        label="attempt directory",
        parent_label="logical-run directory",
    )
    entries = tuple(directory.iterdir())
    if any(not path.is_file() or path.is_symlink() for path in entries):
        raise FormalExecutionError(
            f"attempt-1 bundle contains a non-regular file for {slot.slot_id}"
        )
    actual_names = {path.name for path in entries}
    missing = sorted(_REQUIRED_ATTEMPT_FILES - actual_names)
    if missing:
        raise FormalExecutionError(
            f"incomplete attempt1 for {slot.slot_id}; missing: {', '.join(missing)}"
        )
    unknown = sorted(actual_names - _REQUIRED_ATTEMPT_FILES - _OPTIONAL_ATTEMPT_FILES)
    if unknown:
        raise FormalExecutionError(
            f"attempt-1 bundle contains unknown artifacts for {slot.slot_id}: "
            + ", ".join(unknown)
        )
    artifact_snapshots = _snapshot_attempt_files(directory, actual_names)

    expected_ids = _expected_ids(slot)
    claim = _json_object_from_snapshot(
        artifact_snapshots["claim.json"],
        label="attempt claim",
        path=directory / "claim.json",
    )
    expected_claim = {
        "attempt_id": slot.attempt_id,
        "attempt_number": FORMAL_ATTEMPT_NUMBER,
        "logical_run_id": slot.logical_run_id,
    }
    if claim != expected_claim:
        raise FormalExecutionError(f"attempt claim identity differs for {slot.slot_id}")
    expected_binding = FormalExecutionBinding(
        benchmark_manifest_sha256=campaign.plan.benchmark_manifest_sha256,
        manifest_artifact_sha256=campaign.plan.manifest_artifact_sha256,
        manifest_id=campaign.plan.manifest_id,
        model_id=campaign.plan.model_id,
        formal_execution_implementation_sha256=(
            campaign.plan.formal_execution_implementation_sha256
        ),
        qualification_mode=campaign.plan.qualification_mode,
        gurobi_version=campaign.plan.gurobi_version,
        environment_contract=campaign.plan.environment_contract,
        host_identity_sha256=campaign.identity.host_identity_sha256,
    )
    formal_claim = _json_object_from_snapshot(
        artifact_snapshots["formal_claim.json"],
        label="formal attempt attestation",
        path=directory / "formal_claim.json",
    )
    claim_context = formal_claim.get("formal_context")
    if not isinstance(claim_context, Mapping):
        raise FormalExecutionError(
            f"formal attempt context is missing for {slot.slot_id}"
        )
    session_id = claim_context.get("session_id")
    if not isinstance(session_id, str) or _SESSION_ID_RE.fullmatch(session_id) is None:
        raise FormalExecutionError(
            f"formal attempt session identity is invalid for {slot.slot_id}"
        )
    session_root = campaign.sessions_root / session_id
    _require_direct_child_directory(
        session_root,
        campaign.sessions_root,
        label="formal attempt session root",
        parent_label="sessions root",
    )
    _read_validated_session_start(campaign.identity, session_root)
    expected_formal_context = _formal_context(
        campaign,
        block,
        slot,
        session_id=session_id,
    )
    try:
        input_attestation = FormalInputAttestation.from_payload(
            formal_claim.get("prepared_input_attestation")
        )
        _validate_input_attestation(
            _task_for_slot(campaign, block, slot),
            input_attestation,
        )
    except (FormalExecutionError, TypeError, ValueError) as error:
        raise FormalExecutionError(
            f"formal prepared input attestation differs for {slot.slot_id}: {error}"
        ) from error
    if formal_claim != _formal_claim_payload(
        campaign,
        block,
        slot,
        expected_binding,
        input_attestation,
        session_id=session_id,
    ):
        raise FormalExecutionError(
            f"formal attempt attestation differs for {slot.slot_id}"
        )
    expected_run_formal_context = bind_qualification_trace(
        expected_formal_context,
        input_attestation.qualification_trace,
    )

    pair = _pair_for_block(campaign.plan, block)
    profile = campaign.plan.suite.profile(slot.profile_code)
    expected_core_solver_config = _expected_core_solver_config(
        pair,
        profile,
        campaign.plan.suite.telemetry,
    )
    run_spec = _json_object_from_snapshot(
        artifact_snapshots["run_spec.json"],
        label="run_spec",
        path=directory / "run_spec.json",
    )
    if run_spec.get("schema_version") != 1:
        raise FormalExecutionError(f"run_spec schema differs for {slot.slot_id}")
    if run_spec.get("ids") != expected_ids:
        raise FormalExecutionError(f"run_spec IDs differ for {slot.slot_id}")
    if run_spec.get("formal_context") != expected_run_formal_context:
        raise FormalExecutionError(
            f"run_spec formal context differs for {slot.slot_id}"
        )
    if canonical_json(run_spec.get("pair")) != canonical_json(_json_ready(pair)):
        raise FormalExecutionError(f"run_spec pair differs for {slot.slot_id}")
    if canonical_json(run_spec.get("profile")) != canonical_json(_json_ready(profile)):
        raise FormalExecutionError(f"run_spec profile differs for {slot.slot_id}")
    if canonical_json(run_spec.get("telemetry")) != canonical_json(
        _json_ready(campaign.plan.suite.telemetry)
    ):
        raise FormalExecutionError(f"run_spec telemetry differs for {slot.slot_id}")
    if run_spec.get("profile_label") != profile.label:
        raise FormalExecutionError(f"run_spec profile label differs for {slot.slot_id}")
    if run_spec.get("gurobi_version") != campaign.plan.gurobi_version:
        raise FormalExecutionError(
            f"run_spec Gurobi version differs for {slot.slot_id}"
        )
    if canonical_json(run_spec.get("core_solver_config")) != canonical_json(
        expected_core_solver_config
    ):
        raise FormalExecutionError(
            f"run_spec core solver config differs for {slot.slot_id}"
        )

    run_record = _json_object_from_snapshot(
        artifact_snapshots["run_record.json"],
        label="run_record",
        path=directory / "run_record.json",
    )
    if run_record.get("schema_version") != 1:
        raise FormalExecutionError(f"run_record schema differs for {slot.slot_id}")
    if run_record.get("ids") != expected_ids:
        raise FormalExecutionError(f"run_record IDs differ for {slot.slot_id}")
    if run_record.get("formal_context") != expected_run_formal_context:
        raise FormalExecutionError(
            f"run_record formal context differs for {slot.slot_id}"
        )
    if run_record.get("profile_code") != slot.profile_code.value:
        raise FormalExecutionError(f"run_record profile differs for {slot.slot_id}")
    if run_record.get("profile_label") != profile.label:
        raise FormalExecutionError(
            f"run_record profile label differs for {slot.slot_id}"
        )
    if run_record.get("gurobi_version") != campaign.plan.gurobi_version:
        raise FormalExecutionError(
            f"run_record Gurobi version differs for {slot.slot_id}"
        )
    if canonical_json(run_record.get("core_solver_config")) != canonical_json(
        expected_core_solver_config
    ):
        raise FormalExecutionError(
            f"run_record core solver config differs for {slot.slot_id}"
        )
    outcome = run_record.get("outcome")
    if outcome not in {"completed_with_incumbent", "completed_no_incumbent", "failed"}:
        raise FormalExecutionError(f"run_record outcome is invalid for {slot.slot_id}")
    completed = run_record.get("completed_without_exception")
    if type(completed) is not bool:
        raise FormalExecutionError(
            f"run_record completed flag is invalid for {slot.slot_id}"
        )
    applied_gurobi_parameters = run_record.get("applied_gurobi_parameters")
    _validate_applied_gurobi_parameters(
        applied_gurobi_parameters,
        expected_core_solver_config,
        completed=completed,
        slot_id=slot.slot_id,
    )
    _validate_gurobi_parameter_audit(
        run_record,
        applied_gurobi_parameters,
        completed=completed,
        slot_id=slot.slot_id,
    )
    solver_metrics = run_record.get("solver_metrics")
    if not isinstance(solver_metrics, Mapping):
        raise FormalExecutionError(
            f"run_record solver metrics are missing for {slot.slot_id}"
        )
    has_incumbent = solver_metrics.get("has_incumbent")
    if type(has_incumbent) is not bool:
        raise FormalExecutionError(
            f"run_record incumbent flag is invalid for {slot.slot_id}"
        )
    try:
        solution_structure = validate_solution_structure_payload(
            run_record.get("solution_structure"),
            input_attestation.qualification_trace,
            expected_has_incumbent=has_incumbent,
        )
    except (TypeError, ValueError) as error:
        raise FormalExecutionError(
            f"run_record solution structure differs for {slot.slot_id}: {error}"
        ) from error
    if run_record.get("solution_structure_sha256") != canonical_sha256(
        solution_structure.to_payload()
    ):
        raise FormalExecutionError(
            f"run_record solution structure SHA256 differs for {slot.slot_id}"
        )
    failure = run_record.get("failure")
    expects_failure = outcome == "failed"
    if completed == expects_failure:
        raise FormalExecutionError(
            f"run_record outcome/completed flag conflict for {slot.slot_id}"
        )
    if outcome == "completed_with_incumbent" and not has_incumbent:
        raise FormalExecutionError(
            f"completed_with_incumbent lacks an incumbent for {slot.slot_id}"
        )
    if outcome == "completed_no_incumbent" and has_incumbent:
        raise FormalExecutionError(
            f"completed_no_incumbent has an incumbent for {slot.slot_id}"
        )
    if expects_failure != isinstance(failure, Mapping):
        raise FormalExecutionError(f"failure evidence differs for {slot.slot_id}")
    has_failure_file = "failure.json" in actual_names
    if has_failure_file != expects_failure:
        raise FormalExecutionError(
            f"failure artifact presence differs for {slot.slot_id}"
        )
    if has_failure_file:
        failure_payload = _json_object_from_snapshot(
            artifact_snapshots["failure.json"],
            label="failure artifact",
            path=directory / "failure.json",
        )
        if failure_payload.get("ids") != expected_ids:
            raise FormalExecutionError(
                f"failure artifact IDs differ for {slot.slot_id}"
            )
        if failure_payload.get("formal_context") != expected_run_formal_context:
            raise FormalExecutionError(
                f"failure artifact formal context differs for {slot.slot_id}"
            )

    cut_stats = _json_object_from_snapshot(
        artifact_snapshots["cut_stats.json"],
        label="cut stats",
        path=directory / "cut_stats.json",
    )
    progress = _jsonl_objects_from_snapshot(
        artifact_snapshots["progress.jsonl"],
        label="progress artifact",
        path=directory / "progress.jsonl",
    )
    root_rounds = _jsonl_objects_from_snapshot(
        artifact_snapshots["root_rounds.jsonl"],
        label="root-round artifact",
        path=directory / "root_rounds.jsonl",
    )
    try:
        validate_formal_attempt_paper_metrics(
            run_record=run_record,
            progress=progress,
            root_rounds=root_rounds,
            cut_stats=cut_stats,
            core_solver_config=expected_core_solver_config,
            completed=completed,
        )
    except FormalMetricValidationError as error:
        raise FormalExecutionError(
            f"formal paper metrics are invalid for {slot.slot_id}: {error}"
        ) from error
    file_sha256 = tuple(
        (name, artifact_snapshots[name].sha256) for name in sorted(actual_names)
    )
    attempt_bundle_sha256 = canonical_sha256(
        {
            "domain": FORMAL_SLOT_BUNDLE_HASH_DOMAIN,
            "slot_id": slot.slot_id,
            "slot_index": slot.slot_index,
            "attempt_id": slot.attempt_id,
            "file_sha256": dict(file_sha256),
        }
    )
    return FormalAttemptBundle(
        directory=directory,
        outcome=outcome,
        completed_without_exception=completed,
        has_incumbent=has_incumbent,
        session_id=session_id,
        prepared_input_attestation=input_attestation,
        prepared_input_attestation_sha256=input_attestation.sha256,
        file_sha256=file_sha256,
        attempt_bundle_sha256=attempt_bundle_sha256,
        run_spec=_freeze_json_object(run_spec),
        run_record=_freeze_json_object(run_record),
        progress=tuple(_freeze_json_object(row) for row in progress),
        root_rounds=tuple(_freeze_json_object(row) for row in root_rounds),
        cut_stats=_freeze_json_object(cut_stats),
    )


def _checkpoint_path(campaign: FormalCampaign, slot: FormalSlot) -> Path:
    return campaign.journal_root / f"slot-{slot.slot_index:04d}.json"


@dataclass(frozen=True, slots=True)
class _ValidatedCheckpoint:
    path: Path
    bytes_sha256: str


def _checkpoint_payload(
    campaign: FormalCampaign,
    block: FormalBlock,
    slot: FormalSlot,
    bundle: FormalAttemptBundle,
    *,
    recovered_from_attempt_bundle: bool,
) -> dict[str, Any]:
    relative_attempt = bundle.directory.relative_to(campaign.root).as_posix()
    qualification_trace = bundle.prepared_input_attestation.qualification_trace
    solution_structure = bundle.run_record["solution_structure"]
    return {
        "artifact_schema_version": FORMAL_SLOT_CHECKPOINT_SCHEMA_VERSION,
        "artifact_type": FORMAL_SLOT_CHECKPOINT_ARTIFACT_TYPE,
        "attempt_directory": relative_attempt,
        "attempt_id": slot.attempt_id,
        "attempt_number": FORMAL_ATTEMPT_NUMBER,
        "block_id": block.block_id,
        "block_index": block.block_index,
        "bundle_file_sha256": dict(bundle.file_sha256),
        "campaign_id": campaign.identity.campaign_id,
        "campaign_identity_sha256": campaign.identity.sha256,
        "case_id": block.case_id,
        "completed_without_exception": bundle.completed_without_exception,
        "formal_plan_id": campaign.plan.plan_id,
        "formal_plan_sha256": campaign.plan.sha256,
        "formal_execution_implementation_sha256": (
            campaign.plan.formal_execution_implementation_sha256
        ),
        "has_incumbent": bundle.has_incumbent,
        "logical_run_id": slot.logical_run_id,
        "outcome": bundle.outcome,
        "pair_id": slot.pair_id,
        "profile_code": slot.profile_code.value,
        "profile_id": slot.profile_id,
        "qualification_case_sha256": (
            qualification_trace.qualification_case_sha256
        ),
        "qualification_decision_sha256": (
            qualification_trace.qualification_decision_sha256
        ),
        "qualification_run_id": qualification_trace.qualification_run_id,
        "qualification_trace": qualification_trace.to_payload(),
        "qualification_trace_sha256": qualification_trace.sha256,
        "prepared_input_attestation_sha256": (
            bundle.prepared_input_attestation_sha256
        ),
        "prepared_instance_sha256": (
            bundle.prepared_input_attestation.expected_instance_sha256
        ),
        "recovered_from_attempt_bundle": recovered_from_attempt_bundle,
        "scale": block.scale,
        "session_id": bundle.session_id,
        "solution_structure": solution_structure,
        "solution_structure_sha256": canonical_sha256(solution_structure),
        "slot_id": slot.slot_id,
        "slot_index": slot.slot_index,
        "solver_seed": block.solver_seed,
        "time_budget_seconds": block.time_budget_seconds,
    }


def _validate_checkpoint(
    campaign: FormalCampaign,
    block: FormalBlock,
    slot: FormalSlot,
    bundle: FormalAttemptBundle,
    checkpoint_path: Path,
) -> _ValidatedCheckpoint:
    checkpoint_snapshot = _read_regular_file_once(
        checkpoint_path,
        label="slot checkpoint",
    )
    payload = _json_object_from_snapshot(
        checkpoint_snapshot,
        label="slot checkpoint",
        path=checkpoint_path,
    )
    recovered = payload.get("recovered_from_attempt_bundle")
    if type(recovered) is not bool:
        raise FormalExecutionError(
            f"checkpoint recovery flag is invalid for {slot.slot_id}"
        )
    expected = _checkpoint_payload(
        campaign,
        block,
        slot,
        bundle,
        recovered_from_attempt_bundle=recovered,
    )
    if canonical_json(payload) != canonical_json(expected):
        raise FormalExecutionError(
            f"slot checkpoint differs from its plan/attempt bundle: {slot.slot_id}"
        )
    expected_bytes = (canonical_json(expected) + "\n").encode("utf-8")
    if checkpoint_snapshot.payload != expected_bytes:
        raise FormalExecutionError(
            f"slot checkpoint bytes are not canonical: {slot.slot_id}"
        )
    return _ValidatedCheckpoint(
        path=checkpoint_path,
        bytes_sha256=checkpoint_snapshot.sha256,
    )


def _write_checkpoint(
    campaign: FormalCampaign,
    block: FormalBlock,
    slot: FormalSlot,
    bundle: FormalAttemptBundle,
    *,
    recovered_from_attempt_bundle: bool,
) -> _ValidatedCheckpoint:
    path = _checkpoint_path(campaign, slot)
    payload = _checkpoint_payload(
        campaign,
        block,
        slot,
        bundle,
        recovered_from_attempt_bundle=recovered_from_attempt_bundle,
    )
    try:
        write_canonical_json_exclusive(path, payload)
    except FileExistsError as error:
        raise FormalExecutionError(
            f"slot checkpoint already exists during exclusive publication: {slot.slot_id}"
        ) from error
    except OSError as error:
        raise FormalExecutionError(
            f"cannot publish slot checkpoint for {slot.slot_id}: {error}"
        ) from error
    checkpoint_bytes = (canonical_json(payload) + "\n").encode("utf-8")
    checkpoint_snapshot = _read_regular_file_once(
        path,
        label="new slot checkpoint",
    )
    if checkpoint_snapshot.payload != checkpoint_bytes:
        raise FormalExecutionError(
            f"new slot checkpoint bytes are not canonical: {slot.slot_id}"
        )
    return _ValidatedCheckpoint(
        path=path,
        bytes_sha256=checkpoint_snapshot.sha256,
    )


@dataclass(frozen=True, slots=True)
class FormalCampaignEvidenceSeal:
    """Canonical content root for all 900 formal attempt/checkpoint snapshots."""

    path: Path
    evidence_id: str
    evidence_root_sha256: str
    authenticated_slot_count: int

    def __post_init__(self) -> None:
        if _EVIDENCE_ID_RE.fullmatch(self.evidence_id) is None:
            raise ValueError("evidence_id is not canonical")
        if _SHA256_RE.fullmatch(self.evidence_root_sha256) is None:
            raise ValueError("evidence_root_sha256 is invalid")
        if self.evidence_id != f"formal-evidence-{self.evidence_root_sha256}":
            raise ValueError("evidence_id differs from evidence_root_sha256")
        if self.authenticated_slot_count != FORMAL_SLOT_COUNT:
            raise ValueError("formal campaign evidence requires exactly 900 slots")


def _evidence_slot_payload(
    snapshot: AuthenticatedFormalSlotSnapshot,
) -> dict[str, Any]:
    return {
        "attempt_bundle_sha256": snapshot.attempt_bundle_sha256,
        "block_id": snapshot.block_id,
        "bundle_file_sha256": dict(snapshot.file_sha256),
        "checkpoint_bytes_sha256": snapshot.checkpoint_bytes_sha256,
        "checkpoint_name": f"slot-{snapshot.slot_index:04d}.json",
        "logical_run_id": snapshot.logical_run_id,
        "slot_id": snapshot.slot_id,
        "slot_index": snapshot.slot_index,
    }


def _campaign_evidence_artifact(
    campaign: FormalCampaign,
    authenticated_slots: tuple[AuthenticatedFormalSlotSnapshot, ...],
) -> tuple[dict[str, Any], str, str]:
    expected_slots = tuple(
        (block, slot) for block in campaign.plan.blocks for slot in block.slots
    )
    if len(authenticated_slots) != FORMAL_SLOT_COUNT:
        raise FormalExecutionError(
            "formal campaign evidence requires exactly 900 authenticated slots"
        )
    observed_axes = tuple(
        (
            snapshot.slot_index,
            snapshot.slot_id,
            snapshot.logical_run_id,
            snapshot.block_id,
        )
        for snapshot in authenticated_slots
    )
    expected_axes = tuple(
        (slot.slot_index, slot.slot_id, slot.logical_run_id, block.block_id)
        for block, slot in expected_slots
    )
    if observed_axes != expected_axes:
        raise FormalExecutionError(
            "authenticated slot snapshots are not in exact formal-plan order"
        )
    for snapshot, (_block, slot) in zip(
        authenticated_slots,
        expected_slots,
        strict=True,
    ):
        expected_file_names = _REQUIRED_ATTEMPT_FILES | (
            {"failure.json"} if snapshot.outcome == "failed" else set()
        )
        if snapshot.outcome not in {
            "completed_with_incumbent",
            "completed_no_incumbent",
            "failed",
        }:
            raise FormalExecutionError(
                f"authenticated attempt outcome is invalid for {slot.slot_id}"
            )
        if set(dict(snapshot.file_sha256)) != expected_file_names:
            raise FormalExecutionError(
                f"authenticated attempt file set differs for {slot.slot_id}"
            )
        expected_bundle_sha256 = canonical_sha256(
            {
                "domain": FORMAL_SLOT_BUNDLE_HASH_DOMAIN,
                "slot_id": slot.slot_id,
                "slot_index": slot.slot_index,
                "attempt_id": slot.attempt_id,
                "file_sha256": dict(snapshot.file_sha256),
            }
        )
        if snapshot.attempt_bundle_sha256 != expected_bundle_sha256:
            raise FormalExecutionError(
                f"authenticated attempt bundle digest differs for {slot.slot_id}"
            )
    evidence_payload = {
        "benchmark_manifest_sha256": campaign.plan.benchmark_manifest_sha256,
        "campaign_id": campaign.identity.campaign_id,
        "campaign_identity_sha256": campaign.identity.sha256,
        "domain": FORMAL_CAMPAIGN_EVIDENCE_HASH_DOMAIN,
        "formal_execution_implementation_sha256": (
            campaign.plan.formal_execution_implementation_sha256
        ),
        "formal_plan_id": campaign.plan.plan_id,
        "formal_plan_sha256": campaign.plan.sha256,
        "slot_evidence": [
            _evidence_slot_payload(snapshot) for snapshot in authenticated_slots
        ],
    }
    evidence_root_sha256 = canonical_sha256(evidence_payload)
    evidence_id = f"formal-evidence-{evidence_root_sha256}"
    artifact = {
        "artifact_schema_version": FORMAL_CAMPAIGN_EVIDENCE_SCHEMA_VERSION,
        "artifact_type": FORMAL_CAMPAIGN_EVIDENCE_ARTIFACT_TYPE,
        "authenticated_slot_count": len(authenticated_slots),
        "evidence_id": evidence_id,
        "evidence_payload": evidence_payload,
        "evidence_root_sha256": evidence_root_sha256,
    }
    return artifact, evidence_id, evidence_root_sha256


def seal_formal_campaign_evidence(
    campaign: FormalCampaign,
    authenticated_slots: tuple[AuthenticatedFormalSlotSnapshot, ...],
    *,
    expected_evidence_root_sha256: str | None = None,
) -> FormalCampaignEvidenceSeal:
    """Exclusively publish or strictly revalidate the complete evidence seal.

    The seal intentionally excludes the mutable ``sessions`` directory.  Each
    producing session is already transitively bound through the attempt bundle,
    while later read-only analysis sessions may be appended without changing
    the scientific evidence root.
    """

    _validate_campaign_handle(campaign)
    if expected_evidence_root_sha256 is not None and (
        not isinstance(expected_evidence_root_sha256, str)
        or _SHA256_RE.fullmatch(expected_evidence_root_sha256) is None
    ):
        raise FormalExecutionError(
            "expected_evidence_root_sha256 must be a lowercase SHA256 digest"
        )
    artifact, evidence_id, evidence_root_sha256 = _campaign_evidence_artifact(
        campaign,
        authenticated_slots,
    )
    if (
        expected_evidence_root_sha256 is not None
        and evidence_root_sha256 != expected_evidence_root_sha256
    ):
        raise FormalExecutionError(
            "formal campaign evidence root differs from the externally expected root"
        )
    expected_bytes = (canonical_json(artifact) + "\n").encode("utf-8")
    if _path_exists_lstat(
        campaign.evidence_seal_path,
        label="formal campaign evidence seal",
    ):
        stored = _read_regular_file_once(
            campaign.evidence_seal_path,
            label="formal campaign evidence seal",
        )
        if stored.payload != expected_bytes:
            raise FormalExecutionError(
                "formal campaign evidence seal differs from authenticated slot evidence"
            )
    else:
        try:
            write_canonical_json_exclusive(campaign.evidence_seal_path, artifact)
        except FileExistsError:
            stored = _read_regular_file_once(
                campaign.evidence_seal_path,
                label="formal campaign evidence seal",
            )
            if stored.payload != expected_bytes:
                raise FormalExecutionError(
                    "concurrently published formal evidence seal differs"
                )
        except OSError as error:
            raise FormalExecutionError(
                f"cannot publish formal campaign evidence seal: {error}"
            ) from error
    return FormalCampaignEvidenceSeal(
        path=campaign.evidence_seal_path,
        evidence_id=evidence_id,
        evidence_root_sha256=evidence_root_sha256,
        authenticated_slot_count=len(authenticated_slots),
    )


def _slot_lookup(
    plan: FormalExecutionPlan,
) -> dict[str, tuple[FormalBlock, FormalSlot]]:
    return {
        slot.logical_run_id: (block, slot)
        for block in plan.blocks
        for slot in block.slots
    }


def _validate_attempt_tree(campaign: FormalCampaign) -> set[str]:
    _require_real_directory(campaign.attempts_root, label="attempts root")
    expected = _slot_lookup(campaign.plan)
    present: set[str] = set()
    for logical_path in campaign.attempts_root.iterdir():
        logical_run_id = logical_path.name
        _require_direct_child_directory(
            logical_path,
            campaign.attempts_root,
            label="logical-run directory",
            parent_label="attempts root",
        )
        if (
            _RUN_ID_RE.fullmatch(logical_run_id) is None
            or logical_run_id not in expected
        ):
            raise FormalExecutionError(
                f"attempts root contains an unknown logical run: {logical_run_id}"
            )
        children = tuple(logical_path.iterdir())
        expected_attempt_id = expected[logical_run_id][1].attempt_id
        for child in children:
            if _ATTEMPT_ID_RE.fullmatch(child.name) is None:
                raise FormalExecutionError(
                    f"logical run contains an unknown result entry: {child}"
                )
            _require_direct_child_directory(
                child,
                logical_path,
                label="attempt directory",
                parent_label="logical-run directory",
            )
            if child.name != expected_attempt_id:
                raise FormalExecutionError(
                    f"formal logical run contains a non-attempt1 result: {child.name}"
                )
        if len(children) != 1:
            raise FormalExecutionError(
                f"formal logical run must contain exactly one attempt1 directory: "
                f"{logical_run_id}"
            )
        present.add(logical_run_id)
    return present


def _validate_journal_tree(campaign: FormalCampaign) -> dict[int, Path]:
    _require_real_directory(campaign.journal_root, label="journal root")
    checkpoints: dict[int, Path] = {}
    for path in campaign.journal_root.iterdir():
        if not path.is_file() or path.is_symlink():
            raise FormalExecutionError(f"journal contains an unknown entry: {path}")
        match = _CHECKPOINT_NAME_RE.fullmatch(path.name)
        if match is None:
            raise FormalExecutionError(
                f"journal contains an unknown checkpoint name: {path.name}"
            )
        slot_index = int(match.group(1))
        if not 0 <= slot_index < FORMAL_SLOT_COUNT:
            raise FormalExecutionError(
                f"journal checkpoint is outside the formal plan: {path.name}"
            )
        if slot_index in checkpoints:
            raise FormalExecutionError(f"duplicate checkpoint for slot {slot_index}")
        checkpoints[slot_index] = path
    return checkpoints


def reconcile_formal_campaign(
    campaign: FormalCampaign,
    *,
    expected_evidence_root_sha256: str | None = None,
) -> FormalCampaignState:
    """Validate all existing evidence, recover checkpoints, and list pending slots."""

    _validate_campaign_handle(campaign)
    stored_identity = _read_json_object(
        campaign.identity_path,
        label="campaign identity",
    )
    _validate_campaign_identity_artifact(stored_identity, campaign.identity)
    _validate_campaign_root_entries(campaign)
    _validate_sessions_root(campaign.identity, campaign.sessions_root)
    present_attempts = _validate_attempt_tree(campaign)
    checkpoints = _validate_journal_tree(campaign)

    completed: list[str] = []
    pending: list[str] = []
    recovered: list[str] = []
    authenticated_slots: list[AuthenticatedFormalSlotSnapshot] = []
    for block in campaign.plan.blocks:
        for slot in block.slots:
            checkpoint = checkpoints.get(slot.slot_index)
            attempt_present = slot.logical_run_id in present_attempts
            if checkpoint is not None:
                if not attempt_present:
                    raise FormalExecutionError(
                        f"checkpoint exists without attempt1 for {slot.slot_id}"
                    )
                bundle = _validate_attempt_bundle(campaign, block, slot)
                validated_checkpoint = _validate_checkpoint(
                    campaign,
                    block,
                    slot,
                    bundle,
                    checkpoint,
                )
                completed.append(slot.slot_id)
                authenticated_slots.append(
                    bundle.authenticated_snapshot(
                        block,
                        slot,
                        checkpoint_bytes_sha256=(
                            validated_checkpoint.bytes_sha256
                        ),
                    )
                )
            elif attempt_present:
                bundle = _validate_attempt_bundle(campaign, block, slot)
                validated_checkpoint = _write_checkpoint(
                    campaign,
                    block,
                    slot,
                    bundle,
                    recovered_from_attempt_bundle=True,
                )
                completed.append(slot.slot_id)
                recovered.append(slot.slot_id)
                authenticated_slots.append(
                    bundle.authenticated_snapshot(
                        block,
                        slot,
                        checkpoint_bytes_sha256=(
                            validated_checkpoint.bytes_sha256
                        ),
                    )
                )
            else:
                pending.append(slot.slot_id)
    if pending:
        if _path_exists_lstat(
            campaign.evidence_seal_path,
            label="formal campaign evidence seal",
        ):
            raise FormalExecutionError(
                "formal campaign evidence seal exists before all 900 slots are complete"
            )
        if expected_evidence_root_sha256 is not None:
            raise FormalExecutionError(
                "cannot match an expected formal evidence root with pending slots"
            )
        return FormalCampaignState(
            completed_slot_ids=tuple(completed),
            pending_slot_ids=tuple(pending),
            recovered_slot_ids=tuple(recovered),
            authenticated_slots=tuple(authenticated_slots),
        )
    seal = seal_formal_campaign_evidence(
        campaign,
        tuple(authenticated_slots),
        expected_evidence_root_sha256=expected_evidence_root_sha256,
    )
    return FormalCampaignState(
        completed_slot_ids=tuple(completed),
        pending_slot_ids=(),
        recovered_slot_ids=tuple(recovered),
        authenticated_slots=tuple(authenticated_slots),
        evidence_id=seal.evidence_id,
        evidence_root_sha256=seal.evidence_root_sha256,
    )


def _task_for_slot(
    campaign: FormalCampaign,
    block: FormalBlock,
    slot: FormalSlot,
) -> FormalSlotTask:
    return FormalSlotTask(
        campaign_id=campaign.identity.campaign_id,
        campaign_identity_sha256=campaign.identity.sha256,
        benchmark_manifest_sha256=campaign.plan.benchmark_manifest_sha256,
        manifest_artifact_sha256=campaign.plan.manifest_artifact_sha256,
        formal_plan_id=campaign.plan.plan_id,
        formal_plan_sha256=campaign.plan.sha256,
        formal_execution_implementation_sha256=(
            campaign.plan.formal_execution_implementation_sha256
        ),
        model_id=campaign.plan.model_id,
        gurobi_version=campaign.plan.gurobi_version,
        session_id=campaign.session.session_id,
        block=block,
        slot=slot,
        pair=_pair_for_block(campaign.plan, block),
        profile=campaign.plan.suite.profile(slot.profile_code),
        telemetry=campaign.plan.suite.telemetry,
    )


def _claim_pending_slot(
    campaign: FormalCampaign,
    block: FormalBlock,
    slot: FormalSlot,
) -> tuple[AttemptClaim, FormalSlotTask, PreparedFormalInput]:
    """Revalidate current identity and atomically claim attempt 1."""

    _validate_campaign_handle(campaign)
    with campaign._claim_lock:
        if _path_exists_lstat(campaign.session_end_path, label="formal session end"):
            raise FormalExecutionError(
                "cannot claim a slot after the formal session has ended"
            )
        _read_validated_session_start(campaign.identity, campaign.session_root)
        binding = _validate_current_binding(
            campaign.plan,
            campaign.current_binding_provider,
            campaign.identity,
        )
        task = _task_for_slot(campaign, block, slot)
        try:
            prepared = _validate_prepared_input(
                task,
                campaign.pre_claim_preparer(task),
            )
        except Exception as error:
            raise FormalExecutionError(
                f"pre-claim preparation failed before attempt1 claim for "
                f"{slot.slot_id}: {type(error).__name__}: {error}"
            ) from error
        logical_root = campaign.attempts_root / slot.logical_run_id
        if _path_exists_lstat(logical_root, label="logical-run directory"):
            _require_direct_child_directory(
                logical_root,
                campaign.attempts_root,
                label="logical-run directory",
                parent_label="attempts root",
            )
        try:
            claim = claim_attempt_directory(
                campaign.attempts_root,
                slot.logical_run_id,
                attempt_number=FORMAL_ATTEMPT_NUMBER,
            )
        except FileExistsError as error:
            raise FormalExecutionError(
                f"attempt1 was concurrently or previously claimed for {slot.slot_id}"
            ) from error
        _require_direct_child_directory(
            logical_root,
            campaign.attempts_root,
            label="logical-run directory",
            parent_label="attempts root",
        )
        _require_direct_child_directory(
            claim.directory,
            logical_root,
            label="attempt directory",
            parent_label="logical-run directory",
        )
        try:
            write_canonical_json_exclusive(
                claim.directory / "formal_claim.json",
                _formal_claim_payload(
                    campaign,
                    block,
                    slot,
                    binding,
                    prepared.attestation,
                ),
            )
        except OSError as error:
            cleanup_errors: list[str] = []
            for path in (
                claim.directory / "formal_claim.json",
                claim.claim_file,
            ):
                try:
                    path.unlink(missing_ok=True)
                except OSError as cleanup_error:
                    cleanup_errors.append(f"{path.name}: {cleanup_error}")
            try:
                claim.directory.rmdir()
                claim.directory.parent.rmdir()
            except OSError as cleanup_error:
                cleanup_errors.append(f"attempt directory: {cleanup_error}")
            cleanup_suffix = (
                "; rollback failed: " + "; ".join(cleanup_errors)
                if cleanup_errors
                else ""
            )
            raise FormalExecutionError(
                f"cannot attest formal attempt1 claim for {slot.slot_id}: {error}"
                f"{cleanup_suffix}"
            ) from error
        return claim, task, prepared


def _execute_block_with_completed(
    campaign: FormalCampaign,
    block: FormalBlock,
    execute_slot: SlotExecutionCallback,
    completed_slot_ids: frozenset[str],
) -> FormalBlockExecutionResult:
    if campaign.plan.blocks[block.block_index] != block:
        raise FormalExecutionError("block is not the canonical block at its plan index")
    executed: list[str] = []
    skipped: list[str] = []
    for slot in block.slots:
        if slot.slot_id in completed_slot_ids:
            skipped.append(slot.slot_id)
            continue
        claim, task, prepared = _claim_pending_slot(campaign, block, slot)
        try:
            execute_slot(task, claim, prepared)
        except Exception as error:
            raise FormalExecutionError(
                f"injected slot execution failed after attempt1 claim for "
                f"{slot.slot_id}: {type(error).__name__}: {error}"
            ) from error
        bundle = _validate_attempt_bundle(campaign, block, slot)
        _write_checkpoint(
            campaign,
            block,
            slot,
            bundle,
            recovered_from_attempt_bundle=False,
        )
        executed.append(slot.slot_id)
    return FormalBlockExecutionResult(
        block_id=block.block_id,
        executed_slot_ids=tuple(executed),
        skipped_slot_ids=tuple(skipped),
    )


def execute_formal_block(
    campaign: FormalCampaign,
    block: FormalBlock,
    execute_slot: SlotExecutionCallback,
    current_binding_provider: CurrentBindingProvider,
) -> FormalBlockExecutionResult:
    """Reconcile then serially execute only pending slots in one formal block."""

    _validate_campaign_handle(campaign)
    if current_binding_provider is not campaign.current_binding_provider:
        raise FormalExecutionError(
            "execute_formal_block must reuse the campaign binding provider"
        )
    state = reconcile_formal_campaign(campaign)
    return _execute_block_with_completed(
        campaign,
        block,
        execute_slot,
        frozenset(state.completed_slot_ids),
    )


def _run_concurrent_scale(
    blocks: tuple[FormalBlock, ...],
    callback: BlockCallback,
    *,
    max_workers: int,
    executor_factory: ExecutorFactory,
) -> dict[int, object]:
    results: dict[int, object] = {}
    iterator = iter(blocks)
    with executor_factory(max_workers=max_workers) as executor:
        active: dict[Future[object], FormalBlock] = {}

        def submit_next() -> bool:
            try:
                block = next(iterator)
            except StopIteration:
                return False
            active[executor.submit(callback, block)] = block
            return True

        for _ in range(min(max_workers, len(blocks))):
            submit_next()
        while active:
            done, _not_done = wait(tuple(active), return_when=FIRST_COMPLETED)
            ordered_done = sorted(done, key=lambda future: active[future].block_index)
            for future in ordered_done:
                block = active.pop(future)
                try:
                    results[block.block_index] = future.result()
                except BaseException:
                    for pending in active:
                        pending.cancel()
                    raise
                submit_next()
    return results


def run_formal_block_schedule(
    plan: FormalExecutionPlan,
    callback: BlockCallback,
    *,
    executor_factory: ExecutorFactory = ThreadPoolExecutor,
) -> tuple[object, ...]:
    """Run scale barriers with block concurrency 2/1/1 in canonical order."""

    if not isinstance(plan, FormalExecutionPlan):
        raise FormalExecutionError("plan must be a FormalExecutionPlan")
    results: dict[int, object] = {}
    for contract in FORMAL_SCALE_CONTRACTS:
        blocks = tuple(block for block in plan.blocks if block.scale == contract.scale)
        if contract.max_concurrent_blocks == 1:
            for block in blocks:
                results[block.block_index] = callback(block)
        else:
            results.update(
                _run_concurrent_scale(
                    blocks,
                    callback,
                    max_workers=contract.max_concurrent_blocks,
                    executor_factory=executor_factory,
                )
            )
    return tuple(results[index] for index in range(len(plan.blocks)))


def execute_formal_campaign(
    plan: FormalExecutionPlan,
    results_root: str | Path,
    current_binding_provider: CurrentBindingProvider,
    pre_claim_preparer: PreClaimPreparer,
    execute_slot: SlotExecutionCallback,
    *,
    session_snapshot: SessionSnapshot,
    resource_lock_path: str,
    now_utc: TimestampProvider = _utc_now,
    session_nonce_provider: SessionNonceProvider = _session_nonce,
    executor_factory: ExecutorFactory = ThreadPoolExecutor,
) -> FormalCampaignRunResult:
    """Resume and execute an entire plan under its immutable schedule."""

    campaign = prepare_formal_campaign(
        plan,
        results_root,
        current_binding_provider,
        pre_claim_preparer,
        session_snapshot=session_snapshot,
        resource_lock_path=resource_lock_path,
        now_utc=now_utc,
        session_nonce_provider=session_nonce_provider,
    )
    try:
        initial = reconcile_formal_campaign(campaign)
        completed = frozenset(initial.completed_slot_ids)

        def execute_block(block: FormalBlock) -> FormalBlockExecutionResult:
            return _execute_block_with_completed(
                campaign,
                block,
                execute_slot,
                completed,
            )

        raw_results = run_formal_block_schedule(
            plan,
            execute_block,
            executor_factory=executor_factory,
        )
        block_results = tuple(
            result
            for result in raw_results
            if isinstance(result, FormalBlockExecutionResult)
        )
        if len(block_results) != len(plan.blocks):
            raise FormalExecutionError(
                "formal block scheduler returned incomplete results"
            )
        final = reconcile_formal_campaign(campaign)
        if not final.complete:
            raise FormalExecutionError(
                f"formal execution ended with {len(final.pending_slot_ids)} pending slots"
            )
        executed_count = sum(
            len(result.executed_slot_ids) for result in block_results
        )
        skipped_count = sum(
            len(result.skipped_slot_ids) for result in block_results
        )
    except BaseException as execution_error:
        try:
            finish_formal_session(campaign, status="failed", now_utc=now_utc)
        except Exception as session_error:
            raise FormalExecutionError(
                "formal execution failed and its session end could not be published: "
                f"{type(session_error).__name__}: {session_error}"
            ) from execution_error
        raise
    finish_formal_session(campaign, status="completed", now_utc=now_utc)
    return FormalCampaignRunResult(
        campaign_id=campaign.identity.campaign_id,
        session_id=campaign.session.session_id,
        initial_completed_count=len(initial.completed_slot_ids),
        recovered_count=len(initial.recovered_slot_ids),
        executed_count=executed_count,
        skipped_count=skipped_count,
        final_completed_count=len(final.completed_slot_ids),
        evidence_id=final.evidence_id,
        evidence_root_sha256=final.evidence_root_sha256,
    )


__all__ = [
    "FORMAL_ATTEMPT_ATTESTATION_ARTIFACT_TYPE",
    "FORMAL_ATTEMPT_ATTESTATION_SCHEMA_VERSION",
    "FORMAL_CAMPAIGN_ARTIFACT_TYPE",
    "FORMAL_CAMPAIGN_EVIDENCE_ARTIFACT_TYPE",
    "FORMAL_CAMPAIGN_EVIDENCE_FILENAME",
    "FORMAL_CAMPAIGN_EVIDENCE_HASH_DOMAIN",
    "FORMAL_CAMPAIGN_EVIDENCE_SCHEMA_VERSION",
    "FORMAL_CAMPAIGN_SCHEMA_VERSION",
    "FORMAL_SESSION_END_ARTIFACT_TYPE",
    "FORMAL_SESSION_SCHEMA_VERSION",
    "FORMAL_SESSION_START_ARTIFACT_TYPE",
    "FORMAL_SLOT_CHECKPOINT_ARTIFACT_TYPE",
    "FORMAL_SLOT_CHECKPOINT_SCHEMA_VERSION",
    "FORMAL_SLOT_BUNDLE_HASH_DOMAIN",
    "BlockCallback",
    "AuthenticatedFormalSlotSnapshot",
    "CurrentBindingProvider",
    "ExecutorFactory",
    "FormalAttemptBundle",
    "FormalBlockExecutionResult",
    "FormalCampaign",
    "FormalCampaignEvidenceSeal",
    "FormalCampaignIdentity",
    "FormalCampaignRunResult",
    "FormalCampaignState",
    "FormalExecutionError",
    "FormalExecutionBinding",
    "FormalSlotTask",
    "FormalInputAttestation",
    "FormalRunSession",
    "PreparedFormalInput",
    "PreClaimPreparer",
    "SessionNonceProvider",
    "SlotExecutionCallback",
    "TimestampProvider",
    "execute_formal_block",
    "execute_formal_campaign",
    "finish_formal_session",
    "prepare_formal_campaign",
    "reconcile_formal_campaign",
    "run_formal_block_schedule",
    "seal_formal_campaign_evidence",
]
