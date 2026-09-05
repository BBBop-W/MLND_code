"""One-command orchestration for the paper-grade branch-and-cut benchmark.

The CLI deliberately keeps model construction behind command handlers.  Merely
asking for ``--help``, freezing artifacts, or analysing completed runs does not
import Gurobi.  Every persistent artifact is written exclusively so reruns
cannot silently overwrite the experiment record.
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import stat
from statistics import median
import tempfile
from typing import Any, Callable, Mapping, Sequence

from .analysis import (
    EffectEstimate,
    PairAssessment,
    PreregisteredMetric,
    assess_pair,
    evaluate_success_gate,
    honest_termination_label,
    identify_systematic_regressions,
    shifted_geomean_gap_reduction,
    shifted_geomean_speedup,
    winner_counts,
)
from .artifacts import (
    ATOMIC_WRITE_TEMP_PREFIX,
    claim_attempt_directory,
    write_canonical_json_exclusive,
)
from .case_builder import (
    CANONICAL_INSTANCE_FILES,
    QUALIFICATION_MODEL_IDENTITY_SCHEMA,
    CaseBuildConfig,
    LOCKED_QUALIFICATION_PROFILES,
    QualificationSolveProfile,
    build_locked_candidate_pool,
    canonical_instance_hash,
    qualification_case_sha256,
)
from .execution import ProfileExecutionResult, execute_profile
from .formal_environment import (
    FormalEnvironmentCapture,
    FormalEnvironmentError,
    capture_formal_environment,
)
from .formal_execution import (
    AuthenticatedFormalSlotSnapshot,
    FormalCampaign,
    FormalCampaignRunResult,
    FormalCampaignState,
    FormalExecutionError,
    FormalExecutionBinding,
    FormalInputAttestation,
    FormalSlotTask,
    PreparedFormalInput,
    execute_formal_campaign,
    finish_formal_session,
    prepare_formal_campaign,
    reconcile_formal_campaign,
)
from .formal_plan import (
    FORMAL_AGGREGATE_GATE_POLICY,
    FORMAL_AGGREGATE_SIGNIFICANCE_POLICY,
    FORMAL_ATTEMPT_NUMBER,
    FORMAL_BLOCK_COUNT,
    FORMAL_BOOTSTRAP_METHOD,
    FORMAL_BOOTSTRAP_SAMPLING_UNIT,
    FORMAL_BOOTSTRAP_STRATIFICATION_POLICY,
    FORMAL_COMMON_TARGET_POLICY,
    FORMAL_HORIZON_POLICY,
    FORMAL_MISSING_PROGRESS_POLICY,
    FORMAL_MEDIAN_SUMMARY_POLICY,
    FORMAL_PERFORMANCE_PROFILE_DENOMINATOR_POLICY,
    FORMAL_PERFORMANCE_PROFILE_POLICY,
    FORMAL_PREPARATION_TIME_POLICY,
    FORMAL_STATUS_RECONCILIATION_POLICY,
    FORMAL_SLOT_COUNT,
    FORMAL_SYSTEMATIC_REGRESSION_POLICY,
    FORMAL_TRAJECTORY_POLICY_VERSION,
    FORMAL_RUNTIME_CENSORING_POLICY,
    FORMAL_RUNTIME_EFFECT_POLICY,
    FORMAL_SIGN_TEST_SAMPLING_UNIT,
    FormalAnalysisContract,
    FormalExecutionPlan,
    build_formal_execution_plan,
    formal_plan_from_artifact,
)
from .formal_solution_evidence import (
    FormalQualificationTrace,
    bind_qualification_trace,
)
from .hashing import (
    build_logical_run_id,
    build_model_id,
    build_pair_id,
    canonical_json,
    canonical_sha256,
)
from .manifest import (
    BenchmarkManifest,
    FreezeMetadata,
    FrozenCase,
    OVERCOMPLETE_SELECTION_RULE,
    ReplacementSelectionProtocol,
    SELECTION_RULE,
    SelectionLedgerEntry,
    freeze_manifest,
)
from .pool import CandidateCase, CandidatePool
from .profiles import build_paper_profile_suite
from .qualification import (
    QualificationDecision,
    QualificationEvidence,
    QualificationPolicy,
    assess_qualification,
)
from .qualification_geometry_start import (
    GUROBI_MAX_START_NODE_LIMIT,
    GeometryStartAttemptAudit,
    GeometryStartPolicy,
)
from .qualification_runner import (
    NORMAL_QUALIFICATION_RUNNER_REASONS,
    NORMAL_QUALIFICATION_SOLVER_STATUSES,
    NativeQualificationSettings,
    QualificationModelAudit,
    QualificationModelContract,
    QualificationPhaseTimings,
    QualificationRunnerOutcome,
    QualificationRunnerReason,
    qualify_canonical_instance,
)
from .resource_lock import (
    DEFAULT_BENCHMARK_RESOURCE_LOCK,
    BenchmarkResourceLock,
    BenchmarkResourceLockError,
)
from .schema import (
    BenchmarkMode,
    ModelIdentity,
    PairSpec,
    ProfileCode,
    ProfileConfig,
    SolverConfig,
    StartConfig,
    StartMode,
    TelemetryConfig,
)
from .statistics import (
    CommonTargetTimes,
    ClusterBootstrapEffectEstimate,
    PairedCaseObservation,
    PrimalDualIntegral,
    ProgressTrajectory,
    case_cluster_bootstrap_effect,
    cluster_paired_observations,
    dolan_more_profile_points,
    dolan_more_ratios,
    exact_sign_test,
    paired_censored_time_input,
    primal_dual_integral,
    solver_status_name,
    terminal_best_bound,
    terminal_best_incumbent,
    time_to_common_target,
    time_to_target,
    trajectory_from_progress_events,
)


class CLIError(RuntimeError):
    """A concise user-facing orchestration or artifact error."""


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    )


def _candidate_solve_profile(candidate: CandidateCase) -> QualificationSolveProfile:
    """Return the model parameters pinned by one logical candidate."""

    raw_profile = candidate.generator_parameters.get("qualification_solve_profile")
    if not isinstance(raw_profile, Mapping):
        raise CLIError(
            f"candidate {candidate.case_id!r} is missing qualification_solve_profile"
        )
    try:
        profile = QualificationSolveProfile(**dict(raw_profile))
    except (TypeError, ValueError) as error:
        raise CLIError(
            f"candidate {candidate.case_id!r} has an invalid "
            f"qualification_solve_profile: {error}"
        ) from error
    locked_profile = next(
        (
            item
            for item in LOCKED_QUALIFICATION_PROFILES
            if item.profile_id == profile.profile_id
        ),
        None,
    )
    if profile != locked_profile:
        raise CLIError(
            f"candidate {candidate.case_id!r} does not use one of the locked "
            "v5 qualification profiles"
        )
    return profile


def _candidate_model_contract(candidate: CandidateCase) -> QualificationModelContract:
    """Recompute and authenticate one candidate's complete model identity."""

    profile = _candidate_solve_profile(candidate)
    raw_identity = candidate.generator_parameters.get("qualification_model_identity")
    if not isinstance(raw_identity, Mapping):
        raise CLIError(
            f"candidate {candidate.case_id!r} is missing qualification_model_identity"
        )
    expected_identity = {
        "schema": QUALIFICATION_MODEL_IDENTITY_SCHEMA,
        "effective_model_config_sha256": profile.effective_model_config_sha256(),
        "qualification_case_sha256": qualification_case_sha256(
            candidate.instance_sha256,
            profile,
        ),
        "expected_screening_is_exact": profile.expected_screening_is_exact,
    }
    if dict(raw_identity) != expected_identity:
        raise CLIError(
            f"candidate {candidate.case_id!r} has an inconsistent "
            "qualification_model_identity"
        )
    try:
        return QualificationModelContract(
            qualification_case_sha256=expected_identity["qualification_case_sha256"],
            effective_model_config_sha256=expected_identity[
                "effective_model_config_sha256"
            ],
            demand_scale=profile.demand_scale,
            locker_fixed_cost_per_hour=profile.locker_fixed_cost_per_hour,
            display_origin_choices=profile.display_origin_choices,
            general_corridor_choices=profile.general_corridor_choices,
            screening_is_exact=profile.expected_screening_is_exact,
        )
    except (TypeError, ValueError) as error:
        raise CLIError(
            f"candidate {candidate.case_id!r} has an invalid qualification "
            f"model contract: {error}"
        ) from error


def _default_instance_loader(
    instance_dir: Path,
    candidate: CandidateCase,
) -> object:
    from src.data_processor.loader import DataLoader
    from src.data_processor.instance import Instance

    contract = _candidate_model_contract(candidate)
    profile = _candidate_solve_profile(candidate)
    model_config = profile.effective_model_config()
    instance = DataLoader().load_instance_dir(
        str(instance_dir),
        model_config=model_config,
        demand_scale=profile.demand_scale,
    )
    if not isinstance(instance, Instance):
        return instance
    if instance.config != model_config:
        raise CLIError(
            f"loaded instance for {candidate.case_id!r} does not retain the "
            "qualification model config"
        )
    observed_demand_scale = instance.metadata.get("demand_scale")
    if (
        isinstance(observed_demand_scale, bool)
        or not isinstance(observed_demand_scale, (int, float))
        or float(observed_demand_scale) != profile.demand_scale
    ):
        raise CLIError(
            f"loaded instance for {candidate.case_id!r} does not retain the "
            "qualification demand scale"
        )
    observed_screening = {
        mode: instance.screening_is_exact(mode) for mode in ("MX", "HD", "LO")
    }
    if any(
        observed != contract.screening_is_exact
        for observed in observed_screening.values()
    ):
        raise CLIError(
            f"loaded instance for {candidate.case_id!r} has screening exactness "
            f"{observed_screening}, expected {contract.screening_is_exact}"
        )
    return instance


_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
_MODEL_IMPLEMENTATION_FILES = (
    Path("MLND/main.tex"),
    Path("src/config.py"),
    Path("src/model_contract.py"),
    Path("src/data_processor/loader.py"),
    Path("src/data_processor/instance.py"),
    Path("src/model/gurobi_model.py"),
    Path("src/model/primal_heuristic.py"),
    Path("src/model/valid_inequalities.py"),
)
_QUALIFICATION_IMPLEMENTATION_FILES = (
    Path("src/benchmark/branch_cut/qualification.py"),
    Path("src/benchmark/branch_cut/qualification_geometry_start.py"),
    Path("src/benchmark/branch_cut/qualification_runner.py"),
    Path("src/benchmark/branch_cut/qualification_incumbent_reuse.py"),
    Path("src/benchmark/branch_cut/case_builder.py"),
    Path("src/benchmark/branch_cut/pool.py"),
    Path("src/benchmark/branch_cut/manifest.py"),
    Path("src/benchmark/branch_cut/hashing.py"),
    Path("src/benchmark/branch_cut/artifacts.py"),
    Path("src/benchmark/branch_cut/resource_lock.py"),
    Path("src/benchmark/branch_cut/cli.py"),
)
_FORMAL_EXECUTION_IMPLEMENTATION_FILES = (
    Path("src/benchmark/branch_cut/artifacts.py"),
    Path("src/benchmark/branch_cut/case_builder.py"),
    Path("src/benchmark/branch_cut/cli.py"),
    Path("src/benchmark/branch_cut/execution.py"),
    Path("src/benchmark/branch_cut/formal_environment.py"),
    Path("src/benchmark/branch_cut/formal_execution.py"),
    Path("src/benchmark/branch_cut/formal_metrics.py"),
    Path("src/benchmark/branch_cut/formal_plan.py"),
    Path("src/benchmark/branch_cut/formal_solution_evidence.py"),
    Path("src/benchmark/branch_cut/hashing.py"),
    Path("src/benchmark/branch_cut/manifest.py"),
    Path("src/benchmark/branch_cut/profiles.py"),
    Path("src/benchmark/branch_cut/resource_lock.py"),
    Path("src/benchmark/branch_cut/schema.py"),
    Path("src/benchmark/branch_cut/telemetry.py"),
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _current_model_identity() -> ModelIdentity:
    from src.config import ModelConfig, to_dict
    from src.model_contract import FORMULATION_ID, RESULT_SCHEMA_VERSION

    implementation = {
        path.as_posix(): _sha256_file(_REPOSITORY_ROOT / path)
        for path in _MODEL_IMPLEMENTATION_FILES
    }
    return ModelIdentity(
        formulation_name=FORMULATION_ID,
        formulation_schema_version=RESULT_SCHEMA_VERSION,
        implementation_sha256=canonical_sha256(implementation),
        model_config_sha256=canonical_sha256(to_dict(ModelConfig())),
    )


def _current_qualification_implementation_sha256() -> str:
    implementation = {
        path.as_posix(): _sha256_file(_REPOSITORY_ROOT / path)
        for path in _QUALIFICATION_IMPLEMENTATION_FILES
    }
    return canonical_sha256(implementation)


def _current_formal_execution_implementation_sha256() -> str:
    implementation = {
        path.as_posix(): _sha256_file(_REPOSITORY_ROOT / path)
        for path in _FORMAL_EXECUTION_IMPLEMENTATION_FILES
    }
    return canonical_sha256(implementation)


def _current_gurobi_version() -> str:
    import gurobipy as gp

    version = gp.gurobi.version()
    return ".".join(str(component) for component in version)


def _preload_formal_solver_runtime() -> None:
    """Import every frozen solver module before the first formal claim.

    The formal implementation digest is checked again after this preload.  Any
    later source edit therefore cannot silently alter a lazily imported module
    used by an already-claimed attempt.
    """

    from src import config as _config  # noqa: F401
    from src import model_contract as _model_contract  # noqa: F401
    from src.data_processor import instance as _instance  # noqa: F401
    from src.data_processor import loader as _loader  # noqa: F401
    from src.model import gurobi_model as _gurobi_model  # noqa: F401
    from src.model import primal_heuristic as _primal_heuristic  # noqa: F401
    from src.model import valid_inequalities as _valid_inequalities  # noqa: F401


@dataclass(frozen=True, slots=True)
class _CanonicalInstanceSnapshot:
    """Exact in-memory bytes used to construct one formal model input."""

    ordered_files: tuple[tuple[str, bytes], ...]
    file_sha256: tuple[tuple[str, str], ...]
    instance_sha256: str


def _snapshot_instance_bytes(instance_dir: Path) -> _CanonicalInstanceSnapshot:
    """Read the five canonical files without following links."""

    ordered_files: list[tuple[str, bytes]] = []
    file_sha256: list[tuple[str, str]] = []
    for filename in CANONICAL_INSTANCE_FILES:
        path = instance_dir / filename
        try:
            mode = path.lstat().st_mode
        except OSError as error:
            raise CLIError(
                f"cannot inspect formal instance file {path}: {error}"
            ) from error
        if path.is_symlink() or not stat.S_ISREG(mode):
            raise CLIError(
                f"formal instance file must be a regular non-symlink: {path}"
            )
        try:
            payload = path.read_bytes()
        except OSError as error:
            raise CLIError(
                f"cannot snapshot formal instance file {path}: {error}"
            ) from error
        digest = hashlib.sha256(payload).hexdigest()
        ordered_files.append((filename, payload))
        file_sha256.append((filename, digest))
    aggregate = canonical_sha256(
        {
            "hash_schema": "mlnd-canonical-instance-files-v1",
            "ordered_files": [
                {"name": filename, "sha256": digest} for filename, digest in file_sha256
            ],
        }
    )
    return _CanonicalInstanceSnapshot(
        ordered_files=tuple(ordered_files),
        file_sha256=tuple(file_sha256),
        instance_sha256=aggregate,
    )


def _load_snapshot_instance(
    snapshot: _CanonicalInstanceSnapshot,
    candidate: CandidateCase,
    loader: Callable[[Path, CandidateCase], object],
) -> object:
    """Parse only the captured bytes in a private temporary directory."""

    with tempfile.TemporaryDirectory(prefix="mlnd-formal-input-") as temporary:
        snapshot_root = Path(temporary)
        for filename, payload in snapshot.ordered_files:
            target = snapshot_root / filename
            target.write_bytes(payload)
            target.chmod(0o400)
        before_load, _ = canonical_instance_hash(snapshot_root)
        if before_load != snapshot.instance_sha256:
            raise CLIError("private formal input snapshot changed before parsing")
        instance = loader(snapshot_root, candidate)
        after_load, _ = canonical_instance_hash(snapshot_root)
        if after_load != snapshot.instance_sha256:
            raise CLIError("private formal input snapshot changed while parsing")
        metadata = getattr(instance, "metadata", None)
        if isinstance(metadata, dict):
            metadata["instance_dir"] = f"formal-snapshot:{candidate.instance_ref}"
            metadata["instance_sha256"] = candidate.instance_sha256
        return instance


def _prepare_formal_input_snapshot(
    task: FormalSlotTask,
    *,
    instance_dir: Path,
    candidate: CandidateCase,
    qualification_trace: FormalQualificationTrace,
    canonical_hash: Callable[[Path], tuple[str, Mapping[str, str]]],
    loader: Callable[[Path, CandidateCase], object],
) -> PreparedFormalInput:
    """Hash, snapshot, parse, and re-hash an input before attempt claim."""

    if (
        candidate.case_id != task.block.case_id
        or candidate.instance_ref != task.block.instance_ref
        or candidate.instance_sha256 != task.block.instance_sha256
    ):
        raise CLIError("formal task instance identity differs from the frozen plan")
    source_before, _ = canonical_hash(instance_dir)
    snapshot = _snapshot_instance_bytes(instance_dir)
    payload = _load_snapshot_instance(snapshot, candidate, loader)
    source_after, _ = canonical_hash(instance_dir)
    expected = task.block.instance_sha256
    observed = {
        "source before": source_before,
        "payload": snapshot.instance_sha256,
        "source after": source_after,
    }
    differing = [label for label, digest in observed.items() if digest != expected]
    if differing:
        raise CLIError(
            f"formal instance drift for {task.block.case_id}: " + ", ".join(differing)
        )
    return PreparedFormalInput(
        payload=payload,
        attestation=FormalInputAttestation(
            case_id=task.block.case_id,
            slot_id=task.slot.slot_id,
            expected_instance_sha256=expected,
            payload_instance_sha256=snapshot.instance_sha256,
            source_sha256_before=source_before,
            source_sha256_after=source_after,
            qualification_trace=qualification_trace,
        ),
    )


@dataclass(frozen=True, slots=True)
class CLIHooks:
    """Dependency injection points used by solver-free orchestration tests."""

    build_pool: Callable[..., object] = build_locked_candidate_pool
    canonical_hash: Callable[[Path], tuple[str, Mapping[str, str]]] = (
        canonical_instance_hash
    )
    load_instance: Callable[[Path, CandidateCase], object] = _default_instance_loader
    formal_input_preparer: Callable[..., PreparedFormalInput] | None = None
    qualify_instance: Callable[..., QualificationRunnerOutcome] = (
        qualify_canonical_instance
    )
    freeze: Callable[..., BenchmarkManifest] = freeze_manifest
    execute: Callable[..., ProfileExecutionResult] = execute_profile
    model_identity: Callable[[], ModelIdentity] = _current_model_identity
    qualification_implementation_sha256: Callable[[], str] = (
        _current_qualification_implementation_sha256
    )
    formal_execution_implementation_sha256: Callable[[], str] = (
        _current_formal_execution_implementation_sha256
    )
    execute_formal: Callable[..., FormalCampaignRunResult] = execute_formal_campaign
    prepare_formal: Callable[..., FormalCampaign] = prepare_formal_campaign
    reconcile_formal: Callable[..., FormalCampaignState] = reconcile_formal_campaign
    gurobi_version: Callable[[], str] = _current_gurobi_version
    formal_environment_capture: Callable[[], FormalEnvironmentCapture] = (
        capture_formal_environment
    )
    preload_formal_runtime: Callable[[], None] = _preload_formal_solver_runtime
    now_utc: Callable[[], str] = _utc_now


DEFAULT_HOOKS = CLIHooks()


@dataclass(frozen=True, slots=True)
class LoadedManifest:
    manifest: BenchmarkManifest
    identity_sha256: str
    qualification_mode: str | None
    model_identity: ModelIdentity
    model_id: str

    @property
    def manifest_id(self) -> str:
        return f"manifest-{self.identity_sha256}"


@dataclass(frozen=True, slots=True)
class QualificationTask:
    candidate: CandidateCase
    instance_dir: Path
    mode: str
    qualification_run_id: str
    model_identity: ModelIdentity
    qualification_implementation_sha256: str
    gurobi_version: str
    policy: QualificationPolicy
    settings: NativeQualificationSettings
    geometry_start_policy: GeometryStartPolicy = GeometryStartPolicy()


@dataclass(frozen=True, slots=True)
class QualificationTaskResult:
    case_id: str
    outcome: QualificationRunnerOutcome | None
    error: str | None


@dataclass(frozen=True, slots=True)
class _QualificationStratum:
    stratum_id: str
    target_accepted: int
    candidates_by_rank: tuple[CandidateCase, ...]


def _selection_int(value: Any, *, field_name: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise CLIError(f"{field_name} must be an integer >= {minimum}")
    return value


def _candidate_selection_contract(
    candidate: CandidateCase,
) -> tuple[str, int, int, int] | None:
    raw_selection = candidate.generator_parameters.get("pre_registered_selection")
    if raw_selection is None:
        return None
    if not isinstance(raw_selection, Mapping):
        raise CLIError(
            f"candidate {candidate.case_id!r} has invalid pre_registered_selection"
        )
    if raw_selection.get("selection_rule") != OVERCOMPLETE_SELECTION_RULE:
        raise CLIError(
            f"candidate {candidate.case_id!r} has an unsupported selection rule"
        )
    if raw_selection.get("selection_uses_algorithm_performance") is not False:
        raise CLIError(
            f"candidate {candidate.case_id!r} does not certify "
            "performance-blind selection"
        )
    if raw_selection.get("forced_opening_constraints_added") is not False:
        raise CLIError(f"candidate {candidate.case_id!r} may not force facilities open")
    expected_large_target = [50, 60] if candidate.scale == "large" else None
    if (
        raw_selection.get("large_scale_open_h_qualification_target")
        != expected_large_target
    ):
        raise CLIError(
            f"candidate {candidate.case_id!r} has the wrong large-scale "
            "H-open qualification target"
        )
    _candidate_model_contract(candidate)
    stratum_id = raw_selection.get("stratum_id")
    if stratum_id != f"{candidate.demand_block_id}__{candidate.scale}":
        raise CLIError(
            f"candidate {candidate.case_id!r} has an invalid selection stratum"
        )
    return (
        stratum_id,
        _selection_int(
            raw_selection.get("replacement_rank_within_stratum"),
            field_name="replacement_rank_within_stratum",
            minimum=0,
        ),
        _selection_int(
            raw_selection.get("target_cases_per_stratum"),
            field_name="target_cases_per_stratum",
            minimum=1,
        ),
        _selection_int(
            raw_selection.get("expected_stratum_count"),
            field_name="expected_stratum_count",
            minimum=1,
        ),
    )


def _qualification_strata(
    pool: CandidatePool,
) -> tuple[_QualificationStratum, ...] | None:
    """Validate and materialize the registered replacement ladders."""

    ordered = pool.ordered_candidates()
    contracts = tuple(_candidate_selection_contract(item) for item in ordered)
    if not any(contract is not None for contract in contracts):
        return None
    if not all(contract is not None for contract in contracts):
        raise CLIError(
            "candidate pool may not mix legacy and overcomplete selection contracts"
        )

    by_stratum: dict[str, list[tuple[int, CandidateCase]]] = {}
    targets: set[int] = set()
    expected_counts: set[int] = set()
    for candidate, contract in zip(ordered, contracts, strict=True):
        assert contract is not None
        stratum_id, replacement_rank, target, expected_count = contract
        by_stratum.setdefault(stratum_id, []).append((replacement_rank, candidate))
        targets.add(target)
        expected_counts.add(expected_count)
    if len(targets) != 1 or len(expected_counts) != 1:
        raise CLIError("overcomplete selection metadata is inconsistent across cases")
    target = next(iter(targets))
    expected_count = next(iter(expected_counts))
    if len(by_stratum) != expected_count:
        raise CLIError(
            f"pool has {len(by_stratum)} selection strata; expected {expected_count}"
        )

    strata: list[_QualificationStratum] = []
    for stratum_id, ranked_candidates in by_stratum.items():
        ranked = sorted(ranked_candidates, key=lambda item: item[0])
        ranks = [rank for rank, _candidate in ranked]
        if ranks != list(range(len(ranked))):
            raise CLIError(
                f"replacement ranks for stratum {stratum_id!r} must be "
                "contiguous from zero"
            )
        if len(ranked) < target:
            raise CLIError(
                f"stratum {stratum_id!r} has fewer candidates than its target"
            )
        strata.append(
            _QualificationStratum(
                stratum_id=stratum_id,
                target_accepted=target,
                candidates_by_rank=tuple(candidate for _rank, candidate in ranked),
            )
        )
    return tuple(strata)


@dataclass(frozen=True, slots=True)
class RunTask:
    candidate: CandidateCase
    instance_dir: Path
    model_identity: ModelIdentity
    results_root: Path
    pair: PairSpec
    profile: ProfileConfig
    telemetry: TelemetryConfig
    attempt_number: int | None

    @property
    def case_id(self) -> str:
        return self.candidate.case_id

    @property
    def scale(self) -> str:
        return self.candidate.scale

    @property
    def expected_instance_sha256(self) -> str:
        return self.candidate.instance_sha256


def _read_json_object(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise CLIError(f"artifact does not exist: {source}") from error
    except json.JSONDecodeError as error:
        raise CLIError(f"artifact is not valid JSON: {source}: {error}") from error
    if not isinstance(payload, dict):
        raise CLIError(f"artifact must contain one JSON object: {source}")
    return payload


def _require_absent_output(path: str | Path, *, label: str) -> Path:
    target = Path(path).expanduser().resolve()
    if target.exists():
        raise FileExistsError(
            f"{label} already exists; refusing to overwrite: {target}"
        )
    return target


def _candidate_from_payload(payload: Mapping[str, Any]) -> CandidateCase:
    fields = {
        key: payload[key]
        for key in (
            "case_id",
            "scale",
            "demand_block_id",
            "layout_seed",
            "instance_ref",
            "instance_sha256",
            "generator_parameters",
        )
        if key in payload
    }
    return CandidateCase(**fields)


def load_candidate_pool(path: str | Path) -> CandidatePool:
    """Load and revalidate either a bare pool payload or ``pool.json``."""

    artifact = _read_json_object(path)
    payload = artifact.get("pool", artifact)
    if not isinstance(payload, Mapping):
        raise CLIError("candidate-pool artifact has no object-valued 'pool'")
    candidates_payload = payload.get("candidates")
    if not isinstance(candidates_payload, list):
        raise CLIError("candidate-pool payload must contain a candidates list")
    pool = CandidatePool(
        candidates=tuple(_candidate_from_payload(item) for item in candidates_payload),
        candidate_seed_order=tuple(payload.get("candidate_seed_order", ())),
        pool_label=str(payload.get("pool_label", "")),
        schema_version=int(payload.get("schema_version", 0)),
    )
    reported_hash = artifact.get("pool_sha256")
    if reported_hash is not None and reported_hash != pool.sha256:
        raise CLIError("candidate-pool SHA256 does not match its canonical payload")
    return pool


def _policy_from_payload(payload: Mapping[str, Any]) -> QualificationPolicy:
    values = dict(payload)
    target = values.get("large_scale_open_h_target")
    if target is not None:
        values["large_scale_open_h_target"] = tuple(target)
    return QualificationPolicy(**values)


def _decision_from_payload(payload: Mapping[str, Any]) -> QualificationDecision:
    try:
        values = dict(payload)
        policy = values.get("policy")
        if not isinstance(policy, Mapping):
            raise CLIError("qualification decision is missing its policy")
        values["policy"] = _policy_from_payload(policy)
        values["rejection_reasons"] = tuple(values.get("rejection_reasons", ()))
        return QualificationDecision(**values)
    except CLIError:
        raise
    except (TypeError, ValueError) as error:
        raise CLIError(f"invalid qualification decision: {error}") from error


def _replacement_protocol_from_payload(
    payload: Any,
) -> ReplacementSelectionProtocol | None:
    if payload is None:
        return None
    if not isinstance(payload, Mapping):
        raise CLIError("replacement_selection_protocol must be an object or null")
    try:
        return ReplacementSelectionProtocol(**dict(payload))
    except (TypeError, ValueError) as error:
        raise CLIError(f"invalid replacement selection protocol: {error}") from error


def _selection_entry_from_payload(payload: Any) -> SelectionLedgerEntry:
    if not isinstance(payload, Mapping):
        raise CLIError("selection_ledger entries must be objects")
    values = dict(payload)
    values["rejection_reasons"] = tuple(values.get("rejection_reasons", ()))
    try:
        return SelectionLedgerEntry(**values)
    except (TypeError, ValueError) as error:
        raise CLIError(f"invalid selection-ledger entry: {error}") from error


def load_qualification_ledger(
    path: str | Path,
) -> tuple[dict[str, Any], tuple[QualificationDecision, ...]]:
    artifact = _read_json_object(path)
    raw_decisions = artifact.get("qualification_ledger")
    if not isinstance(raw_decisions, list):
        raise CLIError("qualification artifact must contain qualification_ledger")
    return artifact, tuple(_decision_from_payload(item) for item in raw_decisions)


def load_benchmark_manifest(path: str | Path) -> LoadedManifest:
    """Reconstruct and validate a frozen manifest plus its CLI wrapper identity."""

    artifact = _read_json_object(path)
    payload: Mapping[str, Any]
    raw_manifest = artifact.get("manifest")
    if isinstance(raw_manifest, Mapping):
        payload = raw_manifest
    else:
        payload = artifact
    try:
        seed_order = tuple(payload["candidate_seed_order"])
        frozen_cases = tuple(
            FrozenCase(
                candidate=_candidate_from_payload(item["candidate"]),
                qualification=_decision_from_payload(item["qualification"]),
                selection_rank_within_scale=int(item["selection_rank_within_scale"]),
            )
            for item in payload["cases"]
        )
        ledger = tuple(
            _decision_from_payload(item) for item in payload["qualification_ledger"]
        )
        raw_selection_ledger = payload.get("selection_ledger", ())
        if not isinstance(raw_selection_ledger, (list, tuple)):
            raise CLIError("selection_ledger must be a list")
        replacement_protocol = _replacement_protocol_from_payload(
            payload.get("replacement_selection_protocol")
        )
        selection_ledger = tuple(
            _selection_entry_from_payload(item) for item in raw_selection_ledger
        )
        manifest = BenchmarkManifest(
            cases=frozen_cases,
            qualification_ledger=ledger,
            candidate_seed_order=seed_order,
            source_pool_sha256=str(payload["source_pool_sha256"]),
            source_pool_label=str(payload["source_pool_label"]),
            freeze_metadata=FreezeMetadata(**payload["freeze_metadata"]),
            minimum_case_count=int(payload["minimum_case_count"]),
            minimum_cases_per_scale=int(payload["minimum_cases_per_scale"]),
            schema_version=int(payload["schema_version"]),
            replacement_selection_protocol=replacement_protocol,
            selection_ledger=selection_ledger,
        )
    except CLIError:
        raise
    except (KeyError, TypeError, ValueError) as error:
        raise CLIError(f"invalid benchmark manifest: {error}") from error
    reported_manifest_hash = payload.get("manifest_sha256")
    if reported_manifest_hash is not None and reported_manifest_hash != manifest.sha256:
        raise CLIError("manifest SHA256 does not match its canonical payload")
    qualification_mode = artifact.get("qualification_mode")
    if qualification_mode is not None:
        try:
            qualification_mode = BenchmarkMode(str(qualification_mode)).value
        except ValueError as error:
            raise CLIError("manifest qualification_mode is invalid") from error
    raw_model_identity = artifact.get("model_identity")
    if not isinstance(raw_model_identity, Mapping):
        raise CLIError("benchmark manifest is missing its pinned model_identity")
    try:
        model_identity = ModelIdentity(**raw_model_identity)
    except (TypeError, ValueError) as error:
        raise CLIError(
            f"benchmark manifest model_identity is invalid: {error}"
        ) from error
    model_id = build_model_id(model_identity)
    if artifact.get("model_id") != model_id:
        raise CLIError("benchmark manifest model_id is inconsistent")
    identity_payload = {
        "manifest": manifest.artifact_payload(),
        "qualification_mode": qualification_mode,
        "model_identity": model_identity,
        "model_id": model_id,
    }
    computed_identity = canonical_sha256(identity_payload)
    reported_identity = artifact.get("benchmark_manifest_artifact_sha256")
    if reported_identity is not None and reported_identity != computed_identity:
        raise CLIError("benchmark-manifest wrapper SHA256 is inconsistent")
    return LoadedManifest(
        manifest,
        computed_identity,
        qualification_mode,
        model_identity,
        model_id,
    )


def _resolved_instance_path(root: Path, instance_ref: str) -> Path:
    if Path(instance_ref).is_absolute():
        raise CLIError("instance_ref must be relative to --instances-root")
    resolved_root = root.expanduser().resolve()
    resolved = (resolved_root / instance_ref).resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as error:
        raise CLIError(
            f"instance_ref escapes instances root: {instance_ref!r}"
        ) from error
    return resolved


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0 or parsed > GUROBI_MAX_START_NODE_LIMIT:
        raise argparse.ArgumentTypeError(
            f"value must be an integer in [0, {GUROBI_MAX_START_NODE_LIMIT}]"
        )
    return parsed


def _root_cut_passes(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected an integer") from error
    if not -1 <= parsed <= 2_000_000_000:
        raise argparse.ArgumentTypeError("expected -1 or an integer in [0, 2000000000]")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("value must be positive and finite")
    return parsed


def _nonnegative_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0.0:
        raise argparse.ArgumentTypeError("value must be nonnegative and finite")
    return parsed


def _fraction(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("value must lie in [0, 1]")
    return parsed


def _require_unique(values: Sequence[Any], *, label: str) -> None:
    if len(set(values)) != len(values):
        raise CLIError(f"{label} must not contain duplicates")


def _build_pool_command(args: argparse.Namespace, hooks: CLIHooks) -> Mapping[str, Any]:
    config = CaseBuildConfig(
        output_root=Path(args.output_root),
        nanshan_raw_path=Path(args.nanshan_raw),
        tsl_raw_path=Path(args.tsl_raw),
        candidate_hour_start=args.candidate_hours[0],
        candidate_hour_end=args.candidate_hours[1],
        site_grid_km=args.site_grid_km,
        pool_label=args.pool_label,
    )
    result = hooks.build_pool(config)
    pool = getattr(result, "pool")
    return {
        "command": "build-pool",
        "pool_path": getattr(result, "pool_path"),
        "pool_sha256": pool.sha256,
        "candidate_count": len(pool.candidates),
    }


def _qualification_policy(args: argparse.Namespace) -> QualificationPolicy:
    target = (
        tuple(args.large_scale_open_h_target)
        if args.large_scale_open_h_target is not None
        else (50, 60)
    )
    return QualificationPolicy(
        optimality_tolerance=args.qualification_optimality_tolerance,
        minimum_savings_fraction=args.minimum_savings_fraction,
        minimum_service_fraction=args.minimum_service_fraction,
        minimum_h_open_fraction=args.minimum_h_open_fraction,
        maximum_h_open_fraction=args.maximum_h_open_fraction,
        large_scale_open_h_target=target,
    )


def _qualification_settings(args: argparse.Namespace) -> NativeQualificationSettings:
    return NativeQualificationSettings(
        time_limit_seconds=args.time_limit,
        mip_gap=args.mip_gap,
        solver_seed=args.solver_seed,
        threads=1,
        output_flag=int(args.solver_output),
        method=args.method,
        miqcp_method=args.miqcp_method,
        mip_focus=args.mip_focus,
        heuristics=args.heuristics,
        no_rel_heur_time=args.no_rel_heur_time,
        presolve=args.presolve,
        native_cuts=args.native_cuts,
        root_cut_passes=args.root_cut_passes,
        numeric_focus=args.numeric_focus,
        feasibility_tolerance=args.feasibility_tolerance,
        optimality_tolerance=args.optimality_tolerance,
        integer_feasibility_tolerance=args.integer_feasibility_tolerance,
    )


def _qualification_geometry_start_policy(
    args: argparse.Namespace,
) -> GeometryStartPolicy:
    return GeometryStartPolicy(
        enabled=args.geometry_start,
        per_attempt_seconds=args.geometry_start_seconds,
        start_node_limit=args.geometry_start_node_limit,
    )


def _effective_decision(outcome: QualificationRunnerOutcome) -> QualificationDecision:
    """Merge runner-level failures into the decision used for freezing."""

    if (
        outcome.decision.accepted == outcome.accepted
        and outcome.decision.rejection_reasons == outcome.rejection_reasons
    ):
        return outcome.decision
    return replace(
        outcome.decision,
        accepted=outcome.accepted,
        rejection_reasons=outcome.rejection_reasons,
    )


def _ensure_candidate_qualification_outcome(
    outcome: QualificationRunnerOutcome,
    *,
    case_id: str | None = None,
) -> None:
    """Reject infrastructure records before they can become case evidence.

    A replacement is justified only by a normally terminated solver run whose
    evidence fails the preregistered qualification policy.  Runner mechanism
    errors, unknown future runner reasons, and abnormal Gurobi statuses abort
    the batch and deliberately leave the current case without a checkpoint.
    """

    label = case_id or outcome.evidence.case_id
    decision_reasons = set(outcome.decision.rejection_reasons)
    unexpected_reasons = sorted(
        set(outcome.rejection_reasons)
        - decision_reasons
        - NORMAL_QUALIFICATION_RUNNER_REASONS
    )
    if unexpected_reasons:
        raise CLIError(
            f"qualification infrastructure failure for {label}: runner reasons "
            + ", ".join(unexpected_reasons)
        )

    if outcome.witness_status not in NORMAL_QUALIFICATION_SOLVER_STATUSES:
        raise CLIError(
            f"qualification infrastructure failure for {label}: unexpected "
            f"witness status {outcome.witness_status!r}"
        )
    if outcome.fresh_model_validation_performed:
        if outcome.validation_status not in NORMAL_QUALIFICATION_SOLVER_STATUSES:
            raise CLIError(
                f"qualification infrastructure failure for {label}: unexpected "
                f"fresh-validation status {outcome.validation_status!r}"
            )
    elif outcome.validation_status != "NOT_RUN":
        raise CLIError(
            f"qualification infrastructure failure for {label}: validation status "
            "was recorded without fresh validation"
        )

    planning_errors = {
        attempt.error
        for attempt in outcome.geometry_start_attempts
        if not attempt.plan.prechecks_passed and attempt.error is not None
    }
    unexpected_exceptions = tuple(
        message
        for message in outcome.exception_messages
        if message not in planning_errors
    )
    if unexpected_exceptions:
        raise CLIError(
            f"qualification infrastructure failure for {label}: "
            + "; ".join(unexpected_exceptions)
        )
    for attempt in outcome.geometry_start_attempts:
        if attempt.solve_status not in {
            "NOT_RUN",
            *NORMAL_QUALIFICATION_SOLVER_STATUSES,
        }:
            raise CLIError(
                f"qualification infrastructure failure for {label}: unexpected "
                f"geometry-start status {attempt.solve_status!r}"
            )


def _validate_qualification_model_audits(
    outcome: QualificationRunnerOutcome,
    expected: QualificationModelContract,
    *,
    label: str,
) -> None:
    if outcome.expected_model_contract != expected:
        raise CLIError(
            f"qualification model contract for {label} does not match its candidate"
        )
    witness = outcome.witness_model_audit
    if witness is None or witness.contract_values() != (
        expected.effective_model_config_sha256,
        expected.demand_scale,
        expected.locker_fixed_cost_per_hour,
        expected.display_origin_choices,
        expected.general_corridor_choices,
        expected.screening_is_exact,
    ):
        raise CLIError(f"witness model audit for {label} does not match its candidate")
    fresh = outcome.fresh_model_audit
    if fresh is not None:
        if fresh.contract_values() != witness.contract_values():
            raise CLIError(
                f"fresh model audit for {label} does not match its candidate"
            )
        if (
            fresh.model_fingerprint is None
            or witness.model_fingerprint is None
            or fresh.model_fingerprint != witness.model_fingerprint
        ):
            raise CLIError(
                f"fresh model fingerprint for {label} does not match its witness"
            )
    if outcome.fresh_model_validation_performed and fresh is None:
        raise CLIError(f"fresh model validation for {label} lacks its model audit")


def _validate_qualification_task_integrity(
    task: QualificationTask,
    hooks: CLIHooks,
) -> None:
    """Fail the batch on code, solver, or instance drift.

    Integrity failures are not evidence that an instance is unsuitable and
    therefore must never enter the replacement ladder as ordinary rejections.
    """

    try:
        if hooks.model_identity() != task.model_identity:
            raise CLIError("model implementation/config changed during qualification")
        if (
            hooks.qualification_implementation_sha256()
            != task.qualification_implementation_sha256
        ):
            raise CLIError("qualification implementation changed during qualification")
        if hooks.gurobi_version() != task.gurobi_version:
            raise CLIError("Gurobi version changed during qualification")
        actual_hash, _file_hashes = hooks.canonical_hash(task.instance_dir)
    except (MemoryError, KeyboardInterrupt, SystemExit, CLIError):
        raise
    except Exception as error:
        raise CLIError(
            f"qualification integrity check failed for {task.candidate.case_id}: "
            f"{type(error).__name__}: {error}"
        ) from error
    if actual_hash != task.candidate.instance_sha256:
        raise CLIError(
            f"instance hash mismatch for {task.candidate.case_id}: "
            f"expected {task.candidate.instance_sha256}, got {actual_hash}"
        )


def _qualify_one(task: QualificationTask, hooks: CLIHooks) -> QualificationTaskResult:
    expected_model_contract = _candidate_model_contract(task.candidate)
    _validate_qualification_task_integrity(task, hooks)
    try:
        instance = hooks.load_instance(task.instance_dir, task.candidate)
        outcome = hooks.qualify_instance(
            instance,
            task.mode,
            case_id=task.candidate.case_id,
            scale=task.candidate.scale,
            qualification_run_id=task.qualification_run_id,
            instance_sha256=task.candidate.instance_sha256,
            policy=task.policy,
            settings=task.settings,
            geometry_start_policy=task.geometry_start_policy,
            expected_model_contract=expected_model_contract,
            fail_fast_infrastructure=True,
        )
        _ensure_candidate_qualification_outcome(
            outcome,
            case_id=task.candidate.case_id,
        )
        _validate_qualification_model_audits(
            outcome,
            expected_model_contract,
            label=task.candidate.case_id,
        )
        return QualificationTaskResult(task.candidate.case_id, outcome, None)
    except (MemoryError, KeyboardInterrupt, SystemExit, CLIError):
        raise
    except Exception as error:
        raise CLIError(
            f"qualification infrastructure failure for {task.candidate.case_id}: "
            f"{type(error).__name__}: {error}"
        ) from error


def _qualification_outcome_payload(
    result: QualificationTaskResult,
) -> dict[str, Any]:
    if result.outcome is None:
        raise CLIError("qualification infrastructure failures cannot enter the ledger")
    return {
        "case_id": result.case_id,
        "attempted": True,
        "orchestration_error": None,
        **result.outcome.to_payload(),
    }


def _unattempted_qualification_payload(
    candidate: CandidateCase,
    *,
    reason: str,
) -> dict[str, Any]:
    return {
        "case_id": candidate.case_id,
        "attempted": False,
        "accepted": False,
        "not_attempted_reason": reason,
        "orchestration_error": None,
    }


_QUALIFICATION_JOURNAL_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class _QualificationJournal:
    directory: Path
    identity: Mapping[str, Any]
    identity_sha256: str
    identity_file: Path
    checkpoint_paths_by_case: Mapping[str, Path]
    case_identities_by_case: Mapping[str, Mapping[str, Any]]


def _require_sha256(value: object, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise CLIError(f"{label} must be a lowercase 64-character SHA256 digest")
    return value


def _expected_evidence_root_argument(value: str) -> str:
    try:
        return _require_sha256(value, label="--expected-evidence-root")
    except CLIError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def _read_canonical_journal_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        raw = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise CLIError(f"cannot read {label}: {path}: {error}") from error
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as error:
        raise CLIError(f"{label} is not valid JSON: {path}: {error}") from error
    if not isinstance(payload, dict):
        raise CLIError(f"{label} must contain one JSON object: {path}")
    try:
        canonical = canonical_json(payload) + "\n"
    except (TypeError, ValueError) as error:
        raise CLIError(
            f"{label} is not canonical-JSON-safe: {path}: {error}"
        ) from error
    if raw != canonical:
        raise CLIError(f"{label} is not canonical JSON: {path}")
    return payload


def _qualification_journal_identity_payload(
    *,
    pool: CandidatePool,
    qualification_run_id: str,
    model_id: str,
    qualification_implementation_sha256: str,
    gurobi_version: str,
    mode: str,
    policy: QualificationPolicy,
    settings: NativeQualificationSettings,
    geometry_start_policy: GeometryStartPolicy,
    ordered_candidates: Sequence[CandidateCase],
) -> dict[str, Any]:
    return {
        "pool_sha256": pool.sha256,
        "qualification_run_id": qualification_run_id,
        "model_id": model_id,
        "qualification_implementation_sha256": (qualification_implementation_sha256),
        "gurobi_version": gurobi_version,
        "qualification_mode": mode,
        "policy": policy.to_payload(),
        "native_settings": asdict(settings),
        "geometry_start_policy": geometry_start_policy.to_payload(),
        "ordered_cases": [
            {
                "case_id": candidate.case_id,
                "instance_sha256": candidate.instance_sha256,
                "qualification_case_sha256": _candidate_model_contract(
                    candidate
                ).qualification_case_sha256,
            }
            for candidate in ordered_candidates
        ],
    }


def _qualification_case_journal_identity(
    journal_identity: Mapping[str, Any],
    journal_identity_sha256: str,
    candidate: CandidateCase,
    candidate_order_index: int,
) -> dict[str, Any]:
    model_contract = _candidate_model_contract(candidate)
    return {
        "journal_identity_sha256": journal_identity_sha256,
        "pool_sha256": journal_identity["pool_sha256"],
        "qualification_run_id": journal_identity["qualification_run_id"],
        "model_id": journal_identity["model_id"],
        "qualification_implementation_sha256": journal_identity[
            "qualification_implementation_sha256"
        ],
        "gurobi_version": journal_identity["gurobi_version"],
        "qualification_mode": journal_identity["qualification_mode"],
        "policy": journal_identity["policy"],
        "native_settings": journal_identity["native_settings"],
        "geometry_start_policy": journal_identity["geometry_start_policy"],
        "candidate_order_index": candidate_order_index,
        "case_id": candidate.case_id,
        "instance_sha256": candidate.instance_sha256,
        "qualification_case_sha256": model_contract.qualification_case_sha256,
    }


def _prepare_qualification_journal(
    *,
    output_target: Path,
    journal_dir_override: str | None,
    identity: Mapping[str, Any],
    ordered_candidates: Sequence[CandidateCase],
) -> _QualificationJournal:
    directory = (
        Path(journal_dir_override).expanduser().resolve()
        if journal_dir_override is not None
        else output_target.with_name(f"{output_target.name}.journal")
    )
    if directory == output_target:
        raise CLIError("qualification journal directory may not equal --output")
    if directory.exists() and not directory.is_dir():
        raise CLIError(f"qualification journal path is not a directory: {directory}")
    directory.mkdir(parents=True, exist_ok=True)

    identity_sha256 = canonical_sha256(identity)
    identity_file = directory / "journal_identity.json"
    identity_record = {
        "artifact_type": "mlnd_branch_cut_qualification_journal_identity",
        "artifact_schema_version": _QUALIFICATION_JOURNAL_SCHEMA_VERSION,
        "identity": dict(identity),
        "journal_identity_sha256": identity_sha256,
    }
    if identity_file.exists():
        existing_identity = _read_canonical_journal_object(
            identity_file,
            label="qualification journal identity",
        )
        if canonical_json(existing_identity) != canonical_json(identity_record):
            raise CLIError(
                "qualification journal identity mismatch; refusing checkpoint reuse"
            )
    else:
        existing_entries = tuple(
            item
            for item in directory.iterdir()
            if not item.name.startswith(ATOMIC_WRITE_TEMP_PREFIX)
        )
        if existing_entries:
            raise CLIError(
                "qualification journal directory is nonempty but has no identity file"
            )
        write_canonical_json_exclusive(identity_file, identity_record)

    checkpoint_paths: dict[str, Path] = {}
    case_identities: dict[str, Mapping[str, Any]] = {}
    for index, candidate in enumerate(ordered_candidates):
        case_digest = hashlib.sha256(candidate.case_id.encode("utf-8")).hexdigest()
        checkpoint_paths[candidate.case_id] = (
            directory / f"case-{index:06d}-{case_digest}.json"
        )
        case_identities[candidate.case_id] = _qualification_case_journal_identity(
            identity,
            identity_sha256,
            candidate,
            index,
        )

    expected_names = {identity_file.name} | {
        path.name for path in checkpoint_paths.values()
    }
    unexpected = sorted(
        item.name
        for item in directory.iterdir()
        if item.name not in expected_names
        and not item.name.startswith(ATOMIC_WRITE_TEMP_PREFIX)
    )
    if unexpected:
        raise CLIError(
            "qualification journal contains unexpected files: " + ", ".join(unexpected)
        )
    return _QualificationJournal(
        directory=directory,
        identity=identity,
        identity_sha256=identity_sha256,
        identity_file=identity_file,
        checkpoint_paths_by_case=checkpoint_paths,
        case_identities_by_case=case_identities,
    )


def _qualification_evidence_from_payload(
    payload: object,
) -> QualificationEvidence:
    if not isinstance(payload, Mapping):
        raise CLIError("checkpoint outcome evidence must be an object")
    expected_fields = set(QualificationEvidence.__dataclass_fields__)
    if set(payload) != expected_fields:
        raise CLIError("checkpoint outcome evidence fields are inconsistent")
    values = dict(payload)
    numeric_fields = (
        "courier_only_objective",
        "witness_objective",
        "total_demand_per_hour",
        "witness_service_per_hour",
    )
    for field_name in numeric_fields:
        value = values[field_name]
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, (int, float))
        ):
            raise CLIError(f"checkpoint evidence {field_name} must be a number or null")
    for field_name in (
        "open_facilities",
        "positive_cost_open_facilities",
        "candidate_h",
        "open_h",
    ):
        value = values[field_name]
        if isinstance(value, bool) or not isinstance(value, int):
            raise CLIError(f"checkpoint evidence {field_name} must be an integer")
    active_h = values["active_h"]
    if active_h is not None and (
        isinstance(active_h, bool) or not isinstance(active_h, int)
    ):
        raise CLIError("checkpoint evidence active_h must be an integer or null")
    active_threshold = values["active_h_threshold_per_hour"]
    if active_threshold is not None and (
        isinstance(active_threshold, bool)
        or not isinstance(active_threshold, (int, float))
    ):
        raise CLIError(
            "checkpoint evidence active_h_threshold_per_hour must be a number or null"
        )
    if type(values["witness_feasible"]) is not bool:
        raise CLIError("checkpoint evidence witness_feasible must be boolean")
    for field_name in numeric_fields:
        if values[field_name] is None:
            values[field_name] = math.nan
    try:
        evidence = QualificationEvidence(**values)
    except (TypeError, ValueError) as error:
        raise CLIError(f"invalid checkpoint qualification evidence: {error}") from error
    # Non-finite runner evidence is stored as JSON null by the outcome's safe
    # serializer and restored to NaN here.  The enclosing outcome round-trip
    # below performs the strict canonical comparison after re-sanitizing it.
    return evidence


def _qualification_model_record_from_payload(
    payload: object,
    *,
    record_type: type[QualificationModelContract] | type[QualificationModelAudit],
    label: str,
) -> QualificationModelContract | QualificationModelAudit | None:
    if payload is None:
        return None
    if not isinstance(payload, Mapping):
        raise CLIError(f"checkpoint outcome {label} must be an object or null")
    if set(payload) != set(record_type.__dataclass_fields__):
        raise CLIError(f"checkpoint outcome {label} fields are inconsistent")
    try:
        return record_type(**dict(payload))
    except (TypeError, ValueError) as error:
        raise CLIError(f"invalid checkpoint outcome {label}: {error}") from error


def _qualification_outcome_from_payload(
    payload: object,
) -> QualificationRunnerOutcome:
    if not isinstance(payload, Mapping):
        raise CLIError("checkpoint outcome must be an object")
    expected_fields = set(QualificationRunnerOutcome.__dataclass_fields__)
    if set(payload) != expected_fields:
        raise CLIError("checkpoint outcome fields are inconsistent")
    values = dict(payload)
    values["evidence"] = _qualification_evidence_from_payload(values["evidence"])
    values["expected_model_contract"] = _qualification_model_record_from_payload(
        values["expected_model_contract"],
        record_type=QualificationModelContract,
        label="expected_model_contract",
    )
    values["witness_model_audit"] = _qualification_model_record_from_payload(
        values["witness_model_audit"],
        record_type=QualificationModelAudit,
        label="witness_model_audit",
    )
    values["fresh_model_audit"] = _qualification_model_record_from_payload(
        values["fresh_model_audit"],
        record_type=QualificationModelAudit,
        label="fresh_model_audit",
    )
    raw_decision = values["decision"]
    if not isinstance(raw_decision, Mapping):
        raise CLIError("checkpoint outcome decision must be an object")
    values["decision"] = _decision_from_payload(raw_decision)
    raw_timings = values["phase_timings"]
    if not isinstance(raw_timings, Mapping):
        raise CLIError("checkpoint phase_timings must be an object")
    try:
        values["geometry_start_policy"] = GeometryStartPolicy.from_payload(
            values["geometry_start_policy"]
        )
        raw_geometry_attempts = values["geometry_start_attempts"]
        if not isinstance(raw_geometry_attempts, list):
            raise ValueError("geometry_start_attempts must be a list")
        values["geometry_start_attempts"] = tuple(
            GeometryStartAttemptAudit.from_payload(item)
            for item in raw_geometry_attempts
        )
        values["phase_timings"] = QualificationPhaseTimings(**dict(raw_timings))
        values["rejection_reasons"] = tuple(values["rejection_reasons"])
        values["exception_messages"] = tuple(values["exception_messages"])
        outcome = QualificationRunnerOutcome(**values)
    except (TypeError, ValueError) as error:
        raise CLIError(f"invalid checkpoint qualification outcome: {error}") from error
    for field_name in (
        "accepted",
        "temporary_constraint_used",
        "fresh_model_validation_performed",
        "feasibility_objective_used",
    ):
        if type(getattr(outcome, field_name)) is not bool:
            raise CLIError(f"checkpoint outcome {field_name} must be boolean")
    if (
        outcome.temporary_constraint_removed is not None
        and type(outcome.temporary_constraint_removed) is not bool
    ):
        raise CLIError(
            "checkpoint outcome temporary_constraint_removed must be boolean or null"
        )
    if (
        isinstance(outcome.captured_variable_count, bool)
        or not isinstance(outcome.captured_variable_count, int)
        or outcome.captured_variable_count < 0
    ):
        raise CLIError(
            "checkpoint outcome captured_variable_count must be a non-negative integer"
        )
    if (
        isinstance(outcome.required_service_per_hour, bool)
        or not isinstance(outcome.required_service_per_hour, (int, float))
        or not math.isfinite(float(outcome.required_service_per_hour))
        or float(outcome.required_service_per_hour) <= 0.0
    ):
        raise CLIError(
            "checkpoint outcome required_service_per_hour must be positive and finite"
        )
    objective_upper_bound = outcome.required_objective_upper_bound
    if objective_upper_bound is not None and (
        isinstance(objective_upper_bound, bool)
        or not isinstance(objective_upper_bound, (int, float))
        or not math.isfinite(float(objective_upper_bound))
    ):
        raise CLIError(
            "checkpoint outcome required_objective_upper_bound must be finite or null"
        )
    for field_name in ("witness_status", "validation_status"):
        value = getattr(outcome, field_name)
        if not isinstance(value, str) or not value:
            raise CLIError(f"checkpoint outcome {field_name} must be a nonempty string")
    if outcome.feasibility_objective_used and not outcome.temporary_constraint_used:
        raise CLIError(
            "checkpoint used the feasibility objective without a temporary search"
        )
    if (
        outcome.temporary_constraint_removed is not None
        and not outcome.temporary_constraint_used
    ):
        raise CLIError(
            "checkpoint reports temporary cleanup without a temporary search"
        )
    if outcome.geometry_start_attempts and (
        not outcome.geometry_start_policy.enabled
        or not outcome.temporary_constraint_used
        or not outcome.feasibility_objective_used
    ):
        raise CLIError(
            "checkpoint geometry attempts are inconsistent with fallback policy"
        )
    if outcome.geometry_start_attempts:
        try:
            registered_schedule = outcome.geometry_start_policy.schedule_for(
                outcome.evidence.scale,
                outcome.evidence.candidate_h,
            )
        except ValueError as error:
            raise CLIError(
                f"checkpoint geometry schedule is invalid: {error}"
            ) from error
        observed_targets = tuple(
            attempt.plan.target_support_size
            for attempt in outcome.geometry_start_attempts
        )
        if observed_targets != registered_schedule[: len(observed_targets)]:
            raise CLIError(
                "checkpoint geometry attempt targets are not a registered prefix"
            )
    selected_geometry_sha = outcome.geometry_start_selected_plan_sha256
    if selected_geometry_sha is not None and (
        not isinstance(selected_geometry_sha, str)
        or len(selected_geometry_sha) != 64
        or any(
            character not in "0123456789abcdef" for character in selected_geometry_sha
        )
    ):
        raise CLIError("checkpoint selected geometry plan SHA256 is invalid")
    successful_attempts = tuple(
        attempt
        for attempt in outcome.geometry_start_attempts
        if attempt.full_witness_found
    )
    if len(successful_attempts) > 1:
        raise CLIError("checkpoint has multiple successful geometry attempts")
    matching_selected_attempts = tuple(
        attempt
        for attempt in successful_attempts
        if attempt.plan.plan_sha256 == selected_geometry_sha
    )
    if selected_geometry_sha is not None and len(matching_selected_attempts) != 1:
        raise CLIError(
            "checkpoint selected geometry plan does not match one successful attempt"
        )
    if selected_geometry_sha is not None:
        selected_attempt = matching_selected_attempts[0]
        selected_install = selected_attempt.install
        if (
            selected_install is None
            or not selected_install.safety_preserved
            or selected_attempt.solve_status == "NOT_RUN"
            or selected_attempt.error is not None
            or selected_attempt.cleanup_performed
        ):
            raise CLIError(
                "checkpoint selected geometry plan lacks a safe solved install"
            )
    if selected_geometry_sha is None and any(
        attempt.full_witness_found for attempt in outcome.geometry_start_attempts
    ):
        raise CLIError(
            "checkpoint successful geometry attempt lacks selected plan SHA256"
        )
    final_attempt_index = len(outcome.geometry_start_attempts) - 1
    for attempt_index, attempt in enumerate(outcome.geometry_start_attempts):
        if (
            attempt.per_attempt_seconds
            != outcome.geometry_start_policy.per_attempt_seconds
            or attempt.start_node_limit
            != outcome.geometry_start_policy.start_node_limit
        ):
            raise CLIError(
                "checkpoint geometry attempt settings do not match frozen policy"
            )
        install = attempt.install
        unsafe_install = install is not None and not install.safety_preserved
        if unsafe_install and (
            QualificationRunnerReason.GEOMETRY_START_SAFETY_CHECK_FAILED
            not in outcome.rejection_reasons
        ):
            raise CLIError(
                "checkpoint unsafe geometry start lacks its runner rejection"
            )
        installation_failure = bool(
            attempt.plan.prechecks_passed and attempt.install is None
        )
        if installation_failure and (
            QualificationRunnerReason.GEOMETRY_START_INSTALLATION_FAILED
            not in outcome.rejection_reasons
        ):
            raise CLIError(
                "checkpoint geometry installation failure lacks its runner rejection"
            )
        solve_failure = attempt.solve_status == "ERROR"
        if solve_failure and (
            QualificationRunnerReason.GEOMETRY_START_SOLVE_FAILED
            not in outcome.rejection_reasons
        ):
            raise CLIError(
                "checkpoint geometry solve failure lacks its runner rejection"
            )
        capture_failure = bool(
            install is not None
            and attempt.solve_status not in {"NOT_RUN", "ERROR"}
            and attempt.error is not None
            and attempt.cleanup_succeeded is True
        )
        if capture_failure and (
            QualificationRunnerReason.COMPLETE_INCUMBENT_CAPTURE_FAILED
            not in outcome.rejection_reasons
        ):
            raise CLIError(
                "checkpoint geometry capture failure lacks its runner rejection"
            )
        terminal_attempt = bool(
            attempt.full_witness_found
            or unsafe_install
            or installation_failure
            or solve_failure
            or capture_failure
            or attempt.cleanup_succeeded is False
        )
        if terminal_attempt and attempt_index != final_attempt_index:
            raise CLIError(
                "checkpoint terminal geometry attempt is not the final attempt"
            )
        if (
            attempt.plan.prechecks_passed
            and attempt.install is not None
            and not attempt.full_witness_found
            and not attempt.cleanup_performed
        ):
            raise CLIError("checkpoint failed geometry attempt lacks cleanup audit")
        if attempt.cleanup_succeeded is False and (
            QualificationRunnerReason.GEOMETRY_START_CLEANUP_FAILED
            not in outcome.rejection_reasons
        ):
            raise CLIError(
                "checkpoint geometry cleanup failure lacks its runner rejection"
            )
    if outcome.accepted != (len(outcome.rejection_reasons) == 0):
        raise CLIError(
            "checkpoint outcome accepted flag conflicts with rejection reasons"
        )
    if any(
        not isinstance(reason, str) or not reason
        for reason in outcome.rejection_reasons
    ) or len(set(outcome.rejection_reasons)) != len(outcome.rejection_reasons):
        raise CLIError("checkpoint outcome rejection reasons are invalid")
    if any(
        not isinstance(message, str) or not message
        for message in outcome.exception_messages
    ):
        raise CLIError("checkpoint outcome exception messages are invalid")
    try:
        recomputed_decision = assess_qualification(
            outcome.evidence,
            outcome.decision.policy,
        )
    except (TypeError, ValueError) as error:
        raise CLIError(f"checkpoint evidence cannot be reassessed: {error}") from error
    if outcome.decision != recomputed_decision:
        raise CLIError("checkpoint qualification decision does not match its evidence")
    if not set(outcome.decision.rejection_reasons).issubset(outcome.rejection_reasons):
        raise CLIError(
            "checkpoint runner reasons omit qualification-decision rejections"
        )
    if outcome.accepted:
        if not outcome.fresh_model_validation_performed:
            raise CLIError("accepted checkpoint lacks fresh original-model validation")
        if outcome.captured_variable_count <= 0:
            raise CLIError("accepted checkpoint lacks a complete captured incumbent")
        if outcome.temporary_constraint_used and (
            outcome.temporary_constraint_removed is not True
            or not outcome.feasibility_objective_used
        ):
            raise CLIError(
                "accepted checkpoint has an incomplete temporary-search transaction"
            )
    if canonical_json(outcome.to_payload()) != canonical_json(payload):
        raise CLIError("checkpoint qualification outcome is not self-consistent")
    _ensure_candidate_qualification_outcome(outcome)
    return outcome


def _qualification_task_result_payload(
    result: QualificationTaskResult,
) -> dict[str, Any]:
    if result.outcome is None or result.error is not None:
        raise CLIError("qualification infrastructure failures may not be checkpointed")
    return {
        "case_id": result.case_id,
        "attempted": True,
        "outcome": result.outcome.to_payload() if result.outcome is not None else None,
        "orchestration_error": result.error,
    }


def _qualification_task_result_from_payload(
    payload: object,
    *,
    candidate: CandidateCase,
    qualification_run_id: str,
    policy: QualificationPolicy,
    geometry_start_policy: GeometryStartPolicy,
) -> QualificationTaskResult:
    if not isinstance(payload, Mapping):
        raise CLIError("qualification checkpoint result must be an object")
    if set(payload) != {
        "case_id",
        "attempted",
        "outcome",
        "orchestration_error",
    }:
        raise CLIError("qualification checkpoint result fields are inconsistent")
    if payload["attempted"] is not True or payload["case_id"] != candidate.case_id:
        raise CLIError("qualification checkpoint result identity is inconsistent")
    raw_outcome = payload["outcome"]
    raw_error = payload["orchestration_error"]
    if raw_outcome is None:
        raise CLIError(
            "historical orchestration-error checkpoint is not reusable; "
            "remove it and rerun the case"
        )
    if raw_error is not None:
        raise CLIError("qualification checkpoint may not contain outcome and error")
    outcome = _qualification_outcome_from_payload(raw_outcome)
    expected_model_contract = _candidate_model_contract(candidate)
    _validate_qualification_model_audits(
        outcome,
        expected_model_contract,
        label=candidate.case_id,
    )
    for record in (outcome.evidence, outcome.decision):
        if (
            record.case_id != candidate.case_id
            or record.scale != candidate.scale
            or record.qualification_run_id != qualification_run_id
            or record.instance_sha256 != candidate.instance_sha256
        ):
            raise CLIError("checkpoint outcome identity does not match its candidate")
    if outcome.decision.policy != policy:
        raise CLIError("checkpoint outcome policy does not match journal identity")
    if outcome.geometry_start_policy != geometry_start_policy:
        raise CLIError(
            "checkpoint geometry-start policy does not match journal identity"
        )
    return QualificationTaskResult(candidate.case_id, outcome, None)


def _qualification_checkpoint_payload(
    journal: _QualificationJournal,
    result: QualificationTaskResult,
) -> dict[str, Any]:
    case_identity = journal.case_identities_by_case[result.case_id]
    entry = {
        "identity": case_identity,
        "result": _qualification_task_result_payload(result),
    }
    return {
        "artifact_type": "mlnd_branch_cut_qualification_case_checkpoint",
        "artifact_schema_version": _QUALIFICATION_JOURNAL_SCHEMA_VERSION,
        **entry,
        "checkpoint_sha256": canonical_sha256(entry),
    }


def _load_qualification_checkpoint(
    journal: _QualificationJournal,
    candidate: CandidateCase,
    *,
    qualification_run_id: str,
    policy: QualificationPolicy,
    geometry_start_policy: GeometryStartPolicy,
) -> QualificationTaskResult:
    checkpoint_path = journal.checkpoint_paths_by_case[candidate.case_id]
    payload = _read_canonical_journal_object(
        checkpoint_path,
        label=f"qualification checkpoint for {candidate.case_id}",
    )
    if set(payload) != {
        "artifact_type",
        "artifact_schema_version",
        "identity",
        "result",
        "checkpoint_sha256",
    }:
        raise CLIError(f"checkpoint fields conflict for {candidate.case_id}")
    if (
        payload["artifact_type"] != "mlnd_branch_cut_qualification_case_checkpoint"
        or payload["artifact_schema_version"] != _QUALIFICATION_JOURNAL_SCHEMA_VERSION
        or canonical_json(payload["identity"])
        != canonical_json(journal.case_identities_by_case[candidate.case_id])
    ):
        raise CLIError(f"checkpoint identity conflict for {candidate.case_id}")
    entry = {"identity": payload["identity"], "result": payload["result"]}
    if payload["checkpoint_sha256"] != canonical_sha256(entry):
        raise CLIError(f"checkpoint content hash conflict for {candidate.case_id}")
    return _qualification_task_result_from_payload(
        payload["result"],
        candidate=candidate,
        qualification_run_id=qualification_run_id,
        policy=policy,
        geometry_start_policy=geometry_start_policy,
    )


def _load_qualification_journal_results(
    journal: _QualificationJournal,
    ordered_candidates: Sequence[CandidateCase],
    *,
    qualification_run_id: str,
    policy: QualificationPolicy,
    geometry_start_policy: GeometryStartPolicy,
) -> dict[str, QualificationTaskResult]:
    results: dict[str, QualificationTaskResult] = {}
    for candidate in ordered_candidates:
        checkpoint_path = journal.checkpoint_paths_by_case[candidate.case_id]
        if checkpoint_path.exists():
            results[candidate.case_id] = _load_qualification_checkpoint(
                journal,
                candidate,
                qualification_run_id=qualification_run_id,
                policy=policy,
                geometry_start_policy=geometry_start_policy,
            )
    return results


def _write_qualification_checkpoint(
    journal: _QualificationJournal,
    task: QualificationTask,
    result: QualificationTaskResult,
) -> None:
    if result.case_id != task.candidate.case_id:
        raise CLIError("qualification result case_id does not match its task")
    payload = _qualification_checkpoint_payload(journal, result)
    _qualification_task_result_from_payload(
        payload["result"],
        candidate=task.candidate,
        qualification_run_id=task.qualification_run_id,
        policy=task.policy,
        geometry_start_policy=task.geometry_start_policy,
    )
    checkpoint_path = journal.checkpoint_paths_by_case[result.case_id]
    try:
        write_canonical_json_exclusive(checkpoint_path, payload)
    except FileExistsError as error:
        raise CLIError(
            f"checkpoint already exists for {result.case_id}; refusing overwrite"
        ) from error
    except BaseException as error:
        # The atomic publisher normally leaves no target on failure.  If a
        # failure occurs after the hard-link publication (for example while
        # syncing the directory), remove only the case file that was known to
        # be absent before this exclusive write.  Infrastructure failures must
        # never become resumable candidate rejections.
        cleanup_error: OSError | None = None
        try:
            checkpoint_path.unlink(missing_ok=True)
        except OSError as failure:
            cleanup_error = failure
        if isinstance(error, (MemoryError, KeyboardInterrupt, SystemExit)):
            raise
        if cleanup_error is not None:
            raise CLIError(
                f"qualification checkpoint write and cleanup both failed for "
                f"{result.case_id}: write={type(error).__name__}: {error}; "
                f"cleanup={type(cleanup_error).__name__}: {cleanup_error}"
            ) from error
        raise CLIError(
            f"qualification checkpoint write failed for {result.case_id}: "
            f"{type(error).__name__}: {error}"
        ) from error


def _qualification_journal_file_hashes(
    journal: _QualificationJournal,
    attempted_case_ids: Sequence[str],
) -> dict[str, str]:
    files = [journal.identity_file]
    files.extend(
        journal.checkpoint_paths_by_case[case_id] for case_id in attempted_case_ids
    )
    missing = [path.name for path in files if not path.is_file()]
    if missing:
        raise CLIError("qualification journal files are missing: " + ", ".join(missing))
    return {path.name: _sha256_file(path) for path in sorted(files)}


def _qualification_result_decision(
    result: QualificationTaskResult,
    task: QualificationTask,
) -> QualificationDecision:
    if result.outcome is None:
        raise CLIError("qualification infrastructure failure has no case decision")
    return _effective_decision(result.outcome)


def _qualify_command_locked(
    args: argparse.Namespace,
    hooks: CLIHooks,
) -> Mapping[str, Any]:
    if args.jobs != 1:
        raise CLIError("qualification is hard-limited to --jobs 1 on the 24 GiB host")
    output_target = _require_absent_output(
        args.output,
        label="qualification output",
    )
    pool = load_candidate_pool(args.pool)
    mode = BenchmarkMode(args.mode).value
    policy = _qualification_policy(args)
    settings = _qualification_settings(args)
    geometry_start_policy = _qualification_geometry_start_policy(args)
    try:
        model_identity = hooks.model_identity()
        model_id = build_model_id(model_identity)
        qualification_implementation_sha256 = _require_sha256(
            hooks.qualification_implementation_sha256(),
            label="qualification_implementation_sha256",
        )
        gurobi_version = hooks.gurobi_version()
    except (MemoryError, KeyboardInterrupt, SystemExit, CLIError):
        raise
    except Exception as error:
        raise CLIError(
            "qualification environment initialization failed: "
            f"{type(error).__name__}: {error}"
        ) from error
    if not isinstance(gurobi_version, str) or not gurobi_version.strip():
        raise CLIError("gurobi_version must be a nonempty string")
    qualification_run_id = args.qualification_run_id or (
        "qualification-"
        + canonical_sha256(
            {
                "pool_sha256": pool.sha256,
                "mode": mode,
                "policy": policy,
                "settings": settings,
                "geometry_start_policy": geometry_start_policy,
                "model_id": model_id,
                "qualification_implementation_sha256": (
                    qualification_implementation_sha256
                ),
                "gurobi_version": gurobi_version,
            }
        )
    )
    instances_root = Path(args.instances_root)
    ordered_candidates = pool.ordered_candidates()
    tasks_by_case = {
        candidate.case_id: QualificationTask(
            candidate=candidate,
            instance_dir=_resolved_instance_path(
                instances_root,
                candidate.instance_ref,
            ),
            mode=mode,
            qualification_run_id=qualification_run_id,
            model_identity=model_identity,
            qualification_implementation_sha256=(qualification_implementation_sha256),
            gurobi_version=gurobi_version,
            policy=policy,
            settings=settings,
            geometry_start_policy=geometry_start_policy,
        )
        for candidate in ordered_candidates
    }
    strata = _qualification_strata(pool)
    if strata is not None and policy.large_scale_open_h_target != (50, 60):
        raise CLIError(
            "the v5 overcomplete pool requires the pre-registered large-scale "
            "active-H target 50 60"
        )
    journal_identity = _qualification_journal_identity_payload(
        pool=pool,
        qualification_run_id=qualification_run_id,
        model_id=model_id,
        qualification_implementation_sha256=qualification_implementation_sha256,
        gurobi_version=gurobi_version,
        mode=mode,
        policy=policy,
        settings=settings,
        geometry_start_policy=geometry_start_policy,
        ordered_candidates=ordered_candidates,
    )
    journal = _prepare_qualification_journal(
        output_target=output_target,
        journal_dir_override=args.journal_dir,
        identity=journal_identity,
        ordered_candidates=ordered_candidates,
    )
    checkpoint_results = _load_qualification_journal_results(
        journal,
        ordered_candidates,
        qualification_run_id=qualification_run_id,
        policy=policy,
        geometry_start_policy=geometry_start_policy,
    )

    results_by_case: dict[str, QualificationTaskResult] = {}
    unattempted_reasons: dict[str, str] = {}

    def execute_layer(
        layer_tasks: Sequence[QualificationTask],
    ) -> tuple[QualificationTaskResult, ...]:
        layer_results: dict[str, QualificationTaskResult] = {}
        pending_tasks: list[QualificationTask] = []
        for task in layer_tasks:
            checkpoint = checkpoint_results.get(task.candidate.case_id)
            if checkpoint is None:
                pending_tasks.append(task)
            else:
                # A checkpoint binds the claimed hash, but reuse must also
                # verify the current on-disk instance after a crash or pause.
                _validate_qualification_task_integrity(task, hooks)
                layer_results[task.candidate.case_id] = checkpoint
        for task in pending_tasks:
            result = _qualify_one(task, hooks)
            _write_qualification_checkpoint(journal, task, result)
            layer_results[result.case_id] = result
        return tuple(layer_results[task.candidate.case_id] for task in layer_tasks)

    if strata is None:
        legacy_tasks = tuple(tasks_by_case[item.case_id] for item in ordered_candidates)
        for result in execute_layer(legacy_tasks):
            results_by_case[result.case_id] = result
    else:
        accepted_by_stratum = {item.stratum_id: 0 for item in strata}
        active_strata = {item.stratum_id for item in strata}
        maximum_rank_count = max(len(item.candidates_by_rank) for item in strata)
        for replacement_rank in range(maximum_rank_count):
            layer_strata = tuple(
                item
                for item in strata
                if item.stratum_id in active_strata
                and replacement_rank < len(item.candidates_by_rank)
            )
            if not layer_strata:
                break
            layer_tasks = tuple(
                tasks_by_case[item.candidates_by_rank[replacement_rank].case_id]
                for item in layer_strata
            )
            layer_results = execute_layer(layer_tasks)
            for stratum, result in zip(
                layer_strata,
                layer_results,
                strict=True,
            ):
                results_by_case[result.case_id] = result
                if result.outcome is None:
                    raise CLIError(
                        "qualification infrastructure failure reached the "
                        "replacement scheduler"
                    )
                decision = _effective_decision(result.outcome)
                if decision.accepted:
                    accepted_by_stratum[stratum.stratum_id] += 1
                if accepted_by_stratum[stratum.stratum_id] >= stratum.target_accepted:
                    active_strata.remove(stratum.stratum_id)
                    reason = "not_attempted_after_stratum_target_reached"
                else:
                    reason = ""
                if reason:
                    for candidate in stratum.candidates_by_rank[replacement_rank + 1 :]:
                        unattempted_reasons[candidate.case_id] = reason

            for stratum in layer_strata:
                if stratum.stratum_id in active_strata and replacement_rank + 1 >= len(
                    stratum.candidates_by_rank
                ):
                    active_strata.remove(stratum.stratum_id)

    unused_checkpoints = sorted(set(checkpoint_results).difference(results_by_case))
    if unused_checkpoints:
        raise CLIError(
            "qualification journal contains out-of-schedule checkpoints: "
            + ", ".join(unused_checkpoints)
        )

    results = tuple(
        results_by_case[candidate.case_id]
        for candidate in ordered_candidates
        if candidate.case_id in results_by_case
    )

    decisions = tuple(
        _qualification_result_decision(result, tasks_by_case[result.case_id])
        for result in results
    )
    completed_at = hooks.now_utc()
    artifact = {
        "artifact_type": "mlnd_branch_cut_qualification_ledger",
        "artifact_schema_version": 1,
        "source_pool_sha256": pool.sha256,
        "source_pool_label": pool.pool_label,
        "qualification_mode": mode,
        "qualification_run_id": qualification_run_id,
        "model_identity": model_identity,
        "model_id": model_id,
        "qualification_implementation_sha256": (qualification_implementation_sha256),
        "gurobi_version": gurobi_version,
        "resource_lock": str(Path(args.resource_lock).expanduser().resolve()),
        "completed_at_utc": completed_at,
        "policy": policy.to_payload(),
        "native_settings": asdict(settings),
        "geometry_start_policy": geometry_start_policy.to_payload(),
        "qualification_ledger": [decision.to_payload() for decision in decisions],
        "runner_outcomes": [
            (
                {
                    **_qualification_outcome_payload(
                        results_by_case[candidate.case_id]
                    ),
                    "qualification_solve_profile": candidate.generator_parameters.get(
                        "qualification_solve_profile"
                    ),
                }
                if candidate.case_id in results_by_case
                else {
                    **_unattempted_qualification_payload(
                        candidate,
                        reason=unattempted_reasons.get(
                            candidate.case_id,
                            "not_attempted_by_registered_scheduler",
                        ),
                    ),
                    "qualification_solve_profile": candidate.generator_parameters.get(
                        "qualification_solve_profile"
                    ),
                }
            )
            for candidate in ordered_candidates
        ],
        "replacement_schedule": {
            "enabled": strata is not None,
            "selection_rule": (
                OVERCOMPLETE_SELECTION_RULE if strata is not None else None
            ),
            "stratum_count": len(strata or ()),
            "attempted_candidate_count": len(results),
            "unattempted_candidate_count": len(ordered_candidates) - len(results),
        },
        "qualification_journal": {
            "journal_identity_sha256": journal.identity_sha256,
            "journal_file_sha256": _qualification_journal_file_hashes(
                journal,
                tuple(result.case_id for result in results),
            ),
        },
    }
    output = write_canonical_json_exclusive(output_target, artifact)
    return {
        "command": "qualify",
        "output": output,
        "journal_dir": journal.directory,
        "qualification_run_id": qualification_run_id,
        "candidate_count": len(ordered_candidates),
        "attempted_count": len(results),
        "unattempted_count": len(ordered_candidates) - len(results),
        "decision_count": len(decisions),
        "accepted_count": sum(decision.accepted for decision in decisions),
        "orchestration_error_count": sum(
            result.error is not None for result in results
        ),
        "resource_lock": str(Path(args.resource_lock).expanduser().resolve()),
    }


def _qualify_command(args: argparse.Namespace, hooks: CLIHooks) -> Mapping[str, Any]:
    try:
        with BenchmarkResourceLock(args.resource_lock, command="qualify"):
            return _qualify_command_locked(args, hooks)
    except BenchmarkResourceLockError as error:
        raise CLIError(str(error)) from error


def _freeze_command(args: argparse.Namespace, hooks: CLIHooks) -> Mapping[str, Any]:
    pool = load_candidate_pool(args.pool)
    qualification_strata = _qualification_strata(pool)
    ledger_artifact, decisions = load_qualification_ledger(args.qualification)
    if ledger_artifact.get("source_pool_sha256") != pool.sha256:
        raise CLIError("qualification ledger belongs to a different candidate pool")
    raw_model_identity = ledger_artifact.get("model_identity")
    if not isinstance(raw_model_identity, Mapping):
        raise CLIError("qualification ledger is missing its pinned model_identity")
    try:
        qualification_model_identity = ModelIdentity(**raw_model_identity)
    except (TypeError, ValueError) as error:
        raise CLIError(f"qualification model_identity is invalid: {error}") from error
    qualification_model_id = build_model_id(qualification_model_identity)
    if ledger_artifact.get("model_id") != qualification_model_id:
        raise CLIError("qualification ledger model_id is inconsistent")
    if qualification_model_identity != hooks.model_identity():
        raise CLIError(
            "model implementation/config changed after qualification; "
            "re-run qualification before freezing"
        )
    mode = ledger_artifact.get("qualification_mode")
    try:
        normalized_mode = BenchmarkMode(str(mode)).value
    except ValueError as error:
        raise CLIError(
            "qualification ledger has no valid qualification_mode"
        ) from error
    qualified_at = args.qualification_completed_at or ledger_artifact.get(
        "completed_at_utc"
    )
    if not qualified_at:
        raise CLIError("qualification completion timestamp is unavailable")
    frozen_at = args.manifest_frozen_at or hooks.now_utc()
    manifest = hooks.freeze(
        pool,
        decisions,
        freeze_metadata=FreezeMetadata(
            qualification_completed_at_utc=str(qualified_at),
            manifest_frozen_at_utc=str(frozen_at),
            selection_rule=(
                OVERCOMPLETE_SELECTION_RULE
                if qualification_strata is not None
                else SELECTION_RULE
            ),
        ),
    )
    identity_payload = {
        "manifest": manifest.artifact_payload(),
        "qualification_mode": normalized_mode,
        "model_identity": qualification_model_identity,
        "model_id": qualification_model_id,
    }
    artifact = {
        "artifact_type": "mlnd_branch_cut_benchmark_manifest",
        "artifact_schema_version": 1,
        **identity_payload,
        "benchmark_manifest_artifact_sha256": canonical_sha256(identity_payload),
    }
    output = write_canonical_json_exclusive(args.output, artifact)
    return {
        "command": "freeze",
        "output": output,
        "manifest_sha256": manifest.sha256,
        "benchmark_manifest_artifact_sha256": artifact[
            "benchmark_manifest_artifact_sha256"
        ],
        "case_count": len(manifest.cases),
        "qualification_mode": normalized_mode,
        "model_id": qualification_model_id,
    }


def _freeze_formal_plan_command(
    args: argparse.Namespace,
    hooks: CLIHooks,
) -> Mapping[str, Any]:
    """Freeze the complete 36-case A--E execution matrix.

    This is the only command that accepts formal solver/start settings.  The
    corresponding execution command consumes the resulting plan verbatim and
    deliberately exposes no axes or tuning overrides.
    """

    output_target = _require_absent_output(
        args.output,
        label="formal execution plan",
    )
    loaded = load_benchmark_manifest(args.manifest)
    current_model_identity = hooks.model_identity()
    if loaded.model_identity != current_model_identity:
        raise CLIError(
            "current model implementation/config does not match the frozen manifest"
        )
    if loaded.qualification_mode is None:
        raise CLIError(
            "benchmark manifest has no qualification mode; it cannot seed a "
            "formal execution plan"
        )
    suite = _solver_suite(args)
    implementation_sha256 = hooks.formal_execution_implementation_sha256()
    gurobi_version = hooks.gurobi_version()
    try:
        environment_capture = hooks.formal_environment_capture()
    except FormalEnvironmentError as error:
        raise CLIError(str(error)) from error
    if not isinstance(environment_capture, FormalEnvironmentCapture):
        raise CLIError("formal environment capture hook returned an invalid value")
    if environment_capture.contract.gurobipy.native_version != gurobi_version:
        raise CLIError(
            "formal environment native Gurobi version differs from the solver hook"
        )
    plan = build_formal_execution_plan(
        loaded.manifest,
        manifest_artifact_sha256=loaded.identity_sha256,
        model_id=loaded.model_id,
        formal_execution_implementation_sha256=implementation_sha256,
        qualification_mode=BenchmarkMode(loaded.qualification_mode),
        gurobi_version=gurobi_version,
        environment_contract=environment_capture.contract,
        suite=suite,
        analysis_contract=_formal_analysis_contract_from_freeze_args(args),
    )
    output = write_canonical_json_exclusive(output_target, plan.artifact_payload())
    return {
        "command": "freeze-formal-plan",
        "output": output,
        "formal_plan_id": plan.plan_id,
        "formal_plan_sha256": plan.sha256,
        "formal_execution_implementation_sha256": implementation_sha256,
        "manifest_id": loaded.manifest_id,
        "model_id": loaded.model_id,
        "gurobi_version": gurobi_version,
        "formal_environment_id": environment_capture.contract.environment_id,
        "formal_environment_sha256": environment_capture.contract.sha256,
        "formal_analysis_contract_sha256": plan.analysis_contract.sha256,
        "case_count": plan.expected_case_count,
        "block_count": plan.expected_block_count,
        "slot_count": plan.expected_slot_count,
    }


def _formal_analysis_contract_from_freeze_args(
    args: argparse.Namespace,
) -> FormalAnalysisContract:
    """Build the sole mutable-entry-point preregistration before it is hashed."""

    try:
        return FormalAnalysisContract(
            primary_metric=args.analysis_primary_metric,
            primary_absolute_tolerance=args.analysis_absolute_tolerance,
            primary_relative_tolerance=args.analysis_relative_tolerance,
            minimum_medium_large_win_fraction=(
                args.analysis_minimum_medium_large_win_fraction
            ),
            minimum_improved_scale_tiers=(args.analysis_minimum_improved_scale_tiers),
            minimum_improved_instances=args.analysis_minimum_improved_instances,
            minimum_runtime_speedup=args.analysis_minimum_runtime_speedup,
            minimum_gap_reduction=args.analysis_minimum_gap_reduction,
            confidence_level=args.analysis_confidence,
            bootstrap_seed=args.analysis_bootstrap_seed,
            bootstrap_resamples=args.analysis_bootstrap_resamples,
            minimum_effect_cases=args.analysis_minimum_effect_cases,
            runtime_shift=args.analysis_runtime_shift,
            gap_shift=args.analysis_gap_shift,
            key_regression_metrics=tuple(args.analysis_key_regression_metrics),
            regression_fraction=args.analysis_regression_fraction,
            regression_minimum_pairs=args.analysis_regression_minimum_pairs,
            sign_test_alternative=args.analysis_sign_test_alternative,
            sign_test_zero_tolerance=args.analysis_sign_test_zero_tolerance,
            sign_test_alpha=1.0 - args.analysis_confidence,
            performance_taus=tuple(args.analysis_performance_taus),
            runtime_consistency_relative_tolerance=(
                args.analysis_runtime_consistency_relative_tolerance
            ),
            missing_gap_penalty=args.analysis_missing_gap_penalty,
            target_absolute_tolerance=args.analysis_target_absolute_tolerance,
            target_relative_tolerance=args.analysis_target_relative_tolerance,
        )
    except ValueError as error:
        raise CLIError(f"invalid formal analysis preregistration: {error}") from error


def _solver_suite(args: argparse.Namespace):
    if args.fixed_integer_repair_seconds > 0.0 and args.local_branching_seconds > 0.0:
        raise CLIError(
            "fixed-integer repair and local branching cannot both have positive "
            "budgets; set --fixed-integer-repair-seconds 0 to enable local branching"
        )
    solver = SolverConfig(
        threads=1,
        method=args.method,
        miqcp_method=args.miqcp_method,
        mip_focus=args.mip_focus,
        heuristics=args.heuristics,
        no_rel_heur_time=args.no_rel_heur_time,
        presolve=args.presolve,
        cuts=args.native_cuts,
        root_cut_passes=args.root_cut_passes,
        numeric_focus=args.numeric_focus,
        target_mip_gap=args.target_mip_gap,
        output_flag=int(args.solver_output),
    )
    heuristic = StartConfig(
        start_mode=StartMode.NETWORK_AND_COURIER,
        network_start_seconds=args.network_start_seconds,
        network_start_method=args.network_start_method,
        fixed_integer_repair_seconds=args.fixed_integer_repair_seconds,
        local_branching=args.local_branching_seconds > 0.0,
        local_branching_seconds=args.local_branching_seconds,
        local_branching_radius=args.local_branching_radius,
        local_branching_include_topology=args.local_branching_include_topology,
    )
    telemetry = TelemetryConfig(
        progress_interval_seconds=args.progress_interval_seconds,
        solver_log=not args.no_solver_log,
    )
    return build_paper_profile_suite(
        solver=solver,
        telemetry=telemetry,
        heuristic_start=heuristic,
    )


def _run_one(task: RunTask, hooks: CLIHooks) -> dict[str, Any]:
    profile_code = task.profile.code
    logical_run_id = "NOT_CLAIMED"
    qualification_case_digest: str | None = None
    effective_config_digest: str | None = None
    try:
        model_contract = _candidate_model_contract(task.candidate)
        qualification_case_digest = model_contract.qualification_case_sha256
        effective_config_digest = model_contract.effective_model_config_sha256
        if hooks.model_identity() != task.model_identity:
            raise CLIError("model implementation/config changed during execution batch")
        actual_hash, _file_hashes = hooks.canonical_hash(task.instance_dir)
        if actual_hash != task.expected_instance_sha256:
            raise CLIError(
                f"instance hash mismatch for {task.case_id}: "
                f"expected {task.expected_instance_sha256}, got {actual_hash}"
            )
        logical_run_id = build_logical_run_id(task.pair, task.profile)
        claim = claim_attempt_directory(
            task.results_root,
            logical_run_id,
            task.attempt_number,
        )
        instance = hooks.load_instance(task.instance_dir, task.candidate)
        result = hooks.execute(
            instance,
            task.pair,
            task.profile,
            task.telemetry,
            claim,
        )
        return {
            "case_id": task.case_id,
            "scale": task.scale,
            "mode": task.pair.mode.value,
            "solver_seed": task.pair.solver_seed,
            "time_budget_seconds": task.pair.time_budget_seconds,
            "profile_code": profile_code.value,
            "pair_id": build_pair_id(task.pair),
            "logical_run_id": logical_run_id,
            "attempt_id": result.attempt_id,
            "attempt_directory": result.directory,
            "outcome": result.outcome,
            "has_incumbent": result.has_incumbent,
            "qualification_solve_profile": task.candidate.generator_parameters.get(
                "qualification_solve_profile"
            ),
            "qualification_case_sha256": qualification_case_digest,
            "effective_model_config_sha256": effective_config_digest,
            "orchestration_error": None,
        }
    except Exception as error:
        return {
            "case_id": task.case_id,
            "scale": task.scale,
            "mode": task.pair.mode.value,
            "solver_seed": task.pair.solver_seed,
            "time_budget_seconds": task.pair.time_budget_seconds,
            "profile_code": profile_code.value,
            "pair_id": build_pair_id(task.pair),
            "logical_run_id": logical_run_id,
            "attempt_id": None,
            "attempt_directory": None,
            "outcome": "orchestration_failed",
            "has_incumbent": False,
            "qualification_solve_profile": task.candidate.generator_parameters.get(
                "qualification_solve_profile"
            ),
            "qualification_case_sha256": qualification_case_digest,
            "effective_model_config_sha256": effective_config_digest,
            "orchestration_error": f"{type(error).__name__}: {error}",
        }


def _run_one_default(task: RunTask) -> dict[str, Any]:
    return _run_one(task, DEFAULT_HOOKS)


def _run_command_locked(
    args: argparse.Namespace,
    hooks: CLIHooks,
) -> Mapping[str, Any]:
    batch_target = _require_absent_output(
        args.batch_output,
        label="execution batch output",
    )
    loaded = load_benchmark_manifest(args.manifest)
    if loaded.model_identity != hooks.model_identity():
        raise CLIError(
            "current model implementation/config does not match the frozen manifest"
        )
    suite = _solver_suite(args)
    requested_modes = tuple(
        BenchmarkMode(mode).value
        for mode in (
            args.modes
            or ((loaded.qualification_mode,) if loaded.qualification_mode else ())
        )
    )
    if not requested_modes:
        raise CLIError(
            "manifest has no qualification mode; pass --modes and "
            "--allow-unqualified-modes for exploratory runs"
        )
    if (
        loaded.qualification_mode is None
        or any(mode != loaded.qualification_mode for mode in requested_modes)
    ) and not args.allow_unqualified_modes:
        raise CLIError(
            "run modes must equal the manifest qualification mode; use "
            "--allow-unqualified-modes only for explicitly exploratory runs"
        )
    _require_unique(requested_modes, label="--modes")
    _require_unique(args.profiles, label="--profiles")
    _require_unique(args.solver_seeds, label="--solver-seeds")
    _require_unique(args.time_budgets, label="--time-budgets")
    _require_unique(args.case_ids or (), label="--case-ids")
    wanted_profiles = tuple(ProfileCode(code) for code in args.profiles)
    wanted_case_ids = set(args.case_ids or ())
    known_case_ids = {frozen.candidate.case_id for frozen in loaded.manifest.cases}
    unknown_case_ids = sorted(wanted_case_ids - known_case_ids)
    if unknown_case_ids:
        raise CLIError(f"unknown --case-ids: {unknown_case_ids}")
    selected_cases = tuple(
        frozen
        for frozen in loaded.manifest.cases
        if not wanted_case_ids or frozen.candidate.case_id in wanted_case_ids
    )
    if not selected_cases:
        raise CLIError("run selection contains no frozen cases")
    instances_root = Path(args.instances_root)
    results_root = Path(args.results_root).expanduser().resolve()
    tasks: list[RunTask] = []
    for frozen in selected_cases:
        candidate = frozen.candidate
        instance_dir = _resolved_instance_path(instances_root, candidate.instance_ref)
        for mode in requested_modes:
            for solver_seed in args.solver_seeds:
                for time_budget in args.time_budgets:
                    pair = PairSpec(
                        manifest_id=loaded.manifest_id,
                        case_id=candidate.case_id,
                        mode=BenchmarkMode(mode),
                        solver_seed=solver_seed,
                        time_budget_seconds=time_budget,
                        solver=suite.solver,
                    )
                    for profile_code in wanted_profiles:
                        tasks.append(
                            RunTask(
                                candidate=candidate,
                                instance_dir=instance_dir,
                                model_identity=loaded.model_identity,
                                results_root=results_root,
                                pair=pair,
                                profile=suite.profile(profile_code),
                                telemetry=suite.telemetry,
                                attempt_number=args.attempt_number,
                            )
                        )
    if args.jobs > 1:
        if hooks is not DEFAULT_HOOKS:
            raise CLIError("custom CLI hooks support only --jobs 1")
        with ProcessPoolExecutor(max_workers=args.jobs) as executor:
            results = tuple(executor.map(_run_one_default, tasks, chunksize=1))
    else:
        results = tuple(_run_one(task, hooks) for task in tasks)
    batch = {
        "artifact_type": "mlnd_branch_cut_execution_batch",
        "artifact_schema_version": 1,
        "created_at_utc": hooks.now_utc(),
        "manifest_id": loaded.manifest_id,
        "model_id": loaded.model_id,
        "qualification_mode": loaded.qualification_mode,
        "exploratory_unqualified_modes": any(
            mode != loaded.qualification_mode for mode in requested_modes
        ),
        "profiles": [code.value for code in wanted_profiles],
        "modes": list(requested_modes),
        "solver_seeds": list(args.solver_seeds),
        "time_budgets_seconds": list(args.time_budgets),
        "jobs": args.jobs,
        "resource_lock": str(Path(args.resource_lock).expanduser().resolve()),
        "task_count": len(tasks),
        "results": list(results),
    }
    output = write_canonical_json_exclusive(batch_target, batch)
    return {
        "command": "run",
        "batch_output": output,
        "manifest_id": loaded.manifest_id,
        "task_count": len(tasks),
        "completed_count": sum(
            result["outcome"] != "orchestration_failed" for result in results
        ),
        "orchestration_error_count": sum(
            result["outcome"] == "orchestration_failed" for result in results
        ),
        "resource_lock": str(Path(args.resource_lock).expanduser().resolve()),
    }


def _run_command(args: argparse.Namespace, hooks: CLIHooks) -> Mapping[str, Any]:
    try:
        with BenchmarkResourceLock(args.resource_lock, command="run"):
            return _run_command_locked(args, hooks)
    except BenchmarkResourceLockError as error:
        raise CLIError(str(error)) from error


def _run_formal_command_locked(
    args: argparse.Namespace,
    hooks: CLIHooks,
) -> Mapping[str, Any]:
    """Execute only the externally anchored, immutable formal plan."""

    loaded = load_benchmark_manifest(args.manifest)
    plan_artifact = _read_json_object(args.plan)
    plan = formal_plan_from_artifact(plan_artifact, loaded.manifest)
    if plan.plan_id != args.expected_plan_id:
        raise CLIError("--expected-plan-id does not equal the canonical formal plan ID")

    # Import the complete frozen solver runtime before the first claim, then
    # capture the stable environment/host once for this run session.  Manifest,
    # model, implementation, and Gurobi identities are still recomputed before
    # every pending attempt by current_binding_provider below.
    hooks.preload_formal_runtime()
    try:
        environment_capture = hooks.formal_environment_capture()
    except FormalEnvironmentError as error:
        raise CLIError(str(error)) from error
    if not isinstance(environment_capture, FormalEnvironmentCapture):
        raise CLIError("formal environment capture hook returned an invalid value")

    frozen_by_case = {
        frozen.candidate.case_id: frozen for frozen in loaded.manifest.cases
    }
    if set(frozen_by_case) != {block.case_id for block in plan.blocks}:
        raise CLIError("formal plan case set differs from the frozen manifest")
    instances_root = Path(args.instances_root).expanduser().resolve()

    def current_binding_provider() -> FormalExecutionBinding:
        observed = load_benchmark_manifest(args.manifest)
        if observed.qualification_mode is None:
            raise CLIError("benchmark manifest has no qualification mode")
        current_model_id = build_model_id(hooks.model_identity())
        return FormalExecutionBinding(
            benchmark_manifest_sha256=observed.manifest.sha256,
            manifest_artifact_sha256=observed.identity_sha256,
            manifest_id=observed.manifest_id,
            model_id=current_model_id,
            formal_execution_implementation_sha256=(
                hooks.formal_execution_implementation_sha256()
            ),
            qualification_mode=BenchmarkMode(observed.qualification_mode),
            gurobi_version=hooks.gurobi_version(),
            environment_contract=environment_capture.contract,
            host_identity_sha256=environment_capture.host_identity_sha256,
        )

    def validate_candidate_instance(candidate: CandidateCase) -> Path:
        instance_dir = _resolved_instance_path(
            instances_root,
            candidate.instance_ref,
        )
        actual_sha256, _file_sha256 = hooks.canonical_hash(instance_dir)
        if actual_sha256 != candidate.instance_sha256:
            raise CLIError(
                f"instance hash mismatch for {candidate.case_id}: "
                f"expected {candidate.instance_sha256}, got {actual_sha256}"
            )
        return instance_dir

    # Fail before creating a campaign directory when any one of the 36 frozen
    # instances is absent or has drifted.  Pending slots are checked again
    # immediately before claim so a later mutation cannot enter the journal.
    try:
        current_binding_provider().validate_plan(plan)
    except FormalExecutionError as error:
        raise CLIError(str(error)) from error
    for frozen in loaded.manifest.cases:
        validate_candidate_instance(frozen.candidate)

    def pre_claim_preparer(task: FormalSlotTask) -> PreparedFormalInput:
        frozen_case = frozen_by_case[task.block.case_id]
        candidate = frozen_case.candidate
        qualification_trace = FormalQualificationTrace.from_decision(
            frozen_case.qualification,
            manifest_id=loaded.manifest_id,
            benchmark_manifest_sha256=loaded.manifest.sha256,
            manifest_artifact_sha256=loaded.identity_sha256,
        )
        instance_dir = _resolved_instance_path(
            instances_root,
            candidate.instance_ref,
        )
        if hooks.formal_input_preparer is not None:
            return hooks.formal_input_preparer(
                task,
                instance_dir,
                candidate,
                qualification_trace,
            )
        return _prepare_formal_input_snapshot(
            task,
            instance_dir=instance_dir,
            candidate=candidate,
            qualification_trace=qualification_trace,
            canonical_hash=hooks.canonical_hash,
            loader=hooks.load_instance,
        )

    def execute_slot(
        task: FormalSlotTask,
        claim: Any,
        prepared: PreparedFormalInput,
    ) -> object:
        return hooks.execute(
            prepared.payload,
            task.pair,
            task.profile,
            task.telemetry,
            claim,
            formal_context=bind_qualification_trace(
                task.formal_context,
                prepared.attestation.qualification_trace,
            ),
        )

    try:
        result = hooks.execute_formal(
            plan,
            args.results_root,
            current_binding_provider,
            pre_claim_preparer,
            execute_slot,
            session_snapshot=environment_capture.session,
            resource_lock_path=str(Path(args.resource_lock).expanduser().resolve()),
            now_utc=hooks.now_utc,
        )
    except FormalExecutionError as error:
        raise CLIError(str(error)) from error
    if not isinstance(result, FormalCampaignRunResult):
        raise CLIError("formal executor returned an invalid campaign result")
    campaign_root = (
        Path(args.results_root).expanduser().resolve()
        / plan.plan_id
        / result.campaign_id
    )
    return {
        "command": "run-formal",
        "formal_plan_id": plan.plan_id,
        "campaign_id": result.campaign_id,
        "session_id": result.session_id,
        "campaign_root": campaign_root,
        "initial_completed_count": result.initial_completed_count,
        "recovered_count": result.recovered_count,
        "executed_count": result.executed_count,
        "skipped_count": result.skipped_count,
        "final_completed_count": result.final_completed_count,
        "evidence_id": result.evidence_id,
        "evidence_root_sha256": result.evidence_root_sha256,
        "resource_lock": str(Path(args.resource_lock).expanduser().resolve()),
    }


def _run_formal_command(
    args: argparse.Namespace,
    hooks: CLIHooks,
) -> Mapping[str, Any]:
    try:
        with BenchmarkResourceLock(args.resource_lock, command="run-formal"):
            return _run_formal_command_locked(args, hooks)
    except BenchmarkResourceLockError as error:
        raise CLIError(str(error)) from error


@dataclass(frozen=True, slots=True)
class _ObservedRun:
    pair_id: str
    profile_code: str
    pair: Mapping[str, Any]
    record: Mapping[str, Any]
    attempt_directory: Path | None
    authenticated_progress: tuple[Mapping[str, object], ...] | None = None
    metrics_eligible: bool = True
    metrics_exclusion_reason: str | None = None


@dataclass(frozen=True, slots=True)
class _DerivedProgress:
    """Validated progress evidence for one run on the matched horizon."""

    availability: str
    issues: tuple[str, ...]
    progress_path: Path | None
    event_count: int
    horizon: float
    target_absolute_tolerance: float
    target_relative_tolerance: float
    preparation_time: float | None
    preparation_source: str | None
    trajectory: ProgressTrajectory | None
    integral: PrimalDualIntegral | None
    status_name: str
    status_within_horizon: bool
    termination_label: str
    final_incumbent: float | None
    final_bound: float | None

    @property
    def usable(self) -> bool:
        return self.availability == "available"

    def time_to_target(self, target: float | None) -> float | None:
        if not self.usable or self.trajectory is None or target is None:
            return None
        return time_to_target(
            self.trajectory.points,
            target,
            self.horizon,
            sense="min",
            absolute_tolerance=self.target_absolute_tolerance,
            relative_tolerance=self.target_relative_tolerance,
        )

    def audit_payload(
        self,
        *,
        pair_id: str,
        profile_code: str,
        target: float | None,
        target_unavailable_reason: str | None,
    ) -> dict[str, Any]:
        target_time = self.time_to_target(target)
        target_is_evaluable = self.usable and target is not None
        return {
            "pair_id": pair_id,
            "profile_code": profile_code,
            "progress_artifact": (
                self.progress_path.as_posix()
                if self.progress_path is not None
                else None
            ),
            "progress_availability": self.availability,
            "progress_issues": list(self.issues),
            "progress_event_count": self.event_count,
            "uniform_horizon_seconds": self.horizon,
            "preparation_time_seconds": self.preparation_time,
            "preparation_time_source": self.preparation_source,
            "status_name": self.status_name,
            "status_within_horizon": self.status_within_horizon,
            "termination_label": self.termination_label,
            "proven_optimal_within_horizon": bool(
                self.status_within_horizon
                and self.status_name == "OPTIMAL"
                and self.final_incumbent is not None
            ),
            "time_limited": self.status_name == "TIME_LIMIT",
            "has_incumbent_within_horizon": self.final_incumbent is not None,
            "final_incumbent": self.final_incumbent,
            "final_bound": self.final_bound,
            "terminal_normalized_gap": (
                self.integral.terminal_gap if self.integral is not None else None
            ),
            "first_incumbent_time_seconds": (
                self.integral.first_incumbent_time
                if self.integral is not None
                else None
            ),
            "primal_dual_integral": (
                self.integral.integral if self.integral is not None else None
            ),
            "normalized_primal_dual_integral": (
                self.integral.average_gap if self.integral is not None else None
            ),
            "common_target": target,
            "common_target_unavailable_reason": target_unavailable_reason,
            "time_to_common_target_seconds": target_time,
            "time_to_common_target_censored": (
                target_time is None if target_is_evaluable else None
            ),
        }


def _pair_from_payload(payload: Mapping[str, Any]) -> PairSpec:
    solver = payload.get("solver")
    if not isinstance(solver, Mapping):
        raise CLIError("run_spec pair is missing its matched solver configuration")
    try:
        return PairSpec(
            manifest_id=str(payload["manifest_id"]),
            case_id=str(payload["case_id"]),
            mode=BenchmarkMode(str(payload["mode"])),
            solver_seed=int(payload["solver_seed"]),
            time_budget_seconds=float(payload["time_budget_seconds"]),
            solver=SolverConfig(**solver),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise CLIError(f"invalid pair payload in run_spec: {error}") from error


def _optional_finite(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _load_observed_runs(
    results_root: Path,
    *,
    manifest_id: str,
    attempt_number: int,
) -> tuple[_ObservedRun, ...]:
    observed: dict[tuple[str, str], _ObservedRun] = {}
    for record_path in sorted(
        results_root.expanduser().resolve().rglob("run_record.json")
    ):
        record = _read_json_object(record_path)
        ids = record.get("ids")
        if not isinstance(ids, Mapping) or ids.get("attempt_number") != attempt_number:
            continue
        spec_path = record_path.with_name("run_spec.json")
        spec = _read_json_object(spec_path)
        pair = spec.get("pair")
        if not isinstance(pair, Mapping) or pair.get("manifest_id") != manifest_id:
            continue
        pair_spec = _pair_from_payload(pair)
        profile_code = str(record.get("profile_code", ""))
        try:
            normalized_profile = ProfileCode(profile_code).value
        except ValueError as error:
            raise CLIError(
                f"invalid profile code in {record_path}: {profile_code}"
            ) from error
        pair_id = str(ids.get("pair_id", ""))
        if not pair_id:
            raise CLIError(f"run record is missing pair_id: {record_path}")
        if pair_id != build_pair_id(pair_spec):
            raise CLIError(f"run record pair_id does not match run_spec: {record_path}")
        key = (pair_id, normalized_profile)
        if key in observed:
            raise CLIError(f"duplicate attempt {attempt_number} for pair/profile {key}")
        observed[key] = _ObservedRun(
            pair_id,
            normalized_profile,
            pair,
            record,
            record_path.parent,
        )
    return tuple(observed[key] for key in sorted(observed))


@dataclass(frozen=True, slots=True)
class _FormalAnalysisAudit:
    """Authenticated formal-slot coverage supplied to the shared analyzer."""

    formal_plan_id: str
    formal_plan_sha256: str
    campaign_id: str
    campaign_identity_sha256: str
    analysis_session_id: str
    expected_evidence_id: str
    expected_evidence_root_sha256: str
    actual_evidence_id: str
    actual_evidence_root_sha256: str
    expected_block_count: int
    expected_slot_count: int
    authenticated_slot_count: int
    failed_slot_ids: tuple[str, ...]

    @property
    def all_slots_succeeded(self) -> bool:
        return not self.failed_slot_ids

    def to_payload(self) -> dict[str, Any]:
        return {
            "formal_plan_id": self.formal_plan_id,
            "formal_plan_sha256": self.formal_plan_sha256,
            "campaign_id": self.campaign_id,
            "campaign_identity_sha256": self.campaign_identity_sha256,
            "analysis_session_id": self.analysis_session_id,
            "expected_evidence_id": self.expected_evidence_id,
            "expected_evidence_root_sha256": (self.expected_evidence_root_sha256),
            "actual_evidence_id": self.actual_evidence_id,
            "actual_evidence_root_sha256": self.actual_evidence_root_sha256,
            "expected_block_count": self.expected_block_count,
            "expected_slot_count": self.expected_slot_count,
            "authenticated_slot_count": self.authenticated_slot_count,
            "failed_slot_count": len(self.failed_slot_ids),
            "failed_slot_ids": list(self.failed_slot_ids),
            "all_slots_succeeded": self.all_slots_succeeded,
            "attempt_number": FORMAL_ATTEMPT_NUMBER,
        }


def _load_authenticated_formal_runs(
    args: argparse.Namespace,
    hooks: CLIHooks,
) -> tuple[
    LoadedManifest,
    tuple[_ObservedRun, ...],
    _FormalAnalysisAudit,
    FormalExecutionPlan,
    FormalCampaign,
]:
    """Authenticate the exact frozen 900-slot campaign before analysis.

    The ordinary ``analyze`` command intentionally remains permissive and
    exploratory.  This path instead reopens the formal campaign through the
    same identity/bundle/checkpoint machinery as execution, rejects incomplete
    or extra state, and consumes only the authenticated immutable snapshots
    returned by reconciliation.  Attempt paths are never reopened here.
    """

    loaded = load_benchmark_manifest(args.manifest)
    plan = formal_plan_from_artifact(
        _read_json_object(args.plan),
        loaded.manifest,
    )
    if plan.plan_id != args.expected_plan_id:
        raise CLIError("--expected-plan-id does not equal the canonical formal plan ID")
    if (
        len(plan.blocks) != FORMAL_BLOCK_COUNT
        or sum(len(block.slots) for block in plan.blocks) != FORMAL_SLOT_COUNT
    ):
        raise CLIError(
            "formal plan does not contain the exact 180-block/900-slot matrix"
        )
    expected_evidence_root_sha256 = _require_sha256(
        args.expected_evidence_root,
        label="--expected-evidence-root",
    )
    expected_evidence_id = f"formal-evidence-{expected_evidence_root_sha256}"

    results_root = Path(args.results_root).expanduser().resolve()
    plan_root = results_root / plan.plan_id
    if not plan_root.is_dir():
        raise CLIError(f"formal plan result root is missing: {plan_root}")

    hooks.preload_formal_runtime()
    try:
        environment_capture = hooks.formal_environment_capture()
    except FormalEnvironmentError as error:
        raise CLIError(str(error)) from error
    if not isinstance(environment_capture, FormalEnvironmentCapture):
        raise CLIError("formal environment capture hook returned an invalid value")

    def current_binding_provider() -> FormalExecutionBinding:
        observed = load_benchmark_manifest(args.manifest)
        if observed.qualification_mode is None:
            raise CLIError("benchmark manifest has no qualification mode")
        return FormalExecutionBinding(
            benchmark_manifest_sha256=observed.manifest.sha256,
            manifest_artifact_sha256=observed.identity_sha256,
            manifest_id=observed.manifest_id,
            model_id=build_model_id(hooks.model_identity()),
            formal_execution_implementation_sha256=(
                hooks.formal_execution_implementation_sha256()
            ),
            qualification_mode=BenchmarkMode(observed.qualification_mode),
            gurobi_version=hooks.gurobi_version(),
            environment_contract=environment_capture.contract,
            host_identity_sha256=environment_capture.host_identity_sha256,
        )

    def no_pending_input_may_be_prepared(_task: FormalSlotTask) -> PreparedFormalInput:
        raise FormalExecutionError(
            "formal analysis encountered a pending slot during reconciliation"
        )

    campaign: FormalCampaign | None = None
    try:
        campaign = hooks.prepare_formal(
            plan,
            results_root,
            current_binding_provider,
            no_pending_input_may_be_prepared,
            session_snapshot=environment_capture.session,
            resource_lock_path=str(Path(args.resource_lock).expanduser().resolve()),
            now_utc=hooks.now_utc,
        )
        if not isinstance(campaign, FormalCampaign):
            raise CLIError("formal campaign preparer returned an invalid campaign")
        state = hooks.reconcile_formal(
            campaign,
            expected_evidence_root_sha256=expected_evidence_root_sha256,
        )
        if not isinstance(state, FormalCampaignState):
            raise CLIError("formal campaign reconciler returned an invalid state")

        expected_order = tuple(
            (block, slot) for block in plan.blocks for slot in block.slots
        )
        expected_slot_ids = tuple(slot.slot_id for _block, slot in expected_order)
        if len(set(expected_slot_ids)) != FORMAL_SLOT_COUNT:
            raise CLIError("formal plan slot IDs are not globally unique")
        completed = set(state.completed_slot_ids)
        if (
            state.pending_slot_ids
            or len(state.completed_slot_ids) != FORMAL_SLOT_COUNT
            or completed != set(expected_slot_ids)
        ):
            missing = sorted(set(expected_slot_ids) - completed)
            raise CLIError(
                "formal analysis requires all 900 authenticated attempt-1 slots; "
                f"missing {len(missing)}"
            )
        if (
            state.evidence_root_sha256 != expected_evidence_root_sha256
            or state.evidence_id != expected_evidence_id
        ):
            raise CLIError(
                "formal reconciler evidence identity differs from the external "
                "expected evidence root"
            )
        snapshots = state.authenticated_slots
        if len(snapshots) != FORMAL_SLOT_COUNT:
            raise CLIError(
                "formal analysis requires exactly 900 authenticated in-memory "
                "slot snapshots"
            )

        observed_runs: list[_ObservedRun] = []
        failed_slot_ids: list[str] = []
        for snapshot, (block, slot) in zip(
            snapshots,
            expected_order,
            strict=True,
        ):
            if not isinstance(snapshot, AuthenticatedFormalSlotSnapshot):
                raise CLIError(
                    "formal reconciler returned an invalid authenticated slot snapshot"
                )
            if (
                snapshot.slot_id != slot.slot_id
                or snapshot.slot_index != slot.slot_index
                or snapshot.block_id != block.block_id
                or snapshot.logical_run_id != slot.logical_run_id
            ):
                raise CLIError(
                    f"authenticated snapshot order/identity differs for {slot.slot_id}"
                )
            run_spec = snapshot.run_spec
            record = snapshot.run_record
            pair = run_spec.get("pair")
            ids = record.get("ids")
            formal_context = record.get("formal_context")
            if (
                not isinstance(pair, Mapping)
                or not isinstance(ids, Mapping)
                or not isinstance(formal_context, Mapping)
            ):
                raise CLIError(
                    "authenticated formal snapshot lacks pair, IDs, or formal context"
                )
            pair_spec = _pair_from_payload(pair)
            profile_code = str(record.get("profile_code", ""))
            try:
                normalized_profile = ProfileCode(profile_code).value
            except ValueError as error:
                raise CLIError(
                    f"invalid authenticated profile code for {slot.slot_id}: "
                    f"{profile_code}"
                ) from error
            if (
                pair.get("manifest_id") != loaded.manifest_id
                or build_pair_id(pair_spec) != slot.pair_id
                or ids.get("pair_id") != slot.pair_id
                or ids.get("logical_run_id") != slot.logical_run_id
                or ids.get("attempt_id") != slot.attempt_id
                or ids.get("attempt_number") != FORMAL_ATTEMPT_NUMBER
                or formal_context.get("slot_id") != slot.slot_id
                or normalized_profile != slot.profile_code.value
                or record.get("outcome") != snapshot.outcome
                or record.get("completed_without_exception")
                is not snapshot.completed_without_exception
            ):
                raise CLIError(
                    f"authenticated formal run identity differs for {slot.slot_id}"
                )
            run = _ObservedRun(
                pair_id=slot.pair_id,
                profile_code=normalized_profile,
                pair=pair,
                record=record,
                attempt_directory=None,
                authenticated_progress=snapshot.progress,
            )
            failed = (
                snapshot.outcome == "failed" or not snapshot.completed_without_exception
            )
            if failed:
                failed_slot_ids.append(slot.slot_id)
                run = replace(
                    run,
                    metrics_eligible=False,
                    metrics_exclusion_reason="formal_attempt_failed",
                )
            observed_runs.append(run)

        ordered = tuple(observed_runs)
        audit = _FormalAnalysisAudit(
            formal_plan_id=plan.plan_id,
            formal_plan_sha256=plan.sha256,
            campaign_id=campaign.identity.campaign_id,
            campaign_identity_sha256=campaign.identity.sha256,
            analysis_session_id=campaign.session.session_id,
            expected_evidence_id=expected_evidence_id,
            expected_evidence_root_sha256=expected_evidence_root_sha256,
            actual_evidence_id=state.evidence_id,
            actual_evidence_root_sha256=state.evidence_root_sha256,
            expected_block_count=len(plan.blocks),
            expected_slot_count=len(expected_slot_ids),
            authenticated_slot_count=len(ordered),
            failed_slot_ids=tuple(sorted(failed_slot_ids)),
        )
    except BaseException as error:
        if isinstance(campaign, FormalCampaign):
            try:
                finish_formal_session(
                    campaign,
                    status="failed",
                    now_utc=hooks.now_utc,
                )
            except Exception as session_error:
                raise CLIError(
                    "formal analysis failed and its session end could not be "
                    f"published: {type(session_error).__name__}: {session_error}"
                ) from error
        if isinstance(error, FormalExecutionError):
            raise CLIError(str(error)) from error
        raise
    return loaded, ordered, audit, plan, campaign


def _solver_metrics(run: _ObservedRun | None) -> Mapping[str, Any]:
    if run is None or not run.metrics_eligible:
        return {}
    metrics = run.record.get("solver_metrics")
    return metrics if isinstance(metrics, Mapping) else {}


def _progress_failure(
    run: _ObservedRun | None,
    horizon: float,
    *,
    availability: str,
    issues: Sequence[str],
    progress_path: Path | None,
    event_count: int = 0,
    preparation_time: float | None = None,
    preparation_source: str | None = None,
    target_absolute_tolerance: float = 1e-9,
    target_relative_tolerance: float = 1e-9,
) -> _DerivedProgress:
    metrics = _solver_metrics(run)
    status = solver_status_name(metrics.get("status_name", metrics.get("status")))
    raw_has_incumbent = metrics.get("has_incumbent")
    has_incumbent = raw_has_incumbent if isinstance(raw_has_incumbent, bool) else False
    return _DerivedProgress(
        availability=availability,
        issues=tuple(issues),
        progress_path=progress_path,
        event_count=event_count,
        horizon=horizon,
        target_absolute_tolerance=target_absolute_tolerance,
        target_relative_tolerance=target_relative_tolerance,
        preparation_time=preparation_time,
        preparation_source=preparation_source,
        trajectory=None,
        integral=None,
        status_name=status,
        status_within_horizon=False,
        termination_label=(
            "missing_attempt"
            if run is None
            else honest_termination_label(status, has_incumbent=has_incumbent)
        ),
        final_incumbent=None,
        final_bound=None,
    )


def _read_progress_events(
    run: _ObservedRun | None,
) -> tuple[tuple[Mapping[str, object], ...] | None, Path | None, str, tuple[str, ...]]:
    if run is None:
        return None, None, "missing", ("missing_attempt",)
    progress_path = (
        run.attempt_directory / "progress.jsonl"
        if run.attempt_directory is not None
        else None
    )
    if not run.metrics_eligible:
        return (
            None,
            progress_path,
            "invalid",
            (run.metrics_exclusion_reason or "formal_attempt_not_metric_eligible",),
        )
    if run.authenticated_progress is not None:
        if not run.authenticated_progress:
            return None, None, "missing", ("progress_artifact_has_no_events",)
        return run.authenticated_progress, None, "available", ()
    if progress_path is None:
        return None, None, "invalid", ("missing_progress_evidence_source",)
    if not progress_path.is_file():
        return None, progress_path, "missing", ("missing_progress_artifact",)
    try:
        lines = progress_path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        return (
            None,
            progress_path,
            "invalid",
            (f"unreadable_progress_artifact:{type(error).__name__}",),
        )
    rows: list[Mapping[str, object]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            return (
                None,
                progress_path,
                "invalid",
                (f"invalid_progress_json_line:{line_number}",),
            )
        if not isinstance(row, Mapping):
            return (
                None,
                progress_path,
                "invalid",
                (f"non_object_progress_row:{line_number}",),
            )
        rows.append(row)
    if not rows:
        return None, progress_path, "missing", ("progress_artifact_has_no_events",)
    return tuple(rows), progress_path, "available", ()


def _nonnegative_artifact_number(
    payload: Mapping[str, Any], field_name: str
) -> tuple[float | None, str | None]:
    if field_name not in payload or payload.get(field_name) is None:
        return None, f"missing_{field_name}"
    value = _optional_finite(payload.get(field_name))
    if value is None or value < 0.0:
        return None, f"invalid_{field_name}"
    return value, None


def _preparation_offset(
    metrics: Mapping[str, Any],
    events: Sequence[Mapping[str, object]],
    *,
    runtime_consistency_relative_tolerance: float,
) -> tuple[float | None, str | None, tuple[str, ...]]:
    total, total_issue = _nonnegative_artifact_number(
        metrics, "total_algorithm_runtime_seconds"
    )
    main, main_issue = _nonnegative_artifact_number(
        metrics, "main_solve_runtime_seconds"
    )
    if total is not None and main is not None:
        difference = total - main
        tolerance = runtime_consistency_relative_tolerance * max(1.0, total, main)
        if difference < -tolerance:
            return None, None, ("invalid_total_runtime_less_than_main_runtime",)
        return max(0.0, difference), "telemetry_total_minus_main", ()
    if total_issue is not None and not total_issue.startswith("missing_"):
        return None, None, (total_issue,)
    if main_issue is not None and not main_issue.startswith("missing_"):
        return None, None, (main_issue,)

    for event in reversed(events):
        event_total = _optional_finite(event.get("total_algorithm_runtime_seconds"))
        event_main = _optional_finite(event.get("runtime_seconds"))
        if event_total is None or event_main is None:
            continue
        if event_total < 0.0 or event_main < 0.0:
            return None, None, ("invalid_progress_runtime_for_preparation_offset",)
        difference = event_total - event_main
        tolerance = runtime_consistency_relative_tolerance * max(
            1.0,
            event_total,
            event_main,
        )
        if difference < -tolerance:
            return None, None, ("invalid_progress_total_less_than_main_runtime",)
        return max(0.0, difference), "progress_total_minus_main", ()

    algorithm_prepare, algorithm_issue = _nonnegative_artifact_number(
        metrics, "algorithm_prepare_runtime_seconds"
    )
    start_prepare, start_issue = _nonnegative_artifact_number(
        metrics, "start_preparation_runtime_seconds"
    )
    if algorithm_prepare is not None and start_prepare is not None:
        return (
            algorithm_prepare + start_prepare,
            "telemetry_prepare_components",
            (),
        )
    issues = tuple(
        issue
        for issue in (total_issue, main_issue, algorithm_issue, start_issue)
        if issue is not None
    )
    return None, None, issues or ("missing_preparation_runtime_evidence",)


def _derive_progress(
    run: _ObservedRun | None,
    horizon: float,
    *,
    runtime_consistency_relative_tolerance: float,
    missing_gap_penalty: float,
    target_absolute_tolerance: float,
    target_relative_tolerance: float,
) -> _DerivedProgress:
    rows, progress_path, availability, read_issues = _read_progress_events(run)
    if rows is None:
        return _progress_failure(
            run,
            horizon,
            availability=availability,
            issues=read_issues,
            progress_path=progress_path,
            target_absolute_tolerance=target_absolute_tolerance,
            target_relative_tolerance=target_relative_tolerance,
        )
    metrics = _solver_metrics(run)
    preparation, preparation_source, preparation_issues = _preparation_offset(
        metrics,
        rows,
        runtime_consistency_relative_tolerance=(runtime_consistency_relative_tolerance),
    )
    if preparation is None:
        return _progress_failure(
            run,
            horizon,
            availability="invalid",
            issues=preparation_issues,
            progress_path=progress_path,
            event_count=len(rows),
            target_absolute_tolerance=target_absolute_tolerance,
            target_relative_tolerance=target_relative_tolerance,
        )
    try:
        trajectory = trajectory_from_progress_events(
            rows,
            preparation_time_seconds=preparation,
        )
        integral = primal_dual_integral(
            trajectory.points,
            horizon,
            sense="min",
            missing_gap_penalty=missing_gap_penalty,
        )
        final_incumbent = terminal_best_incumbent(
            trajectory.points,
            horizon,
            sense="min",
        )
        final_bound = terminal_best_bound(
            trajectory.points,
            horizon,
            sense="min",
        )
    except (TypeError, ValueError, OverflowError) as error:
        return _progress_failure(
            run,
            horizon,
            availability="invalid",
            issues=(f"invalid_progress_trajectory:{error}",),
            progress_path=progress_path,
            event_count=len(rows),
            preparation_time=preparation,
            preparation_source=preparation_source,
            target_absolute_tolerance=target_absolute_tolerance,
            target_relative_tolerance=target_relative_tolerance,
        )

    issues: list[str] = []
    progress_status = trajectory.status_name
    telemetry_status = solver_status_name(
        metrics.get("status_name", metrics.get("status"))
    )
    if progress_status == "UNKNOWN" and telemetry_status == "UNKNOWN":
        issues.append("missing_solver_status")
        status = "UNKNOWN"
    elif progress_status == "UNKNOWN":
        issues.append("missing_progress_status_used_telemetry_status")
        status = telemetry_status
    elif telemetry_status == "UNKNOWN":
        issues.append("missing_telemetry_status_used_progress_status")
        status = progress_status
    elif progress_status != telemetry_status:
        return _progress_failure(
            run,
            horizon,
            availability="invalid",
            issues=(
                "solver_status_mismatch:"
                f"progress={progress_status},telemetry={telemetry_status}",
            ),
            progress_path=progress_path,
            event_count=len(rows),
            preparation_time=preparation,
            preparation_source=preparation_source,
            target_absolute_tolerance=target_absolute_tolerance,
            target_relative_tolerance=target_relative_tolerance,
        )
    else:
        status = progress_status

    comparison_horizon = max(horizon, trajectory.observed_end_time)
    progress_has_incumbent = (
        terminal_best_incumbent(
            trajectory.points,
            comparison_horizon,
            sense="min",
        )
        is not None
    )
    raw_has_incumbent = metrics.get("has_incumbent")
    if "has_incumbent" not in metrics:
        issues.append("missing_telemetry_has_incumbent")
    elif not isinstance(raw_has_incumbent, bool):
        return _progress_failure(
            run,
            horizon,
            availability="invalid",
            issues=("invalid_telemetry_has_incumbent",),
            progress_path=progress_path,
            event_count=len(rows),
            preparation_time=preparation,
            preparation_source=preparation_source,
            target_absolute_tolerance=target_absolute_tolerance,
            target_relative_tolerance=target_relative_tolerance,
        )
    elif raw_has_incumbent != progress_has_incumbent:
        return _progress_failure(
            run,
            horizon,
            availability="invalid",
            issues=("incumbent_presence_mismatch_between_progress_and_telemetry",),
            progress_path=progress_path,
            event_count=len(rows),
            preparation_time=preparation,
            preparation_source=preparation_source,
            target_absolute_tolerance=target_absolute_tolerance,
            target_relative_tolerance=target_relative_tolerance,
        )

    status_time = trajectory.status_time
    if status_time is None:
        telemetry_total = _optional_finite(
            metrics.get("total_algorithm_runtime_seconds")
        )
        status_time = (
            telemetry_total
            if telemetry_total is not None and telemetry_total >= 0.0
            else None
        )
    status_within_horizon = status_time is not None and status_time <= horizon
    return _DerivedProgress(
        availability="available",
        issues=tuple(issues),
        progress_path=progress_path,
        event_count=len(rows),
        horizon=horizon,
        target_absolute_tolerance=target_absolute_tolerance,
        target_relative_tolerance=target_relative_tolerance,
        preparation_time=preparation,
        preparation_source=preparation_source,
        trajectory=trajectory,
        integral=integral,
        status_name=status,
        status_within_horizon=status_within_horizon,
        termination_label=honest_termination_label(
            status,
            has_incumbent=progress_has_incumbent,
        ),
        final_incumbent=final_incumbent,
        final_bound=final_bound,
    )


def _status_label(run: _ObservedRun | None) -> str:
    if run is None:
        return "missing_attempt"
    if not run.metrics_eligible:
        return run.metrics_exclusion_reason or "formal_attempt_not_metric_eligible"
    metrics = _solver_metrics(run)
    status = solver_status_name(metrics.get("status_name", metrics.get("status")))
    raw_has_incumbent = metrics.get("has_incumbent")
    return honest_termination_label(
        status,
        has_incumbent=(
            raw_has_incumbent if isinstance(raw_has_incumbent, bool) else False
        ),
    )


def _gap_value(run: _ObservedRun | None) -> float | None:
    metrics = _solver_metrics(run)
    if not bool(metrics.get("has_incumbent", False)):
        return None
    gap = _optional_finite(metrics.get("relative_gap"))
    if gap is None and str(metrics.get("status_name", "")).upper() == "OPTIMAL":
        return 0.0
    return max(0.0, gap) if gap is not None else None


def _proven_runtime_value(run: _ObservedRun | None) -> float | None:
    metrics = _solver_metrics(run)
    if str(metrics.get("status_name", "")).upper() != "OPTIMAL" or not bool(
        metrics.get("has_incumbent", False)
    ):
        return None
    runtime = _optional_finite(metrics.get("total_algorithm_runtime_seconds"))
    return runtime if runtime is not None and runtime >= 0.0 else None


def _metric_value(run: _ObservedRun | None, metric_name: str) -> float | None:
    if metric_name == "gap":
        return _gap_value(run)
    if metric_name == "runtime":
        return _proven_runtime_value(run)
    raise ValueError(f"unsupported telemetry metric: {metric_name}")


def _paired_common_target(
    baseline: _DerivedProgress,
    candidate: _DerivedProgress,
) -> tuple[CommonTargetTimes, str | None]:
    if not baseline.usable or baseline.trajectory is None:
        return (
            CommonTargetTimes(
                target=None,
                baseline_time=None,
                candidate_time=None,
                horizon=baseline.horizon,
                baseline_final_incumbent=baseline.final_incumbent,
                candidate_final_incumbent=candidate.final_incumbent,
            ),
            f"baseline_progress_{baseline.availability}",
        )
    if not candidate.usable or candidate.trajectory is None:
        return (
            CommonTargetTimes(
                target=None,
                baseline_time=None,
                candidate_time=None,
                horizon=baseline.horizon,
                baseline_final_incumbent=baseline.final_incumbent,
                candidate_final_incumbent=candidate.final_incumbent,
            ),
            f"candidate_progress_{candidate.availability}",
        )
    common = time_to_common_target(
        baseline.trajectory.points,
        candidate.trajectory.points,
        baseline.horizon,
        sense="min",
    )
    if common.target is not None:
        return (
            replace(
                common,
                baseline_time=baseline.time_to_target(common.target),
                candidate_time=candidate.time_to_target(common.target),
            ),
            None,
        )
    missing_sides = []
    if common.baseline_final_incumbent is None:
        missing_sides.append("baseline")
    if common.candidate_final_incumbent is None:
        missing_sides.append("candidate")
    return common, "missing_final_incumbent:" + ",".join(missing_sides)


def _progress_metric_value(
    progress: _DerivedProgress,
    common_target: float | None,
    metric_name: str,
) -> float | None:
    if not progress.usable:
        return None
    if metric_name == "time_to_target":
        return progress.time_to_target(common_target)
    if metric_name == "normalized_pdi":
        return progress.integral.average_gap if progress.integral is not None else None
    raise ValueError(f"unsupported progress metric: {metric_name}")


def _case_cluster_effect_estimate(
    observations: Sequence[PairedCaseObservation],
    statistic: Callable[[Sequence[tuple[float, float]]], float],
    *,
    confidence: float,
    resamples: int,
    seed: int,
    minimum_cases: int,
    minimum_cases_per_scale: int,
    expected_solver_seeds: Sequence[int] | None,
) -> tuple[ClusterBootstrapEffectEstimate | None, str | None]:
    case_count = len({row.case_id for row in observations})
    if case_count < minimum_cases:
        return None, (
            f"only {case_count} complete physical-case clusters; "
            f"at least {minimum_cases} are required"
        )
    try:
        effect = case_cluster_bootstrap_effect(
            observations,
            statistic,
            confidence=confidence,
            resamples=resamples,
            seed=seed,
            stratify_by_scale=True,
            minimum_cases=minimum_cases,
            minimum_cases_per_scale=minimum_cases_per_scale,
            expected_solver_seeds=expected_solver_seeds,
            require_balanced_solver_seeds=True,
        )
        effect.validate()
    except (ValueError, OverflowError) as error:
        return None, str(error)
    return effect, None


def _case_cluster_median_summary(
    observations: Sequence[PairedCaseObservation],
    *,
    minimum_cases: int,
    expected_solver_seeds: Sequence[int] | None,
) -> tuple[Mapping[str, Any] | None, str | None]:
    """Return descriptive medians with the physical case as the outer unit."""

    try:
        clusters = cluster_paired_observations(
            observations,
            minimum_cases=minimum_cases,
            expected_solver_seeds=expected_solver_seeds,
            require_balanced_solver_seeds=True,
        )
    except (TypeError, ValueError) as error:
        return None, str(error)

    per_case = tuple(
        {
            "case_id": cluster.case_id,
            "scale": cluster.scale,
            "solver_seed_count": len(cluster.solver_seeds),
            "baseline_within_case_seed_median": median(
                baseline for baseline, _candidate in cluster.pairs
            ),
            "candidate_within_case_seed_median": median(
                candidate for _baseline, candidate in cluster.pairs
            ),
            "candidate_minus_baseline_within_case_seed_median": median(
                candidate - baseline for baseline, candidate in cluster.pairs
            ),
        }
        for cluster in clusters
    )

    def aggregate(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, float | int]:
        return {
            "case_count": len(rows),
            "baseline_median_of_case_seed_medians": median(
                float(row["baseline_within_case_seed_median"]) for row in rows
            ),
            "candidate_median_of_case_seed_medians": median(
                float(row["candidate_within_case_seed_median"]) for row in rows
            ),
            "candidate_minus_baseline_median_of_case_seed_medians": median(
                float(row["candidate_minus_baseline_within_case_seed_median"])
                for row in rows
            ),
        }

    by_scale = {
        scale: aggregate(tuple(row for row in per_case if row["scale"] == scale))
        for scale in sorted({str(row["scale"]) for row in per_case})
    }
    censored_inputs = tuple(
        row.censored_time_input
        for row in observations
        if row.censored_time_input is not None
    )
    return (
        {
            "policy": FORMAL_MEDIAN_SUMMARY_POLICY,
            "sampling_unit": FORMAL_BOOTSTRAP_SAMPLING_UNIT,
            "pair_count": sum(cluster.pair_count for cluster in clusters),
            "case_count": len(clusters),
            "scale_case_counts": {
                scale: sum(cluster.scale == scale for cluster in clusters)
                for scale in sorted({cluster.scale for cluster in clusters})
            },
            "overall": aggregate(per_case),
            "by_scale": by_scale,
            "baseline_censored_count": sum(
                bool(item.baseline_censored) for item in censored_inputs
            ),
            "candidate_censored_count": sum(
                bool(item.candidate_censored) for item in censored_inputs
            ),
        },
        None,
    )


def _case_cluster_sign_test(
    observations: Sequence[PairedCaseObservation],
    *,
    minimum_cases: int,
    expected_solver_seeds: Sequence[int] | None,
    alternative: str,
    zero_tolerance: float,
) -> tuple[Mapping[str, Any] | None, str | None]:
    """Apply the exact sign test to one median contrast per physical case."""

    try:
        clusters = cluster_paired_observations(
            observations,
            minimum_cases=minimum_cases,
            expected_solver_seeds=expected_solver_seeds,
            require_balanced_solver_seeds=True,
        )
    except (TypeError, ValueError) as error:
        return None, str(error)
    case_differences = tuple(
        median(baseline - candidate for baseline, candidate in cluster.pairs)
        for cluster in clusters
    )
    result = exact_sign_test(
        case_differences,
        alternative=alternative,
        zero_tolerance=zero_tolerance,
    )
    return (
        {
            **asdict(result),
            "effective_pairs": result.effective_pairs,
            "effective_cases": result.effective_pairs,
            "case_count": len(clusters),
            "pair_count": sum(cluster.pair_count for cluster in clusters),
            "sampling_unit": FORMAL_SIGN_TEST_SAMPLING_UNIT,
            "case_contrast": "median_baseline_minus_candidate_over_solver_seeds",
        },
        None,
    )


def _runtime_effect_component(
    run: _ObservedRun | None,
    horizon: float,
    *,
    runtime_relative_tolerance: float,
) -> tuple[float | None, bool, str]:
    """Return observed proof time or an audited administrative censoring value."""

    if run is None:
        return None, False, "missing_attempt"
    if not run.metrics_eligible:
        return (
            None,
            False,
            run.metrics_exclusion_reason or "formal_attempt_not_metric_eligible",
        )
    metrics = _solver_metrics(run)
    status = solver_status_name(metrics.get("status_name", metrics.get("status")))
    runtime = _optional_finite(metrics.get("total_algorithm_runtime_seconds"))
    if runtime is None or runtime < 0.0:
        return None, False, "missing_or_invalid_total_algorithm_runtime"
    if status == "TIME_LIMIT":
        return horizon, True, "administratively_censored_time_limit"
    tolerance = max(1e-9, runtime_relative_tolerance * max(1.0, horizon))
    if runtime > horizon + tolerance:
        return None, False, "total_algorithm_runtime_exceeds_matched_horizon"
    if status == "OPTIMAL":
        if metrics.get("has_incumbent") is not True:
            return None, False, "optimal_status_without_incumbent"
        if runtime > horizon:
            return (
                horizon,
                False,
                "observed_optimal_total_runtime_clamped_within_tolerance",
            )
        return runtime, False, "observed_optimal_total_runtime"
    return None, False, f"unsupported_nonoptimal_status:{status}"


def _metric_denominator_audit(
    observations: Sequence[PairedCaseObservation],
    exclusions: Mapping[str, int],
    *,
    expected_pair_count: int,
    expected_case_count: int,
    expected_solver_seeds: Sequence[int] | None,
) -> Mapping[str, Any]:
    seeds_by_case: dict[str, set[int]] = {}
    scales_by_case: dict[str, str] = {}
    observed_keys: set[tuple[str, int]] = set()
    for row in observations:
        key = (row.case_id, row.solver_seed)
        if key in observed_keys:
            raise CLIError(f"duplicate metric observation for {key!r}")
        observed_keys.add(key)
        seeds_by_case.setdefault(row.case_id, set()).add(row.solver_seed)
        scales_by_case[row.case_id] = row.scale
    expected_seed_set = set(expected_solver_seeds or ())
    if expected_seed_set and any(
        not seeds <= expected_seed_set for seeds in seeds_by_case.values()
    ):
        raise CLIError("metric denominator contains an unregistered solver seed")
    excluded_pair_count = expected_pair_count - len(observations)
    if excluded_pair_count < 0:
        raise CLIError("metric denominator exceeds its expected pair population")
    if sum(exclusions.values()) != excluded_pair_count:
        raise CLIError(
            "metric denominator exclusions do not reconcile to the expected population"
        )
    complete_cases = (
        sum(seeds == expected_seed_set for seeds in seeds_by_case.values())
        if expected_seed_set
        else len(seeds_by_case)
    )
    return {
        "expected_pair_count": expected_pair_count,
        "included_pair_count": len(observations),
        "excluded_pair_count": excluded_pair_count,
        "expected_case_count": expected_case_count,
        "represented_case_count": len(seeds_by_case),
        "complete_case_count": complete_cases,
        "expected_solver_seeds": list(expected_solver_seeds or ()),
        "represented_cases_by_scale": dict(
            sorted(Counter(scales_by_case.values()).items())
        ),
        "exclusion_reasons": dict(sorted(exclusions.items())),
    }


def _validate_analysis_policy_namespace(args: argparse.Namespace) -> None:
    """Fail closed if the namespace names a policy this implementation lacks."""

    expected = {
        "aggregate_significance_policy": FORMAL_AGGREGATE_SIGNIFICANCE_POLICY,
        "bootstrap_method": FORMAL_BOOTSTRAP_METHOD,
        "bootstrap_sampling_unit": FORMAL_BOOTSTRAP_SAMPLING_UNIT,
        "bootstrap_stratification_policy": FORMAL_BOOTSTRAP_STRATIFICATION_POLICY,
        "median_summary_policy": FORMAL_MEDIAN_SUMMARY_POLICY,
        "runtime_effect_policy": FORMAL_RUNTIME_EFFECT_POLICY,
        "runtime_censoring_policy": FORMAL_RUNTIME_CENSORING_POLICY,
        "performance_profile_policy": FORMAL_PERFORMANCE_PROFILE_POLICY,
        "performance_profile_denominator_policy": (
            FORMAL_PERFORMANCE_PROFILE_DENOMINATOR_POLICY
        ),
        "sign_test_sampling_unit": FORMAL_SIGN_TEST_SAMPLING_UNIT,
        "systematic_regression_policy": FORMAL_SYSTEMATIC_REGRESSION_POLICY,
        "trajectory_policy_version": FORMAL_TRAJECTORY_POLICY_VERSION,
        "preparation_time_policy": FORMAL_PREPARATION_TIME_POLICY,
        "common_target_policy": FORMAL_COMMON_TARGET_POLICY,
        "missing_progress_policy": FORMAL_MISSING_PROGRESS_POLICY,
        "status_reconciliation_policy": FORMAL_STATUS_RECONCILIATION_POLICY,
        "horizon_policy": FORMAL_HORIZON_POLICY,
    }
    differing = [
        name for name, value in expected.items() if getattr(args, name) != value
    ]
    if differing:
        raise CLIError(
            "analysis namespace contains unsupported policy axes: "
            + ", ".join(differing)
        )
    if args.trajectory_sense != "min":
        raise CLIError("analysis trajectory_sense must be 'min'")


def _analyze_command(
    args: argparse.Namespace,
    hooks: CLIHooks,
    *,
    loaded_override: LoadedManifest | None = None,
    observed_override: Sequence[_ObservedRun] | None = None,
    formal_audit: _FormalAnalysisAudit | None = None,
    formal_analysis_contract: FormalAnalysisContract | None = None,
) -> Mapping[str, Any]:
    _validate_analysis_policy_namespace(args)
    loaded = loaded_override or load_benchmark_manifest(args.manifest)
    observed = (
        tuple(observed_override)
        if observed_override is not None
        else (
            _load_observed_runs(
                Path(args.results_root),
                manifest_id=loaded.manifest_id,
                attempt_number=args.attempt_number,
            )
        )
    )
    if not observed:
        raise CLIError(
            "no run records match this manifest and requested attempt number"
        )
    by_pair: dict[str, dict[str, _ObservedRun]] = {}
    pair_metadata: dict[str, Mapping[str, Any]] = {}
    for run in observed:
        by_pair.setdefault(run.pair_id, {})[run.profile_code] = run
        previous_metadata = pair_metadata.setdefault(run.pair_id, run.pair)
        if previous_metadata != run.pair:
            raise CLIError(
                f"profiles for pair {run.pair_id!r} have inconsistent pair specs"
            )
    scale_by_case = {
        frozen.candidate.case_id: frozen.candidate.scale
        for frozen in loaded.manifest.cases
    }
    baseline_code = ProfileCode(args.baseline_profile).value
    candidate_code = ProfileCode(args.candidate_profile).value
    if baseline_code == candidate_code:
        raise CLIError("baseline and candidate profiles must be different")
    primary_metric = PreregisteredMetric(
        name=args.primary_metric,
        direction=args.primary_direction,
        absolute_tolerance=args.absolute_tolerance,
        relative_tolerance=args.relative_tolerance,
        missing_is_failure=args.primary_missing_is_failure,
    )
    assessments: list[PairAssessment] = []
    metric_observations: dict[str, list[PairedCaseObservation]] = {
        "runtime": [],
        "gap": [],
        "time_to_target": [],
        "normalized_pdi": [],
    }
    metric_exclusions: dict[str, Counter[str]] = {
        name: Counter() for name in metric_observations
    }
    runtime_component_outcomes: dict[str, Counter[str]] = {
        "baseline": Counter(),
        "candidate": Counter(),
    }
    run_progress_metrics: list[Mapping[str, Any]] = []
    pair_progress_metrics: list[Mapping[str, Any]] = []
    key_assessments: dict[str, list[PairAssessment]] = {
        name: [] for name in args.key_regression_metrics
    }
    progress_kwargs = {
        "runtime_consistency_relative_tolerance": (
            args.runtime_consistency_relative_tolerance
        ),
        "missing_gap_penalty": args.missing_gap_penalty,
        "target_absolute_tolerance": args.target_absolute_tolerance,
        "target_relative_tolerance": args.target_relative_tolerance,
    }
    assessed_case_ids: set[str] = set()
    for pair_id in sorted(by_pair):
        profiles = by_pair[pair_id]
        baseline = profiles.get(baseline_code)
        candidate = profiles.get(candidate_code)
        if baseline is None and candidate is None:
            continue
        metadata = pair_metadata[pair_id]
        case_id = str(metadata.get("case_id", ""))
        if case_id not in scale_by_case:
            raise CLIError(f"run record references case outside manifest: {case_id!r}")
        assessed_case_ids.add(case_id)
        validated_pair = _pair_from_payload(metadata)
        seed = validated_pair.solver_seed
        instance_id = f"{case_id}:{metadata.get('mode', '')}"
        horizon = validated_pair.time_budget_seconds
        progress_by_profile = {
            code: _derive_progress(run, horizon, **progress_kwargs)
            for code, run in sorted(profiles.items())
        }
        baseline_progress = progress_by_profile.get(baseline_code) or _derive_progress(
            None,
            horizon,
            **progress_kwargs,
        )
        candidate_progress = progress_by_profile.get(
            candidate_code
        ) or _derive_progress(
            None,
            horizon,
            **progress_kwargs,
        )
        common_target, common_target_unavailable = _paired_common_target(
            baseline_progress,
            candidate_progress,
        )
        baseline_progress_payload = baseline_progress.audit_payload(
            pair_id=pair_id,
            profile_code=baseline_code,
            target=common_target.target,
            target_unavailable_reason=common_target_unavailable,
        )
        candidate_progress_payload = candidate_progress.audit_payload(
            pair_id=pair_id,
            profile_code=candidate_code,
            target=common_target.target,
            target_unavailable_reason=common_target_unavailable,
        )
        selected_progress_payloads = {
            baseline_code: baseline_progress_payload,
            candidate_code: candidate_progress_payload,
        }
        for profile_code in sorted(set(profiles) | {baseline_code, candidate_code}):
            progress = progress_by_profile.get(profile_code)
            if progress is None:
                progress = (
                    baseline_progress
                    if profile_code == baseline_code
                    else candidate_progress
                )
            payload = selected_progress_payloads.get(profile_code)
            if payload is None:
                payload = progress.audit_payload(
                    pair_id=pair_id,
                    profile_code=profile_code,
                    target=common_target.target,
                    target_unavailable_reason=common_target_unavailable,
                )
            observed_run = profiles.get(profile_code)
            raw_run_ids = (
                observed_run.record.get("ids") if observed_run is not None else None
            )
            run_ids = raw_run_ids if isinstance(raw_run_ids, Mapping) else {}
            run_progress_metrics.append(
                {
                    **payload,
                    "logical_run_id": run_ids.get("logical_run_id"),
                    "attempt_id": run_ids.get("attempt_id"),
                    "attempt_number": run_ids.get("attempt_number"),
                    "attempt_directory": (
                        observed_run.attempt_directory.as_posix()
                        if observed_run is not None
                        and observed_run.attempt_directory is not None
                        else None
                    ),
                    "common_target_reference_profiles": [
                        baseline_code,
                        candidate_code,
                    ],
                }
            )
        pair_progress_metrics.append(
            {
                "pair_id": pair_id,
                "case_id": case_id,
                "mode": str(metadata.get("mode", "")),
                "solver_seed": seed,
                "scale_tier": scale_by_case[case_id],
                "uniform_horizon_seconds": horizon,
                "baseline_profile": baseline_code,
                "candidate_profile": candidate_code,
                "common_target": common_target.target,
                "common_target_available": common_target.target_available,
                "common_target_unavailable_reason": common_target_unavailable,
                "common_target_rule": (
                    "maximum_of_paired_final_incumbents_for_minimization"
                ),
                "baseline_final_incumbent": (common_target.baseline_final_incumbent),
                "candidate_final_incumbent": (common_target.candidate_final_incumbent),
                "baseline_time_to_common_target_seconds": (common_target.baseline_time),
                "candidate_time_to_common_target_seconds": (
                    common_target.candidate_time
                ),
                "baseline_time_to_common_target_censored": (
                    common_target.baseline_time is None
                    if common_target.target is not None
                    else None
                ),
                "candidate_time_to_common_target_censored": (
                    common_target.candidate_time is None
                    if common_target.target is not None
                    else None
                ),
                "baseline_normalized_primal_dual_integral": (
                    baseline_progress.integral.average_gap
                    if baseline_progress.usable
                    and baseline_progress.integral is not None
                    else None
                ),
                "candidate_normalized_primal_dual_integral": (
                    candidate_progress.integral.average_gap
                    if candidate_progress.usable
                    and candidate_progress.integral is not None
                    else None
                ),
                "baseline_progress_availability": baseline_progress.availability,
                "candidate_progress_availability": candidate_progress.availability,
            }
        )
        if args.primary_metric in {"gap", "runtime"}:
            baseline_primary = _metric_value(baseline, args.primary_metric)
            candidate_primary = _metric_value(candidate, args.primary_metric)
        else:
            baseline_primary = _progress_metric_value(
                baseline_progress,
                common_target.target,
                args.primary_metric,
            )
            candidate_primary = _progress_metric_value(
                candidate_progress,
                common_target.target,
                args.primary_metric,
            )
        assessments.append(
            assess_pair(
                instance_id=instance_id,
                seed=seed,
                scale_tier=scale_by_case[case_id],
                baseline_value=baseline_primary,
                candidate_value=candidate_primary,
                metric=primary_metric,
                baseline_status=_status_label(baseline),
                candidate_status=_status_label(candidate),
            )
        )
        scale = scale_by_case[case_id]
        baseline_runtime, baseline_runtime_censored, baseline_runtime_outcome = (
            _runtime_effect_component(
                baseline,
                horizon,
                runtime_relative_tolerance=(
                    args.runtime_consistency_relative_tolerance
                ),
            )
        )
        candidate_runtime, candidate_runtime_censored, candidate_runtime_outcome = (
            _runtime_effect_component(
                candidate,
                horizon,
                runtime_relative_tolerance=(
                    args.runtime_consistency_relative_tolerance
                ),
            )
        )
        runtime_component_outcomes["baseline"][baseline_runtime_outcome] += 1
        runtime_component_outcomes["candidate"][candidate_runtime_outcome] += 1
        if baseline_runtime is not None and candidate_runtime is not None:
            runtime_time_input = paired_censored_time_input(
                None if baseline_runtime_censored else baseline_runtime,
                None if candidate_runtime_censored else candidate_runtime,
                horizon,
            )
            metric_observations["runtime"].append(
                PairedCaseObservation.from_censored_time_input(
                    case_id=case_id,
                    solver_seed=seed,
                    scale=scale,
                    censored_time_input=runtime_time_input,
                )
            )
        else:
            metric_exclusions["runtime"][
                f"baseline={baseline_runtime_outcome};"
                f"candidate={candidate_runtime_outcome}"
            ] += 1
        baseline_gap = _gap_value(baseline)
        candidate_gap = _gap_value(candidate)
        if baseline_gap is not None and candidate_gap is not None:
            metric_observations["gap"].append(
                PairedCaseObservation(
                    case_id=case_id,
                    solver_seed=seed,
                    scale=scale,
                    baseline=baseline_gap,
                    candidate=candidate_gap,
                )
            )
        else:
            metric_exclusions["gap"][
                f"baseline={_status_label(baseline)};"
                f"candidate={_status_label(candidate)}"
            ] += 1
        baseline_target_time = baseline_progress.time_to_target(common_target.target)
        candidate_target_time = candidate_progress.time_to_target(common_target.target)
        if common_target.target is not None:
            target_time_input = paired_censored_time_input(
                baseline_target_time,
                candidate_target_time,
                horizon,
            )
            metric_observations["time_to_target"].append(
                PairedCaseObservation.from_censored_time_input(
                    case_id=case_id,
                    solver_seed=seed,
                    scale=scale,
                    censored_time_input=target_time_input,
                )
            )
        else:
            metric_exclusions["time_to_target"][
                common_target_unavailable or "common_target_unavailable"
            ] += 1
        if (
            baseline_progress.usable
            and candidate_progress.usable
            and baseline_progress.integral is not None
            and candidate_progress.integral is not None
        ):
            metric_observations["normalized_pdi"].append(
                PairedCaseObservation(
                    case_id=case_id,
                    solver_seed=seed,
                    scale=scale,
                    baseline=baseline_progress.integral.average_gap,
                    candidate=candidate_progress.integral.average_gap,
                )
            )
        else:
            metric_exclusions["normalized_pdi"][
                f"baseline_progress={baseline_progress.availability};"
                f"candidate_progress={candidate_progress.availability}"
            ] += 1
        for metric_name, baseline_value, candidate_value in (
            (
                "node_count",
                _optional_finite(_solver_metrics(baseline).get("node_count")),
                _optional_finite(_solver_metrics(candidate).get("node_count")),
            ),
            (
                "time_to_first_incumbent",
                (
                    baseline_progress.integral.first_incumbent_time
                    if baseline_progress.usable
                    and baseline_progress.integral is not None
                    else None
                ),
                (
                    candidate_progress.integral.first_incumbent_time
                    if candidate_progress.usable
                    and candidate_progress.integral is not None
                    else None
                ),
            ),
        ):
            if metric_name not in key_assessments:
                continue
            key_assessments[metric_name].append(
                assess_pair(
                    instance_id=instance_id,
                    seed=seed,
                    scale_tier=scale_by_case[case_id],
                    baseline_value=baseline_value,
                    candidate_value=candidate_value,
                    metric=PreregisteredMetric(metric_name, direction="lower"),
                    baseline_status=_status_label(baseline),
                    candidate_status=_status_label(candidate),
                )
            )
    if not assessments:
        raise CLIError("no baseline/candidate pair could be formed")

    expected_bootstrap_solver_seeds = (
        tuple(args.bootstrap_expected_solver_seeds)
        if args.bootstrap_expected_solver_seeds is not None
        else None
    )
    speedup, speedup_unavailable = _case_cluster_effect_estimate(
        metric_observations["runtime"],
        lambda sample: shifted_geomean_speedup(sample, shift=args.runtime_shift),
        confidence=args.confidence,
        resamples=args.bootstrap_resamples,
        seed=args.bootstrap_seed,
        minimum_cases=args.minimum_effect_cases,
        minimum_cases_per_scale=args.bootstrap_minimum_cases_per_scale,
        expected_solver_seeds=expected_bootstrap_solver_seeds,
    )
    gap_reduction, gap_unavailable = _case_cluster_effect_estimate(
        metric_observations["gap"],
        lambda sample: shifted_geomean_gap_reduction(
            sample,
            shift=args.gap_shift,
        ),
        confidence=args.confidence,
        resamples=args.bootstrap_resamples,
        seed=args.bootstrap_seed,
        minimum_cases=args.minimum_effect_cases,
        minimum_cases_per_scale=args.bootstrap_minimum_cases_per_scale,
        expected_solver_seeds=expected_bootstrap_solver_seeds,
    )
    time_to_target_speedup, time_to_target_unavailable = _case_cluster_effect_estimate(
        metric_observations["time_to_target"],
        lambda sample: shifted_geomean_speedup(
            sample,
            shift=args.runtime_shift,
        ),
        confidence=args.confidence,
        resamples=args.bootstrap_resamples,
        seed=args.bootstrap_seed,
        minimum_cases=args.minimum_effect_cases,
        minimum_cases_per_scale=args.bootstrap_minimum_cases_per_scale,
        expected_solver_seeds=expected_bootstrap_solver_seeds,
    )
    normalized_pdi_reduction, normalized_pdi_unavailable = (
        _case_cluster_effect_estimate(
            metric_observations["normalized_pdi"],
            lambda sample: shifted_geomean_gap_reduction(
                sample,
                shift=args.gap_shift,
            ),
            confidence=args.confidence,
            resamples=args.bootstrap_resamples,
            seed=args.bootstrap_seed,
            minimum_cases=args.minimum_effect_cases,
            minimum_cases_per_scale=args.bootstrap_minimum_cases_per_scale,
            expected_solver_seeds=expected_bootstrap_solver_seeds,
        )
    )
    expected_metric_pair_count = (
        FORMAL_BLOCK_COUNT if formal_audit is not None else len(by_pair)
    )
    median_summaries: dict[str, Mapping[str, Any] | None] = {}
    median_unavailable_reasons: dict[str, str | None] = {}
    denominator_audits: dict[str, Mapping[str, Any]] = {}
    for metric_name, observations in metric_observations.items():
        summary, unavailable = _case_cluster_median_summary(
            observations,
            minimum_cases=args.minimum_effect_cases,
            expected_solver_seeds=expected_bootstrap_solver_seeds,
        )
        median_summaries[metric_name] = summary
        median_unavailable_reasons[metric_name] = unavailable
        denominator_audits[metric_name] = _metric_denominator_audit(
            observations,
            metric_exclusions[metric_name],
            expected_pair_count=expected_metric_pair_count,
            expected_case_count=len(scale_by_case),
            expected_solver_seeds=expected_bootstrap_solver_seeds,
        )
    denominator_audits["runtime"] = {
        **denominator_audits["runtime"],
        "effect_policy": FORMAL_RUNTIME_EFFECT_POLICY,
        "censoring_policy": FORMAL_RUNTIME_CENSORING_POLICY,
        "baseline_component_outcomes": dict(
            sorted(runtime_component_outcomes["baseline"].items())
        ),
        "candidate_component_outcomes": dict(
            sorted(runtime_component_outcomes["candidate"].items())
        ),
    }
    regressions = identify_systematic_regressions(
        key_assessments,
        regression_fraction=args.regression_fraction,
        minimum_pairs=args.regression_minimum_pairs,
    )
    primary_effect: EffectEstimate | ClusterBootstrapEffectEstimate | None
    primary_effect_unavailable: str | None
    primary_effect_scale: str
    if args.primary_metric == "runtime":
        primary_effect = speedup
        primary_effect_unavailable = speedup_unavailable
        primary_effect_scale = "baseline_over_candidate_shifted_geomean"
        primary_gate_speedup, primary_gate_reduction = speedup, None
    elif args.primary_metric == "gap":
        primary_effect = gap_reduction
        primary_effect_unavailable = gap_unavailable
        primary_effect_scale = "fractional_shifted_geomean_reduction"
        primary_gate_speedup, primary_gate_reduction = None, gap_reduction
    elif args.primary_metric == "time_to_target":
        primary_effect = time_to_target_speedup
        primary_effect_unavailable = time_to_target_unavailable
        primary_effect_scale = "baseline_over_candidate_shifted_geomean"
        primary_gate_speedup, primary_gate_reduction = time_to_target_speedup, None
    else:
        primary_effect = normalized_pdi_reduction
        primary_effect_unavailable = normalized_pdi_unavailable
        primary_effect_scale = "fractional_shifted_geomean_reduction"
        primary_gate_speedup, primary_gate_reduction = None, normalized_pdi_reduction
    if args.aggregate_gate_policy == FORMAL_AGGREGATE_GATE_POLICY:
        gate_speedup, gate_reduction = speedup, gap_reduction
    elif args.aggregate_gate_policy == "primary_metric_only_v1":
        gate_speedup, gate_reduction = (
            primary_gate_speedup,
            primary_gate_reduction,
        )
    else:
        raise CLIError("unsupported aggregate gate policy")
    gate = evaluate_success_gate(
        assessments,
        speedup=gate_speedup,
        gap_reduction=gate_reduction,
        systematic_regressions=regressions,
        medium_large_tiers=args.medium_large_tiers,
        minimum_medium_large_win_fraction=(args.minimum_medium_large_win_fraction),
        minimum_improved_scale_tiers=args.minimum_improved_scale_tiers,
        minimum_improved_instances=args.minimum_improved_instances,
        minimum_speedup=args.minimum_runtime_speedup,
        minimum_gap_reduction=args.minimum_gap_reduction,
    )
    expected_case_ids = set(scale_by_case)
    missing_case_ids = tuple(sorted(expected_case_ids - assessed_case_ids))
    coverage_complete = not missing_case_ids
    coverage_counts = {
        scale: sum(
            case_id in assessed_case_ids and case_scale == scale
            for case_id, case_scale in scale_by_case.items()
        )
        for scale in ("small", "medium", "large")
    }
    if not coverage_complete:
        gate = replace(
            gate,
            passed=False,
            conditions={**gate.conditions, "formal_manifest_coverage": False},
            reasons=(
                *gate.reasons,
                "formal manifest coverage is incomplete: "
                f"{len(assessed_case_ids)}/{len(expected_case_ids)} cases",
            ),
        )
    else:
        gate = replace(
            gate,
            conditions={**gate.conditions, "formal_manifest_coverage": True},
        )
    if formal_audit is not None:
        if formal_audit.all_slots_succeeded:
            gate = replace(
                gate,
                conditions={
                    **gate.conditions,
                    "formal_all_slots_succeeded": True,
                },
            )
        else:
            gate = replace(
                gate,
                passed=False,
                conditions={
                    **gate.conditions,
                    "formal_all_slots_succeeded": False,
                },
                reasons=(
                    *gate.reasons,
                    "formal campaign contains failed or incomplete attempts: "
                    f"{len(formal_audit.failed_slot_ids)}/{FORMAL_SLOT_COUNT} slots",
                ),
            )
    sign_test_zero_tolerance = (
        args.absolute_tolerance
        if args.sign_test_zero_tolerance is None
        else args.sign_test_zero_tolerance
    )
    sign_test, sign_test_unavailable = _case_cluster_sign_test(
        metric_observations[args.primary_metric],
        minimum_cases=args.minimum_effect_cases,
        expected_solver_seeds=expected_bootstrap_solver_seeds,
        alternative=args.sign_test_alternative,
        zero_tolerance=sign_test_zero_tolerance,
    )
    sign_test_alpha = (
        args.sign_test_alpha
        if args.sign_test_alpha is not None
        else 1.0 - args.confidence
    )

    performance_profile: Mapping[str, Any] | None = None
    performance_profile_unavailable: str | None = None
    performance: dict[str, dict[str, float | None]] = {}
    expected_profile_codes = {code.value for code in ProfileCode}
    for pair_id, profiles in sorted(by_pair.items()):
        if formal_audit is not None and set(profiles) != expected_profile_codes:
            raise CLIError(
                "formal performance-profile matrix lacks the exact A--E profile set "
                f"for pair {pair_id}"
            )
        performance[pair_id] = {
            baseline_code: _proven_runtime_value(profiles.get(baseline_code)),
            candidate_code: _proven_runtime_value(profiles.get(candidate_code)),
            **{code: _proven_runtime_value(run) for code, run in profiles.items()},
        }
    all_profile_codes = tuple(
        sorted({code for values in performance.values() for code in values})
    )
    all_failed_problem_ids = tuple(
        pair_id
        for pair_id, values in sorted(performance.items())
        if all(values.get(code) is None for code in all_profile_codes)
    )
    performance_profile_denominator_audit = {
        "performance_profile_policy": FORMAL_PERFORMANCE_PROFILE_POLICY,
        "denominator_policy": FORMAL_PERFORMANCE_PROFILE_DENOMINATOR_POLICY,
        "expected_problem_count": expected_metric_pair_count,
        "observed_problem_count": len(performance),
        "included_problem_count": len(performance) - len(all_failed_problem_ids),
        "excluded_all_failed_problem_count": len(all_failed_problem_ids),
        "excluded_all_failed_problem_ids": list(all_failed_problem_ids),
        "nonoptimal_or_missing_treated_as_infinite_ratio": True,
        "all_failed_problems_excluded_from_fraction_denominator": True,
        "proven_optimal_counts": {
            code: sum(values.get(code) is not None for values in performance.values())
            for code in all_profile_codes
        },
        "failure_counts": {
            code: sum(values.get(code) is None for values in performance.values())
            for code in all_profile_codes
        },
    }
    if formal_audit is not None and len(performance) != FORMAL_BLOCK_COUNT:
        raise CLIError(
            "formal performance-profile denominator is not the exact 180 problems"
        )
    try:
        ratios = dolan_more_ratios(performance)
        if (
            tuple(ratios.excluded_all_failed) != all_failed_problem_ids
            or len(ratios.included_problems)
            != performance_profile_denominator_audit["included_problem_count"]
        ):
            raise CLIError("performance-profile denominator audit does not reconcile")
        profile = dolan_more_profile_points(ratios, taus=args.performance_taus)
        performance_profile = {
            "metric": "proven_optimal_total_runtime_seconds",
            "policy": FORMAL_PERFORMANCE_PROFILE_POLICY,
            "denominator_policy": FORMAL_PERFORMANCE_PROFILE_DENOMINATOR_POLICY,
            "taus": list(profile.taus),
            "fractions": {
                name: list(values) for name, values in profile.fractions.items()
            },
            "included_problems": list(ratios.included_problems),
            "excluded_all_failed": list(ratios.excluded_all_failed),
            "failed_counts": dict(ratios.failed_counts),
        }
    except ValueError as error:
        performance_profile_unavailable = str(error)

    paper_success_gate = (
        gate
        if formal_audit is not None
        else {
            "passed": False,
            "reasons": [
                "exploratory analyze output is not authenticated formal evidence"
            ],
        }
    )
    artifact = {
        "artifact_type": (
            "mlnd_branch_cut_formal_paired_analysis"
            if formal_audit is not None
            else "mlnd_branch_cut_paired_analysis"
        ),
        "artifact_schema_version": 3,
        "created_at_utc": hooks.now_utc(),
        "evidence_scope": (
            "authenticated_formal_campaign"
            if formal_audit is not None
            else "exploratory"
        ),
        "paper_claim_eligible": bool(
            formal_audit is not None and formal_audit.all_slots_succeeded
        ),
        "paper_success_gate": paper_success_gate,
        "analysis_parameters_source": (
            "formal_plan.analysis_contract"
            if formal_analysis_contract is not None
            else "exploratory_cli"
        ),
        "formal_analysis_contract": (
            formal_analysis_contract.to_payload()
            if formal_analysis_contract is not None
            else None
        ),
        "formal_analysis_contract_sha256": (
            formal_analysis_contract.sha256
            if formal_analysis_contract is not None
            else None
        ),
        "formal_campaign_audit": (
            formal_audit.to_payload() if formal_audit is not None else None
        ),
        "expected_evidence_id": (
            formal_audit.expected_evidence_id if formal_audit is not None else None
        ),
        "expected_evidence_root_sha256": (
            formal_audit.expected_evidence_root_sha256
            if formal_audit is not None
            else None
        ),
        "actual_evidence_id": (
            formal_audit.actual_evidence_id if formal_audit is not None else None
        ),
        "actual_evidence_root_sha256": (
            formal_audit.actual_evidence_root_sha256
            if formal_audit is not None
            else None
        ),
        "manifest_id": loaded.manifest_id,
        "attempt_number": args.attempt_number,
        "baseline_profile": baseline_code,
        "candidate_profile": candidate_code,
        "primary_metric": primary_metric,
        "primary_effect": primary_effect,
        "primary_effect_scale": primary_effect_scale,
        "primary_effect_unavailable_reason": primary_effect_unavailable,
        "pair_assessments": assessments,
        "run_progress_metrics": run_progress_metrics,
        "pair_progress_metrics": pair_progress_metrics,
        "winner_counts": winner_counts(assessments),
        "primary_sign_test": sign_test,
        "primary_sign_test_unavailable_reason": sign_test_unavailable,
        "primary_sign_test_sampling_unit": args.sign_test_sampling_unit,
        "primary_sign_test_zero_tolerance": sign_test_zero_tolerance,
        "primary_sign_test_alpha": sign_test_alpha,
        "primary_sign_test_rejects_null": (
            sign_test is not None and float(sign_test["p_value"]) < sign_test_alpha
        ),
        "aggregate_gate_policy": args.aggregate_gate_policy,
        "bootstrap_method": args.bootstrap_method,
        "bootstrap_sampling_unit": args.bootstrap_sampling_unit,
        "bootstrap_stratification_policy": args.bootstrap_stratification_policy,
        "bootstrap_expected_solver_seeds": list(expected_bootstrap_solver_seeds or ()),
        "median_summary_policy": args.median_summary_policy,
        "runtime_effect_policy": args.runtime_effect_policy,
        "runtime_censoring_policy": args.runtime_censoring_policy,
        "performance_profile_policy": args.performance_profile_policy,
        "performance_profile_denominator_policy": (
            args.performance_profile_denominator_policy
        ),
        "speedup": speedup,
        "speedup_unavailable_reason": speedup_unavailable,
        "gap_reduction": gap_reduction,
        "gap_reduction_unavailable_reason": gap_unavailable,
        "time_to_target_speedup": time_to_target_speedup,
        "time_to_target_speedup_unavailable_reason": time_to_target_unavailable,
        "normalized_pdi_reduction": normalized_pdi_reduction,
        "normalized_pdi_reduction_unavailable_reason": (normalized_pdi_unavailable),
        "median_summaries": median_summaries,
        "median_summary_unavailable_reasons": median_unavailable_reasons,
        "effect_denominator_audits": denominator_audits,
        "systematic_regressions": list(regressions),
        "success_gate_primary_metric": args.primary_metric,
        "success_gate": gate,
        "complete_proven_runtime_pair_count": sum(
            _proven_runtime_value(profiles.get(baseline_code)) is not None
            and _proven_runtime_value(profiles.get(candidate_code)) is not None
            for profiles in by_pair.values()
        ),
        "runtime_effect_input_pair_count": len(metric_observations["runtime"]),
        "complete_gap_pair_count": len(metric_observations["gap"]),
        "complete_time_to_target_pair_count": sum(
            row.censored_time_input is not None
            and not row.censored_time_input.baseline_censored
            and not row.censored_time_input.candidate_censored
            for row in metric_observations["time_to_target"]
        ),
        "time_to_target_effect_input_pair_count": len(
            metric_observations["time_to_target"]
        ),
        "complete_normalized_pdi_pair_count": len(
            metric_observations["normalized_pdi"]
        ),
        "performance_profile": performance_profile,
        "performance_profile_unavailable_reason": performance_profile_unavailable,
        "performance_profile_denominator_audit": (
            performance_profile_denominator_audit
        ),
        "formal_coverage": {
            "complete": coverage_complete,
            "expected_case_count": len(expected_case_ids),
            "observed_case_count": len(assessed_case_ids),
            "observed_cases_by_scale": coverage_counts,
            "missing_case_ids": list(missing_case_ids),
        },
    }
    output = write_canonical_json_exclusive(args.output, artifact)
    return {
        "command": "analyze-formal" if formal_audit is not None else "analyze",
        "output": output,
        "manifest_id": loaded.manifest_id,
        "pair_count": len(assessments),
        "winner_counts": winner_counts(assessments),
        "success_gate_passed": gate.passed,
        "success_gate_reasons": list(gate.reasons),
        "evidence_scope": artifact["evidence_scope"],
        "paper_claim_eligible": artifact["paper_claim_eligible"],
        "paper_success_gate_passed": bool(formal_audit is not None and gate.passed),
        "expected_evidence_id": (
            formal_audit.expected_evidence_id if formal_audit is not None else None
        ),
        "expected_evidence_root_sha256": (
            formal_audit.expected_evidence_root_sha256
            if formal_audit is not None
            else None
        ),
        "actual_evidence_id": (
            formal_audit.actual_evidence_id if formal_audit is not None else None
        ),
        "actual_evidence_root_sha256": (
            formal_audit.actual_evidence_root_sha256
            if formal_audit is not None
            else None
        ),
    }


def _analyze_formal_command_locked(
    args: argparse.Namespace,
    hooks: CLIHooks,
) -> Mapping[str, Any]:
    campaign: FormalCampaign | None = None
    try:
        loaded, observed, audit, plan, campaign = _load_authenticated_formal_runs(
            args, hooks
        )
        contract = plan.analysis_contract
        formal_args = _formal_analysis_namespace(args, contract)
        result = _analyze_command(
            formal_args,
            hooks,
            loaded_override=loaded,
            observed_override=observed,
            formal_audit=audit,
            formal_analysis_contract=contract,
        )
        finish_formal_session(
            campaign,
            status="completed",
            now_utc=hooks.now_utc,
        )
        return result
    except BaseException as error:
        if campaign is not None:
            try:
                finish_formal_session(
                    campaign,
                    status="failed",
                    now_utc=hooks.now_utc,
                )
            except Exception as session_error:
                raise CLIError(
                    "formal analysis failed and its session end could not be "
                    f"published: {type(session_error).__name__}: {session_error}"
                ) from error
        if isinstance(error, FormalExecutionError):
            raise CLIError(str(error)) from error
        raise


def _formal_analysis_namespace(
    args: argparse.Namespace,
    contract: FormalAnalysisContract,
) -> argparse.Namespace:
    """Derive every claim-affecting analyzer input from the hashed plan."""

    if not isinstance(contract, FormalAnalysisContract):
        raise CLIError("formal plan lacks a valid analysis preregistration")
    values = vars(args).copy()
    values.update(
        attempt_number=FORMAL_ATTEMPT_NUMBER,
        baseline_profile=contract.baseline_profile.value,
        candidate_profile=contract.candidate_profile.value,
        primary_metric=contract.primary_metric,
        primary_direction=contract.primary_direction,
        absolute_tolerance=contract.primary_absolute_tolerance,
        relative_tolerance=contract.primary_relative_tolerance,
        primary_missing_is_failure=contract.primary_missing_is_failure,
        aggregate_gate_policy=contract.aggregate_gate_policy,
        aggregate_significance_policy=contract.aggregate_significance_policy,
        medium_large_tiers=contract.medium_large_tiers,
        minimum_medium_large_win_fraction=(contract.minimum_medium_large_win_fraction),
        minimum_improved_scale_tiers=contract.minimum_improved_scale_tiers,
        minimum_improved_instances=contract.minimum_improved_instances,
        minimum_runtime_speedup=contract.minimum_runtime_speedup,
        minimum_gap_reduction=contract.minimum_gap_reduction,
        confidence=contract.confidence_level,
        bootstrap_method=contract.bootstrap_method,
        bootstrap_sampling_unit=contract.bootstrap_sampling_unit,
        bootstrap_stratification_policy=contract.bootstrap_stratification_policy,
        bootstrap_expected_solver_seeds=(contract.bootstrap_expected_solver_seeds),
        bootstrap_minimum_cases_per_scale=(contract.bootstrap_minimum_cases_per_scale),
        bootstrap_resamples=contract.bootstrap_resamples,
        bootstrap_seed=contract.bootstrap_seed,
        minimum_effect_cases=contract.minimum_effect_cases,
        runtime_shift=contract.runtime_shift,
        gap_shift=contract.gap_shift,
        median_summary_policy=contract.median_summary_policy,
        runtime_effect_policy=contract.runtime_effect_policy,
        runtime_censoring_policy=contract.runtime_censoring_policy,
        performance_profile_policy=contract.performance_profile_policy,
        performance_profile_denominator_policy=(
            contract.performance_profile_denominator_policy
        ),
        key_regression_metrics=contract.key_regression_metrics,
        systematic_regression_policy=contract.systematic_regression_policy,
        regression_fraction=contract.regression_fraction,
        regression_minimum_pairs=contract.regression_minimum_pairs,
        sign_test_alternative=contract.sign_test_alternative,
        sign_test_sampling_unit=contract.sign_test_sampling_unit,
        sign_test_zero_tolerance=contract.sign_test_zero_tolerance,
        sign_test_alpha=contract.sign_test_alpha,
        performance_taus=contract.performance_taus,
        trajectory_policy_version=contract.trajectory_policy_version,
        trajectory_sense=contract.trajectory_sense,
        preparation_time_policy=contract.preparation_time_policy,
        runtime_consistency_relative_tolerance=(
            contract.runtime_consistency_relative_tolerance
        ),
        missing_gap_penalty=contract.missing_gap_penalty,
        common_target_policy=contract.common_target_policy,
        target_absolute_tolerance=contract.target_absolute_tolerance,
        target_relative_tolerance=contract.target_relative_tolerance,
        missing_progress_policy=contract.missing_progress_policy,
        status_reconciliation_policy=contract.status_reconciliation_policy,
        horizon_policy=contract.horizon_policy,
        require_complete_manifest_coverage=(
            contract.require_complete_manifest_coverage
        ),
        require_all_formal_slots_succeeded=(
            contract.require_all_formal_slots_succeeded
        ),
    )
    return argparse.Namespace(**values)


def _analyze_formal_command(
    args: argparse.Namespace,
    hooks: CLIHooks,
) -> Mapping[str, Any]:
    try:
        with BenchmarkResourceLock(args.resource_lock, command="analyze-formal"):
            return _analyze_formal_command_locked(args, hooks)
    except BenchmarkResourceLockError as error:
        raise CLIError(str(error)) from error


def _add_formal_suite_arguments(command: argparse.ArgumentParser) -> None:
    """Expose settings that are frozen once, never accepted by ``run-formal``."""

    command.add_argument("--target-mip-gap", type=_fraction, default=0.0)
    command.add_argument("--method", type=int, choices=range(-1, 6), default=-1)
    command.add_argument(
        "--miqcp-method",
        type=int,
        choices=(-1, 0, 1),
        default=-1,
        help="matched Gurobi MIQCPMethod shared by profiles A--E",
    )
    command.add_argument("--mip-focus", type=int, choices=range(0, 4), default=0)
    command.add_argument("--heuristics", type=_fraction, default=0.05)
    command.add_argument("--no-rel-heur-time", type=float, default=0.0)
    command.add_argument("--presolve", type=int, choices=range(-1, 3), default=-1)
    command.add_argument(
        "--native-cuts",
        type=int,
        choices=range(-1, 4),
        default=-1,
    )
    command.add_argument(
        "--root-cut-passes",
        type=_root_cut_passes,
        default=2,
        help="matched Gurobi CutPasses setting shared by profiles A--E",
    )
    command.add_argument("--numeric-focus", type=int, choices=range(0, 4), default=0)
    command.add_argument("--solver-output", action="store_true")
    command.add_argument(
        "--network-start-seconds",
        type=_positive_float,
        default=20.0,
    )
    command.add_argument(
        "--network-start-method",
        type=int,
        choices=range(-1, 6),
        default=2,
        help=(
            "Gurobi Method used only for network-start and fixed-integer "
            "repair solves in profiles D/E"
        ),
    )
    command.add_argument(
        "--fixed-integer-repair-seconds",
        type=_nonnegative_float,
        default=10.0,
        help=(
            "fixed-integer repair budget for heuristic profiles D/E; set to 0 "
            "before enabling local branching"
        ),
    )
    command.add_argument(
        "--local-branching-seconds",
        type=_nonnegative_float,
        default=0.0,
    )
    command.add_argument("--local-branching-radius", type=_positive_int, default=12)
    command.add_argument(
        "--local-branching-include-topology",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    command.add_argument(
        "--progress-interval-seconds",
        type=_positive_float,
        default=1.0,
    )


def _add_formal_analysis_arguments(command: argparse.ArgumentParser) -> None:
    """Expose claim axes only while the immutable formal plan is frozen."""

    command.add_argument(
        "--analysis-primary-metric",
        choices=("gap", "runtime", "time_to_target", "normalized_pdi"),
        default="gap",
    )
    command.add_argument(
        "--analysis-absolute-tolerance",
        type=_nonnegative_float,
        default=0.0,
    )
    command.add_argument(
        "--analysis-relative-tolerance",
        type=_nonnegative_float,
        default=0.0,
    )
    command.add_argument(
        "--analysis-minimum-medium-large-win-fraction",
        type=_fraction,
        default=0.60,
    )
    command.add_argument(
        "--analysis-minimum-improved-scale-tiers",
        type=_positive_int,
        default=2,
    )
    command.add_argument(
        "--analysis-minimum-improved-instances",
        type=_positive_int,
        default=2,
    )
    command.add_argument(
        "--analysis-minimum-runtime-speedup",
        type=_positive_float,
        default=1.50,
    )
    command.add_argument(
        "--analysis-minimum-gap-reduction",
        type=_fraction,
        default=0.20,
    )
    command.add_argument("--analysis-confidence", type=_fraction, default=0.95)
    command.add_argument(
        "--analysis-bootstrap-resamples",
        type=_positive_int,
        default=10_000,
    )
    command.add_argument(
        "--analysis-bootstrap-seed", type=_nonnegative_int, default=20260826
    )
    command.add_argument(
        "--analysis-minimum-effect-cases",
        type=_positive_int,
        default=5,
    )
    command.add_argument(
        "--analysis-runtime-shift",
        type=_positive_float,
        default=1.0,
    )
    command.add_argument(
        "--analysis-gap-shift",
        type=_positive_float,
        default=1e-6,
    )
    command.add_argument(
        "--analysis-key-regression-metrics",
        nargs="+",
        choices=("node_count", "time_to_first_incumbent"),
        default=("node_count", "time_to_first_incumbent"),
    )
    command.add_argument(
        "--analysis-regression-fraction",
        type=_fraction,
        default=0.60,
    )
    command.add_argument(
        "--analysis-regression-minimum-pairs",
        type=_positive_int,
        default=5,
    )
    command.add_argument(
        "--analysis-sign-test-alternative",
        choices=("greater", "less", "two-sided"),
        default="greater",
    )
    command.add_argument(
        "--analysis-sign-test-zero-tolerance",
        type=_nonnegative_float,
        default=0.0,
    )
    command.add_argument(
        "--analysis-performance-taus",
        nargs="+",
        type=_positive_float,
        default=(1.0, 1.25, 1.5, 2.0, 4.0, 8.0),
    )
    command.add_argument(
        "--analysis-runtime-consistency-relative-tolerance",
        type=_nonnegative_float,
        default=1e-8,
    )
    command.add_argument(
        "--analysis-missing-gap-penalty",
        type=_nonnegative_float,
        default=1.0,
    )
    command.add_argument(
        "--analysis-target-absolute-tolerance",
        type=_nonnegative_float,
        default=1e-9,
    )
    command.add_argument(
        "--analysis-target-relative-tolerance",
        type=_nonnegative_float,
        default=1e-9,
    )
    command.add_argument("--no-solver-log", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mlnd-bnc-benchmark",
        description=(
            "Build, qualify, freeze, execute, and analyse the locked MLND "
            "branch-and-cut benchmark."
        ),
    )
    commands = parser.add_subparsers(dest="command", required=True)

    build_pool = commands.add_parser(
        "build-pool",
        help="generate the v5 overcomplete qualification pool",
    )
    build_pool.add_argument("--output-root", required=True)
    build_pool.add_argument("--nanshan-raw", required=True)
    build_pool.add_argument("--tsl-raw", required=True)
    build_pool.add_argument(
        "--candidate-hours",
        nargs=2,
        type=int,
        default=(8, 24),
        metavar=("START", "END"),
    )
    build_pool.add_argument("--site-grid-km", type=_positive_float, default=0.25)
    build_pool.add_argument(
        "--pool-label",
        default="mlnd-real-data-overcomplete-qualification-v5",
    )
    build_pool.set_defaults(handler=_build_pool_command)

    qualify = commands.add_parser(
        "qualify",
        help="find and fresh-model validate non-trivial positive-service witnesses",
    )
    qualify.add_argument("--pool", required=True)
    qualify.add_argument("--instances-root", required=True)
    qualify.add_argument("--output", required=True)
    qualify.add_argument(
        "--mode", choices=tuple(mode.value for mode in BenchmarkMode), default="MX"
    )
    qualify.add_argument("--qualification-run-id")
    qualify.add_argument("--time-limit", type=_positive_float, default=30.0)
    qualify.add_argument("--mip-gap", type=_fraction, default=0.01)
    qualify.add_argument("--solver-seed", type=int, default=42)
    qualify.add_argument("--solver-output", action="store_true")
    qualify.add_argument(
        "--jobs",
        type=_positive_int,
        choices=(1,),
        default=1,
        help="qualification is serialized on the 24 GiB benchmark host",
    )
    qualify.add_argument(
        "--resource-lock",
        default=str(DEFAULT_BENCHMARK_RESOURCE_LOCK),
        help=(
            "shared non-blocking advisory lock used by qualification and formal "
            "benchmark batches"
        ),
    )
    qualify.add_argument("--method", type=int, choices=range(-1, 6), default=-1)
    qualify.add_argument(
        "--miqcp-method",
        type=int,
        choices=range(-1, 2),
        default=-1,
    )
    qualify.add_argument("--mip-focus", type=int, choices=range(0, 4), default=0)
    qualify.add_argument("--heuristics", type=_fraction, default=0.05)
    qualify.add_argument(
        "--no-rel-heur-time",
        type=_nonnegative_float,
        default=0.0,
    )
    qualify.add_argument("--presolve", type=int, choices=range(-1, 3), default=-1)
    qualify.add_argument(
        "--native-cuts",
        type=int,
        choices=range(-1, 4),
        default=-1,
    )
    qualify.add_argument(
        "--root-cut-passes",
        type=_root_cut_passes,
        default=2,
        help="Gurobi CutPasses setting frozen for qualification solves",
    )
    qualify.add_argument(
        "--numeric-focus",
        type=int,
        choices=range(0, 4),
        default=0,
    )
    qualify.add_argument(
        "--feasibility-tolerance",
        type=_positive_float,
        default=1e-6,
    )
    qualify.add_argument(
        "--optimality-tolerance",
        type=_positive_float,
        default=1e-6,
    )
    qualify.add_argument(
        "--integer-feasibility-tolerance",
        type=_positive_float,
        default=1e-5,
    )
    qualify.add_argument(
        "--qualification-optimality-tolerance",
        type=float,
        default=1e-6,
    )
    qualify.add_argument("--minimum-savings-fraction", type=_fraction, default=1e-3)
    qualify.add_argument("--minimum-service-fraction", type=_fraction, default=1e-3)
    qualify.add_argument("--minimum-h-open-fraction", type=_fraction, default=0.15)
    qualify.add_argument("--maximum-h-open-fraction", type=_fraction, default=0.85)
    qualify.add_argument(
        "--geometry-start",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "enable the qualification-only, cost-blind partial geometry-start "
            "fallback (disabled by default until the development pilot is locked)"
        ),
    )
    qualify.add_argument(
        "--geometry-start-seconds",
        type=_positive_float,
        default=10.0,
        help="time budget for each pre-registered geometry support attempt",
    )
    qualify.add_argument(
        "--geometry-start-node-limit",
        type=_nonnegative_int,
        default=500,
        help="Gurobi StartNodeLimit used only for each geometry-start attempt",
    )
    qualify.add_argument(
        "--large-scale-open-h-target",
        nargs=2,
        type=_positive_int,
        metavar=("MIN", "MAX"),
    )
    qualify.add_argument(
        "--journal-dir",
        help=(
            "qualification checkpoint directory; defaults to "
            "<output filename>.journal beside --output"
        ),
    )
    qualify.set_defaults(handler=_qualify_command)

    freeze = commands.add_parser(
        "freeze",
        help="freeze accepted qualifications before comparison starts",
    )
    freeze.add_argument("--pool", required=True)
    freeze.add_argument("--qualification", required=True)
    freeze.add_argument("--output", required=True)
    freeze.add_argument("--qualification-completed-at")
    freeze.add_argument("--manifest-frozen-at")
    freeze.set_defaults(handler=_freeze_command)

    freeze_formal = commands.add_parser(
        "freeze-formal-plan",
        help="freeze the exact 36-case, five-seed, A--E formal execution plan",
    )
    freeze_formal.add_argument("--manifest", required=True)
    freeze_formal.add_argument("--output", required=True)
    _add_formal_suite_arguments(freeze_formal)
    _add_formal_analysis_arguments(freeze_formal)
    freeze_formal.set_defaults(handler=_freeze_formal_plan_command)

    run = commands.add_parser(
        "run",
        help="schedule matched single-thread profile runs",
    )
    run.add_argument("--manifest", required=True)
    run.add_argument("--instances-root", required=True)
    run.add_argument("--results-root", required=True)
    run.add_argument("--batch-output", required=True)
    run.add_argument(
        "--modes",
        nargs="+",
        choices=tuple(mode.value for mode in BenchmarkMode),
    )
    run.add_argument("--allow-unqualified-modes", action="store_true")
    run.add_argument("--case-ids", nargs="+")
    run.add_argument("--solver-seeds", nargs="+", type=int, default=(42,))
    run.add_argument(
        "--time-budgets",
        nargs="+",
        type=_positive_float,
        default=(60.0,),
    )
    run.add_argument(
        "--profiles",
        nargs="+",
        choices=tuple(code.value for code in ProfileCode),
        default=tuple(code.value for code in ProfileCode),
    )
    run.add_argument("--jobs", type=_positive_int, default=1)
    run.add_argument(
        "--resource-lock",
        default=str(DEFAULT_BENCHMARK_RESOURCE_LOCK),
        help=(
            "shared non-blocking advisory lock used by qualification and formal "
            "benchmark batches"
        ),
    )
    run.add_argument("--attempt-number", type=_positive_int)
    run.add_argument("--target-mip-gap", type=_fraction, default=0.0)
    run.add_argument("--method", type=int, choices=range(-1, 6), default=-1)
    run.add_argument(
        "--miqcp-method",
        type=int,
        choices=(-1, 0, 1),
        default=-1,
        help="matched Gurobi MIQCPMethod shared by profiles A--E",
    )
    run.add_argument("--mip-focus", type=int, choices=range(0, 4), default=0)
    run.add_argument("--heuristics", type=_fraction, default=0.05)
    run.add_argument("--no-rel-heur-time", type=float, default=0.0)
    run.add_argument("--presolve", type=int, choices=range(-1, 3), default=-1)
    run.add_argument("--native-cuts", type=int, choices=range(-1, 4), default=-1)
    run.add_argument(
        "--root-cut-passes",
        type=_root_cut_passes,
        default=2,
    )
    run.add_argument("--numeric-focus", type=int, choices=range(0, 4), default=0)
    run.add_argument("--solver-output", action="store_true")
    run.add_argument("--network-start-seconds", type=_positive_float, default=20.0)
    run.add_argument(
        "--network-start-method",
        type=int,
        choices=range(-1, 6),
        default=2,
        help=(
            "Gurobi Method used only for network-start and fixed-integer "
            "repair solves in profiles D/E"
        ),
    )
    run.add_argument(
        "--fixed-integer-repair-seconds",
        type=_nonnegative_float,
        default=10.0,
        help=(
            "fixed-integer repair budget for heuristic profiles D/E; set to 0 "
            "before enabling local branching"
        ),
    )
    run.add_argument(
        "--local-branching-seconds",
        type=_nonnegative_float,
        default=0.0,
    )
    run.add_argument("--local-branching-radius", type=_positive_int, default=12)
    run.add_argument(
        "--local-branching-include-topology",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    run.add_argument("--progress-interval-seconds", type=_positive_float, default=1.0)
    run.add_argument("--no-solver-log", action="store_true")
    run.set_defaults(handler=_run_command)

    run_formal = commands.add_parser(
        "run-formal",
        help="resume the externally anchored formal plan without axis overrides",
    )
    run_formal.add_argument("--manifest", required=True)
    run_formal.add_argument("--plan", required=True)
    run_formal.add_argument(
        "--expected-plan-id",
        required=True,
        help="plan ID copied from the independent freeze-formal-plan output",
    )
    run_formal.add_argument("--instances-root", required=True)
    run_formal.add_argument("--results-root", required=True)
    run_formal.add_argument(
        "--resource-lock",
        default=str(DEFAULT_BENCHMARK_RESOURCE_LOCK),
        help="shared non-blocking advisory lock for formal benchmark execution",
    )
    run_formal.set_defaults(handler=_run_formal_command)

    analyze = commands.add_parser(
        "analyze",
        help="produce paired estimates, confidence intervals, and the success gate",
    )
    analyze.add_argument("--manifest", required=True)
    analyze.add_argument("--results-root", required=True)
    analyze.add_argument("--output", required=True)
    analyze.add_argument("--attempt-number", type=_positive_int, default=1)
    analyze.add_argument(
        "--baseline-profile",
        choices=tuple(code.value for code in ProfileCode),
        default="A",
    )
    analyze.add_argument(
        "--candidate-profile",
        choices=tuple(code.value for code in ProfileCode),
        default="E",
    )
    analyze.add_argument(
        "--primary-metric",
        choices=("gap", "runtime", "time_to_target", "normalized_pdi"),
        default="gap",
    )
    analyze.add_argument("--absolute-tolerance", type=float, default=0.0)
    analyze.add_argument("--relative-tolerance", type=float, default=0.0)
    analyze.add_argument("--confidence", type=float, default=0.95)
    analyze.add_argument("--bootstrap-resamples", type=_positive_int, default=10_000)
    analyze.add_argument("--bootstrap-seed", type=int, default=20260826)
    analyze.add_argument("--minimum-effect-cases", type=_positive_int, default=5)
    analyze.add_argument("--runtime-shift", type=_positive_float, default=1.0)
    analyze.add_argument("--gap-shift", type=_positive_float, default=1e-6)
    analyze.add_argument("--regression-fraction", type=_fraction, default=0.60)
    analyze.add_argument("--regression-minimum-pairs", type=_positive_int, default=5)
    analyze.add_argument(
        "--performance-taus",
        nargs="+",
        type=_positive_float,
        default=(1.0, 1.25, 1.5, 2.0, 4.0, 8.0),
    )
    analyze.add_argument(
        "--aggregate-gate-policy",
        choices=("primary_metric_only_v1", FORMAL_AGGREGATE_GATE_POLICY),
        default="primary_metric_only_v1",
    )
    analyze.add_argument(
        "--medium-large-tiers",
        nargs="+",
        choices=("small", "medium", "large"),
        default=("medium", "large"),
    )
    analyze.add_argument(
        "--minimum-medium-large-win-fraction",
        type=_fraction,
        default=0.60,
    )
    analyze.add_argument(
        "--minimum-improved-scale-tiers",
        type=_positive_int,
        default=2,
    )
    analyze.add_argument(
        "--minimum-improved-instances",
        type=_positive_int,
        default=2,
    )
    analyze.add_argument(
        "--minimum-runtime-speedup",
        type=_positive_float,
        default=1.50,
    )
    analyze.add_argument(
        "--minimum-gap-reduction",
        type=_fraction,
        default=0.20,
    )
    analyze.add_argument(
        "--key-regression-metrics",
        nargs="+",
        choices=("node_count", "time_to_first_incumbent"),
        default=("node_count", "time_to_first_incumbent"),
    )
    analyze.add_argument(
        "--sign-test-alternative",
        choices=("greater", "less", "two-sided"),
        default="greater",
    )
    analyze.add_argument(
        "--sign-test-zero-tolerance",
        type=_nonnegative_float,
    )
    analyze.add_argument(
        "--runtime-consistency-relative-tolerance",
        type=_nonnegative_float,
        default=1e-8,
    )
    analyze.add_argument(
        "--missing-gap-penalty",
        type=_nonnegative_float,
        default=1.0,
    )
    analyze.add_argument(
        "--target-absolute-tolerance",
        type=_nonnegative_float,
        default=1e-9,
    )
    analyze.add_argument(
        "--target-relative-tolerance",
        type=_nonnegative_float,
        default=1e-9,
    )
    analyze.set_defaults(
        primary_direction="lower",
        primary_missing_is_failure=True,
        aggregate_significance_policy=FORMAL_AGGREGATE_SIGNIFICANCE_POLICY,
        bootstrap_method=FORMAL_BOOTSTRAP_METHOD,
        bootstrap_sampling_unit=FORMAL_BOOTSTRAP_SAMPLING_UNIT,
        bootstrap_stratification_policy=FORMAL_BOOTSTRAP_STRATIFICATION_POLICY,
        bootstrap_expected_solver_seeds=None,
        bootstrap_minimum_cases_per_scale=2,
        median_summary_policy=FORMAL_MEDIAN_SUMMARY_POLICY,
        runtime_effect_policy=FORMAL_RUNTIME_EFFECT_POLICY,
        runtime_censoring_policy=FORMAL_RUNTIME_CENSORING_POLICY,
        performance_profile_policy=FORMAL_PERFORMANCE_PROFILE_POLICY,
        performance_profile_denominator_policy=(
            FORMAL_PERFORMANCE_PROFILE_DENOMINATOR_POLICY
        ),
        systematic_regression_policy=FORMAL_SYSTEMATIC_REGRESSION_POLICY,
        sign_test_sampling_unit=FORMAL_SIGN_TEST_SAMPLING_UNIT,
        sign_test_alpha=None,
        trajectory_policy_version=FORMAL_TRAJECTORY_POLICY_VERSION,
        trajectory_sense="min",
        preparation_time_policy=FORMAL_PREPARATION_TIME_POLICY,
        common_target_policy=FORMAL_COMMON_TARGET_POLICY,
        missing_progress_policy=FORMAL_MISSING_PROGRESS_POLICY,
        status_reconciliation_policy=FORMAL_STATUS_RECONCILIATION_POLICY,
        horizon_policy=FORMAL_HORIZON_POLICY,
        require_complete_manifest_coverage=True,
        require_all_formal_slots_succeeded=True,
    )
    analyze.set_defaults(handler=_analyze_command)

    analyze_formal = commands.add_parser(
        "analyze-formal",
        help=("authenticate and analyze exactly the frozen 900-slot formal campaign"),
    )
    analyze_formal.add_argument("--manifest", required=True)
    analyze_formal.add_argument("--plan", required=True)
    analyze_formal.add_argument(
        "--expected-plan-id",
        required=True,
        help="plan ID copied from the independent freeze-formal-plan output",
    )
    analyze_formal.add_argument(
        "--expected-evidence-root",
        required=True,
        type=_expected_evidence_root_argument,
        help=(
            "external 64-character SHA256 root copied from the completed "
            "run-formal output"
        ),
    )
    analyze_formal.add_argument("--results-root", required=True)
    analyze_formal.add_argument("--output", required=True)
    analyze_formal.add_argument(
        "--resource-lock",
        default=str(DEFAULT_BENCHMARK_RESOURCE_LOCK),
        help="shared non-blocking advisory lock for formal campaign analysis",
    )
    analyze_formal.set_defaults(handler=_analyze_formal_command)
    return parser


def dispatch(
    args: argparse.Namespace, hooks: CLIHooks = DEFAULT_HOOKS
) -> Mapping[str, Any]:
    handler = getattr(args, "handler", None)
    if handler is None:
        raise CLIError("no command selected")
    return handler(args, hooks)


def main(argv: Sequence[str] | None = None, *, hooks: CLIHooks = DEFAULT_HOOKS) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    try:
        result = dispatch(arguments, hooks)
    except (CLIError, FileExistsError, FileNotFoundError, ValueError) as error:
        parser.error(str(error))
    print(canonical_json(result))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through main()
    raise SystemExit(main())


__all__ = [
    "CLIError",
    "CLIHooks",
    "DEFAULT_HOOKS",
    "LoadedManifest",
    "build_parser",
    "dispatch",
    "load_benchmark_manifest",
    "load_candidate_pool",
    "load_qualification_ledger",
    "main",
]
