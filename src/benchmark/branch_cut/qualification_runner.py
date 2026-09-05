"""Algorithm-neutral solver adapter for non-trivial-instance qualification.

Qualification deliberately runs outside every paper comparison profile.  A
temporary pair of realized-service and objective-improvement bounds may be used
to *find* a witness.  While those rows are active, the objective is the constant
zero so the auxiliary solve is a pure feasibility search.  Both rows are then
removed and the original objective restored as one transaction.  The complete
incumbent is installed in a fresh copy of the original MLND model and accepted
as evidence only after that unmodified model produces a feasible incumbent
under single-thread native settings.

The module imports Gurobi lazily through :class:`MLNDQualificationAdapter`.
Tests can inject adapters implementing :class:`QualificationAdapter` and do not
need a solver licence.
"""

from __future__ import annotations

from array import array
from contextlib import contextmanager
import gc
import math
import sys
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from time import perf_counter
from typing import Protocol, runtime_checkable

from src.benchmark.branch_cut.qualification import (
    QualificationDecision,
    QualificationEvidence,
    QualificationPolicy,
    assess_qualification,
    objective_margin,
    service_tolerance,
)
from src.benchmark.branch_cut.qualification_geometry_start import (
    GeometryStartAttemptAudit,
    GeometryStartInfeasibilityError,
    GeometryStartInstallAudit,
    GeometryStartPlan,
    GeometryStartPlanAudit,
    GeometryStartPolicy,
    build_geometry_start_plan,
)
from src.benchmark.branch_cut.hashing import canonical_sha256
from src.benchmark.branch_cut.qualification_incumbent_reuse import (
    ConstraintSignatureExtractionError,
    DevelopmentQualificationIncumbentPool,
    QualificationConstraintSignature,
    QualificationReuseProvenance,
    ReusableQualificationIncumbent,
    extract_gurobi_constraint_signature,
    ordered_variable_name_sha256,
)


class QualificationRunnerReason:
    """Stable runner-level rejection reason codes."""

    ADAPTER_INITIALIZATION_FAILED = "qualification_adapter_initialization_failed"
    COURIER_BASELINE_EXTRACTION_FAILED = "courier_baseline_extraction_failed"
    NATIVE_WITNESS_SOLVE_FAILED = "native_witness_solve_failed"
    TEMPORARY_CONSTRAINT_ADD_FAILED = "temporary_service_constraint_add_failed"
    TEMPORARY_CONSTRAINT_REMOVAL_FAILED = "temporary_service_constraint_removal_failed"
    CONSTRAINED_WITNESS_SOLVE_FAILED = "constrained_witness_solve_failed"
    GEOMETRY_START_FALLBACK_FAILED = "geometry_start_fallback_failed"
    GEOMETRY_START_POLICY_FAILED = "geometry_start_policy_failed"
    GEOMETRY_START_INSTALLATION_FAILED = "geometry_start_installation_failed"
    GEOMETRY_START_SOLVE_FAILED = "geometry_start_solve_failed"
    GEOMETRY_START_SAFETY_CHECK_FAILED = "geometry_start_safety_check_failed"
    GEOMETRY_START_CLEANUP_FAILED = "geometry_start_cleanup_failed"
    NO_POSITIVE_SERVICE_INCUMBENT = "no_positive_service_incumbent"
    NO_OBJECTIVE_IMPROVING_INCUMBENT = "no_objective_improving_incumbent"
    COMPLETE_INCUMBENT_CAPTURE_FAILED = "complete_incumbent_capture_failed"
    WITNESS_MODEL_DISPOSAL_FAILED = "witness_model_disposal_failed"
    FRESH_MODEL_INITIALIZATION_FAILED = "fresh_model_initialization_failed"
    FRESH_MODEL_REUSED = "fresh_model_factory_reused_witness_model"
    FRESH_MODEL_BASELINE_EXTRACTION_FAILED = (
        "fresh_model_courier_baseline_extraction_failed"
    )
    FRESH_MODEL_BASELINE_MISMATCH = "fresh_model_courier_baseline_mismatch"
    COMPLETE_START_INSTALLATION_FAILED = "complete_start_installation_failed"
    FRESH_VALIDATION_SOLVE_FAILED = "fresh_original_model_validation_solve_failed"
    FRESH_VALIDATION_NO_INCUMBENT = "fresh_original_model_has_no_incumbent"
    FRESH_MODEL_DISPOSAL_FAILED = "fresh_model_disposal_failed"


class QualificationInfrastructureError(RuntimeError):
    """A solver/runtime failure that must never reject a candidate instance."""


NORMAL_QUALIFICATION_SOLVER_STATUSES = frozenset(
    {"OPTIMAL", "TIME_LIMIT", "INFEASIBLE"}
)
NORMAL_QUALIFICATION_RUNNER_REASONS = frozenset(
    {
        QualificationRunnerReason.GEOMETRY_START_FALLBACK_FAILED,
        QualificationRunnerReason.NO_OBJECTIVE_IMPROVING_INCUMBENT,
        QualificationRunnerReason.NO_POSITIVE_SERVICE_INCUMBENT,
    }
)


def _require_sha256(value: object, *, field_name: str) -> str:
    """Return one canonical lowercase SHA256 digest or fail closed."""

    if not isinstance(value, str) or len(value) != 64 or value.lower() != value:
        raise ValueError(f"{field_name} must be a lowercase SHA256 digest")
    try:
        digest = bytes.fromhex(value)
    except ValueError as error:
        raise ValueError(f"{field_name} must be a lowercase SHA256 digest") from error
    if len(digest) != 32:
        raise ValueError(f"{field_name} must be a lowercase SHA256 digest")
    return value


def _require_screening_limit(value: object, *, field_name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{field_name} must be a positive integer or None")
    return value


@dataclass(frozen=True, slots=True)
class QualificationModelContract:
    """Candidate-owned model identity that both qualification copies must match."""

    qualification_case_sha256: str
    effective_model_config_sha256: str
    demand_scale: float
    locker_fixed_cost_per_hour: float
    display_origin_choices: int | None
    general_corridor_choices: int | None
    screening_is_exact: bool

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "qualification_case_sha256",
            _require_sha256(
                self.qualification_case_sha256,
                field_name="qualification_case_sha256",
            ),
        )
        object.__setattr__(
            self,
            "effective_model_config_sha256",
            _require_sha256(
                self.effective_model_config_sha256,
                field_name="effective_model_config_sha256",
            ),
        )
        demand_scale = float(self.demand_scale)
        if not math.isfinite(demand_scale) or demand_scale <= 0.0:
            raise ValueError("demand_scale must be positive and finite")
        object.__setattr__(self, "demand_scale", demand_scale)
        fixed_cost = float(self.locker_fixed_cost_per_hour)
        if not math.isfinite(fixed_cost) or fixed_cost < 0.0:
            raise ValueError(
                "locker_fixed_cost_per_hour must be finite and nonnegative"
            )
        object.__setattr__(self, "locker_fixed_cost_per_hour", fixed_cost)
        object.__setattr__(
            self,
            "display_origin_choices",
            _require_screening_limit(
                self.display_origin_choices,
                field_name="display_origin_choices",
            ),
        )
        object.__setattr__(
            self,
            "general_corridor_choices",
            _require_screening_limit(
                self.general_corridor_choices,
                field_name="general_corridor_choices",
            ),
        )
        if type(self.screening_is_exact) is not bool:
            raise ValueError("screening_is_exact must be boolean")

    def to_payload(self) -> dict[str, object]:
        return {
            "qualification_case_sha256": self.qualification_case_sha256,
            "effective_model_config_sha256": self.effective_model_config_sha256,
            "demand_scale": self.demand_scale,
            "locker_fixed_cost_per_hour": self.locker_fixed_cost_per_hour,
            "display_origin_choices": self.display_origin_choices,
            "general_corridor_choices": self.general_corridor_choices,
            "screening_is_exact": self.screening_is_exact,
        }


@dataclass(frozen=True, slots=True)
class QualificationModelAudit:
    """Runtime readback from one freshly constructed qualification model."""

    effective_model_config_sha256: str
    demand_scale: float
    locker_fixed_cost_per_hour: float
    display_origin_choices: int | None
    general_corridor_choices: int | None
    screening_is_exact: bool
    model_fingerprint: int | None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "effective_model_config_sha256",
            _require_sha256(
                self.effective_model_config_sha256,
                field_name="effective_model_config_sha256",
            ),
        )
        demand_scale = float(self.demand_scale)
        if not math.isfinite(demand_scale) or demand_scale <= 0.0:
            raise ValueError("demand_scale must be positive and finite")
        object.__setattr__(self, "demand_scale", demand_scale)
        fixed_cost = float(self.locker_fixed_cost_per_hour)
        if not math.isfinite(fixed_cost) or fixed_cost < 0.0:
            raise ValueError(
                "locker_fixed_cost_per_hour must be finite and nonnegative"
            )
        object.__setattr__(self, "locker_fixed_cost_per_hour", fixed_cost)
        object.__setattr__(
            self,
            "display_origin_choices",
            _require_screening_limit(
                self.display_origin_choices,
                field_name="display_origin_choices",
            ),
        )
        object.__setattr__(
            self,
            "general_corridor_choices",
            _require_screening_limit(
                self.general_corridor_choices,
                field_name="general_corridor_choices",
            ),
        )
        if type(self.screening_is_exact) is not bool:
            raise ValueError("screening_is_exact must be boolean")
        if self.model_fingerprint is not None and (
            isinstance(self.model_fingerprint, bool)
            or not isinstance(self.model_fingerprint, int)
        ):
            raise ValueError("model_fingerprint must be an integer or None")

    def contract_values(self) -> tuple[object, ...]:
        return (
            self.effective_model_config_sha256,
            self.demand_scale,
            self.locker_fixed_cost_per_hour,
            self.display_origin_choices,
            self.general_corridor_choices,
            self.screening_is_exact,
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "effective_model_config_sha256": self.effective_model_config_sha256,
            "demand_scale": self.demand_scale,
            "locker_fixed_cost_per_hour": self.locker_fixed_cost_per_hour,
            "display_origin_choices": self.display_origin_choices,
            "general_corridor_choices": self.general_corridor_choices,
            "screening_is_exact": self.screening_is_exact,
            "model_fingerprint": self.model_fingerprint,
        }


def _validate_model_audit(
    audit: QualificationModelAudit,
    contract: QualificationModelContract,
    *,
    stage: str,
    witness_audit: QualificationModelAudit | None = None,
) -> None:
    expected = (
        contract.effective_model_config_sha256,
        contract.demand_scale,
        contract.locker_fixed_cost_per_hour,
        contract.display_origin_choices,
        contract.general_corridor_choices,
        contract.screening_is_exact,
    )
    if audit.contract_values() != expected:
        raise QualificationInfrastructureError(
            f"{stage} model audit does not match qualification case "
            f"{contract.qualification_case_sha256}"
        )
    if witness_audit is not None:
        if audit.contract_values() != witness_audit.contract_values():
            raise QualificationInfrastructureError(
                "fresh qualification model audit differs from witness model audit"
            )
        if (
            audit.model_fingerprint is None
            or witness_audit.model_fingerprint is None
            or audit.model_fingerprint != witness_audit.model_fingerprint
        ):
            raise QualificationInfrastructureError(
                "fresh qualification model fingerprint differs from witness model"
            )


def _read_required_model_audit(
    adapter: QualificationAdapter,
    contract: QualificationModelContract,
    *,
    stage: str,
    witness_audit: QualificationModelAudit | None = None,
) -> QualificationModelAudit:
    """Read and validate a mandatory audit without converting drift to rejection."""

    try:
        audit = adapter.qualification_model_audit()
        if not isinstance(audit, QualificationModelAudit):
            raise TypeError(
                "qualification_model_audit must return QualificationModelAudit"
            )
        _validate_model_audit(
            audit,
            contract,
            stage=stage,
            witness_audit=witness_audit,
        )
    except QualificationInfrastructureError:
        raise
    except Exception as error:
        raise QualificationInfrastructureError(
            f"{stage} qualification model identity audit failed: "
            f"{type(error).__name__}: {error}"
        ) from error
    return audit


def _evidence_model_identity(
    contract: QualificationModelContract | None,
    audit: QualificationModelAudit | None,
    *,
    stage: str,
) -> tuple[str | None, str | None, bool | None]:
    """Return actual audited identity fields, preserving all-None legacy evidence."""

    if contract is None:
        return None, None, None
    if audit is None:
        raise QualificationInfrastructureError(
            f"{stage} qualification evidence is missing its mandatory model audit"
        )
    _validate_model_audit(audit, contract, stage=f"{stage} evidence")
    return (
        contract.qualification_case_sha256,
        audit.effective_model_config_sha256,
        audit.screening_is_exact,
    )


def _require_normal_solver_status(status: object, *, stage: str) -> str:
    """Accept only preregistered terminal statuses during qualification.

    Qualification has no callback-driven termination, node/work/solution
    limits, or user objective limit.  Every other status therefore represents
    an infrastructure, numerical, resource, or unexpected-control failure and
    must stop the batch instead of advancing the replacement ladder.
    """

    if (
        not isinstance(status, str)
        or status not in NORMAL_QUALIFICATION_SOLVER_STATUSES
    ):
        raise QualificationInfrastructureError(
            f"unexpected qualification solver status during {stage}: {status!r}; "
            "allowed statuses are INFEASIBLE, OPTIMAL, and TIME_LIMIT"
        )
    return status


@dataclass(frozen=True, slots=True)
class NativeQualificationSettings:
    """Solver settings shared by witness search and fresh-model validation."""

    time_limit_seconds: float = 30.0
    mip_gap: float = 0.01
    solver_seed: int = 42
    threads: int = 1
    output_flag: int = 0
    method: int = -1
    miqcp_method: int = -1
    mip_focus: int = 0
    heuristics: float = 0.05
    no_rel_heur_time: float = 0.0
    presolve: int = -1
    native_cuts: int = -1
    root_cut_passes: int = 2
    numeric_focus: int = 0
    feasibility_tolerance: float = 1e-6
    optimality_tolerance: float = 1e-6
    integer_feasibility_tolerance: float = 1e-5

    def __post_init__(self) -> None:
        if not math.isfinite(self.time_limit_seconds) or self.time_limit_seconds <= 0.0:
            raise ValueError("time_limit_seconds must be positive and finite")
        if not math.isfinite(self.mip_gap) or not 0.0 <= self.mip_gap <= 1.0:
            raise ValueError("mip_gap must lie in [0, 1]")
        if self.threads != 1:
            raise ValueError("qualification must use matched single-thread settings")
        if isinstance(self.solver_seed, bool) or not isinstance(self.solver_seed, int):
            raise ValueError("solver_seed must be an integer")
        if not 0 <= self.solver_seed <= 2_000_000_000:
            raise ValueError("solver_seed must lie in [0, 2000000000]")
        if self.output_flag not in {0, 1}:
            raise ValueError("output_flag must be zero or one")
        if self.method not in {-1, 0, 1, 2, 3, 4, 5}:
            raise ValueError("method must be a valid Gurobi Method value")
        if self.miqcp_method not in {-1, 0, 1}:
            raise ValueError("miqcp_method must lie in {-1,0,1}")
        if self.mip_focus not in {0, 1, 2, 3}:
            raise ValueError("mip_focus must lie in {0,1,2,3}")
        if not math.isfinite(self.heuristics) or not 0.0 <= self.heuristics <= 1.0:
            raise ValueError("heuristics must lie in [0, 1]")
        if not math.isfinite(self.no_rel_heur_time) or self.no_rel_heur_time < 0.0:
            raise ValueError("no_rel_heur_time must be finite and nonnegative")
        if (
            isinstance(self.root_cut_passes, bool)
            or not isinstance(self.root_cut_passes, int)
            or not -1 <= self.root_cut_passes <= 2_000_000_000
        ):
            raise ValueError(
                "root_cut_passes must be -1 or an integer in [0, 2000000000]"
            )
        for name, value in (
            ("feasibility_tolerance", self.feasibility_tolerance),
            ("optimality_tolerance", self.optimality_tolerance),
            ("integer_feasibility_tolerance", self.integer_feasibility_tolerance),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")


def _ordered_variable_name_sha256(names: Sequence[str]) -> str:
    """Hash an ordered variable-name sequence without ambiguous delimiters."""

    return ordered_variable_name_sha256(names)


@dataclass(frozen=True, slots=True)
class CompleteIncumbent:
    """Compact complete start tied to one exact ordered model variable vector."""

    objective: float
    variable_count: int
    ordered_var_name_sha256: str
    values: Sequence[float] = field(repr=False)
    model_fingerprint: int | None = None

    def __post_init__(self) -> None:
        if not math.isfinite(self.objective):
            raise ValueError("complete incumbent objective must be finite")
        if self.variable_count < 1:
            raise ValueError("complete incumbent must contain every model variable")
        try:
            digest_bytes = bytes.fromhex(self.ordered_var_name_sha256)
        except ValueError as error:
            raise ValueError(
                "ordered_var_name_sha256 must be a SHA256 digest"
            ) from error
        if (
            len(digest_bytes) != 32
            or len(self.ordered_var_name_sha256) != 64
            or self.ordered_var_name_sha256.lower() != self.ordered_var_name_sha256
        ):
            raise ValueError("ordered_var_name_sha256 must be a SHA256 digest")
        compact_values = array("d", map(float, self.values))
        if len(compact_values) != self.variable_count:
            raise ValueError(
                "complete incumbent value count does not match variable_count"
            )
        if not all(math.isfinite(value) for value in compact_values):
            raise ValueError("complete incumbent variable values must be finite")
        if self.model_fingerprint is not None and (
            isinstance(self.model_fingerprint, bool)
            or not isinstance(self.model_fingerprint, int)
        ):
            raise ValueError("model_fingerprint must be an integer when present")
        object.__setattr__(self, "values", compact_values)

    @classmethod
    def from_ordered_values(
        cls,
        *,
        objective: float,
        variable_names: Sequence[str],
        values: Sequence[float],
        model_fingerprint: int | None = None,
    ) -> CompleteIncumbent:
        return cls(
            objective=objective,
            variable_count=len(variable_names),
            ordered_var_name_sha256=_ordered_variable_name_sha256(variable_names),
            values=values,
            model_fingerprint=model_fingerprint,
        )


@dataclass(frozen=True, slots=True)
class TemporaryWitnessConstraintToken:
    """Opaque token owning temporary rows and the displaced original objective."""

    service_constraint: object
    objective_constraint: object
    original_objective: object = field(repr=False)
    original_model_sense: int
    feasibility_objective_used: bool = True

    @property
    def constraints(self) -> tuple[object, object]:
        return (self.service_constraint, self.objective_constraint)


@dataclass(frozen=True, slots=True)
class QualificationSolution:
    """Solver-independent facts extracted from the current incumbent.

    ``active_h`` counts open H sites whose incident flow on physical arcs is
    numerically material.  Merely setting ``x=1`` or carrying flow on a virtual
    reliability/state arc does not make a site active.  The accompanying
    threshold is the demand-scaled :func:`service_tolerance` in orders/h.
    """

    status: str
    has_incumbent: bool
    objective: float | None = None
    realized_service_per_hour: float | None = None
    open_facilities: int = 0
    positive_cost_open_facilities: int = 0
    candidate_h: int = 0
    open_h: int = 0
    active_h: int = 0
    active_h_threshold_per_hour: float | None = None

    def __post_init__(self) -> None:
        if not self.status:
            raise ValueError("solution status must be nonempty")
        if self.has_incumbent:
            if self.objective is None or not math.isfinite(self.objective):
                raise ValueError("an incumbent requires a finite objective")
            if self.realized_service_per_hour is None or not math.isfinite(
                self.realized_service_per_hour
            ):
                raise ValueError("an incumbent requires finite realized service")
        if self.active_h < 0 or self.active_h > self.open_h:
            raise ValueError("active_h must lie between zero and open_h")
        if self.active_h_threshold_per_hour is not None and (
            not math.isfinite(self.active_h_threshold_per_hour)
            or self.active_h_threshold_per_hour <= 0.0
        ):
            raise ValueError("active-H threshold must be positive and finite")


@runtime_checkable
class QualificationAdapter(Protocol):
    """Minimal backend contract used by :func:`qualify_canonical_instance`."""

    def qualification_model_audit(self) -> QualificationModelAudit: ...

    def courier_only_objective(self) -> float: ...

    def total_demand_per_hour(self) -> float: ...

    def candidate_h_count(self) -> int: ...

    def clear_mip_starts(self) -> None: ...

    def qualification_constraint_signature(
        self,
    ) -> QualificationConstraintSignature: ...

    def install_constraint_equivalent_complete_incumbent(
        self,
        incumbent: CompleteIncumbent,
        expected_constraint_signature: QualificationConstraintSignature,
    ) -> None: ...

    def solve_native(self, settings: NativeQualificationSettings) -> str: ...

    def extract_solution(self) -> QualificationSolution: ...

    def add_witness_search_constraints(
        self,
        minimum_service_per_hour: float,
        maximum_objective: float,
    ) -> object: ...

    def remove_temporary_constraint(self, token: object) -> None: ...

    def plan_geometry_start(
        self,
        *,
        scale: str,
        target_support_size: int,
        required_service_per_hour: float,
        active_h_tolerance_per_hour: float,
        policy: GeometryStartPolicy,
    ) -> GeometryStartPlan: ...

    def install_geometry_start(
        self,
        plan: GeometryStartPlan,
    ) -> GeometryStartInstallAudit: ...

    def solve_geometry_start(
        self,
        settings: NativeQualificationSettings,
        policy: GeometryStartPolicy,
    ) -> str: ...

    def clear_geometry_start(self) -> None: ...

    def capture_complete_incumbent(self) -> CompleteIncumbent: ...

    def install_complete_incumbent(self, incumbent: CompleteIncumbent) -> None: ...

    def dispose(self) -> None: ...


AdapterFactory = Callable[[object, str], QualificationAdapter]


@dataclass(frozen=True, slots=True)
class QualificationPhaseTimings:
    """Wall-clock seconds for auditable qualification phases."""

    witness_adapter_initialization_seconds: float = 0.0
    baseline_extraction_seconds: float = 0.0
    native_witness_solve_seconds: float = 0.0
    temporary_constraint_add_seconds: float = 0.0
    constrained_witness_solve_seconds: float = 0.0
    geometry_start_planning_seconds: float = 0.0
    geometry_start_installation_seconds: float = 0.0
    geometry_start_solve_seconds: float = 0.0
    geometry_start_cleanup_seconds: float = 0.0
    complete_incumbent_capture_seconds: float = 0.0
    temporary_constraint_removal_seconds: float = 0.0
    witness_model_disposal_seconds: float = 0.0
    fresh_adapter_initialization_seconds: float = 0.0
    fresh_baseline_extraction_seconds: float = 0.0
    complete_start_installation_seconds: float = 0.0
    fresh_validation_solve_seconds: float = 0.0
    fresh_model_disposal_seconds: float = 0.0
    total_seconds: float = 0.0

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")

    def to_payload(self) -> dict[str, float]:
        return {name: float(getattr(self, name)) for name in self.__dataclass_fields__}


class _QualificationPhaseRecorder:
    """Small mutable recorder kept private from the immutable outcome."""

    _PHASE_NAMES = tuple(
        name
        for name in QualificationPhaseTimings.__dataclass_fields__
        if name != "total_seconds"
    )

    def __init__(self) -> None:
        self._started = perf_counter()
        self._seconds = {name: 0.0 for name in self._PHASE_NAMES}

    @contextmanager
    def measure(self, phase_name: str) -> Iterator[None]:
        if phase_name not in self._seconds:
            raise ValueError(f"unknown qualification phase {phase_name!r}")
        started = perf_counter()
        try:
            yield
        finally:
            self._seconds[phase_name] += max(0.0, perf_counter() - started)

    def snapshot(self) -> QualificationPhaseTimings:
        return QualificationPhaseTimings(
            **self._seconds,
            total_seconds=max(0.0, perf_counter() - self._started),
        )


def _json_safe_payload(payload: dict[str, object]) -> dict[str, object]:
    """Replace non-finite floats recursively so canonical JSON stays strict."""

    def normalize(value: object) -> object:
        if isinstance(value, float):
            return value if math.isfinite(value) else None
        if isinstance(value, dict):
            return {key: normalize(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [normalize(item) for item in value]
        return value

    return {key: normalize(value) for key, value in payload.items()}


@dataclass(frozen=True, slots=True)
class QualificationRunnerOutcome:
    """Structured qualification result, including adapter-level failures."""

    accepted: bool
    rejection_reasons: tuple[str, ...]
    evidence: QualificationEvidence
    decision: QualificationDecision
    required_service_per_hour: float
    required_objective_upper_bound: float | None
    temporary_constraint_used: bool
    temporary_constraint_removed: bool | None
    fresh_model_validation_performed: bool
    witness_status: str
    validation_status: str
    captured_variable_count: int
    exception_messages: tuple[str, ...]
    feasibility_objective_used: bool = False
    expected_model_contract: QualificationModelContract | None = None
    witness_model_audit: QualificationModelAudit | None = None
    fresh_model_audit: QualificationModelAudit | None = None
    geometry_start_policy: GeometryStartPolicy = field(
        default_factory=GeometryStartPolicy
    )
    geometry_start_attempts: tuple[GeometryStartAttemptAudit, ...] = ()
    geometry_start_selected_plan_sha256: str | None = None
    phase_timings: QualificationPhaseTimings = field(
        default_factory=QualificationPhaseTimings
    )

    def to_payload(self) -> dict[str, object]:
        return _json_safe_payload(
            {
                "accepted": self.accepted,
                "rejection_reasons": list(self.rejection_reasons),
                "evidence": self.evidence.to_payload(),
                "decision": self.decision.to_payload(),
                "required_service_per_hour": self.required_service_per_hour,
                "required_objective_upper_bound": self.required_objective_upper_bound,
                "temporary_constraint_used": self.temporary_constraint_used,
                "temporary_constraint_removed": self.temporary_constraint_removed,
                "fresh_model_validation_performed": self.fresh_model_validation_performed,
                "witness_status": self.witness_status,
                "validation_status": self.validation_status,
                "captured_variable_count": self.captured_variable_count,
                "exception_messages": list(self.exception_messages),
                "feasibility_objective_used": self.feasibility_objective_used,
                "expected_model_contract": (
                    self.expected_model_contract.to_payload()
                    if self.expected_model_contract is not None
                    else None
                ),
                "witness_model_audit": (
                    self.witness_model_audit.to_payload()
                    if self.witness_model_audit is not None
                    else None
                ),
                "fresh_model_audit": (
                    self.fresh_model_audit.to_payload()
                    if self.fresh_model_audit is not None
                    else None
                ),
                "geometry_start_policy": self.geometry_start_policy.to_payload(),
                "geometry_start_attempts": [
                    attempt.to_payload() for attempt in self.geometry_start_attempts
                ],
                "geometry_start_selected_plan_sha256": (
                    self.geometry_start_selected_plan_sha256
                ),
                "phase_timings": self.phase_timings.to_payload(),
            }
        )


def _exception_message(
    stage: str,
    error: BaseException,
    *,
    fail_fast: bool = False,
) -> str:
    message = f"{stage}: {type(error).__name__}: {error}"
    if isinstance(error, QualificationInfrastructureError):
        raise error
    if isinstance(error, (MemoryError, KeyboardInterrupt, SystemExit)):
        # Resource exhaustion and process control are never case evidence.
        raise error
    # Gurobi reports native allocation failure as GurobiError rather than
    # Python MemoryError.  Normalize its stable public error number so the CLI
    # stops the batch instead of treating OOM as an instance rejection.
    if getattr(error, "errno", None) == 10001:  # GRB.ERROR_OUT_OF_MEMORY
        raise MemoryError(f"Gurobi out of memory during {stage}: {error}") from error
    if fail_fast:
        raise QualificationInfrastructureError(message) from error
    return message


def _deferred_infrastructure_error(
    stage: str,
    error: BaseException,
) -> BaseException:
    """Normalize a primary that must be re-raised after recovery cleanup."""

    if isinstance(
        error,
        (QualificationInfrastructureError, MemoryError, KeyboardInterrupt, SystemExit),
    ):
        return error
    if not isinstance(error, Exception):
        return error
    if getattr(error, "errno", None) == 10001:  # GRB.ERROR_OUT_OF_MEMORY
        return MemoryError(f"Gurobi out of memory during {stage}: {error}")
    return QualificationInfrastructureError(f"{stage}: {type(error).__name__}: {error}")


def _raise_deferred_infrastructure_error(
    normalized_error: BaseException,
    *,
    original_error: BaseException,
) -> None:
    if normalized_error is original_error:
        raise normalized_error
    raise normalized_error from original_error


def _cleanup_failure_message(stage: str, error: BaseException) -> str:
    return f"{stage}: {type(error).__name__}: {error}"


def _handle_cleanup_failure(
    stage: str,
    error: BaseException,
    *,
    primary_error: BaseException | None,
) -> None:
    """Preserve an active primary, otherwise abort on cleanup failure."""

    message = _cleanup_failure_message(stage, error)
    if primary_error is not None:
        primary_error.add_note(f"secondary cleanup failure: {message}")
        return
    if isinstance(error, (MemoryError, KeyboardInterrupt, SystemExit)):
        raise error
    if not isinstance(error, Exception):
        raise error
    if isinstance(error, QualificationInfrastructureError):
        raise error
    if getattr(error, "errno", None) == 10001:  # GRB.ERROR_OUT_OF_MEMORY
        raise MemoryError(f"Gurobi out of memory during {stage}: {error}") from error
    raise QualificationInfrastructureError(message) from error


def _append_unique(values: list[str], value: str) -> None:
    if value not in values:
        values.append(value)


def _fallback_candidate_h(instance: object) -> int:
    lockers = getattr(instance, "H", ())
    try:
        return len(lockers)
    except TypeError:
        return 0


def _default_adapter_factory(instance: object, mode: str) -> QualificationAdapter:
    return MLNDQualificationAdapter(instance, mode)


def _has_required_service(
    solution: QualificationSolution | None,
    required_service_per_hour: float,
) -> bool:
    return bool(
        solution is not None
        and solution.has_incumbent
        and solution.realized_service_per_hour is not None
        and solution.realized_service_per_hour + 1e-12 >= required_service_per_hour
    )


def _has_required_objective(
    solution: QualificationSolution | None,
    maximum_objective: float | None,
) -> bool:
    if maximum_objective is None:
        return False
    return bool(
        solution is not None
        and solution.has_incumbent
        and solution.objective is not None
        and solution.objective <= maximum_objective
    )


def _solution_fully_qualifies(
    solution: QualificationSolution | None,
    *,
    case_id: str,
    scale: str,
    qualification_run_id: str,
    instance_sha256: str,
    courier_objective: float,
    total_demand: float,
    policy: QualificationPolicy,
    expected_model_contract: QualificationModelContract | None,
    model_audit: QualificationModelAudit | None,
) -> bool:
    """Apply every structural rule before capturing a search-model witness."""

    if solution is None or not solution.has_incumbent:
        return False
    if solution.objective is None or solution.realized_service_per_hour is None:
        return False
    (
        qualification_case_sha256,
        effective_model_config_sha256,
        screening_is_exact,
    ) = _evidence_model_identity(
        expected_model_contract,
        model_audit,
        stage="witness",
    )
    evidence = QualificationEvidence(
        case_id=case_id,
        scale=scale,
        courier_only_objective=courier_objective,
        witness_objective=solution.objective,
        total_demand_per_hour=total_demand,
        witness_service_per_hour=solution.realized_service_per_hour,
        open_facilities=solution.open_facilities,
        positive_cost_open_facilities=solution.positive_cost_open_facilities,
        candidate_h=solution.candidate_h,
        open_h=solution.open_h,
        witness_feasible=True,
        qualification_run_id=qualification_run_id,
        instance_sha256=instance_sha256,
        active_h=solution.active_h,
        active_h_threshold_per_hour=solution.active_h_threshold_per_hour,
        qualification_case_sha256=qualification_case_sha256,
        effective_model_config_sha256=effective_model_config_sha256,
        screening_is_exact=screening_is_exact,
    )
    return assess_qualification(evidence, policy).accepted


def _qualify_canonical_instance_impl(
    instance: object,
    mode: str,
    *,
    case_id: str,
    scale: str,
    qualification_run_id: str,
    instance_sha256: str,
    policy: QualificationPolicy | None = None,
    settings: NativeQualificationSettings | None = None,
    geometry_start_policy: GeometryStartPolicy | None = None,
    adapter_factory: AdapterFactory | None = None,
    expected_model_contract: QualificationModelContract | None = None,
    minimum_service_per_hour: float | None = None,
    fail_fast_infrastructure: bool = False,
    development_incumbent_reuse_pool: (
        DevelopmentQualificationIncumbentPool | None
    ) = None,
    development_reuse_profile_id: str | None = None,
    development_expected_target_constraint_signature: (
        QualificationConstraintSignature | None
    ) = None,
) -> QualificationRunnerOutcome:
    """Qualify one canonical instance without changing its mathematical model.

    The first native solve is unconstrained.  Only if it fails to produce both
    the preregistered positive service and objective improvement are two
    temporary bounds added as one transaction.  Any captured incumbent is then
    checked by a distinct adapter created from the same factory; results from
    the constrained search model are never passed directly to
    :func:`assess_qualification` as feasible evidence.
    """

    phase_recorder = _QualificationPhaseRecorder()
    active_policy = policy or QualificationPolicy()
    native_settings = settings or NativeQualificationSettings()
    active_geometry_policy = geometry_start_policy or GeometryStartPolicy()
    if type(fail_fast_infrastructure) is not bool:
        raise ValueError("fail_fast_infrastructure must be boolean")
    if (development_incumbent_reuse_pool is None) != (
        development_reuse_profile_id is None
    ):
        raise ValueError(
            "development incumbent reuse requires both a pool and profile ID"
        )
    if development_incumbent_reuse_pool is not None:
        if not isinstance(
            development_incumbent_reuse_pool,
            DevelopmentQualificationIncumbentPool,
        ):
            raise TypeError(
                "development_incumbent_reuse_pool must be a "
                "DevelopmentQualificationIncumbentPool"
            )
        if expected_model_contract is None:
            raise ValueError(
                "development incumbent reuse requires an exact model contract"
            )
        if (
            not isinstance(development_reuse_profile_id, str)
            or not development_reuse_profile_id
        ):
            raise ValueError("development_reuse_profile_id must be nonempty")
    if development_expected_target_constraint_signature is not None:
        if development_incumbent_reuse_pool is None:
            raise ValueError(
                "development expected target signature requires incumbent reuse"
            )
        if not isinstance(
            development_expected_target_constraint_signature,
            QualificationConstraintSignature,
        ):
            raise TypeError(
                "development_expected_target_constraint_signature must be a "
                "QualificationConstraintSignature"
            )

    def infrastructure_message(stage: str, error: BaseException) -> str:
        return _exception_message(
            stage,
            error,
            fail_fast=fail_fast_infrastructure,
        )

    factory = adapter_factory or _default_adapter_factory
    reasons: list[str] = []
    exceptions: list[str] = []
    courier_objective = math.nan
    total_demand = math.nan
    candidate_h = _fallback_candidate_h(instance)
    validated_solution: QualificationSolution | None = None
    complete_incumbent: CompleteIncumbent | None = None
    witness_status = "NOT_RUN"
    validation_status = "NOT_RUN"
    temporary_used = False
    temporary_removed: bool | None = None
    feasibility_objective_used = False
    geometry_attempts: list[GeometryStartAttemptAudit] = []
    geometry_selected_plan_sha256: str | None = None
    fresh_validation = False
    witness_adapter: QualificationAdapter | None = None
    witness_model_audit: QualificationModelAudit | None = None
    fresh_model_audit: QualificationModelAudit | None = None
    reuse_provenance: QualificationReuseProvenance | None = None
    reuse_constraint_signature: QualificationConstraintSignature | None = None
    selected_reuse: ReusableQualificationIncumbent | None = None

    try:
        with phase_recorder.measure("witness_adapter_initialization_seconds"):
            witness_adapter = factory(instance, mode)
            if expected_model_contract is not None:
                witness_model_audit = _read_required_model_audit(
                    witness_adapter,
                    expected_model_contract,
                    stage="witness",
                )
    except Exception as error:  # keep a machine-readable rejected record
        if isinstance(error, MemoryError):
            raise
        _append_unique(reasons, QualificationRunnerReason.ADAPTER_INITIALIZATION_FAILED)
        exceptions.append(
            infrastructure_message("witness_adapter_initialization", error)
        )

    if witness_adapter is not None:
        try:
            with phase_recorder.measure("baseline_extraction_seconds"):
                courier_objective = float(witness_adapter.courier_only_objective())
                total_demand = float(witness_adapter.total_demand_per_hour())
                candidate_h = int(witness_adapter.candidate_h_count())
        except Exception as error:
            _append_unique(
                reasons,
                QualificationRunnerReason.COURIER_BASELINE_EXTRACTION_FAILED,
            )
            exceptions.append(infrastructure_message("baseline_extraction", error))

    if minimum_service_per_hour is None:
        if math.isfinite(total_demand) and total_demand >= 0.0:
            required_service = max(
                service_tolerance(total_demand),
                active_policy.minimum_service_fraction * total_demand,
            )
        else:
            required_service = 1e-3
    else:
        required_service = float(minimum_service_per_hour)
        if not math.isfinite(required_service) or required_service <= 0.0:
            raise ValueError("minimum_service_per_hour must be positive and finite")

    required_objective_upper_bound: float | None = None
    if math.isfinite(courier_objective):
        # The feasibility search must enforce every objective-side acceptance
        # rule.  Otherwise a constant-objective solve can stop at the first
        # incumbent that clears the numerical margin but misses the larger
        # preregistered savings fraction, creating a false rejection even when
        # a qualifying witness exists.
        required_improvement = max(
            objective_margin(
                courier_objective,
                active_policy.optimality_tolerance,
            ),
            (
                active_policy.minimum_savings_fraction * courier_objective
                if courier_objective > 0.0
                else 0.0
            ),
        )
        required_objective_upper_bound = courier_objective - required_improvement

    if development_incumbent_reuse_pool is not None and witness_adapter is not None:
        assert expected_model_contract is not None
        assert development_reuse_profile_id is not None
        if witness_model_audit is None:
            raise QualificationInfrastructureError(
                "development incumbent reuse requires the witness model audit"
            )
        try:
            reuse_provenance = QualificationReuseProvenance(
                qualification_run_id=qualification_run_id,
                case_id=case_id,
                profile_id=development_reuse_profile_id,
                scale=scale,
                mode=mode,
                physical_instance_sha256=instance_sha256,
                qualification_case_sha256=(
                    expected_model_contract.qualification_case_sha256
                ),
                effective_model_config_sha256=(
                    witness_model_audit.effective_model_config_sha256
                ),
                demand_scale=witness_model_audit.demand_scale,
                screening_is_exact=witness_model_audit.screening_is_exact,
            )
            reuse_constraint_signature = (
                witness_adapter.qualification_constraint_signature()
            )
            if not isinstance(
                reuse_constraint_signature,
                QualificationConstraintSignature,
            ):
                raise TypeError(
                    "qualification_constraint_signature must return "
                    "QualificationConstraintSignature"
                )
            if (
                development_expected_target_constraint_signature is not None
                and reuse_constraint_signature
                != development_expected_target_constraint_signature
            ):
                raise QualificationInfrastructureError(
                    "actual witness adapter constraint signature differs from "
                    "the development preflight signature"
                )
            selected_reuse = development_incumbent_reuse_pool.select(
                target_provenance=reuse_provenance,
                target_constraint_signature=reuse_constraint_signature,
            )
        except BaseException as error:
            if isinstance(error, (MemoryError, KeyboardInterrupt, SystemExit)):
                raise
            if isinstance(error, QualificationInfrastructureError):
                raise
            raise QualificationInfrastructureError(
                "development incumbent reuse audit failed: "
                f"{type(error).__name__}: {error}"
            ) from error

    current_solution: QualificationSolution | None = None
    if witness_adapter is not None:
        try:
            with phase_recorder.measure("native_witness_solve_seconds"):
                if selected_reuse is None:
                    witness_adapter.clear_mip_starts()
                else:
                    assert development_incumbent_reuse_pool is not None
                    assert reuse_provenance is not None
                    assert reuse_constraint_signature is not None
                    if not isinstance(selected_reuse.incumbent, CompleteIncumbent):
                        raise QualificationInfrastructureError(
                            "reuse pool returned a non-CompleteIncumbent value"
                        )
                    try:
                        witness_adapter.install_constraint_equivalent_complete_incumbent(
                            selected_reuse.incumbent,
                            reuse_constraint_signature,
                        )
                    except BaseException as error:
                        development_incumbent_reuse_pool.record_installation_failure(
                            target_provenance=reuse_provenance,
                            target_constraint_signature=reuse_constraint_signature,
                            selected=selected_reuse,
                            error=error,
                        )
                        if isinstance(
                            error,
                            (MemoryError, KeyboardInterrupt, SystemExit),
                        ):
                            raise
                        raise QualificationInfrastructureError(
                            "constraint-equivalent complete-start installation "
                            f"failed: {type(error).__name__}: {error}"
                        ) from error
                    development_incumbent_reuse_pool.record_installation(
                        target_provenance=reuse_provenance,
                        target_constraint_signature=reuse_constraint_signature,
                        selected=selected_reuse,
                    )
                witness_status = _require_normal_solver_status(
                    witness_adapter.solve_native(native_settings),
                    stage="native_witness_solve",
                )
                current_solution = witness_adapter.extract_solution()
        except Exception as error:
            _append_unique(
                reasons, QualificationRunnerReason.NATIVE_WITNESS_SOLVE_FAILED
            )
            exceptions.append(infrastructure_message("native_witness_solve", error))

    service_found = _has_required_service(current_solution, required_service)
    objective_found = _has_required_objective(
        current_solution,
        required_objective_upper_bound,
    )
    valid_witness_found = _solution_fully_qualifies(
        current_solution,
        case_id=case_id,
        scale=scale,
        qualification_run_id=qualification_run_id,
        instance_sha256=instance_sha256,
        courier_objective=courier_objective,
        total_demand=total_demand,
        policy=active_policy,
        expected_model_contract=expected_model_contract,
        model_audit=witness_model_audit,
    )
    if valid_witness_found and witness_adapter is not None:
        try:
            with phase_recorder.measure("complete_incumbent_capture_seconds"):
                complete_incumbent = witness_adapter.capture_complete_incumbent()
        except Exception as error:
            _append_unique(
                reasons,
                QualificationRunnerReason.COMPLETE_INCUMBENT_CAPTURE_FAILED,
            )
            exceptions.append(
                infrastructure_message("complete_incumbent_capture", error)
            )
    if (
        witness_adapter is not None
        and complete_incumbent is None
        and required_objective_upper_bound is not None
    ):
        temporary_used = True
        temporary_token: object | None = None
        geometry_errors: list[str] = []
        try:
            with phase_recorder.measure("temporary_constraint_add_seconds"):
                temporary_token = witness_adapter.add_witness_search_constraints(
                    required_service,
                    required_objective_upper_bound,
                )
                feasibility_objective_used = True
            try:
                with phase_recorder.measure("constrained_witness_solve_seconds"):
                    witness_status = _require_normal_solver_status(
                        witness_adapter.solve_native(native_settings),
                        stage="constrained_witness_solve",
                    )
                    current_solution = witness_adapter.extract_solution()
                service_found = service_found or _has_required_service(
                    current_solution, required_service
                )
                objective_found = objective_found or _has_required_objective(
                    current_solution, required_objective_upper_bound
                )
                constrained_witness_qualifies = _solution_fully_qualifies(
                    current_solution,
                    case_id=case_id,
                    scale=scale,
                    qualification_run_id=qualification_run_id,
                    instance_sha256=instance_sha256,
                    courier_objective=courier_objective,
                    total_demand=total_demand,
                    policy=active_policy,
                    expected_model_contract=expected_model_contract,
                    model_audit=witness_model_audit,
                )
                if constrained_witness_qualifies:
                    # Capture before model.remove()/update(): changing a Gurobi
                    # model can invalidate access to the current solution.
                    try:
                        with phase_recorder.measure(
                            "complete_incumbent_capture_seconds"
                        ):
                            complete_incumbent = (
                                witness_adapter.capture_complete_incumbent()
                            )
                    except Exception as error:
                        _append_unique(
                            reasons,
                            QualificationRunnerReason.COMPLETE_INCUMBENT_CAPTURE_FAILED,
                        )
                        exceptions.append(
                            infrastructure_message("complete_incumbent_capture", error)
                        )

                if (
                    complete_incumbent is None
                    and active_geometry_policy.enabled
                    and math.isfinite(total_demand)
                    and total_demand >= 0.0
                ):
                    try:
                        schedule = active_geometry_policy.schedule_for(
                            scale, candidate_h
                        )
                    except Exception as error:
                        message = infrastructure_message("geometry_start_policy", error)
                        _append_unique(
                            reasons,
                            QualificationRunnerReason.GEOMETRY_START_POLICY_FAILED,
                        )
                        exceptions.append(message)
                        schedule = ()
                    active_tolerance = service_tolerance(total_demand)
                    for target_support_size in schedule:
                        plan: GeometryStartPlan | None = None
                        failed_plan_audit = GeometryStartPlanAudit.planning_failure(
                            plan_version=active_geometry_policy.plan_version,
                            target_support_size=target_support_size,
                        )
                        try:
                            with phase_recorder.measure(
                                "geometry_start_planning_seconds"
                            ):
                                plan = witness_adapter.plan_geometry_start(
                                    scale=scale,
                                    target_support_size=target_support_size,
                                    required_service_per_hour=required_service,
                                    active_h_tolerance_per_hour=active_tolerance,
                                    policy=active_geometry_policy,
                                )
                            plan_audit = GeometryStartPlanAudit.from_plan(plan)
                        except GeometryStartInfeasibilityError as error:
                            # A conservative topology/flow precheck failure is
                            # a normal "no usable geometry seed" result, not an
                            # infrastructure exception.
                            message = _exception_message(
                                "geometry_start_planning",
                                error,
                            )
                            geometry_attempts.append(
                                GeometryStartAttemptAudit(
                                    plan=failed_plan_audit,
                                    per_attempt_seconds=(
                                        active_geometry_policy.per_attempt_seconds
                                    ),
                                    start_node_limit=(
                                        active_geometry_policy.start_node_limit
                                    ),
                                    error=message,
                                )
                            )
                            continue
                        except Exception as error:
                            raise QualificationInfrastructureError(
                                "unexpected geometry-start planning failure: "
                                f"{type(error).__name__}: {error}"
                            ) from error

                        assert plan is not None
                        try:
                            with phase_recorder.measure(
                                "geometry_start_installation_seconds"
                            ):
                                install_audit = witness_adapter.install_geometry_start(
                                    plan
                                )
                        except BaseException as error:
                            message = _cleanup_failure_message(
                                "geometry_start_installation",
                                error,
                            )
                            installation_error = _deferred_infrastructure_error(
                                "geometry_start_installation",
                                error,
                            )
                            geometry_errors.append(message)
                            _append_unique(
                                reasons,
                                QualificationRunnerReason.GEOMETRY_START_INSTALLATION_FAILED,
                            )
                            # The production installer is transactional, but
                            # the runner still issues a defensive recovery clear
                            # before terminating the registered ladder.
                            recovery_failed = False
                            try:
                                with phase_recorder.measure(
                                    "geometry_start_cleanup_seconds"
                                ):
                                    witness_adapter.clear_geometry_start()
                            except BaseException as cleanup_error:
                                recovery_failed = True
                                cleanup_message = _cleanup_failure_message(
                                    "geometry_start_installation_recovery",
                                    cleanup_error,
                                )
                                geometry_errors.append(cleanup_message)
                                _append_unique(
                                    reasons,
                                    QualificationRunnerReason.GEOMETRY_START_CLEANUP_FAILED,
                                )
                                _handle_cleanup_failure(
                                    "geometry_start_installation_recovery",
                                    cleanup_error,
                                    primary_error=installation_error,
                                )
                            # With no completed install audit, the recovery reset
                            # is not represented as an installed-attempt cleanup.
                            geometry_attempts.append(
                                GeometryStartAttemptAudit(
                                    plan=plan_audit,
                                    per_attempt_seconds=(
                                        active_geometry_policy.per_attempt_seconds
                                    ),
                                    start_node_limit=(
                                        active_geometry_policy.start_node_limit
                                    ),
                                    error=message,
                                )
                            )
                            if (
                                fail_fast_infrastructure
                                or recovery_failed
                                or isinstance(
                                    error,
                                    (
                                        QualificationInfrastructureError,
                                        MemoryError,
                                        KeyboardInterrupt,
                                        SystemExit,
                                    ),
                                )
                                or not isinstance(error, Exception)
                                or getattr(error, "errno", None) == 10001
                            ):
                                _raise_deferred_infrastructure_error(
                                    installation_error,
                                    original_error=error,
                                )
                            break

                        attempt_error: str | None = None
                        solve_status = "NOT_RUN"
                        solve_seconds = 0.0
                        full_witness_found = False
                        solve_failure = False
                        capture_failure = False
                        attempt_primary_error: BaseException | None = None
                        attempt_original_error: BaseException | None = None
                        safety_failure = not install_audit.safety_preserved
                        if safety_failure:
                            _append_unique(
                                reasons,
                                QualificationRunnerReason.GEOMETRY_START_SAFETY_CHECK_FAILED,
                            )
                            attempt_error = (
                                "geometry start changed the start-neutral model "
                                "fingerprint or structure"
                            )
                            attempt_primary_error = QualificationInfrastructureError(
                                attempt_error
                            )
                            attempt_original_error = attempt_primary_error
                        else:
                            solve_started = perf_counter()
                            try:
                                with phase_recorder.measure(
                                    "geometry_start_solve_seconds"
                                ):
                                    solve_status = _require_normal_solver_status(
                                        witness_adapter.solve_geometry_start(
                                            native_settings,
                                            active_geometry_policy,
                                        ),
                                        stage="geometry_start_solve",
                                    )
                                    witness_status = solve_status
                                    current_solution = (
                                        witness_adapter.extract_solution()
                                    )
                                solve_seconds = max(0.0, perf_counter() - solve_started)
                                service_found = service_found or (
                                    _has_required_service(
                                        current_solution,
                                        required_service,
                                    )
                                )
                                objective_found = objective_found or (
                                    _has_required_objective(
                                        current_solution,
                                        required_objective_upper_bound,
                                    )
                                )
                                solution_qualifies = _solution_fully_qualifies(
                                    current_solution,
                                    case_id=case_id,
                                    scale=scale,
                                    qualification_run_id=qualification_run_id,
                                    instance_sha256=instance_sha256,
                                    courier_objective=courier_objective,
                                    total_demand=total_demand,
                                    policy=active_policy,
                                    expected_model_contract=expected_model_contract,
                                    model_audit=witness_model_audit,
                                )
                                if solution_qualifies:
                                    try:
                                        with phase_recorder.measure(
                                            "complete_incumbent_capture_seconds"
                                        ):
                                            complete_incumbent = witness_adapter.capture_complete_incumbent()
                                        geometry_selected_plan_sha256 = plan.plan_sha256
                                        full_witness_found = True
                                    except BaseException as error:
                                        capture_failure = True
                                        attempt_error = _cleanup_failure_message(
                                            "complete_incumbent_capture",
                                            error,
                                        )
                                        attempt_primary_error = (
                                            _deferred_infrastructure_error(
                                                "complete_incumbent_capture",
                                                error,
                                            )
                                        )
                                        attempt_original_error = error
                                        geometry_errors.append(attempt_error)
                                        _append_unique(
                                            reasons,
                                            QualificationRunnerReason.COMPLETE_INCUMBENT_CAPTURE_FAILED,
                                        )
                            except BaseException as error:
                                solve_status = "ERROR"
                                witness_status = solve_status
                                solve_failure = True
                                solve_seconds = max(0.0, perf_counter() - solve_started)
                                attempt_error = _cleanup_failure_message(
                                    "geometry_start_solve",
                                    error,
                                )
                                attempt_primary_error = _deferred_infrastructure_error(
                                    "geometry_start_solve",
                                    error,
                                )
                                attempt_original_error = error
                                geometry_errors.append(attempt_error)
                                _append_unique(
                                    reasons,
                                    QualificationRunnerReason.GEOMETRY_START_SOLVE_FAILED,
                                )

                        if full_witness_found:
                            geometry_attempts.append(
                                GeometryStartAttemptAudit(
                                    plan=plan_audit,
                                    per_attempt_seconds=(
                                        active_geometry_policy.per_attempt_seconds
                                    ),
                                    start_node_limit=(
                                        active_geometry_policy.start_node_limit
                                    ),
                                    install=install_audit,
                                    solve_status=solve_status,
                                    solve_seconds=solve_seconds,
                                    full_witness_found=True,
                                )
                            )
                            break

                        cleanup_succeeded = False
                        cleanup_failed = False
                        try:
                            with phase_recorder.measure(
                                "geometry_start_cleanup_seconds"
                            ):
                                witness_adapter.clear_geometry_start()
                            cleanup_succeeded = True
                        except BaseException as error:
                            cleanup_failed = True
                            message = _cleanup_failure_message(
                                "geometry_start_cleanup",
                                error,
                            )
                            geometry_errors.append(message)
                            _append_unique(
                                reasons,
                                QualificationRunnerReason.GEOMETRY_START_CLEANUP_FAILED,
                            )
                            if attempt_error is None:
                                attempt_error = message
                            if attempt_primary_error is None:
                                attempt_primary_error = _deferred_infrastructure_error(
                                    "geometry_start_cleanup",
                                    error,
                                )
                                attempt_original_error = error
                            else:
                                _handle_cleanup_failure(
                                    "geometry_start_cleanup",
                                    error,
                                    primary_error=attempt_primary_error,
                                )
                        geometry_attempts.append(
                            GeometryStartAttemptAudit(
                                plan=plan_audit,
                                per_attempt_seconds=(
                                    active_geometry_policy.per_attempt_seconds
                                ),
                                start_node_limit=(
                                    active_geometry_policy.start_node_limit
                                ),
                                install=install_audit,
                                solve_status=solve_status,
                                solve_seconds=solve_seconds,
                                cleanup_performed=True,
                                cleanup_succeeded=cleanup_succeeded,
                                error=attempt_error,
                            )
                        )
                        if attempt_primary_error is not None and (
                            fail_fast_infrastructure
                            or cleanup_failed
                            or isinstance(
                                attempt_original_error,
                                (
                                    QualificationInfrastructureError,
                                    MemoryError,
                                    KeyboardInterrupt,
                                    SystemExit,
                                ),
                            )
                            or getattr(attempt_original_error, "errno", None) == 10001
                        ):
                            assert attempt_original_error is not None
                            _raise_deferred_infrastructure_error(
                                attempt_primary_error,
                                original_error=attempt_original_error,
                            )
                        if (
                            not cleanup_succeeded
                            or safety_failure
                            or solve_failure
                            or capture_failure
                        ):
                            break

                    if complete_incumbent is None and geometry_attempts:
                        _append_unique(
                            reasons,
                            QualificationRunnerReason.GEOMETRY_START_FALLBACK_FAILED,
                        )
                        exceptions.extend(geometry_errors)
            except Exception as error:
                _append_unique(
                    reasons,
                    QualificationRunnerReason.CONSTRAINED_WITNESS_SOLVE_FAILED,
                )
                exceptions.append(
                    infrastructure_message("constrained_witness_solve", error)
                )
        except Exception as error:
            _append_unique(
                reasons,
                QualificationRunnerReason.TEMPORARY_CONSTRAINT_ADD_FAILED,
            )
            exceptions.append(infrastructure_message("temporary_constraint_add", error))
        finally:
            primary_error = sys.exception()
            if temporary_token is not None:
                try:
                    with phase_recorder.measure("temporary_constraint_removal_seconds"):
                        witness_adapter.remove_temporary_constraint(temporary_token)
                    temporary_removed = True
                except BaseException as error:
                    temporary_removed = False
                    _append_unique(
                        reasons,
                        QualificationRunnerReason.TEMPORARY_CONSTRAINT_REMOVAL_FAILED,
                    )
                    _handle_cleanup_failure(
                        "temporary_constraint_removal",
                        error,
                        primary_error=primary_error,
                    )

    if not service_found:
        _append_unique(reasons, QualificationRunnerReason.NO_POSITIVE_SERVICE_INCUMBENT)
    if not objective_found:
        _append_unique(
            reasons,
            QualificationRunnerReason.NO_OBJECTIVE_IMPROVING_INCUMBENT,
        )
    if witness_adapter is not None:
        try:
            with phase_recorder.measure("witness_model_disposal_seconds"):
                witness_adapter.dispose()
        except BaseException as error:
            _append_unique(
                reasons,
                QualificationRunnerReason.WITNESS_MODEL_DISPOSAL_FAILED,
            )
            _handle_cleanup_failure(
                "witness_model_disposal",
                error,
                primary_error=None,
            )

    if (
        complete_incumbent is not None
        and temporary_removed is not False
        and QualificationRunnerReason.WITNESS_MODEL_DISPOSAL_FAILED not in reasons
    ):
        validation_adapter: QualificationAdapter | None = None
        validation_ready = False
        try:
            try:
                with phase_recorder.measure("fresh_adapter_initialization_seconds"):
                    validation_adapter = factory(instance, mode)
                    if expected_model_contract is not None:
                        fresh_model_audit = _read_required_model_audit(
                            validation_adapter,
                            expected_model_contract,
                            stage="fresh",
                            witness_audit=witness_model_audit,
                        )
            except Exception as error:
                _append_unique(
                    reasons,
                    QualificationRunnerReason.FRESH_MODEL_INITIALIZATION_FAILED,
                )
                exceptions.append(
                    infrastructure_message("fresh_model_initialization", error)
                )
            if validation_adapter is witness_adapter:
                _append_unique(reasons, QualificationRunnerReason.FRESH_MODEL_REUSED)
                # The witness was already disposed before this factory call.
                validation_adapter = None
            validation_ready = validation_adapter is not None
            if validation_ready:
                assert validation_adapter is not None
                try:
                    with phase_recorder.measure("fresh_baseline_extraction_seconds"):
                        validation_c0 = float(
                            validation_adapter.courier_only_objective()
                        )
                except Exception as error:
                    _append_unique(
                        reasons,
                        QualificationRunnerReason.FRESH_MODEL_BASELINE_EXTRACTION_FAILED,
                    )
                    exceptions.append(
                        infrastructure_message("fresh_model_baseline_extraction", error)
                    )
                    validation_ready = False
                else:
                    tolerance = active_policy.optimality_tolerance * max(
                        1.0, abs(courier_objective), abs(validation_c0)
                    )
                    if (
                        not math.isfinite(courier_objective)
                        or not math.isfinite(validation_c0)
                        or abs(validation_c0 - courier_objective) > tolerance
                    ):
                        _append_unique(
                            reasons,
                            QualificationRunnerReason.FRESH_MODEL_BASELINE_MISMATCH,
                        )
            if validation_ready:
                assert validation_adapter is not None
                try:
                    with phase_recorder.measure("complete_start_installation_seconds"):
                        validation_adapter.install_complete_incumbent(
                            complete_incumbent
                        )
                except Exception as error:
                    _append_unique(
                        reasons,
                        QualificationRunnerReason.COMPLETE_START_INSTALLATION_FAILED,
                    )
                    exceptions.append(
                        infrastructure_message("complete_start_installation", error)
                    )
                    validation_ready = False
            if validation_ready:
                assert validation_adapter is not None
                fresh_validation = True
                try:
                    with phase_recorder.measure("fresh_validation_solve_seconds"):
                        validation_status = _require_normal_solver_status(
                            validation_adapter.solve_native(native_settings),
                            stage="fresh_validation_solve",
                        )
                        # This immutable record is the only solver state retained
                        # after the fresh adapter is disposed in the outer finally.
                        solution = validation_adapter.extract_solution()
                    if solution.has_incumbent:
                        validated_solution = solution
                    else:
                        _append_unique(
                            reasons,
                            QualificationRunnerReason.FRESH_VALIDATION_NO_INCUMBENT,
                        )
                except Exception as error:
                    _append_unique(
                        reasons,
                        QualificationRunnerReason.FRESH_VALIDATION_SOLVE_FAILED,
                    )
                    exceptions.append(
                        infrastructure_message("fresh_validation_solve", error)
                    )
        finally:
            primary_error = sys.exception()
            if validation_adapter is not None:
                try:
                    with phase_recorder.measure("fresh_model_disposal_seconds"):
                        validation_adapter.dispose()
                except BaseException as error:
                    _append_unique(
                        reasons,
                        QualificationRunnerReason.FRESH_MODEL_DISPOSAL_FAILED,
                    )
                    _handle_cleanup_failure(
                        "fresh_model_disposal",
                        error,
                        primary_error=primary_error,
                    )

    witness_feasible = bool(
        validated_solution is not None
        and validated_solution.has_incumbent
        and QualificationRunnerReason.FRESH_MODEL_BASELINE_MISMATCH not in reasons
    )
    active_h_threshold_per_hour = (
        validated_solution.active_h_threshold_per_hour
        if validated_solution is not None
        and validated_solution.active_h_threshold_per_hour is not None
        else (
            service_tolerance(total_demand)
            if math.isfinite(total_demand) and total_demand >= 0.0
            else None
        )
    )
    (
        qualification_case_sha256,
        effective_model_config_sha256,
        screening_is_exact,
    ) = _evidence_model_identity(
        expected_model_contract,
        fresh_model_audit or witness_model_audit,
        stage="final",
    )
    evidence = QualificationEvidence(
        case_id=case_id,
        scale=scale,
        courier_only_objective=courier_objective,
        witness_objective=(
            float(validated_solution.objective)
            if validated_solution is not None
            and validated_solution.objective is not None
            else math.nan
        ),
        total_demand_per_hour=total_demand,
        witness_service_per_hour=(
            float(validated_solution.realized_service_per_hour)
            if validated_solution is not None
            and validated_solution.realized_service_per_hour is not None
            else math.nan
        ),
        open_facilities=(
            validated_solution.open_facilities if validated_solution is not None else 0
        ),
        positive_cost_open_facilities=(
            validated_solution.positive_cost_open_facilities
            if validated_solution is not None
            else 0
        ),
        candidate_h=(
            validated_solution.candidate_h
            if validated_solution is not None
            else candidate_h
        ),
        open_h=validated_solution.open_h if validated_solution is not None else 0,
        witness_feasible=witness_feasible,
        qualification_run_id=qualification_run_id,
        instance_sha256=instance_sha256,
        active_h=validated_solution.active_h if validated_solution is not None else 0,
        active_h_threshold_per_hour=active_h_threshold_per_hour,
        qualification_case_sha256=qualification_case_sha256,
        effective_model_config_sha256=effective_model_config_sha256,
        screening_is_exact=screening_is_exact,
    )
    decision = assess_qualification(evidence, active_policy)
    if fail_fast_infrastructure:
        infrastructure_reasons = tuple(
            reason
            for reason in reasons
            if reason not in NORMAL_QUALIFICATION_RUNNER_REASONS
        )
        if infrastructure_reasons:
            raise QualificationInfrastructureError(
                "qualification mechanism failure: " + ", ".join(infrastructure_reasons)
            )
    combined_reasons = list(reasons)
    for reason in decision.rejection_reasons:
        _append_unique(combined_reasons, reason)
    accepted = not combined_reasons
    if development_incumbent_reuse_pool is not None and accepted:
        if (
            reuse_provenance is None
            or reuse_constraint_signature is None
            or complete_incumbent is None
        ):
            raise QualificationInfrastructureError(
                "accepted reuse-enabled qualification lacks a cacheable witness"
            )
        try:
            development_incumbent_reuse_pool.offer(
                provenance=reuse_provenance,
                constraint_signature=reuse_constraint_signature,
                incumbent=complete_incumbent,
                accepted=True,
                fresh_validation_performed=fresh_validation,
            )
        except BaseException as error:
            if isinstance(error, (MemoryError, KeyboardInterrupt, SystemExit)):
                raise
            raise QualificationInfrastructureError(
                "development incumbent reuse offer failed: "
                f"{type(error).__name__}: {error}"
            ) from error
    return QualificationRunnerOutcome(
        accepted=accepted,
        rejection_reasons=tuple(combined_reasons),
        evidence=evidence,
        decision=decision,
        required_service_per_hour=required_service,
        required_objective_upper_bound=required_objective_upper_bound,
        temporary_constraint_used=temporary_used,
        temporary_constraint_removed=temporary_removed,
        fresh_model_validation_performed=fresh_validation,
        witness_status=witness_status,
        validation_status=validation_status,
        captured_variable_count=(
            complete_incumbent.variable_count if complete_incumbent is not None else 0
        ),
        exception_messages=tuple(exceptions),
        feasibility_objective_used=feasibility_objective_used,
        expected_model_contract=expected_model_contract,
        witness_model_audit=witness_model_audit,
        fresh_model_audit=fresh_model_audit,
        geometry_start_policy=active_geometry_policy,
        geometry_start_attempts=tuple(geometry_attempts),
        geometry_start_selected_plan_sha256=geometry_selected_plan_sha256,
        phase_timings=phase_recorder.snapshot(),
    )


class _LifecycleTrackedQualificationAdapter:
    """Track successful disposal without changing an injected adapter's API."""

    def __init__(self, delegate: QualificationAdapter):
        self.delegate = delegate
        self.disposed = False

    def __getattr__(self, name: str) -> object:
        return getattr(self.delegate, name)

    def dispose(self) -> None:
        if self.disposed:
            return
        self.delegate.dispose()
        self.disposed = True


def qualify_canonical_instance(
    instance: object,
    mode: str,
    *,
    case_id: str,
    scale: str,
    qualification_run_id: str,
    instance_sha256: str,
    policy: QualificationPolicy | None = None,
    settings: NativeQualificationSettings | None = None,
    geometry_start_policy: GeometryStartPolicy | None = None,
    adapter_factory: AdapterFactory | None = None,
    expected_model_contract: QualificationModelContract | None = None,
    minimum_service_per_hour: float | None = None,
    fail_fast_infrastructure: bool = False,
    development_incumbent_reuse_pool: (
        DevelopmentQualificationIncumbentPool | None
    ) = None,
    development_reuse_profile_id: str | None = None,
    development_expected_target_constraint_signature: (
        QualificationConstraintSignature | None
    ) = None,
) -> QualificationRunnerOutcome:
    """Qualify one instance and release every constructed model on all exits."""

    raw_factory = adapter_factory or _default_adapter_factory
    tracked_adapters: list[_LifecycleTrackedQualificationAdapter] = []
    tracked_by_identity: dict[int, _LifecycleTrackedQualificationAdapter] = {}

    def tracked_factory(
        factory_instance: object,
        factory_mode: str,
    ) -> QualificationAdapter:
        delegate = raw_factory(factory_instance, factory_mode)
        identity = id(delegate)
        existing = tracked_by_identity.get(identity)
        if existing is not None:
            if existing.delegate is not delegate:
                raise RuntimeError("adapter identity collision during qualification")
            return existing
        tracked = _LifecycleTrackedQualificationAdapter(delegate)
        tracked_by_identity[identity] = tracked
        tracked_adapters.append(tracked)
        return tracked

    try:
        return _qualify_canonical_instance_impl(
            instance,
            mode,
            case_id=case_id,
            scale=scale,
            qualification_run_id=qualification_run_id,
            instance_sha256=instance_sha256,
            policy=policy,
            settings=settings,
            geometry_start_policy=geometry_start_policy,
            adapter_factory=tracked_factory,
            expected_model_contract=expected_model_contract,
            minimum_service_per_hour=minimum_service_per_hour,
            fail_fast_infrastructure=fail_fast_infrastructure,
            development_incumbent_reuse_pool=development_incumbent_reuse_pool,
            development_reuse_profile_id=development_reuse_profile_id,
            development_expected_target_constraint_signature=(
                development_expected_target_constraint_signature
            ),
        )
    finally:
        primary_error = sys.exception()
        # Fatal process exceptions deliberately propagate, but they must not
        # bypass native model disposal in a long-lived single-process run.
        cleanup_errors: list[BaseException] = []
        for tracked in reversed(tracked_adapters):
            if tracked.disposed:
                continue
            try:
                tracked.dispose()
            except BaseException as cleanup_error:
                cleanup_errors.append(cleanup_error)
        if cleanup_errors:
            if primary_error is not None:
                for cleanup_error in cleanup_errors:
                    _handle_cleanup_failure(
                        "qualification_lifecycle_disposal",
                        cleanup_error,
                        primary_error=primary_error,
                    )
            else:
                first_cleanup_error = cleanup_errors[0]
                for cleanup_error in cleanup_errors[1:]:
                    first_cleanup_error.add_note(
                        "additional lifecycle cleanup failure: "
                        + _cleanup_failure_message(
                            "qualification_lifecycle_disposal",
                            cleanup_error,
                        )
                    )
                _handle_cleanup_failure(
                    "qualification_lifecycle_disposal",
                    first_cleanup_error,
                    primary_error=None,
                )


class MLNDQualificationAdapter:
    """Thin, native-solver adapter around a fresh :class:`MLNDModel`."""

    def __init__(self, instance: object, mode: str):
        # Lazy imports keep pure qualification tests independent of Gurobi.
        import gurobipy as gp
        from gurobipy import GRB

        from src.model.gurobi_model import MLNDModel

        self._gp = gp
        self._GRB = GRB
        self._disposed = False
        self._mode = str(mode).upper()
        # Allocate the Python wrapper separately so it remains reachable for
        # explicit native-model cleanup even if MLNDModel.__init__ raises after
        # creating its gurobipy.Model.
        wrapper = MLNDModel.__new__(MLNDModel)
        try:
            MLNDModel.__init__(wrapper, instance, mode=mode)
            self.wrapper = wrapper.build()
            self.model = self.wrapper.model
            self.instance = instance
            self._temporary_constraint_token: TemporaryWitnessConstraintToken | None = (
                None
            )
            # Cache the fingerprint before qualification-only rows can be added.
            self._base_model_fingerprint = self._read_current_model_fingerprint()
            self._constraint_signature_cache: (
                QualificationConstraintSignature | None
            ) = None
        except BaseException as initialization_error:
            # Construction can fail after Gurobi has allocated a native model
            # but before the lifecycle-tracking factory receives this adapter.
            # Release that partial model in the narrow handoff window.
            partial_model = getattr(wrapper, "model", None)
            if partial_model is not None:
                try:
                    partial_model.dispose()
                except BaseException as cleanup_error:
                    _handle_cleanup_failure(
                        "partial_model_disposal",
                        cleanup_error,
                        primary_error=initialization_error,
                    )
            self._disposed = True
            raise

    def qualification_model_audit(self) -> QualificationModelAudit:
        """Read back the effective scenario from the constructed base model."""

        from src.config import to_dict

        demand_scale = self.instance.metadata.get("demand_scale")
        if isinstance(demand_scale, bool) or not isinstance(demand_scale, (int, float)):
            raise QualificationInfrastructureError(
                "loaded qualification instance has no numeric demand_scale"
            )
        config = self.instance.config
        return QualificationModelAudit(
            effective_model_config_sha256=canonical_sha256(to_dict(config)),
            demand_scale=float(demand_scale),
            locker_fixed_cost_per_hour=float(config.locker_fixed_cost_per_hour),
            display_origin_choices=config.display_origin_choices,
            general_corridor_choices=config.general_corridor_choices,
            screening_is_exact=bool(self.instance.screening_is_exact(self._mode)),
            model_fingerprint=self._base_model_fingerprint,
        )

    def courier_only_objective(self) -> float:
        return float(self.wrapper.courier_only_baseline)

    def total_demand_per_hour(self) -> float:
        return float(self.wrapper.inst.total_demand)

    def candidate_h_count(self) -> int:
        return len(self.wrapper.inst.H)

    def clear_mip_starts(self) -> None:
        self.model.NumStart = 0
        self.model.update()

    def qualification_constraint_signature(
        self,
    ) -> QualificationConstraintSignature:
        """Hash the complete supported feasibility system, excluding objective.

        The extraction is allowed only on the untouched base model.  General
        and native nonlinear constraints fail closed because this v1 extractor
        cannot yet serialize every one of their type-specific payloads.
        """

        if self._temporary_constraint_token is not None:
            raise QualificationInfrastructureError(
                "constraint signature must be computed before temporary rows"
            )
        cached = self._constraint_signature_cache
        if cached is not None:
            if self._constraint_counts() != (
                cached.linear_constraint_count,
                cached.quadratic_constraint_count,
                cached.unsupported_general_constraint_count,
            ):
                raise QualificationInfrastructureError(
                    "base constraint counts changed after signature extraction"
                )
            return cached

        try:
            signature = extract_gurobi_constraint_signature(self.model, self._GRB)
        except ConstraintSignatureExtractionError as error:
            raise QualificationInfrastructureError(
                f"exact qualification constraint extraction failed: {error}"
            ) from error
        self._constraint_signature_cache = signature
        return signature

    def install_constraint_equivalent_complete_incumbent(
        self,
        incumbent: CompleteIncumbent,
        expected_constraint_signature: QualificationConstraintSignature,
    ) -> None:
        """Install a cross-objective start after exact feasibility matching.

        Gurobi's ordinary model fingerprint includes objective coefficients, so
        it is intentionally not compared across profiles.  The complete
        objective-neutral constraint signature and ordered variable-name SHA
        replace that check.  The source fingerprint remains mandatory in the
        reuse envelope as provenance, but is expected to differ here.
        """

        if not isinstance(
            expected_constraint_signature, QualificationConstraintSignature
        ):
            raise TypeError(
                "expected_constraint_signature must be QualificationConstraintSignature"
            )
        observed_signature = self.qualification_constraint_signature()
        if observed_signature != expected_constraint_signature:
            raise ValueError(
                "target feasibility signature changed before start installation"
            )
        if incumbent.model_fingerprint is None:
            raise ValueError("reused complete incumbent lacks source fingerprint")
        variables = tuple(self.model.getVars())
        if len(variables) != incumbent.variable_count:
            raise ValueError(
                "reused complete incumbent variable count differs from target"
            )
        names = tuple(
            str(value)
            for value in self.model.getAttr(self._GRB.Attr.VarName, variables)
        )
        target_name_sha256 = _ordered_variable_name_sha256(names)
        if (
            target_name_sha256 != incumbent.ordered_var_name_sha256
            or target_name_sha256
            != expected_constraint_signature.ordered_var_name_sha256
        ):
            raise ValueError(
                "reused complete incumbent variable order differs from target"
            )
        self._reset_and_clear_all_starts()
        try:
            self.model.NumStart = 1
            self.model.update()
            self.model.Params.StartNumber = 0
            self.model.setAttr(
                self._GRB.Attr.Start,
                variables,
                incumbent.values,
            )
            self.model.update()
            observed_values = self.model.getAttr(self._GRB.Attr.Start, variables)
            if len(observed_values) != incumbent.variable_count:
                raise RuntimeError("reused complete-start readback length mismatch")
            for expected, observed in zip(
                incumbent.values,
                observed_values,
                strict=True,
            ):
                if not math.isclose(
                    float(expected),
                    float(observed),
                    rel_tol=0.0,
                    abs_tol=0.0,
                ):
                    raise RuntimeError("reused complete-start readback mismatch")
        except BaseException as installation_error:
            try:
                self._reset_and_clear_all_starts()
            except BaseException as cleanup_error:
                installation_error.add_note(
                    "reused complete-start rollback also failed: "
                    f"{type(cleanup_error).__name__}: {cleanup_error}"
                )
            raise

    def dispose(self) -> None:
        """Release the native witness model before a fresh copy is built."""

        if getattr(self, "_disposed", False):
            return
        self.model.dispose()
        # Drop the Python-side model wrapper and its million-scale Var objects
        # before constructing the fresh validation copy.
        self.wrapper = None
        self.model = None
        self._disposed = True
        gc.collect()

    def _apply_native_settings(self, settings: NativeQualificationSettings) -> None:
        params = self.model.Params
        params.TimeLimit = settings.time_limit_seconds
        params.MIPGap = settings.mip_gap
        params.Threads = 1
        params.OutputFlag = settings.output_flag
        params.Seed = settings.solver_seed
        params.Method = settings.method
        params.MIQCPMethod = settings.miqcp_method
        params.MIPFocus = settings.mip_focus
        params.Heuristics = settings.heuristics
        params.NoRelHeurTime = settings.no_rel_heur_time
        params.Presolve = settings.presolve
        params.Cuts = settings.native_cuts
        params.CutPasses = settings.root_cut_passes
        params.NumericFocus = settings.numeric_focus
        params.FeasibilityTol = settings.feasibility_tolerance
        params.OptimalityTol = settings.optimality_tolerance
        params.IntFeasTol = settings.integer_feasibility_tolerance
        params.PreCrush = 0
        params.LazyConstraints = 0

    def solve_native(self, settings: NativeQualificationSettings) -> str:
        self._apply_native_settings(settings)
        # No callback is passed: there are no user cuts, lazy constraints,
        # telemetry callbacks, or local-branching searches in qualification.
        self.model.optimize()
        return self._status_name(int(self.model.Status))

    def _status_name(self, status: int) -> str:
        status_names = (
            "LOADED",
            "OPTIMAL",
            "INFEASIBLE",
            "INF_OR_UNBD",
            "UNBOUNDED",
            "CUTOFF",
            "ITERATION_LIMIT",
            "NODE_LIMIT",
            "TIME_LIMIT",
            "SOLUTION_LIMIT",
            "INTERRUPTED",
            "NUMERIC",
            "SUBOPTIMAL",
            "INPROGRESS",
            "USER_OBJ_LIMIT",
            "WORK_LIMIT",
            "MEM_LIMIT",
        )
        names = {
            int(value): name
            for name in status_names
            if (value := getattr(self._GRB, name, None)) is not None
        }
        return names.get(status, f"STATUS_{status}")

    def _realized_service_expression(self) -> object:
        levels = self.wrapper.inst.config.availability_levels
        chi = self.wrapper.vars["chi"]
        return self._gp.quicksum(
            levels[key[4]] * chi[key] for key in self.wrapper.chi_keys
        )

    def add_witness_search_constraints(
        self,
        minimum_service_per_hour: float,
        maximum_objective: float,
    ) -> TemporaryWitnessConstraintToken:
        if self._temporary_constraint_token is not None:
            raise RuntimeError("temporary qualification constraints are already active")
        if (
            not math.isfinite(minimum_service_per_hour)
            or minimum_service_per_hour <= 0.0
        ):
            raise ValueError("minimum service must be positive and finite")
        if not math.isfinite(maximum_objective):
            raise ValueError("maximum objective must be finite")
        if int(self.model.ModelSense) != int(self._GRB.MINIMIZE):
            raise ValueError("qualification objective cutoff requires minimization")

        original_objective = self.model.getObjective().copy()
        original_model_sense = int(self.model.ModelSense)
        added: list[object] = []
        try:
            service_constraint = self.model.addConstr(
                self._realized_service_expression() >= minimum_service_per_hour,
                name="__qualification_minimum_realized_service",
            )
            added.append(service_constraint)
            objective_constraint = self.model.addConstr(
                original_objective <= maximum_objective,
                name="__qualification_maximum_objective",
            )
            added.append(objective_constraint)
            token = TemporaryWitnessConstraintToken(
                service_constraint=service_constraint,
                objective_constraint=objective_constraint,
                original_objective=original_objective,
                original_model_sense=original_model_sense,
            )
            # Service and economic improvement are now hard feasibility
            # conditions.  A constant objective stops the auxiliary MISOCP
            # from spending its bounded search budget optimizing among already
            # qualifying witnesses.  No D/E start, callback, or cut is used.
            self.model.setObjective(
                self._gp.LinExpr(0.0),
                original_model_sense,
            )
            self._temporary_constraint_token = token
            self.model.update()
        except BaseException as setup_error:
            # The runner has no token yet, so the adapter must roll back every
            # row and the objective change made before the failure.
            try:
                self._restore_witness_search_state(
                    original_objective=original_objective,
                    original_model_sense=original_model_sense,
                    constraints=added,
                )
            except BaseException as rollback_error:
                setup_error.add_note(
                    "qualification witness-search rollback also failed: "
                    f"{type(rollback_error).__name__}: {rollback_error}"
                )
            finally:
                self._temporary_constraint_token = None
            raise
        return token

    def _restore_witness_search_state(
        self,
        *,
        original_objective: object,
        original_model_sense: int,
        constraints: Sequence[object],
    ) -> None:
        """Best-effort restoration that attempts every transaction component."""

        first_error: BaseException | None = None
        try:
            self.model.setObjective(original_objective, original_model_sense)
        except BaseException as error:
            first_error = error
        try:
            if constraints:
                self.model.remove(list(constraints))
        except BaseException as error:
            if first_error is None:
                first_error = error
        try:
            self.model.update()
        except BaseException as error:
            if first_error is None:
                first_error = error
        if first_error is not None:
            raise first_error

    def remove_temporary_constraint(self, token: object) -> None:
        if token is not self._temporary_constraint_token:
            raise ValueError("temporary qualification constraint token mismatch")
        assert isinstance(token, TemporaryWitnessConstraintToken)
        self._restore_witness_search_state(
            original_objective=token.original_objective,
            original_model_sense=token.original_model_sense,
            constraints=token.constraints,
        )
        self._temporary_constraint_token = None

    def plan_geometry_start(
        self,
        *,
        scale: str,
        target_support_size: int,
        required_service_per_hour: float,
        active_h_tolerance_per_hour: float,
        policy: GeometryStartPolicy,
    ) -> GeometryStartPlan:
        if not policy.enabled:
            raise ValueError("geometry-start policy is disabled")
        return build_geometry_start_plan(
            self.wrapper,
            scale=scale,
            target_support_size=target_support_size,
            required_service_per_hour=required_service_per_hour,
            active_h_tolerance_per_hour=active_h_tolerance_per_hour,
            active_tolerance_multiplier=policy.active_tolerance_multiplier,
            required_service_multiplier=policy.required_service_multiplier,
        )

    def _constraint_counts(self) -> tuple[int, int, int]:
        return (
            int(self.model.NumConstrs),
            int(self.model.NumQConstrs),
            int(self.model.NumGenConstrs),
        )

    def _reset_and_clear_all_starts(self) -> int:
        """Discard incumbent/search state and leave every Start undefined."""

        self.model.reset(0)
        variables = self.model.getVars()
        existing_start_count = int(self.model.NumStart)
        if existing_start_count > 0 and variables:
            undefined = [self._GRB.UNDEFINED] * len(variables)
            for start_number in range(existing_start_count):
                self.model.Params.StartNumber = start_number
                self.model.setAttr(self._GRB.Attr.Start, variables, undefined)
        self.model.NumStart = 0
        self.model.update()
        return len(variables)

    def install_geometry_start(
        self,
        plan: GeometryStartPlan,
    ) -> GeometryStartInstallAudit:
        """Install and bulk-verify one positive-only partial MIP start.

        Gurobi's native ``Fingerprint`` intentionally changes when a MIP Start
        is present.  The mathematical-model safety check therefore compares
        two *start-neutral* fingerprints: one immediately after the initial
        clear and one after temporarily clearing the installed-and-verified
        start.  The unique start is then reinstalled and verified again.
        """

        if self._temporary_constraint_token is None:
            raise RuntimeError(
                "geometry start is allowed only during temporary feasibility search"
            )
        variable_count = self._reset_and_clear_all_starts()
        fingerprint_before = self._read_current_model_fingerprint()
        constraints_before = self._constraint_counts()
        variables = self.model.getVars()
        if len(variables) != variable_count:
            raise RuntimeError("model variable count changed while clearing starts")

        values_by_identity: dict[int, tuple[object, float, str]] = {}

        def retain(variable: object, value: float, family: str) -> None:
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError("geometry start values must be positive and finite")
            identity = id(variable)
            prior = values_by_identity.get(identity)
            if prior is not None and not math.isclose(
                prior[1], value, rel_tol=1e-12, abs_tol=1e-12
            ):
                raise ValueError("geometry start assigns conflicting variable values")
            values_by_identity[identity] = (variable, value, family)

        x = self.wrapper.vars["x"]
        y = self.wrapper.vars["y"]
        z = self.wrapper.vars["z"]
        psi = self.wrapper.vars["psi"]
        varphi = self.wrapper.vars["varphi"]
        for node in plan.open_nodes:
            retain(x[node], 1.0, "x")
        for arc in plan.activated_y_arcs:
            retain(y[arc], 1.0, "y")
        for choice in plan.capacity_choices:
            retain(z[choice.locker, choice.capacity_index], 1.0, "z")
        for u, v, flow in plan.physical_arc_flows:
            retain(psi[u, v], flow, "psi")
        for u, v, option, flow in plan.route_flows:
            retain(varphi[u, v, option], flow, "varphi")

        model_variable_ids = {id(variable) for variable in variables}
        if not set(values_by_identity).issubset(model_variable_ids):
            raise ValueError("geometry plan refers to a variable outside the model")
        observed_family_counts: dict[str, int] = {}
        for _variable, _value, family in values_by_identity.values():
            observed_family_counts[family] = observed_family_counts.get(family, 0) + 1
        expected_family_counts = dict(plan.family_start_counts)
        if observed_family_counts != expected_family_counts:
            raise ValueError(
                "geometry start family counts do not match the audited plan"
            )

        start_values = [
            values_by_identity.get(id(variable), (None, self._GRB.UNDEFINED, ""))[1]
            for variable in variables
        ]
        fingerprint_with_start: int | None = None

        def install_and_verify() -> None:
            self.model.NumStart = 1
            self.model.update()
            self.model.Params.StartNumber = 0
            self.model.setAttr(self._GRB.Attr.Start, variables, start_values)
            self.model.update()
            if int(self.model.NumStart) != 1:
                raise RuntimeError("geometry start did not leave exactly one MIP start")
            self.model.Params.StartNumber = 0
            observed_values = self.model.getAttr(self._GRB.Attr.Start, variables)
            if len(observed_values) != len(start_values):
                raise RuntimeError("geometry start bulk readback length mismatch")
            for expected, observed in zip(
                start_values,
                observed_values,
                strict=True,
            ):
                expected_value = float(expected)
                observed_value = float(observed)
                if expected_value == float(self._GRB.UNDEFINED):
                    matches = observed_value == expected_value
                else:
                    matches = math.isclose(
                        observed_value,
                        expected_value,
                        rel_tol=1e-12,
                        abs_tol=1e-12,
                    )
                if not matches:
                    raise RuntimeError("geometry start bulk readback mismatch")

        try:
            install_and_verify()
            fingerprint_with_start = self._read_current_model_fingerprint()
            neutral_variable_count = self._reset_and_clear_all_starts()
            if neutral_variable_count != variable_count:
                raise RuntimeError(
                    "model variable count changed during geometry fingerprint audit"
                )
            constraints_after = self._constraint_counts()
            fingerprint_after = self._read_current_model_fingerprint()
            install_and_verify()
        except BaseException as install_error:
            try:
                self._reset_and_clear_all_starts()
            except BaseException as cleanup_error:
                install_error.add_note(
                    "geometry start installation rollback also failed: "
                    f"{type(cleanup_error).__name__}: {cleanup_error}"
                )
            raise
        audit = GeometryStartInstallAudit(
            reset_performed=True,
            undefined_variable_count=variable_count - len(values_by_identity),
            family_start_counts=tuple(sorted(observed_family_counts.items())),
            fingerprint_before=fingerprint_before,
            fingerprint_with_start=fingerprint_with_start,
            fingerprint_after=fingerprint_after,
            fingerprint_unchanged=(
                fingerprint_before is not None
                and fingerprint_after is not None
                and fingerprint_before == fingerprint_after
            ),
            constraint_counts_before=constraints_before,
            constraint_counts_after=constraints_after,
            hard_constraints_added=constraints_before != constraints_after,
            bounds_modified=False,
            objective_modified=False,
        )
        return audit

    def solve_geometry_start(
        self,
        settings: NativeQualificationSettings,
        policy: GeometryStartPolicy,
    ) -> str:
        if not policy.enabled:
            raise ValueError("geometry-start policy is disabled")
        previous_start_node_limit = int(self.model.Params.StartNodeLimit)
        self._apply_native_settings(settings)
        self.model.Params.TimeLimit = policy.per_attempt_seconds
        self.model.Params.StartNodeLimit = policy.start_node_limit
        try:
            self.model.optimize()
            return self._status_name(int(self.model.Status))
        finally:
            # The witness model is normally discarded, but restoring the
            # fallback-only parameter keeps cleanup transactional and makes
            # accidental reuse auditable.  Fresh validation uses a new model.
            primary_error = sys.exception()
            try:
                self.model.Params.StartNodeLimit = previous_start_node_limit
            except BaseException as cleanup_error:
                _handle_cleanup_failure(
                    "geometry_start_node_limit_restore",
                    cleanup_error,
                    primary_error=primary_error,
                )

    def clear_geometry_start(self) -> None:
        self._reset_and_clear_all_starts()

    def _original_objective_value(self) -> float:
        """Read the paper objective even while the auxiliary objective is zero."""

        token = getattr(self, "_temporary_constraint_token", None)
        if token is not None:
            return float(token.original_objective.getValue())
        return float(self.model.ObjVal)

    def _active_h_threshold_per_hour(self) -> float:
        return service_tolerance(self.total_demand_per_hour())

    def _active_h_count(self, activity_tolerance: float | None = None) -> int:
        """Count open lockers with material incident flow on physical arcs.

        ``A_F`` excludes the virtual H--state arcs in ``A_HS``/``A_SH``.  The
        strict activity cutoff is the pre-registered service tolerance
        ``max(1e-3, 1e-5 * total_demand)`` in orders/h.
        """

        if activity_tolerance is None:
            activity_tolerance = self._active_h_threshold_per_hour()

        lockers = tuple(self.wrapper.inst.H)
        locker_set = set(lockers)
        incident_flows = {locker: [] for locker in lockers}
        psi = self.wrapper.vars["psi"]
        for u, v in self.wrapper.inst.A_F:
            physical_flow = max(0.0, float(psi[u, v].X))
            if u in locker_set:
                incident_flows[u].append(physical_flow)
            if v in locker_set:
                incident_flows[v].append(physical_flow)

        x = self.wrapper.vars["x"]
        return sum(
            float(x[locker].X) > 0.5
            and math.fsum(incident_flows[locker]) > activity_tolerance
            for locker in lockers
        )

    def extract_solution(self) -> QualificationSolution:
        status = self._status_name(int(self.model.Status))
        if int(self.model.SolCount) <= 0:
            return QualificationSolution(
                status=status,
                has_incumbent=False,
                candidate_h=len(self.wrapper.inst.H),
                active_h_threshold_per_hour=self._active_h_threshold_per_hour(),
            )
        chi = self.wrapper.vars["chi"]
        levels = self.wrapper.inst.config.availability_levels
        realized = math.fsum(
            levels[key[4]] * float(chi[key].X) for key in self.wrapper.chi_keys
        )
        x = self.wrapper.vars["x"]
        open_nodes = [node for node, variable in x.items() if float(variable.X) > 0.5]
        positive_cost = sum(
            self.wrapper.inst.facility_cost(node) > 0.0 for node in open_nodes
        )
        open_h = sum(float(x[locker].X) > 0.5 for locker in self.wrapper.inst.H)
        active_h_threshold_per_hour = self._active_h_threshold_per_hour()
        active_h = self._active_h_count(active_h_threshold_per_hour)
        return QualificationSolution(
            status=status,
            has_incumbent=True,
            objective=self._original_objective_value(),
            realized_service_per_hour=realized,
            open_facilities=len(open_nodes),
            positive_cost_open_facilities=positive_cost,
            candidate_h=len(self.wrapper.inst.H),
            open_h=open_h,
            active_h=active_h,
            active_h_threshold_per_hour=active_h_threshold_per_hour,
        )

    def _read_current_model_fingerprint(self) -> int | None:
        try:
            value = self.model.Fingerprint
        except MemoryError:
            raise
        except Exception:
            fingerprint_attr = getattr(
                getattr(self._GRB, "Attr", object()),
                "Fingerprint",
                None,
            )
            if fingerprint_attr is None:
                return None
            try:
                value = self.model.getAttr(fingerprint_attr)
            except MemoryError:
                raise
            except Exception:
                return None
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError) as error:
            raise ValueError("model Fingerprint must be an integer") from error

    def _transfer_model_fingerprint(self) -> int | None:
        if hasattr(self, "_base_model_fingerprint"):
            return self._base_model_fingerprint
        return self._read_current_model_fingerprint()

    def capture_complete_incumbent(self) -> CompleteIncumbent:
        if int(self.model.SolCount) <= 0:
            raise ValueError("cannot capture a complete start without an incumbent")
        variables = self.model.getVars()
        variable_count = len(variables)
        if variable_count < 1:
            raise ValueError("cannot capture a complete start from an empty model")
        names = self.model.getAttr(self._GRB.Attr.VarName, variables)
        values = self.model.getAttr(self._GRB.Attr.X, variables)
        if len(names) != variable_count or len(values) != variable_count:
            raise ValueError("bulk model attributes do not cover every variable")
        model_fingerprint = self._transfer_model_fingerprint()
        if model_fingerprint is None:
            raise ValueError(
                "model Fingerprint is unavailable; refusing complete-start capture"
            )
        return CompleteIncumbent.from_ordered_values(
            objective=self._original_objective_value(),
            variable_names=names,
            values=values,
            model_fingerprint=model_fingerprint,
        )

    def install_complete_incumbent(self, incumbent: CompleteIncumbent) -> None:
        variables = self.model.getVars()
        if len(variables) != incumbent.variable_count:
            raise ValueError(
                "complete incumbent variable count does not match fresh model "
                f"({incumbent.variable_count} != {len(variables)})"
            )
        names = self.model.getAttr(self._GRB.Attr.VarName, variables)
        if len(names) != incumbent.variable_count:
            raise ValueError("bulk variable names do not cover every fresh variable")
        fresh_name_sha256 = _ordered_variable_name_sha256(names)
        if fresh_name_sha256 != incumbent.ordered_var_name_sha256:
            raise ValueError(
                "complete incumbent ordered variable-name SHA256 does not match "
                "fresh model"
            )
        fresh_fingerprint = self._transfer_model_fingerprint()
        if incumbent.model_fingerprint is None or fresh_fingerprint is None:
            raise ValueError(
                "model Fingerprint is unavailable; refusing complete-start install"
            )
        if fresh_fingerprint != incumbent.model_fingerprint:
            raise ValueError(
                "complete incumbent model Fingerprint does not match fresh model "
                f"({incumbent.model_fingerprint!r} != {fresh_fingerprint!r})"
            )
        self.model.NumStart = 1
        self.model.update()
        self.model.Params.StartNumber = 0
        self.model.setAttr(
            self._GRB.Attr.Start,
            variables,
            incumbent.values,
        )
        self.model.update()


__all__ = [
    "CompleteIncumbent",
    "MLNDQualificationAdapter",
    "NORMAL_QUALIFICATION_RUNNER_REASONS",
    "NORMAL_QUALIFICATION_SOLVER_STATUSES",
    "NativeQualificationSettings",
    "QualificationAdapter",
    "QualificationInfrastructureError",
    "QualificationModelAudit",
    "QualificationModelContract",
    "QualificationPhaseTimings",
    "QualificationRunnerOutcome",
    "QualificationRunnerReason",
    "QualificationSolution",
    "TemporaryWitnessConstraintToken",
    "qualify_canonical_instance",
]
