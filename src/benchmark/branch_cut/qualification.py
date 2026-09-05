"""Pure qualification rules for non-trivial MLND benchmark instances.

Qualification is deliberately independent of Gurobi.  A solver adapter may
populate :class:`QualificationEvidence`, but the accept/reject decision is a
deterministic function of that evidence and a frozen policy.  In particular,
the temporary positive-service constraint sometimes used to find a witness is
not represented here: ``witness_feasible`` must describe the witness after it
has been checked in the original, unmodified model.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isclose, isfinite
import re
from typing import Any


SCALES = ("small", "medium", "large")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _validate_optional_model_identity(
    *,
    qualification_case_sha256: str | None,
    effective_model_config_sha256: str | None,
    screening_is_exact: bool | None,
) -> None:
    """Require either one complete qualification identity or legacy absence."""

    values = (
        qualification_case_sha256,
        effective_model_config_sha256,
        screening_is_exact,
    )
    if all(value is None for value in values):
        return
    if any(value is None for value in values):
        raise ValueError(
            "qualification model identity fields must be all present or all absent"
        )
    if _SHA256_RE.fullmatch(qualification_case_sha256) is None:
        raise ValueError(
            "qualification_case_sha256 must be a lowercase 64-character SHA256 digest"
        )
    if _SHA256_RE.fullmatch(effective_model_config_sha256) is None:
        raise ValueError(
            "effective_model_config_sha256 must be a lowercase 64-character "
            "SHA256 digest"
        )
    if type(screening_is_exact) is not bool:
        raise ValueError("screening_is_exact must be boolean")


class QualificationReason:
    """Stable machine-readable rejection reason codes."""

    WITNESS_NOT_FEASIBLE = "witness_not_feasible_in_original_model"
    NONFINITE_COURIER_OBJECTIVE = "nonfinite_courier_only_objective"
    NONPOSITIVE_COURIER_OBJECTIVE = "nonpositive_courier_only_objective"
    NONFINITE_WITNESS_OBJECTIVE = "nonfinite_witness_objective"
    NONFINITE_TOTAL_DEMAND = "nonfinite_total_demand"
    NONPOSITIVE_TOTAL_DEMAND = "nonpositive_total_demand"
    NONFINITE_SERVICE = "nonfinite_witness_service"
    NEGATIVE_SERVICE = "negative_witness_service"
    SERVICE_EXCEEDS_DEMAND = "witness_service_exceeds_total_demand"
    SERVICE_BELOW_TOLERANCE = "service_below_tolerance"
    SERVICE_FRACTION_BELOW_MINIMUM = "service_fraction_below_minimum"
    OBJECTIVE_IMPROVEMENT_BELOW_MARGIN = "objective_improvement_below_margin"
    SAVINGS_FRACTION_BELOW_MINIMUM = "savings_fraction_below_minimum"
    NO_OPEN_FACILITY = "no_open_facility"
    NO_POSITIVE_COST_OPEN_FACILITY = "no_positive_cost_open_facility"
    POSITIVE_COST_OPEN_COUNT_EXCEEDS_TOTAL = (
        "positive_cost_open_count_exceeds_total_open_count"
    )
    NO_H_CANDIDATES = "no_h_candidates"
    INVALID_OPEN_H_COUNT = "invalid_open_h_count"
    MISSING_ACTIVE_H_COUNT = "missing_active_h_count"
    NO_ACTIVE_H = "no_active_h_with_physical_service"
    INVALID_ACTIVE_H_COUNT = "invalid_active_h_count"
    MISSING_ACTIVE_H_THRESHOLD = "missing_active_h_threshold_per_hour"
    INVALID_ACTIVE_H_THRESHOLD = "invalid_active_h_threshold_per_hour"
    ACTIVE_H_THRESHOLD_MISMATCH = "active_h_threshold_per_hour_mismatch"
    H_OPEN_FRACTION_BELOW_MINIMUM = "h_open_fraction_below_minimum"
    H_OPEN_FRACTION_ABOVE_MAXIMUM = "h_open_fraction_above_maximum"
    # Retained for reading legacy ledgers; new decisions emit active-H reasons.
    LARGE_OPEN_H_BELOW_TARGET = "large_scale_open_h_below_target"
    LARGE_OPEN_H_ABOVE_TARGET = "large_scale_open_h_above_target"
    LARGE_ACTIVE_H_BELOW_TARGET = "large_scale_active_h_below_target"
    LARGE_ACTIVE_H_ABOVE_TARGET = "large_scale_active_h_above_target"


def objective_margin(
    courier_only_objective: float,
    optimality_tolerance: float,
) -> float:
    """Return the absolute improvement needed to beat the courier baseline.

    The formula is fixed by the experimental protocol and expressed in CNY/h.
    """

    if not isfinite(courier_only_objective):
        raise ValueError("courier_only_objective must be finite")
    if not isfinite(optimality_tolerance) or optimality_tolerance < 0.0:
        raise ValueError("optimality_tolerance must be finite and non-negative")
    magnitude = max(1.0, abs(courier_only_objective))
    return max(
        1.0,
        1e-4 * abs(courier_only_objective),
        100.0 * optimality_tolerance * magnitude,
    )


def service_tolerance(total_demand_per_hour: float) -> float:
    """Return the minimum numerically meaningful service rate in orders/h."""

    if not isfinite(total_demand_per_hour) or total_demand_per_hour < 0.0:
        raise ValueError("total_demand_per_hour must be finite and non-negative")
    return max(1e-3, 1e-5 * total_demand_per_hour)


@dataclass(frozen=True, slots=True)
class QualificationPolicy:
    """Pre-registered thresholds for deciding whether an instance is useful."""

    optimality_tolerance: float = 1e-6
    minimum_savings_fraction: float = 1e-3
    minimum_service_fraction: float = 1e-3
    minimum_h_open_fraction: float = 0.15
    maximum_h_open_fraction: float = 0.85
    # Kept under its original artifact key for compatibility.  The count target
    # is applied to ``active_h``; ``open_h`` still drives only the interior-open
    # fraction and all-open/all-closed checks.
    large_scale_open_h_target: tuple[int, int] | None = None

    def __post_init__(self) -> None:
        if not isfinite(self.optimality_tolerance) or self.optimality_tolerance < 0.0:
            raise ValueError("optimality_tolerance must be finite and non-negative")
        for name, value in (
            ("minimum_savings_fraction", self.minimum_savings_fraction),
            ("minimum_service_fraction", self.minimum_service_fraction),
            ("minimum_h_open_fraction", self.minimum_h_open_fraction),
            ("maximum_h_open_fraction", self.maximum_h_open_fraction),
        ):
            if not isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.minimum_savings_fraction < 0.0:
            raise ValueError("minimum_savings_fraction must be non-negative")
        if self.minimum_service_fraction < 0.0:
            raise ValueError("minimum_service_fraction must be non-negative")
        if (
            not 0.0
            <= self.minimum_h_open_fraction
            < self.maximum_h_open_fraction
            <= 1.0
        ):
            raise ValueError(
                "H-open fraction bounds must satisfy 0 <= minimum < maximum <= 1"
            )
        if self.large_scale_open_h_target is not None:
            lower, upper = self.large_scale_open_h_target
            if lower < 1 or upper < lower:
                raise ValueError(
                    "large_scale_open_h_target must contain positive ordered bounds"
                )

    def to_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        target = self.large_scale_open_h_target
        payload["large_scale_open_h_target"] = list(target) if target else None
        return payload


@dataclass(frozen=True, slots=True)
class QualificationEvidence:
    """Facts obtained before either benchmark method is compared.

    ``positive_cost_open_facilities`` counts every open facility whose fixed
    opening cost is strictly positive.  ``open_h`` and ``candidate_h`` refer to
    the H (locker/display) site family used for the interior-opening check.
    ``active_h`` counts open H sites with incident physical order throughput
    above ``active_h_threshold_per_hour``.  That audit threshold must equal
    :func:`service_tolerance` applied to total demand.  The count excludes a
    bare ``x=1`` decision and flow on virtual reliability/state arcs.
    """

    case_id: str
    scale: str
    courier_only_objective: float
    witness_objective: float
    total_demand_per_hour: float
    witness_service_per_hour: float
    open_facilities: int
    positive_cost_open_facilities: int
    candidate_h: int
    open_h: int
    witness_feasible: bool
    qualification_run_id: str
    instance_sha256: str
    active_h: int | None = None
    active_h_threshold_per_hour: float | None = None
    qualification_case_sha256: str | None = None
    effective_model_config_sha256: str | None = None
    screening_is_exact: bool | None = None

    def __post_init__(self) -> None:
        if not self.case_id:
            raise ValueError("case_id must be non-empty")
        if self.scale not in SCALES:
            raise ValueError(f"scale must be one of {SCALES}")
        if not self.qualification_run_id:
            raise ValueError("qualification_run_id must be non-empty")
        if _SHA256_RE.fullmatch(self.instance_sha256) is None:
            raise ValueError(
                "instance_sha256 must be a lowercase 64-character SHA256 digest"
            )
        _validate_optional_model_identity(
            qualification_case_sha256=self.qualification_case_sha256,
            effective_model_config_sha256=self.effective_model_config_sha256,
            screening_is_exact=self.screening_is_exact,
        )

    def to_payload(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class QualificationDecision:
    """Complete, reproducible qualification decision for one candidate."""

    case_id: str
    scale: str
    accepted: bool
    rejection_reasons: tuple[str, ...]
    qualification_run_id: str
    instance_sha256: str
    courier_only_objective: float | None
    witness_objective: float | None
    objective_improvement: float | None
    objective_margin: float | None
    savings_fraction: float | None
    total_demand_per_hour: float | None
    witness_service_per_hour: float | None
    service_tolerance: float | None
    service_fraction: float | None
    open_facilities: int
    positive_cost_open_facilities: int
    candidate_h: int
    open_h: int
    h_open_fraction: float | None
    policy: QualificationPolicy
    active_h: int | None = None
    active_h_threshold_per_hour: float | None = None
    qualification_case_sha256: str | None = None
    effective_model_config_sha256: str | None = None
    screening_is_exact: bool | None = None

    def __post_init__(self) -> None:
        if not self.case_id:
            raise ValueError("case_id must be non-empty")
        if self.scale not in SCALES:
            raise ValueError(f"scale must be one of {SCALES}")
        if not self.qualification_run_id:
            raise ValueError("qualification_run_id must be non-empty")
        if _SHA256_RE.fullmatch(self.instance_sha256) is None:
            raise ValueError(
                "instance_sha256 must be a lowercase 64-character SHA256 digest"
            )
        if self.accepted == bool(self.rejection_reasons):
            raise ValueError(
                "accepted must be true exactly when rejection_reasons is empty"
            )
        _validate_optional_model_identity(
            qualification_case_sha256=self.qualification_case_sha256,
            effective_model_config_sha256=self.effective_model_config_sha256,
            screening_is_exact=self.screening_is_exact,
        )

    def to_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["rejection_reasons"] = list(self.rejection_reasons)
        payload["policy"] = self.policy.to_payload()
        return payload


def _finite_or_none(value: float) -> float | None:
    return float(value) if isfinite(value) else None


def assess_qualification(
    evidence: QualificationEvidence,
    policy: QualificationPolicy | None = None,
) -> QualificationDecision:
    """Assess all rules and retain every applicable rejection reason."""

    active_policy = policy or QualificationPolicy()
    reasons: list[str] = []

    c0_is_finite = isfinite(evidence.courier_only_objective)
    witness_objective_is_finite = isfinite(evidence.witness_objective)
    demand_is_finite = isfinite(evidence.total_demand_per_hour)
    service_is_finite = isfinite(evidence.witness_service_per_hour)

    if not evidence.witness_feasible:
        reasons.append(QualificationReason.WITNESS_NOT_FEASIBLE)
    if not c0_is_finite:
        reasons.append(QualificationReason.NONFINITE_COURIER_OBJECTIVE)
    elif evidence.courier_only_objective <= 0.0:
        reasons.append(QualificationReason.NONPOSITIVE_COURIER_OBJECTIVE)
    if not witness_objective_is_finite:
        reasons.append(QualificationReason.NONFINITE_WITNESS_OBJECTIVE)
    if not demand_is_finite:
        reasons.append(QualificationReason.NONFINITE_TOTAL_DEMAND)
    elif evidence.total_demand_per_hour <= 0.0:
        reasons.append(QualificationReason.NONPOSITIVE_TOTAL_DEMAND)
    if not service_is_finite:
        reasons.append(QualificationReason.NONFINITE_SERVICE)
    elif evidence.witness_service_per_hour < 0.0:
        reasons.append(QualificationReason.NEGATIVE_SERVICE)
    if (
        demand_is_finite
        and evidence.total_demand_per_hour >= 0.0
        and service_is_finite
        and evidence.witness_service_per_hour
        > evidence.total_demand_per_hour
        + service_tolerance(evidence.total_demand_per_hour)
    ):
        reasons.append(QualificationReason.SERVICE_EXCEEDS_DEMAND)

    margin: float | None = None
    improvement: float | None = None
    savings_fraction: float | None = None
    if c0_is_finite:
        margin = objective_margin(
            evidence.courier_only_objective,
            active_policy.optimality_tolerance,
        )
    if c0_is_finite and witness_objective_is_finite:
        improvement = evidence.courier_only_objective - evidence.witness_objective
        if improvement < (margin or 0.0):
            reasons.append(QualificationReason.OBJECTIVE_IMPROVEMENT_BELOW_MARGIN)
        if evidence.courier_only_objective > 0.0:
            savings_fraction = improvement / evidence.courier_only_objective
            if savings_fraction < active_policy.minimum_savings_fraction:
                reasons.append(QualificationReason.SAVINGS_FRACTION_BELOW_MINIMUM)

    service_tol: float | None = None
    service_fraction: float | None = None
    if demand_is_finite and evidence.total_demand_per_hour >= 0.0:
        service_tol = service_tolerance(evidence.total_demand_per_hour)
    if service_is_finite and service_tol is not None:
        if evidence.witness_service_per_hour < service_tol:
            reasons.append(QualificationReason.SERVICE_BELOW_TOLERANCE)
        if evidence.total_demand_per_hour > 0.0:
            service_fraction = (
                evidence.witness_service_per_hour / evidence.total_demand_per_hour
            )
            if service_fraction < active_policy.minimum_service_fraction:
                reasons.append(QualificationReason.SERVICE_FRACTION_BELOW_MINIMUM)

    if evidence.open_facilities <= 0:
        reasons.append(QualificationReason.NO_OPEN_FACILITY)
    if evidence.positive_cost_open_facilities <= 0:
        reasons.append(QualificationReason.NO_POSITIVE_COST_OPEN_FACILITY)
    if evidence.positive_cost_open_facilities > max(0, evidence.open_facilities):
        reasons.append(QualificationReason.POSITIVE_COST_OPEN_COUNT_EXCEEDS_TOTAL)

    h_open_fraction: float | None = None
    if evidence.candidate_h <= 0:
        reasons.append(QualificationReason.NO_H_CANDIDATES)
    elif evidence.open_h < 0 or evidence.open_h > evidence.candidate_h:
        reasons.append(QualificationReason.INVALID_OPEN_H_COUNT)
    else:
        h_open_fraction = evidence.open_h / evidence.candidate_h
        if h_open_fraction < active_policy.minimum_h_open_fraction:
            reasons.append(QualificationReason.H_OPEN_FRACTION_BELOW_MINIMUM)
        if h_open_fraction > active_policy.maximum_h_open_fraction:
            reasons.append(QualificationReason.H_OPEN_FRACTION_ABOVE_MAXIMUM)

    if evidence.active_h is None:
        reasons.append(QualificationReason.MISSING_ACTIVE_H_COUNT)
    elif evidence.active_h < 0 or evidence.active_h > evidence.open_h:
        reasons.append(QualificationReason.INVALID_ACTIVE_H_COUNT)
    elif evidence.active_h == 0:
        reasons.append(QualificationReason.NO_ACTIVE_H)

    expected_active_h_threshold: float | None = None
    if demand_is_finite and evidence.total_demand_per_hour >= 0.0:
        expected_active_h_threshold = service_tolerance(evidence.total_demand_per_hour)
    if evidence.active_h_threshold_per_hour is None:
        reasons.append(QualificationReason.MISSING_ACTIVE_H_THRESHOLD)
    else:
        if (
            not isfinite(evidence.active_h_threshold_per_hour)
            or evidence.active_h_threshold_per_hour <= 0.0
        ):
            reasons.append(QualificationReason.INVALID_ACTIVE_H_THRESHOLD)
        elif expected_active_h_threshold is not None and not isclose(
            evidence.active_h_threshold_per_hour,
            expected_active_h_threshold,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            reasons.append(QualificationReason.ACTIVE_H_THRESHOLD_MISMATCH)

    if (
        evidence.scale == "large"
        and active_policy.large_scale_open_h_target is not None
    ):
        target_lower, target_upper = active_policy.large_scale_open_h_target
        if evidence.active_h is not None:
            if evidence.active_h < target_lower:
                reasons.append(QualificationReason.LARGE_ACTIVE_H_BELOW_TARGET)
            if evidence.active_h > target_upper:
                reasons.append(QualificationReason.LARGE_ACTIVE_H_ABOVE_TARGET)

    return QualificationDecision(
        case_id=evidence.case_id,
        scale=evidence.scale,
        accepted=not reasons,
        rejection_reasons=tuple(reasons),
        qualification_run_id=evidence.qualification_run_id,
        instance_sha256=evidence.instance_sha256,
        courier_only_objective=_finite_or_none(evidence.courier_only_objective),
        witness_objective=_finite_or_none(evidence.witness_objective),
        objective_improvement=_finite_or_none(improvement)
        if improvement is not None
        else None,
        objective_margin=margin,
        savings_fraction=(
            _finite_or_none(savings_fraction) if savings_fraction is not None else None
        ),
        total_demand_per_hour=_finite_or_none(evidence.total_demand_per_hour),
        witness_service_per_hour=_finite_or_none(evidence.witness_service_per_hour),
        service_tolerance=service_tol,
        service_fraction=(
            _finite_or_none(service_fraction) if service_fraction is not None else None
        ),
        open_facilities=evidence.open_facilities,
        positive_cost_open_facilities=evidence.positive_cost_open_facilities,
        candidate_h=evidence.candidate_h,
        open_h=evidence.open_h,
        h_open_fraction=h_open_fraction,
        policy=active_policy,
        active_h=evidence.active_h,
        active_h_threshold_per_hour=(
            _finite_or_none(evidence.active_h_threshold_per_hour)
            if evidence.active_h_threshold_per_hour is not None
            else None
        ),
        qualification_case_sha256=evidence.qualification_case_sha256,
        effective_model_config_sha256=evidence.effective_model_config_sha256,
        screening_is_exact=evidence.screening_is_exact,
    )
