"""Formal incumbent structure and qualification-trace evidence.

This module deliberately has no Gurobi import.  It reads the small public
surface exposed by the built model wrapper, independently recomputes the
paper-relevant solution structure, and validates its own canonical payload.
Formal orchestration can therefore bind the same frozen qualification decision
to an attempt, its run artifacts, and its immutable slot checkpoint without
expanding the general solver-metric schema.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .hashing import canonical_json, canonical_sha256
from .qualification import (
    QualificationDecision,
    QualificationEvidence,
    QualificationPolicy,
    QualificationReason,
    assess_qualification,
    service_tolerance,
)


FORMAL_QUALIFICATION_TRACE_SCHEMA_VERSION = 1
FORMAL_QUALIFICATION_TRACE_ARTIFACT_TYPE = (
    "mlnd_branch_cut_formal_qualification_trace"
)
FORMAL_SOLUTION_STRUCTURE_SCHEMA_VERSION = 1
FORMAL_SOLUTION_STRUCTURE_SOURCE = "actual_incumbent_variables"
FORMAL_NO_INCUMBENT_SOURCE = "no_incumbent"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_QUALIFICATION_DECISION_FIELDS = frozenset(
    {
        "case_id",
        "scale",
        "accepted",
        "rejection_reasons",
        "qualification_run_id",
        "instance_sha256",
        "courier_only_objective",
        "witness_objective",
        "objective_improvement",
        "objective_margin",
        "savings_fraction",
        "total_demand_per_hour",
        "witness_service_per_hour",
        "service_tolerance",
        "service_fraction",
        "open_facilities",
        "positive_cost_open_facilities",
        "candidate_h",
        "open_h",
        "h_open_fraction",
        "policy",
        "active_h",
        "active_h_threshold_per_hour",
        "qualification_case_sha256",
        "effective_model_config_sha256",
        "screening_is_exact",
    }
)
_QUALIFICATION_POLICY_FIELDS = frozenset(QualificationPolicy().to_payload())
_SOLUTION_STRUCTURE_FIELDS = frozenset(
    {
        "schema_version",
        "source",
        "has_incumbent",
        "qualification_trace_sha256",
        "C0",
        "incumbent_objective",
        "total_demand_per_hour",
        "realized_service_per_hour",
        "realized_service_rate",
        "open_facility_count",
        "positive_cost_open_facility_count",
        "candidate_H_count",
        "open_H_count",
        "active_H_count",
        "active_H_threshold_per_hour",
        "nontriviality_evaluated",
        "nontriviality_passed",
        "nontriviality_criteria",
        "nontriviality_rejection_reasons",
        "nontriviality_decision",
        "nontriviality_decision_sha256",
    }
)
_NONTRIVIALITY_CRITERIA_FIELDS = (
    "objective_improvement_margin_met",
    "minimum_savings_fraction_met",
    "minimum_service_tolerance_met",
    "minimum_service_fraction_met",
    "positive_cost_facility_open",
    "H_open_fraction_interior",
    "active_H_present",
    "large_active_H_target_met",
    "all_criteria_met",
)


def _require_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA256 digest")
    return value


def _require_nonempty(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonempty string")
    return value


def _policy_from_payload(payload: object) -> QualificationPolicy:
    if not isinstance(payload, Mapping):
        raise ValueError("qualification policy must be an object")
    if set(payload) != _QUALIFICATION_POLICY_FIELDS:
        raise ValueError("qualification policy fields are not canonical")
    values = dict(payload)
    target = values.get("large_scale_open_h_target")
    if target is not None:
        if not isinstance(target, (list, tuple)) or len(target) != 2:
            raise ValueError("large_scale_open_h_target must contain two integers")
        values["large_scale_open_h_target"] = tuple(target)
    return QualificationPolicy(**values)


def _decision_from_payload(payload: object) -> QualificationDecision:
    if not isinstance(payload, Mapping):
        raise ValueError("qualification decision must be an object")
    if set(payload) != _QUALIFICATION_DECISION_FIELDS:
        raise ValueError("qualification decision fields are not canonical")
    values = dict(payload)
    values["policy"] = _policy_from_payload(values["policy"])
    reasons = values.get("rejection_reasons")
    if not isinstance(reasons, (list, tuple)) or any(
        not isinstance(reason, str) for reason in reasons
    ):
        raise ValueError("qualification rejection reasons must be strings")
    values["rejection_reasons"] = tuple(reasons)
    decision = QualificationDecision(**values)
    _validate_recomputed_decision(decision)
    return decision


def _validate_recomputed_decision(decision: QualificationDecision) -> None:
    """Recompute every derived decision field from the stored primitive facts."""

    try:
        evidence = QualificationEvidence(
            case_id=decision.case_id,
            scale=decision.scale,
            courier_only_objective=float(decision.courier_only_objective),
            witness_objective=float(decision.witness_objective),
            total_demand_per_hour=float(decision.total_demand_per_hour),
            witness_service_per_hour=float(decision.witness_service_per_hour),
            open_facilities=decision.open_facilities,
            positive_cost_open_facilities=decision.positive_cost_open_facilities,
            candidate_h=decision.candidate_h,
            open_h=decision.open_h,
            witness_feasible=True,
            qualification_run_id=decision.qualification_run_id,
            instance_sha256=decision.instance_sha256,
            active_h=decision.active_h,
            active_h_threshold_per_hour=decision.active_h_threshold_per_hour,
            qualification_case_sha256=decision.qualification_case_sha256,
            effective_model_config_sha256=decision.effective_model_config_sha256,
            screening_is_exact=decision.screening_is_exact,
        )
        recomputed = assess_qualification(evidence, decision.policy)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(
            f"qualification decision primitive facts are invalid: {error}"
        ) from error
    if canonical_json(recomputed.to_payload()) != canonical_json(decision.to_payload()):
        raise ValueError("qualification decision does not match its primitive facts")


@dataclass(frozen=True, slots=True)
class FormalQualificationTrace:
    """Exact accepted qualification decision bound to one formal case."""

    manifest_id: str
    benchmark_manifest_sha256: str
    manifest_artifact_sha256: str
    case_id: str
    scale: str
    instance_sha256: str
    qualification_run_id: str
    qualification_case_sha256: str | None
    effective_model_config_sha256: str | None
    screening_is_exact: bool | None
    qualification_decision: QualificationDecision
    qualification_decision_sha256: str

    def __post_init__(self) -> None:
        _require_nonempty(self.manifest_id, label="manifest_id")
        _require_sha256(
            self.benchmark_manifest_sha256,
            label="benchmark_manifest_sha256",
        )
        _require_sha256(
            self.manifest_artifact_sha256,
            label="manifest_artifact_sha256",
        )
        _require_nonempty(self.case_id, label="case_id")
        _require_nonempty(self.scale, label="scale")
        _require_sha256(self.instance_sha256, label="instance_sha256")
        _require_nonempty(self.qualification_run_id, label="qualification_run_id")
        if self.qualification_case_sha256 is not None:
            _require_sha256(
                self.qualification_case_sha256,
                label="qualification_case_sha256",
            )
        if self.effective_model_config_sha256 is not None:
            _require_sha256(
                self.effective_model_config_sha256,
                label="effective_model_config_sha256",
            )
        if self.screening_is_exact is not None and type(self.screening_is_exact) is not bool:
            raise ValueError("screening_is_exact must be boolean or null")
        if not isinstance(self.qualification_decision, QualificationDecision):
            raise ValueError("qualification_decision has an invalid type")
        _validate_recomputed_decision(self.qualification_decision)
        if not self.qualification_decision.accepted:
            raise ValueError("formal qualification trace requires an accepted decision")
        decision = self.qualification_decision
        duplicated = {
            "case_id": self.case_id,
            "scale": self.scale,
            "instance_sha256": self.instance_sha256,
            "qualification_run_id": self.qualification_run_id,
            "qualification_case_sha256": self.qualification_case_sha256,
            "effective_model_config_sha256": self.effective_model_config_sha256,
            "screening_is_exact": self.screening_is_exact,
        }
        if any(getattr(decision, name) != value for name, value in duplicated.items()):
            raise ValueError("qualification trace identity differs from its decision")
        expected_decision_sha256 = canonical_sha256(decision.to_payload())
        if self.qualification_decision_sha256 != expected_decision_sha256:
            raise ValueError("qualification decision SHA256 is inconsistent")

    @classmethod
    def from_decision(
        cls,
        decision: QualificationDecision,
        *,
        manifest_id: str,
        benchmark_manifest_sha256: str,
        manifest_artifact_sha256: str,
    ) -> FormalQualificationTrace:
        return cls(
            manifest_id=manifest_id,
            benchmark_manifest_sha256=benchmark_manifest_sha256,
            manifest_artifact_sha256=manifest_artifact_sha256,
            case_id=decision.case_id,
            scale=decision.scale,
            instance_sha256=decision.instance_sha256,
            qualification_run_id=decision.qualification_run_id,
            qualification_case_sha256=decision.qualification_case_sha256,
            effective_model_config_sha256=decision.effective_model_config_sha256,
            screening_is_exact=decision.screening_is_exact,
            qualification_decision=decision,
            qualification_decision_sha256=canonical_sha256(decision.to_payload()),
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema_version": FORMAL_QUALIFICATION_TRACE_SCHEMA_VERSION,
            "artifact_type": FORMAL_QUALIFICATION_TRACE_ARTIFACT_TYPE,
            "manifest_id": self.manifest_id,
            "benchmark_manifest_sha256": self.benchmark_manifest_sha256,
            "manifest_artifact_sha256": self.manifest_artifact_sha256,
            "case_id": self.case_id,
            "scale": self.scale,
            "instance_sha256": self.instance_sha256,
            "qualification_run_id": self.qualification_run_id,
            "qualification_case_sha256": self.qualification_case_sha256,
            "effective_model_config_sha256": self.effective_model_config_sha256,
            "screening_is_exact": self.screening_is_exact,
            "qualification_decision": self.qualification_decision.to_payload(),
            "qualification_decision_sha256": self.qualification_decision_sha256,
        }

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.to_payload())

    @classmethod
    def from_payload(cls, payload: object) -> FormalQualificationTrace:
        if not isinstance(payload, Mapping):
            raise ValueError("formal qualification trace must be an object")
        expected_fields = {
            "schema_version",
            "artifact_type",
            "manifest_id",
            "benchmark_manifest_sha256",
            "manifest_artifact_sha256",
            "case_id",
            "scale",
            "instance_sha256",
            "qualification_run_id",
            "qualification_case_sha256",
            "effective_model_config_sha256",
            "screening_is_exact",
            "qualification_decision",
            "qualification_decision_sha256",
        }
        if set(payload) != expected_fields:
            raise ValueError("formal qualification trace fields are not canonical")
        if payload["schema_version"] != FORMAL_QUALIFICATION_TRACE_SCHEMA_VERSION:
            raise ValueError("formal qualification trace schema differs")
        if payload["artifact_type"] != FORMAL_QUALIFICATION_TRACE_ARTIFACT_TYPE:
            raise ValueError("formal qualification trace artifact type differs")
        return cls(
            manifest_id=payload["manifest_id"],
            benchmark_manifest_sha256=payload["benchmark_manifest_sha256"],
            manifest_artifact_sha256=payload["manifest_artifact_sha256"],
            case_id=payload["case_id"],
            scale=payload["scale"],
            instance_sha256=payload["instance_sha256"],
            qualification_run_id=payload["qualification_run_id"],
            qualification_case_sha256=payload["qualification_case_sha256"],
            effective_model_config_sha256=payload[
                "effective_model_config_sha256"
            ],
            screening_is_exact=payload["screening_is_exact"],
            qualification_decision=_decision_from_payload(
                payload["qualification_decision"]
            ),
            qualification_decision_sha256=payload[
                "qualification_decision_sha256"
            ],
        )

    def validate_binding(
        self,
        *,
        manifest_id: str,
        benchmark_manifest_sha256: str,
        manifest_artifact_sha256: str,
        case_id: str,
        scale: str,
        instance_sha256: str,
        qualification_run_id: str,
        qualification_decision_sha256: str,
        qualification_case_sha256: str | None,
        effective_model_config_sha256: str | None,
        screening_is_exact: bool | None,
    ) -> None:
        expected = {
            "manifest_id": manifest_id,
            "benchmark_manifest_sha256": benchmark_manifest_sha256,
            "manifest_artifact_sha256": manifest_artifact_sha256,
            "case_id": case_id,
            "scale": scale,
            "instance_sha256": instance_sha256,
            "qualification_run_id": qualification_run_id,
            "qualification_decision_sha256": qualification_decision_sha256,
            "qualification_case_sha256": qualification_case_sha256,
            "effective_model_config_sha256": effective_model_config_sha256,
            "screening_is_exact": screening_is_exact,
        }
        differing = [
            name for name, value in expected.items() if getattr(self, name) != value
        ]
        if differing:
            raise ValueError(
                "qualification trace binding differs: " + ", ".join(differing)
            )


def bind_qualification_trace(
    formal_context: Mapping[str, Any],
    trace: FormalQualificationTrace,
) -> dict[str, Any]:
    """Attach an exact trace and its digest to one run context."""

    if "qualification_trace" in formal_context or "qualification_trace_sha256" in formal_context:
        raise ValueError("formal context already contains qualification trace fields")
    return {
        **dict(formal_context),
        "qualification_trace": trace.to_payload(),
        "qualification_trace_sha256": trace.sha256,
    }


def qualification_trace_from_context(
    formal_context: Mapping[str, Any],
) -> FormalQualificationTrace:
    trace = FormalQualificationTrace.from_payload(
        formal_context.get("qualification_trace")
    )
    if formal_context.get("qualification_trace_sha256") != trace.sha256:
        raise ValueError("formal context qualification trace SHA256 differs")
    return trace


def _criteria_from_decision(decision: QualificationDecision) -> dict[str, bool]:
    reasons = set(decision.rejection_reasons)
    criteria = {
        "objective_improvement_margin_met": (
            QualificationReason.OBJECTIVE_IMPROVEMENT_BELOW_MARGIN not in reasons
        ),
        "minimum_savings_fraction_met": (
            QualificationReason.SAVINGS_FRACTION_BELOW_MINIMUM not in reasons
        ),
        "minimum_service_tolerance_met": (
            QualificationReason.SERVICE_BELOW_TOLERANCE not in reasons
        ),
        "minimum_service_fraction_met": (
            QualificationReason.SERVICE_FRACTION_BELOW_MINIMUM not in reasons
        ),
        "positive_cost_facility_open": (
            QualificationReason.NO_OPEN_FACILITY not in reasons
            and QualificationReason.NO_POSITIVE_COST_OPEN_FACILITY not in reasons
            and QualificationReason.POSITIVE_COST_OPEN_COUNT_EXCEEDS_TOTAL
            not in reasons
        ),
        "H_open_fraction_interior": (
            QualificationReason.NO_H_CANDIDATES not in reasons
            and QualificationReason.INVALID_OPEN_H_COUNT not in reasons
            and QualificationReason.H_OPEN_FRACTION_BELOW_MINIMUM not in reasons
            and QualificationReason.H_OPEN_FRACTION_ABOVE_MAXIMUM not in reasons
        ),
        "active_H_present": (
            QualificationReason.MISSING_ACTIVE_H_COUNT not in reasons
            and QualificationReason.NO_ACTIVE_H not in reasons
            and QualificationReason.INVALID_ACTIVE_H_COUNT not in reasons
            and QualificationReason.MISSING_ACTIVE_H_THRESHOLD not in reasons
            and QualificationReason.INVALID_ACTIVE_H_THRESHOLD not in reasons
            and QualificationReason.ACTIVE_H_THRESHOLD_MISMATCH not in reasons
        ),
        "large_active_H_target_met": (
            QualificationReason.LARGE_ACTIVE_H_BELOW_TARGET not in reasons
            and QualificationReason.LARGE_ACTIVE_H_ABOVE_TARGET not in reasons
        ),
        "all_criteria_met": decision.accepted,
    }
    return criteria


@dataclass(frozen=True, slots=True)
class FormalSolutionStructure:
    """Canonical structural facts independently extracted from one incumbent."""

    source: str
    has_incumbent: bool
    qualification_trace_sha256: str
    C0: float | None = None
    incumbent_objective: float | None = None
    total_demand_per_hour: float | None = None
    realized_service_per_hour: float | None = None
    realized_service_rate: float | None = None
    open_facility_count: int | None = None
    positive_cost_open_facility_count: int | None = None
    candidate_H_count: int | None = None
    open_H_count: int | None = None
    active_H_count: int | None = None
    active_H_threshold_per_hour: float | None = None
    nontriviality_evaluated: bool = False
    nontriviality_passed: bool | None = None
    nontriviality_criteria: Mapping[str, bool] | None = None
    nontriviality_rejection_reasons: tuple[str, ...] = ()
    nontriviality_decision: QualificationDecision | None = None
    nontriviality_decision_sha256: str | None = None

    def __post_init__(self) -> None:
        _require_sha256(
            self.qualification_trace_sha256,
            label="qualification_trace_sha256",
        )
        if type(self.has_incumbent) is not bool:
            raise ValueError("solution-structure incumbent flag must be boolean")
        if type(self.nontriviality_evaluated) is not bool:
            raise ValueError("nontriviality_evaluated must be boolean")
        if self.has_incumbent:
            if self.source != FORMAL_SOLUTION_STRUCTURE_SOURCE:
                raise ValueError("incumbent structure has an invalid source")
            for name in (
                "C0",
                "incumbent_objective",
                "total_demand_per_hour",
                "realized_service_per_hour",
                "realized_service_rate",
                "active_H_threshold_per_hour",
            ):
                value = getattr(self, name)
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise ValueError(f"{name} must be numeric for an incumbent")
                if not math.isfinite(float(value)):
                    raise ValueError(f"{name} must be finite for an incumbent")
            for name in (
                "open_facility_count",
                "positive_cost_open_facility_count",
                "candidate_H_count",
                "open_H_count",
                "active_H_count",
            ):
                value = getattr(self, name)
                if type(value) is not int or value < 0:
                    raise ValueError(f"{name} must be a nonnegative integer")
            if not self.nontriviality_evaluated:
                raise ValueError("incumbent nontriviality must be evaluated")
            if type(self.nontriviality_passed) is not bool:
                raise ValueError("incumbent nontriviality pass flag must be boolean")
            if not isinstance(self.nontriviality_criteria, Mapping):
                raise ValueError("incumbent nontriviality criteria are missing")
            if set(self.nontriviality_criteria) != set(
                _NONTRIVIALITY_CRITERIA_FIELDS
            ):
                raise ValueError("nontriviality criteria fields are not canonical")
            if any(type(value) is not bool for value in self.nontriviality_criteria.values()):
                raise ValueError("nontriviality criteria values must be boolean")
            if not isinstance(self.nontriviality_decision, QualificationDecision):
                raise ValueError("incumbent nontriviality decision is missing")
            _require_sha256(
                self.nontriviality_decision_sha256,
                label="nontriviality_decision_sha256",
            )
        else:
            if self.source != FORMAL_NO_INCUMBENT_SOURCE:
                raise ValueError("no-incumbent structure has an invalid source")
            nullable = (
                "C0",
                "incumbent_objective",
                "total_demand_per_hour",
                "realized_service_per_hour",
                "realized_service_rate",
                "open_facility_count",
                "positive_cost_open_facility_count",
                "candidate_H_count",
                "open_H_count",
                "active_H_count",
                "active_H_threshold_per_hour",
                "nontriviality_passed",
                "nontriviality_criteria",
                "nontriviality_decision",
                "nontriviality_decision_sha256",
            )
            if self.nontriviality_evaluated or any(
                getattr(self, name) is not None for name in nullable
            ):
                raise ValueError("no-incumbent structure contains evaluated facts")
            if self.nontriviality_rejection_reasons:
                raise ValueError("no-incumbent structure cannot claim rejection reasons")

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema_version": FORMAL_SOLUTION_STRUCTURE_SCHEMA_VERSION,
            "source": self.source,
            "has_incumbent": self.has_incumbent,
            "qualification_trace_sha256": self.qualification_trace_sha256,
            "C0": self.C0,
            "incumbent_objective": self.incumbent_objective,
            "total_demand_per_hour": self.total_demand_per_hour,
            "realized_service_per_hour": self.realized_service_per_hour,
            "realized_service_rate": self.realized_service_rate,
            "open_facility_count": self.open_facility_count,
            "positive_cost_open_facility_count": (
                self.positive_cost_open_facility_count
            ),
            "candidate_H_count": self.candidate_H_count,
            "open_H_count": self.open_H_count,
            "active_H_count": self.active_H_count,
            "active_H_threshold_per_hour": self.active_H_threshold_per_hour,
            "nontriviality_evaluated": self.nontriviality_evaluated,
            "nontriviality_passed": self.nontriviality_passed,
            "nontriviality_criteria": (
                dict(self.nontriviality_criteria)
                if self.nontriviality_criteria is not None
                else None
            ),
            "nontriviality_rejection_reasons": list(
                self.nontriviality_rejection_reasons
            ),
            "nontriviality_decision": (
                self.nontriviality_decision.to_payload()
                if self.nontriviality_decision is not None
                else None
            ),
            "nontriviality_decision_sha256": (
                self.nontriviality_decision_sha256
            ),
        }

    @classmethod
    def from_payload(
        cls,
        payload: object,
    ) -> FormalSolutionStructure:
        if not isinstance(payload, Mapping):
            raise ValueError("formal solution structure must be an object")
        if set(payload) != _SOLUTION_STRUCTURE_FIELDS:
            raise ValueError("formal solution structure fields are not canonical")
        if payload["schema_version"] != FORMAL_SOLUTION_STRUCTURE_SCHEMA_VERSION:
            raise ValueError("formal solution structure schema differs")
        reasons = payload["nontriviality_rejection_reasons"]
        if not isinstance(reasons, (list, tuple)) or any(
            not isinstance(reason, str) for reason in reasons
        ):
            raise ValueError("nontriviality rejection reasons must be strings")
        raw_decision = payload["nontriviality_decision"]
        decision = (
            _decision_from_payload(raw_decision)
            if raw_decision is not None
            else None
        )
        return cls(
            source=payload["source"],
            has_incumbent=payload["has_incumbent"],
            qualification_trace_sha256=payload["qualification_trace_sha256"],
            C0=payload["C0"],
            incumbent_objective=payload["incumbent_objective"],
            total_demand_per_hour=payload["total_demand_per_hour"],
            realized_service_per_hour=payload["realized_service_per_hour"],
            realized_service_rate=payload["realized_service_rate"],
            open_facility_count=payload["open_facility_count"],
            positive_cost_open_facility_count=payload[
                "positive_cost_open_facility_count"
            ],
            candidate_H_count=payload["candidate_H_count"],
            open_H_count=payload["open_H_count"],
            active_H_count=payload["active_H_count"],
            active_H_threshold_per_hour=payload[
                "active_H_threshold_per_hour"
            ],
            nontriviality_evaluated=payload["nontriviality_evaluated"],
            nontriviality_passed=payload["nontriviality_passed"],
            nontriviality_criteria=payload["nontriviality_criteria"],
            nontriviality_rejection_reasons=tuple(reasons),
            nontriviality_decision=decision,
            nontriviality_decision_sha256=payload[
                "nontriviality_decision_sha256"
            ],
        )


def no_incumbent_solution_structure(
    trace: FormalQualificationTrace,
) -> FormalSolutionStructure:
    return FormalSolutionStructure(
        source=FORMAL_NO_INCUMBENT_SOURCE,
        has_incumbent=False,
        qualification_trace_sha256=trace.sha256,
    )


def build_solution_structure_from_facts(
    trace: FormalQualificationTrace,
    *,
    C0: float,
    incumbent_objective: float,
    total_demand_per_hour: float,
    realized_service_per_hour: float,
    open_facility_count: int,
    positive_cost_open_facility_count: int,
    candidate_H_count: int,
    open_H_count: int,
    active_H_count: int,
    active_H_threshold_per_hour: float,
) -> FormalSolutionStructure:
    evidence = QualificationEvidence(
        case_id=trace.case_id,
        scale=trace.scale,
        courier_only_objective=float(C0),
        witness_objective=float(incumbent_objective),
        total_demand_per_hour=float(total_demand_per_hour),
        witness_service_per_hour=float(realized_service_per_hour),
        open_facilities=open_facility_count,
        positive_cost_open_facilities=positive_cost_open_facility_count,
        candidate_h=candidate_H_count,
        open_h=open_H_count,
        witness_feasible=True,
        qualification_run_id=trace.qualification_run_id,
        instance_sha256=trace.instance_sha256,
        active_h=active_H_count,
        active_h_threshold_per_hour=active_H_threshold_per_hour,
        qualification_case_sha256=trace.qualification_case_sha256,
        effective_model_config_sha256=trace.effective_model_config_sha256,
        screening_is_exact=trace.screening_is_exact,
    )
    decision = assess_qualification(evidence, trace.qualification_decision.policy)
    rate = (
        float(realized_service_per_hour) / float(total_demand_per_hour)
        if float(total_demand_per_hour) != 0.0
        else math.nan
    )
    return FormalSolutionStructure(
        source=FORMAL_SOLUTION_STRUCTURE_SOURCE,
        has_incumbent=True,
        qualification_trace_sha256=trace.sha256,
        C0=float(C0),
        incumbent_objective=float(incumbent_objective),
        total_demand_per_hour=float(total_demand_per_hour),
        realized_service_per_hour=float(realized_service_per_hour),
        realized_service_rate=rate,
        open_facility_count=open_facility_count,
        positive_cost_open_facility_count=positive_cost_open_facility_count,
        candidate_H_count=candidate_H_count,
        open_H_count=open_H_count,
        active_H_count=active_H_count,
        active_H_threshold_per_hour=float(active_H_threshold_per_hour),
        nontriviality_evaluated=True,
        nontriviality_passed=decision.accepted,
        nontriviality_criteria=_criteria_from_decision(decision),
        nontriviality_rejection_reasons=decision.rejection_reasons,
        nontriviality_decision=decision,
        nontriviality_decision_sha256=canonical_sha256(decision.to_payload()),
    )


def _require_close(observed: float, expected: float, *, label: str) -> None:
    if not math.isclose(observed, expected, rel_tol=1e-10, abs_tol=1e-8):
        raise ValueError(f"{label} differs from the frozen qualification case")


def recompute_solution_structure(
    model_wrapper: object | None,
    trace: FormalQualificationTrace,
    *,
    expected_has_incumbent: bool,
) -> FormalSolutionStructure:
    """Read one actual incumbent and independently recompute structural facts."""

    if model_wrapper is None:
        if expected_has_incumbent:
            raise ValueError("incumbent exists but the model wrapper is unavailable")
        return no_incumbent_solution_structure(trace)
    backend = getattr(model_wrapper, "model", None)
    if backend is None:
        if expected_has_incumbent:
            raise ValueError("incumbent exists but the backend model is unavailable")
        return no_incumbent_solution_structure(trace)
    backend_has_incumbent = int(getattr(backend, "SolCount")) > 0
    if backend_has_incumbent != expected_has_incumbent:
        raise ValueError("backend and solver-metric incumbent flags differ")
    if not expected_has_incumbent:
        return no_incumbent_solution_structure(trace)

    instance = getattr(model_wrapper, "inst")
    markets = tuple(getattr(instance, "markets"))
    demand_by_market = getattr(instance, "h_od")
    C0 = math.fsum(
        float(instance.direct_rider_cost(*market))
        * float(demand_by_market[market])
        for market in markets
    )
    total_demand = math.fsum(float(demand_by_market[market]) for market in markets)
    _require_close(
        C0,
        float(getattr(model_wrapper, "courier_only_baseline")),
        label="independently recomputed C0 and model C0",
    )
    _require_close(
        total_demand,
        float(getattr(instance, "total_demand")),
        label="independently recomputed demand and instance demand",
    )
    qualification = trace.qualification_decision
    _require_close(
        C0,
        float(qualification.courier_only_objective),
        label="formal C0",
    )
    _require_close(
        total_demand,
        float(qualification.total_demand_per_hour),
        label="formal total demand",
    )

    variables = getattr(model_wrapper, "vars")
    x = variables["x"]
    open_nodes = [node for node, variable in x.items() if float(variable.X) > 0.5]
    positive_cost_open = sum(
        float(instance.facility_cost(node)) > 0.0 for node in open_nodes
    )
    lockers = tuple(instance.H)
    if len(lockers) != qualification.candidate_h:
        raise ValueError("formal H-candidate count differs from qualification")
    open_H = sum(float(x[locker].X) > 0.5 for locker in lockers)

    levels = tuple(instance.config.availability_levels)
    chi = variables["chi"]
    realized_service = math.fsum(
        float(levels[key[4]]) * float(chi[key].X)
        for key in model_wrapper.chi_keys
    )
    active_threshold = service_tolerance(total_demand)
    locker_set = set(lockers)
    incident_flow = {locker: [] for locker in lockers}
    psi = variables["psi"]
    for u, v in instance.A_F:
        flow = max(0.0, float(psi[u, v].X))
        if u in locker_set:
            incident_flow[u].append(flow)
        if v in locker_set:
            incident_flow[v].append(flow)
    active_H = sum(
        float(x[locker].X) > 0.5
        and math.fsum(incident_flow[locker]) > active_threshold
        for locker in lockers
    )
    return build_solution_structure_from_facts(
        trace,
        C0=C0,
        incumbent_objective=float(backend.ObjVal),
        total_demand_per_hour=total_demand,
        realized_service_per_hour=realized_service,
        open_facility_count=len(open_nodes),
        positive_cost_open_facility_count=positive_cost_open,
        candidate_H_count=len(lockers),
        open_H_count=open_H,
        active_H_count=active_H,
        active_H_threshold_per_hour=active_threshold,
    )


def validate_solution_structure_payload(
    payload: object,
    trace: FormalQualificationTrace,
    *,
    expected_has_incumbent: bool,
) -> FormalSolutionStructure:
    structure = FormalSolutionStructure.from_payload(payload)
    if structure.has_incumbent != expected_has_incumbent:
        raise ValueError("solution structure and solver incumbent flags differ")
    if structure.qualification_trace_sha256 != trace.sha256:
        raise ValueError("solution structure qualification trace SHA256 differs")
    if not expected_has_incumbent:
        expected = no_incumbent_solution_structure(trace)
    else:
        expected = build_solution_structure_from_facts(
            trace,
            C0=float(structure.C0),
            incumbent_objective=float(structure.incumbent_objective),
            total_demand_per_hour=float(structure.total_demand_per_hour),
            realized_service_per_hour=float(structure.realized_service_per_hour),
            open_facility_count=int(structure.open_facility_count),
            positive_cost_open_facility_count=int(
                structure.positive_cost_open_facility_count
            ),
            candidate_H_count=int(structure.candidate_H_count),
            open_H_count=int(structure.open_H_count),
            active_H_count=int(structure.active_H_count),
            active_H_threshold_per_hour=float(
                structure.active_H_threshold_per_hour
            ),
        )
    if canonical_json(structure.to_payload()) != canonical_json(expected.to_payload()):
        raise ValueError("formal solution structure is not self-consistent")
    return structure


__all__ = [
    "FORMAL_NO_INCUMBENT_SOURCE",
    "FORMAL_QUALIFICATION_TRACE_ARTIFACT_TYPE",
    "FORMAL_QUALIFICATION_TRACE_SCHEMA_VERSION",
    "FORMAL_SOLUTION_STRUCTURE_SCHEMA_VERSION",
    "FORMAL_SOLUTION_STRUCTURE_SOURCE",
    "FormalQualificationTrace",
    "FormalSolutionStructure",
    "bind_qualification_trace",
    "build_solution_structure_from_facts",
    "no_incumbent_solution_structure",
    "qualification_trace_from_context",
    "recompute_solution_structure",
    "validate_solution_structure_payload",
]
