"""Development-only incumbent reuse for qualification profile search.

This module is deliberately outside the paper A--E algorithms.  It keeps at
most one fresh-validated complete incumbent for each exact feasibility-system
signature and may offer that incumbent only to a distinct objective profile in
the same serial development qualification run.  The signature excludes the
objective by construction, but includes variable order/domains and every
supported linear, quadratic, and SOS constraint record.

The pool never changes a mathematical model, never adds an opening constraint,
and never changes screening.  A caller must still validate any resulting
incumbent in a fresh copy of the target model through the normal qualification
runner before the target case can be accepted.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import math
from typing import Any

from .hashing import canonical_json, canonical_sha256


QUALIFICATION_CONSTRAINT_SIGNATURE_SCHEMA = "mlnd-qualification-constraint-signature-v1"
QUALIFICATION_INCUMBENT_REUSE_POLICY_VERSION = (
    "development-qualification-incumbent-reuse-v1"
)


def _require_sha256(value: object, *, field_name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field_name} must be a lowercase SHA256 digest")
    return value


def _require_nonempty(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} must be a nonempty string")
    return value


def ordered_variable_name_sha256(names: Sequence[str]) -> str:
    """Hash one exact ordered variable-name vector without delimiter ambiguity."""

    digest = hashlib.sha256(b"mlnd-complete-start-var-order-v1\0")
    for name in names:
        if not isinstance(name, str) or not name:
            raise ValueError("complete incumbent variable names must be nonempty")
        encoded = name.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, byteorder="big", signed=False))
        digest.update(encoded)
    return digest.hexdigest()


def _normalized_record(record: Mapping[str, Any], *, field_name: str) -> dict[str, Any]:
    if not isinstance(record, Mapping) or any(
        not isinstance(key, str) for key in record
    ):
        raise TypeError(f"{field_name} must be a mapping with string keys")
    # Canonical serialization performs the recursive type, finiteness, and
    # deterministic-order checks.  Round-tripping through JSON is unnecessary;
    # the original mapping is copied so later caller mutation cannot affect the
    # record stream being hashed.
    copied = dict(record)
    canonical_json(copied)
    return copied


def _stream_record(
    digest: Any, *, family: str, ordinal: int, record: Mapping[str, Any]
) -> dict[str, Any]:
    normalized = _normalized_record(record, field_name=family)
    encoded = canonical_json(
        {
            "family": family,
            "ordinal": ordinal,
            "record": normalized,
        }
    ).encode("utf-8")
    digest.update(len(encoded).to_bytes(8, byteorder="big", signed=False))
    digest.update(encoded)
    return normalized


def _hash_record_family(
    family: str,
    records: Iterable[Mapping[str, Any]],
    *,
    capture_ordered_variable_names: bool = False,
) -> tuple[int, str, str | None]:
    """Hash one potentially very large record stream with bounded memory."""

    digest = hashlib.sha256(
        (QUALIFICATION_CONSTRAINT_SIGNATURE_SCHEMA + "\0" + family + "\0").encode(
            "utf-8"
        )
    )
    variable_name_digest = (
        hashlib.sha256(b"mlnd-complete-start-var-order-v1\0")
        if capture_ordered_variable_names
        else None
    )
    count = 0
    for ordinal, record in enumerate(records):
        normalized = _stream_record(
            digest,
            family=family,
            ordinal=ordinal,
            record=record,
        )
        if variable_name_digest is not None:
            name = _require_nonempty(
                normalized.get("name"),
                field_name="variable name",
            )
            encoded_name = name.encode("utf-8")
            variable_name_digest.update(
                len(encoded_name).to_bytes(8, byteorder="big", signed=False)
            )
            variable_name_digest.update(encoded_name)
        count += 1
    return (
        count,
        digest.hexdigest(),
        variable_name_digest.hexdigest() if variable_name_digest is not None else None,
    )


@dataclass(frozen=True, slots=True)
class QualificationConstraintSignature:
    """Cryptographic identity of a supported feasibility system.

    Objective coefficients, objective constant, model sense, and MIP-start
    attributes are intentionally absent.  Equality therefore means that a
    complete feasible vector can be transferred across objective-only profiles,
    subject to the additional physical-instance and qualification provenance
    checks enforced by :class:`DevelopmentQualificationIncumbentPool`.
    """

    constraint_signature_sha256: str
    ordered_var_name_sha256: str
    variable_count: int
    linear_constraint_count: int
    quadratic_constraint_count: int
    sos_constraint_count: int
    unsupported_general_constraint_count: int = 0
    unsupported_nonlinear_constraint_count: int = 0
    signature_schema: str = QUALIFICATION_CONSTRAINT_SIGNATURE_SCHEMA
    extraction_complete: bool = True
    objective_excluded: bool = True

    def __post_init__(self) -> None:
        if self.signature_schema != QUALIFICATION_CONSTRAINT_SIGNATURE_SCHEMA:
            raise ValueError("constraint signature schema is unsupported")
        object.__setattr__(
            self,
            "constraint_signature_sha256",
            _require_sha256(
                self.constraint_signature_sha256,
                field_name="constraint_signature_sha256",
            ),
        )
        object.__setattr__(
            self,
            "ordered_var_name_sha256",
            _require_sha256(
                self.ordered_var_name_sha256,
                field_name="ordered_var_name_sha256",
            ),
        )
        for name in (
            "variable_count",
            "linear_constraint_count",
            "quadratic_constraint_count",
            "sos_constraint_count",
            "unsupported_general_constraint_count",
            "unsupported_nonlinear_constraint_count",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.variable_count < 1:
            raise ValueError("constraint signature requires at least one variable")
        if self.unsupported_general_constraint_count:
            raise ValueError(
                "constraint signature cannot certify models with general constraints"
            )
        if self.unsupported_nonlinear_constraint_count:
            raise ValueError(
                "constraint signature cannot certify models with nonlinear constraints"
            )
        if self.extraction_complete is not True:
            raise ValueError("constraint signature extraction must be complete")
        if self.objective_excluded is not True:
            raise ValueError("constraint signature must exclude the objective")

    def to_payload(self) -> dict[str, Any]:
        return {
            "signature_schema": self.signature_schema,
            "constraint_signature_sha256": self.constraint_signature_sha256,
            "ordered_var_name_sha256": self.ordered_var_name_sha256,
            "variable_count": self.variable_count,
            "linear_constraint_count": self.linear_constraint_count,
            "quadratic_constraint_count": self.quadratic_constraint_count,
            "sos_constraint_count": self.sos_constraint_count,
            "unsupported_general_constraint_count": (
                self.unsupported_general_constraint_count
            ),
            "unsupported_nonlinear_constraint_count": (
                self.unsupported_nonlinear_constraint_count
            ),
            "extraction_complete": self.extraction_complete,
            "objective_excluded": self.objective_excluded,
        }


def build_qualification_constraint_signature(
    *,
    variable_records: Iterable[Mapping[str, Any]],
    linear_constraint_records: Iterable[Mapping[str, Any]],
    quadratic_constraint_records: Iterable[Mapping[str, Any]],
    sos_constraint_records: Iterable[Mapping[str, Any]],
    unsupported_general_constraint_count: int = 0,
    unsupported_nonlinear_constraint_count: int = 0,
) -> QualificationConstraintSignature:
    """Build an exact, objective-neutral signature from complete model records.

    Unsupported general or nonlinear constraints fail closed because omitting
    even one feasibility construct would make cross-profile reuse unsafe.
    """

    for name, value in (
        ("unsupported_general_constraint_count", unsupported_general_constraint_count),
        (
            "unsupported_nonlinear_constraint_count",
            unsupported_nonlinear_constraint_count,
        ),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    if unsupported_general_constraint_count:
        raise ValueError(
            "cannot certify exact equivalence when general constraints are present"
        )
    if unsupported_nonlinear_constraint_count:
        raise ValueError(
            "cannot certify exact equivalence when nonlinear constraints are present"
        )
    variable_count, variable_sha256, ordered_names_sha256 = _hash_record_family(
        "variable",
        variable_records,
        capture_ordered_variable_names=True,
    )
    if variable_count < 1 or ordered_names_sha256 is None:
        raise ValueError("constraint signature requires at least one variable")
    linear_count, linear_sha256, _ = _hash_record_family(
        "linear_constraint",
        linear_constraint_records,
    )
    quadratic_count, quadratic_sha256, _ = _hash_record_family(
        "quadratic_constraint",
        quadratic_constraint_records,
    )
    sos_count, sos_sha256, _ = _hash_record_family(
        "sos_constraint",
        sos_constraint_records,
    )
    signature_payload = {
        "signature_schema": QUALIFICATION_CONSTRAINT_SIGNATURE_SCHEMA,
        "objective_excluded": True,
        "variable": {"count": variable_count, "records_sha256": variable_sha256},
        "linear_constraint": {
            "count": linear_count,
            "records_sha256": linear_sha256,
        },
        "quadratic_constraint": {
            "count": quadratic_count,
            "records_sha256": quadratic_sha256,
        },
        "sos_constraint": {"count": sos_count, "records_sha256": sos_sha256},
        "unsupported_general_constraint_count": unsupported_general_constraint_count,
        "unsupported_nonlinear_constraint_count": (
            unsupported_nonlinear_constraint_count
        ),
    }

    return QualificationConstraintSignature(
        constraint_signature_sha256=canonical_sha256(signature_payload),
        ordered_var_name_sha256=ordered_names_sha256,
        variable_count=variable_count,
        linear_constraint_count=linear_count,
        quadratic_constraint_count=quadratic_count,
        sos_constraint_count=sos_count,
        unsupported_general_constraint_count=unsupported_general_constraint_count,
        unsupported_nonlinear_constraint_count=(unsupported_nonlinear_constraint_count),
    )


class ConstraintSignatureExtractionError(RuntimeError):
    """A model construct could not be completely and safely serialized."""


def _finite_coefficient(value: object, *, field_name: str) -> float:
    coefficient = float(value)
    if not math.isfinite(coefficient):
        raise ConstraintSignatureExtractionError(
            f"{field_name} must be finite for exact constraint hashing"
        )
    return coefficient


def _constraint_bound(value: object, *, field_name: str) -> float | str:
    bound = float(value)
    if math.isnan(bound):
        raise ConstraintSignatureExtractionError(
            f"{field_name} cannot be NaN in an exact constraint signature"
        )
    if math.isinf(bound):
        return "+infinity" if bound > 0.0 else "-infinity"
    return bound


def _model_count(model: object, name: str) -> int:
    value = getattr(model, name, 0)
    try:
        count = int(value)
    except (TypeError, ValueError) as error:
        raise ConstraintSignatureExtractionError(
            f"model attribute {name} is not an integer"
        ) from error
    if count < 0:
        raise ConstraintSignatureExtractionError(f"model attribute {name} is negative")
    return count


def _constraint_variable_index(
    variable: object,
    *,
    variable_names: Sequence[str],
    field_name: str,
) -> int:
    raw_index = getattr(variable, "index", None)
    try:
        variable_index = int(raw_index)
    except (TypeError, ValueError) as error:
        raise ConstraintSignatureExtractionError(
            f"{field_name} variable has no stable model index"
        ) from error
    name = str(variable.VarName)
    if (
        variable_index < 0
        or variable_index >= len(variable_names)
        or variable_names[variable_index] != name
    ):
        raise ConstraintSignatureExtractionError(
            f"{field_name} variable index/name binding is inconsistent"
        )
    return variable_index


def _aggregate_linear_terms(
    expression: object,
    *,
    variable_names: Sequence[str],
    field_name: str,
) -> tuple[tuple[int, float], ...]:
    totals: dict[int, list[float]] = {}
    size = int(expression.size())
    for term_index in range(size):
        variable_index = _constraint_variable_index(
            expression.getVar(term_index),
            variable_names=variable_names,
            field_name=field_name,
        )
        coefficient = _finite_coefficient(
            expression.getCoeff(term_index),
            field_name=f"{field_name} coefficient",
        )
        totals.setdefault(variable_index, []).append(coefficient)
    aggregated: list[tuple[int, float]] = []
    for variable_index, coefficients in sorted(totals.items()):
        coefficient = math.fsum(coefficients)
        if not math.isfinite(coefficient):
            raise ConstraintSignatureExtractionError(
                f"{field_name} aggregated coefficient is non-finite"
            )
        if coefficient != 0.0:
            aggregated.append((variable_index, coefficient))
    return tuple(aggregated)


def _aggregate_quadratic_terms(
    expression: object,
    *,
    variable_names: Sequence[str],
    field_name: str,
) -> tuple[tuple[int, int, float], ...]:
    totals: dict[tuple[int, int], list[float]] = {}
    size = int(expression.size())
    for term_index in range(size):
        first_index = _constraint_variable_index(
            expression.getVar1(term_index),
            variable_names=variable_names,
            field_name=field_name,
        )
        second_index = _constraint_variable_index(
            expression.getVar2(term_index),
            variable_names=variable_names,
            field_name=field_name,
        )
        pair = tuple(sorted((first_index, second_index)))
        coefficient = _finite_coefficient(
            expression.getCoeff(term_index),
            field_name=f"{field_name} coefficient",
        )
        totals.setdefault(pair, []).append(coefficient)
    aggregated: list[tuple[int, int, float]] = []
    for (first_index, second_index), coefficients in sorted(totals.items()):
        coefficient = math.fsum(coefficients)
        if not math.isfinite(coefficient):
            raise ConstraintSignatureExtractionError(
                f"{field_name} aggregated coefficient is non-finite"
            )
        if coefficient != 0.0:
            aggregated.append((first_index, second_index, coefficient))
    return tuple(aggregated)


def extract_gurobi_constraint_signature(
    model: object,
    grb: object,
) -> QualificationConstraintSignature:
    """Extract an objective-neutral signature through Gurobi's public API.

    The function is duck-typed so its full record logic can be unit-tested
    without importing or invoking Gurobi.  The caller owns model lifecycle.
    """

    model.update()
    variables = tuple(model.getVars())
    if not variables:
        raise ConstraintSignatureExtractionError(
            "cannot hash an empty qualification model"
        )
    names = tuple(str(value) for value in model.getAttr(grb.Attr.VarName, variables))
    if len(names) != len(variables):
        raise ConstraintSignatureExtractionError(
            "bulk variable names do not cover the complete model"
        )
    vtypes = model.getAttr(grb.Attr.VType, variables)
    lower_bounds = model.getAttr(grb.Attr.LB, variables)
    upper_bounds = model.getAttr(grb.Attr.UB, variables)
    if not (len(vtypes) == len(lower_bounds) == len(upper_bounds) == len(variables)):
        raise ConstraintSignatureExtractionError(
            "bulk variable-domain attributes are incomplete"
        )

    def variable_records() -> Iterable[Mapping[str, Any]]:
        for index, (name, vtype, lower_bound, upper_bound) in enumerate(
            zip(names, vtypes, lower_bounds, upper_bounds, strict=True)
        ):
            yield {
                "index": index,
                "name": name,
                "vtype": str(vtype),
                "lower_bound": _constraint_bound(
                    lower_bound,
                    field_name=f"variable {index} lower bound",
                ),
                "upper_bound": _constraint_bound(
                    upper_bound,
                    field_name=f"variable {index} upper bound",
                ),
            }

    def linear_constraint_records() -> Iterable[Mapping[str, Any]]:
        for index, constraint in enumerate(model.getConstrs()):
            row = model.getRow(constraint)
            yield {
                "index": index,
                "name": str(constraint.ConstrName),
                "sense": str(constraint.Sense),
                "rhs": _finite_coefficient(
                    constraint.RHS,
                    field_name=f"linear constraint {index} RHS",
                ),
                "lazy": int(constraint.Lazy),
                "terms": _aggregate_linear_terms(
                    row,
                    variable_names=names,
                    field_name=f"linear constraint {index}",
                ),
            }

    def quadratic_constraint_records() -> Iterable[Mapping[str, Any]]:
        for index, constraint in enumerate(model.getQConstrs()):
            row = model.getQCRow(constraint)
            yield {
                "index": index,
                "name": str(constraint.QCName),
                "sense": str(constraint.QCSense),
                "rhs": _finite_coefficient(
                    constraint.QCRHS,
                    field_name=f"quadratic constraint {index} RHS",
                ),
                "linear_terms": _aggregate_linear_terms(
                    row.getLinExpr(),
                    variable_names=names,
                    field_name=f"quadratic constraint {index} linear part",
                ),
                "quadratic_terms": _aggregate_quadratic_terms(
                    row,
                    variable_names=names,
                    field_name=f"quadratic constraint {index} quadratic part",
                ),
            }

    def sos_constraint_records() -> Iterable[Mapping[str, Any]]:
        for index, sos_constraint in enumerate(model.getSOSs()):
            sos_type, sos_variables, weights = model.getSOS(sos_constraint)
            if len(sos_variables) != len(weights):
                raise ConstraintSignatureExtractionError(
                    f"SOS constraint {index} has mismatched variables and weights"
                )
            entries = tuple(
                (
                    _constraint_variable_index(
                        variable,
                        variable_names=names,
                        field_name=f"SOS constraint {index}",
                    ),
                    _finite_coefficient(
                        raw_weight,
                        field_name=f"SOS constraint {index} weight",
                    ),
                )
                for variable, raw_weight in zip(
                    sos_variables,
                    weights,
                    strict=True,
                )
            )
            yield {
                "index": index,
                "sos_type": int(sos_type),
                "entries": entries,
            }

    general_count = _model_count(model, "NumGenConstrs")
    nonlinear_count = _model_count(model, "NumNLConstrs")
    try:
        signature = build_qualification_constraint_signature(
            variable_records=variable_records(),
            linear_constraint_records=linear_constraint_records(),
            quadratic_constraint_records=quadratic_constraint_records(),
            sos_constraint_records=sos_constraint_records(),
            unsupported_general_constraint_count=general_count,
            unsupported_nonlinear_constraint_count=nonlinear_count,
        )
    except (TypeError, ValueError) as error:
        raise ConstraintSignatureExtractionError(
            f"exact qualification constraint hashing failed: {error}"
        ) from error
    if (
        signature.variable_count != _model_count(model, "NumVars")
        or signature.linear_constraint_count != _model_count(model, "NumConstrs")
        or signature.quadratic_constraint_count != _model_count(model, "NumQConstrs")
        or signature.sos_constraint_count != _model_count(model, "NumSOS")
        or general_count != _model_count(model, "NumGenConstrs")
        or nonlinear_count != _model_count(model, "NumNLConstrs")
    ):
        raise ConstraintSignatureExtractionError(
            "model structure changed during constraint signature extraction"
        )
    if _model_count(model, "NumScenarios"):
        raise ConstraintSignatureExtractionError(
            "multi-scenario models are unsupported by constraint signature v1"
        )
    return signature


@dataclass(frozen=True, slots=True)
class QualificationReuseProvenance:
    """Target/source identity for one development qualification job."""

    qualification_run_id: str
    case_id: str
    profile_id: str
    scale: str
    mode: str
    physical_instance_sha256: str
    qualification_case_sha256: str
    effective_model_config_sha256: str
    demand_scale: float
    screening_is_exact: bool
    development_only: bool = True
    eligible_for_formal_pool: bool = False
    eligible_for_algorithm_comparison: bool = False
    reuse_policy_version: str = QUALIFICATION_INCUMBENT_REUSE_POLICY_VERSION

    def __post_init__(self) -> None:
        for field_name in (
            "qualification_run_id",
            "case_id",
            "profile_id",
            "scale",
            "mode",
        ):
            object.__setattr__(
                self,
                field_name,
                _require_nonempty(getattr(self, field_name), field_name=field_name),
            )
        object.__setattr__(self, "mode", self.mode.upper())
        for field_name in (
            "physical_instance_sha256",
            "qualification_case_sha256",
            "effective_model_config_sha256",
        ):
            object.__setattr__(
                self,
                field_name,
                _require_sha256(getattr(self, field_name), field_name=field_name),
            )
        demand_scale = float(self.demand_scale)
        if not math.isfinite(demand_scale) or demand_scale <= 0.0:
            raise ValueError("demand_scale must be positive and finite")
        object.__setattr__(self, "demand_scale", demand_scale)
        if self.screening_is_exact is not False:
            raise ValueError(
                "development incumbent reuse requires screening_is_exact=false"
            )
        if self.development_only is not True:
            raise ValueError("incumbent reuse provenance must be development-only")
        if self.eligible_for_formal_pool is not False:
            raise ValueError(
                "incumbent reuse provenance must be formal-pool ineligible"
            )
        if self.eligible_for_algorithm_comparison is not False:
            raise ValueError(
                "incumbent reuse provenance must be algorithm-comparison ineligible"
            )
        if self.reuse_policy_version != QUALIFICATION_INCUMBENT_REUSE_POLICY_VERSION:
            raise ValueError("incumbent reuse policy version is unsupported")

    def to_payload(self) -> dict[str, Any]:
        return {
            "reuse_policy_version": self.reuse_policy_version,
            "qualification_run_id": self.qualification_run_id,
            "case_id": self.case_id,
            "profile_id": self.profile_id,
            "scale": self.scale,
            "mode": self.mode,
            "physical_instance_sha256": self.physical_instance_sha256,
            "qualification_case_sha256": self.qualification_case_sha256,
            "effective_model_config_sha256": self.effective_model_config_sha256,
            "demand_scale": self.demand_scale,
            "screening_is_exact": self.screening_is_exact,
            "development_only": self.development_only,
            "eligible_for_formal_pool": self.eligible_for_formal_pool,
            "eligible_for_algorithm_comparison": (
                self.eligible_for_algorithm_comparison
            ),
        }


@dataclass(frozen=True, slots=True)
class ReusableQualificationIncumbent:
    """One complete source incumbent plus its non-formal audit envelope."""

    provenance: QualificationReuseProvenance
    constraint_signature: QualificationConstraintSignature
    incumbent: object = field(repr=False, compare=False)
    source_accepted: bool = True
    source_fresh_validation_performed: bool = True
    reuse_record_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if (
            self.source_accepted is not True
            or self.source_fresh_validation_performed is not True
        ):
            raise ValueError(
                "reusable incumbent must come from a fresh validated accepted source"
            )
        variable_count = getattr(self.incumbent, "variable_count", None)
        ordered_hash = getattr(self.incumbent, "ordered_var_name_sha256", None)
        model_fingerprint = getattr(self.incumbent, "model_fingerprint", None)
        objective = getattr(self.incumbent, "objective", None)
        if variable_count != self.constraint_signature.variable_count:
            raise ValueError("reusable incumbent variable count differs from signature")
        if ordered_hash != self.constraint_signature.ordered_var_name_sha256:
            raise ValueError(
                "reusable incumbent variable-name order differs from signature"
            )
        if isinstance(model_fingerprint, bool) or not isinstance(
            model_fingerprint, int
        ):
            raise ValueError("reusable incumbent requires a source model fingerprint")
        if isinstance(objective, bool) or not isinstance(objective, (int, float)):
            raise ValueError("reusable incumbent requires a numeric source objective")
        objective = float(objective)
        if not math.isfinite(objective):
            raise ValueError("reusable incumbent source objective must be finite")
        object.__setattr__(
            self,
            "reuse_record_sha256",
            canonical_sha256(self.metadata_payload(include_record_sha256=False)),
        )

    def metadata_payload(self, *, include_record_sha256: bool = True) -> dict[str, Any]:
        payload = {
            "reuse_policy_version": QUALIFICATION_INCUMBENT_REUSE_POLICY_VERSION,
            "development_only": True,
            "eligible_for_formal_pool": False,
            "eligible_for_algorithm_comparison": False,
            "source_provenance": self.provenance.to_payload(),
            "constraint_signature": self.constraint_signature.to_payload(),
            "source_accepted": self.source_accepted,
            "source_fresh_validation_performed": (
                self.source_fresh_validation_performed
            ),
            "incumbent_metadata": {
                "objective_under_source_profile": float(
                    getattr(self.incumbent, "objective")
                ),
                "variable_count": int(getattr(self.incumbent, "variable_count")),
                "ordered_var_name_sha256": str(
                    getattr(self.incumbent, "ordered_var_name_sha256")
                ),
                "source_model_fingerprint": int(
                    getattr(self.incumbent, "model_fingerprint")
                ),
                "values_persisted_to_audit": False,
            },
        }
        if include_record_sha256:
            payload["reuse_record_sha256"] = self.reuse_record_sha256
        return payload


@dataclass(frozen=True, slots=True)
class QualificationReuseAuditEvent:
    sequence_index: int
    action: str
    reason: str
    target_provenance: QualificationReuseProvenance
    target_constraint_signature: QualificationConstraintSignature
    source_provenance: QualificationReuseProvenance | None = None
    source_reuse_record_sha256: str | None = None

    def __post_init__(self) -> None:
        if (
            isinstance(self.sequence_index, bool)
            or not isinstance(self.sequence_index, int)
            or self.sequence_index < 0
        ):
            raise ValueError("sequence_index must be a nonnegative integer")
        _require_nonempty(self.action, field_name="action")
        _require_nonempty(self.reason, field_name="reason")
        if (self.source_provenance is None) != (
            self.source_reuse_record_sha256 is None
        ):
            raise ValueError("source provenance and record SHA must appear together")
        if self.source_reuse_record_sha256 is not None:
            _require_sha256(
                self.source_reuse_record_sha256,
                field_name="source_reuse_record_sha256",
            )

    def to_payload(self) -> dict[str, Any]:
        return {
            "sequence_index": self.sequence_index,
            "action": self.action,
            "reason": self.reason,
            "target_provenance": self.target_provenance.to_payload(),
            "target_constraint_signature": (
                self.target_constraint_signature.to_payload()
            ),
            "source_provenance": (
                self.source_provenance.to_payload()
                if self.source_provenance is not None
                else None
            ),
            "source_reuse_record_sha256": self.source_reuse_record_sha256,
        }


class DevelopmentQualificationIncumbentPool:
    """Serial, in-memory pool that cannot be consumed by paper A--E runs."""

    def __init__(self) -> None:
        self._entries: dict[tuple[object, ...], ReusableQualificationIncumbent] = {}
        self._audit_events: list[QualificationReuseAuditEvent] = []

    @property
    def audit_events(self) -> tuple[QualificationReuseAuditEvent, ...]:
        return tuple(self._audit_events)

    @staticmethod
    def _key(
        provenance: QualificationReuseProvenance,
        signature: QualificationConstraintSignature,
    ) -> tuple[object, ...]:
        return (
            provenance.qualification_run_id,
            provenance.physical_instance_sha256,
            provenance.scale,
            provenance.mode,
            provenance.demand_scale,
            provenance.screening_is_exact,
            signature.constraint_signature_sha256,
            signature.ordered_var_name_sha256,
            signature.variable_count,
        )

    def _event(
        self,
        *,
        action: str,
        reason: str,
        target: QualificationReuseProvenance,
        signature: QualificationConstraintSignature,
        source: ReusableQualificationIncumbent | None = None,
    ) -> None:
        self._audit_events.append(
            QualificationReuseAuditEvent(
                sequence_index=len(self._audit_events),
                action=action,
                reason=reason,
                target_provenance=target,
                target_constraint_signature=signature,
                source_provenance=(source.provenance if source is not None else None),
                source_reuse_record_sha256=(
                    source.reuse_record_sha256 if source is not None else None
                ),
            )
        )

    def select(
        self,
        *,
        target_provenance: QualificationReuseProvenance,
        target_constraint_signature: QualificationConstraintSignature,
    ) -> ReusableQualificationIncumbent | None:
        """Return only an exact-constraint source from a distinct profile."""

        key = self._key(target_provenance, target_constraint_signature)
        selected = self._entries.get(key)
        if selected is None:
            self._event(
                action="reuse_cache_miss",
                reason="no_exact_constraint_equivalent_source",
                target=target_provenance,
                signature=target_constraint_signature,
            )
            return None
        source = selected.provenance
        # Exact key equality already binds run, physical instance, scale, mode,
        # demand, screening, signature, variable order, and variable count.
        # Distinct config/case/profile identities prove this is cross-profile
        # reuse rather than a same-profile replay.
        if (
            source.profile_id == target_provenance.profile_id
            or source.effective_model_config_sha256
            == target_provenance.effective_model_config_sha256
            or source.qualification_case_sha256
            == target_provenance.qualification_case_sha256
        ):
            self._event(
                action="reuse_cache_miss",
                reason="source_and_target_profiles_are_not_distinct",
                target=target_provenance,
                signature=target_constraint_signature,
                source=selected,
            )
            return None
        if selected.constraint_signature != target_constraint_signature:
            # Defensive redundancy against key construction or object mutation.
            raise RuntimeError("reuse pool key matched but full signatures differ")
        self._event(
            action="reuse_candidate_selected",
            reason="exact_feasibility_signature_and_provenance_match",
            target=target_provenance,
            signature=target_constraint_signature,
            source=selected,
        )
        return selected

    def record_installation(
        self,
        *,
        target_provenance: QualificationReuseProvenance,
        target_constraint_signature: QualificationConstraintSignature,
        selected: ReusableQualificationIncumbent,
    ) -> None:
        if selected.constraint_signature != target_constraint_signature:
            raise ValueError("installed source signature differs from target")
        if self._key(selected.provenance, selected.constraint_signature) != self._key(
            target_provenance, target_constraint_signature
        ):
            raise ValueError("installed source provenance differs from target key")
        self._event(
            action="reuse_start_installed",
            reason="complete_start_installed_after_exact_constraint_match",
            target=target_provenance,
            signature=target_constraint_signature,
            source=selected,
        )

    def record_installation_failure(
        self,
        *,
        target_provenance: QualificationReuseProvenance,
        target_constraint_signature: QualificationConstraintSignature,
        selected: ReusableQualificationIncumbent,
        error: BaseException,
    ) -> None:
        self._event(
            action="reuse_start_installation_failed",
            reason=f"{type(error).__name__}: {error}",
            target=target_provenance,
            signature=target_constraint_signature,
            source=selected,
        )

    def offer(
        self,
        *,
        provenance: QualificationReuseProvenance,
        constraint_signature: QualificationConstraintSignature,
        incumbent: object,
        accepted: bool,
        fresh_validation_performed: bool,
    ) -> ReusableQualificationIncumbent:
        if accepted is not True or fresh_validation_performed is not True:
            raise ValueError(
                "pool accepts only a fresh validated accepted qualification incumbent"
            )
        reusable = ReusableQualificationIncumbent(
            provenance=provenance,
            constraint_signature=constraint_signature,
            incumbent=incumbent,
            source_accepted=accepted,
            source_fresh_validation_performed=fresh_validation_performed,
        )
        self._entries[self._key(provenance, constraint_signature)] = reusable
        self._event(
            action="incumbent_offered",
            reason="fresh_validated_accepted_source_cached_in_memory",
            target=provenance,
            signature=constraint_signature,
            source=reusable,
        )
        return reusable

    def events_for_case(self, case_id: str) -> tuple[QualificationReuseAuditEvent, ...]:
        return tuple(
            event
            for event in self._audit_events
            if event.target_provenance.case_id == case_id
        )

    def case_audit_payload(self, case_id: str) -> dict[str, Any]:
        case = _require_nonempty(case_id, field_name="case_id")
        entry = {
            "reuse_policy_version": QUALIFICATION_INCUMBENT_REUSE_POLICY_VERSION,
            "development_only": True,
            "qualification_only": True,
            "eligible_for_formal_pool": False,
            "eligible_for_algorithm_comparison": False,
            "case_id": case,
            "audit_events": [
                event.to_payload() for event in self.events_for_case(case)
            ],
        }
        return {**entry, "case_reuse_audit_sha256": canonical_sha256(entry)}

    def to_payload(self) -> dict[str, Any]:
        entries = sorted(
            self._entries.values(),
            key=lambda item: (
                item.provenance.qualification_run_id,
                item.provenance.physical_instance_sha256,
                item.provenance.scale,
                item.provenance.mode,
                item.provenance.demand_scale,
                item.constraint_signature.constraint_signature_sha256,
            ),
        )
        entry = {
            "reuse_policy_version": QUALIFICATION_INCUMBENT_REUSE_POLICY_VERSION,
            "development_only": True,
            "qualification_only": True,
            "eligible_for_formal_pool": False,
            "eligible_for_algorithm_comparison": False,
            "adds_forced_opening_constraint": False,
            "changes_mathematical_model": False,
            "changes_screening": False,
            "concurrency": 1,
            "entry_count": len(entries),
            "entries": [item.metadata_payload() for item in entries],
            "audit_events": [event.to_payload() for event in self._audit_events],
        }
        return {**entry, "reuse_audit_sha256": canonical_sha256(entry)}


__all__ = [
    "ConstraintSignatureExtractionError",
    "DevelopmentQualificationIncumbentPool",
    "QUALIFICATION_CONSTRAINT_SIGNATURE_SCHEMA",
    "QUALIFICATION_INCUMBENT_REUSE_POLICY_VERSION",
    "QualificationConstraintSignature",
    "QualificationReuseAuditEvent",
    "QualificationReuseProvenance",
    "ReusableQualificationIncumbent",
    "build_qualification_constraint_signature",
    "extract_gurobi_constraint_signature",
    "ordered_variable_name_sha256",
]
