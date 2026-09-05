"""Development-only cross-demand positive-support qualification MIP starts.

This module deliberately does not extend the exact-constraint incumbent pool.
It certifies a narrower, incomplete hint: positive binary support from an
accepted, fresh-validated lower-demand MX incumbent may be offered to a
strictly higher-demand MX qualification profile on the same physical instance.
Only ``x``, ``z``, ``y``, ``delta``, and ``alpha`` may be specified.  Every
zero-valued and every unselected variable is written as ``GRB.UNDEFINED`` so
the target solver remains free to repair or reject the hint.

The mechanism is development/qualification-only.  It neither adds a row nor
forces a facility open, and a target can be accepted only after the existing
fresh-target-model validation path succeeds.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import json
import math
from types import MappingProxyType
from typing import Any

from .hashing import canonical_json, canonical_sha256
from .qualification import assess_qualification
from .qualification_incumbent_reuse import ordered_variable_name_sha256
from .qualification_runner import CompleteIncumbent, QualificationRunnerOutcome


CROSS_DEMAND_POSITIVE_START_POLICY_VERSION = (
    "development-qualification-cross-demand-positive-support-v1"
)
CROSS_DEMAND_POSITIVE_START_CERTIFICATE_SCHEMA = (
    "mlnd-qualification-cross-demand-positive-support-certificate-v1"
)
CROSS_DEMAND_POSITIVE_START_SOURCE_SCHEMA = (
    "mlnd-qualification-cross-demand-positive-support-source-v1"
)
ALLOWED_POSITIVE_SUPPORT_FAMILIES = ("alpha", "delta", "x", "y", "z")
ALLOWED_CONFIG_CHANGE_PATHS = ("locker_fixed_cost_per_hour",)


class CrossDemandPositiveStartTransactionError(RuntimeError):
    """A start transaction failed and its rollback was not fully verified."""


def _require_nonempty(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} must be a nonempty string")
    return value


def _require_sha256(value: object, *, field_name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field_name} must be a lowercase SHA256 digest")
    return value


def _finite(value: object, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be numeric")
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{field_name} must be finite")
    return converted


def _require_model_nonnegative_int(model: object, name: str) -> int:
    """Read one critical native-model count without permissive defaults."""

    try:
        value = getattr(model, name)
    except AttributeError as error:
        raise ValueError(f"critical model attribute {name} is unavailable") from error
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(
            f"critical model attribute {name} must be a nonnegative integer"
        )
    return value


def _complete_incumbent_vector_sha256(
    *, ordered_var_name_sha256: str, values: Sequence[float]
) -> str:
    """Bind every ordered source value without persisting the vector itself."""

    return canonical_sha256(
        {
            "vector_schema": "mlnd-complete-incumbent-vector-v1",
            "ordered_var_name_sha256": ordered_var_name_sha256,
            "values": list(values),
        }
    )


def _normalized_json_mapping(
    value: Mapping[str, Any], *, field_name: str
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise TypeError(f"{field_name} must be a mapping with string keys")
    try:
        return json.loads(canonical_json(dict(value)))
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError(f"{field_name} is not canonical-JSON compatible") from error


def _freeze_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {key: _freeze_json(item) for key, item in sorted(value.items())}
        )
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value


def _flatten_json(value: Any, *, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, Mapping):
        flattened: dict[str, Any] = {}
        for key in sorted(value):
            path = f"{prefix}.{key}" if prefix else key
            flattened.update(_flatten_json(value[key], prefix=path))
        return flattened
    # Frozen source sequences are tuples while normalized target sequences are
    # lists.  Compare their canonical JSON values, not their Python containers.
    return {prefix: json.loads(canonical_json(value))}


def _config_change_records(
    source: Mapping[str, Any], target: Mapping[str, Any]
) -> tuple[tuple[str, Any, Any], ...]:
    source_flat = _flatten_json(source)
    target_flat = _flatten_json(target)
    paths = sorted(set(source_flat) | set(target_flat))
    return tuple(
        (path, source_flat.get(path), target_flat.get(path))
        for path in paths
        if source_flat.get(path) != target_flat.get(path)
        or (path in source_flat) != (path in target_flat)
    )


def _normalize_od_demand(
    demand: Mapping[tuple[str, str], float], *, field_name: str
) -> tuple[tuple[str, str, float], ...]:
    if not isinstance(demand, Mapping) or not demand:
        raise ValueError(f"{field_name} must be a nonempty OD mapping")
    records: list[tuple[str, str, float]] = []
    for raw_key, raw_value in demand.items():
        if (
            not isinstance(raw_key, tuple)
            or len(raw_key) != 2
            or any(not isinstance(item, str) or not item for item in raw_key)
        ):
            raise ValueError(
                f"{field_name} keys must be nonempty (origin, destination)"
            )
        value = _finite(raw_value, field_name=f"{field_name}[{raw_key!r}]")
        if value <= 0.0:
            raise ValueError(f"{field_name} values must be strictly positive")
        records.append((raw_key[0], raw_key[1], value))
    records.sort(key=lambda item: (item[0], item[1]))
    if len({(origin, destination) for origin, destination, _ in records}) != len(
        records
    ):
        raise ValueError(f"{field_name} contains duplicate OD keys")
    return tuple(records)


def _od_payload(records: Sequence[tuple[str, str, float]]) -> list[dict[str, Any]]:
    return [
        {"origin": origin, "destination": destination, "demand": demand}
        for origin, destination, demand in records
    ]


@dataclass(frozen=True, slots=True)
class CrossDemandQualificationIdentity:
    """One source or target profile identity for the development heuristic."""

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

    def __post_init__(self) -> None:
        for name in (
            "qualification_run_id",
            "case_id",
            "profile_id",
            "scale",
            "mode",
        ):
            object.__setattr__(
                self,
                name,
                _require_nonempty(getattr(self, name), field_name=name),
            )
        object.__setattr__(self, "mode", self.mode.upper())
        for name in (
            "physical_instance_sha256",
            "qualification_case_sha256",
            "effective_model_config_sha256",
        ):
            object.__setattr__(
                self,
                name,
                _require_sha256(getattr(self, name), field_name=name),
            )
        demand_scale = _finite(self.demand_scale, field_name="demand_scale")
        if demand_scale <= 0.0:
            raise ValueError("demand_scale must be positive")
        object.__setattr__(self, "demand_scale", demand_scale)
        if type(self.screening_is_exact) is not bool:
            raise ValueError("screening_is_exact must be boolean")

    def to_payload(self) -> dict[str, Any]:
        return {
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
        }


@dataclass(frozen=True, slots=True)
class CrossDemandPositiveStartPolicy:
    """Frozen fail-closed scope of the cross-demand development start."""

    allowed_families: tuple[str, ...] = ALLOWED_POSITIVE_SUPPORT_FAMILIES
    allowed_config_change_paths: tuple[str, ...] = ALLOWED_CONFIG_CHANGE_PATHS
    zero_tolerance: float = 1e-6
    integrality_tolerance: float = 1e-5
    demand_tolerance: float = 1e-12
    development_only: bool = True
    qualification_only: bool = True
    eligible_for_formal_pool: bool = False
    eligible_for_algorithm_comparison: bool = False
    adds_forced_opening_constraint: bool = False
    target_fresh_validation_required: bool = True
    values_persisted_to_audit: bool = False
    policy_version: str = CROSS_DEMAND_POSITIVE_START_POLICY_VERSION

    def __post_init__(self) -> None:
        if self.policy_version != CROSS_DEMAND_POSITIVE_START_POLICY_VERSION:
            raise ValueError("unsupported cross-demand start policy")
        if tuple(self.allowed_families) != ALLOWED_POSITIVE_SUPPORT_FAMILIES:
            raise ValueError("positive-support family allowlist is frozen")
        if tuple(self.allowed_config_change_paths) != ALLOWED_CONFIG_CHANGE_PATHS:
            raise ValueError("configuration-change allowlist is frozen")
        object.__setattr__(self, "allowed_families", tuple(self.allowed_families))
        object.__setattr__(
            self,
            "allowed_config_change_paths",
            tuple(self.allowed_config_change_paths),
        )
        for name in ("zero_tolerance", "integrality_tolerance", "demand_tolerance"):
            value = _finite(getattr(self, name), field_name=name)
            if value <= 0.0:
                raise ValueError(f"{name} must be positive")
        if self.zero_tolerance > self.integrality_tolerance:
            raise ValueError("zero_tolerance cannot exceed integrality_tolerance")
        required_flags = {
            "development_only": True,
            "qualification_only": True,
            "eligible_for_formal_pool": False,
            "eligible_for_algorithm_comparison": False,
            "adds_forced_opening_constraint": False,
            "target_fresh_validation_required": True,
            "values_persisted_to_audit": False,
        }
        for name, expected in required_flags.items():
            if getattr(self, name) is not expected:
                raise ValueError(f"{name} must remain {expected!r}")

    def to_payload(self) -> dict[str, Any]:
        return {
            "policy_version": self.policy_version,
            "allowed_families": list(self.allowed_families),
            "allowed_config_change_paths": list(self.allowed_config_change_paths),
            "zero_tolerance": self.zero_tolerance,
            "integrality_tolerance": self.integrality_tolerance,
            "demand_tolerance": self.demand_tolerance,
            "development_only": self.development_only,
            "qualification_only": self.qualification_only,
            "eligible_for_formal_pool": self.eligible_for_formal_pool,
            "eligible_for_algorithm_comparison": (
                self.eligible_for_algorithm_comparison
            ),
            "adds_forced_opening_constraint": self.adds_forced_opening_constraint,
            "target_fresh_validation_required": (self.target_fresh_validation_required),
            "values_persisted_to_audit": self.values_persisted_to_audit,
        }


@dataclass(frozen=True, slots=True)
class VariableDomainRecord:
    index: int
    name: str
    variable_type: str
    lower_bound: float
    upper_bound: float

    def __post_init__(self) -> None:
        if (
            isinstance(self.index, bool)
            or not isinstance(self.index, int)
            or self.index < 0
        ):
            raise ValueError("variable index must be a nonnegative integer")
        _require_nonempty(self.name, field_name="variable name")
        _require_nonempty(self.variable_type, field_name="variable type")
        lower = _finite(self.lower_bound, field_name=f"{self.name} lower bound")
        upper = _finite(self.upper_bound, field_name=f"{self.name} upper bound")
        if lower > upper:
            raise ValueError("variable lower bound exceeds upper bound")
        object.__setattr__(self, "lower_bound", lower)
        object.__setattr__(self, "upper_bound", upper)

    @property
    def family(self) -> str:
        prefix, separator, _suffix = self.name.partition("[")
        return prefix if separator else self.name

    def to_payload(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "name": self.name,
            "variable_type": self.variable_type,
            "lower_bound": self.lower_bound,
            "upper_bound": self.upper_bound,
        }


def _bulk_attr(
    model: object,
    grb: object,
    name: str,
    variables: Sequence[object],
) -> tuple[Any, ...]:
    attr = getattr(getattr(grb, "Attr"), name)
    values = tuple(model.getAttr(attr, variables))
    if len(values) != len(variables):
        raise ValueError(f"bulk {name} attribute did not cover every variable")
    return values


def extract_variable_domain_records(
    model: object, grb: object
) -> tuple[VariableDomainRecord, ...]:
    """Read exact ordered variable names, types, and finite domains."""

    variables = tuple(model.getVars())
    if not variables:
        raise ValueError("cross-demand start requires a nonempty target model")
    names = _bulk_attr(model, grb, "VarName", variables)
    types = _bulk_attr(model, grb, "VType", variables)
    lower_bounds = _bulk_attr(model, grb, "LB", variables)
    upper_bounds = _bulk_attr(model, grb, "UB", variables)
    records = tuple(
        VariableDomainRecord(
            index=index,
            name=str(name),
            variable_type=str(variable_type),
            lower_bound=lower_bound,
            upper_bound=upper_bound,
        )
        for index, (name, variable_type, lower_bound, upper_bound) in enumerate(
            zip(names, types, lower_bounds, upper_bounds, strict=True)
        )
    )
    if len({record.name for record in records}) != len(records):
        raise ValueError("cross-demand start requires unique variable names")
    return records


def variable_domain_sha256(records: Sequence[VariableDomainRecord]) -> str:
    return canonical_sha256([record.to_payload() for record in records])


@dataclass(frozen=True, slots=True)
class CrossDemandPositiveStartSource:
    """In-memory source envelope created only after fresh acceptance."""

    identity: CrossDemandQualificationIdentity
    incumbent_objective: float
    incumbent_variable_count: int
    incumbent_ordered_var_name_sha256: str
    source_model_fingerprint: int
    incumbent_values: tuple[float, ...] = field(repr=False, compare=False)
    complete_incumbent_vector_sha256: str
    qualification_outcome_sha256: str
    variable_domains: tuple[VariableDomainRecord, ...] = field(
        repr=False,
        compare=False,
    )
    effective_config: Mapping[str, Any] = field(repr=False, compare=False)
    od_demand: tuple[tuple[str, str, float], ...] = field(
        repr=False,
        compare=False,
    )
    source_record_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if self.identity.mode != "MX":
            raise ValueError("cross-demand positive starts support MX only")
        if self.identity.screening_is_exact is not False:
            raise ValueError("cross-demand starts require screening_is_exact=false")
        domains = tuple(self.variable_domains)
        if not domains or tuple(record.index for record in domains) != tuple(
            range(len(domains))
        ):
            raise ValueError("source variable domains must be complete and ordered")
        names = tuple(record.name for record in domains)
        if self.incumbent_variable_count != len(domains):
            raise ValueError(
                "source incumbent variable count differs from source model"
            )
        if self.incumbent_ordered_var_name_sha256 != ordered_variable_name_sha256(
            names
        ):
            raise ValueError("source incumbent ordered names differ from source model")
        if isinstance(self.source_model_fingerprint, bool) or not isinstance(
            self.source_model_fingerprint, int
        ):
            raise ValueError("source model fingerprint must be an integer")
        objective = _finite(
            self.incumbent_objective,
            field_name="source incumbent objective",
        )
        values = tuple(
            _finite(value, field_name="source incumbent value")
            for value in self.incumbent_values
        )
        if len(values) != self.incumbent_variable_count:
            raise ValueError("source incumbent snapshot is incomplete")
        vector_sha256 = _complete_incumbent_vector_sha256(
            ordered_var_name_sha256=self.incumbent_ordered_var_name_sha256,
            values=values,
        )
        if vector_sha256 != _require_sha256(
            self.complete_incumbent_vector_sha256,
            field_name="complete_incumbent_vector_sha256",
        ):
            raise ValueError("source complete-incumbent vector SHA256 is inconsistent")
        _require_sha256(
            self.qualification_outcome_sha256,
            field_name="qualification_outcome_sha256",
        )
        config = _normalized_json_mapping(
            self.effective_config,
            field_name="source effective_config",
        )
        if canonical_sha256(config) != self.identity.effective_model_config_sha256:
            raise ValueError("source effective config SHA256 does not match identity")
        demand = tuple(self.od_demand)
        try:
            normalized_demand = _normalize_od_demand(
                {(origin, destination): value for origin, destination, value in demand},
                field_name="source od_demand",
            )
        except (TypeError, ValueError) as error:
            raise ValueError("source OD demand records are invalid") from error
        if normalized_demand != demand:
            raise ValueError("source OD demand records must be unique and ordered")
        object.__setattr__(self, "variable_domains", domains)
        object.__setattr__(self, "incumbent_objective", objective)
        object.__setattr__(self, "incumbent_values", values)
        object.__setattr__(self, "effective_config", _freeze_json(config))
        object.__setattr__(self, "od_demand", demand)
        object.__setattr__(
            self,
            "source_record_sha256",
            canonical_sha256(self.metadata_payload(include_record_sha256=False)),
        )

    def metadata_payload(self, *, include_record_sha256: bool = True) -> dict[str, Any]:
        payload = {
            "source_schema": CROSS_DEMAND_POSITIVE_START_SOURCE_SCHEMA,
            "policy_version": CROSS_DEMAND_POSITIVE_START_POLICY_VERSION,
            "development_only": True,
            "qualification_only": True,
            "eligible_for_formal_pool": False,
            "eligible_for_algorithm_comparison": False,
            "identity": self.identity.to_payload(),
            "source_accepted": True,
            "source_fresh_validation_performed": True,
            "qualification_outcome_sha256": self.qualification_outcome_sha256,
            "incumbent_metadata": {
                "objective_under_source_profile": self.incumbent_objective,
                "variable_count": self.incumbent_variable_count,
                "ordered_var_name_sha256": (self.incumbent_ordered_var_name_sha256),
                "source_model_fingerprint": self.source_model_fingerprint,
                "complete_incumbent_vector_sha256": (
                    self.complete_incumbent_vector_sha256
                ),
                "complete_values_persisted_to_audit": False,
            },
            "variable_domain_sha256": variable_domain_sha256(self.variable_domains),
            "effective_config_sha256": canonical_sha256(self.effective_config),
            "od_demand_sha256": canonical_sha256(_od_payload(self.od_demand)),
            "od_count": len(self.od_demand),
        }
        if include_record_sha256:
            payload["source_record_sha256"] = self.source_record_sha256
        return payload


def _validate_source_qualification_outcome(
    *,
    outcome: QualificationRunnerOutcome,
    identity: CrossDemandQualificationIdentity,
    incumbent: CompleteIncumbent,
    source_model_fingerprint: int,
    source_effective_config: Mapping[str, Any],
    source_od_demand: Sequence[tuple[str, str, float]],
) -> str:
    """Bind acceptance and fresh validation to one immutable runner outcome."""

    if not isinstance(outcome, QualificationRunnerOutcome):
        raise TypeError("qualification_outcome must be QualificationRunnerOutcome")
    if (
        outcome.accepted is not True
        or outcome.rejection_reasons
        or outcome.decision.accepted is not True
        or outcome.decision.rejection_reasons
        or outcome.evidence.witness_feasible is not True
    ):
        raise ValueError("source qualification outcome is not accepted")
    if outcome.fresh_model_validation_performed is not True:
        raise ValueError("source qualification outcome lacks fresh-model validation")
    if outcome.exception_messages:
        raise ValueError("source qualification outcome contains mechanism exceptions")
    if outcome.captured_variable_count != incumbent.variable_count:
        raise ValueError(
            "source outcome captured-variable count differs from incumbent"
        )
    contract = outcome.expected_model_contract
    witness_audit = outcome.witness_model_audit
    fresh_audit = outcome.fresh_model_audit
    if contract is None or witness_audit is None or fresh_audit is None:
        raise ValueError("source outcome lacks complete model-contract audits")
    try:
        expected_contract_values = (
            identity.effective_model_config_sha256,
            identity.demand_scale,
            _finite(
                source_effective_config["locker_fixed_cost_per_hour"],
                field_name="source locker_fixed_cost_per_hour",
            ),
            source_effective_config["display_origin_choices"],
            source_effective_config["general_corridor_choices"],
            False,
        )
    except KeyError as error:
        raise ValueError(
            "source effective config lacks qualification-contract fields"
        ) from error
    contract_values = (
        contract.effective_model_config_sha256,
        contract.demand_scale,
        contract.locker_fixed_cost_per_hour,
        contract.display_origin_choices,
        contract.general_corridor_choices,
        contract.screening_is_exact,
    )
    if (
        contract.qualification_case_sha256 != identity.qualification_case_sha256
        or contract_values != expected_contract_values
    ):
        raise ValueError("source outcome model contract differs from source identity")
    for audit, label in ((witness_audit, "witness"), (fresh_audit, "fresh")):
        if (
            audit.contract_values() != expected_contract_values
            or audit.model_fingerprint != source_model_fingerprint
        ):
            raise ValueError(f"source {label} model audit differs from source model")
    evidence = outcome.evidence
    decision = outcome.decision
    expected_identity = (
        identity.case_id,
        identity.scale,
        identity.qualification_run_id,
        identity.physical_instance_sha256,
        identity.qualification_case_sha256,
        identity.effective_model_config_sha256,
        False,
    )
    if (
        evidence.case_id,
        evidence.scale,
        evidence.qualification_run_id,
        evidence.instance_sha256,
        evidence.qualification_case_sha256,
        evidence.effective_model_config_sha256,
        evidence.screening_is_exact,
    ) != expected_identity:
        raise ValueError("source outcome evidence identity differs from source")
    if (
        decision.case_id,
        decision.scale,
        decision.qualification_run_id,
        decision.instance_sha256,
        decision.qualification_case_sha256,
        decision.effective_model_config_sha256,
        decision.screening_is_exact,
    ) != expected_identity:
        raise ValueError("source outcome decision identity differs from source")
    recomputed_decision = assess_qualification(evidence, decision.policy)
    if recomputed_decision.to_payload() != decision.to_payload():
        raise ValueError("source outcome decision is not reproducible from evidence")
    expected_total_demand = math.fsum(value for _o, _d, value in source_od_demand)
    if not math.isclose(
        evidence.total_demand_per_hour,
        expected_total_demand,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise ValueError("source outcome total demand differs from source OD demand")
    evidence_objective = _finite(
        evidence.witness_objective,
        field_name="source outcome evidence witness objective",
    )
    decision_objective = _finite(
        decision.witness_objective,
        field_name="source outcome decision witness objective",
    )
    objective_tolerance = 1e-9 * max(
        1.0,
        abs(incumbent.objective),
        abs(evidence_objective),
    )
    if not math.isclose(
        incumbent.objective,
        evidence_objective,
        rel_tol=0.0,
        abs_tol=objective_tolerance,
    ) or not math.isclose(
        incumbent.objective,
        decision_objective,
        rel_tol=0.0,
        abs_tol=objective_tolerance,
    ):
        raise ValueError("source incumbent objective differs from accepted outcome")
    return canonical_sha256(outcome.to_payload())


def capture_cross_demand_positive_start_source(
    *,
    identity: CrossDemandQualificationIdentity,
    incumbent: CompleteIncumbent,
    source_model: object,
    grb: object,
    effective_config: Mapping[str, Any],
    od_demand: Mapping[tuple[str, str], float],
    qualification_outcome: QualificationRunnerOutcome,
) -> CrossDemandPositiveStartSource:
    """Capture one immutable, model-bound, fresh-accepted source vector."""

    if not isinstance(incumbent, CompleteIncumbent):
        raise TypeError("incumbent must be CompleteIncumbent")
    source_fingerprint = _read_fingerprint(source_model)
    if incumbent.model_fingerprint != source_fingerprint:
        raise ValueError(
            "source incumbent fingerprint differs from source model Fingerprint"
        )
    if _require_model_nonnegative_int(source_model, "SolCount") < 1:
        raise ValueError("source model has no incumbent")
    try:
        raw_source_model_objective = getattr(source_model, "ObjVal")
    except AttributeError as error:
        raise ValueError("source model ObjVal is unavailable") from error
    source_model_objective = _finite(
        raw_source_model_objective,
        field_name="source model ObjVal",
    )
    objective_tolerance = 1e-9 * max(
        1.0,
        abs(source_model_objective),
        abs(incumbent.objective),
    )
    if not math.isclose(
        source_model_objective,
        incumbent.objective,
        rel_tol=0.0,
        abs_tol=objective_tolerance,
    ):
        raise ValueError("CompleteIncumbent objective differs from source model ObjVal")
    variables = tuple(source_model.getVars())
    domains = extract_variable_domain_records(source_model, grb)
    observed_values = tuple(
        _finite(value, field_name="source model incumbent value")
        for value in _bulk_attr(source_model, grb, "X", variables)
    )
    snapshot = tuple(float(value) for value in incumbent.values)
    if len(snapshot) != len(observed_values) or any(
        not math.isclose(expected, observed, rel_tol=0.0, abs_tol=0.0)
        for expected, observed in zip(snapshot, observed_values, strict=True)
    ):
        raise ValueError("CompleteIncumbent values differ from source model incumbent")
    normalized_demand = _normalize_od_demand(
        od_demand,
        field_name="source od_demand",
    )
    outcome_sha256 = _validate_source_qualification_outcome(
        outcome=qualification_outcome,
        identity=identity,
        incumbent=incumbent,
        source_model_fingerprint=source_fingerprint,
        source_effective_config=effective_config,
        source_od_demand=normalized_demand,
    )
    vector_sha256 = _complete_incumbent_vector_sha256(
        ordered_var_name_sha256=incumbent.ordered_var_name_sha256,
        values=snapshot,
    )
    return CrossDemandPositiveStartSource(
        identity=identity,
        incumbent_objective=incumbent.objective,
        incumbent_variable_count=incumbent.variable_count,
        incumbent_ordered_var_name_sha256=incumbent.ordered_var_name_sha256,
        source_model_fingerprint=source_fingerprint,
        incumbent_values=snapshot,
        complete_incumbent_vector_sha256=vector_sha256,
        qualification_outcome_sha256=outcome_sha256,
        variable_domains=domains,
        effective_config=effective_config,
        od_demand=normalized_demand,
    )


@dataclass(frozen=True, slots=True)
class PositiveStartAssignment:
    index: int
    name: str
    family: str
    value: float

    def __post_init__(self) -> None:
        if (
            isinstance(self.index, bool)
            or not isinstance(self.index, int)
            or self.index < 0
        ):
            raise ValueError("assignment index must be nonnegative")
        _require_nonempty(self.name, field_name="assignment name")
        if self.family not in ALLOWED_POSITIVE_SUPPORT_FAMILIES:
            raise ValueError("assignment family is not allowed")
        name_family, separator, _suffix = self.name.partition("[")
        if not separator or name_family != self.family:
            raise ValueError("assignment family does not match its variable name")
        value = _finite(self.value, field_name="assignment value")
        if not math.isclose(value, 1.0, rel_tol=0.0, abs_tol=0.0):
            raise ValueError("positive-support assignment value must equal one")
        object.__setattr__(self, "value", 1.0)

    def to_payload(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "name": self.name,
            "family": self.family,
            "value": self.value,
        }


@dataclass(frozen=True, slots=True)
class CrossDemandPositiveStartCertificate:
    source_identity: CrossDemandQualificationIdentity
    target_identity: CrossDemandQualificationIdentity
    source_record_sha256: str
    policy: CrossDemandPositiveStartPolicy
    source_od_demand_sha256: str
    target_od_demand_sha256: str
    od_count: int
    pointwise_nondecreasing: bool
    strictly_increased_od_count: int
    minimum_demand_ratio: float
    maximum_demand_ratio: float
    config_change_records: tuple[tuple[str, Any, Any], ...]
    variable_count: int
    variable_domain_sha256: str
    target_base_model_fingerprint: int
    assignments: tuple[PositiveStartAssignment, ...] = field(
        repr=False,
        compare=False,
    )
    selected_assignment_sha256: str
    family_start_counts: tuple[tuple[str, int], ...]
    undefined_variable_count: int
    zeros_installed: bool = False
    certificate_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.policy, CrossDemandPositiveStartPolicy):
            raise TypeError("policy must be CrossDemandPositiveStartPolicy")
        for identity, label in (
            (self.source_identity, "source"),
            (self.target_identity, "target"),
        ):
            if identity.mode != "MX":
                raise ValueError(f"{label} mode must be MX")
            if identity.screening_is_exact is not False:
                raise ValueError(f"{label} screening_is_exact must be false")
        if (
            self.source_identity.qualification_run_id
            != self.target_identity.qualification_run_id
            or self.source_identity.physical_instance_sha256
            != self.target_identity.physical_instance_sha256
            or self.source_identity.scale != self.target_identity.scale
        ):
            raise ValueError("certificate source/target provenance is incompatible")
        _require_sha256(self.source_record_sha256, field_name="source_record_sha256")
        _require_sha256(
            self.source_od_demand_sha256,
            field_name="source_od_demand_sha256",
        )
        _require_sha256(
            self.target_od_demand_sha256,
            field_name="target_od_demand_sha256",
        )
        _require_sha256(
            self.variable_domain_sha256,
            field_name="variable_domain_sha256",
        )
        _require_sha256(
            self.selected_assignment_sha256,
            field_name="selected_assignment_sha256",
        )
        for name in (
            "od_count",
            "strictly_increased_od_count",
            "variable_count",
            "undefined_variable_count",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if (
            self.od_count < 1
            or self.strictly_increased_od_count < 1
            or self.strictly_increased_od_count > self.od_count
        ):
            raise ValueError("certificate requires a strict cross-demand increase")
        if self.pointwise_nondecreasing is not True:
            raise ValueError("target demand must be pointwise nondecreasing")
        if self.target_identity.demand_scale <= self.source_identity.demand_scale:
            raise ValueError("target demand scale must be strictly larger")
        if self.source_od_demand_sha256 == self.target_od_demand_sha256:
            raise ValueError("source and target demand certificates must differ")
        minimum_ratio = _finite(
            self.minimum_demand_ratio,
            field_name="minimum_demand_ratio",
        )
        maximum_ratio = _finite(
            self.maximum_demand_ratio,
            field_name="maximum_demand_ratio",
        )
        if minimum_ratio < 1.0 or maximum_ratio < minimum_ratio:
            raise ValueError("certificate demand ratios are invalid")
        config_records = tuple(self.config_change_records)
        if len({record[0] for record in config_records}) != len(config_records):
            raise ValueError("configuration change paths must be unique")
        if any(
            not isinstance(record, tuple)
            or len(record) != 3
            or record[0] not in self.policy.allowed_config_change_paths
            or record[1] == record[2]
            for record in config_records
        ):
            raise ValueError("certificate configuration changes violate the allowlist")
        assignments = tuple(self.assignments)
        if not assignments:
            raise ValueError("positive-support certificate has no selected values")
        if tuple(item.index for item in assignments) != tuple(
            sorted(item.index for item in assignments)
        ):
            raise ValueError("positive-support assignments must be index ordered")
        if len({item.index for item in assignments}) != len(assignments):
            raise ValueError("positive-support assignments contain duplicates")
        if any(item.index >= self.variable_count for item in assignments):
            raise ValueError("positive-support assignment index is out of range")
        if self.undefined_variable_count != self.variable_count - len(assignments):
            raise ValueError("undefined variable count is inconsistent")
        if self.zeros_installed is not False:
            raise ValueError("zero-valued starts must remain undefined")
        assignment_payload = [item.to_payload() for item in assignments]
        if canonical_sha256(assignment_payload) != self.selected_assignment_sha256:
            raise ValueError("selected-assignment SHA256 is inconsistent")
        observed_counts: dict[str, int] = {}
        for assignment in assignments:
            observed_counts[assignment.family] = (
                observed_counts.get(assignment.family, 0) + 1
            )
        if tuple(sorted(observed_counts.items())) != tuple(self.family_start_counts):
            raise ValueError("family start counts are inconsistent")
        if isinstance(self.target_base_model_fingerprint, bool) or not isinstance(
            self.target_base_model_fingerprint, int
        ):
            raise ValueError("target base model fingerprint must be an integer")
        object.__setattr__(self, "config_change_records", config_records)
        object.__setattr__(self, "assignments", assignments)
        object.__setattr__(
            self,
            "certificate_sha256",
            canonical_sha256(self.to_payload(include_certificate_sha256=False)),
        )

    def to_payload(self, *, include_certificate_sha256: bool = True) -> dict[str, Any]:
        payload = {
            "certificate_schema": (CROSS_DEMAND_POSITIVE_START_CERTIFICATE_SCHEMA),
            "policy": self.policy.to_payload(),
            "development_only": True,
            "qualification_only": True,
            "eligible_for_formal_pool": False,
            "eligible_for_algorithm_comparison": False,
            "adds_forced_opening_constraint": False,
            "changes_mathematical_model": False,
            "target_fresh_validation_required": True,
            "source_identity": self.source_identity.to_payload(),
            "target_identity": self.target_identity.to_payload(),
            "source_record_sha256": self.source_record_sha256,
            "demand_certificate": {
                "source_od_demand_sha256": self.source_od_demand_sha256,
                "target_od_demand_sha256": self.target_od_demand_sha256,
                "od_count": self.od_count,
                "pointwise_nondecreasing": self.pointwise_nondecreasing,
                "strictly_increased_od_count": self.strictly_increased_od_count,
                "minimum_demand_ratio": self.minimum_demand_ratio,
                "maximum_demand_ratio": self.maximum_demand_ratio,
            },
            "config_change_allowlist": list(self.policy.allowed_config_change_paths),
            "observed_config_change_records": [
                {"path": path, "source": source, "target": target}
                for path, source, target in self.config_change_records
            ],
            "variable_domain_certificate": {
                "variable_count": self.variable_count,
                "variable_domain_sha256": self.variable_domain_sha256,
                "source_target_name_type_order_domain_match": True,
            },
            "target_base_model_fingerprint": self.target_base_model_fingerprint,
            "positive_support": {
                "allowed_families": list(self.policy.allowed_families),
                "selected_variable_count": len(self.assignments),
                "selected_assignment_sha256": self.selected_assignment_sha256,
                "family_start_counts": {
                    name: count for name, count in self.family_start_counts
                },
                "undefined_variable_count": self.undefined_variable_count,
                "zeros_installed": self.zeros_installed,
                "complete_source_values_persisted_to_audit": False,
                "selected_assignments_persisted_to_audit": False,
            },
        }
        if include_certificate_sha256:
            payload["certificate_sha256"] = self.certificate_sha256
        return payload


def _read_fingerprint(model: object) -> int:
    value = getattr(model, "Fingerprint", None)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("model Fingerprint is unavailable or not an integer")
    return value


def certify_cross_demand_positive_start(
    *,
    source: CrossDemandPositiveStartSource,
    target_identity: CrossDemandQualificationIdentity,
    target_model: object,
    grb: object,
    target_effective_config: Mapping[str, Any],
    target_od_demand: Mapping[tuple[str, str], float],
    policy: CrossDemandPositiveStartPolicy = CrossDemandPositiveStartPolicy(),
) -> CrossDemandPositiveStartCertificate:
    """Build a fail-closed certificate and the in-memory positive hint."""

    source_identity = source.identity
    for identity, label in (
        (source_identity, "source"),
        (target_identity, "target"),
    ):
        if identity.mode != "MX":
            raise ValueError(f"{label} mode must be MX")
        if identity.screening_is_exact is not False:
            raise ValueError(f"{label} screening_is_exact must be false")
    if source_identity.qualification_run_id != target_identity.qualification_run_id:
        raise ValueError("source and target must belong to one qualification run")
    if source_identity.physical_instance_sha256 != (
        target_identity.physical_instance_sha256
    ):
        raise ValueError("source and target physical instances differ")
    if source_identity.scale != target_identity.scale:
        raise ValueError("source and target scale labels differ")
    if (
        source_identity.case_id == target_identity.case_id
        or source_identity.profile_id == target_identity.profile_id
        or source_identity.qualification_case_sha256
        == target_identity.qualification_case_sha256
    ):
        raise ValueError("source and target qualification profiles must be distinct")
    if target_identity.demand_scale <= source_identity.demand_scale:
        raise ValueError("target demand scale must be strictly larger than source")

    target_config = _normalized_json_mapping(
        target_effective_config,
        field_name="target effective_config",
    )
    if canonical_sha256(target_config) != (
        target_identity.effective_model_config_sha256
    ):
        raise ValueError("target effective config SHA256 does not match identity")
    config_changes = _config_change_records(source.effective_config, target_config)
    changed_paths = {record[0] for record in config_changes}
    disallowed = changed_paths.difference(policy.allowed_config_change_paths)
    if disallowed:
        raise ValueError(
            "target configuration changed outside the allowlist: "
            + ", ".join(sorted(disallowed))
        )

    target_demand = _normalize_od_demand(
        target_od_demand,
        field_name="target od_demand",
    )
    source_by_od = {
        (origin, destination): demand
        for origin, destination, demand in source.od_demand
    }
    target_by_od = {
        (origin, destination): demand for origin, destination, demand in target_demand
    }
    if set(source_by_od) != set(target_by_od):
        raise ValueError("source and target OD key sets differ")
    ratios: list[float] = []
    strict_count = 0
    for od in sorted(source_by_od):
        source_value = source_by_od[od]
        target_value = target_by_od[od]
        tolerance = policy.demand_tolerance * max(1.0, source_value, target_value)
        if target_value < source_value:
            raise ValueError(f"target demand decreases for OD {od!r}")
        if target_value > source_value + tolerance:
            strict_count += 1
        ratios.append(target_value / source_value)
    if strict_count < 1:
        raise ValueError("target demand is not strictly larger on any OD")

    target_domains = extract_variable_domain_records(target_model, grb)
    if target_domains != source.variable_domains:
        raise ValueError(
            "source and target variable name/type/order/domain records differ"
        )
    target_fingerprint = _read_fingerprint(target_model)
    if _require_model_nonnegative_int(target_model, "NumStart") != 0:
        raise ValueError("certificate construction requires an empty target start pool")

    assignments: list[PositiveStartAssignment] = []
    family_counts: dict[str, int] = {}
    for domain, raw_value in zip(
        source.variable_domains,
        source.incumbent_values,
        strict=True,
    ):
        if domain.family not in policy.allowed_families:
            continue
        if (
            domain.variable_type != "B"
            or not math.isclose(
                domain.lower_bound,
                0.0,
                rel_tol=0.0,
                abs_tol=0.0,
            )
            or not math.isclose(
                domain.upper_bound,
                1.0,
                rel_tol=0.0,
                abs_tol=0.0,
            )
        ):
            raise ValueError(
                f"allowed structural variable {domain.name!r} is not binary [0, 1]"
            )
        value = float(raw_value)
        if abs(value) <= policy.zero_tolerance:
            continue
        if not math.isclose(
            value,
            1.0,
            rel_tol=0.0,
            abs_tol=policy.integrality_tolerance,
        ):
            raise ValueError(
                f"source structural value for {domain.name!r} is not binary"
            )
        assignments.append(
            PositiveStartAssignment(
                index=domain.index,
                name=domain.name,
                family=domain.family,
                value=1.0,
            )
        )
        family_counts[domain.family] = family_counts.get(domain.family, 0) + 1
    if not assignments:
        raise ValueError("source has no positive allowed structural support")
    assignment_tuple = tuple(assignments)
    assignment_sha = canonical_sha256(
        [assignment.to_payload() for assignment in assignment_tuple]
    )
    return CrossDemandPositiveStartCertificate(
        source_identity=source_identity,
        target_identity=target_identity,
        source_record_sha256=source.source_record_sha256,
        policy=policy,
        source_od_demand_sha256=canonical_sha256(_od_payload(source.od_demand)),
        target_od_demand_sha256=canonical_sha256(_od_payload(target_demand)),
        od_count=len(source.od_demand),
        pointwise_nondecreasing=True,
        strictly_increased_od_count=strict_count,
        minimum_demand_ratio=min(ratios),
        maximum_demand_ratio=max(ratios),
        config_change_records=config_changes,
        variable_count=len(target_domains),
        variable_domain_sha256=variable_domain_sha256(target_domains),
        target_base_model_fingerprint=target_fingerprint,
        assignments=assignment_tuple,
        selected_assignment_sha256=assignment_sha,
        family_start_counts=tuple(sorted(family_counts.items())),
        undefined_variable_count=len(target_domains) - len(assignments),
        zeros_installed=False,
    )


def _constraint_counts(model: object) -> tuple[tuple[str, int], ...]:
    names = (
        "NumConstrs",
        "NumQConstrs",
        "NumGenConstrs",
        "NumSOS",
        "NumNLConstrs",
    )
    counts: list[tuple[str, int]] = []
    for name in names:
        counts.append((name, _require_model_nonnegative_int(model, name)))
    return tuple(counts)


def _objective_sha256(
    model: object,
    grb: object,
    variables: Sequence[object],
) -> str:
    quadratic_nonzeros = _require_model_nonnegative_int(model, "NumQNZs")
    if quadratic_nonzeros:
        raise ValueError("quadratic objectives are unsupported by start audit v1")
    coefficients = tuple(
        _finite(value, field_name="objective coefficient")
        for value in _bulk_attr(model, grb, "Obj", variables)
    )
    try:
        model_sense = getattr(model, "ModelSense")
    except AttributeError as error:
        raise ValueError(
            "critical model attribute ModelSense is unavailable"
        ) from error
    if isinstance(model_sense, bool) or not isinstance(model_sense, int):
        raise ValueError("ModelSense must be an integer")
    try:
        raw_objective_constant = getattr(model, "ObjCon")
    except AttributeError as error:
        raise ValueError("critical model attribute ObjCon is unavailable") from error
    objective_constant = _finite(raw_objective_constant, field_name="ObjCon")
    return canonical_sha256(
        {
            "model_sense": model_sense,
            "objective_constant": objective_constant,
            "linear_coefficients": list(coefficients),
            "quadratic_nonzero_count": quadratic_nonzeros,
        }
    )


@dataclass(frozen=True, slots=True)
class _MathematicalState:
    fingerprint: int
    constraint_counts: tuple[tuple[str, int], ...]
    variable_domain_sha256: str
    objective_sha256: str


def _mathematical_state(model: object, grb: object) -> _MathematicalState:
    variables = tuple(model.getVars())
    domains = extract_variable_domain_records(model, grb)
    return _MathematicalState(
        fingerprint=_read_fingerprint(model),
        constraint_counts=_constraint_counts(model),
        variable_domain_sha256=variable_domain_sha256(domains),
        objective_sha256=_objective_sha256(model, grb, variables),
    )


def _clear_all_starts(model: object, grb: object) -> int:
    model.reset(0)
    variables = tuple(model.getVars())
    existing = _require_model_nonnegative_int(model, "NumStart")
    if existing and variables:
        undefined = [getattr(grb, "UNDEFINED")] * len(variables)
        for start_number in range(existing):
            model.Params.StartNumber = start_number
            model.setAttr(getattr(grb.Attr, "Start"), variables, undefined)
    model.NumStart = 0
    model.update()
    if _require_model_nonnegative_int(model, "NumStart") != 0:
        raise RuntimeError("start pool clear did not leave NumStart=0")
    return len(variables)


def _best_effort_clear_and_verify_starts(
    model: object,
    grb: object,
) -> tuple[str, ...]:
    """Attempt every rollback component and verify the final native count."""

    failures: list[str] = []

    def attempt(label: str, action: Any) -> Any:
        try:
            return action()
        except BaseException as error:
            failures.append(f"{label}: {type(error).__name__}: {error}")
            return None

    attempt("reset", lambda: model.reset(0))
    variables = attempt("getVars", lambda: tuple(model.getVars()))
    existing = attempt(
        "read_NumStart",
        lambda: _require_model_nonnegative_int(model, "NumStart"),
    )
    undefined = attempt("read_GRB_UNDEFINED", lambda: getattr(grb, "UNDEFINED"))
    start_attr = attempt("read_GRB_Attr_Start", lambda: getattr(grb.Attr, "Start"))
    params = attempt("read_Params", lambda: getattr(model, "Params"))
    if (
        isinstance(variables, tuple)
        and isinstance(existing, int)
        and existing > 0
        and undefined is not None
        and start_attr is not None
        and params is not None
    ):
        values = [undefined] * len(variables)
        for start_number in range(existing):
            attempt(
                f"select_start_{start_number}",
                lambda number=start_number: setattr(params, "StartNumber", number),
            )
            attempt(
                f"undefine_start_{start_number}",
                lambda: model.setAttr(start_attr, variables, values),
            )
    attempt("set_NumStart_zero", lambda: setattr(model, "NumStart", 0))
    attempt("update", model.update)
    observed = attempt(
        "verify_NumStart_zero",
        lambda: _require_model_nonnegative_int(model, "NumStart"),
    )
    if observed != 0:
        failures.append(f"verify_NumStart_zero: observed {observed!r}, expected 0")
    return tuple(failures)


def _raise_after_transaction_rollback(
    primary_error: BaseException,
    rollback_failures: Sequence[str],
) -> None:
    if rollback_failures:
        rollback_message = " | ".join(rollback_failures)
        if isinstance(primary_error, Exception):
            raise CrossDemandPositiveStartTransactionError(
                "cross-demand start transaction failed and rollback was not "
                f"clean; primary={type(primary_error).__name__}: {primary_error}; "
                f"rollback={rollback_message}"
            ) from primary_error
        primary_error.add_note(
            "cross-demand start rollback was not clean: " + rollback_message
        )
    raise primary_error


def _readback_matches(
    observed: Sequence[float],
    expected: Sequence[float],
    *,
    undefined: float,
) -> bool:
    if len(observed) != len(expected):
        return False
    for observed_value, expected_value in zip(observed, expected, strict=True):
        observed_float = float(observed_value)
        expected_float = float(expected_value)
        if expected_float == undefined:
            if observed_float != undefined:
                return False
        elif not math.isclose(
            observed_float,
            expected_float,
            rel_tol=0.0,
            abs_tol=0.0,
        ):
            return False
    return True


@dataclass(frozen=True, slots=True)
class CrossDemandPositiveStartInstallAudit:
    certificate_sha256: str
    reset_performed: bool
    initial_start_count: int
    selected_variable_count: int
    undefined_variable_count: int
    family_start_counts: tuple[tuple[str, int], ...]
    first_readback_verified: bool
    neutral_rollback_performed: bool
    second_readback_verified: bool
    fingerprint_before: int
    fingerprint_with_start: int
    fingerprint_with_start_after_reinstall: int
    fingerprint_after_neutral_rollback: int
    fingerprint_unchanged: bool
    reinstalled_fingerprint_match: bool
    constraint_counts_before: tuple[tuple[str, int], ...]
    constraint_counts_after: tuple[tuple[str, int], ...]
    constraints_unchanged: bool
    variable_domain_sha256_before: str
    variable_domain_sha256_after: str
    bounds_unchanged: bool
    objective_sha256_before: str
    objective_sha256_after: str
    objective_unchanged: bool
    zeros_installed: bool
    all_unselected_variables_undefined: bool
    append_external_start_used: bool = False
    hard_constraints_added: bool = False

    @property
    def safety_preserved(self) -> bool:
        return all(
            (
                self.reset_performed,
                self.first_readback_verified,
                self.neutral_rollback_performed,
                self.second_readback_verified,
                self.fingerprint_unchanged,
                self.reinstalled_fingerprint_match,
                self.constraints_unchanged,
                self.bounds_unchanged,
                self.objective_unchanged,
                not self.zeros_installed,
                self.all_unselected_variables_undefined,
                not self.append_external_start_used,
                not self.hard_constraints_added,
            )
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "certificate_sha256": self.certificate_sha256,
            "installation_api": "bulk_GRB_Attr_Start",
            "reset_performed": self.reset_performed,
            "initial_start_count": self.initial_start_count,
            "selected_variable_count": self.selected_variable_count,
            "undefined_variable_count": self.undefined_variable_count,
            "family_start_counts": {
                name: count for name, count in self.family_start_counts
            },
            "first_readback_verified": self.first_readback_verified,
            "neutral_rollback_performed": self.neutral_rollback_performed,
            "second_readback_verified": self.second_readback_verified,
            "fingerprint_before": self.fingerprint_before,
            "fingerprint_with_start": self.fingerprint_with_start,
            "fingerprint_with_start_after_reinstall": (
                self.fingerprint_with_start_after_reinstall
            ),
            "fingerprint_after_neutral_rollback": (
                self.fingerprint_after_neutral_rollback
            ),
            "fingerprint_unchanged": self.fingerprint_unchanged,
            "reinstalled_fingerprint_match": self.reinstalled_fingerprint_match,
            "constraint_counts_before": dict(self.constraint_counts_before),
            "constraint_counts_after": dict(self.constraint_counts_after),
            "constraints_unchanged": self.constraints_unchanged,
            "variable_domain_sha256_before": self.variable_domain_sha256_before,
            "variable_domain_sha256_after": self.variable_domain_sha256_after,
            "bounds_unchanged": self.bounds_unchanged,
            "objective_sha256_before": self.objective_sha256_before,
            "objective_sha256_after": self.objective_sha256_after,
            "objective_unchanged": self.objective_unchanged,
            "zeros_installed": self.zeros_installed,
            "all_unselected_variables_undefined": (
                self.all_unselected_variables_undefined
            ),
            "append_external_start_used": self.append_external_start_used,
            "hard_constraints_added": self.hard_constraints_added,
            "safety_preserved": self.safety_preserved,
        }


def install_cross_demand_positive_start(
    *,
    model: object,
    grb: object,
    certificate: CrossDemandPositiveStartCertificate,
) -> CrossDemandPositiveStartInstallAudit:
    """Transactionally install, read back, neutralize, audit, and reinstall."""

    try:
        if not isinstance(certificate, CrossDemandPositiveStartCertificate):
            raise TypeError("certificate must be CrossDemandPositiveStartCertificate")
        variables = tuple(model.getVars())
        initial_start_count = _require_model_nonnegative_int(model, "NumStart")
        if len(variables) != certificate.variable_count:
            raise ValueError("target variable count differs from certificate")
        current_domains = extract_variable_domain_records(model, grb)
        if (
            variable_domain_sha256(current_domains)
            != certificate.variable_domain_sha256
        ):
            raise ValueError("target variable domains drifted after certification")
        if (
            canonical_sha256(
                [assignment.to_payload() for assignment in certificate.assignments]
            )
            != certificate.selected_assignment_sha256
        ):
            raise ValueError("certificate runtime assignments failed their SHA256")
        if (
            canonical_sha256(certificate.to_payload(include_certificate_sha256=False))
            != certificate.certificate_sha256
        ):
            raise ValueError("certificate metadata failed its SHA256")
        undefined = _finite(
            getattr(grb, "UNDEFINED"),
            field_name="GRB.UNDEFINED",
        )
        if undefined == 0.0:
            raise ValueError("GRB.UNDEFINED cannot equal zero")
        assignment_by_index: dict[int, PositiveStartAssignment] = {}
        for assignment in certificate.assignments:
            if assignment.index in assignment_by_index:
                raise ValueError("certificate contains duplicate assignment indices")
            if assignment.index < 0 or assignment.index >= len(current_domains):
                raise ValueError("certificate assignment index is out of range")
            domain = current_domains[assignment.index]
            if (
                assignment.family not in certificate.policy.allowed_families
                or assignment.family != domain.family
                or assignment.name != domain.name
                or not math.isclose(
                    assignment.value,
                    1.0,
                    rel_tol=0.0,
                    abs_tol=0.0,
                )
            ):
                raise ValueError(
                    "certificate assignment family/name/value differs from target"
                )
            assignment_by_index[assignment.index] = assignment
        if len(assignment_by_index) != len(certificate.assignments):
            raise ValueError("certificate assignment count is inconsistent")
        start_values = [undefined] * len(variables)
        for index, assignment in assignment_by_index.items():
            start_values[index] = assignment.value
        if any(value == 0.0 for value in start_values if value != undefined):
            raise ValueError("zero-valued structural starts are forbidden")

        _clear_all_starts(model, grb)
        before = _mathematical_state(model, grb)
        if before.fingerprint != certificate.target_base_model_fingerprint:
            raise ValueError("target base fingerprint drifted after certification")

        def install_and_verify() -> None:
            model.NumStart = 1
            model.update()
            model.Params.StartNumber = 0
            model.setAttr(grb.Attr.Start, variables, start_values)
            model.update()
            if _require_model_nonnegative_int(model, "NumStart") != 1:
                raise RuntimeError("partial start did not leave exactly one start")
            model.Params.StartNumber = 0
            observed = tuple(model.getAttr(grb.Attr.Start, variables))
            if not _readback_matches(observed, start_values, undefined=undefined):
                raise RuntimeError("cross-demand partial-start readback mismatch")

        install_and_verify()
        fingerprint_with_start = _read_fingerprint(model)
        if _clear_all_starts(model, grb) != len(variables):
            raise RuntimeError("target variable count changed during rollback")
        after_neutral = _mathematical_state(model, grb)
        if before != after_neutral:
            raise RuntimeError(
                "mathematical model changed during cross-demand start transaction"
            )
        install_and_verify()
        final_state = _mathematical_state(model, grb)
        if (
            final_state.fingerprint != fingerprint_with_start
            or final_state.constraint_counts != before.constraint_counts
            or final_state.variable_domain_sha256 != before.variable_domain_sha256
            or final_state.objective_sha256 != before.objective_sha256
        ):
            raise RuntimeError(
                "target mathematical state changed after final start installation"
            )
        audit = CrossDemandPositiveStartInstallAudit(
            certificate_sha256=certificate.certificate_sha256,
            reset_performed=True,
            initial_start_count=initial_start_count,
            selected_variable_count=len(certificate.assignments),
            undefined_variable_count=certificate.undefined_variable_count,
            family_start_counts=certificate.family_start_counts,
            first_readback_verified=True,
            neutral_rollback_performed=True,
            second_readback_verified=True,
            fingerprint_before=before.fingerprint,
            fingerprint_with_start=fingerprint_with_start,
            fingerprint_with_start_after_reinstall=final_state.fingerprint,
            fingerprint_after_neutral_rollback=after_neutral.fingerprint,
            fingerprint_unchanged=(before.fingerprint == after_neutral.fingerprint),
            reinstalled_fingerprint_match=(
                fingerprint_with_start == final_state.fingerprint
            ),
            constraint_counts_before=before.constraint_counts,
            constraint_counts_after=after_neutral.constraint_counts,
            constraints_unchanged=(
                before.constraint_counts == after_neutral.constraint_counts
            ),
            variable_domain_sha256_before=before.variable_domain_sha256,
            variable_domain_sha256_after=after_neutral.variable_domain_sha256,
            bounds_unchanged=(
                before.variable_domain_sha256 == after_neutral.variable_domain_sha256
            ),
            objective_sha256_before=before.objective_sha256,
            objective_sha256_after=after_neutral.objective_sha256,
            objective_unchanged=(
                before.objective_sha256 == after_neutral.objective_sha256
            ),
            zeros_installed=False,
            all_unselected_variables_undefined=True,
            append_external_start_used=False,
            hard_constraints_added=False,
        )
        if audit.safety_preserved is not True:
            raise RuntimeError("cross-demand start audit did not preserve safety")
        return audit
    except BaseException as transaction_error:
        rollback_failures = _best_effort_clear_and_verify_starts(model, grb)
        _raise_after_transaction_rollback(transaction_error, rollback_failures)


__all__ = [
    "ALLOWED_CONFIG_CHANGE_PATHS",
    "ALLOWED_POSITIVE_SUPPORT_FAMILIES",
    "CROSS_DEMAND_POSITIVE_START_CERTIFICATE_SCHEMA",
    "CROSS_DEMAND_POSITIVE_START_POLICY_VERSION",
    "CROSS_DEMAND_POSITIVE_START_SOURCE_SCHEMA",
    "CrossDemandPositiveStartCertificate",
    "CrossDemandPositiveStartInstallAudit",
    "CrossDemandPositiveStartPolicy",
    "CrossDemandPositiveStartSource",
    "CrossDemandPositiveStartTransactionError",
    "CrossDemandQualificationIdentity",
    "PositiveStartAssignment",
    "VariableDomainRecord",
    "capture_cross_demand_positive_start_source",
    "certify_cross_demand_positive_start",
    "extract_variable_domain_records",
    "install_cross_demand_positive_start",
    "variable_domain_sha256",
]
