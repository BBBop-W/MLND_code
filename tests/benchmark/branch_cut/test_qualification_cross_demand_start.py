from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import pytest

from src.benchmark.branch_cut.calibration_builder import (
    development_calibration_profiles,
)
from src.benchmark.branch_cut.hashing import canonical_sha256
from src.benchmark.branch_cut.qualification import (
    QualificationEvidence,
    QualificationPolicy,
    assess_qualification,
    service_tolerance,
)
from src.benchmark.branch_cut.qualification_cross_demand_start import (
    ALLOWED_CONFIG_CHANGE_PATHS,
    ALLOWED_POSITIVE_SUPPORT_FAMILIES,
    CrossDemandPositiveStartPolicy,
    CrossDemandPositiveStartTransactionError,
    CrossDemandQualificationIdentity,
    PositiveStartAssignment,
    capture_cross_demand_positive_start_source,
    certify_cross_demand_positive_start,
    install_cross_demand_positive_start,
)
from src.benchmark.branch_cut.qualification_runner import (
    CompleteIncumbent,
    QualificationModelAudit,
    QualificationModelContract,
    QualificationRunnerOutcome,
)
from src.config import to_dict


class _FakeGRB:
    UNDEFINED = 1e101

    class Attr:
        VarName = "VarName"
        VType = "VType"
        LB = "LB"
        UB = "UB"
        Obj = "Obj"
        Start = "Start"
        X = "X"


@dataclass
class _FakeVar:
    VarName: str
    VType: str
    LB: float
    UB: float
    Obj: float
    X: float


@dataclass
class _FakeParams:
    StartNumber: int = 0


class _FakeModel:
    def __init__(
        self,
        variables: list[_FakeVar],
        *,
        base_fingerprint: int,
        sol_count: int = 1,
        corrupt_start_readback: bool = False,
        mutate_objective_on_start_write: int | None = None,
        raise_reset_calls: set[int] | None = None,
        raise_set_attr_calls: set[int] | None = None,
        raise_update_calls: set[int] | None = None,
        raise_start_get_calls: set[int] | None = None,
        raise_get_vars_calls: set[int] | None = None,
        objective_value: float = 900.0,
    ) -> None:
        self.variables = list(variables)
        self.base_fingerprint = base_fingerprint
        self.corrupt_start_readback = corrupt_start_readback
        self.mutate_objective_on_start_write = mutate_objective_on_start_write
        self.raise_reset_calls = set(raise_reset_calls or ())
        self.raise_set_attr_calls = set(raise_set_attr_calls or ())
        self.raise_update_calls = set(raise_update_calls or ())
        self.raise_start_get_calls = set(raise_start_get_calls or ())
        self.raise_get_vars_calls = set(raise_get_vars_calls or ())
        self.NumStart = 0
        self.SolCount = sol_count
        self.Params = _FakeParams()
        self.ModelSense = 1
        self.ObjCon = 7.0
        self.ObjVal = objective_value
        self.NumQNZs = 0
        self.NumConstrs = 17
        self.NumQConstrs = 2
        self.NumGenConstrs = 0
        self.NumSOS = 0
        self.NumNLConstrs = 0
        self.start_slots: dict[int, list[float]] = {}
        self.start_write_count = 0
        self.start_read_count = 0
        self.update_count = 0
        self.reset_count = 0
        self.get_vars_count = 0

    @property
    def Fingerprint(self) -> int:
        # A start changes the fake fingerprint, matching the production
        # installer's conservative start-neutral comparison protocol.
        return self.base_fingerprint + int(self.NumStart > 0)

    def getVars(self) -> list[_FakeVar]:
        self.get_vars_count += 1
        if self.get_vars_count in self.raise_get_vars_calls:
            raise RuntimeError(f"fake getVars failure {self.get_vars_count}")
        return list(self.variables)

    def getAttr(
        self,
        attribute: str,
        variables: list[_FakeVar] | tuple[_FakeVar, ...] | None = None,
    ) -> Any:
        if attribute == _FakeGRB.Attr.Start:
            self.start_read_count += 1
            if self.start_read_count in self.raise_start_get_calls:
                raise RuntimeError(
                    f"fake Start getAttr failure {self.start_read_count}"
                )
            values = list(
                self.start_slots.get(
                    self.Params.StartNumber,
                    [_FakeGRB.UNDEFINED] * len(self.variables),
                )
            )
            if self.corrupt_start_readback and values:
                values[0] = 0.0
            return values
        assert variables is not None
        return [getattr(variable, attribute) for variable in variables]

    def setAttr(
        self,
        attribute: str,
        variables: list[_FakeVar] | tuple[_FakeVar, ...],
        values: list[float] | tuple[float, ...],
    ) -> None:
        assert attribute == _FakeGRB.Attr.Start
        assert len(variables) == len(values)
        self.start_write_count += 1
        if self.start_write_count in self.raise_set_attr_calls:
            raise RuntimeError(f"fake setAttr failure {self.start_write_count}")
        self.start_slots[self.Params.StartNumber] = list(values)
        if self.start_write_count == self.mutate_objective_on_start_write:
            self.ObjCon += 1.0

    def update(self) -> None:
        self.update_count += 1
        if self.update_count in self.raise_update_calls:
            raise RuntimeError(f"fake update failure {self.update_count}")
        if self.NumStart == 0:
            self.start_slots.clear()

    def reset(self, clear_all: int) -> None:
        assert clear_all == 0
        self.reset_count += 1
        if self.reset_count in self.raise_reset_calls:
            raise RuntimeError(f"fake reset failure {self.reset_count}")


def _source_values() -> tuple[float, ...]:
    # x[H1] and y are exact zeros and must remain UNDEFINED. Positive gamma
    # and chi are outside the structural allowlist and remain UNDEFINED too.
    return (1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 2.0, 5.0)


def _variables(
    solution_values: tuple[float, ...] | list[float] | None = None,
) -> list[_FakeVar]:
    values = tuple(solution_values or _source_values())
    metadata = (
        ("x[H0]", "B", 0.0, 1.0, 24.0),
        ("x[H1]", "B", 0.0, 1.0, 24.0),
        ("z[H0,0]", "B", 0.0, 1.0, 0.5),
        ("y[H0,H1]", "B", 0.0, 1.0, 0.02),
        ("delta[idle0,2]", "B", 0.0, 1.0, 0.0),
        ("alpha[d0,H0]", "B", 0.0, 1.0, 0.0),
        ("gamma", "I", 0.0, 100.0, 5.0),
        ("chi[o0,d0,H0,H1,2]", "C", 0.0, 200.0, -1.0),
    )
    assert len(values) == len(metadata)
    return [
        _FakeVar(name, variable_type, lower, upper, objective, value)
        for (name, variable_type, lower, upper, objective), value in zip(
            metadata,
            values,
            strict=True,
        )
    ]


def _config(fixed_cost: float, *, feeder_factor: float = 0.35) -> dict[str, Any]:
    return {
        "locker_fixed_cost_per_hour": fixed_cost,
        "feeder_rider_cost_factor": feeder_factor,
        "display_origin_choices": 8,
        "general_corridor_choices": 10,
        "availability_levels": (0.5, 0.8, 0.95),
        "choice": {"locker_asc": -1.0},
    }


def _identity(
    *,
    label: str,
    config: dict[str, Any],
    demand_scale: float,
    mode: str = "MX",
    screening_is_exact: bool = False,
    physical_instance_sha256: str = "a" * 64,
) -> CrossDemandQualificationIdentity:
    return CrossDemandQualificationIdentity(
        qualification_run_id="h23-cross-demand-smoke-run",
        case_id=f"case-{label}",
        profile_id=f"profile-{label}",
        scale="small",
        mode=mode,
        physical_instance_sha256=physical_instance_sha256,
        qualification_case_sha256=("b" if label == "source" else "c") * 64,
        effective_model_config_sha256=canonical_sha256(config),
        demand_scale=demand_scale,
        screening_is_exact=screening_is_exact,
    )


def _qualification_outcome(
    *,
    identity: CrossDemandQualificationIdentity,
    config: dict[str, Any],
    variable_count: int,
    model_fingerprint: int,
    accepted: bool = True,
    fresh_validation_performed: bool = True,
) -> QualificationRunnerOutcome:
    total_demand = 8.0
    evidence = QualificationEvidence(
        case_id=identity.case_id,
        scale=identity.scale,
        courier_only_objective=1000.0,
        witness_objective=900.0,
        total_demand_per_hour=total_demand,
        witness_service_per_hour=2.0,
        open_facilities=4,
        positive_cost_open_facilities=4,
        candidate_h=8,
        open_h=4,
        witness_feasible=True,
        qualification_run_id=identity.qualification_run_id,
        instance_sha256=identity.physical_instance_sha256,
        active_h=2,
        active_h_threshold_per_hour=service_tolerance(total_demand),
        qualification_case_sha256=identity.qualification_case_sha256,
        effective_model_config_sha256=identity.effective_model_config_sha256,
        screening_is_exact=identity.screening_is_exact,
    )
    decision = assess_qualification(evidence, QualificationPolicy())
    assert decision.accepted is True
    rejection_reasons: tuple[str, ...] = ()
    if not accepted:
        rejection_reasons = ("test_runner_rejection",)
        decision = replace(
            decision,
            accepted=False,
            rejection_reasons=rejection_reasons,
        )
    contract = QualificationModelContract(
        qualification_case_sha256=identity.qualification_case_sha256,
        effective_model_config_sha256=identity.effective_model_config_sha256,
        demand_scale=identity.demand_scale,
        locker_fixed_cost_per_hour=config["locker_fixed_cost_per_hour"],
        display_origin_choices=config["display_origin_choices"],
        general_corridor_choices=config["general_corridor_choices"],
        screening_is_exact=identity.screening_is_exact,
    )
    audit = QualificationModelAudit(
        effective_model_config_sha256=identity.effective_model_config_sha256,
        demand_scale=identity.demand_scale,
        locker_fixed_cost_per_hour=config["locker_fixed_cost_per_hour"],
        display_origin_choices=config["display_origin_choices"],
        general_corridor_choices=config["general_corridor_choices"],
        screening_is_exact=identity.screening_is_exact,
        model_fingerprint=model_fingerprint,
    )
    return QualificationRunnerOutcome(
        accepted=accepted,
        rejection_reasons=rejection_reasons,
        evidence=evidence,
        decision=decision,
        required_service_per_hour=service_tolerance(total_demand),
        required_objective_upper_bound=999.0,
        temporary_constraint_used=False,
        temporary_constraint_removed=None,
        fresh_model_validation_performed=fresh_validation_performed,
        witness_status="TIME_LIMIT",
        validation_status="TIME_LIMIT",
        captured_variable_count=variable_count,
        exception_messages=(),
        expected_model_contract=contract,
        witness_model_audit=audit,
        fresh_model_audit=audit,
    )


@dataclass
class _SourceFixture:
    identity: CrossDemandQualificationIdentity
    incumbent: CompleteIncumbent
    model: _FakeModel
    config: dict[str, Any]
    od_demand: dict[tuple[str, str], float]
    outcome: QualificationRunnerOutcome

    def capture(self):
        return capture_cross_demand_positive_start_source(
            identity=self.identity,
            incumbent=self.incumbent,
            source_model=self.model,
            grb=_FakeGRB,
            effective_config=self.config,
            od_demand=self.od_demand,
            qualification_outcome=self.outcome,
        )


def _source_fixture(
    *,
    incumbent_values: tuple[float, ...] | list[float] | None = None,
    model_solution_values: tuple[float, ...] | list[float] | None = None,
    incumbent_fingerprint: int = 101,
    model_fingerprint: int = 101,
    sol_count: int = 1,
    outcome_accepted: bool = True,
    outcome_fresh_validation_performed: bool = True,
    config: dict[str, Any] | None = None,
) -> _SourceFixture:
    source_values = tuple(incumbent_values or _source_values())
    observed_values = tuple(model_solution_values or source_values)
    variables = _variables(observed_values)
    model = _FakeModel(
        variables,
        base_fingerprint=model_fingerprint,
        sol_count=sol_count,
    )
    source_config = config or _config(24.0)
    identity = _identity(
        label="source",
        config=source_config,
        demand_scale=1.0,
    )
    incumbent = CompleteIncumbent.from_ordered_values(
        objective=900.0,
        variable_names=tuple(variable.VarName for variable in variables),
        values=source_values,
        model_fingerprint=incumbent_fingerprint,
    )
    outcome = _qualification_outcome(
        identity=identity,
        config=source_config,
        variable_count=len(variables),
        model_fingerprint=model_fingerprint,
        accepted=outcome_accepted,
        fresh_validation_performed=outcome_fresh_validation_performed,
    )
    return _SourceFixture(
        identity=identity,
        incumbent=incumbent,
        model=model,
        config=source_config,
        od_demand={
            ("o0", "d0"): 3.0,
            ("o1", "d1"): 5.0,
        },
        outcome=outcome,
    )


def _source(**kwargs: Any):
    return _source_fixture(**kwargs).capture()


def _certificate(
    *,
    model: _FakeModel | None = None,
    target_config: dict[str, Any] | None = None,
    target_demand: dict[tuple[str, str], float] | None = None,
    mode: str = "MX",
    screening_is_exact: bool = False,
    physical_instance_sha256: str = "a" * 64,
):
    source = _source()
    config = target_config or _config(48.0)
    target_model = model or _FakeModel(_variables(), base_fingerprint=202)
    target_identity = _identity(
        label="target",
        config=config,
        demand_scale=2.0,
        mode=mode,
        screening_is_exact=screening_is_exact,
        physical_instance_sha256=physical_instance_sha256,
    )
    certificate = certify_cross_demand_positive_start(
        source=source,
        target_identity=target_identity,
        target_model=target_model,
        grb=_FakeGRB,
        target_effective_config=config,
        target_od_demand=target_demand
        or {
            ("o0", "d0"): 6.0,
            ("o1", "d1"): 10.0,
        },
    )
    return source, target_model, certificate


def _seed_start_pool(model: _FakeModel, count: int) -> None:
    assert count >= 1
    model.NumStart = count
    model.start_slots = {index: [1.0] * len(model.variables) for index in range(count)}


def _rehash_tampered_assignment(certificate: Any, *, tamper: str) -> None:
    first = certificate.assignments[0]
    forged = PositiveStartAssignment(
        index=first.index,
        name=first.name,
        family=first.family,
        value=first.value,
    )
    if tamper == "name":
        object.__setattr__(forged, "name", "x[DIFFERENT]")
    elif tamper == "family":
        object.__setattr__(forged, "family", "z")
    elif tamper == "value":
        object.__setattr__(forged, "value", 0.5)
    else:  # pragma: no cover - test helper misuse
        raise AssertionError(tamper)
    assignments = (forged, *certificate.assignments[1:])
    object.__setattr__(certificate, "assignments", assignments)
    object.__setattr__(
        certificate,
        "selected_assignment_sha256",
        canonical_sha256([assignment.to_payload() for assignment in assignments]),
    )
    counts: dict[str, int] = {}
    for assignment in assignments:
        counts[assignment.family] = counts.get(assignment.family, 0) + 1
    object.__setattr__(
        certificate, "family_start_counts", tuple(sorted(counts.items()))
    )
    object.__setattr__(
        certificate,
        "certificate_sha256",
        canonical_sha256(certificate.to_payload(include_certificate_sha256=False)),
    )


def test_policy_is_frozen_qualification_only_and_never_forces_opening() -> None:
    policy = CrossDemandPositiveStartPolicy()

    assert policy.allowed_families == ALLOWED_POSITIVE_SUPPORT_FAMILIES
    assert policy.allowed_config_change_paths == ALLOWED_CONFIG_CHANGE_PATHS
    assert policy.development_only is True
    assert policy.qualification_only is True
    assert policy.eligible_for_formal_pool is False
    assert policy.eligible_for_algorithm_comparison is False
    assert policy.adds_forced_opening_constraint is False
    assert policy.target_fresh_validation_required is True
    with pytest.raises(ValueError, match="family allowlist is frozen"):
        CrossDemandPositiveStartPolicy(allowed_families=("x",))


@pytest.mark.parametrize(
    ("accepted", "fresh", "message"),
    (
        (False, True, "outcome is not accepted"),
        (True, False, "lacks fresh-model validation"),
    ),
)
def test_source_requires_one_accepted_fresh_runner_outcome(
    accepted: bool,
    fresh: bool,
    message: str,
) -> None:
    fixture = _source_fixture(
        outcome_accepted=accepted,
        outcome_fresh_validation_performed=fresh,
    )

    with pytest.raises(ValueError, match=message):
        fixture.capture()


def test_source_rejects_a_boolean_in_place_of_runner_outcome() -> None:
    fixture = _source_fixture()

    with pytest.raises(TypeError, match="QualificationRunnerOutcome"):
        capture_cross_demand_positive_start_source(
            identity=fixture.identity,
            incumbent=fixture.incumbent,
            source_model=fixture.model,
            grb=_FakeGRB,
            effective_config=fixture.config,
            od_demand=fixture.od_demand,
            qualification_outcome=True,  # type: ignore[arg-type]
        )


def test_source_incumbent_fingerprint_must_match_source_model() -> None:
    fixture = _source_fixture(
        incumbent_fingerprint=999,
        model_fingerprint=101,
    )

    with pytest.raises(ValueError, match="fingerprint differs"):
        fixture.capture()


def test_source_model_must_have_an_incumbent() -> None:
    fixture = _source_fixture(sol_count=0)

    with pytest.raises(ValueError, match="has no incumbent"):
        fixture.capture()


def test_source_objective_must_match_live_model_incumbent() -> None:
    fixture = _source_fixture()
    fixture.model.ObjVal = 901.0

    with pytest.raises(ValueError, match="objective differs"):
        fixture.capture()


def test_source_complete_vector_must_equal_source_model_incumbent() -> None:
    observed = list(_source_values())
    observed[0] = 0.0
    fixture = _source_fixture(model_solution_values=observed)

    with pytest.raises(ValueError, match="values differ"):
        fixture.capture()


def test_source_capture_freezes_values_and_full_vector_hash() -> None:
    fixture = _source_fixture()
    source = fixture.capture()
    record_before = source.source_record_sha256
    vector_sha_before = source.complete_incumbent_vector_sha256
    values_before = source.incumbent_values

    fixture.incumbent.values[0] = 0.0
    fixture.incumbent.values[1] = 1.0

    assert source.incumbent_values == values_before == _source_values()
    assert source.source_record_sha256 == record_before
    assert source.complete_incumbent_vector_sha256 == vector_sha_before
    payload = source.metadata_payload()
    assert payload["incumbent_metadata"]["complete_values_persisted_to_audit"] is False
    assert "values" not in payload["incumbent_metadata"]


def test_source_record_hash_binds_the_complete_runner_outcome_payload() -> None:
    first_fixture = _source_fixture()
    first = first_fixture.capture()
    second_fixture = _source_fixture()
    second_fixture.outcome = replace(
        second_fixture.outcome,
        witness_status="OPTIMAL",
    )
    second = second_fixture.capture()

    assert first.qualification_outcome_sha256 != second.qualification_outcome_sha256
    assert first.source_record_sha256 != second.source_record_sha256


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("evidence_run", "evidence identity differs"),
        ("decision_case", "decision identity differs"),
        ("contract_config", "model contract differs"),
        ("witness_fingerprint", "witness model audit differs"),
        ("fresh_fingerprint", "fresh model audit differs"),
    ),
)
def test_source_runner_outcome_identity_mismatch_fails_closed(
    mutation: str,
    message: str,
) -> None:
    fixture = _source_fixture()
    outcome = fixture.outcome
    if mutation == "evidence_run":
        outcome = replace(
            outcome,
            evidence=replace(
                outcome.evidence,
                qualification_run_id="different-run",
            ),
        )
    elif mutation == "decision_case":
        outcome = replace(
            outcome,
            decision=replace(outcome.decision, case_id="different-case"),
        )
    elif mutation == "contract_config":
        assert outcome.expected_model_contract is not None
        outcome = replace(
            outcome,
            expected_model_contract=replace(
                outcome.expected_model_contract,
                effective_model_config_sha256="d" * 64,
            ),
        )
    elif mutation == "witness_fingerprint":
        assert outcome.witness_model_audit is not None
        outcome = replace(
            outcome,
            witness_model_audit=replace(
                outcome.witness_model_audit,
                model_fingerprint=999,
            ),
        )
    else:
        assert outcome.fresh_model_audit is not None
        outcome = replace(
            outcome,
            fresh_model_audit=replace(
                outcome.fresh_model_audit,
                model_fingerprint=999,
            ),
        )
    fixture.outcome = outcome

    with pytest.raises(ValueError, match=message):
        fixture.capture()


def test_source_runner_decision_must_recompute_from_evidence() -> None:
    fixture = _source_fixture()
    fixture.outcome = replace(
        fixture.outcome,
        decision=replace(fixture.outcome.decision, witness_objective=899.0),
    )

    with pytest.raises(ValueError, match="not reproducible"):
        fixture.capture()


def test_certificate_selects_only_positive_allowed_binary_support() -> None:
    source, _model, certificate = _certificate()

    assert certificate.source_record_sha256 == source.source_record_sha256
    assert certificate.pointwise_nondecreasing is True
    assert certificate.strictly_increased_od_count == 2
    assert certificate.minimum_demand_ratio == 2.0
    assert certificate.maximum_demand_ratio == 2.0
    assert [item.name for item in certificate.assignments] == [
        "x[H0]",
        "z[H0,0]",
        "delta[idle0,2]",
        "alpha[d0,H0]",
    ]
    assert {item.value for item in certificate.assignments} == {1.0}
    assert dict(certificate.family_start_counts) == {
        "alpha": 1,
        "delta": 1,
        "x": 1,
        "z": 1,
    }
    assert certificate.undefined_variable_count == 4
    assert certificate.zeros_installed is False
    payload = certificate.to_payload()
    assert payload["adds_forced_opening_constraint"] is False
    assert payload["target_fresh_validation_required"] is True
    assert payload["positive_support"]["zeros_installed"] is False
    assert "assignments" not in payload["positive_support"]


def test_install_uses_undefined_for_every_zero_and_unselected_variable() -> None:
    _source_value, model, certificate = _certificate()

    audit = install_cross_demand_positive_start(
        model=model,
        grb=_FakeGRB,
        certificate=certificate,
    )

    final = model.start_slots[0]
    assert final == [
        1.0,
        _FakeGRB.UNDEFINED,
        1.0,
        _FakeGRB.UNDEFINED,
        1.0,
        1.0,
        _FakeGRB.UNDEFINED,
        _FakeGRB.UNDEFINED,
    ]
    assert 0.0 not in [value for value in final if value != _FakeGRB.UNDEFINED]
    assert model.NumStart == 1
    assert model.reset_count == 2
    assert audit.first_readback_verified is True
    assert audit.neutral_rollback_performed is True
    assert audit.second_readback_verified is True
    assert audit.fingerprint_before == audit.fingerprint_after_neutral_rollback
    assert audit.fingerprint_with_start == audit.fingerprint_with_start_after_reinstall
    assert audit.reinstalled_fingerprint_match is True
    assert audit.constraints_unchanged is True
    assert audit.bounds_unchanged is True
    assert audit.objective_unchanged is True
    assert audit.zeros_installed is False
    assert audit.all_unselected_variables_undefined is True
    assert audit.append_external_start_used is False
    assert audit.hard_constraints_added is False
    assert audit.safety_preserved is True


@pytest.mark.parametrize(
    ("family", "name", "value", "message"),
    (
        ("z", "x[H0]", 1.0, "family does not match"),
        ("x", "x[H0]", 0.0, "value must equal one"),
        ("x", "x[H0]", 0.999, "value must equal one"),
    ),
)
def test_assignment_constructor_rejects_family_name_or_value_drift(
    family: str,
    name: str,
    value: float,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        PositiveStartAssignment(index=0, name=name, family=family, value=value)


@pytest.mark.parametrize(
    ("mode", "screening_is_exact", "message"),
    (
        ("LO", False, "target mode must be MX"),
        ("MX", True, "target screening_is_exact must be false"),
    ),
)
def test_lo_and_exact_screening_fail_closed(
    mode: str,
    screening_is_exact: bool,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _certificate(mode=mode, screening_is_exact=screening_is_exact)


def test_nonmonotone_demand_fails_closed() -> None:
    with pytest.raises(ValueError, match="target demand decreases"):
        _certificate(
            target_demand={
                ("o0", "d0"): 6.0,
                ("o1", "d1"): 4.0,
            }
        )


def test_od_key_drift_fails_closed() -> None:
    with pytest.raises(ValueError, match="OD key sets differ"):
        _certificate(
            target_demand={
                ("o0", "d0"): 6.0,
                ("o2", "d2"): 10.0,
            }
        )


def test_unwhitelisted_configuration_change_fails_closed() -> None:
    with pytest.raises(ValueError, match="outside the allowlist"):
        _certificate(target_config=_config(48.0, feeder_factor=0.4))


def test_unchanged_configuration_is_allowed_by_the_change_allowlist() -> None:
    _source_value, _model, certificate = _certificate(target_config=_config(24.0))

    assert certificate.config_change_records == ()


def test_frozen_h23_profiles_4_to_10_16_22_match_the_config_allowlist() -> None:
    profiles = development_calibration_profiles()
    source_profile = profiles[4]
    source_config = to_dict(source_profile.effective_model_config())
    variables = _variables()
    source_model = _FakeModel(variables, base_fingerprint=101)
    source_identity = _identity(
        label="source",
        config=source_config,
        demand_scale=source_profile.demand_scale,
    )
    incumbent = CompleteIncumbent.from_ordered_values(
        objective=900.0,
        variable_names=tuple(variable.VarName for variable in variables),
        values=_source_values(),
        model_fingerprint=101,
    )
    source = capture_cross_demand_positive_start_source(
        identity=source_identity,
        incumbent=incumbent,
        source_model=source_model,
        grb=_FakeGRB,
        effective_config=source_config,
        od_demand={("o0", "d0"): 3.0, ("o1", "d1"): 5.0},
        qualification_outcome=_qualification_outcome(
            identity=source_identity,
            config=source_config,
            variable_count=len(variables),
            model_fingerprint=101,
        ),
    )

    for index in (10, 16, 22):
        target_profile = profiles[index]
        target_config = to_dict(target_profile.effective_model_config())
        target_scale = target_profile.demand_scale
        certificate = certify_cross_demand_positive_start(
            source=source,
            target_identity=_identity(
                label=f"target-{index}",
                config=target_config,
                demand_scale=target_scale,
            ),
            target_model=_FakeModel(_variables(), base_fingerprint=200 + index),
            grb=_FakeGRB,
            target_effective_config=target_config,
            target_od_demand={
                ("o0", "d0"): 3.0 * target_scale,
                ("o1", "d1"): 5.0 * target_scale,
            },
        )

        assert tuple(record[0] for record in certificate.config_change_records) == (
            "locker_fixed_cost_per_hour",
        )
        assert certificate.minimum_demand_ratio == target_scale
        assert certificate.maximum_demand_ratio == target_scale


@pytest.mark.parametrize("mutation", ("domain", "order", "type"))
def test_variable_name_type_order_domain_mismatch_fails_closed(
    mutation: str,
) -> None:
    variables = _variables()
    if mutation == "domain":
        variables[0] = replace(variables[0], UB=2.0)
    elif mutation == "order":
        variables[0], variables[1] = variables[1], variables[0]
    else:
        variables[0] = replace(variables[0], VType="I")
    model = _FakeModel(variables, base_fingerprint=202)

    with pytest.raises(ValueError, match="variable name/type/order/domain"):
        _certificate(model=model)


def test_physical_instance_mismatch_fails_closed() -> None:
    with pytest.raises(ValueError, match="physical instances differ"):
        _certificate(physical_instance_sha256="d" * 64)


def test_qualification_run_and_scale_mismatch_fail_closed() -> None:
    source, model, certificate = _certificate()
    config = _config(48.0)
    for target, message in (
        (
            replace(
                certificate.target_identity,
                qualification_run_id="different-run",
            ),
            "one qualification run",
        ),
        (replace(certificate.target_identity, scale="medium"), "scale labels differ"),
    ):
        with pytest.raises(ValueError, match=message):
            certify_cross_demand_positive_start(
                source=source,
                target_identity=target,
                target_model=model,
                grb=_FakeGRB,
                target_effective_config=config,
                target_od_demand={
                    ("o0", "d0"): 6.0,
                    ("o1", "d1"): 10.0,
                },
            )


@pytest.mark.parametrize(
    ("mutation", "start_count", "message"),
    (
        ("variable_count", 1, "variable count differs"),
        ("domain", 3, "variable domains drifted"),
        ("fingerprint", 2, "base fingerprint drifted"),
        ("assignment_name", 3, "family/name/value differs"),
    ),
)
def test_preflight_failure_clears_every_preexisting_start(
    mutation: str,
    start_count: int,
    message: str,
) -> None:
    _source_value, model, certificate = _certificate()
    if mutation == "variable_count":
        model.variables.append(_FakeVar("extra", "C", 0.0, 1.0, 0.0, 0.0))
    elif mutation == "domain":
        model.variables[0].UB = 2.0
    elif mutation == "fingerprint":
        model.base_fingerprint += 17
    else:
        _rehash_tampered_assignment(certificate, tamper="name")
    _seed_start_pool(model, start_count)

    with pytest.raises(ValueError, match=message):
        install_cross_demand_positive_start(
            model=model,
            grb=_FakeGRB,
            certificate=certificate,
        )

    assert model.NumStart == 0
    assert model.start_slots == {}


def test_invalid_certificate_type_still_clears_preexisting_starts() -> None:
    model = _FakeModel(_variables(), base_fingerprint=202)
    _seed_start_pool(model, 2)

    with pytest.raises(TypeError, match="certificate must be"):
        install_cross_demand_positive_start(
            model=model,
            grb=_FakeGRB,
            certificate=object(),  # type: ignore[arg-type]
        )

    assert model.NumStart == 0
    assert model.start_slots == {}


@pytest.mark.parametrize("tamper", ("name", "family", "value"))
def test_installer_defends_assignment_name_family_and_value(tamper: str) -> None:
    _source_value, model, certificate = _certificate()
    _rehash_tampered_assignment(certificate, tamper=tamper)
    _seed_start_pool(model, 2)

    with pytest.raises(ValueError, match="family/name/value differs"):
        install_cross_demand_positive_start(
            model=model,
            grb=_FakeGRB,
            certificate=certificate,
        )

    assert model.NumStart == 0
    assert model.start_slots == {}


@pytest.mark.parametrize(
    ("failure_kwargs", "message"),
    (
        ({"raise_reset_calls": {1}}, "fake reset failure 1"),
        ({"raise_set_attr_calls": {1}}, "fake setAttr failure 1"),
        ({"raise_update_calls": {2}}, "fake update failure 2"),
        ({"raise_start_get_calls": {1}}, "fake Start getAttr failure 1"),
        ({"raise_start_get_calls": {2}}, "fake Start getAttr failure 2"),
    ),
)
def test_transaction_faults_roll_back_to_verified_empty_start_pool(
    failure_kwargs: dict[str, Any],
    message: str,
) -> None:
    model = _FakeModel(
        _variables(),
        base_fingerprint=202,
        **failure_kwargs,
    )
    _source_value, _model, certificate = _certificate(model=model)

    with pytest.raises(RuntimeError, match=message):
        install_cross_demand_positive_start(
            model=model,
            grb=_FakeGRB,
            certificate=certificate,
        )

    assert model.NumStart == 0
    assert model.start_slots == {}


def test_primary_and_rollback_errors_are_both_reported_fail_closed() -> None:
    model = _FakeModel(
        _variables(),
        base_fingerprint=202,
        raise_start_get_calls={1},
        # Updates 1--3 are initial clear and first install. Update 4 is the
        # emergency rollback after the injected first-readback failure.
        raise_update_calls={4},
    )
    _source_value, _model, certificate = _certificate(model=model)

    with pytest.raises(CrossDemandPositiveStartTransactionError) as captured:
        install_cross_demand_positive_start(
            model=model,
            grb=_FakeGRB,
            certificate=certificate,
        )

    message = str(captured.value)
    assert "primary=RuntimeError: fake Start getAttr failure 1" in message
    assert "rollback=update: RuntimeError: fake update failure 4" in message
    assert model.NumStart == 0


@pytest.mark.parametrize(
    "attribute",
    ("NumSOS", "NumQNZs", "ModelSense", "ObjCon", "NumStart"),
)
def test_missing_critical_model_attributes_fail_closed_and_clear_starts(
    attribute: str,
) -> None:
    _source_value, model, certificate = _certificate()
    _seed_start_pool(model, 2)
    delattr(model, attribute)

    with pytest.raises((ValueError, CrossDemandPositiveStartTransactionError)):
        install_cross_demand_positive_start(
            model=model,
            grb=_FakeGRB,
            certificate=certificate,
        )

    assert model.NumStart == 0
    assert model.start_slots == {}


def test_readback_failure_rolls_back_every_start() -> None:
    model = _FakeModel(
        _variables(),
        base_fingerprint=202,
        corrupt_start_readback=True,
    )
    _source_value, _model, certificate = _certificate(model=model)

    with pytest.raises(RuntimeError, match="readback mismatch"):
        install_cross_demand_positive_start(
            model=model,
            grb=_FakeGRB,
            certificate=certificate,
        )

    assert model.NumStart == 0
    assert model.start_slots == {}


def test_mathematical_state_mutation_is_detected_and_start_is_rolled_back() -> None:
    model = _FakeModel(
        _variables(),
        base_fingerprint=202,
        mutate_objective_on_start_write=1,
    )
    _source_value, _model, certificate = _certificate(model=model)

    with pytest.raises(RuntimeError, match="mathematical model changed"):
        install_cross_demand_positive_start(
            model=model,
            grb=_FakeGRB,
            certificate=certificate,
        )

    assert model.NumStart == 0
    assert model.start_slots == {}


def test_second_install_state_mutation_is_detected_and_rolled_back() -> None:
    model = _FakeModel(
        _variables(),
        base_fingerprint=202,
        # Write 1 installs, write 2 clears the first slot, write 3 reinstalls.
        mutate_objective_on_start_write=3,
    )
    _source_value, _model, certificate = _certificate(model=model)

    with pytest.raises(RuntimeError, match="mathematical state changed"):
        install_cross_demand_positive_start(
            model=model,
            grb=_FakeGRB,
            certificate=certificate,
        )

    assert model.NumStart == 0
    assert model.start_slots == {}


def test_certificate_rejects_fractional_structural_source_value() -> None:
    bad_values = list(_source_values())
    bad_values[0] = 0.4
    source = _source(incumbent_values=bad_values)
    target_config = _config(48.0)
    target_model = _FakeModel(_variables(), base_fingerprint=202)

    with pytest.raises(ValueError, match="is not binary"):
        certify_cross_demand_positive_start(
            source=source,
            target_identity=_identity(
                label="target",
                config=target_config,
                demand_scale=2.0,
            ),
            target_model=target_model,
            grb=_FakeGRB,
            target_effective_config=target_config,
            target_od_demand={
                ("o0", "d0"): 6.0,
                ("o1", "d1"): 10.0,
            },
        )
