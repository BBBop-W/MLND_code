from __future__ import annotations

from array import array
from collections.abc import Iterable
from dataclasses import dataclass, field, replace
import json
from types import SimpleNamespace

import pytest

from src.benchmark.branch_cut.qualification import (
    QualificationPolicy,
    QualificationReason,
)
from src.benchmark.branch_cut.hashing import canonical_json, canonical_sha256
from src.benchmark.branch_cut.qualification_geometry_start import (
    GEOMETRY_START_PLAN_VERSION,
    GeometryStartInfeasibilityError,
    GeometryStartInstallAudit,
    GeometryStartPlanningError,
    GeometryStartPolicy,
)
from src.benchmark.branch_cut.qualification_runner import (
    CompleteIncumbent,
    MLNDQualificationAdapter,
    NativeQualificationSettings,
    QualificationInfrastructureError,
    QualificationModelAudit,
    QualificationModelContract,
    QualificationRunnerReason,
    QualificationSolution,
    TemporaryWitnessConstraintToken,
    qualify_canonical_instance,
)
from src.config import ModelConfig, to_dict


_INSTANCE_SHA256 = "a" * 64
_MODEL_CONFIG_SHA256 = "b" * 64
_QUALIFICATION_CASE_SHA256 = "c" * 64


def _model_contract() -> QualificationModelContract:
    return QualificationModelContract(
        qualification_case_sha256=_QUALIFICATION_CASE_SHA256,
        effective_model_config_sha256=_MODEL_CONFIG_SHA256,
        demand_scale=1.0,
        locker_fixed_cost_per_hour=24.0,
        display_origin_choices=8,
        general_corridor_choices=10,
        screening_is_exact=False,
    )


def _model_audit(
    *,
    model_config_sha256: str = _MODEL_CONFIG_SHA256,
    demand_scale: float = 1.0,
    locker_fixed_cost_per_hour: float = 24.0,
    display_origin_choices: int | None = 8,
    general_corridor_choices: int | None = 10,
    screening_is_exact: bool = False,
    model_fingerprint: int | None = 12345,
) -> QualificationModelAudit:
    return QualificationModelAudit(
        effective_model_config_sha256=model_config_sha256,
        demand_scale=demand_scale,
        locker_fixed_cost_per_hour=locker_fixed_cost_per_hour,
        display_origin_choices=display_origin_choices,
        general_corridor_choices=general_corridor_choices,
        screening_is_exact=screening_is_exact,
        model_fingerprint=model_fingerprint,
    )


def _courier_solution() -> QualificationSolution:
    return QualificationSolution(
        status="OPTIMAL",
        has_incumbent=True,
        objective=1_000.0,
        realized_service_per_hour=0.0,
        candidate_h=20,
    )


def _positive_solution() -> QualificationSolution:
    return QualificationSolution(
        status="TIME_LIMIT",
        has_incumbent=True,
        objective=900.0,
        realized_service_per_hour=20.0,
        open_facilities=10,
        positive_cost_open_facilities=10,
        candidate_h=20,
        open_h=8,
        active_h=8,
        active_h_threshold_per_hour=0.001,
    )


def _positive_but_not_improving_solution() -> QualificationSolution:
    return QualificationSolution(
        status="TIME_LIMIT",
        has_incumbent=True,
        objective=1_000.0,
        realized_service_per_hour=20.0,
        open_facilities=10,
        positive_cost_open_facilities=10,
        candidate_h=20,
        open_h=8,
        active_h=8,
        active_h_threshold_per_hour=0.001,
    )


def _no_incumbent() -> QualificationSolution:
    return QualificationSolution(
        status="TIME_LIMIT",
        has_incumbent=False,
        candidate_h=20,
    )


@dataclass
class FakeQualificationAdapter:
    solve_results: list[QualificationSolution | BaseException]
    courier_objective: float = 1_000.0
    total_demand: float = 100.0
    candidate_h: int = 20
    baseline_error: BaseException | None = None
    extract_error: BaseException | None = None
    removal_error: BaseException | None = None
    capture_error: BaseException | None = None
    install_error: BaseException | None = None
    dispose_error: BaseException | None = None
    audit_result: QualificationModelAudit | BaseException | None = None
    events: list[str] = field(default_factory=list)
    settings_seen: list[NativeQualificationSettings] = field(default_factory=list)
    temporary_active: bool = False
    temporary_threshold: float | None = None
    temporary_objective_upper_bound: float | None = None
    current_solution: QualificationSolution | None = None
    installed_incumbent: CompleteIncumbent | None = None
    disposed: bool = False

    def qualification_model_audit(self) -> QualificationModelAudit:
        self.events.append("qualification_model_audit")
        if isinstance(self.audit_result, BaseException):
            raise self.audit_result
        if self.audit_result is None:
            raise AssertionError("unexpected qualification model audit")
        return self.audit_result

    def courier_only_objective(self) -> float:
        self.events.append("courier_only_objective")
        if self.baseline_error is not None:
            raise self.baseline_error
        return self.courier_objective

    def total_demand_per_hour(self) -> float:
        self.events.append("total_demand_per_hour")
        return self.total_demand

    def candidate_h_count(self) -> int:
        self.events.append("candidate_h_count")
        return self.candidate_h

    def clear_mip_starts(self) -> None:
        self.events.append("clear_mip_starts")

    def solve_native(self, settings: NativeQualificationSettings) -> str:
        self.events.append("solve_native")
        self.settings_seen.append(settings)
        if not self.solve_results:
            raise AssertionError("unexpected extra native solve")
        result = self.solve_results.pop(0)
        if isinstance(result, BaseException):
            raise result
        self.current_solution = result
        return result.status

    def extract_solution(self) -> QualificationSolution:
        self.events.append("extract_solution")
        if self.extract_error is not None:
            raise self.extract_error
        if self.current_solution is None:
            raise AssertionError("extract_solution called before a successful solve")
        return self.current_solution

    def add_witness_search_constraints(
        self,
        minimum_service_per_hour: float,
        maximum_objective: float,
    ) -> object:
        self.events.append("add_witness_search_constraints")
        if self.temporary_active:
            raise AssertionError("only one temporary constraint is allowed")
        self.temporary_active = True
        self.temporary_threshold = minimum_service_per_hour
        self.temporary_objective_upper_bound = maximum_objective
        return self

    def remove_temporary_constraint(self, token: object) -> None:
        self.events.append("remove_temporary_constraint")
        if token is not self or not self.temporary_active:
            raise AssertionError("temporary constraint token mismatch")
        if self.removal_error is not None:
            raise self.removal_error
        self.temporary_active = False

    def capture_complete_incumbent(self) -> CompleteIncumbent:
        self.events.append("capture_complete_incumbent")
        if self.capture_error is not None:
            raise self.capture_error
        if self.current_solution is None or not self.current_solution.has_incumbent:
            raise AssertionError("there is no incumbent to capture")
        return CompleteIncumbent.from_ordered_values(
            objective=float(self.current_solution.objective),
            variable_names=("x[H0]", "chi[o,d,i,j,0]", "queue_t[H0]"),
            values=(1.0, 25.0, 2.0),
        )

    def install_complete_incumbent(self, incumbent: CompleteIncumbent) -> None:
        self.events.append("install_complete_incumbent")
        if self.install_error is not None:
            raise self.install_error
        if incumbent.variable_count != 3 or tuple(incumbent.values) != (
            1.0,
            25.0,
            2.0,
        ):
            raise AssertionError("the transferred start is not complete")
        self.installed_incumbent = incumbent

    def dispose(self) -> None:
        self.events.append("dispose")
        if self.dispose_error is not None:
            raise self.dispose_error
        self.disposed = True


@dataclass
class FakeAdapterFactory:
    adapters: list[FakeQualificationAdapter]
    calls: list[tuple[object, str]] = field(default_factory=list)

    def __call__(self, instance: object, mode: str) -> FakeQualificationAdapter:
        self.calls.append((instance, mode))
        if not self.adapters:
            raise AssertionError("unexpected extra adapter construction")
        return self.adapters.pop(0)


def _run(
    adapters: Iterable[FakeQualificationAdapter],
    *,
    instance: object | None = None,
    expected_model_contract: QualificationModelContract | None = None,
) -> tuple[object, FakeAdapterFactory]:
    factory = FakeAdapterFactory(list(adapters))
    outcome = qualify_canonical_instance(
        object() if instance is None else instance,
        "MX",
        case_id="small-b0-s11",
        scale="small",
        qualification_run_id="qualification-test",
        instance_sha256=_INSTANCE_SHA256,
        adapter_factory=factory,
        expected_model_contract=expected_model_contract,
        fail_fast_infrastructure=False,
    )
    return outcome, factory


def test_temporary_witness_is_removed_then_validated_in_fresh_model() -> None:
    witness = FakeQualificationAdapter([_courier_solution(), _positive_solution()])
    validation = FakeQualificationAdapter([_positive_solution()])

    outcome, factory = _run([witness, validation])

    assert outcome.accepted
    assert outcome.decision.accepted
    assert outcome.rejection_reasons == ()
    assert outcome.temporary_constraint_used
    assert outcome.temporary_constraint_removed is True
    assert outcome.feasibility_objective_used
    assert outcome.fresh_model_validation_performed
    assert not witness.temporary_active
    assert witness.disposed
    assert witness.temporary_threshold == pytest.approx(0.1)
    assert witness.temporary_objective_upper_bound == pytest.approx(999.0)
    assert outcome.required_objective_upper_bound == pytest.approx(999.0)
    assert outcome.evidence.active_h_threshold_per_hour == pytest.approx(0.001)
    assert outcome.captured_variable_count == 3
    assert outcome.phase_timings.complete_incumbent_capture_seconds >= 0.0
    assert outcome.phase_timings.complete_start_installation_seconds >= 0.0
    assert outcome.phase_timings.fresh_model_disposal_seconds >= 0.0
    assert outcome.phase_timings.total_seconds >= (
        outcome.phase_timings.complete_incumbent_capture_seconds
    )
    timing_payload = outcome.to_payload()["phase_timings"]
    assert outcome.to_payload()["feasibility_objective_used"] is True
    assert isinstance(timing_payload, dict)
    assert {
        "witness_adapter_initialization_seconds",
        "native_witness_solve_seconds",
        "temporary_constraint_add_seconds",
        "constrained_witness_solve_seconds",
        "complete_incumbent_capture_seconds",
        "witness_model_disposal_seconds",
        "fresh_adapter_initialization_seconds",
        "complete_start_installation_seconds",
        "fresh_validation_solve_seconds",
        "fresh_model_disposal_seconds",
        "total_seconds",
    } <= timing_payload.keys()
    assert validation.installed_incumbent is not None
    assert validation.installed_incumbent.objective == pytest.approx(900.0)
    assert validation.disposed
    assert len(factory.calls) == 2
    assert factory.calls[0] == factory.calls[1]
    assert witness is not validation
    assert witness.events.index("capture_complete_incumbent") < witness.events.index(
        "remove_temporary_constraint"
    )
    assert validation.events.index("install_complete_incumbent") < (
        validation.events.index("solve_native")
    )
    assert witness.events == [
        "courier_only_objective",
        "total_demand_per_hour",
        "candidate_h_count",
        "clear_mip_starts",
        "solve_native",
        "extract_solution",
        "add_witness_search_constraints",
        "solve_native",
        "extract_solution",
        "capture_complete_incumbent",
        "remove_temporary_constraint",
        "dispose",
    ]
    assert validation.events == [
        "courier_only_objective",
        "install_complete_incumbent",
        "solve_native",
        "extract_solution",
        "dispose",
    ]
    assert all(
        settings.threads == 1
        for settings in witness.settings_seen + validation.settings_seen
    )


def test_expected_model_contract_is_verified_on_both_fresh_models() -> None:
    witness = FakeQualificationAdapter(
        [_positive_solution()],
        audit_result=_model_audit(),
    )
    validation = FakeQualificationAdapter(
        [_positive_solution()],
        audit_result=_model_audit(),
    )

    outcome, _factory = _run(
        (witness, validation),
        expected_model_contract=_model_contract(),
    )

    assert outcome.accepted
    assert witness.events[0] == "qualification_model_audit"
    assert validation.events[0] == "qualification_model_audit"
    payload = outcome.to_payload()
    assert payload["expected_model_contract"] == _model_contract().to_payload()
    assert payload["witness_model_audit"] == _model_audit().to_payload()
    assert payload["fresh_model_audit"] == _model_audit().to_payload()
    for record in (outcome.evidence, outcome.decision):
        assert record.qualification_case_sha256 == _QUALIFICATION_CASE_SHA256
        assert record.effective_model_config_sha256 == _MODEL_CONFIG_SHA256
        assert record.screening_is_exact is False
    for key in ("evidence", "decision"):
        assert payload[key]["qualification_case_sha256"] == (_QUALIFICATION_CASE_SHA256)
        assert payload[key]["effective_model_config_sha256"] == (_MODEL_CONFIG_SHA256)
        assert payload[key]["screening_is_exact"] is False


def test_rejected_candidate_retains_witness_audited_model_identity() -> None:
    witness = FakeQualificationAdapter(
        [_courier_solution(), _courier_solution()],
        audit_result=_model_audit(),
    )

    outcome, _factory = _run(
        (witness,),
        expected_model_contract=_model_contract(),
    )

    assert not outcome.accepted
    assert not outcome.decision.accepted
    assert not outcome.fresh_model_validation_performed
    assert outcome.fresh_model_audit is None
    assert outcome.evidence.qualification_case_sha256 == _QUALIFICATION_CASE_SHA256
    assert outcome.evidence.effective_model_config_sha256 == _MODEL_CONFIG_SHA256
    assert outcome.evidence.screening_is_exact is False
    assert outcome.decision.qualification_case_sha256 == _QUALIFICATION_CASE_SHA256
    assert outcome.decision.effective_model_config_sha256 == _MODEL_CONFIG_SHA256
    assert outcome.decision.screening_is_exact is False


def test_required_model_audit_failure_is_infrastructure_not_rejection() -> None:
    witness = FakeQualificationAdapter(
        [_positive_solution()],
        audit_result=RuntimeError("synthetic identity readback failure"),
    )

    with pytest.raises(
        QualificationInfrastructureError,
        match="witness qualification model identity audit failed",
    ):
        _run(
            (witness,),
            expected_model_contract=_model_contract(),
        )

    assert witness.events == ["qualification_model_audit", "dispose"]


def test_identically_wrong_witness_and_fresh_models_fail_expected_contract() -> None:
    wrong = _model_audit(
        model_config_sha256="d" * 64,
        locker_fixed_cost_per_hour=1.0,
        display_origin_choices=None,
        general_corridor_choices=None,
        screening_is_exact=True,
    )
    witness = FakeQualificationAdapter(
        [_positive_solution()],
        audit_result=wrong,
    )
    validation = FakeQualificationAdapter(
        [_positive_solution()],
        audit_result=wrong,
    )

    with pytest.raises(
        QualificationInfrastructureError,
        match="witness model audit does not match qualification case",
    ):
        _run(
            (witness, validation),
            expected_model_contract=_model_contract(),
        )

    assert witness.events == ["qualification_model_audit", "dispose"]
    assert validation.events == []


def test_fresh_model_screening_drift_fails_closed() -> None:
    witness = FakeQualificationAdapter(
        [_positive_solution()],
        audit_result=_model_audit(),
    )
    validation = FakeQualificationAdapter(
        [_positive_solution()],
        audit_result=_model_audit(screening_is_exact=True, model_fingerprint=999),
    )

    with pytest.raises(
        QualificationInfrastructureError,
        match="fresh model audit does not match qualification case",
    ):
        _run(
            (witness, validation),
            expected_model_contract=_model_contract(),
        )

    assert witness.disposed
    assert validation.disposed


def test_fresh_model_fingerprint_drift_fails_closed() -> None:
    witness = FakeQualificationAdapter(
        [_positive_solution()],
        audit_result=_model_audit(model_fingerprint=12345),
    )
    validation = FakeQualificationAdapter(
        [_positive_solution()],
        audit_result=_model_audit(model_fingerprint=999),
    )

    with pytest.raises(
        QualificationInfrastructureError,
        match="fresh qualification model fingerprint differs",
    ):
        _run(
            (witness, validation),
            expected_model_contract=_model_contract(),
        )


def test_mlnd_adapter_reads_back_effective_qualification_model() -> None:
    config = replace(
        ModelConfig(),
        locker_fixed_cost_per_hour=24.0,
        feeder_rider_cost_factor=0.30,
        display_origin_choices=8,
        general_corridor_choices=10,
    )
    adapter = MLNDQualificationAdapter.__new__(MLNDQualificationAdapter)
    adapter.instance = SimpleNamespace(
        metadata={"demand_scale": 1.0},
        config=config,
        screening_is_exact=lambda mode: mode != "MX",
    )
    adapter._mode = "MX"
    adapter._base_model_fingerprint = 456

    audit = adapter.qualification_model_audit()

    assert audit == QualificationModelAudit(
        effective_model_config_sha256=canonical_sha256(to_dict(config)),
        demand_scale=1.0,
        locker_fixed_cost_per_hour=24.0,
        display_origin_choices=8,
        general_corridor_choices=10,
        screening_is_exact=False,
        model_fingerprint=456,
    )


def test_mlnd_adapter_rejects_missing_demand_scale_readback() -> None:
    adapter = MLNDQualificationAdapter.__new__(MLNDQualificationAdapter)
    adapter.instance = SimpleNamespace(
        metadata={},
        config=ModelConfig(),
        screening_is_exact=lambda _mode: True,
    )
    adapter._mode = "MX"
    adapter._base_model_fingerprint = 456

    with pytest.raises(
        QualificationInfrastructureError,
        match="no numeric demand_scale",
    ):
        adapter.qualification_model_audit()


@dataclass(frozen=True)
class _FakeGeometryCycle:
    audit_descriptor: tuple[str, ...] = (
        "L0",
        "H0",
        "H1",
        "E0",
        "E0",
        "E0",
    )


@dataclass(frozen=True)
class _FakeGeometryPlan:
    target_support_size: int
    plan_version: str = GEOMETRY_START_PLAN_VERSION
    plan_sha256: str = "b" * 64
    availability_index: int = 1
    availability_probability: float = 0.8
    offered_flow_per_cycle: float = 1.0
    flow_per_cycle: float = 0.8
    planned_service_per_hour: float = 8.8
    family_start_counts: tuple[tuple[str, int], ...] = (
        ("psi", 3),
        ("varphi", 3),
        ("x", 4),
        ("y", 9),
        ("z", 2),
    )
    precheck_metrics: tuple[tuple[str, float], ...] = (("queue_slack", 1.0),)

    @property
    def support_order(self) -> tuple[str, ...]:
        return tuple(f"H{index}" for index in range(self.target_support_size))

    @property
    def cycles(self) -> tuple[_FakeGeometryCycle, ...]:
        return tuple(_FakeGeometryCycle() for _ in range(self.target_support_size - 1))


@dataclass
class FakeGeometryQualificationAdapter(FakeQualificationAdapter):
    unsafe_install: bool = False
    geometry_plan_error: BaseException | None = None
    geometry_install_error: BaseException | None = None
    geometry_cleanup_error: BaseException | None = None
    geometry_plan_targets: list[int] = field(default_factory=list)

    def plan_geometry_start(self, **kwargs: object) -> _FakeGeometryPlan:
        self.events.append("plan_geometry_start")
        target = int(kwargs["target_support_size"])
        self.geometry_plan_targets.append(target)
        if self.geometry_plan_error is not None:
            raise self.geometry_plan_error
        plan_sha256 = ("c" if target == 12 else "d") * 64
        return _FakeGeometryPlan(
            target_support_size=target,
            plan_sha256=plan_sha256,
        )

    def install_geometry_start(
        self,
        plan: _FakeGeometryPlan,
    ) -> GeometryStartInstallAudit:
        self.events.append("install_geometry_start")
        if self.geometry_install_error is not None:
            raise self.geometry_install_error
        after = 18 if self.unsafe_install else 17
        return GeometryStartInstallAudit(
            reset_performed=True,
            undefined_variable_count=100,
            family_start_counts=plan.family_start_counts,
            fingerprint_before=17,
            fingerprint_with_start=19,
            fingerprint_after=after,
            fingerprint_unchanged=after == 17,
            constraint_counts_before=(2, 1, 0),
            constraint_counts_after=(2, 1, 0),
            hard_constraints_added=False,
        )

    def solve_geometry_start(
        self,
        settings: NativeQualificationSettings,
        policy: GeometryStartPolicy,
    ) -> str:
        self.events.append("solve_geometry_start")
        self.settings_seen.append(settings)
        assert policy.enabled
        if not self.solve_results:
            raise AssertionError("unexpected geometry solve")
        result = self.solve_results.pop(0)
        if isinstance(result, BaseException):
            raise result
        self.current_solution = result
        return result.status

    def clear_geometry_start(self) -> None:
        self.events.append("clear_geometry_start")
        if self.geometry_cleanup_error is not None:
            raise self.geometry_cleanup_error


def _small_23_solution() -> QualificationSolution:
    return replace(
        _positive_solution(),
        candidate_h=23,
        open_h=12,
        active_h=12,
    )


def test_geometry_fallback_runs_only_after_no_start_constant_search_and_validates_fresh() -> (
    None
):
    courier_23 = replace(_courier_solution(), candidate_h=23)
    witness = FakeGeometryQualificationAdapter(
        [courier_23, courier_23, _small_23_solution()],
        candidate_h=23,
    )
    validation = FakeQualificationAdapter([_small_23_solution()], candidate_h=23)
    factory = FakeAdapterFactory([witness, validation])
    geometry_policy = GeometryStartPolicy(
        enabled=True,
        per_attempt_seconds=4.0,
        start_node_limit=77,
    )

    outcome = qualify_canonical_instance(
        object(),
        "MX",
        case_id="small-b0-s11",
        scale="small",
        qualification_run_id="qualification-geometry-test",
        instance_sha256=_INSTANCE_SHA256,
        geometry_start_policy=geometry_policy,
        adapter_factory=factory,
    )

    assert outcome.accepted
    assert outcome.geometry_start_policy == geometry_policy
    assert len(outcome.geometry_start_attempts) == 1
    attempt = outcome.geometry_start_attempts[0]
    assert attempt.plan.target_support_size == 12
    assert attempt.full_witness_found
    assert attempt.install is not None
    assert attempt.install.fingerprint_unchanged
    assert not attempt.install.hard_constraints_added
    assert outcome.geometry_start_selected_plan_sha256 == "c" * 64
    assert witness.geometry_plan_targets == [12]
    assert witness.events.index("solve_native") < witness.events.index(
        "add_witness_search_constraints"
    )
    assert witness.events.index(
        "add_witness_search_constraints"
    ) < witness.events.index("plan_geometry_start")
    assert witness.events.index("plan_geometry_start") < witness.events.index(
        "install_geometry_start"
    )
    assert witness.events.index("install_geometry_start") < witness.events.index(
        "solve_geometry_start"
    )
    assert "clear_geometry_start" not in witness.events
    assert validation.events == [
        "courier_only_objective",
        "install_complete_incumbent",
        "solve_native",
        "extract_solution",
        "dispose",
    ]
    assert outcome.phase_timings.geometry_start_solve_seconds >= 0.0


def test_geometry_install_safety_failure_aborts_after_clearing_start() -> None:
    courier_23 = replace(_courier_solution(), candidate_h=23)
    witness = FakeGeometryQualificationAdapter(
        [courier_23, courier_23],
        unsafe_install=True,
        candidate_h=23,
    )

    with pytest.raises(
        QualificationInfrastructureError,
        match="start-neutral model fingerprint",
    ):
        qualify_canonical_instance(
            object(),
            "MX",
            case_id="small-b0-s11",
            scale="small",
            qualification_run_id="qualification-geometry-safety-test",
            instance_sha256=_INSTANCE_SHA256,
            geometry_start_policy=GeometryStartPolicy(enabled=True),
            adapter_factory=FakeAdapterFactory([witness]),
        )

    assert "solve_geometry_start" not in witness.events
    assert "clear_geometry_start" in witness.events


def test_geometry_schedule_mismatch_is_not_misreported_as_constrained_solve_failure() -> (
    None
):
    witness = FakeGeometryQualificationAdapter(
        [_courier_solution(), _courier_solution()]
    )

    outcome = qualify_canonical_instance(
        object(),
        "MX",
        case_id="small-b0-s11",
        scale="small",
        qualification_run_id="qualification-geometry-policy-test",
        instance_sha256=_INSTANCE_SHA256,
        geometry_start_policy=GeometryStartPolicy(enabled=True),
        adapter_factory=FakeAdapterFactory([witness]),
    )

    assert QualificationRunnerReason.GEOMETRY_START_POLICY_FAILED in (
        outcome.rejection_reasons
    )
    assert QualificationRunnerReason.CONSTRAINED_WITNESS_SOLVE_FAILED not in (
        outcome.rejection_reasons
    )
    assert any("candidate-H count" in item for item in outcome.exception_messages)


def test_expected_geometry_precheck_failure_remains_a_normal_candidate_outcome() -> (
    None
):
    courier_23 = replace(_courier_solution(), candidate_h=23)
    witness = FakeGeometryQualificationAdapter(
        [courier_23, courier_23],
        candidate_h=23,
        geometry_plan_error=GeometryStartInfeasibilityError(
            "synthetic candidate topology precheck"
        ),
    )

    outcome = qualify_canonical_instance(
        object(),
        "MX",
        case_id="small-b0-s11",
        scale="small",
        qualification_run_id="qualification-geometry-precheck-test",
        instance_sha256=_INSTANCE_SHA256,
        geometry_start_policy=GeometryStartPolicy(enabled=True),
        adapter_factory=FakeAdapterFactory([witness]),
        fail_fast_infrastructure=True,
    )

    assert not outcome.accepted
    assert witness.geometry_plan_targets == [12, 8, 15]
    assert len(outcome.geometry_start_attempts) == 3
    assert outcome.exception_messages == ()
    assert QualificationRunnerReason.GEOMETRY_START_FALLBACK_FAILED in (
        outcome.rejection_reasons
    )


def test_internal_geometry_planner_error_aborts_instead_of_advancing_ladder() -> None:
    courier_23 = replace(_courier_solution(), candidate_h=23)
    witness = FakeGeometryQualificationAdapter(
        [courier_23, courier_23],
        candidate_h=23,
        geometry_plan_error=GeometryStartPlanningError(
            "synthetic planner identity failure"
        ),
    )

    with pytest.raises(
        QualificationInfrastructureError,
        match="planner identity failure",
    ):
        qualify_canonical_instance(
            object(),
            "MX",
            case_id="small-b0-s11",
            scale="small",
            qualification_run_id="qualification-geometry-internal-test",
            instance_sha256=_INSTANCE_SHA256,
            geometry_start_policy=GeometryStartPolicy(enabled=True),
            adapter_factory=FakeAdapterFactory([witness]),
            fail_fast_infrastructure=True,
        )

    assert witness.geometry_plan_targets == [12]
    assert witness.disposed


def test_geometry_start_cleanup_does_not_mask_primary_solve_failure() -> None:
    courier_23 = replace(_courier_solution(), candidate_h=23)
    witness = FakeGeometryQualificationAdapter(
        [
            courier_23,
            courier_23,
            RuntimeError("PRIMARY_GEOMETRY_SOLVE"),
        ],
        candidate_h=23,
        geometry_cleanup_error=RuntimeError("SECONDARY_GEOMETRY_CLEAR"),
    )

    with pytest.raises(
        QualificationInfrastructureError,
        match="PRIMARY_GEOMETRY_SOLVE",
    ) as caught:
        qualify_canonical_instance(
            object(),
            "MX",
            case_id="small-b0-s11",
            scale="small",
            qualification_run_id="qualification-geometry-cleanup-test",
            instance_sha256=_INSTANCE_SHA256,
            geometry_start_policy=GeometryStartPolicy(enabled=True),
            adapter_factory=FakeAdapterFactory([witness]),
            fail_fast_infrastructure=True,
        )

    assert "clear_geometry_start" in witness.events
    assert any(
        "SECONDARY_GEOMETRY_CLEAR" in note
        for note in getattr(caught.value, "__notes__", ())
    )
    assert witness.disposed


def test_failed_geometry_attempt_is_cleared_before_next_support_succeeds() -> None:
    courier_23 = replace(_courier_solution(), candidate_h=23)
    witness = FakeGeometryQualificationAdapter(
        [
            courier_23,
            courier_23,
            courier_23,
            replace(_small_23_solution(), open_h=8, active_h=8),
        ],
        candidate_h=23,
    )
    validation = FakeQualificationAdapter(
        [replace(_small_23_solution(), open_h=8, active_h=8)],
        candidate_h=23,
    )

    outcome = qualify_canonical_instance(
        object(),
        "MX",
        case_id="small-b0-s11",
        scale="small",
        qualification_run_id="qualification-geometry-retry-test",
        instance_sha256=_INSTANCE_SHA256,
        geometry_start_policy=GeometryStartPolicy(enabled=True),
        adapter_factory=FakeAdapterFactory([witness, validation]),
    )

    assert outcome.accepted
    assert witness.geometry_plan_targets == [12, 8]
    assert len(outcome.geometry_start_attempts) == 2
    first, second = outcome.geometry_start_attempts
    assert not first.full_witness_found
    assert first.cleanup_performed
    assert first.cleanup_succeeded is True
    assert second.full_witness_found
    assert not second.cleanup_performed
    assert outcome.geometry_start_selected_plan_sha256 == "d" * 64
    first_clear = witness.events.index("clear_geometry_start")
    second_plan = witness.events.index("plan_geometry_start", first_clear + 1)
    assert first_clear < second_plan


def test_geometry_install_exception_recovers_and_terminates_support_ladder() -> None:
    courier_23 = replace(_courier_solution(), candidate_h=23)
    witness = FakeGeometryQualificationAdapter(
        [courier_23, courier_23],
        candidate_h=23,
        geometry_install_error=RuntimeError("synthetic geometry install failure"),
    )

    outcome = qualify_canonical_instance(
        object(),
        "MX",
        case_id="small-b0-s11",
        scale="small",
        qualification_run_id="qualification-geometry-install-error-test",
        instance_sha256=_INSTANCE_SHA256,
        geometry_start_policy=GeometryStartPolicy(enabled=True),
        adapter_factory=FakeAdapterFactory([witness]),
    )

    assert witness.geometry_plan_targets == [12]
    assert "clear_geometry_start" in witness.events
    assert "solve_geometry_start" not in witness.events
    assert len(outcome.geometry_start_attempts) == 1
    assert outcome.geometry_start_attempts[0].install is None
    assert QualificationRunnerReason.GEOMETRY_START_INSTALLATION_FAILED in (
        outcome.rejection_reasons
    )
    assert QualificationRunnerReason.CONSTRAINED_WITNESS_SOLVE_FAILED not in (
        outcome.rejection_reasons
    )


def test_geometry_solve_exception_cleans_and_terminates_support_ladder() -> None:
    courier_23 = replace(_courier_solution(), candidate_h=23)
    witness = FakeGeometryQualificationAdapter(
        [
            courier_23,
            courier_23,
            RuntimeError("synthetic geometry solve failure"),
        ],
        candidate_h=23,
    )

    outcome = qualify_canonical_instance(
        object(),
        "MX",
        case_id="small-b0-s11",
        scale="small",
        qualification_run_id="qualification-geometry-solve-error-test",
        instance_sha256=_INSTANCE_SHA256,
        geometry_start_policy=GeometryStartPolicy(enabled=True),
        adapter_factory=FakeAdapterFactory([witness]),
    )

    assert witness.geometry_plan_targets == [12]
    assert "clear_geometry_start" in witness.events
    assert len(outcome.geometry_start_attempts) == 1
    attempt = outcome.geometry_start_attempts[0]
    assert attempt.solve_status == "ERROR"
    assert attempt.cleanup_succeeded is True
    assert QualificationRunnerReason.GEOMETRY_START_SOLVE_FAILED in (
        outcome.rejection_reasons
    )
    assert QualificationRunnerReason.CONSTRAINED_WITNESS_SOLVE_FAILED not in (
        outcome.rejection_reasons
    )


def test_geometry_capture_exception_cleans_and_terminates_support_ladder() -> None:
    courier_23 = replace(_courier_solution(), candidate_h=23)
    witness = FakeGeometryQualificationAdapter(
        [courier_23, courier_23, _small_23_solution()],
        candidate_h=23,
        capture_error=RuntimeError("synthetic geometry capture failure"),
    )

    outcome = qualify_canonical_instance(
        object(),
        "MX",
        case_id="small-b0-s11",
        scale="small",
        qualification_run_id="qualification-geometry-capture-error-test",
        instance_sha256=_INSTANCE_SHA256,
        geometry_start_policy=GeometryStartPolicy(enabled=True),
        adapter_factory=FakeAdapterFactory([witness]),
    )

    assert witness.geometry_plan_targets == [12]
    assert "clear_geometry_start" in witness.events
    assert len(outcome.geometry_start_attempts) == 1
    attempt = outcome.geometry_start_attempts[0]
    assert not attempt.full_witness_found
    assert attempt.cleanup_succeeded is True
    assert attempt.error is not None
    assert outcome.geometry_start_selected_plan_sha256 is None
    assert QualificationRunnerReason.COMPLETE_INCUMBENT_CAPTURE_FAILED in (
        outcome.rejection_reasons
    )


def test_witness_is_disposed_before_fresh_adapter_factory_runs() -> None:
    witness = FakeQualificationAdapter([_positive_solution()])
    validation = FakeQualificationAdapter([_positive_solution()])
    factory_calls = 0

    def checking_factory(
        instance: object,
        mode: str,
    ) -> FakeQualificationAdapter:
        nonlocal factory_calls
        del instance, mode
        factory_calls += 1
        if factory_calls == 1:
            return witness
        assert witness.disposed
        return validation

    outcome = qualify_canonical_instance(
        object(),
        "MX",
        case_id="small-b0-s11",
        scale="small",
        qualification_run_id="qualification-test",
        instance_sha256=_INSTANCE_SHA256,
        adapter_factory=checking_factory,
    )

    assert outcome.accepted
    assert not outcome.feasibility_objective_used
    assert factory_calls == 2
    assert witness.events.index("capture_complete_incumbent") < witness.events.index(
        "dispose"
    )


def test_witness_disposal_failure_aborts_and_blocks_fresh_model_build() -> None:
    witness = FakeQualificationAdapter(
        [_positive_solution()],
        dispose_error=RuntimeError("synthetic dispose failure"),
    )
    validation = FakeQualificationAdapter([_positive_solution()])

    with pytest.raises(
        QualificationInfrastructureError,
        match="witness_model_disposal",
    ) as raised:
        _run([witness, validation])

    assert any(
        "qualification_lifecycle_disposal" in note
        for note in getattr(raised.value, "__notes__", ())
    )
    assert validation.events == []


def test_fresh_disposal_failure_aborts_after_validation() -> None:
    witness = FakeQualificationAdapter([_positive_solution()])
    validation = FakeQualificationAdapter(
        [_positive_solution()],
        dispose_error=RuntimeError("synthetic fresh dispose failure"),
    )

    with pytest.raises(
        QualificationInfrastructureError,
        match="fresh_model_disposal",
    ) as raised:
        _run([witness, validation])

    assert validation.installed_incumbent is not None
    assert any(
        "qualification_lifecycle_disposal" in note
        for note in getattr(raised.value, "__notes__", ())
    )
    assert validation.events[-1] == "dispose"


@pytest.mark.parametrize(
    ("failure_stage", "expected_reason"),
    [
        (
            "baseline",
            QualificationRunnerReason.FRESH_MODEL_BASELINE_EXTRACTION_FAILED,
        ),
        (
            "install",
            QualificationRunnerReason.COMPLETE_START_INSTALLATION_FAILED,
        ),
        ("solve", QualificationRunnerReason.FRESH_VALIDATION_SOLVE_FAILED),
        ("extract", QualificationRunnerReason.FRESH_VALIDATION_SOLVE_FAILED),
    ],
)
def test_fresh_adapter_is_disposed_after_each_validation_stage_failure(
    failure_stage: str,
    expected_reason: str,
) -> None:
    witness = FakeQualificationAdapter([_positive_solution()])
    validation = FakeQualificationAdapter([_positive_solution()])
    if failure_stage == "baseline":
        validation.baseline_error = RuntimeError("synthetic baseline failure")
    elif failure_stage == "install":
        validation.install_error = RuntimeError("synthetic install failure")
    elif failure_stage == "solve":
        validation.solve_results = [RuntimeError("synthetic solve failure")]
    else:
        validation.extract_error = RuntimeError("synthetic extraction failure")

    outcome, _ = _run([witness, validation])

    assert not outcome.accepted
    assert expected_reason in outcome.rejection_reasons
    assert validation.disposed
    assert validation.events[-1] == "dispose"


@pytest.mark.parametrize(
    "fatal_error",
    [
        MemoryError("synthetic memory exhaustion"),
        KeyboardInterrupt("synthetic interrupt"),
        SystemExit("synthetic exit"),
    ],
)
def test_process_level_exceptions_propagate(fatal_error: BaseException) -> None:
    def failing_factory(instance: object, mode: str) -> FakeQualificationAdapter:
        del instance, mode
        raise fatal_error

    with pytest.raises(type(fatal_error)):
        qualify_canonical_instance(
            object(),
            "MX",
            case_id="small-b0-s11",
            scale="small",
            qualification_run_id="qualification-test",
            instance_sha256=_INSTANCE_SHA256,
            adapter_factory=failing_factory,
        )


def test_memory_error_from_solver_stage_is_not_converted_to_rejection() -> None:
    witness = FakeQualificationAdapter(
        [MemoryError("synthetic solver allocation failure")]
    )

    with pytest.raises(MemoryError, match="solver allocation"):
        _run([witness])

    assert witness.disposed


def test_fail_fast_mode_raises_on_solver_exception_before_case_evidence() -> None:
    witness = FakeQualificationAdapter(
        [RuntimeError("synthetic licence or solve failure")]
    )

    with pytest.raises(QualificationInfrastructureError, match="native_witness_solve"):
        qualify_canonical_instance(
            object(),
            "MX",
            case_id="small-b0-s11",
            scale="small",
            qualification_run_id="qualification-fail-fast-test",
            instance_sha256=_INSTANCE_SHA256,
            adapter_factory=FakeAdapterFactory([witness]),
            fail_fast_infrastructure=True,
        )

    assert witness.disposed


@pytest.mark.parametrize("status", ["MEM_LIMIT", "INTERRUPTED", "NUMERIC"])
def test_abnormal_native_status_is_always_an_infrastructure_failure(
    status: str,
) -> None:
    witness = FakeQualificationAdapter([replace(_courier_solution(), status=status)])

    with pytest.raises(QualificationInfrastructureError, match=status):
        _run([witness])

    assert witness.disposed


def test_gurobi_native_out_of_memory_is_normalized_and_stops_the_run() -> None:
    class FakeGurobiOutOfMemory(RuntimeError):
        errno = 10001

    witness = FakeQualificationAdapter(
        [FakeGurobiOutOfMemory("synthetic native allocation failure")]
    )

    with pytest.raises(MemoryError, match="Gurobi out of memory"):
        _run([witness])

    assert witness.disposed


@pytest.mark.parametrize(
    "fatal_error",
    [
        KeyboardInterrupt("synthetic witness solve interrupt"),
        SystemExit("synthetic witness solve exit"),
    ],
)
def test_witness_solver_process_exceptions_propagate_after_disposal(
    fatal_error: BaseException,
) -> None:
    witness = FakeQualificationAdapter([fatal_error])

    with pytest.raises(type(fatal_error)):
        _run([witness])

    assert witness.disposed


def test_invalid_minimum_service_still_disposes_constructed_witness() -> None:
    witness = FakeQualificationAdapter([_positive_solution()])

    with pytest.raises(ValueError, match="minimum_service_per_hour"):
        factory = FakeAdapterFactory([witness])
        qualify_canonical_instance(
            object(),
            "MX",
            case_id="small-b0-s11",
            scale="small",
            qualification_run_id="qualification-test",
            instance_sha256=_INSTANCE_SHA256,
            adapter_factory=factory,
            minimum_service_per_hour=0.0,
        )

    assert witness.disposed


@pytest.mark.parametrize(
    "fatal_error",
    [
        MemoryError("synthetic fresh solve memory exhaustion"),
        KeyboardInterrupt("synthetic fresh solve interrupt"),
        SystemExit("synthetic fresh solve exit"),
    ],
)
def test_fresh_solve_process_exceptions_propagate_after_disposal(
    fatal_error: BaseException,
) -> None:
    witness = FakeQualificationAdapter([_positive_solution()])
    validation = FakeQualificationAdapter([fatal_error])

    with pytest.raises(type(fatal_error)):
        _run([witness, validation])

    assert validation.disposed
    assert validation.events[-1] == "dispose"


@pytest.mark.parametrize(
    "fatal_error",
    [
        MemoryError("synthetic fresh dispose memory exhaustion"),
        KeyboardInterrupt("synthetic fresh dispose interrupt"),
        SystemExit("synthetic fresh dispose exit"),
    ],
)
def test_fresh_disposal_process_exceptions_propagate(
    fatal_error: BaseException,
) -> None:
    witness = FakeQualificationAdapter([_positive_solution()])
    validation = FakeQualificationAdapter(
        [_positive_solution()],
        dispose_error=fatal_error,
    )

    with pytest.raises(type(fatal_error)):
        _run([witness, validation])

    assert validation.events[-1] == "dispose"


def test_temporary_constraint_is_removed_when_constrained_solve_raises() -> None:
    witness = FakeQualificationAdapter(
        [_courier_solution(), RuntimeError("synthetic solver failure")]
    )

    outcome, factory = _run([witness])

    assert not outcome.accepted
    assert outcome.temporary_constraint_removed is True
    assert not witness.temporary_active
    assert "remove_temporary_constraint" in witness.events
    assert QualificationRunnerReason.CONSTRAINED_WITNESS_SOLVE_FAILED in (
        outcome.rejection_reasons
    )
    assert QualificationRunnerReason.NO_POSITIVE_SERVICE_INCUMBENT in (
        outcome.rejection_reasons
    )
    assert len(factory.calls) == 1


def test_positive_but_nonimproving_native_solution_triggers_bounded_search() -> None:
    witness = FakeQualificationAdapter(
        [_positive_but_not_improving_solution(), _positive_solution()]
    )
    validation = FakeQualificationAdapter([_positive_solution()])

    outcome, _ = _run([witness, validation])

    assert outcome.accepted
    assert outcome.temporary_constraint_used
    assert witness.temporary_threshold == pytest.approx(0.1)
    assert witness.temporary_objective_upper_bound == pytest.approx(999.0)
    assert "add_witness_search_constraints" in witness.events


def test_objective_cutoff_uses_the_preregistered_margin_formula() -> None:
    witness = FakeQualificationAdapter([_courier_solution(), _positive_solution()])
    validation = FakeQualificationAdapter([_positive_solution()])
    factory = FakeAdapterFactory([witness, validation])

    outcome = qualify_canonical_instance(
        object(),
        "MX",
        case_id="small-b0-s11",
        scale="small",
        qualification_run_id="qualification-test",
        instance_sha256=_INSTANCE_SHA256,
        policy=QualificationPolicy(optimality_tolerance=1e-3),
        adapter_factory=factory,
    )

    # max(1, 1e-4*C0, 100*tol*C0) = 100 when C0=1000 and tol=1e-3.
    assert outcome.required_objective_upper_bound == pytest.approx(900.0)
    assert witness.temporary_objective_upper_bound == pytest.approx(900.0)
    assert outcome.accepted


def test_objective_cutoff_also_enforces_minimum_savings_fraction() -> None:
    witness = FakeQualificationAdapter(
        [
            replace(_courier_solution(), objective=100_000.0),
            replace(_positive_solution(), objective=99_800.0),
        ],
        courier_objective=100_000.0,
    )
    validation = FakeQualificationAdapter(
        [replace(_positive_solution(), objective=99_800.0)],
        courier_objective=100_000.0,
    )
    factory = FakeAdapterFactory([witness, validation])

    outcome = qualify_canonical_instance(
        object(),
        "MX",
        case_id="small-b0-s11",
        scale="small",
        qualification_run_id="qualification-test",
        instance_sha256=_INSTANCE_SHA256,
        policy=QualificationPolicy(minimum_savings_fraction=2e-3),
        adapter_factory=factory,
    )

    # The numerical margin is only 10, while the registered 0.2% saving is 200.
    assert outcome.required_objective_upper_bound == pytest.approx(99_800.0)
    assert witness.temporary_objective_upper_bound == pytest.approx(99_800.0)
    assert outcome.accepted


def test_constrained_solution_without_objective_improvement_is_not_captured() -> None:
    witness = FakeQualificationAdapter(
        [_courier_solution(), _positive_but_not_improving_solution()]
    )

    outcome, factory = _run([witness])

    assert not outcome.accepted
    assert outcome.temporary_constraint_removed is True
    assert not outcome.fresh_model_validation_performed
    assert outcome.captured_variable_count == 0
    assert "capture_complete_incumbent" not in witness.events
    assert QualificationRunnerReason.NO_OBJECTIVE_IMPROVING_INCUMBENT in (
        outcome.rejection_reasons
    )
    assert len(factory.calls) == 1


def test_failed_temporary_removal_blocks_fresh_validation() -> None:
    witness = FakeQualificationAdapter(
        [_courier_solution(), _positive_solution()],
        removal_error=RuntimeError("synthetic removal failure"),
    )
    validation = FakeQualificationAdapter([_positive_solution()])

    with pytest.raises(
        QualificationInfrastructureError,
        match="temporary_constraint_removal",
    ):
        _run([witness, validation])

    assert witness.temporary_active
    assert "remove_temporary_constraint" in witness.events
    assert validation.events == []


def test_constrained_incumbent_is_not_evidence_without_fresh_incumbent() -> None:
    witness = FakeQualificationAdapter([_courier_solution(), _positive_solution()])
    validation = FakeQualificationAdapter([_no_incumbent()])

    outcome, _ = _run([witness, validation])

    assert not outcome.accepted
    assert outcome.fresh_model_validation_performed
    assert validation.installed_incumbent is not None
    assert validation.disposed
    assert validation.events[-2:] == ["extract_solution", "dispose"]
    assert not outcome.evidence.witness_feasible
    assert QualificationRunnerReason.FRESH_VALIDATION_NO_INCUMBENT in (
        outcome.rejection_reasons
    )
    assert QualificationReason.WITNESS_NOT_FEASIBLE in outcome.rejection_reasons


def test_factory_reusing_witness_model_is_rejected_as_not_fresh() -> None:
    reused = FakeQualificationAdapter([_positive_solution()])
    factory = FakeAdapterFactory([reused, reused])

    outcome = qualify_canonical_instance(
        object(),
        "MX",
        case_id="small-b0-s11",
        scale="small",
        qualification_run_id="qualification-test",
        instance_sha256=_INSTANCE_SHA256,
        adapter_factory=factory,
    )

    assert not outcome.accepted
    assert not outcome.fresh_model_validation_performed
    assert QualificationRunnerReason.FRESH_MODEL_REUSED in outcome.rejection_reasons
    assert "install_complete_incumbent" not in reused.events


def test_fresh_solution_not_constrained_solution_supplies_evidence() -> None:
    witness = FakeQualificationAdapter([_courier_solution(), _positive_solution()])
    validation = FakeQualificationAdapter([_courier_solution()])

    outcome, _ = _run([witness, validation])

    assert not outcome.accepted
    assert outcome.evidence.witness_objective == pytest.approx(1_000.0)
    assert outcome.evidence.witness_service_per_hour == pytest.approx(0.0)
    assert outcome.evidence.open_facilities == 0
    assert QualificationReason.NO_OPEN_FACILITY in outcome.rejection_reasons
    assert QualificationReason.SERVICE_BELOW_TOLERANCE in outcome.rejection_reasons


def test_adapter_initialization_exception_returns_structured_rejection() -> None:
    class InstanceWithH:
        H = ("H0", "H1")

    def failing_factory(instance: object, mode: str) -> FakeQualificationAdapter:
        del instance, mode
        raise RuntimeError("synthetic construction failure")

    outcome = qualify_canonical_instance(
        InstanceWithH(),
        "MX",
        case_id="small-b0-s11",
        scale="small",
        qualification_run_id="qualification-test",
        instance_sha256=_INSTANCE_SHA256,
        adapter_factory=failing_factory,
    )

    assert not outcome.accepted
    assert QualificationRunnerReason.ADAPTER_INITIALIZATION_FAILED in (
        outcome.rejection_reasons
    )
    assert outcome.evidence.candidate_h == 2
    assert outcome.exception_messages

    payload = outcome.to_payload()
    decoded = json.loads(canonical_json(payload))
    assert decoded["evidence"]["courier_only_objective"] is None
    assert decoded["evidence"]["witness_objective"] is None
    assert decoded["evidence"]["total_demand_per_hour"] is None
    assert decoded["evidence"]["witness_service_per_hour"] is None


def test_single_thread_requirement_and_complete_incumbent_are_enforced() -> None:
    with pytest.raises(ValueError, match="single-thread"):
        NativeQualificationSettings(threads=2)
    with pytest.raises(ValueError, match="miqcp_method"):
        NativeQualificationSettings(miqcp_method=2)
    with pytest.raises(ValueError, match="root_cut_passes"):
        NativeQualificationSettings(root_cut_passes=-2)
    with pytest.raises(ValueError, match="every model variable"):
        CompleteIncumbent.from_ordered_values(
            objective=1.0,
            variable_names=(),
            values=(),
        )

    compact = CompleteIncumbent.from_ordered_values(
        objective=1.0,
        variable_names=("x", "y"),
        values=(1.0, 0.0),
        model_fingerprint=123,
    )
    assert isinstance(compact.values, array)
    assert compact.variable_count == 2


def test_qualification_applies_frozen_native_root_cut_passes() -> None:
    adapter = MLNDQualificationAdapter.__new__(MLNDQualificationAdapter)
    params = SimpleNamespace()
    adapter.model = SimpleNamespace(Params=params)

    adapter._apply_native_settings(  # noqa: SLF001 - narrow parameter audit
        NativeQualificationSettings(root_cut_passes=7)
    )

    assert params.CutPasses == 7


class _BulkAttr:
    VarName = "VarName"
    X = "X"
    Start = "Start"


class _BulkGRB:
    Attr = _BulkAttr


class _GeometryGRB:
    Attr = _BulkAttr
    UNDEFINED = 1e101


class _GeometryStartModel:
    def __init__(self, variables: list[object]) -> None:
        self._variables = variables
        self.neutral_fingerprint = 314159
        self.NumConstrs = 10
        self.NumQConstrs = 2
        self.NumGenConstrs = 0
        self.NumStart = 0
        self.Params = SimpleNamespace(StartNumber=-1, StartNodeLimit=-1)
        self.reset_calls: list[int] = []
        self.start_vectors: list[list[float]] = []
        self.current_start = [_GeometryGRB.UNDEFINED] * len(variables)
        self.mutation_on_positive_start: str | None = None
        self._mutation_applied = False
        self.corrupt_readback = False

    @property
    def Fingerprint(self) -> int:
        # Match Gurobi semantics: a MIP Start affects the native fingerprint,
        # while clearing NumStart restores the mathematical-model fingerprint.
        return self.neutral_fingerprint + (1000 if self.NumStart else 0)

    def reset(self, clearall: int) -> None:
        self.reset_calls.append(clearall)

    def getVars(self) -> list[object]:
        return self._variables

    def setAttr(
        self,
        attribute: str,
        variables: list[object],
        values: list[float],
    ) -> None:
        assert attribute == _BulkAttr.Start
        assert variables is self._variables
        self.current_start = list(values)
        self.start_vectors.append(list(values))
        has_positive_hint = any(value != _GeometryGRB.UNDEFINED for value in values)
        if (
            has_positive_hint
            and self.mutation_on_positive_start is not None
            and not self._mutation_applied
        ):
            # Stand in for any mathematical-model mutation that Gurobi's
            # start-neutral Fingerprint must detect.
            self.neutral_fingerprint += 1
            self._mutation_applied = True

    def getAttr(
        self,
        attribute: str,
        variables: list[object],
    ) -> list[float]:
        assert attribute == _BulkAttr.Start
        assert variables is self._variables
        result = list(self.current_start)
        if self.corrupt_readback and self.NumStart and result:
            result[0] = 0.0
        return result

    def update(self) -> None:
        pass


def _geometry_adapter_and_plan(
    model: _GeometryStartModel,
) -> tuple[MLNDQualificationAdapter, SimpleNamespace]:
    variables = model.getVars()
    x_h, x_l, y_arc, z_h, psi_arc, varphi_arc, alpha_unlisted = variables
    adapter = MLNDQualificationAdapter.__new__(MLNDQualificationAdapter)
    adapter.model = model
    adapter._GRB = _GeometryGRB
    adapter._temporary_constraint_token = object()
    adapter.wrapper = SimpleNamespace(
        vars={
            "x": {"H0": x_h, "L0": x_l},
            "y": {("L0", "H0"): y_arc},
            "z": {("H0", 1): z_h},
            "psi": {("L0", "H0"): psi_arc},
            "varphi": {("L0", "H0", "E0"): varphi_arc},
            "alpha": {("D0", "H0"): alpha_unlisted},
        }
    )
    plan = SimpleNamespace(
        open_nodes=("H0", "L0"),
        activated_y_arcs=(("L0", "H0"),),
        capacity_choices=(SimpleNamespace(locker="H0", capacity_index=1),),
        physical_arc_flows=(("L0", "H0", 2.5),),
        route_flows=(("L0", "H0", "E0", 2.5),),
        family_start_counts=(
            ("psi", 1),
            ("varphi", 1),
            ("x", 2),
            ("y", 1),
            ("z", 1),
        ),
    )
    return adapter, plan


def test_geometry_adapter_sets_only_five_positive_families_and_leaves_rest_undefined() -> (
    None
):
    variables = [object() for _ in range(7)]
    model = _GeometryStartModel(variables)
    adapter, plan = _geometry_adapter_and_plan(model)

    audit = adapter.install_geometry_start(plan)

    assert audit.reset_performed
    assert audit.fingerprint_unchanged
    assert audit.fingerprint_before == 314159
    assert audit.fingerprint_with_start == 315159
    assert audit.fingerprint_after == 314159
    assert audit.safety_preserved
    assert not audit.hard_constraints_added
    assert not audit.bounds_modified
    assert not audit.objective_modified
    assert audit.undefined_variable_count == 1
    assert model.reset_calls == [0, 0]
    assert model.NumStart == 1
    installed = model.start_vectors[-1]
    assert installed[:2] == [1.0, 1.0]
    assert installed[2:5] == [1.0, 1.0, 2.5]
    assert installed[5] == pytest.approx(2.5)
    assert installed[6] == _GeometryGRB.UNDEFINED

    adapter.clear_geometry_start()

    assert model.reset_calls == [0, 0, 0]
    assert model.NumStart == 0
    assert model.start_vectors[-1] == [_GeometryGRB.UNDEFINED] * len(variables)


@pytest.mark.parametrize("mutation_kind", ["structure", "bounds", "objective"])
def test_geometry_adapter_rejects_any_start_neutral_model_mutation(
    mutation_kind: str,
) -> None:
    model = _GeometryStartModel([object() for _ in range(7)])
    model.mutation_on_positive_start = mutation_kind
    adapter, plan = _geometry_adapter_and_plan(model)

    audit = adapter.install_geometry_start(plan)

    assert audit.fingerprint_with_start != audit.fingerprint_before
    assert audit.fingerprint_after != audit.fingerprint_before
    assert not audit.fingerprint_unchanged
    assert not audit.safety_preserved
    assert model.NumStart == 1


def test_geometry_adapter_bulk_readback_failure_rolls_back_every_start() -> None:
    model = _GeometryStartModel([object() for _ in range(7)])
    model.corrupt_readback = True
    adapter, plan = _geometry_adapter_and_plan(model)

    with pytest.raises(RuntimeError, match="bulk readback mismatch"):
        adapter.install_geometry_start(plan)

    assert model.NumStart == 0
    assert model.current_start == [_GeometryGRB.UNDEFINED] * 7


class _BulkTransferModel:
    def __init__(
        self,
        names: tuple[str, ...],
        values: tuple[float, ...],
        *,
        fingerprint: int | None,
    ) -> None:
        if len(names) != len(values):
            raise ValueError("fake names and values must have the same length")
        self.names = names
        self.solution_values = values
        self._variables = [object()] * len(names)
        self.Fingerprint = fingerprint
        self.SolCount = 1
        self.ObjVal = 900.0
        self.NumStart = 0
        self.Params = SimpleNamespace(StartNumber=-1)
        self.get_vars_calls = 0
        self.get_attr_calls: list[str] = []
        self.set_attr_calls: list[tuple[str, int, bool, float, float]] = []
        self.update_calls = 0
        self.dispose_calls = 0

    def getVars(self) -> list[object]:
        self.get_vars_calls += 1
        return self._variables

    def getAttr(self, attribute: str, variables: list[object]):
        assert variables is self._variables
        self.get_attr_calls.append(attribute)
        if attribute == _BulkAttr.VarName:
            return self.names
        if attribute == _BulkAttr.X:
            return self.solution_values
        raise AssertionError(f"unexpected bulk attribute {attribute!r}")

    def setAttr(
        self,
        attribute: str,
        variables: list[object],
        values: object,
    ) -> None:
        assert variables is self._variables
        assert isinstance(values, array)
        self.set_attr_calls.append(
            (
                attribute,
                len(values),
                isinstance(values, array),
                float(values[0]),
                float(values[-1]),
            )
        )

    def update(self) -> None:
        self.update_calls += 1

    def dispose(self) -> None:
        self.dispose_calls += 1


def _bulk_transfer_adapter(model: _BulkTransferModel) -> MLNDQualificationAdapter:
    adapter = MLNDQualificationAdapter.__new__(MLNDQualificationAdapter)
    adapter.model = model
    adapter._GRB = _BulkGRB
    adapter._base_model_fingerprint = model.Fingerprint
    return adapter


def test_complete_start_uses_constant_count_bulk_attribute_calls() -> None:
    variable_count = 100_000
    names = tuple(f"v[{index}]" for index in range(variable_count))
    values = tuple(float(index % 7) for index in range(variable_count))
    source_model = _BulkTransferModel(names, values, fingerprint=1_234_567)
    fresh_model = _BulkTransferModel(
        names,
        (0.0,) * variable_count,
        fingerprint=1_234_567,
    )

    source_adapter = _bulk_transfer_adapter(source_model)
    # Qualification-only rows may change the live model Fingerprint; transfer
    # must retain the fingerprint cached for the original model.
    source_model.Fingerprint = 7_654_321
    incumbent = source_adapter.capture_complete_incumbent()
    _bulk_transfer_adapter(fresh_model).install_complete_incumbent(incumbent)

    assert incumbent.variable_count == variable_count
    assert incumbent.model_fingerprint == 1_234_567
    assert isinstance(incumbent.values, array)
    assert source_model.get_vars_calls == 1
    assert source_model.get_attr_calls == [_BulkAttr.VarName, _BulkAttr.X]
    assert fresh_model.get_vars_calls == 1
    assert fresh_model.get_attr_calls == [_BulkAttr.VarName]
    assert fresh_model.set_attr_calls == [
        (_BulkAttr.Start, variable_count, True, values[0], values[-1])
    ]
    assert fresh_model.NumStart == 1
    assert fresh_model.Params.StartNumber == 0


def test_complete_start_captures_original_cost_under_zero_feasibility_objective() -> (
    None
):
    model = _BulkTransferModel(("x", "chi"), (1.0, 2.0), fingerprint=17)
    model.ObjVal = 0.0
    adapter = _bulk_transfer_adapter(model)
    adapter._temporary_constraint_token = TemporaryWitnessConstraintToken(
        service_constraint=object(),
        objective_constraint=object(),
        original_objective=_FakeExpression("objective", 900.0),
        original_model_sense=1,
    )

    incumbent = adapter.capture_complete_incumbent()

    assert incumbent.objective == pytest.approx(900.0)
    assert tuple(incumbent.values) == pytest.approx((1.0, 2.0))


def test_mlnd_adapter_dispose_is_explicit_and_idempotent() -> None:
    model = _BulkTransferModel(("x",), (1.0,), fingerprint=17)
    adapter = _bulk_transfer_adapter(model)

    adapter.dispose()
    adapter.dispose()

    assert model.dispose_calls == 1
    assert adapter.model is None
    assert adapter.wrapper is None


def test_mlnd_adapter_disposes_native_model_when_wrapper_init_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.model.gurobi_model as gurobi_model_module

    class PartialNativeModel:
        dispose_calls = 0

        def dispose(self) -> None:
            self.dispose_calls += 1

    partial_model = PartialNativeModel()

    class FailingMLNDModel:
        def __init__(self, _instance: object, mode: str):
            assert mode == "MX"
            self.model = partial_model
            raise MemoryError("synthetic index allocation failure")

    monkeypatch.setattr(gurobi_model_module, "MLNDModel", FailingMLNDModel)

    with pytest.raises(MemoryError, match="index allocation"):
        MLNDQualificationAdapter(object(), "MX")

    assert partial_model.dispose_calls == 1


def test_complete_start_capture_fails_closed_without_model_fingerprint() -> None:
    model = _BulkTransferModel(("x",), (1.0,), fingerprint=None)

    with pytest.raises(ValueError, match="Fingerprint is unavailable"):
        _bulk_transfer_adapter(model).capture_complete_incumbent()


@pytest.mark.parametrize("missing_side", ["captured", "fresh"])
def test_complete_start_install_fails_closed_if_either_fingerprint_is_missing(
    missing_side: str,
) -> None:
    names = ("x", "y")
    incumbent = CompleteIncumbent.from_ordered_values(
        objective=1.0,
        variable_names=names,
        values=(1.0, 0.0),
        model_fingerprint=None if missing_side == "captured" else 17,
    )
    fresh = _BulkTransferModel(
        names,
        (0.0, 0.0),
        fingerprint=None if missing_side == "fresh" else 17,
    )

    with pytest.raises(ValueError, match="Fingerprint is unavailable"):
        _bulk_transfer_adapter(fresh).install_complete_incumbent(incumbent)

    assert fresh.set_attr_calls == []


@pytest.mark.parametrize(
    ("mismatch", "message"),
    [
        ("count", "variable count"),
        ("order", "ordered variable-name SHA256"),
        ("fingerprint", "Fingerprint"),
    ],
)
def test_complete_start_rejects_every_model_identity_mismatch(
    mismatch: str,
    message: str,
) -> None:
    names = ("x[0]", "x[1]", "x[2]")
    source = _BulkTransferModel(names, (1.0, 0.0, 2.0), fingerprint=17)
    incumbent = _bulk_transfer_adapter(source).capture_complete_incumbent()
    fresh_names = names
    fresh_fingerprint = 17
    if mismatch == "count":
        fresh_names = names[:-1]
    elif mismatch == "order":
        fresh_names = (names[1], names[0], names[2])
    elif mismatch == "fingerprint":
        fresh_fingerprint = 18
    fresh = _BulkTransferModel(
        fresh_names,
        (0.0,) * len(fresh_names),
        fingerprint=fresh_fingerprint,
    )

    with pytest.raises(ValueError, match=message):
        _bulk_transfer_adapter(fresh).install_complete_incumbent(incumbent)

    assert fresh.set_attr_calls == []


@dataclass
class _FakeVariable:
    X: float


class _FakeExtractionGRB:
    OPTIMAL = 2


def test_mlnd_adapter_extracts_only_open_h_with_material_physical_workload() -> None:
    lockers = ("H0", "H1", "H2", "H3", "H4")
    physical_arcs = {
        ("L0", "H0"),
        ("H0", "H1"),
        ("H2", "L0"),
        ("H4", "L0"),
    }
    instance = SimpleNamespace(
        H=lockers,
        A_F=physical_arcs,
        total_demand=22_897.5,
        config=SimpleNamespace(availability_levels=(1.0,)),
        facility_cost=lambda _node: 1.0,
    )
    wrapper = SimpleNamespace(
        inst=instance,
        chi_keys=(("o", "d", "H0", "H1", 0),),
        vars={
            "x": {
                "H0": _FakeVariable(1.0),
                "H1": _FakeVariable(1.0),
                "H2": _FakeVariable(1.0),
                "H3": _FakeVariable(1.0),
                "H4": _FakeVariable(0.0),
            },
            "chi": {("o", "d", "H0", "H1", 0): _FakeVariable(2.0)},
            "psi": {
                ("L0", "H0"): _FakeVariable(2.0),
                ("H0", "H1"): _FakeVariable(2.0),
                ("H2", "L0"): _FakeVariable(0.2),
                ("H4", "L0"): _FakeVariable(3.0),
                # Virtual state flow is deliberately present but not in A_F.
                ("H3", "S3"): _FakeVariable(10.0),
            },
        },
    )
    adapter = MLNDQualificationAdapter.__new__(MLNDQualificationAdapter)
    adapter.wrapper = wrapper
    adapter.model = SimpleNamespace(
        Status=_FakeExtractionGRB.OPTIMAL,
        SolCount=1,
        # The auxiliary feasibility objective is zero; reported economics must
        # still come from the displaced original objective expression.
        ObjVal=0.0,
        Params=SimpleNamespace(FeasibilityTol=1e-5),
    )
    adapter._GRB = _FakeExtractionGRB
    adapter._temporary_constraint_token = TemporaryWitnessConstraintToken(
        service_constraint=object(),
        objective_constraint=object(),
        original_objective=_FakeExpression("objective", 900.0),
        original_model_sense=1,
    )

    solution = adapter.extract_solution()

    assert solution.objective == pytest.approx(900.0)
    assert solution.open_h == 4
    assert solution.active_h == 2
    assert solution.active_h_threshold_per_hour == pytest.approx(0.228975)
    assert solution.realized_service_per_hour == pytest.approx(2.0)


def test_qualification_solution_rejects_active_h_above_open_h() -> None:
    with pytest.raises(ValueError, match="active_h"):
        QualificationSolution(
            status="OPTIMAL",
            has_incumbent=True,
            objective=1.0,
            realized_service_per_hour=1.0,
            open_h=1,
            active_h=2,
        )


class _FakeExpression:
    def __init__(self, label: str, value: float = 0.0):
        self.label = label
        self.value = value

    def copy(self) -> _FakeExpression:
        return _FakeExpression(self.label, self.value)

    def getValue(self) -> float:
        return self.value

    def __ge__(self, value: float) -> tuple[str, str, float]:
        return ("ge", self.label, value)

    def __le__(self, value: float) -> tuple[str, str, float]:
        return ("le", self.label, value)


class _FakeTransactionalModel:
    def __init__(
        self,
        *,
        fail_on_add_number: int | None = None,
        fail_on_remove_number: int | None = None,
        fail_on_set_objective_number: int | None = None,
        fail_on_update_number: int | None = None,
    ) -> None:
        self.ModelSense = 1
        self.fail_on_add_number = fail_on_add_number
        self.fail_on_remove_number = fail_on_remove_number
        self.fail_on_set_objective_number = fail_on_set_objective_number
        self.fail_on_update_number = fail_on_update_number
        self.add_attempts = 0
        self.remove_attempts = 0
        self.set_objective_attempts = 0
        self.update_attempts = 0
        self.created: list[object] = []
        self.removed_batches: list[list[object]] = []
        self.original_objective = _FakeExpression("objective", 900.0)
        self.current_objective = self.original_objective
        self.objective_events: list[tuple[str, int]] = []

    def getObjective(self) -> _FakeExpression:
        return self.current_objective

    def setObjective(self, expression: _FakeExpression, sense: int) -> None:
        self.set_objective_attempts += 1
        if self.set_objective_attempts == self.fail_on_set_objective_number:
            raise RuntimeError("synthetic set objective failure")
        self.current_objective = expression
        self.ModelSense = sense
        self.objective_events.append((expression.label, sense))

    def addConstr(self, expression: object, *, name: str) -> object:
        self.add_attempts += 1
        if self.add_attempts == self.fail_on_add_number:
            raise RuntimeError("synthetic add failure")
        constraint = {"expression": expression, "name": name}
        self.created.append(constraint)
        return constraint

    def remove(self, constraints: list[object]) -> None:
        self.remove_attempts += 1
        if self.remove_attempts == self.fail_on_remove_number:
            raise RuntimeError("synthetic remove failure")
        self.removed_batches.append(list(constraints))

    def update(self) -> None:
        self.update_attempts += 1
        if self.update_attempts == self.fail_on_update_number:
            raise RuntimeError("synthetic update failure")


class _FakeGRB:
    MINIMIZE = 1


class _FakeGP:
    @staticmethod
    def LinExpr(value: float) -> _FakeExpression:
        return _FakeExpression("zero", value)


def _transaction_adapter(model: _FakeTransactionalModel) -> MLNDQualificationAdapter:
    adapter = MLNDQualificationAdapter.__new__(MLNDQualificationAdapter)
    adapter.model = model
    adapter._GRB = _FakeGRB
    adapter._gp = _FakeGP
    adapter._temporary_constraint_token = None
    adapter._realized_service_expression = lambda: _FakeExpression("service")
    return adapter


def test_adapter_adds_and_removes_service_and_objective_rows_as_one_token() -> None:
    model = _FakeTransactionalModel()
    adapter = _transaction_adapter(model)

    token = adapter.add_witness_search_constraints(0.1, 999.0)

    assert [constraint["name"] for constraint in model.created] == [
        "__qualification_minimum_realized_service",
        "__qualification_maximum_objective",
    ]
    assert [constraint["expression"] for constraint in model.created] == [
        ("ge", "service", 0.1),
        ("le", "objective", 999.0),
    ]
    assert model.current_objective.label == "zero"
    assert token.original_objective is not model.original_objective
    assert token.original_objective.label == "objective"
    assert token.original_model_sense == _FakeGRB.MINIMIZE
    adapter.remove_temporary_constraint(token)
    assert model.removed_batches == [model.created]
    assert model.current_objective.label == "objective"
    assert model.ModelSense == _FakeGRB.MINIMIZE
    assert adapter._temporary_constraint_token is None


@pytest.mark.parametrize(
    (
        "fail_on_add_number",
        "fail_on_set_objective_number",
        "fail_on_update_number",
        "removed_count",
    ),
    [
        (2, None, None, 1),
        (None, 1, None, 2),
        (None, None, 1, 2),
    ],
)
def test_adapter_rolls_back_every_added_row_when_transaction_creation_fails(
    fail_on_add_number: int | None,
    fail_on_set_objective_number: int | None,
    fail_on_update_number: int | None,
    removed_count: int,
) -> None:
    model = _FakeTransactionalModel(
        fail_on_add_number=fail_on_add_number,
        fail_on_set_objective_number=fail_on_set_objective_number,
        fail_on_update_number=fail_on_update_number,
    )
    adapter = _transaction_adapter(model)

    with pytest.raises(RuntimeError, match="synthetic"):
        adapter.add_witness_search_constraints(0.1, 999.0)

    assert len(model.removed_batches) == 1
    assert len(model.removed_batches[0]) == removed_count
    assert model.current_objective.label == "objective"
    assert model.ModelSense == _FakeGRB.MINIMIZE
    assert adapter._temporary_constraint_token is None


@pytest.mark.parametrize("failure_stage", ["remove", "update"])
def test_adapter_restores_objective_but_keeps_token_if_cleanup_fails(
    failure_stage: str,
) -> None:
    model = _FakeTransactionalModel(
        fail_on_remove_number=1 if failure_stage == "remove" else None,
        # Setup consumes update #1; fail the cleanup update instead.
        fail_on_update_number=2 if failure_stage == "update" else None,
    )
    adapter = _transaction_adapter(model)
    token = adapter.add_witness_search_constraints(0.1, 999.0)

    with pytest.raises(RuntimeError, match=f"synthetic {failure_stage} failure"):
        adapter.remove_temporary_constraint(token)

    assert model.current_objective.label == "objective"
    assert model.ModelSense == _FakeGRB.MINIMIZE
    assert adapter._temporary_constraint_token is token
