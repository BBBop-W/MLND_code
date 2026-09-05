from __future__ import annotations

from dataclasses import dataclass, field, replace

import pytest

from src.benchmark.branch_cut.qualification_incumbent_reuse import (
    DevelopmentQualificationIncumbentPool,
    QualificationReuseProvenance,
    build_qualification_constraint_signature,
)
from src.benchmark.branch_cut.qualification_runner import (
    CompleteIncumbent,
    MLNDQualificationAdapter,
    NativeQualificationSettings,
    QualificationInfrastructureError,
    QualificationModelAudit,
    QualificationModelContract,
    QualificationSolution,
    qualify_canonical_instance,
)


_INSTANCE_SHA256 = "a" * 64
_SOURCE_CONFIG_SHA256 = "b" * 64
_TARGET_CONFIG_SHA256 = "c" * 64
_SOURCE_CASE_SHA256 = "d" * 64
_TARGET_CASE_SHA256 = "e" * 64


def _constraint_signature(*, rhs: float = 5.0, upper_bound: float = 10.0):
    return build_qualification_constraint_signature(
        variable_records=(
            {
                "index": 0,
                "name": "x[H0]",
                "vtype": "B",
                "lower_bound": 0.0,
                "upper_bound": 1.0,
            },
            {
                "index": 1,
                "name": "chi[o,d,i,j,0]",
                "vtype": "C",
                "lower_bound": 0.0,
                "upper_bound": upper_bound,
            },
            {
                "index": 2,
                "name": "queue_t[H0]",
                "vtype": "C",
                "lower_bound": 0.0,
                "upper_bound": 100.0,
            },
        ),
        linear_constraint_records=(
            {
                "index": 0,
                "name": "flow",
                "sense": "<",
                "rhs": rhs,
                "lazy": 0,
                "terms": ((0, 1.0), (1, 2.0)),
            },
        ),
        quadratic_constraint_records=(
            {
                "index": 0,
                "name": "queue_soc",
                "sense": "<",
                "rhs": 0.0,
                "linear_terms": ((2, -1.0),),
                "quadratic_terms": ((1, 1, 1.0),),
            },
        ),
        sos_constraint_records=(),
        unsupported_general_constraint_count=0,
        unsupported_nonlinear_constraint_count=0,
    )


def _provenance(
    *,
    profile_id: str,
    config_sha256: str,
    case_sha256: str,
    demand_scale: float = 1.0,
    instance_sha256: str = _INSTANCE_SHA256,
) -> QualificationReuseProvenance:
    return QualificationReuseProvenance(
        qualification_run_id="development-qualification-run",
        case_id=f"case-{profile_id}",
        profile_id=profile_id,
        scale="small",
        mode="MX",
        physical_instance_sha256=instance_sha256,
        qualification_case_sha256=case_sha256,
        effective_model_config_sha256=config_sha256,
        demand_scale=demand_scale,
        screening_is_exact=False,
    )


def _incumbent(*, model_fingerprint: int = 101) -> CompleteIncumbent:
    return CompleteIncumbent.from_ordered_values(
        objective=900.0,
        variable_names=("x[H0]", "chi[o,d,i,j,0]", "queue_t[H0]"),
        values=(1.0, 5.0, 0.5),
        model_fingerprint=model_fingerprint,
    )


def test_constraint_signature_excludes_objective_but_binds_every_feasibility_axis() -> (
    None
):
    first = _constraint_signature()
    # There is deliberately no objective argument: changing objective-only
    # costs cannot change the feasibility signature.
    second = _constraint_signature()

    assert first == second
    assert first.objective_excluded is True
    assert first.extraction_complete is True
    assert first.variable_count == 3
    assert first.linear_constraint_count == 1
    assert first.quadratic_constraint_count == 1
    assert _constraint_signature(rhs=5.1) != first
    assert _constraint_signature(upper_bound=9.0) != first


def test_constraint_signature_accepts_one_shot_record_streams() -> None:
    signature = build_qualification_constraint_signature(
        variable_records=(
            record
            for record in (
                {
                    "index": 0,
                    "name": "x",
                    "vtype": "B",
                    "lower_bound": 0.0,
                    "upper_bound": 1.0,
                },
            )
        ),
        linear_constraint_records=(record for record in ()),
        quadratic_constraint_records=(record for record in ()),
        sos_constraint_records=(record for record in ()),
    )

    assert signature.variable_count == 1
    assert signature.linear_constraint_count == 0


@dataclass
class _FakeVar:
    index: int
    VarName: str
    VType: str
    LB: float
    UB: float
    Start: float = 0.0


@dataclass
class _FakeLinearExpression:
    terms: tuple[tuple[_FakeVar, float], ...]

    def size(self) -> int:
        return len(self.terms)

    def getVar(self, index: int) -> _FakeVar:
        return self.terms[index][0]

    def getCoeff(self, index: int) -> float:
        return self.terms[index][1]


@dataclass
class _FakeQuadraticExpression:
    linear: _FakeLinearExpression
    terms: tuple[tuple[_FakeVar, _FakeVar, float], ...]

    def getLinExpr(self) -> _FakeLinearExpression:
        return self.linear

    def size(self) -> int:
        return len(self.terms)

    def getVar1(self, index: int) -> _FakeVar:
        return self.terms[index][0]

    def getVar2(self, index: int) -> _FakeVar:
        return self.terms[index][1]

    def getCoeff(self, index: int) -> float:
        return self.terms[index][2]


@dataclass(frozen=True)
class _FakeLinearConstraint:
    ConstrName: str
    Sense: str
    RHS: float
    Lazy: int = 0


@dataclass(frozen=True)
class _FakeQuadraticConstraint:
    QCName: str
    QCSense: str
    QCRHS: float


class _FakeGRB:
    UNDEFINED = 1e101

    class Attr:
        VarName = "VarName"
        VType = "VType"
        LB = "LB"
        UB = "UB"
        Start = "Start"


@dataclass
class _FakeParams:
    StartNumber: int = 0


class _FakeConstraintModel:
    def __init__(self, *, linear_rhs: float = 5.0) -> None:
        self.variables = (
            _FakeVar(0, "x[H0]", "B", 0.0, 1.0),
            _FakeVar(1, "chi[o,d,i,j,0]", "C", 0.0, 10.0),
            _FakeVar(2, "queue_t[H0]", "C", 0.0, 100.0),
        )
        self.linear_constraint = _FakeLinearConstraint("flow", "<", linear_rhs)
        self.quadratic_constraint = _FakeQuadraticConstraint("queue_soc", "<", 0.0)
        self.linear_row = _FakeLinearExpression(
            ((self.variables[0], 1.0), (self.variables[1], 2.0))
        )
        self.quadratic_row = _FakeQuadraticExpression(
            _FakeLinearExpression(((self.variables[2], -1.0),)),
            ((self.variables[1], self.variables[1], 1.0),),
        )
        self.NumConstrs = 1
        self.NumQConstrs = 1
        self.NumVars = 3
        self.NumGenConstrs = 0
        self.NumNLConstrs = 0
        self.NumSOS = 0
        self.NumScenarios = 0
        self.NumStart = 0
        self.Params = _FakeParams()
        self.reset_calls = 0

    def update(self) -> None:
        return None

    def getVars(self) -> list[_FakeVar]:
        return list(self.variables)

    def getAttr(self, attribute: str, values: object = None):
        if values is None:
            return getattr(self, attribute)
        return [getattr(value, attribute) for value in values]

    def setAttr(self, attribute: str, variables: object, values: object) -> None:
        for variable, value in zip(variables, values, strict=True):
            setattr(variable, attribute, float(value))

    def getConstrs(self) -> list[_FakeLinearConstraint]:
        return [self.linear_constraint]

    def getRow(self, constraint: _FakeLinearConstraint) -> _FakeLinearExpression:
        assert constraint is self.linear_constraint
        return self.linear_row

    def getQConstrs(self) -> list[_FakeQuadraticConstraint]:
        return [self.quadratic_constraint]

    def getQCRow(
        self, constraint: _FakeQuadraticConstraint
    ) -> _FakeQuadraticExpression:
        assert constraint is self.quadratic_constraint
        return self.quadratic_row

    def getSOSs(self) -> list[object]:
        return []

    def reset(self, clearall: int) -> None:
        assert clearall == 0
        self.reset_calls += 1


def _raw_mlnd_adapter(model: _FakeConstraintModel) -> MLNDQualificationAdapter:
    adapter = MLNDQualificationAdapter.__new__(MLNDQualificationAdapter)
    adapter.model = model
    adapter._GRB = _FakeGRB
    adapter._temporary_constraint_token = None
    adapter._constraint_signature_cache = None
    return adapter


def test_production_adapter_hashes_constraints_and_installs_without_objective_fingerprint() -> (
    None
):
    target_model = _FakeConstraintModel()
    target_adapter = _raw_mlnd_adapter(target_model)
    target_signature = target_adapter.qualification_constraint_signature()
    repeated = target_adapter.qualification_constraint_signature()
    drifted = _raw_mlnd_adapter(
        _FakeConstraintModel(linear_rhs=6.0)
    ).qualification_constraint_signature()

    assert repeated == target_signature
    assert drifted != target_signature
    incumbent = _incumbent(model_fingerprint=999)
    target_adapter.install_constraint_equivalent_complete_incumbent(
        incumbent,
        target_signature,
    )
    assert target_model.NumStart == 1
    assert tuple(variable.Start for variable in target_model.variables) == tuple(
        incumbent.values
    )
    assert target_model.reset_calls == 1


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"unsupported_general_constraint_count": 1}, "general constraints"),
        ({"unsupported_nonlinear_constraint_count": 1}, "nonlinear constraints"),
    ],
)
def test_constraint_signature_fails_closed_on_unsupported_constructs(
    kwargs: dict[str, int], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        build_qualification_constraint_signature(
            variable_records=(),
            linear_constraint_records=(),
            quadratic_constraint_records=(),
            sos_constraint_records=(),
            **kwargs,
        )


def test_pool_reuses_only_a_fresh_validated_incumbent_across_distinct_profiles() -> (
    None
):
    pool = DevelopmentQualificationIncumbentPool()
    signature = _constraint_signature()
    source = _provenance(
        profile_id="source-cost-profile",
        config_sha256=_SOURCE_CONFIG_SHA256,
        case_sha256=_SOURCE_CASE_SHA256,
    )
    target = _provenance(
        profile_id="target-cost-profile",
        config_sha256=_TARGET_CONFIG_SHA256,
        case_sha256=_TARGET_CASE_SHA256,
    )

    pool.offer(
        provenance=source,
        constraint_signature=signature,
        incumbent=_incumbent(),
        accepted=True,
        fresh_validation_performed=True,
    )
    selected = pool.select(
        target_provenance=target,
        target_constraint_signature=signature,
    )

    assert selected is not None
    assert selected.provenance == source
    assert selected.constraint_signature == signature
    assert selected.reuse_record_sha256
    pool.record_installation(
        target_provenance=target,
        target_constraint_signature=signature,
        selected=selected,
    )
    payload = pool.to_payload()
    assert payload["development_only"] is True
    assert payload["eligible_for_formal_pool"] is False
    assert payload["eligible_for_algorithm_comparison"] is False
    assert len(payload["reuse_audit_sha256"]) == 64
    assert len(pool.case_audit_payload(target.case_id)["case_reuse_audit_sha256"]) == 64
    assert [event["action"] for event in payload["audit_events"]] == [
        "incumbent_offered",
        "reuse_candidate_selected",
        "reuse_start_installed",
    ]


@pytest.mark.parametrize(
    "target",
    [
        _provenance(
            profile_id="different-demand",
            config_sha256=_TARGET_CONFIG_SHA256,
            case_sha256=_TARGET_CASE_SHA256,
            demand_scale=2.0,
        ),
        _provenance(
            profile_id="different-instance",
            config_sha256=_TARGET_CONFIG_SHA256,
            case_sha256=_TARGET_CASE_SHA256,
            instance_sha256="f" * 64,
        ),
    ],
)
def test_pool_refuses_non_equivalent_instance_or_demand(
    target: QualificationReuseProvenance,
) -> None:
    pool = DevelopmentQualificationIncumbentPool()
    signature = _constraint_signature()
    pool.offer(
        provenance=_provenance(
            profile_id="source",
            config_sha256=_SOURCE_CONFIG_SHA256,
            case_sha256=_SOURCE_CASE_SHA256,
        ),
        constraint_signature=signature,
        incumbent=_incumbent(),
        accepted=True,
        fresh_validation_performed=True,
    )

    assert (
        pool.select(
            target_provenance=target,
            target_constraint_signature=signature,
        )
        is None
    )
    assert pool.audit_events[-1].action == "reuse_cache_miss"


def test_pool_refuses_constraint_drift_and_same_profile_replay() -> None:
    pool = DevelopmentQualificationIncumbentPool()
    signature = _constraint_signature()
    source = _provenance(
        profile_id="source",
        config_sha256=_SOURCE_CONFIG_SHA256,
        case_sha256=_SOURCE_CASE_SHA256,
    )
    pool.offer(
        provenance=source,
        constraint_signature=signature,
        incumbent=_incumbent(),
        accepted=True,
        fresh_validation_performed=True,
    )

    assert (
        pool.select(
            target_provenance=replace(
                source,
                case_id="same-profile-second-job",
            ),
            target_constraint_signature=signature,
        )
        is None
    )
    assert (
        pool.select(
            target_provenance=_provenance(
                profile_id="target",
                config_sha256=_TARGET_CONFIG_SHA256,
                case_sha256=_TARGET_CASE_SHA256,
            ),
            target_constraint_signature=_constraint_signature(rhs=6.0),
        )
        is None
    )
    assert pool.audit_events[-1].action == "reuse_cache_miss"


def test_pool_rejects_unvalidated_source_and_formal_or_exact_provenance() -> None:
    signature = _constraint_signature()
    source = _provenance(
        profile_id="source",
        config_sha256=_SOURCE_CONFIG_SHA256,
        case_sha256=_SOURCE_CASE_SHA256,
    )
    pool = DevelopmentQualificationIncumbentPool()
    with pytest.raises(ValueError, match="fresh validated accepted"):
        pool.offer(
            provenance=source,
            constraint_signature=signature,
            incumbent=_incumbent(),
            accepted=False,
            fresh_validation_performed=True,
        )
    with pytest.raises(ValueError, match="screening_is_exact=false"):
        replace(source, screening_is_exact=True)
    with pytest.raises(ValueError, match="formal-pool ineligible"):
        replace(source, eligible_for_formal_pool=True)
    with pytest.raises(ValueError, match="algorithm-comparison ineligible"):
        replace(source, eligible_for_algorithm_comparison=True)


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


@dataclass
class _Adapter:
    audit: QualificationModelAudit
    signature: object
    events: list[str] = field(default_factory=list)
    current_solution: QualificationSolution | None = None
    disposed: bool = False

    def qualification_model_audit(self) -> QualificationModelAudit:
        self.events.append("model_audit")
        return self.audit

    def qualification_constraint_signature(self):
        self.events.append("constraint_signature")
        return self.signature

    def courier_only_objective(self) -> float:
        self.events.append("courier_baseline")
        return 1_000.0

    def total_demand_per_hour(self) -> float:
        return 100.0

    def candidate_h_count(self) -> int:
        return 20

    def clear_mip_starts(self) -> None:
        self.events.append("clear_mip_starts")

    def install_constraint_equivalent_complete_incumbent(
        self, incumbent: CompleteIncumbent, expected_constraint_signature: object
    ) -> None:
        self.events.append("install_reused_start")
        assert expected_constraint_signature == self.signature
        assert incumbent.variable_count == 3

    def solve_native(self, settings: NativeQualificationSettings) -> str:
        assert settings.threads == 1
        self.events.append("solve_native")
        self.current_solution = _positive_solution()
        return self.current_solution.status

    def extract_solution(self) -> QualificationSolution:
        assert self.current_solution is not None
        return self.current_solution

    def capture_complete_incumbent(self) -> CompleteIncumbent:
        self.events.append("capture_complete_incumbent")
        return _incumbent(model_fingerprint=int(self.audit.model_fingerprint))

    def install_complete_incumbent(self, incumbent: CompleteIncumbent) -> None:
        self.events.append("install_complete_incumbent")

    def add_witness_search_constraints(self, *_args: object) -> object:
        raise AssertionError("positive solution must not need constrained search")

    def remove_temporary_constraint(self, _token: object) -> None:
        raise AssertionError("positive solution must not add temporary rows")

    def plan_geometry_start(self, **_kwargs: object) -> object:
        raise AssertionError("positive solution must not need geometry start")

    def install_geometry_start(self, _plan: object) -> object:
        raise AssertionError("positive solution must not need geometry start")

    def solve_geometry_start(self, *_args: object) -> str:
        raise AssertionError("positive solution must not need geometry start")

    def clear_geometry_start(self) -> None:
        raise AssertionError("positive solution must not need geometry start")

    def dispose(self) -> None:
        self.events.append("dispose")
        self.disposed = True


@dataclass
class _Factory:
    adapters: list[_Adapter]

    def __call__(self, _instance: object, _mode: str) -> _Adapter:
        return self.adapters.pop(0)


def _contract(
    *, config_sha256: str, case_sha256: str, fixed_cost: float
) -> QualificationModelContract:
    return QualificationModelContract(
        qualification_case_sha256=case_sha256,
        effective_model_config_sha256=config_sha256,
        demand_scale=1.0,
        locker_fixed_cost_per_hour=fixed_cost,
        display_origin_choices=8,
        general_corridor_choices=10,
        screening_is_exact=False,
    )


def _audit(
    *, config_sha256: str, fixed_cost: float, fingerprint: int
) -> QualificationModelAudit:
    return QualificationModelAudit(
        effective_model_config_sha256=config_sha256,
        demand_scale=1.0,
        locker_fixed_cost_per_hour=fixed_cost,
        display_origin_choices=8,
        general_corridor_choices=10,
        screening_is_exact=False,
        model_fingerprint=fingerprint,
    )


def test_runner_installs_reused_start_only_after_exact_constraint_match() -> None:
    pool = DevelopmentQualificationIncumbentPool()
    signature = _constraint_signature()
    source_contract = _contract(
        config_sha256=_SOURCE_CONFIG_SHA256,
        case_sha256=_SOURCE_CASE_SHA256,
        fixed_cost=24.0,
    )
    source_witness = _Adapter(
        _audit(
            config_sha256=_SOURCE_CONFIG_SHA256,
            fixed_cost=24.0,
            fingerprint=101,
        ),
        signature,
    )
    source_fresh = _Adapter(source_witness.audit, signature)
    source_outcome = qualify_canonical_instance(
        object(),
        "MX",
        case_id="source-job",
        scale="small",
        qualification_run_id="development-qualification-run",
        instance_sha256=_INSTANCE_SHA256,
        adapter_factory=_Factory([source_witness, source_fresh]),
        expected_model_contract=source_contract,
        development_incumbent_reuse_pool=pool,
        development_reuse_profile_id="source-profile",
    )
    assert source_outcome.accepted
    assert "clear_mip_starts" in source_witness.events

    target_contract = _contract(
        config_sha256=_TARGET_CONFIG_SHA256,
        case_sha256=_TARGET_CASE_SHA256,
        fixed_cost=28.0,
    )
    target_witness = _Adapter(
        _audit(
            config_sha256=_TARGET_CONFIG_SHA256,
            fixed_cost=28.0,
            fingerprint=202,
        ),
        signature,
    )
    target_fresh = _Adapter(target_witness.audit, signature)
    target_outcome = qualify_canonical_instance(
        object(),
        "MX",
        case_id="target-job",
        scale="small",
        qualification_run_id="development-qualification-run",
        instance_sha256=_INSTANCE_SHA256,
        adapter_factory=_Factory([target_witness, target_fresh]),
        expected_model_contract=target_contract,
        development_incumbent_reuse_pool=pool,
        development_reuse_profile_id="target-profile",
    )

    assert target_outcome.accepted
    assert "install_reused_start" in target_witness.events
    assert "clear_mip_starts" not in target_witness.events
    actions = [event.action for event in pool.audit_events]
    assert actions.count("reuse_start_installed") == 1
    assert actions.count("incumbent_offered") == 2


def test_runner_rechecks_actual_witness_signature_before_select_install_or_solve() -> (
    None
):
    pool = DevelopmentQualificationIncumbentPool()
    preflight_signature = _constraint_signature(rhs=5.0)
    pool.offer(
        provenance=_provenance(
            profile_id="source-profile",
            config_sha256=_SOURCE_CONFIG_SHA256,
            case_sha256=_SOURCE_CASE_SHA256,
        ),
        constraint_signature=preflight_signature,
        incumbent=_incumbent(),
        accepted=True,
        fresh_validation_performed=True,
    )
    target_contract = _contract(
        config_sha256=_TARGET_CONFIG_SHA256,
        case_sha256=_TARGET_CASE_SHA256,
        fixed_cost=28.0,
    )
    actual_witness = _Adapter(
        _audit(
            config_sha256=_TARGET_CONFIG_SHA256,
            fixed_cost=28.0,
            fingerprint=202,
        ),
        _constraint_signature(rhs=6.0),
    )

    with pytest.raises(
        QualificationInfrastructureError,
        match="actual witness adapter constraint signature differs",
    ):
        qualify_canonical_instance(
            object(),
            "MX",
            case_id="target-job",
            scale="small",
            qualification_run_id="development-qualification-run",
            instance_sha256=_INSTANCE_SHA256,
            adapter_factory=_Factory([actual_witness]),
            expected_model_contract=target_contract,
            fail_fast_infrastructure=True,
            development_incumbent_reuse_pool=pool,
            development_reuse_profile_id="target-profile",
            development_expected_target_constraint_signature=preflight_signature,
        )

    assert "constraint_signature" in actual_witness.events
    assert "install_reused_start" not in actual_witness.events
    assert "clear_mip_starts" not in actual_witness.events
    assert "solve_native" not in actual_witness.events
    assert actual_witness.events[-1] == "dispose"
    assert all(
        event.target_provenance.case_id != "target-job" for event in pool.audit_events
    )


def test_no_reuse_path_remains_signature_neutral() -> None:
    contract = _contract(
        config_sha256=_TARGET_CONFIG_SHA256,
        case_sha256=_TARGET_CASE_SHA256,
        fixed_cost=28.0,
    )
    witness = _Adapter(
        _audit(
            config_sha256=_TARGET_CONFIG_SHA256,
            fixed_cost=28.0,
            fingerprint=202,
        ),
        _constraint_signature(rhs=6.0),
    )
    fresh = _Adapter(witness.audit, witness.signature)

    outcome = qualify_canonical_instance(
        object(),
        "MX",
        case_id="ordinary-no-reuse-job",
        scale="small",
        qualification_run_id="ordinary-qualification-run",
        instance_sha256=_INSTANCE_SHA256,
        adapter_factory=_Factory([witness, fresh]),
        expected_model_contract=contract,
    )

    assert outcome.accepted
    assert "constraint_signature" not in witness.events
    assert "solve_native" in witness.events


def test_runner_clears_starts_when_constraint_signature_does_not_match() -> None:
    pool = DevelopmentQualificationIncumbentPool()
    source = _provenance(
        profile_id="source-profile",
        config_sha256=_SOURCE_CONFIG_SHA256,
        case_sha256=_SOURCE_CASE_SHA256,
    )
    pool.offer(
        provenance=source,
        constraint_signature=_constraint_signature(),
        incumbent=_incumbent(),
        accepted=True,
        fresh_validation_performed=True,
    )
    target_contract = _contract(
        config_sha256=_TARGET_CONFIG_SHA256,
        case_sha256=_TARGET_CASE_SHA256,
        fixed_cost=28.0,
    )
    target_signature = _constraint_signature(rhs=6.0)
    target_witness = _Adapter(
        _audit(
            config_sha256=_TARGET_CONFIG_SHA256,
            fixed_cost=28.0,
            fingerprint=202,
        ),
        target_signature,
    )
    target_fresh = _Adapter(target_witness.audit, target_signature)

    outcome = qualify_canonical_instance(
        object(),
        "MX",
        case_id="target-job",
        scale="small",
        qualification_run_id="development-qualification-run",
        instance_sha256=_INSTANCE_SHA256,
        adapter_factory=_Factory([target_witness, target_fresh]),
        expected_model_contract=target_contract,
        development_incumbent_reuse_pool=pool,
        development_reuse_profile_id="target-profile",
    )

    assert outcome.accepted
    assert "clear_mip_starts" in target_witness.events
    assert "install_reused_start" not in target_witness.events
    assert any(
        event.action == "reuse_cache_miss"
        and event.reason == "no_exact_constraint_equivalent_source"
        for event in pool.audit_events
    )
