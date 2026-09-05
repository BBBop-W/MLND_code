from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.benchmark.branch_cut.formal_solution_evidence import (
    FormalQualificationTrace,
    bind_qualification_trace,
    build_solution_structure_from_facts,
    no_incumbent_solution_structure,
    qualification_trace_from_context,
    recompute_solution_structure,
    validate_solution_structure_payload,
)
from src.benchmark.branch_cut.hashing import canonical_sha256
from src.benchmark.branch_cut.qualification import (
    QualificationEvidence,
    QualificationReason,
    assess_qualification,
)


class _Var:
    def __init__(self, value: float) -> None:
        self.X = value


class _Instance:
    H = ("H0", "H1", "H2", "H3")
    A_F = {
        ("L0", "H0"),
        ("H0", "H1"),
        ("H2", "E0"),
    }
    markets = (("O0", "D0"),)
    h_od = {("O0", "D0"): 100.0}
    total_demand = 100.0
    config = SimpleNamespace(availability_levels=(0.5, 0.8))

    @staticmethod
    def direct_rider_cost(origin: str, destination: str) -> float:
        assert (origin, destination) == ("O0", "D0")
        return 10.0

    @staticmethod
    def facility_cost(node: str) -> float:
        return 0.0 if node == "E0" else 5.0


def _trace() -> FormalQualificationTrace:
    decision = assess_qualification(
        QualificationEvidence(
            case_id="case-small-00",
            scale="small",
            courier_only_objective=1_000.0,
            witness_objective=900.0,
            total_demand_per_hour=100.0,
            witness_service_per_hour=20.0,
            open_facilities=3,
            positive_cost_open_facilities=3,
            candidate_h=4,
            open_h=2,
            witness_feasible=True,
            qualification_run_id="qualification-run-00",
            instance_sha256="1" * 64,
            active_h=2,
            active_h_threshold_per_hour=0.001,
            qualification_case_sha256="2" * 64,
            effective_model_config_sha256="3" * 64,
            screening_is_exact=False,
        )
    )
    assert decision.accepted
    return FormalQualificationTrace.from_decision(
        decision,
        manifest_id="manifest-" + "4" * 64,
        benchmark_manifest_sha256="5" * 64,
        manifest_artifact_sha256="6" * 64,
    )


def _wrapper(*, sol_count: int = 1, objective: float = 950.0):
    chi_key = ("O0", "D0", "H0", "H1", 0)
    return SimpleNamespace(
        model=SimpleNamespace(SolCount=sol_count, ObjVal=objective),
        inst=_Instance(),
        courier_only_baseline=1_000.0,
        chi_keys=(chi_key,),
        vars={
            "x": {
                "L0": _Var(1.0),
                "H0": _Var(1.0),
                "H1": _Var(1.0),
                "H2": _Var(0.0),
                "H3": _Var(0.0),
                "E0": _Var(0.0),
            },
            "chi": {chi_key: _Var(40.0)},
            "psi": {
                ("L0", "H0"): _Var(10.0),
                ("H0", "H1"): _Var(5.0),
                ("H2", "E0"): _Var(0.0),
            },
        },
    )


def test_recomputes_formal_solution_structure_from_actual_incumbent() -> None:
    trace = _trace()
    structure = recompute_solution_structure(
        _wrapper(),
        trace,
        expected_has_incumbent=True,
    )

    assert structure.C0 == pytest.approx(1_000.0)
    assert structure.open_H_count == 2
    assert structure.active_H_count == 2
    assert structure.realized_service_per_hour == pytest.approx(20.0)
    assert structure.realized_service_rate == pytest.approx(0.2)
    assert structure.nontriviality_passed is True
    assert structure.nontriviality_criteria["all_criteria_met"] is True
    assert validate_solution_structure_payload(
        structure.to_payload(),
        trace,
        expected_has_incumbent=True,
    ) == structure


def test_no_incumbent_payload_is_explicit_and_status_consistent() -> None:
    trace = _trace()
    structure = recompute_solution_structure(
        _wrapper(sol_count=0),
        trace,
        expected_has_incumbent=False,
    )
    assert structure == no_incumbent_solution_structure(trace)
    assert structure.nontriviality_evaluated is False
    assert structure.C0 is None
    with pytest.raises(ValueError, match="incumbent flags differ"):
        validate_solution_structure_payload(
            structure.to_payload(),
            trace,
            expected_has_incumbent=True,
        )


def test_solution_structure_rejects_self_rehashed_metric_tampering() -> None:
    trace = _trace()
    payload = recompute_solution_structure(
        _wrapper(),
        trace,
        expected_has_incumbent=True,
    ).to_payload()
    payload["open_H_count"] = 3

    with pytest.raises(ValueError, match="not self-consistent"):
        validate_solution_structure_payload(
            payload,
            trace,
            expected_has_incumbent=True,
        )


def test_qualification_trace_rejects_rehashed_decision_tampering() -> None:
    payload = _trace().to_payload()
    decision = payload["qualification_decision"]
    decision["witness_service_per_hour"] = 0.0
    payload["qualification_decision_sha256"] = canonical_sha256(decision)

    with pytest.raises(ValueError, match="does not match its primitive facts"):
        FormalQualificationTrace.from_payload(payload)


def test_context_requires_exact_trace_and_hash() -> None:
    trace = _trace()
    context = bind_qualification_trace({"slot_id": "slot-00"}, trace)
    assert qualification_trace_from_context(context) == trace
    context["qualification_trace_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="SHA256 differs"):
        qualification_trace_from_context(context)


def test_all_open_incumbent_is_retained_but_fails_nontriviality() -> None:
    trace = _trace()
    structure = build_solution_structure_from_facts(
        trace,
        C0=1_000.0,
        incumbent_objective=950.0,
        total_demand_per_hour=100.0,
        realized_service_per_hour=20.0,
        open_facility_count=5,
        positive_cost_open_facility_count=5,
        candidate_H_count=4,
        open_H_count=4,
        active_H_count=4,
        active_H_threshold_per_hour=0.001,
    )

    assert structure.nontriviality_passed is False
    assert QualificationReason.H_OPEN_FRACTION_ABOVE_MAXIMUM in (
        structure.nontriviality_rejection_reasons
    )
    assert structure.nontriviality_criteria["H_open_fraction_interior"] is False


def test_independent_C0_must_match_model_and_qualification() -> None:
    wrapper = _wrapper()
    wrapper.courier_only_baseline = 999.0
    with pytest.raises(ValueError, match="recomputed C0"):
        recompute_solution_structure(
            wrapper,
            _trace(),
            expected_has_incumbent=True,
        )
