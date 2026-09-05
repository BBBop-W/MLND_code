from __future__ import annotations

from dataclasses import replace
import hashlib
import math

import pytest

from src.benchmark.branch_cut.case_builder import (
    LOCKED_QUALIFICATION_PROFILES,
    qualification_case_sha256,
)
from src.benchmark.branch_cut.manifest import (
    FORMAL_QUALIFICATION_POOL_LABEL_V5,
    OVERCOMPLETE_SELECTION_RULE,
    QUALIFICATION_MODEL_IDENTITY_SCHEMA,
    FreezeMetadata,
    freeze_manifest,
)
from src.benchmark.branch_cut.pool import CandidateCase, build_candidate_pool
from src.benchmark.branch_cut.qualification import (
    QualificationEvidence,
    QualificationPolicy,
    QualificationReason,
    assess_qualification,
    objective_margin,
    service_tolerance,
)


def _evidence(
    case_id: str = "small-block-a-s11",
    *,
    scale: str = "small",
    candidate_h: int = 20,
    open_h: int = 8,
    active_h: int | None = None,
    active_h_threshold_per_hour: float | None = None,
    instance_sha256: str | None = None,
    qualification_case_sha256: str | None = None,
    effective_model_config_sha256: str | None = None,
    screening_is_exact: bool | None = None,
) -> QualificationEvidence:
    digest = instance_sha256 or hashlib.sha256(case_id.encode()).hexdigest()
    return QualificationEvidence(
        case_id=case_id,
        scale=scale,
        courier_only_objective=100_000.0,
        witness_objective=99_800.0,
        total_demand_per_hour=10_000.0,
        witness_service_per_hour=500.0,
        open_facilities=open_h + 3,
        positive_cost_open_facilities=open_h + 3,
        candidate_h=candidate_h,
        open_h=open_h,
        witness_feasible=True,
        qualification_run_id=f"qual-{case_id}",
        instance_sha256=digest,
        active_h=open_h if active_h is None else active_h,
        active_h_threshold_per_hour=(
            service_tolerance(10_000.0)
            if active_h_threshold_per_hour is None
            else active_h_threshold_per_hour
        ),
        qualification_case_sha256=qualification_case_sha256,
        effective_model_config_sha256=effective_model_config_sha256,
        screening_is_exact=screening_is_exact,
    )


def test_protocol_tolerances_follow_fixed_formulas() -> None:
    assert objective_margin(20_000.0, 1e-6) == pytest.approx(2.0)
    assert objective_margin(100.0, 1e-3) == pytest.approx(10.0)
    assert service_tolerance(20.0) == pytest.approx(1e-3)
    assert service_tolerance(22_897.5) == pytest.approx(0.228975)
    assert service_tolerance(200_000.0) == pytest.approx(2.0)


def test_nontrivial_witness_is_accepted() -> None:
    decision = assess_qualification(_evidence())

    assert decision.accepted
    assert decision.rejection_reasons == ()
    assert decision.objective_margin == pytest.approx(10.0)
    assert decision.objective_improvement == pytest.approx(200.0)
    assert decision.savings_fraction == pytest.approx(0.002)
    assert decision.service_tolerance == pytest.approx(0.1)
    assert decision.service_fraction == pytest.approx(0.05)
    assert decision.h_open_fraction == pytest.approx(0.4)


def test_model_identity_is_all_or_none_and_propagates_to_decision() -> None:
    evidence = _evidence(
        qualification_case_sha256="a" * 64,
        effective_model_config_sha256="b" * 64,
        screening_is_exact=False,
    )

    decision = assess_qualification(evidence)

    assert decision.qualification_case_sha256 == "a" * 64
    assert decision.effective_model_config_sha256 == "b" * 64
    assert decision.screening_is_exact is False
    assert decision.to_payload()["qualification_case_sha256"] == "a" * 64
    assert decision.to_payload()["effective_model_config_sha256"] == "b" * 64
    assert decision.to_payload()["screening_is_exact"] is False

    with pytest.raises(ValueError, match="all present or all absent"):
        _evidence(qualification_case_sha256="a" * 64)
    with pytest.raises(ValueError, match="screening_is_exact must be boolean"):
        _evidence(
            qualification_case_sha256="a" * 64,
            effective_model_config_sha256="b" * 64,
            screening_is_exact=1,  # type: ignore[arg-type]
        )


def test_qualification_keeps_every_applicable_triviality_reason() -> None:
    evidence = replace(
        _evidence(),
        witness_feasible=False,
        witness_objective=100_000.0,
        witness_service_per_hour=0.0,
        open_facilities=0,
        positive_cost_open_facilities=0,
        open_h=0,
        active_h=0,
    )
    decision = assess_qualification(evidence)

    assert not decision.accepted
    assert set(decision.rejection_reasons) >= {
        QualificationReason.WITNESS_NOT_FEASIBLE,
        QualificationReason.OBJECTIVE_IMPROVEMENT_BELOW_MARGIN,
        QualificationReason.SAVINGS_FRACTION_BELOW_MINIMUM,
        QualificationReason.SERVICE_BELOW_TOLERANCE,
        QualificationReason.SERVICE_FRACTION_BELOW_MINIMUM,
        QualificationReason.NO_OPEN_FACILITY,
        QualificationReason.NO_POSITIVE_COST_OPEN_FACILITY,
        QualificationReason.NO_ACTIVE_H,
        QualificationReason.H_OPEN_FRACTION_BELOW_MINIMUM,
    }


def test_near_all_open_and_large_active_target_are_explicit_rejections() -> None:
    near_all = assess_qualification(
        _evidence(candidate_h=20, open_h=18),
    )
    assert QualificationReason.H_OPEN_FRACTION_ABOVE_MAXIMUM in (
        near_all.rejection_reasons
    )

    target_policy = QualificationPolicy(large_scale_open_h_target=(50, 60))
    below_target = assess_qualification(
        _evidence(scale="large", candidate_h=90, open_h=55, active_h=47),
        target_policy,
    )
    assert QualificationReason.LARGE_ACTIVE_H_BELOW_TARGET in (
        below_target.rejection_reasons
    )
    assert below_target.open_h == 55
    assert below_target.active_h == 47

    within_target = assess_qualification(
        _evidence(scale="large", candidate_h=90, open_h=55, active_h=51),
        target_policy,
    )
    assert within_target.accepted
    assert within_target.active_h == 51


def test_active_h_cannot_exceed_open_h_and_is_serialised() -> None:
    decision = assess_qualification(_evidence(open_h=8, active_h=9))

    assert not decision.accepted
    assert QualificationReason.INVALID_ACTIVE_H_COUNT in decision.rejection_reasons
    assert decision.to_payload()["active_h"] == 9
    assert decision.to_payload()["active_h_threshold_per_hour"] == pytest.approx(0.1)


def test_every_scale_requires_physical_throughput_at_an_active_h() -> None:
    no_active_h = assess_qualification(_evidence(scale="small", active_h=0))
    assert not no_active_h.accepted
    assert QualificationReason.NO_ACTIVE_H in no_active_h.rejection_reasons

    missing_threshold = assess_qualification(
        replace(_evidence(scale="medium"), active_h_threshold_per_hour=None)
    )
    assert not missing_threshold.accepted
    assert QualificationReason.MISSING_ACTIVE_H_THRESHOLD in (
        missing_threshold.rejection_reasons
    )


def test_active_h_threshold_must_match_demand_scaled_service_tolerance() -> None:
    decision = assess_qualification(
        _evidence(active_h_threshold_per_hour=1e-6),
    )

    assert not decision.accepted
    assert QualificationReason.ACTIVE_H_THRESHOLD_MISMATCH in (
        decision.rejection_reasons
    )


def test_large_target_requires_active_h_in_legacy_evidence() -> None:
    decision = assess_qualification(
        replace(
            _evidence(scale="large", candidate_h=90, open_h=55),
            active_h=None,
        ),
        QualificationPolicy(large_scale_open_h_target=(50, 60)),
    )

    assert not decision.accepted
    assert QualificationReason.MISSING_ACTIVE_H_COUNT in decision.rejection_reasons

    missing_threshold = assess_qualification(
        replace(
            _evidence(scale="large", candidate_h=90, open_h=55, active_h=51),
            active_h_threshold_per_hour=None,
        ),
        QualificationPolicy(large_scale_open_h_target=(50, 60)),
    )
    assert not missing_threshold.accepted
    assert QualificationReason.MISSING_ACTIVE_H_THRESHOLD in (
        missing_threshold.rejection_reasons
    )


def test_nonfinite_evidence_is_serialisable_and_retains_reasons() -> None:
    decision = assess_qualification(
        replace(
            _evidence(),
            courier_only_objective=math.inf,
            witness_objective=math.nan,
        )
    )
    assert not decision.accepted
    assert decision.courier_only_objective is None
    assert decision.witness_objective is None
    assert QualificationReason.NONFINITE_COURIER_OBJECTIVE in (
        decision.rejection_reasons
    )
    assert QualificationReason.NONFINITE_WITNESS_OBJECTIVE in (
        decision.rejection_reasons
    )


def _formal_pool_and_decisions(
    *,
    reverse_input: bool = False,
) -> tuple[object, list[object]]:
    candidates: list[CandidateCase] = []
    decisions = []
    for scale in ("small", "medium", "large"):
        candidate_h = {"small": 23, "medium": 51, "large": 90}[scale]
        open_h = {"small": 8, "medium": 20, "large": 55}[scale]
        for block_index in range(6):
            for seed in (29, 11):
                case_id = f"{scale}-b{block_index}-s{seed}"
                candidate = CandidateCase(
                    case_id=case_id,
                    scale=scale,
                    demand_block_id=f"b{block_index}",
                    layout_seed=seed,
                    instance_ref=f"instances/{case_id}",
                    instance_sha256=hashlib.sha256(case_id.encode()).hexdigest(),
                    generator_parameters={
                        "map_half_size_km": {
                            "small": 4.0,
                            "medium": 6.0,
                            "large": 8.0,
                        }[scale]
                    },
                )
                candidates.append(candidate)
                decisions.append(
                    assess_qualification(
                        _evidence(
                            case_id,
                            scale=scale,
                            candidate_h=candidate_h,
                            open_h=open_h,
                            instance_sha256=candidate.instance_sha256,
                        ),
                        QualificationPolicy(large_scale_open_h_target=(50, 60)),
                    )
                )
    if reverse_input:
        candidates.reverse()
        decisions.reverse()
    pool = build_candidate_pool(
        candidates,
        candidate_seed_order=(11, 29),
        pool_label="locked-nanshan-v1",
    )
    return pool, decisions


def _freeze_metadata() -> FreezeMetadata:
    return FreezeMetadata(
        qualification_completed_at_utc="2026-08-26T04:00:00Z",
        manifest_frozen_at_utc="2026-08-26T04:05:00Z",
    )


def test_manifest_freezes_36_balanced_cases_in_candidate_order() -> None:
    pool, decisions = _formal_pool_and_decisions()
    manifest = freeze_manifest(
        pool,
        decisions,
        freeze_metadata=_freeze_metadata(),
    )

    assert len(manifest.cases) == 36
    assert {
        scale: sum(case.candidate.scale == scale for case in manifest.cases)
        for scale in ("small", "medium", "large")
    } == {
        "small": 12,
        "medium": 12,
        "large": 12,
    }
    first_two = [case.candidate.layout_seed for case in manifest.cases[:2]]
    assert first_two == [11, 29]
    assert manifest.artifact_payload()["manifest_sha256"] == manifest.sha256
    assert all(
        "active_h" in case["qualification"] for case in manifest.to_payload()["cases"]
    )
    assert all(
        case["qualification"]["active_h_threshold_per_hour"] == pytest.approx(0.1)
        for case in manifest.to_payload()["cases"]
    )


def test_manifest_hash_is_deterministic_under_input_permutation() -> None:
    pool_a, decisions_a = _formal_pool_and_decisions()
    pool_b, decisions_b = _formal_pool_and_decisions(reverse_input=True)

    manifest_a = freeze_manifest(
        pool_a,
        decisions_a,
        freeze_metadata=_freeze_metadata(),
    )
    manifest_b = freeze_manifest(
        pool_b,
        decisions_b,
        freeze_metadata=_freeze_metadata(),
    )
    assert pool_a.sha256 == pool_b.sha256
    assert manifest_a.sha256 == manifest_b.sha256


def test_manifest_rejects_padded_large_decision_even_if_marked_accepted() -> None:
    pool, decisions = _formal_pool_and_decisions()
    padded = [
        replace(decision, active_h=47)
        if decision.scale == "large" and decision.case_id.endswith("s11")
        else decision
        for decision in decisions
    ]

    with pytest.raises(ValueError, match="active-H target"):
        freeze_manifest(
            pool,
            padded,
            freeze_metadata=_freeze_metadata(),
        )


def test_manifest_rejects_unauditable_large_activity_threshold() -> None:
    pool, decisions = _formal_pool_and_decisions()
    tampered = [
        replace(decision, active_h_threshold_per_hour=1e-6)
        if decision.scale == "large" and decision.case_id.endswith("s11")
        else decision
        for decision in decisions
    ]

    with pytest.raises(ValueError, match="auditable active-H threshold"):
        freeze_manifest(
            pool,
            tampered,
            freeze_metadata=_freeze_metadata(),
        )


def test_freeze_rejects_underfilled_scale_and_comparison_started_metadata() -> None:
    pool, decisions = _formal_pool_and_decisions()
    rejected_large = [
        replace(
            decision,
            accepted=False,
            rejection_reasons=(QualificationReason.NO_OPEN_FACILITY,),
        )
        if decision.scale == "large" and decision.case_id.endswith("s29")
        else decision
        for decision in decisions
    ]
    with pytest.raises(ValueError, match="requires at least 36"):
        freeze_manifest(
            pool,
            rejected_large,
            freeze_metadata=_freeze_metadata(),
        )

    with pytest.raises(ValueError, match="before comparison starts"):
        FreezeMetadata(
            qualification_completed_at_utc="2026-08-26T04:00:00Z",
            manifest_frozen_at_utc="2026-08-26T04:05:00Z",
            comparison_started_at_utc="2026-08-26T04:06:00Z",
        )


def test_pool_rejects_duplicate_ids_unregistered_seeds_and_performance_fields() -> None:
    case = CandidateCase(
        case_id="small-b0-s11",
        scale="small",
        demand_block_id="b0",
        layout_seed=11,
        instance_ref="instances/small-b0-s11",
        instance_sha256="a" * 64,
    )
    with pytest.raises(ValueError, match="unique"):
        build_candidate_pool(
            (case, case),
            candidate_seed_order=(11,),
            pool_label="bad",
        )
    with pytest.raises(ValueError, match="unregistered seed"):
        build_candidate_pool(
            (case,),
            candidate_seed_order=(29,),
            pool_label="bad",
        )
    with pytest.raises(ValueError, match="performance-based"):
        CandidateCase(
            case_id="small-b0-s11",
            scale="small",
            demand_block_id="b0",
            layout_seed=11,
            instance_ref="instances/small-b0-s11",
            instance_sha256="a" * 64,
            generator_parameters={"pilot": {"bnc_gap": 0.1}},
        )


def _overcomplete_pool_and_prefix() -> tuple[object, list[object]]:
    candidates: list[CandidateCase] = []
    decisions = []
    seeds = (11, 29, 47, 83)
    for scale in ("small", "medium", "large"):
        candidate_h = {"small": 23, "medium": 51, "large": 90}[scale]
        open_h = {"small": 8, "medium": 20, "large": 55}[scale]
        for block_index in range(6):
            block_id = f"b{block_index}"
            stratum_id = f"{block_id}__{scale}"
            for rank, seed in enumerate(seeds):
                physical_id = f"{scale}-{block_id}-s{seed}"
                case_id = f"{physical_id}-replacement-{rank}"
                digest = hashlib.sha256(physical_id.encode()).hexdigest()
                candidate = CandidateCase(
                    case_id=case_id,
                    scale=scale,
                    demand_block_id=block_id,
                    layout_seed=seed,
                    instance_ref=f"instances/{physical_id}",
                    instance_sha256=digest,
                    generator_parameters={
                        "candidate_counts": {
                            "H": candidate_h,
                            "L": 2,
                            "E": 3,
                        },
                        "qualification_solve_profile": {
                            "profile_id": "demand-3x-feeder-035",
                            "demand_scale": 3.0,
                            "feeder_rider_cost_factor": 0.35,
                            "adds_forced_opening_constraint": False,
                        },
                        "pre_registered_selection": {
                            "selection_rule": OVERCOMPLETE_SELECTION_RULE,
                            "stratum_id": stratum_id,
                            "replacement_rank_within_stratum": rank,
                            "target_cases_per_stratum": 2,
                            "expected_stratum_count": 18,
                            "selection_uses_algorithm_performance": False,
                            "large_scale_open_h_qualification_target": (
                                [50, 60] if scale == "large" else None
                            ),
                            "forced_opening_constraints_added": False,
                        },
                    },
                )
                candidates.append(candidate)
                if rank <= 2:
                    accepted = assess_qualification(
                        _evidence(
                            case_id,
                            scale=scale,
                            candidate_h=candidate_h,
                            open_h=open_h,
                            instance_sha256=digest,
                        ),
                        QualificationPolicy(large_scale_open_h_target=(50, 60)),
                    )
                    decisions.append(
                        replace(
                            accepted,
                            accepted=False,
                            rejection_reasons=(QualificationReason.NO_OPEN_FACILITY,),
                        )
                        if rank == 0
                        else accepted
                    )
    return (
        build_candidate_pool(
            reversed(candidates),
            candidate_seed_order=seeds,
            pool_label="overcomplete-v2",
        ),
        list(reversed(decisions)),
    )


def _v5_overcomplete_pool_and_prefix() -> tuple[object, list[object]]:
    legacy_pool, legacy_decisions = _overcomplete_pool_and_prefix()
    identities: dict[str, tuple[str, str, bool]] = {}
    v5_case_ids: dict[str, str] = {}
    candidates: list[CandidateCase] = []
    profile = LOCKED_QUALIFICATION_PROFILES[0]
    for candidate in legacy_pool.candidates:
        v5_case_id = f"{candidate.case_id}__profile-{profile.profile_id}"
        v5_case_ids[candidate.case_id] = v5_case_id
        case_sha256 = qualification_case_sha256(
            candidate.instance_sha256,
            profile,
        )
        effective_model_config_sha256 = profile.effective_model_config_sha256()
        screening_is_exact = profile.expected_screening_is_exact
        identities[v5_case_id] = (
            case_sha256,
            effective_model_config_sha256,
            screening_is_exact,
        )
        parameters = dict(candidate.generator_parameters)
        parameters["qualification_solve_profile"] = profile.to_payload()
        parameters["qualification_model_identity"] = {
            "schema": QUALIFICATION_MODEL_IDENTITY_SCHEMA,
            "effective_model_config_sha256": effective_model_config_sha256,
            "qualification_case_sha256": case_sha256,
            "expected_screening_is_exact": screening_is_exact,
        }
        candidates.append(
            replace(
                candidate,
                case_id=v5_case_id,
                generator_parameters=parameters,
            )
        )

    decisions = []
    for decision in legacy_decisions:
        v5_case_id = v5_case_ids[decision.case_id]
        case_digest, config_digest, screening_is_exact = identities[v5_case_id]
        decisions.append(
            replace(
                decision,
                case_id=v5_case_id,
                qualification_run_id=f"qual-{v5_case_id}",
                qualification_case_sha256=case_digest,
                effective_model_config_sha256=config_digest,
                screening_is_exact=screening_is_exact,
            )
        )
    return (
        build_candidate_pool(
            candidates,
            candidate_seed_order=legacy_pool.candidate_seed_order,
            pool_label=FORMAL_QUALIFICATION_POOL_LABEL_V5,
        ),
        decisions,
    )


def _overcomplete_freeze_metadata() -> FreezeMetadata:
    return FreezeMetadata(
        qualification_completed_at_utc="2026-08-27T01:00:00Z",
        manifest_frozen_at_utc="2026-08-27T01:05:00Z",
        selection_rule=OVERCOMPLETE_SELECTION_RULE,
    )


def test_overcomplete_pool_selects_first_two_qualified_per_stratum() -> None:
    pool, decisions = _overcomplete_pool_and_prefix()
    manifest = freeze_manifest(
        pool,
        decisions,
        freeze_metadata=_overcomplete_freeze_metadata(),
    )

    assert len(manifest.cases) == 36
    assert manifest.replacement_selection_protocol is not None
    assert manifest.replacement_selection_protocol.target_cases_per_stratum == 2
    assert len(manifest.selection_ledger) == 72
    assert (
        sum(
            entry.qualification_status == "rejected"
            for entry in manifest.selection_ledger
        )
        == 18
    )
    assert (
        sum(
            entry.qualification_status == "not_attempted"
            for entry in manifest.selection_ledger
        )
        == 18
    )
    assert {
        entry.replacement_rank_within_stratum
        for entry in manifest.selection_ledger
        if entry.selected
    } == {1, 2}
    artifact = manifest.artifact_payload()
    assert len(artifact["rejection_ledger"]) == 18
    assert all(
        item["rejection_reasons"] == [QualificationReason.NO_OPEN_FACILITY]
        for item in artifact["rejection_ledger"]
    )
    assert (
        artifact["replacement_selection_protocol"][
            "selection_uses_algorithm_performance"
        ]
        is False
    )
    assert (
        artifact["replacement_selection_protocol"]["forced_opening_constraints_added"]
        is False
    )


def test_v5_manifest_persists_and_matches_actual_model_identity() -> None:
    pool, decisions = _v5_overcomplete_pool_and_prefix()

    manifest = freeze_manifest(
        pool,
        decisions,
        freeze_metadata=_overcomplete_freeze_metadata(),
    )

    artifact = manifest.artifact_payload()
    assert artifact["source_pool_label"] == FORMAL_QUALIFICATION_POOL_LABEL_V5
    assert all(
        decision["qualification_case_sha256"] is not None
        and decision["effective_model_config_sha256"] is not None
        and decision["screening_is_exact"] is True
        for decision in artifact["qualification_ledger"]
    )
    assert all(
        case["qualification"]["qualification_case_sha256"]
        == case["candidate"]["generator_parameters"]["qualification_model_identity"][
            "qualification_case_sha256"
        ]
        and case["qualification"]["effective_model_config_sha256"]
        == case["candidate"]["generator_parameters"]["qualification_model_identity"][
            "effective_model_config_sha256"
        ]
        and case["qualification"]["screening_is_exact"]
        is case["candidate"]["generator_parameters"]["qualification_model_identity"][
            "expected_screening_is_exact"
        ]
        for case in artifact["cases"]
    )


@pytest.mark.parametrize(
    ("field_name", "tampered_value"),
    [
        ("qualification_case_sha256", "d" * 64),
        ("effective_model_config_sha256", "e" * 64),
        ("screening_is_exact", False),
    ],
)
def test_v5_manifest_rejects_tampered_decision_model_identity(
    field_name: str,
    tampered_value: object,
) -> None:
    pool, decisions = _v5_overcomplete_pool_and_prefix()
    original = decisions[0]
    decisions[0] = replace(original, **{field_name: tampered_value})

    with pytest.raises(ValueError, match="qualification model identity mismatch"):
        freeze_manifest(
            pool,
            decisions,
            freeze_metadata=_overcomplete_freeze_metadata(),
        )


def test_v5_manifest_rejects_missing_candidate_or_decision_identity() -> None:
    pool, decisions = _v5_overcomplete_pool_and_prefix()
    original_candidate = pool.candidates[0]
    parameters = dict(original_candidate.generator_parameters)
    del parameters["qualification_model_identity"]
    missing_candidate_pool = build_candidate_pool(
        (
            replace(original_candidate, generator_parameters=parameters),
            *pool.candidates[1:],
        ),
        candidate_seed_order=pool.candidate_seed_order,
        pool_label=pool.pool_label,
    )

    with pytest.raises(ValueError, match="missing qualification_model_identity"):
        freeze_manifest(
            missing_candidate_pool,
            decisions,
            freeze_metadata=_overcomplete_freeze_metadata(),
        )

    original_decision = decisions[0]
    decisions[0] = replace(
        original_decision,
        qualification_case_sha256=None,
        effective_model_config_sha256=None,
        screening_is_exact=None,
    )
    with pytest.raises(ValueError, match="qualification model identity mismatch"):
        freeze_manifest(
            pool,
            decisions,
            freeze_metadata=_overcomplete_freeze_metadata(),
        )


def test_v5_manifest_rejects_non_strict_candidate_identity_object() -> None:
    pool, decisions = _v5_overcomplete_pool_and_prefix()
    original = pool.candidates[0]
    parameters = dict(original.generator_parameters)
    model_identity = dict(parameters["qualification_model_identity"])
    model_identity["unexpected_default"] = None
    parameters["qualification_model_identity"] = model_identity
    bad_pool = build_candidate_pool(
        (replace(original, generator_parameters=parameters), *pool.candidates[1:]),
        candidate_seed_order=pool.candidate_seed_order,
        pool_label=pool.pool_label,
    )

    with pytest.raises(ValueError, match="exactly the locked four fields"):
        freeze_manifest(
            bad_pool,
            decisions,
            freeze_metadata=_overcomplete_freeze_metadata(),
        )


def test_v5_manifest_recomputes_identity_and_rejects_profile_only_tampering() -> None:
    pool, decisions = _v5_overcomplete_pool_and_prefix()
    original = pool.candidates[0]
    parameters = dict(original.generator_parameters)
    profile = dict(parameters["qualification_solve_profile"])
    profile["locker_fixed_cost_per_hour"] = 24.0
    parameters["qualification_solve_profile"] = profile
    tampered_pool = build_candidate_pool(
        (replace(original, generator_parameters=parameters), *pool.candidates[1:]),
        candidate_seed_order=pool.candidate_seed_order,
        pool_label=pool.pool_label,
    )

    with pytest.raises(ValueError, match="locked qualification profile"):
        freeze_manifest(
            tampered_pool,
            decisions,
            freeze_metadata=_overcomplete_freeze_metadata(),
        )


def test_v5_manifest_rejects_consistently_rehashed_unlocked_profile() -> None:
    pool, decisions = _v5_overcomplete_pool_and_prefix()
    target_decision = decisions[0]
    target_index = next(
        index
        for index, candidate in enumerate(pool.candidates)
        if candidate.case_id == target_decision.case_id
    )
    original = pool.candidates[target_index]
    tampered_profile = LOCKED_QUALIFICATION_PROFILES[1]
    tampered_case_id = (
        original.case_id.rsplit("__profile-", maxsplit=1)[0]
        + f"__profile-{tampered_profile.profile_id}"
    )
    tampered_effective_sha256 = tampered_profile.effective_model_config_sha256()
    tampered_case_sha256 = qualification_case_sha256(
        original.instance_sha256,
        tampered_profile,
    )
    parameters = dict(original.generator_parameters)
    parameters["qualification_solve_profile"] = tampered_profile.to_payload()
    parameters["qualification_model_identity"] = {
        "schema": QUALIFICATION_MODEL_IDENTITY_SCHEMA,
        "effective_model_config_sha256": tampered_effective_sha256,
        "qualification_case_sha256": tampered_case_sha256,
        "expected_screening_is_exact": (tampered_profile.expected_screening_is_exact),
    }
    candidates = list(pool.candidates)
    candidates[target_index] = replace(
        original,
        case_id=tampered_case_id,
        generator_parameters=parameters,
    )
    tampered_pool = build_candidate_pool(
        candidates,
        candidate_seed_order=pool.candidate_seed_order,
        pool_label=pool.pool_label,
    )
    decisions[0] = replace(
        target_decision,
        case_id=tampered_case_id,
        qualification_run_id=f"qual-{tampered_case_id}",
        qualification_case_sha256=tampered_case_sha256,
        effective_model_config_sha256=tampered_effective_sha256,
        screening_is_exact=tampered_profile.expected_screening_is_exact,
    )

    with pytest.raises(ValueError, match="locked replacement rank"):
        freeze_manifest(
            tampered_pool,
            decisions,
            freeze_metadata=_overcomplete_freeze_metadata(),
        )


def test_overcomplete_freeze_requires_prefix_and_enough_qualified_replacements() -> (
    None
):
    pool, decisions = _overcomplete_pool_and_prefix()
    missing_middle = [
        decision
        for decision in decisions
        if not (
            decision.case_id.startswith("small-b0-")
            and decision.case_id.endswith("replacement-1")
        )
    ]
    with pytest.raises(ValueError, match="contiguous prefix"):
        freeze_manifest(
            pool,
            missing_middle,
            freeze_metadata=_overcomplete_freeze_metadata(),
        )

    only_one_accepted = [
        decision
        for decision in decisions
        if not (
            decision.case_id.startswith("small-b0-")
            and decision.case_id.endswith("replacement-2")
        )
    ]
    with pytest.raises(ValueError, match="only 1 qualified cases"):
        freeze_manifest(
            pool,
            only_one_accepted,
            freeze_metadata=_overcomplete_freeze_metadata(),
        )


def test_overcomplete_freeze_rejects_any_forced_opening_contract() -> None:
    pool, decisions = _overcomplete_pool_and_prefix()
    candidates = list(pool.candidates)
    original = candidates[0]
    parameters = dict(original.generator_parameters)
    selection = dict(parameters["pre_registered_selection"])
    selection["forced_opening_constraints_added"] = True
    parameters["pre_registered_selection"] = selection
    candidates[0] = replace(original, generator_parameters=parameters)
    bad_pool = build_candidate_pool(
        candidates,
        candidate_seed_order=pool.candidate_seed_order,
        pool_label=pool.pool_label,
    )

    with pytest.raises(ValueError, match="may not force facilities open"):
        freeze_manifest(
            bad_pool,
            decisions,
            freeze_metadata=_overcomplete_freeze_metadata(),
        )


def test_large_replacement_requires_actual_50_to_60_h_qualification_policy() -> None:
    pool, decisions = _overcomplete_pool_and_prefix()
    tampered = list(decisions)
    index = next(
        index
        for index, decision in enumerate(tampered)
        if decision.scale == "large" and decision.accepted
    )
    tampered[index] = replace(
        tampered[index],
        policy=QualificationPolicy(),
    )

    with pytest.raises(ValueError, match="locked 50--60 H target"):
        freeze_manifest(
            pool,
            tampered,
            freeze_metadata=_overcomplete_freeze_metadata(),
        )
