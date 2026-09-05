from __future__ import annotations

import math

import pytest

from src.benchmark.branch_cut.analysis import (
    EffectEstimate,
    PairAssessment,
    PairWinner,
    PreregisteredMetric,
    assess_pair,
    evaluate_success_gate,
    honest_termination_label,
    identify_systematic_regressions,
    shifted_geomean_gap_reduction,
    shifted_geomean_speedup,
    winner_counts,
)
from src.benchmark.branch_cut.statistics import (
    PairedCaseObservation,
    TrajectoryPoint,
    case_cluster_bootstrap_effect,
    cluster_paired_observations,
    dolan_more_profile_points,
    dolan_more_ratios,
    exact_sign_test,
    mean_candidate_minus_baseline,
    normalized_gap,
    paired_censored_time_input,
    paired_bootstrap_ci,
    primal_dual_integral,
    shifted_geometric_mean,
    step_integral,
    time_to_common_target,
    time_to_target,
)


def test_shifted_geometric_mean_handles_zero_without_epsilon() -> None:
    assert shifted_geometric_mean((0.0, 3.0), shift=1.0) == pytest.approx(1.0)
    assert shifted_geometric_mean((4.0, 4.0), shift=2.0) == pytest.approx(4.0)
    with pytest.raises(ValueError, match="at least one"):
        shifted_geometric_mean((), shift=1.0)
    with pytest.raises(ValueError, match="nonnegative"):
        shifted_geometric_mean((-0.1, 1.0), shift=1.0)


def test_normalized_gap_is_sign_robust_and_keeps_missing_explicit() -> None:
    assert normalized_gap(110.0, 100.0) == pytest.approx(10.0 / 110.0)
    assert normalized_gap(-90.0, -100.0) == pytest.approx(10.0 / 100.0)
    assert normalized_gap(1.0, -1.0) == pytest.approx(2.0)
    assert normalized_gap(None, 100.0) is None
    assert normalized_gap(float("inf"), 100.0) is None


def test_normalized_gap_rejects_materially_invalid_primal_dual_order() -> None:
    with pytest.raises(ValueError, match="ordering"):
        normalized_gap(90.0, 100.0)
    assert normalized_gap(100.0 - 1e-10, 100.0) == pytest.approx(0.0)
    assert normalized_gap(90.0, 100.0, sense="max") == pytest.approx(0.1)


def test_step_integral_uses_right_continuous_events_and_censors_late_points() -> None:
    assert step_integral(
        ((2.0, 3.0), (5.0, 1.0), (20.0, 99.0)),
        10.0,
        initial_value=2.0,
    ) == pytest.approx(18.0)


def test_primal_dual_integral_penalizes_no_incumbent_period() -> None:
    result = primal_dual_integral(
        (
            TrajectoryPoint(0.0, bound=80.0),
            TrajectoryPoint(2.0, incumbent=100.0),
            TrajectoryPoint(4.0, bound=90.0),
            TrajectoryPoint(6.0, incumbent=90.0),
            TrajectoryPoint(12.0, incumbent=1.0),
        ),
        10.0,
    )
    assert result.integral == pytest.approx(2.6)
    assert result.average_gap == pytest.approx(0.26)
    assert result.terminal_gap == pytest.approx(0.0)
    assert result.first_incumbent_time == pytest.approx(2.0)


def test_time_to_common_final_target_and_explicit_censoring() -> None:
    baseline = (
        TrajectoryPoint(1.0, incumbent=120.0),
        TrajectoryPoint(8.0, incumbent=100.0),
    )
    candidate = (
        TrajectoryPoint(2.0, incumbent=110.0),
        TrajectoryPoint(6.0, incumbent=90.0),
    )
    common = time_to_common_target(baseline, candidate, 10.0)
    assert common.target == pytest.approx(100.0)
    assert common.baseline_time == pytest.approx(8.0)
    assert common.candidate_time == pytest.approx(6.0)
    assert common.baseline_hit and common.candidate_hit

    strict = time_to_common_target(baseline, candidate, 10.0, target=95.0)
    assert strict.baseline_time is None
    assert strict.candidate_time == pytest.approx(6.0)
    assert time_to_target(candidate, 95.0, 5.0) is None

    no_common = time_to_common_target(baseline, (), 10.0)
    assert no_common.target is None
    assert not no_common.baseline_hit and not no_common.candidate_hit


def test_paired_bootstrap_is_deterministic_and_resamples_whole_pairs() -> None:
    pairs = ((10.0, 8.0), (6.0, 5.0), (9.0, 7.5), (4.0, 2.5))
    first = paired_bootstrap_ci(
        pairs,
        mean_candidate_minus_baseline,
        confidence=0.90,
        resamples=2_000,
        seed=20260826,
    )
    second = paired_bootstrap_ci(
        pairs,
        mean_candidate_minus_baseline,
        confidence=0.90,
        resamples=2_000,
        seed=20260826,
    )
    assert first == second
    assert first.estimate == pytest.approx(-1.5)
    assert first.lower <= first.estimate <= first.upper
    assert first.pair_count == 4
    with pytest.raises(ValueError):
        paired_bootstrap_ci(pairs, mean_candidate_minus_baseline, seed=None)  # type: ignore[arg-type]


def _case_observation(
    case_id: str,
    solver_seed: int,
    scale: str,
    baseline: float,
    candidate: float,
) -> PairedCaseObservation:
    return PairedCaseObservation(
        case_id=case_id,
        solver_seed=solver_seed,
        scale=scale,
        baseline=baseline,
        candidate=candidate,
    )


def test_case_clusters_are_canonical_and_keep_the_complete_solver_seed_set() -> None:
    observations = (
        _case_observation("case-b", 29, "Medium", 12.0, 10.0),
        _case_observation("case-a", 29, " small ", 8.0, 7.0),
        _case_observation("case-b", 11, "medium", 11.0, 9.0),
        _case_observation("case-a", 11, "small", 7.0, 6.0),
    )
    clusters = cluster_paired_observations(
        observations,
        expected_solver_seeds=(29, 11),
    )
    assert tuple(cluster.case_id for cluster in clusters) == ("case-a", "case-b")
    assert tuple(cluster.scale for cluster in clusters) == ("small", "medium")
    assert all(cluster.solver_seeds == (11, 29) for cluster in clusters)
    assert clusters[0].pairs == ((7.0, 6.0), (8.0, 7.0))


def test_case_cluster_bootstrap_is_deterministic_and_resamples_whole_cases() -> None:
    observations: list[PairedCaseObservation] = []
    for case_index, case_id in enumerate(("a", "b", "c"), start=1):
        for solver_seed in (1, 2):
            baseline = float(100 * case_index + solver_seed)
            observations.append(
                _case_observation(
                    case_id,
                    solver_seed,
                    "medium",
                    baseline,
                    baseline - case_index,
                )
            )

    def whole_case_statistic(sample: tuple[tuple[float, float], ...]) -> float:
        counts: dict[tuple[int, int], int] = {}
        for baseline, _candidate in sample:
            case_index = int(baseline) // 100
            solver_seed = int(baseline) % 100
            key = (case_index, solver_seed)
            counts[key] = counts.get(key, 0) + 1
        for case_index in (1, 2, 3):
            assert counts.get((case_index, 1), 0) == counts.get((case_index, 2), 0)
        return mean_candidate_minus_baseline(sample)

    first = case_cluster_bootstrap_effect(
        observations,
        whole_case_statistic,
        confidence=0.90,
        resamples=500,
        seed=20260827,
        expected_solver_seeds=(1, 2),
    )
    second = case_cluster_bootstrap_effect(
        reversed(observations),
        whole_case_statistic,
        confidence=0.90,
        resamples=500,
        seed=20260827,
        expected_solver_seeds=(1, 2),
    )
    assert first == second
    assert first.estimate == pytest.approx(-2.0)
    assert first.pair_count == 6
    assert first.case_count == 3
    assert first.scale_case_counts == (("medium", 3),)
    assert not first.stratified_by_scale


def test_scale_stratified_cluster_bootstrap_preserves_scale_case_counts() -> None:
    observations: list[PairedCaseObservation] = []
    case_specs = (
        ("s1", "small", 100),
        ("s2", "small", 110),
        ("l1", "large", 200),
        ("l2", "large", 210),
        ("l3", "large", 220),
    )
    for case_id, scale, code in case_specs:
        for solver_seed in (1, 2):
            observations.append(
                _case_observation(
                    case_id,
                    solver_seed,
                    scale,
                    float(code + solver_seed),
                    float(code + solver_seed - 1),
                )
            )

    def fixed_strata_statistic(sample: tuple[tuple[float, float], ...]) -> float:
        small_rows = sum(baseline < 200.0 for baseline, _candidate in sample)
        large_rows = sum(baseline >= 200.0 for baseline, _candidate in sample)
        assert (small_rows, large_rows) == (4, 6)
        return mean_candidate_minus_baseline(sample)

    result = case_cluster_bootstrap_effect(
        observations,
        fixed_strata_statistic,
        resamples=250,
        seed=7,
        stratify_by_scale=True,
        expected_solver_seeds=(1, 2),
    )
    assert result.estimate == pytest.approx(-1.0)
    assert result.stratified_by_scale
    assert result.scale_case_counts == (("large", 3), ("small", 2))


def test_cluster_bootstrap_accepts_shifted_geomean_effects_with_five_seeds() -> None:
    runtime_rows: list[PairedCaseObservation] = []
    gap_rows: list[PairedCaseObservation] = []
    for case_id, scale in (("s1", "small"), ("m1", "medium"), ("l1", "large")):
        for solver_seed in (11, 29, 47, 83, 101):
            runtime_rows.append(
                _case_observation(case_id, solver_seed, scale, 10.0, 5.0)
            )
            gap_rows.append(_case_observation(case_id, solver_seed, scale, 0.20, 0.10))
    expected_seeds = (11, 29, 47, 83, 101)
    speedup = case_cluster_bootstrap_effect(
        runtime_rows,
        shifted_geomean_speedup,
        resamples=100,
        seed=20260827,
        expected_solver_seeds=expected_seeds,
    )
    reduction = case_cluster_bootstrap_effect(
        gap_rows,
        shifted_geomean_gap_reduction,
        resamples=100,
        seed=20260827,
        expected_solver_seeds=expected_seeds,
    )
    assert speedup.estimate == pytest.approx(2.0)
    assert reduction.estimate == pytest.approx(0.5)
    assert speedup.pair_count == reduction.pair_count == 15
    assert speedup.case_count == reduction.case_count == 3


def test_case_cluster_validation_rejects_duplicate_unbalanced_and_invalid_rows() -> (
    None
):
    valid = (
        _case_observation("a", 1, "small", 2.0, 1.0),
        _case_observation("a", 2, "small", 3.0, 2.0),
        _case_observation("b", 1, "small", 4.0, 3.0),
        _case_observation("b", 2, "small", 5.0, 4.0),
    )
    with pytest.raises(ValueError, match="duplicate paired observation"):
        cluster_paired_observations(valid + (valid[0],))
    with pytest.raises(ValueError, match="unbalanced solver-seed set"):
        cluster_paired_observations(valid[:-1])
    with pytest.raises(ValueError, match="expected solver-seed set"):
        cluster_paired_observations(valid[:-1], expected_solver_seeds=(1, 2))
    with pytest.raises(ValueError, match="multiple scales"):
        cluster_paired_observations(
            valid[:1] + (_case_observation("a", 2, "large", 3.0, 2.0),) + valid[2:]
        )
    with pytest.raises(ValueError, match="finite"):
        cluster_paired_observations(
            valid[:-1] + (_case_observation("b", 2, "small", math.nan, 4.0),)
        )
    with pytest.raises(ValueError, match="at least 3 distinct cases"):
        cluster_paired_observations(valid, minimum_cases=3)


def test_cluster_bootstrap_rejects_undersized_strata_and_nonfinite_statistics() -> None:
    observations = (
        _case_observation("s1", 1, "small", 2.0, 1.0),
        _case_observation("s2", 1, "small", 3.0, 2.0),
        _case_observation("l1", 1, "large", 4.0, 3.0),
    )
    with pytest.raises(ValueError, match="at least 2 cases per scale"):
        case_cluster_bootstrap_effect(
            observations,
            mean_candidate_minus_baseline,
            resamples=10,
            seed=1,
            stratify_by_scale=True,
        )
    with pytest.raises(ValueError, match="must be finite"):
        case_cluster_bootstrap_effect(
            observations,
            lambda _sample: math.inf,
            resamples=10,
            seed=1,
        )


def test_paired_censored_time_input_retains_miss_flags_for_horizon_and_cap() -> None:
    restricted = paired_censored_time_input(None, 7.0, 10.0)
    assert restricted.pair == (10.0, 7.0)
    assert restricted.baseline_censored
    assert not restricted.candidate_censored
    assert restricted.strategy == "horizon"
    assert restricted.censor_value == pytest.approx(10.0)

    capped = paired_censored_time_input(4.0, None, 10.0, censor_cap=20.0)
    assert capped.pair == (4.0, 20.0)
    assert not capped.baseline_censored
    assert capped.candidate_censored
    assert capped.strategy == "fixed_cap"
    assert capped.censor_value == pytest.approx(20.0)


def test_censored_time_inputs_include_misses_in_case_cluster_effect() -> None:
    times = (
        paired_censored_time_input(None, 6.0, 10.0),
        paired_censored_time_input(8.0, None, 10.0),
        paired_censored_time_input(None, None, 10.0),
        paired_censored_time_input(4.0, 3.0, 10.0),
    )
    observations = tuple(
        PairedCaseObservation.from_censored_time_input(
            case_id=f"case-{index // 2}",
            solver_seed=index % 2,
            scale="small",
            censored_time_input=time_input,
        )
        for index, time_input in enumerate(times)
    )
    result = case_cluster_bootstrap_effect(
        observations,
        mean_candidate_minus_baseline,
        resamples=100,
        seed=11,
        expected_solver_seeds=(0, 1),
    )
    assert result.pair_count == 4
    assert result.estimate == pytest.approx(-0.75)
    assert result.censored_time_contract is not None
    assert result.censored_time_contract.horizon == pytest.approx(10.0)
    assert result.censored_time_contract.censor_value == pytest.approx(10.0)
    assert result.censored_time_contract.strategy == "horizon"
    assert result.baseline_censored_count == 2
    assert result.candidate_censored_count == 2
    assert result.either_censored_count == 3
    assert result.both_censored_count == 1
    clusters = cluster_paired_observations(
        observations,
        expected_solver_seeds=(0, 1),
    )
    assert all(cluster.censored_time_inputs is not None for cluster in clusters)

    reversed_result = case_cluster_bootstrap_effect(
        reversed(observations),
        mean_candidate_minus_baseline,
        resamples=100,
        seed=11,
        expected_solver_seeds=(0, 1),
    )
    assert reversed_result == result


def test_censored_cluster_rejects_mixed_contracts_and_partial_metadata() -> None:
    horizon_input = paired_censored_time_input(None, 6.0, 10.0)
    cap_input = paired_censored_time_input(None, 6.0, 10.0, censor_cap=20.0)
    audited_rows = (
        PairedCaseObservation.from_censored_time_input(
            case_id="a",
            solver_seed=1,
            scale="small",
            censored_time_input=horizon_input,
        ),
        PairedCaseObservation.from_censored_time_input(
            case_id="b",
            solver_seed=1,
            scale="small",
            censored_time_input=cap_input,
        ),
    )
    with pytest.raises(ValueError, match="mixed censor contracts"):
        cluster_paired_observations(audited_rows)

    fixed_cap_rows = tuple(
        PairedCaseObservation.from_censored_time_input(
            case_id=case_id,
            solver_seed=1,
            scale="small",
            censored_time_input=cap_input,
        )
        for case_id in ("a", "b")
    )
    fixed_cap_result = case_cluster_bootstrap_effect(
        fixed_cap_rows,
        mean_candidate_minus_baseline,
        resamples=10,
        seed=1,
    )
    assert fixed_cap_result.censored_time_contract is not None
    assert fixed_cap_result.censored_time_contract.strategy == "fixed_cap"
    assert fixed_cap_result.censored_time_contract.horizon == pytest.approx(10.0)
    assert fixed_cap_result.censored_time_contract.censor_value == pytest.approx(20.0)
    assert fixed_cap_result.censored_time_contracts_by_scale == (
        ("small", fixed_cap_result.censored_time_contract),
    )
    assert fixed_cap_result.baseline_censored_count == 2
    assert fixed_cap_result.candidate_censored_count == 0
    assert fixed_cap_result.either_censored_count == 2
    assert fixed_cap_result.both_censored_count == 0

    large_input = paired_censored_time_input(None, 12.0, 20.0)
    scale_specific_rows = (
        *tuple(
            PairedCaseObservation.from_censored_time_input(
                case_id=case_id,
                solver_seed=1,
                scale="small",
                censored_time_input=horizon_input,
            )
            for case_id in ("small-a", "small-b")
        ),
        *tuple(
            PairedCaseObservation.from_censored_time_input(
                case_id=case_id,
                solver_seed=1,
                scale="large",
                censored_time_input=large_input,
            )
            for case_id in ("large-a", "large-b")
        ),
    )
    scale_specific = case_cluster_bootstrap_effect(
        scale_specific_rows,
        mean_candidate_minus_baseline,
        resamples=10,
        seed=1,
        stratify_by_scale=True,
    )
    assert scale_specific.censored_time_contract is None
    assert [
        scale for scale, _contract in scale_specific.censored_time_contracts_by_scale
    ] == [
        "large",
        "small",
    ]
    assert [
        contract.horizon
        for _scale, contract in scale_specific.censored_time_contracts_by_scale
    ] == [20.0, 10.0]

    partial_rows = (
        audited_rows[0],
        _case_observation("b", 1, "small", 10.0, 6.0),
    )
    with pytest.raises(ValueError, match="partial censored-time metadata"):
        cluster_paired_observations(partial_rows)

    mismatched_values = (
        PairedCaseObservation(
            case_id="a",
            solver_seed=1,
            scale="small",
            baseline=9.0,
            candidate=6.0,
            censored_time_input=horizon_input,
        ),
        PairedCaseObservation.from_censored_time_input(
            case_id="b",
            solver_seed=1,
            scale="small",
            censored_time_input=horizon_input,
        ),
    )
    with pytest.raises(ValueError, match="do not match"):
        cluster_paired_observations(mismatched_values)


def test_minimum_cases_per_scale_is_ignored_without_stratification() -> None:
    observations = (
        _case_observation("a", 1, "small", 2.0, 1.0),
        _case_observation("b", 1, "small", 3.0, 2.0),
    )
    result = case_cluster_bootstrap_effect(
        observations,
        mean_candidate_minus_baseline,
        resamples=10,
        seed=1,
        stratify_by_scale=False,
        minimum_cases_per_scale=1,
    )
    assert result.estimate == pytest.approx(-1.0)
    with pytest.raises(ValueError, match="at least two"):
        case_cluster_bootstrap_effect(
            observations,
            mean_candidate_minus_baseline,
            resamples=10,
            seed=1,
            stratify_by_scale=True,
            minimum_cases_per_scale=1,
        )


@pytest.mark.parametrize(
    ("baseline_time", "candidate_time", "horizon", "censor_cap", "message"),
    (
        (1.0, 2.0, 0.0, None, "strictly positive"),
        (11.0, 2.0, 10.0, None, "exceeds"),
        (1.0, math.inf, 10.0, None, "finite"),
        (1.0, 2.0, 10.0, 9.0, "at least"),
    ),
)
def test_paired_censored_time_input_rejects_invalid_preregistered_inputs(
    baseline_time: float | None,
    candidate_time: float | None,
    horizon: float,
    censor_cap: float | None,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        paired_censored_time_input(
            baseline_time,
            candidate_time,
            horizon,
            censor_cap=censor_cap,
        )


def test_direct_censored_time_input_validation_rejects_boolean_contract() -> None:
    valid = paired_censored_time_input(None, 2.0, 10.0)
    malformed = type(valid)(
        baseline_value=valid.baseline_value,
        candidate_value=valid.candidate_value,
        baseline_censored=valid.baseline_censored,
        candidate_censored=valid.candidate_censored,
        horizon=True,
        censor_value=valid.censor_value,
        strategy=valid.strategy,
    )
    with pytest.raises(ValueError, match="censor horizon"):
        malformed.validate()


def test_exact_sign_test_reports_ties_and_exact_binomial_probability() -> None:
    two_sided = exact_sign_test((1.0, 2.0, 3.0, -1.0, 0.0))
    assert (two_sided.positive, two_sided.negative, two_sided.ties) == (3, 1, 1)
    assert two_sided.effective_pairs == 4
    assert two_sided.p_value == pytest.approx(0.625)
    assert exact_sign_test(
        (1.0, 2.0, 3.0, -1.0), alternative="greater"
    ).p_value == pytest.approx(5.0 / 16.0)
    assert exact_sign_test((0.0, 0.0)).p_value == pytest.approx(1.0)


def test_dolan_more_profiles_keep_failures_and_disclose_all_failed_problems() -> None:
    ratios = dolan_more_ratios(
        {
            "p1": {"native": 1.0, "bnc": 2.0, "warm": None},
            "p2": {"native": 4.0, "bnc": 2.0, "warm": 8.0},
            "p3": {"native": None, "bnc": None, "warm": None},
        }
    )
    assert ratios.included_problems == ("p1", "p2")
    assert ratios.excluded_all_failed == ("p3",)
    assert ratios.ratios["p1"]["native"] == pytest.approx(1.0)
    assert ratios.ratios["p2"]["native"] == pytest.approx(2.0)
    assert math.isinf(ratios.ratios["p1"]["warm"])
    assert ratios.failed_counts["warm"] == 1

    profile = dolan_more_profile_points(ratios, taus=(1.0, 2.0, 4.0))
    assert profile.fractions["native"] == pytest.approx((0.5, 1.0, 1.0))
    assert profile.fractions["bnc"] == pytest.approx((0.5, 1.0, 1.0))
    assert profile.fractions["warm"] == pytest.approx((0.0, 0.0, 0.5))

    finite_failure = dolan_more_ratios(
        {"p1": {"a": 1.0, "b": None}, "p2": {"a": 4.0, "b": 1.0}},
        failure_ratio=5.0,
    )
    assert finite_failure.ratios["p1"]["b"] == pytest.approx(5.0)
    with pytest.raises(ValueError, match="cover"):
        dolan_more_ratios(
            {"p1": {"a": 1.0, "b": None}, "p2": {"a": 4.0, "b": 1.0}},
            failure_ratio=2.0,
        )


def test_time_limit_status_is_never_labeled_optimal() -> None:
    assert honest_termination_label("TIME_LIMIT", has_incumbent=True) == (
        "time_limit_with_incumbent"
    )
    assert honest_termination_label("TIME_LIMIT", has_incumbent=False) == (
        "time_limit_without_incumbent"
    )
    assert honest_termination_label("OPTIMAL", has_incumbent=True) == "proven_optimal"


def _assessment(
    instance: str,
    tier: str,
    winner: PairWinner,
    seed: int = 1,
) -> PairAssessment:
    values = {
        PairWinner.CANDIDATE: (10.0, 8.0),
        PairWinner.BASELINE: (8.0, 10.0),
        PairWinner.TIE: (10.0, 10.0),
        PairWinner.UNRESOLVED: (None, None),
    }[winner]
    return assess_pair(
        instance_id=instance,
        seed=seed,
        scale_tier=tier,
        baseline_value=values[0],
        candidate_value=values[1],
        metric=PreregisteredMetric("primary_gap", direction="lower"),
        baseline_status="TIME_LIMIT",
        candidate_status="TIME_LIMIT",
    )


def test_pair_winner_respects_direction_tolerance_and_missing_failures() -> None:
    lower = PreregisteredMetric("gap", direction="lower", absolute_tolerance=0.01)
    candidate = assess_pair(
        instance_id="i1",
        seed=1,
        scale_tier="medium",
        baseline_value=0.20,
        candidate_value=0.15,
        metric=lower,
        baseline_status="TIME_LIMIT",
        candidate_status="TIME_LIMIT",
    )
    tie = assess_pair(
        instance_id="i2",
        seed=1,
        scale_tier="medium",
        baseline_value=0.20,
        candidate_value=0.205,
        metric=lower,
        baseline_status="TIME_LIMIT",
        candidate_status="TIME_LIMIT",
    )
    missing = assess_pair(
        instance_id="i3",
        seed=1,
        scale_tier="large",
        baseline_value=0.20,
        candidate_value=None,
        metric=lower,
        baseline_status="TIME_LIMIT",
        candidate_status="TIME_LIMIT",
    )
    assert candidate.winner == PairWinner.CANDIDATE
    assert tie.winner == PairWinner.TIE
    assert missing.winner == PairWinner.BASELINE
    assert winner_counts((candidate, tie, missing)) == {
        "candidate": 1,
        "baseline": 1,
        "tie": 1,
        "unresolved": 0,
    }


def test_shifted_aggregate_effect_helpers() -> None:
    assert shifted_geomean_speedup(((10.0, 5.0), (10.0, 5.0))) == pytest.approx(2.0)
    assert shifted_geomean_gap_reduction(((0.20, 0.10), (0.20, 0.10))) == pytest.approx(
        0.5
    )


def test_success_gate_requires_broad_wins_effect_ci_and_no_regression() -> None:
    pairs = (
        _assessment("s1", "small", PairWinner.CANDIDATE),
        _assessment("m1", "medium", PairWinner.CANDIDATE),
        _assessment("m2", "medium", PairWinner.CANDIDATE),
        _assessment("m3", "medium", PairWinner.BASELINE),
        _assessment("l1", "large", PairWinner.CANDIDATE),
        _assessment("l2", "large", PairWinner.CANDIDATE),
    )
    result = evaluate_success_gate(
        pairs,
        speedup=EffectEstimate(
            estimate=1.60, ci_lower=1.10, ci_upper=2.10, pair_count=6
        ),
        gap_reduction=None,
    )
    assert result.passed
    assert result.medium_large_candidate_wins == 4
    assert result.medium_large_pair_count == 5
    assert result.medium_large_win_rate == pytest.approx(0.8)
    assert result.improved_scale_tiers == ("large", "medium", "small")
    assert result.speed_gate_passed and not result.gap_gate_passed

    weak_ci = evaluate_success_gate(
        pairs,
        speedup=EffectEstimate(
            estimate=1.60, ci_lower=0.98, ci_upper=2.10, pair_count=6
        ),
        gap_reduction=None,
    )
    assert not weak_ci.passed
    assert not weak_ci.conditions["substantive_aggregate_improvement"]

    regression = evaluate_success_gate(
        pairs,
        speedup=None,
        gap_reduction=EffectEstimate(
            estimate=0.25, ci_lower=0.04, ci_upper=0.40, pair_count=6
        ),
        systematic_regressions=("time_to_first_incumbent",),
    )
    assert not regression.passed
    assert regression.gap_gate_passed
    assert "time_to_first_incumbent" in regression.reasons[-1]


def test_systematic_regression_uses_all_pairs_in_denominator() -> None:
    key_pairs = (
        _assessment("i1", "medium", PairWinner.BASELINE),
        _assessment("i2", "medium", PairWinner.BASELINE),
        _assessment("i3", "medium", PairWinner.BASELINE),
        _assessment("i4", "medium", PairWinner.TIE),
        _assessment("i5", "medium", PairWinner.UNRESOLVED),
    )
    assert identify_systematic_regressions(
        {"nodes": key_pairs},
        regression_fraction=0.60,
        minimum_pairs=5,
    ) == ("nodes",)
