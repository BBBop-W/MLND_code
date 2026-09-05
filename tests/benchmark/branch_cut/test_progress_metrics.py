from __future__ import annotations

import pytest

from src.benchmark.branch_cut.analysis import summarize_minimization_progress
from src.benchmark.branch_cut.statistics import (
    primal_dual_integral_from_progress_events,
    time_to_common_target_from_progress_events,
    time_to_target_from_progress_events,
    trajectory_from_progress_events,
)


def test_progress_metrics_charge_preparation_and_keep_time_limit_honest() -> None:
    events = (
        {
            "event": "mip_progress",
            "runtime_seconds": 1.0,
            "incumbent": 1e100,
            "bound": 80.0,
        },
        {
            "event": "incumbent",
            "runtime_seconds": 3.0,
            "incumbent": 120.0,
            "bound": 80.0,
        },
        {
            "event": "mip_progress",
            "runtime_seconds": 3.0,
            "incumbent": 100.0,
            "bound": 85.0,
        },
        {
            "event": "final",
            "runtime_seconds": 6.0,
            "total_algorithm_runtime_seconds": 8.0,
            "incumbent": 100.0,
            "bound": 90.0,
            "status": 9,
        },
    )

    metrics = summarize_minimization_progress(
        events,
        target=105.0,
        horizon_seconds=10.0,
        preparation_time_seconds=2.0,
    )

    assert metrics.time_to_target_seconds == pytest.approx(5.0)
    assert not metrics.time_to_target_censored
    assert metrics.first_incumbent_time_seconds == pytest.approx(5.0)
    assert metrics.primal_dual_integral == pytest.approx(5.65)
    assert metrics.time_normalized_primal_dual_integral == pytest.approx(0.565)
    assert metrics.terminal_gap == pytest.approx(0.1)
    assert metrics.final_incumbent == pytest.approx(100.0)
    assert metrics.final_bound == pytest.approx(90.0)
    assert metrics.status_name == "TIME_LIMIT"
    assert metrics.termination_label == "time_limit_with_incumbent"
    assert metrics.time_limited
    assert not metrics.proven_optimal


def test_no_incumbent_is_censored_and_penalized_for_the_whole_horizon() -> None:
    events = (
        {
            "event": "mip_progress",
            "runtime_seconds": 0.0,
            "incumbent": float("inf"),
            "bound": -1e100,
        },
        {
            "event": "final",
            "runtime_seconds": 4.0,
            "incumbent": None,
            "bound": 50.0,
            "status": "TIME_LIMIT",
        },
    )

    metrics = summarize_minimization_progress(
        events,
        target=60.0,
        horizon_seconds=6.0,
        preparation_time_seconds=1.0,
        missing_gap_penalty=0.75,
    )

    assert metrics.time_to_target_seconds is None
    assert metrics.time_to_target_censored
    assert metrics.primal_dual_integral == pytest.approx(4.5)
    assert metrics.time_normalized_primal_dual_integral == pytest.approx(0.75)
    assert metrics.terminal_gap is None
    assert metrics.final_incumbent is None
    assert metrics.final_bound == pytest.approx(50.0)
    assert metrics.termination_label == "time_limit_without_incumbent"
    assert not metrics.proven_optimal


def test_event_after_horizon_cannot_supply_target_or_optimal_status() -> None:
    events = (
        {
            "event": "final",
            "runtime_seconds": 8.0,
            "incumbent": 90.0,
            "bound": 90.0,
            "status": "OPTIMAL",
        },
    )

    metrics = summarize_minimization_progress(
        events,
        target=95.0,
        horizon_seconds=10.0,
        preparation_time_seconds=3.0,
    )

    assert metrics.time_to_target_seconds is None
    assert metrics.time_to_target_censored
    assert metrics.final_incumbent is None
    assert metrics.status_name == "UNKNOWN"
    assert metrics.termination_label == "terminated_without_incumbent"
    assert not metrics.proven_optimal


def test_raw_progress_helpers_use_stable_same_time_order_and_total_clock() -> None:
    events = (
        {"runtime_seconds": 2.0, "incumbent": 120.0, "bound": 80.0},
        {"runtime_seconds": 2.0, "incumbent": 100.0, "bound": 90.0},
    )
    trajectory = trajectory_from_progress_events(
        events,
        preparation_time_seconds=1.5,
    )

    assert tuple(point.time for point in trajectory.points) == pytest.approx((3.5, 3.5))
    assert time_to_target_from_progress_events(
        events,
        105.0,
        5.0,
        preparation_time_seconds=1.5,
    ) == pytest.approx(3.5)
    integral = primal_dual_integral_from_progress_events(
        events,
        5.0,
        preparation_time_seconds=1.5,
    )
    assert integral.integral == pytest.approx(3.65)
    assert integral.average_gap == pytest.approx(0.73)


def test_paired_common_target_is_worse_final_incumbent_and_retains_missingness() -> (
    None
):
    baseline = (
        {"runtime_seconds": 1.0, "incumbent": 120.0, "bound": 70.0},
        {"runtime_seconds": 8.0, "incumbent": 100.0, "bound": 90.0},
    )
    candidate = (
        {"runtime_seconds": 2.0, "incumbent": 110.0, "bound": 80.0},
        {"runtime_seconds": 4.0, "incumbent": 100.0, "bound": 85.0},
        {"runtime_seconds": 7.0, "incumbent": 90.0, "bound": 88.0},
    )

    common = time_to_common_target_from_progress_events(
        baseline,
        candidate,
        10.0,
        baseline_preparation_time_seconds=2.0,
        candidate_preparation_time_seconds=1.0,
    )
    assert common.target == pytest.approx(100.0)
    assert common.baseline_final_incumbent == pytest.approx(100.0)
    assert common.candidate_final_incumbent == pytest.approx(90.0)
    assert common.baseline_time == pytest.approx(10.0)
    assert common.candidate_time == pytest.approx(5.0)
    assert common.target_available
    assert not common.baseline_incumbent_missing
    assert not common.candidate_incumbent_missing

    missing_candidate = time_to_common_target_from_progress_events(
        baseline,
        ({"runtime_seconds": 7.0, "incumbent": 1e100, "bound": 88.0},),
        10.0,
        baseline_preparation_time_seconds=2.0,
        candidate_preparation_time_seconds=1.0,
    )
    assert missing_candidate.target is None
    assert missing_candidate.baseline_final_incumbent == pytest.approx(100.0)
    assert missing_candidate.candidate_final_incumbent is None
    assert missing_candidate.baseline_time is None
    assert missing_candidate.candidate_time is None
    assert not missing_candidate.target_available
    assert not missing_candidate.baseline_incumbent_missing
    assert missing_candidate.candidate_incumbent_missing
