"""Paper-facing paired analysis and preregistered success-gate evaluation."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum

from src.benchmark.branch_cut.statistics import (
    Pair,
    primal_dual_integral,
    shifted_geometric_mean,
    terminal_best_bound,
    terminal_best_incumbent,
    time_to_target,
    trajectory_from_progress_events,
)


class PairWinner(str, Enum):
    """Outcome of one paired comparison."""

    CANDIDATE = "candidate"
    BASELINE = "baseline"
    TIE = "tie"
    UNRESOLVED = "unresolved"


@dataclass(frozen=True)
class PreregisteredMetric:
    """Direction and tie tolerance fixed before examining test outcomes."""

    name: str
    direction: str = "lower"
    absolute_tolerance: float = 0.0
    relative_tolerance: float = 0.0
    missing_is_failure: bool = True

    def validate(self) -> None:
        if not self.name.strip():
            raise ValueError("metric name cannot be empty")
        if self.direction not in {"lower", "higher"}:
            raise ValueError("metric direction must be 'lower' or 'higher'")
        for name, value in (
            ("absolute_tolerance", self.absolute_tolerance),
            ("relative_tolerance", self.relative_tolerance),
        ):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")


def honest_termination_label(status: str, *, has_incumbent: bool) -> str:
    """Return a status label that never upgrades a time-limited incumbent.

    Only an explicit ``OPTIMAL`` solver status is called proven optimal.  A
    time-limited run remains a time-limited run even when its reported gap is
    small or zero to display precision.
    """

    normalized = status.strip().upper()
    if not normalized:
        raise ValueError("status cannot be empty")
    if normalized == "OPTIMAL":
        return (
            "proven_optimal" if has_incumbent else "invalid_optimal_without_incumbent"
        )
    if normalized == "TIME_LIMIT":
        return (
            "time_limit_with_incumbent"
            if has_incumbent
            else "time_limit_without_incumbent"
        )
    if normalized == "INFEASIBLE":
        return (
            "proven_infeasible"
            if not has_incumbent
            else "invalid_infeasible_with_incumbent"
        )
    if normalized in {"UNBOUNDED", "INF_OR_UNBD"}:
        return normalized.lower()
    return (
        "terminated_with_incumbent" if has_incumbent else "terminated_without_incumbent"
    )


@dataclass(frozen=True)
class MinimizationProgressMetrics:
    """Paper-facing progress metrics on one matched total-runtime horizon."""

    target: float
    time_to_target_seconds: float | None
    time_to_target_censored: bool
    primal_dual_integral: float
    time_normalized_primal_dual_integral: float
    terminal_gap: float | None
    first_incumbent_time_seconds: float | None
    final_incumbent: float | None
    final_bound: float | None
    has_incumbent: bool
    status_name: str
    termination_label: str
    proven_optimal: bool
    time_limited: bool
    horizon_seconds: float
    preparation_time_seconds: float


def summarize_minimization_progress(
    events: Iterable[Mapping[str, object]],
    *,
    target: float,
    horizon_seconds: float,
    preparation_time_seconds: float = 0.0,
    missing_gap_penalty: float = 1.0,
    objective_scale: float = 1.0,
    absolute_target_tolerance: float = 1e-9,
    relative_target_tolerance: float = 1e-9,
) -> MinimizationProgressMetrics:
    """Compute honest time-to-target and normalized primal-dual integral.

    Callback runtimes are shifted by charged algorithm-preparation time before
    either metric is evaluated.  A target not reached by ``horizon_seconds`` is
    explicitly right-censored.  Only a finite incumbent inside the horizon
    counts as an incumbent, and only an ``OPTIMAL`` status observed no later
    than the horizon is labeled proven optimal; ``TIME_LIMIT`` remains
    time-limited even if its displayed terminal gap is zero.
    """

    trajectory = trajectory_from_progress_events(
        events,
        preparation_time_seconds=preparation_time_seconds,
    )
    integral = primal_dual_integral(
        trajectory.points,
        horizon_seconds,
        sense="min",
        missing_gap_penalty=missing_gap_penalty,
        objective_scale=objective_scale,
    )
    target_time = time_to_target(
        trajectory.points,
        target,
        horizon_seconds,
        sense="min",
        absolute_tolerance=absolute_target_tolerance,
        relative_tolerance=relative_target_tolerance,
    )
    incumbent = terminal_best_incumbent(
        trajectory.points,
        horizon_seconds,
        sense="min",
    )
    bound = terminal_best_bound(
        trajectory.points,
        horizon_seconds,
        sense="min",
    )
    status_within_horizon = bool(
        trajectory.status_time is None
        or trajectory.status_time <= float(horizon_seconds)
    )
    status_name = trajectory.status_name if status_within_horizon else "UNKNOWN"
    has_incumbent = incumbent is not None
    termination = honest_termination_label(
        status_name,
        has_incumbent=has_incumbent,
    )
    return MinimizationProgressMetrics(
        target=float(target),
        time_to_target_seconds=target_time,
        time_to_target_censored=target_time is None,
        primal_dual_integral=integral.integral,
        time_normalized_primal_dual_integral=integral.average_gap,
        terminal_gap=integral.terminal_gap,
        first_incumbent_time_seconds=integral.first_incumbent_time,
        final_incumbent=incumbent,
        final_bound=bound,
        has_incumbent=has_incumbent,
        status_name=status_name,
        termination_label=termination,
        proven_optimal=termination == "proven_optimal",
        time_limited=status_name == "TIME_LIMIT",
        horizon_seconds=integral.horizon,
        preparation_time_seconds=trajectory.preparation_time,
    )


@dataclass(frozen=True)
class PairAssessment:
    """One same-instance, same-seed comparison under one primary metric."""

    instance_id: str
    seed: int
    scale_tier: str
    metric_name: str
    baseline_value: float | None
    candidate_value: float | None
    winner: PairWinner
    baseline_status: str
    candidate_status: str


def assess_pair(
    *,
    instance_id: str,
    seed: int,
    scale_tier: str,
    baseline_value: float | None,
    candidate_value: float | None,
    metric: PreregisteredMetric,
    baseline_status: str,
    candidate_status: str,
) -> PairAssessment:
    """Choose a pair winner without changing the preregistered metric post hoc.

    Missing values normally lose to finite values, so a no-incumbent or
    no-target-hit run cannot disappear from the denominator.  If both values are
    missing the pair is unresolved.  ``missing_is_failure=False`` keeps any pair
    with missing data unresolved instead.
    """

    if not instance_id.strip() or not scale_tier.strip():
        raise ValueError("instance_id and scale_tier cannot be empty")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    metric.validate()

    def finite_or_none(value: float | None, name: str) -> float | None:
        if value is None:
            return None
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"{name} must be finite or None")
        return number

    baseline = finite_or_none(baseline_value, "baseline_value")
    candidate = finite_or_none(candidate_value, "candidate_value")
    if baseline is None and candidate is None:
        winner = PairWinner.UNRESOLVED
    elif baseline is None or candidate is None:
        if not metric.missing_is_failure:
            winner = PairWinner.UNRESOLVED
        elif baseline is None:
            winner = PairWinner.CANDIDATE
        else:
            winner = PairWinner.BASELINE
    else:
        tolerance = metric.absolute_tolerance + metric.relative_tolerance * max(
            1.0, abs(baseline), abs(candidate)
        )
        difference = candidate - baseline
        if abs(difference) <= tolerance:
            winner = PairWinner.TIE
        elif metric.direction == "lower":
            winner = PairWinner.CANDIDATE if difference < 0.0 else PairWinner.BASELINE
        else:
            winner = PairWinner.CANDIDATE if difference > 0.0 else PairWinner.BASELINE
    return PairAssessment(
        instance_id=instance_id,
        seed=seed,
        scale_tier=scale_tier.lower(),
        metric_name=metric.name,
        baseline_value=baseline,
        candidate_value=candidate,
        winner=winner,
        baseline_status=baseline_status,
        candidate_status=candidate_status,
    )


def shifted_geomean_speedup(pairs: Sequence[Pair], *, shift: float = 1.0) -> float:
    """Return baseline/candidate shifted-geometric-mean speedup."""

    if not pairs:
        raise ValueError("speedup requires at least one complete pair")
    baseline = shifted_geometric_mean((pair[0] for pair in pairs), shift)
    candidate = shifted_geometric_mean((pair[1] for pair in pairs), shift)
    if candidate == 0.0:
        return math.inf if baseline > 0.0 else 1.0
    return baseline / candidate


def shifted_geomean_gap_reduction(
    pairs: Sequence[Pair], *, shift: float = 1e-6
) -> float:
    """Return fractional reduction in shifted-geometric-mean final gap."""

    if not pairs:
        raise ValueError("gap reduction requires at least one complete pair")
    baseline = shifted_geometric_mean((pair[0] for pair in pairs), shift)
    candidate = shifted_geometric_mean((pair[1] for pair in pairs), shift)
    if baseline == 0.0:
        if candidate == 0.0:
            return 0.0
        raise ValueError(
            "gap reduction is undefined when baseline aggregate gap is zero"
        )
    return 1.0 - candidate / baseline


@dataclass(frozen=True)
class EffectEstimate:
    """Point estimate and paired confidence interval on one declared scale."""

    estimate: float
    ci_lower: float
    ci_upper: float
    pair_count: int

    def validate(self) -> None:
        if any(
            not math.isfinite(value)
            for value in (self.estimate, self.ci_lower, self.ci_upper)
        ):
            raise ValueError("effect estimate and confidence limits must be finite")
        if self.ci_lower > self.ci_upper:
            raise ValueError("confidence interval lower limit exceeds upper limit")
        if (
            isinstance(self.pair_count, bool)
            or not isinstance(self.pair_count, int)
            or self.pair_count < 1
        ):
            raise ValueError("pair_count must be a positive integer")


def identify_systematic_regressions(
    key_metrics: Mapping[str, Sequence[PairAssessment]],
    *,
    regression_fraction: float = 0.60,
    minimum_pairs: int = 5,
) -> tuple[str, ...]:
    """Flag key metrics for which baseline wins form a preregistered majority.

    Ties and unresolved pairs remain in the denominator.  This conservative
    convention prevents failed candidate runs from being dropped.  The caller
    should preregister the key metrics and thresholds before locked-test runs.
    """

    if not math.isfinite(regression_fraction) or not 0.0 < regression_fraction <= 1.0:
        raise ValueError("regression_fraction must lie in (0, 1]")
    if (
        isinstance(minimum_pairs, bool)
        or not isinstance(minimum_pairs, int)
        or minimum_pairs < 1
    ):
        raise ValueError("minimum_pairs must be a positive integer")
    regressions: list[str] = []
    for name, pairs in sorted(key_metrics.items()):
        if len(pairs) < minimum_pairs:
            continue
        baseline_wins = sum(pair.winner == PairWinner.BASELINE for pair in pairs)
        if baseline_wins / len(pairs) >= regression_fraction:
            regressions.append(name)
    return tuple(regressions)


@dataclass(frozen=True)
class SuccessGateResult:
    """Auditable decision and each independently checkable gate condition."""

    passed: bool
    conditions: Mapping[str, bool]
    reasons: tuple[str, ...]
    medium_large_win_rate: float
    medium_large_candidate_wins: int
    medium_large_pair_count: int
    improved_scale_tiers: tuple[str, ...]
    improved_instance_count: int
    speed_gate_passed: bool
    gap_gate_passed: bool


def evaluate_success_gate(
    primary_pairs: Sequence[PairAssessment],
    *,
    speedup: EffectEstimate | None,
    gap_reduction: EffectEstimate | None,
    systematic_regressions: Iterable[str] = (),
    medium_large_tiers: Iterable[str] = ("medium", "large"),
    minimum_medium_large_win_fraction: float = 0.60,
    minimum_improved_scale_tiers: int = 2,
    minimum_improved_instances: int = 2,
    minimum_speedup: float = 1.50,
    minimum_gap_reduction: float = 0.20,
) -> SuccessGateResult:
    """Evaluate the preregistered paper-level B&C success criteria.

    The aggregate effect must clear its substantive threshold while its paired
    confidence interval excludes the neutral value (1 for speedup, 0 for gap
    reduction).  Ties and unresolved medium/large pairs count against the 60%
    win threshold.  Any caller-declared systematic key regression fails the
    gate even if the primary aggregate metric improves.
    """

    if not primary_pairs:
        raise ValueError("success evaluation requires primary paired results")
    if not math.isfinite(minimum_medium_large_win_fraction) or not (
        0.0 < minimum_medium_large_win_fraction <= 1.0
    ):
        raise ValueError("minimum_medium_large_win_fraction must lie in (0, 1]")
    for name, value in (
        ("minimum_improved_scale_tiers", minimum_improved_scale_tiers),
        ("minimum_improved_instances", minimum_improved_instances),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if not math.isfinite(minimum_speedup) or minimum_speedup <= 1.0:
        raise ValueError("minimum_speedup must exceed one")
    if (
        not math.isfinite(minimum_gap_reduction)
        or not 0.0 < minimum_gap_reduction < 1.0
    ):
        raise ValueError("minimum_gap_reduction must lie in (0, 1)")

    target_tiers = {tier.lower() for tier in medium_large_tiers}
    if not target_tiers:
        raise ValueError("medium_large_tiers cannot be empty")
    medium_large = [
        pair for pair in primary_pairs if pair.scale_tier.lower() in target_tiers
    ]
    medium_large_wins = sum(
        pair.winner == PairWinner.CANDIDATE for pair in medium_large
    )
    medium_large_rate = medium_large_wins / len(medium_large) if medium_large else 0.0
    improved_pairs = [
        pair for pair in primary_pairs if pair.winner == PairWinner.CANDIDATE
    ]
    improved_tiers = tuple(sorted({pair.scale_tier.lower() for pair in improved_pairs}))
    improved_instances = len({pair.instance_id for pair in improved_pairs})

    if speedup is not None:
        speedup.validate()
    if gap_reduction is not None:
        gap_reduction.validate()
    speed_gate = bool(
        speedup is not None
        and speedup.estimate >= minimum_speedup
        and speedup.ci_lower > 1.0
    )
    gap_gate = bool(
        gap_reduction is not None
        and gap_reduction.estimate >= minimum_gap_reduction
        and gap_reduction.ci_lower > 0.0
    )
    regression_names = tuple(sorted(set(systematic_regressions)))

    conditions = {
        "medium_large_win_fraction": (
            bool(medium_large)
            and medium_large_rate >= minimum_medium_large_win_fraction
        ),
        "multiple_scale_tiers": len(improved_tiers) >= minimum_improved_scale_tiers,
        "multiple_instances": improved_instances >= minimum_improved_instances,
        "substantive_aggregate_improvement": speed_gate or gap_gate,
        "no_systematic_key_regression": not regression_names,
    }
    reasons: list[str] = []
    if not conditions["medium_large_win_fraction"]:
        reasons.append(
            "medium/large candidate-win fraction is below the preregistered threshold"
        )
    if not conditions["multiple_scale_tiers"]:
        reasons.append("candidate improvements do not span enough scale tiers")
    if not conditions["multiple_instances"]:
        reasons.append("candidate improvements occur on too few distinct instances")
    if not conditions["substantive_aggregate_improvement"]:
        reasons.append(
            "the selected aggregate effect does not clear its effect and CI gate"
        )
    if regression_names:
        reasons.append("systematic key regressions: " + ", ".join(regression_names))
    return SuccessGateResult(
        passed=all(conditions.values()),
        conditions=conditions,
        reasons=tuple(reasons),
        medium_large_win_rate=medium_large_rate,
        medium_large_candidate_wins=medium_large_wins,
        medium_large_pair_count=len(medium_large),
        improved_scale_tiers=improved_tiers,
        improved_instance_count=improved_instances,
        speed_gate_passed=speed_gate,
        gap_gate_passed=gap_gate,
    )


def winner_counts(pairs: Iterable[PairAssessment]) -> Mapping[str, int]:
    """Return stable counts including ties and unresolved pairs."""

    counts = Counter(pair.winner.value for pair in pairs)
    return {winner.value: counts[winner.value] for winner in PairWinner}


__all__ = [
    "EffectEstimate",
    "MinimizationProgressMetrics",
    "PairAssessment",
    "PairWinner",
    "PreregisteredMetric",
    "SuccessGateResult",
    "assess_pair",
    "evaluate_success_gate",
    "honest_termination_label",
    "identify_systematic_regressions",
    "shifted_geomean_gap_reduction",
    "shifted_geomean_speedup",
    "summarize_minimization_progress",
    "winner_counts",
]
