"""Dependency-light statistics for paired branch-and-cut experiments.

The functions in this module deliberately keep censoring and missing solver
states explicit.  A run without an incumbent is not silently assigned a zero
gap, and a failed run is not silently removed from a performance profile.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast


Pair = tuple[float, float]  # (baseline, candidate)
Statistic = Callable[[Sequence[Pair]], float]

_STATUS_NAMES = {
    1: "LOADED",
    2: "OPTIMAL",
    3: "INFEASIBLE",
    4: "INF_OR_UNBD",
    5: "UNBOUNDED",
    6: "CUTOFF",
    7: "ITERATION_LIMIT",
    8: "NODE_LIMIT",
    9: "TIME_LIMIT",
    10: "SOLUTION_LIMIT",
    11: "INTERRUPTED",
    12: "NUMERIC",
    13: "SUBOPTIMAL",
    14: "INPROGRESS",
    15: "USER_OBJ_LIMIT",
    16: "WORK_LIMIT",
    17: "MEM_LIMIT",
}

_GUROBI_INFINITY_THRESHOLD = 0.99e100


def _finite(value: float, name: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _finite_observation(value: Any) -> float | None:
    """Convert solver numeric output to a finite observation or missingness."""

    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(number) or abs(number) >= _GUROBI_INFINITY_THRESHOLD:
        return None
    return number


def solver_status_name(value: Any) -> str:
    """Normalize an integer or textual solver status without inferring success."""

    if value is None or isinstance(value, bool):
        return "UNKNOWN"
    if isinstance(value, str):
        stripped = value.strip().upper()
        if not stripped:
            return "UNKNOWN"
        try:
            numeric = int(stripped)
        except ValueError:
            return stripped
        return _STATUS_NAMES.get(numeric, f"STATUS_{numeric}")
    try:
        numeric = int(value)
    except (TypeError, ValueError, OverflowError):
        return "UNKNOWN"
    return _STATUS_NAMES.get(numeric, f"STATUS_{numeric}")


def shifted_geometric_mean(values: Iterable[float], shift: float = 1.0) -> float:
    """Return ``exp(mean(log(value + shift))) - shift``.

    The paper-facing performance metrics handled here are nonnegative.  A
    strictly positive shift permits zero runtimes or zero final gaps without an
    arbitrary epsilon substitution.
    """

    offset = _finite(shift, "shift")
    if offset <= 0.0:
        raise ValueError("shift must be strictly positive")
    sample = tuple(_finite(value, "value") for value in values)
    if not sample:
        raise ValueError("shifted geometric mean requires at least one value")
    if any(value < 0.0 for value in sample):
        raise ValueError("shifted geometric mean values must be nonnegative")
    result = math.exp(
        math.fsum(math.log(value + offset) for value in sample) / len(sample)
    )
    result -= offset
    return 0.0 if -1e-15 < result < 0.0 else result


def normalized_gap(
    incumbent: float | None,
    bound: float | None,
    *,
    sense: str = "min",
    objective_scale: float = 1.0,
    feasibility_tolerance: float = 1e-9,
) -> float | None:
    """Return a sign-robust relative primal-dual gap.

    ``None`` means that a finite incumbent or bound is unavailable.  For a
    minimization problem the numerator is ``incumbent - bound``; it is reversed
    for maximization.  The denominator
    ``max(abs(incumbent), abs(bound), objective_scale)`` remains meaningful for
    zero and sign-changing objectives.  A material primal/dual ordering error is
    rejected instead of being disguised as a zero gap.
    """

    if incumbent is None or bound is None:
        return None
    primal = float(incumbent)
    dual = float(bound)
    if not math.isfinite(primal) or not math.isfinite(dual):
        return None
    scale_floor = _finite(objective_scale, "objective_scale")
    tolerance = _finite(feasibility_tolerance, "feasibility_tolerance")
    if scale_floor <= 0.0:
        raise ValueError("objective_scale must be strictly positive")
    if tolerance < 0.0:
        raise ValueError("feasibility_tolerance must be nonnegative")
    normalized_sense = sense.lower()
    if normalized_sense == "min":
        difference = primal - dual
    elif normalized_sense == "max":
        difference = dual - primal
    else:
        raise ValueError("sense must be 'min' or 'max'")
    denominator = max(abs(primal), abs(dual), scale_floor)
    if difference < -tolerance * denominator:
        raise ValueError("incumbent and bound have an invalid primal-dual ordering")
    return max(0.0, difference) / denominator


def step_integral(
    points: Iterable[tuple[float, float]],
    horizon: float,
    *,
    initial_value: float = 0.0,
) -> float:
    """Integrate a right-continuous, piecewise-constant trajectory.

    Points after ``horizon`` are censored.  Repeated timestamps are applied in
    input order and therefore contribute no artificial area.
    """

    end = _finite(horizon, "horizon")
    current_value = _finite(initial_value, "initial_value")
    if end < 0.0:
        raise ValueError("horizon must be nonnegative")
    indexed_points: list[tuple[float, int, float]] = []
    for index, (time_value, value) in enumerate(points):
        event_time = _finite(time_value, "trajectory time")
        event_value = _finite(value, "trajectory value")
        if event_time < 0.0:
            raise ValueError("trajectory times must be nonnegative")
        if event_time <= end:
            indexed_points.append((event_time, index, event_value))
    indexed_points.sort(key=lambda item: (item[0], item[1]))

    area = 0.0
    current_time = 0.0
    for event_time, _index, event_value in indexed_points:
        area += (event_time - current_time) * current_value
        current_time = event_time
        current_value = event_value
    area += (end - current_time) * current_value
    return area


@dataclass(frozen=True)
class TrajectoryPoint:
    """One solver progress event measured from the start of the run."""

    time: float
    incumbent: float | None = None
    bound: float | None = None


@dataclass(frozen=True)
class ProgressTrajectory:
    """Sanitized progress events on the total algorithm-runtime clock.

    ``preparation_time`` shifts callback runtimes, which start at the main
    ``optimize`` call, onto the matched algorithm clock that also charges
    static-cut and MIP-start preparation.  ``status_name`` is merely the last
    reported solver status; only ``OPTIMAL`` is evidence of proven optimality.
    """

    points: tuple[TrajectoryPoint, ...]
    preparation_time: float
    observed_end_time: float
    status_name: str
    status_time: float | None


def trajectory_from_progress_events(
    events: Iterable[Mapping[str, object]],
    *,
    preparation_time_seconds: float = 0.0,
) -> ProgressTrajectory:
    """Sanitize raw progress mappings and align them to algorithm runtime.

    Infinite/NaN incumbent and bound values, including Gurobi's ``1e100``
    sentinel, are retained as explicit missing observations.  Events sharing a
    timestamp remain in input order.  A malformed finite-time field is rejected
    because silently reordering or timing such an event would corrupt both
    time-to-target and integral statistics.
    """

    preparation = _finite(preparation_time_seconds, "preparation_time_seconds")
    if preparation < 0.0:
        raise ValueError("preparation_time_seconds must be nonnegative")

    indexed_points: list[tuple[float, int, TrajectoryPoint]] = []
    observed_end = preparation
    status_name = "UNKNOWN"
    status_time: float | None = None
    for index, event in enumerate(events):
        if not isinstance(event, Mapping):
            raise TypeError("every progress event must be a mapping")
        reported_status = (
            solver_status_name(event.get("status")) if "status" in event else None
        )

        total_runtime = _finite_observation(
            event.get("total_algorithm_runtime_seconds")
        )
        if total_runtime is not None:
            if total_runtime < 0.0:
                raise ValueError("total algorithm runtime must be nonnegative")
            observed_end = max(observed_end, total_runtime)

        raw_runtime = event.get("runtime_seconds", event.get("time_seconds"))
        if raw_runtime is None:
            if reported_status is not None:
                status_name = reported_status
                status_time = total_runtime
            continue
        if isinstance(raw_runtime, bool):
            raise ValueError("progress runtime must be finite and nonnegative")
        solve_runtime = _finite_observation(raw_runtime)
        if solve_runtime is None or solve_runtime < 0.0:
            raise ValueError("progress runtime must be finite and nonnegative")
        algorithm_time = preparation + solve_runtime
        observed_end = max(observed_end, algorithm_time)
        if reported_status is not None:
            status_name = reported_status
            status_time = algorithm_time
        point = TrajectoryPoint(
            time=algorithm_time,
            incumbent=_finite_observation(event.get("incumbent")),
            bound=_finite_observation(event.get("bound")),
        )
        indexed_points.append((algorithm_time, index, point))

    indexed_points.sort(key=lambda item: (item[0], item[1]))
    return ProgressTrajectory(
        points=tuple(item[2] for item in indexed_points),
        preparation_time=preparation,
        observed_end_time=observed_end,
        status_name=status_name,
        status_time=status_time,
    )


@dataclass(frozen=True)
class PrimalDualIntegral:
    """Piecewise-constant primal-dual integral and terminal diagnostics."""

    integral: float
    average_gap: float
    terminal_gap: float | None
    first_incumbent_time: float | None
    horizon: float


def _trajectory_events(
    points: Iterable[TrajectoryPoint], horizon: float
) -> tuple[TrajectoryPoint, ...]:
    indexed: list[tuple[float, int, TrajectoryPoint]] = []
    for index, point in enumerate(points):
        time_value = _finite(point.time, "trajectory time")
        if time_value < 0.0:
            raise ValueError("trajectory times must be nonnegative")
        if time_value <= horizon:
            indexed.append(
                (
                    time_value,
                    index,
                    TrajectoryPoint(
                        time=time_value,
                        incumbent=_finite_observation(point.incumbent),
                        bound=_finite_observation(point.bound),
                    ),
                )
            )
    indexed.sort(key=lambda item: (item[0], item[1]))
    return tuple(item[2] for item in indexed)


def _update_best(
    current: float | None,
    proposed: float | None,
    *,
    prefer: str,
) -> float | None:
    if proposed is None:
        return current
    value = float(proposed)
    if current is None:
        return value
    return min(current, value) if prefer == "smaller" else max(current, value)


def primal_dual_integral(
    points: Iterable[TrajectoryPoint],
    horizon: float,
    *,
    sense: str = "min",
    missing_gap_penalty: float = 1.0,
    objective_scale: float = 1.0,
) -> PrimalDualIntegral:
    """Integrate the best-so-far normalized gap over a fixed horizon.

    Until both a finite incumbent and bound exist, the interval receives the
    explicit ``missing_gap_penalty``.  This prevents no-incumbent periods from
    looking artificially good.  Incumbents and bounds are monotonized in the
    correct direction before integration, as solver progress logs may repeat or
    report non-improving values.
    """

    end = _finite(horizon, "horizon")
    penalty = _finite(missing_gap_penalty, "missing_gap_penalty")
    if end <= 0.0:
        raise ValueError("horizon must be strictly positive")
    if penalty < 0.0:
        raise ValueError("missing_gap_penalty must be nonnegative")
    normalized_sense = sense.lower()
    if normalized_sense not in {"min", "max"}:
        raise ValueError("sense must be 'min' or 'max'")
    events = _trajectory_events(points, end)

    incumbent: float | None = None
    bound: float | None = None
    first_incumbent_time: float | None = None
    current_time = 0.0
    area = 0.0

    def current_gap() -> float:
        gap = normalized_gap(
            incumbent,
            bound,
            sense=normalized_sense,
            objective_scale=objective_scale,
        )
        return penalty if gap is None else gap

    for point in events:
        event_time = float(point.time)
        area += (event_time - current_time) * current_gap()
        current_time = event_time
        if point.incumbent is not None and first_incumbent_time is None:
            first_incumbent_time = event_time
        incumbent = _update_best(
            incumbent,
            point.incumbent,
            prefer="smaller" if normalized_sense == "min" else "larger",
        )
        bound = _update_best(
            bound,
            point.bound,
            prefer="larger" if normalized_sense == "min" else "smaller",
        )
    area += (end - current_time) * current_gap()
    terminal = normalized_gap(
        incumbent,
        bound,
        sense=normalized_sense,
        objective_scale=objective_scale,
    )
    return PrimalDualIntegral(
        integral=area,
        average_gap=area / end,
        terminal_gap=terminal,
        first_incumbent_time=first_incumbent_time,
        horizon=end,
    )


def primal_dual_integral_from_progress_events(
    events: Iterable[Mapping[str, object]],
    horizon: float,
    *,
    preparation_time_seconds: float = 0.0,
    missing_gap_penalty: float = 1.0,
    objective_scale: float = 1.0,
) -> PrimalDualIntegral:
    """Compute a minimization primal-dual integral from raw progress events.

    ``average_gap`` in the result is the time-normalized integral.  The fixed
    ``horizon`` is the matched total algorithm budget, so preparation time and
    no-incumbent intervals are charged rather than silently discarded.
    """

    trajectory = trajectory_from_progress_events(
        events,
        preparation_time_seconds=preparation_time_seconds,
    )
    return primal_dual_integral(
        trajectory.points,
        horizon,
        sense="min",
        missing_gap_penalty=missing_gap_penalty,
        objective_scale=objective_scale,
    )


def terminal_best_incumbent(
    points: Iterable[TrajectoryPoint], horizon: float, *, sense: str = "min"
) -> float | None:
    """Return the best incumbent available no later than ``horizon``."""

    end = _finite(horizon, "horizon")
    if end < 0.0:
        raise ValueError("horizon must be nonnegative")
    normalized_sense = sense.lower()
    if normalized_sense not in {"min", "max"}:
        raise ValueError("sense must be 'min' or 'max'")
    best: float | None = None
    for point in _trajectory_events(points, end):
        best = _update_best(
            best,
            point.incumbent,
            prefer="smaller" if normalized_sense == "min" else "larger",
        )
    return best


def terminal_best_bound(
    points: Iterable[TrajectoryPoint], horizon: float, *, sense: str = "min"
) -> float | None:
    """Return the strongest finite dual bound available by ``horizon``."""

    end = _finite(horizon, "horizon")
    if end < 0.0:
        raise ValueError("horizon must be nonnegative")
    normalized_sense = sense.lower()
    if normalized_sense not in {"min", "max"}:
        raise ValueError("sense must be 'min' or 'max'")
    best: float | None = None
    for point in _trajectory_events(points, end):
        best = _update_best(
            best,
            point.bound,
            prefer="larger" if normalized_sense == "min" else "smaller",
        )
    return best


def time_to_target(
    points: Iterable[TrajectoryPoint],
    target: float,
    horizon: float,
    *,
    sense: str = "min",
    absolute_tolerance: float = 1e-9,
    relative_tolerance: float = 1e-9,
) -> float | None:
    """Return first time an incumbent reaches a target, or ``None`` if censored."""

    goal = _finite(target, "target")
    end = _finite(horizon, "horizon")
    abs_tol = _finite(absolute_tolerance, "absolute_tolerance")
    rel_tol = _finite(relative_tolerance, "relative_tolerance")
    if end < 0.0 or abs_tol < 0.0 or rel_tol < 0.0:
        raise ValueError("horizon and tolerances must be nonnegative")
    normalized_sense = sense.lower()
    if normalized_sense not in {"min", "max"}:
        raise ValueError("sense must be 'min' or 'max'")
    tolerance = abs_tol + rel_tol * max(1.0, abs(goal))
    best: float | None = None
    for point in _trajectory_events(points, end):
        best = _update_best(
            best,
            point.incumbent,
            prefer="smaller" if normalized_sense == "min" else "larger",
        )
        if best is None:
            continue
        reached = (
            best <= goal + tolerance
            if normalized_sense == "min"
            else best >= goal - tolerance
        )
        if reached:
            return float(point.time)
    return None


def time_to_target_from_progress_events(
    events: Iterable[Mapping[str, object]],
    target: float,
    horizon: float,
    *,
    preparation_time_seconds: float = 0.0,
    absolute_tolerance: float = 1e-9,
    relative_tolerance: float = 1e-9,
) -> float | None:
    """Return minimization first-hit time, or ``None`` for right-censoring."""

    trajectory = trajectory_from_progress_events(
        events,
        preparation_time_seconds=preparation_time_seconds,
    )
    return time_to_target(
        trajectory.points,
        target,
        horizon,
        sense="min",
        absolute_tolerance=absolute_tolerance,
        relative_tolerance=relative_tolerance,
    )


@dataclass(frozen=True)
class CommonTargetTimes:
    """Paired times to one identical target; ``None`` denotes right-censoring."""

    target: float | None
    baseline_time: float | None
    candidate_time: float | None
    horizon: float
    baseline_final_incumbent: float | None = None
    candidate_final_incumbent: float | None = None

    @property
    def baseline_hit(self) -> bool:
        return self.baseline_time is not None

    @property
    def candidate_hit(self) -> bool:
        return self.candidate_time is not None

    @property
    def target_available(self) -> bool:
        return self.target is not None

    @property
    def baseline_incumbent_missing(self) -> bool:
        return self.baseline_final_incumbent is None

    @property
    def candidate_incumbent_missing(self) -> bool:
        return self.candidate_final_incumbent is None


@dataclass(frozen=True)
class CensoredTimeContract:
    """One preregistered administrative-censoring contract."""

    horizon: float
    censor_value: float
    strategy: str

    def validate(self) -> None:
        if isinstance(self.horizon, bool):
            raise ValueError("censor horizon must be finite and strictly positive")
        horizon = _finite(self.horizon, "censor horizon")
        if horizon <= 0.0:
            raise ValueError("censor horizon must be strictly positive")
        if isinstance(self.censor_value, bool):
            raise ValueError("censor value must be finite")
        censor_value = _finite(self.censor_value, "censor value")
        if self.strategy == "horizon":
            if censor_value != horizon:
                raise ValueError(
                    "horizon censoring requires censor_value to equal horizon"
                )
        elif self.strategy == "fixed_cap":
            if censor_value < horizon:
                raise ValueError(
                    "fixed-cap censoring requires censor_value at least horizon"
                )
        else:
            raise ValueError("censor strategy must be 'horizon' or 'fixed_cap'")


@dataclass(frozen=True)
class PairedCensoredTimeInput:
    """Numeric paired time input that retains both right-censoring flags.

    ``baseline_value`` and ``candidate_value`` are suitable for an ordinary
    paired lower-is-better statistic.  A missing hit is represented by the
    preregistered ``censor_value`` rather than being removed from the sample.
    The Boolean flags must travel with the numeric values so that tables and
    sensitivity analyses can still disclose which observations were censored.
    """

    baseline_value: float
    candidate_value: float
    baseline_censored: bool
    candidate_censored: bool
    horizon: float
    censor_value: float
    strategy: str

    @property
    def contract(self) -> CensoredTimeContract:
        """Return the complete preregistered censoring contract."""

        if isinstance(self.horizon, bool):
            raise ValueError("censor horizon must be finite and strictly positive")
        if isinstance(self.censor_value, bool):
            raise ValueError("censor value must be finite")
        return CensoredTimeContract(
            horizon=_finite(self.horizon, "censor horizon"),
            censor_value=_finite(self.censor_value, "censor value"),
            strategy=self.strategy,
        )

    def validate(self) -> None:
        """Reject malformed direct construction and inconsistent censor flags."""

        contract = self.contract
        contract.validate()
        for name, value, censored in (
            ("baseline", self.baseline_value, self.baseline_censored),
            ("candidate", self.candidate_value, self.candidate_censored),
        ):
            if not isinstance(censored, bool):
                raise ValueError(f"{name}_censored must be Boolean")
            if isinstance(value, bool):
                raise ValueError(f"{name}_value must be finite")
            observed = _finite(value, f"{name}_value")
            if observed < 0.0:
                raise ValueError(f"{name}_value must be nonnegative")
            if censored:
                if observed != contract.censor_value:
                    raise ValueError(
                        f"censored {name}_value must equal the censor value"
                    )
            elif observed > contract.horizon:
                raise ValueError(f"uncensored {name}_value exceeds the common horizon")

    @property
    def pair(self) -> Pair:
        """Return only the numeric pair for a statistic.

        Use ``PairedCaseObservation.from_censored_time_input`` when entering a
        case-cluster analysis; constructing a generic observation from this
        pair would intentionally omit its censor audit.
        """

        return (self.baseline_value, self.candidate_value)


def paired_censored_time_input(
    baseline_time: float | None,
    candidate_time: float | None,
    horizon: float,
    *,
    censor_cap: float | None = None,
) -> PairedCensoredTimeInput:
    """Convert paired right-censored hit times without complete-hit filtering.

    With ``censor_cap=None``, a missed target is assigned the common horizon,
    which is the restricted-time input used by an RMST-style analysis.  With an
    explicit finite ``censor_cap >= horizon``, a miss receives that fixed
    preregistered penalty, supporting a PAR-cap-style sensitivity analysis.
    Observed hits remain unchanged and must lie inside the matched horizon.

    This helper deliberately returns censor flags alongside the numeric pair.
    It does not estimate a survival curve and does not claim that cap-imputed
    values are uncensored event times.
    """

    if isinstance(horizon, bool):
        raise ValueError("horizon must be finite and strictly positive")
    end = _finite(horizon, "horizon")
    if end <= 0.0:
        raise ValueError("horizon must be strictly positive")
    if censor_cap is None:
        cap = end
        strategy = "horizon"
    else:
        if isinstance(censor_cap, bool):
            raise ValueError("censor_cap must be finite")
        cap = _finite(censor_cap, "censor_cap")
        if cap < end:
            raise ValueError("censor_cap must be at least the common horizon")
        strategy = "fixed_cap"

    def converted(value: float | None, name: str) -> tuple[float, bool]:
        if value is None:
            return cap, True
        if isinstance(value, bool):
            raise ValueError(f"{name} must be finite or None")
        observed = _finite(value, name)
        if observed < 0.0:
            raise ValueError(f"{name} must be nonnegative")
        if observed > end:
            raise ValueError(f"{name} exceeds the common horizon")
        return observed, False

    baseline_value, baseline_censored = converted(baseline_time, "baseline_time")
    candidate_value, candidate_censored = converted(candidate_time, "candidate_time")
    result = PairedCensoredTimeInput(
        baseline_value=baseline_value,
        candidate_value=candidate_value,
        baseline_censored=baseline_censored,
        candidate_censored=candidate_censored,
        horizon=end,
        censor_value=cap,
        strategy=strategy,
    )
    result.validate()
    return result


def time_to_common_target(
    baseline: Iterable[TrajectoryPoint],
    candidate: Iterable[TrajectoryPoint],
    horizon: float,
    *,
    target: float | None = None,
    sense: str = "min",
) -> CommonTargetTimes:
    """Compute paired first-hit times for an explicit or common-final target.

    If ``target`` is omitted, the target is the worse of the two terminal best
    incumbents (maximum for minimization, minimum for maximization), making it a
    quality level demonstrably reached by both runs.  If either run has no
    incumbent, no common-final target exists and both times remain censored.
    """

    end = _finite(horizon, "horizon")
    if end < 0.0:
        raise ValueError("horizon must be nonnegative")
    normalized_sense = sense.lower()
    if normalized_sense not in {"min", "max"}:
        raise ValueError("sense must be 'min' or 'max'")
    baseline_points = tuple(baseline)
    candidate_points = tuple(candidate)
    baseline_final = terminal_best_incumbent(
        baseline_points, end, sense=normalized_sense
    )
    candidate_final = terminal_best_incumbent(
        candidate_points, end, sense=normalized_sense
    )
    common_target = target
    if common_target is None:
        if baseline_final is None or candidate_final is None:
            return CommonTargetTimes(
                target=None,
                baseline_time=None,
                candidate_time=None,
                horizon=end,
                baseline_final_incumbent=baseline_final,
                candidate_final_incumbent=candidate_final,
            )
        common_target = (
            max(baseline_final, candidate_final)
            if normalized_sense == "min"
            else min(baseline_final, candidate_final)
        )
    common_target = _finite(common_target, "target")
    return CommonTargetTimes(
        target=common_target,
        baseline_time=time_to_target(
            baseline_points, common_target, end, sense=normalized_sense
        ),
        candidate_time=time_to_target(
            candidate_points, common_target, end, sense=normalized_sense
        ),
        horizon=end,
        baseline_final_incumbent=baseline_final,
        candidate_final_incumbent=candidate_final,
    )


def time_to_common_target_from_progress_events(
    baseline_events: Iterable[Mapping[str, object]],
    candidate_events: Iterable[Mapping[str, object]],
    horizon: float,
    *,
    baseline_preparation_time_seconds: float = 0.0,
    candidate_preparation_time_seconds: float = 0.0,
) -> CommonTargetTimes:
    """Use the worse finite final incumbent as a paired minimization target.

    The target exists only when both runs have an incumbent within the common
    horizon.  This prevents the method with the better final solution from
    receiving a target that its paired comparator could never attain.  Which
    side is missing remains explicit in the returned terminal incumbents.
    """

    baseline = trajectory_from_progress_events(
        baseline_events,
        preparation_time_seconds=baseline_preparation_time_seconds,
    )
    candidate = trajectory_from_progress_events(
        candidate_events,
        preparation_time_seconds=candidate_preparation_time_seconds,
    )
    return time_to_common_target(
        baseline.points,
        candidate.points,
        horizon,
        sense="min",
    )


@dataclass(frozen=True)
class BootstrapInterval:
    """Percentile paired-bootstrap interval with reproducibility metadata."""

    estimate: float
    lower: float
    upper: float
    confidence: float
    resamples: int
    seed: int
    pair_count: int


@dataclass(frozen=True)
class PairedCaseObservation:
    """One same-case, same-solver-seed paired benchmark observation."""

    case_id: str
    solver_seed: int
    scale: str
    baseline: float
    candidate: float
    censored_time_input: PairedCensoredTimeInput | None = None

    @classmethod
    def from_censored_time_input(
        cls,
        *,
        case_id: str,
        solver_seed: int,
        scale: str,
        censored_time_input: PairedCensoredTimeInput,
    ) -> PairedCaseObservation:
        """Construct an audited row without separating values from censor flags."""

        if not isinstance(censored_time_input, PairedCensoredTimeInput):
            raise TypeError("censored_time_input must be a PairedCensoredTimeInput")
        censored_time_input.validate()
        return cls(
            case_id=case_id,
            solver_seed=solver_seed,
            scale=scale,
            baseline=float(censored_time_input.baseline_value),
            candidate=float(censored_time_input.candidate_value),
            censored_time_input=censored_time_input,
        )

    @property
    def pair(self) -> Pair:
        return (self.baseline, self.candidate)


@dataclass(frozen=True)
class CasePairCluster:
    """All solver-seed pairs and any aligned censor audit for one case."""

    case_id: str
    scale: str
    solver_seeds: tuple[int, ...]
    pairs: tuple[Pair, ...]
    censored_time_inputs: tuple[PairedCensoredTimeInput, ...] | None = None

    @property
    def pair_count(self) -> int:
        return len(self.pairs)


@dataclass(frozen=True)
class ClusterBootstrapEffectEstimate:
    """Case-clustered effect estimate with percentile confidence limits.

    The first four fields intentionally match the paper-facing
    ``EffectEstimate`` interface.  ``pair_count`` counts solver-seed pairs,
    while ``case_count`` is the independent sampling-unit count used by the
    bootstrap.  The remaining fields make the resampling design reproducible.
    ``censored_time_contract=None`` explicitly identifies an ordinary numeric
    analysis with no attached censor audit; a non-``None`` contract certifies
    that every input row carried one identical contract and discloses all four
    censor-count summaries.
    """

    estimate: float
    ci_lower: float
    ci_upper: float
    pair_count: int
    case_count: int
    confidence: float
    resamples: int
    seed: int
    stratified_by_scale: bool
    scale_case_counts: tuple[tuple[str, int], ...]
    censored_time_contract: CensoredTimeContract | None = None
    censored_time_contracts_by_scale: tuple[tuple[str, CensoredTimeContract], ...] = ()
    baseline_censored_count: int = 0
    candidate_censored_count: int = 0
    either_censored_count: int = 0
    both_censored_count: int = 0

    def validate(self) -> None:
        for name, value in (
            ("estimate", self.estimate),
            ("ci_lower", self.ci_lower),
            ("ci_upper", self.ci_upper),
            ("confidence", self.confidence),
        ):
            _finite(value, name)
        if self.ci_lower > self.ci_upper:
            raise ValueError("confidence interval lower limit exceeds upper limit")
        if not 0.0 < self.confidence < 1.0:
            raise ValueError("confidence must lie strictly between zero and one")
        for name, value, minimum in (
            ("pair_count", self.pair_count, 1),
            ("case_count", self.case_count, 2),
            ("resamples", self.resamples, 1),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer of at least {minimum}")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("seed must be an explicit integer")
        if not isinstance(self.stratified_by_scale, bool):
            raise ValueError("stratified_by_scale must be Boolean")
        if not self.scale_case_counts:
            raise ValueError("scale_case_counts cannot be empty")
        scales: set[str] = set()
        total_cases = 0
        for scale, count in self.scale_case_counts:
            if not isinstance(scale, str) or not scale.strip() or scale in scales:
                raise ValueError("scale_case_counts must use unique nonempty scales")
            if isinstance(count, bool) or not isinstance(count, int) or count < 1:
                raise ValueError("every scale case count must be a positive integer")
            scales.add(scale)
            total_cases += count
        if total_cases != self.case_count:
            raise ValueError("scale case counts do not sum to case_count")
        censor_counts = (
            self.baseline_censored_count,
            self.candidate_censored_count,
            self.either_censored_count,
            self.both_censored_count,
        )
        if any(
            isinstance(count, bool)
            or not isinstance(count, int)
            or not 0 <= count <= self.pair_count
            for count in censor_counts
        ):
            raise ValueError(
                "censored counts must be integers between zero and pair_count"
            )
        contracts_by_scale: dict[str, CensoredTimeContract] = {}
        for scale, contract in self.censored_time_contracts_by_scale:
            if (
                not isinstance(scale, str)
                or not scale.strip()
                or scale in contracts_by_scale
            ):
                raise ValueError(
                    "censored_time_contracts_by_scale must use unique nonempty scales"
                )
            if not isinstance(contract, CensoredTimeContract):
                raise TypeError(
                    "censored_time_contracts_by_scale values must be contracts"
                )
            contract.validate()
            contracts_by_scale[scale] = contract
        if self.censored_time_contract is not None:
            if not isinstance(self.censored_time_contract, CensoredTimeContract):
                raise TypeError("censored_time_contract must be a CensoredTimeContract")
            self.censored_time_contract.validate()
        if contracts_by_scale:
            if set(contracts_by_scale) != scales:
                raise ValueError(
                    "censored-time contract scales must match scale_case_counts"
                )
            unique_contracts = set(contracts_by_scale.values())
            expected_common = (
                next(iter(unique_contracts)) if len(unique_contracts) == 1 else None
            )
            if self.censored_time_contract != expected_common:
                raise ValueError(
                    "common censored-time contract is inconsistent with scale contracts"
                )
        elif any(censor_counts):
            raise ValueError(
                "censored counts require per-scale censored-time contracts"
            )
        elif self.censored_time_contract is not None:
            raise ValueError(
                "common censored-time contract requires per-scale contracts"
            )
        if self.both_censored_count > min(
            self.baseline_censored_count, self.candidate_censored_count
        ):
            raise ValueError("both-censored count exceeds a marginal count")
        expected_either = (
            self.baseline_censored_count
            + self.candidate_censored_count
            - self.both_censored_count
        )
        if self.either_censored_count != expected_either:
            raise ValueError("either-censored count is inconsistent with marginals")


def _minimum_cluster_count(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 2:
        raise ValueError(f"{name} must be an integer of at least two")
    return value


def _validated_seed_set(
    seeds: Iterable[int] | None,
) -> tuple[int, ...] | None:
    if seeds is None:
        return None
    values: list[int] = []
    for seed in seeds:
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("expected solver seeds must be integers")
        values.append(seed)
    if not values:
        raise ValueError("expected solver seeds cannot be empty")
    if len(values) != len(set(values)):
        raise ValueError("expected solver seeds contain duplicates")
    return tuple(sorted(values))


def cluster_paired_observations(
    observations: Iterable[PairedCaseObservation],
    *,
    minimum_cases: int = 2,
    expected_solver_seeds: Iterable[int] | None = None,
    require_balanced_solver_seeds: bool = True,
) -> tuple[CasePairCluster, ...]:
    """Validate and group paired rows by the physical benchmark case.

    The returned order is canonical by ``case_id`` and ``solver_seed``.  A
    ``(case_id, solver_seed)`` key may occur exactly once, each case must belong
    to one scale, and all numeric values must be finite.  By default every case
    must expose the same complete solver-seed set; callers can additionally
    freeze that set with ``expected_solver_seeds`` (for example, five
    preregistered seeds).
    """

    required_cases = _minimum_cluster_count(minimum_cases, "minimum_cases")
    expected = _validated_seed_set(expected_solver_seeds)
    if not isinstance(require_balanced_solver_seeds, bool):
        raise ValueError("require_balanced_solver_seeds must be Boolean")

    grouped: dict[str, dict[int, Pair]] = {}
    grouped_censoring: dict[str, dict[int, PairedCensoredTimeInput | None]] = {}
    scales: dict[str, str] = {}
    censor_contracts_by_scale: dict[str, CensoredTimeContract] = {}
    saw_censor_metadata = False
    saw_generic_row = False
    for row in observations:
        if not isinstance(row, PairedCaseObservation):
            raise TypeError("every observation must be a PairedCaseObservation")
        if not isinstance(row.case_id, str) or not row.case_id.strip():
            raise ValueError("case_id must be a nonempty string")
        if not isinstance(row.scale, str) or not row.scale.strip():
            raise ValueError("scale must be a nonempty string")
        if isinstance(row.solver_seed, bool) or not isinstance(row.solver_seed, int):
            raise ValueError("solver_seed must be an integer")
        case_id = row.case_id.strip()
        scale = row.scale.strip().lower()
        if case_id in scales and scales[case_id] != scale:
            raise ValueError(f"case {case_id!r} belongs to multiple scales")
        scales[case_id] = scale
        case_rows = grouped.setdefault(case_id, {})
        if row.solver_seed in case_rows:
            raise ValueError(
                f"duplicate paired observation for case {case_id!r} "
                f"and solver seed {row.solver_seed}"
            )
        if isinstance(row.baseline, bool) or isinstance(row.candidate, bool):
            raise ValueError("baseline and candidate must be finite numeric values")
        pair = (
            _finite(row.baseline, "baseline"),
            _finite(row.candidate, "candidate"),
        )
        case_rows[row.solver_seed] = pair
        censored_time_input = row.censored_time_input
        if censored_time_input is None:
            saw_generic_row = True
        else:
            if not isinstance(censored_time_input, PairedCensoredTimeInput):
                raise TypeError(
                    "censored_time_input must be a PairedCensoredTimeInput or None"
                )
            censored_time_input.validate()
            if pair != (
                float(censored_time_input.baseline_value),
                float(censored_time_input.candidate_value),
            ):
                raise ValueError(
                    "censored-time values do not match the paired observation"
                )
            row_contract = censored_time_input.contract
            previous_contract = censor_contracts_by_scale.setdefault(
                scale, row_contract
            )
            if row_contract != previous_contract:
                raise ValueError(
                    "clustered censored-time observations use mixed censor contracts "
                    f"within scale {scale!r}"
                )
            saw_censor_metadata = True
        grouped_censoring.setdefault(case_id, {})[row.solver_seed] = censored_time_input

    if saw_censor_metadata and saw_generic_row:
        raise ValueError(
            "clustered observations contain partial censored-time metadata"
        )

    if len(grouped) < required_cases:
        raise ValueError(
            f"case-cluster analysis requires at least {required_cases} distinct cases"
        )

    reference_seeds = expected
    clusters: list[CasePairCluster] = []
    for case_id in sorted(grouped):
        case_rows = grouped[case_id]
        solver_seeds = tuple(sorted(case_rows))
        if reference_seeds is None and require_balanced_solver_seeds:
            reference_seeds = solver_seeds
        if expected is not None and solver_seeds != expected:
            raise ValueError(
                f"case {case_id!r} does not contain the expected solver-seed set"
            )
        if (
            expected is None
            and require_balanced_solver_seeds
            and solver_seeds != reference_seeds
        ):
            raise ValueError(f"case {case_id!r} has an unbalanced solver-seed set")
        clusters.append(
            CasePairCluster(
                case_id=case_id,
                scale=scales[case_id],
                solver_seeds=solver_seeds,
                pairs=tuple(case_rows[seed] for seed in solver_seeds),
                censored_time_inputs=(
                    tuple(
                        cast(
                            PairedCensoredTimeInput,
                            grouped_censoring[case_id][seed],
                        )
                        for seed in solver_seeds
                    )
                    if saw_censor_metadata
                    else None
                ),
            )
        )
    return tuple(clusters)


def _quantile(sorted_values: Sequence[float], probability: float) -> float:
    if not sorted_values:
        raise ValueError("quantile requires a nonempty sample")
    if probability <= 0.0:
        return float(sorted_values[0])
    if probability >= 1.0:
        return float(sorted_values[-1])
    position = probability * (len(sorted_values) - 1)
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    weight = position - lower_index
    return (1.0 - weight) * sorted_values[lower_index] + weight * sorted_values[
        upper_index
    ]


def paired_bootstrap_ci(
    pairs: Iterable[Pair],
    statistic: Statistic,
    *,
    confidence: float = 0.95,
    resamples: int = 10_000,
    seed: int,
) -> BootstrapInterval:
    """Return a deterministic percentile CI by resampling complete pairs."""

    sample = tuple(
        (_finite(baseline, "baseline"), _finite(candidate, "candidate"))
        for baseline, candidate in pairs
    )
    if not sample:
        raise ValueError("paired bootstrap requires at least one pair")
    level = _finite(confidence, "confidence")
    if not 0.0 < level < 1.0:
        raise ValueError("confidence must lie strictly between zero and one")
    if isinstance(resamples, bool) or not isinstance(resamples, int) or resamples < 1:
        raise ValueError("resamples must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an explicit integer")
    estimate = _finite(statistic(sample), "bootstrap statistic")
    rng = random.Random(seed)
    draws: list[float] = []
    for _ in range(resamples):
        resample = tuple(sample[rng.randrange(len(sample))] for _ in sample)
        draws.append(_finite(statistic(resample), "bootstrap statistic"))
    draws.sort()
    tail = 0.5 * (1.0 - level)
    return BootstrapInterval(
        estimate=estimate,
        lower=_quantile(draws, tail),
        upper=_quantile(draws, 1.0 - tail),
        confidence=level,
        resamples=resamples,
        seed=seed,
        pair_count=len(sample),
    )


def case_cluster_bootstrap_effect(
    observations: Iterable[PairedCaseObservation],
    statistic: Statistic,
    *,
    confidence: float = 0.95,
    resamples: int = 10_000,
    seed: int,
    stratify_by_scale: bool = False,
    minimum_cases: int = 2,
    minimum_cases_per_scale: int = 2,
    expected_solver_seeds: Iterable[int] | None = None,
    require_balanced_solver_seeds: bool = True,
) -> ClusterBootstrapEffectEstimate:
    """Estimate an arbitrary paired effect by resampling whole cases.

    A bootstrap draw samples physical cases with replacement and then includes
    *all* solver-seed pairs from every selected case.  Solver seeds are never
    treated as independent bootstrap units.  If ``stratify_by_scale`` is true,
    the original number of cases in each scale is sampled independently and
    retained in every draw.

    The statistic receives a flat sequence of ``(baseline, candidate)`` pairs,
    so the same API supports shifted-geometric-mean speedup, shifted gap
    reduction, paired means, and preregistered capped time-to-target effects.
    It must return a finite value for both the point sample and every resample.
    """

    if not callable(statistic):
        raise TypeError("statistic must be callable")
    level = _finite(confidence, "confidence")
    if not 0.0 < level < 1.0:
        raise ValueError("confidence must lie strictly between zero and one")
    if isinstance(resamples, bool) or not isinstance(resamples, int) or resamples < 1:
        raise ValueError("resamples must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an explicit integer")
    if not isinstance(stratify_by_scale, bool):
        raise ValueError("stratify_by_scale must be Boolean")
    clusters = cluster_paired_observations(
        observations,
        minimum_cases=minimum_cases,
        expected_solver_seeds=expected_solver_seeds,
        require_balanced_solver_seeds=require_balanced_solver_seeds,
    )

    clusters_by_scale: dict[str, list[CasePairCluster]] = {}
    for cluster in clusters:
        clusters_by_scale.setdefault(cluster.scale, []).append(cluster)
    if stratify_by_scale:
        required_per_scale = _minimum_cluster_count(
            minimum_cases_per_scale, "minimum_cases_per_scale"
        )
        undersized = {
            scale: len(scale_clusters)
            for scale, scale_clusters in clusters_by_scale.items()
            if len(scale_clusters) < required_per_scale
        }
        if undersized:
            details = ", ".join(
                f"{scale}={count}" for scale, count in sorted(undersized.items())
            )
            raise ValueError(
                "scale-stratified cluster bootstrap requires at least "
                f"{required_per_scale} cases per scale; found {details}"
            )

    def flattened(selected: Iterable[CasePairCluster]) -> tuple[Pair, ...]:
        return tuple(pair for cluster in selected for pair in cluster.pairs)

    point_pairs = flattened(clusters)
    censored_time_inputs = tuple(
        item for cluster in clusters for item in (cluster.censored_time_inputs or ())
    )
    censored_time_contracts_by_scale = tuple(
        sorted(
            {
                cluster.scale: cast(
                    tuple[PairedCensoredTimeInput, ...],
                    cluster.censored_time_inputs,
                )[0].contract
                for cluster in clusters
                if cluster.censored_time_inputs
            }.items()
        )
    )
    unique_censored_time_contracts = {
        contract for _scale, contract in censored_time_contracts_by_scale
    }
    censored_time_contract = (
        next(iter(unique_censored_time_contracts))
        if len(unique_censored_time_contracts) == 1
        else None
    )
    baseline_censored_count = sum(
        item.baseline_censored for item in censored_time_inputs
    )
    candidate_censored_count = sum(
        item.candidate_censored for item in censored_time_inputs
    )
    either_censored_count = sum(
        item.baseline_censored or item.candidate_censored
        for item in censored_time_inputs
    )
    both_censored_count = sum(
        item.baseline_censored and item.candidate_censored
        for item in censored_time_inputs
    )
    estimate = _finite(statistic(point_pairs), "cluster bootstrap statistic")
    rng = random.Random(seed)
    draws: list[float] = []
    if stratify_by_scale:
        scale_pools = tuple(
            (scale, tuple(clusters_by_scale[scale]))
            for scale in sorted(clusters_by_scale)
        )
        for _ in range(resamples):
            selected: list[CasePairCluster] = []
            for _scale, pool in scale_pools:
                selected.extend(pool[rng.randrange(len(pool))] for _ in pool)
            draws.append(
                _finite(
                    statistic(flattened(selected)),
                    "cluster bootstrap statistic",
                )
            )
    else:
        for _ in range(resamples):
            selected = [clusters[rng.randrange(len(clusters))] for _ in clusters]
            draws.append(
                _finite(
                    statistic(flattened(selected)),
                    "cluster bootstrap statistic",
                )
            )
    draws.sort()
    tail = 0.5 * (1.0 - level)
    result = ClusterBootstrapEffectEstimate(
        estimate=estimate,
        ci_lower=_quantile(draws, tail),
        ci_upper=_quantile(draws, 1.0 - tail),
        pair_count=len(point_pairs),
        case_count=len(clusters),
        confidence=level,
        resamples=resamples,
        seed=seed,
        stratified_by_scale=stratify_by_scale,
        scale_case_counts=tuple(
            (scale, len(clusters_by_scale[scale]))
            for scale in sorted(clusters_by_scale)
        ),
        censored_time_contract=censored_time_contract,
        censored_time_contracts_by_scale=censored_time_contracts_by_scale,
        baseline_censored_count=baseline_censored_count,
        candidate_censored_count=candidate_censored_count,
        either_censored_count=either_censored_count,
        both_censored_count=both_censored_count,
    )
    result.validate()
    return result


def mean_candidate_minus_baseline(pairs: Sequence[Pair]) -> float:
    """Mean paired difference, with negative values favoring minimization."""

    if not pairs:
        raise ValueError("mean paired difference requires at least one pair")
    return math.fsum(candidate - baseline for baseline, candidate in pairs) / len(pairs)


@dataclass(frozen=True)
class SignTestResult:
    """Exact binomial sign test after removing tolerance-level ties."""

    positive: int
    negative: int
    ties: int
    alternative: str
    p_value: float

    @property
    def effective_pairs(self) -> int:
        return self.positive + self.negative


def exact_sign_test(
    differences: Iterable[float],
    *,
    alternative: str = "two-sided",
    zero_tolerance: float = 0.0,
) -> SignTestResult:
    """Perform an exact sign test without SciPy.

    Positive differences are treated as candidate improvements.  Supported
    alternatives are ``two-sided``, ``greater`` (more positive differences),
    and ``less``.
    """

    tolerance = _finite(zero_tolerance, "zero_tolerance")
    if tolerance < 0.0:
        raise ValueError("zero_tolerance must be nonnegative")
    normalized_alternative = alternative.lower()
    if normalized_alternative not in {"two-sided", "greater", "less"}:
        raise ValueError("alternative must be 'two-sided', 'greater', or 'less'")
    positive = negative = ties = 0
    for difference in differences:
        value = _finite(difference, "difference")
        if value > tolerance:
            positive += 1
        elif value < -tolerance:
            negative += 1
        else:
            ties += 1
    trials = positive + negative
    if trials == 0:
        p_value = 1.0
    else:
        denominator = 2**trials

        def probability(start: int, stop: int) -> float:
            return (
                math.fsum(math.comb(trials, count) for count in range(start, stop))
                / denominator
            )

        upper = probability(positive, trials + 1)
        lower = probability(0, positive + 1)
        if normalized_alternative == "greater":
            p_value = upper
        elif normalized_alternative == "less":
            p_value = lower
        else:
            p_value = min(1.0, 2.0 * min(lower, upper))
    return SignTestResult(
        positive=positive,
        negative=negative,
        ties=ties,
        alternative=normalized_alternative,
        p_value=p_value,
    )


@dataclass(frozen=True)
class DolanMoreRatios:
    """Performance ratios with explicit all-failed and censored bookkeeping."""

    ratios: Mapping[str, Mapping[str, float]]
    solvers: tuple[str, ...]
    included_problems: tuple[str, ...]
    excluded_all_failed: tuple[str, ...]
    failed_counts: Mapping[str, int]
    failure_ratio: float


def dolan_more_ratios(
    performance: Mapping[str, Mapping[str, float | None]],
    *,
    failure_ratio: float = math.inf,
) -> DolanMoreRatios:
    """Build Dolan--Moré ratios for lower-is-better performance measures.

    ``None`` and positive infinity denote a censored/failed run.  Problems on
    which every solver failed are reported separately and excluded because no
    best observed performance exists.  By default failures retain an infinite
    ratio and therefore never count as solved at a finite profile threshold.
    A finite ``failure_ratio`` may be supplied, but it must be no smaller than
    every observed finite ratio.
    """

    failure = float(failure_ratio)
    if math.isnan(failure) or failure < 1.0:
        raise ValueError("failure_ratio must be at least one or positive infinity")
    solvers = tuple(
        sorted({solver for values in performance.values() for solver in values})
    )
    if not solvers:
        raise ValueError("performance data must contain at least one solver")

    ratios: dict[str, dict[str, float]] = {}
    excluded: list[str] = []
    failures = {solver: 0 for solver in solvers}
    maximum_finite_ratio = 1.0
    for problem in sorted(performance):
        values = performance[problem]
        finite_values: dict[str, float] = {}
        for solver in solvers:
            raw_value = values.get(solver)
            if raw_value is None:
                continue
            value = float(raw_value)
            if math.isnan(value):
                raise ValueError(f"performance[{problem!r}][{solver!r}] is NaN")
            if math.isinf(value):
                if value > 0.0:
                    continue
                raise ValueError("negative-infinite performance is invalid")
            if value <= 0.0:
                raise ValueError("finite performance values must be strictly positive")
            finite_values[solver] = value
        if not finite_values:
            excluded.append(problem)
            continue
        best = min(finite_values.values())
        problem_ratios: dict[str, float] = {}
        for solver in solvers:
            if solver in finite_values:
                ratio = max(1.0, finite_values[solver] / best)
                maximum_finite_ratio = max(maximum_finite_ratio, ratio)
            else:
                ratio = failure
                failures[solver] += 1
            problem_ratios[solver] = ratio
        ratios[problem] = problem_ratios
    if not ratios:
        raise ValueError("all benchmark problems failed for every solver")
    if math.isfinite(failure) and failure < maximum_finite_ratio:
        raise ValueError("finite failure_ratio must cover every observed finite ratio")
    return DolanMoreRatios(
        ratios=ratios,
        solvers=solvers,
        included_problems=tuple(ratios),
        excluded_all_failed=tuple(excluded),
        failed_counts=failures,
        failure_ratio=failure,
    )


@dataclass(frozen=True)
class DolanMoreProfile:
    """Finite performance-profile thresholds and empirical CDF values."""

    taus: tuple[float, ...]
    fractions: Mapping[str, tuple[float, ...]]
    problem_count: int


def dolan_more_profile_points(
    ratio_data: DolanMoreRatios,
    taus: Iterable[float] | None = None,
) -> DolanMoreProfile:
    """Evaluate a Dolan--Moré performance profile at finite thresholds."""

    if taus is None:
        threshold_values = {1.0}
        threshold_values.update(
            ratio
            for values in ratio_data.ratios.values()
            for ratio in values.values()
            if math.isfinite(ratio)
        )
        thresholds = tuple(sorted(threshold_values))
    else:
        thresholds = tuple(sorted({_finite(value, "tau") for value in taus}))
        if not thresholds:
            raise ValueError("taus must contain at least one finite threshold")
        if thresholds[0] < 1.0:
            raise ValueError("every performance-profile threshold must be at least one")
    count = len(ratio_data.included_problems)
    fractions = {
        solver: tuple(
            sum(
                ratio_data.ratios[problem][solver] <= tau
                for problem in ratio_data.included_problems
            )
            / count
            for tau in thresholds
        )
        for solver in ratio_data.solvers
    }
    return DolanMoreProfile(thresholds, fractions, count)


__all__ = [
    "BootstrapInterval",
    "CasePairCluster",
    "CensoredTimeContract",
    "ClusterBootstrapEffectEstimate",
    "CommonTargetTimes",
    "DolanMoreProfile",
    "DolanMoreRatios",
    "Pair",
    "PairedCaseObservation",
    "PairedCensoredTimeInput",
    "ProgressTrajectory",
    "PrimalDualIntegral",
    "SignTestResult",
    "TrajectoryPoint",
    "case_cluster_bootstrap_effect",
    "cluster_paired_observations",
    "dolan_more_profile_points",
    "dolan_more_ratios",
    "exact_sign_test",
    "mean_candidate_minus_baseline",
    "normalized_gap",
    "paired_censored_time_input",
    "paired_bootstrap_ci",
    "primal_dual_integral",
    "primal_dual_integral_from_progress_events",
    "shifted_geometric_mean",
    "solver_status_name",
    "step_integral",
    "terminal_best_bound",
    "terminal_best_incumbent",
    "time_to_common_target",
    "time_to_common_target_from_progress_events",
    "time_to_target",
    "time_to_target_from_progress_events",
    "trajectory_from_progress_events",
]
