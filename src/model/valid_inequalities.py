"""Solver-independent valid inequalities for the OD-aware MLND model.

The routines in this module only separate algebraic inequalities.  They do not
import Gurobi, mutate an optimization model, or retain separation state.  This
makes the formulas independently testable before a caller translates the
returned coefficients into ordinary constraints or callback cuts.

The module also provides solver-independent indexing and coefficient checks for
aggregate swap-route activation bounds.  Those helpers distinguish route flow
from geometric-leg incidence, so one swap route is indexed on two directed
legs but only once in its swap-station throughput bundle.

``QueuePerspectiveCut``
    Supporting hyperplanes for the union of the discrete M/M/1 queue graphs.

``ChoiceCapacityCover``
    Locker-only covers linking destination-locker display decisions to a
    locker's selected service capacity.  They are not valid for mixed delivery,
    where displaying a locker does not force any order into the drone network.

``NearestDisplaySuffixRow``
    Static rows aggregating the existing nearest-open display implications for
    all strictly farther lockers in one destination's display choice set.

``DirectedReturnCut``
    Directed connectivity cuts.  Positive offered flow on corridor ``(i,j)``
    requires a return path from ``j`` to ``i`` in every integral balanced
    network.  Co-located virtual holding states may be contracted into their
    physical lockers before the minimum-cut calculation.

``ODDisaggregatedDirectedReturnCut``
    The corresponding OD-disaggregated inequality.  It replaces the aggregate
    corridor coefficient by the tighter OD/corridor upper bound while reusing
    one directed minimum cut for every OD group on the same corridor.
"""

from __future__ import annotations

import math
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import AbstractSet, Iterable, Mapping, Sequence

Node = str
Arc = tuple[Node, Node]
Market = tuple[Node, Node]
ChoiceKey = tuple[Node, Node, Node]  # (origin, destination, locker)
AlphaKey = tuple[Node, Node]  # (destination, locker)
RouteOptionKey = tuple[Node, Node, Node]  # (physical tail, physical head, option)
ODCorridorKey = tuple[Node, Node, Node, Node]  # (origin, destination, i, j)


def _finite(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _unit_interval(value: float, name: str, tolerance: float) -> float:
    result = _finite(value, name)
    if result < -tolerance or result > 1.0 + tolerance:
        raise ValueError(f"{name} must lie in [0, 1]")
    return min(1.0, max(0.0, result))


def _nonnegative(value: float, name: str, tolerance: float = 0.0) -> float:
    result = _finite(value, name)
    if result < -tolerance:
        raise ValueError(f"{name} must be nonnegative")
    return max(0.0, result)


# ---------------------------------------------------------------------------
# Nearest-open display suffix rows
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NearestDisplaySuffixRow:
    r"""One static nearest-open display aggregation.

    For destination ``d`` and anchor locker ``k``, the represented row is

    ``sum(alpha[d,j] for j in farther_lockers) <= 1 - x[k]``.

    ``farther_lockers`` is derived from the *existing* strict-nearness relation,
    not from the stable locker-ID tie breaker used to order display choices.
    Rows with fewer than two terms are omitted because they are either empty or
    exact duplicates of an existing pairwise ``display_nearest`` constraint.
    """

    destination: Node
    anchor_locker: Node
    farther_lockers: tuple[Node, ...]

    def signature(self) -> tuple[str, Node, Node, tuple[Node, ...]]:
        return (
            "nearest_display_suffix",
            self.destination,
            self.anchor_locker,
            self.farther_lockers,
        )


def build_nearest_display_suffix_rows(
    display_lockers: Mapping[Node, Sequence[Node]],
    closer_lockers: Mapping[AlphaKey, Iterable[Node]],
) -> tuple[NearestDisplaySuffixRow, ...]:
    r"""Build deterministic nonredundant nearest-display suffix rows.

    The base model contains, for each destination ``d``,

    ``sum_j alpha[d,j] <= 1``

    and ``alpha[d,j] <= 1 - x[k]`` whenever ``k`` belongs to the precomputed
    strict-nearness set ``closer_lockers[d,j]``.  With binary ``x[k]``, these
    rows imply the returned aggregation: if ``x[k] = 1``, every farther display
    is zero; if ``x[k] = 0``, the destination display limit bounds their sum by
    one.  The aggregation can nevertheless strengthen the LP relaxation.

    The input display sequence is kept in its canonical instance order.  In the
    current instance builder that order is ``(distance, locker_id)``, while the
    supplied strict-nearness relation uses the tolerance-aware distance test.
    Consequently distance ties are never inferred from sequence position.
    """

    rows: list[NearestDisplaySuffixRow] = []
    for destination in sorted(display_lockers):
        candidates = tuple(display_lockers[destination])
        if len(candidates) != len(set(candidates)):
            raise ValueError(
                f"display_lockers[{destination!r}] contains duplicate lockers"
            )
        candidate_set = set(candidates)
        farther_by_anchor: dict[Node, list[Node]] = {
            locker: [] for locker in candidates
        }

        for farther in candidates:
            key = (destination, farther)
            if key not in closer_lockers:
                raise ValueError(f"missing strict-nearness set for {key!r}")
            nearer = set(closer_lockers[key])
            if farther in nearer:
                raise ValueError(
                    f"strict-nearness set for {key!r} contains the locker itself"
                )
            for anchor in nearer.intersection(candidate_set):
                farther_by_anchor[anchor].append(farther)

        for anchor in candidates:
            farther = tuple(farther_by_anchor[anchor])
            if len(farther) < 2:
                continue
            rows.append(
                NearestDisplaySuffixRow(
                    destination=destination,
                    anchor_locker=anchor,
                    farther_lockers=farther,
                )
            )
    return tuple(rows)


# ---------------------------------------------------------------------------
# Aggregate swap-route activation bounds
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RouteAggregateVUBBound:
    """Safe coefficient for one aggregate route activation inequality.

    ``summed_individual_upper`` is the coefficient obtained by summing the
    existing per-route big-M rows.  The aggregate inequality is worth
    separating only when ``coefficient`` is strictly smaller.
    """

    coefficient: float
    summed_individual_upper: float

    def strictly_strengthens(self, tolerance: float = 1e-12) -> bool:
        """Return whether the aggregate coefficient improves the existing sum."""

        tol = _nonnegative(tolerance, "tolerance")
        scale = max(1.0, self.summed_individual_upper)
        return self.coefficient < self.summed_individual_upper - tol * scale


@dataclass(frozen=True)
class RouteBundleViolation:
    """Evaluation of ``sum(route flow) <= coefficient * activation``."""

    lhs_value: float
    rhs_value: float
    violation: float


def build_swap_route_bundles(
    route_options: Mapping[Arc, Sequence[Node]],
    *,
    direct_option: Node,
) -> tuple[
    dict[Arc, tuple[RouteOptionKey, ...]],
    dict[Node, tuple[RouteOptionKey, ...]],
]:
    """Index swap-route variables by directed leg and by swap station.

    A key ``(u, v, e)`` represents one physical-arc flow routed through swap
    station ``e``.  It therefore enters exactly the two leg bundles ``(u,e)``
    and ``(e,v)``, but only one station-throughput bundle for ``e``.  Direct
    route options enter neither index.
    """

    by_leg: dict[Arc, list[RouteOptionKey]] = defaultdict(list)
    by_station: dict[Node, list[RouteOptionKey]] = defaultdict(list)
    for (u, v), raw_options in sorted(route_options.items()):
        options = tuple(raw_options)
        if len(set(options)) != len(options):
            raise ValueError(f"route options for {(u, v)!r} must be unique")
        for option in options:
            if option == direct_option:
                continue
            key = (u, v, option)
            by_leg[u, option].append(key)
            by_leg[option, v].append(key)
            by_station[option].append(key)

    return (
        {leg: tuple(sorted(keys)) for leg, keys in sorted(by_leg.items())},
        {
            station: tuple(sorted(keys))
            for station, keys in sorted(by_station.items())
        },
    )


def _route_bundle_arc_upper_sum(
    route_keys: Iterable[RouteOptionKey],
    original_arc_upper_bounds: Mapping[Arc, float],
) -> float:
    """Sum original-arc bounds once, even if route keys repeat an arc."""

    arcs = {(u, v) for u, v, _option in route_keys}
    total = 0.0
    for arc in sorted(arcs):
        if arc not in original_arc_upper_bounds:
            raise KeyError(f"missing original-arc upper bound for {arc!r}")
        total += _nonnegative(
            original_arc_upper_bounds[arc],
            f"original_arc_upper_bounds[{arc!r}]",
        )
    return total


def _unique_route_keys(
    route_keys: Iterable[RouteOptionKey],
) -> tuple[RouteOptionKey, ...]:
    """Materialize one route bundle and reject duplicate variable keys.

    Distinct route options on the same original arc may legitimately share its
    single ``psi`` upper bound because route decomposition bounds their sum.
    Repeating an identical key is different: it would count the same variable
    twice in the aggregate row while the coefficient counts its original arc
    only once.  Rejecting that malformed input keeps the helper safe even when
    it is used outside ``build_swap_route_bundles``.
    """

    keys = tuple(route_keys)
    if not keys:
        raise ValueError("route_keys cannot be empty")
    if len(keys) != len(set(keys)):
        raise ValueError("route_keys cannot contain duplicate variable keys")
    return keys


def route_swap_leg_vub_bound(
    route_keys: Iterable[RouteOptionKey],
    original_arc_upper_bounds: Mapping[Arc, float],
    *,
    endpoint_kind: str,
    realized_demand_upper: float,
    locker_service_rates: Sequence[float] = (),
    max_locker_time: float | None = None,
) -> RouteAggregateVUBBound:
    """Return a safe aggregate coefficient for one shared directed swap leg.

    A base's physical inflow/outflow is at most total realized loaded demand.
    A locker's physical inflow/outflow is at most its largest QoS-compatible
    arrival capacity.  The locker departure statement uses the current flow
    balance at the locker and its paired virtual state.  These implications use
    the source/loaded-flow identities and queue/flow balance, not a local energy
    constraint, so they remain valid with formulation v3's system-wide energy
    pool.
    """

    keys = _unique_route_keys(route_keys)
    individual_sum = _route_bundle_arc_upper_sum(
        keys, original_arc_upper_bounds
    )
    demand_upper = _nonnegative(
        realized_demand_upper, "realized_demand_upper"
    )
    normalized_kind = endpoint_kind.lower()
    if normalized_kind == "base":
        structural_upper = demand_upper
    elif normalized_kind == "locker":
        if not locker_service_rates or max_locker_time is None:
            raise ValueError(
                "locker leg bounds require service rates and max_locker_time"
            )
        structural_upper = max(
            stable_arrival_capacity(rate, max_locker_time)
            for rate in locker_service_rates
        )
    else:
        raise ValueError("endpoint_kind must be 'base' or 'locker'")
    return RouteAggregateVUBBound(
        coefficient=min(individual_sum, structural_upper),
        summed_individual_upper=individual_sum,
    )


def route_swap_station_vub_bound(
    route_keys: Iterable[RouteOptionKey],
    original_arc_upper_bounds: Mapping[Arc, float],
    original_arc_categories: Mapping[Arc, str],
    *,
    realized_demand_upper: float,
) -> RouteAggregateVUBBound:
    """Return a safe station-throughput coefficient without double counting.

    The three physical-arc categories ``LH``, ``HH``, and ``HL`` each carry at
    most total realized loaded demand in the current one-HH-corridor-per-``chi``
    formulation.  This follows from the source/loaded-flow identities plus
    aggregate flow balance and is independent of formulation v3's system-wide
    energy pool.  The HH statement must be re-audited if ``chi`` is later
    replaced by multi-hop commodity flow.  A route through the station is
    counted once here, even though it appears in two directed-leg bundles.
    """

    keys = _unique_route_keys(route_keys)
    demand_upper = _nonnegative(
        realized_demand_upper, "realized_demand_upper"
    )
    arcs = {(u, v) for u, v, _option in keys}
    category_upper: dict[str, float] = defaultdict(float)
    for arc in sorted(arcs):
        if arc not in original_arc_upper_bounds:
            raise KeyError(f"missing original-arc upper bound for {arc!r}")
        if arc not in original_arc_categories:
            raise KeyError(f"missing original-arc category for {arc!r}")
        category = str(original_arc_categories[arc]).upper()
        if category not in {"LH", "HH", "HL"}:
            raise ValueError(
                f"original-arc category for {arc!r} must be LH, HH, or HL"
            )
        category_upper[category] += _nonnegative(
            original_arc_upper_bounds[arc],
            f"original_arc_upper_bounds[{arc!r}]",
        )
    individual_sum = sum(category_upper.values())
    coefficient = sum(
        min(demand_upper, category_upper.get(category, 0.0))
        for category in ("LH", "HH", "HL")
    )
    return RouteAggregateVUBBound(
        coefficient=coefficient,
        summed_individual_upper=individual_sum,
    )


def evaluate_route_bundle_vub(
    route_keys: Iterable[RouteOptionKey],
    route_values: Mapping[RouteOptionKey, float],
    *,
    activation_value: float,
    coefficient: float,
    feasibility_tolerance: float = 1e-9,
) -> RouteBundleViolation:
    """Evaluate one aggregate VUB at a supplied fractional point."""

    keys = _unique_route_keys(route_keys)
    tol = _nonnegative(feasibility_tolerance, "feasibility_tolerance")
    activation = _unit_interval(activation_value, "activation_value", tol)
    upper = _nonnegative(coefficient, "coefficient")
    lhs = 0.0
    for key in keys:
        if key not in route_values:
            raise KeyError(f"missing route value for {key!r}")
        lhs += _nonnegative(route_values[key], f"route_values[{key!r}]", tol)
    rhs = upper * activation
    return RouteBundleViolation(
        lhs_value=lhs,
        rhs_value=rhs,
        violation=lhs - rhs,
    )


# ---------------------------------------------------------------------------
# Discrete-capacity queue perspective cuts
# ---------------------------------------------------------------------------


def stable_arrival_capacity(service_rate: float, max_system_time: float) -> float:
    """Return the largest arrival rate allowed by the queue QoS constraint.

    For an M/M/1 queue with service rate ``mu`` and maximum system time ``tau``,
    ``1 / (mu - lambda) <= tau`` is equivalent to

    ``lambda <= kappa = mu - 1 / tau``.

    A nonnegative feasible interval requires ``mu * tau >= 1``.
    """

    mu = _finite(service_rate, "service_rate")
    tau = _finite(max_system_time, "max_system_time")
    if mu <= 0.0 or tau <= 0.0:
        raise ValueError("service_rate and max_system_time must be positive")
    kappa = mu - 1.0 / tau
    if kappa < -1e-12 * max(1.0, mu):
        raise ValueError("queue QoS leaves no nonnegative arrival-rate interval")
    return max(0.0, kappa)


def queue_perspective_intercept(
    service_rate: float,
    arrival_capacity: float,
    pi: float,
) -> float:
    r"""Return :math:`b_c(\pi)` for one discrete queue capacity.

    The queue graph is

    .. math::

        f_c(t)=\frac{t}{\mu_c-t},\qquad 0\le t\le\kappa_c<\mu_c.

    Its support-function intercept is

    .. math::

        b_c(\pi)=\min_{0\le t\le\kappa_c}\{f_c(t)-\pi t\}.

    The implementation uses the exact three-piece formula and is continuous at
    both breakpoints.  Consequently

    ``xi >= pi * lambda + sum_c b_c(pi) * z_c``

    is valid for every ``pi >= 0`` over the discrete-capacity queue union.
    """

    mu = _finite(service_rate, "service_rate")
    kappa = _finite(arrival_capacity, "arrival_capacity")
    slope = _finite(pi, "pi")
    if mu <= 0.0:
        raise ValueError("service_rate must be positive")
    if kappa < 0.0 or kappa >= mu:
        raise ValueError("arrival_capacity must lie in [0, service_rate)")
    if slope < 0.0:
        raise ValueError("pi must be nonnegative")
    if kappa == 0.0:
        return 0.0

    lower_breakpoint = 1.0 / mu
    upper_breakpoint = mu / (mu - kappa) ** 2
    if slope <= lower_breakpoint:
        return 0.0
    if slope >= upper_breakpoint:
        return kappa / (mu - kappa) - slope * kappa
    return -((math.sqrt(mu * slope) - 1.0) ** 2)


def queue_support_minimizer(
    service_rate: float,
    arrival_capacity: float,
    pi: float,
) -> float:
    """Return the minimizing arrival rate in ``queue_perspective_intercept``."""

    mu = _finite(service_rate, "service_rate")
    kappa = _finite(arrival_capacity, "arrival_capacity")
    slope = _finite(pi, "pi")
    if mu <= 0.0:
        raise ValueError("service_rate must be positive")
    if kappa < 0.0 or kappa >= mu:
        raise ValueError("arrival_capacity must lie in [0, service_rate)")
    if slope < 0.0:
        raise ValueError("pi must be nonnegative")
    if kappa == 0.0 or slope <= 1.0 / mu:
        return 0.0
    if slope >= mu / (mu - kappa) ** 2:
        return kappa
    return mu - math.sqrt(mu / slope)


@dataclass(frozen=True)
class QueuePerspectiveCut:
    """Linear coefficients of one separated queue perspective inequality.

    The represented inequality is

    ``xi >= lambda_coefficient * lambda + sum(z_coefficients[c] * z[c])``.
    """

    pi: float
    lambda_coefficient: float
    z_coefficients: tuple[float, ...]
    support_arrivals: tuple[float, ...]
    rhs_at_point: float
    violation: float

    def signature(self, digits: int = 12) -> tuple[str, float]:
        """Return a stable signature suitable for external cut de-duplication."""

        return "queue_perspective", round(self.pi, digits)


def most_violated_queue_perspective_cut(
    lambda_value: float,
    xi_value: float,
    z_values: Sequence[float],
    service_rates: Sequence[float],
    max_system_time: float,
    *,
    feasibility_tolerance: float = 1e-9,
    bisection_tolerance: float = 1e-12,
    max_bisection_iterations: int = 120,
) -> QueuePerspectiveCut:
    r"""Maximize perspective-cut violation over :math:`\pi\ge0`.

    If ``zbar`` is the fractional discrete-capacity vector, the function
    maximizes the concave one-dimensional expression

    .. math::

        V(\pi)=\pi\bar\lambda+
        \sum_c\bar z_c b_c(\pi)-\bar\xi.

    By the envelope theorem,
    ``V'(pi) = lambda_bar - sum_c zbar_c * t_c(pi)``.  The minimizing arrivals
    ``t_c(pi)`` are monotone, so a safeguarded bisection finds a global
    maximizer.  If the supplied point violates
    ``lambda <= sum_c kappa_c z_c``, the maximization is unbounded and a
    ``ValueError`` is raised; the caller should first enforce that simpler
    linear QoS inequality.
    """

    if len(z_values) != len(service_rates) or not service_rates:
        raise ValueError(
            "z_values and service_rates must have the same positive length"
        )
    if feasibility_tolerance < 0.0 or bisection_tolerance <= 0.0:
        raise ValueError("tolerances must be positive (feasibility may be zero)")
    if max_bisection_iterations < 1:
        raise ValueError("max_bisection_iterations must be positive")

    lam = _nonnegative(lambda_value, "lambda_value", feasibility_tolerance)
    xi = _finite(xi_value, "xi_value")
    tau = _finite(max_system_time, "max_system_time")
    if tau <= 0.0:
        raise ValueError("max_system_time must be positive")
    mus = tuple(
        _finite(value, f"service_rates[{c}]") for c, value in enumerate(service_rates)
    )
    if any(mu <= 0.0 for mu in mus):
        raise ValueError("service rates must be positive")
    zbar = tuple(
        _unit_interval(value, f"z_values[{c}]", feasibility_tolerance)
        for c, value in enumerate(z_values)
    )
    if sum(zbar) > 1.0 + feasibility_tolerance:
        raise ValueError("discrete-capacity weights must sum to at most one")

    kappas = tuple(stable_arrival_capacity(mu, tau) for mu in mus)
    weighted_capacity = sum(weight * kappa for weight, kappa in zip(zbar, kappas))
    scale = max(1.0, lam, weighted_capacity)
    if lam > weighted_capacity + feasibility_tolerance * scale:
        raise ValueError(
            "lambda_value exceeds the fractional QoS capacity; perspective separation is unbounded"
        )

    if lam <= feasibility_tolerance:
        pi_star = 0.0
    else:
        upper = max(
            (mu / (mu - kappa) ** 2 for mu, kappa in zip(mus, kappas)),
            default=0.0,
        )

        def derivative(slope: float) -> float:
            return lam - sum(
                weight * queue_support_minimizer(mu, kappa, slope)
                for weight, mu, kappa in zip(zbar, mus, kappas)
            )

        high_derivative = derivative(upper)
        if high_derivative >= -feasibility_tolerance * scale:
            # Equality gives a flat tail after all support arrivals reach kappa.
            pi_star = upper
        else:
            low = 0.0
            high = upper
            for _ in range(max_bisection_iterations):
                midpoint = 0.5 * (low + high)
                if derivative(midpoint) > 0.0:
                    low = midpoint
                else:
                    high = midpoint
                if high - low <= bisection_tolerance * max(1.0, high):
                    break
            pi_star = 0.5 * (low + high)

    coefficients = tuple(
        queue_perspective_intercept(mu, kappa, pi_star)
        for mu, kappa in zip(mus, kappas)
    )
    arrivals = tuple(
        queue_support_minimizer(mu, kappa, pi_star) for mu, kappa in zip(mus, kappas)
    )
    rhs = pi_star * lam + sum(coef * weight for coef, weight in zip(coefficients, zbar))
    return QueuePerspectiveCut(
        pi=pi_star,
        lambda_coefficient=pi_star,
        z_coefficients=coefficients,
        support_arrivals=arrivals,
        rhs_at_point=rhs,
        violation=rhs - xi,
    )


def separate_queue_perspective_cut(
    lambda_value: float,
    xi_value: float,
    z_values: Sequence[float],
    service_rates: Sequence[float],
    max_system_time: float,
    *,
    violation_tolerance: float = 1e-6,
    **kwargs: object,
) -> QueuePerspectiveCut | None:
    """Return the globally most violated queue cut, or ``None`` if satisfied."""

    if violation_tolerance < 0.0:
        raise ValueError("violation_tolerance must be nonnegative")
    cut = most_violated_queue_perspective_cut(
        lambda_value,
        xi_value,
        z_values,
        service_rates,
        max_system_time,
        **kwargs,
    )
    return cut if cut.violation > violation_tolerance else None


# ---------------------------------------------------------------------------
# Choice-to-corridor coefficients and capacity covers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ChoiceCorridorCoverData:
    """Locker-only offered demand induced by one ``alpha[d,j]`` decision."""

    destination: Node
    locker: Node
    offered_demand: float
    market_contributions: tuple[tuple[Market, float], ...]
    incoming_corridors: tuple[Arc, ...]

    @property
    def alpha_key(self) -> AlphaKey:
        return self.destination, self.locker

    def signature(self) -> tuple[str, Node, Node]:
        return "choice_corridor_data", self.destination, self.locker


def build_choice_corridor_cover_data(
    h_od: Mapping[Market, float],
    choice_share: Mapping[ChoiceKey, float],
    display_origins: Mapping[ChoiceKey, Iterable[Node]],
    *,
    feasibility_tolerance: float = 1e-9,
) -> tuple[ChoiceCorridorCoverData, ...]:
    r"""Build :math:`B_{dj}=\sum_o h_{od}\eta_{odj}` and its corridors.

    ``display_origins[(o,d,j)]`` lists every candidate origin locker ``i`` to
    which the offer may be allocated.  Its union is retained as provenance for
    the structural condition that all corresponding corridors end at ``j``.
    The sum does *not* multiply by the number of candidate origins: allocation
    among those origins is endogenous and the locker-only allocation equality
    fixes their total to ``h_od[o,d] * choice_share[o,d,j] * alpha[d,j]``.
    """

    if feasibility_tolerance < 0.0:
        raise ValueError("feasibility_tolerance must be nonnegative")
    demand: dict[Market, float] = {}
    for market, value in h_od.items():
        if len(market) != 2:
            raise ValueError("every market key must be (origin, destination)")
        demand[market] = _nonnegative(value, f"h_od[{market}]", feasibility_tolerance)

    contributions: dict[AlphaKey, list[tuple[Market, float]]] = defaultdict(list)
    corridors: dict[AlphaKey, set[Arc]] = defaultdict(set)
    for key in sorted(choice_share):
        if len(key) != 3:
            raise ValueError("every choice key must be (origin, destination, locker)")
        origin, destination, locker = key
        market = (origin, destination)
        if market not in demand:
            raise ValueError(f"choice key {key} has no matching OD demand")
        share = _unit_interval(
            choice_share[key], f"choice_share[{key}]", feasibility_tolerance
        )
        if key not in display_origins:
            raise ValueError(f"choice key {key} has no candidate origin locker")
        origins = tuple(sorted(set(display_origins[key])))
        if not origins:
            raise ValueError(f"choice key {key} has an empty origin-locker set")
        alpha_key = (destination, locker)
        amount = demand[market] * share
        contributions[alpha_key].append((market, amount))
        corridors[alpha_key].update((i, locker) for i in origins)

    result = []
    for destination, locker in sorted(contributions):
        market_values = tuple(sorted(contributions[destination, locker]))
        result.append(
            ChoiceCorridorCoverData(
                destination=destination,
                locker=locker,
                offered_demand=sum(value for _, value in market_values),
                market_contributions=market_values,
                incoming_corridors=tuple(sorted(corridors[destination, locker])),
            )
        )
    return tuple(result)


@dataclass(frozen=True)
class ChoiceCapacityCover:
    """A lifted cover over display decisions and discrete capacity choices.

    The represented inequality is

    ``sum(alpha[d,j] for d in P)``
    ``<= len(P) - 1 + sum(z[j,c] for c in adequate_capacity_indices)``.
    """

    locker: Node
    destinations: tuple[Node, ...]
    offered_demand: float
    effective_arrival: float
    adequate_capacity_indices: tuple[int, ...]
    lhs_at_point: float
    rhs_at_point: float
    violation: float

    def signature(self) -> tuple[str, Node, tuple[Node, ...], tuple[int, ...]]:
        return (
            "choice_capacity_cover",
            self.locker,
            self.destinations,
            self.adequate_capacity_indices,
        )


def separate_choice_capacity_covers(
    data: Sequence[ChoiceCorridorCoverData],
    capacity_limits: Sequence[float],
    rho_min: float,
    alpha_values: Mapping[AlphaKey, float],
    z_values: Mapping[tuple[Node, int], float],
    *,
    violation_tolerance: float = 1e-6,
    feasibility_tolerance: float = 1e-9,
    max_cuts_per_locker: int | None = 5,
    seen_signatures: AbstractSet[object] | None = None,
) -> tuple[ChoiceCapacityCover, ...]:
    """Heuristically separate valid locker-only choice-capacity covers.

    For a destination subset ``P`` at locker ``j``, let

    ``W(P) = rho_min * sum(B[d,j] for d in P)``.

    Every capacity level with ``kappa[c] < W(P)`` is insufficient if all
    displays in ``P`` are selected.  The returned cover allows all of ``P``
    only when an adequate level is selected.  Candidate subsets are generated
    by several deficit/weight orderings and seeded greedy prefixes.  This is a
    separation heuristic, but every returned subset defines a globally valid
    inequality for the locker-only formulation.  Callers must not install these
    cuts in the mixed-delivery formulation because ``alpha[d,j]`` does not force
    positive offered flow there.
    """

    if violation_tolerance < 0.0 or feasibility_tolerance < 0.0:
        raise ValueError("tolerances must be nonnegative")
    if max_cuts_per_locker is not None and max_cuts_per_locker < 0:
        raise ValueError("max_cuts_per_locker must be nonnegative or None")
    rho = _finite(rho_min, "rho_min")
    if rho <= 0.0 or rho > 1.0:
        raise ValueError("rho_min must lie in (0, 1]")
    if not capacity_limits:
        raise ValueError("capacity_limits cannot be empty")
    capacities = tuple(
        _nonnegative(value, f"capacity_limits[{c}]", feasibility_tolerance)
        for c, value in enumerate(capacity_limits)
    )

    by_locker: dict[Node, list[ChoiceCorridorCoverData]] = defaultdict(list)
    alpha_bar: dict[AlphaKey, float] = {}
    for item in data:
        amount = _nonnegative(
            item.offered_demand,
            f"offered_demand[{item.destination},{item.locker}]",
            feasibility_tolerance,
        )
        if amount <= feasibility_tolerance:
            continue
        if any(corridor[1] != item.locker for corridor in item.incoming_corridors):
            raise ValueError(
                "choice-corridor data contains a corridor ending at another locker"
            )
        if item.alpha_key not in alpha_values:
            raise ValueError(f"missing alpha value for {item.alpha_key}")
        alpha_bar[item.alpha_key] = _unit_interval(
            alpha_values[item.alpha_key],
            f"alpha_values[{item.alpha_key}]",
            feasibility_tolerance,
        )
        by_locker[item.locker].append(item)

    z_bar: dict[tuple[Node, int], float] = {}
    for locker in by_locker:
        total = 0.0
        for c in range(len(capacities)):
            key = (locker, c)
            if key not in z_values:
                raise ValueError(f"missing z value for {key}")
            value = _unit_interval(
                z_values[key], f"z_values[{key}]", feasibility_tolerance
            )
            z_bar[key] = value
            total += value
        if total > 1.0 + feasibility_tolerance:
            raise ValueError(
                f"capacity weights for locker {locker} sum to more than one"
            )

    seen = set(seen_signatures or ())
    separated: list[ChoiceCapacityCover] = []
    for locker in sorted(by_locker):
        items = sorted(by_locker[locker], key=lambda item: item.destination)
        # Zero-alpha items cannot improve the cover violation by more than the
        # unit capacity mass they might exclude, so retaining only positive
        # fractional displays is both safe and much cheaper.
        candidates = [
            item for item in items if alpha_bar[item.alpha_key] > feasibility_tolerance
        ]
        if not candidates:
            continue

        def deficit(item: ChoiceCorridorCoverData) -> float:
            return 1.0 - alpha_bar[item.alpha_key]

        orderings = [
            sorted(
                candidates,
                key=lambda item: (
                    deficit(item) / (rho * item.offered_demand),
                    deficit(item),
                    -item.offered_demand,
                    item.destination,
                ),
            ),
            sorted(
                candidates,
                key=lambda item: (
                    deficit(item),
                    -item.offered_demand,
                    item.destination,
                ),
            ),
            sorted(
                candidates,
                key=lambda item: (
                    -item.offered_demand,
                    deficit(item),
                    item.destination,
                ),
            ),
        ]

        subset_keys: set[tuple[Node, ...]] = set()
        for ordering in orderings:
            prefix: list[Node] = []
            for item in ordering:
                prefix.append(item.destination)
                subset_keys.add(tuple(sorted(prefix)))

        # Seeded prefixes prevent one large item from masking a stronger cover
        # composed around a different high-alpha destination.
        primary_order = orderings[0]
        for seed in candidates:
            prefix = [seed.destination]
            subset_keys.add(tuple(prefix))
            for item in primary_order:
                if item.destination == seed.destination:
                    continue
                prefix.append(item.destination)
                subset_keys.add(tuple(sorted(prefix)))

        item_by_destination = {item.destination: item for item in candidates}
        locker_cuts: list[ChoiceCapacityCover] = []
        for destinations in sorted(subset_keys, key=lambda value: (len(value), value)):
            selected = [
                item_by_destination[destination] for destination in destinations
            ]
            offered = sum(item.offered_demand for item in selected)
            effective = rho * offered
            # With no insufficient level the inequality is dominated by the
            # existing alpha <= x and sum(z)=x relations.
            if all(
                capacity + feasibility_tolerance * max(1.0, effective) >= effective
                for capacity in capacities
            ):
                continue
            adequate = tuple(
                c
                for c, capacity in enumerate(capacities)
                if capacity + feasibility_tolerance * max(1.0, effective) >= effective
            )
            lhs = sum(alpha_bar[item.alpha_key] for item in selected)
            rhs = len(selected) - 1.0 + sum(z_bar[locker, c] for c in adequate)
            violation = lhs - rhs
            if violation <= violation_tolerance:
                continue
            cut = ChoiceCapacityCover(
                locker=locker,
                destinations=destinations,
                offered_demand=offered,
                effective_arrival=effective,
                adequate_capacity_indices=adequate,
                lhs_at_point=lhs,
                rhs_at_point=rhs,
                violation=violation,
            )
            if cut.signature() in seen:
                continue
            seen.add(cut.signature())
            locker_cuts.append(cut)

        locker_cuts.sort(
            key=lambda cut: (-cut.violation, len(cut.destinations), cut.destinations)
        )
        if max_cuts_per_locker is not None:
            locker_cuts = locker_cuts[:max_cuts_per_locker]
        separated.extend(locker_cuts)

    separated.sort(key=lambda cut: (-cut.violation, cut.locker, cut.destinations))
    return tuple(separated)


# ---------------------------------------------------------------------------
# Directed return-path cuts
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DirectedReturnCut:
    """A directed cut certifying the return path of an offered corridor.

    The represented inequality is

    ``offered[corridor] <= big_m * sum(y[arc] for arc in crossing_arcs)``.
    """

    corridor: Arc
    big_m: float
    source_side: tuple[Node, ...]
    crossing_arcs: tuple[Arc, ...]
    cut_capacity: float
    offered_value: float
    violation: float

    def signature(self) -> tuple[str, Arc, tuple[Arc, ...]]:
        return "directed_return", self.corridor, self.crossing_arcs


@dataclass(frozen=True)
class ODDisaggregatedDirectedReturnCut:
    r"""One OD-disaggregated directed return-connectivity cut.

    For ``od_corridor=(o,d,i,j)``, the represented inequality is

    ``sum_r chi[o,d,i,j,r] <= upper_bound * sum(y[a] for a in crossing_arcs)``.

    ``upper_bound`` must be a valid upper bound on the OD flow on corridor
    ``(i,j)``.  The caller uses ``min(h_od, M_offer[i,j])`` in HD/MX and
    ``min(h_od * p_odj, M_offer[i,j])`` in LO.
    """

    od_corridor: ODCorridorKey
    upper_bound: float
    source_side: tuple[Node, ...]
    crossing_arcs: tuple[Arc, ...]
    cut_capacity: float
    offered_value: float
    violation: float

    @property
    def market(self) -> Market:
        return self.od_corridor[0], self.od_corridor[1]

    @property
    def corridor(self) -> Arc:
        return self.od_corridor[2], self.od_corridor[3]

    def signature(
        self,
    ) -> tuple[str, ODCorridorKey, tuple[Arc, ...]]:
        return (
            "od_disaggregated_directed_return",
            self.od_corridor,
            self.crossing_arcs,
        )


@dataclass
class _ResidualEdge:
    destination: int
    reverse_index: int
    capacity: float


def _minimum_directed_cut(
    nodes: Sequence[Node],
    capacities: Mapping[Arc, float],
    source: Node,
    sink: Node,
    residual_tolerance: float,
) -> tuple[float, set[Node]]:
    """Compute a floating-point directed minimum cut with Dinic's algorithm."""

    if source == sink:
        raise ValueError("source and sink of a directed cut must differ")
    node_order = tuple(sorted(set(nodes) | {source, sink}))
    index = {node: position for position, node in enumerate(node_order)}
    graph: list[list[_ResidualEdge]] = [[] for _ in node_order]

    def add_edge(origin: Node, destination: Node, capacity: float) -> None:
        u = index[origin]
        v = index[destination]
        forward = _ResidualEdge(v, len(graph[v]), capacity)
        reverse = _ResidualEdge(u, len(graph[u]), 0.0)
        graph[u].append(forward)
        graph[v].append(reverse)

    for (origin, destination), capacity in sorted(capacities.items()):
        if origin != destination:
            add_edge(origin, destination, max(0.0, capacity))

    source_index = index[source]
    sink_index = index[sink]
    max_flow = 0.0
    while True:
        level = [-1] * len(node_order)
        level[source_index] = 0
        queue = deque([source_index])
        while queue:
            u = queue.popleft()
            for edge in graph[u]:
                if edge.capacity > residual_tolerance and level[edge.destination] < 0:
                    level[edge.destination] = level[u] + 1
                    queue.append(edge.destination)
        if level[sink_index] < 0:
            break

        next_edge = [0] * len(node_order)

        def send(u: int, available: float) -> float:
            if u == sink_index:
                return available
            while next_edge[u] < len(graph[u]):
                edge_index = next_edge[u]
                edge = graph[u][edge_index]
                if (
                    edge.capacity > residual_tolerance
                    and level[edge.destination] == level[u] + 1
                ):
                    amount = send(edge.destination, min(available, edge.capacity))
                    if amount > residual_tolerance:
                        edge.capacity -= amount
                        reverse = graph[edge.destination][edge.reverse_index]
                        reverse.capacity += amount
                        return amount
                next_edge[u] += 1
            return 0.0

        while True:
            amount = send(source_index, math.inf)
            if amount <= residual_tolerance:
                break
            max_flow += amount

    reachable_indices = {source_index}
    queue = deque([source_index])
    while queue:
        u = queue.popleft()
        for edge in graph[u]:
            if (
                edge.capacity > residual_tolerance
                and edge.destination not in reachable_indices
            ):
                reachable_indices.add(edge.destination)
                queue.append(edge.destination)
    return max_flow, {node_order[position] for position in reachable_indices}


def _contract_representative(node: Node, contraction: Mapping[Node, Node]) -> Node:
    representative = node
    visited: set[Node] = set()
    while (
        representative in contraction and contraction[representative] != representative
    ):
        if representative in visited:
            raise ValueError("contraction mapping contains a directed cycle")
        visited.add(representative)
        representative = contraction[representative]
    return representative


def separate_directed_return_cuts(
    physical_arcs: Iterable[Arc],
    y_values: Mapping[Arc, float],
    offered_values: Mapping[Arc, float],
    big_m: Mapping[Arc, float],
    *,
    contraction: Mapping[Node, Node] | None = None,
    violation_tolerance: float = 1e-6,
    feasibility_tolerance: float = 1e-9,
    max_candidates: int | None = None,
    max_cuts: int | None = None,
    seen_signatures: AbstractSet[object] | None = None,
) -> tuple[DirectedReturnCut, ...]:
    """Separate offered-flow return-connectivity cuts by directed minimum cuts.

    ``physical_arcs`` should be the aggregate-flight arcs ``A_F`` with their
    main activation values ``y``.  The optional mapping contracts a virtual
    state into its co-located physical locker, e.g. ``instance.s_to_h``.  Arcs
    becoming self-loops after contraction never cross a cut.

    At a fractional point, a minimum ``j``-to-``i`` cut of capacity ``C``
    violates the inequality exactly when
    ``offered_bar[i,j] > M[i,j] * C``.  The offered value may be supplied as
    ``sum(w[l,i,j,r])`` or the equivalent ``sum(chi[o,d,i,j,r])``; no auxiliary
    corridor-demand variable is required.  The implementation is pure Python
    and combines parallel arcs only for the max-flow calculation; every
    original crossing arc is retained in the returned linear expression.
    """

    if violation_tolerance < 0.0 or feasibility_tolerance < 0.0:
        raise ValueError("tolerances must be nonnegative")
    if max_candidates is not None and max_candidates < 0:
        raise ValueError("max_candidates must be nonnegative or None")
    if max_cuts is not None and max_cuts < 0:
        raise ValueError("max_cuts must be nonnegative or None")
    contraction_map = contraction or {}
    arcs = tuple(sorted(set(physical_arcs)))
    ybar: dict[Arc, float] = {}
    contracted_arcs: dict[Arc, Arc] = {}
    aggregate_capacity: dict[Arc, float] = defaultdict(float)
    nodes: set[Node] = set()
    for arc in arcs:
        if len(arc) != 2:
            raise ValueError("every physical arc must be an ordered node pair")
        if arc not in y_values:
            raise ValueError(f"missing y value for physical arc {arc}")
        value = _unit_interval(y_values[arc], f"y_values[{arc}]", feasibility_tolerance)
        ybar[arc] = value
        origin = _contract_representative(arc[0], contraction_map)
        destination = _contract_representative(arc[1], contraction_map)
        contracted_arcs[arc] = (origin, destination)
        nodes.update((origin, destination))
        if origin != destination:
            aggregate_capacity[origin, destination] += value

    candidates: list[tuple[float, Arc, float]] = []
    for corridor, raw_offered in offered_values.items():
        if corridor not in big_m:
            raise ValueError(f"missing big-M value for corridor {corridor}")
        offered_bar = _nonnegative(
            raw_offered,
            f"offered_values[{corridor}]",
            feasibility_tolerance,
        )
        upper = _finite(big_m[corridor], f"big_m[{corridor}]")
        if upper <= 0.0:
            raise ValueError("every corridor big-M value must be positive")
        if offered_bar <= violation_tolerance:
            continue
        candidates.append((offered_bar, corridor, upper))
    candidates.sort(key=lambda item: (-item[0], item[1]))
    if max_candidates is not None:
        candidates = candidates[:max_candidates]

    seen = set(seen_signatures or ())
    cuts: list[DirectedReturnCut] = []
    residual_tolerance = max(1e-14, feasibility_tolerance * 0.01)
    for offered_bar, corridor, upper in candidates:
        origin = _contract_representative(corridor[0], contraction_map)
        destination = _contract_representative(corridor[1], contraction_map)
        if origin == destination:
            continue
        nodes.update((origin, destination))
        _, source_side = _minimum_directed_cut(
            tuple(nodes),
            aggregate_capacity,
            destination,
            origin,
            residual_tolerance,
        )
        crossing = tuple(
            arc
            for arc in arcs
            if contracted_arcs[arc][0] in source_side
            and contracted_arcs[arc][1] not in source_side
        )
        # Recompute from original values so the reported violation and the
        # returned linear expression agree even after contraction/aggregation.
        cut_capacity = sum(ybar[arc] for arc in crossing)
        violation = offered_bar - upper * cut_capacity
        if violation <= violation_tolerance:
            continue
        cut = DirectedReturnCut(
            corridor=corridor,
            big_m=upper,
            source_side=tuple(sorted(source_side)),
            crossing_arcs=crossing,
            cut_capacity=cut_capacity,
            offered_value=offered_bar,
            violation=violation,
        )
        if cut.signature() in seen:
            continue
        seen.add(cut.signature())
        cuts.append(cut)

    cuts.sort(key=lambda cut: (-cut.violation, cut.corridor, cut.crossing_arcs))
    if max_cuts is not None:
        cuts = cuts[:max_cuts]
    return tuple(cuts)


def separate_od_disaggregated_directed_return_cuts(
    physical_arcs: Iterable[Arc],
    y_values: Mapping[Arc, float],
    offered_values: Mapping[ODCorridorKey, float],
    upper_bounds: Mapping[ODCorridorKey, float],
    *,
    contraction: Mapping[Node, Node] | None = None,
    violation_tolerance: float = 1e-6,
    feasibility_tolerance: float = 1e-9,
    max_corridors: int | None = None,
    max_cuts: int | None = None,
    seen_signatures: AbstractSet[object] | None = None,
) -> tuple[ODDisaggregatedDirectedReturnCut, ...]:
    r"""Separate OD-disaggregated directed return-connectivity cuts.

    For every retained ``(o,d,i,j)`` group the valid inequality is

    ``sum_r chi[o,d,i,j,r] <= U[o,d,i,j] * sum(a in delta+(S)) y[a]``,

    where ``S`` is the source side of a directed ``j``-to-``i`` cut and
    ``U`` is a supplied valid OD/corridor flow upper bound.  All positive OD
    groups on the same corridor share the *same* minimum-cut calculation.  The
    optional ``max_corridors`` budget therefore limits expensive max-flow
    calls, not cheap OD-group checks.

    The caller is responsible for constructing ``U``.  The MLND callback uses
    ``min(h_od, M_offer[i,j])`` in HD/MX and
    ``min(h_od * p_odj, M_offer[i,j])`` in LO.  Nonfinite or malformed inputs
    are rejected before any cut is returned.
    """

    violation_tol = _nonnegative(
        violation_tolerance,
        "violation_tolerance",
    )
    feasibility_tol = _nonnegative(
        feasibility_tolerance,
        "feasibility_tolerance",
    )
    for value, name in (
        (max_corridors, "max_corridors"),
        (max_cuts, "max_cuts"),
    ):
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int) or value < 0
        ):
            raise ValueError(f"{name} must be a nonnegative integer or None")

    contraction_map = contraction or {}
    arcs = tuple(sorted(set(physical_arcs)))
    ybar: dict[Arc, float] = {}
    contracted_arcs: dict[Arc, Arc] = {}
    aggregate_capacity: dict[Arc, float] = defaultdict(float)
    nodes: set[Node] = set()
    for arc in arcs:
        if not isinstance(arc, tuple) or len(arc) != 2:
            raise ValueError("every physical arc must be an ordered node pair")
        if arc not in y_values:
            raise ValueError(f"missing y value for physical arc {arc}")
        value = _unit_interval(
            y_values[arc],
            f"y_values[{arc}]",
            feasibility_tol,
        )
        ybar[arc] = value
        origin = _contract_representative(arc[0], contraction_map)
        destination = _contract_representative(arc[1], contraction_map)
        contracted_arcs[arc] = (origin, destination)
        nodes.update((origin, destination))
        if origin != destination:
            aggregate_capacity[origin, destination] += value

    candidates_by_corridor: dict[
        Arc,
        list[tuple[ODCorridorKey, float, float]],
    ] = defaultdict(list)
    for raw_key, raw_offered in offered_values.items():
        if not isinstance(raw_key, tuple) or len(raw_key) != 4:
            raise ValueError(
                "every offered-value key must be an (origin,destination,i,j) tuple"
            )
        od_corridor: ODCorridorKey = raw_key
        if od_corridor not in upper_bounds:
            raise ValueError(f"missing upper bound for OD corridor {od_corridor}")
        offered_bar = _nonnegative(
            raw_offered,
            f"offered_values[{od_corridor}]",
            feasibility_tol,
        )
        upper = _finite(
            upper_bounds[od_corridor],
            f"upper_bounds[{od_corridor}]",
        )
        if upper <= 0.0:
            raise ValueError("every OD/corridor upper bound must be positive")
        if offered_bar <= violation_tol:
            continue
        corridor = od_corridor[2], od_corridor[3]
        origin = _contract_representative(corridor[0], contraction_map)
        destination = _contract_representative(corridor[1], contraction_map)
        nodes.update((origin, destination))
        if origin == destination:
            continue
        candidates_by_corridor[corridor].append(
            (od_corridor, offered_bar, upper)
        )

    ranked_corridors = sorted(
        candidates_by_corridor,
        key=lambda corridor: (
            -sum(
                offered
                for _, offered, _ in candidates_by_corridor[corridor]
            ),
            corridor,
        ),
    )
    if max_corridors is not None:
        ranked_corridors = ranked_corridors[:max_corridors]

    # One cache entry equals one max-flow call.  Every OD group on a corridor
    # reuses the same source side, crossing arcs, and cut capacity.
    corridor_cut_cache: dict[Arc, tuple[tuple[Node, ...], tuple[Arc, ...], float]] = {}
    residual_tolerance = max(1e-14, feasibility_tol * 0.01)
    for corridor in ranked_corridors:
        origin = _contract_representative(corridor[0], contraction_map)
        destination = _contract_representative(corridor[1], contraction_map)
        _, raw_source_side = _minimum_directed_cut(
            tuple(nodes),
            aggregate_capacity,
            destination,
            origin,
            residual_tolerance,
        )
        source_side = tuple(sorted(raw_source_side))
        crossing = tuple(
            arc
            for arc in arcs
            if contracted_arcs[arc][0] in raw_source_side
            and contracted_arcs[arc][1] not in raw_source_side
        )
        cut_capacity = sum(ybar[arc] for arc in crossing)
        corridor_cut_cache[corridor] = source_side, crossing, cut_capacity

    seen = set(seen_signatures or ())
    cuts: list[ODDisaggregatedDirectedReturnCut] = []
    for corridor in ranked_corridors:
        source_side, crossing, cut_capacity = corridor_cut_cache[corridor]
        for od_corridor, offered_bar, upper in candidates_by_corridor[corridor]:
            violation = offered_bar - upper * cut_capacity
            if violation <= violation_tol:
                continue
            cut = ODDisaggregatedDirectedReturnCut(
                od_corridor=od_corridor,
                upper_bound=upper,
                source_side=source_side,
                crossing_arcs=crossing,
                cut_capacity=cut_capacity,
                offered_value=offered_bar,
                violation=violation,
            )
            if cut.signature() in seen:
                continue
            seen.add(cut.signature())
            cuts.append(cut)

    cuts.sort(
        key=lambda cut: (
            -cut.violation,
            cut.od_corridor,
            cut.crossing_arcs,
        )
    )
    if max_cuts is not None:
        cuts = cuts[:max_cuts]
    return tuple(cuts)


__all__ = [
    "AlphaKey",
    "Arc",
    "ChoiceCapacityCover",
    "ChoiceCorridorCoverData",
    "DirectedReturnCut",
    "ODCorridorKey",
    "ODDisaggregatedDirectedReturnCut",
    "NearestDisplaySuffixRow",
    "QueuePerspectiveCut",
    "RouteAggregateVUBBound",
    "RouteBundleViolation",
    "RouteOptionKey",
    "build_choice_corridor_cover_data",
    "build_nearest_display_suffix_rows",
    "build_swap_route_bundles",
    "evaluate_route_bundle_vub",
    "most_violated_queue_perspective_cut",
    "queue_perspective_intercept",
    "queue_support_minimizer",
    "separate_choice_capacity_covers",
    "separate_directed_return_cuts",
    "separate_od_disaggregated_directed_return_cuts",
    "separate_queue_perspective_cut",
    "stable_arrival_capacity",
    "route_swap_leg_vub_bound",
    "route_swap_station_vub_bound",
]
