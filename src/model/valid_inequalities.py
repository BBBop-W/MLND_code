"""Solver-independent valid inequalities for the OD-aware MLND model.

The routines in this module only separate algebraic inequalities.  They do not
import Gurobi, mutate an optimization model, or retain separation state.  This
makes the formulas independently testable before a caller translates the
returned coefficients into ordinary constraints or callback cuts.

The module contains three families.

``QueuePerspectiveCut``
    Supporting hyperplanes for the union of the discrete M/M/1 queue graphs.

``ChoiceCapacityCover``
    Covers linking destination-locker display decisions to a locker's selected
    service capacity.  Their validity uses the MLND relation
    ``psi[i,j] >= rho_min * h[i,j]`` and the fact that every locker-choice offer
    for locker ``j`` enters a corridor ending at ``j``.

``DirectedReturnCut``
    Directed connectivity cuts.  Positive offered flow on corridor ``(i,j)``
    requires a return path from ``j`` to ``i`` in every integral balanced
    network.  Co-located virtual holding states may be contracted into their
    physical lockers before the minimum-cut calculation.
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
    return -(math.sqrt(mu * slope) - 1.0) ** 2


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
        raise ValueError("z_values and service_rates must have the same positive length")
    if feasibility_tolerance < 0.0 or bisection_tolerance <= 0.0:
        raise ValueError("tolerances must be positive (feasibility may be zero)")
    if max_bisection_iterations < 1:
        raise ValueError("max_bisection_iterations must be positive")

    lam = _nonnegative(lambda_value, "lambda_value", feasibility_tolerance)
    xi = _finite(xi_value, "xi_value")
    tau = _finite(max_system_time, "max_system_time")
    if tau <= 0.0:
        raise ValueError("max_system_time must be positive")
    mus = tuple(_finite(value, f"service_rates[{c}]") for c, value in enumerate(service_rates))
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
        queue_support_minimizer(mu, kappa, pi_star)
        for mu, kappa in zip(mus, kappas)
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
    """Deterministic offered demand induced by one ``alpha[d,j]`` decision."""

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
    locker_origins: Mapping[ChoiceKey, Iterable[Node]],
    *,
    feasibility_tolerance: float = 1e-9,
) -> tuple[ChoiceCorridorCoverData, ...]:
    r"""Build :math:`B_{dj}=\sum_o h_{od}\eta_{odj}` and its corridors.

    ``locker_origins[(o,d,j)]`` lists every candidate origin locker ``i`` to
    which the offer may be allocated.  Its union is retained as provenance for
    the structural condition that all corresponding corridors end at ``j``.
    The sum does *not* multiply by the number of candidate origins: allocation
    among those origins is endogenous and the model's locker-offer equality
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
        if key not in locker_origins:
            raise ValueError(f"choice key {key} has no candidate origin locker")
        origins = tuple(sorted(set(locker_origins[key])))
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
    """Heuristically separate valid choice-capacity covers.

    For a destination subset ``P`` at locker ``j``, let

    ``W(P) = rho_min * sum(B[d,j] for d in P)``.

    Every capacity level with ``kappa[c] < W(P)`` is insufficient if all
    displays in ``P`` are selected.  The returned cover allows all of ``P``
    only when an adequate level is selected.  Candidate subsets are generated
    by several deficit/weight orderings and seeded greedy prefixes.  This is a
    separation heuristic, but every returned subset defines a globally valid
    inequality.
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
            raise ValueError("choice-corridor data contains a corridor ending at another locker")
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
            value = _unit_interval(z_values[key], f"z_values[{key}]", feasibility_tolerance)
            z_bar[key] = value
            total += value
        if total > 1.0 + feasibility_tolerance:
            raise ValueError(f"capacity weights for locker {locker} sum to more than one")

    seen = set(seen_signatures or ())
    separated: list[ChoiceCapacityCover] = []
    for locker in sorted(by_locker):
        items = sorted(by_locker[locker], key=lambda item: item.destination)
        # Zero-alpha items cannot improve the cover violation by more than the
        # unit capacity mass they might exclude, so retaining only positive
        # fractional displays is both safe and much cheaper.
        candidates = [
            item
            for item in items
            if alpha_bar[item.alpha_key] > feasibility_tolerance
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
                key=lambda item: (deficit(item), -item.offered_demand, item.destination),
            ),
            sorted(
                candidates,
                key=lambda item: (-item.offered_demand, deficit(item), item.destination),
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
            selected = [item_by_destination[destination] for destination in destinations]
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

    ``h[corridor] <= big_m * sum(y[arc] for arc in crossing_arcs)``.
    """

    corridor: Arc
    big_m: float
    source_side: tuple[Node, ...]
    crossing_arcs: tuple[Arc, ...]
    cut_capacity: float
    h_value: float
    violation: float

    def signature(self) -> tuple[str, Arc, tuple[Arc, ...]]:
        return "directed_return", self.corridor, self.crossing_arcs


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
                if edge.capacity > residual_tolerance and level[edge.destination] == level[u] + 1:
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
            if edge.capacity > residual_tolerance and edge.destination not in reachable_indices:
                reachable_indices.add(edge.destination)
                queue.append(edge.destination)
    return max_flow, {node_order[position] for position in reachable_indices}


def _contract_representative(node: Node, contraction: Mapping[Node, Node]) -> Node:
    representative = node
    visited: set[Node] = set()
    while representative in contraction and contraction[representative] != representative:
        if representative in visited:
            raise ValueError("contraction mapping contains a directed cycle")
        visited.add(representative)
        representative = contraction[representative]
    return representative


def separate_directed_return_cuts(
    physical_arcs: Iterable[Arc],
    y_values: Mapping[Arc, float],
    h_values: Mapping[Arc, float],
    big_m: Mapping[Arc, float],
    *,
    contraction: Mapping[Node, Node] | None = None,
    violation_tolerance: float = 1e-6,
    feasibility_tolerance: float = 1e-9,
    max_candidates: int | None = None,
    max_cuts: int | None = None,
    seen_signatures: AbstractSet[object] | None = None,
) -> tuple[DirectedReturnCut, ...]:
    """Separate ``h_ij`` return-connectivity cuts by directed minimum cuts.

    ``physical_arcs`` should be the aggregate-flight arcs ``A_F`` with their
    main activation values ``y``.  The optional mapping contracts a virtual
    state into its co-located physical locker, e.g. ``instance.s_to_h``.  Arcs
    becoming self-loops after contraction never cross a cut.

    At a fractional point, a minimum ``j``-to-``i`` cut of capacity ``C``
    violates the inequality exactly when ``hbar[i,j] > M[i,j] * C``.  The
    implementation is pure Python and combines parallel arcs only for the
    max-flow calculation; every original crossing arc is retained in the
    returned linear expression.
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
    for corridor, raw_h in h_values.items():
        if corridor not in big_m:
            raise ValueError(f"missing big-M value for corridor {corridor}")
        hbar = _nonnegative(raw_h, f"h_values[{corridor}]", feasibility_tolerance)
        upper = _finite(big_m[corridor], f"big_m[{corridor}]")
        if upper <= 0.0:
            raise ValueError("every corridor big-M value must be positive")
        if hbar <= violation_tolerance:
            continue
        candidates.append((hbar, corridor, upper))
    candidates.sort(key=lambda item: (-item[0], item[1]))
    if max_candidates is not None:
        candidates = candidates[:max_candidates]

    seen = set(seen_signatures or ())
    cuts: list[DirectedReturnCut] = []
    residual_tolerance = max(1e-14, feasibility_tolerance * 0.01)
    for hbar, corridor, upper in candidates:
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
        violation = hbar - upper * cut_capacity
        if violation <= violation_tolerance:
            continue
        cut = DirectedReturnCut(
            corridor=corridor,
            big_m=upper,
            source_side=tuple(sorted(source_side)),
            crossing_arcs=crossing,
            cut_capacity=cut_capacity,
            h_value=hbar,
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


__all__ = [
    "AlphaKey",
    "Arc",
    "ChoiceCapacityCover",
    "ChoiceCorridorCoverData",
    "DirectedReturnCut",
    "QueuePerspectiveCut",
    "build_choice_corridor_cover_data",
    "most_violated_queue_perspective_cut",
    "queue_perspective_intercept",
    "queue_support_minimizer",
    "separate_choice_capacity_covers",
    "separate_directed_return_cuts",
    "separate_queue_perspective_cut",
    "stable_arrival_capacity",
]
