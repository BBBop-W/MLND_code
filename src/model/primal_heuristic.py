"""Exact-conic local branching for primal MIP-start improvement.

The heuristic changes only the binary design neighbourhood.  All continuous
allocation, flow, fleet, and queueing variables, including the original
second-order-cone constraints, remain in the subproblem.  Consequently, a
solution returned by this module is feasible for the original MISOCP after the
temporary local-branching and objective-cutoff constraints are removed.

The intended integration with :class:`src.model.gurobi_model.MLNDModel` is::

    seed = capture_incumbent(mlnd.model, mlnd.vars)
    result = improve_mip_start(mlnd.model, mlnd.vars, seed)

``improve_mip_start`` installs the best complete solution as a MIP start.  It
does not leave a neighbourhood constraint, a changed variable bound, or a
changed search parameter in the final model, even if optimization raises.
"""

from __future__ import annotations

import math
import time
from array import array
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
from typing import Any, Hashable

import gurobipy as gp
from gurobipy import GRB


DEFAULT_STRUCTURE_GROUPS = ("x", "z", "delta", "alpha")
OBJECTIVE_START_PROXY_QOS_ABS_TOLERANCE = 1e-9
OBJECTIVE_START_PROXY_ENERGY_REL_TOLERANCE = 1e-9
OBJECTIVE_START_PROXY_SEARCH_REL_TOLERANCE = 1e-12

MarketKey = tuple[str, str]
ArcKey = tuple[str, str]
SupportKey = tuple[str, str, int]


@dataclass(frozen=True)
class ApproximateObjectiveBreakdown:
    """Auditable objective proxy for one partial network support.

    The direct-courier term is the cost avoided by the estimated successful
    service.  Every other field is a cost of that service.  The proxy is used
    only to rank MIP starts; it is never added to the mathematical model.
    """

    direct_courier_baseline: float = 0.0
    first_mile_rider: float = 0.0
    last_mile_rider: float = 0.0
    mx_locker_pickup: float = 0.0
    flight: float = 0.0
    facility: float = 0.0
    capacity: float = 0.0
    connection: float = 0.0
    queue: float = 0.0
    fleet: float = 0.0

    def validate(self) -> None:
        values = (
            self.direct_courier_baseline,
            self.first_mile_rider,
            self.last_mile_rider,
            self.mx_locker_pickup,
            self.flight,
            self.facility,
            self.capacity,
            self.connection,
            self.queue,
            self.fleet,
        )
        if any(not math.isfinite(value) or value < 0.0 for value in values):
            raise ValueError(
                "approximate objective terms must be finite and nonnegative"
            )

    @property
    def network_cost(self) -> float:
        return math.fsum(
            (
                self.first_mile_rider,
                self.last_mile_rider,
                self.mx_locker_pickup,
                self.flight,
                self.facility,
                self.capacity,
                self.connection,
                self.queue,
                self.fleet,
            )
        )

    @property
    def net_savings(self) -> float:
        return self.direct_courier_baseline - self.network_cost


@dataclass(frozen=True)
class CapacityOption:
    """One locker service-rate choice used by the objective proxy."""

    index: int
    service_rate: float
    fixed_cost: float

    def validate(self) -> None:
        if (
            isinstance(self.index, bool)
            or not isinstance(self.index, int)
            or self.index < 0
        ):
            raise ValueError("capacity index must be a nonnegative integer")
        if not math.isfinite(self.service_rate) or self.service_rate <= 0.0:
            raise ValueError("capacity service_rate must be positive and finite")
        if not math.isfinite(self.fixed_cost) or self.fixed_cost < 0.0:
            raise ValueError("capacity fixed_cost must be finite and nonnegative")


@dataclass(frozen=True)
class ObjectiveAwareOpportunity:
    """One market/corridor/reliability contribution to a cheap start proxy."""

    market: MarketKey
    corridor: ArcKey
    reliability_index: int
    reliability: float
    offered_demand: float
    support_facilities: tuple[str, ...]
    open_h: tuple[str, ...]
    active_connections: tuple[ArcKey, ...]
    facility_costs: tuple[tuple[str, float], ...]
    connection_costs: tuple[tuple[ArcKey, float], ...]
    idle_state: str
    incoming_h: tuple[str, ...]
    outgoing_route_time_hours: tuple[tuple[str, float], ...]
    display_locker: str | None
    display_blockers: tuple[str, ...]
    direct_unit_cost: float
    first_mile_unit_cost: float
    last_mile_unit_cost: float
    mx_pickup_unit_cost: float
    flight_unit_cost: float
    route_time_hours: float
    # Exact coefficients of the f884 system-wide energy pool per unit of
    # successful service represented by this cycle.  A physical-base
    # departure contributes one full usable battery even when its route option
    # is DIRECT; every non-DIRECT route option contributes one additional swap
    # event.  These fields deliberately do not allocate energy by facility.
    base_departures_per_unit: float = 1.0
    swap_events_per_unit: float = 0.0

    @property
    def support_key(self) -> SupportKey:
        return self.corridor[0], self.corridor[1], self.reliability_index

    @property
    def successful_service(self) -> float:
        return self.reliability * self.offered_demand

    def variable_net_savings(self, drone_cost_per_hour: float) -> float:
        unit_network_cost = math.fsum(
            (
                self.first_mile_unit_cost,
                self.last_mile_unit_cost,
                self.mx_pickup_unit_cost,
                self.flight_unit_cost,
                drone_cost_per_hour * self.route_time_hours,
            )
        )
        return self.successful_service * (self.direct_unit_cost - unit_network_cost)

    def validate(self) -> None:
        if len(self.market) != 2 or any(not value for value in self.market):
            raise ValueError("opportunity market must contain two nonempty node ids")
        if len(self.corridor) != 2 or self.corridor[0] == self.corridor[1]:
            raise ValueError("opportunity corridor must have two distinct endpoints")
        if (
            isinstance(self.reliability_index, bool)
            or not isinstance(self.reliability_index, int)
            or self.reliability_index < 0
        ):
            raise ValueError("reliability_index must be a nonnegative integer")
        if not math.isfinite(self.reliability) or not 0.0 < self.reliability < 1.0:
            raise ValueError("reliability must lie strictly between zero and one")
        nonnegative = (
            self.offered_demand,
            self.direct_unit_cost,
            self.first_mile_unit_cost,
            self.last_mile_unit_cost,
            self.mx_pickup_unit_cost,
            self.flight_unit_cost,
            self.route_time_hours,
            self.base_departures_per_unit,
            self.swap_events_per_unit,
            *(route_time for _, route_time in self.outgoing_route_time_hours),
            *(cost for _, cost in self.facility_costs),
            *(cost for _, cost in self.connection_costs),
        )
        if any(not math.isfinite(value) or value < 0.0 for value in nonnegative):
            raise ValueError(
                "opportunity quantities and costs must be finite and nonnegative"
            )
        if self.base_departures_per_unit <= 0.0:
            raise ValueError("an opportunity cycle must contain a base departure")
        if (
            not self.support_facilities
            or not self.open_h
            or not self.active_connections
        ):
            raise ValueError(
                "an opportunity must have nonempty facility and connection support"
            )
        if not self.idle_state:
            raise ValueError("an opportunity must identify its source idle state")
        if len(self.incoming_h) != len(set(self.incoming_h)):
            raise ValueError("incoming_h must contain unique lockers")
        if set(self.incoming_h) != set(self.open_h):
            raise ValueError("every open H in the cycle must receive incoming flow")
        outgoing_lockers = tuple(locker for locker, _ in self.outgoing_route_time_hours)
        if len(outgoing_lockers) != len(set(outgoing_lockers)):
            raise ValueError("outgoing route times must identify each locker once")
        if set(outgoing_lockers) != set(self.open_h):
            raise ValueError("every open H in the cycle must have outgoing route time")
        if self.display_locker is not None:
            if not self.display_locker:
                raise ValueError("display_locker must be nonempty or None")
            if self.display_locker not in self.open_h:
                raise ValueError("display_locker must be an open H")
            if self.display_locker in self.display_blockers:
                raise ValueError("display locker cannot block itself")
        elif self.display_blockers:
            raise ValueError("implicit display opportunities cannot have blockers")
        if len(self.display_blockers) != len(set(self.display_blockers)):
            raise ValueError("display_blockers must be unique")


@dataclass(frozen=True)
class ObjectiveAwarePlanningConfig:
    """Solver-free controls for deterministic partial-start construction."""

    candidate_lockers: tuple[str, ...]
    capacity_options: tuple[CapacityOption, ...]
    max_locker_time_hours: float
    usable_battery_time_hours: float
    drone_cost_per_hour: float
    max_candidates: int = 3
    min_open_h: int = 2
    min_service: float = 0.0
    min_savings: float = 0.0
    improvement_tolerance: float = 1e-9

    def validate(self) -> None:
        if len(self.candidate_lockers) != len(set(self.candidate_lockers)):
            raise ValueError("candidate_lockers must be unique")
        if not self.capacity_options:
            raise ValueError("at least one capacity option is required")
        for option in self.capacity_options:
            option.validate()
        if len({option.index for option in self.capacity_options}) != len(
            self.capacity_options
        ):
            raise ValueError("capacity option indices must be unique")
        if (
            not math.isfinite(self.max_locker_time_hours)
            or self.max_locker_time_hours <= 0.0
        ):
            raise ValueError("max_locker_time_hours must be positive and finite")
        if (
            not math.isfinite(self.usable_battery_time_hours)
            or self.usable_battery_time_hours <= 0.0
        ):
            raise ValueError("usable_battery_time_hours must be positive and finite")
        if (
            not math.isfinite(self.drone_cost_per_hour)
            or self.drone_cost_per_hour <= 0.0
        ):
            raise ValueError("drone_cost_per_hour must be positive and finite")
        if (
            isinstance(self.max_candidates, bool)
            or not isinstance(self.max_candidates, int)
            or not 1 <= self.max_candidates <= 16
        ):
            raise ValueError("max_candidates must be an integer in [1, 16]")
        if (
            isinstance(self.min_open_h, bool)
            or not isinstance(self.min_open_h, int)
            or self.min_open_h < 1
        ):
            raise ValueError("min_open_h must be a positive integer")
        thresholds = (self.min_service, self.min_savings, self.improvement_tolerance)
        if any(not math.isfinite(value) or value < 0.0 for value in thresholds):
            raise ValueError("planning thresholds must be finite and nonnegative")


@dataclass(frozen=True)
class ObjectiveAwareSupportCandidate:
    """One gated network support to install as a partial MIP start."""

    candidate_id: str
    move: str
    selected_supports: tuple[SupportKey, ...]
    open_h: tuple[str, ...]
    open_facilities: tuple[str, ...]
    active_connections: tuple[ArcKey, ...]
    capacity_levels: tuple[tuple[str, int], ...]
    forced_closed_h: tuple[str, ...]
    forced_off_connections: tuple[ArcKey, ...]
    estimated_service: float
    objective: ApproximateObjectiveBreakdown
    service_fraction: float = 1.0
    system_energy_use: float = 0.0
    system_energy_supply: float = 0.0
    base_departure_population: float = 0.0
    swap_event_population: float = 0.0

    @property
    def score(self) -> float:
        return self.objective.net_savings


@dataclass(frozen=True)
class ObjectiveAwareStartPlan:
    """Outcome and audit trail of bounded, solver-free start preparation."""

    budget_seconds: float
    elapsed_seconds: float
    opportunity_count: int
    singleton_support_count: int
    evaluated_support_count: int
    candidates: tuple[ObjectiveAwareSupportCandidate, ...]
    rejection_reasons: tuple[str, ...]
    timed_out: bool


class _ObjectiveStartTimeout(RuntimeError):
    pass


def effective_objective_start_budget(
    total_time_limit: float,
    requested_seconds: float | None = None,
) -> float:
    """Return a hard preparation budget capped at five percent of the solve."""

    if not math.isfinite(total_time_limit) or total_time_limit <= 0.0:
        raise ValueError("total_time_limit must be positive and finite")
    cap = 0.05 * total_time_limit
    if requested_seconds is None:
        return cap
    if not math.isfinite(requested_seconds) or requested_seconds < 0.0:
        raise ValueError("requested start seconds must be finite and nonnegative")
    return min(requested_seconds, cap)


def _objective_start_check_timeout(
    should_stop: Callable[[], bool] | None,
) -> None:
    if should_stop is not None and should_stop():
        raise _ObjectiveStartTimeout


def evaluate_objective_support(
    opportunities: Sequence[ObjectiveAwareOpportunity],
    selected_supports: Sequence[SupportKey],
    config: ObjectiveAwarePlanningConfig,
    *,
    should_stop: Callable[[], bool] | None = None,
    opportunity_index: Mapping[SupportKey, Sequence[ObjectiveAwareOpportunity]]
    | None = None,
) -> tuple[ObjectiveAwareSupportCandidate | None, str | None]:
    """Evaluate a support with a full but deliberately approximate objective.

    Each OD market is credited at most once, using its best positive variable
    saving among the selected corridor/reliability variants.  Facility and
    connection costs are charged on their unions.  The chosen opportunities
    share a deterministic service fraction in ``(0, 1]``: one means their full
    successful demand, while a smaller value is the largest common fraction
    allowed by locker QoS capacity and the system-wide energy pool.  This is a
    solver-free scalar search because, for fixed service rates,
    ``lambda*A / (mu - lambda*A) / lambda`` is increasing in ``lambda``.
    Locker capacity is then chosen by capacity cost plus the implied M/M/1
    queue-population cost.  Total queue, travel, and idle population is rounded
    up to the exact integer fleet scale.
    """

    config.validate()
    selected = tuple(sorted(set(selected_supports)))
    if not selected:
        return None, "empty_support"
    selected_set = set(selected)
    retained: list[ObjectiveAwareOpportunity] = []
    if opportunity_index is None:
        index: dict[SupportKey, list[ObjectiveAwareOpportunity]] = {}
        for opportunity in opportunities:
            _objective_start_check_timeout(should_stop)
            opportunity.validate()
            index.setdefault(opportunity.support_key, []).append(opportunity)
        opportunity_index = index
    for support in selected:
        _objective_start_check_timeout(should_stop)
        retained.extend(opportunity_index.get(support, ()))
    present = {opportunity.support_key for opportunity in retained}
    if present != selected_set:
        return None, "missing_support_data"

    by_support: dict[SupportKey, list[ObjectiveAwareOpportunity]] = {}
    by_market: dict[MarketKey, list[ObjectiveAwareOpportunity]] = {}
    for opportunity in retained:
        by_support.setdefault(opportunity.support_key, []).append(opportunity)
        by_market.setdefault(opportunity.market, []).append(opportunity)

    facility_costs: dict[str, float] = {}
    connection_costs: dict[ArcKey, float] = {}
    open_h: set[str] = set()
    open_facilities: set[str] = set()
    active_connections: set[ArcKey] = set()
    reliability_by_idle: dict[str, tuple[int, float]] = {}
    for support in selected:
        _objective_start_check_timeout(should_stop)
        representative = min(
            by_support[support],
            key=lambda item: (item.market, item.corridor, item.reliability_index),
        )
        open_h.update(representative.open_h)
        open_facilities.update(representative.support_facilities)
        active_connections.update(representative.active_connections)
        previous_reliability = reliability_by_idle.setdefault(
            representative.idle_state,
            (representative.reliability_index, representative.reliability),
        )
        if previous_reliability[0] != representative.reliability_index:
            return None, "inconsistent_idle_reliability"
        if not math.isclose(
            previous_reliability[1],
            representative.reliability,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            return None, "inconsistent_idle_reliability_value"
        for node, cost in representative.facility_costs:
            previous = facility_costs.setdefault(node, cost)
            if not math.isclose(previous, cost, rel_tol=0.0, abs_tol=1e-9):
                return None, "inconsistent_facility_cost"
        for arc, cost in representative.connection_costs:
            previous = connection_costs.setdefault(arc, cost)
            if not math.isclose(previous, cost, rel_tol=0.0, abs_tol=1e-9):
                return None, "inconsistent_connection_cost"

    display_by_destination: dict[str, set[str]] = {}
    destinations_with_implicit_display: set[str] = set()
    for opportunity in retained:
        destination = opportunity.market[1]
        if opportunity.display_locker is None:
            destinations_with_implicit_display.add(destination)
        else:
            display_by_destination.setdefault(destination, set()).add(
                opportunity.display_locker
            )
    ambiguous_display = destinations_with_implicit_display & set(display_by_destination)
    if ambiguous_display:
        return None, "inconsistent_display_semantics"

    def display_is_unblocked(opportunity: ObjectiveAwareOpportunity) -> bool:
        return not any(blocker in open_h for blocker in opportunity.display_blockers)

    selected_display: dict[str, str | None] = {}
    for destination in sorted(display_by_destination):
        _objective_start_check_timeout(should_stop)
        candidates: list[tuple[float, str]] = []
        for locker in sorted(display_by_destination[destination]):
            feasible_opportunities = [
                opportunity
                for market in by_market
                if market[1] == destination
                for opportunity in by_market[market]
                if opportunity.display_locker == locker
                and display_is_unblocked(opportunity)
            ]
            if not feasible_opportunities:
                continue
            total_saving = 0.0
            for market in sorted(
                market for market in by_market if market[1] == destination
            ):
                eligible = [
                    opportunity
                    for opportunity in by_market[market]
                    if opportunity.display_locker == locker
                    and display_is_unblocked(opportunity)
                ]
                best_saving = max(
                    (
                        opportunity.variable_net_savings(config.drone_cost_per_hour)
                        for opportunity in eligible
                    ),
                    default=0.0,
                )
                total_saving += max(0.0, best_saving)
            candidates.append((total_saving, locker))
        selected_display[destination] = (
            min(
                candidates,
                key=lambda item: (-item[0], item[1]),
            )[1]
            if candidates
            else None
        )

    chosen: list[ObjectiveAwareOpportunity] = []
    for market in sorted(by_market):
        _objective_start_check_timeout(should_stop)
        destination = market[1]
        required_display = selected_display.get(destination)
        if destination in selected_display and required_display is None:
            continue
        eligible = [
            opportunity
            for opportunity in by_market[market]
            if required_display is None
            or opportunity.display_locker == required_display
        ]
        if not eligible:
            continue
        ranked = sorted(
            eligible,
            key=lambda item: (
                -item.variable_net_savings(config.drone_cost_per_hour),
                item.support_key,
            ),
        )
        best = ranked[0]
        if best.variable_net_savings(config.drone_cost_per_hour) > 0.0:
            chosen.append(best)

    full_arrivals = {locker: 0.0 for locker in open_h}
    full_route_population = 0.0
    full_base_departure_population = 0.0
    full_swap_event_population = 0.0
    full_service = 0.0
    for opportunity in chosen:
        _objective_start_check_timeout(should_stop)
        amount = opportunity.successful_service
        full_service += amount
        full_route_population += amount * opportunity.route_time_hours
        full_base_departure_population += amount * opportunity.base_departures_per_unit
        full_swap_event_population += amount * opportunity.swap_events_per_unit
        for locker in opportunity.incoming_h:
            full_arrivals[locker] += amount

    # A continuous chi/w/psi flow need not take the entire OD upper bound.  Let
    # lambda scale every selected opportunity's successful-service upper bound.
    # The demand constraints give lambda <= 1.  Locker QoS gives
    # lambda*A_h <= max_c(mu_c - 1/T), so their intersection supplies a closed
    # scalar upper interval without changing or solving the original model.
    service_fraction = 1.0
    inverse_wait_limit = 1.0 / config.max_locker_time_hours
    maximum_service_rate = max(
        option.service_rate for option in config.capacity_options
    )
    for locker in sorted(open_h):
        _objective_start_check_timeout(should_stop)
        full_arrival = full_arrivals.get(locker, 0.0)
        maximum_stable_arrival = maximum_service_rate - inverse_wait_limit
        if full_arrival > 0.0:
            if maximum_stable_arrival <= 0.0:
                return None, "capacity_infeasible"
            service_fraction = min(
                service_fraction,
                maximum_stable_arrival / full_arrival,
            )
        elif maximum_stable_arrival < -OBJECTIVE_START_PROXY_QOS_ABS_TOLERANCE:
            # Opening a locker still requires one selected capacity whose
            # service rate satisfies the QoS margin, even at zero arrival.
            return None, "capacity_infeasible"
    service_fraction = min(1.0, max(0.0, service_fraction))
    if full_service > 0.0 and service_fraction <= 0.0:
        return None, "capacity_infeasible"

    def maximum_capacity_energy(scale: float) -> tuple[float, float]:
        queue_population = math.fsum(
            (scale * full_arrival / (maximum_service_rate - scale * full_arrival))
            for full_arrival in full_arrivals.values()
            if full_arrival > 0.0
        )
        use = queue_population + scale * full_route_population
        supply = (
            scale
            * config.usable_battery_time_hours
            * math.fsum(
                (
                    full_base_departure_population,
                    full_swap_event_population,
                )
            )
        )
        return use, supply

    if full_service > 0.0:
        # At lambda -> 0+, divide the energy inequality by lambda.  The queue
        # term tends to A_h/mu_h.  If that limiting slope is not strictly below
        # the available energy slope, convex queue growth makes every positive
        # amount infeasible; accepting numerical lambda ~= 0 would be spurious.
        limiting_use_slope = full_route_population + math.fsum(
            full_arrival / maximum_service_rate
            for full_arrival in full_arrivals.values()
        )
        supply_slope = config.usable_battery_time_hours * math.fsum(
            (
                full_base_departure_population,
                full_swap_event_population,
            )
        )
        slope_tolerance = OBJECTIVE_START_PROXY_SEARCH_REL_TOLERANCE * max(
            1.0,
            abs(limiting_use_slope),
            abs(supply_slope),
        )
        if limiting_use_slope >= supply_slope - slope_tolerance:
            return None, "energy_infeasible"

        maximum_use, maximum_supply = maximum_capacity_energy(service_fraction)
        boundary_tolerance = OBJECTIVE_START_PROXY_SEARCH_REL_TOLERANCE * max(
            1.0,
            abs(maximum_use),
            abs(maximum_supply),
        )
        if maximum_use > maximum_supply + boundary_tolerance:
            # For maximum service rates, use(lambda)/lambda is increasing, so
            # the positive feasible set is one interval.  Bisection returns
            # its deterministic upper boundary without calling optimize().
            lower = 0.0
            upper = service_fraction
            for _ in range(60):
                _objective_start_check_timeout(should_stop)
                midpoint = 0.5 * (lower + upper)
                use, supply = maximum_capacity_energy(midpoint)
                tolerance = OBJECTIVE_START_PROXY_SEARCH_REL_TOLERANCE * max(
                    1.0, abs(use), abs(supply)
                )
                if use <= supply + tolerance:
                    lower = midpoint
                else:
                    upper = midpoint
                if upper - lower <= OBJECTIVE_START_PROXY_SEARCH_REL_TOLERANCE * max(
                    1.0, service_fraction
                ):
                    break
            service_fraction = lower
            if service_fraction <= 0.0:
                return None, "energy_infeasible"

    # Recompute every proxy term from the actual partial amount rather than
    # retaining a full-demand numerator with only capacity/energy scaled down.
    arrivals = {locker: 0.0 for locker in open_h}
    direct = first = last = pickup = flight = route_population = 0.0
    base_departure_population = 0.0
    swap_event_population = 0.0
    service = 0.0
    for opportunity in chosen:
        _objective_start_check_timeout(should_stop)
        amount = service_fraction * opportunity.successful_service
        service += amount
        direct += amount * opportunity.direct_unit_cost
        first += amount * opportunity.first_mile_unit_cost
        last += amount * opportunity.last_mile_unit_cost
        pickup += amount * opportunity.mx_pickup_unit_cost
        flight += amount * opportunity.flight_unit_cost
        route_population += amount * opportunity.route_time_hours
        base_departure_population += amount * opportunity.base_departures_per_unit
        swap_event_population += amount * opportunity.swap_events_per_unit
        for locker in opportunity.incoming_h:
            arrivals[locker] += amount

    # First enumerate every QoS-feasible capacity at each open H.  Energy is a
    # single f884 system pool, so capacity choices cannot be screened locker by
    # locker.  We start from the cheapest option and deterministically upgrade
    # queue capacity only when the aggregate energy budget requires it.
    choices_by_locker: dict[str, list[tuple[float, int, float, float, float]]] = {}
    for locker in sorted(open_h):
        _objective_start_check_timeout(should_stop)
        arrival = arrivals.get(locker, 0.0)
        choices: list[tuple[float, int, float, float, float]] = []
        for option in config.capacity_options:
            stable_capacity = option.service_rate - 1.0 / config.max_locker_time_hours
            if (
                arrival
                > stable_capacity + OBJECTIVE_START_PROXY_QOS_ABS_TOLERANCE
            ):
                continue
            queue_population = (
                arrival / (option.service_rate - arrival) if arrival > 0.0 else 0.0
            )
            queue_component = config.drone_cost_per_hour * queue_population
            choices.append(
                (
                    option.fixed_cost + queue_component,
                    option.index,
                    option.fixed_cost,
                    queue_component,
                    queue_population,
                )
            )
        if not choices:
            return None, "capacity_infeasible"
        choices_by_locker[locker] = sorted(
            choices,
            key=lambda item: (item[0], item[4], item[1]),
        )

    selected_choices = {
        locker: choices[0] for locker, choices in choices_by_locker.items()
    }
    energy_supply = config.usable_battery_time_hours * math.fsum(
        (base_departure_population, swap_event_population)
    )

    def queue_population_total() -> float:
        return math.fsum(choice[4] for choice in selected_choices.values())

    def energy_violation() -> float:
        return queue_population_total() + route_population - energy_supply

    def energy_tolerance() -> float:
        return OBJECTIVE_START_PROXY_ENERGY_REL_TOLERANCE * max(
            1.0,
            abs(energy_supply),
            abs(route_population),
            abs(queue_population_total()),
        )

    # Each upgrade is a direct move to any lower-queue option for one locker.
    # Ranking by incremental proxy cost per unit queue reduction is cheap,
    # deterministic, and preserves the hard energy gate even on large supports.
    while energy_violation() > energy_tolerance():
        _objective_start_check_timeout(should_stop)
        upgrades: list[
            tuple[float, float, str, int, tuple[float, int, float, float, float]]
        ] = []
        for locker, current in selected_choices.items():
            for alternative in choices_by_locker[locker]:
                reduction = current[4] - alternative[4]
                if reduction <= 1e-12:
                    continue
                incremental_cost = alternative[0] - current[0]
                upgrades.append(
                    (
                        incremental_cost / reduction,
                        incremental_cost,
                        locker,
                        alternative[1],
                        alternative,
                    )
                )
        if not upgrades:
            return None, "energy_infeasible"
        _, _, locker, _, alternative = min(upgrades)
        selected_choices[locker] = alternative

    capacity_cost = 0.0
    queue_cost = 0.0
    total_queue_population = 0.0
    capacity_levels: list[tuple[str, int]] = []
    for locker in sorted(selected_choices):
        (
            _,
            capacity_index,
            fixed_component,
            queue_component,
            selected_queue_population,
        ) = selected_choices[locker]
        capacity_levels.append((locker, capacity_index))
        capacity_cost += fixed_component
        queue_cost += queue_component
        total_queue_population += selected_queue_population

    idle_population = math.fsum(
        rho / (1.0 - rho) for _, rho in reliability_by_idle.values()
    )
    total_fleet_population = math.fsum(
        (total_queue_population, route_population, idle_population)
    )
    rounded_fleet_population = max(
        0,
        math.ceil(total_fleet_population - 1e-9),
    )
    # Keep queue as an auditable subterm while making queue + fleet equal the
    # integer-gamma fleet proxy used by the exact objective.
    fleet_cost = max(
        0.0,
        config.drone_cost_per_hour * rounded_fleet_population - queue_cost,
    )
    objective = ApproximateObjectiveBreakdown(
        direct_courier_baseline=direct,
        first_mile_rider=first,
        last_mile_rider=last,
        mx_locker_pickup=pickup,
        flight=flight,
        facility=math.fsum(facility_costs.values()),
        capacity=capacity_cost,
        connection=math.fsum(connection_costs.values()),
        queue=queue_cost,
        fleet=fleet_cost,
    )
    objective.validate()
    return (
        ObjectiveAwareSupportCandidate(
            candidate_id="evaluation",
            move="evaluation",
            selected_supports=selected,
            open_h=tuple(sorted(open_h)),
            open_facilities=tuple(sorted(open_facilities)),
            active_connections=tuple(sorted(active_connections)),
            capacity_levels=tuple(capacity_levels),
            forced_closed_h=(),
            forced_off_connections=(),
            estimated_service=service,
            objective=objective,
            service_fraction=service_fraction,
            system_energy_use=total_queue_population + route_population,
            system_energy_supply=energy_supply,
            base_departure_population=base_departure_population,
            swap_event_population=swap_event_population,
        ),
        None,
    )


def _objective_candidate_gate(
    candidate: ObjectiveAwareSupportCandidate,
    config: ObjectiveAwarePlanningConfig,
) -> str | None:
    if len(candidate.open_h) < config.min_open_h:
        return "material_open_h"
    if set(candidate.open_h) == set(config.candidate_lockers):
        return "all_candidate_h_open"
    if candidate.estimated_service + 1e-12 < config.min_service:
        return "material_service"
    if candidate.score + 1e-12 < config.min_savings:
        return "material_savings"
    return None


def plan_objective_aware_partial_starts(
    opportunities: Sequence[ObjectiveAwareOpportunity],
    config: ObjectiveAwarePlanningConfig,
    *,
    budget_seconds: float,
    clock: Callable[[], float] = time.perf_counter,
) -> ObjectiveAwareStartPlan:
    """Build deterministic singleton/reciprocal starts within a hard wall budget."""

    config.validate()
    if not math.isfinite(budget_seconds) or budget_seconds < 0.0:
        raise ValueError("budget_seconds must be finite and nonnegative")
    started = clock()
    deadline = started + budget_seconds
    evaluated_count = 0
    rejection_reasons: list[str] = []
    singleton_by_key: dict[SupportKey, ObjectiveAwareSupportCandidate] = {}

    def should_stop() -> bool:
        return clock() > deadline + 1e-12

    def finish(
        candidates: Sequence[ObjectiveAwareSupportCandidate],
        *,
        singleton_count: int,
        timed_out: bool,
    ) -> ObjectiveAwareStartPlan:
        elapsed = max(0.0, clock() - started)
        if timed_out or elapsed > budget_seconds + 1e-12:
            candidates = ()
            timed_out = True
            rejection_reasons.append("budget_exhausted")
        return ObjectiveAwareStartPlan(
            budget_seconds=budget_seconds,
            elapsed_seconds=elapsed,
            opportunity_count=len(opportunities),
            singleton_support_count=singleton_count,
            evaluated_support_count=evaluated_count,
            candidates=tuple(candidates),
            rejection_reasons=tuple(dict.fromkeys(rejection_reasons)),
            timed_out=timed_out,
        )

    if budget_seconds <= 0.0:
        return finish((), singleton_count=0, timed_out=True)

    try:
        opportunity_index: dict[SupportKey, list[ObjectiveAwareOpportunity]] = {}
        for opportunity in opportunities:
            _objective_start_check_timeout(should_stop)
            opportunity.validate()
            opportunity_index.setdefault(opportunity.support_key, []).append(
                opportunity
            )
        support_keys = sorted(opportunity_index)
        if not support_keys:
            rejection_reasons.append("no_opportunities")
            return finish((), singleton_count=0, timed_out=False)

        for key in support_keys:
            _objective_start_check_timeout(should_stop)
            candidate, reason = evaluate_objective_support(
                opportunities,
                (key,),
                config,
                should_stop=should_stop,
                opportunity_index=opportunity_index,
            )
            evaluated_count += 1
            if candidate is None:
                rejection_reasons.append(reason or "singleton_unreliable")
                continue
            singleton_by_key[key] = candidate

        starter_by_keys: dict[
            tuple[SupportKey, ...], ObjectiveAwareSupportCandidate
        ] = {
            candidate.selected_supports: candidate
            for candidate in singleton_by_key.values()
            if candidate.score > config.improvement_tolerance
        }
        support_key_set = set(support_keys)
        reciprocal_pairs = [
            (key, reciprocal)
            for key in support_keys
            for reciprocal in ((key[1], key[0], key[2]),)
            if key < reciprocal and reciprocal in support_key_set
        ]
        for pair in reciprocal_pairs:
            _objective_start_check_timeout(should_stop)
            candidate, reason = evaluate_objective_support(
                opportunities,
                pair,
                config,
                should_stop=should_stop,
                opportunity_index=opportunity_index,
            )
            evaluated_count += 1
            if candidate is None:
                rejection_reasons.append(reason or "reciprocal_pair_unreliable")
                continue
            if candidate.score > config.improvement_tolerance:
                starter_by_keys[candidate.selected_supports] = candidate

        positive_starters = sorted(
            starter_by_keys.values(),
            key=lambda item: (
                -item.score,
                -item.estimated_service,
                item.selected_supports,
            ),
        )
        if not positive_starters:
            rejection_reasons.append("no_positive_support")
            return finish((), singleton_count=len(singleton_by_key), timed_out=False)

        base_candidate = positive_starters[0]
        base_keys = base_candidate.selected_supports
        current_score = base_candidate.score
        for starter in positive_starters[1:]:
            _objective_start_check_timeout(should_stop)
            trial_keys = tuple(sorted(set(base_keys) | set(starter.selected_supports)))
            if trial_keys == base_keys:
                continue
            trial, reason = evaluate_objective_support(
                opportunities,
                trial_keys,
                config,
                should_stop=should_stop,
                opportunity_index=opportunity_index,
            )
            evaluated_count += 1
            if trial is None:
                rejection_reasons.append(reason or "greedy_support_unreliable")
                continue
            if trial.score > current_score + config.improvement_tolerance:
                base_keys = trial_keys
                base_candidate = trial
                current_score = trial.score

        variants: list[
            tuple[str, tuple[SupportKey, ...], ObjectiveAwareSupportCandidate]
        ] = [("base", base_keys, base_candidate)]
        for starter in positive_starters[1:]:
            if starter.selected_supports == base_keys:
                continue
            variants.append(("alternate", starter.selected_supports, starter))
        accepted: list[ObjectiveAwareSupportCandidate] = []
        seen: set[tuple[SupportKey, ...]] = set()
        for move, keys, candidate in variants:
            _objective_start_check_timeout(should_stop)
            if keys in seen:
                continue
            seen.add(keys)
            gate = _objective_candidate_gate(candidate, config)
            if gate is not None:
                rejection_reasons.append(gate)
                continue
            accepted.append(
                replace(
                    candidate,
                    candidate_id=f"objective_partial_{move}_{len(accepted) + 1:02d}",
                    move=move,
                    forced_closed_h=(),
                    forced_off_connections=(),
                )
            )
            if len(accepted) >= config.max_candidates:
                break
        if not accepted:
            rejection_reasons.append("no_material_network_candidate")
        return finish(
            accepted,
            singleton_count=len(singleton_by_key),
            timed_out=False,
        )
    except _ObjectiveStartTimeout:
        return finish(
            (),
            singleton_count=len(singleton_by_key),
            timed_out=True,
        )


def build_candidate_partial_assignment(
    candidate: ObjectiveAwareSupportCandidate,
    variables: Mapping[str, object],
    *,
    mode: str,
) -> dict[gp.Var, float]:
    """Translate one support into a genuinely partial structural assignment.

    In particular, MX ``alpha`` variables are deliberately not assigned.  The
    same conservative choice is used for LO: the exact nearest/display logic
    remains available to Gurobi when it completes the start.
    """

    normalized_mode = mode.upper()
    if normalized_mode not in {"MX", "HD", "LO"}:
        raise ValueError("mode must be MX, HD, or LO")
    required = {"x", "z", "y", "alpha"}
    missing = required - set(variables)
    if missing:
        raise KeyError(f"missing structural variable groups: {sorted(missing)}")
    x = dict(_container_items(variables["x"], "x"))
    z = dict(_container_items(variables["z"], "z"))
    y = dict(_container_items(variables["y"], "y"))
    assignment: dict[gp.Var, float] = {}

    for facility in candidate.open_facilities:
        if facility not in x:
            raise KeyError(f"partial start references unknown facility {facility!r}")
        assignment[x[facility]] = 1.0
    for locker in candidate.forced_closed_h:
        if locker not in x:
            raise KeyError(f"partial start references unknown locker {locker!r}")
        assignment[x[locker]] = 0.0

    capacities_by_locker: dict[str, list[tuple[Hashable, gp.Var]]] = {}
    for key, variable in z.items():
        if not isinstance(key, tuple) or len(key) != 2:
            raise ValueError("z variable keys must be (locker, capacity_index)")
        capacities_by_locker.setdefault(str(key[0]), []).append((key, variable))
    for locker, selected_index in candidate.capacity_levels:
        if locker not in capacities_by_locker:
            raise KeyError(f"partial start has no capacity variables for {locker!r}")
        found = False
        for key, variable in capacities_by_locker[locker]:
            value = 1.0 if key[1] == selected_index else 0.0
            found = found or value == 1.0
            assignment[variable] = value
        if not found:
            raise KeyError(
                f"partial start references unknown capacity {selected_index} for {locker!r}"
            )
    for locker in candidate.forced_closed_h:
        for _, variable in capacities_by_locker.get(locker, ()):
            assignment[variable] = 0.0

    for arc in candidate.active_connections:
        if arc not in y:
            raise KeyError(f"partial start references unknown connection {arc!r}")
        assignment[y[arc]] = 1.0
    for arc in candidate.forced_off_connections:
        if arc in y and arc not in candidate.active_connections:
            assignment[y[arc]] = 0.0
    # No alpha assignment here.  In MX this is essential: setting every alpha
    # to zero would turn the hint into the harmful historical HD-like seed.
    return assignment


def build_all_courier_partial_assignment(
    variables: Mapping[str, object],
) -> dict[gp.Var, float]:
    """Return the safe all-courier structural start, leaving repair undefined."""

    assignment: dict[gp.Var, float] = {}
    for group in ("x", "z", "y", "delta", "alpha"):
        if group not in variables:
            raise KeyError(f"missing structural variable group {group!r}")
        for _, variable in _container_items(variables[group], group):
            assignment[variable] = 0.0
    return assignment


def build_partial_start_vector(
    model_variables: Sequence[gp.Var],
    assignments: Mapping[gp.Var, float],
) -> array[float]:
    """Build one vector with every undecided variable set to ``UNDEFINED``."""

    known = set(model_variables)
    unknown = [variable for variable in assignments if variable not in known]
    if unknown:
        raise ValueError("partial MIP start contains variables outside the model")
    for value in assignments.values():
        number = float(value)
        if not math.isfinite(number) or number == float(GRB.UNDEFINED):
            raise ValueError("assigned partial-start values must be finite and decided")
    return array(
        "d",
        (
            float(assignments.get(variable, GRB.UNDEFINED))
            for variable in model_variables
        ),
    )


def install_partial_mip_starts(
    model: gp.Model,
    assignments: Sequence[Mapping[gp.Var, float]],
) -> tuple[int, ...]:
    """Replace the start pool with sparse partial starts; never optimize.

    A newly allocated Gurobi start slot gives every unassigned variable the
    ``GRB.UNDEFINED`` value.  Writing only the decided variables therefore has
    exactly the same semantics as a dense vector while avoiding one full-model
    attribute write per candidate.  This distinction is material for the
    five-percent preparation budget on the large formulation.
    """

    if not assignments:
        raise ValueError("at least one partial MIP start is required")
    sparse_starts: list[tuple[tuple[gp.Var, ...], tuple[float, ...]]] = []
    for assignment in assignments:
        variables = tuple(assignment)
        values = tuple(float(assignment[variable]) for variable in variables)
        if any(
            not math.isfinite(value) or value == float(GRB.UNDEFINED)
            for value in values
        ):
            raise ValueError("assigned partial-start values must be finite and decided")
        sparse_starts.append((variables, values))

    model.update()
    # Dropping the old pool first prevents any stale value from being inherited
    # if a wrapper is deliberately reused for a development run.
    model.NumStart = 0
    model.update()
    try:
        model.NumStart = len(sparse_starts)
        model.update()
        for index, (variables, values) in enumerate(sparse_starts):
            model.Params.StartNumber = index
            if variables:
                model.setAttr(GRB.Attr.Start, list(variables), list(values))
        model.Params.StartNumber = 0
        model.update()
    except Exception:
        # A foreign/stale variable or backend error must not leave a partially
        # installed start pool behind.
        model.NumStart = 0
        model.update()
        raise
    return tuple(len(variables) for variables, _ in sparse_starts)


@dataclass(frozen=True)
class PrimalSeed:
    """A feasible incumbent and its binary design projection.

    ``values`` contains every model variable.  Keeping the full incumbent is
    important: the local subproblem is the actual conic model, so its start
    should include the continuous queueing and network repair as well as the
    structural binaries.
    """

    objective: float
    structure: Mapping[str, Mapping[Hashable, int]]
    values: Mapping[gp.Var, float]


@dataclass(frozen=True)
class FixedIntegerRepairConfig:
    """Controls a short feasibility repair with the integer design fixed."""

    time_limit: float = 5.0
    mip_gap: float = 0.01
    method: int = 2
    miqcp_method: int = -1
    integrality_tolerance: float = 1e-5
    output_flag: int | None = None

    def validate(self) -> None:
        if not math.isfinite(self.time_limit) or self.time_limit <= 0.0:
            raise ValueError(
                "fixed-integer repair time_limit must be positive and finite"
            )
        if not math.isfinite(self.mip_gap) or not 0.0 <= self.mip_gap <= 1.0:
            raise ValueError("fixed-integer repair mip_gap must be in [0, 1]")
        if self.method not in {-1, 0, 1, 2, 3, 4, 5}:
            raise ValueError(
                "fixed-integer repair method must be a valid Gurobi Method value"
            )
        if self.miqcp_method not in {-1, 0, 1}:
            raise ValueError(
                "fixed-integer repair miqcp_method must be a valid Gurobi "
                "MIQCPMethod value"
            )
        if (
            not math.isfinite(self.integrality_tolerance)
            or self.integrality_tolerance < 0.0
        ):
            raise ValueError(
                "fixed-integer repair integrality_tolerance must be finite and nonnegative"
            )
        if self.output_flag not in {None, 0, 1}:
            raise ValueError("output_flag must be None, 0, or 1")


@dataclass(frozen=True)
class FixedIntegerRepairResult:
    """Outcome of validating and repairing one complete primal seed."""

    found_incumbent: bool
    installed_source: str
    seed_objective: float
    candidate_objective: float | None
    installed_objective: float
    fixed_integer_count: int
    status: int
    solution_count: int
    runtime: float


@dataclass(frozen=True)
class LocalBranchingConfig:
    """Controls a short exact-MISOCP local-branching search."""

    radius: int = 10
    time_limit: float = 10.0
    mip_gap: float = 0.01
    include_y: bool = False
    improvement_abs_tol: float = 1e-6
    improvement_rel_tol: float = 1e-8
    mip_focus: int = 1
    heuristics: float = 0.5
    no_rel_heur_time: float = 0.0
    output_flag: int | None = None

    def validate(self) -> None:
        if (
            isinstance(self.radius, bool)
            or not isinstance(self.radius, int)
            or self.radius < 0
        ):
            raise ValueError("local-branching radius must be a nonnegative integer")
        if not math.isfinite(self.time_limit) or self.time_limit <= 0.0:
            raise ValueError("local-branching time_limit must be positive and finite")
        if not math.isfinite(self.mip_gap) or not 0.0 <= self.mip_gap <= 1.0:
            raise ValueError("local-branching mip_gap must be in [0, 1]")
        tolerances = (self.improvement_abs_tol, self.improvement_rel_tol)
        if any(not math.isfinite(value) or value < 0.0 for value in tolerances):
            raise ValueError("improvement tolerances must be finite and nonnegative")
        if self.mip_focus not in {0, 1, 2, 3}:
            raise ValueError("mip_focus must be one of 0, 1, 2, 3")
        if not math.isfinite(self.heuristics) or not 0.0 <= self.heuristics <= 1.0:
            raise ValueError("heuristics must be in [0, 1]")
        if not math.isfinite(self.no_rel_heur_time) or self.no_rel_heur_time < 0.0:
            raise ValueError("no_rel_heur_time must be finite and nonnegative")
        if self.output_flag not in {None, 0, 1}:
            raise ValueError("output_flag must be None, 0, or 1")


@dataclass(frozen=True)
class LocalBranchingResult:
    """Outcome of one local-neighbourhood solve."""

    improved: bool
    seed_objective: float
    candidate_objective: float | None
    installed_objective: float
    selected_binary_count: int
    hamming_distance: int | None
    radius: int
    status: int
    solution_count: int
    runtime: float


@dataclass(frozen=True)
class _MIPStartSnapshot:
    num_start: int
    start_number: int
    values: tuple[tuple[float, ...], ...]


@dataclass(frozen=True)
class _MIPStartPoolMark:
    num_start: int
    start_number: int


def _structure_groups(include_y: bool) -> tuple[str, ...]:
    return DEFAULT_STRUCTURE_GROUPS + (("y",) if include_y else ())


def _container_items(container: object, group: str) -> list[tuple[Hashable, gp.Var]]:
    if not hasattr(container, "items"):
        raise TypeError(f"variables[{group!r}] must be a keyed variable container")
    return list(container.items())  # type: ignore[union-attr, no-any-return]


def _selected_variables(
    variables: Mapping[str, object], include_y: bool
) -> dict[str, list[tuple[Hashable, gp.Var]]]:
    selected: dict[str, list[tuple[Hashable, gp.Var]]] = {}
    for group in _structure_groups(include_y):
        if group not in variables:
            raise KeyError(f"missing structural variable group {group!r}")
        items = _container_items(variables[group], group)
        for _, variable in items:
            if variable.VType != GRB.BINARY:
                raise TypeError(f"variables[{group!r}] contains a nonbinary variable")
        selected[group] = items
    return selected


def capture_incumbent(
    model: gp.Model,
    variables: Mapping[str, object],
    *,
    include_y: bool = False,
    integrality_tolerance: float = 1e-5,
) -> PrimalSeed:
    """Capture the current feasible incumbent as a complete primal seed."""

    model.update()
    if model.SolCount <= 0:
        raise ValueError(
            "cannot capture a primal seed before a feasible incumbent exists"
        )
    if not math.isfinite(integrality_tolerance) or integrality_tolerance < 0.0:
        raise ValueError("integrality_tolerance must be finite and nonnegative")

    selected = _selected_variables(variables, include_y)
    structure: dict[str, dict[Hashable, int]] = {}
    for group, items in selected.items():
        group_values: dict[Hashable, int] = {}
        for key, variable in items:
            value = float(variable.X)
            binary_value = int(value >= 0.5)
            if abs(value - binary_value) > integrality_tolerance:
                raise ValueError(
                    f"incumbent value {value} for {group}[{key!r}] is not binary"
                )
            group_values[key] = binary_value
        structure[group] = group_values

    model_variables = model.getVars()
    values = dict(zip(model_variables, model.getAttr(GRB.Attr.X, model_variables)))
    objective = float(model.ObjVal)
    if not math.isfinite(objective):
        raise ValueError("incumbent objective must be finite")
    return PrimalSeed(objective=objective, structure=structure, values=values)


def build_hamming_expression(
    variables: Mapping[str, object],
    structure: Mapping[str, Mapping[Hashable, int]],
    *,
    include_y: bool = False,
) -> tuple[gp.LinExpr, int]:
    r"""Build the exact Hamming distance from a binary design seed.

    For selected binary variables :math:`b` and seed :math:`\hat b`, the
    returned expression is

    .. math::

        \sum_{k:\hat b_k=0} b_k + \sum_{k:\hat b_k=1}(1-b_k).

    By default the design vector is ``(x, z, delta, alpha)``.  Arc decisions
    ``y`` can be included explicitly because they make the neighbourhood much
    larger and are partly implied by the facility decisions.
    """

    selected = _selected_variables(variables, include_y)
    expression = gp.LinExpr()
    count = 0
    for group, items in selected.items():
        if group not in structure:
            raise KeyError(f"primal seed is missing structural group {group!r}")
        expected_keys = {key for key, _ in items}
        supplied_keys = set(structure[group])
        if supplied_keys != expected_keys:
            missing = expected_keys - supplied_keys
            extra = supplied_keys - expected_keys
            raise ValueError(
                f"seed keys for {group!r} do not match model variables "
                f"(missing={len(missing)}, extra={len(extra)})"
            )
        for key, variable in items:
            value = structure[group][key]
            if isinstance(value, bool):
                value = int(value)
            if value not in {0, 1}:
                raise ValueError(f"seed value for {group}[{key!r}] must be 0 or 1")
            if value == 0:
                expression += variable
            else:
                expression += 1.0 - variable
            count += 1
    if count == 0:
        raise ValueError("the selected local-branching design vector is empty")
    return expression, count


@contextmanager
def temporary_local_branching_neighborhood(
    model: gp.Model,
    hamming_expression: gp.LinExpr,
    radius: int,
    *,
    objective_cutoff: float | None = None,
    parameter_overrides: Mapping[str, Any] | None = None,
) -> Iterator[tuple[gp.Constr | gp.QConstr, ...]]:
    """Temporarily add a local branch and restore the model transactionally.

    The snapshot uses Gurobi's bulk attribute interface, so every existing
    variable lower and upper bound is restored even if code executed inside the
    context changes a bound or raises an exception.  Only parameters listed in
    ``parameter_overrides`` are changed, and each of those is restored.
    """

    if isinstance(radius, bool) or not isinstance(radius, int) or radius < 0:
        raise ValueError("local-branching radius must be a nonnegative integer")
    if objective_cutoff is not None and not math.isfinite(objective_cutoff):
        raise ValueError("objective_cutoff must be finite or None")
    overrides = dict(parameter_overrides or {})

    model.update()
    model_variables = model.getVars()
    lower_bounds = model.getAttr(GRB.Attr.LB, model_variables)
    upper_bounds = model.getAttr(GRB.Attr.UB, model_variables)
    old_parameters = {name: getattr(model.Params, name) for name in overrides}
    temporary_constraints: list[gp.Constr | gp.QConstr] = []
    try:
        for name, value in overrides.items():
            setattr(model.Params, name, value)
        temporary_constraints.append(
            model.addConstr(
                hamming_expression <= radius, name="__local_branching_hamming"
            )
        )
        if objective_cutoff is not None:
            if model.ModelSense != GRB.MINIMIZE:
                raise ValueError(
                    "objective improvement cutoff currently requires minimization"
                )
            temporary_constraints.append(
                model.addConstr(
                    model.getObjective() <= objective_cutoff,
                    name="__local_branching_improvement",
                )
            )
        model.update()
        yield tuple(temporary_constraints)
    finally:
        if temporary_constraints:
            model.remove(temporary_constraints)
        model.setAttr(GRB.Attr.LB, model_variables, lower_bounds)
        model.setAttr(GRB.Attr.UB, model_variables, upper_bounds)
        model.update()
        for name, value in old_parameters.items():
            setattr(model.Params, name, value)


def _snapshot_mip_starts(
    model: gp.Model, model_variables: list[gp.Var]
) -> _MIPStartSnapshot:
    model.update()
    num_start = int(model.NumStart)
    start_number = int(model.Params.StartNumber)
    values: list[tuple[float, ...]] = []
    for index in range(num_start):
        model.Params.StartNumber = index
        values.append(tuple(model.getAttr(GRB.Attr.Start, model_variables)))
    model.Params.StartNumber = start_number
    return _MIPStartSnapshot(num_start, start_number, tuple(values))


def _restore_mip_starts(
    model: gp.Model, model_variables: list[gp.Var], snapshot: _MIPStartSnapshot
) -> None:
    model.NumStart = snapshot.num_start
    model.update()
    for index, values in enumerate(snapshot.values):
        model.Params.StartNumber = index
        model.setAttr(GRB.Attr.Start, model_variables, list(values))
    model.Params.StartNumber = snapshot.start_number
    model.update()


def _complete_start_vector(
    model_variables: Sequence[gp.Var], values: Mapping[gp.Var, float]
) -> array[float]:
    if len(values) != len(model_variables) or any(
        variable not in values for variable in model_variables
    ):
        raise ValueError("a primal seed must provide a value for every model variable")
    start_values = array("d", (float(values[variable]) for variable in model_variables))
    if any(not math.isfinite(value) for value in start_values):
        raise ValueError("all primal seed values must be finite")
    return start_values


def _install_start_vector(
    model: gp.Model,
    model_variables: list[gp.Var],
    start_values: Sequence[float],
) -> None:
    if len(start_values) != len(model_variables):
        raise ValueError("a MIP-start vector must provide one value per model variable")
    if any(not math.isfinite(float(value)) for value in start_values):
        raise ValueError("all MIP-start values must be finite")
    model.NumStart = 1
    model.update()
    model.Params.StartNumber = 0
    model.setAttr(GRB.Attr.Start, model_variables, start_values)
    model.Params.StartNumber = 0
    model.update()


def _install_complete_start(
    model: gp.Model, model_variables: list[gp.Var], values: Mapping[gp.Var, float]
) -> None:
    _install_start_vector(
        model, model_variables, _complete_start_vector(model_variables, values)
    )


def _append_start_vector(
    model: gp.Model,
    model_variables: list[gp.Var],
    start_values: Sequence[float],
) -> _MIPStartPoolMark:
    """Append a temporary start without copying the existing start pool."""

    mark = _MIPStartPoolMark(
        num_start=int(model.NumStart),
        start_number=int(model.Params.StartNumber),
    )
    model.NumStart = mark.num_start + 1
    model.update()
    model.Params.StartNumber = mark.num_start
    model.setAttr(GRB.Attr.Start, model_variables, start_values)
    model.update()
    return mark


def _restore_mip_start_pool_mark(model: gp.Model, mark: _MIPStartPoolMark) -> None:
    """Drop only starts appended after ``mark``, preserving earlier values."""

    model.NumStart = mark.num_start
    model.update()
    model.Params.StartNumber = mark.start_number
    model.update()


def _fixed_integer_values(
    model: gp.Model,
    model_variables: list[gp.Var],
    start_values: Sequence[float],
    *,
    integrality_tolerance: float,
) -> tuple[list[gp.Var], array[float]]:
    variable_types = model.getAttr(GRB.Attr.VType, model_variables)
    integer_variables: list[gp.Var] = []
    integer_values = array("d")
    for variable, variable_type, seed_value in zip(
        model_variables, variable_types, start_values
    ):
        if variable_type not in {GRB.BINARY, GRB.INTEGER}:
            continue
        rounded_value = int(round(float(seed_value)))
        if abs(float(seed_value) - rounded_value) > integrality_tolerance:
            raise ValueError(
                f"seed value {seed_value} for integer variable "
                f"{variable.VarName!r} is not integral"
            )
        if variable_type == GRB.BINARY and rounded_value not in {0, 1}:
            raise ValueError(
                f"seed value {seed_value} for binary variable "
                f"{variable.VarName!r} is not 0 or 1"
            )
        if (
            rounded_value < float(variable.LB) - integrality_tolerance
            or rounded_value > float(variable.UB) + integrality_tolerance
        ):
            raise ValueError(
                f"seed value {seed_value} for integer variable "
                f"{variable.VarName!r} violates its bounds"
            )
        integer_variables.append(variable)
        integer_values.append(float(rounded_value))
    return integer_variables, integer_values


@contextmanager
def _temporary_fixed_integer_design(
    model: gp.Model,
    model_variables: list[gp.Var],
    integer_variables: list[gp.Var],
    integer_values: Sequence[float],
    parameter_overrides: Mapping[str, Any],
) -> Iterator[None]:
    """Fix an integer design and restore all bounds and changed parameters."""

    # Compact C-double arrays keep a million-variable bound snapshot near 16 MB
    # instead of retaining two Python-float lists for the duration of the solve.
    lower_bounds = array("d", model.getAttr(GRB.Attr.LB, model_variables))
    upper_bounds = array("d", model.getAttr(GRB.Attr.UB, model_variables))
    old_parameters = {name: getattr(model.Params, name) for name in parameter_overrides}
    try:
        for name, value in parameter_overrides.items():
            setattr(model.Params, name, value)
        if integer_variables:
            model.setAttr(GRB.Attr.LB, integer_variables, integer_values)
            model.setAttr(GRB.Attr.UB, integer_variables, integer_values)
        model.update()
        yield
    finally:
        try:
            model.setAttr(GRB.Attr.LB, model_variables, lower_bounds)
            model.setAttr(GRB.Attr.UB, model_variables, upper_bounds)
            model.update()
        finally:
            for name, value in old_parameters.items():
                setattr(model.Params, name, value)


def repair_fixed_integer_design_start(
    model: gp.Model,
    seed: PrimalSeed,
    config: FixedIntegerRepairConfig | None = None,
) -> FixedIntegerRepairResult:
    """Repair a complete seed while holding every discrete decision fixed.

    The auxiliary solve is the unchanged original model: it retains every
    continuous variable, objective term, linear constraint, and conic
    constraint.  Only binary and general-integer bounds are temporarily fixed
    to the seed.  A single thread and short time limit are enforced.  If the
    solve finds an incumbent, its complete solution vector replaces the seed
    as the sole MIP start; otherwise the input seed remains installed.

    Existing MIP starts are not copied for the auxiliary solve.  The seed is
    appended transactionally, which keeps exception rollback practical even
    for models with millions of variables.
    """

    config = config or FixedIntegerRepairConfig()
    config.validate()
    if not math.isfinite(seed.objective):
        raise ValueError("seed objective must be finite")

    model.update()
    model_variables = model.getVars()
    start_values = _complete_start_vector(model_variables, seed.values)
    integer_variables, integer_values = _fixed_integer_values(
        model,
        model_variables,
        start_values,
        integrality_tolerance=config.integrality_tolerance,
    )
    parameter_overrides: dict[str, Any] = {
        "TimeLimit": config.time_limit,
        "MIPGap": config.mip_gap,
        "Threads": 1,
        "Method": config.method,
        "MIQCPMethod": config.miqcp_method,
    }
    if config.output_flag is not None:
        parameter_overrides["OutputFlag"] = config.output_flag

    temporary_start_mark: _MIPStartPoolMark | None = None
    candidate_values: array[float] | None = None
    candidate_objective: float | None = None
    status = GRB.LOADED
    solution_count = 0
    runtime = 0.0
    try:
        temporary_start_mark = _append_start_vector(
            model, model_variables, start_values
        )
        with _temporary_fixed_integer_design(
            model,
            model_variables,
            integer_variables,
            integer_values,
            parameter_overrides,
        ):
            model.optimize()
            status = int(model.Status)
            solution_count = int(model.SolCount)
            runtime = float(model.Runtime)
            if solution_count > 0:
                candidate_objective = float(model.ObjVal)
                candidate_values = array(
                    "d", model.getAttr(GRB.Attr.X, model_variables)
                )
                if not math.isfinite(candidate_objective) or any(
                    not math.isfinite(value) for value in candidate_values
                ):
                    raise ValueError(
                        "fixed-integer repair returned a nonfinite incumbent"
                    )
    except BaseException:
        if temporary_start_mark is not None:
            _restore_mip_start_pool_mark(model, temporary_start_mark)
        raise

    found_incumbent = candidate_values is not None
    if candidate_values is not None:
        assert candidate_objective is not None
        _install_start_vector(model, model_variables, candidate_values)
        installed_source = "repaired"
        installed_objective = float(candidate_objective)
    else:
        _install_start_vector(model, model_variables, start_values)
        installed_source = "seed"
        installed_objective = float(seed.objective)

    return FixedIntegerRepairResult(
        found_incumbent=found_incumbent,
        installed_source=installed_source,
        seed_objective=float(seed.objective),
        candidate_objective=candidate_objective,
        installed_objective=installed_objective,
        fixed_integer_count=len(integer_variables),
        status=status,
        solution_count=solution_count,
        runtime=runtime,
    )


def _candidate_hamming_distance(
    variables: Mapping[str, object],
    structure: Mapping[str, Mapping[Hashable, int]],
    *,
    include_y: bool,
) -> int:
    distance = 0
    for group, items in _selected_variables(variables, include_y).items():
        for key, variable in items:
            distance += int(int(float(variable.X) >= 0.5) != structure[group][key])
    return distance


def improve_mip_start(
    model: gp.Model,
    variables: Mapping[str, object],
    seed: PrimalSeed,
    config: LocalBranchingConfig | None = None,
) -> LocalBranchingResult:
    """Search an exact conic Hamming neighbourhood and install the best start.

    The center seed remains feasible inside the Hamming ball, giving the local
    MISOCP an immediate incumbent and objective bound.  Only a strictly better
    completed solution is accepted and installed.  The global solve that
    follows remains the unmodified original MISOCP.
    """

    config = config or LocalBranchingConfig()
    config.validate()
    if not math.isfinite(seed.objective):
        raise ValueError("seed objective must be finite")
    if model.ModelSense != GRB.MINIMIZE:
        raise ValueError("exact-conic local branching currently requires minimization")

    hamming_expression, selected_count = build_hamming_expression(
        variables, seed.structure, include_y=config.include_y
    )
    tolerance = max(
        config.improvement_abs_tol,
        config.improvement_rel_tol * max(1.0, abs(seed.objective)),
    )
    parameter_overrides: dict[str, Any] = {
        "TimeLimit": config.time_limit,
        "MIPGap": config.mip_gap,
        "MIPFocus": config.mip_focus,
        "Heuristics": config.heuristics,
        "NoRelHeurTime": config.no_rel_heur_time,
    }
    if config.output_flag is not None:
        parameter_overrides["OutputFlag"] = config.output_flag

    model.update()
    model_variables = model.getVars()
    previous_starts = _snapshot_mip_starts(model, model_variables)
    candidate_values: dict[gp.Var, float] | None = None
    candidate_objective: float | None = None
    hamming_distance: int | None = None
    status = GRB.LOADED
    solution_count = 0
    runtime = 0.0
    try:
        _install_complete_start(model, model_variables, seed.values)
        with temporary_local_branching_neighborhood(
            model,
            hamming_expression,
            config.radius,
            parameter_overrides=parameter_overrides,
        ):
            model.optimize()
            status = int(model.Status)
            solution_count = int(model.SolCount)
            runtime = float(model.Runtime)
            if solution_count > 0:
                candidate_objective = float(model.ObjVal)
                hamming_distance = _candidate_hamming_distance(
                    variables, seed.structure, include_y=config.include_y
                )
                if candidate_objective < seed.objective - tolerance:
                    candidate_values = dict(
                        zip(
                            model_variables,
                            model.getAttr(GRB.Attr.X, model_variables),
                        )
                    )
    except BaseException:
        _restore_mip_starts(model, model_variables, previous_starts)
        raise

    improved = candidate_values is not None
    if candidate_values is not None:
        assert candidate_objective is not None
        _install_complete_start(model, model_variables, candidate_values)
        installed_objective = float(candidate_objective)
    else:
        _install_complete_start(model, model_variables, seed.values)
        installed_objective = float(seed.objective)

    return LocalBranchingResult(
        improved=improved,
        seed_objective=float(seed.objective),
        candidate_objective=candidate_objective,
        installed_objective=installed_objective,
        selected_binary_count=selected_count,
        hamming_distance=hamming_distance,
        radius=config.radius,
        status=status,
        solution_count=solution_count,
        runtime=runtime,
    )
