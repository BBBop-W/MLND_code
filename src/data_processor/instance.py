"""Canonical OD-aware instance and derived physical-network relations."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from src.config import ModelConfig

Arc = tuple[str, str]
Market = tuple[str, str]
DisplayKey = tuple[str, str, str]  # (o, d, j)
DIRECT_ROUTE = "__direct__"


@dataclass
class Instance:
    """A model instance that keeps customer OD markets separate from lockers."""

    L: list[str] = field(default_factory=list)
    H: list[str] = field(default_factory=list)
    E: list[str] = field(default_factory=list)
    S: list[str] = field(default_factory=list)
    N: list[str] = field(default_factory=list)
    coords: dict[str, tuple[float, float]] = field(default_factory=dict)
    activity_rate: dict[str, float] = field(default_factory=dict)
    h_od: dict[Market, float] = field(default_factory=dict)
    order_count: dict[Market, int] = field(default_factory=dict)
    locker_pickup_cost_by_locker: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    config: ModelConfig = field(default_factory=ModelConfig)

    h_to_s: dict[str, str] = field(default_factory=dict, init=False)
    s_to_h: dict[str, str] = field(default_factory=dict, init=False)
    A_LH: set[Arc] = field(default_factory=set, init=False)
    A_SH: set[Arc] = field(default_factory=set, init=False)
    A_HH: set[Arc] = field(default_factory=set, init=False)
    A_HL: set[Arc] = field(default_factory=set, init=False)
    A_HS: set[Arc] = field(default_factory=set, init=False)
    A_F: set[Arc] = field(default_factory=set, init=False)
    A: set[Arc] = field(default_factory=set, init=False)
    A_Y: set[Arc] = field(default_factory=set, init=False)
    route_options: dict[Arc, tuple[str, ...]] = field(default_factory=dict, init=False)
    route_distance: dict[tuple[str, str, str], float] = field(
        default_factory=dict, init=False
    )
    route_time: dict[tuple[str, str, str], float] = field(
        default_factory=dict, init=False
    )
    arc_activation_cost: dict[Arc, float] = field(default_factory=dict, init=False)
    eligible_idle: dict[str, tuple[str, ...]] = field(default_factory=dict, init=False)
    display_lockers: dict[str, tuple[str, ...]] = field(
        default_factory=dict, init=False
    )
    closer_lockers: dict[tuple[str, str], tuple[str, ...]] = field(
        default_factory=dict, init=False
    )
    display_origins: dict[DisplayKey, tuple[str, ...]] = field(
        default_factory=dict, init=False
    )
    general_corridors: dict[Market, tuple[Arc, ...]] = field(
        default_factory=dict, init=False
    )
    choice_share: dict[tuple[str, str, str], float] = field(
        default_factory=dict, init=False
    )
    M_psi: dict[Arc, float] = field(default_factory=dict, init=False)
    M_xi: dict[str, float] = field(default_factory=dict, init=False)

    def build(self) -> "Instance":
        self.config.validate()
        self._validate_inputs()
        self.L.sort()
        self.H.sort()
        self.E.sort()
        self.N.sort()
        self.h_od = {
            market: float(rate)
            for market, rate in self.h_od.items()
            if float(rate) > 0.0
        }
        self.order_count = {
            market: count
            for market, count in self.order_count.items()
            if market in self.h_od
        }
        self._build_virtual_states()
        self._build_physical_network()
        self._build_customer_interface()
        self._build_bounds()
        return self

    @property
    def markets(self) -> tuple[Market, ...]:
        return tuple(sorted(self.h_od))

    @property
    def total_demand(self) -> float:
        return sum(self.h_od.values())

    @property
    def idle_states(self) -> tuple[str, ...]:
        return tuple(self.L + self.S)

    def distance(self, u: str, v: str) -> float:
        ux, uy = self.coords[u]
        vx, vy = self.coords[v]
        return math.hypot(ux - vx, uy - vy)

    def rider_cost(self, u: str, v: str) -> float:
        """Return the direct-courier tariff for one origin-destination leg."""

        cfg = self.config
        road_distance = cfg.road_distance_factor * self.distance(u, v)
        excess_distance = max(0.0, road_distance - cfg.rider_included_distance_km)
        return (
            cfg.rider_initial_charge_cny
            + cfg.rider_incremental_cost_per_km * excess_distance
        )

    def direct_rider_cost(self, u: str, v: str) -> float:
        """Return :math:`f_{od}^{rider}` for direct courier fulfillment."""

        return self.rider_cost(u, v)

    def feeder_rider_cost(self, u: str, v: str) -> float:
        """Return :math:`f_{oi}^{rider}` or :math:`f_{jd}^{rider}`."""

        return self.config.feeder_rider_cost_factor * self.rider_cost(u, v)

    def facility_cost(self, node: str) -> float:
        if node in self.L:
            return self.config.base_cost_per_hour
        if node in self.H:
            return self.config.locker_fixed_cost_per_hour
        if node in self.E:
            return self.config.swap_cost_per_hour
        raise KeyError(f"unknown physical facility {node!r}")

    def locker_pickup_cost(self, locker: str) -> float:
        """Return the pickup-fulfillment cost for one destination locker."""

        if locker not in self.H:
            raise KeyError(f"unknown smart locker {locker!r}")
        return self.locker_pickup_cost_by_locker.get(
            locker,
            self.config.locker_pickup_cost_per_order,
        )

    def corridors_for_mode(self, mode: str, market: Market) -> tuple[Arc, ...]:
        """Return the retained loaded corridors available to one formulation."""

        normalized_mode = mode.upper()
        if normalized_mode not in {"MX", "HD", "LO"}:
            raise ValueError("mode must be one of MX, HD, or LO")
        if market not in self.h_od:
            raise KeyError(f"unknown positive-demand market {market!r}")

        general = self.general_corridors.get(market, ())
        if normalized_mode == "HD":
            return general

        origin, destination = market
        displayed = tuple(
            (locker_origin, locker)
            for locker in self.display_lockers.get(destination, ())
            for locker_origin in self.display_origins.get(
                (origin, destination, locker),
                (),
            )
        )
        displayed = tuple(dict.fromkeys(displayed))
        if normalized_mode == "LO":
            return displayed
        return tuple(dict.fromkeys((*general, *displayed)))

    def screening_is_exact(self, mode: str) -> bool:
        """Report whether preprocessing retains the full formulation index set."""

        normalized_mode = mode.upper()
        if normalized_mode not in {"MX", "HD", "LO"}:
            raise ValueError("mode must be one of MX, HD, or LO")
        full_corridors = set(self.A_HH)
        for market in self.markets:
            if normalized_mode == "LO":
                destination = market[1]
                eligible_destinations = set(self.display_lockers.get(destination, ()))
                expected = {(i, j) for i, j in self.A_HH if j in eligible_destinations}
            else:
                expected = full_corridors
            if set(self.corridors_for_mode(normalized_mode, market)) != expected:
                return False
        return True

    def endpoint_facility(self, node: str) -> str:
        return self.s_to_h.get(node, node)

    def _validate_inputs(self) -> None:
        node_groups = {
            "L": self.L,
            "H": self.H,
            "E": self.E,
            "N": self.N,
        }
        for group_name, nodes in node_groups.items():
            if len(nodes) != len(set(nodes)):
                raise ValueError(f"duplicate node identifier in {group_name}")
            if any(not isinstance(node, str) or not node for node in nodes):
                raise ValueError(
                    f"every node identifier in {group_name} must be a nonempty string"
                )

        physical_groups = (("L", set(self.L)), ("H", set(self.H)), ("E", set(self.E)))
        for index, (left_name, left) in enumerate(physical_groups):
            for right_name, right in physical_groups[index + 1 :]:
                overlap = sorted(left & right)
                if overlap:
                    raise ValueError(
                        f"facility sets {left_name} and {right_name} must be disjoint; "
                        f"overlap={overlap}"
                    )
        physical_nodes = set(self.L) | set(self.H) | set(self.E)
        demand_nodes = set(self.N)
        overlap = sorted(physical_nodes & demand_nodes)
        if overlap:
            raise ValueError(
                "physical-facility and demand-node identifiers must be distinct; "
                f"overlap={overlap}"
            )

        expected_virtual = {f"S_{locker}" for locker in self.H}
        known_coordinate_nodes = physical_nodes | demand_nodes | expected_virtual
        unknown_coordinate_nodes = sorted(set(self.coords) - known_coordinate_nodes)
        if unknown_coordinate_nodes:
            raise ValueError(
                f"coordinates supplied for unknown nodes: {unknown_coordinate_nodes}"
            )
        missing_coordinates = sorted((physical_nodes | demand_nodes) - set(self.coords))
        if missing_coordinates:
            raise ValueError(f"missing coordinates for nodes: {missing_coordinates}")
        validated_coordinates: dict[str, tuple[float, float]] = {}
        for node, coordinate in self.coords.items():
            if not isinstance(coordinate, (tuple, list)) or len(coordinate) != 2:
                raise ValueError(
                    f"coordinate for {node!r} must contain exactly two values"
                )
            try:
                normalized_coordinate = (float(coordinate[0]), float(coordinate[1]))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"coordinate for {node!r} must be numeric") from exc
            if any(not math.isfinite(value) for value in normalized_coordinate):
                raise ValueError(f"coordinate for {node!r} must be finite")
            validated_coordinates[node] = normalized_coordinate
        self.coords = validated_coordinates

        unknown_activity_nodes = sorted(set(self.activity_rate) - physical_nodes)
        if unknown_activity_nodes:
            raise ValueError(
                f"activity rates supplied for unknown sites: {unknown_activity_nodes}"
            )
        validated_activity: dict[str, float] = {}
        for node, value in self.activity_rate.items():
            try:
                normalized_value = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"activity rate for {node!r} must be numeric") from exc
            if not math.isfinite(normalized_value) or normalized_value < 0.0:
                raise ValueError(
                    f"activity rate for {node!r} must be finite and nonnegative"
                )
            validated_activity[node] = normalized_value
        self.activity_rate = validated_activity

        validated_demand: dict[Market, float] = {}
        for market, raw_rate in self.h_od.items():
            if not isinstance(market, tuple) or len(market) != 2:
                raise ValueError(
                    f"market key must be (origin, destination), got {market!r}"
                )
            origin, destination = market
            if origin not in demand_nodes or destination not in demand_nodes:
                raise ValueError(f"market {market!r} references an unknown demand node")
            try:
                rate = float(raw_rate)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"demand for market {market!r} must be numeric"
                ) from exc
            if not math.isfinite(rate) or rate < 0.0:
                raise ValueError(
                    f"demand for market {market!r} must be finite and nonnegative"
                )
            if origin == destination and rate > 0.0:
                raise ValueError(f"self-market {market!r} must have zero demand")
            validated_demand[market] = rate
        self.h_od = validated_demand

        for market, raw_count in self.order_count.items():
            if market not in self.h_od:
                raise ValueError(f"order count supplied for unknown market {market!r}")
            if (
                isinstance(raw_count, bool)
                or int(raw_count) != raw_count
                or int(raw_count) < 0
            ):
                raise ValueError(
                    f"order count for market {market!r} must be a nonnegative integer"
                )
        self.order_count = {
            market: int(count) for market, count in self.order_count.items()
        }

        unknown_locker_costs = sorted(
            set(self.locker_pickup_cost_by_locker) - set(self.H)
        )
        if unknown_locker_costs:
            raise ValueError(
                "locker pickup costs supplied for unknown lockers: "
                f"{unknown_locker_costs}"
            )
        for locker, raw_cost in self.locker_pickup_cost_by_locker.items():
            try:
                cost = float(raw_cost)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"locker pickup cost for {locker!r} must be numeric"
                ) from exc
            if not math.isfinite(cost) or cost < 0.0:
                raise ValueError(
                    f"locker pickup cost for {locker!r} must be finite and nonnegative"
                )
            self.locker_pickup_cost_by_locker[locker] = cost

    def _build_virtual_states(self) -> None:
        self.S.clear()
        self.h_to_s.clear()
        self.s_to_h.clear()
        reserved_nodes = set(self.L) | set(self.H) | set(self.E) | set(self.N)
        for locker in self.H:
            state = f"S_{locker}"
            if state in reserved_nodes:
                raise ValueError(
                    f"generated virtual-state identifier {state!r} conflicts with an input node"
                )
            self.S.append(state)
            self.h_to_s[locker] = state
            self.s_to_h[state] = locker
            self.coords[state] = self.coords[locker]

    def _route_option_data(self, u: str, v: str) -> list[tuple[str, float, float]]:
        cfg = self.config
        direct_distance = self.distance(u, v)
        options: list[tuple[str, float, float]] = []
        if direct_distance <= cfg.battery_range_km + 1e-9:
            options.append(
                (
                    DIRECT_ROUTE,
                    direct_distance,
                    direct_distance / cfg.drone_speed_kmph,
                )
            )

        swap_options: list[tuple[str, float, float]] = []
        for station in self.E:
            first = self.distance(u, station)
            second = self.distance(station, v)
            total = first + second
            if (
                first <= cfg.battery_range_km + 1e-9
                and second <= cfg.battery_range_km + 1e-9
            ):
                swap_options.append(
                    (
                        station,
                        total,
                        total / cfg.drone_speed_kmph + cfg.swap_service_time_hours,
                    )
                )
        swap_options.sort(key=lambda item: (item[1], item[0]))
        return [*options, *swap_options]

    def _add_physical_arc(self, u: str, v: str, bucket: set[Arc]) -> None:
        options = self._route_option_data(u, v)
        if not options:
            return
        arc = (u, v)
        bucket.add(arc)
        self.route_options[arc] = tuple(option for option, _, _ in options)
        for option, distance, travel_time in options:
            self.route_distance[u, v, option] = distance
            self.route_time[u, v, option] = travel_time

    def _build_physical_network(self) -> None:
        self.A_LH.clear()
        self.A_HH.clear()
        self.A_HL.clear()
        self.A_SH.clear()
        self.A_HS.clear()
        self.route_options.clear()
        self.route_distance.clear()
        self.route_time.clear()

        for base in self.L:
            for locker in self.H:
                self._add_physical_arc(base, locker, self.A_LH)
        for origin in self.H:
            for destination in self.H:
                if origin != destination:
                    self._add_physical_arc(origin, destination, self.A_HH)
        for locker in self.H:
            for base in self.L:
                self._add_physical_arc(locker, base, self.A_HL)

        for locker in self.H:
            state = self.h_to_s[locker]
            self.A_HS.add((locker, state))
            self.A_SH.add((state, locker))

        self.A_F = self.A_LH | self.A_HH | self.A_HL
        self.A = self.A_F | self.A_HS | self.A_SH
        self.A_Y = set(self.A_F)
        self.arc_activation_cost = {
            arc: self.config.corridor_cost_per_km_hour * self.distance(*arc)
            for arc in self.A_F
        }
        swap_legs: set[Arc] = set()
        for (u, v), options in self.route_options.items():
            for option in options:
                if option == DIRECT_ROUTE:
                    continue
                for leg in ((u, option), (option, v)):
                    swap_legs.add(leg)
        self.A_Y.update(swap_legs)
        for leg in swap_legs:
            self.arc_activation_cost[leg] = (
                self.config.connection_cost_per_km_hour * self.distance(*leg)
            )

        for locker in self.H:
            sources = [base for base in self.L if (base, locker) in self.A_LH]
            sources.append(self.h_to_s[locker])
            self.eligible_idle[locker] = tuple(sources)

    def _build_customer_interface(self) -> None:
        self.display_lockers.clear()
        self.closer_lockers.clear()
        self.display_origins.clear()
        self.general_corridors.clear()
        self.choice_share.clear()

        markets_by_destination: dict[str, list[Market]] = {}
        for market in self.markets:
            markets_by_destination.setdefault(market[1], []).append(market)

        incoming_by_locker = {
            locker: tuple(origin for origin in self.H if (origin, locker) in self.A_HH)
            for locker in self.H
        }
        choice = self.config.choice
        for destination, destination_markets in sorted(markets_by_destination.items()):
            candidates = [
                locker
                for locker in self.H
                if choice.max_walk_km is None
                or self.distance(destination, locker) <= choice.max_walk_km + 1e-9
            ]
            candidates.sort(
                key=lambda locker: (self.distance(destination, locker), locker)
            )
            self.display_lockers[destination] = tuple(candidates)
            for locker in candidates:
                for origin, d_node in destination_markets:
                    ranked = sorted(
                        incoming_by_locker[locker],
                        key=lambda i: (self.distance(origin, i), i),
                    )
                    limit = self.config.display_origin_choices
                    if limit is not None:
                        ranked = ranked[:limit]
                    self.display_origins[origin, d_node, locker] = tuple(ranked)

                locker_distance = self.distance(destination, locker)
                self.closer_lockers[destination, locker] = tuple(
                    other
                    for other in self.H
                    if self.distance(destination, other) < locker_distance - 1e-9
                )
                utility_gap = (
                    choice.locker_asc
                    + choice.delivery_time_coefficient
                    * (choice.home_eta_minutes - choice.locker_eta_minutes)
                    + choice.price_coefficient * choice.locker_discount_cny
                    - choice.walk_distance_coefficient * locker_distance
                )
                share = _logistic(utility_gap)
                for origin, d_node in destination_markets:
                    self.choice_share[origin, d_node, locker] = share

        all_corridors = tuple(sorted(self.A_HH))
        for origin, destination in self.markets:
            selected = sorted(
                all_corridors,
                key=lambda ij: (
                    self.feeder_rider_cost(origin, ij[0])
                    + self.feeder_rider_cost(ij[1], destination),
                    ij,
                ),
            )
            limit = self.config.general_corridor_choices
            if limit is not None:
                selected = selected[:limit]
            self.general_corridors[origin, destination] = tuple(selected)

    def _build_bounds(self) -> None:
        # The loaded and source-flow identities bound all forward flow by
        # rho_max times total OD demand.  Summed flow balance over H union S
        # gives the same bound for every H-to-L return arc.  Independently,
        # every physical arc enters or leaves a locker, whose QoS constraint
        # limits throughput to max_c(mu_c - 1 / tau_H).
        demand_flow_bound = max(self.config.availability_levels) * self.total_demand
        capacity_flow_bound = max(
            service_rate - 1.0 / self.config.max_locker_time_hours
            for service_rate in self.config.locker_service_rates
        )
        # A nonpositive capacity bound means no locker can carry positive flow.
        # Clamping at zero avoids a negative big-M without weakening the model.
        flow_bound = max(0.0, min(demand_flow_bound, capacity_flow_bound))
        self.M_psi = {arc: flow_bound for arc in self.A_F}
        population_bound = (
            max(self.config.locker_service_rates) * self.config.max_locker_time_hours
        )
        self.M_xi = {locker: population_bound for locker in self.H}


def _logistic(value: float) -> float:
    """Evaluate a logistic probability without overflowing at extreme utilities."""

    if value >= 0.0:
        exponential = math.exp(-value)
        return 1.0 / (1.0 + exponential)
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)
