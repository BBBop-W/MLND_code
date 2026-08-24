"""Canonical OD-aware instance and derived physical-network relations."""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field
from typing import Any

from src.config import ModelConfig

Arc = tuple[str, str]
Market = tuple[str, str]
OrderCorridor = tuple[str, str, str, str]  # (o, i, j, d)
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
    route_distance: dict[tuple[str, str, str], float] = field(default_factory=dict, init=False)
    route_time: dict[tuple[str, str, str], float] = field(default_factory=dict, init=False)
    arc_activation_cost: dict[Arc, float] = field(default_factory=dict, init=False)
    eligible_idle: dict[str, tuple[str, ...]] = field(default_factory=dict, init=False)
    display_lockers: dict[str, tuple[str, ...]] = field(default_factory=dict, init=False)
    closer_lockers: dict[tuple[str, str], tuple[str, ...]] = field(default_factory=dict, init=False)
    locker_origins: dict[tuple[str, str, str], tuple[str, ...]] = field(default_factory=dict, init=False)
    home_corridors: dict[Market, tuple[Arc, ...]] = field(default_factory=dict, init=False)
    choice_share: dict[tuple[str, str, str], float] = field(default_factory=dict, init=False)
    locker_keys: list[OrderCorridor] = field(default_factory=list, init=False)
    home_keys: list[OrderCorridor] = field(default_factory=list, init=False)
    M_w: dict[Arc, float] = field(default_factory=dict, init=False)
    M_psi: dict[Arc, float] = field(default_factory=dict, init=False)
    M_xi: dict[str, float] = field(default_factory=dict, init=False)

    def build(self) -> "Instance":
        self.config.validate()
        self.L.sort()
        self.H.sort()
        self.E.sort()
        self.N.sort()
        self.h_od = {market: float(rate) for market, rate in self.h_od.items() if rate > 0}
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
        cfg = self.config
        return cfg.rider_base_cost + cfg.rider_cost_per_km * cfg.road_distance_factor * self.distance(u, v)

    def facility_cost(self, node: str) -> float:
        if node in self.L:
            return self.config.base_cost_per_hour
        if node in self.H:
            return self.config.locker_fixed_cost_per_hour
        return self.config.swap_cost_per_hour

    def endpoint_facility(self, node: str) -> str:
        return self.s_to_h.get(node, node)

    def _build_virtual_states(self) -> None:
        self.S.clear()
        self.h_to_s.clear()
        self.s_to_h.clear()
        for locker in self.H:
            state = f"S_{locker}"
            self.S.append(state)
            self.h_to_s[locker] = state
            self.s_to_h[state] = locker
            self.coords[state] = self.coords[locker]

    def _route_option_data(self, u: str, v: str) -> list[tuple[str, float, float]]:
        cfg = self.config
        direct_distance = self.distance(u, v)
        if direct_distance <= cfg.max_direct_range_km + 1e-9:
            # Every one-swap route is weakly longer by the triangle inequality
            # and has an additional positive swap-service time.  It also needs
            # two extra activated legs.  Replacing its flow by the direct option
            # therefore preserves feasibility and cannot increase any cost.
            return [
                (
                    DIRECT_ROUTE,
                    direct_distance,
                    direct_distance / cfg.drone_speed_kmph,
                )
            ]

        options: list[tuple[str, float, float]] = []
        for station in self.E:
            first = self.distance(u, station)
            second = self.distance(station, v)
            total = first + second
            if (
                first <= cfg.max_swap_leg_km + 1e-9
                and second <= cfg.max_swap_leg_km + 1e-9
            ):
                options.append(
                    (
                        station,
                        total,
                        total / cfg.drone_speed_kmph + cfg.swap_service_time_hours,
                    )
                )
        options.sort(key=lambda item: (item[1], item[0]))
        return options

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
        self.locker_origins.clear()
        self.home_corridors.clear()
        self.choice_share.clear()
        self.locker_keys.clear()
        self.home_keys.clear()

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
            candidates.sort(key=lambda locker: (self.distance(destination, locker), locker))
            valid_candidates: list[str] = []
            for locker in candidates:
                valid_for_all_markets = True
                for origin, d_node in destination_markets:
                    ranked = heapq.nsmallest(
                        self.config.origin_locker_choices or len(incoming_by_locker[locker]),
                        incoming_by_locker[locker],
                        key=lambda i: (self.distance(origin, i), i),
                    )
                    if not ranked:
                        valid_for_all_markets = False
                        break
                    self.locker_origins[origin, d_node, locker] = tuple(ranked)
                if valid_for_all_markets:
                    valid_candidates.append(locker)

            self.display_lockers[destination] = tuple(valid_candidates)
            for locker in valid_candidates:
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
                share = 1.0 / (1.0 + math.exp(-utility_gap))
                for origin, d_node in destination_markets:
                    self.choice_share[origin, d_node, locker] = share
                    for i in self.locker_origins[origin, d_node, locker]:
                        self.locker_keys.append((origin, i, locker, d_node))

        all_corridors = tuple(self.A_HH)
        for origin, destination in self.markets:
            selected = heapq.nsmallest(
                self.config.home_corridor_choices or len(all_corridors),
                all_corridors,
                key=lambda ij: (
                    self.rider_cost(origin, ij[0]) + self.rider_cost(ij[1], destination),
                    ij,
                ),
            )
            self.home_corridors[origin, destination] = tuple(selected)
            self.home_keys.extend((origin, i, j, destination) for i, j in selected)

        self.locker_keys.sort()
        self.home_keys.sort()

    def _build_bounds(self) -> None:
        corridor_demand: dict[Arc, float] = {arc: 0.0 for arc in self.A_HH}
        market_corridors: dict[Market, set[Arc]] = {market: set() for market in self.markets}
        for o, i, j, d in self.locker_keys:
            market_corridors[o, d].add((i, j))
        for o, i, j, d in self.home_keys:
            market_corridors[o, d].add((i, j))
        for market, corridors in market_corridors.items():
            for corridor in corridors:
                corridor_demand[corridor] += self.h_od[market]
        self.M_w = {arc: max(corridor_demand[arc], 1e-6) for arc in self.A_HH}
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
        population_bound = max(self.config.locker_service_rates) * self.config.max_locker_time_hours
        self.M_xi = {locker: population_bound for locker in self.H}
