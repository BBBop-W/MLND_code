"""Gurobi implementation of the unified chi/epsilon-energy MLND MISOCP."""

from __future__ import annotations

import math
import time
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass

import gurobipy as gp
from gurobipy import GRB

from src.config import SolverConfig
from src.data_processor.instance import DIRECT_ROUTE, Arc, Instance, Market
from src.model_contract import FORMULATION_ID
from src.model.primal_heuristic import (
    DEFAULT_STRUCTURE_GROUPS,
    LocalBranchingConfig,
    PrimalSeed,
    capture_incumbent,
    improve_mip_start,
)
from src.model.valid_inequalities import (
    ChoiceCorridorCoverData,
    build_choice_corridor_cover_data,
    queue_perspective_intercept,
    separate_choice_capacity_covers,
    separate_directed_return_cuts,
    separate_queue_perspective_cut,
    stable_arrival_capacity,
)

ChiKey = tuple[str, str, str, str, int]  # (o, d, i, j, r)
EpsilonKey = tuple[str, str, str]  # (o, d, j)
WKey = tuple[str, str, str, int]  # (idle, i, j, r)

_MAX_LO_NETWORK_START_ATTEMPTS = 5
_NETWORK_START_PARAMETER_NAMES = (
    "TimeLimit",
    "MIPGap",
    "MIPFocus",
    "Heuristics",
    "NoRelHeurTime",
)


@dataclass(frozen=True)
class _LOStartCandidate:
    """One capacity-safe locker-choice candidate for the LO start search."""

    alpha_key: tuple[str, str]
    offered_demand: float
    effective_min_arrival: float
    optimistic_saving: float


@dataclass(frozen=True)
class _NetworkStartAttempt:
    """Feasible incumbent captured from one temporary auxiliary solve."""

    seed: PrimalSeed
    objective: float
    values: Mapping[gp.Var, float]


@dataclass(frozen=True)
class _MIPStartState:
    """Exact snapshot of the MIP-start pool used around an auxiliary solve."""

    num_start: int
    start_number: int
    values: tuple[tuple[float, ...], ...]


class MLNDModel:
    """Implement the mixed, home-assistance, or locker-only MLND model."""

    FORMULATION_ID = FORMULATION_ID
    MODES = {"MX", "HD", "LO"}

    def __init__(self, instance: Instance, mode: str = "MX"):
        self.inst = instance
        self.mode = mode.upper()
        if self.mode not in self.MODES:
            raise ValueError(f"mode must be one of {sorted(self.MODES)}")
        self.model = gp.Model(f"MLND_{self.mode}_CHI_EPSILON_ENERGY")
        self.solver_config: SolverConfig | None = None
        self.vars: dict[str, object] = {}
        self.cost_expressions: dict[str, gp.LinExpr] = {}
        self.paper_objective_expressions: dict[str, gp.LinExpr] = {}
        self.courier_only_baseline = 0.0
        self.network_start_runtime = 0.0
        self.network_start_found = False
        self.network_start_used = False
        self.network_start_key: tuple[str, str] | None = None
        self.network_start_objective: float | None = None
        self.network_start_required_nodes: tuple[str, ...] = ()
        self.network_start_allowed_node_counts: dict[str, int] = {}
        self.network_start_seed_open_counts: dict[str, int] = {}
        self.network_start_candidate_count = 0
        self.network_start_capacity_rejected_count = 0
        self.network_start_attempted_keys: tuple[tuple[str, str], ...] = ()
        self.network_start_candidate_selection_seconds = 0.0
        self.network_partial_start_installed = False
        self.network_partial_start_index: int | None = None
        self.network_partial_start_variable_count = 0
        self.network_partial_start_groups = DEFAULT_STRUCTURE_GROUPS
        self.external_start_installed = False
        self.external_start_source: str | None = None
        self.external_start_matched_variables = 0
        self.main_solve_runtime = 0.0
        self.total_runtime = 0.0
        self.static_cut_counts: dict[str, int] = {}
        self.user_cut_counts: dict[str, int] = {}
        self.root_cut_rounds = 0
        self.local_branching_runtime = 0.0
        self.local_branching_improved = False
        self.local_branching_objective: float | None = None
        self.local_branching_status: int | None = None
        self.local_branching_solution_count = 0
        self.local_branching_hamming_distance: int | None = None
        self._valid_inequalities_added = False
        self._user_cut_signatures: set[tuple[object, ...]] = set()
        self._choice_cover_data: tuple[ChoiceCorridorCoverData, ...] = ()
        self._built = False
        self._prepare_indices()

    def _prepare_indices(self) -> None:
        """Build the sparse OD-corridor-reliability index sets for one mode."""

        level_count = len(self.inst.config.availability_levels)
        self.market_corridors: dict[Market, tuple[Arc, ...]] = {}
        chi_keys: list[ChiKey] = []
        self.chi_by_market: dict[Market, list[ChiKey]] = defaultdict(list)
        self.chi_by_market_j: dict[EpsilonKey, list[ChiKey]] = defaultdict(list)
        self.chi_by_corridor: dict[Arc, list[ChiKey]] = defaultdict(list)
        self.chi_by_corridor_level: dict[tuple[str, str, int], list[ChiKey]] = (
            defaultdict(list)
        )

        for market in self.inst.markets:
            corridors = tuple(
                dict.fromkeys(self.inst.corridors_for_mode(self.mode, market))
            )
            invalid = [arc for arc in corridors if arc not in self.inst.A_HH]
            if invalid:
                raise ValueError(
                    f"mode corridor set contains non-HH arcs: {invalid[:3]}"
                )
            self.market_corridors[market] = corridors
            o, d = market
            for i, j in corridors:
                for r in range(level_count):
                    key = (o, d, i, j, r)
                    chi_keys.append(key)
                    self.chi_by_market[market].append(key)
                    self.chi_by_market_j[o, d, j].append(key)
                    self.chi_by_corridor[i, j].append(key)
                    self.chi_by_corridor_level[i, j, r].append(key)
        self.chi_keys = tuple(chi_keys)
        self.corridors = tuple(sorted(self.chi_by_corridor))

        self.alpha_keys: tuple[tuple[str, str], ...] = ()
        if self.mode in {"MX", "LO"}:
            self.alpha_keys = tuple(
                (destination, locker)
                for destination in sorted(self.inst.display_lockers)
                for locker in self.inst.display_lockers[destination]
            )
        self.epsilon_keys: tuple[EpsilonKey, ...] = ()
        if self.mode == "MX":
            self.epsilon_keys = tuple(
                (o, d, locker)
                for o, d in self.inst.markets
                for locker in self.inst.display_lockers.get(d, ())
            )

        rho_min = min(self.inst.config.availability_levels)
        self.M_offer: dict[Arc, float] = {}
        for arc in self.corridors:
            raw_bound = 0.0
            for o, d in self.inst.markets:
                if arc not in self.market_corridors[o, d]:
                    continue
                if self.mode == "LO":
                    raw_bound += self.inst.h_od[o, d] * self.inst.choice_share.get(
                        (o, d, arc[1]),
                        1.0,
                    )
                else:
                    raw_bound += self.inst.h_od[o, d]
            realized_bound = self.inst.M_psi.get(
                arc,
                max(self.inst.M_psi.values(), default=0.0),
            )
            self.M_offer[arc] = max(0.0, min(raw_bound, realized_bound / rho_min))

        self.w_keys: tuple[WKey, ...] = tuple(
            (idle, i, j, r)
            for i, j in self.corridors
            for idle in self.inst.eligible_idle[i]
            for r in range(level_count)
        )
        self.w_by_source: dict[tuple[str, str], list[WKey]] = defaultdict(list)
        self.w_by_corridor: dict[Arc, list[WKey]] = defaultdict(list)
        self.w_by_corridor_level: dict[tuple[str, str, int], list[WKey]] = defaultdict(
            list
        )
        for key in self.w_keys:
            idle, i, j, r = key
            self.w_by_source[idle, i].append(key)
            self.w_by_corridor[i, j].append(key)
            self.w_by_corridor_level[i, j, r].append(key)

        self.varphi_keys = tuple(
            (u, v, option)
            for u, v in sorted(self.inst.A_F)
            for option in self.inst.route_options[u, v]
        )

    def _successful_market_expression(
        self,
        chi: gp.tupledict,
        market: Market,
    ) -> gp.LinExpr:
        levels = self.inst.config.availability_levels
        return gp.quicksum(
            levels[key[4]] * chi[key] for key in self.chi_by_market[market]
        )

    def _successful_market_locker_expression(
        self,
        chi: gp.tupledict,
        key: EpsilonKey,
    ) -> gp.LinExpr:
        levels = self.inst.config.availability_levels
        return gp.quicksum(
            levels[item[4]] * chi[item] for item in self.chi_by_market_j[key]
        )

    def build(self) -> "MLNDModel":
        if self._built:
            return self

        cfg = self.inst.config
        capacities = range(len(cfg.locker_service_rates))
        levels = range(len(cfg.availability_levels))
        physical_nodes = self.inst.L + self.inst.H + self.inst.E
        balance_nodes = self.inst.L + self.inst.H + self.inst.S

        x = self.model.addVars(physical_nodes, vtype=GRB.BINARY, name="x")
        z = self.model.addVars(self.inst.H, capacities, vtype=GRB.BINARY, name="z")
        y = self.model.addVars(sorted(self.inst.A_Y), vtype=GRB.BINARY, name="y")
        delta = self.model.addVars(
            [(idle, r) for idle in self.inst.idle_states for r in levels],
            vtype=GRB.BINARY,
            name="delta",
        )
        alpha = self.model.addVars(self.alpha_keys, vtype=GRB.BINARY, name="alpha")
        gamma = self.model.addVar(lb=0.0, vtype=GRB.INTEGER, name="gamma")

        chi = self.model.addVars(self.chi_keys, lb=0.0, name="chi")
        epsilon = self.model.addVars(self.epsilon_keys, lb=0.0, name="epsilon")
        w = self.model.addVars(self.w_keys, lb=0.0, name="w")
        psi = self.model.addVars(sorted(self.inst.A), lb=0.0, name="psi")
        varphi = self.model.addVars(self.varphi_keys, lb=0.0, name="varphi")
        xi = self.model.addVars(self.inst.H, lb=0.0, name="xi")

        facility_cost = gp.quicksum(
            self.inst.facility_cost(node) * x[node] for node in physical_nodes
        )
        capacity_cost = gp.quicksum(
            cfg.locker_capacity_costs[c] * z[i, c]
            for i in self.inst.H
            for c in capacities
        )
        connection_cost = gp.quicksum(
            self.inst.arc_activation_cost[arc] * y[arc] for arc in sorted(self.inst.A_Y)
        )
        route_operation_cost = gp.quicksum(
            cfg.flight_cost_per_km * self.inst.route_distance[key] * varphi[key]
            for key in self.varphi_keys
        )
        fleet_cost = cfg.drone_cost_per_hour * gamma

        direct_courier_fallback = gp.LinExpr()
        for market in self.inst.markets:
            o, d = market
            direct_courier_fallback += self.inst.direct_rider_cost(o, d) * (
                self.inst.h_od[market] - self._successful_market_expression(chi, market)
            )
        first_mile_courier = gp.quicksum(
            self.inst.feeder_rider_cost(o, i)
            * cfg.availability_levels[r]
            * chi[o, d, i, j, r]
            for o, d, i, j, r in self.chi_keys
        )

        last_mile_courier = gp.LinExpr()
        if self.mode in {"MX", "HD"}:
            last_mile_courier += gp.quicksum(
                self.inst.feeder_rider_cost(j, d)
                * cfg.availability_levels[r]
                * chi[o, d, i, j, r]
                for o, d, i, j, r in self.chi_keys
            )
        if self.mode == "MX":
            last_mile_courier -= gp.quicksum(
                self.inst.choice_share[o, d, j]
                * self.inst.feeder_rider_cost(j, d)
                * epsilon[o, d, j]
                for o, d, j in self.epsilon_keys
            )

        locker_handling = gp.LinExpr()
        if self.mode == "MX":
            locker_handling += gp.quicksum(
                self.inst.choice_share[o, d, j]
                * self.inst.locker_pickup_cost(j)
                * epsilon[o, d, j]
                for o, d, j in self.epsilon_keys
            )
        elif self.mode == "LO":
            locker_handling += gp.quicksum(
                self.inst.locker_pickup_cost(j)
                * cfg.availability_levels[r]
                * chi[o, d, i, j, r]
                for o, d, i, j, r in self.chi_keys
            )

        self.cost_expressions = {
            "facility": facility_cost,
            "capacity": capacity_cost,
            "connection": connection_cost,
            "route_operation": route_operation_cost,
            "fleet": fleet_cost,
            "direct_courier_fallback": direct_courier_fallback,
            "first_mile_courier": first_mile_courier,
            "last_mile_courier": last_mile_courier,
            "locker_handling": locker_handling,
        }

        self.courier_only_baseline = sum(
            self.inst.direct_rider_cost(o, d) * self.inst.h_od[o, d]
            for o, d in self.inst.markets
        )
        courier_baseline = gp.LinExpr(self.courier_only_baseline)
        if self.mode == "LO":
            realized_adjustment = gp.quicksum(
                cfg.availability_levels[r]
                * (
                    self.inst.feeder_rider_cost(o, i)
                    + self.inst.locker_pickup_cost(j)
                    - self.inst.direct_rider_cost(o, d)
                )
                * chi[o, d, i, j, r]
                for o, d, i, j, r in self.chi_keys
            )
            locker_adjustment = gp.LinExpr()
        else:
            realized_adjustment = gp.quicksum(
                cfg.availability_levels[r]
                * (
                    self.inst.feeder_rider_cost(o, i)
                    + self.inst.feeder_rider_cost(j, d)
                    - self.inst.direct_rider_cost(o, d)
                )
                * chi[o, d, i, j, r]
                for o, d, i, j, r in self.chi_keys
            )
            locker_adjustment = gp.LinExpr()
            if self.mode == "MX":
                locker_adjustment += gp.quicksum(
                    self.inst.choice_share[o, d, j]
                    * (
                        self.inst.locker_pickup_cost(j)
                        - self.inst.feeder_rider_cost(j, d)
                    )
                    * epsilon[o, d, j]
                    for o, d, j in self.epsilon_keys
                )
        self.paper_objective_expressions = {
            "courier_baseline": courier_baseline,
            "realized_drone_adjustment": realized_adjustment,
            "locker_pickup_adjustment": locker_adjustment,
        }
        self.model.setObjective(
            gp.quicksum(self.cost_expressions.values()), GRB.MINIMIZE
        )

        for i in self.inst.H:
            self.model.addConstr(
                gp.quicksum(z[i, c] for c in capacities) == x[i],
                name=f"capacity_select[{i}]",
            )
        for u, v in sorted(self.inst.A_Y):
            self.model.addConstr(y[u, v] <= x[u], name=f"arc_tail[{u},{v}]")
            self.model.addConstr(y[u, v] <= x[v], name=f"arc_head[{u},{v}]")

        if self.alpha_keys:
            alpha_by_destination: dict[str, list[str]] = defaultdict(list)
            for destination, locker in self.alpha_keys:
                alpha_by_destination[destination].append(locker)
            for destination, lockers in sorted(alpha_by_destination.items()):
                self.model.addConstr(
                    gp.quicksum(alpha[destination, locker] for locker in lockers) <= 1,
                    name=f"display_limit[{destination}]",
                )
                for locker in lockers:
                    self.model.addConstr(
                        alpha[destination, locker] <= x[locker],
                        name=f"display_open[{destination},{locker}]",
                    )
                    for closer in self.inst.closer_lockers[destination, locker]:
                        self.model.addConstr(
                            alpha[destination, locker] <= 1 - x[closer],
                            name=f"display_nearest[{destination},{locker},{closer}]",
                        )

        for market in self.inst.markets:
            o, d = market
            self.model.addConstr(
                gp.quicksum(chi[key] for key in self.chi_by_market[market])
                <= self.inst.h_od[market],
                name=f"od_demand_bound[{o},{d}]",
            )
        if self.mode == "LO":
            for o, d in self.inst.markets:
                for locker in self.inst.display_lockers.get(d, ()):
                    self.model.addConstr(
                        gp.quicksum(
                            chi[key] for key in self.chi_by_market_j[o, d, locker]
                        )
                        == self.inst.h_od[o, d]
                        * self.inst.choice_share[o, d, locker]
                        * alpha[d, locker],
                        name=f"lo_locker_allocation[{o},{d},{locker}]",
                    )

        for i, j in self.corridors:
            for r in levels:
                self.model.addConstr(
                    gp.quicksum(chi[key] for key in self.chi_by_corridor_level[i, j, r])
                    == gp.quicksum(w[key] for key in self.w_by_corridor_level[i, j, r]),
                    name=f"od_reliability_match[{i},{j},{r}]",
                )
            self.model.addConstr(
                gp.quicksum(w[key] for key in self.w_by_corridor[i, j])
                <= self.M_offer[i, j] * y[i, j],
                name=f"corridor_offer_activation[{i},{j}]",
            )

        for base in self.inst.L:
            self.model.addConstr(
                gp.quicksum(delta[base, r] for r in levels) == x[base],
                name=f"availability_base[{base}]",
            )
        for locker in self.inst.H:
            state = self.inst.h_to_s[locker]
            self.model.addConstr(
                gp.quicksum(delta[state, r] for r in levels) <= x[locker],
                name=f"availability_virtual[{state}]",
            )
        for idle, i, j, r in self.w_keys:
            self.model.addConstr(
                w[idle, i, j, r] <= self.M_offer[i, j] * delta[idle, r],
                name=f"w_reliability[{idle},{i},{j},{r}]",
            )

        for i in self.inst.H:
            for idle in self.inst.eligible_idle[i]:
                self.model.addConstr(
                    psi[idle, i]
                    == gp.quicksum(
                        cfg.availability_levels[key[3]] * w[key]
                        for key in self.w_by_source.get((idle, i), ())
                    ),
                    name=f"source_flow[{idle},{i}]",
                )
        for i, j in sorted(self.inst.A_HH):
            self.model.addConstr(
                psi[i, j]
                == gp.quicksum(
                    cfg.availability_levels[key[3]] * w[key]
                    for key in self.w_by_corridor.get((i, j), ())
                ),
                name=f"loaded_flow[{i},{j}]",
            )

        for node in balance_nodes:
            self.model.addConstr(
                gp.quicksum(psi[u, v] for u, v in self.inst.A if v == node)
                == gp.quicksum(psi[u, v] for u, v in self.inst.A if u == node),
                name=f"flow_balance[{node}]",
            )
        for arc in sorted(self.inst.A_F):
            self.model.addConstr(
                psi[arc] <= self.inst.M_psi[arc] * y[arc],
                name=f"physical_flow[{arc[0]},{arc[1]}]",
            )

        for u, v in sorted(self.inst.A_F):
            self.model.addConstr(
                gp.quicksum(
                    varphi[u, v, option] for option in self.inst.route_options[u, v]
                )
                == psi[u, v],
                name=f"route_decomposition[{u},{v}]",
            )
            for option in self.inst.route_options[u, v]:
                if option == DIRECT_ROUTE:
                    continue
                upper = self.inst.M_psi[u, v]
                self.model.addConstr(
                    varphi[u, v, option] <= upper * y[u, option],
                    name=f"route_swap_in[{u},{v},{option}]",
                )
                self.model.addConstr(
                    varphi[u, v, option] <= upper * y[option, v],
                    name=f"route_swap_out[{u},{v},{option}]",
                )

        # The paper uses (1-epsilon) * Omega / omega. The confirmed usable
        # range and speed identify this ratio directly, so no arbitrary energy
        # scale is introduced into the implementation. Every departure from a
        # physical base starts with one full battery, and every one-swap route
        # supplies one additional full battery. A system-wide balance lets that
        # energy support subsequent direct movements in the closed network.
        usable_battery_time = cfg.usable_battery_time_hours
        energy_use = gp.quicksum(xi[i] for i in self.inst.H) + gp.quicksum(
            self.inst.route_time[key] * varphi[key] for key in self.varphi_keys
        )
        base_battery_departures = gp.quicksum(
            varphi[u, v, option]
            for u, v in sorted(self.inst.A_LH)
            for option in self.inst.route_options[u, v]
        )
        swap_events = gp.quicksum(
            varphi[u, v, option]
            for u, v, option in self.varphi_keys
            if option != DIRECT_ROUTE
        )
        self.model.addConstr(
            energy_use <= usable_battery_time * (base_battery_departures + swap_events),
            name="energy_system",
        )

        queue_v1 = self.model.addVars(self.inst.H, lb=-GRB.INFINITY, name="queue_v1")
        queue_v2 = self.model.addVars(self.inst.H, lb=-GRB.INFINITY, name="queue_v2")
        queue_t = self.model.addVars(self.inst.H, lb=0.0, name="queue_t")
        for i in self.inst.H:
            incoming = gp.quicksum(psi[u, v] for u, v in self.inst.A if v == i)
            service_rate = gp.quicksum(
                cfg.locker_service_rates[c] * z[i, c] for c in capacities
            )
            self.model.addConstr(
                service_rate - incoming >= x[i] / cfg.max_locker_time_hours,
                name=f"locker_qos[{i}]",
            )
            self.model.addConstr(
                xi[i] <= self.inst.M_xi[i] * x[i],
                name=f"locker_population[{i}]",
            )
            self.model.addConstr(
                queue_v1[i]
                == 2.0
                * gp.quicksum(
                    math.sqrt(cfg.locker_service_rates[c]) * z[i, c] for c in capacities
                ),
                name=f"queue_v1_definition[{i}]",
            )
            self.model.addConstr(
                queue_v2[i] == xi[i] + incoming - service_rate + 1.0,
                name=f"queue_v2_definition[{i}]",
            )
            self.model.addConstr(
                queue_t[i] == xi[i] - incoming + service_rate + 1.0,
                name=f"queue_t_definition[{i}]",
            )
            self.model.addQConstr(
                queue_v1[i] * queue_v1[i] + queue_v2[i] * queue_v2[i]
                <= queue_t[i] * queue_t[i],
                name=f"locker_queue_soc[{i}]",
            )
        for locker in self.inst.H:
            state = self.inst.h_to_s[locker]
            self.model.addConstr(
                gp.quicksum(
                    cfg.availability_levels[r]
                    / (1.0 - cfg.availability_levels[r])
                    * delta[state, r]
                    for r in levels
                )
                <= cfg.max_virtual_wait_hours * psi[state, locker],
                name=f"holding_qos[{state}]",
            )
        self.model.addConstr(
            gp.quicksum(
                cfg.availability_levels[r]
                / (1.0 - cfg.availability_levels[r])
                * delta[idle, r]
                for idle in self.inst.idle_states
                for r in levels
            )
            + gp.quicksum(xi[i] for i in self.inst.H)
            + gp.quicksum(
                self.inst.route_time[key] * varphi[key] for key in self.varphi_keys
            )
            <= gamma,
            name="fleet_size",
        )

        if self.mode == "MX":
            rho_max = max(cfg.availability_levels)
            for o, d, locker in self.epsilon_keys:
                successful = self._successful_market_locker_expression(
                    chi,
                    (o, d, locker),
                )
                big_m = rho_max * self.inst.h_od[o, d]
                self.model.addConstr(
                    epsilon[o, d, locker] <= successful,
                    name=f"epsilon_flow[{o},{d},{locker}]",
                )
                self.model.addConstr(
                    epsilon[o, d, locker] <= big_m * alpha[d, locker],
                    name=f"epsilon_display[{o},{d},{locker}]",
                )
                self.model.addConstr(
                    epsilon[o, d, locker]
                    >= successful - big_m * (1.0 - alpha[d, locker]),
                    name=f"epsilon_lower[{o},{d},{locker}]",
                )

        self.vars = {
            "x": x,
            "z": z,
            "y": y,
            "delta": delta,
            "alpha": alpha,
            "gamma": gamma,
            "chi": chi,
            "epsilon": epsilon,
            "w": w,
            "psi": psi,
            "varphi": varphi,
            "xi": xi,
            "queue_v1": queue_v1,
            "queue_v2": queue_v2,
            "queue_t": queue_t,
        }
        self._built = True
        self.model.update()
        return self

    def prepare_algorithm(self, solver: SolverConfig | None = None) -> "MLNDModel":
        """Add mode-valid strengthening rows before branch-and-cut starts."""

        self.build()
        solver = solver or SolverConfig()
        solver.validate()
        if self.mode == "LO" and self.alpha_keys and not self._choice_cover_data:
            cover_alpha_keys = {
                (destination, locker)
                for destination, locker in self.alpha_keys
                if all(
                    self.inst.display_origins.get((origin, d_node, locker), ())
                    for origin, d_node in self.inst.markets
                    if d_node == destination
                )
            }
            self._choice_cover_data = build_choice_corridor_cover_data(
                self.inst.h_od,
                {
                    key: self.inst.choice_share[key]
                    for key in self.inst.display_origins
                    if (key[1], key[2]) in cover_alpha_keys
                },
                {
                    key: origins
                    for key, origins in self.inst.display_origins.items()
                    if (key[1], key[2]) in cover_alpha_keys
                },
            )
        if solver.use_valid_inequalities and not self._valid_inequalities_added:
            self._add_static_valid_inequalities()
            self._valid_inequalities_added = True
            self.model.update()
        return self

    def _add_static_valid_inequalities(self) -> None:
        """Add compact projections that remain valid for the selected mode."""

        cfg = self.inst.config
        z = self.vars["z"]
        y = self.vars["y"]
        delta = self.vars["delta"]
        alpha = self.vars["alpha"]
        w = self.vars["w"]
        psi = self.vars["psi"]
        xi = self.vars["xi"]
        counts: dict[str, int] = defaultdict(int)
        rho_min = min(cfg.availability_levels)
        rho_max = max(cfg.availability_levels)
        max_locker_flow = max(
            0.0,
            min(
                rho_max * self.inst.total_demand,
                max(
                    rate - 1.0 / cfg.max_locker_time_hours
                    for rate in cfg.locker_service_rates
                ),
            ),
        )

        for locker in self.inst.H:
            for idle in self.inst.eligible_idle[locker]:
                source_keys = self.w_by_source.get((idle, locker), ())
                for r, rho in enumerate(cfg.availability_levels):
                    offered = gp.quicksum(w[key] for key in source_keys if key[3] == r)
                    self.model.addConstr(
                        rho * offered <= max_locker_flow * delta[idle, r],
                        name=f"vi_source_level_upper[{idle},{locker},{r}]",
                    )
                    counts["source_level_upper"] += 1
                    if idle in self.inst.S:
                        lower = 1.0 / ((1.0 - rho) * cfg.max_virtual_wait_hours)
                        self.model.addConstr(
                            offered >= lower * delta[idle, r],
                            name=f"vi_virtual_level_lower[{idle},{r}]",
                        )
                        counts["virtual_level_lower"] += 1

        capacity_slack = tuple(
            rate - 1.0 / cfg.max_locker_time_hours for rate in cfg.locker_service_rates
        )
        if self.mode == "LO":
            display_volume: dict[tuple[str, str], float] = {}
            display_arcs: dict[tuple[str, str], set[Arc]] = defaultdict(set)
            for destination, locker in self.alpha_keys:
                volume = 0.0
                for origin, d_node in self.inst.markets:
                    if d_node != destination:
                        continue
                    volume += (
                        self.inst.h_od[origin, destination]
                        * self.inst.choice_share[origin, destination, locker]
                    )
                    display_arcs[destination, locker].update(
                        (i, locker)
                        for i in self.inst.display_origins.get(
                            (origin, destination, locker),
                            (),
                        )
                        if (i, locker) in self.M_offer
                    )
                display_volume[destination, locker] = volume
                if volume <= 0.0:
                    continue
                self.model.addConstr(
                    gp.quicksum(
                        min(self.M_offer[arc], volume) * y[arc]
                        for arc in display_arcs[destination, locker]
                    )
                    >= volume * alpha[destination, locker],
                    name=f"vi_lo_choice_corridor_cover[{destination},{locker}]",
                )
                counts["lo_choice_corridor_cover"] += 1

            for locker in self.inst.H:
                forced = gp.quicksum(
                    display_volume[destination, candidate]
                    * alpha[destination, candidate]
                    for destination, candidate in self.alpha_keys
                    if candidate == locker
                )
                self.model.addConstr(
                    rho_min * forced
                    <= gp.quicksum(
                        capacity_slack[c] * z[locker, c]
                        for c in range(len(capacity_slack))
                    ),
                    name=f"vi_lo_choice_capacity[{locker}]",
                )
                counts["lo_choice_capacity_projection"] += 1

        queue_slopes = sorted(
            {
                slope
                for rate, slack in zip(cfg.locker_service_rates, capacity_slack)
                for slope in (
                    1.0 / rate,
                    rate / (rate - max(0.0, slack)) ** 2,
                )
                if slope > 0.0 and math.isfinite(slope)
            }
        )
        for locker in self.inst.H:
            incoming = gp.quicksum(psi[u, v] for u, v in self.inst.A if v == locker)
            for slope_index, slope in enumerate(queue_slopes):
                intercepts = tuple(
                    queue_perspective_intercept(
                        rate,
                        stable_arrival_capacity(rate, cfg.max_locker_time_hours),
                        slope,
                    )
                    for rate in cfg.locker_service_rates
                )
                self.model.addConstr(
                    xi[locker]
                    >= slope * incoming
                    + gp.quicksum(
                        intercepts[c] * z[locker, c] for c in range(len(intercepts))
                    ),
                    name=f"vi_queue_perspective[{locker},{slope_index}]",
                )
                counts["queue_perspective_support"] += 1

        self.static_cut_counts = dict(sorted(counts.items()))

    @staticmethod
    def _callback_values(
        callback_model: gp.Model,
        container: gp.tupledict,
    ) -> dict[object, float]:
        """Read one tupledict from a MIPNODE relaxation in one bulk call."""

        items = list(container.items())
        if not items:
            return {}
        values = callback_model.cbGetNodeRel([variable for _, variable in items])
        return {key: float(value) for (key, _), value in zip(items, values)}

    def _add_root_user_cuts(
        self,
        callback_model: gp.Model,
        solver: SolverConfig,
    ) -> int:
        """Separate one bounded round of globally valid root-node cuts."""

        tolerance = solver.cut_violation_tolerance
        budget = solver.max_cuts_per_round
        if budget <= 0:
            return 0

        cfg = self.inst.config
        chi = self.vars["chi"]
        alpha = self.vars["alpha"]
        w = self.vars["w"]
        psi = self.vars["psi"]
        y = self.vars["y"]
        z = self.vars["z"]
        xi = self.vars["xi"]

        y_values = self._callback_values(callback_model, y)
        counts: dict[str, int] = defaultdict(int)
        added = 0

        chi_values = self._callback_values(callback_model, chi)
        vub_candidates: list[
            tuple[float, tuple[object, ...], gp.Var, float, gp.Var]
        ] = []
        for key, value in chi_values.items():
            o, d, i, j, _r = key
            coefficient = self.inst.h_od[o, d]
            if self.mode == "LO":
                coefficient *= self.inst.choice_share[o, d, j]
            violation = value - coefficient * y_values[i, j]
            signature = ("chi_od_vub", key)
            if violation > tolerance and signature not in self._user_cut_signatures:
                vub_candidates.append(
                    (violation, signature, chi[key], coefficient, y[i, j])
                )
        vub_candidates.sort(key=lambda item: (-item[0], item[1]))
        vub_quota = max(1, budget // 2)
        for _, signature, offer, coefficient, corridor_open in vub_candidates[
            :vub_quota
        ]:
            callback_model.cbCut(offer <= coefficient * corridor_open)
            self._user_cut_signatures.add(signature)
            counts["chi_od_vub"] += 1
            added += 1

        remaining = budget - added
        queue_quota = min(remaining, max(1, budget // 6))
        queue_added = 0
        if queue_quota > 0:
            psi_values = self._callback_values(callback_model, psi)
            xi_values = self._callback_values(callback_model, xi)
            z_values = self._callback_values(callback_model, z)
            for locker in self.inst.H:
                if queue_added >= queue_quota:
                    break
                arrival_value = sum(
                    psi_values[u, v] for u, v in self.inst.A if v == locker
                )
                capacity_values = tuple(
                    z_values[locker, c] for c in range(len(cfg.locker_service_rates))
                )
                weighted_capacity = sum(
                    stable_arrival_capacity(rate, cfg.max_locker_time_hours)
                    * capacity_values[c]
                    for c, rate in enumerate(cfg.locker_service_rates)
                )
                if arrival_value > weighted_capacity + 10.0 * tolerance:
                    continue
                cut = separate_queue_perspective_cut(
                    arrival_value,
                    xi_values[locker],
                    capacity_values,
                    cfg.locker_service_rates,
                    cfg.max_locker_time_hours,
                    violation_tolerance=tolerance,
                    feasibility_tolerance=max(1e-9, 10.0 * tolerance),
                )
                if cut is None:
                    continue
                signature = ("queue", locker, *cut.signature())
                if signature in self._user_cut_signatures:
                    continue
                incoming = gp.quicksum(psi[u, v] for u, v in self.inst.A if v == locker)
                callback_model.cbCut(
                    xi[locker]
                    >= cut.lambda_coefficient * incoming
                    + gp.quicksum(
                        cut.z_coefficients[c] * z[locker, c]
                        for c in range(len(cut.z_coefficients))
                    )
                )
                self._user_cut_signatures.add(signature)
                counts["queue_perspective"] += 1
                added += 1
                queue_added += 1

        if self.mode == "LO" and added < budget and self._choice_cover_data:
            alpha_values = self._callback_values(callback_model, alpha)
            z_values = self._callback_values(callback_model, z)
            cover_cuts = separate_choice_capacity_covers(
                self._choice_cover_data,
                tuple(
                    stable_arrival_capacity(rate, cfg.max_locker_time_hours)
                    for rate in cfg.locker_service_rates
                ),
                min(cfg.availability_levels),
                alpha_values,
                z_values,
                violation_tolerance=tolerance,
                feasibility_tolerance=max(1e-8, 10.0 * tolerance),
                max_cuts_per_locker=2,
                seen_signatures=self._user_cut_signatures,
            )
            cover_quota = min(budget - added, max(1, budget // 5))
            for cut in cover_cuts[:cover_quota]:
                callback_model.cbCut(
                    gp.quicksum(alpha[d, cut.locker] for d in cut.destinations)
                    <= len(cut.destinations)
                    - 1
                    + gp.quicksum(
                        z[cut.locker, c] for c in cut.adequate_capacity_indices
                    )
                )
                self._user_cut_signatures.add(cut.signature())
                counts["lo_choice_capacity_cover"] += 1
                added += 1

        if added < budget:
            w_values = self._callback_values(callback_model, w)
            offered_values = {
                arc: sum(w_values[key] for key in self.w_by_corridor[arc])
                for arc in self.corridors
                if self.M_offer[arc] > 0.0
            }
            return_cuts = separate_directed_return_cuts(
                self.inst.A_F,
                {arc: y_values[arc] for arc in self.inst.A_F},
                offered_values,
                {arc: self.M_offer[arc] for arc in offered_values},
                contraction=self.inst.s_to_h,
                violation_tolerance=tolerance,
                feasibility_tolerance=max(1e-8, 10.0 * tolerance),
                max_candidates=min(100, len(offered_values)),
                max_cuts=budget - added,
                seen_signatures=self._user_cut_signatures,
            )
            for cut in return_cuts:
                callback_model.cbCut(
                    gp.quicksum(w[key] for key in self.w_by_corridor[cut.corridor])
                    <= cut.big_m * gp.quicksum(y[arc] for arc in cut.crossing_arcs)
                )
                self._user_cut_signatures.add(cut.signature())
                counts["directed_return"] += 1
                added += 1

        for family, count in counts.items():
            self.user_cut_counts[family] = self.user_cut_counts.get(family, 0) + count
        return added

    def _root_user_cut_callback(self, solver: SolverConfig):
        """Create the root-only callback used by the branch-and-cut solve."""

        def callback(callback_model: gp.Model, where: int) -> None:
            if where != GRB.Callback.MIPNODE:
                return
            if self.root_cut_rounds >= solver.max_root_cut_rounds:
                return
            if callback_model.cbGet(GRB.Callback.MIPNODE_STATUS) != GRB.OPTIMAL:
                return
            if callback_model.cbGet(GRB.Callback.MIPNODE_NODCNT) > 0.5:
                return
            self.root_cut_rounds += 1
            self._add_root_user_cuts(callback_model, solver)

        return callback

    def solve(
        self,
        solver: SolverConfig | None = None,
        *,
        external_start: Mapping[str, float] | None = None,
        external_start_source: str | None = None,
    ) -> bool:
        solver = solver or SolverConfig()
        self.solver_config = solver
        self.prepare_algorithm(solver)
        self.network_start_runtime = 0.0
        self.network_start_found = False
        self.network_start_used = False
        self.network_start_key = None
        self.network_start_objective = None
        self.network_start_required_nodes = ()
        self.network_start_allowed_node_counts = {}
        self.network_start_seed_open_counts = {}
        self.network_start_candidate_count = 0
        self.network_start_capacity_rejected_count = 0
        self.network_start_attempted_keys = ()
        self.network_start_candidate_selection_seconds = 0.0
        self.network_partial_start_installed = False
        self.network_partial_start_index = None
        self.network_partial_start_variable_count = 0
        self.external_start_installed = False
        self.external_start_source = None
        self.external_start_matched_variables = 0
        self.local_branching_runtime = 0.0
        self.local_branching_improved = False
        self.local_branching_objective = None
        self.local_branching_status = None
        self.local_branching_solution_count = 0
        self.local_branching_hamming_distance = None
        self.main_solve_runtime = 0.0
        self.total_runtime = 0.0
        self.user_cut_counts = {}
        self.root_cut_rounds = 0
        self._user_cut_signatures.clear()
        self.model.Params.TimeLimit = solver.time_limit
        self.model.Params.MIPGap = solver.mip_gap
        self.model.Params.Threads = solver.threads
        self.model.Params.OutputFlag = solver.output_flag
        self.model.Params.Seed = solver.seed
        self.model.Params.Method = solver.root_relaxation_method
        self.model.Params.MIQCPMethod = solver.miqcp_method
        start_budget = min(solver.network_start_seconds, 0.20 * solver.time_limit)
        if start_budget > 1e-9 and self.chi_keys:
            self._construct_network_start(start_budget, solver)
        else:
            self._set_courier_only_start()
        if external_start is not None:
            self._append_external_start(external_start, external_start_source)
        remaining = max(0.1, solver.time_limit - self.network_start_runtime)
        self.model.Params.TimeLimit = remaining
        self.model.Params.MIPGap = solver.mip_gap
        # Limit native root cut passes only for the full solve.  The compact
        # auxiliary network-start model benefits from Gurobi's automatic cut
        # policy and should not inherit this large-model time safeguard.
        self.model.Params.CutPasses = solver.root_cut_passes
        if self.network_start_used:
            self.model.Params.MIPFocus = 1
            self.model.Params.Heuristics = 0.5
            self.model.Params.NoRelHeurTime = min(
                solver.no_rel_heur_seconds,
                0.05 * remaining,
            )
        else:
            # Failure of the short auxiliary search is evidence that primal
            # discovery is difficult, not a reason to switch the main solve
            # to a proof-oriented default.  Keep an incumbent-focused search
            # so the large screened instances still have a realistic chance
            # to improve on the always-feasible all-courier start.
            self.model.Params.MIPFocus = 1
            self.model.Params.Heuristics = 0.25
            self.model.Params.NoRelHeurTime = min(
                solver.no_rel_heur_seconds,
                0.05 * remaining,
            )
        if (
            solver.use_user_cuts
            and solver.max_root_cut_rounds > 0
            and solver.max_cuts_per_round > 0
        ):
            self.model.Params.PreCrush = 1
            self.model.optimize(self._root_user_cut_callback(solver))
        else:
            self.model.optimize()
        self.main_solve_runtime = float(self.model.Runtime)
        self.total_runtime = self.network_start_runtime + self.main_solve_runtime
        return self.model.SolCount > 0

    def export_incumbent_start(self) -> dict[str, float]:
        """Export a complete incumbent by stable Gurobi variable name."""

        self.model.update()
        if self.model.SolCount <= 0:
            raise ValueError("cannot export a start without a feasible incumbent")
        return {
            variable.VarName: float(variable.X) for variable in self.model.getVars()
        }

    def _append_external_start(
        self,
        values_by_name: Mapping[str, float],
        source: str | None,
    ) -> None:
        """Append a complete named start, setting variables absent upstream to zero."""

        normalized: dict[str, float] = {}
        for raw_name, raw_value in values_by_name.items():
            name = str(raw_name)
            value = float(raw_value)
            if not math.isfinite(value):
                raise ValueError(
                    f"external start contains nonfinite value for {name!r}"
                )
            normalized[name] = value

        self.model.update()
        variables = self.model.getVars()
        matched = sum(variable.VarName in normalized for variable in variables)
        if matched == 0:
            raise ValueError(
                "external start has no variables in common with this model"
            )
        start_index = int(self.model.NumStart)
        self.model.NumStart = start_index + 1
        self.model.update()
        self.model.Params.StartNumber = start_index
        for variable in variables:
            variable.Start = normalized.get(variable.VarName, 0.0)
        self.model.Params.StartNumber = 0
        self.model.update()
        self.external_start_installed = True
        self.external_start_source = source
        self.external_start_matched_variables = matched

    def _rank_lo_network_start_candidates(
        self, solver: SolverConfig
    ) -> tuple[tuple[_LOStartCandidate, ...], int]:
        """Rank LO display candidates after a necessary capacity screen.

        The screen uses only the precomputed choice-cover coefficients.  It
        therefore scales with the number of ``alpha`` candidates and their OD
        contributions, rather than repeatedly traversing the full ``chi``
        index.  A candidate is rejected only when even the smallest available
        realization rate exceeds the largest admissible receiving-locker
        arrival rate.
        """

        cfg = self.inst.config
        rho_min = min(cfg.availability_levels)
        max_arrival_capacity = max(
            stable_arrival_capacity(mu, cfg.max_locker_time_hours)
            for mu in cfg.locker_service_rates
        )
        feasibility_tolerance = max(
            float(self.model.Params.FeasibilityTol),
            solver.cut_violation_tolerance,
        )
        alpha_keys = set(self.alpha_keys)
        ranked: list[_LOStartCandidate] = []
        capacity_rejected = 0
        for data in self._choice_cover_data:
            if data.alpha_key not in alpha_keys:
                raise ValueError(
                    "choice-cover candidate has no matching alpha variable: "
                    f"{data.alpha_key!r}"
                )
            offered_demand = math.fsum(
                amount for _market, amount in data.market_contributions
            )
            effective_min_arrival = rho_min * offered_demand
            tolerance = feasibility_tolerance * max(
                1.0,
                max_arrival_capacity,
                effective_min_arrival,
            )
            if effective_min_arrival > max_arrival_capacity + tolerance:
                capacity_rejected += 1
                continue

            saving_terms: list[float] = []
            for (origin, destination), amount in data.market_contributions:
                origins = self.inst.display_origins.get(
                    (origin, destination, data.locker), ()
                )
                if not origins:
                    raise ValueError(
                        "choice-cover candidate has no retained origin locker: "
                        f"{(origin, destination, data.locker)!r}"
                    )
                best_unit_saving = max(
                    self.inst.direct_rider_cost(origin, destination)
                    - self.inst.feeder_rider_cost(origin, locker_origin)
                    - self.inst.locker_pickup_cost(data.locker)
                    - cfg.flight_cost_per_km
                    * self.inst.distance(locker_origin, data.locker)
                    for locker_origin in origins
                )
                saving_terms.append(amount * best_unit_saving)
            ranked.append(
                _LOStartCandidate(
                    alpha_key=data.alpha_key,
                    offered_demand=offered_demand,
                    effective_min_arrival=effective_min_arrival,
                    optimistic_saving=math.fsum(saving_terms),
                )
            )

        ranked.sort(
            key=lambda candidate: (
                -candidate.optimistic_saving,
                -candidate.offered_demand,
                candidate.alpha_key[0],
                candidate.alpha_key[1],
            )
        )
        return tuple(ranked), capacity_rejected

    def _snapshot_mip_start_state(
        self, model_variables: list[gp.Var]
    ) -> _MIPStartState:
        """Capture every installed MIP start before a temporary solve."""

        self.model.update()
        num_start = int(self.model.NumStart)
        start_number = int(self.model.Params.StartNumber)
        values: list[tuple[float, ...]] = []
        for index in range(num_start):
            self.model.Params.StartNumber = index
            values.append(tuple(self.model.getAttr(GRB.Attr.Start, model_variables)))
        self.model.Params.StartNumber = start_number
        return _MIPStartState(num_start, start_number, tuple(values))

    def _restore_mip_start_state(
        self,
        model_variables: list[gp.Var],
        state: _MIPStartState,
    ) -> None:
        """Restore an exact MIP-start snapshot after a temporary solve."""

        self.model.NumStart = state.num_start
        self.model.update()
        for index, values in enumerate(state.values):
            self.model.Params.StartNumber = index
            self.model.setAttr(GRB.Attr.Start, model_variables, list(values))
        self.model.Params.StartNumber = state.start_number
        self.model.update()

    def _solve_network_start_attempt(
        self,
        *,
        time_limit: float,
        alpha_target: tuple[str, str] | None,
        required_physical_nodes: frozenset[str] | None = None,
        allowed_physical_nodes: frozenset[str] | None = None,
        positive_chi_keys: tuple[ChiKey, ...] = (),
        positive_offer: float = 0.0,
    ) -> _NetworkStartAttempt | None:
        """Solve one auxiliary start problem and restore all temporary state."""

        if not math.isfinite(time_limit) or time_limit <= 0.0:
            raise ValueError("network-start attempt time_limit must be positive")
        if positive_offer < 0.0 or not math.isfinite(positive_offer):
            raise ValueError("positive_offer must be finite and nonnegative")

        x = self.vars["x"]
        alpha = self.vars["alpha"]
        chi = self.vars["chi"]
        self.model.update()
        bounded_variables = [
            *(variable for variable in x.values()),
            *(variable for variable in alpha.values()),
        ]
        lower_bounds = self.model.getAttr(GRB.Attr.LB, bounded_variables)
        upper_bounds = self.model.getAttr(GRB.Attr.UB, bounded_variables)
        model_variables = self.model.getVars()
        start_state = self._snapshot_mip_start_state(model_variables)
        parameter_state = {
            name: getattr(self.model.Params, name)
            for name in _NETWORK_START_PARAMETER_NAMES
        }
        temporary_constraints: list[gp.Constr] = []
        attempt: _NetworkStartAttempt | None = None
        try:
            # A failed candidate must not inherit an incumbent or MIP start
            # from an earlier candidate.  The original pool is restored in the
            # finally block and is replaced only after a feasible seed exists.
            self.model.reset(0)
            self.model.NumStart = 0
            self.model.update()
            if allowed_physical_nodes is not None:
                if required_physical_nodes is None:
                    raise ValueError(
                        "allowed_physical_nodes requires required_physical_nodes"
                    )
                for key, variable in x.items():
                    if key in required_physical_nodes:
                        variable.LB = 1.0
                        variable.UB = 1.0
                    elif key not in allowed_physical_nodes:
                        variable.LB = 0.0
                        variable.UB = 0.0
            for key, variable in alpha.items():
                fixed = 1.0 if key == alpha_target else 0.0
                variable.LB = fixed
                variable.UB = fixed
            if positive_offer > 0.0:
                if not positive_chi_keys:
                    raise ValueError(
                        "a positive network offer requires at least one chi variable"
                    )
                temporary_constraints.append(
                    self.model.addConstr(
                        gp.quicksum(chi[key] for key in positive_chi_keys)
                        >= positive_offer,
                        name="__network_start_positive_chi",
                    )
                )
            self.model.Params.TimeLimit = time_limit
            self.model.Params.MIPGap = 0.001
            self.model.Params.MIPFocus = 1
            self.model.Params.Heuristics = 0.5
            self.model.Params.NoRelHeurTime = min(5.0, 0.25 * time_limit)
            self.model.update()
            self.model.optimize()
            if self.model.SolCount > 0:
                objective = float(self.model.ObjVal)
                values = {variable: float(variable.X) for variable in model_variables}
                attempt = _NetworkStartAttempt(
                    seed=capture_incumbent(self.model, self.vars),
                    objective=objective,
                    values=values,
                )
        finally:
            self.model.reset(0)
            if temporary_constraints:
                self.model.remove(temporary_constraints)
            self.model.setAttr(GRB.Attr.LB, bounded_variables, lower_bounds)
            self.model.setAttr(GRB.Attr.UB, bounded_variables, upper_bounds)
            self.model.update()
            self._restore_mip_start_state(model_variables, start_state)
            for name, value in parameter_state.items():
                setattr(self.model.Params, name, value)
        return attempt

    def _construct_network_start(self, time_limit: float, solver: SolverConfig) -> None:
        """Construct a nontrivial mode-valid start within one wall-clock budget."""

        start_wall_time = time.perf_counter()
        deadline = start_wall_time + time_limit
        attempt: _NetworkStartAttempt | None = None

        if self.mode == "LO":
            selection_started = time.perf_counter()
            candidates, capacity_rejected = self._rank_lo_network_start_candidates(
                solver
            )
            self.network_start_candidate_selection_seconds = (
                time.perf_counter() - selection_started
            )
            self.network_start_candidate_count = len(self._choice_cover_data)
            self.network_start_capacity_rejected_count = capacity_rejected
            shortlisted = candidates[:_MAX_LO_NETWORK_START_ATTEMPTS]
            attempted_keys: list[tuple[str, str]] = []
            for index, candidate in enumerate(shortlisted):
                remaining = deadline - time.perf_counter()
                attempts_left = len(shortlisted) - index
                attempt_time = remaining / attempts_left
                if remaining < 0.1 or attempt_time < 0.1:
                    break
                attempted_keys.append(candidate.alpha_key)
                self.network_start_attempted_keys = tuple(attempted_keys)
                self.network_start_key = candidate.alpha_key
                attempt = self._solve_network_start_attempt(
                    time_limit=attempt_time,
                    alpha_target=candidate.alpha_key,
                )
                if attempt is not None:
                    break
        else:
            # MX and HD can first be seeded by a home-delivery corridor.  This
            # avoids fixing a customer display in MX, whose nearest-open-locker
            # implications can otherwise suppress an economically useful part
            # of the physical network.  HD has no alpha variables at all.
            def corridor_score(arc: Arc) -> tuple[float, float, Arc]:
                i, j = arc
                saving = 0.0
                volume = 0.0
                for origin, destination in self.inst.markets:
                    if arc not in self.market_corridors[origin, destination]:
                        continue
                    amount = self.inst.h_od[origin, destination]
                    unit_saving = (
                        self.inst.direct_rider_cost(origin, destination)
                        - self.inst.feeder_rider_cost(origin, i)
                        - self.inst.feeder_rider_cost(j, destination)
                        - self.inst.config.flight_cost_per_km * self.inst.distance(i, j)
                    )
                    saving += amount * max(0.0, unit_saving)
                    volume += amount
                return saving, volume, arc

            if self.corridors:
                required_physical_nodes: frozenset[str] | None = None
                allowed_physical_nodes: frozenset[str] | None = None

                def supported_cycle(arc: Arc) -> frozenset[str] | None:
                    i, j = arc
                    for base in sorted(self.inst.L):
                        cycle_arcs = ((base, i), (i, j), (j, base))
                        if all(item in self.inst.A_F for item in cycle_arcs):
                            return frozenset((base, i, j))
                    return None

                supported = [
                    (arc, cycle)
                    for arc in self.corridors
                    if (cycle := supported_cycle(arc)) is not None
                ]
                if supported:
                    target, required_physical_nodes = max(
                        supported,
                        key=lambda item: corridor_score(item[0]),
                    )
                    locker_support_limit = min(
                        len(self.inst.H),
                        60,
                        max(16, math.ceil(0.5 * len(self.inst.H))),
                    )
                    locker_support = set(
                        sorted(
                            self.inst.H,
                            key=lambda node: (
                                -self.inst.activity_rate.get(node, 0.0),
                                node,
                            ),
                        )[:locker_support_limit]
                    )
                    allowed_physical_nodes = frozenset(
                        (
                            *required_physical_nodes,
                            *locker_support,
                            *self.inst.L,
                            *self.inst.E,
                        )
                    )
                else:
                    target = max(self.corridors, key=corridor_score)

                self.network_start_key = target
                if required_physical_nodes is not None:
                    self.network_start_required_nodes = tuple(
                        sorted(required_physical_nodes)
                    )
                if allowed_physical_nodes is not None:
                    self.network_start_allowed_node_counts = {
                        "bases": sum(
                            node in self.inst.L for node in allowed_physical_nodes
                        ),
                        "lockers": sum(
                            node in self.inst.H for node in allowed_physical_nodes
                        ),
                        "swap_stations": sum(
                            node in self.inst.E for node in allowed_physical_nodes
                        ),
                    }
                relevant = tuple(self.chi_by_corridor[target])
                accessible_markets = {(key[0], key[1]) for key in relevant}
                market_mass = math.fsum(
                    self.inst.h_od[market] for market in accessible_markets
                )
                positive_offer = min(1.0, 0.01 * market_mass)
                remaining = deadline - time.perf_counter()
                auxiliary_time = min(
                    remaining,
                    max(10.0, 0.75 * time_limit),
                )
                if auxiliary_time >= 0.1:
                    attempt = self._solve_network_start_attempt(
                        time_limit=auxiliary_time,
                        alpha_target=None,
                        required_physical_nodes=required_physical_nodes,
                        allowed_physical_nodes=allowed_physical_nodes,
                        positive_chi_keys=relevant,
                        positive_offer=positive_offer,
                    )

        start_values: Mapping[gp.Var, float] = {}
        seed: PrimalSeed | None = None
        if attempt is not None:
            seed = attempt.seed
            start_values = attempt.values
            self.network_start_found = True
            self.network_start_used = True
            self.network_start_objective = attempt.objective
            seed_x = seed.structure["x"]
            self.network_start_seed_open_counts = {
                "bases": sum(seed_x[node] for node in self.inst.L),
                "lockers": sum(seed_x[node] for node in self.inst.H),
                "swap_stations": sum(seed_x[node] for node in self.inst.E),
            }

        local_time = max(0.0, deadline - time.perf_counter())
        if seed is not None and solver.use_local_branching and local_time >= 0.1:
            result = improve_mip_start(
                self.model,
                self.vars,
                seed,
                LocalBranchingConfig(
                    radius=solver.local_branching_radius,
                    time_limit=local_time,
                    mip_gap=min(0.01, solver.mip_gap),
                    output_flag=solver.output_flag,
                ),
            )
            self.local_branching_runtime = result.runtime
            self.local_branching_improved = result.improved
            self.local_branching_objective = result.installed_objective
            self.local_branching_status = result.status
            self.local_branching_solution_count = result.solution_count
            self.local_branching_hamming_distance = result.hamming_distance
            self.network_start_objective = result.installed_objective
        elif self.network_start_used:
            self.model.NumStart = 1
            self.model.update()
            self.model.Params.StartNumber = 0
            for variable, value in start_values.items():
                variable.Start = value
            self.model.update()

        if self.network_start_used:
            self._append_network_fallback_starts()
        else:
            self._set_courier_only_start()
        self.network_start_runtime = time.perf_counter() - start_wall_time

    def _append_network_fallback_starts(self) -> None:
        """Keep the complete network start and append partial and courier fallbacks.

        The partial start fixes only the structural integer design
        ``(x, z, delta, alpha)``.  Every other variable is explicitly left
        undefined so Gurobi can reconstruct the continuous conic solution if
        a tiny numerical residual causes it to reject the complete start.
        """

        self.model.update()
        if int(self.model.NumStart) < 1:
            raise ValueError("a complete network start must be installed first")

        model_variables = self.model.getVars()
        self.model.Params.StartNumber = 0
        structural_values: list[tuple[gp.Var, float]] = []
        for group in self.network_partial_start_groups:
            container = self.vars[group]
            if not hasattr(container, "values"):
                raise TypeError(f"variables[{group!r}] must be a keyed container")
            for variable in container.values():
                value = float(variable.Start)
                if abs(value) >= 0.5 * GRB.UNDEFINED:
                    raise ValueError(
                        f"complete network start is missing {group}[{variable.VarName!r}]"
                    )
                structural_values.append((variable, float(value >= 0.5)))

        partial_index = int(self.model.NumStart)
        courier_index = partial_index + 1
        self.model.NumStart = courier_index + 1
        self.model.update()

        self.model.Params.StartNumber = partial_index
        self.model.setAttr(
            GRB.Attr.Start,
            model_variables,
            [GRB.UNDEFINED] * len(model_variables),
        )
        for variable, value in structural_values:
            variable.Start = value

        self.model.Params.StartNumber = courier_index
        self._set_courier_only_start(reset_start_pool=False)
        self.model.Params.StartNumber = 0
        self.model.update()

        self.network_partial_start_installed = True
        self.network_partial_start_index = partial_index
        self.network_partial_start_variable_count = len(structural_values)

    def _set_courier_only_start(self, *, reset_start_pool: bool = True) -> None:
        """Seed the always-feasible all-courier counterfactual."""

        if reset_start_pool:
            self.model.NumStart = 1
            self.model.update()
            self.model.Params.StartNumber = 0
        for name, container in self.vars.items():
            if isinstance(container, gp.Var):
                container.Start = 0.0
                continue
            start_value = 1.0 if name in {"queue_v2", "queue_t"} else 0.0
            for variable in container.values():
                variable.Start = start_value

    def write_model(self, path: str) -> None:
        self.build()
        self.model.write(path)
