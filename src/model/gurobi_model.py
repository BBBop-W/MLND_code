"""Gurobi implementation of the unified chi/epsilon-energy MLND MISOCP."""

from __future__ import annotations

import itertools
import math
import time
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import asdict, dataclass

import gurobipy as gp
from gurobipy import GRB

from src.config import SolverConfig
from src.data_processor.instance import DIRECT_ROUTE, Arc, Instance, Market
from src.model_contract import FORMULATION_ID
from src.model.primal_heuristic import (
    DEFAULT_STRUCTURE_GROUPS,
    OBJECTIVE_START_PROXY_ENERGY_REL_TOLERANCE,
    OBJECTIVE_START_PROXY_QOS_ABS_TOLERANCE,
    OBJECTIVE_START_PROXY_SEARCH_REL_TOLERANCE,
    CapacityOption,
    FixedIntegerRepairConfig,
    LocalBranchingConfig,
    ObjectiveAwareOpportunity,
    ObjectiveAwarePlanningConfig,
    PrimalSeed,
    build_all_courier_partial_assignment,
    build_candidate_partial_assignment,
    capture_incumbent,
    effective_objective_start_budget,
    improve_mip_start,
    install_partial_mip_starts,
    plan_objective_aware_partial_starts,
    repair_fixed_integer_design_start,
)
from src.model.valid_inequalities import (
    ChoiceCorridorCoverData,
    RouteAggregateVUBBound,
    build_choice_corridor_cover_data,
    build_nearest_display_suffix_rows,
    build_swap_route_bundles,
    evaluate_route_bundle_vub,
    queue_perspective_intercept,
    route_swap_leg_vub_bound,
    route_swap_station_vub_bound,
    separate_choice_capacity_covers,
    separate_directed_return_cuts,
    separate_od_disaggregated_directed_return_cuts,
    separate_queue_perspective_cut,
    stable_arrival_capacity,
)

ChiKey = tuple[str, str, str, str, int]  # (o, d, i, j, r)
EpsilonKey = tuple[str, str, str]  # (o, d, j)
WKey = tuple[str, str, str, int]  # (idle, i, j, r)
VarphiKey = tuple[str, str, str]  # (physical tail, physical head, route option)

_MAX_LO_NETWORK_START_ATTEMPTS = 5
_OBJECTIVE_PARTIAL_START_MODES = frozenset({"HD", "MX"})
_NETWORK_START_PARAMETER_NAMES = (
    "TimeLimit",
    "MIPGap",
    "MIPFocus",
    "Heuristics",
    "NoRelHeurTime",
    "Method",
    "MIQCPMethod",
)


def _aggregate_raw_offer_bounds(
    market_corridors: Mapping[Market, tuple[Arc, ...]],
    h_od: Mapping[Market, float],
    choice_share: Mapping[tuple[str, str, str], float],
    mode: str,
) -> dict[Arc, float]:
    """Accumulate the exact demand-side offer bound in retained-index time."""

    bounds: dict[Arc, float] = defaultdict(float)
    for (origin, destination), corridors in market_corridors.items():
        demand = h_od[origin, destination]
        for corridor in corridors:
            coefficient = demand
            if mode == "LO":
                coefficient *= choice_share.get(
                    (origin, destination, corridor[1]),
                    1.0,
                )
            bounds[corridor] += coefficient
    return dict(bounds)


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

    def __init__(
        self,
        instance: Instance,
        mode: str = "MX",
        *,
        environment: gp.Env | None = None,
    ):
        self.inst = instance
        self.mode = mode.upper()
        if self.mode not in self.MODES:
            raise ValueError(f"mode must be one of {sorted(self.MODES)}")
        model_name = f"MLND_{self.mode}_CHI_EPSILON_ENERGY"
        if environment is None:
            self.model = gp.Model(model_name)
        else:
            self.model = gp.Model(model_name, env=environment)
        self.solver_config: SolverConfig | None = None
        self.vars: dict[str, object] = {}
        self.cost_expressions: dict[str, gp.LinExpr] = {}
        self.paper_objective_expressions: dict[str, gp.LinExpr] = {}
        self.courier_only_baseline = 0.0
        self.network_start_runtime = 0.0
        self.network_construction_runtime = 0.0
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
        self.network_start_open_h_count = 0
        self.network_start_open_facility_count = 0
        self.network_start_service_per_hour = 0.0
        self.network_start_relevant_chi_count = 0
        self.start_preparation_runtime = 0.0
        self.objective_partial_start_telemetry = self._empty_objective_start_telemetry()
        self.main_solve_runtime = 0.0
        self.main_solve_started = False
        self.optimize_called = False
        self.total_runtime = 0.0
        self.budget_exhausted_before_main_solve = False
        self.algorithm_prepare_runtime = 0.0
        self.static_cut_counts: dict[str, int] = {}
        self.user_cut_counts: dict[str, int] = {}
        self.user_cut_candidate_counts: dict[str, int] = {}
        self.user_cut_duplicate_counts: dict[str, int] = {}
        self.user_cut_rejection_counts: dict[str, int] = {}
        self.user_cut_separation_seconds: dict[str, float] = {}
        self.root_cut_rounds = 0
        self.root_bound_before_cuts: float | None = None
        self.root_bound_after_cuts: float | None = None
        self.root_round_events: list[dict[str, object]] = []
        self.progress_events: list[dict[str, object]] = []
        self.callback_runtime = 0.0
        self.first_incumbent_time: float | None = None
        self.first_incumbent_objective: float | None = None
        self._last_progress_time = -math.inf
        self._last_progress_incumbent: float | None = None
        self._last_progress_bound: float | None = None
        self.local_branching_runtime = 0.0
        self.local_branching_improved = False
        self.local_branching_objective: float | None = None
        self.local_branching_status: int | None = None
        self.local_branching_solution_count = 0
        self.local_branching_hamming_distance: int | None = None
        self.fixed_integer_repair_runtime = 0.0
        self.fixed_integer_repair_found = False
        self.fixed_integer_repair_objective: float | None = None
        self.fixed_integer_repair_status: int | None = None
        self.fixed_integer_repair_solution_count = 0
        self.fixed_integer_repair_fixed_count = 0
        self._valid_inequalities_added = False
        self._user_cut_signatures: set[tuple[object, ...]] = set()
        self._static_user_cut_signatures: set[tuple[object, ...]] = set()
        self._choice_cover_data: tuple[ChoiceCorridorCoverData, ...] = ()
        self._built = False
        self._prepare_indices()

    @staticmethod
    def _empty_objective_start_telemetry() -> dict[str, object]:
        return {
            "policy_version": "objective_partial_v3_system_energy",
            "proxy_only": True,
            "proxy_feasibility_tolerances": {
                "qos_arrival_absolute": (
                    OBJECTIVE_START_PROXY_QOS_ABS_TOLERANCE
                ),
                "system_energy_relative": (
                    OBJECTIVE_START_PROXY_ENERGY_REL_TOLERANCE
                ),
                "construction_search_relative": (
                    OBJECTIVE_START_PROXY_SEARCH_REL_TOLERANCE
                ),
                "solver_feasibility_tolerance_linked": False,
            },
            "enabled": False,
            "used": False,
            "budget_seconds": 0.0,
            "preparation_seconds": 0.0,
            "opportunity_count": 0,
            "singleton_support_count": 0,
            "evaluated_support_count": 0,
            "network_candidate_count": 0,
            "heuristic_candidate_count": 0,
            "safety_start_count": 0,
            "installed_start_count": 0,
            "candidate_supports": [],
            "rejection_reasons": [],
            "fallback_reason": None,
            "timed_out": False,
        }

    def _prepare_indices(self) -> None:
        """Build the sparse OD-corridor-reliability index sets for one mode."""

        level_count = len(self.inst.config.availability_levels)
        self.market_corridors: dict[Market, tuple[Arc, ...]] = {}
        chi_keys: list[ChiKey] = []
        self.chi_by_market: dict[Market, list[ChiKey]] = defaultdict(list)
        self.chi_by_market_i: dict[EpsilonKey, list[ChiKey]] = defaultdict(list)
        self.chi_by_market_j: dict[EpsilonKey, list[ChiKey]] = defaultdict(list)
        self.chi_by_market_incident: dict[EpsilonKey, list[ChiKey]] = defaultdict(
            list
        )
        self.chi_by_destination_locker: dict[
            tuple[str, str], list[ChiKey]
        ] = defaultdict(list)
        self.chi_by_od_corridor: dict[
            tuple[str, str, str, str], list[ChiKey]
        ] = defaultdict(list)
        self.chi_by_corridor: dict[Arc, list[ChiKey]] = defaultdict(list)
        self.chi_by_corridor_level: dict[tuple[str, str, int], list[ChiKey]] = defaultdict(list)

        for market in self.inst.markets:
            corridors = tuple(dict.fromkeys(self.inst.corridors_for_mode(self.mode, market)))
            invalid = [arc for arc in corridors if arc not in self.inst.A_HH]
            if invalid:
                raise ValueError(f"mode corridor set contains non-HH arcs: {invalid[:3]}")
            self.market_corridors[market] = corridors
            o, d = market
            for i, j in corridors:
                for r in range(level_count):
                    key = (o, d, i, j, r)
                    chi_keys.append(key)
                    self.chi_by_market[market].append(key)
                    self.chi_by_market_i[o, d, i].append(key)
                    self.chi_by_market_j[o, d, j].append(key)
                    self.chi_by_market_incident[o, d, i].append(key)
                    if j != i:
                        self.chi_by_market_incident[o, d, j].append(key)
                    self.chi_by_destination_locker[d, j].append(key)
                    self.chi_by_od_corridor[o, d, i, j].append(key)
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
        raw_offer_bound_by_corridor = _aggregate_raw_offer_bounds(
            self.market_corridors,
            self.inst.h_od,
            self.inst.choice_share,
            self.mode,
        )
        self.M_offer: dict[Arc, float] = {}
        for arc in self.corridors:
            realized_bound = self.inst.M_psi.get(
                arc,
                max(self.inst.M_psi.values(), default=0.0),
            )
            self.M_offer[arc] = max(
                0.0,
                min(raw_offer_bound_by_corridor[arc], realized_bound / rho_min),
            )

        self.w_keys: tuple[WKey, ...] = tuple(
            (idle, i, j, r)
            for i, j in self.corridors
            for idle in self.inst.eligible_idle[i]
            for r in range(level_count)
        )
        self.w_by_source: dict[tuple[str, str], list[WKey]] = defaultdict(list)
        self.w_by_corridor: dict[Arc, list[WKey]] = defaultdict(list)
        self.w_by_corridor_level: dict[tuple[str, str, int], list[WKey]] = defaultdict(list)
        for key in self.w_keys:
            idle, i, j, r = key
            self.w_by_source[idle, i].append(key)
            self.w_by_corridor[i, j].append(key)
            self.w_by_corridor_level[i, j, r].append(key)

        self.varphi_keys: tuple[VarphiKey, ...] = tuple(
            (u, v, option)
            for u, v in sorted(self.inst.A_F)
            for option in self.inst.route_options[u, v]
        )
        self.swap_varphi_by_leg: dict[Arc, tuple[VarphiKey, ...]] = {}
        self.swap_varphi_by_station: dict[str, tuple[VarphiKey, ...]] = {}
        self.swap_route_leg_vub_bounds: dict[
            Arc, RouteAggregateVUBBound
        ] = {}
        self.swap_route_station_vub_bounds: dict[
            str, RouteAggregateVUBBound
        ] = {}
        self.route_bundle_index_counts: dict[str, int] = {}
        self._route_bundle_indices_prepared = False

    def _prepare_route_bundle_indices(self) -> None:
        """Build development-only route bundles inside the algorithm budget."""

        if self._route_bundle_indices_prepared:
            return
        (
            self.swap_varphi_by_leg,
            self.swap_varphi_by_station,
        ) = build_swap_route_bundles(
            self.inst.route_options,
            direct_option=DIRECT_ROUTE,
        )
        realized_demand_upper = (
            max(self.inst.config.availability_levels) * self.inst.total_demand
        )
        self.swap_route_leg_vub_bounds: dict[
            Arc, RouteAggregateVUBBound
        ] = {}
        for leg, keys in self.swap_varphi_by_leg.items():
            if leg[0] in self.inst.E:
                endpoint = leg[1]
            elif leg[1] in self.inst.E:
                endpoint = leg[0]
            else:  # pragma: no cover - guarded by the route-option builder
                raise ValueError(f"swap leg {leg!r} has no swap-station endpoint")
            if endpoint in self.inst.L:
                endpoint_kind = "base"
            elif endpoint in self.inst.H:
                endpoint_kind = "locker"
            else:  # pragma: no cover - input node sets are validated as disjoint
                raise ValueError(f"swap leg {leg!r} has an unknown physical endpoint")
            self.swap_route_leg_vub_bounds[leg] = route_swap_leg_vub_bound(
                keys,
                self.inst.M_psi,
                endpoint_kind=endpoint_kind,
                realized_demand_upper=realized_demand_upper,
                locker_service_rates=self.inst.config.locker_service_rates,
                max_locker_time=self.inst.config.max_locker_time_hours,
            )

        arc_categories = {
            **{arc: "LH" for arc in self.inst.A_LH},
            **{arc: "HH" for arc in self.inst.A_HH},
            **{arc: "HL" for arc in self.inst.A_HL},
        }
        self.swap_route_station_vub_bounds: dict[
            str, RouteAggregateVUBBound
        ] = {
            station: route_swap_station_vub_bound(
                keys,
                self.inst.M_psi,
                arc_categories,
                realized_demand_upper=realized_demand_upper,
            )
            for station, keys in self.swap_varphi_by_station.items()
        }
        self.route_bundle_index_counts = {
            "swap_route_options": sum(
                len(keys) for keys in self.swap_varphi_by_station.values()
            ),
            "swap_leg_bundles": len(self.swap_varphi_by_leg),
            "swap_station_bundles": len(self.swap_varphi_by_station),
            "strengthening_swap_leg_bundles": sum(
                bound.strictly_strengthens()
                for bound in self.swap_route_leg_vub_bounds.values()
            ),
            "strengthening_swap_station_bundles": sum(
                bound.strictly_strengthens()
                for bound in self.swap_route_station_vub_bounds.values()
            ),
        }
        self._route_bundle_indices_prepared = True

    def _successful_market_expression(
        self,
        chi: gp.tupledict,
        market: Market,
    ) -> gp.LinExpr:
        levels = self.inst.config.availability_levels
        return gp.quicksum(levels[key[4]] * chi[key] for key in self.chi_by_market[market])

    def _successful_market_locker_expression(
        self,
        chi: gp.tupledict,
        key: EpsilonKey,
    ) -> gp.LinExpr:
        levels = self.inst.config.availability_levels
        return gp.quicksum(levels[item[4]] * chi[item] for item in self.chi_by_market_j[key])

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
        self.model.setObjective(gp.quicksum(self.cost_expressions.values()), GRB.MINIMIZE)

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
                    gp.quicksum(
                        chi[key] for key in self.chi_by_corridor_level[i, j, r]
                    )
                    == gp.quicksum(
                        w[key] for key in self.w_by_corridor_level[i, j, r]
                    ),
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
                    math.sqrt(cfg.locker_service_rates[c]) * z[i, c]
                    for c in capacities
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
        prepare_started = time.perf_counter()
        did_work = False
        # f884's LO network-start search ranks up to five exact display
        # candidates from this same data.  Build it for every LO model, even
        # when static and dynamic choice-cut families are both disabled.
        needs_choice_data = self.mode == "LO" and bool(self.alpha_keys)
        if needs_choice_data and not self._choice_cover_data:
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
            did_work = True
        needs_route_bundle_data = solver.use_user_cuts and (
            solver.cut_route_swap_leg_vub
            or solver.cut_route_swap_station_vub
        )
        if needs_route_bundle_data and not self._route_bundle_indices_prepared:
            self._prepare_route_bundle_indices()
            did_work = True
        if solver.use_valid_inequalities and not self._valid_inequalities_added:
            self._add_static_valid_inequalities(solver)
            self._valid_inequalities_added = True
            self.model.update()
            did_work = True
        if did_work:
            self.algorithm_prepare_runtime += time.perf_counter() - prepare_started
        return self

    def _add_static_valid_inequalities(self, solver: SolverConfig) -> None:
        """Add compact projections that remain valid for the selected mode."""

        cfg = self.inst.config
        z = self.vars["z"]
        x = self.vars["x"]
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

        if solver.static_nearest_display_suffix:
            # Access the key even when the family has no nonredundant rows so
            # cut telemetry distinguishes "enabled but degenerate" from off.
            counts["nearest_display_suffix"] += 0
            if self.alpha_keys:
                for row in build_nearest_display_suffix_rows(
                    self.inst.display_lockers,
                    self.inst.closer_lockers,
                ):
                    self.model.addConstr(
                        gp.quicksum(
                            alpha[row.destination, locker]
                            for locker in row.farther_lockers
                        )
                        <= 1.0 - x[row.anchor_locker],
                        name=(
                            "vi_nearest_display_suffix"
                            f"[{row.destination},{row.anchor_locker}]"
                        ),
                    )
                    counts["nearest_display_suffix"] += 1

        if solver.static_source_level or solver.static_virtual_level_lower:
            for locker in self.inst.H:
                for idle in self.inst.eligible_idle[locker]:
                    source_keys = self.w_by_source.get((idle, locker), ())
                    for r, rho in enumerate(cfg.availability_levels):
                        offered = gp.quicksum(
                            w[key] for key in source_keys if key[3] == r
                        )
                        if solver.static_source_level:
                            self.model.addConstr(
                                rho * offered
                                <= max_locker_flow * delta[idle, r],
                                name=(
                                    "vi_source_level_upper"
                                    f"[{idle},{locker},{r}]"
                                ),
                            )
                            counts["source_level_upper"] += 1
                        if solver.static_virtual_level_lower and idle in self.inst.S:
                            lower = 1.0 / (
                                (1.0 - rho) * cfg.max_virtual_wait_hours
                            )
                            self.model.addConstr(
                                offered >= lower * delta[idle, r],
                                name=f"vi_virtual_level_lower[{idle},{r}]",
                            )
                            counts["virtual_level_lower"] += 1

        capacity_slack = tuple(
            rate - 1.0 / cfg.max_locker_time_hours
            for rate in cfg.locker_service_rates
        )
        if self.mode == "LO" and (
            solver.static_lo_choice_corridor or solver.static_lo_choice_capacity
        ):
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
                if solver.static_lo_choice_corridor:
                    self.model.addConstr(
                        gp.quicksum(
                            min(self.M_offer[arc], volume) * y[arc]
                            for arc in display_arcs[destination, locker]
                        )
                        >= volume * alpha[destination, locker],
                        name=(
                            "vi_lo_choice_corridor_cover"
                            f"[{destination},{locker}]"
                        ),
                    )
                    counts["lo_choice_corridor_cover"] += 1

            if solver.static_lo_choice_capacity:
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
        if solver.static_queue_perspective:
            for locker in self.inst.H:
                incoming = gp.quicksum(
                    psi[u, v] for u, v in self.inst.A if v == locker
                )
                for slope_index, slope in enumerate(queue_slopes):
                    intercepts = tuple(
                        queue_perspective_intercept(
                            rate,
                            stable_arrival_capacity(
                                rate, cfg.max_locker_time_hours
                            ),
                            slope,
                        )
                        for rate in cfg.locker_service_rates
                    )
                    self.model.addConstr(
                        xi[locker]
                        >= slope * incoming
                        + gp.quicksum(
                            intercepts[c] * z[locker, c]
                            for c in range(len(intercepts))
                        ),
                        name=f"vi_queue_perspective[{locker},{slope_index}]",
                    )
                    counts["queue_perspective_support"] += 1

        if solver.static_chi_vub_top_k_per_corridor:
            chi = self.vars["chi"]
            groups_by_corridor: dict[Arc, list[tuple[str, str]]] = defaultdict(list)
            for o, d, i, j in self.chi_by_od_corridor:
                groups_by_corridor[i, j].append((o, d))

            top_k = solver.static_chi_vub_top_k_per_corridor
            for i, j in sorted(groups_by_corridor):
                ranked = sorted(
                    groups_by_corridor[i, j],
                    key=lambda market: (
                        -self.inst.h_od[market],
                        market[0],
                        market[1],
                    ),
                )
                for o, d in ranked[:top_k]:
                    keys, coefficient, corridor = self._chi_od_corridor_vub_data(
                        o, d, i, j
                    )
                    self.model.addConstr(
                        gp.quicksum(chi[key] for key in keys)
                        <= coefficient * y[corridor],
                        name=f"vi_chi_od_corridor_vub[{o},{d},{i},{j}]",
                    )
                    signature = ("chi_od_corridor_vub", o, d, i, j)
                    self._static_user_cut_signatures.add(signature)
                    self._user_cut_signatures.add(signature)
                    counts["chi_od_corridor_vub_top_k"] += 1

        self.static_cut_counts = dict(sorted(counts.items()))

    def _chi_od_corridor_vub_data(
        self,
        o: str,
        d: str,
        i: str,
        j: str,
    ) -> tuple[tuple[ChiKey, ...], float, Arc]:
        """Return the shared static/dynamic payload for one exact VUB."""

        coefficient = self.inst.h_od[o, d]
        if self.mode == "LO":
            coefficient *= self.inst.choice_share[o, d, j]
        return tuple(self.chi_by_od_corridor[o, d, i, j]), coefficient, (i, j)

    def _chi_od_facility_vub_coefficient(
        self,
        o: str,
        d: str,
        facility: str,
        *,
        head: bool,
    ) -> float:
        """Return the valid OD bound for one incident open-facility link."""

        coefficient = self.inst.h_od[o, d]
        if head and self.mode == "LO":
            coefficient *= self.inst.choice_share[o, d, facility]
        return coefficient

    def _od_disaggregated_return_upper_bound(
        self,
        o: str,
        d: str,
        i: str,
        j: str,
    ) -> float:
        """Return the tight valid OD/corridor coefficient for a return cut."""

        _, od_bound, corridor = self._chi_od_corridor_vub_data(o, d, i, j)
        return min(od_bound, self.M_offer[corridor])

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
        """Separate one globally ranked, bounded round of valid root cuts."""

        tolerance = solver.cut_violation_tolerance
        budget = solver.max_cuts_per_round
        if budget <= 0:
            return 0
        if (
            solver.cut_route_swap_leg_vub
            or solver.cut_route_swap_station_vub
        ) and not self._route_bundle_indices_prepared:
            raise RuntimeError(
                "route bundle indices must be prepared before root separation"
            )

        cfg = self.inst.config
        chi = self.vars["chi"]
        alpha = self.vars["alpha"]
        psi = self.vars["psi"]
        x = self.vars["x"]
        y = self.vars["y"]
        z = self.vars["z"]
        xi = self.vars["xi"]
        varphi = self.vars["varphi"]

        relaxation_cache: dict[str, dict[object, float]] = {}

        def relaxation_values(name: str, container: gp.tupledict) -> dict[object, float]:
            if name not in relaxation_cache:
                relaxation_cache[name] = self._callback_values(
                    callback_model, container
                )
            return relaxation_cache[name]

        candidates: list[dict[str, object]] = []
        candidate_counts: dict[str, int] = defaultdict(int)
        duplicate_counts: dict[str, int] = defaultdict(int)
        rejection_counts: dict[str, int] = defaultdict(int)
        separation_seconds: dict[str, float] = defaultdict(float)

        def retain(
            family: str,
            signature: tuple[object, ...],
            violation: float,
            scale: float,
            payload: object,
        ) -> None:
            if violation <= tolerance:
                return
            candidate_counts[family] += 1
            if signature in self._user_cut_signatures:
                duplicate_counts[family] += 1
                return
            efficacy = violation / max(1.0, abs(scale))
            if efficacy < solver.cut_efficacy_tolerance:
                rejection_counts[family] += 1
                return
            candidates.append(
                {
                    "efficacy": efficacy,
                    "violation": violation,
                    "family": family,
                    "signature": signature,
                    "payload": payload,
                }
            )

        if solver.cut_chi_vub:
            family_started = time.perf_counter()
            y_values = relaxation_values("y", y)
            chi_values = relaxation_values("chi", chi)
            for o, d, i, j in self.chi_by_od_corridor:
                keys, coefficient, corridor = self._chi_od_corridor_vub_data(
                    o, d, i, j
                )
                offered_value = sum(chi_values[key] for key in keys)
                violation = offered_value - coefficient * y_values[corridor]
                retain(
                    "chi_od_corridor_vub",
                    ("chi_od_corridor_vub", o, d, i, j),
                    violation,
                    coefficient,
                    (keys, coefficient, corridor),
                )
            separation_seconds["chi_od_corridor_vub"] += (
                time.perf_counter() - family_started
            )

        if solver.cut_chi_facility_vub:
            family_started = time.perf_counter()
            x_values = relaxation_values("x", x)
            chi_values = relaxation_values("chi", chi)
            for o, d, facility in sorted(self.chi_by_market_incident):
                keys = self.chi_by_market_incident[o, d, facility]
                coefficient = self._chi_od_facility_vub_coefficient(
                    o, d, facility, head=False
                )
                offered_value = sum(chi_values[key] for key in keys)
                violation = offered_value - coefficient * x_values[facility]
                retain(
                    "chi_od_incident_star_vub",
                    ("chi_od_incident_star_vub", o, d, facility),
                    violation,
                    coefficient,
                    (tuple(keys), coefficient, facility),
                )
            separation_seconds["chi_od_incident_star_vub"] += (
                time.perf_counter() - family_started
            )
            if self.mode == "LO":
                family_started = time.perf_counter()
                for o, d, facility in sorted(self.chi_by_market_j):
                    keys = self.chi_by_market_j[o, d, facility]
                    coefficient = self._chi_od_facility_vub_coefficient(
                        o, d, facility, head=True
                    )
                    offered_value = sum(chi_values[key] for key in keys)
                    violation = offered_value - coefficient * x_values[facility]
                    retain(
                        "chi_od_head_facility_vub",
                        ("chi_od_head_facility_vub", o, d, facility),
                        violation,
                        coefficient,
                        (tuple(keys), coefficient, facility),
                    )
                separation_seconds["chi_od_head_facility_vub"] += (
                    time.perf_counter() - family_started
                )

        if solver.cut_route_swap_leg_vub:
            family_started = time.perf_counter()
            y_values = relaxation_values("y", y)
            varphi_values = relaxation_values("varphi", varphi)
            for leg in sorted(self.swap_varphi_by_leg):
                bound = self.swap_route_leg_vub_bounds[leg]
                if not bound.strictly_strengthens():
                    continue
                keys = self.swap_varphi_by_leg[leg]
                evaluated = evaluate_route_bundle_vub(
                    keys,
                    varphi_values,
                    activation_value=y_values[leg],
                    coefficient=bound.coefficient,
                    feasibility_tolerance=max(1e-9, 10.0 * tolerance),
                )
                retain(
                    "route_swap_leg_bundle_vub",
                    ("route_swap_leg_bundle_vub", *leg),
                    evaluated.violation,
                    max(
                        bound.coefficient,
                        evaluated.lhs_value,
                        evaluated.rhs_value,
                    ),
                    (keys, bound.coefficient, leg),
                )
            separation_seconds["route_swap_leg_bundle_vub"] += (
                time.perf_counter() - family_started
            )

        if solver.cut_route_swap_station_vub:
            family_started = time.perf_counter()
            x_values = relaxation_values("x", x)
            varphi_values = relaxation_values("varphi", varphi)
            for station in sorted(self.swap_varphi_by_station):
                bound = self.swap_route_station_vub_bounds[station]
                if not bound.strictly_strengthens():
                    continue
                keys = self.swap_varphi_by_station[station]
                evaluated = evaluate_route_bundle_vub(
                    keys,
                    varphi_values,
                    activation_value=x_values[station],
                    coefficient=bound.coefficient,
                    feasibility_tolerance=max(1e-9, 10.0 * tolerance),
                )
                retain(
                    "route_swap_station_throughput_vub",
                    ("route_swap_station_throughput_vub", station),
                    evaluated.violation,
                    max(
                        bound.coefficient,
                        evaluated.lhs_value,
                        evaluated.rhs_value,
                    ),
                    (keys, bound.coefficient, station),
                )
            separation_seconds["route_swap_station_throughput_vub"] += (
                time.perf_counter() - family_started
            )

        if solver.cut_queue_perspective:
            family_started = time.perf_counter()
            psi_values = relaxation_values("psi", psi)
            xi_values = relaxation_values("xi", xi)
            z_values = relaxation_values("z", z)
            for locker in self.inst.H:
                arrival_value = sum(
                    psi_values[u, v] for u, v in self.inst.A if v == locker
                )
                capacity_values = tuple(
                    z_values[locker, c]
                    for c in range(len(cfg.locker_service_rates))
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
                retain(
                    "queue_perspective",
                    ("queue", locker, *cut.signature()),
                    cut.violation,
                    max(1.0, abs(cut.rhs_at_point), abs(xi_values[locker])),
                    (locker, cut),
                )
            separation_seconds["queue_perspective"] += (
                time.perf_counter() - family_started
            )

        alpha_values: dict[object, float] = {}
        z_values: dict[object, float] = {}
        if self.mode == "LO" and self._choice_cover_data and (
            solver.cut_lo_choice_corridor or solver.cut_lo_choice_capacity
        ):
            alpha_values = relaxation_values("alpha", alpha)
            if solver.cut_lo_choice_capacity:
                z_values = relaxation_values("z", z)

        if (
            self.mode == "LO"
            and solver.cut_lo_choice_corridor
            and self._choice_cover_data
        ):
            family_started = time.perf_counter()
            y_values = relaxation_values("y", y)
            for item in self._choice_cover_data:
                lhs_value = item.offered_demand * alpha_values[item.alpha_key]
                corridor_coefficients = tuple(
                    (arc, min(self.M_offer.get(arc, 0.0), item.offered_demand))
                    for arc in item.incoming_corridors
                    if arc in y_values and self.M_offer.get(arc, 0.0) > 0.0
                )
                rhs_value = sum(
                    coefficient * y_values[arc]
                    for arc, coefficient in corridor_coefficients
                )
                retain(
                    "lo_choice_corridor",
                    ("lo_choice_corridor", item.destination, item.locker),
                    lhs_value - rhs_value,
                    item.offered_demand,
                    (item, corridor_coefficients),
                )
            separation_seconds["lo_choice_corridor"] += (
                time.perf_counter() - family_started
            )

        if (
            self.mode == "LO"
            and solver.cut_lo_choice_capacity
            and self._choice_cover_data
        ):
            family_started = time.perf_counter()
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
                seen_signatures=None,
            )
            for cut in cover_cuts:
                retain(
                    "lo_choice_capacity_cover",
                    cut.signature(),
                    cut.violation,
                    len(cut.destinations),
                    cut,
                )
            separation_seconds["lo_choice_capacity_cover"] += (
                time.perf_counter() - family_started
            )

        if solver.cut_directed_return:
            family_started = time.perf_counter()
            y_values = relaxation_values("y", y)
            chi_values = relaxation_values("chi", chi)
            offered_values = {
                arc: sum(chi_values[key] for key in self.chi_by_corridor[arc])
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
                max_cuts=None,
                seen_signatures=None,
            )
            for cut in return_cuts:
                retain(
                    "directed_return",
                    cut.signature(),
                    cut.violation,
                    max(1.0, cut.offered_value, cut.big_m),
                    cut,
                )
            separation_seconds["directed_return"] += (
                time.perf_counter() - family_started
            )

        if solver.cut_od_disaggregated_return:
            family_started = time.perf_counter()
            y_values = relaxation_values("y", y)
            chi_values = relaxation_values("chi", chi)
            offered_values: dict[tuple[str, str, str, str], float] = {}
            upper_bounds: dict[tuple[str, str, str, str], float] = {}
            for od_corridor in sorted(self.chi_by_od_corridor):
                o, d, i, j = od_corridor
                upper = self._od_disaggregated_return_upper_bound(o, d, i, j)
                if upper <= 0.0:
                    continue
                offered_values[od_corridor] = sum(
                    chi_values[key]
                    for key in self.chi_by_od_corridor[od_corridor]
                )
                upper_bounds[od_corridor] = upper
            od_return_cuts = separate_od_disaggregated_directed_return_cuts(
                self.inst.A_F,
                {arc: y_values[arc] for arc in self.inst.A_F},
                offered_values,
                upper_bounds,
                contraction=self.inst.s_to_h,
                violation_tolerance=tolerance,
                feasibility_tolerance=max(1e-8, 10.0 * tolerance),
                max_corridors=min(100, len(self.corridors)),
                max_cuts=None,
                seen_signatures=None,
            )
            for cut in od_return_cuts:
                retain(
                    "od_disaggregated_directed_return",
                    cut.signature(),
                    cut.violation,
                    max(1.0, cut.offered_value, cut.upper_bound),
                    cut,
                )
            separation_seconds["od_disaggregated_directed_return"] += (
                time.perf_counter() - family_started
            )

        candidates.sort(
            key=lambda item: (
                -float(item["efficacy"]),
                -float(item["violation"]),
                str(item["family"]),
                repr(item["signature"]),
            )
        )
        counts: dict[str, int] = defaultdict(int)
        for candidate in candidates[:budget]:
            family = str(candidate["family"])
            signature = candidate["signature"]
            payload = candidate["payload"]
            if family == "chi_od_corridor_vub":
                keys, coefficient, arc = payload
                callback_model.cbCut(
                    gp.quicksum(chi[key] for key in keys)
                    <= coefficient * y[arc]
                )
            elif family in {
                "chi_od_incident_star_vub",
                "chi_od_head_facility_vub",
            }:
                keys, coefficient, facility = payload
                callback_model.cbCut(
                    gp.quicksum(chi[key] for key in keys)
                    <= coefficient * x[facility]
                )
            elif family == "route_swap_leg_bundle_vub":
                keys, coefficient, leg = payload
                callback_model.cbCut(
                    gp.quicksum(varphi[key] for key in keys)
                    <= coefficient * y[leg]
                )
            elif family == "route_swap_station_throughput_vub":
                keys, coefficient, station = payload
                callback_model.cbCut(
                    gp.quicksum(varphi[key] for key in keys)
                    <= coefficient * x[station]
                )
            elif family == "queue_perspective":
                locker, cut = payload
                incoming = gp.quicksum(
                    psi[u, v] for u, v in self.inst.A if v == locker
                )
                callback_model.cbCut(
                    xi[locker]
                    >= cut.lambda_coefficient * incoming
                    + gp.quicksum(
                        cut.z_coefficients[c] * z[locker, c]
                        for c in range(len(cut.z_coefficients))
                    )
                )
            elif family == "lo_choice_corridor":
                item, corridor_coefficients = payload
                callback_model.cbCut(
                    item.offered_demand * alpha[item.alpha_key]
                    <= gp.quicksum(
                        coefficient * y[arc]
                        for arc, coefficient in corridor_coefficients
                    )
                )
            elif family == "lo_choice_capacity_cover":
                cut = payload
                callback_model.cbCut(
                    gp.quicksum(
                        alpha[d, cut.locker] for d in cut.destinations
                    )
                    <= len(cut.destinations)
                    - 1
                    + gp.quicksum(
                        z[cut.locker, c]
                        for c in cut.adequate_capacity_indices
                    )
                )
            elif family == "directed_return":
                cut = payload
                callback_model.cbCut(
                    gp.quicksum(
                        chi[key] for key in self.chi_by_corridor[cut.corridor]
                    )
                    <= cut.big_m
                        * gp.quicksum(y[arc] for arc in cut.crossing_arcs)
                )
            elif family == "od_disaggregated_directed_return":
                cut = payload
                callback_model.cbCut(
                    gp.quicksum(
                        chi[key]
                        for key in self.chi_by_od_corridor[cut.od_corridor]
                    )
                    <= cut.upper_bound
                    * gp.quicksum(y[arc] for arc in cut.crossing_arcs)
                )
            else:  # pragma: no cover - guarded by construction above
                raise RuntimeError(f"unknown cut family {family!r}")
            self._user_cut_signatures.add(signature)
            counts[family] += 1

        for family, count in counts.items():
            self.user_cut_counts[family] = self.user_cut_counts.get(family, 0) + count
        for family, count in candidate_counts.items():
            self.user_cut_candidate_counts[family] = (
                self.user_cut_candidate_counts.get(family, 0) + count
            )
        for family, count in duplicate_counts.items():
            self.user_cut_duplicate_counts[family] = (
                self.user_cut_duplicate_counts.get(family, 0) + count
            )
        for family, count in rejection_counts.items():
            self.user_cut_rejection_counts[family] = (
                self.user_cut_rejection_counts.get(family, 0) + count
            )
        for family, seconds in separation_seconds.items():
            self.user_cut_separation_seconds[family] = (
                self.user_cut_separation_seconds.get(family, 0.0) + seconds
            )
        return sum(counts.values())

    @staticmethod
    def _finite_callback_value(value: object) -> float | None:
        number = float(value)
        if not math.isfinite(number) or abs(number) >= 0.99 * GRB.INFINITY:
            return None
        return number

    @staticmethod
    def _relative_gap(incumbent: float | None, bound: float | None) -> float | None:
        if incumbent is None or bound is None:
            return None
        return abs(incumbent - bound) / max(1e-10, abs(incumbent))

    def _record_mip_progress(self, callback_model: gp.Model, solver: SolverConfig) -> None:
        runtime = float(callback_model.cbGet(GRB.Callback.RUNTIME))
        incumbent = self._finite_callback_value(
            callback_model.cbGet(GRB.Callback.MIP_OBJBST)
        )
        bound = self._finite_callback_value(callback_model.cbGet(GRB.Callback.MIP_OBJBND))
        materially_changed = (
            incumbent != self._last_progress_incumbent
            or bound != self._last_progress_bound
        )
        if (
            runtime - self._last_progress_time < solver.progress_interval_seconds
            and not materially_changed
        ):
            return
        self.progress_events.append(
            {
                "event": "mip_progress",
                "runtime_seconds": runtime,
                "incumbent": incumbent,
                "bound": bound,
                "relative_gap": self._relative_gap(incumbent, bound),
                "node_count": float(
                    callback_model.cbGet(GRB.Callback.MIP_NODCNT)
                ),
                "solution_count": int(
                    callback_model.cbGet(GRB.Callback.MIP_SOLCNT)
                ),
            }
        )
        self._last_progress_time = runtime
        self._last_progress_incumbent = incumbent
        self._last_progress_bound = bound

    def _solver_callback(self, solver: SolverConfig):
        """Create the matched telemetry callback with optional root separation."""

        cuts_enabled = (
            solver.use_user_cuts
            and solver.max_root_cut_rounds > 0
            and solver.max_cuts_per_round > 0
        )

        def callback(callback_model: gp.Model, where: int) -> None:
            callback_started = time.perf_counter()
            try:
                if where == GRB.Callback.MIP and solver.collect_progress:
                    self._record_mip_progress(callback_model, solver)
                    return
                if where == GRB.Callback.MIPSOL and solver.collect_progress:
                    runtime = float(callback_model.cbGet(GRB.Callback.RUNTIME))
                    objective = self._finite_callback_value(
                        callback_model.cbGet(GRB.Callback.MIPSOL_OBJ)
                    )
                    bound = self._finite_callback_value(
                        callback_model.cbGet(GRB.Callback.MIPSOL_OBJBND)
                    )
                    if self.first_incumbent_time is None:
                        self.first_incumbent_time = runtime
                        self.first_incumbent_objective = objective
                    self.progress_events.append(
                        {
                            "event": "incumbent",
                            "runtime_seconds": runtime,
                            "incumbent": objective,
                            "bound": bound,
                            "relative_gap": self._relative_gap(objective, bound),
                        }
                    )
                    return
                if where != GRB.Callback.MIPNODE:
                    return
                if callback_model.cbGet(GRB.Callback.MIPNODE_STATUS) != GRB.OPTIMAL:
                    return
                node_count = float(
                    callback_model.cbGet(GRB.Callback.MIPNODE_NODCNT)
                )
                if node_count > 0.5:
                    return
                bound = self._finite_callback_value(
                    callback_model.cbGet(GRB.Callback.MIPNODE_OBJBND)
                )
                if self.root_bound_before_cuts is None:
                    self.root_bound_before_cuts = bound
                self.root_bound_after_cuts = bound
                if not cuts_enabled or self.root_cut_rounds >= solver.max_root_cut_rounds:
                    return
                self.root_cut_rounds += 1
                before_counts = dict(self.user_cut_counts)
                added = self._add_root_user_cuts(callback_model, solver)
                added_by_family = {
                    family: count - before_counts.get(family, 0)
                    for family, count in self.user_cut_counts.items()
                    if count - before_counts.get(family, 0) > 0
                }
                self.root_round_events.append(
                    {
                        "round": self.root_cut_rounds,
                        "runtime_seconds": float(
                            callback_model.cbGet(GRB.Callback.RUNTIME)
                        ),
                        "root_bound_before_round": bound,
                        "cuts_added": added,
                        "cuts_added_by_family": added_by_family,
                    }
                )
            finally:
                self.callback_runtime += time.perf_counter() - callback_started

        return callback

    def solve(
        self,
        solver: SolverConfig | None = None,
        *,
        external_start: Mapping[str, float] | None = None,
        external_start_source: str | None = None,
    ) -> bool:
        solver = solver or SolverConfig()
        solver.validate()
        self.solver_config = solver
        # Model/index construction is common to every matched profile and is
        # deliberately outside the solver budget.  Static strengthening,
        # start preparation, callbacks, and the main optimization are inside.
        self.build()
        budget_wall_start = time.perf_counter()
        algorithm_deadline = budget_wall_start + solver.time_limit
        self.algorithm_prepare_runtime = 0.0
        self.network_start_runtime = 0.0
        self.network_construction_runtime = 0.0
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
        self.network_start_open_h_count = 0
        self.network_start_open_facility_count = 0
        self.network_start_service_per_hour = 0.0
        self.network_start_relevant_chi_count = 0
        self.start_preparation_runtime = 0.0
        self.objective_partial_start_telemetry = self._empty_objective_start_telemetry()
        self.local_branching_runtime = 0.0
        self.local_branching_improved = False
        self.local_branching_objective = None
        self.local_branching_status = None
        self.local_branching_solution_count = 0
        self.local_branching_hamming_distance = None
        self.fixed_integer_repair_runtime = 0.0
        self.fixed_integer_repair_found = False
        self.fixed_integer_repair_objective = None
        self.fixed_integer_repair_status = None
        self.fixed_integer_repair_solution_count = 0
        self.fixed_integer_repair_fixed_count = 0
        self.main_solve_runtime = 0.0
        self.main_solve_started = False
        self.optimize_called = False
        self.total_runtime = 0.0
        self.budget_exhausted_before_main_solve = False
        self.user_cut_counts = {}
        self.user_cut_candidate_counts = {}
        self.user_cut_duplicate_counts = {}
        self.user_cut_rejection_counts = {}
        self.user_cut_separation_seconds = {}
        self.root_cut_rounds = 0
        self.root_bound_before_cuts = None
        self.root_bound_after_cuts = None
        self.root_round_events = []
        self.progress_events = []
        self.callback_runtime = 0.0
        self.first_incumbent_time = None
        self.first_incumbent_objective = None
        self._last_progress_time = -math.inf
        self._last_progress_incumbent = None
        self._last_progress_bound = None
        self._user_cut_signatures = set(
            getattr(self, "_static_user_cut_signatures", ())
        )
        try:
            self.prepare_algorithm(solver)
            self.model.Params.TimeLimit = solver.time_limit
            self.model.Params.MIPGap = solver.mip_gap
            self.model.Params.Threads = solver.threads
            self.model.Params.OutputFlag = solver.output_flag
            self.model.Params.Seed = solver.seed
            self.model.Params.Method = solver.root_relaxation_method
            self.model.Params.Presolve = solver.presolve
            self.model.Params.Cuts = solver.cuts
            self.model.Params.NumericFocus = solver.numeric_focus
            self.model.Params.FeasibilityTol = solver.feasibility_tol
            self.model.Params.OptimalityTol = solver.optimality_tol
            self.model.Params.IntFeasTol = solver.integer_feasibility_tol
            self.model.Params.MIQCPMethod = solver.miqcp_method

            remaining_before_start = algorithm_deadline - time.perf_counter()
            if remaining_before_start <= 0.0:
                self.budget_exhausted_before_main_solve = True
            else:
                if solver.use_objective_partial_start:
                    start_preparation_started = time.perf_counter()
                    objective_budget = min(
                        effective_objective_start_budget(
                            solver.time_limit,
                            solver.objective_partial_start_seconds,
                        ),
                        remaining_before_start,
                    )
                    self._construct_objective_partial_starts(
                        objective_budget,
                        solver,
                    )
                    self.start_preparation_runtime = (
                        time.perf_counter() - start_preparation_started
                    )
                else:
                    maximum_start_budget = min(
                        0.20 * solver.time_limit,
                        remaining_before_start,
                    )
                    network_budget = min(
                        solver.network_start_seconds,
                        maximum_start_budget,
                    )
                    local_budget = (
                        solver.local_branching_seconds
                        if solver.use_local_branching
                        else 0.0
                    )
                    repair_budget = solver.fixed_integer_repair_seconds
                    no_start_fast_path = (
                        solver.start_mode == "none" and int(self.model.NumStart) == 0
                    )
                    if not no_start_fast_path:
                        start_preparation_started = time.perf_counter()
                        if solver.start_mode == "none":
                            self._clear_mip_starts()
                        elif solver.start_mode == "courier":
                            self._set_courier_only_start()
                        elif solver.start_mode in {"network", "network_and_courier"}:
                            include_courier = solver.start_mode == "network_and_courier"
                            if network_budget > 1e-9 and self.chi_keys:
                                self._construct_network_start_with_budgets(
                                    network_budget,
                                    repair_budget,
                                    local_budget,
                                    maximum_start_budget,
                                    solver,
                                    include_courier_start=include_courier,
                                )
                            elif include_courier:
                                self._set_courier_only_start()
                            else:
                                self._clear_mip_starts()
                        elif network_budget > 1e-9 and self.chi_keys:
                            self._construct_network_start_with_budgets(
                                network_budget,
                                repair_budget,
                                local_budget,
                                maximum_start_budget,
                                solver,
                                include_courier_start=True,
                            )
                        else:
                            self._set_courier_only_start()
                        self.start_preparation_runtime = (
                            time.perf_counter() - start_preparation_started
                        )

            if external_start is not None:
                external_started = time.perf_counter()
                self._append_external_start(
                    external_start,
                    external_start_source,
                )
                self.start_preparation_runtime += (
                    time.perf_counter() - external_started
                )

            # Auxiliary solves restore these transactionally.  Set the full
            # matched A--E parameter contract again at the main-solve boundary,
            # even when no budget remains and optimize must be skipped.
            self.model.Params.MIPGap = solver.mip_gap
            self.model.Params.Method = solver.root_relaxation_method
            self.model.Params.MIQCPMethod = solver.miqcp_method
            self.model.Params.CutPasses = solver.root_cut_passes
            legacy_heuristics = 0.5 if self.network_start_used else 0.25
            self.model.Params.MIPFocus = (
                1 if solver.mip_focus is None else solver.mip_focus
            )
            self.model.Params.Heuristics = (
                legacy_heuristics if solver.heuristics is None else solver.heuristics
            )
            self.model.Params.NoRelHeurTime = (
                min(
                    solver.no_rel_heur_seconds,
                    0.05
                    * max(
                        0.1,
                        solver.time_limit - self.network_start_runtime,
                    ),
                )
                if solver.no_rel_heur_time is None
                else solver.no_rel_heur_time
            )
            cuts_enabled = (
                solver.use_user_cuts
                and solver.max_root_cut_rounds > 0
                and solver.max_cuts_per_round > 0
            )
            if cuts_enabled:
                self.model.Params.PreCrush = 1

            remaining = algorithm_deadline - time.perf_counter()
            if remaining <= 0.0:
                self.budget_exhausted_before_main_solve = True
                self.model.Params.TimeLimit = 0.0
            else:
                self.model.Params.TimeLimit = remaining
                main_solve_wall_started = time.perf_counter()
                self.main_solve_started = True
                self.optimize_called = True
                try:
                    if cuts_enabled or solver.collect_progress:
                        self.model.optimize(self._solver_callback(solver))
                    else:
                        self.model.optimize()
                finally:
                    try:
                        self.main_solve_runtime = float(self.model.Runtime)
                    except MemoryError:
                        raise
                    except Exception:
                        self.main_solve_runtime = max(
                            0.0,
                            time.perf_counter() - main_solve_wall_started,
                        )
        finally:
            self.total_runtime = time.perf_counter() - budget_wall_start

        if solver.collect_progress:
            if self.optimize_called:
                solution_count = int(self.model.SolCount)
                incumbent = (
                    self._finite_callback_value(self.model.ObjVal)
                    if solution_count > 0
                    else None
                )
                bound = self._finite_callback_value(self.model.ObjBound)
                node_count = float(self.model.NodeCount)
                status = int(self.model.Status)
            else:
                solution_count = 0
                incumbent = None
                bound = None
                node_count = 0.0
                status = int(GRB.LOADED)
            if self.first_incumbent_time is None and incumbent is not None:
                self.first_incumbent_time = self.main_solve_runtime
                self.first_incumbent_objective = incumbent
            self.progress_events.append(
                {
                    "event": "final",
                    "runtime_seconds": self.main_solve_runtime,
                    "total_algorithm_runtime_seconds": self.total_runtime,
                    "budget_exhausted_before_main_solve": (
                        self.budget_exhausted_before_main_solve
                    ),
                    "main_solve_started": self.main_solve_started,
                    "optimize_called": self.optimize_called,
                    "incumbent": incumbent,
                    "bound": bound,
                    "relative_gap": self._relative_gap(incumbent, bound),
                    "node_count": node_count,
                    "solution_count": solution_count,
                    "status": status,
                }
            )
        if not self.optimize_called:
            return False
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

    def _objective_start_cycle_support(
        self,
        corridor: Arc,
        *,
        deadline: float | None = None,
    ) -> tuple[
        tuple[str, ...],
        tuple[Arc, ...],
        str,
        float,
        float,
        tuple[str, ...],
        tuple[tuple[str, float], ...],
        float,
        float,
    ] | None:
        """Return the cheapest cycle valid for f884's system-wide energy pool."""

        i, j = corridor
        cfg = self.inst.config
        usable_time = cfg.usable_battery_time_hours
        candidates: list[
            tuple[
                tuple[float, float, float, str, tuple[str, ...]],
                tuple[str, ...],
                tuple[Arc, ...],
                str,
                float,
                float,
                tuple[str, ...],
                tuple[tuple[str, float], ...],
                float,
                float,
            ]
        ] = []
        for base in sorted(self.inst.L):
            if deadline is not None and time.perf_counter() > deadline:
                raise TimeoutError("objective partial-start preparation timed out")
            cycle_arcs = ((base, i), (i, j), (j, base))
            if any(arc not in self.inst.A_F for arc in cycle_arcs):
                continue
            option_sets = tuple(
                tuple(sorted(self.inst.route_options[arc]))
                for arc in cycle_arcs
            )
            if any(not options for options in option_sets):
                continue
            for selected_options in itertools.product(*option_sets):
                if deadline is not None and time.perf_counter() > deadline:
                    raise TimeoutError(
                        "objective partial-start preparation timed out"
                    )
                facilities = {base, i, j}
                active_connections: set[Arc] = set(cycle_arcs)
                route_distance = 0.0
                route_time = 0.0
                incoming_h: set[str] = set()
                outgoing_route_time_by_h: dict[str, float] = {}
                swap_events = 0.0
                feasible = True
                for (u, v), option in zip(cycle_arcs, selected_options):
                    if deadline is not None and time.perf_counter() > deadline:
                        raise TimeoutError(
                            "objective partial-start preparation timed out"
                        )
                    if option != DIRECT_ROUTE:
                        if (
                            (u, option) not in self.inst.A_Y
                            or (option, v) not in self.inst.A_Y
                        ):
                            feasible = False
                            break
                        facilities.add(option)
                        active_connections.update(((u, option), (option, v)))
                        swap_events += 1.0
                    route_distance += self.inst.route_distance[u, v, option]
                    selected_route_time = self.inst.route_time[u, v, option]
                    route_time += selected_route_time
                    if v in self.inst.H:
                        incoming_h.add(v)
                    if u in self.inst.H:
                        outgoing_route_time_by_h[u] = selected_route_time
                if not feasible:
                    continue
                base_departures = float(
                    sum(u in self.inst.L for u, _v in cycle_arcs)
                )
                # Queue population is nonnegative, so a route that already
                # exceeds the exact system pool cannot become feasible later.
                if (
                    route_time
                    > usable_time * (base_departures + swap_events) + 1e-9
                ):
                    continue
                energy_reserve = (
                    usable_time * (base_departures + swap_events) - route_time
                )
                fixed_proxy = math.fsum(
                    self.inst.facility_cost(node) for node in facilities
                ) + math.fsum(
                    self.inst.arc_activation_cost[arc]
                    for arc in active_connections
                )
                variable_proxy = (
                    cfg.flight_cost_per_km * route_distance
                    + cfg.drone_cost_per_hour * route_time
                )
                # Prefer the cheapest route once it reserves a modest quarter
                # battery-hour fraction for queue population.  If no option
                # reaches that reserve, retain the energy-richest route.  The
                # exact aggregate gate in evaluate_objective_support remains
                # authoritative, so this ranking never certifies feasibility.
                candidates.append(
                    (
                        (
                            float(energy_reserve + 1e-9 < 0.25 * usable_time),
                            (
                                fixed_proxy + variable_proxy
                                if energy_reserve + 1e-9 >= 0.25 * usable_time
                                else -energy_reserve
                            ),
                            fixed_proxy + variable_proxy,
                            base,
                            tuple(selected_options),
                        ),
                        tuple(sorted(facilities)),
                        tuple(sorted(active_connections)),
                        base,
                        route_distance,
                        route_time,
                        tuple(sorted(incoming_h)),
                        tuple(sorted(outgoing_route_time_by_h.items())),
                        base_departures,
                        swap_events,
                    )
                )
        if not candidates:
            return None
        (
            _,
            facilities,
            active_connections,
            base,
            distance,
            route_time,
            incoming_h,
            outgoing_route_times,
            base_departures,
            swap_events,
        ) = min(candidates)
        return (
            facilities,
            active_connections,
            base,
            distance,
            route_time,
            incoming_h,
            outgoing_route_times,
            base_departures,
            swap_events,
        )

    def _objective_start_market_costs(
        self,
        market: Market,
        corridor: Arc,
    ) -> tuple[float, float, float, float, float]:
        """Return offered demand and rider/pickup costs for one proxy option."""

        origin, destination = market
        i, j = corridor
        demand = self.inst.h_od[market]
        first = self.inst.feeder_rider_cost(origin, i)
        direct = self.inst.direct_rider_cost(origin, destination)
        if self.mode == "LO":
            offered = demand * self.inst.choice_share.get((origin, destination, j), 0.0)
            return offered, direct, first, 0.0, self.inst.locker_pickup_cost(j)
        last = self.inst.feeder_rider_cost(j, destination)
        # MX alpha remains undefined.  Conservatively charge the full last-mile
        # rider cost instead of crediting mutually incompatible display lockers
        # independently across OD markets.
        return demand, direct, first, last, 0.0

    def _build_objective_start_opportunities(
        self,
        *,
        deadline: float,
        max_candidates: int,
    ) -> tuple[list[ObjectiveAwareOpportunity], dict[str, int]]:
        """Create a deterministic short list without solving an auxiliary model."""

        cfg = self.inst.config

        def check_budget() -> None:
            if time.perf_counter() > deadline:
                raise TimeoutError("objective partial-start preparation timed out")

        rough: list[tuple[float, float, Arc]] = []
        for corridor in self.corridors:
            check_budget()
            i, j = corridor
            markets = sorted({(key[0], key[1]) for key in self.chi_by_corridor[corridor]})
            saving = 0.0
            volume = 0.0
            for market in markets:
                offered, direct, first, last, pickup = self._objective_start_market_costs(
                    market, corridor
                )
                if offered <= 0.0:
                    continue
                flight_lower = cfg.flight_cost_per_km * self.inst.distance(i, j)
                saving += offered * max(
                    0.0,
                    direct - first - last - pickup - flight_lower,
                )
                volume += offered
            rough.append((saving, volume, corridor))
        rough.sort(key=lambda item: (-item[0], -item[1], item[2]))
        # The downstream planner can retain at most ``max_candidates`` starts.
        # A deterministic three-to-one corridor pool leaves room for alternate
        # supports without spending the five-percent budget materializing OD
        # opportunities that cannot reach the installed start pool.
        shortlist_size = min(len(rough), max(6, 3 * max_candidates))
        shortlisted = rough[:shortlist_size]

        opportunities: list[ObjectiveAwareOpportunity] = []
        supported_cycles = 0
        for _, _, corridor in shortlisted:
            check_budget()
            support = self._objective_start_cycle_support(
                corridor,
                deadline=deadline,
            )
            if support is None:
                continue
            supported_cycles += 1
            (
                facilities,
                active_connections,
                base,
                route_distance,
                route_time,
                incoming_h,
                outgoing_route_times,
                base_departures,
                swap_events,
            ) = support
            facility_costs = tuple(
                (node, self.inst.facility_cost(node)) for node in facilities
            )
            connection_costs = tuple(
                (arc, self.inst.arc_activation_cost[arc])
                for arc in active_connections
            )
            markets = sorted({(key[0], key[1]) for key in self.chi_by_corridor[corridor]})
            for reliability_index, reliability in enumerate(cfg.availability_levels):
                for market in markets:
                    check_budget()
                    offered, direct, first, last, pickup = (
                        self._objective_start_market_costs(market, corridor)
                    )
                    if offered <= 0.0:
                        continue
                    opportunities.append(
                        ObjectiveAwareOpportunity(
                            market=market,
                            corridor=corridor,
                            reliability_index=reliability_index,
                            reliability=reliability,
                            offered_demand=offered,
                            support_facilities=facilities,
                            open_h=tuple(sorted(corridor)),
                            active_connections=active_connections,
                            facility_costs=facility_costs,
                            connection_costs=connection_costs,
                            idle_state=base,
                            incoming_h=incoming_h,
                            outgoing_route_time_hours=outgoing_route_times,
                            display_locker=j if self.mode == "LO" else None,
                            display_blockers=(
                                tuple(self.inst.closer_lockers[market[1], j])
                                if self.mode == "LO"
                                else ()
                            ),
                            direct_unit_cost=direct,
                            first_mile_unit_cost=first,
                            last_mile_unit_cost=last,
                            mx_pickup_unit_cost=pickup,
                            flight_unit_cost=cfg.flight_cost_per_km * route_distance,
                            route_time_hours=route_time,
                            base_departures_per_unit=base_departures,
                            swap_events_per_unit=swap_events,
                        )
                    )
        return opportunities, {
            "eligible_corridor_count": len(rough),
            "shortlisted_corridor_count": len(shortlisted),
            "supported_cycle_count": supported_cycles,
            "mx_alpha_key_count": len(self.alpha_keys),
        }

    def _construct_objective_partial_starts(
        self,
        budget_seconds: float,
        solver: SolverConfig,
    ) -> None:
        """Install bounded objective-aware partial starts without optimizing."""

        if self.mode not in _OBJECTIVE_PARTIAL_START_MODES:
            supported = ", ".join(sorted(_OBJECTIVE_PARTIAL_START_MODES))
            raise ValueError(
                "objective-partial start is supported only for modes "
                f"{supported}; got {self.mode!r}"
            )

        started = time.perf_counter()
        telemetry = self._empty_objective_start_telemetry()
        telemetry.update(
            {
                "enabled": True,
                "budget_seconds": budget_seconds,
                "legacy_start_mode_ignored": solver.start_mode,
            }
        )
        self.objective_partial_start_telemetry = telemetry
        if budget_seconds <= 0.0:
            self._clear_mip_starts()
            telemetry["timed_out"] = True
            telemetry["fallback_reason"] = "zero_preparation_budget"
            telemetry["rejection_reasons"] = ["budget_exhausted"]
            telemetry["preparation_seconds"] = time.perf_counter() - started
            return

        deadline = started + budget_seconds
        try:
            opportunities, construction_counts = (
                self._build_objective_start_opportunities(
                    deadline=deadline,
                    max_candidates=solver.objective_partial_start_max_candidates,
                )
            )
            telemetry.update(construction_counts)
            remaining = max(0.0, deadline - time.perf_counter())
            direct_baseline = max(self.courier_only_baseline, 0.0)
            planning_config = ObjectiveAwarePlanningConfig(
                candidate_lockers=tuple(sorted(self.inst.H)),
                capacity_options=tuple(
                    CapacityOption(index, rate, cfg_cost)
                    for index, (rate, cfg_cost) in enumerate(
                        zip(
                            self.inst.config.locker_service_rates,
                            self.inst.config.locker_capacity_costs,
                        )
                    )
                ),
                max_locker_time_hours=self.inst.config.max_locker_time_hours,
                usable_battery_time_hours=(
                    self.inst.config.usable_battery_time_hours
                ),
                drone_cost_per_hour=self.inst.config.drone_cost_per_hour,
                max_candidates=solver.objective_partial_start_max_candidates,
                min_open_h=2,
                min_service=max(1e-6, 0.01 * self.inst.total_demand),
                min_savings=max(1e-6, 1e-4 * direct_baseline),
            )
            plan = plan_objective_aware_partial_starts(
                opportunities,
                planning_config,
                budget_seconds=remaining,
            )
            telemetry.update(
                {
                    "opportunity_count": plan.opportunity_count,
                    "singleton_support_count": plan.singleton_support_count,
                    "evaluated_support_count": plan.evaluated_support_count,
                    "network_candidate_count": len(plan.candidates),
                    "heuristic_candidate_count": len(plan.candidates),
                    "rejection_reasons": list(plan.rejection_reasons),
                    "timed_out": plan.timed_out,
                }
            )
            if plan.timed_out or not plan.candidates:
                self._clear_mip_starts()
                telemetry["fallback_reason"] = (
                    "budget_exhausted" if plan.timed_out else "no_reliable_network_candidate"
                )
                return

            assignments = [
                build_candidate_partial_assignment(
                    candidate,
                    self.vars,
                    mode=self.mode,
                )
                for candidate in plan.candidates
            ]
            # Network hints occupy the first start slots.  The all-courier
            # completion is a last-resort safety start, and exists only because
            # at least one material network candidate passed every gate above.
            assignments.append(build_all_courier_partial_assignment(self.vars))
            install_partial_mip_starts(self.model, assignments)
            elapsed = time.perf_counter() - started
            if elapsed > budget_seconds + 1e-9:
                self._clear_mip_starts()
                telemetry["timed_out"] = True
                telemetry["fallback_reason"] = "budget_exhausted_after_install"
                telemetry["rejection_reasons"] = list(
                    dict.fromkeys(
                        [*telemetry["rejection_reasons"], "budget_exhausted"]
                    )
                )
                return
            telemetry.update(
                {
                    "used": True,
                    "safety_start_count": 1,
                    "installed_start_count": len(assignments),
                    "candidate_supports": [
                        {
                            "candidate_id": candidate.candidate_id,
                            "move": candidate.move,
                            "selected_supports": [
                                list(item) for item in candidate.selected_supports
                            ],
                            "open_h": list(candidate.open_h),
                            "open_facilities": list(candidate.open_facilities),
                            "active_connections": [
                                list(item) for item in candidate.active_connections
                            ],
                            "capacity_levels": [
                                list(item) for item in candidate.capacity_levels
                            ],
                            "forced_closed_h": list(candidate.forced_closed_h),
                            "forced_off_connections": [
                                list(item)
                                for item in candidate.forced_off_connections
                            ],
                            "estimated_service": candidate.estimated_service,
                            "estimated_service_fraction": (
                                candidate.service_fraction
                            ),
                            "fraction_basis_service": (
                                candidate.estimated_service
                                / candidate.service_fraction
                            ),
                            "proxy_only": True,
                            "score": candidate.score,
                            "objective_breakdown": asdict(candidate.objective),
                            "energy_proxy": {
                                "system_energy_use": candidate.system_energy_use,
                                "system_energy_supply": candidate.system_energy_supply,
                                "base_departure_population": (
                                    candidate.base_departure_population
                                ),
                                "swap_event_population": (
                                    candidate.swap_event_population
                                ),
                            },
                            "assigned_variable_count": len(assignments[index]),
                        }
                        for index, candidate in enumerate(plan.candidates)
                    ],
                }
            )
        except MemoryError:
            raise
        except TimeoutError:
            self._clear_mip_starts()
            telemetry["timed_out"] = True
            telemetry["fallback_reason"] = "budget_exhausted"
            telemetry["rejection_reasons"] = ["budget_exhausted"]
        except Exception as exc:
            # This development heuristic must never jeopardize the exact solve.
            # The exception type is enough to audit the fail-closed fallback;
            # instance contents and environment details are deliberately omitted.
            self._clear_mip_starts()
            telemetry["fallback_reason"] = f"unreliable_{type(exc).__name__}"
            telemetry["rejection_reasons"] = ["unreliable_preparation"]
        finally:
            telemetry["preparation_seconds"] = time.perf_counter() - started

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
        solver: SolverConfig | None = None,
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
        active_solver = solver or SolverConfig()

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
            self.model.Params.Method = active_solver.network_start_method
            self.model.Params.MIQCPMethod = active_solver.miqcp_method
            self.model.update()
            self.model.optimize()
            if self.model.SolCount > 0:
                objective = float(self.model.ObjVal)
                values = {variable: float(variable.X) for variable in model_variables}
                attempt = _NetworkStartAttempt(
                    seed=capture_incumbent(
                        self.model,
                        self.vars,
                        include_y=active_solver.local_branching_include_y,
                    ),
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

    def _construct_network_start_with_budgets(
        self,
        network_time_limit: float,
        fixed_integer_repair_time_limit: float,
        local_branching_time_limit: float,
        maximum_start_budget: float,
        solver: SolverConfig,
        *,
        include_courier_start: bool,
    ) -> None:
        """Construct an f884 network start inside one absolute preparation budget."""

        start_wall_time = time.perf_counter()
        network_deadline = start_wall_time + network_time_limit
        overall_deadline = start_wall_time + maximum_start_budget
        attempt: _NetworkStartAttempt | None = None
        relevant: tuple[ChiKey, ...] = ()

        def install_fallback_start() -> None:
            if include_courier_start:
                self._set_courier_only_start()
            else:
                self._clear_mip_starts()

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
                remaining = network_deadline - time.perf_counter()
                attempts_left = len(shortlisted) - index
                attempt_time = remaining / attempts_left
                if remaining < 0.1 or attempt_time < 0.1:
                    break
                attempted_keys.append(candidate.alpha_key)
                self.network_start_attempted_keys = tuple(attempted_keys)
                self.network_start_key = candidate.alpha_key
                relevant_by_destination = getattr(
                    self,
                    "chi_by_destination_locker",
                    {},
                )
                relevant = tuple(
                    relevant_by_destination.get(candidate.alpha_key, ())
                )
                attempt = self._solve_network_start_attempt(
                    time_limit=attempt_time,
                    solver=solver,
                    alpha_target=candidate.alpha_key,
                )
                if attempt is not None:
                    break
        else:
            # MX and HD retain the f884 exact auxiliary: one profitable H-H
            # corridor, one required physical cycle, and a broad but bounded H
            # support.  Only the temporary auxiliary bounds are restricted.
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
                        - self.inst.config.flight_cost_per_km
                        * self.inst.distance(i, j)
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
                remaining = network_deadline - time.perf_counter()
                auxiliary_time = min(
                    remaining,
                    max(10.0, 0.75 * network_time_limit),
                )
                if auxiliary_time >= 0.1:
                    attempt = self._solve_network_start_attempt(
                        time_limit=auxiliary_time,
                        solver=solver,
                        alpha_target=None,
                        required_physical_nodes=required_physical_nodes,
                        allowed_physical_nodes=allowed_physical_nodes,
                        positive_chi_keys=relevant,
                        positive_offer=positive_offer,
                    )

        self.network_construction_runtime = time.perf_counter() - start_wall_time
        seed: PrimalSeed | None = None
        if attempt is not None:
            seed = attempt.seed
            self.network_start_found = True
            self.network_start_used = True
            self.network_start_objective = attempt.objective
            seed_x = seed.structure["x"]
            self.network_start_seed_open_counts = {
                "bases": sum(seed_x[node] for node in self.inst.L),
                "lockers": sum(seed_x[node] for node in self.inst.H),
                "swap_stations": sum(seed_x[node] for node in self.inst.E),
            }
            self.network_start_open_h_count = self.network_start_seed_open_counts[
                "lockers"
            ]
            self.network_start_open_facility_count = sum(
                self.network_start_seed_open_counts.values()
            )
            self.network_start_relevant_chi_count = len(relevant)
            levels = getattr(
                getattr(self.inst, "config", None),
                "availability_levels",
                (),
            )
            chi = self.vars.get("chi")
            self.network_start_service_per_hour = (
                math.fsum(
                    levels[key[4]] * float(attempt.values[chi[key]])
                    for key in relevant
                )
                if relevant and levels and chi is not None
                else 0.0
            )

        remaining_start_budget = max(0.0, overall_deadline - time.perf_counter())
        repair_time = min(
            fixed_integer_repair_time_limit,
            remaining_start_budget,
        )
        start_is_installed = False
        if seed is not None and repair_time >= 0.1:
            repair = repair_fixed_integer_design_start(
                self.model,
                seed,
                FixedIntegerRepairConfig(
                    time_limit=repair_time,
                    mip_gap=min(0.01, solver.mip_gap),
                    method=solver.network_start_method,
                    miqcp_method=solver.miqcp_method,
                    integrality_tolerance=solver.integer_feasibility_tol,
                    output_flag=solver.output_flag,
                ),
            )
            self.fixed_integer_repair_runtime = repair.runtime
            self.fixed_integer_repair_found = repair.found_incumbent
            self.fixed_integer_repair_objective = repair.installed_objective
            self.fixed_integer_repair_status = repair.status
            self.fixed_integer_repair_solution_count = repair.solution_count
            self.fixed_integer_repair_fixed_count = repair.fixed_integer_count
            self.network_start_objective = repair.installed_objective
            start_is_installed = True

        remaining_after_repair = max(0.0, overall_deadline - time.perf_counter())
        local_time = min(
            local_branching_time_limit if repair_time < 0.1 else 0.0,
            remaining_after_repair,
        )
        if seed is not None and solver.use_local_branching and local_time >= 0.1:
            result = improve_mip_start(
                self.model,
                self.vars,
                seed,
                LocalBranchingConfig(
                    radius=solver.local_branching_radius,
                    time_limit=local_time,
                    mip_gap=min(0.01, solver.mip_gap),
                    include_y=solver.local_branching_include_y,
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
            start_is_installed = True
        elif self.network_start_used and not start_is_installed:
            if seed is None:
                raise RuntimeError("network start was marked used without a seed")
            model_variables = self.model.getVars()
            self.model.NumStart = 1
            self.model.update()
            self.model.Params.StartNumber = 0
            self.model.setAttr(
                GRB.Attr.Start,
                model_variables,
                [float(seed.values[variable]) for variable in model_variables],
            )
            self.model.update()

        if self.network_start_used:
            if include_courier_start:
                # Preserve f884's no-keyword helper call so tests and narrow
                # development hooks that replace the method remain compatible.
                self._append_network_fallback_starts()
            else:
                self._append_network_fallback_starts(
                    include_courier_start=False
                )
        else:
            install_fallback_start()
        self.network_start_runtime = time.perf_counter() - start_wall_time

    def _construct_network_start(
        self,
        time_limit: float,
        solver: SolverConfig,
    ) -> None:
        """f884-compatible wrapper around the explicit-budget implementation."""

        self._construct_network_start_with_budgets(
            time_limit,
            solver.fixed_integer_repair_seconds,
            time_limit if solver.use_local_branching else 0.0,
            time_limit,
            solver,
            include_courier_start=True,
        )

    def _append_network_fallback_starts(
        self,
        *,
        include_courier_start: bool = True,
    ) -> None:
        """Keep the complete start and append structural and optional courier hints.

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
        self.model.NumStart = partial_index + 1 + int(include_courier_start)
        self.model.update()

        self.model.Params.StartNumber = partial_index
        self.model.setAttr(
            GRB.Attr.Start,
            model_variables,
            [GRB.UNDEFINED] * len(model_variables),
        )
        for variable, value in structural_values:
            variable.Start = value

        if include_courier_start:
            self.model.Params.StartNumber = courier_index
            self._set_courier_only_start(reset_start_pool=False)
        self.model.Params.StartNumber = 0
        self.model.update()

        self.network_partial_start_installed = True
        self.network_partial_start_index = partial_index
        self.network_partial_start_variable_count = len(structural_values)

    def _clear_mip_starts(self) -> None:
        """Remove every user start, including stale per-variable Start values."""

        # A freshly built formal A--C model has no start pool.  Querying
        # NumStart is constant time; avoid an otherwise million-variable scan
        # and avoid charging the native baseline for work it does not need.
        if int(self.model.NumStart) == 0:
            return
        self.model.update()
        for variable in self.model.getVars():
            variable.Start = GRB.UNDEFINED
        self.model.update()
        # Assigning the Start attribute itself creates a start slot in Gurobi,
        # even when every value is undefined.  Drop that slot last so the
        # native profile is observably and semantically start-free.
        self.model.NumStart = 0
        self.model.update()

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
