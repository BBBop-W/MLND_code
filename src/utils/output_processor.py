"""Solution reporting for the OD-aware mixed-delivery model."""

from __future__ import annotations

import csv
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from matplotlib.collections import LineCollection

from src.data_processor.instance import DIRECT_ROUTE, Instance

if TYPE_CHECKING:
    from src.model.gurobi_model import MLNDModel


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


class ResultProcessor:
    """Extract, print, and persist one incumbent from :class:`MLNDModel`."""

    TOL = 1e-7

    def __init__(self, model: "MLNDModel", instance: Instance):
        self.wrapper = model
        self.model = model.model
        self.inst = instance
        self.mode = model.mode
        source = str(instance.metadata.get("source", "data"))
        exposure = instance.metadata.get("demand_exposure_hours")
        self.data_granularity = (
            f"{source}; hourly OD rates averaged over {exposure:g} exposure-hours"
            if isinstance(exposure, (int, float))
            else f"{source}; hourly OD rates"
        )

        self.sol_count = int(self.model.SolCount)
        if self.sol_count == 0:
            raise ValueError("ResultProcessor requires a model with an incumbent solution")

        self.status = int(self.model.Status)
        self.status_name = _STATUS_NAMES.get(self.status, f"STATUS_{self.status}")
        self.objective = float(self.model.ObjVal)
        self.bound = self._finite_or_none(self.model.ObjBound)
        self.gap = self._finite_or_none(self.model.MIPGap)
        self.main_runtime = float(getattr(model, "main_solve_runtime", self.model.Runtime))
        self.network_start_runtime = float(getattr(model, "network_start_runtime", 0.0))
        self.runtime = float(getattr(model, "total_runtime", self.model.Runtime))
        self.requested_mip_gap = float(self.model.Params.MIPGap)

        self.x = self._tupledict_values("x")
        self.z = self._tupledict_values("z")
        self.y = self._tupledict_values("y")
        self.delta = self._tupledict_values("delta")
        self.alpha = self._tupledict_values("alpha")
        self.h = self._tupledict_values("h")
        self.w = self._tupledict_values("w")
        self.psi = self._tupledict_values("psi")
        self.varphi = self._tupledict_values("varphi")
        self.lambd = self._tupledict_values("lambda")
        self.xi = self._tupledict_values("xi")
        self.a_lk = self._tupledict_values("a_lk")
        self.a_hm = self._tupledict_values("a_hm")
        self.q_lk = self._tupledict_values("q_lk")
        self.q_hm = self._tupledict_values("q_hm")
        self.gamma = self._clean(float(self.wrapper.vars["gamma"].X))

        self.costs = {
            name: self._clean(float(expression.getValue()))
            for name, expression in self.wrapper.cost_expressions.items()
        }
        self.cost_total = self._clean(sum(self.costs.values()))

        self.rho_by_idle, self.rho_level_by_idle = self._extract_rho_levels()
        self.capacity_by_locker = self._extract_capacities()
        self.od_rows = self._build_od_rows()
        self.corridor_rows = self._build_corridor_rows()
        self.locker_rows = self._build_locker_rows()
        self.facility_rows = self._build_facility_rows()
        self.summary = self._build_summary()

    @staticmethod
    def _finite_or_none(value: Any) -> float | None:
        number = float(value)
        return number if math.isfinite(number) else None

    @classmethod
    def _clean(cls, value: float) -> float:
        return 0.0 if abs(value) <= cls.TOL else float(value)

    @staticmethod
    def _ratio(numerator: float, denominator: float) -> float | None:
        return float(numerator / denominator) if denominator > ResultProcessor.TOL else None

    def _tupledict_values(self, name: str) -> dict[Any, float]:
        return {key: self._clean(float(var.X)) for key, var in self.wrapper.vars[name].items()}

    def _extract_rho_levels(self) -> tuple[dict[str, float], dict[str, int]]:
        levels = self.inst.config.availability_levels
        rho_by_idle: dict[str, float] = {}
        level_by_idle: dict[str, int] = {}
        for idle in self.inst.idle_states:
            selected = [(r, self.delta.get((idle, r), 0.0)) for r in range(len(levels))]
            r, value = max(selected, key=lambda item: item[1])
            if value > 0.5:
                rho_by_idle[idle] = float(levels[r])
                level_by_idle[idle] = r + 1
        return rho_by_idle, level_by_idle

    def _extract_capacities(self) -> dict[str, dict[str, float | int | None]]:
        rates = self.inst.config.locker_service_rates
        result: dict[str, dict[str, float | int | None]] = {}
        for locker in self.inst.H:
            selected = [(c, self.z.get((locker, c), 0.0)) for c in range(len(rates))]
            c, value = max(selected, key=lambda item: item[1])
            result[locker] = {
                "level": c + 1 if value > 0.5 else None,
                "service_rate": float(rates[c]) if value > 0.5 else 0.0,
            }
        return result

    @staticmethod
    def _market_totals(values: dict[tuple[str, str, str, str], float]) -> dict[tuple[str, str], float]:
        totals: dict[tuple[str, str], float] = defaultdict(float)
        for (origin, _i, _j, destination), value in values.items():
            totals[origin, destination] += value
        return totals

    def _build_od_rows(self) -> list[dict[str, Any]]:
        lk_offered = self._market_totals(self.a_lk)
        hm_offered = self._market_totals(self.a_hm)
        lk_served = self._market_totals(self.q_lk)
        hm_served = self._market_totals(self.q_hm)
        displayed: dict[str, list[str]] = defaultdict(list)
        for (destination, locker), value in self.alpha.items():
            if value > 0.5:
                displayed[destination].append(locker)

        rows: list[dict[str, Any]] = []
        for origin, destination in self.inst.markets:
            market = (origin, destination)
            demand = self.inst.h_od[market]
            locker_offer = lk_offered.get(market, 0.0)
            home_offer = hm_offered.get(market, 0.0)
            locker_service = lk_served.get(market, 0.0)
            home_service = hm_served.get(market, 0.0)
            served = locker_service + home_service
            fallback = max(0.0, demand - served)
            direct_unit_cost = self.inst.rider_cost(origin, destination)
            rows.append(
                {
                    "origin": origin,
                    "destination": destination,
                    "demand_per_hour": demand,
                    "order_count": self.inst.order_count.get(market, ""),
                    "displayed_locker": ";".join(sorted(displayed.get(destination, ()))),
                    "locker_offered_per_hour": locker_offer,
                    "home_offered_per_hour": home_offer,
                    "total_offered_per_hour": locker_offer + home_offer,
                    "locker_served_per_hour": locker_service,
                    "home_served_per_hour": home_service,
                    "drone_served_per_hour": served,
                    "offered_not_realized_per_hour": max(0.0, locker_offer + home_offer - served),
                    "rider_fallback_per_hour": fallback,
                    "drone_service_rate": self._ratio(served, demand) or 0.0,
                    "locker_realization_rate": self._ratio(locker_service, locker_offer),
                    "home_realization_rate": self._ratio(home_service, home_offer),
                    "direct_rider_unit_cost": direct_unit_cost,
                    "all_rider_baseline_cost_per_hour": direct_unit_cost * demand,
                    "fallback_rider_cost_per_hour": direct_unit_cost * fallback,
                    "avoided_direct_rider_cost_per_hour": direct_unit_cost * served,
                }
            )
        return rows

    @staticmethod
    def _corridor_totals(
        values: dict[tuple[str, str, str, str], float],
    ) -> dict[tuple[str, str], float]:
        totals: dict[tuple[str, str], float] = defaultdict(float)
        for (_o, i, j, _d), value in values.items():
            totals[i, j] += value
        return totals

    def _build_corridor_rows(self) -> list[dict[str, Any]]:
        lk_offered = self._corridor_totals(self.a_lk)
        hm_offered = self._corridor_totals(self.a_hm)
        lk_served = self._corridor_totals(self.q_lk)
        hm_served = self._corridor_totals(self.q_hm)
        direct: dict[tuple[str, str], float] = defaultdict(float)
        swap: dict[tuple[str, str], float] = defaultdict(float)
        flight_km: dict[tuple[str, str], float] = defaultdict(float)
        for (u, v, option), flow in self.varphi.items():
            if (u, v) not in self.inst.A_HH:
                continue
            bucket = direct if option == DIRECT_ROUTE else swap
            bucket[u, v] += flow
            flight_km[u, v] += self.inst.route_distance[u, v, option] * flow

        rows: list[dict[str, Any]] = []
        for i, j in self.wrapper.corridors:
            offered = lk_offered[i, j] + hm_offered[i, j]
            served = lk_served[i, j] + hm_served[i, j]
            rows.append(
                {
                    "origin_locker": i,
                    "destination_locker": j,
                    "connection_open": int(self.y.get((i, j), 0.0) > 0.5),
                    "model_offered_h_per_hour": self.h.get((i, j), 0.0),
                    "locker_offered_per_hour": lk_offered[i, j],
                    "home_offered_per_hour": hm_offered[i, j],
                    "total_offered_per_hour": offered,
                    "locker_served_per_hour": lk_served[i, j],
                    "home_served_per_hour": hm_served[i, j],
                    "total_served_per_hour": served,
                    "loaded_psi_per_hour": self.psi.get((i, j), 0.0),
                    "direct_flow_per_hour": direct[i, j],
                    "swap_flow_per_hour": swap[i, j],
                    "swap_share": self._ratio(swap[i, j], direct[i, j] + swap[i, j]),
                    "flight_km_per_hour": flight_km[i, j],
                }
            )
        return rows

    def _build_locker_rows(self) -> list[dict[str, Any]]:
        displayed_count = Counter()
        for (_destination, locker), value in self.alpha.items():
            if value > 0.5:
                displayed_count[locker] += 1

        rows: list[dict[str, Any]] = []
        for locker in self.inst.H:
            capacity = self.capacity_by_locker[locker]
            mu = float(capacity["service_rate"])
            workload = self.lambd.get(locker, 0.0)
            population = self.xi.get(locker, 0.0)
            headroom = mu - workload
            queue_population = workload / headroom if headroom > self.TOL else None
            state = self.inst.h_to_s[locker]
            chain_flow = self.psi.get((state, locker), 0.0)
            state_population = self.xi.get(state, 0.0)
            rows.append(
                {
                    "locker": locker,
                    "open": int(self.x.get(locker, 0.0) > 0.5),
                    "x_km": self.inst.coords[locker][0],
                    "y_km": self.inst.coords[locker][1],
                    "capacity_level": capacity["level"],
                    "service_rate_per_hour": mu,
                    "workload_lambda_per_hour": workload,
                    "service_headroom_per_hour": headroom,
                    "utilization": self._ratio(workload, mu),
                    "queue_population_formula": queue_population,
                    "mean_handling_time_formula_minutes": (
                        60.0 * queue_population / workload
                        if workload > self.TOL and queue_population is not None
                        else 0.0
                    ),
                    "conic_epigraph_xi": population,
                    "displayed_destination_count": displayed_count[locker],
                    "virtual_state": state,
                    "virtual_state_rho": self.rho_by_idle.get(state),
                    "virtual_state_rho_level": self.rho_level_by_idle.get(state),
                    "chained_source_flow_per_hour": chain_flow,
                    "virtual_state_expected_drones": state_population,
                    "virtual_state_wait_minutes": 60.0 * state_population / chain_flow if chain_flow > self.TOL else 0.0,
                }
            )
        return rows

    def _build_facility_rows(self) -> list[dict[str, Any]]:
        swap_throughput = Counter()
        for (_u, _v, option), flow in self.varphi.items():
            if option != DIRECT_ROUTE:
                swap_throughput[option] += flow

        rows: list[dict[str, Any]] = []
        for node in self.inst.L + self.inst.H + self.inst.E:
            if self.x.get(node, 0.0) <= 0.5:
                continue
            if node in self.inst.L:
                kind = "base"
                idle_state = node
            elif node in self.inst.H:
                kind = "locker"
                idle_state = self.inst.h_to_s[node]
            else:
                kind = "swap"
                idle_state = None
            capacity = self.capacity_by_locker.get(node, {})
            incident_flow = sum(
                flow for (u, v), flow in self.psi.items() if (u == node or v == node) and (u, v) in self.inst.A_F
            )
            rows.append(
                {
                    "facility": node,
                    "type": kind,
                    "x_km": self.inst.coords[node][0],
                    "y_km": self.inst.coords[node][1],
                    "fixed_cost_per_hour": self.inst.facility_cost(node),
                    "idle_state": idle_state or "",
                    "rho_level": self.rho_level_by_idle.get(idle_state) if idle_state else None,
                    "rho": self.rho_by_idle.get(idle_state) if idle_state else None,
                    "capacity_level": capacity.get("level"),
                    "locker_service_rate_per_hour": capacity.get("service_rate"),
                    "locker_workload_per_hour": self.lambd.get(node) if node in self.inst.H else None,
                    "incident_physical_flow_per_hour": incident_flow,
                    "swap_throughput_per_hour": swap_throughput[node] if node in self.inst.E else None,
                    "active_incident_connections": sum(
                        1 for (u, v), value in self.y.items() if value > 0.5 and (u == node or v == node)
                    ),
                }
            )
        return rows

    def _route_metrics(self) -> dict[str, Any]:
        metrics = defaultdict(float)
        swap_by_station = Counter()
        for (u, v, option), flow in self.varphi.items():
            load = "loaded" if (u, v) in self.inst.A_HH else "empty"
            route = "direct" if option == DIRECT_ROUTE else "swap"
            distance = self.inst.route_distance[u, v, option]
            travel_time = self.inst.route_time[u, v, option]
            metrics[f"{load}_{route}_flow"] += flow
            metrics[f"{load}_{route}_km"] += flow * distance
            metrics[f"{load}_{route}_transit_drones"] += flow * travel_time
            if option != DIRECT_ROUTE and flow > self.TOL:
                swap_by_station[option] += flow

        loaded = metrics["loaded_direct_flow"] + metrics["loaded_swap_flow"]
        empty = metrics["empty_direct_flow"] + metrics["empty_swap_flow"]
        direct = metrics["loaded_direct_flow"] + metrics["empty_direct_flow"]
        swap = metrics["loaded_swap_flow"] + metrics["empty_swap_flow"]
        total_km = sum(value for key, value in metrics.items() if key.endswith("_km"))
        transit = sum(value for key, value in metrics.items() if key.endswith("_transit_drones"))
        return {
            "loaded_flow_per_hour": loaded,
            "empty_flow_per_hour": empty,
            "empty_dispatch_per_hour": sum(self.psi.get(arc, 0.0) for arc in self.inst.A_LH),
            "empty_recovery_per_hour": sum(self.psi.get(arc, 0.0) for arc in self.inst.A_HL),
            "loaded_direct_flow_per_hour": metrics["loaded_direct_flow"],
            "loaded_swap_flow_per_hour": metrics["loaded_swap_flow"],
            "loaded_swap_share": self._ratio(metrics["loaded_swap_flow"], loaded),
            "all_direct_flow_per_hour": direct,
            "all_swap_flow_per_hour": swap,
            "all_swap_share": self._ratio(swap, direct + swap),
            "loaded_flight_km_per_hour": metrics["loaded_direct_km"] + metrics["loaded_swap_km"],
            "empty_flight_km_per_hour": metrics["empty_direct_km"] + metrics["empty_swap_km"],
            "total_flight_km_per_hour": total_km,
            "transit_drones": transit,
            "swap_flow_by_station_per_hour": dict(sorted(swap_by_station.items())),
        }

    def _build_summary(self) -> dict[str, Any]:
        demand = sum(row["demand_per_hour"] for row in self.od_rows)
        locker_offered = sum(row["locker_offered_per_hour"] for row in self.od_rows)
        home_offered = sum(row["home_offered_per_hour"] for row in self.od_rows)
        locker_served = sum(row["locker_served_per_hour"] for row in self.od_rows)
        home_served = sum(row["home_served_per_hour"] for row in self.od_rows)
        offered = locker_offered + home_offered
        served = locker_served + home_served
        fallback = max(0.0, demand - served)

        rider_baseline = sum(row["all_rider_baseline_cost_per_hour"] for row in self.od_rows)
        savings = self._clean(rider_baseline - self.cost_total)
        source_from_bases = sum(self.psi.get(arc, 0.0) for arc in self.inst.A_LH)
        source_from_virtual = sum(self.psi.get(arc, 0.0) for arc in self.inst.A_SH)
        state_entries = sum(self.psi.get(arc, 0.0) for arc in self.inst.A_HS)
        route = self._route_metrics()

        base_idle_population = sum(self.xi.get(base, 0.0) for base in self.inst.L)
        virtual_idle_population = sum(self.xi.get(state, 0.0) for state in self.inst.S)
        locker_population = sum(self.xi.get(locker, 0.0) for locker in self.inst.H)
        fleet_accounted = base_idle_population + virtual_idle_population + locker_population + route["transit_drones"]
        rho_counts = Counter(f"{rho:.2f}" for rho in self.rho_by_idle.values())
        base_rho_counts = Counter(f"{self.rho_by_idle[b]:.2f}" for b in self.inst.L if b in self.rho_by_idle)
        virtual_rho_counts = Counter(f"{self.rho_by_idle[s]:.2f}" for s in self.inst.S if s in self.rho_by_idle)

        open_counts = {
            "bases": sum(self.x.get(node, 0.0) > 0.5 for node in self.inst.L),
            "lockers": sum(self.x.get(node, 0.0) > 0.5 for node in self.inst.H),
            "swap_stations": sum(self.x.get(node, 0.0) > 0.5 for node in self.inst.E),
        }
        capacity_counts = Counter(
            str(row["capacity_level"])
            for row in self.locker_rows
            if row["open"] and row["capacity_level"] is not None
        )
        total_lambda = sum(row["workload_lambda_per_hour"] for row in self.locker_rows)
        total_locker_xi = sum(row["conic_epigraph_xi"] for row in self.locker_rows)
        total_queue_population = sum(
            row["queue_population_formula"] or 0.0 for row in self.locker_rows
        )
        max_util = max((row["utilization"] or 0.0 for row in self.locker_rows), default=0.0)
        max_wait = max(
            (row["mean_handling_time_formula_minutes"] for row in self.locker_rows),
            default=0.0,
        )
        cfg = self.inst.config
        screening_is_exact = (
            cfg.choice.max_walk_km is None
            and cfg.origin_locker_choices is None
            and cfg.home_corridor_choices is None
        )
        active_virtual_state_count = sum(
            state in self.rho_by_idle for state in self.inst.S
        )
        selected_base_rho = {
            base: self.rho_by_idle[base]
            for base in self.inst.L
            if base in self.rho_by_idle
        }
        selected_virtual_rho_by_locker = {
            self.inst.s_to_h[state]: self.rho_by_idle[state]
            for state in self.inst.S
            if state in self.rho_by_idle
        }
        network_is_nontrivial = served > self.TOL
        solver_config = getattr(self.wrapper, "solver_config", None)

        return {
            "schema_version": 1,
            "mode": self.mode,
            "status": self.status,
            "status_name": self.status_name,
            "objective": self.objective,
            "objective_bound": self.bound,
            "runtime_seconds": self.runtime,
            "mip_gap": self.gap,
            "requested_mip_gap": self.requested_mip_gap,
            "courier_only_baseline": rider_baseline,
            "cost_saving": savings,
            "cost_saving_percent": (
                100.0 * savings / rider_baseline if rider_baseline > self.TOL else None
            ),
            "drone_served_rate": self._ratio(served, demand),
            "fleet_size": self.gamma,
            "screening_is_exact": screening_is_exact,
            "active_virtual_state_count": active_virtual_state_count,
            "network_is_nontrivial": network_is_nontrivial,
            "courier_only_incumbent": not network_is_nontrivial,
            "open_bases": open_counts["bases"],
            "open_lockers": open_counts["lockers"],
            "open_swap_stations": open_counts["swap_stations"],
            "selected_rho_state_count": len(self.rho_by_idle),
            "selected_rho_by_idle_state": dict(sorted(self.rho_by_idle.items())),
            "solver": {
                "status": self.status,
                "status_name": self.status_name,
                "solution_count": self.sol_count,
                "objective_cost_per_hour": self.objective,
                "objective_bound": self.bound,
                "mip_gap": self.gap,
                "requested_mip_gap": self.requested_mip_gap,
                "runtime_seconds": self.runtime,
                "main_solve_runtime_seconds": self.main_runtime,
                "network_start_runtime_seconds": self.network_start_runtime,
                "network_start_found": bool(getattr(self.wrapper, "network_start_found", False)),
                "network_start_used": bool(getattr(self.wrapper, "network_start_used", False)),
                "network_start_key": list(self.wrapper.network_start_key)
                if getattr(self.wrapper, "network_start_key", None)
                else None,
                "network_start_objective": getattr(
                    self.wrapper, "network_start_objective", None
                ),
                "algorithm": {
                    "name": "demand_aware_perspective_branch_and_cut",
                    "root_relaxation_method": getattr(
                        solver_config, "root_relaxation_method", None
                    ),
                    "static_pool_enabled": bool(
                        getattr(solver_config, "use_valid_inequalities", False)
                    ),
                    "user_cuts_enabled": bool(
                        getattr(solver_config, "use_user_cuts", False)
                    ),
                    "maximum_root_cut_rounds": getattr(
                        solver_config, "max_root_cut_rounds", None
                    ),
                    "maximum_cuts_per_round": getattr(
                        solver_config, "max_cuts_per_round", None
                    ),
                    "local_branching_enabled": bool(
                        getattr(solver_config, "use_local_branching", False)
                    ),
                    "local_branching_radius": getattr(
                        solver_config, "local_branching_radius", None
                    ),
                    "static_valid_inequality_counts": dict(
                        getattr(self.wrapper, "static_cut_counts", {})
                    ),
                    "root_user_cut_counts": dict(
                        getattr(self.wrapper, "user_cut_counts", {})
                    ),
                    "root_cut_rounds": int(
                        getattr(self.wrapper, "root_cut_rounds", 0)
                    ),
                    "local_branching_runtime_seconds": float(
                        getattr(self.wrapper, "local_branching_runtime", 0.0)
                    ),
                    "local_branching_improved": bool(
                        getattr(self.wrapper, "local_branching_improved", False)
                    ),
                    "local_branching_objective": getattr(
                        self.wrapper, "local_branching_objective", None
                    ),
                    "local_branching_status": getattr(
                        self.wrapper, "local_branching_status", None
                    ),
                    "local_branching_solution_count": int(
                        getattr(self.wrapper, "local_branching_solution_count", 0)
                    ),
                    "local_branching_hamming_distance": getattr(
                        self.wrapper, "local_branching_hamming_distance", None
                    ),
                    "installed_mip_start_count": 2
                    if getattr(self.wrapper, "network_start_used", False)
                    else 1,
                    "all_courier_start_installed": True,
                },
            },
            "instance": {
                "data_source": self.inst.metadata.get("source"),
                "data_granularity": self.data_granularity,
                "markets": len(self.inst.markets),
                "od_nodes": len(self.inst.N),
                "endogenous_corridors": len(self.wrapper.corridors),
                "locker_assignment_keys": len(self.wrapper.locker_keys),
                "home_assignment_keys": len(self.wrapper.home_keys),
                "candidate_bases": len(self.inst.L),
                "candidate_lockers": len(self.inst.H),
                "candidate_swap_stations": len(self.inst.E),
                "candidate_physical_arcs": len(self.inst.A_F),
                "candidate_activation_arcs": len(self.inst.A_Y),
                "route_options": len(self.wrapper.varphi_keys),
                "direct_route_options": sum(
                    option == DIRECT_ROUTE
                    for _u, _v, option in self.wrapper.varphi_keys
                ),
                "swap_route_options": sum(
                    option != DIRECT_ROUTE
                    for _u, _v, option in self.wrapper.varphi_keys
                ),
                "maximum_physical_arc_flow_bound": max(
                    self.inst.M_psi.values(), default=0.0
                ),
                "maximum_offered_corridor_bound": max(
                    self.wrapper.M_w.values(), default=0.0
                ),
                "screening_is_exact": screening_is_exact,
            },
            "model_size": {
                "variables": int(self.model.NumVars),
                "binary_variables": int(self.model.NumBinVars),
                "integer_variables_including_binary": int(self.model.NumIntVars),
                "general_integer_variables": max(
                    int(self.model.NumIntVars) - int(self.model.NumBinVars), 0
                ),
                "linear_constraints": int(self.model.NumConstrs),
                "quadratic_constraints": int(self.model.NumQConstrs),
            },
            "costs_cny_per_hour": {
                **self.costs,
                "optimized_total": self.cost_total,
                "objective_reconciliation_error": self._clean(self.objective - self.cost_total),
            },
            "all_rider_baseline": {
                "cost_per_hour": rider_baseline,
                "optimized_cost_per_hour": self.cost_total,
                "savings_per_hour": savings,
                "savings_fraction": self._ratio(savings, rider_baseline),
                "avoided_direct_rider_cost_per_hour": self._clean(
                    rider_baseline - self.costs["fallback_home"]
                ),
            },
            "demand_funnel_orders_per_hour": {
                "total_demand": demand,
                "locker_offered": locker_offered,
                "home_offered": home_offered,
                "total_drone_offered": offered,
                "not_offered": max(0.0, demand - offered),
                "locker_served": locker_served,
                "home_served": home_served,
                "total_drone_served": served,
                "offered_not_realized": max(0.0, offered - served),
                "direct_rider_fallback": fallback,
                "drone_service_fraction_of_demand": self._ratio(served, demand),
                "realization_fraction_of_offered": self._ratio(served, offered),
            },
            "facilities": {
                "open": open_counts,
                "candidates": {
                    "bases": len(self.inst.L),
                    "lockers": len(self.inst.H),
                    "swap_stations": len(self.inst.E),
                },
                "locker_capacity_level_counts": dict(sorted(capacity_counts.items())),
                "active_connections": sum(value > 0.5 for value in self.y.values()),
            },
            "discrete_availability": {
                "allowed_rho_levels": list(self.inst.config.availability_levels),
                "selected_state_count_by_rho": dict(sorted(rho_counts.items())),
                "selected_base_count_by_rho": dict(sorted(base_rho_counts.items())),
                "selected_virtual_count_by_rho": dict(sorted(virtual_rho_counts.items())),
                "selected_rho_by_idle_state": dict(sorted(self.rho_by_idle.items())),
                "selected_base_rho": selected_base_rho,
                "selected_virtual_rho_by_locker": selected_virtual_rho_by_locker,
                "active_virtual_state_count": active_virtual_state_count,
                "offered_flow_weighted_rho": self._ratio(served, sum(self.w.values())),
            },
            "fleet": {
                "gamma_drones": self.gamma,
                "base_idle_drones": base_idle_population,
                "virtual_idle_drones": virtual_idle_population,
                "locker_conic_epigraph_xi": locker_population,
                "transit_drones": route["transit_drones"],
                "accounted_drones": fleet_accounted,
                "fleet_slack": self._clean(self.gamma - fleet_accounted),
                "flying_fraction_of_fleet": self._ratio(route["transit_drones"], self.gamma),
            },
            "flight_operations": route,
            "chaining": {
                "base_origin_orders_per_hour": source_from_bases,
                "virtual_state_origin_orders_per_hour": source_from_virtual,
                "virtual_state_entry_flow_per_hour": state_entries,
                "chaining_order_fraction": self._ratio(source_from_virtual, served),
                "source_flow_reconciliation_error": self._clean(
                    source_from_bases + source_from_virtual - served
                ),
                "virtual_flow_reconciliation_error": self._clean(
                    state_entries - source_from_virtual
                ),
                "mean_response_minutes_from_base": 60.0 * base_idle_population / source_from_bases if source_from_bases > self.TOL else None,
                "mean_response_minutes_from_virtual_state": 60.0 * virtual_idle_population / source_from_virtual if source_from_virtual > self.TOL else None,
            },
            "locker_queue": {
                "total_workload_per_hour": total_lambda,
                "queue_population_formula_total": total_queue_population,
                "conic_epigraph_xi_total": total_locker_xi,
                "workload_weighted_mean_handling_formula_minutes": (
                    60.0 * total_queue_population / total_lambda
                    if total_lambda > self.TOL
                    else None
                ),
                "maximum_utilization": max_util,
                "maximum_mean_handling_formula_minutes": max_wait,
                "handling_time_limit_minutes": 60.0 * self.inst.config.max_locker_time_hours,
            },
            "artifacts": {
                "summary": "summary.json",
                "open_facilities": "open_facilities.csv",
                "corridor_flows": "corridor_flows.csv",
                "od_service": "od_service.csv",
                "locker_metrics": "locker_metrics.csv",
                "figures": [
                    "network_map",
                    "od_service_map",
                    "cost_breakdown",
                    "demand_funnel",
                    "locker_utilization",
                ],
            },
        }

    @staticmethod
    def _fmt(value: float | None, digits: int = 2) -> str:
        return "N/A" if value is None else f"{value:,.{digits}f}"

    @staticmethod
    def _pct(value: float | None) -> str:
        return "N/A" if value is None else f"{100.0 * value:.2f}%"

    def print_summary(self) -> None:
        """Print the decision and operating metrics needed to audit a run."""

        solver = self.summary["solver"]
        baseline = self.summary["all_rider_baseline"]
        funnel = self.summary["demand_funnel_orders_per_hour"]
        facilities = self.summary["facilities"]
        availability = self.summary["discrete_availability"]
        fleet = self.summary["fleet"]
        flight = self.summary["flight_operations"]
        chaining = self.summary["chaining"]
        queue = self.summary["locker_queue"]
        instance = self.summary["instance"]
        model_size = self.summary["model_size"]

        print("\n" + "=" * 72)
        status_text = solver["status_name"]
        if solver["status_name"] == "OPTIMAL":
            status_text += (
                f" within requested {100.0 * solver['requested_mip_gap']:.2f}% MIP tolerance"
            )
        print(f"MLND {self.mode} SOLUTION  |  {status_text}  |  {solver['runtime_seconds']:.2f}s")
        print("=" * 72)
        print(
            f"Input: {instance['data_source']}  |  OD={instance['markets']:,}, "
            f"OD nodes={instance['od_nodes']:,}, endogenous H-H={instance['endogenous_corridors']:,}  |  "
            f"candidates L/H/E={instance['candidate_bases']}/{instance['candidate_lockers']}/"
            f"{instance['candidate_swap_stations']}"
        )
        print(
            f"Model size: {model_size['variables']:,} vars "
            f"({model_size['binary_variables']:,} binary, "
            f"{model_size['general_integer_variables']:,} general integer), "
            f"{model_size['linear_constraints']:,} linear + "
            f"{model_size['quadratic_constraints']:,} quadratic constraints"
        )
        algorithm = solver["algorithm"]
        print(
            "Algorithm: demand-aware perspective branch-and-cut  |  "
            f"static rows={sum(algorithm['static_valid_inequality_counts'].values()):,}, "
            f"root user cuts={sum(algorithm['root_user_cut_counts'].values()):,} "
            f"in {algorithm['root_cut_rounds']} rounds, "
            f"local branching={algorithm['local_branching_runtime_seconds']:.2f}s"
        )
        print(
            f"Objective: {self._fmt(solver['objective_cost_per_hour'])} CNY/h  "
            f"bound={self._fmt(solver['objective_bound'])}  gap={self._pct(solver['mip_gap'])}"
        )
        print(
            f"All-rider baseline: {self._fmt(baseline['cost_per_hour'])} CNY/h  |  "
            f"saving: {self._fmt(baseline['savings_per_hour'])} CNY/h "
            f"({self._pct(baseline['savings_fraction'])})"
        )
        print("Cost breakdown (CNY/h):")
        for name, value in self.costs.items():
            print(f"  {name:<20} {value:>12,.2f}")

        print("Demand funnel (orders/h):")
        print(
            f"  demand={funnel['total_demand']:,.2f}  offered={funnel['total_drone_offered']:,.2f}  "
            f"served={funnel['total_drone_served']:,.2f}  fallback={funnel['direct_rider_fallback']:,.2f}"
        )
        print(
            f"  served locker/home={funnel['locker_served']:,.2f}/{funnel['home_served']:,.2f}  "
            f"demand service={self._pct(funnel['drone_service_fraction_of_demand'])}  "
            f"offered realization={self._pct(funnel['realization_fraction_of_offered'])}"
        )

        opened = facilities["open"]
        candidates = facilities["candidates"]
        print(
            "Open facilities: "
            f"L {opened['bases']}/{candidates['bases']}, "
            f"H {opened['lockers']}/{candidates['lockers']}, "
            f"E {opened['swap_stations']}/{candidates['swap_stations']}  |  "
            f"active connections={facilities['active_connections']}"
        )
        print(
            f"Locker capacity levels: {facilities['locker_capacity_level_counts'] or '{}'}  |  "
            f"selected discrete rho: {availability['selected_state_count_by_rho'] or '{}'}"
        )
        if self.facility_rows:
            print("Opened-site decisions (rho applies to base idle or locker virtual state):")
            for row in self.facility_rows:
                rho = row["rho"]
                rho_text = "N/A" if rho is None else f"{rho:.2f} (level {row['rho_level']})"
                detail = f"rho={rho_text}"
                if row["type"] == "locker":
                    detail += f", capacity={row['capacity_level']}"
                print(f"  {row['facility']} [{row['type']}]  {detail}")
        else:
            print("Opened-site decisions: none; selected rho: none")
        if self.summary["courier_only_incumbent"]:
            print(
                "WARNING: incumbent uses no realized drone service; cost equals the all-rider "
                "counterfactual when no positive-cost facilities are opened."
            )
        print(
            f"Exact candidate screening: {self.summary['instance']['screening_is_exact']}  |  "
            f"active virtual states={availability['active_virtual_state_count']}"
        )
        print(
            f"Fleet gamma={fleet['gamma_drones']:.0f}: idle L={fleet['base_idle_drones']:.2f}, "
            f"idle S={fleet['virtual_idle_drones']:.2f}, "
            f"locker conic-xi={fleet['locker_conic_epigraph_xi']:.2f}, "
            f"flying={fleet['transit_drones']:.2f} ({self._pct(fleet['flying_fraction_of_fleet'])})"
        )
        print(
            f"Physical flow: loaded={flight['loaded_flow_per_hour']:.2f}, "
            f"empty={flight['empty_flow_per_hour']:.2f} orders/h-equivalent; "
            f"loaded direct/swap={flight['loaded_direct_flow_per_hour']:.2f}/"
            f"{flight['loaded_swap_flow_per_hour']:.2f} "
            f"(swap {self._pct(flight['loaded_swap_share'])})"
        )
        print(
            f"Chaining: base-origin={chaining['base_origin_orders_per_hour']:.2f}, "
            f"virtual-origin={chaining['virtual_state_origin_orders_per_hour']:.2f} orders/h "
            f"({self._pct(chaining['chaining_order_fraction'])})"
        )
        print(
            f"Locker queue: workload={queue['total_workload_per_hour']:.2f}/h, "
            f"weighted handling={self._fmt(queue['workload_weighted_mean_handling_formula_minutes'])} min, "
            f"max handling={queue['maximum_mean_handling_formula_minutes']:.2f} min, "
            f"max utilization={self._pct(queue['maximum_utilization'])}"
        )
        print(
            f"  M/M/1 queue population={queue['queue_population_formula_total']:.3f}; "
            f"conic epigraph xi={queue['conic_epigraph_xi_total']:.3f} (not treated as exact queue length)"
        )
        print("=" * 72 + "\n")

    @staticmethod
    def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: Iterable[str]) -> None:
        fields = list(fieldnames)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    @staticmethod
    def _save_figure(fig: plt.Figure, stem: Path) -> None:
        fig.savefig(stem.with_suffix(".png"), dpi=220, bbox_inches="tight")
        fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
        plt.close(fig)

    def _plot_network_map(self, output_dir: Path) -> None:
        fig, ax = plt.subplots(figsize=(11, 9))

        endpoint_demand: dict[str, float] = defaultdict(float)
        for (origin, destination), rate in self.inst.h_od.items():
            endpoint_demand[origin] += rate
            endpoint_demand[destination] += rate
        demand_nodes = [node for node in sorted(endpoint_demand) if endpoint_demand[node] > self.TOL]
        max_endpoint_demand = max(
            (endpoint_demand[node] for node in demand_nodes),
            default=1.0,
        )
        if demand_nodes:
            ax.scatter(
                [self.inst.coords[node][0] for node in demand_nodes],
                [self.inst.coords[node][1] for node in demand_nodes],
                s=[
                    8.0 + 62.0 * math.sqrt(endpoint_demand[node] / max_endpoint_demand)
                    for node in demand_nodes
                ],
                color="#9ca3af",
                alpha=0.20,
                linewidths=0,
                label="OD endpoint demand (orders/h)",
                zorder=0,
            )

        station_styles = {
            "L": (self.inst.L, "s", "#d62728", "base"),
            "H": (self.inst.H, "o", "#1f77b4", "locker"),
            "E": (self.inst.E, "D", "#2ca02c", "swap"),
        }
        for code, (nodes, marker, _color, label) in station_styles.items():
            if not nodes:
                continue
            ax.scatter(
                [self.inst.coords[node][0] for node in nodes],
                [self.inst.coords[node][1] for node in nodes],
                marker=marker,
                s=42,
                facecolors="none",
                edgecolors="#8b8f94",
                linewidths=0.9,
                alpha=0.62,
                label=f"Candidate {label} {code}",
                zorder=1,
            )

        max_flow = max(self.varphi.values(), default=0.0)
        flow_labeled = False
        for (u, v, option), flow in self.varphi.items():
            if flow <= self.TOL:
                continue
            points = [self.inst.coords[u], self.inst.coords[v]]
            if option != DIRECT_ROUTE:
                points.insert(1, self.inst.coords[option])
            width = 0.7 + 4.3 * math.sqrt(flow / max_flow) if max_flow > self.TOL else 0.8
            ax.plot(
                [point[0] for point in points],
                [point[1] for point in points],
                color="#374151",
                linestyle="-",
                linewidth=width,
                alpha=0.64,
                solid_capstyle="round",
                label="Physical service flow (orders/h)" if not flow_labeled else None,
                zorder=2,
            )
            flow_labeled = True
        if not flow_labeled:
            ax.plot(
                [],
                [],
                color="#374151",
                linewidth=1.3,
                label="Physical service flow (none)",
            )

        label_offsets = ((3, 3), (3, -3), (-3, 3), (-3, -3))
        label_index = 0
        for code, (nodes, marker, color, label) in station_styles.items():
            opened = [node for node in nodes if self.x.get(node, 0.0) > 0.5]
            if opened:
                ax.scatter(
                    [self.inst.coords[node][0] for node in opened],
                    [self.inst.coords[node][1] for node in opened],
                    marker=marker,
                    s=100,
                    color=color,
                    edgecolors="white",
                    linewidths=0.8,
                    label=f"Open {label} {code}",
                    zorder=4,
                )
                for node in opened:
                    x_coord, y_coord = self.inst.coords[node]
                    if code == "L":
                        rho = self.rho_by_idle.get(node)
                    elif code == "H":
                        rho = self.rho_by_idle.get(self.inst.h_to_s[node])
                    else:
                        continue
                    if rho is None:
                        continue
                    rho_value = f"{rho:.2f}"[1:]
                    text = f"{node}  ρ={rho_value}" if code == "L" else f"{node}  ρv={rho_value}"
                    offset_x, offset_y = label_offsets[label_index % len(label_offsets)]
                    label_index += 1
                    ax.annotate(
                        text,
                        (x_coord, y_coord),
                        xytext=(offset_x, offset_y),
                        textcoords="offset points",
                        ha="left" if offset_x > 0 else "right",
                        va="bottom" if offset_y > 0 else "top",
                        fontsize=6.5,
                        color="#202124",
                        bbox={
                            "boxstyle": "round,pad=0.10",
                            "facecolor": "white",
                            "edgecolor": "none",
                            "alpha": 0.82,
                        },
                        zorder=5,
                    )

        funnel = self.summary["demand_funnel_orders_per_hour"]
        opened_counts = self.summary["facilities"]["open"]
        selected_rhos = sorted(set(self.rho_by_idle.values()))
        rho_text = ", ".join(f"{rho:.2f}" for rho in selected_rhos) or "none"
        gap_text = "N/A" if self.gap is None else f"{100.0 * self.gap:.2f}%"
        fig.suptitle(
            f"MLND {self.mode}: optimized station network",
            y=0.975,
            fontsize=15,
            fontweight="semibold",
            color="#202124",
        )
        fig.text(
            0.5,
            0.925,
            f"Open L/H/E = {opened_counts['bases']}/{opened_counts['lockers']}/"
            f"{opened_counts['swap_stations']}  •  served = "
            f"{funnel['total_drone_served']:,.2f} orders/h  •  gap = {gap_text}  •  "
            f"selected rho levels = {rho_text}",
            ha="center",
            fontsize=9.5,
            color="#4b5563",
        )
        ax.set_xlabel("x (km)")
        ax.set_ylabel("y (km)")
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(color="#9ca3af", linestyle="--", linewidth=0.55, alpha=0.24)
        ax.set_axisbelow(True)
        ax.legend(
            loc="upper left",
            fontsize=7.5,
            ncol=2,
            frameon=True,
            framealpha=0.92,
            facecolor="white",
            edgecolor="#d1d5db",
        )
        fig.text(
            0.5,
            0.018,
            f"Line width scales with physical flow in orders/hour; demand bubbles are hourly OD endpoint rates.  "
            f"{self.data_granularity}",
            ha="center",
            fontsize=8,
            color="#6b7280",
        )
        fig.tight_layout(rect=(0.0, 0.055, 1.0, 0.885))
        self._save_figure(fig, output_dir / "network_map")

    def _plot_od_service_map(self, output_dir: Path) -> None:
        fig, ax = plt.subplots(figsize=(11, 9))
        rows = [row for row in self.od_rows if row["demand_per_hour"] > self.TOL]
        segments = [
            [self.inst.coords[row["origin"]], self.inst.coords[row["destination"]]] for row in rows
        ]
        max_demand = max((row["demand_per_hour"] for row in rows), default=1.0)
        service = [row["drone_service_rate"] for row in rows]
        widths = [0.15 + 2.2 * math.sqrt(row["demand_per_hour"] / max_demand) for row in rows]
        norm = colors.Normalize(vmin=0.0, vmax=1.0)
        if segments:
            collection = LineCollection(
                segments,
                cmap="cividis",
                norm=norm,
                linewidths=widths,
                alpha=0.42,
                zorder=1,
            )
            collection.set_array(service)
            ax.add_collection(collection)
        od_nodes = sorted(set(self.inst.N))
        ax.scatter(
            [self.inst.coords[n][0] for n in od_nodes],
            [self.inst.coords[n][1] for n in od_nodes],
            s=8,
            color="#4b5563",
            alpha=0.45,
            label="OD grid nodes",
            zorder=2,
        )
        open_lockers = [node for node in self.inst.H if self.x.get(node, 0.0) > 0.5]
        if open_lockers:
            ax.scatter(
                [self.inst.coords[n][0] for n in open_lockers],
                [self.inst.coords[n][1] for n in open_lockers],
                s=70,
                marker="o",
                color="#2563eb",
                edgecolor="white",
                linewidth=0.7,
                label="Open lockers",
                zorder=3,
            )
        scalar = cm.ScalarMappable(norm=norm, cmap="cividis")
        scalar.set_array([])
        fig.colorbar(scalar, ax=ax, label="Drone-served fraction of OD demand")
        ax.set_title(
            f"MLND {self.mode}: OD demand and realized drone service (orders/h)\n"
            f"{self.data_granularity}"
        )
        ax.set_xlabel("x (km)")
        ax.set_ylabel("y (km)")
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(alpha=0.15)
        ax.legend(loc="best")
        fig.tight_layout()
        self._save_figure(fig, output_dir / "od_service_map")

    def _plot_cost_breakdown(self, output_dir: Path) -> None:
        fig, ax = plt.subplots(figsize=(11, 6))
        labels = [name.replace("_", "\n") for name in self.costs]
        values = list(self.costs.values())
        palette = [
            "#cbd5e1",
            "#94a3b8",
            "#64748b",
            "#dbeafe",
            "#bfdbfe",
            "#93c5fd",
            "#3b82f6",
            "#1d4ed8",
        ]
        bars = ax.bar(labels, values, color=palette[: len(values)])
        baseline = self.summary["all_rider_baseline"]["cost_per_hour"]
        ax.axhline(baseline, color="#c2410c", linestyle="--", linewidth=1.5, label="All-rider baseline")
        ax.bar_label(bars, labels=[f"{value:,.0f}" for value in values], padding=3, fontsize=8)
        ax.set_ylabel("CNY per hour")
        ax.set_title(
            f"Cost breakdown — total {self.cost_total:,.2f} CNY/h\n"
            f"{self.data_granularity}"
        )
        ax.grid(axis="y", alpha=0.2)
        ax.legend()
        fig.tight_layout()
        self._save_figure(fig, output_dir / "cost_breakdown")

    def _plot_demand_funnel(self, output_dir: Path) -> None:
        funnel = self.summary["demand_funnel_orders_per_hour"]
        total = funnel["total_demand"]
        values = [total, funnel["total_drone_offered"], funnel["total_drone_served"]]
        labels = ["All OD demand", "Offered to drone network", "Realized drone service"]
        fig, ax = plt.subplots(figsize=(9, 5.5))
        y_pos = list(range(len(labels)))
        ax.barh(y_pos, [total] * len(labels), color="#e5e7eb", height=0.68)
        bars = ax.barh(y_pos, values, color=["#64748b", "#60a5fa", "#1d4ed8"], height=0.68)
        ax.set_yticks(y_pos, labels)
        ax.invert_yaxis()
        ax.bar_label(bars, labels=[f"{value:,.2f}/h" for value in values], padding=5)
        ax.set_xlim(0.0, max(total * 1.18, 1.0))
        ax.set_xlabel("Orders per hour")
        ax.set_title(
            f"Demand funnel (orders/h): locker/home served "
            f"{funnel['locker_served']:,.2f} / {funnel['home_served']:,.2f}\n"
            f"{self.data_granularity}"
        )
        ax.grid(axis="x", alpha=0.2)
        fig.tight_layout()
        self._save_figure(fig, output_dir / "demand_funnel")

    def _plot_locker_utilization(self, output_dir: Path) -> None:
        rows = [row for row in self.locker_rows if row["open"]]
        fig, ax = plt.subplots(figsize=(max(9, 0.42 * len(rows) + 4), 6))
        fig.suptitle(
            "Open smart-locker utilization and M/M/1 handling time",
            y=0.97,
            fontweight="semibold",
        )
        fig.text(
            0.5,
            0.915,
            f"Rates in orders/hour; {self.data_granularity}",
            ha="center",
            color="#475569",
            fontsize=9,
        )
        if not rows:
            ax.text(0.5, 0.5, "No open smart lockers", ha="center", va="center", fontsize=15)
            ax.set_axis_off()
        else:
            names = [row["locker"] for row in rows]
            utilization = [row["utilization"] or 0.0 for row in rows]
            waiting = [row["mean_handling_time_formula_minutes"] for row in rows]
            x_pos = list(range(len(rows)))
            blue_scale = plt.get_cmap("Blues")
            bars = ax.bar(
                x_pos,
                utilization,
                color=[blue_scale(0.42 + 0.48 * min(value, 1.0)) for value in utilization],
            )
            ax.set_xticks(x_pos, names, rotation=55, ha="right")
            ax.set_ylabel("Locker workload / service rate")
            ax.set_ylim(0.0, max(1.02, max(utilization, default=0.0) * 1.15))
            ax.grid(axis="y", alpha=0.2)
            wait_ax = ax.twinx()
            wait_ax.plot(x_pos, waiting, color="#111827", marker="o", linewidth=1.2, label="Mean handling time")
            wait_ax.axhline(
                60.0 * self.inst.config.max_locker_time_hours,
                color="#c2410c",
                linestyle="--",
                label="Handling-time limit",
            )
            wait_ax.set_ylabel("Mean handling time (minutes)")
            wait_ax.set_ylim(bottom=0.0)
            wait_ax.legend(loc="upper right", fontsize=8)
            ax.bar_label(bars, labels=[f"{100 * value:.0f}%" for value in utilization], padding=2, fontsize=7)
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.88))
        self._save_figure(fig, output_dir / "locker_utilization")

    def write_all(self, output_dir: str) -> dict[str, Any]:
        """Write the complete result package and return its summary dictionary."""

        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        with (destination / "summary.json").open("w", encoding="utf-8") as handle:
            json.dump(self.summary, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)

        self._write_csv(
            destination / "open_facilities.csv",
            self.facility_rows,
            (
                "facility",
                "type",
                "x_km",
                "y_km",
                "fixed_cost_per_hour",
                "idle_state",
                "rho_level",
                "rho",
                "capacity_level",
                "locker_service_rate_per_hour",
                "locker_workload_per_hour",
                "incident_physical_flow_per_hour",
                "swap_throughput_per_hour",
                "active_incident_connections",
            ),
        )
        self._write_csv(destination / "corridor_flows.csv", self.corridor_rows, self.corridor_rows[0].keys() if self.corridor_rows else (
            "origin_locker", "destination_locker", "connection_open", "model_offered_h_per_hour",
            "locker_offered_per_hour", "home_offered_per_hour", "total_offered_per_hour",
            "locker_served_per_hour", "home_served_per_hour", "total_served_per_hour",
            "loaded_psi_per_hour", "direct_flow_per_hour", "swap_flow_per_hour", "swap_share", "flight_km_per_hour",
        ))
        self._write_csv(destination / "od_service.csv", self.od_rows, self.od_rows[0].keys() if self.od_rows else ())
        self._write_csv(destination / "locker_metrics.csv", self.locker_rows, self.locker_rows[0].keys() if self.locker_rows else ())

        self._plot_network_map(destination)
        self._plot_od_service_map(destination)
        self._plot_cost_breakdown(destination)
        self._plot_demand_funnel(destination)
        self._plot_locker_utilization(destination)
        return self.summary
