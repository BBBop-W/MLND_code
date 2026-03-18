"""Benders decomposition utilities for the MLND model.

This module is organized around the standard Benders objects:

1. The master problem over facility activation and service levels.
2. The primal subproblem under fixed master decisions.
3. The dual subproblem used to generate Benders cuts.

It also provides a fixed-master audit against the benchmark monolithic model.
"""

from dataclasses import dataclass, field
import math
import time
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import gurobipy as gp
from gurobipy import GRB

from .. import config
from ..data_processor.instance import Instance


XHat = Mapping[str, int]
UHat = Mapping[Tuple[str, int], int]


@dataclass
class PrimalSubproblemResult:
    """Solve summary for the primal subproblem."""

    status: int
    objective: Optional[float] = None
    gamma: Optional[float] = None
    slack_sum: float = 0.0
    slack_values: Dict[str, float] = field(default_factory=dict)

    @property
    def is_optimal(self) -> bool:
        return self.status == GRB.OPTIMAL

    def is_feasible(self, tolerance: float = 1e-6) -> bool:
        return self.is_optimal and self.slack_sum <= tolerance


@dataclass
class DualSubproblemResult:
    """Solve summary for the dual subproblem."""

    status: int
    objective: Optional[float] = None
    multipliers: Optional[Dict[str, Any]] = None
    cut_kind: Optional[str] = None

    @property
    def is_optimal(self) -> bool:
        return self.status == GRB.OPTIMAL


@dataclass
class FixedMasterAuditReport:
    """Comparison report for a fixed master solution."""

    x_hat: Dict[str, int]
    u_hat: Dict[Tuple[str, int], int]
    primal: PrimalSubproblemResult
    dual: DualSubproblemResult
    reference_status: Optional[int] = None
    reference_objective: Optional[float] = None
    primal_dual_gap: Optional[float] = None
    primal_reference_gap: Optional[float] = None
    notes: Sequence[str] = field(default_factory=tuple)

    @property
    def is_consistent(self) -> bool:
        return (
            self.primal.is_feasible()
            and self.dual.is_optimal
            and self.primal_dual_gap is not None
            and abs(self.primal_dual_gap) <= 1e-4
        )


class IndexSets:
    """Index sets and helper expressions shared across the formulation."""

    def __init__(self, instance: Instance):
        self.inst = instance
        self.L_hat = instance.L_hat
        self.L_v = instance.L_v if config.USE_VIRTUAL_BASES_OPT else []
        self.L = self.L_hat + self.L_v
        self.P = instance.P
        self.D = instance.D
        self.E = instance.E
        self.N = instance.N
        self.S = list(range(len(config.SERVICE_LEVELS)))
        self.rho = config.SERVICE_LEVELS

        self.A = [arc for arc in instance.A if self._arc_is_allowed(arc)]
        self.A_set = set(self.A)
        self.A_PD = [arc for arc in instance.A_PD if arc in self.A_set]
        self.A_LP = [arc for arc in instance.A_LP if arc in self.A_set]
        self.A_DL = [arc for arc in instance.A_DL if arc in self.A_set]
        self.A_PD_set = set(self.A_PD)
        self.A_LP_set = set(self.A_LP)

        self.out_arcs: Dict[str, list[Tuple[str, str]]] = {}
        self.in_arcs: Dict[str, list[Tuple[str, str]]] = {}
        for arc in self.A:
            self.out_arcs.setdefault(arc[0], []).append(arc)
            self.in_arcs.setdefault(arc[1], []).append(arc)

        self.lp_arcs_from_base = {
            location: [arc for arc in self.A_LP if arc[0] == location]
            for location in self.L
        }
        self.pd_arcs_from_pickup = {
            pickup: [arc for arc in self.A_PD if arc[0] == pickup]
            for pickup in self.P
        }

        self.nearby_bases = {
            pickup: sorted({base for base, node in self.A_LP if node == pickup and base in self.L_hat})
            for pickup in self.P
        }
        self.nearby_bases.update(
            {
                delivery: sorted({base for node, base in self.A_DL if node == delivery and base in self.L_hat})
                for delivery in self.D
            }
        )

        self.nearby_swaps = {
            lv: sorted(
                {
                    swap
                    for _, pickup in self.lp_arcs_from_base.get(lv, [])
                    for swap in instance.E_uv.get((lv, pickup), [])
                }
            )
            for lv in self.L_v
        }
        self.nearby_swaps.update(
            {
                pickup: sorted(
                    {
                        swap
                        for _, delivery in self.pd_arcs_from_pickup.get(pickup, [])
                        for swap in instance.E_uv.get((pickup, delivery), [])
                    }
                )
                for pickup in self.P
            }
        )

        self.H = {
            pickup: sum(
                instance.h.get((pickup, delivery), 0.0)
                for delivery in self.D
                if (pickup, delivery) in self.A_PD_set
            )
            for pickup in self.P
        }

        self.H_hat = {
            delivery: sum(
                instance.h.get((pickup, delivery), 0.0)
                for pickup in self.P
                if (pickup, delivery) in self.A_PD_set
            )
            for delivery in self.D
        }

        self.delivery_demand = {
            delivery: sum(
                instance.h.get((pickup, delivery), 0.0)
                for pickup in self.P
                if (pickup, delivery) in self.A_PD_set
            )
            for delivery in self.D
        }
        self.total_demand = sum(instance.h.values())

        self.phi_keys: list[Tuple[str, str, str]] = []
        self.phi_for_arc: Dict[Tuple[str, str], list[str]] = {}
        self.swap_flow_keys: Dict[str, list[Tuple[str, str, str]]] = {}
        for u_node, v_node in self.A:
            if instance.get_dist(u_node, v_node) <= config.MAX_DIRECT_DISTANCE or (
                u_node in self.D and v_node.startswith("V_")
            ):
                self.phi_keys.append((u_node, v_node, "0"))
                self.phi_for_arc.setdefault((u_node, v_node), []).append("0")
            for swap in instance.E_uv.get((u_node, v_node), []):
                self.phi_keys.append((u_node, v_node, swap))
                self.phi_for_arc.setdefault((u_node, v_node), []).append(swap)
                self.swap_flow_keys.setdefault(swap, []).append((u_node, v_node, swap))

        self.cap_eff = (
            1 - config.ENERGY_SAFETY_MARGIN_EPSILON
        ) * config.BATTERY_CAPACITY_OMEGA_UPPER
        self.energy_exchange_limit = self.cap_eff / config.ENERGY_RATE_OMEGA
        self.service_wait = {
            level: self.rho[level] / (1 - self.rho[level])
            for level in self.S
            if self.rho[level] < 1.0
        }

    @staticmethod
    def _arc_is_allowed(arc: Tuple[str, str]) -> bool:
        if config.USE_VIRTUAL_BASES_OPT:
            return True
        return not arc[0].startswith("V_") and not arc[1].startswith("V_")

    def service_wait_value(
        self,
        u_hat: UHat,
        locations: Optional[Iterable[str]] = None,
    ) -> float:
        active_locations = list(locations) if locations is not None else self.L
        return sum(
            self.service_wait.get(level, 0.0) * u_hat[(location, level)]
            for location in active_locations
            for level in self.S
        )

    def service_wait_expr(
        self,
        u_vars: Mapping[Tuple[str, int], gp.Var],
        locations: Optional[Iterable[str]] = None,
    ) -> gp.LinExpr:
        active_locations = list(locations) if locations is not None else self.L
        return gp.quicksum(
            self.service_wait.get(level, 0.0) * u_vars[location, level]
            for location in active_locations
            for level in self.S
        )

    def outflow_expr(self, psi: Mapping[Tuple[str, str], gp.Var], node: str) -> gp.LinExpr:
        return gp.quicksum(psi[arc] for arc in self.out_arcs.get(node, []))

    def inflow_expr(self, psi: Mapping[Tuple[str, str], gp.Var], node: str) -> gp.LinExpr:
        return gp.quicksum(psi[arc] for arc in self.in_arcs.get(node, []))

    def path_time(self, u_node: str, v_node: str, swap: str) -> float:
        if swap == "0":
            return self.inst.get_time(u_node, v_node)
        return self.inst.get_time(u_node, swap) + self.inst.get_time(swap, v_node)

    def phi_travel_expr(
        self,
        phi: Mapping[Tuple[str, str, str], gp.Var],
        arc_keys: Iterable[Tuple[str, str, str]],
    ) -> gp.LinExpr:
        return gp.quicksum(
            phi[u_node, v_node, swap] * self.path_time(u_node, v_node, swap)
            for u_node, v_node, swap in arc_keys
        )


class PrimalSubproblem:
    """Primal subproblem for fixed master decisions."""

    def __init__(
        self,
        instance: Instance,
        data: IndexSets,
        allow_feasibility_slacks: bool = False,
        slack_penalty: float = 1e6,
    ):
        self.inst = instance
        self.data = data
        self.allow_feasibility_slacks = allow_feasibility_slacks
        self.slack_penalty = slack_penalty

    def solve(
        self,
        x_hat: XHat,
        u_hat: UHat,
    ) -> PrimalSubproblemResult:
        data = self.data
        inst = self.inst
        q_hat = {location: data.service_wait_value(u_hat, [location]) for location in data.L}

        model = gp.Model("Primal_Subproblem")
        model.setParam("OutputFlag", 0)

        y = model.addVars(data.A_LP, lb=0.0, ub=1.0, name="y")
        w_keys = [
            (base, pickup, delivery)
            for base, pickup in data.A_LP
            for delivery in data.D
            if (pickup, delivery) in data.A_PD_set
        ]
        w = model.addVars(w_keys, lb=0.0, name="w")
        demand = model.addVars(data.A_LP, lb=0.0, name="D")
        z_keys = [(base, pickup, level) for base, pickup in data.A_LP for level in data.S]
        z = model.addVars(z_keys, lb=0.0, name="z")
        psi = model.addVars(data.A, lb=0.0, name="psi")
        phi = model.addVars(data.phi_keys, lb=0.0, name="phi")
        xi = model.addVars(data.P + data.D, lb=0.0, name="xi")
        lambd = model.addVars(data.P + data.D, lb=0.0, name="lambda")
        gamma = model.addVar(lb=0.0, name="gamma")

        lv_wait_slack = {}
        lv_energy_slack = {}
        fleet_slack = {}
        slack_expr = gp.LinExpr()
        if self.allow_feasibility_slacks:
            lv_wait_slack = model.addVars(data.L_v, lb=0.0, name="slack_lv_wait")
            lv_energy_slack = model.addVars(data.L_v, lb=0.0, name="slack_lv_energy")
            fleet_slack = model.addVars(["fleet", "budget", "limit"], lb=0.0, name="slack_fleet")
            slack_expr += gp.quicksum(lv_wait_slack[lv] for lv in data.L_v)
            slack_expr += gp.quicksum(lv_energy_slack[lv] for lv in data.L_v)
            slack_expr += gp.quicksum(fleet_slack[key] for key in fleet_slack)

        served_demand = gp.quicksum(psi[pickup, delivery] for pickup, delivery in data.A_PD)
        if self.allow_feasibility_slacks:
            model.setObjective(served_demand - self.slack_penalty * slack_expr, GRB.MAXIMIZE)
        else:
            model.setObjective(served_demand, GRB.MAXIMIZE)

        for pickup in data.P:
            reachable_bases = [base for base in data.L if (base, pickup) in data.A_LP_set]
            if reachable_bases:
                model.addConstr(
                    gp.quicksum(y[base, pickup] for base in reachable_bases) <= x_hat[pickup],
                    name=f"assign_pickup_{pickup}",
                )

        for base, pickup in data.A_LP:
            model.addConstr(y[base, pickup] <= x_hat[base], name=f"assign_base_{base}_{pickup}")
            model.addConstr(
                demand[base, pickup]
                == gp.quicksum(
                    inst.h.get((pickup, delivery), 0.0) * w[base, pickup, delivery]
                    for delivery in data.D
                    if (pickup, delivery) in data.A_PD_set
                ),
                name=f"aggregate_demand_{base}_{pickup}",
            )

            for level in data.S:
                max_pickup_demand = data.H.get(pickup, 0.0)
                model.addConstr(
                    z[base, pickup, level] <= demand[base, pickup],
                    name=f"z_demand_ub_{base}_{pickup}_{level}",
                )
                model.addConstr(
                    z[base, pickup, level] <= max_pickup_demand * u_hat[base, level],
                    name=f"z_service_ub_{base}_{pickup}_{level}",
                )
                model.addConstr(
                    demand[base, pickup] - z[base, pickup, level]
                    <= max_pickup_demand * (1 - u_hat[base, level]),
                    name=f"z_service_lb_{base}_{pickup}_{level}",
                )

            model.addConstr(
                psi[base, pickup]
                == gp.quicksum(data.rho[level] * z[base, pickup, level] for level in data.S),
                name=f"pickup_flow_{base}_{pickup}",
            )

        for base, pickup, delivery in w_keys:
            model.addConstr(w[base, pickup, delivery] <= y[base, pickup], name=f"w_y_ub_{base}_{pickup}_{delivery}")
            model.addConstr(w[base, pickup, delivery] <= x_hat[delivery], name=f"w_x_ub_{base}_{pickup}_{delivery}")
            model.addConstr(
                y[base, pickup] - w[base, pickup, delivery] <= 1 - x_hat[delivery],
                name=f"w_x_lb_{base}_{pickup}_{delivery}",
            )

        for node in data.N + data.L_v:
            model.addConstr(
                data.inflow_expr(psi, node) == data.outflow_expr(psi, node),
                name=f"flow_balance_{node}",
            )

        for pickup, delivery in data.A_PD:
            model.addConstr(psi[pickup, delivery] <= inst.h[pickup, delivery], name=f"demand_cap_{pickup}_{delivery}")

        for u_node, v_node in data.A:
            model.addConstr(
                psi[u_node, v_node]
                == gp.quicksum(
                    phi[u_node, v_node, swap]
                    for swap in data.phi_for_arc.get((u_node, v_node), [])
                ),
                name=f"flow_decomposition_{u_node}_{v_node}",
            )

        for node in data.P + data.D:
            qos_limit = config.MAX_WAITING_FACTOR_P if node in data.P else config.MAX_WAITING_FACTOR_D
            model.addConstr(
                data.outflow_expr(psi, node) <= qos_limit * x_hat[node],
                name=f"qos_{node}",
            )

        for swap in data.E:
            model.addConstr(
                gp.quicksum(phi[key] for key in data.swap_flow_keys.get(swap, []))
                <= config.MAX_FLOW_E * x_hat[swap],
                name=f"swap_capacity_{swap}",
            )

        for node in data.P + data.D:
            mu = inst.mu[node]
            lam_expr = data.outflow_expr(psi, node)
            model.addConstr(lambd[node] == lam_expr, name=f"lambda_def_{node}")
            model.addQConstr(
                4 * mu + (xi[node] + lambd[node] - mu + 1) ** 2
                <= (xi[node] - lambd[node] + mu + 1) ** 2,
                name=f"queue_soc_{node}",
            )
            model.addConstr(
                xi[node] - lambd[node] + mu + 1 >= 0,
                name=f"queue_pos_{node}",
            )

        for lv in data.L_v:
            lv_outflow = gp.quicksum(psi[arc] for arc in data.lp_arcs_from_base.get(lv, []))
            wait_rhs = config.MAX_WAITING_FACTOR_LV * lv_outflow
            wait_slack = lv_wait_slack[lv] if self.allow_feasibility_slacks else 0.0
            model.addConstr(q_hat[lv] <= wait_rhs + wait_slack, name=f"lv_wait_{lv}")

            lv_path_keys = [
                (u_node, v_node, swap)
                for u_node, v_node in data.lp_arcs_from_base.get(lv, [])
                for swap in data.phi_for_arc.get((u_node, v_node), [])
            ]
            lv_travel = data.phi_travel_expr(phi, lv_path_keys)
            lv_replenish = gp.quicksum(
                phi[u_node, v_node, swap]
                for u_node, v_node, swap in lv_path_keys
                if swap != "0"
            )
            energy_slack = lv_energy_slack[lv] if self.allow_feasibility_slacks else 0.0
            model.addConstr(
                q_hat[lv] + lv_travel <= data.energy_exchange_limit * lv_replenish + energy_slack,
                name=f"lv_energy_{lv}",
            )

        for pickup in data.P:
            pickup_path_keys = [
                (u_node, v_node, swap)
                for u_node, v_node in data.pd_arcs_from_pickup.get(pickup, [])
                for swap in data.phi_for_arc.get((u_node, v_node), [])
            ]
            pickup_travel = data.phi_travel_expr(phi, pickup_path_keys)
            pickup_replenish = gp.quicksum(
                phi[u_node, v_node, swap]
                for u_node, v_node, swap in pickup_path_keys
                if swap != "0"
            )
            model.addConstr(
                xi[pickup] + pickup_travel <= data.energy_exchange_limit * pickup_replenish,
                name=f"pickup_energy_{pickup}",
            )

        fleet_expr = data.service_wait_value(u_hat)
        fleet_expr += gp.quicksum(xi[node] for node in data.P + data.D)
        fleet_expr += data.phi_travel_expr(phi, data.phi_keys)

        fleet_overflow = fleet_slack["fleet"] if self.allow_feasibility_slacks else 0.0
        budget_overflow = fleet_slack["budget"] if self.allow_feasibility_slacks else 0.0
        limit_overflow = fleet_slack["limit"] if self.allow_feasibility_slacks else 0.0

        model.addConstr(fleet_expr <= gamma + fleet_overflow, name="fleet_size")
        model.addConstr(
            config.COST_DRONE * gamma
            <= config.BUDGET - sum(inst.f[node] * x_hat[node] for node in data.N) + budget_overflow,
            name="fleet_budget",
        )
        model.addConstr(
            gamma <= config.MAX_FLEET_PER_BASE * sum(x_hat[base] for base in data.L_hat) + limit_overflow,
            name="fleet_limit",
        )

        model.optimize()

        if model.Status != GRB.OPTIMAL:
            return PrimalSubproblemResult(status=model.Status)

        slack_values: Dict[str, float] = {}
        slack_sum = 0.0
        if self.allow_feasibility_slacks:
            for lv in data.L_v:
                slack_values[f"lv_wait[{lv}]"] = lv_wait_slack[lv].X
                slack_values[f"lv_energy[{lv}]"] = lv_energy_slack[lv].X
            for key in fleet_slack:
                slack_values[f"fleet[{key}]"] = fleet_slack[key].X
            slack_sum = sum(slack_values.values())

        return PrimalSubproblemResult(
            status=model.Status,
            objective=model.ObjVal,
            gamma=gamma.X,
            slack_sum=slack_sum,
            slack_values=slack_values,
        )


class DualSubproblem:
    """Dual subproblem used for Benders cuts."""

    def __init__(self, instance: Instance, data: IndexSets):
        self.inst = instance
        self.data = data
        self.model: Optional[gp.Model] = None
        self.vars: Dict[str, Any] = {}

    def _build_model(self) -> None:
        data = self.data
        inst = self.inst

        model = gp.Model("Dual_Subproblem")
        model.setParam("OutputFlag", 0)
        model.setParam("InfUnbdInfo", 1)
        model.setParam("DualReductions", 0)

        a1 = model.addVars(data.P, lb=0.0, name="a1")
        a2 = model.addVars(data.A_LP, lb=0.0, name="a2")
        w_keys = [
            (base, pickup, delivery)
            for base, pickup in data.A_LP
            for delivery in data.D
            if (pickup, delivery) in data.A_PD_set
        ]
        a3 = model.addVars(w_keys, lb=0.0, name="a3")
        a4 = model.addVars(w_keys, lb=0.0, name="a4")
        a5 = model.addVars(w_keys, lb=0.0, name="a5")
        a6 = model.addVars(data.A_LP, lb=-GRB.INFINITY, name="a6")

        z_keys = [(base, pickup, level) for base, pickup in data.A_LP for level in data.S]
        e1 = model.addVars(z_keys, lb=0.0, name="e1")
        e2 = model.addVars(z_keys, lb=0.0, name="e2")
        e3 = model.addVars(z_keys, lb=0.0, name="e3")
        e4 = model.addVars(data.A_LP, lb=-GRB.INFINITY, name="e4")

        b1 = model.addVars(data.N + data.L_v, lb=-GRB.INFINITY, name="b1")
        b2 = model.addVars(data.A_PD, lb=0.0, name="b2")
        b3 = model.addVars(data.A, lb=-GRB.INFINITY, name="b3")

        p1 = model.addVars(data.L_v, lb=0.0, name="p1")
        p2 = model.addVars(data.P + data.D, lb=0.0, name="p2")
        p3 = model.addVars(data.E, lb=0.0, name="p3")

        d1 = model.addVars(data.P + data.D, lb=-GRB.INFINITY, name="d1")
        d2 = model.addVars(data.P + data.D, lb=-GRB.INFINITY, name="d2")
        delta = model.addVars(data.P + data.D, lb=0.0, name="Delta")

        s1 = model.addVars(data.L_v, lb=0.0, name="s1")
        s2 = model.addVars(data.P, lb=0.0, name="s2")
        k1 = model.addVar(lb=0.0, name="k1")
        k2 = model.addVar(lb=0.0, name="k2")
        k3 = model.addVar(lb=0.0, name="k3")

        for base, pickup in data.A_LP:
            model.addConstr(
                a1[pickup]
                + a2[base, pickup]
                + gp.quicksum(
                    a5[base, pickup, delivery] - a3[base, pickup, delivery]
                    for delivery in data.D
                    if (base, pickup, delivery) in w_keys
                )
                >= 0,
                name=f"dual_y_{base}_{pickup}",
            )
            model.addConstr(
                a6[base, pickup]
                - gp.quicksum(e1[base, pickup, level] for level in data.S)
                + gp.quicksum(e3[base, pickup, level] for level in data.S)
                >= 0,
                name=f"dual_D_{base}_{pickup}",
            )

        for base, pickup, delivery in w_keys:
            model.addConstr(
                a3[base, pickup, delivery]
                + a4[base, pickup, delivery]
                - a5[base, pickup, delivery]
                - inst.h[pickup, delivery] * a6[base, pickup]
                >= 0,
                name=f"dual_w_{base}_{pickup}_{delivery}",
            )

        for base, pickup, level in z_keys:
            model.addConstr(
                e1[base, pickup, level]
                + e2[base, pickup, level]
                - e3[base, pickup, level]
                - data.rho[level] * e4[base, pickup]
                >= 0,
                name=f"dual_z_{base}_{pickup}_{level}",
            )

        for u_node, v_node in data.A:
            expr = b1[u_node] - b1[v_node] + b3[u_node, v_node]
            if (u_node, v_node) in data.A_LP_set:
                expr += e4[u_node, v_node]
                if u_node in data.L_v:
                    expr += (config.TAU_LV / math.log(config.GAMMA)) * p1[u_node]
            if (u_node, v_node) in data.A_PD_set:
                expr += b2[u_node, v_node]
            if u_node in data.P + data.D:
                expr += p2[u_node] + delta[u_node] - d2[u_node]
            rhs = 1.0 if (u_node, v_node) in data.A_PD_set else 0.0
            model.addConstr(expr >= rhs, name=f"dual_psi_{u_node}_{v_node}")

        for u_node, v_node, swap in data.phi_keys:
            travel_time = data.path_time(u_node, v_node, swap)
            expr = -b3[u_node, v_node] + travel_time * k1
            if swap != "0":
                expr += p3[swap]
            if u_node in data.L_v:
                expr += (travel_time - (data.energy_exchange_limit if swap != "0" else 0.0)) * s1[u_node]
            if (u_node, v_node) in data.A_PD_set:
                expr += (travel_time - (data.energy_exchange_limit if swap != "0" else 0.0)) * s2[u_node]
            model.addConstr(expr >= 0, name=f"dual_phi_{u_node}_{v_node}_{swap}")

        for node in data.P + data.D:
            model.addConstr(
                -delta[node] - d2[node] + k1 + (s2[node] if node in data.P else 0.0) >= 0,
                name=f"dual_xi_{node}",
            )
            model.addConstr(
                d1[node] * d1[node] + d2[node] * d2[node] <= delta[node] * delta[node],
                name=f"dual_soc_{node}",
            )

        model.addConstr(-k1 + config.COST_DRONE * k2 + k3 >= 0, name="dual_gamma")

        self.model = model
        self.vars = {
            "a1": a1,
            "a2": a2,
            "a4": a4,
            "a5": a5,
            "e2": e2,
            "e3": e3,
            "b2": b2,
            "p1": p1,
            "p2": p2,
            "p3": p3,
            "d1": d1,
            "d2": d2,
            "Delta": delta,
            "s1": s1,
            "k1": k1,
            "k2": k2,
            "k3": k3,
        }

    @staticmethod
    def _extract_container_values(container: Any, attr_name: str) -> Any:
        if hasattr(container, "items"):
            values = {}
            for key, var in container.items():
                try:
                    values[key] = var.getAttr(attr_name)
                except gp.GurobiError:
                    values[key] = 0.0
                except AttributeError:
                    values[key] = 0.0
            return values

        try:
            return container.getAttr(attr_name)
        except gp.GurobiError:
            return 0.0
        except AttributeError:
            return 0.0

    def solve(
        self,
        x_hat: XHat,
        u_hat: UHat,
    ) -> DualSubproblemResult:
        if self.model is None:
            self._build_model()

        assert self.model is not None
        data = self.data
        inst = self.inst
        vars_map = self.vars
        q_hat = {location: data.service_wait_value(u_hat, [location]) for location in data.L}

        objective = gp.LinExpr()
        for pickup in data.P:
            objective += vars_map["a1"][pickup] * x_hat[pickup]

        for base, pickup in data.A_LP:
            objective += vars_map["a2"][base, pickup] * x_hat[base]

        for base, pickup, delivery in vars_map["a4"].keys():
            objective += vars_map["a4"][base, pickup, delivery] * x_hat[delivery]
            objective += vars_map["a5"][base, pickup, delivery] * (1 - x_hat[delivery])

        for base, pickup, level in vars_map["e2"].keys():
            pickup_demand = data.H.get(pickup, 0.0)
            objective += pickup_demand * (
                vars_map["e2"][base, pickup, level] * u_hat[base, level]
                + vars_map["e3"][base, pickup, level] * (1 - u_hat[base, level])
            )

        for pickup, delivery in data.A_PD:
            objective += vars_map["b2"][pickup, delivery] * inst.h[pickup, delivery]

        for lv in data.L_v:
            objective -= (vars_map["p1"][lv] + vars_map["s1"][lv]) * q_hat[lv]

        for node in data.P + data.D:
            qos_limit = config.MAX_WAITING_FACTOR_P if node in data.P else config.MAX_WAITING_FACTOR_D
            objective += vars_map["p2"][node] * qos_limit * x_hat[node]
            mu = inst.mu[node]
            objective += 2 * math.sqrt(mu) * vars_map["d1"][node]
            objective += (1 - mu) * vars_map["d2"][node]
            objective += (mu + 1) * vars_map["Delta"][node]

        for swap in data.E:
            objective += vars_map["p3"][swap] * config.MAX_FLOW_E * x_hat[swap]

        objective -= vars_map["k1"] * data.service_wait_value(u_hat)
        objective += vars_map["k2"] * (
            config.BUDGET - sum(inst.f[node] * x_hat[node] for node in data.N)
        )
        objective += vars_map["k3"] * config.MAX_FLEET_PER_BASE * sum(
            x_hat[base] for base in data.L_hat
        )

        self.model.setObjective(objective, GRB.MINIMIZE)
        self.model.optimize()

        if self.model.Status == GRB.OPTIMAL:
            multipliers: Dict[str, Any] = {}
            for name, container in vars_map.items():
                multipliers[name] = self._extract_container_values(container, "X")
            return DualSubproblemResult(
                status=self.model.Status,
                objective=self.model.ObjVal,
                multipliers=multipliers,
                cut_kind="optimality",
            )

        if self.model.Status == GRB.UNBOUNDED:
            ray: Dict[str, Any] = {}
            for name, container in vars_map.items():
                ray[name] = self._extract_container_values(container, "UnbdRay")
            return DualSubproblemResult(
                status=self.model.Status,
                objective=None,
                multipliers=ray,
                cut_kind="feasibility",
            )

        return DualSubproblemResult(status=self.model.Status)


class BendersMaster:
    """Shared Benders master problem and cut builder."""

    def __init__(
        self,
        instance: Instance,
        max_iter: int = 100,
        gap_tol: float = 1e-4,
        time_limit: float = 3600,
        verbose: bool = True,
    ):
        self.inst = instance
        self.data = IndexSets(instance)
        self.max_iter = max_iter
        self.gap_tol = gap_tol
        self.time_limit = time_limit
        self.verbose = verbose
        self.best_obj = -math.inf
        self.best_sol: Optional[Dict[str, Any]] = None
        self._evaluated_signatures: set[Tuple[Tuple[int, ...], Tuple[int, ...]]] = set()
        self.model: Optional[gp.Model] = None
        self.x: Optional[gp.tupledict] = None
        self.u: Optional[gp.tupledict] = None
        self.theta: Optional[gp.Var] = None

    def _build_master_problem(self) -> None:
        data = self.data
        inst = self.inst

        model = gp.Model("Benders_Master")
        model.setParam("OutputFlag", 1 if self.verbose else 0)

        x = model.addVars(data.N + data.L_v, vtype=GRB.BINARY, name="x")
        u = model.addVars(data.L, data.S, vtype=GRB.BINARY, name="u")
        theta = model.addVar(lb=0.0, ub=data.total_demand, name="theta")

        model.setObjective(theta, GRB.MAXIMIZE)

        for location in data.L:
            model.addConstr(
                gp.quicksum(u[location, level] for level in data.S) == x[location],
                name=f"service_select_{location}",
            )

        for node in data.P + data.D:
            nearby_bases = data.nearby_bases.get(node, [])
            if nearby_bases:
                model.addConstr(
                    x[node] <= gp.quicksum(x[base] for base in nearby_bases),
                    name=f"nearby_base_{node}",
                )
            else:
                model.addConstr(x[node] == 0, name=f"forbid_node_{node}")

        for node in data.L_v + data.P:
            nearby_swaps = data.nearby_swaps.get(node, [])
            if nearby_swaps:
                model.addConstr(
                    x[node] <= gp.quicksum(x[swap] for swap in nearby_swaps),
                    name=f"nearby_swap_{node}",
                )
            else:
                model.addConstr(x[node] == 0, name=f"forbid_swap_{node}")

        if config.USE_VIRTUAL_BASES_OPT:
            for delivery in data.D:
                model.addConstr(
                    x[inst.d_to_lv[delivery]] <= x[delivery],
                    name=f"virtual_base_link_{delivery}",
                )

        fixed_cost = gp.quicksum(inst.f[node] * x[node] for node in data.N)
        minimum_wait = data.service_wait_expr(u)
        model.addConstr(fixed_cost + config.COST_DRONE * minimum_wait <= config.BUDGET, name="budget_lb")
        model.addConstr(
            minimum_wait <= config.MAX_FLEET_PER_BASE * gp.quicksum(x[base] for base in data.L_hat),
            name="fleet_lb",
        )
        model.addConstr(
            theta <= gp.quicksum(data.H[pickup] * x[pickup] for pickup in data.P),
            name="theta_pickup_bound",
        )
        model.addConstr(
            theta <= gp.quicksum(data.delivery_demand[delivery] * x[delivery] for delivery in data.D),
            name="theta_delivery_bound",
        )
        model.addConstr(
            theta <= data.total_demand * gp.quicksum(x[base] for base in data.L_hat),
            name="theta_base_bound",
        )

        self.model = model
        self.x = x
        self.u = u
        self.theta = theta

    def _solution_signature(
        self,
        x_hat: XHat,
        u_hat: UHat,
    ) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
        x_sig = tuple(x_hat[node] for node in self.data.N + self.data.L_v)
        u_sig = tuple(u_hat[location, level] for location in self.data.L for level in self.data.S)
        return x_sig, u_sig

    def _extract_callback_solution(
        self,
        model: gp.Model,
    ) -> Tuple[Dict[str, int], Dict[Tuple[str, int], int], float]:
        assert self.x is not None and self.u is not None and self.theta is not None
        x_hat = {
            node: 1 if model.cbGetSolution(self.x[node]) > 0.5 else 0
            for node in self.data.N + self.data.L_v
        }
        u_hat = {
            (location, level): 1 if model.cbGetSolution(self.u[location, level]) > 0.5 else 0
            for location in self.data.L
            for level in self.data.S
        }
        theta_value = model.cbGetSolution(self.theta)
        return x_hat, u_hat, theta_value

    def _build_cut_expression(self, multipliers: Dict[str, Any]) -> gp.LinExpr:
        assert self.x is not None and self.u is not None
        data = self.data
        inst = self.inst

        expr = gp.LinExpr()
        for pickup in data.P:
            expr += multipliers["a1"][pickup] * self.x[pickup]

        for base, pickup in data.A_LP:
            expr += multipliers["a2"][base, pickup] * self.x[base]

        for base, pickup, delivery in multipliers["a4"].keys():
            expr += multipliers["a4"][base, pickup, delivery] * self.x[delivery]
            expr += multipliers["a5"][base, pickup, delivery] * (1 - self.x[delivery])

        for base, pickup, level in multipliers["e2"].keys():
            pickup_demand = data.H.get(pickup, 0.0)
            expr += pickup_demand * (
                multipliers["e2"][base, pickup, level] * self.u[base, level]
                + multipliers["e3"][base, pickup, level] * (1 - self.u[base, level])
            )

        for pickup, delivery in data.A_PD:
            expr += multipliers["b2"][pickup, delivery] * inst.h[pickup, delivery]

        for lv in data.L_v:
            expr -= (multipliers["p1"][lv] + multipliers["s1"][lv]) * data.service_wait_expr(self.u, [lv])

        for node in data.P + data.D:
            qos_limit = config.MAX_WAITING_FACTOR_P if node in data.P else config.MAX_WAITING_FACTOR_D
            expr += multipliers["p2"][node] * qos_limit * self.x[node]
            mu = inst.mu[node]
            expr += 2 * math.sqrt(mu) * multipliers["d1"][node]
            expr += (1 - mu) * multipliers["d2"][node]
            expr += (mu + 1) * multipliers["Delta"][node]

        for swap in data.E:
            expr += multipliers["p3"][swap] * config.MAX_FLOW_E * self.x[swap]

        expr -= multipliers["k1"] * data.service_wait_expr(self.u)
        expr += multipliers["k2"] * (
            config.BUDGET - gp.quicksum(inst.f[node] * self.x[node] for node in data.N)
        )
        expr += multipliers["k3"] * config.MAX_FLEET_PER_BASE * gp.quicksum(
            self.x[base] for base in data.L_hat
        )
        return expr

    def get_solution(self) -> Optional[Dict[str, Any]]:
        return self.best_sol


class BendersDual(BendersMaster):
    """Benders decomposition driven by the dual subproblem."""

    def solve(self) -> Optional[float]:
        self._build_master_problem()
        assert self.model is not None and self.theta is not None
        dual_problem = DualSubproblem(self.inst, self.data)
        start_time = time.time()

        def callback(model: gp.Model, where: int) -> None:
            if where != GRB.Callback.MIPSOL:
                return
            if time.time() - start_time > self.time_limit:
                model.terminate()
                return
            if len(self._evaluated_signatures) >= self.max_iter:
                model.terminate()
                return

            x_hat, u_hat, theta_value = self._extract_callback_solution(model)
            signature = self._solution_signature(x_hat, u_hat)
            if signature in self._evaluated_signatures:
                return
            self._evaluated_signatures.add(signature)

            dual_result = dual_problem.solve(x_hat, u_hat)
            if dual_result.multipliers is None:
                return

            cut_expr = self._build_cut_expression(dual_result.multipliers)
            if dual_result.cut_kind == "feasibility":
                model.cbLazy(cut_expr >= 0)
                return

            assert dual_result.objective is not None
            if theta_value > dual_result.objective + self.gap_tol:
                model.cbLazy(self.theta <= cut_expr)


        self.model.setParam("LazyConstraints", 1)
        self.model.optimize(callback)

        return self.model.ObjVal if self.model.Status == GRB.OPTIMAL else None


class BendersPrimal(BendersMaster):
    """Benders decomposition with primal subproblem validation."""

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.validation_issues: list[str] = []

    def solve(self) -> Optional[float]:
        self._build_master_problem()
        assert self.model is not None and self.theta is not None
        primal_problem = PrimalSubproblem(self.inst, self.data)
        start_time = time.time()

        def callback(model: gp.Model, where: int) -> None:
            if where != GRB.Callback.MIPSOL:
                return
            if time.time() - start_time > self.time_limit:
                model.terminate()
                return
            if len(self._evaluated_signatures) >= self.max_iter:
                model.terminate()
                return

            x_hat, u_hat, theta_value = self._extract_callback_solution(model)
            signature = self._solution_signature(x_hat, u_hat)
            if signature in self._evaluated_signatures:
                return
            self._evaluated_signatures.add(signature)

            primal_result = primal_problem.solve(x_hat, u_hat)


            if primal_result.multipliers is not None:
                cut_expr = self._build_cut_expression(primal_result.multipliers)
                if primal_result.cut_kind == "feasibility":
                    model.cbLazy(cut_expr >= 0)
                elif primal_result.objective is not None and theta_value > primal_result.objective + self.gap_tol:
                    model.cbLazy(self.theta <= cut_expr)


        self.model.setParam("LazyConstraints", 1)
        self.model.optimize(callback)

        return  self.model.ObjVal if self.model.Status == GRB.OPTIMAL else None

    def get_validation_issues(self) -> Sequence[str]:
        return tuple(self.validation_issues)


class FixedMasterAudit:
    """Compare primal, dual, and benchmark objectives for fixed binaries."""

    def __init__(self, instance: Instance):
        self.inst = instance
        self.data = IndexSets(instance)
        self.primal = PrimalSubproblem(instance, self.data)
        self.dual = DualSubproblem(instance, self.data)

    def solve_benchmark(
        self,
        time_limit: float = 300,
        mip_gap: float = 0.01,
    ) -> Dict[str, Any]:
        from .gurobi_model import MLNDModel

        benchmark = MLNDModel(self.inst)
        benchmark.build()
        benchmark.model.update()
        benchmark.model.setParam("OutputFlag", 0)
        benchmark.model.setParam("TimeLimit", time_limit)
        benchmark.model.setParam("MIPGap", mip_gap)
        benchmark.model.optimize()

        result: Dict[str, Any] = {"status": benchmark.model.Status}
        if benchmark.model.SolCount == 0:
            return result

        x_hat = {
            node: int(round(benchmark.model.getVarByName(f"x[{node}]").X))
            for node in self.data.N + self.data.L_v
        }
        u_hat = {
            (location, level): int(round(benchmark.model.getVarByName(f"u[{location},{level}]").X))
            for location in self.data.L
            for level in self.data.S
        }
        result.update({
            "objective": benchmark.model.ObjVal,
            "x": x_hat,
            "u": u_hat,
        })
        return result

    def solve_fixed_master_benchmark(
        self,
        x_hat: XHat,
        u_hat: UHat,
        time_limit: float = 300,
    ) -> Dict[str, Any]:
        from .gurobi_model import MLNDModel

        reference = MLNDModel(self.inst)
        reference.build()
        reference.model.update()
        reference.model.setParam("OutputFlag", 0)
        reference.model.setParam("TimeLimit", time_limit)

        for node in self.data.N + self.data.L_v:
            reference.model.addConstr(
                reference.model.getVarByName(f"x[{node}]") == x_hat[node],
                name=f"fix_x_{node}",
            )

        for location in self.data.L:
            for level in self.data.S:
                reference.model.addConstr(
                    reference.model.getVarByName(f"u[{location},{level}]") == u_hat[location, level],
                    name=f"fix_u_{location}_{level}",
                )

        reference.model.optimize()
        result = {"status": reference.model.Status}
        if reference.model.SolCount > 0:
            result["objective"] = reference.model.ObjVal
        return result

    def audit_fixed_master(
        self,
        x_hat: XHat,
        u_hat: UHat,
        compare_to_reference: bool = True,
        reference_time_limit: float = 300,
    ) -> FixedMasterAuditReport:
        primal_result = self.primal.solve(x_hat, u_hat)
        dual_result = self.dual.solve(x_hat, u_hat)

        notes = []
        primal_dual_gap = None
        if primal_result.is_feasible() and dual_result.is_optimal:
            assert primal_result.objective is not None and dual_result.objective is not None
            primal_dual_gap = primal_result.objective - dual_result.objective
            if abs(primal_dual_gap) > 1e-4:
                notes.append("Primal and dual subproblem objectives do not match within tolerance.")
        elif not primal_result.is_feasible():
            notes.append("Primal subproblem is infeasible or only feasible with positive slacks.")
        elif not dual_result.is_optimal:
            notes.append("Dual subproblem did not return an optimal objective for this master solution.")

        reference_status = None
        reference_objective = None
        primal_reference_gap = None
        if compare_to_reference:
            reference_result = self.solve_fixed_master_benchmark(x_hat, u_hat, reference_time_limit)
            reference_status = reference_result["status"]
            reference_objective = reference_result.get("objective")
            if primal_result.objective is not None and reference_objective is not None:
                primal_reference_gap = primal_result.objective - reference_objective
                if abs(primal_reference_gap) > 1e-4:
                    notes.append(
                        "Primal subproblem and fixed-master benchmark objective do not match within tolerance."
                    )

        return FixedMasterAuditReport(
            x_hat=dict(x_hat),
            u_hat=dict(u_hat),
            primal=primal_result,
            dual=dual_result,
            reference_status=reference_status,
            reference_objective=reference_objective,
            primal_dual_gap=primal_dual_gap,
            primal_reference_gap=primal_reference_gap,
            notes=tuple(notes),
        )
