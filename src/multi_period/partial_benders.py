"""Partial Benders solver for the multi-period Regime II model.

The master keeps the shared facility decisions plus period-specific logical
flow and queue variables. Route-level physical flows ``varphi`` stay in a
period LP recourse, so each iteration adds one optimality cut per period.
"""

import math
import os
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, Tuple

import gurobipy as gp
from gurobipy import GRB

if __package__ in (None, ""):
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)

from src import config
from src.model.benders import RegimeIIData
from src.multi_period.gurobi_model import (
    MultiPeriodInstance,
    _as_period_ids,
    load_multi_period_instance,
)


TOL = 1e-6
SLACK_TOL = 1e-5


@dataclass
class PartialMasterSolution:
    obj: float
    x: Dict[str, float]
    y: Dict[Tuple[str, str], float]
    z: Dict[Tuple[str, int], float]
    Gamma: float
    theta: Dict[str, float]
    psi: Dict[str, Dict[Tuple[str, str], float]]
    xi: Dict[str, Dict[str, float]]


class RouteRecourseSubproblem:
    """LP recourse for route decomposition, energy, swap capacity, and fleet."""

    def __init__(self, data: RegimeIIData, penalty: float = 1e5, output: bool = False):
        self.data = data
        self.penalty = float(penalty)
        self.output = output

    def solve(
        self,
        x_hat: Dict[str, float],
        psi_hat: Dict[Tuple[str, str], float],
        xi_hat: Dict[str, float],
        gamma_hat: float,
    ) -> Dict[str, Any]:
        data = self.data
        m = gp.Model("RegimeII_RouteRecourse")
        m.setParam("OutputFlag", 1 if self.output else 0)
        m.setParam("DualReductions", 0)
        m.setParam("FeasibilityTol", 1e-9)
        m.setParam("OptimalityTol", 1e-9)

        varphi = m.addVars(data.varphi_keys, lb=0.0, name="varphi")
        route_pos = m.addVars(data.A_F, lb=0.0, name="route_pos")
        route_neg = m.addVars(data.A_F, lb=0.0, name="route_neg")
        swap_slack = m.addVars(data.E, lb=0.0, name="swap_slack")
        energy_l_slack = m.addVars(data.L, lb=0.0, name="energy_l_slack")
        energy_hs_slack = m.addVars(data.A_HS, lb=0.0, name="energy_hs_slack")
        fleet_slack = m.addVar(lb=0.0, name="fleet_slack")

        m.setObjective(
            gp.quicksum(data.flight_cost[u, v, k] * varphi[u, v, k] for u, v, k in data.varphi_keys)
            + self.penalty
            * (
                gp.quicksum(route_pos[u, v] + route_neg[u, v] for u, v in data.A_F)
                + gp.quicksum(swap_slack[k] for k in data.E)
                + gp.quicksum(energy_l_slack[l] for l in data.L)
                + gp.quicksum(energy_hs_slack[i, s] for i, s in data.A_HS)
                + fleet_slack
            ),
            GRB.MINIMIZE,
        )

        cons: Dict[str, Any] = {"route": {}, "swap": {}, "energyL": {}, "energyHS": {}}
        for u, v in data.A_F:
            cons["route"][u, v] = m.addConstr(
                gp.quicksum(
                    varphi[u, v, k]
                    for k in data.route_options.get((u, v), [])
                    if (u, v, k) in varphi
                )
                + route_pos[u, v]
                - route_neg[u, v]
                == psi_hat.get((u, v), 0.0),
                name=f"route_{u}_{v}",
            )

        for k in data.E:
            cons["swap"][k] = m.addConstr(
                gp.quicksum(varphi[u, v, k] for u, v, kk in data.varphi_keys if kk == k)
                - swap_slack[k]
                <= data.swap_capacity(k) * x_hat.get(k, 0.0),
                name=f"swap_{k}",
            )

        for l in data.L:
            cons["energyL"][l] = m.addConstr(
                gp.quicksum(
                    data.route_energy_coef[l, i, k] * varphi[l, i, k]
                    for u, i, k in data.varphi_keys
                    if u == l and (l, i) in data.A_LH_set
                )
                - energy_l_slack[l]
                <= 0.0,
                name=f"energyL_{l}",
            )

        for i, s in data.A_HS:
            cons["energyHS"][i, s] = m.addConstr(
                gp.quicksum(
                    data.route_energy_coef[i, u, k] * varphi[i, u, k]
                    for h, u, k in data.varphi_keys
                    if h == i and (i, u) in data.A_HH_set
                )
                + gp.quicksum(
                    data.route_energy_coef[s, u, k] * varphi[s, u, k]
                    for h, u, k in data.varphi_keys
                    if h == s and ((s, u) in data.A_SL or (s, u) in data.A_SS)
                )
                - energy_hs_slack[i, s]
                <= -data.omega * (xi_hat.get(i, 0.0) + xi_hat.get(s, 0.0)),
                name=f"energyHS_{i}_{s}",
            )

        cons["fleet"] = m.addConstr(
            gp.quicksum(data.t[u, v, k] * varphi[u, v, k] for u, v, k in data.varphi_keys)
            - fleet_slack
            <= gamma_hat - sum(xi_hat.get(u, 0.0) for u in data.HLS),
            name="fleet",
        )

        m.optimize()
        if m.Status != GRB.OPTIMAL:
            return {"status": m.Status, "obj": None, "cut": None, "slack": math.inf}

        def pi(constr: gp.Constr) -> float:
            value = float(constr.Pi)
            return 0.0 if abs(value) <= TOL else value

        duals: Dict[str, Any] = {
            "route": {key: pi(constr) for key, constr in cons["route"].items()},
            "swap": {key: pi(constr) for key, constr in cons["swap"].items()},
            "energyHS": {key: pi(constr) for key, constr in cons["energyHS"].items()},
            "fleet": pi(cons["fleet"]),
        }

        rhs_at_hat = self.dual_expr_value(duals, x_hat, psi_hat, xi_hat, gamma_hat)
        flight_obj = sum(
            data.flight_cost[u, v, k] * max(0.0, varphi[u, v, k].X)
            for u, v, k in data.varphi_keys
        )
        slack_total = (
            sum(max(0.0, route_pos[u, v].X) + max(0.0, route_neg[u, v].X) for u, v in data.A_F)
            + sum(max(0.0, swap_slack[k].X) for k in data.E)
            + sum(max(0.0, energy_l_slack[l].X) for l in data.L)
            + sum(max(0.0, energy_hs_slack[i, s].X) for i, s in data.A_HS)
            + max(0.0, fleet_slack.X)
        )
        corrected_obj = flight_obj + self.penalty * slack_total

        return {
            "status": m.Status,
            "obj": float(corrected_obj),
            "model_obj": float(m.ObjVal),
            "duals": duals,
            "rhs_at_hat": rhs_at_hat,
            "slack": float(slack_total),
        }

    def dual_expr_value(
        self,
        duals: Dict[str, Any],
        x_ref: Dict[str, float],
        psi_ref: Dict[Tuple[str, str], float],
        xi_ref: Dict[str, float],
        gamma_ref: float,
    ) -> float:
        data = self.data
        expr = 0.0
        expr += sum(duals["route"][u, v] * psi_ref.get((u, v), 0.0) for u, v in data.A_F)
        expr += sum(duals["swap"][k] * data.swap_capacity(k) * x_ref.get(k, 0.0) for k in data.E)
        expr += sum(
            duals["energyHS"][i, s] * (-data.omega * (xi_ref.get(i, 0.0) + xi_ref.get(s, 0.0)))
            for i, s in data.A_HS
        )
        expr += duals["fleet"] * (gamma_ref - sum(xi_ref.get(u, 0.0) for u in data.HLS))
        return expr

    def dual_expr_master(
        self,
        duals: Dict[str, Any],
        x: Any,
        psi: Any,
        xi: Any,
        gamma: Any,
        period: str,
    ) -> gp.LinExpr:
        data = self.data
        expr = gp.LinExpr()
        expr += gp.quicksum(duals["route"][u, v] * psi[period, u, v] for u, v in data.A_F)
        expr += gp.quicksum(duals["swap"][k] * data.swap_capacity(k) * x[k] for k in data.E)
        expr += gp.quicksum(
            duals["energyHS"][i, s] * (-data.omega * (xi[period, i] + xi[period, s]))
            for i, s in data.A_HS
        )
        expr += duals["fleet"] * (gamma - gp.quicksum(xi[period, u] for u in data.HLS))
        return expr


class MultiPeriodPartialBenders:
    """Partial Benders: logical flow and queue constraints in master, routes in LP recourse."""

    def __init__(
        self,
        mp_instance: MultiPeriodInstance,
        *,
        max_iter: int = 50,
        gap_tol: float = 1e-4,
        master_time_limit: float = 30.0,
        master_mip_gap: float = 1e-5,
        recourse_penalty: float = 1e5,
        verbose: bool = True,
    ):
        self.mp = mp_instance
        self.base = mp_instance.base
        self.periods = list(mp_instance.periods)
        self.weights = dict(mp_instance.weights)
        self.max_iter = int(max_iter)
        self.gap_tol = float(gap_tol)
        self.master_time_limit = float(master_time_limit)
        self.master_mip_gap = float(master_mip_gap)
        self.recourse_penalty = float(recourse_penalty)
        self.verbose = verbose

        self.data = RegimeIIData(self.base)
        self.model = None
        self.x = None
        self.y = None
        self.z = None
        self.rho_l = None
        self.rho_s = None
        self.alpha = None
        self.beta = None
        self.psi = None
        self.lambd = None
        self.xi = None
        self.theta = None
        self.Gamma = None

        self.best_obj = -math.inf
        self.best_solution = None
        self.history: list[Dict[str, Any]] = []
        self.subproblem_time = 0.0
        self.converged = False
        self.stop_reason = None

    def _demand(self, period: str, i: str, j: str) -> float:
        return float(self.mp.demand.get(period, {}).get((i, j), 0.0))

    def _p_chain(self, hub: str) -> float:
        return self.data.p_chain(hub)

    def _build_master(self) -> None:
        data = self.data
        m = gp.Model("RegimeII_MultiPeriod_PartialBendersMaster")
        m.setParam("OutputFlag", 1 if self.verbose else 0)
        m.setParam("Threads", 1)
        m.setParam("TimeLimit", self.master_time_limit)
        m.setParam("MIPGap", self.master_mip_gap)

        C = data.C
        q_nodes = data.L + data.S
        alpha_keys = [(period, l, i, j) for period in self.periods for l, i, j in data.alpha_keys]
        beta_keys = [(period, i, j) for period in self.periods for i, j in data.beta_keys]
        psi_keys = [(period, u, v) for period in self.periods for u, v in data.A]

        self.x = m.addVars(data.N_x, vtype=GRB.BINARY, name="x")
        self.y = m.addVars([(i, l) for l, i in data.A_LH], vtype=GRB.BINARY, name="y")
        self.z = m.addVars(data.H, C, vtype=GRB.BINARY, name="z")
        self.Gamma = m.addVar(lb=0.0, vtype=GRB.INTEGER, name="Gamma")
        self.rho_l = m.addVars(self.periods, data.L, lb=0.0, name="rho_l")
        self.rho_s = m.addVars(self.periods, data.S, lb=0.0, name="rho_s")
        self.alpha = m.addVars(alpha_keys, lb=0.0, name="alpha")
        self.beta = m.addVars(beta_keys, lb=0.0, name="beta")
        self.psi = m.addVars(psi_keys, lb=0.0, name="psi")
        self.lambd = m.addVars(self.periods, data.H, lb=0.0, name="lambda")
        self.xi = m.addVars(self.periods, data.HLS, lb=0.0, name="xi")
        self.theta = m.addVars(self.periods, lb=-GRB.INFINITY, ub=0.0, name="theta")

        q_v1 = m.addVars(self.periods, q_nodes, lb=-GRB.INFINITY, name="q_v1")
        q_v2 = m.addVars(self.periods, q_nodes, lb=-GRB.INFINITY, name="q_v2")
        q_t = m.addVars(self.periods, q_nodes, lb=0.0, name="q_t")
        h_v1 = m.addVars(self.periods, data.H, lb=-GRB.INFINITY, name="h_v1")
        h_v2 = m.addVars(self.periods, data.H, lb=-GRB.INFINITY, name="h_v2")
        h_t = m.addVars(self.periods, data.H, lb=0.0, name="h_t")

        fixed = gp.quicksum(float(self.base.f.get(l, config.COST_BASE)) * self.x[l] for l in data.L)
        fixed += gp.quicksum(float(getattr(config, "COST_HUB_FIXED", 0.0)) * self.x[i] for i in data.H)
        fixed += gp.quicksum(float(self.base.f.get(k, config.COST_SWAPPING)) * self.x[k] for k in data.E)
        fixed += gp.quicksum(float(config.COST_HUB[c]) * self.z[i, c] for i in data.H for c in C)

        revenue = gp.quicksum(
            self.weights[period] * data.revenue * self.psi[period, i, j]
            for period in self.periods
            for i, j in data.A_HH
        )
        recourse = gp.quicksum(self.weights[period] * self.theta[period] for period in self.periods)
        m.setObjective(revenue + recourse - fixed - data.gamma_cost * self.Gamma, GRB.MAXIMIZE)

        for i in data.H:
            bases = [l for l, h in data.A_LH if h == i]
            if bases:
                m.addConstr(gp.quicksum(self.y[i, l] for l in bases) == self.x[i], name=f"assign_{i}")
            else:
                m.addConstr(self.x[i] == 0, name=f"no_assign_{i}")
            m.addConstr(gp.quicksum(self.z[i, c] for c in C) == self.x[i], name=f"capacity_{i}")

        for l, i in data.A_LH:
            m.addConstr(self.y[i, l] <= self.x[l], name=f"assign_active_{i}_{l}")

        mu_expr = {i: gp.quicksum(data.mu_levels[c] * self.z[i, c] for c in C) for i in data.H}
        sqrt_mu_expr = {i: gp.quicksum(math.sqrt(data.mu_levels[c]) * self.z[i, c] for c in C) for i in data.H}
        min_route_time = {
            (u, v): min(data.t[u, v, k] for k in data.route_options.get((u, v), []))
            for u, v in data.A_F
            if data.route_options.get((u, v))
        }
        min_route_energy = {
            (u, v): min(data.route_energy_coef[u, v, k] for k in data.route_options.get((u, v), []))
            for u, v in data.A_F
            if data.route_options.get((u, v))
        }

        for period in self.periods:
            for l in data.L:
                m.addConstr(self.rho_l[period, l] <= self.x[l], name=f"rho_L_{period}_{l}")
            for i, s in data.A_HS:
                m.addConstr(self.rho_s[period, s] <= self.x[i], name=f"rho_S_{period}_{s}")

            for l, i, j in data.alpha_keys:
                m.addConstr(self.alpha[period, l, i, j] <= self.rho_l[period, l], name=f"alpha_1_{period}_{l}_{i}_{j}")
                m.addConstr(self.alpha[period, l, i, j] <= self.y[i, l], name=f"alpha_2_{period}_{l}_{i}_{j}")
                m.addConstr(self.alpha[period, l, i, j] <= self.x[j], name=f"alpha_3_{period}_{l}_{i}_{j}")
                m.addConstr(
                    self.alpha[period, l, i, j] >= self.rho_l[period, l] + self.y[i, l] + self.x[j] - 2,
                    name=f"alpha_4_{period}_{l}_{i}_{j}",
                )

            for s, i, j in data.beta_support:
                m.addConstr(self.beta[period, i, j] <= self.rho_s[period, s], name=f"beta_1_{period}_{i}_{j}")
                m.addConstr(self.beta[period, i, j] <= self.x[j], name=f"beta_2_{period}_{i}_{j}")
                m.addConstr(
                    self.beta[period, i, j] >= self.rho_s[period, s] + self.x[j] - 1,
                    name=f"beta_3_{period}_{i}_{j}",
                )

            for i, j in data.A_HH:
                p_i = self._p_chain(i)
                demand = self._demand(period, i, j)
                m.addConstr(
                    self.psi[period, i, j]
                    == demand
                    * (
                        (1.0 - p_i)
                        * gp.quicksum(
                            self.alpha[period, l, i, j]
                            for l, h in data.A_LH
                            if h == i and (period, l, i, j) in self.alpha
                        )
                        + p_i * self.beta[period, i, j]
                    ),
                    name=f"psi_HH_{period}_{i}_{j}",
                )

            for l, i in data.A_LH:
                m.addConstr(
                    self.psi[period, l, i]
                    == gp.quicksum(
                        self._demand(period, i, j) * (1.0 - self._p_chain(i)) * self.alpha[period, l, i, j]
                        for h, j in data.A_HH
                        if h == i and (period, l, i, j) in self.alpha
                    ),
                    name=f"psi_LH_{period}_{l}_{i}",
                )

            for s, i in data.A_SH:
                m.addConstr(
                    self.psi[period, s, i]
                    == gp.quicksum(
                        self._demand(period, i, j) * self._p_chain(i) * self.beta[period, i, j]
                        for h, j in data.A_HH
                        if h == i
                    ),
                    name=f"psi_SH_{period}_{s}_{i}",
                )

            for q in data.HLS:
                m.addConstr(
                    gp.quicksum(self.psi[period, u, q] for u, v in data.A if v == q)
                    == gp.quicksum(self.psi[period, q, v] for u, v in data.A if u == q),
                    name=f"flow_balance_{period}_{q}",
                )

            for i, s in data.A_HS:
                m.addConstr(
                    gp.quicksum(self.psi[period, s, u] for v, u in data.A if v == s) <= data.max_flow * self.x[i],
                    name=f"pool_flow_active_{period}_{s}",
                )

            for u, v in data.A_F:
                if not data.route_options.get((u, v)):
                    m.addConstr(self.psi[period, u, v] == 0.0, name=f"no_route_{period}_{u}_{v}")

            if data.E:
                total_period_demand = sum(self._demand(period, i, j) for i, j in data.A_HH)
                if total_period_demand > TOL:
                    m.addConstr(
                        gp.quicksum(self.psi[period, i, j] for i, j in data.A_HH)
                        <= total_period_demand * gp.quicksum(self.x[k] for k in data.E),
                        name=f"served_requires_swap_{period}",
                    )

            for l in data.L:
                energy_lb = gp.quicksum(
                    min_route_energy[l, i] * self.psi[period, l, i]
                    for u, i in data.A_LH
                    if u == l and (l, i) in min_route_energy
                )
                m.addConstr(energy_lb <= 0.0, name=f"energy_L_lb_{period}_{l}")

            for i, s in data.A_HS:
                energy_lb = data.omega * (self.xi[period, i] + self.xi[period, s])
                energy_lb += gp.quicksum(
                    min_route_energy[i, u] * self.psi[period, i, u]
                    for h, u in data.A_HH
                    if h == i and (i, u) in min_route_energy
                )
                energy_lb += gp.quicksum(
                    min_route_energy[s, u] * self.psi[period, s, u]
                    for h, u in data.A
                    if h == s and ((s, u) in data.A_SL or (s, u) in data.A_SS) and (s, u) in min_route_energy
                )
                m.addConstr(energy_lb <= 0.0, name=f"energy_HS_lb_{period}_{i}")

            for i in data.H:
                m.addConstr(
                    self.lambd[period, i] == gp.quicksum(self.psi[period, u, i] for u, v in data.A if v == i),
                    name=f"lambda_{period}_{i}",
                )
                m.addConstr(mu_expr[i] - self.lambd[period, i] >= self.x[i] / data.tau_H, name=f"hub_wait_{period}_{i}")

            for q in q_nodes:
                rho_q = self.rho_l[period, q] if q in data.L else self.rho_s[period, q]
                m.addConstr(q_v1[period, q] == 2.0, name=f"q_v1_{period}_{q}")
                m.addConstr(q_v2[period, q] == self.xi[period, q] + rho_q, name=f"q_v2_{period}_{q}")
                m.addConstr(q_t[period, q] == self.xi[period, q] - rho_q + 2.0, name=f"q_t_{period}_{q}")
                m.addQConstr(
                    q_v1[period, q] * q_v1[period, q] + q_v2[period, q] * q_v2[period, q]
                    <= q_t[period, q] * q_t[period, q],
                    name=f"queue_{period}_{q}",
                )

            for i in data.H:
                m.addConstr(h_v1[period, i] == 2.0 * sqrt_mu_expr[i], name=f"h_v1_{period}_{i}")
                m.addConstr(
                    h_v2[period, i] == self.xi[period, i] + self.lambd[period, i] - mu_expr[i] + 1.0,
                    name=f"h_v2_{period}_{i}",
                )
                m.addConstr(
                    h_t[period, i] == self.xi[period, i] - self.lambd[period, i] + mu_expr[i] + 1.0,
                    name=f"h_t_{period}_{i}",
                )
                m.addQConstr(
                    h_v1[period, i] * h_v1[period, i] + h_v2[period, i] * h_v2[period, i]
                    <= h_t[period, i] * h_t[period, i],
                    name=f"queue_H_{period}_{i}",
                )

            for s in data.S:
                m.addConstr(
                    self.xi[period, s]
                    <= data.tau_S * gp.quicksum(self.psi[period, s, v] for u, v in data.A if u == s),
                    name=f"mean_S_{period}_{s}",
                )

            m.addConstr(
                gp.quicksum(self.xi[period, u] for u in data.HLS)
                + gp.quicksum(
                    min_route_time[u, v] * self.psi[period, u, v]
                    for u, v in data.A_F
                    if (u, v) in min_route_time
                )
                <= self.Gamma,
                name=f"fleet_lb_{period}",
            )

        self.model = m

    def _solution(self) -> PartialMasterSolution:
        data = self.data
        return PartialMasterSolution(
            obj=float(self.model.ObjVal),
            x={u: float(self.x[u].X) for u in data.N_x},
            y={(i, l): float(self.y[i, l].X) for l, i in data.A_LH},
            z={(i, c): float(self.z[i, c].X) for i in data.H for c in data.C},
            Gamma=float(self.Gamma.X),
            theta={period: float(self.theta[period].X) for period in self.periods},
            psi={period: {(u, v): float(self.psi[period, u, v].X) for u, v in data.A} for period in self.periods},
            xi={period: {u: float(self.xi[period, u].X) for u in data.HLS} for period in self.periods},
        )

    def _fixed_cost_value(self, sol: PartialMasterSolution) -> float:
        return self.data.fixed_cost_value(sol.x, sol.z)

    def _logical_revenue_value(self, sol: PartialMasterSolution, period: str) -> float:
        return self.data.revenue * sum(sol.psi[period].get((i, j), 0.0) for i, j in self.data.A_HH)

    def _add_cut(self, period: str, oracle: RouteRecourseSubproblem, duals: Dict[str, Any], name: str) -> None:
        rhs = oracle.dual_expr_master(duals, self.x, self.psi, self.xi, self.Gamma, period)
        self.model.addConstr(self.theta[period] <= -rhs, name=name)

    def solve(self) -> float | None:
        self._build_master()
        oracles = {
            period: RouteRecourseSubproblem(self.data, penalty=self.recourse_penalty, output=False)
            for period in self.periods
        }

        for it in range(1, self.max_iter + 1):
            start = time.perf_counter()
            self.model.optimize()
            if self.model.SolCount == 0:
                self.stop_reason = "master_no_incumbent"
                return None

            sol = self._solution()
            period_q: Dict[str, float] = {}
            period_slack: Dict[str, float] = {}
            theta_gap: Dict[str, float] = {}

            sub_start = time.perf_counter()
            for period, oracle in oracles.items():
                result = oracle.solve(sol.x, sol.psi[period], sol.xi[period], sol.Gamma)
                if result["status"] != GRB.OPTIMAL:
                    self.stop_reason = f"subproblem_status_{result['status']}"
                    return None
                period_q[period] = float(result["obj"])
                period_slack[period] = float(result["slack"])
                theta_gap[period] = sol.theta[period] - (-period_q[period])
                self._add_cut(period, oracle, result["duals"], f"partial_benders_{it}_{period}")
            self.subproblem_time += time.perf_counter() - sub_start

            candidate_obj = (
                sum(
                    self.weights[period] * (self._logical_revenue_value(sol, period) - period_q[period])
                    for period in self.periods
                )
                - self._fixed_cost_value(sol)
                - self.data.gamma_cost * sol.Gamma
            )
            max_theta_gap = max(theta_gap.values())
            max_slack = max(period_slack.values())
            is_recourse_feasible = max_slack <= SLACK_TOL
            if is_recourse_feasible and candidate_obj > self.best_obj:
                self.best_obj = candidate_obj
                self.best_solution = sol

            bound = float(self.model.ObjBound) if self.model.Status in {GRB.OPTIMAL, GRB.TIME_LIMIT} else sol.obj
            gap = math.inf
            if self.best_obj > -math.inf:
                gap = (bound - self.best_obj) / max(1.0, abs(self.best_obj))

            row = {
                "iter": it,
                "master_obj": sol.obj,
                "master_bound": bound,
                "candidate_obj": candidate_obj,
                "best_obj": self.best_obj,
                "gap": gap,
                "max_theta_gap": max_theta_gap,
                "max_slack": max_slack,
                "Gamma": sol.Gamma,
                "subproblem_time": self.subproblem_time,
                "time": time.perf_counter() - start,
            }
            self.history.append(row)
            if self.verbose:
                best_display = self.best_obj if self.best_obj > -math.inf else float("nan")
                gap_display = f"{gap:.2%}" if math.isfinite(gap) else "inf"
                print(
                    "[MP-Partial-Benders] "
                    f"it={it} master={sol.obj:.4f} candidate={candidate_obj:.4f} "
                    f"best={best_display:.4f} gap={gap_display} "
                    f"theta_gap={max_theta_gap:.4g} slack={max_slack:.4g}"
                )

            theta_tol = 1e-4 + 1e-7 * max(1.0, abs(candidate_obj))
            if is_recourse_feasible and max_theta_gap <= theta_tol and gap <= self.gap_tol:
                self.converged = True
                self.stop_reason = "gap_tol"
                break

        if self.stop_reason is None:
            self.stop_reason = "max_iter"
        return self.best_obj if self.best_obj > -math.inf else None


def run_multi_period_partial_benders(
    mp_instance: MultiPeriodInstance,
    *,
    max_iter: int = 50,
    gap_tol: float = 1e-4,
    master_time_limit: float = 30.0,
    master_mip_gap: float = 1e-5,
    recourse_penalty: float = 1e5,
    verbose: bool = True,
) -> MultiPeriodPartialBenders:
    solver = MultiPeriodPartialBenders(
        mp_instance,
        max_iter=max_iter,
        gap_tol=gap_tol,
        master_time_limit=master_time_limit,
        master_mip_gap=master_mip_gap,
        recourse_penalty=recourse_penalty,
        verbose=verbose,
    )
    solver.solve()
    return solver


def run_meituan_multi_period_partial_benders(
    *,
    project_root=None,
    instance_dir=None,
    period_ids=None,
    demand_hours=None,
    period_weights=None,
    **kwargs,
) -> MultiPeriodPartialBenders:
    if project_root is None:
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if instance_dir is None:
        instance_dir = os.path.join(project_root, "data", "instances", "MeituanMultiPeriod")
    periods = _as_period_ids(period_ids=period_ids, demand_hours=demand_hours)
    mp_inst = load_multi_period_instance(instance_dir, periods, period_weights=period_weights)
    return run_multi_period_partial_benders(mp_inst, **kwargs)


if __name__ == "__main__":
    run_meituan_multi_period_partial_benders(demand_hours=[12, 17, 20])
