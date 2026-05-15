"""Scenario-retention Partial Benders for the multi-period Regime II model.

This follows the Transportation Science PBD idea: retain whole scenario
subproblems in the master. Here, periods are scenarios. Retained periods keep
all period-specific Regime II variables and constraints in the master, while
the remaining periods are handled by standard Benders cuts.
"""

import math
import os
import sys
import time
from typing import Any, Dict, Optional, Sequence

import gurobipy as gp
from gurobipy import GRB

if __package__ in (None, ""):
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)

from src import config
from src.model.benders import RegimeIIDualObjective, RegimeIIPrimalSubproblem
from src.multi_period.benders import MultiPeriodMasterSolution, MultiPeriodRegimeIIBenders
from src.multi_period.gurobi_model import (
    MultiPeriodInstance,
    _as_period_ids,
    load_multi_period_instance,
)


TOL = 1e-6


def _period_total_demand(mp_instance: MultiPeriodInstance, period: str) -> float:
    return sum(float(v) for v in mp_instance.demand.get(period, {}).values())


def select_retained_periods(
    mp_instance: MultiPeriodInstance,
    retained_periods: Sequence[str] | None = None,
    retained_count: int = 1,
    rule: str = "largest_demand",
) -> list[str]:
    """Pick periods whose full second-stage blocks are kept in the master."""
    periods = list(mp_instance.periods)
    if retained_periods is not None:
        selected = [str(p) for p in retained_periods]
        missing = [p for p in selected if p not in periods]
        if missing:
            raise ValueError(f"retained_periods not in instance periods: {missing}")
        return selected

    retained_count = max(0, min(int(retained_count), len(periods)))
    if retained_count == 0:
        return []

    if rule == "first":
        return periods[:retained_count]
    if rule == "largest_demand":
        return sorted(periods, key=lambda p: (-_period_total_demand(mp_instance, p), periods.index(p)))[:retained_count]
    raise ValueError("retention rule must be 'largest_demand' or 'first'.")


class MultiPeriodScenarioRetentionBenders(MultiPeriodRegimeIIBenders):
    """Partial Benders with complete retained-period blocks in the master."""

    def __init__(
        self,
        mp_instance: MultiPeriodInstance,
        *,
        retained_periods: Sequence[str] | None = None,
        retained_count: int = 1,
        retention_rule: str = "largest_demand",
        subproblem: str = "dual",
        use_pareto: bool = True,
        use_valid_inequalities: bool = True,
        use_integer_nogood: bool = False,
        max_iter: int = 30,
        gap_tol: float = 1e-3,
        master_time_limit: float = 30.0,
        master_mip_gap: float = 1e-4,
        verbose: bool = True,
    ):
        super().__init__(
            mp_instance,
            subproblem=subproblem,
            use_pareto=use_pareto,
            use_valid_inequalities=use_valid_inequalities,
            use_integer_nogood=use_integer_nogood,
            max_iter=max_iter,
            gap_tol=gap_tol,
            master_time_limit=master_time_limit,
            master_mip_gap=master_mip_gap,
            verbose=verbose,
        )
        self.retained_periods = select_retained_periods(
            mp_instance,
            retained_periods=retained_periods,
            retained_count=retained_count,
            rule=retention_rule,
        )
        retained_set = set(self.retained_periods)
        self.decomposed_periods = [p for p in self.periods if p not in retained_set]
        self.retained_vars: Dict[str, Any] = {}

    def _build_master(self) -> None:
        data = self.data0
        m = gp.Model("RegimeII_MultiPeriod_TSPartialBendersMaster")
        m.setParam("OutputFlag", 1 if self.verbose else 0)
        m.setParam("Threads", 1)
        m.setParam("TimeLimit", self.master_time_limit)
        m.setParam("MIPGap", self.master_mip_gap)

        self.x = m.addVars(data.N_x, vtype=GRB.BINARY, name="x")
        self.y = m.addVars([(i, l) for l, i in data.A_LH], vtype=GRB.BINARY, name="y")
        self.z = m.addVars(data.H, data.C, vtype=GRB.BINARY, name="z")

        gamma_ub = sum(self.weights[p] * self.data_by_period[p].total_revenue_ub for p in self.periods)
        gamma_ub = max(1.0, gamma_ub / data.gamma_cost) if data.gamma_cost > TOL else GRB.INFINITY
        self.Gamma = m.addVar(lb=0.0, ub=gamma_ub, vtype=GRB.INTEGER, name="Gamma")

        theta_ub = {
            period: max(1.0, self.data_by_period[period].total_revenue_ub)
            for period in self.decomposed_periods
        }
        self.Theta = m.addVars(self.decomposed_periods, lb=0.0, ub=theta_ub, name="Theta")

        retained_obj = self._add_retained_period_blocks(m)
        theta_obj = gp.quicksum(self.weights[p] * self.Theta[p] for p in self.decomposed_periods)
        m.setObjective(retained_obj + theta_obj - self._fixed_cost_expr() - data.gamma_cost * self.Gamma, GRB.MAXIMIZE)

        for i in data.H:
            bases = [l for l, h in data.A_LH if h == i]
            if bases:
                m.addConstr(gp.quicksum(self.y[i, l] for l in bases) == self.x[i], name=f"assign_{i}")
            else:
                m.addConstr(self.x[i] == 0, name=f"no_assign_{i}")
            m.addConstr(gp.quicksum(self.z[i, c] for c in data.C) == self.x[i], name=f"capacity_{i}")

        for l, i in data.A_LH:
            m.addConstr(self.y[i, l] <= self.x[l], name=f"assign_active_{i}_{l}")

        if self.use_valid_inequalities:
            self._add_retention_master_valid_inequalities(m)

        self.Gamma.Start = 0.0
        for v in self.x.values():
            v.Start = 0.0
        for v in self.y.values():
            v.Start = 0.0
        for v in self.z.values():
            v.Start = 0.0
        for period in self.decomposed_periods:
            self.Theta[period].Start = 0.0

        self.model = m

    def _add_retained_period_blocks(self, m: gp.Model) -> gp.LinExpr:
        data = self.data0
        periods = self.retained_periods
        if not periods:
            self.retained_vars = {}
            return gp.LinExpr()

        alpha_keys = [(p, l, i, j) for p in periods for l, i, j in data.alpha_keys]
        beta_keys = [(p, i, j) for p in periods for i, j in data.beta_keys]
        psi_keys = [(p, u, v) for p in periods for u, v in data.A]
        varphi_keys = [
            (p, u, v, k)
            for p in periods
            for u, v in data.A_F
            for k in data.route_options.get((u, v), [])
        ]

        rho_l = m.addVars(periods, data.L, lb=0.0, name="ret_rho_l")
        rho_s = m.addVars(periods, data.S, lb=0.0, name="ret_rho_s")
        alpha = m.addVars(alpha_keys, lb=0.0, name="ret_alpha")
        beta = m.addVars(beta_keys, lb=0.0, name="ret_beta")
        psi = m.addVars(psi_keys, lb=0.0, name="ret_psi")
        varphi = m.addVars(varphi_keys, lb=0.0, name="ret_varphi")
        lambd = m.addVars(periods, data.H, lb=0.0, name="ret_lambda")
        xi = m.addVars(periods, data.HLS, lb=0.0, name="ret_xi")

        q_nodes = data.L + data.S
        q_v1 = m.addVars(periods, q_nodes, lb=-GRB.INFINITY, name="ret_q_v1")
        q_v2 = m.addVars(periods, q_nodes, lb=-GRB.INFINITY, name="ret_q_v2")
        q_t = m.addVars(periods, q_nodes, lb=0.0, name="ret_q_t")
        h_v1 = m.addVars(periods, data.H, lb=-GRB.INFINITY, name="ret_h_v1")
        h_v2 = m.addVars(periods, data.H, lb=-GRB.INFINITY, name="ret_h_v2")
        h_t = m.addVars(periods, data.H, lb=0.0, name="ret_h_t")

        self.retained_vars = {
            "rho_l": rho_l,
            "rho_s": rho_s,
            "alpha": alpha,
            "beta": beta,
            "psi": psi,
            "varphi": varphi,
            "lambda": lambd,
            "xi": xi,
            "q_v1": q_v1,
            "q_v2": q_v2,
            "q_t": q_t,
            "h_v1": h_v1,
            "h_v2": h_v2,
            "h_t": h_t,
            "varphi_keys": varphi_keys,
        }

        mu_expr = {
            i: gp.quicksum(data.mu_levels[c] * self.z[i, c] for c in data.C)
            for i in data.H
        }
        sqrt_mu_expr = {
            i: gp.quicksum(math.sqrt(data.mu_levels[c]) * self.z[i, c] for c in data.C)
            for i in data.H
        }

        for period in periods:
            pdata = self.data_by_period[period]
            for l in data.L:
                m.addConstr(rho_l[period, l] <= self.x[l], name=f"ret_rho_L_{period}_{l}")
            for i, s in data.A_HS:
                m.addConstr(rho_s[period, s] <= self.x[i], name=f"ret_rho_S_{period}_{s}")

            for l, i, j in data.alpha_keys:
                m.addConstr(alpha[period, l, i, j] <= rho_l[period, l], name=f"ret_alpha_1_{period}_{l}_{i}_{j}")
                m.addConstr(alpha[period, l, i, j] <= self.y[i, l], name=f"ret_alpha_2_{period}_{l}_{i}_{j}")
                m.addConstr(alpha[period, l, i, j] <= self.x[j], name=f"ret_alpha_3_{period}_{l}_{i}_{j}")
                m.addConstr(
                    alpha[period, l, i, j] >= rho_l[period, l] + self.y[i, l] + self.x[j] - 2,
                    name=f"ret_alpha_4_{period}_{l}_{i}_{j}",
                )

            for s, i, j in data.beta_support:
                m.addConstr(beta[period, i, j] <= rho_s[period, s], name=f"ret_beta_1_{period}_{i}_{j}")
                m.addConstr(beta[period, i, j] <= self.x[j], name=f"ret_beta_2_{period}_{i}_{j}")
                m.addConstr(
                    beta[period, i, j] >= rho_s[period, s] + self.x[j] - 1,
                    name=f"ret_beta_3_{period}_{i}_{j}",
                )

            for i, j in data.A_HH:
                demand = float(pdata.inst.h.get((i, j), 0.0))
                p_i = pdata.p_chain(i)
                m.addConstr(
                    psi[period, i, j]
                    == demand
                    * (
                        (1.0 - p_i)
                        * gp.quicksum(
                            alpha[period, l, i, j]
                            for l, h in data.A_LH
                            if h == i and (period, l, i, j) in alpha
                        )
                        + p_i * beta[period, i, j]
                    ),
                    name=f"ret_psi_HH_{period}_{i}_{j}",
                )

            for l, i in data.A_LH:
                m.addConstr(
                    psi[period, l, i]
                    == gp.quicksum(
                        float(pdata.inst.h.get((i, j), 0.0))
                        * (1.0 - pdata.p_chain(i))
                        * alpha[period, l, i, j]
                        for h, j in data.A_HH
                        if h == i and (period, l, i, j) in alpha
                    ),
                    name=f"ret_psi_LH_{period}_{l}_{i}",
                )

            for s, i in data.A_SH:
                m.addConstr(
                    psi[period, s, i]
                    == gp.quicksum(
                        float(pdata.inst.h.get((i, j), 0.0)) * pdata.p_chain(i) * beta[period, i, j]
                        for h, j in data.A_HH
                        if h == i
                    ),
                    name=f"ret_psi_SH_{period}_{s}_{i}",
                )

            for q in data.HLS:
                m.addConstr(
                    gp.quicksum(psi[period, u, q] for u, v in data.A if v == q)
                    == gp.quicksum(psi[period, q, v] for u, v in data.A if u == q),
                    name=f"ret_flow_balance_{period}_{q}",
                )

            for i, s in data.A_HS:
                m.addConstr(
                    gp.quicksum(psi[period, s, u] for v, u in data.A if v == s) <= data.max_flow * self.x[i],
                    name=f"ret_pool_flow_active_{period}_{s}",
                )

            for u, v in data.A_F:
                m.addConstr(
                    gp.quicksum(varphi[period, u, v, k] for k in data.route_options.get((u, v), []) if (period, u, v, k) in varphi)
                    == psi[period, u, v],
                    name=f"ret_route_{period}_{u}_{v}",
                )

            for k in data.E:
                m.addConstr(
                    gp.quicksum(varphi[period, u, v, kk] for tp, u, v, kk in varphi_keys if tp == period and kk == k)
                    <= data.swap_capacity(k) * self.x[k],
                    name=f"ret_swap_capacity_{period}_{k}",
                )

            total_queue_length = gp.quicksum(xi[period, u] for u in data.HLS)
            total_flight_time = gp.quicksum(
                data.t[u, v, k] * varphi[period, u, v, k]
                for tp, u, v, k in varphi_keys
                if tp == period
            )

            for l in data.L:
                m.addConstr(
                    gp.quicksum(
                        data.route_energy_coef[l, i, k] * varphi[period, l, i, k]
                        for tp, u, i, k in varphi_keys
                        if tp == period and u == l and (l, i) in data.A_LH_set
                    )
                    <= 0.0,
                    name=f"ret_energy_L_{period}_{l}",
                )

            for i in data.H:
                s = data.inst.h_to_s[i]
                m.addConstr(
                    lambd[period, i] == gp.quicksum(psi[period, u, i] for u, v in data.A if v == i),
                    name=f"ret_lambda_{period}_{i}",
                )
                m.addConstr(
                    mu_expr[i] - lambd[period, i] >= self.x[i] / data.tau_H,
                    name=f"ret_hub_wait_{period}_{i}",
                )
                m.addConstr(
                    data.omega * (xi[period, i] + xi[period, s])
                    + gp.quicksum(
                        data.route_energy_coef[i, j, k] * varphi[period, i, j, k]
                        for tp, h, j, k in varphi_keys
                        if tp == period and h == i and (i, j) in data.A_HH_set
                    )
                    + gp.quicksum(
                        data.route_energy_coef[s, v, k] * varphi[period, s, v, k]
                        for tp, u, v, k in varphi_keys
                        if tp == period and u == s and ((s, v) in data.A_SL or (s, v) in data.A_SS)
                    )
                    <= 0.0,
                    name=f"ret_energy_HS_{period}_{i}",
                )

            for q in q_nodes:
                rho_q = rho_l[period, q] if q in data.L else rho_s[period, q]
                m.addConstr(q_v1[period, q] == 2.0, name=f"ret_q_v1_{period}_{q}")
                m.addConstr(q_v2[period, q] == xi[period, q] + rho_q, name=f"ret_q_v2_{period}_{q}")
                m.addConstr(q_t[period, q] == xi[period, q] - rho_q + 2.0, name=f"ret_q_t_{period}_{q}")
                m.addQConstr(
                    q_v1[period, q] * q_v1[period, q] + q_v2[period, q] * q_v2[period, q]
                    <= q_t[period, q] * q_t[period, q],
                    name=f"ret_queue_{period}_{q}",
                )

            for i in data.H:
                m.addConstr(h_v1[period, i] == 2.0 * sqrt_mu_expr[i], name=f"ret_h_v1_{period}_{i}")
                m.addConstr(
                    h_v2[period, i] == xi[period, i] + lambd[period, i] - mu_expr[i] + 1.0,
                    name=f"ret_h_v2_{period}_{i}",
                )
                m.addConstr(
                    h_t[period, i] == xi[period, i] - lambd[period, i] + mu_expr[i] + 1.0,
                    name=f"ret_h_t_{period}_{i}",
                )
                m.addQConstr(
                    h_v1[period, i] * h_v1[period, i] + h_v2[period, i] * h_v2[period, i]
                    <= h_t[period, i] * h_t[period, i],
                    name=f"ret_queue_H_{period}_{i}",
                )

            for s in data.S:
                m.addConstr(
                    xi[period, s] <= data.tau_S * gp.quicksum(psi[period, s, v] for u, v in data.A if u == s),
                    name=f"ret_mean_S_{period}_{s}",
                )

            m.addConstr(total_queue_length + total_flight_time <= self.Gamma, name=f"ret_fleet_{period}")

        retained_revenue = gp.quicksum(
            self.weights[p] * self.data_by_period[p].revenue * psi[p, i, j]
            for p in periods
            for i, j in data.A_HH
        )
        retained_flight_cost = gp.quicksum(
            self.weights[p] * data.flight_cost[u, v, k] * varphi[p, u, v, k]
            for p, u, v, k in varphi_keys
        )
        return retained_revenue - retained_flight_cost

    def _add_retention_master_valid_inequalities(self, m: gp.Model) -> None:
        data = self.data0
        for i in data.H:
            m.addConstr(
                gp.quicksum(data.mu_levels[c] * self.z[i, c] for c in data.C) >= self.x[i] / data.tau_H,
                name=f"master_hub_wait_feas_{i}",
            )

        if data.E:
            for period in self.decomposed_periods:
                m.addConstr(
                    self.Theta[period]
                    <= self.data_by_period[period].total_revenue_ub * gp.quicksum(self.x[k] for k in data.E),
                    name=f"theta_requires_swap_open_{period}",
                )

        for period in self.decomposed_periods:
            pdata = self.data_by_period[period]
            m.addConstr(
                self.Theta[period]
                <= 0.5
                * pdata.revenue
                * gp.quicksum(
                    gp.quicksum(pdata.mu_levels[c] * self.z[i, c] for c in pdata.C) - self.x[i] / pdata.tau_H
                    for i in pdata.H
                ),
                name=f"theta_hub_capacity_bound_{period}",
            )

    def _solution(self) -> MultiPeriodMasterSolution:
        assert self.model is not None
        data = self.data0
        return MultiPeriodMasterSolution(
            x={u: float(self.x[u].X) for u in data.N_x},
            y={(i, l): float(self.y[i, l].X) for l, i in data.A_LH},
            z={(i, c): float(self.z[i, c].X) for i in data.H for c in data.C},
            Gamma=float(self.Gamma.X),
            theta={period: float(self.Theta[period].X) for period in self.decomposed_periods},
            obj=float(self.model.ObjVal),
        )

    def _retained_period_obj_value(self, period: str) -> float:
        if period not in self.retained_periods:
            return 0.0
        data = self.data_by_period[period]
        psi = self.retained_vars["psi"]
        varphi = self.retained_vars["varphi"]
        varphi_keys = self.retained_vars["varphi_keys"]
        revenue = data.revenue * sum(float(psi[period, i, j].X) for i, j in data.A_HH)
        flight_cost = sum(
            data.flight_cost[u, v, k] * float(varphi[period, u, v, k].X)
            for tp, u, v, k in varphi_keys
            if tp == period
        )
        return revenue - flight_cost

    def _add_cut(self, period: str, duals: Dict[str, Any], name: str) -> None:
        assert self.model is not None
        data = self.data_by_period[period]
        rhs = RegimeIIDualObjective.expr(data, duals, self.x, self.y, self.z, self.Gamma, True)
        self.model.addConstr(self.Theta[period] <= rhs, name=name)

    def solve(self) -> Optional[float]:
        self._build_master()
        assert self.model is not None

        for it in range(1, self.max_iter + 1):
            iter_start = time.perf_counter()
            self.model.optimize()
            if self.model.SolCount == 0:
                self.stop_reason = "master_no_incumbent"
                if self.verbose:
                    print("[MP-TS-PBD] master has no incumbent.")
                return None

            sol = self._solution()
            period_objs: Dict[str, float] = {}
            period_theta_gap: Dict[str, float] = {}
            cut_sources: Dict[str, str] = {}
            all_optimal = True

            sub_start = time.perf_counter()
            for period in self.retained_periods:
                solver = RegimeIIPrimalSubproblem(self.data_by_period[period], output=False)
                status, retained_obj, _ = solver.solve(sol.x, sol.y, sol.z, sol.Gamma)
                if status != GRB.OPTIMAL or retained_obj is None:
                    all_optimal = False
                    cut_sources[period] = f"retained_reprice_status_{status}"
                    continue
                period_objs[period] = float(retained_obj)
                cut_sources[period] = "retained_repriced"

            for period in self.decomposed_periods:
                status, sub_obj, cut_data = self._solve_period(period, sol)
                if status != GRB.OPTIMAL or sub_obj is None or cut_data is None:
                    all_optimal = False
                    cut_sources[period] = f"status_{status}"
                    continue

                chosen = cut_data
                if self.use_pareto:
                    mw_start = time.perf_counter()
                    mw_status, _, mw_cut = self._pareto_cut(period, sol, cut_data)
                    self.mw_time += time.perf_counter() - mw_start
                    if mw_status == GRB.OPTIMAL and mw_cut is not None:
                        chosen = mw_cut

                self._add_cut(period, chosen["duals"], f"ts_pbd_{it}_{period}_{chosen.get('source', self.subproblem_mode)}")
                period_objs[period] = float(sub_obj)
                period_theta_gap[period] = sol.theta[period] - float(sub_obj)
                cut_sources[period] = chosen.get("source", self.subproblem_mode)

            self.subproblem_time += time.perf_counter() - sub_start
            if not all_optimal:
                if self.use_integer_nogood:
                    self._add_nogood(sol, f"ts_pbd_nogood_{it}")
                continue

            true_obj = float(
                sum(self.weights[p] * period_objs[p] for p in self.periods)
                - self._fixed_cost_value(sol)
                - self.data0.gamma_cost * sol.Gamma
            )
            if true_obj > self.best_obj:
                self.best_obj = true_obj
                self.best_solution = sol

            master_bound = float(self.model.ObjBound) if self.model.Status in {GRB.OPTIMAL, GRB.TIME_LIMIT} else sol.obj
            gap = (master_bound - self.best_obj) / max(1.0, abs(self.best_obj))
            max_theta_gap = max(period_theta_gap.values()) if period_theta_gap else 0.0

            row = {
                "iter": it,
                "master_obj": sol.obj,
                "master_bound": master_bound,
                "true_obj": true_obj,
                "best_obj": self.best_obj,
                "gap": gap,
                "max_theta_gap": max_theta_gap,
                "Gamma": sol.Gamma,
                "retained_periods": list(self.retained_periods),
                "decomposed_periods": list(self.decomposed_periods),
                "period_objs": dict(period_objs),
                "cut_sources": dict(cut_sources),
                "time": time.perf_counter() - iter_start,
            }
            self.history.append(row)

            if self.verbose:
                print(
                    "[MP-TS-PBD] "
                    f"it={it} master={sol.obj:.4f} true={true_obj:.4f} "
                    f"best={self.best_obj:.4f} gap={gap:.2%} "
                    f"theta_gap={max_theta_gap:.4g} retained={self.retained_periods} cuts={cut_sources}"
                )

            self._update_core(sol)
            theta_tol = 1e-4 + 1e-7 * max(1.0, abs(true_obj))
            if self.use_integer_nogood and max_theta_gap > theta_tol:
                self._add_nogood(sol, f"ts_pbd_integer_nogood_{it}")
            if max_theta_gap <= theta_tol and gap <= self.gap_tol:
                self.converged = True
                self.stop_reason = "gap_tol"
                break

        if self.stop_reason is None:
            self.stop_reason = "max_iter"
        return self.best_obj if self.best_obj > -math.inf else None


def run_multi_period_ts_partial_benders(
    mp_instance: MultiPeriodInstance,
    *,
    retained_periods: Sequence[str] | None = None,
    retained_count: int = 1,
    retention_rule: str = "largest_demand",
    subproblem: str = "dual",
    use_pareto: bool = True,
    use_valid_inequalities: bool = True,
    use_integer_nogood: bool = False,
    max_iter: int = 30,
    gap_tol: float = 1e-3,
    master_time_limit: float = 30.0,
    master_mip_gap: float = 1e-4,
    verbose: bool = True,
) -> MultiPeriodScenarioRetentionBenders:
    solver = MultiPeriodScenarioRetentionBenders(
        mp_instance,
        retained_periods=retained_periods,
        retained_count=retained_count,
        retention_rule=retention_rule,
        subproblem=subproblem,
        use_pareto=use_pareto,
        use_valid_inequalities=use_valid_inequalities,
        use_integer_nogood=use_integer_nogood,
        max_iter=max_iter,
        gap_tol=gap_tol,
        master_time_limit=master_time_limit,
        master_mip_gap=master_mip_gap,
        verbose=verbose,
    )
    solver.solve()
    return solver


def run_meituan_multi_period_ts_partial_benders(
    *,
    project_root=None,
    instance_dir=None,
    period_ids=None,
    demand_hours=None,
    period_weights=None,
    retained_periods: Sequence[str] | None = None,
    retained_count: int = 1,
    retention_rule: str = "largest_demand",
    **kwargs,
) -> MultiPeriodScenarioRetentionBenders:
    if project_root is None:
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if instance_dir is None:
        instance_dir = os.path.join(project_root, "data", "instances", "MeituanMultiPeriod")
    periods = _as_period_ids(period_ids=period_ids, demand_hours=demand_hours)
    mp_inst = load_multi_period_instance(instance_dir, periods, period_weights=period_weights)
    return run_multi_period_ts_partial_benders(
        mp_inst,
        retained_periods=retained_periods,
        retained_count=retained_count,
        retention_rule=retention_rule,
        **kwargs,
    )


if __name__ == "__main__":
    run_meituan_multi_period_ts_partial_benders(demand_hours=[12, 17, 20])
