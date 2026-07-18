import math
import os
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import gurobipy as gp
from gurobipy import GRB

# Support direct script execution.
if __package__ in (None, ""):
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)

from src import config
from src.data_processor.instance import Instance
from src.model.backup.benders import (
    RegimeIIData,
    RegimeIIDualObjective,
    RegimeIIDualSubproblem,
    RegimeIIPrimalSubproblem,
)
from src.multi_period.gurobi_model import (
    MultiPeriodInstance,
    _as_period_ids,
    load_multi_period_instance,
)


TOL = 1e-6


@dataclass
class MultiPeriodMasterSolution:
    x: Dict[str, float]
    y: Dict[Tuple[str, str], float]
    z: Dict[Tuple[str, int], float]
    Gamma: float
    theta: Dict[str, float]
    obj: float


def _period_instance(base: Instance, demand: Dict[Tuple[str, str], float]) -> Instance:
    """Create a read-only period view with shared network and period demand."""
    inst = Instance()
    inst.L = list(base.L)
    inst.H = list(base.H)
    inst.E = list(base.E)
    inst.S = list(base.S)
    inst.N = list(base.N)
    inst.h_to_s = dict(base.h_to_s)
    inst.s_to_h = dict(base.s_to_h)

    inst.A = set(base.A)
    inst.A_HH = set(base.A_HH)
    inst.A_LH = set(base.A_LH)
    inst.A_HS = set(base.A_HS)
    inst.A_SH = set(base.A_SH)
    inst.A_SL = set(base.A_SL)
    inst.A_SS = set(base.A_SS)
    inst.A_phi = set(base.A_phi)
    inst.A_F = set(base.A_F)
    inst.E_uv = {arc: list(values) for arc, values in base.E_uv.items()}

    inst.coords = dict(base.coords)
    inst.dist_matrix = dict(base.dist_matrix)
    inst.time_matrix = dict(base.time_matrix)
    inst.mu = dict(base.mu)
    inst.f = dict(base.f)
    inst.h = {arc: float(demand.get(arc, 0.0)) for arc in base.A_HH}
    return inst


class MultiPeriodRegimeIIBenders:
    """Multi-cut Benders for the multi-period Regime II monolithic model."""

    def __init__(
        self,
        mp_instance: MultiPeriodInstance,
        *,
        subproblem: str = "dual",
        use_pareto: bool = True,
        use_valid_inequalities: bool = True,
        use_integer_nogood: bool = True,
        max_iter: int = 30,
        gap_tol: float = 1e-3,
        master_time_limit: float = 30.0,
        master_mip_gap: float = 1e-4,
        verbose: bool = True,
    ):
        if subproblem not in {"primal", "dual"}:
            raise ValueError("subproblem must be 'primal' or 'dual'.")

        self.mp = mp_instance
        self.base = mp_instance.base
        self.periods = list(mp_instance.periods)
        self.weights = dict(mp_instance.weights)
        self.subproblem_mode = subproblem
        self.use_pareto = use_pareto
        self.use_valid_inequalities = use_valid_inequalities
        self.use_integer_nogood = use_integer_nogood
        self.max_iter = int(max_iter)
        self.gap_tol = float(gap_tol)
        self.master_time_limit = float(master_time_limit)
        self.master_mip_gap = float(master_mip_gap)
        self.verbose = verbose

        self.period_instances = {
            period: _period_instance(self.base, self.mp.demand[period])
            for period in self.periods
        }
        self.data_by_period = {
            period: RegimeIIData(inst)
            for period, inst in self.period_instances.items()
        }
        self.data0 = self.data_by_period[self.periods[0]]

        self.model: Optional[gp.Model] = None
        self.x = None
        self.y = None
        self.z = None
        self.Gamma = None
        self.Theta = None
        self.q_open = None

        self.best_obj = -math.inf
        self.best_solution: Optional[MultiPeriodMasterSolution] = None
        self.history: list[Dict[str, Any]] = []
        self.subproblem_time = 0.0
        self.mw_time = 0.0
        self.converged = False
        self.stop_reason: Optional[str] = None

        self.core_x, self.core_y, self.core_z, self.core_gamma = self._initial_core_point()

    def _initial_core_point(self):
        data = self.data0
        core_x = {u: 1.0 for u in data.N_x}
        core_y: Dict[Tuple[str, str], float] = {}
        for i in data.H:
            bases = [l for l, h in data.A_LH if h == i]
            denom = max(1, len(bases))
            for l in bases:
                core_y[i, l] = 1.0 / denom
        for l, i in data.A_LH:
            core_y.setdefault((i, l), 0.0)
        core_z = {(i, c): 1.0 / max(1, len(data.C)) for i in data.H for c in data.C}
        core_gamma = max(1.0, max((data.core_gamma for data in self.data_by_period.values()), default=1.0))
        return core_x, core_y, core_z, core_gamma

    def _fixed_cost_expr(self) -> gp.LinExpr:
        data = self.data0
        fixed = gp.quicksum(float(self.base.f.get(l, config.COST_BASE)) * self.x[l] for l in data.L)
        fixed += gp.quicksum(float(getattr(config, "COST_HUB_FIXED", 0.0)) * self.x[i] for i in data.H)
        fixed += gp.quicksum(float(self.base.f.get(k, config.COST_SWAPPING)) * self.x[k] for k in data.E)
        fixed += gp.quicksum(float(config.COST_HUB[c]) * self.z[i, c] for i in data.H for c in data.C)
        return fixed

    def _fixed_cost_value(self, sol: MultiPeriodMasterSolution) -> float:
        return self.data0.fixed_cost_value(sol.x, sol.z)

    def _build_master(self) -> None:
        data = self.data0
        m = gp.Model("RegimeII_MultiPeriod_BendersMaster")
        m.setParam("OutputFlag", 1 if self.verbose else 0)
        m.setParam("Threads", 1)
        m.setParam("TimeLimit", self.master_time_limit)
        m.setParam("MIPGap", self.master_mip_gap)

        self.x = m.addVars(data.N_x, vtype=GRB.BINARY, name="x")
        self.y = m.addVars([(i, l) for l, i in data.A_LH], vtype=GRB.BINARY, name="y")
        self.z = m.addVars(data.H, data.C, vtype=GRB.BINARY, name="z")
        gamma_ub = sum(self.weights[p] * self.data_by_period[p].total_revenue_ub for p in self.periods)
        if data.gamma_cost > TOL:
            gamma_ub = max(1.0, gamma_ub / data.gamma_cost)
        else:
            gamma_ub = GRB.INFINITY
        self.Gamma = m.addVar(lb=0.0, ub=gamma_ub, vtype=GRB.INTEGER, name="Gamma")
        theta_ub = {
            period: max(1.0, self.data_by_period[period].total_revenue_ub)
            for period in self.periods
        }
        self.Theta = m.addVars(self.periods, lb=-GRB.INFINITY, ub=theta_ub, name="Theta")

        m.setObjective(
            gp.quicksum(self.weights[period] * self.Theta[period] for period in self.periods)
            - self._fixed_cost_expr()
            - self.data0.gamma_cost * self.Gamma,
            GRB.MAXIMIZE,
        )

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
            self._add_valid_inequalities(m)

        self.Gamma.Start = 0.0
        for v in self.x.values():
            v.Start = 0.0
        for v in self.y.values():
            v.Start = 0.0
        for v in self.z.values():
            v.Start = 0.0
        for period in self.periods:
            self.Theta[period].Start = 0.0

        self.model = m

    def _add_valid_inequalities(self, m: gp.Model) -> None:
        data = self.data0
        for i in data.H:
            m.addConstr(
                gp.quicksum(data.mu_levels[c] * self.z[i, c] for c in data.C) >= self.x[i] / data.tau_H,
                name=f"master_hub_wait_feas_{i}",
            )

        if data.E:
            for period in self.periods:
                m.addConstr(
                    self.Theta[period]
                    <= self.data_by_period[period].total_revenue_ub * gp.quicksum(self.x[k] for k in data.E),
                    name=f"theta_requires_swap_open_{period}",
                )

        serviceable_hh = [(i, j) for i, j in data.A_HH if data.route_options.get((i, j))]
        if serviceable_hh:
            self.q_open = m.addVars(serviceable_hh, lb=0.0, ub=1.0, name="q_open_hh")
            for i, j in serviceable_hh:
                hh_swaps = [k for k in data.route_options.get((i, j), []) if k in data.E]
                hh_direct = 1.0 if "0" in data.route_options.get((i, j), []) else 0.0
                m.addConstr(self.q_open[i, j] <= self.x[i], name=f"q_open_origin_{i}_{j}")
                m.addConstr(self.q_open[i, j] <= self.x[j], name=f"q_open_dest_{i}_{j}")
                m.addConstr(
                    self.q_open[i, j] <= hh_direct + gp.quicksum(self.x[k] for k in hh_swaps),
                    name=f"q_open_route_{i}_{j}",
                )

            base_service_keys = [
                (l, i, j)
                for l, i, j in data.alpha_keys
                if (i, j) in serviceable_hh
                and data.p_chain(i) < 1.0 - TOL
                and data.route_options.get((l, i))
            ]
            q_lij = m.addVars(base_service_keys, lb=0.0, ub=1.0, name="q_lij") if base_service_keys else {}
            for l, i, j in base_service_keys:
                lh_swaps = [k for k in data.route_options.get((l, i), []) if k in data.E]
                lh_direct = 1.0 if "0" in data.route_options.get((l, i), []) else 0.0
                m.addConstr(q_lij[l, i, j] <= self.y[i, l], name=f"q_lij_assign_{l}_{i}_{j}")
                m.addConstr(q_lij[l, i, j] <= self.x[j], name=f"q_lij_dest_{l}_{i}_{j}")
                m.addConstr(q_lij[l, i, j] <= self.q_open[i, j], name=f"q_lij_hh_{l}_{i}_{j}")
                m.addConstr(
                    q_lij[l, i, j] <= lh_direct + gp.quicksum(self.x[k] for k in lh_swaps),
                    name=f"q_lij_route_{l}_{i}_{j}",
                )

            for period in self.periods:
                pdata = self.data_by_period[period]
                chain_bound = gp.quicksum(
                    pdata.p_chain(i) * float(pdata.inst.h.get((i, j), 0.0)) * self.q_open[i, j]
                    for i, j in serviceable_hh
                    if pdata.p_chain(i) > TOL
                )
                base_bound = gp.quicksum(
                    (1.0 - pdata.p_chain(i)) * float(pdata.inst.h.get((i, j), 0.0)) * q_lij[l, i, j]
                    for l, i, j in base_service_keys
                )
                m.addConstr(
                    self.Theta[period] <= pdata.revenue * (chain_bound + base_bound),
                    name=f"theta_assignment_demand_bound_{period}",
                )

                service_time_expr = gp.quicksum(
                    pdata.p_chain(i)
                    * float(pdata.inst.h.get((i, j), 0.0))
                    * min(pdata.t[i, j, k] for k in pdata.route_options.get((i, j), []))
                    * self.q_open[i, j]
                    for i, j in serviceable_hh
                    if pdata.p_chain(i) > TOL
                )
                service_time_expr += gp.quicksum(
                    (1.0 - pdata.p_chain(i))
                    * float(pdata.inst.h.get((i, j), 0.0))
                    * (
                        min(pdata.t[l, i, k] for k in pdata.route_options.get((l, i), []))
                        + min(pdata.t[i, j, k] for k in pdata.route_options.get((i, j), []))
                    )
                    * q_lij[l, i, j]
                    for l, i, j in base_service_keys
                )
                m.addConstr(
                    service_time_expr <= self.Gamma,
                    name=f"fleet_assignment_time_bound_{period}",
                )

                min_service_time = min(
                    (
                        pdata.t[i, j, k]
                        for i, j in serviceable_hh
                        for k in pdata.route_options.get((i, j), [])
                    ),
                    default=math.inf,
                )
                if math.isfinite(min_service_time) and min_service_time > TOL:
                    m.addConstr(
                        self.Theta[period] <= (pdata.revenue / min_service_time) * self.Gamma,
                        name=f"theta_fleet_time_bound_{period}",
                    )

        for period in self.periods:
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
            theta={period: float(self.Theta[period].X) for period in self.periods},
            obj=float(self.model.ObjVal),
        )

    def _add_cut(self, period: str, duals: Dict[str, Any], name: str) -> None:
        assert self.model is not None
        data = self.data_by_period[period]
        rhs = RegimeIIDualObjective.expr(data, duals, self.x, self.y, self.z, self.Gamma, True)
        self.model.addConstr(self.Theta[period] <= rhs, name=name)

    def _add_nogood(self, sol: MultiPeriodMasterSolution, name: str) -> None:
        assert self.model is not None
        binary_terms = []

        for u, val in sol.x.items():
            binary_terms.append(1 - self.x[u] if val > 0.5 else self.x[u])
        for key, val in sol.y.items():
            binary_terms.append(1 - self.y[key] if val > 0.5 else self.y[key])
        for key, val in sol.z.items():
            binary_terms.append(1 - self.z[key] if val > 0.5 else self.z[key])

        gamma_value = int(round(sol.Gamma))
        gamma_diff = self.model.addVar(lb=0.0, name=f"{name}_gamma_diff")
        self.model.addConstr(gamma_diff >= self.Gamma - gamma_value, name=f"{name}_gamma_pos")
        self.model.addConstr(gamma_diff >= gamma_value - self.Gamma, name=f"{name}_gamma_neg")
        self.model.addConstr(gp.quicksum(binary_terms) + gamma_diff >= 1.0, name=name)

    def _solve_period(self, period: str, sol: MultiPeriodMasterSolution):
        data = self.data_by_period[period]
        if self.subproblem_mode == "primal":
            solver = RegimeIIPrimalSubproblem(data, output=False)
            status, obj, cut_data = solver.solve(sol.x, sol.y, sol.z, sol.Gamma)
            if status == GRB.OPTIMAL and obj is not None and cut_data is not None:
                return status, float(obj), cut_data
            fallback = RegimeIIDualSubproblem(data, output=False)
            d_status, d_obj, d_cut = fallback.solve(sol.x, sol.y, sol.z, sol.Gamma)
            if d_status == GRB.OPTIMAL and d_obj is not None and d_cut is not None:
                d_cut["source"] = f"primal_status_{status}_fallback_dual"
                return d_status, float(d_obj), d_cut
            return status, obj, cut_data

        solver = RegimeIIDualSubproblem(data, output=False)
        status, obj, cut_data = solver.solve(sol.x, sol.y, sol.z, sol.Gamma)
        if status == GRB.OPTIMAL and obj is not None and cut_data is not None:
            return status, float(obj), cut_data
        fallback = RegimeIIPrimalSubproblem(data, output=False)
        p_status, p_obj, p_cut = fallback.solve(sol.x, sol.y, sol.z, sol.Gamma)
        if p_status == GRB.OPTIMAL and p_obj is not None and p_cut is not None:
            p_cut["source"] = f"dual_status_{status}_fallback_{p_cut.get('source', 'primal')}"
            return p_status, float(p_obj), p_cut
        return status, obj, cut_data

    def _pareto_cut(self, period: str, sol: MultiPeriodMasterSolution, cut_data: Dict[str, Any]):
        data = self.data_by_period[period]
        solver = RegimeIIDualSubproblem(data, output=False)
        return solver.solve_pareto(
            sol.x,
            sol.y,
            sol.z,
            sol.Gamma,
            self.core_x,
            self.core_y,
            self.core_z,
            self.core_gamma,
            cut_data.get("rhs_at_hat", cut_data.get("z_star", 0.0)),
        )

    def _update_core(self, sol: MultiPeriodMasterSolution) -> None:
        for u in self.data0.N_x:
            self.core_x[u] = 0.5 * self.core_x.get(u, 0.0) + 0.5 * sol.x.get(u, 0.0)
        for key in self.core_y:
            self.core_y[key] = 0.5 * self.core_y.get(key, 0.0) + 0.5 * sol.y.get(key, 0.0)
        for key in self.core_z:
            self.core_z[key] = 0.5 * self.core_z.get(key, 0.0) + 0.5 * sol.z.get(key, 0.0)
        self.core_gamma = 0.5 * self.core_gamma + 0.5 * sol.Gamma

    def solve(self) -> Optional[float]:
        self._build_master()
        assert self.model is not None

        for it in range(1, self.max_iter + 1):
            iter_start = time.perf_counter()
            self.model.optimize()
            if self.model.SolCount == 0:
                self.stop_reason = "master_no_incumbent"
                if self.verbose:
                    print("[MP-Benders] master has no incumbent.")
                return None

            sol = self._solution()
            period_objs: Dict[str, float] = {}
            period_theta_gap: Dict[str, float] = {}
            cut_sources: Dict[str, str] = {}
            all_optimal = True

            sub_start = time.perf_counter()
            for period in self.periods:
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

                self._add_cut(period, chosen["duals"], f"mp_benders_{it}_{period}_{chosen.get('source', self.subproblem_mode)}")
                period_objs[period] = float(sub_obj)
                period_theta_gap[period] = sol.theta[period] - float(sub_obj)
                cut_sources[period] = chosen.get("source", self.subproblem_mode)

            self.subproblem_time += time.perf_counter() - sub_start
            if not all_optimal:
                self._add_nogood(sol, f"mp_nogood_{it}")
                continue

            true_obj = (
                sum(self.weights[p] * period_objs[p] for p in self.periods)
                - self._fixed_cost_value(sol)
                - self.data0.gamma_cost * sol.Gamma
            )
            if true_obj > self.best_obj:
                self.best_obj = true_obj
                self.best_solution = sol

            master_bound = float(self.model.ObjBound) if self.model.Status in {GRB.OPTIMAL, GRB.TIME_LIMIT} else sol.obj
            gap = (master_bound - self.best_obj) / max(1.0, abs(self.best_obj))
            max_theta_gap = max(period_theta_gap.values()) if period_theta_gap else math.inf

            row = {
                "iter": it,
                "master_obj": sol.obj,
                "master_bound": master_bound,
                "true_obj": true_obj,
                "best_obj": self.best_obj,
                "gap": gap,
                "max_theta_gap": max_theta_gap,
                "Gamma": sol.Gamma,
                "open_L": sum(1 for l in self.data0.L if sol.x.get(l, 0.0) > 0.5),
                "open_H": sum(1 for h in self.data0.H if sol.x.get(h, 0.0) > 0.5),
                "open_E": sum(1 for e in self.data0.E if sol.x.get(e, 0.0) > 0.5),
                "theta": dict(sol.theta),
                "period_objs": dict(period_objs),
                "cut_sources": dict(cut_sources),
                "time": time.perf_counter() - iter_start,
            }
            self.history.append(row)

            if self.verbose:
                print(
                    "[MP-Benders] "
                    f"it={it} master={sol.obj:.4f} true={true_obj:.4f} "
                    f"best={self.best_obj:.4f} gap={gap:.2%} "
                    f"theta_gap={max_theta_gap:.4g} cuts={cut_sources}"
                )

            self._update_core(sol)
            theta_tol = 1e-4 + 1e-7 * max(1.0, abs(true_obj))
            if self.use_integer_nogood and max_theta_gap > theta_tol:
                self._add_nogood(sol, f"mp_integer_nogood_{it}")
            if max_theta_gap <= theta_tol and gap <= self.gap_tol:
                self.converged = True
                self.stop_reason = "gap_tol"
                break

        if self.stop_reason is None:
            self.stop_reason = "max_iter"
        return self.best_obj if self.best_obj > -math.inf else None


def run_multi_period_benders(
    mp_instance: MultiPeriodInstance,
    *,
    subproblem: str = "dual",
    use_pareto: bool = True,
    use_valid_inequalities: bool = True,
    use_integer_nogood: bool = True,
    max_iter: int = 30,
    gap_tol: float = 1e-3,
    master_time_limit: float = 30.0,
    master_mip_gap: float = 1e-4,
    verbose: bool = True,
) -> MultiPeriodRegimeIIBenders:
    solver = MultiPeriodRegimeIIBenders(
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
    solver.solve()
    return solver


def run_meituan_multi_period_benders(
    *,
    project_root=None,
    instance_dir=None,
    period_ids=None,
    demand_hours=None,
    period_weights=None,
    **kwargs,
) -> MultiPeriodRegimeIIBenders:
    if project_root is None:
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if instance_dir is None:
        instance_dir = os.path.join(project_root, "data", "instances", "MeituanMultiPeriod")
    periods = _as_period_ids(period_ids=period_ids, demand_hours=demand_hours)
    mp_inst = load_multi_period_instance(instance_dir, periods, period_weights=period_weights)
    return run_multi_period_benders(mp_inst, **kwargs)


if __name__ == "__main__":
    run_meituan_multi_period_benders(demand_hours=[12, 17, 20])
