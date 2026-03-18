import math
import time
from typing import Any, Dict, List, Optional, Tuple

import gurobipy as gp
from gurobipy import GRB

from src import config
from src.data_processor.instance import Instance
from src.model.benders_model import IndexSets

XHat = Dict[str, int]
UHat = Dict[Tuple[str, int], int]
TOL = 5e-6

# ======================================================================
# Primal subproblem
# ======================================================================

class PrimalSubproblem:

    def __init__(self, instance: Instance, data: IndexSets):
        self.inst = instance
        self.data = data
        self.model = None  # Store the Gurobi model for debugging/inspection

    def solve(
        self,
        x_hat: XHat,
        u_hat: UHat,
    ) -> Tuple[int, Optional[float], Optional[Dict[str, Any]]]:
        """Build, solve, extract duals, return (status, obj, cut_data)."""
        data = self.data
        inst = self.inst
        q_hat = {loc: data.service_wait_value(u_hat, [loc]) for loc in data.L}
        sw_total = data.service_wait_value(u_hat)

        m = gp.Model("PrimalSub")
        
        m.setParam("OutputFlag", 0)
        m.setParam("QCPDual", 1)

        # ---------- variables ----------
        y = m.addVars(data.A_LP, lb=0.0, ub=1.0, name="y")

        w_keys: List[Tuple[str, str, str]] = [
            (l, i, j)
            for l, i in data.A_LP
            for j in data.D
            if (i, j) in data.A_PD_set
        ]
        w = m.addVars(w_keys, lb=0.0, name="w")
        D_var = m.addVars(data.A_LP, lb=0.0, name="D")

        z_keys = [(l, i, s) for l, i in data.A_LP for s in data.S]
        z = m.addVars(z_keys, lb=0.0, name="z")

        psi = m.addVars(data.A, lb=0.0, name="psi")
        phi = m.addVars(data.phi_keys, lb=0.0, name="phi")

        xi = m.addVars(data.P + data.D, lb=0.0, name="xi")
        lambd = m.addVars(data.P + data.D, lb=0.0, name="lambda")
        gamma_var = m.addVar(lb=0.0, name="gamma")

        # ---------- objective ----------
        m.setObjective(
            gp.quicksum(psi[i, j] for i, j in data.A_PD), GRB.MAXIMIZE
        )

        # -- (C1) assign pickup:  Σ_l y[l,i] ≤ x̂[i]
        alpha_1 = {}
        for i in data.P:
            bases = [l for l in data.L if (l, i) in data.A_LP_set]
            if bases:
                alpha_1[i] = m.addConstr(
                    gp.quicksum(y[l, i] for l in bases) <= x_hat[i],
                    name=f"alpha_1_{i}",
                )

        # -- (C2) assign base:  y[l,i] ≤ x̂[l]
        alpha_2 = {}
        for l, i in data.A_LP:
            alpha_2[l, i] = m.addConstr(
                y[l, i] <= x_hat[l], name=f"alpha_2_{l}_{i}"
            )

        # -- (C3a) w ≤ y  (no master var – skip)
        alpha_3 = {}
        alpha_4 = {}
        alpha_5 = {}
        for l, i, j in w_keys:
            alpha_3[l, i, j] = m.addConstr(w[l, i, j] <= y[l, i], name=f"alpha_3_{l}_{i}_{j}")
        # -- (C3b) w ≤ x̂[j]
            alpha_4[l, i, j] = m.addConstr(w[l, i, j] <= x_hat[j], name=f"alpha_4_{l}_{i}_{j}")
        # -- (C3c) y - w ≤ 1 - x̂[j]
            alpha_5[l, i, j] = m.addConstr(y[l, i] - w[l, i, j] <= 1 - x_hat[j],name=f"alpha_5_{l}_{i}_{j}",)

        # -- (C4) demand aggregation  D = Σ h·w  (equality, no master var)
        alpha_6 = {}
        for l, i in data.A_LP:
            alpha_6[l, i] = m.addConstr(
                D_var[l, i]
                == gp.quicksum(
                    inst.h.get((i, j), 0.0) * w[l, i, j]
                    for j in data.D
                    if (i, j) in data.A_PD_set
                ),
                name=f"alpha_6_{l}_{i}",
            )

        epsilon_1 = {}
        epsilon_2 = {}
        epsilon_3 = {}
        # -- (C5a) z ≤ D  (no master var)
        for l, i, s in z_keys:
            epsilon_1[l, i, s] = m.addConstr( z[l, i, s] <= D_var[l, i],name=f"epsilon_1_{l}_{i}_{s}",)
        # -- (C5b) z ≤ H_i · û[l,s]
            H_i = data.H.get(i, 0.0)
            epsilon_2[l, i, s] = m.addConstr( z[l, i, s] <= H_i * u_hat[l, s], name=f"epsilon_2_{l}_{i}_{s}",)
        # -- (C5c) D - z ≤ H_i · (1 - û[l,s])
            epsilon_3[l, i, s] = m.addConstr(D_var[l, i] - z[l, i, s] <= H_i * (1 - u_hat[l, s]),name=f"epsilon_3_{l}_{i}_{s}",)

        epsilon_4 = {}
        # -- (C6) pickup flow:  psi[l,i] = Σ ρ_s z[l,i,s]  (no master var)
        for l, i in data.A_LP:
            epsilon_4[l, i] = m.addConstr( psi[l, i]
                == gp.quicksum(data.rho[s] * z[l, i, s] for s in data.S),
                name=f"epsilon_4_{l}_{i}",
            )
        
        beta_1 = {}
        # -- (C7) flow balance  (no master var)
        for v in data.N + data.L_v:
            beta_1[v] = m.addConstr(
                data.inflow_expr(psi, v) == data.outflow_expr(psi, v),
                name=f"beta_1_{v}",
            )

        # -- (C8) demand cap  (no master var – h is data)
        beta_2 = {}
        for i, j in data.A_PD:
            beta_2[i, j] = m.addConstr(
                psi[i, j] <= inst.h[i, j], name=f"beta_2_{i}_{j}"
            )

        beta_3 = {}
        # -- (C9) flow decomposition  (no master var)
        for u_n, v_n in data.A:
            beta_3[u_n, v_n] = m.addConstr(
                psi[u_n, v_n]
                == gp.quicksum(
                    phi[u_n, v_n, k]
                    for k in data.phi_for_arc.get((u_n, v_n), [])
                ),
                name=f"beta_3_{u_n}_{v_n}",
            )

        pi_1 = {}
        for n in data.L_v:
            pi_1[n] = m.addConstr(
                -config.MAX_WAITING_FACTOR_LV * data.outflow_expr(psi, n) <= -q_hat[n],
                name=f"pi_1_{n}",
            )

        # -- (C10) QoS:  outflow(n) ≤ M_n · x̂[n]
        pi_2 = {}
        for n in data.P + data.D:
            M_n = (
                config.MAX_WAITING_FACTOR_P
                if n in data.P
                else config.MAX_WAITING_FACTOR_D
            )
            pi_2[n] = m.addConstr(
                data.outflow_expr(psi, n) <= M_n * x_hat[n],
                name=f"pi_2_{n}",
            )

        # -- (C11) swap capacity:  Σ phi_k ≤ M_E · x̂[k]
        pi_3 = {}
        for k in data.E:
            pi_3[k] = m.addConstr(
                gp.quicksum(
                    phi[key] for key in data.swap_flow_keys.get(k, [])
                )
                <= config.MAX_FLOW_E * x_hat[k],
                name=f"pi_3_{k}",
            )

        # -- (C12) lambda definition  (no master var)
        for n in data.P + data.D:
            m.addConstr(
                lambd[n] == data.outflow_expr(psi, n),
                name=f"pi_4_{n}",
            )

        # -- (C13) SOCP queue  (no master var in cone data)

        v1 = m.addVars(data.P + data.D, name="v1", lb=-GRB.INFINITY)
        v2 = m.addVars(data.P + data.D, name="v2", lb=-GRB.INFINITY)
        t  = m.addVars(data.P + data.D, name="t",  lb=0.0)
        delta_1 = {}
        delta_2 = {}
        Delta = {}

        for n in data.P + data.D:
            mu = inst.mu[n]
            # 2. 添加绑定常数项的线性约束 (注意把含有参数 mu 的项放在 RHS)
            delta_1[n] = m.addConstr(v1[n] == 2 * math.sqrt(mu), name=f"eq1_{n}")
            delta_2[n] = m.addConstr(v2[n] - xi[n] - lambd[n] == 1 - mu, name=f"eq2_{n}")
            Delta[n] = m.addConstr(t[n] - xi[n] + lambd[n] == 1 + mu, name=f"eq3_{n}")
            
            # 3. 添加标准二阶锥约束 v1^2 + v2^2 <= t^2
            m.addQConstr(v1[n]*v1[n] + v2[n]*v2[n] <= t[n]*t[n], name=f"c13_soc_{n}")


        # -- (C15) LV energy:  q̂[lv] + travel ≤ E_lim · replenish
        #    Same parametric structure: b_eff = −q̂[lv] when q̂ moved right.
        sigma_1 = {}
        for lv in data.L_v:
            lv_path_keys = [
                (u_n, v_n, k)
                for u_n, v_n in data.lp_arcs_from_base.get(lv, [])
                for k in data.phi_for_arc.get((u_n, v_n), [])
            ]
            lv_travel = data.phi_travel_expr(phi, lv_path_keys)
            lv_replenish = gp.quicksum(
                phi[u_n, v_n, k]
                for u_n, v_n, k in lv_path_keys
                if k != "0"
            )

            sigma_1[lv] = m.addConstr(
                -data.energy_exchange_limit * lv_replenish + lv_travel
                <=  - q_hat[lv],
                name=f"sigma_1_{lv}",
            )

        # -- (C16) pickup energy  (no master var – xi is subproblem var)
        for i in data.P:
            p_keys = [
                (u_n, v_n, k)
                for u_n, v_n in data.pd_arcs_from_pickup.get(i, [])
                for k in data.phi_for_arc.get((u_n, v_n), [])
            ]
            m.addConstr(
                xi[i] + data.phi_travel_expr(phi, p_keys)
                -data.energy_exchange_limit
                * gp.quicksum(
                    phi[u_n, v_n, k]
                    for u_n, v_n, k in p_keys
                    if k != "0"
                ) <= 0,
                name=f"c16_p_energy_{i}",
            )

        fleet_travel = data.phi_travel_expr(phi, data.phi_keys)
        gamma_1 = m.addConstr(
            - gamma_var
            + gp.quicksum(xi[n] for n in data.P + data.D)
            + fleet_travel
            <= - sw_total,
            name="c17_fleet",
        )

        # -- (C18) budget:  C_d · γ ≤ B − Σ f_n · x̂[n]
        budget_rhs = config.BUDGET - sum(
            inst.f[n] * x_hat[n] for n in data.N
        )
        gamma_2 = m.addConstr(
            config.COST_DRONE * gamma_var <= budget_rhs,
            name="gamma_2",
        )

        # -- (C19) fleet limit:  γ ≤ M · Σ x̂[l]
        fleet_limit_rhs = config.MAX_FLEET_PER_BASE * sum(
            x_hat[l] for l in data.L_hat
        )
        gamma_3 = m.addConstr(
            gamma_var <= fleet_limit_rhs, name="gamma_3"
        )

        # ---------- solve ----------
        m.optimize()
        m.write("benders_primal_subproblem.lp")

        if m.Status == GRB.INFEASIBLE:
            return (
                m.Status,
                None,
                {"kind": "feasibility", "x_hat": dict(x_hat), "u_hat": dict(u_hat)},
            )

        if m.Status != GRB.OPTIMAL:
            return m.Status, None, None

        obj = m.ObjVal

        # ---------- build Benders cut coefficients from dual multipliers ----------
        def pi(c):
            """Safe extraction of a linear constraint dual multiplier."""
            try:
                return c.Pi
            except (gp.GurobiError, AttributeError):
                return 0.0

        # Cut coefficients grouped by master variables.
        cut_coef_x: Dict[str, float] = {}
        cut_coef_u: Dict[Tuple[str, int], float] = {}
        const = 0.0  # constant term in the cut (from duals of constraints without master vars)

        def add_coef_x(n, val):
            if abs(val) > 1e-12:
                cut_coef_x[n] = cut_coef_x.get(n, 0.0) + val

        def add_coef_u(key, val):
            if abs(val) > 1e-12:
                cut_coef_u[key] = cut_coef_u.get(key, 0.0) + val

        # C1 contributes to x[i] cut coefficient.
        for i, c in alpha_1.items():
            add_coef_x(i, pi(c))

        # C2 contributes to x[l] cut coefficient.
        for (l, i), c in alpha_2.items():
            add_coef_x(l, pi(c))

        # C3b contributes to x[j] cut coefficient.
        for (l, i, j), c in alpha_4.items():
            add_coef_x(j, pi(c))

        # C3c contributes to x[j] cut coefficient.
        for (l, i, j), c in alpha_5.items():
            add_coef_x(j, -pi(c))
            const += pi(c)

        # C5b contributes to u[l,s] cut coefficient.
        for (l, i, s), c in epsilon_2.items():
            H_i = data.H.get(i, 0.0)
            add_coef_u((l, s), pi(c) * H_i)

        # C5c contributes to u[l,s] cut coefficient.
        for (l, i, s), c in epsilon_3.items():
            H_i = data.H.get(i, 0.0)
            add_coef_u((l, s), -pi(c) * H_i)
            const += pi(c) * H_i
        
        for (i,j), c in beta_2.items():
            const += pi(c) * inst.h.get((i, j), 0.0)

        for lv, c in pi_1.items():
            p = pi(c)
            if abs(p) > 1e-12:
                for s in data.S:
                    sw = data.service_wait.get(s, 0.0)
                    add_coef_u((lv, s), -p * sw)
        
        # C10 contributes to x[n] cut coefficient.
        for n, c in pi_2.items():
            M_n = (
                config.MAX_WAITING_FACTOR_P
                if n in data.P
                else config.MAX_WAITING_FACTOR_D
            )
            add_coef_x(n, pi(c) * M_n)

        # C11 contributes to x[k] cut coefficient.
        for k, c in pi_3.items():
            add_coef_x(k, pi(c) * config.MAX_FLOW_E)

        # LV waiting constraints contribute to u[lv,s] coefficients.

        for n in data.P + data.D:
            delta_1_val = pi(delta_1[n]) if abs(pi(delta_1[n])) > TOL else 0.0
            delta_2_val = pi(delta_2[n]) if abs(pi(delta_2[n])) > TOL else 0.0
            Delta_val = pi(Delta[n]) if abs(pi(Delta[n])) > TOL else 0.0
            mu = inst.mu[n]
            const += 2 * delta_1_val * math.sqrt(mu) + delta_2_val * (1 - mu) + Delta_val * (mu + 1)

        # LV energy constraints contribute to u[lv,s] coefficients.
        # for lv, c in sigma_1.items():
        #     p = pi(c)
        #     if abs(p) > TOL:
        #         for s in data.S:
        #             sw = data.service_wait.get(s, 0.0)
        #             add_coef_u((lv, s), -p * sw)

        # Fleet constraint contributes to all u[l,s] coefficients.
        if abs(pi(gamma_1)) > TOL:
            for l in data.L:
                for s in data.S:
                    sw = data.service_wait.get(s, 0.0)
                    add_coef_u((l, s), -pi(gamma_1) * sw)

        # Budget constraint contributes to x[n] coefficients.
        if abs(pi(gamma_2)) > TOL:
            for n in data.N:
                add_coef_x(n, pi(gamma_2) * (-inst.f[n]))
            const += pi(gamma_2) * config.BUDGET

        # Fleet-limit constraint contributes to x[l] coefficients.
        if abs(pi(gamma_3)) > TOL:
            for l in data.L_hat:
                add_coef_x(l, pi(gamma_3) * config.MAX_FLEET_PER_BASE)

        cut_data = {
            "kind": "optimality",
            "z_star": obj,
            "cut_coef_x": cut_coef_x,
            "cut_coef_u": cut_coef_u,
            "x_hat": dict(x_hat),
            "u_hat": dict(u_hat),
            "const" : const,
        }
        self.model = m
        return m.Status, obj, cut_data


# ======================================================================
# Full Benders driver
# ======================================================================

class BendersPrimal:
    """Benders decomposition with cuts from primal-subproblem duals."""

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

        self.model: Optional[gp.Model] = None
        self.x: Optional[gp.tupledict] = None
        self.u: Optional[gp.tupledict] = None
        self.theta: Optional[gp.Var] = None
        self.best_obj = -math.inf
        self.best_sol: Optional[Dict[str, Any]] = None
        self._visited_master_keys: set = set()

    def _build_master(self) -> None:
        data = self.data
        inst = self.inst

        mdl = gp.Model("BendersMaster_Primal")
        mdl.setParam("OutputFlag", 1 if self.verbose else 0)

        x = mdl.addVars(data.N + data.L_v, vtype=GRB.BINARY, name="x")
        u = mdl.addVars(data.L, data.S, vtype=GRB.BINARY, name="u")
        theta = mdl.addVar(lb=0.0, ub=data.total_demand, name="theta")
        mdl.setObjective(theta, GRB.MAXIMIZE)

        # service level selection
        for l in data.L:
            mdl.addConstr(
                gp.quicksum(u[l, s] for s in data.S) == x[l],
                name=f"svc_{l}",
            )

        # connectivity
        for n in data.P + data.D:
            bases = data.nearby_bases.get(n, [])
            if bases:
                mdl.addConstr(
                    x[n] <= gp.quicksum(x[b] for b in bases),
                    name=f"near_base_{n}",
                )
            else:
                mdl.addConstr(x[n] == 0, name=f"forbid_{n}")

        for n in data.L_v + data.P:
            swaps = data.nearby_swaps.get(n, [])
            if swaps:
                mdl.addConstr(
                    x[n] <= gp.quicksum(x[e] for e in swaps),
                    name=f"near_swap_{n}",
                )
            else:
                mdl.addConstr(x[n] == 0, name=f"forbid_swap_{n}")

        if config.USE_VIRTUAL_BASES_OPT:
            for d in data.D:
                mdl.addConstr(
                    x[inst.d_to_lv[d]] <= x[d],
                    name=f"vbase_{d}",
                )

        fc = gp.quicksum(inst.f[n] * x[n] for n in data.N)
        mw = data.service_wait_expr(u)
        mdl.addConstr(fc + config.COST_DRONE * mw <= config.BUDGET, name="budget_lb")
        mdl.addConstr(
            mw <= config.MAX_FLEET_PER_BASE * gp.quicksum(x[b] for b in data.L_hat),
            name="fleet_lb",
        )
        mdl.addConstr(
            theta <= gp.quicksum(data.H[i] * x[i] for i in data.P),
            name="theta_P",
        )


        self.model = mdl
        self.x = x
        self.u = u
        self.theta = theta

    def _cb_solution(self, model: gp.Model):
        x_hat = {
            n: 1 if model.cbGetSolution(self.x[n]) > 0.5 else 0
            for n in self.data.N + self.data.L_v
        }
        u_hat = {
            (l, s): 1 if model.cbGetSolution(self.u[l, s]) > 0.5 else 0
            for l in self.data.L
            for s in self.data.S
        }
        tv = model.cbGetSolution(self.theta)
        return x_hat, u_hat, tv

    def _master_solution_key(self, x_hat, u_hat):
        return (
            tuple(x_hat[n] for n in self.data.N + self.data.L_v),
            tuple(u_hat[l, s] for l in self.data.L for s in self.data.S),
        )

    def _cut_rhs_at_incumbent(self, cd: dict) -> float:
        """Evaluate affine cut RHS at (x_hat, u_hat) for consistency check."""
        rhs = float(cd["const"])
        for n, coef in cd["cut_coef_x"].items():
            rhs += coef * cd["x_hat"].get(n, 0)
        for (l, s), coef in cd["cut_coef_u"].items():
            rhs += coef * cd["u_hat"].get((l, s), 0)
        return rhs

    def _add_cut(self, model: gp.Model, cd: dict) -> None:
        """Add optimality cut using coefficients computed from subproblem dual multipliers."""
        expr = gp.LinExpr(0)
        for n, c in cd["cut_coef_x"].items():
            if n in self.x:
                expr += c * self.x[n]
        for (l, s), c in cd["cut_coef_u"].items():
            expr += c * self.u[l, s]
        expr += cd["const"]
        model.cbLazy(self.theta <= expr)
        

    # def _add_nogood(self, model: gp.Model, cd: dict) -> None:
    #     """Feasibility cut: exclude the current binary combination."""
    #     xh = cd["x_hat"]
    #     ones = [self.x[n] for n in self.data.N + self.data.L_v if xh[n] == 1]
    #     zeros = [self.x[n] for n in self.data.N + self.data.L_v if xh[n] == 0]
    #     model.cbLazy(
    #         gp.quicksum(1 - v for v in ones) + gp.quicksum(zeros) >= 1
    #     )

    def solve(self) -> Optional[float]:
        self._build_master()
        sub = PrimalSubproblem(self.inst, self.data)
        t0 = time.time()

        def callback(model: gp.Model, where: int):
            if where != GRB.Callback.MIPSOL:
                return
            if time.time() - t0 > self.time_limit:
                model.terminate()
                return
            if len(self._visited_master_keys) >= self.max_iter:
                model.terminate()
                return

            xh, uh, tv = self._cb_solution(model)
            master_key = self._master_solution_key(xh, uh)
            if master_key in self._visited_master_keys:
                return
            self._visited_master_keys.add(master_key)

            status, obj, cd = sub.solve(xh, uh)
            for v in sub.model.getVars():
                if v.VarName.startswith("psi["):
                    if v.X > 0.00001:
                        print(f"Nonzero psi: {v.VarName} = {v.X:.6f}")


            if cd is None:
                return

            if cd["kind"] == "optimality" and obj is not None:
                rhs_at_inc = self._cut_rhs_at_incumbent(cd)
                tol = 1e-5 * max(1.0, abs(obj))
                if abs(rhs_at_inc - obj) > tol:
                    print(
                            "Inconsistent cut: "
                            f"RHS(x_hat,u_hat)={rhs_at_inc:.6f}, obj={obj:.6f}, "
                            f"diff={rhs_at_inc - obj:.6e}"
                        )


            # if cd["kind"] == "feasibility":
            #     self._add_nogood(model, cd)
            #     return

            if obj is not None:
                self._add_cut(model, cd)

            if obj is not None and obj > self.best_obj:
                self.best_obj = obj
                self.best_sol = {"x": xh, "u": uh, "obj": obj}

        self.model.setParam("LazyConstraints", 1)
        self.model.optimize(callback)
        self.model.write("benders_primal_master.lp")
        return self.model.ObjVal if self.model.Status == GRB.OPTIMAL else None
