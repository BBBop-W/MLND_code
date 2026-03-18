"""Benders decomposition – explicit dual SOCP subproblem approach.

Solves the full dual SOCP formulation for fixed master decisions (x̂, û).
The dual has second-order cone constraints that mirror the primal's SOCP
queue constraints.

Based on the provided LaTeX formulation, the dual minimization problem is:

    min U = Σ (RHS coefficients) × (x̂, û parameters)
    s.t. dual constraints including SOC: ||δ_u||₂ ≤ Δ_u

The dual optimal value equals the primal optimal value (strong duality for QCP).
"""

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


class DualSubproblem:
    """Explicit SOCP dual of the primal subproblem."""

    def __init__(self, instance: Instance, data: IndexSets):
        self.inst = instance
        self.data = data

    def solve(
        self,
        x_hat: XHat,
        u_hat: UHat,
    ) -> Tuple[int, Optional[float], Optional[Dict[str, Any]]]:
        """Build and solve the dual SOCP, return (status, obj, cut_data)."""
        data = self.data
        inst = self.inst
        q_hat = {loc: data.service_wait_value(u_hat, [loc]) for loc in data.L}
        sw_total = data.service_wait_value(u_hat)

        m = gp.Model("DualSub")
        m.setParam("OutputFlag", 1)
        m.setParam("DualReductions", 0)  # Disable dual reductions for better dual variable stability

        # ========== Dual variables ==========

        # α (for y constraints and w linearization)
        alpha1 = m.addVars(data.P, lb=0.0, name="alpha1")  # α^1_i
        alpha2 = m.addVars(data.A_LP, lb=0.0, name="alpha2")  # α^2_{li}
        alpha3 = m.addVars(
            [(l, i, j) for l, i in data.A_LP for j in data.D if (i, j) in data.A_PD_set],
            lb=0.0,
            name="alpha3",
        )  # α^3
        alpha4 = m.addVars(
            [(l, i, j) for l, i in data.A_LP for j in data.D if (i, j) in data.A_PD_set],
            lb=0.0,
            name="alpha4",
        )  # α^4
        alpha5 = m.addVars(
            [(l, i, j) for l, i in data.A_LP for j in data.D if (i, j) in data.A_PD_set],
            lb=0.0,
            name="alpha5",
        )  # α^5
        alpha6 = m.addVars(data.A_LP, lb=-GRB.INFINITY, name="alpha6")  # α^6

        # ε (for z constraints)
        epsilon1 = m.addVars(
            [(l, i, s) for l, i in data.A_LP for s in data.S],
            lb=0.0,
            name="epsilon1",
        )  # ε^1
        epsilon2 = m.addVars(
            [(l, i, s) for l, i in data.A_LP for s in data.S],
            lb=0.0,
            name="epsilon2",
        )  # ε^2
        epsilon3 = m.addVars(
            [(l, i, s) for l, i in data.A_LP for s in data.S],
            lb=0.0,
            name="epsilon3",
        )  # ε^3
        epsilon4 = m.addVars(data.A_LP, lb=-GRB.INFINITY, name="epsilon4")  # ε^4

        # β (for flow conservation and decomposition)
        beta1 = m.addVars(data.N + data.L_v, lb=-GRB.INFINITY, name="beta1")  # β^1
        beta2 = m.addVars(data.A_PD, lb=0.0, name="beta2")  # β^2
        beta3 = m.addVars(data.A, lb=-GRB.INFINITY, name="beta3")  # β^3

        # π (for LV wait, QoS, swap constraints)
        pi1 = m.addVars(data.L_v, lb=0.0, name="pi1")  # π^1
        pi2 = m.addVars(data.P + data.D, lb=0.0, name="pi2")  # π^2
        pi3 = m.addVars(data.E, lb=0.0, name="pi3")  # π^3

        # δ, Δ (conic duals for SOCP queue constraint)
        # δ_u is a 2-vector for each u ∈ P∪D
        delta1 = m.addVars(data.P + data.D, lb=-GRB.INFINITY, name="delta1")
        delta2 = m.addVars(data.P + data.D, lb=-GRB.INFINITY, name="delta2")
        Delta = m.addVars(
            data.P + data.D, lb=0,  name="Delta"
        )  # Δ_u

        # σ (for energy balance constraints)
        sigma1 = m.addVars(data.L_v, lb=0.0, name="sigma1")  # σ^1
        sigma2 = m.addVars(data.P, lb=0.0, name="sigma2")  # σ^2

        # κ (for fleet sizing and budget)
        kappa1 = m.addVar(lb=0.0, name="kappa1")  # κ^1
        kappa2 = m.addVar(lb=0.0, name="kappa2")  # κ^2
        kappa3 = m.addVar(lb=0.0, name="kappa3")  # κ^3

        # ========== Dual objective ==========
        obj = gp.LinExpr()

        # α^1_i · x̂_i
        obj += gp.quicksum(alpha1[i] * x_hat[i] for i in data.P)

        # α^2_li · x̂_l
        obj += gp.quicksum(alpha2[l, i] * x_hat[l] for l, i in data.A_LP)

        # α^4_lij · x̂_j
        w_keys = [
            (l, i, j) for l, i in data.A_LP for j in data.D if (i, j) in data.A_PD_set
        ]
        obj += gp.quicksum(alpha4[l, i, j] * x_hat[j] for l, i, j in w_keys)

        # α^5_lij · (1 - x̂_j)
        obj += gp.quicksum(alpha5[l, i, j] * (1 - x_hat[j]) for l, i, j in w_keys)

        # ε^2_lis · H_i · û_{ls}
        z_keys = [(l, i, s) for l, i in data.A_LP for s in data.S]
        obj += gp.quicksum(
            epsilon2[l, i, s] * data.H.get(i, 0.0) * u_hat[l, s] for l, i, s in z_keys
        )

        # ε^3_lis · H_i · (1 - û_{ls})
        obj += gp.quicksum(
            epsilon3[l, i, s] * data.H.get(i, 0.0) * (1 - u_hat[l, s])
            for l, i, s in z_keys
        )

        # β^2_ij · h_ij
        obj += gp.quicksum(beta2[i, j] * inst.h[i, j] for i, j in data.A_PD)

        # -π^1_ℓ · q̂_ℓ
        obj += gp.quicksum(-pi1[lv] * q_hat[lv] for lv in data.L_v)

        # π^2_u · M_u · x̂_u
        for n in data.P + data.D:
            M_n = (
                config.MAX_WAITING_FACTOR_P
                if n in data.P
                else config.MAX_WAITING_FACTOR_D
            )
            obj += pi2[n] * M_n * x_hat[n]

        # π^3_k · M_E · x̂_k
        obj += gp.quicksum(
            pi3[k] * config.MAX_FLOW_E * x_hat[k] for k in data.E
        )

        # Queue SOCP contribution: 2√μ_u · δ^1_u + (1-μ_u) · δ^2_u + (μ_u+1) · Δ_u
        for u in data.P + data.D:
            mu = inst.mu[u]
            obj += 2 * math.sqrt(mu) * delta1[u]
            obj += (1 - mu) * delta2[u]
            obj += (mu + 1) * Delta[u]

        # -Σ q̂_ℓ · σ^1_ℓ
        # obj += gp.quicksum(-q_hat[lv] * sigma1[lv] for lv in data.L_v)

        # -κ^1 · Σ q̂_l
        obj += -kappa1 * sum(q_hat[l] for l in data.L)

        # κ^2 · (C - Σ f_u · x̂_u)
        obj += kappa2 * (config.BUDGET - sum(inst.f[u] * x_hat[u] for u in data.N))

        # κ^3 · V · Σ x̂_l
        obj += kappa3 * config.MAX_FLEET_PER_BASE * sum(x_hat[l] for l in data.L_hat)

        m.setObjective(obj, GRB.MINIMIZE)

        # m.setObjective(0, GRB.MINIMIZE)

        # ========== Dual constraints ==========

        # y_li: α^1_i + α^2_{li} + Σ_j (α^5_{lij} - α^3_{lij}) ≥ 0
        for l, i in data.A_LP:
            expr = alpha1[i] + alpha2[l, i]
            expr += gp.quicksum(
                alpha5[l, i, j] - alpha3[l, i, j]
                for j in data.D
                if (i, j) in data.A_PD_set
            )
            m.addConstr(expr >= 0, name=f"dual_y_{l}_{i}")

        # w_lij: α^3 + α^4 - α^5 - h_ij·α^6 ≥ 0
        for l, i, j in w_keys:
            expr = alpha3[l, i, j] + alpha4[l, i, j] - alpha5[l, i, j]
            expr -= inst.h[i, j] * alpha6[l, i]
            m.addConstr(expr >= 0, name=f"dual_w_{l}_{i}_{j}")

        # D_li: α^6 - Σ_s ε^1 + Σ_s ε^3 ≥ 0
        for l, i in data.A_LP:
            expr = alpha6[l, i]
            expr -= gp.quicksum(epsilon1[l, i, s] for s in data.S)
            expr += gp.quicksum(epsilon3[l, i, s] for s in data.S)
            m.addConstr(expr >= 0, name=f"dual_D_{l}_{i}")

        # z_lis: ε^1 + ε^2 - ε^3 - ρ_s·ε^4 ≥ 0
        for l, i, s in z_keys:
            expr = epsilon1[l, i, s] + epsilon2[l, i, s] - epsilon3[l, i, s]
            expr -= data.rho[s] * epsilon4[l, i]
            m.addConstr(expr >= 0, name=f"dual_z_{l}_{i}_{s}")

        # ψ_uv (complex constraint with multiple terms)
        for u_node, v_node in data.A:
            expr = gp.LinExpr()

            # ε^4 if (u,v) ∈ A_LP
            if (u_node, v_node) in data.A_LP_set:
                expr += epsilon4[u_node, v_node]

                # π^1 term if u ∈ L_v
                if u_node in data.L_v:
                    expr += -config.MAX_WAITING_FACTOR_LV * pi1[u_node]

            # Flow conservation β^1
            expr += beta1[u_node] - beta1[v_node]

            # Flow decomposition β^3
            expr += beta3[u_node, v_node]

            # β^2 if (u,v) ∈ A_PD
            if (u_node, v_node) in data.A_PD_set:
                expr += beta2[u_node, v_node]

            # π^2 and conic duals if u ∈ P∪D
            if u_node in data.P + data.D:
                expr += pi2[u_node]
                expr += Delta[u_node] - delta2[u_node]

            # RHS
            rhs = 1.0 if (u_node, v_node) in data.A_PD_set else 0.0
            m.addConstr(expr >= rhs, name=f"dual_psi_{u_node}_{v_node}")

        # φ_uv^k (complex constraint with energy terms)
        for u_node, v_node, k in data.phi_keys:
            expr = -beta3[u_node, v_node]
            if k in data.E:
                expr += pi3[k]

            travel_time = data.path_time(u_node, v_node, k)

            # # σ^1 terms if u ∈ L_v
            if u_node in data.L_v:
                expr += travel_time * sigma1[u_node]
                if k != "0":
                    expr -= (
                        data.energy_exchange_limit
                    ) * sigma1[u_node]

            # # σ^2 terms if u ∈ P
            if u_node in data.P:
                expr += travel_time * sigma2[u_node]
                if k != "0":
                    expr -= (
                        data.energy_exchange_limit
                    ) * sigma2[u_node]

            # κ^1 travel term
            expr += travel_time * kappa1

            m.addConstr(expr >= 0, name=f"dual_phi_{u_node}_{v_node}_{k}")

        # ξ_u: -Δ_u - δ^2_u + [σ^2_u if u∈P] + κ^1 ≥ 0
        for u in data.P + data.D:
            expr = -Delta[u] - delta2[u]
            expr+= kappa1
            if u in data.P:
                expr += sigma2[u]
            m.addConstr(expr >= 0, name=f"dual_xi_{u}")

        # Γ: -κ^1 + c·κ^2 + κ^3 ≥ 0
        m.addConstr(
            -kappa1 + config.COST_DRONE * kappa2 + kappa3 >= 0,
            name="dual_Gamma",
        )

        # ========== SOC constraint in dual ==========
        # ||δ_u||₂ ≤ Δ_u  for each u ∈ P∪D
        for u in data.P + data.D:
            m.addQConstr(
                delta1[u] ** 2 + delta2[u] ** 2 <= Delta[u] ** 2,
                name=f"dual_soc_{u}",
            )

        # ========== Solve ==========
        m.optimize()
        m.write("dual_subproblem.lp")  # For debugging

        if m.Status == GRB.INFEASIBLE:
            return m.Status, None, {"kind": "feasibility", "x_hat": dict(x_hat), "u_hat": dict(u_hat)}

        if m.Status != GRB.OPTIMAL:
            return m.Status, None, None

        obj_val = m.ObjVal
        cut_data = {
            "kind": "optimality",
            "z_star": obj_val,
            "x_hat": dict(x_hat),
            "u_hat": dict(u_hat),
        }
        return m.Status, obj_val, cut_data


# ======================================================================
# Full Benders driver
# ======================================================================

class BendersDual:
    """Benders decomposition using the explicit dual SOCP subproblem."""

    def __init__(
        self,
        instance: Instance,
        max_iter: int = 100,
        gap_tol: float = 1e-6,
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
        self._seen: set = set()

    def _build_master(self) -> None:
        data = self.data
        inst = self.inst

        mdl = gp.Model("BendersMaster_Dual")
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
        mdl.addConstr(
            theta <= gp.quicksum(data.delivery_demand[j] * x[j] for j in data.D),
            name="theta_D",
        )
        mdl.addConstr(
            theta <= data.total_demand * gp.quicksum(x[b] for b in data.L_hat),
            name="theta_L",
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

    def _sig(self, x_hat, u_hat):
        return (
            tuple(x_hat[n] for n in self.data.N + self.data.L_v),
            tuple(u_hat[l, s] for l in self.data.L for s in self.data.S),
        )

    def _add_cut(self, model: gp.Model, cd: dict) -> None:
        """Optimality cut: θ ≤ z* from dual objective."""
        model.cbLazy(self.theta <= cd["z_star"])

    def _add_nogood(self, model: gp.Model, cd: dict) -> None:
        """Feasibility cut: exclude the current binary combination."""
        xh = cd["x_hat"]
        ones = [self.x[n] for n in self.data.N + self.data.L_v if xh[n] == 1]
        zeros = [self.x[n] for n in self.data.N + self.data.L_v if xh[n] == 0]
        model.cbLazy(
            gp.quicksum(1 - v for v in ones) + gp.quicksum(zeros) >= 1
        )

    def solve(self) -> Optional[float]:
        self._build_master()
        sub = DualSubproblem(self.inst, self.data)
        t0 = time.time()

        def callback(model: gp.Model, where: int):
            if where != GRB.Callback.MIPSOL:
                return
            if time.time() - t0 > self.time_limit:
                model.terminate()
                return
            if len(self._seen) >= self.max_iter:
                model.terminate()
                return

            xh, uh, tv = self._cb_solution(model)
            sig = self._sig(xh, uh)
            if sig in self._seen:
                return
            self._seen.add(sig)

            status, obj, cd = sub.solve(xh, uh)
            if cd is None:
                return

            if cd["kind"] == "feasibility":
                self._add_nogood(model, cd)
                return

            if obj is not None and tv > obj + self.gap_tol:
                self._add_cut(model, cd)

            if obj is not None and obj > self.best_obj:
                self.best_obj = obj
                self.best_sol = {"x": xh, "u": uh, "obj": obj}

        self.model.setParam("LazyConstraints", 1)
        self.model.optimize(callback)
        return self.model.ObjVal if self.model.Status == GRB.OPTIMAL else None
