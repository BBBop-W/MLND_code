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

    @staticmethod
    def build_objective_expr(
        data: IndexSets,
        inst: Instance,
        x_ref,
        u_ref,
        q_ref,
        alpha1,
        alpha2,
        alpha4,
        alpha5,
        epsilon2,
        epsilon3,
        beta2,
        pi1,
        pi2,
        pi3,
        delta1,
        delta2,
        Delta,
        sigma1,
        kappa1,
        kappa3,
        use_gurobi_expr: bool,
    ):
        """Build the dual objective expression in numeric or symbolic form."""

        def sum_terms(terms):
            return gp.quicksum(terms) if use_gurobi_expr else sum(terms)

        obj = gp.LinExpr() if use_gurobi_expr else 0.0

        obj += sum_terms(alpha1[i] * x_ref[i] for i in data.P)
        obj += sum_terms(alpha2[l, i] * x_ref[l] for l, i in data.A_LP)

        w_keys = [
            (l, i, j) for l, i in data.A_LP for j in data.D if (i, j) in data.A_PD_set
        ]
        obj += sum_terms(alpha4[l, i, j] * x_ref[j] for l, i, j in w_keys)
        obj += sum_terms(alpha5[l, i, j] * (1 - x_ref[j]) for l, i, j in w_keys)

        z_keys = [(l, i, s) for l, i in data.A_LP for s in data.S]
        obj += sum_terms(
            epsilon2[l, i, s] * data.H.get(i, 0.0) * u_ref[l, s]
            for l, i, s in z_keys
        )
        obj += sum_terms(
            epsilon3[l, i, s] * data.H.get(i, 0.0) * (1 - u_ref[l, s])
            for l, i, s in z_keys
        )

        obj += sum_terms(beta2[i, j] * inst.h[i, j] for i, j in data.A_PD)
        obj += sum_terms(-pi1[lv] * q_ref[lv] for lv in data.L_v)

        for n in data.P + data.D:
            m_n = config.MAX_WAITING_FACTOR_P if n in data.P else config.MAX_WAITING_FACTOR_D
            obj += pi2[n] * m_n * x_ref[n]

        obj += sum_terms(pi3[k] * config.MAX_FLOW_E * x_ref[k] for k in data.E)

        for u in data.P + data.D:
            mu = inst.mu[u]
            obj += 2 * math.sqrt(mu) * delta1[u]
            obj += (1 - mu) * delta2[u]
            obj += (mu + 1) * Delta[u]

        # obj += sum_terms(-q_ref[lv] * sigma1[lv] for lv in data.L_v)
        obj += -kappa1 * sum_terms(q_ref[l] for l in data.L)
        # obj += kappa3 * config.MAX_FLEET_PER_BASE * sum_terms(x_ref[l] for l in data.L_hat)
        obj += -sum_terms(inst.f[u] * x_ref[u] for u in data.N)

        return obj

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
        m.setParam("OutputFlag", 0)
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
        epsilon4 = m.addVars(data.A_LP, lb=0.0, name="epsilon4")  # ε^4

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
        sigma1 = m.addVars(data.D, lb=0.0, name="sigma1")  # σ^1 (indexed by delivery node)
        sigma2 = m.addVars(data.P, lb=0.0, name="sigma2")  # σ^2

        # κ (for fleet sizing and budget)
        kappa1 = m.addVar(lb=0.0, name="kappa1")  # κ^1
        kappa3 = m.addVar(lb=0.0, name="kappa3")  # κ^3

        # ========== Dual objective ==========
        w_keys = [
            (l, i, j) for l, i in data.A_LP for j in data.D if (i, j) in data.A_PD_set
        ]
        z_keys = [(l, i, s) for l, i in data.A_LP for s in data.S]

        obj = self.build_objective_expr(
            data=data,
            inst=inst,
            x_ref=x_hat,
            u_ref=u_hat,
            q_ref=q_hat,
            alpha1=alpha1,
            alpha2=alpha2,
            alpha4=alpha4,
            alpha5=alpha5,
            epsilon2=epsilon2,
            epsilon3=epsilon3,
            beta2=beta2,
            pi1=pi1,
            pi2=pi2,
            pi3=pi3,
            delta1=delta1,
            delta2=delta2,
            Delta=Delta,
            sigma1=sigma1,
            kappa1=kappa1,
            kappa3=kappa3,
            use_gurobi_expr=True,
        )

        m.setObjective(obj, GRB.MINIMIZE)



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
            rhs = config.ORDER_REVENUE_COEF if (u_node, v_node) in data.A_PD_set else 0.0
            m.addConstr(expr >= rhs, name=f"dual_psi_{u_node}_{v_node}")

        # φ_uv^k (complex constraint with energy terms)
        for u_node, v_node, k in data.phi_keys:
            expr = -beta3[u_node, v_node]
            if k in data.E:
                expr += pi3[k]

            travel_time = data.path_time(u_node, v_node, k)

            # σ^1 terms from LV energy constraints indexed by delivery node d.
            for d in data.D:
                lv = inst.d_to_lv[d]
                if u_node == lv and (u_node, v_node) in data.A_LP_set:
                    expr += travel_time * sigma1[d]
                    if k != "0":
                        expr -= data.energy_exchange_limit * sigma1[d]
                if u_node == d and v_node in data.L_hat and k == "0":
                    expr -= data.energy_exchange_limit * sigma1[d]

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
            if u in data.D:
                expr += sigma1[u]
            m.addConstr(expr >= 0, name=f"dual_xi_{u}")

        # Γ: -κ^1 + c·κ^2 + κ^3 ≥ 0
        m.addConstr(
            -kappa1  >= -config.COST_DRONE,
            # -kappa1 + kappa3 >= -config.COST_DRONE,
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
            print("Dual subproblem infeasible. Adding feasibility cut.")
            return m.Status, None, {"kind": "feasibility", "x_hat": dict(x_hat), "u_hat": dict(u_hat)}

        if m.Status == GRB.INF_OR_UNBD:
            print("Dual subproblem unbounded.")

            return m.Status, None, {"kind": "unbounded", "x_hat": dict(x_hat), "u_hat": dict(u_hat)}

        if m.Status != GRB.OPTIMAL:
            print(f"Unexpected dual subproblem status: {m.Status}. Treating as unbounded.")
            return m.Status, None, {"kind": "unbounded", "x_hat": dict(x_hat), "u_hat": dict(u_hat)}



        obj_val = m.ObjVal
        duals = {
            "alpha1": {i: alpha1[i].X for i in data.P},
            "alpha2": {(l, i): alpha2[l, i].X for l, i in data.A_LP},
            "alpha3": {(l, i, j): alpha3[l, i, j].X for l, i, j in w_keys},
            "alpha4": {(l, i, j): alpha4[l, i, j].X for l, i, j in w_keys},
            "alpha5": {(l, i, j): alpha5[l, i, j].X for l, i, j in w_keys},
            "alpha6": {(l, i): alpha6[l, i].X for l, i in data.A_LP},
            "epsilon1": {(l, i, s): epsilon1[l, i, s].X for l, i, s in z_keys},
            "epsilon2": {(l, i, s): epsilon2[l, i, s].X for l, i, s in z_keys},
            "epsilon3": {(l, i, s): epsilon3[l, i, s].X for l, i, s in z_keys},
            "epsilon4": {(l, i): epsilon4[l, i].X for l, i in data.A_LP},
            "beta1": {n: beta1[n].X for n in data.N + data.L_v},
            "beta2": {(i, j): beta2[i, j].X for i, j in data.A_PD},
            "beta3": {(u, v): beta3[u, v].X for u, v in data.A},
            "pi1": {lv: pi1[lv].X for lv in data.L_v},
            "pi2": {n: pi2[n].X for n in data.P + data.D},
            "pi3": {k: pi3[k].X for k in data.E},
            "delta1": {u: delta1[u].X for u in data.P + data.D},
            "delta2": {u: delta2[u].X for u in data.P + data.D},
            "Delta": {u: Delta[u].X for u in data.P + data.D},
            "sigma1": {d: sigma1[d].X for d in data.D},
            "sigma2": {p: sigma2[p].X for p in data.P},
            "kappa1": kappa1.X,
            "kappa3": kappa3.X,
        }
        cut_data = {
            "kind": "optimality",
            "z_star": obj_val,
            "x_hat": dict(x_hat),
            "u_hat": dict(u_hat),
            "duals": duals,
        }
        return m.Status, obj_val, cut_data


