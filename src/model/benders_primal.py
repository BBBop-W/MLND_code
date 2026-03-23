import math
import time
from typing import Any, Dict, List, Optional, Tuple

import gurobipy as gp
from gurobipy import GRB

from .. import config
from ..data_processor.instance import Instance
from .benders_model import IndexSets

XHat = Dict[str, int]
UHat = Dict[Tuple[str, int], int]
TOL = 1e-7

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
        m.setParam("DualReductions", 0)

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
            config.ORDER_REVENUE_COEF * gp.quicksum(psi[i, j] for i, j in data.A_PD)
            - config.COST_DRONE * gamma_var,
            GRB.MAXIMIZE,
        )

        # -- (C1) assign pickup:  Σ_l y[l,i] ≤ x̂[i]

        Constr_dict = {}

        alpha_1 = {}
        for i in data.P:
            bases = [l for l in data.L if (l, i) in data.A_LP_set]
            if bases:
                alpha_1[i] = m.addConstr(
                    gp.quicksum(y[l, i] for l in bases) <= x_hat[i],
                    name=f"alpha_1_{i}",
                )
        
        Constr_dict["alpha1"] = alpha_1

        # -- (C2) assign base:  y[l,i] ≤ x̂[l]
        alpha_2 = {}
        for l, i in data.A_LP:
            alpha_2[l, i] = m.addConstr(
                y[l, i] <= x_hat[l], name=f"alpha_2_{l}_{i}"
            )
        Constr_dict["alpha2"] = alpha_2

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

        Constr_dict["alpha3"] = alpha_3
        Constr_dict["alpha4"] = alpha_4
        Constr_dict["alpha5"] = alpha_5


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

        Constr_dict["alpha6"] = alpha_6

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
                <= gp.quicksum(data.rho[s] * z[l, i, s] for s in data.S),
                name=f"epsilon_4_{l}_{i}",
            )

        Constr_dict["epsilon1"] = epsilon_1
        Constr_dict["epsilon2"] = epsilon_2
        Constr_dict["epsilon3"] = epsilon_3
        Constr_dict["epsilon4"] = epsilon_4
        
        beta_1 = {}
        # -- (C7) flow balance  (no master var)
        for v in data.N + data.L_v:
            beta_1[v] = m.addConstr(
                data.inflow_expr(psi, v) == data.outflow_expr(psi, v),
                name=f"beta_1_{v}",
            )
        
        Constr_dict["beta1"] = beta_1

        # -- (C8) demand cap  (no master var – h is data)
        beta_2 = {}
        for i, j in data.A_PD:
            beta_2[i, j] = m.addConstr(
                psi[i, j] <= inst.h[i, j], name=f"beta_2_{i}_{j}"
            )
        Constr_dict["beta2"] = beta_2

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
        Constr_dict["beta3"] = beta_3

        pi_1 = {}
        for lv in data.L_v:
            pi_1[lv] = m.addConstr(
                -config.MAX_WAITING_FACTOR_LV * data.outflow_expr(psi, lv)
                <= -q_hat[lv],
                name=f"pi_1_{lv}",
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

        Constr_dict["pi_1"] = pi_1
        Constr_dict["pi2"] = pi_2
        Constr_dict["pi3"] = pi_3

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

        Constr_dict["delta1"] = delta_1
        Constr_dict["delta2"] = delta_2
        Constr_dict["Delta"] = Delta

        # -- (C15) LV energy:  q̂[lv] + travel ≤ E_lim · replenish
        #    Same parametric structure: b_eff = −q̂[lv] when q̂ moved right.
        sigma_1 = {}
        phi_key_set = set(data.phi_keys)
        for d in data.D:
            lv = inst.d_to_lv[d]
            lv_path_keys = [
                (u_n, v_n, k)
                for u_n, v_n in data.lp_arcs_from_base.get(lv, [])
                for k in data.phi_for_arc.get((u_n, v_n), [])
            ]
            lv_travel = data.phi_travel_expr(phi, lv_path_keys)

            lv_replenish = gp.quicksum(
                phi[d, l_hat, "0"]
                for l_hat in data.L_hat
                if (d, l_hat, "0") in phi_key_set
            )
            lv_replenish += gp.quicksum(
                phi[u_n, v_n, k]
                for u_n, v_n, k in lv_path_keys
                if k != "0"
            )

            sigma_1[d] = m.addConstr(
                xi[d] + lv_travel - data.energy_exchange_limit * lv_replenish <= 0,
                name=f"sigma_1_{d}",
            )

        Constr_dict["sigma1"] = sigma_1

        
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
                name=f"sigma_2_{i}",
            )

        fleet_travel = data.phi_travel_expr(phi, data.phi_keys)
        kappa_1 = m.addConstr(
            - gamma_var
            + gp.quicksum(xi[n] for n in data.P + data.D)
            + fleet_travel
            <= - sw_total,
            name="kappa_1",
        )

        # -- (C18) fleet limit:  γ ≤ M · Σ x̂[l]
        fleet_limit_rhs = config.MAX_FLEET_PER_BASE * sum(
            x_hat[l] for l in data.L_hat
        )
        # kappa_3 = m.addConstr(
        #     gamma_var <= fleet_limit_rhs, name="kappa_3"
        # )

        Constr_dict["kappa1"] = kappa_1
        # Constr_dict["kappa3"] = kappa_3

        # ---------- solve ----------
        m.optimize()
        m.write("benders_primal_subproblem.lp")

        if m.Status == GRB.INFEASIBLE:
            print("Primal subproblem is infeasible. Attempting IIS...")
            m.computeIIS()
            m.write("benders_primal_subproblem.ilp")

            return (
                m.Status,
                None,
                {"kind": "feasibility", "x_hat": dict(x_hat), "u_hat": dict(u_hat)},
            )

        if m.Status != GRB.OPTIMAL:
            print(f"Primal subproblem solve ended with status {m.Status}, not optimal.")
            print("=" * 40)
            return m.Status, None, None

        def _safe_pi(constr: gp.Constr) -> float:
            try:
                return float(constr.Pi) if abs(constr.Pi) > TOL else 0.0
            except (gp.GurobiError, AttributeError):
                return 0.0

        obj = m.ObjVal
        duals: Dict[str, Any] = {}
        for name, item in Constr_dict.items():
            if hasattr(item, "items"):
                duals[name] = {key: _safe_pi(c) for key, c in item.items()}
            else:
                duals[name] = _safe_pi(item)

        cut_data = {
            "kind": "optimality",
            "z_star": obj,
            "duals": duals,
            "x_hat": dict(x_hat),
            "u_hat": dict(u_hat),
        }
        self.model = m
        
        return m.Status, obj, cut_data


# ======================================================================
# Full Benders driver
# ======================================================================

# class BendersPrimal:
#     """Benders decomposition with cuts from primal-subproblem duals."""

#     def __init__(
#         self,
#         instance: Instance,
#         max_iter: int = 100,
#         gap_tol: float = 1e-4,
#         time_limit: float = 3600,
#         verbose: bool = True,
#     ):
#         self.inst = instance
#         self.data = IndexSets(instance)
#         self.max_iter = max_iter
#         self.gap_tol = gap_tol
#         self.time_limit = time_limit
#         self.verbose = verbose

#         self.model: Optional[gp.Model] = None
#         self.x: Optional[gp.tupledict] = None
#         self.u: Optional[gp.tupledict] = None
#         self.theta: Optional[gp.Var] = None
#         self.best_obj = -math.inf
#         self.best_sol: Optional[Dict[str, Any]] = None
#         self._visited_master_keys: set = set()

#     def _build_master(self) -> None:
#         data = self.data
#         inst = self.inst

#         mdl = gp.Model("BendersMaster_Primal")
#         mdl.setParam("OutputFlag", 1 if self.verbose else 0)

#         x = mdl.addVars(data.N + data.L_v, vtype=GRB.BINARY, name="x")
#         u = mdl.addVars(data.L, data.S, vtype=GRB.BINARY, name="u")
#         theta = mdl.addVar(lb=0.0, ub=data.total_demand, name="theta")
#         mdl.setObjective(theta, GRB.MAXIMIZE)

#         # service level selection
#         for l in data.L:
#             mdl.addConstr(
#                 gp.quicksum(u[l, s] for s in data.S) == x[l],
#                 name=f"svc_{l}",
#             )

#         # connectivity
#         for n in data.P + data.D:
#             bases = data.nearby_bases.get(n, [])
#             if bases:
#                 mdl.addConstr(
#                     x[n] <= gp.quicksum(x[b] for b in bases),
#                     name=f"near_base_{n}",
#                 )
#             else:
#                 mdl.addConstr(x[n] == 0, name=f"forbid_{n}")

#         for n in data.L_v + data.P:
#             swaps = data.nearby_swaps.get(n, [])
#             if swaps:
#                 mdl.addConstr(
#                     x[n] <= gp.quicksum(x[e] for e in swaps),
#                     name=f"near_swap_{n}",
#                 )
#             else:
#                 mdl.addConstr(x[n] == 0, name=f"forbid_swap_{n}")

#         if config.USE_VIRTUAL_BASES_OPT:
#             for d in data.D:
#                 mdl.addConstr(
#                     x[inst.d_to_lv[d]] <= x[d],
#                     name=f"vbase_{d}",
#                 )

#         fc = gp.quicksum(inst.f[n] * x[n] for n in data.N)
#         mw = data.service_wait_expr(u)
#         mdl.addConstr(fc + config.COST_DRONE * mw <= config.BUDGET, name="budget_lb")
#         mdl.addConstr(
#             mw <= config.MAX_FLEET_PER_BASE * gp.quicksum(x[b] for b in data.L_hat),
#             name="fleet_lb",
#         )
#         mdl.addConstr(
#             theta <= gp.quicksum(data.H[i] * x[i] for i in data.P),
#             name="theta_P",
#         )
#         mdl.addConstr(
#             theta <= gp.quicksum(x[j] for j in data.D) * data.total_demand,
#             name="theta_D",
#         )


#         self.model = mdl
#         self.x = x
#         self.u = u
#         self.theta = theta

#     def get_solution_binary(self, model: gp.Model):
#         x_hat = {
#             n: 1 if model.cbGetSolution(self.x[n]) > 0.5 else 0
#             for n in self.data.N + self.data.L_v
#         }
#         u_hat = {
#             (l, s): 1 if model.cbGetSolution(self.u[l, s]) > 0.5 else 0
#             for l in self.data.L
#             for s in self.data.S
#         }
#         tv = model.cbGetSolution(self.theta)
#         return x_hat, u_hat, tv

#     def _master_solution_key(self, x_hat, u_hat):
#         return (
#             tuple(x_hat[n] for n in self.data.N + self.data.L_v),
#             tuple(u_hat[l, s] for l in self.data.L for s in self.data.S),
#         )

#     def _cut_check(self, cd: dict) -> float:
#         """Evaluate affine cut RHS at (x_hat, u_hat) for consistency check."""
        
#         duals = cd["duals"]
#         data = self.data
#         inst = self.inst
#         XHat = cd["x_hat"]
#         UHat = cd["u_hat"]

#         rhs = 0
        
#         rhs += sum(pi_i * XHat[i] for i, pi_i in duals["alpha_1"].items())

#         rhs += sum(pi_li * XHat[l] for (l, i), pi_li in duals["alpha_2"].items())

#         rhs += sum(pi_lij * XHat[j] for (l, i, j), pi_lij in duals["alpha_4"].items())

#         rhs += sum(pi_lij * (1 - XHat[j]) for (l, i, j), pi_lij in duals["alpha_5"].items())

#         # rhs += sum(pi_lis * data.H.get(i, 0.0) * UHat[l, s] for (l, i, s), pi_lis in duals["epsilon_2"].items())

#         rhs += sum(pi_lis * data.H.get(i, 0.0) * (1 - UHat[l, s]) for (l, i, s), pi_lis in duals["epsilon_3"].items())

#         rhs += sum(pi_ij * inst.h.get((i, j), 0.0) for (i, j), pi_ij in duals["beta_2"].items())

#         rhs -= sum(pi_lv * data.service_wait.get(s, 0.0) * UHat[lv, s] for lv, pi_lv in duals["pi_1"].items() for s in data.S)

#         rhs += sum(pi_n * config.MAX_WAITING_FACTOR_P * XHat[n] for n, pi_n in duals["pi_2"].items() if n in data.P)
#         rhs += sum(pi_n * config.MAX_WAITING_FACTOR_D * XHat[n] for n, pi_n in duals["pi_2"].items() if n in data.D)

#         rhs += sum(pi_k * config.MAX_FLOW_E * XHat[k] for k, pi_k in duals["pi_3"].items())

#         # rhs += sum(2 * pi_delta_1 * math.sqrt(inst.mu[n]) for n, pi_delta_1 in duals["delta_1"].items())
#         # rhs += sum(pi_delta_2 * (1 - inst.mu[n]) for n, pi_delta_2 in duals["delta_2"].items())
#         # rhs += sum(pi_Delta * (inst.mu[n] + 1) for n, pi_Delta in duals["Delta"].items())
        
#         # rhs -= sum(pi_sigma_1 * data.service_wait.get(s, 0.0) * UHat[lv, s] for lv, pi_sigma_1 in duals["sigma_1"].items() for s in data.S)
        
#         rhs -= duals["kappa_1"] * sum(data.service_wait.get(s, 0.0) * UHat[l, s] for l in data.L for s in data.S)
#         rhs += duals["kappa_2"] *(config.BUDGET - sum(inst.f[n]* XHat[n] for n in data.N) )/config.COST_DRONE
#         rhs += duals["kappa_3"] *sum(config.MAX_FLEET_PER_BASE * XHat[l] for l in data.L_hat)

#         return rhs

#     def _add_cut(self, model: gp.Model, cut_data: dict) -> None:
#         """Add optimality cut using coefficients computed from subproblem dual multipliers."""
#         expr = 0
#         duals = cut_data["duals"]
#         data = self.data
#         inst = self.inst

#         expr += gp.quicksum(pi_i * self.x[i] for i, pi_i in duals["alpha_1"].items())

#         expr += gp.quicksum(pi_li * self.x[l] for (l, i), pi_li in duals["alpha_2"].items())

#         expr += gp.quicksum(pi_lij * self.x[j] for (l, i, j), pi_lij in duals["alpha_4"].items())

#         expr += gp.quicksum(pi_lij * (1 - self.x[j]) for (l, i, j), pi_lij in duals["alpha_5"].items())

#         # expr += gp.quicksum(pi_lis * data.H.get(i, 0.0) * self.u[l, s] for (l, i, s), pi_lis in duals["epsilon_2"].items())

#         expr += gp.quicksum(pi_lis * data.H.get(i, 0.0) * (1 - self.u[l, s]) for (l, i, s), pi_lis in duals["epsilon_3"].items())

#         expr += gp.quicksum(pi_ij * inst.h.get((i, j), 0.0) for (i, j), pi_ij in duals["beta_2"].items())

#         expr -= gp.quicksum(pi_lv * data.service_wait.get(s, 0.0) * self.u[lv, s] for lv, pi_lv in duals["pi_1"].items() for s in data.S)

#         expr += gp.quicksum(pi_n * config.MAX_WAITING_FACTOR_P * self.x[n] for n, pi_n in duals["pi_2"].items() if n in data.P)
#         expr += gp.quicksum(pi_n * config.MAX_WAITING_FACTOR_D * self.x[n] for n, pi_n in duals["pi_2"].items() if n in data.D)

#         expr += gp.quicksum(pi_k * config.MAX_FLOW_E * self.x[k] for k, pi_k in duals["pi_3"].items())

#         # expr += gp.quicksum(2 * pi_delta_1 * math.sqrt(inst.mu[n]) for n, pi_delta_1 in duals["delta_1"].items())
#         # expr += gp.quicksum(pi_delta_2 * (1 - inst.mu[n]) for n, pi_delta_2 in duals["delta_2"].items())
#         # expr += gp.quicksum(pi_Delta * (inst.mu[n] + 1) for n, pi_Delta in duals["Delta"].items())
        
#         # expr -= gp.quicksum(pi_sigma_1 * data.service_wait.get(s, 0.0) * self.u[lv, s] for lv, pi_sigma_1 in duals["sigma_1"].items() for s in data.S)
        
#         expr -= duals["kappa_1"] * gp.quicksum(data.service_wait.get(s, 0.0) * self.u[l, s] for l in data.L for s in data.S)
#         expr += duals["kappa_2"] * ( config.BUDGET - gp.quicksum(inst.f[n]* self.x[n] for n in data.N) )/config.COST_DRONE
#         expr += duals["kappa_3"] * gp.quicksum(config.MAX_FLEET_PER_BASE * self.x[l] for l in data.L_hat)


#         model.cbLazy(self.theta <= expr)
    
#     # alpha_2
#     # epsilon_1
#     # kappa_3


#     # def _add_nogood(self, model: gp.Model, cd: dict) -> None:
#     #     """Feasibility cut: exclude the current binary combination."""
#     #     xh = cd["x_hat"]
#     #     ones = [self.x[n] for n in self.data.N + self.data.L_v if xh[n] == 1]
#     #     zeros = [self.x[n] for n in self.data.N + self.data.L_v if xh[n] == 0]
#     #     model.cbLazy(
#     #         gp.quicksum(1 - v for v in ones) + gp.quicksum(zeros) >= 1
#     #     )

#     def solve(self) -> Optional[float]:
#         self._build_master()
#         sub = PrimalSubproblem(self.inst, self.data)
#         t0 = time.time()

#         def callback(model: gp.Model, where: int):
#             if where != GRB.Callback.MIPSOL:
#                 return
#             # if time.time() - t0 > self.time_limit:
#             #     model.terminate()
#             #     return
#             # if len(self._visited_master_keys) >= self.max_iter:
#             #     model.terminate()
#             #     return

#             xh, uh, tv = self.get_solution_binary(model)
#             master_key = self._master_solution_key(xh, uh)
#             if master_key in self._visited_master_keys:
#                 return
#             self._visited_master_keys.add(master_key)

#             status, obj, cd = sub.solve(xh, uh)
#             # for v in sub.model.getVars():
#             #     if v.VarName.startswith("psi["):
#             #         if v.X > 0.00001:
#             #             print(f"Nonzero psi: {v.VarName} = {v.X:.6f}")

#             if obj is not None:
#                 rhs_at_inc = self._cut_check(cd)
#                 tol = 1e-5 * max(1.0, abs(obj))
#                 if abs(rhs_at_inc - obj) > tol:
#                     print(
#                             "Inconsistent cut: "
#                             f"RHS(x_hat,u_hat)={rhs_at_inc:.6f}, obj={obj:.6f}, "
#                             f"diff={rhs_at_inc - obj:.6e}"
#                         )
#                 # else:
#                     # print(
#                     #     f"Cut check passed: RHS(x_hat,u_hat)={rhs_at_inc:.6f}, obj={obj:.6f}"
#                     # )
                    
#             # if cd["kind"] == "feasibility":
#             #     self._add_nogood(model, cd)
#             #     return

#             if obj is not None:
#                 self._add_cut(model, cd)

#             if obj is not None and obj > self.best_obj:
#                 self.best_obj = obj
#                 self.best_sol = {"x": xh, "u": uh, "obj": obj}

#         self.model.setParam("LazyConstraints", 1)
#         self.model.optimize(callback)
#         self.model.write("benders_primal_master.lp")
#         return self.model.ObjVal if self.model.Status == GRB.OPTIMAL else None
