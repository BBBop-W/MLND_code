import math
import time
from typing import Any, Dict, List, Optional, Tuple

import gurobipy as gp
from gurobipy import GRB

from .. import config
from ..data_processor.instance import Instance
from .benders_main import IndexSets

XHat = Dict[str, int]
EtaHat = Dict[Tuple[str, int], int]
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
        eta_hat: EtaHat,
    ) -> Tuple[int, Optional[float], Optional[Dict[str, Any]]]:
        """Build, solve, extract duals, return (status, obj, cut_data)."""
        data = self.data
        inst = self.inst
        q_hat = {loc: data.service_wait_value(eta_hat, [loc]) for loc in data.L_hat}
        sw_total = data.service_wait_value(eta_hat)

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
            bases = [l for l in data.L_hat if (l, i) in data.A_LP_set]
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
            epsilon_2[l, i, s] = m.addConstr( z[l, i, s] <= H_i * eta_hat[l, s], name=f"epsilon_2_{l}_{i}_{s}",)
        # -- (C5c) D - z ≤ H_i · (1 - û[l,s])
            epsilon_3[l, i, s] = m.addConstr(D_var[l, i] - z[l, i, s] <= H_i * (1 - eta_hat[l, s]),name=f"epsilon_3_{l}_{i}_{s}",)

        epsilon_4 = {}
        # -- (C6) pickup flow:  psi[l,i] = Σ ρ_s z[l,i,s]  (no master var)
        for l, i in data.A_LP:
            epsilon_4[l, i] = m.addConstr( psi[l, i]
                == gp.quicksum(data.rho[s] * z[l, i, s] for s in data.S),
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
        # sigma_1 = {}
        # phi_key_set = set(data.phi_keys)
        # for d in data.D:
        #     lv = inst.d_to_lv[d]
        #     lv_path_keys = [
        #         (u_n, v_n, k)
        #         for u_n, v_n in data.lp_arcs_from_base.get(lv, [])
        #         for k in data.phi_for_arc.get((u_n, v_n), [])
        #     ]
        #     lv_travel = data.phi_travel_expr(phi, lv_path_keys)

        #     lv_replenish = gp.quicksum(
        #         phi[d, l_hat, "0"]
        #         for l_hat in data.L
        #         if (d, l_hat, "0") in phi_key_set
        #     )
        #     lv_replenish += gp.quicksum(
        #         phi[u_n, v_n, k]
        #         for u_n, v_n, k in lv_path_keys
        #         if k != "0"
        #     )

        #     sigma_1[d] = m.addConstr(
        #         xi[d] + lv_travel - data.energy_exchange_limit * lv_replenish <= 0,
        #         name=f"sigma_1_{d}",
        #     )

        # Constr_dict["sigma1"] = sigma_1

        
        # for i in data.P:
        #     p_keys = [
        #         (u_n, v_n, k)
        #         for u_n, v_n in data.pd_arcs_from_pickup.get(i, [])
        #         for k in data.phi_for_arc.get((u_n, v_n), [])
        #     ]
        #     m.addConstr(
        #         xi[i] + data.phi_travel_expr(phi, p_keys)
        #         -data.energy_exchange_limit
        #         * gp.quicksum(
        #             phi[u_n, v_n, k]
        #             for u_n, v_n, k in p_keys
        #             if k != "0"
        #         ) <= 0,
        #         name=f"sigma_2_{i}",
        #     )

        fleet_travel = data.phi_travel_expr(phi, data.phi_keys)
        kappa_1 = m.addConstr(
            - gamma_var
            + gp.quicksum(xi[n] for n in data.P + data.D)
            + fleet_travel
            <= - sw_total,
            name="kappa_1",
        )

        # -- (C18) fleet limit:  γ ≤ M · Σ x̂[l]
        # fleet_limit_rhs = config.MAX_FLEET_PER_BASE * sum(
        #     x_hat[l] for l in data.L
        # )
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
                {"kind": "feasibility", "x_hat": dict(x_hat), "eta_hat": dict(eta_hat)},
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
            "eta_hat": dict(eta_hat),
        }
        self.model = m
        
        return m.Status, obj, cut_data

