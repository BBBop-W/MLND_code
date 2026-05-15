"""Benders-dual-decomposition cut helpers for Regime II.

The helpers here implement the C.3-style strengthening used by the callback
TS-PBD solver: fix the dual multipliers from a Benders cut, relax the linking
constraints to local first-stage copies, and solve the resulting SOCP to get a
valid intercept for a Benders-dual cut.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import gurobipy as gp
from gurobipy import GRB

from src.model.benders import RegimeIIData, RegimeIIDualObjective


TOL = 1e-7


def _add_to(target: Dict[Any, float], key: Any, value: float) -> None:
    target[key] = target.get(key, 0.0) + float(value)


def benders_affine_coefficients(data: RegimeIIData, duals: Dict[str, Any]) -> Dict[str, Any]:
    """Return ``constant + beta * master_vars`` for a dual Benders RHS."""
    coeff_x = {u: 0.0 for u in data.N_x}
    coeff_y = {(i, l): 0.0 for l, i in data.A_LH}
    coeff_z = {(i, c): 0.0 for i in data.H for c in data.C}
    constant = 0.0

    for l in data.L:
        _add_to(coeff_x, l, duals["rhoL"][l])
    for i, s in data.A_HS:
        _add_to(coeff_x, i, duals["rhoS"][i, s])

    for l, i, j in data.alpha_keys:
        a2 = float(duals["alpha2"][l, i, j])
        a3 = float(duals["alpha3"][l, i, j])
        a4 = float(duals["alpha4"][l, i, j])
        _add_to(coeff_y, (i, l), a2 - a4)
        _add_to(coeff_x, j, a3 - a4)
        constant += 2.0 * a4

    for s, i, j in data.beta_support:
        b2 = float(duals["beta2"][s, i, j])
        b3 = float(duals["beta3"][s, i, j])
        _add_to(coeff_x, j, b2 - b3)
        constant += b3

    for i, s in data.A_HS:
        _add_to(coeff_x, i, float(duals["pool"][i, s]) * data.max_flow)
    for k in data.E:
        _add_to(coeff_x, k, float(duals["swap"][k]) * data.swap_capacity(k))

    for i in data.H:
        hub_wait = float(duals["hubWait"][i])
        h1 = float(duals["h1"][i])
        h2 = float(duals["h2"][i])
        h3 = float(duals["h3"][i])
        _add_to(coeff_x, i, -hub_wait / data.tau_H)
        constant += h2 + h3
        for c in data.C:
            mu = data.mu_levels[c]
            _add_to(coeff_z, (i, c), hub_wait * mu)
            _add_to(coeff_z, (i, c), h1 * 2.0 * math.sqrt(mu))
            _add_to(coeff_z, (i, c), -h2 * mu)
            _add_to(coeff_z, (i, c), h3 * mu)

    for q in data.L + data.S:
        constant += 2.0 * float(duals["q1"][q]) + 2.0 * float(duals["q3"][q])

    return {
        "constant": constant,
        "x": coeff_x,
        "y": coeff_y,
        "z": coeff_z,
        "Gamma": float(duals["fleet"]),
    }


def bdd_rhs_expr(coeffs: Dict[str, Any], x_ref: Any, y_ref: Any, z_ref: Any, gamma_ref: Any, use_gurobi_expr: bool) -> Any:
    """Evaluate/build ``intercept + beta * master_vars``."""
    expr = gp.LinExpr(float(coeffs["constant"])) if use_gurobi_expr else float(coeffs["constant"])
    add = gp.quicksum if use_gurobi_expr else sum
    expr += add(coeff * x_ref[u] for u, coeff in coeffs["x"].items())
    expr += add(coeff * y_ref[key] for key, coeff in coeffs["y"].items())
    expr += add(coeff * z_ref[key] for key, coeff in coeffs["z"].items())
    expr += float(coeffs["Gamma"]) * gamma_ref
    return expr


class RegimeIIBDDStrengthener:
    """Solve the Lagrangian local-copy SOCP for a strengthened BDD cut."""

    def __init__(self, data: RegimeIIData, gamma_ub: float, output: bool = False):
        self.data = data
        self.gamma_ub = max(1.0, float(gamma_ub))
        self.output = output

    def solve(
        self,
        duals: Dict[str, Any],
        x_ref: Dict[str, float],
        y_ref: Dict[Tuple[str, str], float],
        z_ref: Dict[Tuple[str, int], float],
        gamma_ref: float,
    ) -> Tuple[int, Optional[Dict[str, Any]]]:
        data = self.data
        coeffs = benders_affine_coefficients(data, duals)
        standard_rhs = RegimeIIDualObjective.expr(data, duals, x_ref, y_ref, z_ref, gamma_ref, False)

        m = gp.Model("RegimeII_BDD_Strengthening")
        m.setParam("OutputFlag", 1 if self.output else 0)
        m.setParam("Threads", 1)
        m.setParam("DualReductions", 0)
        m.setParam("BarHomogeneous", 1)

        x_loc = m.addVars(data.N_x, lb=0.0, ub=1.0, name="bdd_x")
        y_loc = m.addVars([(i, l) for l, i in data.A_LH], lb=0.0, ub=1.0, name="bdd_y")
        z_loc = m.addVars(data.H, data.C, lb=0.0, ub=1.0, name="bdd_z")
        gamma_loc = m.addVar(lb=0.0, ub=self.gamma_ub, name="bdd_Gamma")

        for i in data.H:
            bases = [l for l, h in data.A_LH if h == i]
            if bases:
                m.addConstr(gp.quicksum(y_loc[i, l] for l in bases) == x_loc[i], name=f"bdd_assign_{i}")
            else:
                m.addConstr(x_loc[i] == 0.0, name=f"bdd_no_assign_{i}")
            m.addConstr(gp.quicksum(z_loc[i, c] for c in data.C) == x_loc[i], name=f"bdd_capacity_{i}")
        for l, i in data.A_LH:
            m.addConstr(y_loc[i, l] <= x_loc[l], name=f"bdd_assign_active_{i}_{l}")

        rho_l = m.addVars(data.L, lb=0.0, name="rho_l")
        rho_s = m.addVars(data.S, lb=0.0, name="rho_s")
        alpha = m.addVars(data.alpha_keys, lb=0.0, name="alpha")
        beta = m.addVars(data.beta_keys, lb=0.0, name="beta")
        psi = m.addVars(data.A, lb=0.0, name="psi")
        varphi = m.addVars(data.varphi_keys, lb=0.0, name="varphi")
        lambd = m.addVars(data.H, lb=0.0, name="lambda")
        xi = m.addVars(data.HLS, lb=0.0, name="xi")

        recourse_obj = data.revenue * gp.quicksum(psi[i, j] for i, j in data.A_HH)
        recourse_obj -= gp.quicksum(data.flight_cost[u, v, k] * varphi[u, v, k] for u, v, k in data.varphi_keys)
        lagrangian_penalty = gp.quicksum(coeffs["x"][u] * x_loc[u] for u in data.N_x)
        lagrangian_penalty += gp.quicksum(coeffs["y"][i, l] * y_loc[i, l] for l, i in data.A_LH)
        lagrangian_penalty += gp.quicksum(coeffs["z"][i, c] * z_loc[i, c] for i in data.H for c in data.C)
        lagrangian_penalty += coeffs["Gamma"] * gamma_loc
        m.setObjective(recourse_obj - lagrangian_penalty, GRB.MAXIMIZE)

        for l in data.L:
            m.addConstr(rho_l[l] <= x_loc[l], name=f"rhoL_{l}")
        for i, s in data.A_HS:
            m.addConstr(rho_s[s] <= x_loc[i], name=f"rhoS_{i}_{s}")

        for l, i, j in data.alpha_keys:
            m.addConstr(alpha[l, i, j] <= rho_l[l], name=f"alpha1_{l}_{i}_{j}")
            m.addConstr(alpha[l, i, j] <= y_loc[i, l], name=f"alpha2_{l}_{i}_{j}")
            m.addConstr(alpha[l, i, j] <= x_loc[j], name=f"alpha3_{l}_{i}_{j}")
            m.addConstr(alpha[l, i, j] >= rho_l[l] + y_loc[i, l] + x_loc[j] - 2.0, name=f"alpha4_{l}_{i}_{j}")

        for s, i, j in data.beta_support:
            m.addConstr(beta[i, j] <= rho_s[s], name=f"beta1_{s}_{i}_{j}")
            m.addConstr(beta[i, j] <= x_loc[j], name=f"beta2_{s}_{i}_{j}")
            m.addConstr(beta[i, j] >= rho_s[s] + x_loc[j] - 1.0, name=f"beta3_{s}_{i}_{j}")

        for i, j in data.A_HH:
            p_i = data.p_chain(i)
            demand = float(data.inst.h.get((i, j), 0.0))
            m.addConstr(
                psi[i, j]
                == demand
                * (
                    (1.0 - p_i) * gp.quicksum(alpha[l, i, j] for l, h in data.A_LH if h == i and (l, i, j) in alpha)
                    + p_i * beta[i, j]
                ),
                name=f"psiHH_{i}_{j}",
            )

        for l, i in data.A_LH:
            m.addConstr(
                psi[l, i]
                == gp.quicksum(
                    float(data.inst.h.get((i, j), 0.0)) * (1.0 - data.p_chain(i)) * alpha[l, i, j]
                    for h, j in data.A_HH
                    if h == i and (l, i, j) in alpha
                ),
                name=f"psiLH_{l}_{i}",
            )

        for s, i in data.A_SH:
            m.addConstr(
                psi[s, i]
                == gp.quicksum(
                    float(data.inst.h.get((i, j), 0.0)) * data.p_chain(i) * beta[i, j]
                    for h, j in data.A_HH
                    if h == i
                ),
                name=f"psiSH_{s}_{i}",
            )

        for q in data.HLS:
            m.addConstr(
                gp.quicksum(psi[u, q] for u, v in data.A if v == q)
                == gp.quicksum(psi[q, v] for u, v in data.A if u == q),
                name=f"flow_{q}",
            )

        for i, s in data.A_HS:
            m.addConstr(
                gp.quicksum(psi[s, u] for v, u in data.A if v == s) <= data.max_flow * x_loc[i],
                name=f"pool_{i}_{s}",
            )

        for u, v in data.A_F:
            m.addConstr(
                gp.quicksum(varphi[u, v, k] for k in data.route_options.get((u, v), []) if (u, v, k) in varphi)
                == psi[u, v],
                name=f"route_{u}_{v}",
            )

        for k in data.E:
            m.addConstr(
                gp.quicksum(varphi[u, v, kk] for u, v, kk in data.varphi_keys if kk == k)
                <= data.swap_capacity(k) * x_loc[k],
                name=f"swap_{k}",
            )

        for l in data.L:
            m.addConstr(
                gp.quicksum(
                    data.route_energy_coef[l, i, k] * varphi[l, i, k]
                    for u, i, k in data.varphi_keys
                    if u == l and (l, i) in data.A_LH_set
                )
                <= 0.0,
                name=f"energyL_{l}",
            )

        for i, s in data.A_HS:
            m.addConstr(
                data.omega * (xi[i] + xi[s])
                + gp.quicksum(
                    data.route_energy_coef[i, u, k] * varphi[i, u, k]
                    for h, u, k in data.varphi_keys
                    if h == i and (i, u) in data.A_HH_set
                )
                + gp.quicksum(
                    data.route_energy_coef[s, u, k] * varphi[s, u, k]
                    for h, u, k in data.varphi_keys
                    if h == s and ((s, u) in data.A_SL or (s, u) in data.A_SS)
                )
                <= 0.0,
                name=f"energyHS_{i}_{s}",
            )

        mu_expr = {i: gp.quicksum(data.mu_levels[c] * z_loc[i, c] for c in data.C) for i in data.H}
        sqrt_mu_expr = {i: gp.quicksum(math.sqrt(data.mu_levels[c]) * z_loc[i, c] for c in data.C) for i in data.H}

        for i in data.H:
            m.addConstr(lambd[i] == gp.quicksum(psi[u, i] for u, v in data.A if v == i), name=f"lambdaDef_{i}")
            m.addConstr(lambd[i] <= mu_expr[i] - x_loc[i] / data.tau_H, name=f"hubWait_{i}")

        for s in data.S:
            m.addConstr(
                xi[s] <= data.tau_S * gp.quicksum(psi[s, v] for u, v in data.A if u == s),
                name=f"meanS_{s}",
            )

        m.addConstr(
            gp.quicksum(xi[u] for u in data.HLS)
            + gp.quicksum(data.t[u, v, k] * varphi[u, v, k] for u, v, k in data.varphi_keys)
            <= gamma_loc,
            name="fleet",
        )

        v1_q = m.addVars(data.L + data.S, lb=-GRB.INFINITY, name="q_v1")
        v2_q = m.addVars(data.L + data.S, lb=-GRB.INFINITY, name="q_v2")
        t_q = m.addVars(data.L + data.S, lb=0.0, name="q_t")
        for q in data.L + data.S:
            rho_q = rho_l[q] if q in data.L else rho_s[q]
            m.addConstr(v1_q[q] == 2.0, name=f"q1_{q}")
            m.addConstr(v2_q[q] == xi[q] + rho_q, name=f"q2_{q}")
            m.addConstr(t_q[q] == xi[q] - rho_q + 2.0, name=f"q3_{q}")
            m.addQConstr(v1_q[q] * v1_q[q] + v2_q[q] * v2_q[q] <= t_q[q] * t_q[q], name=f"soc_Q_{q}")

        v1_h = m.addVars(data.H, lb=-GRB.INFINITY, name="h_v1")
        v2_h = m.addVars(data.H, lb=-GRB.INFINITY, name="h_v2")
        t_h = m.addVars(data.H, lb=0.0, name="h_t")
        for i in data.H:
            m.addConstr(v1_h[i] == 2.0 * sqrt_mu_expr[i], name=f"h1_{i}")
            m.addConstr(v2_h[i] == xi[i] + lambd[i] - mu_expr[i] + 1.0, name=f"h2_{i}")
            m.addConstr(t_h[i] == xi[i] - lambd[i] + mu_expr[i] + 1.0, name=f"h3_{i}")
            m.addQConstr(v1_h[i] * v1_h[i] + v2_h[i] * v2_h[i] <= t_h[i] * t_h[i], name=f"soc_H_{i}")

        m.optimize()
        if m.Status != GRB.OPTIMAL:
            return m.Status, None

        strengthened_coeffs = {
            "constant": float(m.ObjVal),
            "x": coeffs["x"],
            "y": coeffs["y"],
            "z": coeffs["z"],
            "Gamma": coeffs["Gamma"],
        }
        rhs_at_ref = bdd_rhs_expr(strengthened_coeffs, x_ref, y_ref, z_ref, gamma_ref, False)
        guard = 1e-5 + 1e-6 * max(1.0, abs(float(standard_rhs)))
        if rhs_at_ref < float(standard_rhs) - guard:
            return GRB.NUMERIC, None

        return m.Status, {
            "kind": "optimality",
            "source": "bdd_strengthened",
            "bdd_coeffs": strengthened_coeffs,
            "rhs_at_hat": float(rhs_at_ref),
            "standard_rhs_at_hat": float(standard_rhs),
            "bdd_intercept": float(m.ObjVal),
        }
