import math
import os
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Tuple

import gurobipy as gp
from gurobipy import GRB

# Support both package execution and direct script execution.
if __package__ in (None, ""):
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)

    from src import config
    from src.data_processor.instance import Instance
else:
    from .. import config
    from ..data_processor.instance import Instance


TOL = 1e-7
Arc = Tuple[str, str]
Route = Tuple[str, str, str]


def _val(container: Any, key: Any = None) -> Any:
    if key is None:
        return container
    return container[key]


def _sum(terms: Iterable[Any], use_gurobi_expr: bool) -> Any:
    return gp.quicksum(terms) if use_gurobi_expr else sum(terms)


@dataclass
class MasterSolution:
    x: Dict[str, float]
    y: Dict[Tuple[str, str], float]
    z: Dict[Tuple[str, int], float]
    Gamma: float
    theta: float
    obj: float


class RegimeIIData:
    """Index sets and route data for the final Regime-II notation."""

    def __init__(self, instance: Instance):
        self.inst = instance
        self.L = list(instance.L)
        self.H = list(instance.H)
        self.S = list(instance.S)
        self.E = list(instance.E)
        self.C = list(range(len(config.MU_HUB)))
        self.mu_levels = [float(v) for v in config.MU_HUB]

        self.A = sorted(instance.A)
        self.A_HH = sorted(instance.A_HH)
        self.A_LH = sorted(instance.A_LH)
        self.A_HS = sorted(instance.A_HS)
        self.A_SH = sorted(instance.A_SH)
        self.A_SL = sorted(instance.A_SL)
        self.A_SS = sorted(instance.A_SS)
        self.A_F = sorted(instance.A_F)

        self.A_set = set(self.A)
        self.A_HH_set = set(self.A_HH)
        self.A_LH_set = set(self.A_LH)
        self.A_HS_set = set(self.A_HS)
        self.A_SH_set = set(self.A_SH)
        self.A_F_set = set(self.A_F)

        self.HLS = self.L + self.H + self.S
        self.N_x = self.L + self.H + self.E

        self.out_arcs: Dict[str, list[Arc]] = {}
        self.in_arcs: Dict[str, list[Arc]] = {}
        for arc in self.A:
            self.out_arcs.setdefault(arc[0], []).append(arc)
            self.in_arcs.setdefault(arc[1], []).append(arc)

        self.route_options: Dict[Arc, list[str]] = {}
        self.t: Dict[Route, float] = {}
        self.flight_cost: Dict[Route, float] = {}
        self.varphi_keys: list[Route] = []
        self.varphi_by_arc: Dict[Arc, list[str]] = {}
        self._build_route_options()

        self.alpha_keys = [
            (l, i, j)
            for (l, i) in self.A_LH
            for (i2, j) in self.A_HH
            if i2 == i
        ]
        self.beta_keys = list(self.A_HH)
        self.beta_support = [
            (s, i, j)
            for (s, i) in self.A_SH
            for (i2, j) in self.A_HH
            if i2 == i
        ]

        self.gamma_cost = float(config.COST_DRONE)
        self.revenue = float(getattr(config, "ORDER_REVENUE_COEF", 0.0))
        self.max_flow = float(config.MAX_FLOW_E)
        self.tau_H = float(config.TAU_H)
        self.tau_S = float(config.TAU_S)
        self.omega = float(config.ENERGY_RATE_OMEGA)
        self.Omega_eff = (
            1.0 - float(getattr(config, "ENERGY_EPSILON", getattr(config, "EPS", 0.0)))
        ) * float(config.BATTERY_CAPACITY_OMEGA_UPPER)
        self.route_energy_coef = {
            (u, v, k): self.omega * self.t[u, v, k] - self.Omega_eff * (1.0 if k in self.E else 0.0)
            for (u, v, k) in self.varphi_keys
        }
        self.total_revenue_ub = self.revenue * sum(float(v) for v in instance.h.values())
        self.core_gamma = max(1.0, sum(float(v) for v in instance.h.values()))

    def _build_route_options(self) -> None:
        flight_cost_rate = float(getattr(config, "COST_FLIGHT_TIME", 0.1))
        for u, v in self.A_F:
            options = []
            if self.inst.get_dist(u, v) <= config.MAX_DIRECT_DISTANCE:
                options.append("0")
                self.t[u, v, "0"] = self.inst.get_time(u, v)
            for k in self.inst.E_uv.get((u, v), []):
                options.append(k)
                self.t[u, v, k] = self.inst.get_time(u, k) + self.inst.get_time(k, v)

            ordered = []
            seen = set()
            for k in options:
                if k not in seen:
                    ordered.append(k)
                    seen.add(k)
            self.route_options[u, v] = ordered
            self.varphi_by_arc[u, v] = ordered
            for k in ordered:
                self.varphi_keys.append((u, v, k))
                self.flight_cost[u, v, k] = flight_cost_rate * self.t[u, v, k]

    def p_chain(self, hub: str) -> float:
        p_by_hub = getattr(config, "P_CHAIN_BY_HUB", None)
        if isinstance(p_by_hub, dict):
            return float(p_by_hub.get(hub, config.P_CHAIN))
        return float(config.P_CHAIN)

    def swap_capacity(self, k: str) -> float:
        return float(self.inst.mu.get(k, getattr(config, "SWAP_CAPACITY", config.MU_SWAPPING)))

    def fixed_cost_value(self, x_ref: Dict[str, float], z_ref: Dict[Tuple[str, int], float]) -> float:
        fixed = sum(float(self.inst.f.get(l, config.COST_BASE)) * x_ref.get(l, 0.0) for l in self.L)
        fixed += sum(float(getattr(config, "COST_HUB_FIXED", 0.0)) * x_ref.get(i, 0.0) for i in self.H)
        fixed += sum(float(self.inst.f.get(k, config.COST_SWAPPING)) * x_ref.get(k, 0.0) for k in self.E)
        fixed += sum(float(config.COST_HUB[c]) * z_ref.get((i, c), 0.0) for i in self.H for c in self.C)
        return fixed


class RegimeIIDualObjective:
    """Build the Benders cut RHS from a feasible dual solution."""

    @staticmethod
    def mu_expr(data: RegimeIIData, z_ref: Any, i: str, use_gurobi_expr: bool) -> Any:
        return _sum((data.mu_levels[c] * _val(z_ref, (i, c)) for c in data.C), use_gurobi_expr)

    @staticmethod
    def sqrt_mu_expr(data: RegimeIIData, z_ref: Any, i: str, use_gurobi_expr: bool) -> Any:
        return _sum((math.sqrt(data.mu_levels[c]) * _val(z_ref, (i, c)) for c in data.C), use_gurobi_expr)

    @staticmethod
    def expr(
        data: RegimeIIData,
        duals: Dict[str, Any],
        x_ref: Any,
        y_ref: Any,
        z_ref: Any,
        gamma_ref: Any,
        use_gurobi_expr: bool,
    ) -> Any:
        obj = gp.LinExpr() if use_gurobi_expr else 0.0

        obj += _sum((duals["rhoL"][l] * _val(x_ref, l) for l in data.L), use_gurobi_expr)
        obj += _sum((duals["rhoS"][i, s] * _val(x_ref, i) for i, s in data.A_HS), use_gurobi_expr)

        obj += _sum(
            (
                duals["alpha2"][l, i, j] * _val(y_ref, (i, l))
                + duals["alpha3"][l, i, j] * _val(x_ref, j)
                + duals["alpha4"][l, i, j] * (2.0 - _val(y_ref, (i, l)) - _val(x_ref, j))
                for l, i, j in data.alpha_keys
            ),
            use_gurobi_expr,
        )

        obj += _sum(
            (
                duals["beta2"][s, i, j] * _val(x_ref, j)
                + duals["beta3"][s, i, j] * (1.0 - _val(x_ref, j))
                for s, i, j in data.beta_support
            ),
            use_gurobi_expr,
        )

        obj += _sum(
            (duals["pool"][i, s] * data.max_flow * _val(x_ref, i) for i, s in data.A_HS),
            use_gurobi_expr,
        )
        obj += _sum(
            (duals["swap"][k] * data.swap_capacity(k) * _val(x_ref, k) for k in data.E),
            use_gurobi_expr,
        )

        for i in data.H:
            mu_i = RegimeIIDualObjective.mu_expr(data, z_ref, i, use_gurobi_expr)
            sqrt_mu_i = RegimeIIDualObjective.sqrt_mu_expr(data, z_ref, i, use_gurobi_expr)
            obj += duals["hubWait"][i] * (mu_i - _val(x_ref, i) / data.tau_H)
            obj += duals["h1"][i] * (2.0 * sqrt_mu_i)
            obj += duals["h2"][i] * (1.0 - mu_i)
            obj += duals["h3"][i] * (1.0 + mu_i)

        obj += _sum((duals["q1"][q] * 2.0 + duals["q3"][q] * 2.0 for q in data.L + data.S), use_gurobi_expr)
        obj += duals["fleet"] * gamma_ref
        return obj


class RegimeIIPrimalSubproblem:
    """Continuous SOCP subproblem; cuts are obtained from primal dual multipliers."""

    def __init__(self, data: RegimeIIData, output: bool = False):
        self.data = data
        self.output = output
        self.model: Optional[gp.Model] = None

    def solve(
        self,
        x_hat: Dict[str, float],
        y_hat: Dict[Tuple[str, str], float],
        z_hat: Dict[Tuple[str, int], float],
        gamma_hat: float,
    ) -> Tuple[int, Optional[float], Optional[Dict[str, Any]]]:
        data = self.data
        inst = data.inst
        m = gp.Model("RegimeII_PrimalSub")
        m.setParam("OutputFlag", 1 if self.output else 0)
        m.setParam("QCPDual", 1)
        m.setParam("DualReductions", 0)
        m.setParam("BarHomogeneous", 1)

        rho_l = m.addVars(data.L, lb=0.0, name="rho_l")
        rho_s = m.addVars(data.S, lb=0.0, name="rho_s")
        alpha = m.addVars(data.alpha_keys, lb=0.0, name="alpha")
        beta = m.addVars(data.beta_keys, lb=0.0, name="beta")
        psi = m.addVars(data.A, lb=0.0, name="psi")
        varphi = m.addVars(data.varphi_keys, lb=0.0, name="varphi")
        lambd = m.addVars(data.H, lb=0.0, name="lambda")
        xi = m.addVars(data.HLS, lb=0.0, name="xi")

        m.setObjective(
            data.revenue * gp.quicksum(psi[i, j] for i, j in data.A_HH)
            - gp.quicksum(data.flight_cost[u, v, k] * varphi[u, v, k] for u, v, k in data.varphi_keys),
            GRB.MAXIMIZE,
        )

        cons: Dict[str, Any] = {}
        cons["rhoL"] = {l: m.addConstr(rho_l[l] <= x_hat.get(l, 0.0), name=f"rhoL_{l}") for l in data.L}
        cons["rhoS"] = {
            (i, s): m.addConstr(rho_s[s] <= x_hat.get(i, 0.0), name=f"rhoS_{i}_{s}")
            for i, s in data.A_HS
        }

        cons["alpha1"], cons["alpha2"], cons["alpha3"], cons["alpha4"] = {}, {}, {}, {}
        for l, i, j in data.alpha_keys:
            cons["alpha1"][l, i, j] = m.addConstr(alpha[l, i, j] - rho_l[l] <= 0.0, name=f"alpha1_{l}_{i}_{j}")
            cons["alpha2"][l, i, j] = m.addConstr(alpha[l, i, j] <= y_hat.get((i, l), 0.0), name=f"alpha2_{l}_{i}_{j}")
            cons["alpha3"][l, i, j] = m.addConstr(alpha[l, i, j] <= x_hat.get(j, 0.0), name=f"alpha3_{l}_{i}_{j}")
            cons["alpha4"][l, i, j] = m.addConstr(
                -alpha[l, i, j] + rho_l[l] <= 2.0 - y_hat.get((i, l), 0.0) - x_hat.get(j, 0.0),
                name=f"alpha4_{l}_{i}_{j}",
            )

        cons["beta1"], cons["beta2"], cons["beta3"] = {}, {}, {}
        for s, i, j in data.beta_support:
            cons["beta1"][s, i, j] = m.addConstr(beta[i, j] - rho_s[s] <= 0.0, name=f"beta1_{s}_{i}_{j}")
            cons["beta2"][s, i, j] = m.addConstr(beta[i, j] <= x_hat.get(j, 0.0), name=f"beta2_{s}_{i}_{j}")
            cons["beta3"][s, i, j] = m.addConstr(-beta[i, j] + rho_s[s] <= 1.0 - x_hat.get(j, 0.0), name=f"beta3_{s}_{i}_{j}")

        cons["psiHH"], cons["psiLH"], cons["psiSH"] = {}, {}, {}
        for i, j in data.A_HH:
            p_i = data.p_chain(i)
            demand = float(inst.h.get((i, j), 0.0))
            cons["psiHH"][i, j] = m.addConstr(
                psi[i, j]
                - demand
                * (
                    (1.0 - p_i)
                    * gp.quicksum(alpha[l, i, j] for l, h in data.A_LH if h == i and (l, i, j) in alpha)
                    + p_i * beta[i, j]
                )
                == 0.0,
                name=f"psiHH_{i}_{j}",
            )

        for l, i in data.A_LH:
            cons["psiLH"][l, i] = m.addConstr(
                psi[l, i]
                - gp.quicksum(
                    float(inst.h.get((i, j), 0.0)) * (1.0 - data.p_chain(i)) * alpha[l, i, j]
                    for h, j in data.A_HH
                    if h == i and (l, i, j) in alpha
                )
                == 0.0,
                name=f"psiLH_{l}_{i}",
            )

        for s, i in data.A_SH:
            cons["psiSH"][s, i] = m.addConstr(
                psi[s, i]
                - gp.quicksum(
                    float(inst.h.get((i, j), 0.0)) * data.p_chain(i) * beta[i, j]
                    for h, j in data.A_HH
                    if h == i
                )
                == 0.0,
                name=f"psiSH_{s}_{i}",
            )

        cons["flow"] = {}
        for q in data.HLS:
            cons["flow"][q] = m.addConstr(
                gp.quicksum(psi[u, q] for u, v in data.A if v == q)
                - gp.quicksum(psi[q, v] for u, v in data.A if u == q)
                == 0.0,
                name=f"flow_{q}",
            )

        cons["pool"] = {}
        for i, s in data.A_HS:
            cons["pool"][i, s] = m.addConstr(
                gp.quicksum(psi[s, u] for v, u in data.A if v == s) <= data.max_flow * x_hat.get(i, 0.0),
                name=f"pool_{i}_{s}",
            )

        cons["route"] = {}
        for u, v in data.A_F:
            cons["route"][u, v] = m.addConstr(
                gp.quicksum(varphi[u, v, k] for k in data.route_options.get((u, v), []) if (u, v, k) in varphi)
                - psi[u, v]
                == 0.0,
                name=f"route_{u}_{v}",
            )

        cons["swap"] = {}
        for k in data.E:
            cons["swap"][k] = m.addConstr(
                gp.quicksum(varphi[u, v, k] for u, v, kk in data.varphi_keys if kk == k)
                <= data.swap_capacity(k) * x_hat.get(k, 0.0),
                name=f"swap_{k}",
            )

        cons["energyL"] = {}
        for l in data.L:
            cons["energyL"][l] = m.addConstr(
                gp.quicksum(
                    data.route_energy_coef[l, i, k] * varphi[l, i, k]
                    for u, i, k in data.varphi_keys
                    if u == l and (l, i) in data.A_LH_set
                )
                <= 0.0,
                name=f"energyL_{l}",
            )

        cons["energyHS"] = {}
        for i, s in data.A_HS:
            cons["energyHS"][i, s] = m.addConstr(
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

        cons["lambdaDef"], cons["hubWait"], cons["meanS"] = {}, {}, {}
        mu_hat = {
            i: sum(data.mu_levels[c] * z_hat.get((i, c), 0.0) for c in data.C)
            for i in data.H
        }
        for i in data.H:
            cons["lambdaDef"][i] = m.addConstr(
                lambd[i] - gp.quicksum(psi[u, i] for u, v in data.A if v == i) == 0.0,
                name=f"lambdaDef_{i}",
            )
            cons["hubWait"][i] = m.addConstr(
                lambd[i] <= mu_hat[i] - x_hat.get(i, 0.0) / data.tau_H,
                name=f"hubWait_{i}",
            )

        for s in data.S:
            cons["meanS"][s] = m.addConstr(
                xi[s] - data.tau_S * gp.quicksum(psi[s, v] for u, v in data.A if u == s) <= 0.0,
                name=f"meanS_{s}",
            )

        cons["fleet"] = m.addConstr(
            gp.quicksum(xi[u] for u in data.HLS)
            + gp.quicksum(data.t[u, v, k] * varphi[u, v, k] for u, v, k in data.varphi_keys)
            <= gamma_hat,
            name="fleet",
        )

        v1_q = m.addVars(data.L + data.S, lb=-GRB.INFINITY, name="q_v1")
        v2_q = m.addVars(data.L + data.S, lb=-GRB.INFINITY, name="q_v2")
        t_q = m.addVars(data.L + data.S, lb=0.0, name="q_t")
        cons["q1"], cons["q2"], cons["q3"] = {}, {}, {}
        for q in data.L + data.S:
            rho_q = rho_l[q] if q in data.L else rho_s[q]
            cons["q1"][q] = m.addConstr(v1_q[q] == 2.0, name=f"q1_{q}")
            cons["q2"][q] = m.addConstr(v2_q[q] - xi[q] - rho_q == 0.0, name=f"q2_{q}")
            cons["q3"][q] = m.addConstr(t_q[q] - xi[q] + rho_q == 2.0, name=f"q3_{q}")
            m.addQConstr(v1_q[q] * v1_q[q] + v2_q[q] * v2_q[q] <= t_q[q] * t_q[q], name=f"soc_Q_{q}")

        v1_h = m.addVars(data.H, lb=-GRB.INFINITY, name="h_v1")
        v2_h = m.addVars(data.H, lb=-GRB.INFINITY, name="h_v2")
        t_h = m.addVars(data.H, lb=0.0, name="h_t")
        cons["h1"], cons["h2"], cons["h3"] = {}, {}, {}
        for i in data.H:
            cons["h1"][i] = m.addConstr(v1_h[i] == 2.0 * math.sqrt(max(mu_hat[i], 0.0)), name=f"h1_{i}")
            cons["h2"][i] = m.addConstr(v2_h[i] - xi[i] - lambd[i] == 1.0 - mu_hat[i], name=f"h2_{i}")
            cons["h3"][i] = m.addConstr(t_h[i] - xi[i] + lambd[i] == 1.0 + mu_hat[i], name=f"h3_{i}")
            m.addQConstr(v1_h[i] * v1_h[i] + v2_h[i] * v2_h[i] <= t_h[i] * t_h[i], name=f"soc_H_{i}")

        m.optimize()
        self.model = m

        if m.Status != GRB.OPTIMAL:
            return m.Status, None, None

        pi_failures: list[str] = []

        def pi(name: str, c: gp.Constr) -> float:
            try:
                val = float(c.Pi)
            except (AttributeError, gp.GurobiError):
                pi_failures.append(name)
                return 0.0
            return 0.0 if abs(val) <= TOL else val

        duals: Dict[str, Any] = {}
        for name, item in cons.items():
            if hasattr(item, "items"):
                duals[name] = {key: pi(f"{name}{key}", constr) for key, constr in item.items()}
            else:
                duals[name] = pi(name, item)

        rhs_value = RegimeIIDualObjective.expr(data, duals, x_hat, y_hat, z_hat, gamma_hat, use_gurobi_expr=False)
        tight_tol = 1e-7 + 1e-7 * max(1.0, abs(float(m.ObjVal)))
        degenerate_zero = abs(float(m.ObjVal)) <= 1e-6
        if pi_failures or degenerate_zero or abs(rhs_value - float(m.ObjVal)) > tight_tol:
            if pi_failures:
                reason = "pi"
            elif degenerate_zero:
                reason = "degenerate"
            else:
                reason = "tightness"
            fallback = self._dual_fallback(
                x_hat,
                y_hat,
                z_hat,
                gamma_hat,
                float(m.ObjVal),
                reason,
            )
            if fallback is not None:
                return m.Status, m.ObjVal, fallback
            if pi_failures:
                return m.Status, m.ObjVal, None

        return m.Status, m.ObjVal, {
            "kind": "optimality",
            "z_star": m.ObjVal,
            "duals": duals,
            "rhs_at_hat": rhs_value,
            "source": "primal",
        }

    def _dual_fallback(
        self,
        x_hat: Dict[str, float],
        y_hat: Dict[Tuple[str, str], float],
        z_hat: Dict[Tuple[str, int], float],
        gamma_hat: float,
        primal_obj: float,
        reason: str,
    ) -> Optional[Dict[str, Any]]:
        solver = RegimeIIDualSubproblem(self.data, output=False)
        status, _, cut_data = solver.solve(x_hat, y_hat, z_hat, gamma_hat)
        if status == GRB.OPTIMAL and cut_data is not None:
            cut_data["source"] = f"primal_fallback_dual_{reason}"
            cut_data["primal_obj"] = primal_obj
            return cut_data
        return None


class RegimeIIDualSubproblem:
    """Explicit SOCP dual. Also used for Magnanti-Wong Pareto cuts."""

    def __init__(self, data: RegimeIIData, output: bool = False):
        self.data = data
        self.output = output

    def _build_model(self) -> Tuple[gp.Model, Dict[str, Any]]:
        data = self.data
        inst = data.inst
        m = gp.Model("RegimeII_DualSub")
        m.setParam("OutputFlag", 1 if self.output else 0)
        m.setParam("DualReductions", 0)
        m.setParam("BarHomogeneous", 1)

        d: Dict[str, Any] = {}
        d["rhoL"] = m.addVars(data.L, lb=0.0, name="rhoL")
        d["rhoS"] = m.addVars(data.A_HS, lb=0.0, name="rhoS")
        for name in ["alpha1", "alpha2", "alpha3", "alpha4"]:
            d[name] = m.addVars(data.alpha_keys, lb=0.0, name=name)
        for name in ["beta1", "beta2", "beta3"]:
            d[name] = m.addVars(data.beta_support, lb=0.0, name=name)

        d["psiHH"] = m.addVars(data.A_HH, lb=-GRB.INFINITY, name="psiHH")
        d["psiLH"] = m.addVars(data.A_LH, lb=-GRB.INFINITY, name="psiLH")
        d["psiSH"] = m.addVars(data.A_SH, lb=-GRB.INFINITY, name="psiSH")
        d["flow"] = m.addVars(data.HLS, lb=-GRB.INFINITY, name="flow")
        d["pool"] = m.addVars(data.A_HS, lb=0.0, name="pool")
        d["route"] = m.addVars(data.A_F, lb=-GRB.INFINITY, name="route")
        d["swap"] = m.addVars(data.E, lb=0.0, name="swap")
        d["energyL"] = m.addVars(data.L, lb=0.0, name="energyL")
        d["energyHS"] = m.addVars(data.A_HS, lb=0.0, name="energyHS")
        d["lambdaDef"] = m.addVars(data.H, lb=-GRB.INFINITY, name="lambdaDef")
        d["hubWait"] = m.addVars(data.H, lb=0.0, name="hubWait")
        d["meanS"] = m.addVars(data.S, lb=0.0, name="meanS")
        d["fleet"] = m.addVar(lb=0.0, name="fleet")
        d["q1"] = m.addVars(data.L + data.S, lb=-GRB.INFINITY, name="q1")
        d["q2"] = m.addVars(data.L + data.S, lb=-GRB.INFINITY, name="q2")
        d["q3"] = m.addVars(data.L + data.S, lb=0.0, name="q3")
        d["h1"] = m.addVars(data.H, lb=-GRB.INFINITY, name="h1")
        d["h2"] = m.addVars(data.H, lb=-GRB.INFINITY, name="h2")
        d["h3"] = m.addVars(data.H, lb=0.0, name="h3")

        for l in data.L:
            expr = d["rhoL"][l]
            expr += gp.quicksum(-d["alpha1"][l, i, j] + d["alpha4"][l, i, j] for ll, i, j in data.alpha_keys if ll == l)
            expr += -d["q2"][l] + d["q3"][l]
            m.addConstr(expr >= 0.0, name=f"dual_rhoL_{l}")

        for s in data.S:
            expr = gp.quicksum(d["rhoS"][i, ss] for i, ss in data.A_HS if ss == s)
            expr += gp.quicksum(-d["beta1"][ss, i, j] + d["beta3"][ss, i, j] for ss, i, j in data.beta_support if ss == s)
            expr += -d["q2"][s] + d["q3"][s]
            m.addConstr(expr >= 0.0, name=f"dual_rhoS_{s}")

        for l, i, j in data.alpha_keys:
            p_i = data.p_chain(i)
            demand = float(inst.h.get((i, j), 0.0))
            expr = d["alpha1"][l, i, j] + d["alpha2"][l, i, j] + d["alpha3"][l, i, j] - d["alpha4"][l, i, j]
            expr += -demand * (1.0 - p_i) * d["psiHH"][i, j]
            expr += -demand * (1.0 - p_i) * d["psiLH"][l, i]
            m.addConstr(expr >= 0.0, name=f"dual_alpha_{l}_{i}_{j}")

        for i, j in data.beta_keys:
            p_i = data.p_chain(i)
            demand = float(inst.h.get((i, j), 0.0))
            expr = gp.quicksum(d["beta1"][s, i, j] + d["beta2"][s, i, j] - d["beta3"][s, i, j] for s, h, jj in data.beta_support if h == i and jj == j)
            expr += -demand * p_i * d["psiHH"][i, j]
            expr += gp.quicksum(-demand * p_i * d["psiSH"][s, i] for s, h in data.A_SH if h == i)
            m.addConstr(expr >= 0.0, name=f"dual_beta_{i}_{j}")

        for u, v in data.A:
            expr = gp.LinExpr()
            if (u, v) in data.A_HH_set:
                expr += d["psiHH"][u, v]
            if (u, v) in data.A_LH_set:
                expr += d["psiLH"][u, v]
            if (u, v) in data.A_SH_set:
                expr += d["psiSH"][u, v]
            if v in data.HLS:
                expr += d["flow"][v]
            if u in data.HLS:
                expr -= d["flow"][u]
            if u in data.S:
                expr += gp.quicksum(d["pool"][i, s] for i, s in data.A_HS if s == u)
                expr -= data.tau_S * d["meanS"][u]
            if (u, v) in data.A_F_set:
                expr -= d["route"][u, v]
            if v in data.H:
                expr -= d["lambdaDef"][v]
            rhs = data.revenue if (u, v) in data.A_HH_set else 0.0
            m.addConstr(expr >= rhs, name=f"dual_psi_{u}_{v}")

        for u, v, k in data.varphi_keys:
            expr = d["route"][u, v] + d["fleet"] * data.t[u, v, k]
            if k in data.E:
                expr += d["swap"][k]
            if u in data.L and (u, v) in data.A_LH_set:
                expr += d["energyL"][u] * data.route_energy_coef[u, v, k]
            expr += gp.quicksum(
                d["energyHS"][i, s] * data.route_energy_coef[u, v, k]
                for i, s in data.A_HS
                if u == i or u == s
            )
            m.addConstr(expr >= -data.flight_cost[u, v, k], name=f"dual_varphi_{u}_{v}_{k}")

        for i in data.H:
            expr = d["lambdaDef"][i] + d["hubWait"][i] - d["h2"][i] + d["h3"][i]
            m.addConstr(expr >= 0.0, name=f"dual_lambda_{i}")

        for l in data.L:
            m.addConstr(d["fleet"] - d["q2"][l] - d["q3"][l] >= 0.0, name=f"dual_xi_L_{l}")

        for s in data.S:
            expr = d["fleet"] + d["meanS"][s] - d["q2"][s] - d["q3"][s]
            expr += gp.quicksum(data.omega * d["energyHS"][i, ss] for i, ss in data.A_HS if ss == s)
            m.addConstr(expr >= 0.0, name=f"dual_xi_S_{s}")

        for i in data.H:
            expr = d["fleet"] - d["h2"][i] - d["h3"][i]
            expr += gp.quicksum(data.omega * d["energyHS"][ii, s] for ii, s in data.A_HS if ii == i)
            m.addConstr(expr >= 0.0, name=f"dual_xi_H_{i}")

        for q in data.L + data.S:
            m.addQConstr(d["q1"][q] * d["q1"][q] + d["q2"][q] * d["q2"][q] <= d["q3"][q] * d["q3"][q], name=f"dual_soc_Q_{q}")

        for i in data.H:
            m.addQConstr(d["h1"][i] * d["h1"][i] + d["h2"][i] * d["h2"][i] <= d["h3"][i] * d["h3"][i], name=f"dual_soc_H_{i}")

        return m, d

    @staticmethod
    def _extract(container: Any) -> Any:
        if hasattr(container, "items"):
            return {k: float(v.X) for k, v in container.items()}
        return float(container.X)

    def _collect(self, d: Dict[str, Any]) -> Dict[str, Any]:
        return {name: self._extract(container) for name, container in d.items()}

    def solve(
        self,
        x_ref: Dict[str, float],
        y_ref: Dict[Tuple[str, str], float],
        z_ref: Dict[Tuple[str, int], float],
        gamma_ref: float,
    ) -> Tuple[int, Optional[float], Optional[Dict[str, Any]]]:
        m, d = self._build_model()
        m.setObjective(RegimeIIDualObjective.expr(self.data, d, x_ref, y_ref, z_ref, gamma_ref, True), GRB.MINIMIZE)
        m.optimize()
        if m.Status != GRB.OPTIMAL:
            return m.Status, None, None
        duals = self._collect(d)
        return m.Status, m.ObjVal, {
            "kind": "optimality",
            "z_star": m.ObjVal,
            "duals": duals,
            "rhs_at_hat": RegimeIIDualObjective.expr(self.data, duals, x_ref, y_ref, z_ref, gamma_ref, False),
            "source": "dual",
        }

    def solve_pareto(
        self,
        x_hat: Dict[str, float],
        y_hat: Dict[Tuple[str, str], float],
        z_hat: Dict[Tuple[str, int], float],
        gamma_hat: float,
        core_x: Dict[str, float],
        core_y: Dict[Tuple[str, str], float],
        core_z: Dict[Tuple[str, int], float],
        core_gamma: float,
        z_star: float,
    ) -> Tuple[int, Optional[float], Optional[Dict[str, Any]]]:
        m, d = self._build_model()
        anchor = RegimeIIDualObjective.expr(self.data, d, x_hat, y_hat, z_hat, gamma_hat, True)
        eps = 1e-5 + 1e-5 * max(1.0, abs(float(z_star)))
        m.addConstr(anchor >= z_star - eps, name="mw_anchor_lb")
        m.addConstr(anchor <= z_star + eps, name="mw_anchor_ub")
        m.setObjective(RegimeIIDualObjective.expr(self.data, d, core_x, core_y, core_z, core_gamma, True), GRB.MINIMIZE)
        m.optimize()
        if m.Status != GRB.OPTIMAL:
            return m.Status, None, None
        duals = self._collect(d)
        return m.Status, m.ObjVal, {
            "kind": "optimality",
            "z_star": z_star,
            "duals": duals,
            "rhs_at_hat": RegimeIIDualObjective.expr(self.data, duals, x_hat, y_hat, z_hat, gamma_hat, False),
            "mw_obj": m.ObjVal,
            "source": "pareto",
        }


class RegimeIIBenders:
    """Iterative Benders solver for the final Regime-II formulation."""

    def __init__(
        self,
        instance: Instance,
        subproblem: str = "dual",
        use_pareto: bool = True,
        max_iter: int = 30,
        gap_tol: float = 1e-3,
        master_time_limit: float = 30.0,
        master_mip_gap: float = 0.2,
        verbose: bool = True,
        use_trust_region: bool = False,
        trust_radius: Optional[int] = None,
        trust_radius_min: int = 1,
        trust_radius_max: Optional[int] = None,
        initial_solution: Optional[Any] = None,
    ):
        if subproblem not in {"primal", "dual"}:
            raise ValueError("subproblem must be 'primal' or 'dual'.")
        self.inst = instance
        self.data = RegimeIIData(instance)
        self.subproblem_mode = subproblem
        self.use_pareto = use_pareto
        self.max_iter = max_iter
        self.gap_tol = gap_tol
        self.master_time_limit = master_time_limit
        self.master_mip_gap = master_mip_gap
        self.verbose = verbose
        self.use_trust_region = use_trust_region
        self.trust_radius_min = max(1, int(trust_radius_min))
        self.trust_radius_max = trust_radius_max
        self.trust_radius = trust_radius
        self.initial_solution = initial_solution

        self.model: Optional[gp.Model] = None
        self.x = None
        self.y = None
        self.z = None
        self.Gamma = None
        self.Theta = None
        self.trust_constr = None
        self.trust_center_x: Optional[Dict[str, float]] = None
        self.best_obj = -math.inf
        self.best_solution: Optional[MasterSolution] = None
        self.history: list[Dict[str, Any]] = []
        self.subproblem_time = 0.0
        self.mw_time = 0.0
        self.global_check_time = 0.0
        self.stop_reason: Optional[str] = None
        self.converged = False

        self.core_x, self.core_y, self.core_z, self.core_gamma = self._initial_core_point()

    def _initial_core_point(self):
        data = self.data
        core_x = {u: 1.0 for u in data.N_x}
        core_y: Dict[Tuple[str, str], float] = {}
        for i in data.H:
            bases = [l for l, h in data.A_LH if h == i]
            denom = max(1, len(bases))
            for l in bases:
                core_y[i, l] = 1.0 / denom
        for i, l in [(h, b) for b, h in data.A_LH]:
            core_y.setdefault((i, l), 0.0)
        core_z = {(i, c): 1.0 / max(1, len(data.C)) for i in data.H for c in data.C}
        return core_x, core_y, core_z, data.core_gamma

    def _update_core(self, sol: MasterSolution) -> None:
        for u in self.data.N_x:
            self.core_x[u] = 0.5 * self.core_x.get(u, 0.0) + 0.5 * sol.x.get(u, 0.0)
        for key in self.core_y:
            self.core_y[key] = 0.5 * self.core_y.get(key, 0.0) + 0.5 * sol.y.get(key, 0.0)
        for key in self.core_z:
            self.core_z[key] = 0.5 * self.core_z.get(key, 0.0) + 0.5 * sol.z.get(key, 0.0)
        self.core_gamma = 0.5 * self.core_gamma + 0.5 * sol.Gamma

    def _build_master(self) -> None:
        data = self.data
        m = gp.Model(f"RegimeII_BendersMaster_{self.subproblem_mode}")
        m.setParam("OutputFlag", 1 if self.verbose else 0)
        m.setParam("TimeLimit", self.master_time_limit)
        m.setParam("MIPGap", self.master_mip_gap)

        self.x = m.addVars(data.N_x, vtype=GRB.BINARY, name="x")
        self.y = m.addVars([(i, l) for l, i in data.A_LH], vtype=GRB.BINARY, name="y")
        self.z = m.addVars(data.H, data.C, vtype=GRB.BINARY, name="z")
        self.Gamma = m.addVar(lb=0.0, vtype=GRB.INTEGER, name="Gamma")
        self.Theta = m.addVar(lb=0.0, ub=max(1.0, data.total_revenue_ub), name="Theta")

        fixed = gp.quicksum(float(self.inst.f.get(l, config.COST_BASE)) * self.x[l] for l in data.L)
        fixed += gp.quicksum(float(getattr(config, "COST_HUB_FIXED", 0.0)) * self.x[i] for i in data.H)
        fixed += gp.quicksum(float(self.inst.f.get(k, config.COST_SWAPPING)) * self.x[k] for k in data.E)
        fixed += gp.quicksum(float(config.COST_HUB[c]) * self.z[i, c] for i in data.H for c in data.C)

        m.setObjective(self.Theta - fixed - data.gamma_cost * self.Gamma, GRB.MAXIMIZE)

        for i in data.H:
            bases = [l for l, h in data.A_LH if h == i]
            if bases:
                m.addConstr(gp.quicksum(self.y[i, l] for l in bases) == self.x[i], name=f"assign_{i}")
            else:
                m.addConstr(self.x[i] == 0, name=f"no_assign_{i}")
            m.addConstr(gp.quicksum(self.z[i, c] for c in data.C) == self.x[i], name=f"capacity_{i}")

        for l, i in data.A_LH:
            m.addConstr(self.y[i, l] <= self.x[l], name=f"assign_active_{i}_{l}")

        self.model = m

        if self.use_trust_region:
            n_x = max(1, len(data.N_x))
            if self.trust_radius_max is None:
                self.trust_radius_max = n_x
            else:
                self.trust_radius_max = max(self.trust_radius_min, min(int(self.trust_radius_max), n_x))
            if self.trust_radius is None:
                self.trust_radius = max(self.trust_radius_min, min(self.trust_radius_max, max(2, n_x // 2)))
            else:
                self.trust_radius = max(self.trust_radius_min, min(int(self.trust_radius), self.trust_radius_max))

    def _coerce_initial_solution(self) -> Optional[MasterSolution]:
        if self.initial_solution is None:
            return None
        init = self.initial_solution
        gamma = getattr(init, "Gamma", getattr(init, "gamma", None))
        if gamma is None:
            return None
        return MasterSolution(
            x=dict(getattr(init, "x")),
            y=dict(getattr(init, "y")),
            z=dict(getattr(init, "z")),
            Gamma=float(gamma),
            theta=float(getattr(init, "theta", 0.0) or 0.0),
            obj=float(getattr(init, "obj", 0.0) or 0.0),
        )

    def _apply_initial_start(self, sol: MasterSolution) -> None:
        for u, value in sol.x.items():
            if u in self.x:
                self.x[u].Start = value
        for key, value in sol.y.items():
            if key in self.y:
                self.y[key].Start = value
        for key, value in sol.z.items():
            if key in self.z:
                self.z[key].Start = value
        self.Gamma.Start = sol.Gamma
        if sol.theta:
            self.Theta.Start = sol.theta

    def _load_initial_solution(self) -> None:
        sol = self._coerce_initial_solution()
        if sol is None:
            return
        self._apply_initial_start(sol)
        status, sub_obj, cut_data = self._solve_subproblem(sol)
        if status != GRB.OPTIMAL or sub_obj is None or cut_data is None:
            if self.verbose:
                print(f"[Benders-RII] initial solution rejected by subproblem status={status}.")
            return
        self._add_cut(cut_data["duals"], f"initial_{cut_data.get('source', self.subproblem_mode)}")
        fixed = self.data.fixed_cost_value(sol.x, sol.z)
        self.best_obj = float(sub_obj) - fixed - self.data.gamma_cost * sol.Gamma
        self.best_solution = sol
        if self.use_trust_region:
            self.trust_center_x = dict(sol.x)
        if self.verbose:
            print(f"[Benders-RII] loaded initial feasible solution obj={self.best_obj:.4f}.")

    def _refresh_trust_region(self) -> None:
        if not self.use_trust_region or self.trust_center_x is None:
            return
        assert self.model is not None
        if self.trust_constr is not None:
            self.model.remove(self.trust_constr)
            self.trust_constr = None

        expr = gp.LinExpr()
        for u in self.data.N_x:
            if self.trust_center_x.get(u, 0.0) > 0.5:
                expr += 1.0 - self.x[u]
            else:
                expr += self.x[u]
        self.trust_constr = self.model.addConstr(
            expr <= int(self.trust_radius),
            name=f"trust_region_r{int(self.trust_radius)}",
        )

    def _update_trust_region(self, sol: MasterSolution, improved: bool) -> None:
        if not self.use_trust_region:
            return
        if self.trust_center_x is None or improved:
            self.trust_center_x = dict(sol.x)
            if self.trust_radius is not None and self.trust_radius_max is not None:
                self.trust_radius = min(self.trust_radius_max, int(self.trust_radius) + 1)
            return
        if self.trust_radius is not None:
            self.trust_radius = max(self.trust_radius_min, int(self.trust_radius) - 1)

    def _unrestricted_master_solution(self) -> Optional[Tuple[MasterSolution, float]]:
        if not self.use_trust_region or self.trust_constr is None:
            return None
        assert self.model is not None

        saved = self.trust_constr
        self.model.remove(saved)
        self.trust_constr = None
        self.model.update()
        start = time.perf_counter()
        self.model.optimize()
        self.global_check_time += time.perf_counter() - start
        if self.model.SolCount == 0:
            self._refresh_trust_region()
            return None
        sol = self._solution()
        bound = float(self.model.ObjBound)
        self._refresh_trust_region()
        return sol, bound

    def _solution(self) -> MasterSolution:
        assert self.model is not None
        x_hat = {u: float(self.x[u].X) for u in self.data.N_x}
        y_hat = {(i, l): float(self.y[i, l].X) for l, i in self.data.A_LH}
        z_hat = {(i, c): float(self.z[i, c].X) for i in self.data.H for c in self.data.C}
        return MasterSolution(
            x=x_hat,
            y=y_hat,
            z=z_hat,
            Gamma=float(self.Gamma.X),
            theta=float(self.Theta.X),
            obj=float(self.model.ObjVal),
        )

    def _add_cut(self, duals: Dict[str, Any], name: str) -> None:
        assert self.model is not None
        rhs = RegimeIIDualObjective.expr(self.data, duals, self.x, self.y, self.z, self.Gamma, True)
        self.model.addConstr(self.Theta <= rhs, name=name)

    def _solve_subproblem(self, sol: MasterSolution):
        if self.subproblem_mode == "primal":
            solver = RegimeIIPrimalSubproblem(self.data, output=False)
            status, obj, cut_data = solver.solve(sol.x, sol.y, sol.z, sol.Gamma)
            if status == GRB.OPTIMAL and cut_data is not None:
                return status, obj, cut_data

            fallback = RegimeIIDualSubproblem(self.data, output=False)
            d_status, d_obj, d_cut_data = fallback.solve(sol.x, sol.y, sol.z, sol.Gamma)
            if d_status == GRB.OPTIMAL and d_cut_data is not None:
                d_cut_data["source"] = f"primal_status_{status}_fallback_dual"
                return d_status, d_obj, d_cut_data
            return status, obj, cut_data

        solver = RegimeIIDualSubproblem(self.data, output=False)
        status, obj, cut_data = solver.solve(sol.x, sol.y, sol.z, sol.Gamma)
        if status == GRB.OPTIMAL and cut_data is not None:
            return status, obj, cut_data

        fallback = RegimeIIPrimalSubproblem(self.data, output=False)
        p_status, p_obj, p_cut_data = fallback.solve(sol.x, sol.y, sol.z, sol.Gamma)
        if p_status == GRB.OPTIMAL and p_cut_data is not None:
            p_cut_data["source"] = f"dual_fallback_{p_cut_data.get('source', 'primal')}"
            return p_status, p_obj, p_cut_data

        return status, obj, cut_data

    def solve(self) -> Optional[float]:
        self._build_master()
        assert self.model is not None
        dual_solver_for_pareto = RegimeIIDualSubproblem(self.data, output=False)
        self._load_initial_solution()

        for it in range(1, self.max_iter + 1):
            t0 = time.perf_counter()
            self._refresh_trust_region()
            self.model.optimize()
            if self.model.SolCount == 0:
                if self.verbose:
                    print("[Benders-RII] master has no incumbent.")
                self.stop_reason = "master_no_incumbent"
                return None

            sol = self._solution()
            sub_start = time.perf_counter()
            status, sub_obj, cut_data = self._solve_subproblem(sol)
            sub_elapsed = time.perf_counter() - sub_start
            self.subproblem_time += sub_elapsed
            if status != GRB.OPTIMAL or cut_data is None or sub_obj is None:
                if self.verbose:
                    print(f"[Benders-RII] subproblem status={status}; adding no-good fallback.")
                self._add_nogood(sol, f"nogood_{it}")
                continue

            chosen_cut = cut_data
            mw_elapsed = 0.0
            if self.use_pareto:
                mw_start = time.perf_counter()
                mw_status, mw_obj, mw_cd = dual_solver_for_pareto.solve_pareto(
                    sol.x,
                    sol.y,
                    sol.z,
                    sol.Gamma,
                    self.core_x,
                    self.core_y,
                    self.core_z,
                    self.core_gamma,
                    cut_data.get("rhs_at_hat", sub_obj),
                )
                mw_elapsed = time.perf_counter() - mw_start
                self.mw_time += mw_elapsed
                if mw_status == GRB.OPTIMAL and mw_cd is not None:
                    chosen_cut = mw_cd

            self._add_cut(chosen_cut["duals"], f"benders_{it}_{chosen_cut.get('source', self.subproblem_mode)}")
            fixed = self.data.fixed_cost_value(sol.x, sol.z)
            eval_sub_obj = sub_obj
            if self.subproblem_mode == "dual":
                p_status, p_obj, _ = RegimeIIPrimalSubproblem(self.data, output=False).solve(
                    sol.x,
                    sol.y,
                    sol.z,
                    sol.Gamma,
                )
                if p_status == GRB.OPTIMAL and p_obj is not None:
                    eval_sub_obj = p_obj

            true_obj = eval_sub_obj - fixed - self.data.gamma_cost * sol.Gamma
            improved = true_obj > self.best_obj + 1e-6
            if true_obj > self.best_obj:
                self.best_obj = true_obj
                self.best_solution = sol
            self._update_trust_region(sol, improved)

            master_bound = float(self.model.ObjBound) if self.model.Status in {GRB.OPTIMAL, GRB.TIME_LIMIT} else sol.obj
            gap = (master_bound - self.best_obj) / max(1.0, abs(self.best_obj))
            global_bound = None
            global_gap = None
            global_cut_source = None
            global_sub_obj = None
            global_status = None
            if self.use_trust_region and gap <= self.gap_tol:
                global_result = self._unrestricted_master_solution()
                if global_result is not None:
                    global_sol, global_bound = global_result
                    global_gap = (global_bound - self.best_obj) / max(1.0, abs(self.best_obj))
                    if global_gap > self.gap_tol:
                        global_sub_start = time.perf_counter()
                        g_status, g_sub_obj, g_cut_data = self._solve_subproblem(global_sol)
                        global_sub_elapsed = time.perf_counter() - global_sub_start
                        self.subproblem_time += global_sub_elapsed
                        global_status = g_status
                        if g_status == GRB.OPTIMAL and g_cut_data is not None and g_sub_obj is not None:
                            global_chosen_cut = g_cut_data
                            if self.use_pareto:
                                mw_start = time.perf_counter()
                                g_mw_status, _, g_mw_cd = dual_solver_for_pareto.solve_pareto(
                                    global_sol.x,
                                    global_sol.y,
                                    global_sol.z,
                                    global_sol.Gamma,
                                    self.core_x,
                                    self.core_y,
                                    self.core_z,
                                    self.core_gamma,
                                    g_cut_data.get("rhs_at_hat", g_sub_obj),
                                )
                                g_mw_elapsed = time.perf_counter() - mw_start
                                self.mw_time += g_mw_elapsed
                                if g_mw_status == GRB.OPTIMAL and g_mw_cd is not None:
                                    global_chosen_cut = g_mw_cd
                            self._add_cut(
                                global_chosen_cut["duals"],
                                f"global_benders_{it}_{global_chosen_cut.get('source', self.subproblem_mode)}",
                            )
                            global_cut_source = global_chosen_cut.get("source")
                            global_sub_obj = g_sub_obj
                            global_true_obj = (
                                g_sub_obj
                                - self.data.fixed_cost_value(global_sol.x, global_sol.z)
                                - self.data.gamma_cost * global_sol.Gamma
                            )
                            if global_true_obj > self.best_obj + 1e-6:
                                self.best_obj = global_true_obj
                                self.best_solution = global_sol
                                self._update_trust_region(global_sol, True)
                        else:
                            self._add_nogood(global_sol, f"global_nogood_{it}_{g_status}")
                            global_cut_source = f"nogood_status_{g_status}"
            row = {
                "iter": it,
                "master_obj": sol.obj,
                "master_bound": master_bound,
                "global_bound": global_bound,
                "theta": sol.theta,
                "sub_obj": sub_obj,
                "eval_sub_obj": eval_sub_obj,
                "true_obj": true_obj,
                "best_obj": self.best_obj,
                "gap": gap,
                "global_gap": global_gap,
                "sub_source": cut_data.get("source"),
                "cut_source": chosen_cut.get("source"),
                "global_cut_source": global_cut_source,
                "global_sub_obj": global_sub_obj,
                "global_status": global_status,
                "rhs_at_hat": chosen_cut.get("rhs_at_hat"),
                "subproblem_time": sub_elapsed,
                "mw_time": mw_elapsed,
                "trust_radius": self.trust_radius if self.use_trust_region else None,
                "trust_active": self.use_trust_region and self.trust_center_x is not None,
                "time": time.perf_counter() - t0,
            }
            self.history.append(row)
            if self.verbose:
                trust_msg = ""
                if self.use_trust_region:
                    trust_msg = f" tr={self.trust_radius}"
                print(
                    "[Benders-RII] "
                    f"it={it} master={sol.obj:.4f} theta={sol.theta:.4f} "
                    f"sub={sub_obj:.4f} true={true_obj:.4f} best={self.best_obj:.4f} "
                    f"gap={gap:.2%} cut={row['cut_source']}{trust_msg}"
                )

            self._update_core(sol)
            opt_tol = 1e-4 + 1e-7 * max(1.0, abs(float(sub_obj)))
            if sol.theta <= sub_obj + opt_tol and gap <= self.gap_tol:
                if self.use_trust_region and (global_gap is None or global_gap > self.gap_tol):
                    continue
                self.converged = True
                self.stop_reason = "gap_tol"
                break

        if self.stop_reason is None:
            self.stop_reason = "max_iter"
        return self.best_obj if self.best_obj > -math.inf else None

    def _add_nogood(self, sol: MasterSolution, name: str) -> None:
        assert self.model is not None
        ones = [self.x[u] for u, val in sol.x.items() if val > 0.5]
        zeros = [self.x[u] for u, val in sol.x.items() if val <= 0.5]
        self.model.addConstr(gp.quicksum(1 - v for v in ones) + gp.quicksum(zeros) >= 1, name=name)


def run_benders_regime_ii(
    instance: Instance,
    subproblem: str = "dual",
    use_pareto: bool = True,
    max_iter: int = 20,
    master_time_limit: float = 20.0,
    master_mip_gap: float = 0.2,
    verbose: bool = True,
    use_trust_region: bool = False,
    trust_radius: Optional[int] = None,
    trust_radius_min: int = 1,
    trust_radius_max: Optional[int] = None,
    initial_solution: Optional[Any] = None,
) -> RegimeIIBenders:
    solver = RegimeIIBenders(
        instance,
        subproblem=subproblem,
        use_pareto=use_pareto,
        max_iter=max_iter,
        master_time_limit=master_time_limit,
        master_mip_gap=master_mip_gap,
        verbose=verbose,
        use_trust_region=use_trust_region,
        trust_radius=trust_radius,
        trust_radius_min=trust_radius_min,
        trust_radius_max=trust_radius_max,
        initial_solution=initial_solution,
    )
    solver.solve()
    return solver
