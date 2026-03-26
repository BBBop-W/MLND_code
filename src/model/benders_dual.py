"""Benders decomposition - explicit dual SOCP subproblem approach.

This version keeps two persistent models:
- sub_model: regular dual subproblem for incumbent evaluation.
- mw_model: Magnanti-Wong (MW) subproblem.

Both models are built once and updated by objective/constraints for each call.
"""

import math
import time
from typing import Any, Dict, List, Optional, Tuple

import gurobipy as gp
from gurobipy import GRB

from src import config
from src.data_processor.instance import Instance
from src.model.benders_main import IndexSets

XHat = Dict[str, int]
EtaHat = Dict[Tuple[str, int], int]


class DualSubproblem:
    """Explicit SOCP dual of the primal subproblem with model reuse."""

    def __init__(self, instance: Instance, data: IndexSets):
        self.inst = instance
        self.data = data

        self.sub_model: Optional[gp.Model] = None
        self.sub_vars: Dict[str, Any] = {}

        self.mw_model: Optional[gp.Model] = None
        self.mw_vars: Dict[str, Any] = {}
        self._mw_anchor_lb_constr: Optional[gp.Constr] = None
        self._mw_anchor_ub_constr: Optional[gp.Constr] = None
        self.mw_anchor_abs_tol: float = 1
        self.mw_anchor_rel_tol: float = 1e-3

        self.w_keys: List[Tuple[str, str, str]] = [
            (l, i, j) for l, i in data.A_LP for j in data.D if (i, j) in data.A_PD_set
        ]
        self.z_keys: List[Tuple[str, str, int]] = [(l, i, s) for l, i in data.A_LP for s in data.S]

        self.stats: Dict[str, float] = {
            "sub_calls": 0,
            "sub_optimal": 0,
            "sub_nonoptimal": 0,
            "sub_total_time": 0.0,
            "mw_calls": 0,
            "mw_optimal": 0,
            "mw_nonoptimal": 0,
            "mw_total_time": 0.0,
        }

    @staticmethod
    def build_objective_expr(
        data: IndexSets,
        inst: Instance,
        x_ref,
        eta_ref,
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
        """Build dual objective expression in numeric or symbolic form."""

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
            epsilon2[l, i, s] * data.H.get(i, 0.0) * eta_ref[l, s]
            for l, i, s in z_keys
        )
        obj += sum_terms(
            epsilon3[l, i, s] * data.H.get(i, 0.0) * (1 - eta_ref[l, s])
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

        obj += -kappa1 * sum_terms(q_ref[l] for l in data.L_hat)

        return obj

    def _build_single_model(self, model_name: str) -> Tuple[gp.Model, Dict[str, Any]]:
        data = self.data
        inst = self.inst

        m = gp.Model(model_name)
        m.setParam("OutputFlag", 0)
        m.setParam("DualReductions", 0)

        alpha1 = m.addVars(data.P, lb=0.0, name="alpha1")
        alpha2 = m.addVars(data.A_LP, lb=0.0, name="alpha2")
        alpha3 = m.addVars(self.w_keys, lb=0.0, name="alpha3")
        alpha4 = m.addVars(self.w_keys, lb=0.0, name="alpha4")
        alpha5 = m.addVars(self.w_keys, lb=0.0, name="alpha5")
        alpha6 = m.addVars(data.A_LP, lb=-GRB.INFINITY, name="alpha6")

        epsilon1 = m.addVars(self.z_keys, lb=0.0, name="epsilon1")
        epsilon2 = m.addVars(self.z_keys, lb=0.0, name="epsilon2")
        epsilon3 = m.addVars(self.z_keys, lb=0.0, name="epsilon3")
        epsilon4 = m.addVars(data.A_LP, lb=0.0, name="epsilon4")

        beta1 = m.addVars(data.N + data.L_v, lb=-GRB.INFINITY, name="beta1")
        beta2 = m.addVars(data.A_PD, lb=0.0, name="beta2")
        beta3 = m.addVars(data.A, lb=-GRB.INFINITY, name="beta3")

        pi1 = m.addVars(data.L_v, lb=0.0, name="pi1")
        pi2 = m.addVars(data.P + data.D, lb=0.0, name="pi2")
        pi3 = m.addVars(data.E, lb=0.0, name="pi3")

        delta1 = m.addVars(data.P + data.D, lb=-GRB.INFINITY, name="delta1")
        delta2 = m.addVars(data.P + data.D, lb=-GRB.INFINITY, name="delta2")
        Delta = m.addVars(data.P + data.D, lb=0.0, name="Delta")

        sigma1 = m.addVars(data.D, lb=0.0, name="sigma1")
        sigma2 = m.addVars(data.P, lb=0.0, name="sigma2")

        kappa1 = m.addVar(lb=0.0, name="kappa1")
        kappa3 = m.addVar(lb=0.0, name="kappa3")

        for l, i in data.A_LP:
            expr = alpha1[i] + alpha2[l, i]
            expr += gp.quicksum(
                alpha5[l, i, j] - alpha3[l, i, j]
                for j in data.D
                if (i, j) in data.A_PD_set
            )
            m.addConstr(expr >= 0, name=f"dual_y_{l}_{i}")

        for l, i, j in self.w_keys:
            expr = alpha3[l, i, j] + alpha4[l, i, j] - alpha5[l, i, j]
            expr -= inst.h[i, j] * alpha6[l, i]
            m.addConstr(expr >= 0, name=f"dual_w_{l}_{i}_{j}")

        for l, i in data.A_LP:
            expr = alpha6[l, i]
            expr -= gp.quicksum(epsilon1[l, i, s] for s in data.S)
            expr += gp.quicksum(epsilon3[l, i, s] for s in data.S)
            m.addConstr(expr >= 0, name=f"dual_D_{l}_{i}")

        for l, i, s in self.z_keys:
            expr = epsilon1[l, i, s] + epsilon2[l, i, s] - epsilon3[l, i, s]
            expr -= data.rho[s] * epsilon4[l, i]
            m.addConstr(expr >= 0, name=f"dual_z_{l}_{i}_{s}")

        for u_node, v_node in data.A:
            expr = gp.LinExpr()
            if (u_node, v_node) in data.A_LP_set:
                expr += epsilon4[u_node, v_node]
                if u_node in data.L_v:
                    expr += -config.MAX_WAITING_FACTOR_LV * pi1[u_node]

            expr += beta1[u_node] - beta1[v_node]
            expr += beta3[u_node, v_node]

            if (u_node, v_node) in data.A_PD_set:
                expr += beta2[u_node, v_node]

            if u_node in data.P + data.D:
                expr += pi2[u_node]
                expr += Delta[u_node] - delta2[u_node]

            rhs = config.ORDER_REVENUE_COEF if (u_node, v_node) in data.A_PD_set else 0.0
            m.addConstr(expr >= rhs, name=f"dual_psi_{u_node}_{v_node}")

        for u_node, v_node, k in data.phi_keys:
            expr = -beta3[u_node, v_node]
            if k in data.E:
                expr += pi3[k]
            expr += data.path_time(u_node, v_node, k) * kappa1
            m.addConstr(expr >= 0, name=f"dual_phi_{u_node}_{v_node}_{k}")

        for u in data.P + data.D:
            expr = -Delta[u] - delta2[u] + kappa1
            m.addConstr(expr >= 0, name=f"dual_xi_{u}")

        m.addConstr(-kappa1 >= -config.COST_DRONE, name="dual_Gamma")

        for u in data.P + data.D:
            m.addQConstr(
                delta1[u] * delta1[u] + delta2[u] * delta2[u] <= Delta[u] * Delta[u],
                name=f"dual_soc_{u}",
            )

        vars_map = {
            "alpha1": alpha1,
            "alpha2": alpha2,
            "alpha3": alpha3,
            "alpha4": alpha4,
            "alpha5": alpha5,
            "alpha6": alpha6,
            "epsilon1": epsilon1,
            "epsilon2": epsilon2,
            "epsilon3": epsilon3,
            "epsilon4": epsilon4,
            "beta1": beta1,
            "beta2": beta2,
            "beta3": beta3,
            "pi1": pi1,
            "pi2": pi2,
            "pi3": pi3,
            "delta1": delta1,
            "delta2": delta2,
            "Delta": Delta,
            "sigma1": sigma1,
            "sigma2": sigma2,
            "kappa1": kappa1,
            "kappa3": kappa3,
        }
        return m, vars_map

    def _ensure_sub_model(self) -> None:
        if self.sub_model is None:
            self.sub_model, self.sub_vars = self._build_single_model("DualSub")

    def _ensure_mw_model(self) -> None:
        if self.mw_model is None:
            self.mw_model, self.mw_vars = self._build_single_model("DualMW")

    @staticmethod
    def _squared_norm_expr(vars_map: Dict[str, Any]) -> gp.QuadExpr:
        quad = gp.QuadExpr()
        for container in vars_map.values():
            if hasattr(container, "values"):
                for var in container.values():
                    quad += var * var
            else:
                quad += container * container
        return quad

    def _set_objective(
        self,
        model: gp.Model,
        vars_map: Dict[str, Any],
        x_ref: XHat,
        eta_ref: EtaHat,
        reg_weight: float = 0.0,
    ) -> None:
        q_ref = {loc: self.data.service_wait_value(eta_ref, [loc]) for loc in self.data.L_hat}
        obj = self.build_objective_expr(
            data=self.data,
            inst=self.inst,
            x_ref=x_ref,
            eta_ref=eta_ref,
            q_ref=q_ref,
            alpha1=vars_map["alpha1"],
            alpha2=vars_map["alpha2"],
            alpha4=vars_map["alpha4"],
            alpha5=vars_map["alpha5"],
            epsilon2=vars_map["epsilon2"],
            epsilon3=vars_map["epsilon3"],
            beta2=vars_map["beta2"],
            pi1=vars_map["pi1"],
            pi2=vars_map["pi2"],
            pi3=vars_map["pi3"],
            delta1=vars_map["delta1"],
            delta2=vars_map["delta2"],
            Delta=vars_map["Delta"],
            sigma1=vars_map["sigma1"],
            kappa1=vars_map["kappa1"],
            kappa3=vars_map["kappa3"],
            use_gurobi_expr=True,
        )
        if reg_weight > 0.0:
            obj += reg_weight * self._squared_norm_expr(vars_map)
        model.setObjective(obj, GRB.MINIMIZE)

    def _set_mw_anchor(self, x_hat: XHat, eta_hat: EtaHat, z_star: float) -> None:
        assert self.mw_model is not None
        q_hat = {loc: self.data.service_wait_value(eta_hat, [loc]) for loc in self.data.L_hat}
        v = self.mw_vars
        anchor_expr = self.build_objective_expr(
            data=self.data,
            inst=self.inst,
            x_ref=x_hat,
            eta_ref=eta_hat,
            q_ref=q_hat,
            alpha1=v["alpha1"],
            alpha2=v["alpha2"],
            alpha4=v["alpha4"],
            alpha5=v["alpha5"],
            epsilon2=v["epsilon2"],
            epsilon3=v["epsilon3"],
            beta2=v["beta2"],
            pi1=v["pi1"],
            pi2=v["pi2"],
            pi3=v["pi3"],
            delta1=v["delta1"],
            delta2=v["delta2"],
            Delta=v["Delta"],
            sigma1=v["sigma1"],
            kappa1=v["kappa1"],
            kappa3=v["kappa3"],
            use_gurobi_expr=True,
        )

        if self._mw_anchor_lb_constr is not None:
            self.mw_model.remove(self._mw_anchor_lb_constr)
        if self._mw_anchor_ub_constr is not None:
            self.mw_model.remove(self._mw_anchor_ub_constr)
        eps = self.mw_anchor_abs_tol + self.mw_anchor_rel_tol * max(1.0, abs(float(z_star)))
        self._mw_anchor_lb_constr = self.mw_model.addConstr(
            anchor_expr >= z_star - eps,
            name="mw_anchor_lb",
        )
        self._mw_anchor_ub_constr = self.mw_model.addConstr(
            anchor_expr <= z_star + eps,
            name="mw_anchor_ub",
        )

    @staticmethod
    def _extract_values(container: Any) -> Any:
        if hasattr(container, "items"):
            return {k: v.X for k, v in container.items()}
        return container.X

    def _collect_duals(self, vars_map: Dict[str, Any]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for name, container in vars_map.items():
            out[name] = self._extract_values(container)
        return out

    def solve(self, x_hat: XHat, eta_hat: EtaHat) -> Tuple[int, Optional[float], Optional[Dict[str, Any]]]:
        """Solve regular dual subproblem on persistent sub_model."""
        self._ensure_sub_model()
        assert self.sub_model is not None

        t0 = time.perf_counter()
        self.stats["sub_calls"] += 1

        self._set_objective(self.sub_model, self.sub_vars, x_hat, eta_hat)
        self.sub_model.optimize()

        self.stats["sub_total_time"] += time.perf_counter() - t0

        if self.sub_model.Status != GRB.OPTIMAL:
            self.stats["sub_nonoptimal"] += 1
            return self.sub_model.Status, None, {
                "kind": "unbounded",
                "x_hat": dict(x_hat),
                "eta_hat": dict(eta_hat),
            }

        self.stats["sub_optimal"] += 1
        obj_val = self.sub_model.ObjVal
        duals = self._collect_duals(self.sub_vars)
        cut_data = {
            "kind": "optimality",
            "z_star": obj_val,
            "x_hat": dict(x_hat),
            "eta_hat": dict(eta_hat),
            "duals": duals,
        }
        return self.sub_model.Status, obj_val, cut_data

    def solve_mw(
        self,
        x_hat: XHat,
        eta_hat: EtaHat,
        core_x: Dict[str, float],
        core_eta: Dict[Tuple[str, int], float],
        z_star: float,
    ) -> Tuple[int, Optional[float], Optional[Dict[str, Any]]]:
        """Solve Magnanti-Wong subproblem on persistent mw_model."""
        self._ensure_mw_model()
        assert self.mw_model is not None

        t0 = time.perf_counter()
        self.stats["mw_calls"] += 1

        self._set_mw_anchor(x_hat, eta_hat, z_star)
        # Keep MW always stabilized to avoid objective-face rays.
        self._set_objective(self.mw_model, self.mw_vars, core_x, core_eta, reg_weight=1e-6)
        self.mw_model.optimize()

        self.stats["mw_total_time"] += time.perf_counter() - t0

        if self.mw_model.Status != GRB.OPTIMAL:
            self.stats["mw_nonoptimal"] += 1
            return self.mw_model.Status, None, None

        self.stats["mw_optimal"] += 1
        mw_obj = self.mw_model.ObjVal
        duals = self._collect_duals(self.mw_vars)
        cut_data = {
            "kind": "optimality",
            "z_star": z_star,
            "x_hat": dict(x_hat),
            "eta_hat": dict(eta_hat),
            "duals": duals,
            "mw_obj": mw_obj,
        }
        return self.mw_model.Status, mw_obj, cut_data

    def solve_mw_independent(
        self,
        core_x: Dict[str, float],
        core_eta: Dict[Tuple[str, int], float],
    ) -> Tuple[int, Optional[float], Optional[Dict[str, Any]]]:
        """Solve independent MW subproblem using the regular subproblem model at core point."""
        self._ensure_sub_model()
        assert self.sub_model is not None

        t0 = time.perf_counter()
        self.stats["mw_calls"] += 1

        self._set_objective(
            self.sub_model,
            self.sub_vars,
            core_x,
            core_eta,
            reg_weight=1e-6,
        )
        self.sub_model.optimize()

        self.stats["mw_total_time"] += time.perf_counter() - t0

        if self.sub_model.Status != GRB.OPTIMAL:
            self.stats["mw_nonoptimal"] += 1
            return self.sub_model.Status, None, None

        self.stats["mw_optimal"] += 1
        mw_obj = self.sub_model.ObjVal
        duals = self._collect_duals(self.sub_vars)
        cut_data = {
            "kind": "optimality",
            "z_star": mw_obj,
            "x_hat": dict(core_x),
            "eta_hat": dict(core_eta),
            "duals": duals,
            "mw_obj": mw_obj,
        }
        return self.sub_model.Status, mw_obj, cut_data

    def get_stats(self) -> Dict[str, float]:
        stats = dict(self.stats)
        stats["sub_avg_time"] = (
            stats["sub_total_time"] / stats["sub_calls"] if stats["sub_calls"] > 0 else 0.0
        )
        stats["mw_avg_time"] = (
            stats["mw_total_time"] / stats["mw_calls"] if stats["mw_calls"] > 0 else 0.0
        )
        return stats
