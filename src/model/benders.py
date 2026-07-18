"""Benders decomposition for the Regime-II MLND model.

This file is intended to be placed next to the original Gurobi model file.
It uses the same ``src.config`` and ``src.data_processor.instance.Instance``
objects as the monolithic model.

Implementation choice
---------------------
I use an explicit multi-tree Benders loop instead of a lazy-callback
implementation.  The reason is practical: each Benders iteration solves a
continuous SOCP subproblem and then, optionally, a Pareto/core-point SOCP.
Solving conic subproblems inside a Gurobi MIP callback is possible in principle
but is much less stable and much harder to debug.  The explicit loop also makes
all generated cuts visible and reproducible.

Cut strategy
------------
The master contains only strategic variables x, y, z, Gamma and a recourse
upper-estimator eta.  Given a master solution, the subproblem fixes the design
variables and optimizes all continuous operational variables.  Gurobi QCP duals
are used to construct conic Benders optimality cuts.

In addition to the incumbent cut, the solver can generate a core-point
Pareto-stabilized cut.  This is an in-out / Magnanti-Wong-style stabilization:
we solve an additional SOCP at a core reference design and add the resulting
valid Benders cut.  For convex-conic subproblems, this cut is globally valid;
it is often much stronger than relying on the single incumbent cut only.

Important note
--------------
Gurobi returns dual values for convex QCP/SOCP models only when QCPDual=1.
The subproblem is therefore solved with QCPDual=1.  If Gurobi reports that QCP
duals are unavailable for your version/settings, set USE_PARETO_CUTS=False and
fall back to ordinary Benders cuts, or replace the quadratic SOC constraints by
native cone constraints in a solver that exposes conic duals directly.
"""

from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Tuple, Union

# Allow absolute imports when running this file directly from a model folder.
if __name__ == "__main__":
    _proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if _proj_root not in sys.path:
        sys.path.insert(0, _proj_root)

import gurobipy as gp
from gurobipy import GRB

from src import config
from src.data_processor.instance import Instance


Number = Union[int, float]
Node = Union[str, int]
Arc = Tuple[Node, Node]
RouteKey = Tuple[Node, Node, Node]
AlphaKey = Tuple[Node, Node, Node]


@dataclass
class SubproblemResult:
    """Container for one solved SOCP subproblem."""

    status: int
    obj_value: Optional[float]
    cut_expr_data: Optional[List[Tuple[float, Callable[[Mapping[str, object]], gp.LinExpr]]]]
    rhs_at_reference: Optional[List[float]]
    design: Mapping[str, object]
    label: str


class BendersRegimeIISolver:
    """Branch-and-Benders-cut solver for the Regime-II MLND model.

    Parameters
    ----------
    instance:
        The generated/loaded MLND instance.
    use_pareto_cuts:
        Whether to generate an additional core-point Pareto-stabilized cut.
        This is strongly recommended for this model because a single aggregate
        eta cut is usually weak.
    pareto_lambda:
        Weight of the incumbent design in the stabilized reference point:
        x_ref = lambda * x_incumbent + (1-lambda) * x_core.
        Values in [0.3, 0.7] usually work.  Smaller values emphasize the core.
    core_update_weight:
        Core-point update: core <- w * core + (1-w) * incumbent.
    """

    def __init__(
        self,
        instance: Instance,
        use_pareto_cuts: bool = True,
        pareto_lambda: float = 0.5,
        core_update_weight: float = 0.7,
    ):
        self.inst = instance
        self.use_pareto_cuts = bool(use_pareto_cuts)
        self.pareto_lambda = float(pareto_lambda)
        self.core_update_weight = float(core_update_weight)

        # Sets copied from the monolithic model.
        self.L = list(self.inst.L)
        self.H = list(self.inst.H)
        self.S = list(self.inst.S)
        self.E = list(self.inst.E)
        self.N_x = self.L + self.H + self.E
        self.HLS = self.L + self.H + self.S

        self.A = sorted(self.inst.A)
        self.A_HH = sorted(self.inst.A_HH)
        self.A_LH = sorted(self.inst.A_LH)
        self.A_HS = sorted(self.inst.A_HS)
        self.A_SH = sorted(self.inst.A_SH)
        self.A_SL = sorted(self.inst.A_SL)
        self.A_SS = sorted(self.inst.A_SS)
        self.A_F = sorted(self.inst.A_F)
        self.A_virtual = sorted(set(self.A_HS) | set(self.A_SH))
        self.A_hat = sorted(set(self.A) - set(self.A_virtual))

        self.route_options: Dict[Arc, List[Node]] = {}
        self.t: Dict[RouteKey, float] = {}
        self.flight_cost: Dict[RouteKey, float] = {}
        self.route_big_m: Dict[RouteKey, float] = {}
        self._build_route_options()

        self.AY: List[Arc] = []
        self.arc_cost: Dict[Arc, float] = {}
        self._build_y_arcs()

        self.master: Optional[gp.Model] = None
        self.master_vars: Dict[str, object] = {}
        self.cut_count = 0
        self.core_point: Optional[Dict[str, object]] = None
        self.best_design: Optional[Dict[str, object]] = None
        self.best_total_obj = -GRB.INFINITY
        self.best_operational_obj = -GRB.INFINITY
        self.history: List[Dict[str, Number]] = []
        self.zero_flow_signature_counts: Dict[Tuple[int, ...], int] = {}
        self.initial_designs: List[Dict[str, object]] = []
        self._initial_designs_processed = False

    # ------------------------------------------------------------------
    # Basic data helpers
    # ------------------------------------------------------------------

    def _build_route_options(self) -> None:
        flight_cost_rate = float(getattr(config, "COST_FLIGHT_TIME", 0.1))
        for u, v in self.A_F:
            options: List[Node] = []
            if self.inst.get_dist(u, v) <= config.MAX_DIRECT_DISTANCE:
                options.append("0")
                self.t[u, v, "0"] = float(self.inst.get_time(u, v))

            for k in self.inst.E_uv.get((u, v), []):
                options.append(k)
                self.t[u, v, k] = float(self.inst.get_time(u, k) + self.inst.get_time(k, v))

            # Deterministic order and duplicate removal.
            seen = set()
            ordered_options: List[Node] = []
            for k in options:
                if k not in seen:
                    ordered_options.append(k)
                    seen.add(k)

            self.route_options[u, v] = ordered_options
            for k in ordered_options:
                self.flight_cost[u, v, k] = flight_cost_rate * self.t[u, v, k]
                self.route_big_m[u, v, k] = self._route_flow_big_m(u, v, k)

    def _build_y_arcs(self) -> None:
        y_arcs = set(self.A_hat)
        for u, v in self.A_hat:
            for k in self.inst.E_uv.get((u, v), []):
                y_arcs.add((u, k))
                y_arcs.add((k, v))
        self.AY = sorted(y_arcs)
        for u, v in self.AY:
            self.arc_cost[u, v] = self._arc_activation_cost(u, v)

    def _route_flow_big_m(self, u: Node, v: Node, k: Node) -> float:
        del u, v, k
        total_demand = sum(float(value) for value in self.inst.h.values())
        configured = float(getattr(config, "MAX_FLOW_E", 0.0))
        return max(1.0, configured, total_demand)

    def _arc_activation_cost(self, u: Node, v: Node) -> float:
        cost = getattr(config, "COST_ARC", 0.0)
        if isinstance(cost, dict):
            return float(cost.get((u, v), cost.get(f"{u},{v}", cost.get(f"{u}->{v}", 0.0))))
        return float(cost)

    def _hub_pool(self, h: Node) -> Node:
        return self.inst.h_to_s[h]

    def _p_chain(self, hub: Node) -> float:
        p_by_hub = getattr(config, "P_CHAIN_BY_HUB", None)
        if isinstance(p_by_hub, dict):
            return float(p_by_hub.get(hub, config.P_CHAIN))
        return float(config.P_CHAIN)

    def _node_activation_value(self, design: Mapping[str, object], node: Node) -> float:
        """Return the fixed activation value of a node in a subproblem."""
        x_val = design["x"]
        if node in self.S:
            return float(x_val[self.inst.s_to_h[node]])
        return float(x_val[node])

    def _node_activation_expr(self, master_vars: Mapping[str, object], node: Node) -> gp.Var:
        """Return the master activation variable/expression of a node."""
        x = master_vars["x"]
        if node in self.S:
            return x[self.inst.s_to_h[node]]
        return x[node]

    # ------------------------------------------------------------------
    # Master problem
    # ------------------------------------------------------------------

    def build_master(self) -> None:
        if self.master is not None:
            return

        m = gp.Model("MLND_RegimeII_Benders_Master")
        m.setParam("OutputFlag", int(getattr(config, "BENDERS_MASTER_LOG", 0)))
        m.setParam("Threads", int(getattr(config, "BENDERS_MASTER_THREADS", 1)))
        if getattr(config, "BENDERS_USE_INTEGER_REFERENCE_CUTS", False):
            pool_solutions = int(getattr(config, "BENDERS_REFERENCE_POOL_SOLUTIONS", 4))
            if pool_solutions > 1:
                m.setParam("PoolSolutions", pool_solutions)
                m.setParam("PoolSearchMode", int(getattr(config, "BENDERS_REFERENCE_POOL_SEARCH_MODE", 2)))
                pool_gap = getattr(config, "BENDERS_REFERENCE_POOL_GAP", None)
                if pool_gap is not None:
                    m.setParam("PoolGap", float(pool_gap))

        C = range(len(config.MU_HUB))
        mu_c = config.MU_HUB
        tau_H = float(config.TAU_H)

        x = m.addVars(self.N_x, vtype=GRB.BINARY, name="x")
        y = m.addVars(self.AY, vtype=GRB.BINARY, name="y")
        z = m.addVars(self.H, C, vtype=GRB.BINARY, name="z")
        Gamma = m.addVar(lb=0.0, vtype=GRB.INTEGER, name="Gamma")
        eta = m.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="eta")

        # Fixed-cost part of the objective.
        fixed_cost = gp.quicksum(float(self.inst.f.get(l, config.COST_BASE)) * x[l] for l in self.L)
        fixed_cost += gp.quicksum(float(getattr(config, "COST_HUB_FIXED", 0.0)) * x[i] for i in self.H)
        fixed_cost += gp.quicksum(float(self.inst.f.get(k, config.COST_SWAPPING)) * x[k] for k in self.E)
        fixed_cost += gp.quicksum(float(config.COST_HUB[c]) * z[i, c] for i in self.H for c in C)
        arc_activation_cost = gp.quicksum(self.arc_cost[u, v] * y[u, v] for (u, v) in self.AY)
        fleet_cost = float(config.COST_DRONE) * Gamma
        m.setObjective(eta - fixed_cost - arc_activation_cost - fleet_cost, GRB.MAXIMIZE)

        # First-stage assignment and activation constraints.
        for i in self.H:
            assigned_bases = [l for (l, h) in self.A_LH if h == i]
            m.addConstr(gp.quicksum(y[l, i] for l in assigned_bases if (l, i) in y) == x[i], name=f"assign_{i}")

        for l, i in self.A_LH:
            m.addConstr(y[l, i] <= x[l], name=f"assign_active_{l}_{i}")

        for i in self.H:
            m.addConstr(gp.quicksum(z[i, c] for c in C) == x[i], name=f"capacity_{i}")

        for u, v in self.AY:
            if (u, v) in self.A_LH:
                continue
            m.addConstr(y[u, v] <= self._node_activation_expr({"x": x}, u), name=f"arc_tail_active_{u}_{v}")
            m.addConstr(y[u, v] <= self._node_activation_expr({"x": x}, v), name=f"arc_head_active_{u}_{v}")

        # A cheap implied feasibility cut from the hub service constraint with zero workload.
        # Without this, the master may select a capacity level that cannot even satisfy the
        # required slack x_i/tau_H.
        for i in self.H:
            m.addConstr(
                gp.quicksum(float(mu_c[c]) * z[i, c] for c in C) >= x[i] / tau_H,
                name=f"master_hub_min_slack_{i}",
            )

        # Initial eta upper bound.  The operational subproblem maximizes revenue minus
        # flight operating cost, so total potential revenue is a safe upper bound.
        revenue_coef = float(getattr(config, "ORDER_REVENUE_COEF", 0.0))
        eta_ub = revenue_coef * sum(float(self.inst.h.get((i, j), 0.0)) for i, j in self.A_HH)
        m.addConstr(eta <= eta_ub, name="eta_trivial_ub")

        # Stronger but still cheap upper bound: revenue can only be obtained on opened HH arcs.
        # This is valid because operating costs are nonnegative.
        m.addConstr(
            eta <= gp.quicksum(revenue_coef * float(self.inst.h.get((i, j), 0.0)) * y[i, j] for i, j in self.A_HH),
            name="eta_open_loaded_arc_ub",
        )

        od_service_ub = {}
        route_ready = {}
        if getattr(config, "BENDERS_USE_MASTER_RECOURSE_BOUNDS", True):
            od_service_ub, route_ready = self._add_master_recourse_bounds(
                m=m,
                x=x,
                y=y,
                z=z,
                Gamma=Gamma,
                eta=eta,
                revenue_coef=revenue_coef,
                mu_c=mu_c,
                tau_H=tau_H,
            )

        self.master = m
        self.master_vars = {
            "x": x,
            "y": y,
            "z": z,
            "Gamma": Gamma,
            "eta": eta,
            "od_service_ub": od_service_ub,
            "route_ready": route_ready,
        }

    def _add_master_recourse_bounds(
        self,
        m: gp.Model,
        x,
        y,
        z,
        Gamma,
        eta,
        revenue_coef: float,
        mu_c: Iterable[float],
        tau_H: float,
    ) -> Tuple[object, object]:
        """Add master-only upper bounds on operational recourse.

        These variables do not approximate the subproblem objective directly.
        They describe a superset of service that any feasible subproblem can
        realize, which gives valid upper bounds on eta before Benders cuts
        arrive.
        """
        od_service_ub = m.addVars(self.A_HH, lb=0.0, ub=1.0, name="od_service_ub")
        route_ready_keys = [
            (i, j, k)
            for i, j in self.A_HH
            for k in self.route_options.get((i, j), [])
            if k in self.E
        ]
        route_ready = m.addVars(route_ready_keys, lb=0.0, ub=1.0, name="route_ready")

        eta_terms = []
        fleet_terms = []
        for i, j in self.A_HH:
            demand = float(self.inst.h.get((i, j), 0.0))
            options = self.route_options.get((i, j), [])
            ready_terms = []

            if (i, j) in y:
                m.addConstr(od_service_ub[i, j] <= y[i, j], name=f"od_arc_open_ub_{i}_{j}")

            if "0" in options:
                ready_terms.append(y[i, j])

            for k in options:
                if k not in self.E:
                    continue
                ready = route_ready[i, j, k]
                m.addConstr(ready <= y[i, j], name=f"route_ready_arc_{i}_{j}_{k}")
                m.addConstr(ready <= y[i, k], name=f"route_ready_swap_in_{i}_{j}_{k}")
                m.addConstr(ready <= y[k, j], name=f"route_ready_swap_out_{i}_{j}_{k}")
                ready_terms.append(ready)

            if ready_terms:
                m.addConstr(
                    od_service_ub[i, j] <= gp.quicksum(ready_terms),
                    name=f"od_route_ready_ub_{i}_{j}",
                )
                min_flight_cost = min(float(self.flight_cost[i, j, k]) for k in options)
                min_time = min(float(self.t[i, j, k]) for k in options)
                unit_profit_ub = max(0.0, revenue_coef - min_flight_cost)
                eta_terms.append(unit_profit_ub * demand * od_service_ub[i, j])
                fleet_terms.append(min_time * demand * od_service_ub[i, j])
            else:
                m.addConstr(od_service_ub[i, j] == 0.0, name=f"od_no_route_ub_{i}_{j}")

        if eta_terms:
            m.addConstr(
                eta <= gp.quicksum(eta_terms),
                name="eta_route_profit_ub",
            )
        else:
            m.addConstr(eta <= 0.0, name="eta_no_route_profit_ub")

        if fleet_terms:
            # Loaded HH flight time is only part of total fleet time, so this is
            # a conservative but valid upper-bound link between eta service and Gamma.
            m.addConstr(
                gp.quicksum(fleet_terms) <= Gamma,
                name="fleet_loaded_time_ub",
            )

        for j in self.H:
            incoming_terms = [
                float(self.inst.h.get((i, j), 0.0)) * od_service_ub[i, j]
                for i, h in self.A_HH
                if h == j
            ]
            if incoming_terms:
                m.addConstr(
                    gp.quicksum(incoming_terms)
                    <= gp.quicksum(float(mu_c[c]) * z[j, c] for c in range(len(config.MU_HUB)))
                    - x[j] / tau_H,
                    name=f"hub_loaded_in_capacity_ub_{j}",
                )

        self._add_master_flow_relaxation_ub(m, x, y, z, Gamma, eta, revenue_coef, mu_c, tau_H)

        return od_service_ub, route_ready

    def _add_master_flow_relaxation_ub(
        self,
        m: gp.Model,
        x,
        y,
        z,
        Gamma,
        eta,
        revenue_coef: float,
        mu_c: Iterable[float],
        tau_H: float,
    ) -> None:
        """Add a linear operational-flow relaxation to upper-bound eta."""
        total_demand = max(1.0, sum(float(value) for value in self.inst.h.values()))
        varphi_keys: List[RouteKey] = [
            (u, v, k)
            for (u, v) in self.A_F
            for k in self.route_options.get((u, v), [])
        ]
        psi_ub = m.addVars(self.A, lb=0.0, name="psi_flow_ub")
        varphi_ub = m.addVars(varphi_keys, lb=0.0, name="varphi_flow_ub")
        rho_l_ub = m.addVars(self.L, lb=0.0, name="rho_l_flow_ub")
        rho_s_ub = m.addVars(self.S, lb=0.0, name="rho_s_flow_ub")
        alpha_keys: List[AlphaKey] = [
            (l, i, j)
            for (l, i) in self.A_LH
            for (i2, j) in self.A_HH
            if i2 == i
        ]
        alpha_ub = m.addVars(alpha_keys, lb=0.0, name="alpha_flow_ub")
        beta_ub = m.addVars(list(self.A_HH), lb=0.0, name="beta_flow_ub")

        for l in self.L:
            m.addConstr(rho_l_ub[l] <= x[l], name=f"flow_relax_rho_l_{l}")
        for i, s in self.A_HS:
            m.addConstr(rho_s_ub[s] <= x[i], name=f"flow_relax_rho_s_{s}")

        for l, i, j in alpha_keys:
            m.addConstr(alpha_ub[l, i, j] <= rho_l_ub[l], name=f"flow_relax_alpha_rho_{l}_{i}_{j}")
            m.addConstr(alpha_ub[l, i, j] <= y[l, i], name=f"flow_relax_alpha_assign_{l}_{i}_{j}")
            m.addConstr(alpha_ub[l, i, j] <= y[i, j], name=f"flow_relax_alpha_arc_{l}_{i}_{j}")
            m.addConstr(
                alpha_ub[l, i, j] >= rho_l_ub[l] + y[l, i] + y[i, j] - 2.0,
                name=f"flow_relax_alpha_lb_{l}_{i}_{j}",
            )

        for s, i in self.A_SH:
            for h, j in self.A_HH:
                if h != i:
                    continue
                m.addConstr(beta_ub[i, j] <= rho_s_ub[s], name=f"flow_relax_beta_rho_{i}_{j}")
                m.addConstr(beta_ub[i, j] <= y[i, j], name=f"flow_relax_beta_arc_{i}_{j}")
                m.addConstr(
                    beta_ub[i, j] >= rho_s_ub[s] + y[i, j] - 1.0,
                    name=f"flow_relax_beta_lb_{i}_{j}",
                )

        for i, j in self.A_HH:
            demand = float(self.inst.h.get((i, j), 0.0))
            p_i = self._p_chain(i)
            m.addConstr(
                psi_ub[i, j]
                == demand
                * (
                    (1.0 - p_i)
                    * gp.quicksum(
                        alpha_ub[l, i, j]
                        for (l, h) in self.A_LH
                        if h == i and (l, i, j) in alpha_ub
                    )
                    + p_i * beta_ub[i, j]
                ),
                name=f"flow_relax_psi_hh_{i}_{j}",
            )
            m.addConstr(psi_ub[i, j] <= demand * y[i, j], name=f"psi_hh_demand_ub_{i}_{j}")

        for l, i in self.A_LH:
            m.addConstr(
                psi_ub[l, i]
                == gp.quicksum(
                    float(self.inst.h.get((i, j), 0.0))
                    * (1.0 - self._p_chain(i))
                    * alpha_ub[l, i, j]
                    for (h, j) in self.A_HH
                    if h == i and (l, i, j) in alpha_ub
                ),
                name=f"flow_relax_psi_lh_{l}_{i}",
            )

        for s, i in self.A_SH:
            m.addConstr(
                psi_ub[s, i]
                == gp.quicksum(
                    float(self.inst.h.get((i, j), 0.0))
                    * self._p_chain(i)
                    * beta_ub[i, j]
                    for (h, j) in self.A_HH
                    if h == i
                ),
                name=f"flow_relax_psi_sh_{s}_{i}",
            )

        for i, s in self.A_HS:
            m.addConstr(psi_ub[i, s] <= total_demand * x[i], name=f"psi_hs_active_ub_{i}_{s}")

        for u, v in self.A_hat:
            m.addConstr(
                gp.quicksum(
                    varphi_ub[u, v, k]
                    for k in self.route_options.get((u, v), [])
                    if (u, v, k) in varphi_ub
                )
                == psi_ub[u, v],
                name=f"flow_relax_route_{u}_{v}",
            )

        for u, v, k in varphi_keys:
            big_m = self.route_big_m[u, v, k]
            m.addConstr(varphi_ub[u, v, k] <= big_m * y[u, v], name=f"flow_relax_arc_active_{u}_{v}_{k}")
            if k in self.E:
                m.addConstr(varphi_ub[u, v, k] <= big_m * y[u, k], name=f"flow_relax_swap_in_{u}_{v}_{k}")
                m.addConstr(varphi_ub[u, v, k] <= big_m * y[k, v], name=f"flow_relax_swap_out_{u}_{v}_{k}")

        for q in self.HLS:
            m.addConstr(
                gp.quicksum(psi_ub[u, q] for (u, v) in self.A if v == q)
                == gp.quicksum(psi_ub[q, v] for (u, v) in self.A if u == q),
                name=f"flow_relax_balance_{q}",
            )

        total_loaded_and_empty_time = gp.quicksum(self.t[u, v, k] * varphi_ub[u, v, k] for u, v, k in varphi_keys)
        m.addConstr(total_loaded_and_empty_time <= Gamma, name="flow_relax_fleet_time")

        for j in self.H:
            m.addConstr(
                gp.quicksum(psi_ub[u, j] for (u, v) in self.A if v == j)
                <= gp.quicksum(float(mu_c[c]) * z[j, c] for c in range(len(config.MU_HUB)))
                - x[j] / tau_H,
                name=f"flow_relax_hub_capacity_{j}",
            )

        operating_cost = gp.quicksum(self.flight_cost[u, v, k] * varphi_ub[u, v, k] for u, v, k in varphi_keys)
        served_benefit = gp.quicksum(revenue_coef * psi_ub[i, j] for i, j in self.A_HH)
        m.addConstr(eta <= served_benefit - operating_cost, name="eta_flow_relax_ub")

    def apply_master_solver_config(self) -> None:
        assert self.master is not None
        time_limit = getattr(config, "BENDERS_MASTER_TIME_LIMIT", None)
        mip_gap = getattr(config, "BENDERS_MASTER_MIP_GAP", None)
        if time_limit is not None:
            self.master.setParam("TimeLimit", float(time_limit))
        if mip_gap is not None:
            self.master.setParam("MIPGap", float(mip_gap))

    def _extract_master_design(self, solution_number: int = 0) -> Dict[str, object]:
        assert self.master is not None
        if solution_number > 0:
            self.master.Params.SolutionNumber = int(solution_number)

        def _value(var) -> float:
            return float(var.Xn if solution_number > 0 else var.X)

        x = self.master_vars["x"]
        y = self.master_vars["y"]
        z = self.master_vars["z"]
        Gamma = self.master_vars["Gamma"]
        C = range(len(config.MU_HUB))
        return {
            "x": {u: _value(x[u]) for u in self.N_x},
            "y": {(u, v): _value(y[u, v]) for (u, v) in self.AY},
            "z": {(i, c): _value(z[i, c]) for i in self.H for c in C},
            "Gamma": _value(Gamma),
        }

    def _fixed_cost_value(self, design: Mapping[str, object]) -> float:
        x = design["x"]
        y = design["y"]
        z = design["z"]
        Gamma = float(design["Gamma"])
        C = range(len(config.MU_HUB))
        value = sum(float(self.inst.f.get(l, config.COST_BASE)) * float(x[l]) for l in self.L)
        value += sum(float(getattr(config, "COST_HUB_FIXED", 0.0)) * float(x[i]) for i in self.H)
        value += sum(float(self.inst.f.get(k, config.COST_SWAPPING)) * float(x[k]) for k in self.E)
        value += sum(float(config.COST_HUB[c]) * float(z[i, c]) for i in self.H for c in C)
        value += sum(self.arc_cost[u, v] * float(y[u, v]) for (u, v) in self.AY)
        value += float(config.COST_DRONE) * Gamma
        return value

    def _normalize_master_design(self, design_like: object) -> Dict[str, object]:
        """Coerce a heuristic/master solution to the current master index sets."""
        if hasattr(design_like, "to_master_design"):
            design_like = design_like.to_master_design()
        if not isinstance(design_like, Mapping):
            raise TypeError("Initial design must be a mapping or expose to_master_design().")

        x_raw = design_like.get("x", {})
        y_raw = design_like.get("y", {})
        z_raw = design_like.get("z", {})
        gamma_raw = design_like.get("Gamma", design_like.get("gamma", 0.0))
        eta_raw = design_like.get("eta", design_like.get("theta", None))
        C = range(len(config.MU_HUB))
        normalized: Dict[str, object] = {
            "x": {u: float(x_raw.get(u, 0.0)) for u in self.N_x},
            "y": {(u, v): float(y_raw.get((u, v), 0.0)) for (u, v) in self.AY},
            "z": {(i, c): float(z_raw.get((i, c), 0.0)) for i in self.H for c in C},
            "Gamma": float(gamma_raw or 0.0),
        }
        if eta_raw is not None:
            normalized["eta"] = float(eta_raw)
        return normalized

    def add_initial_design(self, design_like: object) -> None:
        """Register a first-stage design for MIP start and initial cut generation.

        Only master variables x/y/z/Gamma are required.  The continuous
        operational variables are intentionally omitted and will be recovered by
        solving the Benders subproblem at this design.
        """
        self.initial_designs.append(self._normalize_master_design(design_like))

    add_initial_solution = add_initial_design

    def _apply_master_start(self, design: Mapping[str, object]) -> None:
        assert self.master is not None
        x = self.master_vars["x"]
        y = self.master_vars["y"]
        z = self.master_vars["z"]
        Gamma = self.master_vars["Gamma"]
        eta = self.master_vars["eta"]
        C = range(len(config.MU_HUB))
        for u in self.N_x:
            x[u].Start = float(design["x"].get(u, 0.0))
        for u, v in self.AY:
            y[u, v].Start = float(design["y"].get((u, v), 0.0))
        for i in self.H:
            for c in C:
                z[i, c].Start = float(design["z"].get((i, c), 0.0))
        Gamma.Start = float(design.get("Gamma", 0.0))
        eta_start = design.get("eta", design.get("theta", None))
        if eta_start is not None:
            eta.Start = float(eta_start)

    def _process_initial_designs(self, lower_bound: float) -> Tuple[float, int]:
        if self._initial_designs_processed or not self.initial_designs:
            return lower_bound, 0

        added = 0
        best_initial: Optional[Dict[str, object]] = None
        best_initial_obj = -GRB.INFINITY
        for idx, design in enumerate(self.initial_designs, 1):
            sub = self.solve_subproblem(design, label=f"initial{idx}")
            if sub.obj_value is None:
                print(f"Initial design {idx}: subproblem status={sub.status}; skipped.")
                continue

            total_obj = sub.obj_value - self._fixed_cost_value(design)
            if total_obj > lower_bound + 1e-7:
                lower_bound = total_obj
                self.best_total_obj = total_obj
                self.best_operational_obj = sub.obj_value
                self.best_design = design
            if total_obj > best_initial_obj + 1e-7:
                best_initial_obj = total_obj
                best_initial = design

            if sub.cut_expr_data is None or sub.rhs_at_reference is None:
                print(f"Initial design {idx}: dual cut unavailable; kept as MIP start only.")
                continue
            self._add_benders_cut(sub, "initial")
            added += 1

        if best_initial is not None:
            self._apply_master_start(best_initial)
            self._initialize_or_update_core(best_initial)

        self._initial_designs_processed = True
        if added:
            print(f"Loaded {added} initial Benders cut(s); initial LB={lower_bound:.4f}.")
        return lower_bound, added

    # ------------------------------------------------------------------
    # Subproblem construction and cut extraction
    # ------------------------------------------------------------------

    def _rhs_x(self, u: Node) -> Callable[[Mapping[str, object]], Union[float, gp.LinExpr]]:
        return lambda d: d["x"][u]

    def _rhs_y(self, u: Node, v: Node, coef: float = 1.0) -> Callable[[Mapping[str, object]], Union[float, gp.LinExpr]]:
        return lambda d: coef * d["y"][u, v]

    def _rhs_node_open(self, node: Node) -> Callable[[Mapping[str, object]], Union[float, gp.LinExpr]]:
        def _f(d: Mapping[str, object]) -> Union[float, gp.LinExpr]:
            if node in self.S:
                return d["x"][self.inst.s_to_h[node]]
            return d["x"][node]
        return _f

    def _rhs_alpha_lb(self, l: Node, i: Node, j: Node) -> Callable[[Mapping[str, object]], Union[float, gp.LinExpr]]:
        return lambda d: d["y"][l, i] + d["y"][i, j] - 2.0

    def _rhs_beta_lb(self, i: Node, j: Node) -> Callable[[Mapping[str, object]], Union[float, gp.LinExpr]]:
        return lambda d: d["y"][i, j] - 1.0

    def _rhs_mu(self, i: Node) -> Callable[[Mapping[str, object]], Union[float, gp.LinExpr]]:
        C = range(len(config.MU_HUB))
        mu_c = config.MU_HUB
        return lambda d: sum(float(mu_c[c]) * d["z"][i, c] for c in C)

    def _rhs_sqrt_mu(self, i: Node) -> Callable[[Mapping[str, object]], Union[float, gp.LinExpr]]:
        C = range(len(config.MU_HUB))
        mu_c = config.MU_HUB
        return lambda d: sum(2.0 * math.sqrt(float(mu_c[c])) * d["z"][i, c] for c in C)

    def _rhs_hub_wait(self, i: Node) -> Callable[[Mapping[str, object]], Union[float, gp.LinExpr]]:
        tau_H = float(config.TAU_H)
        return lambda d: d["x"][i] / tau_H

    def _rhs_gamma(self) -> Callable[[Mapping[str, object]], Union[float, gp.LinExpr]]:
        return lambda d: d["Gamma"]

    @staticmethod
    def _rhs_value(rhs_fun: Callable[[Mapping[str, object]], Union[float, gp.LinExpr]], design: Mapping[str, object]) -> float:
        value = rhs_fun(design)
        if isinstance(value, (int, float)):
            return float(value)
        # This branch is not normally used for fixed designs.
        return float(value.getValue())

    def solve_subproblem(
        self,
        design: Mapping[str, object],
        label: str = "incumbent",
        bar_qcp_conv_tol: Optional[float] = None,
        dual_retry_index: int = 0,
    ) -> SubproblemResult:
        """Solve the continuous SOCP subproblem for a fixed design.

        The method stores all constraints whose RHS depends on master variables.
        The returned dual multipliers and RHS functions are used to construct a
        Benders cut in the master.
        """
        m = gp.Model(f"MLND_RegimeII_SOCP_Subproblem_{label}")
        m.Params.OutputFlag = int(getattr(config, "BENDERS_SUB_LOG", 0))
        m.Params.Threads = int(getattr(config, "BENDERS_SUB_THREADS", 1))
        m.Params.QCPDual = 1
        # Barrier is usually the most stable method for convex QCP/SOCP duals.
        if getattr(config, "BENDERS_SUB_USE_BARRIER", True):
            m.Params.Method = 2
            m.Params.Crossover = 0
            bar_qcp_tol = (
                bar_qcp_conv_tol
                if bar_qcp_conv_tol is not None
                else getattr(config, "BENDERS_SUB_BAR_QCP_CONV_TOL", None)
            )
            if bar_qcp_tol is not None:
                m.Params.BarQCPConvTol = float(bar_qcp_tol)

        C = range(len(config.MU_HUB))
        mu_c = config.MU_HUB
        tau_H = float(config.TAU_H)
        tau_S = float(config.TAU_S)
        omega = float(config.ENERGY_RATE_OMEGA)
        Omega = float(config.BATTERY_CAPACITY_OMEGA_UPPER)
        energy_epsilon = float(getattr(config, "ENERGY_EPSILON", getattr(config, "EPS", 0.0)))
        usable_battery = (1.0 - energy_epsilon) * Omega

        # Operational variables only.
        rho_l = m.addVars(self.L, lb=0.0, name="rho_l")
        rho_s = m.addVars(self.S, lb=0.0, name="rho_s")
        alpha_keys: List[AlphaKey] = [(l, i, j) for (l, i) in self.A_LH for (i2, j) in self.A_HH if i2 == i]
        alpha = m.addVars(alpha_keys, lb=0.0, name="alpha")
        beta = m.addVars(list(self.A_HH), lb=0.0, name="beta")
        psi = m.addVars(self.A, lb=0.0, name="psi")
        varphi_keys: List[RouteKey] = [(u, v, k) for (u, v) in self.A_F for k in self.route_options.get((u, v), [])]
        varphi = m.addVars(varphi_keys, lb=0.0, name="varphi")
        lambd = m.addVars(self.H, lb=0.0, name="lambda")
        mu = m.addVars(self.H, lb=0.0, name="mu")
        xi = m.addVars(self.HLS, lb=0.0, name="xi")

        revenue_coef = float(getattr(config, "ORDER_REVENUE_COEF", 0.0))
        served_benefit = gp.quicksum(revenue_coef * psi[i, j] for i, j in self.A_HH)
        operating_cost = gp.quicksum(self.flight_cost[u, v, k] * varphi[u, v, k] for (u, v, k) in varphi_keys)
        m.setObjective(served_benefit - operating_cost, GRB.MAXIMIZE)

        # List of parameterized constraints for Benders cut extraction.
        # Each item is (constraint, rhs_function, rhs_value_at_reference).
        param_constrs: List[Tuple[gp.Constr, Callable[[Mapping[str, object]], Union[float, gp.LinExpr]], float]] = []

        def add_le(lhs, rhs_fun, name: str):
            rhs_val = self._rhs_value(rhs_fun, design)
            con = m.addConstr(lhs <= rhs_val, name=name)
            param_constrs.append((con, rhs_fun, rhs_val))
            return con

        def add_ge(lhs, rhs_fun, name: str):
            rhs_val = self._rhs_value(rhs_fun, design)
            con = m.addConstr(lhs >= rhs_val, name=name)
            param_constrs.append((con, rhs_fun, rhs_val))
            return con

        def add_eq(lhs, rhs_fun, name: str):
            rhs_val = self._rhs_value(rhs_fun, design)
            con = m.addConstr(lhs == rhs_val, name=name)
            param_constrs.append((con, rhs_fun, rhs_val))
            return con

        # Availability bounds.
        for l in self.L:
            add_le(rho_l[l], self._rhs_x(l), f"rho_L_{l}")
        for i, s in self.A_HS:
            add_le(rho_s[s], self._rhs_x(i), f"rho_S_{s}")

        # alpha_lij linearizes rho_l * y_li * y_ij with y fixed.
        for l, i, j in alpha_keys:
            m.addConstr(alpha[l, i, j] <= rho_l[l], name=f"alpha_1_{l}_{i}_{j}")
            add_le(alpha[l, i, j], self._rhs_y(l, i), f"alpha_2_{l}_{i}_{j}")
            add_le(alpha[l, i, j], self._rhs_y(i, j), f"alpha_3_{l}_{i}_{j}")
            add_ge(alpha[l, i, j] - rho_l[l], self._rhs_alpha_lb(l, i, j), f"alpha_4_{l}_{i}_{j}")

        # beta_ij linearizes rho_s * y_ij with y fixed.
        for s, i in self.A_SH:
            for h, j in self.A_HH:
                if h != i:
                    continue
                m.addConstr(beta[i, j] <= rho_s[s], name=f"beta_1_{i}_{j}")
                add_le(beta[i, j], self._rhs_y(i, j), f"beta_2_{i}_{j}")
                add_ge(beta[i, j] - rho_s[s], self._rhs_beta_lb(i, j), f"beta_3_{i}_{j}")

        # Loaded hub-to-hub flow.
        for i, j in self.A_HH:
            p_i = self._p_chain(i)
            demand = float(self.inst.h.get((i, j), 0.0))
            m.addConstr(
                psi[i, j]
                == demand
                * (
                    (1.0 - p_i)
                    * gp.quicksum(alpha[l, i, j] for (l, h) in self.A_LH if h == i and (l, i, j) in alpha)
                    + p_i * beta[i, j]
                ),
                name=f"psi_HH_{i}_{j}",
            )

        # Empty base-to-hub flow.
        for l, i in self.A_LH:
            m.addConstr(
                psi[l, i]
                == gp.quicksum(
                    float(self.inst.h.get((i, j), 0.0)) * (1.0 - self._p_chain(i)) * alpha[l, i, j]
                    for (h, j) in self.A_HH
                    if h == i and (l, i, j) in alpha
                ),
                name=f"psi_LH_{l}_{i}",
            )

        # Empty holding-pool-to-hub flow.
        for s, i in self.A_SH:
            m.addConstr(
                psi[s, i]
                == gp.quicksum(
                    float(self.inst.h.get((i, j), 0.0)) * self._p_chain(i) * beta[i, j]
                    for (h, j) in self.A_HH
                    if h == i
                ),
                name=f"psi_SH_{s}_{i}",
            )

        # Closed-network flow balance.
        for q in self.HLS:
            m.addConstr(
                gp.quicksum(psi[u, q] for (u, v) in self.A if v == q)
                == gp.quicksum(psi[q, v] for (u, v) in self.A if u == q),
                name=f"flow_balance_{q}",
            )

        # Route decomposition.
        for u, v in self.A_hat:
            m.addConstr(
                gp.quicksum(varphi[u, v, k] for k in self.route_options.get((u, v), []) if (u, v, k) in varphi)
                == psi[u, v],
                name=f"route_{u}_{v}",
            )

        for u, v, k in varphi_keys:
            big_m = self.route_big_m[u, v, k]
            add_le(varphi[u, v, k], self._rhs_y(u, v, big_m), f"route_arc_active_{u}_{v}_{k}")

        for u, v, k in varphi_keys:
            if k not in self.E:
                continue
            big_m = self.route_big_m[u, v, k]
            add_le(varphi[u, v, k], self._rhs_y(u, k, big_m), f"route_swap_in_active_{u}_{v}_{k}")
            add_le(varphi[u, v, k], self._rhs_y(k, v, big_m), f"route_swap_out_active_{u}_{v}_{k}")

        # Energy constraints.  These do not contain first-stage RHS terms after
        # route-activation constraints are enforced.
        for l in self.L:
            m.addConstr(
                omega
                * gp.quicksum(
                    self.t[l, i, k] * varphi[l, i, k]
                    for (u, i, k) in varphi_keys
                    if u == l and (l, i) in self.A_hat
                )
                <= usable_battery
                * gp.quicksum(
                    varphi[l, i, k]
                    for (u, i, k) in varphi_keys
                    if u == l and (l, i) in self.A_hat and k in self.E
                ),
                name=f"energy_L_{l}",
            )

        for i in self.H:
            s = self._hub_pool(i)
            m.addConstr(
                omega
                * (
                    xi[i]
                    + xi[s]
                    + gp.quicksum(self.t[i, j, k] * varphi[i, j, k] for (h, j, k) in varphi_keys if h == i and (i, j) in self.A_HH)
                    + gp.quicksum(
                        self.t[s, v, k] * varphi[s, v, k]
                        for (u, v, k) in varphi_keys
                        if u == s and ((s, v) in self.A_SL or (s, v) in self.A_SS)
                    )
                )
                <= usable_battery
                * (
                    gp.quicksum(varphi[i, j, k] for (h, j, k) in varphi_keys if h == i and (i, j) in self.A_HH and k in self.E)
                    + gp.quicksum(
                        varphi[s, v, k]
                        for (u, v, k) in varphi_keys
                        if u == s and ((s, v) in self.A_SL or (s, v) in self.A_SS) and k in self.E
                    )
                ),
                name=f"energy_HS_{i}",
            )

        # lambda_i equals total incoming aggregate flow to hub i.
        for i in self.H:
            m.addConstr(lambd[i] == gp.quicksum(psi[u, i] for (u, v) in self.A if v == i), name=f"lambda_{i}")

        # mu_i and hub service stability; both depend on master variables.
        for i in self.H:
            add_eq(mu[i], self._rhs_mu(i), f"mu_{i}")
            add_ge(mu[i] - lambd[i], self._rhs_hub_wait(i), f"hub_wait_{i}")

        # Holding-pool mean dwell bound.
        for s in self.S:
            m.addConstr(xi[s] <= tau_S * gp.quicksum(psi[s, v] for (u, v) in self.A if u == s), name=f"mean_S_{s}")

        # SOC: xi_u >= rho_u / (1-rho_u) for bases and pools.
        q_nodes = self.L + self.S
        q_v1 = m.addVars(q_nodes, lb=-GRB.INFINITY, name="q_v1")
        q_v2 = m.addVars(q_nodes, lb=-GRB.INFINITY, name="q_v2")
        q_t = m.addVars(q_nodes, lb=0.0, name="q_t")
        for q in q_nodes:
            rho_q = rho_l[q] if q in self.L else rho_s[q]
            m.addConstr(q_v1[q] == 2.0, name=f"q_v1_{q}")
            m.addConstr(q_v2[q] == xi[q] + rho_q, name=f"q_v2_{q}")
            m.addConstr(q_t[q] == xi[q] - rho_q + 2.0, name=f"q_t_{q}")
            m.addQConstr(q_v1[q] * q_v1[q] + q_v2[q] * q_v2[q] <= q_t[q] * q_t[q], name=f"queue_{q}")

        # SOC: xi_i >= lambda_i / (mu_i-lambda_i) for hubs.
        h_v1 = m.addVars(self.H, lb=-GRB.INFINITY, name="h_v1")
        h_v2 = m.addVars(self.H, lb=-GRB.INFINITY, name="h_v2")
        h_t = m.addVars(self.H, lb=0.0, name="h_t")
        for i in self.H:
            add_eq(h_v1[i], self._rhs_sqrt_mu(i), f"h_v1_{i}")
            m.addConstr(h_v2[i] == xi[i] + lambd[i] - mu[i] + 1.0, name=f"h_v2_{i}")
            m.addConstr(h_t[i] == xi[i] - lambd[i] + mu[i] + 1.0, name=f"h_t_{i}")
            m.addQConstr(h_v1[i] * h_v1[i] + h_v2[i] * h_v2[i] <= h_t[i] * h_t[i], name=f"queue_H_{i}")

        total_queue_length = gp.quicksum(xi[u] for u in self.HLS)
        total_flight_time = gp.quicksum(self.t[u, v, k] * varphi[u, v, k] for (u, v, k) in varphi_keys)
        add_le(total_queue_length + total_flight_time, self._rhs_gamma(), "fleet")

        sub_time_limit = getattr(config, "BENDERS_SUB_TIME_LIMIT", None)
        if sub_time_limit is not None:
            m.Params.TimeLimit = float(sub_time_limit)

        m.optimize()

        if m.status != GRB.OPTIMAL or m.SolCount == 0:
            if (
                m.SolCount > 0
                and m.status in {GRB.SUBOPTIMAL, GRB.TIME_LIMIT}
            ):
                return SubproblemResult(
                    status=m.status,
                    obj_value=float(m.ObjVal),
                    cut_expr_data=None,
                    rhs_at_reference=None,
                    design=design,
                    label=label,
                )
            return SubproblemResult(status=m.status, obj_value=None, cut_expr_data=None, rhs_at_reference=None, design=design, label=label)

        # Extract linear dual multipliers for the parameterized constraints.
        # Gurobi's Pi is the sensitivity of the objective to the constraint RHS.
        cut_expr_data: List[Tuple[float, Callable[[Mapping[str, object]], gp.LinExpr]]] = []
        rhs_values: List[float] = []
        dual_tol = float(getattr(config, "BENDERS_DUAL_TOL", 1e-8))
        for con, rhs_fun, rhs_val in param_constrs:
            try:
                pi = float(con.Pi)
            except (gp.GurobiError, AttributeError):
                retry_tols = getattr(config, "BENDERS_SUB_DUAL_RETRY_BAR_QCP_CONV_TOLS", (1e-7, 1e-8))
                if dual_retry_index < len(retry_tols):
                    return self.solve_subproblem(
                        design,
                        label=f"{label}_dual_retry{dual_retry_index + 1}",
                        bar_qcp_conv_tol=float(retry_tols[dual_retry_index]),
                        dual_retry_index=dual_retry_index + 1,
                    )
                return SubproblemResult(
                    status=m.status,
                    obj_value=float(m.ObjVal),
                    cut_expr_data=None,
                    rhs_at_reference=None,
                    design=design,
                    label=label,
                )
            if abs(pi) > dual_tol:
                cut_expr_data.append((pi, rhs_fun))
                rhs_values.append(rhs_val)

        return SubproblemResult(
            status=m.status,
            obj_value=float(m.ObjVal),
            cut_expr_data=cut_expr_data,
            rhs_at_reference=rhs_values,
            design=design,
            label=label,
        )

    def _add_benders_cut(self, sub_result: SubproblemResult, cut_name_prefix: str) -> None:
        """Add eta <= Q(x_ref) + pi^T [rhs(x)-rhs(x_ref)]."""
        assert self.master is not None
        if sub_result.obj_value is None or sub_result.cut_expr_data is None or sub_result.rhs_at_reference is None:
            raise ValueError("Cannot add a Benders cut from an unsolved subproblem.")

        eta = self.master_vars["eta"]
        # Build expression for master RHS at current master variables.
        master_dict = {
            "x": self.master_vars["x"],
            "y": self.master_vars["y"],
            "z": self.master_vars["z"],
            "Gamma": self.master_vars["Gamma"],
        }
        cut_rhs = gp.LinExpr(float(sub_result.obj_value))
        for (pi, rhs_fun), rhs_ref in zip(sub_result.cut_expr_data, sub_result.rhs_at_reference):
            rhs_expr = rhs_fun(master_dict)
            cut_rhs += pi * (rhs_expr - rhs_ref)

        self.cut_count += 1
        self.master.addConstr(eta <= cut_rhs, name=f"{cut_name_prefix}_{self.cut_count}")

    # ------------------------------------------------------------------
    # Core-point/Pareto stabilization
    # ------------------------------------------------------------------

    def _initialize_or_update_core(self, design: Mapping[str, object]) -> None:
        """Maintain a fractional core point as a convex combination of incumbents."""
        C = range(len(config.MU_HUB))
        if self.core_point is None:
            self.core_point = {
                "x": {u: float(design["x"][u]) for u in self.N_x},
                "y": {(u, v): float(design["y"][u, v]) for (u, v) in self.AY},
                "z": {(i, c): float(design["z"][i, c]) for i in self.H for c in C},
                "Gamma": float(design["Gamma"]),
            }
            return

        w = self.core_update_weight
        for u in self.N_x:
            self.core_point["x"][u] = w * float(self.core_point["x"][u]) + (1.0 - w) * float(design["x"][u])
        for u, v in self.AY:
            self.core_point["y"][u, v] = w * float(self.core_point["y"][u, v]) + (1.0 - w) * float(design["y"][u, v])
        for i in self.H:
            for c in C:
                self.core_point["z"][i, c] = w * float(self.core_point["z"][i, c]) + (1.0 - w) * float(design["z"][i, c])
        self.core_point["Gamma"] = w * float(self.core_point["Gamma"]) + (1.0 - w) * float(design["Gamma"])

    def _pareto_reference_design(self, incumbent_design: Mapping[str, object]) -> Dict[str, object]:
        """Return a stabilized core/incumbent reference design.

        This is the point used to generate the additional Pareto-stabilized cut.
        It remains feasible for the linear master constraints because it is a
        convex combination of feasible master designs.
        """
        if self.core_point is None:
            return {
                "x": dict(incumbent_design["x"]),
                "y": dict(incumbent_design["y"]),
                "z": dict(incumbent_design["z"]),
                "Gamma": float(incumbent_design["Gamma"]),
            }
        lam = self.pareto_lambda
        C = range(len(config.MU_HUB))
        ref = {
            "x": {},
            "y": {},
            "z": {},
            "Gamma": lam * float(incumbent_design["Gamma"]) + (1.0 - lam) * float(self.core_point["Gamma"]),
        }
        for u in self.N_x:
            ref["x"][u] = lam * float(incumbent_design["x"][u]) + (1.0 - lam) * float(self.core_point["x"][u])
        for u, v in self.AY:
            ref["y"][u, v] = lam * float(incumbent_design["y"][u, v]) + (1.0 - lam) * float(self.core_point["y"][u, v])
        for i in self.H:
            for c in C:
                ref["z"][i, c] = lam * float(incumbent_design["z"][i, c]) + (1.0 - lam) * float(self.core_point["z"][i, c])
        return ref

    # ------------------------------------------------------------------
    # Integer reference cuts
    # ------------------------------------------------------------------

    def _collect_integer_reference_designs(self, incumbent_design: Mapping[str, object]) -> List[Dict[str, object]]:
        """Collect extra integer master solutions for additional full subproblem cuts."""
        if not getattr(config, "BENDERS_USE_INTEGER_REFERENCE_CUTS", False):
            return []

        assert self.master is not None
        max_refs = int(getattr(config, "BENDERS_REFERENCE_MAX_PER_ITER", 2))
        pool_limit = int(getattr(config, "BENDERS_REFERENCE_POOL_SOLUTIONS", 4))
        if max_refs <= 0 or pool_limit <= 1 or self.master.SolCount <= 1:
            return []

        reference_designs: List[Dict[str, object]] = []
        seen = {self._reference_design_signature(incumbent_design)}
        for solution_number in range(1, min(int(self.master.SolCount), pool_limit)):
            try:
                design = self._extract_master_design(solution_number=solution_number)
            except (gp.GurobiError, AttributeError):
                continue
            signature = self._reference_design_signature(design)
            if signature in seen:
                continue
            seen.add(signature)
            reference_designs.append(design)
            if len(reference_designs) >= max_refs:
                break
        return reference_designs

    def _add_integer_reference_cuts(
        self,
        iteration: int,
        reference_designs: List[Dict[str, object]],
        lower_bound: float,
    ) -> Tuple[float, int]:
        """Solve full subproblems at extra integer designs and add global cuts."""
        if not reference_designs:
            return lower_bound, 0

        added = 0
        for ref_idx, ref_design in enumerate(reference_designs, 1):
            sub = self.solve_subproblem(ref_design, label=f"iref{iteration}_{ref_idx}")
            if sub.obj_value is None:
                continue

            total_obj = sub.obj_value - self._fixed_cost_value(ref_design)
            if total_obj > lower_bound + 1e-7:
                lower_bound = total_obj
                self.best_total_obj = total_obj
                self.best_operational_obj = sub.obj_value
                self.best_design = ref_design

            if sub.cut_expr_data is None or sub.rhs_at_reference is None:
                continue

            self._add_benders_cut(sub, "integer_ref")
            added += 1
        return lower_bound, added

    # ------------------------------------------------------------------
    # Main Benders loop
    # ------------------------------------------------------------------

    def solve(
        self,
        max_iterations: Optional[int] = None,
        tolerance: Optional[float] = None,
        time_limit: Optional[float] = None,
    ) -> bool:
        """Solve the Benders master/subproblem loop.

        Returns True if at least one feasible incumbent is obtained.
        """
        self.build_master()
        assert self.master is not None
        self.apply_master_solver_config()

        max_iterations = int(max_iterations if max_iterations is not None else getattr(config, "BENDERS_MAX_ITER", 100))
        tolerance = float(tolerance if tolerance is not None else getattr(config, "BENDERS_TOL", 1e-4))
        time_limit = float(time_limit if time_limit is not None else getattr(config, "BENDERS_TIME_LIMIT", getattr(config, "MODEL_TIME_LIMIT", 3600.0)))
        start_time = gp.tupledict()  # dummy to avoid importing time lazily in old environments
        del start_time
        import time

        t0 = time.time()
        lower_bound = -GRB.INFINITY
        upper_bound = GRB.INFINITY
        lower_bound, initial_cut_count = self._process_initial_designs(lower_bound)

        for it in range(1, max_iterations + 1):
            elapsed = time.time() - t0
            if elapsed >= time_limit:
                print(f"Benders stopped by time limit before iteration {it}.")
                break
            self.master.Params.TimeLimit = max(1.0, time_limit - elapsed)
            self.master.optimize()

            if self.master.status not in {GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL} or self.master.SolCount == 0:
                print(f"Master did not return a solution. status={self.master.status}")
                break

            master_bound = float(self.master.ObjBound)
            if not math.isfinite(master_bound):
                master_bound = float(self.master.ObjVal)
            upper_bound = min(upper_bound, master_bound)
            design = self._extract_master_design()
            integer_reference_designs = self._collect_integer_reference_designs(design)
            eta_val = float(self.master_vars["eta"].X)
            fixed_value = self._fixed_cost_value(design)

            # Exact incumbent subproblem.
            sub = self.solve_subproblem(design, label=f"iter{it}")
            if sub.obj_value is None:
                # This should be rare because zero flow is normally feasible. Only an
                # actual infeasibility status justifies excluding the integer design.
                if sub.status in {GRB.INFEASIBLE, GRB.INF_OR_UNBD}:
                    print(f"Iteration {it}: subproblem infeasible/status={sub.status}; adding fallback no-good cut.")
                    self._add_no_good_cut(design)
                    continue
                print(f"Iteration {it}: subproblem did not solve to optimality, status={sub.status}; stopping.")
                break

            total_obj = sub.obj_value - fixed_value
            if total_obj > lower_bound + 1e-7:
                lower_bound = total_obj
                self.best_total_obj = total_obj
                self.best_operational_obj = sub.obj_value
                self.best_design = design

            zero_flow_cut_added = self._add_zero_flow_design_cut(
                design,
                eta_val=eta_val,
                sub_obj=sub.obj_value,
                fixed_value=fixed_value,
            )

            # Add the incumbent exact conic Benders cut.
            if sub.cut_expr_data is None or sub.rhs_at_reference is None:
                if zero_flow_cut_added or self._add_zero_flow_design_cut(
                    design,
                    eta_val=eta_val,
                    sub_obj=sub.obj_value,
                    fixed_value=fixed_value,
                ):
                    print(f"Iteration {it}: zero-flow subproblem duals unavailable; added Gamma-lift cut.")
                    continue
                print(f"Iteration {it}: incumbent subproblem duals are unavailable; stopping.")
                break
            self._add_benders_cut(sub, "benders")

            # Add the Pareto/core-point cut.  It is valid as an ordinary Benders cut
            # generated at a stabilized reference design, and usually improves convergence
            # when the incumbent cut is weak/degenerate.
            if self.use_pareto_cuts:
                self._initialize_or_update_core(design)
                ref_design = self._pareto_reference_design(design)
                pareto_sub = self.solve_subproblem(ref_design, label=f"pareto{it}")
                if pareto_sub.obj_value is not None and pareto_sub.cut_expr_data is not None:
                    self._add_benders_cut(pareto_sub, "pareto")
                else:
                    print(f"Iteration {it}: Pareto reference subproblem failed, status={pareto_sub.status}.")

            lower_bound, integer_ref_cuts = self._add_integer_reference_cuts(
                iteration=it,
                reference_designs=integer_reference_designs,
                lower_bound=lower_bound,
            )

            abs_gap = upper_bound - lower_bound
            rel_gap = abs_gap / max(1.0, abs(lower_bound))
            self.history.append(
                {
                    "iter": it,
                    "UB": upper_bound,
                    "LB": lower_bound,
                    "abs_gap": abs_gap,
                    "rel_gap": rel_gap,
                    "eta": eta_val,
                    "sub_obj": sub.obj_value,
                    "fixed_cost": fixed_value,
                    "cuts": self.cut_count,
                    "initial_cuts": initial_cut_count if it == 1 else 0,
                    "integer_ref_cuts": integer_ref_cuts,
                    "time": time.time() - t0,
                }
            )
            print(
                f"Iter {it:03d}: UB={upper_bound:.4f}, LB={lower_bound:.4f}, "
                f"gap={rel_gap:.4%}, eta={eta_val:.4f}, sub={sub.obj_value:.4f}, cuts={self.cut_count}"
            )

            if abs_gap <= tolerance * max(1.0, abs(lower_bound)):
                print("Benders converged.")
                break

        return self.best_design is not None

    def _add_no_good_cut(self, design: Mapping[str, object]) -> None:
        """Fallback no-good cut for an infeasible integer master design.

        For this model, zero flow is normally feasible, so this cut is only a
        defensive safeguard.  It excludes the exact binary design x,y,z.  Gamma
        is not included because increasing fleet size might repair infeasibility.
        """
        assert self.master is not None
        x = self.master_vars["x"]
        y = self.master_vars["y"]
        z = self.master_vars["z"]
        C = range(len(config.MU_HUB))
        terms = []
        for u in self.N_x:
            terms.append(1 - x[u] if float(design["x"][u]) > 0.5 else x[u])
        for u, v in self.AY:
            terms.append(1 - y[u, v] if float(design["y"][u, v]) > 0.5 else y[u, v])
        for i in self.H:
            for c in C:
                terms.append(1 - z[i, c] if float(design["z"][i, c]) > 0.5 else z[i, c])
        self.master.addConstr(gp.quicksum(terms) >= 1, name=f"nogood_{self.cut_count + 1}")
        self.cut_count += 1

    def _add_zero_flow_design_cut(
        self,
        design: Mapping[str, object],
        eta_val: float,
        sub_obj: float,
        fixed_value: float,
    ) -> bool:
        """Conditionally lift Gamma for a zero-recourse design.

        If the same binary network has zero operational value at the current
        Gamma, then all lower Gamma values also have zero operational value.
        Since close-all is feasible with objective 0, such positive-fixed-cost
        points cannot be optimal.  The cut still allows the same network to be
        retried with a larger Gamma.
        """
        if not getattr(config, "BENDERS_ADD_ZERO_FLOW_DESIGN_CUT", True):
            return False

        eps = float(getattr(config, "EPS", 1e-6))
        if sub_obj > eps or eta_val <= eps or fixed_value <= eps:
            return False

        assert self.master is not None
        x = self.master_vars["x"]
        y = self.master_vars["y"]
        z = self.master_vars["z"]
        Gamma = self.master_vars["Gamma"]
        C = range(len(config.MU_HUB))
        terms = []
        for u in self.N_x:
            terms.append(1 - x[u] if float(design["x"][u]) > 0.5 else x[u])
        for u, v in self.AY:
            terms.append(1 - y[u, v] if float(design["y"][u, v]) > 0.5 else y[u, v])
        for i in self.H:
            for c in C:
                terms.append(1 - z[i, c] if float(design["z"][i, c]) > 0.5 else z[i, c])

        signature = self._binary_design_signature(design)
        repeat_count = self.zero_flow_signature_counts.get(signature, 0) + 1
        self.zero_flow_signature_counts[signature] = repeat_count
        nogood_after = int(getattr(config, "BENDERS_ZERO_FLOW_NOGOOD_AFTER", 3))
        if nogood_after > 0 and repeat_count >= nogood_after:
            self.cut_count += 1
            self.master.addConstr(
                gp.quicksum(terms) >= 1.0,
                name=f"zero_flow_nogood_{self.cut_count}",
            )
            return True

        gamma_next = math.floor(float(design["Gamma"]) + eps) + 1.0
        big_m = float(getattr(config, "BENDERS_ZERO_FLOW_GAMMA_BIG_M", 1e6))
        self.cut_count += 1
        self.master.addConstr(
            Gamma >= gamma_next - big_m * gp.quicksum(terms),
            name=f"zero_flow_design_{self.cut_count}",
        )
        return True

    def _binary_design_signature(self, design: Mapping[str, object]) -> Tuple[int, ...]:
        values: List[int] = []
        values.extend(1 if float(design["x"][u]) > 0.5 else 0 for u in self.N_x)
        values.extend(1 if float(design["y"][u, v]) > 0.5 else 0 for u, v in self.AY)
        for i in self.H:
            values.extend(1 if float(design["z"][i, c]) > 0.5 else 0 for c in range(len(config.MU_HUB)))
        return tuple(values)

    def _reference_design_signature(self, design: Mapping[str, object]) -> Tuple[int, ...]:
        gamma_value = int(round(float(design["Gamma"])))
        return self._binary_design_signature(design) + (gamma_value,)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def write_master(self, filename: str = "benders_master.lp") -> None:
        self.build_master()
        assert self.master is not None
        self.master.write(filename)

    def save_history_csv(self, filename: str) -> None:
        import csv

        if not self.history:
            return
        keys = list(self.history[0].keys())
        with open(filename, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(self.history)


# ----------------------------------------------------------------------
# Convenience runner mirroring the original Regime-II script.
# ----------------------------------------------------------------------


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "n"}


def run_meituan_regime_ii_benders(
    generate_sample: Optional[bool] = None,
    date_filter: Optional[str] = None,
    project_root: Optional[str] = None,
    save_outputs: bool = True,
):
    """Generate/load the Meituan instance and solve it by Benders."""
    from src.data_processor.generator import DataGenerator
    from src.data_processor.loader import DataLoader

    if project_root is None:
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if generate_sample is None:
        generate_sample = _env_flag("MLND_GENERATE_SAMPLE", True)
    if date_filter is None:
        date_filter = config.MEITUAN_DEMAND_DATES[0]

    period_id = f"h{config.MEITUAN_DEMAND_HOUR:02d}"
    meituan_instance_dir = os.path.join(project_root, "data", "instances", "Meituan")
    meituan_results_dir = os.path.join(project_root, "results", "Meituan_Benders")
    meituan_raw_file = os.path.join(project_root, "Meituan-TSL", "all_waybill_info_meituan_0322.csv")
    if save_outputs:
        os.makedirs(meituan_results_dir, exist_ok=True)

    print(
        f"Regime II Benders run: generate={generate_sample}, period={period_id}, "
        f"max_iter={getattr(config, 'BENDERS_MAX_ITER', 100)}, tol={getattr(config, 'BENDERS_TOL', 1e-4)}"
    )

    if generate_sample:
        DataGenerator(seed=42).generate_meituan_hourly(
            raw_data_path=meituan_raw_file,
            save_dir=meituan_instance_dir,
            num_hubs=config.MEITUAN_NUM_HUBS,
            num_bases=config.MEITUAN_NUM_BASES,
            num_swapping=config.MEITUAN_NUM_SWAPPING,
            date_filter=date_filter,
            map_half_size_km=config.MEITUAN_MAP_HALF_SIZE_KM,
            candidate_date_filter=config.MEITUAN_CANDIDATE_DATES,
            demand_date_filter=config.MEITUAN_DEMAND_DATES,
            candidate_hour_start=config.MEITUAN_CANDIDATE_HOUR_START,
            candidate_hour_end=config.MEITUAN_CANDIDATE_HOUR_END,
            demand_hour_start=config.MEITUAN_DEMAND_HOUR,
            colocate_base_swaps=config.MEITUAN_COLOCATE_BASE_SWAPS,
        )

    period_path = os.path.join(meituan_instance_dir, "orders", f"{period_id}.csv")
    load_period_id = period_id if os.path.exists(period_path) else None
    inst = DataLoader().load_instance_dir(meituan_instance_dir, period_id=load_period_id)
    print(
        f"Loaded: H={len(inst.H)}, L={len(inst.L)}, S={len(inst.S)}, E={len(inst.E)}, "
        f"OD={len(inst.h)}, A={len(inst.A)}, A_F={len(inst.A_F)}"
    )

    solver = BendersRegimeIISolver(
        inst,
        use_pareto_cuts=bool(getattr(config, "BENDERS_USE_PARETO", True)),
        pareto_lambda=float(getattr(config, "BENDERS_PARETO_LAMBDA", 0.5)),
        core_update_weight=float(getattr(config, "BENDERS_CORE_UPDATE_WEIGHT", 0.7)),
    )
    success = solver.solve()

    if save_outputs:
        solver.write_master(os.path.join(meituan_results_dir, "benders_master_final.lp"))
        solver.save_history_csv(os.path.join(meituan_results_dir, "benders_history.csv"))
        print(f"Saved Benders outputs to {meituan_results_dir}")

    if success:
        print(f"Best Benders objective: {solver.best_total_obj:.4f}")
    else:
        print("No feasible Benders incumbent found.")
    return solver


if __name__ == "__main__":
    run_meituan_regime_ii_benders()
