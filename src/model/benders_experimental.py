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
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

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


@dataclass
class PersistentSubproblemState:
    model: gp.Model
    param_constrs: List[Tuple[gp.Constr, Callable[[Mapping[str, object]], Union[float, gp.LinExpr]]]]


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
# Experimental accelerations.  These classes are intentionally kept in this
# copied file so the production Benders implementation remains unchanged.
# ----------------------------------------------------------------------


@dataclass
class PartialMasterSolution:
    obj: float
    x: Dict[Node, float]
    y: Dict[Arc, float]
    z: Dict[Tuple[Node, int], float]
    Gamma: float
    theta: float
    psi: Dict[Arc, float]
    xi: Dict[Node, float]


@dataclass
class RouteRecourseResult:
    status: int
    obj_value: Optional[float]
    cut_expr_data: Optional[List[Tuple[float, Callable[[Mapping[str, object]], gp.LinExpr]]]]
    rhs_at_reference: Optional[List[float]]
    slack: float
    label: str
    feasibility_obj: Optional[float] = None
    feasibility_cut_expr_data: Optional[List[Tuple[float, Callable[[Mapping[str, object]], gp.LinExpr]]]] = None
    feasibility_rhs_at_reference: Optional[List[float]] = None
    slack_components: Optional[Dict[str, float]] = None


class ExperimentalBendersRegimeIISolver(BendersRegimeIISolver):
    """Standard Benders plus low-risk experimental accelerations.

    This keeps the same subproblem as ``BendersRegimeIISolver`` and adds:
    - multiple initial cuts from local-search reference designs,
    - an optional eta-by-origin split in the master,
    - Pareto core updates only from positive designs, with retry references.
    """

    def __init__(
        self,
        instance: Instance,
        use_pareto_cuts: bool = True,
        pareto_lambda: float = 0.5,
        core_update_weight: float = 0.7,
        *,
        use_eta_origin_split: bool = True,
        pareto_positive_core_only: bool = True,
        pareto_retry_lambdas: Sequence[float] = (0.75, 0.9),
        use_persistent_subproblem: bool = True,
        use_local_branching: bool = False,
        local_branching_radius: int = 6,
        local_branching_time_limit: float = 3.0,
        local_branching_max_calls: int = 3,
    ):
        super().__init__(
            instance,
            use_pareto_cuts=use_pareto_cuts,
            pareto_lambda=pareto_lambda,
            core_update_weight=core_update_weight,
        )
        self.use_eta_origin_split = bool(use_eta_origin_split)
        self.pareto_positive_core_only = bool(pareto_positive_core_only)
        self.pareto_retry_lambdas = tuple(float(v) for v in pareto_retry_lambdas)
        self.use_persistent_subproblem = bool(use_persistent_subproblem)
        self.use_local_branching = bool(use_local_branching)
        self.local_branching_radius = int(local_branching_radius)
        self.local_branching_time_limit = float(local_branching_time_limit)
        self.local_branching_max_calls = int(local_branching_max_calls)
        self.local_search_reference_solutions: List[object] = []
        self._persistent_subproblem: Optional[PersistentSubproblemState] = None
        self._local_branching_calls = 0
        self._local_branching_seen: set[Tuple[int, ...]] = set()

    def build_master(self) -> None:
        super().build_master()
        if not self.use_eta_origin_split:
            return
        assert self.master is not None
        if "eta_origin" in self.master_vars:
            return

        eta = self.master_vars["eta"]
        y = self.master_vars["y"]
        revenue_coef = float(getattr(config, "ORDER_REVENUE_COEF", 0.0))
        eta_origin = self.master.addVars(self.H, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="eta_origin")
        self.master.addConstr(eta == gp.quicksum(eta_origin[i] for i in self.H), name="eta_origin_sum")
        for i in self.H:
            terms = []
            for h, j in self.A_HH:
                if h != i:
                    continue
                demand = float(self.inst.h.get((i, j), 0.0))
                options = self.route_options.get((i, j), [])
                if not options or (i, j) not in y:
                    continue
                min_cost = min(float(self.flight_cost[i, j, k]) for k in options)
                unit_profit_ub = max(0.0, revenue_coef - min_cost)
                terms.append(unit_profit_ub * demand * y[i, j])
            if terms:
                self.master.addConstr(eta_origin[i] <= gp.quicksum(terms), name=f"eta_origin_profit_ub_{i}")
            else:
                self.master.addConstr(eta_origin[i] <= 0.0, name=f"eta_origin_no_profit_{i}")
        self.master_vars["eta_origin"] = eta_origin

    def add_local_search_reference_designs(
        self,
        *,
        top_k: int = 3,
        min_obj: float = -GRB.INFINITY,
        min_hamming_distance: Optional[int] = None,
        diversity_fill: bool = True,
        search_kwargs: Optional[Mapping[str, object]] = None,
    ) -> List[object]:
        """Run local search once and register the best distinct designs as initial cuts."""
        from src.model.local_search_heuristic import RegimeIILocalSearch

        kwargs = dict(search_kwargs or {})
        search = RegimeIILocalSearch(self.inst, **kwargs)
        best = search.solve()
        candidates = list(search.evaluator.cache.values())
        candidates.append(best)
        candidates = [
            sol for sol in candidates
            if sol.obj is not None and float(sol.obj) >= min_obj and sol.gamma is not None
        ]
        candidates.sort(key=lambda sol: float(sol.obj), reverse=True)

        min_hamming = int(
            min_hamming_distance
            if min_hamming_distance is not None
            else getattr(config, "BENDERS_REFERENCE_MIN_HAMMING_DISTANCE", 2)
        )
        selected: List[object] = []
        selected_designs: List[Dict[str, object]] = []
        seen = set()
        for sol in candidates:
            design = self._normalize_master_design(sol)
            signature = self._reference_design_signature(design)
            if signature in seen:
                continue
            seen.add(signature)
            if min_hamming > 0 and any(
                self._binary_design_distance(design, chosen) < min_hamming
                for chosen in selected_designs
            ):
                continue
            self.add_initial_design(design)
            selected.append(sol)
            selected_designs.append(design)
            if len(selected) >= int(top_k):
                break
        if diversity_fill and len(selected) < int(top_k):
            selected_signatures = {
                self._reference_design_signature(self._normalize_master_design(sol))
                for sol in selected
            }
            for sol in candidates:
                design = self._normalize_master_design(sol)
                signature = self._reference_design_signature(design)
                if signature in selected_signatures:
                    continue
                selected_signatures.add(signature)
                self.add_initial_design(design)
                selected.append(sol)
                if len(selected) >= int(top_k):
                    break
        self.local_search_reference_solutions = selected
        return selected

    def _binary_design_distance(
        self,
        left_design: Mapping[str, object],
        right_design: Mapping[str, object],
    ) -> int:
        left = self._binary_design_signature(left_design)
        right = self._binary_design_signature(right_design)
        return sum(1 for left_value, right_value in zip(left, right) if left_value != right_value)

    def _collect_integer_reference_designs(self, incumbent_design: Mapping[str, object]) -> List[Dict[str, object]]:
        if not getattr(config, "BENDERS_USE_INTEGER_REFERENCE_CUTS", False):
            return []

        assert self.master is not None
        max_refs = int(getattr(config, "BENDERS_REFERENCE_MAX_PER_ITER", 2))
        pool_limit = int(getattr(config, "BENDERS_REFERENCE_POOL_SOLUTIONS", 4))
        if max_refs <= 0 or pool_limit <= 1 or self.master.SolCount <= 1:
            return []

        min_hamming = int(getattr(config, "BENDERS_REFERENCE_MIN_HAMMING_DISTANCE", 2))
        incumbent_signature = self._reference_design_signature(incumbent_design)
        candidates: List[Dict[str, object]] = []
        seen = {incumbent_signature}
        for solution_number in range(1, min(int(self.master.SolCount), pool_limit)):
            try:
                design = self._extract_master_design(solution_number=solution_number)
            except (gp.GurobiError, AttributeError):
                continue
            signature = self._reference_design_signature(design)
            if signature in seen:
                continue
            seen.add(signature)
            candidates.append(design)

        selected: List[Dict[str, object]] = []
        for design in candidates:
            if min_hamming > 0 and self._binary_design_distance(design, incumbent_design) < min_hamming:
                continue
            if min_hamming > 0 and any(self._binary_design_distance(design, chosen) < min_hamming for chosen in selected):
                continue
            selected.append(design)
            if len(selected) >= max_refs:
                return selected

        if len(selected) < max_refs:
            selected_signatures = {self._reference_design_signature(design) for design in selected}
            for design in candidates:
                signature = self._reference_design_signature(design)
                if signature in selected_signatures:
                    continue
                selected_signatures.add(signature)
                selected.append(design)
                if len(selected) >= max_refs:
                    break
        return selected

    def solve_subproblem(
        self,
        design: Mapping[str, object],
        label: str = "incumbent",
        bar_qcp_conv_tol: Optional[float] = None,
        dual_retry_index: int = 0,
    ) -> SubproblemResult:
        """Solve the fixed-design subproblem, optionally reusing one model."""
        if (
            self.use_persistent_subproblem
            and bar_qcp_conv_tol is None
            and dual_retry_index == 0
        ):
            return self._solve_persistent_subproblem(design, label=label)
        return super().solve_subproblem(
            design,
            label=label,
            bar_qcp_conv_tol=bar_qcp_conv_tol,
            dual_retry_index=dual_retry_index,
        )

    def _build_persistent_subproblem(self) -> PersistentSubproblemState:
        m = gp.Model("MLND_RegimeII_SOCP_Subproblem_Persistent")
        m.Params.OutputFlag = int(getattr(config, "BENDERS_SUB_LOG", 0))
        m.Params.Threads = int(getattr(config, "BENDERS_SUB_THREADS", 1))
        m.Params.QCPDual = 1
        if getattr(config, "BENDERS_SUB_USE_BARRIER", True):
            m.Params.Method = 2
            m.Params.Crossover = 0
            bar_qcp_tol = getattr(config, "BENDERS_SUB_BAR_QCP_CONV_TOL", None)
            if bar_qcp_tol is not None:
                m.Params.BarQCPConvTol = float(bar_qcp_tol)

        C = range(len(config.MU_HUB))
        mu_c = config.MU_HUB
        tau_S = float(config.TAU_S)
        omega = float(config.ENERGY_RATE_OMEGA)
        Omega = float(config.BATTERY_CAPACITY_OMEGA_UPPER)
        energy_epsilon = float(getattr(config, "ENERGY_EPSILON", getattr(config, "EPS", 0.0)))
        usable_battery = (1.0 - energy_epsilon) * Omega

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

        param_constrs: List[Tuple[gp.Constr, Callable[[Mapping[str, object]], Union[float, gp.LinExpr]]]] = []

        def add_le(lhs, rhs_fun, name: str):
            con = m.addConstr(lhs <= 0.0, name=name)
            param_constrs.append((con, rhs_fun))
            return con

        def add_ge(lhs, rhs_fun, name: str):
            con = m.addConstr(lhs >= 0.0, name=name)
            param_constrs.append((con, rhs_fun))
            return con

        def add_eq(lhs, rhs_fun, name: str):
            con = m.addConstr(lhs == 0.0, name=name)
            param_constrs.append((con, rhs_fun))
            return con

        for l in self.L:
            add_le(rho_l[l], self._rhs_x(l), f"rho_L_{l}")
        for i, s in self.A_HS:
            add_le(rho_s[s], self._rhs_x(i), f"rho_S_{s}")

        for l, i, j in alpha_keys:
            m.addConstr(alpha[l, i, j] <= rho_l[l], name=f"alpha_1_{l}_{i}_{j}")
            add_le(alpha[l, i, j], self._rhs_y(l, i), f"alpha_2_{l}_{i}_{j}")
            add_le(alpha[l, i, j], self._rhs_y(i, j), f"alpha_3_{l}_{i}_{j}")
            add_ge(alpha[l, i, j] - rho_l[l], self._rhs_alpha_lb(l, i, j), f"alpha_4_{l}_{i}_{j}")

        for s, i in self.A_SH:
            for h, j in self.A_HH:
                if h != i:
                    continue
                m.addConstr(beta[i, j] <= rho_s[s], name=f"beta_1_{i}_{j}")
                add_le(beta[i, j], self._rhs_y(i, j), f"beta_2_{i}_{j}")
                add_ge(beta[i, j] - rho_s[s], self._rhs_beta_lb(i, j), f"beta_3_{i}_{j}")

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

        for q in self.HLS:
            m.addConstr(
                gp.quicksum(psi[u, q] for (u, v) in self.A if v == q)
                == gp.quicksum(psi[q, v] for (u, v) in self.A if u == q),
                name=f"flow_balance_{q}",
            )

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

        for i in self.H:
            m.addConstr(lambd[i] == gp.quicksum(psi[u, i] for (u, v) in self.A if v == i), name=f"lambda_{i}")

        for i in self.H:
            add_eq(mu[i], self._rhs_mu(i), f"mu_{i}")
            add_ge(mu[i] - lambd[i], self._rhs_hub_wait(i), f"hub_wait_{i}")

        for s in self.S:
            m.addConstr(xi[s] <= tau_S * gp.quicksum(psi[s, v] for (u, v) in self.A if u == s), name=f"mean_S_{s}")

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

        m.update()
        return PersistentSubproblemState(model=m, param_constrs=param_constrs)

    def _solve_persistent_subproblem(
        self,
        design: Mapping[str, object],
        label: str,
    ) -> SubproblemResult:
        if self._persistent_subproblem is None:
            self._persistent_subproblem = self._build_persistent_subproblem()

        state = self._persistent_subproblem
        m = state.model
        rhs_values: List[float] = []
        try:
            for con, rhs_fun in state.param_constrs:
                rhs_val = self._rhs_value(rhs_fun, design)
                con.RHS = rhs_val
                rhs_values.append(rhs_val)
            sub_time_limit = getattr(config, "BENDERS_SUB_TIME_LIMIT", None)
            if sub_time_limit is not None:
                m.Params.TimeLimit = float(sub_time_limit)
            m.update()
            m.optimize()
        except gp.GurobiError as exc:
            print(f"Persistent subproblem failed for {label}: {exc}; falling back to rebuilt subproblem.")
            return super().solve_subproblem(design, label=f"{label}_fallback")

        if m.status != GRB.OPTIMAL or m.SolCount == 0:
            return SubproblemResult(status=m.status, obj_value=None, cut_expr_data=None, rhs_at_reference=None, design=design, label=label)

        cut_expr_data: List[Tuple[float, Callable[[Mapping[str, object]], gp.LinExpr]]] = []
        active_rhs_values: List[float] = []
        dual_tol = float(getattr(config, "BENDERS_DUAL_TOL", 1e-8))
        for (con, rhs_fun), rhs_val in zip(state.param_constrs, rhs_values):
            try:
                pi = float(con.Pi)
            except (gp.GurobiError, AttributeError):
                retry_tols = getattr(config, "BENDERS_SUB_DUAL_RETRY_BAR_QCP_CONV_TOLS", (1e-7, 1e-8))
                if retry_tols:
                    return super().solve_subproblem(
                        design,
                        label=f"{label}_dual_retry1",
                        bar_qcp_conv_tol=float(retry_tols[0]),
                        dual_retry_index=1,
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
                active_rhs_values.append(rhs_val)

        return SubproblemResult(
            status=m.status,
            obj_value=float(m.ObjVal),
            cut_expr_data=cut_expr_data,
            rhs_at_reference=active_rhs_values,
            design=design,
            label=label,
        )

    def _pareto_reference_design_with_lambda(
        self,
        incumbent_design: Mapping[str, object],
        lam: float,
    ) -> Dict[str, object]:
        old_lam = self.pareto_lambda
        try:
            self.pareto_lambda = float(lam)
            return self._pareto_reference_design(incumbent_design)
        finally:
            self.pareto_lambda = old_lam

    def _local_branching_terms(self, center_design: Mapping[str, object]) -> List[gp.LinExpr]:
        x = self.master_vars["x"]
        y = self.master_vars["y"]
        z = self.master_vars["z"]
        C = range(len(config.MU_HUB))
        terms: List[gp.LinExpr] = []
        for u in self.N_x:
            terms.append(1 - x[u] if float(center_design["x"][u]) > 0.5 else x[u])
        for u, v in self.AY:
            terms.append(1 - y[u, v] if float(center_design["y"][u, v]) > 0.5 else y[u, v])
        for i in self.H:
            for c in C:
                terms.append(1 - z[i, c] if float(center_design["z"][i, c]) > 0.5 else z[i, c])
        return terms

    def _add_local_branching_constraints(
        self,
        center_design: Mapping[str, object],
        radius: int,
        name: str,
    ) -> List[gp.Constr]:
        assert self.master is not None
        distance_expr = gp.quicksum(self._local_branching_terms(center_design))
        constraints = [self.master.addConstr(distance_expr <= float(radius), name=f"{name}_ub")]
        if radius >= 1:
            constraints.append(self.master.addConstr(distance_expr >= 1.0, name=f"{name}_lb"))
        return constraints

    def _run_local_branching_reference(
        self,
        *,
        iteration: int,
        center_design: Mapping[str, object],
        lower_bound: float,
        remaining_time: float,
    ) -> Tuple[float, int]:
        if (
            not self.use_local_branching
            or self.local_branching_radius <= 0
            or self.local_branching_time_limit <= 0.0
            or self._local_branching_calls >= self.local_branching_max_calls
            or remaining_time <= 1.0
        ):
            return lower_bound, 0

        assert self.master is not None
        center_signature = self._reference_design_signature(center_design)
        if center_signature in self._local_branching_seen:
            return lower_bound, 0
        self._local_branching_seen.add(center_signature)
        self._local_branching_calls += 1

        constraints = self._add_local_branching_constraints(
            center_design,
            radius=self.local_branching_radius,
            name=f"local_branch_{iteration}_{self._local_branching_calls}",
        )
        old_time_limit = float(self.master.Params.TimeLimit)
        candidate: Optional[Dict[str, object]] = None
        try:
            self.master.Params.TimeLimit = max(0.5, min(self.local_branching_time_limit, remaining_time))
            self.master.optimize()
            if self.master.status in {GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL} and self.master.SolCount > 0:
                candidate = self._extract_master_design()
        finally:
            for constr in constraints:
                self.master.remove(constr)
            self.master.update()
            self.master.Params.TimeLimit = old_time_limit

        if candidate is None:
            return lower_bound, 0
        if self._reference_design_signature(candidate) == center_signature:
            return lower_bound, 0

        sub = self.solve_subproblem(candidate, label=f"local_branch{iteration}_{self._local_branching_calls}")
        if sub.obj_value is None:
            if sub.status in {GRB.INFEASIBLE, GRB.INF_OR_UNBD}:
                self._add_no_good_cut(candidate)
                return lower_bound, 1
            return lower_bound, 0

        total_obj = sub.obj_value - self._fixed_cost_value(candidate)
        if total_obj > lower_bound + 1e-7:
            lower_bound = total_obj
            self.best_total_obj = total_obj
            self.best_operational_obj = sub.obj_value
            self.best_design = candidate

        if sub.cut_expr_data is None or sub.rhs_at_reference is None:
            return lower_bound, 0

        self._add_benders_cut(sub, "local_branch")
        return lower_bound, 1

    def solve(
        self,
        max_iterations: Optional[int] = None,
        tolerance: Optional[float] = None,
        time_limit: Optional[float] = None,
    ) -> bool:
        """Solve with stable Pareto updates while preserving the base loop."""
        self.build_master()
        assert self.master is not None
        self.apply_master_solver_config()

        max_iterations = int(max_iterations if max_iterations is not None else getattr(config, "BENDERS_MAX_ITER", 100))
        tolerance = float(tolerance if tolerance is not None else getattr(config, "BENDERS_TOL", 1e-4))
        time_limit = float(time_limit if time_limit is not None else getattr(config, "BENDERS_TIME_LIMIT", getattr(config, "MODEL_TIME_LIMIT", 3600.0)))
        import time

        t0 = time.time()
        lower_bound = -GRB.INFINITY
        upper_bound = GRB.INFINITY
        lower_bound, initial_cut_count = self._process_initial_designs(lower_bound)
        core_min_obj = float(getattr(config, "BENDERS_PARETO_CORE_MIN_OBJ", 1e-6))

        for it in range(1, max_iterations + 1):
            elapsed = time.time() - t0
            if elapsed >= time_limit:
                print(f"Experimental Benders stopped by time limit before iteration {it}.")
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

            sub = self.solve_subproblem(design, label=f"exp_iter{it}")
            if sub.obj_value is None:
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
            if sub.cut_expr_data is None or sub.rhs_at_reference is None:
                if zero_flow_cut_added:
                    print(f"Iteration {it}: zero-flow subproblem duals unavailable; added Gamma-lift cut.")
                    continue
                print(f"Iteration {it}: incumbent subproblem duals are unavailable; stopping.")
                break
            self._add_benders_cut(sub, "benders")

            pareto_added = 0
            if self.use_pareto_cuts:
                allow_core_update = (not self.pareto_positive_core_only) or total_obj > core_min_obj
                if allow_core_update:
                    self._initialize_or_update_core(design)
                    for lam in (self.pareto_lambda, *self.pareto_retry_lambdas):
                        ref_design = self._pareto_reference_design_with_lambda(design, lam)
                        pareto_sub = self.solve_subproblem(ref_design, label=f"exp_pareto{it}_lam{lam:g}")
                        if pareto_sub.obj_value is not None and pareto_sub.cut_expr_data is not None:
                            self._add_benders_cut(pareto_sub, "pareto")
                            pareto_added = 1
                            break
                    if pareto_added == 0:
                        print(f"Iteration {it}: Pareto reference subproblem failed for all retry lambdas.")

            lower_bound, integer_ref_cuts = self._add_integer_reference_cuts(
                iteration=it,
                reference_designs=integer_reference_designs,
                lower_bound=lower_bound,
            )
            local_branching_cuts = 0
            if self.best_design is not None:
                lower_bound, local_branching_cuts = self._run_local_branching_reference(
                    iteration=it,
                    center_design=self.best_design,
                    lower_bound=lower_bound,
                    remaining_time=time_limit - (time.time() - t0),
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
                    "pareto_added": pareto_added,
                    "integer_ref_cuts": integer_ref_cuts,
                    "local_branching_cuts": local_branching_cuts,
                    "time": time.time() - t0,
                }
            )
            print(
                f"ExpIter {it:03d}: UB={upper_bound:.4f}, LB={lower_bound:.4f}, "
                f"gap={rel_gap:.4%}, eta={eta_val:.4f}, sub={sub.obj_value:.4f}, "
                f"cuts={self.cut_count}, lb_cuts={local_branching_cuts}"
            )

            if abs_gap <= tolerance * max(1.0, abs(lower_bound)):
                print("Experimental Benders converged.")
                break

        return self.best_design is not None


class SinglePeriodPartialBendersRegimeIISolver(BendersRegimeIISolver):
    """Single-period partial Benders.

    The master keeps aggregate logical flow and queue variables
    ``rho/alpha/beta/psi/xi``.  The recourse LP keeps only route decomposition,
    route activation, energy and fleet feasibility through ``varphi``.
    """

    def __init__(
        self,
        instance: Instance,
        *,
        max_iter: int = 100,
        gap_tol: float = 1e-4,
        master_time_limit: float = 30.0,
        master_mip_gap: float = 1e-5,
        recourse_penalty: float = 1e5,
        keep_route_feasibility: bool = True,
        embed_route_cost: bool = False,
        validate_incumbents: bool = True,
        verbose: bool = True,
    ):
        super().__init__(instance, use_pareto_cuts=False)
        self.max_iter = int(max_iter)
        self.gap_tol = float(gap_tol)
        self.master_time_limit = float(master_time_limit)
        self.master_mip_gap = float(master_mip_gap)
        self.recourse_penalty = float(recourse_penalty)
        self.keep_route_feasibility = bool(keep_route_feasibility)
        self.embed_route_cost = bool(embed_route_cost)
        self.validate_incumbents = bool(validate_incumbents)
        self.verbose = bool(verbose)
        self.partial_master: Optional[gp.Model] = None
        self.partial_vars: Dict[str, object] = {}
        self.partial_history: List[Dict[str, Number]] = []
        self.partial_cut_count = 0

    def build_partial_master(self) -> None:
        if self.partial_master is not None:
            return

        m = gp.Model("MLND_RegimeII_SinglePeriod_PartialBenders_Master")
        m.setParam("OutputFlag", int(getattr(config, "BENDERS_PARTIAL_MASTER_LOG", 0)))
        m.setParam("Threads", int(getattr(config, "BENDERS_MASTER_THREADS", 1)))
        m.setParam("TimeLimit", self.master_time_limit)
        m.setParam("MIPGap", self.master_mip_gap)

        C = range(len(config.MU_HUB))
        mu_c = config.MU_HUB
        tau_H = float(config.TAU_H)
        tau_S = float(config.TAU_S)
        revenue_coef = float(getattr(config, "ORDER_REVENUE_COEF", 0.0))
        total_demand = max(1.0, sum(float(value) for value in self.inst.h.values()))

        x = m.addVars(self.N_x, vtype=GRB.BINARY, name="x")
        y = m.addVars(self.AY, vtype=GRB.BINARY, name="y")
        z = m.addVars(self.H, C, vtype=GRB.BINARY, name="z")
        Gamma = m.addVar(lb=0.0, vtype=GRB.INTEGER, name="Gamma")
        theta = m.addVar(lb=-GRB.INFINITY, ub=0.0, name="theta_route")
        rho_l = m.addVars(self.L, lb=0.0, name="rho_l")
        rho_s = m.addVars(self.S, lb=0.0, name="rho_s")
        alpha_keys: List[AlphaKey] = [(l, i, j) for (l, i) in self.A_LH for (i2, j) in self.A_HH if i2 == i]
        alpha = m.addVars(alpha_keys, lb=0.0, name="alpha")
        beta = m.addVars(list(self.A_HH), lb=0.0, name="beta")
        psi = m.addVars(self.A, lb=0.0, name="psi")
        lambd = m.addVars(self.H, lb=0.0, name="lambda")
        xi = m.addVars(self.HLS, lb=0.0, name="xi")
        route_ready_keys: List[RouteKey] = [
            (u, v, k)
            for (u, v) in self.A_hat
            for k in self.route_options.get((u, v), [])
            if k in self.E
        ]
        route_ready = m.addVars(route_ready_keys, lb=0.0, ub=1.0, name="partial_route_ready")
        varphi_keys: List[RouteKey] = [
            (u, v, k)
            for (u, v) in self.A_F
            for k in self.route_options.get((u, v), [])
        ]
        route_varphi = (
            m.addVars(varphi_keys, lb=0.0, name="partial_varphi")
            if self.keep_route_feasibility
            else None
        )

        q_nodes = self.L + self.S
        q_v1 = m.addVars(q_nodes, lb=-GRB.INFINITY, name="q_v1")
        q_v2 = m.addVars(q_nodes, lb=-GRB.INFINITY, name="q_v2")
        q_t = m.addVars(q_nodes, lb=0.0, name="q_t")
        h_v1 = m.addVars(self.H, lb=-GRB.INFINITY, name="h_v1")
        h_v2 = m.addVars(self.H, lb=-GRB.INFINITY, name="h_v2")
        h_t = m.addVars(self.H, lb=0.0, name="h_t")

        fixed_cost = gp.quicksum(float(self.inst.f.get(l, config.COST_BASE)) * x[l] for l in self.L)
        fixed_cost += gp.quicksum(float(getattr(config, "COST_HUB_FIXED", 0.0)) * x[i] for i in self.H)
        fixed_cost += gp.quicksum(float(self.inst.f.get(k, config.COST_SWAPPING)) * x[k] for k in self.E)
        fixed_cost += gp.quicksum(float(config.COST_HUB[c]) * z[i, c] for i in self.H for c in C)
        fixed_cost += gp.quicksum(self.arc_cost[u, v] * y[u, v] for u, v in self.AY)
        fixed_cost += float(config.COST_DRONE) * Gamma
        revenue = gp.quicksum(revenue_coef * psi[i, j] for i, j in self.A_HH)
        m.setObjective(revenue + theta - fixed_cost, GRB.MAXIMIZE)

        for i in self.H:
            assigned_bases = [l for (l, h) in self.A_LH if h == i]
            m.addConstr(gp.quicksum(y[l, i] for l in assigned_bases if (l, i) in y) == x[i], name=f"assign_{i}")
            m.addConstr(gp.quicksum(z[i, c] for c in C) == x[i], name=f"capacity_{i}")
            m.addConstr(gp.quicksum(float(mu_c[c]) * z[i, c] for c in C) >= x[i] / tau_H, name=f"hub_min_slack_{i}")

        for l, i in self.A_LH:
            m.addConstr(y[l, i] <= x[l], name=f"assign_active_{l}_{i}")
        for u, v in self.AY:
            if (u, v) in self.A_LH:
                continue
            m.addConstr(y[u, v] <= self._node_activation_expr({"x": x}, u), name=f"arc_tail_active_{u}_{v}")
            m.addConstr(y[u, v] <= self._node_activation_expr({"x": x}, v), name=f"arc_head_active_{u}_{v}")

        for l in self.L:
            m.addConstr(rho_l[l] <= x[l], name=f"rho_L_{l}")
        for i, s in self.A_HS:
            m.addConstr(rho_s[s] <= x[i], name=f"rho_S_{s}")

        for l, i, j in alpha_keys:
            m.addConstr(alpha[l, i, j] <= rho_l[l], name=f"alpha_1_{l}_{i}_{j}")
            m.addConstr(alpha[l, i, j] <= y[l, i], name=f"alpha_2_{l}_{i}_{j}")
            m.addConstr(alpha[l, i, j] <= y[i, j], name=f"alpha_3_{l}_{i}_{j}")
            m.addConstr(alpha[l, i, j] >= rho_l[l] + y[l, i] + y[i, j] - 2.0, name=f"alpha_4_{l}_{i}_{j}")

        for s, i in self.A_SH:
            for h, j in self.A_HH:
                if h != i:
                    continue
                m.addConstr(beta[i, j] <= rho_s[s], name=f"beta_1_{i}_{j}")
                m.addConstr(beta[i, j] <= y[i, j], name=f"beta_2_{i}_{j}")
                m.addConstr(beta[i, j] >= rho_s[s] + y[i, j] - 1.0, name=f"beta_3_{i}_{j}")

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

        for i, s in self.A_HS:
            m.addConstr(psi[i, s] <= total_demand * x[i], name=f"psi_hs_active_{i}_{s}")

        for u, v in self.A_hat:
            if (u, v) in y:
                m.addConstr(psi[u, v] <= total_demand * y[u, v], name=f"psi_arc_active_{u}_{v}")
            options = self.route_options.get((u, v), [])
            if not options:
                m.addConstr(psi[u, v] == 0.0, name=f"psi_no_route_{u}_{v}")
                continue
            ready_terms = []
            if "0" in options and (u, v) in y:
                ready_terms.append(y[u, v])
            for k in options:
                if k not in self.E:
                    continue
                ready = route_ready[u, v, k]
                m.addConstr(ready <= y[u, v], name=f"partial_ready_arc_{u}_{v}_{k}")
                m.addConstr(ready <= y[u, k], name=f"partial_ready_in_{u}_{v}_{k}")
                m.addConstr(ready <= y[k, v], name=f"partial_ready_out_{u}_{v}_{k}")
                ready_terms.append(ready)
            if ready_terms:
                m.addConstr(psi[u, v] <= total_demand * gp.quicksum(ready_terms), name=f"psi_route_ready_{u}_{v}")
            else:
                m.addConstr(psi[u, v] == 0.0, name=f"psi_no_ready_route_{u}_{v}")

        for q in self.HLS:
            m.addConstr(
                gp.quicksum(psi[u, q] for (u, v) in self.A if v == q)
                == gp.quicksum(psi[q, v] for (u, v) in self.A if u == q),
                name=f"flow_balance_{q}",
            )

        mu_expr = {i: gp.quicksum(float(mu_c[c]) * z[i, c] for c in C) for i in self.H}
        sqrt_mu_expr = {i: gp.quicksum(math.sqrt(float(mu_c[c])) * z[i, c] for c in C) for i in self.H}
        for i in self.H:
            m.addConstr(lambd[i] == gp.quicksum(psi[u, i] for (u, v) in self.A if v == i), name=f"lambda_{i}")
            m.addConstr(mu_expr[i] - lambd[i] >= x[i] / tau_H, name=f"hub_wait_{i}")

        for s in self.S:
            m.addConstr(xi[s] <= tau_S * gp.quicksum(psi[s, v] for (u, v) in self.A if u == s), name=f"mean_S_{s}")

        for q in q_nodes:
            rho_q = rho_l[q] if q in self.L else rho_s[q]
            m.addConstr(q_v1[q] == 2.0, name=f"q_v1_{q}")
            m.addConstr(q_v2[q] == xi[q] + rho_q, name=f"q_v2_{q}")
            m.addConstr(q_t[q] == xi[q] - rho_q + 2.0, name=f"q_t_{q}")
            m.addQConstr(q_v1[q] * q_v1[q] + q_v2[q] * q_v2[q] <= q_t[q] * q_t[q], name=f"queue_{q}")

        for i in self.H:
            m.addConstr(h_v1[i] == 2.0 * sqrt_mu_expr[i], name=f"h_v1_{i}")
            m.addConstr(h_v2[i] == xi[i] + lambd[i] - mu_expr[i] + 1.0, name=f"h_v2_{i}")
            m.addConstr(h_t[i] == xi[i] - lambd[i] + mu_expr[i] + 1.0, name=f"h_t_{i}")
            m.addQConstr(h_v1[i] * h_v1[i] + h_v2[i] * h_v2[i] <= h_t[i] * h_t[i], name=f"queue_H_{i}")

        if self.keep_route_feasibility:
            omega = float(config.ENERGY_RATE_OMEGA)
            usable_battery = (
                1.0 - float(getattr(config, "ENERGY_EPSILON", getattr(config, "EPS", 0.0)))
            ) * float(config.BATTERY_CAPACITY_OMEGA_UPPER)

            for u, v in self.A_hat:
                m.addConstr(
                    gp.quicksum(
                        route_varphi[u, v, k]
                        for k in self.route_options.get((u, v), [])
                        if (u, v, k) in route_varphi
                    )
                    == psi[u, v],
                    name=f"partial_route_flow_{u}_{v}",
                )

            for u, v, k in varphi_keys:
                big_m = self.route_big_m[u, v, k]
                m.addConstr(route_varphi[u, v, k] <= big_m * y[u, v], name=f"partial_route_arc_active_{u}_{v}_{k}")
                if k in self.E:
                    m.addConstr(route_varphi[u, v, k] <= big_m * y[u, k], name=f"partial_route_swap_in_{u}_{v}_{k}")
                    m.addConstr(route_varphi[u, v, k] <= big_m * y[k, v], name=f"partial_route_swap_out_{u}_{v}_{k}")

            for l in self.L:
                m.addConstr(
                    omega
                    * gp.quicksum(
                        self.t[l, i, k] * route_varphi[l, i, k]
                        for (u, i, k) in varphi_keys
                        if u == l and (l, i) in self.A_LH
                    )
                    - usable_battery
                    * gp.quicksum(
                        route_varphi[l, i, k]
                        for (u, i, k) in varphi_keys
                        if u == l and (l, i) in self.A_LH and k in self.E
                    )
                    <= 0.0,
                    name=f"partial_energy_L_{l}",
                )

            for i, s in self.A_HS:
                lhs = omega * (
                    gp.quicksum(
                        self.t[i, j, k] * route_varphi[i, j, k]
                        for (h, j, k) in varphi_keys
                        if h == i and (i, j) in self.A_HH
                    )
                    + gp.quicksum(
                        self.t[s, v, k] * route_varphi[s, v, k]
                        for (u, v, k) in varphi_keys
                        if u == s and ((s, v) in self.A_SL or (s, v) in self.A_SS)
                    )
                )
                lhs -= usable_battery * (
                    gp.quicksum(
                        route_varphi[i, j, k]
                        for (h, j, k) in varphi_keys
                        if h == i and (i, j) in self.A_HH and k in self.E
                    )
                    + gp.quicksum(
                        route_varphi[s, v, k]
                        for (u, v, k) in varphi_keys
                        if u == s and ((s, v) in self.A_SL or (s, v) in self.A_SS) and k in self.E
                    )
                )
                m.addConstr(lhs <= -omega * (xi[i] + xi[s]), name=f"partial_energy_HS_{i}")

            route_total_flight_time = gp.quicksum(
                self.t[u, v, k] * route_varphi[u, v, k]
                for u, v, k in varphi_keys
            )
            m.addConstr(
                gp.quicksum(xi[u] for u in self.HLS) + route_total_flight_time <= Gamma,
                name="partial_route_fleet",
            )
            if self.embed_route_cost:
                m.addConstr(
                    theta
                    <= -gp.quicksum(
                        self.flight_cost[u, v, k] * route_varphi[u, v, k]
                        for u, v, k in varphi_keys
                    ),
                    name="partial_embedded_route_cost",
                )

        min_route_time = {
            (u, v): min(float(self.t[u, v, k]) for k in self.route_options.get((u, v), []))
            for u, v in self.A_hat
            if self.route_options.get((u, v), [])
        }
        min_route_cost = {
            (u, v): min(
                float(self.flight_cost[u, v, k])
                for k in self.route_options.get((u, v), [])
            )
            for u, v in self.A_hat
            if self.route_options.get((u, v), [])
        }
        m.addConstr(
            theta
            <= -gp.quicksum(
                min_route_cost[u, v] * psi[u, v]
                for u, v in self.A_hat
                if (u, v) in min_route_cost
            ),
            name="partial_min_route_cost_lb",
        )
        m.addConstr(
            gp.quicksum(xi[u] for u in self.HLS)
            + gp.quicksum(min_route_time[u, v] * psi[u, v] for u, v in self.A_hat if (u, v) in min_route_time)
            <= Gamma,
            name="fleet_time_lb",
        )

        self.partial_master = m
        self.partial_vars = {
            "x": x,
            "y": y,
            "z": z,
            "Gamma": Gamma,
            "theta": theta,
            "rho_l": rho_l,
            "rho_s": rho_s,
            "alpha": alpha,
            "beta": beta,
            "psi": psi,
            "route_ready": route_ready,
            "route_varphi": route_varphi,
            "lambda": lambd,
            "xi": xi,
        }

    def _extract_partial_solution(self) -> PartialMasterSolution:
        assert self.partial_master is not None
        x = self.partial_vars["x"]
        y = self.partial_vars["y"]
        z = self.partial_vars["z"]
        Gamma = self.partial_vars["Gamma"]
        theta = self.partial_vars["theta"]
        psi = self.partial_vars["psi"]
        xi = self.partial_vars["xi"]
        C = range(len(config.MU_HUB))
        return PartialMasterSolution(
            obj=float(self.partial_master.ObjVal),
            x={u: float(x[u].X) for u in self.N_x},
            y={(u, v): float(y[u, v].X) for u, v in self.AY},
            z={(i, c): float(z[i, c].X) for i in self.H for c in C},
            Gamma=float(Gamma.X),
            theta=float(theta.X),
            psi={(u, v): float(psi[u, v].X) for u, v in self.A},
            xi={u: float(xi[u].X) for u in self.HLS},
        )

    @staticmethod
    def _rhs_partial_value(
        rhs_fun: Callable[[Mapping[str, object]], Union[float, gp.LinExpr]],
        design: Mapping[str, object],
    ) -> float:
        value = rhs_fun(design)
        if isinstance(value, (int, float)):
            return float(value)
        return float(value.getValue())

    def _solve_route_recourse_soft(self, sol: PartialMasterSolution, label: str = "partial") -> RouteRecourseResult:
        m = gp.Model(f"MLND_RegimeII_RouteRecourse_{label}")
        m.setParam("OutputFlag", int(getattr(config, "BENDERS_PARTIAL_RECOURSE_LOG", 0)))
        m.setParam("Threads", int(getattr(config, "BENDERS_SUB_THREADS", 1)))
        m.setParam("DualReductions", 0)
        m.setParam("FeasibilityTol", float(getattr(config, "BENDERS_PARTIAL_RECOURSE_FEAS_TOL", 1e-5)))
        m.setParam("OptimalityTol", 1e-8)

        omega = float(config.ENERGY_RATE_OMEGA)
        usable_battery = (1.0 - float(getattr(config, "ENERGY_EPSILON", getattr(config, "EPS", 0.0)))) * float(config.BATTERY_CAPACITY_OMEGA_UPPER)
        varphi_keys: List[RouteKey] = [
            (u, v, k)
            for (u, v) in self.A_F
            for k in self.route_options.get((u, v), [])
        ]
        varphi = m.addVars(varphi_keys, lb=0.0, name="varphi")
        route_pos = m.addVars(self.A_hat, lb=0.0, name="route_pos")
        route_neg = m.addVars(self.A_hat, lb=0.0, name="route_neg")
        energy_l_slack = m.addVars(self.L, lb=0.0, name="energy_l_slack")
        energy_hs_slack = m.addVars(self.A_HS, lb=0.0, name="energy_hs_slack")
        fleet_slack = m.addVar(lb=0.0, name="fleet_slack")

        m.setObjective(
            gp.quicksum(self.flight_cost[u, v, k] * varphi[u, v, k] for u, v, k in varphi_keys)
            + self.recourse_penalty
            * (
                gp.quicksum(route_pos[u, v] + route_neg[u, v] for u, v in self.A_hat)
                + gp.quicksum(energy_l_slack[l] for l in self.L)
                + gp.quicksum(energy_hs_slack[i, s] for i, s in self.A_HS)
                + fleet_slack
            ),
            GRB.MINIMIZE,
        )

        fixed_design = {
            "y": sol.y,
            "psi": sol.psi,
            "xi": sol.xi,
            "Gamma": sol.Gamma,
        }
        param_constrs: List[Tuple[gp.Constr, Callable[[Mapping[str, object]], Union[float, gp.LinExpr]], float]] = []

        def add_le(lhs, rhs_fun, name: str):
            rhs_val = self._rhs_partial_value(rhs_fun, fixed_design)
            con = m.addConstr(lhs <= rhs_val, name=name)
            param_constrs.append((con, rhs_fun, rhs_val))
            return con

        def add_eq(lhs, rhs_fun, name: str):
            rhs_val = self._rhs_partial_value(rhs_fun, fixed_design)
            con = m.addConstr(lhs == rhs_val, name=name)
            param_constrs.append((con, rhs_fun, rhs_val))
            return con

        for u, v in self.A_hat:
            add_eq(
                gp.quicksum(varphi[u, v, k] for k in self.route_options.get((u, v), []) if (u, v, k) in varphi)
                + route_pos[u, v]
                - route_neg[u, v],
                lambda d, u=u, v=v: d["psi"][u, v],
                f"route_{u}_{v}",
            )

        for u, v, k in varphi_keys:
            big_m = self.route_big_m[u, v, k]
            add_le(varphi[u, v, k], lambda d, u=u, v=v, big_m=big_m: big_m * d["y"][u, v], f"route_arc_active_{u}_{v}_{k}")
            if k in self.E:
                add_le(varphi[u, v, k], lambda d, u=u, k=k, big_m=big_m: big_m * d["y"][u, k], f"route_swap_in_{u}_{v}_{k}")
                add_le(varphi[u, v, k], lambda d, k=k, v=v, big_m=big_m: big_m * d["y"][k, v], f"route_swap_out_{u}_{v}_{k}")

        for l in self.L:
            m.addConstr(
                omega
                * gp.quicksum(
                    self.t[l, i, k] * varphi[l, i, k]
                    for (u, i, k) in varphi_keys
                    if u == l and (l, i) in self.A_LH
                )
                - usable_battery
                * gp.quicksum(
                    varphi[l, i, k]
                    for (u, i, k) in varphi_keys
                    if u == l and (l, i) in self.A_LH and k in self.E
                )
                - energy_l_slack[l]
                <= 0.0,
                name=f"energy_L_{l}",
            )

        for i, s in self.A_HS:
            lhs = omega * (
                gp.quicksum(self.t[i, j, k] * varphi[i, j, k] for (h, j, k) in varphi_keys if h == i and (i, j) in self.A_HH)
                + gp.quicksum(
                    self.t[s, v, k] * varphi[s, v, k]
                    for (u, v, k) in varphi_keys
                    if u == s and ((s, v) in self.A_SL or (s, v) in self.A_SS)
                )
            )
            lhs -= usable_battery * (
                gp.quicksum(varphi[i, j, k] for (h, j, k) in varphi_keys if h == i and (i, j) in self.A_HH and k in self.E)
                + gp.quicksum(
                    varphi[s, v, k]
                    for (u, v, k) in varphi_keys
                    if u == s and ((s, v) in self.A_SL or (s, v) in self.A_SS) and k in self.E
                )
            )
            lhs -= energy_hs_slack[i, s]
            add_le(lhs, lambda d, i=i, s=s, omega=omega: -omega * (d["xi"][i] + d["xi"][s]), f"energy_HS_{i}_{s}")

        total_flight_time = gp.quicksum(self.t[u, v, k] * varphi[u, v, k] for u, v, k in varphi_keys)
        add_le(
            total_flight_time - fleet_slack,
            lambda d: d["Gamma"] - sum(d["xi"][u] for u in self.HLS),
            "fleet",
        )

        m.optimize()
        if m.status != GRB.OPTIMAL or m.SolCount == 0:
            return RouteRecourseResult(m.status, None, None, None, math.inf, label)

        cut_expr_data: List[Tuple[float, Callable[[Mapping[str, object]], gp.LinExpr]]] = []
        rhs_values: List[float] = []
        dual_tol = float(getattr(config, "BENDERS_DUAL_TOL", 1e-8))
        for con, rhs_fun, rhs_val in param_constrs:
            pi = float(con.Pi)
            if abs(pi) > dual_tol:
                cut_expr_data.append((pi, rhs_fun))
                rhs_values.append(rhs_val)

        slack_total = (
            sum(max(0.0, route_pos[u, v].X) + max(0.0, route_neg[u, v].X) for u, v in self.A_hat)
            + sum(max(0.0, energy_l_slack[l].X) for l in self.L)
            + sum(max(0.0, energy_hs_slack[i, s].X) for i, s in self.A_HS)
            + max(0.0, fleet_slack.X)
        )
        return RouteRecourseResult(
            status=m.status,
            obj_value=float(m.ObjVal),
            cut_expr_data=cut_expr_data,
            rhs_at_reference=rhs_values,
            slack=float(slack_total),
            label=label,
        )

    def _build_route_recourse_lp(
        self,
        sol: PartialMasterSolution,
        label: str,
        *,
        allow_slack: bool,
    ) -> Tuple[
        gp.Model,
        List[Tuple[gp.Constr, Callable[[Mapping[str, object]], Union[float, gp.LinExpr]], float]],
        Dict[str, object],
    ]:
        m = gp.Model(f"MLND_RegimeII_RouteRecourse_{label}")
        m.setParam("OutputFlag", int(getattr(config, "BENDERS_PARTIAL_RECOURSE_LOG", 0)))
        m.setParam("Threads", int(getattr(config, "BENDERS_SUB_THREADS", 1)))
        m.setParam("DualReductions", 0)
        m.setParam(
            "FeasibilityTol",
            float(getattr(config, "BENDERS_PARTIAL_RECOURSE_FEAS_TOL", 1e-6)),
        )
        m.setParam("OptimalityTol", 1e-8)
        m.setParam(
            "NumericFocus",
            int(getattr(config, "BENDERS_PARTIAL_RECOURSE_NUMERIC_FOCUS", 2)),
        )

        omega = float(config.ENERGY_RATE_OMEGA)
        usable_battery = (
            1.0 - float(getattr(config, "ENERGY_EPSILON", getattr(config, "EPS", 0.0)))
        ) * float(config.BATTERY_CAPACITY_OMEGA_UPPER)
        varphi_keys: List[RouteKey] = [
            (u, v, k)
            for (u, v) in self.A_F
            for k in self.route_options.get((u, v), [])
        ]
        varphi = m.addVars(varphi_keys, lb=0.0, name="varphi")
        if allow_slack:
            route_pos = m.addVars(self.A_hat, lb=0.0, name="route_pos")
            route_neg = m.addVars(self.A_hat, lb=0.0, name="route_neg")
            energy_l_slack = m.addVars(self.L, lb=0.0, name="energy_l_slack")
            energy_hs_slack = m.addVars(self.A_HS, lb=0.0, name="energy_hs_slack")
            fleet_slack = m.addVar(lb=0.0, name="fleet_slack")
        else:
            route_pos = None
            route_neg = None
            energy_l_slack = None
            energy_hs_slack = None
            fleet_slack = None

        fixed_design = {
            "y": sol.y,
            "psi": sol.psi,
            "xi": sol.xi,
            "Gamma": sol.Gamma,
        }
        param_constrs: List[Tuple[gp.Constr, Callable[[Mapping[str, object]], Union[float, gp.LinExpr]], float]] = []

        def add_le(lhs, rhs_fun, name: str):
            rhs_val = self._rhs_partial_value(rhs_fun, fixed_design)
            con = m.addConstr(lhs <= rhs_val, name=name)
            param_constrs.append((con, rhs_fun, rhs_val))
            return con

        def add_eq(lhs, rhs_fun, name: str):
            rhs_val = self._rhs_partial_value(rhs_fun, fixed_design)
            con = m.addConstr(lhs == rhs_val, name=name)
            param_constrs.append((con, rhs_fun, rhs_val))
            return con

        for u, v in self.A_hat:
            route_expr = gp.quicksum(
                varphi[u, v, k]
                for k in self.route_options.get((u, v), [])
                if (u, v, k) in varphi
            )
            if allow_slack:
                route_expr += route_pos[u, v] - route_neg[u, v]
            add_eq(route_expr, lambda d, u=u, v=v: d["psi"][u, v], f"route_{u}_{v}")

        for u, v, k in varphi_keys:
            big_m = self.route_big_m[u, v, k]
            add_le(varphi[u, v, k], lambda d, u=u, v=v, big_m=big_m: big_m * d["y"][u, v], f"route_arc_active_{u}_{v}_{k}")
            if k in self.E:
                add_le(varphi[u, v, k], lambda d, u=u, k=k, big_m=big_m: big_m * d["y"][u, k], f"route_swap_in_{u}_{v}_{k}")
                add_le(varphi[u, v, k], lambda d, k=k, v=v, big_m=big_m: big_m * d["y"][k, v], f"route_swap_out_{u}_{v}_{k}")

        for l in self.L:
            lhs = (
                omega
                * gp.quicksum(
                    self.t[l, i, k] * varphi[l, i, k]
                    for (u, i, k) in varphi_keys
                    if u == l and (l, i) in self.A_LH
                )
                - usable_battery
                * gp.quicksum(
                    varphi[l, i, k]
                    for (u, i, k) in varphi_keys
                    if u == l and (l, i) in self.A_LH and k in self.E
                )
            )
            if allow_slack:
                lhs -= energy_l_slack[l]
            m.addConstr(lhs <= 0.0, name=f"energy_L_{l}")

        for i, s in self.A_HS:
            lhs = omega * (
                gp.quicksum(self.t[i, j, k] * varphi[i, j, k] for (h, j, k) in varphi_keys if h == i and (i, j) in self.A_HH)
                + gp.quicksum(
                    self.t[s, v, k] * varphi[s, v, k]
                    for (u, v, k) in varphi_keys
                    if u == s and ((s, v) in self.A_SL or (s, v) in self.A_SS)
                )
            )
            lhs -= usable_battery * (
                gp.quicksum(varphi[i, j, k] for (h, j, k) in varphi_keys if h == i and (i, j) in self.A_HH and k in self.E)
                + gp.quicksum(
                    varphi[s, v, k]
                    for (u, v, k) in varphi_keys
                    if u == s and ((s, v) in self.A_SL or (s, v) in self.A_SS) and k in self.E
                )
            )
            if allow_slack:
                lhs -= energy_hs_slack[i, s]
            add_le(lhs, lambda d, i=i, s=s, omega=omega: -omega * (d["xi"][i] + d["xi"][s]), f"energy_HS_{i}_{s}")

        total_flight_time = gp.quicksum(self.t[u, v, k] * varphi[u, v, k] for u, v, k in varphi_keys)
        fleet_lhs = total_flight_time
        if allow_slack:
            fleet_lhs -= fleet_slack
        add_le(
            fleet_lhs,
            lambda d: d["Gamma"] - sum(d["xi"][u] for u in self.HLS),
            "fleet",
        )

        if allow_slack:
            total_slack = (
                gp.quicksum(route_pos[u, v] + route_neg[u, v] for u, v in self.A_hat)
                + gp.quicksum(energy_l_slack[l] for l in self.L)
                + gp.quicksum(energy_hs_slack[i, s] for i, s in self.A_HS)
                + fleet_slack
            )
            m.setObjective(total_slack, GRB.MINIMIZE)
        else:
            m.setObjective(
                gp.quicksum(self.flight_cost[u, v, k] * varphi[u, v, k] for u, v, k in varphi_keys),
                GRB.MINIMIZE,
            )

        return m, param_constrs, {
            "route_pos": route_pos,
            "route_neg": route_neg,
            "energy_l_slack": energy_l_slack,
            "energy_hs_slack": energy_hs_slack,
            "fleet_slack": fleet_slack,
        }

    def _collect_recourse_duals(
        self,
        param_constrs: List[Tuple[gp.Constr, Callable[[Mapping[str, object]], Union[float, gp.LinExpr]], float]],
    ) -> Tuple[List[Tuple[float, Callable[[Mapping[str, object]], gp.LinExpr]]], List[float]]:
        cut_expr_data: List[Tuple[float, Callable[[Mapping[str, object]], gp.LinExpr]]] = []
        rhs_values: List[float] = []
        dual_tol = float(getattr(config, "BENDERS_DUAL_TOL", 1e-8))
        for con, rhs_fun, rhs_val in param_constrs:
            pi = float(con.Pi)
            if abs(pi) > dual_tol:
                cut_expr_data.append((pi, rhs_fun))
                rhs_values.append(rhs_val)
        return cut_expr_data, rhs_values

    def _route_slack_components(self, aux: Mapping[str, object]) -> Dict[str, float]:
        route_pos = aux["route_pos"]
        route_neg = aux["route_neg"]
        energy_l_slack = aux["energy_l_slack"]
        energy_hs_slack = aux["energy_hs_slack"]
        fleet_slack = aux["fleet_slack"]
        route_slack = sum(max(0.0, route_pos[u, v].X) + max(0.0, route_neg[u, v].X) for u, v in self.A_hat)
        energy_l = sum(max(0.0, energy_l_slack[l].X) for l in self.L)
        energy_hs = sum(max(0.0, energy_hs_slack[i, s].X) for i, s in self.A_HS)
        fleet = max(0.0, fleet_slack.X)
        return {
            "route": float(route_slack),
            "energy_l": float(energy_l),
            "energy_hs": float(energy_hs),
            "fleet": float(fleet),
            "total": float(route_slack + energy_l + energy_hs + fleet),
        }

    def solve_route_recourse(self, sol: PartialMasterSolution, label: str = "partial") -> RouteRecourseResult:
        feas_model, feas_constrs, feas_aux = self._build_route_recourse_lp(sol, f"{label}_feas", allow_slack=True)
        feas_model.optimize()
        if feas_model.status != GRB.OPTIMAL or feas_model.SolCount == 0:
            return RouteRecourseResult(feas_model.status, None, None, None, math.inf, label)

        feas_cut_data, feas_rhs_values = self._collect_recourse_duals(feas_constrs)
        slack_components = self._route_slack_components(feas_aux)
        slack_total = float(slack_components["total"])
        exact_slack_tol = float(
            getattr(
                config,
                "BENDERS_PARTIAL_EXACT_SLACK_TOL",
                getattr(config, "BENDERS_PARTIAL_SLACK_TOL", 1e-5),
            )
        )
        if slack_total > exact_slack_tol:
            return RouteRecourseResult(
                status=feas_model.status,
                obj_value=None,
                cut_expr_data=None,
                rhs_at_reference=None,
                slack=slack_total,
                label=label,
                feasibility_obj=float(feas_model.ObjVal),
                feasibility_cut_expr_data=feas_cut_data,
                feasibility_rhs_at_reference=feas_rhs_values,
                slack_components=slack_components,
            )

        cost_model, cost_constrs, _ = self._build_route_recourse_lp(sol, f"{label}_cost", allow_slack=False)
        cost_model.optimize()
        if cost_model.status != GRB.OPTIMAL or cost_model.SolCount == 0:
            accept_slack_tol = float(getattr(config, "BENDERS_PARTIAL_ACCEPT_SLACK_TOL", getattr(config, "BENDERS_PARTIAL_SLACK_TOL", 1e-5)))
            if slack_total <= accept_slack_tol:
                soft = self._solve_route_recourse_soft(sol, f"{label}_soft")
                if soft.obj_value is not None:
                    soft.feasibility_obj = float(feas_model.ObjVal)
                    soft.feasibility_cut_expr_data = feas_cut_data
                    soft.feasibility_rhs_at_reference = feas_rhs_values
                    soft.slack_components = slack_components
                    soft.slack = slack_total
                    return soft
            return RouteRecourseResult(
                status=cost_model.status,
                obj_value=None,
                cut_expr_data=None,
                rhs_at_reference=None,
                slack=slack_total,
                label=label,
                feasibility_obj=float(feas_model.ObjVal),
                feasibility_cut_expr_data=feas_cut_data,
                feasibility_rhs_at_reference=feas_rhs_values,
                slack_components=slack_components,
            )

        cost_cut_data, cost_rhs_values = self._collect_recourse_duals(cost_constrs)
        return RouteRecourseResult(
            status=cost_model.status,
            obj_value=float(cost_model.ObjVal),
            cut_expr_data=cost_cut_data,
            rhs_at_reference=cost_rhs_values,
            slack=slack_total,
            label=label,
            feasibility_obj=float(feas_model.ObjVal),
            feasibility_cut_expr_data=feas_cut_data,
            feasibility_rhs_at_reference=feas_rhs_values,
            slack_components=slack_components,
        )

    def _add_partial_feasibility_cut(self, recourse: RouteRecourseResult, name: str) -> None:
        assert self.partial_master is not None
        if (
            recourse.feasibility_obj is None
            or recourse.feasibility_cut_expr_data is None
            or recourse.feasibility_rhs_at_reference is None
        ):
            return

        master_dict = {
            "y": self.partial_vars["y"],
            "psi": self.partial_vars["psi"],
            "xi": self.partial_vars["xi"],
            "Gamma": self.partial_vars["Gamma"],
        }
        feas_lb = gp.LinExpr(float(recourse.feasibility_obj))
        for (pi, rhs_fun), rhs_ref in zip(recourse.feasibility_cut_expr_data, recourse.feasibility_rhs_at_reference):
            feas_lb += pi * (rhs_fun(master_dict) - rhs_ref)

        self.partial_cut_count += 1
        self.partial_master.addConstr(feas_lb <= 0.0, name=f"{name}_{self.partial_cut_count}")

    def _add_partial_cut(self, recourse: RouteRecourseResult, name: str) -> None:
        assert self.partial_master is not None
        if recourse.obj_value is None or recourse.cut_expr_data is None or recourse.rhs_at_reference is None:
            return

        theta = self.partial_vars["theta"]
        master_dict = {
            "y": self.partial_vars["y"],
            "psi": self.partial_vars["psi"],
            "xi": self.partial_vars["xi"],
            "Gamma": self.partial_vars["Gamma"],
        }
        recourse_lb = gp.LinExpr(float(recourse.obj_value))
        for (pi, rhs_fun), rhs_ref in zip(recourse.cut_expr_data, recourse.rhs_at_reference):
            recourse_lb += pi * (rhs_fun(master_dict) - rhs_ref)

        self.partial_cut_count += 1
        self.partial_master.addConstr(theta <= -recourse_lb, name=f"{name}_{self.partial_cut_count}")

    def _partial_revenue_value(self, sol: PartialMasterSolution) -> float:
        revenue_coef = float(getattr(config, "ORDER_REVENUE_COEF", 0.0))
        return revenue_coef * sum(sol.psi.get((i, j), 0.0) for i, j in self.A_HH)

    def solve(
        self,
        max_iterations: Optional[int] = None,
        tolerance: Optional[float] = None,
        time_limit: Optional[float] = None,
    ) -> bool:
        self.build_partial_master()
        assert self.partial_master is not None
        max_iterations = int(max_iterations if max_iterations is not None else self.max_iter)
        tolerance = float(tolerance if tolerance is not None else self.gap_tol)
        time_limit = float(time_limit if time_limit is not None else getattr(config, "BENDERS_TIME_LIMIT", getattr(config, "MODEL_TIME_LIMIT", 3600.0)))
        import time

        t0 = time.time()
        lower_bound = -GRB.INFINITY
        upper_bound = GRB.INFINITY
        slack_tol = float(getattr(config, "BENDERS_PARTIAL_SLACK_TOL", 1e-5))
        incumbent_floor = None
        incumbent_floor_tol = float(
            getattr(config, "BENDERS_PARTIAL_INCUMBENT_FLOOR_TOL", 1e-5)
        )

        for idx, design in enumerate(self.initial_designs, 1):
            x = self.partial_vars["x"]
            y = self.partial_vars["y"]
            z = self.partial_vars["z"]
            Gamma = self.partial_vars["Gamma"]
            for u in self.N_x:
                x[u].Start = float(design["x"].get(u, 0.0))
            for u, v in self.AY:
                y[u, v].Start = float(design["y"].get((u, v), 0.0))
            for i in self.H:
                for c in range(len(config.MU_HUB)):
                    z[i, c].Start = float(design["z"].get((i, c), 0.0))
            Gamma.Start = float(design.get("Gamma", 0.0))

            # A registered design is more than a MIP start: evaluating it in
            # the exact continuous subproblem provides a certified primal
            # bound before the partial master starts.  The objective-floor
            # constraint is a numerically useful cutoff and cannot remove an
            # optimal solution because it only discards points worse than an
            # already feasible incumbent.
            if time.time() - t0 < time_limit:
                initial_sub = self.solve_subproblem(
                    design,
                    label=f"partial_initial_{idx}",
                )
                if initial_sub.obj_value is not None:
                    initial_obj = float(initial_sub.obj_value) - self._fixed_cost_value(design)
                    if initial_obj > lower_bound + 1e-7:
                        lower_bound = initial_obj
                        self.best_total_obj = initial_obj
                        self.best_operational_obj = float(initial_sub.obj_value)
                        self.best_design = design

        has_certified_incumbent = lower_bound > -0.5 * float(GRB.INFINITY)
        if has_certified_incumbent:
            incumbent_floor = self.partial_master.addConstr(
                self.partial_master.getObjective()
                >= lower_bound - incumbent_floor_tol,
                name="partial_known_incumbent_floor",
            )
            self.partial_master.update()
            print(f"Partial Benders initial certified LB={lower_bound:.4f}.")

        for it in range(1, max_iterations + 1):
            elapsed = time.time() - t0
            if elapsed >= time_limit:
                print(f"Partial Benders stopped by time limit before iteration {it}.")
                break
            if self.best_design is not None:
                x = self.partial_vars["x"]
                y = self.partial_vars["y"]
                z = self.partial_vars["z"]
                Gamma = self.partial_vars["Gamma"]
                for u in self.N_x:
                    x[u].Start = float(self.best_design["x"].get(u, 0.0))
                for u, v in self.AY:
                    y[u, v].Start = float(self.best_design["y"].get((u, v), 0.0))
                for i in self.H:
                    for c in range(len(config.MU_HUB)):
                        z[i, c].Start = float(
                            self.best_design["z"].get((i, c), 0.0)
                        )
                Gamma.Start = float(self.best_design.get("Gamma", 0.0))
            self.partial_master.Params.TimeLimit = max(1.0, min(self.master_time_limit, time_limit - elapsed))
            self.partial_master.optimize()
            if (
                self.partial_master.status == GRB.TIME_LIMIT
                and self.partial_master.SolCount == 0
            ):
                retry_factors = tuple(
                    float(value)
                    for value in getattr(
                        config,
                        "BENDERS_PARTIAL_MASTER_RETRY_FACTORS",
                        (3.0, 6.0, 12.0),
                    )
                )
                for retry_factor in retry_factors:
                    remaining = time_limit - (time.time() - t0)
                    retry_limit = min(
                        retry_factor * self.master_time_limit,
                        remaining,
                    )
                    if retry_limit <= 1.0:
                        break
                    print(
                        f"Partial master iteration {it} returned no incumbent; "
                        f"retrying for {retry_limit:.1f}s."
                    )
                    self.partial_master.Params.TimeLimit = retry_limit
                    self.partial_master.optimize()
                    if self.partial_master.SolCount > 0:
                        break
                if self.partial_master.SolCount == 0:
                    remaining = time_limit - (time.time() - t0)
                    if remaining > 1.0:
                        print(
                            f"Partial master iteration {it} still has no incumbent; "
                            f"using the remaining {remaining:.1f}s."
                        )
                        self.partial_master.Params.TimeLimit = remaining
                        self.partial_master.optimize()
            if self.partial_master.status not in {GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL} or self.partial_master.SolCount == 0:
                print(f"Partial master did not return a solution. status={self.partial_master.status}")
                break

            master_bound = float(self.partial_master.ObjBound)
            if not math.isfinite(master_bound):
                master_bound = float(self.partial_master.ObjVal)
            # Numerical MIQCP solves have occasionally reported a bound below a
            # separately certified feasible incumbent.  Never propagate that
            # contradiction into the algorithmic gap.
            if lower_bound > -0.5 * float(GRB.INFINITY):
                master_bound = max(master_bound, lower_bound)
            upper_bound = min(upper_bound, master_bound)
            sol = self._extract_partial_solution()
            previous_start = {
                var: float(var.X)
                for var in self.partial_master.getVars()
            }
            recourse = self.solve_route_recourse(sol, label=f"partial_iter{it}")

            fixed_value = self._fixed_cost_value({"x": sol.x, "y": sol.y, "z": sol.z, "Gamma": sol.Gamma})
            revenue_value = self._partial_revenue_value(sol)
            cut_type = "optimality"
            if recourse.obj_value is None:
                if recourse.feasibility_cut_expr_data is None:
                    print(f"Route recourse failed. status={recourse.status}")
                    break
                self._add_partial_feasibility_cut(recourse, "partial_feasibility")
                cut_type = "feasibility"
            elif recourse.slack > slack_tol:
                self._add_partial_feasibility_cut(recourse, "partial_feasibility")
                cut_type = "feasibility"
            else:
                design = {"x": sol.x, "y": sol.y, "z": sol.z, "Gamma": sol.Gamma}
                operational_obj = revenue_value - float(recourse.obj_value)
                true_obj = operational_obj - fixed_value
                if self.validate_incumbents:
                    full_sub = self.solve_subproblem(design, label=f"partial_incumbent_{it}")
                    if full_sub.obj_value is not None:
                        operational_obj = float(full_sub.obj_value)
                        true_obj = operational_obj - fixed_value
                if true_obj > lower_bound + 1e-7:
                    lower_bound = true_obj
                    self.best_total_obj = true_obj
                    self.best_operational_obj = operational_obj
                    self.best_design = design
                    if incumbent_floor is not None:
                        incumbent_floor.RHS = lower_bound - incumbent_floor_tol
                self._add_partial_cut(recourse, "partial_benders")

            # Reuse the complete previous master solution.  For an optimality
            # cut, adjusting theta to the exact recourse value makes this a
            # feasible start for the newly cut master in the common case.
            for var, value in previous_start.items():
                var.Start = value
            if recourse.obj_value is not None and recourse.slack <= slack_tol:
                self.partial_vars["theta"].Start = min(
                    float(sol.theta),
                    -float(recourse.obj_value),
                )

            abs_gap = upper_bound - lower_bound
            rel_gap = abs_gap / max(1.0, abs(lower_bound))
            slack_components = recourse.slack_components or {}
            self.partial_history.append(
                {
                    "iter": it,
                    "UB": upper_bound,
                    "LB": lower_bound,
                    "abs_gap": abs_gap,
                    "rel_gap": rel_gap,
                    "master_obj": sol.obj,
                    "theta": sol.theta,
                    "route_obj": None if recourse.obj_value is None else float(recourse.obj_value),
                    "feasibility_obj": recourse.feasibility_obj,
                    "route_slack": recourse.slack,
                    "route_slack_route": slack_components.get("route"),
                    "route_slack_energy_l": slack_components.get("energy_l"),
                    "route_slack_energy_hs": slack_components.get("energy_hs"),
                    "route_slack_fleet": slack_components.get("fleet"),
                    "cut_type": cut_type,
                    "cuts": self.partial_cut_count,
                    "time": time.time() - t0,
                }
            )
            route_text = "NA" if recourse.obj_value is None else f"{recourse.obj_value:.4f}"
            print(
                f"PartialIter {it:03d}: UB={upper_bound:.4f}, LB={lower_bound:.4f}, "
                f"gap={rel_gap:.4%}, route={route_text}, slack={recourse.slack:.4g}, "
                f"cut={cut_type}, cuts={self.partial_cut_count}"
            )

            if recourse.slack <= slack_tol and abs_gap <= tolerance * max(1.0, abs(lower_bound)):
                print("Partial Benders converged.")
                break

        self.history = self.partial_history
        return self.best_design is not None


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
