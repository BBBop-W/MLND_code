import math
import os
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import gurobipy as gp
from gurobipy import GRB

# Support both package execution and direct script execution.
if __package__ in (None, ""):
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)

    from src import config
    from src.data_processor.instance import Instance
    from src.model.benders import (
        RegimeIIData,
        RegimeIIDualObjective,
        RegimeIIDualSubproblem,
    )
else:
    from .. import config
    from ..data_processor.instance import Instance
    from .benders import (
        RegimeIIData,
        RegimeIIDualObjective,
        RegimeIIDualSubproblem,
    )


TOL = 1e-6
RAY_TOL = 1e-9


@dataclass
class CallbackBendersSolution:
    obj: Optional[float]
    master_obj: Optional[float]
    bound: Optional[float]
    gap: Optional[float]
    runtime: float
    wall_time: float
    x: Dict[str, float]
    y: Dict[Tuple[str, str], float]
    z: Dict[Tuple[str, int], float]
    Gamma: Optional[float]
    theta: Optional[float]


class DualCutOracle:
    """Dual-only SOCP oracle used by the callback Benders master."""

    def __init__(self, data: RegimeIIData, use_cache: bool = True):
        self.data = data
        self.use_cache = use_cache
        self.cache: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
        self.solve_count = 0
        self.cache_hits = 0
        self.status_counts: Dict[int, int] = {}
        self.model_build_count = 0
        self.subproblem_time = 0.0
        self.mw_time = 0.0
        self.recession_time = 0.0
        self._dual_params = {"InfUnbdInfo": 1, "NumericFocus": 2}
        self._resolve_profiles = (
            {"InfUnbdInfo": 1, "NumericFocus": 2},
            {"InfUnbdInfo": 1, "ScaleFlag": 2, "NumericFocus": 3},
            {"InfUnbdInfo": 1, "ScaleFlag": 2, "NumericFocus": 3, "Presolve": 0},
        )
        self._profile_defaults = {
            "InfUnbdInfo": 0,
            "NumericFocus": 0,
            "ScaleFlag": -1,
            "Presolve": -1,
        }
        self._dual_context: Optional[Tuple[RegimeIIDualSubproblem, gp.Model, Dict[str, Any]]] = None
        self._recession_context: Optional[Tuple[RegimeIIDualSubproblem, gp.Model, Dict[str, Any]]] = None
        self._pareto_context: Optional[Tuple[RegimeIIDualSubproblem, gp.Model, Dict[str, Any]]] = None
        self._pareto_anchor: list[gp.Constr] = []

    def _new_dual_context(
        self,
        params: Dict[str, Any],
        model_name: str,
    ) -> Tuple[RegimeIIDualSubproblem, gp.Model, Dict[str, Any]]:
        solver = RegimeIIDualSubproblem(self.data, output=False)
        model, dual_vars = solver._build_model()
        model.ModelName = model_name
        for param, value in params.items():
            model.setParam(param, value)
        model.update()
        self.model_build_count += 1
        return solver, model, dual_vars

    def _main_dual_context(self):
        if self._dual_context is None:
            self._dual_context = self._new_dual_context(
                self._dual_params,
                "RegimeII_DualSub_Persistent",
            )
        return self._dual_context

    def _apply_resolve_profile(self, model: gp.Model, params: Dict[str, Any]) -> None:
        for param, default in self._profile_defaults.items():
            model.setParam(param, params.get(param, default))

    def _recession_dual_context(self):
        if self._recession_context is None:
            solver, model, dual_vars = self._new_dual_context(
                {"NumericFocus": 3},
                "RegimeII_DualSub_Recession_Persistent",
            )
            for constr in model.getConstrs():
                constr.RHS = 0.0
            for var in model.getVars():
                if var.LB <= -GRB.INFINITY / 2:
                    var.LB = -1.0
                else:
                    var.LB = max(float(var.LB), -1.0)
                if var.UB >= GRB.INFINITY / 2:
                    var.UB = 1.0
                else:
                    var.UB = min(float(var.UB), 1.0)
            model.update()
            self._recession_context = (solver, model, dual_vars)
        return self._recession_context

    def _pareto_dual_context(self):
        if self._pareto_context is None:
            self._pareto_context = self._new_dual_context(
                {},
                "RegimeII_DualSub_Pareto_Persistent",
            )
        return self._pareto_context

    @staticmethod
    def _extract_attr(container: Any, attr: str) -> Any:
        if hasattr(container, "items"):
            return {k: float(getattr(v, attr)) for k, v in container.items()}
        return float(getattr(container, attr))

    def _collect_ray(self, dual_vars: Dict[str, Any]) -> Dict[str, Any]:
        return {name: self._extract_attr(container, "UnbdRay") for name, container in dual_vars.items()}

    @staticmethod
    def _scale_duals(duals: Dict[str, Any], scale: float) -> Dict[str, Any]:
        scaled: Dict[str, Any] = {}
        for name, item in duals.items():
            if isinstance(item, dict):
                scaled[name] = {key: value * scale for key, value in item.items()}
            else:
                scaled[name] = item * scale
        return scaled

    def _solve_recession_cut(
        self,
        x_ref: Dict[str, float],
        y_ref: Dict[Tuple[str, str], float],
        z_ref: Dict[Tuple[str, int], float],
        gamma_ref: float,
    ) -> Optional[Dict[str, Any]]:
        """Find a feasibility cut from the dual recession cone.

        Gurobi does not reliably expose UnbdRay for conic dual models.  This
        auxiliary SOCP keeps the same recession cone, bounds the ray by a unit
        box for normalization, and minimizes the current master RHS.
        """

        solver, model, dual_vars = self._recession_dual_context()
        model.setObjective(
            RegimeIIDualObjective.expr(self.data, dual_vars, x_ref, y_ref, z_ref, gamma_ref, True),
            GRB.MINIMIZE,
        )
        start = time.perf_counter()
        model.optimize()
        elapsed = time.perf_counter() - start
        self.recession_time += elapsed
        self.solve_count += 1
        self.status_counts[model.Status] = self.status_counts.get(model.Status, 0) + 1
        if model.Status != GRB.OPTIMAL or model.ObjVal >= -RAY_TOL:
            return None

        ray = solver._collect(dual_vars)
        ray_rhs = RegimeIIDualObjective.expr(
            self.data,
            ray,
            x_ref,
            y_ref,
            z_ref,
            gamma_ref,
            False,
        )
        if ray_rhs >= -RAY_TOL:
            return None
        scale = 1.0 / (-ray_rhs)
        ray = self._scale_duals(ray, scale)
        ray_rhs = RegimeIIDualObjective.expr(
            self.data,
            ray,
            x_ref,
            y_ref,
            z_ref,
            gamma_ref,
            False,
        )
        return {
            "kind": "feasibility",
            "duals": ray,
            "rhs_at_hat": ray_rhs,
            "z_star": None,
            "source": "dual_recession_cut",
            "status": GRB.UNBOUNDED,
        }

    def _point_key(
        self,
        x_ref: Dict[str, float],
        y_ref: Dict[Tuple[str, str], float],
        z_ref: Dict[Tuple[str, int], float],
        gamma_ref: float,
        decimals: int,
    ) -> Tuple[Any, ...]:
        return (
            tuple(round(x_ref.get(u, 0.0), decimals) for u in self.data.N_x),
            tuple(round(y_ref.get((i, l), 0.0), decimals) for l, i in self.data.A_LH),
            tuple(round(z_ref.get((i, c), 0.0), decimals) for i in self.data.H for c in self.data.C),
            round(float(gamma_ref), decimals),
        )

    def solve(
        self,
        x_ref: Dict[str, float],
        y_ref: Dict[Tuple[str, str], float],
        z_ref: Dict[Tuple[str, int], float],
        gamma_ref: float,
        *,
        cache_decimals: int = 6,
        accept_feasible: bool = False,
    ) -> Optional[Dict[str, Any]]:
        key = self._point_key(x_ref, y_ref, z_ref, gamma_ref, cache_decimals) + (accept_feasible,)
        if self.use_cache and key in self.cache:
            self.cache_hits += 1
            return self.cache[key]

        solver, model, dual_vars = self._main_dual_context()
        candidate_feasible = None
        for params in self._resolve_profiles:
            self._apply_resolve_profile(model, params)
            model.setObjective(
                RegimeIIDualObjective.expr(self.data, dual_vars, x_ref, y_ref, z_ref, gamma_ref, True),
                GRB.MINIMIZE,
            )
            start = time.perf_counter()
            model.optimize()
            elapsed = time.perf_counter() - start
            self.subproblem_time += elapsed
            self.solve_count += 1
            self.status_counts[model.Status] = self.status_counts.get(model.Status, 0) + 1

            if model.Status == GRB.OPTIMAL:
                duals = solver._collect(dual_vars)
                rhs_at_ref = RegimeIIDualObjective.expr(
                    self.data,
                    duals,
                    x_ref,
                    y_ref,
                    z_ref,
                    gamma_ref,
                    False,
                )
                cut_data = {
                    "duals": duals,
                    "rhs_at_hat": rhs_at_ref,
                    "z_star": model.ObjVal,
                    "source": "dual",
                    "status": model.Status,
                }
                if self.use_cache:
                    self.cache[key] = cut_data
                return cut_data

            if model.Status == GRB.UNBOUNDED:
                try:
                    ray = self._collect_ray(dual_vars)
                    ray_rhs = RegimeIIDualObjective.expr(
                        self.data,
                        ray,
                        x_ref,
                        y_ref,
                        z_ref,
                        gamma_ref,
                        False,
                    )
                except (AttributeError, gp.GurobiError):
                    ray = None
                    ray_rhs = None

                if ray is not None and ray_rhs is not None and ray_rhs < -RAY_TOL:
                    ray = self._scale_duals(ray, 1.0 / (-ray_rhs))
                    ray_rhs = RegimeIIDualObjective.expr(
                        self.data,
                        ray,
                        x_ref,
                        y_ref,
                        z_ref,
                        gamma_ref,
                        False,
                    )
                    cut_data = {
                        "kind": "feasibility",
                        "duals": ray,
                        "rhs_at_hat": ray_rhs,
                        "z_star": None,
                        "source": "dual_unbounded_ray",
                        "status": model.Status,
                    }
                    if self.use_cache:
                        self.cache[key] = cut_data
                    return cut_data

                recession_cut = self._solve_recession_cut(x_ref, y_ref, z_ref, gamma_ref)
                if recession_cut is not None:
                    if self.use_cache:
                        self.cache[key] = recession_cut
                    return recession_cut

            if accept_feasible and model.SolCount > 0:
                duals = solver._collect(dual_vars)
                rhs_at_ref = RegimeIIDualObjective.expr(
                    self.data,
                    duals,
                    x_ref,
                    y_ref,
                    z_ref,
                    gamma_ref,
                    False,
                )
                candidate_feasible = {
                    "duals": duals,
                    "rhs_at_hat": rhs_at_ref,
                    "z_star": model.ObjVal,
                    "source": f"dual_feasible_status_{model.Status}",
                    "status": model.Status,
                }

        if candidate_feasible is not None:
            if self.use_cache:
                self.cache[key] = candidate_feasible
            return candidate_feasible
        return None

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
    ) -> Optional[Dict[str, Any]]:
        solver, model, dual_vars = self._pareto_dual_context()
        if self._pareto_anchor:
            model.remove(self._pareto_anchor)
            self._pareto_anchor = []
            model.update()

        anchor = RegimeIIDualObjective.expr(
            self.data,
            dual_vars,
            x_hat,
            y_hat,
            z_hat,
            gamma_hat,
            True,
        )
        eps = 1e-5 + 1e-5 * max(1.0, abs(float(z_star)))
        self._pareto_anchor = [
            model.addConstr(anchor >= z_star - eps, name="mw_anchor_lb"),
            model.addConstr(anchor <= z_star + eps, name="mw_anchor_ub"),
        ]
        model.setObjective(
            RegimeIIDualObjective.expr(self.data, dual_vars, core_x, core_y, core_z, core_gamma, True),
            GRB.MINIMIZE,
        )
        start = time.perf_counter()
        model.optimize()
        elapsed = time.perf_counter() - start
        self.mw_time += elapsed
        self.solve_count += 1
        self.status_counts[model.Status] = self.status_counts.get(model.Status, 0) + 1
        if model.Status == GRB.OPTIMAL:
            duals = solver._collect(dual_vars)
            return {
                "kind": "optimality",
                "z_star": z_star,
                "duals": duals,
                "rhs_at_hat": RegimeIIDualObjective.expr(
                    self.data,
                    duals,
                    x_hat,
                    y_hat,
                    z_hat,
                    gamma_hat,
                    False,
                ),
                "mw_obj": model.ObjVal,
                "source": "pareto",
            }
        return None


class RegimeIICallbackBenders:
    """Dual-only branch-and-Benders implementation for Regime II."""

    def __init__(
        self,
        instance: Instance,
        *,
        use_valid_inequalities: bool = True,
        use_initial_cuts: bool = True,
        use_pareto: bool = True,
        use_user_cuts: bool = True,
        use_cut_cache: bool = True,
        time_limit: float = 120.0,
        mip_gap: float = 1e-5,
        max_user_cuts: int = 120,
        node_cut_interval: int = 25,
        pareto_gap_trigger: float = 1e-4,
        initial_solution: Optional[Any] = None,
        verbose: bool = True,
    ):
        self.inst = instance
        self.data = RegimeIIData(instance)
        self.use_valid_inequalities = use_valid_inequalities
        self.use_initial_cuts = use_initial_cuts
        self.use_pareto = use_pareto
        self.use_user_cuts = use_user_cuts
        self.use_cut_cache = use_cut_cache
        self.time_limit = time_limit
        self.mip_gap = mip_gap
        self.max_user_cuts = max_user_cuts
        self.node_cut_interval = max(1, node_cut_interval)
        self.pareto_gap_trigger = pareto_gap_trigger
        self.initial_solution = initial_solution
        self.verbose = verbose

        self.model: Optional[gp.Model] = None
        self.x = None
        self.y = None
        self.z = None
        self.Gamma = None
        self.Theta = None
        self.q = None

        self.oracle = DualCutOracle(self.data, use_cache=use_cut_cache)
        self.core_x, self.core_y, self.core_z, self.core_gamma = self._initial_core_point()

        self.lazy_cuts = 0
        self.user_cuts = 0
        self.initial_cuts = 0
        self.final_cuts = 0
        self.callback_points = 0
        self.node_callbacks = 0
        self.uncertified_incumbents = 0
        self.last_uncertified_point: Optional[Dict[str, Any]] = None
        self.best_obj = -math.inf
        self.best_solution: Optional[CallbackBendersSolution] = None
        self.mip_start_loaded = False

    def _initial_core_point(self):
        data = self.data
        core_x = {u: 1.0 for u in data.N_x}
        core_y: Dict[Tuple[str, str], float] = {}
        for i in data.H:
            bases = [l for l, h in data.A_LH if h == i]
            denom = max(1, len(bases))
            for l in bases:
                core_y[i, l] = 1.0 / denom
        for l, i in data.A_LH:
            core_y.setdefault((i, l), 0.0)
        core_z = {(i, c): 1.0 / max(1, len(data.C)) for i in data.H for c in data.C}
        return core_x, core_y, core_z, data.core_gamma

    def _build_master(self) -> None:
        data = self.data
        model = gp.Model("RegimeII_CallbackBenders_Dual")
        model.setParam("OutputFlag", 1 if self.verbose else 0)
        model.setParam("TimeLimit", self.time_limit)
        model.setParam("MIPGap", self.mip_gap)
        model.setParam("LazyConstraints", 1)
        model.setParam("PreCrush", 1)
        model.setParam("FeasibilityTol", 1e-8)
        model.setParam("OptimalityTol", 1e-8)

        self.x = model.addVars(data.N_x, vtype=GRB.BINARY, name="x")
        self.y = model.addVars([(i, l) for l, i in data.A_LH], vtype=GRB.BINARY, name="y")
        self.z = model.addVars(data.H, data.C, vtype=GRB.BINARY, name="z")
        self.Gamma = model.addVar(lb=0.0, vtype=GRB.INTEGER, name="Gamma")
        self.Theta = model.addVar(lb=0.0, ub=max(1.0, data.total_revenue_ub), name="Theta")

        fixed = gp.quicksum(float(self.inst.f.get(l, config.COST_BASE)) * self.x[l] for l in data.L)
        fixed += gp.quicksum(float(getattr(config, "COST_HUB_FIXED", 0.0)) * self.x[i] for i in data.H)
        fixed += gp.quicksum(float(self.inst.f.get(k, config.COST_SWAPPING)) * self.x[k] for k in data.E)
        fixed += gp.quicksum(float(config.COST_HUB[c]) * self.z[i, c] for i in data.H for c in data.C)
        model.setObjective(self.Theta - fixed - data.gamma_cost * self.Gamma, GRB.MAXIMIZE)

        for i in data.H:
            bases = [l for l, h in data.A_LH if h == i]
            if bases:
                model.addConstr(gp.quicksum(self.y[i, l] for l in bases) == self.x[i], name=f"assign_{i}")
            else:
                model.addConstr(self.x[i] == 0, name=f"no_assign_{i}")
            model.addConstr(gp.quicksum(self.z[i, c] for c in data.C) == self.x[i], name=f"capacity_{i}")

        for l, i in data.A_LH:
            model.addConstr(self.y[i, l] <= self.x[l], name=f"assign_active_{i}_{l}")

        if self.use_valid_inequalities:
            self._add_valid_inequalities(model)

        self.model = model
        if self.use_initial_cuts:
            self._add_initial_cuts()
        if self.initial_solution is not None:
            self._load_mip_start(self.initial_solution)

    def _load_mip_start(self, solution: Any) -> None:
        if self.model is None:
            return

        x_start = getattr(solution, "x", None)
        y_start = getattr(solution, "y", None)
        z_start = getattr(solution, "z", None)
        gamma_start = getattr(solution, "gamma", getattr(solution, "Gamma", None))
        theta_start = getattr(solution, "theta", None)
        if x_start is None or y_start is None or z_start is None:
            return

        for u in self.data.N_x:
            self.x[u].Start = float(x_start.get(u, 0.0))
        for l, i in self.data.A_LH:
            self.y[i, l].Start = float(y_start.get((i, l), 0.0))
        for i in self.data.H:
            for c in self.data.C:
                self.z[i, c].Start = float(z_start.get((i, c), 0.0))
        if gamma_start is not None:
            self.Gamma.Start = max(0.0, float(gamma_start))
        if theta_start is not None:
            self.Theta.Start = max(0.0, min(float(theta_start), max(1.0, self.data.total_revenue_ub)))
        self.mip_start_loaded = True

    def _add_valid_inequalities(self, model: gp.Model) -> None:
        data = self.data

        for i in data.H:
            model.addConstr(
                gp.quicksum(data.mu_levels[c] * self.z[i, c] for c in data.C) >= self.x[i] / data.tau_H,
                name=f"master_hub_wait_feas_{i}",
            )

        if data.E:
            model.addConstr(
                self.Theta <= data.total_revenue_ub * gp.quicksum(self.x[k] for k in data.E),
                name="theta_requires_swap_open",
            )

        serviceable_hh = [(i, j) for i, j in data.A_HH if data.route_options.get((i, j))]
        if serviceable_hh:
            self.q = model.addVars(serviceable_hh, lb=0.0, ub=1.0, name="q_hh")
            for i, j in serviceable_hh:
                model.addConstr(self.q[i, j] <= self.x[i], name=f"q_origin_{i}_{j}")
                model.addConstr(self.q[i, j] <= self.x[j], name=f"q_dest_{i}_{j}")
                model.addConstr(self.q[i, j] >= self.x[i] + self.x[j] - 1.0, name=f"q_and_{i}_{j}")

            model.addConstr(
                self.Theta
                <= data.revenue
                * gp.quicksum(float(self.inst.h.get((i, j), 0.0)) * self.q[i, j] for i, j in serviceable_hh),
                name="theta_open_demand_bound",
            )

            base_service_keys = [
                (l, i, j)
                for l, i, j in data.alpha_keys
                if (i, j) in data.A_HH_set and data.p_chain(i) < 1.0 - TOL
            ]
            if base_service_keys:
                q_lij = model.addVars(base_service_keys, lb=0.0, ub=1.0, name="q_lij")
                service_time_expr = gp.LinExpr()
                for l, i, j in base_service_keys:
                    model.addConstr(q_lij[l, i, j] <= self.y[i, l], name=f"q_lij_assign_{l}_{i}_{j}")
                    model.addConstr(q_lij[l, i, j] <= self.x[j], name=f"q_lij_dest_{l}_{i}_{j}")
                    usable_swaps = [k for k in data.route_options.get((l, i), []) if k in data.E]
                    if usable_swaps:
                        model.addConstr(
                            q_lij[l, i, j] <= gp.quicksum(self.x[k] for k in usable_swaps),
                            name=f"q_lij_swap_{l}_{i}_{j}",
                        )
                    else:
                        model.addConstr(q_lij[l, i, j] <= 0.0, name=f"q_lij_no_swap_{l}_{i}_{j}")

                    lh_time = min(
                        (data.t[l, i, k] for k in data.route_options.get((l, i), []) if k in data.E),
                        default=math.inf,
                    )
                    hh_time = min((data.t[i, j, k] for k in data.route_options.get((i, j), [])), default=math.inf)
                    if math.isfinite(lh_time) and math.isfinite(hh_time):
                        service_time_expr += (
                            (1.0 - data.p_chain(i))
                            * float(self.inst.h.get((i, j), 0.0))
                            * (lh_time + hh_time)
                            * q_lij[l, i, j]
                        )

                chain_bound = gp.quicksum(
                    data.p_chain(i) * float(self.inst.h.get((i, j), 0.0)) * self.q[i, j]
                    for i, j in serviceable_hh
                    if data.p_chain(i) > TOL
                )
                base_bound = gp.quicksum(
                    (1.0 - data.p_chain(i)) * float(self.inst.h.get((i, j), 0.0)) * q_lij[l, i, j]
                    for l, i, j in base_service_keys
                )
                model.addConstr(
                    self.Theta <= data.revenue * (base_bound + chain_bound),
                    name="theta_assignment_swap_demand_bound",
                )
                model.addConstr(service_time_expr <= self.Gamma, name="fleet_assignment_time_bound")

            min_loaded_time = min(
                (data.t[i, j, k] for i, j in serviceable_hh for k in data.route_options.get((i, j), [])),
                default=0.0,
            )
            if min_loaded_time > TOL:
                model.addConstr(
                    self.Theta <= (data.revenue / min_loaded_time) * self.Gamma,
                    name="theta_fleet_time_bound",
                )

            min_service_time = math.inf
            min_route_time = {
                (u, v): min((data.t[u, v, k] for k in data.route_options.get((u, v), [])), default=math.inf)
                for u, v in data.A_F
            }
            for i, j in serviceable_hh:
                hh_time = min_route_time.get((i, j), math.inf)
                if not math.isfinite(hh_time):
                    continue
                if data.p_chain(i) > TOL:
                    min_service_time = min(min_service_time, hh_time)
                if 1.0 - data.p_chain(i) > TOL:
                    for l, h in data.A_LH:
                        if h != i:
                            continue
                        lh_time = min_route_time.get((l, i), math.inf)
                        if math.isfinite(lh_time):
                            min_service_time = min(min_service_time, lh_time + hh_time)
            if math.isfinite(min_service_time) and min_service_time > TOL:
                model.addConstr(
                    self.Theta <= (data.revenue / min_service_time) * self.Gamma,
                    name="theta_service_cycle_time_bound",
                )

        model.addConstr(
            self.Theta
            <= 0.5
            * data.revenue
            * gp.quicksum(
                gp.quicksum(data.mu_levels[c] * self.z[i, c] for c in data.C) - self.x[i] / data.tau_H
                for i in data.H
            ),
            name="theta_hub_capacity_bound",
        )

    def _reference_points(self):
        data = self.data
        points = []

        closed_x = {u: 0.0 for u in data.N_x}
        closed_y = {(i, l): 0.0 for l, i in data.A_LH}
        closed_z = {(i, c): 0.0 for i in data.H for c in data.C}
        points.append((closed_x, closed_y, closed_z, 0.0))

        def repaired_point(open_hubs, gamma_value):
            x_ref = {u: 0.0 for u in data.N_x}
            y_ref = {(i, l): 0.0 for l, i in data.A_LH}
            z_ref = {(i, c): 0.0 for i in data.H for c in data.C}
            cap = max(data.C, key=lambda c: data.mu_levels[c]) if data.C else 0
            for k in data.E:
                x_ref[k] = 1.0
            for i in open_hubs:
                bases = [l for l, h in data.A_LH if h == i]
                if not bases:
                    continue
                best_l = min(bases, key=lambda l: self.inst.get_dist(l, i))
                x_ref[i] = 1.0
                x_ref[best_l] = 1.0
                y_ref[i, best_l] = 1.0
                z_ref[i, cap] = 1.0
            return x_ref, y_ref, z_ref, float(gamma_value)

        all_hubs = list(data.H)
        points.append(repaired_point(all_hubs, max(1.0, math.ceil(data.core_gamma))))

        hub_score = {i: 0.0 for i in data.H}
        for (i, j), demand in self.inst.h.items():
            hub_score[i] = hub_score.get(i, 0.0) + float(demand)
            hub_score[j] = hub_score.get(j, 0.0) + float(demand)
        top_hubs = [
            h
            for h, _ in sorted(hub_score.items(), key=lambda item: item[1], reverse=True)[: max(1, min(4, len(data.H)))]
        ]
        points.append(repaired_point(top_hubs, max(1.0, math.ceil(data.core_gamma / 2.0))))
        return points

    def _add_initial_cuts(self) -> None:
        assert self.model is not None
        for idx, (x_ref, y_ref, z_ref, gamma_ref) in enumerate(self._reference_points(), start=1):
            cut_data = self.oracle.solve(x_ref, y_ref, z_ref, gamma_ref)
            if cut_data is None:
                continue
            self._add_master_cut(cut_data, f"initial_benders_{idx}_{cut_data['source']}")
            self.initial_cuts += 1

            if cut_data.get("kind") == "feasibility":
                continue
            if not self.use_pareto or abs(cut_data["rhs_at_hat"]) <= TOL:
                continue
            pareto = self.oracle.solve_pareto(
                x_ref,
                y_ref,
                z_ref,
                gamma_ref,
                self.core_x,
                self.core_y,
                self.core_z,
                self.core_gamma,
                cut_data["rhs_at_hat"],
            )
            if pareto is not None:
                self._add_master_cut(pareto, f"initial_pareto_{idx}")
                self.initial_cuts += 1

    def _add_master_cut(self, cut_data: Dict[str, Any], name: str) -> None:
        assert self.model is not None
        rhs = RegimeIIDualObjective.expr(self.data, cut_data["duals"], self.x, self.y, self.z, self.Gamma, True)
        if cut_data.get("kind") == "feasibility":
            self.model.addConstr(rhs >= 0.0, name=name)
        else:
            self.model.addConstr(self.Theta <= rhs, name=name)

    def _callback_values(self, model: gp.Model, getter) -> Tuple[Dict[str, float], Dict[Tuple[str, str], float], Dict[Tuple[str, int], float], float, float]:
        data = self.data
        x_ref = {u: float(getter(self.x[u])) for u in data.N_x}
        y_ref = {(i, l): float(getter(self.y[i, l])) for l, i in data.A_LH}
        z_ref = {(i, c): float(getter(self.z[i, c])) for i in data.H for c in data.C}
        gamma_ref = float(getter(self.Gamma))
        theta_ref = float(getter(self.Theta))
        return x_ref, y_ref, z_ref, gamma_ref, theta_ref

    def _fixed_cost(self, x_ref: Dict[str, float], z_ref: Dict[Tuple[str, int], float]) -> float:
        return self.data.fixed_cost_value(x_ref, z_ref)

    def _select_cut(
        self,
        x_ref: Dict[str, float],
        y_ref: Dict[Tuple[str, str], float],
        z_ref: Dict[Tuple[str, int], float],
        gamma_ref: float,
        theta_ref: float,
        *,
        allow_pareto: bool,
        cache_decimals: int,
    ) -> Optional[Dict[str, Any]]:
        standard = self.oracle.solve(x_ref, y_ref, z_ref, gamma_ref, cache_decimals=cache_decimals)
        if standard is None:
            feasible = self.oracle.solve(
                x_ref,
                y_ref,
                z_ref,
                gamma_ref,
                cache_decimals=cache_decimals,
                accept_feasible=True,
            )
            if feasible is not None and theta_ref > float(feasible["rhs_at_hat"]) + TOL:
                return feasible
            return None

        if standard.get("kind") == "feasibility":
            if float(standard["rhs_at_hat"]) < -TOL:
                return standard
            return None

        if (
            not allow_pareto
            or not self.use_pareto
            or abs(standard["rhs_at_hat"]) <= TOL
            or theta_ref <= standard["rhs_at_hat"] + self.pareto_gap_trigger
        ):
            return standard

        pareto = self.oracle.solve_pareto(
            x_ref,
            y_ref,
            z_ref,
            gamma_ref,
            self.core_x,
            self.core_y,
            self.core_z,
            self.core_gamma,
            standard["rhs_at_hat"],
        )
        if pareto is not None and pareto["rhs_at_hat"] <= theta_ref - TOL:
            return pareto
        return standard

    def _lazy_callback(self, model: gp.Model, where: int) -> None:
        if where == GRB.Callback.MIPSOL:
            self.callback_points += 1
            x_ref, y_ref, z_ref, gamma_ref, theta_ref = self._callback_values(model, model.cbGetSolution)
            cut_data = self._select_cut(
                x_ref,
                y_ref,
                z_ref,
                gamma_ref,
                theta_ref,
                allow_pareto=True,
                cache_decimals=6,
            )
            if cut_data is None:
                self.uncertified_incumbents += 1
                self.last_uncertified_point = {
                    "x": x_ref,
                    "y": y_ref,
                    "z": z_ref,
                    "Gamma": gamma_ref,
                    "theta": theta_ref,
                }
                model.terminate()
                return

            rhs_at_ref = float(cut_data["rhs_at_hat"])
            rhs_expr = RegimeIIDualObjective.expr(self.data, cut_data["duals"], self.x, self.y, self.z, self.Gamma, True)
            if cut_data.get("kind") == "feasibility":
                if rhs_at_ref < -TOL:
                    model.cbLazy(rhs_expr >= 0.0)
                    self.lazy_cuts += 1
                return

            if theta_ref > rhs_at_ref + TOL:
                model.cbLazy(self.Theta <= rhs_expr)
                self.lazy_cuts += 1
                return

            true_obj = rhs_at_ref - self._fixed_cost(x_ref, z_ref) - self.data.gamma_cost * gamma_ref
            if true_obj > self.best_obj:
                self.best_obj = true_obj
                self.best_solution = CallbackBendersSolution(
                    obj=true_obj,
                    master_obj=theta_ref - self._fixed_cost(x_ref, z_ref) - self.data.gamma_cost * gamma_ref,
                    bound=None,
                    gap=None,
                    runtime=0.0,
                    wall_time=0.0,
                    x=x_ref,
                    y=y_ref,
                    z=z_ref,
                    Gamma=gamma_ref,
                    theta=theta_ref,
                )

        elif self.use_user_cuts and where == GRB.Callback.MIPNODE and self.user_cuts < self.max_user_cuts:
            if int(model.cbGet(GRB.Callback.MIPNODE_STATUS)) != GRB.OPTIMAL:
                return
            self.node_callbacks += 1
            if self.node_callbacks % self.node_cut_interval != 0:
                return
            x_ref, y_ref, z_ref, gamma_ref, theta_ref = self._callback_values(model, model.cbGetNodeRel)
            cut_data = self._select_cut(
                x_ref,
                y_ref,
                z_ref,
                gamma_ref,
                theta_ref,
                allow_pareto=False,
                cache_decimals=4,
            )
            if cut_data is None:
                return
            rhs_expr = RegimeIIDualObjective.expr(self.data, cut_data["duals"], self.x, self.y, self.z, self.Gamma, True)
            if cut_data.get("kind") == "feasibility":
                if float(cut_data["rhs_at_hat"]) >= -TOL:
                    return
                model.cbCut(rhs_expr >= 0.0)
                self.user_cuts += 1
                return
            if theta_ref <= float(cut_data["rhs_at_hat"]) + TOL:
                return
            model.cbCut(self.Theta <= rhs_expr)
            self.user_cuts += 1

    def solve(self) -> CallbackBendersSolution:
        self._build_master()
        assert self.model is not None
        start = time.perf_counter()
        while True:
            elapsed = time.perf_counter() - start
            remaining = max(1.0, self.time_limit - elapsed)
            self.model.setParam("TimeLimit", remaining)
            self.model.optimize(self._lazy_callback)

            if self.model.SolCount == 0:
                break
            cut_data = self._final_certification_cut()
            if cut_data is None:
                break
            self._add_master_cut(cut_data, f"final_cert_{self.final_cuts + 1}_{cut_data['source']}")
            self.final_cuts += 1
            if self.final_cuts >= 50 or time.perf_counter() - start >= self.time_limit:
                break

        wall = time.perf_counter() - start

        solution = self._extract_solution(wall)
        self.best_solution = solution if solution.obj is not None and solution.obj >= self.best_obj - TOL else self.best_solution
        if self.verbose:
            print(
                "[Callback-Benders-RII] "
                f"obj={solution.obj} master={solution.master_obj} bound={solution.bound} "
                f"lazy={self.lazy_cuts} user={self.user_cuts} initial={self.initial_cuts} final={self.final_cuts} "
                f"sub_solves={self.oracle.solve_count} sub_models={self.oracle.model_build_count} "
                f"sub_time={self.oracle.subproblem_time:.4f}s mw_time={self.oracle.mw_time:.4f}s "
                f"ray_time={self.oracle.recession_time:.4f}s "
                f"cache_hits={self.oracle.cache_hits} "
                f"start={self.mip_start_loaded} uncertified={self.uncertified_incumbents}"
            )
        return solution

    def _final_certification_cut(self) -> Optional[Dict[str, Any]]:
        assert self.model is not None
        x_ref = {u: float(self.x[u].X) for u in self.data.N_x}
        y_ref = {(i, l): float(self.y[i, l].X) for l, i in self.data.A_LH}
        z_ref = {(i, c): float(self.z[i, c].X) for i in self.data.H for c in self.data.C}
        gamma_ref = float(self.Gamma.X)
        theta_ref = float(self.Theta.X)
        cut_data = self._select_cut(
            x_ref,
            y_ref,
            z_ref,
            gamma_ref,
            theta_ref,
            allow_pareto=True,
            cache_decimals=6,
        )
        if cut_data is None:
            return None
        if cut_data.get("kind") == "feasibility":
            return cut_data if float(cut_data["rhs_at_hat"]) < -TOL else None
        return cut_data if theta_ref > float(cut_data["rhs_at_hat"]) + TOL else None

    def _extract_solution(self, wall: float) -> CallbackBendersSolution:
        assert self.model is not None
        if self.model.SolCount == 0:
            return CallbackBendersSolution(
                obj=None,
                master_obj=None,
                bound=None,
                gap=None,
                runtime=float(self.model.Runtime),
                wall_time=wall,
                x={},
                y={},
                z={},
                Gamma=None,
                theta=None,
            )

        x_ref = {u: float(self.x[u].X) for u in self.data.N_x}
        y_ref = {(i, l): float(self.y[i, l].X) for l, i in self.data.A_LH}
        z_ref = {(i, c): float(self.z[i, c].X) for i in self.data.H for c in self.data.C}
        gamma_ref = float(self.Gamma.X)
        theta_ref = float(self.Theta.X)
        master_obj = float(self.model.ObjVal)

        cut_data = self.oracle.solve(x_ref, y_ref, z_ref, gamma_ref)
        obj = None
        if cut_data is not None and cut_data.get("kind") != "feasibility":
            obj = float(cut_data["rhs_at_hat"]) - self._fixed_cost(x_ref, z_ref) - self.data.gamma_cost * gamma_ref

        return CallbackBendersSolution(
            obj=obj,
            master_obj=master_obj,
            bound=float(self.model.ObjBound) if self.model.Status in {GRB.OPTIMAL, GRB.TIME_LIMIT} else None,
            gap=float(self.model.MIPGap) if self.model.SolCount else None,
            runtime=float(self.model.Runtime),
            wall_time=wall,
            x=x_ref,
            y=y_ref,
            z=z_ref,
            Gamma=gamma_ref,
            theta=theta_ref,
        )


def run_benders_regime_ii_callback(
    instance: Instance,
    *,
    time_limit: float = 120.0,
    mip_gap: float = 1e-5,
    verbose: bool = True,
    **kwargs,
) -> RegimeIICallbackBenders:
    solver = RegimeIICallbackBenders(
        instance,
        time_limit=time_limit,
        mip_gap=mip_gap,
        verbose=verbose,
        **kwargs,
    )
    solver.solve()
    return solver

