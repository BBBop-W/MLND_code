"""Branch-and-Benders callback variant for multi-period TS-style PBD.

The retained periods are kept as full second-stage blocks in the master, as in
``ts_partial_benders``.  The decomposed periods are separated by lazy Benders
cuts inside one Gurobi branch-and-cut search.  This is an experimental
acceleration path inspired by the branch-and-Benders cut strategy in the
Transportation Science crowdsourced-capacity paper.
"""

from __future__ import annotations

import math
import os
import sys
import time
from typing import Any, Dict, Optional, Sequence, Tuple

import gurobipy as gp
from gurobipy import GRB

if __package__ in (None, ""):
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)

from src.model.benders import RegimeIIDualObjective, RegimeIIPrimalSubproblem
from src.model.benders_bdd import RegimeIIBDDStrengthener, bdd_rhs_expr
from src.model.benders_callback import DualCutOracle
from src.multi_period.gurobi_model import (
    MultiPeriodInstance,
    _as_period_ids,
    load_multi_period_instance,
)
from src.multi_period.benders import MultiPeriodMasterSolution
from src.multi_period.ts_partial_benders import (
    TOL,
    MultiPeriodScenarioRetentionBenders,
    _period_total_demand,
)


class MultiPeriodScenarioRetentionCallbackBenders(MultiPeriodScenarioRetentionBenders):
    """Scenario-retention PBD solved with lazy/Benders-dual cuts in one MIP tree."""

    def __init__(
        self,
        mp_instance: MultiPeriodInstance,
        *,
        retained_periods: Sequence[str] | None = None,
        retained_count: int = 1,
        retention_rule: str = "largest_demand",
        use_valid_inequalities: bool = True,
        use_initial_cuts: bool = True,
        use_pareto: bool = False,
        use_bdd_cuts: bool = False,
        use_user_cuts: bool | None = None,
        use_cut_cache: bool = True,
        use_bdd_strengthening: bool = True,
        selective_fraction: float = 1.0,
        selective_min: int = 1,
        time_limit: float = 60.0,
        mip_gap: float = 1e-4,
        gap_tol: float = 1e-3,
        max_user_cuts: int = 80,
        node_cut_interval: int = 25,
        pareto_gap_trigger: float = 1e-4,
        initial_solution: Optional[Any] = None,
        initial_obj: Optional[float] = None,
        verbose: bool = True,
    ):
        super().__init__(
            mp_instance,
            retained_periods=retained_periods,
            retained_count=retained_count,
            retention_rule=retention_rule,
            subproblem="dual",
            use_pareto=use_pareto,
            use_valid_inequalities=use_valid_inequalities,
            use_integer_nogood=False,
            max_iter=1,
            gap_tol=gap_tol,
            master_time_limit=time_limit,
            master_mip_gap=mip_gap,
            verbose=verbose,
        )
        self.total_time_limit = float(time_limit)
        self.use_initial_cuts = bool(use_initial_cuts)
        if use_user_cuts is not None:
            use_bdd_cuts = use_user_cuts
        self.use_bdd_cuts = bool(use_bdd_cuts)
        self.use_user_cuts = self.use_bdd_cuts  # legacy statistic/argument name
        self.use_cut_cache = bool(use_cut_cache)
        self.use_bdd_strengthening = bool(use_bdd_strengthening)
        self.selective_fraction = max(0.0, min(1.0, float(selective_fraction)))
        self.selective_min = max(1, int(selective_min))
        self.max_user_cuts = max(0, int(max_user_cuts))
        self.node_cut_interval = max(1, int(node_cut_interval))
        self.pareto_gap_trigger = float(pareto_gap_trigger)
        self.initial_solution = initial_solution
        self.initial_obj = initial_obj

        self.oracles: Dict[str, DualCutOracle] = {}
        self.bdd_strengtheners: Dict[str, RegimeIIBDDStrengthener] = {}
        self.scenario_scores = {
            period: max(1.0, _period_total_demand(mp_instance, period))
            for period in self.decomposed_periods
        }
        self.scenario_score_counts = {period: 0 for period in self.decomposed_periods}

        self.lazy_cuts = 0
        self.bdd_cuts = 0
        self.user_cuts = 0
        self.initial_cuts = 0
        self.final_cuts = 0
        self.callback_points = 0
        self.node_callbacks = 0
        self.uncertified_incumbents = 0
        self.last_uncertified_point: Optional[Dict[str, Any]] = None
        self.candidate_evals = 0
        self.best_period_objs: Dict[str, float] = {}
        self.mip_start_loaded = False
        self._start_time: Optional[float] = None

    def _build_master(self) -> None:
        super()._build_master()
        assert self.model is not None
        self.model.setParam("TimeLimit", self.total_time_limit)
        self.model.setParam("MIPGap", self.master_mip_gap)
        self.model.setParam("Threads", 1)
        self.model.setParam("LazyConstraints", 1)
        self.model.setParam("PreCrush", 1)
        self.model.setParam("FeasibilityTol", 1e-8)
        self.model.setParam("OptimalityTol", 1e-8)
        self.model.update()
        self.oracles = {
            period: DualCutOracle(self.data_by_period[period], use_cache=self.use_cut_cache)
            for period in self.decomposed_periods
        }
        gamma_ub = float(self.Gamma.UB) if self.Gamma.UB < GRB.INFINITY / 2 else 1e6
        self.bdd_strengtheners = {
            period: RegimeIIBDDStrengthener(self.data_by_period[period], gamma_ub=gamma_ub, output=False)
            for period in self.decomposed_periods
        }
        if self.initial_solution is not None:
            self._load_mip_start(self.initial_solution)
        if self.initial_obj is not None:
            self.best_obj = max(self.best_obj, float(self.initial_obj))

    def _load_mip_start(self, solution: Any) -> None:
        if self.model is None:
            return

        x_start = getattr(solution, "x", None)
        y_start = getattr(solution, "y", None)
        z_start = getattr(solution, "z", None)
        gamma_start = getattr(solution, "Gamma", getattr(solution, "gamma", None))
        if x_start is None or y_start is None or z_start is None:
            return

        for u in self.data0.N_x:
            self.x[u].Start = float(x_start.get(u, 0.0))
        for l, i in self.data0.A_LH:
            self.y[i, l].Start = float(y_start.get((i, l), 0.0))
        for i in self.data0.H:
            for c in self.data0.C:
                self.z[i, c].Start = float(z_start.get((i, c), 0.0))
        if gamma_start is not None:
            self.Gamma.Start = max(0.0, float(gamma_start))
        for period in self.decomposed_periods:
            self.Theta[period].Start = 0.0
        self._load_retained_zero_flow_start(x_start, z_start)
        self.mip_start_loaded = True

    def _load_retained_zero_flow_start(self, x_start: Dict[str, float], z_start: Dict[Tuple[str, int], float]) -> None:
        if not self.retained_vars:
            return
        data = self.data0

        for key in ["rho_l", "rho_s", "alpha", "beta", "psi", "varphi", "lambda", "xi"]:
            container = self.retained_vars.get(key)
            if hasattr(container, "values"):
                for var in container.values():
                    var.Start = 0.0

        q_v1 = self.retained_vars.get("q_v1")
        q_v2 = self.retained_vars.get("q_v2")
        q_t = self.retained_vars.get("q_t")
        if q_v1 is not None and q_v2 is not None and q_t is not None:
            for period in self.retained_periods:
                for q in data.L + data.S:
                    q_v1[period, q].Start = 2.0
                    q_v2[period, q].Start = 0.0
                    q_t[period, q].Start = 2.0

        h_v1 = self.retained_vars.get("h_v1")
        h_v2 = self.retained_vars.get("h_v2")
        h_t = self.retained_vars.get("h_t")
        if h_v1 is not None and h_v2 is not None and h_t is not None:
            for period in self.retained_periods:
                for i in data.H:
                    mu_i = sum(data.mu_levels[c] * float(z_start.get((i, c), 0.0)) for c in data.C)
                    h_v1[period, i].Start = 2.0 * math.sqrt(max(0.0, mu_i))
                    h_v2[period, i].Start = 1.0 - mu_i
                    h_t[period, i].Start = 1.0 + mu_i

    def _reference_points(self):
        data = self.data0
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
                best_l = min(bases, key=lambda l: self.base.get_dist(l, i))
                x_ref[i] = 1.0
                x_ref[best_l] = 1.0
                y_ref[i, best_l] = 1.0
                z_ref[i, cap] = 1.0
            return x_ref, y_ref, z_ref, float(gamma_value)

        all_hubs = list(data.H)
        points.append(repaired_point(all_hubs, max(1.0, math.ceil(data.core_gamma))))

        hub_score = {i: 0.0 for i in data.H}
        for period in self.periods:
            for (i, j), demand in self.mp.demand.get(period, {}).items():
                hub_score[i] = hub_score.get(i, 0.0) + float(demand)
                hub_score[j] = hub_score.get(j, 0.0) + float(demand)
        top_hubs = [
            h
            for h, _ in sorted(
                hub_score.items(),
                key=lambda item: item[1],
                reverse=True,
            )[: max(1, min(4, len(data.H)))]
        ]
        points.append(repaired_point(top_hubs, max(1.0, math.ceil(data.core_gamma / 2.0))))
        return points

    def _cut_rhs_expr(self, period: str, cut_data: Dict[str, Any]) -> gp.LinExpr:
        if "bdd_coeffs" in cut_data:
            return bdd_rhs_expr(cut_data["bdd_coeffs"], self.x, self.y, self.z, self.Gamma, True)
        data = self.data_by_period[period]
        return RegimeIIDualObjective.expr(data, cut_data["duals"], self.x, self.y, self.z, self.Gamma, True)

    def _add_static_cut(self, period: str, cut_data: Dict[str, Any], name: str) -> None:
        assert self.model is not None
        rhs_expr = self._cut_rhs_expr(period, cut_data)
        if cut_data.get("kind") == "feasibility":
            self.model.addConstr(rhs_expr >= 0.0, name=name)
        else:
            self.model.addConstr(self.Theta[period] <= rhs_expr, name=name)

    def _add_initial_cuts(self) -> None:
        for idx, (x_ref, y_ref, z_ref, gamma_ref) in enumerate(self._reference_points(), start=1):
            for period in self.decomposed_periods:
                cut_data = self.oracles[period].solve(x_ref, y_ref, z_ref, gamma_ref)
                if cut_data is None:
                    continue
                self._add_static_cut(period, cut_data, f"mp_ts_cb_initial_{idx}_{period}_{cut_data['source']}")
                self.initial_cuts += 1

    def _callback_values(
        self,
        getter,
    ) -> Tuple[
        Dict[str, float],
        Dict[Tuple[str, str], float],
        Dict[Tuple[str, int], float],
        float,
        Dict[str, float],
    ]:
        data = self.data0
        x_ref = {u: float(getter(self.x[u])) for u in data.N_x}
        y_ref = {(i, l): float(getter(self.y[i, l])) for l, i in data.A_LH}
        z_ref = {(i, c): float(getter(self.z[i, c])) for i in data.H for c in data.C}
        gamma_ref = float(getter(self.Gamma))
        theta_ref = {period: float(getter(self.Theta[period])) for period in self.decomposed_periods}
        return x_ref, y_ref, z_ref, gamma_ref, theta_ref

    def _ordered_decomposed_periods(self) -> list[str]:
        position = {period: idx for idx, period in enumerate(self.decomposed_periods)}
        return sorted(
            self.decomposed_periods,
            key=lambda p: (-self.scenario_scores.get(p, 0.0), position[p]),
        )

    def _initial_selective_count(self) -> int:
        n_periods = len(self.decomposed_periods)
        if n_periods <= 1 or self.selective_fraction >= 1.0 - TOL:
            return n_periods
        return max(self.selective_min, min(n_periods, math.ceil(self.selective_fraction * n_periods)))

    def _update_score(self, period: str, violation: float) -> None:
        count = self.scenario_score_counts.get(period, 0) + 1
        previous = self.scenario_scores.get(period, 0.0)
        self.scenario_scores[period] = previous + (max(0.0, violation) - previous) / count
        self.scenario_score_counts[period] = count

    def _select_cut(
        self,
        period: str,
        x_ref: Dict[str, float],
        y_ref: Dict[Tuple[str, str], float],
        z_ref: Dict[Tuple[str, int], float],
        gamma_ref: float,
        theta_ref: float,
        *,
        allow_pareto: bool,
        cache_decimals: int,
    ) -> Optional[Dict[str, Any]]:
        cut_data = self.oracles[period].solve(
            x_ref,
            y_ref,
            z_ref,
            gamma_ref,
            cache_decimals=cache_decimals,
        )
        if cut_data is None:
            return None
        if cut_data.get("kind") == "feasibility":
            self._update_score(period, -float(cut_data["rhs_at_hat"]))
            return cut_data

        rhs_at_ref = float(cut_data["rhs_at_hat"])
        violation = theta_ref - rhs_at_ref
        self._update_score(period, violation)
        if (
            not allow_pareto
            or not self.use_pareto
            or abs(rhs_at_ref) <= TOL
            or violation <= self.pareto_gap_trigger
        ):
            return cut_data

        pareto = self.oracles[period].solve_pareto(
            x_ref,
            y_ref,
            z_ref,
            gamma_ref,
            self.core_x,
            self.core_y,
            self.core_z,
            self.core_gamma,
            rhs_at_ref,
        )
        if pareto is not None and float(pareto["rhs_at_hat"]) <= theta_ref - TOL:
            return pareto
        return cut_data

    def _violated_period_cut(
        self,
        period: str,
        x_ref: Dict[str, float],
        y_ref: Dict[Tuple[str, str], float],
        z_ref: Dict[Tuple[str, int], float],
        gamma_ref: float,
        theta_ref: float,
        *,
        allow_pareto: bool,
        cache_decimals: int,
    ) -> Optional[Dict[str, Any]]:
        cut_data = self._select_cut(
            period,
            x_ref,
            y_ref,
            z_ref,
            gamma_ref,
            theta_ref,
            allow_pareto=allow_pareto,
            cache_decimals=cache_decimals,
        )
        if cut_data is None:
            return None
        if cut_data.get("kind") == "feasibility":
            return cut_data if float(cut_data["rhs_at_hat"]) < -TOL else None
        return cut_data if theta_ref > float(cut_data["rhs_at_hat"]) + TOL else None

    def _violated_bdd_period_cut(
        self,
        period: str,
        x_ref: Dict[str, float],
        y_ref: Dict[Tuple[str, str], float],
        z_ref: Dict[Tuple[str, int], float],
        gamma_ref: float,
        theta_ref: float,
    ) -> Optional[Dict[str, Any]]:
        standard = self._select_cut(
            period,
            x_ref,
            y_ref,
            z_ref,
            gamma_ref,
            theta_ref,
            allow_pareto=False,
            cache_decimals=4,
        )
        if standard is None:
            return None
        if standard.get("kind") == "feasibility":
            if float(standard["rhs_at_hat"]) >= -TOL:
                return None
            standard = dict(standard)
            standard["source"] = f"bdd_{standard.get('source', 'dual')}"
            return standard

        if not self.use_bdd_strengthening:
            if theta_ref <= float(standard["rhs_at_hat"]) + TOL:
                return None
            if standard is not None:
                standard = dict(standard)
                standard["source"] = f"bdd_{standard.get('source', 'dual')}"
            return standard

        status, strengthened = self.bdd_strengtheners[period].solve(
            standard["duals"],
            x_ref,
            y_ref,
            z_ref,
            gamma_ref,
        )
        if status == GRB.OPTIMAL and strengthened is not None and theta_ref > float(strengthened["rhs_at_hat"]) + TOL:
            return strengthened

        if theta_ref <= float(standard["rhs_at_hat"]) + TOL:
            return None
        standard = dict(standard)
        standard["source"] = f"bdd_{standard.get('source', 'dual')}"
        return standard

    def _add_lazy_cut(self, model: gp.Model, period: str, cut_data: Dict[str, Any]) -> None:
        rhs_expr = self._cut_rhs_expr(period, cut_data)
        if cut_data.get("kind") == "feasibility":
            model.cbLazy(rhs_expr >= 0.0)
        else:
            model.cbLazy(self.Theta[period] <= rhs_expr)
        self.lazy_cuts += 1

    def _add_user_cut(self, model: gp.Model, period: str, cut_data: Dict[str, Any]) -> None:
        rhs_expr = self._cut_rhs_expr(period, cut_data)
        if cut_data.get("kind") == "feasibility":
            model.cbCut(rhs_expr >= 0.0)
        else:
            model.cbCut(self.Theta[period] <= rhs_expr)
        self.bdd_cuts += 1
        self.user_cuts += 1

    def _periods_to_check(self) -> tuple[list[str], list[str]]:
        ordered = self._ordered_decomposed_periods()
        first_count = self._initial_selective_count()
        return ordered[:first_count], ordered[first_count:]

    def _lazy_callback(self, model: gp.Model, where: int) -> None:
        if where == GRB.Callback.MIPSOL:
            self.callback_points += 1
            x_ref, y_ref, z_ref, gamma_ref, theta_ref = self._callback_values(model.cbGetSolution)
            first_batch, remainder = self._periods_to_check()
            added = 0

            for period in first_batch:
                cut_data = self._violated_period_cut(
                    period,
                    x_ref,
                    y_ref,
                    z_ref,
                    gamma_ref,
                    theta_ref[period],
                    allow_pareto=True,
                    cache_decimals=6,
                )
                if cut_data is not None:
                    self._add_lazy_cut(model, period, cut_data)
                    added += 1

            self._record_candidate_from_refs(x_ref, y_ref, z_ref, gamma_ref, theta_ref)
            if added > 0:
                return

            for period in remainder:
                cut_data = self._violated_period_cut(
                    period,
                    x_ref,
                    y_ref,
                    z_ref,
                    gamma_ref,
                    theta_ref[period],
                    allow_pareto=True,
                    cache_decimals=6,
                )
                if cut_data is not None:
                    self._add_lazy_cut(model, period, cut_data)
                    return

        elif self.use_bdd_cuts and where == GRB.Callback.MIPNODE and self.bdd_cuts < self.max_user_cuts:
            if int(model.cbGet(GRB.Callback.MIPNODE_STATUS)) != GRB.OPTIMAL:
                return
            self.node_callbacks += 1
            if self.node_callbacks % self.node_cut_interval != 0:
                return
            x_ref, y_ref, z_ref, gamma_ref, theta_ref = self._callback_values(model.cbGetNodeRel)
            first_batch, remainder = self._periods_to_check()
            added = 0
            for period in first_batch:
                cut_data = self._violated_bdd_period_cut(
                    period,
                    x_ref,
                    y_ref,
                    z_ref,
                    gamma_ref,
                    theta_ref[period],
                )
                if cut_data is not None:
                    self._add_user_cut(model, period, cut_data)
                    added += 1
                    if self.bdd_cuts >= self.max_user_cuts:
                        return

            if added > 0:
                return

            for period in remainder:
                cut_data = self._violated_bdd_period_cut(
                    period,
                    x_ref,
                    y_ref,
                    z_ref,
                    gamma_ref,
                    theta_ref[period],
                )
                if cut_data is not None:
                    self._add_user_cut(model, period, cut_data)
                    return

    def _decomposed_obj_values(
        self,
        x_ref: Dict[str, float],
        y_ref: Dict[Tuple[str, str], float],
        z_ref: Dict[Tuple[str, int], float],
        gamma_ref: float,
    ) -> Optional[Dict[str, float]]:
        period_objs: Dict[str, float] = {}
        for period in self.decomposed_periods:
            cut_data = self.oracles[period].solve(x_ref, y_ref, z_ref, gamma_ref)
            if cut_data is None or cut_data.get("kind") == "feasibility":
                return None
            period_objs[period] = float(cut_data["rhs_at_hat"])
        return period_objs

    def _evaluate_reference_solution(
        self,
        x_ref: Dict[str, float],
        y_ref: Dict[Tuple[str, str], float],
        z_ref: Dict[Tuple[str, int], float],
        gamma_ref: float,
    ) -> Optional[Tuple[float, Dict[str, float]]]:
        period_objs: Dict[str, float] = {}
        for period in self.retained_periods:
            solver = RegimeIIPrimalSubproblem(self.data_by_period[period], output=False)
            status, retained_obj, _ = solver.solve(x_ref, y_ref, z_ref, gamma_ref)
            if status != GRB.OPTIMAL or retained_obj is None:
                return None
            period_objs[period] = float(retained_obj)

        decomposed_objs = self._decomposed_obj_values(x_ref, y_ref, z_ref, gamma_ref)
        if decomposed_objs is None:
            return None
        period_objs.update(decomposed_objs)

        true_obj = float(
            sum(self.weights[p] * period_objs[p] for p in self.periods)
            - self.data0.fixed_cost_value(x_ref, z_ref)
            - self.data0.gamma_cost * gamma_ref
        )
        return true_obj, period_objs

    def _record_candidate_from_refs(
        self,
        x_ref: Dict[str, float],
        y_ref: Dict[Tuple[str, str], float],
        z_ref: Dict[Tuple[str, int], float],
        gamma_ref: float,
        theta_ref: Dict[str, float],
    ) -> None:
        start = time.perf_counter()
        evaluated = self._evaluate_reference_solution(x_ref, y_ref, z_ref, gamma_ref)
        self.subproblem_time += time.perf_counter() - start
        self.candidate_evals += 1
        if evaluated is None:
            self.uncertified_incumbents += 1
            self.last_uncertified_point = {
                "x": x_ref,
                "y": y_ref,
                "z": z_ref,
                "Gamma": gamma_ref,
                "theta": theta_ref,
            }
            return

        true_obj, period_objs = evaluated
        if true_obj > self.best_obj:
            self.best_obj = true_obj
            self.best_period_objs = dict(period_objs)
            self.best_solution = MultiPeriodMasterSolution(
                x=dict(x_ref),
                y=dict(y_ref),
                z=dict(z_ref),
                Gamma=float(gamma_ref),
                theta=dict(theta_ref),
                obj=true_obj,
            )

    def _final_certification_cuts(self) -> list[tuple[str, Dict[str, Any]]]:
        assert self.model is not None
        x_ref, y_ref, z_ref, gamma_ref, theta_ref = self._callback_values(lambda var: var.X)
        cuts = []
        for period in self.decomposed_periods:
            cut_data = self._violated_period_cut(
                period,
                x_ref,
                y_ref,
                z_ref,
                gamma_ref,
                theta_ref[period],
                allow_pareto=True,
                cache_decimals=6,
            )
            if cut_data is not None:
                cuts.append((period, cut_data))
        return cuts

    def solve(self) -> Optional[float]:
        self._build_master()
        assert self.model is not None
        if self.use_initial_cuts:
            self._add_initial_cuts()

        self._start_time = time.perf_counter()
        final_rounds = 0
        while True:
            elapsed = time.perf_counter() - self._start_time
            remaining = self.total_time_limit - elapsed
            if remaining <= TOL:
                break
            self.model.setParam("TimeLimit", max(1.0, remaining))
            self.model.optimize(self._lazy_callback)

            if self.model.SolCount == 0:
                break
            cuts = self._final_certification_cuts()
            if not cuts:
                break
            for period, cut_data in cuts:
                self._add_static_cut(period, cut_data, f"mp_ts_cb_final_{self.final_cuts + 1}_{period}_{cut_data['source']}")
                self.final_cuts += 1
            final_rounds += 1
            if final_rounds >= 25:
                break

        wall = time.perf_counter() - self._start_time
        obj = self._extract_solution(wall)
        if self.stop_reason is None:
            if self.model.SolCount == 0:
                self.stop_reason = "master_no_incumbent"
            elif self.converged:
                self.stop_reason = "gap_tol"
            elif self.model.Status == GRB.OPTIMAL:
                self.stop_reason = "gurobi_optimal"
            elif self.model.Status == GRB.TIME_LIMIT:
                self.stop_reason = "time_limit"
            else:
                self.stop_reason = f"status_{self.model.Status}"
        return obj

    def _extract_solution(self, wall: float) -> Optional[float]:
        assert self.model is not None
        if self.model.SolCount == 0:
            return None

        sol = self._solution()
        period_objs: Dict[str, float] = {}
        cut_sources: Dict[str, str] = {}
        all_optimal = True

        sub_start = time.perf_counter()
        for period in self.retained_periods:
            solver = RegimeIIPrimalSubproblem(self.data_by_period[period], output=False)
            status, retained_obj, _ = solver.solve(sol.x, sol.y, sol.z, sol.Gamma)
            if status != GRB.OPTIMAL or retained_obj is None:
                all_optimal = False
                cut_sources[period] = f"retained_reprice_status_{status}"
                continue
            period_objs[period] = float(retained_obj)
            cut_sources[period] = "retained_repriced"

        decomposed_objs = self._decomposed_obj_values(sol.x, sol.y, sol.z, sol.Gamma)
        if decomposed_objs is None:
            all_optimal = False
        else:
            for period, value in decomposed_objs.items():
                period_objs[period] = value
                cut_sources[period] = "callback_dual"
        self.subproblem_time += time.perf_counter() - sub_start

        true_obj = None
        if all_optimal:
            true_obj = float(
                sum(self.weights[p] * period_objs[p] for p in self.periods)
                - self._fixed_cost_value(sol)
                - self.data0.gamma_cost * sol.Gamma
            )
            if true_obj > self.best_obj:
                self.best_obj = true_obj
                self.best_period_objs = dict(period_objs)
                self.best_solution = sol

        if self.best_obj <= -math.inf:
            return None

        bound = float(self.model.ObjBound) if self.model.Status in {GRB.OPTIMAL, GRB.TIME_LIMIT} else sol.obj
        gap = (bound - self.best_obj) / max(1.0, abs(self.best_obj))
        self.converged = gap <= self.gap_tol

        row = {
            "iter": 1,
            "master_obj": sol.obj,
            "master_bound": bound,
            "true_obj": true_obj,
            "best_obj": self.best_obj,
            "gap": gap,
            "max_theta_gap": 0.0,
            "Gamma": sol.Gamma,
            "retained_periods": list(self.retained_periods),
            "decomposed_periods": list(self.decomposed_periods),
            "period_objs": dict(self.best_period_objs or period_objs),
            "cut_sources": dict(cut_sources),
            "time": wall,
            "lazy_cuts": self.lazy_cuts,
            "bdd_cuts": self.bdd_cuts,
            "user_cuts": self.user_cuts,
            "initial_cuts": self.initial_cuts,
            "final_cuts": self.final_cuts,
            "callback_points": self.callback_points,
            "candidate_evals": self.candidate_evals,
            "uncertified_incumbents": self.uncertified_incumbents,
        }
        self.history.append(row)

        if self.verbose:
            sub_solves = sum(oracle.solve_count for oracle in self.oracles.values())
            sub_time = sum(oracle.subproblem_time for oracle in self.oracles.values())
            print(
                "[MP-TS-PBD-CB] "
                f"obj={self.best_obj:.4f} current={true_obj} master={sol.obj:.4f} bound={bound:.4f} gap={gap:.2%} "
                f"retained={self.retained_periods} lazy={self.lazy_cuts} bdd={self.bdd_cuts} "
                f"initial={self.initial_cuts} final={self.final_cuts} "
                f"callbacks={self.callback_points} evals={self.candidate_evals} sub_solves={sub_solves} "
                f"oracle_time={sub_time:.4f}s wall={wall:.2f}s"
            )
        return self.best_obj


def run_multi_period_ts_partial_benders_callback(
    mp_instance: MultiPeriodInstance,
    *,
    retained_periods: Sequence[str] | None = None,
    retained_count: int = 1,
    retention_rule: str = "largest_demand",
    **kwargs,
) -> MultiPeriodScenarioRetentionCallbackBenders:
    solver = MultiPeriodScenarioRetentionCallbackBenders(
        mp_instance,
        retained_periods=retained_periods,
        retained_count=retained_count,
        retention_rule=retention_rule,
        **kwargs,
    )
    solver.solve()
    return solver


def run_meituan_multi_period_ts_partial_benders_callback(
    *,
    project_root=None,
    instance_dir=None,
    period_ids=None,
    demand_hours=None,
    period_weights=None,
    retained_periods: Sequence[str] | None = None,
    retained_count: int = 1,
    retention_rule: str = "largest_demand",
    **kwargs,
) -> MultiPeriodScenarioRetentionCallbackBenders:
    if project_root is None:
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if instance_dir is None:
        instance_dir = os.path.join(project_root, "data", "instances", "MeituanMultiPeriod")
    periods = _as_period_ids(period_ids=period_ids, demand_hours=demand_hours)
    mp_inst = load_multi_period_instance(instance_dir, periods, period_weights=period_weights)
    return run_multi_period_ts_partial_benders_callback(
        mp_inst,
        retained_periods=retained_periods,
        retained_count=retained_count,
        retention_rule=retention_rule,
        **kwargs,
    )


if __name__ == "__main__":
    run_meituan_multi_period_ts_partial_benders_callback(
        demand_hours=[8, 10, 12, 14, 17],
        retained_count=2,
        time_limit=60.0,
        verbose=True,
    )
