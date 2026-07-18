"""Independent LR-only and strict conic Benders solvers dated 2026-06-23."""

from __future__ import annotations

import math
import time
from typing import Dict, Mapping, Optional, Tuple

import gurobipy as gp
import networkx as nx
from gurobipy import GRB

from src import config
from src.data_processor.instance import Instance
from src.model.Benders import BendersRegimeIISolver, SubproblemResult
from src.model.gurobi_model import MLNDModel

from .bdd import BDDCutGenerator20260623, solve_fractional_master
from .local_branching import AdaptiveLocalBranching20260623
from .linearized import (
    add_queue_supporting_lines,
    remove_all_quadratic_constraints,
)
from .lr import LagrangianEngine20260623
from .master import assert_pure_master, build_pure_master
from .pareto import ParetoCutSelector20260623
from .profiling import PhaseProfiler
from .types import (
    CutAuditRecord,
    Design,
    PhaseTiming,
    SolveResult20260623,
    design_signature,
)


def _status_label(status: int) -> str:
    names = {
        GRB.OPTIMAL: "OPTIMAL",
        GRB.INFEASIBLE: "INFEASIBLE",
        GRB.INF_OR_UNBD: "INF_OR_UNBD",
        GRB.UNBOUNDED: "UNBOUNDED",
        GRB.CUTOFF: "CUTOFF",
        GRB.ITERATION_LIMIT: "ITERATION_LIMIT",
        GRB.NODE_LIMIT: "NODE_LIMIT",
        GRB.TIME_LIMIT: "TIME_LIMIT",
        GRB.SOLUTION_LIMIT: "SOLUTION_LIMIT",
        GRB.INTERRUPTED: "INTERRUPTED",
        GRB.NUMERIC: "NUMERIC",
        GRB.SUBOPTIMAL: "SUBOPTIMAL",
    }
    return names.get(int(status), str(status))


def _candidate_log_fields(design: Mapping[str, object]) -> Dict[str, object]:
    metadata = design.get("_lr_meta", {})
    if not isinstance(metadata, Mapping):
        metadata = {}
    x_values = design.get("x", {})
    y_values = design.get("y", {})
    z_values = design.get("z", {})
    return {
        "candidate_source": metadata.get("source"),
        "candidate_seed_policy": metadata.get("seed_policy"),
        "candidate_seed_index": metadata.get("seed_index"),
        "candidate_selected_pairs": metadata.get("selected_pairs"),
        "candidate_hub_fraction": metadata.get("hub_fraction"),
        "candidate_gamma_center": metadata.get("gamma_center"),
        "candidate_fix_hub_set": metadata.get("fix_hub_set"),
        "candidate_surrogate_score": metadata.get("surrogate_score"),
        "open_x": sum(float(value) > 0.5 for value in x_values.values()),
        "open_y": sum(float(value) > 0.5 for value in y_values.values()),
        "open_z": sum(float(value) > 0.5 for value in z_values.values()),
    }


class RecourseOracle20260623(BendersRegimeIISolver):
    """Stable fixed-design SOCP oracle plus shared strategic data helpers."""

    def __init__(self, instance: Instance) -> None:
        super().__init__(instance, use_pareto_cuts=False)
        self.profiler = PhaseProfiler()

    def route_requirements(self, u, v, k) -> Tuple[Tuple, ...]:
        if k in self.E:
            return ((u, v), (u, k), (k, v))
        return ((u, v),)

    def nearest_base(self, hub):
        bases = [base for base, h in self.A_LH if h == hub]
        if not bases:
            return None
        return min(bases, key=lambda base: self.inst.get_dist(base, hub))

    def recovery_route(self, u, v):
        options = list(self.route_options.get((u, v), []))
        if not options:
            return None
        swap_options = [option for option in options if option in self.E]
        return min(swap_options or options, key=lambda k: self.t[u, v, k])

    def solve_subproblem_with_deadline(
        self,
        design: Mapping[str, object],
        *,
        label: str,
        deadline: float,
    ) -> SubproblemResult:
        remaining = deadline - time.perf_counter()
        if remaining <= 0.01:
            return SubproblemResult(
                status=GRB.TIME_LIMIT,
                obj_value=None,
                cut_expr_data=None,
                rhs_at_reference=None,
                design=design,
                label=label,
            )
        had_limit = hasattr(config, "BENDERS_SUB_TIME_LIMIT")
        old_limit = getattr(config, "BENDERS_SUB_TIME_LIMIT", None)
        configured = (
            remaining
            if old_limit is None
            else min(float(old_limit), remaining)
        )
        setattr(config, "BENDERS_SUB_TIME_LIMIT", max(0.01, configured))
        start = time.perf_counter()
        try:
            result = self.solve_subproblem(design, label=label)
        finally:
            if had_limit:
                setattr(config, "BENDERS_SUB_TIME_LIMIT", old_limit)
            else:
                delattr(config, "BENDERS_SUB_TIME_LIMIT")
        elapsed = time.perf_counter() - start
        timing = self.profiler._timings.setdefault(
            "exact_socp_recourse",
            PhaseTiming(phase="exact_socp_recourse"),
        )
        timing.wall_time += elapsed
        # The inherited SOCP oracle does not expose Gurobi Runtime in
        # SubproblemResult.  Count this phase as solver-dominated so LR C++
        # decisions are not distorted by exact validation calls.
        timing.solver_time += elapsed
        timing.calls += 1
        return result

    def _gamma_dominance_upper_bound(self) -> int:
        drone_cost = float(config.COST_DRONE)
        if drone_cost <= 1e-9:
            return 0
        revenue = float(getattr(config, "ORDER_REVENUE_COEF", 0.0))
        total_revenue = revenue * sum(
            max(0.0, float(self.inst.h.get(pair, 0.0)))
            for pair in self.A_HH
        )
        return max(0, int(math.ceil(total_revenue / drone_cost)))

    def _clone_design_with_gamma(
        self,
        design: Mapping[str, object],
        gamma: int,
        *,
        source: str,
        center_gamma: int,
    ) -> Design:
        metadata = design.get("_lr_meta", {})
        if not isinstance(metadata, Mapping):
            metadata = {}
        new_metadata = dict(metadata)
        base_source = new_metadata.get("source")
        new_metadata.update(
            {
                "source": source,
                "base_source": base_source,
                "gamma_center": int(center_gamma),
            }
        )
        return {
            "x": dict(design["x"]),
            "y": dict(design["y"]),
            "z": dict(design["z"]),
            "Gamma": float(max(0, int(gamma))),
            "_lr_meta": new_metadata,
        }

    def _gamma_sweep_values(
        self,
        center_gamma: float,
        *,
        radius: int,
        step: int,
    ) -> list[int]:
        """Deterministic near-neighborhood for exact fixed-network validation."""

        upper = self._gamma_dominance_upper_bound()
        center = max(0, min(upper, int(round(float(center_gamma)))))
        step = max(1, int(step))
        radius = max(0, int(radius))
        values: list[int] = []
        seen: set[int] = set()

        def add(value: float) -> None:
            gamma = max(0, min(upper, int(round(float(value)))))
            if gamma in seen:
                return
            seen.add(gamma)
            values.append(gamma)

        add(center)
        for delta in range(step, radius + 1, step):
            add(center + delta)
        for delta in range(step, radius + 1, step):
            add(center - delta)
        for factor in (0.85, 0.90, 0.95, 1.05, 1.10, 1.15, 1.20):
            add(center * factor)
        if radius > 0:
            add(center + radius)
            add(center - radius)
        return values

    def solve_global_linearized_upper_bound(
        self,
        *,
        deadline: float,
        total_time_limit: float,
        time_fraction: float,
        time_limit: float,
        label: str,
    ) -> Dict[str, object]:
        """Solve a cone-free full-network relaxation and return its ObjBound.

        The original queueing cones are removed and replaced by globally valid
        supporting-line lower bounds.  The resulting feasible set contains the
        original feasible set, so the maximization ObjBound is a valid upper
        bound for the original MISOCP objective.  The relaxation incumbent is
        intentionally not used as a primal solution.
        """

        remaining = deadline - time.perf_counter()
        budget = min(
            remaining,
            max(0.0, float(time_limit)),
            max(0.0, float(time_fraction)) * float(total_time_limit),
        )
        if budget <= 0.05:
            return {
                "global_linearized_bound": None,
                "global_linearized_status": None,
                "global_linearized_wall_time": 0.0,
                "global_linearized_qconstrs_before": None,
                "global_linearized_qconstrs_after": None,
                "global_linearized_queue_tangents": 0,
            }

        relaxation = MLNDModel(self.inst)
        relaxation.build()
        model = relaxation.model
        model.ModelName = f"MLND_20260623_Global_Linearized_UB_{label}"
        model.Params.OutputFlag = 0
        model.Params.Threads = 1
        model.Params.Seed = int(getattr(config, "BENDERS_RANDOM_SEED", 0))
        model.Params.TimeLimit = budget
        model.Params.MIPGap = float(
            getattr(config, "BDD_GLOBAL_LINEAR_MIP_GAP", 0.002)
        )
        q_metadata = remove_all_quadratic_constraints(relaxation)
        queue_tangents = add_queue_supporting_lines(
            relaxation, prefix=f"{label}_queue_tangent"
        )
        start = time.perf_counter()
        with self.profiler.phase("global_linearized_upper_bound_milp"):
            model.optimize()
        wall_time = time.perf_counter() - start
        self.profiler.add_solver_time(
            "global_linearized_upper_bound_milp", model.Runtime
        )
        raw_bound = float(model.ObjBound)
        bound = raw_bound if math.isfinite(raw_bound) else None
        return {
            "global_linearized_bound": bound,
            "global_linearized_status": int(model.Status),
            "global_linearized_wall_time": wall_time,
            "global_linearized_qconstrs_before": q_metadata[
                "qconstrs_before"
            ],
            "global_linearized_qconstrs_after": q_metadata["qconstrs_after"],
            "global_linearized_queue_tangents": queue_tangents,
        }


class LROnlySolver20260623(RecourseOracle20260623):
    """LR dual plus recovered designs, with exact SOCP used only for evaluation."""

    def __init__(
        self,
        instance: Instance,
        *,
        lr_iterations: int = 50,
        lr_initial_step: float = 20.0,
        lr_max_designs: int = 12,
        lr_use_queue_tangents: bool = False,
        lr_restricted_solution_pool_size: int = 1,
        lr_anchor_top_pairs: int = 0,
        lr_restricted_time_limit: Optional[float] = None,
        gamma_sweep_max_calls: int = 0,
        gamma_sweep_radius: int = 40,
        gamma_sweep_step: int = 8,
        gamma_sweep_time_reserve: float = 0.0,
        use_global_linearized_bound: bool = True,
        global_bound_time_fraction: float = 0.10,
        global_bound_time_limit: float = 60.0,
    ) -> None:
        super().__init__(instance)
        self.lr_engine = LagrangianEngine20260623(
            self,
            iterations=lr_iterations,
            initial_step=lr_initial_step,
            max_designs=lr_max_designs,
            use_queue_tangents=lr_use_queue_tangents,
            restricted_solution_pool_size=lr_restricted_solution_pool_size,
            anchor_top_pairs=lr_anchor_top_pairs,
            restricted_time_limit=lr_restricted_time_limit,
        )
        self.gamma_sweep_max_calls = max(0, int(gamma_sweep_max_calls))
        self.gamma_sweep_radius = max(0, int(gamma_sweep_radius))
        self.gamma_sweep_step = max(1, int(gamma_sweep_step))
        self.gamma_sweep_time_reserve = max(
            0.0, float(gamma_sweep_time_reserve)
        )
        self.use_global_linearized_bound = bool(use_global_linearized_bound)
        self.global_bound_time_fraction = max(
            0.0, float(global_bound_time_fraction)
        )
        self.global_bound_time_limit = max(
            0.0, float(global_bound_time_limit)
        )

    def solve(self, *, time_limit: float) -> SolveResult20260623:
        start = time.perf_counter()
        deadline = start + max(0.0, float(time_limit))
        total_limit = max(0.0, float(time_limit))
        validation_reserve = (
            5.0
            if total_limit <= 60.0
            else 30.0
            if total_limit <= 300.0
            else 90.0
        )
        generation_budget = max(
            0.05,
            min(0.80 * total_limit, total_limit - validation_reserve),
        )
        lr_result = self.lr_engine.generate(time_limit=generation_budget)
        best_obj = -math.inf
        best_design = None
        time_to_best = None
        initial_history = []
        gamma_sweep_history = []
        evaluated_signatures = set()
        candidate_deadline = deadline
        if (
            self.gamma_sweep_max_calls > 0
            and self.gamma_sweep_time_reserve > 0.0
        ):
            candidate_deadline = max(
                time.perf_counter(),
                deadline - self.gamma_sweep_time_reserve,
            )
        for index, design in enumerate(lr_result.designs, 1):
            if time.perf_counter() >= candidate_deadline:
                break
            evaluated_signatures.add(design_signature(design))
            eval_start = time.perf_counter()
            sub = self.solve_subproblem_with_deadline(
                design,
                label=f"lr_only_candidate_{index}",
                deadline=candidate_deadline,
            )
            eval_wall_time = time.perf_counter() - eval_start
            fixed_cost = self._fixed_cost_value(design)
            objective = (
                None
                if sub.obj_value is None
                else float(sub.obj_value) - fixed_cost
            )
            status_name = _status_label(int(sub.status))
            initial_history.append(
                {
                    "index": index,
                    "source": "lagrangian",
                    "status": int(sub.status),
                    "status_name": status_name,
                    "sub_obj": sub.obj_value,
                    "fixed_cost": fixed_cost,
                    "total_obj": objective,
                    "Gamma": float(design.get("Gamma", 0.0)),
                    "candidate_eval_wall_time": eval_wall_time,
                    "failure_reason": (
                        None
                        if objective is not None
                        else f"exact_socp_{status_name}"
                    ),
                    **_candidate_log_fields(design),
                }
            )
            if objective is not None and objective > best_obj + 1e-7:
                best_obj = objective
                best_design = design
                time_to_best = time.perf_counter() - start

        best_initial_obj = best_obj if best_design is not None else None
        if (
            best_design is not None
            and self.gamma_sweep_max_calls > 0
            and time.perf_counter() < deadline
        ):
            center_gamma = int(round(float(best_design.get("Gamma", 0.0))))
            evaluated = 0
            base_obj = float(best_obj)
            for gamma in self._gamma_sweep_values(
                center_gamma,
                radius=self.gamma_sweep_radius,
                step=self.gamma_sweep_step,
            ):
                if evaluated >= self.gamma_sweep_max_calls:
                    break
                if time.perf_counter() >= deadline:
                    break
                candidate = self._clone_design_with_gamma(
                    best_design,
                    gamma,
                    source="lr_gamma_sweep",
                    center_gamma=center_gamma,
                )
                signature = design_signature(candidate)
                if signature in evaluated_signatures:
                    continue
                evaluated_signatures.add(signature)
                evaluated += 1
                eval_start = time.perf_counter()
                sub = self.solve_subproblem_with_deadline(
                    candidate,
                    label=f"lr_gamma_sweep_{evaluated}",
                    deadline=deadline,
                )
                eval_wall_time = time.perf_counter() - eval_start
                fixed_cost = self._fixed_cost_value(candidate)
                objective = (
                    None
                    if sub.obj_value is None
                    else float(sub.obj_value) - fixed_cost
                )
                status_name = _status_label(int(sub.status))
                improved = (
                    objective is not None
                    and objective > best_obj + 1e-7
                )
                gamma_sweep_history.append(
                    {
                        "call": evaluated,
                        "source": "lr_gamma_sweep",
                        "center_gamma": center_gamma,
                        "Gamma": float(gamma),
                        "status": int(sub.status),
                        "status_name": status_name,
                        "sub_obj": sub.obj_value,
                        "fixed_cost": fixed_cost,
                        "total_obj": objective,
                        "base_obj": base_obj,
                        "improvement_vs_base": (
                            None
                            if objective is None
                            else float(objective) - base_obj
                        ),
                        "improved_incumbent": bool(improved),
                        "candidate_eval_wall_time": eval_wall_time,
                        "failure_reason": (
                            None
                            if objective is not None
                            else f"exact_socp_{status_name}"
                        ),
                    }
                )
                if improved:
                    best_obj = float(objective)
                    best_design = candidate
                    time_to_best = time.perf_counter() - start

        success = best_design is not None
        objective = best_obj if success else None
        raw_lr_bound = (
            lr_result.best_dual_bound
            if math.isfinite(lr_result.best_dual_bound)
            else None
        )
        global_bound_metadata = {
            "global_linearized_bound": None,
            "global_linearized_status": None,
            "global_linearized_wall_time": 0.0,
            "global_linearized_qconstrs_before": None,
            "global_linearized_qconstrs_after": None,
            "global_linearized_queue_tangents": 0,
        }
        if (
            self.use_global_linearized_bound
            and time.perf_counter() < deadline
        ):
            global_bound_metadata = self.solve_global_linearized_upper_bound(
                deadline=deadline,
                total_time_limit=total_limit,
                time_fraction=self.global_bound_time_fraction,
                time_limit=self.global_bound_time_limit,
                label="lr_only",
            )
        linearized_bound = global_bound_metadata.get(
            "global_linearized_bound"
        )
        candidate_bounds = [
            float(value)
            for value in (raw_lr_bound, linearized_bound)
            if value is not None and math.isfinite(float(value))
        ]
        bound = min(candidate_bounds) if candidate_bounds else None
        if objective is not None and bound is not None:
            bound = max(float(bound), float(objective))
        gap = (
            None
            if objective is None or bound is None
            else (bound - objective) / max(1.0, abs(objective))
        )
        wall_time = time.perf_counter() - start
        timing_rows = [
            *self.lr_engine.profiler.rows(),
            *self.profiler.rows(),
        ]
        cpp_decision = self.lr_engine.profiler.cpp_acceleration_decision(
            total_lr_time=lr_result.wall_time
        )
        return SolveResult20260623(
            method="lr",
            success=success,
            objective=objective,
            bound=bound,
            gap=gap,
            wall_time=wall_time,
            time_to_best=time_to_best,
            best_design=best_design,
            initial_design_history=initial_history,
            gamma_sweep_history=gamma_sweep_history,
            lr_result=lr_result,
            metadata={
                "lr_bound": raw_lr_bound,
                "raw_lr_bound": raw_lr_bound,
                "lr_reported_bound_uses_global_linearized_min": (
                    linearized_bound is not None
                    and raw_lr_bound is not None
                    and float(linearized_bound) < float(raw_lr_bound)
                ),
                "lr_wall_time": lr_result.wall_time,
                "timing_rows": timing_rows,
                "cpp_acceleration_decision": cpp_decision,
                "master_is_milp": None,
                "master_qconstrs": None,
                "initial_design_count": len(lr_result.designs),
                "validated_initial_design_count": len(initial_history),
                "failed_initial_design_count": sum(
                    row.get("total_obj") is None for row in initial_history
                ),
                "best_initial_design_obj": best_initial_obj,
                "best_initial_design_validation_obj": best_initial_obj,
                "gamma_sweep_enabled": self.gamma_sweep_max_calls > 0,
                "gamma_sweep_max_calls": self.gamma_sweep_max_calls,
                "gamma_sweep_radius": self.gamma_sweep_radius,
                "gamma_sweep_step": self.gamma_sweep_step,
                "gamma_sweep_time_reserve": self.gamma_sweep_time_reserve,
                "gamma_sweep_evaluated": len(gamma_sweep_history),
                "gamma_sweep_best_obj": objective,
                "gamma_sweep_improved_count": sum(
                    1
                    for row in gamma_sweep_history
                    if row.get("improved_incumbent")
                ),
                "lr_candidate_count": len(lr_result.candidate_history),
                "exact_socp_solver_time_is_wall_proxy": True,
                "global_linearized_bound_enabled": (
                    self.use_global_linearized_bound
                ),
                "global_bound_time_fraction": self.global_bound_time_fraction,
                "global_bound_time_limit": self.global_bound_time_limit,
                **global_bound_metadata,
                **lr_result.metadata,
            },
        )


class ConicBendersSolver20260623(RecourseOracle20260623):
    """Strict MILP-master BD with optional LR start and local branching."""

    def __init__(
        self,
        instance: Instance,
        *,
        use_lr: bool,
        use_pareto_cuts: bool = True,
        use_bdd_strengthened: bool = True,
        use_bdd_lagrangian: bool = True,
        use_local_branching: bool = False,
        lr_iterations: int = 50,
        lr_initial_step: float = 20.0,
        lr_max_designs: int = 12,
        lr_use_queue_tangents: bool = False,
        lr_restricted_solution_pool_size: int = 1,
        lr_anchor_top_pairs: int = 0,
        lr_restricted_time_limit: Optional[float] = None,
        local_branching_radius: int = 6,
        local_branching_gamma_radius: int = 8,
        local_branching_max_calls: int = 32,
        local_branching_max_radius: int = 160,
        local_branching_max_gamma_radius: int = 240,
        local_branching_distance_scope: str = "all",
        local_branching_start_pool_size: int = 1,
        local_branching_max_calls_per_center: int = 0,
        local_branching_arc_radius: int = 64,
        local_branching_max_arc_radius: int = 260,
        local_branching_solution_pool_size: int = 1,
        local_branching_benders_envelope_cuts: int = 0,
        local_branching_benders_envelope_weight: float = 1.0,
        local_branching_benders_envelope_cutoff: bool = False,
        local_branching_max_removals: Optional[int] = None,
        local_branching_protect_top_pairs: int = 0,
        gamma_sweep_max_calls: int = 0,
        gamma_sweep_radius: int = 40,
        gamma_sweep_step: int = 8,
        gamma_sweep_time_reserve: float = 0.0,
        bdd_root_rounds: int = 3,
        bdd_inner_time_limit: float = 10.0,
        bdd_full_integrality: bool = False,
        bdd_epsilon_abs: Optional[float] = None,
        use_global_linearized_bound: bool = True,
        use_master_proxy: bool = True,
        global_bound_time_fraction: float = 0.10,
        global_bound_time_limit: float = 60.0,
        verbose: bool = False,
    ) -> None:
        super().__init__(instance)
        self.use_lr = bool(use_lr)
        self.use_pareto_cuts = bool(use_pareto_cuts)
        self.use_bdd_strengthened = bool(use_bdd_strengthened)
        self.use_bdd_lagrangian = bool(use_bdd_lagrangian)
        self.use_local_branching = bool(use_local_branching)
        self.verbose = bool(verbose)
        self.lr_engine = LagrangianEngine20260623(
            self,
            iterations=lr_iterations,
            initial_step=lr_initial_step,
            max_designs=lr_max_designs,
            use_queue_tangents=lr_use_queue_tangents,
            restricted_solution_pool_size=lr_restricted_solution_pool_size,
            anchor_top_pairs=lr_anchor_top_pairs,
            restricted_time_limit=lr_restricted_time_limit,
        )
        self.local_branching = AdaptiveLocalBranching20260623(
            self,
            initial_radius=local_branching_radius,
            gamma_radius=local_branching_gamma_radius,
            max_gamma_radius=local_branching_max_gamma_radius,
            max_calls=local_branching_max_calls,
            max_radius=local_branching_max_radius,
            distance_scope=local_branching_distance_scope,
            arc_radius=local_branching_arc_radius,
            max_arc_radius=local_branching_max_arc_radius,
            solution_pool_size=local_branching_solution_pool_size,
            benders_envelope_cuts=local_branching_benders_envelope_cuts,
            benders_envelope_weight=local_branching_benders_envelope_weight,
            benders_envelope_cutoff=local_branching_benders_envelope_cutoff,
            max_removals=local_branching_max_removals,
            protect_top_pairs=local_branching_protect_top_pairs,
        )
        self.local_branching_start_pool_size = max(
            1, int(local_branching_start_pool_size)
        )
        self.local_branching_max_calls_per_center = max(
            0, int(local_branching_max_calls_per_center)
        )
        self.gamma_sweep_max_calls = max(0, int(gamma_sweep_max_calls))
        self.gamma_sweep_radius = max(0, int(gamma_sweep_radius))
        self.gamma_sweep_step = max(1, int(gamma_sweep_step))
        self.gamma_sweep_time_reserve = max(
            0.0, float(gamma_sweep_time_reserve)
        )
        self.pareto_selector = ParetoCutSelector20260623(self)
        self.bdd_generator = BDDCutGenerator20260623(
            self,
            full_integrality=bdd_full_integrality,
            epsilon_abs=bdd_epsilon_abs,
        )
        self.bdd_root_rounds = max(0, int(bdd_root_rounds))
        self.bdd_inner_time_limit = max(0.05, float(bdd_inner_time_limit))
        self.use_global_linearized_bound = bool(
            use_global_linearized_bound
        )
        self.use_master_proxy = bool(use_master_proxy)
        self.proxy_top_pairs: Tuple[Tuple, ...] = ()
        self.master_proxy_metadata: Dict[str, object] = {}
        self.global_bound_time_fraction = max(
            0.0, float(global_bound_time_fraction)
        )
        self.global_bound_time_limit = max(
            0.0, float(global_bound_time_limit)
        )
        self.global_linearized_bound: Optional[float] = None
        self.global_linearized_status: Optional[int] = None
        self.global_linearized_wall_time = 0.0
        self.global_linearized_qconstrs_before: Optional[int] = None
        self.global_linearized_qconstrs_after: Optional[int] = None
        self.global_linearized_queue_tangents = 0
        self.cut_audit_history: list[CutAuditRecord] = []
        self._benders_cut_pool: list[SubproblemResult] = []
        self.initial_design_history: list[Dict[str, object]] = []
        self.gamma_sweep_history: list[Dict[str, object]] = []
        self._local_branching_start_pool: list[
            Tuple[float, Design]
        ] = []
        self.current_iteration = 0
        self.time_to_best: Optional[float] = None
        self._solve_start = 0.0
        self._evaluated_signatures = set()
        self._known_exact_recourse: list[
            Tuple[Design, float]
        ] = []
        self._topology_cut_signatures = set()
        self._swap_cut_signatures = set()
        self.topology_cuts = 0
        self.swap_reachability_cuts = 0
        self.exact_design_nogood_cuts = 0

    def build_master(self) -> None:
        build_pure_master(self)

    def _add_benders_cut(
        self,
        sub_result: SubproblemResult,
        cut_name_prefix: str,
    ) -> None:
        super()._add_benders_cut(sub_result, cut_name_prefix)
        if (
            sub_result.obj_value is not None
            and sub_result.cut_expr_data is not None
            and sub_result.rhs_at_reference is not None
            and sub_result.cut_expr_data
        ):
            self._benders_cut_pool.append(sub_result)

    @staticmethod
    def remaining_time(deadline: float) -> float:
        return deadline - time.perf_counter()

    @staticmethod
    def _cut_rhs_value(
        sub: SubproblemResult, design: Mapping[str, object]
    ) -> Optional[float]:
        if (
            sub.obj_value is None
            or sub.cut_expr_data is None
            or sub.rhs_at_reference is None
        ):
            return None
        value = float(sub.obj_value)
        for (pi, rhs_fun), rhs_reference in zip(
            sub.cut_expr_data, sub.rhs_at_reference
        ):
            rhs_value = rhs_fun(design)
            if not isinstance(rhs_value, (int, float)):
                rhs_value = rhs_value.getValue()
            value += float(pi) * (float(rhs_value) - float(rhs_reference))
        return value

    def register_known_exact(
        self, design: Mapping[str, object], operational_value: float
    ) -> None:
        signature = design_signature(design)
        if any(
            design_signature(existing) == signature
            for existing, _ in self._known_exact_recourse
        ):
            return
        normalized = {
            "x": dict(design["x"]),
            "y": dict(design["y"]),
            "z": dict(design["z"]),
            "Gamma": float(design["Gamma"]),
        }
        self._known_exact_recourse.append(
            (normalized, float(operational_value))
        )

    def sample_validate_sub_cut(
        self, sub: SubproblemResult
    ) -> Tuple[int, Optional[float]]:
        violations = []
        for design, exact_value in self._known_exact_recourse:
            rhs = self._cut_rhs_value(sub, design)
            if rhs is not None:
                violations.append(exact_value - rhs)
        return (
            len(violations),
            max(violations) if violations else None,
        )

    def sample_validate_affine_cut(
        self,
        *,
        intercept: float,
        coefficients,
    ) -> Tuple[int, Optional[float]]:
        from .bdd import coefficient_value

        violations = [
            exact_value
            - (
                float(intercept)
                + coefficient_value(coefficients, design)
            )
            for design, exact_value in self._known_exact_recourse
        ]
        return (
            len(violations),
            max(violations) if violations else None,
        )

    def add_classical_cut(
        self,
        sub: SubproblemResult,
        *,
        iteration: int,
        prefix: str = "classic_conic",
    ) -> bool:
        if (
            sub.obj_value is None
            or sub.cut_expr_data is None
            or sub.rhs_at_reference is None
            or not sub.cut_expr_data
        ):
            return False
        reference_rhs = self._cut_rhs_value(sub, sub.design)
        audit_core = self.pareto_selector.core_point
        core_rhs = (
            None
            if audit_core is None
            else self._cut_rhs_value(sub, audit_core)
        )
        sampled, max_violation = self.sample_validate_sub_cut(sub)
        sample_valid = max_violation is None or max_violation <= 1e-5
        if sample_valid:
            self._add_benders_cut(sub, prefix)
        self.cut_audit_history.append(
            CutAuditRecord(
                iteration=iteration,
                cut_family="classic_conic_dual",
                reference_value=float(sub.obj_value),
                rhs_at_reference=reference_rhs,
                rhs_at_core=core_rhs,
                classical_rhs_at_core=core_rhs,
                core_improvement=0.0 if core_rhs is not None else None,
                reference_error=(
                    None
                    if reference_rhs is None
                    else reference_rhs - float(sub.obj_value)
                ),
                globally_validated=sample_valid,
                sampled_designs=sampled,
                max_sample_violation=max_violation,
                note="QCPDual=1; cut added only with nonempty parameter duals.",
            )
        )
        return sample_valid

    def add_pareto_cut(
        self,
        reference_sub: SubproblemResult,
        *,
        iteration: int,
        deadline: float,
    ) -> bool:
        if not self.use_pareto_cuts:
            return False
        classical_rhs = (
            None
            if self.pareto_selector.core_point is None
            else self._cut_rhs_value(
                reference_sub, self.pareto_selector.core_point
            )
        )
        record = self.pareto_selector.select_and_add(
            reference_sub=reference_sub,
            classical_rhs_at_core=classical_rhs,
            iteration=iteration,
            deadline=deadline,
        )
        if record is None:
            return False
        self.cut_audit_history.append(record)
        return bool(
            record.globally_validated
            and record.core_improvement is not None
            and record.core_improvement
            >= self.pareto_selector.minimum_core_improvement
        )

    def _empty_design(self) -> Design:
        return {
            "x": {node: 0.0 for node in self.N_x},
            "y": {arc: 0.0 for arc in self.AY},
            "z": {
                (hub, c): 0.0
                for hub in self.H
                for c in range(len(config.MU_HUB))
            },
            "Gamma": 0.0,
        }

    def _route_is_available(
        self, y: Mapping[Tuple, float], u, v
    ) -> bool:
        return any(
            all(
                float(y.get(arc, 0.0)) > 0.5
                for arc in self.route_requirements(u, v, option)
            )
            for option in self.route_options.get((u, v), [])
        )

    def _active_aggregate_graph(
        self,
        design: Mapping[str, object],
        *,
        swap_aware: bool,
    ) -> nx.DiGraph:
        graph = nx.DiGraph()
        graph.add_nodes_from(self.HLS)
        y = design["y"]
        for u, v in self.A_hat:
            if float(y.get((u, v), 0.0)) <= 0.5:
                continue
            if swap_aware and not self._route_is_available(y, u, v):
                continue
            graph.add_edge(u, v)
        graph.add_edges_from(self.A_virtual)
        return graph

    def _missing_route_arcs(self, y, u, v) -> set[Tuple]:
        missing = set()
        options = self.route_options.get((u, v), [])
        if not options:
            if (u, v) in self.AY:
                missing.add((u, v))
            return missing
        best_missing = None
        for option in options:
            required = self.route_requirements(u, v, option)
            option_missing = {
                arc
                for arc in required
                if float(y.get(arc, 0.0)) <= 0.5
            }
            if best_missing is None or len(option_missing) < len(best_missing):
                best_missing = option_missing
        missing.update(best_missing or ())
        return missing

    def _add_reachability_cut(
        self,
        *,
        pair,
        missing_arcs,
        prefix: str,
        signatures: set,
    ) -> bool:
        missing = tuple(sorted(set(missing_arcs)))
        signature = (pair, missing)
        if not missing or signature in signatures:
            return False
        signatures.add(signature)
        repair = gp.quicksum(
            self.master_vars["y"][arc]
            for arc in missing
            if arc in self.master_vars["y"]
        )
        self.cut_count += 1
        top_pair_possible = self.master_vars.get("top_pair_possible")
        if (
            top_pair_possible is None
            or pair not in top_pair_possible
        ):
            return False
        self.master.addConstr(
            top_pair_possible[pair] <= repair,
            name=f"{prefix}_{self.cut_count}",
        )
        return True

    def add_topology_connectivity_cuts(
        self, design: Mapping[str, object], limit: int = 30
    ) -> int:
        graph = self._active_aggregate_graph(
            design, swap_aware=False
        )
        y = design["y"]
        added = 0
        pairs = sorted(
            self.proxy_top_pairs,
            key=lambda pair: float(self.inst.h.get(pair, 0.0)),
            reverse=True,
        )
        for i, j in pairs:
            if (
                float(
                    self.master_vars["top_pair_possible"][i, j].X
                )
                <= 0.5
                or float(y.get((i, j), 0.0)) <= 0.5
            ):
                continue
            reachable = {j} | (
                nx.descendants(graph, j) if j in graph else set()
            )
            if i in reachable:
                continue
            missing = {
                (u, v)
                for u, v in self.A_hat
                if u in reachable and v not in reachable
            }
            if self._add_reachability_cut(
                pair=(i, j),
                missing_arcs=missing,
                prefix="topology_return_path",
                signatures=self._topology_cut_signatures,
            ):
                added += 1
                if added >= limit:
                    break
        self.topology_cuts += added
        return added

    def add_swap_reachability_cuts(
        self, design: Mapping[str, object], limit: int = 30
    ) -> int:
        topology_graph = self._active_aggregate_graph(
            design, swap_aware=False
        )
        swap_graph = self._active_aggregate_graph(
            design, swap_aware=True
        )
        y = design["y"]
        added = 0
        pairs = sorted(
            self.proxy_top_pairs,
            key=lambda pair: float(self.inst.h.get(pair, 0.0)),
            reverse=True,
        )
        for i, j in pairs:
            if (
                float(
                    self.master_vars["top_pair_possible"][i, j].X
                )
                <= 0.5
            ):
                continue
            topology_reachable = {j} | (
                nx.descendants(topology_graph, j)
                if j in topology_graph
                else set()
            )
            if (
                float(y.get((i, j), 0.0)) <= 0.5
                or i not in topology_reachable
            ):
                continue
            missing = self._missing_route_arcs(y, i, j)
            if not missing:
                reachable = {j} | (
                    nx.descendants(swap_graph, j)
                    if j in swap_graph
                    else set()
                )
                if i in reachable:
                    continue
                for u, v in self.A_hat:
                    if u in reachable and v not in reachable:
                        missing.update(self._missing_route_arcs(y, u, v))
            if self._add_reachability_cut(
                pair=(i, j),
                missing_arcs=missing,
                prefix="swap_route_reachability",
                signatures=self._swap_cut_signatures,
            ):
                added += 1
                if added >= limit:
                    break
        self.swap_reachability_cuts += added
        return added

    def _record_incumbent(
        self,
        design: Design,
        operational_obj: float,
        total_obj: float,
    ) -> bool:
        if total_obj <= self.best_total_obj + 1e-7:
            return False
        self.best_total_obj = total_obj
        self.best_operational_obj = operational_obj
        self.best_design = design
        self.time_to_best = time.perf_counter() - self._solve_start
        return True

    def _add_exact_design_nogood(
        self,
        design: Mapping[str, object],
    ) -> bool:
        """Exclude one exact integer design without excluding other Gamma."""

        assert self.master is not None
        x = self.master_vars["x"]
        y = self.master_vars["y"]
        z = self.master_vars["z"]
        gamma = self.master_vars["Gamma"]
        mismatch_terms = []
        for key, variable in x.items():
            mismatch_terms.append(
                1.0 - variable
                if float(design["x"].get(key, 0.0)) > 0.5
                else variable
            )
        for key, variable in y.items():
            mismatch_terms.append(
                1.0 - variable
                if float(design["y"].get(key, 0.0)) > 0.5
                else variable
            )
        for key, variable in z.items():
            mismatch_terms.append(
                1.0 - variable
                if float(design["z"].get(key, 0.0)) > 0.5
                else variable
            )

        drone_cost = float(config.COST_DRONE)
        revenue = float(getattr(config, "ORDER_REVENUE_COEF", 0.0))
        if drone_cost <= 1e-9:
            return False
        total_revenue = revenue * sum(
            max(0.0, float(self.inst.h.get(pair, 0.0)))
            for pair in self.A_HH
        )
        gamma_max = int(math.ceil(total_revenue / drone_cost))
        gamma_value = int(round(float(design.get("Gamma", 0.0))))
        big_m = float(gamma_max + 1)
        cut_index = self.exact_design_nogood_cuts + 1
        gamma_lower = self.master.addVar(
            vtype=GRB.BINARY,
            name=f"nogood_gamma_lower_{cut_index}",
        )
        gamma_higher = self.master.addVar(
            vtype=GRB.BINARY,
            name=f"nogood_gamma_higher_{cut_index}",
        )
        self.master.addConstr(
            gamma
            <= gamma_value - 1 + big_m * (1.0 - gamma_lower),
            name=f"nogood_gamma_lower_link_{cut_index}",
        )
        self.master.addConstr(
            gamma
            >= gamma_value + 1 - big_m * (1.0 - gamma_higher),
            name=f"nogood_gamma_higher_link_{cut_index}",
        )
        self.master.addConstr(
            gamma_lower + gamma_higher <= 1.0,
            name=f"nogood_gamma_side_{cut_index}",
        )
        self.master.addConstr(
            gp.quicksum(mismatch_terms)
            + gamma_lower
            + gamma_higher
            >= 1.0,
            name=f"exact_design_nogood_{cut_index}",
        )
        self.master.update()
        self.exact_design_nogood_cuts += 1
        return True

    def _evaluate_initial_designs(
        self,
        designs,
        *,
        deadline: float,
    ) -> None:
        best_start = None
        best_start_sub = None
        best_start_obj = -math.inf
        for index, design in enumerate(designs, 1):
            if time.perf_counter() >= deadline:
                break
            signature = design_signature(design)
            if signature in self._evaluated_signatures:
                continue
            self._evaluated_signatures.add(signature)
            eval_start = time.perf_counter()
            sub = self.solve_subproblem_with_deadline(
                design,
                label=f"initial_{index}",
                deadline=deadline,
            )
            eval_wall_time = time.perf_counter() - eval_start
            fixed_cost = self._fixed_cost_value(design)
            total_obj = (
                None
                if sub.obj_value is None
                else float(sub.obj_value) - fixed_cost
            )
            status_name = _status_label(int(sub.status))
            self.initial_design_history.append(
                {
                    "index": index,
                    "source": "lagrangian" if self.use_lr else "empty",
                    "status": int(sub.status),
                    "status_name": status_name,
                    "sub_obj": sub.obj_value,
                    "fixed_cost": fixed_cost,
                    "total_obj": total_obj,
                    "Gamma": float(design.get("Gamma", 0.0)),
                    "candidate_eval_wall_time": eval_wall_time,
                    "failure_reason": (
                        None
                        if total_obj is not None
                        else f"exact_socp_{status_name}"
                    ),
                    **_candidate_log_fields(design),
                }
            )
            if total_obj is not None:
                self.register_known_exact(design, float(sub.obj_value))
                self._local_branching_start_pool.append(
                    (float(total_obj), dict(design))
                )
                self._record_incumbent(
                    design, float(sub.obj_value), total_obj
                )
                if total_obj > best_start_obj + 1e-7:
                    best_start_obj = total_obj
                    best_start = design
                    best_start_sub = sub
            self.add_classical_cut(
                sub, iteration=0, prefix="initial_classic_conic"
            )
        if best_start is not None:
            self._apply_master_start(best_start)
            self._initialize_or_update_core(best_start)
            if best_start_sub is not None:
                self.add_pareto_cut(
                    best_start_sub,
                    iteration=0,
                    deadline=deadline,
                )

    @staticmethod
    def _is_integral_design(
        design: Mapping[str, object], tolerance: float = 1e-6
    ) -> bool:
        for family in ("x", "y", "z"):
            for value in design[family].values():
                if abs(float(value) - round(float(value))) > tolerance:
                    return False
        return (
            abs(float(design["Gamma"]) - round(float(design["Gamma"])))
            <= tolerance
        )

    def _run_bdd_root_strengthening(
        self,
        *,
        deadline: float,
    ) -> None:
        if (
            self.bdd_root_rounds <= 0
            or (
                not self.use_bdd_strengthened
                and not self.use_bdd_lagrangian
            )
        ):
            return
        for root_round in range(1, self.bdd_root_rounds + 1):
            remaining = deadline - time.perf_counter()
            if remaining <= 0.10:
                break
            reference = solve_fractional_master(
                self, time_limit=min(5.0, remaining)
            )
            if reference is None:
                break
            sub = self.solve_subproblem_with_deadline(
                reference,
                label=f"bdd_root_reference_{root_round}",
                deadline=deadline,
            )
            if sub.obj_value is None:
                break
            self.add_classical_cut(
                sub,
                iteration=-root_round,
                prefix="bdd_root_classic",
            )
            self.add_pareto_cut(
                sub,
                iteration=-root_round,
                deadline=deadline,
            )
            if self.use_bdd_strengthened and root_round >= 2:
                remaining = deadline - time.perf_counter()
                if remaining <= 0.10:
                    break
                record = self.bdd_generator.strengthened_cut(
                    classical_sub=sub,
                    reference_design=reference,
                    iteration=-root_round,
                    time_limit=min(self.bdd_inner_time_limit, remaining),
                )
                if record is not None:
                    self.cut_audit_history.append(record)
            if (
                self.use_bdd_lagrangian
                and root_round == self.bdd_root_rounds
            ):
                record = self.bdd_generator.lagrangian_cut(
                    classical_sub=sub,
                    reference_design=reference,
                    iteration=-root_round,
                    deadline=deadline,
                    per_inner_time_limit=self.bdd_inner_time_limit,
                )
                if record is not None:
                    self.cut_audit_history.append(record)

        # Local-copy solutions are heuristic candidates only.  Accept no
        # incumbent here; integral candidates are evaluated by the exact SOCP.
        for index, candidate in enumerate(
            self.bdd_generator.local_candidates, 1
        ):
            if (
                time.perf_counter() >= deadline
                or not self._is_integral_design(candidate)
            ):
                continue
            signature = design_signature(candidate)
            if signature in self._evaluated_signatures:
                continue
            self._evaluated_signatures.add(signature)
            sub = self.solve_subproblem_with_deadline(
                candidate,
                label=f"bdd_local_candidate_{index}",
                deadline=deadline,
            )
            if sub.obj_value is None:
                continue
            self.register_known_exact(candidate, float(sub.obj_value))
            total = float(sub.obj_value) - self._fixed_cost_value(candidate)
            self._record_incumbent(candidate, float(sub.obj_value), total)
            self.initial_design_history.append(
                {
                    "index": index,
                    "source": "bdd_local_copy_exact_validated",
                    "status": int(sub.status),
                    "sub_obj": float(sub.obj_value),
                    "fixed_cost": self._fixed_cost_value(candidate),
                    "total_obj": total,
                    "Gamma": float(candidate["Gamma"]),
                }
            )

    def _solve_global_linearized_upper_bound(
        self,
        *,
        deadline: float,
        total_time_limit: float,
    ) -> None:
        if not self.use_global_linearized_bound:
            return
        remaining = deadline - time.perf_counter()
        budget = min(
            remaining,
            self.global_bound_time_limit,
            self.global_bound_time_fraction * float(total_time_limit),
        )
        if budget <= 0.05:
            return

        relaxation = MLNDModel(self.inst)
        relaxation.build()
        model = relaxation.model
        model.ModelName = "MLND_20260623_Global_Linearized_Upper_Bound"
        model.Params.OutputFlag = 0
        model.Params.Threads = 1
        model.Params.Seed = int(getattr(config, "BENDERS_RANDOM_SEED", 0))
        model.Params.TimeLimit = budget
        model.Params.MIPGap = float(
            getattr(config, "BDD_GLOBAL_LINEAR_MIP_GAP", 0.002)
        )
        q_metadata = remove_all_quadratic_constraints(relaxation)
        self.global_linearized_qconstrs_before = q_metadata[
            "qconstrs_before"
        ]
        self.global_linearized_qconstrs_after = q_metadata["qconstrs_after"]
        self.global_linearized_queue_tangents = add_queue_supporting_lines(
            relaxation, prefix="global_queue_tangent"
        )
        start = time.perf_counter()
        model.optimize()
        self.global_linearized_wall_time = time.perf_counter() - start
        self.global_linearized_status = int(model.Status)
        bound = float(model.ObjBound)
        if not math.isfinite(bound):
            return
        self.global_linearized_bound = bound
        tolerance = 1e-7 * max(1.0, abs(bound))
        self.master.addConstr(
            self.master.getObjective() <= bound + tolerance,
            name="global_linearized_network_upper_bound",
        )

    def _run_initial_local_branching(
        self,
        *,
        deadline: float,
        per_call_time_limit: float,
    ) -> None:
        if (
            not self.use_local_branching
            or self.best_design is None
            or time.perf_counter() >= deadline
        ):
            return
        pool = sorted(
            self._local_branching_start_pool,
            key=lambda item: item[0],
            reverse=True,
        )
        centers: list[Tuple[float, Design]] = []
        seen = set()

        def candidate_source(design: Mapping[str, object]) -> str:
            metadata = design.get("_lr_meta", {})
            if isinstance(metadata, Mapping):
                return str(metadata.get("source") or "")
            return ""

        def add_center(objective: float, design: Design) -> bool:
            signature = design_signature(design)
            if signature in seen:
                return False
            seen.add(signature)
            centers.append((float(objective), design))
            return True

        if self.local_branching_start_pool_size > 1:
            for source in (
                "lr_restricted_network_milp",
                "lr_fixed_hub_network_milp",
                "lr_recovery",
                "",
            ):
                for objective, design in pool:
                    if candidate_source(design) != source:
                        continue
                    if add_center(float(objective), design):
                        break
                    if len(centers) >= self.local_branching_start_pool_size:
                        break
                if len(centers) >= self.local_branching_start_pool_size:
                    break
        for objective, design in pool:
            if len(centers) >= self.local_branching_start_pool_size:
                break
            add_center(float(objective), design)
        if not centers:
            centers = [(float(self.best_total_obj), self.best_design)]

        for center_index, (center_objective, center) in enumerate(centers):
            if (
                self.local_branching.calls >= self.local_branching.max_calls
                or time.perf_counter() >= deadline
            ):
                break
            self.local_branching.reset_radius()
            center_start_calls = self.local_branching.calls
            while (
                self.local_branching.calls
                < self.local_branching.max_calls
                and time.perf_counter() < deadline
            ):
                if (
                    self.local_branching_max_calls_per_center > 0
                    and self.local_branching.calls - center_start_calls
                    >= self.local_branching_max_calls_per_center
                ):
                    break
                candidate, candidate_obj = self.local_branching.search(
                    center=center,
                    center_objective=center_objective,
                    deadline=deadline,
                    per_call_time_limit=per_call_time_limit,
                )
                if (
                    candidate is not None
                    and candidate_obj is not None
                    and candidate_obj > center_objective + 1e-7
                ):
                    if candidate_obj > self.best_total_obj + 1e-7:
                        self.best_design = candidate
                        self.best_total_obj = candidate_obj
                        self.time_to_best = (
                            time.perf_counter() - self._solve_start
                        )
                    center = candidate
                    center_objective = candidate_obj
                    continue
                if (
                    self.local_branching.radius
                    >= self.local_branching.max_radius
                    and self.local_branching.gamma_radius
                    >= self.local_branching.max_gamma_radius
                ):
                    break
            if center_index + 1 < len(centers):
                remaining = deadline - time.perf_counter()
                if remaining <= per_call_time_limit + 0.05:
                    break

    def _signature_known_or_evaluated(self, signature: Tuple[int, ...]) -> bool:
        if signature in self._evaluated_signatures:
            return True
        return any(
            design_signature(existing) == signature
            for existing, _ in self._known_exact_recourse
        )

    def _run_gamma_sweep(self, *, deadline: float) -> None:
        if (
            self.gamma_sweep_max_calls <= 0
            or self.best_design is None
            or time.perf_counter() >= deadline
        ):
            return
        base_design = self.best_design
        center_gamma = int(round(float(base_design.get("Gamma", 0.0))))
        base_obj = float(self.best_total_obj)
        evaluated = 0
        source = "lr_bd_gamma_sweep" if self.use_lr else "bd_gamma_sweep"
        for gamma in self._gamma_sweep_values(
            center_gamma,
            radius=self.gamma_sweep_radius,
            step=self.gamma_sweep_step,
        ):
            if evaluated >= self.gamma_sweep_max_calls:
                break
            if time.perf_counter() >= deadline:
                break
            candidate = self._clone_design_with_gamma(
                base_design,
                gamma,
                source=source,
                center_gamma=center_gamma,
            )
            signature = design_signature(candidate)
            if self._signature_known_or_evaluated(signature):
                continue
            self._evaluated_signatures.add(signature)
            evaluated += 1
            eval_start = time.perf_counter()
            sub = self.solve_subproblem_with_deadline(
                candidate,
                label=f"{source}_{evaluated}",
                deadline=deadline,
            )
            eval_wall_time = time.perf_counter() - eval_start
            fixed_cost = self._fixed_cost_value(candidate)
            total_obj = (
                None
                if sub.obj_value is None
                else float(sub.obj_value) - fixed_cost
            )
            status_name = _status_label(int(sub.status))
            cut_added = False
            pareto_added = False
            improved = False
            if total_obj is not None:
                self.register_known_exact(candidate, float(sub.obj_value))
                cut_added = self.add_classical_cut(
                    sub,
                    iteration=self.current_iteration,
                    prefix="gamma_sweep_classic",
                )
                improved = self._record_incumbent(
                    candidate, float(sub.obj_value), float(total_obj)
                )
                if improved:
                    self._apply_master_start(candidate)
                    self._initialize_or_update_core(candidate)
                    pareto_added = self.add_pareto_cut(
                        sub,
                        iteration=self.current_iteration,
                        deadline=deadline,
                    )
            self.gamma_sweep_history.append(
                {
                    "call": evaluated,
                    "source": source,
                    "center_gamma": center_gamma,
                    "Gamma": float(gamma),
                    "status": int(sub.status),
                    "status_name": status_name,
                    "sub_obj": sub.obj_value,
                    "fixed_cost": fixed_cost,
                    "total_obj": total_obj,
                    "base_obj": base_obj,
                    "improvement_vs_base": (
                        None
                        if total_obj is None
                        else float(total_obj) - base_obj
                    ),
                    "improved_incumbent": bool(improved),
                    "classic_cut_added": int(cut_added),
                    "pareto_cut_added": int(pareto_added),
                    "candidate_eval_wall_time": eval_wall_time,
                    "failure_reason": (
                        None
                        if total_obj is not None
                        else f"exact_socp_{status_name}"
                    ),
                }
            )

    def solve(
        self,
        *,
        time_limit: float,
        lr_time_limit: float,
        max_iterations: int = 10000,
        tolerance: float = 1e-4,
        local_branching_time_limit: float = 5.0,
    ) -> SolveResult20260623:
        self._solve_start = time.perf_counter()
        deadline = self._solve_start + max(0.0, float(time_limit))
        self.build_master()
        assert self.master is not None
        self.apply_master_solver_config()
        invariant = assert_pure_master(self.master)
        if self.use_pareto_cuts:
            self.pareto_selector.initialize_core()

        lr_result = None
        lr_bound = math.inf
        if self.use_lr and time.perf_counter() < deadline:
            lr_budget = min(
                max(0.0, float(lr_time_limit)),
                deadline - time.perf_counter(),
            )
            lr_result = self.lr_engine.generate(time_limit=lr_budget)
            lr_bound = lr_result.best_dual_bound
            initial_designs = lr_result.designs
        else:
            initial_designs = [self._empty_design()]
        self._evaluate_initial_designs(initial_designs, deadline=deadline)
        self._solve_global_linearized_upper_bound(
            deadline=deadline,
            total_time_limit=time_limit,
        )
        self._run_bdd_root_strengthening(deadline=deadline)
        recovery_deadline = deadline
        if (
            self.gamma_sweep_max_calls > 0
            and self.gamma_sweep_time_reserve > 0.0
        ):
            recovery_deadline = max(
                time.perf_counter(),
                deadline - self.gamma_sweep_time_reserve,
            )
        self._run_initial_local_branching(
            deadline=recovery_deadline,
            per_call_time_limit=local_branching_time_limit,
        )
        self._run_gamma_sweep(deadline=deadline)

        lower_bound = (
            float(self.best_total_obj)
            if self.best_design is not None
            else -math.inf
        )
        upper_bound = min(
            lr_bound,
            (
                math.inf
                if self.global_linearized_bound is None
                else self.global_linearized_bound
            ),
        )

        for iteration in range(1, max(1, int(max_iterations)) + 1):
            self.current_iteration = iteration
            remaining = deadline - time.perf_counter()
            if remaining <= 0.01:
                break
            self.master.Params.TimeLimit = max(0.01, remaining)
            self.master.optimize()
            if (
                self.master.Status
                not in {GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL}
                or self.master.SolCount == 0
            ):
                break

            master_bound = float(self.master.ObjBound)
            if math.isfinite(master_bound):
                upper_bound = min(upper_bound, master_bound)
            design = self._extract_master_design()
            signature = design_signature(design)
            topology_added = self.add_topology_connectivity_cuts(design)
            swap_added = self.add_swap_reachability_cuts(design)
            if topology_added or swap_added:
                self.history.append(
                    {
                        "iteration": iteration,
                        "LB": (
                            None
                            if not math.isfinite(lower_bound)
                            else lower_bound
                        ),
                        "UB": (
                            None
                            if not math.isfinite(upper_bound)
                            else upper_bound
                        ),
                        "gap": None,
                        "master_bound": master_bound,
                        "topology_cuts": topology_added,
                        "swap_reachability_cuts": swap_added,
                        "classic_cut_added": 0,
                        "pareto_cut_added": 0,
                        "cuts": self.cut_count,
                        "time": time.perf_counter() - self._solve_start,
                    }
                )
                continue
            exact = self.solve_subproblem_with_deadline(
                design,
                label=f"bd_exact_{iteration}",
                deadline=deadline,
            )
            fixed_cost = self._fixed_cost_value(design)
            exact_total = (
                None
                if exact.obj_value is None
                else float(exact.obj_value) - fixed_cost
            )
            improved = False
            if exact_total is not None:
                self.register_known_exact(design, float(exact.obj_value))
                improved = self._record_incumbent(
                    design, float(exact.obj_value), exact_total
                )
                lower_bound = max(lower_bound, exact_total)
            cut_added = self.add_classical_cut(
                exact, iteration=iteration
            )
            pareto_added = self.add_pareto_cut(
                exact,
                iteration=iteration,
                deadline=deadline,
            )

            if self.core_point is None:
                self._initialize_or_update_core(design)
            else:
                self._initialize_or_update_core(design)

            if (
                improved
                and self.use_local_branching
                and self.best_design is not None
                and time.perf_counter() < deadline
            ):
                candidate, candidate_obj = self.local_branching.search(
                    center=self.best_design,
                    center_objective=float(self.best_total_obj),
                    deadline=deadline,
                    per_call_time_limit=local_branching_time_limit,
                )
                if (
                    candidate is not None
                    and candidate_obj is not None
                    and candidate_obj > self.best_total_obj + 1e-7
                ):
                    # The local branching module already solved exact recourse.
                    self.best_design = candidate
                    self.best_total_obj = candidate_obj
                    self.time_to_best = (
                        time.perf_counter() - self._solve_start
                    )
                    lower_bound = max(lower_bound, candidate_obj)

            if math.isfinite(lower_bound):
                upper_bound = max(upper_bound, lower_bound)
            gap = (
                math.inf
                if not math.isfinite(lower_bound)
                or not math.isfinite(upper_bound)
                else (upper_bound - lower_bound)
                / max(1.0, abs(lower_bound))
            )
            self.history.append(
                {
                    "iteration": iteration,
                    "LB": None if not math.isfinite(lower_bound) else lower_bound,
                    "UB": None if not math.isfinite(upper_bound) else upper_bound,
                    "gap": gap,
                    "master_bound": master_bound,
                    "exact_obj": exact.obj_value,
                    "exact_total_obj": exact_total,
                    "classic_cut_added": int(cut_added),
                    "pareto_cut_added": int(pareto_added),
                    "topology_cuts": topology_added,
                    "swap_reachability_cuts": swap_added,
                    "cuts": self.cut_count,
                    "time": time.perf_counter() - self._solve_start,
                }
            )
            if self.verbose:
                print(
                    f"20260623 BD {iteration}: LB={lower_bound:.6f}, "
                    f"UB={upper_bound:.6f}, gap={gap:.4%}, "
                    f"cut={int(cut_added)}",
                    flush=True,
                )
            if (
                (cut_added or pareto_added)
                and math.isfinite(gap)
                and gap <= float(tolerance)
            ):
                break
            if (
                signature in self._evaluated_signatures
                and not cut_added
                and not pareto_added
            ):
                # Degenerate SOCPs can return no usable dual cut.  Exclude
                # only this exact integer (x,y,z,Gamma) point and continue.
                # The two Gamma-side binaries preserve all other fleet sizes.
                if not self._add_exact_design_nogood(design):
                    break
                continue
            self._evaluated_signatures.add(signature)

        wall_time = time.perf_counter() - self._solve_start
        success = self.best_design is not None
        objective = float(self.best_total_obj) if success else None
        bound = (
            None if not math.isfinite(upper_bound) else float(upper_bound)
        )
        if objective is not None and bound is not None:
            bound = max(bound, objective)
        gap = (
            None
            if objective is None or bound is None
            else (bound - objective) / max(1.0, abs(objective))
        )
        timing_rows = [
            *self.lr_engine.profiler.rows(),
            *self.profiler.rows(),
        ]
        cpp_decision = (
            None
            if lr_result is None
            else self.lr_engine.profiler.cpp_acceleration_decision(
                total_lr_time=lr_result.wall_time
            )
        )
        return SolveResult20260623(
            method="lr_bd" if self.use_lr else "bd",
            success=success,
            objective=objective,
            bound=bound,
            gap=gap,
            wall_time=wall_time,
            time_to_best=self.time_to_best,
            best_design=self.best_design,
            history=list(self.history),
            initial_design_history=list(self.initial_design_history),
            cut_history=list(self.cut_audit_history),
            local_branching_history=list(self.local_branching.history),
            gamma_sweep_history=list(self.gamma_sweep_history),
            lr_result=lr_result,
            metadata={
                **invariant,
                **self.master_proxy_metadata,
                "lr_bound": (
                    None if not math.isfinite(lr_bound) else lr_bound
                ),
                "raw_lr_bound": (
                    None if not math.isfinite(lr_bound) else lr_bound
                ),
                "lr_wall_time": (
                    0.0 if lr_result is None else lr_result.wall_time
                ),
                "iterations": len(self.history),
                "cuts": self.cut_count,
                "timing_rows": timing_rows,
                "cpp_acceleration_decision": cpp_decision,
                "initial_design_count": len(initial_designs),
                "validated_initial_design_count": len(
                    self.initial_design_history
                ),
                "failed_initial_design_count": sum(
                    row.get("total_obj") is None
                    for row in self.initial_design_history
                ),
                "best_initial_design_obj": (
                    None
                    if not self.initial_design_history
                    else max(
                        (
                            float(row["total_obj"])
                            for row in self.initial_design_history
                            if row.get("total_obj") is not None
                        ),
                        default=None,
                    )
                ),
                "best_initial_design_validation_obj": (
                    None
                    if not self.initial_design_history
                    else max(
                        (
                            float(row["total_obj"])
                            for row in self.initial_design_history
                            if row.get("total_obj") is not None
                        ),
                        default=None,
                    )
                ),
                "lr_candidate_count": (
                    0
                    if lr_result is None
                    else len(lr_result.candidate_history)
                ),
                "exact_socp_solver_time_is_wall_proxy": True,
                "cut_strategy": (
                    "classic_conic_dual+pareto_mw_conic"
                    if self.use_pareto_cuts
                    else "classic_conic_dual"
                ),
                "pareto_cut_implemented": True,
                "pareto_cut_enabled": self.use_pareto_cuts,
                "pareto_core_metadata": self.pareto_selector.core_metadata,
                "bdd_cut_implemented": True,
                "bdd_strengthened_enabled": self.use_bdd_strengthened,
                "bdd_lagrangian_enabled": self.use_bdd_lagrangian,
                "bdd_root_rounds": self.bdd_root_rounds,
                "local_branching_enabled": self.use_local_branching,
                "local_branching_distance_scope": (
                    self.local_branching.distance_scope
                ),
                "local_branching_start_pool_size": (
                    self.local_branching_start_pool_size
                ),
                "local_branching_max_calls_per_center": (
                    self.local_branching_max_calls_per_center
                ),
                "local_branching_arc_radius": (
                    self.local_branching.initial_arc_radius
                ),
                "local_branching_max_arc_radius": (
                    self.local_branching.max_arc_radius
                ),
                "local_branching_solution_pool_size": (
                    self.local_branching.solution_pool_size
                ),
                "local_branching_benders_envelope_cuts": (
                    self.local_branching.benders_envelope_cuts
                ),
                "local_branching_benders_envelope_weight": (
                    self.local_branching.benders_envelope_weight
                ),
                "local_branching_benders_envelope_cutoff": (
                    self.local_branching.benders_envelope_cutoff
                ),
                "local_branching_max_removals": (
                    self.local_branching.max_removals
                ),
                "local_branching_protect_top_pairs": (
                    self.local_branching.protect_top_pairs
                ),
                "gamma_sweep_enabled": self.gamma_sweep_max_calls > 0,
                "gamma_sweep_max_calls": self.gamma_sweep_max_calls,
                "gamma_sweep_radius": self.gamma_sweep_radius,
                "gamma_sweep_step": self.gamma_sweep_step,
                "gamma_sweep_time_reserve": self.gamma_sweep_time_reserve,
                "gamma_sweep_evaluated": len(self.gamma_sweep_history),
                "gamma_sweep_best_obj": objective,
                "gamma_sweep_improved_count": sum(
                    1
                    for row in self.gamma_sweep_history
                    if row.get("improved_incumbent")
                ),
                "benders_cut_pool_size": len(self._benders_cut_pool),
                "topology_cuts": self.topology_cuts,
                "swap_reachability_cuts": self.swap_reachability_cuts,
                "exact_design_nogood_cuts": (
                    self.exact_design_nogood_cuts
                ),
                "global_linearized_bound": self.global_linearized_bound,
                "global_linearized_status": self.global_linearized_status,
                "global_linearized_wall_time": (
                    self.global_linearized_wall_time
                ),
                "global_linearized_qconstrs_before": (
                    self.global_linearized_qconstrs_before
                ),
                "global_linearized_qconstrs_after": (
                    self.global_linearized_qconstrs_after
                ),
                "global_linearized_queue_tangents": (
                    self.global_linearized_queue_tangents
                ),
                **self.local_branching.linearized_metadata,
                **({} if lr_result is None else lr_result.metadata),
            },
        )
