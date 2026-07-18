"""Adaptive local branching over strategic variables only."""

from __future__ import annotations

import time
from typing import Mapping, Optional, Tuple

import gurobipy as gp
from gurobipy import GRB

from src import config
from src.model.gurobi_model import MLNDModel

from .linearized import (
    add_queue_supporting_lines,
    remove_all_quadratic_constraints,
)
from .types import Design, LocalBranchingRecord, design_signature


class AdaptiveLocalBranching20260623:
    """Search Hamming neighborhoods and certify candidates by exact SOCP."""

    def __init__(
        self,
        owner,
        *,
        initial_radius: int = 6,
        min_radius: int = 2,
        max_radius: int = 160,
        gamma_radius: int = 8,
        max_gamma_radius: int = 240,
        max_calls: int = 32,
        distance_scope: str = "all",
        arc_radius: int = 64,
        max_arc_radius: int = 260,
        solution_pool_size: int = 1,
        benders_envelope_cuts: int = 0,
        benders_envelope_weight: float = 1.0,
        benders_envelope_cutoff: bool = False,
        max_removals: Optional[int] = None,
        protect_top_pairs: int = 0,
    ) -> None:
        self.owner = owner
        self.initial_radius = max(1, int(initial_radius))
        self.radius = self.initial_radius
        self.min_radius = max(1, int(min_radius))
        self.max_radius = max(self.radius, int(max_radius))
        self.initial_gamma_radius = max(0, int(gamma_radius))
        self.gamma_radius = self.initial_gamma_radius
        self.max_gamma_radius = max(
            self.gamma_radius, int(max_gamma_radius)
        )
        if distance_scope not in {"all", "facility", "layered"}:
            raise ValueError(
                "distance_scope must be 'all', 'facility', or 'layered'"
            )
        self.distance_scope = distance_scope
        self.initial_arc_radius = max(1, int(arc_radius))
        self.arc_radius = self.initial_arc_radius
        self.max_arc_radius = max(self.arc_radius, int(max_arc_radius))
        self.solution_pool_size = max(1, int(solution_pool_size))
        self.benders_envelope_cuts = max(0, int(benders_envelope_cuts))
        self.benders_envelope_weight = min(
            1.0, max(0.0, float(benders_envelope_weight))
        )
        self.benders_envelope_cutoff = bool(benders_envelope_cutoff)
        self.max_removals = (
            None if max_removals is None or int(max_removals) < 0 else int(max_removals)
        )
        self.protect_top_pairs = max(0, int(protect_top_pairs))
        self.max_calls = max(0, int(max_calls))
        self.calls = 0
        self.seen_centers = set()
        self.seen_candidates = set()
        self.history: list[LocalBranchingRecord] = []
        self.linearized_metadata = {
            "local_branching_qconstrs_before": None,
            "local_branching_qconstrs_after": None,
            "local_branching_queue_tangents": 0,
            "local_branching_benders_envelope_enabled": False,
            "local_branching_benders_envelope_cuts": 0,
            "local_branching_protect_top_pairs": self.protect_top_pairs,
        }

    def reset_radius(self) -> None:
        self.radius = self.initial_radius
        self.gamma_radius = self.initial_gamma_radius
        self.arc_radius = self.initial_arc_radius

    def _distance_expressions(
        self,
        center: Mapping[str, object],
        *,
        x,
        y,
        z,
    ):
        owner = self.owner
        capacities = range(len(config.MU_HUB))
        facility_terms = []
        arc_terms = []
        removal_terms = []
        for node in owner.N_x:
            if float(center["x"].get(node, 0.0)) > 0.5:
                term = 1 - x[node]
                removal_terms.append(term)
            else:
                term = x[node]
            facility_terms.append(term)
        for arc in owner.AY:
            if float(center["y"].get(arc, 0.0)) > 0.5:
                term = 1 - y[arc]
                removal_terms.append(term)
            else:
                term = y[arc]
            arc_terms.append(term)
        for hub in owner.H:
            for c in capacities:
                if float(center["z"].get((hub, c), 0.0)) > 0.5:
                    term = 1 - z[hub, c]
                    removal_terms.append(term)
                else:
                    term = z[hub, c]
                facility_terms.append(term)
        facility_distance = gp.quicksum(facility_terms)
        arc_distance = gp.quicksum(arc_terms)
        removal_distance = gp.quicksum(removal_terms)
        if self.distance_scope == "facility":
            total_distance = facility_distance
        elif self.distance_scope == "layered":
            total_distance = facility_distance + arc_distance
        else:
            total_distance = facility_distance + arc_distance
        return facility_distance, arc_distance, total_distance, removal_distance

    def _fixed_cost_expression(self, *, x, y, z, gamma):
        owner = self.owner
        capacities = range(len(config.MU_HUB))
        fixed_cost = gp.quicksum(
            float(owner.inst.f.get(base, config.COST_BASE)) * x[base]
            for base in owner.L
        )
        fixed_cost += gp.quicksum(
            float(getattr(config, "COST_HUB_FIXED", 0.0)) * x[hub]
            for hub in owner.H
        )
        fixed_cost += gp.quicksum(
            float(owner.inst.f.get(swap, config.COST_SWAPPING)) * x[swap]
            for swap in owner.E
        )
        fixed_cost += gp.quicksum(
            float(config.COST_HUB[c]) * z[hub, c]
            for hub in owner.H
            for c in capacities
        )
        fixed_cost += gp.quicksum(owner.arc_cost[arc] * y[arc] for arc in owner.AY)
        fixed_cost += float(config.COST_DRONE) * gamma
        return fixed_cost

    def _apply_benders_envelope_objective(
        self,
        *,
        model,
        x,
        y,
        z,
        gamma,
        center_objective: float,
    ) -> Tuple[int, bool]:
        if self.benders_envelope_cuts <= 0:
            return 0, False
        cut_pool = list(getattr(self.owner, "_benders_cut_pool", ()))
        if not cut_pool:
            return 0, False
        selected = [
            sub
            for sub in cut_pool[-self.benders_envelope_cuts :]
            if sub.obj_value is not None
            and sub.cut_expr_data is not None
            and sub.rhs_at_reference is not None
            and sub.cut_expr_data
        ]
        if not selected:
            return 0, False

        eta_envelope = model.addVar(lb=0.0, name=f"lb_benders_eta_{self.calls}")
        local_design = {"x": x, "y": y, "z": z, "Gamma": gamma}
        for cut_index, sub in enumerate(selected, 1):
            cut_rhs = gp.LinExpr(float(sub.obj_value))
            for (pi, rhs_fun), rhs_ref in zip(
                sub.cut_expr_data, sub.rhs_at_reference
            ):
                cut_rhs += float(pi) * (rhs_fun(local_design) - float(rhs_ref))
            model.addConstr(
                eta_envelope <= cut_rhs,
                name=f"lb_benders_envelope_{self.calls}_{cut_index}",
            )

        envelope_total = eta_envelope - self._fixed_cost_expression(
            x=x, y=y, z=z, gamma=gamma
        )
        if self.benders_envelope_cutoff:
            model.addConstr(
                envelope_total >= float(center_objective) + 1e-6,
                name=f"lb_benders_envelope_cutoff_{self.calls}",
            )

        original_objective = model.getObjective()
        weight = self.benders_envelope_weight
        model.setObjective(
            weight * envelope_total + (1.0 - weight) * original_objective,
            GRB.MAXIMIZE,
        )
        model.update()
        self.linearized_metadata.update(
            {
                "local_branching_benders_envelope_enabled": True,
                "local_branching_benders_envelope_cuts": len(selected),
            }
        )
        return len(selected), True

    def _center_route_requirements(
        self,
        center: Mapping[str, object],
        pair: Tuple[object, object],
    ) -> Tuple[Tuple[object, object], ...]:
        owner = self.owner
        y = center.get("y", {})
        for option in owner.route_options.get(pair, []):
            required = owner.route_requirements(pair[0], pair[1], option)
            if all(float(y.get(arc, 0.0)) > 0.5 for arc in required):
                return tuple(required)
        return ()

    def _add_service_preservation_constraints(
        self,
        *,
        model,
        center: Mapping[str, object],
        y,
    ) -> int:
        if self.protect_top_pairs <= 0:
            return 0
        owner = self.owner
        protected = 0
        for pair in sorted(
            owner.A_HH,
            key=lambda item: float(owner.inst.h.get(item, 0.0)),
            reverse=True,
        ):
            required = self._center_route_requirements(center, pair)
            if not required:
                continue
            if any(arc not in y for arc in required):
                continue
            for arc in required:
                model.addConstr(
                    y[arc] >= 1.0,
                    name=(
                        f"lb_{self.calls}_protect_route_"
                        f"{protected}_{arc[0]}_{arc[1]}"
                    ),
                )
            protected += 1
            if protected >= self.protect_top_pairs:
                break
        return protected

    def search(
        self,
        *,
        center: Design,
        center_objective: float,
        deadline: float,
        per_call_time_limit: float,
    ) -> Tuple[Optional[Design], Optional[float]]:
        if self.calls >= self.max_calls or time.perf_counter() >= deadline:
            return None, None
        center_signature = design_signature(center)
        center_key = (
            center_signature,
            self.distance_scope,
            self.radius,
            self.arc_radius,
            self.gamma_radius,
        )
        if center_key in self.seen_centers:
            return None, None
        self.seen_centers.add(center_key)
        self.calls += 1

        owner = self.owner
        neighborhood = MLNDModel(owner.inst)
        neighborhood.build()
        model = neighborhood.model
        model.ModelName = f"MLND_20260623_LocalBranch_{self.calls}"
        model.Params.OutputFlag = 0
        model.Params.Threads = 1
        model.Params.Seed = int(getattr(config, "BENDERS_RANDOM_SEED", 0))
        model.Params.MIPGap = float(
            getattr(config, "BDD_LOCAL_BRANCH_MIP_GAP", 0.01)
        )
        if self.solution_pool_size > 1:
            model.Params.PoolSolutions = self.solution_pool_size
            model.Params.PoolSearchMode = 2
        q_metadata = remove_all_quadratic_constraints(neighborhood)
        self.linearized_metadata.update(
            {
                "local_branching_qconstrs_before": q_metadata[
                    "qconstrs_before"
                ],
                "local_branching_qconstrs_after": q_metadata[
                    "qconstrs_after"
                ],
            }
        )
        queue_tangents = add_queue_supporting_lines(
            neighborhood, prefix=f"local_branch_queue_{self.calls}"
        )
        self.linearized_metadata[
            "local_branching_queue_tangents"
        ] = queue_tangents

        x = neighborhood.vars["x"]
        y = neighborhood.vars["y"]
        z = neighborhood.vars["z"]
        gamma = neighborhood.vars["Gamma"]
        (
            facility_distance,
            arc_distance,
            total_distance,
            removal_distance,
        ) = self._distance_expressions(
            center, x=x, y=y, z=z
        )
        gamma_center = int(round(float(center.get("Gamma", 0.0))))
        if self.distance_scope == "layered":
            model.addConstr(
                total_distance >= 1.0,
                name=f"lb_{self.calls}_distance_lb",
            )
            model.addConstr(
                facility_distance <= float(self.radius),
                name=f"lb_{self.calls}_facility_distance_ub",
            )
            model.addConstr(
                arc_distance <= float(self.arc_radius),
                name=f"lb_{self.calls}_arc_distance_ub",
            )
        else:
            model.addConstr(
                total_distance >= 1.0,
                name=f"lb_{self.calls}_distance_lb",
            )
            model.addConstr(
                total_distance <= float(self.radius),
                name=f"lb_{self.calls}_distance_ub",
            )
        model.addConstr(
            gamma >= max(0, gamma_center - self.gamma_radius),
            name=f"lb_{self.calls}_gamma_lb",
        )
        model.addConstr(
            gamma <= gamma_center + self.gamma_radius,
            name=f"lb_{self.calls}_gamma_ub",
        )
        if self.max_removals is not None:
            model.addConstr(
                removal_distance <= float(self.max_removals),
                name=f"lb_{self.calls}_removal_distance_ub",
            )
        protected_pairs = self._add_service_preservation_constraints(
            model=model,
            center=center,
            y=y,
        )
        envelope_cuts, envelope_active = self._apply_benders_envelope_objective(
            model=model,
            x=x,
            y=y,
            z=z,
            gamma=gamma,
            center_objective=center_objective,
        )
        candidates: list[Tuple[Design, Optional[float]]] = []
        start = time.perf_counter()
        model.Params.TimeLimit = max(
            0.05,
            min(
                float(per_call_time_limit),
                deadline - time.perf_counter(),
            ),
        )
        model.optimize()
        if (
            model.Status in {GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL}
            and model.SolCount > 0
        ):
            for solution_index in range(
                min(int(model.SolCount), self.solution_pool_size)
            ):
                model.Params.SolutionNumber = solution_index
                try:
                    proxy_objective = float(model.PoolObjVal)
                except (gp.GurobiError, AttributeError):
                    proxy_objective = None
                candidates.append(
                    (
                        {
                            "x": {
                                key: float(variable.Xn)
                                for key, variable in x.items()
                            },
                            "y": {
                                key: float(variable.Xn)
                                for key, variable in y.items()
                            },
                            "z": {
                                key: float(variable.Xn)
                                for key, variable in z.items()
                            },
                            "Gamma": float(gamma.Xn),
                        },
                        proxy_objective,
                    )
                )

        best_candidate = None
        candidate_objective_best = None
        candidate_objective = None
        candidate_proxy_best = None
        accepted = False
        pool_evaluated = 0
        for candidate, proxy_objective in candidates:
            signature = design_signature(candidate)
            if signature not in self.seen_candidates:
                self.seen_candidates.add(signature)
                remaining = deadline - time.perf_counter()
                if remaining > 0.01:
                    pool_evaluated += 1
                    sub = owner.solve_subproblem_with_deadline(
                        candidate,
                        label=f"local_branch_{self.calls}",
                        deadline=deadline,
                    )
                    if sub.obj_value is not None:
                        owner.register_known_exact(
                            candidate, float(sub.obj_value)
                        )
                        candidate_objective = (
                            float(sub.obj_value)
                            - owner._fixed_cost_value(candidate)
                        )
                        if (
                            best_candidate is None
                            or candidate_objective
                            > (float(candidate_objective_best) + 1e-7)
                        ):
                            best_candidate = candidate
                            candidate_objective_best = candidate_objective
                            candidate_proxy_best = proxy_objective
                        if (
                            sub.cut_expr_data is not None
                            and sub.rhs_at_reference is not None
                        ):
                            owner.add_classical_cut(
                                sub,
                                iteration=owner.current_iteration,
                                prefix="local_branch_classic",
                            )
        if best_candidate is not None:
            candidate_objective = candidate_objective_best
            accepted = candidate_objective > center_objective + 1e-7

        improvement = (
            None
            if candidate_objective is None
            else candidate_objective - center_objective
        )
        self.history.append(
            LocalBranchingRecord(
                call=self.calls,
                center_objective=center_objective,
                radius=self.radius,
                gamma_radius=self.gamma_radius,
                arc_radius=(
                    self.arc_radius
                    if self.distance_scope == "layered"
                    else None
                ),
                wall_time=time.perf_counter() - start,
                candidate_objective=candidate_objective,
                improvement=improvement,
                accepted=accepted,
                distance_scope=self.distance_scope,
                solution_pool_size=self.solution_pool_size,
                pool_solutions=len(candidates),
                pool_evaluated=pool_evaluated,
                benders_envelope_cuts=envelope_cuts,
                benders_envelope_cutoff=(
                    envelope_active and self.benders_envelope_cutoff
                ),
                candidate_proxy_objective=candidate_proxy_best,
                max_removals=self.max_removals,
                protected_pairs=protected_pairs,
            )
        )
        if accepted:
            self.radius = max(self.min_radius, self.radius // 2)
            self.gamma_radius = max(1, self.gamma_radius // 2)
            self.arc_radius = max(1, self.arc_radius // 2)
            return best_candidate, candidate_objective
        self.radius = min(self.max_radius, max(self.radius + 2, 2 * self.radius))
        self.gamma_radius = min(
            self.max_gamma_radius,
            max(self.gamma_radius + 2, 2 * self.gamma_radius),
        )
        self.arc_radius = min(
            self.max_arc_radius,
            max(self.arc_radius + 2, 2 * self.arc_radius),
        )
        return None, candidate_objective
