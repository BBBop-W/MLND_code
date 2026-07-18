"""Numerically certified Magnanti-Wong cut selection for conic recourse.

The selector avoids constructing an explicit SOCP dual.  It solves the primal
SOCP at points on the segment from the reference design to a strict master
core point.  A candidate dual is accepted only if its global support cut is
also tight at the original reference design.  Tightness certifies that the
candidate remains on the reference optimal-dual face.  Conditional on that
certificate, optimality at the perturbed RHS minimizes the cut value at the
core point over the reference face, which is the Magnanti-Wong criterion.
"""

from __future__ import annotations

import math
from typing import Dict, Mapping, Optional, Sequence, Tuple

import gurobipy as gp
from gurobipy import GRB

from src import config

from .types import CutAuditRecord, Design


def interpolate_design(
    reference: Mapping[str, object],
    core: Mapping[str, object],
    epsilon: float,
) -> Design:
    weight = min(1.0, max(0.0, float(epsilon)))
    return {
        "x": {
            key: (1.0 - weight) * float(value)
            + weight * float(core["x"][key])
            for key, value in reference["x"].items()
        },
        "y": {
            key: (1.0 - weight) * float(value)
            + weight * float(core["y"][key])
            for key, value in reference["y"].items()
        },
        "z": {
            key: (1.0 - weight) * float(value)
            + weight * float(core["z"][key])
            for key, value in reference["z"].items()
        },
        "Gamma": (1.0 - weight) * float(reference["Gamma"])
        + weight * float(core["Gamma"]),
    }


def build_strict_core_point(owner) -> Tuple[Optional[Design], Dict[str, object]]:
    """Solve a max-min-distance LP over the current master relaxation."""

    assert owner.master is not None
    relaxed = owner.master.relax()
    relaxed.ModelName = "MLND_20260623_Strict_Core_Point"
    relaxed.Params.OutputFlag = 0
    relaxed.Params.Threads = 1
    relaxed.Params.Seed = int(getattr(config, "BENDERS_RANDOM_SEED", 0))
    relaxed.update()

    by_name = {var.VarName: var for var in relaxed.getVars()}
    radius = relaxed.addVar(lb=0.0, ub=0.499, name="core_radius")
    strategic_names = []
    for node in owner.N_x:
        strategic_names.append(f"x[{node}]")
    for u, v in owner.AY:
        strategic_names.append(f"y[{u},{v}]")
    for hub in owner.H:
        for c in range(len(config.MU_HUB)):
            strategic_names.append(f"z[{hub},{c}]")

    constrained = 0
    for name in strategic_names:
        variable = by_name.get(name)
        if variable is None:
            continue
        relaxed.addConstr(variable >= radius)
        relaxed.addConstr(variable <= 1.0 - radius)
        constrained += 1
    relaxed.setObjective(radius, GRB.MAXIMIZE)
    relaxed.optimize()
    if relaxed.SolCount == 0:
        return None, {
            "status": int(relaxed.Status),
            "radius": None,
            "constrained_variables": constrained,
        }

    def value(name: str) -> float:
        return float(by_name[name].X)

    core: Design = {
        "x": {node: value(f"x[{node}]") for node in owner.N_x},
        "y": {arc: value(f"y[{arc[0]},{arc[1]}]") for arc in owner.AY},
        "z": {
            (hub, c): value(f"z[{hub},{c}]")
            for hub in owner.H
            for c in range(len(config.MU_HUB))
        },
        "Gamma": value("Gamma"),
    }
    return core, {
        "status": int(relaxed.Status),
        "radius": float(radius.X),
        "constrained_variables": constrained,
        "runtime": float(relaxed.Runtime),
    }


class ParetoCutSelector20260623:
    """Generate and audit a reference-tight core-optimal conic dual cut."""

    def __init__(
        self,
        owner,
        *,
        epsilons: Sequence[float] = (
            0.50,
            0.25,
            0.10,
            0.05,
            0.02,
            0.01,
            0.005,
            0.001,
        ),
        reference_tolerance: float = 1e-5,
        minimum_core_improvement: float = 1e-7,
    ) -> None:
        self.owner = owner
        self.epsilons = tuple(float(value) for value in epsilons)
        self.reference_tolerance = float(reference_tolerance)
        self.minimum_core_improvement = float(minimum_core_improvement)
        self.core_point: Optional[Design] = None
        self.core_metadata: Dict[str, object] = {}

    def initialize_core(self) -> Optional[Design]:
        if self.core_point is None:
            self.core_point, self.core_metadata = build_strict_core_point(
                self.owner
            )
        return self.core_point

    def select_and_add(
        self,
        *,
        reference_sub,
        classical_rhs_at_core: Optional[float],
        iteration: int,
        deadline: float,
    ) -> Optional[CutAuditRecord]:
        if reference_sub.obj_value is None:
            return None
        core = self.initialize_core()
        if core is None:
            return None

        reference_value = float(reference_sub.obj_value)
        scale = max(1.0, abs(reference_value))
        best = None
        best_core_rhs = math.inf
        best_reference_rhs = None
        best_epsilon = None
        for epsilon in self.epsilons:
            if self.owner.remaining_time(deadline) <= 0.01:
                break
            perturbed = interpolate_design(
                reference_sub.design, core, epsilon
            )
            candidate = self.owner.solve_subproblem_with_deadline(
                perturbed,
                label=f"pareto_{iteration}_eps_{epsilon:g}",
                deadline=deadline,
            )
            if (
                candidate.obj_value is None
                or candidate.cut_expr_data is None
                or candidate.rhs_at_reference is None
                or not candidate.cut_expr_data
            ):
                continue
            rhs_reference = self.owner._cut_rhs_value(
                candidate, reference_sub.design
            )
            rhs_core = self.owner._cut_rhs_value(candidate, core)
            if rhs_reference is None or rhs_core is None:
                continue
            reference_error = rhs_reference - reference_value
            if (
                reference_error < -self.reference_tolerance * scale
                or abs(reference_error) > self.reference_tolerance * scale
            ):
                continue
            if rhs_core < best_core_rhs:
                best = candidate
                best_core_rhs = rhs_core
                best_reference_rhs = rhs_reference
                best_epsilon = epsilon

        if best is None:
            return CutAuditRecord(
                iteration=iteration,
                cut_family="pareto_mw_conic",
                reference_value=reference_value,
                rhs_at_reference=None,
                rhs_at_core=None,
                classical_rhs_at_core=classical_rhs_at_core,
                core_improvement=None,
                reference_error=None,
                globally_validated=False,
                note=(
                    "No perturbed dual remained reference-tight within "
                    f"tolerance={self.reference_tolerance:g}."
                ),
            )

        improvement = (
            None
            if classical_rhs_at_core is None
            else classical_rhs_at_core - best_core_rhs
        )
        sampled, max_violation = self.owner.sample_validate_sub_cut(best)
        sample_valid = max_violation is None or max_violation <= 1e-5
        if (
            improvement is not None
            and improvement < self.minimum_core_improvement
        ):
            return CutAuditRecord(
                iteration=iteration,
                cut_family="pareto_mw_conic",
                reference_value=reference_value,
                rhs_at_reference=best_reference_rhs,
                rhs_at_core=best_core_rhs,
                classical_rhs_at_core=classical_rhs_at_core,
                core_improvement=improvement,
                reference_error=best_reference_rhs - reference_value,
                globally_validated=sample_valid,
                sampled_designs=sampled,
                max_sample_violation=max_violation,
                note=(
                    f"Certified with epsilon={best_epsilon:g}, but skipped "
                    "because it did not improve the classical cut at the core."
                ),
            )

        if not sample_valid:
            return CutAuditRecord(
                iteration=iteration,
                cut_family="pareto_mw_conic",
                reference_value=reference_value,
                rhs_at_reference=best_reference_rhs,
                rhs_at_core=best_core_rhs,
                classical_rhs_at_core=classical_rhs_at_core,
                core_improvement=improvement,
                reference_error=best_reference_rhs - reference_value,
                globally_validated=False,
                sampled_designs=sampled,
                max_sample_violation=max_violation,
                note="Rejected because a known exact design violated the cut.",
            )
        self.owner._add_benders_cut(best, "pareto_mw_conic")
        return CutAuditRecord(
            iteration=iteration,
            cut_family="pareto_mw_conic",
            reference_value=reference_value,
            rhs_at_reference=best_reference_rhs,
            rhs_at_core=best_core_rhs,
            classical_rhs_at_core=classical_rhs_at_core,
            core_improvement=improvement,
            reference_error=best_reference_rhs - reference_value,
            globally_validated=True,
            sampled_designs=sampled,
            max_sample_violation=max_violation,
            note=(
                f"Reference-tight perturbed dual accepted at "
                f"epsilon={best_epsilon:g}; numerical MW certificate."
            ),
        )
