"""Shared result and logging types for the isolated 2026-06-23 framework."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Tuple


Number = float | int
Node = str | int
Arc = Tuple[Node, Node]
Pair = Tuple[Node, Node]
Design = Dict[str, object]


@dataclass
class PhaseTiming:
    phase: str
    wall_time: float = 0.0
    solver_time: float = 0.0
    calls: int = 0

    @property
    def non_solver_time(self) -> float:
        return max(0.0, self.wall_time - self.solver_time)

    def as_dict(self) -> Dict[str, Number | str]:
        return {
            "phase": self.phase,
            "wall_time": self.wall_time,
            "solver_time": self.solver_time,
            "non_solver_time": self.non_solver_time,
            "calls": self.calls,
        }


@dataclass
class CutAuditRecord:
    iteration: int
    cut_family: str
    reference_value: Optional[float]
    rhs_at_reference: Optional[float]
    rhs_at_core: Optional[float]
    classical_rhs_at_core: Optional[float]
    core_improvement: Optional[float]
    reference_error: Optional[float]
    globally_validated: bool
    sampled_designs: int = 0
    max_sample_violation: Optional[float] = None
    note: str = ""

    def as_dict(self) -> Dict[str, Number | str | bool | None]:
        return {
            "iteration": self.iteration,
            "cut_family": self.cut_family,
            "reference_value": self.reference_value,
            "rhs_at_reference": self.rhs_at_reference,
            "rhs_at_core": self.rhs_at_core,
            "classical_rhs_at_core": self.classical_rhs_at_core,
            "core_improvement": self.core_improvement,
            "reference_error": self.reference_error,
            "globally_validated": self.globally_validated,
            "sampled_designs": self.sampled_designs,
            "max_sample_violation": self.max_sample_violation,
            "note": self.note,
        }


@dataclass
class LagrangianIteration:
    iteration: int
    dual_bound: float
    best_dual_bound: float
    selected_pairs: int
    open_hubs: int
    violation_norm: float
    step_size: float

    def as_dict(self) -> Dict[str, Number]:
        return {
            "iteration": self.iteration,
            "dual_bound": self.dual_bound,
            "best_dual_bound": self.best_dual_bound,
            "selected_pairs": self.selected_pairs,
            "open_hubs": self.open_hubs,
            "violation_norm": self.violation_norm,
            "step_size": self.step_size,
        }


@dataclass
class LagrangianResult:
    designs: List[Design]
    best_dual_bound: float
    history: List[LagrangianIteration] = field(default_factory=list)
    candidate_history: List[Dict[str, object]] = field(default_factory=list)
    wall_time: float = 0.0
    metadata: Dict[str, object] = field(default_factory=dict)


@dataclass
class LocalBranchingRecord:
    call: int
    center_objective: float
    radius: int
    gamma_radius: int
    arc_radius: Optional[int]
    wall_time: float
    candidate_objective: Optional[float]
    improvement: Optional[float]
    accepted: bool
    distance_scope: str = "all"
    solution_pool_size: int = 1
    pool_solutions: int = 0
    pool_evaluated: int = 0
    benders_envelope_cuts: int = 0
    benders_envelope_cutoff: bool = False
    candidate_proxy_objective: Optional[float] = None
    max_removals: Optional[int] = None
    protected_pairs: int = 0

    def as_dict(self) -> Dict[str, Number | bool | None]:
        return {
            "call": self.call,
            "center_objective": self.center_objective,
            "radius": self.radius,
            "gamma_radius": self.gamma_radius,
            "arc_radius": self.arc_radius,
            "distance_scope": self.distance_scope,
            "wall_time": self.wall_time,
            "candidate_objective": self.candidate_objective,
            "improvement": self.improvement,
            "accepted": self.accepted,
            "solution_pool_size": self.solution_pool_size,
            "pool_solutions": self.pool_solutions,
            "pool_evaluated": self.pool_evaluated,
            "benders_envelope_cuts": self.benders_envelope_cuts,
            "benders_envelope_cutoff": self.benders_envelope_cutoff,
            "candidate_proxy_objective": self.candidate_proxy_objective,
            "max_removals": self.max_removals,
            "protected_pairs": self.protected_pairs,
        }


@dataclass
class SolveResult20260623:
    method: str
    success: bool
    objective: Optional[float]
    bound: Optional[float]
    gap: Optional[float]
    wall_time: float
    time_to_best: Optional[float]
    best_design: Optional[Design]
    history: List[Dict[str, object]] = field(default_factory=list)
    initial_design_history: List[Dict[str, object]] = field(
        default_factory=list
    )
    cut_history: List[CutAuditRecord] = field(default_factory=list)
    local_branching_history: List[LocalBranchingRecord] = field(
        default_factory=list
    )
    gamma_sweep_history: List[Dict[str, object]] = field(
        default_factory=list
    )
    lr_result: Optional[LagrangianResult] = None
    metadata: Dict[str, object] = field(default_factory=dict)

    def summary(self) -> Dict[str, object]:
        return {
            "method": self.method,
            "status_name": "FEASIBLE" if self.success else "NO_INCUMBENT",
            "sol_count": 1 if self.success else 0,
            "obj": self.objective,
            "bound": self.bound,
            "gap": self.gap,
            "wall_time": self.wall_time,
            "time_to_best": self.time_to_best,
            **self.metadata,
        }


def design_signature(
    design: Mapping[str, object],
) -> Tuple[int, ...]:
    """Return a deterministic signature including integer fleet size."""

    x = tuple(
        int(float(value) > 0.5)
        for _, value in sorted(design.get("x", {}).items())
    )
    y = tuple(
        int(float(value) > 0.5)
        for _, value in sorted(design.get("y", {}).items())
    )
    z = tuple(
        int(float(value) > 0.5)
        for _, value in sorted(design.get("z", {}).items())
    )
    gamma = int(round(float(design.get("Gamma", 0.0))))
    return (*x, *y, *z, gamma)
