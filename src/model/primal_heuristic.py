"""Exact-conic local branching for primal MIP-start improvement.

The heuristic changes only the binary design neighbourhood.  All continuous
allocation, flow, fleet, and queueing variables, including the original
second-order-cone constraints, remain in the subproblem.  Consequently, a
solution returned by this module is feasible for the original MISOCP after the
temporary local-branching and objective-cutoff constraints are removed.

The intended integration with :class:`src.model.gurobi_model.MLNDModel` is::

    seed = capture_incumbent(mlnd.model, mlnd.vars)
    result = improve_mip_start(mlnd.model, mlnd.vars, seed)

``improve_mip_start`` installs the best complete solution as a MIP start.  It
does not leave a neighbourhood constraint, a changed variable bound, or a
changed search parameter in the final model, even if optimization raises.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Hashable

import gurobipy as gp
from gurobipy import GRB


DEFAULT_STRUCTURE_GROUPS = ("x", "z", "delta", "alpha")


@dataclass(frozen=True)
class PrimalSeed:
    """A feasible incumbent and its binary design projection.

    ``values`` contains every model variable.  Keeping the full incumbent is
    important: the local subproblem is the actual conic model, so its start
    should include the continuous queueing and network repair as well as the
    structural binaries.
    """

    objective: float
    structure: Mapping[str, Mapping[Hashable, int]]
    values: Mapping[gp.Var, float]


@dataclass(frozen=True)
class LocalBranchingConfig:
    """Controls a short exact-MISOCP local-branching search."""

    radius: int = 10
    time_limit: float = 10.0
    mip_gap: float = 0.01
    include_y: bool = False
    improvement_abs_tol: float = 1e-6
    improvement_rel_tol: float = 1e-8
    mip_focus: int = 1
    heuristics: float = 0.5
    no_rel_heur_time: float = 0.0
    output_flag: int | None = None

    def validate(self) -> None:
        if (
            isinstance(self.radius, bool)
            or not isinstance(self.radius, int)
            or self.radius < 0
        ):
            raise ValueError("local-branching radius must be a nonnegative integer")
        if not math.isfinite(self.time_limit) or self.time_limit <= 0.0:
            raise ValueError("local-branching time_limit must be positive and finite")
        if not math.isfinite(self.mip_gap) or not 0.0 <= self.mip_gap <= 1.0:
            raise ValueError("local-branching mip_gap must be in [0, 1]")
        tolerances = (self.improvement_abs_tol, self.improvement_rel_tol)
        if any(not math.isfinite(value) or value < 0.0 for value in tolerances):
            raise ValueError("improvement tolerances must be finite and nonnegative")
        if self.mip_focus not in {0, 1, 2, 3}:
            raise ValueError("mip_focus must be one of 0, 1, 2, 3")
        if not math.isfinite(self.heuristics) or not 0.0 <= self.heuristics <= 1.0:
            raise ValueError("heuristics must be in [0, 1]")
        if not math.isfinite(self.no_rel_heur_time) or self.no_rel_heur_time < 0.0:
            raise ValueError("no_rel_heur_time must be finite and nonnegative")
        if self.output_flag not in {None, 0, 1}:
            raise ValueError("output_flag must be None, 0, or 1")


@dataclass(frozen=True)
class LocalBranchingResult:
    """Outcome of one local-neighbourhood solve."""

    improved: bool
    seed_objective: float
    candidate_objective: float | None
    installed_objective: float
    selected_binary_count: int
    hamming_distance: int | None
    radius: int
    status: int
    solution_count: int
    runtime: float


@dataclass(frozen=True)
class _MIPStartSnapshot:
    num_start: int
    start_number: int
    values: tuple[tuple[float, ...], ...]


def _structure_groups(include_y: bool) -> tuple[str, ...]:
    return DEFAULT_STRUCTURE_GROUPS + (("y",) if include_y else ())


def _container_items(container: object, group: str) -> list[tuple[Hashable, gp.Var]]:
    if not hasattr(container, "items"):
        raise TypeError(f"variables[{group!r}] must be a keyed variable container")
    return list(container.items())  # type: ignore[union-attr, no-any-return]


def _selected_variables(
    variables: Mapping[str, object], include_y: bool
) -> dict[str, list[tuple[Hashable, gp.Var]]]:
    selected: dict[str, list[tuple[Hashable, gp.Var]]] = {}
    for group in _structure_groups(include_y):
        if group not in variables:
            raise KeyError(f"missing structural variable group {group!r}")
        items = _container_items(variables[group], group)
        for _, variable in items:
            if variable.VType != GRB.BINARY:
                raise TypeError(f"variables[{group!r}] contains a nonbinary variable")
        selected[group] = items
    return selected


def capture_incumbent(
    model: gp.Model,
    variables: Mapping[str, object],
    *,
    include_y: bool = False,
    integrality_tolerance: float = 1e-5,
) -> PrimalSeed:
    """Capture the current feasible incumbent as a complete primal seed."""

    model.update()
    if model.SolCount <= 0:
        raise ValueError(
            "cannot capture a primal seed before a feasible incumbent exists"
        )
    if not math.isfinite(integrality_tolerance) or integrality_tolerance < 0.0:
        raise ValueError("integrality_tolerance must be finite and nonnegative")

    selected = _selected_variables(variables, include_y)
    structure: dict[str, dict[Hashable, int]] = {}
    for group, items in selected.items():
        group_values: dict[Hashable, int] = {}
        for key, variable in items:
            value = float(variable.X)
            binary_value = int(value >= 0.5)
            if abs(value - binary_value) > integrality_tolerance:
                raise ValueError(
                    f"incumbent value {value} for {group}[{key!r}] is not binary"
                )
            group_values[key] = binary_value
        structure[group] = group_values

    model_variables = model.getVars()
    values = dict(zip(model_variables, model.getAttr(GRB.Attr.X, model_variables)))
    objective = float(model.ObjVal)
    if not math.isfinite(objective):
        raise ValueError("incumbent objective must be finite")
    return PrimalSeed(objective=objective, structure=structure, values=values)


def build_hamming_expression(
    variables: Mapping[str, object],
    structure: Mapping[str, Mapping[Hashable, int]],
    *,
    include_y: bool = False,
) -> tuple[gp.LinExpr, int]:
    r"""Build the exact Hamming distance from a binary design seed.

    For selected binary variables :math:`b` and seed :math:`\hat b`, the
    returned expression is

    .. math::

        \sum_{k:\hat b_k=0} b_k + \sum_{k:\hat b_k=1}(1-b_k).

    By default the design vector is ``(x, z, delta, alpha)``.  Arc decisions
    ``y`` can be included explicitly because they make the neighbourhood much
    larger and are partly implied by the facility decisions.
    """

    selected = _selected_variables(variables, include_y)
    expression = gp.LinExpr()
    count = 0
    for group, items in selected.items():
        if group not in structure:
            raise KeyError(f"primal seed is missing structural group {group!r}")
        expected_keys = {key for key, _ in items}
        supplied_keys = set(structure[group])
        if supplied_keys != expected_keys:
            missing = expected_keys - supplied_keys
            extra = supplied_keys - expected_keys
            raise ValueError(
                f"seed keys for {group!r} do not match model variables "
                f"(missing={len(missing)}, extra={len(extra)})"
            )
        for key, variable in items:
            value = structure[group][key]
            if isinstance(value, bool):
                value = int(value)
            if value not in {0, 1}:
                raise ValueError(f"seed value for {group}[{key!r}] must be 0 or 1")
            if value == 0:
                expression += variable
            else:
                expression += 1.0 - variable
            count += 1
    if count == 0:
        raise ValueError("the selected local-branching design vector is empty")
    return expression, count


@contextmanager
def temporary_local_branching_neighborhood(
    model: gp.Model,
    hamming_expression: gp.LinExpr,
    radius: int,
    *,
    objective_cutoff: float | None = None,
    parameter_overrides: Mapping[str, Any] | None = None,
) -> Iterator[tuple[gp.Constr, ...]]:
    """Temporarily add a local branch and restore the model transactionally.

    The snapshot uses Gurobi's bulk attribute interface, so every existing
    variable lower and upper bound is restored even if code executed inside the
    context changes a bound or raises an exception.  Only parameters listed in
    ``parameter_overrides`` are changed, and each of those is restored.
    """

    if isinstance(radius, bool) or not isinstance(radius, int) or radius < 0:
        raise ValueError("local-branching radius must be a nonnegative integer")
    if objective_cutoff is not None and not math.isfinite(objective_cutoff):
        raise ValueError("objective_cutoff must be finite or None")
    overrides = dict(parameter_overrides or {})

    model.update()
    model_variables = model.getVars()
    lower_bounds = model.getAttr(GRB.Attr.LB, model_variables)
    upper_bounds = model.getAttr(GRB.Attr.UB, model_variables)
    old_parameters = {name: getattr(model.Params, name) for name in overrides}
    temporary_constraints: list[gp.Constr] = []
    try:
        for name, value in overrides.items():
            setattr(model.Params, name, value)
        temporary_constraints.append(
            model.addConstr(
                hamming_expression <= radius, name="__local_branching_hamming"
            )
        )
        if objective_cutoff is not None:
            if model.ModelSense != GRB.MINIMIZE:
                raise ValueError(
                    "objective improvement cutoff currently requires minimization"
                )
            temporary_constraints.append(
                model.addConstr(
                    model.getObjective() <= objective_cutoff,
                    name="__local_branching_improvement",
                )
            )
        model.update()
        yield tuple(temporary_constraints)
    finally:
        if temporary_constraints:
            model.remove(temporary_constraints)
        model.setAttr(GRB.Attr.LB, model_variables, lower_bounds)
        model.setAttr(GRB.Attr.UB, model_variables, upper_bounds)
        model.update()
        for name, value in old_parameters.items():
            setattr(model.Params, name, value)


def _snapshot_mip_starts(
    model: gp.Model, model_variables: list[gp.Var]
) -> _MIPStartSnapshot:
    model.update()
    num_start = int(model.NumStart)
    start_number = int(model.Params.StartNumber)
    values: list[tuple[float, ...]] = []
    for index in range(num_start):
        model.Params.StartNumber = index
        values.append(tuple(model.getAttr(GRB.Attr.Start, model_variables)))
    model.Params.StartNumber = start_number
    return _MIPStartSnapshot(num_start, start_number, tuple(values))


def _restore_mip_starts(
    model: gp.Model, model_variables: list[gp.Var], snapshot: _MIPStartSnapshot
) -> None:
    model.NumStart = snapshot.num_start
    model.update()
    for index, values in enumerate(snapshot.values):
        model.Params.StartNumber = index
        model.setAttr(GRB.Attr.Start, model_variables, list(values))
    model.Params.StartNumber = snapshot.start_number
    model.update()


def _install_complete_start(
    model: gp.Model, model_variables: list[gp.Var], values: Mapping[gp.Var, float]
) -> None:
    if len(values) != len(model_variables) or any(
        variable not in values for variable in model_variables
    ):
        raise ValueError("a primal seed must provide a value for every model variable")
    start_values = [float(values[variable]) for variable in model_variables]
    if any(not math.isfinite(value) for value in start_values):
        raise ValueError("all primal seed values must be finite")
    old_start_number = int(model.Params.StartNumber)
    model.NumStart = 1
    model.update()
    model.Params.StartNumber = 0
    model.setAttr(GRB.Attr.Start, model_variables, start_values)
    model.Params.StartNumber = 0 if old_start_number < 0 else min(old_start_number, 0)
    model.update()


def _candidate_hamming_distance(
    variables: Mapping[str, object],
    structure: Mapping[str, Mapping[Hashable, int]],
    *,
    include_y: bool,
) -> int:
    distance = 0
    for group, items in _selected_variables(variables, include_y).items():
        for key, variable in items:
            distance += int(int(float(variable.X) >= 0.5) != structure[group][key])
    return distance


def improve_mip_start(
    model: gp.Model,
    variables: Mapping[str, object],
    seed: PrimalSeed,
    config: LocalBranchingConfig | None = None,
) -> LocalBranchingResult:
    """Search an exact conic Hamming neighbourhood and install the best start.

    The center seed remains feasible inside the Hamming ball, giving the local
    MISOCP an immediate incumbent and objective bound.  Only a strictly better
    completed solution is accepted and installed.  The global solve that
    follows remains the unmodified original MISOCP.
    """

    config = config or LocalBranchingConfig()
    config.validate()
    if not math.isfinite(seed.objective):
        raise ValueError("seed objective must be finite")
    if model.ModelSense != GRB.MINIMIZE:
        raise ValueError("exact-conic local branching currently requires minimization")

    hamming_expression, selected_count = build_hamming_expression(
        variables, seed.structure, include_y=config.include_y
    )
    tolerance = max(
        config.improvement_abs_tol,
        config.improvement_rel_tol * max(1.0, abs(seed.objective)),
    )
    parameter_overrides: dict[str, Any] = {
        "TimeLimit": config.time_limit,
        "MIPGap": config.mip_gap,
        "MIPFocus": config.mip_focus,
        "Heuristics": config.heuristics,
        "NoRelHeurTime": config.no_rel_heur_time,
    }
    if config.output_flag is not None:
        parameter_overrides["OutputFlag"] = config.output_flag

    model.update()
    model_variables = model.getVars()
    previous_starts = _snapshot_mip_starts(model, model_variables)
    candidate_values: dict[gp.Var, float] | None = None
    candidate_objective: float | None = None
    hamming_distance: int | None = None
    status = GRB.LOADED
    solution_count = 0
    runtime = 0.0
    try:
        _install_complete_start(model, model_variables, seed.values)
        with temporary_local_branching_neighborhood(
            model,
            hamming_expression,
            config.radius,
            parameter_overrides=parameter_overrides,
        ):
            model.optimize()
            status = int(model.Status)
            solution_count = int(model.SolCount)
            runtime = float(model.Runtime)
            if solution_count > 0:
                candidate_objective = float(model.ObjVal)
                hamming_distance = _candidate_hamming_distance(
                    variables, seed.structure, include_y=config.include_y
                )
                if candidate_objective < seed.objective - tolerance:
                    candidate_values = dict(
                        zip(
                            model_variables,
                            model.getAttr(GRB.Attr.X, model_variables),
                        )
                    )
    except BaseException:
        _restore_mip_starts(model, model_variables, previous_starts)
        raise

    improved = candidate_values is not None
    if improved:
        _install_complete_start(model, model_variables, candidate_values)
        installed_objective = float(candidate_objective)
    else:
        _install_complete_start(model, model_variables, seed.values)
        installed_objective = float(seed.objective)

    return LocalBranchingResult(
        improved=improved,
        seed_objective=float(seed.objective),
        candidate_objective=candidate_objective,
        installed_objective=installed_objective,
        selected_binary_count=selected_count,
        hamming_distance=hamming_distance,
        radius=config.radius,
        status=status,
        solution_count=solution_count,
        runtime=runtime,
    )
