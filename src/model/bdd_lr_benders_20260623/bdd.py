"""Rahmaniani et al. (2020) Benders dual-decomposition cut machinery.

The original model builder is used only as a convenient constructor for the
local-copy MISOCP feasible region.  Its objective is replaced, its variables
are not fixed to a monolithic incumbent, and only a rigorous maximization
``ObjBound`` is used in a cut.  A local-copy incumbent is merely a candidate
until the separate fixed-design SOCP oracle validates it.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import gurobipy as gp
from gurobipy import GRB

from src import config
from src.model.gurobi_model import MLNDModel

from .types import CutAuditRecord, Design


CoefficientKey = Tuple[str, object]


@dataclass
class BDDInnerPoint:
    operational_value: float
    design: Design


@dataclass
class BDDInnerResult:
    status: int
    intercept_bound: Optional[float]
    incumbent_intercept: Optional[float]
    gap: Optional[float]
    runtime: float
    local_design: Optional[Design]
    local_operational_value: Optional[float]
    epsilon_correction: float


def zero_design(owner) -> Design:
    return {
        "x": {node: 0.0 for node in owner.N_x},
        "y": {arc: 0.0 for arc in owner.AY},
        "z": {
            (hub, c): 0.0
            for hub in owner.H
            for c in range(len(config.MU_HUB))
        },
        "Gamma": 0.0,
    }


def affine_cut_coefficients(
    owner, sub
) -> Tuple[float, Dict[CoefficientKey, float]]:
    """Recover the affine classical-cut slope by basis evaluation."""

    origin = zero_design(owner)
    intercept = owner._cut_rhs_value(sub, origin)
    if intercept is None:
        raise ValueError("Classical cut is unavailable")
    coefficients: Dict[CoefficientKey, float] = {}

    for node in owner.N_x:
        basis = zero_design(owner)
        basis["x"][node] = 1.0
        value = owner._cut_rhs_value(sub, basis)
        coefficients["x", node] = float(value) - intercept
    for arc in owner.AY:
        basis = zero_design(owner)
        basis["y"][arc] = 1.0
        value = owner._cut_rhs_value(sub, basis)
        coefficients["y", arc] = float(value) - intercept
    for hub in owner.H:
        for c in range(len(config.MU_HUB)):
            basis = zero_design(owner)
            basis["z"][hub, c] = 1.0
            value = owner._cut_rhs_value(sub, basis)
            coefficients["z", (hub, c)] = float(value) - intercept
    basis = zero_design(owner)
    basis["Gamma"] = 1.0
    value = owner._cut_rhs_value(sub, basis)
    coefficients["Gamma", None] = float(value) - intercept
    return float(intercept), coefficients


def coefficient_value(
    coefficients: Mapping[CoefficientKey, float],
    design: Mapping[str, object],
) -> float:
    value = 0.0
    for (family, key), coefficient in coefficients.items():
        if family == "Gamma":
            design_value = float(design["Gamma"])
        else:
            design_value = float(design[family][key])
        value += float(coefficient) * design_value
    return value


def master_coefficient_expression(owner, coefficients):
    expression = gp.LinExpr()
    for (family, key), coefficient in coefficients.items():
        if abs(float(coefficient)) <= 1e-12:
            continue
        if family == "Gamma":
            variable = owner.master_vars["Gamma"]
        else:
            variable = owner.master_vars[family][key]
        expression += float(coefficient) * variable
    return expression


class BDDLocalCopyOracle20260623:
    """Solve the priced local-copy MISOCP for one multiplier vector."""

    def __init__(
        self,
        owner,
        *,
        fractional_lower: float = 0.05,
        fractional_upper: float = 0.95,
        full_integrality: bool = False,
    ) -> None:
        self.owner = owner
        self.fractional_lower = float(fractional_lower)
        self.fractional_upper = float(fractional_upper)
        self.full_integrality = bool(full_integrality)

    def _should_be_integral(self, value: float) -> bool:
        return self.full_integrality or (
            self.fractional_lower
            <= float(value)
            <= self.fractional_upper
        )

    def solve(
        self,
        coefficients: Mapping[CoefficientKey, float],
        *,
        reference_design: Mapping[str, object],
        time_limit: float,
        epsilon_abs: Optional[float] = None,
        label: str,
    ) -> BDDInnerResult:
        local = MLNDModel(self.owner.inst)
        local.build()
        model = local.model
        model.ModelName = f"MLND_20260623_BDD_LocalCopy_{label}"
        model.Params.OutputFlag = 0
        model.Params.Threads = 1
        model.Params.Seed = int(getattr(config, "BENDERS_RANDOM_SEED", 0))
        model.Params.TimeLimit = max(0.05, float(time_limit))
        model.Params.MIPGap = float(
            getattr(config, "BDD_LOCAL_MIP_GAP", 0.005)
        )

        x = local.vars["x"]
        y = local.vars["y"]
        z = local.vars["z"]
        gamma = local.vars["Gamma"]
        for node in self.owner.N_x:
            if not self._should_be_integral(
                float(reference_design["x"][node])
            ):
                x[node].VType = GRB.CONTINUOUS
        for arc in self.owner.AY:
            if not self._should_be_integral(
                float(reference_design["y"][arc])
            ):
                y[arc].VType = GRB.CONTINUOUS
        for hub in self.owner.H:
            for c in range(len(config.MU_HUB)):
                if not self._should_be_integral(
                    float(reference_design["z"][hub, c])
                ):
                    z[hub, c].VType = GRB.CONTINUOUS
        if not self._should_be_integral(
            float(reference_design["Gamma"])
            - math.floor(float(reference_design["Gamma"]))
        ):
            gamma.VType = GRB.CONTINUOUS

        served = gp.quicksum(
            float(getattr(config, "ORDER_REVENUE_COEF", 0.0))
            * local.vars["psi"][i, j]
            for i, j in self.owner.A_HH
        )
        operating_cost = gp.quicksum(
            local.flight_cost[u, v, k] * variable
            for (u, v, k), variable in local.vars["varphi"].items()
        )
        priced_design = gp.LinExpr()
        for (family, key), coefficient in coefficients.items():
            if family == "x":
                variable = x[key]
            elif family == "y":
                variable = y[key]
            elif family == "z":
                variable = z[key]
            else:
                variable = gamma
            priced_design += float(coefficient) * variable
        model.setObjective(served - operating_cost - priced_design, GRB.MAXIMIZE)
        model.optimize()

        if model.SolCount == 0:
            return BDDInnerResult(
                status=int(model.Status),
                intercept_bound=None,
                incumbent_intercept=None,
                gap=None,
                runtime=float(model.Runtime),
                local_design=None,
                local_operational_value=None,
                epsilon_correction=0.0,
            )

        incumbent_intercept = float(model.ObjVal)
        bound = float(model.ObjBound)
        epsilon_correction = 0.0
        if epsilon_abs is not None:
            certified_gap = max(0.0, bound - incumbent_intercept)
            if certified_gap <= float(epsilon_abs) + 1e-9:
                epsilon_correction = float(epsilon_abs)
                bound = min(bound, incumbent_intercept + epsilon_correction)

        local_design: Design = {
            "x": {key: float(variable.X) for key, variable in x.items()},
            "y": {key: float(variable.X) for key, variable in y.items()},
            "z": {key: float(variable.X) for key, variable in z.items()},
            "Gamma": float(gamma.X),
        }
        operational_value = float(
            served.getValue() - operating_cost.getValue()
        )
        return BDDInnerResult(
            status=int(model.Status),
            intercept_bound=bound,
            incumbent_intercept=incumbent_intercept,
            gap=(
                None
                if abs(bound) >= 0.5 * GRB.INFINITY
                else max(0.0, bound - incumbent_intercept)
                / max(1.0, abs(incumbent_intercept))
            ),
            runtime=float(model.Runtime),
            local_design=local_design,
            local_operational_value=operational_value,
            epsilon_correction=epsilon_correction,
        )


class BDDCutGenerator20260623:
    """Generate strengthened and regularized Lagrangian BDD cuts."""

    def __init__(
        self,
        owner,
        *,
        full_integrality: bool = False,
        epsilon_abs: Optional[float] = None,
        lagrangian_iterations: int = 3,
        regularization: float = 1e-3,
        minimum_lift: float = 1e-5,
    ) -> None:
        self.owner = owner
        self.oracle = BDDLocalCopyOracle20260623(
            owner, full_integrality=full_integrality
        )
        self.epsilon_abs = epsilon_abs
        self.lagrangian_iterations = max(1, int(lagrangian_iterations))
        self.regularization = max(1e-12, float(regularization))
        self.minimum_lift = float(minimum_lift)
        self.local_candidates: List[Design] = []

    def _add_cut(
        self,
        *,
        coefficients: Mapping[CoefficientKey, float],
        intercept_bound: float,
        prefix: str,
    ) -> None:
        self.owner.cut_count += 1
        self.owner.master.addConstr(
            self.owner.master_vars["eta"]
            <= float(intercept_bound)
            + master_coefficient_expression(self.owner, coefficients),
            name=f"{prefix}_{self.owner.cut_count}",
        )

    def strengthened_cut(
        self,
        *,
        classical_sub,
        reference_design: Mapping[str, object],
        iteration: int,
        time_limit: float,
    ) -> Optional[CutAuditRecord]:
        if (
            classical_sub.obj_value is None
            or classical_sub.cut_expr_data is None
            or not classical_sub.cut_expr_data
        ):
            return None
        classical_intercept, coefficients = affine_cut_coefficients(
            self.owner, classical_sub
        )
        result = self.oracle.solve(
            coefficients,
            reference_design=reference_design,
            time_limit=time_limit,
            epsilon_abs=self.epsilon_abs,
            label=f"strengthened_{iteration}",
        )
        if result.intercept_bound is None:
            return None
        if result.local_design is not None:
            self.local_candidates.append(result.local_design)
        classical_at_reference = (
            classical_intercept
            + coefficient_value(coefficients, reference_design)
        )
        strengthened_at_reference = (
            float(result.intercept_bound)
            + coefficient_value(coefficients, reference_design)
        )
        sampled, max_violation = self.owner.sample_validate_affine_cut(
            intercept=float(result.intercept_bound),
            coefficients=coefficients,
        )
        sample_valid = max_violation is None or max_violation <= 1e-5
        if sample_valid:
            self._add_cut(
                coefficients=coefficients,
                intercept_bound=result.intercept_bound,
                prefix="bdd_strengthened",
            )
        return CutAuditRecord(
            iteration=iteration,
            cut_family="bdd_strengthened",
            reference_value=float(classical_sub.obj_value),
            rhs_at_reference=strengthened_at_reference,
            rhs_at_core=None,
            classical_rhs_at_core=None,
            core_improvement=(
                classical_at_reference - strengthened_at_reference
            ),
            reference_error=(
                strengthened_at_reference
                - float(classical_sub.obj_value)
            ),
            globally_validated=sample_valid,
            sampled_designs=sampled,
            max_sample_violation=max_violation,
            note=(
                "Local-copy MISOCP; cut intercept uses maximization ObjBound. "
                f"status={result.status}, gap={result.gap}, "
                f"epsilon_correction={result.epsilon_correction}."
            ),
        )

    def _regularized_multiplier_update(
        self,
        *,
        reference_design: Mapping[str, object],
        previous: Mapping[CoefficientKey, float],
        points: Sequence[BDDInnerPoint],
    ) -> Dict[CoefficientKey, float]:
        keys = list(previous)
        model = gp.Model("MLND_20260623_BDD_Multiplier_Master")
        model.Params.OutputFlag = 0
        model.Params.Threads = 1
        multiplier = model.addVars(
            range(len(keys)), lb=-GRB.INFINITY, name="lambda"
        )
        eta = model.addVar(lb=-GRB.INFINITY, name="eta")
        for point_index, point in enumerate(points):
            expression = float(point.operational_value)
            for position, key in enumerate(keys):
                family, item = key
                if family == "Gamma":
                    reference_value = float(reference_design["Gamma"])
                    point_value = float(point.design["Gamma"])
                else:
                    reference_value = float(
                        reference_design[family][item]
                    )
                    point_value = float(point.design[family][item])
                expression += (
                    reference_value - point_value
                ) * multiplier[position]
            model.addConstr(
                eta >= expression, name=f"bundle_point_{point_index}"
            )
        regularizer = gp.quicksum(
            (
                multiplier[position] - float(previous[key])
            )
            * (
                multiplier[position] - float(previous[key])
            )
            for position, key in enumerate(keys)
        )
        model.setObjective(
            eta + 0.5 * self.regularization * regularizer,
            GRB.MINIMIZE,
        )
        model.optimize()
        if model.SolCount == 0:
            return dict(previous)
        return {
            key: float(multiplier[position].X)
            for position, key in enumerate(keys)
        }

    def lagrangian_cut(
        self,
        *,
        classical_sub,
        reference_design: Mapping[str, object],
        iteration: int,
        deadline: float,
        per_inner_time_limit: float,
    ) -> Optional[CutAuditRecord]:
        if (
            classical_sub.obj_value is None
            or classical_sub.cut_expr_data is None
            or not classical_sub.cut_expr_data
        ):
            return None
        _, coefficients = affine_cut_coefficients(
            self.owner, classical_sub
        )
        points: List[BDDInnerPoint] = []
        best_result = None
        best_coefficients = dict(coefficients)
        best_reference_rhs = math.inf

        for inner_iteration in range(1, self.lagrangian_iterations + 1):
            remaining = deadline - time.perf_counter()
            if remaining <= 0.05:
                break
            result = self.oracle.solve(
                coefficients,
                reference_design=reference_design,
                time_limit=min(float(per_inner_time_limit), remaining),
                epsilon_abs=self.epsilon_abs,
                label=f"lagrangian_{iteration}_{inner_iteration}",
            )
            if result.intercept_bound is None:
                break
            reference_rhs = (
                float(result.intercept_bound)
                + coefficient_value(coefficients, reference_design)
            )
            if reference_rhs < best_reference_rhs:
                best_reference_rhs = reference_rhs
                best_result = result
                best_coefficients = dict(coefficients)
            if (
                result.local_design is None
                or result.local_operational_value is None
            ):
                break
            self.local_candidates.append(result.local_design)
            points.append(
                BDDInnerPoint(
                    operational_value=result.local_operational_value,
                    design=result.local_design,
                )
            )
            updated = self._regularized_multiplier_update(
                reference_design=reference_design,
                previous=coefficients,
                points=points,
            )
            movement = math.sqrt(
                sum(
                    (updated[key] - coefficients[key]) ** 2
                    for key in coefficients
                )
            )
            coefficients = updated
            if movement <= self.minimum_lift:
                break

        if best_result is None or best_result.intercept_bound is None:
            return None
        sampled, max_violation = self.owner.sample_validate_affine_cut(
            intercept=float(best_result.intercept_bound),
            coefficients=best_coefficients,
        )
        sample_valid = max_violation is None or max_violation <= 1e-5
        if sample_valid:
            self._add_cut(
                coefficients=best_coefficients,
                intercept_bound=best_result.intercept_bound,
                prefix="bdd_lagrangian",
            )
        return CutAuditRecord(
            iteration=iteration,
            cut_family="bdd_lagrangian",
            reference_value=float(classical_sub.obj_value),
            rhs_at_reference=best_reference_rhs,
            rhs_at_core=None,
            classical_rhs_at_core=None,
            core_improvement=(
                float(classical_sub.obj_value) - best_reference_rhs
            ),
            reference_error=(
                best_reference_rhs - float(classical_sub.obj_value)
            ),
            globally_validated=sample_valid,
            sampled_designs=sampled,
            max_sample_violation=max_violation,
            note=(
                "Regularized multiplier bundle with local-copy MISOCP "
                f"ObjBound; inner_points={len(points)}."
            ),
        )


def solve_fractional_master(owner, *, time_limit: float) -> Optional[Design]:
    """Solve the current LP master and map its strategic solution by name."""

    assert owner.master is not None
    relaxation = owner.master.relax()
    relaxation.ModelName = "MLND_20260623_BDD_Fractional_Master"
    relaxation.Params.OutputFlag = 0
    relaxation.Params.Threads = 1
    relaxation.Params.TimeLimit = max(0.05, float(time_limit))
    relaxation.optimize()
    if relaxation.SolCount == 0:
        return None
    by_name = {variable.VarName: variable for variable in relaxation.getVars()}
    return {
        "x": {
            node: float(by_name[f"x[{node}]"].X) for node in owner.N_x
        },
        "y": {
            arc: float(by_name[f"y[{arc[0]},{arc[1]}]"].X)
            for arc in owner.AY
        },
        "z": {
            (hub, c): float(by_name[f"z[{hub},{c}]"].X)
            for hub in owner.H
            for c in range(len(config.MU_HUB))
        },
        "Gamma": float(by_name["Gamma"].X),
    }
