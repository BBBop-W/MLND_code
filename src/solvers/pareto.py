"""Certified Magnanti--Wong-style selection for the lifted conic recourse.

Exact Pareto optimality requires an exact optimal face and exact auxiliary
optimality. This implementation reports the numerical face and verification
tolerances explicitly and always keeps the ordinary certified cut as fallback.
"""

from __future__ import annotations

import time
import numpy as np
from gurobipy import GRB, GurobiError

from .dual import (
    DUAL_TIMING_PHASES,
    solve_dual,
    price_catalogue,
    certified_cut,
    cut_value,
)
from .formulation import BuildStopped, shapes


def _validate_design_space(data, point, *, strict):
    for name, shape in shapes(data).items():
        if name not in point:
            raise ValueError(f"Missing design component: {name}")
        array = np.asarray(point[name], dtype=float)
        if array.shape != shape or not np.isfinite(array).all():
            raise ValueError(f"Invalid design component: {name}")
        if np.any(array < -1e-10) or np.any(array > 1 + 1e-10):
            raise ValueError(f"Design component outside [0, 1]: {name}")
    x, z, a, y = (np.asarray(point[name]) for name in shapes(data))
    if not np.allclose(z.sum(axis=1), x[data.n_l :], atol=1e-10, rtol=0):
        raise ValueError("Capacity levels must sum to locker opening")
    if np.any(a.sum(axis=(1, 2)) > 1 + 1e-10):
        raise ValueError("An OD cannot have more than one assignment")
    if np.any(a > x[: data.n_l][None, :, None] + 1e-10) or np.any(
        a > x[: data.n_l][None, None, :] + 1e-10
    ):
        raise ValueError("Assignment exceeds base opening")
    allowed = np.asarray(data.allowed_links, dtype=bool)
    if np.any(np.abs(y[~allowed]) > 1e-10):
        raise ValueError("A forbidden link must remain zero")
    if np.any(y > x[:, None] + 1e-10) or np.any(y > x[None, :] + 1e-10):
        raise ValueError("Airline exceeds endpoint opening")
    if strict:
        if np.any(x <= 0) or np.any(x >= 1) or np.any(z <= 0):
            raise ValueError("The core lies on a facility or capacity boundary")
        if np.any(a <= 0) or np.any(a.sum(axis=(1, 2)) >= 1):
            raise ValueError("The core lies on an assignment boundary")
        if np.any(a >= x[: data.n_l][None, :, None]) or np.any(
            a >= x[: data.n_l][None, None, :]
        ):
            raise ValueError("The core lies on an assignment activation boundary")
        if np.any(y[allowed] <= 0) or np.any(
            y[allowed] >= np.minimum(x[:, None], x[None, :])[allowed]
        ):
            raise ValueError("The core lies on a link activation boundary")
    return True


def validate_core_point(data, point):
    """Check the design-space equations and strict non-fixed inequalities.

    These checks are a numerical guard, not an independent convex-hull proof.
    The full-support distribution in CorePoint supplies that proof.
    """
    return _validate_design_space(data, point, strict=True)


class CorePoint:
    """Mean of a distribution with positive mass on every legal binary design.

    Open each facility independently with probability 1/2; conditionally choose
    a locker level uniformly and each permitted active-endpoint link with
    probability 1/2. Each OD proposes uniformly one of L^2 base pairs or no
    assignment, accepting a pair if its endpoints open. This full-support mean
    belongs to ri(conv(X)). Updates retain positive initial mass indefinitely.
    """

    def __init__(self, data):
        self.data = data
        point = {name: np.zeros(shape) for name, shape in shapes(data).items()}
        point["facility"][:] = 0.5
        point["capacity_level"][:] = 0.5 / data.slot_counts.shape[1]
        endpoint_probability = np.full((data.n_l, data.n_l), 0.25)
        np.fill_diagonal(endpoint_probability, 0.5)
        point["assignment"][:] = endpoint_probability / (data.n_l**2 + 1)
        link_probability = np.full((data.n_nodes, data.n_nodes), 0.125)
        np.fill_diagonal(link_probability, 0.25)
        point["airline"][:] = link_probability * data.allowed_links
        validate_core_point(data, point)
        self._initial = {name: value.copy() for name, value in point.items()}
        self._point = point

    @property
    def point(self):
        return {name: value.copy() for name, value in self._point.items()}

    def update(self, design):
        _validate_design_space(self.data, design, strict=False)
        for name, value in design.items():
            if name in self._point and np.any(
                (np.asarray(value) != 0) & (np.asarray(value) != 1)
            ):
                raise ValueError("Core updates require a feasible binary design")
        self._point = {
            name: 0.5 * self._point[name]
            + 0.49 * np.asarray(design[name])
            + 0.01 * initial
            for name, initial in self._initial.items()
        }
        validate_core_point(self.data, self._point)


def solve_pareto_cut(
    built,
    ordinary_cut,
    anchor_design,
    core_point,
    anchor_value,
    *,
    time_limit=None,
    threads=1,
    seed=43,
    stop_requested=None,
    face_tolerance=None,
    timings=None,
):
    """Select a certified cut and optionally accumulate exclusive timing leaves.

    ``wall`` is the total. Its disjoint leaves are the four ``dual`` phases,
    ``pricing``, and ``validation``. Nested ``solver_runtime`` is diagnostic.
    Validation includes anchor/core checks, cut certification, and control
    overhead. All early returns and propagated exceptions retain partial times.
    """
    started = time.perf_counter()
    if timings is not None:
        details = timings.setdefault("dual", {})
        for key in (*DUAL_TIMING_PHASES, "solver_runtime"):
            details.setdefault(key, 0.0)
        for key in ("pricing", "validation", "wall"):
            timings.setdefault(key, 0.0)
        before_dual = sum(details[key] for key in DUAL_TIMING_PHASES)
        before_pricing = timings["pricing"]
    result = None
    try:
        result = _solve_pareto_cut_impl(
            built,
            ordinary_cut,
            anchor_design,
            core_point,
            anchor_value,
            time_limit=time_limit,
            threads=threads,
            seed=seed,
            stop_requested=stop_requested,
            face_tolerance=face_tolerance,
            _started=started,
            _timings=timings,
        )
        return result
    finally:
        elapsed = time.perf_counter() - started
        if result is not None:
            result["wall"] = elapsed
        if timings is not None:
            dual_elapsed = (
                sum(timings["dual"][key] for key in DUAL_TIMING_PHASES) - before_dual
            )
            pricing_elapsed = timings["pricing"] - before_pricing
            timings["wall"] += elapsed
            timings["validation"] += max(0.0, elapsed - dual_elapsed - pricing_elapsed)


def _solve_pareto_cut_impl(
    built,
    ordinary_cut,
    anchor_design,
    core_point,
    anchor_value,
    *,
    time_limit=None,
    threads=1,
    seed=43,
    stop_requested=None,
    face_tolerance=None,
    _started=None,
    _timings=None,
):
    """Return an accepted complete-catalogue auxiliary cut or the original cut.

    The auxiliary includes every omitted path's dual stationarity row and
    finite-upper-bound price. A fresh complete scan independently checks it;
    a certificate for the ordinary dual is never reused for the auxiliary.
    """
    started = _started
    evidence = dict(pricing_complete=False, auxiliary_status=None)

    def finish(reason, candidate=None):
        return dict(
            cut=ordinary_cut if candidate is None else candidate,
            accepted=candidate is not None,
            certified=candidate is not None,
            reason=reason,
            wall=time.perf_counter() - started,
            **evidence,
        )

    def remaining():
        if stop_requested and stop_requested():
            raise BuildStopped("user_stop")
        if time_limit is None:
            return None
        left = time_limit - (time.perf_counter() - started)
        if left <= 0:
            raise BuildStopped("time_limit")
        return left

    if (
        ordinary_cut is None
        or not ordinary_cut.get("pricing_complete")
        or not built.paths.complete
    ):
        return finish("uncertified_anchor")
    validate_core_point(built.data, core_point)
    _validate_design_space(built.data, anchor_design, strict=False)
    for (name, key), row in built.constraints["design_fix"].items():
        if float(anchor_design[name][key]) != float(row.RHS):
            raise ValueError("The anchor design differs from the fixed-design primal")
    if not np.isfinite(anchor_value):
        raise ValueError("anchor_value must be finite")
    scale = 1 + abs(float(anchor_value))
    face_tolerance = 1e-6 * scale if face_tolerance is None else float(face_tolerance)
    if not np.isfinite(face_tolerance) or face_tolerance < 0:
        raise ValueError("face_tolerance must be finite and nonnegative")
    verification_tolerance = 2e-7 * scale
    ordinary_gap = float(anchor_value) - cut_value(ordinary_cut, anchor_design)
    evidence.update(
        face_tolerance=face_tolerance,
        verification_tolerance=verification_tolerance,
        ordinary_anchor_gap=ordinary_gap,
    )
    if (
        ordinary_gap < -verification_tolerance
        or ordinary_gap > face_tolerance + verification_tolerance
    ):
        return finish("ordinary_anchor_not_tight")
    try:
        budget = remaining()
        evidence["auxiliary_called"] = True
        dual = solve_dual(
            built,
            core_point=core_point,
            anchor_value=float(anchor_value),
            face_tolerance=face_tolerance,
            time_limit=budget,
            threads=threads,
            seed=seed,
            stop_requested=stop_requested,
            timings=None if _timings is None else _timings["dual"],
        )
        if dual is None:
            return finish("auxiliary_unavailable")
        evidence["auxiliary_status"] = dual["status"]
        if dual["status"] != GRB.OPTIMAL:
            return finish("auxiliary_not_optimal")
        pricing_budget = remaining()
        pricing_started = time.perf_counter()
        try:
            pricing = price_catalogue(
                built, dual, time_limit=pricing_budget, stop_requested=stop_requested
            )
        finally:
            if _timings is not None:
                _timings["pricing"] += time.perf_counter() - pricing_started
        evidence.update(
            pricing_complete=bool(pricing.get("complete")),
            scanned=pricing.get("scanned", 0),
            minimum_reduced_cost=pricing.get("minimum_reduced_cost"),
        )
        candidate = certified_cut(dual, pricing)
        if candidate is None:
            return finish(
                "incomplete_pricing"
                if not pricing.get("complete")
                else "pricing_violation"
            )
        anchor = cut_value(candidate, anchor_design)
        core_value = cut_value(candidate, core_point)
        gap = float(anchor_value) - anchor
        gain = core_value - cut_value(ordinary_cut, core_point)
        core_loss = float(dual["raw_core_objective"]) - core_value
        evidence.update(
            anchor=anchor,
            anchor_gap=gap,
            core_gain=gain,
            core_value=core_value,
            raw_core_objective=float(dual["raw_core_objective"]),
            raw_anchor_objective=float(dual["raw_anchor_objective"]),
            core_optimality_loss=core_loss,
        )
        if not np.isfinite([anchor, core_value, gap, gain, core_loss]).all():
            return finish("nonfinite_certificate")
        if (
            gap < -verification_tolerance
            or gap > face_tolerance + verification_tolerance
        ):
            return finish("anchor_check_failed")
        if gain < -verification_tolerance:
            return finish("core_check_failed")
        if abs(core_loss) > verification_tolerance:
            return finish("core_optimality_check_failed")
        candidate.update(
            anchor=anchor,
            duality_gap=gap,
            cut_type="pareto",
            pareto_face_tolerance=face_tolerance,
        )
        return finish("optimal_face", candidate)
    except BuildStopped as exc:
        return finish(exc.reason)
    except GurobiError as exc:
        evidence["solver_error_code"] = exc.errno
        return finish("auxiliary_solver_error")
