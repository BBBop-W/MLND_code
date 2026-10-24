"""Explicit conic duals and global cuts for the fixed lifted matrix.

The separately solved dual includes variable bounds and all base-cone duals.
Numerical stationarity residuals are bounded using finite, design-independent
primal bounds. A cut is released only after scanning the complete catalogue.
"""

from __future__ import annotations

import time
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from .formulation import BuildStopped, path_column, shapes, setup


DUAL_TIMING_PHASES = ("build", "optimize", "postprocess", "dispose")


def solve_dual(
    built,
    *,
    time_limit=None,
    threads=1,
    seed=43,
    stop_requested=None,
    core_point=None,
    anchor_value=None,
    face_tolerance=None,
    timings=None,
):
    """Solve the dual, optionally accumulating mutually exclusive wall times.

    ``solver_runtime`` is a separate backend diagnostic and must not be added
    to build/optimize/postprocess/dispose. Exceptions retain partial timings.
    """
    started = time.perf_counter()
    if timings is not None:
        for key in (*DUAL_TIMING_PHASES, "solver_runtime"):
            timings.setdefault(key, 0.0)
    active, phase_started = "build", started

    def phase(name):
        nonlocal active, phase_started
        now = time.perf_counter()
        if timings is not None and active is not None:
            timings[active] += now - phase_started
        active, phase_started = name, now

    try:
        return _solve_dual_impl(
            built,
            time_limit=time_limit,
            threads=threads,
            seed=seed,
            stop_requested=stop_requested,
            core_point=core_point,
            anchor_value=anchor_value,
            face_tolerance=face_tolerance,
            _started=started,
            _phase=phase,
            _timings=timings,
        )
    finally:
        phase(None)


def _solve_dual_impl(
    built,
    *,
    time_limit=None,
    threads=1,
    seed=43,
    stop_requested=None,
    core_point=None,
    anchor_value=None,
    face_tolerance=None,
    _started=None,
    _phase=None,
    _timings=None,
):
    """Solve the ordinary dual or its complete-catalogue optimal-face auxiliary.

    The auxiliary replaces only the fixed-design RHS in the objective. Missing
    path columns receive their own stationarity rows and finite-bound prices;
    their bound prices are essential because these bounds occur in the primal.
    """
    started = _started

    def check():
        if stop_requested and stop_requested():
            raise BuildStopped("user_stop")
        if time_limit is not None and time.perf_counter() - started >= time_limit:
            raise BuildStopped("time_limit")

    check()
    primal = built.model
    primal.update()
    variables, rows = primal.getVars(), primal.getConstrs()
    matrix = primal.getA().tocsc()
    objective = np.array([v.Obj for v in variables])
    lower, upper = (
        np.array([getattr(v, name) for v in variables]) for name in ("LB", "UB")
    )
    if np.any(lower < 0) or np.any(upper >= GRB.INFINITY):
        raise ValueError(
            "Every lifted primal variable needs a finite nonnegative bound"
        )
    rhs = np.array([r.RHS for r in rows])
    core_rhs = rhs.copy()
    if core_point is not None:
        if anchor_value is None or face_tolerance is None:
            raise ValueError("The core objective requires an anchor and face tolerance")
        for name, shape in shapes(built.data).items():
            values = np.asarray(core_point[name], dtype=float)
            if values.shape != shape or not np.isfinite(values).all():
                raise ValueError(f"Invalid core_point.{name}")
        for (name, key), row in built.constraints["design_fix"].items():
            core_rhs[row.index] = core_point[name][key]
        if not built.constraints["design_fix"]:
            raise ValueError("A Pareto auxiliary requires a fixed-design primal")
        if (
            not np.isfinite(anchor_value)
            or not np.isfinite(face_tolerance)
            or face_tolerance < 0
        ):
            raise ValueError("Invalid anchor or face tolerance")
    check()
    dual = gp.Model("consecutive_dual")
    setup(dual, threads, seed)
    try:
        pi = [
            dual.addVar(
                lb=0 if r.Sense == ">" else -GRB.INFINITY,
                ub=0 if r.Sense == "<" else GRB.INFINITY,
                name=f"pi_{i}",
            )
            for i, r in enumerate(rows)
        ]
        ub = dual.addVars(len(variables), name="upper")
        lb = {
            i: dual.addVar(name=f"lower_{i}")
            for i in range(len(variables))
            if lower[i] > 0
        }
        cone_terms, cone_vars = {}, []
        for j, (up, down, one) in enumerate(built.cones):
            su, sd = dual.addVar(name=f"su_{j}"), dual.addVar(name=f"sd_{j}")
            so = dual.addVar(lb=-GRB.INFINITY, name=f"so_{j}")
            dual.addQConstr(so * so <= 4 * su * sd)
            for source, target in ((up, su), (down, sd), (one, so)):
                cone_terms[source.index] = target
            cone_vars.append((up.index, down.index, one.index, su, sd, so))
        for column in range(len(variables)):
            check()
            begin, end = matrix.indptr[column : column + 2]
            expr = (
                gp.quicksum(
                    float(matrix.data[k]) * pi[int(matrix.indices[k])]
                    for k in range(begin, end)
                )
                - ub[column]
            )
            if column in lb:
                expr += lb[column]
            if column in cone_terms:
                expr += cone_terms[column]
            dual.addConstr(expr <= objective[column])
        missing_upper, missing_columns = {}, {}
        if core_point is not None:
            for is_empty, paths, selected in (
                (False, built.paths.service_paths, set(built.service_indices)),
                (True, built.paths.empty_paths, set(built.empty_indices)),
            ):
                for index, path in enumerate(paths):
                    check()
                    if index in selected:
                        continue
                    key = is_empty, index
                    column = tuple(
                        (row.index, value)
                        for row, value in path_column(built, path, empty=is_empty)
                    )
                    missing_columns[key] = (column, float(path.cost))
                    missing_upper[key] = dual.addVar(
                        name=f"missing_upper_{int(is_empty)}_{index}"
                    )
                    dual.addConstr(
                        gp.quicksum(value * pi[row] for row, value in column)
                        - missing_upper[key]
                        <= path.cost,
                        name=f"missing_column_{int(is_empty)}_{index}",
                    )
        total = float(built.data.demand.sum())
        bound_objective = (
            -gp.quicksum(upper[i] * ub[i] for i in range(len(variables)))
            + gp.quicksum(lower[i] * value for i, value in lb.items())
            - total * gp.quicksum(missing_upper.values())
        )
        anchor_objective = (
            gp.quicksum(rhs[i] * pi[i] for i in range(len(rows))) + bound_objective
        )
        core_objective = (
            gp.quicksum(core_rhs[i] * pi[i] for i in range(len(rows))) + bound_objective
        )
        if core_point is not None:
            dual.addConstr(
                anchor_objective >= anchor_value - face_tolerance, name="optimal_face"
            )
        dual.setObjective(
            core_objective,
            GRB.MAXIMIZE,
        )
        check()
        if time_limit is not None:
            dual.Params.TimeLimit = max(
                1e-6, time_limit - (time.perf_counter() - started)
            )

        def callback(model, where):
            if stop_requested and stop_requested():
                model.terminate()

        _phase("optimize")
        try:
            dual.optimize(callback if stop_requested else None)
        finally:
            _phase("postprocess")
            if _timings is not None:
                try:
                    _timings["solver_runtime"] += float(dual.Runtime)
                except (AttributeError, gp.GurobiError):
                    pass
        if not dual.SolCount:
            if dual.Status == GRB.TIME_LIMIT:
                raise BuildStopped("time_limit")
            if dual.Status == GRB.INTERRUPTED:
                raise BuildStopped("user_stop")
            return None
        prices = np.array([v.X for v in pi])
        for i, row in enumerate(rows):
            if row.Sense == ">":
                prices[i] = max(0.0, prices[i])
            elif row.Sense == "<":
                prices[i] = min(0.0, prices[i])
        upper_prices = np.maximum(
            0.0, np.array([ub[i].X for i in range(len(variables))])
        )
        lower_prices, cone = np.zeros(len(variables)), np.zeros(len(variables))
        for i, value in lb.items():
            lower_prices[i] = max(0.0, value.X)
        for ui, di, oi, su, sd, so in cone_vars:
            u, d, o = max(0.0, su.X), max(0.0, sd.X), so.X
            if 4 * u * d < o * o:
                if u >= d and u > 0:
                    d = float(np.nextafter(o * o / (4 * u), np.inf))
                elif d > 0:
                    u = float(np.nextafter(o * o / (4 * d), np.inf))
                else:
                    u = d = float(np.nextafter(abs(o) / 2, np.inf))
            cone[ui] += u
            cone[di] += d
            cone[oi] += o
        residual = np.maximum(
            0.0,
            np.asarray(matrix.T @ prices).ravel()
            - upper_prices
            + lower_prices
            + cone
            - objective,
        )
        missing_prices = {}
        for key, variable in missing_upper.items():
            column, cost = missing_columns[key]
            value = max(0.0, float(variable.X))
            # Repair the omitted stationarity rows after row-sign projection.
            value = max(
                value,
                sum(coefficient * prices[row] for row, coefficient in column) - cost,
            )
            missing_prices[key] = (
                float(np.nextafter(value, np.inf)) if value > 0 else 0.0
            )
        bound_terms = float(
            -upper @ upper_prices
            + lower @ lower_prices
            - total * sum(missing_prices.values())
        )
        correction = float(upper @ residual)
        coefficients = {
            name: np.zeros(shape) for name, shape in shapes(built.data).items()
        }
        intercept = float(rhs @ prices) + bound_terms
        for (name, key), row in built.constraints["design_fix"].items():
            coefficients[name][key] = prices[row.index]
            intercept -= row.RHS * prices[row.index]
        margin = 1e-8 * (1 + float(built.data.demand @ built.data.ground_cost))
        margin += (
            128
            * np.finfo(float).eps
            * (
                1
                + abs(float(rhs @ prices))
                + abs(bound_terms)
                + sum(np.abs(a).sum() for a in coefficients.values())
            )
        )
        return dict(
            intercept=intercept - correction - margin,
            coefficients=coefficients,
            correction=correction + margin,
            stationarity_error=float(residual.max(initial=0)),
            raw_dual_objective=float(dual.ObjVal),
            raw_anchor_objective=float(anchor_objective.getValue()),
            raw_core_objective=float(core_objective.getValue()),
            auxiliary_complete_columns=core_point is not None,
            auxiliary_missing_columns=len(missing_upper),
            dual_cones=len(cone_vars),
            status=int(dual.Status),
            _pi=prices,
            _missing_upper_prices=missing_prices,
            row_prices={row.ConstrName: float(prices[i]) for i, row in enumerate(rows)},
        )
    finally:
        _phase("dispose")
        dual.dispose()


def cut_value(cut, design):
    return float(
        cut["intercept"]
        + sum(
            np.sum(values * design[name])
            for name, values in cut["coefficients"].items()
        )
    )


def price_catalogue(
    built, dual, *, tolerance=1e-7, stop_requested=None, time_limit=None
):
    """Test every missing scalar column, irrespective of the current design."""
    started = time.perf_counter()
    service, empty, minimum, correction, scanned = [], [], 0.0, 0.0, 0
    prices = dual["_pi"]
    for is_empty, paths, selected, target in (
        (False, built.paths.service_paths, set(built.service_indices), service),
        (True, built.paths.empty_paths, set(built.empty_indices), empty),
    ):
        for index, path in enumerate(paths):
            if stop_requested and stop_requested():
                return dict(
                    complete=False,
                    reason="user_stop",
                    service=service,
                    empty=empty,
                    scanned=scanned,
                )
            if time_limit is not None and time.perf_counter() - started >= time_limit:
                return dict(
                    complete=False,
                    reason="time_limit",
                    service=service,
                    empty=empty,
                    scanned=scanned,
                )
            scanned += 1
            if index in selected:
                continue
            reduced = float(
                path.cost
                + dual.get("_missing_upper_prices", {}).get((is_empty, index), 0.0)
                - sum(
                    value * prices[row.index]
                    for row, value in path_column(built, path, empty=is_empty)
                )
            )
            minimum = min(minimum, reduced)
            correction += max(0.0, -reduced) * float(built.data.demand.sum())
            if reduced < -tolerance:
                target.append((reduced, index))
    return dict(
        complete=bool(built.paths.complete),
        reason="complete",
        service=sorted(service),
        empty=sorted(empty),
        minimum_reduced_cost=minimum,
        missing_stationarity_correction=correction,
        scanned=scanned,
    )


def certified_cut(dual, pricing):
    """No partial or negative-reduced-cost pricing result can issue a cut."""
    if (
        dual is None
        or not pricing.get("complete")
        or pricing["service"]
        or pricing["empty"]
    ):
        return None
    cut = {key: value for key, value in dual.items() if not key.startswith("_")}
    correction = pricing.get("missing_stationarity_correction", 0.0)
    cut["intercept"] -= correction
    cut["correction"] += correction
    cut["pricing_complete"] = True
    return cut
