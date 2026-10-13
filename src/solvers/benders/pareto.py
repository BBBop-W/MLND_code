"""Magnanti--Wong cut selection on the common, all-assignment recourse dual.

The anchor keeps the current optimal face; the objective selects its strongest
support at a relative-interior core point. Every auxiliary solve uses universal
pricing. Incomplete solves retain the previously certified ordinary cut.
"""

import time

import gurobipy as gp
import numpy as np
from gurobipy import GRB

from src.model.design import fixed_design
from src.model.network import incidence, levels, network, price_matrix, route_key
from src.solvers.benders.cuts import _idle_conjugate
from src.solvers.pricing.backend import PricingStopped, price_design
from src.solvers.pricing.routes import trace_empty, trace_mission
from src.solvers.settings import settings


class CorePoint:
    """Expectation of a distribution supporting every feasible binary design.

    Facilities open independently; an open locker chooses a capacity level
    uniformly. Links activate conditionally on their endpoints. Each OD either
    stays unassigned or proposes a uniformly chosen base pair, accepting it if
    both endpoints open. Fixed facilities/levels restrict the distribution to
    that face. Keeping a small initial weight protects numerical interiority.
    """

    def __init__(self, data, fixed_facility=None, fixed_capacity_level=None):
        L, H, K, D = (data[k] for k in ("n_l", "n_h", "n_k", "n_od"))
        net = network(data)
        _, capacity, _ = levels(data)
        x = np.full(L + H + K, 0.5)
        if fixed_facility is not None:
            x[:] = np.asarray(fixed_facility, float)
        z = np.broadcast_to(
            x[L : L + H, None] / capacity.shape[1], capacity.shape
        ).copy()
        if fixed_capacity_level is not None:
            z[:] = np.asarray(fixed_capacity_level, float)
            x[L : L + H] = z.sum(axis=1)
        probability = x[:L, None] * x[None, :L]
        np.fill_diagonal(probability, x[:L])
        assignment = np.broadcast_to(0.5 * probability / L**2, (D, L, L)).copy()
        y = 0.5 * x[net["uv"][:, 0]] * x[net["uv"][:, 1]]
        self._initial = (x, z, assignment, y)
        self._point = tuple(v.copy() for v in self._initial)

    @property
    def point(self):
        return tuple(v.copy() for v in self._point)

    def update(self, *design):
        self._point = tuple(
            0.5 * old + 0.49 * np.asarray(new) + 0.01 * initial
            for old, new, initial in zip(self._point, design, self._initial)
        )


def cut_value(cut, point):
    x, z, assignment, y = point
    return float(
        cut["cut_intercept"]
        + cut["cut_facility"] @ x
        + np.sum(cut["cut_level"] * z)
        + np.sum(cut["cut_B"] * assignment)
        + cut["cut_airline"] @ y
    )


def _complete_cut(data, design, values, priced):
    """Repair all dual column constraints, then evaluate idle conjugates exactly."""
    a, u, eta, pi, sigma = (values[k] for k in ("a", "u", "eta", "pi", "sigma"))
    pi = pi.copy()
    L = data["n_l"]
    scale = 1.0
    empty = priced["empty_pricing_cost"]
    for ell in range(L):
        for m in range(L):
            difference = pi[m] - pi[ell]
            if difference > 0 and np.isfinite(empty[ell, m]):
                scale = min(scale, max(0.0, float(empty[ell, m])) / difference)
    pi *= scale
    b = np.maximum(a, 0.0)
    g = np.maximum(
        0.0,
        a[:, :, None] - pi[None, :, None] + pi[None, None, :] - priced["pricing_cost"],
    )
    demand = np.asarray(data["demand"])
    coefficient = demand @ a + sigma
    cut = dict(
        cut_facility=np.r_[-sigma, np.zeros(data["n_h"] + data["n_k"])],
        cut_level=-design.capacity * u[:, None],
        cut_B=demand[:, None, None] * (b[:, :, None] - g),
        cut_airline=-design.net["upper"] * eta,
        cut_intercept=float(
            design.baseline_cost
            + sum(_idle_conjugate(k, data["fleet_cost"]) for k in coefficient)
            - demand @ b.sum(axis=1)
        ),
    )
    norm = (
        1
        + abs(cut["cut_intercept"])
        + sum(
            np.abs(cut[k]).sum()
            for k in ("cut_facility", "cut_level", "cut_B", "cut_airline")
        )
    )
    margin = 64 * np.finfo(float).eps * norm + 1e-8 * (1 + abs(design.baseline_cost))
    cut["cut_intercept"] -= margin
    cut.update(cut_correction=margin, dual_potential_scale=scale)
    return cut


def strengthen(
    data,
    design,
    core,
    ordinary,
    *,
    time_limit=30.0,
    threads=1,
    seed=43,
    pricing="label",
    stop_requested=None,
    max_iterations=300,
):
    """Return a fully priced optimal-face cut, or the unchanged ordinary cut.

    ``certified`` denotes auxiliary optimality and universal pricing to the
    reported numerical tolerances. The anchor tolerance covers the ordinary
    primal/dual gap and a small feasibility margin; it is recorded explicitly.
    """
    started = time.perf_counter()
    deadline = float("inf") if time_limit is None else started + time_limit
    stats = dict(cg_iterations=0, pricing_calls=0, pricing_wall=0.0)

    def finish(reason, cut=None, **details):
        result = dict(ordinary if cut is None else cut)
        result.update(
            accepted=cut is not None,
            certified=cut is not None,
            reason=reason,
            **stats,
            wall=time.perf_counter() - started,
            **details,
        )
        return result

    def check_stop():
        if stop_requested is not None and stop_requested():
            raise PricingStopped("user_stop")
        if time.perf_counter() >= deadline:
            raise PricingStopped("time_limit")

    def price(**kwargs):
        check_stop()
        tick = time.perf_counter()
        stats["pricing_calls"] += 1
        try:
            answer = price_design(
                data,
                backend=pricing,
                threads=threads,
                seed=seed,
                time_limit=None if time_limit is None else deadline - tick,
                stop_requested=stop_requested,
                **kwargs,
            )
        finally:
            stats["pricing_wall"] += time.perf_counter() - tick
        if not (answer.get("complete") and answer.get("full_graph")):
            raise PricingStopped("incomplete_pricing")
        return answer

    if not (ordinary.get("cg_complete") and ordinary.get("cut_valid")):
        return finish("uncertified_anchor")
    model = None
    try:
        check_stop()
        anchor = fixed_design(data, *design)
        L, H, D = (data[k] for k in ("n_l", "n_h", "n_od"))
        net = anchor.net
        demand = np.asarray(data["demand"])
        objective = float(ordinary["objective"])
        anchor_tolerance = max(
            1e-7 * (1 + abs(objective)),
            abs(objective - cut_value(ordinary, design)) + 1e-8 * (1 + abs(objective)),
        )
        # Seed all groups: the auxiliary has prices for inactive assignments too.
        initial = price()
        model = gp.Model("pareto_dual_column_generation")
        settings(model, time_limit, threads, seed)
        # The nearly tight optimal face is ill-conditioned at closed designs.
        # Check the repaired support independently below instead of forcing
        # excessive barrier accuracy on this auxiliary model.
        model.Params.BarQCPConvTol = 1e-7
        a = model.addMVar((D, L), lb=-GRB.INFINITY, name="source_price")
        b = model.addMVar((D, L), name="source_lower_price")
        g = model.addMVar((D, L, L), name="assignment_price")
        u = model.addMVar(H, name="locker_price")
        eta = model.addMVar(net["count"], name="link_price")
        pi = model.addMVar(L, lb=-GRB.INFINITY, name="balance_price")
        sigma = model.addMVar(L, name="rho_cap_price")
        model.addConstr(b >= a)
        model.addConstr(pi[0] == 0)
        cones = []
        for ell in range(L):
            t = model.addVar(lb=0)
            s = model.addVar(lb=-GRB.INFINITY)
            z = model.addVar(lb=-GRB.INFINITY)
            model.addQConstr(s * s + z * z <= t * t)
            model.addConstr(t + z <= data["fleet_cost"])
            model.addConstr(demand @ a[:, ell] + sigma[ell] + t - z >= 0)
            cones.append((t, s, z))

        def value_at(point):
            x, z, assignment, y = point
            return (
                anchor.baseline_cost
                - (anchor.capacity * z).sum(axis=1) @ u
                - (net["upper"] * y) @ eta
                - x[:L] @ sigma
                - gp.quicksum(2 * (t + s) for t, s, _ in cones)
                - (demand[:, None] * (1 - assignment.sum(axis=2)) * b).sum()
                - (demand[:, None, None] * assignment * g).sum()
            )

        model.addConstr(
            value_at(design) >= objective - anchor_tolerance, name="optimal_face"
        )
        model.setObjective(value_at(core), GRB.MAXIMIZE)
        keys, empty_keys = set(), set()

        def insert_service(od, ell, m, trace):
            check_stop()
            key = (int(od), int(ell), int(m), route_key(trace))
            if key in keys:
                return 0
            charge = gp.quicksum(k * eta[e].item() for e, k in incidence(net, trace))
            model.addConstr(
                a[od, ell]
                - g[od, ell, m]
                - pi[ell]
                + pi[m]
                - u[trace["i"]]
                - u[trace["j"]]
                - charge
                <= trace["cost"]
            )
            keys.add(key)
            return 1

        def insert_empty(ell, m, trace):
            check_stop()
            key = (int(ell), int(m), route_key(trace))
            if key in empty_keys:
                return 0
            charge = gp.quicksum(k * eta[e].item() for e, k in incidence(net, trace))
            model.addConstr(trace["cost"] + charge + pi[ell] - pi[m] >= 0)
            empty_keys.add(key)
            return 1

        def insert_oracle(oracle, groups, pairs):
            added = 0
            for od, ell, m in groups:
                added += insert_service(od, ell, m, trace_mission(oracle, od, ell, m))
            for ell, m in pairs:
                added += insert_empty(ell, m, trace_empty(oracle, ell, m))
            return added

        insert_oracle(
            initial,
            np.argwhere(np.isfinite(initial["pricing_cost"])),
            np.argwhere(np.isfinite(initial["empty_pricing_cost"])),
        )
        for column in ordinary.get("pool", {}).get("missions", []):
            insert_service(column["od"], column["l"], column["m"], column["trace"])
        for column in ordinary.get("pool", {}).get("relocations", []):
            insert_empty(column["l"], column["m"], column["trace"])

        def callback(mm, where):
            if stop_requested is not None and stop_requested():
                mm.terminate()

        for _ in range(max_iterations):
            check_stop()
            model.Params.TimeLimit = (
                max(0.01, deadline - time.perf_counter())
                if time_limit is not None
                else GRB.INFINITY
            )
            model.optimize(callback)
            stats["cg_iterations"] += 1
            check_stop()
            if model.Status != GRB.OPTIMAL:
                return finish("auxiliary_status_" + str(model.Status))
            values = dict(
                a=a.X.copy(),
                u=np.maximum(0, u.X),
                eta=np.maximum(0, eta.X),
                pi=pi.X.copy(),
                sigma=np.maximum(0, sigma.X),
            )
            priced = price(
                hub_prices=values["u"], arc_prices=price_matrix(net, values["eta"])
            )
            potential = values["pi"]
            violations = (
                values["a"][:, :, None]
                - g.X
                - potential[None, :, None]
                + potential[None, None, :]
                - priced["pricing_cost"]
            )
            empty_slack = (
                priced["empty_pricing_cost"] + potential[:, None] - potential[None, :]
            )
            groups = np.argwhere(violations > 1e-7)
            pairs = np.argwhere(empty_slack < -1e-7)
            if len(groups) or len(pairs):
                if not insert_oracle(priced, groups, pairs):
                    return finish("pricing_stalled")
                continue
            cut = _complete_cut(data, anchor, values, priced)
            bound = cut_value(cut, design)
            error = objective - bound
            gain = cut_value(cut, core) - cut_value(ordinary, core)
            tolerance = 2e-6 * (1 + abs(objective))
            finite = all(np.isfinite(v).all() for v in cut.values())
            details = dict(
                anchor_tolerance=anchor_tolerance,
                tightness_error=error,
                core_gain=gain,
                auxiliary_objective=float(model.ObjVal),
                core_optimality_loss=float(model.ObjVal) - cut_value(cut, core),
                pricing_complete=True,
            )
            if not finite or error < -tolerance or error > tolerance:
                return finish("anchor_check_failed", **details)
            if gain < -1e-7 * (1 + abs(cut_value(ordinary, core))):
                return finish("core_check_failed", **details)
            if abs(details["core_optimality_loss"]) > 2e-6 * (
                1 + abs(float(model.ObjVal))
            ):
                return finish("core_optimality_check_failed", **details)
            cut.update(bound=bound, cut_valid=True, duality_gap=error)
            return finish("optimal_face", cut, **details)
        return finish("iteration_limit")
    except PricingStopped as error:
        return finish(str(error))
    finally:
        if model is not None:
            model.dispose()
