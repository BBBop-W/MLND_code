"""Gurobi shortest-path pricing on the exact resource-state graph.

The graph describes partial routes; it does not enumerate complete columns.
Each requested OD/base-pair group is solved as a unit-flow linear program.
Its network matrix is totally unimodular, so simplex yields a route.  A price
is returned only after every requested group and empty route is certified.
"""

from __future__ import annotations

import math
import time

import gurobipy as gp
import numpy as np
from gurobipy import GRB

from src.model.path_graph import build_empty, build_service, reconstruct
from src.solvers.pricing.inputs import _validated
from src.solvers.pricing.routes import (
    trace_empty as trace_empty,
    trace_mission as trace_mission,
)


class PricingStopped(RuntimeError):
    """Pricing ended without the complete prices needed for a valid cut."""


def _check_stop(deadline, stop_requested):
    if stop_requested is not None and stop_requested():
        raise PricingStopped("user_stop")
    if time.perf_counter() >= deadline:
        raise PricingStopped("time_limit")


def _path_edges(graph, values):
    """Read the integral simplex solution in source-to-sink order."""
    if np.max(np.abs(values - np.rint(values)), initial=0) > 1e-6:
        raise PricingStopped("Shortest-path LP returned a nonintegral solution")
    selected = np.flatnonzero(values > 0.5).tolist()
    outgoing = {}
    for index in selected:
        outgoing.setdefault(graph.edges[index].tail, []).append(index)
    current, path = graph.source, []
    while current != graph.sink:
        candidates = outgoing.get(current, [])
        if len(candidates) != 1 or len(path) >= len(selected):
            raise PricingStopped("Shortest-path solution has no unique traceback")
        index = candidates[0]
        path.append(index)
        current = graph.edges[index].head
    if len(path) != len(selected):
        raise PricingStopped("Shortest-path solution includes disconnected flow")
    return path


def _solve_graph(c, graph, env, deadline, stop_requested, threads, seed):
    _check_stop(deadline, stop_requested)
    if not graph.edges:
        return None, 0.0
    costs = np.asarray(
        [
            edge.operating_cost
            + c["fleet"] * edge.time
            + sum(c["hub_prices"][i] for i in edge.hub_visits)
            + sum(
                c["arc_prices"].ravel()[arc] * count for arc, count in edge.arc_counts
            )
            for edge in graph.edges
        ]
    )
    incoming, outgoing = graph.adjacency()

    with gp.Model("route_pricing", env=env) as model:
        model.Params.Method = 1
        model.Params.Threads = int(threads)
        model.Params.Seed = int(seed)
        model.Params.FeasibilityTol = 1e-9
        model.Params.OptimalityTol = 1e-9
        flow = model.addVars(len(graph.edges), ub=1.0, obj=costs.tolist(), name="flow")
        for node in range(graph.nodes):
            supply = int(node == graph.source) - int(node == graph.sink)
            if incoming[node] or outgoing[node] or supply:
                model.addConstr(
                    gp.quicksum(flow[e] for e in outgoing[node])
                    - gp.quicksum(flow[e] for e in incoming[node])
                    == supply
                )

        def stop_callback(active_model, where):
            if stop_requested is not None and stop_requested():
                active_model.terminate()

        _check_stop(deadline, stop_requested)
        if math.isfinite(deadline):
            model.Params.TimeLimit = max(1e-6, deadline - time.perf_counter())
        model.optimize(stop_callback if stop_requested is not None else None)
        _check_stop(deadline, stop_requested)
        if model.Status == GRB.INFEASIBLE:
            return None, float(model.Runtime)
        if model.Status == GRB.TIME_LIMIT:
            raise PricingStopped("time_limit")
        if model.Status != GRB.OPTIMAL:
            raise PricingStopped(f"Shortest-path LP ended with status {model.Status}")
        values = np.asarray([flow[e].X for e in range(len(graph.edges))])
        path = _path_edges(graph, values)
        trace = reconstruct(c, graph, path)
        baseline = c["ground"][trace["od"]] if "od" in trace else 0.0
        if not math.isclose(
            trace["pricing_cost"],
            model.ObjVal - baseline,
            rel_tol=1e-9,
            abs_tol=1e-7,
        ):
            raise AssertionError("Solver pricing and physical traceback disagree")
        return trace, float(model.Runtime)


def price_design(
    data,
    facility=None,
    hub_prices=None,
    arc_prices=None,
    groups=None,
    *,
    threads=1,
    seed=43,
    time_limit=math.inf,
    stop_requested=None,
):
    """Price requested service groups and every ordered empty base pair.

    ``complete`` means all OD/base pairs were requested; ``full_graph`` means
    all candidate facilities were available.  Timeouts and interruptions
    raise :class:`PricingStopped`, including during a universal pricing pass.
    No partial result is returned as an optimality certificate.
    """
    start = time.perf_counter()
    if threads < 1 or int(threads) != threads:
        raise ValueError("threads must be a positive integer")
    if not 0 <= int(seed) <= 2_000_000_000 or int(seed) != seed:
        raise ValueError("seed must be an integer from 0 to 2000000000")
    time_limit = math.inf if time_limit is None else float(time_limit)
    if math.isnan(time_limit) or time_limit < 0:
        raise ValueError("time_limit must be nonnegative")
    deadline = start + time_limit
    _check_stop(deadline, stop_requested)
    c = _validated(data, facility, hub_prices, arc_prices, groups)
    D, L, H = c["D"], c["L"], c["H"]
    shape = (D, L, L)
    price, actual = np.full(shape, np.inf), np.full(shape, np.inf)
    elapsed = np.zeros(shape)
    choices = np.full(shape + (5,), -1, dtype=np.int64)
    empty_price = np.full((L, L), np.inf)
    empty_cost = np.full((L, L), np.inf)
    empty_time = np.zeros((L, L))
    empty_choice = np.full((L, L), -2, dtype=np.int64)
    models, solver_wall, graph_edges = 0, 0.0, 0

    with gp.Env(empty=True) as env:
        env.setParam("OutputFlag", 0)
        env.start()
        for od, l, m in np.argwhere(c["selected"]):
            _check_stop(deadline, stop_requested)
            graph = build_service(
                c,
                int(od),
                int(l),
                int(m),
                check=lambda _: _check_stop(deadline, stop_requested),
            )
            graph_edges += len(graph.edges)
            trace, runtime = _solve_graph(
                c, graph, env, deadline, stop_requested, threads, seed
            )
            models += bool(graph.edges)
            solver_wall += runtime
            if trace is not None:
                price[od, l, m] = trace["pricing_cost"]
                actual[od, l, m] = trace["cost"]
                elapsed[od, l, m] = trace["time"]
                choices[od, l, m] = [trace["i"], trace["j"], *trace["swap_choices"]]
        for l in range(L):
            for m in range(L):
                if l == m:
                    continue
                _check_stop(deadline, stop_requested)
                graph = build_empty(
                    c, l, m, check=lambda _: _check_stop(deadline, stop_requested)
                )
                graph_edges += len(graph.edges)
                trace, runtime = _solve_graph(
                    c, graph, env, deadline, stop_requested, threads, seed
                )
                models += bool(graph.edges)
                solver_wall += runtime
                if trace is not None:
                    empty_price[l, m] = trace["pricing_cost"]
                    empty_cost[l, m] = trace["cost"]
                    empty_time[l, m] = trace["time"]
                    empty_choice[l, m] = trace["swap_choices"][0]

    c["choices"], c["empty_choice"] = choices, empty_choice
    return dict(
        cost=actual,
        pricing_cost=price,
        time=elapsed,
        i=choices[:, :, :, 0],
        j=choices[:, :, :, 1],
        empty_cost=empty_cost,
        empty_pricing_cost=empty_price,
        empty_time=empty_time,
        facility=c["facility"].tolist(),
        hub_prices=c["hub_prices"].tolist(),
        arc_prices=c["arc_prices"],
        arc_node_count=c["N"],
        complete=bool(c["selected"].all()),
        full_graph=bool(c["facility"].all()),
        groups_priced=int(c["selected"].sum()),
        implicit_basic_paths=int(D * L * L * H * (H - 1)),
        enumerated_complete_paths=0,
        pricing_backend="solver",
        solver_models=models,
        solver_wall=solver_wall,
        state_graph_edges=graph_edges,
        wall=time.perf_counter() - start,
        _context=c,
    )
