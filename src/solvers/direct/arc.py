"""Direct state-expanded arc-flow MISOCP for algorithm comparisons.

The graph contains physical flight edges and resource-feasible service states.
Continuous edge flows may split among paths, exactly as in the path formulation.
It does not create one variable per complete service path. Exact resource states
can still make this formulation large; the API does not screen OD/base pairs.
"""

from __future__ import annotations

from dataclasses import dataclass
import time

import gurobipy as gp
from gurobipy import GRB
import numpy as np

from src.model.network import facility_costs, levels, network, vector
from src.model.path_graph import build_empty, build_service, reconstruct
from src.solvers.pricing.inputs import _validated
from src.solvers.settings import settings, mip_bound


@dataclass
class ArcFlow:
    graph: object
    variables: object
    total: object


@dataclass
class ArcModel:
    model: object
    recourse: object
    continuous: bool
    context: dict
    flows: list[ArcFlow]
    x: object
    z: object
    alpha: object
    y: object
    rho: object
    idle: object
    fleet: object


class BuildStopped(Exception):
    """Graph construction was interrupted before optimization."""


class GraphSizeLimit(ValueError):
    """An explicitly requested arc count limit was reached; nothing is truncated."""


def build_model(
    data,
    *,
    threads=1,
    seed=43,
    mip_gap=1e-4,
    mip_gap_abs=1e-6,
    fixed_facility=None,
    fixed_capacity_level=None,
    fixed_assignment=None,
    fixed_airlines=None,
    stop_requested=None,
    time_limit=None,
    max_state_arcs=None,
):
    """Construct an equivalent arc model; supplied design variables are fixed."""
    c = _validated(data)
    deadline = (
        None if time_limit is None else time.perf_counter() + max(0.0, time_limit)
    )
    L, H, D = c["L"], c["H"], c["D"]
    net = network(data)
    sizes, capacity, levelcost = levels(data)
    fixed = [fixed_facility, fixed_capacity_level, fixed_assignment, fixed_airlines]
    continuous = all(value is not None for value in fixed)
    model = gp.Model("arc_flow")
    try:
        settings(model, None, threads, seed)
        if mip_gap is not None:
            model.Params.MIPGap = mip_gap
        if mip_gap_abs is not None:
            model.Params.MIPGapAbs = mip_gap_abs
        dtype = GRB.CONTINUOUS if continuous else GRB.BINARY
        x = model.addMVar(net["N"], lb=0, ub=1, vtype=dtype, name="x")
        z = model.addMVar(capacity.shape, lb=0, ub=1, vtype=dtype, name="z")
        alpha = model.addMVar((D, L, L), lb=0, ub=1, vtype=dtype, name="alpha")
        y = model.addMVar(net["count"], lb=0, ub=1, vtype=dtype, name="y")
        specs = [
            (x, fixed_facility, (net["N"],)),
            (z, fixed_capacity_level, capacity.shape),
            (alpha, fixed_assignment, (D, L, L)),
            (y, fixed_airlines, (net["count"],)),
        ]
        for variable, values, shape in specs:
            if values is None:
                continue
            values = (
                vector(data, values, net)
                if variable is y
                else np.asarray(values, float)
            )
            if (
                values.shape != shape
                or not np.isfinite(values).all()
                or np.any((values != 0) & (values != 1))
            ):
                raise ValueError(
                    "Fixed strategic variables must be binary with required dimensions"
                )
            variable.LB, variable.UB = values, values
        rho = model.addVars(L, lb=0, ub=1, name="rho")
        idle = model.addVars(L, lb=0, name="idle")
        fleet = model.addVar(lb=0, name="Gamma")
        model.addConstr(z.sum(axis=1) == x[L : L + H], name="capacity_choice")
        model.addConstr(alpha.sum(axis=(1, 2)) <= 1, name="one_assignment")
        model.addConstr(alpha.sum(axis=2) <= x[:L], name="source_open")
        model.addConstr(alpha.sum(axis=1) <= x[:L], name="destination_open")
        model.addConstr(y <= x[net["uv"][:, 0]], name="link_origin")
        model.addConstr(y <= x[net["uv"][:, 1]], name="link_destination")
        flows, groups = [], {}
        locker = [gp.LinExpr() for _ in range(H)]
        links = [gp.LinExpr() for _ in range(net["count"])]
        outgoing, incoming = ([gp.LinExpr() for _ in range(L)] for _ in range(2))
        workload, operating = gp.LinExpr(), gp.LinExpr()
        completed_arcs = 0

        def check(graph=None):
            if stop_requested and stop_requested():
                raise BuildStopped("user_stop")
            if deadline is not None and time.perf_counter() >= deadline:
                raise BuildStopped("time_limit")
            count = completed_arcs + (0 if graph is None else len(graph.edges))
            if max_state_arcs is not None and count > max_state_arcs:
                raise GraphSizeLimit(
                    f"State graph exceeds max_state_arcs={max_state_arcs}"
                )

        def add_graph(graph):
            nonlocal workload, operating
            k = len(flows)
            variables = model.addVars(len(graph.edges), lb=0, name=f"flow_{k}")
            ins, outs = graph.adjacency()
            total = gp.quicksum(variables[e] for e in outs[graph.source])
            for node in range(graph.nodes):
                if node not in (graph.source, graph.sink):
                    model.addConstr(
                        gp.quicksum(variables[e] for e in ins[node])
                        == gp.quicksum(variables[e] for e in outs[node]),
                        name=f"state_{k}_{node}",
                    )
            for e, edge in enumerate(graph.edges):
                workload += edge.time * variables[e]
                operating += edge.operating_cost * variables[e]
                for i in edge.hub_visits:
                    locker[i] += variables[e]
                for flat, count in edge.arc_counts:
                    links[int(net["lookup"][flat])] += count * variables[e]
            if graph.od is not None:
                operating -= float(data["ground"][graph.od]) * total
                groups[graph.od, graph.l, graph.m] = total
            outgoing[graph.l] += total
            incoming[graph.m] += total
            flows.append(ArcFlow(graph, variables, total))

        for l in range(L):
            for m in range(L):
                for od in range(D):
                    check()
                    graph = build_service(c, od, l, m, check=check)
                    check(graph)
                    add_graph(graph)
                    completed_arcs += len(graph.edges)
                if l != m:
                    graph = build_empty(c, l, m, check=check)
                    check(graph)
                    add_graph(graph)
                    completed_arcs += len(graph.edges)
        for od in range(D):
            check()
            h = float(data["demand"][od])
            for l in range(L):
                source = gp.quicksum(groups[od, l, m] for m in range(L))
                for m in range(L):
                    model.addConstr(
                        groups[od, l, m] <= h * alpha[od, l, m],
                        name=f"pair_{od}_{l}_{m}",
                    )
                model.addConstr(source <= h * rho[l], name=f"source_upper_{od}_{l}")
                model.addConstr(
                    h * rho[l] - source <= h * (1 - alpha[od, l, :].sum()),
                    name=f"source_lower_{od}_{l}",
                )
        for l in range(L):
            check()
            model.addConstr(rho[l] <= x[l], name=f"available_{l}")
            model.addConstr(outgoing[l] == incoming[l], name=f"balance_{l}")
            up = model.addVar(lb=0, name=f"cone_u_{l}")
            down = model.addVar(lb=0, name=f"cone_v_{l}")
            one = model.addVar(lb=1, ub=1, name=f"cone_one_{l}")
            model.addConstr(up == idle[l] + 1)
            model.addConstr(down == 1 - rho[l])
            model.addQConstr(one * one <= up * down, name=f"idle_cone_{l}")
        for i in range(H):
            check()
            model.addConstr(
                locker[i]
                <= gp.quicksum(
                    float(capacity[i, s]) * z[i, s].item() for s in range(len(sizes))
                ),
                name=f"locker_{i}",
            )
        for e in range(net["count"]):
            if e % 1024 == 0:
                check()
            model.addConstr(links[e] <= net["upper"][e] * y[e], name=f"link_{e}")
        model.addConstr(gp.quicksum(idle.values()) + workload <= fleet, name="fleet")
        baseline = float(np.asarray(data["demand"]) @ np.asarray(data["ground"]))
        recourse = baseline + data["fleet_cost"] * fleet + operating
        strategy = facility_costs(data) @ x + (levelcost * z).sum() + net["fees"] @ y
        model.setObjective(strategy + recourse)
        return ArcModel(
            model, recourse, continuous, c, flows, x, z, alpha, y, rho, idle, fleet
        )
    except Exception:
        model.dispose()
        raise


def decompose(c, graph, values, tolerance=1e-7):
    """Decompose DAG flow, bounding numerical loss on every edge and in total.

    Positive edges below the feasibility tolerance still belong to paths. Only
    a stranded numerical residual may be removed, by its bottleneck amount;
    removing an entire prefix could silently erase substantial valid flow.
    """
    values = np.asarray(values, float)
    if values.shape != (len(graph.edges),) or not np.isfinite(values).all():
        raise ValueError("Arc-flow values must be finite and match the graph")
    if np.min(values, initial=0.0) < -tolerance:
        raise ValueError("Arc-flow solution has negative flow beyond tolerance")
    remaining = np.maximum(values, 0.0).copy()
    reconstructed = np.zeros_like(remaining)
    incoming, outgoing = graph.adjacency()
    balance = np.asarray(
        [sum(values[e] for e in incoming[n]) - sum(values[e] for e in outgoing[n])
         for n in range(graph.nodes)]
    )
    internal = np.delete(balance, [graph.source, graph.sink])
    if (
        np.max(np.abs(internal), initial=0.0) > tolerance
        or abs(balance[graph.source] + balance[graph.sink]) > tolerance
    ):
        raise ValueError("Arc-flow solution does not conserve flow within tolerance")
    records = []
    discarded = 0.0
    while sum(remaining[e] for e in outgoing[graph.source]) > 0.0:
        node, path = graph.source, []
        while node != graph.sink:
            candidates = [e for e in outgoing[node] if remaining[e] > 0.0]
            if not candidates:
                amount = min((remaining[e] for e in path), default=float("inf"))
                if discarded + amount <= tolerance:
                    remaining[path] -= amount
                    discarded += amount
                    path = []
                    break
                raise ValueError(
                    "Arc-flow solution does not conserve flow within tolerance"
                )
            e = max(candidates, key=lambda index: remaining[index])
            path.append(e)
            node = graph.edges[e].head
        if not path:
            continue
        amount = min(remaining[e] for e in path)
        remaining[path] -= amount
        reconstructed[path] += amount
        trace = reconstruct(c, graph, path)
        record = dict(
            trace=trace,
            l=graph.l,
            m=graph.m,
            flow=float(amount),
            time=trace["time"],
            cost=trace["cost"],
        )
        if graph.od is not None:
            record.update(od=graph.od, i=trace["i"], j=trace["j"])
        records.append(record)
    source_flow = sum(values[e] for e in outgoing[graph.source])
    if (
        np.max(np.abs(values - reconstructed), initial=0.0) > tolerance
        or abs(source_flow - sum(row["flow"] for row in records)) > tolerance
    ):
        raise ValueError("Arc-flow decomposition lost flow beyond tolerance")
    return records


def solve(
    data,
    *,
    time_limit=60.0,
    threads=1,
    seed=43,
    mip_gap=1e-4,
    mip_gap_abs=1e-6,
    stop_requested=None,
    progress=None,
    max_state_arcs=None,
    **fixed_design,
):
    """Solve the arc model and expose the same result fields as ``direct.solve``."""
    started = time.perf_counter()
    result = dict(
        model_version="path_cqn_v1",
        method="arc_MISOCP",
        objective=None,
        recourse_objective=None,
        bound=None,
        gap=None,
    )
    try:
        bundle = build_model(
            data,
            threads=threads,
            seed=seed,
            mip_gap=mip_gap,
            mip_gap_abs=mip_gap_abs,
            stop_requested=stop_requested,
            time_limit=time_limit,
            max_state_arcs=max_state_arcs,
            **fixed_design,
        )
    except (BuildStopped, GraphSizeLimit) as stopped:
        if isinstance(stopped, GraphSizeLimit):
            return dict(
                result,
                status=GRB.SUBOPTIMAL,
                wall=time.perf_counter() - started,
                termination_reason="graph_limit",
                construction_complete=False,
            )
        reason = str(stopped)
        return dict(
            result,
            status=GRB.TIME_LIMIT if reason == "time_limit" else GRB.INTERRUPTED,
            wall=time.perf_counter() - started,
            termination_reason=reason,
            construction_complete=False,
        )
    model = bundle.model
    if time_limit is not None:
        model.Params.TimeLimit = max(0.01, time_limit - (time.perf_counter() - started))
    last_progress = [0.0]

    def callback(active, where):
        if stop_requested and stop_requested():
            active.terminate()
        elapsed = time.perf_counter() - started
        if progress and where == GRB.Callback.MIP and elapsed - last_progress[0] >= 1.0:
            progress(
                dict(
                    stage="arc",
                    wall=elapsed,
                    objective=active.cbGet(GRB.Callback.MIP_OBJBST),
                    bound=active.cbGet(GRB.Callback.MIP_OBJBND),
                )
            )
            last_progress[0] = elapsed

    try:
        model.optimize(callback)
        result.update(
            status=int(model.Status),
            wall=time.perf_counter() - started,
            construction_complete=True,
            method="arc_SOCP" if bundle.continuous else "arc_MISOCP",
            params=dict(
                Seed=seed,
                Threads=threads,
                MIPGap=float(model.Params.MIPGap),
                MIPGapAbs=float(model.Params.MIPGapAbs),
            ),
            state_nodes=sum(flow.graph.nodes for flow in bundle.flows),
            state_arcs=sum(len(flow.graph.edges) for flow in bundle.flows),
            service_graphs=sum(flow.graph.od is not None for flow in bundle.flows),
            empty_graphs=sum(flow.graph.od is None for flow in bundle.flows),
        )
        if not bundle.continuous:
            result["bound"] = mip_bound(model)
        if model.SolCount:
            missions, relocations = [], []
            for flow in bundle.flows:
                rows = decompose(
                    bundle.context,
                    flow.graph,
                    [flow.variables[e].X for e in range(len(flow.graph.edges))],
                )
                (missions if flow.graph.od is not None else relocations).extend(rows)
            bound = (
                (float(model.ObjVal) if model.Status == GRB.OPTIMAL else None)
                if bundle.continuous
                else float(model.ObjBound)
            )
            gap = (
                (0.0 if model.Status == GRB.OPTIMAL else None)
                if bundle.continuous
                else float(model.MIPGap)
            )
            result.update(
                objective=float(model.ObjVal),
                recourse_objective=float(bundle.recourse.getValue()),
                bound=bound,
                gap=gap,
                facility=bundle.x.X.copy(),
                capacity_level=bundle.z.X.copy(),
                assignment=bundle.alpha.X.copy(),
                airline=bundle.y.X.copy(),
                rho=np.asarray([bundle.rho[l].X for l in range(data["n_l"])]),
                idle=np.asarray([bundle.idle[l].X for l in range(data["n_l"])]),
                fleet=float(bundle.fleet.X),
                served=sum(row["flow"] for row in missions),
                missions=missions,
                relocations=relocations,
            )
        return result
    finally:
        model.dispose()
