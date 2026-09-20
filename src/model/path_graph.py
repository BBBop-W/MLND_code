"""Exact acyclic state graph for the current three-leg mission domain.

Nodes record service stage, remaining battery and, before delivery, the clock
needed by the deadline constraint. Edges represent individual physical flights;
a swap is charged on arrival and handling is charged on arrival at a locker.
This is a state-expanded arc formulation, not a compact physical-node model.
It shares feasible prefixes and suffixes without enumerating complete paths.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from src.solvers.pricing.routes import _materialize

TOL = 1e-10


@dataclass(frozen=True)
class Edge:
    tail: int
    head: int
    time: float
    operating_cost: float
    arc_counts: tuple
    hub_visits: tuple = ()
    choice: tuple | None = None


@dataclass
class PathGraph:
    nodes: int
    edges: list[Edge]
    od: int | None
    l: int
    m: int
    source: int = 0
    sink: int = 1

    def adjacency(self):
        incoming = [[] for _ in range(self.nodes)]
        outgoing = [[] for _ in range(self.nodes)]
        for k, edge in enumerate(self.edges):
            incoming[edge.head].append(k)
            outgoing[edge.tail].append(k)
        return incoming, outgoing


@dataclass(frozen=True)
class Leg:
    nodes: tuple
    swap: int
    duration: float
    first: float
    last: float


def _legs(c, u, v):
    if not c["facility"][u] or not c["facility"][v]:
        return
    flight = float(c["flight"][u, v])
    if math.isfinite(flight) and flight <= c["battery"] + TOL:
        yield Leg((u, v), -1, flight, flight, flight)
    for k in range(c["K"]):
        w = c["L"] + c["H"] + k
        a, b = float(c["flight"][u, w]), float(c["flight"][w, v])
        if (
            c["facility"][w]
            and math.isfinite(a + b)
            and max(a, b) <= c["battery"] + TOL
        ):
            yield Leg((u, w, v), k, a + b + float(c["swap"][k]), a, b)


def _energy(c, used, leg, extra=0.0):
    if used + leg.first > c["battery"] + TOL:
        return None
    after = (leg.last if leg.swap >= 0 else used + leg.last) + extra
    return after if after <= c["battery"] + TOL else None


def _node(graph):
    value = graph.nodes
    graph.nodes += 1
    return value


def _add_leg(
    c, graph, tail, head, leg, *, extra_time=0.0, extra_cost=0.0, hub=None, choice=None
):
    """Split a single-swap leg into physical flight edges."""
    for index, (u, v) in enumerate(zip(leg.nodes, leg.nodes[1:])):
        last = index == len(leg.nodes) - 2
        target = head if last else _node(graph)
        flight = float(c["flight"][u, v])
        swap = float(c["swap"][v - c["L"] - c["H"]]) if v >= c["L"] + c["H"] else 0.0
        graph.edges.append(
            Edge(
                tail,
                target,
                flight + swap + (extra_time if last else 0.0),
                c["flightcost"] * flight + (extra_cost if last else 0.0),
                ((u * c["N"] + v, 1),),
                (int(hub),) if last and hub is not None else (),
                choice if last else None,
            )
        )
        tail = target


def _prune(graph):
    """Keep only edges on a source-to-sink route; renumber surviving states."""
    incoming, outgoing = graph.adjacency()

    def reach(start, adjacency, reverse=False):
        visited = {start}
        stack = [start]
        while stack:
            node = stack.pop()
            for k in adjacency[node]:
                edge = graph.edges[k]
                other = edge.tail if reverse else edge.head
                if other not in visited:
                    visited.add(other)
                    stack.append(other)
        return visited

    keep = reach(graph.source, outgoing) & reach(graph.sink, incoming, True)
    if graph.sink not in keep:
        return PathGraph(2, [], graph.od, graph.l, graph.m)
    ordered = [graph.source, graph.sink] + sorted(keep - {graph.source, graph.sink})
    index = {old: new for new, old in enumerate(ordered)}
    edges = [
        Edge(
            index[e.tail],
            index[e.head],
            e.time,
            e.operating_cost,
            e.arc_counts,
            e.hub_visits,
            e.choice,
        )
        for e in graph.edges
        if e.tail in keep and e.head in keep
    ]
    return PathGraph(len(ordered), edges, graph.od, graph.l, graph.m)


def build_service(c, od, l, m, *, check=None):
    """Build the exact feasible state graph for one OD and ordered base pair."""
    graph = PathGraph(2, [], int(od), int(l), int(m))
    if not c["facility"][l] or not c["facility"][m]:
        return graph
    o, d = int(c["oi"][od]), int(c["di"][od])
    pickup_states = {}
    for i in range(c["H"]):
        if check:
            check(graph)
        mean = float(c["mean"][o, i])
        quantile = (
            mean if c["family"] == "deterministic" else -mean * math.log(c["epsilon"])
        )
        handling = float(c["pickup"][i])
        for leg in _legs(c, l, c["L"] + i):
            wait = (
                max(0.0, mean - leg.duration)
                if c["family"] == "deterministic"
                else (mean * math.exp(-leg.duration / mean) if mean > 0 else 0.0)
            )
            quantile_wait = max(0.0, quantile - leg.duration)
            used = _energy(c, 0.0, leg, quantile_wait + handling)
            clock = leg.duration + quantile_wait + handling
            if used is None or clock > c["deadline"][od] + TOL:
                continue
            state = (i, used, clock)
            if state not in pickup_states:
                pickup_states[state] = _node(graph)
            _add_leg(
                c,
                graph,
                graph.source,
                pickup_states[state],
                leg,
                extra_time=wait + handling,
                extra_cost=float(c["first"][o, i]) + c["hovercost"] * wait,
                hub=i,
                choice=(0, i, leg.swap),
            )

    delivery_states = {}
    for (i, used, clock), tail in pickup_states.items():
        if check:
            check(graph)
        for j in range(c["H"]):
            if i == j:
                continue
            handling = float(c["delivery"][j])
            for leg in _legs(c, c["L"] + i, c["L"] + j):
                after = _energy(c, used, leg, handling)
                deadline_time = (
                    clock + leg.duration + handling + c["terminaltime"][d, j]
                )
                if after is None or deadline_time > c["deadline"][od] + TOL:
                    continue
                state = (j, after)
                if state not in delivery_states:
                    delivery_states[state] = _node(graph)
                _add_leg(
                    c,
                    graph,
                    tail,
                    delivery_states[state],
                    leg,
                    extra_time=handling,
                    extra_cost=float(c["terminal"][d, j]),
                    hub=j,
                    choice=(1, j, leg.swap),
                )
    for (j, used), tail in delivery_states.items():
        if check:
            check(graph)
        for leg in _legs(c, c["L"] + j, m):
            if _energy(c, used, leg) is not None:
                _add_leg(c, graph, tail, graph.sink, leg, choice=(2, m, leg.swap))
    return _prune(graph)


def build_empty(c, l, m, *, check=None):
    """Build feasible direct and single-swap empty-movement alternatives."""
    graph = PathGraph(2, [], None, int(l), int(m))
    if l == m:
        return graph
    for leg in _legs(c, l, m):
        if check:
            check(graph)
        if _energy(c, 0.0, leg) is not None:
            _add_leg(c, graph, graph.source, graph.sink, leg, choice=(0, m, leg.swap))
    return _prune(graph)


def reconstruct(c, graph, edge_indices):
    """Recover the physical trace from one source-to-sink state-graph path."""
    choices = {}
    for k in edge_indices:
        choice = graph.edges[int(k)].choice
        if choice is not None:
            stage, node, swap = choice
            if stage in choices:
                raise ValueError("Repeated service stage in state-graph path")
            choices[stage] = (node, swap)
    if graph.od is None:
        if set(choices) != {0}:
            raise ValueError("Incomplete empty state-graph path")
        return _materialize(c, None, graph.l, graph.m, None, None, [choices[0][1]])
    if set(choices) != {0, 1, 2}:
        raise ValueError("Incomplete service state-graph path")
    return _materialize(
        c,
        graph.od,
        graph.l,
        graph.m,
        choices[0][0],
        choices[1][0],
        [choices[s][1] for s in range(3)],
    )
