"""Root-relaxation max-flow cuts for the single-period Regime-II model.

The cuts are generated outside a Gurobi callback.  This is deliberate: on the
current Gurobi 13.0.2/macOS environment, merely enabling a Python callback on
this MIQCP can change the reported optimum.  The safe workflow is:

1. solve a continuous copy of the model;
2. separate directed minimum cuts in that copy;
3. add the resulting globally valid inequalities as ordinary constraints to
   both the relaxation and the untouched integer model;
4. solve the integer model normally, without a callback.

For an OD arc (i,j), positive loaded flow psi[i,j] requires a return path from
j to i because the aggregate network is flow-balanced.  If a directed cut
separates j from i, at least one activated aggregate arc must cross it.  Since
0 <= psi[i,j] <= demand[i,j], the following inequality is valid:

    psi[i,j] <= demand[i,j] * sum(y[u,v] for (u,v) in delta+(R)).

Co-located H<->S handoff arcs are treated as always available in the separation
network.  Cuts crossed by such a constant-capacity arc are skipped because they
reduce to the already-valid demand upper bound and provide no strengthening.
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

import networkx as nx
from gurobipy import GRB


Node = str
Arc = Tuple[Node, Node]


@dataclass(frozen=True)
class MaxFlowRootCut:
    origin: Node
    destination: Node
    demand: float
    crossing_arcs: Tuple[Arc, ...]
    violation: float
    round_index: int

    def signature(self) -> Tuple[Node, Node, Tuple[Arc, ...]]:
        return self.origin, self.destination, self.crossing_arcs


@dataclass
class MaxFlowCutStats:
    rounds: int = 0
    relaxation_solves: int = 0
    candidate_pairs: int = 0
    min_cut_solves: int = 0
    cuts_added: int = 0
    initial_relaxation_bound: Optional[float] = None
    final_relaxation_bound: Optional[float] = None
    separation_time: float = 0.0
    total_time: float = 0.0
    stopped_by_time_limit: bool = False

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def add_root_max_flow_cuts(
    mlnd_model,
    *,
    time_limit: float = 60.0,
    max_rounds: int = 5,
    max_cuts_per_round: int = 50,
    max_od_pairs: Optional[int] = 500,
    violation_tolerance: float = 1e-6,
    output_flag: int = 0,
    threads: int = 1,
) -> Tuple[List[MaxFlowRootCut], MaxFlowCutStats]:
    """Separate max-flow cuts at the continuous root relaxation.

    Parameters
    ----------
    mlnd_model:
        A built ``MLNDModel``.  The original integer model receives ordinary
        constraints only after each cut has been verified in the relaxation.
    time_limit:
        Wall-clock budget for all relaxation solves and min-cut separation.
        Callers should subtract this time from the final 900-second solve budget.
    max_od_pairs:
        At most this many positive-flow OD pairs are checked in one round,
        prioritizing larger fractional flow.  ``None`` checks all pairs.
    """
    if not getattr(mlnd_model, "_built", False):
        mlnd_model.build()

    start = time.perf_counter()
    deadline = start + max(0.0, float(time_limit))
    stats = MaxFlowCutStats()
    cuts: List[MaxFlowRootCut] = []
    seen = set()

    original = mlnd_model.model
    original.update()
    relaxation = original.relax()
    relaxation.Params.OutputFlag = int(output_flag)
    relaxation.Params.Threads = max(1, int(threads))

    relaxed_vars = {
        var.VarName: relaxation.getVarByName(var.VarName)
        for var in original.getVars()
    }
    actual_arcs = tuple(
        sorted(
            (u, v)
            for u, v in mlnd_model.A_hat
            if (u, v) in mlnd_model.vars["y"]
        )
    )
    virtual_arcs = tuple(sorted(set(mlnd_model.A_HS) | set(mlnd_model.A_SH)))

    for round_index in range(1, max(0, int(max_rounds)) + 1):
        remaining = deadline - time.perf_counter()
        if remaining <= 0.0:
            stats.stopped_by_time_limit = True
            break

        relaxation.Params.TimeLimit = max(0.01, remaining)
        relaxation.optimize()
        stats.relaxation_solves += 1
        if relaxation.Status not in {GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL}:
            break
        if relaxation.SolCount <= 0:
            break

        bound = float(relaxation.ObjVal)
        if stats.initial_relaxation_bound is None:
            stats.initial_relaxation_bound = bound
        stats.final_relaxation_bound = bound
        stats.rounds = round_index

        graph = nx.DiGraph()
        graph.add_nodes_from(mlnd_model.HLS)
        for u, v in actual_arcs:
            var = relaxed_vars[mlnd_model.vars["y"][u, v].VarName]
            capacity = min(1.0, max(0.0, float(var.X)))
            graph.add_edge(u, v, capacity=capacity, is_virtual=False)
        for u, v in virtual_arcs:
            graph.add_edge(u, v, capacity=1.0, is_virtual=True)

        positive_pairs = []
        for i, j in mlnd_model.A_HH:
            demand = float(mlnd_model.inst.h.get((i, j), 0.0))
            if demand <= violation_tolerance:
                continue
            psi_var = relaxed_vars[mlnd_model.vars["psi"][i, j].VarName]
            flow = max(0.0, float(psi_var.X))
            if flow <= violation_tolerance:
                continue
            positive_pairs.append((flow, demand, i, j))
        positive_pairs.sort(reverse=True)
        if max_od_pairs is not None:
            positive_pairs = positive_pairs[: max(0, int(max_od_pairs))]
        stats.candidate_pairs += len(positive_pairs)

        separation_start = time.perf_counter()
        violated: List[MaxFlowRootCut] = []
        for flow, demand, i, j in positive_pairs:
            if time.perf_counter() >= deadline:
                stats.stopped_by_time_limit = True
                break
            try:
                cut_value, partition = nx.minimum_cut(
                    graph,
                    j,
                    i,
                    capacity="capacity",
                )
            except (nx.NetworkXError, nx.NetworkXUnbounded):
                continue
            stats.min_cut_solves += 1
            reachable, non_reachable = partition
            violation = flow - demand * float(cut_value)
            if violation <= violation_tolerance:
                continue

            crossing_actual = []
            crosses_virtual = False
            for u, v, data in graph.edges(data=True):
                if u not in reachable or v not in non_reachable:
                    continue
                if bool(data.get("is_virtual", False)):
                    crosses_virtual = True
                    break
                crossing_actual.append((u, v))
            if crosses_virtual or not crossing_actual:
                continue

            crossing_tuple = tuple(sorted(crossing_actual))
            cut = MaxFlowRootCut(
                origin=i,
                destination=j,
                demand=demand,
                crossing_arcs=crossing_tuple,
                violation=float(violation),
                round_index=round_index,
            )
            if cut.signature() in seen:
                continue
            violated.append(cut)

        stats.separation_time += time.perf_counter() - separation_start
        violated.sort(key=lambda item: item.violation, reverse=True)
        selected = violated[: max(0, int(max_cuts_per_round))]
        if not selected:
            break

        for cut in selected:
            relaxed_psi = relaxed_vars[
                mlnd_model.vars["psi"][cut.origin, cut.destination].VarName
            ]
            relaxed_rhs = sum(
                (
                    relaxed_vars[mlnd_model.vars["y"][arc].VarName]
                    for arc in cut.crossing_arcs
                ),
                start=0,
            )
            relaxation.addConstr(
                relaxed_psi <= cut.demand * relaxed_rhs,
                name=f"maxflow_root_r{round_index}_{len(cuts) + 1}",
            )

            integer_rhs = sum(
                (mlnd_model.vars["y"][arc] for arc in cut.crossing_arcs),
                start=0,
            )
            original.addConstr(
                mlnd_model.vars["psi"][cut.origin, cut.destination]
                <= cut.demand * integer_rhs,
                name=f"maxflow_root_r{round_index}_{len(cuts) + 1}",
            )
            seen.add(cut.signature())
            cuts.append(cut)

        stats.cuts_added = len(cuts)
        relaxation.update()
        original.update()

    stats.total_time = time.perf_counter() - start
    if stats.final_relaxation_bound is not None and not math.isfinite(
        stats.final_relaxation_bound
    ):
        stats.final_relaxation_bound = None
    return cuts, stats
