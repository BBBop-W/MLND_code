"""Independent LR dual and primal recovery for the 2026-06-23 framework."""

from __future__ import annotations

import math
import time
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import gurobipy as gp
import networkx as nx
from gurobipy import GRB

from src import config
from src.model.gurobi_model import MLNDModel

from .linearized import (
    add_queue_supporting_lines,
    remove_all_quadratic_constraints,
)
from .profiling import PhaseProfiler
from .types import (
    Design,
    LagrangianIteration,
    LagrangianResult,
    Pair,
    design_signature,
)


class LagrangianEngine20260623:
    """OD-service LR with solver/Python timing separated by phase."""

    def __init__(
        self,
        owner,
        *,
        iterations: int = 50,
        initial_step: float = 20.0,
        max_designs: int = 12,
        use_queue_tangents: bool = False,
        restricted_solution_pool_size: int = 1,
        anchor_top_pairs: int = 0,
        restricted_time_limit: Optional[float] = None,
    ) -> None:
        self.owner = owner
        self.iterations = max(1, int(iterations))
        self.initial_step = float(initial_step)
        self.max_designs = max(1, int(max_designs))
        self.use_queue_tangents = bool(use_queue_tangents)
        self.restricted_solution_pool_size = max(
            1, int(restricted_solution_pool_size)
        )
        self.anchor_top_pairs = max(0, int(anchor_top_pairs))
        self.restricted_time_limit = (
            None
            if restricted_time_limit is None
            else max(0.05, float(restricted_time_limit))
        )
        self.profiler = PhaseProfiler()
        self.linearized_metadata: Dict[str, object] = {
            "restricted_network_qconstrs_before": None,
            "restricted_network_qconstrs_after": None,
            "restricted_network_queue_tangents": 0,
            "restricted_network_solution_pool_size": (
                self.restricted_solution_pool_size
            ),
            "lr_anchor_top_pairs": self.anchor_top_pairs,
            "lr_anchor_refinement_seeds": 0,
            "lr_restricted_time_limit": self.restricted_time_limit,
        }
        self._return_path_cache: Dict[
            Pair, Optional[List[Tuple[List[Tuple], float]]]
        ] = {}

    def _potential_profit(self, pair: Pair) -> float:
        revenue = float(getattr(config, "ORDER_REVENUE_COEF", 0.0))
        return revenue * max(
            0.0, float(self.owner.inst.h.get(pair, 0.0))
        )

    def _build_design_problem(
        self,
        lambda_origin: Mapping[Pair, float],
        lambda_destination: Mapping[Pair, float],
        lambda_arc: Mapping[Pair, float],
    ) -> Tuple[gp.Model, Dict[str, object]]:
        model = gp.Model("MLND_RegimeII_LR_20260623_Design")
        model.Params.OutputFlag = 0
        model.Params.Threads = 1
        model.Params.Seed = int(getattr(config, "BENDERS_RANDOM_SEED", 0))

        capacities = range(len(config.MU_HUB))
        x = model.addVars(self.owner.N_x, vtype=GRB.BINARY, name="x")
        y = model.addVars(self.owner.AY, vtype=GRB.BINARY, name="y")
        z = model.addVars(
            self.owner.H, capacities, vtype=GRB.BINARY, name="z"
        )

        objective = gp.quicksum(
            lambda_origin.get((i, j), 0.0) * x[i]
            + lambda_destination.get((i, j), 0.0) * x[j]
            + lambda_arc.get((i, j), 0.0) * y[i, j]
            for i, j in self.owner.A_HH
            if (i, j) in y
        )
        objective -= gp.quicksum(
            float(self.owner.inst.f.get(l, config.COST_BASE)) * x[l]
            for l in self.owner.L
        )
        objective -= gp.quicksum(
            float(getattr(config, "COST_HUB_FIXED", 0.0)) * x[i]
            for i in self.owner.H
        )
        objective -= gp.quicksum(
            float(self.owner.inst.f.get(k, config.COST_SWAPPING)) * x[k]
            for k in self.owner.E
        )
        objective -= gp.quicksum(
            float(config.COST_HUB[c]) * z[i, c]
            for i in self.owner.H
            for c in capacities
        )
        objective -= gp.quicksum(
            self.owner.arc_cost[arc] * y[arc] for arc in self.owner.AY
        )
        model.setObjective(objective, GRB.MAXIMIZE)

        for hub in self.owner.H:
            assignments = [
                y[l, h]
                for l, h in self.owner.A_LH
                if h == hub and (l, h) in y
            ]
            model.addConstr(gp.quicksum(assignments) == x[hub])
            model.addConstr(
                gp.quicksum(z[hub, c] for c in capacities) == x[hub]
            )
            model.addConstr(
                gp.quicksum(
                    float(config.MU_HUB[c]) * z[hub, c]
                    for c in capacities
                )
                >= x[hub] / float(config.TAU_H)
            )
        for base, hub in self.owner.A_LH:
            if (base, hub) in y:
                model.addConstr(y[base, hub] <= x[base])
        for u, v in self.owner.AY:
            if (u, v) in self.owner.A_LH:
                continue
            model.addConstr(
                y[u, v]
                <= self.owner._node_activation_expr({"x": x}, u)
            )
            model.addConstr(
                y[u, v]
                <= self.owner._node_activation_expr({"x": x}, v)
            )
        return model, {"x": x, "y": y, "z": z}

    def _return_paths(
        self, selected_pairs: Sequence[Pair]
    ) -> Tuple[List[Pair], Dict[Pair, List[Tuple[List[Tuple], float]]]]:
        with self.profiler.phase("lr_python_return_path_cache"):
            graph = nx.DiGraph()
            graph.add_nodes_from(self.owner.HLS)
            for u, v in [*self.owner.A_SS, *self.owner.A_SL]:
                route = self.owner.recovery_route(u, v)
                if route is not None:
                    graph.add_edge(
                        u, v, weight=float(self.owner.t[u, v, route])
                    )
            graph.add_edges_from(
                (u, v, {"weight": 0.0})
                for u, v in self.owner.A_virtual
            )

        accepted: List[Pair] = []
        paths: Dict[Pair, List[Tuple[List[Tuple], float]]] = {}
        for i, j in selected_pairs:
            cached = self._return_path_cache.get((i, j), ...)
            if cached is not ...:
                if cached is not None:
                    accepted.append((i, j))
                    paths[i, j] = cached
                continue
            if self.owner.recovery_route(i, j) is None:
                self._return_path_cache[i, j] = None
                continue
            endpoints = []
            chain_probability = self.owner._p_chain(i)
            if chain_probability > 1e-9:
                endpoints.append(
                    (self.owner._hub_pool(i), chain_probability)
                )
            if 1.0 - chain_probability > 1e-9:
                base = self.owner.nearest_base(i)
                if base is not None:
                    endpoints.append((base, 1.0 - chain_probability))
            recovered = []
            for endpoint, fraction in endpoints:
                try:
                    nodes = nx.shortest_path(
                        graph, j, endpoint, weight="weight"
                    )
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    recovered = []
                    break
                recovered.append(
                    (list(zip(nodes[:-1], nodes[1:])), fraction)
                )
            if recovered:
                accepted.append((i, j))
                paths[i, j] = recovered
                self._return_path_cache[i, j] = recovered
            else:
                self._return_path_cache[i, j] = None
        return accepted, paths

    def _build_recovery_pair_sets(
        self,
        pairs: Sequence[Pair],
        selected_count: Mapping[Pair, int],
    ) -> List[Tuple[Pair, ...]]:
        """Create demand-ranked and hub-coverage-balanced OD subsets."""

        with self.profiler.phase("lr_python_pair_set_construction"):
            ranked = sorted(
                pairs,
                key=lambda pair: (
                    self._potential_profit(pair)
                    * (1.0 + 0.02 * selected_count.get(pair, 0)),
                    float(self.owner.inst.h.get(pair, 0.0)),
                ),
                reverse=True,
            )
            count = len(ranked)
            if count == 0:
                return []
            hub_count = max(1, len(self.owner.H))
            target_sizes = {
                min(count, value)
                for value in (
                    5,
                    10,
                    20,
                    50,
                    2 * hub_count,
                    4 * hub_count,
                    int(math.ceil(0.15 * count)),
                    int(math.ceil(0.30 * count)),
                    int(math.ceil(0.50 * count)),
                    int(math.ceil(0.75 * count)),
                )
                if value > 0
            }

            pair_sets: List[Tuple[Pair, ...]] = []
            seen = set()

            def add_subset(subset: Sequence[Pair]) -> None:
                normalized = tuple(sorted(set(subset)))
                if normalized and normalized not in seen:
                    seen.add(normalized)
                    pair_sets.append(normalized)

            # Demand-ranked sets preserve the original LR signal.
            for target in sorted(target_sizes):
                add_subset(ranked[:target])

            best_outgoing: Dict[object, List[Pair]] = {}
            best_incoming: Dict[object, List[Pair]] = {}
            for pair in ranked:
                best_outgoing.setdefault(pair[0], []).append(pair)
                best_incoming.setdefault(pair[1], []).append(pair)

            # Balanced sets prevent a few high-demand hubs from consuming the
            # whole recovery budget as H and |OD| grow.
            balanced_targets = sorted(
                target
                for target in target_sizes
                if target >= min(count, hub_count)
            )
            for target in balanced_targets:
                chosen: List[Pair] = []
                chosen_set = set()
                depth = 0
                while len(chosen) < target:
                    added = False
                    for hub in self.owner.H:
                        for family in (best_outgoing, best_incoming):
                            options = family.get(hub, [])
                            if depth >= len(options):
                                continue
                            pair = options[depth]
                            if pair not in chosen_set:
                                chosen.append(pair)
                                chosen_set.add(pair)
                                added = True
                                if len(chosen) >= target:
                                    break
                        if len(chosen) >= target:
                            break
                    if not added:
                        break
                    depth += 1
                if len(chosen) < target:
                    for pair in ranked:
                        if pair in chosen_set:
                            continue
                        chosen.append(pair)
                        chosen_set.add(pair)
                        if len(chosen) >= target:
                            break
                add_subset(chosen)

            # Process informative medium/large sets first.  Tiny sets remain
            # as diversification candidates but no longer monopolize the
            # restricted-network refinement seeds.
            pair_sets.sort(
                key=lambda subset: (
                    abs(len(subset) - min(count, max(2 * hub_count, count // 3))),
                    -len({node for pair in subset for node in pair}),
                    -sum(self._potential_profit(pair) for pair in subset),
                )
            )
            return pair_sets

    def _build_anchor_pair_sets(
        self, pairs: Sequence[Pair]
    ) -> List[Tuple[Pair, ...]]:
        if self.anchor_top_pairs <= 0:
            return []
        ranked = sorted(
            pairs,
            key=lambda pair: (
                self._potential_profit(pair),
                float(self.owner.inst.h.get(pair, 0.0)),
                str(pair),
            ),
            reverse=True,
        )
        count = min(len(ranked), self.anchor_top_pairs)
        if count <= 0:
            return []
        return [tuple(sorted(ranked[:count]))]

    def _build_hub_coverage_seeds(
        self,
        pairs: Sequence[Pair],
        selected_count: Mapping[Pair, int],
    ) -> List[Design]:
        """Build diverse LR-guided facility sets for large-scale recovery.

        The first large-scale recovery versions used one hub ranking and one
        fraction schedule.  That made H=50 fragile: a sparse or dense seed
        could crowd out a medium fixed-hub design that later validated better.
        This ensemble keeps the Master boundary unchanged and only diversifies
        first-stage recovery seeds.
        """

        with self.profiler.phase("lr_python_hub_seed_construction"):
            lr_score = {hub: 0.0 for hub in self.owner.H}
            demand_score = {hub: 0.0 for hub in self.owner.H}
            pair_weight = {}
            for pair in pairs:
                weight = self._potential_profit(pair) * (
                    1.0 + 0.02 * selected_count.get(pair, 0)
                )
                pair_weight[pair] = weight
                lr_score[pair[0]] = lr_score.get(pair[0], 0.0) + weight
                lr_score[pair[1]] = lr_score.get(pair[1], 0.0) + weight
                raw_demand = float(self.owner.inst.h.get(pair, 0.0))
                demand_score[pair[0]] = (
                    demand_score.get(pair[0], 0.0) + raw_demand
                )
                demand_score[pair[1]] = (
                    demand_score.get(pair[1], 0.0) + raw_demand
                )

            ranked_lr = sorted(
                self.owner.H,
                key=lambda hub: (lr_score.get(hub, 0.0), str(hub)),
                reverse=True,
            )
            ranked_demand = sorted(
                self.owner.H,
                key=lambda hub: (demand_score.get(hub, 0.0), str(hub)),
                reverse=True,
            )

            remaining = set(self.owner.H)
            ranked_coverage = []
            uncovered = dict(pair_weight)
            while remaining:
                best = max(
                    remaining,
                    key=lambda hub: (
                        sum(
                            weight
                            for pair, weight in uncovered.items()
                            if pair[0] == hub or pair[1] == hub
                        ),
                        lr_score.get(hub, 0.0),
                        str(hub),
                    ),
                )
                ranked_coverage.append(best)
                remaining.remove(best)
                uncovered = {
                    pair: weight
                    for pair, weight in uncovered.items()
                    if pair[0] != best and pair[1] != best
                }
                if not uncovered:
                    ranked_coverage.extend(
                        hub
                        for hub in ranked_lr
                        if hub in remaining
                    )
                    break

            hub_count = len(self.owner.H)
            if hub_count >= 80:
                fractions = (0.95, 0.90, 0.85, 0.75, 1.00)
                max_seed_count = 9
            elif hub_count < 60 and hub_count >= 40:
                fractions = (0.55, 0.70, 0.85, 1.00, 0.40)
                max_seed_count = 10
            elif hub_count >= 40:
                fractions = (0.70, 0.85, 0.55, 1.00, 0.40)
                max_seed_count = 8
            else:
                fractions = (0.70, 0.85, 1.00, 0.50)
                max_seed_count = 6
            seeds = []
            seen_hub_sets = set()
            capacities = range(len(config.MU_HUB))
            ranking_policies = (
                ("lr_weighted", ranked_lr),
                ("coverage_greedy", ranked_coverage),
                ("demand_degree", ranked_demand),
            )
            for policy_name, ranked_hubs in ranking_policies:
                for fraction in fractions:
                    if len(seeds) >= max_seed_count:
                        break
                    count = max(
                        2,
                        min(
                            len(ranked_hubs),
                            int(math.ceil(fraction * len(ranked_hubs))),
                        ),
                    )
                    selected_hubs = set(ranked_hubs[:count])
                    hub_signature = tuple(
                        int(hub in selected_hubs)
                        for hub in sorted(self.owner.H)
                    )
                    if hub_signature in seen_hub_sets:
                        continue
                    seen_hub_sets.add(hub_signature)
                    x = {node: 0.0 for node in self.owner.N_x}
                    for hub in selected_hubs:
                        x[hub] = 1.0
                    y = {arc: 0.0 for arc in self.owner.AY}
                    z = {
                        (hub, c): 0.0
                        for hub in self.owner.H
                        for c in capacities
                    }
                    for hub in selected_hubs:
                        z[hub, len(config.MU_HUB) - 1] = 1.0
                    seeds.append(
                        {
                            "x": x,
                            "y": y,
                            "z": z,
                            "Gamma": 0.0,
                            "_lr_meta": {
                                "source": "lr_hub_coverage_seed",
                                "seed_policy": policy_name,
                                "hub_fraction": fraction,
                                "open_hubs": len(selected_hubs),
                            },
                        }
                    )
                if len(seeds) >= max_seed_count:
                    break
            return seeds

    @staticmethod
    def _with_candidate_metadata(
        design: Design,
        *,
        source: str,
        score: float,
        seed_index: Optional[int] = None,
        selected_pairs: Optional[int] = None,
        gamma_center: Optional[int] = None,
        fix_hub_set: Optional[bool] = None,
    ) -> Design:
        enriched = dict(design)
        metadata = dict(enriched.get("_lr_meta", {}))
        metadata.update(
            {
                "source": source,
                "surrogate_score": float(score),
            }
        )
        if seed_index is not None:
            metadata["seed_index"] = int(seed_index)
        if selected_pairs is not None:
            metadata["selected_pairs"] = int(selected_pairs)
        if gamma_center is not None:
            metadata["gamma_center"] = int(gamma_center)
        if fix_hub_set is not None:
            metadata["fix_hub_set"] = bool(fix_hub_set)
        enriched["_lr_meta"] = metadata
        return enriched

    @staticmethod
    def _facility_signature(design: Mapping[str, object]) -> Tuple[int, ...]:
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
        return (*x, *y, *z)

    def _candidate_source(self, design: Mapping[str, object]) -> str:
        metadata = design.get("_lr_meta", {})
        if isinstance(metadata, Mapping):
            return str(metadata.get("source", "unknown"))
        return "unknown"

    def _candidate_sort_key(
        self,
        design: Mapping[str, object],
        scores: Mapping[Tuple[int, ...], float],
    ) -> Tuple[float, float, int, float]:
        metadata = design.get("_lr_meta", {})
        score = scores.get(design_signature(design), -math.inf)
        gamma_center = None
        if isinstance(metadata, Mapping):
            gamma_center = metadata.get("gamma_center")
        gamma = float(design.get("Gamma", 0.0))
        gamma_distance = (
            abs(gamma - float(gamma_center))
            if gamma_center is not None
            else 0.0
        )
        source_priority = {
            "lr_fixed_hub_network_milp": 3,
            "lr_restricted_network_milp": 2,
            "lr_recovery": 1,
        }.get(self._candidate_source(design), 0)
        return (
            float(source_priority),
            float(score),
            int(self._candidate_source(design) == "lr_fixed_hub_network_milp"),
            -gamma_distance,
        )

    @staticmethod
    def _register_candidate(
        candidates: Dict[Tuple[int, ...], Design],
        candidate_scores: Dict[Tuple[int, ...], float],
        candidate: Design,
        score: float,
    ) -> None:
        signature = design_signature(candidate)
        if score >= candidate_scores.get(signature, -math.inf):
            candidates[signature] = candidate
            candidate_scores[signature] = float(score)

    def _recover_design(
        self,
        selected_pairs: Sequence[Pair],
        *,
        recovery_time_limit: float,
    ) -> Optional[Design]:
        owner = self.owner
        capacities = range(len(config.MU_HUB))
        accepted, return_paths = self._return_paths(selected_pairs)
        if not accepted:
            return None

        aggregate_arcs = set(accepted)
        for pair in accepted:
            for path, _ in return_paths[pair]:
                aggregate_arcs.update(path)

        touched_hubs = set(node for pair in accepted for node in pair)
        required_bases = set()
        for u, v in aggregate_arcs:
            for node in (u, v):
                if node in owner.H:
                    touched_hubs.add(node)
                elif node in owner.S:
                    touched_hubs.add(owner.inst.s_to_h[node])
                elif node in owner.L:
                    required_bases.add(node)

        model = gp.Model("MLND_RegimeII_LR_20260623_Recovery")
        model.Params.OutputFlag = 0
        model.Params.Threads = 1
        model.Params.Seed = int(getattr(config, "BENDERS_RANDOM_SEED", 0))
        model.Params.TimeLimit = max(0.05, float(recovery_time_limit))

        base_open = model.addVars(owner.L, vtype=GRB.BINARY, name="base")
        swap_open = model.addVars(owner.E, vtype=GRB.BINARY, name="swap")
        assignments = [
            (l, h) for l, h in owner.A_LH if h in touched_hubs
        ]
        assign = model.addVars(assignments, vtype=GRB.BINARY, name="assign")
        route_keys = [
            (u, v, k)
            for u, v in aggregate_arcs
            if (u, v) not in owner.A_virtual
            for k in owner.route_options.get((u, v), [])
        ]
        route = model.addVars(route_keys, vtype=GRB.BINARY, name="route")
        physical_arcs = sorted(
            {
                arc
                for u, v, k in route_keys
                for arc in owner.route_requirements(u, v, k)
                if arc in owner.AY
            }
            | set(assignments)
        )
        physical = model.addVars(
            physical_arcs, vtype=GRB.BINARY, name="physical"
        )

        for hub in touched_hubs:
            options = [assign[l, h] for l, h in assignments if h == hub]
            if not options:
                return None
            model.addConstr(gp.quicksum(options) == 1.0)
        for l, h in assignments:
            model.addConstr(assign[l, h] <= base_open[l])
            model.addConstr(physical[l, h] == assign[l, h])
        for base in required_bases:
            model.addConstr(base_open[base] == 1.0)

        for u, v in aggregate_arcs:
            if (u, v) in owner.A_virtual:
                continue
            options = [
                route[u, v, k]
                for k in owner.route_options.get((u, v), [])
                if (u, v, k) in route
            ]
            if not options:
                return None
            model.addConstr(gp.quicksum(options) == 1.0)
        for u, v, k in route_keys:
            for arc in owner.route_requirements(u, v, k):
                if arc in physical:
                    model.addConstr(route[u, v, k] <= physical[arc])
            if k in owner.E:
                model.addConstr(route[u, v, k] <= swap_open[k])

        flow_weight = {arc: 0.0 for arc in aggregate_arcs}
        for pair in accepted:
            demand = float(owner.inst.h.get(pair, 0.0))
            flow_weight[pair] += demand
            for path, fraction in return_paths[pair]:
                for arc in path:
                    flow_weight[arc] = (
                        flow_weight.get(arc, 0.0) + fraction * demand
                    )

        objective = gp.quicksum(
            float(owner.inst.f.get(l, config.COST_BASE)) * base_open[l]
            for l in owner.L
        )
        objective += gp.quicksum(
            float(owner.inst.f.get(k, config.COST_SWAPPING)) * swap_open[k]
            for k in owner.E
        )
        objective += gp.quicksum(
            owner.arc_cost[arc] * physical[arc] for arc in physical_arcs
        )
        objective += gp.quicksum(
            (
                owner.flight_cost[u, v, k]
                + float(config.COST_DRONE) * owner.t[u, v, k]
            )
            * flow_weight.get((u, v), 0.0)
            * route[u, v, k]
            for u, v, k in route_keys
        )
        model.setObjective(objective, GRB.MINIMIZE)

        with self.profiler.phase("lr_recovery_consolidation_milp"):
            model.optimize()
        self.profiler.add_solver_time(
            "lr_recovery_consolidation_milp", model.Runtime
        )
        if model.SolCount == 0:
            return None

        x = {node: 0.0 for node in owner.N_x}
        y = {arc: 0.0 for arc in owner.AY}
        z = {
            (hub, c): 0.0 for hub in owner.H for c in capacities
        }
        for hub in touched_hubs:
            x[hub] = 1.0
        for base in owner.L:
            x[base] = float(base_open[base].X > 0.5)
        for swap in owner.E:
            x[swap] = float(swap_open[swap].X > 0.5)
        for arc in physical_arcs:
            y[arc] = float(physical[arc].X > 0.5)

        incoming = {hub: 0.0 for hub in owner.H}
        for i, j in accepted:
            incoming[j] += float(owner.inst.h.get((i, j), 0.0))
        for hub in touched_hubs:
            required = incoming[hub] + 1.0 / float(config.TAU_H)
            level = len(config.MU_HUB) - 1
            for c, capacity in enumerate(config.MU_HUB):
                if float(capacity) >= required:
                    level = c
                    break
            z[hub, level] = 1.0

        total_flight_time = 0.0
        for pair in accepted:
            demand = float(owner.inst.h.get(pair, 0.0))
            u, v = pair
            chosen = next(
                (
                    k
                    for k in owner.route_options.get(pair, [])
                    if (u, v, k) in route and route[u, v, k].X > 0.5
                ),
                None,
            )
            if chosen is not None:
                total_flight_time += demand * float(owner.t[u, v, chosen])
            for path, fraction in return_paths[pair]:
                for a, b in path:
                    if (a, b) in owner.A_virtual:
                        continue
                    chosen_return = next(
                        (
                            k
                            for k in owner.route_options.get((a, b), [])
                            if (a, b, k) in route
                            and route[a, b, k].X > 0.5
                        ),
                        None,
                    )
                    if chosen_return is not None:
                        total_flight_time += (
                            fraction
                            * demand
                            * float(owner.t[a, b, chosen_return])
                        )

        gamma = max(0, int(math.ceil(1.50 * total_flight_time)))
        return {"x": x, "y": y, "z": z, "Gamma": float(gamma)}

    def _restricted_network_refinement(
        self,
        seed: Mapping[str, object],
        *,
        time_limit: float,
        hamming_radius: int,
        fix_hub_set: bool = False,
    ) -> List[Tuple[Design, float]]:
        """Solve an LR-centered MILP after removing every queueing cone."""

        linear = MLNDModel(self.owner.inst)
        linear.build()
        model = linear.model
        model.ModelName = "MLND_RegimeII_LR_20260623_RestrictedNetworkMILP"
        model.Params.OutputFlag = 0
        model.Params.Threads = 1
        model.Params.Seed = int(getattr(config, "BENDERS_RANDOM_SEED", 0))
        model.Params.TimeLimit = max(0.05, float(time_limit))
        model.Params.MIPGap = float(
            getattr(config, "BENDERS_LR_NETWORK_MIP_GAP", 0.01)
        )
        model.Params.MIPFocus = 1
        model.Params.Heuristics = 0.25
        if self.restricted_solution_pool_size > 1:
            model.Params.PoolSolutions = self.restricted_solution_pool_size
            model.Params.PoolSearchMode = 2
        model.Params.ImproveStartTime = min(
            5.0, 0.20 * max(0.05, float(time_limit))
        )
        q_metadata = remove_all_quadratic_constraints(linear)
        self.linearized_metadata.update(
            {
                "restricted_network_qconstrs_before": q_metadata[
                    "qconstrs_before"
                ],
                "restricted_network_qconstrs_after": q_metadata[
                    "qconstrs_after"
                ],
            }
        )

        # Supporting-line lower bounds preserve relaxation validity while
        # preventing the cone-free model from treating all queue length as 0.
        if self.use_queue_tangents:
            queue_tangents = add_queue_supporting_lines(
                linear, prefix="lr_queue_tangent"
            )
            self.linearized_metadata[
                "restricted_network_queue_tangents"
            ] = queue_tangents

        x = linear.vars["x"]
        y = linear.vars["y"]
        z = linear.vars["z"]
        gamma = linear.vars["Gamma"]
        for key, variable in x.items():
            variable.Start = float(seed["x"].get(key, 0.0))
        for key, variable in y.items():
            variable.Start = float(seed["y"].get(key, 0.0))
        for key, variable in z.items():
            variable.Start = float(seed["z"].get(key, 0.0))
        gamma.Start = float(seed.get("Gamma", 0.0))
        if fix_hub_set:
            for hub in self.owner.H:
                value = float(
                    float(seed["x"].get(hub, 0.0)) > 0.5
                )
                x[hub].LB = value
                x[hub].UB = value
        else:
            distance_terms = []
            for node in self.owner.N_x:
                distance_terms.append(
                    1 - x[node]
                    if float(seed["x"].get(node, 0.0)) > 0.5
                    else x[node]
                )
            model.addConstr(
                gp.quicksum(distance_terms)
                <= max(0, int(hamming_radius)),
                name="lr_facility_hamming_neighborhood",
            )

        with self.profiler.phase("lr_restricted_network_milp"):
            model.optimize()
        self.profiler.add_solver_time(
            "lr_restricted_network_milp", model.Runtime
        )
        if model.SolCount == 0:
            return []
        refinements: list[Tuple[Design, float]] = []
        for solution_index in range(
            min(int(model.SolCount), self.restricted_solution_pool_size)
        ):
            model.Params.SolutionNumber = solution_index
            design = {
                "x": {
                    key: float(variable.Xn)
                    for key, variable in linear.vars["x"].items()
                },
                "y": {
                    key: float(variable.Xn)
                    for key, variable in linear.vars["y"].items()
                },
                "z": {
                    key: float(variable.Xn)
                    for key, variable in linear.vars["z"].items()
                },
                "Gamma": float(linear.vars["Gamma"].Xn),
            }
            refinements.append((design, float(model.PoolObjVal)))
        return refinements

    def generate(
        self,
        *,
        time_limit: float,
    ) -> LagrangianResult:
        start = time.perf_counter()
        deadline = start + max(0.0, float(time_limit))
        lr_deadline = start + 0.35 * max(0.0, float(time_limit))

        pairs = [
            pair
            for pair in self.owner.A_HH
            if self._potential_profit(pair) > 1e-9
            and pair in self.owner.AY
        ]
        lambda_origin = {pair: 0.0 for pair in pairs}
        lambda_destination = {pair: 0.0 for pair in pairs}
        lambda_arc = {pair: 0.0 for pair in pairs}
        best_dual = math.inf
        history: List[LagrangianIteration] = []
        pair_sets: List[Tuple[Pair, ...]] = []
        pair_set_seen = set()
        selected_count = {pair: 0 for pair in pairs}
        anchor_pair_sets = self._build_anchor_pair_sets(pairs)
        anchor_pair_set_signatures = set(anchor_pair_sets)
        for subset in anchor_pair_sets:
            if subset and subset not in pair_set_seen:
                pair_set_seen.add(subset)
                pair_sets.append(subset)

        for iteration in range(1, self.iterations + 1):
            if time.perf_counter() >= lr_deadline:
                break
            with self.profiler.phase("lr_python_od_selection"):
                selected = {
                    pair: int(
                        self._potential_profit(pair)
                        - lambda_origin[pair]
                        - lambda_destination[pair]
                        - lambda_arc[pair]
                        > 0.0
                    )
                    for pair in pairs
                }
                for pair, value in selected.items():
                    selected_count[pair] += int(value)
                service_value = sum(
                    max(
                        0.0,
                        self._potential_profit(pair)
                        - lambda_origin[pair]
                        - lambda_destination[pair]
                        - lambda_arc[pair],
                    )
                    for pair in pairs
                )

            model, variables = self._build_design_problem(
                lambda_origin, lambda_destination, lambda_arc
            )
            model.Params.TimeLimit = max(
                0.05, min(10.0, lr_deadline - time.perf_counter())
            )
            with self.profiler.phase("lr_design_milp"):
                model.optimize()
            self.profiler.add_solver_time("lr_design_milp", model.Runtime)
            if model.SolCount == 0:
                break

            raw_x = {
                node: float(variables["x"][node].X)
                for node in self.owner.N_x
            }
            raw_y = {
                arc: float(variables["y"][arc].X)
                for arc in self.owner.AY
            }
            design_bound = (
                float(model.ObjVal)
                if model.Status == GRB.OPTIMAL
                else float(model.ObjBound)
            )
            dual_bound = service_value + design_bound
            best_dual = min(best_dual, dual_bound)

            active = [pair for pair, value in selected.items() if value]
            active.sort(key=self._potential_profit, reverse=True)
            for limit in (5, 10, 20, 50):
                subset = tuple(sorted(active[:limit]))
                if subset and subset not in pair_set_seen:
                    pair_set_seen.add(subset)
                    pair_sets.append(subset)

            with self.profiler.phase("lr_python_multiplier_update"):
                violations = []
                for i, j in pairs:
                    q = float(selected[i, j])
                    violations.extend(
                        (
                            q - raw_x[i],
                            q - raw_x[j],
                            q - raw_y.get((i, j), 0.0),
                        )
                    )
                norm_sq = sum(value * value for value in violations)
                step = self.initial_step / math.sqrt(float(iteration))
                if norm_sq <= 1e-12:
                    step = 0.0
                for i, j in pairs:
                    q = float(selected[i, j])
                    lambda_origin[i, j] = max(
                        0.0,
                        lambda_origin[i, j] + step * (q - raw_x[i]),
                    )
                    lambda_destination[i, j] = max(
                        0.0,
                        lambda_destination[i, j] + step * (q - raw_x[j]),
                    )
                    lambda_arc[i, j] = max(
                        0.0,
                        lambda_arc[i, j]
                        + step * (q - raw_y.get((i, j), 0.0)),
                    )
            history.append(
                LagrangianIteration(
                    iteration=iteration,
                    dual_bound=dual_bound,
                    best_dual_bound=best_dual,
                    selected_pairs=sum(selected.values()),
                    open_hubs=sum(
                        int(raw_x[hub] > 0.5) for hub in self.owner.H
                    ),
                    violation_norm=math.sqrt(norm_sq),
                    step_size=step,
                )
            )

        candidates: Dict[Tuple[int, ...], Design] = {}
        candidate_scores: Dict[Tuple[int, ...], float] = {}
        diversified_candidates: List[Design] = []
        anchor_refinement_seeds: List[Design] = []
        candidate_history: List[Dict[str, object]] = []
        for subset in self._build_recovery_pair_sets(
            pairs, selected_count
        ):
            if subset not in pair_set_seen:
                pair_set_seen.add(subset)
                pair_sets.append(subset)
        ranked_pair_sets = pair_sets
        for index, subset in enumerate(ranked_pair_sets):
            if time.perf_counter() >= deadline:
                break
            remaining = deadline - time.perf_counter()
            recovered = self._recover_design(
                subset,
                recovery_time_limit=min(
                    2.0 if len(self.owner.H) <= 20 else 5.0,
                    remaining,
                ),
            )
            if recovered is None:
                continue
            center = int(round(float(recovered["Gamma"])))
            surrogate_score = (
                sum(self._potential_profit(pair) for pair in subset)
                - self.owner._fixed_cost_value(recovered)
            )
            is_anchor_subset = subset in anchor_pair_set_signatures
            if is_anchor_subset:
                anchor_seed = self._with_candidate_metadata(
                    {
                        "x": dict(recovered["x"]),
                        "y": dict(recovered["y"]),
                        "z": dict(recovered["z"]),
                        "Gamma": float(recovered["Gamma"]),
                    },
                    source="lr_anchor_recovery_seed",
                    score=surrogate_score,
                    seed_index=index,
                    selected_pairs=len(subset),
                    gamma_center=center,
                )
                anchor_seed["_lr_meta"]["seed_policy"] = "top_demand_anchor"
                anchor_refinement_seeds.append(anchor_seed)
            gamma_values = {
                max(0, int(math.ceil(center * factor)))
                for factor in (
                    0.08,
                    0.10,
                    0.12,
                    0.15,
                    0.20,
                    0.30,
                    0.50,
                    0.75,
                    1.00,
                )
            }
            network_candidates = []
            for gamma in gamma_values:
                base_candidate = {
                    "x": dict(recovered["x"]),
                    "y": dict(recovered["y"]),
                    "z": dict(recovered["z"]),
                    "Gamma": float(gamma),
                }
                gamma_surrogate_score = (
                    sum(self._potential_profit(pair) for pair in subset)
                    - self.owner._fixed_cost_value(base_candidate)
                )
                score = gamma_surrogate_score - float(config.COST_DRONE) * abs(
                    float(gamma) - float(center)
                )
                candidate = self._with_candidate_metadata(
                    base_candidate,
                    source="lr_recovery",
                    score=score,
                    seed_index=index,
                    selected_pairs=len(subset),
                    gamma_center=center,
                )
                self._register_candidate(
                    candidates, candidate_scores, candidate, score
                )
                network_candidates.append(candidate)
                candidate_history.append(
                    {
                        "source": "lr_recovery",
                        "pair_set_index": index,
                        "selected_pairs": len(subset),
                        "Gamma": gamma,
                        "surrogate_score": gamma_surrogate_score,
                        "ranking_score": score,
                        "gamma_center": center,
                        "is_anchor": is_anchor_subset,
                    }
                )
            diversified_candidates.extend(
                sorted(
                    network_candidates,
                    key=lambda design: candidate_scores[
                        design_signature(design)
                    ],
                    reverse=True,
                )[:2]
            )

        empty: Design = {
            "x": {node: 0.0 for node in self.owner.N_x},
            "y": {arc: 0.0 for arc in self.owner.AY},
            "z": {
                (hub, c): 0.0
                for hub in self.owner.H
                for c in range(len(config.MU_HUB))
            },
            "Gamma": 0.0,
        }
        empty_signature = design_signature(empty)
        candidates[empty_signature] = empty
        candidate_scores[empty_signature] = 0.0

        # The LR subsets identify promising facilities; this cone-free MILP
        # jointly reoptimizes flow, arcs, capacity and fleet in a bounded
        # facility neighborhood.  It is not the original MISOCP.
        refinement_seeds = []
        refinement_seen = set()
        max_refinement_seeds = (
            4 if len(self.owner.H) <= 20 else 6 if len(self.owner.H) <= 50 else 8
        )

        def add_refinement_seed(design: Design) -> bool:
            facility_signature = tuple(
                int(float(value) > 0.5)
                for _, value in sorted(design["x"].items())
            )
            if facility_signature in refinement_seen:
                return False
            refinement_seen.add(facility_signature)
            refinement_seeds.append(design)
            return True

        anchor_added = 0
        for design in anchor_refinement_seeds:
            if len(refinement_seeds) >= max_refinement_seeds:
                break
            if add_refinement_seed(design):
                anchor_added += 1
        self.linearized_metadata["lr_anchor_refinement_seeds"] = anchor_added

        seed_pool = sorted(
            candidates.values(),
            key=lambda item: (
                candidate_scores.get(design_signature(item), -math.inf),
                sum(float(value) > 0.5 for value in item["x"].values()),
            ),
            reverse=True,
        )
        hub_coverage_seeds = self._build_hub_coverage_seeds(
            pairs, selected_count
        )
        for design in seed_pool:
            add_refinement_seed(design)
            if len(refinement_seeds) >= max_refinement_seeds:
                break

        # Dense hub-set refinements are solved first on large instances.
        staged_refinements = []
        anchor_signatures = {
            self._facility_signature(design) for design in anchor_refinement_seeds
        }
        if self.anchor_top_pairs > 0:
            staged_refinements.extend(
                (seed, False)
                for seed in refinement_seeds
                if self._facility_signature(seed) in anchor_signatures
            )
        if len(self.owner.H) >= 40:
            staged_refinements.extend(
                (seed, True) for seed in hub_coverage_seeds
            )
        non_anchor_refinement_seeds = [
            seed
            for seed in refinement_seeds
            if self._facility_signature(seed) not in anchor_signatures
        ]
        if len(self.owner.H) < 80:
            staged_refinements.extend(
                (seed, False) for seed in non_anchor_refinement_seeds
            )
        else:
            staged_refinements.extend(
                (seed, False) for seed in non_anchor_refinement_seeds[:2]
            )

        for seed_index, (seed, fix_hub_set) in enumerate(
            staged_refinements, 1
        ):
            remaining = deadline - time.perf_counter()
            if remaining <= 0.10:
                break
            radius = (
                2
                if len(self.owner.H) <= 10
                else 8
                if len(self.owner.H) <= 20
                else max(12, len(self.owner.H) // 4)
            )
            per_seed_limit = (
                5.0
                if len(self.owner.H) <= 10
                else 8.0
                if len(self.owner.H) <= 20
                else 12.0
                if len(self.owner.H) <= 50
                else 30.0
            )
            if self.restricted_time_limit is not None:
                per_seed_limit = self.restricted_time_limit
            refinements = self._restricted_network_refinement(
                seed,
                time_limit=min(per_seed_limit, remaining),
                hamming_radius=radius,
                fix_hub_set=fix_hub_set,
            )
            if not refinements:
                continue
            for pool_index, (refined, refinement_score) in enumerate(
                refinements
            ):
                center = max(0, int(round(float(refined["Gamma"]))))
                gamma_values = {
                    max(0, center + delta)
                    for delta in (-2, -1, 0, 1, 2)
                }
                gamma_values.update(
                    {
                        int(math.ceil(center * factor))
                        for factor in (1.15, 1.30, 1.50, 1.75, 2.00)
                    }
                )
                refined_candidates = []
                for gamma in gamma_values:
                    seed_metadata = dict(seed.get("_lr_meta", {}))
                    base_candidate = {
                        "x": dict(refined["x"]),
                        "y": dict(refined["y"]),
                        "z": dict(refined["z"]),
                        "Gamma": float(gamma),
                        "_lr_meta": seed_metadata,
                    }
                    # The cone-free MILP objective is useful for ranking but
                    # is not an exact incumbent value.
                    score = refinement_score - float(
                        config.COST_DRONE
                    ) * abs(float(gamma) - float(center))
                    source = (
                        "lr_fixed_hub_network_milp"
                        if fix_hub_set
                        else "lr_restricted_network_milp"
                    )
                    candidate = self._with_candidate_metadata(
                        base_candidate,
                        source=source,
                        score=score,
                        seed_index=seed_index,
                        gamma_center=center,
                        fix_hub_set=fix_hub_set,
                    )
                    self._register_candidate(
                        candidates, candidate_scores, candidate, score
                    )
                    refined_candidates.append(candidate)
                    candidate_history.append(
                        {
                            "source": source,
                            "pair_set_index": seed_index,
                            "pool_index": pool_index,
                            "selected_pairs": None,
                            "Gamma": gamma,
                            "surrogate_score": score,
                            "ranking_score": score,
                            "gamma_center": center,
                            "fix_hub_set": fix_hub_set,
                        }
                    )
                priority_gamma = (
                    center,
                    int(math.ceil(1.30 * center)),
                    int(math.ceil(1.50 * center)),
                )
                refined_candidates.sort(
                    key=lambda candidate: min(
                        abs(int(round(float(candidate["Gamma"]))) - value)
                        for value in priority_gamma
                    )
                )
                diversified_candidates = [
                    *refined_candidates[:3],
                    *diversified_candidates,
                ]

        with self.profiler.phase("lr_python_candidate_ranking"):
            ranked = sorted(
                [
                    design
                    for signature, design in candidates.items()
                    if signature != empty_signature
                ],
                key=lambda design: self._candidate_sort_key(
                    design, candidate_scores
                ),
                reverse=True,
            )
            designs = []
            selected = set()
            slots = max(0, self.max_designs - 1)

            def add_design(design: Design) -> bool:
                signature = design_signature(design)
                if signature in selected or signature == empty_signature:
                    return False
                designs.append(design)
                selected.add(signature)
                return True

            # Reserve part of the validation budget for each recovery source.
            # This separates bound-improving BD work from incumbent-improving
            # LR recovery and prevents one seed family from hiding another.
            source_quotas = (
                ("lr_fixed_hub_network_milp", max(2, slots // 3)),
                ("lr_restricted_network_milp", max(2, slots // 4)),
                ("lr_recovery", max(1, slots // 6)),
            )
            for source, quota in source_quotas:
                used = 0
                seen_facilities = set()
                for design in ranked:
                    if used >= quota or len(designs) >= slots:
                        break
                    if self._candidate_source(design) != source:
                        continue
                    facility_signature = self._facility_signature(design)
                    if facility_signature in seen_facilities:
                        continue
                    if add_design(design):
                        seen_facilities.add(facility_signature)
                        used += 1
            for design in ranked:
                if len(designs) >= slots:
                    break
                add_design(design)
            # The zero design is a mandatory safety incumbent for every LR run.
            designs.append(empty)

        wall_time = time.perf_counter() - start
        return LagrangianResult(
            designs=designs,
            best_dual_bound=best_dual,
            history=history,
            candidate_history=candidate_history,
            wall_time=wall_time,
            metadata={
                **self.linearized_metadata,
                "lr_queue_tangents_enabled": self.use_queue_tangents,
            },
        )
