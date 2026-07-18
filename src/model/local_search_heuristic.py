import math
import time
from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Optional, Sequence, Set, Tuple

from gurobipy import GRB

from src import config
from src.data_processor.instance import Instance

try:
    from src.model.Benders import BendersRegimeIISolver
except ModuleNotFoundError:  # pragma: no cover - kept for case-insensitive worktrees.
    from src.model.benders import BendersRegimeIISolver


TOL = 1e-6


@dataclass(frozen=True)
class FacilityCandidate:
    bases: frozenset[str]
    hubs: frozenset[str]
    swaps: frozenset[str]


@dataclass
class LocalSearchSolution:
    obj: Optional[float]
    theta: Optional[float]
    gamma: Optional[float]
    x: Dict[str, float]
    y: Dict[Tuple[str, str], float]
    z: Dict[Tuple[str, int], float]
    status: int
    runtime: float
    wall_time: float
    evaluations: int
    cache_hits: int
    candidate: FacilityCandidate

    def to_master_design(self) -> Dict[str, object]:
        """Return only first-stage decisions needed by Benders."""
        return {
            "x": dict(self.x),
            "y": dict(self.y),
            "z": dict(self.z),
            "Gamma": float(self.gamma or 0.0),
            "eta": float(self.theta or 0.0),
        }


class RegimeIIFixedEvaluator:
    """Evaluate fixed x/y/z/Gamma designs through the current Benders subproblem.

    The old local-search heuristic fixed the monolithic Regime-II model.  For
    Benders warm starts we only need a master design, so this evaluator solves
    the continuous subproblem for several simple Gamma candidates and keeps the
    best total objective.
    """

    def __init__(
        self,
        instance: Instance,
        *,
        eval_time_limit: float = 10.0,
        eval_mip_gap: float = 1e-5,
        gamma_multipliers: Optional[Sequence[float]] = None,
        max_gamma_trials: int = 10,
    ):
        self.inst = instance
        self.solver = BendersRegimeIISolver(instance, use_pareto_cuts=True)
        self.data = self.solver
        self.eval_time_limit = eval_time_limit
        self.eval_mip_gap = eval_mip_gap
        self.gamma_multipliers = tuple(gamma_multipliers or (0.05, 0.10, 0.20, 0.30, 0.40, 0.65, 1.00))
        self.max_gamma_trials = max(1, int(max_gamma_trials))
        self.cache: Dict[Tuple[Tuple[str, ...], Tuple[str, ...], Tuple[str, ...]], LocalSearchSolution] = {}
        self.evaluations = 0
        self.cache_hits = 0

    def repair(
        self,
        bases: Iterable[str],
        hubs: Iterable[str],
        swaps: Iterable[str],
    ) -> FacilityCandidate:
        open_bases = {l for l in bases if l in self.data.L}
        open_hubs = {h for h in hubs if h in self.data.H}
        open_swaps = {k for k in swaps if k in self.data.E}
        a_lh = set(self.data.A_LH)

        repaired_hubs = {
            h
            for h in open_hubs
            if any((l, h) in a_lh for l in self.data.L)
        }
        if not repaired_hubs:
            open_bases = set()
            open_swaps = set()

        for h in repaired_hubs:
            if any((l, h) in a_lh for l in open_bases):
                continue
            feasible_bases = [l for l in self.data.L if (l, h) in a_lh]
            if feasible_bases:
                open_bases.add(min(feasible_bases, key=lambda l: self.inst.get_dist(l, h)))

        return FacilityCandidate(
            frozenset(sorted(open_bases)),
            frozenset(sorted(repaired_hubs)),
            frozenset(sorted(open_swaps)),
        )

    def _capacity_level_for_hub(self, hub: str) -> int:
        mu_levels = list(config.MU_HUB)
        if not mu_levels:
            return 0
        outgoing = sum(float(demand) for (i, _), demand in self.inst.h.items() if i == hub)
        required = outgoing + 1.0 / float(config.TAU_H)
        for c, mu_c in enumerate(mu_levels):
            if float(mu_c) >= 1.05 * required:
                return c
        return len(mu_levels) - 1

    def build_fixed_decisions(
        self,
        candidate: FacilityCandidate,
    ) -> Tuple[Dict[str, float], Dict[Tuple[str, str], float], Dict[Tuple[str, int], float]]:
        x = {u: 0.0 for u in self.data.N_x}
        y = {key: 0.0 for key in self.data.AY}
        z = {(i, c): 0.0 for i in self.data.H for c in range(len(config.MU_HUB))}
        a_lh = set(self.data.A_LH)

        for k in candidate.swaps:
            x[k] = 1.0
        for l in candidate.bases:
            x[l] = 1.0

        for i in candidate.hubs:
            bases = [l for l in candidate.bases if (l, i) in a_lh]
            if not bases:
                continue
            best_l = min(bases, key=lambda l: self.inst.get_dist(l, i))
            x[i] = 1.0
            if (best_l, i) in y and self._has_available_route_option(best_l, i, x):
                y[best_l, i] = 1.0
            if config.MU_HUB:
                z[i, self._capacity_level_for_hub(i)] = 1.0

        def is_open(node: str) -> bool:
            if node in self.inst.S:
                return x.get(self.inst.s_to_h.get(node, ""), 0.0) > 0.5
            return x.get(node, 0.0) > 0.5

        for u, v in self.data.A_HH:
            if (u, v) in y and is_open(u) and is_open(v) and self._has_available_route_option(u, v, x):
                y[u, v] = 1.0

        if len(candidate.hubs) >= 3:
            for i, j in self.data.A_HH:
                if y.get((i, j), 0.0) <= 0.5:
                    continue
                forward = float(self.inst.h.get((i, j), 0.0))
                reverse = float(self.inst.h.get((j, i), 0.0))
                if forward <= reverse + TOL:
                    continue
                s_i = self.inst.h_to_s.get(i)
                s_j = self.inst.h_to_s.get(j)
                if s_i is None or s_j is None:
                    continue
                if (s_j, s_i) in y and self._has_available_route_option(s_j, s_i, x):
                    y[s_j, s_i] = 1.0

        for u, v in self.data.A_hat:
            if y.get((u, v), 0.0) <= 0.5:
                continue
            if (u, v) in self.data.A_LH and self.data._p_chain(v) >= 1.0 - TOL:
                continue
            options = self.data.route_options.get((u, v), [])
            if any(k not in self.data.E for k in options):
                continue
            swap_options = [k for k in options if k in self.data.E and x.get(k, 0.0) > 0.5]
            if not swap_options:
                continue
            k = min(swap_options, key=lambda swap: self.data.t[u, v, swap])
            if (u, k) in y and (k, v) in y:
                y[u, k] = 1.0
                y[k, v] = 1.0

        return x, y, z

    @staticmethod
    def _cache_key(candidate: FacilityCandidate) -> Tuple[Tuple[str, ...], Tuple[str, ...], Tuple[str, ...]]:
        return tuple(sorted(candidate.bases)), tuple(sorted(candidate.hubs)), tuple(sorted(candidate.swaps))

    def _has_available_route_option(self, u: str, v: str, x: Mapping[str, float]) -> bool:
        for k in self.data.route_options.get((u, v), []):
            if k not in self.data.E or x.get(k, 0.0) > 0.5:
                return True
        return False

    def _route_available(self, y: Mapping[Tuple[str, str], float], u: str, v: str, k: str) -> bool:
        if y.get((u, v), 0.0) <= 0.5:
            return False
        if k not in self.data.E:
            return True
        return y.get((u, k), 0.0) > 0.5 and y.get((k, v), 0.0) > 0.5

    def _min_route_time(self, y: Mapping[Tuple[str, str], float], u: str, v: str) -> Optional[float]:
        times = [
            self.data.t[u, v, k]
            for k in self.data.route_options.get((u, v), [])
            if self._route_available(y, u, v, k)
        ]
        if not times:
            return None
        return min(float(value) for value in times)

    def _estimated_fleet_time(self, x: Mapping[str, float], y: Mapping[Tuple[str, str], float]) -> float:
        estimate = 0.0
        for (i, j), demand in self.inst.h.items():
            if x.get(i, 0.0) <= 0.5 or x.get(j, 0.0) <= 0.5:
                continue
            travel_time = self._min_route_time(y, i, j)
            if travel_time is None:
                continue
            estimate += float(demand) * max(travel_time, 1.0 / 120.0)
        return estimate

    def _gamma_candidates(
        self,
        candidate: FacilityCandidate,
        x: Mapping[str, float],
        y: Mapping[Tuple[str, str], float],
    ) -> list[float]:
        if not candidate.hubs:
            return [0.0]

        values = {0}
        base_time = self._estimated_fleet_time(x, y)
        if base_time > TOL:
            values.update(max(1, int(math.ceil(base_time * multiplier))) for multiplier in self.gamma_multipliers)

        hub_count = max(1, len(candidate.hubs))
        values.update({hub_count, 2 * hub_count, 4 * hub_count, 8 * hub_count})
        ordered = sorted(float(value) for value in values if value >= 0)
        if len(ordered) <= self.max_gamma_trials:
            return ordered
        return ordered[: self.max_gamma_trials]

    def _solve_design(self, design: Mapping[str, object], label: str):
        old_limit = getattr(config, "BENDERS_SUB_TIME_LIMIT", None)
        if self.eval_time_limit is not None and self.eval_time_limit > 0:
            setattr(config, "BENDERS_SUB_TIME_LIMIT", float(self.eval_time_limit))
        try:
            return self.solver.solve_subproblem(design, label=label)
        finally:
            if old_limit is None and hasattr(config, "BENDERS_SUB_TIME_LIMIT"):
                delattr(config, "BENDERS_SUB_TIME_LIMIT")
            elif old_limit is not None:
                setattr(config, "BENDERS_SUB_TIME_LIMIT", old_limit)

    def evaluate(self, candidate: FacilityCandidate) -> LocalSearchSolution:
        candidate = self.repair(candidate.bases, candidate.hubs, candidate.swaps)
        key = self._cache_key(candidate)
        if key in self.cache:
            self.cache_hits += 1
            cached = self.cache[key]
            return LocalSearchSolution(
                obj=cached.obj,
                theta=cached.theta,
                gamma=cached.gamma,
                x=dict(cached.x),
                y=dict(cached.y),
                z=dict(cached.z),
                status=cached.status,
                runtime=cached.runtime,
                wall_time=0.0,
                evaluations=self.evaluations,
                cache_hits=self.cache_hits,
                candidate=candidate,
            )

        x, y, z = self.build_fixed_decisions(candidate)
        gamma_values = self._gamma_candidates(candidate, x, y)

        self.evaluations += 1
        start = time.perf_counter()
        best_obj: Optional[float] = None
        best_theta: Optional[float] = None
        best_gamma: Optional[float] = None
        best_status = GRB.INFEASIBLE
        runtime = 0.0

        for g_idx, gamma in enumerate(gamma_values):
            design = {"x": x, "y": y, "z": z, "Gamma": float(gamma)}
            sub = self._solve_design(design, label=f"local_{self.evaluations}_{g_idx}")
            best_status = int(sub.status)
            runtime += float(getattr(sub, "runtime", 0.0) or 0.0)
            if sub.obj_value is None:
                continue
            obj = float(sub.obj_value) - self.solver._fixed_cost_value(design)
            if best_obj is None or obj > best_obj + 1e-7:
                best_obj = obj
                best_theta = float(sub.obj_value)
                best_gamma = float(gamma)

        wall = time.perf_counter() - start
        solution = LocalSearchSolution(
            obj=best_obj,
            theta=best_theta,
            gamma=best_gamma,
            x=x,
            y=y,
            z=z,
            status=best_status,
            runtime=runtime,
            wall_time=wall,
            evaluations=self.evaluations,
            cache_hits=self.cache_hits,
            candidate=candidate,
        )
        self.cache[key] = solution
        return solution


class RegimeIILocalSearch:
    """Add/drop/swap local search over opened hubs and swapping stations."""

    def __init__(
        self,
        instance: Instance,
        *,
        eval_time_limit: float = 10.0,
        eval_mip_gap: float = 1e-5,
        max_seed_hubs: int = 6,
        max_seed_candidates: Optional[int] = 250,
        max_iter: int = 25,
        gamma_multipliers: Optional[Sequence[float]] = None,
        max_gamma_trials: int = 10,
        verbose: bool = False,
    ):
        self.inst = instance
        self.evaluator = RegimeIIFixedEvaluator(
            instance,
            eval_time_limit=eval_time_limit,
            eval_mip_gap=eval_mip_gap,
            gamma_multipliers=gamma_multipliers,
            max_gamma_trials=max_gamma_trials,
        )
        self.data = self.evaluator.data
        self.max_seed_hubs = max_seed_hubs
        self.max_seed_candidates = max_seed_candidates
        self.max_iter = max_iter
        self.verbose = verbose

    def _hub_scores(self) -> Dict[str, float]:
        scores = {h: 0.0 for h in self.data.H}
        for (i, j), demand in self.inst.h.items():
            scores[i] = scores.get(i, 0.0) + float(demand)
            scores[j] = scores.get(j, 0.0) + float(demand)
        return scores

    def _seed_candidates(self) -> list[FacilityCandidate]:
        scores = self._hub_scores()
        ranked_hubs = [h for h, _ in sorted(scores.items(), key=lambda item: item[1], reverse=True)]
        swap_sets: list[Set[str]] = [set()]
        swap_sets.extend([{k} for k in self.data.E])
        if self.data.E:
            swap_sets.append(set(self.data.E))

        base_sets: list[Set[str]] = [{l} for l in self.data.L]
        if self.data.L:
            base_sets.append(set(self.data.L))

        seeds = [FacilityCandidate(frozenset(), frozenset(), frozenset())]
        top_pairs = sorted(self.inst.h.items(), key=lambda item: float(item[1]), reverse=True)
        for (i, j), _ in top_pairs[: max(1, 2 * self.max_seed_hubs)]:
            if i not in self.data.H or j not in self.data.H:
                continue
            hubs = {i, j}
            bases = set()
            for h in hubs:
                feasible = [l for l in self.data.L if (l, h) in self.data.A_LH]
                if feasible:
                    bases.add(min(feasible, key=lambda l: self.inst.get_dist(l, h)))
            route_swaps = {
                k
                for arc in ((i, j), (j, i))
                for k in self.data.route_options.get(arc, [])
                if k in self.data.E
            }
            od_swap_sets: list[Set[str]] = [set()]
            od_swap_sets.extend({k} for k in sorted(route_swaps))
            if len(route_swaps) > 1:
                od_swap_sets.append(set(route_swaps))
            for swaps in od_swap_sets:
                seeds.append(FacilityCandidate(frozenset(bases), frozenset(hubs), frozenset(swaps)))

        for n in range(1, min(self.max_seed_hubs, len(ranked_hubs)) + 1):
            hubs = set(ranked_hubs[:n])
            nearest_bases = set()
            for h in hubs:
                feasible = [l for l in self.data.L if (l, h) in self.data.A_LH]
                if feasible:
                    nearest_bases.add(min(feasible, key=lambda l: self.inst.get_dist(l, h)))
            needed_swaps = {
                k
                for (i, j), _ in top_pairs
                if i in hubs and j in hubs
                for k in self.data.route_options.get((i, j), [])
                if k in self.data.E
            }
            seeds.append(FacilityCandidate(frozenset(nearest_bases), frozenset(hubs), frozenset(needed_swaps)))

        for n in range(1, min(self.max_seed_hubs, len(ranked_hubs)) + 1):
            hubs = set(ranked_hubs[:n])
            for bases in base_sets:
                for swaps in swap_sets:
                    seeds.append(FacilityCandidate(frozenset(bases), frozenset(hubs), frozenset(swaps)))
        for h in ranked_hubs[: self.max_seed_hubs]:
            for bases in base_sets:
                for swaps in swap_sets:
                    seeds.append(FacilityCandidate(frozenset(bases), frozenset({h}), frozenset(swaps)))
        unique = list(dict.fromkeys(self.evaluator.repair(s.bases, s.hubs, s.swaps) for s in seeds))
        if self.max_seed_candidates is not None:
            return unique[: max(1, int(self.max_seed_candidates))]
        return unique

    def _neighbors(self, candidate: FacilityCandidate) -> list[FacilityCandidate]:
        bases = set(candidate.bases)
        hubs = set(candidate.hubs)
        swaps = set(candidate.swaps)
        neighbors = []

        for l in self.data.L:
            if l not in bases:
                neighbors.append(FacilityCandidate(frozenset(bases | {l}), frozenset(hubs), frozenset(swaps)))
        for l in list(bases):
            neighbors.append(FacilityCandidate(frozenset(bases - {l}), frozenset(hubs), frozenset(swaps)))
        for l_out in list(bases):
            for l_in in self.data.L:
                if l_in not in bases:
                    neighbors.append(FacilityCandidate(frozenset((bases - {l_out}) | {l_in}), frozenset(hubs), frozenset(swaps)))

        for h in self.data.H:
            if h not in hubs:
                neighbors.append(FacilityCandidate(frozenset(bases), frozenset(hubs | {h}), frozenset(swaps)))
        for h in list(hubs):
            neighbors.append(FacilityCandidate(frozenset(bases), frozenset(hubs - {h}), frozenset(swaps)))
        for h_out in list(hubs):
            for h_in in self.data.H:
                if h_in not in hubs:
                    neighbors.append(FacilityCandidate(frozenset(bases), frozenset((hubs - {h_out}) | {h_in}), frozenset(swaps)))

        for k in self.data.E:
            if k not in swaps:
                neighbors.append(FacilityCandidate(frozenset(bases), frozenset(hubs), frozenset(swaps | {k})))
        for k in list(swaps):
            neighbors.append(FacilityCandidate(frozenset(bases), frozenset(hubs), frozenset(swaps - {k})))
        for k_out in list(swaps):
            for k_in in self.data.E:
                if k_in not in swaps:
                    neighbors.append(FacilityCandidate(frozenset(bases), frozenset(hubs), frozenset((swaps - {k_out}) | {k_in})))

        repaired = [self.evaluator.repair(n.bases, n.hubs, n.swaps) for n in neighbors]
        return list(dict.fromkeys(repaired))

    @staticmethod
    def _is_better(candidate: LocalSearchSolution, incumbent: Optional[LocalSearchSolution]) -> bool:
        if candidate.obj is None:
            return False
        if incumbent is None or incumbent.obj is None:
            return True
        return candidate.obj > incumbent.obj + 1e-5

    def solve(self) -> LocalSearchSolution:
        start = time.perf_counter()
        best: Optional[LocalSearchSolution] = None
        current: Optional[LocalSearchSolution] = None

        for seed in self._seed_candidates():
            sol = self.evaluator.evaluate(seed)
            if self._is_better(sol, best):
                best = sol
                current = sol

        if current is None:
            current = self.evaluator.evaluate(FacilityCandidate(frozenset(), frozenset(), frozenset()))
            best = current

        for _ in range(self.max_iter):
            next_sol = current
            for neighbor in self._neighbors(current.candidate):
                sol = self.evaluator.evaluate(neighbor)
                if self._is_better(sol, next_sol):
                    next_sol = sol
            if next_sol is current or not self._is_better(next_sol, current):
                break
            current = next_sol
            if self._is_better(current, best):
                best = current

        assert best is not None
        best.wall_time = time.perf_counter() - start
        best.evaluations = self.evaluator.evaluations
        best.cache_hits = self.evaluator.cache_hits
        return best
