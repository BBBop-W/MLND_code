import math
import time
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Set, Tuple

from gurobipy import GRB

from src import config
from src.data_processor.instance import Instance
from src.model.benders import RegimeIIData
from src.model.gurobi_model import MLNDModel


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


class RegimeIIFixedEvaluator:
    """Evaluate fixed x/y/z decisions with one persistent Regime-II model."""

    def __init__(
        self,
        instance: Instance,
        *,
        eval_time_limit: float = 10.0,
        eval_mip_gap: float = 1e-5,
    ):
        self.inst = instance
        self.data = RegimeIIData(instance)
        self.eval_time_limit = eval_time_limit
        self.eval_mip_gap = eval_mip_gap
        self.model = MLNDModel(instance)
        self.model.build()
        self.model.model.setParam("OutputFlag", 0)
        self.model.model.setParam("TimeLimit", eval_time_limit)
        self.model.model.setParam("MIPGap", eval_mip_gap)
        self.cache: Dict[Tuple[Tuple[str, ...], Tuple[str, ...]], LocalSearchSolution] = {}
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
        repaired_hubs = {
            h
            for h in open_hubs
            if any((l, h) in self.data.A_LH_set for l in self.data.L)
        }
        if not repaired_hubs:
            open_bases = set()
            open_swaps = set()
        for h in repaired_hubs:
            if any((l, h) in self.data.A_LH_set for l in open_bases):
                continue
            feasible_bases = [l for l in self.data.L if (l, h) in self.data.A_LH_set]
            if feasible_bases:
                open_bases.add(min(feasible_bases, key=lambda l: self.inst.get_dist(l, h)))
        return FacilityCandidate(
            frozenset(sorted(open_bases)),
            frozenset(sorted(repaired_hubs)),
            frozenset(sorted(open_swaps)),
        )

    def build_fixed_decisions(
        self,
        candidate: FacilityCandidate,
    ) -> Tuple[Dict[str, float], Dict[Tuple[str, str], float], Dict[Tuple[str, int], float]]:
        x = {u: 0.0 for u in self.data.N_x}
        y = {(i, l): 0.0 for l, i in self.data.A_LH}
        z = {(i, c): 0.0 for i in self.data.H for c in self.data.C}
        cap = max(self.data.C, key=lambda c: self.data.mu_levels[c]) if self.data.C else None

        for k in candidate.swaps:
            x[k] = 1.0
        for l in candidate.bases:
            x[l] = 1.0

        for i in candidate.hubs:
            bases = [l for l in candidate.bases if (l, i) in self.data.A_LH_set]
            if not bases:
                continue
            best_l = min(bases, key=lambda l: self.inst.get_dist(l, i))
            x[i] = 1.0
            y[i, best_l] = 1.0
            if cap is not None:
                z[i, cap] = 1.0

        return x, y, z

    @staticmethod
    def _cache_key(candidate: FacilityCandidate) -> Tuple[Tuple[str, ...], Tuple[str, ...], Tuple[str, ...]]:
        return tuple(sorted(candidate.bases)), tuple(sorted(candidate.hubs)), tuple(sorted(candidate.swaps))

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
        x_vars = self.model.vars["x"]
        y_vars = self.model.vars["y"]
        z_vars = self.model.vars["z"]

        for u, var in x_vars.items():
            value = x.get(u, 0.0)
            var.LB = value
            var.UB = value
        for key_y, var in y_vars.items():
            value = y.get(key_y, 0.0)
            var.LB = value
            var.UB = value
        for key_z, var in z_vars.items():
            value = z.get(key_z, 0.0)
            var.LB = value
            var.UB = value

        self.evaluations += 1
        start = time.perf_counter()
        self.model.model.optimize()
        wall = time.perf_counter() - start

        status = int(self.model.model.Status)
        obj = None
        theta = None
        gamma = None
        if self.model.model.SolCount > 0:
            obj = float(self.model.model.ObjVal)
            gamma = float(self.model.vars["Gamma"].X)
            theta = obj + self.data.fixed_cost_value(x, z) + self.data.gamma_cost * gamma

        solution = LocalSearchSolution(
            obj=obj,
            theta=theta,
            gamma=gamma,
            x=x,
            y=y,
            z=z,
            status=status,
            runtime=float(self.model.model.Runtime),
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
        max_iter: int = 25,
        verbose: bool = False,
    ):
        self.inst = instance
        self.evaluator = RegimeIIFixedEvaluator(
            instance,
            eval_time_limit=eval_time_limit,
            eval_mip_gap=eval_mip_gap,
        )
        self.data = self.evaluator.data
        self.max_seed_hubs = max_seed_hubs
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
        for n in range(1, min(self.max_seed_hubs, len(ranked_hubs)) + 1):
            hubs = set(ranked_hubs[:n])
            for bases in base_sets:
                for swaps in swap_sets:
                    seeds.append(FacilityCandidate(frozenset(bases), frozenset(hubs), frozenset(swaps)))
        for h in ranked_hubs[: self.max_seed_hubs]:
            for bases in base_sets:
                for swaps in swap_sets:
                    seeds.append(FacilityCandidate(frozenset(bases), frozenset({h}), frozenset(swaps)))
        return list(dict.fromkeys(self.evaluator.repair(s.bases, s.hubs, s.swaps) for s in seeds))

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
