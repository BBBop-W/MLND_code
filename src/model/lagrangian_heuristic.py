import math
import time
from dataclasses import dataclass, field
from typing import Dict, Iterable, Optional, Sequence, Tuple

from src import config
from src.data_processor.instance import Instance
from src.model.benders import RegimeIIData
from src.model.local_search_heuristic import (
    FacilityCandidate,
    LocalSearchSolution,
    RegimeIIFixedEvaluator,
)


Pair = Tuple[str, str]


def _solution_value(sol: Optional[LocalSearchSolution]) -> float:
    if sol is None or sol.obj is None:
        return -math.inf
    return float(sol.obj)


@dataclass
class LagrangianIteration:
    iteration: int
    lagrangian_bound: float
    open_hubs: int
    open_swaps: int
    max_violation: float
    step_size: float


@dataclass
class HeuristicRun:
    name: str
    solution: LocalSearchSolution
    wall_time: float
    evaluations: int
    cache_hits: int
    history: list[dict] = field(default_factory=list)


class CandidateBuilder:
    """Convert hub/swap scores into feasible first-stage candidates."""

    def __init__(self, data: RegimeIIData):
        self.data = data
        self.inst = data.inst
        self.hub_activity = self._hub_activity()

    def _hub_activity(self) -> Dict[str, float]:
        activity = {h: 0.0 for h in self.data.H}
        for (i, j), demand in self.inst.h.items():
            val = float(demand)
            activity[i] = activity.get(i, 0.0) + val
            activity[j] = activity.get(j, 0.0) + val
        return activity

    def _base_score(self, base: str, hubs: Iterable[str]) -> float:
        return sum(
            self.hub_activity.get(h, 0.0)
            for h in hubs
            if (base, h) in self.data.A_LH_set
        )

    def choose_bases(self, hubs: Iterable[str]) -> set[str]:
        remaining = set(hubs)
        bases: set[str] = set()
        while remaining:
            ranked = sorted(
                self.data.L,
                key=lambda l: (self._base_score(l, remaining), -self.inst.get_dist(l, next(iter(remaining)))),
                reverse=True,
            )
            if not ranked:
                break
            best = ranked[0]
            covered = {h for h in remaining if (best, h) in self.data.A_LH_set}
            if not covered:
                break
            bases.add(best)
            remaining -= covered
        return bases

    def choose_swaps(
        self,
        scores: Dict[str, float],
        *,
        max_swaps: Optional[int],
        min_positive: bool = True,
    ) -> set[str]:
        ranked = sorted(self.data.E, key=lambda k: scores.get(k, 0.0), reverse=True)
        if max_swaps is not None:
            ranked = ranked[: max(0, int(max_swaps))]
        if min_positive:
            return {k for k in ranked if scores.get(k, 0.0) > 0.0}
        return set(ranked)

    def build(
        self,
        hubs: Iterable[str],
        swap_scores: Optional[Dict[str, float]] = None,
        *,
        max_swaps: Optional[int] = None,
        swaps: Optional[Iterable[str]] = None,
    ) -> FacilityCandidate:
        open_hubs = {h for h in hubs if h in self.data.H}
        if not open_hubs:
            return FacilityCandidate(frozenset(), frozenset(), frozenset())

        open_bases = self.choose_bases(open_hubs)
        if swaps is None:
            swap_scores = swap_scores or {}
            open_swaps = self.choose_swaps(swap_scores, max_swaps=max_swaps)
        else:
            open_swaps = {k for k in swaps if k in self.data.E}

        return FacilityCandidate(
            frozenset(sorted(open_bases)),
            frozenset(sorted(open_hubs)),
            frozenset(sorted(open_swaps)),
        )


class RegimeIIConstructiveHeuristic:
    """Simple constructive heuristic for a single Regime-II instance.

    The heuristic ranks hubs by OD activity, selects a small set of ranked hub
    prefixes, repairs bases/swaps into feasible first-stage candidates, and
    evaluates the best candidates with the fixed first-stage SOCP evaluator.
    """

    def __init__(
        self,
        instance: Instance,
        *,
        evaluator: Optional[RegimeIIFixedEvaluator] = None,
        max_hubs: Optional[int] = None,
        max_swaps: Optional[int] = None,
        candidate_hubs: int = 15,
        candidate_swaps: int = 10,
        eval_top: int = 25,
        verbose: bool = False,
    ):
        self.inst = instance
        self.data = RegimeIIData(instance)
        self.evaluator = evaluator or RegimeIIFixedEvaluator(instance)
        self.builder = CandidateBuilder(self.data)
        self.max_hubs = max_hubs
        self.max_swaps = max_swaps
        self.candidate_hubs = max(1, int(candidate_hubs))
        self.candidate_swaps = max(1, int(candidate_swaps))
        self.eval_top = max(1, int(eval_top))
        self.verbose = verbose
        self.history: list[dict] = []

    def _hub_fixed_cost(self) -> float:
        hub_fixed = float(getattr(config, "COST_HUB_FIXED", 0.0))
        if self.data.C:
            hub_fixed += min(float(config.COST_HUB[c]) for c in self.data.C)
        return hub_fixed

    def _hub_scores(self) -> Dict[str, float]:
        scores = {h: -self._hub_fixed_cost() for h in self.data.H}
        for (i, j), demand in self.inst.h.items():
            value = self.data.revenue * float(demand)
            scores[i] = scores.get(i, 0.0) + value
            scores[j] = scores.get(j, 0.0) + value
        return scores

    def _swap_scores(self) -> Dict[str, float]:
        scores = {k: -float(self.inst.f.get(k, config.COST_SWAPPING)) for k in self.data.E}
        for i, j in self.data.A_HH:
            demand = float(self.inst.h.get((i, j), 0.0))
            options = self.data.route_options.get((i, j), [])
            if demand <= 0.0 or not options:
                continue
            direct_cost = self.data.flight_cost[i, j, "0"] if "0" in options else None
            for k in options:
                if k not in self.data.E:
                    continue
                if direct_cost is None:
                    scores[k] += 0.25 * self.data.revenue * demand
                else:
                    scores[k] += max(0.0, direct_cost - self.data.flight_cost[i, j, k]) * demand
        return scores

    def _hub_sets(self, hub_scores: Dict[str, float]) -> list[set[str]]:
        ranked = sorted(self.data.H, key=lambda h: hub_scores.get(h, 0.0), reverse=True)
        ranked = ranked[: min(self.candidate_hubs, len(ranked))]
        cap = len(ranked) if self.max_hubs is None else min(int(self.max_hubs), len(ranked))
        checkpoints = {1, 2, 3, 5, 8, 13, cap}
        positive = [h for h in ranked if hub_scores.get(h, 0.0) > 0.0]
        if self.max_hubs is not None:
            positive = positive[: int(self.max_hubs)]

        hub_sets: list[set[str]] = []
        if positive:
            hub_sets.append(set(positive))
        for size in sorted(checkpoints):
            if size <= 0 or not ranked:
                continue
            hub_sets.append(set(ranked[: min(size, cap)]))

        unique: list[set[str]] = []
        seen = set()
        for hubs in hub_sets:
            key = tuple(sorted(hubs))
            if key in seen:
                continue
            seen.add(key)
            unique.append(hubs)
        return unique

    def _swap_sets(self, swap_scores: Dict[str, float]) -> list[set[str]]:
        if not self.data.E:
            return [set()]
        ranked = sorted(self.data.E, key=lambda k: swap_scores.get(k, 0.0), reverse=True)
        ranked = ranked[: min(self.candidate_swaps, len(ranked))]
        cap = len(ranked) if self.max_swaps is None else min(int(self.max_swaps), len(ranked))
        positive = [k for k in ranked if swap_scores.get(k, 0.0) > 0.0]
        if self.max_swaps is not None:
            positive = positive[: int(self.max_swaps)]

        swap_sets: list[set[str]] = [set()]
        if positive:
            swap_sets.append(set(positive))
        for size in (1, 2, 3, 5, cap):
            if size <= 0:
                continue
            swap_sets.append(set(ranked[: min(size, cap)]))

        unique: list[set[str]] = []
        seen = set()
        for swaps in swap_sets:
            key = tuple(sorted(swaps))
            if key in seen:
                continue
            seen.add(key)
            unique.append(swaps)
        return unique

    def solve(self) -> HeuristicRun:
        start = time.perf_counter()
        eval_start = self.evaluator.evaluations
        cache_start = self.evaluator.cache_hits

        hub_scores = self._hub_scores()
        swap_scores = self._swap_scores()
        candidates: Dict[FacilityCandidate, float] = {
            FacilityCandidate(frozenset(), frozenset(), frozenset()): 0.0
        }
        for hubs in self._hub_sets(hub_scores):
            for swaps in self._swap_sets(swap_scores):
                candidate = self.builder.build(hubs, swaps=swaps)
                score = sum(hub_scores.get(h, 0.0) for h in candidate.hubs)
                score += sum(swap_scores.get(k, 0.0) for k in candidate.swaps)
                candidates[candidate] = max(candidates.get(candidate, -math.inf), score)

        ranked_candidates = sorted(candidates.items(), key=lambda item: item[1], reverse=True)
        best_sol: Optional[LocalSearchSolution] = None
        for candidate, score in ranked_candidates[: self.eval_top]:
            sol = self.evaluator.evaluate(candidate)
            if _solution_value(sol) > _solution_value(best_sol):
                best_sol = sol
            self.history.append(
                {
                    "candidate_score": score,
                    "open_hubs": len(candidate.hubs),
                    "open_bases": len(candidate.bases),
                    "open_swaps": len(candidate.swaps),
                    "obj": sol.obj,
                }
            )

        if best_sol is None:
            empty = FacilityCandidate(frozenset(), frozenset(), frozenset())
            best_sol = self.evaluator.evaluate(empty)

        wall = time.perf_counter() - start
        best_sol.wall_time = wall
        best_sol.evaluations = self.evaluator.evaluations - eval_start
        best_sol.cache_hits = self.evaluator.cache_hits - cache_start
        return HeuristicRun(
            name="constructive",
            solution=best_sol,
            wall_time=wall,
            evaluations=best_sol.evaluations,
            cache_hits=best_sol.cache_hits,
            history=self.history,
        )


class RegimeIILagrangianHeuristic:
    """Lagrangian-relaxation candidate generator for the single-period model.

    This is a pragmatic LR heuristic, not a replacement for the exact conic
    model.  It relaxes simplified OD-pair activation constraints, then repairs
    the resulting open-hub set and evaluates it with the exact fixed first-stage
    SOCP used by the main model.
    """

    def __init__(
        self,
        instance: Instance,
        *,
        decomposition: str = "coverage",
        evaluator: Optional[RegimeIIFixedEvaluator] = None,
        max_iter: int = 40,
        initial_step: float = 20.0,
        max_hubs: Optional[int] = None,
        max_swaps: Optional[int] = None,
        eval_top: int = 30,
        verbose: bool = False,
    ):
        if decomposition not in {"coverage", "route_swap"}:
            raise ValueError("decomposition must be 'coverage' or 'route_swap'.")
        self.inst = instance
        self.data = RegimeIIData(instance)
        self.evaluator = evaluator or RegimeIIFixedEvaluator(instance)
        self.decomposition = decomposition
        self.max_iter = max(1, int(max_iter))
        self.initial_step = float(initial_step)
        self.max_hubs = max_hubs
        self.max_swaps = max_swaps
        self.eval_top = max(1, int(eval_top))
        self.verbose = verbose
        self.builder = CandidateBuilder(self.data)
        self.history: list[LagrangianIteration] = []
        self.last_hub_scores: Dict[str, float] = {h: 0.0 for h in self.data.H}
        self.last_swap_scores: Dict[str, float] = {k: 0.0 for k in self.data.E}

    def _hub_fixed_cost(self) -> float:
        hub_fixed = float(getattr(config, "COST_HUB_FIXED", 0.0))
        if self.data.C:
            hub_fixed += min(float(config.COST_HUB[c]) for c in self.data.C)
        return hub_fixed

    def _base_share_cost(self) -> float:
        if not self.data.L or not self.data.H:
            return 0.0
        avg_base = sum(float(self.inst.f.get(l, config.COST_BASE)) for l in self.data.L) / len(self.data.L)
        expected_hubs_per_base = max(1.0, len(self.data.H) / max(1.0, len(self.data.L)))
        return avg_base / expected_hubs_per_base

    def _pair_values(self) -> Dict[Pair, float]:
        values: Dict[Pair, float] = {}
        for i, j in self.data.A_HH:
            demand = float(self.inst.h.get((i, j), 0.0))
            if demand <= 0.0:
                continue
            value = self.data.revenue * demand
            if self.decomposition == "route_swap":
                options = self.data.route_options.get((i, j), [])
                if not options:
                    value = 0.0
                else:
                    min_cost = min(self.data.flight_cost[i, j, k] for k in options)
                    value -= demand * min_cost
            if value > 0.0:
                values[i, j] = value
        return values

    def _swap_scores(self, active_pairs: Dict[Pair, int], pair_values: Dict[Pair, float]) -> Dict[str, float]:
        scores = {
            k: -float(self.inst.f.get(k, config.COST_SWAPPING))
            for k in self.data.E
        }
        if self.decomposition != "route_swap":
            return scores

        for (i, j), is_active in active_pairs.items():
            if not is_active:
                continue
            options = self.data.route_options.get((i, j), [])
            if not options:
                continue
            demand = float(self.inst.h.get((i, j), 0.0))
            direct_cost = None
            if "0" in options:
                direct_cost = self.data.flight_cost[i, j, "0"]
            for k in options:
                if k not in self.data.E:
                    continue
                if direct_cost is None:
                    scores[k] += 0.25 * pair_values.get((i, j), 0.0)
                else:
                    scores[k] += max(0.0, direct_cost - self.data.flight_cost[i, j, k]) * demand
        return scores

    def _ranked_hub_sets(self, scores: Dict[str, float], open_hubs: set[str]) -> list[set[str]]:
        ranked = sorted(self.data.H, key=lambda h: scores.get(h, 0.0), reverse=True)
        hub_sets: list[set[str]] = [set(open_hubs)]
        checkpoints = {1, 2, 3, 5, 8, 13}
        if self.max_hubs is not None:
            checkpoints.add(int(self.max_hubs))
        for k in sorted(checkpoints):
            if k <= 0:
                continue
            if self.max_hubs is not None:
                k = min(k, int(self.max_hubs))
            hub_sets.append(set(ranked[: min(k, len(ranked))]))

        positive = [h for h in ranked if scores.get(h, 0.0) > 0.0]
        if self.max_hubs is not None:
            positive = positive[: int(self.max_hubs)]
        hub_sets.append(set(positive))

        unique: list[set[str]] = []
        seen = set()
        for hubs in hub_sets:
            key = tuple(sorted(hubs))
            if key not in seen:
                seen.add(key)
                unique.append(hubs)
        return unique

    def _swap_variants(self, swap_scores: Dict[str, float]) -> list[Optional[set[str]]]:
        variants: list[Optional[set[str]]] = [None, set()]
        if not self.data.E:
            return variants

        ranked = sorted(self.data.E, key=lambda k: swap_scores.get(k, 0.0), reverse=True)
        cap = len(ranked) if self.max_swaps is None else min(int(self.max_swaps), len(ranked))
        if cap > 0:
            variants.append(set(ranked[:cap]))
        for k in (1, 2, 3, 5):
            if cap <= 0:
                break
            variants.append(set(ranked[: min(k, cap)]))

        unique: list[Optional[set[str]]] = []
        seen = set()
        for swaps in variants:
            key = None if swaps is None else tuple(sorted(swaps))
            if key in seen:
                continue
            seen.add(key)
            unique.append(swaps)
        return unique

    def _relaxed_step(
        self,
        pair_values: Dict[Pair, float],
        lambda_o: Dict[Pair, float],
        lambda_d: Dict[Pair, float],
    ) -> tuple[Dict[Pair, int], set[str], Dict[str, float], float]:
        active_pairs: Dict[Pair, int] = {}
        bound = 0.0
        for pair, value in pair_values.items():
            coeff = value - lambda_o[pair] - lambda_d[pair]
            active = 1 if coeff > 0.0 else 0
            active_pairs[pair] = active
            if active:
                bound += coeff

        hub_cost = self._hub_fixed_cost() + self._base_share_cost()
        hub_scores = {h: -hub_cost for h in self.data.H}
        for (i, j), value in pair_values.items():
            hub_scores[i] = hub_scores.get(i, -hub_cost) + lambda_o[i, j]
            hub_scores[j] = hub_scores.get(j, -hub_cost) + lambda_d[i, j]

        ranked = sorted(self.data.H, key=lambda h: hub_scores.get(h, 0.0), reverse=True)
        open_hubs = {h for h in ranked if hub_scores.get(h, 0.0) > 0.0}
        if self.max_hubs is not None:
            open_hubs = set(ranked[: min(int(self.max_hubs), len(ranked))]) & open_hubs
        if not open_hubs and ranked and pair_values:
            open_hubs = {ranked[0]}

        for h in open_hubs:
            bound += max(0.0, hub_scores.get(h, 0.0))
        return active_pairs, open_hubs, hub_scores, bound

    def solve(self) -> HeuristicRun:
        start = time.perf_counter()
        eval_start = self.evaluator.evaluations
        cache_start = self.evaluator.cache_hits

        pair_values = self._pair_values()
        lambda_o = {pair: 0.0 for pair in pair_values}
        lambda_d = {pair: 0.0 for pair in pair_values}
        candidates: Dict[FacilityCandidate, float] = {}
        best_violation = 0.0

        empty = FacilityCandidate(frozenset(), frozenset(), frozenset())
        candidates[empty] = 0.0

        for it in range(1, self.max_iter + 1):
            active_pairs, open_hubs, hub_scores, lag_bound = self._relaxed_step(
                pair_values,
                lambda_o,
                lambda_d,
            )
            swap_scores = self._swap_scores(active_pairs, pair_values)
            self.last_hub_scores = hub_scores
            self.last_swap_scores = swap_scores

            for hubs in self._ranked_hub_sets(hub_scores, open_hubs):
                for swaps in self._swap_variants(swap_scores):
                    candidate = self.builder.build(
                        hubs,
                        swap_scores,
                        max_swaps=self.max_swaps,
                        swaps=swaps,
                    )
                    score = sum(hub_scores.get(h, 0.0) for h in hubs)
                    score += sum(max(0.0, swap_scores.get(k, 0.0)) for k in candidate.swaps)
                    candidates[candidate] = max(candidates.get(candidate, -math.inf), score)

            step = self.initial_step / math.sqrt(it)
            max_violation = 0.0
            for (i, j), active in active_pairs.items():
                vi = float(active) - (1.0 if i in open_hubs else 0.0)
                vj = float(active) - (1.0 if j in open_hubs else 0.0)
                lambda_o[i, j] = max(0.0, lambda_o[i, j] + step * vi)
                lambda_d[i, j] = max(0.0, lambda_d[i, j] + step * vj)
                max_violation = max(max_violation, abs(vi), abs(vj))

            best_violation = max_violation
            self.history.append(
                LagrangianIteration(
                    iteration=it,
                    lagrangian_bound=lag_bound,
                    open_hubs=len(open_hubs),
                    open_swaps=len([k for k, v in swap_scores.items() if v > 0.0]),
                    max_violation=max_violation,
                    step_size=step,
                )
            )
            if self.verbose:
                print(
                    f"[LR-{self.decomposition}] it={it} bound={lag_bound:.4f} "
                    f"hubs={len(open_hubs)} violation={max_violation:.3f}"
                )

        ranked_candidates = sorted(candidates.items(), key=lambda item: item[1], reverse=True)
        best_sol: Optional[LocalSearchSolution] = None
        for candidate, _ in ranked_candidates[: self.eval_top]:
            sol = self.evaluator.evaluate(candidate)
            if _solution_value(sol) > _solution_value(best_sol):
                best_sol = sol

        if best_sol is None:
            best_sol = self.evaluator.evaluate(empty)

        wall = time.perf_counter() - start
        best_sol.wall_time = wall
        best_sol.evaluations = self.evaluator.evaluations - eval_start
        best_sol.cache_hits = self.evaluator.cache_hits - cache_start

        return HeuristicRun(
            name=f"lr_{self.decomposition}",
            solution=best_sol,
            wall_time=wall,
            evaluations=best_sol.evaluations,
            cache_hits=best_sol.cache_hits,
            history=[
                {
                    "iteration": item.iteration,
                    "lagrangian_bound": item.lagrangian_bound,
                    "open_hubs": item.open_hubs,
                    "open_swaps": item.open_swaps,
                    "max_violation": item.max_violation,
                    "step_size": item.step_size,
                }
                for item in self.history
            ]
            + [{"final_max_violation": best_violation}],
        )


class RegimeIILagrangianTabuSearch:
    """Tabu search seeded and ranked by LR reduced-cost style scores."""

    def __init__(
        self,
        instance: Instance,
        *,
        evaluator: Optional[RegimeIIFixedEvaluator] = None,
        hub_scores: Optional[Dict[str, float]] = None,
        swap_scores: Optional[Dict[str, float]] = None,
        max_iter: int = 25,
        tenure: int = 5,
        candidate_hubs: int = 15,
        candidate_swaps: int = 10,
        max_neighbors: int = 80,
        max_swaps: Optional[int] = None,
        verbose: bool = False,
    ):
        self.inst = instance
        self.data = RegimeIIData(instance)
        self.evaluator = evaluator or RegimeIIFixedEvaluator(instance)
        self.builder = CandidateBuilder(self.data)
        self.hub_scores = hub_scores or self.builder.hub_activity
        self.swap_scores = swap_scores or {k: 0.0 for k in self.data.E}
        self.max_iter = max(1, int(max_iter))
        self.tenure = max(1, int(tenure))
        self.candidate_hubs = max(1, int(candidate_hubs))
        self.candidate_swaps = max(1, int(candidate_swaps))
        self.max_neighbors = max(1, int(max_neighbors))
        self.max_swaps = max_swaps
        self.verbose = verbose
        self.history: list[dict] = []

    def _ranked_hubs(self, current: FacilityCandidate) -> list[str]:
        pool = set(current.hubs)
        ranked = sorted(self.data.H, key=lambda h: self.hub_scores.get(h, 0.0), reverse=True)
        pool.update(ranked[: min(self.candidate_hubs, len(ranked))])
        return sorted(pool, key=lambda h: self.hub_scores.get(h, 0.0), reverse=True)

    def _ranked_swaps(self, current: FacilityCandidate) -> list[str]:
        pool = set(current.swaps)
        ranked = sorted(self.data.E, key=lambda k: self.swap_scores.get(k, 0.0), reverse=True)
        pool.update(ranked[: min(self.candidate_swaps, len(ranked))])
        return sorted(pool, key=lambda k: self.swap_scores.get(k, 0.0), reverse=True)

    def _candidate(self, hubs: Iterable[str], swaps: Iterable[str]) -> FacilityCandidate:
        open_swaps = set(swaps)
        if self.max_swaps is not None and len(open_swaps) > self.max_swaps:
            ranked = sorted(open_swaps, key=lambda k: self.swap_scores.get(k, 0.0), reverse=True)
            open_swaps = set(ranked[: int(self.max_swaps)])
        return self.builder.build(hubs, swaps=open_swaps)

    def _neighbors(self, current: FacilityCandidate) -> list[tuple[FacilityCandidate, tuple]]:
        hubs = set(current.hubs)
        swaps = set(current.swaps)
        neighbors: list[tuple[FacilityCandidate, tuple]] = []

        for h in self._ranked_hubs(current):
            if h in hubs:
                neighbors.append((self._candidate(hubs - {h}, swaps), ("drop_h", h)))
            else:
                neighbors.append((self._candidate(hubs | {h}, swaps), ("add_h", h)))

        ranked_hubs = self._ranked_hubs(current)
        open_ranked = [h for h in ranked_hubs if h in hubs]
        closed_ranked = [h for h in ranked_hubs if h not in hubs]
        for h_out in open_ranked[:5]:
            for h_in in closed_ranked[:5]:
                neighbors.append(
                    (
                        self._candidate((hubs - {h_out}) | {h_in}, swaps),
                        ("swap_h", h_out, h_in),
                    )
                )

        for k in self._ranked_swaps(current):
            if k in swaps:
                neighbors.append((self._candidate(hubs, swaps - {k}), ("drop_e", k)))
            else:
                neighbors.append((self._candidate(hubs, swaps | {k}), ("add_e", k)))

        ranked_swaps = self._ranked_swaps(current)
        open_swaps = [k for k in ranked_swaps if k in swaps]
        closed_swaps = [k for k in ranked_swaps if k not in swaps]
        for k_out in open_swaps[:4]:
            for k_in in closed_swaps[:4]:
                neighbors.append(
                    (
                        self._candidate(hubs, (swaps - {k_out}) | {k_in}),
                        ("swap_e", k_out, k_in),
                    )
                )

        unique: list[tuple[FacilityCandidate, tuple]] = []
        seen = {current}
        for candidate, move in neighbors:
            if candidate in seen:
                continue
            seen.add(candidate)
            unique.append((candidate, move))
            if len(unique) >= self.max_neighbors:
                break
        return unique

    def solve(self, seeds: Sequence[FacilityCandidate]) -> HeuristicRun:
        start = time.perf_counter()
        eval_start = self.evaluator.evaluations
        cache_start = self.evaluator.cache_hits

        seed_list = list(dict.fromkeys(seeds)) or [FacilityCandidate(frozenset(), frozenset(), frozenset())]
        best: Optional[LocalSearchSolution] = None
        current: Optional[LocalSearchSolution] = None
        for candidate in seed_list:
            sol = self.evaluator.evaluate(candidate)
            if _solution_value(sol) > _solution_value(best):
                best = sol
                current = sol
        assert current is not None
        assert best is not None

        tabu_until: Dict[tuple, int] = {}
        for it in range(1, self.max_iter + 1):
            chosen: Optional[LocalSearchSolution] = None
            chosen_move: Optional[tuple] = None
            for candidate, move in self._neighbors(current.candidate):
                sol = self.evaluator.evaluate(candidate)
                is_tabu = tabu_until.get(move, 0) >= it
                aspiration = _solution_value(sol) > _solution_value(best)
                if is_tabu and not aspiration:
                    continue
                if _solution_value(sol) > _solution_value(chosen):
                    chosen = sol
                    chosen_move = move

            if chosen is None:
                break
            current = chosen
            if chosen_move is not None:
                tabu_until[chosen_move] = it + self.tenure
            if _solution_value(current) > _solution_value(best):
                best = current

            self.history.append(
                {
                    "iteration": it,
                    "current_obj": current.obj,
                    "best_obj": best.obj,
                    "move": chosen_move,
                    "open_hubs": len(current.candidate.hubs),
                    "open_swaps": len(current.candidate.swaps),
                }
            )
            if self.verbose:
                print(
                    f"[LR-Tabu] it={it} current={current.obj} best={best.obj} "
                    f"hubs={len(current.candidate.hubs)}"
                )

        wall = time.perf_counter() - start
        best.wall_time = wall
        best.evaluations = self.evaluator.evaluations - eval_start
        best.cache_hits = self.evaluator.cache_hits - cache_start
        return HeuristicRun(
            name="lr_tabu",
            solution=best,
            wall_time=wall,
            evaluations=best.evaluations,
            cache_hits=best.cache_hits,
            history=self.history,
        )


def run_lagrangian_suite(
    instance: Instance,
    *,
    eval_time_limit: float = 5.0,
    eval_mip_gap: float = 0.05,
    lr_iterations: int = 40,
    lr_eval_top: int = 30,
    tabu_iterations: int = 20,
    tabu_tenure: int = 5,
    tabu_candidate_hubs: int = 15,
    tabu_candidate_swaps: int = 10,
    tabu_max_neighbors: int = 80,
    max_hubs: Optional[int] = None,
    max_swaps: Optional[int] = None,
    verbose: bool = False,
) -> list[HeuristicRun]:
    evaluator = RegimeIIFixedEvaluator(
        instance,
        eval_time_limit=eval_time_limit,
        eval_mip_gap=eval_mip_gap,
    )
    runs: list[HeuristicRun] = []

    coverage = RegimeIILagrangianHeuristic(
        instance,
        decomposition="coverage",
        evaluator=evaluator,
        max_iter=lr_iterations,
        max_hubs=max_hubs,
        max_swaps=max_swaps,
        eval_top=lr_eval_top,
        verbose=verbose,
    )
    runs.append(coverage.solve())

    route = RegimeIILagrangianHeuristic(
        instance,
        decomposition="route_swap",
        evaluator=evaluator,
        max_iter=lr_iterations,
        max_hubs=max_hubs,
        max_swaps=max_swaps,
        eval_top=lr_eval_top,
        verbose=verbose,
    )
    runs.append(route.solve())

    seeds = [run.solution.candidate for run in runs]
    tabu = RegimeIILagrangianTabuSearch(
        instance,
        evaluator=evaluator,
        hub_scores=route.last_hub_scores or coverage.last_hub_scores,
        swap_scores=route.last_swap_scores or coverage.last_swap_scores,
        max_iter=tabu_iterations,
        tenure=tabu_tenure,
        candidate_hubs=tabu_candidate_hubs,
        candidate_swaps=tabu_candidate_swaps,
        max_neighbors=tabu_max_neighbors,
        max_swaps=max_swaps,
        verbose=verbose,
    )
    runs.append(tabu.solve(seeds))
    return runs


STANDARD_HEURISTIC_METHODS = (
    "constructive",
    "lagrangian_relaxation",
    "tabu_lagrangian_relaxation",
)


def run_constructive_heuristic(
    instance: Instance,
    *,
    evaluator: Optional[RegimeIIFixedEvaluator] = None,
    eval_time_limit: float = 5.0,
    eval_mip_gap: float = 0.05,
    constructive_candidate_hubs: int = 15,
    constructive_candidate_swaps: int = 10,
    constructive_eval_top: int = 25,
    max_hubs: Optional[int] = None,
    max_swaps: Optional[int] = None,
    verbose: bool = False,
) -> HeuristicRun:
    evaluator = evaluator or RegimeIIFixedEvaluator(
        instance,
        eval_time_limit=eval_time_limit,
        eval_mip_gap=eval_mip_gap,
    )
    return RegimeIIConstructiveHeuristic(
        instance,
        evaluator=evaluator,
        max_hubs=max_hubs,
        max_swaps=max_swaps,
        candidate_hubs=constructive_candidate_hubs,
        candidate_swaps=constructive_candidate_swaps,
        eval_top=constructive_eval_top,
        verbose=verbose,
    ).solve()


def run_lagrangian_relaxation(
    instance: Instance,
    *,
    evaluator: Optional[RegimeIIFixedEvaluator] = None,
    eval_time_limit: float = 5.0,
    eval_mip_gap: float = 0.05,
    lr_iterations: int = 40,
    lr_eval_top: int = 30,
    lr_decomposition: str = "route_swap",
    max_hubs: Optional[int] = None,
    max_swaps: Optional[int] = None,
    verbose: bool = False,
) -> HeuristicRun:
    evaluator = evaluator or RegimeIIFixedEvaluator(
        instance,
        eval_time_limit=eval_time_limit,
        eval_mip_gap=eval_mip_gap,
    )
    run = RegimeIILagrangianHeuristic(
        instance,
        decomposition=lr_decomposition,
        evaluator=evaluator,
        max_iter=lr_iterations,
        max_hubs=max_hubs,
        max_swaps=max_swaps,
        eval_top=lr_eval_top,
        verbose=verbose,
    ).solve()
    run.name = "lagrangian_relaxation"
    return run


def run_tabu_lagrangian_relaxation(
    instance: Instance,
    *,
    evaluator: Optional[RegimeIIFixedEvaluator] = None,
    seed_runs: Optional[Sequence[HeuristicRun]] = None,
    lr_seed_run: Optional[HeuristicRun] = None,
    eval_time_limit: float = 5.0,
    eval_mip_gap: float = 0.05,
    lr_iterations: int = 40,
    lr_eval_top: int = 30,
    lr_decomposition: str = "route_swap",
    tabu_iterations: int = 20,
    tabu_tenure: int = 5,
    tabu_candidate_hubs: int = 15,
    tabu_candidate_swaps: int = 10,
    tabu_max_neighbors: int = 80,
    max_hubs: Optional[int] = None,
    max_swaps: Optional[int] = None,
    verbose: bool = False,
) -> HeuristicRun:
    evaluator = evaluator or RegimeIIFixedEvaluator(
        instance,
        eval_time_limit=eval_time_limit,
        eval_mip_gap=eval_mip_gap,
    )

    lr_solver = RegimeIILagrangianHeuristic(
        instance,
        decomposition=lr_decomposition,
        evaluator=evaluator,
        max_iter=lr_iterations,
        max_hubs=max_hubs,
        max_swaps=max_swaps,
        eval_top=lr_eval_top,
        verbose=verbose,
    )
    if lr_seed_run is None:
        lr_seed_run = lr_solver.solve()
    else:
        # Still run the LR iterations when only a solution seed is supplied,
        # because tabu needs the final multiplier-derived scores.
        lr_solver.solve()

    seeds = [lr_seed_run.solution.candidate]
    if seed_runs is not None:
        seeds.extend(run.solution.candidate for run in seed_runs)

    run = RegimeIILagrangianTabuSearch(
        instance,
        evaluator=evaluator,
        hub_scores=lr_solver.last_hub_scores,
        swap_scores=lr_solver.last_swap_scores,
        max_iter=tabu_iterations,
        tenure=tabu_tenure,
        candidate_hubs=tabu_candidate_hubs,
        candidate_swaps=tabu_candidate_swaps,
        max_neighbors=tabu_max_neighbors,
        max_swaps=max_swaps,
        verbose=verbose,
    ).solve(seeds)
    run.name = "tabu_lagrangian_relaxation"
    return run


def run_standard_heuristics(
    instance: Instance,
    *,
    methods: Sequence[str] = STANDARD_HEURISTIC_METHODS,
    eval_time_limit: float = 5.0,
    eval_mip_gap: float = 0.05,
    constructive_candidate_hubs: int = 15,
    constructive_candidate_swaps: int = 10,
    constructive_eval_top: int = 25,
    lr_iterations: int = 40,
    lr_eval_top: int = 30,
    lr_decomposition: str = "route_swap",
    tabu_iterations: int = 20,
    tabu_tenure: int = 5,
    tabu_candidate_hubs: int = 15,
    tabu_candidate_swaps: int = 10,
    tabu_max_neighbors: int = 80,
    max_hubs: Optional[int] = None,
    max_swaps: Optional[int] = None,
    verbose: bool = False,
) -> list[HeuristicRun]:
    unknown = [method for method in methods if method not in STANDARD_HEURISTIC_METHODS]
    if unknown:
        raise ValueError(f"Unknown heuristic methods: {', '.join(unknown)}")

    evaluator = RegimeIIFixedEvaluator(
        instance,
        eval_time_limit=eval_time_limit,
        eval_mip_gap=eval_mip_gap,
    )
    requested = list(dict.fromkeys(methods))
    runs: list[HeuristicRun] = []
    by_name: Dict[str, HeuristicRun] = {}

    if "constructive" in requested:
        run = run_constructive_heuristic(
            instance,
            evaluator=evaluator,
            constructive_candidate_hubs=constructive_candidate_hubs,
            constructive_candidate_swaps=constructive_candidate_swaps,
            constructive_eval_top=constructive_eval_top,
            max_hubs=max_hubs,
            max_swaps=max_swaps,
            verbose=verbose,
        )
        by_name[run.name] = run
        runs.append(run)

    lr_run: Optional[HeuristicRun] = None
    lr_solver: Optional[RegimeIILagrangianHeuristic] = None
    if "lagrangian_relaxation" in requested or "tabu_lagrangian_relaxation" in requested:
        lr_solver = RegimeIILagrangianHeuristic(
            instance,
            evaluator=evaluator,
            decomposition=lr_decomposition,
            max_iter=lr_iterations,
            max_hubs=max_hubs,
            max_swaps=max_swaps,
            eval_top=lr_eval_top,
            verbose=verbose,
        )
        lr_run = lr_solver.solve()
        lr_run.name = "lagrangian_relaxation"
        by_name[lr_run.name] = lr_run
        if "lagrangian_relaxation" in requested:
            runs.append(lr_run)

    if "tabu_lagrangian_relaxation" in requested:
        seed_runs = [run for run in by_name.values()]
        assert lr_solver is not None
        tabu_run = RegimeIILagrangianTabuSearch(
            instance,
            evaluator=evaluator,
            hub_scores=lr_solver.last_hub_scores,
            swap_scores=lr_solver.last_swap_scores,
            max_iter=tabu_iterations,
            tenure=tabu_tenure,
            candidate_hubs=tabu_candidate_hubs,
            candidate_swaps=tabu_candidate_swaps,
            max_neighbors=tabu_max_neighbors,
            max_swaps=max_swaps,
            verbose=verbose,
        ).solve([run.solution.candidate for run in seed_runs])
        tabu_run.name = "tabu_lagrangian_relaxation"
        runs.append(tabu_run)

    order = {method: idx for idx, method in enumerate(requested)}
    return sorted(runs, key=lambda run: order.get(run.name, len(order)))
