"""Input-derived fixed task rates and complete-domain preprocessing checks."""
from dataclasses import dataclass
import math
import numpy as np
from .paths import iter_paths

def physical_task_statistics(data, *, time_limit=None, stop_requested=None):
    """Find task cost/support over the input's complete physical domain.

    Output witnesses can prove that a path exists, but cannot prove a global
    minimum or absence of other supported departures. Consume the same physical
    iterator as enumeration without retaining its completed paths. An interrupted
    scan raises; it never returns partial statistics as reconstruction evidence.
    """
    r, bases = data.n_od, data.n_l
    single_cost, double_cost = np.full(r, np.inf), np.full((r, r), np.inf)
    single_support = np.zeros((r, bases), dtype=bool)
    double_support = np.zeros((r, r, bases), dtype=bool)
    service_paths = 0
    for family, path in iter_paths(data, time_limit=time_limit, stop_requested=stop_requested):
        if family != "service":
            continue
        service_paths += 1
        first, second, origin = path.first_od, path.second_od, path.b[0]
        if second is None:
            single_cost[first] = min(single_cost[first], path.cost)
            single_support[first, origin] = True
        else:
            double_cost[first, second] = min(double_cost[first, second], path.cost)
            double_support[first, second, origin] = True
    return single_cost, double_cost, single_support, double_support, service_paths


@dataclass(frozen=True)
class GeneralizedDemand:
    single_rates: np.ndarray
    pair_rates: np.ndarray
    common_origin_support: np.ndarray
    candidate_order: tuple
    processing_trace: tuple

    @property
    def head_rates(self):
        return self.single_rates + self.pair_rates.sum(axis=1)

    @property
    def reconstructed_orders(self):
        return self.head_rates + self.pair_rates.sum(axis=0)


def generalized_demand(
    demand,
    delta,
    kappa,
    single_cost,
    double_cost,
    single_support,
    double_support,
    *,
    candidate_order=None,
):
    """Reconstruct fixed rates from complete-path costs and departure support.

    Candidate pairs have binary compatibility and positive proxy savings,
    defined by the two cheapest separate service paths less the cheapest
    ordered double path.  The deterministic traversal uses decreasing savings,
    then OD indices.  All positive branches sharing a first OD retain at least
    one common supported departure base.  Scores exclude facility costs and
    are an exogenous ranking heuristic, not an optimal matching certificate.
    """
    h = np.asarray(demand, dtype=float)
    if h.ndim != 1 or not h.size or not np.isfinite(h).all() or np.any(h < 0):
        raise ValueError("demand must be a finite nonnegative vector")
    if not math.isfinite(float(delta)) or delta < 0:
        raise ValueError("delta must be finite and nonnegative")
    n = len(h)
    compatibility = np.asarray(kappa, dtype=float)
    c1, c2 = np.asarray(single_cost, float), np.asarray(double_cost, float)
    support1 = np.asarray(single_support, bool)
    support2 = np.asarray(double_support, bool)
    if compatibility.shape != (n, n) or not np.isfinite(compatibility).all():
        raise ValueError("Invalid compatibility matrix")
    if delta > 0 and np.any((compatibility != 0) & (compatibility != 1)):
        raise ValueError("preprocessed demand requires binary kappa")
    if c1.shape != (n,) or c2.shape != (n, n):
        raise ValueError("Invalid path-cost dimensions")
    if np.isnan(c1).any() or np.isnan(c2).any() or np.any(c1 < 0) or np.any(c2 < 0):
        raise ValueError("Path costs must be nonnegative or positive infinity")
    if (
        support1.ndim != 2
        or support1.shape[0] != n
        or support2.shape != (n, n, support1.shape[1])
    ):
        raise ValueError("Invalid departure-support dimensions")
    if not np.array_equal(np.isfinite(c1), support1.any(axis=1)) or not np.array_equal(
        np.isfinite(c2), support2.any(axis=2)
    ):
        raise ValueError("Finite path costs and departure support disagree")
    candidates = []
    scores, score_tolerances = {}, {}
    for r, s in np.ndindex(n, n):
        if compatibility[r, s] != 1 or not all(
            math.isfinite(c) for c in (c1[r], c1[s], c2[r, s])
        ):
            continue
        score = float(c1[r] + c1[s] - c2[r, s])
        scores[r, s] = score
        score_tolerances[r, s] = 1e-12 * (1 + c1[r] + c1[s] + c2[r, s])
        if score > 0:
            candidates.append((r, s, score))
    candidates.sort(key=lambda item: (-item[2], item[0], item[1]))
    if candidate_order is not None:
        order = tuple(tuple(row) for row in candidate_order)
        # Physical costs are independently recomputed in Python. A saving of
        # zero can have either sign after C++/Python floating-point summation.
        # Preserve the saved decision only at this numerical boundary; require
        # every clearly positive pair and reject incompatible/negative pairs.
        required = {pair for pair, score in scores.items()
                    if score > score_tolerances[pair]}
        allowed = {pair for pair, score in scores.items()
                   if score >= -score_tolerances[pair]}
        if (len(order) != len(set(order)) or not required <= set(order)
                or not set(order) <= allowed):
            raise ValueError(
                "Saved candidate order does not cover the eligible pairs exactly"
            )
        candidates = [(r, s, scores[r, s]) for r, s in order]
        if any(
            following[2] - current[2] > 1e-10 * (1 + abs(current[2]))
            for current, following in zip(candidates, candidates[1:])
        ):
            raise ValueError("Saved candidate order violates decreasing proxy savings")
    residual = h.copy()
    pair = np.zeros((n, n))
    common = support1.copy()
    trace = []
    for r, s, score in candidates:
        compatible_origins = common[r] & support2[r, s]
        q = -math.expm1(-float(residual[s]) * float(delta))
        amount = (residual[r] / 2 if r == s else min(residual[r], residual[s])) * q
        status = "accepted"
        if amount <= 0:
            status = "zero_residual"
        elif not compatible_origins.any():
            status = "no_common_origin"
            amount = 0.0
        else:
            common[r] = compatible_origins
            pair[r, s] = amount
            residual[r] -= amount
            residual[s] -= amount
        trace.append(
            dict(r=r, s=s, score=score, q=q, gamma=float(amount), status=status)
        )
    return GeneralizedDemand(
        residual, pair, common, tuple((r, s) for r, s, _ in candidates), tuple(trace)
    )

