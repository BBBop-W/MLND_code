"""Order-conserving mean task reconstruction with spatial compatibility."""

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class DemandRates:
    pool_rate: float
    q: float
    theta: np.ndarray
    single_rates: np.ndarray
    pair_rates: np.ndarray
    task_rate: float
    od_share: np.ndarray

    @property
    def reconstructed_orders(self):
        return (
            self.single_rates
            + self.pair_rates.sum(axis=0)
            + self.pair_rates.sum(axis=1)
        )


def demand_rates(demand, assignment=None, kappa=None, delta=0.0, *, max_orders=2):
    """Solve (I + q diag(pi) kappa.T) theta = assigned demand.

    This reconstructs offered task rates before applying base availability.
    Repeated OD pairs count two distinct occurrences of the same OD.
    """
    h = np.asarray(demand, dtype=float)
    if h.ndim != 1 or not h.size or not np.isfinite(h).all() or np.any(h < 0):
        raise ValueError("demand must be a finite nonnegative vector")
    alpha = (
        np.ones_like(h) if assignment is None else np.asarray(assignment, dtype=float)
    )
    if alpha.shape != h.shape or np.any((alpha != 0) & (alpha != 1)):
        raise ValueError("assignment must be a binary OD vector")
    compatibility = (
        np.ones((h.size, h.size)) if kappa is None else np.asarray(kappa, dtype=float)
    )
    if (
        compatibility.shape != (h.size, h.size)
        or not np.isfinite(compatibility).all()
        or np.any((compatibility < 0) | (compatibility > 1))
    ):
        raise ValueError("kappa must be an OD-by-OD matrix in [0, 1]")
    if isinstance(max_orders, (bool, np.bool_)) or max_orders not in (1, 2):
        raise ValueError("max_orders must be 1 or 2")
    delta = float(delta)
    if not math.isfinite(delta) or delta < 0:
        raise ValueError("delta must be finite and nonnegative")
    delta = delta if max_orders == 2 else 0.0
    assigned = h * alpha
    total = float(assigned.sum())
    if not math.isfinite(total):
        raise ValueError("Total assigned demand must be finite")
    q = -math.expm1(-total * delta)
    share = assigned / total if total else np.zeros_like(h)
    conditional = q * compatibility * share[None, :]
    theta = np.linalg.solve(np.eye(h.size) + conditional.T, assigned)
    if np.any(theta < -1e-10 * max(1.0, total)):
        raise ArithmeticError("Task reconstruction produced negative initiation rates")
    theta = np.maximum(theta, 0.0)
    pair = theta[:, None] * conditional
    # Computing the probability as a weighted sum avoids cancellation when q≈1.
    no_second = math.exp(-total * delta) + q * ((1 - compatibility) @ share)
    single = theta * no_second
    for values in (theta, pair, single, share):
        values.setflags(write=False)
    return DemandRates(total, q, theta, single, pair, float(theta.sum()), share)
