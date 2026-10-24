"""Poisson mean thresholds for finite locker compartments."""

from __future__ import annotations

from functools import lru_cache
import math

import numpy as np
from scipy.optimize import brentq
from scipy.stats import poisson


@lru_cache(maxsize=4096)
def poisson_mean_threshold(compartments: int, epsilon: float) -> float:
    """Largest mean satisfying P(Poisson(mean) > compartments) <= epsilon."""
    if (
        isinstance(compartments, bool)
        or not math.isfinite(float(compartments))
        or int(compartments) != compartments
        or compartments < 0
    ):
        raise ValueError("Compartment capacity must be a nonnegative integer")
    if not math.isfinite(epsilon) or not 0 < epsilon < 1:
        raise ValueError("Overflow probability must be strictly between zero and one")
    compartments = int(compartments)
    upper = max(1.0, float(compartments + 1))
    while poisson.sf(compartments, upper) < epsilon:
        upper *= 2.0
    return float(
        brentq(
            lambda mean: poisson.sf(compartments, mean) - epsilon,
            0.0,
            upper,
            xtol=1e-13,
            rtol=4 * 2.220446049250313e-16,
        )
    )


slot_threshold = poisson_mean_threshold


def occupancy_limits(compartments, epsilon):
    """Return mean limits with NumPy broadcasting over capacities and risks.

    For an H-by-S capacity table and a separate risk for each locker, pass
    epsilon[:, None]; a scalar risk broadcasts to every capacity level.
    """
    try:
        capacities, risks = np.broadcast_arrays(
            np.asarray(compartments), np.asarray(epsilon, dtype=float)
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Compartment counts and overflow risks must broadcast"
        ) from exc
    result = np.empty(capacities.shape, dtype=float)
    for index in np.ndindex(result.shape):
        result[index] = poisson_mean_threshold(
            capacities[index].item(), float(risks[index])
        )
    return result
