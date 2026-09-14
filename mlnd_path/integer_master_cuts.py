"""Binary-only pooled-idle cuts; no operating variables enter the master.

For nonnegative assignment benefits w_g and a fixed source base,
  F(W) = min_{0<=rho<=rho_max} cFleet*rho/(1-rho) - W*rho
is concave and nonincreasing, with F(0)=0. Thus F(sum_{g in S}w_g) is a
submodular set function. Every greedy chain produces a valid modular lower
bound, giving the pure-binary cut zeta >= C0 + sum_g slope_g * B_g.
At a binary assignment sorted with selected items first, the unrounded chain
cut equals the pooled-idle relaxation at that assignment. The small downward
coefficient reserve below makes floating-point construction conservative.
"""
from __future__ import annotations

import math
import sys

import numpy as np


def _parameters(fleet_cost: float, rho_max: float) -> tuple[float, float]:
    cost, maximum = float(fleet_cost), float(rho_max)
    if not math.isfinite(cost) or cost <= 0:
        raise ValueError("fleet_cost must be finite and strictly positive")
    if not math.isfinite(maximum) or not 0 <= maximum < 1:
        raise ValueError("rho_max must belong to [0,1)")
    return cost, maximum


def idle_relaxation_value(weight: float, fleet_cost: float, rho_max: float) -> float:
    """Closed-form F(weight), including zero weight and an active rho cap."""
    cost, maximum = _parameters(fleet_cost, rho_max)
    weight = float(weight)
    if not math.isfinite(weight) or weight < 0:
        raise ValueError("weight must be finite and nonnegative")
    if maximum == 0 or weight <= cost:
        return 0.0
    if weight >= cost/(1.0-maximum)**2:
        return cost*maximum/(1.0-maximum)-weight*maximum
    # Equivalent to -(sqrt(weight)-sqrt(cost))**2, with a stable difference
    # when weight is close to the idle-cost threshold.
    difference = (weight-cost)/(math.sqrt(weight)+math.sqrt(cost))
    return -difference*difference


def _weights(weights) -> np.ndarray:
    answer = np.asarray(weights, dtype=float)
    if answer.ndim != 3 or answer.shape[1] != answer.shape[2] or answer.shape[1] == 0:
        raise ValueError("weights must have shape (nOD,nL,nL), with nL positive")
    if not np.all(np.isfinite(answer)) or np.any(answer < 0):
        raise ValueError("assignment benefit weights must be finite and nonnegative")
    return answer


def pooled_idle_value(weights, assignment, *, fleet_cost: float, rho_max: float) -> float:
    """Evaluate sum_l F(sum_od,m weights[od,l,m]*assignment[od,l,m]).

    Its use as a physical lower bound is for binary assignment vectors; this
    function also accepts continuous vectors for algebraic diagnostics.
    """
    benefits = _weights(weights)
    point = np.asarray(assignment, dtype=float)
    if point.shape != benefits.shape or not np.all(np.isfinite(point)) or np.any(point < 0):
        raise ValueError("assignment must be finite, nonnegative, and match weights")
    cost, maximum = _parameters(fleet_cost, rho_max)
    return sum(idle_relaxation_value(math.fsum((benefits[:, l, :]*point[:, l, :]).ravel()),
                                     cost, maximum) for l in range(benefits.shape[1]))


def pooled_idle_slopes(weights, assignment=None, *, fleet_cost: float, rho_max: float,
                       order: str = "point") -> np.ndarray:
    """Return an equally shaped globally valid binary-master slope array.

    ``order='point'`` sorts by assignment descending, then benefit descending;
    this is the separating Lovasz chain for a fractional point, and is tight
    at a binary point before rounding reserves. If assignment is omitted, it
    uses descending benefits. Other initial orders are ``descending_weight``,
    ``ascending_weight``, and ``lexicographic``. Any such order is valid.
    """
    benefits = _weights(weights)
    cost, maximum = _parameters(fleet_cost, rho_max)
    if assignment is None:
        point = np.zeros_like(benefits)
        if order == "point":
            order = "descending_weight"
    else:
        point = np.asarray(assignment, dtype=float)
        if point.shape != benefits.shape or not np.all(np.isfinite(point)):
            raise ValueError("assignment must be finite and match the weight array")
    valid_orders = {"point", "descending_weight", "ascending_weight", "lexicographic"}
    if order not in valid_orders:
        raise ValueError(f"Unknown chain order: {order}")
    slopes = np.zeros_like(benefits)
    for l in range(benefits.shape[1]):
        values = benefits[:, l, :].ravel()
        indices = np.arange(len(values))
        if order == "point":
            sequence = np.lexsort((indices, -values, -point[:, l, :].ravel()))
        elif order == "descending_weight":
            sequence = np.lexsort((indices, -values))
        elif order == "ascending_weight":
            sequence = np.lexsort((indices, values))
        else:
            sequence = indices
        output = np.zeros_like(values)
        cumulative = compensation = previous = 0.0
        for item in sequence:
            if values[item] == 0:
                continue
            # Compensated positive summation controls errors in long chains.
            increment = values[item]-compensation
            updated = cumulative+increment
            compensation = (updated-cumulative)-increment
            cumulative = updated
            if not math.isfinite(cumulative):
                raise ValueError("cumulative assignment benefit overflows float64")
            current = idle_relaxation_value(cumulative, cost, maximum)
            reserve = 16.0*sys.float_info.epsilon*(1.0+cumulative+cost+abs(current)+abs(previous))
            output[item] = min(0.0, current-previous-reserve)
            previous = current
        slopes[:, l, :] = output.reshape(benefits.shape[0], benefits.shape[2])
    return slopes


def initial_pooled_idle_cuts(weights, *, fleet_cost: float, rho_max: float,
                             orders=("descending_weight", "ascending_weight", "lexicographic")):
    """Return deterministic initial chain cuts, without introducing variables."""
    return [pooled_idle_slopes(weights, fleet_cost=fleet_cost, rho_max=rho_max, order=order)
            for order in orders]
