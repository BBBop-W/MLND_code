"""Globally valid affine Benders cuts from complete candidate-graph pricing."""

import numpy as np


def _idle_conjugate(coefficient, cost):
    return (
        0.0 if coefficient >= -cost else -((np.sqrt(-coefficient) - np.sqrt(cost)) ** 2)
    )


def build_cut(data, design, dual, objective):
    """Complete source on/off duals and apply a floating-point safety margin."""
    L, H, K, D = (data[k] for k in ("n_l", "n_h", "n_k", "n_od"))
    last = dual
    net = design.net
    capacity = design.capacity
    selected = design.selected
    T = len(selected)
    os, ls, ms = design.origins, design.sources, design.returns
    f, zlevel, B, y = (
        design.facility,
        design.capacity_level,
        design.assignment,
        design.airlines,
    )
    c0 = design.baseline_cost
    # Extend the selected-group dual to the affine on/off source formulation.
    prices = last["u"]
    ap = last["eta"]
    op = last["priced"]
    low = op["pricing_cost"]
    alpha_cost = np.minimum(last["a"], low[os, ls, ms]) if T else np.empty(0)
    piv = last["pi"].copy()
    sig = last["sigma"]
    elow = op["empty_pricing_cost"]
    # Empty cost is nonnegative. Scaling the potential toward zero restores
    # all empty-column inequalities, even after an interrupted CG solve.
    scale = 1.0
    for l in range(L):
        for m in range(L):
            difference = piv[m] - piv[l]
            if l != m and difference > 0 and np.isfinite(elow[l, m]):
                scale = min(scale, max(0.0, float(elow[l, m])) / difference)
    piv *= scale
    alpha = np.zeros((D, L))
    for g, (o, l, m) in enumerate(selected):
        alpha[o, l] = alpha_cost[g] + piv[l] - piv[m]
    coefficient = data["demand"] @ alpha + sig
    M = data["demand"]
    beta = np.maximum(alpha, 0.0)
    gamma = np.maximum(
        0.0, alpha[:, :, None] - piv[None, :, None] + piv[None, None, :] - low
    )
    cutB = M[:, None, None] * (beta[:, :, None] - gamma)
    cutf = np.zeros(L + H + K)
    cutf[:L] = -sig
    cutz = -capacity * prices[:, None]
    cuty = -net["upper"] * ap
    intercept = float(
        c0
        + sum(_idle_conjugate(x, data["fleet_cost"]) for x in coefficient)
        - M @ beta.sum(axis=1)
    )
    norm = (
        1
        + abs(intercept)
        + np.abs(cutB).sum()
        + np.abs(cutf).sum()
        + np.abs(cutz).sum()
        + np.abs(cuty).sum()
    )
    margin = 64 * np.finfo(float).eps * norm + 1e-8 * (1 + abs(c0))
    intercept -= margin
    value = float(
        intercept + cutf @ f + np.sum(cutz * zlevel) + cuty @ y + np.sum(cutB * B)
    )
    gap = objective - value
    return dict(
        bound=value,
        cut_intercept=intercept,
        cut_B=cutB,
        cut_facility=cutf,
        cut_level=cutz,
        cut_airline=cuty,
        cut_valid=bool(np.isfinite(value)),
        duality_gap=gap,
        capacity_prices=prices,
        airline_prices=ap,
        cut_correction=margin,
        dual_potential_scale=scale,
    )
