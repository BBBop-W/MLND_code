"""Pure-algebra certificates and later small integer-master checks."""
from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

from mlnd_path.integer_master_cuts import (
    idle_relaxation_value,
    initial_pooled_idle_cuts,
    pooled_idle_slopes,
    pooled_idle_value,
)


def test_pooled_idle_closed_form_matches_stationary_point_and_cap():
    for weight, cost, cap in itertools.product((0.0, 0.2, 1.0, 3.0, 25.0),
                                                (0.5, 2.0), (0.0, 0.4, 0.9)):
        rho = 0.0 if weight <= cost else min(cap, 1.0-math.sqrt(cost/weight))
        reference = cost*rho/(1.0-rho)-weight*rho
        assert idle_relaxation_value(weight, cost, cap) == pytest.approx(reference, abs=1e-12)


def test_binary_only_idle_cuts_are_valid_for_every_binary_subset():
    # Eight assignment items include two source bases and competing destinations.
    # Checking every bit pattern is stronger than checking only admissible OD
    # assignments, since submodularity itself needs no assignment restriction.
    weights = np.asarray([[[0.2, 1.3], [2.0, 0.0]], [[4.1, 0.7], [0.4, 3.2]]])
    points = [np.asarray(bits, dtype=float).reshape(weights.shape)
              for bits in itertools.product((0.0, 1.0), repeat=weights.size)]
    cuts = initial_pooled_idle_cuts(weights, fleet_cost=2.0, rho_max=0.8)
    cuts += [pooled_idle_slopes(weights, point, fleet_cost=2.0, rho_max=0.8)
             for point in points]
    truths = np.asarray([pooled_idle_value(weights, point, fleet_cost=2.0, rho_max=0.8)
                         for point in points])
    slopes = np.asarray([cut.ravel() for cut in cuts])
    vectors = np.asarray([point.ravel() for point in points])
    assert np.max(slopes@vectors.T-truths[None, :]) <= 1e-11
    for point, cut, truth in zip(points, cuts[3:], truths):
        assert float(np.sum(cut*point)) == pytest.approx(truth, abs=2e-11)


def test_lovasz_chain_at_fractional_point_equals_convex_combination_of_binary_values():
    weights = np.asarray([[[0.2]], [[1.7]], [[3.0]], [[0.8]]])
    point = np.asarray([[[0.8]], [[0.35]], [[0.6]], [[0.15]]])
    slope = pooled_idle_slopes(weights, point, fleet_cost=1.0, rho_max=0.85)
    ordering = np.argsort(-point.ravel(), kind="stable")
    ordered_point = point.ravel()[ordering]
    selected = np.zeros_like(point)
    expected = 0.0
    for k, item in enumerate(ordering):
        selected.ravel()[item] = 1.0
        next_level = ordered_point[k+1] if k+1 < len(ordering) else 0.0
        expected += (ordered_point[k]-next_level)*pooled_idle_value(
            weights, selected, fleet_cost=1.0, rho_max=0.85)
    assert float(np.sum(slope*point)) == pytest.approx(expected, abs=1e-12)


def test_pooled_relaxation_is_below_every_feasible_assigned_service_cost():
    # Directly retain the shared idle cost once per base and check its global
    # minimum, independently of the combinatorial chain construction.
    weights = np.asarray([[[0.2, 1.3], [2.0, 0.0]], [[4.1, 0.7], [0.4, 3.2]]])
    point = np.zeros_like(weights)
    point[0, 0, 1] = 1.0
    point[1, 0, 0] = 1.0
    lower = pooled_idle_value(weights, point, fleet_cost=2.0, rho_max=0.8)
    for rho0, rho1 in itertools.product(np.linspace(0.0, 0.8, 11), repeat=2):
        actual = 2.0*(rho0/(1-rho0)+rho1/(1-rho1))-5.4*rho0
        assert lower <= actual+1e-12


def test_pooled_idle_rejects_invalid_benefits_and_preserves_zero_weight():
    zeros = np.zeros((2, 2, 2))
    assert np.array_equal(pooled_idle_slopes(zeros, fleet_cost=1.0, rho_max=0.9), zeros)
    with pytest.raises(ValueError):
        pooled_idle_slopes(-np.ones((1, 1, 1)), fleet_cost=1.0, rho_max=0.9)
