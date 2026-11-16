"""Direct-route acceleration must preserve the complete scalar catalogue."""

from dataclasses import replace

import numpy as np
import pytest

from src.data.instances import to_planning_data
from src.model.paths import EnumerationStopped, enumerate_paths
from src.model.types import PlanningData


def reference_catalogue(data, **kwargs):
    # Supplied one-arc routes invoke the original scalar enumeration path; the
    # same-origin transfer is included by RouteCatalogue in both domains.
    routes = tuple(
        (u, v)
        for u in range(data.n_nodes)
        for v in range(data.n_nodes)
        if data.allowed_links[u, v]
    )
    return enumerate_paths(replace(data, candidate_routes=routes), **kwargs)


def assert_same_catalogue(data):
    fast, reference = enumerate_paths(data), reference_catalogue(data)
    assert fast.initial_segments == reference.initial_segments
    assert fast.service_paths == reference.service_paths
    assert fast.empty_paths == reference.empty_paths
    assert fast.complete and reference.complete
    return fast


@pytest.mark.parametrize("seed", range(10))
def test_direct_prefilter_equals_full_scalar_catalogue(seed):
    rng = np.random.default_rng(seed)
    n_l, n_h, n_od = 2, 3, 4
    n = n_l + n_h
    flight = rng.uniform(0, 0.16, (n, n))
    np.fill_diagonal(flight, 0)
    allowed = rng.random((n, n)) > 0.15
    np.fill_diagonal(allowed, False)
    if seed % 2:
        flight[~allowed] = np.inf
        np.fill_diagonal(flight, 0)
    rider = rng.uniform(0, 0.2, (n_od, n_h))
    rider[0, 0] = 0
    receipt = rng.uniform(0, 0.1, (n_od, n_h))
    kappa = rng.random((n_od, n_od))
    kappa[kappa < 0.25] = 0
    data = PlanningData(
        n_l=n_l,
        n_h=n_h,
        demand=rng.uniform(1, 10, n_od),
        flight_times=flight,
        allowed_links=allowed,
        rider_mean=rider,
        first_cost=rng.random((n_od, n_h)),
        terminal_cost=rng.random((n_od, n_h)),
        receipt_mean=receipt,
        removal_mean=receipt / 2,
        ground_cost=5,
        deadline=rng.uniform(0.15, 0.4, n_od),
        base_cost=0.5,
        slot_counts=[10, 20],
        slot_cost=[0.1, 0.2],
        slot_epsilon=0.05,
        link_cost=0,
        kappa=kappa,
        delta=0.01 if seed < 8 else 0,
        operating_limit=float(rng.uniform(0.15, 0.6)),
        fleet_cost=1,
        flight_cost=0.2,
        waiting_cost=0.3,
    )
    assert_same_catalogue(data)


@pytest.mark.parametrize("shift", [-2e-12, -1e-12, -1e-15, 0, 1e-15, 1e-12, 2e-12])
def test_direct_prefilter_preserves_boundary_paths_and_ids(small_input, shift):
    data = to_planning_data(small_input)
    original = enumerate_paths(data)
    path = next(p for p in original.service_paths if p.second_od is not None)
    assert_same_catalogue(replace(data, operating_limit=path.time + shift))
    deadlines = data.deadline.copy()
    deadlines[path.second_od] = path.delivery_times[1] + shift
    assert_same_catalogue(replace(data, deadline=deadlines))


def test_direct_prefilter_stop_and_column_limits_match(small_input):
    data = to_planning_data(small_input)
    total = len(enumerate_paths(data).service_paths)
    for limit in (0, 1, total - 1):
        with pytest.raises(EnumerationStopped) as fast:
            enumerate_paths(data, column_limit=limit)
        with pytest.raises(EnumerationStopped) as reference:
            reference_catalogue(data, column_limit=limit)
        assert fast.value.reason == reference.value.reason == "column_limit"
        assert fast.value.columns == reference.value.columns
        assert fast.value.initial_segments == reference.value.initial_segments
    for kwargs, reason in (
        ({"time_limit": 0}, "time_limit"),
        ({"stop_requested": lambda: True}, "user_stop"),
    ):
        with pytest.raises(EnumerationStopped) as error:
            enumerate_paths(data, **kwargs)
        assert error.value.reason == reason
        assert error.value.columns == error.value.initial_segments == 0
