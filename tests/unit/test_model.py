"""Independent mathematical checks of the consecutive physical model."""

from dataclasses import replace
import math

import numpy as np
import pytest
from scipy.integrate import quad

from src.model.demand import demand_rates
from src.model.paths import EnumerationStopped, enumerate_paths
from src.model.resources import (
    PathInfeasible,
    build_empty_path,
    build_initial_segment,
    build_service_path,
    rebuild_path,
    second_wait,
)
from src.model.types import PlanningData


def make_data(**changes):
    values = dict(
        n_l=1,
        n_h=2,
        demand=np.array([4.0, 7.0]),
        flight_times=np.array([[0, 0.05, 0.07], [0.05, 0, 0.10], [0.07, 0.10, 0]]),
        rider_mean=np.array([[0.08, 0.12], [0.10, 0.06]]),
        first_cost=1.0,
        terminal_cost=0.5,
        receipt_mean=0.08,
        removal_mean=0.03,
        ground_cost=5.0,
        deadline=1.0,
        base_cost=0.5,
        slot_counts=[10, 20],
        slot_cost=[0.1, 0.2],
        slot_epsilon=0.05,
        link_cost=0.0,
        kappa=np.array([[0.3, 0.9], [0.2, 0.7]]),
        delta=0.04,
        operating_limit=1.0,
        fleet_cost=1.0,
        flight_cost=0.2,
        waiting_cost=0.3,
    )
    values.update(changes)
    return PlanningData(**values)


@pytest.mark.parametrize(
    "kappa",
    [
        np.ones((3, 3)),
        np.zeros((3, 3)),
        np.array([[0.2, 0.8, 0.1], [0.4, 0.6, 1], [0.9, 0, 0.3]]),
    ],
)
def test_reconstruction_conserves_each_od_and_task_counts(kappa):
    h = np.array([3.0, 5.0, 9.0])
    rates = demand_rates(h, [1, 1, 1], kappa, 0.17)
    np.testing.assert_allclose(rates.reconstructed_orders, h, atol=1e-12)
    assert rates.task_rate == pytest.approx(
        rates.single_rates.sum() + rates.pair_rates.sum()
    )
    assert rates.pool_rate == pytest.approx(
        rates.single_rates.sum() + 2 * rates.pair_rates.sum()
    )
    assert np.all(rates.theta >= (1 - rates.q) * h - 1e-12)
    assert np.all(rates.theta <= h + 1e-12)
    if np.all(kappa == 1):
        np.testing.assert_allclose(rates.theta, h / (1 + rates.q))
    if np.all(kappa == 0):
        np.testing.assert_array_equal(rates.single_rates, h)


def test_empty_pool_zero_window_k1_and_aa_accounting():
    for kwargs in ({"delta": 0.0}, {"delta": 0.5, "max_orders": 1}):
        rates = demand_rates([6.0], [1], [[1.0]], **kwargs)
        assert rates.task_rate == 6.0
        assert rates.pair_rates[0, 0] == 0.0
    empty = demand_rates([6.0], [0], [[1.0]], 0.5)
    assert empty.pool_rate == empty.task_rate == empty.q == 0
    aa = demand_rates([6.0], [1], [[1.0]], 0.5)
    assert aa.single_rates[0] + 2 * aa.pair_rates[0, 0] == pytest.approx(6.0)


def test_asymmetric_kappa_requires_od_specific_theta():
    h = np.array([10.0, 10.0])
    rates = demand_rates(h, [1, 1], [[0, 1], [0, 0]], 0.1)
    assert rates.theta[0] == pytest.approx(10)
    assert rates.theta[1] < rates.theta[0]
    np.testing.assert_allclose(rates.reconstructed_orders, h)


@pytest.mark.parametrize(
    "mean,transfer,delta",
    [(0.12, 0.03, 0.08), (0.06, 0.0, 0.04), (0.10, 0.2, 1e-10), (0.0, 0.1, 0.03)],
)
def test_uniform_wait_and_pickup_match_independent_integral(mean, transfer, delta):
    # E[(R-a)+] = integral_a^infinity P(R>t) dt; then average over uniform V.
    def conditional(v):
        if mean == 0:
            return 0.0
        return quad(lambda t: math.exp(-t / mean), transfer + v, np.inf, epsabs=1e-12)[
            0
        ]

    expected_wait = quad(lambda v: conditional(v * delta), 0, 1, epsabs=1e-12)[0]
    actual_wait = second_wait(mean, transfer, delta)
    assert actual_wait == pytest.approx(expected_wait, abs=1e-11)
    expected_pickup = quad(
        lambda v: transfer + v * delta + conditional(v * delta), 0, 1, epsabs=1e-12
    )[0]
    assert transfer + delta / 2 + actual_wait == pytest.approx(
        expected_pickup, abs=1e-11
    )
    assert second_wait(0.1, 0.03, 0) == pytest.approx(0.1 * math.exp(-0.3))


def test_aa_shared_lockers_count_both_reservations_and_own_clocks():
    data = make_data()
    segment = build_initial_segment(data, (0, 0), 0, 0, 1, ((0, 1), (1, 2)), id=3)
    path = build_service_path(
        data,
        segment,
        ((0, 1), (1, 2), (2,), (2, 1), (1, 0)),
        second_od=0,
        second_pickup=1,
        second_delivery=0,
        id=7,
    )
    assert path.served_counts == ((0, 2),)
    q1, q2 = path.pickup_times
    slots = dict(path.slot_times)
    assert slots[0] == pytest.approx(q1 + q2 + 0.1 + 0.03)
    assert slots[1] == pytest.approx(q1 + 0.1 + 0.03 + q2)
    assert path.delivery_times == pytest.approx((q1 + 0.1 + 0.08, q2 + 0.1 + 0.08))
    assert path.time == pytest.approx(path.flight_time + path.wait)
    assert path.cost == pytest.approx(3.0 + 0.2 * path.flight_time + 0.3 * path.wait)
    # Independent rebuild does not trust saved time or cost fields.
    assert rebuild_path(data, replace(path, time=999, cost=999)) == path


def test_mean_deadlines_first_flight_window_and_operating_limit():
    data = make_data()
    with pytest.raises(PathInfeasible, match="window"):
        build_initial_segment(
            replace(data, delta=0.11), (0, 0), 0, 0, 1, ((0, 1), (1, 2))
        )
    with pytest.raises(PathInfeasible, match="first order"):
        build_initial_segment(
            replace(data, deadline=[0.01, 1]), (0, 0), 0, 0, 1, ((0, 1), (1, 2))
        )
    segment = build_initial_segment(data, (0, 0), 0, 0, 1, ((0, 1), (1, 2)))
    legs = ((0, 1), (1, 2), (2,), (2, 1), (1, 0))
    with pytest.raises(PathInfeasible, match="own mean"):
        build_service_path(
            replace(data, deadline=[1, 0.01]),
            segment,
            legs,
            second_od=1,
            second_pickup=1,
            second_delivery=0,
        )
    with pytest.raises(PathInfeasible, match="operating"):
        build_service_path(
            replace(data, operating_limit=0.1), segment, ((0, 1), (1, 2), (2, 0))
        )
    with pytest.raises(PathInfeasible, match="spatially"):
        build_service_path(
            replace(data, kappa=np.zeros((2, 2))),
            segment,
            legs,
            second_od=1,
            second_pickup=1,
            second_delivery=0,
        )


def test_enumeration_keeps_first_leg_when_required_completion_is_absent():
    allowed = np.zeros((3, 3), dtype=bool)
    allowed[0, 1] = allowed[1, 2] = True
    paths = enumerate_paths(make_data(allowed_links=allowed))
    assert len(paths.initial_segments) == 2
    assert paths.service_paths == ()
    assert paths.complete


def test_all_simple_and_supplied_multi_arc_route_scopes():
    direct = make_data(max_orders=1)
    simple = replace(direct, route_mode="simple")
    paths = enumerate_paths(simple)
    assert paths.scope == "simple"
    assert len(paths.service_paths) > len(enumerate_paths(direct).service_paths)
    assert any(len(leg) > 2 for path in paths.service_paths for leg in path.legs)
    routes = ((0, 2, 1), (1, 2), (2, 0))
    supplied = enumerate_paths(replace(direct, candidate_routes=routes))
    assert supplied.scope == "supplied_routes"
    assert len(supplied.service_paths) == 2
    assert all(path.legs == routes for path in supplied.service_paths)


def test_budget_guards_raise_without_claiming_complete():
    data = make_data()
    for kwargs, reason in (
        (dict(column_limit=0), "column_limit"),
        (dict(time_limit=0), "time_limit"),
        (dict(stop_requested=lambda: True), "user_stop"),
    ):
        with pytest.raises(EnumerationStopped) as caught:
            enumerate_paths(data, **kwargs)
        assert caught.value.reason == reason


def test_empty_activity_uses_flight_time_without_turnaround():
    data = make_data(
        n_l=2,
        flight_times=np.array(
            [
                [0, 0.08, 0.05, 0.07],
                [0.08, 0, 0.06, 0.04],
                [0.05, 0.06, 0, 0.10],
                [0.07, 0.04, 0.10, 0],
            ]
        ),
    )
    path = build_empty_path(data, (0, 1), (0, 1))
    assert path.time == path.flight_time == 0.08
    assert path.cost == pytest.approx(0.08 * data.flight_cost)


def test_schema_readonly_and_rejects_inconsistent_parameters():
    data = make_data()
    with pytest.raises(ValueError):
        data.demand[0] = 10
    with pytest.raises(ValueError, match="kappa"):
        make_data(kappa=1.1)
    with pytest.raises(ValueError, match="removal"):
        make_data(removal_mean=0.2)
    with pytest.raises(ValueError, match="integer"):
        make_data(slot_counts=[1.5])
    with pytest.raises(ValueError, match="elementary"):
        make_data(candidate_routes=((0, 1, 0, 2),))
