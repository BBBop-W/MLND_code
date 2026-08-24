from __future__ import annotations

import math

import pytest

from src.model.valid_inequalities import (
    ChoiceCorridorCoverData,
    build_choice_corridor_cover_data,
    most_violated_queue_perspective_cut,
    queue_perspective_intercept,
    queue_support_minimizer,
    separate_choice_capacity_covers,
    separate_directed_return_cuts,
    separate_queue_perspective_cut,
    stable_arrival_capacity,
)


def _queue_value(mu: float, arrival: float) -> float:
    return arrival / (mu - arrival)


def test_queue_intercept_matches_exact_piecewise_minimum() -> None:
    mu = 12.0
    kappa = 8.0
    lower = 1.0 / mu
    upper = mu / (mu - kappa) ** 2
    for pi in (0.0, lower, 0.2, upper, 2.0):
        intercept = queue_perspective_intercept(mu, kappa, pi)
        minimizer = queue_support_minimizer(mu, kappa, pi)
        assert 0.0 <= minimizer <= kappa
        assert intercept == pytest.approx(
            _queue_value(mu, minimizer) - pi * minimizer,
            abs=1e-12,
        )
        brute = min(
            _queue_value(mu, kappa * step / 20000.0)
            - pi * kappa * step / 20000.0
            for step in range(20001)
        )
        assert intercept == pytest.approx(brute, abs=2e-7)


def test_queue_support_cut_is_valid_for_every_capacity_disjunct() -> None:
    service_rates = (12.0, 18.0, 24.0)
    tau = 0.25
    kappas = tuple(stable_arrival_capacity(mu, tau) for mu in service_rates)
    for pi in (0.0, 0.1, 0.3, 1.0):
        coefficients = tuple(
            queue_perspective_intercept(mu, kappa, pi)
            for mu, kappa in zip(service_rates, kappas)
        )
        for c, (mu, kappa) in enumerate(zip(service_rates, kappas)):
            for step in range(101):
                arrival = kappa * step / 100.0
                rhs = pi * arrival + coefficients[c]
                assert _queue_value(mu, arrival) + 1e-10 >= rhs


def test_queue_separator_finds_global_one_dimensional_maximum() -> None:
    service_rates = (12.0, 18.0, 24.0)
    tau = 0.25
    zbar = (0.25, 0.35, 0.20)
    lambda_bar = 7.0
    xi_bar = 0.05
    cut = most_violated_queue_perspective_cut(
        lambda_bar,
        xi_bar,
        zbar,
        service_rates,
        tau,
    )
    assert cut.violation > 0.0
    assert sum(z * t for z, t in zip(zbar, cut.support_arrivals)) == pytest.approx(
        lambda_bar, abs=1e-8
    )

    max_breakpoint = max(
        mu / (mu - stable_arrival_capacity(mu, tau)) ** 2 for mu in service_rates
    )
    brute_rhs = -math.inf
    for step in range(50001):
        pi = max_breakpoint * step / 50000.0
        rhs = pi * lambda_bar + sum(
            z * queue_perspective_intercept(
                mu, stable_arrival_capacity(mu, tau), pi
            )
            for z, mu in zip(zbar, service_rates)
        )
        brute_rhs = max(brute_rhs, rhs)
    assert cut.rhs_at_point == pytest.approx(brute_rhs, abs=2e-6)
    assert separate_queue_perspective_cut(
        lambda_bar,
        cut.rhs_at_point,
        zbar,
        service_rates,
        tau,
    ) is None


def test_queue_separator_rejects_point_outside_fractional_qos_domain() -> None:
    with pytest.raises(ValueError, match="exceeds the fractional QoS capacity"):
        most_violated_queue_perspective_cut(
            lambda_value=5.0,
            xi_value=0.0,
            z_values=(0.1,),
            service_rates=(12.0,),
            max_system_time=0.25,
        )


def test_choice_corridor_data_counts_each_market_once_not_each_origin() -> None:
    data = build_choice_corridor_cover_data(
        h_od={
            ("O1", "D1"): 10.0,
            ("O2", "D1"): 20.0,
            ("O3", "D2"): 5.0,
        },
        choice_share={
            ("O1", "D1", "H2"): 0.4,
            ("O2", "D1", "H2"): 0.5,
            ("O3", "D2", "H2"): 0.8,
        },
        locker_origins={
            ("O1", "D1", "H2"): ("H1", "H3"),
            ("O2", "D1", "H2"): ("H1",),
            ("O3", "D2", "H2"): ("H3",),
        },
    )
    assert [item.alpha_key for item in data] == [("D1", "H2"), ("D2", "H2")]
    assert data[0].offered_demand == pytest.approx(14.0)
    assert data[0].incoming_corridors == (("H1", "H2"), ("H3", "H2"))
    assert data[1].offered_demand == pytest.approx(4.0)


def test_choice_capacity_cover_separator_returns_valid_violated_subset() -> None:
    data = (
        ChoiceCorridorCoverData("D1", "H1", 8.0, (), (("H2", "H1"),)),
        ChoiceCorridorCoverData("D2", "H1", 7.0, (), (("H3", "H1"),)),
        ChoiceCorridorCoverData("D3", "H1", 2.0, (), (("H2", "H1"),)),
    )
    cuts = separate_choice_capacity_covers(
        data=data,
        capacity_limits=(5.0, 10.0, 20.0),
        rho_min=0.8,
        alpha_values={
            ("D1", "H1"): 0.95,
            ("D2", "H1"): 0.90,
            ("D3", "H1"): 0.10,
        },
        z_values={
            ("H1", 0): 0.75,
            ("H1", 1): 0.20,
            ("H1", 2): 0.05,
        },
        max_cuts_per_locker=None,
    )
    assert cuts
    best = cuts[0]
    assert best.destinations == ("D1", "D2")
    assert best.effective_arrival == pytest.approx(12.0)
    assert best.adequate_capacity_indices == (2,)
    assert best.violation == pytest.approx(0.80)

    # Exhaustively verify the returned combinatorial inequality for every
    # integral capacity and display selection satisfying the aggregate lower
    # bound W(alpha) <= kappa[z].
    weights = {item.destination: 0.8 * item.offered_demand for item in data}
    for capacity_index, capacity in enumerate((5.0, 10.0, 20.0)):
        for mask in range(1 << len(data)):
            selected = {
                item.destination for bit, item in enumerate(data) if mask & (1 << bit)
            }
            if sum(weights[destination] for destination in selected) > capacity + 1e-12:
                continue
            lhs = sum(destination in selected for destination in best.destinations)
            rhs = len(best.destinations) - 1 + (
                capacity_index in best.adequate_capacity_indices
            )
            assert lhs <= rhs


def test_choice_capacity_cover_deduplicates_external_signatures() -> None:
    item = ChoiceCorridorCoverData("D1", "H1", 10.0, (), (("H2", "H1"),))
    first = separate_choice_capacity_covers(
        (item,),
        capacity_limits=(5.0,),
        rho_min=1.0,
        alpha_values={("D1", "H1"): 0.8},
        z_values={("H1", 0): 0.0},
    )
    assert len(first) == 1
    second = separate_choice_capacity_covers(
        (item,),
        capacity_limits=(5.0,),
        rho_min=1.0,
        alpha_values={("D1", "H1"): 0.8},
        z_values={("H1", 0): 0.0},
        seen_signatures={first[0].signature()},
    )
    assert second == ()


def test_directed_return_separator_finds_expected_min_cut() -> None:
    arcs = (
        ("H1", "H2"),
        ("H2", "H3"),
        ("H2", "H4"),
        ("H3", "H1"),
        ("H4", "H1"),
    )
    cuts = separate_directed_return_cuts(
        physical_arcs=arcs,
        y_values={
            ("H1", "H2"): 1.0,
            ("H2", "H3"): 0.10,
            ("H2", "H4"): 0.20,
            ("H3", "H1"): 0.80,
            ("H4", "H1"): 0.70,
        },
        h_values={("H1", "H2"): 5.0},
        big_m={("H1", "H2"): 10.0},
    )
    assert len(cuts) == 1
    cut = cuts[0]
    assert cut.source_side == ("H2",)
    assert cut.crossing_arcs == (("H2", "H3"), ("H2", "H4"))
    assert cut.cut_capacity == pytest.approx(0.30)
    assert cut.violation == pytest.approx(2.0)


def test_directed_return_separator_contracts_virtual_holding_state() -> None:
    arcs = (
        ("H1", "H2"),
        ("H2", "S_H2"),  # contracts to a self-loop and cannot cross a cut
        ("S_H2", "H3"),
        ("H3", "H1"),
    )
    cuts = separate_directed_return_cuts(
        physical_arcs=arcs,
        y_values={arc: value for arc, value in zip(arcs, (1.0, 0.0, 0.10, 0.20))},
        h_values={("H1", "H2"): 3.0},
        big_m={("H1", "H2"): 10.0},
        contraction={"S_H2": "H2"},
    )
    assert len(cuts) == 1
    assert ("H2", "S_H2") not in cuts[0].crossing_arcs
    assert cuts[0].crossing_arcs == (("S_H2", "H3"),)
    assert cuts[0].violation == pytest.approx(2.0)


def test_directed_return_separator_handles_zero_capacity_and_deduplication() -> None:
    arcs = (("H1", "H2"), ("H2", "H1"))
    first = separate_directed_return_cuts(
        arcs,
        y_values={("H1", "H2"): 1.0, ("H2", "H1"): 0.0},
        h_values={("H1", "H2"): 1.0},
        big_m={("H1", "H2"): 4.0},
    )
    assert len(first) == 1
    assert first[0].crossing_arcs == (("H2", "H1"),)
    assert first[0].cut_capacity == 0.0
    assert separate_directed_return_cuts(
        arcs,
        y_values={("H1", "H2"): 1.0, ("H2", "H1"): 0.0},
        h_values={("H1", "H2"): 1.0},
        big_m={("H1", "H2"): 4.0},
        seen_signatures={first[0].signature()},
    ) == ()


def test_directed_return_separator_returns_no_cut_when_satisfied() -> None:
    assert separate_directed_return_cuts(
        (("H1", "H2"), ("H2", "H1")),
        y_values={("H1", "H2"): 1.0, ("H2", "H1"): 0.6},
        h_values={("H1", "H2"): 5.0},
        big_m={("H1", "H2"): 10.0},
    ) == ()
