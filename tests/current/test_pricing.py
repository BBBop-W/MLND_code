"""Independent finite-path checks for the current manuscript's oracle."""

import itertools
import math

import numpy as np
import pytest
from scipy.integrate import quad

import src.solvers.pricing.label as oracle


def _fixture():
    bases = np.asarray([[0.0, 0.0], [4.0, 0.0]])
    hubs = np.asarray([[0.8, 0.2], [2.0, 0.4], [3.2, 0.2]])
    remote = np.asarray([[1.2, 0.0], [2.8, 0.0]])
    xy = np.vstack((bases, hubs, remote))
    ft = np.linalg.norm(xy[:, None] - xy[None, :], axis=2) * 0.03
    return dict(
        bases=bases,
        hubs=hubs,
        remote=remote,
        n_l=2,
        n_h=3,
        n_k=2,
        n_od=3,
        flight_times=ft,
        battery_time=0.17,
        swap_time=np.asarray([0.012, 0.016]),
        pickup_handling_time=np.asarray([0.007, 0.010, 0.006]),
        delivery_handling_time=0.009,
        ground_time=np.asarray([[0.018, 0.031, 0.038], [0.04, 0.021, 0.015]]),
        first_cost=np.asarray([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]]),
        terminal_cost=np.asarray([[0.8, 0.4, 0.1], [0.1, 0.4, 0.8]]),
        terminal_time=np.asarray([[0.035, 0.021, 0.008], [0.008, 0.021, 0.035]]),
        delivery_deadline=np.asarray([0.20, 0.21, 0.18]),
        od_origin=np.asarray([0, 1, 0]),
        od_destination=np.asarray([0, 1, 1]),
        ground=np.asarray([8.0, 9.0, 7.0]),
        fleet_cost=2.0,
        drone_time_cost=3.0,
        hover_cost=4.0,
        epsilon=0.1,
        rider_time_family="exponential",
    )


def _brute(data, fac, od, l, m, hp, ap, all_paths=False):
    """Independently replay every finite route, interval by interval."""
    L, H, K = (data[k] for k in ("n_l", "n_h", "n_k"))
    ft = data["flight_times"]
    B = data["battery_time"]

    def val(key, index):
        a = np.asarray(data[key])
        return float(a if a.ndim == 0 else a[index])

    if not fac[l] or not fac[m]:
        return []
    paths = []
    ks = [-1] + [k for k in range(K) if fac[L + H + k]]
    pairs = itertools.permutations(range(H), 2) if od is not None else [(None, None)]
    for i, j in pairs:
        if od is not None and (not fac[L + i] or not fac[L + j]):
            continue
        endpoints = (
            [(l, L + i), (L + i, L + j), (L + j, m)] if od is not None else [(l, m)]
        )
        for swaps in itertools.product(ks, repeat=len(endpoints)):
            routes = [
                [u, v] if k < 0 else [u, L + H + k, v]
                for (u, v), k in zip(endpoints, swaps)
            ]
            fly = [sum(ft[u, v] for u, v in zip(route, route[1:])) for route in routes]
            if not np.isfinite(fly).all():
                continue
            sw = [0.0 if k < 0 else val("swap_time", k) for k in swaps]
            edges = [(u, v) for route in routes for u, v in zip(route, route[1:])]
            used = 0.0
            intervals = []
            w = 0.0
            qw = 0.0
            handling = 0.0
            fixed = 0.0
            hub = 0.0
            deadline = 0.0
            if od is not None:
                o, d = data["od_origin"][od], data["od_destination"][od]
                mean = data["ground_time"][o, i]
                arrival = fly[0] + sw[0]
                if data["rider_time_family"] == "deterministic":
                    w = max(0.0, mean - arrival)
                    q = mean
                else:
                    w = (
                        quad(
                            lambda t: (t - arrival) * math.exp(-t / mean) / mean,
                            arrival,
                            np.inf,
                            epsabs=1e-12,
                        )[0]
                        if mean
                        else 0.0
                    )
                    q = -mean * math.log(data["epsilon"])
                qw = max(0.0, q - arrival)
                handling = val("pickup_handling_time", i) + val(
                    "delivery_handling_time", j
                )
                deadline = (
                    max(arrival, q)
                    + handling
                    + fly[1]
                    + sw[1]
                    + data["terminal_time"][d, j]
                )
                if deadline > data["delivery_deadline"][od] + 1e-10:
                    continue
                fixed = (
                    data["first_cost"][o, i]
                    + data["terminal_cost"][d, j]
                    - data["ground"][od]
                )
                hub = hp[i] + hp[j]
            for stage, route in enumerate(routes):
                if od is not None and stage == 1:
                    used += qw + val("pickup_handling_time", i)
                if od is not None and stage == 2:
                    used += val("delivery_handling_time", j)
                for u, v in zip(route, route[1:]):
                    used += ft[u, v]
                    if v >= L + H:
                        intervals.append(used)
                        used = 0.0
            intervals.append(used)
            if max(intervals) > B + 1e-10:
                continue
            duration = sum(fly) + sum(sw) + handling + w
            real = (
                fixed
                + data["drone_time_cost"] * sum(fly)
                + data["hover_cost"] * w
                + data["fleet_cost"] * duration
            )
            priced = real + hub + sum(ap[u, v] for u, v in edges)
            paths.append(
                dict(
                    cost=real,
                    pricing_cost=priced,
                    time=duration,
                    i=i,
                    j=j,
                    swaps=swaps,
                    intervals=intervals,
                    deadline=deadline,
                    edges=edges,
                )
            )
    return paths


@pytest.mark.parametrize("family", ["deterministic", "exponential"])
@pytest.mark.parametrize("seed", [0, 17, 92])
def test_exact_pricing_against_independent_enumeration(family, seed):
    d = _fixture()
    d["rider_time_family"] = family
    rng = np.random.default_rng(seed)
    N = len(d["flight_times"])
    hp = rng.uniform(0, 3, d["n_h"])
    ap = rng.uniform(0, 2, (N, N))
    fac = np.ones(N, dtype=int)
    if seed == 17:
        fac[-1] = 0
    if seed == 92:
        fac[d["n_l"] + 1] = 0
    result = oracle.price_design(d, fac, hp, ap)
    for od, l, m in itertools.product(
        range(d["n_od"]), range(d["n_l"]), range(d["n_l"])
    ):
        paths = _brute(d, fac, od, l, m, hp, ap)
        best = min([p["pricing_cost"] for p in paths], default=np.inf)
        assert result["pricing_cost"][od, l, m] == pytest.approx(best, abs=1e-9)
        if paths:
            tr = oracle.trace_mission(result, od, l, m)
            matching = [p for p in paths if abs(p["pricing_cost"] - best) < 1e-8]
            assert any(
                abs(p["time"] - tr["time"]) < 1e-9
                and abs(p["cost"] - tr["cost"]) < 1e-9
                for p in matching
            )
            assert max(tr["battery_interval_consumptions"]) <= d["battery_time"] + 1e-9
            assert tr["deadline_quantile_time"] <= d["delivery_deadline"][od] + 1e-9
            assert sum(tr["duration_components"].values()) == pytest.approx(tr["time"])
            assert not tr["segments"][-1]["swap_at_head"]
    for l, m in itertools.permutations(range(d["n_l"]), 2):
        paths = _brute(d, fac, None, l, m, hp, ap)
        assert result["empty_pricing_cost"][l, m] == pytest.approx(
            min([p["pricing_cost"] for p in paths], default=np.inf)
        )
        if paths:
            oracle.trace_empty(result, l, m)


def test_time_label_survives_after_pickup_until_destination_deadline():
    # Direct loaded leg is fast but dear; its cheap swapping detour is too late.
    d = _fixture()
    d.update(
        n_l=1,
        n_h=2,
        n_k=1,
        n_od=1,
        rider_time_family="deterministic",
        bases=np.zeros((1, 2)),
        hubs=np.zeros((2, 2)),
        remote=np.zeros((1, 2)),
        ground_time=np.zeros((1, 2)),
        first_cost=np.asarray([[0.0, 100.0]]),
        terminal_cost=np.zeros((1, 2)),
        terminal_time=np.zeros((1, 2)),
        od_origin=np.asarray([0]),
        od_destination=np.asarray([0]),
        ground=np.asarray([10.0]),
        delivery_deadline=np.asarray([0.12]),
        battery_time=1.0,
        pickup_handling_time=0.0,
        delivery_handling_time=0.0,
        swap_time=0.01,
    )
    ft = np.full((4, 4), np.inf)
    np.fill_diagonal(ft, 0.0)
    ft[0, 1] = 0.02
    ft[1, 2] = 0.02
    ft[1, 3] = 0.06
    ft[3, 2] = 0.06
    ft[2, 0] = 0.01
    d["flight_times"] = ft
    ap = np.zeros((4, 4))
    ap[1, 2] = 10.0
    result = oracle.price_design(d, arc_prices=ap)
    tr = oracle.trace_mission(result, 0, 0, 0)
    assert tr["swap_choices"] == [-1, -1, -1]
    assert tr["deadline_quantile_time"] == pytest.approx(0.04)
    d["delivery_deadline"] = np.asarray([0.3])
    later = oracle.trace_mission(oracle.price_design(d, arc_prices=ap), 0, 0, 0)
    assert later["swap_choices"] == [-1, 0, -1]


def test_quantile_wait_and_handling_consume_battery_not_mean_wait():
    d = _fixture()
    d.update(
        n_l=1,
        n_h=2,
        n_k=0,
        n_od=1,
        bases=np.zeros((1, 2)),
        hubs=np.zeros((2, 2)),
        remote=np.zeros((0, 2)),
        ground_time=np.asarray([[0.04, 0.04]]),
        first_cost=np.zeros((1, 2)),
        terminal_cost=np.zeros((1, 2)),
        terminal_time=np.zeros((1, 2)),
        od_origin=np.asarray([0]),
        od_destination=np.asarray([0]),
        ground=np.asarray([10.0]),
        delivery_deadline=np.asarray([1.0]),
        battery_time=0.15,
        pickup_handling_time=0.01,
        delivery_handling_time=0.01,
        swap_time=0.0,
    )
    d["flight_times"] = np.full((3, 3), 0.02)
    np.fill_diagonal(d["flight_times"], 0.0)
    # Quantile T=.0921, extra wait=.0721, flight=.06, handling=.02 => .1521.
    assert not np.isfinite(oracle.price_design(d)["cost"]).any()
    d["battery_time"] = 0.16
    tr = oracle.trace_mission(oracle.price_design(d), 0, 0, 0)
    assert tr["wait_quantile"] > tr["expected_synchronization_wait_hours"]
    assert tr["battery_interval_consumptions"] == pytest.approx(
        [0.06 + 0.02 - 0.04 * math.log(0.1) - 0.02]
    )
    assert tr["time"] == pytest.approx(0.08 + 0.04 * math.exp(-0.5))


def test_same_station_can_be_used_on_different_legs_no_final_swap():
    d = _fixture()
    d.update(
        n_l=1,
        n_h=2,
        n_k=1,
        n_od=1,
        bases=np.zeros((1, 2)),
        hubs=np.zeros((2, 2)),
        remote=np.zeros((1, 2)),
        ground_time=np.zeros((1, 2)),
        first_cost=np.zeros((1, 2)),
        terminal_cost=np.zeros((1, 2)),
        terminal_time=np.zeros((1, 2)),
        od_origin=np.asarray([0]),
        od_destination=np.asarray([0]),
        ground=np.asarray([10.0]),
        delivery_deadline=np.asarray([1.0]),
        battery_time=0.05,
        pickup_handling_time=0.001,
        delivery_handling_time=0.001,
        swap_time=0.01,
    )
    ft = np.full((4, 4), np.inf)
    np.fill_diagonal(ft, 0.0)
    for v in range(3):
        ft[v, 3] = ft[3, v] = 0.02
    d["flight_times"] = ft
    tr = oracle.trace_mission(oracle.price_design(d), 0, 0, 0)
    assert tr["swap_choices"] == [0, 0, 0]
    assert tr["swap_count"] == 3
    assert tr["time"] == pytest.approx(0.12 + 0.03 + 0.002)
    assert len(tr["battery_interval_consumptions"]) == 4
    assert tr["duration_components"]["return_swap"] == pytest.approx(0.01)
    assert tr["segments"][-1]["swap_hours"] == 0.0


def test_full_graph_facility_masks_and_group_scope():
    d = _fixture()
    N = len(d["flight_times"])
    full = oracle.price_design(d)
    fac = np.ones(N, dtype=int)
    fac[d["n_l"] : d["n_l"] + d["n_h"]] = 0
    closed = oracle.price_design(d, facility=fac)
    assert np.isfinite(full["cost"]).any() and not np.isfinite(closed["cost"]).any()
    assert full["full_graph"] and not closed["full_graph"]
    subset = oracle.price_design(d, groups=[(0, 0, 1), (2, 1, 0)])
    assert not subset["complete"] and subset["groups_priced"] == 2
    assert subset["cost"][0, 0, 1] == pytest.approx(full["cost"][0, 0, 1])
    assert subset["cost"][2, 1, 0] == pytest.approx(full["cost"][2, 1, 0])
    assert not np.isfinite(subset["cost"][1]).any()
    assert np.array_equal(subset["empty_cost"], full["empty_cost"])


def test_public_full_columns_match_independent_path_count():
    d = _fixture()
    N = len(d["flight_times"])
    fac = np.ones(N)
    hp = np.zeros(d["n_h"])
    ap = np.zeros((N, N))
    columns = oracle.enumerate_all_columns(d)
    expected = sum(
        len(_brute(d, fac, od, l, m, hp, ap))
        for od, l, m in itertools.product(range(d["n_od"]), range(2), range(2))
    )
    assert len(columns["service"]) == expected
    expected_empty = sum(
        len(_brute(d, fac, None, l, m, hp, ap))
        for l, m in itertools.permutations(range(2), 2)
    )
    assert len(columns["empty"]) == expected_empty


def test_zero_mean_and_absent_arcs():
    d = _fixture()
    d["ground_time"][:] = 0.0
    d["flight_times"][0, :] = np.inf
    d["flight_times"][0, 0] = 0.0
    result = oracle.price_design(d)
    assert not np.isfinite(result["cost"][:, 0]).any()
    for od, l, m in np.argwhere(np.isfinite(result["cost"])):
        assert (
            oracle.trace_mission(result, int(od), int(l), int(m))["wait_quantile"]
            == 0.0
        )


@pytest.mark.parametrize(
    "field,value",
    [("epsilon", 0.0), ("epsilon", 1.0), ("battery_time", 0.0), ("hover_cost", -1.0)],
)
def test_invalid_physical_inputs_rejected(field, value):
    d = _fixture()
    d[field] = value
    with pytest.raises(ValueError):
        oracle.price_design(d)
