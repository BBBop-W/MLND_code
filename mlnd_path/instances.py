"""Synthetic, fully reproducible instances for the base--path model.

Time is in hours, distance in km, demand in orders/hour, and costs in a common
currency.  These are controlled mathematical fixtures, not calibrated data.

The path preprocessing is exact for the integer facility designs in the main
model (positive fleet cost, activation gates, no binding physical station or
fleet capacities).  For every allowed swap-facility subset and every fixed
OD/base pair/locker pair it retains a cheapest complete mission.  Its union
therefore contains a cheapest feasible mission for every integer design.
It does not claim to preserve the full-column fractional facility relaxation.
"""

from __future__ import annotations

import heapq
import itertools
import math
import random
import time
from dataclasses import dataclass
from typing import Any

from .data import Instance

TOL = 1e-9


def expected_sync(arrival: float, ground: float, mean: float) -> float:
    """E[max(arrival, X+ground)], X exponential with strictly positive mean."""
    if mean <= 0:
        return max(arrival, ground)
    if arrival <= ground:
        return ground + mean
    return arrival + mean * math.exp(-(arrival - ground) / mean)


@dataclass(frozen=True)
class Geometry:
    bases: tuple[tuple[float, float], ...]
    hubs: tuple[tuple[float, float], ...]
    remote: tuple[tuple[float, float], ...]
    battery: float = 15.0
    reserve: float = 0.5
    loaded_energy: float = 1.20
    speed: float = 60.0
    flight_cost: float = 0.12
    swap_time: float = 0.025
    swap_fee: float = 0.25
    handling_pickup: float = 0.008
    handling_delivery: float = 0.006
    fleet_cost: float = 3.0

    @property
    def stations(self):
        return self.bases + self.remote

    def location(self, node):
        return self.hubs[node[1]] if node[0] == "H" else self.stations[node[1]]

    def dist(self, left, right):
        return math.dist(self.location(left), self.location(right))


@dataclass(eq=False)
class Label:
    stage: int
    node: tuple[str, int]
    energy: float
    elapsed: float
    flight_cost: float
    arrival: float
    route: tuple[tuple[str, int], ...]
    swaps: tuple[int, ...]
    energy_trace: tuple[tuple[float, float], ...]
    score: float
    active: bool = True


def _dominates(left: Label, right: Label):
    if left.energy < right.energy - TOL:
        return False
    if left.stage == 0:
        # Before pickup, both early arrival and cheap travel are useful; the
        # max synchronization transformation is monotone in arrival time.
        return (left.elapsed <= right.elapsed + TOL
                and left.flight_cost <= right.flight_cost + TOL)
    return left.score <= right.score + TOL


def _label_route(geo: Geometry, l: int, m: int, allowed: tuple[int, ...],
                 i: int | None = None, j: int | None = None,
                 ready_mean: float = 0.0, ground_time: float = 0.0):
    """Exact resource-label search; no hop truncation or top-k filtering.

    A state is stage/current node/residual energy.  Before the pickup target,
    travel time and monetary cost are compared separately.  After applying the
    expected synchronization time, objective cost is additive.  Every swap
    restores a common full state, so same-stage positive-cost cycles are
    dominated automatically; a station may be used again in another stage.
    """
    is_empty = i is None
    first_stage = 2 if is_empty else 0
    start = Label(first_stage, ("S", l), geo.battery, 0.0, 0.0, 0.0,
                  (("L", l),), (0,) * len(geo.stations), (), 0.0)
    fronts: dict[tuple[int, tuple[str, int]], list[Label]] = {}
    heap = []
    serial = itertools.count()
    n_labels = 0
    n_expanded = 0
    best = None

    def offer(label):
        nonlocal n_labels
        key = (label.stage, label.node)
        front = fronts.setdefault(key, [])
        if any(_dominates(old, label) for old in front):
            return
        survivors = []
        for old in front:
            if _dominates(label, old):
                old.active = False
            else:
                survivors.append(old)
        survivors.append(label)
        fronts[key] = survivors
        heapq.heappush(heap, (label.score, next(serial), label))
        n_labels += 1

    offer(start)
    while heap:
        _, _, cur = heapq.heappop(heap)
        if not cur.active:
            continue
        if best is not None and cur.score > best.score + TOL:
            continue
        n_expanded += 1
        # A swap at the final target is the terminal action, not an extra
        # intermediate action.  At earlier stages that base remains usable.
        next_nodes = [("S", k) for k in allowed
                      if ("S", k) != cur.node and not (cur.stage == 2 and k == m)]
        target = (("H", i) if cur.stage == 0 else
                  ("H", j) if cur.stage == 1 else ("S", m))
        next_nodes.append(target)
        for node in next_nodes:
            distance = geo.dist(cur.node, node)
            factor = geo.loaded_energy if cur.stage == 1 else 1.0
            residual = cur.energy - factor * distance
            if residual < geo.reserve - TOL:
                continue
            elapsed = cur.elapsed + distance / geo.speed
            cost = cur.flight_cost + geo.flight_cost * distance * (1.06 if cur.stage == 1 else 1.0)
            stage = cur.stage
            counts = cur.swaps
            energy = residual
            arrival = cur.arrival
            if node == target:
                if cur.stage == 0:
                    arrival = elapsed
                    elapsed = expected_sync(elapsed, ground_time, ready_mean) + geo.handling_pickup
                    stage = 1
                    route_node = ("P", i)
                elif cur.stage == 1:
                    elapsed += geo.handling_delivery
                    stage = 2
                    route_node = ("D", j)
                else:
                    elapsed += geo.swap_time
                    cost += geo.swap_fee
                    energy = geo.battery
                    counts = tuple(c + (k == m) for k, c in enumerate(counts))
                    stage = 3
                    route_node = ("T", m)
            else:
                elapsed += geo.swap_time
                cost += geo.swap_fee
                energy = geo.battery
                counts = tuple(c + (k == node[1]) for k, c in enumerate(counts))
                route_node = node
            score = cost + geo.fleet_cost * elapsed
            new = Label(stage, node, energy, elapsed, cost, arrival,
                        cur.route + (route_node,), counts,
                        cur.energy_trace + ((residual, energy),), score)
            if stage == 3:
                if best is None or new.score < best.score - TOL:
                    best = new
            else:
                offer(new)
    stats = {"labels_generated": n_labels, "labels_expanded": n_expanded}
    if best is None:
        return None, stats
    return {"l": l, "m": m, "i": i, "j": j,
            "time": best.elapsed, "cost": best.flight_cost,
            "swaps": best.swaps, "route": best.route,
            "energy_trace": best.energy_trace, "pickup_arrival": best.arrival,
            "search_objective": best.score}, stats


def _allowed_sets(n_stations, l, m):
    mandatory = {l, m}
    optional = [k for k in range(n_stations) if k not in mandatory]
    for mask in range(1 << len(optional)):
        yield tuple(sorted(mandatory | {k for bit, k in enumerate(optional) if mask & (1 << bit)}))


def _coordinates(n_l, n_h, n_k, rng):
    # Bases on an inner ring, customer lockers on an outer ring, and optional
    # remote swap facilities through the interior produce genuine range gaps.
    bases = tuple((4.8 * math.cos(2 * math.pi * k / n_l),
                   4.8 * math.sin(2 * math.pi * k / n_l)) for k in range(n_l))
    hubs = tuple(((6.0 + rng.uniform(-0.5, 0.5)) * math.cos(2 * math.pi * (k + .18) / n_h),
                  (6.0 + rng.uniform(-0.5, 0.5)) * math.sin(2 * math.pi * (k + .18) / n_h))
                 for k in range(n_h))
    remote = tuple((0.0, 0.0) if k == 0 else
                   (2.3 * math.cos(2 * math.pi * (k - 1) / max(1, n_k - 1)),
                    2.3 * math.sin(2 * math.pi * (k - 1) / max(1, n_k - 1)))
                   for k in range(n_k))
    return bases, hubs, remote


def make_instance(n_l: int, n_h: int, n_od: int, n_k: int = 1,
                  seed: int = 17, name: str | None = None) -> Instance:
    """Build exact design-preserving columns for a controlled geometric case.

    OD records share merchant preparation distributions to make geometry-path
    caching explicit.  All potential H pairs are present; none is screened by
    merchant proximity, endpoint openness, reduced cost, or arbitrary length.
    """
    if n_l < 1 or n_h < 2 or n_od < 1 or n_k < 0:
        raise ValueError("Require n_l >= 1, n_h >= 2, n_od >= 1, n_k >= 0")
    start_time = time.perf_counter()
    rng = random.Random(seed)
    bases, hubs, remote = _coordinates(n_l, n_h, n_k, rng)
    geo = Geometry(bases=bases, hubs=hubs, remote=remote)
    n_merchant = min(n_od, max(2, (n_od + 2) // 3))
    merchant_points = tuple((hubs[(o * 2) % n_h][0] + rng.uniform(-.6, .6),
                             hubs[(o * 2) % n_h][1] + rng.uniform(-.6, .6))
                            for o in range(n_merchant))
    ready_mean = tuple(rng.uniform(.08, .15) for _ in range(n_merchant))
    od_merchant = [od % n_merchant for od in range(n_od)]
    customer_points = []
    demand = []
    ground_cost = []
    for od, merchant in enumerate(od_merchant):
        target = (2 * merchant + n_h // 2 + od // n_merchant) % n_h
        point = (hubs[target][0] + rng.uniform(-1.0, 1.0),
                 hubs[target][1] + rng.uniform(-1.0, 1.0))
        customer_points.append(point)
        demand.append(rng.uniform(2.5, 4.5))
        direct = 1.2 * math.dist(merchant_points[merchant], point)
        ground_cost.append(5.0 + .68 * direct)
    instance = Instance(name=name or f"L{n_l}_H{n_h}_OD{n_od}_K{n_k}_s{seed}",
                        n_l=n_l, n_h=n_h, n_k=n_k, demand=demand,
                        ground_cost=ground_cost,
                        base_cost=[rng.uniform(10.5, 14.5) for _ in range(n_l)],
                        hub_cost=[rng.uniform(2.0, 3.5) for _ in range(n_h)],
                        swap_cost=[rng.uniform(2.0, 3.0) for _ in range(n_k)],
                        fleet_cost=geo.fleet_cost)
    generated = expanded = searches = 0
    shared = {}
    n_stations = n_l + n_k
    for merchant in range(n_merchant):
        for l in range(n_l):
            for m in range(n_l):
                subsets = tuple(_allowed_sets(n_stations, l, m))
                for i in range(n_h):
                    g = 1.2 * math.dist(merchant_points[merchant], hubs[i]) / 18.0
                    for j in range(n_h):
                        if i == j:
                            continue
                        unique = {}
                        for allowed in subsets:
                            path, counts = _label_route(geo, l, m, allowed, i, j, ready_mean[merchant], g)
                            generated += counts["labels_generated"]
                            expanded += counts["labels_expanded"]
                            searches += 1
                            if path is not None:
                                unique[path["route"]] = path
                        shared[merchant, l, m, i, j] = tuple(unique.values())
    for od, merchant in enumerate(od_merchant):
        for l in range(n_l):
            for m in range(n_l):
                for i in range(n_h):
                    first_distance = 1.2 * math.dist(merchant_points[merchant], hubs[i])
                    first_cost = .65 + .25 * first_distance
                    for j in range(n_h):
                        if i == j:
                            continue
                        last_distance = 1.2 * math.dist(hubs[j], customer_points[od])
                        # The terminal H affects utility and the courier leg;
                        # preparation randomness does not change quoted time.
                        utility_pickup = 1.65 - 1.15 * last_distance
                        utility_delivery = -.45
                        pickup_share = 1.0 / (1.0 + math.exp(utility_delivery - utility_pickup))
                        pickup_cost = .10
                        delivery_cost = 1.10 + .40 * last_distance
                        last_cost = pickup_share * pickup_cost + (1.0 - pickup_share) * delivery_cost
                        for path in shared[merchant, l, m, i, j]:
                            col = dict(path)
                            col.update(od=od, merchant=merchant,
                                       cost=path["cost"] + first_cost + last_cost,
                                       pickup_share=pickup_share, ready_mean=ready_mean[merchant],
                                       first_ground_time=first_distance / 18.0,
                                       first_ground_cost=first_cost,
                                       terminal_expected_cost=last_cost)
                            instance.service.append(col)
    for l in range(n_l):
        for m in range(n_l):
            if l == m:
                continue
            unique = {}
            for allowed in _allowed_sets(n_stations, l, m):
                path, counts = _label_route(geo, l, m, allowed)
                generated += counts["labels_generated"]
                expanded += counts["labels_expanded"]
                searches += 1
                if path is not None:
                    unique[path["route"]] = path
            instance.empty.extend(unique.values())
    instance.metadata.update(
        synthetic=True, seed=seed, coordinates={"bases": bases, "hubs": hubs, "remote_swaps": remote,
                                                "merchants": merchant_points, "customers": customer_points},
        od_merchant=od_merchant, merchant_ready_means=ready_mean,
        battery_capacity=geo.battery, battery_reserve=geo.reserve,
        loaded_energy_factor=geo.loaded_energy, flight_speed_kmh=geo.speed,
        swap_time_hours=geo.swap_time, swap_fee=geo.swap_fee,
        pickup_distribution="Exponential; analytic E[max(A, xi + g)]", n_merchant=n_merchant,
        preprocessing="all swap subsets; exact resource labels; integer-design value preserving union",
        preprocessing_scope="activation gates only; positive fixed fleet price; no hard fleet capacity",
        pricing_universe="fixed preprocessed exact-for-integer-design column union; no open-only pricing",
        preserves_fractional_full_route_relaxation=False,
        service_columns=len(instance.service), empty_columns=len(instance.empty),
        service_columns_using_remote_swap=sum(any(p["swaps"][n_l:]) for p in instance.service),
        service_columns_using_intermediate_swap=sum(sum(p["swaps"]) > 1 for p in instance.service),
        route_searches=searches, labels_generated=generated, labels_expanded=expanded,
        generation_seconds=time.perf_counter() - start_time,
    )
    return instance


if __name__ == "__main__":
    import argparse
    import json
    parser = argparse.ArgumentParser()
    parser.add_argument("--bases", type=int, default=2)
    parser.add_argument("--hubs", type=int, default=3)
    parser.add_argument("--ods", type=int, default=2)
    parser.add_argument("--swaps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    case = make_instance(args.bases, args.hubs, args.ods, args.swaps, args.seed)
    print(json.dumps(case.metadata, indent=2))
