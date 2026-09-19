"""Independent reconstruction of physical path clocks, energy and reliability."""

from __future__ import annotations

from collections import Counter
import math
import numpy as np
from .arrays import _array, _broadcast, _maximum


def _cdf(threshold, mean, family):
    if threshold < 0:
        return 0.0
    if mean == 0:
        return 1.0
    if family == "deterministic":
        return float(threshold + 1e-12 >= mean)
    return -math.expm1(-threshold / mean)


def _trace(data, record, *, service):
    """Recompute durations and resources without the oracle's time helpers."""
    L, H, K = (int(data[k]) for k in ("n_l", "n_h", "n_k"))
    N = L + H + K
    tr = record.get("trace", record)
    get = lambda key: record.get(key, tr.get(key))
    l, m = int(get("l")), int(get("m"))
    if not (0 <= l < L and 0 <= m < L):
        raise ValueError("Path endpoints must be bases")
    raw = tr.get("legs")
    if raw is None:
        raise ValueError("Independent validation requires explicit path legs")
    legs = [[int(v) for v in leg] for leg in raw]
    if any(
        any(v != original for v, original in zip(leg, orig))
        for leg, orig in zip(legs, raw)
    ):
        raise ValueError("Path node indices must be integers")
    if service:
        od, i, j = (int(get(k)) for k in ("od", "i", "j"))
        if not (0 <= od < data["n_od"] and 0 <= i < H and 0 <= j < H and i != j):
            raise ValueError("Service path has an invalid OD or locker pair")
        endpoints = [(l, L + i), (L + i, L + j), (L + j, m)]
    else:
        if l == m:
            raise ValueError("Empty paths must connect distinct bases")
        od = i = j = None
        endpoints = [(l, m)]
    if len(legs) != len(endpoints):
        raise ValueError("Wrong number of physical path legs")
    flight = _array(data["flight_times"], (N, N), "flight_times")
    swaps = _broadcast(data.get("swap_time", 0.0), K, "swap_time")
    pickup = _broadcast(
        data.get("pickup_handling_time", 0.0), H, "pickup_handling_time"
    )
    delivery = _broadcast(
        data.get("delivery_handling_time", 0.0), H, "delivery_handling_time"
    )
    elapsed, interval = 0.0, 0.0
    intervals, segments, arc_counts, nodes = [], [], Counter(), set()
    stage_times, components = [], {}
    waiting_interval = None
    phases = ("pre_pickup", "loaded", "return") if service else ("empty",)
    for stage, (leg, (start, end), phase) in enumerate(zip(legs, endpoints, phases)):
        if len(leg) not in (2, 3) or leg[0] != start or leg[-1] != end:
            raise ValueError("A leg must be direct or visit one swapping station")
        if len(leg) == 3 and not L + H <= leg[1] < N:
            raise ValueError(
                "Only an independent swapping station may be internal to a leg"
            )
        if len(set(leg)) != len(leg):
            raise ValueError("A leg cannot repeat a node")
        elapsed_before = elapsed
        fly, sw = 0.0, 0.0
        if service and stage == 1:
            waiting_interval = len(intervals)
            interval += float(pickup[i])
            elapsed += float(pickup[i])
        for u, v in zip(leg[:-1], leg[1:]):
            if not (0 <= u < N and 0 <= v < N) or u == v:
                raise ValueError("Invalid physical flight arc")
            duration = float(flight[u, v])
            if duration < 0:
                raise ValueError("Negative flight time")
            nodes.update((u, v))
            arc_counts[u * N + v] += 1
            fly += duration
            elapsed += duration
            interval += duration
            swap = float(swaps[v - L - H]) if v >= L + H else 0.0
            if v >= L + H:
                intervals.append(interval)
                interval = 0.0
                sw += swap
                elapsed += swap
            segments.append(
                dict(
                    stage=stage,
                    phase=phase,
                    u_index=u,
                    v_index=v,
                    flight_hours=duration,
                    swap_hours=swap,
                    swap_at_head=v >= L + H,
                )
            )
        if service and stage == 1:
            interval += float(delivery[j])
            elapsed += float(delivery[j])
        stage_times.append(elapsed - elapsed_before)
        components[phase + "_flight"] = fly
        components[phase + "_swap"] = sw
    intervals.append(interval)
    battery = float(data["battery_time"])
    operating_cost = float(data["drone_time_cost"]) * sum(
        s["flight_hours"] for s in segments
    )
    family = data.get("rider_time_family", "exponential")
    if family not in ("exponential", "deterministic"):
        raise ValueError(
            "Operation validation supports exponential or deterministic rider times"
        )
    result = dict(
        od=od,
        l=l,
        m=m,
        i=i,
        j=j,
        legs=legs,
        nodes=sorted(nodes),
        segments=segments,
        arc_counts=arc_counts,
        deterministic_intervals=intervals,
        time=elapsed,
        operating_cost=operating_cost,
        components=components,
        battery_violation=_maximum(np.asarray(intervals) - battery),
        delivery_violation=0.0,
        joint_reliability=1.0,
        wait=0.0,
        wait_quantile=0.0,
        first_rider=0.0,
        terminal=0.0,
        mean_delivery=0.0,
        quantile_delivery=0.0,
    )
    if service:
        o = int(data["od_origin"][od])
        d = int(data["od_destination"][od])
        mean = float(data["ground_time"][o, i])
        epsilon = float(data["epsilon"])
        if not 0 < epsilon < 1 or mean < 0:
            raise ValueError("Invalid reliability parameter or rider mean")
        t1, t2, t3 = stage_times
        quantile = mean if family == "deterministic" else -mean * math.log(epsilon)
        wait_quantile = max(quantile - t1, 0.0)
        wait = (
            max(mean - t1, 0.0)
            if family == "deterministic"
            else (mean * math.exp(-t1 / mean) if mean else 0.0)
        )
        used = np.asarray(intervals, dtype=float)
        used[waiting_interval] += wait_quantile
        terminal = float(data["terminal_time"][d, j])
        deadline = float(data["delivery_deadline"][od])
        delivery_threshold = deadline - t2 - terminal
        energy_threshold = t1 + battery - intervals[waiting_interval]
        energy_deterministic_ok = all(v <= battery + 1e-12 for v in intervals)
        delivery_deterministic_ok = t1 <= delivery_threshold + 1e-12
        reliability = _cdf(min(delivery_threshold, energy_threshold), mean, family)
        if not (energy_deterministic_ok and delivery_deterministic_ok):
            reliability = 0.0
        operating_cost += (
            float(data["first_cost"][o, i])
            + float(data["terminal_cost"][d, j])
            + float(data.get("hover_cost", 0.0)) * wait
        )
        components.update(
            pickup_handling=float(pickup[i]),
            delivery_handling=float(delivery[j]),
            sync_wait=wait,
        )
        result.update(
            time=elapsed + wait,
            operating_cost=operating_cost,
            battery_violation=_maximum(used - battery),
            delivery_violation=max(0.0, t1 + wait_quantile + t2 + terminal - deadline),
            battery_interval_consumptions=used.tolist(),
            waiting_interval=waiting_interval,
            joint_reliability=reliability,
            wait=wait,
            wait_quantile=wait_quantile,
            first_rider=mean,
            terminal=terminal,
            mean_delivery=t1 + wait + t2 + terminal,
            quantile_delivery=t1 + wait_quantile + t2 + terminal,
            prefix=t1,
            t2=t2,
            t3=t3,
            first_cost=float(data["first_cost"][o, i]),
            terminal_cost=float(data["terminal_cost"][d, j]),
            pickup_share=float(data["pickup_share"][d, j]),
            deadline=deadline,
        )
    else:
        result["battery_interval_consumptions"] = intervals
    errors = {}
    comparisons = {
        "time": result["time"],
        "operating_cost": operating_cost,
        "wait_quantile": result["wait_quantile"],
    }
    if service:
        comparisons.update(
            deadline_quantile_time=result["quantile_delivery"],
            pickup_drone_arrival_hours=result["prefix"],
            expected_synchronization_wait_hours=result["wait"],
            first_rider_hours=result["first_rider"],
        )
    for key, actual in comparisons.items():
        if key in tr:
            errors[key] = abs(float(tr[key]) - actual)
    if "cost" in tr:
        combined = operating_cost + float(data["fleet_cost"]) * result["time"]
        if service:
            combined -= float(data["ground"][od])
        errors["cost"] = abs(float(tr["cost"]) - combined)
    if "duration_components" in tr:
        supplied = tr["duration_components"]
        errors["duration_components"] = max(
            (abs(float(supplied.get(k, 0.0)) - v) for k, v in components.items()),
            default=0.0,
        )
        errors["duration_components_total"] = abs(
            sum(map(float, supplied.values())) - result["time"]
        )
    if "battery_interval_consumptions" in tr:
        supplied = _array(
            tr["battery_interval_consumptions"],
            (len(intervals),),
            "battery_interval_consumptions",
        )
        errors["battery_intervals"] = _maximum(
            abs(supplied - result["battery_interval_consumptions"])
        )
    if "arc_counts" in tr:
        declared = Counter()
        for e, count in tr["arc_counts"]:
            if int(e) != e or int(count) != count or count <= 0:
                raise ValueError("Arc incidences must be positive integer counts")
            declared[int(e)] += int(count)
        if declared != arc_counts:
            raise ValueError("Saved path arc counts differ from physical legs")
    if "segments" in tr:
        actual = tr["segments"]
        if len(actual) != len(segments):
            raise ValueError("Saved segment count differs from physical legs")
        for declared, rebuilt in zip(actual, segments):
            u, v = int(declared["u"]["global_index"]), int(
                declared["v"]["global_index"]
            )
            if (u, v) != (rebuilt["u_index"], rebuilt["v_index"]):
                raise ValueError("Saved segments differ from physical legs")
            for key in ("flight_hours", "swap_hours"):
                errors[key] = max(
                    errors.get(key, 0.0), abs(float(declared[key]) - rebuilt[key])
                )
            if bool(declared.get("swap_at_head", False)) != rebuilt["swap_at_head"]:
                raise ValueError("Saved swap marker differs from physical leg")
    result["coefficient_errors"] = errors
    return result
