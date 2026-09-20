"""Physical route reconstruction and independent coefficient checks."""

import math
from collections import Counter
import numpy as np


def _leg_nodes(c, u, v, k):
    return [int(u), int(v)] if k < 0 else [int(u), c["L"] + c["H"] + int(k), int(v)]


def _node(c, v):
    if v < c["L"]:
        kind, index = "L", v
    elif v < c["L"] + c["H"]:
        kind, index = "H", v - c["L"]
    else:
        kind, index = "E", v - c["L"] - c["H"]
    node = dict(
        global_index=int(v), type=kind, index=int(index), id=f"{kind}{index + 1:03d}"
    )
    coords = c["data"].get({"L": "bases", "H": "hubs", "E": "remote"}[kind])
    if coords is not None:
        node["xy"] = np.asarray(coords[index]).tolist()
    return node


def _physical_trace(c, legs, stages):
    segments, actions, intervals = [], [], []
    consumption = 0.0
    for stage, nodes in zip(stages, legs):
        for u, v in zip(nodes, nodes[1:]):
            ft = float(c["flight"][u, v])
            consumption += ft
            sw = v >= c["L"] + c["H"]
            st = float(c["swap"][v - c["L"] - c["H"]]) if sw else 0.0
            segments.append(
                dict(
                    stage=stage,
                    phase=stage,
                    u=_node(c, u),
                    v=_node(c, v),
                    arc_index=int(u * c["N"] + v),
                    flight_hours=ft,
                    swap_hours=st,
                    swap_at_head=sw,
                )
            )
            actions.append(
                dict(type="flight", stage=stage, u=int(u), v=int(v), duration=ft)
            )
            if sw:
                intervals.append(consumption)
                consumption = 0.0
                actions.append(dict(type="swap", stage=stage, node=int(v), duration=st))
    return segments, actions


def _materialize(c, od, l, m, i, j, swaps):
    empty = od is None
    if empty:
        legs = [_leg_nodes(c, l, m, swaps[0])]
        stages = ["reposition"]
    else:
        legs = [
            _leg_nodes(c, l, c["L"] + i, swaps[0]),
            _leg_nodes(c, c["L"] + i, c["L"] + j, swaps[1]),
            _leg_nodes(c, c["L"] + j, m, swaps[2]),
        ]
        stages = ["before_pickup", "loaded", "return"]
    segments, actions = _physical_trace(c, legs, stages)
    counts = sorted(Counter(s["arc_index"] for s in segments).items())
    flown = sum(s["flight_hours"] for s in segments)
    price = sum(c["arc_prices"].ravel()[idx] * count for idx, count in counts)
    if empty:
        parts = dict(
            empty_flight=flown, empty_swap=sum(s["swap_hours"] for s in segments)
        )
        total = sum(parts.values())
        raw = c["flightcost"] * flown
        adjusted = raw + c["fleet"] * total
        qw = 0.0
        deadline = None
    else:
        o, d = int(c["oi"][od]), int(c["di"][od])
        parts = {}
        for stage, prefix in zip(stages, ["pre_pickup", "loaded", "return"]):
            parts[prefix + "_flight"] = sum(
                s["flight_hours"] for s in segments if s["stage"] == stage
            )
            parts[prefix + "_swap"] = sum(
                s["swap_hours"] for s in segments if s["stage"] == stage
            )
        arrival = parts["pre_pickup_flight"] + parts["pre_pickup_swap"]
        mean = float(c["mean"][o, i])
        w = (
            max(0.0, mean - arrival)
            if c["family"] == "deterministic"
            else (mean * math.exp(-arrival / mean) if mean > 0 else 0.0)
        )
        qt = mean if c["family"] == "deterministic" else -mean * math.log(c["epsilon"])
        qw = max(0.0, qt - arrival)
        parts.update(
            pickup_handling=float(c["pickup"][i]),
            delivery_handling=float(c["delivery"][j]),
            sync_wait=w,
        )
        total = sum(parts.values())
        raw = (
            c["first"][o, i]
            + c["terminal"][d, j]
            + c["flightcost"] * flown
            + c["hovercost"] * w
        )
        adjusted = raw - c["ground"][od] + c["fleet"] * total
        price += c["hub_prices"][i] + c["hub_prices"][j]
        deadline = (
            arrival
            + qw
            + parts["pickup_handling"]
            + parts["loaded_flight"]
            + parts["loaded_swap"]
            + parts["delivery_handling"]
            + c["terminaltime"][d, j]
        )
    # Replay energy actions including the waiting interval and both handling
    # actions; swapping time occupies the drone but consumes no flight budget.
    intervals = []
    used = 0.0
    ordered_actions = []
    for index, (stage, nodes) in enumerate(zip(stages, legs)):
        if not empty and index == 1:
            used += qw + c["pickup"][i]
            ordered_actions.extend(
                [
                    dict(
                        type="hover",
                        stage="pickup",
                        node=c["L"] + i,
                        duration=parts["sync_wait"],
                        energy_quantile=qw,
                    ),
                    dict(
                        type="handling",
                        stage="pickup",
                        node=c["L"] + i,
                        duration=float(c["pickup"][i]),
                    ),
                ]
            )
        if not empty and index == 2:
            used += c["delivery"][j]
            ordered_actions.append(
                dict(
                    type="handling",
                    stage="delivery",
                    node=c["L"] + j,
                    duration=float(c["delivery"][j]),
                )
            )
        for u, v in zip(nodes, nodes[1:]):
            ft = float(c["flight"][u, v])
            used += ft
            ordered_actions.append(
                dict(type="flight", stage=stage, u=u, v=v, duration=ft)
            )
            if v >= c["L"] + c["H"]:
                intervals.append(used)
                used = 0.0
                ordered_actions.append(
                    dict(
                        type="swap",
                        stage=stage,
                        node=v,
                        duration=float(c["swap"][v - c["L"] - c["H"]]),
                    )
                )
    intervals.append(used)
    trace = dict(
        l=int(l),
        m=int(m),
        legs=legs,
        swap_choices=[int(k) for k in swaps],
        segments=segments,
        physical_actions=ordered_actions,
        arc_counts=counts,
        arc_price_charge=float(
            sum(c["arc_prices"].ravel()[idx] * n for idx, n in counts)
        ),
        time=float(total),
        cost=float(adjusted),
        operating_cost=float(raw),
        pricing_cost=float(adjusted + price),
        duration_components={k: float(v) for k, v in parts.items()},
        battery_interval_consumptions=[float(v) for v in intervals],
        wait_quantile=float(qw),
        deadline_quantile_time=None if deadline is None else float(deadline),
        flight_hours=float(flown),
        swap_count=sum(k >= 0 for k in swaps),
        swap_stations=[c["L"] + int(k) for k in swaps if k >= 0],
        station_visit_counts=np.bincount(
            [c["L"] + int(k) for k in swaps if k >= 0], minlength=c["L"] + c["K"]
        ).tolist(),
        complete=True,
    )
    if not empty:
        meal = (
            arrival
            + parts["sync_wait"]
            + parts["pickup_handling"]
            + parts["loaded_flight"]
            + parts["loaded_swap"]
            + parts["delivery_handling"]
        )
        trace.update(
            od=int(od),
            i=int(i),
            j=int(j),
            hub_visits=[int(i), int(j)],
            pickup_drone_arrival_hours=float(arrival),
            expected_synchronization_wait_hours=float(parts["sync_wait"]),
            pickup_synchronization_hours=float(arrival + parts["sync_wait"]),
            first_rider_hours=mean,
            last_rider_hours=float(c["terminaltime"][d, j]),
            home_delivery_hours=float(meal + c["terminaltime"][d, j]),
            delivery_arrival_hours=float(meal),
            acceptance_to_delivery_arrival_hours=float(meal),
            rider_time_family=c["family"],
            rider_time_distribution=c["family"],
            cross_base=bool(l != m),
            pre_dispatch_mean_hours=0.0,
            clock_mode="fetch",
            delivery_deadline=float(c["deadline"][od]),
        )
    return trace


def trace_mission(result, od, l, m):
    if not np.isfinite(result["pricing_cost"][od, l, m]):
        raise ValueError("No feasible mission to trace")
    c = result["_context"]
    i, j, ka, kb, kc = map(int, c["choices"][od, l, m])
    tr = _materialize(c, int(od), int(l), int(m), i, j, [ka, kb, kc])
    for field, target in [
        ("cost", result["cost"][od, l, m]),
        ("time", result["time"][od, l, m]),
        ("pricing_cost", result["pricing_cost"][od, l, m]),
    ]:
        if not math.isclose(tr[field], target, rel_tol=1e-10, abs_tol=1e-8):
            raise AssertionError(f"Pricing traceback mismatch: {field}")
    if max(tr["battery_interval_consumptions"]) > c["battery"] + 1e-8:
        raise AssertionError("Pricing traceback violates battery chance constraint")
    if tr["deadline_quantile_time"] > c["deadline"][od] + 1e-8:
        raise AssertionError("Pricing traceback violates delivery chance constraint")
    return tr


def trace_empty(result, l, m):
    if not np.isfinite(result["empty_pricing_cost"][l, m]):
        raise ValueError("No feasible empty route to trace")
    c = result["_context"]
    tr = _materialize(
        c, None, int(l), int(m), None, None, [int(c["empty_choice"][l, m])]
    )
    for field, target in [
        ("cost", result["empty_cost"][l, m]),
        ("time", result["empty_time"][l, m]),
        ("pricing_cost", result["empty_pricing_cost"][l, m]),
    ]:
        if not math.isclose(tr[field], target, rel_tol=1e-10, abs_tol=1e-8):
            raise AssertionError(f"Empty traceback mismatch: {field}")
    return tr
