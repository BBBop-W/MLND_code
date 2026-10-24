"""Fixed path coefficients for exponential courier times and uniform V.

All builders use the same physical checks.  A second request is measured
backwards from the first deposit, V ~ U[0, delta], not from first dispatch.
"""

from collections import Counter, defaultdict
import math

from .types import EmptyPath, InitialSegment, ServicePath


class PathInfeasible(ValueError):
    """A physically valid candidate cannot meet the model's mean limits."""


def first_wait(mean, travel):
    if mean < 0 or travel < 0:
        raise ValueError("Courier mean and flight time must be nonnegative")
    return mean * math.exp(-travel / mean) if mean > 0 else 0.0


def second_wait(mean, travel, delta):
    """E[(R - travel - V)+], including the continuous delta=0 limit."""
    if mean < 0 or travel < 0 or delta < 0:
        raise ValueError("Courier mean, flight time and window must be nonnegative")
    if mean == 0:
        return 0.0
    if delta == 0:
        return first_wait(mean, travel)
    return first_wait(mean, travel) * (-math.expm1(-delta / mean)) / (delta / mean)


def route_time(data, route, origin=None, destination=None):
    """Validate a route against the selected finite candidate-route universe."""
    route = tuple(route)
    if not route or any(
        isinstance(v, bool) or int(v) != v or not 0 <= v < data.n_nodes for v in route
    ):
        raise ValueError("Invalid route nodes")
    if origin is not None and route[0] != origin:
        raise ValueError("Route has an incorrect origin")
    if destination is not None and route[-1] != destination:
        raise ValueError("Route has an incorrect destination")
    if any(not data.allowed_links[u, v] for u, v in zip(route, route[1:])):
        raise ValueError("Route uses an unavailable directed link")
    if len(set(route)) != len(route):
        raise ValueError("Each flight leg must be an elementary route")
    if len(route) == 1:
        return 0.0
    if data.candidate_routes is not None:
        if route not in data.candidate_routes:
            raise ValueError("Route is outside the supplied candidate set")
    elif data.route_mode == "direct" and len(route) != 2:
        raise ValueError("A direct-flight candidate must have one arc")
    elif data.route_mode == "simple" and len(set(route)) != len(route):
        raise ValueError("An elementary route cannot repeat a node")
    return math.fsum(float(data.flight_times[u, v]) for u, v in zip(route, route[1:]))


def arc_counts(legs):
    counts = Counter((u, v) for leg in legs for u, v in zip(leg, leg[1:]))
    return tuple((u, v, count) for (u, v), count in sorted(counts.items()))


def _indices(data, b, od, pickup, delivery):
    if tuple(b) not in data.base_pairs:
        raise ValueError("Invalid departure-return base pair")
    if not 0 <= od < data.n_od or int(od) != od:
        raise ValueError("Invalid OD index")
    if any(not 0 <= i < data.n_h or int(i) != i for i in (pickup, delivery)):
        raise ValueError("Invalid locker index")
    if pickup == delivery:
        raise PathInfeasible("Each order needs distinct pickup and delivery lockers")


def build_initial_segment(data, b, first_od, pickup, delivery, legs, *, id=-1):
    """Build a feasible first service leg even if no complete path exists."""
    b = tuple(b)
    _indices(data, b, first_od, pickup, delivery)
    legs = tuple(tuple(route) for route in legs)
    if len(legs) != 2:
        raise ValueError("A first leg needs base-to-pickup and loaded routes")
    t0 = route_time(data, legs[0], b[0], data.n_l + pickup)
    t1 = route_time(data, legs[1], data.n_l + pickup, data.n_l + delivery)
    if t1 + 1e-12 < data.effective_delta:
        raise PathInfeasible("The continuation window exceeds the first loaded flight")
    q1 = t0 + first_wait(float(data.rider_mean[first_od, pickup]), t0)
    final = q1 + t1 + float(data.receipt_mean[first_od, delivery])
    if final > data.deadline[first_od] + 1e-12:
        raise PathInfeasible("The first order violates its mean receipt-time limit")
    return InitialSegment(
        id, b, int(first_od), int(pickup), int(delivery), legs, t0, t1
    )


def build_service_path(
    data,
    segment,
    legs,
    *,
    second_od=None,
    second_pickup=None,
    second_delivery=None,
    id=-1,
):
    """Construct a complete path, recomputing all physical coefficients.

    ``delivery_times`` measures own-request-to-final-receipt time; reservation
    release uses removal_mean separately.  No handling or recharge time is added.
    """
    legs = tuple(tuple(route) for route in legs)
    expected = 3 if second_od is None else 5
    if len(legs) != expected:
        raise ValueError(f"A complete service path needs {expected} legs")
    if legs[:2] != segment.legs:
        raise ValueError("Complete path must retain its shared first-leg routes")
    initial = build_initial_segment(
        data,
        segment.b,
        segment.first_od,
        segment.pickup,
        segment.delivery,
        legs[:2],
        id=segment.id,
    )
    r, i, j = initial.first_od, initial.pickup, initial.delivery
    q1 = initial.t0 + first_wait(float(data.rider_mean[r, i]), initial.t0)
    wait = q1 - initial.t0
    pickups, deliveries, ods = (i,), (j,), (r,)
    pickup_times = (q1,)
    loaded_times = (initial.t1,)
    durations = [initial.t0, initial.t1]
    if second_od is None:
        if second_pickup is not None or second_delivery is not None:
            raise ValueError("Single-order paths cannot specify second lockers")
        durations.append(route_time(data, legs[2], data.n_l + j, initial.b[1]))
    else:
        if second_pickup is None or second_delivery is None:
            raise ValueError(
                "Two-order paths require second pickup and delivery lockers"
            )
        _indices(data, initial.b, second_od, second_pickup, second_delivery)
        if data.effective_delta <= 0:
            raise PathInfeasible(
                "Two-order paths require a positive continuation window"
            )
        if data.kappa[r, second_od] <= 0:
            raise PathInfeasible("The OD transition is spatially excluded")
        i2, j2, r2 = int(second_pickup), int(second_delivery), int(second_od)
        transfer = route_time(data, legs[2], data.n_l + j, data.n_l + i2)
        loaded = route_time(data, legs[3], data.n_l + i2, data.n_l + j2)
        back = route_time(data, legs[4], data.n_l + j2, initial.b[1])
        w2 = second_wait(float(data.rider_mean[r2, i2]), transfer, data.effective_delta)
        q2 = transfer + data.effective_delta / 2 + w2
        durations.extend((transfer, loaded, back))
        wait += w2
        pickup_times = (q1, q2)
        loaded_times = (initial.t1, loaded)
        pickups, deliveries, ods = (i, i2), (j, j2), (r, r2)
    flight = math.fsum(durations)
    operating = flight + wait
    if operating > data.operating_limit + 1e-12:
        raise PathInfeasible("The complete path violates its mean operating-time limit")
    receipt = tuple(
        q + t + float(data.receipt_mean[rn, jn])
        for rn, jn, q, t in zip(ods, deliveries, pickup_times, loaded_times)
    )
    if any(value > data.deadline[rn] + 1e-12 for rn, value in zip(ods, receipt)):
        raise PathInfeasible("An order violates its own mean receipt-time limit")
    slots = defaultdict(float)
    for rn, i_n, j_n, q, t in zip(ods, pickups, deliveries, pickup_times, loaded_times):
        slots[i_n] += q
        slots[j_n] += q + t + float(data.removal_mean[rn, j_n])
    courier = math.fsum(
        float(data.first_cost[rn, i_n] + data.terminal_cost[rn, j_n])
        for rn, i_n, j_n in zip(ods, pickups, deliveries)
    )
    cost = courier + data.flight_cost * flight + data.waiting_cost * wait
    return ServicePath(
        id=id,
        segment_id=initial.id,
        b=initial.b,
        first_od=r,
        second_od=second_od,
        mode="single" if second_od is None else "double",
        pickups=pickups,
        deliveries=deliveries,
        legs=legs,
        served_counts=tuple(sorted(Counter(ods).items())),
        arc_counts=arc_counts(legs),
        flight_time=flight,
        time=operating,
        wait=wait,
        slot_times=tuple(sorted(slots.items())),
        pickup_times=pickup_times,
        delivery_times=receipt,
        cost=cost,
    )


def build_empty_path(data, b, route, *, id=-1):
    b, route = tuple(b), tuple(route)
    if b not in data.base_pairs or b[0] == b[1]:
        raise ValueError("Empty activities must connect distinct bases")
    flight = route_time(data, route, b[0], b[1])
    if flight > data.operating_limit + 1e-12:
        raise PathInfeasible("Empty flight violates the operating-time limit")
    legs = (route,)
    return EmptyPath(
        id, b, legs, arc_counts(legs), flight, flight, data.flight_cost * flight
    )


def rebuild_path(data, path):
    """Recompute a saved/supplied path without trusting its cached resources."""
    if isinstance(path, EmptyPath):
        if len(path.legs) != 1:
            raise ValueError("An empty activity needs one base-to-base route")
        return build_empty_path(data, path.b, path.legs[0], id=path.id)
    if not isinstance(path, ServicePath):
        raise TypeError("Expected ServicePath or EmptyPath")
    if path.mode not in ("single", "double"):
        raise ValueError("Unknown service path mode")
    count = 1 if path.second_od is None else 2
    if (
        path.mode != ("single" if count == 1 else "double")
        or len(path.pickups) != count
        or len(path.deliveries) != count
    ):
        raise ValueError("Order occurrences and service mode disagree")
    segment = build_initial_segment(
        data,
        path.b,
        path.first_od,
        path.pickups[0],
        path.deliveries[0],
        path.legs[:2],
        id=path.segment_id,
    )
    return build_service_path(
        data,
        segment,
        path.legs,
        id=path.id,
        second_od=path.second_od,
        second_pickup=path.pickups[1] if count == 2 else None,
        second_delivery=path.deliveries[1] if count == 2 else None,
    )


def path_feasible(path, data):
    try:
        rebuild_path(data, path)
        return True
    except (ValueError, TypeError):
        return False


def resource_coefficients(path, data):
    """Return a rebuilt path, whose time/cost/slot fields are fixed coefficients."""
    return rebuild_path(data, path)


build_path = build_service_path
