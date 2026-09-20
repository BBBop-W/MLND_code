"""Complete physical-path enumeration for small-instance reference solves."""

import itertools
import math
import time
from src.solvers.pricing.inputs import _validated
from src.solvers.pricing.routes import _materialize

_TOL = 1e-10


class EnumerationStopped(RuntimeError):
    """The complete path set could not be constructed within the limits."""


def enumerate_all_columns(
    data,
    facility=None,
    *,
    time_limit=None,
    column_limit=None,
    stop_requested=None,
    progress=None,
):
    """Exhaustive physical-path reference with exact feasibility pruning.

    Returns ``{'service': list, 'empty': list}``; no dominance or pricing
    optimization is used. Every direct/single-swap leg remains available.
    Infeasible prefixes are discarded before constructing their full trace;
    nonnegative flight, waiting, and handling times cannot repair them later.
    This implementation is deliberately separate from the pricing kernel and
    state graph.
    """
    started = time.perf_counter()
    c = _validated(data, facility)
    service = []
    empty = []
    L, H, K = c["L"], c["H"], c["K"]
    options = [-1] + [k for k in range(K) if c["facility"][L + H + k]]

    attempts = 0

    def check_limits():
        nonlocal attempts
        attempts += 1
        if column_limit is not None and len(service) + len(empty) >= column_limit:
            raise EnumerationStopped("column_limit")
        if attempts % 256 != 1:
            return
        if stop_requested is not None and stop_requested():
            raise EnumerationStopped("user_stop")
        if time_limit is not None and time.perf_counter() - started >= time_limit:
            raise EnumerationStopped("time_limit")
        if progress and attempts % 65536 == 1:
            progress(
                dict(
                    stage="path_enumeration",
                    columns=len(service) + len(empty),
                    elapsed=time.perf_counter() - started,
                )
            )

    battery = c["battery"] + _TOL
    bases = [base for base in range(L) if c["facility"][base]]
    hubs = [i for i in range(H) if c["facility"][L + i]]
    leg_cache = {}

    def legs(u, v):
        """(swap choice, first flight, last flight, total flight, swap time)."""
        key = (u, v)
        if key not in leg_cache:
            found = []
            for k in options:
                check_limits()
                if k < 0:
                    first, last, swap = float(c["flight"][u, v]), 0.0, 0.0
                else:
                    w = L + H + k
                    first, last = float(c["flight"][u, w]), float(c["flight"][w, v])
                    swap = float(c["swap"][k])
                if (
                    math.isfinite(first)
                    and math.isfinite(last)
                    and first <= battery
                    and last <= battery
                ):
                    found.append((k, first, last, first + last, swap))
            leg_cache[key] = found
        return leg_cache[key]

    def advance(used, leg):
        # Match the action-by-action addition order in _materialize, including
        # the reset at a swap. The completed interval cannot be repaired by a
        # later reset; neither can an already excessive unfinished interval.
        used += leg[1]
        if used > battery:
            return None
        return leg[2] if leg[0] >= 0 else used

    check_limits()
    for base, m in itertools.product(bases, repeat=2):
        if base != m:
            for leg in legs(base, m):
                check_limits()
                empty.append(_materialize(c, None, base, m, None, None, [leg[0]]))
        for od in range(c["D"]):
            o, d = int(c["oi"][od]), int(c["di"][od])
            deadline = c["deadline"][od] + _TOL
            for i in hubs:
                check_limits()
                mean = float(c["mean"][o, i])
                quantile = (
                    mean
                    if c["family"] == "deterministic"
                    else -mean * math.log(c["epsilon"])
                )
                pickup_prefixes = []
                for first_leg in legs(base, L + i):
                    arrival = first_leg[3] + first_leg[4]
                    wait = max(0.0, quantile - arrival)
                    used = first_leg[2] if first_leg[0] >= 0 else first_leg[1]
                    used += wait + c["pickup"][i]
                    pickup_time = arrival + wait + float(c["pickup"][i])
                    if used <= battery and pickup_time <= deadline:
                        pickup_prefixes.append((first_leg[0], used, pickup_time))
                if not pickup_prefixes:
                    continue
                for j in hubs:
                    check_limits()
                    if i == j:
                        continue
                    loaded_legs = legs(L + i, L + j)
                    return_legs = legs(L + j, m)
                    if not loaded_legs or not return_legs:
                        continue
                    # Keep the old enumeration order: i, j, then the three
                    # swap choices. Prefix reuse changes neither columns nor
                    # their order and never compares costs or dominance.
                    for ka, pickup_used, pickup_time in pickup_prefixes:
                        for loaded_leg in loaded_legs:
                            check_limits()
                            used = advance(pickup_used, loaded_leg)
                            if used is None:
                                continue
                            delivered = (
                                pickup_time
                                + loaded_leg[3]
                                + loaded_leg[4]
                                + float(c["delivery"][j])
                                + c["terminaltime"][d, j]
                            )
                            if delivered > deadline:
                                continue
                            used += c["delivery"][j]
                            if used > battery:
                                continue
                            for return_leg in return_legs:
                                check_limits()
                                if advance(used, return_leg) is not None:
                                    service.append(
                                        _materialize(
                                            c,
                                            od,
                                            base,
                                            m,
                                            i,
                                            j,
                                            [ka, loaded_leg[0], return_leg[0]],
                                        )
                                    )
    return dict(service=service, empty=empty)
