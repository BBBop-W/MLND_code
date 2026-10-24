"""Exact enumeration of the selected finite physical-route universe."""

from dataclasses import replace
from itertools import product
import math
import time

import numpy as np

from .resources import (
    PathInfeasible,
    build_initial_segment,
    build_service_path,
    build_empty_path,
    first_wait,
    second_wait,
)
from .types import PathCollection, PlanningData


class EnumerationStopped(RuntimeError):
    def __init__(self, reason, *, columns=0, initial_segments=0):
        super().__init__(f"Path enumeration stopped: {reason}")
        self.reason = reason
        self.columns = columns
        self.initial_segments = initial_segments


class RouteCatalogue:
    """Lazily cache all routes between endpoints; never silently truncate."""

    def __init__(self, data, check=None):
        self.data = data
        self.check = check or (lambda: None)
        self.cache = {}
        self.adjacent = tuple(
            tuple(v for v in range(data.n_nodes) if data.allowed_links[u, v])
            for u in range(data.n_nodes)
        )
        if data.candidate_routes is not None:
            groups = {}
            for route in data.candidate_routes:
                self.check()
                groups.setdefault((route[0], route[-1]), []).append(route)
            self.cache.update({key: tuple(routes) for key, routes in groups.items()})

    def routes(self, origin, destination):
        self.check()
        if origin == destination:
            supplied = self.cache.get((origin, destination), ())
            return tuple(dict.fromkeys(((origin,), *supplied)))
        key = (origin, destination)
        if key in self.cache:
            return self.cache[key]
        if self.data.candidate_routes is not None:
            answer = ()
        elif self.data.route_mode == "direct":
            answer = ((origin, destination),) if self.data.allowed_links[key] else ()
        else:
            routes = []

            def visit(route, elapsed):
                self.check()
                for nxt in self.adjacent[route[-1]]:
                    self.check()
                    if nxt in route:
                        continue
                    travel = elapsed + self.data.flight_times[route[-1], nxt]
                    # A leg longer than the full mission limit cannot occur in
                    # any feasible path because every remaining term is >= 0.
                    if travel > self.data.operating_limit + 1e-12:
                        continue
                    following = (*route, nxt)
                    if nxt == destination:
                        routes.append(following)
                    else:
                        visit(following, travel)

            visit((origin,), 0.0)
            answer = tuple(routes)
        self.cache[key] = answer
        return answer


class DirectContinuationFilter:
    """Necessary physical checks before constructing direct-route columns.

    The cache contains all second-order locker triples in their original order.
    Every survivor still passes through the authoritative scalar path builder.
    No first segment is filtered: even an empty continuation branch remains in
    the solver's compulsory branch equations.
    """

    def __init__(self, data, check):
        self.data, self.check, self.cache = data, check, {}
        self.s, self.i, self.j = np.indices((data.n_od, data.n_h, data.n_h))
        self.s, self.i, self.j = (v.ravel() for v in (self.s, self.i, self.j))

    @staticmethod
    def _slack(value):
        # Scalar math.fsum and vector additions can round differently. Keeping
        # a few extra candidates near a boundary preserves exact membership.
        return 1e-12 + 64 * np.finfo(float).eps * np.maximum(1.0, np.abs(value))

    def candidates(self, segment):
        self.check()
        data = self.data
        key = segment.delivery, segment.b[1]
        if key not in self.cache:
            transfer = data.flight_times[data.n_l + segment.delivery, data.n_l :]
            loaded = data.flight_times[data.n_l :, data.n_l :]
            back = data.flight_times[data.n_l :, segment.b[1]]
            waits = np.zeros((data.n_od, data.n_h))
            for s in range(data.n_od):
                self.check()
                for i in range(data.n_h):
                    waits[s, i] = second_wait(
                        float(data.rider_mean[s, i]),
                        float(transfer[i]),
                        data.effective_delta,
                    )
            second_waits = waits[self.s, self.i]
            downstream = (
                transfer[self.i] + loaded[self.i, self.j] + back[self.j] + second_waits
            )
            receipt = (
                transfer[self.i]
                + data.effective_delta / 2
                + second_waits
                + loaded[self.i, self.j]
                + data.receipt_mean[self.s, self.j]
            )
            allowed_transfer = data.allowed_links[
                data.n_l + segment.delivery, data.n_l + self.i
            ] | (self.i == segment.delivery)
            feasible = (
                (self.i != self.j)
                & allowed_transfer
                & data.allowed_links[data.n_l + self.i, data.n_l + self.j]
                & data.allowed_links[data.n_l + self.j, segment.b[1]]
                & np.isfinite(downstream)
                & (
                    receipt
                    <= data.deadline[self.s] + self._slack(data.deadline[self.s])
                )
            )
            indices = np.flatnonzero(feasible)
            self.cache[key] = indices, downstream[indices]
        indices, downstream = self.cache[key]
        initial = (
            segment.t0
            + segment.t1
            + first_wait(
                float(data.rider_mean[segment.first_od, segment.pickup]), segment.t0
            )
        )
        keep = (data.kappa[segment.first_od, self.s[indices]] > 0) & (
            initial + downstream
            <= data.operating_limit + self._slack(data.operating_limit)
        )
        for index in indices[keep]:
            self.check()
            s, i, j = int(self.s[index]), int(self.i[index]), int(self.j[index])
            start, pickup, delivery = (
                segment.delivery + data.n_l,
                i + data.n_l,
                j + data.n_l,
            )
            transfer = (start,) if start == pickup else (start, pickup)
            yield s, i, j, (transfer, (pickup, delivery), (delivery, segment.b[1]))


def enumerate_paths(data, column_limit=None, time_limit=None, stop_requested=None):
    """Return complete candidate paths or raise EnumerationStopped.

    First legs remain present even when a required continuation family is empty.
    An absent physical branch must therefore still be imposed by the solver.
    """
    if not isinstance(data, PlanningData):
        raise TypeError("Expected PlanningData")
    if column_limit is not None and (
        isinstance(column_limit, bool)
        or int(column_limit) != column_limit
        or column_limit < 0
    ):
        raise ValueError("column_limit must be a nonnegative integer or None")
    if time_limit is not None and (not math.isfinite(time_limit) or time_limit < 0):
        raise ValueError("time_limit must be finite and nonnegative or None")
    started = time.perf_counter()
    segments, service, empty = [], [], []

    def stop(reason):
        raise EnumerationStopped(
            reason, columns=len(service) + len(empty), initial_segments=len(segments)
        )

    def check():
        if stop_requested is not None and stop_requested():
            stop("user_stop")
        if time_limit is not None and time.perf_counter() - started >= time_limit:
            stop("time_limit")

    def append(target, path):
        check()
        if column_limit is not None and len(service) + len(empty) >= column_limit:
            stop("column_limit")
        target.append(replace(path, id=len(target)))

    routes = RouteCatalogue(data, check)
    direct_filter = (
        DirectContinuationFilter(data, check)
        if data.route_mode == "direct" and data.candidate_routes is None
        else None
    )
    for b in data.base_pairs:
        check()
        origin, destination = b
        if origin != destination:
            for route in routes.routes(origin, destination):
                check()
                try:
                    append(empty, build_empty_path(data, b, route))
                except PathInfeasible:
                    pass
        for r, i, j in product(range(data.n_od), range(data.n_h), range(data.n_h)):
            check()
            if i == j:
                continue
            for first, loaded in product(
                routes.routes(origin, data.n_l + i),
                routes.routes(data.n_l + i, data.n_l + j),
            ):
                check()
                try:
                    segment = build_initial_segment(
                        data, b, r, i, j, (first, loaded), id=len(segments)
                    )
                except PathInfeasible:
                    continue
                segments.append(segment)
                for back in routes.routes(data.n_l + j, destination):
                    check()
                    try:
                        append(
                            service,
                            build_service_path(data, segment, (*segment.legs, back)),
                        )
                    except PathInfeasible:
                        pass
                if data.effective_delta == 0:
                    continue
                if direct_filter is not None:
                    for s, i2, j2, continuation in direct_filter.candidates(segment):
                        try:
                            append(
                                service,
                                build_service_path(
                                    data,
                                    segment,
                                    segment.legs + continuation,
                                    second_od=s,
                                    second_pickup=i2,
                                    second_delivery=j2,
                                ),
                            )
                        except PathInfeasible:
                            pass
                    continue
                for s, i2, j2 in product(
                    range(data.n_od), range(data.n_h), range(data.n_h)
                ):
                    check()
                    if data.kappa[r, s] == 0 or i2 == j2:
                        continue
                    families = (
                        routes.routes(data.n_l + j, data.n_l + i2),
                        routes.routes(data.n_l + i2, data.n_l + j2),
                        routes.routes(data.n_l + j2, destination),
                    )
                    for continuation in product(*families):
                        check()
                        try:
                            path = build_service_path(
                                data,
                                segment,
                                segment.legs + continuation,
                                second_od=s,
                                second_pickup=i2,
                                second_delivery=j2,
                            )
                            append(service, path)
                        except PathInfeasible:
                            pass
    scope = "supplied_routes" if data.candidate_routes is not None else data.route_mode
    return PathCollection(tuple(segments), tuple(service), tuple(empty), True, scope)
