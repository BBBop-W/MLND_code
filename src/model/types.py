"""The single input and path contract for the consecutive-delivery model.

Times are hours, demand is orders per hour, and costs use the same hour.
Coordinates, raw observations and provenance belong in the input metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

MODEL_VERSION = "consecutive"
BasePair = tuple[int, int]
Legs = tuple[tuple[int, ...], ...]
ArcCounts = tuple[tuple[int, int, int], ...]


def _array(value, shape, name):
    try:
        result = np.broadcast_to(np.asarray(value, dtype=float), shape).copy()
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must broadcast to {shape}") from exc
    if not np.isfinite(result).all() or np.any(result < 0):
        raise ValueError(f"{name} must be finite and nonnegative")
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class PlanningData:
    n_l: int
    n_h: int
    demand: np.ndarray
    flight_times: np.ndarray
    rider_mean: np.ndarray
    first_cost: np.ndarray
    terminal_cost: np.ndarray
    receipt_mean: np.ndarray
    removal_mean: np.ndarray
    ground_cost: np.ndarray
    deadline: np.ndarray
    base_cost: np.ndarray
    slot_counts: np.ndarray
    slot_cost: np.ndarray
    slot_epsilon: np.ndarray
    link_cost: np.ndarray
    kappa: np.ndarray
    delta: float
    operating_limit: float
    fleet_cost: float
    flight_cost: float
    waiting_cost: float
    max_orders: int = 2
    allowed_links: np.ndarray | None = None
    name: str = "consecutive"
    candidate_routes: tuple[tuple[int, ...], ...] | None = None
    route_mode: str = "direct"
    flow_policy: str = "aggregate"

    def __post_init__(self):
        if self.flow_policy not in ("prefix", "aggregate"):
            raise ValueError("flow_policy must be prefix or aggregate")
        for name in ("n_l", "n_h"):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or int(value) != value or value < 1:
                raise ValueError(f"{name} must be a positive integer")
            object.__setattr__(self, name, int(value))
        if isinstance(self.max_orders, (bool, np.bool_)) or self.max_orders not in (
            1,
            2,
        ):
            raise ValueError("max_orders must be 1 or 2")
        object.__setattr__(self, "max_orders", int(self.max_orders))
        demand = np.asarray(self.demand, dtype=float)
        if demand.ndim != 1 or not demand.size:
            raise ValueError("demand must be a nonempty one-dimensional vector")
        object.__setattr__(self, "demand", _array(demand, demand.shape, "demand"))
        r, h, bases, n = self.n_od, self.n_h, self.n_l, self.n_nodes
        shapes = {
            "rider_mean": (r, h),
            "first_cost": (r, h),
            "terminal_cost": (r, h),
            "receipt_mean": (r, h),
            "removal_mean": (r, h),
            "ground_cost": (r,),
            "deadline": (r,),
            "base_cost": (bases,),
            "slot_epsilon": (h,),
            "link_cost": (n, n),
            "kappa": (r, r),
        }
        for name, shape in shapes.items():
            object.__setattr__(self, name, _array(getattr(self, name), shape, name))
        if np.any(self.kappa > 1):
            raise ValueError("kappa must lie in [0, 1]")
        if np.any((self.slot_epsilon <= 0) | (self.slot_epsilon >= 1)):
            raise ValueError("slot_epsilon must lie strictly between zero and one")
        if np.any(self.removal_mean > self.receipt_mean + 1e-12):
            raise ValueError("Meal removal cannot follow final receipt")
        counts = np.asarray(self.slot_counts, dtype=float)
        if counts.ndim not in (1, 2) or counts.shape[-1] == 0:
            raise ValueError("slot_counts must contain at least one capacity level")
        counts = _array(counts, (h, counts.shape[-1]), "slot_counts")
        if np.any(counts != np.rint(counts)):
            raise ValueError("slot_counts must contain integer compartment counts")
        counts = counts.astype(np.int64)
        counts.setflags(write=False)
        object.__setattr__(self, "slot_counts", counts)
        object.__setattr__(
            self, "slot_cost", _array(self.slot_cost, counts.shape, "slot_cost")
        )
        for name in (
            "delta",
            "operating_limit",
            "fleet_cost",
            "flight_cost",
            "waiting_cost",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
            if name in ("operating_limit", "fleet_cost") and value == 0:
                raise ValueError(f"{name} must be positive")
            object.__setattr__(self, name, value)
        flight = np.asarray(self.flight_times, dtype=float).copy()
        if flight.shape != (n, n) or np.isnan(flight).any() or np.any(flight < 0):
            raise ValueError(
                "flight_times must be nonnegative NxN; inf marks absent links"
            )
        if np.any(np.diag(flight) != 0):
            raise ValueError("flight_times must have a zero diagonal")
        if self.allowed_links is None:
            allowed = np.isfinite(flight)
        else:
            raw = np.asarray(self.allowed_links)
            if raw.shape != (n, n) or np.any((raw != 0) & (raw != 1)):
                raise ValueError("allowed_links must be a binary NxN array")
            allowed = raw.astype(bool)
            if np.any(allowed & ~np.isfinite(flight)):
                raise ValueError("Allowed links must have finite flight times")
        np.fill_diagonal(allowed, False)
        flight.setflags(write=False)
        allowed.setflags(write=False)
        object.__setattr__(self, "flight_times", flight)
        object.__setattr__(self, "allowed_links", allowed)
        if self.route_mode not in ("direct", "simple"):
            raise ValueError("route_mode must be direct or simple")
        if self.candidate_routes is not None:
            routes = []
            for raw_route in self.candidate_routes:
                route = tuple(raw_route)
                if not route or any(
                    isinstance(v, (bool, np.bool_)) or int(v) != v or not 0 <= v < n
                    for v in route
                ):
                    raise ValueError("Candidate route nodes must be valid node indices")
                route = tuple(int(v) for v in route)
                if len(set(route)) != len(route):
                    raise ValueError(
                        "Each candidate flight leg must be an elementary route"
                    )
                if any(not allowed[u, v] for u, v in zip(route, route[1:])):
                    raise ValueError("A candidate route uses an unavailable link")
                routes.append(route)
            object.__setattr__(self, "candidate_routes", tuple(dict.fromkeys(routes)))

    @property
    def n_od(self):
        return len(self.demand)

    @property
    def n_nodes(self):
        return self.n_l + self.n_h

    @property
    def base_pairs(self):
        return tuple(
            (origin, destination)
            for origin in range(self.n_l)
            for destination in range(self.n_l)
        )

    @property
    def effective_delta(self):
        return self.delta if self.max_orders == 2 else 0.0


@dataclass(frozen=True)
class InitialSegment:
    id: int
    b: BasePair
    first_od: int
    pickup: int
    delivery: int
    legs: Legs
    t0: float
    t1: float


@dataclass(frozen=True)
class ServicePath:
    id: int
    segment_id: int
    b: BasePair
    first_od: int
    second_od: int | None
    mode: str
    pickups: tuple[int, ...]
    deliveries: tuple[int, ...]
    legs: Legs
    served_counts: tuple[tuple[int, int], ...]
    arc_counts: ArcCounts
    flight_time: float
    time: float
    wait: float
    slot_times: tuple[tuple[int, float], ...]
    pickup_times: tuple[float, ...]
    delivery_times: tuple[float, ...]
    cost: float


@dataclass(frozen=True)
class EmptyPath:
    id: int
    b: BasePair
    legs: Legs
    arc_counts: ArcCounts
    flight_time: float
    time: float
    cost: float


@dataclass(frozen=True)
class PathCollection:
    initial_segments: tuple[InitialSegment, ...]
    service_paths: tuple[ServicePath, ...]
    empty_paths: tuple[EmptyPath, ...]
    complete: bool = True
    scope: str = "direct"
