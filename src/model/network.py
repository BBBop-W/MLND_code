"""Physical graph, facility charges, and cabinet capacity levels."""

from __future__ import annotations

import numpy as np


def levels(data):
    H = data["n_h"]
    sizes = np.asarray(
        data.get("capacity_levels", np.arange(1, int(data.get("max_cabinets", 5)) + 1)),
        float,
    )
    unit = np.broadcast_to(np.asarray(data.get("unit_capacity", 38.0), float), (H,))
    capacity = np.asarray(data.get("level_capacity", unit[:, None] * sizes), float)
    cost = np.asarray(
        data.get(
            "level_cost",
            np.broadcast_to(data.get("hub_cost", 3.0) * sizes, capacity.shape),
        ),
        float,
    )
    if (
        sizes.ndim != 1
        or len(sizes) == 0
        or np.any(sizes <= 0)
        or capacity.shape != (H, len(sizes))
        or cost.shape != capacity.shape
        or not np.isfinite(capacity).all()
        or not np.isfinite(cost).all()
        or np.any(capacity <= 0)
        or np.any(cost < 0)
    ):
        raise ValueError("Invalid cabinet capacity levels")
    return sizes, capacity, cost


def network(data):
    L, H, K = (data[k] for k in ("n_l", "n_h", "n_k"))
    xy = np.vstack((data["bases"], data["hubs"], data["remote"]))
    N = L + H + K
    distance = np.linalg.norm(xy[:, None, :] - xy[None, :, :], axis=2)
    uv = np.argwhere(~np.eye(N, dtype=bool))
    flat = uv[:, 0] * N + uv[:, 1]
    lookup = np.full(N * N, -1, dtype=int)
    lookup[flat] = np.arange(len(flat))
    _, capacities, _ = levels(data)
    # Removing redundant base-level empty cycles preserves an optimal plan.
    # This bound requires no availability cap or dispatch-request-rate cap.
    upper = np.full(len(flat), (L + 2) * float(np.sum(data["demand"])))
    for e, (u, v) in enumerate(uv):
        for node in (u, v):
            if L <= node < L + H:
                upper[e] = min(upper[e], capacities[node - L].max())
    lengths = distance[uv[:, 0], uv[:, 1]]
    physical = lengths > 1e-9
    hh = (uv[:, 0] >= L) & (uv[:, 0] < L + H) & (uv[:, 1] >= L) & (uv[:, 1] < L + H)
    fees = np.where(
        hh,
        data.get("airline_hh_cost_per_km_hour", 0.02),
        data.get("airline_connection_cost_per_km_hour", 0.05),
    )
    fees = fees * lengths * data.get("airline_cost_multiplier", 1.0)
    if "airline_fees" in data:
        supplied = np.asarray(data["airline_fees"], float)
        fees = supplied.ravel()[flat] if supplied.shape == (N, N) else supplied.copy()
    if fees.shape != (len(flat),) or not np.isfinite(fees).all() or np.any(fees < 0):
        raise ValueError("Flight-link fees must be finite and nonnegative")
    return dict(
        N=N,
        xy=xy,
        uv=uv,
        flat=flat,
        lookup=lookup,
        upper=upper,
        distance=lengths,
        physical=physical,
        fees=fees,
        count=len(flat),
    )


def vector(data, values, net=None):
    net = network(data) if net is None else net
    a = np.asarray(values, float)
    if a.shape == (net["N"], net["N"]):
        a = a.ravel()[net["flat"]]
    if a.shape != (net["count"],) or not np.isfinite(a).all():
        raise ValueError("Invalid flight-link vector")
    return a


def price_matrix(net, prices):
    result = np.zeros((net["N"], net["N"]))
    result.ravel()[net["flat"]] = prices
    return result


def incidence(net, trace):
    result = []
    for flat, count in trace["arc_counts"]:
        e = int(net["lookup"][int(flat)])
        if e < 0 or count <= 0:
            raise ValueError("Invalid physical link in path")
        result.append((e, int(count)))
    return result


def route_key(trace):
    # Columns with identical coefficients are interchangeable in the SOCP.
    return (
        trace.get("i"),
        trace.get("j"),
        tuple(map(tuple, trace["arc_counts"])),
        round(float(trace["time"]), 12),
        round(float(trace["cost"]), 12),
    )


def facility_costs(data):
    """Hourly costs paid once per open location, excluding cabinet units.

    Optional ``*_site_cost`` entries are amortized setup costs in CNY per
    operating hour, supplied as scalars or vectors over candidate locations.
    They do not depend on the number of cabinets installed at a location.
    """
    L, H, K = (data[k] for k in ("n_l", "n_h", "n_k"))

    def values(key, count, default=0.0):
        value = np.asarray(data.get(key, default), float)
        if not np.isfinite(value).all() or np.any(value < 0):
            raise ValueError(f"{key} must be finite and nonnegative")
        return np.broadcast_to(value, (count,))

    return np.r_[
        values("base_cost", L) + values("base_site_cost", L),
        values("hub_site_cost", H),
        values("swap_cost", K) + values("swap_site_cost", K),
    ]
