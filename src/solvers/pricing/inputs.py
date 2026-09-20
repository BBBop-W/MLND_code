"""Validated arrays and optional resource prices supplied to path pricing."""

from __future__ import annotations
import math
import numpy as np


def _vector(value, size, name):
    a = np.asarray(value, dtype=float)
    if a.ndim == 0:
        a = np.full(size, float(a))
    if a.shape != (size,) or not np.isfinite(a).all() or np.any(a < 0):
        raise ValueError(f"{name} must be nonnegative scalar or length {size}")
    return a


def _validated(data, facility=None, hub_prices=None, arc_prices=None, groups=None):
    L, H, K, D = (int(data[k]) for k in ("n_l", "n_h", "n_k", "n_od"))
    N = L + H + K
    ft = np.asarray(data["flight_times"], dtype=float)
    if ft.shape != (N, N) or np.isnan(ft).any() or np.any(ft < 0):
        raise ValueError(
            "flight_times must be a nonnegative NxN matrix; infinity means absent arc"
        )
    battery = float(data["battery_time"])
    eps = float(data["epsilon"])
    if not math.isfinite(battery) or battery <= 0 or not 0 < eps < 1:
        raise ValueError("battery_time must be positive and epsilon must lie in (0,1)")
    family = data.get("rider_time_family", "exponential")
    if family not in ("exponential", "deterministic"):
        raise ValueError(
            "Path pricing supports exponential or deterministic rider times"
        )
    fac = np.ones(N, dtype=np.int64) if facility is None else np.asarray(facility)
    if fac.shape != (N,) or np.any((fac != 0) & (fac != 1)):
        raise ValueError("facility must be a binary vector in L,H,E order")
    hp = np.zeros(H) if hub_prices is None else np.asarray(hub_prices, dtype=float)
    ap = np.zeros((N, N)) if arc_prices is None else np.asarray(arc_prices, dtype=float)
    if hp.shape != (H,) or ap.shape != (N, N):
        raise ValueError("dual price shapes must be H and NxN")
    if (
        not np.isfinite(hp).all()
        or not np.isfinite(ap).all()
        or hp.min(initial=0) < -1e-10
        or ap.min(initial=0) < -1e-10
    ):
        raise ValueError("resource prices must be finite and nonnegative")
    hp, ap = np.maximum(hp, 0), np.maximum(ap, 0)
    groundtime = np.asarray(data["ground_time"], dtype=float)
    first = np.asarray(data["first_cost"], dtype=float)
    terminal = np.asarray(data["terminal_cost"], dtype=float)
    terminaltime = np.asarray(data["terminal_time"], dtype=float)
    oi, di = (
        np.asarray(data[k], dtype=np.int64) for k in ("od_origin", "od_destination")
    )
    ground = np.asarray(data["ground"], dtype=float)
    deadline = np.asarray(data["delivery_deadline"], dtype=float)
    if (
        groundtime.ndim != 2
        or groundtime.shape[1] != H
        or first.shape != groundtime.shape
    ):
        raise ValueError("ground_time and first_cost must have shape origins,H")
    if (
        terminal.ndim != 2
        or terminal.shape[1] != H
        or terminaltime.shape != terminal.shape
    ):
        raise ValueError(
            "terminal_cost and terminal_time must have shape destinations,H"
        )
    if any(a.shape != (D,) for a in (oi, di, ground, deadline)):
        raise ValueError("OD arrays must have length n_od")
    if (
        np.any(oi < 0)
        or np.any(oi >= len(first))
        or np.any(di < 0)
        or np.any(di >= len(terminal))
    ):
        raise ValueError("OD row indices out of bounds")
    for name, a in [
        ("ground_time", groundtime),
        ("first_cost", first),
        ("terminal_cost", terminal),
        ("terminal_time", terminaltime),
        ("ground", ground),
        ("delivery_deadline", deadline),
    ]:
        if (
            np.isnan(a).any()
            or np.any(a < 0)
            or (name != "delivery_deadline" and not np.isfinite(a).all())
        ):
            raise ValueError(name + " must be nonnegative and finite")
    scalars = [float(data[k]) for k in ("drone_time_cost", "hover_cost", "fleet_cost")]
    if any(not math.isfinite(v) or v < 0 for v in scalars):
        raise ValueError("cost rates must be nonnegative and finite")
    selected = (
        np.ones((D, L, L), dtype=np.bool_)
        if groups is None
        else np.zeros((D, L, L), dtype=np.bool_)
    )
    if groups is not None:
        a = np.asarray(groups)
        if a.dtype == np.bool_ and a.shape == selected.shape:
            selected[:] = a
        else:
            for od, l, m in groups:
                if not (0 <= od < D and 0 <= l < L and 0 <= m < L):
                    raise ValueError("pricing group out of bounds")
                selected[int(od), int(l), int(m)] = True
    return dict(
        data=data,
        L=L,
        H=H,
        K=K,
        D=D,
        N=N,
        flight=ft,
        battery=battery,
        epsilon=eps,
        family=family,
        facility=fac.astype(np.int64),
        hub_prices=hp,
        arc_prices=ap,
        swap=_vector(data["swap_time"], K, "swap_time"),
        pickup=_vector(data["pickup_handling_time"], H, "pickup_handling_time"),
        delivery=_vector(data["delivery_handling_time"], H, "delivery_handling_time"),
        mean=groundtime,
        first=first,
        terminal=terminal,
        terminaltime=terminaltime,
        oi=oi,
        di=di,
        ground=ground,
        deadline=deadline,
        flightcost=scalars[0],
        hovercost=scalars[1],
        fleet=scalars[2],
        selected=selected,
    )
