"""Public path-pricing interface for service and empty missions.

Pricing considers the complete candidate graph unless a facility mask is
explicitly supplied. Each of three service legs permits one battery swap.
"""

import math
import time
import numpy as np
from .inputs import _validated
from .legs import _leg_tables, _return_envelopes
from .kernels import _service_kernel
from .routes import (
    _materialize as _materialize,
    trace_mission as trace_mission,
    trace_empty as trace_empty,
)
from .enumeration import enumerate_all_columns as enumerate_all_columns


def price_design(data, facility=None, hub_prices=None, arc_prices=None, groups=None):
    """Return exact minimum columns for requested OD,L,L groups and all empties.

    ``groups=None`` prices every group over the complete candidate graph.
    A facility mask restricts the graph only when explicitly supplied.  A
    subset is marked ``complete=False`` until a full pricing pass is made.
    """
    start = time.perf_counter()
    c = _validated(data, facility, hub_prices, arc_prices, groups)
    L, H, K, D = c["L"], c["H"], c["K"], c["D"]
    tables = _leg_tables(
        c["flight"], c["arc_prices"], c["facility"], c["swap"], L, H, K, c["battery"]
    )
    c["tables"] = tables
    lf, ld, lp, first, last = tables
    envelope = _return_envelopes(
        *tables, L, H, K, c["battery"], c["flightcost"], c["fleet"]
    )
    price, actual, elapsed, choices = _service_kernel(
        *tables,
        *envelope,
        c["facility"],
        c["hub_prices"],
        c["pickup"],
        c["delivery"],
        c["mean"],
        c["first"],
        c["terminal"],
        c["terminaltime"],
        c["oi"],
        c["di"],
        c["ground"],
        c["deadline"],
        c["selected"],
        L,
        H,
        K,
        c["battery"],
        1.0 if c["family"] == "deterministic" else -math.log(c["epsilon"]),
        0 if c["family"] == "deterministic" else 1,
        c["flightcost"],
        c["hovercost"],
        c["fleet"],
    )
    empty_cost = np.full((L, L), np.inf)
    empty_price = np.full((L, L), np.inf)
    empty_time = np.zeros((L, L))
    empty_choice = np.full((L, L), -2, np.int64)
    for l in range(L):
        for m in range(L):
            if l == m:
                continue
            vals = c["flightcost"] * lf[l, m] + c["fleet"] * ld[l, m] + lp[l, m]
            k = int(np.argmin(vals))
            if np.isfinite(vals[k]):
                empty_price[l, m] = vals[k]
                empty_cost[l, m] = (
                    c["flightcost"] * lf[l, m, k] + c["fleet"] * ld[l, m, k]
                )
                empty_time[l, m] = ld[l, m, k]
                empty_choice[l, m] = k - 1
    c["choices"], c["empty_choice"] = choices, empty_choice
    return dict(
        cost=actual,
        pricing_cost=price,
        time=elapsed,
        i=choices[:, :, :, 0],
        j=choices[:, :, :, 1],
        empty_cost=empty_cost,
        empty_pricing_cost=empty_price,
        empty_time=empty_time,
        facility=c["facility"].tolist(),
        hub_prices=c["hub_prices"].tolist(),
        arc_prices=c["arc_prices"],
        arc_node_count=c["N"],
        complete=bool(c["selected"].all()),
        full_graph=bool(c["facility"].all()),
        groups_priced=int(c["selected"].sum()),
        implicit_basic_paths=int(D * L * L * H * (H - 1)),
        enumerated_complete_paths=0,
        wall=time.perf_counter() - start,
        _context=c,
    )
