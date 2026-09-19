"""Complete physical-path enumeration for small-instance reference solves."""

import itertools
import numpy as np
from .inputs import _validated
from .routes import _materialize

_TOL = 1e-10


def enumerate_all_columns(data, facility=None):
    """Exhaustive physical-path reference for small instances only.

    Returns ``{'service': list, 'empty': list}``; no dominance or pricing
    optimization is used.  This is deliberately separate from the kernel.
    """
    c = _validated(data, facility)
    service = []
    empty = []
    L, H, K = c["L"], c["H"], c["K"]
    options = [-1] + [k for k in range(K) if c["facility"][L + H + k]]

    def feasible(tr):
        if any(not np.isfinite(s["flight_hours"]) for s in tr["segments"]):
            return False
        if max(tr["battery_interval_consumptions"]) > c["battery"] + _TOL:
            return False
        if "od" in tr and tr["deadline_quantile_time"] > c["deadline"][tr["od"]] + _TOL:
            return False
        return True

    for l, m in itertools.product(range(L), repeat=2):
        if not c["facility"][l] or not c["facility"][m]:
            continue
        if l != m:
            for k in options:
                tr = _materialize(c, None, l, m, None, None, [k])
                if feasible(tr):
                    empty.append(tr)
        for od in range(c["D"]):
            for i, j in itertools.permutations(range(H), 2):
                if not c["facility"][L + i] or not c["facility"][L + j]:
                    continue
                for swaps in itertools.product(options, repeat=3):
                    tr = _materialize(c, od, l, m, i, j, swaps)
                    if feasible(tr):
                        service.append(tr)
    return dict(service=service, empty=empty)
