"""Direct and single-swap leg tables, plus exact return-cost envelopes."""

import numpy as np
from numba import njit

_TOL = 1e-10


@njit(cache=True)
def _leg_tables(ft, ap, facility, swap, L, H, K, battery):
    N = len(ft)
    shape = (N, N, K + 1)
    flight = np.full(shape, np.inf)
    duration = np.full(shape, np.inf)
    price = np.full(shape, np.inf)
    first = np.full(shape, np.inf)
    last = np.full(shape, np.inf)
    for u in range(N):
        if not facility[u]:
            continue
        for v in range(N):
            if u == v or not facility[v]:
                continue
            if ft[u, v] <= battery + _TOL:
                flight[u, v, 0] = duration[u, v, 0] = first[u, v, 0] = last[u, v, 0] = (
                    ft[u, v]
                )
                price[u, v, 0] = ap[u, v]
            for k in range(K):
                w = L + H + k
                if not facility[w] or u == w or v == w:
                    continue
                a, b = ft[u, w], ft[w, v]
                if a <= battery + _TOL and b <= battery + _TOL:
                    flight[u, v, k + 1] = a + b
                    duration[u, v, k + 1] = a + b + swap[k]
                    price[u, v, k + 1] = ap[u, w] + ap[w, v]
                    first[u, v, k + 1] = a
                    last[u, v, k + 1] = b
    return flight, duration, price, first, last


@njit(cache=True)
def _return_envelopes(lf, ld, lp, first, last, L, H, K, battery, flightcost, fleet):
    req = np.full((H, L, K + 1), np.inf)
    val = np.full((H, L, K + 1), np.inf)
    choice = np.full((H, L, K + 1), -1, np.int64)
    for j in range(H):
        for m in range(L):
            demands = first[L + j, m, :]
            order = np.argsort(demands)
            best = np.inf
            bestk = -1
            for a in range(K + 1):
                k = order[a]
                req[j, m, a] = demands[k]
                cost = (
                    flightcost * lf[L + j, m, k]
                    + fleet * ld[L + j, m, k]
                    + lp[L + j, m, k]
                )
                if cost < best:
                    best, bestk = cost, k
                val[j, m, a] = best
                choice[j, m, a] = bestk
    return req, val, choice
