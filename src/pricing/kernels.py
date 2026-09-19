"""Compiled exact service-path pricing with battery and delivery limits."""

import math
import numpy as np
from numba import njit

_TOL = 1e-10


@njit(cache=True)
def _service_kernel(
    lf,
    ld,
    lp,
    first,
    last,
    req,
    retprice,
    retchosen,
    facility,
    hp,
    pickup,
    delivery,
    mean,
    firstcost,
    terminal,
    terminaltime,
    oi,
    di,
    ground,
    deadline,
    selected,
    L,
    H,
    K,
    battery,
    quantilefactor,
    family,
    flightcost,
    hovercost,
    fleet,
):
    D = len(oi)
    shape = (D, L, L)
    price = np.full(shape, np.inf)
    actual = np.full(shape, np.inf)
    elapsed = np.zeros(shape)
    choices = np.full((D, L, L, 5), -1, np.int64)
    O = len(mean)
    maxlabels = max(1, H * (K + 1) * (K + 1))
    # A full list of time/resource labels is retained until delivery.  Sorting
    # by delivery quantile gives exact threshold minima for each deadline.
    theta = np.empty(maxlabels)
    resid = np.empty(maxlabels)
    baseprice = np.empty(maxlabels)
    basereal = np.empty(maxlabels)
    basetime = np.empty(maxlabels)
    pick = np.empty(maxlabels, np.int64)
    kprefix = np.empty(maxlabels, np.int64)
    kloaded = np.empty(maxlabels, np.int64)
    prefixbest = np.empty(maxlabels, np.int64)
    returnbest = np.empty(maxlabels, np.int64)
    for o in range(O):
        ods = np.where(oi == o)[0]
        if len(ods) == 0:
            continue
        for l in range(L):
            if not facility[l]:
                continue
            active = False
            for od in ods:
                for m in range(L):
                    if selected[od, l, m]:
                        active = True
            if not active:
                continue
            for j in range(H):
                if not facility[L + j]:
                    continue
                n = 0
                for i in range(H):
                    if i == j or not facility[L + i]:
                        continue
                    g = mean[o, i]
                    qt = g * quantilefactor
                    for ka in range(K + 1):
                        t1 = ld[l, L + i, ka]
                        if not np.isfinite(t1):
                            continue
                        qw = max(0.0, qt - t1)
                        w = (
                            max(0.0, g - t1)
                            if family == 0
                            else (g * math.exp(-t1 / g) if g > 0 else 0.0)
                        )
                        used = last[l, L + i, ka] + qw + pickup[i]
                        if used > battery + _TOL:
                            continue
                        for kb in range(K + 1):
                            if not np.isfinite(ld[L + i, L + j, kb]):
                                continue
                            if first[L + i, L + j, kb] > battery - used + _TOL:
                                continue
                            after = (
                                used + lf[L + i, L + j, kb]
                                if kb == 0
                                else last[L + i, L + j, kb]
                            ) + delivery[j]
                            if after > battery + _TOL:
                                continue
                            t12 = (
                                t1 + w + pickup[i] + ld[L + i, L + j, kb] + delivery[j]
                            )
                            fly = lf[l, L + i, ka] + lf[L + i, L + j, kb]
                            raw = (
                                firstcost[o, i]
                                + flightcost * fly
                                + hovercost * w
                                + fleet * t12
                            )
                            theta[n] = (
                                t1 + qw + pickup[i] + ld[L + i, L + j, kb] + delivery[j]
                            )
                            resid[n] = max(0.0, battery - after)
                            basetime[n] = t12
                            basereal[n] = raw
                            baseprice[n] = (
                                raw
                                + hp[i]
                                + hp[j]
                                + lp[l, L + i, ka]
                                + lp[L + i, L + j, kb]
                            )
                            pick[n] = i
                            kprefix[n] = ka
                            kloaded[n] = kb
                            n += 1
                if n == 0:
                    continue
                order = np.argsort(theta[:n])
                sortedtheta = theta[:n][order]
                for m in range(L):
                    if not facility[m]:
                        continue
                    active = False
                    for od in ods:
                        if selected[od, l, m]:
                            active = True
                    if not active:
                        continue
                    best, bestlabel, bestret = np.inf, -1, -1
                    for position in range(n):
                        label = order[position]
                        pos = (
                            np.searchsorted(
                                req[j, m], resid[label] + _TOL, side="right"
                            )
                            - 1
                        )
                        if pos >= 0:
                            kc = retchosen[j, m, pos]
                            value = baseprice[label] + retprice[j, m, pos]
                            if value < best:
                                best, bestlabel, bestret = value, label, kc
                        prefixbest[position] = bestlabel
                        returnbest[position] = bestret
                    for od in ods:
                        if not selected[od, l, m]:
                            continue
                        budget = deadline[od] - terminaltime[di[od], j]
                        pos = (
                            np.searchsorted(sortedtheta, budget + _TOL, side="right")
                            - 1
                        )
                        if pos < 0:
                            continue
                        label, kc = prefixbest[pos], returnbest[pos]
                        if label < 0 or kc < 0:
                            continue
                        add = terminal[di[od], j] - ground[od]
                        retreal = (
                            flightcost * lf[L + j, m, kc] + fleet * ld[L + j, m, kc]
                        )
                        value = baseprice[label] + retreal + lp[L + j, m, kc] + add
                        if value < price[od, l, m]:
                            price[od, l, m] = value
                            actual[od, l, m] = basereal[label] + retreal + add
                            elapsed[od, l, m] = basetime[label] + ld[L + j, m, kc]
                            choices[od, l, m, 0] = pick[label]
                            choices[od, l, m, 1] = j
                            choices[od, l, m, 2] = kprefix[label] - 1
                            choices[od, l, m, 3] = kloaded[label] - 1
                            choices[od, l, m, 4] = kc - 1
    return price, actual, elapsed, choices
