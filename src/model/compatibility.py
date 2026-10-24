"""Freeze OD compatibility using complete, physically feasible two-order paths.

Compatibility is screened before assignment or facility optimization.  It is a
global existence condition over the declared candidate network, not a promise
that every departure-return base pair can serve every retained OD transition.
"""

from dataclasses import replace

import numpy as np

from .paths import enumerate_paths
from .resources import (
    PathInfeasible,
    build_initial_segment,
    build_service_path,
    first_wait,
    second_wait,
)
from .types import PlanningData


def _direct_witness(data, b, r, s, i, j, i2, j2):
    """Check one witness using exactly the same scalar builders as enumeration."""
    n_l = data.n_l
    try:
        initial = build_initial_segment(
            data, b, r, i, j, ((b[0], n_l + i), (n_l + i, n_l + j))
        )
        transfer = (n_l + j,) if j == i2 else (n_l + j, n_l + i2)
        build_service_path(
            data,
            initial,
            (*initial.legs, transfer, (n_l + i2, n_l + j2), (n_l + j2, b[1])),
            second_od=s,
            second_pickup=i2,
            second_delivery=j2,
        )
        return True
    except PathInfeasible:
        return False


def _boundary_exists(data, b, r, s):
    """Rare floating-point boundary fallback; exhaustive, without a cutoff."""
    n_l, h = data.n_l, data.n_h
    for i in range(h):
        if not data.allowed_links[b[0], n_l + i]:
            continue
        for j in range(h):
            if i == j or not data.allowed_links[n_l + i, n_l + j]:
                continue
            try:
                build_initial_segment(
                    data, b, r, i, j, ((b[0], n_l + i), (n_l + i, n_l + j))
                )
            except PathInfeasible:
                continue
            for i2 in range(h):
                if j != i2 and not data.allowed_links[n_l + j, n_l + i2]:
                    continue
                for j2 in range(h):
                    if i2 == j2 or not data.allowed_links[n_l + i2, n_l + j2]:
                        continue
                    if not data.allowed_links[n_l + j2, b[1]]:
                        continue
                    if _direct_witness(data, b, r, s, i, j, i2, j2):
                        return True
    return False


def _direct_support(data, proposed):
    """Exact direct-route existence via separable minimum operating times.

    For a fixed first delivery j, minimize the first prefix over its pickup i,
    and the continuation over (i2,j2), subject to each order's own deadline.
    Their only remaining coupling is the additive mission operating limit.
    This is an existence oracle, not a catalogue or a count of all paths.
    Retained pairs have scalar-builder witnesses. Conservative rounding slack
    and an exhaustive fallback settle the exceptional boundary cases.
    """
    data = replace(data, max_orders=2)
    n_l, h, r_count = data.n_l, data.n_h, data.n_od
    loaded = data.flight_times[n_l:, n_l:]
    allowed_loaded = data.allowed_links[n_l:, n_l:]
    first_min = np.full((n_l, r_count, h), np.inf)
    first_pick = np.zeros((n_l, r_count, h), dtype=int)
    for origin in range(n_l):
        travel = data.flight_times[origin, n_l:]
        waits = np.array(
            [
                [first_wait(float(mu), float(t)) for mu, t in zip(row, travel)]
                for row in data.rider_mean
            ]
        )
        q = travel[None, :] + waits
        prefix = q[:, :, None] + loaded[None, :, :]
        receipt = prefix + data.receipt_mean[:, None, :]
        feasible = (
            data.allowed_links[origin, n_l:][None, :, None]
            & allowed_loaded[None, :, :]
            & (loaded[None, :, :] + 1e-12 >= data.delta)
            & (receipt <= data.deadline[:, None, None] + 1e-12)
        )
        values = np.where(feasible, prefix, np.inf)
        first_min[origin] = values.min(axis=1)
        first_pick[origin] = values.argmin(axis=1)

    second_min = np.full((n_l, r_count, h), np.inf)
    second_pick = np.zeros((n_l, r_count, h), dtype=int)
    second_drop = np.zeros((n_l, r_count, h), dtype=int)
    for j in range(h):
        transfer = loaded[j]
        waits = np.array(
            [
                [
                    second_wait(float(mu), float(t), data.delta)
                    for mu, t in zip(row, transfer)
                ]
                for row in data.rider_mean
            ]
        )
        receipt = (
            (transfer[None, :] + data.delta / 2 + waits)[:, :, None]
            + loaded[None, :, :]
            + data.receipt_mean[:, None, :]
        )
        transfer_allowed = data.allowed_links[n_l + j, n_l:].copy()
        transfer_allowed[j] = True  # zero-length repositioning needs no arc
        feasible = (
            transfer_allowed[None, :, None]
            & allowed_loaded[None, :, :]
            & (receipt <= data.deadline[:, None, None] + 1e-12)
        )
        for destination in range(n_l):
            back = data.flight_times[n_l:, destination]
            values = (
                transfer[None, :, None]
                + loaded[None, :, :]
                + back[None, None, :]
                + waits[:, :, None]
            )
            values = np.where(
                feasible & data.allowed_links[n_l:, destination][None, None, :],
                values,
                np.inf,
            ).reshape(r_count, h * h)
            best = values.argmin(axis=1)
            second_min[destination, :, j] = values[np.arange(r_count), best]
            second_pick[destination, :, j] = best // h
            second_drop[destination, :, j] = best % h

    by_base = {}
    # Scalar fsum and reassociated additions differ only at numerical boundaries.
    slack = 128 * np.finfo(float).eps * max(1.0, data.operating_limit)
    pairs = np.argwhere(proposed)
    for b in data.base_pairs:
        local = np.zeros_like(proposed)
        costs = first_min[b[0], pairs[:, 0]] + second_min[b[1], pairs[:, 1]]
        best_j = costs.argmin(axis=1)
        minima = costs[np.arange(len(pairs)), best_j]
        for p in np.flatnonzero(minima <= data.operating_limit + 1e-12 + slack):
            r, s = map(int, pairs[p])
            j = int(best_j[p])
            witness = (
                int(first_pick[b[0], r, j]),
                j,
                int(second_pick[b[1], s, j]),
                int(second_drop[b[1], s, j]),
            )
            if _direct_witness(data, b, r, s, *witness):
                local[r, s] = True
            else:
                # No negative conclusion can be based on a rounded minimum.
                local[r, s] = _boundary_exists(data, b, r, s)
        by_base[b] = local
    return by_base


def screen_compatibility(data):
    """Return ``(screened_data, diagnostics)`` without changing the input.

    Intersect the supplied positive kappa support with existence of a complete
    feasible double path.  Preserve retained fractional coefficients exactly.
    Use K=2 physics even for a K=1 comparison, so changing only ``max_orders``
    cannot change the frozen compatibility scenario.  All physical checks are
    those of the authoritative resource builders used by path enumeration.

    Existence search has no column/time cutoff. An exception or incomplete catalogue
    aborts screening: absence from a truncated search is never proof of physical
    impossibility.  Existing saved inputs are not screened by their replay path.
    """
    if not isinstance(data, PlanningData):
        raise TypeError("Expected PlanningData")
    proposed = data.kappa > 0
    support = np.zeros_like(proposed)
    by_base = {b: np.zeros_like(proposed) for b in data.base_pairs}
    path_counts = {b: 0 for b in data.base_pairs}
    algorithm = "enumeration"
    scope = "supplied_routes" if data.candidate_routes is not None else data.route_mode
    if proposed.any() and data.delta > 0:
        if data.route_mode == "direct" and data.candidate_routes is None:
            algorithm = "separable_direct_existence_with_scalar_witnesses"
            by_base = _direct_support(data, proposed)
            for b, local in by_base.items():
                support |= local
                path_counts[b] = None if local.any() else 0
        else:
            catalogue = enumerate_paths(replace(data, max_orders=2))
            if not catalogue.complete:
                raise ValueError(
                    "Compatibility screening requires a complete path catalogue"
                )
            scope = catalogue.scope
            for path in catalogue.service_paths:
                if path.second_od is None:
                    continue
                pair = path.first_od, path.second_od
                support[pair] = True
                by_base[path.b][pair] = True
                path_counts[path.b] += 1
    retained = proposed & support
    kappa = np.where(retained, data.kappa, 0.0)
    diagnostics = {
        "method": "complete_two_order_path_existence",
        "algorithm": algorithm,
        "complete": True,
        "scope": scope,
        "network_scope": "all fixed candidate facilities and allowed routes, before optimization",
        "screening_max_orders": 2,
        "window_hours": float(data.delta),
        "proposed_pairs": int(proposed.sum()),
        "retained_pairs": int(retained.sum()),
        "removed_pairs": int((proposed & ~retained).sum()),
        "removed_od_pairs": np.argwhere(proposed & ~retained).tolist(),
        "retained_self_pairs": int(np.diag(retained).sum()),
        "complete_double_paths": (
            None
            if any(v is None for v in path_counts.values())
            else int(sum(path_counts.values()))
        ),
        "path_count_interpretation": "null means existence was proved without enumerating/counting all paths",
        "base_pairs": [
            {
                "departure": b[0],
                "return": b[1],
                "supported_pairs": int(local.sum()),
                "globally_retained_but_unavailable": int((retained & ~local).sum()),
                "complete_double_paths": path_counts[b],
            }
            for b, local in by_base.items()
        ],
        "interpretation": "positive kappa requires one complete physical double path somewhere in the declared network; selected base-pair feasibility remains a solver constraint",
    }
    return replace(data, kappa=kappa), diagnostics
