"""Complete-catalogue operating lower bounds for the Benders master.

This module never constructs or solves an optimization model. Call it with
the same physically validated complete catalogue used by the solver. Returned
perassignment_savings is currency per served order, indexed by the strategic assignment shape;
multiply by demand when forming coefficients of assignment variables.
"""

from __future__ import annotations

import math

import numpy as np

from src.model.assignment import assignment_shape, assignment_key


def master_bounds(data, paths):
    """Compute operating lower bounds without assuming assignment means service.

    For every service path p, distribute cost_p + fleet_cost * time_p equally
    over its order occurrences. An AA mission has two occurrences. Each OD's
    minimum share is consequently a valid simultaneous per-order lower bound.

    The locker coefficient is the largest positive net saving per unit of
    aggregate slot occupancy. A profitable zero-occupancy path disables this
    second bound rather than allowing division by zero or an invalid finite
    coefficient. All-courier, unavailable-pool and no-profit cases are included.
    """
    if not paths.complete:
        raise ValueError("Master bounds require the complete path catalogue")
    ground = np.asarray(data.ground_cost, dtype=float)
    demand = np.asarray(data.demand, dtype=float)
    fleet_cost = float(data.fleet_cost)
    if (
        not np.isfinite(ground).all()
        or not np.isfinite(demand).all()
        or np.any(ground < 0)
        or np.any(demand < 0)
        or not math.isfinite(fleet_cost)
        or fleet_cost < 0
    ):
        raise ValueError("Demand and operating cost coefficients must be nonnegative")
    minimum = np.full(assignment_shape(data), np.inf)
    beta = 0.0
    zero_load_profitable_paths = []
    zero_load_paths = 0
    positive_saving_paths = 0
    for index, path in enumerate(paths.service_paths):
        cost, duration = float(path.cost), float(path.time)
        if not np.isfinite([cost, duration]).all() or min(cost, duration) < 0:
            raise ValueError("Path costs and busy times must be finite and nonnegative")
        if len(path.b) != 2 or any(
            isinstance(base, (bool, np.bool_))
            or int(base) != base
            or not 0 <= base < data.n_l
            for base in path.b
        ):
            raise ValueError("Service path has an invalid base pair")
        pair = tuple(map(int, path.b))
        counts = []
        seen = set()
        for od, count in path.served_counts:
            if (
                isinstance(od, (bool, np.bool_))
                or int(od) != od
                or not 0 <= od < data.n_od
                or od in seen
                or isinstance(count, (bool, np.bool_))
                or int(count) != count
                or count < 1
            ):
                raise ValueError("Service path order occurrences must be valid counts")
            seen.add(od)
            counts.append((int(od), int(count)))
        occurrences = sum(count for _, count in counts)
        if occurrences not in (1, 2) or occurrences > data.max_orders:
            raise ValueError("Path order occurrences disagree with max_orders")
        occupancy = 0.0
        for locker, value in path.slot_times:
            if (
                isinstance(locker, (bool, np.bool_))
                or int(locker) != locker
                or not 0 <= locker < data.n_h
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError("Locker occupancies must be finite and nonnegative")
            occupancy += value
        augmented = cost + fleet_cost * duration
        if not math.isfinite(augmented) or not math.isfinite(occupancy):
            raise ValueError("Derived path costs and occupancies must be finite")
        # Round shares downward and savings upward: both make the bounds weaker.
        share = float(np.nextafter(augmented / occurrences, -np.inf))
        share = max(0.0, share)
        for od, _ in counts:
            key = assignment_key(data, od, pair)
            minimum[key] = min(minimum[key], share)
        saving = (
            math.fsum(float(ground[od]) * count for od, count in counts) - augmented
        )
        if saving > 0:
            positive_saving_paths += 1
        if occupancy == 0:
            zero_load_paths += 1
            if saving > 0:
                zero_load_profitable_paths.append(index)
        elif saving > 0:
            beta = max(beta, float(np.nextafter(saving / occupancy, np.inf)))
    supported = np.isfinite(minimum)
    gains = np.zeros_like(minimum)
    broadcast_ground = np.broadcast_to(
        ground.reshape((-1,) + (1,) * (minimum.ndim - 1)), minimum.shape
    )
    gains[supported] = np.maximum(0.0, broadcast_ground[supported] - minimum[supported])
    gains[gains > 0] = np.nextafter(gains[gains > 0], np.inf)
    if beta > 0:
        beta = float(np.nextafter(beta * (1 + 64 * np.finfo(float).eps), np.inf))
    locker_value = None if zero_load_profitable_paths else beta
    baseline = float(demand @ ground)
    numerical_margin = 1e-8 * (1 + abs(baseline))
    global_floor = baseline - float(
        demand @ gains.max(axis=tuple(range(1, gains.ndim)))
    )
    return dict(
        perassignment_savings=gains,
        locker_value_per_load=locker_value,
        metadata=dict(
            method="complete_catalogue_cost_and_locker_load_bounds",
            catalogue_complete=True,
            service_paths=len(paths.service_paths),
            supported_assignments=int(supported.sum()),
            positive_saving_assignments=int((gains > 0).sum()),
            unsupported_ods=np.flatnonzero(
                ~supported.any(axis=tuple(range(1, supported.ndim)))
            ).tolist(),
            all_courier_cost=baseline,
            numerical_margin=numerical_margin,
            assignment_global_operating_floor=global_floor - numerical_margin,
            positive_saving_paths=positive_saving_paths,
            zero_load_paths=zero_load_paths,
            zero_load_profitable_path_indices=zero_load_profitable_paths,
            locker_cut_enabled=locker_value is not None,
            locker_value_per_load=locker_value,
            locker_cut_disabled_reason="profitable_zero_occupancy_path"
            if zero_load_profitable_paths
            else None,
            savings_units="currency_per_order",
            locker_coefficient_units="currency_per_hour_per_mean_occupied_compartment",
            costs_included="path operating cost plus fleet cost times complete busy time",
            costs_relaxed="nonnegative idle fleet and empty repositioning costs",
            occupancy_source="sum of the existing physical path slot_times; no added return time",
        ),
    )
