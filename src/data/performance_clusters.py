"""Reproducible spatial aggregation for algorithm-performance instances.

OD clustering uses the joint four-dimensional pickup/dropoff coordinates.
Repeated observations retain their multiplicities, and each representative is
an observed tuple. The returned membership assigns every input order exactly
once; it does not remove orders or change their demand weights.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree
from sklearn.cluster import BisectingKMeans
from threadpoolctl import threadpool_limits


def _points(points, dimensions):
    values = np.asarray(points, dtype=float)
    if values.ndim != 2 or values.shape[1] != dimensions or not len(values):
        raise ValueError(f"Expected a nonempty N by {dimensions} coordinate array")
    if not np.isfinite(values).all():
        raise ValueError("Coordinates must be finite")
    return values


def _cluster(points, count, seed):
    """Cluster weighted unique observations and canonically number medoids."""
    if isinstance(count, bool) or int(count) != count or count <= 0:
        raise ValueError("Cluster count must be a positive integer")
    count = int(count)
    unique, inverse, weights = np.unique(
        points, axis=0, return_inverse=True, return_counts=True
    )
    if count > len(unique):
        raise ValueError(f"Requested {count} clusters from {len(unique)} unique points")
    if count == len(unique):
        membership = np.arange(count)
    elif count == 1:
        membership = np.zeros(len(unique), dtype=int)
    else:
        # Bisecting K-means avoids repeatedly comparing all observations with
        # all 1,200 centers. Every unique point enters the weighted fit.
        with threadpool_limits(limits=1):
            model = BisectingKMeans(
                n_clusters=count,
                random_state=int(seed),
                n_init=3,
                max_iter=100,
                bisecting_strategy="biggest_inertia",
            ).fit(unique, sample_weight=weights)
        membership = model.labels_.copy()

    # Numerical degeneracy in a bisection must not silently reduce the count.
    # Split a distinct observation from a nonsingleton cluster if necessary.
    for missing in sorted(set(range(count)) - set(membership.tolist())):
        sizes = np.bincount(membership, minlength=count)
        eligible = sizes[membership] > 1
        score = np.full(len(unique), -np.inf)
        for group in np.unique(membership[eligible]):
            members = np.flatnonzero(membership == group)
            center = np.average(unique[members], axis=0, weights=weights[members])
            score[members] = weights[members] * np.sum(
                (unique[members] - center) ** 2, axis=1
            )
        membership[int(np.argmax(score))] = missing

    representatives = np.empty((count, unique.shape[1]))
    for group in range(count):
        members = np.flatnonzero(membership == group)
        center = np.average(unique[members], axis=0, weights=weights[members])
        # np.unique has lexicographic order, also fixing equal-distance ties.
        nearest = np.argmin(np.sum((unique[members] - center) ** 2, axis=1))
        representatives[group] = unique[members[nearest]]
    order = np.lexsort(representatives.T[::-1])
    canonical = np.empty(count, dtype=int)
    canonical[order] = np.arange(count)
    labels = canonical[membership][inverse]
    representatives = representatives[order]
    counts = np.bincount(labels, minlength=count)
    return labels, representatives, counts, len(unique)


def _distortion(distances):
    return {
        "mean_km": float(np.mean(distances)),
        "rms_km": float(np.sqrt(np.mean(distances**2))),
        "max_km": float(np.max(distances)),
    }


def cluster_od(points, count, seed=43):
    """Return ``(labels, observed_representatives, audit)`` for exact K ODs.

    ``points`` has one row per raw order: ``[ox, oy, dx, dy]`` in kilometers.
    Origin and destination coordinates enter with equal scaling. The audit's
    distortion statistics are order-weighted, including repeated raw tuples.
    """
    points = _points(points, 4)
    labels, representatives, counts, unique_count = _cluster(points, count, seed)
    difference = points - representatives[labels]
    audit = {
        "method": "weighted_bisecting_kmeans_observed_joint_od_representatives",
        "seed": int(seed),
        "orders": len(points),
        "unique_input_od": unique_count,
        "clusters": len(representatives),
        "assigned_orders": int(counts.sum()),
        "unassigned_orders": 0,
        "smallest_cluster_orders": int(counts.min()),
        "largest_cluster_orders": int(counts.max()),
        "origin_distortion": _distortion(np.linalg.norm(difference[:, :2], axis=1)),
        "destination_distortion": _distortion(np.linalg.norm(difference[:, 2:], axis=1)),
        "joint_distortion": _distortion(np.linalg.norm(difference, axis=1)),
    }
    return labels, representatives, audit


def select_sites(points, count, seed=43):
    """Return exact K observed site coordinates with activity weights (K by 3)."""
    points = _points(points, 2)
    _, representatives, weights, _ = _cluster(points, count, seed)
    return np.column_stack((representatives, weights))


def _separated_pair(centers, counts, minimum_separation):
    """Maximize combined count with nonoverlap along at least one axis."""
    best = None
    for axis in (0, 1):
        ordered = np.argsort(centers[:, axis], kind="stable")
        coordinates = centers[ordered, axis]
        prefix_best = np.empty(len(ordered), dtype=int)
        incumbent = int(ordered[0])
        for position, current in enumerate(ordered):
            if (counts[current], -int(current)) > (counts[incumbent], -incumbent):
                incumbent = int(current)
            prefix_best[position] = incumbent
        for current in ordered:
            boundary = np.searchsorted(
                coordinates, centers[current, axis] - minimum_separation, side="right"
            )
            if boundary:
                previous = int(prefix_best[boundary - 1])
                pair = tuple(sorted((previous, int(current))))
                key = (int(counts[pair[0]] + counts[pair[1]]), -pair[0], -pair[1])
                if best is None or key > best[0]:
                    best = key, pair
    return None if best is None else best[1]


def choose_regions(orders, half_size_h10=10 * np.sqrt(10 / 120), seed=43):
    """Choose two data-rich H10 squares, preferring spatially disjoint areas.

    The input DataFrame contains ``ox, oy, dx, dy, hour`` in kilometers.
    Candidate centers are a fixed 0.5 km lattice over observed noon-order
    midpoints. Among squares containing at least 200 distinct ODs, maximize
    the combined order count subject to nonoverlap along x or y. If the data
    cannot support two disjoint squares, use the busiest square and then the
    busiest square with minimum unavoidable overlap. No optimization results
    enter this selection; seed is recorded for the instance specification.
    """
    if not np.isfinite(half_size_h10) or half_size_h10 <= 0:
        raise ValueError("half_size_h10 must be positive")
    noon = orders.loc[orders["hour"] == 12, ["ox", "oy", "dx", "dy"]]
    points = _points(noon.to_numpy(), 4)
    midpoint = (points[:, :2] + points[:, 2:]) / 2
    grid = 0.5
    centers = (np.unique(np.floor(midpoint / grid), axis=0) + 0.5) * grid
    queries = np.column_stack((centers, centers))
    tree = cKDTree(points)
    counts = tree.query_ball_point(queries, half_size_h10, p=np.inf, return_length=True)
    centers, counts = centers[counts >= 200], counts[counts >= 200]
    if len(centers) < 2:
        raise ValueError("Data cannot support two region candidates with 200 orders")
    rejected = 0
    while True:
        pair = _separated_pair(centers, counts, 2 * half_size_h10)
        nonoverlap = pair is not None
        if pair is None:
            first = int(np.argmax(counts))
            gap = np.maximum(0, 2 * half_size_h10 - np.abs(centers - centers[first]))
            overlaps = gap[:, 0] * gap[:, 1]
            overlaps[first] = np.inf
            second = int(np.lexsort((np.arange(len(counts)), -counts, overlaps))[0])
            pair = (first, second)
        invalid = []
        audits = []
        for index in pair:
            query = np.tile(centers[index], 2)
            rows = tree.query_ball_point(query, half_size_h10, p=np.inf)
            distinct = len(np.unique(points[rows], axis=0))
            if distinct < 200:
                invalid.append(index)
            audits.append({"orders_h10": len(rows), "distinct_od_h10": distinct})
        if not invalid:
            chosen = centers[list(pair)]
            overlap_sides = np.maximum(
                0, 2 * half_size_h10 - np.abs(chosen[0] - chosen[1])
            )
            return chosen, {
                "method": "noon_joint_endpoint_square_counts_on_0.5km_midpoint_lattice",
                "seed": int(seed),
                "half_size_h10_km": float(half_size_h10),
                "grid_km": grid,
                "nonoverlap_h10": nonoverlap,
                "overlap_h10_km2": float(np.prod(overlap_sides)),
                "center_linf_distance_km": float(np.max(np.abs(chosen[0] - chosen[1]))),
                "center_distance_km": float(np.linalg.norm(chosen[0] - chosen[1])),
                "rejected_low_unique_od_centers": rejected,
                "regions": audits,
            }
        keep = np.ones(len(centers), dtype=bool)
        keep[invalid] = False
        centers, counts = centers[keep], counts[keep]
        rejected += len(invalid)
        if len(centers) < 2:
            raise ValueError("Data cannot support two regions with 200 distinct ODs")
