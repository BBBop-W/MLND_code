"""Reproducible spatial aggregation with independent OD and facility counts.

OD clustering uses the joint four-dimensional pickup/dropoff coordinates.
Repeated observations retain their multiplicities, and each representative is
an observed tuple. The returned membership assigns every input order exactly
once; it does not remove orders or change their demand weights.
"""

from __future__ import annotations

import numpy as np
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
        "destination_distortion": _distortion(
            np.linalg.norm(difference[:, 2:], axis=1)
        ),
        "joint_distortion": _distortion(np.linalg.norm(difference, axis=1)),
    }
    return labels, representatives, audit


def select_sites(points, count, seed=43):
    """Return exact K observed site coordinates with activity weights (K by 3)."""
    points = _points(points, 2)
    _, representatives, weights, _ = _cluster(points, count, seed)
    return np.column_stack((representatives, weights))
