"""Nested facility candidates on one frozen geographic and OD market."""

from __future__ import annotations

from copy import deepcopy
import csv
import json
from pathlib import Path

import numpy as np

from .choice import pickup_probability, rider_tariff
from .demand import demand_signature
from .instance_io import sha256


LEVELS = ((10, 2, 4), (15, 3, 6), (25, 5, 10), (30, 6, 12))
ROLES = {"H": "hubs", "L": "bases", "E": "remote"}


def _key(point):
    return tuple(round(float(x), 8) for x in point)


def _pool(base, base_path):
    """Use existing observed-activity candidates inside the unchanged H20 crop."""
    metadata = base["provenance"]["instance_metadata"]
    bounds = metadata["region_bounds_local_km"]
    lower = np.array([bounds["x_min"], bounds["y_min"]])
    upper = np.array([bounds["x_max"], bounds["y_max"]])
    points = {role: {} for role in ROLES}
    sources = []
    for directory in sorted(Path(base_path).resolve().parent.glob("H*_L*_E*_OD*")):
        meta = json.loads((directory / "metadata.json").read_text())
        for field in ("source", "region", "projection_center_lon", "projection_center_lat"):
            if meta[field] != metadata[field]:
                raise ValueError(f"Candidate mother set differs in {field}: {directory}")
        path = directory / "sites.csv"
        retained = dict.fromkeys(ROLES, 0)
        with path.open(newline="") as stream:
            for row in csv.DictReader(stream):
                role = row["type"]
                point = np.array([float(row["x_km"]), float(row["y_km"])])
                if role in ROLES and np.all(point >= lower) and np.all(point <= upper):
                    points[role].setdefault(_key(point), point)
                    retained[role] += 1
        sources.append(dict(path=str(path), sha256=sha256(path), retained=retained))
    return {
        role: np.asarray([values[key] for key in sorted(values)]).reshape(-1, 2)
        for role, values in points.items()
    }, sources


def _coverage_order(points):
    """Farthest-point prefixes, starting nearest the original-set centroid."""
    points = np.asarray(points)
    order = []
    remaining = np.ones(len(points), dtype=bool)
    nearest = np.full(len(points), np.inf)
    index = int(np.argmin(np.sum((points - points.mean(axis=0)) ** 2, axis=1)))
    for _ in points:
        order.append(index)
        remaining[index] = False
        nearest = np.minimum(nearest, np.sum((points - points[index]) ** 2, axis=1))
        nearest[~remaining] = -np.inf
        index = int(np.argmax(nearest))
    return order


def _extended(original, pool, maximum):
    """Keep every baseline coordinate and greedily fill its largest coverage gaps."""
    existing = {_key(point) for point in original}
    available = np.asarray([point for point in pool if _key(point) not in existing])
    needed = maximum - len(original)
    if needed > len(available):
        raise ValueError(f"Need {needed} extra candidates; only {len(available)} available")
    if not needed:
        return original.copy()
    nearest = np.min(np.sum((available[:, None] - original[None, :]) ** 2, axis=2), axis=1)
    added = []
    for _ in range(needed):
        index = int(np.argmax(nearest))
        point = available[index]
        added.append(point)
        nearest = np.minimum(nearest, np.sum((available - point) ** 2, axis=1))
        nearest[index] = -np.inf
    return np.vstack((original, added))


def _geometry(data):
    """Refresh only facility-dependent quantities using the frozen calibration."""
    config = data["provenance"]["resolved_config"]
    first = np.linalg.norm(data["origins"][:, None] - data["hubs"][None, :], axis=2)
    last = np.linalg.norm(data["destinations"][:, None] - data["hubs"][None, :], axis=2)
    multiplier = config["costs"]["feeder_rider_multiplier"]
    data["first_cost"] = multiplier * rider_tariff(first, config)
    data["last_rider_cost"] = multiplier * rider_tariff(last, config)
    data["pickup_share"] = pickup_probability(last, **config["choice"])
    data["terminal_cost"] = (1 - data["pickup_share"]) * data["last_rider_cost"]
    slope = data["rider_slope_minutes_per_straight_km"]
    data["ground_time"] = first * slope / 60
    data["last_ground_time"] = last * slope / 60
    data["terminal_time"] = data["last_ground_time"] + data["terminal_dispatch_time"]
    xy = np.vstack((data["bases"], data["hubs"], data["remote"]))
    data["flight_times"] = np.linalg.norm(xy[:, None] - xy[None, :], axis=2) / data["speed"]
    shape = (data["n_h"], data["max_cabinets"])
    data["level_capacity"] = np.broadcast_to(data["unit_capacity"] * data["capacity_levels"], shape).copy()
    data["level_cost"] = np.broadcast_to(data["hub_cost"] * data["capacity_levels"], shape).copy()


def build_density_variants(base, base_path):
    """Return H10/15/25/30 variants; H20 is the previously saved shared baseline.

    Candidate quantities vary. Opening decisions remain free in every solve.
    OD identities, rates, endpoints, cluster-mean courier costs, and all empirical
    time-calibration arrays are copied unchanged from the saved H20 instance.
    """
    if (base["n_h"], base["n_l"], base["n_k"]) != (20, 4, 8):
        raise ValueError("The density family requires the frozen H20/L4/E8 baseline")
    pools, sources = _pool(base, base_path)
    orders = {role: _coverage_order(base[key]) for role, key in ROLES.items()}
    extended = {
        role: _extended(base[key], pools[role], maximum)
        for role, key, maximum in (("H", "hubs", 30), ("L", "bases", 6), ("E", "remote", 12))
    }
    results = []
    for h, l, e in LEVELS:
        counts = dict(H=h, L=l, E=e)
        data = deepcopy(base)
        for role, key in ROLES.items():
            count = counts[role]
            data[key] = (base[key][sorted(orders[role][:count])].copy()
                         if count < len(base[key]) else extended[role][:count].copy())
        data.update(n_h=h, n_l=l, n_k=e)
        _geometry(data)
        case_id = f"density_H{h:02d}"
        data["name"] = f"{base['name']}_{case_id}"
        provenance = data["provenance"]
        metadata = provenance["instance_metadata"]
        metadata["family"] = "fixed_region_density"
        metadata["case_id"] = data["name"]
        metadata["data_config"].update(candidate_count=h, num_bases=l, num_swaps=e)
        area = (2 * metadata["crop_half_size_km"]) ** 2
        provenance["sensitivity"] = dict(
            id=case_id, factor="density", multiplier=h / 20,
            parent_name=base["name"], parent_input_path=str(Path(base_path).resolve()),
            parent_demand_signature=demand_signature(base),
            candidate_counts=counts, candidate_density_per_km2={key: value / area for key, value in counts.items()},
            area_km2=area, candidate_coordinates_nested=True,
            original_h20_coordinates_retained_when_expanding=True,
            all_original_ods_retained=True, opening_counts_fixed=False,
            generation="Farthest-point subsets of baseline; farthest-point additions from existing same-region candidates inside baseline crop",
            candidate_source_files=sources,
            unchanged="OD ids, rates, endpoints, ground costs, deadlines, timing calibration, costs, and crop",
            authoritative_replay="Saved input_arrays.npz plus input_metadata.json",
        )
        provenance["demand_selection"]["mode"] = "fixed"
        data["demand_signature"] = demand_signature(data)
        provenance["demand_selection"]["signature"] = data["demand_signature"]
        if data["demand_signature"] != demand_signature(base):
            raise AssertionError("Density generation changed the demand market")
        results.append(dict(id=case_id, factor="density", multiplier=h / 20, data=data))
    return results
