"""Canonical CSV reading and source checks for geographic demand instances."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def project_path(path):
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _rows(path, required):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        missing = set(required) - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path.name}: missing columns {sorted(missing)}")
        return list(reader)


def coordinates(rows):
    values = np.asarray(
        [[float(row["x_km"]), float(row["y_km"])] for row in rows], dtype=float
    ).reshape(-1, 2)
    if not np.isfinite(values).all():
        raise ValueError("Coordinates must be finite local kilometers")
    return values


def read_instance(hubs, instance_root):
    """Read sites and OD demand without any model or result-directory input."""
    if int(hubs) != hubs or hubs < 2:
        raise ValueError("hubs must be a positive integer of at least two")
    directories = sorted(project_path(instance_root).glob(f"H{int(hubs):03d}_*"))
    if len(directories) != 1:
        raise ValueError(
            f"Expected exactly one canonical instance for H{hubs}: {directories}"
        )
    directory = directories[0]
    metadata = json.loads((directory / "metadata.json").read_text())
    if metadata.get("coordinate_system") != "local_km":
        raise ValueError("Canonical instances must use local_km coordinates")
    sites = _rows(directory / "sites.csv", ("site_id", "type", "x_km", "y_km"))
    nodes = _rows(directory / "od_nodes.csv", ("node_id", "role", "x_km", "y_km"))
    orders = _rows(
        directory / "od_demand.csv",
        ("origin_node", "destination_node", "lambda_per_hour", "direct_distance_km"),
    )
    if len({r["site_id"] for r in sites}) != len(sites):
        raise ValueError("Duplicate site IDs")
    if len({r["node_id"] for r in nodes}) != len(nodes):
        raise ValueError("Duplicate OD node IDs")
    if any(row["type"] not in ("L", "H", "E") for row in sites):
        raise ValueError("Unknown facility type")
    origins = [row for row in nodes if row["role"] == "origin"]
    destinations = [row for row in nodes if row["role"] == "destination"]
    oi = {row["node_id"]: index for index, row in enumerate(origins)}
    di = {row["node_id"]: index for index, row in enumerate(destinations)}
    demand = np.asarray([float(row["lambda_per_hour"]) for row in orders])
    if not np.isfinite(demand).all() or np.any(demand < 0) or demand.sum() <= 0:
        raise ValueError("Demand must be finite, nonnegative and nonempty")
    od_pairs = [(row["origin_node"], row["destination_node"]) for row in orders]
    if len(set(od_pairs)) != len(od_pairs):
        raise ValueError("Duplicate OD pairs")
    data = dict(
        name=f"Nanshan_H{hubs}",
        bases=coordinates([row for row in sites if row["type"] == "L"]),
        hubs=coordinates([row for row in sites if row["type"] == "H"]),
        remote=coordinates([row for row in sites if row["type"] == "E"]),
        origins=coordinates(origins),
        destinations=coordinates(destinations),
        od_origin=np.asarray([oi[row["origin_node"]] for row in orders]),
        od_destination=np.asarray([di[row["destination_node"]] for row in orders]),
        demand=demand,
    )
    data.update(
        n_l=len(data["bases"]),
        n_h=len(data["hubs"]),
        n_k=len(data["remote"]),
        n_od=len(orders),
    )
    if data["n_h"] != hubs or data["n_l"] < 1:
        raise ValueError("Candidate counts do not match the instance")
    direct = np.asarray([float(row["direct_distance_km"]) for row in orders])
    if not np.isfinite(direct).all() or np.any(direct < 0):
        raise ValueError("Direct distances must be finite and nonnegative")
    data["provenance"] = dict(
        directory=str(directory),
        instance_metadata=metadata,
        sha256={p.name: sha256(p) for p in directory.glob("*.csv")},
        orders_sampled=False,
        demand_rescaled=False,
        screening_is_exact=True,
        historical_top_k_screening_used=False,
    )
    return data, direct, metadata
