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


def instance_directory(hubs, instance_root):
    if isinstance(hubs, bool) or int(hubs) != hubs or hubs < 2:
        raise ValueError("hubs must be a positive integer of at least two")
    directories = sorted(project_path(instance_root).glob(f"H{int(hubs):03d}_*"))
    if len(directories) != 1:
        raise ValueError(
            f"Expected exactly one canonical instance for H{hubs}: {directories}"
        )
    return directories[0]


def read_instance(hubs, instance_root, *, demand_hubs=None):
    """Read candidates and a separately selected complete demand market."""
    directory = instance_directory(hubs, instance_root)
    source_hubs = hubs if demand_hubs is None else demand_hubs
    demand_directory = instance_directory(source_hubs, instance_root)
    metadata = json.loads((directory / "metadata.json").read_text())
    demand_metadata = json.loads((demand_directory / "metadata.json").read_text())
    if metadata.get("coordinate_system") != "local_km":
        raise ValueError("Canonical instances must use local_km coordinates")
    if demand_directory != directory:
        for key in (
            "coordinate_system",
            "projection_center_lon",
            "projection_center_lat",
        ):
            if (
                key not in metadata
                or key not in demand_metadata
                or metadata[key] != demand_metadata[key]
            ):
                raise ValueError(
                    f"Candidate and demand coordinate systems differ: {key}"
                )
    sites = _rows(directory / "sites.csv", ("site_id", "type", "x_km", "y_km"))
    nodes = _rows(
        demand_directory / "od_nodes.csv", ("node_id", "role", "x_km", "y_km")
    )
    orders = _rows(
        demand_directory / "od_demand.csv",
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
        od_pairs=[list(pair) for pair in od_pairs],
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
    sources = {"sites.csv": directory / "sites.csv"}
    sources.update(
        {
            name: demand_directory / name
            for name in ("od_nodes.csv", "od_demand.csv", "periods.csv")
        }
    )
    data["provenance"] = dict(
        directory=str(directory),
        instance_metadata=metadata,
        sha256={name: sha256(path) for name, path in sources.items()},
        source_files={name: str(path) for name, path in sources.items()},
        demand_selection=dict(
            mode="instance" if demand_hubs is None else "fixed",
            source_hubs=int(source_hubs),
            directory=str(demand_directory),
            instance_metadata=demand_metadata,
            n_od=len(orders),
            hourly_demand=float(demand.sum()),
            all_source_ods_included=True,
        ),
        orders_sampled=False,
        demand_rescaled=False,
        screening_is_exact=True,
        historical_top_k_screening_used=False,
    )
    return data, direct, metadata
