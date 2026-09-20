"""Deterministic candidate families with one unchanged H120 demand market.

Run ``python -m src.data.candidates`` to build both families. All locations
are selected from existing observed-activity medoids, never from a map grid.
"""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import numpy as np

from .instance_io import ROOT, instance_directory, sha256

SOURCE_ROOT = ROOT / "data/instances/nanshan_nested_h120"
AREA_ROOT = ROOT / "data/instances/nanshan_area_candidates"
NESTED_ROOT = ROOT / "data/instances/nanshan_nested_candidates"
AREA_LEVELS = (11, 24, 43, 68, 87, 120)
NESTED_LEVELS = (10, 20, 40, 60, 80, 100, 120)
DEMAND_FILES = ("od_nodes.csv", "od_demand.csv", "periods.csv")


def _read_sites(directory):
    with (directory / "sites.csv").open(newline="", encoding="utf-8-sig") as stream:
        return [row for row in csv.DictReader(stream) if row["type"] == "H"]


def coverage_order(rows, start_rank=0):
    """Farthest-point traversal, starting at a near-centroid observed site.

    Each prefix covers the same geographic mother set. Ties use the source site
    ID, so row permutations do not change the generated IDs or coordinates.
    """
    rows = sorted(rows, key=lambda row: row["site_id"])
    xy = np.asarray([[float(r["x_km"]), float(r["y_km"])] for r in rows])
    if not len(rows) or not 0 <= start_rank < len(rows):
        raise ValueError("Invalid candidate mother set or starting rank")
    center_distance = np.sum((xy - xy.mean(axis=0)) ** 2, axis=1)
    start = int(np.argsort(center_distance, kind="stable")[start_rank])
    selected, remaining = [], np.ones(len(rows), dtype=bool)
    nearest = np.full(len(rows), np.inf)
    index = start
    for _ in rows:
        selected.append(rows[index])
        remaining[index] = False
        nearest = np.minimum(nearest, np.sum((xy - xy[index]) ** 2, axis=1))
        nearest[~remaining] = -np.inf
        index = int(np.argmax(nearest))
    return selected


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def _role_sites(rows, role, count):
    return [
        dict(row, site_id=role + row["site_id"][1:], type=role) for row in rows[:count]
    ]


def build_family(kind, *, source_root=SOURCE_ROOT, output_root=None):
    """Write a candidate family; demand CSVs remain byte-for-byte unchanged."""
    if kind not in ("area", "nested"):
        raise ValueError("kind must be area or nested")
    source_root = Path(source_root)
    output_root = Path(output_root or (AREA_ROOT if kind == "area" else NESTED_ROOT))
    if output_root.resolve() == source_root.resolve():
        raise ValueError("Output must not replace the source instances")
    full = instance_directory(120, source_root)
    full_metadata = json.loads((full / "metadata.json").read_text())
    mother = _read_sites(full)
    orders = {
        role: coverage_order(mother, rank)
        for role, rank in (("H", 0), ("L", 1), ("E", 2))
    }
    levels = AREA_LEVELS if kind == "area" else NESTED_LEVELS
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = dict(
        schema_version=1,
        family=kind,
        site_coordinates_are_nested=kind == "nested",
        algorithm="Deterministic farthest-point traversal of existing observed-activity medoids",
        start_rule="Nearest-to-centroid ranks H=0, L=1, E=2; ties by source site ID",
        stable_id_rule="Role prefix plus unchanged source H numeric ID",
        co_location_allowed=True,
        demand_source=str(full),
        demand_file_sha256={name: sha256(full / name) for name in DEMAND_FILES},
        calibration_sha256=sha256(ROOT / "data/model_inputs/calibration.json"),
        levels=[],
    )
    for hubs in levels:
        source = instance_directory(hubs if kind == "area" else 120, source_root)
        source_metadata = json.loads((source / "metadata.json").read_text())
        half_size = source_metadata["data_config"]["map_half_size_km"]
        counts = dict(
            H=hubs, L=max(2, int(0.15 * hubs + 0.5)), E=max(2, int(0.25 * hubs + 0.5))
        )
        if kind == "area":
            area_mother = _read_sites(source)
            selected = _role_sites(area_mother, "H", hubs)
            selected += _role_sites(coverage_order(area_mother, 1), "L", counts["L"])
            selected += _role_sites(coverage_order(area_mother, 2), "E", counts["E"])
        else:
            selected = [
                row
                for role in ("H", "L", "E")
                for row in _role_sites(orders[role], role, counts[role])
            ]
        directory = output_root / f"H{hubs:03d}_L{counts['L']:02d}_E{counts['E']:02d}"
        directory.mkdir(exist_ok=True)
        with (directory / "sites.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(selected[0]))
            writer.writeheader()
            writer.writerows(selected)
        for name in DEMAND_FILES:
            shutil.copyfile(full / name, directory / name)
        metadata = dict(full_metadata)
        metadata.pop("nested_scale", None)
        metadata["schema_version"] = 3
        metadata["candidate_orders_in_area"] = source_metadata[
            "candidate_orders_in_area"
        ]
        metadata["data_config"] = dict(
            full_metadata["data_config"],
            output_dir=str(directory),
            candidate_count=hubs,
            num_bases=counts["L"],
            num_swaps=counts["E"],
        )
        metadata["candidate_generation"] = dict(
            family=kind,
            source_instance=str(source),
            source_sites_sha256=sha256(source / "sites.csv"),
            candidate_half_size_km=half_size,
            candidate_area_km2=(2 * half_size) ** 2,
            counts=counts,
            site_coordinates_are_nested=kind == "nested",
            method=manifest["algorithm"],
            locker_policy="Preserve source coordinates and order"
            if kind == "area"
            else "Prefixes of the H120 farthest-point traversal",
            base_swap_policy="Independent farthest-point prefixes of the same activity-site mother set",
            stable_id_rule=manifest["stable_id_rule"],
            source_site_ids={
                row["site_id"]: "H" + row["site_id"][1:] for row in selected
            },
        )
        metadata["demand_generation"] = dict(
            source_instance=str(full),
            half_size_km=10.0,
            unchanged_files=manifest["demand_file_sha256"],
            od_market_count=full_metadata["od_market_count"],
            total_demand_per_hour=full_metadata["total_demand_per_hour"],
        )
        _write_json(directory / "metadata.json", metadata)
        manifest["levels"].append(
            dict(
                hubs=hubs,
                directory=str(directory),
                counts=counts,
                candidate_half_size_km=half_size,
                sites_sha256=sha256(directory / "sites.csv"),
            )
        )
    _write_json(output_root / "candidate_manifest.json", manifest)
    return manifest


if __name__ == "__main__":
    for family in ("area", "nested"):
        result = build_family(family)
        print(
            json.dumps(
                dict(family=family, counts=[row["counts"] for row in result["levels"]])
            )
        )
