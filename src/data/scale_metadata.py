"""Record canonical scale-family metadata and source hashes."""

from __future__ import annotations
import json
from typing import Any, Mapping
from .scale_config import NESTED_SCALE_SCHEMA_VERSION, _GeneratedLevel, _RawOrderAudit
from .scale_validation import _multiset_hash, _site_counts, _nonnegative_float


def _annotate_level(
    *,
    instance: _GeneratedLevel,
    audit: _RawOrderAudit,
    family_id: str,
    level_index: int,
    level_count: int,
    full: _GeneratedLevel,
    full_parent_hashes: Mapping[str, str] | None,
    immediate_superset: _GeneratedLevel | None,
    immediate_superset_hashes: Mapping[str, str] | None,
) -> None:
    level = instance.level
    full_radius = full.level.half_size_km
    instance.metadata["nested_scale"] = {
        "schema_version": NESTED_SCALE_SCHEMA_VERSION,
        "family_id": family_id,
        "level_index_small_to_large": level_index,
        "level_count": level_count,
        "is_full_parent": instance is full,
        "full_parent_instance": str(full.directory),
        "full_parent_canonical_file_sha256": (
            None if full_parent_hashes is None else dict(full_parent_hashes)
        ),
        "immediate_superset_instance": (
            None if immediate_superset is None else str(immediate_superset.directory)
        ),
        "immediate_superset_canonical_file_sha256": (
            None
            if immediate_superset_hashes is None
            else dict(immediate_superset_hashes)
        ),
        "crop_half_size_km": level.half_size_km,
        "crop_side_km": 2.0 * level.half_size_km,
        "crop_area_km2": (2.0 * level.half_size_km) ** 2,
        "area_fraction_of_full_parent": (level.half_size_km / full_radius) ** 2,
        "crop_bounds_local_km": {
            "x_min": -level.half_size_km,
            "x_max": level.half_size_km,
            "y_min": -level.half_size_km,
            "y_max": level.half_size_km,
        },
        "facility_target_counts": {
            "H": level.lockers,
            "L": level.bases,
            "E": level.swaps,
        },
        "facility_actual_counts": _site_counts(instance.sites),
        "site_policy": "recluster_within_crop",
        "site_generation": (
            "deterministic weighted K-means followed by mapping each centroid "
            "to the nearest observed active-grid point"
        ),
        "site_coordinates_are_nested": False,
        "candidate_order_multiset_sha256": _multiset_hash(audit.candidate_tokens),
        "candidate_order_count": len(audit.candidate_tokens),
        "od_policy": (
            "regenerate canonical OD aggregation from every retained demand-period raw order"
        ),
        "od_identifiers_are_nested": False,
        "raw_demand_orders_are_nested": True,
        "demand_order_multiset_sha256": _multiset_hash(audit.demand_tokens),
        "demand_order_count": len(audit.demand_tokens),
        "orders_sampled": False,
        "demand_rescaled": False,
        "required_solve_demand_scale": 1.0,
    }
    (instance.directory / "metadata.json").write_text(
        json.dumps(instance.metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _level_record(
    instance: _GeneratedLevel,
    audit: _RawOrderAudit,
    canonical_hashes: Mapping[str, str],
    full: _GeneratedLevel,
    full_parent_hashes: Mapping[str, str],
) -> dict[str, Any]:
    metadata = instance.metadata
    radius = instance.level.half_size_km
    return {
        "instance_dir": str(instance.directory),
        "is_full_parent": instance is full,
        "full_parent_instance": str(full.directory),
        "full_parent_canonical_file_sha256": dict(full_parent_hashes),
        "crop_half_size_km": radius,
        "crop_side_km": 2.0 * radius,
        "crop_area_km2": (2.0 * radius) ** 2,
        "area_fraction_of_full_parent": (radius / full.level.half_size_km) ** 2,
        "candidate_counts": _site_counts(instance.sites),
        "candidate_order_count": len(audit.candidate_tokens),
        "candidate_order_multiset_sha256": _multiset_hash(audit.candidate_tokens),
        "od_market_count": len(instance.demand),
        "od_node_count": len(instance.nodes),
        "demand_order_count": len(audit.demand_tokens),
        "demand_order_multiset_sha256": _multiset_hash(audit.demand_tokens),
        "total_demand_per_hour": _nonnegative_float(
            metadata.get("total_demand_per_hour"),
            "metadata total_demand_per_hour",
        ),
        "canonical_file_sha256": dict(canonical_hashes),
    }
