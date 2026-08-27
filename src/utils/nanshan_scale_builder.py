"""Build calibrated, same-center Nanshan geographic scale instances.

Every level is regenerated from the same raw file, dates, and time windows.
Raw orders are retained exactly when both endpoints lie in the centered square.
Facility candidates are generated separately inside each crop by deterministic
weighted K-means followed by mapping each centroid to the nearest observed
active-grid point; demand is never sampled, multiplied, or otherwise rescaled.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.config import DEFAULT_NANSHAN_PATH, DataConfig
from src.data_processor.generator import DataGenerator, NanshanAdapter, RawOrder


CANONICAL_FILES = (
    "metadata.json",
    "sites.csv",
    "od_nodes.csv",
    "od_demand.csv",
    "periods.csv",
)
NESTED_SCALE_SCHEMA_VERSION = 2


@dataclass(frozen=True, order=True)
class NanshanScaleLevel:
    """One concentric square crop and its calibrated candidate targets."""

    half_size_km: float
    lockers: int
    bases: int
    swaps: int

    def validate(self) -> None:
        if not math.isfinite(self.half_size_km) or self.half_size_km <= 0:
            raise ValueError("crop half-size must be positive and finite")
        if self.lockers <= 0 or self.bases <= 0 or self.swaps < 0:
            raise ValueError("H/L targets must be positive and E must be nonnegative")


DEFAULT_NANSHAN_LEVELS = (
    NanshanScaleLevel(3.0, 11, 2, 2),
    NanshanScaleLevel(4.5, 24, 2, 3),
    NanshanScaleLevel(6.0, 43, 3, 5),
    NanshanScaleLevel(7.5, 68, 5, 7),
    NanshanScaleLevel(8.5, 87, 6, 9),
    NanshanScaleLevel(10.0, 120, 8, 13),
)


@dataclass(frozen=True)
class NanshanScaleBuildConfig:
    """Configuration shared by every member of a scale family."""

    output_root: Path = Path("data/instances/nanshan_nested_h120")
    raw_path: Path = Path(DEFAULT_NANSHAN_PATH)
    levels: tuple[NanshanScaleLevel, ...] = DEFAULT_NANSHAN_LEVELS
    candidate_hour_start: int = 8
    candidate_hour_end: int = 24
    demand_hour_start: int = 12
    demand_hour_end: int = 13
    od_grid_km: float = 0.5
    site_grid_km: float = 0.25
    dates: tuple[str, ...] = ("1", "2", "3", "4", "5", "6", "7")
    seed: int = 42

    def validate(self) -> None:
        if not self.levels:
            raise ValueError("levels cannot be empty")
        for level in self.levels:
            level.validate()
        if tuple(sorted(self.levels)) != self.levels:
            raise ValueError("levels must be strictly increasing by crop half-size")
        radii = [level.half_size_km for level in self.levels]
        if len(radii) != len(set(radii)):
            raise ValueError("crop half-sizes must be distinct")
        for field in ("lockers", "bases", "swaps"):
            values = [getattr(level, field) for level in self.levels]
            if any(smaller > larger for smaller, larger in zip(values, values[1:])):
                raise ValueError(
                    f"{field} targets must be nondecreasing with crop size"
                )
        if not self.dates:
            raise ValueError("dates cannot be empty for a reproducible scale family")
        DataConfig(
            source="nanshan",
            raw_path=str(self.raw_path),
            output_dir=str(self.output_root),
            candidate_count=1,
            num_bases=1,
            num_swaps=0,
            candidate_hour_start=self.candidate_hour_start,
            candidate_hour_end=self.candidate_hour_end,
            demand_hour_start=self.demand_hour_start,
            demand_hour_end=self.demand_hour_end,
            map_half_size_km=self.levels[-1].half_size_km,
            od_grid_km=self.od_grid_km,
            site_grid_km=self.site_grid_km,
            dates=self.dates,
            seed=self.seed,
        ).validate()


@dataclass
class _GeneratedLevel:
    level: NanshanScaleLevel
    directory: Path
    metadata: dict[str, Any]
    sites: list[dict[str, str]]
    nodes: list[dict[str, str]]
    demand: list[dict[str, str]]
    periods: list[dict[str, str]]


@dataclass
class _RawOrderAudit:
    candidate_tokens: list[str]
    demand_tokens: list[str]


def build_nanshan_nested_scale(config: NanshanScaleBuildConfig) -> Path:
    """Generate and provenance-check one calibrated nested Nanshan family."""

    config.validate()
    raw_path = config.raw_path.expanduser().resolve()
    if not raw_path.is_file():
        raise FileNotFoundError(f"Nanshan raw data does not exist: {raw_path}")
    output_root = config.output_root.expanduser().resolve()
    _require_empty_root(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    generated: list[_GeneratedLevel] = []
    for level in config.levels:
        level_dir = output_root / _level_name(level)
        data_config = DataConfig(
            source="nanshan",
            raw_path=str(raw_path),
            output_dir=str(level_dir),
            candidate_count=level.lockers,
            num_bases=level.bases,
            num_swaps=level.swaps,
            candidate_hour_start=config.candidate_hour_start,
            candidate_hour_end=config.candidate_hour_end,
            demand_hour_start=config.demand_hour_start,
            demand_hour_end=config.demand_hour_end,
            map_half_size_km=level.half_size_km,
            od_grid_km=config.od_grid_km,
            site_grid_km=config.site_grid_km,
            dates=config.dates,
            seed=config.seed,
        )
        DataGenerator().generate(data_config)
        instance = _read_generated_level(level_dir, level)
        _validate_generated_level(instance, config)
        generated.append(instance)

    _validate_common_provenance(generated, config)
    full = generated[-1]
    center = (
        _finite_float(
            full.metadata.get("projection_center_lon"), "projection center longitude"
        ),
        _finite_float(
            full.metadata.get("projection_center_lat"), "projection center latitude"
        ),
    )
    selected_days = tuple(str(value) for value in full.metadata["selected_days"])
    raw_audits = _audit_raw_orders(raw_path, config, center, selected_days)
    _validate_raw_order_audits(generated, raw_audits)

    family_contract = {
        "source": "nanshan",
        "raw_file_sha256": full.metadata["raw_file_sha256"],
        "projection_center_lon": center[0],
        "projection_center_lat": center[1],
        "selected_days": list(selected_days),
        "candidate_hours": [config.candidate_hour_start, config.candidate_hour_end],
        "demand_hours": [config.demand_hour_start, config.demand_hour_end],
        "candidate_exposure_hours": full.metadata["candidate_exposure_hours"],
        "demand_exposure_hours": full.metadata["demand_exposure_hours"],
        "od_grid_km": config.od_grid_km,
        "site_grid_km": config.site_grid_km,
        "seed": config.seed,
        "levels": [
            {
                "half_size_km": item.level.half_size_km,
                "H": item.level.lockers,
                "L": item.level.bases,
                "E": item.level.swaps,
            }
            for item in generated
        ],
    }
    family_id = hashlib.sha256(
        json.dumps(family_contract, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()[:16]

    hashes_by_radius: dict[float, dict[str, str]] = {}
    full_parent_hashes: dict[str, str] | None = None
    immediate_superset: _GeneratedLevel | None = None
    immediate_superset_hashes: dict[str, str] | None = None
    for index in range(len(generated) - 1, -1, -1):
        instance = generated[index]
        is_full = index == len(generated) - 1
        _annotate_level(
            instance=instance,
            audit=raw_audits[instance.level.half_size_km],
            family_id=family_id,
            level_index=index + 1,
            level_count=len(generated),
            full=full,
            full_parent_hashes=None if is_full else full_parent_hashes,
            immediate_superset=immediate_superset,
            immediate_superset_hashes=immediate_superset_hashes,
        )
        level_hashes = _canonical_hashes(instance.directory)
        hashes_by_radius[instance.level.half_size_km] = level_hashes
        if is_full:
            full_parent_hashes = level_hashes
        immediate_superset = instance
        immediate_superset_hashes = level_hashes

    assert full_parent_hashes is not None
    manifest = {
        "schema_version": NESTED_SCALE_SCHEMA_VERSION,
        "family_id": family_id,
        "design": "same-center nested raw-order crops with per-crop site reclustering",
        "source": "nanshan",
        "full_parent_instance": str(full.directory),
        "full_parent_canonical_file_sha256": full_parent_hashes,
        "family_contract": family_contract,
        "crop_rule": "retain a raw order only when both endpoints are inside [-r,r]^2",
        "site_policy": "recluster_within_crop",
        "site_generation": (
            "deterministic weighted K-means followed by mapping each centroid "
            "to the nearest observed active-grid point"
        ),
        "od_policy": (
            "regenerate canonical OD aggregation from every retained demand-period raw order"
        ),
        "raw_candidate_order_multisets_nested": True,
        "raw_demand_order_multisets_nested": True,
        "orders_sampled": False,
        "demand_rescaled": False,
        "required_solve_demand_scale": 1.0,
        "levels": [
            _level_record(
                instance,
                raw_audits[instance.level.half_size_km],
                hashes_by_radius[instance.level.half_size_km],
                full,
                full_parent_hashes,
            )
            for instance in generated
        ],
    }
    manifest_path = output_root / "scale_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest_path


def _read_generated_level(directory: Path, level: NanshanScaleLevel) -> _GeneratedLevel:
    missing = [name for name in CANONICAL_FILES if not (directory / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"canonical instance {directory} is missing: {', '.join(missing)}"
        )
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError("metadata.json must contain an object")
    return _GeneratedLevel(
        level=level,
        directory=directory,
        metadata=metadata,
        sites=_read_csv(directory / "sites.csv"),
        nodes=_read_csv(directory / "od_nodes.csv"),
        demand=_read_csv(directory / "od_demand.csv"),
        periods=_read_csv(directory / "periods.csv"),
    )


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"{path} has no header")
        return [dict(row) for row in reader]


def _validate_generated_level(
    instance: _GeneratedLevel,
    config: NanshanScaleBuildConfig,
) -> None:
    metadata = instance.metadata
    level = instance.level
    if metadata.get("source") != "nanshan":
        raise ValueError(f"{instance.directory} is not a Nanshan instance")
    if metadata.get("coordinate_system") != "local_km":
        raise ValueError(f"{instance.directory} does not use local_km coordinates")
    data_config = metadata.get("data_config")
    if not isinstance(data_config, Mapping):
        raise ValueError(f"{instance.directory} metadata is missing data_config")
    expected_config: dict[str, object] = {
        "candidate_count": level.lockers,
        "num_bases": level.bases,
        "num_swaps": level.swaps,
        "candidate_hour_start": config.candidate_hour_start,
        "candidate_hour_end": config.candidate_hour_end,
        "demand_hour_start": config.demand_hour_start,
        "demand_hour_end": config.demand_hour_end,
        "map_half_size_km": level.half_size_km,
        "od_grid_km": config.od_grid_km,
        "site_grid_km": config.site_grid_km,
        "seed": config.seed,
        "dates": list(sorted(set(config.dates))),
    }
    mismatches = [
        f"{field}: expected {expected!r}, got {data_config.get(field)!r}"
        for field, expected in expected_config.items()
        if data_config.get(field) != expected
    ]
    if mismatches:
        raise ValueError(
            f"{instance.directory} preprocessing contract mismatch: "
            + "; ".join(mismatches)
        )

    actual_counts = _site_counts(instance.sites)
    expected_counts = {"H": level.lockers, "L": level.bases, "E": level.swaps}
    if actual_counts != expected_counts:
        raise ValueError(
            f"{instance.directory} facility counts mismatch: "
            f"expected={expected_counts}, actual={actual_counts}"
        )
    _validate_rows_inside_crop(instance)
    _validate_demand_totals(instance)
    _validate_site_activity(instance)


def _validate_rows_inside_crop(instance: _GeneratedLevel) -> None:
    radius = instance.level.half_size_km
    for artifact, rows in (
        ("sites.csv", instance.sites),
        ("od_nodes.csv", instance.nodes),
    ):
        for row in rows:
            x = _finite_float(row.get("x_km"), f"{artifact}.x_km")
            y = _finite_float(row.get("y_km"), f"{artifact}.y_km")
            if abs(x) > radius + 1e-8 or abs(y) > radius + 1e-8:
                raise ValueError(
                    f"{artifact} contains a point outside the requested crop"
                )


def _validate_demand_totals(instance: _GeneratedLevel) -> None:
    if len(instance.periods) != 1:
        raise ValueError("nested Nanshan builder requires exactly one period row")
    order_count = sum(
        _nonnegative_int(row.get("order_count"), "OD order_count")
        for row in instance.demand
    )
    total_rate = sum(
        _nonnegative_float(row.get("lambda_per_hour"), "OD lambda_per_hour")
        for row in instance.demand
    )
    for row in instance.demand:
        exposure = _nonnegative_float(row.get("exposure_hours"), "OD exposure_hours")
        if exposure <= 0:
            raise ValueError("OD exposure_hours must be positive")
        count = _nonnegative_int(row.get("order_count"), "OD order_count")
        rate = _nonnegative_float(row.get("lambda_per_hour"), "OD lambda_per_hour")
        if not math.isclose(rate, count / exposure, abs_tol=5e-10):
            raise ValueError("OD rate is inconsistent with order count and exposure")
    comparisons = {
        "period order count": (
            _nonnegative_int(
                instance.periods[0].get("order_count"), "period order_count"
            ),
            order_count,
        ),
        "metadata demand order count": (
            _nonnegative_int(
                instance.metadata.get("demand_orders_in_area"),
                "metadata demand_orders_in_area",
            ),
            order_count,
        ),
    }
    for label, (actual, expected) in comparisons.items():
        if actual != expected:
            raise ValueError(f"{label} does not match canonical OD demand")
    period_rate = _nonnegative_float(
        instance.periods[0].get("total_rate_per_hour"),
        "period total_rate_per_hour",
    )
    metadata_rate = _nonnegative_float(
        instance.metadata.get("total_demand_per_hour"),
        "metadata total_demand_per_hour",
    )
    if not math.isclose(period_rate, total_rate, abs_tol=1e-5) or not math.isclose(
        metadata_rate,
        total_rate,
        abs_tol=1e-5,
    ):
        raise ValueError("period or metadata rate does not match canonical OD demand")


def _validate_site_activity(instance: _GeneratedLevel) -> None:
    exposure = _nonnegative_float(
        instance.metadata.get("candidate_exposure_hours"),
        "candidate exposure",
    )
    candidate_orders = _nonnegative_int(
        instance.metadata.get("candidate_orders_in_area"),
        "candidate orders",
    )
    if exposure <= 0:
        raise ValueError("candidate exposure must be positive")
    activity = {kind: 0.0 for kind in ("H", "L", "E")}
    for row in instance.sites:
        kind = row.get("type")
        if kind not in activity:
            raise ValueError(f"unknown facility type {kind!r}")
        activity[kind] += _nonnegative_float(row.get("activity_rate"), "activity rate")
    expected = {
        "H": 2.0 * candidate_orders / exposure,
        "L": candidate_orders / exposure,
        "E": candidate_orders / exposure if instance.level.swaps else 0.0,
    }
    for kind, expected_rate in expected.items():
        if not math.isclose(activity[kind], expected_rate, abs_tol=2e-6):
            raise ValueError(
                f"{kind} activity does not account for all retained candidate orders"
            )


def _validate_common_provenance(
    generated: Sequence[_GeneratedLevel],
    config: NanshanScaleBuildConfig,
) -> None:
    full = generated[-1]
    invariant_fields = (
        "raw_file_sha256",
        "projection_center_lon",
        "projection_center_lat",
        "selected_days",
        "candidate_exposure_hours",
        "demand_exposure_hours",
        "date_basis",
        "hour_basis",
    )
    for instance in generated[:-1]:
        changed = [
            field
            for field in invariant_fields
            if instance.metadata.get(field) != full.metadata.get(field)
        ]
        if changed:
            raise ValueError(
                f"{instance.directory} differs from the full parent on: {', '.join(changed)}"
            )
    expected_days = list(sorted(set(config.dates)))
    if full.metadata.get("selected_days") != expected_days:
        raise ValueError("generated selected days do not match the requested dates")


def _audit_raw_orders(
    raw_path: Path,
    config: NanshanScaleBuildConfig,
    center: tuple[float, float],
    selected_days: tuple[str, ...],
) -> dict[float, _RawOrderAudit]:
    audits = {
        level.half_size_km: _RawOrderAudit(candidate_tokens=[], demand_tokens=[])
        for level in config.levels
    }
    selected = set(selected_days)
    for order in NanshanAdapter().iter_orders(str(raw_path)):
        if order.day_id not in selected:
            continue
        coordinates = _project_order(order, center)
        token = _order_token(order)
        for level in config.levels:
            if not _both_endpoints_inside(coordinates, level.half_size_km):
                continue
            audit = audits[level.half_size_km]
            if _in_hour_window(
                order.hour,
                config.candidate_hour_start,
                config.candidate_hour_end,
            ):
                audit.candidate_tokens.append(token)
            if _in_hour_window(
                order.hour,
                config.demand_hour_start,
                config.demand_hour_end,
            ):
                audit.demand_tokens.append(token)
    return audits


def _project_order(
    order: RawOrder,
    center: tuple[float, float],
) -> tuple[float, float, float, float]:
    center_lon, center_lat = center
    km_per_lon = 111.32 * math.cos(math.radians(center_lat))
    return (
        (order.pickup_lon - center_lon) * km_per_lon,
        (order.pickup_lat - center_lat) * 111.32,
        (order.dropoff_lon - center_lon) * km_per_lon,
        (order.dropoff_lat - center_lat) * 111.32,
    )


def _both_endpoints_inside(
    coordinates: tuple[float, float, float, float],
    half_size: float,
) -> bool:
    ox, oy, dx, dy = coordinates
    return (
        abs(ox) <= half_size
        and abs(oy) <= half_size
        and abs(dx) <= half_size
        and abs(dy) <= half_size
    )


def _order_token(order: RawOrder) -> str:
    return json.dumps(
        [
            order.order_id,
            order.day_id,
            order.hour,
            order.pickup_lon,
            order.pickup_lat,
            order.dropoff_lon,
            order.dropoff_lat,
        ],
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _validate_raw_order_audits(
    generated: Sequence[_GeneratedLevel],
    audits: Mapping[float, _RawOrderAudit],
) -> None:
    previous_candidate: Counter[str] = Counter()
    previous_demand: Counter[str] = Counter()
    for instance in generated:
        audit = audits[instance.level.half_size_km]
        candidate = Counter(audit.candidate_tokens)
        demand = Counter(audit.demand_tokens)
        if not previous_candidate <= candidate or not previous_demand <= demand:
            raise ValueError(
                "raw order multisets are not nested across geographic crops"
            )
        previous_candidate = candidate
        previous_demand = demand
        expected_counts = {
            "candidate_orders_in_area": len(audit.candidate_tokens),
            "demand_orders_in_area": len(audit.demand_tokens),
        }
        for field, expected in expected_counts.items():
            actual = _nonnegative_int(instance.metadata.get(field), f"metadata {field}")
            if actual != expected:
                raise ValueError(
                    f"{instance.directory} {field}={actual} but raw audit found {expected}"
                )


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


def _site_counts(rows: Sequence[Mapping[str, str]]) -> dict[str, int]:
    return {
        kind: sum(row.get("type") == kind for row in rows) for kind in ("H", "L", "E")
    }


def _multiset_hash(tokens: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for token in sorted(tokens):
        payload = token.encode("utf-8")
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def _canonical_hashes(directory: Path) -> dict[str, str]:
    return {name: _sha256_file(directory / name) for name in CANONICAL_FILES}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite_float(value: object, field: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be finite")
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be finite") from exc
    if not math.isfinite(number):
        raise ValueError(f"{field} must be finite")
    return number


def _nonnegative_float(value: object, field: str) -> float:
    number = _finite_float(value, field)
    if number < 0:
        raise ValueError(f"{field} must be nonnegative")
    return number


def _nonnegative_int(value: object, field: str) -> int:
    try:
        number = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a nonnegative integer") from exc
    if number < 0:
        raise ValueError(f"{field} must be a nonnegative integer")
    return number


def _in_hour_window(hour: int, start: int, end: int) -> bool:
    if end == 24 or start < end:
        return start <= hour < end
    return hour >= start or hour < end


def _level_name(level: NanshanScaleLevel) -> str:
    radius = f"{level.half_size_km:05.2f}".replace(".", "p")
    return f"H{level.lockers:03d}_L{level.bases:02d}_E{level.swaps:02d}_R{radius}"


def _require_empty_root(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty output root: {path}")


def _string_tuple(raw: str) -> tuple[str, ...]:
    values = tuple(item.strip() for item in raw.split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("at least one date is required")
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build calibrated same-center Nanshan crops at 3/4.5/6/7.5/8.5/10 km "
            "with H=11/24/43/68/87/120 and demand_scale=1."
        )
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/instances/nanshan_nested_h120"),
    )
    parser.add_argument("--raw-data", type=Path, default=Path(DEFAULT_NANSHAN_PATH))
    parser.add_argument("--candidate-hours", nargs=2, type=int, default=(8, 24))
    parser.add_argument("--demand-hours", nargs=2, type=int, default=(12, 13))
    parser.add_argument(
        "--dates",
        type=_string_tuple,
        default=("1", "2", "3", "4", "5", "6", "7"),
    )
    parser.add_argument("--od-grid-km", type=float, default=0.5)
    parser.add_argument("--site-grid-km", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = NanshanScaleBuildConfig(
        output_root=args.output_root,
        raw_path=args.raw_data,
        candidate_hour_start=args.candidate_hours[0],
        candidate_hour_end=args.candidate_hours[1],
        demand_hour_start=args.demand_hours[0],
        demand_hour_end=args.demand_hours[1],
        od_grid_km=args.od_grid_km,
        site_grid_km=args.site_grid_km,
        dates=tuple(args.dates),
        seed=args.seed,
    )
    manifest = build_nanshan_nested_scale(config)
    print(f"Nested Nanshan scale manifest: {manifest}")


if __name__ == "__main__":
    main()
