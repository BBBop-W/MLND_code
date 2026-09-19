"""Regenerate the six Nanshan geographic scales from raw orders.

Run ``python -m src.data.scales --help`` for the input and output options.
Each crop reclusters candidate facilities and preserves every retained OD.
"""

from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
from .generation_config import DataConfig
from .generator import DataGenerator
from .scale_config import (
    NanshanScaleLevel,
    NanshanScaleBuildConfig,
    DEFAULT_NANSHAN_PATH,
    NESTED_SCALE_SCHEMA_VERSION,
    _GeneratedLevel,
)
from .scale_validation import (
    _read_generated_level,
    _validate_generated_level,
    _validate_common_provenance,
    _finite_float,
    _audit_raw_orders,
    _validate_raw_order_audits,
    _canonical_hashes,
)
from .scale_metadata import _annotate_level, _level_record


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


def build(raw_path=None, output_root=None, **options):
    """Build the default six-size family, accepting configuration overrides."""
    if raw_path is not None:
        options["raw_path"] = Path(raw_path)
    if output_root is not None:
        options["output_root"] = Path(output_root)
    return build_nanshan_nested_scale(NanshanScaleBuildConfig(**options))


if __name__ == "__main__":
    main()
