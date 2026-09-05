"""Collect and visualize current unified chi-epsilon-energy scale results."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from src.model_contract import FORMULATION_ID as EXPECTED_FORMULATION_ID
from src.model_contract import (
    require_result_contract, require_same_result_schema,
)
from src.model_contract import RESULT_SCHEMA_VERSION as EXPECTED_SCHEMA_VERSION


_SCALE_PARAMETER_FIELDS = (
    "battery_range_km",
    "drone_speed_mps",
    "drone_speed_kmph",
    "usable_battery_time_minutes",
    "swap_service_time_hours",
    "rider_initial_charge_cny",
    "rider_included_distance_km",
    "rider_incremental_cost_per_km",
    "road_distance_factor",
    "feeder_rider_cost_factor",
)

_CANONICAL_INSTANCE_FILES = (
    "metadata.json",
    "sites.csv",
    "od_nodes.csv",
    "od_demand.csv",
    "periods.csv",
)
_NESTED_SCALE_SCHEMA_VERSION = 2
_FORMAL_NANSHAN_SCALE_LEVELS = (
    (11, 2, 2, 3.0),
    (24, 2, 3, 4.5),
    (43, 3, 5, 6.0),
    (68, 5, 7, 7.5),
    (87, 6, 9, 8.5),
    (120, 8, 13, 10.0),
)
_FORMAL_NANSHAN_SCALE_LOCKERS = tuple(
    lockers for lockers, _bases, _swaps, _radius in _FORMAL_NANSHAN_SCALE_LEVELS
)


@dataclass(frozen=True)
class _ScaleRunEvidence:
    result_directory: Path
    result_directory_lockers: int
    instance_directory: Path | None
    metadata: Mapping[str, Any]
    nested_scale: Mapping[str, Any] | None
    canonical_hashes: Mapping[str, Any]
    summary_instance: Mapping[str, Any]


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _number(value: Any, field: str) -> float:
    if value is None or isinstance(value, bool):
        raise ValueError(f"missing finite {field}")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid {field}: {value!r}") from exc
    if not math.isfinite(number):
        raise ValueError(f"nonfinite {field}: {value!r}")
    return number


def _optional_number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _integer(value: Any, field: str) -> int:
    number = _number(value, field)
    integer = round(number)
    if not math.isclose(number, integer, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"{field} must be an integer; got {value!r}")
    return int(integer)


def _required_mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must contain an object")
    return value


def _required_sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{field} must contain a 64-character sha256")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError(f"{field} must contain hexadecimal sha256 text") from exc
    return value


def _same_number(left: Any, right: Any) -> bool:
    try:
        return math.isclose(
            float(left),
            float(right),
            rel_tol=1e-9,
            abs_tol=1e-7,
        )
    except (TypeError, ValueError):
        return False


def _require_current_contract(payload: Mapping[str, Any], artifact: str) -> None:
    require_result_contract(payload, artifact)


def _require_current_summary(summary: Mapping[str, Any]) -> None:
    _require_current_contract(summary, "summary")


def _scale_directories(root: Path) -> list[Path]:
    candidates = [
        path for path in root.glob("H*") if path.is_dir() and path.name[1:].isdigit()
    ]
    return sorted(candidates, key=lambda path: int(path.name[1:]))


def _payload_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _implementation_sha256(run_config: Mapping[str, Any]) -> str:
    value = _mapping(run_config.get("implementation")).get("sha256")
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError("run_config is missing a 64-character implementation sha256")
    return value


def _read_json_object(path: Path, artifact: str) -> Mapping[str, Any]:
    if not path.is_file():
        raise ValueError(f"missing required {artifact}: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read valid {artifact}: {path}: {exc}") from exc
    return _required_mapping(value, artifact)


def _canonical_hash_map(value: Any, artifact: str) -> dict[str, str]:
    mapping = _required_mapping(value, artifact)
    return {
        filename: _required_sha256(
            mapping.get(filename),
            f"{artifact}.{filename}",
        )
        for filename in _CANONICAL_INSTANCE_FILES
    }


def _require_false(payload: Mapping[str, Any], field: str, artifact: str) -> None:
    if payload.get(field) is not False:
        raise ValueError(f"{artifact}.{field} must be false")


def _require_true(payload: Mapping[str, Any], field: str, artifact: str) -> None:
    if payload.get(field) is not True:
        raise ValueError(f"{artifact}.{field} must be true")


def _validate_manifest_contract(
    manifest: Mapping[str, Any],
    baseline: Mapping[str, object],
) -> None:
    contract = _required_mapping(manifest.get("family_contract"), "family_contract")
    direct_expectations = {
        "source": baseline["raw_data_source"],
        "raw_file_sha256": baseline["raw_file_sha256"],
        "selected_days": json.loads(str(baseline["selected_days"])),
    }
    mismatches = [
        field
        for field, expected in direct_expectations.items()
        if contract.get(field) != expected
    ]
    numeric_expectations = {
        "projection_center_lon": baseline["projection_center_lon"],
        "projection_center_lat": baseline["projection_center_lat"],
        "candidate_exposure_hours": baseline["candidate_exposure_hours"],
        "demand_exposure_hours": baseline["demand_exposure_hours"],
        "od_grid_km": baseline["od_grid_km"],
        "site_grid_km": baseline["site_grid_km"],
        "seed": baseline["data_seed"],
    }
    mismatches.extend(
        field
        for field, expected in numeric_expectations.items()
        if not _same_number(contract.get(field), expected)
    )
    expected_windows = {
        "candidate_hours": [
            baseline["candidate_hour_start"],
            baseline["candidate_hour_end"],
        ],
        "demand_hours": [
            baseline["demand_hour_start"],
            baseline["demand_hour_end"],
        ],
    }
    mismatches.extend(
        field
        for field, expected in expected_windows.items()
        if contract.get(field) != expected
    )
    expected_levels = [
        {"half_size_km": radius, "H": h, "L": bases, "E": swaps}
        for h, bases, swaps, radius in _FORMAL_NANSHAN_SCALE_LEVELS
    ]
    if contract.get("levels") != expected_levels:
        mismatches.append("levels")
    if mismatches:
        raise ValueError(
            "scale_manifest family_contract mismatch: " + ", ".join(mismatches)
        )


def _validate_nested_scale_family(
    rows: list[dict[str, object]],
    evidence: list[_ScaleRunEvidence],
) -> None:
    nested_evidence = [item for item in evidence if item.nested_scale is not None]
    if len(nested_evidence) != len(evidence):
        raise ValueError("scale root mixes nested_scale and non-nested run metadata")

    actual_h = tuple(item.result_directory_lockers for item in evidence)
    if actual_h != _FORMAL_NANSHAN_SCALE_LOCKERS:
        raise ValueError(
            "formal nested scale benchmark requires complete H set "
            f"{_FORMAL_NANSHAN_SCALE_LOCKERS}; got {actual_h}"
        )

    instance_directories = [item.instance_directory for item in evidence]
    if any(path is None for path in instance_directories):
        raise ValueError("nested_scale run_config is missing instance.directory")
    manifest_paths = {
        path.parent / "scale_manifest.json"
        for path in instance_directories
        if path is not None
    }
    if len(manifest_paths) != 1:
        raise ValueError(
            "nested_scale runs do not resolve to one common scale_manifest.json"
        )
    manifest_path = next(iter(manifest_paths))
    manifest = _read_json_object(manifest_path, "scale_manifest.json")

    if manifest.get("schema_version") != _NESTED_SCALE_SCHEMA_VERSION:
        raise ValueError(
            "scale_manifest schema_version mismatch: expected "
            f"{_NESTED_SCALE_SCHEMA_VERSION}, got {manifest.get('schema_version')!r}"
        )
    family_id = manifest.get("family_id")
    if not isinstance(family_id, str) or not family_id:
        raise ValueError("scale_manifest family_id must be nonempty text")
    expected_manifest_fields = {
        "source": "nanshan",
        "design": "same-center nested raw-order crops with per-crop site reclustering",
        "site_policy": "recluster_within_crop",
    }
    changed_manifest_fields = [
        field
        for field, expected in expected_manifest_fields.items()
        if manifest.get(field) != expected
    ]
    if changed_manifest_fields:
        raise ValueError(
            "scale_manifest contract mismatch: " + ", ".join(changed_manifest_fields)
        )
    _require_true(
        manifest,
        "raw_candidate_order_multisets_nested",
        "scale_manifest",
    )
    _require_true(
        manifest,
        "raw_demand_order_multisets_nested",
        "scale_manifest",
    )
    _require_false(manifest, "orders_sampled", "scale_manifest")
    _require_false(manifest, "demand_rescaled", "scale_manifest")
    if not _same_number(manifest.get("required_solve_demand_scale"), 1.0):
        raise ValueError("scale_manifest required_solve_demand_scale must equal 1")
    _validate_manifest_contract(manifest, rows[0])

    raw_levels = manifest.get("levels")
    if not isinstance(raw_levels, list):
        raise ValueError("scale_manifest levels must contain a list")
    if len(raw_levels) != len(_FORMAL_NANSHAN_SCALE_LEVELS):
        raise ValueError("scale_manifest must contain exactly six formal nested levels")

    manifest_levels: dict[int, Mapping[str, Any]] = {}
    manifest_schedule: list[tuple[int, int, int, float]] = []
    for index, raw_level in enumerate(raw_levels, start=1):
        level = _required_mapping(raw_level, f"scale_manifest levels[{index}]")
        counts = _required_mapping(
            level.get("candidate_counts"),
            f"scale_manifest levels[{index}].candidate_counts",
        )
        h = _integer(counts.get("H"), f"scale_manifest levels[{index}] H")
        bases = _integer(counts.get("L"), f"scale_manifest levels[{index}] L")
        swaps = _integer(counts.get("E"), f"scale_manifest levels[{index}] E")
        radius = _number(
            level.get("crop_half_size_km"),
            f"scale_manifest levels[{index}] crop_half_size_km",
        )
        if h in manifest_levels:
            raise ValueError(f"scale_manifest contains duplicate H{h} level")
        manifest_levels[h] = level
        manifest_schedule.append((h, bases, swaps, radius))
    if tuple(manifest_schedule) != _FORMAL_NANSHAN_SCALE_LEVELS:
        raise ValueError(
            "scale_manifest levels do not match the formal H/L/E/crop schedule"
        )

    final_level_hashes = _canonical_hash_map(
        manifest_levels[120].get("canonical_file_sha256"),
        "scale_manifest H120 canonical_file_sha256",
    )
    full_parent_hashes = _canonical_hash_map(
        manifest.get("full_parent_canonical_file_sha256"),
        "scale_manifest full_parent_canonical_file_sha256",
    )
    if full_parent_hashes != final_level_hashes:
        raise ValueError("scale_manifest full-parent hashes do not match H120")

    full_parent_value = manifest.get("full_parent_instance")
    if not isinstance(full_parent_value, str) or not full_parent_value:
        raise ValueError("scale_manifest full_parent_instance must be nonempty text")
    full_parent = Path(full_parent_value).expanduser().resolve()

    for index, (row, item) in enumerate(zip(rows, evidence), start=1):
        h, expected_bases, expected_swaps, radius = _FORMAL_NANSHAN_SCALE_LEVELS[
            index - 1
        ]
        level = manifest_levels[h]
        nested = _required_mapping(item.nested_scale, f"H{h} nested_scale")
        if nested.get("schema_version") != _NESTED_SCALE_SCHEMA_VERSION:
            raise ValueError(f"H{h} nested_scale schema_version mismatch")
        if nested.get("family_id") != family_id:
            raise ValueError(f"H{h} nested_scale family_id mismatch")
        if _integer(nested.get("level_count"), f"H{h} level_count") != 6:
            raise ValueError(f"H{h} nested_scale level_count must equal 6")
        if (
            _integer(
                nested.get("level_index_small_to_large"),
                f"H{h} level_index_small_to_large",
            )
            != index
        ):
            raise ValueError(f"H{h} nested_scale level index mismatch")
        if nested.get("is_full_parent") is not (h == 120):
            raise ValueError(f"H{h} nested_scale is_full_parent mismatch")
        if not _same_number(nested.get("crop_half_size_km"), radius):
            raise ValueError(f"H{h} nested_scale crop half-size mismatch")
        if not _same_number(nested.get("crop_side_km"), 2.0 * radius):
            raise ValueError(f"H{h} nested_scale crop side mismatch")
        if not _same_number(nested.get("crop_area_km2"), (2.0 * radius) ** 2):
            raise ValueError(f"H{h} nested_scale crop area mismatch")
        _require_true(nested, "raw_demand_orders_are_nested", f"H{h} nested_scale")
        _require_false(nested, "orders_sampled", f"H{h} nested_scale")
        _require_false(nested, "demand_rescaled", f"H{h} nested_scale")
        if not _same_number(nested.get("required_solve_demand_scale"), 1.0):
            raise ValueError(
                f"H{h} nested_scale required_solve_demand_scale must equal 1"
            )

        target_counts = _required_mapping(
            nested.get("facility_target_counts"),
            f"H{h} nested_scale facility_target_counts",
        )
        actual_counts = _required_mapping(
            nested.get("facility_actual_counts"),
            f"H{h} nested_scale facility_actual_counts",
        )
        expected_counts = {"H": h, "L": expected_bases, "E": expected_swaps}
        for field, expected in expected_counts.items():
            if _integer(target_counts.get(field), f"H{h} target {field}") != expected:
                raise ValueError(f"H{h} nested_scale target {field} mismatch")
            if _integer(actual_counts.get(field), f"H{h} actual {field}") != expected:
                raise ValueError(f"H{h} nested_scale actual {field} mismatch")

        summary_counts = {
            "H": _integer(
                item.summary_instance.get("candidate_lockers"),
                f"H{h} summary candidate lockers",
            ),
            "L": _integer(
                item.summary_instance.get("candidate_bases"),
                f"H{h} summary candidate bases",
            ),
            "E": _integer(
                item.summary_instance.get("candidate_swap_stations"),
                f"H{h} summary candidate swap stations",
            ),
        }
        if summary_counts != expected_counts:
            raise ValueError(f"H{h} summary candidate H/L/E counts mismatch")

        data_config = _required_mapping(
            item.metadata.get("data_config"),
            f"H{h} metadata data_config",
        )
        config_counts = {
            "H": _integer(data_config.get("candidate_count"), f"H{h} candidate_count"),
            "L": _integer(data_config.get("num_bases"), f"H{h} num_bases"),
            "E": _integer(data_config.get("num_swaps"), f"H{h} num_swaps"),
        }
        if config_counts != expected_counts:
            raise ValueError(f"H{h} data_config candidate H/L/E counts mismatch")

        if item.instance_directory is None:
            raise ValueError(f"H{h} run_config is missing instance.directory")
        level_instance = level.get("instance_dir")
        if not isinstance(level_instance, str) or not level_instance:
            raise ValueError(f"scale_manifest H{h} instance_dir must be nonempty text")
        if item.instance_directory != Path(level_instance).expanduser().resolve():
            raise ValueError(f"H{h} instance.directory does not match scale_manifest")
        if h == 120 and item.instance_directory != full_parent:
            raise ValueError("H120 instance.directory does not match full parent")
        nested_parent = nested.get("full_parent_instance")
        if not isinstance(nested_parent, str) or (
            Path(nested_parent).expanduser().resolve() != full_parent
        ):
            raise ValueError(f"H{h} nested_scale full_parent_instance mismatch")

        run_hashes = _canonical_hash_map(
            item.canonical_hashes,
            f"H{h} run_config canonical_file_sha256",
        )
        level_hashes = _canonical_hash_map(
            level.get("canonical_file_sha256"),
            f"scale_manifest H{h} canonical_file_sha256",
        )
        if run_hashes != level_hashes:
            raise ValueError(f"H{h} canonical file hashes do not match scale_manifest")

        for count_field in ("candidate_order_count", "demand_order_count"):
            nested_count = _integer(
                nested.get(count_field),
                f"H{h} nested_scale {count_field}",
            )
            manifest_count = _integer(
                level.get(count_field),
                f"scale_manifest H{h} {count_field}",
            )
            if nested_count != manifest_count:
                raise ValueError(f"H{h} {count_field} does not match scale_manifest")
        for hash_field in (
            "candidate_order_multiset_sha256",
            "demand_order_multiset_sha256",
        ):
            nested_hash = _required_sha256(
                nested.get(hash_field),
                f"H{h} nested_scale {hash_field}",
            )
            manifest_hash = _required_sha256(
                level.get(hash_field),
                f"scale_manifest H{h} {hash_field}",
            )
            if nested_hash != manifest_hash:
                raise ValueError(f"H{h} {hash_field} does not match scale_manifest")
        if not _same_number(level.get("total_demand_per_hour"), row["total_demand"]):
            raise ValueError(f"H{h} demand does not match scale_manifest")


def _validate_scale_invariants(
    rows: list[dict[str, object]],
    evidence: list[_ScaleRunEvidence],
) -> None:
    """Classify scale evidence only after its full provenance contract passes."""

    if not rows:
        return
    if len(rows) != len(evidence):
        raise ValueError("internal scale evidence count mismatch")
    invariant_fields = (
        "summary_schema_version",
        "formulation_source_sha256",
        "implementation_sha256",
        "model_config_sha256",
        "solver_config_sha256",
        "configured_time_limit_seconds",
        "target_mip_gap",
        "raw_data_source",
        "raw_file_sha256",
        "selected_days",
        "period_id",
        "date_basis",
        "hour_basis",
        "candidate_hour_start",
        "candidate_hour_end",
        "demand_hour_start",
        "demand_hour_end",
        "candidate_exposure_hours",
        "demand_exposure_hours",
        "projection_center_lon",
        "projection_center_lat",
        "od_grid_km",
        "site_grid_km",
        "data_seed",
    )
    baseline = rows[0]
    if len(rows) > 1:
        for row in rows[1:]:
            mismatches = [
                field for field in invariant_fields if row[field] != baseline[field]
            ]
            if mismatches:
                raise ValueError(
                    "inconsistent candidate-scale run contract; differing fields: "
                    + ", ".join(mismatches)
                )

        missing_provenance = [
            field
            for field in invariant_fields[5:]
            if baseline.get(field) in (None, "", "[]")
        ]
        if missing_provenance:
            raise ValueError(
                "candidate-scale provenance is incomplete; missing fields: "
                + ", ".join(missing_provenance)
            )

    required_operating_values = {
        "drone_speed_mps": 20.0,
        "battery_range_km": 10.0,
    }
    for field, expected in required_operating_values.items():
        for row in rows:
            actual = float(row[field])
            if not math.isclose(
                actual,
                expected,
                rel_tol=0.0,
                abs_tol=1e-9,
            ):
                raise ValueError(
                    f"candidate-scale runs require {field}={expected:g}; got {actual:g}"
                )

    for row, item in zip(rows, evidence):
        if int(row["candidate_lockers"]) != item.result_directory_lockers:
            raise ValueError(
                f"{item.result_directory.name} directory H does not match "
                f"summary candidate_lockers={row['candidate_lockers']}"
            )

    has_nested = [item.nested_scale is not None for item in evidence]
    if any(has_nested):
        _validate_nested_scale_family(rows, evidence)
    elif len(rows) == 1:
        rows[0]["benchmark_design"] = "single_instance"
        return

    for row in rows:
        if not math.isclose(float(row["demand_scale"]), 1.0, abs_tol=1e-12):
            raise ValueError(
                "candidate-scale runs must use demand_scale=1 without an order multiplier"
            )
        metadata_demand = row.get("metadata_total_demand_per_hour")
        if metadata_demand is None or not math.isclose(
            float(row["total_demand"]),
            float(metadata_demand),
            rel_tol=1e-9,
            abs_tol=1e-7,
        ):
            raise ValueError(
                "reported total demand does not match unscaled metadata demand"
            )

    half_sizes = [row.get("map_half_size_km") for row in rows]
    if any(value is None for value in half_sizes):
        raise ValueError("candidate-scale provenance is missing map_half_size_km")
    normalized_half_sizes = [float(value) for value in half_sizes]
    all_same_map = all(
        math.isclose(value, normalized_half_sizes[0], abs_tol=1e-12)
        for value in normalized_half_sizes[1:]
    )
    if any(has_nested):
        if any(
            right <= left + 1e-12
            for left, right in zip(normalized_half_sizes, normalized_half_sizes[1:])
        ):
            raise ValueError("nested crop half-sizes must increase strictly with H")
        demands = [float(row["total_demand"]) for row in rows]
        if any(right + 1e-7 < left for left, right in zip(demands, demands[1:])):
            raise ValueError(
                "nested geographic crops must have nondecreasing unscaled demand"
            )
        for row in rows:
            row["benchmark_design"] = "nested_geographic_crop"
        return

    if not all_same_map:
        raise ValueError(
            "geographic crop benchmark is missing nested_scale/scale_manifest provenance"
        )
    fixed_map_fields = (
        "od_nodes_sha256",
        "od_demand_sha256",
        "periods_sha256",
        "total_demand",
    )
    for row in rows[1:]:
        mismatches = [
            field for field in fixed_map_fields if row[field] != baseline[field]
        ]
        if mismatches:
            raise ValueError(
                "fixed-map candidate benchmark has inconsistent OD inputs: "
                + ", ".join(mismatches)
            )
    for row in rows:
        row["benchmark_design"] = "fixed_map_candidate_scale"


def collect_scale_results(results_root: str | Path) -> list[dict[str, object]]:
    """Collect current scale runs, failing closed on every discovered artifact."""

    root = Path(results_root).expanduser().resolve()
    run_directories = _scale_directories(root)
    if not run_directories:
        raise ValueError(f"no H<number> scale run directories under {root}")
    rows: list[dict[str, object]] = []
    evidence: list[_ScaleRunEvidence] = []
    for run_dir in run_directories:
        hd_summary_path = run_dir / "HD" / "summary.json"
        mx_summary_path = run_dir / "MX" / "summary.json"
        run_config_path = run_dir / "run_config.json"
        missing = [
            str(path.relative_to(run_dir))
            for path in (hd_summary_path, mx_summary_path, run_config_path)
            if not path.is_file()
        ]
        if missing:
            raise ValueError(
                f"{run_dir.name}: missing required scale artifact(s): "
                + ", ".join(missing)
            )
        try:
            hd_summary = json.loads(hd_summary_path.read_text(encoding="utf-8"))
            summary = json.loads(mx_summary_path.read_text(encoding="utf-8"))
            run_config = json.loads(run_config_path.read_text(encoding="utf-8"))
            if (
                not isinstance(hd_summary, Mapping)
                or not isinstance(summary, Mapping)
                or not isinstance(run_config, Mapping)
            ):
                raise ValueError("top-level JSON value is not an object")
            _require_current_summary(hd_summary)
            _require_current_summary(summary)
            _require_current_contract(run_config, "run_config")
            require_same_result_schema(run_config, hd_summary, "HD summary")
            require_same_result_schema(run_config, summary, "MX summary")
            if run_config.get("modes") != ["HD", "MX"]:
                raise ValueError(
                    "scale run_config.modes must equal ['HD', 'MX']; "
                    f"got {run_config.get('modes')!r}"
                )
            if (
                run_config.get("cross_mode_start_policy")
                != "HD_to_later_MX_by_variable_name"
            ):
                raise ValueError(
                    "scale run_config.cross_mode_start_policy must equal "
                    "'HD_to_later_MX_by_variable_name'"
                )
            if run_config.get("warm_start_from") is not None:
                raise ValueError("scale run_config.warm_start_from must be null")
            if hd_summary.get("mode") != "HD":
                raise ValueError(
                    f"scale HD summary mode must be 'HD'; got {hd_summary.get('mode')!r}"
                )
            if summary.get("mode") != "MX":
                raise ValueError(
                    f"scale summary mode must be 'MX'; got {summary.get('mode')!r}"
                )
            run_signature_sha256 = _required_sha256(
                run_config.get("run_signature_sha256"),
                "run_config.run_signature_sha256",
            )
            hd_run_signature_sha256 = _required_sha256(
                hd_summary.get("run_signature_sha256"),
                "HD summary.run_signature_sha256",
            )
            mx_run_signature_sha256 = _required_sha256(
                summary.get("run_signature_sha256"),
                "MX summary.run_signature_sha256",
            )
            if hd_run_signature_sha256 != run_signature_sha256:
                raise ValueError("HD summary run signature disagrees with run_config")
            if mx_run_signature_sha256 != run_signature_sha256:
                raise ValueError("MX summary run signature disagrees with run_config")

            hd_solver = _required_mapping(hd_summary.get("solver"), "HD summary solver")
            solver = _required_mapping(summary.get("solver"), "MX summary solver")
            if hd_solver.get("external_start_installed") is not False:
                raise ValueError(
                    "scale HD summary must record external_start_installed=false"
                )
            if solver.get("external_start_installed") is not True:
                raise ValueError(
                    "scale MX summary must record external_start_installed=true"
                )
            if solver.get("external_start_source") != "HD":
                raise ValueError("scale MX external_start_source must equal 'HD'")
            matched_variables = _integer(
                solver.get("external_start_matched_variables"),
                "MX external-start matched variables",
            )
            if matched_variables <= 0:
                raise ValueError(
                    "scale MX external_start_matched_variables must be positive"
                )
            solver_config = _mapping(run_config.get("solver"))
            model_config = _mapping(run_config.get("model"))
            model_derived = _mapping(run_config.get("model_derived"))
            formulation_source = _required_mapping(
                run_config.get("formulation_source"),
                "run_config formulation_source",
            )
            formulation_source_sha256 = _required_sha256(
                formulation_source.get("sha256"),
                "run_config formulation_source.sha256",
            )
            instance_config = _required_mapping(
                run_config.get("instance"), "run_config instance"
            )
            metadata = _required_mapping(
                instance_config.get("metadata"), "run_config instance metadata"
            )
            data_config = _mapping(metadata.get("data_config"))
            canonical_hashes = _mapping(instance_config.get("canonical_file_sha256"))
            nested_raw = metadata.get("nested_scale")
            nested_scale = (
                None
                if nested_raw is None
                else _required_mapping(nested_raw, "metadata nested_scale")
            )
            instance_directory_raw = instance_config.get("directory")
            if instance_directory_raw is None:
                instance_directory = None
            elif isinstance(instance_directory_raw, str) and instance_directory_raw:
                instance_directory = Path(instance_directory_raw).expanduser().resolve()
            else:
                raise ValueError("run_config instance.directory must be nonempty text")
            facility_summary = _mapping(summary.get("facilities"))
            facilities = _mapping(facility_summary.get("open"))
            funnel = _mapping(summary.get("demand_funnel_orders_per_hour"))
            instance = _mapping(summary.get("instance"))
            model_size = _mapping(summary.get("model_size"))
            discrete = _mapping(summary.get("discrete_availability"))
            energy = _mapping(summary.get("energy_balance"))
            fleet = _mapping(summary.get("fleet"))
            queue = _mapping(summary.get("locker_queue"))
            flights = _mapping(summary.get("flight_operations"))
            chaining = _mapping(summary.get("chaining"))

            open_bases = _integer(facilities.get("bases"), "open bases")
            open_lockers = _integer(facilities.get("lockers"), "open lockers")
            open_swap_stations = _integer(
                facilities.get("swap_stations"),
                "open swap stations",
            )
            active_lockers = _integer(
                summary.get("active_lockers"),
                "active lockers",
            )
            nested_active_lockers = _integer(
                facility_summary.get("active_lockers"),
                "facilities active lockers",
            )
            idle_open_lockers = _integer(
                summary.get("idle_open_lockers"),
                "idle open lockers",
            )
            nested_idle_open_lockers = _integer(
                facility_summary.get("idle_open_lockers"),
                "facilities idle open lockers",
            )
            if active_lockers != nested_active_lockers:
                raise ValueError(
                    "top-level active lockers disagree with facilities summary"
                )
            if idle_open_lockers != nested_idle_open_lockers:
                raise ValueError(
                    "top-level idle open lockers disagree with facilities summary"
                )
            if not 0 <= active_lockers <= open_lockers:
                raise ValueError(
                    "active_lockers must satisfy 0 <= active <= open lockers"
                )
            if idle_open_lockers != open_lockers - active_lockers:
                raise ValueError(
                    "idle_open_lockers must equal open_lockers - active_lockers"
                )
            network_opened = any(
                count > 0 for count in (open_bases, open_lockers, open_swap_stations)
            )
            if summary.get("network_opened") is not network_opened:
                raise ValueError("network_opened flag disagrees with open counts")

            target_gap = _number(solver_config.get("mip_gap"), "solver target gap")
            time_limit = _number(solver_config.get("time_limit"), "solver time limit")
            if time_limit <= 0.0:
                raise ValueError("solver time limit must be positive")
            hd_runtime = _number(
                hd_solver.get("runtime_seconds"), "HD solver runtime"
            )
            mx_runtime = _number(solver.get("runtime_seconds"), "MX solver runtime")
            end_to_end_runtime = hd_runtime + mx_runtime
            gap = _optional_number(solver.get("mip_gap"))

            objective = _number(
                solver.get("objective_cost_per_hour", summary.get("objective")),
                "objective",
            )
            baseline = _number(
                summary.get("courier_only_baseline"),
                "courier-only baseline",
            )
            cost_saving = _optional_number(summary.get("cost_saving"))
            if cost_saving is None:
                cost_saving = baseline - objective
            cost_saving_percent = _optional_number(summary.get("cost_saving_percent"))
            if cost_saving_percent is None:
                cost_saving_percent = (
                    100.0 * cost_saving / baseline if abs(baseline) > 1e-12 else 0.0
                )

            total_demand = _number(funnel.get("total_demand"), "total_demand")
            offered = _number(funnel.get("total_drone_offered"), "total_drone_offered")
            realized = _number(
                funnel.get("total_drone_realized"), "total_drone_realized"
            )
            positive_drone_service = realized > 1e-7
            if summary.get("network_is_nontrivial") is not positive_drone_service:
                raise ValueError(
                    "network_is_nontrivial must mean positive realized drone service"
                )
            locker_pickup = _number(funnel.get("locker_pickup"), "locker_pickup")
            assisted_home = _number(funnel.get("assisted_home"), "assisted_home")
            courier_fallback = _number(
                funnel.get("direct_courier_fallback"),
                "direct_courier_fallback",
            )
            service_error = (
                locker_pickup + assisted_home + courier_fallback - total_demand
            )
            service_fraction = _optional_number(
                funnel.get("drone_service_fraction_of_demand")
            )
            if service_fraction is None:
                service_fraction = (
                    realized / total_demand if total_demand > 0.0 else 0.0
                )
            energy_violation = _number(
                energy.get("maximum_constraint_violation"),
                "maximum energy constraint violation",
            )
            if energy_violation > 2e-6:
                raise ValueError(
                    "maximum energy constraint violation exceeds tolerance: "
                    f"{energy_violation:g}"
                )
            model_config_sha256 = _payload_sha256(model_config)
            solver_config_sha256 = _payload_sha256(solver_config)
            comparable_hashes = {}
            for filename in ("od_nodes.csv", "od_demand.csv", "periods.csv"):
                comparable_hashes[filename] = _required_sha256(
                    canonical_hashes.get(filename),
                    f"run_config canonical_file_sha256.{filename}",
                )
            parameter_values: dict[str, float] = {}
            for field in _SCALE_PARAMETER_FIELDS:
                raw_value = model_config.get(field, model_derived.get(field))
                if field == "feeder_rider_cost_factor" and raw_value is None:
                    # Schema-3 runs created before this explicit parameter used
                    # the same tariff for direct and feeder courier legs.
                    raw_value = 1.0
                parameter_values[field] = _number(raw_value, field)
            demand_scale = _number(run_config.get("demand_scale"), "demand scale")
            map_half_size = _optional_number(data_config.get("map_half_size_km"))
            map_side = 2.0 * map_half_size if map_half_size is not None else None
            map_area = map_side**2 if map_side is not None else None
            density = total_demand / map_area if map_area not in (None, 0.0) else None
            raw_file_sha256 = metadata.get("raw_file_sha256")
            if raw_file_sha256 is not None:
                raw_file_sha256 = _required_sha256(
                    raw_file_sha256,
                    "metadata raw_file_sha256",
                )
            selected_days = metadata.get("selected_days")
            selected_days_json = (
                json.dumps(selected_days, ensure_ascii=False, separators=(",", ":"))
                if isinstance(selected_days, list)
                else None
            )
            period_start = _optional_number(data_config.get("demand_hour_start"))
            period_end = _optional_number(data_config.get("demand_hour_end"))
            period_id = (
                f"h{int(period_start):02d}_{int(period_end):02d}"
                if period_start is not None and period_end is not None
                else None
            )

            rho_by_facility: dict[str, float] = {}
            facilities_path = run_dir / "MX" / "open_facilities.csv"
            if facilities_path.is_file():
                with facilities_path.open(newline="", encoding="utf-8") as handle:
                    for facility in csv.DictReader(handle):
                        rho = _optional_number(facility.get("rho"))
                        if rho is not None:
                            rho_by_facility[str(facility["facility"])] = rho

            rows.append(
                {
                    "formulation_id": EXPECTED_FORMULATION_ID,
                    "summary_schema_version": summary["schema_version"],
                    "run_signature_sha256": run_signature_sha256,
                    "formulation_source_sha256": formulation_source_sha256,
                    "implementation_sha256": _implementation_sha256(run_config),
                    "model_config_sha256": model_config_sha256,
                    "solver_config_sha256": solver_config_sha256,
                    "od_nodes_sha256": comparable_hashes["od_nodes.csv"],
                    "od_demand_sha256": comparable_hashes["od_demand.csv"],
                    "periods_sha256": comparable_hashes["periods.csv"],
                    **parameter_values,
                    "scale_family_id": (
                        nested_scale.get("family_id")
                        if nested_scale is not None
                        else None
                    ),
                    "scale_level_index": (
                        nested_scale.get("level_index_small_to_large")
                        if nested_scale is not None
                        else None
                    ),
                    "orders_sampled": (
                        nested_scale.get("orders_sampled")
                        if nested_scale is not None
                        else None
                    ),
                    "demand_rescaled": (
                        nested_scale.get("demand_rescaled")
                        if nested_scale is not None
                        else None
                    ),
                    "data_source": str(instance.get("data_source") or "unknown"),
                    "raw_data_source": metadata.get("source"),
                    "raw_file_sha256": raw_file_sha256,
                    "selected_days": selected_days_json,
                    "period_id": period_id,
                    "date_basis": metadata.get("date_basis"),
                    "hour_basis": metadata.get("hour_basis"),
                    "candidate_hour_start": data_config.get("candidate_hour_start"),
                    "candidate_hour_end": data_config.get("candidate_hour_end"),
                    "demand_hour_start": data_config.get("demand_hour_start"),
                    "demand_hour_end": data_config.get("demand_hour_end"),
                    "candidate_exposure_hours": metadata.get(
                        "candidate_exposure_hours"
                    ),
                    "demand_exposure_hours": metadata.get("demand_exposure_hours"),
                    "projection_center_lon": metadata.get("projection_center_lon"),
                    "projection_center_lat": metadata.get("projection_center_lat"),
                    "od_grid_km": data_config.get("od_grid_km"),
                    "site_grid_km": data_config.get("site_grid_km"),
                    "data_seed": data_config.get("seed"),
                    "demand_scale": demand_scale,
                    "metadata_total_demand_per_hour": _optional_number(
                        metadata.get("total_demand_per_hour")
                    ),
                    "map_half_size_km": map_half_size,
                    "map_side_km": map_side,
                    "map_area_sq_km": map_area,
                    "demand_density_orders_per_hour_sq_km": density,
                    "display_origin_choices": model_config.get(
                        "display_origin_choices"
                    ),
                    "general_corridor_choices": model_config.get(
                        "general_corridor_choices"
                    ),
                    "solver_threads": int(
                        round(_number(solver_config.get("threads"), "solver threads"))
                    ),
                    "solver_seed": int(
                        round(_number(solver_config.get("seed"), "solver seed"))
                    ),
                    "candidate_lockers": _integer(
                        instance.get("candidate_lockers"), "candidate lockers"
                    ),
                    "screening_is_exact": bool(
                        summary.get("screening_is_exact", False)
                    ),
                    "status": solver.get("status_name", summary.get("status_name", "")),
                    "runtime_seconds": mx_runtime,
                    "hd_runtime_seconds": hd_runtime,
                    "mx_runtime_seconds": mx_runtime,
                    "end_to_end_runtime_seconds": end_to_end_runtime,
                    "configured_time_limit_seconds": time_limit,
                    "configured_end_to_end_budget_seconds": 2.0 * time_limit,
                    "runtime_over_limit_ratio": mx_runtime / time_limit,
                    "mx_runtime_over_per_mode_budget_ratio": mx_runtime / time_limit,
                    "end_to_end_runtime_over_configured_budget_ratio": (
                        end_to_end_runtime / (2.0 * time_limit)
                    ),
                    "objective_cny_per_hour": objective,
                    "best_bound_cny_per_hour": _optional_number(
                        solver.get("objective_bound")
                    ),
                    "courier_only_baseline_cny_per_hour": baseline,
                    "cost_saving_cny_per_hour": cost_saving,
                    "cost_saving_percent": cost_saving_percent,
                    "mip_gap": gap,
                    "target_mip_gap": target_gap,
                    "meets_target_gap": (gap is not None and gap <= target_gap + 1e-9),
                    "open_bases": open_bases,
                    "open_lockers": open_lockers,
                    "active_lockers": active_lockers,
                    "idle_open_lockers": idle_open_lockers,
                    "open_swap_stations": open_swap_stations,
                    "active_connections": int(
                        round(
                            _number(
                                facility_summary.get("active_connections"),
                                "active connections",
                            )
                        )
                    ),
                    "fleet_size_drones": _number(
                        fleet.get("gamma_drones", summary.get("fleet_size")),
                        "fleet size",
                    ),
                    "maximum_locker_utilization": _number(
                        queue.get("maximum_utilization"),
                        "maximum locker utilization",
                    ),
                    "total_flight_km_per_hour": _number(
                        flights.get("total_flight_km_per_hour"),
                        "total flight kilometres per hour",
                    ),
                    "swap_flow_share": _optional_number(flights.get("all_swap_share")),
                    "chaining_order_fraction": _optional_number(
                        chaining.get("chaining_order_fraction")
                    ),
                    "offered_flow_weighted_rho": _optional_number(
                        discrete.get("offered_flow_weighted_rho")
                    ),
                    "selected_rho_by_facility": dict(sorted(rho_by_facility.items())),
                    "active_virtual_states": int(
                        round(
                            _number(
                                discrete.get("active_virtual_state_count"),
                                "active virtual states",
                            )
                        )
                    ),
                    "total_demand": total_demand,
                    "total_drone_offered": offered,
                    "total_drone_realized": realized,
                    "drone_service_fraction": service_fraction,
                    "drone_service_percent": 100.0 * service_fraction,
                    "locker_pickup": locker_pickup,
                    "assisted_home": assisted_home,
                    "direct_courier_fallback": courier_fallback,
                    "service_accounting_error": service_error,
                    "maximum_energy_constraint_violation": energy_violation,
                    "positive_drone_service": positive_drone_service,
                    "network_opened": network_opened,
                    "network_is_nontrivial": positive_drone_service,
                    "model_variables": int(
                        round(_number(model_size.get("variables"), "model variables"))
                    ),
                    "binary_variables": int(
                        round(
                            _number(
                                model_size.get("binary_variables"),
                                "binary variables",
                            )
                        )
                    ),
                    "chi_variables": int(
                        round(_number(instance.get("chi_keys"), "chi variables"))
                    ),
                    "linear_constraints": int(
                        round(
                            _number(
                                model_size.get("linear_constraints"),
                                "linear constraints",
                            )
                        )
                    ),
                    "quadratic_constraints": int(
                        round(
                            _number(
                                model_size.get("quadratic_constraints"),
                                "quadratic constraints",
                            )
                        )
                    ),
                }
            )
            evidence.append(
                _ScaleRunEvidence(
                    result_directory=run_dir,
                    result_directory_lockers=int(run_dir.name[1:]),
                    instance_directory=instance_directory,
                    metadata=metadata,
                    nested_scale=nested_scale,
                    canonical_hashes=canonical_hashes,
                    summary_instance=instance,
                )
            )
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
            raise ValueError(f"{run_dir.name}: {exc}") from exc
    _validate_scale_invariants(rows, evidence)
    return rows


def _choice_count_label(value: object) -> str:
    if value is None:
        return "all"
    return f"top-{int(value)}"


def _benchmark_context(rows: list[dict[str, object]]) -> str:
    """Describe the recorded run contract without assuming local software."""

    first = rows[0]
    if all(bool(row["screening_is_exact"]) for row in rows):
        index_label = "complete index sets"
    else:
        origins = _choice_count_label(first["display_origin_choices"])
        corridors = _choice_count_label(first["general_corridor_choices"])
        index_label = f"screened origins {origins} / corridors {corridors}"
    threads = int(first["solver_threads"])
    thread_label = "solver-default threads" if threads == 0 else f"{threads} threads"
    limit = float(first["configured_time_limit_seconds"])
    target = 100.0 * float(first["target_mip_gap"])
    return (
        f"Schema {first['summary_schema_version']} · {index_label} · {thread_label} · "
        f"seed {int(first['solver_seed'])} · {limit:.0f}s configured budget per mode · "
        f"{2.0 * limit:.0f}s configured HD→MX end-to-end budget "
        "(barrier timeout checks may delay wall-clock return) · "
        f"target gap {target:g}% · demand scale {float(first['demand_scale']):g}"
    )


def _scale_design_note(rows: list[dict[str, object]]) -> str:
    design = rows[0]["benchmark_design"]
    if design == "nested_geographic_crop":
        return (
            "Concentric square crops use the same raw file, dates, time windows, "
            "center, model, and solver; unscaled demand grows with covered area."
        )
    if design == "fixed_map_candidate_scale":
        return (
            "Legacy fixed-map benchmark: OD demand and map extent are identical; "
            "only the recorded candidate-instance scale changes."
        )
    return "Single recorded instance; no cross-scale geographic claim is made."


def _network_mix_note(rows: list[dict[str, object]]) -> str:
    opened = sum(bool(row["network_opened"]) for row in rows)
    service = sum(bool(row["positive_drone_service"]) for row in rows)
    if opened == 0:
        return "No incumbent opens drone facilities; all report zero positive drone service."
    if opened == len(rows) and service == len(rows):
        return (
            "Every incumbent opens facilities and provides positive drone service; "
            "exact metrics remain in benchmark_summary.csv."
        )
    return (
        f"Facilities open in {opened}/{len(rows)} incumbents; positive drone service "
        f"occurs in {service}/{len(rows)}."
    )


def _scale_tick_labels(rows: list[dict[str, object]]) -> list[str]:
    """Keep repeated panel ticks compact; row-level context stays in the CSV."""

    is_crop = rows[0]["benchmark_design"] == "nested_geographic_crop"
    labels: list[str] = []
    for row in rows:
        model_variables = float(row["model_variables"]) / 1_000.0
        if is_crop:
            labels.append(
                f"H{row['candidate_lockers']}\n"
                f"{float(row['map_side_km']):g} km\n"
                f"{model_variables:.0f}k vars"
            )
        else:
            labels.append(f"H{row['candidate_lockers']}\n{model_variables:.0f}k vars")
    return labels


def _scale_data_note(rows: list[dict[str, object]]) -> str:
    """Summarize crop context once instead of repeating it under every bar."""

    if rows[0]["benchmark_design"] != "nested_geographic_crop":
        return "Ticks show candidate lockers and model size; exact rows: benchmark_summary.csv."
    areas = [float(row["map_area_sq_km"]) for row in rows]
    demands = [float(row["total_demand"]) for row in rows]
    densities = [float(row["demand_density_orders_per_hour_sq_km"]) for row in rows]
    return (
        "Ticks: candidate H / crop side / model size.  "
        f"Area {min(areas):g}–{max(areas):g} km² · "
        f"demand {min(demands):,.0f}–{max(demands):,.0f} orders/h · "
        f"density {min(densities):.1f}–{max(densities):.1f} orders/h/km².  "
        "Exact rows: benchmark_summary.csv."
    )


def _service_bar_label(row: Mapping[str, object], value: float) -> str:
    """Use full network detail only where a network is actually active."""

    if not bool(row["network_opened"]):
        return f"{value:.0f}%"
    fleet_size = float(row["fleet_size_drones"])
    utilization = 100.0 * float(row["maximum_locker_utilization"])
    return (
        f"{value:.1f}%\n"
        f"L/H/E {row['open_bases']}/{row['open_lockers']}/"
        f"{row['open_swap_stations']}\n"
        f"active H {row['active_lockers']} · fleet {fleet_size:g}\n"
        f"util {utilization:.0f}%"
    )


def write_scale_report(results_root: str | Path) -> list[dict[str, object]]:
    root = Path(results_root).expanduser().resolve()
    rows = collect_scale_results(root)
    if not rows:
        raise ValueError(
            f"no compatible scale runs under {root}; expected "
            f"schema_version={EXPECTED_SCHEMA_VERSION} "
            f"and formulation_id={EXPECTED_FORMULATION_ID!r}"
        )

    fields = list(rows[0])
    with (root / "benchmark_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            payload = dict(row)
            payload["selected_rho_by_facility"] = json.dumps(
                payload["selected_rho_by_facility"], sort_keys=True
            )
            writer.writerow(payload)

    is_crop = rows[0]["benchmark_design"] == "nested_geographic_crop"
    labels = _scale_tick_labels(rows)
    positions = list(range(len(rows)))
    hd_runtime = [float(row["hd_runtime_seconds"]) for row in rows]
    mx_runtime = [float(row["mx_runtime_seconds"]) for row in rows]
    end_to_end_runtime = [
        float(row["end_to_end_runtime_seconds"]) for row in rows
    ]
    gaps = [
        100.0 * float(row["mip_gap"]) if row["mip_gap"] is not None else math.nan
        for row in rows
    ]
    finite_gaps = [gap for gap in gaps if not math.isnan(gap)]
    limit = float(rows[0]["configured_time_limit_seconds"])
    target_gap_percent = 100.0 * float(rows[0]["target_mip_gap"])
    solved_color = "#2563eb"
    unsolved_color = "#d97706"
    bar_colors = [
        solved_color if row["meets_target_gap"] else unsolved_color for row in rows
    ]
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(max(14.8, 1.55 * len(rows) + 5.5), 10.4),
    )
    runtime_ax, gap_ax = axes[0]
    saving_ax, service_ax = axes[1]
    hd_runtime_bars = runtime_ax.bar(
        positions,
        hd_runtime,
        color="#bfdbfe",
        edgecolor="#1d4ed8",
        linewidth=0.8,
        width=0.68,
        label="HD solve",
    )
    mx_runtime_bars = runtime_ax.bar(
        positions,
        mx_runtime,
        bottom=hd_runtime,
        color=bar_colors,
        edgecolor="#334155",
        linewidth=0.8,
        width=0.68,
        label="MX solve after HD start",
    )
    for hd_bar, mx_bar, row, hd_time, mx_time, total_time in zip(
        hd_runtime_bars,
        mx_runtime_bars,
        rows,
        hd_runtime,
        mx_runtime,
        end_to_end_runtime,
    ):
        if not row["meets_target_gap"]:
            mx_bar.set_hatch("//")
        runtime_ax.text(
            hd_bar.get_x() + hd_bar.get_width() / 2,
            total_time + max(0.012 * max(end_to_end_runtime), 1.0),
            f"{total_time:.1f}s\nHD {hd_time:.1f} + MX {mx_time:.1f}",
            ha="center",
            va="bottom",
            fontsize=7.6,
            color="#334155",
        )
    end_to_end_budget = 2.0 * limit
    runtime_ax.axhline(
        end_to_end_budget,
        color="#64748b",
        linestyle="--",
        linewidth=1.2,
    )
    runtime_ax.text(
        0.985,
        end_to_end_budget,
        f"{end_to_end_budget:.0f}s configured HD→MX budget ({limit:.0f}s/mode)",
        transform=runtime_ax.get_yaxis_transform(),
        ha="right",
        va="bottom",
        fontsize=8.1,
        color="#64748b",
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.2},
    )
    runtime_ax.set_ylabel("Algorithm wall-clock time (seconds)")
    runtime_ax.set_title("End-to-end HD→MX runtime", loc="left", fontsize=11)
    runtime_ax.set_ylim(
        0.0,
        max(end_to_end_budget * 1.18, max(end_to_end_runtime) * 1.22),
    )
    runtime_ax.legend(loc="upper left", frameon=False, fontsize=8.0)

    gap_bars = gap_ax.bar(
        positions,
        [0.0 if math.isnan(gap) else gap for gap in gaps],
        color="#bfdbfe",
        edgecolor="#1d4ed8",
        linewidth=1.0,
        width=0.68,
    )
    gap_label_offset = max(
        0.012 * max(finite_gaps or [target_gap_percent]),
        0.12,
    )
    for bar, gap in zip(gap_bars, gaps):
        if math.isnan(gap):
            bar.set_height(0.0)
        gap_ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + gap_label_offset,
            "N/A" if math.isnan(gap) else f"{gap:.2f}%",
            ha="center",
            va="bottom",
            fontsize=8.2,
            color="#334155",
        )
    gap_ax.axhline(
        target_gap_percent,
        color="#d97706",
        linestyle="--",
        linewidth=1.2,
    )
    gap_ax.text(
        0.985,
        target_gap_percent,
        f"target {target_gap_percent:.1f}%",
        transform=gap_ax.get_yaxis_transform(),
        ha="right",
        va="bottom",
        fontsize=8.1,
        color="#9a3412",
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.2},
    )
    gap_ax.set_ylabel("Final MIP gap (%)")
    gap_ax.set_title("Final proof gap", loc="left", fontsize=11)
    gap_ax.set_ylim(
        0.0,
        max(target_gap_percent * 1.3, max(finite_gaps, default=0.0) * 1.16),
    )

    savings = [float(row["cost_saving_percent"]) for row in rows]
    saving_colors = [
        "#2563eb" if value > 1e-9 else "#d97706" if value < -1e-9 else "#cbd5e1"
        for value in savings
    ]
    saving_bars = saving_ax.bar(
        positions,
        savings,
        color=saving_colors,
        edgecolor="#334155",
        linewidth=0.8,
        width=0.68,
    )
    saving_ax.axhline(0.0, color="#64748b", linewidth=0.9)
    saving_span = max(max(savings, default=0.0) - min(savings, default=0.0), 1.0)
    for bar, value in zip(saving_bars, savings):
        saving_ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + (0.035 * saving_span if value >= 0.0 else -0.035 * saving_span),
            f"{value:.2f}%",
            ha="center",
            va="bottom" if value >= 0.0 else "top",
            fontsize=8.3,
            color="#334155",
        )
    saving_ax.set_ylim(
        min(-0.20, min(savings, default=0.0) - 0.14 * saving_span),
        max(0.80, max(savings, default=0.0) + 0.20 * saving_span),
    )
    saving_ax.set_ylabel("Saving vs. courier-only baseline (%)")
    saving_ax.set_title("Incumbent cost saving", loc="left", fontsize=11)

    service = [float(row["drone_service_percent"]) for row in rows]
    service_colors = [
        "#2563eb" if bool(row["positive_drone_service"]) else "#cbd5e1" for row in rows
    ]
    service_bars = service_ax.bar(
        positions,
        service,
        color=service_colors,
        edgecolor="#334155",
        linewidth=0.8,
        width=0.68,
    )
    service_ceiling = max(12.0, max(service, default=0.0) * 1.75 + 4.0)
    for index, (bar, row, value) in enumerate(zip(service_bars, rows, service)):
        bar_center = bar.get_x() + bar.get_width() / 2
        label_y = max(
            value + 0.025 * service_ceiling,
            service_ceiling * (0.56 if index % 2 == 0 else 0.77),
        )
        service_ax.annotate(
            _service_bar_label(row, value),
            xy=(bar_center, value),
            xytext=(bar_center, label_y),
            ha="center",
            va="bottom",
            fontsize=7.2,
            color="#334155",
            linespacing=1.08,
            arrowprops={
                "arrowstyle": "-",
                "color": "#94a3b8",
                "linewidth": 0.65,
            },
        )
    service_ax.set_ylim(0.0, service_ceiling)
    service_ax.set_ylabel("Demand realized through drone network (%)")
    service_ax.set_title(
        "Positive drone service and opened facilities", loc="left", fontsize=11
    )

    source = str(rows[0]["data_source"]).strip().title() or "Recorded"
    design_title = (
        "nested geographic-crop benchmark"
        if is_crop
        else "fixed-map candidate-scale benchmark"
        if rows[0]["benchmark_design"] == "fixed_map_candidate_scale"
        else "single-instance benchmark"
    )
    fig.suptitle(
        f"{source} MX {design_title}",
        x=0.06,
        y=0.987,
        ha="left",
        fontweight="semibold",
    )
    fig.text(
        0.06,
        0.956,
        _benchmark_context(rows),
        ha="left",
        color="#475569",
        fontsize=9,
    )
    fig.text(
        0.06,
        0.928,
        _scale_data_note(rows),
        ha="left",
        color="#475569",
        fontsize=8.5,
    )
    fig.legend(
        handles=[
            Patch(
                facecolor=solved_color, edgecolor="#334155", label="Reached target gap"
            ),
            Patch(
                facecolor=unsolved_color,
                edgecolor="#334155",
                hatch="//",
                label="Did not reach target gap",
            ),
        ],
        loc="upper right",
        bbox_to_anchor=(0.985, 0.91),
        ncol=2,
        frameon=False,
        fontsize=8.2,
    )
    for ax in (runtime_ax, gap_ax, saving_ax, service_ax):
        ax.set_xticks(positions, labels)
        ax.grid(axis="y", color="#e2e8f0", linewidth=0.7)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="x", labelsize=8.0, pad=4)
        for tick in ax.get_xticklabels():
            tick.set_linespacing(1.15)
        ax.margins(x=0.035)
    fig.text(
        0.5,
        0.047,
        _scale_design_note(rows),
        ha="center",
        color="#475569",
        fontsize=8.1,
    )
    fig.text(
        0.5,
        0.023,
        _network_mix_note(rows),
        ha="center",
        color="#475569",
        fontsize=8.1,
    )
    fig.tight_layout(rect=(0.035, 0.105, 0.99, 0.875), h_pad=3.5, w_pad=2.8)
    fig.savefig(root / "benchmark_runtime.png", dpi=220, bbox_inches="tight")
    fig.savefig(root / "benchmark_runtime.svg", bbox_inches="tight")
    plt.close(fig)
    return rows


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            f"Summarize schema-{EXPECTED_SCHEMA_VERSION} unified chi-epsilon "
            "scale runs."
        )
    )
    parser.add_argument("results_root")
    write_scale_report(parser.parse_args().results_root)
