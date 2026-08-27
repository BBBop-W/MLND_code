from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import pytest

from src.data_processor.loader import DataLoader
from src.utils.nanshan_scale_builder import (
    DEFAULT_NANSHAN_LEVELS,
    NanshanScaleBuildConfig,
    NanshanScaleLevel,
    build_nanshan_nested_scale,
)


def _to_lonlat(x_km: float, y_km: float) -> tuple[float, float]:
    center_lon = 113.94
    center_lat = 22.55
    km_per_lon = 111.32 * math.cos(math.radians(center_lat))
    return center_lon + x_km / km_per_lon, center_lat + y_km / 111.32


def _write_raw_orders(path: Path) -> None:
    fields = [
        "order_seq_id",
        "dt_seq_id",
        "customer_pay_time",
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
    ]
    geometry = [
        ("inner", (-0.2, -0.2), (0.2, 0.2)),
        ("middle", (-0.8, 0.4), (0.8, 0.5)),
        ("outer", (-1.5, -1.5), (1.5, 1.5)),
    ]
    rows: list[dict[str, object]] = []
    for order_id, pickup, dropoff in geometry:
        pickup_lon, pickup_lat = _to_lonlat(*pickup)
        dropoff_lon, dropoff_lat = _to_lonlat(*dropoff)
        rows.append(
            {
                "order_seq_id": order_id,
                "dt_seq_id": "1",
                "customer_pay_time": "12:10:00",
                "pickup_longitude": pickup_lon,
                "pickup_latitude": pickup_lat,
                "dropoff_longitude": dropoff_lon,
                "dropoff_latitude": dropoff_lat,
            }
        )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _test_config(output_root: Path, raw_path: Path) -> NanshanScaleBuildConfig:
    return NanshanScaleBuildConfig(
        output_root=output_root,
        raw_path=raw_path,
        levels=(
            NanshanScaleLevel(0.5, 1, 1, 1),
            NanshanScaleLevel(1.0, 2, 1, 1),
            NanshanScaleLevel(2.0, 3, 2, 2),
        ),
        dates=("1",),
        od_grid_km=0.1,
        site_grid_km=0.1,
    )


def test_builder_reclusters_sites_and_keeps_raw_order_crops_nested(
    tmp_path: Path,
) -> None:
    raw_path = tmp_path / "nanshan.csv"
    _write_raw_orders(raw_path)

    manifest_path = build_nanshan_nested_scale(
        _test_config(tmp_path / "family", raw_path)
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    levels = manifest["levels"]
    assert manifest["site_policy"] == "recluster_within_crop"
    assert manifest["raw_candidate_order_multisets_nested"] is True
    assert manifest["raw_demand_order_multisets_nested"] is True
    assert manifest["orders_sampled"] is False
    assert manifest["demand_rescaled"] is False
    assert manifest["required_solve_demand_scale"] == 1.0
    assert [row["crop_half_size_km"] for row in levels] == [0.5, 1.0, 2.0]
    assert [row["candidate_counts"] for row in levels] == [
        {"H": 1, "L": 1, "E": 1},
        {"H": 2, "L": 1, "E": 1},
        {"H": 3, "L": 2, "E": 2},
    ]
    assert [row["candidate_order_count"] for row in levels] == [1, 2, 3]
    assert [row["demand_order_count"] for row in levels] == [1, 2, 3]
    assert [row["total_demand_per_hour"] for row in levels] == [1.0, 2.0, 3.0]
    assert len({row["demand_order_multiset_sha256"] for row in levels}) == 3

    centers: set[tuple[float, float]] = set()
    for index, row in enumerate(levels, start=1):
        level_dir = Path(row["instance_dir"])
        metadata = json.loads((level_dir / "metadata.json").read_text(encoding="utf-8"))
        nested = metadata["nested_scale"]
        centers.add(
            (
                metadata["projection_center_lon"],
                metadata["projection_center_lat"],
            )
        )
        assert "demand_scale" not in metadata
        assert nested["level_index_small_to_large"] == index
        assert nested["site_policy"] == "recluster_within_crop"
        assert nested["site_coordinates_are_nested"] is False
        assert nested["raw_demand_orders_are_nested"] is True
        assert nested["orders_sampled"] is False
        assert nested["demand_rescaled"] is False
        assert nested["required_solve_demand_scale"] == 1.0
        assert nested["crop_area_km2"] == pytest.approx(
            (2.0 * row["crop_half_size_km"]) ** 2
        )

        loaded = DataLoader().load_instance_dir(str(level_dir), demand_scale=1.0)
        assert sum(loaded.h_od.values()) == pytest.approx(row["total_demand_per_hour"])
        assert loaded.metadata["demand_scale"] == 1.0

    assert len(centers) == 1
    assert (
        levels[-1]["canonical_file_sha256"]
        == manifest["full_parent_canonical_file_sha256"]
    )
    smallest_metadata = json.loads(
        (Path(levels[0]["instance_dir"]) / "metadata.json").read_text(encoding="utf-8")
    )
    assert (
        smallest_metadata["nested_scale"]["immediate_superset_instance"]
        == levels[1]["instance_dir"]
    )


def test_default_calibrated_scale_schedule_is_exact() -> None:
    assert [
        (level.half_size_km, level.lockers, level.bases, level.swaps)
        for level in DEFAULT_NANSHAN_LEVELS
    ] == [
        (3.0, 11, 2, 2),
        (4.5, 24, 2, 3),
        (6.0, 43, 3, 5),
        (7.5, 68, 5, 7),
        (8.5, 87, 6, 9),
        (10.0, 120, 8, 13),
    ]


def test_builder_refuses_to_overwrite_nonempty_family(tmp_path: Path) -> None:
    raw_path = tmp_path / "nanshan.csv"
    _write_raw_orders(raw_path)
    output_root = tmp_path / "family"
    output_root.mkdir()
    (output_root / "keep.txt").write_text("user data", encoding="utf-8")

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        build_nanshan_nested_scale(_test_config(output_root, raw_path))
    assert (output_root / "keep.txt").read_text(encoding="utf-8") == "user data"
