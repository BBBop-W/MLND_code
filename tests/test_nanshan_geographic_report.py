from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from src.utils.nanshan_geographic_report import (
    _outer_rings,
    _window_bounds,
    gcj02_to_wgs84,
    local_km_to_plot_lonlat,
    lonlat_to_tile,
    tile_to_lonlat,
)


def _metadata() -> dict[str, float]:
    return {
        "projection_center_lon": 113.94075,
        "projection_center_lat": 22.55175,
    }


def test_local_projection_inverse_without_datum_transform() -> None:
    metadata = _metadata()
    lon, lat = local_km_to_plot_lonlat(0.0, 0.0, metadata, assume_gcj02=False)
    assert lon == pytest.approx(metadata["projection_center_lon"])
    assert lat == pytest.approx(metadata["projection_center_lat"])

    lon, lat = local_km_to_plot_lonlat(1.0, 1.0, metadata, assume_gcj02=False)
    assert lat == pytest.approx(metadata["projection_center_lat"] + 1.0 / 111.32)
    assert lon == pytest.approx(
        metadata["projection_center_lon"]
        + 1.0 / (111.32 * math.cos(math.radians(metadata["projection_center_lat"])))
    )


def test_historical_gcj_transform_is_visible_and_bounded() -> None:
    source = (113.94075, 22.55175)
    plotted = gcj02_to_wgs84(*source)
    assert plotted[0] < source[0]
    assert plotted[1] > source[1]
    assert 0.003 < source[0] - plotted[0] < 0.008
    assert 0.001 < plotted[1] - source[1] < 0.006


@pytest.mark.parametrize("zoom", [10, 12, 14])
def test_tile_conversion_round_trip(zoom: int) -> None:
    source = (113.936, 22.548)
    tile = lonlat_to_tile(*source, zoom)
    recovered = tile_to_lonlat(*tile, zoom)
    assert recovered == pytest.approx(source, abs=1e-10)


def test_window_bounds_enclose_all_four_corners() -> None:
    metadata = _metadata()
    bounds = _window_bounds(metadata, 10.0, padding_km=0.0)
    west, south, east, north = bounds
    for x in (-10.0, 10.0):
        for y in (-10.0, 10.0):
            lon, lat = local_km_to_plot_lonlat(x, y, metadata)
            assert west <= lon <= east
            assert south <= lat <= north


def test_outer_rings_accepts_polygon_and_multipolygon(tmp_path: Path) -> None:
    polygon = {
        "type": "Polygon",
        "coordinates": [[[1, 2], [3, 4], [1, 2]]],
    }
    multi = {
        "type": "MultiPolygon",
        "coordinates": [[[[1, 2], [3, 4], [1, 2]]]],
    }
    assert len(_outer_rings(polygon)) == 1
    assert len(_outer_rings(multi)) == 1
    # JSON round-trip guards the exact GeoJSON-shaped input accepted by the loader.
    path = tmp_path / "boundary.json"
    path.write_text(json.dumps(multi), encoding="utf-8")
    assert len(_outer_rings(json.loads(path.read_text(encoding="utf-8")))) == 1
