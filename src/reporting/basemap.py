"""Offline OpenStreetMap tile and administrative-boundary loading."""

from __future__ import annotations

import math
import json
from pathlib import Path
from PIL import Image, ImageEnhance
from .io import _sha
from .projection import _mercator, _inverse_mercator, lonlat_to_tile, tile_to_lonlat

CACHE = Path(__file__).resolve().parents[2] / "data" / "map_cache"


def _basemap(bounds, cache):
    """Load only cached OSM tiles, with a matching EPSG:3857 image extent."""
    minx, maxx, miny, maxy = bounds
    corners = _inverse_mercator([[minx, miny], [maxx, maxy]])
    west, south = corners[0]
    east, north = corners[1]
    x0, y0 = lonlat_to_tile(west, north, 14)
    x1, y1 = lonlat_to_tile(east, south, 14)
    x0, y0, x1, y1 = map(math.floor, (x0, y0, x1, y1))
    image = Image.new("RGB", (256 * (x1 - x0 + 1), 256 * (y1 - y0 + 1)))
    sources = []
    for x in range(x0, x1 + 1):
        for y in range(y0, y1 + 1):
            path = cache / "tile_openstreetmap_org" / "14" / str(x) / f"{y}.png"
            if not path.is_file():
                raise FileNotFoundError(
                    f"Missing cached basemap tile: {path}; no network fallback is used"
                )
            image.paste(
                Image.open(path).convert("RGB"), (256 * (x - x0), 256 * (y - y0))
            )
            sources.append(dict(path=str(path), sha256=_sha(path)))
    ll = _mercator([tile_to_lonlat(x0, y1 + 1, 14), tile_to_lonlat(x1 + 1, y0, 14)])
    image = ImageEnhance.Color(image).enhance(0.50)
    image = ImageEnhance.Contrast(image).enhance(0.83)
    image = Image.blend(image, Image.new("RGB", image.size, "white"), 0.15)
    return image, (ll[0, 0], ll[1, 0], ll[0, 1], ll[1, 1]), sources


def _boundary(cache):
    path = cache / "nanshan_R5664195.geojson"
    geom = json.loads(path.read_text())["features"][0]["geometry"]
    polygons = (
        geom["coordinates"] if geom["type"] == "MultiPolygon" else [geom["coordinates"]]
    )
    return [_mercator(p[0]) for p in polygons], dict(path=str(path), sha256=_sha(path))
