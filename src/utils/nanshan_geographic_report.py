"""Build a source-audited Chinese report and geographic Nanshan maps.

The current canonical instances store local kilometre coordinates.  This module
reconstructs the source longitude/latitude coordinates with the projection
recorded in ``metadata.json`` and, for visualisation only, applies the same
GCJ-02-to-WGS84 assumption used by the repository's historical Nanshan map
script.  The assumption is surfaced on every figure and in the output manifest;
it never changes model distances, instances, or optimisation results.

Basemap tiles are requested one at a time, cached on disk, and visibly
attributed.  Generated report files live below ``results/`` and are therefore
kept separate from the model implementation committed to source control.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import urlparse

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mlnd-matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/mlnd-xdg-cache")

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import requests
from matplotlib import colors
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon
from PIL import Image, ImageEnhance

from src.model_contract import FORMULATION_ID, RESULT_SCHEMA_VERSION
from src.utils.instance_plotter import StudyLayout, load_study_layout


SCALE_LEVELS = (11, 24, 43, 68, 87, 120)
SCALE_MODES = ("HD", "MX")
FORMAL_MODES = ("HD", "MX", "LO")
MODE_NAMES = {"HD": "到家配送", "MX": "混合配送", "LO": "机柜自取"}
SITE_STYLE = {
    "H": {"marker": "o", "color": "#2563EB", "label": "开放机柜 H"},
    "L": {"marker": "s", "color": "#DC2626", "label": "开放机库 L"},
    "E": {"marker": "D", "color": "#059669", "label": "开放换电站 E"},
}
ROUTE_STYLE = {
    "locker": {"color": "#334155", "label": "H–H 运输"},
    "base": {"color": "#D97706", "label": "机库发出/回收"},
    "swap": {"color": "#7C3AED", "label": "经换电站路线"},
}

TILE_URL = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
TILE_ZOOM = 14
TILE_SIZE = 256
TILE_USER_AGENT = "MLND-Nanshan-Research-Figures/1.0 (Tsinghua SIGS research)"
BASEMAP_ATTRIBUTION = "© OpenStreetMap contributors"
OSM_COPYRIGHT_URL = "https://www.openstreetmap.org/copyright"
NANSHAN_RELATION = "R5664195"
NANSHAN_LOOKUP_URL = (
    "https://nominatim.openstreetmap.org/lookup?"
    "osm_ids=R5664195&format=geojson&polygon_geojson=1"
)

_FLOW_TOL = 1e-7
_COORDINATE_NOTE = (
    "定位假设：来源坐标按 GCJ-02 处理并转换为 WGS84；仅用于制图，待抽查原始坐标基准"
)


@dataclass(frozen=True)
class Basemap:
    image: Image.Image
    extent: tuple[float, float, float, float]
    zoom: int
    tile_count: int


@dataclass(frozen=True)
class ScaleCase:
    h: int
    instance_dir: Path
    result_dir: Path
    layout: StudyLayout
    run_config: dict[str, Any]
    summaries: dict[str, dict[str, Any]]


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write an empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_current_contract(payload: Mapping[str, Any], path: Path) -> None:
    if payload.get("schema_version") != RESULT_SCHEMA_VERSION:
        raise ValueError(
            f"{path} has schema {payload.get('schema_version')!r}; "
            f"expected {RESULT_SCHEMA_VERSION}"
        )
    if payload.get("formulation_id") != FORMULATION_ID:
        raise ValueError(
            f"{path} has formulation {payload.get('formulation_id')!r}; "
            f"expected {FORMULATION_ID!r}"
        )


def _mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"expected a mapping for {context}")
    return value


def _number(value: Any, context: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"invalid numeric value for {context}: {value!r}")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid numeric value for {context}: {value!r}") from exc
    if not math.isfinite(result):
        raise ValueError(f"non-finite numeric value for {context}: {value!r}")
    return result


def _out_of_china(lon: float, lat: float) -> bool:
    return not (72.004 <= lon <= 137.8347 and 0.8293 <= lat <= 55.8271)


def _transform_lat(lon: float, lat: float) -> float:
    result = (
        -100.0
        + 2.0 * lon
        + 3.0 * lat
        + 0.2 * lat * lat
        + 0.1 * lon * lat
        + 0.2 * math.sqrt(abs(lon))
    )
    result += (
        20.0 * math.sin(6.0 * lon * math.pi)
        + 20.0 * math.sin(2.0 * lon * math.pi)
    ) * 2.0 / 3.0
    result += (
        20.0 * math.sin(lat * math.pi)
        + 40.0 * math.sin(lat / 3.0 * math.pi)
    ) * 2.0 / 3.0
    result += (
        160.0 * math.sin(lat / 12.0 * math.pi)
        + 320.0 * math.sin(lat * math.pi / 30.0)
    ) * 2.0 / 3.0
    return result


def _transform_lon(lon: float, lat: float) -> float:
    result = (
        300.0
        + lon
        + 2.0 * lat
        + 0.1 * lon * lon
        + 0.1 * lon * lat
        + 0.1 * math.sqrt(abs(lon))
    )
    result += (
        20.0 * math.sin(6.0 * lon * math.pi)
        + 20.0 * math.sin(2.0 * lon * math.pi)
    ) * 2.0 / 3.0
    result += (
        20.0 * math.sin(lon * math.pi)
        + 40.0 * math.sin(lon / 3.0 * math.pi)
    ) * 2.0 / 3.0
    result += (
        150.0 * math.sin(lon / 12.0 * math.pi)
        + 300.0 * math.sin(lon / 30.0 * math.pi)
    ) * 2.0 / 3.0
    return result


def gcj02_to_wgs84(lon: float, lat: float) -> tuple[float, float]:
    """Approximate inverse transform used by the historical Nanshan figures."""

    if _out_of_china(lon, lat):
        return lon, lat
    semi_major = 6378245.0
    eccentricity = 0.00669342162296594323
    delta_lat = _transform_lat(lon - 105.0, lat - 35.0)
    delta_lon = _transform_lon(lon - 105.0, lat - 35.0)
    rad_lat = math.radians(lat)
    magic = 1.0 - eccentricity * math.sin(rad_lat) ** 2
    sqrt_magic = math.sqrt(magic)
    delta_lat = (delta_lat * 180.0) / (
        (semi_major * (1.0 - eccentricity)) / (magic * sqrt_magic) * math.pi
    )
    delta_lon = (delta_lon * 180.0) / (
        semi_major / sqrt_magic * math.cos(rad_lat) * math.pi
    )
    return lon - delta_lon, lat - delta_lat


def local_km_to_plot_lonlat(
    x_km: float,
    y_km: float,
    metadata: Mapping[str, Any],
    *,
    assume_gcj02: bool = True,
) -> tuple[float, float]:
    """Invert the canonical local projection and return basemap coordinates."""

    center_lon = _number(metadata.get("projection_center_lon"), "projection center lon")
    center_lat = _number(metadata.get("projection_center_lat"), "projection center lat")
    source_lon = center_lon + x_km / (111.32 * math.cos(math.radians(center_lat)))
    source_lat = center_lat + y_km / 111.32
    if assume_gcj02:
        return gcj02_to_wgs84(source_lon, source_lat)
    return source_lon, source_lat


def lonlat_to_tile(lon: float, lat: float, zoom: int) -> tuple[float, float]:
    latitude = max(min(lat, 85.05112878), -85.05112878)
    n = 2.0**zoom
    x = (lon + 180.0) / 360.0 * n
    y = (
        1.0
        - math.asinh(math.tan(math.radians(latitude))) / math.pi
    ) / 2.0 * n
    return x, y


def tile_to_lonlat(x: float, y: float, zoom: int) -> tuple[float, float]:
    n = 2.0**zoom
    lon = x / n * 360.0 - 180.0
    lat = math.degrees(math.atan(math.sinh(math.pi * (1.0 - 2.0 * y / n))))
    return lon, lat


def _tile_path(cache_dir: Path, x: int, y: int, zoom: int) -> Path:
    host = urlparse(TILE_URL).netloc.replace(".", "_")
    return cache_dir / host / str(zoom) / str(x) / f"{y}.png"


def _fetch_tile(
    session: requests.Session,
    cache_dir: Path,
    x: int,
    y: int,
    zoom: int,
) -> Image.Image:
    path = _tile_path(cache_dir, x, y, zoom)
    if path.is_file():
        return Image.open(path).convert("RGB")
    url = TILE_URL.format(z=zoom, x=x, y=y)
    response = session.get(url, timeout=30)
    response.raise_for_status()
    if "image" not in response.headers.get("content-type", ""):
        raise ValueError(f"tile server returned non-image content: {url}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(response.content)
    return Image.open(io.BytesIO(response.content)).convert("RGB")


def build_basemap(
    bounds: tuple[float, float, float, float],
    cache_dir: Path,
    *,
    zoom: int = TILE_ZOOM,
) -> Basemap:
    """Fetch only the tiles intersecting one requested bounding box."""

    west, south, east, north = bounds
    x0, y_top = lonlat_to_tile(west, north, zoom)
    x1, y_bottom = lonlat_to_tile(east, south, zoom)
    xt0, xt1 = math.floor(x0), math.floor(x1)
    yt0, yt1 = math.floor(y_top), math.floor(y_bottom)
    mosaic = Image.new(
        "RGB",
        ((xt1 - xt0 + 1) * TILE_SIZE, (yt1 - yt0 + 1) * TILE_SIZE),
        "white",
    )
    session = requests.Session()
    session.headers.update({"User-Agent": TILE_USER_AGENT})
    for x in range(xt0, xt1 + 1):
        for y in range(yt0, yt1 + 1):
            tile = _fetch_tile(session, cache_dir, x, y, zoom)
            mosaic.paste(tile, ((x - xt0) * TILE_SIZE, (y - yt0) * TILE_SIZE))
    west_full, north_full = tile_to_lonlat(xt0, yt0, zoom)
    east_full, south_full = tile_to_lonlat(xt1 + 1, yt1 + 1, zoom)
    # Make the network layer the visual foreground.  Cached source tiles remain
    # unchanged; this presentation-only treatment is applied to the mosaic.
    mosaic = ImageEnhance.Color(mosaic).enhance(0.48)
    mosaic = ImageEnhance.Contrast(mosaic).enhance(0.78)
    mosaic = Image.blend(
        mosaic,
        Image.new("RGB", mosaic.size, "white"),
        alpha=0.16,
    )
    return Basemap(
        image=mosaic,
        extent=(west_full, east_full, south_full, north_full),
        zoom=zoom,
        tile_count=(xt1 - xt0 + 1) * (yt1 - yt0 + 1),
    )


def _load_nanshan_boundary(cache_path: Path) -> dict[str, Any] | None:
    if cache_path.is_file():
        payload = _read_json(cache_path)
    else:
        session = requests.Session()
        session.headers.update({"User-Agent": TILE_USER_AGENT})
        response = session.get(NANSHAN_LOOKUP_URL, timeout=60)
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise ValueError("Nominatim returned a non-object GeoJSON payload")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    features = payload.get("features")
    if not isinstance(features, list) or not features:
        return None
    geometry = features[0].get("geometry")
    return geometry if isinstance(geometry, dict) else None


def _outer_rings(geometry: Mapping[str, Any] | None) -> list[np.ndarray]:
    if not geometry:
        return []
    kind = geometry.get("type")
    coordinates = geometry.get("coordinates")
    rings: list[np.ndarray] = []
    if kind == "Polygon" and isinstance(coordinates, list) and coordinates:
        rings.append(np.asarray(coordinates[0], dtype=float))
    elif kind == "MultiPolygon" and isinstance(coordinates, list):
        for polygon in coordinates:
            if isinstance(polygon, list) and polygon:
                rings.append(np.asarray(polygon[0], dtype=float))
    return [ring for ring in rings if ring.ndim == 2 and ring.shape[1] == 2]


def _window_bounds(
    metadata: Mapping[str, Any],
    half_size_km: float,
    *,
    padding_km: float = 0.45,
) -> tuple[float, float, float, float]:
    radius = half_size_km + padding_km
    corners = [
        local_km_to_plot_lonlat(x, y, metadata)
        for x in (-radius, radius)
        for y in (-radius, radius)
    ]
    longitudes = [point[0] for point in corners]
    latitudes = [point[1] for point in corners]
    return min(longitudes), min(latitudes), max(longitudes), max(latitudes)


def _window_polygon(
    metadata: Mapping[str, Any],
    half_size_km: float,
) -> np.ndarray:
    points = [
        (-half_size_km, -half_size_km),
        (half_size_km, -half_size_km),
        (half_size_km, half_size_km),
        (-half_size_km, half_size_km),
    ]
    return np.asarray(
        [local_km_to_plot_lonlat(x, y, metadata) for x, y in points],
        dtype=float,
    )


def _load_case(
    h: int,
    instance_dir: Path,
    result_dir: Path,
    modes: Sequence[str],
) -> ScaleCase:
    run_config_path = result_dir / "run_config.json"
    run_config = _read_json(run_config_path)
    _require_current_contract(run_config, run_config_path)
    if run_config.get("modes") != list(modes):
        raise ValueError(
            f"{run_config_path} modes {run_config.get('modes')!r} != {list(modes)!r}"
        )
    summaries: dict[str, dict[str, Any]] = {}
    for mode in modes:
        summary_path = result_dir / mode / "summary.json"
        summary = _read_json(summary_path)
        _require_current_contract(summary, summary_path)
        if summary.get("mode") != mode:
            raise ValueError(f"{summary_path} reports mode {summary.get('mode')!r}")
        if summary.get("run_signature_sha256") != run_config.get(
            "run_signature_sha256"
        ):
            raise ValueError(f"run signature mismatch: {summary_path}")
        summaries[mode] = summary
    layout = load_study_layout(instance_dir)
    actual_h = sum(site["type"] == "H" for site in layout.sites)
    if actual_h != h:
        raise ValueError(f"{instance_dir} contains H{actual_h}, expected H{h}")
    return ScaleCase(h, instance_dir, result_dir, layout, run_config, summaries)


def _load_scale_cases(instance_root: Path, scale_results: Path) -> list[ScaleCase]:
    manifest = _read_json(instance_root / "scale_manifest.json")
    levels = manifest.get("levels")
    if not isinstance(levels, list) or len(levels) != len(SCALE_LEVELS):
        raise ValueError("scale manifest must contain the six formal levels")
    cases: list[ScaleCase] = []
    for expected_h, level in zip(SCALE_LEVELS, levels):
        level_map = _mapping(level, f"scale H{expected_h}")
        counts = _mapping(level_map.get("candidate_counts"), "candidate counts")
        h = int(_number(counts.get("H"), "candidate H count"))
        if h != expected_h:
            raise ValueError(f"scale manifest order mismatch: H{h} != H{expected_h}")
        instance_name = Path(str(level_map["instance_dir"])).name
        cases.append(
            _load_case(
                h,
                instance_root / instance_name,
                scale_results / f"H{h}",
                SCALE_MODES,
            )
        )
    return cases


def _site_coordinates(
    layout: StudyLayout,
) -> tuple[dict[str, tuple[float, float]], dict[str, str]]:
    metadata = layout.metadata
    coordinates: dict[str, tuple[float, float]] = {}
    types: dict[str, str] = {}
    for site in layout.sites:
        site_id = str(site["id"])
        coordinates[site_id] = local_km_to_plot_lonlat(
            float(site["x"]), float(site["y"]), metadata
        )
        types[site_id] = str(site["type"])
    return coordinates, types


def _node_coordinates(layout: StudyLayout) -> dict[str, tuple[float, float]]:
    return {
        str(node["id"]): local_km_to_plot_lonlat(
            float(node["x"]), float(node["y"]), layout.metadata
        )
        for node in layout.nodes
    }


def _physical_routes(mode_dir: Path) -> list[dict[str, Any]]:
    path = mode_dir / "incumbent.sol"
    rows: list[dict[str, Any]] = []
    variable_count = 0
    with path.open(encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line.startswith("varphi["):
                continue
            variable_count += 1
            try:
                name, value_text = line.rsplit(maxsplit=1)
                parts = name[len("varphi[") : -1].split(",")
                value = float(value_text)
            except (ValueError, IndexError) as exc:
                raise ValueError(f"malformed route variable {path}:{line_number}") from exc
            if len(parts) != 3 or not math.isfinite(value) or value < -_FLOW_TOL:
                raise ValueError(f"invalid route variable {path}:{line_number}")
            if value > _FLOW_TOL:
                rows.append(
                    {
                        "origin": parts[0],
                        "destination": parts[1],
                        "option": parts[2],
                        "flow": value,
                    }
                )
    if variable_count == 0:
        raise ValueError(f"no physical route variables found: {path}")
    return rows


def _open_facilities(mode_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in _read_csv(mode_dir / "open_facilities.csv"):
        parsed: dict[str, Any] = dict(row)
        for field in (
            "rho",
            "locker_service_rate_per_hour",
            "locker_workload_per_hour",
            "incident_physical_flow_per_hour",
            "swap_throughput_per_hour",
        ):
            text = row.get(field, "")
            parsed[field] = float(text) if text not in ("", None) else None
        rows.append(parsed)
    return rows


def _route_kind(
    origin: str,
    destination: str,
    option: str,
    site_types: Mapping[str, str],
) -> str:
    if option != "__direct__":
        return "swap"
    if site_types.get(origin) == site_types.get(destination) == "H":
        return "locker"
    return "base"


def _draw_basemap(ax: plt.Axes, basemap: Basemap) -> None:
    ax.imshow(
        basemap.image,
        extent=basemap.extent,
        origin="upper",
        alpha=0.91,
        zorder=-10,
    )


def _draw_boundary(ax: plt.Axes, rings: Iterable[np.ndarray]) -> None:
    for ring in rings:
        ax.plot(
            ring[:, 0],
            ring[:, 1],
            color="#E11D48",
            linewidth=1.1,
            linestyle=(0, (4, 3)),
            alpha=0.72,
            zorder=1,
        )


def _draw_demand_activity(ax: plt.Axes, layout: StudyLayout) -> None:
    coordinates = _node_coordinates(layout)
    active = [
        (node_id, rate)
        for node_id, rate in layout.endpoint_rates.items()
        if rate > 0.0 and node_id in coordinates
    ]
    if not active:
        return
    max_rate = max(rate for _node_id, rate in active)
    ax.scatter(
        [coordinates[node_id][0] for node_id, _rate in active],
        [coordinates[node_id][1] for node_id, _rate in active],
        s=[4.0 + 32.0 * math.sqrt(rate / max_rate) for _node_id, rate in active],
        c="#F97316",
        alpha=0.17,
        edgecolors="none",
        zorder=0,
    )


def _draw_routes(
    ax: plt.Axes,
    layout: StudyLayout,
    rows: Sequence[Mapping[str, Any]],
    *,
    emphasize_structure: bool = False,
) -> None:
    coordinates, site_types = _site_coordinates(layout)
    if not rows:
        return
    maximum = max(float(row["flow"]) for row in rows)
    ordered = sorted(rows, key=lambda row: float(row["flow"]))
    for row in ordered:
        origin = str(row["origin"])
        destination = str(row["destination"])
        option = str(row["option"])
        flow = float(row["flow"])
        points = [coordinates[origin]]
        if option != "__direct__":
            points.append(coordinates[option])
        points.append(coordinates[destination])
        kind = _route_kind(origin, destination, option, site_types)
        color_value = ROUTE_STYLE[kind]["color"]
        width = 0.30 + (3.2 if emphasize_structure else 2.2) * math.sqrt(
            flow / maximum
        )
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        ax.plot(
            xs,
            ys,
            color="white",
            linewidth=width + 1.4,
            alpha=0.70,
            solid_capstyle="round",
            zorder=2,
        )
        ax.plot(
            xs,
            ys,
            color=color_value,
            linewidth=width,
            alpha=0.70 if emphasize_structure else 0.54,
            solid_capstyle="round",
            zorder=3,
        )


def _draw_sites(
    ax: plt.Axes,
    layout: StudyLayout,
    opened: Sequence[Mapping[str, Any]],
    *,
    utilization: bool = False,
) -> Any:
    coordinates, _site_types = _site_coordinates(layout)
    for site_type in ("L", "E", "H"):
        style = SITE_STYLE[site_type]
        candidates = [site for site in layout.sites if site["type"] == site_type]
        ax.scatter(
            [coordinates[str(site["id"])][0] for site in candidates],
            [coordinates[str(site["id"])][1] for site in candidates],
            s={"H": 18.0, "L": 28.0, "E": 24.0}[site_type],
            marker=style["marker"],
            facecolors="white",
            edgecolors="#64748B",
            linewidths=0.55,
            alpha=0.48,
            zorder=4,
        )

    color_artist = None
    for site_type in ("L", "E", "H"):
        style = SITE_STYLE[site_type]
        part = [row for row in opened if row.get("type") == {
            "H": "locker", "L": "base", "E": "swap"
        }[site_type]]
        if not part:
            continue
        xs = [coordinates[str(row["facility"])][0] for row in part]
        ys = [coordinates[str(row["facility"])][1] for row in part]
        if utilization and site_type == "H":
            ratios = [
                float(row["locker_workload_per_hour"])
                / float(row["locker_service_rate_per_hour"])
                for row in part
            ]
            workloads = [float(row["locker_workload_per_hour"]) for row in part]
            maximum = max(workloads, default=1.0)
            color_artist = ax.scatter(
                xs,
                ys,
                s=[72.0 + 165.0 * math.sqrt(value / maximum) for value in workloads],
                marker=style["marker"],
                c=ratios,
                cmap="YlOrRd",
                norm=colors.Normalize(vmin=0.0, vmax=1.0),
                edgecolors="white",
                linewidths=1.0,
                zorder=7,
            )
        else:
            size = {"H": 72.0, "L": 98.0, "E": 84.0}[site_type]
            ax.scatter(
                xs,
                ys,
                s=size,
                marker=style["marker"],
                c=style["color"],
                edgecolors="white",
                linewidths=1.05,
                zorder=7,
            )
    return color_artist


def _add_map_furniture(
    ax: plt.Axes,
    bounds: tuple[float, float, float, float],
    *,
    compact: bool = False,
) -> None:
    west, south, east, north = bounds
    center_lat = 0.5 * (south + north)
    width_km = (east - west) * 111.32 * math.cos(math.radians(center_lat))
    scale_km = 2.0 if width_km <= 9.0 else 5.0
    scale_lon = scale_km / (111.32 * math.cos(math.radians(center_lat)))
    x0 = west + 0.055 * (east - west)
    y0 = south + 0.055 * (north - south)
    ax.plot([x0, x0 + scale_lon], [y0, y0], color="#0F172A", lw=2.3, zorder=20)
    ax.text(
        x0 + scale_lon / 2.0,
        y0 + 0.012 * (north - south),
        f"{scale_km:g} km",
        ha="center",
        va="bottom",
        fontsize=6.5 if compact else 8.0,
        color="#0F172A",
        zorder=20,
    )
    ax.annotate(
        "N",
        xy=(0.94, 0.93),
        xytext=(0.94, 0.82),
        xycoords="axes fraction",
        ha="center",
        fontsize=7.0 if compact else 8.5,
        fontweight="bold",
        arrowprops={"arrowstyle": "-|>", "color": "#0F172A", "lw": 1.2},
        zorder=20,
    )


def _finish_map_axis(
    ax: plt.Axes,
    bounds: tuple[float, float, float, float],
    *,
    compact: bool,
) -> None:
    west, south, east, north = bounds
    ax.set_xlim(west, east)
    ax.set_ylim(south, north)
    ax.set_aspect(1.0 / math.cos(math.radians((south + north) / 2.0)))
    ax.set_axis_off()
    _add_map_furniture(ax, bounds, compact=compact)
    if not compact:
        ax.text(
            0.99,
            0.01,
            BASEMAP_ATTRIBUTION,
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=5.8,
            color="#475569",
            bbox={"facecolor": "white", "alpha": 0.76, "edgecolor": "none", "pad": 1.2},
            zorder=30,
        )


def _summary_metrics(summary: Mapping[str, Any]) -> dict[str, Any]:
    facilities = _mapping(summary.get("facilities"), "facilities")
    opened = _mapping(facilities.get("open"), "open facilities")
    funnel = _mapping(summary.get("demand_funnel_orders_per_hour"), "demand funnel")
    model_size = _mapping(summary.get("model_size"), "model size")
    return {
        "status": str(summary.get("status_name")),
        "objective": _number(summary.get("objective"), "objective"),
        "bound": _number(summary.get("objective_bound"), "objective bound"),
        "gap_percent": 100.0 * _number(summary.get("mip_gap"), "MIP gap"),
        "runtime": _number(summary.get("runtime_seconds"), "runtime"),
        "open_l": int(_number(opened.get("bases"), "open bases")),
        "open_h": int(_number(opened.get("lockers"), "open lockers")),
        "open_e": int(_number(opened.get("swap_stations"), "open swaps")),
        "active_h": int(_number(facilities.get("active_lockers"), "active lockers")),
        "fleet": int(round(_number(summary.get("fleet_size"), "fleet size"))),
        "service": _number(funnel.get("total_drone_realized"), "drone service"),
        "service_percent": 100.0
        * _number(funnel.get("drone_service_fraction_of_demand"), "service fraction"),
        "locker_pickup": _number(funnel.get("locker_pickup"), "locker pickup"),
        "assisted_home": _number(funnel.get("assisted_home"), "assisted home"),
        "variables": int(_number(model_size.get("variables"), "model variables")),
        "cost_saving_percent": _number(
            summary.get("cost_saving_percent"), "cost saving percent"
        ),
    }


def _status_zh(metrics: Mapping[str, Any]) -> str:
    return "达到1%容差" if metrics["status"] == "OPTIMAL" else "时限停止"


def _draw_case_panel(
    ax: plt.Axes,
    case: ScaleCase,
    mode: str,
    basemap: Basemap,
    boundary_rings: Sequence[np.ndarray],
    *,
    compact: bool,
    show_demand: bool = True,
    emphasize_structure: bool = False,
    utilization: bool = False,
) -> Any:
    summary = case.summaries[mode]
    metrics = _summary_metrics(summary)
    mode_dir = case.result_dir / mode
    bounds = _window_bounds(case.layout.metadata, case.layout.half_size_km)
    _draw_basemap(ax, basemap)
    _draw_boundary(ax, boundary_rings)
    if show_demand:
        _draw_demand_activity(ax, case.layout)
    routes = _physical_routes(mode_dir)
    _draw_routes(
        ax,
        case.layout,
        routes,
        emphasize_structure=emphasize_structure,
    )
    color_artist = _draw_sites(
        ax,
        case.layout,
        _open_facilities(mode_dir),
        utilization=utilization,
    )
    title = (
        f"H{case.h} · {MODE_NAMES[mode]}\n"
        f"{_status_zh(metrics)} · Gap {metrics['gap_percent']:.2f}% · "
        f"开放 L/H/E {metrics['open_l']}/{metrics['open_h']}/{metrics['open_e']} · "
        f"有流量 H {metrics['active_h']}"
    )
    ax.set_title(
        title,
        loc="left",
        fontsize=8.4 if compact else 11.0,
        color="#172033",
        fontweight="semibold",
        pad=6,
    )
    _finish_map_axis(ax, bounds, compact=compact)
    return color_artist


def _legend_handles(*, route_structure: bool = False) -> list[Line2D]:
    handles: list[Line2D] = [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            markerfacecolor="#F97316",
            markeredgecolor="none",
            alpha=0.32,
            markersize=6,
            label="OD 端点活动",
        ),
        Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            markerfacecolor="white",
            markeredgecolor="#64748B",
            markersize=6,
            label="候选设施",
        ),
    ]
    for site_type in ("H", "L", "E"):
        style = SITE_STYLE[site_type]
        handles.append(
            Line2D(
                [],
                [],
                marker=style["marker"],
                linestyle="none",
                markerfacecolor=style["color"],
                markeredgecolor="white",
                markersize=7,
                label=style["label"],
            )
        )
    if route_structure:
        for kind in ("locker", "base", "swap"):
            style = ROUTE_STYLE[kind]
            handles.append(
                Line2D([], [], color=style["color"], lw=2.2, label=style["label"])
            )
    else:
        handles.append(Line2D([], [], color="#334155", lw=2.2, label="正流量航线"))
    return handles


def _save_figure(figure: plt.Figure, stem: Path, *, dpi: int = 220) -> tuple[Path, Path]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    png = stem.with_suffix(".png")
    svg = stem.with_suffix(".svg")
    figure.savefig(png, dpi=dpi, bbox_inches="tight", facecolor="white")
    figure.savefig(svg, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return png, svg


def _plot_scale_maps(
    cases: Sequence[ScaleCase],
    basemap: Basemap,
    boundary_rings: Sequence[np.ndarray],
    output_dir: Path,
) -> list[Path]:
    outputs: list[Path] = []
    figure, axes = plt.subplots(2, 3, figsize=(17.2, 11.6))
    for axis, case in zip(axes.flat, cases):
        _draw_case_panel(
            axis,
            case,
            "MX",
            basemap,
            boundary_rings,
            compact=True,
        )
    figure.suptitle(
        "南山同中心地图裁剪：六档混合配送网络",
        x=0.04,
        y=0.993,
        ha="left",
        fontsize=16,
        fontweight="semibold",
        color="#172033",
    )
    figure.legend(
        handles=_legend_handles(),
        loc="lower center",
        ncol=6,
        frameon=False,
        fontsize=8.5,
        bbox_to_anchor=(0.5, 0.025),
    )
    figure.text(
        0.04,
        0.006,
        f"{_COORDINATE_NOTE} · {BASEMAP_ATTRIBUTION}",
        fontsize=7.4,
        color="#64748B",
    )
    figure.subplots_adjust(top=0.93, bottom=0.075, hspace=0.14, wspace=0.04)
    png, _svg = _save_figure(figure, output_dir / "nanshan_six_scales_MX")
    outputs.append(png)

    for case in cases:
        figure, axis = plt.subplots(figsize=(8.2, 8.2))
        _draw_case_panel(
            axis,
            case,
            "MX",
            basemap,
            boundary_rings,
            compact=False,
        )
        axis.legend(
            handles=_legend_handles(),
            loc="upper left",
            frameon=True,
            framealpha=0.92,
            fontsize=7.7,
            ncol=2,
        )
        figure.text(0.02, 0.008, _COORDINATE_NOTE, fontsize=7.0, color="#64748B")
        png, _svg = _save_figure(
            figure,
            output_dir / f"nanshan_H{case.h}_MX",
            dpi=240,
        )
        outputs.append(png)
    return outputs


def _plot_formal_modes(
    case: ScaleCase,
    basemap: Basemap,
    boundary_rings: Sequence[np.ndarray],
    output_dir: Path,
) -> Path:
    figure, axes = plt.subplots(1, 3, figsize=(18.2, 6.9))
    for axis, mode in zip(axes, FORMAL_MODES):
        _draw_case_panel(
            axis,
            case,
            mode,
            basemap,
            boundary_rings,
            compact=True,
        )
    figure.suptitle(
        "南山 H120 正式算例：三种服务方式的网络设计",
        x=0.035,
        y=0.992,
        ha="left",
        fontsize=16,
        fontweight="semibold",
        color="#172033",
    )
    figure.legend(
        handles=_legend_handles(),
        loc="lower center",
        ncol=6,
        frameon=False,
        fontsize=8.5,
        bbox_to_anchor=(0.5, 0.04),
    )
    figure.text(
        0.035,
        0.012,
        f"{_COORDINATE_NOTE} · {BASEMAP_ATTRIBUTION}",
        fontsize=7.3,
        color="#64748B",
    )
    figure.subplots_adjust(top=0.88, bottom=0.105, wspace=0.04)
    png, _svg = _save_figure(figure, output_dir / "nanshan_H120_three_modes")
    return png


def _plot_route_structure(
    case: ScaleCase,
    basemap: Basemap,
    boundary_rings: Sequence[np.ndarray],
    output_dir: Path,
) -> Path:
    figure, axis = plt.subplots(figsize=(9.2, 9.0))
    _draw_case_panel(
        axis,
        case,
        "MX",
        basemap,
        boundary_rings,
        compact=False,
        show_demand=False,
        emphasize_structure=True,
    )
    axis.set_title(
        "H120 · 混合配送路线结构\n线宽表示每小时物理流量；紫色折线经过开放换电站",
        loc="left",
        fontsize=12.0,
        fontweight="semibold",
        color="#172033",
        pad=8,
    )
    axis.legend(
        handles=_legend_handles(route_structure=True)[1:],
        loc="upper left",
        ncol=2,
        frameon=True,
        framealpha=0.94,
        fontsize=8.0,
    )
    figure.text(0.02, 0.008, _COORDINATE_NOTE, fontsize=7.0, color="#64748B")
    png, _svg = _save_figure(
        figure,
        output_dir / "nanshan_H120_MX_route_structure",
        dpi=240,
    )
    return png


def _plot_utilization(
    case: ScaleCase,
    basemap: Basemap,
    boundary_rings: Sequence[np.ndarray],
    output_dir: Path,
) -> Path:
    figure, axis = plt.subplots(figsize=(9.2, 9.0))
    color_artist = _draw_case_panel(
        axis,
        case,
        "MX",
        basemap,
        boundary_rings,
        compact=False,
        show_demand=False,
        utilization=True,
    )
    axis.set_title(
        "H120 · 混合配送机柜利用率\n颜色表示工作负荷/处理能力，点大小表示机柜工作负荷",
        loc="left",
        fontsize=12.0,
        fontweight="semibold",
        color="#172033",
        pad=8,
    )
    if color_artist is not None:
        colorbar = figure.colorbar(color_artist, ax=axis, fraction=0.036, pad=0.012)
        colorbar.set_label("机柜利用率", fontsize=9)
        colorbar.ax.tick_params(labelsize=8)
    handles = _legend_handles()[1:5]
    axis.legend(
        handles=handles,
        loc="upper left",
        ncol=2,
        frameon=True,
        framealpha=0.94,
        fontsize=8.0,
    )
    figure.text(0.02, 0.008, _COORDINATE_NOTE, fontsize=7.0, color="#64748B")
    png, _svg = _save_figure(
        figure,
        output_dir / "nanshan_H120_MX_locker_utilization",
        dpi=240,
    )
    return png


def _plot_scale_windows(
    cases: Sequence[ScaleCase],
    basemap: Basemap,
    boundary_rings: Sequence[np.ndarray],
    output_dir: Path,
) -> Path:
    full_case = cases[-1]
    figure, axis = plt.subplots(figsize=(9.2, 9.0))
    _draw_basemap(axis, basemap)
    _draw_boundary(axis, boundary_rings)
    palette = ["#7C3AED", "#2563EB", "#0891B2", "#059669", "#D97706", "#DC2626"]
    for case, color_value in reversed(list(zip(cases, palette))):
        polygon = _window_polygon(case.layout.metadata, case.layout.half_size_km)
        axis.add_patch(
            Polygon(
                polygon,
                closed=True,
                fill=False,
                edgecolor=color_value,
                linewidth=2.0 if case.h in (11, 120) else 1.35,
                alpha=0.84,
                zorder=4,
            )
        )
        label_at = polygon[2]
        axis.text(
            label_at[0],
            label_at[1],
            f" H{case.h}",
            color=color_value,
            fontsize=8.0,
            fontweight="bold",
            ha="left",
            va="bottom",
            zorder=6,
        )
    bounds = _window_bounds(full_case.layout.metadata, full_case.layout.half_size_km)
    _finish_map_axis(axis, bounds, compact=False)
    axis.set_title(
        "南山六档规模的同中心裁剪窗口\n缩小规模仅裁剪地图；订单不抽样、需求不放缩，候选点在各窗口内重新聚类",
        loc="left",
        fontsize=12.0,
        fontweight="semibold",
        color="#172033",
        pad=8,
    )
    figure.text(0.02, 0.008, _COORDINATE_NOTE, fontsize=7.0, color="#64748B")
    png, _svg = _save_figure(
        figure,
        output_dir / "nanshan_scale_windows",
        dpi=240,
    )
    return png


def _format_list(values: Iterable[Any]) -> str:
    return "、".join(f"{float(value):g}" for value in values)


def _parameter_rows(scale_config: Mapping[str, Any]) -> list[dict[str, Any]]:
    model = _mapping(scale_config.get("model"), "model parameters")
    choice = _mapping(model.get("choice"), "choice parameters")
    metadata = _mapping(
        _mapping(scale_config.get("instance"), "instance").get("metadata"),
        "instance metadata",
    )
    data = _mapping(metadata.get("data_config"), "data config")
    solver = _mapping(scale_config.get("solver"), "solver config")
    rows: list[dict[str, Any]] = []

    def add(
        category: str,
        parameter: str,
        value: Any,
        unit: str,
        scope: str,
        note: str,
    ) -> None:
        rows.append(
            {
                "类别": category,
                "参数": parameter,
                "本次取值": value,
                "单位": unit,
                "适用范围": scope,
                "性质或说明": note,
            }
        )

    add("物理与服务", "无人机飞行速度", model["drone_speed_mps"], "m/s", "全部方式", "用户给定")
    add("物理与服务", "单块满电池可用航程", model["battery_range_km"], "km/航段", "全部方式", "用户给定；每一物理航段分别受限")
    add("物理与服务", "换电服务时间", 60 * float(model["swap_service_time_hours"]), "min/次", "全部方式", "场景假设")
    add("物理与服务", "机柜最大平均处理时间", 60 * float(model["max_locker_time_hours"]), "min", "全部方式", "场景假设")
    add("物理与服务", "虚拟待机最大平均等待时间", 60 * float(model["max_virtual_wait_hours"]), "min", "全部方式", "场景假设")
    add("可靠性与容量", "可用性等级", _format_list(model["availability_levels"]), "—", "全部方式", "离散等级")
    add("可靠性与容量", "机柜处理能力等级", _format_list(model["locker_service_rates"]), "次/h", "全部方式", "三档容量")
    add("可靠性与容量", "机柜容量成本", _format_list(model["locker_capacity_costs"]), "CNY/h", "全部方式", "与三档处理能力一一对应")
    cost_specs = [
        ("无人机库固定成本", "base_cost_per_hour", "CNY/h", "全部方式"),
        ("智能机柜固定成本", "locker_fixed_cost_per_hour", "CNY/h", "全部方式"),
        ("换电站固定成本", "swap_cost_per_hour", "CNY/h", "全部方式"),
        ("普通物理连接建设成本", "corridor_cost_per_km_hour", "CNY/(km·h)", "全部方式"),
        ("涉及换电站的连接建设成本", "connection_cost_per_km_hour", "CNY/(km·h)", "全部方式"),
        ("航程运营成本", "flight_cost_per_km", "CNY/(无人机·km)", "全部方式"),
        ("无人机机队成本", "drone_cost_per_hour", "CNY/(架·h)", "全部方式"),
        ("骑手基础收费", "rider_initial_charge_cny", "CNY/单", "全部方式"),
        ("基础收费包含道路距离", "rider_included_distance_km", "km", "全部方式"),
        ("超出部分骑手成本", "rider_incremental_cost_per_km", "CNY/(单·km)", "全部方式"),
        ("直线距离转道路距离系数", "road_distance_factor", "—", "全部方式"),
        ("首尾接驳骑手成本系数", "feeder_rider_cost_factor", "—", "全部方式"),
        ("成功机柜自取处理成本", "locker_pickup_cost_per_order", "CNY/单", "MX、LO"),
    ]
    for label, key, unit, scope in cost_specs:
        add("成本", label, model[key], unit, scope, "统一场景值")
    choice_specs = [
        ("机柜自取常数项", "locker_asc", "无量纲"),
        ("配送时间敏感系数", "delivery_time_coefficient", "每 min"),
        ("价格敏感系数", "price_coefficient", "每 CNY"),
        ("步行距离敏感系数", "walk_distance_coefficient", "每 km"),
        ("到家配送展示时长", "home_eta_minutes", "min"),
        ("机柜配送展示时长", "locker_eta_minutes", "min"),
        ("机柜自取优惠", "locker_discount_cny", "CNY"),
        ("最大步行距离", "max_walk_km", "km"),
    ]
    for label, key, unit in choice_specs:
        add("用户选择", label, choice[key], unit, "MX、LO", "未用企业数据校准的透明场景值")
    add("计算筛选", "每个展示机柜保留的候选始发机柜数", model["display_origin_choices"], "个", "MX、LO", "因此筛选非精确")
    add("计算筛选", "每个 OD 保留的一般运输走廊数", model["general_corridor_choices"], "条", "HD、MX", "因此筛选非精确")
    data_specs = [
        ("候选位置活动窗口", f"{int(data['candidate_hour_start']):02d}:00–{int(data['candidate_hour_end']):02d}:00", "每日", "7天共112小时"),
        ("求解需求窗口", f"{int(data['demand_hour_start']):02d}:00–{int(data['demand_hour_end']):02d}:00", "每日", "7天共7小时"),
        ("OD 网格", data["od_grid_km"], "km", "数据构造"),
        ("站点活动网格", data["site_grid_km"], "km", "数据构造"),
        ("数据构造随机种子", data["seed"], "—", "数据构造"),
        ("需求放缩系数", scale_config["demand_scale"], "—", "全部规模；未放缩"),
    ]
    for label, value, unit, note in data_specs:
        add("数据构造", label, value, unit, "六档规模", note)
    solver_specs = [
        ("单方式时间预算", "time_limit", "s"),
        ("目标 Gap", "mip_gap", "比例"),
        ("线程数", "threads", "个"),
        ("求解种子", "seed", "—"),
        ("网络初始解预算", "network_start_seconds", "s"),
        ("NoRel 启发式预算", "no_rel_heur_seconds", "s"),
        ("根松弛方法", "root_relaxation_method", "—"),
        ("混合整数锥策略", "miqcp_method", "—"),
        ("原生根节点 cut passes", "root_cut_passes", "轮"),
        ("最大用户割轮数", "max_root_cut_rounds", "轮"),
        ("每轮最大割数", "max_cuts_per_round", "条"),
        ("割违反容差", "cut_violation_tolerance", "—"),
        ("局部分支半径", "local_branching_radius", "—"),
    ]
    for label, key, unit in solver_specs:
        add(
            "规模实验求解",
            label,
            solver.get(key, "未记录"),
            unit,
            "H11–H120 的 HD/MX",
            "正式运行设置",
        )
    return rows


def _solver_setting_rows(
    scale_config: Mapping[str, Any],
    formal_config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    scale = _mapping(scale_config.get("solver"), "scale solver config")
    formal = _mapping(formal_config.get("solver"), "formal solver config")

    def enabled(value: Any) -> str:
        return "开启" if value is True else "关闭"

    return [
        {
            "设置": "求解方式",
            "六档规模实验": "HD、MX",
            "正式H120": "HD、MX、LO",
            "说明": "同一数据模型，不同实验任务",
        },
        {
            "设置": "单方式时间预算",
            "六档规模实验": f"{float(scale['time_limit']):g} s",
            "正式H120": f"{float(formal['time_limit']):g} s",
            "说明": "Gurobi TimeLimit",
        },
        {
            "设置": "网络初始解预算上限",
            "六档规模实验": f"{float(scale['network_start_seconds']):g} s",
            "正式H120": f"{float(formal['network_start_seconds']):g} s",
            "说明": "计入总时间",
        },
        {
            "设置": "目标 Gap",
            "六档规模实验": f"{100 * float(scale['mip_gap']):g}%",
            "正式H120": f"{100 * float(formal['mip_gap']):g}%",
            "说明": "OPTIMAL 表示达到该容差",
        },
        {
            "设置": "线程数",
            "六档规模实验": int(scale["threads"]),
            "正式H120": int(formal["threads"]),
            "说明": "固定并行度",
        },
        {
            "设置": "求解种子",
            "六档规模实验": int(scale["seed"]),
            "正式H120": int(formal["seed"]),
            "说明": "固定随机性",
        },
        {
            "设置": "NoRel 启发式预算",
            "六档规模实验": f"{float(scale['no_rel_heur_seconds']):g} s",
            "正式H120": f"{float(formal['no_rel_heur_seconds']):g} s",
            "说明": "一致",
        },
        {
            "设置": "根松弛方法",
            "六档规模实验": int(scale["root_relaxation_method"]),
            "正式H120": int(formal["root_relaxation_method"]),
            "说明": "Method=2",
        },
        {
            "设置": "混合整数锥策略",
            "六档规模实验": scale.get("miqcp_method", "未记录"),
            "正式H120": formal.get("miqcp_method", "未记录"),
            "说明": "MIQCPMethod；与 Method 分开记录",
        },
        {
            "设置": "原生根节点 cut passes",
            "六档规模实验": int(scale["root_cut_passes"]),
            "正式H120": int(formal["root_cut_passes"]),
            "说明": "一致",
        },
        {
            "设置": "用户割",
            "六档规模实验": enabled(scale["use_user_cuts"]),
            "正式H120": enabled(formal["use_user_cuts"]),
            "说明": f"最多 {int(scale['max_root_cut_rounds'])} 轮、每轮 {int(scale['max_cuts_per_round'])} 条",
        },
        {
            "设置": "割违反容差",
            "六档规模实验": f"{float(scale['cut_violation_tolerance']):g}",
            "正式H120": f"{float(formal['cut_violation_tolerance']):g}",
            "说明": "一致",
        },
        {
            "设置": "局部分支",
            "六档规模实验": f"{enabled(scale['use_local_branching'])}，半径 {int(scale['local_branching_radius'])}",
            "正式H120": f"{enabled(formal['use_local_branching'])}，半径 {int(formal['local_branching_radius'])}",
            "说明": "一致",
        },
        {
            "设置": "静态有效不等式池",
            "六档规模实验": enabled(scale["use_valid_inequalities"]),
            "正式H120": enabled(formal["use_valid_inequalities"]),
            "说明": "当前关闭",
        },
        {
            "设置": "外部文件热启动",
            "六档规模实验": "无",
            "正式H120": "仅 HD 首方式",
            "说明": "正式实验的文件哈希记录在 run_config",
        },
        {
            "设置": "跨方式热启动",
            "六档规模实验": "HD→MX（按变量名）",
            "正式H120": "HD→MX（按变量名）",
            "说明": "LO 不接收跨方式启动",
        },
    ]


def _scale_result_rows(cases: Sequence[ScaleCase]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case in cases:
        metadata = case.layout.metadata
        for mode in SCALE_MODES:
            metrics = _summary_metrics(case.summaries[mode])
            rows.append(
                {
                    "H候选数": case.h,
                    "方式": mode,
                    "中文方式": MODE_NAMES[mode],
                    "地图边长km": 2.0 * case.layout.half_size_km,
                    "地图面积km2": 4.0 * case.layout.half_size_km**2,
                    "候选L/H/E": "/".join(
                        str(sum(site["type"] == typ for site in case.layout.sites))
                        for typ in ("L", "H", "E")
                    ),
                    "OD市场数": int(metadata["od_market_count"]),
                    "需求单每小时": round(float(metadata["total_demand_per_hour"]), 3),
                    "状态": _status_zh(metrics),
                    "目标值CNY每小时": round(metrics["objective"], 3),
                    "下界CNY每小时": round(metrics["bound"], 3),
                    "Gap百分比": round(metrics["gap_percent"], 3),
                    "运行时间s": round(metrics["runtime"], 3),
                    "开放L/H/E": f"{metrics['open_l']}/{metrics['open_h']}/{metrics['open_e']}",
                    "有流量H": metrics["active_h"],
                    "机队架数": metrics["fleet"],
                    "无人机服务单每小时": round(metrics["service"], 3),
                    "服务率百分比": round(metrics["service_percent"], 3),
                    "变量数": metrics["variables"],
                    "screening_is_exact": False,
                }
            )
    return rows


def _formal_result_rows(case: ScaleCase) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for mode in FORMAL_MODES:
        metrics = _summary_metrics(case.summaries[mode])
        rows.append(
            {
                "方式": mode,
                "中文方式": MODE_NAMES[mode],
                "状态": _status_zh(metrics),
                "目标值CNY每小时": round(metrics["objective"], 3),
                "下界CNY每小时": round(metrics["bound"], 3),
                "Gap百分比": round(metrics["gap_percent"], 3),
                "运行时间s": round(metrics["runtime"], 3),
                "开放L/H/E": f"{metrics['open_l']}/{metrics['open_h']}/{metrics['open_e']}",
                "有流量H": metrics["active_h"],
                "机队架数": metrics["fleet"],
                "无人机服务单每小时": round(metrics["service"], 3),
                "服务率百分比": round(metrics["service_percent"], 3),
                "机柜自取单每小时": round(metrics["locker_pickup"], 3),
                "骑手末端接驳单每小时": round(metrics["assisted_home"], 3),
                "成本节约率百分比": round(metrics["cost_saving_percent"], 3),
                "screening_is_exact": False,
            }
        )
    return rows


def _scale_input_rows(cases: Sequence[ScaleCase]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case in cases:
        metadata = case.layout.metadata
        nested = _mapping(metadata.get("nested_scale"), "nested scale metadata")
        rows.append(
            {
                "档位": f"H{case.h}",
                "地图边长km": 2.0 * case.layout.half_size_km,
                "地图面积km2": 4.0 * case.layout.half_size_km**2,
                "候选L/H/E": "/".join(
                    str(sum(site["type"] == typ for site in case.layout.sites))
                    for typ in ("L", "H", "E")
                ),
                "候选期原始订单": int(nested["candidate_order_count"]),
                "需求期原始订单": int(nested["demand_order_count"]),
                "OD市场数": int(metadata["od_market_count"]),
                "总需求单每小时": round(float(metadata["total_demand_per_hour"]), 3),
                "订单抽样": "否",
                "需求放缩": "否",
            }
        )
    return rows


def _write_markdown_report(
    output_dir: Path,
    parameter_rows: Sequence[Mapping[str, Any]],
    scale_inputs: Sequence[Mapping[str, Any]],
    scale_results: Sequence[Mapping[str, Any]],
    formal_results: Sequence[Mapping[str, Any]],
    solver_settings: Sequence[Mapping[str, Any]],
) -> Path:
    mx_rows = [row for row in scale_results if row["方式"] == "MX"]
    lines = [
        "# 当前 MLND 参数、规模结果与南山地图汇总",
        "",
        "## 结论摘要",
        "",
        "- 当前物理输入为飞行速度 20 m/s、单块满电池可用航程 10 km/航段；表中未重复列出可直接推导的速度或续航时间，也不含大 M。",
        "- 六档规模通过同一中心方形窗口裁剪构造；订单不抽样、需求不放缩，各窗口内重新聚类候选位置。",
        "- H11 的 HD/MX 均达到 1% 目标容差；H24 的 HD 达到目标、MX 为 1.926%；H43 以上的大规模 MX 在 300 s 内下界明显偏弱。",
        "- 正式 H120 中 HD 达到 0.935% Gap；MX/LO 是通过可行性核验的 incumbent，但不能声称接近全局最优。",
        "- 正式 H120 的 HD/MX 均有 52 个有流量 H，位于目标的 50–60 个实际建设规模内。",
        "",
        "## 六档输入规模",
        "",
        "|档位|地图边长/面积|候选 L/H/E|OD 市场|需求（单/h）|",
        "|---|---:|---:|---:|---:|",
    ]
    for row in scale_inputs:
        lines.append(
            f"|{row['档位']}|{row['地图边长km']:g} km / {row['地图面积km2']:g} km²|"
            f"{row['候选L/H/E']}|{int(row['OD市场数']):,}|{float(row['总需求单每小时']):,.3f}|"
        )
    lines.extend(
        [
            "",
            "## 六档 MX 求解能力",
            "",
            "|H|状态|目标值|Gap|运行时间|开放 L/H/E|有流量 H|变量数|",
            "|---:|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in mx_rows:
        lines.append(
            f"|{row['H候选数']}|{row['状态']}|{float(row['目标值CNY每小时']):,.2f}|"
            f"{float(row['Gap百分比']):.3f}%|{float(row['运行时间s']):.1f}s|"
            f"{row['开放L/H/E']}|{row['有流量H']}|{int(row['变量数']):,}|"
        )
    lines.extend(
        [
            "",
            "## 正式 H120 三方式",
            "",
            "|方式|状态|目标值|Gap|运行时间|开放 L/H/E|有流量 H|服务量（单/h）|",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in formal_results:
        lines.append(
            f"|{row['方式']}（{row['中文方式']}）|{row['状态']}|"
            f"{float(row['目标值CNY每小时']):,.2f}|{float(row['Gap百分比']):.3f}%|"
            f"{float(row['运行时间s']):.1f}s|{row['开放L/H/E']}|{row['有流量H']}|"
            f"{float(row['无人机服务单每小时']):,.2f}|"
        )
    lines.extend(
        [
            "",
            "## 参数表范围",
            "",
            f"完整参数表共 {len(parameter_rows)} 行，见 `parameters_used_zh.csv`；两套求解设置共 {len(solver_settings)} 行，见 `solver_settings_zh.csv`。用户选择参数被明确标为未用企业数据校准的场景值；top-8/top-10 筛选使所有 Gap 仅属于筛选后的模型。",
            "",
            "## 地图口径",
            "",
            "底图使用 OpenStreetMap Standard，并仅做去饱和、提亮的视觉处理。模型的本地公里坐标依据实例投影中心反算；为延续旧南山地图，制图副本假定来源坐标为 GCJ-02 并转换到 WGS84。当前元数据没有声明原始坐标基准，因此地图适合网络结构展示，不能作为测绘级站点定位图。",
        ]
    )
    path = output_dir / "report_summary_zh.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _write_mcp_artifact(
    output_dir: Path,
    parameter_rows: Sequence[Mapping[str, Any]],
    scale_inputs: Sequence[Mapping[str, Any]],
    scale_results: Sequence[Mapping[str, Any]],
    formal_results: Sequence[Mapping[str, Any]],
    solver_settings: Sequence[Mapping[str, Any]],
    generated_at: str,
) -> Path:
    """Write the bounded snapshot consumed by the Data Analytics report app."""

    scale_dataset = [
        {
            "h_candidates": int(row["H候选数"]),
            "mode": str(row["方式"]),
            "mode_name": str(row["中文方式"]),
            "map_side_km": float(row["地图边长km"]),
            "map_area_km2": float(row["地图面积km2"]),
            "candidate_sites": str(row["候选L/H/E"]),
            "od_markets": int(row["OD市场数"]),
            "demand_per_hour": float(row["需求单每小时"]),
            "status": str(row["状态"]),
            "objective_cny_per_hour": float(row["目标值CNY每小时"]),
            "best_bound_cny_per_hour": float(row["下界CNY每小时"]),
            "gap_rate": float(row["Gap百分比"]) / 100.0,
            "runtime_seconds": float(row["运行时间s"]),
            "open_sites": str(row["开放L/H/E"]),
            "active_lockers": int(row["有流量H"]),
            "fleet_drones": int(row["机队架数"]),
            "drone_service_per_hour": float(row["无人机服务单每小时"]),
            "service_rate": float(row["服务率百分比"]) / 100.0,
            "model_variables": int(row["变量数"]),
            "screening_is_exact": bool(row["screening_is_exact"]),
        }
        for row in scale_results
    ]
    scale_mx_dataset = [row for row in scale_dataset if row["mode"] == "MX"]
    formal_dataset = [
        {
            "mode": str(row["方式"]),
            "mode_name": str(row["中文方式"]),
            "status": str(row["状态"]),
            "objective_cny_per_hour": float(row["目标值CNY每小时"]),
            "best_bound_cny_per_hour": float(row["下界CNY每小时"]),
            "gap_rate": float(row["Gap百分比"]) / 100.0,
            "runtime_seconds": float(row["运行时间s"]),
            "open_sites": str(row["开放L/H/E"]),
            "active_lockers": int(row["有流量H"]),
            "fleet_drones": int(row["机队架数"]),
            "drone_service_per_hour": float(row["无人机服务单每小时"]),
            "service_rate": float(row["服务率百分比"]) / 100.0,
            "locker_pickup_per_hour": float(row["机柜自取单每小时"]),
            "assisted_home_per_hour": float(row["骑手末端接驳单每小时"]),
            "cost_saving_rate": float(row["成本节约率百分比"]) / 100.0,
            "screening_is_exact": bool(row["screening_is_exact"]),
        }
        for row in formal_results
    ]
    parameter_dataset = [
        {
            "category": str(row["类别"]),
            "parameter": str(row["参数"]),
            "value": str(row["本次取值"]),
            "unit": str(row["单位"]),
            "scope": str(row["适用范围"]),
            "note": str(row["性质或说明"]),
        }
        for row in parameter_rows
    ]
    solver_dataset = [
        {
            "setting": str(row["设置"]),
            "scale_experiment": str(row["六档规模实验"]),
            "formal_h120": str(row["正式H120"]),
            "note": str(row["说明"]),
        }
        for row in solver_settings
    ]
    input_dataset = [
        {
            "level": str(row["档位"]),
            "map_side_km": float(row["地图边长km"]),
            "map_area_km2": float(row["地图面积km2"]),
            "candidate_sites": str(row["候选L/H/E"]),
            "candidate_orders": int(row["候选期原始订单"]),
            "demand_orders": int(row["需求期原始订单"]),
            "od_markets": int(row["OD市场数"]),
            "demand_per_hour": float(row["总需求单每小时"]),
            "orders_sampled": False,
            "demand_rescaled": False,
        }
        for row in scale_inputs
    ]
    formal_by_mode = {row["mode"]: row for row in formal_dataset}
    scale_mx_by_h = {row["h_candidates"]: row for row in scale_mx_dataset}
    headline = [
        {
            "formal_effective_h": formal_by_mode["MX"]["active_lockers"],
            "target_h_low": 50,
            "target_h_high": 60,
            "formal_hd_gap": formal_by_mode["HD"]["gap_rate"],
            "formal_hd_runtime": formal_by_mode["HD"]["runtime_seconds"],
            "h11_mx_gap": scale_mx_by_h[11]["gap_rate"],
            "h24_mx_gap": scale_mx_by_h[24]["gap_rate"],
            "h120_mx_gap": scale_mx_by_h[120]["gap_rate"],
            "h120_mx_variables": scale_mx_by_h[120]["model_variables"],
        }
    ]
    sources = [
        {
            "id": "report_bundle",
            "label": "当前南山报告清单",
            "path": "results/nanshan_current_report_schema5/report_manifest.json",
            "query": {
                "engine": "DuckDB",
                "sql": "SELECT * FROM read_json_auto('results/nanshan_current_report_schema5/report_manifest.json')",
                "language": "sql",
                "description": "由当前 schema-5 参数、实例、summary.json 与 CSV 汇总生成。",
                "tables_used": [
                    "results/nanshan_current_report_schema5/report_manifest.json"
                ],
                "filters": ["仅使用当前 schema-5 正式结果"],
                "metric_definitions": [
                    "所有读者可见数字均来自本报告 bounded snapshot。"
                ],
            },
        },
        {
            "id": "scale_results_source",
            "label": "六档 HD/MX 正式规模实验",
            "path": "results/nanshan_current_report_schema5/scale_results_HD_MX_zh.csv",
            "query": {
                "engine": "DuckDB",
                "sql": "SELECT * FROM read_csv_auto('results/nanshan_current_report_schema5/scale_results_HD_MX_zh.csv', header=true)",
                "language": "sql",
                "description": "逐档读取 HD/MX summary.json，并与实例 metadata 交叉核对。",
                "tables_used": [
                    "results/nanshan_nested_scale_final_schema5/benchmark_summary.csv",
                    "results/nanshan_nested_scale_final_schema5/H*/{HD,MX}/summary.json",
                ],
                "filters": [
                    "H in {11,24,43,68,87,120}",
                    "demand_scale=1",
                    "screening_is_exact=false",
                ],
                "metric_definitions": [
                    "Gap = summary.mip_gap，按小数比例储存并按百分比显示。",
                    "有流量 H = facilities.active_lockers。",
                    "服务率 = total_drone_realized / total_demand。",
                ],
            },
        },
        {
            "id": "formal_results_source",
            "label": "正式 H120 三方式结果",
            "path": "results/nanshan_current_report_schema5/H120_three_modes_zh.csv",
            "query": {
                "engine": "DuckDB",
                "sql": "SELECT * FROM read_csv_auto('results/nanshan_current_report_schema5/H120_three_modes_zh.csv', header=true)",
                "language": "sql",
                "description": "读取正式 H120 的 HD、MX、LO summary.json。",
                "tables_used": [
                    "results/nanshan_h120_final_schema5/{HD,MX,LO}/summary.json"
                ],
                "filters": ["H=120", "screening_is_exact=false"],
                "metric_definitions": [
                    "服务量 = demand_funnel_orders_per_hour.total_drone_realized。",
                    "成本节约率 = summary.cost_saving_percent / 100。",
                ],
            },
        },
        {
            "id": "parameters_source",
            "label": "当前模型与数据参数",
            "path": "results/nanshan_current_report_schema5/parameters_used_zh.csv",
            "query": {
                "engine": "DuckDB",
                "sql": "SELECT * FROM read_csv_auto('results/nanshan_current_report_schema5/parameters_used_zh.csv', header=true)",
                "language": "sql",
                "description": "读取 run_config 中实际进入模型和数据构造的输入；排除大 M 与可直接推导量。",
                "tables_used": [
                    "results/nanshan_nested_scale_final_schema5/H11/run_config.json",
                    "results/nanshan_h120_final_schema5/run_config.json",
                ],
                "filters": ["不含大 M", "不重复 72 km/h 或 8.33 min 等推导值"],
                "metric_definitions": [
                    "20 m/s 与 10 km/航段为用户给定物理输入。"
                ],
            },
        },
        {
            "id": "solver_source",
            "label": "两套求解设置",
            "path": "results/nanshan_current_report_schema5/solver_settings_zh.csv",
            "query": {
                "engine": "DuckDB",
                "sql": "SELECT * FROM read_csv_auto('results/nanshan_current_report_schema5/solver_settings_zh.csv', header=true)",
                "language": "sql",
                "description": "对照六档规模实验与正式 H120 的 run_config.solver。",
                "tables_used": [
                    "results/nanshan_nested_scale_final_schema5/H11/run_config.json",
                    "results/nanshan_h120_final_schema5/run_config.json",
                ],
                "filters": ["正式记录设置，不使用默认值推断"],
            },
        },
    ]
    cards = [
        {
            "id": "effective_h_card",
            "dataset": "headline",
            "sourceId": "formal_results_source",
            "description": "正式 H120 混合配送中承担正流量的机柜数量。",
            "metrics": [
                {"label": "H120 有流量 H", "field": "formal_effective_h", "format": "number"},
                {"label": "目标下限", "field": "target_h_low", "format": "number"},
                {"label": "目标上限", "field": "target_h_high", "format": "number"},
            ],
        },
        {
            "id": "formal_hd_card",
            "dataset": "headline",
            "sourceId": "formal_results_source",
            "description": "正式 H120 到家配送达到配置的 1% 最优性容差。",
            "metrics": [
                {"label": "H120 HD Gap", "field": "formal_hd_gap", "format": "percent"},
                {"label": "运行秒数", "field": "formal_hd_runtime", "format": "number"},
            ],
        },
        {
            "id": "small_scale_card",
            "dataset": "headline",
            "sourceId": "scale_results_source",
            "description": "小规模 MX 的求解质量；H24 在 300 s 时略高于 1%。",
            "metrics": [
                {"label": "H11 MX Gap", "field": "h11_mx_gap", "format": "percent"},
                {"label": "H24 MX Gap", "field": "h24_mx_gap", "format": "percent"},
            ],
        },
        {
            "id": "large_scale_card",
            "dataset": "headline",
            "sourceId": "scale_results_source",
            "description": "H120 MX 的 incumbent 可行，但筛选后模型下界仍很弱。",
            "metrics": [
                {"label": "H120 MX Gap", "field": "h120_mx_gap", "format": "percent"},
                {"label": "模型变量", "field": "h120_mx_variables", "format": "compact"},
            ],
        },
    ]
    charts = [
        {
            "id": "gap_by_scale",
            "title": "六档规模的求解 Gap",
            "subtitle": "H43 起 MX 的下界明显弱化；HD 在同一时限下更稳定。",
            "intent": "comparison",
            "question": "候选 H 规模增加时，HD 与 MX 的求解 Gap 如何变化？",
            "rationale": "按候选 H 数排序的双系列折线可直接比较两种方式的规模退化。",
            "type": "line",
            "dataset": "scale_results",
            "sourceId": "scale_results_source",
            "encodings": {
                "x": {"field": "h_candidates", "type": "quantitative", "label": "候选 H 数"},
                "y": {"field": "gap_rate", "type": "quantitative", "format": "percent", "label": "Gap"},
                "color": {"field": "mode", "type": "nominal", "label": "方式"},
                "tooltip": [
                    {"field": "status", "type": "text", "label": "状态"},
                    {"field": "runtime_seconds", "type": "quantitative", "label": "运行时间", "unit": "s"},
                    {"field": "model_variables", "type": "quantitative", "format": "compact", "label": "变量数"},
                    {"field": "active_lockers", "type": "quantitative", "label": "有流量 H"},
                ],
            },
            "xAxisTitle": "候选 H 数",
            "yAxisTitle": "Gap",
            "valueFormat": "percent",
            "referenceLines": [
                {"axis": "y", "value": 0.01, "label": "1% 目标", "lineStyle": "dashed", "color": "red"}
            ],
            "legend": {"position": "bottom", "title": "方式"},
            "layout": "full",
        }
    ]
    tables = [
        {
            "id": "scale_mx_table",
            "title": "六档 MX 结果",
            "subtitle": "每档 300 s；所有 Gap 均为筛选后模型 Gap。",
            "dataset": "scale_mx_results",
            "sourceId": "scale_results_source",
            "defaultSort": {"field": "h_candidates", "direction": "asc"},
            "density": "dense",
            "layout": "full",
            "columns": [
                {"field": "h_candidates", "label": "候选 H", "format": "number"},
                {"field": "map_side_km", "label": "边长 km", "format": "number"},
                {"field": "candidate_sites", "label": "候选 L/H/E", "type": "text"},
                {"field": "status", "label": "状态", "type": "text"},
                {"field": "gap_rate", "label": "Gap", "format": "percent"},
                {"field": "runtime_seconds", "label": "时间 s", "format": "number"},
                {"field": "open_sites", "label": "开放 L/H/E", "type": "text"},
                {"field": "active_lockers", "label": "有流量 H", "format": "number"},
                {"field": "model_variables", "label": "变量数", "format": "compact"},
            ],
        },
        {
            "id": "formal_h120_table",
            "title": "正式 H120 三方式结果",
            "subtitle": "正式实验为 900 s 设置；HD 先接收外部可行解，MX 再接收 HD 变量名热启动。",
            "dataset": "formal_results",
            "sourceId": "formal_results_source",
            "defaultSort": {"field": "mode", "direction": "asc"},
            "density": "dense",
            "layout": "full",
            "columns": [
                {"field": "mode", "label": "方式", "type": "text"},
                {"field": "status", "label": "状态", "type": "text"},
                {"field": "objective_cny_per_hour", "label": "目标值 CNY/h", "format": "number"},
                {"field": "gap_rate", "label": "Gap", "format": "percent"},
                {"field": "runtime_seconds", "label": "时间 s", "format": "number"},
                {"field": "open_sites", "label": "开放 L/H/E", "type": "text"},
                {"field": "active_lockers", "label": "有流量 H", "format": "number"},
                {"field": "drone_service_per_hour", "label": "服务单/h", "format": "number"},
                {"field": "cost_saving_rate", "label": "节约率", "format": "percent"},
            ],
        },
        {
            "id": "parameter_table",
            "title": "当前实际使用参数",
            "subtitle": "不含大 M，也不重复列出由 20 m/s 和 10 km 可直接推导的量。",
            "dataset": "parameters",
            "sourceId": "parameters_source",
            "defaultSort": {"field": "category", "direction": "asc"},
            "density": "dense",
            "layout": "full",
            "columns": [
                {"field": "category", "label": "类别", "type": "text"},
                {"field": "parameter", "label": "参数", "type": "text"},
                {"field": "value", "label": "取值", "type": "text"},
                {"field": "unit", "label": "单位", "type": "text"},
                {"field": "scope", "label": "适用范围", "type": "text"},
                {"field": "note", "label": "说明", "type": "text"},
            ],
        },
        {
            "id": "solver_table",
            "title": "求解设置对照",
            "subtitle": "规模实验与正式 H120 分开报告，避免把 300 s 和 900 s 结果混在一起。",
            "dataset": "solver_settings",
            "sourceId": "solver_source",
            "defaultSort": {"field": "setting", "direction": "asc"},
            "density": "dense",
            "layout": "full",
            "columns": [
                {"field": "setting", "label": "设置", "type": "text"},
                {"field": "scale_experiment", "label": "六档规模实验", "type": "text"},
                {"field": "formal_h120", "label": "正式 H120", "type": "text"},
                {"field": "note", "label": "说明", "type": "text"},
            ],
        },
    ]
    blocks = [
        {"id": "title", "type": "markdown", "body": "# 当前 MLND 参数、规模结果与南山地图汇总", "layout": "full"},
        {
            "id": "executive_summary",
            "type": "markdown",
            "sourceId": "report_bundle",
            "body": (
                "## Executive Summary\n\n"
                "当前网络已经在南山 H120 正式算例中形成 52 个有流量 H，落在目标的 50–60 个实际建设规模内。"
                "小规模 H11 的 HD/MX 均达到 1% 容差，H24 的 MX 为 1.926%；H43 以后 MX 的主要瓶颈是下界弱化。"
                "正式 H120 的 HD 达到 0.935% Gap；MX 和 LO 应报告为已核验可行解，不能声称接近全局最优。"
            ),
            "layout": "full",
        },
        {"id": "headline_metrics", "type": "metric-strip", "cardIds": [card["id"] for card in cards], "layout": "full"},
        {
            "id": "scale_findings",
            "type": "markdown",
            "sourceId": "scale_results_source",
            "body": (
                "## 不同规模的求解表现\n\n"
                "规模递减严格采用同中心地图裁剪：订单不抽样、需求不放缩；候选点在各裁剪窗口内重新聚类。"
                "因此下图反映地理范围和候选网络共同增长，而不是人为增减订单。"
            ),
            "layout": "full",
        },
        {"id": "gap_chart_block", "type": "chart", "chartId": "gap_by_scale", "layout": "full"},
        {"id": "scale_table_block", "type": "table", "tableId": "scale_mx_table", "layout": "full"},
        {
            "id": "formal_findings",
            "type": "markdown",
            "sourceId": "formal_results_source",
            "body": (
                "## 正式 H120 三方式\n\n"
                "HD 与 MX 均服务 562.58 单/h，且全部表现为骑手末端接驳；LO 服务 37.11 单/h，全部为机柜自取。"
                "HD/MX 的网络规模合理，但 MX 与 LO 在时限内的下界不具决策解释力。"
            ),
            "layout": "full",
        },
        {"id": "formal_table_block", "type": "table", "tableId": "formal_h120_table", "layout": "full"},
        {
            "id": "parameter_section",
            "type": "markdown",
            "sourceId": "parameters_source",
            "body": (
                "## 当前模型参数\n\n"
                "20 m/s 与 10 km/航段已作为直接物理输入进入模型。"
                "代码没有再分别输入电池额定能量、耗电率和储备比例；用户选择参数仍是透明但未校准的场景值。"
            ),
            "layout": "full",
        },
        {"id": "parameter_table_block", "type": "table", "tableId": "parameter_table", "layout": "full"},
        {
            "id": "solver_section",
            "type": "markdown",
            "sourceId": "solver_source",
            "body": "## 求解设置\n\n六档规模实验采用每方式 300 s；正式 H120 采用每方式 900 s，并单独记录热启动来源。",
            "layout": "full",
        },
        {"id": "solver_table_block", "type": "table", "tableId": "solver_table", "layout": "full"},
        {
            "id": "map_section",
            "type": "markdown",
            "sourceId": "report_bundle",
            "body": (
                "## 地图交付\n\n"
                "已生成六档 MX 合图、六张单档地图、H120 三方式图、路线结构图、机柜利用率图和裁剪窗口图。"
                "底图为 OpenStreetMap Standard，红色虚线为 OSM 南山区关系 R5664195。"
                "制图副本沿用旧代码的 GCJ-02→WGS84 假设；当前实例元数据没有声明来源坐标基准，所以这些图用于网络结构展示，不用于测绘级定位。"
            ),
            "layout": "full",
        },
        {
            "id": "limitations",
            "type": "markdown",
            "sourceId": "report_bundle",
            "body": (
                "## 局限与稳健性\n\n"
                "1. MNL 用户选择参数尚未用企业数据校准。\n"
                "2. top-8/top-10 筛选使 Gap 只属于筛选后的模型。\n"
                "3. 大规模 MX/LO 的 incumbent 可行，但下界偏弱。\n"
                "4. 地图坐标基准仍需用一个已知地标做人工抽查。"
            ),
            "layout": "full",
        },
        {
            "id": "next_steps",
            "type": "markdown",
            "body": (
                "## 下一步\n\n"
                "当前结果可直接用于内部汇报。若要进入论文主文，优先补做坐标基准抽查，并把 H43、H68、H87、H120 的 MX 求解设置作为下界强化实验。"
            ),
            "layout": "full",
        },
        {
            "id": "further_questions",
            "type": "markdown",
            "body": (
                "## Further Questions\n\n"
                "企业侧最有价值的后续数据是逐站固定/容量成本、换电服务时间分布，以及能够校准用户选择参数的自取与骑手接驳行为记录。"
            ),
            "layout": "full",
        },
    ]
    artifact = {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": "当前 MLND 参数、规模结果与南山地图汇总",
            "description": "当前 schema-5 南山算例的参数、规模求解能力、正式 H120 结果与地图口径。",
            "generatedAt": generated_at,
            "blocks": blocks,
            "cards": cards,
            "charts": charts,
            "tables": tables,
            "sources": sources,
        },
        "snapshot": {
            "version": 1,
            "generatedAt": generated_at,
            "status": "ready",
            "datasets": {
                "headline": headline,
                "scale_inputs": input_dataset,
                "scale_results": scale_dataset,
                "scale_mx_results": scale_mx_dataset,
                "formal_results": formal_dataset,
                "parameters": parameter_dataset,
                "solver_settings": solver_dataset,
            },
        },
        "sources": sources,
    }
    path = output_dir / "mcp_artifact.json"
    path.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


def _git_head(repo_root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def write_nanshan_geographic_report(
    instance_root: str | Path,
    scale_results: str | Path,
    formal_results: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Generate audited Chinese tables plus multiple geographic maps."""

    instance_root = Path(instance_root).expanduser().resolve()
    scale_results = Path(scale_results).expanduser().resolve()
    formal_results = Path(formal_results).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    maps_dir = output_dir / "maps"
    cache_dir = output_dir / "basemap_cache"

    plt.rcParams.update(
        {
            "font.family": ["Arial Unicode MS", "Lantinghei SC", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "figure.dpi": 120,
        }
    )

    cases = _load_scale_cases(instance_root, scale_results)
    full_instance = cases[-1].instance_dir
    formal_case = _load_case(120, full_instance, formal_results, FORMAL_MODES)
    scale_config = cases[0].run_config
    formal_model = formal_case.run_config.get("model")
    if formal_model != scale_config.get("model"):
        raise ValueError("formal H120 and scale experiments use different model parameters")

    full_bounds = _window_bounds(
        cases[-1].layout.metadata,
        cases[-1].layout.half_size_km,
        padding_km=0.8,
    )
    basemap = build_basemap(full_bounds, cache_dir, zoom=TILE_ZOOM)
    boundary = _load_nanshan_boundary(cache_dir / "nanshan_R5664195.geojson")
    boundary_rings = _outer_rings(boundary)

    parameter_rows = _parameter_rows(scale_config)
    solver_setting_rows = _solver_setting_rows(
        scale_config,
        formal_case.run_config,
    )
    scale_input_rows = _scale_input_rows(cases)
    scale_result_rows = _scale_result_rows(cases)
    formal_result_rows = _formal_result_rows(formal_case)
    parameter_csv = output_dir / "parameters_used_zh.csv"
    scale_input_csv = output_dir / "scale_inputs_zh.csv"
    scale_result_csv = output_dir / "scale_results_HD_MX_zh.csv"
    formal_result_csv = output_dir / "H120_three_modes_zh.csv"
    solver_setting_csv = output_dir / "solver_settings_zh.csv"
    _write_csv(parameter_csv, parameter_rows)
    _write_csv(scale_input_csv, scale_input_rows)
    _write_csv(scale_result_csv, scale_result_rows)
    _write_csv(formal_result_csv, formal_result_rows)
    _write_csv(solver_setting_csv, solver_setting_rows)

    map_paths = _plot_scale_maps(cases, basemap, boundary_rings, maps_dir)
    map_paths.extend(
        [
            _plot_formal_modes(formal_case, basemap, boundary_rings, maps_dir),
            _plot_route_structure(formal_case, basemap, boundary_rings, maps_dir),
            _plot_utilization(formal_case, basemap, boundary_rings, maps_dir),
            _plot_scale_windows(cases, basemap, boundary_rings, maps_dir),
        ]
    )
    markdown = _write_markdown_report(
        output_dir,
        parameter_rows,
        scale_input_rows,
        scale_result_rows,
        formal_result_rows,
        solver_setting_rows,
    )

    repo_root = Path(__file__).resolve().parents[2]
    generated_at = datetime.now(UTC).isoformat()
    artifact_path = _write_mcp_artifact(
        output_dir,
        parameter_rows,
        scale_input_rows,
        scale_result_rows,
        formal_result_rows,
        solver_setting_rows,
        generated_at,
    )
    manifest = {
        "schema_version": 1,
        "generated_at": generated_at,
        "git_head": _git_head(repo_root),
        "result_contract": {
            "schema_version": RESULT_SCHEMA_VERSION,
            "formulation_id": FORMULATION_ID,
        },
        "inputs": {
            "instance_root": str(instance_root),
            "scale_results": str(scale_results),
            "formal_results": str(formal_results),
            "scale_manifest_sha256": _sha256(instance_root / "scale_manifest.json"),
            "scale_summary_sha256": _sha256(scale_results / "benchmark_summary.csv"),
            "formal_run_config_sha256": _sha256(formal_results / "run_config.json"),
        },
        "coordinate_contract": {
            "instance_coordinate_system": "local_km",
            "source_datum_assumption_for_plotting_only": "GCJ-02",
            "basemap_datum": "WGS84 / Web Mercator tiles",
            "model_coordinates_modified": False,
            "caveat": _COORDINATE_NOTE,
        },
        "basemap": {
            "provider": "OpenStreetMap Standard (lightened for presentation)",
            "data": "OpenStreetMap contributors",
            "tile_url": TILE_URL,
            "zoom": basemap.zoom,
            "tiles_in_mosaic": basemap.tile_count,
            "cache": str(cache_dir),
            "attribution": BASEMAP_ATTRIBUTION,
            "copyright_url": OSM_COPYRIGHT_URL,
            "nanshan_boundary_relation": NANSHAN_RELATION,
            "boundary_loaded": bool(boundary_rings),
        },
        "tables": [
            str(parameter_csv),
            str(scale_input_csv),
            str(scale_result_csv),
            str(formal_result_csv),
            str(solver_setting_csv),
        ],
        "maps": [str(path) for path in map_paths],
        "report": str(markdown),
        "mcp_artifact": str(artifact_path),
        "limitations": [
            "MNL choice parameters are transparent uncalibrated scenario values.",
            "top-8/top-10 screening makes every reported gap a screened-model gap.",
            "MX and LO formal H120 lower bounds are weak at the time limit.",
            "The source datum is not declared in current instance metadata; GCJ-02 is a historical plotting assumption.",
        ],
    }
    manifest_path = output_dir / "report_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    manifest["manifest"] = str(manifest_path)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create current-contract Chinese parameter/result tables and Nanshan maps."
    )
    parser.add_argument(
        "--instance-root",
        default="data/instances/nanshan_nested_h120",
    )
    parser.add_argument(
        "--scale-results",
        default="results/nanshan_nested_scale_final_schema5",
    )
    parser.add_argument(
        "--formal-results",
        default="results/nanshan_h120_final_schema5",
    )
    parser.add_argument(
        "--output-dir",
        default="results/nanshan_current_report_schema5",
    )
    args = parser.parse_args()
    manifest = write_nanshan_geographic_report(
        args.instance_root,
        args.scale_results,
        args.formal_results,
        args.output_dir,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
