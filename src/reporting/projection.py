"""Coordinate transforms used only for geographic plotting."""

from __future__ import annotations

import math
from typing import Any, Mapping
import numpy as np

R = 6378137.0


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
        (20.0 * math.sin(6.0 * lon * math.pi) + 20.0 * math.sin(2.0 * lon * math.pi))
        * 2.0
        / 3.0
    )
    result += (
        (20.0 * math.sin(lat * math.pi) + 40.0 * math.sin(lat / 3.0 * math.pi))
        * 2.0
        / 3.0
    )
    result += (
        (
            160.0 * math.sin(lat / 12.0 * math.pi)
            + 320.0 * math.sin(lat * math.pi / 30.0)
        )
        * 2.0
        / 3.0
    )
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
        (20.0 * math.sin(6.0 * lon * math.pi) + 20.0 * math.sin(2.0 * lon * math.pi))
        * 2.0
        / 3.0
    )
    result += (
        (20.0 * math.sin(lon * math.pi) + 40.0 * math.sin(lon / 3.0 * math.pi))
        * 2.0
        / 3.0
    )
    result += (
        (
            150.0 * math.sin(lon / 12.0 * math.pi)
            + 300.0 * math.sin(lon / 30.0 * math.pi)
        )
        * 2.0
        / 3.0
    )
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
    y = (1.0 - math.asinh(math.tan(math.radians(latitude))) / math.pi) / 2.0 * n
    return x, y


def tile_to_lonlat(x: float, y: float, zoom: int) -> tuple[float, float]:
    n = 2.0**zoom
    lon = x / n * 360.0 - 180.0
    lat = math.degrees(math.atan(math.sinh(math.pi * (1.0 - 2.0 * y / n))))
    return lon, lat


def _mercator(points):
    p = np.asarray(points, dtype=float)
    return np.column_stack(
        (
            R * np.radians(p[:, 0]),
            R * np.log(np.tan(np.pi / 4 + np.radians(p[:, 1]) / 2)),
        )
    )


def _inverse_mercator(points):
    p = np.asarray(points, dtype=float)
    return np.column_stack(
        (
            np.degrees(p[:, 0] / R),
            np.degrees(2 * np.arctan(np.exp(p[:, 1] / R)) - np.pi / 2),
        )
    )
