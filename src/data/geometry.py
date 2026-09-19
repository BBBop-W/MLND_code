"""Projection, square-crop membership and raw-order identity."""

from __future__ import annotations
import json
import math
from .generator import RawOrder


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


def _in_hour_window(hour: int, start: int, end: int) -> bool:
    if end == 24 or start < end:
        return start <= hour < end
    return hour >= start or hour < end
