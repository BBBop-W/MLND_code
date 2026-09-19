"""Raw TSL/Nanshan adapters and canonical OD-instance generation."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import platform
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterator
from zoneinfo import ZoneInfo

import numpy as np

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
import sklearn
from sklearn.cluster import KMeans

from .generation_config import DataConfig, _window_hours, to_dict


@dataclass(frozen=True)
class RawOrder:
    order_id: str
    day_id: str
    hour: int
    pickup_lon: float
    pickup_lat: float
    dropoff_lon: float
    dropoff_lat: float
    poi_id: str = ""
    event_day_id: str | None = None


class TSLAdapter:
    """Read one accepted waybill per TSL order in Asia/Shanghai time."""

    timezone = ZoneInfo("Asia/Shanghai")

    def iter_orders(self, path: str) -> Iterator[RawOrder]:
        with open(path, newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                if row["is_courier_grabbed"] != "1":
                    continue
                timestamp = int(row["platform_order_time"])
                local_time = datetime.fromtimestamp(timestamp, self.timezone)
                yield RawOrder(
                    order_id=row["order_id"],
                    day_id=row["dt"],
                    hour=local_time.hour,
                    pickup_lon=float(row["sender_lng"]) / 1_000_000.0,
                    pickup_lat=float(row["sender_lat"]) / 1_000_000.0,
                    dropoff_lon=float(row["recipient_lng"]) / 1_000_000.0,
                    dropoff_lat=float(row["recipient_lat"]) / 1_000_000.0,
                    poi_id=row["poi_id"],
                    event_day_id=local_time.strftime("%Y%m%d"),
                )


class NanshanAdapter:
    """Read the anonymized Nanshan order sample."""

    def iter_orders(self, path: str) -> Iterator[RawOrder]:
        with open(path, newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                hour = int(row["customer_pay_time"].split(":", 1)[0])
                yield RawOrder(
                    order_id=row["order_seq_id"],
                    day_id=row["dt_seq_id"],
                    hour=hour,
                    pickup_lon=float(row["pickup_longitude"]),
                    pickup_lat=float(row["pickup_latitude"]),
                    dropoff_lon=float(row["dropoff_longitude"]),
                    dropoff_lat=float(row["dropoff_latitude"]),
                )


ADAPTERS = {"tsl": TSLAdapter, "nanshan": NanshanAdapter}


class DataGenerator:
    """Generate a canonical instance while preserving OD independently of sites."""

    def generate(self, config: DataConfig) -> str:
        config.validate()
        adapter = ADAPTERS[config.source]()
        requested_days = tuple(sorted(set(config.dates)))
        available_days, center = self._scan_center(adapter, config, set(requested_days))
        selected_days = (
            requested_days if requested_days else tuple(sorted(available_days))
        )
        selected_day_set = set(selected_days)
        projection = _Projection(*center)

        site_activity: dict[tuple[int, int], list[float]] = {}
        pickup_activity: dict[tuple[int, int], list[float]] = {}
        midpoint_activity: dict[tuple[int, int], list[float]] = {}
        origin_nodes: dict[tuple[int, int], list[float]] = {}
        destination_nodes: dict[tuple[int, int], list[float]] = {}
        od_counts: dict[tuple[tuple[int, int], tuple[int, int]], int] = defaultdict(int)
        accepted_orders = 0
        candidate_orders = 0
        demand_orders = 0
        timestamp_date_mismatch_orders = 0

        for order in adapter.iter_orders(config.raw_path):
            accepted_orders += 1
            if order.event_day_id is not None and order.event_day_id != order.day_id:
                timestamp_date_mismatch_orders += 1
            if order.day_id not in selected_day_set:
                continue
            ox, oy = projection.project(order.pickup_lon, order.pickup_lat)
            dx, dy = projection.project(order.dropoff_lon, order.dropoff_lat)
            if not _inside_square(
                ox, oy, config.map_half_size_km
            ) or not _inside_square(dx, dy, config.map_half_size_km):
                continue

            if _in_hour_window(
                order.hour, config.candidate_hour_start, config.candidate_hour_end
            ):
                candidate_orders += 1
                _accumulate(site_activity, ox, oy, config.site_grid_km)
                _accumulate(site_activity, dx, dy, config.site_grid_km)
                _accumulate(pickup_activity, ox, oy, config.site_grid_km)
                _accumulate(
                    midpoint_activity,
                    (ox + dx) / 2.0,
                    (oy + dy) / 2.0,
                    config.site_grid_km,
                )

            if _in_hour_window(
                order.hour, config.demand_hour_start, config.demand_hour_end
            ):
                demand_orders += 1
                o_cell = _accumulate(origin_nodes, ox, oy, config.od_grid_km)
                d_cell = _accumulate(destination_nodes, dx, dy, config.od_grid_km)
                od_counts[o_cell, d_cell] += 1

        candidate_exposure = len(selected_days) * _window_hours(
            config.candidate_hour_start, config.candidate_hour_end
        )
        demand_exposure = len(selected_days) * _window_hours(
            config.demand_hour_start, config.demand_hour_end
        )
        lockers = _weighted_medoids(site_activity, config.candidate_count, config.seed)
        bases = _weighted_medoids(pickup_activity, config.num_bases, config.seed + 1)
        swaps = _weighted_medoids(midpoint_activity, config.num_swaps, config.seed + 2)

        output_dir = Path(config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self._write_sites(
            output_dir / "sites.csv", lockers, bases, swaps, candidate_exposure
        )
        origin_ids, destination_ids = self._write_od_nodes(
            output_dir / "od_nodes.csv", origin_nodes, destination_nodes
        )
        total_rate = self._write_od_demand(
            output_dir / "od_demand.csv",
            od_counts,
            origin_ids,
            destination_ids,
            origin_nodes,
            destination_nodes,
            demand_exposure,
        )
        self._write_period(
            output_dir / "periods.csv",
            config,
            selected_days,
            demand_exposure,
            demand_orders,
            total_rate,
        )
        normalized_data_config = to_dict(config)
        normalized_data_config["dates"] = list(selected_days)
        metadata = {
            "schema_version": 2,
            "source": config.source,
            "raw_path": os.path.abspath(config.raw_path),
            "raw_file_size_bytes": os.path.getsize(config.raw_path),
            "raw_file_sha256": _sha256_file(config.raw_path),
            "date_basis": (
                "source_partition_dt" if config.source == "tsl" else "dt_seq_id"
            ),
            "hour_basis": (
                "Asia/Shanghai event timestamp"
                if config.source == "tsl"
                else "customer_pay_time"
            ),
            "timestamp_date_mismatch_orders_all_days": timestamp_date_mismatch_orders,
            "coordinate_system": "local_km",
            "projection_center_lon": center[0],
            "projection_center_lat": center[1],
            "selected_days": list(selected_days),
            "source_orders_all_days": accepted_orders,
            "accepted_waybills_all_days": (
                accepted_orders if config.source == "tsl" else None
            ),
            "candidate_orders_in_area": candidate_orders,
            "demand_orders_in_area": demand_orders,
            "candidate_exposure_hours": candidate_exposure,
            "demand_exposure_hours": demand_exposure,
            "total_demand_per_hour": total_rate,
            "od_market_count": len(od_counts),
            "origin_node_count": len(origin_ids),
            "destination_node_count": len(destination_ids),
            "data_config": normalized_data_config,
            "software_versions": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "scikit_learn": sklearn.__version__,
            },
        }
        with open(output_dir / "metadata.json", "w", encoding="utf-8") as handle:
            json.dump(metadata, handle, ensure_ascii=False, indent=2)

        print(
            f"Prepared {config.source}: OD={len(od_counts):,}, demand={total_rate:,.2f}/h, "
            f"H={len(lockers)}, L={len(bases)}, E={len(swaps)}, exposure={demand_exposure}h"
        )
        if timestamp_date_mismatch_orders:
            print(
                "Data QA: "
                f"{timestamp_date_mismatch_orders:,} accepted TSL rows have event-date != dt; "
                "dt is used as the complete statistical-day partition."
            )
        print(f"Canonical instance: {output_dir}")
        return str(output_dir)

    def _scan_center(
        self,
        adapter: object,
        config: DataConfig,
        requested_days: set[str],
    ) -> tuple[set[str], tuple[float, float]]:
        days: set[str] = set()
        if config.source == "tsl":
            poi_count: dict[str, int] = defaultdict(int)
            poi_location: dict[str, tuple[float, float]] = {}
            for order in adapter.iter_orders(config.raw_path):
                days.add(order.day_id)
                if requested_days and order.day_id not in requested_days:
                    continue
                if _in_hour_window(
                    order.hour, config.candidate_hour_start, config.candidate_hour_end
                ):
                    poi_count[order.poi_id] += 1
                    poi_location.setdefault(
                        order.poi_id, (order.pickup_lon, order.pickup_lat)
                    )
            _validate_requested_days(requested_days, days)
            if not poi_count:
                raise ValueError(
                    "no TSL candidate-period orders for the selected dates"
                )
            center_poi = max(poi_count, key=lambda poi: (poi_count[poi], poi))
            return days, poi_location[center_poi]

        min_lon = min_lat = math.inf
        max_lon = max_lat = -math.inf
        for order in adapter.iter_orders(config.raw_path):
            days.add(order.day_id)
            if requested_days and order.day_id not in requested_days:
                continue
            if not _in_hour_window(
                order.hour, config.candidate_hour_start, config.candidate_hour_end
            ):
                continue
            min_lon = min(min_lon, order.pickup_lon, order.dropoff_lon)
            max_lon = max(max_lon, order.pickup_lon, order.dropoff_lon)
            min_lat = min(min_lat, order.pickup_lat, order.dropoff_lat)
            max_lat = max(max_lat, order.pickup_lat, order.dropoff_lat)
        _validate_requested_days(requested_days, days)
        if not math.isfinite(min_lon):
            raise ValueError(
                "no Nanshan candidate-period orders for the selected dates"
            )
        return days, ((min_lon + max_lon) / 2.0, (min_lat + max_lat) / 2.0)

    @staticmethod
    def _write_sites(
        path: Path,
        lockers: list[tuple[float, float, float]],
        bases: list[tuple[float, float, float]],
        swaps: list[tuple[float, float, float]],
        exposure: int,
    ) -> None:
        with open(path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=["site_id", "type", "x_km", "y_km", "activity_rate"]
            )
            writer.writeheader()
            for node_type, prefix, values in (
                ("H", "H", lockers),
                ("L", "L", bases),
                ("E", "E", swaps),
            ):
                for index, (x, y, weight) in enumerate(values, 1):
                    writer.writerow(
                        {
                            "site_id": f"{prefix}{index:03d}",
                            "type": node_type,
                            "x_km": f"{x:.8f}",
                            "y_km": f"{y:.8f}",
                            "activity_rate": f"{weight / exposure:.8f}",
                        }
                    )

    @staticmethod
    def _write_od_nodes(
        path: Path,
        origins: dict[tuple[int, int], list[float]],
        destinations: dict[tuple[int, int], list[float]],
    ) -> tuple[dict[tuple[int, int], str], dict[tuple[int, int], str]]:
        origin_ids = {
            cell: f"O{index:04d}" for index, cell in enumerate(sorted(origins), 1)
        }
        destination_ids = {
            cell: f"D{index:04d}" for index, cell in enumerate(sorted(destinations), 1)
        }
        with open(path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["node_id", "role", "x_km", "y_km", "raw_endpoint_count"],
            )
            writer.writeheader()
            for role, values, identifiers in (
                ("origin", origins, origin_ids),
                ("destination", destinations, destination_ids),
            ):
                for cell in sorted(values):
                    count, sum_x, sum_y = values[cell]
                    writer.writerow(
                        {
                            "node_id": identifiers[cell],
                            "role": role,
                            "x_km": f"{sum_x / count:.8f}",
                            "y_km": f"{sum_y / count:.8f}",
                            "raw_endpoint_count": int(count),
                        }
                    )
        return origin_ids, destination_ids

    @staticmethod
    def _write_od_demand(
        path: Path,
        od_counts: dict[tuple[tuple[int, int], tuple[int, int]], int],
        origin_ids: dict[tuple[int, int], str],
        destination_ids: dict[tuple[int, int], str],
        origin_nodes: dict[tuple[int, int], list[float]],
        destination_nodes: dict[tuple[int, int], list[float]],
        exposure: int,
    ) -> float:
        total_rate = 0.0
        with open(path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "od_id",
                    "origin_node",
                    "destination_node",
                    "order_count",
                    "exposure_hours",
                    "lambda_per_hour",
                    "direct_distance_km",
                ],
            )
            writer.writeheader()
            for index, ((o_cell, d_cell), count) in enumerate(
                sorted(od_counts.items()), 1
            ):
                ox = origin_nodes[o_cell][1] / origin_nodes[o_cell][0]
                oy = origin_nodes[o_cell][2] / origin_nodes[o_cell][0]
                dx = destination_nodes[d_cell][1] / destination_nodes[d_cell][0]
                dy = destination_nodes[d_cell][2] / destination_nodes[d_cell][0]
                rate = count / exposure
                total_rate += rate
                writer.writerow(
                    {
                        "od_id": f"OD{index:06d}",
                        "origin_node": origin_ids[o_cell],
                        "destination_node": destination_ids[d_cell],
                        "order_count": count,
                        "exposure_hours": exposure,
                        "lambda_per_hour": f"{rate:.10f}",
                        "direct_distance_km": f"{math.hypot(ox - dx, oy - dy):.8f}",
                    }
                )
        return total_rate

    @staticmethod
    def _write_period(
        path: Path,
        config: DataConfig,
        days: tuple[str, ...],
        exposure: int,
        orders: int,
        rate: float,
    ) -> None:
        with open(path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "period_id",
                    "start_hour",
                    "end_hour",
                    "dates",
                    "exposure_hours",
                    "order_count",
                    "total_rate_per_hour",
                ],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "period_id": f"h{config.demand_hour_start:02d}_{config.demand_hour_end:02d}",
                    "start_hour": config.demand_hour_start,
                    "end_hour": config.demand_hour_end,
                    "dates": ";".join(days),
                    "exposure_hours": exposure,
                    "order_count": orders,
                    "total_rate_per_hour": f"{rate:.10f}",
                }
            )


@dataclass(frozen=True)
class _Projection:
    center_lon: float
    center_lat: float

    def project(self, lon: float, lat: float) -> tuple[float, float]:
        km_per_lon = 111.32 * math.cos(math.radians(self.center_lat))
        return (lon - self.center_lon) * km_per_lon, (lat - self.center_lat) * 111.32


def _in_hour_window(hour: int, start: int, end: int) -> bool:
    if end == 24 or start < end:
        return start <= hour < end
    return hour >= start or hour < end


def _inside_square(x: float, y: float, half_size: float) -> bool:
    return -half_size <= x <= half_size and -half_size <= y <= half_size


def _accumulate(
    storage: dict[tuple[int, int], list[float]], x: float, y: float, grid_size: float
) -> tuple[int, int]:
    cell = (math.floor(x / grid_size), math.floor(y / grid_size))
    if cell not in storage:
        storage[cell] = [0.0, 0.0, 0.0]
    storage[cell][0] += 1.0
    storage[cell][1] += x
    storage[cell][2] += y
    return cell


def _weighted_medoids(
    activity: dict[tuple[int, int], list[float]], count: int, seed: int
) -> list[tuple[float, float, float]]:
    if count == 0:
        return []
    points = np.asarray(
        [
            (value[1] / value[0], value[2] / value[0])
            for _, value in sorted(activity.items())
        ],
        dtype=float,
    )
    weights = np.asarray(
        [value[0] for _, value in sorted(activity.items())], dtype=float
    )
    if len(points) < count:
        raise ValueError(
            f"cannot select {count} sites from {len(points)} active grid cells"
        )
    clustering = KMeans(n_clusters=count, random_state=seed, n_init=20)
    labels = clustering.fit_predict(points, sample_weight=weights)
    medoids: list[tuple[float, float, float]] = []
    for cluster in range(count):
        members = np.flatnonzero(labels == cluster)
        center = clustering.cluster_centers_[cluster]
        best = min(
            members,
            key=lambda index: (
                float(np.sum((points[index] - center) ** 2)),
                -weights[index],
                points[index, 0],
                points[index, 1],
            ),
        )
        medoids.append(
            (
                float(points[best, 0]),
                float(points[best, 1]),
                float(weights[members].sum()),
            )
        )
    medoids.sort(key=lambda item: (-item[2], item[0], item[1]))
    return medoids


def _validate_requested_days(requested: set[str], available: set[str]) -> None:
    missing = sorted(requested - available)
    if missing:
        raise ValueError(
            f"requested dates are not present in the source: {', '.join(missing)}; "
            f"available={','.join(sorted(available))}"
        )


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
