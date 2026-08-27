"""Loader for the canonical OD/site instance shared by both raw-data sources."""

from __future__ import annotations

import csv
import json
import math
import os
from collections.abc import Mapping
from typing import Any

from src.config import ModelConfig
from src.data_processor.instance import Instance


class DataLoader:
    def load_instance_dir(
        self,
        instance_dir: str,
        model_config: ModelConfig | None = None,
        demand_scale: float = 1.0,
    ) -> Instance:
        if not math.isfinite(demand_scale) or demand_scale <= 0:
            raise ValueError("demand_scale must be positive and finite")
        with open(
            os.path.join(instance_dir, "metadata.json"), encoding="utf-8"
        ) as handle:
            metadata = json.load(handle)
        if not isinstance(metadata, dict):
            raise ValueError("metadata.json must contain a JSON object")
        if metadata.get("coordinate_system") != "local_km":
            raise ValueError(
                "metadata.json must declare coordinate_system='local_km'; "
                "latitude/longitude values cannot be used as model distances"
            )
        metadata["instance_dir"] = os.path.abspath(instance_dir)
        metadata["demand_scale"] = demand_scale

        instance = Instance(metadata=metadata, config=model_config or ModelConfig())
        seen_nodes: set[str] = set()
        with open(
            os.path.join(instance_dir, "sites.csv"), newline="", encoding="utf-8"
        ) as handle:
            reader = csv.DictReader(handle)
            _require_columns(
                reader,
                {"site_id", "type", "x_km", "y_km", "activity_rate"},
                "sites.csv",
            )
            has_locker_cost = "locker_pickup_cost_per_order" in (
                reader.fieldnames or ()
            )
            for row_number, row in enumerate(reader, start=2):
                node = _required_identifier(row, "site_id", "sites.csv", row_number)
                _reject_duplicate(node, seen_nodes, "node", "sites.csv", row_number)
                node_type = _required_identifier(row, "type", "sites.csv", row_number)
                instance.coords[node] = (
                    _finite_float(row, "x_km", "sites.csv", row_number),
                    _finite_float(row, "y_km", "sites.csv", row_number),
                )
                instance.activity_rate[node] = _nonnegative_float(
                    row,
                    "activity_rate",
                    "sites.csv",
                    row_number,
                )
                if node_type == "L":
                    instance.L.append(node)
                elif node_type == "H":
                    instance.H.append(node)
                    if has_locker_cost:
                        locker_cost = _optional_nonnegative_float(
                            row,
                            "locker_pickup_cost_per_order",
                            "sites.csv",
                            row_number,
                        )
                        if locker_cost is not None:
                            instance.locker_pickup_cost_by_locker[node] = locker_cost
                elif node_type == "E":
                    instance.E.append(node)
                else:
                    raise ValueError(
                        f"sites.csv row {row_number}: unknown site type {node_type!r}"
                    )
                if has_locker_cost and node_type != "H":
                    locker_cost_raw = row.get("locker_pickup_cost_per_order")
                    if locker_cost_raw is not None and locker_cost_raw.strip():
                        raise ValueError(
                            "sites.csv row "
                            f"{row_number}: locker_pickup_cost_per_order is valid only for H sites"
                        )

        origin_nodes: set[str] = set()
        destination_nodes: set[str] = set()
        with open(
            os.path.join(instance_dir, "od_nodes.csv"), newline="", encoding="utf-8"
        ) as handle:
            reader = csv.DictReader(handle)
            _require_columns(
                reader,
                {"node_id", "role", "x_km", "y_km"},
                "od_nodes.csv",
            )
            for row_number, row in enumerate(reader, start=2):
                node = _required_identifier(row, "node_id", "od_nodes.csv", row_number)
                _reject_duplicate(node, seen_nodes, "node", "od_nodes.csv", row_number)
                role = _required_identifier(row, "role", "od_nodes.csv", row_number)
                if role == "origin":
                    origin_nodes.add(node)
                elif role == "destination":
                    destination_nodes.add(node)
                else:
                    raise ValueError(
                        f"od_nodes.csv row {row_number}: unknown node role {role!r}"
                    )
                instance.N.append(node)
                instance.coords[node] = (
                    _finite_float(row, "x_km", "od_nodes.csv", row_number),
                    _finite_float(row, "y_km", "od_nodes.csv", row_number),
                )

        seen_markets: set[tuple[str, str]] = set()
        seen_od_ids: set[str] = set()
        with open(
            os.path.join(instance_dir, "od_demand.csv"), newline="", encoding="utf-8"
        ) as handle:
            reader = csv.DictReader(handle)
            _require_columns(
                reader,
                {
                    "od_id",
                    "origin_node",
                    "destination_node",
                    "order_count",
                    "lambda_per_hour",
                },
                "od_demand.csv",
            )
            for row_number, row in enumerate(reader, start=2):
                od_id = _required_identifier(row, "od_id", "od_demand.csv", row_number)
                _reject_duplicate(
                    od_id, seen_od_ids, "OD identifier", "od_demand.csv", row_number
                )
                origin = _required_identifier(
                    row,
                    "origin_node",
                    "od_demand.csv",
                    row_number,
                )
                destination = _required_identifier(
                    row,
                    "destination_node",
                    "od_demand.csv",
                    row_number,
                )
                if origin not in origin_nodes:
                    raise ValueError(
                        f"od_demand.csv row {row_number}: unknown origin node {origin!r}"
                    )
                if destination not in destination_nodes:
                    raise ValueError(
                        f"od_demand.csv row {row_number}: unknown destination node {destination!r}"
                    )
                market = (origin, destination)
                if market in seen_markets:
                    raise ValueError(
                        f"od_demand.csv row {row_number}: duplicate OD market {market!r}"
                    )
                seen_markets.add(market)
                raw_rate = _nonnegative_float(
                    row,
                    "lambda_per_hour",
                    "od_demand.csv",
                    row_number,
                )
                scaled_rate = raw_rate * demand_scale
                if not math.isfinite(scaled_rate):
                    raise ValueError(
                        f"od_demand.csv row {row_number}: scaled demand must be finite"
                    )
                order_count = _nonnegative_int(
                    row,
                    "order_count",
                    "od_demand.csv",
                    row_number,
                )
                if scaled_rate == 0.0:
                    continue
                instance.h_od[market] = scaled_rate
                instance.order_count[market] = order_count

        return instance.build()


def _require_columns(reader: csv.DictReader, required: set[str], filename: str) -> None:
    fieldnames = set(reader.fieldnames or ())
    missing = sorted(required - fieldnames)
    if missing:
        raise ValueError(f"{filename} is missing required columns: {missing}")


def _required_identifier(
    row: Mapping[str, Any],
    column: str,
    filename: str,
    row_number: int,
) -> str:
    raw_value = row.get(column)
    value = "" if raw_value is None else str(raw_value).strip()
    if not value:
        raise ValueError(f"{filename} row {row_number}: {column} must be nonempty")
    return value


def _finite_float(
    row: Mapping[str, Any],
    column: str,
    filename: str,
    row_number: int,
) -> float:
    raw_value = row.get(column)
    try:
        value = float(raw_value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{filename} row {row_number}: {column} must be a number"
        ) from exc
    if not math.isfinite(value):
        raise ValueError(f"{filename} row {row_number}: {column} must be finite")
    return value


def _nonnegative_float(
    row: Mapping[str, Any],
    column: str,
    filename: str,
    row_number: int,
) -> float:
    value = _finite_float(row, column, filename, row_number)
    if value < 0.0:
        raise ValueError(f"{filename} row {row_number}: {column} must be nonnegative")
    return value


def _optional_nonnegative_float(
    row: Mapping[str, Any],
    column: str,
    filename: str,
    row_number: int,
) -> float | None:
    raw_value = row.get(column)
    if raw_value is None or not str(raw_value).strip():
        return None
    return _nonnegative_float(row, column, filename, row_number)


def _nonnegative_int(
    row: Mapping[str, Any],
    column: str,
    filename: str,
    row_number: int,
) -> int:
    raw_value = row.get(column)
    try:
        value = int(raw_value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{filename} row {row_number}: {column} must be an integer"
        ) from exc
    if value < 0:
        raise ValueError(f"{filename} row {row_number}: {column} must be nonnegative")
    return value


def _reject_duplicate(
    value: str,
    seen: set[str],
    label: str,
    filename: str,
    row_number: int,
) -> None:
    if value in seen:
        raise ValueError(f"{filename} row {row_number}: duplicate {label} {value!r}")
    seen.add(value)
