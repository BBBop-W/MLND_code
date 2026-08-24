"""Loader for the canonical OD/site instance shared by both raw-data sources."""

from __future__ import annotations

import csv
import json
import math
import os

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
        with open(os.path.join(instance_dir, "metadata.json"), encoding="utf-8") as handle:
            metadata = json.load(handle)
        metadata["instance_dir"] = os.path.abspath(instance_dir)
        metadata["demand_scale"] = demand_scale

        instance = Instance(metadata=metadata, config=model_config or ModelConfig())
        with open(os.path.join(instance_dir, "sites.csv"), newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                node = row["site_id"]
                node_type = row["type"]
                instance.coords[node] = (float(row["x_km"]), float(row["y_km"]))
                instance.activity_rate[node] = float(row["activity_rate"])
                if node_type == "L":
                    instance.L.append(node)
                elif node_type == "H":
                    instance.H.append(node)
                elif node_type == "E":
                    instance.E.append(node)
                else:
                    raise ValueError(f"unknown site type {node_type!r}")

        with open(os.path.join(instance_dir, "od_nodes.csv"), newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                node = row["node_id"]
                instance.N.append(node)
                instance.coords[node] = (float(row["x_km"]), float(row["y_km"]))

        with open(os.path.join(instance_dir, "od_demand.csv"), newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                market = (row["origin_node"], row["destination_node"])
                instance.h_od[market] = float(row["lambda_per_hour"]) * demand_scale
                instance.order_count[market] = int(row["order_count"])

        return instance.build()
