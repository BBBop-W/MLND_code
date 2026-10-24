"""Validated configuration for the single consecutive-delivery model."""

from copy import deepcopy
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/default.json"


def _merge(base, changes, prefix=""):
    if not isinstance(changes, dict):
        raise ValueError(f"{prefix or 'Configuration'} must be an object")
    result = deepcopy(base)
    for key, value in changes.items():
        if key not in base:
            raise ValueError(f"Unknown configuration key: {prefix}{key}")
        result[key] = (
            _merge(base[key], value, f"{prefix}{key}.")
            if isinstance(base[key], dict)
            else deepcopy(value)
        )
    return result


def validate_config(config):
    if config["schema_version"] != 2 or config["model"] != "consecutive":
        raise ValueError("Only consecutive configuration schema 2 is supported")
    if isinstance(config["max_orders"], bool) or config["max_orders"] not in (1, 2):
        raise ValueError("max_orders must be 1 or 2")
    if config["flow_policy"] not in ("prefix", "aggregate"):
        raise ValueError("flow_policy must be prefix or aggregate")
    if not isinstance(config["compatibility"]["require_feasible_path"], bool):
        raise ValueError("compatibility.require_feasible_path must be boolean")
    if config["operations"]["route_mode"] not in ("direct", "simple"):
        raise ValueError("route_mode must be direct or simple")
    if config["compatibility"]["mode"] not in ("all", "none", "distance", "matrix"):
        raise ValueError("compatibility.mode must be all, none, distance or matrix")
    nullable = {
        "operations.rider_minutes_per_km",
        "operations.deadline_minutes",
        "terminal.courier_receipt_minutes",
        "solver.time_limit",
        "solver.mip_gap",
        "solver.max_iterations",
        "solver.column_limit",
    }
    signed = {
        "choice.beta0",
        "choice.distance_difference_km",
        "choice.time_difference_minutes",
        "choice.price_difference_cny",
    }
    for section in (
        "operations",
        "terminal",
        "choice",
        "costs",
        "depreciation",
        "solver",
    ):
        for key, value in config[section].items():
            if section == "operations" and key == "route_mode":
                continue
            if value is None:
                if f"{section}.{key}" not in nullable:
                    raise ValueError(f"{section}.{key} cannot be null")
                continue
            array = np.asarray(value)
            if array.dtype.kind not in "iuf" or not np.isfinite(array).all():
                raise ValueError(f"{section}.{key} must be finite numeric values")
            if f"{section}.{key}" not in signed and np.any(array < 0):
                raise ValueError(f"{section}.{key} must be nonnegative")
    for key in ("drone_speed_kmh", "operating_limit_minutes"):
        value = config["operations"][key]
        if np.ndim(value) != 0 or value <= 0:
            raise ValueError(f"operations.{key} must be positive")
    for key in ("window_minutes", "deadline_multiplier", "road_distance_factor"):
        if np.ndim(config["operations"][key]) != 0:
            raise ValueError(f"operations.{key} must be scalar")
    slope = config["operations"]["rider_minutes_per_km"]
    if slope is not None and (np.ndim(slope) != 0 or slope <= 0):
        raise ValueError("rider_minutes_per_km must be a positive scalar or null")
    slots = np.asarray(config["operations"]["slot_counts"])
    if slots.size == 0 or np.any(slots <= 0) or np.any(slots != np.floor(slots)):
        raise ValueError("slot_counts must contain positive integer capacities")
    epsilon = np.asarray(config["operations"]["slot_epsilon"])
    if np.any((epsilon <= 0) | (epsilon >= 1)):
        raise ValueError("slot_epsilon must be strictly between zero and one")
    scale = config["data"]["demand_scale"]
    if (
        isinstance(scale, bool)
        or not isinstance(scale, (float, int))
        or not np.isfinite(scale)
        or scale <= 0
    ):
        raise ValueError("data.demand_scale must be positive")
    for key in ("input_path", "instance_root", "calibration"):
        if not isinstance(config["data"][key], str) or not config["data"][key]:
            raise ValueError(f"data.{key} must be a path string")
    hubs = config["data"]["demand_hubs"]
    if hubs is not None and (
        isinstance(hubs, bool) or not isinstance(hubs, int) or hubs < 2
    ):
        raise ValueError("data.demand_hubs must be an integer >= 2 or null")
    if any(v <= 0 for v in config["depreciation"].values()):
        raise ValueError("Depreciation factors must be positive")
    if (
        config["compatibility"]["mode"] == "matrix"
        and config["compatibility"]["matrix"] is None
    ):
        raise ValueError("compatibility.matrix is required for matrix mode")
    threshold = config["compatibility"]["max_distance_km"]
    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, (int, float))
        or not np.isfinite(threshold)
        or threshold < 0
    ):
        raise ValueError(
            "compatibility.max_distance_km must be a finite nonnegative scalar"
        )
    matrix = config["compatibility"]["matrix"]
    if matrix is not None:
        value = np.asarray(matrix)
        if (
            value.dtype.kind not in "iuf"
            or value.ndim != 2
            or value.shape[0] != value.shape[1]
            or not np.isfinite(value).all()
            or np.any((value < 0) | (value > 1))
        ):
            raise ValueError(
                "compatibility.matrix must be a finite square matrix in [0, 1]"
            )
    for key in ("threads", "seed", "max_iterations", "column_limit"):
        value = config["solver"][key]
        if value is not None and (
            np.ndim(value) != 0 or int(value) != value or (key != "seed" and value < 1)
        ):
            raise ValueError(f"solver.{key} must be an integer in its valid range")
    if (
        config["solver"]["time_limit"] is not None
        and config["solver"]["time_limit"] <= 0
    ):
        raise ValueError("solver.time_limit must be positive or null")
    return config


def load_config(path=None, overrides=None):
    """Load a partial dict/JSON file over the documented explicit scenario."""
    result = json.loads(DEFAULT_CONFIG.read_text())
    if path is not None:
        if isinstance(path, dict):
            changes = path
        else:
            target = Path(path)
            changes = json.loads(
                (target if target.is_absolute() else ROOT / target).read_text()
            )
        result = _merge(result, changes)
    if overrides is not None:
        result = _merge(result, overrides)
    return validate_config(result)
