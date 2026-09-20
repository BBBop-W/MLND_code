"""Load the single production parameter configuration.

Paths are resolved relative to the repository. Partial JSON configurations and
nested overrides inherit the checked defaults in ``configs/default.json``.
"""

from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/default.json"


def _merge(base, changes, prefix=""):
    result = deepcopy(base)
    if not isinstance(changes, dict):
        raise ValueError(f"{prefix or 'Configuration'} must be an object")
    for key, value in changes.items():
        if key not in base:
            raise ValueError(f"Unknown configuration key: {prefix}{key}")
        if isinstance(base[key], dict):
            result[key] = _merge(base[key], value, f"{prefix}{key}.")
        else:
            result[key] = deepcopy(value)
    return result


def _number(value, key, *, positive=False, nullable=False, integer=False):
    if value is None and nullable:
        return
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ValueError(f"{key} must be finite")
    if value < 0 or (positive and value == 0):
        raise ValueError(f"{key} must be {'positive' if positive else 'nonnegative'}")
    if integer and int(value) != value:
        raise ValueError(f"{key} must be an integer")


def validate_config(config):
    if config["schema_version"] != 1:
        raise ValueError("Unsupported configuration schema")
    if not isinstance(config["name"], str) or not config["name"]:
        raise ValueError("Configuration name must be nonempty")
    for name in ("instance_root", "calibration"):
        if not isinstance(config["data"][name], str) or not config["data"][name]:
            raise ValueError(f"data.{name} must be a path")
    _number(config["data"]["demand_scale"], "data.demand_scale", positive=True)
    demand_hubs = config["data"]["demand_hubs"]
    _number(demand_hubs, "data.demand_hubs", nullable=True, integer=True)
    if demand_hubs is not None and demand_hubs < 2:
        raise ValueError(
            "data.demand_hubs must be at least two, or null for per-instance demand"
        )
    for key, value in config["depreciation"].items():
        _number(value, f"depreciation.{key}", positive=True)
    for key, value in config["costs"].items():
        _number(value, f"costs.{key}", nullable=key == "hover_per_hour")
    for key, value in config["operations"].items():
        if key == "rider_time_family":
            if value not in ("exponential", "deterministic"):
                raise ValueError(
                    "operations.rider_time_family must be exponential or deterministic"
                )
            continue
        _number(
            value,
            f"operations.{key}",
            positive=key
            in (
                "cabinet_capacity",
                "max_cabinets",
                "drone_speed_kmh",
                "battery_endurance_minutes",
                "road_distance_factor",
                "rider_minutes_per_km",
                "deadline_multiplier",
            ),
            nullable=key in ("rider_minutes_per_km", "terminal_dispatch_minutes"),
            integer=key == "max_cabinets",
        )
    if not 0 < config["operations"]["failure_probability"] < 1:
        raise ValueError(
            "operations.failure_probability must be strictly between 0 and 1"
        )
    for key, value in config["choice"].items():
        if key == "locker_asc":
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError("choice.locker_asc must be finite")
        else:
            _number(value, f"choice.{key}")
    for key, value in config["solver"].items():
        _number(
            value,
            f"solver.{key}",
            nullable=key not in ("threads", "seed"),
            positive=key
            in (
                "time_limit",
                "threads",
                "max_iterations",
                "column_limit",
                "graph_limit",
            ),
            integer=key
            in ("threads", "max_iterations", "seed", "column_limit", "graph_limit"),
        )
    if config["solver"]["seed"] > 2**31 - 1:
        raise ValueError("solver.seed must be at most 2147483647")
    return config


def load_config(path=None, overrides=None):
    """Return a fresh complete configuration from optional JSON and overrides.

    ``path`` may also be a partial configuration dictionary. Relative paths
    within the configuration are interpreted from the project root.
    """
    config = json.loads(DEFAULT_CONFIG.read_text())
    if path is not None:
        if isinstance(path, dict):
            changes = path
        else:
            target = Path(path)
            if not target.is_absolute():
                target = ROOT / target
            changes = json.loads(target.read_text())
        config = _merge(config, changes)
    if overrides is not None:
        config = _merge(config, overrides)
    return validate_config(config)
