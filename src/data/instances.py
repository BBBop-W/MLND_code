"""Assemble the current path-network model from canonical data and parameters.

All durations in the returned model dictionary are hours. Geographic and
calibration inputs are separate from experiment outputs and solver choices.
"""

from __future__ import annotations

import numpy as np
from src.config import load_config
from .instance_io import project_path, read_instance
from .calibration import load_calibration
from .choice import pickup_probability, rider_tariff
from .provenance import attach_provenance

DEFAULT = project_path("data/model_inputs")


def load_instance(hubs, config=None):
    """Build one canonical size using a config dictionary, JSON path, or defaults."""
    cfg = load_config(config)
    costs, op = cfg["costs"], cfg["operations"]
    hours = np.prod(list(cfg["depreciation"].values()))
    data, direct_distances, _ = read_instance(hubs, cfg["data"]["instance_root"])
    cal = load_calibration(hubs, data, cfg["data"]["calibration"])
    slope = op["rider_minutes_per_km"]
    if slope is None:
        slope = float(cal["mean_fit"]["slope_minutes_per_straight_km"])
    dispatch = op["terminal_dispatch_minutes"]
    if dispatch is None:
        dispatch = float(cal["manifest"]["dispatch"]["mean_minutes"])
    distance_oh = np.linalg.norm(
        data["origins"][:, None, :] - data["hubs"][None, :, :], axis=2
    )
    distance_dh = np.linalg.norm(
        data["destinations"][:, None, :] - data["hubs"][None, :, :], axis=2
    )
    data["demand"] *= cfg["data"]["demand_scale"]
    data["ground"] = rider_tariff(direct_distances, cfg)
    data["first_cost"] = costs["feeder_rider_multiplier"] * rider_tariff(
        distance_oh, cfg
    )
    data["last_rider_cost"] = costs["feeder_rider_multiplier"] * rider_tariff(
        distance_dh, cfg
    )
    data["pickup_share"] = pickup_probability(distance_dh, **cfg["choice"])
    data["terminal_cost"] = (1 - data["pickup_share"]) * data["last_rider_cost"]
    data["ground_time"] = slope * distance_oh / 60
    data["last_ground_time"] = slope * distance_dh / 60
    data.update(
        speed=float(op["drone_speed_kmh"]),
        battery_time=op["battery_endurance_minutes"] / 60,
        swap_time=op["swap_minutes"] / 60,
        pickup_handling_time=op["pickup_minutes"] / 60,
        delivery_handling_time=op["unloading_minutes"] / 60,
        drone_time_cost=costs["flight_per_hour"],
        hover_cost=(
            costs["flight_per_hour"]
            if costs["hover_per_hour"] is None
            else costs["hover_per_hour"]
        ),
        fleet_cost=costs["drone_investment"] / hours,
        base_cost=costs["base_investment"] / hours,
        hub_cost=costs["cabinet_investment"] / hours,
        swap_cost=costs["swap_investment"] / hours,
        airline_hh_cost_per_km_hour=costs["directed_route_investment_per_km"] / hours,
        airline_connection_cost_per_km_hour=costs["directed_route_investment_per_km"]
        / hours,
        airline_cost_multiplier=1.0,
        epsilon=op["failure_probability"],
        unit_capacity=op["cabinet_capacity"],
        max_cabinets=int(op["max_cabinets"]),
        rider_slope_minutes_per_straight_km=slope,
        rider_speed=60 * op["road_distance_factor"] / slope,
        road_factor=op["road_distance_factor"],
        rider_time_family=op["rider_time_family"],
        rider_time_cv=1.0 if op["rider_time_family"] == "exponential" else 0.0,
        dispatch_clock="fetch",
        terminal_dispatch_time=dispatch / 60,
        pre_dispatch_mean_hours=float(cal["pre_dispatch"]["mean_hours"]),
        model_version="path_cqn_v1",
    )
    data["battery"] = data["battery_time"] * data["speed"]
    data["flight_cost"] = data["drone_time_cost"] / data["speed"]
    xy = np.vstack((data["bases"], data["hubs"], data["remote"]))
    data["flight_times"] = (
        np.linalg.norm(xy[:, None, :] - xy[None, :, :], axis=2) / data["speed"]
    )
    data["terminal_time"] = data["last_ground_time"] + data["terminal_dispatch_time"]
    table = cal["table"]
    data["direct_rider_hours"] = table.deadline_minutes.to_numpy() / 60
    data["delivery_deadline"] = data["direct_rider_hours"] * op["deadline_multiplier"]
    data["observed_postfetch_hours"] = table.observed_postfetch_minutes.to_numpy() / 60
    data["observed_postfetch_count"] = table.observed_count.to_numpy(dtype=int)
    data["capacity_levels"] = np.arange(1, data["max_cabinets"] + 1)
    shape = (data["n_h"], data["max_cabinets"])
    data["level_capacity"] = np.broadcast_to(
        data["unit_capacity"] * data["capacity_levels"], shape
    ).copy()
    data["level_cost"] = np.broadcast_to(
        data["hub_cost"] * data["capacity_levels"], shape
    ).copy()
    hours = np.prod(list(cfg["depreciation"].values()))
    for field, investment in [
        ("base_site_cost", "base_site_investment"),
        ("hub_site_cost", "locker_site_investment"),
        ("swap_site_cost", "swap_site_investment"),
    ]:
        data[field] = costs[investment] / hours
    data["c0"] = float(data["demand"] @ data["ground"])
    attach_provenance(data, cfg, cal)
    return data


def configure(
    hubs,
    unit=None,
    maximum=None,
    *,
    config=None,
    cost_multiplier=1.0,
    epsilon=None,
    calibration=None,
    pickup_handling_minutes=None,
    delivery_handling_minutes=None,
    hover_cost=None,
    settings=None,
):
    """Convenience constructor with explicit overrides for focused experiments.

    New experiment configurations should use ``load_instance``. The keyword
    form also supports concise capacity and choice sensitivity checks.
    """
    cfg = load_config(config)
    updates = dict(
        cabinet_capacity=unit,
        max_cabinets=maximum,
        failure_probability=epsilon,
        pickup_minutes=pickup_handling_minutes,
        unloading_minutes=delivery_handling_minutes,
    )
    cfg["operations"].update({k: v for k, v in updates.items() if v is not None})
    if calibration is not None:
        cfg["data"]["calibration"] = str(calibration)
    if hover_cost is not None:
        cfg["costs"]["hover_per_hour"] = hover_cost
    mapping = dict(
        hub_cost="cabinet_investment",
        base_cost="base_investment",
        swap_cost="swap_investment",
        fleet_cost="drone_investment",
    )
    for key, value in (settings or {}).items():
        if key == "choice":
            cfg = load_config(cfg, overrides={"choice": value})
        elif key in mapping:
            cfg["costs"][mapping[key]] = value * np.prod(
                list(cfg["depreciation"].values())
            )
        elif key in (
            "airline_hh_cost_per_km_hour",
            "airline_connection_cost_per_km_hour",
        ):
            cfg["costs"]["directed_route_investment_per_km"] = value * np.prod(
                list(cfg["depreciation"].values())
            )
        elif key in ("base_site_cost", "hub_site_cost", "swap_site_cost"):
            investment = {
                "base_site_cost": "base_site_investment",
                "hub_site_cost": "locker_site_investment",
                "swap_site_cost": "swap_site_investment",
            }[key]
            cfg["costs"][investment] = value * np.prod(
                list(cfg["depreciation"].values())
            )
        else:
            raise ValueError(f"Unknown scenario setting: {key}")
    if not np.isfinite(cost_multiplier) or cost_multiplier < 0:
        raise ValueError("cost_multiplier must be finite and nonnegative")
    data = load_instance(hubs, cfg)
    data["airline_cost_multiplier"] = float(cost_multiplier)
    data["provenance"]["airlines"]["multiplier"] = float(cost_multiplier)
    # Keep both link-category keyword overrides when explicitly supplied.
    for key in ("airline_hh_cost_per_km_hour", "airline_connection_cost_per_km_hour"):
        if settings and key in settings:
            if not np.isfinite(settings[key]) or settings[key] < 0:
                raise ValueError(f"Invalid {key}")
            data[key] = float(settings[key])
            data["provenance"]["scenario_settings"][key] = float(settings[key])
            short = (
                "hh_cost_per_km_hour"
                if key.startswith("airline_hh")
                else "connection_cost_per_km_hour"
            )
            data["provenance"]["airlines"][short] = float(settings[key])
    return data
