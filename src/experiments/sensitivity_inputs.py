"""One-factor sensitivity inputs derived from an immutable saved instance."""

from __future__ import annotations

from copy import deepcopy
import numpy as np

from src.data.demand import demand_signature
from src.data.choice import pickup_probability


FACTORS = {
    "demand": (0.5, 0.75, 1.25, 1.5),
    "rider_cost": (0.5, 0.75, 1.25, 1.5),
    "speed": (0.75, 1.25),
    "battery": (0.75, 1.25),
}


def scale_instance(base, factor="baseline", multiplier=1.0):
    """Copy one frozen input and change only the selected economic/physical factor.

    Rider costs are platform tariffs for direct, first-mile and home last-mile
    service. Customer pickup discount and therefore mode probabilities stay fixed.
    Battery denotes usable flight/handling/hover endurance; changing speed keeps
    this endurance fixed, while changing battery keeps speed fixed.
    """
    multiplier = float(multiplier)
    if not np.isfinite(multiplier) or multiplier <= 0:
        raise ValueError("Sensitivity multiplier must be finite and positive")
    if factor not in ("baseline", "fleet_cost", "max_cabinets", "walk_sensitivity", *FACTORS):
        raise ValueError(f"Unknown sensitivity factor: {factor}")
    if factor == "baseline" and multiplier != 1:
        raise ValueError("The baseline multiplier must equal one")

    data = deepcopy(base)
    provenance = data.setdefault("provenance", {})
    config = provenance.setdefault("resolved_config", {})
    settings = provenance.setdefault("scenario_settings", {})
    changed = []
    value = None
    if factor == "demand":
        data["demand"] *= multiplier
        demand_config = config.setdefault("data", {})
        demand_config["demand_scale"] = (
            float(demand_config.get("demand_scale", 1.0)) * multiplier
        )
        provenance["demand_scale"] = demand_config["demand_scale"]
        provenance["demand_rescaled"] = demand_config["demand_scale"] != 1
        changed = ["demand", "c0"]
    elif factor == "rider_cost":
        changed = ["ground", "first_cost", "last_rider_cost", "terminal_cost"]
        for key in changed:
            data[key] *= multiplier
        costs = config.setdefault("costs", {})
        for key in ("rider_initial_charge", "rider_additional_per_km"):
            if key in costs:
                costs[key] *= multiplier
        for key in ("rider_initial_charge_cny", "rider_incremental_cost_per_km"):
            if key in settings:
                settings[key] *= multiplier
        changed.append("c0")
    elif factor == "speed":
        data["speed"] *= multiplier
        data["flight_times"] /= multiplier
        data["flight_cost"] = data["drone_time_cost"] / data["speed"]
        data["battery"] = data["battery_time"] * data["speed"]
        config.setdefault("operations", {})["drone_speed_kmh"] = data["speed"]
        changed = ["speed", "flight_times", "flight_cost", "battery"]
    elif factor == "battery":
        data["battery_time"] *= multiplier
        data["battery"] = data["battery_time"] * data["speed"]
        config.setdefault("operations", {})["battery_endurance_minutes"] = (
            data["battery_time"] * 60
        )
        changed = ["battery_time", "battery"]
    elif factor == "fleet_cost":
        data["fleet_cost"] *= multiplier
        config.setdefault("costs", {})["drone_investment"] *= multiplier
        settings["fleet_cost"] = data["fleet_cost"]
        changed = ["fleet_cost"]
    elif factor == "max_cabinets":
        scaled_maximum = float(data["max_cabinets"]) * multiplier
        maximum = round(scaled_maximum)
        if maximum < 1 or not np.isclose(scaled_maximum, maximum, rtol=0, atol=1e-10):
            raise ValueError("Cabinet limit must scale to a positive integer")
        data["max_cabinets"] = maximum
        data["capacity_levels"] = np.arange(1, maximum + 1)
        shape = (data["n_h"], maximum)
        data["level_capacity"] = np.broadcast_to(
            data["unit_capacity"] * data["capacity_levels"], shape
        ).copy()
        data["level_cost"] = np.broadcast_to(
            data["hub_cost"] * data["capacity_levels"], shape
        ).copy()
        config.setdefault("operations", {})["max_cabinets"] = maximum
        provenance.setdefault("cabinet_capacity", {})["max_units_per_site"] = maximum
        changed = ["max_cabinets", "capacity_levels", "level_capacity", "level_cost"]
        value = maximum
    elif factor == "walk_sensitivity":
        choice = config["choice"]
        choice["walk_distance_coefficient"] *= multiplier
        settings["choice"] = deepcopy(choice)
        distance = np.linalg.norm(
            data["destinations"][:, None] - data["hubs"][None, :], axis=2
        )
        data["pickup_share"] = pickup_probability(distance, **choice)
        data["terminal_cost"] = (1 - data["pickup_share"]) * data["last_rider_cost"]
        changed = ["pickup_share", "terminal_cost"]
        value = choice["walk_distance_coefficient"]

    case_id = (
        "baseline" if factor == "baseline"
        else f"max_cabinets_{data['max_cabinets']}" if factor == "max_cabinets"
        else f"{factor}_{round(100 * multiplier):03d}"
    )
    data["name"] = f"{base['name']}_{case_id}"
    data["c0"] = float(data["demand"] @ data["ground"])
    data["demand_signature"] = demand_signature(data)
    provenance.setdefault("demand_selection", {}).update(
        hourly_demand=float(data["demand"].sum()),
        all_ground_cost=data["c0"],
        demand_scale=config.get("data", {}).get("demand_scale", 1.0),
        signature=data["demand_signature"],
    )
    provenance["sensitivity"] = dict(
        id=case_id,
        factor=factor,
        multiplier=multiplier,
        parent_name=base["name"],
        parent_demand_signature=demand_signature(base),
        changed_model_fields=changed,
        od_pairs_and_candidate_coordinates_unchanged=True,
        source_observations_unchanged=True,
        authoritative_replay="Saved input_arrays.npz plus input_metadata.json",
        rider_cost_scope="Direct, first-mile and home-delivery rider tariffs",
        utility=(
            "Walk-distance coefficient scaled; customer discount, ETA, distances and walk cutoff fixed"
            if factor == "walk_sensitivity" else
            "Fixed customer discount, ETA and walk-distance inputs; unchanged"
        ),
        speed_scope="Flight durations change inversely; battery endurance stays fixed",
        battery_scope="Flight, handling and pickup-hover endurance; speed stays fixed",
        fleet_cost_scope="Drone purchase cost amortized over the unchanged five-year operating horizon",
        max_cabinets_scope="Maximum cabinet units per candidate locker site; per-unit capacity and cost fixed",
        unscaled=("Facility and route costs, hourly flight/hover costs, deadlines and rider times"
                  if factor == "fleet_cost" else
                  "Fixed costs, route costs, hourly flight/hover costs, deadlines and rider times"),
    )
    if value is not None:
        provenance["sensitivity"]["value"] = value
    return data


def build_variants(base, *, factors=None, include_baseline=True):
    """Return one-factor variants, optionally including one shared baseline."""
    values = ([("baseline", 1.0)] if include_baseline else []) + [
        (factor, multiplier)
        for factor, multipliers in (FACTORS if factors is None else factors).items()
        for multiplier in multipliers
    ]
    result = []
    for factor, multiplier in values:
        data = scale_instance(base, factor, multiplier)
        result.append(dict(
            id=data["provenance"]["sensitivity"]["id"],
            factor=factor, multiplier=multiplier, data=data,
        ))
    return result
