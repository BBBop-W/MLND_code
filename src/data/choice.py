"""Destination-locker pickup probabilities and rider tariffs."""

from __future__ import annotations
import numpy as np
from src.config import load_config


def pickup_probability(distance_km, **choice):
    p = load_config(overrides={"choice": choice})["choice"]
    distance = np.asarray(distance_km, dtype=float)
    if np.any(~np.isfinite(distance)) or np.any(distance < 0):
        raise ValueError("Distance must be finite and nonnegative")
    utility = (
        p["locker_asc"]
        + p["delivery_time_coefficient"]
        * (p["home_eta_minutes"] - p["locker_eta_minutes"])
        + p["price_coefficient"] * p["locker_discount_cny"]
        - p["walk_distance_coefficient"] * distance
    )
    return np.where(
        distance < p["max_walk_km"],
        1.0 / (1.0 + np.exp(-np.clip(utility, -700.0, 700.0))),
        0.0,
    )


def rider_tariff(distance_km, config):
    costs, operations = config["costs"], config["operations"]
    road_distance = operations["road_distance_factor"] * np.asarray(distance_km)
    return costs["rider_initial_charge"] + costs[
        "rider_additional_per_km"
    ] * np.maximum(0.0, road_distance - costs["rider_included_distance_km"])
