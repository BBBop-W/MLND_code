"""Fixed conditional choice attributes; all differences are pickup minus home."""

import numpy as np
from scipy.special import expit
from src.config import load_config


def pickup_probability(distance_km, **choice):
    """Evaluate the manuscript's distance, time, price and walking utility."""
    p = load_config(overrides={"choice": choice})["choice"]
    distance = np.asarray(distance_km, dtype=float) * p["walking_distance_factor"]
    if np.any(~np.isfinite(distance)) or np.any(distance < 0):
        raise ValueError("Walking distance must be finite and nonnegative")
    utility = (
        p["beta0"]
        - p["beta_distance"] * np.asarray(p["distance_difference_km"])
        - p["beta_time"] * np.asarray(p["time_difference_minutes"])
        - p["beta_price"] * np.asarray(p["price_difference_cny"])
        - p["beta_walk"] * distance
    )
    return np.where(distance <= p["max_walk_km"], expit(utility), 0.0)


def rider_tariff(distance_km, config):
    costs = config["costs"]
    distance = config["operations"]["road_distance_factor"] * np.asarray(distance_km)
    return costs["rider_initial_charge"] + costs[
        "rider_additional_per_km"
    ] * np.maximum(0.0, distance - costs["rider_included_distance_km"])
