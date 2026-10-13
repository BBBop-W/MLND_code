"""Sensitivity replay keeps one-factor changes internally consistent."""

from copy import deepcopy
from pathlib import Path
import numpy as np
import pytest

from src.data.demand import demand_signature
from src.data.choice import pickup_probability
from src.experiments.sensitivity_inputs import build_variants, scale_instance
from src.experiments.storage import load_inputs, save_inputs
from src.solvers.pricing.inputs import _validated


@pytest.fixture(scope="module")
def base():
    directory = (Path(__file__).resolve().parents[2]
                 / "data/instances/algorithm_performance/nanshan/region_2"
                 / "H020_L04_E08_OD0400")
    return load_inputs(directory)


def test_thirteen_variants_preserve_geometry_and_choice(base):
    original = deepcopy(base)
    variants = build_variants(base)
    assert len(variants) == len({v["id"] for v in variants}) == 13
    assert variants[0]["id"] == "baseline"
    for record in variants:
        data = record["data"]
        _validated(data)
        assert data["n_od"] == 400
        assert data["od_pairs"] == base["od_pairs"]
        for key in ("bases", "hubs", "remote", "origins", "destinations",
                    "od_origin", "od_destination", "pickup_share",
                    "delivery_deadline", "direct_rider_hours", "ground_time",
                    "terminal_time", "level_capacity", "level_cost"):
            np.testing.assert_array_equal(data[key], base[key])
        for key in ("fleet_cost", "base_cost", "swap_cost", "hub_cost",
                    "drone_time_cost", "hover_cost", "airline_hh_cost_per_km_hour"):
            assert data[key] == base[key]
        assert data["c0"] == pytest.approx(data["demand"] @ data["ground"])
        assert data["demand_signature"] == demand_signature(data)
        assert data["provenance"]["demand_selection"]["signature"] == demand_signature(data)
    assert base["provenance"] == original["provenance"]
    for key, value in original.items():
        if isinstance(value, np.ndarray):
            np.testing.assert_array_equal(base[key], value)


@pytest.mark.parametrize("factor", ["demand", "rider_cost", "speed", "battery"])
def test_one_factor_updates_only_its_dependent_arrays(base, factor):
    value = 0.75
    data = scale_instance(base, factor, value)
    if factor in ("demand", "rider_cost"):
        assert data["c0"] == pytest.approx(base["c0"] * value)
        assert data["demand_signature"] != base["demand_signature"]
    else:
        assert data["c0"] == base["c0"]
        assert data["demand_signature"] == base["demand_signature"]
    np.testing.assert_allclose(data["demand"], base["demand"] * (value if factor == "demand" else 1))
    for key in ("ground", "first_cost", "last_rider_cost", "terminal_cost"):
        np.testing.assert_allclose(data[key], base[key] * (value if factor == "rider_cost" else 1))
    np.testing.assert_allclose(data["terminal_cost"], (1-data["pickup_share"]) * data["last_rider_cost"])
    np.testing.assert_allclose(data["flight_times"], base["flight_times"] / (value if factor == "speed" else 1))
    assert data["battery_time"] == pytest.approx(base["battery_time"] * (value if factor == "battery" else 1))
    assert data["battery"] == pytest.approx(data["battery_time"] * data["speed"])
    assert data["flight_cost"] == pytest.approx(data["drone_time_cost"] / data["speed"])


def test_saved_variant_can_be_replayed(base, tmp_path):
    data = scale_instance(base, "rider_cost", 1.25)
    save_inputs(data, tmp_path)
    replay = load_inputs(tmp_path)
    assert demand_signature(data) == demand_signature(replay)
    assert replay["provenance"] == data["provenance"]
    np.testing.assert_array_equal(replay["first_cost"], data["first_cost"])


@pytest.mark.parametrize("factor", ["speed", "battery", "fleet_cost"])
@pytest.mark.parametrize("scale", [.5, 1.5])
def test_round_two_extremes_keep_other_factors_fixed(base, factor, scale):
    data = scale_instance(base, factor, scale)
    _validated(data)
    assert data["c0"] == base["c0"]
    assert data["demand_signature"] == base["demand_signature"]
    for key in ("demand", "origins", "destinations", "hubs", "bases", "remote",
                "ground", "first_cost", "terminal_cost", "pickup_share", "level_cost"):
        np.testing.assert_array_equal(data[key], base[key])
    assert data["fleet_cost"] == pytest.approx(
        base["fleet_cost"] * (scale if factor == "fleet_cost" else 1))
    assert data["speed"] == pytest.approx(base["speed"] * (scale if factor == "speed" else 1))
    assert data["battery_time"] == pytest.approx(
        base["battery_time"] * (scale if factor == "battery" else 1))
    assert data["provenance"]["resolved_config"]["costs"]["drone_investment"] == pytest.approx(
        base["provenance"]["resolved_config"]["costs"]["drone_investment"]
        * (scale if factor == "fleet_cost" else 1))
    assert data["provenance"]["scenario_settings"]["fleet_cost"] == data["fleet_cost"]


@pytest.mark.parametrize("factor,scales", [
    ("demand", (.2, .3, .4)), ("speed", (.6, .7, .8, .9)),
])
def test_round_three_demand_and_speed_preserve_market(base, factor, scales):
    for scale in scales:
        data = scale_instance(base, factor, scale)
        _validated(data)
        assert data["od_pairs"] == base["od_pairs"]
        for key in ("hubs", "bases", "remote", "origins", "destinations",
                    "pickup_share", "terminal_cost", "level_capacity", "level_cost"):
            np.testing.assert_array_equal(data[key], base[key])
        np.testing.assert_allclose(data["demand"], base["demand"] * (scale if factor == "demand" else 1))
        np.testing.assert_allclose(data["flight_times"], base["flight_times"] / (scale if factor == "speed" else 1))
        assert data["battery_time"] == base["battery_time"]


@pytest.mark.parametrize("maximum", [3, 4, 6, 7])
def test_cabinet_maximum_updates_all_capacity_levels(base, maximum):
    data = scale_instance(base, "max_cabinets", maximum / base["max_cabinets"])
    _validated(data)
    assert data["max_cabinets"] == maximum
    np.testing.assert_array_equal(data["capacity_levels"], np.arange(1, maximum + 1))
    assert data["level_capacity"].shape == data["level_cost"].shape == (20, maximum)
    np.testing.assert_allclose(data["level_capacity"],
                               np.tile(base["unit_capacity"] * np.arange(1, maximum + 1), (20, 1)))
    np.testing.assert_allclose(data["level_cost"],
                               np.tile(base["hub_cost"] * np.arange(1, maximum + 1), (20, 1)))
    assert data["provenance"]["resolved_config"]["operations"]["max_cabinets"] == maximum
    assert data["provenance"]["cabinet_capacity"]["max_units_per_site"] == maximum
    assert data["provenance"]["sensitivity"]["id"] == f"max_cabinets_{maximum}"
    assert data["provenance"]["sensitivity"]["value"] == maximum
    for key in ("demand", "hubs", "bases", "remote", "origins", "destinations",
                "pickup_share", "terminal_cost", "flight_times"):
        np.testing.assert_array_equal(data[key], base[key])
    assert data["demand_signature"] == base["demand_signature"]


@pytest.mark.parametrize("scale", [.5, .75, 1.25, 1.5])
def test_walk_sensitivity_recomputes_choice_and_last_leg_cost(base, scale):
    data = scale_instance(base, "walk_sensitivity", scale)
    _validated(data)
    choice = data["provenance"]["resolved_config"]["choice"]
    original_choice = base["provenance"]["resolved_config"]["choice"]
    assert choice["walk_distance_coefficient"] == pytest.approx(
        original_choice["walk_distance_coefficient"] * scale)
    assert choice["max_walk_km"] == original_choice["max_walk_km"] == .8
    assert data["provenance"]["scenario_settings"]["choice"] == choice
    assert data["provenance"]["sensitivity"]["value"] == choice["walk_distance_coefficient"]
    distance = np.linalg.norm(base["destinations"][:, None] - base["hubs"][None, :], axis=2)
    np.testing.assert_allclose(data["pickup_share"], pickup_probability(distance, **choice))
    np.testing.assert_allclose(data["terminal_cost"], (1 - data["pickup_share"]) * base["last_rider_cost"])
    np.testing.assert_array_equal(data["pickup_share"][distance >= .8], 0)
    within = (distance > 0) & (distance < .8)
    assert np.any(within)
    if scale < 1:
        assert np.all(data["pickup_share"][within] > base["pickup_share"][within])
        assert np.all(data["terminal_cost"][within] < base["terminal_cost"][within])
    else:
        assert np.all(data["pickup_share"][within] < base["pickup_share"][within])
        assert np.all(data["terminal_cost"][within] > base["terminal_cost"][within])
    for key in ("demand", "hubs", "bases", "remote", "origins", "destinations",
                "last_rider_cost", "first_cost", "ground", "terminal_time",
                "flight_times", "level_capacity", "level_cost"):
        np.testing.assert_array_equal(data[key], base[key])
    assert data["demand_signature"] == base["demand_signature"]
    assert data["c0"] == base["c0"]


@pytest.mark.parametrize("factor,value", [("unknown", 1), ("baseline", 2),
                                           ("speed", 0), ("demand", float("nan")),
                                           ("max_cabinets", .7), ("max_cabinets", .01)])
def test_invalid_variants_are_rejected(base, factor, value):
    with pytest.raises(ValueError):
        scale_instance(base, factor, value)
