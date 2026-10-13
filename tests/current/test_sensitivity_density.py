"""Fixed-area density changes preserve the frozen demand and nested sites."""

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from src.data.demand import demand_signature
from src.data.sensitivity_density import build_density_variants
from src.experiments.storage import load_inputs, save_inputs
from src.solvers.pricing.inputs import _validated


@pytest.fixture(scope="module")
def cases():
    path = (Path(__file__).resolve().parents[2]
            / "data/instances/algorithm_performance/nanshan/region_2/H020_L04_E08_OD0400")
    base = load_inputs(path)
    original = deepcopy(base)
    variants = build_density_variants(base, path)
    assert original["provenance"] == base["provenance"]
    for key, value in original.items():
        if isinstance(value, np.ndarray):
            np.testing.assert_array_equal(base[key], value)
    return base, variants


def test_density_keeps_every_original_od_and_calibration(cases):
    base, variants = cases
    assert len(variants) == 4
    assert [(x["data"]["n_h"], x["data"]["n_l"], x["data"]["n_k"])
            for x in variants] == [(10, 2, 4), (15, 3, 6), (25, 5, 10), (30, 6, 12)]
    for row in variants:
        data = row["data"]
        _validated(data)
        assert data["n_od"] == base["n_od"] == 400
        assert data["od_pairs"] == base["od_pairs"]
        assert demand_signature(data) == demand_signature(base)
        assert data["c0"] == base["c0"]
        for key in ("demand", "ground", "origins", "destinations", "od_origin",
                    "od_destination", "delivery_deadline", "direct_rider_hours",
                    "observed_postfetch_hours", "observed_postfetch_count"):
            np.testing.assert_array_equal(data[key], base[key])
        assert data["provenance"]["calibration"] == base["provenance"]["calibration"]
        assert data["provenance"]["instance_metadata"]["region_bounds_local_km"] == base["provenance"]["instance_metadata"]["region_bounds_local_km"]


def test_facilities_are_nested_and_stay_inside_original_crop(cases):
    base, variants = cases
    ordered = [x["data"] for x in variants[:2]] + [base] + [x["data"] for x in variants[2:]]
    bounds = base["provenance"]["instance_metadata"]["region_bounds_local_km"]
    for key in ("bases", "hubs", "remote"):
        previous = set()
        for data in ordered:
            points = data[key]
            current = set(map(tuple, points))
            assert previous <= current
            assert len(current) == len(points)
            assert np.all(points >= [bounds["x_min"], bounds["y_min"]])
            assert np.all(points <= [bounds["x_max"], bounds["y_max"]])
            previous = current


def test_retained_hubs_reproduce_original_costs_choice_and_times(cases):
    base, variants = cases
    columns = {tuple(point): index for index, point in enumerate(base["hubs"])}
    for row in variants:
        data = row["data"]
        for index, point in enumerate(data["hubs"]):
            if tuple(point) not in columns:
                continue
            old_index = columns[tuple(point)]
            for field in ("first_cost", "last_rider_cost", "terminal_cost", "pickup_share",
                          "ground_time", "last_ground_time", "terminal_time"):
                np.testing.assert_allclose(data[field][:, index], base[field][:, old_index], rtol=1e-14)
        xy = np.vstack((data["bases"], data["hubs"], data["remote"]))
        np.testing.assert_allclose(data["flight_times"], np.linalg.norm(xy[:, None]-xy[None, :], axis=2)/data["speed"])


def test_density_saved_inputs_replay(cases, tmp_path):
    _, variants = cases
    data = variants[-1]["data"]
    save_inputs(data, tmp_path)
    replay = load_inputs(tmp_path)
    assert replay["provenance"] == data["provenance"]
    assert replay["demand_signature"] == data["demand_signature"]
    np.testing.assert_array_equal(replay["first_cost"], data["first_cost"])
