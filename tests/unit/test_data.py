"""Input provenance, independent handoff clocks and preservation regressions."""

import json
import numpy as np
import pandas as pd
import pytest
from scipy.special import expit

from src.config import ROOT, load_config
from src.data.choice import pickup_probability
from src.data.instances import load_saved, to_planning_data, pack_planning_data
from src.data.demand import demand_signature
from src.data.prepare import prepare_orders
from src.experiments.storage import save_inputs

SAVED = (
    ROOT / "data/instances/algorithm_performance/nanshan/region_1/H010_L02_E04_OD0200"
)


def test_saved_market_preserves_orders_and_ground_costs():
    with np.load(SAVED / "input_arrays.npz") as old:
        data = load_saved(SAVED)
        np.testing.assert_array_equal(data["demand"], old["demand"])
        np.testing.assert_array_equal(data["ground_cost"], old["ground"])
        np.testing.assert_array_equal(data["origins"], old["origins"])
        np.testing.assert_array_equal(data["first_cost"], old["first_cost"])
        np.testing.assert_array_equal(data["deadline"], old["delivery_deadline"])
    planning = to_planning_data(data)
    assert (planning.n_l, planning.n_h, planning.n_od) == (2, 10, 200)
    assert planning.flight_times.shape == (12, 12)
    assert "remote" not in data and data["model_version"] == "consecutive"
    assert data["metadata"]["source"]["model"] == "path_cqn_v1"
    assert len(data["metadata"]["source"]["hashes"]) == 2


def test_removal_and_receipt_are_separate():
    data = load_saved(
        SAVED,
        {
            "terminal": {
                "self_pickup_removal_minutes": 3,
                "self_pickup_receipt_minutes": 7,
                "courier_removal_minutes": 2,
                "courier_receipt_minutes": 11,
            }
        },
    )
    sigma = data["pickup_share"]
    np.testing.assert_allclose(data["removal_mean"], (3 * sigma + 2 * (1 - sigma)) / 60)
    np.testing.assert_allclose(
        data["receipt_mean"], (7 * sigma + 11 * (1 - sigma)) / 60
    )
    assert np.all(data["receipt_mean"] > data["removal_mean"])
    with pytest.raises(ValueError, match="cannot precede"):
        load_saved(
            SAVED,
            {"terminal": {"courier_removal_minutes": 5, "courier_receipt_minutes": 1}},
        )


def test_full_choice_differences_and_range_boundary():
    p = pickup_probability(
        np.array([0.2, 0.8, 0.81]),
        beta0=0.5,
        beta_distance=0.3,
        beta_time=0.1,
        beta_price=0.4,
        beta_walk=2,
        distance_difference_km=0.7,
        time_difference_minutes=-2,
        price_difference_cny=-1,
        max_walk_km=0.8,
    )
    np.testing.assert_allclose(
        p[:2], expit(0.5 - 0.3 * 0.7 + 0.1 * 2 + 0.4 - 2 * np.array([0.2, 0.8]))
    )
    assert p[-1] == 0


def test_new_snapshot_exact_replay(tmp_path):
    data = load_saved(SAVED, {"max_orders": 1, "compatibility": {"mode": "none"}})
    signature = demand_signature(data)
    save_inputs(data, tmp_path)
    replay = load_saved(tmp_path)
    assert demand_signature(replay) == signature
    assert to_planning_data(replay).effective_delta == 0
    np.testing.assert_array_equal(replay["kappa"], data["kappa"])
    assert replay["metadata"] == data["metadata"]


def test_historical_name_replay_changes_no_physical_inputs(tmp_path):
    data = load_saved(SAVED)
    data["model_version"] = "consecutive_v2"
    data["metadata"]["resolved_config"]["model"] = "consecutive_v2"
    save_inputs(data, tmp_path)
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    replay = load_saved(tmp_path, {"solver": {"time_limit": 2}})
    assert replay["model_version"] == "consecutive"
    assert replay["metadata"]["resolved_config"]["model"] == "consecutive"
    for name, value in data.items():
        if isinstance(value, np.ndarray):
            np.testing.assert_array_equal(replay[name], value)
        elif name not in ("model_version", "metadata"):
            assert replay[name] == value
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == before


def test_reject_old_config_and_noninteger_capacity():
    with pytest.raises(ValueError, match="Unknown"):
        load_config({"continuation": {"enabled": True}})
    with pytest.raises(ValueError, match="integer"):
        load_config({"operations": {"slot_counts": [30.4]}})


def test_prepare_independent_counts_and_order_conservation(tmp_path):
    rows = []
    for day in ("1", "2"):
        for i in range(8):
            rows.append(
                dict(
                    order_seq_id=f"{day}-{i}",
                    dt_seq_id=day,
                    customer_pay_time="12:00",
                    fetch_time="12:03",
                    arrived_time=f"12:{10 + i:02d}",
                    pickup_longitude=113.9 + i * 0.001,
                    pickup_latitude=22.5 + i * 0.0003,
                    dropoff_longitude=113.902 + i * 0.0015,
                    dropoff_latitude=22.502 + i * 0.0004,
                )
            )
    raw = tmp_path / "raw.csv"
    pd.DataFrame(rows).to_csv(raw, index=False)
    out = tmp_path / "prepared"
    report = prepare_orders("nanshan", out, n_l=1, n_h=3, n_od=2, raw_path=raw)
    data = load_saved(out)
    assert (report["n_l"], report["n_h"], report["n_od"]) == (1, 3, 2)
    assert data["demand"].sum() == 8
    sites = pd.read_csv(out / "sites.csv")
    assert set(sites.type) == {"L", "H"}
    assert len(sites) == 4
    assert (
        json.loads((out / "metadata.json").read_text())["aggregation"][
            "assigned_orders"
        ]
        == 16
    )
    membership = pd.read_csv(out / "order_membership.csv.gz")
    assert len(membership) == 16
    assert not membership.duplicated(["day", "order_id"]).any()
    counts = membership.groupby("od_id").size().sort_index().to_numpy()
    np.testing.assert_array_equal(counts / 2, data["demand"])
    periods = pd.read_csv(out / "periods.csv")
    assert periods.demand_orders.sum() == 16
    assert periods.demand_exposure_hours.sum() == 2
    with pytest.raises(FileExistsError):
        prepare_orders("nanshan", out, raw_path=raw)


def test_prepare_rejects_original_data_destination():
    with pytest.raises(ValueError, match="original data"):
        prepare_orders("nanshan", ROOT / "data/new_disallowed")


def test_candidate_routes_and_dtypes_survive_snapshot(tmp_path):
    from dataclasses import replace

    planning = to_planning_data(load_saved(SAVED))
    routes = ((0, 2, 3, 1), (1, 3, 2, 0))
    planning = replace(planning, candidate_routes=routes)
    packed = pack_planning_data(planning, {"purpose": "route replay"})
    save_inputs(packed, tmp_path)
    loaded = load_saved(tmp_path)
    replay = to_planning_data(loaded)
    assert replay.candidate_routes == routes
    assert replay.allowed_links.dtype == np.bool_
    assert replay.slot_counts.dtype.kind in "iu"
    assert replay.kappa.dtype.kind == "f"
    np.testing.assert_array_equal(replay.kappa, planning.kappa)


def test_unknown_legacy_version_rejected(tmp_path):
    save_inputs(dict(model_version="unrecognized", demand=np.ones(2)), tmp_path)
    with pytest.raises(ValueError, match="Unknown saved input"):
        load_saved(tmp_path)


def test_missing_deadline_and_travel_evidence_rejected(tmp_path):
    with np.load(SAVED / "input_arrays.npz") as arrays:
        old = {key: arrays[key].copy() for key in arrays.files}
    old.update(json.loads((SAVED / "input_metadata.json").read_text())["values"])
    old.pop("delivery_deadline")
    save_inputs(old, tmp_path / "no_deadline")
    with pytest.raises(ValueError, match="deadline"):
        load_saved(tmp_path / "no_deadline")
    old["delivery_deadline"] = np.ones(200)
    for key in (
        "ground_time",
        "last_ground_time",
        "rider_slope_minutes_per_straight_km",
    ):
        old.pop(key)
    save_inputs(old, tmp_path / "no_travel")
    with pytest.raises(ValueError, match="calibrated rider slope"):
        load_saved(tmp_path / "no_travel")


@pytest.mark.parametrize(
    "changes",
    [
        {"terminal": {"self_pickup_receipt_minutes": None}},
        {"operations": {"rider_minutes_per_km": "fast"}},
        {"operations": {"window_minutes": [1, 2]}},
        {"data": {"demand_scale": True}},
        {"compatibility": {"mode": "matrix", "matrix": [[1, 0, 1], [0, 1, 0]]}},
        {"compatibility": {"max_distance_km": float("nan")}},
        {"solver": {"threads": 1.5}},
    ],
)
def test_invalid_config_fails_before_model(changes):
    with pytest.raises(ValueError):
        load_config(changes)


def test_wrong_terminal_shape_fails_loading():
    with pytest.raises(ValueError, match="broadcast"):
        load_saved(SAVED, {"terminal": {"courier_receipt_minutes": [1, 2, 3]}})


def test_solver_only_config_does_not_change_new_snapshot(tmp_path):
    data = load_saved(
        SAVED,
        {
            "terminal": {"courier_receipt_minutes": 9},
            "operations": {"window_minutes": 0.8},
            "compatibility": {"mode": "none"},
        },
    )
    save_inputs(data, tmp_path)
    replay = load_saved(tmp_path, {"solver": {"time_limit": 3}})
    for key, value in data.items():
        if isinstance(value, np.ndarray):
            np.testing.assert_array_equal(replay[key], value)
        else:
            assert replay[key] == value


def test_explicit_physical_change_merges_saved_config_once(tmp_path):
    data = load_saved(
        SAVED,
        {
            "data": {"demand_scale": 2},
            "operations": {"deadline_multiplier": 1.5, "window_minutes": 0.8},
            "terminal": {"courier_receipt_minutes": 9},
        },
    )
    save_inputs(data, tmp_path)
    updated = load_saved(tmp_path, {"terminal": {"courier_receipt_minutes": 10}})
    np.testing.assert_array_equal(updated["demand"], data["demand"])
    np.testing.assert_allclose(updated["deadline"], data["deadline"])
    assert updated["delta"] == data["delta"]
    assert updated["metadata"]["resolved_config"]["data"]["demand_scale"] == 2
    sigma = updated["pickup_share"]
    np.testing.assert_allclose(
        updated["receipt_mean"], (5 * sigma + 10 * (1 - sigma)) / 60
    )


def test_max_orders_override_preserves_all_physics(tmp_path):
    data = load_saved(SAVED, {"operations": {"window_minutes": 0.8}})
    data["candidate_routes"] = ((0, 2, 3, 1),)
    save_inputs(data, tmp_path)
    updated = load_saved(tmp_path, {"max_orders": 1})
    p = to_planning_data(updated)
    assert p.max_orders == 1 and p.effective_delta == 0
    assert p.candidate_routes == ((0, 2, 3, 1),)
    assert updated["delta"] == data["delta"]
    np.testing.assert_array_equal(updated["flight_times"], data["flight_times"])
    assert updated["metadata"]["resolved_config"]["max_orders"] == 1


def test_physical_update_without_saved_scenario_requires_complete_input(tmp_path):
    packed = pack_planning_data(to_planning_data(load_saved(SAVED)))
    save_inputs(packed, tmp_path)
    load_saved(tmp_path, {"solver": {"time_limit": 1}})
    with pytest.raises(ValueError, match="complete explicit scenario"):
        load_saved(tmp_path, {"terminal": {"courier_receipt_minutes": 10}})
