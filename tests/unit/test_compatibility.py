"""Physical compatibility must be proven by complete double-task paths."""

from dataclasses import replace
import json

import numpy as np
import pytest

from src.config import load_config
from src.data.instances import (
    _scenario,
    load_saved,
    to_planning_data,
)
from src.experiments.storage import save_inputs
from src.model.compatibility import screen_compatibility
from src.model.paths import enumerate_paths
from src.model.types import PathCollection, PlanningData


def test_screen_preserves_fractional_coefficients_and_input(small_input):
    original = to_planning_data(small_input)
    supplied = original.kappa.copy()
    supplied[0, 1] = 0
    original = replace(original, kappa=supplied)
    screened, diagnostic = screen_compatibility(original)
    np.testing.assert_array_equal(screened.kappa, supplied)
    np.testing.assert_array_equal(original.kappa, supplied)
    assert screened is not original
    assert diagnostic["proposed_pairs"] == diagnostic["retained_pairs"] == 3
    assert diagnostic["complete_double_paths"] is None
    assert diagnostic["algorithm"] == "separable_direct_existence_with_scalar_witnesses"
    assert diagnostic["removed_pairs"] == 0
    json.dumps(diagnostic)  # The evidence is saved directly in input metadata.


def test_complete_trip_limit_excludes_pair_even_when_single_task_exists(small_input):
    original = replace(to_planning_data(small_input), operating_limit=0.3)
    paths = enumerate_paths(original)
    assert any(p.second_od is None for p in paths.service_paths)
    assert not any(p.second_od is not None for p in paths.service_paths)
    screened, diagnostic = screen_compatibility(original)
    assert not screened.kappa.any()
    assert diagnostic["removed_pairs"] == 4
    assert diagnostic["removed_od_pairs"] == [[0, 0], [0, 1], [1, 0], [1, 1]]


def test_k1_and_k2_use_identical_two_order_screen(small_input):
    original = replace(to_planning_data(small_input), operating_limit=0.3)
    k1, d1 = screen_compatibility(replace(original, max_orders=1))
    k2, d2 = screen_compatibility(replace(original, max_orders=2))
    np.testing.assert_array_equal(k1.kappa, k2.kappa)
    assert k1.max_orders == 1 and k2.max_orders == 2
    assert d1 == d2


def test_zero_window_cannot_produce_physical_double_tasks(small_input):
    screened, diagnostic = screen_compatibility(
        replace(to_planning_data(small_input), delta=0)
    )
    assert not screened.kappa.any()
    assert diagnostic["complete"] and diagnostic["complete_double_paths"] == 0


def test_global_existence_is_not_existence_for_every_base_pair():
    # Base 0 is close to both lockers; base 1 cannot fit in any complete task.
    flight = np.full((4, 4), 0.05)
    flight[1, :] = flight[:, 1] = 1.0
    np.fill_diagonal(flight, 0)
    data = PlanningData(
        n_l=2,
        n_h=2,
        demand=[1],
        flight_times=flight,
        rider_mean=0,
        first_cost=0,
        terminal_cost=0,
        receipt_mean=0,
        removal_mean=0,
        ground_cost=1,
        deadline=2,
        base_cost=0,
        slot_counts=[10],
        slot_cost=0,
        slot_epsilon=0.05,
        link_cost=0,
        kappa=0.7,
        delta=0.01,
        operating_limit=0.3,
        fleet_cost=1,
        flight_cost=0,
        waiting_cost=0,
    )
    screened, diagnostic = screen_compatibility(data)
    assert screened.kappa[0, 0] == 0.7
    counts = {
        (row["departure"], row["return"]): row for row in diagnostic["base_pairs"]
    }
    assert counts[(0, 0)]["supported_pairs"] == 1
    for b in ((0, 1), (1, 0), (1, 1)):
        assert counts[b]["supported_pairs"] == 0
        assert counts[b]["globally_retained_but_unavailable"] == 1
    # Removing every locker-to-base return arc invalidates all complete paths,
    # despite leaving physically feasible first segments in the catalogue.
    allowed = data.allowed_links.copy()
    allowed[2:, :2] = False
    unavailable = replace(data, allowed_links=allowed)
    assert enumerate_paths(unavailable).initial_segments
    screened, diagnostic = screen_compatibility(unavailable)
    assert not screened.kappa.any()


def test_incomplete_catalogue_never_justifies_zeroing_compatibility(
    small_input, monkeypatch
):
    monkeypatch.setattr(
        "src.model.compatibility.enumerate_paths",
        lambda data: PathCollection((), (), (), complete=False),
    )
    with pytest.raises(ValueError, match="complete path catalogue"):
        screen_compatibility(
            replace(to_planning_data(small_input), route_mode="simple")
        )


@pytest.mark.parametrize("seed", range(12))
def test_direct_existence_matches_complete_enumeration(small_input, seed):
    rng = np.random.default_rng(seed)
    p = to_planning_data(small_input)
    n = 5
    flight = rng.uniform(0.005, 0.15, (n, n))
    np.fill_diagonal(flight, 0)
    allowed = rng.random((n, n)) > 0.15
    np.fill_diagonal(allowed, False)
    p = replace(
        p,
        n_l=2,
        n_h=3,
        flight_times=flight,
        allowed_links=allowed,
        demand=[2, 5],
        rider_mean=rng.uniform(0, 0.1, (2, 3)),
        first_cost=1,
        terminal_cost=1,
        receipt_mean=rng.uniform(0, 0.05, (2, 3)),
        removal_mean=0,
        deadline=rng.uniform(0.08, 0.4, 2),
        base_cost=1,
        slot_counts=[10, 20],
        slot_cost=1,
        slot_epsilon=0.05,
        link_cost=0,
        delta=0.025,
        operating_limit=float(rng.uniform(0.15, 0.5)),
    )
    expected = {b: np.zeros((2, 2), dtype=bool) for b in p.base_pairs}
    for path in enumerate_paths(p).service_paths:
        if path.second_od is not None:
            expected[path.b][path.first_od, path.second_od] = True
    screened, diag = screen_compatibility(p)
    union = np.logical_or.reduce(list(expected.values()))
    np.testing.assert_array_equal(screened.kappa, np.where(union, p.kappa, 0))
    for row in diag["base_pairs"]:
        b = row["departure"], row["return"]
        assert row["supported_pairs"] == int(expected[b].sum())


def _fresh_scenario(require_feasible_path=True):
    raw = {
        "bases": [[0, 0]],
        "hubs": [[1, 0], [2, 0]],
        "origins": [[1, 0]],
        "destinations": [[2, 0]],
        "od_origin": [0],
        "od_destination": [0],
        "demand": [10],
    }
    cfg = load_config(
        {
            "compatibility": {
                "mode": "all",
                "require_feasible_path": require_feasible_path,
            },
            "operations": {
                "rider_minutes_per_km": 1,
                "deadline_minutes": 30,
                "operating_limit_minutes": 0.1,
            },
        }
    )
    return _scenario(raw, cfg, {"kind": "synthetic_test"})


def test_fresh_scenario_records_screen_and_explicit_opt_out():
    fresh = _fresh_scenario()
    assert fresh["kappa"][0, 0] == 0
    assert fresh["metadata"]["compatibility_screen"]["removed_pairs"] == 1
    old_policy = _fresh_scenario(False)
    assert old_policy["kappa"][0, 0] == 1
    assert "compatibility_screen" not in old_policy["metadata"]


def test_old_snapshot_replay_and_flow_override_do_not_rescreen(tmp_path, monkeypatch):
    old = _fresh_scenario(False)
    old.pop("flow_policy")
    old["metadata"]["resolved_config"].pop("flow_policy")
    old["metadata"]["resolved_config"]["compatibility"].pop("require_feasible_path")
    save_inputs(old, tmp_path)

    def forbidden(data):
        raise AssertionError("replay must not apply the new compatibility screen")

    monkeypatch.setattr("src.model.compatibility.screen_compatibility", forbidden)
    replay = load_saved(tmp_path)
    assert to_planning_data(replay).flow_policy == "prefix"
    assert replay["kappa"][0, 0] == 1
    changed = load_saved(tmp_path, {"flow_policy": "aggregate"})
    assert to_planning_data(changed).flow_policy == "aggregate"
    for key, value in old.items():
        if isinstance(value, np.ndarray):
            np.testing.assert_array_equal(changed[key], value)
    # Unrelated explicit physical edits also retain the historical defaults.
    changed = load_saved(tmp_path, {"costs": {"flight_per_hour": 0.1}})
    assert to_planning_data(changed).flow_policy == "prefix"
    assert (
        changed["metadata"]["resolved_config"]["compatibility"]["require_feasible_path"]
        is False
    )


def test_explicit_screen_on_historical_snapshot_is_applied(tmp_path):
    old = _fresh_scenario(False)
    old.pop("flow_policy")
    old["metadata"]["resolved_config"].pop("flow_policy")
    old["metadata"]["resolved_config"]["compatibility"].pop("require_feasible_path")
    save_inputs(old, tmp_path)
    changed = load_saved(tmp_path, {"compatibility": {"require_feasible_path": True}})
    assert changed["kappa"][0, 0] == 0
    assert to_planning_data(changed).flow_policy == "prefix"
    assert changed["metadata"]["compatibility_screen"]["removed_pairs"] == 1
