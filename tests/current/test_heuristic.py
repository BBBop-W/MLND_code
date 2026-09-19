"""Restricted-column primal designs must remain feasible in the full model."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from src.optimization import direct as direct_solver, heuristic
from src.model.network import levels, network
from src.operations.validation import build_operations
from src.pricing.oracle import enumerate_all_columns

_spec = importlib.util.spec_from_file_location(
    "optimization_fixture", Path(__file__).with_name("test_optimization.py")
)
_fixture = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_fixture)


def _data():
    data = _fixture._tiny()
    data["pickup_share"] = np.zeros((2, 2))
    data.update(base_site_cost=0.5, hub_site_cost=0.8, swap_site_cost=0.3)
    return data


def _baseline(data):
    net = network(data)
    _, capacity, _ = levels(data)
    L, H, D = data["n_l"], data["n_h"], data["n_od"]
    return dict(
        _solution=dict(
            facility=np.zeros(net["N"]),
            capacity_level=np.zeros_like(capacity),
            assignment=np.zeros((D, L, L)),
            airline=np.zeros(net["count"]),
            rho=np.zeros(L),
            idle=np.zeros(L),
            fleet=0.0,
            hub_count=np.zeros(H),
            empty=np.zeros((L, L)),
        ),
        _missions=[],
        _relocations=[],
        objective=float(data["demand"] @ data["ground"]),
    )


def test_complete_pool_matches_direct_and_does_not_export_restricted_bound():
    data = _data()
    cols = enumerate_all_columns(data)
    direct = direct_solver.solve(data, columns=cols, time_limit=30.0, mip_gap=1e-4)
    improved = heuristic.solve(
        data,
        _baseline(data),
        columns=cols,
        add_oracle_pairs=0,
        time_limit=30.0,
        threads=1,
        global_lower_bound=1.0,
    )
    assert improved["objective"] == pytest.approx(direct["objective"], abs=3e-3)
    assert improved["served"] > 1.0
    assert improved["bound"] == 1.0
    assert improved["heuristic_stats"]["restricted_bound_exported"] is False
    assert improved["heuristic_stats"]["improved"]
    assert build_operations(data, improved)["summary"]["checks"]["passed"]


def test_empty_seed_generates_new_oracle_columns_and_serves_orders():
    data = _data()
    improved = heuristic.solve(data, _baseline(data), time_limit=30.0, threads=1)
    assert improved["served"] > 1.0
    assert improved["objective"] < improved["heuristic_seed_objective"]
    assert improved["heuristic_stats"]["assignment_binaries"] <= 2 * data["n_od"]
    assert build_operations(data, improved)["summary"]["checks"]["passed"]


def test_changed_costs_rebuild_seed_and_can_close_facilities():
    data = _data()
    old = direct_solver.solve(data, time_limit=30.0)
    assert old["served"] > 1.0
    old_objective = old["objective"]
    new = dict(data)
    new.update(
        fleet_cost=2 * data["fleet_cost"],
        drone_time_cost=2 * data["drone_time_cost"],
        hover_cost=2 * data["hover_cost"],
        hub_site_cost=10000.0,
        base_site_cost=10000.0,
    )
    new["ground"] = 0.8 * data["ground"]
    new["first_cost"] = 0.8 * data["first_cost"]
    new["terminal_cost"] = 0.8 * data["terminal_cost"]
    improved = heuristic.solve(new, old, time_limit=30.0, threads=1)
    assert improved["heuristic_seed_objective"] > old_objective + 10000.0
    assert improved["objective"] == pytest.approx(
        float(new["demand"] @ new["ground"]), abs=1e-5
    )
    assert improved["opened"] == dict(L=0, H=0, E=0)
    assert improved["bound"] == 0.0
    assert old["objective"] == old_objective
    assert build_operations(new, improved)["summary"]["checks"]["passed"]


def test_empty_pool_retains_feasible_baseline():
    data = _data()
    improved = heuristic.solve(
        data, _baseline(data), add_oracle_pairs=0, time_limit=10.0, threads=1
    )
    assert improved["served"] == 0.0
    assert improved["objective"] == pytest.approx(
        float(data["demand"] @ data["ground"])
    )
    assert improved["bound"] == 0.0


def test_user_stop_preserves_a_feasible_incumbent_and_global_bound():
    data = _data()
    result = heuristic.solve(
        data,
        _baseline(data),
        add_oracle_pairs=0,
        time_limit=30.0,
        threads=1,
        global_lower_bound=1.0,
        stop_requested=lambda: True,
    )
    assert result["termination_reason"] == "user_stop"
    assert result["bound"] == 1.0
    assert build_operations(data, result)["summary"]["checks"]["passed"]
