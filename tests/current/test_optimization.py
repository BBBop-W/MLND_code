"""Full-column reference, CG equality, and cross-design cut certification."""

import functools
import importlib.util
from pathlib import Path

import gurobipy as gp
import numpy as np
import pytest

import src.solvers.direct.path as direct_solver
import src.solvers.benders.subproblem as recourse
import src.solvers.benders.engine as benders
from src.model.network import network
from src.solvers.pricing.label import enumerate_all_columns

_spec = importlib.util.spec_from_file_location(
    "physical_fixture", Path(__file__).with_name("test_pricing.py")
)
_fixturemod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_fixturemod)


def _data():
    d = _fixturemod._fixture()
    d.update(
        demand=np.asarray([8.0, 10.0, 7.0]),
        base_cost=0.1,
        swap_cost=0.1,
        capacity_levels=np.asarray([1.0, 2.0]),
        level_capacity=np.tile([8.0, 30.0], (3, 1)),
        level_cost=np.tile([0.1, 0.2], (3, 1)),
    )
    return d


def _assignment(pairs, D=3, L=2):
    a = np.zeros((D, L, L))
    for od, pair in enumerate(pairs):
        if pair is not None:
            a[od, pair[0], pair[1]] = 1
    return a


def _designs(d):
    net = network(d)
    L, H = d["n_l"], d["n_h"]
    allf = np.ones(net["N"])
    high = np.zeros((H, 2))
    high[:, 1] = 1
    pairs = [
        [(0, 1), (1, 0), (0, 0)],
        [(0, 0), (1, 1), (0, 0)],
        [(1, 1), (0, 0), (1, 0)],
        [(0, 1), None, (1, 1)],
        [None, None, None],
    ]
    designs = [
        (allf.copy(), high.copy(), _assignment(p), np.ones(net["count"])) for p in pairs
    ]
    for h in range(H):
        z = high.copy()
        z[h] = [1, 0]
        designs.append((allf.copy(), z, _assignment(pairs[0]), np.ones(net["count"])))
    f = allf.copy()
    f[-1] = 0
    designs.append(
        (f, high.copy(), _assignment(pairs[0]), f[net["uv"][:, 0]] * f[net["uv"][:, 1]])
    )
    f = allf.copy()
    f[L + 1] = 0
    z = high.copy()
    z[1] = 0
    designs.append(
        (f, z, _assignment(pairs[0]), f[net["uv"][:, 0]] * f[net["uv"][:, 1]])
    )
    for edge in [(0, L), (L, L + 1), (L + 2, 1)]:
        yy = np.ones(net["count"])
        yy[net["lookup"][edge[0] * net["N"] + edge[1]]] = 0
        designs.append((allf.copy(), high.copy(), _assignment(pairs[0]), yy))
    designs.append(
        (allf.copy(), high.copy(), _assignment(pairs[0]), np.zeros(net["count"]))
    )
    return designs


@functools.lru_cache(maxsize=1)
def _reference_results():
    d = _data()
    columns = enumerate_all_columns(d)
    designs = _designs(d)
    results = []
    for f, z, a, y in designs:
        result = direct_solver.solve(
            d,
            fixed_facility=f,
            fixed_capacity_level=z,
            fixed_assignment=a,
            fixed_airlines=y,
            columns=columns,
            time_limit=20.0,
        )
        assert result["status"] == gp.GRB.OPTIMAL
        results.append(result)
    return d, designs, results


@pytest.mark.parametrize("index", [0, 1, 2, 3, 4, 5, 8, 9, 10, 13])
def test_cg_matches_all_columns_on_nontrivial_and_closed_designs(index):
    d, designs, references = _reference_results()
    f, z, a, y = designs[index]
    result = recourse.solve(d, f, z, a, y, time_limit=30.0)
    reference = references[index]
    assert result["cg_complete"], result["duality_gap"]
    assert result["objective"] == pytest.approx(
        reference["recourse_objective"], abs=2e-5
    )
    assert result["bound"] <= reference["recourse_objective"] + 2e-6
    assert result["primal_residual"] < 1e-6
    if index in (0, 1, 2, 3, 5, 8, 9, 10):
        assert reference["served"] > 1.0
        assert result["fleet"] > 0.0
    if index == 13:
        assert result["served"] == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("source", [0, 1, 3, 4, 5, 8, 10, 13])
def test_benders_cut_valid_across_assignments_links_levels_and_facilities(source):
    d, designs, references = _reference_results()
    f, z, a, y = designs[source]
    cut = recourse.solve(d, f, z, a, y, time_limit=30.0)
    assert cut["cut_valid"]
    for target, (ff, zz, aa, yy) in enumerate(designs):
        value = (
            cut["cut_intercept"]
            + cut["cut_facility"] @ ff
            + np.sum(cut["cut_level"] * zz)
            + np.sum(cut["cut_B"] * aa)
            + cut["cut_airline"] @ yy
        )
        assert value <= references[target]["recourse_objective"] + 2e-5, (
            source,
            target,
            value,
            references[target]["recourse_objective"],
        )
    assert abs(cut["objective"] - cut["bound"]) < 2e-4


def _tiny():
    d = _data()
    d.update(
        n_l=2,
        n_h=2,
        n_k=0,
        n_od=2,
        battery_time=1.0,
        hubs=np.asarray([[1.0, 0.0], [3.0, 0.0]]),
        remote=np.empty((0, 2)),
        ground_time=np.asarray([[0.01, 0.04], [0.04, 0.01]]),
        first_cost=np.asarray([[0.2, 2.0], [2.0, 0.2]]),
        terminal_cost=np.asarray([[1.0, 0.1], [0.1, 1.0]]),
        terminal_time=np.full((2, 2), 0.01),
        od_origin=np.asarray([0, 1]),
        od_destination=np.asarray([0, 1]),
        ground=np.asarray([8.0, 8.0]),
        demand=np.asarray([9.0, 9.0]),
        delivery_deadline=np.asarray([0.5, 0.5]),
        pickup_handling_time=0.004,
        delivery_handling_time=0.004,
        swap_time=0.0,
        base_cost=0.2,
        swap_cost=0.2,
        level_capacity=np.tile([6.0, 30.0], (2, 1)),
        level_cost=np.tile([0.1, 0.2], (2, 1)),
        airline_cost_multiplier=0.2,
    )
    xy = np.vstack((d["bases"], d["hubs"], d["remote"]))
    d["flight_times"] = np.linalg.norm(xy[:, None] - xy[None, :], axis=2) * 0.02
    return d


def test_complete_binary_master_benders_matches_full_misocp():
    d = _tiny()
    direct = direct_solver.solve(d, time_limit=30.0)
    assert direct["status"] == gp.GRB.OPTIMAL
    assert direct["served"] > 5.0 and direct["fleet"] > 0.0
    progress = []
    result = benders.solve(
        d, time_limit=60.0, mip_gap=1e-6, max_iterations=200, progress=progress.append
    )
    assert result["objective"] == pytest.approx(direct["objective"], abs=2e-4), (
        result["termination_reason"],
        result["gap"],
    )
    assert result["bound"] <= direct["objective"] + 2e-5
    assert result["gap"] < 2e-5
    assert result["stats"]["master_continuous"] == 1
    assert result["stats"]["master_integers"] == result["stats"]["master_binaries"]
    assert (
        progress[0]["stage"] == "master_ready" and progress[-1]["stage"] == "finished"
    )
    assert all(p["bound"] <= p["objective"] + 2e-5 for p in progress)
    assert progress[-1]["objective"] == pytest.approx(result["objective"])


def test_full_model_reports_gamma_as_fleet_occupancy():
    d = _tiny()
    result = direct_solver.solve(d, time_limit=30.0)
    occupancy = sum(result["idle"]) + sum(
        p["time"] * p["flow"] for p in result["missions"] + result["relocations"]
    )
    assert result["fleet"] == pytest.approx(occupancy, abs=2e-5)
    assert np.all(result["rho"] < 1)
    assert sum(result["capacity_level"].ravel()) > 0
    assert np.allclose(
        result["capacity_level"], np.rint(result["capacity_level"]), atol=1e-7
    )


def test_location_setup_costs_match_full_misocp_and_operational_reconstruction():
    from src.operations.validation import build_operations

    d = _tiny()
    d.update(
        base_site_cost=np.array([0.4, 0.8]),
        hub_site_cost=1.7,
        swap_site_cost=3.0,
        pickup_share=np.zeros((2, 2)),
    )
    direct = direct_solver.solve(d, time_limit=30.0)
    result = benders.solve(d, time_limit=45.0, mip_gap=1e-6, max_iterations=200)
    assert direct["status"] == gp.GRB.OPTIMAL
    assert direct["served"] > 5.0
    assert result["objective"] == pytest.approx(direct["objective"], abs=2e-4)
    assert result["bound"] <= direct["objective"] + 2e-5
    assert result["gap"] < 2e-5
    assert result["stats"]["master_continuous"] == 1
    for solved in (direct, result):
        ops = build_operations(d, solved)
        assert ops["summary"]["checks"]["passed"], ops["summary"]["checks"]
        assert sum(row["cny_per_hour"] for row in ops["tables"]["site_costs"]) > 3.0


def test_interrupted_column_generation_still_returns_global_lower_cut():
    d, designs, references = _reference_results()
    f, z, a, y = designs[0]
    cut = recourse.solve(d, f, z, a, y, time_limit=30.0, max_iterations=0)
    assert cut["cut_valid"] and not cut["fully_priced"]
    for target, (ff, zz, aa, yy) in enumerate(designs):
        value = (
            cut["cut_intercept"]
            + cut["cut_facility"] @ ff
            + np.sum(cut["cut_level"] * zz)
            + np.sum(cut["cut_B"] * aa)
            + cut["cut_airline"] @ yy
        )
        assert value <= references[target]["recourse_objective"] + 2e-5


def test_unlimited_default_gap_checkpoints_reconstruct_complete_feasible_plans(
    tmp_path,
):
    from src.operations.validation import build_operations

    d = _tiny()
    d["pickup_share"] = np.zeros((2, 2))
    checkpoints = []
    result = benders.solve(
        d,
        time_limit=None,
        max_iterations=None,
        mip_gap=None,
        mip_gap_abs=None,
        checkpoint=checkpoints.append,
        log=tmp_path / "master.log",
    )
    reference = direct_solver.solve(d, time_limit=30.0)
    default = gp.Model()
    default.Params.OutputFlag = 0
    try:
        assert result["params"]["MIPGap"] == default.Params.MIPGap
        assert result["params"]["MIPGapAbs"] == default.Params.MIPGapAbs
    finally:
        default.dispose()
    assert result["budget"] is None and result["params"]["max_iterations"] is None
    assert result["params"]["cg_max_iterations"] is None
    assert result["termination_reason"] == "gap_limit"
    assert result["objective"] - reference["objective"] < 2e-4
    assert result["gap"] <= result["params"]["MIPGap"]
    assert checkpoints[0]["checkpoint_stage"] == "master_ready"
    assert checkpoints[-1]["checkpoint_stage"] == "finished"
    assert any(p["checkpoint_stage"] == "incumbent_improved" for p in checkpoints)
    for saved in checkpoints:
        checks = build_operations(d, saved)["summary"]["checks"]
        assert checks["passed"], checks
    assert "Gurobi Optimizer" in (tmp_path / "master.log").read_text()


def test_initial_result_and_bound_survive_immediate_user_stop():
    from src.operations.validation import build_operations

    d = _tiny()
    d["pickup_share"] = np.zeros((2, 2))
    initial = benders.solve(d, time_limit=30.0)
    checkpoints = []
    result = benders.solve(
        d,
        time_limit=None,
        max_iterations=None,
        mip_gap=None,
        mip_gap_abs=None,
        initial_result=initial,
        stop_requested=lambda: True,
        checkpoint=checkpoints.append,
    )
    assert result["termination_reason"] == "user_stop"
    assert result["status"] == gp.GRB.INTERRUPTED
    assert result["objective"] == initial["objective"]
    assert result["bound"] == initial["bound"]
    assert result["served"] > 0 and result["_missions"]
    assert checkpoints[-1]["termination_reason"] == "user_stop"
    assert build_operations(d, result)["summary"]["checks"]["passed"]


def test_user_stop_during_master_callback_preserves_latest_incumbent():
    from src.operations.validation import build_operations

    d = _tiny()
    d["pickup_share"] = np.zeros((2, 2))
    state = dict(stop=False)

    def progress(event):
        if event["stage"] == "master_start":
            state["stop"] = True

    result = benders.solve(
        d,
        time_limit=None,
        max_iterations=None,
        mip_gap=None,
        mip_gap_abs=None,
        progress=progress,
        stop_requested=lambda: state["stop"],
    )
    assert result["termination_reason"] == "user_stop"
    assert result["status"] == gp.GRB.INTERRUPTED
    assert result["served"] > 0
    assert build_operations(d, result)["summary"]["checks"]["passed"]


def test_recourse_user_stop_inside_optimizer_callback():
    d = _data()
    f, z, a, y = _designs(d)[0]
    state = dict(calls=0)

    def stop():
        state["calls"] += 1
        # First calls occur at entry, at the CG boundary and before optimize;
        # the fourth call is the first optimizer callback.
        return state["calls"] >= 4

    with pytest.raises(recourse.RecourseStopped, match="user_stop"):
        recourse.solve(
            d, f, z, a, y, time_limit=None, max_iterations=None, stop_requested=stop
        )
    assert state["calls"] >= 4


def test_unlimited_recourse_matches_full_columns():
    d, designs, references = _reference_results()
    f, z, a, y = designs[0]
    result = recourse.solve(d, f, z, a, y, time_limit=None, max_iterations=None)
    assert result["cg_complete"]
    assert result["objective"] == pytest.approx(
        references[0]["recourse_objective"], abs=2e-5
    )


def test_interrupted_recourse_retains_previous_full_result(monkeypatch):
    from src.operations.validation import build_operations

    d = _tiny()
    d["pickup_share"] = np.zeros((2, 2))
    initial = benders.solve(d, time_limit=30.0)

    def stopped(*args, **kwargs):
        raise recourse.RecourseStopped("user_stop")

    monkeypatch.setattr(benders, "recourse", stopped)
    checkpoints = []
    result = benders.solve(
        d,
        time_limit=None,
        max_iterations=None,
        mip_gap=None,
        mip_gap_abs=None,
        initial_result=initial,
        checkpoint=checkpoints.append,
    )
    assert result["termination_reason"] == "user_stop"
    assert result["objective"] == initial["objective"]
    assert result["bound"] == initial["bound"]
    assert checkpoints[-1]["_missions"]
    assert build_operations(d, checkpoints[-1])["summary"]["checks"]["passed"]
