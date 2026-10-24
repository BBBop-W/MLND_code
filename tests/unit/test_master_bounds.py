"""Valid operating bounds across every design of small physical instances."""

from dataclasses import replace
from itertools import product
from types import SimpleNamespace

import numpy as np
import pytest
from gurobipy import GRB

from src.model.locker_capacity import occupancy_limits
from src.model.paths import enumerate_paths
from src.model.types import PlanningData
from src.solvers.formulation import build_model, shapes
from src.solvers.strengthening import master_bounds


def tiny(max_orders=2, **changes):
    times = np.array([[0, 0.01, np.inf], [np.inf, 0, 0.04], [0.02, 0.02, 0]])
    values = dict(
        n_l=1,
        n_h=2,
        demand=[4.0, 7.0],
        flight_times=times,
        rider_mean=0.03,
        first_cost=1.0,
        terminal_cost=0.5,
        receipt_mean=0.03,
        removal_mean=0.02,
        ground_cost=[3.0, 5.0],
        deadline=0.5,
        base_cost=0.5,
        slot_counts=[1, 4],
        slot_cost=[0.1, 0.3],
        slot_epsilon=0.05,
        link_cost=0.05,
        kappa=[[1.0, 0.6], [0.3, 1.0]],
        delta=0.005,
        operating_limit=0.5,
        fleet_cost=1.0,
        flight_cost=0.2,
        waiting_cost=0.3,
        max_orders=max_orders,
    )
    values.update(changes)
    return PlanningData(**values)


def catalogue(*paths, complete=True):
    return SimpleNamespace(complete=complete, service_paths=paths)


def path(*, counts=((0, 1),), cost=2.0, time=0.0, slots=((0, 0.2),), pair=(0, 0)):
    return SimpleNamespace(
        served_counts=counts, cost=cost, time=time, slot_times=slots, b=pair
    )


def legal_designs(d):
    # Every legal binary design for this one-base two-locker directed network.
    for base_open in (0, 1):
        for levels in product(range(-1, d.slot_counts.shape[1]), repeat=d.n_h):
            design = {name: np.zeros(shape) for name, shape in shapes(d).items()}
            design["facility"][0] = base_open
            for locker, level in enumerate(levels):
                if level >= 0:
                    design["facility"][1 + locker] = 1
                    design["capacity_level"][locker, level] = 1
            permitted = [
                tuple(edge)
                for edge in np.argwhere(d.allowed_links)
                if design["facility"][edge[0]] and design["facility"][edge[1]]
            ]
            for assignment in product((0, 1) if base_open else (0,), repeat=d.n_od):
                design["assignment"][
                    (slice(None),) + (0,) * (design["assignment"].ndim - 1)
                ] = assignment
                for airline in product((0, 1), repeat=len(permitted)):
                    design["airline"][:] = 0
                    for edge, value in zip(permitted, airline):
                        design["airline"][edge] = value
                    yield {name: value.copy() for name, value in design.items()}


@pytest.mark.parametrize("max_orders", [1, 2])
def test_exhaustive_all_tiny_binary_designs_have_valid_cost_bounds(max_orders):
    d = tiny(max_orders)
    paths = enumerate_paths(d)
    bounds = master_bounds(d, paths)
    thresholds = occupancy_limits(d.slot_counts, d.slot_epsilon[:, None])
    assert any(
        p.mode == ("double" if max_orders == 2 else "single")
        for p in paths.service_paths
    )
    count = 0
    for design in legal_designs(d):
        built = build_model(d, paths, fixed_design=design)
        try:
            built.model.Params.TimeLimit = 2
            built.model.optimize()
            assert built.model.Status == GRB.OPTIMAL
            operating = float(built.model.ObjVal)
            first = float(d.demand @ d.ground_cost) - float(
                np.sum(
                    d.demand.reshape((-1,) + (1,) * (design["assignment"].ndim - 1))
                    * bounds["perassignment_savings"]
                    * design["assignment"]
                )
            )
            second = float(d.demand @ d.ground_cost) - bounds[
                "locker_value_per_load"
            ] * float(np.sum(thresholds * design["capacity_level"]))
            assert first <= operating + 2e-6
            assert second <= operating + 2e-6
            count += 1
        finally:
            built.model.dispose()
    assert count == 313


def test_aa_uses_two_order_occurrences_and_ab_uses_simultaneously_valid_shares():
    d = tiny()
    aa = path(counts=((0, 2),), cost=4.0)
    bounds = master_bounds(d, catalogue(aa))
    assert bounds["perassignment_savings"][0, 0, 0] == pytest.approx(1.0)
    assert bounds["perassignment_savings"][1, 0, 0] == 0
    ab = path(counts=((0, 1), (1, 1)), cost=4.0)
    bounds = master_bounds(d, catalogue(ab))
    np.testing.assert_allclose(bounds["perassignment_savings"][:, 0, 0], [1, 3])
    # OD 0 can individually lose money while the paired route remains profitable.
    mixed = replace(d, ground_cost=np.array([1.0, 6.0]))
    bounds = master_bounds(mixed, catalogue(ab))
    np.testing.assert_allclose(bounds["perassignment_savings"][:, 0, 0], [0, 4])
    assert bounds["locker_value_per_load"] == pytest.approx(15.0)


def test_busy_time_uses_complete_mission_and_slots_use_supplied_total_exactly():
    d = tiny()
    p = path(cost=1.0, time=0.5, slots=((0, 0.2), (1, 0.3)))
    bounds = master_bounds(d, catalogue(p))
    assert bounds["perassignment_savings"][0, 0, 0] == pytest.approx(1.5)
    assert bounds["locker_value_per_load"] == pytest.approx(3.0)


def test_profitable_zero_occupancy_disables_only_locker_cut_without_threshold():
    d = tiny()
    answer = master_bounds(d, catalogue(path(cost=1, time=0, slots=())))
    assert answer["locker_value_per_load"] is None
    assert answer["perassignment_savings"][0, 0, 0] == pytest.approx(2)
    assert answer["metadata"]["zero_load_profitable_path_indices"] == [0]
    assert (
        master_bounds(d, catalogue(path(cost=1, slots=((0, 1e-20),))))[
            "locker_value_per_load"
        ]
        is not None
    )


def test_no_paths_and_no_profitable_zero_occupancy_have_zero_gain():
    d = tiny()
    for paths in (catalogue(), catalogue(path(cost=10.0, slots=()))):
        answer = master_bounds(d, paths)
        assert not answer["perassignment_savings"].any()
        assert answer["locker_value_per_load"] == 0
        assert answer["metadata"]["assignment_global_operating_floor"] == pytest.approx(
            float(d.demand @ d.ground_cost)
        )


@pytest.mark.parametrize(
    "bad",
    [path(cost=-1), path(time=-1), path(slots=((0, -1),)), path(counts=((0, 0),))],
)
def test_unsafe_path_coefficients_are_rejected(bad):
    with pytest.raises(ValueError):
        master_bounds(tiny(), catalogue(bad))


def test_incomplete_catalogue_and_k1_double_are_rejected():
    with pytest.raises(ValueError, match="complete"):
        master_bounds(tiny(), catalogue(complete=False))
    with pytest.raises(ValueError, match="max_orders"):
        master_bounds(tiny(1), catalogue(path(counts=((0, 2),))))


def test_zero_demand_unsupported_od_never_multiplies_infinity():
    d = tiny(demand=[0.0, 7.0])
    answer = master_bounds(d, catalogue(path(counts=((1, 1),))))
    assert np.isfinite(answer["perassignment_savings"]).all()
    assert np.isfinite(answer["metadata"]["assignment_global_operating_floor"])


@pytest.mark.parametrize("max_orders", [1, 2])
def test_enabled_bounds_preserve_full_model_optimum_and_record_their_metadata(
    max_orders,
):
    from src.solvers.api import solve

    d = tiny(max_orders)
    full = solve(d, method="full_path", time_limit=10, mip_gap=1e-7)
    assert full["termination_reason"] == "gap_limit"
    for enabled in (False, True):
        answer = solve(
            d,
            method="benders_cg_plain",
            time_limit=10,
            mip_gap=1e-6,
            accelerations={"courier_savings_bound": enabled},
        )
        assert answer["termination_reason"] == "gap_limit"
        assert answer["objective"] == pytest.approx(full["objective"], abs=2e-5)
        assert answer["bound"] <= full["objective"] + 2e-6
        assert answer["master_bound_metadata"]["enabled"] is enabled
        if enabled:
            assert answer["master_bound_metadata"]["locker_cut_enabled"]
            assert answer["master_bound_metadata"]["numerical_margin"] > 0
