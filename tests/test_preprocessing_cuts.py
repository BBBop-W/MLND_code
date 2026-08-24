from __future__ import annotations

import pytest

from src.config import ModelConfig
from src.data_processor.instance import DIRECT_ROUTE, Instance
from src.model.gurobi_model import MLNDModel


def _single_locker_route_instance(max_direct_range_km: float) -> Instance:
    config = ModelConfig(
        max_direct_range_km=max_direct_range_km,
        max_swap_leg_km=2.0,
    )
    return Instance(
        L=["L001"],
        H=["H001"],
        E=["E001"],
        coords={
            "L001": (0.0, 0.0),
            "H001": (4.0, 0.0),
            "E001": (2.0, 0.0),
        },
        config=config,
    ).build()


def test_direct_route_removes_all_dominated_swap_options() -> None:
    instance = _single_locker_route_instance(max_direct_range_km=4.0)

    assert instance.route_options["L001", "H001"] == (DIRECT_ROUTE,)
    assert ("L001", "H001", "E001") not in instance.route_distance
    assert ("L001", "E001") not in instance.A_Y
    assert ("E001", "H001") not in instance.A_Y


def test_swap_option_is_retained_when_direct_route_is_infeasible() -> None:
    instance = _single_locker_route_instance(max_direct_range_km=3.9)

    assert instance.route_options["L001", "H001"] == ("E001",)
    assert instance.route_distance["L001", "H001", "E001"] == pytest.approx(4.0)
    assert ("L001", "E001") in instance.A_Y
    assert ("E001", "H001") in instance.A_Y


def _bounded_flow_instance(total_demand: float, service_rates: tuple[float, ...]) -> Instance:
    config = ModelConfig(
        availability_levels=(0.50, 0.90),
        locker_service_rates=service_rates,
        locker_capacity_costs=tuple(float(index + 1) for index in range(len(service_rates))),
        max_direct_range_km=10.0,
        max_locker_time_hours=0.1,
        origin_locker_choices=None,
        home_corridor_choices=None,
    )
    return Instance(
        L=["L001"],
        H=["H001", "H002"],
        N=["O001", "D001"],
        coords={
            "L001": (0.0, 0.0),
            "H001": (1.0, 0.0),
            "H002": (2.0, 0.0),
            "O001": (0.0, 0.0),
            "D001": (2.0, 0.0),
        },
        h_od={("O001", "D001"): total_demand},
        config=config,
    ).build()


@pytest.mark.parametrize(
    ("total_demand", "expected_bound"),
    [
        (50.0, 45.0),
        (1_000.0, 90.0),
    ],
)
def test_physical_flow_bound_uses_demand_and_capacity_minimum(
    total_demand: float,
    expected_bound: float,
) -> None:
    instance = _bounded_flow_instance(total_demand, service_rates=(50.0, 100.0))

    assert instance.A_LH
    assert instance.A_HH
    assert instance.A_HL
    assert set(instance.M_psi) == instance.A_F
    assert all(bound == pytest.approx(expected_bound) for bound in instance.M_psi.values())


def test_capacity_levels_that_cannot_meet_qos_are_rejected() -> None:
    with pytest.raises(ValueError, match="mu >= 1 / max_locker_time_hours"):
        _bounded_flow_instance(100.0, service_rates=(9.0,))


def test_offered_corridor_bound_inherits_realized_capacity_bound() -> None:
    instance = _bounded_flow_instance(1_000.0, service_rates=(50.0, 100.0))
    model = MLNDModel(instance, "HD")

    assert model.M_w
    inherited_bound = max(instance.M_psi.values()) / min(
        instance.config.availability_levels
    )
    assert max(model.M_w.values()) <= inherited_bound + 1e-9
