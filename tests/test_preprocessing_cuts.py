from __future__ import annotations

import pytest

from src.config import ModelConfig
from src.data_processor.instance import DIRECT_ROUTE, Instance
from src.model.gurobi_model import MLNDModel, _aggregate_raw_offer_bounds


def _single_locker_route_instance(
    battery_range_km: float,
    destination_x_km: float = 4.0,
) -> Instance:
    config = ModelConfig(
        battery_range_km=battery_range_km,
    )
    return Instance(
        L=["L001"],
        H=["H001"],
        E=["E001"],
        coords={
            "L001": (0.0, 0.0),
            "H001": (destination_x_km, 0.0),
            "E001": (destination_x_km / 2.0, 0.0),
        },
        config=config,
    ).build()


def test_direct_and_swap_options_are_both_retained_when_feasible() -> None:
    instance = _single_locker_route_instance(battery_range_km=4.0)

    assert instance.route_options["L001", "H001"] == (DIRECT_ROUTE, "E001")
    assert instance.route_distance["L001", "H001", "E001"] == pytest.approx(4.0)
    assert ("L001", "E001") in instance.A_Y
    assert ("E001", "H001") in instance.A_Y


def test_swap_option_is_retained_when_direct_route_is_infeasible() -> None:
    instance = _single_locker_route_instance(battery_range_km=3.9)

    assert instance.route_options["L001", "H001"] == ("E001",)
    assert instance.route_distance["L001", "H001", "E001"] == pytest.approx(4.0)
    assert ("L001", "E001") in instance.A_Y
    assert ("E001", "H001") in instance.A_Y


def test_confirmed_range_boundary_and_speed_define_route_times() -> None:
    instance = _single_locker_route_instance(
        battery_range_km=10.0,
        destination_x_km=10.0,
    )

    key = ("L001", "H001")
    assert instance.route_options[key] == (DIRECT_ROUTE, "E001")
    assert instance.route_time[*key, DIRECT_ROUTE] == pytest.approx(10.0 / 72.0)
    assert instance.route_time[*key, "E001"] == pytest.approx(10.0 / 72.0 + 3.0 / 60.0)

    infeasible = _single_locker_route_instance(
        battery_range_km=4.99,
        destination_x_km=10.0,
    )
    assert key not in infeasible.A_F
    assert key not in infeasible.route_options


def _bounded_flow_instance(
    total_demand: float, service_rates: tuple[float, ...]
) -> Instance:
    config = ModelConfig(
        availability_levels=(0.50, 0.90),
        locker_service_rates=service_rates,
        locker_capacity_costs=tuple(
            float(index + 1) for index in range(len(service_rates))
        ),
        battery_range_km=10.0,
        max_locker_time_hours=0.1,
        display_origin_choices=None,
        general_corridor_choices=None,
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
    assert all(
        bound == pytest.approx(expected_bound) for bound in instance.M_psi.values()
    )


def test_capacity_levels_that_cannot_meet_qos_are_rejected() -> None:
    with pytest.raises(ValueError, match="mu >= 1 / max_locker_time_hours"):
        _bounded_flow_instance(100.0, service_rates=(9.0,))


@pytest.mark.parametrize("mode", ["MX", "HD", "LO"])
def test_raw_offer_bounds_accumulate_once_per_retained_market_corridor(
    mode: str,
) -> None:
    market_corridors = {
        ("O1", "D1"): (("H1", "H2"), ("H2", "H3")),
        ("O2", "D2"): (("H1", "H2"),),
    }
    h_od = {("O1", "D1"): 7.0, ("O2", "D2"): 11.0}
    choice_share = {
        ("O1", "D1", "H2"): 0.25,
        ("O1", "D1", "H3"): 0.50,
        ("O2", "D2", "H2"): 0.75,
    }

    bounds = _aggregate_raw_offer_bounds(
        market_corridors,
        h_od,
        choice_share,
        mode,
    )

    if mode == "LO":
        assert bounds == pytest.approx(
            {
                ("H1", "H2"): 7.0 * 0.25 + 11.0 * 0.75,
                ("H2", "H3"): 7.0 * 0.50,
            }
        )
    else:
        assert bounds == pytest.approx(
            {("H1", "H2"): 18.0, ("H2", "H3"): 7.0}
        )


@pytest.mark.parametrize("mode", ["MX", "HD", "LO"])
def test_offered_corridor_bound_is_the_tight_demand_capacity_minimum(
    mode: str,
) -> None:
    instance = _bounded_flow_instance(1_000.0, service_rates=(50.0, 100.0))
    model = MLNDModel(instance, mode)

    assert model.M_offer
    assert set(model.M_offer) == set(model.corridors)
    rho_min = min(instance.config.availability_levels)
    for arc, bound in model.M_offer.items():
        demand_bound = sum(
            instance.h_od[market]
            * (
                instance.choice_share[market[0], market[1], arc[1]]
                if mode == "LO"
                else 1.0
            )
            for market in instance.markets
            if arc in model.market_corridors[market]
        )
        capacity_bound = instance.M_psi[arc] / rho_min
        assert bound == pytest.approx(min(demand_bound, capacity_bound))
        assert bound <= demand_bound + 1e-9
        assert bound <= capacity_bound + 1e-9

    model.build()
    assert all(
        model.model.getConstrByName(f"corridor_offer_activation[{i},{j}]") is not None
        for i, j in model.corridors
    )
