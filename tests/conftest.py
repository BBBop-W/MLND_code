"""Small physical network shared by end-to-end tests."""

import numpy as np
import pytest
from src.model.types import PlanningData


@pytest.fixture
def small_input():
    from src.data.instances import pack_planning_data

    p = PlanningData(
        n_l=1,
        n_h=2,
        demand=[4.0, 7.0],
        flight_times=[[0, 0.05, 0.07], [0.05, 0, 0.1], [0.07, 0.1, 0]],
        rider_mean=[[0.08, 0.12], [0.1, 0.06]],
        first_cost=1.0,
        terminal_cost=0.5,
        receipt_mean=0.08,
        removal_mean=0.03,
        ground_cost=5.0,
        deadline=1.0,
        base_cost=0.5,
        slot_counts=[10, 20],
        slot_cost=[0.1, 0.2],
        slot_epsilon=0.05,
        link_cost=0.0,
        kappa=[[0.3, 0.9], [0.2, 0.7]],
        delta=0.04,
        operating_limit=1.0,
        fleet_cost=1.0,
        flight_cost=0.2,
        waiting_cost=0.3,
    )
    data = pack_planning_data(p)
    data.update(bases=np.array([[0.0, 0.0]]), hubs=np.array([[1.0, 0.0], [1.0, 1.0]]))
    return data
