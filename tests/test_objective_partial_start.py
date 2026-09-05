from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from gurobipy import GRB

from main import _solve, build_parser
import src.model.gurobi_model as gurobi_model_module
import src.model.primal_heuristic as primal_heuristic
from src.benchmark.branch_cut.telemetry import collect_solver_metrics
from src.config import SolverConfig
from src.data_processor.instance import DIRECT_ROUTE
from src.model.gurobi_model import MLNDModel
from src.model.primal_heuristic import (
    ApproximateObjectiveBreakdown,
    CapacityOption,
    ObjectiveAwareOpportunity,
    ObjectiveAwarePlanningConfig,
    ObjectiveAwareSupportCandidate,
    build_all_courier_partial_assignment,
    build_candidate_partial_assignment,
    build_partial_start_vector,
    effective_objective_start_budget,
    evaluate_objective_support,
    install_partial_mip_starts,
    plan_objective_aware_partial_starts,
)
from src.utils.output_processor import _json_safe_mapping


class FakeVar:
    def __init__(self, name: str):
        self.VarName = name


class MockStartModel:
    def __init__(self, variables: list[FakeVar]):
        self._variables = variables
        self.NumStart = 0
        self.Params = SimpleNamespace(StartNumber=-1)
        self.sparse_values: dict[int, dict[FakeVar, float]] = {}

    def update(self) -> None:
        return None

    def getVars(self) -> list[FakeVar]:
        return self._variables

    def setAttr(self, attribute, variables, values) -> None:
        assert attribute == GRB.Attr.Start
        self.sparse_values[self.Params.StartNumber] = dict(zip(variables, values))


def _opportunity(
    corridor: tuple[str, str],
    market: tuple[str, str],
    *,
    direct: float = 20.0,
    reliability_index: int = 0,
    reliability: float = 0.5,
    idle_state: str = "L1",
    display_locker: str | None = None,
    display_blockers: tuple[str, ...] = (),
    first_mile: float = 1.0,
    last_mile: float = 1.0,
    pickup: float = 0.0,
    outgoing_route_time_hours: tuple[tuple[str, float], ...] | None = None,
    base_departures_per_unit: float = 1.0,
    swap_events_per_unit: float = 0.0,
    offered_demand: float = 4.0,
) -> ObjectiveAwareOpportunity:
    i, j = corridor
    return ObjectiveAwareOpportunity(
        market=market,
        corridor=corridor,
        reliability_index=reliability_index,
        reliability=reliability,
        offered_demand=offered_demand,
        support_facilities=("L1", "E1", i, j),
        open_h=(i, j),
        active_connections=(("L1", i), (i, j), (j, "L1")),
        facility_costs=(("L1", 1.0), ("E1", 1.0), (i, 1.0), (j, 1.0)),
        connection_costs=(
            (("L1", i), 0.1),
            ((i, j), 0.1),
            ((j, "L1"), 0.1),
        ),
        idle_state=idle_state,
        incoming_h=(i, j),
        outgoing_route_time_hours=(
            outgoing_route_time_hours
            if outgoing_route_time_hours is not None
            else ((i, 0.03), (j, 0.03))
        ),
        display_locker=display_locker,
        display_blockers=display_blockers,
        direct_unit_cost=direct,
        first_mile_unit_cost=first_mile,
        last_mile_unit_cost=last_mile,
        mx_pickup_unit_cost=pickup,
        flight_unit_cost=1.0,
        route_time_hours=0.1,
        base_departures_per_unit=base_departures_per_unit,
        swap_events_per_unit=swap_events_per_unit,
    )


def _planning_config(*, max_candidates: int = 3) -> ObjectiveAwarePlanningConfig:
    return ObjectiveAwarePlanningConfig(
        candidate_lockers=("h1", "h2", "h3", "h4", "h5", "h6"),
        capacity_options=(CapacityOption(0, 20.0, 1.0),),
        max_locker_time_hours=0.1,
        usable_battery_time_hours=1.0,
        drone_cost_per_hour=1.0,
        max_candidates=max_candidates,
        min_open_h=2,
        min_service=1.0,
        min_savings=0.1,
    )


def test_objective_breakdown_is_complete_and_auditable() -> None:
    terms = ApproximateObjectiveBreakdown(
        direct_courier_baseline=100.0,
        first_mile_rider=10.0,
        last_mile_rider=8.0,
        mx_locker_pickup=2.0,
        flight=7.0,
        facility=6.0,
        capacity=5.0,
        connection=4.0,
        queue=3.0,
        fleet=2.0,
    )
    terms.validate()
    assert terms.network_cost == pytest.approx(47.0)
    assert terms.net_savings == pytest.approx(53.0)


def test_capacity_and_queue_proxy_counts_every_cycle_h_arrival() -> None:
    opportunity = _opportunity(("h1", "h2"), ("o1", "d1"))
    candidate, rejection = evaluate_objective_support(
        (opportunity,),
        (opportunity.support_key,),
        _planning_config(),
    )
    assert rejection is None
    assert candidate is not None
    # Successful service is 2.  The physical cycle is L1->h1->h2->L1, so both
    # h1 and h2 have incoming drone flow and an M/M/1 population of 2/(20-2).
    assert candidate.objective.queue == pytest.approx(2.0 * 2.0 / 18.0)
    assert candidate.capacity_levels == (("h1", 0), ("h2", 0))


def test_fractional_service_stops_exactly_at_the_locker_capacity_boundary() -> None:
    opportunity = _opportunity(
        ("h1", "h2"),
        ("o1", "d1"),
        offered_demand=100.0,
        swap_events_per_unit=0.5,
    )

    candidate, rejection = evaluate_objective_support(
        (opportunity,),
        (opportunity.support_key,),
        _planning_config(),
    )

    assert rejection is None
    assert candidate is not None
    # Full successful demand is 50, but mu=20 and T=0.1 impose arrival <= 10
    # at both H lockers.  Continuous OD flow therefore uses lambda=10/50.
    assert 0.0 < candidate.estimated_service < opportunity.successful_service
    assert candidate.estimated_service == pytest.approx(10.0)
    assert candidate.service_fraction == pytest.approx(0.2)
    assert candidate.objective.direct_courier_baseline == pytest.approx(200.0)
    assert candidate.objective.first_mile_rider == pytest.approx(10.0)
    assert candidate.objective.last_mile_rider == pytest.approx(10.0)
    assert candidate.objective.flight == pytest.approx(10.0)
    assert candidate.objective.queue == pytest.approx(2.0)
    assert candidate.base_departure_population == pytest.approx(10.0)
    assert candidate.swap_event_population == pytest.approx(5.0)
    assert candidate.system_energy_supply == pytest.approx(15.0)
    assert candidate.system_energy_use == pytest.approx(3.0)


def test_fractional_service_stops_at_the_system_energy_boundary() -> None:
    opportunity = _opportunity(
        ("h1", "h2"),
        ("o1", "d1"),
        offered_demand=20.0,
    )
    config = replace(
        _planning_config(),
        usable_battery_time_hours=0.25,
    )

    candidate, rejection = evaluate_objective_support(
        (opportunity,),
        (opportunity.support_key,),
        config,
    )

    assert rejection is None
    assert candidate is not None
    # For served amount a, pooled energy is active at
    #   2a/(20-a) + 0.1a = 0.25a,
    # whose positive root is a=20/3.  Every flow-derived proxy uses that same a.
    expected = 20.0 / 3.0
    assert candidate.estimated_service == pytest.approx(expected, abs=1e-8)
    assert candidate.service_fraction == pytest.approx(
        expected / opportunity.successful_service,
        abs=1e-8,
    )
    assert candidate.objective.direct_courier_baseline == pytest.approx(
        20.0 * expected,
        abs=1e-7,
    )
    assert candidate.objective.queue == pytest.approx(1.0, abs=1e-8)
    assert candidate.base_departure_population == pytest.approx(expected, abs=1e-8)
    assert candidate.swap_event_population == pytest.approx(0.0)
    assert candidate.system_energy_use == pytest.approx(
        candidate.system_energy_supply,
        abs=2e-9,
    )


def test_fractional_support_evaluation_is_deterministic() -> None:
    first = _opportunity(
        ("h1", "h2"),
        ("o1", "d1"),
        offered_demand=60.0,
    )
    second = _opportunity(
        ("h1", "h2"),
        ("o2", "d2"),
        offered_demand=40.0,
    )

    forward = evaluate_objective_support(
        (first, second),
        (first.support_key,),
        _planning_config(),
    )
    reverse = evaluate_objective_support(
        (second, first),
        (first.support_key,),
        _planning_config(),
    )

    assert forward == reverse
    candidate, rejection = forward
    assert rejection is None
    assert candidate is not None
    assert (
        0.0
        < candidate.estimated_service
        < math.fsum((first.successful_service, second.successful_service))
    )


def test_energy_proxy_uses_the_f884_system_pool_across_all_h() -> None:
    opportunity = _opportunity(
        ("h1", "h2"),
        ("o1", "d1"),
        outgoing_route_time_hours=(("h1", 0.05), ("h2", 0.05)),
        swap_events_per_unit=1.0,
    )
    config = replace(
        _planning_config(),
        capacity_options=(
            CapacityOption(0, 20.0, 1.0),
            CapacityOption(1, 30.0, 2.0),
        ),
        usable_battery_time_hours=0.1,
    )

    candidate, rejection = evaluate_objective_support(
        (opportunity,),
        (opportunity.support_key,),
        config,
    )

    assert rejection is None
    assert candidate is not None
    # At service 2, the single base departure plus one swap gives 0.4 hours of
    # pooled energy.  Two mu=20 choices need 2*(2/18)+0.2 > 0.4, while one
    # deterministic upgrade gives 2/28+2/18+0.2 <= 0.4.  This is possible only
    # under the system pool; no per-H energy allocation is introduced.
    assert candidate.capacity_levels == (("h1", 1), ("h2", 0))
    assert candidate.objective.queue == pytest.approx(2.0 / 28.0 + 2.0 / 18.0)
    assert candidate.base_departure_population == pytest.approx(2.0)
    assert candidate.swap_event_population == pytest.approx(2.0)
    assert candidate.system_energy_supply == pytest.approx(0.4)
    assert candidate.system_energy_use <= candidate.system_energy_supply + 1e-9

    rejected, reason = evaluate_objective_support(
        (opportunity,),
        (opportunity.support_key,),
        replace(
            config,
            capacity_options=(CapacityOption(0, 20.0, 1.0),),
        ),
    )
    assert rejected is None
    assert reason == "energy_infeasible"


def test_energy_proxy_allows_a_direct_cycle_from_base_battery_energy() -> None:
    direct_cycle = _opportunity(
        ("h1", "h2"),
        ("o1", "d1"),
        base_departures_per_unit=1.0,
        swap_events_per_unit=0.0,
    )
    candidate, rejection = evaluate_objective_support(
        (direct_cycle,),
        (direct_cycle.support_key,),
        replace(_planning_config(), usable_battery_time_hours=0.3),
    )

    assert rejection is None
    assert candidate is not None
    assert candidate.capacity_levels == (("h1", 0), ("h2", 0))
    assert candidate.swap_event_population == pytest.approx(0.0)
    assert candidate.base_departure_population == pytest.approx(2.0)
    assert candidate.system_energy_use <= candidate.system_energy_supply + 1e-9


def test_fleet_proxy_rounds_total_population_once_with_boundary_tolerance() -> None:
    opportunity = _opportunity(("h1", "h2"), ("o1", "d1"))
    noninteger, rejection = evaluate_objective_support(
        (opportunity,),
        (opportunity.support_key,),
        _planning_config(),
    )
    assert rejection is None
    assert noninteger is not None
    # queue + route + idle = 2/18 + 2/18 + 0.2 + 1, hence gamma=2.
    assert noninteger.objective.queue + noninteger.objective.fleet == pytest.approx(2.0)

    boundary = replace(opportunity, route_time_hours=0.3750000000005)
    boundary_config = replace(
        _planning_config(),
        capacity_options=(CapacityOption(0, 18.0, 1.0),),
    )
    candidate, rejection = evaluate_objective_support(
        (boundary,),
        (boundary.support_key,),
        boundary_config,
    )
    assert rejection is None
    assert candidate is not None
    # The exact population is 2 plus a 1e-12 floating error; tolerance must
    # avoid charging a spurious third integer drone.
    assert candidate.objective.queue == pytest.approx(0.25)
    assert candidate.objective.queue + candidate.objective.fleet == pytest.approx(2.0)


def test_proxy_rejects_mixed_reliability_levels_for_one_idle_base() -> None:
    first = _opportunity(("h1", "h2"), ("o1", "d1"))
    second = _opportunity(
        ("h2", "h3"),
        ("o2", "d2"),
        reliability_index=1,
        reliability=0.7,
    )

    candidate, rejection = evaluate_objective_support(
        (first, second),
        (first.support_key, second.support_key),
        _planning_config(),
    )

    assert candidate is None
    assert rejection == "inconsistent_idle_reliability"


def test_lo_proxy_uses_one_display_locker_for_all_od_with_same_destination() -> None:
    opportunities = (
        _opportunity(
            ("h1", "h2"),
            ("o1", "d"),
            display_locker="h2",
            first_mile=1.0,
        ),
        _opportunity(
            ("h1", "h2"),
            ("o2", "d"),
            display_locker="h2",
            first_mile=8.0,
        ),
        _opportunity(
            ("h1", "h3"),
            ("o1", "d"),
            display_locker="h3",
            first_mile=15.0,
        ),
        _opportunity(
            ("h1", "h3"),
            ("o2", "d"),
            display_locker="h3",
            first_mile=1.0,
        ),
    )
    supports = tuple(dict.fromkeys(item.support_key for item in opportunities))

    candidate, rejection = evaluate_objective_support(
        opportunities,
        supports,
        _planning_config(),
    )

    assert rejection is None
    assert candidate is not None
    # h2 wins in aggregate (both origins must share one alpha[d,j]).  Choosing
    # h2 for o1 and h3 for o2 independently would incorrectly give cost 4.
    assert candidate.objective.first_mile_rider == pytest.approx(18.0)
    assert candidate.estimated_service == pytest.approx(4.0)


def test_lo_proxy_does_not_credit_display_locker_blocked_by_open_closer_h() -> None:
    blocked = _opportunity(
        ("h1", "h3"),
        ("o1", "d"),
        display_locker="h3",
        display_blockers=("h2",),
        first_mile=1.0,
    )
    opens_closer = _opportunity(
        ("h2", "h4"),
        ("o2", "other"),
        display_locker="h4",
        first_mile=1.0,
    )

    candidate, rejection = evaluate_objective_support(
        (blocked, opens_closer),
        (blocked.support_key, opens_closer.support_key),
        _planning_config(),
    )

    assert rejection is None
    assert candidate is not None
    assert candidate.estimated_service == pytest.approx(opens_closer.successful_service)


def test_mx_market_proxy_conservatively_does_not_credit_pickup_alpha() -> None:
    wrapper = MLNDModel.__new__(MLNDModel)
    wrapper.mode = "MX"
    wrapper.inst = SimpleNamespace(
        h_od={("o", "d"): 4.0},
        feeder_rider_cost=lambda u, _v: 7.0 if u == "h2" else 2.0,
        direct_rider_cost=lambda _o, _d: 20.0,
        choice_share={("o", "d", "h2"): 0.8},
        locker_pickup_cost=lambda _h: 1.0,
    )

    costs = wrapper._objective_start_market_costs(("o", "d"), ("h1", "h2"))

    assert costs == pytest.approx((4.0, 20.0, 2.0, 7.0, 0.0))


def test_cycle_support_can_select_direct_routes_and_reports_system_coefficients() -> (
    None
):
    wrapper = MLNDModel.__new__(MLNDModel)
    cycle_arcs = (("L1", "h1"), ("h1", "h2"), ("h2", "L1"))
    route_time = {
        ("L1", "h1", DIRECT_ROUTE): 0.04,
        ("h1", "h2", DIRECT_ROUTE): 0.05,
        ("h2", "L1", DIRECT_ROUTE): 0.06,
        ("L1", "h1", "E1"): 0.11,
        ("h1", "h2", "E1"): 0.12,
        ("h2", "L1", "E1"): 0.13,
    }
    route_distance = {
        key: (0.5 if key[2] == DIRECT_ROUTE else 1.0) for key in route_time
    }
    active_legs = {leg for u, v in cycle_arcs for leg in ((u, "E1"), ("E1", v))}
    wrapper.inst = SimpleNamespace(
        L=("L1",),
        H=("h1", "h2"),
        A_F=set(cycle_arcs),
        A_Y=active_legs,
        route_options={arc: (DIRECT_ROUTE, "E1") for arc in cycle_arcs},
        route_time=route_time,
        route_distance=route_distance,
        config=SimpleNamespace(
            usable_battery_time_hours=0.2,
            flight_cost_per_km=1.0,
            drone_cost_per_hour=1.0,
        ),
        facility_cost=lambda _node: 1.0,
        arc_activation_cost={arc: 1.0 for arc in {*cycle_arcs, *active_legs}},
    )

    support = wrapper._objective_start_cycle_support(("h1", "h2"))

    assert support is not None
    assert support[5] == ("h1", "h2")
    assert support[6] == (("h1", 0.05), ("h2", 0.06))
    assert support[7] == pytest.approx(1.0)
    assert support[8] == pytest.approx(0.0)
    assert "E1" not in support[0]

    with pytest.raises(TimeoutError, match="preparation timed out"):
        wrapper._objective_start_cycle_support(("h1", "h2"), deadline=-1.0)


def test_planner_is_deterministic_and_retains_alternate_starters() -> None:
    opportunities = (
        _opportunity(("h1", "h2"), ("o1", "d1")),
        _opportunity(("h2", "h3"), ("o2", "d2")),
        _opportunity(("h3", "h4"), ("o3", "d3")),
        _opportunity(("h4", "h5"), ("o4", "d4"), direct=2.0),
    )
    config = _planning_config(max_candidates=3)
    first = plan_objective_aware_partial_starts(
        opportunities,
        config,
        budget_seconds=1.0,
        clock=lambda: 0.0,
    )
    second = plan_objective_aware_partial_starts(
        tuple(reversed(opportunities)),
        config,
        budget_seconds=1.0,
        clock=lambda: 0.0,
    )

    def signature(plan):
        return [
            (candidate.move, candidate.selected_supports, candidate.open_h)
            for candidate in plan.candidates
        ]

    assert signature(first) == signature(second)
    assert len(first.candidates) == 3
    assert [candidate.move for candidate in first.candidates] == [
        "base",
        "alternate",
        "alternate",
    ]


def test_reciprocal_pair_can_seed_when_each_singleton_is_negative() -> None:
    forward = _opportunity(
        ("h1", "h2"),
        ("o1", "d1"),
        direct=6.8,
    )
    reverse = _opportunity(
        ("h2", "h1"),
        ("o2", "d2"),
        direct=6.8,
    )
    config = _planning_config()
    for opportunity in (forward, reverse):
        singleton, rejection = evaluate_objective_support(
            (forward, reverse),
            (opportunity.support_key,),
            config,
        )
        assert rejection is None
        assert singleton is not None
        assert singleton.score < 0.0

    plan = plan_objective_aware_partial_starts(
        (forward, reverse),
        config,
        budget_seconds=1.0,
        clock=lambda: 0.0,
    )

    assert not plan.timed_out
    assert "no_positive_support" not in plan.rejection_reasons
    assert len(plan.candidates) == 1
    assert plan.candidates[0].move == "base"
    assert plan.candidates[0].selected_supports == tuple(
        sorted((forward.support_key, reverse.support_key))
    )
    assert plan.candidates[0].score > 0.0


def test_budget_is_five_percent_capped_and_timeout_discards_candidates() -> None:
    assert effective_objective_start_budget(100.0) == pytest.approx(5.0)
    assert effective_objective_start_budget(100.0, 2.0) == pytest.approx(2.0)
    assert effective_objective_start_budget(100.0, 20.0) == pytest.approx(5.0)

    clock_values = iter((0.0, 1.0, 1.0, 1.0))
    plan = plan_objective_aware_partial_starts(
        (_opportunity(("h1", "h2"), ("o1", "d1")),),
        _planning_config(),
        budget_seconds=0.5,
        clock=lambda: next(clock_values, 1.0),
    )
    assert plan.timed_out
    assert plan.candidates == ()
    assert "budget_exhausted" in plan.rejection_reasons


def test_timeout_preserves_completed_singleton_count(monkeypatch) -> None:
    forward = _opportunity(("h1", "h2"), ("o1", "d1"))
    reverse = _opportunity(("h2", "h1"), ("o2", "d2"))
    original_evaluate = primal_heuristic.evaluate_objective_support

    def timeout_on_reciprocal_pair(*args, **kwargs):
        if len(args[1]) == 2:
            raise primal_heuristic._ObjectiveStartTimeout
        return original_evaluate(*args, **kwargs)

    monkeypatch.setattr(
        primal_heuristic,
        "evaluate_objective_support",
        timeout_on_reciprocal_pair,
    )
    plan = plan_objective_aware_partial_starts(
        (forward, reverse),
        _planning_config(),
        budget_seconds=1.0,
        clock=lambda: 0.0,
    )

    assert plan.timed_out
    assert plan.candidates == ()
    assert plan.singleton_support_count == 2
    assert plan.evaluated_support_count == 2


@pytest.mark.parametrize(
    ("config", "reason"),
    (
        (replace(_planning_config(), min_open_h=3), "material_open_h"),
        (replace(_planning_config(), min_service=3.0), "material_service"),
        (replace(_planning_config(), min_savings=1_000.0), "material_savings"),
        (
            replace(_planning_config(), candidate_lockers=("h1", "h2")),
            "all_candidate_h_open",
        ),
    ),
)
def test_material_network_gates_reject_weak_or_all_open_supports(
    config: ObjectiveAwarePlanningConfig,
    reason: str,
) -> None:
    plan = plan_objective_aware_partial_starts(
        (_opportunity(("h1", "h2"), ("o1", "d1")),),
        config,
        budget_seconds=1.0,
        clock=lambda: 0.0,
    )
    assert plan.candidates == ()
    assert reason in plan.rejection_reasons


def test_mx_candidate_assignment_leaves_alpha_and_other_variables_undefined() -> None:
    x = {name: FakeVar(f"x[{name}]") for name in ("h1", "h2", "h3")}
    z = {
        (name, level): FakeVar(f"z[{name},{level}]")
        for name in ("h1", "h2", "h3")
        for level in (0, 1)
    }
    y = {arc: FakeVar(f"y[{arc[0]},{arc[1]}]") for arc in (("h1", "h2"), ("h2", "h3"))}
    alpha = {("d1", "h2"): FakeVar("alpha[d1,h2]")}
    variables = {"x": x, "z": z, "y": y, "alpha": alpha}
    candidate = ObjectiveAwareSupportCandidate(
        candidate_id="c1",
        move="base",
        selected_supports=(("h1", "h2", 0),),
        open_h=("h1", "h2"),
        open_facilities=("h1", "h2"),
        active_connections=(("h1", "h2"),),
        capacity_levels=(("h1", 0), ("h2", 1)),
        forced_closed_h=(),
        forced_off_connections=(),
        estimated_service=2.0,
        objective=ApproximateObjectiveBreakdown(direct_courier_baseline=10.0),
    )
    assignment = build_candidate_partial_assignment(candidate, variables, mode="MX")
    assert alpha[("d1", "h2")] not in assignment
    assert x["h3"] not in assignment
    assert y[("h2", "h3")] not in assignment
    assert assignment[z[("h1", 0)]] == 1.0
    assert assignment[z[("h1", 1)]] == 0.0
    assert assignment[z[("h2", 1)]] == 1.0

    model_variables = [*x.values(), *z.values(), *y.values(), *alpha.values()]
    vector = build_partial_start_vector(model_variables, assignment)
    assert vector[model_variables.index(alpha[("d1", "h2")])] == GRB.UNDEFINED
    assert vector[model_variables.index(x["h3"])] == GRB.UNDEFINED


def test_partial_start_installer_uses_undefined_and_never_optimizes() -> None:
    first, second, third = (FakeVar(name) for name in ("first", "second", "third"))
    model = MockStartModel([first, second, third])
    assigned_counts = install_partial_mip_starts(
        model,
        ({first: 1.0}, {second: 0.0}),
    )
    assert model.NumStart == 2
    assert model.Params.StartNumber == 0
    assert assigned_counts == (1, 1)
    assert model.sparse_values == {0: {first: 1.0}, 1: {second: 0.0}}
    assert not hasattr(model, "optimize")


def test_all_courier_safety_start_is_partial() -> None:
    structural = {
        name: {name: FakeVar(name)} for name in ("x", "z", "y", "delta", "alpha")
    }
    gamma = FakeVar("gamma")
    structural["gamma"] = gamma
    assignment = build_all_courier_partial_assignment(structural)
    assert set(assignment.values()) == {0.0}
    assert gamma not in assignment


def test_no_network_candidate_falls_back_to_zero_starts() -> None:
    wrapper = MLNDModel.__new__(MLNDModel)
    wrapper.model = SimpleNamespace(NumStart=4)
    wrapper.vars = {}
    wrapper.mode = "MX"
    wrapper.courier_only_baseline = 100.0
    wrapper.inst = SimpleNamespace(
        H=["h1", "h2", "h3"],
        total_demand=10.0,
        config=SimpleNamespace(
            locker_service_rates=(20.0,),
            locker_capacity_costs=(1.0,),
            max_locker_time_hours=0.1,
            usable_battery_time_hours=1.0,
            drone_cost_per_hour=1.0,
        ),
    )
    wrapper._build_objective_start_opportunities = lambda **_: (
        [],
        {
            "eligible_corridor_count": 0,
            "shortlisted_corridor_count": 0,
            "supported_cycle_count": 0,
        },
    )
    wrapper._clear_mip_starts = lambda: setattr(wrapper.model, "NumStart", 0)
    wrapper._construct_objective_partial_starts(
        1.0,
        SolverConfig(
            time_limit=20.0,
            use_objective_partial_start=True,
            objective_partial_start_max_candidates=3,
        ),
    )
    telemetry = wrapper.objective_partial_start_telemetry
    assert wrapper.model.NumStart == 0
    assert telemetry["used"] is False
    assert telemetry["installed_start_count"] == 0
    assert telemetry["safety_start_count"] == 0
    assert telemetry["fallback_reason"] == "no_reliable_network_candidate"


@pytest.mark.parametrize("mode", ("MX", "HD"))
def test_objective_partial_start_accepts_only_validated_modes(mode: str) -> None:
    wrapper = MLNDModel.__new__(MLNDModel)
    wrapper.model = SimpleNamespace(NumStart=1)
    wrapper.mode = mode
    wrapper._clear_mip_starts = lambda: setattr(wrapper.model, "NumStart", 0)

    wrapper._construct_objective_partial_starts(
        0.0,
        SolverConfig(time_limit=20.0, use_objective_partial_start=True),
    )

    assert wrapper.model.NumStart == 0
    assert wrapper.objective_partial_start_telemetry["proxy_only"] is True
    assert wrapper.objective_partial_start_telemetry["timed_out"] is True


def test_objective_partial_start_explicitly_rejects_lo() -> None:
    wrapper = MLNDModel.__new__(MLNDModel)
    wrapper.model = SimpleNamespace(NumStart=0)
    wrapper.mode = "LO"

    with pytest.raises(
        ValueError,
        match="supported only for modes HD, MX; got 'LO'",
    ):
        wrapper._construct_objective_partial_starts(
            0.0,
            SolverConfig(time_limit=20.0, use_objective_partial_start=True),
        )

    assert not hasattr(wrapper.model, "optimize")


def test_candidate_telemetry_labels_fraction_as_proxy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = ObjectiveAwareSupportCandidate(
        candidate_id="candidate",
        move="base",
        selected_supports=(("h1", "h2", 0),),
        open_h=("h1", "h2"),
        open_facilities=("h1", "h2"),
        active_connections=(("h1", "h2"),),
        capacity_levels=(("h1", 0), ("h2", 0)),
        forced_closed_h=(),
        forced_off_connections=(),
        estimated_service=2.0,
        objective=ApproximateObjectiveBreakdown(direct_courier_baseline=10.0),
        service_fraction=0.25,
    )
    plan = SimpleNamespace(
        opportunity_count=1,
        singleton_support_count=1,
        evaluated_support_count=1,
        candidates=(candidate,),
        rejection_reasons=(),
        timed_out=False,
    )
    monkeypatch.setattr(
        gurobi_model_module,
        "plan_objective_aware_partial_starts",
        lambda *_args, **_kwargs: plan,
    )
    monkeypatch.setattr(
        gurobi_model_module,
        "build_candidate_partial_assignment",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        gurobi_model_module,
        "build_all_courier_partial_assignment",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        gurobi_model_module,
        "install_partial_mip_starts",
        lambda *_args, **_kwargs: (0, 0),
    )

    wrapper = MLNDModel.__new__(MLNDModel)
    wrapper.model = SimpleNamespace(NumStart=0)
    wrapper.vars = {}
    wrapper.mode = "MX"
    wrapper.courier_only_baseline = 100.0
    wrapper.inst = SimpleNamespace(
        H=["h1", "h2", "h3"],
        total_demand=10.0,
        config=SimpleNamespace(
            locker_service_rates=(20.0,),
            locker_capacity_costs=(1.0,),
            max_locker_time_hours=0.1,
            usable_battery_time_hours=1.0,
            drone_cost_per_hour=1.0,
        ),
    )
    wrapper._build_objective_start_opportunities = lambda **_: ([], {})
    wrapper._clear_mip_starts = lambda: setattr(wrapper.model, "NumStart", 0)

    wrapper._construct_objective_partial_starts(
        1.0,
        SolverConfig(time_limit=20.0, use_objective_partial_start=True),
    )

    telemetry = wrapper.objective_partial_start_telemetry
    support = telemetry["candidate_supports"][0]
    assert telemetry["proxy_only"] is True
    assert support["proxy_only"] is True
    assert support["estimated_service_fraction"] == pytest.approx(0.25)
    assert support["fraction_basis_service"] == pytest.approx(8.0)
    assert "service_fraction" not in support
    assert not hasattr(wrapper.model, "optimize")


def test_construction_timeout_is_audited_as_budget_exhaustion() -> None:
    wrapper = MLNDModel.__new__(MLNDModel)
    wrapper.model = SimpleNamespace(NumStart=2)
    wrapper.vars = {}
    wrapper.mode = "MX"
    wrapper.courier_only_baseline = 100.0
    wrapper.inst = SimpleNamespace(H=["h1", "h2", "h3"], total_demand=10.0)

    def timeout(**_):
        raise TimeoutError

    wrapper._build_objective_start_opportunities = timeout
    wrapper._clear_mip_starts = lambda: setattr(wrapper.model, "NumStart", 0)
    wrapper._construct_objective_partial_starts(
        1.0,
        SolverConfig(time_limit=20.0, use_objective_partial_start=True),
    )
    telemetry = wrapper.objective_partial_start_telemetry
    assert wrapper.model.NumStart == 0
    assert telemetry["timed_out"] is True
    assert telemetry["fallback_reason"] == "budget_exhausted"
    assert telemetry["rejection_reasons"] == ["budget_exhausted"]


def test_partial_start_telemetry_is_json_safe_in_both_reporting_paths() -> None:
    raw = {
        "enabled": True,
        "used": True,
        "candidate_supports": [{"move": "base", "score": 12.5}],
    }
    detached = _json_safe_mapping(raw)
    assert detached == raw
    assert detached is not raw

    backend = SimpleNamespace(
        Status=GRB.LOADED,
        SolCount=0,
        ObjBound=0.0,
        Runtime=0.0,
        NodeCount=0.0,
    )
    wrapper = SimpleNamespace(
        model=backend,
        objective_partial_start_telemetry=raw,
    )
    metrics = collect_solver_metrics(wrapper)
    assert metrics["objective_partial_start"] == raw


@pytest.mark.parametrize(
    "kwargs",
    (
        {"use_objective_partial_start": 1},
        {"objective_partial_start_seconds": -1.0},
        {"objective_partial_start_max_candidates": 0},
        {"objective_partial_start_max_candidates": 17},
    ),
)
def test_solver_config_rejects_invalid_partial_start_controls(kwargs) -> None:
    with pytest.raises(ValueError):
        SolverConfig(**kwargs).validate()


def test_development_cli_is_default_off_and_exposes_bounded_controls() -> None:
    parser = build_parser()
    default = parser.parse_args(["solve", "--instance-dir", "/tmp/example"])
    assert default.dev_objective_partial_start is False
    assert default.objective_partial_start_seconds is None
    assert default.objective_partial_start_max_candidates == 3

    enabled = parser.parse_args(
        [
            "solve",
            "--instance-dir",
            "/tmp/example",
            "--dev-objective-partial-start",
            "--objective-partial-start-seconds",
            "1.5",
            "--objective-partial-start-max-candidates",
            "4",
        ]
    )
    assert enabled.dev_objective_partial_start is True
    assert enabled.objective_partial_start_seconds == pytest.approx(1.5)
    assert enabled.objective_partial_start_max_candidates == 4


@pytest.mark.parametrize("modes", ("LO", "HD,LO", "HD,MX,LO"))
def test_cli_rejects_unsupported_partial_start_modes_before_loading_or_solving(
    tmp_path: Path, modes: str
) -> None:
    instance = tmp_path / "unread_instance"
    output = tmp_path / "results"
    args = build_parser().parse_args(
        [
            "solve",
            "--instance-dir",
            str(instance),
            "--results-dir",
            str(output),
            "--modes",
            modes,
            "--dev-objective-partial-start",
        ]
    )
    with pytest.raises(ValueError, match="only HD/MX"):
        _solve(args, str(instance))
    assert not output.exists()
