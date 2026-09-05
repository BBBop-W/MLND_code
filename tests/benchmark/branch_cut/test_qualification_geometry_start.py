from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from src.benchmark.branch_cut.qualification_geometry_start import (
    GEOMETRY_START_PLAN_VERSION,
    GUROBI_MAX_START_NODE_LIMIT,
    GeometryStartAttemptAudit,
    GeometryStartInfeasibilityError,
    GeometryStartInstallAudit,
    GeometryStartPlanAudit,
    GeometryStartPlanningError,
    GeometryStartPolicy,
    build_geometry_start_plan,
)
from src.data_processor.instance import DIRECT_ROUTE


def _fake_wrapper(*, mode: str = "MX") -> SimpleNamespace:
    lockers = [f"H{index}" for index in range(4)]
    base = "B0"
    swaps = ("E_slow", "E_fast")
    corridors = (("H1", "H0"), ("H1", "H2"), ("H2", "H3"))
    physical_arcs = {
        *((base, origin) for origin, _destination in corridors),
        *corridors,
        *((destination, base) for _origin, destination in corridors),
    }
    route_options = {
        arc: (DIRECT_ROUTE, *swaps) for arc in sorted(physical_arcs)
    }
    route_time = {}
    route_distance = {}
    for u, v in physical_arcs:
        route_time[u, v, DIRECT_ROUTE] = 0.2
        route_distance[u, v, DIRECT_ROUTE] = 1.0
        route_time[u, v, "E_slow"] = 0.5
        route_distance[u, v, "E_slow"] = 2.0
        route_time[u, v, "E_fast"] = 0.4
        route_distance[u, v, "E_fast"] = 3.0
    activated = set(physical_arcs)
    for u, v in physical_arcs:
        for option in swaps:
            activated.add((u, option))
            activated.add((option, v))

    markets = {(f"O{index}", f"D{index}"): 100.0 for index in range(3)}
    chi_by_corridor = {}
    chi_by_level = {}
    for index, corridor in enumerate(corridors):
        key = (f"O{index}", f"D{index}", *corridor, 1)
        chi_by_corridor[corridor] = [key]
        chi_by_level[(*corridor, 1)] = [key]
    instance = SimpleNamespace(
        L=[base],
        H=lockers,
        E=list(swaps),
        coords={
            base: (0.0, -1.0),
            "H0": (0.0, 0.0),
            "H1": (1.0, 0.0),
            "H2": (2.0, 0.0),
            "H3": (3.0, 0.0),
            "E_slow": (0.0, 1.0),
            "E_fast": (0.0, 2.0),
        },
        A_F=physical_arcs,
        A_LH={(base, origin) for origin, _destination in corridors},
        A_Y=activated,
        route_options=route_options,
        route_time=route_time,
        route_distance=route_distance,
        M_psi={arc: 100.0 for arc in physical_arcs},
        M_xi={locker: 100.0 for locker in lockers},
        h_od=markets,
        config=SimpleNamespace(
            usable_battery_time_hours=1.0,
            availability_levels=(0.5, 0.8),
            locker_service_rates=(10.0, 100.0),
            max_locker_time_hours=1.0,
        ),
    )

    def forbidden_cost(*_args: object) -> float:
        raise AssertionError("geometry planning may not read a cost")

    instance.direct_rider_cost = forbidden_cost
    instance.feeder_rider_cost = forbidden_cost
    instance.facility_cost = forbidden_cost
    return SimpleNamespace(
        mode=mode,
        inst=instance,
        chi_by_corridor=chi_by_corridor,
        chi_by_corridor_level=chi_by_level,
        w_keys=tuple((base, *corridor, 1) for corridor in corridors),
        M_offer={corridor: 100.0 for corridor in corridors},
    )


def test_frozen_policy_serializes_registered_schedules() -> None:
    policy = GeometryStartPolicy(enabled=True, per_attempt_seconds=7.5)

    assert policy.schedule_for("small", 23) == (12, 8, 15)
    assert policy.schedule_for("medium", 51) == (26, 17, 34)
    assert policy.schedule_for("large", 90) == (55, 50, 60)
    assert GeometryStartPolicy.from_payload(policy.to_payload()) == policy
    with pytest.raises(ValueError, match="pre-registered"):
        replace(policy, small_support_schedule=(8, 12, 15))
    with pytest.raises(GeometryStartPlanningError, match="candidate-H count"):
        policy.schedule_for("large", 89)


@pytest.mark.parametrize("bad_seconds", [True, "5", None, float("inf")])
def test_policy_rejects_non_numeric_or_nonfinite_attempt_budget(
    bad_seconds: object,
) -> None:
    payload = GeometryStartPolicy().to_payload()
    payload["per_attempt_seconds"] = bad_seconds

    with pytest.raises(ValueError, match="per_attempt_seconds"):
        GeometryStartPolicy.from_payload(payload)


@pytest.mark.parametrize(
    "bad_limit",
    [True, -1, GUROBI_MAX_START_NODE_LIMIT + 1],
)
def test_policy_rejects_start_node_limit_outside_gurobi_domain(
    bad_limit: object,
) -> None:
    payload = GeometryStartPolicy().to_payload()
    payload["start_node_limit"] = bad_limit

    with pytest.raises(ValueError, match="start_node_limit"):
        GeometryStartPolicy.from_payload(payload)


def test_plan_is_cost_blind_connected_and_model_consistent() -> None:
    plan = build_geometry_start_plan(
        _fake_wrapper(),
        scale="small",
        target_support_size=4,
        required_service_per_hour=3.0,
        active_h_tolerance_per_hour=0.01,
    )

    assert plan.plan_version == GEOMETRY_START_PLAN_VERSION
    assert plan.support_order == ("H1", "H0", "H2", "H3")
    assert len(plan.cycles) == 3
    assert set(plan.support_order) == {
        node for cycle in plan.cycles for node in (cycle.origin_h, cycle.destination_h)
    }
    # The v3 planner admits direct routes and chooses the cost-blind
    # minimum-time option.  Base departures provide the initial batteries.
    assert all(
        option == DIRECT_ROUTE
        for cycle in plan.cycles
        for option in cycle.route_options
    )
    metrics = dict(plan.precheck_metrics)
    assert metrics["swap_event_flow"] == pytest.approx(0.0)
    assert metrics["base_battery_departure_flow"] > 0.0
    assert metrics["system_energy_slack"] >= 0.0
    assert plan.availability_index == 1
    assert plan.availability_probability == pytest.approx(0.8)
    assert plan.offered_flow_per_cycle * 0.8 == pytest.approx(plan.flow_per_cycle)
    assert sum(item[-1] for item in plan.chi_flows) == pytest.approx(
        len(plan.cycles) * plan.offered_flow_per_cycle
    )
    assert all(item[3] == 1 for item in plan.w_flows)
    assert plan.delta_levels == (("B0", 1),)
    assert all(choice.capacity_index == 1 for choice in plan.capacity_choices)
    assert len(plan.xi_values) == len(plan.support_order)
    assert plan.gamma_value >= 1
    assert set(dict(plan.family_start_counts)) == {"x", "y", "z", "psi", "varphi"}
    assert GeometryStartPlanAudit.from_payload(
        GeometryStartPlanAudit.from_plan(plan).to_payload()
    ) == GeometryStartPlanAudit.from_plan(plan)


def test_planner_is_deterministic_and_hashes_aggregated_plan() -> None:
    kwargs = {
        "scale": "small",
        "target_support_size": 4,
        "required_service_per_hour": 3.0,
        "active_h_tolerance_per_hour": 0.01,
    }
    first = build_geometry_start_plan(_fake_wrapper(), **kwargs)
    second = build_geometry_start_plan(_fake_wrapper(), **kwargs)

    assert first == second
    assert first.plan_sha256 == second.plan_sha256
    source_flow = sum(
        flow
        for u, v, flow in first.physical_arc_flows
        if (u, v) == ("B0", "H1")
    )
    assert source_flow == pytest.approx(2.0 * first.flow_per_cycle)


def test_lo_and_unsafe_big_m_plan_fail_closed() -> None:
    with pytest.raises(
        GeometryStartPlanningError,
        match="fail-closed for LO",
    ) as lo_error:
        build_geometry_start_plan(
            _fake_wrapper(mode="LO"),
            scale="small",
            target_support_size=3,
            required_service_per_hour=1.0,
            active_h_tolerance_per_hour=0.01,
        )
    assert not isinstance(lo_error.value, GeometryStartInfeasibilityError)

    unsafe = _fake_wrapper()
    unsafe.M_offer["H1", "H2"] = 0.001
    with pytest.raises(
        GeometryStartInfeasibilityError,
        match="violates M_offer",
    ):
        build_geometry_start_plan(
            unsafe,
            scale="small",
            target_support_size=4,
            required_service_per_hour=3.0,
            active_h_tolerance_per_hour=0.01,
        )


def test_planner_internal_identity_failure_is_not_candidate_infeasibility() -> None:
    inconsistent = _fake_wrapper()
    inconsistent.inst.A_Y.remove(("B0", "H1"))

    with pytest.raises(
        GeometryStartPlanningError,
        match="planned y arc is absent",
    ) as caught:
        build_geometry_start_plan(
            inconsistent,
            scale="small",
            target_support_size=4,
            required_service_per_hour=3.0,
            active_h_tolerance_per_hour=0.01,
        )

    assert not isinstance(caught.value, GeometryStartInfeasibilityError)


def test_attempt_audits_have_strict_canonical_roundtrip_types() -> None:
    plan = build_geometry_start_plan(
        _fake_wrapper(),
        scale="small",
        target_support_size=4,
        required_service_per_hour=3.0,
        active_h_tolerance_per_hour=0.01,
    )
    install = GeometryStartInstallAudit(
        reset_performed=True,
        undefined_variable_count=12,
        family_start_counts=plan.family_start_counts,
        fingerprint_before=17,
        fingerprint_with_start=19,
        fingerprint_after=17,
        fingerprint_unchanged=True,
        constraint_counts_before=(10, 4, 0),
        constraint_counts_after=(10, 4, 0),
        hard_constraints_added=False,
    )
    attempt = GeometryStartAttemptAudit(
        plan=GeometryStartPlanAudit.from_plan(plan),
        per_attempt_seconds=5.0,
        start_node_limit=42,
        install=install,
        solve_status="TIME_LIMIT",
        solve_seconds=4.9,
        full_witness_found=True,
    )

    assert GeometryStartAttemptAudit.from_payload(attempt.to_payload()) == attempt

    malformed = attempt.to_payload()
    malformed["start_node_limit"] = True
    with pytest.raises(ValueError, match="StartNodeLimit"):
        GeometryStartAttemptAudit.from_payload(malformed)

    inconsistent_install = install.to_payload()
    inconsistent_install["hard_constraints_added"] = True
    with pytest.raises(ValueError, match="hard-constraint audit"):
        GeometryStartInstallAudit.from_payload(inconsistent_install)


def test_attempt_state_machine_rejects_tampered_witness_and_cleanup_states() -> None:
    plan = build_geometry_start_plan(
        _fake_wrapper(),
        scale="small",
        target_support_size=4,
        required_service_per_hour=3.0,
        active_h_tolerance_per_hour=0.01,
    )
    plan_audit = GeometryStartPlanAudit.from_plan(plan)
    safe_install = GeometryStartInstallAudit(
        reset_performed=True,
        undefined_variable_count=12,
        family_start_counts=plan.family_start_counts,
        fingerprint_before=17,
        fingerprint_with_start=19,
        fingerprint_after=17,
        fingerprint_unchanged=True,
        constraint_counts_before=(10, 4, 0),
        constraint_counts_after=(10, 4, 0),
        hard_constraints_added=False,
    )
    successful = GeometryStartAttemptAudit(
        plan=plan_audit,
        per_attempt_seconds=5.0,
        start_node_limit=42,
        install=safe_install,
        solve_status="TIME_LIMIT",
        solve_seconds=4.9,
        full_witness_found=True,
    )

    with pytest.raises(ValueError, match="safe install"):
        replace(
            successful,
            install=replace(
                safe_install,
                fingerprint_after=18,
                fingerprint_unchanged=False,
            ),
        )
    with pytest.raises(ValueError, match="cannot contain an error"):
        replace(successful, error="tampered")
    with pytest.raises(ValueError, match="cannot be cleaned up"):
        replace(successful, cleanup_performed=True, cleanup_succeeded=True)
    with pytest.raises(ValueError, match="requires cleanup"):
        replace(successful, full_witness_found=False)
    with pytest.raises(ValueError, match="prechecked plan"):
        replace(
            successful,
            plan=GeometryStartPlanAudit.planning_failure(
                plan_version=GEOMETRY_START_PLAN_VERSION,
                target_support_size=4,
            ),
            full_witness_found=False,
            cleanup_performed=True,
            cleanup_succeeded=True,
            error="tampered",
        )
    with pytest.raises(ValueError, match="requires an error audit"):
        GeometryStartAttemptAudit(
            plan=plan_audit,
            per_attempt_seconds=5.0,
            start_node_limit=42,
        )
    with pytest.raises(ValueError, match="StartNodeLimit"):
        replace(successful, start_node_limit=GUROBI_MAX_START_NODE_LIMIT + 1)
    with pytest.raises(ValueError, match="budget"):
        replace(successful, per_attempt_seconds=True)
    with pytest.raises(ValueError, match="budget"):
        replace(successful, per_attempt_seconds="5")


def test_system_energy_precheck_includes_queue_population_and_all_routes() -> None:
    wrapper = _fake_wrapper()
    # Direct cycles use 0.6 battery-hours per unit of cycle flow and receive
    # one full battery per base departure.  This allowance covers neither all
    # route population nor the strictly positive queue populations.
    wrapper.inst.config.usable_battery_time_hours = 0.59

    with pytest.raises(
        GeometryStartPlanningError,
        match="system energy precheck",
    ):
        build_geometry_start_plan(
            wrapper,
            scale="small",
            target_support_size=4,
            required_service_per_hour=3.0,
            active_h_tolerance_per_hour=0.01,
        )


def test_system_energy_is_pooled_across_h_and_base_departures_supply_it() -> None:
    wrapper = _fake_wrapper()
    # Make one H departure locally consume more than one battery while the
    # three direct cycles remain feasible in the exact system-wide balance.
    for key in list(wrapper.inst.route_time):
        u, _v, option = key
        if option == DIRECT_ROUTE:
            wrapper.inst.route_time[key] = 0.7 if u == "H1" else 0.05
    for arc in wrapper.inst.A_F:
        wrapper.inst.route_options[arc] = (DIRECT_ROUTE,)

    plan = build_geometry_start_plan(
        wrapper,
        scale="small",
        target_support_size=4,
        required_service_per_hour=3.0,
        active_h_tolerance_per_hour=0.01,
    )

    metrics = dict(plan.precheck_metrics)
    assert metrics["swap_event_flow"] == pytest.approx(0.0)
    assert metrics["system_energy_supply"] == pytest.approx(
        metrics["base_battery_departure_flow"]
    )
    assert metrics["system_energy_use"] == pytest.approx(
        metrics["queue_population_total"] + metrics["route_population_total"]
    )
    assert metrics["system_energy_slack"] >= 0.0


def test_each_selected_swap_route_contributes_one_system_supply_event() -> None:
    wrapper = _fake_wrapper()
    # Remove direct alternatives so every selected physical arc uses exactly
    # one swap option.  The energy audit must count one event per route flow,
    # not one event per activated swap leg.
    for arc in wrapper.inst.A_F:
        wrapper.inst.route_options[arc] = ("E_fast",)

    plan = build_geometry_start_plan(
        wrapper,
        scale="small",
        target_support_size=4,
        required_service_per_hour=3.0,
        active_h_tolerance_per_hour=0.01,
    )

    metrics = dict(plan.precheck_metrics)
    total_route_flow = sum(item[-1] for item in plan.route_flows)
    assert metrics["swap_event_flow"] == pytest.approx(total_route_flow)
