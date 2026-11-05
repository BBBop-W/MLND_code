"""Independent-task audit regressions; no solver or production artifact is modified."""
from copy import deepcopy
from decimal import Decimal
import math

import pytest

from src.experiments import task_cost_audit as task
from src.experiments import baseline_cut_audit_20261029 as audit


@pytest.fixture
def independent_case(tmp_path):
    csvs = {
        "coefficients/od_costs.csv": "od_id,ground_cost\nr,10\n",
        "od_demand.csv": "od_id,lambda_per_hour\nr,3\n",
        "coefficients/locker_levels.csv": "hub_id,level_id,occupancy_limit\nh0,0,10\nh1,0,10\n",
        "coefficients/links.csv": "from_site,to_site,allowed_links\nh0,h1,1\n",
    }
    for name, content in csvs.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    model = dict(values=dict(n_od=1, n_l=2, n_h=2, max_orders=2, fleet_cost=1),
                 ids=dict(ods=["r"], hubs=["h0", "h1"], levels=[0], flight_nodes=["l0", "l1", "h0", "h1"]))
    demand = dict(single_rates=[1], pair_rates=[[1]], head_rates=[2], assignment_scope="task", max_task_orders=2)
    catalogue = dict(missions=[
        dict(flow=.5, path=dict(id=0, first_od=0, second_od=None, b=[0, 0], cost=1, time=0,
                               slot_times=[[0, .1]], arc_counts=[[2, 3, 1]])),
        dict(flow=.25, path=dict(id=1, first_od=0, second_od=0, b=[1, 1], cost=2, time=0,
                                slot_times=[[0, .2]], arc_counts=[[2, 3, 2]])),
    ], relocations=[], segments=[])
    constants = audit.input_constants(model, demand, catalogue, tmp_path)
    solution = dict(assignment_scope="task", assignment=[[1, 0], [0, 1]], rho=[.5, .25], flow=[.5, .25],
                    airline=[[int((u, v) == (2, 3)) for v in range(4)] for u in range(4)])
    return model, demand, catalogue, constants, solution, tmp_path


def test_task_minima_support_and_arc_maxima_are_independent(independent_case):
    model, demand, catalogue, c, solution, instance = independent_case
    assert (c["R"], c["Q"], c["G"]) == (1, 2, 4)
    assert c["saving"] == [9, 0, 0, 18]
    assert c["supported"] == [True, False, False, True]
    assert c["task_upper"][11] == 3  # max_l single + max_l pair, including same-OD multiplicity.
    old = audit.input_constants(model, demand, catalogue, instance, assignment_scope="origin")
    assert old["Q"] == 1 and old["saving"] == [0, 0]
    assert old["task_upper"][11] == 2  # Historical coupled artifacts retain their old certificate domain.
    errors = []
    checked = task.flow_checks(catalogue, solution, demand, c["task_upper"], c, errors, "task")
    assert not errors and checked["maximum_task_flow_identity_error"] == 0
    wrong = deepcopy(solution)
    wrong["assignment"][1] = [1, 0]
    errors = []
    task.flow_checks(catalogue, wrong, demand, c["task_upper"], c, errors, "wrong")
    assert any("task flow identity" in message for message in errors)


def profile(c, priced=False):
    result = dict(profile_id=0, kind="resource_price_epi" if priced else "nonnegative_savings_epi",
        assignment_scope="task", heads=c["Q"], bases=c["L"], fleet_cost=1, constant=30,
        complete_domain=True, catalogue_complete=True, support_complete=True, service_paths_scanned=2,
        empty_paths_scanned=0, negative_empty_paths=[], signed_savings=[9, 0, 0, 18],
        unsupported=[False] * 4, lower_coefficients=[0] * c["design_count"], prices=None)
    if priced:
        result.update(unsupported=[False, True, True, False], rounding_relative_guard=1e-12,
            positive_service_paths_scanned=2, missing_branches=[[0, 0, 0], [0, 1, -1]],
            total_demand=3, airline_upper=list(map(float, c["task_upper"])),
            prices=dict(locker=[0, 0], airline=[0] * 16, balance=[0, 0], branch_gates=[]))
    return result


@pytest.mark.parametrize("priced", [False, True])
def test_queue_profiles_use_task_groups_and_physical_branch_keys(independent_case, priced):
    _, demand, catalogue, c, _, _ = independent_case
    p = profile(c, priced)
    assert audit.validate_profile(p, c, demand, catalogue)["group_checks"] == 4
    p["assignment_scope"] = "origin"
    with pytest.raises(ValueError, match="assignment scope"):
        audit.validate_profile(p, c, demand, catalogue)


def test_queue_chain_replays_over_all_task_variables(independent_case):
    _, _, _, c, _, _ = independent_case
    p, start, count = profile(c), c["assignment"], c["design_count"]
    design = [0.] * count
    design[start:start + 4] = [.7, 0, 0, .6]
    coefficients, bases = [[count, 1.]], []
    for base, weights in enumerate(([9, 0], [0, 18])):
        transformed = [design[start + q * 2 + base] for q in range(2)]
        order = sorted(range(2), key=lambda q: -transformed[q])
        deltas, prefixes, cumulative, previous = [0., 0.], [0.], 0, Decimal(0)
        for q in order:
            cumulative += weights[q]
            value = audit.conjugate(Decimal(cumulative), Decimal(1))
            deltas[q] = float(previous - value)
            if deltas[q]:
                coefficients.append([start + q * 2 + base, -deltas[q]])
            prefixes.append(float(value))
            previous = value
        bases.append(dict(base=base, necessary_gates=[False, False], complemented=[False, False],
            shift=0, weights=weights, transformed_point=transformed, order=order,
            prefix_savings=prefixes, deltas=deltas))
    margin = 31e-8
    rhs = 30 - margin
    value = rhs - math.fsum(a * design[j] for j, a in coefficients if j != count)
    certificate = dict(kind="queue_polymatroid", complete_domain=True, global_integer_valid=True,
        direct_global_bound_update=False, ordinary_auxiliary_optimization_calls=0, profile=p,
        separation_design=design, separation_theta=0, bases=bases, numerical_margin=margin,
        conversion_error=0, arithmetic_error=0, increment_error_allowance=0,
        intercept_correction=margin, value_at_separation_point=value)
    row = dict(certificate=certificate, rhs=rhs, coefficients=coefficients, violation=value, stage="test")
    checked = audit.replay_queue_row(row, c)
    assert checked["recomputed_violation"] == pytest.approx(value)
    assert not checked["separation_point_binary"]


def test_task_support_canonicalization_does_not_drop_supported_siblings(independent_case):
    model, demand, catalogue, c, solution, _ = independent_case
    solution = deepcopy(solution)
    solution.update(assignment=[[1, 0], [1, 0]], rho=[0, 0], flow=[0, 0])
    point = [0] * c["design_count"]
    point[c["assignment"]] = point[c["assignment"] + 2] = 1
    normalized, checked = audit.canonicalize_support_point(model, demand, catalogue, solution, ("fixture", point, 30))
    assert checked["passed"] and len(checked["changed_assignments"]) == 1
    assert checked["changed_assignments"][0]["entity"] == 1
    assert normalized[1][c["assignment"]] == 1 and normalized[1][c["assignment"] + 2] == 0


def test_joint_queue_replays_vertices_and_rejects_forged_plane(independent_case):
    _, _, _, c, _, _ = independent_case
    p, count, start = profile(c), c["design_count"], c["assignment"]
    first, second = audit.conjugate(Decimal(9), Decimal(1)), audit.conjugate(Decimal(18), Decimal(1))
    affine = [0, -float(first), 0, 0, -float(second)]
    vertices = [dict(states=[a, b], queue_value=float(-(first if a == 1 else 0) - (second if b == 2 else 0)),
                     safe_queue_value=float(-(first if a == 1 else 0) - (second if b == 2 else 0)))
                for a in range(3) for b in range(3)]
    margin = 31e-8
    coefficients = [[count, 1], [start, float(first)], [start + 3, float(second)]]
    cert = dict(kind="queue_joint", complete_domain=True, global_integer_valid=True,
        direct_global_bound_update=False, ordinary_auxiliary_optimization_calls=0, assignment_scope="task",
        profile=p, selected_heads=[0, 1], selected_tasks=[0, 1], vertices=vertices,
        affine_coefficients=affine, lifting_rule="JL_positive_marginal_at_full_positive_sum",
        external_positive_sums=[9, 18], external_penalties=[0] * 4, numerical_margin=margin,
        conversion_error=0, arithmetic_error=0, intercept_correction=margin, facet_validation_correction=0,
        separation_design=[0] * count, separation_theta=0, value_at_separation_point=30 - margin)
    row = dict(certificate=cert, rhs=30 - margin, coefficients=coefficients, violation=30 - margin)
    assert audit.replay_queue_row(row, c)["vertex_checks"] == 9
    forged = deepcopy(row)
    forged["certificate"]["affine_coefficients"][1] += 1
    with pytest.raises(ValueError, match="excludes a vertex"):
        audit.replay_queue_row(forged, c)
    forged = deepcopy(row)
    forged["certificate"]["external_penalties"][1] = -1
    with pytest.raises(ValueError, match="unsafe joint external penalty"):
        audit.replay_queue_row(forged, c)


def test_k1_has_only_single_tasks_and_rejects_hidden_pairs(independent_case):
    model, demand, _, _, _, _ = independent_case
    with pytest.raises(ValueError, match="K1"):
        task.assignment_groups(model, demand, max_orders=1)
    demand = deepcopy(demand)
    demand["pair_rates"] = [[0]]
    scope, groups = task.assignment_groups(model, demand, max_orders=1)
    assert scope == "task" and groups == [[(0, None, 1)]]
