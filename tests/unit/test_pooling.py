"""Exact aggregate lifts, zero-window elimination, and segment reconstruction."""

from types import SimpleNamespace

import numpy as np
import pytest

from test_solver import data, design
from src.model.paths import enumerate_paths
from src.operations.validation import build_operations
from src.solvers.formulation import build_model, read_solution


@pytest.mark.parametrize("policy", ["aggregate", "prefix"])
def test_zero_window_has_no_continuation_lift(policy):
    d = data(max_orders=1, flow_policy=policy)
    paths = enumerate_paths(d)
    built = build_model(d, paths, fixed_design=design(d))
    try:
        assert not built.variables["_active_lifts"]
        assert all(
            len(built.variables[name]) == 0
            for name in ("eta", "u", "chain", "chain_product")
        )
        assert not built.constraints["double"]
        built.model.optimize()
        result = read_solution(built)
        assert result["_solution"]["eta"].shape == (len(paths.initial_segments),)
        assert not result["_solution"]["eta"].any()
        assert build_operations(d, result)["summary"]["checks"]["passed"]
    finally:
        built.model.dispose()


def test_aggregate_groups_are_catalogue_defined_and_design_independent():
    d = data(max_orders=2, flow_policy="aggregate")
    paths = enumerate_paths(d)
    chosen = design(d)
    built = [
        build_model(d, paths, fixed_design=chosen),
        build_model(
            d, paths, fixed_design=chosen, service_indices=(), empty_indices=()
        ),
        build_model(d, paths, fixed_design=design(d, all_closed=True)),
    ]
    try:
        groups = built[0].variables["_lift_groups"]
        assert len(groups) < len(paths.initial_segments)
        assert all(item.variables["_lift_groups"] == groups for item in built)
        assert all(len(item.variables["eta"]) == len(groups) for item in built)
        first, second = built[0].model, built[2].model
        assert (first.getA() != second.getA()).nnz == 0
        for attribute in ("LB", "UB", "Obj"):
            np.testing.assert_array_equal(
                first.getAttr(attribute), second.getAttr(attribute)
            )
    finally:
        for item in built:
            item.model.dispose()
    prefix = data(max_orders=2, flow_policy="prefix")
    paths = enumerate_paths(prefix)
    built = build_model(prefix, paths, fixed_design=design(prefix))
    try:
        assert len(built.variables["eta"]) == len(paths.initial_segments)
        assert all(len(group) == 1 for group in built.variables["_lift_groups"])
    finally:
        built.model.dispose()


def test_zero_kappa_od_still_enters_the_assignment_arrival_chain():
    d = data(max_orders=2, flow_policy="aggregate", kappa=[[1, 0], [0, 0]])
    paths = enumerate_paths(d)
    built = build_model(d, paths, fixed_design=design(d))
    try:
        groups = built.variables["_lift_groups"]
        assert len(built.variables["u"]) == len(groups) * d.n_od
        assert len(built.variables["chain"]) == len(groups) * (d.n_od + 1)
        built.model.optimize()
        result = read_solution(built)
        head, eta = (result["_solution"][name] for name in ("head", "eta"))
        assert head.sum() > 1
        rate = float(d.demand.sum())
        factor = -np.expm1(-d.effective_delta * rate) / rate
        np.testing.assert_allclose(eta, factor * head, atol=1e-7)
        assert build_operations(d, result)["summary"]["checks"]["passed"]
    finally:
        built.model.dispose()


def test_positive_tiny_heads_use_actual_head_sum_and_are_not_erased():
    d = data(max_orders=2, flow_policy="aggregate")
    paths = enumerate_paths(d)
    built = build_model(d, paths, fixed_design=design(d))
    try:
        built.model.optimize()
        groups = built.variables["_lift_groups"]
        g = next(i for i, indices in enumerate(groups) if len(indices) >= 2)
        first, second = groups[g][:2]
        heads = np.zeros(len(paths.initial_segments))
        heads[first], heads[second] = 1e-16, 3e-16
        built.variables["head"] = {
            j: SimpleNamespace(X=float(value)) for j, value in enumerate(heads)
        }
        built.variables["eta"] = {
            j: SimpleNamespace(X=1e-17 if j == g else 0.0) for j in range(len(groups))
        }
        # Deliberately inconsistent stored group-head value models numerical
        # equality residuals; export must normalize by actual segment heads.
        built.variables["group_head"][g] = SimpleNamespace(X=1.0)
        result = read_solution(built)
        eta = result["_solution"]["eta"]
        assert eta[first] > 0 and eta[second] > 0
        np.testing.assert_allclose(
            eta[[first, second]], [2.5e-18, 7.5e-18], rtol=1e-14, atol=0
        )
        built.variables["head"] = {j: SimpleNamespace(X=0.0) for j in range(len(heads))}
        assert not read_solution(built)["_solution"]["eta"].any()
    finally:
        built.model.dispose()
