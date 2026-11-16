"""Compact physical witnesses preserve the independent native-output audit.

No optimizer is used.  The two-base fixture has unused single/double types and
an origin whose cheapest path is more expensive than the global minimum, so a
minimum-cost witness alone cannot represent all supported departures.
"""

from copy import deepcopy

import numpy as np
import pytest

from src.experiments.native import load_native_result, save_json_result
from src.operations.validation import build_operations
from test_generalized_demand import _manual_preprocessed
from test_operations import instance


@pytest.fixture(params=[1, 2], ids=["single", "double"])
def physical_result(request):
    return _manual_preprocessed(instance(
        n_l=2, demand=[4.0, 3.0], kappa=np.ones((2, 2)),
        max_orders=request.param, flow_policy="aggregate",
        flight_times=[
            [0, .11, .05, .07], [.11, 0, .16, .18],
            [.05, .16, 0, .1], [.07, .18, .1, 0],
        ],
    ))


def _compact(result):
    """Make the proposed portable payload while preserving original path IDs."""
    out = deepcopy(result)
    for family in ("_missions", "_relocations"):
        records = result[family]
        keep = {i for i, row in enumerate(records) if row["flow"] != 0}
        cheapest, arc_maximum = {}, {}
        for i, row in enumerate(records):
            path = row["path"]
            if family == "_missions":
                key = path["first_od"], path["second_od"], path["b"][0]
                if (key not in cheapest
                        or path["cost"] < records[cheapest[key]]["path"]["cost"]):
                    cheapest[key] = i
            for u, v, count in path["arc_counts"]:
                if (u, v) not in arc_maximum or count > arc_maximum[u, v][0]:
                    arc_maximum[u, v] = count, i
        keep.update(cheapest.values())
        keep.update(i for _, i in arc_maximum.values())
        out[family] = [row for i, row in enumerate(records) if i in keep]
    return out


def _statistics(records):
    """Statistics the demand reconstruction and link audit consume."""
    cost, support, traversals = {}, set(), {}
    for row in records:
        p = row["path"]
        if "first_od" in p:
            task = p["first_od"], p["second_od"]
            cost[task] = min(cost.get(task, np.inf), p["cost"])
            support.add((*task, p["b"][0]))
        for u, v, count in p["arc_counts"]:
            traversals[u, v] = max(traversals.get((u, v), 0), count)
    return cost, support, traversals


def test_compact_witnesses_preserve_physical_statistics_and_full_audit(physical_result):
    data, dense = physical_result
    compact = _compact(dense)
    assert len(compact["_missions"]) < len(dense["_missions"])
    for family in ("_missions", "_relocations"):
        assert _statistics(compact[family]) == _statistics(dense[family])
    assert any(row["flow"] == 0 and row["path"]["b"][0] == 1
               for row in compact["_missions"])
    if data.max_orders == 2:
        assert any(row["flow"] == 0 and row["path"]["second_od"] is not None
                   for row in compact["_missions"])
    # The unused second departure needs a witness even though it never attains
    # a global task minimum.  This is not just removing duplicate JSON rows.
    global_cost = _statistics(dense["_missions"])[0]
    for row in compact["_missions"]:
        p = row["path"]
        if p["b"][0] == 1:
            assert p["cost"] > global_cost[p["first_od"], p["second_od"]]
    before, after = build_operations(data, dense), build_operations(data, compact)
    assert before["summary"]["checks"]["passed"]
    assert after["summary"]["checks"]["passed"]
    assert before["summary"] == after["summary"]
    assert before["tables"] == after["tables"]


def test_tiny_nonzero_flow_is_retained_without_a_cutoff(physical_result):
    data, original = physical_result
    dense = deepcopy(original)
    # Move a tiny portion of one singleton to another path of the same task,
    # retaining both endpoint bases and all order totals.
    source = next(row for row in dense["_missions"]
                  if row["flow"] > 0 and row["path"]["second_od"] is None)
    p = source["path"]
    target = next(row for row in dense["_missions"]
                  if row["flow"] == 0 and row["path"]["first_od"] == p["first_od"]
                  and row["path"]["second_od"] is None and row["path"]["b"] == p["b"])
    q, epsilon = target["path"], 1e-12
    source["flow"] -= epsilon
    target["flow"] = epsilon
    solution = dense["_solution"]
    solution["flow"][p["id"]] -= epsilon
    solution["flow"][q["id"]] = epsilon
    solution["head"][p["segment_id"]] -= epsilon
    solution["head"][q["segment_id"]] += epsilon
    extra_busy = epsilon * (q["time"] - p["time"])
    solution["fleet"] += extra_busy
    dense["objective"] += data.fleet_cost * extra_busy + epsilon * (q["cost"] - p["cost"])
    compact = _compact(dense)
    kept = {row["path"]["id"]: row["flow"] for row in compact["_missions"]}
    assert kept[q["id"]] == epsilon
    before, after = build_operations(data, dense), build_operations(data, compact)
    assert before["summary"]["checks"]["passed"]
    assert before["summary"] == after["summary"]
    assert before["tables"] == after["tables"]


def test_omitting_a_nonzero_record_fails_flow_storage_audit(physical_result):
    data, dense = physical_result
    compact = _compact(dense)
    missing = next(row for row in compact["_missions"] if row["flow"] > .01)
    compact["_missions"].remove(missing)
    # The witness could also be essential to the reconstruction; either a
    # reconstruction rejection or a failed numerical audit is acceptable.
    try:
        checks = build_operations(data, compact)["summary"]["checks"]
    except ValueError as error:
        assert "candidate order" in str(error)
    else:
        assert not checks["passed"]
        assert checks["residuals"]["service_flow_storage_total"] >= missing["flow"]


def test_corrupt_zero_flow_witness_is_still_physically_checked(physical_result):
    data, dense = physical_result
    compact = _compact(dense)
    witness = next(row for row in compact["_missions"]
                   if row["flow"] == 0 and row["path"]["b"][0] == 1)
    witness["path"]["time"] += .1
    try:
        checks = build_operations(data, compact)["summary"]["checks"]
    except ValueError as error:
        assert "candidate order" in str(error)
    else:
        assert not checks["passed"]
        assert any("Cached time" in error for error in checks["physical_errors"])


def test_compact_audit_does_not_trust_saved_reconstructed_demand(physical_result):
    data, dense = physical_result
    compact = _compact(dense)
    compact["preprocessed_demand"]["single_rates"][0] += .125
    checks = build_operations(data, compact)["summary"]["checks"]
    assert not checks["passed"]
    assert checks["residuals"]["preprocessed_single_rates"] == pytest.approx(.125)


@pytest.mark.parametrize("compact_payload", [False, True], ids=["legacy_dense", "compact"])
def test_native_loader_preserves_both_output_forms(physical_result, tmp_path, compact_payload):
    data, dense = physical_result
    result = _compact(dense) if compact_payload else dense
    save_json_result(tmp_path, result)
    loaded = load_native_result(tmp_path)
    for key in result["_solution"]:
        np.testing.assert_array_equal(loaded["_solution"][key], result["_solution"][key])
    assert len(loaded["_missions"]) == len(result["_missions"])
    assert build_operations(data, loaded)["summary"]["checks"]["passed"]
