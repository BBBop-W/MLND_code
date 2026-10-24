"""Wall-time accounting uses disjoint phases, including terminal solver calls."""

import json

import pytest

from src.solvers import algorithm
from src.solvers.api import solve
from src.solvers.dual import DUAL_TIMING_PHASES
from src.solvers.formulation import BuildStopped
from test_solver import data


def _check_subproblem_accounting(stats):
    phases = stats["subproblem_timings"]
    details = stats["subproblem_timing_details"]
    assert all(value >= 0 for value in phases.values())
    assert sum(phases.values()) <= stats["subproblem_wall"] + 1e-8
    assert details["primal_solver"]["optimize"] <= phases["primal"] + 1e-8
    # Backend Runtime is a second measurement of optimize, not another phase.
    dual = sum(details["dual"][key] for key in DUAL_TIMING_PHASES)
    assert dual <= phases["dual"] + 1e-8
    pareto = details["pareto"]
    leaves = sum(pareto["dual"][key] for key in DUAL_TIMING_PHASES)
    leaves += pareto["pricing"] + pareto["validation"]
    assert leaves == pytest.approx(pareto["wall"], abs=1e-8)
    assert pareto["wall"] <= phases["pareto"] + 1e-8
    assert stats["pareto_seconds"] == phases["pareto"]


@pytest.mark.parametrize("max_orders", [1, 2])
def test_real_benders_times_are_additive_and_include_final_master(max_orders):
    result = solve(data(max_orders=max_orders), time_limit=10, mip_gap=1e-6)
    assert result["termination_reason"] == "gap_limit"
    stats = result["acceleration_statistics"]
    master = stats["master_solver_timings"]
    # The final master closes the gap before another subproblem/record is made.
    assert master["calls"] == len(result["iterations"]) + 1
    assert master["optimize"] > 0
    _check_subproblem_accounting(stats)
    accounted = sum(stats["algorithm_timings"].values())
    accounted += master["optimize"] + stats["subproblem_wall"]
    assert accounted <= result["algorithm_wall"] + 1e-8
    assert result["algorithm_wall"] + result["enumeration_only_seconds"] <= (
        result["solver_wall"] + 1e-8
    )
    # Initial-design work is already included in the aggregate, not extra time.
    records = result["iterations"] + result["acceleration_records"]
    for key, aggregate in stats["subproblem_timings"].items():
        assert sum(row["timings"][key] for row in records) == pytest.approx(aggregate)


def test_runner_checkpoint_leaves_replace_parent_without_double_count(
    tmp_path, small_input, monkeypatch
):
    from src.experiments import runner

    monkeypatch.setattr(runner, "source_hashes", lambda: {"test": "stable"})
    output = tmp_path / "timed"
    runner.run_case(small_input, output, time_limit=10, mip_gap=1e-6, make_report=False)
    result = json.loads((output / "result.json").read_text())
    assert result["termination_reason"] == "gap_limit"
    stats, external = result["acceleration_statistics"], result["runner_timings"]
    _check_subproblem_accounting(stats)
    algorithm_phases = stats["algorithm_timings"]
    checkpoint = external["checkpoint_validation"] + external["checkpoint_write"]
    assert checkpoint > 0
    assert checkpoint <= algorithm_phases["checkpoint"] + 1e-8
    # Subproblem wall contains its progress callbacks. Use its disjoint phases
    # instead when progress I/O is counted separately at the runner boundary.
    accounted = result["enumeration_only_seconds"]
    accounted += stats["master_solver_timings"]["optimize"]
    accounted += sum(stats["subproblem_timings"].values())
    accounted += sum(v for k, v in algorithm_phases.items() if k != "checkpoint")
    accounted += checkpoint + external["progress_io"]
    assert accounted <= result["solver_wall"] + 1e-8
    assert result["solver_wall"] + external["final_validation"] <= (
        result["wall"] + 1e-8
    )


def test_optimize_exception_retains_call_time_but_pre_call_stop_does_not():
    class InterruptedModel:
        Runtime = 0.0

        def optimize(self, callback):
            raise BuildStopped("user_stop")

    model, timings = InterruptedModel(), {}
    with pytest.raises(BuildStopped, match="user_stop"):
        algorithm.optimize(model, timings=timings)
    assert timings["calls"] == 1
    assert timings["optimize"] > 0
    assert timings["solver_runtime"] == 0
    before = timings.copy()
    with pytest.raises(BuildStopped, match="user_stop"):
        algorithm.optimize(model, stop_requested=lambda: True, timings=timings)
    assert timings == before
