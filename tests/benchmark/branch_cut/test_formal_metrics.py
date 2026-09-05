from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

import src.benchmark.branch_cut.cli as cli_module
from src.benchmark.branch_cut.formal_metrics import (
    FormalMetricValidationError,
    validate_formal_attempt_paper_metrics,
)
from src.benchmark.branch_cut.schema import TelemetryConfig
from src.benchmark.branch_cut.telemetry import (
    collect_cut_stats,
    collect_progress_rows,
    collect_root_round_rows,
    collect_solver_metrics,
)


def _core(*, user_cuts: bool = True) -> dict[str, Any]:
    return {
        "use_user_cuts": user_cuts,
        "max_root_cut_rounds": 3,
        "max_cuts_per_round": 250,
        "static_source_level": False,
        "static_virtual_level_lower": False,
        "static_lo_choice_corridor": False,
        "static_lo_choice_capacity": False,
        "static_queue_perspective": False,
        "static_nearest_display_suffix": False,
        "static_chi_vub_top_k_per_corridor": 0,
        "cut_chi_vub": user_cuts,
        "cut_chi_facility_vub": False,
        "cut_route_swap_station_vub": False,
        "cut_route_swap_leg_vub": False,
        "cut_queue_perspective": False,
        "cut_lo_choice_corridor": False,
        "cut_lo_choice_capacity": False,
        "cut_directed_return": False,
        "cut_od_disaggregated_return": False,
    }


def _route_counts() -> dict[str, int]:
    return {
        "swap_route_options": 19,
        "swap_leg_bundles": 8,
        "swap_station_bundles": 2,
        "strengthening_swap_leg_bundles": 5,
        "strengthening_swap_station_bundles": 2,
    }


def _valid_artifacts(
    *,
    has_incumbent: bool = True,
    with_root_round: bool = True,
) -> dict[str, Any]:
    objective = 100.0 if has_incumbent else None
    bound = 95.0
    gap = 0.05 if has_incumbent else None
    root_before = 90.0 if with_root_round else None
    root_after = 92.0 if with_root_round else None
    callback_runtime = 1.0
    metrics = {
        "status": 9,
        "status_name": "TIME_LIMIT",
        "has_incumbent": has_incumbent,
        "solution_count": int(has_incumbent),
        "objective": objective,
        "best_bound": bound,
        "relative_gap": gap,
        "node_count": 12.0,
        "main_solve_runtime_seconds": 40.0,
        "total_algorithm_runtime_seconds": 45.0,
        "algorithm_prepare_runtime_seconds": 3.0,
        "start_preparation_runtime_seconds": 2.0,
        "first_incumbent_time_seconds": 5.0 if has_incumbent else None,
        "first_incumbent_objective": 110.0 if has_incumbent else None,
        "root_bound_before_cuts": root_before,
        "root_bound_after_cuts": root_after,
        "callback_runtime_seconds": callback_runtime,
        "budget_exhausted_before_main_solve": False,
        "main_solve_started": True,
        "optimize_called": True,
    }
    progress: list[dict[str, Any]] = [
        {
            "event": "mip_progress",
            "runtime_seconds": 1.0,
            "incumbent": None,
            "bound": 90.0,
            "relative_gap": None,
            "node_count": 0.0,
            "solution_count": 0,
        }
    ]
    if has_incumbent:
        progress.append(
            {
                "event": "incumbent",
                "runtime_seconds": 5.0,
                "incumbent": 110.0,
                "bound": 92.0,
                "relative_gap": 18.0 / 110.0,
            }
        )
    progress.append(
        {
            "event": "final",
            "runtime_seconds": 40.0,
            "total_algorithm_runtime_seconds": 45.0,
            "budget_exhausted_before_main_solve": False,
            "main_solve_started": True,
            "optimize_called": True,
            "incumbent": objective,
            "bound": bound,
            "relative_gap": gap,
            "node_count": 12.0,
            "solution_count": int(has_incumbent),
            "status": 9,
        }
    )
    root_rounds = (
        [
            {
                "round": 1,
                "runtime_seconds": 0.5,
                "root_bound_before_round": 90.0,
                "cuts_added": 3,
                "cuts_added_by_family": {"chi_od_corridor_vub": 3},
            }
        ]
        if with_root_round
        else []
    )
    user_added = {"chi_od_corridor_vub": 3} if with_root_round else {}
    candidates = {"chi_od_corridor_vub": 5} if with_root_round else {}
    duplicates = {"chi_od_corridor_vub": 1} if with_root_round else {}
    separation = {"chi_od_corridor_vub": 0.2} if with_root_round else {}
    cut_stats = {
        "captured": True,
        "static_rows_added_by_family": {},
        "user_cuts_added_by_family": user_added,
        "user_cut_candidates_by_family": candidates,
        "user_cut_duplicates_by_family": duplicates,
        "user_cut_rejections_by_family": {},
        "user_cut_separation_seconds_by_family": separation,
        "route_bundle_index_counts": _route_counts(),
        "static_rows_added_total": 0,
        "user_cuts_added_total": 3 if with_root_round else 0,
        "root_cut_rounds": int(with_root_round),
        "callback_runtime_seconds": callback_runtime,
    }
    run_record = {
        "completed_without_exception": True,
        "solve_returned_incumbent": has_incumbent,
        "solver_metrics": metrics,
        "execution_wall_runtime_seconds": 46.0,
        "budget_exhausted_before_main_solve": False,
        "main_solve_started": True,
        "optimize_called": True,
        "telemetry_counts": {
            "progress_events": len(progress),
            "root_round_events": len(root_rounds),
            "static_rows_added": 0,
            "user_cuts_added": 3 if with_root_round else 0,
        },
    }
    return {
        "run_record": run_record,
        "progress": progress,
        "root_rounds": root_rounds,
        "cut_stats": cut_stats,
        "core_solver_config": _core(),
        "completed": True,
    }


def _validate(artifacts: dict[str, Any]) -> None:
    validate_formal_attempt_paper_metrics(**artifacts)


def test_formal_metric_validator_is_bound_into_formal_implementation_identity() -> None:
    assert Path("src/benchmark/branch_cut/formal_metrics.py") in (
        cli_module._FORMAL_EXECUTION_IMPLEMENTATION_FILES
    )


def test_current_telemetry_collectors_produce_a_compatible_no_incumbent_timeout() -> None:
    class Backend:
        Status = 9
        SolCount = 0
        ObjBound = 95.0
        MIPGap = None
        NodeCount = 12.0
        Runtime = 40.0
        NumStart = 0

    class Wrapper:
        model = Backend()
        main_solve_runtime = 40.0
        total_runtime = 45.0
        algorithm_prepare_runtime = 3.0
        start_preparation_runtime = 2.0
        callback_runtime = 1.0
        first_incumbent_time = None
        first_incumbent_objective = None
        root_bound_before_cuts = 90.0
        root_bound_after_cuts = 92.0
        budget_exhausted_before_main_solve = False
        main_solve_started = True
        optimize_called = True
        progress_events = (
            {
                "event": "mip_progress",
                "runtime_seconds": 1.0,
                "incumbent": None,
                "bound": 90.0,
                "relative_gap": None,
                "node_count": 0.0,
                "solution_count": 0,
            },
            {
                "event": "final",
                "runtime_seconds": 40.0,
                "total_algorithm_runtime_seconds": 45.0,
                "budget_exhausted_before_main_solve": False,
                "main_solve_started": True,
                "optimize_called": True,
                "incumbent": None,
                "bound": 95.0,
                "relative_gap": None,
                "node_count": 12.0,
                "solution_count": 0,
                "status": 9,
            },
        )
        root_round_events = (
            {
                "round": 1,
                "runtime_seconds": 0.5,
                "root_bound_before_round": 90.0,
                "cuts_added": 3,
                "cuts_added_by_family": {"chi_od_corridor_vub": 3},
            },
        )
        static_cut_counts: dict[str, int] = {}
        user_cut_counts = {"chi_od_corridor_vub": 3}
        user_cut_candidate_counts = {"chi_od_corridor_vub": 5}
        user_cut_duplicate_counts = {"chi_od_corridor_vub": 1}
        user_cut_rejection_counts: dict[str, int] = {}
        user_cut_separation_seconds = {"chi_od_corridor_vub": 0.2}
        route_bundle_index_counts = _route_counts()
        root_cut_rounds = 1

    wrapper = Wrapper()
    telemetry = TelemetryConfig()
    metrics = collect_solver_metrics(wrapper)
    metrics.update(
        {
            "budget_exhausted_before_main_solve": False,
            "main_solve_started": True,
            "optimize_called": True,
        }
    )
    progress = collect_progress_rows(wrapper, telemetry)
    root_rounds = collect_root_round_rows(wrapper, telemetry)
    cut_stats = collect_cut_stats(wrapper, telemetry)
    run_record = {
        "solver_metrics": metrics,
        "solve_returned_incumbent": False,
        "execution_wall_runtime_seconds": 46.0,
        "budget_exhausted_before_main_solve": False,
        "main_solve_started": True,
        "optimize_called": True,
        "telemetry_counts": {
            "progress_events": len(progress),
            "root_round_events": len(root_rounds),
            "static_rows_added": cut_stats["static_rows_added_total"],
            "user_cuts_added": cut_stats["user_cuts_added_total"],
        },
    }
    validate_formal_attempt_paper_metrics(
        run_record=run_record,
        progress=progress,
        root_rounds=root_rounds,
        cut_stats=cut_stats,
        core_solver_config=_core(),
        completed=True,
    )


@pytest.mark.parametrize("has_incumbent", (False, True))
def test_time_limit_with_or_without_incumbent_is_a_valid_completed_state(
    has_incumbent: bool,
) -> None:
    _validate(_valid_artifacts(has_incumbent=has_incumbent))


def test_time_limit_without_incumbent_or_finite_bound_remains_legal() -> None:
    artifacts = _valid_artifacts(has_incumbent=False, with_root_round=False)
    metrics = artifacts["run_record"]["solver_metrics"]
    metrics["best_bound"] = None
    artifacts["progress"][0]["bound"] = None
    artifacts["progress"][-1]["bound"] = None
    _validate(artifacts)


def test_budget_exhausted_before_optimize_has_canonical_loaded_state() -> None:
    artifacts = _valid_artifacts(has_incumbent=False, with_root_round=False)
    metrics = artifacts["run_record"]["solver_metrics"]
    metrics.update(
        {
            "status": 1,
            "status_name": "LOADED",
            "best_bound": None,
            "node_count": None,
            "main_solve_runtime_seconds": 0.0,
            "total_algorithm_runtime_seconds": 10.0,
            "algorithm_prepare_runtime_seconds": 5.0,
            "start_preparation_runtime_seconds": 5.0,
            "callback_runtime_seconds": 0.0,
            "budget_exhausted_before_main_solve": True,
            "main_solve_started": False,
            "optimize_called": False,
        }
    )
    record = artifacts["run_record"]
    record.update(
        {
            "execution_wall_runtime_seconds": 11.0,
            "budget_exhausted_before_main_solve": True,
            "main_solve_started": False,
            "optimize_called": False,
        }
    )
    artifacts["progress"] = [
        {
            "event": "final",
            "runtime_seconds": 0.0,
            "total_algorithm_runtime_seconds": 10.0,
            "budget_exhausted_before_main_solve": True,
            "main_solve_started": False,
            "optimize_called": False,
            "incumbent": None,
            "bound": None,
            "relative_gap": None,
            "node_count": 0.0,
            "solution_count": 0,
            "status": 1,
        }
    ]
    artifacts["cut_stats"]["callback_runtime_seconds"] = 0.0
    record["telemetry_counts"]["progress_events"] = 1
    _validate(artifacts)


@pytest.mark.parametrize(
    "missing_field",
    (
        "status",
        "objective",
        "best_bound",
        "relative_gap",
        "main_solve_runtime_seconds",
        "first_incumbent_time_seconds",
        "node_count",
        "root_bound_before_cuts",
        "callback_runtime_seconds",
    ),
)
def test_completed_attempt_rejects_each_missing_paper_metric(
    missing_field: str,
) -> None:
    artifacts = _valid_artifacts()
    del artifacts["run_record"]["solver_metrics"][missing_field]
    with pytest.raises(FormalMetricValidationError, match="missing"):
        _validate(artifacts)


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (
            lambda a: a["run_record"]["solver_metrics"].__setitem__(
                "status_name", "OPTIMAL"
            ),
            "status_name",
        ),
        (
            lambda a: a["run_record"]["solver_metrics"].__setitem__(
                "solution_count", 0
            ),
            "solution_count",
        ),
        (
            lambda a: a["run_record"]["solver_metrics"].__setitem__(
                "relative_gap", 0.2
            ),
            "differs from objective",
        ),
        (
            lambda a: a["run_record"]["solver_metrics"].__setitem__(
                "main_solve_runtime_seconds", -1.0
            ),
            "nonnegative",
        ),
        (
            lambda a: a["run_record"]["solver_metrics"].__setitem__(
                "first_incumbent_time_seconds", 41.0
            ),
            "first incumbent exceeds",
        ),
        (
            lambda a: a["run_record"]["solver_metrics"].__setitem__(
                "node_count", -1.0
            ),
            "nonnegative",
        ),
        (
            lambda a: a["run_record"]["solver_metrics"].__setitem__(
                "root_bound_after_cuts", None
            ),
            "jointly present",
        ),
        (
            lambda a: a["cut_stats"].__setitem__(
                "callback_runtime_seconds", 0.5
            ),
            "callback runtime differs",
        ),
    ),
)
def test_solver_metric_type_range_and_cross_artifact_drift_fail_closed(
    mutation,
    message: str,
) -> None:
    artifacts = _valid_artifacts()
    mutation(artifacts)
    with pytest.raises(FormalMetricValidationError, match=message):
        _validate(artifacts)


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (
            lambda a: a["progress"].pop(),
            "trailing final",
        ),
        (
            lambda a: a["progress"][-1].__setitem__("status", 2),
            "status differs",
        ),
        (
            lambda a: a["progress"][-1].__setitem__("runtime_seconds", 0.5),
            "runtimes are not nondecreasing",
        ),
        (
            lambda a: a["progress"][-1].__setitem__("unexpected", 1),
            "fields differ",
        ),
        (
            lambda a: a["run_record"]["telemetry_counts"].__setitem__(
                "progress_events", 99
            ),
            "telemetry_counts differ",
        ),
    ),
)
def test_progress_and_telemetry_count_corruption_fail_closed(
    mutation,
    message: str,
) -> None:
    artifacts = _valid_artifacts()
    mutation(artifacts)
    with pytest.raises(FormalMetricValidationError, match=message):
        _validate(artifacts)


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (
            lambda a: a["cut_stats"].__setitem__("user_cuts_added_total", 4),
            "differs from its family counts",
        ),
        (
            lambda a: a["cut_stats"].__setitem__("root_cut_rounds", 2),
            "row count differs",
        ),
        (
            lambda a: a["root_rounds"][0].__setitem__("cuts_added", 2),
            "differs from family counts",
        ),
        (
            lambda a: a["cut_stats"]["user_cut_candidates_by_family"].__setitem__(
                "chi_od_corridor_vub", 3
            ),
            "candidate count undercounts",
        ),
        (
            lambda a: a["cut_stats"]["user_cut_separation_seconds_by_family"].__setitem__(
                "chi_od_corridor_vub", 2.0
            ),
            "separation time exceeds",
        ),
        (
            lambda a: a["cut_stats"]["route_bundle_index_counts"].pop(
                "swap_route_options"
            ),
            "route_bundle_index_counts fields differ",
        ),
    ),
)
def test_root_round_and_cut_family_corruption_fail_closed(
    mutation,
    message: str,
) -> None:
    artifacts = _valid_artifacts()
    mutation(artifacts)
    with pytest.raises(FormalMetricValidationError, match=message):
        _validate(artifacts)


def test_user_cut_activity_is_rejected_when_the_profile_disables_cuts() -> None:
    artifacts = _valid_artifacts()
    artifacts["core_solver_config"] = _core(user_cuts=False)
    with pytest.raises(FormalMetricValidationError, match="disabled or unknown"):
        _validate(artifacts)


def test_od_disaggregated_return_telemetry_is_bound_to_its_own_flag() -> None:
    family = "od_disaggregated_directed_return"
    artifacts = _valid_artifacts()
    artifacts["root_rounds"][0]["cuts_added_by_family"] = {family: 3}
    stats = artifacts["cut_stats"]
    stats["user_cuts_added_by_family"] = {family: 3}
    stats["user_cut_candidates_by_family"] = {family: 5}
    stats["user_cut_duplicates_by_family"] = {family: 1}
    stats["user_cut_separation_seconds_by_family"] = {family: 0.2}

    with pytest.raises(FormalMetricValidationError, match="disabled or unknown"):
        _validate(artifacts)

    artifacts["core_solver_config"]["cut_od_disaggregated_return"] = True
    _validate(artifacts)


def test_failed_attempt_preserves_nullable_unavailable_telemetry() -> None:
    artifacts = {
        "run_record": {
            "solver_metrics": {"has_incumbent": False},
            "telemetry_counts": {
                "progress_events": 0,
                "root_round_events": 0,
                "static_rows_added": None,
                "user_cuts_added": None,
            },
        },
        "progress": [],
        "root_rounds": [],
        "cut_stats": {"captured": False},
        "core_solver_config": _core(user_cuts=False),
        "completed": False,
    }
    _validate(artifacts)


def test_mutation_helpers_do_not_share_nested_state() -> None:
    original = _valid_artifacts()
    changed = deepcopy(original)
    changed["cut_stats"]["user_cuts_added_by_family"]["chi_od_corridor_vub"] = 9
    assert original["cut_stats"]["user_cuts_added_by_family"] == {
        "chi_od_corridor_vub": 3
    }
