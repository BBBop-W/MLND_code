"""Fail-closed semantic validation for paper-facing formal-run telemetry.

The formal journal authenticates bytes, but byte authentication alone does not
make a metric meaningful.  This module validates the solver summary and the
three telemetry artifacts as one coherent observation before a formal attempt
can be recovered or sealed.  It deliberately contains no Gurobi dependency.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any


class FormalMetricValidationError(ValueError):
    """Formal telemetry is missing, malformed, or internally inconsistent."""


_STATUS_NAMES = {
    1: "LOADED",
    2: "OPTIMAL",
    3: "INFEASIBLE",
    4: "INF_OR_UNBD",
    5: "UNBOUNDED",
    6: "CUTOFF",
    7: "ITERATION_LIMIT",
    8: "NODE_LIMIT",
    9: "TIME_LIMIT",
    10: "SOLUTION_LIMIT",
    11: "INTERRUPTED",
    12: "NUMERIC",
    13: "SUBOPTIMAL",
    14: "INPROGRESS",
    15: "USER_OBJ_LIMIT",
    16: "WORK_LIMIT",
    17: "MEM_LIMIT",
}
_NO_INCUMBENT_STATUSES = frozenset(
    {"LOADED", "INFEASIBLE", "INF_OR_UNBD", "UNBOUNDED", "CUTOFF"}
)
_INCUMBENT_REQUIRED_STATUSES = frozenset({"OPTIMAL", "SOLUTION_LIMIT"})

_REQUIRED_COMPLETED_SOLVER_METRICS = frozenset(
    {
        "status",
        "status_name",
        "has_incumbent",
        "solution_count",
        "objective",
        "best_bound",
        "relative_gap",
        "node_count",
        "main_solve_runtime_seconds",
        "total_algorithm_runtime_seconds",
        "algorithm_prepare_runtime_seconds",
        "start_preparation_runtime_seconds",
        "first_incumbent_time_seconds",
        "first_incumbent_objective",
        "root_bound_before_cuts",
        "root_bound_after_cuts",
        "callback_runtime_seconds",
        "budget_exhausted_before_main_solve",
        "main_solve_started",
        "optimize_called",
    }
)

_PROGRESS_FIELDS = {
    "mip_progress": frozenset(
        {
            "event",
            "runtime_seconds",
            "incumbent",
            "bound",
            "relative_gap",
            "node_count",
            "solution_count",
        }
    ),
    "incumbent": frozenset(
        {
            "event",
            "runtime_seconds",
            "incumbent",
            "bound",
            "relative_gap",
        }
    ),
    "final": frozenset(
        {
            "event",
            "runtime_seconds",
            "total_algorithm_runtime_seconds",
            "budget_exhausted_before_main_solve",
            "main_solve_started",
            "optimize_called",
            "incumbent",
            "bound",
            "relative_gap",
            "node_count",
            "solution_count",
            "status",
        }
    ),
}
_ROOT_ROUND_FIELDS = frozenset(
    {
        "round",
        "runtime_seconds",
        "root_bound_before_round",
        "cuts_added",
        "cuts_added_by_family",
    }
)
_CUT_STATS_FIELDS = frozenset(
    {
        "captured",
        "static_rows_added_by_family",
        "user_cuts_added_by_family",
        "user_cut_candidates_by_family",
        "user_cut_duplicates_by_family",
        "user_cut_rejections_by_family",
        "user_cut_separation_seconds_by_family",
        "route_bundle_index_counts",
        "static_rows_added_total",
        "user_cuts_added_total",
        "root_cut_rounds",
        "callback_runtime_seconds",
    }
)
_TELEMETRY_COUNT_FIELDS = frozenset(
    {
        "progress_events",
        "root_round_events",
        "static_rows_added",
        "user_cuts_added",
    }
)
_ROUTE_BUNDLE_INDEX_FIELDS = frozenset(
    {
        "swap_route_options",
        "swap_leg_bundles",
        "swap_station_bundles",
        "strengthening_swap_leg_bundles",
        "strengthening_swap_station_bundles",
    }
)
_STATIC_FAMILY_CONTROLS = {
    "source_level_upper": "static_source_level",
    "virtual_level_lower": "static_virtual_level_lower",
    "lo_choice_corridor_cover": "static_lo_choice_corridor",
    "lo_choice_capacity_projection": "static_lo_choice_capacity",
    "queue_perspective_support": "static_queue_perspective",
    "nearest_display_suffix": "static_nearest_display_suffix",
    "chi_od_corridor_vub_top_k": "static_chi_vub_top_k_per_corridor",
}
_USER_FAMILY_CONTROLS = {
    "chi_od_corridor_vub": "cut_chi_vub",
    "chi_od_incident_star_vub": "cut_chi_facility_vub",
    "chi_od_head_facility_vub": "cut_chi_facility_vub",
    "route_swap_leg_bundle_vub": "cut_route_swap_leg_vub",
    "route_swap_station_throughput_vub": "cut_route_swap_station_vub",
    "queue_perspective": "cut_queue_perspective",
    "lo_choice_corridor": "cut_lo_choice_corridor",
    "lo_choice_capacity_cover": "cut_lo_choice_capacity",
    "directed_return": "cut_directed_return",
    "od_disaggregated_directed_return": "cut_od_disaggregated_return",
}


def _fail(message: str) -> None:
    raise FormalMetricValidationError(message)


def _finite_number(
    value: object,
    label: str,
    *,
    nullable: bool = False,
    nonnegative: bool = False,
) -> float | None:
    if value is None and nullable:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(f"{label} must be a finite number" + (" or null" if nullable else ""))
    number = float(value)
    if not math.isfinite(number) or abs(number) >= 0.99e100:
        _fail(f"{label} must be a finite Gurobi observation")
    if nonnegative and number < 0.0:
        _fail(f"{label} must be nonnegative")
    return number


def _nonnegative_integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        _fail(f"{label} must be a nonnegative integer")
    return value


def _boolean(value: object, label: str) -> bool:
    if type(value) is not bool:
        _fail(f"{label} must be boolean")
    return value


def _tolerance(*values: float) -> float:
    return 1e-8 + 1e-7 * max((1.0, *(abs(value) for value in values)))


def _runtime_tolerance(*values: float) -> float:
    return 0.05 + 1e-3 * max((1.0, *(abs(value) for value in values)))


def _numbers_match(left: float | None, right: float | None) -> bool:
    if left is None or right is None:
        return left is right
    return abs(left - right) <= _tolerance(left, right)


def _validate_gap(
    incumbent: float | None,
    bound: float | None,
    raw_gap: object,
    *,
    label: str,
) -> float | None:
    gap = _finite_number(raw_gap, label, nullable=True, nonnegative=True)
    if incumbent is None or bound is None:
        if gap is not None:
            _fail(f"{label} must be null when incumbent or bound is unavailable")
        return None
    if bound > incumbent + _tolerance(bound, incumbent):
        _fail(f"{label} has a minimization bound above the incumbent")
    expected = abs(incumbent - bound) / max(1e-10, abs(incumbent))
    if gap is None or abs(gap - expected) > _tolerance(gap or 0.0, expected):
        _fail(f"{label} differs from objective and best bound")
    return gap


def _validate_family_count_mapping(value: object, label: str) -> dict[str, int]:
    if not isinstance(value, Mapping):
        _fail(f"{label} must be a mapping")
    result: dict[str, int] = {}
    for raw_family, raw_count in value.items():
        if not isinstance(raw_family, str) or not raw_family.strip():
            _fail(f"{label} contains an invalid family name")
        result[raw_family] = _nonnegative_integer(
            raw_count,
            f"{label}[{raw_family}]",
        )
    return result


def _validate_family_seconds_mapping(
    value: object,
    label: str,
) -> dict[str, float]:
    if not isinstance(value, Mapping):
        _fail(f"{label} must be a mapping")
    result: dict[str, float] = {}
    for raw_family, raw_seconds in value.items():
        if not isinstance(raw_family, str) or not raw_family.strip():
            _fail(f"{label} contains an invalid family name")
        seconds = _finite_number(
            raw_seconds,
            f"{label}[{raw_family}]",
            nonnegative=True,
        )
        assert seconds is not None
        result[raw_family] = seconds
    return result


def _enabled_families(
    controls: Mapping[str, str],
    core_solver_config: Mapping[str, Any],
) -> frozenset[str]:
    enabled: set[str] = set()
    for family, control in controls.items():
        value = core_solver_config.get(control)
        if control == "static_chi_vub_top_k_per_corridor":
            if _nonnegative_integer(value, f"core_solver_config.{control}") > 0:
                enabled.add(family)
            continue
        if _boolean(value, f"core_solver_config.{control}"):
            enabled.add(family)
    return frozenset(enabled)


def _validate_solver_metrics(
    run_record: Mapping[str, Any],
    *,
    completed: bool,
) -> dict[str, Any]:
    metrics = run_record.get("solver_metrics")
    if not isinstance(metrics, Mapping):
        _fail("solver_metrics must be a mapping")
    if not completed:
        # Failed attempts are excluded from formal effects.  Preserve their
        # canonical nullable telemetry while still requiring the incumbent flag
        # already used by the journal state machine.
        _boolean(metrics.get("has_incumbent"), "solver_metrics.has_incumbent")
        return dict(metrics)

    missing = sorted(_REQUIRED_COMPLETED_SOLVER_METRICS - set(metrics))
    if missing:
        _fail("completed solver_metrics are missing: " + ", ".join(missing))

    status = metrics["status"]
    if isinstance(status, bool) or not isinstance(status, int) or status not in _STATUS_NAMES:
        _fail("solver_metrics.status must be a recognized Gurobi status integer")
    status_name = metrics["status_name"]
    if status_name != _STATUS_NAMES[status]:
        _fail("solver_metrics.status_name differs from status")
    if status_name == "INPROGRESS":
        _fail("a completed formal solve cannot retain INPROGRESS status")

    has_incumbent = _boolean(
        metrics["has_incumbent"],
        "solver_metrics.has_incumbent",
    )
    solution_count = _nonnegative_integer(
        metrics["solution_count"],
        "solver_metrics.solution_count",
    )
    if has_incumbent != (solution_count > 0):
        _fail("solver_metrics incumbent flag differs from solution_count")
    if status_name in _NO_INCUMBENT_STATUSES and has_incumbent:
        _fail(f"solver status {status_name} cannot carry an incumbent")
    if status_name in _INCUMBENT_REQUIRED_STATUSES and not has_incumbent:
        _fail(f"solver status {status_name} requires an incumbent")

    objective = _finite_number(
        metrics["objective"],
        "solver_metrics.objective",
        nullable=True,
    )
    if has_incumbent != (objective is not None):
        _fail("solver_metrics objective presence differs from incumbent state")
    bound = _finite_number(
        metrics["best_bound"],
        "solver_metrics.best_bound",
        nullable=True,
    )
    gap = _validate_gap(
        objective,
        bound,
        metrics["relative_gap"],
        label="solver_metrics.relative_gap",
    )
    node_count = _finite_number(
        metrics["node_count"],
        "solver_metrics.node_count",
        nullable=True,
        nonnegative=True,
    )

    runtime_names = (
        "main_solve_runtime_seconds",
        "total_algorithm_runtime_seconds",
        "algorithm_prepare_runtime_seconds",
        "start_preparation_runtime_seconds",
        "callback_runtime_seconds",
    )
    runtimes: dict[str, float] = {}
    for name in runtime_names:
        value = _finite_number(
            metrics[name],
            f"solver_metrics.{name}",
            nonnegative=True,
        )
        assert value is not None
        runtimes[name] = value
    main_runtime = runtimes["main_solve_runtime_seconds"]
    total_runtime = runtimes["total_algorithm_runtime_seconds"]
    if main_runtime > total_runtime + _runtime_tolerance(main_runtime, total_runtime):
        _fail("main solve runtime exceeds total algorithm runtime")
    charged_components = (
        main_runtime
        + runtimes["algorithm_prepare_runtime_seconds"]
        + runtimes["start_preparation_runtime_seconds"]
    )
    if charged_components > total_runtime + _runtime_tolerance(
        charged_components,
        total_runtime,
    ):
        _fail("charged runtime components exceed total algorithm runtime")
    if runtimes["callback_runtime_seconds"] > total_runtime + _runtime_tolerance(
        runtimes["callback_runtime_seconds"],
        total_runtime,
    ):
        _fail("callback runtime exceeds total algorithm runtime")

    first_time = _finite_number(
        metrics["first_incumbent_time_seconds"],
        "solver_metrics.first_incumbent_time_seconds",
        nullable=True,
        nonnegative=True,
    )
    first_objective = _finite_number(
        metrics["first_incumbent_objective"],
        "solver_metrics.first_incumbent_objective",
        nullable=True,
    )
    if has_incumbent:
        if first_time is None or first_objective is None:
            _fail("an incumbent requires finite time-to-first-incumbent metrics")
        if first_time > main_runtime + _runtime_tolerance(first_time, main_runtime):
            _fail("time to first incumbent exceeds main solve runtime")
        assert objective is not None
        if first_objective < objective - _tolerance(first_objective, objective):
            _fail("first incumbent is better than the final minimization incumbent")
    elif first_time is not None or first_objective is not None:
        _fail("a no-incumbent solve must have null first-incumbent metrics")

    root_before = _finite_number(
        metrics["root_bound_before_cuts"],
        "solver_metrics.root_bound_before_cuts",
        nullable=True,
    )
    root_after = _finite_number(
        metrics["root_bound_after_cuts"],
        "solver_metrics.root_bound_after_cuts",
        nullable=True,
    )
    if (root_before is None) != (root_after is None):
        _fail("root bounds before and after cuts must be jointly present or null")
    if (
        root_before is not None
        and root_after is not None
        and root_after < root_before - _tolerance(root_before, root_after)
    ):
        _fail("root bound after cuts is worse for a minimization problem")

    budget_exhausted = _boolean(
        metrics["budget_exhausted_before_main_solve"],
        "solver_metrics.budget_exhausted_before_main_solve",
    )
    main_started = _boolean(
        metrics["main_solve_started"],
        "solver_metrics.main_solve_started",
    )
    optimize_called = _boolean(
        metrics["optimize_called"],
        "solver_metrics.optimize_called",
    )
    if main_started != optimize_called:
        _fail("main_solve_started differs from optimize_called")
    if optimize_called:
        if budget_exhausted:
            _fail("an executed main solve cannot be marked exhausted before start")
        if status_name == "LOADED":
            _fail("an executed main solve cannot finish with LOADED status")
        if node_count is None:
            _fail("an executed main solve requires a finite node count")
    else:
        if not budget_exhausted:
            _fail("a skipped main solve must exhaust its budget before start")
        if status_name != "LOADED":
            _fail("a skipped main solve must retain LOADED status")
        if has_incumbent or bound is not None or gap is not None or node_count is not None:
            _fail("a skipped main solve carries stale solver outcomes")
        if main_runtime != 0.0:
            _fail("a skipped main solve must have zero main-solve runtime")
        if runtimes["callback_runtime_seconds"] != 0.0:
            _fail("a skipped main solve must have zero callback runtime")

    execution_wall = _finite_number(
        run_record.get("execution_wall_runtime_seconds"),
        "execution_wall_runtime_seconds",
        nonnegative=True,
    )
    assert execution_wall is not None
    if total_runtime > execution_wall + _runtime_tolerance(total_runtime, execution_wall):
        _fail("total algorithm runtime exceeds execution wall runtime")
    solve_returned = _boolean(
        run_record.get("solve_returned_incumbent"),
        "solve_returned_incumbent",
    )
    if solve_returned != has_incumbent:
        _fail("solve_returned_incumbent differs from the solver incumbent state")
    for name, expected in (
        ("budget_exhausted_before_main_solve", budget_exhausted),
        ("main_solve_started", main_started),
        ("optimize_called", optimize_called),
    ):
        if run_record.get(name) is not expected:
            _fail(f"run_record {name} differs from solver_metrics")

    return {
        "status": status,
        "status_name": status_name,
        "has_incumbent": has_incumbent,
        "solution_count": solution_count,
        "objective": objective,
        "bound": bound,
        "gap": gap,
        "node_count": node_count,
        "main_runtime": main_runtime,
        "total_runtime": total_runtime,
        "first_time": first_time,
        "root_before": root_before,
        "root_after": root_after,
        "callback_runtime": runtimes["callback_runtime_seconds"],
        "budget_exhausted": budget_exhausted,
        "main_started": main_started,
        "optimize_called": optimize_called,
    }


def _validate_progress(
    rows: Sequence[Mapping[str, Any]],
    solver: Mapping[str, Any],
    *,
    completed: bool,
) -> None:
    previous_runtime: float | None = None
    previous_node_count: float | None = None
    observed_incumbent_times: list[float] = []
    final_rows: list[tuple[int, Mapping[str, Any]]] = []

    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            _fail(f"progress row {index} must be a mapping")
        event = row.get("event")
        if event not in _PROGRESS_FIELDS:
            _fail(f"progress row {index} has an unknown event type")
        expected_fields = _PROGRESS_FIELDS[str(event)]
        if set(row) != expected_fields:
            _fail(f"progress row {index} fields differ for event {event}")
        runtime = _finite_number(
            row["runtime_seconds"],
            f"progress row {index} runtime_seconds",
            nonnegative=True,
        )
        assert runtime is not None
        if previous_runtime is not None and runtime < previous_runtime - _runtime_tolerance(
            runtime,
            previous_runtime,
        ):
            _fail("progress runtimes are not nondecreasing")
        previous_runtime = runtime

        incumbent = _finite_number(
            row["incumbent"],
            f"progress row {index} incumbent",
            nullable=True,
        )
        bound = _finite_number(
            row["bound"],
            f"progress row {index} bound",
            nullable=True,
        )
        _validate_gap(
            incumbent,
            bound,
            row["relative_gap"],
            label=f"progress row {index} relative_gap",
        )
        if incumbent is not None:
            observed_incumbent_times.append(runtime)
        if event == "incumbent" and incumbent is None:
            _fail(f"progress incumbent row {index} lacks an incumbent")

        if event in {"mip_progress", "final"}:
            node_count = _finite_number(
                row["node_count"],
                f"progress row {index} node_count",
                nonnegative=True,
            )
            assert node_count is not None
            if previous_node_count is not None and node_count < (
                previous_node_count - _tolerance(node_count, previous_node_count)
            ):
                _fail("progress node counts are not nondecreasing")
            previous_node_count = node_count
            _nonnegative_integer(
                row["solution_count"],
                f"progress row {index} solution_count",
            )

        if event == "final":
            final_rows.append((index, row))
            _finite_number(
                row["total_algorithm_runtime_seconds"],
                f"progress row {index} total_algorithm_runtime_seconds",
                nonnegative=True,
            )
            for marker in (
                "budget_exhausted_before_main_solve",
                "main_solve_started",
                "optimize_called",
            ):
                _boolean(row[marker], f"progress row {index} {marker}")
            status = row["status"]
            if (
                isinstance(status, bool)
                or not isinstance(status, int)
                or status not in _STATUS_NAMES
            ):
                _fail(f"progress row {index} has an invalid final status")

    if not completed:
        if len(final_rows) > 1:
            _fail("failed attempt progress contains multiple final rows")
        return
    if len(final_rows) != 1 or final_rows[0][0] != len(rows) - 1:
        _fail("completed attempt requires exactly one trailing final progress row")

    final = final_rows[0][1]
    final_runtime = float(final["runtime_seconds"])
    final_total = float(final["total_algorithm_runtime_seconds"])
    comparisons = (
        (final_runtime, solver["main_runtime"], "final main runtime"),
        (final_total, solver["total_runtime"], "final total runtime"),
        (
            _finite_number(final["incumbent"], "final incumbent", nullable=True),
            solver["objective"],
            "final incumbent",
        ),
        (
            _finite_number(final["bound"], "final bound", nullable=True),
            solver["bound"],
            "final bound",
        ),
        (
            _finite_number(
                final["relative_gap"],
                "final relative_gap",
                nullable=True,
            ),
            solver["gap"],
            "final relative gap",
        ),
    )
    for observed, expected, label in comparisons:
        if not _numbers_match(observed, expected):
            _fail(f"{label} differs from solver_metrics")
    if final["status"] != solver["status"]:
        _fail("final progress status differs from solver_metrics")
    if final["solution_count"] != solver["solution_count"]:
        _fail("final progress solution count differs from solver_metrics")
    if solver["optimize_called"]:
        final_nodes = float(final["node_count"])
        if not _numbers_match(final_nodes, solver["node_count"]):
            _fail("final progress node count differs from solver_metrics")
    elif float(final["node_count"]) != 0.0:
        _fail("skipped main solve final progress must report zero nodes")
    marker_keys = {
        "budget_exhausted_before_main_solve": "budget_exhausted",
        "main_solve_started": "main_started",
        "optimize_called": "optimize_called",
    }
    for marker, solver_key in marker_keys.items():
        if final[marker] is not solver[solver_key]:
            _fail(f"final progress {marker} differs from solver_metrics")

    if solver["has_incumbent"]:
        if not observed_incumbent_times:
            _fail("incumbent solve has no incumbent-bearing progress event")
        first_time = solver["first_time"]
        assert first_time is not None
        earliest = min(observed_incumbent_times)
        if first_time > earliest + _runtime_tolerance(first_time, earliest):
            _fail("time to first incumbent follows the first observed incumbent")
    elif observed_incumbent_times:
        _fail("no-incumbent solve has incumbent-bearing progress events")


def _validate_cut_stats(
    cut_stats: Mapping[str, Any],
    core_solver_config: Mapping[str, Any],
    *,
    completed: bool,
) -> dict[str, Any]:
    captured = _boolean(cut_stats.get("captured"), "cut_stats.captured")
    if not captured:
        if set(cut_stats) != {"captured"}:
            _fail("uncaptured cut_stats must contain only captured=false")
        if completed:
            _fail("completed formal attempt requires captured cut statistics")
        return {
            "captured": False,
            "static_total": None,
            "user_total": None,
            "root_rounds": 0,
            "callback_runtime": None,
            "user_added": {},
        }
    if set(cut_stats) != _CUT_STATS_FIELDS:
        _fail("captured cut_stats fields differ from the formal schema")

    static_added = _validate_family_count_mapping(
        cut_stats["static_rows_added_by_family"],
        "cut_stats.static_rows_added_by_family",
    )
    user_added = _validate_family_count_mapping(
        cut_stats["user_cuts_added_by_family"],
        "cut_stats.user_cuts_added_by_family",
    )
    candidates = _validate_family_count_mapping(
        cut_stats["user_cut_candidates_by_family"],
        "cut_stats.user_cut_candidates_by_family",
    )
    duplicates = _validate_family_count_mapping(
        cut_stats["user_cut_duplicates_by_family"],
        "cut_stats.user_cut_duplicates_by_family",
    )
    rejections = _validate_family_count_mapping(
        cut_stats["user_cut_rejections_by_family"],
        "cut_stats.user_cut_rejections_by_family",
    )
    separation = _validate_family_seconds_mapping(
        cut_stats["user_cut_separation_seconds_by_family"],
        "cut_stats.user_cut_separation_seconds_by_family",
    )
    route_counts = _validate_family_count_mapping(
        cut_stats["route_bundle_index_counts"],
        "cut_stats.route_bundle_index_counts",
    )
    if set(route_counts) != _ROUTE_BUNDLE_INDEX_FIELDS:
        _fail("route_bundle_index_counts fields differ from the formal schema")

    static_total = _nonnegative_integer(
        cut_stats["static_rows_added_total"],
        "cut_stats.static_rows_added_total",
    )
    user_total = _nonnegative_integer(
        cut_stats["user_cuts_added_total"],
        "cut_stats.user_cuts_added_total",
    )
    root_rounds = _nonnegative_integer(
        cut_stats["root_cut_rounds"],
        "cut_stats.root_cut_rounds",
    )
    callback_runtime = _finite_number(
        cut_stats["callback_runtime_seconds"],
        "cut_stats.callback_runtime_seconds",
        nonnegative=True,
    )
    assert callback_runtime is not None
    if static_total != sum(static_added.values()):
        _fail("static_rows_added_total differs from its family counts")
    if user_total != sum(user_added.values()):
        _fail("user_cuts_added_total differs from its family counts")

    enabled_static = _enabled_families(
        _STATIC_FAMILY_CONTROLS,
        core_solver_config,
    )
    if not set(static_added) <= enabled_static:
        _fail("static cut statistics contain a disabled or unknown family")
    enabled_user = _enabled_families(
        _USER_FAMILY_CONTROLS,
        core_solver_config,
    )
    observed_user_families = (
        set(user_added)
        | set(candidates)
        | set(duplicates)
        | set(rejections)
        | set(separation)
    )
    if not observed_user_families <= enabled_user:
        _fail("user-cut statistics contain a disabled or unknown family")
    use_user_cuts = _boolean(
        core_solver_config.get("use_user_cuts"),
        "core_solver_config.use_user_cuts",
    )
    if not use_user_cuts and (observed_user_families or user_total or root_rounds):
        _fail("user-cut-off profile contains user-cut activity")

    for family in observed_user_families:
        candidate_count = candidates.get(family, 0)
        accounted = (
            user_added.get(family, 0)
            + duplicates.get(family, 0)
            + rejections.get(family, 0)
        )
        if candidate_count < accounted:
            _fail(f"user-cut candidate count undercounts family {family}")
    if sum(separation.values()) > callback_runtime + _runtime_tolerance(
        sum(separation.values()),
        callback_runtime,
    ):
        _fail("user-cut separation time exceeds callback runtime")

    return {
        "captured": True,
        "static_total": static_total,
        "user_total": user_total,
        "root_rounds": root_rounds,
        "callback_runtime": callback_runtime,
        "user_added": user_added,
    }


def _validate_root_rounds(
    rows: Sequence[Mapping[str, Any]],
    cut_summary: Mapping[str, Any],
    solver: Mapping[str, Any],
    core_solver_config: Mapping[str, Any],
) -> None:
    if not cut_summary["captured"]:
        if rows:
            _fail("uncaptured cut statistics cannot have root-round rows")
        return
    if len(rows) != cut_summary["root_rounds"]:
        _fail("root-round row count differs from cut_stats.root_cut_rounds")
    max_rounds = _nonnegative_integer(
        core_solver_config.get("max_root_cut_rounds"),
        "core_solver_config.max_root_cut_rounds",
    )
    max_cuts = _nonnegative_integer(
        core_solver_config.get("max_cuts_per_round"),
        "core_solver_config.max_cuts_per_round",
    )
    if len(rows) > max_rounds:
        _fail("root-round telemetry exceeds the frozen round budget")

    previous_runtime: float | None = None
    added_by_family: Counter[str] = Counter()
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, Mapping) or set(row) != _ROOT_ROUND_FIELDS:
            _fail(f"root-round row {index} fields differ from the formal schema")
        if row["round"] != index:
            _fail("root-round indices must be contiguous and one-based")
        runtime = _finite_number(
            row["runtime_seconds"],
            f"root-round row {index} runtime_seconds",
            nonnegative=True,
        )
        assert runtime is not None
        if previous_runtime is not None and runtime < previous_runtime - _runtime_tolerance(
            runtime,
            previous_runtime,
        ):
            _fail("root-round runtimes are not nondecreasing")
        previous_runtime = runtime
        if (
            solver.get("main_runtime") is not None
            and runtime
            > float(solver["main_runtime"])
            + _runtime_tolerance(runtime, float(solver["main_runtime"]))
        ):
            _fail("root-round runtime exceeds main solve runtime")
        root_bound = _finite_number(
            row["root_bound_before_round"],
            f"root-round row {index} root_bound_before_round",
            nullable=True,
        )
        if (
            index == 1
            and "root_before" in solver
            and not _numbers_match(root_bound, solver["root_before"])
        ):
            _fail("first root-round bound differs from solver root-before metric")
        cuts_added = _nonnegative_integer(
            row["cuts_added"],
            f"root-round row {index} cuts_added",
        )
        if cuts_added > max_cuts:
            _fail("root-round cuts exceed the frozen per-round budget")
        families = _validate_family_count_mapping(
            row["cuts_added_by_family"],
            f"root-round row {index} cuts_added_by_family",
        )
        if any(count == 0 for count in families.values()):
            _fail("root-round family mappings must omit zero counts")
        if sum(families.values()) != cuts_added:
            _fail("root-round cuts_added differs from family counts")
        added_by_family.update(families)
    if dict(added_by_family) != dict(cut_summary["user_added"]):
        _fail("root-round family totals differ from cut_stats")


def _validate_telemetry_counts(
    run_record: Mapping[str, Any],
    progress: Sequence[Mapping[str, Any]],
    root_rounds: Sequence[Mapping[str, Any]],
    cut_summary: Mapping[str, Any],
    *,
    completed: bool,
) -> None:
    raw = run_record.get("telemetry_counts")
    if not isinstance(raw, Mapping) or set(raw) != _TELEMETRY_COUNT_FIELDS:
        if completed:
            _fail("completed run_record telemetry_counts differ from the schema")
        return
    expected = {
        "progress_events": len(progress),
        "root_round_events": len(root_rounds),
        "static_rows_added": cut_summary["static_total"],
        "user_cuts_added": cut_summary["user_total"],
    }
    if dict(raw) != expected:
        _fail("run_record telemetry_counts differ from telemetry artifacts")


def validate_formal_attempt_paper_metrics(
    *,
    run_record: Mapping[str, Any],
    progress: Sequence[Mapping[str, Any]],
    root_rounds: Sequence[Mapping[str, Any]],
    cut_stats: Mapping[str, Any],
    core_solver_config: Mapping[str, Any],
    completed: bool,
) -> None:
    """Validate all paper metrics and raw telemetry as one formal observation.

    Completed runs are strict: every paper-facing metric is required and must
    agree with final progress, root-round, cut-family, callback, and runtime
    evidence.  Failed runs remain representable and auditable, but are not
    upgraded into complete paper observations.
    """

    if type(completed) is not bool:
        _fail("completed must be boolean")
    if not isinstance(run_record, Mapping):
        _fail("run_record must be a mapping")
    if not isinstance(core_solver_config, Mapping):
        _fail("core_solver_config must be a mapping")
    if not isinstance(cut_stats, Mapping):
        _fail("cut_stats must be a mapping")
    if not isinstance(progress, Sequence) or isinstance(progress, (str, bytes)):
        _fail("progress must be a sequence of mappings")
    if not isinstance(root_rounds, Sequence) or isinstance(root_rounds, (str, bytes)):
        _fail("root_rounds must be a sequence of mappings")

    solver = _validate_solver_metrics(run_record, completed=completed)
    _validate_progress(progress, solver, completed=completed)
    cut_summary = _validate_cut_stats(
        cut_stats,
        core_solver_config,
        completed=completed,
    )
    if completed and not _numbers_match(
        cut_summary["callback_runtime"],
        solver["callback_runtime"],
    ):
        _fail("cut_stats callback runtime differs from solver_metrics")
    _validate_root_rounds(
        root_rounds,
        cut_summary,
        solver,
        core_solver_config,
    )
    _validate_telemetry_counts(
        run_record,
        progress,
        root_rounds,
        cut_summary,
        completed=completed,
    )


__all__ = [
    "FormalMetricValidationError",
    "validate_formal_attempt_paper_metrics",
]
