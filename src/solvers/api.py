"""Public solve interface for the sole consecutive model."""

from __future__ import annotations

from src.model.assignment import assignment_scope
import math
import time
from src.model.types import PlanningData, MODEL_VERSION
from src.model.paths import enumerate_paths, EnumerationStopped
from .algorithm import baseline, solve_full_path, solve_benders
from .methods import resolve_accelerations


def solve(
    data,
    *,
    method="benders_cg",
    time_limit=600.0,
    threads=1,
    seed=43,
    mip_gap=1e-4,
    progress=None,
    checkpoint=None,
    stop_requested=None,
    column_limit=500_000,
    max_iterations=1000,
    accelerations=None,
    **kwargs,
):
    if not isinstance(data, PlanningData):
        raise TypeError("solve requires consecutive PlanningData")
    resolved_accelerations = resolve_accelerations(method, accelerations)
    if kwargs:
        raise TypeError(f"Unknown solver options: {', '.join(kwargs)}")
    if time_limit is not None and (not math.isfinite(time_limit) or time_limit < 0):
        raise ValueError("time_limit must be finite and nonnegative, or None")
    if not math.isfinite(mip_gap) or mip_gap < 0:
        raise ValueError("mip_gap must be finite and nonnegative")
    if isinstance(threads, bool) or int(threads) != threads or threads < 1:
        raise ValueError("threads must be a positive integer")
    if (
        isinstance(max_iterations, bool)
        or int(max_iterations) != max_iterations
        or max_iterations < 1
    ):
        raise ValueError("max_iterations must be a positive integer")
    started = time.perf_counter()
    scope = "supplied" if data.candidate_routes is not None else data.route_mode
    result = baseline(data)
    if progress:
        progress(dict(stage="enumeration_started", path_scope=scope))
    enumeration_started = time.perf_counter()
    try:
        paths = enumerate_paths(
            data,
            column_limit=column_limit,
            time_limit=time_limit,
            stop_requested=stop_requested,
        )
    except EnumerationStopped as exc:
        enumeration_only_seconds = time.perf_counter() - enumeration_started
        enumeration_seconds = time.perf_counter() - started
        result.update(
            termination_reason=exc.reason,
            catalogue_complete=False,
            construction_complete=False,
            pricing_complete=False,
            enumerated_columns=exc.columns,
            enumerated_segments=exc.initial_segments,
        )
    else:
        enumeration_only_seconds = time.perf_counter() - enumeration_started
        enumeration_seconds = time.perf_counter() - started
        if progress:
            progress(
                dict(
                    stage="enumeration_complete",
                    enumeration_seconds=enumeration_seconds,
                    segments=len(paths.initial_segments),
                    service_columns=len(paths.service_paths),
                    empty_columns=len(paths.empty_paths),
                )
            )
        remaining = (
            None
            if time_limit is None
            else max(0.0, time_limit - (time.perf_counter() - started))
        )
        options = dict(
            time_limit=remaining,
            threads=threads,
            seed=seed,
            mip_gap=mip_gap,
            progress=progress,
            checkpoint=checkpoint,
            stop_requested=stop_requested,
        )
        if method == "full_path":
            result = solve_full_path(data, paths, **options)
        else:
            result = solve_benders(
                data,
                paths,
                accelerated=method == "benders_cg",
                accelerations=resolved_accelerations,
                max_iterations=max_iterations,
                **options,
            )
        result.setdefault("construction_complete", True)
        result.update(
            catalogue_complete=True,
            catalogue=dict(
                segments=len(paths.initial_segments),
                service=len(paths.service_paths),
                empty=len(paths.empty_paths),
            ),
        )
        scope = paths.scope
    result.update(
        model_version=MODEL_VERSION,
        method=method,
        accelerations=resolved_accelerations,
        path_scope=scope,
        pricing_implementation="complete_finite_catalogue_scan",
        wall=time.perf_counter() - started,
        budget=time_limit,
        max_orders=data.max_orders,
        flow_policy=data.flow_policy,
        assignment_scope=assignment_scope(data),
        effective_delta=data.effective_delta,
        enumeration_seconds=enumeration_seconds,
        enumeration_only_seconds=enumeration_only_seconds,
        solve_seconds=max(0.0, time.perf_counter() - started - enumeration_seconds),
    )
    if progress:
        progress(
            dict(
                stage="finished",
                objective=result["objective"],
                bound=result["bound"],
                gap=result["gap"],
                termination_reason=result["termination_reason"],
                pricing_complete=result["pricing_complete"],
            )
        )
    result["solver_wall"] = time.perf_counter() - started
    return result
