"""Benders run and optional incumbent improvement within one solve budget."""

import time
from gurobipy import GRB


def gap_met(result, mip_gap, mip_gap_abs):
    return (
        not result.get("errors")
        and result.get("objective") is not None
        and result.get("bound") is not None
        and result["objective"] - result["bound"]
        <= max(mip_gap_abs, mip_gap * abs(result["objective"]))
    )


def solve_benders(
    data,
    method,
    *,
    time_limit,
    threads,
    seed,
    mip_gap,
    mip_gap_abs,
    max_iterations,
    log_dir,
    progress,
    checkpoint,
    stop_requested,
):
    from .engine import solve
    from ..heuristic import search
    from ..heuristic.refinement import refine, update_incumbent
    from ..pricing.backend import PricingStopped

    started = time.perf_counter()
    common = dict(threads=threads, seed=seed, mip_gap=mip_gap, mip_gap_abs=mip_gap_abs)
    extras = method.accelerations.primal_search and time_limit is not None

    def stopped():
        return stop_requested is not None and stop_requested()

    def remaining():
        return (
            float("inf")
            if time_limit is None
            else max(0.0, time_limit - (time.perf_counter() - started))
        )

    def completed():
        return (
            stopped() or result.get("errors") or gap_met(result, mip_gap, mip_gap_abs)
        )

    def record(stage_result):
        return {
            key: stage_result.get(key)
            for key in (
                "wall",
                "budget",
                "objective",
                "bound",
                "status",
                "termination_reason",
                "params",
            )
        }

    benders_budget = 0.65 * time_limit if extras else time_limit
    result = solve(
        data,
        time_limit=benders_budget,
        max_iterations=max_iterations,
        pricing=method.pricing,
        **method.accelerations.benders_options(),
        log=log_dir / "benders.log" if log_dir else None,
        progress=progress,
        checkpoint=checkpoint,
        stop_requested=stop_requested,
        **common,
    )
    stages = {"benders": record(result)}
    if extras and remaining() > 1.0 and not completed():
        try:
            improved = search.solve(
                data,
                result,
                time_limit=remaining() * 5 / 7,
                global_lower_bound=result["bound"],
                pricing=method.pricing,
                progress=progress,
                stop_requested=stop_requested,
                log=log_dir / "design_search.log" if log_dir else None,
                **common,
            )
            stages["design_search"] = dict(
                record(improved), statistics=improved.get("heuristic_stats")
            )
            result = update_incumbent(result, improved)
        except PricingStopped as error:
            stages["design_search"] = {
                "accepted": False,
                "termination_reason": str(error),
            }
        if checkpoint:
            checkpoint(result)
    if extras and remaining() > 1.0 and not completed():
        result, stages["refinement"] = refine(
            data,
            result,
            seconds=remaining(),
            threads=threads,
            seed=seed,
            pricing=method.pricing,
            progress=progress,
            stop_requested=stop_requested,
        )
        if checkpoint:
            checkpoint(result)
    result.update(
        stages=stages,
        wall=time.perf_counter() - started,
        budget=time_limit,
        primal_search_scheduled=extras,
    )
    result["params"] = dict(result.get("params", {}), TimeLimit=time_limit)
    if stopped():
        result.update(termination_reason="user_stop", status=GRB.INTERRUPTED)
    elif gap_met(result, mip_gap, mip_gap_abs):
        result.update(termination_reason="gap_limit", status=GRB.OPTIMAL)
    elif extras and not result.get("errors"):
        exhausted = remaining() <= 0.0
        result.update(
            termination_reason="time_limit" if exhausted else "stages_finished",
            status=GRB.TIME_LIMIT if exhausted else GRB.SUBOPTIMAL,
        )
    return result
