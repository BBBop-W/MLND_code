"""Experiment strategies with one recorded budget and complete-model bounds."""

import time

from gurobipy import GRB
from src.optimization import benders, direct, heuristic
from src.optimization.column_pool import build_result
from src.optimization.settings import DEFAULT_MIP_GAP, DEFAULT_MIP_GAP_ABS
from .refinement import refine, update_incumbent


def _gap_met(result, options):
    return (
        not result.get("errors")
        and result.get("objective") is not None
        and result.get("bound") is not None
        and result["objective"] - result["bound"]
        <= max(options["mip_gap_abs"], options["mip_gap"] * abs(result["objective"]))
    )


def solve(data, method, options, output, monitor):
    started = time.perf_counter()
    seconds = options["time_limit"]
    gap = options.get("mip_gap")
    abs_gap = options.get("mip_gap_abs")
    common = dict(
        threads=options["threads"],
        seed=options.get("seed", 43),
        mip_gap=DEFAULT_MIP_GAP if gap is None else gap,
        mip_gap_abs=DEFAULT_MIP_GAP_ABS if abs_gap is None else abs_gap,
    )
    if method == "direct":
        # Reference enumeration grows with OD, base pairs, lockers and swaps.
        count = (
            data["n_od"]
            * data["n_l"] ** 2
            * data["n_h"]
            * (data["n_h"] - 1)
            * (data["n_k"] + 1) ** 3
        )
        if count > 500_000:
            raise ValueError(
                f"Direct reference would examine up to {count:,} service paths; use benders or hybrid"
            )
        monitor.stage = "direct"
        raw = direct.solve(data, time_limit=seconds, **common)
        if raw.get("objective") is None:
            return raw
        solution = {
            key: raw[key]
            for key in (
                "facility",
                "capacity_level",
                "assignment",
                "airline",
                "rho",
                "idle",
                "fleet",
            )
        }
        result = build_result(
            data, solution, raw["missions"], raw["relocations"], raw["bound"]
        )
        result.update(
            {
                key: raw[key]
                for key in (
                    "status",
                    "method",
                    "params",
                    "service_columns",
                    "empty_columns",
                )
            }
        )
        reason = {
            GRB.OPTIMAL: "gap_limit",
            GRB.TIME_LIMIT: "time_limit",
            GRB.INTERRUPTED: "user_stop",
        }.get(raw["status"], "solver_stopped")
        result.update(
            termination_reason=reason,
            budget=seconds,
            wall=time.perf_counter() - started,
        )
        monitor.checkpoint(result)
        return result
    if method not in ("benders", "hybrid"):
        raise ValueError(f"Unknown method: {method}")
    if method == "hybrid" and seconds is None:
        raise ValueError("Hybrid needs a finite budget to allocate its three stages")
    benders_budget = seconds * 0.65 if method == "hybrid" else seconds
    result = benders.solve(
        data,
        time_limit=benders_budget,
        max_iterations=options.get("max_iterations"),
        log=output / "solver.log",
        progress=monitor.progress,
        checkpoint=monitor.checkpoint,
        stop_requested=monitor.stop_requested,
        **common,
    )
    stages = {
        "benders": {
            "wall": time.perf_counter() - started,
            "budget": benders_budget,
            **{
                key: result.get(key)
                for key in (
                    "objective",
                    "bound",
                    "status",
                    "termination_reason",
                    "params",
                )
            },
        }
    }
    if method == "hybrid" and not monitor.stop_requested() and not result.get("errors"):
        remaining = seconds - (time.perf_counter() - started)
        if remaining > 1.0 and not _gap_met(result, common):
            monitor.stage = "design_search"
            improved = heuristic.solve(
                data,
                result,
                time_limit=remaining * 5 / 7,
                global_lower_bound=result["bound"],
                progress=monitor.progress,
                log=output / "design_search.log",
                stop_requested=monitor.stop_requested,
                **common,
            )
            stages["design_search"] = {
                key: improved.get(key)
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
            stages["design_search"]["statistics"] = improved.get("heuristic_stats")
            result = update_incumbent(result, improved)
            monitor.checkpoint(result)
        remaining = seconds - (time.perf_counter() - started)
        if (
            remaining > 1.0
            and not monitor.stop_requested()
            and not _gap_met(result, common)
        ):
            monitor.stage = "refinement"
            result, stages["refinement"] = refine(
                data,
                result,
                seconds=remaining,
                threads=common["threads"],
                seed=common["seed"],
                monitor=monitor,
            )
            monitor.checkpoint(result)
    result.update(
        method=method, stages=stages, wall=time.perf_counter() - started, budget=seconds
    )
    result["params"] = dict(result.get("params", {}), TimeLimit=seconds)
    if monitor.stop_requested():
        result.update(termination_reason="user_stop", status=GRB.INTERRUPTED)
    elif _gap_met(result, common):
        result.update(termination_reason="gap_limit", status=GRB.OPTIMAL)
    elif method == "hybrid" and not result.get("errors"):
        exhausted = result["wall"] >= seconds
        result.update(
            termination_reason="time_limit" if exhausted else "hybrid_finished",
            status=GRB.TIME_LIMIT if exhausted else GRB.SUBOPTIMAL,
        )
    return result
