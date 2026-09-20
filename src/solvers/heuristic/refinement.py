"""Continuous full-graph refinement of an already selected binary design."""

import time

from src.solvers.benders.subproblem import solve, RecourseStopped
from src.solvers.result import build_result
from src.solvers.pricing.backend import PricingStopped


def update_incumbent(previous, candidate):
    """Replace operational values while keeping the enclosing solve's metadata."""
    result = {**previous, **candidate}
    for key in (
        "method",
        "stats",
        "trace",
        "errors",
        "status",
        "termination_reason",
        "params",
        "source_sha256",
        "master_runtime",
        "nodes",
        "budget",
        "wall",
        "checkpoint_stage",
    ):
        if key in previous:
            result[key] = previous[key]
    return result


def refine(
    data,
    result,
    *,
    seconds,
    threads=1,
    seed=43,
    pricing="label",
    progress=None,
    stop_requested=None,
):
    """Keep the global bound and accept only validated, nonworsening solutions."""
    solution = result["_solution"]
    started = time.perf_counter()
    try:
        sub = solve(
            data,
            solution["facility"],
            solution["capacity_level"],
            solution["assignment"],
            solution["airline"],
            time_limit=seconds,
            threads=threads,
            max_iterations=None,
            seed=seed,
            pricing=pricing,
            initial_pool={
                "missions": result["_missions"],
                "relocations": result["_relocations"],
            },
            progress=progress,
            stop_requested=stop_requested,
        )
    except (RecourseStopped, PricingStopped):
        return result, {
            "accepted": False,
            "stopped": True,
            "wall": time.perf_counter() - started,
            "budget": seconds,
        }
    metadata = {
        "accepted": False,
        "budget": seconds,
        "full_graph_pricing": True,
        **{
            key: sub.get(key)
            for key in (
                "status",
                "cg_complete",
                "cut_valid",
                "duality_gap",
                "primal_residual",
            )
        },
    }
    try:
        # A fixed-design subproblem bound is not a complete-model lower bound.
        candidate = build_result(
            data,
            dict(
                solution,
                rho=sub["rho"],
                idle=sub["idle"],
                empty=sub["empty"],
                fleet=sub["fleet"],
            ),
            sub["missions"],
            sub["relocations"],
            result["bound"],
        )
    except ValueError as error:
        metadata.update(
            rejection="candidate_validation_failed",
            detail=str(error),
            wall=time.perf_counter() - started,
        )
        return result, metadata
    metadata.update(
        accepted=candidate["objective"] <= result["objective"] + 1e-7,
        objective=candidate["objective"],
        bound=result["bound"],
        wall=time.perf_counter() - started,
    )
    if metadata["accepted"]:
        result = update_incumbent(result, candidate)
    return result, metadata
