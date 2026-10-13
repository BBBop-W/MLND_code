"""Unified method selection and normalized results for all five comparisons."""

from pathlib import Path
import time

from .methods import get_method
from .settings import DEFAULT_MIP_GAP, DEFAULT_MIP_GAP_ABS


def solve(
    data,
    *,
    method="benders_cg",
    time_limit=600.0,
    threads=1,
    seed=43,
    mip_gap=None,
    mip_gap_abs=None,
    max_iterations=None,
    accelerations=None,
    column_limit=500_000,
    graph_limit=1_000_000,
    log_dir=None,
    progress=None,
    checkpoint=None,
    stop_requested=None,
):
    """Solve one model. Reading method descriptions never calls this function."""
    spec = get_method(method, accelerations)
    if data.get("same_base_service", False) and spec.formulation != "arc":
        raise ValueError("Same-base service experiments currently require the arc method")
    common = dict(
        threads=threads,
        seed=seed,
        mip_gap=DEFAULT_MIP_GAP if mip_gap is None else mip_gap,
        mip_gap_abs=DEFAULT_MIP_GAP_ABS if mip_gap_abs is None else mip_gap_abs,
    )
    started = time.perf_counter()
    directory = Path(log_dir) if log_dir is not None else None
    if directory:
        directory.mkdir(parents=True, exist_ok=True)
    if spec.formulation == "benders":
        from .benders.driver import solve_benders

        result = solve_benders(
            data,
            spec,
            time_limit=time_limit,
            max_iterations=max_iterations,
            log_dir=directory,
            progress=progress,
            checkpoint=checkpoint,
            stop_requested=stop_requested,
            **common,
        )
    else:
        from .direct import arc, path

        backend = arc if spec.formulation == "arc" else path
        limits = (
            {"max_state_arcs": graph_limit}
            if spec.formulation == "arc"
            else {"column_limit": column_limit}
        )
        raw = backend.solve(
            data,
            time_limit=time_limit,
            stop_requested=stop_requested,
            progress=progress,
            **limits,
            **common,
        )
        result = _direct_result(data, raw)
        if checkpoint and result.get("_solution") is not None:
            checkpoint(result)
    result.update(
        method=spec.name,
        method_spec=spec.describe(),
        budget=time_limit,
        wall=time.perf_counter() - started,
    )
    return result


def _direct_result(data, raw):
    from .result import build_result

    raw = dict(raw)
    raw.setdefault(
        "termination_reason",
        {
            2: "gap_limit",
            3: "infeasible",
            4: "infeasible_or_unbounded",
            5: "unbounded",
            9: "time_limit",
            11: "user_stop",
            12: "numerical_failure",
        }.get(raw.get("status"), "solver_stopped"),
    )
    if raw.get("objective") is None:
        return raw
    if "_solution" in raw:
        return raw
    keys = (
        "facility",
        "capacity_level",
        "assignment",
        "airline",
        "rho",
        "idle",
        "fleet",
    )
    result = build_result(
        data,
        {key: raw[key] for key in keys},
        raw["missions"],
        raw["relocations"],
        raw["bound"],
    )
    # Preserve formulation dimensions, solver parameters and timing measurements.
    result.update(
        {
            key: value
            for key, value in raw.items()
            if key not in (*keys, "missions", "relocations")
        }
    )
    return result
