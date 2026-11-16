"""One model, one run lifecycle, explicit saved inputs and numerical evidence."""

from __future__ import annotations
from dataclasses import replace
from copy import deepcopy
from pathlib import Path
import time

from src.data.instances import to_planning_data, pack_planning_data
from src.model.assignment import assignment_scope
from src.operations.export import export_operations
from src.operations.validation import build_operations
from .storage import save_inputs, save_solution, write_json, source_hashes
from .progress import RunMonitor, utc_now
from .status import runtime
from src.solvers.methods import resolve_accelerations
from src.solvers.algorithm import _phase


def run_case(
    data,
    output,
    *,
    method="benders_cg",
    time_limit=600,
    threads=1,
    seed=43,
    mip_gap=1e-4,
    column_limit=500000,
    max_iterations=1000,
    make_report=True,
    accelerations=None,
):
    from src.solvers.api import solve

    resolved_accelerations = resolve_accelerations(method, accelerations)
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Output directory must be new or empty: {output}")
    planning = to_planning_data(data)
    if not isinstance(data, dict):
        data = pack_planning_data(planning)
    output.mkdir(parents=True, exist_ok=True)
    fingerprint = source_hashes()
    manifest = dict(
        model_version="consecutive",
        flow_policy=planning.flow_policy,
        assignment_scope=assignment_scope(planning),
        started_utc=utc_now(),
        status="running",
        input_hashes=save_inputs(data, output),
        source_hashes=fingerprint,
        runtime=runtime(),
        options=dict(
            method=method,
            time_limit=time_limit,
            threads=threads,
            seed=seed,
            mip_gap=mip_gap,
            column_limit=column_limit,
            max_iterations=max_iterations,
            accelerations=resolved_accelerations,
        ),
    )
    write_json(output / "manifest.json", manifest)
    started = time.perf_counter()
    runner_timings = dict(
        checkpoint_validation=0.0,
        checkpoint_write=0.0,
        progress_io=0.0,
        final_validation=0.0,
    )
    try:
        with RunMonitor(output) as monitor:

            def checkpoint(candidate):
                if candidate.get("_solution") is not None:
                    with _phase(runner_timings, "checkpoint_validation"):
                        checked = build_operations(data, candidate)
                    if checked["summary"]["checks"]["passed"]:
                        with _phase(runner_timings, "checkpoint_write"):
                            monitor.checkpoint(candidate)

            def progress(record):
                with _phase(runner_timings, "progress_io"):
                    monitor.progress(record)

            result = solve(
                planning,
                method=method,
                time_limit=time_limit,
                threads=threads,
                seed=seed,
                mip_gap=mip_gap,
                column_limit=column_limit,
                max_iterations=max_iterations,
                accelerations=resolved_accelerations,
                progress=progress,
                checkpoint=checkpoint,
                stop_requested=monitor.stop_requested,
            )
        with _phase(runner_timings, "final_validation"):
            operations = build_operations(data, result)
        s = operations["summary"]
        result.update(
            {
                k: s[k]
                for k in (
                    "served",
                    "direct",
                    "fleet",
                    "task_flights",
                    "continued_orders",
                    "slot_total",
                    "opened",
                )
            }
        )
        result["wall"] = time.perf_counter() - started
        result["runner_timings"] = runner_timings
        result["input_hashes"] = manifest["input_hashes"]
        save_solution(result, output)
        export_operations(data, result, output)
        if source_hashes() != fingerprint:
            raise RuntimeError(
                "Source changed during the run; results are not from a frozen version"
            )
        summary = dict(
            model_version="consecutive",
            flow_policy=planning.flow_policy,
            assignment_scope=assignment_scope(planning),
            accelerations=resolved_accelerations,
            passed=s["checks"]["passed"],
            objective=result["objective"],
            bound=result.get("bound"),
            gap=result.get("gap"),
            pricing_complete=result.get("pricing_complete"),
            path_scope=result.get("path_scope"),
            termination_reason=result.get("termination_reason"),
            served=s["served"],
            task_flights=s["task_flights"],
            continued_orders=s["continued_orders"],
            wall=result["wall"],
            output=str(output),
        )
        write_json(output / "summary.json", summary)
        if make_report:
            from src.reporting.report import run

            run(output)
        manifest.update(status="completed", finished_utc=utc_now(), passed=True)
        write_json(output / "manifest.json", manifest)
        return summary
    except BaseException as exc:
        manifest.update(
            status="failed",
            finished_utc=utc_now(),
            passed=False,
            error=f"{type(exc).__name__}: {exc}",
        )
        write_json(output / "manifest.json", manifest)
        raise


def compare(data, output, **options):
    """Run K=1 and K=2 against identical market inputs and explicit scenario costs."""
    output = Path(output)
    planning = to_planning_data(data)
    rows = []
    for k in (1, 2):
        p = replace(planning, max_orders=k)
        case = deepcopy(data) if isinstance(data, dict) else {}
        metadata = case.get("metadata", {})
        if "resolved_config" in metadata:
            metadata["resolved_config"]["max_orders"] = k
        metadata.setdefault("run_overrides", {})["max_orders"] = k
        case.update(pack_planning_data(p, metadata=metadata))
        rows.append(dict(max_orders=k, **run_case(case, output / f"K{k}", **options)))
    write_json(output / "comparison.json", rows)
    return rows
