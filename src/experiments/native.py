"""Independent validation and Gurobi reference for native consecutive CSV runs.

This adapter is optional analysis tooling. The C++ executable reads instances,
optimizes and writes its solution without requiring this module or Python.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import math
from pathlib import Path
import time

import numpy as np

from src.data.instances import to_planning_data
from src.data.standard import load_standard
from src.experiments.storage import json_payload, sha256, source_hashes, write_json
from src.operations import export_operations
from src.experiments.progress import RunMonitor, utc_now


def load_native_result(directory):
    """Restore the ordinary Python result contract from native JSON outputs."""
    directory = Path(directory)
    result = json.loads((directory / "result.json").read_text())
    solution = json.loads((directory / "solution.json").read_text())
    scope = solution.pop("assignment_scope", None)
    if scope is not None:
        if result.get("assignment_scope", scope) != scope:
            raise ValueError("Native result and solution assignment_scope disagree")
        result["assignment_scope"] = scope
    records = json.loads((directory / "path_flows.json").read_text())
    if set(records) != {"segments", "missions", "relocations"}:
        raise ValueError(
            "Native path output must contain segments, missions, relocations"
        )
    result["_solution"] = {
        key: np.asarray(value, dtype=float) for key, value in solution.items()
    }
    for name, values in result["_solution"].items():
        if not np.isfinite(values).all():
            raise ValueError(f"Native decision array must be finite: {name}")
    for key, value in records.items():
        result["_" + key] = value
    return result


def _input_reference(directory):
    filename = Path(directory) / "input_reference.json"
    if not filename.is_file():
        raise ValueError("Native run is missing required input_reference.json")
    reference = json.loads(filename.read_text())
    if (
        not isinstance(reference, dict)
        or "manifest" not in reference
        or "max_orders" not in reference
    ):
        raise ValueError("Native input reference must contain manifest and max_orders")
    return reference


def operations_file_hashes(directory):
    """Fingerprint every exported table, JSON payload and explanatory document."""
    directory = Path(directory)
    return {
        str(path.relative_to(directory)): sha256(path)
        for path in sorted(directory.rglob("*"))
        if path.is_file() and path.suffix.lower() in {".csv", ".json", ".md"}
    }


def validate_native(directory, instance, *, max_orders=None):
    """Rebuild all physical resources and check every operational constraint."""
    started = time.perf_counter()
    directory, instance = Path(directory), Path(instance)
    result = load_native_result(directory)
    provenance = _input_reference(directory)
    k = (
        max_orders
        if max_orders is not None
        else result.get("max_orders", provenance["max_orders"])
    )
    if isinstance(k, bool) or k not in (1, 2):
        raise ValueError("Cannot determine K=1/2 from result/provenance; supply --k")
    if result.get("max_orders", k) != k or provenance["max_orders"] != k:
        raise ValueError("Requested K differs from native run provenance")
    manifest = json.loads((instance / "manifest.json").read_text())
    if provenance["manifest"] != manifest:
        raise ValueError("Native run manifest differs from the requested instance")
    manifest_sha = sha256(instance / "manifest.json")
    if result.get("input_manifest_sha256", manifest_sha) != manifest_sha:
        raise ValueError(
            "Native result input manifest SHA-256 differs from the requested instance"
        )
    inputs = load_standard(instance, case=f"K{k}")
    read_seconds = time.perf_counter() - started
    checked = time.perf_counter()
    operations = export_operations(inputs, result, directory / "operations")
    validation_seconds = time.perf_counter() - checked
    report = {
        "passed": operations["summary"]["checks"]["passed"],
        "max_orders": k,
        "assignment_scope": operations["summary"]["assignment_scope"],
        "instance": str(instance.resolve()),
        "manifest_sha256": manifest_sha,
        "native_manifest_matches": True,
        "result_sha256": sha256(directory / "result.json"),
        "solution_sha256": sha256(directory / "solution.json"),
        "path_flows_sha256": sha256(directory / "path_flows.json"),
        "operations_sha256": sha256(directory / "operations" / "operations.json"),
        "operations_files_sha256": operations_file_hashes(directory / "operations"),
        "read_seconds": read_seconds,
        "validation_and_export_seconds": validation_seconds,
        "checks": operations["summary"]["checks"],
        "summary": operations["summary"],
    }
    write_json(directory / "native_validation.json", report)
    return report


def load_solution_file(path):
    """Read a native JSON solution or existing Python solution.npz seed."""
    path = Path(path)
    if path.is_dir():
        path = path / (
            "solution.json" if (path / "solution.json").exists() else "solution.npz"
        )
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as source:
            return {key: source[key].copy() for key in source.files}
    payload = json.loads(path.read_text())
    value = payload.get("_solution", payload)
    scope = value.get("assignment_scope", payload.get("assignment_scope"))
    if "assignment_scope" in payload and payload["assignment_scope"] != scope:
        raise ValueError("Solution wrapper and decision assignment_scope disagree")
    result = {
        key: np.asarray(item, dtype=float)
        for key, item in value.items()
        if key != "assignment_scope"
    }
    if scope is not None:
        result["assignment_scope"] = scope
    return result


def save_json_result(directory, result):
    """Use the same portable JSON filenames as the native executable."""
    directory = Path(directory)
    write_json(
        directory / "result.json",
        {k: v for k, v in result.items() if not k.startswith("_")},
    )
    write_json(directory / "solution.json", result["_solution"])
    payload = {
        key: result["_" + key] for key in ("segments", "missions", "relocations")
    }
    directory.mkdir(parents=True, exist_ok=True)
    temporary = directory / "path_flows.json.tmp"
    temporary.write_text(
        json.dumps(
            json_payload(payload),
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )
        + "\n"
    )
    temporary.replace(directory / "path_flows.json")


def _gurobi_status_name(status):
    """Keep non-optimal solver outcomes explicit in portable result metadata."""
    from gurobipy import GRB

    names = (
        "LOADED",
        "OPTIMAL",
        "INFEASIBLE",
        "INF_OR_UNBD",
        "UNBOUNDED",
        "CUTOFF",
        "ITERATION_LIMIT",
        "NODE_LIMIT",
        "TIME_LIMIT",
        "SOLUTION_LIMIT",
        "INTERRUPTED",
        "NUMERIC",
        "SUBOPTIMAL",
        "INPROGRESS",
        "USER_OBJ_LIMIT",
        "WORK_LIMIT",
        "MEM_LIMIT",
    )
    mapping = {getattr(GRB, name): name.lower() for name in names if hasattr(GRB, name)}
    return mapping.get(status, str(status))


def align_native_arc_bounds(built, *, check=None, tight_airline_bounds=False):
    """Match the C++ input-derived activation rows for a solver comparison.

    The ordinary Python formulation retains its catalogue-based bounds.  This
    adapter changes only the airline coefficient in each activation row, so
    full CPLEX and Gurobi runs also share the same continuous relaxation.
    """
    data, model = built.data, built.model
    if data.route_mode != "direct" or data.candidate_routes is not None:
        raise ValueError(
            "Native arc bounds require the unrestricted direct-flight domain"
        )
    if data.max_orders not in (1, 2):
        raise ValueError("Native arc bounds require K=1 or K=2")
    demand = float(data.demand.sum())
    changed = 0
    for u, v in np.ndindex(data.allowed_links.shape):
        if check is not None and v == 0:
            check()
        bound = 0.0
        if u != v and data.allowed_links[u, v]:
            if u < data.n_l and v < data.n_l:
                bound = demand * max(0, data.n_l - 1)
            elif u >= data.n_l and v >= data.n_l:
                bound = demand * (1 if tight_airline_bounds else data.max_orders)
            else:
                bound = demand
        row = built.constraints["link"][u, v]
        airline = built.variables["airline"][u, v]
        changed += model.getCoeff(row, airline) != -bound
        model.chgCoeff(row, airline, -bound)
    model.update()
    return {
        "scheme": "native_input_arc_bounds",
        "rows_checked": int(data.n_nodes**2),
        "changed_airline_coefficients": changed,
        "tight_airline_bounds": bool(tight_airline_bounds),
        "formula": (
            "D for base-locker and locker-locker; (L-1)*D for base-base; 0 for forbidden/self arcs"
            if tight_airline_bounds
            else "D for base-locker; K*D for locker-locker; (L-1)*D for base-base; 0 for forbidden/self arcs"
        ),
    }


class _ReferenceStopped(RuntimeError):
    def __init__(self, reason):
        super().__init__(reason)
        self.reason = reason


class _ReferenceProgress(RunMonitor):
    """Persist inclusive-clock progress without claiming solver candidates verified."""

    def __init__(self, output, started, seconds):
        super().__init__(output)
        self.started = started
        self.deadline = started + seconds
        self.stage = "setup"
        self.last_event = float("-inf")
        self.bound_scope = "global"
        self.snapshot = {
            "objective": None,
            "bound": None,
            "gap": None,
            "incumbent_validated": False,
            "bound_source": None,
            "status": "not_started",
            "pricing_complete": None,
        }

    def remaining(self):
        return max(0.0, self.deadline - time.perf_counter())

    def emit(self, *, force=False, **record):
        now = time.perf_counter()
        if not force and now - self.last_event < 1.0:
            return
        self.last_event = now
        payload = {**self.snapshot, **record}
        payload.update(
            timestamp_utc=utc_now(),
            elapsed_seconds=now - self.started,
            upper_bound=payload.get("objective"),
            lower_bound=payload.get("bound"),
            bound_scope=self.bound_scope,
            validated_incumbent=payload.get("incumbent_validated", False),
        )
        self.progress(payload)

    def check(self):
        self.emit(event="heartbeat")
        if self.stop_requested():
            raise _ReferenceStopped("user_stop")
        if self.remaining() <= 0:
            raise _ReferenceStopped("time_limit")

    def stop_check(self):
        # Enumeration/build enforce their own remaining deadline. This callback
        # supplies graceful STOP/signal handling and progress during construction.
        self.emit(event="heartbeat")
        return self.stop_requested()

    def bounds(self, objective, bound, *, validated=False, source=None):
        def finite(value):
            if value is None:
                return None
            value = float(value)
            return value if math.isfinite(value) and abs(value) < 1e99 else None

        objective, bound = finite(objective), finite(bound)
        self.snapshot.update(
            objective=objective,
            bound=bound,
            gap=None
            if objective is None or bound is None
            else max(0.0, objective - bound) / max(1.0, abs(objective)),
            incumbent_validated=validated,
            bound_source=source,
        )


@contextmanager
def _reference_phase(monitor, timings, name):
    monitor.stage = name
    monitor.snapshot["status"] = "running"
    started = time.perf_counter()
    monitor.emit(force=True, event="phase_started")
    state = "completed"
    try:
        yield
    except BaseException:
        state = "interrupted"
        raise
    finally:
        timings[name] = timings.get(name, 0.0) + time.perf_counter() - started
        monitor.emit(
            force=True,
            event="phase_finished",
            phase_state=state,
            phase_seconds=timings[name],
        )


def _save_reference_checkpoint(
    output,
    directory,
    result,
    provenance,
    checks,
    monitor,
    *,
    fallback,
    write_payload=True,
):
    """Publish only after all raw files and independent physical checks exist."""
    if not checks.get("passed"):
        raise ValueError("Checkpoint must pass independent physical validation")
    if write_payload:
        save_json_result(directory, result)
        write_json(directory / "input_reference.json", provenance)
    files = {
        name: {
            "path": str((directory / name).relative_to(output)),
            "sha256": sha256(directory / name),
        }
        for name in (
            "result.json",
            "solution.json",
            "path_flows.json",
            "input_reference.json",
        )
    }
    write_json(
        output / "checkpoint.json",
        {
            "schema": 1,
            "timestamp_utc": utc_now(),
            "elapsed_seconds": time.perf_counter() - monitor.started,
            "directory": str(directory.relative_to(output)),
            "validated": True,
            "fallback": fallback,
            "kind": "all_courier_fallback" if fallback else "global_feasible_incumbent",
            "bound_scope": monitor.bound_scope,
            "objective": result["objective"],
            "bound": result.get("bound"),
            "files": files,
            "checks": checks,
        },
    )


def _gurobi_progress_callback(monitor, grb, checkpoint=None):
    """Callback values are solver reports; final physical validation stays separate."""
    mip_seen = False

    def callback(model, where):
        nonlocal mip_seen
        if monitor.stop_requested() or monitor.remaining() <= 0:
            model.terminate()
        if where == grb.Callback.MIP:
            objective = model.cbGet(grb.Callback.MIP_OBJBST)
            bound = model.cbGet(grb.Callback.MIP_OBJBND)
            monitor.bounds(objective, bound, source="solver_full_model")
            monitor.emit(
                force=not mip_seen,
                event="solver_progress",
                node_count=model.cbGet(grb.Callback.MIP_NODCNT),
            )
            mip_seen = True
        elif where == grb.Callback.MIPSOL:
            objective = model.cbGet(grb.Callback.MIPSOL_OBJ)
            bound = model.cbGet(grb.Callback.MIPSOL_OBJBND)
            monitor.bounds(objective, bound, source="solver_full_model")
            monitor.emit(force=True, event="solver_incumbent")
            if checkpoint is not None:
                checkpoint(model)
        else:
            monitor.emit(event="solver_heartbeat")

    return callback


def gurobi_reference(
    instance,
    output,
    *,
    max_orders,
    seconds=600,
    gap=0.005,
    threads=1,
    seed=43,
    fixed_design=None,
    start=None,
    log=False,
    max_model_variables=2_000_000,
    checkpoint_seconds=300.0,
    tight_airline_bounds=False,
):
    """Exhaustive full-model reference with a shared construction/solve deadline.

    No candidate path cap or restricted-column fallback is used. Expected stops
    retain an explicitly identified all-courier solution. Solver candidates in
    progress.jsonl are unvalidated until final physical reconstruction succeeds.
    """
    if max_orders not in (1, 2):
        raise ValueError("K must be 1 or 2")
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("seconds must be finite and positive")
    if not math.isfinite(gap) or gap < 0:
        raise ValueError("gap must be finite and nonnegative")
    if threads < 1:
        raise ValueError("threads must be positive")
    if (
        isinstance(max_model_variables, bool)
        or max_model_variables is not None
        and (int(max_model_variables) != max_model_variables or max_model_variables < 0)
    ):
        raise ValueError("max_model_variables must be a nonnegative integer or None")
    max_model_variables = (
        None if max_model_variables in (None, 0) else int(max_model_variables)
    )
    if not math.isfinite(checkpoint_seconds) or checkpoint_seconds <= 0:
        raise ValueError("checkpoint_seconds must be finite and positive")
    output, instance = Path(output), Path(instance)
    if (output / "result.json").exists():
        raise FileExistsError("Output already contains a result; use a new directory")
    output.mkdir(parents=True, exist_ok=True)
    begun = time.perf_counter()
    import gurobipy as gp
    from gurobipy import GRB
    from src.model.paths import EnumerationStopped, enumerate_paths
    from src.solvers.algorithm import baseline
    from src.operations.validation import build_operations
    from src.solvers.formulation import (
        DESIGN_NAMES,
        BuildStopped,
        build_model,
        read_solution,
        validate_design_arrays,
    )

    timings = {}
    inputs = data = paths = built = result = None
    fixed = initial = None
    failure = None
    stopped_stage = None
    formulation_alignment = None
    catalogue_complete = construction_complete = False
    status_name, status_code = "not_started", None
    has_solver_solution = False
    model_size = node_count = None
    bound = 0.0
    provenance = None
    checkpoint = output / "recovery" / "all_courier"
    checkpoint_stats = {"attempts": 0, "accepted": 0, "seconds_in_optimize": 0.0}
    last_checkpoint = begun
    with _ReferenceProgress(output, begun, seconds) as monitor:
        monitor.bound_scope = "fixed_design" if fixed_design is not None else "global"
        monitor.emit(
            force=True,
            event="run_started",
            budget=seconds,
            method="full_path",
            path_source="exhaustive_enumeration",
        )
        try:
            with _reference_phase(monitor, timings, "read"):
                provenance = {
                    "instance_path": str(instance.resolve()),
                    "manifest": json.loads((instance / "manifest.json").read_text()),
                    "model": json.loads((instance / "model.json").read_text()),
                    "max_orders": max_orders,
                    "source_hashes": source_hashes(),
                }
                write_json(output / "input_reference.json", provenance)
                inputs = load_standard(instance, case=f"K{max_orders}")
                data = to_planning_data(inputs)
                if data.route_mode != "direct" or data.candidate_routes is not None:
                    raise ValueError(
                        "Native full reference requires unrestricted direct-flight paths"
                    )
            # Save a small complete-recourse recovery point before potentially
            # expensive enumeration. This is not a final result or a solver claim.
            if fixed_design is None:
                with _reference_phase(monitor, timings, "initial_checkpoint"):
                    initial_fallback = baseline(data)
                    initial_fallback.update(
                        max_orders=max_orders,
                        is_final=False,
                        checkpoint_kind="all_courier_complete_recourse",
                    )
                    initial_checks = build_operations(inputs, initial_fallback)[
                        "summary"
                    ]["checks"]
                    if not initial_checks["passed"]:
                        raise ValueError(
                            "All-courier recovery checkpoint failed physical validation"
                        )
                    initial_fallback["independently_validated"] = True
                    _save_reference_checkpoint(
                        output,
                        checkpoint,
                        initial_fallback,
                        provenance,
                        initial_checks,
                        monitor,
                        fallback=True,
                    )
                    monitor.bounds(
                        initial_fallback["objective"],
                        0.0,
                        validated=True,
                        source="nonnegative_cost_bound",
                    )
                    monitor.emit(
                        force=True,
                        event="recovery_checkpoint",
                        directory=str(checkpoint.relative_to(output)),
                    )
            monitor.check()
            with _reference_phase(monitor, timings, "enumeration"):
                paths = enumerate_paths(
                    data,
                    time_limit=monitor.remaining(),
                    stop_requested=monitor.stop_check,
                )
                if not paths.complete:
                    raise ValueError("Full reference received an incomplete catalogue")
                catalogue_complete = True
                monitor.emit(
                    force=True,
                    event="catalogue_complete",
                    catalogue={
                        "segments": len(paths.initial_segments),
                        "service": len(paths.service_paths),
                        "empty": len(paths.empty_paths),
                    },
                    pricing_complete=True,
                )
                monitor.snapshot["pricing_complete"] = True
                monitor.snapshot["path_sizes"] = {
                    "segments": len(paths.initial_segments),
                    "service": len(paths.service_paths),
                    "empty": len(paths.empty_paths),
                }
            monitor.check()
            with _reference_phase(monitor, timings, "build"):
                fixed = (
                    None
                    if fixed_design is None
                    else {
                        key: value
                        for key, value in load_solution_file(fixed_design).items()
                        if key in (*DESIGN_NAMES, "assignment_scope")
                    }
                )
                initial = None if start is None else load_solution_file(start)
                built = build_model(
                    data,
                    paths,
                    fixed_design=fixed,
                    threads=threads,
                    seed=seed,
                    mip_gap=gap,
                    time_limit=monitor.remaining(),
                    stop_requested=monitor.stop_check,
                    max_lifted_variables=max_model_variables,
                )
                formulation_alignment = align_native_arc_bounds(
                    built,
                    check=monitor.check,
                    tight_airline_bounds=tight_airline_bounds,
                )
                built.model.Params.OutputFlag = int(bool(log))
                built.model.Params.LogFile = str((output / "solver.log").resolve())
                if initial is not None and fixed is None:
                    initial = validate_design_arrays(data, initial, source="warm_start")
                    for name in DESIGN_NAMES:
                        for key, variable in built.variables[name].items():
                            variable.Start = float(initial[name][key])
                model_size = {
                    "variables": built.model.NumVars,
                    "linear_rows": built.model.NumConstrs,
                    "cones": built.model.NumQConstrs,
                }
                construction_complete = True
                monitor.snapshot["model_size"] = model_size
                monitor.emit(force=True, event="model_complete", model_size=model_size)
            monitor.check()
            built.model.Params.TimeLimit = monitor.remaining()

            def save_callback_checkpoint(model):
                nonlocal last_checkpoint
                now = time.perf_counter()
                if (
                    now - last_checkpoint < checkpoint_seconds
                    or monitor.remaining() <= 0
                ):
                    return
                last_checkpoint = now
                checkpoint_stats["attempts"] += 1
                try:
                    variables = built.model.getVars()
                    values = model.cbGetSolution(variables)
                    candidate = read_solution(
                        built, value=lambda variable: values[variable.index]
                    )
                    checks = build_operations(inputs, candidate)["summary"]["checks"]
                    if not checks["passed"]:
                        monitor.emit(
                            force=True,
                            event="checkpoint_rejected",
                            reason="physical_validation",
                        )
                        return
                    candidate.update(
                        max_orders=max_orders,
                        bound=monitor.snapshot["bound"],
                        independently_validated=True,
                        is_final=False,
                    )
                    directory = (
                        output
                        / "recovery"
                        / f"incumbent_{checkpoint_stats['attempts']:06d}"
                    )
                    _save_reference_checkpoint(
                        output,
                        directory,
                        candidate,
                        provenance,
                        checks,
                        monitor,
                        fallback=False,
                    )
                    checkpoint_stats["accepted"] += 1
                    monitor.bounds(
                        candidate["objective"],
                        candidate["bound"],
                        validated=True,
                        source="solver_full_model",
                    )
                    monitor.emit(
                        force=True,
                        event="checkpoint_saved",
                        directory=str(directory.relative_to(output)),
                    )
                except Exception as exc:
                    # A callback checkpoint must not abort an otherwise valid
                    # solve. The previous complete checkpoint remains available.
                    monitor.emit(
                        force=True,
                        event="checkpoint_rejected",
                        reason=type(exc).__name__,
                        message=str(exc),
                    )
                finally:
                    checkpoint_stats["seconds_in_optimize"] += time.perf_counter() - now

            with _reference_phase(monitor, timings, "optimize"):
                built.model.optimize(
                    _gurobi_progress_callback(monitor, GRB, save_callback_checkpoint)
                )
            with _reference_phase(monitor, timings, "read_solution"):
                has_solver_solution = bool(built.model.SolCount)
                if has_solver_solution:
                    result = read_solution(built)
                elif fixed is not None:
                    raise RuntimeError(
                        "Gurobi returned no feasible solution for the fixed design"
                    )
                else:
                    result = baseline(data)
                status_code = int(built.model.Status)
                status_name = _gurobi_status_name(status_code)
                if fixed is None:
                    try:
                        bound = float(built.model.ObjBound)
                    except (AttributeError, gp.GurobiError):
                        bound = None
                    if bound is not None and (
                        not math.isfinite(bound) or status_code == GRB.NUMERIC
                    ):
                        bound = None
                else:
                    bound = result["objective"] if status_code == GRB.OPTIMAL else None
                if bound is not None and bound > result["objective"] + 1e-6 * (
                    1 + abs(result["objective"])
                ):
                    raise RuntimeError(
                        "Gurobi bound exceeds the reconstructed objective"
                    )
                if bound is not None:
                    bound = min(max(0.0, bound), result["objective"])
                node_count = float(built.model.NodeCount)
                if status_code == GRB.INTERRUPTED:
                    status_name = (
                        "user_stop"
                        if monitor.requested
                        else "time_limit"
                        if monitor.remaining() <= 0
                        else "interrupted"
                    )
        except (EnumerationStopped, BuildStopped, _ReferenceStopped) as exc:
            stopped_stage = monitor.stage
            status_name = exc.reason
            if data is not None and fixed_design is None:
                result = baseline(data)
                result["preprocessing_stop"] = {
                    "reason": exc.reason,
                    "stage": stopped_stage,
                    "columns_constructed": getattr(exc, "columns", None),
                    "initial_segments_constructed": getattr(
                        exc, "initial_segments", None
                    ),
                }
                bound = 0.0
            else:
                failure = exc
        except Exception as exc:
            failure = exc.with_traceback(None)
            stopped_stage = monitor.stage
            status_name = "error"
            if data is not None and fixed_design is None:
                result = baseline(data)
                bound = 0.0
                has_solver_solution = False
        finally:
            if built is not None:
                try:
                    with _reference_phase(monitor, timings, "dispose"):
                        built.model.dispose()
                except Exception as exc:
                    failure = exc
                    stopped_stage = "dispose"
                    status_name = "error"
                    if result is None and data is not None and fixed_design is None:
                        result = baseline(data)
                        bound = 0.0
        if failure is not None:
            write_json(
                output / "failure.json",
                {
                    "stage": stopped_stage,
                    "reason": status_name,
                    "error_type": type(failure).__name__,
                    "message": str(failure),
                    "elapsed": time.perf_counter() - begun,
                    "recovery_directory": str(checkpoint.relative_to(output))
                    if checkpoint.exists()
                    else None,
                },
            )
        if result is None:
            monitor.emit(force=True, event="failed", reason=status_name)
            raise (
                failure
                if failure is not None
                else RuntimeError("Reference run produced no result")
            )
        result.update(
            solver="Gurobi",
            solver_version=".".join(map(str, gp.gurobi.version())),
            implementation="python",
            method="fixed_design" if fixed_design is not None else "full_path",
            formulation_alignment=formulation_alignment,
            max_orders=max_orders,
            bound=bound,
            gap=None
            if bound is None
            else (result["objective"] - bound) / max(1.0, abs(result["objective"])),
            solver_status=status_name,
            solver_status_code=status_code,
            has_solver_solution=has_solver_solution,
            termination_reason=status_name,
            stopped_stage=stopped_stage,
            execution_failed=failure is not None,
            catalogue_complete=catalogue_complete,
            construction_complete=construction_complete,
            pricing_complete=catalogue_complete and construction_complete,
            path_scope=data.route_mode,
            path_source="exhaustive_enumeration",
            path_storage="complete_catalogue"
            if has_solver_solution
            else "all_courier_no_active_flows",
            catalogue={
                "segments": len(paths.initial_segments),
                "service": len(paths.service_paths),
                "empty": len(paths.empty_paths),
            }
            if paths
            else None,
            model_size=model_size,
            node_count=node_count,
            budget=seconds,
            budget_scope="read_enumeration_model_construction_and_optimization",
            threads=threads,
            seed=seed,
            mip_gap_target=gap,
            input_manifest_sha256=sha256(instance / "manifest.json"),
            fixed_design_source=None
            if fixed_design is None
            else str(Path(fixed_design).resolve()),
            warm_start_source=None if start is None else str(Path(start).resolve()),
            warm_start_scope=None if start is None else "binary_design_only",
            progress_schema="native_full_v1",
            progress_file="progress.jsonl",
            max_model_variables=max_model_variables,
            checkpoint_seconds=checkpoint_seconds,
            checkpoint_statistics=checkpoint_stats,
            solve_wall_before_validation=time.perf_counter() - begun,
            independently_validated=False,
            is_final=True,
        )
        result["timings"] = timings
        # Persist raw numerical output first, so an exporter/validation error does
        # not erase a long solve. Such a result stays explicitly unvalidated.
        with _reference_phase(monitor, timings, "raw_save"):
            save_json_result(output, result)
        try:
            with _reference_phase(monitor, timings, "validation_and_export"):
                operations = export_operations(inputs, result, output / "operations")
            result.update(
                independently_validated=operations["summary"]["checks"]["passed"],
                served=operations["summary"]["served"],
                demand=float(data.demand.sum()),
                fleet=float(result["_solution"]["fleet"]),
            )
            with _reference_phase(monitor, timings, "save"):
                write_json(
                    output / "result.json",
                    {
                        key: value
                        for key, value in result.items()
                        if not key.startswith("_")
                    },
                )
            timings["total"] = time.perf_counter() - begun
            write_json(
                output / "result.json",
                {
                    key: value
                    for key, value in result.items()
                    if not key.startswith("_")
                },
            )
            write_json(
                output / "native_validation.json",
                {
                    "passed": result["independently_validated"],
                    "max_orders": max_orders,
                    "instance": str(instance.resolve()),
                    "manifest_sha256": result["input_manifest_sha256"],
                    "native_manifest_matches": True,
                    "result_sha256": sha256(output / "result.json"),
                    "solution_sha256": sha256(output / "solution.json"),
                    "path_flows_sha256": sha256(output / "path_flows.json"),
                    "operations_sha256": sha256(
                        output / "operations" / "operations.json"
                    ),
                    "operations_files_sha256": operations_file_hashes(
                        output / "operations"
                    ),
                    "read_seconds": 0.0,
                    "validation_and_export_seconds": timings["validation_and_export"],
                    "validation_origin": "independent_python_physical_reconstruction",
                    "checks": operations["summary"]["checks"],
                    "summary": operations["summary"],
                },
            )
        except Exception as exc:
            write_json(
                output / "failure.json",
                {
                    "stage": monitor.stage,
                    "reason": "validation_or_export_failed",
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                    "elapsed": time.perf_counter() - begun,
                    "raw_solution_saved": True,
                },
            )
            monitor.emit(
                force=True, event="failed", reason="validation_or_export_failed"
            )
            raise
        _save_reference_checkpoint(
            output,
            output,
            result,
            provenance,
            operations["summary"]["checks"],
            monitor,
            fallback=not has_solver_solution,
            write_payload=False,
        )
        monitor.stage = "completed"
        monitor.snapshot.update(
            status=status_name, pricing_complete=result["pricing_complete"]
        )
        monitor.bounds(
            result["objective"],
            result["bound"],
            validated=result["independently_validated"],
            source="solver_full_model"
            if construction_complete and status_code is not None
            else "nonnegative_cost_bound",
        )
        monitor.emit(
            force=True,
            event="run_finished",
            termination_reason=status_name,
            solve_wall_before_validation=result["solve_wall_before_validation"],
            catalogue_complete=catalogue_complete,
            construction_complete=construction_complete,
        )
        if failure is not None:
            raise RuntimeError(
                "Reference run failed; raw result and validated fallback were retained"
            ) from failure
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser(
        "validate", help="Independently validate native JSON outputs"
    )
    validate.add_argument("run", type=Path)
    validate.add_argument("--instance", type=Path, required=True)
    validate.add_argument("--k", type=int, choices=(1, 2))
    reference = commands.add_parser(
        "gurobi", help="Python/Gurobi complete-path reference on the same CSV instance"
    )
    reference.add_argument("--instance", type=Path, required=True)
    reference.add_argument("--k", type=int, choices=(1, 2), required=True)
    reference.add_argument("--seconds", type=float, default=600)
    reference.add_argument("--gap", type=float, default=0.005)
    reference.add_argument("--threads", type=int, default=1)
    reference.add_argument("--seed", type=int, default=43)
    reference.add_argument("--out", type=Path, required=True)
    reference.add_argument("--fixed-design", type=Path)
    reference.add_argument("--start", type=Path)
    reference.add_argument("--log", action="store_true")
    reference.add_argument("--tight-airline-bounds", action="store_true")
    reference.add_argument(
        "--max-model-variables",
        type=int,
        default=2_000_000,
        help="Construction variable limit; 0 disables the guard",
    )
    reference.add_argument("--checkpoint-seconds", type=float, default=300.0)
    args = parser.parse_args()
    if args.command == "validate":
        report = validate_native(args.run, args.instance, max_orders=args.k)
        print(
            json.dumps(
                {
                    "passed": report["passed"],
                    "max_orders": report["max_orders"],
                    "validation_seconds": report["validation_and_export_seconds"],
                    "operations": str((args.run / "operations").resolve()),
                },
                indent=2,
            )
        )
        if not report["passed"]:
            raise SystemExit(1)
    else:
        result = gurobi_reference(
            args.instance,
            args.out,
            max_orders=args.k,
            seconds=args.seconds,
            gap=args.gap,
            threads=args.threads,
            seed=args.seed,
            fixed_design=args.fixed_design,
            start=args.start,
            log=args.log,
            max_model_variables=args.max_model_variables,
            checkpoint_seconds=args.checkpoint_seconds,
            tight_airline_bounds=args.tight_airline_bounds,
        )
        print(
            json.dumps(
                {
                    key: result.get(key)
                    for key in (
                        "objective",
                        "bound",
                        "gap",
                        "served",
                        "fleet",
                        "solver_status",
                        "timings",
                        "independently_validated",
                    )
                },
                indent=2,
            )
        )
        if not result["independently_validated"]:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
