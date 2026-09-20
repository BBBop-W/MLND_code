"""One-case and batch runs with immutable inputs and independent validation."""

from pathlib import Path
import time
import traceback

from src.config import load_config
from src.data.instances import load_instance
from src.data.demand import audit_instances, describe_demand
from src.operations.export import export_operations
from .progress import RunMonitor, utc_now
from src.solvers import solve
from src.solvers.methods import get_method
from .plan import build_plan
from .status import runtime, summarize
from .storage import (
    save_inputs,
    save_solution,
    write_json,
    snapshot_sources,
    snapshot_calibration,
    verify_sources,
    sha256,
    load_inputs,
)


def run_case(
    hubs,
    output,
    *,
    config=None,
    method="benders_cg",
    options=None,
    input_path=None,
    expected_demand_signature=None,
):
    """Use a new case directory. Parameter overrides are saved before solving."""
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Preserving existing run: {output}")
    specification = get_method(method)
    configuration = load_config(config)
    settings = dict(configuration["solver"], **(options or {}))
    configuration["solver"] = settings
    # Validate the combined settings, including CLI overrides, before writing files.
    configuration = load_config(overrides=configuration)
    data = load_inputs(input_path) if input_path else load_instance(hubs, configuration)
    demand = describe_demand(data, expected_demand_signature)
    output.mkdir(parents=True)
    started = time.perf_counter()
    manifest = dict(
        schema=1,
        model_version=data.get("model_version", "path_cqn_v1"),
        started_utc=utc_now(),
        status="running",
        runtime=runtime(),
        method=method,
        method_spec=specification.describe(),
        solver=settings,
        demand=demand,
        output=str(output),
        input_source=str(input_path) if input_path else None,
    )
    write_json(output / "manifest.json", manifest)
    result = None
    phase = "preparation"
    try:
        saved_config = (
            configuration
            if input_path is None
            else {
                "input_mode": "saved_arrays",
                "source": str(Path(input_path).resolve()),
                "model_config": data.get("provenance", {}).get("resolved_config"),
                "solver": settings,
            }
        )
        write_json(output / "config.json", saved_config)
        manifest["inputs"] = save_inputs(data, output)
        manifest["sources"] = snapshot_sources(output)
        manifest["calibration"] = snapshot_calibration(output, data=data)
        write_json(output / "manifest.json", manifest)
        phase = "solve"
        with RunMonitor(output) as monitor:
            result = solve(
                data,
                method=method,
                **settings,
                log_dir=output,
                progress=monitor.progress,
                checkpoint=monitor.checkpoint,
                stop_requested=monitor.stop_requested,
            )
            save_solution(result, output)
        verify_sources(manifest["sources"])
        for name, digest in result.get("source_sha256", {}).items():
            record = manifest["sources"].get(f"src/{name}")
            if record is None or record["sha256"] != digest:
                raise RuntimeError(
                    f"Solver source fingerprint differs from its snapshot: {name}"
                )
        phase = "validation"
        if result.get("_solution") is None:
            summary = dict(
                case=output.name,
                method=method,
                status="no_incumbent",
                passed=False,
                termination_reason=result.get("termination_reason"),
                solver_status=result.get("status"),
                bound=result.get("bound"),
                wall=result.get("wall"),
                output=str(output),
            )
        else:
            operations = export_operations(data, result, output)
            summary = summarize(data, result, operations["summary"]["checks"], output)
    except Exception as error:
        detail = dict(
            type=type(error).__name__,
            message=str(error),
            traceback=traceback.format_exc(),
        )
        write_json(output / "error.json", detail)
        summary = dict(
            case=output.name,
            status="failed",
            passed=False,
            failed_phase=phase,
            error=detail,
            output=str(output),
        )
        if result is not None:
            summary.update(
                {
                    k: result.get(k)
                    for k in ("objective", "bound", "gap", "served", "fleet")
                }
            )
    summary["total_wall"] = time.perf_counter() - started
    summary["demand_input"] = demand
    write_json(output / "summary.json", summary)
    manifest.update(
        status=summary["status"],
        passed=summary["passed"],
        finished_utc=utc_now(),
        total_wall=summary["total_wall"],
    )
    manifest["artifacts"] = {
        str(p.relative_to(output)): {"bytes": p.stat().st_size, "sha256": sha256(p)}
        for p in sorted(output.rglob("*"))
        if p.is_file() and p != output / "manifest.json"
    }
    write_json(output / "manifest.json", manifest)
    return summary


def run_batch(
    hubs, output, *, config=None, methods=("benders_cg",), options=None, progress=None
):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Preserving existing experiment batch: {output}")
    plan = build_plan(hubs, methods, config=config, options=options, output=output)
    demand_audit = audit_instances([case["hubs"] for case in plan["cases"]], config)
    output.mkdir(parents=True)
    batch = {
        "started_utc": utc_now(),
        "plan": plan,
        "demand_audit": demand_audit,
        "runs": [],
    }
    write_json(output / "batch_manifest.json", batch)
    for case in plan["cases"]:
        if (output / "STOP").exists():
            batch["stopped"] = True
            break
        try:
            row = run_case(
                case["hubs"],
                case["output"],
                config=config,
                method=case["method"],
                options=options,
                expected_demand_signature=demand_audit["expected_signature"],
            )
        except Exception as error:
            row = {
                "hubs": case["hubs"],
                "method": case["method"],
                "passed": False,
                "status": "setup_failed",
                "error": str(error),
            }
        batch["runs"].append(row)
        batch["passed"] = all(r["passed"] for r in batch["runs"])
        write_json(output / "batch_manifest.json", batch)
        if progress:
            progress(row)
    batch["finished_utc"] = utc_now()
    batch["passed"] = len(batch["runs"]) == len(plan["cases"]) and all(
        r["passed"] for r in batch["runs"]
    )
    write_json(output / "batch_manifest.json", batch)
    return batch
