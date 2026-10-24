"""Freeze, inspect, and explicitly run the 60-job real TSL scale experiment.

Preparing and preflighting never optimize. Only the explicit ``run`` command
launches solvers. Each retry gets a separate attempt directory; prior outputs
and the immutable experiment manifest are never replaced.
"""

from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import contextmanager
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback

from src.experiments.storage import write_json

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BATCH = ROOT / "outputs/tsl_scale_ladder"
DEFAULT_CONFIG = ROOT / "configs/tsl_scale_ladder.json"
THREAD_ENV = {
    key: "1"
    for key in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "BLIS_NUM_THREADS",
    )
}
PACKAGES = (
    "numpy",
    "scipy",
    "pandas",
    "matplotlib",
    "scikit-learn",
    "gurobipy",
    "psutil",
)
ARTIFACTS = (
    "result.json",
    "solution.json",
    "path_flows.json",
    "input_reference.json",
    "progress.jsonl",
    "progress.json",
    "checkpoint.json",
    "stdout.log",
    "stderr.log",
    "execution.json",
    "resource_usage.json",
    "resources.csv",
    "native_validation.json",
    "operations/operations.json",
    "operations/operations_summary.json",
    "operations/bases.csv",
    "operations/lockers.csv",
    "operations/links.csv",
    "operations/pools.csv",
    "operations/fleet.csv",
    "operations/orders.csv",
    "operations/paths.csv",
    "operations/flight_stages.csv",
    "operations/metric_dictionary.csv",
    "operations/costs.csv",
    "operations/cost_detail.csv",
    "operations/busy_components.csv",
    "operations/od_assignments.csv",
    "operations/order_paths.csv",
    "operations/task_pairs.csv",
    "operations/task_singles.csv",
    "operations/nodes.csv",
    "operations/market_nodes.csv",
    "operations/arcs.csv",
    "operations/OPERATIONS.md",
)


def utc():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def versions():
    return {name: importlib.metadata.version(name) for name in PACKAGES}


def settings(path):
    cfg = read(path)
    required = {
        "levels": list(range(1, 11)),
        "cases": [1, 2],
        "methods": ["full_cplex", "benders_cplex", "full_gurobi"],
        "seconds_per_job": 18000,
        "threads_per_job": 1,
        "parallel_jobs": 3,
        "full_path_column_limit": 0,
        "full_model_variable_limit": 0,
        "external_warm_start": None,
        "fixed_design": None,
    }
    for name, expected in required.items():
        if cfg.get(name) != expected:
            raise ValueError(f"This experiment requires {name}={expected!r}")
    return cfg


def make_jobs(cfg):
    return [
        dict(
            id=f"L{n:02d}_K{k}_{method}",
            level=n,
            K=k,
            method=method,
            instance=f"instances/L{n:02d}",
            output=f"runs/L{n:02d}/{method}/K{k}",
        )
        for n in cfg["levels"]
        for k in cfg["cases"]
        for method in cfg["methods"]
    ]


def command_for(batch, manifest, job, attempt):
    batch, attempt = Path(batch), Path(attempt)
    cfg = manifest["protocol"]
    instance = str(batch / job["instance"])
    common = [
        "--k",
        str(job["K"]),
        "--seconds",
        str(cfg["seconds_per_job"]),
        "--threads",
        str(cfg["threads_per_job"]),
        "--seed",
        str(cfg["seed"]),
        "--gap",
        str(cfg["relative_gap_target"]),
        "--log",
        "--out",
        str(attempt),
    ]
    if job["method"] == "full_gurobi":
        return [
            manifest["python"],
            "-m",
            "src.experiments.native",
            "gurobi",
            "--instance",
            instance,
            *common,
            "--max-model-variables",
            "0",
            "--checkpoint-seconds",
            str(cfg["checkpoint_seconds"]),
        ]
    command = [
        str(batch / "frozen/bin/mlnd_cpp"),
        "--input",
        instance,
        "--method",
        "full_path" if job["method"] == "full_cplex" else "benders_cg",
        *common,
        "--path-column-limit",
        "0",
    ]
    if job["method"] == "benders_cplex":
        for flag, key in (
            ("cg-max-columns", "cg_max_columns_per_iteration"),
            ("max-iterations", "max_benders_iterations"),
            ("capacity-cut-limit", "capacity_cut_limit"),
            ("root-cut-rounds", "root_cut_rounds"),
            ("root-budget-fraction", "root_budget_fraction"),
            ("pareto-frequency", "pareto_frequency"),
            ("pareto-budget-fraction", "pareto_budget_fraction"),
            ("pareto-call-seconds", "pareto_call_seconds"),
            ("heuristic-budget-fraction", "heuristic_budget_fraction"),
            ("local-branching-radius", "local_branching_radius"),
            ("cg-duality-tolerance", "cg_duality_tolerance"),
            ("pricing-tolerance", "pricing_tolerance"),
        ):
            command.extend(["--" + flag, str(cfg[key])])
    return command


def prepare(batch=DEFAULT_BATCH, config=DEFAULT_CONFIG, binary=None):
    """Freeze existing validated instances, sources, binary and 60 commands."""
    batch = Path(batch).resolve()
    cfg = settings(config)
    prepared = read(batch / "preparation_manifest.json")
    if [v["level"] for v in prepared["levels"]] != cfg["levels"]:
        raise ValueError("Generate and validate all ten levels first")
    if (batch / "experiment_manifest.json").exists() or (batch / "frozen").exists():
        raise FileExistsError("Refusing to replace a frozen experiment")
    binary = Path(binary or ROOT / "cpp/build/mlnd_cpp").resolve()
    if not binary.is_file():
        raise FileNotFoundError(
            "Build cpp/build/mlnd_cpp first; preparing never builds or solves"
        )
    package_versions = versions()
    frozen = batch / "frozen"
    frozen.mkdir()
    for folder in ("src", "configs", "cpp"):
        shutil.copytree(
            ROOT / folder,
            frozen / folder,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "build", ".DS_Store"),
        )
    for name in ("main.py", "pyproject.toml"):
        shutil.copy2(ROOT / name, frozen / name)
    (frozen / "bin").mkdir()
    shutil.copy2(binary, frozen / "bin/mlnd_cpp")
    write_json(frozen / "protocol.json", cfg)
    # This launcher imports the frozen dispatcher, so later repo edits cannot
    # alter long-running jobs or their analysis. No action defaults to status.
    launcher = (
        "from pathlib import Path\nimport sys, os\n"
        f"os.environ.update({THREAD_ENV!r})\n"
        "batch = Path(__file__).resolve().parent\n"
        "sys.path.insert(0, str(batch / 'frozen'))\n"
        "from src.experiments.scale_ladder import main\n"
        "main([*(sys.argv[1:] or ['status']), '--batch', str(batch)])\n"
    )
    (batch / "run.py").write_text(launcher)
    source_files = sorted(p for p in frozen.rglob("*") if p.is_file())
    input_files = sorted(p for p in (batch / "instances").rglob("*") if p.is_file())
    provenance_files = sorted(
        p for p in (batch / "provenance").rglob("*") if p.is_file()
    )
    import psutil

    manifest = dict(
        schema=cfg["schema"],
        created_utc=utc(),
        status="prepared_not_started",
        protocol=cfg,
        jobs=make_jobs(cfg),
        python=sys.executable,
        python_version=platform.python_version(),
        packages=package_versions,
        platform=platform.platform(),
        cpu_count=os.cpu_count(),
        physical_cpu_count=psutil.cpu_count(logical=False),
        memory_bytes=psutil.virtual_memory().total,
        disk_free_bytes_at_preparation=shutil.disk_usage(batch).free,
        thread_environment=THREAD_ENV,
        solver_jobs_started_during_preparation=0,
        git_branch=subprocess.check_output(
            ["git", "branch", "--show-current"], cwd=ROOT, text=True
        ).strip(),
        git_head=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        git_dirty=True,
        source_of_truth="frozen source and binary hashes, including uncommitted authorized changes",
        source_sha256={str(p.relative_to(batch)): digest(p) for p in source_files},
        input_sha256={str(p.relative_to(batch)): digest(p) for p in input_files},
        provenance_sha256={
            str(p.relative_to(batch)): digest(p) for p in provenance_files
        },
        preparation_manifest_sha256=digest(batch / "preparation_manifest.json"),
        launcher_sha256=digest(batch / "run.py"),
        output_contract=list(ARTIFACTS),
        schedule="ascending L then K, rolling queue with at most three running jobs; no shared starts",
    )
    for job in manifest["jobs"]:
        job["initial_command"] = command_for(
            batch, manifest, job, batch / job["output"] / "attempts/001"
        )
    write_json(batch / "experiment_manifest.json", manifest)
    write_json(
        batch / "state.json",
        dict(
            schema="consecutive-scale-state/1",
            status="prepared_not_started",
            updated_utc=utc(),
            jobs={
                j["id"]: {"status": "pending", "attempts": []} for j in manifest["jobs"]
            },
        ),
    )
    return preflight(batch)


def preflight(batch):
    """Read/hash/import checks only. No Env, Model, solve, or path enumeration."""
    from src.data.standard import load_standard

    batch = Path(batch).resolve()
    m = read(batch / "experiment_manifest.json")
    frozen_protocol = settings(batch / "frozen/protocol.json")
    if m["protocol"] != frozen_protocol:
        raise ValueError("Manifest protocol differs from the frozen protocol")
    expected_jobs = make_jobs(frozen_protocol)
    for job in expected_jobs:
        job["initial_command"] = command_for(
            batch, m, job, batch / job["output"] / "attempts/001"
        )
    if m["jobs"] != expected_jobs:
        raise ValueError("Manifest job list/commands differ from the frozen experiment")
    for group in ("source_sha256", "input_sha256", "provenance_sha256"):
        for name, expected in m[group].items():
            if digest(batch / name) != expected:
                raise ValueError(f"Frozen {group} file changed: {name}")
    for filename, key in (
        ("preparation_manifest.json", "preparation_manifest_sha256"),
        ("run.py", "launcher_sha256"),
    ):
        if digest(batch / filename) != m[key]:
            raise ValueError(f"Frozen provenance changed: {filename}")
    if platform.python_version() != m["python_version"] or versions() != m["packages"]:
        raise ValueError("Python/package versions changed since freezing")
    for n in m["protocol"]["levels"]:
        for k in m["protocol"]["cases"]:
            inp = load_standard(batch / f"instances/L{n:02d}", case=f"K{k}")
            if (len(inp["base_cost"]), len(inp["slot_counts"]), len(inp["demand"])) != (
                n,
                10 * n,
                25 * n,
            ):
                raise ValueError(f"Incorrect dimensions at L{n:02d}/K{k}")
            if inp["max_orders"] != k:
                raise ValueError("K case override not applied")
    if len(m["jobs"]) != 60 or len({j["id"] for j in m["jobs"]}) != 60:
        raise ValueError("Expected exactly 60 distinct jobs")
    result = dict(
        passed=True,
        checked_utc=utc(),
        solver_invoked=False,
        instances=10,
        case_inputs=20,
        jobs=60,
        parallel_jobs=3,
        threads_per_job=1,
        seconds_per_job=18000,
        all_hashes_match=True,
        license_status="not checked: no optimizer environment opened",
        manifest_sha256=digest(batch / "experiment_manifest.json"),
    )
    write_json(batch / "preflight.json", result)
    return result


@contextmanager
def batch_lock(batch):
    path = Path(batch) / ".dispatcher.lock"
    with path.open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(
                "Another dispatcher already owns this experiment"
            ) from exc
        lock.seek(0)
        lock.truncate()
        lock.write(json.dumps({"pid": os.getpid(), "started_utc": utc()}))
        lock.flush()
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def _terminate(process):
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()
    except ProcessLookupError:
        pass


def _resource_sample(pid, elapsed):
    import psutil

    row = dict(
        timestamp_utc=utc(),
        elapsed_seconds=elapsed,
        rss_bytes=0,
        cpu_user_seconds=0,
        cpu_system_seconds=0,
        process_count=0,
        os_threads=0,
        sampling_status="complete",
    )
    try:
        parent = psutil.Process(pid)
        children = [parent, *parent.children(recursive=True)]
    except psutil.NoSuchProcess:
        return row
    except (psutil.AccessDenied, PermissionError) as exc:
        row.update(
            sampling_status=f"unavailable: {type(exc).__name__}",
            rss_bytes=None,
            cpu_user_seconds=None,
            cpu_system_seconds=None,
            process_count=None,
            os_threads=None,
        )
        return row
    for process in children:
        try:
            row["rss_bytes"] += process.memory_info().rss
            cpu = process.cpu_times()
            row["cpu_user_seconds"] += cpu.user
            row["cpu_system_seconds"] += cpu.system
            row["os_threads"] += process.num_threads()
            row["process_count"] += 1
        except (psutil.NoSuchProcess, psutil.AccessDenied, PermissionError):
            row["sampling_status"] = "partial_process_access"
            continue
    return row


def execute(command, cwd, output, cfg, cancel, on_start=None):
    """Run one process group, collecting logs/resources and enforcing watchdog."""
    output = Path(output)
    started = time.monotonic()
    execution = dict(
        command=command,
        cwd=str(cwd),
        started_utc=utc(),
        status="starting",
        thread_environment=THREAD_ENV,
        sampled_peak_rss_bytes=None,
        resource_note="RSS is sampled across the process tree when permitted; resource_usage.json records exact child wait4 usage",
    )
    env = os.environ.copy()
    env.update(THREAD_ENV)
    env.update(
        PYTHONUNBUFFERED="1", MPLBACKEND="Agg", MPLCONFIGDIR=str(output / "matplotlib")
    )
    wrapped = [
        sys.executable,
        str(Path(__file__).with_name("process_worker.py")),
        "--usage-file",
        str(output / "resource_usage.json"),
        "--",
        *command,
    ]
    execution["wrapped_command"] = wrapped
    write_json(output / "execution.json", execution)
    process = None
    try:
        with (
            (output / "stdout.log").open("w") as stdout,
            (output / "stderr.log").open("w") as stderr,
            (output / "resources.csv").open("w", newline="") as stream,
        ):
            process = subprocess.Popen(
                wrapped,
                cwd=cwd,
                env=env,
                stdout=stdout,
                stderr=stderr,
                start_new_session=True,
            )
            execution.update(pid=process.pid, status="running")
            write_json(output / "execution.json", execution)
            if on_start:
                on_start(process.pid)
            writer = None
            next_sample = 0.0
            while process.poll() is None:
                elapsed = time.monotonic() - started
                if cancel.is_set():
                    execution["stop_reason"] = "dispatcher_interrupted"
                    _terminate(process)
                    break
                if elapsed > cfg["seconds_per_job"] + cfg["watchdog_grace_seconds"]:
                    execution["stop_reason"] = "watchdog_deadline_exceeded"
                    _terminate(process)
                    break
                if elapsed >= next_sample:
                    row = _resource_sample(process.pid, elapsed)
                    if writer is None:
                        writer = csv.DictWriter(stream, fieldnames=list(row))
                        writer.writeheader()
                    writer.writerow(row)
                    stream.flush()
                    if row["rss_bytes"] is not None:
                        execution["sampled_peak_rss_bytes"] = max(
                            execution["sampled_peak_rss_bytes"] or 0, row["rss_bytes"]
                        )
                    execution["elapsed_seconds"] = elapsed
                    write_json(output / "execution.json", execution)
                    next_sample = elapsed + cfg["resource_sample_seconds"]
                cancel.wait(min(1.0, cfg["resource_sample_seconds"]))
            execution.update(returncode=process.wait(), status="exited")
    except BaseException as exc:
        if process is not None:
            _terminate(process)
        execution.update(
            status="launch_or_monitor_failed",
            error=f"{type(exc).__name__}: {exc}",
            traceback=traceback.format_exc(),
        )
    execution.update(
        finished_utc=utc(), external_wall_seconds=time.monotonic() - started
    )
    if (output / "resource_usage.json").is_file():
        usage = read(output / "resource_usage.json")
        execution["resource_usage"] = usage
        execution["peak_rss_bytes"] = usage.get("peak_rss_bytes")
    write_json(output / "execution.json", execution)
    return execution


def _recover_checkpoint(attempt):
    """Save interrupted-run evidence separately; never replace the raw result."""
    from src.experiments.native import load_native_result, save_json_result

    checkpoint = read(attempt / "checkpoint.json")
    reference = attempt / "input_reference.json"
    if "files" in checkpoint:
        if checkpoint.get("validated") is not True:
            raise ValueError("Referenced checkpoint was not validated")
        sources = {}
        for name in (
            "result.json",
            "solution.json",
            "path_flows.json",
            "input_reference.json",
        ):
            entry = checkpoint["files"][name]
            path = (attempt / entry["path"]).resolve()
            if attempt.resolve() not in path.parents or digest(path) != entry["sha256"]:
                raise ValueError(f"Referenced checkpoint integrity failed: {name}")
            sources[name] = path
        if len({p.parent for p in sources.values()}) != 1:
            raise ValueError("Checkpoint payload must belong to one complete snapshot")
        reference = sources["input_reference.json"]
        checkpoint = load_native_result(reference.parent)
    if "_solution" not in checkpoint and "result" in checkpoint:
        checkpoint = checkpoint["result"]
    if "_solution" not in checkpoint:
        raise ValueError("Checkpoint lacks a recoverable raw solution")
    checkpoint_hash = digest(attempt / "checkpoint.json")
    recovered = attempt / "recovered"
    number = 1
    while recovered.exists():
        if (recovered / "result.json").is_file() and read(
            recovered / "result.json"
        ).get("recovery_source_sha256") == checkpoint_hash:
            return recovered
        number += 1
        recovered = attempt / f"recovered_{number:03d}"
    recovered.mkdir()
    checkpoint.update(
        recovered_from_checkpoint=True,
        termination_reason="interrupted_checkpoint",
        recovery_source_sha256=checkpoint_hash,
    )
    save_json_result(recovered, checkpoint)
    shutil.copy2(reference, recovered / "input_reference.json")
    return recovered


def audit_attempt(batch, manifest, job, attempt):
    """Validate current files, export operations, and retain failures explicitly."""
    from src.experiments.native import validate_native

    batch, attempt = Path(batch), Path(attempt)
    started = time.monotonic()
    execution = (
        read(attempt / "execution.json")
        if (attempt / "execution.json").exists()
        else {}
    )
    report = dict(
        checked_utc=utc(),
        passed=False,
        job_id=job["id"],
        errors=[],
        execution_status=execution.get("status"),
        returncode=execution.get("returncode"),
    )
    source = attempt
    try:
        if not all(
            (attempt / name).is_file()
            for name in ("result.json", "solution.json", "path_flows.json")
        ):
            source = _recover_checkpoint(attempt)
        try:
            checked = validate_native(
                source, batch / job["instance"], max_orders=job["K"]
            )
        except Exception as exc:
            if source != attempt or not (attempt / "checkpoint.json").is_file():
                raise
            report["errors"].append(
                f"Final output validation failed; recovering checkpoint: {type(exc).__name__}: {exc}"
            )
            source = _recover_checkpoint(attempt)
            checked = validate_native(
                source, batch / job["instance"], max_orders=job["K"]
            )
        report["feasible_solution_validated"] = checked["passed"]
        result = read(source / "result.json")
        full = job["method"] != "benders_cplex"
        report["catalogue_complete"] = result.get("catalogue_complete")
        report["construction_complete"] = result.get("construction_complete")
        report["pricing_complete"] = result.get("pricing_complete")
        report["termination_reason"] = result.get("termination_reason")
        report["result_directory"] = str(source.relative_to(batch))
        report["fallback"] = bool(
            result.get("fallback")
            or result.get("recovered_from_checkpoint")
            or result.get("incumbent_kind")
            in ("all_courier_fallback", "all_courier_baseline")
        )
        missing = [
            name
            for name in ARTIFACTS
            if not (attempt / name).is_file() and not (source / name).is_file()
        ]
        report["missing_artifacts"] = missing
        if missing:
            report["errors"].append("Missing required artifacts: " + ", ".join(missing))
        if execution.get("returncode") != 0 or execution.get("stop_reason"):
            report["errors"].append("Process did not finish normally")
        if full and not (
            result.get("catalogue_complete") and result.get("construction_complete")
        ):
            report["errors"].append(
                "Full path enumeration/model construction not completed"
            )
        if not full and result.get("support_complete") is False:
            report["errors"].append("Implicit support preprocessing not completed")
        report["passed"] = bool(checked["passed"] and not report["errors"])
        report["artifact_sha256"] = {
            str(p.relative_to(attempt)): digest(p)
            for p in attempt.rglob("*")
            if p.is_file()
            and p.name not in {"artifact_audit.json"}
            and "matplotlib" not in p.parts
        }
    except Exception as exc:
        report["errors"].append(f"{type(exc).__name__}: {exc}")
        report["traceback"] = traceback.format_exc()
    report["validation_wall_seconds"] = time.monotonic() - started
    write_json(attempt / "artifact_audit.json", report)
    return report


def dispatch(batch, *, retry_failed=False, worker=None):
    """Resume pending jobs; explicit retries append attempts, never overwrite."""
    import psutil

    batch = Path(batch).resolve()
    preflight(batch)
    manifest = read(batch / "experiment_manifest.json")
    cfg = manifest["protocol"]
    cancel = threading.Event()
    state_lock = threading.Lock()
    with batch_lock(batch):
        state = read(batch / "state.json")
        job_by_id = {j["id"]: j for j in manifest["jobs"]}
        for job_id, value in state["jobs"].items():
            if value["status"] == "running":
                current = value["attempts"][-1]
                execution_path = batch / current["directory"] / "execution.json"
                execution = read(execution_path) if execution_path.exists() else {}
                pid = execution.get("pid")
                if pid and psutil.pid_exists(pid):
                    raise RuntimeError(
                        f"Previously launched process {pid} is still alive; no duplicate job started"
                    )
                audit = audit_attempt(
                    batch, manifest, job_by_id[job_id], batch / current["directory"]
                )
                status = "completed" if audit["passed"] else "interrupted"
                current.update(
                    status=status, audit=audit, recovered_dispatch_state_utc=utc()
                )
                value.update(
                    status=status,
                    result_directory=audit.get(
                        "result_directory", current["directory"]
                    ),
                )

        def save():
            state["updated_utc"] = utc()
            write_json(batch / "state.json", state)

        state["status"] = "running"
        save()
        eligible = [
            j
            for j in manifest["jobs"]
            if state["jobs"][j["id"]]["status"] == "pending"
            or (
                retry_failed
                and state["jobs"][j["id"]]["status"] in ("failed", "interrupted")
            )
        ]

        def one(job):
            with state_lock:
                record = state["jobs"][job["id"]]
                number = len(record["attempts"]) + 1
                relative = f"{job['output']}/attempts/{number:03d}"
                attempt = batch / relative
                attempt.mkdir(parents=True, exist_ok=False)
                current = dict(
                    number=number,
                    directory=relative,
                    started_utc=utc(),
                    status="running",
                )
                record.update(status="running", directory=relative)
                record["attempts"].append(current)
                save()
            command = command_for(batch, manifest, job, attempt)
            write_json(
                attempt / "job.json",
                dict(
                    job=job,
                    protocol=cfg,
                    command=command,
                    experiment_manifest_sha256=digest(
                        batch / "experiment_manifest.json"
                    ),
                ),
            )
            print(
                json.dumps(
                    dict(event="job_started", job=job["id"], attempt=number, utc=utc())
                ),
                flush=True,
            )
            try:
                if (
                    worker is not None
                ):  # dependency injection for no-solver scheduler tests
                    audit = worker(job, attempt, cancel)
                else:
                    execute(command, batch / "frozen", attempt, cfg, cancel)
                    audit = audit_attempt(batch, manifest, job, attempt)
                status = (
                    "completed"
                    if audit["passed"]
                    else ("interrupted" if cancel.is_set() else "failed")
                )
            except BaseException as exc:
                audit = dict(
                    passed=False,
                    errors=[f"{type(exc).__name__}: {exc}"],
                    traceback=traceback.format_exc(),
                )
                write_json(attempt / "dispatcher_failure.json", audit)
                status = "failed"
            with state_lock:
                current.update(status=status, finished_utc=utc(), audit=audit)
                record.update(
                    status=status,
                    result_directory=audit.get("result_directory", relative),
                )
                save()
            print(
                json.dumps(
                    dict(event="job_finished", job=job["id"], status=status, utc=utc())
                ),
                flush=True,
            )

        handlers = {}
        if threading.current_thread() is threading.main_thread():
            for sig in (signal.SIGINT, signal.SIGTERM):
                handlers[sig] = signal.signal(sig, lambda *_: cancel.set())
        try:
            with ThreadPoolExecutor(max_workers=cfg["parallel_jobs"]) as pool:
                iterator = iter(eligible)
                pending = set()
                while not cancel.is_set():
                    while len(pending) < cfg["parallel_jobs"]:
                        job = next(iterator, None)
                        if job is None:
                            break
                        pending.add(pool.submit(one, job))
                    if not pending:
                        break
                    done, pending = wait(
                        pending, timeout=1, return_when=FIRST_COMPLETED
                    )
                    for future in done:
                        future.result()
        finally:
            for sig, handler in handlers.items():
                signal.signal(sig, handler)
            state["status"] = (
                "interrupted"
                if cancel.is_set()
                else "finished_with_failures"
                if any(r["status"] != "completed" for r in state["jobs"].values())
                else "completed"
            )
            save()
    if worker is None:
        from src.reporting.scale_ladder import build_scale_report

        build_scale_report(batch)
    return status_summary(batch)


def status_summary(batch):
    state = read(Path(batch) / "state.json")
    counts = {}
    for value in state["jobs"].values():
        counts[value["status"]] = counts.get(value["status"], 0) + 1
    return dict(
        status=state["status"],
        jobs=counts,
        updated_utc=state["updated_utc"],
        solver_invoked_by_status=False,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        nargs="?",
        default="status",
        choices=("prepare", "preflight", "status", "run", "report"),
    )
    parser.add_argument("--batch", type=Path, default=DEFAULT_BATCH)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--binary", type=Path)
    parser.add_argument("--retry-failed", action="store_true")
    args = parser.parse_args(argv)
    if args.action == "prepare":
        value = prepare(args.batch, args.config, args.binary)
    elif args.action == "preflight":
        value = preflight(args.batch)
    elif args.action == "status":
        value = status_summary(args.batch)
    elif args.action == "run":
        value = dispatch(args.batch, retry_failed=args.retry_failed)
    else:
        from src.reporting.scale_ladder import build_scale_report

        value = build_scale_report(args.batch)
    print(json.dumps(value, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
