"""K=2 algorithm comparison on existing standard inputs, never generated inputs.

``prepare`` freezes current sources, executable and byte-for-byte input copies.
Only ``run`` launches solvers. Logs, attempts, certificates and failures survive.
The native/Gurobi adapters and process/resource monitor are reused unchanged.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import threading
import time
from zoneinfo import ZoneInfo

from src.data.standard import load_standard
from src.experiments import scale_ladder as existing
from src.experiments.storage import write_json
from src.reporting.native import native_phase_times

ROOT = Path(__file__).resolve().parents[2]
METHODS = ("full_cplex", "full_gurobi", "benders_lifted", "benders_conditional")
DEFAULT_BATCH = ROOT / "outputs/conditional_benchmark_20261022"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def optional_read(path):
    try:
        value = read(path)
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            value.update(chunk)
    return value.hexdigest()


def command_for(batch, manifest, job, attempt):
    batch = Path(batch)
    cfg = manifest["protocol"]
    common = ["--k", "2", "--seconds", str(cfg["seconds_per_job"]),
              "--threads", str(cfg["threads_per_job"]), "--seed", str(cfg["seed"]),
              "--gap", str(cfg["relative_gap_target"]), "--log", "--out", str(attempt)]
    instance = str(batch / job["instance"])
    if job["method"] == "full_gurobi":
        return [manifest["python"], "-m", "src.experiments.native", "gurobi",
                "--instance", instance, *common, "--max-model-variables", "0",
                "--checkpoint-seconds", str(cfg["checkpoint_seconds"])]
    result = [str(batch / "frozen/bin/mlnd_cpp"), "--input", instance,
              "--method", "full_path" if job["method"] == "full_cplex" else "benders_cg",
              *common, "--path-column-limit", "0"]
    if job["method"].startswith("benders_"):
        result += ["--demand-formulation",
                   "conditional" if job["method"] == "benders_conditional" else "lifted",
                   "--cg-max-columns", str(cfg["cg_max_columns_per_iteration"]),
                   "--max-iterations", str(cfg["max_benders_iterations"])]
        if cfg["plain"]:
            result += ["--plain"]
    return result


def prepare(batch, instances, *, binary=None, seconds=180, threads=1,
            parallel=1, gap=0.005, seed=43, plain=False, job_rss_gib=4,
            batch_rss_gib=8, system_available_gib=2, hard_wall_seconds=None,
            resource_sample_seconds=.5, terminate_grace_seconds=5):
    batch = Path(batch).resolve()
    if batch.exists():
        raise FileExistsError(f"Refusing to replace existing batch: {batch}")
    binary = Path(binary or ROOT / "cpp/build/mlnd_cpp").resolve()
    if not binary.is_file():
        raise FileNotFoundError(binary)
    if seconds <= 0 or threads < 1 or parallel < 1 or not 0 <= gap < 1:
        raise ValueError("Invalid solve budget, thread count, concurrency or gap")
    if min(job_rss_gib, batch_rss_gib, system_available_gib) <= 0 or resource_sample_seconds <= 0 or terminate_grace_seconds <= 0:
        raise ValueError("Resource protections require positive limits and sampling/grace intervals")
    hard_wall_seconds = seconds + 120 if hard_wall_seconds is None else hard_wall_seconds
    if hard_wall_seconds <= seconds:
        raise ValueError("External hard wall limit must exceed the optimizer budget")
    inputs = []
    for value in instances:
        source = Path(value).resolve()
        if not (source / "manifest.json").is_file():
            raise FileNotFoundError(f"An existing standard CSV input is required: {source}")
        data = load_standard(source, case="K2")  # verifies saved file hashes, no solve
        inputs.append((source, data))
    if len({p.name for p, _ in inputs}) != len(inputs):
        raise ValueError("Input directory names must be unique")
    batch.mkdir(parents=True)
    frozen = batch / "frozen"
    frozen.mkdir()
    for folder in ("src", "configs", "cpp"):
        shutil.copytree(ROOT / folder, frozen / folder,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "build*", ".DS_Store"))
    for name in ("main.py", "pyproject.toml"):
        shutil.copy2(ROOT / name, frozen / name)
    (frozen / "bin").mkdir()
    shutil.copy2(binary, frozen / "bin/mlnd_cpp")
    cfg = dict(seconds_per_job=seconds, threads_per_job=threads, parallel_jobs=parallel,
               relative_gap_target=gap, seed=seed, plain=plain,
               cg_max_columns_per_iteration=512, max_benders_iterations=1000000,
               checkpoint_seconds=min(60, max(5, seconds / 4)),
               resource_sample_seconds=resource_sample_seconds, watchdog_grace_seconds=hard_wall_seconds-seconds,
               hard_wall_seconds=hard_wall_seconds,
               resource_limits=dict(job_rss_bytes=int(job_rss_gib*1024**3),
                   batch_rss_bytes=int(batch_rss_gib*1024**3),
                   system_available_floor_bytes=int(system_available_gib*1024**3),
                   sample_seconds=resource_sample_seconds, terminate_grace_seconds=terminate_grace_seconds,
                   pressure_cooldown_seconds=max(10, terminate_grace_seconds+2*resource_sample_seconds)),
               max_orders=2, external_warm_start=None, fixed_design=None,
               full_path_column_limit=0, full_model_variable_limit=0,
               budget_scope="input read, path preparation, model construction and optimization; validation reported separately")
    jobs, identities = [], []
    for source, _ in inputs:
        relative = f"instances/{source.name}"
        shutil.copytree(source, batch / relative)
        identities.append(dict(id=source.name, source=str(source), instance=relative,
                               manifest_sha256=sha256(source / "manifest.json"),
                               source_manifest=read(source / "manifest.json")))
        for method in METHODS:
            jobs.append(dict(id=f"{source.name}_K2_{method}", instance=relative, K=2,
                             method=method, output=f"runs/{source.name}/{method}/K2"))
    fingerprints = {str(p.relative_to(batch)): sha256(p)
                    for base in (frozen, batch / "instances")
                    for p in sorted(base.rglob("*")) if p.is_file()}
    version = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                             capture_output=True, text=True, check=True).stdout.strip()
    manifest = dict(schema="conditional-benchmark/1", created_utc=existing.utc(),
                    protocol=cfg, python=sys.executable, git_head=version,
                    git_status=subprocess.run(["git", "status", "--short"], cwd=ROOT,
                                              capture_output=True, text=True, check=True).stdout,
                    input_origins=identities, jobs=jobs, frozen_sha256=fingerprints)
    write_json(batch / "experiment_manifest.json", manifest)
    write_json(batch / "resource_limits.json", cfg["resource_limits"])
    write_json(batch / "state.json", dict(status="prepared_not_started", updated_utc=existing.utc(),
                                          jobs={j["id"]: dict(status="pending", attempts=[]) for j in jobs}))
    (batch / "run.py").write_text(
        "from pathlib import Path\nimport sys\n"
        "batch = Path(__file__).resolve().parent\n"
        "sys.path.insert(0, str(batch / 'frozen'))\n"
        "from src.experiments.conditional_benchmark_20261022 import main\n"
        "main([*sys.argv[1:], '--batch', str(batch)])\n", encoding="utf-8")
    return dict(batch=str(batch), jobs=len(jobs), input_generation=False, solver_started=False)


def verify(batch):
    batch = Path(batch).resolve()
    manifest = read(batch / "experiment_manifest.json")
    for relative, expected in manifest["frozen_sha256"].items():
        path = batch / relative
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"Frozen artifact changed or missing: {relative}")
    return manifest


def guarded_execute(batch, manifest, job_id, command, attempt, cancel, *, remaining=None,
                    initial_phase="construction", stage_directory=None):
    batch, attempt = Path(batch), Path(attempt)
    cfg = manifest["protocol"]
    if "resource_limits" not in cfg:  # Compatibility with previously frozen batches.
        return existing.execute(command, batch / "frozen", attempt, cfg, cancel)
    limits_file = attempt / "guard_limits.json"
    write_json(limits_file, cfg["resource_limits"])
    wall = cfg["hard_wall_seconds"] if remaining is None else remaining
    guarded = [manifest["python"], "-m", "src.experiments.resource_guard",
               "--output", str(attempt), "--registry", str(batch / "resource_guards"),
               "--key", job_id + "_" + attempt.name, "--limits", str(limits_file),
               "--wall-limit", str(wall), "--initial-phase", initial_phase,
               "--stage-directory", str(stage_directory or attempt), "--", *command]
    execution = existing.execute(guarded, batch / "frozen", attempt,
                                 dict(cfg, seconds_per_job=wall, watchdog_grace_seconds=cfg["resource_limits"]["terminate_grace_seconds"]+5), cancel)
    guard = optional_read(attempt / "guard_status.json")
    execution["solver_command"] = command
    execution["resource_guard"] = guard
    if guard.get("termination_reason"):
        execution["stop_reason"] = guard["termination_reason"]
        execution["resource_limit_kind"] = guard.get("resource_limit_kind")
        execution["stopped_stage"] = guard.get("stage")
    if guard.get("peak_sampled_rss_bytes") is not None:
        execution["sampled_peak_rss_bytes"] = guard["peak_sampled_rss_bytes"]
    if guard.get("solver_usage"):
        execution["solver_resource_usage"] = guard["solver_usage"]
        execution["peak_rss_bytes"] = guard["solver_usage"].get("peak_rss_bytes")
    write_json(attempt / "execution.json", execution)
    return execution


def audit_worker(batch, job_id, attempt):
    batch, attempt = Path(batch).resolve(), Path(attempt).resolve()
    manifest = read(batch / "experiment_manifest.json")
    job = next(job for job in manifest["jobs"] if job["id"] == job_id)
    audit_job = dict(job)
    if job["method"].startswith("benders_"):
        audit_job["method"] = "benders_cplex"
    audit = existing.audit_attempt(batch, manifest, audit_job, attempt)
    state = dict(status="completed" if audit.get("passed") else "failed",
                 directory=str(attempt.relative_to(batch)),
                 result_directory=audit.get("result_directory", str(attempt.relative_to(batch))))
    write_json(attempt / "report_row.json", _record(batch, job, state))
    return audit


def launch(batch):
    """Detach one frozen dispatcher; the tool/terminal lifetime is irrelevant."""
    import psutil
    batch = Path(batch).resolve()
    verify(batch)
    with (batch / ".launch.lock").open("a+") as launch_lock:
        fcntl.flock(launch_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with existing.batch_lock(batch):
            pass  # Refuse launch while a dispatcher already owns the batch.
        previous = optional_read(batch / "dispatcher.json")
        if previous.get("pid"):
            try:
                process = psutil.Process(previous["pid"])
                if process.create_time() == previous.get("create_time") and process.is_running():
                    raise RuntimeError("The recorded background dispatcher is still running")
            except psutil.NoSuchProcess:
                pass
        manifest = read(batch / "experiment_manifest.json")
        with (batch / "dispatcher.log").open("ab", buffering=0) as stream:
            process = subprocess.Popen([manifest["python"], str(batch / "run.py"), "run"],
                cwd=batch / "frozen", stdin=subprocess.DEVNULL, stdout=stream, stderr=stream,
                close_fds=True, start_new_session=True,
                env={**os.environ, **existing.THREAD_ENV, "PYTHONUNBUFFERED": "1"})
        record = dict(pid=process.pid, create_time=psutil.Process(process.pid).create_time(),
                      started_utc=existing.utc(), command=[manifest["python"], str(batch / "run.py"), "run"],
                      log=str(batch / "dispatcher.log"), detached=True)
        write_json(batch / "dispatcher.json", record)
    return record


def live_report(batch, manifest, state):
    """Aggregate only small guarded summaries; never read solution/path JSON."""
    rows = []
    for job in manifest["jobs"]:
        item = state["jobs"][job["id"]]
        attempt = batch / item.get("directory", "_missing")
        row = optional_read(attempt / "report_row.json")
        row.update(job=job["id"], method=job["method"], instance=job["instance"],
                   scheduler_status=item["status"])
        execution = optional_read(attempt / "execution.json")
        guard = optional_read(attempt / "guard_status.json")
        validation = optional_read(attempt / "postprocess/guard_status.json")
        row["stop_reason"] = execution.get("stop_reason") or guard.get("termination_reason") or validation.get("termination_reason") or row.get("stop_reason")
        row["resource_limit_kind"] = guard.get("resource_limit_kind") or validation.get("resource_limit_kind")
        row["stopped_stage"] = guard.get("stage") if guard.get("termination_reason") else validation.get("stage") if validation.get("termination_reason") else row.get("stopped_stage")
        row["peak_sampled_tree_rss_bytes"] = guard.get("peak_sampled_rss_bytes")
        row["validation_peak_sampled_tree_rss_bytes"] = validation.get("peak_sampled_rss_bytes")
        rows.append(row)
    write_json(batch / "live_status.json", dict(updated_utc=existing.utc(), rows=rows))
    fields = ("job", "scheduler_status", "upper_bound", "lower_bound", "gap", "stop_reason", "resource_limit_kind", "stopped_stage", "peak_sampled_tree_rss_bytes", "validation_peak_sampled_tree_rss_bytes")
    temporary = batch / "live_status.csv.tmp"
    with temporary.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(batch / "live_status.csv")
    local_time = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d %H:%M:%S %Z")
    lines = ["# 夜间比较实时状态", "", f"更新时间：{local_time}（北京时间）。只读取受监护验证进程写出的轻量摘要。", "",
             "| 任务 | 状态 | 已验证UB | 全局LB | gap | 停止原因 | 阶段 |", "|---|---|---:|---:|---:|---|---|"]
    for row in rows:
        gap = None if row.get("gap") is None else f"{100*row['gap']:.3f}%"
        values = [row["job"], row["scheduler_status"], row.get("upper_bound"), row.get("lower_bound"), gap, row.get("stop_reason"), row.get("stopped_stage")]
        lines.append("| " + " | ".join("—" if value is None else f"{value:.6g}" if isinstance(value,float) else str(value) for value in values) + " |")
    temporary = batch / "live_status.md.tmp"
    temporary.write_text("\n".join(lines)+"\n", encoding="utf-8")
    temporary.replace(batch / "live_status.md")


def run(batch, *, only=None, retry_failed=False):
    batch = Path(batch).resolve()
    manifest = verify(batch)
    cancel, lock = threading.Event(), threading.Lock()
    with existing.batch_lock(batch):
        state = read(batch / "state.json")
        if any(item["status"] == "running" for item in state["jobs"].values()):
            raise RuntimeError("A running/interrupted dispatcher must be reconciled before resuming")
        selected = set(only or [j["id"] for j in manifest["jobs"]])
        if selected - state["jobs"].keys():
            raise ValueError(f"Unknown job IDs: {selected - state['jobs'].keys()}")

        def save():
            state["updated_utc"] = existing.utc()
            write_json(batch / "state.json", state)
            live_report(batch, manifest, state)

        def one(job):
            if cancel.is_set():
                return
            with lock:
                item = state["jobs"][job["id"]]
                number = len(item["attempts"]) + 1
                relative = f"{job['output']}/attempts/{number:03d}"
                attempt = batch / relative
                attempt.mkdir(parents=True, exist_ok=False)
                record = dict(directory=relative, status="running", started_utc=existing.utc())
                item["attempts"].append(record)
                item.update(status="running", directory=relative)
                save()
            command = command_for(batch, manifest, job, attempt)
            write_json(attempt / "job.json", dict(job=job, command=command,
                       experiment_manifest_sha256=sha256(batch / "experiment_manifest.json")))
            print(json.dumps(dict(event="job_started", job=job["id"], utc=existing.utc())), flush=True)
            started = time.monotonic()
            try:
                execution = guarded_execute(batch, manifest, job["id"], command, attempt, cancel)
                remaining = manifest["protocol"].get("hard_wall_seconds", manifest["protocol"]["seconds_per_job"]+120) - (time.monotonic()-started)
                if remaining > 1 and not cancel.is_set():
                    postprocess = attempt / "postprocess"
                    postprocess.mkdir()
                    audit_command = [manifest["python"], "-m", "src.experiments.conditional_benchmark_20261022",
                                     "audit-worker", "--batch", str(batch), "--job-id", job["id"], "--attempt", str(attempt)]
                    checked = guarded_execute(batch, manifest, job["id"]+"_audit", audit_command,
                        postprocess, cancel, remaining=remaining, initial_phase="validation", stage_directory=postprocess)
                    audit = optional_read(attempt / "artifact_audit.json")
                    if not audit or checked.get("stop_reason") or checked.get("returncode") != 0:
                        audit.update(passed=False, errors=[*(audit.get("errors") or []),
                                     checked.get("stop_reason") or "validation_process_failed"],
                                     validation_stop_reason=checked.get("stop_reason"),
                                     validation_resource_limit_kind=checked.get("resource_limit_kind"))
                        write_json(attempt / "artifact_audit.json", audit)
                else:
                    audit = dict(passed=False, errors=["validation_not_run_after_external_wall_limit_or_interrupt"],
                                 execution_stop_reason=execution.get("stop_reason"))
                    write_json(attempt / "artifact_audit.json", audit)
            except Exception as exc:
                audit = dict(passed=False, errors=[f"{type(exc).__name__}: {exc}"])
                write_json(attempt / "dispatcher_failure.json", audit)
            with lock:
                status = "completed" if audit["passed"] else "interrupted" if cancel.is_set() else "failed"
                record.update(status=status, finished_utc=existing.utc(), audit=audit)
                item.update(status=status, result_directory=audit.get("result_directory", relative))
                save()
            print(json.dumps(dict(event="job_finished", job=job["id"], status=status,
                                  utc=existing.utc())), flush=True)

        jobs = [j for j in manifest["jobs"] if j["id"] in selected and
                (state["jobs"][j["id"]]["status"] == "pending" or
                 retry_failed and state["jobs"][j["id"]]["status"] in ("failed", "interrupted"))]
        handlers = {}
        if threading.current_thread() is threading.main_thread():
            for signum in (signal.SIGINT, signal.SIGTERM):
                handlers[signum] = signal.signal(signum, lambda *_: cancel.set())
        state["status"] = "running"
        save()
        try:
            with ThreadPoolExecutor(max_workers=manifest["protocol"]["parallel_jobs"]) as pool:
                list(pool.map(one, jobs))
        finally:
            for signum, handler in handlers.items():
                signal.signal(signum, handler)
            state["status"] = "interrupted" if cancel.is_set() else "finished_dispatch"
            save()
    try:
        if "resource_limits" not in manifest["protocol"]:
            return report(batch)
        report_dir = batch / "report_process" / existing.utc().replace(":", "-")
        report_dir.mkdir(parents=True)
        command = [manifest["python"], "-m", "src.experiments.conditional_benchmark_20261022",
                   "report", "--batch", str(batch)]
        execution = guarded_execute(batch, manifest, "final_report", command, report_dir,
                                    threading.Event(), remaining=300, initial_phase="validation")
        if execution.get("returncode") != 0 or execution.get("stop_reason"):
            raise RuntimeError(f"Final report guard stopped: {execution.get('stop_reason')} / {execution.get('returncode')}")
        return dict(report=str(batch / "analysis/REPORT.md"), live_status=str(batch / "live_status.md"))
    except Exception as exc:
        failure = dict(error=f"{type(exc).__name__}: {exc}", utc=existing.utc(),
                       jobs_finished=True, raw_evidence_preserved=True)
        write_json(batch / "report_failure.json", failure)
        return failure


def certified_bound(result):
    if result.get("bound_scope") == "global":
        return result.get("bound")
    if (result.get("bound_scope") is None and result.get("method") == "full_path"
            and result.get("catalogue_complete") is True
            and result.get("construction_complete") is True):
        return result.get("bound")
    return None


def _size(row, prefix, value):
    for key in ("variables", "linear_rows", "cones"):
        row[f"{prefix}_{key}"] = (value or {}).get(key)


def _record(batch, job, state):
    row = dict(job=job["id"], instance=job["instance"], K=2, method=job["method"],
               scheduler_status=state["status"], validation_passed=False,
               upper_bound=None, lower_bound=None, gap=None, all_service_paths=None,
               all_empty_paths=None, all_paths=None, full_path_count_source=None)
    source = batch / state.get("result_directory", state.get("directory", "_missing"))
    attempt = batch / state.get("directory", "_missing")
    execution = optional_read(attempt / "execution.json")
    guard = optional_read(attempt / "guard_status.json")
    validation_execution = optional_read(attempt / "postprocess/execution.json")
    validation_guard = optional_read(attempt / "postprocess/guard_status.json")
    stop = execution.get("stop_reason") or guard.get("termination_reason")
    validation_stop = validation_execution.get("stop_reason") or validation_guard.get("termination_reason")
    kind = execution.get("resource_limit_kind") or guard.get("resource_limit_kind")
    row.update(returncode=execution.get("returncode"), stop_reason=stop,
               resource_limit_kind=kind, stopped_stage=guard.get("stage"),
               validation_stop_reason=validation_stop,
               validation_resource_limit_kind=validation_guard.get("resource_limit_kind"),
               resource_unaffordable=bool(kind and ("rss_limit" in kind or "memory_floor" in kind)
                   or validation_guard.get("resource_limit_kind") in ("per_job_rss_limit", "batch_rss_limit", "system_available_memory_floor")),
               termination_reason=stop or validation_stop or state["status"],
               external_wall_seconds=execution.get("external_wall_seconds"),
               peak_rss_bytes=execution.get("peak_rss_bytes"),
               peak_sampled_tree_rss_bytes=guard.get("peak_sampled_rss_bytes"),
               validation_peak_sampled_tree_rss_bytes=validation_guard.get("peak_sampled_rss_bytes"))
    result_path = source / "result.json"
    if not result_path.is_file():
        return row
    result = optional_read(result_path)
    if not result:
        row["result_error"] = "missing_or_invalid_result_json"
        return row
    validation = optional_read(source / "native_validation.json")
    valid = bool(validation.get("passed"))
    for name, key in (("result.json", "result_sha256"), ("solution.json", "solution_sha256"),
                      ("path_flows.json", "path_flows_sha256")):
        valid = valid and (source / name).is_file() and validation.get(key) == sha256(source / name)
    input_sha = sha256(batch / job["instance"] / "manifest.json")
    valid = valid and validation.get("manifest_sha256") == input_sha
    expected_solver = "Gurobi" if job["method"] == "full_gurobi" else "CPLEX"
    expected_method = "full_path" if job["method"].startswith("full_") else "benders_cg"
    valid = valid and result.get("max_orders") == 2 and result.get("solver") == expected_solver
    valid = valid and result.get("method") == expected_method
    if job["method"] == "benders_conditional":
        valid = valid and result.get("demand_formulation") == "conditional"
    row.update(result_directory=str(source), result_sha256=sha256(result_path),
               input_manifest_sha256=input_sha, validation_passed=valid,
               termination_reason=stop or validation_stop or result.get("termination_reason"),
               saved_result_termination_reason=result.get("termination_reason"),
               solver_status=result.get("solver_status"),
               fallback=result.get("fallback", result.get("incumbent_kind") in ("all_courier_fallback", "all_courier_baseline")),
               external_wall_seconds=execution.get("external_wall_seconds"),
               peak_rss_bytes=execution.get("peak_rss_bytes"),
               catalogue_complete=result.get("catalogue_complete"),
               construction_complete=result.get("construction_complete"),
               pricing_complete=result.get("pricing_complete"),
               pricing_complete_scope=result.get("pricing_complete_scope"),
               demand_formulation=result.get("demand_formulation"),
               flow_policy=result.get("flow_policy"),
               validation_seconds=validation.get("validation_and_export_seconds"))
    if valid:
        row["upper_bound"] = result.get("objective")
        row["lower_bound"] = certified_bound(result)
        if row["upper_bound"] is not None and row["lower_bound"] is not None:
            row["gap"] = (row["upper_bound"] - row["lower_bound"]) / max(1, abs(row["upper_bound"]))
    timing = result.get("timings", {})
    try:
        phases = native_phase_times(result)
        row["solve_seconds"] = phases["solve_seconds"]
    except (ValueError, TypeError) as exc:
        row["solve_seconds"] = None
        row["timing_audit_error"] = str(exc)
    for key in ("build", "optimize", "master_build", "master_optimize", "subproblem_wall"):
        row[key + "_seconds"] = timing.get(key)
    for key, value in timing.get("subproblem", {}).items():
        row["subproblem_" + key + "_seconds"] = value
    row["enumeration_seconds"] = result.get("enumeration_seconds", timing.get("enumeration"))
    row["support_preprocessing_seconds"] = result.get("support_preprocessing_seconds")
    _size(row, "full_model", result.get("model_size") if job["method"].startswith("full_") else None)
    _size(row, "master", result.get("master_model_size"))
    _size(row, "subproblem_last", result.get("subproblem_model_size"))
    _size(row, "subproblem_peak", result.get("subproblem_model_size_peak"))
    generated = result.get("generated_columns") or {}
    row.update(generated_service_columns=generated.get("service"), generated_empty_columns=generated.get("empty"),
               initial_service_columns=result.get("initial_service_columns"),
               initial_empty_columns=result.get("initial_empty_columns"))
    for key in ("iterations", "cg_iterations", "cuts"):
        value = result.get(key)
        row[key + "_count"] = len(value) if isinstance(value, list) else value
    for key in ("acceleration_statistics", "accelerations", "conditional_statistics"):
        value = result.get(key)
        row[key] = json.dumps(value, sort_keys=True) if isinstance(value, (list, dict)) else value
    catalogue = result.get("catalogue") or {}
    if result.get("catalogue_complete") is True and "service" in catalogue and "empty" in catalogue:
        row.update(all_service_paths=catalogue["service"], all_empty_paths=catalogue["empty"],
                   all_paths=catalogue["service"] + catalogue["empty"],
                   full_path_count_source=job["id"], initial_segments=catalogue.get("segments"))
    return row


def report(batch):
    batch = Path(batch).resolve()
    manifest = verify(batch)
    state = read(batch / "state.json")
    rows = [_record(batch, job, state["jobs"][job["id"]]) for job in manifest["jobs"]]
    counts = {}
    for row in rows:
        if row["all_paths"] is not None:
            identity = row["instance"]
            previous = counts.get(identity)
            if previous and (previous["all_service_paths"], previous["all_empty_paths"]) != (row["all_service_paths"], row["all_empty_paths"]):
                raise ValueError(f"Complete catalogues disagree for {identity}")
            counts[identity] = row
    for row in rows:
        if row["all_paths"] is None and row["instance"] in counts:
            reference = counts[row["instance"]]
            for key in ("all_paths", "all_service_paths", "all_empty_paths", "initial_segments"):
                row[key] = reference.get(key)
            row["full_path_count_source"] = reference["job"]
    output = batch / "analysis"
    output.mkdir(exist_ok=True)
    write_json(output / "comparison.json", dict(protocol=manifest["protocol"], rows=rows))
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (output / "comparison.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    def fmt(value):
        return "—" if value is None else f"{value:,.3f}" if isinstance(value, float) else str(value)
    lines = ["# K=2 条件 Benders 算法比较", "",
             "所有输入均为既有标准实例的逐文件复制。模型、需求重构、首段流量策略及物理限制沿用冻结代码。",
             f"每组预算 {manifest['protocol']['seconds_per_job']} 秒，线程 {manifest['protocol']['threads_per_job']}，"
             f"并发 {manifest['protocol']['parallel_jobs']}，目标 gap={manifest['protocol']['relative_gap_target']}。", "",
             "UB 仅展示独立物理验证及文件哈希通过的可行解；LB 仅展示具有全局作用域的界。"
             "限时结果不表示已证明最优。失败状态与可行性、构造完成性及间隙分别记录。", "",
             "| 实例 | 方法 | UB | LB | gap | 预算内耗时 s | 进程墙钟 s | 终止 | 验证 |",
             "|---|---|---:|---:|---:|---:|---:|---|---|"]
    for row in rows:
        lines.append("| " + " | ".join(fmt(v) for v in (
            Path(row["instance"]).name, row["method"], row["upper_bound"], row["lower_bound"],
            None if row["gap"] is None else f"{100*row['gap']:.3f}%", row.get("solve_seconds"),
            row.get("external_wall_seconds"), row.get("termination_reason", row["scheduler_status"]),
            row["validation_passed"])) + " |")
    lines += ["", "| 实例 | 方法 | 完整模型变量/行/锥 | 主问题变量/行/锥 | 子问题最大变量/行/锥 | 全路径 | 已生成服务/空飞列 |",
              "|---|---|---:|---:|---:|---:|---:|"]
    for row in rows:
        sizes = [" / ".join(fmt(row.get(prefix + "_" + key)) for key in ("variables", "linear_rows", "cones"))
                 for prefix in ("full_model", "master", "subproblem_peak")]
        lines.append("| " + " | ".join([Path(row["instance"]).name, row["method"], *sizes,
            fmt(row["all_paths"]), f"{fmt(row.get('generated_service_columns'))} / {fmt(row.get('generated_empty_columns'))}"]) + " |")
    lines += ["", "全路径计数仅来自同输入、同 K 的完整目录，来源在 CSV 中逐行列明。"
              "列生成的已生成列数不等于完整路径数。Benders 子问题规模随列池变化，报告末次与最大规模；"
              "主问题与子问题规模不能简单相加并与完整模型比较。", "",
              "原始日志、解、路径、验证和资源采样保存在各 attempts 目录。"
              "comparison.csv/JSON 另含各阶段耗时、构造/定价状态、源码与输入清单引用。"]
    if manifest["protocol"].get("resource_limits"):
        limits = manifest["protocol"]["resource_limits"]
        lines += ["", "## 资源终止与外部时限", "",
                  f"单任务进程树RSS上限 {limits['job_rss_bytes']/1024**3:g} GiB，批次同时运行RSS合计上限 "
                  f"{limits['batch_rss_bytes']/1024**3:g} GiB，系统可用内存保底 "
                  f"{limits['system_available_floor_bytes']/1024**3:g} GiB，"
                  f"每任务含验证外部墙钟上限 {manifest['protocol']['hard_wall_seconds']:g} 秒。", "",
                  "construction_memory_limit / solve_memory_limit / validation_memory_limit 表示当前资源限额下不可承受，"
                  "不表示数学不可行。读取RSS失败会终止相应任务。没有最终JSON时仍保留监护记录、日志和已落盘检查点。", "",
                  "| 任务 | 资源原因 | 阶段 | 求解进程树峰值 GiB | 验证终止 | 验证峰值 GiB |", "|---|---|---|---:|---|---:|"]
        for row in rows:
            lines.append("| " + " | ".join(fmt(value) for value in (
                row["job"], row.get("resource_limit_kind"), row.get("stopped_stage"),
                None if row.get("peak_sampled_tree_rss_bytes") is None else row["peak_sampled_tree_rss_bytes"]/1024**3,
                row.get("validation_stop_reason"), None if row.get("validation_peak_sampled_tree_rss_bytes") is None
                else row["validation_peak_sampled_tree_rss_bytes"]/1024**3)) + " |")
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return dict(report=str(output / "REPORT.md"), jobs=len(rows),
                validated=sum(row["validation_passed"] for row in rows))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "verify", "run", "launch", "report", "status", "audit-worker"), nargs="?", default="status")
    parser.add_argument("--batch", type=Path, default=DEFAULT_BATCH)
    parser.add_argument("--instances", type=Path, nargs="+")
    parser.add_argument("--levels", type=int, nargs="+", choices=range(1, 11))
    parser.add_argument("--binary", type=Path)
    parser.add_argument("--seconds", type=float, default=180)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--parallel", type=int, default=1)
    parser.add_argument("--gap", type=float, default=0.005)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--plain", action="store_true")
    parser.add_argument("--jobs", nargs="+")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--job-rss-gib", type=float, default=4)
    parser.add_argument("--batch-rss-gib", type=float, default=8)
    parser.add_argument("--system-available-gib", type=float, default=2)
    parser.add_argument("--hard-wall-seconds", type=float)
    parser.add_argument("--resource-sample-seconds", type=float, default=.5)
    parser.add_argument("--terminate-grace-seconds", type=float, default=5)
    parser.add_argument("--job-id")
    parser.add_argument("--attempt", type=Path)
    args = parser.parse_args(argv)
    if args.action == "prepare":
        if args.instances and args.levels:
            parser.error("Choose existing --instances or --levels, not both")
        levels = args.levels or (1, 2, 3)
        if len(set(levels)) != len(levels):
            parser.error("Levels must not repeat")
        instances = args.instances or [ROOT / f"outputs/tsl_scale_ladder/instances/L{n:02d}" for n in levels]
        result = prepare(args.batch, instances, binary=args.binary, seconds=args.seconds,
                         threads=args.threads, parallel=args.parallel, gap=args.gap,
                         seed=args.seed, plain=args.plain, job_rss_gib=args.job_rss_gib,
                         batch_rss_gib=args.batch_rss_gib, system_available_gib=args.system_available_gib,
                         hard_wall_seconds=args.hard_wall_seconds,
                         resource_sample_seconds=args.resource_sample_seconds,
                         terminate_grace_seconds=args.terminate_grace_seconds)
    elif args.action == "verify":
        result = dict(passed=bool(verify(args.batch)), solver_started=False)
    elif args.action == "run":
        result = run(args.batch, only=args.jobs, retry_failed=args.retry_failed)
    elif args.action == "launch":
        result = launch(args.batch)
    elif args.action == "audit-worker":
        if not args.job_id or not args.attempt:
            parser.error("audit-worker requires --job-id and --attempt")
        result = audit_worker(args.batch, args.job_id, args.attempt)
    elif args.action == "report":
        result = report(args.batch)
    else:
        result = existing.status_summary(args.batch)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
