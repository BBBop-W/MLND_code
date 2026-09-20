"""Run saved algorithm-performance cases in a detached, bounded process pool.

Each batch executes an immutable source snapshot. ``STOP`` in its directory
stops admission and requests a checkpoint-preserving stop of active runs.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import traceback

from src.config import load_config
from src.solvers.methods import METHODS
from .progress import utc_now
from .storage import ROOT, sha256, snapshot_sources, write_json


THREAD_ENV = {
    name: "1"
    for name in (
        "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS",
    )
}
SUMMARY_FIELDS = (
    "id", "source", "region", "H", "L", "E", "OD", "method", "state",
    "pid", "exit_code", "stop_reason", "elapsed_seconds", "peak_rss_bytes",
    "status", "passed", "objective", "bound", "gap", "wall", "served", "output",
)


def _absolute(path, root=ROOT):
    path = Path(path)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _gurobi_defaults():
    import gurobipy as gp

    with gp.Env(params={"OutputFlag": 0}) as environment:
        with gp.Model(env=environment) as model:
            return {"mip_gap": model.Params.MIPGap, "mip_gap_abs": model.Params.MIPGapAbs}


def prepare(manifest, output, *, config=None, hubs=(10, 20, 30), workers=3,
            seconds=7200, methods=tuple(METHODS)):
    """Freeze the requested matrix and code without starting any optimization."""
    output = _absolute(output)
    if output.exists():
        raise FileExistsError(f"Preserving existing batch: {output}")
    if workers < 1 or seconds <= 0:
        raise ValueError("Workers and seconds must be positive")
    manifest = _absolute(manifest)
    family = json.loads(manifest.read_text())
    methods = list(dict.fromkeys(methods))
    if not methods or any(name not in METHODS for name in methods):
        raise ValueError("Choose one or more of the five registered methods")
    cases = [case for case in family["cases"] if case["H"] in hubs]
    if not cases or set(hubs) != {case["H"] for case in cases}:
        raise ValueError("Every requested H must occur in the saved manifest")
    ids = [case["id"] for case in cases]
    if len(set(ids)) != len(ids) or any(
        not re.fullmatch(r"[A-Za-z0-9_-]+", name) for name in ids
    ):
        raise ValueError("Case IDs must be unique safe directory names")
    defaults = _gurobi_defaults()
    options = dict(time_limit=float(seconds), threads=1, **defaults,
                   column_limit=None, graph_limit=None)
    records = []
    configurations = {}
    for case in sorted(cases, key=lambda row: (row["H"], row["source"], str(row["region"]))):
        source = _absolute(case["input_path"])
        hashes = {name: sha256(source / name)
                  for name in ("input_arrays.npz", "input_metadata.json")}
        configurations[case["id"]] = load_config(
            case.get("config") or config or family.get("config"),
            overrides={"solver": options},
        )
        for method in methods:
            job_id = f"{case['id']}__{method}"
            records.append({
                **{key: case[key] for key in ("source", "region", "H", "L", "E", "OD")},
                "id": job_id, "instance": case["id"], "method": method,
                "input_path": str(source), "input_sha256": hashes,
                "config": str(output / "configs" / f"{case['id']}.json"),
                "output": str(output / "runs" / job_id),
                "log": str(output / "logs" / f"{job_id}.log"),
            })
    output.mkdir(parents=True)
    write_json(output / "input_manifest.json", family)
    for name, configuration in configurations.items():
        write_json(output / "configs" / f"{name}.json", configuration)
    plan = {
        "schema": 1, "family": "algorithm performance", "created_utc": utc_now(),
        "manifest": str(manifest), "manifest_sha256": sha256(manifest),
        "workers": workers, "seconds": float(seconds), "solver": options,
        "gap_policy": "Defaults read from a fresh local Gurobi model", "gurobi_defaults": defaults,
        "termination_grace_seconds": 120, "kill_grace_seconds": 180,
        "memory_limit_bytes": None, "source_snapshot": str(output / "source_snapshot"),
        "python": sys.executable, "sources": snapshot_sources(output), "jobs": records,
    }
    write_json(output / "plan.json", plan)
    _write_status(output, {"status": "prepared", "jobs": [dict(job, state="queued")
                                                          for job in records]})
    return plan


def _environment():
    return dict(os.environ, **THREAD_ENV, PYTHONUNBUFFERED="1")


def launch(batch):
    """Detach the prepared supervisor; its subprocesses inherit no terminal."""
    batch = _absolute(batch)
    plan = json.loads((batch / "plan.json").read_text())
    if (batch / "launcher.json").exists():
        raise FileExistsError("This batch has already been launched")
    with (batch / "supervisor.log").open("ab", buffering=0) as log:
        process = subprocess.Popen(
            [plan["python"], "-m", "src.experiments.performance", "supervise",
             "--batch", str(batch)], cwd=plan["source_snapshot"], env=_environment(),
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    record = {"pid": process.pid, "started_utc": utc_now(), "output": str(batch)}
    write_json(batch / "launcher.json", record)
    return record


def _spawn(plan, batch, job):
    Path(job["log"]).parent.mkdir(parents=True, exist_ok=True)
    with Path(job["log"]).open("ab", buffering=0) as log:
        return subprocess.Popen(
            [plan["python"], "-m", "src.experiments.performance", "worker",
             "--batch", str(batch), "--job", job["id"]],
            cwd=plan["source_snapshot"], env=_environment(),
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True,
        )


def _write_status(batch, state):
    state["updated_utc"] = utc_now()
    state["counts"] = {
        key: sum(row["state"] == key for row in state["jobs"])
        for key in ("queued", "running", "finished", "failed", "cancelled")
    }
    write_json(batch / "status.json", state)
    temporary = batch / "summary.csv.tmp"
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(state["jobs"])
    temporary.replace(batch / "summary.csv")


def _resources(pid):
    try:
        import psutil
        available = psutil.virtual_memory().available
        try:
            rss = psutil.Process(pid).memory_info().rss
        except (psutil.Error, OSError):
            rss = None
        return rss, available
    except (ImportError, OSError):
        return None, None


def _signal_child(process, *, force=False):
    try:
        process.kill() if force else process.terminate()
    except ProcessLookupError:
        pass  # The child may exit between poll() and the signal.


def _request_stop(batch, job, reason, now):
    if job.get("stop_reason"):
        return
    job.update(stop_reason=reason, stop_requested_utc=utc_now(), stop_clock=now)
    output = Path(job["output"])
    # Do not create this directory before run_case, which preserves existing runs.
    if output.exists():
        (output / "STOP").touch()
    write_json(batch / "stop_requests" / f"{job['id']}.json",
               {"reason": reason, "utc": utc_now()})


def supervise(batch, *, poll_seconds=1.0):
    """Keep at most the requested workers alive; one failed run never aborts others."""
    batch = _absolute(batch)
    plan = json.loads((batch / "plan.json").read_text())
    lock = batch / "supervisor.lock"
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.write(descriptor, str(os.getpid()).encode())
    os.close(descriptor)
    state = {"status": "running", "pid": os.getpid(), "started_utc": utc_now(),
             "jobs": [dict(job, state="queued") for job in plan["jobs"]]}
    running = {}
    handlers = {}

    def stopped(*_):
        (batch / "STOP").touch()

    for sig in (signal.SIGTERM, signal.SIGINT):
        handlers[sig] = signal.signal(sig, stopped)
    try:
        while True:
            now = time.monotonic()
            stopping = (batch / "STOP").exists()
            for job in state["jobs"]:
                if stopping and job["state"] == "queued":
                    job["state"] = "cancelled"
                if job["id"] not in running:
                    continue
                process, started = running[job["id"]]
                job["elapsed_seconds"] = now - started
                code = process.poll()
                if code is not None:
                    job.update(exit_code=code, finished_utc=utc_now(), state="finished")
                    summary_path = Path(job["output"]) / "summary.json"
                    if summary_path.exists():
                        try:
                            summary = json.loads(summary_path.read_text())
                            job.update({key: summary.get(key) for key in (
                                "status", "passed", "objective", "bound", "gap", "wall", "served"
                            )})
                        except (OSError, ValueError) as error:
                            job.update(status="failed", error=f"Cannot read result: {error}")
                    if code != 0 or not summary_path.exists() or job.get("status") == "failed":
                        job["state"] = "failed"
                    del running[job["id"]]
                    continue
                rss, available = _resources(process.pid)
                state["available_memory_bytes"] = available
                job["rss_bytes"] = rss
                if rss is not None:
                    job["peak_rss_bytes"] = max(job.get("peak_rss_bytes", 0), rss)
                if stopping or now - started >= plan["seconds"]:
                    _request_stop(batch, job, "user_stop" if stopping else "time_limit", now)
                if job.get("stop_reason"):
                    output = Path(job["output"])
                    if output.exists():
                        (output / "STOP").touch()
                    delay = now - job["stop_clock"]
                    if delay >= plan["kill_grace_seconds"] and not job.get("kill_sent"):
                        _signal_child(process, force=True)
                        job["kill_sent"] = True
                    elif delay >= plan["termination_grace_seconds"] and not job.get("term_sent"):
                        _signal_child(process)
                        job["term_sent"] = True
            if not stopping:
                for job in state["jobs"]:
                    if len(running) >= plan["workers"]:
                        break
                    if job["state"] != "queued":
                        continue
                    try:
                        process = _spawn(plan, batch, job)
                        running[job["id"]] = (process, time.monotonic())
                        job.update(state="running", pid=process.pid, started_utc=utc_now())
                    except Exception as error:
                        job.update(state="failed", status="launch_failed", error=str(error))
            _write_status(batch, state)
            if not running and not any(job["state"] == "queued" for job in state["jobs"]):
                break
            time.sleep(poll_seconds)
        state.update(status="stopped" if stopping else "finished", finished_utc=utc_now())
        _write_status(batch, state)
        return state
    finally:
        # Unexpected supervisor errors must not orphan unrestricted workers.
        for job_id, (process, _) in running.items():
            output = Path(next(row["output"] for row in state["jobs"] if row["id"] == job_id))
            if output.exists():
                (output / "STOP").touch()
            if process.poll() is None:
                _signal_child(process)
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    _signal_child(process, force=True)
                    process.wait()
        for sig, handler in handlers.items():
            signal.signal(sig, handler)


def worker(batch, job_id):
    from .runner import run_case

    batch = _absolute(batch)
    plan = json.loads((batch / "plan.json").read_text())
    job = next(job for job in plan["jobs"] if job["id"] == job_id)
    for name, digest in job["input_sha256"].items():
        if sha256(Path(job["input_path"]) / name) != digest:
            raise RuntimeError(f"Saved input changed after batch launch: {job_id}/{name}")
    summary = run_case(job["H"], job["output"], config=job["config"],
                       method=job["method"], options=plan["solver"],
                       input_path=job["input_path"])
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 1 if summary["status"] == "failed" else 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("plan", "launch"):
        command_parser = subparsers.add_parser(command)
        command_parser.add_argument("--manifest", required=True)
        command_parser.add_argument("--out", required=True)
        command_parser.add_argument("--config")
        command_parser.add_argument("--hubs", nargs="+", type=int, default=[10, 20, 30])
        command_parser.add_argument("--workers", type=int, default=3)
        command_parser.add_argument("--seconds", type=float, default=7200)
    for command in ("supervise", "worker", "jobs"):
        command_parser = subparsers.add_parser(command)
        command_parser.add_argument("--batch", required=True)
        if command == "worker":
            command_parser.add_argument("--job", required=True)
    args = parser.parse_args(argv)
    if args.command in ("plan", "launch"):
        plan = prepare(args.manifest, args.out, config=args.config, hubs=args.hubs,
                       workers=args.workers, seconds=args.seconds)
        result = launch(args.out) if args.command == "launch" else {
            "jobs": len(plan["jobs"]), "output": str(_absolute(args.out)), "status": "prepared"
        }
    elif args.command == "supervise":
        result = supervise(args.batch)
    elif args.command == "worker":
        try:
            return worker(args.batch, args.job)
        except Exception:
            traceback.print_exc()
            return 1
    else:
        result = json.loads((_absolute(args.batch) / "status.json").read_text())
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
