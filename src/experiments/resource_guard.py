"""Per-process-group RSS/wall guardian, with a locked batch pressure lease.

Runs inside process_worker.py's dedicated process group. It survives a solver
SIGTERM long enough to save the stop reason, then escalates to group SIGKILL.
Only our recorded process groups are signalled; global process enumeration is
not needed on macOS. The dispatcher never imports or validates large solutions.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
import ctypes
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import psutil


def utc():
    return datetime.now(timezone.utc).isoformat()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def read(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {} if default is None else default


def group_pids(group):
    if sys.platform != "darwin":
        found = []
        for pid in psutil.pids():
            try:
                if os.getpgid(pid) == group:
                    found.append(pid)
            except ProcessLookupError:
                pass
        return found
    lib = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
    function = lib.proc_listpgrppids
    function.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_int]
    function.restype = ctypes.c_int
    capacity = 64
    while capacity <= 65536:
        buffer = (ctypes.c_int * capacity)()
        ctypes.set_errno(0)
        count = function(group, buffer, ctypes.sizeof(buffer))
        if count < 0 or (count == 0 and ctypes.get_errno() not in (0, 3)):
            raise PermissionError(ctypes.get_errno(), "scoped proc_listpgrppids failed")
        if count < capacity:
            return [value for value in buffer[:count] if value > 0]
        capacity *= 2
    raise RuntimeError("Process tree exceeds scoped monitoring capacity")


def tree_sample(root):
    rss, processes = 0, 0
    members = group_pids(root)
    for pid in members:
        try:
            process = psutil.Process(pid)
            rss += process.memory_info().rss
            processes += 1
        except psutil.NoSuchProcess:
            continue
        # Access failures deliberately propagate: unavailable RSS is not zero.
    if not processes:
        raise RuntimeError("No live process was accessible to RSS monitor")
    memory = psutil.virtual_memory()
    return dict(rss_bytes=rss, process_count=processes, process_ids=members,
                system_available_bytes=memory.available, system_total_bytes=memory.total)


def resource_decision(entries, current, limits, available, lease, now):
    """Pure selection: at most one global-pressure victim during a lease."""
    own = entries[current]
    if limits["job_rss_bytes"] and own["rss_bytes"] > limits["job_rss_bytes"]:
        return dict(victim=current, reason="per_job_rss_limit", requested=now,
                    expires=now + limits["pressure_cooldown_seconds"])
    if lease and lease.get("expires", 0) > now:
        return lease
    active = {key: value for key, value in entries.items() if value.get("state") == "running"}
    if any(now - value.get("sample_monotonic", 0) >= 10 for value in active.values()):
        raise RuntimeError("A registered running guardian has stopped reporting RSS")
    total = sum(value["rss_bytes"] for value in active.values())
    reason = None
    if limits["batch_rss_bytes"] and total > limits["batch_rss_bytes"]:
        reason = "batch_rss_limit"
    elif limits["system_available_floor_bytes"] and available < limits["system_available_floor_bytes"]:
        reason = "system_available_memory_floor"
    if not reason or not active:
        return None
    victim = max(active, key=lambda key: (active[key]["rss_bytes"], key))
    return dict(victim=victim, reason=reason, requested=now,
                expires=now + limits["pressure_cooldown_seconds"],
                batch_rss_bytes=total, system_available_bytes=available)


@contextmanager
def registry_lock(directory):
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".lock").open("a+") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def update_registry(directory, key, entry, limits, sample):
    with registry_lock(directory):
        write(directory / (key + ".json"), entry)
        entries = {path.stem: read(path) for path in directory.glob("*.json")
                   if path.name != "pressure_lease.json"}
        if any("rss_bytes" not in value for value in entries.values()):
            raise RuntimeError("A registered guardian has an unreadable RSS record")
        lease = read(directory / "pressure_lease.json")
        decision = resource_decision(entries, key, limits, sample["system_available_bytes"],
                                     lease, entry["sample_monotonic"])
        if decision:
            write(directory / "pressure_lease.json", decision)
        return decision


def phase_from(progress, previous="construction"):
    stage = str(progress.get("stage", "")).lower()
    event = str(progress.get("event", "")).lower()
    if "validation" in stage or "export" in stage or stage == "audit":
        return "validation"
    if any(word in event for word in ("model_build", "extraction")):
        return "construction"
    if any(word in stage for word in ("construction", "enumeration", "support", "setup")) or stage in ("read", "input_read", "build"):
        return "construction"
    if "solver_solve" in event or stage in ("optimize", "root", "solve"):
        return "solve"
    return previous


def supervise(command, output, registry, key, limits, wall_limit, *, stage_directory=None, initial_phase="construction"):
    output, registry = Path(output), Path(registry)
    stage_directory = Path(stage_directory or output)
    output.mkdir(parents=True, exist_ok=True)
    group = os.getpgrp()
    if group != os.getppid():
        raise RuntimeError("Guardian must run under process_worker in its dedicated process group")
    begun = time.monotonic()
    received = []
    handlers = {}
    for sig in (signal.SIGTERM, signal.SIGINT):
        handlers[sig] = signal.signal(sig, lambda value, _frame: received.append(value))
    child = subprocess.Popen(command)
    record = dict(started_utc=utc(), supervisor_pid=os.getpid(), process_group=group,
                  solver_pid=child.pid, solver_create_time=psutil.Process(child.pid).create_time(),
                  command=command, limits=limits, wall_limit_seconds=wall_limit,
                  status="running", peak_sampled_rss_bytes=None, stage=initial_phase)
    write(output / "guard_status.json", record)
    reason, stopped, phase, usage = None, None, initial_phase, None
    entry = dict(state="running", rss_bytes=0, sample_monotonic=begun, process_group=group)
    code = 125
    child_exited = False
    with (output / "guard_resources.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("utc", "elapsed_seconds", "rss_bytes", "process_count", "system_available_bytes", "system_total_bytes", "phase"))
        writer.writeheader()
        while True:
            if not child_exited:
                waited, status, usage_now = os.wait4(child.pid, os.WNOHANG)
                if waited:
                    code, usage = os.waitstatus_to_exitcode(status), usage_now
                    child.returncode = code
                    child_exited = True
            now = time.monotonic()
            progress = read(stage_directory / "progress.json")
            phase = phase_from(progress, phase)
            try:
                sample = tree_sample(group)
                record["peak_sampled_rss_bytes"] = max(record["peak_sampled_rss_bytes"] or 0, sample["rss_bytes"])
                entry.update(state="terminating" if reason else "running", sample_monotonic=now,
                             rss_bytes=sample["rss_bytes"], phase=phase)
                decision = update_registry(registry, key, entry, limits, sample)
                writer.writerow(dict(utc=utc(), elapsed_seconds=now - begun, phase=phase,
                                     **{key:value for key,value in sample.items() if key != "process_ids"}))
                stream.flush()
                record["last_sample"] = sample
                remaining = [pid for pid in sample["process_ids"] if pid not in (os.getpid(), group)]
                if child_exited and not remaining:
                    break
            except (OSError, psutil.Error, RuntimeError) as exc:
                decision = dict(victim=key, reason="resource_monitor_unavailable")
                record["monitor_error"] = f"{type(exc).__name__}: {exc}"
            if not reason:
                if received:
                    reason = "dispatcher_interrupted"
                elif child_exited:
                    reason = "orphaned_descendant_cleanup"
                elif now - begun >= max(0, wall_limit - limits["terminate_grace_seconds"]):
                    reason = "external_wall_limit"
                elif decision and decision["victim"] == key:
                    reason = decision["reason"]
                if reason:
                    stopped = now
                    record.update(status="terminating", resource_limit_kind=reason, stage=phase,
                                  termination_reason=f"{phase}_memory_limit" if "rss_limit" in reason or "memory_floor" in reason else reason,
                                  stop_elapsed_seconds=now - begun, last_progress=progress,
                                  term_sent_utc=utc())
                    write(output / "guard_status.json", record)
                    # Our own handler and process_worker's handler survive TERM;
                    # the solver and all descendants receive the same signal.
                    os.killpg(group, signal.SIGTERM)
            elif now >= min(stopped + limits["terminate_grace_seconds"], begun + wall_limit):
                record.update(status="kill_escalated", kill_sent_utc=utc(),
                              external_wall_seconds=now - begun)
                write(output / "guard_status.json", record)
                os.killpg(group, signal.SIGKILL)
            record.update(elapsed_seconds=now - begun, stage=phase)
            write(output / "guard_status.json", record)
            time.sleep(limits["sample_seconds"])
    record.update(status="finished", returncode=code, finished_utc=utc(),
                  external_wall_seconds=time.monotonic() - begun, received_signals=received)
    if usage:
        record["solver_usage"] = dict(cpu_user_seconds=usage.ru_utime, cpu_system_seconds=usage.ru_stime,
                                     peak_rss_bytes=usage.ru_maxrss * (1 if sys.platform == "darwin" else 1024))
    write(output / "guard_status.json", record)
    with registry_lock(registry):
        entry.update(state="finished", sample_monotonic=time.monotonic(), rss_bytes=0)
        write(registry / (key + ".json"), entry)
    for sig, handler in handlers.items():
        signal.signal(sig, handler)
    return code if code >= 0 else 128 - code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--key", required=True)
    parser.add_argument("--limits", type=Path, required=True)
    parser.add_argument("--wall-limit", type=float, required=True)
    parser.add_argument("--stage-directory", type=Path)
    parser.add_argument("--initial-phase", default="construction")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("A child command is required")
    try:
        code = supervise(command, args.output, args.registry, args.key, read(args.limits),
                         args.wall_limit, stage_directory=args.stage_directory,
                         initial_phase=args.initial_phase)
    except Exception as exc:
        try:
            record = read(args.output / "guard_status.json")
            record.update(status="guardian_failure", termination_reason="resource_monitor_unavailable",
                          monitor_error=f"{type(exc).__name__}: {exc}", finished_utc=utc())
            write(args.output / "guard_status.json", record)
        finally:
            if os.getpgrp() == os.getppid():
                os.killpg(os.getpgrp(), signal.SIGKILL)
        raise
    raise SystemExit(code)


if __name__ == "__main__":
    main()
