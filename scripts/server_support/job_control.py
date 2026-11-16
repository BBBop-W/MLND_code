"""Unprivileged Linux resource guard; never claim a hard aggregate memory cap.

Each payload process inherits RLIMIT_AS. An independent guardian samples RSS
and owns the supported process group through its entire lifetime.
"""
from __future__ import annotations
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid
from linux_process import require_linux, same_process

GIB = 1024 ** 3
GUARDIAN = Path(__file__).with_name("linux_guardian.py")


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


class ControlledProcess:
    """Own one independent guardian; communicate only through its private pipe."""
    def __init__(self, command, cwd, directory, memory_bytes, wall_seconds, env=None, *, sample_seconds=1):
        require_linux()
        self.directory = Path(directory)
        self.memory_bytes, self.wall_seconds = int(memory_bytes), float(wall_seconds)
        self.stop_reason = self.returncode = self.monitor_error = None
        self.last_sample = {"rss_bytes": 0, "pids": [], "process_count": 0}
        self.latest = {}
        self.control_fd = None
        self.process = None
        self.stdout = (self.directory / "guard.stdout.log").open("wb")
        self.stderr = (self.directory / "guard.stderr.log").open("wb")
        spec = self.directory / "launch_command.json"
        atomic_json(spec, {"command": command, "cwd": str(cwd), "token": uuid.uuid4().hex,
                    "memory_bytes": self.memory_bytes, "wall_seconds": self.wall_seconds, "sample_seconds": sample_seconds})
        read_fd, self.control_fd = os.pipe()
        try:
            self.process = subprocess.Popen([sys.executable, str(GUARDIAN), str(spec), str(read_fd)],
                pass_fds=(read_fd,), start_new_session=True, stdout=self.stdout, stderr=self.stderr, env=env)
        except BaseException:
            os.close(self.control_fd)
            self.control_fd = None
            self.stdout.close()
            self.stderr.close()
            raise
        finally:
            os.close(read_fd)
        try:
            deadline = time.monotonic() + 20
            while time.monotonic() < deadline:
                self._read()
                if self.latest.get("rlimit_readback"):
                    return
                if self.process.poll() is not None:
                    raise RuntimeError("Guardian/gate could not establish RLIMIT_AS; see guard.json and guard.stderr.log")
                time.sleep(.02)
            raise RuntimeError("Guardian setup did not finish within 20 seconds")
        except BaseException:
            self.close()
            raise

    def _read(self):
        try:
            self.latest = json.loads((self.directory / "guard.json").read_text(encoding="utf-8"))
        except FileNotFoundError:
            return
        self.last_sample = self.latest.get("last_sample", self.last_sample)
        self.stop_reason = self.latest.get("stop_reason") or self.stop_reason
        self.monitor_error = self.latest.get("monitor_error")
        self.returncode = self.latest.get("returncode")

    def stop(self, reason):
        self.stop_reason = self.stop_reason or reason
        if self.control_fd is not None:
            try:
                os.write(self.control_fd, (json.dumps({"stop": self.stop_reason}) + "\n").encode())
            except BrokenPipeError:
                pass

    def tick(self):
        self._read()
        guardian_returncode = self.process.poll()
        if guardian_returncode is None:
            return False
        self._read()
        if self.latest.get("status") != "finished" or not self.latest.get("group_cleanup_complete"):
            self.stop_reason = self.stop_reason or "guardian_or_group_cleanup_failed"
            self.returncode = self.returncode if self.returncode is not None else guardian_returncode or 125
        return True

    def record(self):
        return dict(self.latest, stop_reason=self.stop_reason, returncode=self.returncode,
                    monitor_error=self.monitor_error, last_sample=self.last_sample)

    def close(self):
        # EOF requests guardian cleanup, even on dispatcher interruption. Never
        # kill the guardian itself while it protects a live payload subtree.
        if self.control_fd is not None:
            os.close(self.control_fd)
            self.control_fd = None
        try:
            if self.process is not None:
                self.process.wait(timeout=25)
                self._read()
        finally:
            self.stdout.close()
            self.stderr.close()


def ensure_no_live_attempts(output):
    """Resume may not reserve over an old guardian that is still cleaning up."""
    for path in (Path(output) / "jobs").glob("*/attempts/*/attempt.json"):
        own = json.loads(path.read_text(encoding="utf-8"))
        if own.get("status") in ("running", "interrupted") and not (path.parent / "guard.json").exists():
            raise RuntimeError(f"Previous launch/cleanup is unconfirmed: attempt has no guardian record; inspect before resume: {path}")
    for path in (Path(output) / "jobs").glob("*/attempts/*/guard.json"):
        record = json.loads(path.read_text(encoding="utf-8"))
        for prefix in ("guardian", "bootstrap"):
            pid, started = record.get(prefix + "_pid"), record.get(prefix + "_start_ticks")
            if pid and started and same_process(pid, started):
                raise RuntimeError(f"Owned process from a previous attempt is still alive; wait for cleanup before resume: {path}")
        if record.get("status") != "finished" or record.get("group_cleanup_complete") is not True:
            raise RuntimeError(f"Prior attempt reported incomplete process-group cleanup; inspect it before using this output: {path}")


def reservation_fits(job, active, available_bytes, resources):
    if len(active) >= resources["max_parallel"]:
        return False
    if job["exclusive"] and active or any(x["job"]["exclusive"] for x in active):
        return False
    reserved = sum(x["job"]["memory_bytes"] for x in active)
    if reserved + job["memory_bytes"] > resources["reservation_budget_bytes"]:
        return False
    if sum(x["job"]["threads"] for x in active) + job["threads"] > resources["max_threads"]:
        return False
    unfilled = sum(max(0, x["job"]["memory_bytes"] - x["sample"].get("rss_bytes", 0)) for x in active)
    return available_bytes >= resources["available_floor_bytes"] + job["memory_bytes"] + unfilled


def pressure_victim(active, available_bytes, resources, consecutive_low, cooldown_until, now):
    low = consecutive_low + 1 if available_bytes < resources["available_floor_bytes"] else 0
    eligible = [x for x in active if not x.get("stopping")]
    if low < 3 or now < cooldown_until or not eligible:
        return None, low
    victim = max(eligible, key=lambda x: (x["sample"].get("rss_bytes", 0), x["job"]["id"]))
    return victim["job"]["id"], 0
