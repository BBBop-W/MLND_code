"""Independent owner of one Linux job group; retained leader prevents PGID reuse.

RLIMIT_AS is hard per process. Group RSS is sampled and may overshoot. A process
deliberately changing session/group is outside the supported execution model.
"""
from __future__ import annotations
import json
import os
from pathlib import Path
import select
import signal
import subprocess
import sys
import time
from job_control import atomic_json
from linux_process import (become_subreaper, descendants, pidfd_open,
                          pidfd_send_signal, process_info, reap_adopted)


def run(spec_path, control_fd):
    spec = json.loads(Path(spec_path).read_text(encoding="utf-8"))
    directory = Path(spec_path).parent
    become_subreaper()
    own = process_info(os.getpid())
    start = time.monotonic()
    leader = leader_pidfd = stop_reason = monitor_error = exit_status = leader_exited_at = None
    peak_rss = peak_virtual = 0
    sample = {"rss_bytes": 0, "max_process_virtual_bytes": 0, "pids": [], "process_count": 0, "escaped_pids": []}
    ready, control_buffer, snapshots = None, b"", {}

    def record(status):
        value = {"status": status, "guardian_pid": os.getpid(), "guardian_start_ticks": own["start_ticks"],
                 "bootstrap_pid": None if leader is None else leader.pid,
                 "bootstrap_start_ticks": snapshots.get("leader_start_ticks"),
                 "process_group_id": None if leader is None else leader.pid,
                 "address_space_hard_limit_bytes_per_process": spec["memory_bytes"],
                 "sampled_tree_rss_limit_bytes": spec["memory_bytes"], "rss_limit_is_hard": False,
                 "memory_limit_metric": "RLIMIT_AS hard per process; whole supported job group RSS sampled only",
                 "sample_interval_seconds": spec["sample_seconds"],
                 "wall_limit_seconds": spec["wall_seconds"], "elapsed_seconds": time.monotonic()-start,
                 "stop_reason": stop_reason, "monitor_error": monitor_error, "returncode": exit_status,
                 "sampled_rss_peak_bytes": peak_rss, "sampled_max_process_virtual_peak_bytes": peak_virtual,
                 "last_sample": sample, "rlimit_readback": ready,
                 "group_cleanup_complete": snapshots.get("group_cleanup_complete", False),
                 "scope_limitation": "Inherited process group/session only; setsid/daemon escape unsupported; swap allowed."}
        atomic_json(directory / "guard.json", value)
        return value

    def read_control():
        nonlocal control_buffer, stop_reason
        readable, _, _ = select.select([control_fd], [], [], 0)
        if not readable:
            return
        data = os.read(control_fd, 65536)
        if not data:
            stop_reason = stop_reason or "dispatcher_pipe_closed"
            return
        control_buffer += data
        while b"\n" in control_buffer:
            line, control_buffer = control_buffer.split(b"\n", 1)
            message = json.loads(line)
            if isinstance(message.get("stop"), str):
                stop_reason = stop_reason or message["stop"]

    def refresh():
        nonlocal sample, peak_rss, peak_virtual
        members, escaped = descendants(os.getpid(), leader.pid)
        sample = {"rss_bytes": sum(member["rss_bytes"] for member in members),
                  "max_process_virtual_bytes": max([member["virtual_bytes"] for member in members] or [0]),
                  "pids": [member["pid"] for member in members], "process_count": len(members),
                  "escaped_pids": [member["pid"] for member in escaped]}
        peak_rss = max(peak_rss, sample["rss_bytes"])
        peak_virtual = max(peak_virtual, sample["max_process_virtual_bytes"])

    def terminate_owned_group():
        # Never poll/wait/reap the leader before the last killpg: its retained
        # PID prevents that numeric group identity being reused for another job.
        identity = process_info(leader.pid)
        if (identity["start_ticks"] != snapshots["leader_start_ticks"] or identity["pgid"] != leader.pid
                or identity["sid"] != leader.pid):
            raise RuntimeError("Owned session leader identity mismatch; refusing killpg")
        try:
            if leader_pidfd is not None:
                pidfd_send_signal(leader_pidfd, signal.SIGKILL)
        except ProcessLookupError:
            pass
        deadline = time.monotonic() + 10
        while True:
            try:
                os.killpg(leader.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            refresh()
            if not sample["pids"]:
                snapshots["group_cleanup_complete"] = not sample["escaped_pids"]
                return
            if time.monotonic() >= deadline:
                raise RuntimeError("Owned group still live after SIGKILL; possible uninterruptible kernel sleep")
            time.sleep(.05)

    try:
        leader = subprocess.Popen([sys.executable, str(Path(__file__).with_name("gate_worker.py")), str(spec_path)],
                                  start_new_session=True, close_fds=True)
        snapshots["leader_start_ticks"] = process_info(leader.pid)["start_ticks"]
        leader_pidfd = pidfd_open(leader.pid)
        record("initializing")
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            read_control()
            if stop_reason:
                break
            failed = directory / "gate_error.json"
            if failed.exists():
                monitor_error = failed.read_text(encoding="utf-8")
                stop_reason = "rlimit_setup_failed"
                break
            if (directory / "gate_ready.json").exists():
                ready = json.loads((directory / "gate_ready.json").read_text(encoding="utf-8"))
                if (ready.get("pid") != leader.pid or ready.get("pgid") != leader.pid or ready.get("sid") != leader.pid
                        or ready.get("rlimit_as_soft") != spec["memory_bytes"] or ready.get("rlimit_as_hard") != spec["memory_bytes"]):
                    raise RuntimeError("Gate limit/session readback mismatch")
                atomic_json(directory / "launch_gate.json", {"token": spec["token"]})
                break
            if os.waitid(os.P_PID, leader.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT):
                stop_reason = "gate_failed_before_launch"
                break
            time.sleep(.02)
        if ready is None and stop_reason is None:
            stop_reason = "gate_setup_timeout"
        record("running")
        while not stop_reason:
            read_control()
            outcome = os.waitid(os.P_PID, leader.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
            if outcome is not None and leader_exited_at is None:
                leader_exited_at = time.monotonic()
                exit_status = outcome.si_status if outcome.si_code == os.CLD_EXITED else -outcome.si_status
            refresh()
            if sample["escaped_pids"]:
                stop_reason = "unsupported_process_group_escape"
            elif sample["rss_bytes"] > spec["memory_bytes"]:
                stop_reason = "sampled_tree_rss_limit"
            elif time.monotonic() - start >= spec["wall_seconds"]:
                stop_reason = "external_job_wall_limit"
            elif outcome is not None:
                if not sample["pids"]:
                    snapshots["group_cleanup_complete"] = True
                    break
                if time.monotonic() - leader_exited_at >= 2:
                    stop_reason = "orphaned_descendant_cleanup"
            record("running")
            if not stop_reason:
                select.select([control_fd], [], [], spec["sample_seconds"])
    except BaseException as error:
        monitor_error = f"{type(error).__name__}: {error}"
        stop_reason = stop_reason or "resource_guard_unavailable"
    finally:
        if leader is not None:
            try:
                # Always signal the anchored group before reaping, including a
                # normal exit whose final descendant sample appeared empty.
                snapshots["group_cleanup_complete"] = False
                terminate_owned_group()
                actual = leader.wait(timeout=10)
                exit_status = actual if exit_status is None else exit_status
                reap_adopted()
            except BaseException as error:
                snapshots["group_cleanup_complete"] = False
                monitor_error = (monitor_error or "") + f"; cleanup: {type(error).__name__}: {error}"
                stop_reason = stop_reason or "owned_group_cleanup_incomplete"
            if leader_pidfd is not None:
                os.close(leader_pidfd)
        os.close(control_fd)
        record("finished")
    return 0 if snapshots.get("group_cleanup_complete") else 2


if __name__ == "__main__":
    raise SystemExit(run(Path(sys.argv[1]).resolve(), int(sys.argv[2])))
