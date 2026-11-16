"""Ordinary-user target test: hard per-process AS, sampled group RSS and cleanup.

No solver, cgroup modification, sudo, global process scan or swap reconfiguration.
Run on the actual Linux host; portable mocked tests cannot certify this behavior.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import time
import psutil
from job_control import ControlledProcess, atomic_json
from linux_process import (become_subreaper, pidfd_open, pidfd_send_signal,
                          require_linux, same_process)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def finish(process, timeout=20):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.tick():
            return process.record()
        time.sleep(.05)
    raise RuntimeError("Self-test job did not finish within its external deadline")


def run(output):
    process_capabilities = require_linux()
    become_subreaper()
    if psutil.virtual_memory().available < 512*1024**2:
        raise RuntimeError("Self-test requires at least 512 MiB available memory")
    output.mkdir(parents=True, exist_ok=False)
    report = {"schema": "mlnd-linux-unprivileged-selftest/1", "passed": False,
              "started_utc": datetime.now(timezone.utc).isoformat(), "platform": platform.platform(),
              "python": sys.version, "linux_process_capabilities": process_capabilities,
              "checks": {}, "probes": {}, "errors": [],
              "limitation": "RLIMIT_AS is hard per process. Group RSS is sampled and may overshoot. No hard tree-memory or no-swap claim; setsid/daemon escape unsupported."}
    processes = []
    dispatcher = None
    sentinel = subprocess.Popen([sys.executable, "-c", "import time;time.sleep(120)"], start_new_session=True)
    try:
        as_probe = output / "as_probe.py"
        as_probe.write_text('''import json,resource,subprocess,sys
from pathlib import Path
soft,hard=resource.getrlimit(resource.RLIMIT_AS)
denied=False
try:
    value=bytearray(512*1024*1024)
except MemoryError:
    denied=True
child=json.loads(subprocess.check_output([sys.executable,"-c","import json,resource;print(json.dumps(resource.getrlimit(resource.RLIMIT_AS)))"],text=True))
Path(sys.argv[1]).write_text(json.dumps({"limit":[soft,hard],"child_limit":child,"allocation_denied":denied}))
''', encoding="utf-8")
        directory = output / "address_space"
        directory.mkdir()
        cap = 128*1024**2
        process = ControlledProcess([sys.executable, str(as_probe), str(directory / "probe.json")], output,
                                    directory, cap, 15, sample_seconds=.05)
        processes.append(process)
        record = finish(process)
        probe = read(directory / "probe.json")
        report["checks"]["address_space_allocation_denied"] = probe["allocation_denied"] is True
        report["checks"]["hard_and_soft_limits_inherited"] = probe["limit"] == [cap, cap] and probe["child_limit"] == [cap, cap]
        report["checks"]["normal_completion_cleanup"] = record["returncode"] == 0 and record["stop_reason"] is None and record["group_cleanup_complete"]
        report["probes"]["address_space"] = record
        process.close()

        rss_probe = output / "rss_probe.py"
        rss_probe.write_text('''import subprocess,sys,time
if len(sys.argv)>1:
    allocation=bytearray(72*1024*1024)
    time.sleep(60)
else:
    subprocess.Popen([sys.executable,__file__,"child"])
    subprocess.Popen([sys.executable,__file__,"child"])
    time.sleep(60)
''', encoding="utf-8")
        directory = output / "sampled_rss"
        directory.mkdir()
        process = ControlledProcess([sys.executable, str(rss_probe)], output, directory, cap, 15, sample_seconds=.05)
        processes.append(process)
        record = finish(process)
        report["checks"]["aggregate_rss_polling_terminates_group"] = record["stop_reason"] == "sampled_tree_rss_limit" and record["sampled_rss_peak_bytes"] > cap and record["group_cleanup_complete"]
        report["checks"]["individual_address_spaces_stay_within_limit"] = record["sampled_max_process_virtual_peak_bytes"] <= cap
        report["checks"]["rss_is_reported_as_nonhard"] = record["rss_limit_is_hard"] is False
        report["probes"]["sampled_rss"] = record
        process.close()

        tree_probe = output / "tree_probe.py"
        tree_probe.write_text('''import json,os,subprocess,sys,time
from pathlib import Path
level=int(sys.argv[1])
with open(sys.argv[2],"a") as stream:
    ticks=int(Path("/proc",str(os.getpid()),"stat").read_text().rsplit(")",1)[1].split()[19])
    stream.write(json.dumps({"pid":os.getpid(),"start_ticks":ticks,"pgid":os.getpgrp(),"sid":os.getsid(0)})+"\\n")
if level:
    subprocess.Popen([sys.executable,__file__,str(level-1),sys.argv[2]])
time.sleep(60)
''', encoding="utf-8")
        directory = output / "wall_tree"
        directory.mkdir()
        process = ControlledProcess([sys.executable, str(tree_probe), "2", str(directory / "members.jsonl")], output,
                                    directory, 256*1024**2, 1.5, sample_seconds=.05)
        processes.append(process)
        record = finish(process)
        members = [json.loads(line) for line in (directory / "members.jsonl").read_text().splitlines()]
        report["checks"]["three_generations_inherit_same_group"] = len(members) == 3 and all(m["pgid"] == record["process_group_id"] == m["sid"] for m in members)
        report["checks"]["wall_limit_cleans_descendants"] = record["stop_reason"] == "external_job_wall_limit" and record["group_cleanup_complete"] and all(not same_process(m["pid"],m["start_ticks"]) for m in members)
        report["probes"]["wall_tree"] = record
        process.close()

        # Kill a dedicated fake dispatcher, not its independently running
        # guardian. EOF must trigger cleanup without this self-test doing it.
        directory = output / "parent_death"
        directory.mkdir()
        launcher = output / "fake_dispatcher.py"
        launcher.write_text('''import sys,time
from pathlib import Path
sys.path.insert(0,sys.argv[1])
from job_control import ControlledProcess
p=ControlledProcess([sys.executable,sys.argv[2],"2",sys.argv[3]],Path(sys.argv[2]).parent,Path(sys.argv[4]),256*1024**2,60,sample_seconds=.05)
Path(sys.argv[4],"dispatcher_ready").touch()
time.sleep(60)
''', encoding="utf-8")
        dispatcher = subprocess.Popen([sys.executable, str(launcher), str(Path(__file__).parent), str(tree_probe),
            str(directory / "members.jsonl"), str(directory)], start_new_session=True)
        deadline = time.monotonic() + 15
        while not (directory / "dispatcher_ready").exists():
            if dispatcher.poll() is not None or time.monotonic() >= deadline:
                raise RuntimeError("Fake dispatcher did not establish its guarded job")
            time.sleep(.05)
        # Allow the recorded descendants to exist before forcing the crash.
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if (directory / "members.jsonl").exists() and len((directory / "members.jsonl").read_text().splitlines()) >= 3:
                break
            time.sleep(.05)
        if not (directory / "members.jsonl").exists() or len((directory / "members.jsonl").read_text().splitlines()) != 3:
            raise RuntimeError("Parent-death probe did not establish all three descendant generations")
        descriptor = pidfd_open(dispatcher.pid)
        try:
            pidfd_send_signal(descriptor, signal.SIGKILL)
        finally:
            os.close(descriptor)
        dispatcher.wait(timeout=5)
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            record = read(directory / "guard.json")
            if record.get("status") == "finished":
                break
            time.sleep(.05)
        members = [json.loads(line) for line in (directory / "members.jsonl").read_text().splitlines()]
        report["checks"]["dispatcher_crash_eof_cleans_group"] = record.get("status") == "finished" and record.get("stop_reason") == "dispatcher_pipe_closed" and record.get("group_cleanup_complete") is True and len(members) == 3 and all(not same_process(m["pid"],m["start_ticks"]) for m in members)
        report["probes"]["parent_death"] = record
        report["checks"]["outside_sentinel_unaffected"] = sentinel.poll() is None
        report["passed"] = all(report["checks"].values())
    except BaseException as error:
        report["errors"].append(f"{type(error).__name__}: {error}")
    finally:
        if dispatcher is not None and dispatcher.poll() is None:
            dispatcher.kill()
            dispatcher.wait(timeout=5)
        for process in processes:
            try:
                process.close()
            except BaseException as error:
                report["errors"].append(f"Cleanup: {type(error).__name__}: {error}")
        if sentinel.poll() is None:
            sentinel.kill()
        sentinel.wait(timeout=5)
        report["passed"] = report["passed"] and not report["errors"]
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        atomic_json(output / "report.json", report)
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="New self-test output directory")
    args = parser.parse_args()
    try:
        return run(args.out.resolve())
    except (OSError, RuntimeError) as error:
        print(f"Self-test not run: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
