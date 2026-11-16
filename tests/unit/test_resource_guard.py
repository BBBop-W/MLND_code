"""Resource enforcement with small disposable processes, never solver instances."""

import json
import sys
import threading

import psutil
import pytest

from src.experiments import conditional_benchmark_20261022 as benchmark
from src.experiments import resource_guard as guard
from src.experiments import scale_ladder


def limits(**change):
    result = dict(job_rss_bytes=100*1024**2, batch_rss_bytes=1024**4,
                system_available_floor_bytes=1, sample_seconds=.05,
                terminate_grace_seconds=.3, pressure_cooldown_seconds=1)
    result.update(change)
    return result


def execute(tmp_path, code, *, phase="construction", wall=5, cap=None):
    batch = tmp_path / "batch"
    batch.mkdir()
    (batch / "frozen").symlink_to(benchmark.ROOT, target_is_directory=True)
    attempt = batch / "attempt"
    attempt.mkdir()
    cfg = dict(resource_limits=cap or limits(), hard_wall_seconds=wall,
               seconds_per_job=wall, resource_sample_seconds=.05, watchdog_grace_seconds=1)
    manifest = dict(python=sys.executable, protocol=cfg)
    result = benchmark.guarded_execute(batch, manifest, "probe",
        [sys.executable, "-c", code], attempt, threading.Event(), initial_phase=phase)
    return result, attempt


def process_gone(pid):
    try:
        return not psutil.Process(pid).is_running() or psutil.Process(pid).status() == psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return True


@pytest.mark.parametrize("phase", ["construction", "solve", "validation"])
def test_real_rss_limit_keeps_checkpoint_and_classifies_phase(tmp_path, phase):
    checkpoint = tmp_path / "checkpoint.json"
    code = (f"from pathlib import Path; import time; Path({str(checkpoint)!r}).write_text('{{\"saved\":true}}'); "
            "a=bytearray(160*1024*1024); a[::4096]=b'x'*len(a[::4096]); time.sleep(20)")
    result, attempt = execute(tmp_path, code, phase=phase)
    assert result["stop_reason"] == phase + "_memory_limit"
    assert result["resource_limit_kind"] == "per_job_rss_limit"
    assert result["sampled_peak_rss_bytes"] > 100*1024**2
    assert json.loads(checkpoint.read_text())["saved"]
    assert (attempt / "stdout.log").is_file()
    assert result["external_wall_seconds"] < 5


def test_wall_limit_kills_term_ignoring_descendant_after_parent_exits(tmp_path):
    pid_file = tmp_path / "descendant.pid"
    descendant = "import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(30)"
    code = ("import subprocess,sys,time; from pathlib import Path; "
            f"p=subprocess.Popen([sys.executable,'-c',{descendant!r}]); Path({str(pid_file)!r}).write_text(str(p.pid)); "
            "time.sleep(30)")
    result, attempt = execute(tmp_path, code, wall=.5)
    saved = json.loads((attempt / "guard_status.json").read_text())
    assert saved["status"] == "kill_escalated"
    assert result["stop_reason"] == "external_wall_limit"
    assert process_gone(int(pid_file.read_text()))


def test_normal_parent_exit_does_not_leave_unguarded_descendant(tmp_path):
    pid_file = tmp_path / "descendant.pid"
    descendant = "import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(30)"
    code = ("import subprocess,sys,time; from pathlib import Path; "
            f"p=subprocess.Popen([sys.executable,'-c',{descendant!r}]); Path({str(pid_file)!r}).write_text(str(p.pid)); time.sleep(.2)")
    result, attempt = execute(tmp_path, code)
    assert result["stop_reason"] == "orphaned_descendant_cleanup"
    assert json.loads((attempt / "guard_status.json").read_text())["status"] == "kill_escalated"
    assert process_gone(int(pid_file.read_text()))


def test_memory_stop_cleans_term_ignoring_descendant(tmp_path):
    pid_file = tmp_path / "descendant.pid"
    descendant = "import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(30)"
    code = ("import subprocess,sys,time; from pathlib import Path; "
            f"p=subprocess.Popen([sys.executable,'-c',{descendant!r}]); Path({str(pid_file)!r}).write_text(str(p.pid)); "
            "time.sleep(.2); a=bytearray(160*1024*1024); a[::4096]=b'x'*len(a[::4096]); time.sleep(30)")
    result, attempt = execute(tmp_path, code)
    assert result["stop_reason"] == "construction_memory_limit"
    assert json.loads((attempt / "guard_status.json").read_text())["status"] == "kill_escalated"
    assert process_gone(int(pid_file.read_text()))


def test_global_pressure_lease_selects_only_largest_registered_job():
    cfg = limits()
    cfg.update(job_rss_bytes=1000, batch_rss_bytes=250, system_available_floor_bytes=50)
    entries = {"a": dict(state="running", rss_bytes=100, sample_monotonic=1),
               "b": dict(state="running", rss_bytes=200, sample_monotonic=1)}
    first = guard.resource_decision(entries, "a", cfg, 100, {}, 1)
    assert first["victim"] == "b" and first["reason"] == "batch_rss_limit"
    assert guard.resource_decision(entries, "b", cfg, 100, first, 1.1) == first
    cfg["batch_rss_bytes"] = 1000
    system = guard.resource_decision(entries, "a", cfg, 10, {}, 1)
    assert system["victim"] == "b" and system["reason"] == "system_available_memory_floor"
    with pytest.raises(RuntimeError, match="stopped reporting"):
        guard.resource_decision(entries, "a", cfg, 100, {}, 20)


def test_unavailable_rss_stops_closed_and_never_reports_zero_peak(tmp_path):
    output = tmp_path / "probe"
    output.mkdir()
    code = ("import sys; from pathlib import Path; from src.experiments import resource_guard as g; "
            "g.tree_sample=lambda *_: (_ for _ in ()).throw(PermissionError('denied')); "
            f"sys.exit(g.supervise([sys.executable,'-c','import time;time.sleep(20)'],Path({str(output)!r}),"
            f"Path({str(tmp_path/'registry')!r}),'failure',{limits()!r},3))")
    cfg = dict(seconds_per_job=4, watchdog_grace_seconds=1, resource_sample_seconds=.05)
    scale_ladder.execute([sys.executable, "-c", code], benchmark.ROOT, output, cfg, threading.Event())
    result = json.loads((output / "guard_status.json").read_text())
    assert result["termination_reason"] == "resource_monitor_unavailable"
    assert result["peak_sampled_rss_bytes"] is None


def test_report_retains_kill_reason_without_final_result(tmp_path):
    directory = tmp_path / "run"
    directory.mkdir()
    (directory / "execution.json").write_text(json.dumps(dict(returncode=137, stop_reason="construction_memory_limit")))
    (directory / "guard_status.json").write_text(json.dumps(dict(termination_reason="construction_memory_limit",
        resource_limit_kind="per_job_rss_limit", stage="construction", peak_sampled_rss_bytes=123)))
    row = benchmark._record(tmp_path, dict(id="a", method="full_cplex", instance="instances/L01"),
                            dict(status="failed", directory="run"))
    assert row["resource_unaffordable"] and row["termination_reason"] == "construction_memory_limit"
    assert row["peak_sampled_tree_rss_bytes"] == 123 and row["upper_bound"] is None
    (directory / "result.json").write_text('{"broken":')
    assert benchmark._record(tmp_path, dict(id="a", method="full_cplex", instance="instances/L01"),
                             dict(status="failed", directory="run"))["result_error"] == "missing_or_invalid_result_json"
