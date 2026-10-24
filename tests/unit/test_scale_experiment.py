"""Exercise the real dispatcher with fake jobs; no optimizer is ever invoked."""

import json
import sys
import threading
import time

import pytest

from src.experiments import scale_ladder as batch
from src.experiments.storage import write_json


@pytest.fixture
def protocol():
    return batch.settings(batch.DEFAULT_CONFIG)


def test_sixty_commands_share_protocol_and_are_exhaustive(tmp_path, protocol):
    manifest = {"protocol": protocol, "python": sys.executable}
    jobs = batch.make_jobs(protocol)
    assert len(jobs) == len({v["id"] for v in jobs}) == 60
    for job in jobs:
        command = batch.command_for(tmp_path, manifest, job, tmp_path / "new-output")
        assert command[command.index("--seconds") + 1] == "18000"
        assert command[command.index("--threads") + 1] == "1"
        assert "--start" not in command and "--fixed-design" not in command
        if job["method"] == "full_gurobi":
            assert command[command.index("--max-model-variables") + 1] == "0"
        else:
            assert command[command.index("--path-column-limit") + 1] == "0"
        if job["method"] == "benders_cplex":
            assert command[command.index("--root-cut-rounds") + 1] == "3"
            assert "--disable" not in command


def _fake_batch(tmp_path, protocol, monkeypatch, count=9):
    jobs = batch.make_jobs(protocol)[:count]
    manifest = {"protocol": protocol, "python": sys.executable, "jobs": jobs}
    write_json(tmp_path / "experiment_manifest.json", manifest)
    write_json(
        tmp_path / "state.json",
        {
            "status": "prepared_not_started",
            "updated_utc": batch.utc(),
            "jobs": {j["id"]: {"status": "pending", "attempts": []} for j in jobs},
        },
    )
    monkeypatch.setattr(batch, "preflight", lambda *_: {"passed": True})
    return jobs


def test_dispatch_max_three_resume_and_separate_retry(tmp_path, protocol, monkeypatch):
    jobs = _fake_batch(tmp_path, protocol, monkeypatch)
    lock = threading.Lock()
    active = peak = called = 0

    def fake(job, attempt, cancel):
        nonlocal active, peak, called
        with lock:
            active += 1
            peak = max(active, peak)
            called += 1
        time.sleep(0.025)
        (attempt / "evidence.txt").write_text("unchanged")
        with lock:
            active -= 1
        return {"passed": job["id"] != jobs[0]["id"] or attempt.name == "002"}

    batch.dispatch(tmp_path, worker=fake)
    assert called == 9 and peak == 3
    state = batch.read(tmp_path / "state.json")
    assert state["jobs"][jobs[0]["id"]]["status"] == "failed"
    batch.dispatch(tmp_path, worker=fake)
    assert called == 9  # pending-only resume does not rerun failed or completed jobs
    batch.dispatch(tmp_path, worker=fake, retry_failed=True)
    assert called == 10
    state = batch.read(tmp_path / "state.json")
    first = state["jobs"][jobs[0]["id"]]
    assert first["status"] == "completed" and len(first["attempts"]) == 2
    assert (
        tmp_path / first["attempts"][0]["directory"] / "evidence.txt"
    ).read_text() == "unchanged"


def test_lock_blocks_second_dispatcher(tmp_path):
    with batch.batch_lock(tmp_path):
        with pytest.raises(RuntimeError, match="already owns"):
            with batch.batch_lock(tmp_path):
                pass


def test_status_and_default_cli_cannot_launch_processes(
    tmp_path, protocol, monkeypatch, capsys
):
    _fake_batch(tmp_path, protocol, monkeypatch, count=3)
    monkeypatch.setattr(
        batch.subprocess,
        "Popen",
        lambda *a, **k: pytest.fail("Unexpected process launch"),
    )
    batch.main(["--batch", str(tmp_path)])
    assert json.loads(capsys.readouterr().out)["jobs"] == {"pending": 3}


def test_execute_captures_failure_and_environment_without_solver(tmp_path, protocol):
    cfg = dict(
        protocol,
        resource_sample_seconds=0.01,
        seconds_per_job=2,
        watchdog_grace_seconds=1,
    )
    code = "import os,sys,time; print(os.environ['OMP_NUM_THREADS'],flush=True); print('synthetic failure',file=sys.stderr); time.sleep(.05); sys.exit(13)"
    result = batch.execute(
        [sys.executable, "-c", code], tmp_path, tmp_path, cfg, threading.Event()
    )
    assert result["returncode"] == 13
    assert (tmp_path / "stdout.log").read_text().strip() == "1"
    assert "synthetic failure" in (tmp_path / "stderr.log").read_text()
    assert (tmp_path / "resources.csv").read_text().startswith("timestamp_utc,")
    assert (
        result["external_wall_seconds"] > 0 and (tmp_path / "execution.json").is_file()
    )


def test_watchdog_stops_fake_process_group(tmp_path, protocol):
    cfg = dict(
        protocol,
        resource_sample_seconds=0.01,
        seconds_per_job=0.05,
        watchdog_grace_seconds=0.01,
    )
    result = batch.execute(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        tmp_path,
        tmp_path,
        cfg,
        threading.Event(),
    )
    assert result["stop_reason"] == "watchdog_deadline_exceeded"
    assert result["external_wall_seconds"] < 5


def test_artifact_audit_rejects_missing_outputs(tmp_path, protocol):
    job = batch.make_jobs(protocol)[0]
    attempt = tmp_path / "attempt"
    attempt.mkdir()
    write_json(attempt / "execution.json", {"status": "exited", "returncode": 0})
    audit = batch.audit_attempt(tmp_path, {"protocol": protocol}, job, attempt)
    assert not audit["passed"] and audit["errors"]
    assert (attempt / "artifact_audit.json").is_file()


def test_referenced_checkpoint_integrity_and_repeat_recovery(tmp_path):
    from src.experiments.native import save_json_result

    source = tmp_path / "recovery" / "all_courier"
    payload = {
        "objective": 4,
        "bound": 0,
        "max_orders": 1,
        "_solution": {"fleet": 0},
        "_segments": [],
        "_missions": [],
        "_relocations": [],
    }
    save_json_result(source, payload)
    write_json(source / "input_reference.json", {"max_orders": 1, "manifest": {}})
    files = {
        name: {
            "path": str((source / name).relative_to(tmp_path)),
            "sha256": batch.digest(source / name),
        }
        for name in (
            "result.json",
            "solution.json",
            "path_flows.json",
            "input_reference.json",
        )
    }
    write_json(tmp_path / "checkpoint.json", {"validated": True, "files": files})
    recovered = batch._recover_checkpoint(tmp_path)
    assert batch.read(recovered / "result.json")["objective"] == 4
    assert batch._recover_checkpoint(tmp_path) == recovered
    (source / "solution.json").write_text('{"fleet": 1}')
    with pytest.raises(ValueError, match="integrity"):
        batch._recover_checkpoint(tmp_path)


def test_termination_waits_for_child_finalization(tmp_path, protocol):
    cfg = dict(
        protocol,
        resource_sample_seconds=0.01,
        seconds_per_job=0.3,
        watchdog_grace_seconds=0,
    )
    code = """import signal,time,sys
def finish(*args):
    time.sleep(.1)
    print('final child artifact',flush=True)
    sys.exit(0)
signal.signal(signal.SIGTERM,finish)
print('ready',flush=True)
time.sleep(30)
"""
    result = batch.execute(
        [sys.executable, "-c", code], tmp_path, tmp_path, cfg, threading.Event()
    )
    assert result["stop_reason"] == "watchdog_deadline_exceeded"
    assert "final child artifact" in (tmp_path / "stdout.log").read_text()
    usage = batch.read(tmp_path / "resource_usage.json")
    assert usage["returncode"] == 0 and usage["peak_rss_bytes"] > 0


def test_preflight_rejects_changed_protocol_before_loading_inputs(tmp_path, protocol):
    write_json(tmp_path / "frozen/protocol.json", protocol)
    write_json(
        tmp_path / "experiment_manifest.json",
        {"protocol": dict(protocol, parallel_jobs=4)},
    )
    with pytest.raises(ValueError, match="differs from the frozen protocol"):
        batch.preflight(tmp_path)


def test_preflight_rejects_changed_job_command(tmp_path, protocol):
    manifest = {
        "protocol": protocol,
        "python": sys.executable,
        "jobs": batch.make_jobs(protocol),
    }
    for job in manifest["jobs"]:
        job["initial_command"] = batch.command_for(
            tmp_path, manifest, job, tmp_path / job["output"] / "attempts/001"
        )
    manifest["jobs"][0]["initial_command"].extend(["--start", "unapproved.json"])
    write_json(tmp_path / "frozen/protocol.json", protocol)
    write_json(tmp_path / "experiment_manifest.json", manifest)
    with pytest.raises(ValueError, match="job list/commands"):
        batch.preflight(tmp_path)


def test_restart_audits_finished_output_without_rerunning(
    tmp_path, protocol, monkeypatch
):
    jobs = _fake_batch(tmp_path, protocol, monkeypatch, count=1)
    relative = jobs[0]["output"] + "/attempts/001"
    attempt = tmp_path / relative
    attempt.mkdir(parents=True)
    write_json(attempt / "execution.json", {"status": "exited", "returncode": 0})
    state = batch.read(tmp_path / "state.json")
    state["jobs"][jobs[0]["id"]] = {
        "status": "running",
        "directory": relative,
        "attempts": [{"directory": relative, "number": 1, "status": "running"}],
    }
    write_json(tmp_path / "state.json", state)
    monkeypatch.setattr(
        batch,
        "audit_attempt",
        lambda *_: {"passed": True, "result_directory": relative},
    )
    result = batch.dispatch(
        tmp_path, worker=lambda *_: pytest.fail("Completed job must not rerun")
    )
    assert result["jobs"] == {"completed": 1}
