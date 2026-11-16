"""Portable scheduler policy tests; Linux primitive acceptance remains on-host."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "server_suite_under_test",
    Path(__file__).resolve().parents[2] / "scripts/server_suite.py",
)
suite = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(suite)


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def prepared(tmp_path):
    instance = tmp_path / "inputs/a"
    put(instance / "model.json", {"values": {"n_l": 2, "n_h": 2, "n_od": 1}})
    put(
        instance / "manifest.json",
        {"files": {"model.json": suite.digest(instance / "model.json")}},
    )
    document = {
        "schema": suite.SCHEMA,
        "defaults": {"seconds": 7200, "gap": 0.01, "seed": 43, "threads": 1},
        "resources": {"max_parallel": 8, "sample_seconds": 0.001},
        "jobs": [
            {
                "id": "p1",
                "stage": "preflight",
                "instance": "inputs/a",
                "k": 1,
                "memory_gib": 3,
            },
            {
                "id": "p2",
                "stage": "preflight",
                "instance": "inputs/a",
                "k": 2,
                "memory_gib": 6,
            },
            {
                "id": "c1",
                "stage": "core",
                "instance": "inputs/a",
                "k": 1,
                "memory_gib": 3,
            },
        ],
    }
    put(tmp_path / "suite.json", document)
    args = suite.parser().parse_args(
        [
            "--suite-root",
            str(tmp_path),
            "--stage",
            "all",
            "--binary",
            str(tmp_path / "binary"),
            "--build-manifest",
            str(tmp_path / "build.json"),
            "--output",
            str(tmp_path / "results"),
        ]
    )
    return tmp_path, document, args


def test_real_protocol_defaults_and_new_map_budget(prepared):
    root, _, args = prepared
    _, jobs, resources = suite.load_suite(root, args)
    assert resources["max_parallel"] == 1
    assert [j["seconds"] for j in jobs] == [7200] * 3
    assert [j["threads"] for j in jobs] == [1] * 3
    assert [j["exclusive"] for j in jobs] == [True, True, False]
    assert jobs[0]["job_wall_seconds"] == 9450
    command = suite.make_command(
        jobs[0],
        Path("/binary"),
        Path("/build.json"),
        Path("/attempt"),
        Path("/python"),
        root / "maps",
    )
    assert "server_run.py" in command[1]
    assert command[command.index("--method") + 1] == "algorithm"
    assert "--preset" not in command
    assert command[command.index("--maps-root") + 1] == str(root / "maps")


@pytest.mark.parametrize(
    "mutation",
    [
        lambda d: d.update(schema="old"),
        lambda d: d["defaults"].update(seconds=60),
        lambda d: d["defaults"].update(threads=True),
        lambda d: d["jobs"][0].update(threads=True),
        lambda d: d["jobs"][0].update(seconds=60),
        lambda d: d["jobs"][0].update(k=3),
        lambda d: d["jobs"][0].update(instance="../escape"),
        lambda d: d["jobs"][0].update(id="../escape"),
        lambda d: d["jobs"][1].update(id="p1"),
        lambda d: d["jobs"][1].update(enabled=False),
        lambda d: d["jobs"][0].update(memory_gib=7),
        lambda d: d["resources"].update(system_available_floor_gib=1),
    ],
)
def test_reject_invalid_or_changed_protocol_before_launch(prepared, mutation):
    root, document, args = prepared
    mutation(document)
    put(root / "suite.json", document)
    with pytest.raises((ValueError, FileNotFoundError)):
        suite.load_suite(root, args)


def test_input_hashes_checked_and_unknown_jobs_rejected(prepared):
    root, _, args = prepared
    _, jobs, _ = suite.load_suite(root, args)
    args.job = ["absent"]
    with pytest.raises(ValueError):
        suite.select_jobs(jobs, args)
    (root / "inputs/a/model.json").write_text("{}")
    with pytest.raises(ValueError, match="SHA-256"):
        suite.load_suite(root, args)


def successful_native(attempt, *, qualified=True):
    own = suite.read(attempt / "attempt.json")
    native = attempt / "native"
    values = {
        "result.json": {"objective": 100, "bound": 99.5, "gap": 0.005},
        "solution.json": {},
        "path_flows.json": {},
        "input_reference.json": {},
    }
    for name, value in values.items():
        put(native / name, value)
    cert = {
        "passed": True,
        "max_orders": own["job"]["k"],
        "native_manifest_matches": True,
        "assignment_scope": "task",
        "manifest_sha256": own["job"]["input_sha256"]["manifest.json"],
    }
    for name in values:
        cert[name.replace(".json", "_sha256")] = suite.digest(native / name)
    put(native / "native_validation.json", cert)
    put(
        native / "task_status.json",
        {
            "status": "completed",
            "state": "qualified_baseline" if qualified else "validated_not_qualified",
            "qualified_baseline": qualified,
            "validation_passed": True,
            "solver_returncode": 0,
            "validation_returncode": 0,
            "binary_sha256": own["binary_sha256"],
            "build_manifest_sha256": own["build_manifest_sha256"],
            "input_sha256": own["job"]["input_sha256"],
            "objective": 100,
            "bound": 99.5,
            "gap": 0.005,
            "bound_scope": "global",
            "termination_reason": "gap_limit",
        },
    )


def saved_attempt(root, job, identity, qualified=True):
    attempt = root / "jobs" / job["id"] / "attempts/001"
    own = {
        "job": job,
        "signature": suite.job_signature(job, identity),
        "status": "completed",
        "returncode": 0,
        "stop_reason": None,
        "group_cleanup_complete": True,
        "binary_sha256": "binary",
        "build_manifest_sha256": "manifest",
    }
    put(attempt / "attempt.json", own)
    successful_native(attempt, qualified=qualified)
    own["artifact_sha256"] = suite.attempt_hashes(attempt)
    put(attempt / "attempt.json", own)
    return attempt


def test_resume_requires_certificate_all_artifacts_and_exact_identity(prepared):
    root, _, args = prepared
    _, jobs, _ = suite.load_suite(root, args)
    identity = {"binary_sha256": "binary", "build_manifest_sha256": "manifest"}
    attempt = saved_attempt(root, jobs[0], identity)
    signature = suite.job_signature(jobs[0], identity)
    assert suite.completed_attempt(attempt, signature, qualified=True)
    assert not suite.completed_attempt(attempt, "wrong")
    cert = attempt / "native/native_validation.json"
    original = cert.read_text()
    cert.unlink()
    assert not suite.completed_attempt(attempt, signature)
    cert.write_text(original)
    (attempt / "native/solution.json").write_text('{"changed":true}')
    assert not suite.completed_attempt(attempt, signature)


def test_resume_cannot_promote_abnormal_validated_or_nonqualified_preflight(prepared):
    root, _, args = prepared
    _, jobs, _ = suite.load_suite(root, args)
    identity = {"binary_sha256": "binary", "build_manifest_sha256": "manifest"}
    attempt = saved_attempt(root, jobs[0], identity, qualified=False)
    signature = suite.job_signature(jobs[0], identity)
    assert suite.completed_attempt(attempt, signature)
    assert not suite.completed_attempt(attempt, signature, qualified=True)
    with pytest.raises(ValueError, match="1%"):
        suite.require_preflight(root, jobs, identity)
    own = suite.read(attempt / "attempt.json")
    own.update(returncode=-9, status="failed")
    put(attempt / "attempt.json", own)
    assert not suite.completed_attempt(attempt, signature)
    suite.write_summary(root)
    row = suite.read(root / "summary.json")["attempts"][0]
    assert row["state"] == "validated_abnormal_termination"
    assert not row["qualified_baseline"]


def test_output_lock_excludes_second_controller(tmp_path):
    with suite.OutputLock(tmp_path):
        with pytest.raises(RuntimeError, match="lock"):
            with suite.OutputLock(tmp_path):
                pass
    with suite.OutputLock(tmp_path):
        pass


def test_memory_reservation_is_exclusive_and_accounts_unfilled_rss(prepared):
    root, _, args = prepared
    _, jobs, policy = suite.load_suite(root, args)
    policy["max_parallel"] = 8
    active = [{"job": jobs[2], "sample": {"rss_bytes": 0}}]
    assert not suite.reservation_fits(jobs[0], active, 100 * suite.GIB, policy)
    assert not suite.reservation_fits(jobs[2], active, 13 * suite.GIB, policy)
    assert suite.reservation_fits(jobs[2], active, 14 * suite.GIB, policy)


def fake_scheduler(monkeypatch, qualified=True, fail_core=False):
    identity = {"binary_sha256": "binary", "build_manifest_sha256": "manifest"}
    launches = []
    monkeypatch.setattr(suite, "require_linux", lambda: None)
    monkeypatch.setattr(suite, "run_identity", lambda *a: identity)
    monkeypatch.setattr(
        suite.psutil,
        "virtual_memory",
        lambda: type(
            "Memory", (), {"available": 40 * suite.GIB, "total": 48 * suite.GIB}
        )(),
    )

    class FakeProcess:
        def __init__(
            self, command, cwd, directory, memory_bytes, wall_seconds, env, **kwargs
        ):
            self.attempt = Path(directory)
            own = suite.read(self.attempt / "attempt.json")
            launches.append(own["job"]["id"])
            successful_native(self.attempt, qualified=qualified)
            self.returncode = 1 if fail_core and own["job"]["stage"] == "core" else 0
            self.stop_reason = None
            self.last_sample = {"rss_bytes": 0}

        def tick(self):
            return True

        def close(self):
            pass

        def stop(self, reason):
            self.stop_reason = reason

        def record(self):
            # The resource guardian and wrapper both use a status field. This
            # exercises the old duplicate-key crash path through completion.
            return {
                "status": "finished",
                "returncode": self.returncode,
                "stop_reason": self.stop_reason,
                "group_cleanup_complete": True,
                "sampled_rss_peak_bytes": 10,
            }

    monkeypatch.setattr(suite, "ControlledProcess", FakeProcess)
    return launches


def test_preflight_blocks_entire_matrix_when_not_qualified(prepared, monkeypatch):
    root, _, args = prepared
    launches = fake_scheduler(monkeypatch, qualified=False)
    path, jobs, resources = suite.load_suite(root, args)
    assert suite.execute(args, path, jobs, resources) == 1
    assert launches == ["p1"]
    assert (
        suite.read(args.output / "suite_status.json")["dispatch_stop_reason"]
        == "preflight_not_validated_at_one_percent"
    )


def test_complete_then_resume_has_no_duplicate_launch_or_status_collision(
    prepared, monkeypatch
):
    root, _, args = prepared
    launches = fake_scheduler(monkeypatch)
    path, jobs, resources = suite.load_suite(root, args)
    assert suite.execute(args, path, jobs, resources) == 0
    assert launches == ["p1", "p2", "c1"]
    args.resume = True
    assert suite.execute(args, path, jobs, resources) == 0
    assert launches == ["p1", "p2", "c1"]
    assert len(suite.read(args.output / "summary.json")["attempts"]) == 3


def test_failed_tasks_need_explicit_retry_and_keep_previous_attempt(
    prepared, monkeypatch
):
    root, _, args = prepared
    launches = fake_scheduler(monkeypatch, fail_core=True)
    path, jobs, resources = suite.load_suite(root, args)
    assert suite.execute(args, path, jobs, resources) == 1
    args.resume = True
    assert suite.execute(args, path, jobs, resources) == 1
    assert launches == ["p1", "p2", "c1"]
    args.retry_failed = True
    more = fake_scheduler(monkeypatch)
    assert suite.execute(args, path, jobs, resources) == 0
    assert more == ["c1"]
    assert (args.output / "jobs/c1/attempts/001/attempt.json").is_file()
    assert (args.output / "jobs/c1/attempts/002/attempt.json").is_file()


def test_guardian_completion_requires_confirmed_cleanup(monkeypatch):
    import job_control

    guard = job_control.ControlledProcess.__new__(job_control.ControlledProcess)
    guard._read = lambda: None
    guard.process = type("Exited", (), {"poll": lambda self: 0})()
    guard.latest = {"status": "finished", "group_cleanup_complete": False}
    guard.stop_reason = guard.returncode = None
    assert guard.tick()
    assert guard.returncode == 125
    assert guard.stop_reason == "guardian_or_group_cleanup_failed"


def test_resume_rejects_live_guardian_even_if_record_says_finished(
    tmp_path, monkeypatch
):
    import job_control

    path = tmp_path / "jobs/a/attempts/001/guard.json"
    put(
        path,
        {
            "status": "finished",
            "group_cleanup_complete": True,
            "guardian_pid": 123,
            "guardian_start_ticks": 456,
        },
    )
    monkeypatch.setattr(job_control, "same_process", lambda pid, ticks: True)
    with pytest.raises(RuntimeError, match="still alive"):
        job_control.ensure_no_live_attempts(tmp_path)
    monkeypatch.setattr(job_control, "same_process", lambda pid, ticks: False)
    job_control.ensure_no_live_attempts(tmp_path)


def test_missing_python_pidfd_binding_uses_typed_syscall(monkeypatch):
    import linux_process

    calls = []
    monkeypatch.setattr(linux_process, "_PY_PIDFD_OPEN", None)
    monkeypatch.setattr(linux_process, "_PY_PIDFD_SEND_SIGNAL", None)
    monkeypatch.setattr(
        linux_process,
        "_syscall",
        lambda number, operation, *arguments: calls.append(
            (number, operation, arguments)
        )
        or 17,
    )
    assert linux_process.pidfd_open(100) == 17
    linux_process.pidfd_send_signal(17, 0)
    assert [c[0] for c in calls] == [434, 424]
    assert calls[0][2][0].value == 100
    assert calls[1][2][2].value is None  # pointer-sized NULL siginfo


def test_kernel_pidfd_denial_is_not_downgraded_to_pid_signaling(monkeypatch):
    import errno
    import linux_process

    details = {
        "platform": "linux",
        "machine": "x86_64",
        "kernel_release": "fixture",
        "pointer_bytes": 8,
        "long_bytes": 8,
        "maxsize": 2**63 - 1,
    }
    monkeypatch.setattr(linux_process, "_platform_details", lambda: details)
    monkeypatch.setattr(
        linux_process,
        "_wait_capabilities",
        lambda: {"os.waitid": True, "os.WNOWAIT": True},
    )

    def denied(pid):
        raise OSError(errno.EPERM, "permission denied")

    monkeypatch.setattr(linux_process, "pidfd_open", denied)
    with pytest.raises(RuntimeError, match="errno=1"):
        linux_process.require_linux()
