"""Version isolation, resume environment and inherited batch-lock regressions."""
from argparse import Namespace
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
import server_build as build  # noqa: E402 -- test the standalone script imports
import server_start as start  # noqa: E402


def write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def write_json(path, value):
    return write(path, json.dumps(value))


def git(root, *args):
    return subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True)


@pytest.fixture
def clean_repository(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    git(root, "init", "-b", "MLND_sever")
    source = write(root / "solver.cpp", "current solver\n")
    write_json(root / "SOURCE_VERSION.json", {"files": {"solver.cpp": {"sha256": build.sha256(source)}}})
    write(root / "scripts/entry.py", "print('committed entry')\n")
    write(root / ".gitignore", "/outputs/\n")
    git(root, "add", ".")
    git(root, "-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-m", "fixture")
    return root


def test_snapshot_excludes_local_data_and_remains_bound_to_commit(clean_repository, tmp_path):
    root = clean_repository
    write(root / "outputs/results.json", "private experiment result")
    write(root / "untracked.py", "not committed")
    destination = tmp_path / "snapshot"
    record = build.snapshot(root, destination)
    assert record["git_commit"] == git(root, "rev-parse", "HEAD").stdout.decode().strip()
    assert not (destination / "outputs").exists()
    assert not (destination / "untracked.py").exists()
    assert (destination / "scripts/entry.py").read_bytes() == (root / "scripts/entry.py").read_bytes()
    start.verify_files(destination, "SNAPSHOT.json")
    write(destination / "scripts/entry.py", "changed frozen entry")
    with pytest.raises(ValueError, match="changed"):
        start.verify_files(destination, "SNAPSHOT.json")


def test_snapshot_rejects_concurrent_staged_changes(clean_repository, tmp_path, monkeypatch):
    root = clean_repository
    original = build.source_hashes
    staged = False

    def stage_after_initial_clean_check(path):
        nonlocal staged
        result = original(path)
        if not staged:
            staged = True
            write(root / "scripts/entry.py", "print('new staged entry')\n")
            git(root, "add", "scripts/entry.py")
        return result

    monkeypatch.setattr(build, "source_hashes", stage_after_initial_clean_check)
    with pytest.raises(ValueError, match="changed|changes"):
        build.snapshot(root, tmp_path / "snapshot")


def test_snapshot_rejects_tracked_symlink(clean_repository, tmp_path):
    root = clean_repository
    (root / "outside-link").symlink_to(tmp_path / "outside")
    git(root, "add", "outside-link")
    git(root, "-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-m", "link fixture")
    with pytest.raises(ValueError, match="regular files"):
        build.snapshot(root, tmp_path / "snapshot")


def test_input_snapshot_preserves_verified_inputs_not_old_solver(tmp_path):
    suite = tmp_path / "old-bundle/suite"
    model = write_json(suite / "instances/example/model.json", {"version": "input"})
    write_json(suite / "instances/example/manifest.json", {"files": {"model.json": build.sha256(model)}})
    write_json(suite / "suite.json", {"schema": "mlnd-server-capacity-suite/1", "jobs": [
        {"id": "example_K1", "instance": "instances/example", "k": 1}]})
    write_json(suite / "provenance/map_backgrounds.json", {"backgrounds": []})
    write(suite / "maps/background.png", "existing local basemap")
    write(suite / "tools/run_suite.py", "obsolete launcher")
    write(suite / "source/mlnd", "obsolete binary")
    frozen = tmp_path / "frozen-inputs"
    record = start.freeze_suite(suite.parent, frozen)
    assert record["jobs"] == 1
    assert (frozen / "instances/example/model.json").read_bytes() == model.read_bytes()
    assert not (frozen / "tools").exists()
    assert not (frozen / "source").exists()
    start.verify_files(frozen, "SERVER_INPUT_SNAPSHOT.json")
    write(frozen / "maps/background.png", "changed basemap")
    with pytest.raises(ValueError, match="changed"):
        start.verify_files(frozen, "SERVER_INPUT_SNAPSHOT.json")


def test_input_snapshot_rejects_manifest_escape(tmp_path):
    suite = tmp_path / "suite"
    write_json(suite / "suite.json", {"schema": "mlnd-server-capacity-suite/1", "jobs": [
        {"instance": "../outside", "k": 1}]})
    write_json(tmp_path / "outside/manifest.json", {"files": {}})
    with pytest.raises(ValueError, match="relative path"):
        start.freeze_suite(suite, tmp_path / "copy")


def test_background_batch_keeps_global_flock_after_launcher_closes(tmp_path, monkeypatch):
    run = tmp_path / "run-test"
    write(run / "source/scripts/server_suite.py", "import time\ntime.sleep(30)\n")
    binary = write(run / "build/mlnd", "fake binary")
    write_json(run / "build/build_manifest.json", {"binary_sha256": build.sha256(binary)})
    monkeypatch.setattr(start, "verify_files", lambda *args: None)
    actual_popen = subprocess.Popen
    children = []

    def capture_popen(*args, **kwargs):
        process = actual_popen(*args, **kwargs)
        children.append(process)
        return process

    monkeypatch.setattr(start.subprocess, "Popen", capture_popen)
    lock_path = tmp_path / "global.lock"
    lock = lock_path.open("a+")
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    probe = [sys.executable, "-c", (
        "import fcntl,sys\n"
        "f=open(sys.argv[1],'a+')\n"
        "try: fcntl.flock(f.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)\n"
        "except BlockingIOError: sys.exit(75)\n"
    ), str(lock_path)]
    try:
        assert start.launch(run, lock, Namespace(retry_failed=False, foreground=False),
                            {"stage": "all", "max_parallel": 8}) == 0
        lock.close()
        check = actual_popen(probe)
        assert check.wait(timeout=5) == 75
    finally:
        lock.close()
        for child in children:
            if child.poll() is None:
                child.terminate()
                child.wait(timeout=5)
    check = actual_popen(probe)
    assert check.wait(timeout=5) == 0


def test_resume_restores_original_cplex_runtime_path(tmp_path, monkeypatch):
    runs = tmp_path / "runs"
    run = runs / "run-existing"
    cplex = tmp_path / "custom-cplex"
    write(runs / "CURRENT", "run-existing\n")
    write_json(run / "prepared.json", {"stage": "all", "max_parallel": 1,
                                       "python": sys.executable, "cplex_root": str(cplex),
                                       "runtime_environment": start.runtime_environment()})
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(sys, "argv", ["server_start.py", "--resume", "--runs-root", str(runs)])
    monkeypatch.setenv("MLND_LOCK_FILE", str(tmp_path / "global.lock"))
    monkeypatch.delenv("MLND_LAUNCH_LOCK_FD", raising=False)
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    monkeypatch.setenv("LD_LIBRARY_PATH", "/existing/lib")
    launched = []

    def launch(run_path, lock, args, config):
        expected = str(cplex / "cplex/bin/x86-64_linux")
        assert os.environ["PATH"].split(os.pathsep)[0] == expected
        assert os.environ["LD_LIBRARY_PATH"].split(os.pathsep)[0] == expected
        assert run_path == run
        launched.append(True)
        return 0

    monkeypatch.setattr(start, "launch", launch)
    assert start.main() == 0
    assert launched == [True]


def test_new_batch_builds_then_launches_eight_without_acceptance(clean_repository, tmp_path, monkeypatch):
    root = clean_repository
    suite = tmp_path / "suite"
    model = write_json(suite / "instances/example/model.json", {"version": "input"})
    write_json(suite / "instances/example/manifest.json", {"files": {"model.json": build.sha256(model)}})
    write_json(suite / "suite.json", {"schema": "mlnd-server-capacity-suite/1", "jobs": [
        {"id": "example_K1", "instance": "instances/example", "k": 1}]})
    write_json(suite / "provenance/map_backgrounds.json", {"backgrounds": []})
    monkeypatch.setattr(start, "ROOT", root)
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(sys, "argv", ["server_start.py", "--suite-root", str(suite),
                                     "--runs-root", str(tmp_path / "runs")])
    monkeypatch.setenv("MLND_LOCK_FILE", str(tmp_path / "global.lock"))
    monkeypatch.delenv("MLND_LAUNCH_LOCK_FD", raising=False)
    commands = []

    def build_only(command, **kwargs):
        assert Path(command[1]).name == "server_build.py", "Startup must not run test jobs"
        commands.append(command)
        return subprocess.CompletedProcess(command, 0)

    def no_test_subprocess(*args, **kwargs):
        pytest.fail("Startup invoked an acceptance process before the experiment batch")

    def launch(run, lock, args, config):
        assert config["max_parallel"] == 8
        assert config["stage"] == "all"
        assert len(commands) == 1
        assert not (run / "acceptance.json").exists()
        return 0

    # The source fixture is already copied before the subprocess boundary is mocked.
    original_snapshot = start.snapshot
    def snapshot_then_mock(source, destination):
        result = original_snapshot(source, destination)
        monkeypatch.setattr(start.subprocess, "run", build_only)
        monkeypatch.setattr(start.subprocess, "Popen", no_test_subprocess)
        return result

    monkeypatch.setattr(start, "snapshot", snapshot_then_mock)
    monkeypatch.setattr(start, "launch", launch)
    assert start.main() == 0
