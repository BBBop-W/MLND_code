"""Regression checks for version provenance and truthful task completion."""
import importlib.util
import json
import os
from pathlib import Path
import sys

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/server_run.py"
SPEC = importlib.util.spec_from_file_location("mlnd_server_run", SCRIPT)
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def write(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return path


def write_json(path, content):
    return write(path, json.dumps(content))


@pytest.fixture
def portable_build(tmp_path):
    root = tmp_path / "repo"
    files = ("cpp/src/main.cpp", "cpp/include/mlnd/input.hpp", "mlnd_runtime/__init__.py",
             "CMakeLists.txt", "requirements.txt")
    hashes = {}
    for name in files:
        hashes[name] = runner.digest(write(root / name, name))
    version = write_json(root / "SOURCE_VERSION.json", {"files": {
        name: {"sha256": sha} for name, sha in hashes.items()}})
    binary = write(root / "build/mlnd", "#!/bin/sh\nexit 0\n")
    binary.chmod(0o755)
    manifest = write_json(root / "build/build_manifest.json", {
        "schema_version": 1, "source_sha256": hashes,
        "source_version_sha256": runner.digest(version),
        "binary_sha256": runner.digest(binary), "git_commit": "test-source-commit",
        "source_root": "/old/build/machine/path", "binary_path": "/old/build/machine/path/mlnd",
    })
    return root, binary, manifest


def test_build_hashes_survive_relocation_but_reject_old_binary(portable_build):
    root, binary, manifest = portable_build
    assert runner.verify_build(binary, manifest, root)[0]["git_commit"] == "test-source-commit"
    binary.write_text("#!/bin/sh\nexit 1\n")
    with pytest.raises(ValueError, match="Binary SHA-256"):
        runner.verify_build(binary, manifest, root)


def test_build_rejects_changed_or_unlisted_solver_source(portable_build):
    root, binary, manifest = portable_build
    path = root / "cpp/src/main.cpp"
    original = path.read_text()
    path.write_text("changed solver")
    with pytest.raises(ValueError, match="Source differs"):
        runner.verify_build(binary, manifest, root)
    path.write_text(original)
    write(root / "cpp/src/extra.cpp", "unexpected compilation unit")
    with pytest.raises(ValueError, match="file set"):
        runner.verify_build(binary, manifest, root)


def test_input_manifest_rejects_escape_and_corruption(tmp_path):
    instance = tmp_path / "input"
    model = write_json(instance / "model.json", {})
    write_json(instance / "manifest.json", {"files": {"model.json": runner.digest(model)}})
    assert "manifest.json" in runner.input_fingerprints(instance)
    model.write_text("modified")
    with pytest.raises(ValueError, match="mismatch"):
        runner.input_fingerprints(instance)
    write_json(instance / "manifest.json", {"files": {
        "model.json": runner.digest(model), "../outside": "a" * 64}})
    with pytest.raises(ValueError, match="leaves its instance"):
        runner.input_fingerprints(instance)


def completed_record(**updates):
    record = dict(validation_passed=True, solver_returncode=0, validation_returncode=0,
                  solver_timed_out=False, validation_timed_out=False, optimization_completed=True,
                  has_feasible_solution=True, objective=100., bound=99.5, gap=.005,
                  bound_scope="global", target_gap=.01, status="running")
    for key in ("source_unchanged", "input_unchanged", "binary_unchanged", "runner_unchanged",
                "build_manifest_unchanged"):
        record[key] = True
    record.update(updates)
    return record


@pytest.mark.parametrize("updates,state,status", [
    ({}, "qualified_baseline", "completed"),
    ({"bound": 95., "gap": .05}, "validated_not_qualified", "completed"),
    ({"solver_returncode": 1}, "validated_abnormal_termination", "failed"),
    ({"solver_timed_out": True}, "validated_abnormal_termination", "failed"),
    ({"validation_passed": False}, "unvalidated_failed", "failed"),
    ({"source_unchanged": False}, "validated_abnormal_termination", "failed"),
    ({"optimization_completed": False}, "validated_abnormal_termination", "failed"),
    ({"bound_scope": "fixed_design"}, "validated_not_qualified", "completed"),
    ({"bound": 101.}, "validated_not_qualified", "completed"),
    ({"gap": .001}, "validated_not_qualified", "completed"),
])
def test_completion_does_not_conflate_primal_validation_and_gap(updates, state, status):
    record = runner.classify(completed_record(**updates))
    assert (record["state"], record["status"]) == (state, status)
    assert record["qualified_baseline"] is (state == "qualified_baseline")
    assert record["global_lower_bound_independently_certified"] is False


def test_external_timeout_reaps_its_owned_child(tmp_path):
    command = [sys.executable, "-c", "import os,time; print(os.getpid(), flush=True); time.sleep(60)"]
    result = runner.invoke(command, tmp_path, "solver", .5)
    assert result["timed_out"] is True
    assert result["returncode"] != 0
    pid = int((tmp_path / "solver.stdout.log").read_text().strip())
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


@pytest.mark.parametrize("solver_code,expected_state,map_failure", [(0, "qualified_baseline", False),
                                                        (7, "validated_abnormal_termination", False),
                                                        (0, "qualified_baseline", True)])
def test_run_validates_saved_incumbent_after_nonzero_exit(tmp_path, monkeypatch, solver_code, expected_state, map_failure):
    instance = tmp_path / "input"
    model = write_json(instance / "model.json", {})
    write_json(instance / "manifest.json", {"files": {"model.json": runner.digest(model)}})
    binary = write(tmp_path / "mlnd", "stand-in solver")
    manifest = write_json(tmp_path / "build_manifest.json", {})
    output = tmp_path / "result"
    args = runner.parser().parse_args([
        "--binary", str(binary), "--build-manifest", str(manifest),
        "--input", str(instance), "--output", str(output), "--k", "2",
        "--seconds", "5", "--validation-seconds", "5",
    ])
    if map_failure:
        args.maps_root = tmp_path / "maps"
    sources = {"fake.cpp": "test-sha"}
    monkeypatch.setattr(runner, "verify_build", lambda *args: ({"git_commit": "test"}, sources))
    monkeypatch.setattr(runner, "source_fingerprints", lambda: sources)
    phases = []

    def invoke(command, directory, prefix, timeout):
        phases.append(prefix)
        if prefix == "solver":
            result = dict(objective=100., bound=99.5, gap=.005, termination_reason="gap_limit",
                          bound_scope="global", optimization_completed=True, has_feasible_solution=True,
                          assignment_scope="task", delivery_time_scope="courier_home",
                          demand_formulation="preprocessed", max_orders=2, requested_method="benders_cg",
                          threads=1, seed=43, requested_gap=.01, budget=5.)
            write_json(directory / "result.json", result)
            for name in runner.OUTPUT_FILES[1:]:
                write_json(directory / name, {})
            code = solver_code
        elif prefix == "validation":
            write_json(directory / "native_validation.json", {
                "passed": True, "result_sha256": runner.digest(directory / "result.json")})
            code = 0
        else:
            write(directory / "map.stderr.log", "missing map background")
            code = 2
        return dict(returncode=code, timed_out=False, wall_seconds=.01, sampled_peak_rss_bytes=None)

    monkeypatch.setattr(runner, "invoke", invoke)
    record = runner.run(args)
    assert phases == (["solver", "validation", "map"] if map_failure else ["solver", "validation"])
    assert record["state"] == expected_state
    assert record["solver_returncode"] == solver_code
    assert record["validation_passed"] is True
    assert runner.read_json(output / "summary.json")["state"] == expected_state
    if map_failure:
        assert record["map_state"] == "failed"
        assert runner.read_json(output / "map_status.json")["map_error"] == "missing map background"
    before = {p.name: p.read_bytes() for p in output.iterdir()}
    with pytest.raises(FileExistsError):
        runner.run(args)
    assert before == {p.name: p.read_bytes() for p in output.iterdir()}
