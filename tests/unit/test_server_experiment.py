"""Portable experiment lifecycle, failure evidence and provenance contracts."""

import hashlib
import json
import subprocess

import pytest

from scripts import run_server_experiment as runner


def setup_case(tmp_path, monkeypatch, *, bases=3, k=2):
    binary = tmp_path / "solver with spaces.exe"
    binary.write_bytes(b"frozen-test-executable")
    instance = tmp_path / "instance"
    instance.mkdir()
    model = instance / "model.json"
    model.write_text(json.dumps({"values": {"n_l": bases}}))
    (instance / "manifest.json").write_text(json.dumps({
        "files": {"model.json": hashlib.sha256(model.read_bytes()).hexdigest()}
    }))
    monkeypatch.setattr(runner, "source_fingerprints", lambda: {"test": "fixed"})
    args = runner.parser().parse_args([
        "--binary", str(binary), "--instance", str(instance),
        "--out", str(tmp_path / "new output"), "--preset", "performance-20261105",
        "--k", str(k), "--seconds", "1",
    ])
    return args


def fake_solver(args, *, valid=True, solver_code=0, mutate=None):
    def invoke(command, output, prefix, timeout):
        assert isinstance(command, list) and timeout > 0
        target = output / "run"
        target.mkdir(exist_ok=True)
        if prefix == "solver":
            (target / "result.json").write_text(json.dumps(dict(
                assignment_scope="task", delivery_time_scope="courier_home", max_orders=args.k,
                objective=20., bound=10., gap=.5, termination_reason="time_limit",
                pricing_complete=False,
            )))
            return solver_code
        (target / "native_validation.json").write_text(json.dumps({"passed": valid}))
        if mutate:
            mutate()
        return 0
    return invoke


@pytest.mark.parametrize("bases,k,alns", [(2, 2, True), (2, 1, False), (3, 2, False)])
def test_run_records_bounds_time_limit_and_scope_disables(tmp_path, monkeypatch, bases, k, alns):
    args = setup_case(tmp_path, monkeypatch, bases=bases, k=k)
    monkeypatch.setattr(runner, "invoke", fake_solver(args))
    result = runner.run(args)
    assert result["status"] == "completed" and result["validation_passed"]
    assert result["termination_reason"] == "time_limit" and result["gap"] == .5
    assert result["objective"] == 20. and result["bound"] == 10.
    assert result["disabled_accelerations"] == ([] if alns else sorted(runner.ALNS_FLAGS))
    assert result["command"][0] == str(args.binary)
    assert result["source_unchanged"] and result["input_unchanged"] and result["binary_unchanged"]
    with pytest.raises(FileExistsError):
        runner.run(args)


@pytest.mark.parametrize("valid,code", [(False, 0), (True, 7)])
def test_solver_and_validation_failure_never_complete(tmp_path, monkeypatch, valid, code):
    args = setup_case(tmp_path, monkeypatch)
    monkeypatch.setattr(runner, "invoke", fake_solver(args, valid=valid, solver_code=code))
    result = runner.run(args)
    assert result["status"] == "failed" and "error" in result
    assert result["solver_returncode"] == code


def test_timeout_is_persisted(tmp_path, monkeypatch):
    args = setup_case(tmp_path, monkeypatch)
    def timeout(*_):
        raise subprocess.TimeoutExpired("test", 1)
    monkeypatch.setattr(runner, "invoke", timeout)
    result = runner.run(args)
    assert result["status"] == "failed" and result["error"] == "solver_external_wall_limit"
    assert not result["validation_passed"]
    assert json.loads((args.out / "experiment.json").read_text())["status"] == "failed"


def test_changed_binary_fails_even_after_validation(tmp_path, monkeypatch):
    args = setup_case(tmp_path, monkeypatch)
    monkeypatch.setattr(runner, "invoke", fake_solver(args, mutate=lambda: args.binary.write_bytes(b"changed")))
    result = runner.run(args)
    assert result["validation_passed"] and not result["binary_unchanged"]
    assert result["status"] == "failed" and "provenance_error" in result


def test_tampered_inputs_fail_before_start(tmp_path, monkeypatch):
    args = setup_case(tmp_path, monkeypatch)
    (args.instance / "model.json").write_text("{}")
    with pytest.raises(ValueError, match="mismatch"):
        runner.run(args)
    assert not args.out.exists()


def test_manifest_cannot_escape_instance(tmp_path, monkeypatch):
    args = setup_case(tmp_path, monkeypatch)
    path = args.instance / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["files"]["../solver with spaces.exe"] = runner.digest(args.binary)
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="leaves instance"):
        runner.run(args)


def test_legacy_model_result_rejected(tmp_path, monkeypatch):
    args = setup_case(tmp_path, monkeypatch)
    regular = fake_solver(args)
    def invoke(*parameters):
        code = regular(*parameters)
        target = args.out / "run" / "result.json"
        payload = json.loads(target.read_text())
        payload["assignment_scope"] = "origin"
        target.write_text(json.dumps(payload))
        return code
    monkeypatch.setattr(runner, "invoke", invoke)
    result = runner.run(args)
    assert result["status"] == "failed" and "current task" in result["error"]


def test_external_limit_does_not_silently_shorten_internal_budget(tmp_path, monkeypatch):
    args = setup_case(tmp_path, monkeypatch)
    args.seconds, args.wall_limit = 600, 1
    with pytest.raises(ValueError, match="cover"):
        runner.run(args)
    assert not args.out.exists()


@pytest.mark.parametrize("seconds", ["0", "-1", "nan", "inf"])
def test_invalid_budget_rejected_before_start(seconds):
    with pytest.raises(SystemExit):
        runner.parser().parse_args([
            "--binary", "unused", "--instance", "unused", "--out", "unused",
            "--preset", "performance-20261105", "--k", "2", f"--seconds={seconds}",
        ])
