"""Public serial-batch contracts; synthetic fixtures, never a solver run."""
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

import pytest

from src.experiments import l2_series_20261029 as bench


def planned_instances():
    return [dict(id=f"l2_h{h:02}_od{r:02}_r{k:02}", instance=f"instances/l2_h{h:02}_od{r:02}_r{k:02}",
                 n_l=2, n_h=h, n_od=r, replicate=k, seed=2026102900+k,
                 input_manifest_sha256="0" * 64)
            for h, r in ((10, 20), (20, 20), (20, 40), (40, 40)) for k in range(1, 6)]


def test_twenty_inputs_create_forty_serial_equal_budget_commands_without_catalogue_export():
    jobs = bench.jobs_for(planned_instances())
    assert len(jobs) == len({j["id"] for j in jobs}) == 40
    cfg = bench.protocol()
    assert cfg["parallel_jobs"] == 1
    assert cfg["resource_limits"]["job_rss_bytes"] == 8 * 1024**3
    assert cfg["resource_limits"]["batch_rss_bytes"] == 12 * 1024**3
    assert cfg["resource_limits"]["system_available_floor_bytes"] == 2 * 1024**3
    assert cfg["hard_wall_seconds"] == 720
    for job in jobs:
        command = bench.command_for(Path("/batch"), {"protocol": cfg}, job, "/attempt")
        value = lambda name: command[command.index(name) + 1]
        assert command[0] == "/batch/frozen/bin/mlnd_cpp"
        assert value("--preset") == ("full-cplex-20261029" if job["method"] == "full_cplex" else "baseline-20261029")
        for name, expected in (("--seconds", "600"), ("--gap", "0.01"), ("--threads", "1"),
                               ("--seed", "43"), ("--k", "2"), ("--path-column-limit", "0"),
                               ("--demand-formulation", "preprocessed")):
            assert value(name) == expected
        assert not {"--export-catalogue", "--start", "--fixed-design", "--first-leg-consistency"} & set(command)


def prepared_inputs(root):
    rows = planned_instances()
    for row in rows:
        folder = root / row["instance"]
        folder.mkdir(parents=True)
        (folder / "model.json").write_text(json.dumps(dict(values={k: row[k] for k in ("n_l", "n_h", "n_od")})))
        (folder / "manifest.json").write_text(json.dumps({"case": row["id"]}))
        row["manifest_sha256"] = bench.base.sha256(folder / "manifest.json")
        row["status"] = "prepared_validated_not_solved"
    manifest = dict(schema="tsl-matched-l2-preparation/1", status="prepared_validated_not_solved",
        physical_instances=20, expected_solve_jobs=40, K=2, demand_formulation="preprocessed",
        assignment_scope="origin", flow_policy="aggregate", solver_invoked=False, instances=rows)
    path = root / "preparation_manifest.json"
    path.write_text(json.dumps(manifest))
    return path


def test_inspect_rejects_missing_duplicate_and_changed_input_without_solving(tmp_path):
    path = prepared_inputs(tmp_path)
    checked = bench.inspect(path)
    assert checked["solver_started"] is False and len(checked["jobs"]) == 40
    raw = json.loads(path.read_text())
    for rows in (raw["instances"][:-1], [raw["instances"][0]] * 20):
        path.write_text(json.dumps(dict(raw, instances=rows)))
        with pytest.raises(ValueError): bench.inspect(path)
    path.write_text(json.dumps(raw))
    folder = tmp_path / raw["instances"][0]["instance"]
    (folder / "model.json").write_text(json.dumps({"values": {"n_l": 2, "n_h": 11, "n_od": 20}}))
    with pytest.raises(ValueError, match="dimension"): bench.inspect(path)


def resolved_payload(job):
    full = job["method"] == "full_cplex"
    preset = "full-cplex-20261029" if full else "baseline-20261029"
    enabled = bench.policy.SHARED_NAMES if full else bench.policy.BASELINE_ENABLED
    return dict(preset=preset, resolved_configuration=dict(schema="mlnd-solver-configuration/1", preset=preset,
        method="full_path" if full else "benders_cg", demand_formulation="preprocessed", first_leg_consistency=False,
        seconds=600, gap=.01, threads=1, seed=43, max_iterations=1000000, cg_max_columns=512,
        path_column_limit_argument=0, export_catalogue=False,
        flags={k: k in enabled for k in bench.policy.FLAG_NAMES}), assignment_scope="origin", max_orders=2,
        **{k: job[k] for k in ("n_l", "n_h", "n_od")})


def test_preflight_checks_all_forty_read_only_commands_and_rejects_export_drift(tmp_path, monkeypatch):
    path = prepared_inputs(tmp_path)
    jobs = bench.inspect(path)["jobs"]
    binary = tmp_path / "binary"; binary.write_text("binary identity")
    seen = []
    def invoke(command, **kwargs):
        assert "--inspect" in command and "--out" not in command and "--export-catalogue" not in command
        assert kwargs["timeout"] == 30 and kwargs["check"]
        row = jobs[len(seen)]["job"]; seen.append(command)
        return SimpleNamespace(stdout=json.dumps(resolved_payload(row)), stderr="")
    monkeypatch.setattr(bench.subprocess, "run", invoke)
    evidence = bench.preflight(binary, path)
    assert evidence["passed"] and evidence["solver_invoked"] is False
    assert len(seen) == len(evidence["rows"]) == 40
    payload = resolved_payload(jobs[1]["job"])
    assert not bench.configuration_errors(payload, "full_cplex", check_structure=False)
    payload["resolved_configuration"]["export_catalogue"] = True
    assert bench.configuration_errors(payload, "full_cplex", check_structure=False)


def frozen_batch(tmp_path):
    source = tmp_path / "source"; source.mkdir()
    preparation = prepared_inputs(source)
    batch = tmp_path / "batch"; batch.mkdir()
    shutil.copytree(source / "instances", batch / "instances")
    provenance = batch / "provenance"; provenance.mkdir()
    shutil.copyfile(preparation, provenance / preparation.name)
    binary = batch / "frozen/bin/mlnd_cpp"; binary.parent.mkdir(parents=True); binary.write_text("binary")
    jobs = bench.jobs_for(json.loads(preparation.read_text())["instances"])
    rows = []
    for job in jobs:
        payload = resolved_payload(job); stdout = json.dumps(payload)
        rows.append(dict(job=job["id"], payload=payload, stdout=stdout,
                         stdout_sha256=bench.hashlib.sha256(stdout.encode()).hexdigest()))
    bench.write_json(provenance / "PREFLIGHT.json", dict(passed=True, solver_invoked=False,
        binary_sha256=bench.base.sha256(binary), rows=rows))
    for name in bench.REQUIRED_PROVENANCE:
        if not (provenance / name).exists(): (provenance / name).write_text("{}")
    bench.write_json(provenance / "BINARY_IDENTITY.json", dict(passed=True,
        binary_sha256=bench.base.sha256(binary), source_matches_validated_release=True))
    (batch / "run.py").write_text("# frozen runner")
    manifest = dict(schema="l2-series-benchmark/1", jobs=jobs, protocol=bench.protocol(), python=sys.executable,
        binary_sha256=bench.base.sha256(binary), frozen_sha256={str(p.relative_to(batch)): bench.base.sha256(p)
        for p in batch.rglob("*") if p.is_file()})
    bench.write_json(batch / "experiment_manifest.json", manifest)
    bench.write_json(batch / "state.json", dict(status="prepared_not_started",
        jobs={j["id"]: dict(status="pending", attempts=[]) for j in jobs}))
    return batch


def test_verify_rejects_parallel_resource_and_frozen_input_drift(tmp_path):
    batch = frozen_batch(tmp_path)
    original = bench.verify(batch)
    for key, value in (("parallel_jobs", 2), ("total_jobs", 39)):
        changed = deepcopy(original); changed["protocol"][key] = value
        bench.write_json(batch / "experiment_manifest.json", changed)
        with pytest.raises(ValueError): bench.verify(batch)
    changed = deepcopy(original); changed["protocol"]["resource_limits"]["job_rss_bytes"] = 4 * 1024**3
    bench.write_json(batch / "experiment_manifest.json", changed)
    with pytest.raises(ValueError): bench.verify(batch)
    bench.write_json(batch / "experiment_manifest.json", original)
    (batch / original["jobs"][0]["instance"] / "model.json").write_text("changed")
    with pytest.raises(ValueError, match="Frozen artifact"): bench.verify(batch)


def test_serial_dispatch_records_failure_then_continues_and_routes_lightweight_children(tmp_path, monkeypatch):
    batch = frozen_batch(tmp_path)
    state = bench.base.read(batch / "state.json")
    jobs = bench.verify(batch)["jobs"]
    for job in jobs[2:]: state["jobs"][job["id"]]["status"] = "failed"
    bench.write_json(batch / "state.json", state)
    events = []; running = []
    def worker(batch, manifest, job_id, command, attempt, cancel, **kwargs):
        assert not running
        running.append(job_id)
        try:
            action = command[3] if "-m" in command else "solver"
            if "-m" in command: assert command[2] == bench.MODULE
            events.append((job_id, action))
            if job_id == jobs[0]["id"]: raise RuntimeError("synthetic solver process failure")
            if action == "audit-worker":
                target = Path(command[command.index("--attempt") + 1])
                bench.write_json(target / "artifact_audit.json", dict(passed=True))
            return dict(returncode=0)
        finally: running.pop()
    # The guard is the external process boundary; no native solver is invoked.
    monkeypatch.setattr(bench, "_BASE_GUARD", worker)
    bench.run(batch)
    status = bench.base.read(batch / "state.json")
    assert status["status"] == "finished_dispatch"
    assert status["jobs"][jobs[0]["id"]]["status"] == "failed"
    assert status["jobs"][jobs[1]["id"]]["status"] == "completed"
    assert [action for _, action in events] == ["solver", "solver", "audit-worker", "report"]
    assert all(len(j["attempts"]) <= 1 for j in status["jobs"].values())


def test_launch_is_detached_returns_identity_and_rejects_parallel_override(tmp_path, monkeypatch):
    batch = frozen_batch(tmp_path)
    calls = []
    def detached(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(pid=os.getpid())
    monkeypatch.setattr(bench.base.subprocess, "Popen", detached)
    result = bench.launch(batch)
    assert result["detached"] and result["pid"] == os.getpid()
    assert result["manifest_sha256"] == bench.base.sha256(batch / "experiment_manifest.json")
    command, opts = calls[0]
    assert Path(command[0]).is_absolute() and command[-1] == "run"
    assert opts["stdin"] == bench.subprocess.DEVNULL and opts["start_new_session"] and opts["close_fds"]
    assert opts["stdout"] is opts["stderr"] and opts["cwd"] == batch / "frozen"
    with pytest.raises(SystemExit): bench.main(["launch", "--batch", str(batch), "--parallel", "2"])
    with pytest.raises(SystemExit): bench.main(["run", "--batch", str(batch), "--retry-failed"])
    assert len(calls) == 1


def test_terminal_report_preserves_resource_failures_without_full_catalogue_or_cut_audit(tmp_path):
    batch = frozen_batch(tmp_path)
    manifest = bench.verify(batch); state = bench.base.read(batch / "state.json")
    for job in manifest["jobs"]:
        state["jobs"][job["id"]]["status"] = "failed"
    first = manifest["jobs"][0]
    attempt = batch / first["output"] / "attempts/001"; attempt.mkdir(parents=True)
    state["jobs"][first["id"]].update(directory=str(attempt.relative_to(batch)))
    bench.write_json(attempt / "execution.json", dict(returncode=-15, stop_reason="construction_memory_limit"))
    bench.write_json(attempt / "guard_status.json", dict(termination_reason="construction_memory_limit",
        resource_limit_kind="per_job_rss_limit", stage="construction"))
    bench.write_json(batch / "state.json", state)
    result = bench.report(batch)
    payload = bench.base.read(batch / "analysis/comparison.json")
    assert result["jobs"] == 40 and payload["declared_jobs_terminal"]
    assert payload["validated_runs"] == 0 and not payload["all_jobs_completed"]
    assert payload["independent_cut_audit_performed"] is False
    assert payload["rows"][0]["termination_reason"] == "construction_memory_limit"
    assert all(row["upper_bound"] is None and row["lower_bound"] is None for row in payload["rows"])
    assert not (batch / "analysis/CUT_AUDIT.json").exists()


def test_report_requires_current_native_hashes_and_exact_preflight_configuration(tmp_path):
    batch = frozen_batch(tmp_path)
    manifest = bench.verify(batch); job = manifest["jobs"][0]
    attempt = batch / job["output"] / "attempts/001"; attempt.mkdir(parents=True)
    state = bench.base.read(batch / "state.json")
    state["jobs"][job["id"]].update(status="completed", directory=str(attempt.relative_to(batch)))
    bench.write_json(batch / "state.json", state)
    result = dict(resolved_payload(job), max_orders=2, solver="CPLEX", method="benders_cg",
        demand_formulation="preprocessed", flow_policy="aggregate", objective=100.0, bound=99.5,
        bound_scope="global", path_column_limit_argument=0, binary_sha256=manifest["binary_sha256"],
        threads=1, seed=43, budget=600, requested_gap=.01, task_airline_bounds=True,
        structural_strengthening={k: True for k in ("shared_rho_joint", "queue_perspective", "task_airline_bounds")},
        wall_before_output=30, timings={}, termination_reason="gap_limit", target_gap_met=True)
    for name, content in (("result.json", result), ("solution.json", {}), ("path_flows.json", {}),
                          ("input_reference.json", {"first_leg_consistency": False}), ("preprocessed_demand.json", {})):
        bench.write_json(attempt / name, content)
    validation = dict(passed=True, manifest_sha256=bench.base.sha256(batch / job["instance"] / "manifest.json"),
        **{key: bench.base.sha256(attempt / name) for name, key in (("result.json", "result_sha256"),
          ("solution.json", "solution_sha256"), ("path_flows.json", "path_flows_sha256"))})
    bench.write_json(attempt / "native_validation.json", validation)
    bench.write_json(attempt / "artifact_audit.json", dict(passed=True,
        artifact_sha256={p.name: bench.base.sha256(p) for p in attempt.iterdir()}))
    bench.report(batch)
    row = bench.base.read(batch / "analysis/comparison.json")["rows"][0]
    assert row["validation_passed"] and row["upper_bound"] == 100 and row["lower_bound"] == 99.5 and row["gap"] == .005
    result["resolved_configuration"]["seconds"] = 599
    bench.write_json(attempt / "result.json", result)
    # Even a newly signed native report cannot make policy drift acceptable.
    validation["result_sha256"] = bench.base.sha256(attempt / "result.json")
    bench.write_json(attempt / "native_validation.json", validation)
    bench.write_json(attempt / "artifact_audit.json", dict(passed=True,
        artifact_sha256={p.name: bench.base.sha256(p) for p in attempt.iterdir() if p.name != "artifact_audit.json"}))
    bench.report(batch)
    row = bench.base.read(batch / "analysis/comparison.json")["rows"][0]
    assert not row["validation_passed"] and row["upper_bound"] is None and row["lower_bound"] is None
    assert row["configuration_errors"]
    (attempt / "solution.json").write_text("changed after validation")
    bench.report(batch)
    assert not bench.base.read(batch / "analysis/comparison.json")["rows"][0]["validation_passed"]
