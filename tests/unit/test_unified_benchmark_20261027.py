import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.experiments import unified_benchmark_20261027 as bench


def test_jobs_cover_each_physical_instance_exactly_four_times():
    origins = [dict(id=f"{source}_L{level}_{map_id}", instance=f"instances/{source}_L{level}_{map_id}",
                    dataset=source, map_id=f"M{map_id:02d}", n_l=level)
               for source in ("tsl", "nanshan") for level in (1, 2) for map_id in range(1, 11)]
    jobs = bench.jobs_for(origins)
    assert len(jobs) == len({job["id"] for job in jobs}) == 160
    for origin in origins:
        matching = [job for job in jobs if job["instance"] == origin["instance"]]
        assert {(j["K"], j["method"]) for j in matching} == {
            (1, "full_cplex"), (1, "benders_unified"),
            (2, "full_cplex"), (2, "benders_unified")}


def test_both_methods_use_same_unified_model_and_budget():
    cfg = dict(seconds_per_job=3600, threads_per_job=1, seed=43, relative_gap_target=.01,
               common_solver_args=list(bench.COMMON_FLAGS), algorithm_solver_args=list(bench.ALGORITHM_FLAGS),
               cg_max_columns_per_iteration=512, max_benders_iterations=1000000)
    origin = dict(id="tsl_L01_M01", instance="instances/tsl_L01_M01", dataset="tsl", map_id="M01", n_l=1)
    for job in bench.jobs_for([origin]):
        command = bench.command_for(Path("/batch"), {"protocol": cfg}, job, "/attempt")
        option = lambda flag: command[command.index(flag) + 1]
        assert option("--demand-formulation") == "preprocessed"
        assert option("--k") == str(job["K"])
        assert option("--seconds") == "3600"
        assert option("--threads") == "1"
        assert option("--gap") == "0.01"
        assert all(flag in command for flag in bench.COMMON_FLAGS)
        assert ("--topology-cuts" in command) == (job["method"] == "benders_unified")
        assert ("--od-airline-bounds" in command) == (job["method"] == "benders_unified")
        assert "--preprocessed-first-leg" not in command


def test_center_selection_is_deterministic_preserves_anchor_and_uses_both_endpoints():
    rows = []
    for center in ((0, 0), (5, 0), (0, 5), (5, 5)):
        for i in range(60):
            delta = (i - 30) * .001
            rows.append(dict(hour=12, ox=center[0] + delta, oy=center[1],
                             dx=center[0], dy=center[1] + delta))
    # These pickup points never have an endpoint-complete order in the window.
    rows.extend(dict(hour=12, ox=20 + i * .001, oy=20, dx=100, dy=100) for i in range(60))
    orders = pd.DataFrame(rows)
    result = bench.choose_centers(orders, (0, 0), count=4, half=1, min_orders=50)
    assert result == bench.choose_centers(orders, (0, 0), count=4, half=1, min_orders=50)
    assert result[0]["center"] == [0, 0]
    assert result[1]["center"] == [5, 5]
    assert len({tuple(item["center"]) for item in result}) == 4
    assert all(item["l1_noon_orders"] == 60 for item in result)
    with pytest.raises(ValueError, match="Insufficient"):
        bench.choose_centers(orders, (0, 0), count=5, half=1, min_orders=50)


def test_report_does_not_mix_single_and_double_path_counts(tmp_path, monkeypatch):
    origins = [dict(id="tsl_L01_M01", instance="instances/tsl_L01_M01", dataset="tsl", map_id="M01", n_l=1)]
    jobs = bench.jobs_for(origins)
    manifest = dict(protocol=dict(physical_instances=1, seconds_per_job=3600,
                    relative_gap_target=.01, threads_per_job=1, parallel_jobs=4), jobs=jobs)
    (tmp_path / "state.json").write_text(json.dumps({"jobs": {job["id"]: {} for job in jobs}}))
    monkeypatch.setattr(bench.base, "verify", lambda *_: manifest)
    def fake_record(batch, job, state):
        full = job["method"] == "full_cplex"
        count = 100 if job["K"] == 1 else 900
        return dict(job=job["id"], instance=job["instance"], K=job["K"], task_mode=job["task_mode"],
                    method=job["method"], upper_bound=None, lower_bound=None, gap=None,
                    all_paths=count if full else None, all_service_paths=count - 10 if full else None,
                    all_empty_paths=10 if full else None, validation_passed=False)
    monkeypatch.setattr(bench, "record", fake_record)
    bench.report(tmp_path)
    rows = json.loads((tmp_path / "analysis/comparison.json").read_text())["rows"]
    assert [row["all_paths"] for row in rows] == [100, 100, 900, 900]


def test_adapter_restores_original_runner_functions():
    original = bench.base.command_for
    with bench.adapters():
        assert bench.base.command_for is bench.command_for
    assert bench.base.command_for is original


@pytest.mark.parametrize("mode", (1, 2))
def test_audit_worker_passes_actual_mode_and_benders_identity(tmp_path, monkeypatch, mode):
    job = dict(id="job", instance="instances/map", K=mode, method="benders_unified")
    attempt = tmp_path / "runs/attempt"
    attempt.mkdir(parents=True)
    (tmp_path / "experiment_manifest.json").write_text(json.dumps({"jobs": [job]}))
    seen = []
    def fake_audit(batch, manifest, actual_job, actual_attempt):
        seen.append((actual_job["K"], actual_job["method"], actual_attempt))
        return dict(passed=True, result_directory="runs/attempt")
    monkeypatch.setattr(bench.base.existing, "audit_attempt", fake_audit)
    monkeypatch.setattr(bench.base, "_record", lambda *args: {"mode": mode})
    result = bench.base.audit_worker(tmp_path, "job", attempt)
    assert result["passed"]
    assert seen == [(mode, "benders_cplex", attempt)]
    assert json.loads((attempt / "report_row.json").read_text())["mode"] == mode


@pytest.mark.parametrize("action", ("audit-worker", "report"))
def test_child_guard_uses_frozen_unified_module(monkeypatch, action):
    captured = []
    monkeypatch.setattr(bench, "_BASE_GUARD", lambda *args, **kw: captured.append(args[3]))
    command = ["python", "-m", bench.base.__name__, action, "--batch", "/batch"]
    bench._guarded_execute("/batch", {}, "job", command, "/attempt", None)
    assert captured[0][2] == bench.MODULE
    assert command[2] == bench.base.__name__  # Caller command is not mutated.


@pytest.mark.parametrize("mode", (1, 2))
def test_checkpoint_recovery_keeps_unified_mode_and_demand_identity(tmp_path, mode):
    from src.experiments.native import save_json_result
    source = tmp_path / "recovery/all_courier"
    payload = dict(objective=4, bound=0, max_orders=mode, demand_formulation="preprocessed",
                   binary_sha256="frozen-binary", method="benders_cg", solver="CPLEX",
                   _solution={"fleet": 0}, _segments=[], _missions=[], _relocations=[])
    save_json_result(source, payload)
    (source / "input_reference.json").write_text(json.dumps({"max_orders": mode, "manifest": {}}))
    names = ("result.json", "solution.json", "path_flows.json", "input_reference.json")
    checkpoint = dict(validated=True, files={name: dict(path=str((source / name).relative_to(tmp_path)),
                      sha256=bench.base.sha256(source / name)) for name in names})
    (tmp_path / "checkpoint.json").write_text(json.dumps(checkpoint))
    recovered = bench.base.existing._recover_checkpoint(tmp_path)
    result = json.loads((recovered / "result.json").read_text())
    assert result["max_orders"] == mode
    assert result["demand_formulation"] == "preprocessed"
    assert result["binary_sha256"] == "frozen-binary"
    assert result["method"] == "benders_cg"
    assert result["recovered_from_checkpoint"]


def test_stop_checks_pid_creation_and_frozen_entrypoint(tmp_path, monkeypatch):
    import psutil
    calls = []
    (tmp_path / "dispatcher.json").write_text(json.dumps(dict(pid=1234, create_time=100)))
    class FakeProcess:
        def __init__(self, pid):
            assert pid == 1234
        def create_time(self):
            return 100
        def cmdline(self):
            return ["python", str(tmp_path / "run.py"), "run"]
        def send_signal(self, value):
            calls.append(value)
    monkeypatch.setattr(psutil, "Process", FakeProcess)
    assert bench.stop(tmp_path)["stop_requested"]
    assert calls == [bench.signal.SIGTERM]
    monkeypatch.setattr(FakeProcess, "create_time", lambda self: 101)
    with pytest.raises(RuntimeError, match="reused"):
        bench.stop(tmp_path)
    assert len(calls) == 1


def test_prepare_freezes_unified_launcher_and_all_four_jobs(tmp_path, monkeypatch):
    maps = tmp_path / "maps"
    inp = maps / "instances/tsl_L01_M01"
    provenance = maps / "provenance/tsl_L01_M01"
    inp.mkdir(parents=True)
    provenance.mkdir(parents=True)
    (inp / "manifest.json").write_text("{}")
    (provenance / "preparation.json").write_text("{}")
    item = dict(id="tsl_L01_M01", dataset="tsl", map_id="M01", n_l=1, n_h=10, n_od=25,
                hourly_demand=123, instance="instances/tsl_L01_M01", provenance="provenance/tsl_L01_M01",
                manifest_sha256=bench.base.sha256(inp / "manifest.json"))
    (maps / "preparation_manifest.json").write_text(json.dumps(dict(status="prepared_validated_not_solved", instances=[item])))
    def fake_prepare(batch, instances, **kwargs):
        assert kwargs["binary"] == bench.ROOT / "cpp/build-unified/mlnd_cpp"
        Path(batch).mkdir()
        binary = Path(batch) / "frozen/bin/mlnd_cpp"
        binary.parent.mkdir(parents=True)
        binary.write_text("fixture binary, never executed")
        frozen = dict(input_origins=[dict(id=item["id"], instance=item["instance"])], frozen_sha256={}, protocol=kwargs)
        (Path(batch) / "experiment_manifest.json").write_text(json.dumps(frozen, default=str))
    monkeypatch.setattr(bench.base, "prepare", fake_prepare)
    batch = tmp_path / "batch"
    result = bench.prepare(batch, maps=maps)
    assert result["jobs"] == 4 and not result["solver_started"]
    manifest = json.loads((batch / "experiment_manifest.json").read_text())
    assert manifest["frozen_sha256"]["run.py"] == bench.base.sha256(batch / "run.py")
    entry = (batch / "run.py").read_text()
    assert "sys.path.insert(0, str(batch / 'frozen'))" in entry
    assert f"from {bench.MODULE} import main" in entry
    assert manifest["protocol"]["common_solver_args"] == list(bench.COMMON_FLAGS)
    state = json.loads((batch / "state.json").read_text())
    assert {item["status"] for item in state["jobs"].values()} == {"pending"}
