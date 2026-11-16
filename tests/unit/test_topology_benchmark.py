"""Ablation protocol and reporting contracts; no performance solves or new inputs."""

import json
import sys

import pytest

from src.experiments import conditional_benchmark_20261022 as base
from src.experiments import topology_benchmark_20261022 as topology
from src.experiments.storage import write_json


def manifest():
    return dict(python=sys.executable, input_origins=[dict(id="L01", instance="instances/L01",
        all_courier_cost=120)], protocol=dict(seconds_per_job=600, threads_per_job=1,
        seed=43, relative_gap_target=.005, checkpoint_seconds=60,
        cg_max_columns_per_iteration=512, max_benders_iterations=1000000, plain=False))


def test_two_by_two_ablation_and_reference_keep_underlying_methods():
    origins = [dict(id=f"L{n:02}", instance=f"instances/L{n:02}") for n in (1, 2)]
    jobs = topology.jobs_for(origins)
    assert len(jobs) == 10 and len({job["id"] for job in jobs}) == 10
    assert len(topology.jobs_for(origins, full_reference=False)) == 8
    for origin in origins:
        group = [job for job in jobs if job["instance"] == origin["instance"]]
        assert sum(job["tight_airline_bounds"] and job["topology_cuts"] for job in group) == 2
        assert {job["method"] for job in group} == {"benders_lifted", "benders_conditional", "full_cplex"}


def test_same_binary_baseline_is_exact_legacy_command_and_only_two_flags_change(tmp_path):
    cfg = manifest()
    jobs = topology.jobs_for(cfg["input_origins"])
    for job in jobs:
        legacy = base.command_for(tmp_path, cfg, job, tmp_path / "attempt")
        command = topology.command_for(tmp_path, cfg, job, tmp_path / "attempt")
        expected = legacy + (["--tight-airline-bounds", "--topology-cuts", "--topology-seconds", "2",
                              "--topology-cut-limit", "1000"] if job["topology_cuts"] else [])
        assert command == expected
        assert command[0] == str(tmp_path / "frozen/bin/mlnd_cpp")
        assert command[command.index("--seconds")+1] == "600"
        assert command[command.index("--threads")+1] == "1"


def test_adapters_restore_original_runner_even_after_failure():
    before = {name: getattr(base, name) for name in ("command_for", "_record", "guarded_execute", "report")}
    with pytest.raises(RuntimeError, match="test"):
        with topology.adapters():
            assert base.command_for is topology.command_for
            assert base._record is topology._record
            raise RuntimeError("test")
    assert all(getattr(base, key) is value for key, value in before.items())


def test_only_internal_audit_and_report_module_is_rerouted(monkeypatch):
    seen = []
    monkeypatch.setattr(topology, "_BASE_GUARD", lambda *args, **kwargs: seen.append(args[3]))
    internal = [sys.executable, "-m", "src.experiments.conditional_benchmark_20261022", "audit-worker"]
    topology._guarded_execute(None, None, None, internal, None, None)
    assert seen[-1][2] == topology.MODULE
    native = ["/frozen/bin/mlnd_cpp", "--method", "benders_cg"]
    topology._guarded_execute(None, None, None, native, None, None)
    assert seen[-1] == native and internal[2] != topology.MODULE


def test_record_keeps_invalid_root_unreported_and_distinguishes_candidate_ratios(tmp_path, monkeypatch):
    cfg = manifest()
    write_json(tmp_path / "experiment_manifest.json", cfg)
    job = topology.jobs_for(cfg["input_origins"])[3]
    write_json(tmp_path / "attempt/result.json", dict(
        continuous_root=dict(bound_valid=False, certified_lower_bound=88, termination_reason="dual_unavailable"),
        master_solves=[dict(certified_bound=40), dict(certified_bound=50), dict(certified_bound=0)],
        acceleration_statistics=dict(integer_cg_calls=4, integer_master_solves=9, all_courier_candidates=1,
            topology_attempts=9, topology_cuts=3, topology_skipped_socp=5),
        topology_summary=dict(enabled=True, attempts=9, cuts=3, skipped_integer_subproblems=5,
                              zero_service_certificates=6, seconds=.123),
        topology_cut_records=[{}, {}, {}], topology_attempts=[{}]*9,
        accelerations=dict(topology_cuts=True, tight_airline_bounds=True)))
    monkeypatch.setattr(topology, "_BASE_RECORD", lambda *_: dict(upper_bound=100, lower_bound=50))
    row = topology._record(tmp_path, job, dict(directory="attempt"))
    assert row["continuous_root_lower_bound"] is None
    assert row["integer_master_best_lower_bound"] == 50
    assert row["integer_master_last_reported_lower_bound"] == 0
    assert row["all_courier_candidate_ratio"] == .25
    assert row["upper_bound_ratio_to_all_courier"] == 100/120
    assert row["topology_skipped_socp"] == 5
    assert row["topology_skipped_integer_subproblems"] == 5
    assert row["topology_zero_service_certificates"] == 6
    assert row["topology_cut_records_record_count"] == 3
    assert row["topology_cuts_enabled"]


def test_prepare_uses_existing_freezer_and_own_entrypoint(tmp_path, monkeypatch):
    def freeze(batch, inputs, **kwargs):
        assert inputs == ["existing/L01"]
        assert kwargs["seconds"] == 600 and kwargs["parallel"] == 4
        assert kwargs["job_rss_gib"] == 4 and kwargs["hard_wall_seconds"] == 720
        batch.mkdir()
        write_json(batch / "experiment_manifest.json", manifest())
    monkeypatch.setattr(base, "prepare", freeze)
    monkeypatch.setattr(topology, "load_standard", lambda *_args, **_kwargs: dict(
        demand=[2, 3], ground_cost=[10, 20], n_l=1, n_h=10, n_od=2))
    batch = tmp_path / "batch"
    result = topology.prepare(batch, ["existing/L01"])
    saved = json.loads((batch / "experiment_manifest.json").read_text())
    assert result["jobs"] == 5 and not result["solver_started"]
    assert saved["input_origins"][0]["all_courier_cost"] == 80
    assert saved["schema"] == "topology-benchmark/1"
    assert topology.MODULE in (batch / "run.py").read_text()
    assert len(json.loads((batch / "state.json").read_text())["jobs"]) == 5


def test_incomplete_report_handles_pending_jobs_without_loading_solver_results(tmp_path):
    cfg = manifest()
    cfg["frozen_sha256"] = {}
    cfg["protocol"].update(parallel_jobs=4, hard_wall_seconds=720)
    cfg["input_origins"][0]["manifest_sha256"] = "test_identity"
    cfg["jobs"] = topology.jobs_for(cfg["input_origins"])
    write_json(tmp_path / "experiment_manifest.json", cfg)
    write_json(tmp_path / "state.json", dict(jobs={job["id"]: dict(status="pending", attempts=[])
                                                for job in cfg["jobs"]}))
    with topology.adapters():
        result = topology._report(tmp_path)
    assert not result["complete"] and result["validated"] == 0
    markdown = (tmp_path / "analysis/REPORT.md").read_text()
    assert "尚未完成" in markdown and "600" in markdown
    assert "topology_skipped_integer_subproblems" in markdown
    data = json.loads((tmp_path / "analysis/comparison.json").read_text())
    assert len(data["rows"]) == 5 and all(row["upper_bound"] is None for row in data["rows"])
