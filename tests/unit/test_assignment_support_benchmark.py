"""Compare new cuts to immutable, validated prior runs without performance solves."""

import json
import sys

import pytest

from src.experiments import assignment_support_benchmark_20261022 as support
from src.experiments import conditional_benchmark_20261022 as base
from src.experiments.storage import write_json


def manifest():
    return dict(python=sys.executable, input_origins=[dict(id="L01", instance="instances/L01",
        manifest_sha256="identity", all_courier_cost=120)], frozen_sha256={},
        protocol=dict(seconds_per_job=600, threads_per_job=1, parallel_jobs=4, seed=43,
        relative_gap_target=.005, checkpoint_seconds=60, cg_max_columns_per_iteration=512,
        max_benders_iterations=1000000, plain=False))


def test_four_jobs_only_add_support_to_existing_strengthened_setting(tmp_path):
    cfg = manifest()
    origins = [dict(id=f"L{n:02}", instance=f"instances/L{n:02}") for n in (1, 2)]
    jobs = support.jobs_for(origins)
    assert len(jobs) == len({job["id"] for job in jobs}) == 4
    assert {job["method"] for job in jobs} == {"benders_lifted", "benders_conditional"}
    for job in jobs:
        assert job["tight_airline_bounds"] and job["topology_cuts"] and job["assignment_support_cuts"]
        legacy = base.command_for(tmp_path, cfg, job, tmp_path / "attempt")
        assert support.command_for(tmp_path, cfg, job, tmp_path / "attempt") == legacy + [
            "--tight-airline-bounds", "--topology-cuts", "--topology-seconds", "2",
            "--topology-cut-limit", "1000", "--assignment-support-cuts"]


def test_cut_limits_are_explicit_when_protocol_defines_them(tmp_path):
    cfg = manifest()
    cfg["protocol"].update(assignment_support_seconds=.25, assignment_support_cut_limit=2000,
                           assignment_support_max_cuts=16, assignment_support_floor=1e-4)
    command = support.command_for(tmp_path, cfg, support.jobs_for(cfg["input_origins"])[0], tmp_path)
    assert command[-8:] == ["--assignment-support-seconds", "0.25", "--assignment-support-cut-limit", "2000",
                            "--assignment-support-max-cuts", "16", "--assignment-support-floor", "0.0001"]
    assert command[command.index("--seconds")+1] == "600"
    assert command[command.index("--threads")+1] == "1"


def test_adapters_restore_and_route_child_audits(monkeypatch):
    before = {key:getattr(base,key) for key in ("command_for", "_record", "guarded_execute", "report")}
    with pytest.raises(RuntimeError):
        with support.adapters():
            assert base._record is support._record
            raise RuntimeError("test")
    assert all(getattr(base,key) is value for key,value in before.items())
    seen = []
    monkeypatch.setattr(support, "_BASE_GUARD", lambda *args, **kwargs: seen.append(args[3]))
    original = [sys.executable, "-m", base.__name__, "audit-worker"]
    support._guarded_execute(None,None,None,original,None,None)
    assert seen[-1][2] == support.MODULE and original[2] == base.__name__


def test_record_reports_certified_root_improvement_and_previous_delta(tmp_path, monkeypatch):
    job = support.jobs_for(manifest()["input_origins"])[0]
    row = dict(upper_bound=100, lower_bound=92, continuous_root_lower_bound=90,
        integer_master_best_lower_bound=92, all_paths=None)
    monkeypatch.setattr(support, "_TOPOLOGY_RECORD", lambda *_:dict(row))
    write_json(tmp_path / "attempt/result.json", dict(assignment_support_summary=dict(
        enabled=True, cuts=3, attempts=9, skipped_integer_subproblems=2, seconds=.2),
        assignment_support_cut_records=[{}, {}, {}]))
    write_json(tmp_path / "reference/validated_comparison.json", dict(rows=[dict(
        instance="instances/L01", variant="lifted_strengthened", upper_bound=101, lower_bound=90,
        gap=11/101, all_paths=12, all_service_paths=10, all_empty_paths=2,
        full_path_count_source="L01_K2_full_cplex_reference")]))
    result = support._record(tmp_path, job, dict(directory="attempt"))
    assert result["assignment_support_cuts_enabled"]
    assert result["assignment_support_cuts"] == 3
    assert result["assignment_support_cut_records_record_count"] == 3
    assert result["global_lb_exceeds_continuous_root"]
    assert result["global_lb_minus_continuous_root"] == result["change_from_previous_lower_bound"] == 2
    assert result["change_from_previous_upper_bound"] == -1
    assert result["all_paths"] == 12 and result["full_path_count_source"].startswith("reference/")
    row["lower_bound"] = 90+1e-7
    assert not support._record(tmp_path, job, dict(directory="attempt"))["global_lb_exceeds_continuous_root"]
    row["continuous_root_lower_bound"] = None
    assert support._record(tmp_path, job, dict(directory="attempt"))["global_lb_exceeds_continuous_root"] is None


def test_prepare_refuses_mismatched_budget_before_freezing(tmp_path, monkeypatch):
    monkeypatch.setattr(support, "_reference_snapshot", lambda _: (manifest(), [], []))
    monkeypatch.setattr(support.topology, "prepare", lambda *_a, **_k:pytest.fail("must reject before freezing"))
    with pytest.raises(ValueError, match="matching seconds"):
        support.prepare(tmp_path / "new", [], reference=tmp_path, seconds=180)


def test_prepare_copies_reference_without_altering_it(tmp_path, monkeypatch):
    old = tmp_path / "old"
    old.mkdir()
    cfg = manifest()
    source = tmp_path / "inputs/L01"
    write_json(source / "manifest.json", {"existing":True})
    cfg["input_origins"][0]["manifest_sha256"] = base.sha256(source / "manifest.json")
    write_json(old / "experiment_manifest.json", cfg)
    write_json(old / "analysis/comparison.json", dict(rows=[]))
    trace = old / "progress.jsonl"
    trace.write_text("{}\n")
    original = {str(path.relative_to(old)):base.sha256(path) for path in old.rglob("*") if path.is_file()}
    monkeypatch.setattr(support, "_reference_snapshot", lambda _: (cfg, [], [("old_job",trace)]))
    def freeze(batch, inputs, **kwargs):
        assert kwargs["seconds"] == 600 and kwargs["parallel"] == 4 and kwargs["threads"] == 1
        assert inputs == [source]
        batch.mkdir()
        write_json(batch / "experiment_manifest.json", cfg)
    monkeypatch.setattr(support.topology, "prepare", freeze)
    batch = tmp_path / "new"
    result = support.prepare(batch, [source], reference=old, assignment_support_seconds=.25)
    saved = json.loads((batch / "experiment_manifest.json").read_text())
    assert result["jobs"] == 2 and result["reference_unchanged"] and not result["solver_started"]
    assert saved["schema"] == "assignment-support-benchmark/1"
    assert saved["protocol"]["assignment_support_seconds"] == .25
    assert saved["protocol"]["cross_batch_single_seed"]
    assert support.MODULE in (batch / "run.py").read_text()
    assert saved["frozen_sha256"]["reference/progress/old_job/progress.jsonl"] == base.sha256(trace)
    assert original == {str(path.relative_to(old)):base.sha256(path) for path in old.rglob("*") if path.is_file()}


def test_pending_report_explicitly_preserves_missing_bounds(tmp_path):
    cfg = manifest()
    cfg["jobs"] = support.jobs_for(cfg["input_origins"])
    write_json(tmp_path / "experiment_manifest.json", cfg)
    write_json(tmp_path / "state.json", dict(jobs={job["id"]:dict(status="pending",attempts=[])
                                                 for job in cfg["jobs"]}))
    write_json(tmp_path / "reference/validated_comparison.json", dict(rows=[]))
    with support.adapters():
        outcome = support._report(tmp_path)
    assert not outcome["complete"] and outcome["validated"] == 0
    content = (tmp_path / "analysis/REPORT.md").read_text()
    assert "尚未完成" in content and "跨批次单种子" in content and "是否越过连续根界" in content
    rows = json.loads((tmp_path / "analysis/comparison.json").read_text())["rows"]
    assert all(row["lower_bound"] is None and row["global_lb_exceeds_continuous_root"] is None for row in rows)
