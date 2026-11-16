"""Benchmark protocol/record checks without solves or generated standard inputs."""

import json
import sys

import pytest

from src.experiments import branch_joint_benchmark_20261022 as branch
from src.experiments import conditional_benchmark_20261022 as base
from src.experiments.storage import write_json


def manifest():
    return dict(schema="assignment-support-benchmark/1",python=sys.executable,frozen_sha256={},
        input_origins=[dict(id="L01",instance="instances/L01",manifest_sha256="identity",all_courier_cost=120,
                            size=dict(bases=1,lockers=10,orders=25))],
        protocol=dict(seconds_per_job=600,threads_per_job=1,parallel_jobs=4,seed=43,relative_gap_target=.005,
            checkpoint_seconds=60,cg_max_columns_per_iteration=512,max_benders_iterations=1000000,plain=False,
            topology_seconds=2,topology_cut_limit=1000,assignment_support_seconds=.25,
            assignment_support_cut_limit=2000,assignment_support_max_cuts=16,assignment_support_floor=1e-4,
            new_solver_args=list(branch.NEW_SOLVER_ARGS),hard_wall_seconds=720,
            resource_limits=dict(job_rss_bytes=4*1024**3,batch_rss_bytes=8*1024**3,
                system_available_floor_bytes=2*1024**3,sample_seconds=.5,terminate_grace_seconds=5)))


def test_exactly_four_jobs_append_reviewed_new_options_only(tmp_path):
    cfg = manifest()
    origins = [dict(id=f"L{n:02}",instance=f"instances/L{n:02}") for n in (1,2)]
    jobs = branch.jobs_for(origins)
    assert len(jobs)==len({job["id"] for job in jobs})==4
    for job in jobs:
        assert job["branch_joint_cuts"] and job["assignment_support_cuts"] and job["tight_airline_bounds"] and job["topology_cuts"]
        previous = branch.assignment.command_for(tmp_path,cfg,job,tmp_path/"attempt")
        new = branch.command_for(tmp_path,cfg,job,tmp_path/"attempt")
        assert new==previous+cfg["protocol"]["new_solver_args"]
        for key,value in (("--seconds","600"),("--threads","1"),("--topology-seconds","2"),
                          ("--assignment-support-seconds","0.25"),("--assignment-support-floor","0.0001")):
            assert new[new.index(key)+1]==value
        assert "--double-support-cuts" in new and "--joint-base-cost-cuts" in new
        assert new[new.index("--branch-joint-preprocessing-seconds")+1]=="2"


def test_new_options_cannot_override_budget_or_legacy_algorithm(tmp_path):
    with pytest.raises(ValueError,match="reviewed"):
        branch.prepare(tmp_path/"new",[],new_solver_args=[])
    with pytest.raises(ValueError,match="reviewed"):
        branch.prepare(tmp_path/"new",[],new_solver_args="--branch-support-cuts")
    with pytest.raises(ValueError,match="override"):
        branch.prepare(tmp_path/"new",[],new_solver_args=["--seconds","60"])


def test_adapter_restoration_and_child_module_routing(monkeypatch):
    previous = {key:getattr(base,key) for key in ("command_for","_record","guarded_execute","report")}
    with pytest.raises(RuntimeError):
        with branch.adapters():
            assert base._record is branch._record
            raise RuntimeError("test")
    assert all(getattr(base,key) is value for key,value in previous.items())
    seen=[]
    monkeypatch.setattr(branch,"_BASE_GUARD",lambda *a,**kw:seen.append(a[3]))
    command=[sys.executable,"-m",base.__name__,"audit-worker"]
    branch._guarded_execute(None,None,None,command,None,None)
    assert seen[-1][2]==branch.MODULE and command[2]==base.__name__


def test_record_compares_direct_prior_and_retains_raw_applied_roots(tmp_path,monkeypatch):
    job=branch.jobs_for(manifest()["input_origins"])[0]
    initial=dict(upper_bound=100,lower_bound=92,continuous_root_lower_bound=90.00001,
        continuous_root_applied_lower_bound=90,integer_master_best_lower_bound=92,
        all_paths=None,previous_strengthened_lower_bound=88,change_from_previous_lower_bound=4)
    monkeypatch.setattr(branch,"_ASSIGNMENT_RECORD",lambda *_:dict(initial))
    write_json(tmp_path/"attempt/result.json",dict(branch_joint_summary=dict(
        enabled=True,double_cuts=7,joint_cuts=3,seconds=.1,preprocessing_seconds=.2),
        branch_joint_cut_records=[dict(kind="double_branch")]*7+[dict(kind="joint_bases")]*3,
        branch_joint_preprocessing=dict(complete_domain=True)))
    write_json(tmp_path/"reference/validated_comparison.json",dict(rows=[dict(
        instance="instances/L01",method="benders_lifted",job="L01_K2_lifted_assignment_support",
        lower_bound=90,upper_bound=101,all_paths=12,all_service_paths=10,all_empty_paths=2,
        full_path_count_source="older_full_reference")]))
    row=branch._record(tmp_path,job,dict(directory="attempt"))
    assert row["previous_assignment_lower_bound"]==90 and row["change_from_previous_lower_bound"]==2
    assert "previous_strengthened_lower_bound" not in row
    assert row["global_lb_minus_applied_root"]==row["integer_lb_minus_applied_root"]==2
    assert row["global_lb_exceeds_applied_root"]
    assert row["branch_joint_double_cuts"]==7 and row["branch_joint_joint_cuts"]==3
    assert row["branch_joint_cut_records_record_count"]==10
    assert row["branch_joint_double_cut_records"]==7 and row["branch_joint_joint_cut_records"]==3
    assert json.loads(row["branch_joint_preprocessing_json"])["complete_domain"]
    assert row["all_paths"]==12 and row["full_path_count_source"]=="reference/older_full_reference"


def test_prepare_copies_readonly_reference_and_inherits_original_protocol(tmp_path,monkeypatch):
    old=tmp_path/"old"
    cfg=manifest()
    source=tmp_path/"inputs/L01"
    write_json(source/"manifest.json",dict(existing=True))
    cfg["input_origins"][0]["manifest_sha256"]=base.sha256(source/"manifest.json")
    for name,value in (("experiment_manifest.json",cfg),("analysis/comparison.json",dict(rows=[])),
                       ("reference/validated_comparison.json",dict(rows=[]))):
        write_json(old/name,value)
    trace=old/"progress.jsonl"
    trace.write_text("{}\n")
    before={str(path.relative_to(old)):base.sha256(path) for path in old.rglob("*") if path.is_file()}
    monkeypatch.setattr(branch,"_reference_snapshot",lambda _: (cfg,[],[("old_job",trace)]))
    def freeze(batch,inputs,**kwargs):
        assert inputs==[source] and kwargs["seconds"]==600 and kwargs["threads"]==1 and kwargs["parallel"]==4
        assert kwargs["job_rss_gib"]==4 and kwargs["batch_rss_gib"]==8 and kwargs["hard_wall_seconds"]==720
        batch.mkdir()
        write_json(batch/"experiment_manifest.json",cfg)
    monkeypatch.setattr(base,"prepare",freeze)
    new=tmp_path/"new"
    result=branch.prepare(new,[source],reference=old,new_solver_args=["--branch-support-cuts","--joint-base-cuts"])
    saved=json.loads((new/"experiment_manifest.json").read_text())
    assert result["jobs"]==2 and not result["solver_started"] and result["reference_unchanged"]
    assert saved["schema"]=="branch-joint-benchmark/1"
    assert saved["protocol"]["assignment_support_floor"]==1e-4
    assert saved["protocol"]["new_solver_args"]==["--branch-support-cuts","--joint-base-cuts"]
    assert saved["protocol"]["cross_batch_single_seed"]
    assert branch.MODULE in (new/"run.py").read_text()
    assert saved["frozen_sha256"]["reference/progress/old_job/progress.jsonl"]==base.sha256(trace)
    assert before=={str(path.relative_to(old)):base.sha256(path) for path in old.rglob("*") if path.is_file()}


def test_prepare_rejects_reference_parameter_drift_before_freezer(tmp_path,monkeypatch):
    cfg=manifest()
    cfg["protocol"]["assignment_support_floor"]=0
    monkeypatch.setattr(branch,"_reference_snapshot",lambda _: (cfg,[],[]))
    monkeypatch.setattr(base,"prepare",lambda *_a,**_kw:pytest.fail("must reject first"))
    with pytest.raises(ValueError,match="protocol"):
        branch.prepare(tmp_path/"new",[],reference=tmp_path,new_solver_args=["--branch-support-cuts"])


def test_pending_report_preserves_missing_certified_values(tmp_path):
    cfg=manifest()
    cfg["jobs"]=branch.jobs_for(cfg["input_origins"])
    write_json(tmp_path/"experiment_manifest.json",cfg)
    write_json(tmp_path/"state.json",dict(jobs={job["id"]:dict(status="pending",attempts=[]) for job in cfg["jobs"]}))
    write_json(tmp_path/"reference/validated_comparison.json",dict(rows=[]))
    with branch.adapters():
        outcome=branch._report(tmp_path)
    assert not outcome["complete"] and outcome["validated"]==0
    content=(tmp_path/"analysis/REPORT.md").read_text()
    assert "尚未完成" in content and "跨批次单种子" in content
    rows=json.loads((tmp_path/"analysis/comparison.json").read_text())["rows"]
    assert all(row["lower_bound"] is None and row["global_lb_exceeds_applied_root"] is None for row in rows)


def test_display_suppresses_negative_zero_without_changing_raw_values():
    assert branch._difference_display(-1e-12)=="0.000"
    assert branch._difference_display(None)=="—"
    assert branch._difference_display(.25)=="0.250000"
