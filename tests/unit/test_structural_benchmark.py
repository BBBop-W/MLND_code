"""Frozen algorithm identity, scope and comparison checks without optimization."""

import json
import sys

import pytest

from src.experiments import conditional_benchmark_20261022 as base
from src.experiments import structural_benchmark as bench
from src.experiments.storage import write_json
from src.reporting.structural_comparison import histories, report, structural_point_violations


def manifest():
    return dict(python=sys.executable, frozen_sha256={},
                binaries=dict(old=dict(path="frozen/bin/old", sha256="old_hash"), new=dict(path="frozen/bin/new", sha256="new_hash")),
                input_origins=[dict(id="L02", instance="instances/L02")],
                protocol=dict(seconds_per_job=600, threads_per_job=1, parallel_jobs=3, seed=43,
                              relative_gap_target=.005, checkpoint_seconds=60, cg_max_columns_per_iteration=512,
                              max_benders_iterations=1000000, plain=False, common_solver_args=list(bench.COMMON_SOLVER_ARGS)))


def fake_result(batch, job, *, wrong=None):
    active = job["structural_strengthening"]
    result = dict(max_orders=job["K"], solver="CPLEX", method="benders_cg",
                  demand_formulation=job["method"].removeprefix("benders_"), binary_sha256=f"{job['version']}_hash",
                  objective=100, bound=91 if active else 90, bound_scope="global", wall_before_output=10,
                  timings=dict(master_build=1, master_optimize=4, subproblem_wall=3, subproblem=dict(pricing=1)),
                  continuous_root=dict(bound_valid=True, certified_lower_bound=85),
                  root_diagnostics=dict(termination_reason="round_limit"), resource_lp_cuts=[dict(value=5)])
    if active:
        result.update({flag: True for flag in bench.FLAGS})
    if wrong == "k":
        result["max_orders"] = 3 - job["K"]
    elif wrong == "binary":
        result["binary_sha256"] = "wrong"
    elif wrong == "flag":
        result[bench.FLAGS[0]] = not active
    elif wrong == "formulation":
        result["demand_formulation"] = "conditional" if result["demand_formulation"] == "lifted" else "lifted"
    directory = batch / "runs" / job["id"]
    for name, value in (("result.json", result), ("solution.json", {}), ("path_flows.json", {})):
        write_json(directory / name, value)
    validation = dict(passed=True, manifest_sha256=base.sha256(batch / job["instance"] / "manifest.json"))
    for name, key in (("result.json", "result_sha256"), ("solution.json", "solution_sha256"), ("path_flows.json", "path_flows_sha256")):
        validation[key] = base.sha256(directory / name)
    if wrong == "stale":
        validation["result_sha256"] = "stale"
    write_json(directory / "native_validation.json", validation)
    return dict(status="completed", directory=str(directory.relative_to(batch)))


def test_eight_distinct_jobs_cover_two_formulations_and_two_implementations():
    jobs = bench.jobs_for(manifest()["input_origins"])
    assert len(jobs) == len({j["id"] for j in jobs}) == 8
    assert {(j["K"], j["method"], j["version"]) for j in jobs} == {(k, method, version) for k in (1, 2) for method in ("benders_lifted", "benders_conditional") for version in ("old", "new")}


def test_commands_route_old_and_new_binaries_keep_common_budget(tmp_path):
    cfg = manifest()
    for job in bench.jobs_for(cfg["input_origins"]):
        cmd = bench.command_for(tmp_path, cfg, job, tmp_path / "attempt")
        assert cmd[0] == str(tmp_path / cfg["binaries"][job["version"]]["path"])
        assert ("--structural-strengthening" in cmd) is job["structural_strengthening"]
        for flag, value in (("--seconds", "600"), ("--k", str(job["K"])), ("--threads", "1"), ("--seed", "43"), ("--cg-max-columns", "512")):
            assert cmd[cmd.index(flag) + 1] == value
        assert "--od-airline-bounds" not in cmd and "--start" not in cmd and "--continuous-bound-file" not in cmd


@pytest.mark.parametrize("index", range(8))
def test_records_validate_either_k_form_and_binary(tmp_path, index):
    cfg = manifest()
    job = bench.jobs_for(cfg["input_origins"])[index]
    write_json(tmp_path / job["instance"] / "manifest.json", {})
    row = bench.record(tmp_path, cfg, job, fake_result(tmp_path, job))
    assert row["validation_passed"] and row["binary_identity_verified"]
    assert row["K"] == job["K"] and row["upper_bound"] == 100
    assert all(row[flag + "_enabled"] is job["structural_strengthening"] for flag in bench.FLAGS)


@pytest.mark.parametrize("wrong", ("k", "binary", "flag", "formulation", "stale"))
def test_mismatched_identity_cannot_enter_bounds(tmp_path, wrong):
    cfg = manifest()
    job = bench.jobs_for(cfg["input_origins"])[0]
    write_json(tmp_path / job["instance"] / "manifest.json", {})
    row = bench.record(tmp_path, cfg, job, fake_result(tmp_path, job, wrong=wrong))
    assert not row["validation_passed"]
    assert row["upper_bound"] is row["lower_bound"] is row["gap"] is None


def test_four_paired_changes_and_disjoint_times(tmp_path):
    cfg = manifest()
    cfg["jobs"] = bench.jobs_for(cfg["input_origins"])
    write_json(tmp_path / "instances/L02/manifest.json", {})
    states = {job["id"]: fake_result(tmp_path, job) for job in cfg["jobs"]}
    write_json(tmp_path / "experiment_manifest.json", cfg)
    write_json(tmp_path / "state.json", dict(jobs=states))
    outcome = report(tmp_path, draw=False)
    assert outcome["validated"] == 8
    data = json.loads((tmp_path / "analysis/comparison.json").read_text())
    assert len(data["pairs"]) == 4
    assert all(p["new_minus_old_lower_bound"] == 1 and abs(p["new_minus_old_gap"] + .01) < 1e-12 for p in data["pairs"])
    assert all(r["exclusive_phase_sum_seconds"] == 10 for r in data["rows"])
    states[cfg["jobs"][0]["id"]]["status"] = "failed"
    write_json(tmp_path / "state.json", dict(jobs=states))
    report(tmp_path, draw=False)
    pairs = json.loads((tmp_path / "analysis/comparison.json").read_text())["pairs"]
    assert sum(p["controlled_pair_available"] for p in pairs) == 3
    assert pairs[0]["new_minus_old_lower_bound"] is None


def test_adapters_restore_base_on_exception():
    keys = ("command_for", "_record", "guarded_execute", "audit_worker", "report")
    original = {key: getattr(base, key) for key in keys}
    with pytest.raises(RuntimeError):
        with bench.adapters():
            assert base.command_for is bench.command_for
            raise RuntimeError("test")
    assert all(getattr(base, key) is value for key, value in original.items())


def test_root_ablation_plan_is_separate_and_starts_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(bench, "load_standard", lambda *_args, **_kwargs: {})
    instance = tmp_path / "source"
    write_json(instance / "manifest.json", {})
    probe = tmp_path / "probe"
    probe.write_text("test-only executable placeholder")
    output = tmp_path / "root-plan"
    result = bench.prepare_root_ablations(output, instance, probe=probe)
    assert result["jobs"] == 8 and result["solver_started"] is False
    plan = base.read(output / "manifest.json")
    assert plan["seconds_per_job"] == 180 and not plan["root_certificates_supplied_to_production"]
    assert {job["variant"] for job in plan["jobs"]} == {"baseline", "joint", "perspective", "all"}
    for job in plan["jobs"]:
        assert job["command"][3] == "180"
        assert not (output / "runs").exists()


def test_preparation_freezes_old_binary_identity_and_starts_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(bench, "load_standard", lambda *_args, **_kwargs: dict(demand=[2], ground_cost=[10], n_l=2, n_h=20, n_od=50))
    old = tmp_path / "old_binary"
    old.write_text("old executable fixture")
    freeze = tmp_path / "baseline_freeze.json"
    write_json(freeze, dict(frozen_binary=str(old), binary_sha256=base.sha256(old), source_hashes={"cpp/src/main.cpp": "old_source"}))

    def fake_prepare(batch, _instances, **_kwargs):
        (batch / "frozen/bin").mkdir(parents=True)
        (batch / "frozen/bin/mlnd_cpp").write_text("new executable fixture")
        cfg = manifest()
        cfg["frozen_sha256"]["frozen/bin/mlnd_cpp"] = base.sha256(batch / "frozen/bin/mlnd_cpp")
        write_json(batch / "experiment_manifest.json", cfg)

    monkeypatch.setattr(base, "prepare", fake_prepare)
    target = tmp_path / "new_batch"
    result = bench.prepare(target, [tmp_path / "existing_input"], baseline_freeze=freeze)
    assert result["jobs"] == 8 and result["solver_started"] is False
    frozen = base.read(target / "experiment_manifest.json")
    assert frozen["binaries"]["old"]["sha256"] == base.sha256(old)
    assert len(frozen["jobs"]) == 8 and frozen["protocol"]["root_initialization"].startswith("inside each job")
    assert all(item["status"] == "pending" for item in base.read(target / "state.json")["jobs"].values())
    assert not (target / "runs").exists()


def test_nested_structural_metadata_is_read_without_guessing_missing_new_flags():
    for flag in bench.FLAGS:
        assert bench.enabled({"structural_strengthening": {flag: True}}, flag) is True
        assert bench.enabled({"accelerations": {flag: False}}, flag) is False
        assert bench.enabled({}, flag) is None


def test_wrong_identity_suppresses_root_bound_and_progress_curve(tmp_path):
    cfg = manifest()
    job = bench.jobs_for(cfg["input_origins"])[1]
    write_json(tmp_path / job["instance"] / "manifest.json", {})
    state = fake_result(tmp_path, job, wrong="binary")
    row = bench.record(tmp_path, cfg, job, state)
    source = tmp_path / state["directory"]
    (source / "progress.jsonl").write_text(json.dumps(dict(elapsed_seconds=3, bound_scope="global", lower_bound=99))+"\n")
    assert row["continuous_root_certified_lower_bound"] is None
    assert histories(source, row) == []


def test_identity_audit_failure_marks_scheduler_artifact_failed(tmp_path, monkeypatch):
    def fake_audit(*_args):
        write_json(tmp_path / "report_row.json", dict(validation_passed=False))
        return dict(passed=True, errors=[])
    monkeypatch.setattr(bench, "_BASE_AUDIT", fake_audit)
    audit = bench.audit_worker(tmp_path, "job", tmp_path)
    assert not audit["passed"] and not audit["method_and_binary_identity_verified"]
    assert base.read(tmp_path / "artifact_audit.json") == audit


def test_structural_row_count_includes_both_shared_rho_rows(tmp_path):
    cfg = manifest()
    cfg["input_origins"][0]["size"] = dict(bases=2, lockers=20, orders=50)
    job = bench.jobs_for(cfg["input_origins"])[1]
    write_json(tmp_path / job["instance"] / "manifest.json", {})
    row = bench.record(tmp_path, cfg, job, fake_result(tmp_path, job))
    assert row["theoretical_structural_master_added_rows"] == 100
    assert row["theoretical_structural_subproblem_added_rows"] == 304
    assert row["theoretical_structural_added_variables"] == 0


def test_saved_point_violation_audit_uses_shared_availability_and_incidence_once():
    import numpy as np
    data = dict(n_l=2,n_h=0,n_od=1,slot_counts=np.empty((0,1)))
    result = dict(primal_valid=True,design=[.5,1,.5,.5,0,0],
                  operating_point=dict(v_order="r_then_departure_then_return",rho=[.5,0],idle=[1,0],v=[.5,.5,0,0]))
    summary, details = structural_point_violations(result,data)
    assert summary["available"] and len(details) == 8
    assert summary["assignment_incident_clique_count"] == 1
    assert summary["assignment_incident_clique_max"] == .5
    assert summary["shared_rho_executed_count"] == 1
    assert summary["shared_rho_executed_max"] == .5
    assert summary["shared_rho_unexecuted_count"] == 0
    assert summary["queue_perspective_cone_max"] == .25
    result["design"][0] = 1
    result["operating_point"]["v"] = [0,0,0,0]
    summary, _ = structural_point_violations(result,data)
    assert summary["shared_rho_executed_count"] == 0
    assert summary["shared_rho_unexecuted_count"] == 1
    assert summary["shared_rho_unexecuted_max"] == .5
    result["operating_point"]["v_order"] = "unknown"
    unavailable, details = structural_point_violations(result,data)
    assert not unavailable["available"] and not details


def test_root_dispatch_preserves_evidence_and_does_not_invent_integer_ub(tmp_path, monkeypatch):
    monkeypatch.setattr(bench, "load_standard", lambda *_args, **_kwargs: {})
    instance = tmp_path / "source"
    write_json(instance / "manifest.json", {})
    write_json(instance / "model.json", {})
    probe = tmp_path / "probe"
    probe.write_text("test-only executable placeholder")
    output = tmp_path / "root-plan"
    bench.prepare_root_ablations(output, instance, probe=probe)
    calls = []

    def fake_execute(command, _cwd, directory, cfg, _cancel):
        calls.append(command)
        assert cfg["seconds_per_job"] == 180 and cfg["watchdog_grace_seconds"] == 30
        result = dict(K=int(command[2]), input_manifest={}, input_model_sha256=base.sha256(instance / "model.json"),
                      method="full_continuous_cg", scope="full_path_continuous_relaxation", objective_includes_fixed_costs=True,
                      bound_valid=True, primal_valid=True, pricing_complete=True, certified_lower_bound=89,
                      continuous_objective=90, end_to_end_seconds=5, termination_reason="pricing_complete", design=[0, .5, 1])
        for feature in bench.FLAGS:
            flag = "--" + feature.replace("_", "-")
            result[feature] = command[command.index(flag) + 1] == "true"
        if any(result[feature] for feature in bench.FLAGS):
            result["scope"] = "structurally_strengthened_continuous_relaxation"
        write_json(Path(command[4]), result)
        execution = dict(returncode=0, status="exited", external_wall_seconds=5.1)
        write_json(directory / "execution.json", execution)
        return execution

    from pathlib import Path
    monkeypatch.setattr(base.existing, "execute", fake_execute)
    summary = bench.run_root_ablations(output)
    assert len(calls) == 8 and summary["verified"] == 8
    rows = base.read(output / "analysis/comparison.json")["rows"]
    assert all(not row["is_integer_upper_bound"] and row["fractional_strategic_variables"] == 1 for row in rows)
    assert all(abs(row["root_gap"] - 1 / 90) < 1e-12 for row in rows)
    with pytest.raises(FileExistsError):
        bench.run_root_ablations(output)
