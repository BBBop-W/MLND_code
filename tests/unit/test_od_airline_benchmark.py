"""Comparison identity and validity checks without launching a solver."""

import json
import sys

import pytest

from src.experiments import conditional_benchmark_20261022 as base
from src.experiments import od_airline_benchmark_20261022 as bench
from src.experiments.storage import write_json


def manifest():
    return dict(python=sys.executable, frozen_sha256={},
        input_origins=[dict(id="L02", instance="instances/L02")],
        protocol=dict(seconds_per_job=600, threads_per_job=1, parallel_jobs=3,
        seed=43, relative_gap_target=.005, checkpoint_seconds=60,
        cg_max_columns_per_iteration=512, max_benders_iterations=1000000,
        plain=False, common_solver_args=list(bench.COMMON_SOLVER_ARGS)))


def fake_result(batch, job, *, result_k=None, gate=None, stale=False):
    result = dict(max_orders=job["K"] if result_k is None else result_k,
        solver="CPLEX", method="benders_cg", demand_formulation=job["method"].removeprefix("benders_"),
        od_airline_bounds=job["od_airline_bounds"] if gate is None else gate,
        objective=100, bound=91 if job["od_airline_bounds"] else 90, bound_scope="global",
        wall_before_output=10, timings=dict(master_build=1, master_optimize=4, subproblem_wall=3,
                                           subproblem=dict(pricing=1)),
        continuous_root=dict(bound_valid=True, certified_lower_bound=85,
                             applied_total_cost_lower_bound=84.999),
        root_diagnostics=dict(lower_bound=91, termination_reason="strengthened_cut_closure"),
        resource_lp_cuts=[dict(value=5, domain="integer")])
    directory = batch / "runs" / job["id"]
    write_json(directory / "result.json", result)
    write_json(directory / "solution.json", {})
    write_json(directory / "path_flows.json", {})
    validation = dict(passed=True, manifest_sha256=base.sha256(batch / job["instance"] / "manifest.json"))
    for name, key in (("result.json", "result_sha256"), ("solution.json", "solution_sha256"),
                      ("path_flows.json", "path_flows_sha256")):
        validation[key] = base.sha256(directory / name)
    if stale:
        validation["result_sha256"] = "old"
    write_json(directory / "native_validation.json", validation)
    return dict(status="completed", directory=str(directory.relative_to(batch)))


def test_eight_jobs_have_two_k_two_formulations_and_two_cut_states():
    jobs = bench.jobs_for(manifest()["input_origins"])
    assert len(jobs) == len({j["id"] for j in jobs}) == 8
    assert {(j["K"], j["method"], j["od_airline_bounds"]) for j in jobs} == {
        (k, method, flag) for k in (1, 2) for method in ("benders_lifted", "benders_conditional") for flag in (False, True)}


def test_same_binary_same_common_settings_only_cut_flag_changes(tmp_path):
    cfg = manifest()
    for job in bench.jobs_for(cfg["input_origins"]):
        command = bench.command_for(tmp_path, cfg, job, tmp_path / "attempt")
        expected = base.command_for(tmp_path, cfg, job, tmp_path / "attempt")
        expected[expected.index("--k")+1] = str(job["K"])
        expected += list(bench.COMMON_SOLVER_ARGS)
        if job["od_airline_bounds"]:
            expected.append("--od-airline-bounds")
        assert command == expected
        assert command[command.index("--seconds")+1] == "600"
        assert command[command.index("--threads")+1] == "1"
        assert "--start" not in command and "--fixed-design" not in command


@pytest.mark.parametrize("index", range(8))
def test_record_accepts_both_k_and_preserves_original_continuous_domain(tmp_path, index):
    cfg = manifest()
    job = bench.jobs_for(cfg["input_origins"])[index]
    write_json(tmp_path / job["instance"] / "manifest.json", {})
    state = fake_result(tmp_path, job)
    row = bench._record(tmp_path, job, state)
    assert row["validation_passed"] and row["K"] == job["K"]
    assert row["upper_bound"] == 100 and row["lower_bound"] >= 90
    assert row["root_diagnostics_lower_bound"] > row["continuous_root_certified_lower_bound"]
    assert row["resource_lp_cuts_record_count"] == 1


@pytest.mark.parametrize("failure", ("stale", "wrong_k", "wrong_gate", "wrong_formulation"))
def test_invalid_or_mismatched_evidence_cannot_enter_bounds(tmp_path, failure):
    cfg = manifest()
    job = bench.jobs_for(cfg["input_origins"])[0]
    write_json(tmp_path / job["instance"] / "manifest.json", {})
    state = fake_result(tmp_path, job, result_k=2 if failure == "wrong_k" else None,
                        gate=True if failure == "wrong_gate" else None, stale=failure == "stale")
    if failure == "wrong_formulation":
        job = dict(job, method="benders_conditional")
    row = bench._record(tmp_path, job, state)
    assert not row["validation_passed"]
    assert row["upper_bound"] is row["lower_bound"] is row["gap"] is None


def test_report_compares_within_same_k_and_method_and_exclusive_times_sum(tmp_path):
    cfg = manifest()
    cfg["jobs"] = bench.jobs_for(cfg["input_origins"])
    write_json(tmp_path / "instances/L02/manifest.json", {})
    states = {job["id"]: fake_result(tmp_path, job) for job in cfg["jobs"]}
    write_json(tmp_path / "experiment_manifest.json", cfg)
    write_json(tmp_path / "state.json", dict(jobs=states))
    outcome = bench.report(tmp_path)
    assert outcome["validated"] == 8
    report = json.loads((tmp_path / "analysis/comparison.json").read_text())
    assert len(report["pairs"]) == 4
    assert all(p["on_minus_off_lower_bound"] == 1 and abs(p["on_minus_off_gap"]+.01) < 1e-12 for p in report["pairs"])
    import csv
    phases = list(csv.DictReader((tmp_path / "analysis/stage_times.csv").open(encoding="utf-8-sig")))
    for job in cfg["jobs"]:
        assert sum(float(row["seconds"]) for row in phases if row["job"] == job["id"]) == 10


def test_adapter_restores_base_runner_after_error():
    keys = ("command_for", "_record", "guarded_execute", "report")
    previous = {key: getattr(base, key) for key in keys}
    with pytest.raises(RuntimeError):
        with bench.adapters():
            assert base._record is bench._record
            raise RuntimeError("probe")
    assert all(getattr(base, key) is value for key, value in previous.items())


def test_prepare_freezes_current_implementation_without_starting_solver(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(bench, "load_standard", lambda *_args, **_kwargs: dict(
        demand=[2, 3], ground_cost=[10, 20], n_l=2, n_h=20, n_od=50))

    def freeze(batch, inputs, **kwargs):
        calls.append(kwargs)
        batch.mkdir()
        write_json(batch / "experiment_manifest.json", manifest())

    monkeypatch.setattr(base, "prepare", freeze)
    batch = tmp_path / "new"
    result = bench.prepare(batch, ["existing/L02"])
    assert result["jobs"] == 8 and result["solver_started"] is False
    assert calls[0]["parallel"] == 3 and calls[0]["seconds"] == 600
    assert bench.MODULE in (batch / "run.py").read_text()
    assert set(v["status"] for v in base.read(batch / "state.json")["jobs"].values()) == {"pending"}
