"""Comparison bookkeeping tests; no benchmark instance or solve is generated."""

import json
import sys

import pytest

from src.experiments import conditional_benchmark_20261022 as bench
from src.experiments.storage import sha256, write_json


def test_four_commands_preserve_common_budget_and_full_domain(tmp_path):
    manifest = dict(python=sys.executable, protocol=dict(seconds_per_job=180,
        threads_per_job=1, seed=43, relative_gap_target=.005, checkpoint_seconds=45,
        cg_max_columns_per_iteration=512, max_benders_iterations=1000000, plain=False))
    for method in bench.METHODS:
        command = bench.command_for(tmp_path, manifest,
            dict(method=method, instance="instances/L01"), tmp_path / method)
        assert command[command.index("--k") + 1] == "2"
        assert command[command.index("--seconds") + 1] == "180"
        assert command[command.index("--threads") + 1] == "1"
        assert "--plain" not in command
        assert "--start" not in command and "--fixed-design" not in command
        if method == "full_gurobi":
            assert command[command.index("--max-model-variables") + 1] == "0"
        else:
            assert command[command.index("--path-column-limit") + 1] == "0"
        if method.startswith("benders_"):
            assert command[command.index("--demand-formulation") + 1] == method.split("_")[1]


def test_only_certified_global_lower_bounds_are_reported():
    assert bench.certified_bound(dict(bound=12, bound_scope="fixed_design")) is None
    assert bench.certified_bound(dict(bound=12, method="benders_cg")) is None
    assert bench.certified_bound(dict(bound=12, bound_scope="global")) == 12
    assert bench.certified_bound(dict(bound=12, method="full_path", catalogue_complete=True,
                                     construction_complete=False)) is None
    assert bench.certified_bound(dict(bound=12, method="full_path", catalogue_complete=True,
                                     construction_complete=True)) == 12


def fake_result(batch, method, *, full=False, stale=False):
    job = dict(id=method, method=method, K=2, instance="instances/L01")
    directory = batch / "runs" / method
    directory.mkdir(parents=True)
    result = dict(max_orders=2, solver="CPLEX", method="full_path" if full else "benders_cg",
                  demand_formulation="lifted", objective=100, bound=90, bound_scope="global",
                  catalogue_complete=full, construction_complete=True,
                  catalogue=dict(service=23, empty=2) if full else None,
                  generated_columns=dict(service=23 if full else 4, empty=2 if full else 1),
                  model_size=dict(variables=50, linear_rows=20, cones=1),
                  wall_before_output=5, timings={})
    write_json(directory / "result.json", result)
    for name in ("solution.json", "path_flows.json"):
        write_json(directory / name, {})
    write_json(directory / "native_validation.json", dict(passed=True,
        result_sha256="stale" if stale else sha256(directory / "result.json"),
        solution_sha256=sha256(directory / "solution.json"),
        path_flows_sha256=sha256(directory / "path_flows.json"),
        manifest_sha256=sha256(batch / "instances/L01/manifest.json")))
    state = dict(status="completed", directory=str(directory.relative_to(batch)))
    return job, state


def test_report_full_catalogue_is_separate_from_generated_columns(tmp_path):
    write_json(tmp_path / "instances/L01/manifest.json", {})
    full_job, full_state = fake_result(tmp_path, "full_cplex", full=True)
    cg_job, cg_state = fake_result(tmp_path, "benders_lifted")
    write_json(tmp_path / "experiment_manifest.json", dict(frozen_sha256={},
        jobs=[full_job, cg_job], protocol=dict(seconds_per_job=180, threads_per_job=1,
        parallel_jobs=2, relative_gap_target=.005)))
    write_json(tmp_path / "state.json", dict(jobs={full_job["id"]: full_state, cg_job["id"]: cg_state}))
    bench.report(tmp_path)
    rows = json.loads((tmp_path / "analysis/comparison.json").read_text())["rows"]
    assert rows[0]["all_paths"] == rows[1]["all_paths"] == 25
    assert rows[1]["full_path_count_source"] == "full_cplex"
    assert rows[1]["generated_service_columns"] == 4
    assert rows[1]["full_model_variables"] is None
    assert rows[1]["upper_bound"] == 100 and rows[1]["gap"] == .1


def test_stale_validation_cannot_report_bound_or_incumbent(tmp_path):
    write_json(tmp_path / "instances/L01/manifest.json", {})
    job, state = fake_result(tmp_path, "full_cplex", full=True, stale=True)
    row = bench._record(tmp_path, job, state)
    assert not row["validation_passed"]
    assert row["upper_bound"] is None and row["lower_bound"] is None and row["gap"] is None


def test_frozen_hash_mismatch_is_rejected(tmp_path):
    (tmp_path / "source.py").write_text("original\n")
    write_json(tmp_path / "experiment_manifest.json", dict(frozen_sha256={"source.py": sha256(tmp_path / "source.py")}))
    (tmp_path / "source.py").write_text("changed\n")
    with pytest.raises(ValueError, match="Frozen artifact changed"):
        bench.verify(tmp_path)


def test_live_report_uses_small_summaries_and_local_display(tmp_path, monkeypatch):
    attempt = tmp_path / "runs/probe"
    write_json(attempt / "report_row.json", dict(upper_bound=100, lower_bound=90, gap=.1))
    write_json(attempt / "guard_status.json", dict(termination_reason="solve_memory_limit",
        stage="solve", resource_limit_kind="per_job_rss_limit", peak_sampled_rss_bytes=123456))
    original = bench.optional_read

    def only_small_files(path):
        assert path.name not in ("result.json", "solution.json", "path_flows.json")
        return original(path)

    monkeypatch.setattr(bench, "optional_read", only_small_files)
    job = dict(id="probe", method="full_cplex", instance="instances/L01")
    bench.live_report(tmp_path, dict(jobs=[job]), dict(jobs={"probe": dict(
        status="failed", directory="runs/probe")}))
    markdown = (tmp_path / "live_status.md").read_text()
    assert "北京时间" in markdown and "10.000%" in markdown
    assert "solve_memory_limit" in markdown
    row = json.loads((tmp_path / "live_status.json").read_text())["rows"][0]
    assert row["gap"] == .1 and row["peak_sampled_tree_rss_bytes"] == 123456
