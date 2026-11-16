"""Saved-evidence report tests. No optimizer is constructed or invoked."""

import csv
import json
from pathlib import Path
import shutil

import pytest

from src.experiments.storage import sha256, write_json
from src.reporting import scale_ladder as report

ROOT = Path(__file__).resolve().parents[2]
ANCHOR = ROOT / "examples/tsl_2l20h50od"
SAVED = ROOT / "outputs/cpp_cplex_20261016/full_cplex/K1"


@pytest.fixture(autouse=True)
def no_charts_or_optimizer(monkeypatch):
    for name in ("plot_preparation", "plot_comparisons", "plot_phases", "plot_operations", "plot_progress"):
        monkeypatch.setattr(report, name, lambda *_args, **_kwargs: [])
    import gurobipy as gp

    def forbidden(*_args, **_kwargs):
        raise AssertionError("Report generation must not invoke an optimizer")

    monkeypatch.setattr(gp.Model, "optimize", forbidden)


def _read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _batch(tmp_path, *, with_run=False):
    batch = tmp_path / "batch"
    shutil.copytree(ANCHOR, batch / "instances/L02")
    job = dict(id="L02_K1_full_cplex", level=2, K=1, method="full_cplex",
               instance="instances/L02", output="runs/L02/full_cplex/K1")
    write_json(batch / "preparation_manifest.json", {"levels": [{"level": 2}]})
    write_json(batch / "experiment_manifest.json", {"jobs": [job], "protocol": {}})
    state = {"status": "pending", "attempts": []}
    if with_run:
        if not SAVED.is_dir():
            pytest.skip("Saved independent-validation fixture unavailable")
        relative = job["output"] + "/attempts/001"
        attempt = batch / relative
        shutil.copytree(SAVED, attempt)
        certificate = json.loads((attempt / "native_validation.json").read_text())
        certificate["operations_sha256"] = sha256(attempt / "operations/operations.json")
        certificate["operations_files_sha256"] = {
            str(path.relative_to(attempt / "operations")): sha256(path)
            for path in (attempt / "operations").rglob("*") if path.is_file()
        }
        write_json(attempt / "native_validation.json", certificate)
        write_json(attempt / "execution.json", dict(returncode=0, external_wall_seconds=70,
                   sampled_peak_rss_bytes=1024**3, status="exited"))
        (attempt / "resources.csv").write_text("elapsed_seconds,rss_bytes\n0,1048576\n1,1073741824\n")
        state = dict(status="completed", directory=relative, result_directory=relative,
                     attempts=[dict(number=1, directory=relative)])
    write_json(batch / "state.json", {"jobs": {job["id"]: state}})
    return batch, job, state


def test_preparation_only_plan_has_sixty_jobs(tmp_path):
    write_json(tmp_path / "preparation_manifest.json", {"levels": [{"level": n} for n in range(1, 11)]})
    _, jobs = report._jobs(tmp_path)
    assert len(jobs) == len({job["id"] for job in jobs}) == 60
    assert {job["method"] for job in jobs} == {"full_cplex", "full_gurobi", "benders_cplex"}


def test_unrun_rows_are_visible_and_have_no_fake_numbers(tmp_path):
    batch, _, _ = _batch(tmp_path)
    result = report.build_scale_report(batch)
    rows = _read_csv(batch / "analysis/comparison.csv")
    assert result["solver_invoked"] is False
    assert result["validated_jobs"] == 0
    assert rows[0]["state"] == "not_run"
    assert rows[0]["upper_bound"] == rows[0]["gap"] == rows[0]["solve_seconds"] == ""
    assert not _read_csv(batch / "analysis/global_progress.csv")


def test_validated_saved_run_produces_metrics_and_point_tables(tmp_path):
    batch, _, state = _batch(tmp_path, with_run=True)
    result = report.build_scale_report(batch)
    row = _read_csv(batch / "analysis/comparison.csv")[0]
    assert result["validated_jobs"] == 1
    assert row["state"] == "validated"
    assert float(row["upper_bound"]) > float(row["lower_bound"])
    assert row["operation_tables_state"] == "current_hashes_verified"
    assert float(row["sampled_peak_rss_gib"]) == 1
    assert _read_csv(batch / "analysis/operations/bases.csv")
    links = _read_csv(batch / "analysis/large_table_links.csv")
    assert any(item["table"] == "paths" and item["available"] == "True" for item in links)
    assert not (batch / "analysis/operations/paths.csv").exists()
    assert (batch / state["directory"] / "result.json").exists()


def test_exact_process_usage_is_reported_separately_from_sampling(tmp_path):
    batch, _, state = _batch(tmp_path, with_run=True)
    attempt = batch / state["directory"]
    write_json(attempt / "resource_usage.json", dict(peak_rss_bytes=2 * 1024**3,
               cpu_user_seconds=12.5, cpu_system_seconds=1.25))
    execution = json.loads((attempt / "execution.json").read_text())
    execution["peak_rss_bytes"] = 1024**3
    write_json(attempt / "execution.json", execution)
    report.build_scale_report(batch)
    row = _read_csv(batch / "analysis/comparison.csv")[0]
    assert float(row["peak_rss_gib"]) == 2
    assert float(row["sampled_peak_rss_gib"]) == 1
    assert float(row["cpu_user_seconds"]) == 12.5
    assert float(row["cpu_system_seconds"]) == 1.25


def test_missing_sampled_usage_does_not_hide_exact_usage(tmp_path):
    write_json(tmp_path / "resource_usage.json", dict(peak_rss_bytes=123,
               cpu_user_seconds=4, cpu_system_seconds=0))
    actual = report._resources(tmp_path, {"sampled_peak_rss_bytes": None})
    assert actual["peak_rss_bytes"] == 123
    assert actual["cpu_system_seconds"] == 0
    assert actual["sampled_peak_rss_bytes"] is None


def test_sampled_usage_does_not_invent_exact_usage(tmp_path):
    actual = report._resources(tmp_path, {"sampled_peak_rss_bytes": 123})
    assert actual["sampled_peak_rss_bytes"] == 123
    assert actual["peak_rss_bytes"] is None
    assert actual["cpu_user_seconds"] is None


@pytest.mark.parametrize("result,expected", [
    ({"acceleration_statistics": {"capacity_cuts": 3}}, 3),
    ({"acceleration_statistics": {"capacity_cuts_added": 4}}, 4),
    ({"capacity_cuts": [{"base": 0}, {"base": 1}]}, 2),
    ({"capacity_cuts": []}, 0),
    ({}, None),
])
def test_capacity_cut_count_supports_current_and_saved_schemas(result, expected):
    assert report._capacity_cuts(result) == expected


@pytest.mark.parametrize("filename", ["result.json", "operations/bases.csv", "operations/operations.json"])
def test_stale_evidence_never_enters_numeric_comparison(tmp_path, filename):
    batch, _, state = _batch(tmp_path, with_run=True)
    path = batch / state["directory"] / filename
    path.write_text(path.read_text() + "\n")
    result = report.build_scale_report(batch)
    row = _read_csv(batch / "analysis/comparison.csv")[0]
    assert result["validated_jobs"] == 0
    assert row["state"] == "invalid_evidence"
    assert row["upper_bound"] == row["lower_bound"] == row["solve_seconds"] == ""
    assert not _read_csv(batch / "analysis/operations/bases.csv")


def test_progress_uses_only_validated_global_events(tmp_path):
    batch, _, state = _batch(tmp_path, with_run=True)
    attempt = batch / state["directory"]
    final = json.loads((attempt / "result.json").read_text())
    events = [
        dict(elapsed_seconds=1, bound_scope="global", validated_incumbent=True,
             upper_bound=final["objective"] + 100, lower_bound=9000),
        dict(elapsed_seconds=2, bound_scope="fixed_design_recourse", validated_incumbent=True,
             upper_bound=1, lower_bound=1),
        dict(elapsed_seconds=3, bound_scope="global", validated_incumbent=False,
             upper_bound=100, lower_bound=50),
        dict(elapsed_seconds=4, bound_scope="global", validated_incumbent=True,
             upper_bound=final["objective"], lower_bound=final["bound"]),
    ]
    (attempt / "progress.jsonl").write_text("\n".join(map(json.dumps, events)) + "\n")
    report.build_scale_report(batch)
    rows = _read_csv(batch / "analysis/global_progress.csv")
    assert [float(row["elapsed_seconds"]) for row in rows] == [1, 4]


@pytest.mark.parametrize("change", [{"method": "benders_cg"}, {"solver": "Gurobi"}])
def test_validated_result_from_another_method_is_rejected(tmp_path, change):
    batch, _, state = _batch(tmp_path, with_run=True)
    attempt = batch / state["directory"]
    result_path = attempt / "result.json"
    result = json.loads(result_path.read_text())
    result.update(change)
    write_json(result_path, result)
    certificate = json.loads((attempt / "native_validation.json").read_text())
    certificate["result_sha256"] = sha256(result_path)
    write_json(attempt / "native_validation.json", certificate)
    report.build_scale_report(batch)
    row = _read_csv(batch / "analysis/comparison.csv")[0]
    assert row["state"] == "invalid_evidence"
    assert "method/solver" in row["validation_error"]
    assert row["upper_bound"] == row["lower_bound"] == ""


def test_partial_checkpoint_keeps_upper_bound_without_unknown_lower(tmp_path):
    batch, job, state = _batch(tmp_path, with_run=True)
    attempt = batch / state["directory"]
    result_path = attempt / "result.json"
    result = json.loads(result_path.read_text())
    result.update(recovered_from_checkpoint=True, catalogue_complete=False,
                  construction_complete=False, termination_reason="interrupted_checkpoint")
    write_json(result_path, result)
    certificate = json.loads((attempt / "native_validation.json").read_text())
    certificate["result_sha256"] = sha256(result_path)
    write_json(attempt / "native_validation.json", certificate)
    execution = json.loads((attempt / "execution.json").read_text())
    execution.update(returncode=-9, stop_reason="external_memory_limit")
    write_json(attempt / "execution.json", execution)
    state["status"] = "interrupted"
    write_json(batch / "state.json", {"jobs": {job["id"]: state}})
    report.build_scale_report(batch)
    row = _read_csv(batch / "analysis/comparison.csv")[0]
    assert row["state"] == "validated"
    assert row["completion_kind"] == "partial_validated_incumbent"
    assert float(row["upper_bound"]) > 0
    assert row["lower_bound"] == row["gap"] == ""
    assert row["lower_bound_source"] == "global_validity_not_established"


def test_invalidated_result_removes_previous_owned_figures(tmp_path):
    batch, _, _ = _batch(tmp_path)
    owned = batch / "analysis/figures/previous.png"
    owned.parent.mkdir(parents=True)
    owned.write_bytes(b"obsolete")
    write_json(batch / "analysis/report_manifest.json", {"figures": ["figures/previous.png"]})
    report.build_scale_report(batch)
    assert not owned.exists()
