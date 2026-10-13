"""Check the saved sensitivity batch and its automatic result persistence."""

import json
from pathlib import Path

import pytest

from src.experiments import performance, sensitivity
from src.experiments.storage import sha256, write_json


def test_h20_plan_preserves_all_cases_and_requested_limits(tmp_path, monkeypatch):
    if not sensitivity.DEFAULT_BASE.exists():
        pytest.skip("Frozen Nanshan H20 inputs unavailable")
    monkeypatch.setattr(performance, "_gurobi_defaults",
                        lambda: dict(mip_gap=1e-4, mip_gap_abs=1e-10))
    plan = sensitivity.prepare(tmp_path / "experiment")
    assert len(plan["jobs"]) == 13
    assert plan["workers"] == 3 and plan["seconds"] == 3600
    assert plan["solver"]["mip_gap"] == .005
    assert plan["postprocess"]["kind"] == "sensitivity"
    assert sum(job["factor"] == "baseline" for job in plan["jobs"]) == 1
    assert {job["method"] for job in plan["jobs"]} == {"arc"}
    for job in plan["jobs"]:
        config = json.loads(Path(job["config"]).read_text())
        assert config["solver"]["mip_gap"] == .005
        assert config["solver"]["time_limit"] == 3600
        for name, digest in job["input_sha256"].items():
            assert sha256(Path(job["input_path"]) / name) == digest
    for module in ("experiments/sensitivity.py", "reporting/sensitivity.py"):
        assert (Path(plan["source_snapshot"]) / "src" / module).is_file()


def test_postprocessing_extends_summary_and_artifact_manifest(tmp_path, monkeypatch):
    from src.reporting import sensitivity as report

    monkeypatch.setattr(report, "export_case", lambda *args, **kwargs: dict(
        summary=dict(saving_pct=12.3, fleet=999, swap_visits_per_hour=4.5)))
    write_json(tmp_path / "manifest.json", dict(passed=True, artifacts={}))
    summary = sensitivity.finalize_job(
        dict(postprocess=dict(kind="sensitivity", map_cache=str(tmp_path))),
        dict(output=str(tmp_path), factor="speed", multiplier=1.25),
        dict(status="ok", passed=True, fleet=3.0),
    )
    assert summary["postprocess_status"] == "complete"
    assert summary["saving_pct"] == 12.3 and summary["fleet"] == 3.0
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["artifacts"]["summary.json"]["sha256"] == sha256(tmp_path / "summary.json")


def test_round_two_has_twelve_additions_and_two_workers(tmp_path, monkeypatch):
    if not sensitivity.DEFAULT_BASE.exists():
        pytest.skip("Frozen Nanshan H20 inputs unavailable")
    monkeypatch.setattr(performance, "_gurobi_defaults",
                        lambda: dict(mip_gap=1e-4, mip_gap_abs=1e-10))
    output = tmp_path / "round2"
    plan = sensitivity.prepare(output, round_number=2)
    assert len(plan["jobs"]) == 12 and plan["workers"] == 2
    assert {job["round"] for job in plan["jobs"]} == {2}
    assert {job["method"] for job in plan["jobs"]} == {"arc"}
    assert plan["solver"]["threads"] == 1
    assert plan["solver"]["mip_gap"] == .005 and plan["seconds"] == 3600
    for job in plan["jobs"]:
        assert job["OD"] == 400
        assert job["factor"] != "baseline"
    density = [job for job in plan["jobs"] if job["factor"] == "density"]
    assert {(job["H"], job["L"], job["E"]) for job in density} == {
        (10, 2, 4), (15, 3, 6), (25, 5, 10), (30, 6, 12)}
    manifest = json.loads((output / "instances.json").read_text())
    assert manifest["round"] == 2 and manifest["total_cases"] == 12
    assert manifest["baseline_run"] == str(sensitivity.ROUND_ONE / "batch/runs/H20_baseline__arc")


def test_round_three_is_fifteen_cases_with_full_solution_storage(tmp_path, monkeypatch):
    if not sensitivity.DEFAULT_BASE.exists():
        pytest.skip("Frozen Nanshan H20 inputs unavailable")
    monkeypatch.setattr(performance, "_gurobi_defaults",
                        lambda: dict(mip_gap=1e-4, mip_gap_abs=1e-10))
    output = tmp_path / "round3"
    plan = sensitivity.prepare(output, round_number=3)
    assert len(plan["jobs"]) == 15 and plan["workers"] == 2
    assert {job["round"] for job in plan["jobs"]} == {3}
    assert {job["method"] for job in plan["jobs"]} == {"arc"}
    assert {(job["H"], job["L"], job["E"], job["OD"])
            for job in plan["jobs"]} == {(20, 4, 8, 400)}
    assert plan["solver"]["threads"] == 1
    assert plan["solver"]["mip_gap"] == .005 and plan["seconds"] == 3600
    manifest = json.loads((output / "instances.json").read_text())
    assert manifest["round"] == 3 and manifest["total_cases"] == 15
    assert manifest["factors"]["max_cabinets"] == [3, 4, 6, 7]
    assert manifest["solution_storage"]["primal_solution"] == "solution.npz"
    assert manifest["solution_storage"]["per_od_times"] == "sensitivity/od_times.csv"
