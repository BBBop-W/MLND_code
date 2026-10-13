"""Incomplete direct-model construction cannot count as a successful run."""

from contextlib import nullcontext
import json
from types import SimpleNamespace

import numpy as np
import pytest

from src.experiments import runner


@pytest.fixture
def run_stub(monkeypatch, tmp_path):
    data = {"n_l": 2, "n_h": 10, "n_k": 4, "n_od": 200}
    monkeypatch.setattr(runner, "load_inputs", lambda _: data)
    monkeypatch.setattr(runner, "describe_demand", lambda *_: {"signature": "fixture"})
    monkeypatch.setattr(runner, "runtime", lambda: {})
    monkeypatch.setattr(runner, "save_inputs", lambda *_: {})
    monkeypatch.setattr(runner, "snapshot_sources", lambda *_: {})
    monkeypatch.setattr(runner, "snapshot_calibration", lambda *_, **__: {})
    monitor = SimpleNamespace(progress=None, checkpoint=None, stop_requested=None)
    monkeypatch.setattr(runner, "RunMonitor", lambda _: nullcontext(monitor))
    exports = []

    def export_operations(*args):
        exports.append(args)
        return {"summary": {"checks": {"passed": True}}}

    monkeypatch.setattr(runner, "export_operations", export_operations)

    def run(result, method):
        monkeypatch.setattr(runner, "solve", lambda *_, **__: result)
        output = tmp_path / method
        summary = runner.run_case(
            None,
            output,
            input_path=tmp_path / "input",
            method=method,
            options={"time_limit": 7200.0, "threads": 1},
        )
        return output, summary, exports

    return run


@pytest.mark.parametrize("method", ["arc", "full_path"])
@pytest.mark.parametrize("retained_baseline", [False, True])
@pytest.mark.parametrize("reason", ["time_limit", "user_stop"])
def test_incomplete_construction_fails_even_with_saved_baseline(
    run_stub, method, retained_baseline, reason
):
    result = {
        "method": method,
        "construction_complete": False,
        "status": 9 if reason == "time_limit" else 11,
        "termination_reason": reason,
        "objective": 123.0 if retained_baseline else None,
        "served": 0,
        "wall": 7200.0,
    }
    if retained_baseline:
        result["_solution"] = {"rho": np.zeros(2)}
    output, summary, exports = run_stub(result, method)

    assert summary["status"] == "failed"
    assert summary["passed"] is False
    assert summary["failed_phase"] == "construction"
    assert summary["construction_complete"] is False
    assert summary["termination_reason"] == reason
    assert reason in summary["error"]["message"]
    assert not exports
    assert (output / "solution.npz").exists() is retained_baseline
    saved = json.loads((output / "result.json").read_text())
    assert saved["construction_complete"] is False
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == "failed" and manifest["passed"] is False
    assert json.loads((output / "summary.json").read_text())["failed_phase"] == "construction"


@pytest.mark.parametrize("method", ["arc", "full_path"])
@pytest.mark.parametrize("served", [0.0, 50.0])
def test_complete_construction_with_time_limited_solution_is_valid(
    run_stub, method, served
):
    result = {
        "method": method,
        "construction_complete": True,
        "status": 9,
        "termination_reason": "time_limit",
        "objective": 123.0,
        "served": served,
        "_solution": {"rho": np.zeros(2)},
    }
    _, summary, exports = run_stub(result, method)
    assert summary["passed"] is True
    assert summary["status"] == (
        "feasible_limit" if served else "feasible_baseline_only"
    )
    assert "failed_phase" not in summary
    assert len(exports) == 1
