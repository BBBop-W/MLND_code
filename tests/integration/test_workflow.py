"""A real solver -> saved arrays -> independent audit -> figures -> CLI replay."""

import json
import numpy as np
import pytest
from src.cli import main
from src.experiments.runner import run_case, compare
from src.experiments.storage import load_seed, save_inputs, load_inputs
from src.operations.validation import build_operations


def test_saved_run_reports_and_cli_replay(tmp_path, small_input, capsys):
    output = tmp_path / "run"
    summary = run_case(small_input, output, method="full_path", time_limit=20)
    assert summary["passed"]
    assert summary["served"] > 0
    data, result = load_seed(output)
    independent = build_operations(data, result)
    assert independent["summary"]["checks"]["passed"]
    assert independent["summary"]["double_tasks"] > 0
    report = output / "report"
    manifest = json.loads((report / "source_manifest.json").read_text())
    assert (
        manifest["independent_validation_passed"] and not manifest["optimization_rerun"]
    )
    for name in ("network", "workload", "cost_and_fleet", "receipt_times"):
        assert (report / f"{name}.png").stat().st_size > 1000
        assert (report / f"{name}.pdf").stat().st_size > 1000
    assert main(["validate", str(output)]) == 0
    assert main(["status", str(output)]) == 0
    assert main(["inspect", "--input", str(output)]) == 0
    assert main(["stop", str(output)]) == 0
    assert (output / "STOP").exists()
    assert main(["run", "--input", str(output), "--out", str(output)]) == 2
    capsys.readouterr()


def test_stop_before_construction_still_saves_valid_baseline(tmp_path, small_input):
    output = tmp_path / "stopped"
    summary = run_case(small_input, output, time_limit=0, make_report=False)
    assert summary["passed"] and summary["served"] == 0
    assert not summary["pricing_complete"]
    assert summary["bound"] == 0
    assert summary["termination_reason"] == "time_limit"


def test_k1_k2_share_market_and_keep_separate_results(tmp_path, small_input):
    rows = compare(
        small_input,
        tmp_path / "comparison",
        method="full_path",
        time_limit=20,
        make_report=False,
    )
    assert [r["max_orders"] for r in rows] == [1, 2]
    first, _ = load_seed(tmp_path / "comparison" / "K1")
    second, _ = load_seed(tmp_path / "comparison" / "K2")
    np.testing.assert_array_equal(first["demand"], second["demand"])
    np.testing.assert_array_equal(first["flight_times"], second["flight_times"])
    assert rows[0]["continued_orders"] == 0
    assert rows[1]["continued_orders"] > 0


def test_saved_array_shape_tampering_is_rejected(tmp_path, small_input):
    save_inputs(small_input, tmp_path)
    path = tmp_path / "input_metadata.json"
    meta = json.loads(path.read_text())
    meta["arrays"]["demand"]["shape"] = [99]
    path.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="metadata mismatch"):
        load_inputs(tmp_path)
