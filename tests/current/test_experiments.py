"""Experiment boundary tests: persistence, stopping, and actual solver runs."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from src.cli import main
from src.experiments import storage
from src.experiments.progress import RunMonitor
from src.experiments.runner import run_case

spec = importlib.util.spec_from_file_location(
    "experiment_fixture", Path(__file__).with_name("test_optimization.py")
)
fixture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixture)


def test_input_roundtrip_and_object_rejection(tmp_path):
    data = fixture._data()
    storage.save_inputs(data, tmp_path)
    loaded = storage.load_inputs(tmp_path)
    for key, value in data.items():
        if isinstance(value, np.ndarray):
            np.testing.assert_array_equal(value, loaded[key])
        else:
            assert value == loaded[key]
    with pytest.raises(ValueError, match="Object arrays"):
        storage.save_inputs({"bad": np.asarray([{}], dtype=object)}, tmp_path)


def test_nonfinite_json_is_explicit(tmp_path):
    path = tmp_path / "status.json"
    storage.write_json(path, {"gap": float("inf"), "x": np.int64(2)})
    value = json.loads(path.read_text())
    assert value["gap"] is None and value["x"] == 2
    assert value["nonfinite_fields_serialized_as_null"]
    assert not path.with_suffix(".json.tmp").exists()


def test_checkpoint_keeps_complete_generations(tmp_path):
    data = fixture._data()
    storage.save_inputs(data, tmp_path)
    monitor = RunMonitor(tmp_path)
    for i in range(5):
        monitor.checkpoint(
            {
                "objective": float(10 - i),
                "_solution": {"rho": np.zeros(2)},
                "_missions": [],
                "_relocations": [],
            }
        )
    loaded, result = storage.load_seed(tmp_path)
    assert loaded["n_l"] == 2 and result["objective"] == 6
    assert len(list((tmp_path / "checkpoints").iterdir())) == 3
    assert not monitor.stop_requested()
    (tmp_path / "STOP").touch()
    assert monitor.stop_requested()


def test_source_mutation_is_detected(tmp_path):
    source = tmp_path / "model.py"
    source.write_text("original")
    record = {"model.py": {"source": str(source), "sha256": storage.sha256(source)}}
    storage.verify_sources(record)
    source.write_text("changed")
    with pytest.raises(RuntimeError, match="changed"):
        storage.verify_sources(record)


@pytest.mark.parametrize("method", ["direct", "benders", "hybrid"])
def test_run_positive_solution_and_replay(tmp_path, method):
    inputs = tmp_path / "inputs"
    output = tmp_path / method
    data = fixture._data()
    data["pickup_share"] = np.zeros((2, 3))
    storage.save_inputs(data, inputs)
    summary = run_case(
        None,
        output,
        input_path=inputs,
        method=method,
        options={"time_limit": 12.0, "threads": 1, "max_iterations": 200},
    )
    assert summary["passed"], summary.get("error")
    assert summary["served"] > 1
    assert summary["fleet"] > 0
    assert summary["bound"] <= summary["objective"] + 1e-4
    assert summary["candidates"] == {"bases": 2, "lockers": 3, "swaps": 2}
    assert main(["validate", str(output)]) == 0
    assert main(["status", str(output)]) == 0
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["passed"] and "src/optimization/benders.py" in manifest["sources"]
    assert "result.json" in manifest["artifacts"]
    with pytest.raises(FileExistsError, match="Preserving"):
        run_case(None, output, input_path=inputs)


def test_no_historical_module_dependency():
    import ast

    root = Path(__file__).resolve().parents[2]
    for path in (root / "src").rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            modules = (
                [node.module or ""]
                if isinstance(node, ast.ImportFrom)
                else (
                    [alias.name for alias in node.names]
                    if isinstance(node, ast.Import)
                    else []
                )
            )
            assert not any(
                name.startswith(
                    ("mlnd_path", "src.data_processor", "src.benchmark", "src.utils")
                )
                for name in modules
            ), path


def test_invalid_solver_options_do_not_create_output(tmp_path):
    output = tmp_path / "invalid"
    with pytest.raises(SystemExit):
        main(["run", "--seconds", "-1", "--out", str(output)])
    assert not output.exists()
