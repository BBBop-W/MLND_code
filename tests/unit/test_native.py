"""Native output restoration and provenance must fail closed on corruption."""

import json
from dataclasses import replace
from pathlib import Path
import shutil
import sys

import numpy as np
import pytest

from src.experiments.native import load_native_result, validate_native
from src.experiments import native
from src.experiments.storage import sha256


EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "tsl_2l20h50od"


def write(path, value):
    path.write_text(json.dumps(value))


@pytest.fixture
def native_run(tmp_path):
    directory = tmp_path / "run"
    directory.mkdir()
    manifest = json.loads((EXAMPLE / "manifest.json").read_text())
    write(
        directory / "result.json",
        {
            "model_version": "consecutive",
            "max_orders": 1,
            "input_manifest_sha256": sha256(EXAMPLE / "manifest.json"),
        },
    )
    write(
        directory / "solution.json",
        {"facility": [0, 1], "fleet": 2.5, "assignment": [[[1, 0], [0, 0]]]},
    )
    write(
        directory / "path_flows.json",
        {"segments": [], "missions": [], "relocations": []},
    )
    write(directory / "input_reference.json", {"max_orders": 1, "manifest": manifest})
    return directory


def test_native_solution_arrays_and_scalar_restored(native_run):
    result = load_native_result(native_run)
    assert result["_solution"]["fleet"].shape == ()
    assert result["_solution"]["fleet"] == 2.5
    np.testing.assert_array_equal(result["_solution"]["facility"], [0.0, 1.0])
    assert result["_solution"]["assignment"].shape == (1, 2, 2)
    assert result["_solution"]["facility"].dtype == np.float64
    assert result["_segments"] == result["_missions"] == result["_relocations"] == []


def test_missing_path_record_family_rejected(native_run):
    write(native_run / "path_flows.json", {"segments": [], "missions": []})
    with pytest.raises(ValueError, match="segments, missions, relocations"):
        load_native_result(native_run)


@pytest.mark.parametrize("value", [None, float("nan"), float("inf"), -float("inf")])
def test_nonfinite_decision_rejected(native_run, value):
    write(native_run / "solution.json", {"fleet": value})
    with pytest.raises(ValueError, match="must be finite"):
        load_native_result(native_run)


def test_missing_native_provenance_rejected(native_run):
    (native_run / "input_reference.json").unlink()
    with pytest.raises(ValueError, match="missing required input_reference"):
        validate_native(native_run, EXAMPLE)


def test_incomplete_native_provenance_rejected(native_run):
    write(native_run / "input_reference.json", {"max_orders": 1})
    with pytest.raises(ValueError, match="must contain manifest and max_orders"):
        validate_native(native_run, EXAMPLE)


def test_changed_case_rejected(native_run):
    with pytest.raises(ValueError, match="Requested K differs"):
        validate_native(native_run, EXAMPLE, max_orders=2)


def test_result_case_conflicts_with_provenance(native_run):
    metadata = json.loads((native_run / "result.json").read_text())
    metadata["max_orders"] = 2
    write(native_run / "result.json", metadata)
    with pytest.raises(ValueError, match="Requested K differs"):
        validate_native(native_run, EXAMPLE)


def test_changed_manifest_rejected(native_run):
    provenance = json.loads((native_run / "input_reference.json").read_text())
    provenance["manifest"]["files"]["model.json"] = "0" * 64
    write(native_run / "input_reference.json", provenance)
    with pytest.raises(ValueError, match="manifest differs"):
        validate_native(native_run, EXAMPLE)


def test_changed_result_manifest_hash_rejected(native_run):
    metadata = json.loads((native_run / "result.json").read_text())
    metadata["input_manifest_sha256"] = "0" * 64
    write(native_run / "result.json", metadata)
    with pytest.raises(ValueError, match="manifest SHA-256 differs"):
        validate_native(native_run, EXAMPLE)


def test_changed_csv_rejected_before_operational_checks(native_run, tmp_path):
    instance = tmp_path / "instance"
    shutil.copytree(EXAMPLE, instance)
    with (instance / "od_demand.csv").open("a") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="SHA-256 mismatch: od_demand.csv"):
        validate_native(native_run, instance)


def test_validation_cli_returns_failure_status(monkeypatch, tmp_path):
    monkeypatch.setattr(
        sys, "argv", ["native", "validate", str(tmp_path), "--instance", str(EXAMPLE)]
    )
    monkeypatch.setattr(
        native,
        "validate_native",
        lambda *a, **k: {
            "passed": False,
            "max_orders": 1,
            "validation_and_export_seconds": 0,
        },
    )
    with pytest.raises(SystemExit) as stopped:
        native.main()
    assert stopped.value.code == 1


def test_gurobi_cli_returns_failure_status(monkeypatch, tmp_path):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "native",
            "gurobi",
            "--instance",
            str(EXAMPLE),
            "--k",
            "1",
            "--out",
            str(tmp_path),
        ],
    )
    monkeypatch.setattr(
        native, "gurobi_reference", lambda *a, **k: {"independently_validated": False}
    )
    with pytest.raises(SystemExit) as stopped:
        native.main()
    assert stopped.value.code == 1


def test_reference_refuses_overwriting_existing_result(tmp_path):
    write(tmp_path / "result.json", {"preserved": True})
    with pytest.raises(FileExistsError, match="use a new directory"):
        native.gurobi_reference(EXAMPLE, tmp_path, max_orders=1)
    assert json.loads((tmp_path / "result.json").read_text()) == {"preserved": True}


def test_suboptimal_status_remains_explicit():
    from gurobipy import GRB

    assert native._gurobi_status_name(GRB.SUBOPTIMAL) == "suboptimal"
    assert native._gurobi_status_name(GRB.OPTIMAL) == "optimal"
    assert native._gurobi_status_name(GRB.TIME_LIMIT) == "time_limit"


@pytest.mark.parametrize("k", [1, 2])
@pytest.mark.parametrize("fixed", [False, True])
@pytest.mark.parametrize("tight", [False, True])
def test_reference_arc_alignment_changes_only_native_airline_bounds(small_input, k, fixed, tight):
    from src.data.instances import to_planning_data
    from src.model.paths import enumerate_paths
    from src.solvers.formulation import activation_bounds, build_model, shapes

    data = replace(to_planning_data(small_input), max_orders=k)
    paths = enumerate_paths(data)
    design = {name: np.zeros(shape) for name, shape in shapes(data).items()} if fixed else None
    built = build_model(data, paths, fixed_design=design)
    try:
        model = built.model
        before = model.getA().toarray()
        rhs, costs = model.getAttr("RHS"), model.getAttr("Obj")
        dimensions = (model.NumVars, model.NumConstrs, model.NumQConstrs)
        catalogue_bounds = activation_bounds(data, paths).copy()
        metadata = native.align_native_arc_bounds(built, tight_airline_bounds=tight)
        expected = before.copy()
        demand = float(data.demand.sum())
        for u, v in np.ndindex(data.allowed_links.shape):
            if not data.allowed_links[u, v] or u == v:
                bound = 0.0
            elif u < data.n_l and v < data.n_l:
                bound = (data.n_l - 1) * demand
            elif u >= data.n_l and v >= data.n_l:
                bound = (1 if tight else k) * demand
            else:
                bound = demand
            row = built.constraints["link"][u, v]
            variable = built.variables["airline"][u, v]
            expected[row.index, variable.index] = -bound
        np.testing.assert_array_equal(model.getA().toarray(), expected)
        assert model.getAttr("RHS") == rhs
        assert model.getAttr("Obj") == costs
        assert (model.NumVars, model.NumConstrs, model.NumQConstrs) == dimensions
        assert metadata["scheme"] == "native_input_arc_bounds"
        assert metadata["rows_checked"] == data.n_nodes**2
        assert metadata["changed_airline_coefficients"] == np.count_nonzero(expected != before)
        np.testing.assert_array_equal(activation_bounds(data, paths), catalogue_bounds)
        assert native.align_native_arc_bounds(built, tight_airline_bounds=tight)["changed_airline_coefficients"] == 0
        assert metadata["tight_airline_bounds"] == tight
    finally:
        built.model.dispose()
