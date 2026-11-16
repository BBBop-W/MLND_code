"""Independent algorithm switches and provenance without altering saved inputs."""

import json

import numpy as np
import pytest

from src.cli import _accelerations, main, parser
from src.data.instances import to_planning_data
from src.experiments.storage import save_inputs
from src.solvers.methods import ACCELERATIONS, get_method, resolve_accelerations


@pytest.mark.parametrize(
    "method, enabled", [("benders_cg", True), ("benders_cg_plain", False)]
)
def test_method_defaults_expose_all_seven_switches(method, enabled):
    expected = {name: enabled for name in ACCELERATIONS}
    assert len(expected) == 7
    assert resolve_accelerations(method) == expected
    assert {name: get_method(method)[name] for name in ACCELERATIONS} == expected


@pytest.mark.parametrize("name", ACCELERATIONS)
@pytest.mark.parametrize(
    "method, enabled", [("benders_cg", True), ("benders_cg_plain", False)]
)
def test_each_switch_can_be_ablated_independently(name, method, enabled):
    overrides = {name: not enabled}
    actual = resolve_accelerations(method, overrides)
    assert actual[name] is not enabled
    assert all(value is enabled for key, value in actual.items() if key != name)
    assert overrides == {name: not enabled}
    assert all(value is enabled for value in resolve_accelerations(method).values())


@pytest.mark.parametrize("value", [1, 0, "false", None, np.bool_(True)])
def test_acceleration_values_require_actual_booleans(value):
    with pytest.raises(ValueError, match="boolean"):
        resolve_accelerations("benders_cg", {"pareto_cuts": value})


@pytest.mark.parametrize("overrides", [False, [], "pareto_cuts"])
def test_acceleration_overrides_require_a_dict(overrides):
    with pytest.raises(ValueError, match="dict"):
        resolve_accelerations("benders_cg", overrides)


def test_unknown_method_or_switch_is_rejected():
    with pytest.raises(ValueError, match="Unknown method"):
        resolve_accelerations("unknown")
    with pytest.raises(ValueError, match="Unknown acceleration"):
        resolve_accelerations("benders_cg", {"pareto_cut": True})


def test_full_path_has_no_benders_accelerations():
    assert resolve_accelerations("full_path") == {}
    assert resolve_accelerations("full_path", {}) == {}
    for enabled in (False, True):
        with pytest.raises(ValueError, match="full_path"):
            resolve_accelerations("full_path", {"pareto_cuts": enabled})


def test_repeated_cli_flags_resolve_to_independent_overrides():
    args = parser().parse_args(
        [
            "plan",
            "--enable-acceleration",
            "pareto_cuts",
            "--enable-acceleration",
            "warm_start",
            "--disable-acceleration",
            "reuse_columns",
            "--disable-acceleration",
            "primal_search",
        ]
    )
    assert _accelerations(args) == {
        "pareto_cuts": True,
        "warm_start": True,
        "reuse_columns": False,
        "primal_search": False,
    }
    assert _accelerations(parser().parse_args(["plan"])) is None


def test_conflicting_cli_flags_fail_before_loading_input(capsys):
    code = main(
        [
            "plan",
            "--input",
            "nonexistent-input",
            "--enable-acceleration",
            "pareto_cuts",
            "--disable-acceleration",
            "pareto_cuts",
        ]
    )
    assert code == 2
    assert "both enable and disable" in json.loads(capsys.readouterr().out)["error"]


def test_full_path_cli_rejects_switch_before_loading_input(capsys):
    assert (
        main(
            [
                "plan",
                "--input",
                "nonexistent-input",
                "--method",
                "full_path",
                "--disable-acceleration",
                "pareto_cuts",
            ]
        )
        == 2
    )
    assert "full_path" in json.loads(capsys.readouterr().out)["error"]


def test_cli_plan_preserves_saved_input_and_resolves_ablation(
    tmp_path, small_input, capsys
):
    save_inputs(small_input, tmp_path)
    before = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    assert (
        main(
            [
                "plan",
                "--input",
                str(tmp_path),
                "--method",
                "benders_cg_plain",
                "--enable-acceleration",
                "pareto_cuts",
                "--seconds",
                "600",
                "--threads",
                "1",
                "--seed",
                "43",
                "--gap",
                "0.005",
                "--column-limit",
                "500000",
                "--max-iterations",
                "100",
            ]
        )
        == 0
    )
    planned = json.loads(capsys.readouterr().out)
    assert planned["options"]["accelerations"] == {"pareto_cuts": True}
    assert planned["resolved_accelerations"] == {
        name: name == "pareto_cuts" for name in ACCELERATIONS
    }
    assert before == {path.name: path.read_bytes() for path in tmp_path.iterdir()}


def test_api_passes_resolved_switches_to_benders(monkeypatch, small_input):
    from src.solvers import api

    calls = []

    def fake_benders(data, paths, **options):
        calls.append(options)
        return api.baseline(data)

    monkeypatch.setattr(api, "solve_benders", fake_benders)
    result = api.solve(
        to_planning_data(small_input),
        method="benders_cg_plain",
        accelerations={"pareto_cuts": True},
    )
    expected = resolve_accelerations("benders_cg_plain", {"pareto_cuts": True})
    assert calls[0]["accelerations"] == expected
    assert calls[0]["accelerated"] is False
    assert result["accelerations"] == expected


def test_stopped_enumeration_still_records_accelerations(small_input):
    from src.solvers.api import solve

    result = solve(
        to_planning_data(small_input),
        time_limit=0,
        accelerations={"pareto_cuts": False},
    )
    assert not result["catalogue_complete"]
    assert result["accelerations"] == resolve_accelerations(
        "benders_cg", {"pareto_cuts": False}
    )


def test_runner_saves_switches_in_manifest_result_and_summary(
    tmp_path, small_input, monkeypatch
):
    from src.experiments import runner

    monkeypatch.setattr(runner, "source_hashes", lambda: {"test": "stable"})
    summary = runner.run_case(
        small_input,
        tmp_path / "run",
        method="benders_cg_plain",
        accelerations={"pareto_cuts": True},
        time_limit=0,
        make_report=False,
    )
    expected = resolve_accelerations("benders_cg_plain", {"pareto_cuts": True})
    result = json.loads((tmp_path / "run/result.json").read_text())
    manifest = json.loads((tmp_path / "run/manifest.json").read_text())
    assert result["accelerations"] == expected
    assert summary["accelerations"] == expected
    assert manifest["options"]["accelerations"] == expected


def test_runner_rejects_bad_switch_before_creating_output(tmp_path, small_input):
    from src.experiments.runner import run_case

    output = tmp_path / "not-created"
    with pytest.raises(ValueError, match="full_path"):
        run_case(
            small_input, output, method="full_path", accelerations={"pareto_cuts": True}
        )
    assert not output.exists()
