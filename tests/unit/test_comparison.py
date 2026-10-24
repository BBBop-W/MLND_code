"""Comparison evidence must retain source scope, real timing and valid flows."""

from copy import deepcopy
from dataclasses import replace
import csv
import json

import pytest

from src.data.instances import pack_planning_data, to_planning_data
from src.experiments.runner import run_case
from src.reporting.comparison import _history, _map_context, produce_comparison


def test_comparison_recomputes_both_policies_and_exports(tmp_path, small_input):
    paths = []
    for k in (1, 2):
        case = deepcopy(small_input)
        case.update(pack_planning_data(replace(to_planning_data(case), max_orders=k)))
        path = tmp_path / "runs" / f"K{k}"
        run_case(case, path, method="full_path", time_limit=20, make_report=False)
        paths.append(path)
    output = tmp_path / "comparison"
    result = produce_comparison(tmp_path / "runs", output)
    assert result["cases"] == 2 and result["independent_validation_passed"]
    rows = list(csv.DictReader((output / "comparison.csv").open(encoding="utf-8-sig")))
    assert rows[0]["scenario_sha256"] == rows[1]["scenario_sha256"]
    assert float(rows[0]["double_tasks_per_hour"]) == 0
    assert float(rows[1]["double_tasks_per_hour"]) > 0
    for row in rows:
        assert float(row["single_tasks_per_hour"]) + 2 * float(
            row["double_tasks_per_hour"]
        ) == pytest.approx(float(row["served_per_hour"]))
        assert sum(
            float(value) for key, value in row.items() if key.startswith("cost_")
        ) == pytest.approx(float(row["objective"]))
    for name in (
        "performance",
        "convergence",
        "operating_costs",
        "service_fleet",
        "study_area",
        "selected_facilities",
    ):
        for extension in ("png", "pdf"):
            assert (output / f"{name}.{extension}").stat().st_size > 1000
    manifest = json.loads((output / "source_manifest.json").read_text())
    assert not manifest["optimization_rerun"]
    assert len(manifest["sources"]) == 2
    assert "K=2 必然降低成本" in (output / "COMPARISON.md").read_text()


def test_mismatched_scenarios_cannot_be_compared(tmp_path, small_input):
    first = tmp_path / "first"
    second = tmp_path / "second"
    run_case(small_input, first, time_limit=0, make_report=False)
    changed = deepcopy(small_input)
    changed["ground_cost"] = changed["ground_cost"] * 2
    run_case(changed, second, time_limit=0, make_report=False)
    with pytest.raises(ValueError, match="identical physical inputs"):
        produce_comparison([first, second], tmp_path / "comparison")
    assert not (tmp_path / "comparison").exists()


def test_missing_history_is_not_invented_and_solver_points_are_distinguished(tmp_path):
    result = dict(objective=10, bound=5, wall=9, pricing_complete=False)
    assert _history(tmp_path, result) == [
        dict(
            elapsed=9.0,
            incumbent=10.0,
            bound=5.0,
            kind="validated_final",
            iteration=None,
            stage="final",
            pricing_complete=False,
        )
    ]
    events = [
        dict(elapsed=2, solver_incumbent=1e100, solver_bound=3),
        dict(elapsed=4, objective=12, bound=4, iteration=1),
    ]
    (tmp_path / "progress.jsonl").write_text(
        "\n".join(json.dumps(event) for event in events)
    )
    history = _history(tmp_path, result)
    assert [row["elapsed"] for row in history] == [2, 4, 9]
    assert history[0]["incumbent"] is None
    assert history[0]["kind"] == "solver_reported"
    assert history[1]["kind"] == "validated_iteration"


def test_unknown_crs_is_never_silently_projected_onto_osm(small_input):
    small_input["metadata"] = dict(
        projection_center_lon=113.9, projection_center_lat=22.5
    )
    context = _map_context(small_input)
    assert "CRS unspecified" in context["label"]
    with pytest.raises(ValueError, match="documented WGS84 or GCJ02"):
        _map_context(small_input, cached_basemap=True)
