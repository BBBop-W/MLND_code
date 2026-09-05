from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.benchmark.branch_cut.artifacts import write_canonical_json_exclusive
from src.benchmark.branch_cut.cli import (
    CLIHooks,
    build_parser,
    dispatch,
    load_benchmark_manifest,
)
from src.benchmark.branch_cut.hashing import build_pair_id, canonical_json
from src.benchmark.branch_cut.schema import BenchmarkMode, PairSpec, SolverConfig
from src.benchmark.branch_cut.telemetry import write_jsonl_exclusive
from tests.benchmark.branch_cut.test_cli import _NOW, _formal_manifest


def _write_progress_attempt(
    root: Path,
    *,
    profile_code: str,
    manifest_id: str,
    case_id: str,
    preparation_seconds: float,
    events: tuple[dict[str, object], ...],
    include_progress: bool = True,
    malformed_progress: bool = False,
) -> Path:
    pair = PairSpec(
        manifest_id=manifest_id,
        case_id=case_id,
        mode=BenchmarkMode.MX,
        solver_seed=42,
        time_budget_seconds=60.0,
        solver=SolverConfig(),
    )
    pair_id = build_pair_id(pair)
    directory = root / pair_id / profile_code
    directory.mkdir(parents=True)
    write_canonical_json_exclusive(
        directory / "run_spec.json",
        {"pair": json.loads(canonical_json(pair))},
    )
    final = events[-1]
    final_incumbent = final.get("incumbent")
    final_bound = final.get("bound")
    raw_main_runtime = final["runtime_seconds"]
    assert isinstance(raw_main_runtime, (int, float)) and not isinstance(
        raw_main_runtime, bool
    )
    main_runtime = float(raw_main_runtime)
    write_canonical_json_exclusive(
        directory / "run_record.json",
        {
            "ids": {"pair_id": pair_id, "attempt_number": 1},
            "profile_code": profile_code,
            "solver_metrics": {
                "status_name": "TIME_LIMIT",
                "has_incumbent": final_incumbent is not None,
                "objective": final_incumbent,
                "best_bound": final_bound,
                "relative_gap": 0.0,
                "main_solve_runtime_seconds": main_runtime,
                "total_algorithm_runtime_seconds": (main_runtime + preparation_seconds),
                "algorithm_prepare_runtime_seconds": preparation_seconds,
                "start_preparation_runtime_seconds": 0.0,
                "node_count": 100.0 if profile_code == "A" else 50.0,
            },
        },
    )
    if malformed_progress:
        (directory / "progress.jsonl").write_text("{not-json}\n", encoding="utf-8")
    elif include_progress:
        write_jsonl_exclusive(directory / "progress.jsonl", events)
    return directory


def _baseline_events() -> tuple[dict[str, object], ...]:
    return (
        {
            "event": "mip_progress",
            "runtime_seconds": 0.0,
            "incumbent": None,
            "bound": 50.0,
        },
        {
            "event": "incumbent",
            "runtime_seconds": 8.0,
            "incumbent": 120.0,
            "bound": 60.0,
        },
        {
            "event": "incumbent",
            "runtime_seconds": 50.0,
            "incumbent": 100.0,
            "bound": 90.0,
        },
        {
            "event": "final",
            "runtime_seconds": 58.0,
            "total_algorithm_runtime_seconds": 60.0,
            "incumbent": 100.0,
            "bound": 100.0,
            "status": 9,
        },
    )


def _candidate_events() -> tuple[dict[str, object], ...]:
    return (
        {
            "event": "mip_progress",
            "runtime_seconds": 0.0,
            "incumbent": None,
            "bound": 60.0,
        },
        {
            "event": "incumbent",
            "runtime_seconds": 6.0,
            "incumbent": 110.0,
            "bound": 70.0,
        },
        {
            "event": "incumbent",
            "runtime_seconds": 20.0,
            "incumbent": 100.0,
            "bound": 80.0,
        },
        {
            "event": "final",
            "runtime_seconds": 56.0,
            "total_algorithm_runtime_seconds": 60.0,
            "incumbent": 90.0,
            "bound": 90.0,
            "status": "TIME_LIMIT",
        },
    )


def test_cli_analyze_uses_common_target_progress_metric_in_primary_gate(
    tmp_path: Path,
) -> None:
    manifest_path, manifest = _formal_manifest(tmp_path)
    loaded = load_benchmark_manifest(manifest_path)
    results_root = tmp_path / "progress-runs"
    for frozen in manifest.cases:
        case_id = frozen.candidate.case_id
        _write_progress_attempt(
            results_root,
            profile_code="A",
            manifest_id=loaded.manifest_id,
            case_id=case_id,
            preparation_seconds=2.0,
            events=_baseline_events(),
        )
        _write_progress_attempt(
            results_root,
            profile_code="E",
            manifest_id=loaded.manifest_id,
            case_id=case_id,
            preparation_seconds=4.0,
            events=_candidate_events(),
        )
    _write_progress_attempt(
        results_root,
        profile_code="B",
        manifest_id=loaded.manifest_id,
        case_id=manifest.cases[0].candidate.case_id,
        preparation_seconds=4.0,
        events=_candidate_events(),
    )

    output = tmp_path / "progress-analysis.json"
    args = build_parser().parse_args(
        [
            "analyze",
            "--manifest",
            str(manifest_path),
            "--results-root",
            str(results_root),
            "--output",
            str(output),
            "--primary-metric",
            "time_to_target",
            "--bootstrap-resamples",
            "100",
        ]
    )
    result = dispatch(args, CLIHooks(now_utc=lambda: _NOW))
    artifact = json.loads(output.read_text(encoding="utf-8"))

    assert result["pair_count"] == 36
    assert result["winner_counts"] == {
        "baseline": 0,
        "candidate": 36,
        "tie": 0,
        "unresolved": 0,
    }
    assert result["success_gate_passed"]
    assert artifact["primary_metric"]["name"] == "time_to_target"
    assert artifact["primary_effect"]["estimate"] > 1.5
    assert artifact["primary_effect_scale"] == (
        "baseline_over_candidate_shifted_geomean"
    )
    assert artifact["complete_time_to_target_pair_count"] == 36
    assert artifact["complete_normalized_pdi_pair_count"] == 36
    assert artifact["complete_proven_runtime_pair_count"] == 0
    assert len(artifact["run_progress_metrics"]) == 73
    assert any(
        run["profile_code"] == "B"
        and run["common_target_reference_profiles"] == ["A", "E"]
        for run in artifact["run_progress_metrics"]
    )
    pair = artifact["pair_progress_metrics"][0]
    assert pair["common_target"] == pytest.approx(100.0)
    assert pair["common_target_rule"] == (
        "maximum_of_paired_final_incumbents_for_minimization"
    )
    assert pair["baseline_time_to_common_target_seconds"] == pytest.approx(52.0)
    assert pair["candidate_time_to_common_target_seconds"] == pytest.approx(24.0)
    assert pair["baseline_time_to_common_target_censored"] is False
    assert pair["candidate_time_to_common_target_censored"] is False
    run = artifact["run_progress_metrics"][0]
    assert run["termination_label"] == "time_limit_with_incumbent"
    assert run["time_limited"] is True
    assert run["proven_optimal_within_horizon"] is False
    assert run["terminal_normalized_gap"] == pytest.approx(0.0)
    assert run["normalized_primal_dual_integral"] is not None


def test_cli_analyze_reports_missing_and_invalid_progress_without_zero_fill(
    tmp_path: Path,
) -> None:
    manifest_path, manifest = _formal_manifest(tmp_path)
    loaded = load_benchmark_manifest(manifest_path)
    results_root = tmp_path / "incomplete-progress-runs"
    first_case = manifest.cases[0].candidate.case_id
    second_case = manifest.cases[1].candidate.case_id
    for case_id in (first_case, second_case):
        _write_progress_attempt(
            results_root,
            profile_code="A",
            manifest_id=loaded.manifest_id,
            case_id=case_id,
            preparation_seconds=2.0,
            events=_baseline_events(),
        )
    _write_progress_attempt(
        results_root,
        profile_code="E",
        manifest_id=loaded.manifest_id,
        case_id=first_case,
        preparation_seconds=4.0,
        events=_candidate_events(),
        include_progress=False,
    )
    _write_progress_attempt(
        results_root,
        profile_code="E",
        manifest_id=loaded.manifest_id,
        case_id=second_case,
        preparation_seconds=4.0,
        events=_candidate_events(),
        malformed_progress=True,
    )

    output = tmp_path / "incomplete-progress-analysis.json"
    args = build_parser().parse_args(
        [
            "analyze",
            "--manifest",
            str(manifest_path),
            "--results-root",
            str(results_root),
            "--output",
            str(output),
            "--primary-metric",
            "normalized_pdi",
            "--bootstrap-resamples",
            "10",
        ]
    )
    dispatch(args, CLIHooks(now_utc=lambda: _NOW))
    artifact = json.loads(output.read_text(encoding="utf-8"))

    candidate_runs = {
        item["pair_id"]: item
        for item in artifact["run_progress_metrics"]
        if item["profile_code"] == "E"
    }
    assert {item["progress_availability"] for item in candidate_runs.values()} == {
        "missing",
        "invalid",
    }
    for item in candidate_runs.values():
        assert item["normalized_primal_dual_integral"] is None
        assert item["time_to_common_target_seconds"] is None
        assert item["time_to_common_target_censored"] is None
        assert item["progress_issues"]
    reasons = {
        item["common_target_unavailable_reason"]
        for item in artifact["pair_progress_metrics"]
    }
    assert reasons == {"candidate_progress_missing", "candidate_progress_invalid"}
    assert artifact["complete_normalized_pdi_pair_count"] == 0
    assert artifact["primary_effect"] is None
    assert (
        "only 0 complete physical-case clusters"
        in artifact["primary_effect_unavailable_reason"]
    )
    assert all(
        assessment["candidate_value"] is None
        for assessment in artifact["pair_assessments"]
    )
