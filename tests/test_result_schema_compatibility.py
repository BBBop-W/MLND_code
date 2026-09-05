"""Behavior at the historical-mathematics/current-telemetry boundary."""

from __future__ import annotations

from copy import deepcopy

import pytest

from src.model_contract import (
    FORMULATION_ID,
    require_result_contract,
    require_same_result_schema,
    require_validation_binding,
    result_contract_metadata,
    runtime_contract,
)
from src.utils.instance_plotter import _require_current_summary_contract
from src.utils.nanshan_benchmark_report import _validate_runtime_accounting
from src.utils.parameter_report import _require_current_contract as parameter_contract
from src.utils.scale_report import _require_current_contract as scale_contract


def _summary(version: int, total: float = 13.0) -> dict:
    return {
        "schema_version": version,
        "formulation_id": FORMULATION_ID,
        "runtime_seconds": total,
        "run_signature_sha256": "a" * 64,
        "solver": {
            "main_solve_runtime_seconds": 10.0,
            "network_start_runtime_seconds": 3.0,
        },
    }


@pytest.mark.parametrize("version", [5, 6])
def test_readers_preserve_original_version_and_missing_telemetry(version: int) -> None:
    summary = _summary(version)
    original = deepcopy(summary)
    assert require_result_contract(summary, "summary") == version
    _require_current_summary_contract(summary)
    parameter_contract(summary, "summary")
    scale_contract(summary, "summary")
    assert summary == original
    assert "installed_mip_start_count" not in summary["solver"]
    assert result_contract_metadata([version])["schema_version"] == version


@pytest.mark.parametrize("version", [None, True, 4, 7, 5.0, 6.0, "5"])
def test_contract_rejects_coerced_and_unrecognized_versions(version: object) -> None:
    summary = {**_summary(5), "schema_version": version}
    with pytest.raises(ValueError, match="incompatible"):
        require_result_contract(summary, "summary")


def test_same_schema_cannot_make_another_formulation_compatible() -> None:
    summary = {**_summary(5), "formulation_id": "mlnd-legacy"}
    with pytest.raises(ValueError, match="incompatible"):
        require_result_contract(summary, "summary")


def test_copied_signature_does_not_allow_cross_version_run_assembly() -> None:
    with pytest.raises(ValueError, match="schema disagrees"):
        require_same_result_schema(_summary(5), _summary(6), "summary")
    with pytest.raises(ValueError, match="validation result contract"):
        require_validation_binding(
            {
                "result_schema_version": 6,
                "formulation_id": FORMULATION_ID,
                "run_signature_sha256": "a" * 64,
            },
            _summary(5),
            "validation",
        )


def test_runtime_contracts_do_not_turn_wall_overhead_into_solver_phase_time() -> None:
    assert runtime_contract(5) != runtime_contract(6)
    _validate_runtime_accounting(_summary(5), "historical")
    with pytest.raises(ValueError, match="runtime phases"):
        _validate_runtime_accounting(_summary(5, 17.0), "historical")
    current = _summary(6, 17.0)
    current["solver"].update(
        {
            "algorithm_prepare_runtime_seconds": 2.0,
            "start_preparation_runtime_seconds": 4.0,
            "network_construction_runtime_seconds": 3.5,
        }
    )
    _validate_runtime_accounting(current, "current")
    # These are overlapping subphases, deliberately not an additive partition.
    current["solver"]["start_preparation_runtime_seconds"] = 9.0
    _validate_runtime_accounting(current, "current")
    current["solver"]["main_solve_runtime_seconds"] = 20.0
    with pytest.raises(ValueError, match="exceeds algorithm wall"):
        _validate_runtime_accounting(current, "current")


@pytest.mark.parametrize("value", [-1.0, float("nan"), float("inf")])
def test_current_runtime_rejects_invalid_phase_measurements(value: float) -> None:
    summary = _summary(6)
    summary["solver"]["main_solve_runtime_seconds"] = value
    with pytest.raises(ValueError):
        _validate_runtime_accounting(summary, "current")


def test_mixed_source_manifest_never_claims_a_single_current_schema() -> None:
    contract = result_contract_metadata([6, 5, 5])
    assert contract["schema_version"] is None
    assert contract["schema_versions"] == [5, 6]
    assert contract["runtime_contracts"] == {
        "5": "network_start_plus_main_solver",
        "6": "algorithm_wall_excluding_model_build",
    }
