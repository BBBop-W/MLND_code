from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from src.model_contract import FORMULATION_ID, RESULT_SCHEMA_VERSION
from src.utils.parameter_report import (
    _require_current_contract,
    _require_result_binding,
    collect_parameter_results,
)


RUN_SIGNATURE = "a" * 64
IMPLEMENTATION_SHA256 = "b" * 64


def _run_config(*, modes: list[str] | None = None) -> dict[str, Any]:
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "formulation_id": FORMULATION_ID,
        "run_signature_sha256": RUN_SIGNATURE,
        "modes": ["MX"] if modes is None else modes,
        "implementation": {"sha256": IMPLEMENTATION_SHA256},
        "model": {},
        "model_derived": {},
        "solver": {},
        "instance": {},
    }


def _summary(mode: str = "MX") -> dict[str, Any]:
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "formulation_id": FORMULATION_ID,
        "run_signature_sha256": RUN_SIGNATURE,
        "mode": mode,
        "objective": 90.0,
        "demand_funnel_orders_per_hour": {
            "total_demand": 10.0,
            "total_drone_offered": 6.0,
            "total_drone_realized": 5.0,
            "locker_pickup": 2.0,
            "assisted_home": 3.0,
            "direct_courier_fallback": 5.0,
        },
        "energy_balance": {"maximum_constraint_violation": 0.0},
    }


def _write_scenario(
    root: Path,
    name: str,
    *,
    modes: list[str] | None = None,
) -> Path:
    scenario = root / name
    run_config = _run_config(modes=modes)
    scenario.mkdir(parents=True)
    (scenario / "run_config.json").write_text(
        json.dumps(run_config),
        encoding="utf-8",
    )
    for mode in run_config["modes"]:
        mode_dir = scenario / mode
        mode_dir.mkdir()
        (mode_dir / "summary.json").write_text(
            json.dumps(_summary(mode)),
            encoding="utf-8",
        )
    return scenario


def test_parameter_binding_accepts_current_bound_mx_summary() -> None:
    assert _require_result_binding(_run_config(), _summary(), "MX") == RUN_SIGNATURE


@pytest.mark.parametrize("schema_version", [None, 3, 5.0, True])
def test_parameter_contract_requires_integer_schema_version(
    schema_version: object,
) -> None:
    payload = {
        "schema_version": schema_version,
        "formulation_id": FORMULATION_ID,
    }
    with pytest.raises(ValueError, match="legacy/incompatible summary"):
        _require_current_contract(payload, "summary")


@pytest.mark.parametrize(
    ("target", "field", "value", "match"),
    [
        ("run", "run_signature_sha256", None, "run_config.run_signature_sha256"),
        ("summary", "run_signature_sha256", None, "summary.run_signature_sha256"),
        ("summary", "run_signature_sha256", "c" * 64, "disagrees"),
        ("summary", "run_signature_sha256", "A" * 64, "lowercase sha256"),
        ("summary", "mode", None, "missing text field 'mode'"),
        ("summary", "mode", "HD", "disagrees with directory"),
        ("run", "modes", None, "nonempty JSON array"),
        ("run", "modes", ["HD"], "not declared"),
    ],
)
def test_parameter_binding_rejects_missing_or_mismatched_fields(
    target: str,
    field: str,
    value: object,
    match: str,
) -> None:
    run_config = _run_config()
    summary = _summary()
    (run_config if target == "run" else summary)[field] = value
    with pytest.raises(ValueError, match=match):
        _require_result_binding(run_config, summary, "MX")


def test_parameter_collection_binds_every_configured_summary(tmp_path: Path) -> None:
    scenario = _write_scenario(
        tmp_path,
        "all_modes",
        modes=["HD", "MX", "LO"],
    )
    lo_path = scenario / "LO" / "summary.json"
    lo_summary = json.loads(lo_path.read_text(encoding="utf-8"))
    lo_summary["run_signature_sha256"] = "c" * 64
    lo_path.write_text(json.dumps(lo_summary), encoding="utf-8")

    with pytest.raises(ValueError, match="LO.*run signature disagrees"):
        collect_parameter_results(tmp_path)


def test_parameter_collection_requires_every_configured_summary(tmp_path: Path) -> None:
    scenario = _write_scenario(
        tmp_path,
        "incomplete",
        modes=["HD", "MX", "LO"],
    )
    (scenario / "LO" / "summary.json").unlink()

    with pytest.raises(ValueError, match="LO/summary.json"):
        collect_parameter_results(tmp_path)


def test_parameter_collection_rejects_undeclared_summary_directory(
    tmp_path: Path,
) -> None:
    scenario = _write_scenario(tmp_path, "stale_summary")
    hd_dir = scenario / "HD"
    hd_dir.mkdir()
    (hd_dir / "summary.json").write_text(
        json.dumps(_summary("HD")),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="not declared in run_config.modes"):
        collect_parameter_results(tmp_path)


def test_parameter_collection_returns_bound_signature(tmp_path: Path) -> None:
    _write_scenario(tmp_path, "valid")

    rows = collect_parameter_results(tmp_path)

    assert len(rows) == 1
    assert rows[0]["run_signature_sha256"] == RUN_SIGNATURE
    assert rows[0]["summary_schema_version"] == RESULT_SCHEMA_VERSION
    assert rows[0]["formulation_id"] == FORMULATION_ID
