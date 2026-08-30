from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from src.model_contract import FORMULATION_ID, RESULT_SCHEMA_VERSION
from src.utils.paper_experiment_ledger import build_paper_experiment_ledger


def _sha(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


def _write_run(root: Path, name: str, *, complete: bool = True) -> Path:
    run = root / "core" / name
    mode_dir = run / "MX"
    mode_dir.mkdir(parents=True)
    canonical = {"metadata.json": "1" * 64, "sites.csv": "2" * 64}
    signature = "a" * 64
    config: dict[str, Any] = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "formulation_id": FORMULATION_ID,
        "run_signature_sha256": signature,
        "experiment_id": "paper-main",
        "scenario_id": name,
        "modes": ["MX"],
        "git": {"head": "deadbeef", "branch": "User_choice"},
        "implementation": {"sha256": "b" * 64},
        "instance": {
            "canonical_file_sha256": canonical,
            "metadata": {
                "demand_orders_in_area": 1400,
                "total_demand_per_hour": 200.0,
                "od_market_count": 50,
                "nested_scale": {
                    "family_id": "family-1",
                    "crop_side_km": 9.0,
                    "crop_area_km2": 81.0,
                },
            },
        },
        "demand_scale": 1.0,
        "screening_is_exact": {"MX": True},
        "model": {"battery_range_km": 10.0, "drone_speed_mps": 20.0},
        "solver": {
            "time_limit": 900.0,
            "mip_gap": 0.01,
            "threads": 4,
            "seed": 43,
            "root_relaxation_method": 2,
            "miqcp_method": -1,
        },
    }
    summary: dict[str, Any] = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "formulation_id": FORMULATION_ID,
        "run_signature_sha256": signature,
        "mode": "MX",
        "status": 2,
        "status_name": "OPTIMAL",
        "objective": 900.0,
        "objective_bound": 895.5,
        "mip_gap": 0.005,
        "runtime_seconds": 80.0,
        "screening_is_exact": True,
        "courier_only_baseline": 1000.0,
        "cost_saving": 100.0,
        "cost_saving_percent": 10.0,
        "costs_cny_per_hour": {
            "optimized_total": 900.0,
            "facility": 10.0,
            "fleet": 20.0,
            "direct_courier_fallback": 700.0,
        },
        "demand_funnel_orders_per_hour": {
            "total_demand": 200.0,
            "total_drone_offered": 50.0,
            "total_drone_realized": 40.0,
            "drone_service_fraction_of_demand": 0.2,
            "locker_pickup": 10.0,
            "assisted_home": 30.0,
            "direct_courier_fallback": 160.0,
            "realization_fraction_of_offered": 0.8,
        },
        "facilities": {
            "candidates": {"bases": 2, "lockers": 24, "swap_stations": 3},
            "open": {"bases": 1, "lockers": 12, "swap_stations": 1},
            "active_connections": 30,
        },
        "instance": {"markets": 50, "od_nodes": 20},
        "fleet": {"gamma_drones": 15.0, "transit_drones": 7.0},
        "locker_queue": {
            "maximum_utilization": 0.7,
            "workload_weighted_mean_handling_formula_minutes": 3.0,
        },
        "energy_balance": {
            "total_energy_use_equivalent_drones": 6.0,
            "total_energy_supply_equivalent_drones": 6.0,
            "maximum_constraint_violation": 0.0,
        },
        "solver": {"search_statistics": {"work_units": 12.5}},
    }
    (run / "run_config.json").write_text(json.dumps(config), encoding="utf-8")
    (mode_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (run / "validation.json").write_text(
        json.dumps({"valid": True}), encoding="utf-8"
    )
    if complete:
        (run / "run_complete.json").write_text(
            json.dumps({"completed": True, "modes": ["MX"]}), encoding="utf-8"
        )
    return run


def test_builds_deterministic_mode_and_scenario_ledgers(tmp_path: Path) -> None:
    root = tmp_path / "paper_experiment"
    _write_run(root, "H24_base")

    outputs = build_paper_experiment_ledger(root)
    first_bytes = {
        path.name: path.read_bytes()
        for path in (outputs.run_ledger, outputs.scenario_summary, outputs.manifest)
    }
    outputs_again = build_paper_experiment_ledger(root)

    assert outputs.row_count == 1
    assert outputs.run_count == 1
    assert first_bytes == {
        path.name: path.read_bytes()
        for path in (
            outputs_again.run_ledger,
            outputs_again.scenario_summary,
            outputs_again.manifest,
        )
    }
    with outputs.run_ledger.open(encoding="utf-8", newline="") as handle:
        row = next(csv.DictReader(handle))
    assert row["run_path"] == "core/H24_base"
    assert row["scenario_id"] == "H24_base"
    assert row["mode"] == "MX"
    assert row["input_sha256"] == _sha(
        {"metadata.json": "1" * 64, "sites.csv": "2" * 64}
    )
    assert row["mip_gap_percent"] == "0.5"
    assert row["work_units"] == "12.5"
    assert row["validation_pass"] == "true"
    with outputs.scenario_summary.open(encoding="utf-8", newline="") as handle:
        scenario = next(csv.DictReader(handle))
    assert scenario["mx_gap_within_one_percent"] == "true"
    assert scenario["all_validations_pass"] == "true"
    manifest = json.loads(outputs.manifest.read_text(encoding="utf-8"))
    assert manifest["mode_row_count"] == 1
    assert manifest["run_count"] == 1
    assert manifest["outputs"]["run_ledger.csv"] == hashlib.sha256(
        outputs.run_ledger.read_bytes()
    ).hexdigest()


def test_rejects_incomplete_run_instead_of_silently_skipping_it(tmp_path: Path) -> None:
    root = tmp_path / "paper_experiment"
    _write_run(root, "complete")
    _write_run(root, "interrupted", complete=False)

    with pytest.raises(ValueError, match="incomplete run.*run_complete.json"):
        build_paper_experiment_ledger(root)
