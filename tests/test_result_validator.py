from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from src.model_contract import FORMULATION_ID, RESULT_SCHEMA_VERSION
from src.utils.result_validator import _payload_sha256, main, validate_run


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    assert rows
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _seal(root: Path) -> None:
    artifacts = []
    for path in sorted(root.rglob("*")):
        if (
            not path.is_file()
            or path.name in {"run_complete.json", "validation.json", "validation_checks.csv"}
        ):
            continue
        artifacts.append(
            {
                "path": path.relative_to(root).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    summary = json.loads((root / "MX" / "summary.json").read_text())
    payload = {
        "schema_version": 1,
        "completed": True,
        "started_at_utc": "2026-08-31T00:00:00+00:00",
        "completed_at_utc": "2026-08-31T00:01:00+00:00",
        "modes": ["MX"],
        "mode_status": {
            "MX": {
                "status": summary["status_name"],
                "objective": summary["objective"],
                "objective_bound": summary["objective_bound"],
                "mip_gap": summary["mip_gap"],
                "runtime_seconds": summary["runtime_seconds"],
                "screening_is_exact": summary["screening_is_exact"],
            }
        },
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
    }
    (root / "run_complete.json").write_text(json.dumps(payload), encoding="utf-8")


def _make_package(tmp_path: Path) -> Path:
    root = tmp_path / "run"
    mode_root = root / "MX"
    instance = tmp_path / "instance"
    mode_root.mkdir(parents=True)
    instance.mkdir()
    canonical_hashes = {}
    for filename in ("metadata.json", "sites.csv", "od_nodes.csv", "od_demand.csv", "periods.csv"):
        path = instance / filename
        path.write_text(f"canonical:{filename}\n", encoding="utf-8")
        canonical_hashes[filename] = _sha256(path)

    formulation_source = {"path": "/frozen/main.tex", "sha256": "b" * 64}
    implementation = {"sha256": "a" * 64, "files": {"main.py": "c" * 64}}
    signature_payload = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "formulation_id": FORMULATION_ID,
        "formulation_source": formulation_source,
        "implementation": implementation,
        "instance_dir": str(instance),
        "canonical_file_sha256": canonical_hashes,
        "demand_scale": 1.0,
        "modes": ["MX"],
        "cross_mode_start_policy": "none",
        "warm_start_from": None,
        "file_warm_start_policy": "first_mode_only",
        "screening_is_exact": {"MX": False},
        "model": {"availability_levels": [0.75]},
        "model_derived": {"usable_battery_time_hours": 5.0 / 36.0},
        "solver": {"mip_gap": 0.02},
    }
    signature = _payload_sha256(signature_payload)
    run_config = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "formulation_id": FORMULATION_ID,
        "run_signature_sha256": signature,
        "formulation_source": formulation_source,
        "implementation": implementation,
        "instance": {
            "directory": str(instance),
            "metadata": {"source": "test"},
            "canonical_file_sha256": canonical_hashes,
        },
        "demand_scale": 1.0,
        "modes": ["MX"],
        "cross_mode_start_policy": "none",
        "warm_start_from": None,
        "file_warm_start_policy": "first_mode_only",
        "screening_is_exact": {"MX": False},
        "model": {"availability_levels": [0.75]},
        "model_derived": {"usable_battery_time_hours": 5.0 / 36.0},
        "solver": {"mip_gap": 0.02},
    }
    (root / "run_config.json").write_text(json.dumps(run_config), encoding="utf-8")

    objective = 53.96
    bound = 53.0
    gap = abs(objective - bound) / objective
    costs = {
        "facility": 10.0,
        "capacity": 1.0,
        "connection": 2.0,
        "route_operation": 3.0,
        "fleet": 5.0,
        "direct_courier_fallback": 25.6,
        "first_mile_courier": 3.0,
        "last_mile_courier": 4.0,
        "locker_handling": 0.36,
        "optimized_total": objective,
        "objective_reconciliation_error": 0.0,
    }
    summary = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "formulation_id": FORMULATION_ID,
        "run_signature_sha256": signature,
        "mode": "MX",
        "status": 9,
        "status_name": "TIME_LIMIT",
        "objective": objective,
        "objective_bound": bound,
        "mip_gap": gap,
        "requested_mip_gap": 0.02,
        "runtime_seconds": 60.0,
        "courier_only_baseline": 64.0,
        "cost_saving": 10.04,
        "cost_saving_percent": 15.6875,
        "drone_served_rate": 0.6,
        "fleet_size": 2.0,
        "screening_is_exact": False,
        "solver": {
            "objective_cost_per_hour": objective,
            "objective_bound": bound,
            "mip_gap": gap,
            "requested_mip_gap": 0.02,
            "runtime_seconds": 60.0,
        },
        "instance": {"markets": 1, "screening_is_exact": False},
        "costs_cny_per_hour": costs,
        "paper_objective_terms_cny_per_hour": {
            "courier_baseline": 64.0,
            "locker_pickup_adjustment": -0.36,
            "realized_drone_adjustment": -30.68,
            "network_design_and_operation": 21.0,
            "reconstructed_total": objective,
            "objective_reconciliation_error": 0.0,
        },
        "all_rider_baseline": {"cost_per_hour": 64.0},
        "demand_funnel_orders_per_hour": {
            "total_demand": 10.0,
            "total_drone_offered": 8.0,
            "total_drone_realized": 6.0,
            "locker_pickup": 1.2,
            "assisted_home": 4.8,
            "direct_courier_fallback": 4.0,
        },
        "reconciliation": {
            "maximum_od_service_conservation_error": 0.0,
            "maximum_corridor_offered_flow_error": 0.0,
            "maximum_corridor_realized_flow_error": 0.0,
            "maximum_epsilon_product_error": 0.0,
            "maximum_energy_constraint_violation": 0.0,
        },
        "energy_balance": {
            "total_energy_use_equivalent_drones": 2.0,
            "total_energy_supply_equivalent_drones": 25.0 / 9.0,
            "base_full_battery_departures_per_hour": 20.0,
            "swap_events_per_hour": 0.0,
            "battery_supply_events_per_hour": 20.0,
            "maximum_constraint_violation": 0.0,
        },
    }
    (mode_root / "summary.json").write_text(json.dumps(summary), encoding="utf-8")

    _write_csv(
        mode_root / "od_service.csv",
        [
            {
                "origin": "O1",
                "destination": "D1",
                "demand_per_hour": 10.0,
                "drone_offered_per_hour": 8.0,
                "drone_realized_per_hour": 6.0,
                "offered_not_realized_per_hour": 2.0,
                "not_offered_per_hour": 2.0,
                "locker_pickup_per_hour": 1.2,
                "assisted_home_per_hour": 4.8,
                "direct_courier_fallback_per_hour": 4.0,
                "service_conservation_error": 0.0,
                "drone_service_rate": 0.6,
                "direct_rider_unit_cost": 6.4,
                "all_rider_baseline_cost_per_hour": 64.0,
                "direct_courier_fallback_cost_per_hour": 25.6,
                "first_mile_courier_cost_per_hour": 3.0,
                "last_mile_courier_cost_per_hour": 4.0,
                "locker_handling_cost_per_hour": 0.36,
            }
        ],
    )
    _write_csv(
        mode_root / "od_corridor_reliability.csv",
        [
            {
                "origin": "O1",
                "destination": "D1",
                "origin_locker": "H1",
                "destination_locker": "H2",
                "rho_level": 1,
                "rho": 0.75,
                "chi_offered_per_hour": 8.0,
                "realized_per_hour": 6.0,
                "displayed_for_destination": 1,
                "conditional_locker_share_eta": 0.2,
                "locker_pickup_per_hour": 1.2,
                "assisted_home_per_hour": 4.8,
            }
        ],
    )
    _write_csv(
        mode_root / "displayed_locker_service.csv",
        [
            {
                "origin": "O1",
                "destination": "D1",
                "locker": "H2",
                "alpha": 1.0,
                "conditional_locker_share_eta": 0.2,
                "successful_to_locker_per_hour": 6.0,
                "epsilon_per_hour": 6.0,
                "alpha_times_successful_per_hour": 6.0,
                "epsilon_product_error": 0.0,
                "locker_pickup_per_hour": 1.2,
            }
        ],
    )
    _write_csv(
        mode_root / "corridor_flows.csv",
        [
            {
                "origin_locker": "H1",
                "destination_locker": "H2",
                "od_offered_per_hour": 8.0,
                "source_allocated_offered_per_hour": 8.0,
                "offered_reconciliation_error": 0.0,
                "realized_from_chi_per_hour": 6.0,
                "loaded_psi_per_hour": 6.0,
                "realized_reconciliation_error": 0.0,
            }
        ],
    )
    _write_csv(
        mode_root / "energy_balance.csv",
        [
            {
                "locker_handling_population_drones": 1.0,
                "route_transit_and_handling_population_drones": 1.0,
                "energy_use_equivalent_drones": 2.0,
                "base_full_battery_departures_per_hour": 20.0,
                "swap_events_per_hour": 0.0,
                "battery_supply_events_per_hour": 20.0,
                "usable_battery_time_hours": 5.0 / 36.0,
                "energy_supply_equivalent_drones": 25.0 / 9.0,
                "energy_slack_equivalent_drones": 7.0 / 9.0,
                "energy_violation_equivalent_drones": 0.0,
            }
        ],
    )
    _write_csv(
        root / "mode_comparison.csv",
        [
            {
                "mode": "MX",
                "run_signature_sha256": signature,
                "status_name": "TIME_LIMIT",
                "objective": objective,
                "courier_only_baseline": 64.0,
                "cost_saving": 10.04,
                "runtime_seconds": 60.0,
                "mip_gap": gap,
                "requested_mip_gap": 0.02,
                "screening_is_exact": False,
            }
        ],
    )
    _seal(root)
    return root


def test_valid_package_writes_validation_artifacts(tmp_path: Path) -> None:
    root = _make_package(tmp_path)
    report = validate_run(root)

    assert report["valid"] is True
    assert report["check_counts"]["fail"] == 0
    assert report["interpretation_scope"] == "screened_candidate_model"
    assert report["eligible_for_unscreened_exactness_claim"] is False
    assert (root / "validation.json").is_file()
    assert (root / "validation_checks.csv").is_file()
    assert main([str(root)]) == 0


@pytest.mark.parametrize("tamper", ["cost", "od"])
def test_semantic_tampering_is_rejected_even_after_resealing(
    tmp_path: Path, tamper: str
) -> None:
    root = _make_package(tmp_path)
    if tamper == "cost":
        path = root / "MX" / "summary.json"
        summary = json.loads(path.read_text())
        summary["costs_cny_per_hour"]["facility"] += 1.0
        path.write_text(json.dumps(summary), encoding="utf-8")
    else:
        path = root / "MX" / "od_service.csv"
        fields, row = None, None
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            fields = reader.fieldnames
            row = next(reader)
        assert fields is not None and row is not None
        row["direct_courier_fallback_per_hour"] = "3.0"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerow(row)
    _seal(root)

    report = validate_run(root)

    assert report["valid"] is False
    assert report["check_counts"]["fail"] >= 1


def test_hash_tampering_is_rejected(tmp_path: Path) -> None:
    root = _make_package(tmp_path)
    path = root / "MX" / "od_service.csv"
    path.write_text(path.read_text() + "\n", encoding="utf-8")

    report = validate_run(root)

    assert report["valid"] is False
    failed = {check["check_id"] for check in report["checks"] if check["status"] == "FAIL"}
    assert "artifact_hashes" in failed
    assert main([str(root)]) == 1
