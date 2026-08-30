from __future__ import annotations

import csv
import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from src.model_contract import FORMULATION_ID, RESULT_SCHEMA_VERSION
from src.utils import paper_experiment_report as report
from src.utils.paper_experiment_ledger import build_paper_experiment_ledger


_BASELINE_MODEL: dict[str, object] = {
    "display_origin_choices": 8,
    "general_corridor_choices": 10,
    "choice": {"locker_discount_cny": 1.0},
    "rider_base_cost": 6.4,
    "rider_incremental_cost_per_km": 1.8,
    "base_cost_per_hour": 14.0,
    "locker_fixed_cost_per_hour": 1.0,
    "swap_cost_per_hour": 3.0,
    "drone_cost_per_hour": 5.0,
}

_BASELINE_SOLVER: dict[str, object] = {
    "time_limit": 300.0,
    "mip_gap": 0.01,
    "threads": 4,
    "seed": 43,
    "root_relaxation_method": 2,
    "miqcp_method": -1,
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _ledger_row(
    run_path: str,
    *,
    mode: str = "MX",
    gap: float = 0.008,
    validation: bool = True,
    screening_is_exact: bool = False,
    h: int = 24,
    instance_key: str | None = None,
    model_updates: dict[str, object] | None = None,
    solver_miqcp_method: int = -1,
    solver_seed: int = 43,
    solver_threads: int = 4,
    solver_time_limit_seconds: float = 300.0,
) -> dict[str, object]:
    objective = 32000 + {"HD": 100, "MX": 0, "LO": 200}[mode]
    return {
        "run_path": run_path,
        "scenario_id": run_path.replace("/", "_"),
        "mode": mode,
        "schema_version": RESULT_SCHEMA_VERSION,
        "formulation_id": FORMULATION_ID,
        "candidate_lockers": h,
        "crop_side_km": 9,
        "crop_area_km2": 81,
        "demand_per_hour": 4906.857,
        "od_markets": 6445,
        "objective_cny_per_hour": objective,
        "objective_bound_cny_per_hour": objective * (1.0 - gap),
        "mip_gap_fraction": gap,
        "runtime_seconds": 220,
        "cost_saving_percent": 2.1,
        "drone_service_fraction": 0.06,
        "open_bases": 2,
        "open_lockers": 21,
        "open_swap_stations": 1,
        "fleet_size": 62,
        "screening_is_exact": str(screening_is_exact).lower(),
        "validation_pass": str(validation).lower(),
        "_instance_key": instance_key or f"H{h}",
        "_model_updates": dict(model_updates or {}),
        "_solver_miqcp_method": solver_miqcp_method,
        "_solver_seed": solver_seed,
        "_solver_threads": solver_threads,
        "_solver_time_limit_seconds": solver_time_limit_seconds,
    }


def _write_instance(root: Path, key: str, h: int) -> tuple[Path, dict[str, str]]:
    instance_dir = root / "instances" / key
    instance_dir.mkdir(parents=True, exist_ok=True)
    metadata = instance_dir / "metadata.json"
    sites = instance_dir / "sites.csv"
    od_nodes = instance_dir / "od_nodes.csv"
    od_demand = instance_dir / "od_demand.csv"
    periods = instance_dir / "periods.csv"
    if not metadata.exists():
        metadata.write_text(
            json.dumps(
                {
                    "fixture_instance": key,
                    "candidate_lockers": h,
                    "coordinate_system": "fixture-local-km",
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    if not sites.exists():
        sites.write_text(
            "site_id,site_type,x_km,y_km\n" f"{key}-L1,locker,0.0,0.0\n",
            encoding="utf-8",
        )
    if not od_nodes.exists():
        od_nodes.write_text(
            "node_id,x_km,y_km\nO1,0.0,0.0\nD1,1.0,0.0\n", encoding="utf-8"
        )
    if not od_demand.exists():
        od_demand.write_text(
            "origin_id,destination_id,demand\nO1,D1,1.0\n", encoding="utf-8"
        )
    if not periods.exists():
        periods.write_text("period_id,hours\nP1,1.0\n", encoding="utf-8")
    return instance_dir, {
        filename: _sha256(instance_dir / filename)
        for filename in (
            "metadata.json",
            "sites.csv",
            "od_nodes.csv",
            "od_demand.csv",
            "periods.csv",
        )
    }


def _apply_model_updates(
    baseline: dict[str, object], updates: dict[str, object]
) -> dict[str, object]:
    model = deepcopy(baseline)
    paths = {
        "rider_initial_charge_cny": ("rider_base_cost",),
        "rider_extra_cost_cny_per_km": ("rider_incremental_cost_per_km",),
        "locker_discount_cny": ("choice", "locker_discount_cny"),
        "base_cost_cny_per_hour": ("base_cost_per_hour",),
        "locker_fixed_cost_cny_per_hour": ("locker_fixed_cost_per_hour",),
        "swap_cost_cny_per_hour": ("swap_cost_per_hour",),
        "drone_cost_cny_per_hour": ("drone_cost_per_hour",),
    }
    for key, value in updates.items():
        path = paths.get(key, (key,))
        current: dict[str, object] = model
        for part in path[:-1]:
            child = current.get(part)
            if not isinstance(child, dict):
                child = {}
                current[part] = child
            current = child
        current[path[-1]] = None if value == "all" else value
    return model


def _write_run_sources(
    root: Path, run_path: str, rows: list[dict[str, object]]
) -> None:
    first = rows[0]
    h = int(first["candidate_lockers"])
    instance_key = str(first["_instance_key"])
    _instance_dir, canonical_hashes = _write_instance(root, instance_key, h)
    model = _apply_model_updates(_BASELINE_MODEL, dict(first["_model_updates"]))
    solver = deepcopy(_BASELINE_SOLVER)
    solver.update(
        {
            "miqcp_method": first["_solver_miqcp_method"],
            "seed": first["_solver_seed"],
            "threads": first["_solver_threads"],
            "time_limit": first["_solver_time_limit_seconds"],
        }
    )
    modes = [
        mode for mode in ("HD", "MX", "LO") if any(row["mode"] == mode for row in rows)
    ]
    signature = hashlib.sha256(run_path.encode("utf-8")).hexdigest()
    run_dir = root / run_path
    run_dir.mkdir(parents=True, exist_ok=True)
    run_config: dict[str, Any] = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "formulation_id": FORMULATION_ID,
        "run_signature_sha256": signature,
        "experiment_id": "paper-report-fixture",
        "scenario_id": first["scenario_id"],
        "modes": modes,
        "git": {"head": "d" * 40, "branch": "fixture"},
        "implementation": {"sha256": "b" * 64},
        "instance": {
            "directory": str(_instance_dir.resolve()),
            "canonical_file_sha256": canonical_hashes,
            "metadata": {
                "demand_orders_in_area": 34348,
                "total_demand_per_hour": first["demand_per_hour"],
                "od_market_count": first["od_markets"],
                "nested_scale": {
                    "family_id": "fixture-scale-family",
                    "crop_side_km": first["crop_side_km"],
                    "crop_area_km2": first["crop_area_km2"],
                    "facility_actual_counts": {
                        "H": h,
                        "L": 2,
                        "E": 2 if h <= 11 else 3,
                    },
                },
            },
        },
        "demand_scale": 1.0,
        "screening_is_exact": {
            str(row["mode"]): row["screening_is_exact"] == "true" for row in rows
        },
        "model": model,
        "solver": solver,
    }
    (run_dir / "run_config.json").write_text(
        json.dumps(run_config, ensure_ascii=False, sort_keys=True), encoding="utf-8"
    )
    for row in rows:
        mode = str(row["mode"])
        mode_dir = run_dir / mode
        mode_dir.mkdir(parents=True, exist_ok=True)
        objective = float(row["objective_cny_per_hour"])
        saving_percent = float(row["cost_saving_percent"])
        courier_baseline = objective / (1.0 - saving_percent / 100.0)
        demand = float(row["demand_per_hour"])
        drone_fraction = float(row["drone_service_fraction"])
        drone_demand = demand * drone_fraction
        summary = {
            "schema_version": RESULT_SCHEMA_VERSION,
            "formulation_id": FORMULATION_ID,
            "run_signature_sha256": signature,
            "mode": mode,
            "status": 2,
            "status_name": "OPTIMAL",
            "objective": objective,
            "objective_bound": row["objective_bound_cny_per_hour"],
            "mip_gap": row["mip_gap_fraction"],
            "runtime_seconds": row["runtime_seconds"],
            "screening_is_exact": row["screening_is_exact"] == "true",
            "courier_only_baseline": courier_baseline,
            "cost_saving": courier_baseline - objective,
            "cost_saving_percent": saving_percent,
            "costs_cny_per_hour": {
                "optimized_total": objective,
                "facility": 100.0,
                "fleet": 500.0,
                "direct_courier_fallback": objective - 600.0,
            },
            "demand_funnel_orders_per_hour": {
                "total_demand": demand,
                "total_drone_offered": drone_demand,
                "total_drone_realized": drone_demand,
                "drone_service_fraction_of_demand": drone_fraction,
                "locker_pickup": drone_demand / 2.0,
                "assisted_home": drone_demand / 2.0,
                "direct_courier_fallback": demand - drone_demand,
                "realization_fraction_of_offered": 1.0,
            },
            "facilities": {
                "candidates": {"bases": 2, "lockers": h, "swap_stations": 3},
                "open": {
                    "bases": row["open_bases"],
                    "lockers": row["open_lockers"],
                    "swap_stations": row["open_swap_stations"],
                },
                "active_connections": 30,
            },
            "instance": {"markets": row["od_markets"], "od_nodes": 200},
            "fleet": {
                "gamma_drones": row["fleet_size"],
                "transit_drones": 7.0,
            },
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
        (mode_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, sort_keys=True), encoding="utf-8"
        )
    validations = {row["validation_pass"] for row in rows}
    if len(validations) != 1:
        raise AssertionError("fixture requires one run-level validation result")
    (run_dir / "validation.json").write_text(
        json.dumps({"valid": validations == {"true"}}), encoding="utf-8"
    )
    (run_dir / "run_complete.json").write_text(
        json.dumps({"completed": True, "modes": modes}), encoding="utf-8"
    )


def _write_ledger_bundle(root: Path, rows: list[dict[str, object]]) -> None:
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row["run_path"]), []).append(row)
    for run_path, group in grouped.items():
        _write_run_sources(root, run_path, group)
    build_paper_experiment_ledger(root, root / "ledger")


def _scenario_config(root: Path) -> dict[str, object]:
    return {
        "schema_version": 1,
        "artifact_status": "final_evidence_registry",
        "experiment_root": str(root),
        "repository_commit_required": "d" * 40,
        "instance_family": {
            "family_id": "fixture-scale-family",
            "required_solve_demand_scale": 1.0,
        },
        "baseline": {
            "formulation_id": FORMULATION_ID,
            "result_schema_version": RESULT_SCHEMA_VERSION,
            "model": deepcopy(_BASELINE_MODEL),
            "solver": {
                "target_mip_gap_fraction": 0.01,
                "threads": 4,
                "solver_seed": 43,
                "root_method": 2,
                "miqcp_method": -1,
            },
        },
        "ledger": {
            "run_ledger": "ledger/run_ledger.csv",
            "scenario_summary": "ledger/scenario_summary.csv",
            "manifest": "ledger/ledger_manifest.json",
        },
        "roles": {
            "main": {
                "id": "main_h24",
                "label": "H24 正式主算例",
                "run_path": "main/h24",
                "instance_dir": str(root / "instances/H24"),
                "modes": ["HD", "MX", "LO"],
                "role": "selected",
            },
            "scale": [
                {
                    "id": "scale_h24",
                    "label": "H24",
                    "run_path": "scale/h24",
                    "instance_dir": str(root / "instances/H24"),
                    "modes": ["MX"],
                    "scale": {
                        "H": 24,
                        "L": 2,
                        "E": 3,
                        "crop_side_km": 9,
                        "crop_area_km2": 81,
                    },
                    "solver_overrides": {"time_limit_seconds_per_mode": 300},
                },
                {
                    "id": "scale_h43_pending",
                    "label": "H43",
                    "run_path": "scale/h43",
                    "instance_dir": str(root / "instances/H43"),
                    "modes": ["MX"],
                    "scale": {
                        "H": 43,
                        "L": 3,
                        "E": 5,
                        "crop_side_km": 12,
                        "crop_area_km2": 144,
                    },
                    "solver_overrides": {"time_limit_seconds_per_mode": 300},
                    "status": "planned",
                },
            ],
            "algorithm": [
                {
                    "id": "method_auto",
                    "label": "MIQCPMethod=-1",
                    "run_path": "algorithm/auto",
                    "instance_dir": str(root / "instances/H24"),
                    "modes": ["MX"],
                    "solver_overrides": {
                        "miqcp_method": -1,
                        "seed": 43,
                        "threads": 4,
                        "time_limit_seconds": 300,
                    },
                }
            ],
            "exactness": [
                {
                    "id": "h11_screened",
                    "label": "H11 screened",
                    "run_path": "exact/screened",
                    "instance_dir": str(root / "instances/H11"),
                    "modes": ["MX"],
                    "role": "screened",
                },
                {
                    "id": "h11_complete",
                    "label": "H11 complete-index",
                    "run_path": "exact/complete",
                    "instance_dir": str(root / "instances/H11"),
                    "modes": ["MX"],
                    "role": "complete",
                    "model_overrides": {
                        "display_origin_choices": "all",
                        "general_corridor_choices": "all",
                    },
                    "solver_overrides": {"time_limit_seconds": 3600},
                },
            ],
            "sensitivity": [
                {
                    "id": "rider_080",
                    "label": "Rider x0.8",
                    "run_path": "sensitivity/rider080",
                    "instance_dir": str(root / "instances/H24"),
                    "modes": ["MX"],
                    "group": "rider_cost",
                    "factor": 0.8,
                    "model_overrides": {
                        "rider_initial_charge_cny": 5.12,
                        "rider_extra_cost_cny_per_km": 1.44,
                    },
                    "solver_overrides": {"time_limit_seconds": 3600},
                },
                {
                    "id": "rider_120",
                    "label": "Rider x1.2",
                    "run_path": "sensitivity/rider120",
                    "instance_dir": str(root / "instances/H24"),
                    "modes": ["MX"],
                    "group": "rider_cost",
                    "factor": 1.2,
                    "model_overrides": {
                        "rider_initial_charge_cny": 7.68,
                        "rider_extra_cost_cny_per_km": 2.16,
                    },
                    "solver_overrides": {"time_limit_seconds": 3600},
                },
            ],
        },
        "choice_curve": {
            "distances_km": [0, 0.5, 1.0, 1.5],
            "discounts_cny": [0, 1, 2],
            "parameters": {
                "locker_asc": -1,
                "delivery_time_coefficient": 0.06,
                "price_coefficient": 0.35,
                "walk_distance_coefficient": 1.2,
                "home_eta_minutes": 30,
                "locker_eta_minutes": 20,
                "locker_discount_cny": 1,
                "max_walk_km": 1.5,
            },
        },
    }


def _selected_main_role(
    root: Path,
    *,
    run_path: str = "main/h24",
    h: int = 24,
    modes: list[str] | None = None,
    scenario_id: str = "main_h24",
) -> dict[str, object]:
    return {
        "id": scenario_id,
        "label": f"H{h} selected main",
        "run_path": run_path,
        "instance_dir": str(root / f"instances/H{h}"),
        "modes": list(modes or ["MX"]),
        "role": "selected",
    }


def _config_with_roles(
    root: Path,
    roles: dict[str, object],
    *,
    artifact_status: str = "final_evidence_registry",
) -> dict[str, object]:
    config = _scenario_config(root)
    config["artifact_status"] = artifact_status
    config["roles"] = roles
    return config


def _write_config(path: Path, config: dict[str, object]) -> Path:
    path.write_text(
        json.dumps(config, ensure_ascii=False, sort_keys=True), encoding="utf-8"
    )
    return path


def _reseal_manifest_file(
    experiment_root: Path, relative_path: str, *, output: bool = False
) -> None:
    manifest_path = experiment_root / "ledger/ledger_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    target = experiment_root / relative_path
    if output:
        manifest["outputs"][target.name] = _sha256(target)
    else:
        source = next(
            item for item in manifest["source_files"] if item["path"] == relative_path
        )
        source["size_bytes"] = target.stat().st_size
        source["sha256"] = _sha256(target)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True), encoding="utf-8"
    )


def _artifact_result_status(artifact: dict[str, Any], scenario_id: str) -> str:
    statuses = {
        row["evidence_status"]
        for row in artifact["snapshot"]["datasets"]["results"]
        if row["scenario_id"] == scenario_id
    }
    assert len(statuses) == 1
    return statuses.pop()


def test_report_is_dynamic_and_keeps_pending_roles_out_of_claims(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    experiment_root = tmp_path / "experiment"
    experiment_root.mkdir()
    rows = [
        *[
            _ledger_row("main/h24", mode=mode, solver_time_limit_seconds=900)
            for mode in ("HD", "MX", "LO")
        ],
        _ledger_row("scale/h24", gap=0.009),
        _ledger_row("algorithm/auto", gap=0.021),
        _ledger_row("exact/screened", gap=0.007, screening_is_exact=False, h=11),
        _ledger_row(
            "exact/complete",
            gap=0.009,
            screening_is_exact=True,
            h=11,
            model_updates={
                "display_origin_choices": "all",
                "general_corridor_choices": "all",
            },
            solver_time_limit_seconds=3600,
        ),
        _ledger_row(
            "sensitivity/rider080",
            gap=0.008,
            model_updates={
                "rider_initial_charge_cny": 5.12,
                "rider_extra_cost_cny_per_km": 1.44,
            },
            solver_time_limit_seconds=3600,
        ),
        _ledger_row(
            "sensitivity/rider120",
            gap=0.013,
            model_updates={
                "rider_initial_charge_cny": 7.68,
                "rider_extra_cost_cny_per_km": 2.16,
            },
            solver_time_limit_seconds=3600,
        ),
    ]
    _write_ledger_bundle(experiment_root, rows)
    config_path = tmp_path / "scenario_config.json"
    config_path.write_text(
        json.dumps(_scenario_config(experiment_root), ensure_ascii=False),
        encoding="utf-8",
    )

    def fake_map(selected, experiment_root, basemap_cache, stem):  # noqa: ANN001
        png, svg = report._pending_figure(stem, "测试地图", "地图接口已调用")
        return report.MapRender(
            png,
            svg,
            "ready",
            "测试中以占位图替代 OSM 渲染。",
        )

    monkeypatch.setattr(report, "_plot_selected_map", fake_map)
    output = tmp_path / "report"
    outputs = report.write_paper_experiment_report(config_path, output)

    markdown = outputs.markdown.read_text(encoding="utf-8")
    assert "H24 MX 主算例 Gap 为 0.80%" in markdown
    assert "最大达标档为 H24" in markdown
    assert "scale_h43_pending" in markdown
    assert "H120" not in markdown
    assert "不是营业收入或利润" in markdown
    assert "只属于预设筛选模型" in markdown

    artifact = json.loads(outputs.artifact.read_text(encoding="utf-8"))
    assert artifact["surface"] == "report"
    assert artifact["snapshot"]["status"] == "partial"
    result_rows = artifact["snapshot"]["datasets"]["results"]
    pending = next(
        row for row in result_rows if row["scenario_id"] == "scale_h43_pending"
    )
    assert pending["evidence_status"] == "pending"
    scale = artifact["snapshot"]["datasets"]["scale_ready"]
    assert [row["x_value"] for row in scale] == [24.0]

    manifest = json.loads(outputs.manifest.read_text(encoding="utf-8"))
    assert manifest["evidence_state"] == {
        "ready": 6,
        "pilot": 1,
        "pending": 1,
        "invalid": 0,
    }
    assert len(outputs.figures) == 12
    assert all(path.is_file() and path.stat().st_size > 0 for path in outputs.figures)
    assert all(path.is_file() for path in outputs.tables)


def test_role_assignment_uses_exact_explicit_run_path(tmp_path: Path) -> None:
    experiment_root = tmp_path / "experiment"
    experiment_root.mkdir()
    rows = [_ledger_row("formal/main/H24_finished")]
    _write_ledger_bundle(experiment_root, rows)
    config = {
        "schema_version": 1,
        "experiment_root": str(experiment_root),
        "roles": {
            "main": {
                "id": "selected",
                "label": "Selected H24",
                "run_path": "formal/main/H24_requested",
                "modes": ["MX"],
                "role": "selected",
            }
        },
    }
    roles = report._parse_roles(config, experiment_root)
    evidence = report._evidence_for_roles(roles, rows)
    assert evidence[0].status == "pending"
    assert not evidence[0].rows


def test_ledger_hash_mismatch_fails_closed(tmp_path: Path) -> None:
    experiment_root = tmp_path / "experiment"
    experiment_root.mkdir()
    _write_ledger_bundle(experiment_root, [_ledger_row("main/h24")])
    ledger = experiment_root / "ledger/run_ledger.csv"
    ledger.write_text(
        ledger.read_text(encoding="utf-8") + "tampered\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="ledger output hash mismatch"):
        report._load_verified_ledger(
            {"ledger": {}},
            experiment_root,
        )


def test_source_file_tamper_fails_closed_through_public_writer(tmp_path: Path) -> None:
    experiment_root = tmp_path / "experiment"
    experiment_root.mkdir()
    _write_ledger_bundle(experiment_root, [_ledger_row("main/h24")])
    config_path = _write_config(
        tmp_path / "scenario_config.json",
        _config_with_roles(
            experiment_root, {"main": _selected_main_role(experiment_root)}
        ),
    )
    source = experiment_root / "main/h24/run_config.json"
    tampered = bytearray(source.read_bytes())
    tampered[len(tampered) // 2] ^= 1
    source.write_bytes(tampered)

    with pytest.raises(ValueError, match=r"source file hash mismatch"):
        report.write_paper_experiment_report(config_path, tmp_path / "report")


def test_resealed_mode_summary_tamper_disagrees_with_ledger_row(
    tmp_path: Path,
) -> None:
    experiment_root = tmp_path / "experiment"
    experiment_root.mkdir()
    _write_ledger_bundle(experiment_root, [_ledger_row("main/h24")])
    config_path = _write_config(
        tmp_path / "scenario_config.json",
        _config_with_roles(
            experiment_root, {"main": _selected_main_role(experiment_root)}
        ),
    )
    relative = "main/h24/MX/summary.json"
    summary_path = experiment_root / relative
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["objective"] = float(summary["objective"]) + 1.0
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, sort_keys=True), encoding="utf-8"
    )
    _reseal_manifest_file(experiment_root, relative)

    with pytest.raises(ValueError, match=r"summary hash disagrees"):
        report.write_paper_experiment_report(config_path, tmp_path / "report")


def test_resealed_scenario_summary_semantic_tamper_fails_closed(
    tmp_path: Path,
) -> None:
    experiment_root = tmp_path / "experiment"
    experiment_root.mkdir()
    _write_ledger_bundle(experiment_root, [_ledger_row("main/h24")])
    config_path = _write_config(
        tmp_path / "scenario_config.json",
        _config_with_roles(
            experiment_root, {"main": _selected_main_role(experiment_root)}
        ),
    )
    summary_path = experiment_root / "ledger/scenario_summary.csv"
    with summary_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows[0]["modes"] = "HD"
    _write_csv(summary_path, rows)
    _reseal_manifest_file(experiment_root, "ledger/scenario_summary.csv", output=True)

    with pytest.raises(ValueError, match=r"scenario_summary.*(mismatch|disagree)"):
        report.write_paper_experiment_report(config_path, tmp_path / "report")


def test_multiple_main_roles_use_explicit_selected_even_when_not_first(
    tmp_path: Path,
) -> None:
    experiment_root = tmp_path / "experiment"
    experiment_root.mkdir()
    _write_ledger_bundle(
        experiment_root,
        [
            _ledger_row("main/h11", h=11),
            _ledger_row("main/h24", h=24),
        ],
    )
    first = {
        "id": "main_h11_candidate",
        "label": "H11 candidate main",
        "run_path": "main/h11",
        "instance_dir": str(experiment_root / "instances/H11"),
        "modes": ["MX"],
        "role": "candidate",
    }
    selected = _selected_main_role(experiment_root)
    config_path = _write_config(
        tmp_path / "scenario_config.json",
        _config_with_roles(experiment_root, {"main": [first, selected]}),
    )

    outputs = report.write_paper_experiment_report(config_path, tmp_path / "report")
    artifact = json.loads(outputs.artifact.read_text(encoding="utf-8"))
    headline = artifact["snapshot"]["datasets"]["headline"][0]
    assert headline["selected_h"] == 24
    assert "H24 MX 主算例" in outputs.markdown.read_text(encoding="utf-8")


def test_empty_role_modes_fail_closed_through_public_writer(tmp_path: Path) -> None:
    experiment_root = tmp_path / "experiment"
    experiment_root.mkdir()
    _write_ledger_bundle(experiment_root, [_ledger_row("main/h24")])
    main = _selected_main_role(experiment_root)
    main["modes"] = []
    config_path = _write_config(
        tmp_path / "scenario_config.json",
        _config_with_roles(experiment_root, {"main": main}),
    )

    with pytest.raises(ValueError, match=r"modes must be a nonempty"):
        report.write_paper_experiment_report(config_path, tmp_path / "report")


def test_scale_role_with_mismatched_h_is_invalid_not_ready(tmp_path: Path) -> None:
    experiment_root = tmp_path / "experiment"
    experiment_root.mkdir()
    _write_ledger_bundle(
        experiment_root,
        [
            _ledger_row("main/h24"),
            _ledger_row("scale/h23", h=23, instance_key="H24-scale"),
        ],
    )
    scale_role = {
        "id": "scale_h24_mismatch",
        "label": "Declared H24 but actual H23",
        "run_path": "scale/h23",
        "instance_dir": str(experiment_root / "instances/H24-scale"),
        "modes": ["MX"],
        "scale": {
            "H": 24,
            "L": 2,
            "E": 3,
            "crop_side_km": 9,
            "crop_area_km2": 81,
        },
        "solver_overrides": {"time_limit_seconds_per_mode": 300},
    }
    config_path = _write_config(
        tmp_path / "scenario_config.json",
        _config_with_roles(
            experiment_root,
            {
                "main": _selected_main_role(experiment_root),
                "scale": [scale_role],
            },
        ),
    )

    outputs = report.write_paper_experiment_report(config_path, tmp_path / "report")
    artifact = json.loads(outputs.artifact.read_text(encoding="utf-8"))
    assert _artifact_result_status(artifact, "scale_h24_mismatch") == "invalid"
    manifest = json.loads(outputs.manifest.read_text(encoding="utf-8"))
    assert manifest["evidence_state"]["invalid"] == 1


@pytest.mark.parametrize(
    ("mismatch", "row_options", "instance_key"),
    [
        ("method", {"solver_miqcp_method": 0}, "H24"),
        ("seed", {"solver_seed": 99}, "H24"),
        ("threads", {"solver_threads": 8}, "H24"),
        ("time", {"solver_time_limit_seconds": 600}, "H24"),
        ("instance", {}, "H24-algorithm-other"),
    ],
)
def test_algorithm_comparator_contract_mismatch_is_invalid(
    tmp_path: Path,
    mismatch: str,
    row_options: dict[str, object],
    instance_key: str,
) -> None:
    experiment_root = tmp_path / "experiment"
    experiment_root.mkdir()
    algorithm_row = _ledger_row(
        "algorithm/auto", instance_key=instance_key, **row_options
    )
    _write_ledger_bundle(
        experiment_root,
        [_ledger_row("main/h24"), algorithm_row],
    )
    algorithm_role = {
        "id": f"algorithm_{mismatch}_mismatch",
        "label": "MIQCPMethod=-1 matched comparator",
        "run_path": "algorithm/auto",
        "instance_dir": str(experiment_root / f"instances/{instance_key}"),
        "modes": ["MX"],
        "solver_overrides": {
            "miqcp_method": -1,
            "seed": 43,
            "threads": 4,
            "time_limit_seconds": 300,
        },
    }
    config_path = _write_config(
        tmp_path / "scenario_config.json",
        _config_with_roles(
            experiment_root,
            {
                "main": _selected_main_role(experiment_root),
                "algorithm": [algorithm_role],
            },
        ),
    )

    outputs = report.write_paper_experiment_report(config_path, tmp_path / "report")
    artifact = json.loads(outputs.artifact.read_text(encoding="utf-8"))
    assert (
        _artifact_result_status(artifact, f"algorithm_{mismatch}_mismatch") == "invalid"
    )


def test_sensitivity_declared_override_must_match_run_config(tmp_path: Path) -> None:
    experiment_root = tmp_path / "experiment"
    experiment_root.mkdir()
    _write_ledger_bundle(
        experiment_root,
        [
            _ledger_row("main/h24"),
            _ledger_row("sensitivity/rider080", solver_time_limit_seconds=3600),
        ],
    )
    sensitivity = {
        "id": "rider_override_mismatch",
        "label": "Rider x0.8",
        "run_path": "sensitivity/rider080",
        "instance_dir": str(experiment_root / "instances/H24"),
        "modes": ["MX"],
        "group": "rider_cost",
        "factor": 0.8,
        "model_overrides": {
            "rider_initial_charge_cny": 5.12,
            "rider_extra_cost_cny_per_km": 1.44,
        },
        "solver_overrides": {"time_limit_seconds": 3600},
    }
    config_path = _write_config(
        tmp_path / "scenario_config.json",
        _config_with_roles(
            experiment_root,
            {
                "main": _selected_main_role(experiment_root),
                "sensitivity": [sensitivity],
            },
        ),
    )

    outputs = report.write_paper_experiment_report(config_path, tmp_path / "report")
    artifact = json.loads(outputs.artifact.read_text(encoding="utf-8"))
    assert _artifact_result_status(artifact, "rider_override_mismatch") == "invalid"


def test_complete_exactness_with_screening_false_is_invalid(tmp_path: Path) -> None:
    experiment_root = tmp_path / "experiment"
    experiment_root.mkdir()
    _write_ledger_bundle(
        experiment_root,
        [
            _ledger_row("main/h24"),
            _ledger_row(
                "exact/complete",
                h=11,
                screening_is_exact=False,
                model_updates={
                    "display_origin_choices": "all",
                    "general_corridor_choices": "all",
                },
                solver_time_limit_seconds=3600,
            ),
        ],
    )
    complete = {
        "id": "complete_but_screened",
        "label": "Complete-index declaration with screened result",
        "run_path": "exact/complete",
        "instance_dir": str(experiment_root / "instances/H11"),
        "modes": ["MX"],
        "role": "complete",
        "model_overrides": {
            "display_origin_choices": "all",
            "general_corridor_choices": "all",
        },
        "solver_overrides": {"time_limit_seconds": 3600},
    }
    config_path = _write_config(
        tmp_path / "scenario_config.json",
        _config_with_roles(
            experiment_root,
            {
                "main": _selected_main_role(experiment_root),
                "exactness": [complete],
            },
        ),
    )

    outputs = report.write_paper_experiment_report(config_path, tmp_path / "report")
    artifact = json.loads(outputs.artifact.read_text(encoding="utf-8"))
    assert _artifact_result_status(artifact, "complete_but_screened") == "invalid"


def test_selected_instance_canonical_file_tamper_is_fatal(tmp_path: Path) -> None:
    experiment_root = tmp_path / "experiment"
    experiment_root.mkdir()
    _write_ledger_bundle(experiment_root, [_ledger_row("main/h24")])
    config_path = _write_config(
        tmp_path / "scenario_config.json",
        _config_with_roles(
            experiment_root, {"main": _selected_main_role(experiment_root)}
        ),
    )
    canonical = experiment_root / "instances/H24/sites.csv"
    tampered = bytearray(canonical.read_bytes())
    tampered[-2] ^= 1
    canonical.write_bytes(tampered)

    with pytest.raises(ValueError, match=r"(canonical|规范文件).*(hash|哈希)"):
        report.write_paper_experiment_report(config_path, tmp_path / "report")


def test_selected_map_failure_marks_partial_and_records_access_issue(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    experiment_root = tmp_path / "experiment"
    experiment_root.mkdir()
    _write_ledger_bundle(
        experiment_root,
        [
            _ledger_row("main/h24", mode=mode, solver_time_limit_seconds=900)
            for mode in ("HD", "MX", "LO")
        ],
    )
    config_path = _write_config(
        tmp_path / "scenario_config.json",
        _config_with_roles(
            experiment_root,
            {"main": _selected_main_role(experiment_root, modes=["HD", "MX", "LO"])},
        ),
    )

    def fail_map_load(*args: object, **kwargs: object) -> None:
        raise PermissionError("fixture map source access denied")

    monkeypatch.setattr(report.geo, "_load_case", fail_map_load)
    outputs = report.write_paper_experiment_report(config_path, tmp_path / "report")
    artifact = json.loads(outputs.artifact.read_text(encoding="utf-8"))
    assert artifact["snapshot"]["status"] == "partial"
    assert any(
        issue.get("scope") == "selected_map"
        for issue in artifact["snapshot"]["accessIssues"]
    )
    manifest = json.loads(outputs.manifest.read_text(encoding="utf-8"))
    assert manifest["map_status"] == "failed"


def test_final_registry_with_all_roles_pending_refuses_result_artifact(
    tmp_path: Path,
) -> None:
    experiment_root = tmp_path / "experiment"
    experiment_root.mkdir()
    _write_ledger_bundle(experiment_root, [_ledger_row("unassigned/finished")])
    pending = _selected_main_role(
        experiment_root, run_path="main/not_finished", modes=["MX"]
    )
    config_path = _write_config(
        tmp_path / "scenario_config.json",
        _config_with_roles(experiment_root, {"main": pending}),
    )

    with pytest.raises(ValueError, match=r"no evidence-ready roles"):
        report.write_paper_experiment_report(config_path, tmp_path / "report")


def test_choice_curve_matches_current_logit_definition() -> None:
    parameters = {
        "locker_asc": -1.0,
        "delivery_time_coefficient": 0.06,
        "price_coefficient": 0.35,
        "walk_distance_coefficient": 1.2,
        "home_eta_minutes": 30.0,
        "locker_eta_minutes": 20.0,
        "locker_discount_cny": 1.0,
        "max_walk_km": 1.5,
    }
    rows = report._choice_curve_rows(parameters, [0.0, 1.5], [1.0])
    assert rows[0]["到柜选择比例"] == pytest.approx(0.4875026035)
    assert rows[1]["到柜选择比例"] == pytest.approx(0.1358728970)
    assert all(row["是否在模型覆盖范围"] for row in rows)


def test_repo_relative_run_path_is_normalized_against_experiment_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    root = repo / "results/paper"
    root.mkdir(parents=True)
    monkeypatch.setattr(report, "_repo_root", lambda: repo)
    assert (
        report._normalise_run_path("results/paper/formal/main/H24", root)
        == "formal/main/H24"
    )
