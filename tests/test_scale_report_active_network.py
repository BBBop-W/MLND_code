from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from src.model_contract import FORMULATION_ID, RESULT_SCHEMA_VERSION
from src.utils.scale_report import (
    _benchmark_context,
    _network_mix_note,
    _scale_data_note,
    _scale_design_note,
    _scale_tick_labels,
    _service_bar_label,
    collect_scale_results,
    write_scale_report,
)


FORMAL_LEVELS = (
    (11, 2, 2, 3.0),
    (24, 2, 3, 4.5),
    (43, 3, 5, 6.0),
    (68, 5, 7, 7.5),
    (87, 6, 9, 8.5),
    (120, 8, 13, 10.0),
)
CANONICAL_FILES = (
    "metadata.json",
    "sites.csv",
    "od_nodes.csv",
    "od_demand.csv",
    "periods.csv",
)


def _digest(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _instance_name(h: int, bases: int, swaps: int, radius: float) -> str:
    radius_token = f"{radius:05.2f}".replace(".", "p")
    return f"H{h:03d}_L{bases:02d}_E{swaps:02d}_R{radius_token}"


def _write_scale_run(
    root: Path,
    candidate_lockers: int,
    *,
    realized: float,
    candidate_bases: int = 1,
    candidate_swaps: int = 1,
    map_half_size_km: float | None = None,
    total_demand: float | None = None,
    od_hash_token: str | None = None,
    demand_scale: float = 1.0,
    raw_hash: str = "e" * 64,
    instance_directory: Path | None = None,
    nested_scale: dict[str, object] | None = None,
) -> dict[str, object]:
    run_dir = root / f"H{candidate_lockers}"
    hd_mode_dir = run_dir / "HD"
    mx_mode_dir = run_dir / "MX"
    hd_mode_dir.mkdir(parents=True)
    mx_mode_dir.mkdir(parents=True)
    built = realized > 0.0
    half_size = map_half_size_km or candidate_lockers / 5.0
    demand = total_demand or 5.0 * candidate_lockers
    od_token = od_hash_token or f"{candidate_lockers:x}"
    run_signature_sha256 = _digest(f"run:{candidate_lockers}:{od_token}")
    instance_directory = instance_directory or (
        root.parent / f"{root.name}_instances" / f"H{candidate_lockers}"
    )
    canonical_hashes = {
        filename: _digest(f"{od_token}:{filename}") for filename in CANONICAL_FILES
    }
    baseline = 1_000.0
    objective = 900.0 if built else baseline
    model = {
        "battery_range_km": 10.0,
        "drone_speed_mps": 20.0,
        "swap_service_time_hours": 0.05,
        "rider_initial_charge_cny": 6.4,
        "rider_included_distance_km": 3.0,
        "rider_incremental_cost_per_km": 1.8,
        "road_distance_factor": 1.25,
        "feeder_rider_cost_factor": 0.8,
        "display_origin_choices": 2,
        "general_corridor_choices": 5,
    }
    run_config = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "formulation_id": FORMULATION_ID,
        "run_signature_sha256": run_signature_sha256,
        "formulation_source": {
            "path": "/study/MLND/main.tex",
            "sha256": "b" * 64,
        },
        "implementation": {"sha256": "a" * 64},
        "modes": ["HD", "MX"],
        "cross_mode_start_policy": "HD_to_later_MX_by_variable_name",
        "warm_start_from": None,
        "instance": {
            "directory": str(instance_directory.resolve()),
            "metadata": {
                "source": "nanshan",
                "raw_file_sha256": raw_hash,
                "selected_days": ["1", "2", "3"],
                "date_basis": "dt_seq_id",
                "hour_basis": "customer_pay_time",
                "candidate_exposure_hours": 48,
                "demand_exposure_hours": 3,
                "candidate_orders_in_area": 100 * candidate_lockers,
                "demand_orders_in_area": 10 * candidate_lockers,
                "total_demand_per_hour": demand,
                "projection_center_lon": 113.94,
                "projection_center_lat": 22.55,
                "data_config": {
                    "candidate_count": candidate_lockers,
                    "num_bases": candidate_bases,
                    "num_swaps": candidate_swaps,
                    "candidate_hour_start": 8,
                    "candidate_hour_end": 24,
                    "demand_hour_start": 12,
                    "demand_hour_end": 13,
                    "map_half_size_km": half_size,
                    "od_grid_km": 0.5,
                    "site_grid_km": 0.25,
                    "seed": 42,
                },
                **({"nested_scale": nested_scale} if nested_scale is not None else {}),
            },
            "canonical_file_sha256": canonical_hashes,
        },
        "demand_scale": demand_scale,
        "model": model,
        "model_derived": {
            "drone_speed_kmph": 72.0,
            "usable_battery_time_minutes": 10.0 / 72.0 * 60.0,
        },
        "solver": {
            "time_limit": 60.0,
            "mip_gap": 0.02,
            "threads": 7,
            "seed": 19,
        },
    }
    locker_pickup = 20.0 if built else 0.0
    assisted_home = realized - locker_pickup
    hd_summary = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "formulation_id": FORMULATION_ID,
        "run_signature_sha256": run_signature_sha256,
        "mode": "HD",
        "solver": {
            "status_name": "OPTIMAL",
            "runtime_seconds": 10.0 + candidate_lockers,
            "external_start_installed": False,
            "external_start_source": None,
            "external_start_matched_variables": 0,
        },
    }
    summary = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "formulation_id": FORMULATION_ID,
        "run_signature_sha256": run_signature_sha256,
        "mode": "MX",
        "active_lockers": 3 if built else 0,
        "idle_open_lockers": 1 if built else 0,
        "network_opened": built,
        "network_is_nontrivial": built,
        "status_name": "OPTIMAL",
        "objective": objective,
        "courier_only_baseline": baseline,
        "cost_saving": baseline - objective,
        "cost_saving_percent": 100.0 * (baseline - objective) / baseline,
        "screening_is_exact": False,
        "solver": {
            "status_name": "OPTIMAL",
            "runtime_seconds": 18.0 + candidate_lockers,
            "objective_cost_per_hour": objective,
            "objective_bound": objective - 10.0,
            "mip_gap": 0.01,
            "external_start_installed": True,
            "external_start_source": "HD",
            "external_start_matched_variables": 1_000 + candidate_lockers,
        },
        "instance": {
            "data_source": "nanshan",
            "candidate_bases": candidate_bases,
            "candidate_lockers": candidate_lockers,
            "candidate_swap_stations": candidate_swaps,
            "chi_keys": 10_000 * candidate_lockers,
        },
        "model_size": {
            "variables": 20_000 * candidate_lockers,
            "binary_variables": 100 * candidate_lockers,
            "linear_constraints": 5_000 * candidate_lockers,
            "quadratic_constraints": candidate_lockers,
        },
        "facilities": {
            "active_lockers": 3 if built else 0,
            "idle_open_lockers": 1 if built else 0,
            "open": {
                "bases": 1 if built else 0,
                "lockers": 4 if built else 0,
                "swap_stations": 1 if built else 0,
            },
            "active_connections": 8 if built else 0,
        },
        "fleet": {"gamma_drones": 12.0 if built else 0.0},
        "locker_queue": {"maximum_utilization": 0.72 if built else 0.0},
        "flight_operations": {
            "total_flight_km_per_hour": 200.0 if built else 0.0,
            "all_swap_share": 0.15 if built else None,
        },
        "chaining": {"chaining_order_fraction": 0.25 if built else None},
        "demand_funnel_orders_per_hour": {
            "total_demand": demand,
            "total_drone_offered": 40.0 if built else 0.0,
            "total_drone_realized": realized,
            "drone_service_fraction_of_demand": realized / demand,
            "locker_pickup": locker_pickup,
            "assisted_home": assisted_home,
            "direct_courier_fallback": demand - realized,
        },
        "discrete_availability": {
            "active_virtual_state_count": 2 if built else 0,
            "offered_flow_weighted_rho": 0.8 if built else None,
        },
        "energy_balance": {"maximum_constraint_violation": 0.0},
    }
    (run_dir / "run_config.json").write_text(json.dumps(run_config), encoding="utf-8")
    (hd_mode_dir / "summary.json").write_text(
        json.dumps(hd_summary), encoding="utf-8"
    )
    (mx_mode_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    if built:
        with (mx_mode_dir / "open_facilities.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=("facility", "rho"))
            writer.writeheader()
            writer.writerow({"facility": "L001", "rho": 0.8})
    return {
        "instance_dir": str(instance_directory.resolve()),
        "is_full_parent": candidate_lockers == 120,
        "crop_half_size_km": half_size,
        "crop_side_km": 2.0 * half_size,
        "crop_area_km2": (2.0 * half_size) ** 2,
        "candidate_counts": {
            "H": candidate_lockers,
            "L": candidate_bases,
            "E": candidate_swaps,
        },
        "candidate_order_count": 100 * candidate_lockers,
        "candidate_order_multiset_sha256": _digest(
            f"candidate-orders:{candidate_lockers}"
        ),
        "demand_order_count": 10 * candidate_lockers,
        "demand_order_multiset_sha256": _digest(f"demand-orders:{candidate_lockers}"),
        "total_demand_per_hour": demand,
        "canonical_file_sha256": canonical_hashes,
    }


def _write_formal_crop_family(
    root: Path,
    *,
    realized_by_h: dict[int, float] | None = None,
    demand_scale_by_h: dict[int, float] | None = None,
    raw_hash_by_h: dict[int, str] | None = None,
    omit_h: int | None = None,
) -> Path:
    realized_by_h = realized_by_h or {}
    demand_scale_by_h = demand_scale_by_h or {}
    raw_hash_by_h = raw_hash_by_h or {}
    instance_root = root.parent / f"{root.name}_instances"
    family_id = "formal-family-01"
    instance_paths = {
        h: (instance_root / _instance_name(h, bases, swaps, radius)).resolve()
        for h, bases, swaps, radius in FORMAL_LEVELS
    }
    full_parent = instance_paths[120]
    level_records: list[dict[str, object]] = []

    for index, (h, bases, swaps, radius) in enumerate(FORMAL_LEVELS, start=1):
        demand = 50.0 * index
        canonical_hashes = {
            filename: _digest(f"{h:x}:{filename}") for filename in CANONICAL_FILES
        }
        candidate_hash = _digest(f"candidate-orders:{h}")
        demand_hash = _digest(f"demand-orders:{h}")
        nested_scale = {
            "schema_version": 2,
            "family_id": family_id,
            "level_index_small_to_large": index,
            "level_count": 6,
            "is_full_parent": h == 120,
            "full_parent_instance": str(full_parent),
            "crop_half_size_km": radius,
            "crop_side_km": 2.0 * radius,
            "crop_area_km2": (2.0 * radius) ** 2,
            "facility_target_counts": {"H": h, "L": bases, "E": swaps},
            "facility_actual_counts": {"H": h, "L": bases, "E": swaps},
            "site_policy": "recluster_within_crop",
            "raw_demand_orders_are_nested": True,
            "candidate_order_count": 100 * h,
            "candidate_order_multiset_sha256": candidate_hash,
            "demand_order_count": 10 * h,
            "demand_order_multiset_sha256": demand_hash,
            "orders_sampled": False,
            "demand_rescaled": False,
            "required_solve_demand_scale": 1.0,
        }
        level_record = {
            "instance_dir": str(instance_paths[h]),
            "is_full_parent": h == 120,
            "full_parent_instance": str(full_parent),
            "crop_half_size_km": radius,
            "crop_side_km": 2.0 * radius,
            "crop_area_km2": (2.0 * radius) ** 2,
            "candidate_counts": {"H": h, "L": bases, "E": swaps},
            "candidate_order_count": 100 * h,
            "candidate_order_multiset_sha256": candidate_hash,
            "demand_order_count": 10 * h,
            "demand_order_multiset_sha256": demand_hash,
            "total_demand_per_hour": demand,
            "canonical_file_sha256": canonical_hashes,
        }
        level_records.append(level_record)
        if h == omit_h:
            continue
        _write_scale_run(
            root,
            h,
            realized=realized_by_h.get(h, 0.0),
            candidate_bases=bases,
            candidate_swaps=swaps,
            map_half_size_km=radius,
            total_demand=demand,
            demand_scale=demand_scale_by_h.get(h, 1.0),
            raw_hash=raw_hash_by_h.get(h, "e" * 64),
            instance_directory=instance_paths[h],
            nested_scale=nested_scale,
        )

    instance_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 2,
        "family_id": family_id,
        "design": "same-center nested raw-order crops with per-crop site reclustering",
        "source": "nanshan",
        "full_parent_instance": str(full_parent),
        "full_parent_canonical_file_sha256": level_records[-1]["canonical_file_sha256"],
        "family_contract": {
            "source": "nanshan",
            "raw_file_sha256": "e" * 64,
            "projection_center_lon": 113.94,
            "projection_center_lat": 22.55,
            "selected_days": ["1", "2", "3"],
            "candidate_hours": [8, 24],
            "demand_hours": [12, 13],
            "candidate_exposure_hours": 48,
            "demand_exposure_hours": 3,
            "od_grid_km": 0.5,
            "site_grid_km": 0.25,
            "seed": 42,
            "levels": [
                {"half_size_km": radius, "H": h, "L": bases, "E": swaps}
                for h, bases, swaps, radius in FORMAL_LEVELS
            ],
        },
        "site_policy": "recluster_within_crop",
        "raw_candidate_order_multisets_nested": True,
        "raw_demand_order_multisets_nested": True,
        "orders_sampled": False,
        "demand_rescaled": False,
        "required_solve_demand_scale": 1.0,
        "levels": level_records,
    }
    (instance_root / "scale_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    return instance_root


def test_active_network_metrics_and_mixed_report(tmp_path: Path) -> None:
    _write_formal_crop_family(tmp_path, realized_by_h={24: 30.0})

    rows = collect_scale_results(tmp_path)
    assert [row["candidate_lockers"] for row in rows] == [
        11,
        24,
        43,
        68,
        87,
        120,
    ]
    no_build, built = rows[:2]
    assert no_build["network_is_nontrivial"] is False
    assert no_build["formulation_source_sha256"] == "b" * 64
    assert no_build["cost_saving_percent"] == pytest.approx(0.0)
    assert no_build["benchmark_design"] == "nested_geographic_crop"
    assert no_build["periods_sha256"] != built["periods_sha256"]
    assert built["period_id"] == "h12_13"
    assert built["network_is_nontrivial"] is True
    assert built["cost_saving_cny_per_hour"] == pytest.approx(100.0)
    assert built["cost_saving_percent"] == pytest.approx(10.0)
    assert built["drone_service_percent"] == pytest.approx(30.0)
    assert built["map_side_km"] == pytest.approx(9.0)
    assert built["map_area_sq_km"] == pytest.approx(81.0)
    assert built["demand_density_orders_per_hour_sq_km"] == pytest.approx(100.0 / 81.0)
    assert built["active_connections"] == 8
    assert built["active_lockers"] == 3
    assert built["idle_open_lockers"] == 1
    assert built["fleet_size_drones"] == pytest.approx(12.0)
    assert built["maximum_locker_utilization"] == pytest.approx(0.72)
    assert built["total_flight_km_per_hour"] == pytest.approx(200.0)
    assert built["swap_flow_share"] == pytest.approx(0.15)
    assert built["chaining_order_fraction"] == pytest.approx(0.25)
    assert built["selected_rho_by_facility"] == {"L001": 0.8}
    assert built["hd_runtime_seconds"] == pytest.approx(34.0)
    assert built["mx_runtime_seconds"] == pytest.approx(42.0)
    assert built["runtime_seconds"] == pytest.approx(42.0)
    assert built["end_to_end_runtime_seconds"] == pytest.approx(76.0)
    assert built["configured_end_to_end_budget_seconds"] == pytest.approx(120.0)

    written_rows = write_scale_report(tmp_path)
    assert len(written_rows) == 6
    for filename in (
        "benchmark_summary.csv",
        "benchmark_runtime.png",
        "benchmark_runtime.svg",
    ):
        assert (tmp_path / filename).stat().st_size > 0
    with (tmp_path / "benchmark_summary.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        csv_rows = list(csv.DictReader(handle))
    assert csv_rows[1]["open_lockers"] == "4"
    assert csv_rows[1]["active_lockers"] == "3"
    assert csv_rows[1]["idle_open_lockers"] == "1"
    assert csv_rows[1]["active_connections"] == "8"
    assert float(csv_rows[1]["drone_service_percent"]) == pytest.approx(30.0)
    assert csv_rows[1]["benchmark_design"] == "nested_geographic_crop"
    assert float(csv_rows[1]["map_area_sq_km"]) == pytest.approx(81.0)
    assert float(csv_rows[1]["hd_runtime_seconds"]) == pytest.approx(34.0)
    assert float(csv_rows[1]["mx_runtime_seconds"]) == pytest.approx(42.0)
    assert float(csv_rows[1]["end_to_end_runtime_seconds"]) == pytest.approx(76.0)
    assert float(csv_rows[1]["configured_end_to_end_budget_seconds"]) == pytest.approx(120.0)


def test_dynamic_context_and_network_notes(tmp_path: Path) -> None:
    _write_formal_crop_family(tmp_path, realized_by_h={24: 30.0})
    rows = collect_scale_results(tmp_path)

    context = _benchmark_context(rows)
    assert "screened origins top-2 / corridors top-5" in context
    assert "7 threads" in context
    assert "seed 19" in context
    assert "60s configured budget per mode" in context
    assert "120s configured HD→MX end-to-end budget" in context
    assert "barrier timeout checks may delay wall-clock return" in context
    assert "demand scale 1" in context
    assert "Gurobi" not in context
    assert _network_mix_note(rows).startswith("Facilities open in 1/6 incumbents")
    assert _scale_design_note(rows).startswith("Concentric square crops")

    labels = _scale_tick_labels(rows)
    assert labels == [
        "H11\n6 km\n220k vars",
        "H24\n9 km\n480k vars",
        "H43\n12 km\n860k vars",
        "H68\n15 km\n1360k vars",
        "H87\n17 km\n1740k vars",
        "H120\n20 km\n2400k vars",
    ]
    assert all("orders" not in label and "km²" not in label for label in labels)
    data_note = _scale_data_note(rows)
    assert "Area 36–400 km²" in data_note
    assert "demand 50–300 orders/h" in data_note
    assert "density 0.8–1.4 orders/h/km²" in data_note
    assert data_note.endswith("Exact rows: benchmark_summary.csv.")

    assert _service_bar_label(rows[0], 0.0) == "0%"
    active_label = _service_bar_label(rows[1], 30.0)
    assert active_label == (
        "30.0%\nL/H/E 1/4/1\nactive H 3 · fleet 12\nutil 72%"
    )
    assert "arcs" not in active_label

    exact_rows = [
        {
            **row,
            "screening_is_exact": True,
            "display_origin_choices": None,
            "general_corridor_choices": None,
        }
        for row in rows
    ]
    assert "complete index sets" in _benchmark_context(exact_rows)
    no_build_rows = [
        {
            **row,
            "network_is_nontrivial": False,
            "positive_drone_service": False,
            "network_opened": False,
        }
        for row in rows
    ]
    assert _network_mix_note(no_build_rows).startswith(
        "No incumbent opens drone facilities"
    )


def test_fixed_map_results_remain_separate_from_crop_results(tmp_path: Path) -> None:
    fixed_root = tmp_path / "fixed"
    _write_scale_run(
        fixed_root,
        10,
        realized=0.0,
        map_half_size_km=4.0,
        total_demand=100.0,
        od_hash_token="same",
    )
    _write_scale_run(
        fixed_root,
        20,
        realized=30.0,
        map_half_size_km=4.0,
        total_demand=100.0,
        od_hash_token="same",
    )
    fixed_rows = collect_scale_results(fixed_root)
    assert all(
        row["benchmark_design"] == "fixed_map_candidate_scale" for row in fixed_rows
    )
    assert _scale_design_note(fixed_rows).startswith("Legacy fixed-map benchmark")

    mixed_root = tmp_path / "mixed"
    _write_scale_run(mixed_root, 10, realized=0.0, map_half_size_km=2.0)
    _write_scale_run(mixed_root, 20, realized=20.0, map_half_size_km=2.0)
    _write_scale_run(mixed_root, 30, realized=30.0, map_half_size_km=6.0)
    with pytest.raises(ValueError, match="missing nested_scale/scale_manifest"):
        collect_scale_results(mixed_root)


def test_crop_contract_rejects_scaling_and_source_changes(tmp_path: Path) -> None:
    scaled_root = tmp_path / "scaled"
    _write_formal_crop_family(
        scaled_root,
        demand_scale_by_h={24: 1.2},
    )
    with pytest.raises(ValueError, match="demand_scale=1"):
        collect_scale_results(scaled_root)

    source_root = tmp_path / "source"
    _write_formal_crop_family(
        source_root,
        raw_hash_by_h={24: "f" * 64},
    )
    with pytest.raises(ValueError, match="raw_file_sha256"):
        collect_scale_results(source_root)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("drone_speed_mps", 19.0, "require drone_speed_mps=20"),
        ("battery_range_km", 9.0, "require battery_range_km=10"),
    ),
)
def test_scale_report_requires_confirmed_speed_and_range(
    tmp_path: Path,
    field: str,
    value: float,
    message: str,
) -> None:
    root = tmp_path / f"invalid_{field}"
    _write_formal_crop_family(root)
    for run_config_path in root.glob("H*/run_config.json"):
        run_config = json.loads(run_config_path.read_text(encoding="utf-8"))
        run_config["model"][field] = value
        run_config_path.write_text(json.dumps(run_config), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        collect_scale_results(root)


def test_scale_report_rejects_active_lockers_above_open_lockers(tmp_path: Path) -> None:
    root = tmp_path / "invalid_active"
    _write_formal_crop_family(root, realized_by_h={24: 30.0})
    summary_path = root / "H24" / "MX" / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["active_lockers"] = 5
    summary["facilities"]["active_lockers"] = 5
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    with pytest.raises(ValueError, match="0 <= active <= open"):
        collect_scale_results(root)


def test_scale_report_rejects_mismatched_run_signature(tmp_path: Path) -> None:
    root = tmp_path / "signature_mismatch"
    _write_formal_crop_family(root)
    summary_path = root / "H24" / "MX" / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["run_signature_sha256"] = "0" * 64
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    with pytest.raises(ValueError, match="run signature disagrees"):
        collect_scale_results(root)

    hd_root = tmp_path / "hd_signature_mismatch"
    _write_formal_crop_family(hd_root)
    hd_summary_path = hd_root / "H24" / "HD" / "summary.json"
    hd_summary = json.loads(hd_summary_path.read_text(encoding="utf-8"))
    hd_summary["run_signature_sha256"] = "0" * 64
    hd_summary_path.write_text(json.dumps(hd_summary), encoding="utf-8")

    with pytest.raises(ValueError, match="HD summary run signature disagrees"):
        collect_scale_results(hd_root)


def test_scale_report_requires_mx_summary_mode(tmp_path: Path) -> None:
    root = tmp_path / "wrong_mode"
    _write_formal_crop_family(root)
    summary_path = root / "H24" / "MX" / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["mode"] = "HD"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    with pytest.raises(ValueError, match="mode must be 'MX'"):
        collect_scale_results(root)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("modes", ["MX"], r"modes must equal \['HD', 'MX'\]"),
        (
            "cross_mode_start_policy",
            None,
            "cross_mode_start_policy must equal",
        ),
        (
            "warm_start_from",
            {"path": "/tmp/start.json"},
            "warm_start_from must be null",
        ),
    ),
)
def test_scale_report_rejects_noncomparable_run_configuration(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    root = tmp_path / f"invalid_{field}"
    _write_formal_crop_family(root)
    run_config_path = root / "H24" / "run_config.json"
    run_config = json.loads(run_config_path.read_text(encoding="utf-8"))
    run_config[field] = value
    run_config_path.write_text(json.dumps(run_config), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        collect_scale_results(root)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("external_start_installed", False, "external_start_installed=true"),
        ("external_start_source", "LO", "external_start_source must equal 'HD'"),
        (
            "external_start_matched_variables",
            0,
            "external_start_matched_variables must be positive",
        ),
    ),
)
def test_scale_report_rejects_invalid_hd_to_mx_start(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    root = tmp_path / f"invalid_external_{field}"
    _write_formal_crop_family(root)
    summary_path = root / "H24" / "MX" / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["solver"][field] = value
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        collect_scale_results(root)


def test_scale_report_rejects_missing_or_mixed_manuscript_hash(tmp_path: Path) -> None:
    missing_root = tmp_path / "missing_manuscript_hash"
    _write_formal_crop_family(missing_root)
    missing_path = missing_root / "H24" / "run_config.json"
    missing = json.loads(missing_path.read_text(encoding="utf-8"))
    del missing["formulation_source"]["sha256"]
    missing_path.write_text(json.dumps(missing), encoding="utf-8")
    with pytest.raises(ValueError, match="formulation_source.sha256"):
        collect_scale_results(missing_root)

    mixed_root = tmp_path / "mixed_manuscript_hash"
    _write_formal_crop_family(mixed_root)
    mixed_path = mixed_root / "H24" / "run_config.json"
    mixed = json.loads(mixed_path.read_text(encoding="utf-8"))
    mixed["formulation_source"]["sha256"] = "c" * 64
    mixed_path.write_text(json.dumps(mixed), encoding="utf-8")
    with pytest.raises(ValueError, match="formulation_source_sha256"):
        collect_scale_results(mixed_root)


def test_crop_contract_rejects_changed_center_or_time_window(tmp_path: Path) -> None:
    center_root = tmp_path / "center"
    _write_formal_crop_family(center_root)
    center_config_path = center_root / "H24" / "run_config.json"
    center_config = json.loads(center_config_path.read_text(encoding="utf-8"))
    center_config["instance"]["metadata"]["projection_center_lon"] = 113.95
    center_config_path.write_text(json.dumps(center_config), encoding="utf-8")
    with pytest.raises(ValueError, match="projection_center_lon"):
        collect_scale_results(center_root)

    window_root = tmp_path / "window"
    _write_formal_crop_family(window_root)
    window_config_path = window_root / "H24" / "run_config.json"
    window_config = json.loads(window_config_path.read_text(encoding="utf-8"))
    window_config["instance"]["metadata"]["data_config"]["demand_hour_end"] = 14
    window_config_path.write_text(json.dumps(window_config), encoding="utf-8")
    with pytest.raises(ValueError, match="demand_hour_end"):
        collect_scale_results(window_root)


def test_formal_crop_requires_all_six_runs_and_every_artifact(tmp_path: Path) -> None:
    incomplete_root = tmp_path / "incomplete"
    _write_formal_crop_family(incomplete_root, omit_h=87)
    with pytest.raises(ValueError, match="requires complete H set"):
        collect_scale_results(incomplete_root)

    missing_root = tmp_path / "missing"
    _write_formal_crop_family(missing_root)
    (missing_root / "H43" / "MX" / "summary.json").unlink()
    with pytest.raises(ValueError, match="missing required scale artifact"):
        collect_scale_results(missing_root)

    missing_hd_root = tmp_path / "missing_hd"
    _write_formal_crop_family(missing_hd_root)
    (missing_hd_root / "H43" / "HD" / "summary.json").unlink()
    with pytest.raises(ValueError, match="HD/summary.json"):
        collect_scale_results(missing_hd_root)

    corrupt_root = tmp_path / "corrupt"
    _write_formal_crop_family(corrupt_root)
    (corrupt_root / "H68" / "run_config.json").write_text("{not-json", encoding="utf-8")
    with pytest.raises(ValueError, match="H68"):
        collect_scale_results(corrupt_root)


def test_formal_crop_requires_manifest_and_directory_h_match(tmp_path: Path) -> None:
    missing_manifest_root = tmp_path / "no_manifest"
    instance_root = _write_formal_crop_family(missing_manifest_root)
    (instance_root / "scale_manifest.json").unlink()
    with pytest.raises(ValueError, match="missing required scale_manifest.json"):
        collect_scale_results(missing_manifest_root)

    mismatched_root = tmp_path / "directory_mismatch"
    _write_formal_crop_family(mismatched_root)
    summary_path = mismatched_root / "H24" / "MX" / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["instance"]["candidate_lockers"] = 25
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(ValueError, match="directory H does not match"):
        collect_scale_results(mismatched_root)


@pytest.mark.parametrize(
    ("artifact", "field", "value", "message"),
    (
        ("manifest", "orders_sampled", True, "scale_manifest.orders_sampled"),
        ("nested", "demand_rescaled", True, "nested_scale.demand_rescaled"),
        ("nested", "family_id", "wrong-family", "family_id mismatch"),
    ),
)
def test_formal_crop_rejects_manifest_or_nested_tampering(
    tmp_path: Path,
    artifact: str,
    field: str,
    value: object,
    message: str,
) -> None:
    root = tmp_path / f"{artifact}_{field}"
    instance_root = _write_formal_crop_family(root)
    if artifact == "manifest":
        path = instance_root / "scale_manifest.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload[field] = value
    else:
        path = root / "H43" / "run_config.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["instance"]["metadata"]["nested_scale"][field] = value
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        collect_scale_results(root)


def test_formal_crop_cross_checks_manifest_hashes(tmp_path: Path) -> None:
    root = tmp_path / "hash_mismatch"
    instance_root = _write_formal_crop_family(root)
    manifest_path = instance_root / "scale_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["levels"][2]["canonical_file_sha256"]["od_demand.csv"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="canonical file hashes"):
        collect_scale_results(root)
