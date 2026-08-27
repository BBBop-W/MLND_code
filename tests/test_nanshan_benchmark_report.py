from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import pytest

from src.utils import nanshan_benchmark_report as report


RUN_SIGNATURE_SHA256 = "a" * 64


def test_benchmark_report_requires_integer_schema_version(tmp_path: Path) -> None:
    payload = {
        "schema_version": 5.0,
        "formulation_id": report.FORMULATION_ID,
    }
    with pytest.raises(ValueError, match="legacy/incompatible result"):
        report._require_current_contract(payload, tmp_path / "summary.json")


def _summary(
    mode: str,
    *,
    network: bool,
    candidate_lockers: int = 120,
    objective: float = 900.0,
    bound: float = 880.0,
    baseline: float = 1_000.0,
) -> dict[str, object]:
    realized = 250.0 if network else 0.0
    incumbent = objective if network else baseline
    saving = baseline - incumbent
    gap = abs(incumbent - bound) / incumbent
    return {
        "run_signature_sha256": RUN_SIGNATURE_SHA256,
        "mode": mode,
        "status": 9,
        "status_name": "TIME_LIMIT",
        "objective": incumbent,
        "objective_bound": bound,
        "mip_gap": gap,
        "requested_mip_gap": 0.01,
        "runtime_seconds": 30.0,
        "courier_only_baseline": baseline,
        "all_rider_baseline": {"cost_per_hour": baseline},
        "cost_saving": saving,
        "cost_saving_percent": 100.0 * saving / baseline,
        "drone_served_rate": realized / 1_000.0,
        "network_is_nontrivial": network,
        "network_opened": network,
        "courier_only_incumbent": not network,
        "open_bases": 2 if network else 0,
        "open_lockers": 7 if network else 0,
        "active_lockers": 5 if network else 0,
        "idle_open_lockers": 2 if network else 0,
        "open_swap_stations": 3 if network else 0,
        "screening_is_exact": False,
        "solver": {
            "status": 9,
            "status_name": "TIME_LIMIT",
            "solution_count": 1,
            "objective_cost_per_hour": incumbent,
            "objective_bound": bound,
            "mip_gap": gap,
            "requested_mip_gap": 0.01,
            "runtime_seconds": 30.0,
            "network_start_runtime_seconds": 8.0,
            "main_solve_runtime_seconds": 22.0,
            "network_start_found": network,
            "network_start_key": ["H001", "H002"] if network else None,
            "algorithm": {
                "network_start_candidate_count": 0,
                "network_start_capacity_rejected_count": 0,
                "network_start_attempted_keys": [],
            },
            "external_start_installed": mode == "MX",
            "external_start_source": "HD" if mode == "MX" else None,
            "external_start_matched_variables": 500 if mode == "MX" else 0,
        },
        "model_size": {
            "variables": 10_000,
            "binary_variables": 500,
            "linear_constraints": 8_000,
            "quadratic_constraints": 34,
        },
        "instance": {
            "chi_keys": 7_500,
            "endogenous_corridors": 1,
            "candidate_bases": 8,
            "candidate_lockers": candidate_lockers,
            "candidate_swap_stations": 13,
            "markets": 11_427,
            "screening_is_exact": False,
        },
        "demand_funnel_orders_per_hour": {
            "total_demand": 1_000.0,
            "total_drone_realized": realized,
            "drone_service_fraction_of_demand": realized / 1_000.0,
            "direct_courier_fallback": 1_000.0 - realized,
        },
        "facilities": {
            "active_connections": 18 if network else 0,
            "active_lockers": 5 if network else 0,
            "idle_open_lockers": 2 if network else 0,
            "candidates": {
                "bases": 8,
                "lockers": candidate_lockers,
                "swap_stations": 13,
            },
            "open": {
                "bases": 2 if network else 0,
                "lockers": 7 if network else 0,
                "swap_stations": 3 if network else 0,
            },
        },
        "fleet": {"gamma_drones": 12.0 if network else 0.0},
        "fleet_size": 12.0 if network else 0.0,
        "locker_queue": {"maximum_utilization": 0.72 if network else 0.0},
        "flight_operations": {
            "loaded_flow_per_hour": realized,
            "loaded_flight_km_per_hour": 2.0 * realized,
            "loaded_direct_flow_per_hour": realized,
            "loaded_swap_flow_per_hour": 0.0,
            "empty_flow_per_hour": 50.0 if network else 0.0,
            "all_direct_flow_per_hour": realized + (40.0 if network else 0.0),
            "all_swap_flow_per_hour": 10.0 if network else 0.0,
            "empty_dispatch_per_hour": 40.0 if network else 0.0,
            "empty_recovery_per_hour": 10.0 if network else 0.0,
        },
        "energy_balance": {
            "balance_scope": "system_wide_base_departure_plus_swap",
            "drone_speed_mps": 20.0,
            "battery_range_km": 10.0,
            "usable_battery_time_hours": 10.0 / 72.0,
            "usable_battery_time_minutes": 60.0 * 10.0 / 72.0,
            "total_energy_use_equivalent_drones": 6.0 if network else 0.0,
            "total_energy_supply_equivalent_drones": (
                50.0 * 10.0 / 72.0 if network else 0.0
            ),
            "base_full_battery_departures_per_hour": 40.0 if network else 0.0,
            "swap_events_per_hour": 10.0 if network else 0.0,
            "battery_supply_events_per_hour": 50.0 if network else 0.0,
            "system_slack_equivalent_drones": (
                50.0 * 10.0 / 72.0 - 6.0 if network else 0.0
            ),
            "maximum_constraint_violation": 0.0,
        },
        "artifacts": {
            "open_facilities": "open_facilities.csv",
            "corridor_flows": "corridor_flows.csv",
        },
    }


def _summaries(network_modes: set[str]) -> dict[str, dict[str, object]]:
    return {
        mode: _summary(mode, network=mode in network_modes) for mode in report.MODES
    }


def _run_config(candidate_lockers: int = 120) -> dict[str, object]:
    return {
        "run_signature_sha256": RUN_SIGNATURE_SHA256,
        "modes": ["HD", "MX", "LO"],
        "cross_mode_start_policy": "HD_to_later_MX_by_variable_name",
        "screening_is_exact": {mode: False for mode in report.MODES},
        "model": {
            "display_origin_choices": 8,
            "general_corridor_choices": 10,
            "drone_speed_mps": 20.0,
            "battery_range_km": 10.0,
        },
        "solver": {"mip_gap": 0.01},
        "instance": {
            "metadata": {
                "data_config": {
                    "candidate_count": candidate_lockers,
                    "num_bases": 8,
                    "num_swaps": 13,
                }
            }
        },
    }


def _layout(candidate_lockers: int = 120) -> report.StudyLayout:
    sites = [
        {"id": f"L{index:03d}", "type": "L", "x": 0.0, "y": 0.0}
        for index in range(1, 9)
    ]
    sites.extend(
        {"id": f"H{index:03d}", "type": "H", "x": 0.0, "y": 0.0}
        for index in range(1, candidate_lockers + 1)
    )
    sites.extend(
        {"id": f"E{index:03d}", "type": "E", "x": 0.0, "y": 0.0}
        for index in range(1, 14)
    )
    return report.StudyLayout(
        metadata={"data_config": {"map_half_size_km": 10.0}},
        nodes=(),
        sites=tuple(sites),
        endpoint_rates={},
    )


def _write_mode_artifacts(
    root: Path,
    summaries: dict[str, dict[str, object]],
) -> None:
    for mode, summary in summaries.items():
        mode_dir = root / mode
        mode_dir.mkdir(parents=True)
        open_rows: list[tuple[str, str]] = []
        for prefix, kind, count in (
            ("L", "base", int(summary["open_bases"])),
            ("H", "locker", int(summary["open_lockers"])),
            ("E", "swap", int(summary["open_swap_stations"])),
        ):
            open_rows.extend(
                (f"{prefix}{index:03d}", kind) for index in range(1, count + 1)
            )
        with (mode_dir / "open_facilities.csv").open(
            "w",
            newline="",
            encoding="utf-8",
        ) as handle:
            writer = csv.writer(handle)
            writer.writerow(("facility", "type"))
            writer.writerows(open_rows)
        with (mode_dir / "corridor_flows.csv").open(
            "w",
            newline="",
            encoding="utf-8",
        ) as handle:
            writer = csv.writer(handle)
            writer.writerow(
                (
                    "origin_locker",
                    "destination_locker",
                    "connection_open",
                    "od_offered_per_hour",
                    "source_allocated_offered_per_hour",
                    "offered_reconciliation_error",
                    "realized_from_chi_per_hour",
                    "loaded_psi_per_hour",
                    "realized_reconciliation_error",
                    "offered_flow_weighted_rho",
                    "direct_flow_per_hour",
                    "swap_flow_per_hour",
                    "swap_share",
                    "flight_km_per_hour",
                )
            )
            realized = float(
                summary["demand_funnel_orders_per_hour"]["total_drone_realized"]
            )
            network = bool(summary["network_opened"])
            writer.writerow(
                (
                    "H001",
                    "H002",
                    int(network),
                    realized,
                    realized,
                    0.0,
                    realized,
                    realized,
                    0.0,
                    1.0 if realized > 0.0 else "",
                    realized,
                    0.0,
                    0.0 if realized > 0.0 else "",
                    2.0 * realized,
                )
            )
        solution_lines = ["# Objective value = 0"]
        if network:
            solution_lines.extend(
                (
                    f"varphi[H001,H002,__direct__] {realized}",
                    "varphi[L001,H001,__direct__] 40",
                    "varphi[H002,L001,E001] 10",
                )
            )
        else:
            solution_lines.append("varphi[H001,H002,__direct__] 0")
        (mode_dir / "incumbent.sol").write_text(
            "\n".join(solution_lines) + "\n",
            encoding="utf-8",
        )


def test_solution_metrics_cover_network_design_and_no_build() -> None:
    network = report._solution_metrics(_summary("MX", network=True))
    assert network["service_fraction"] == pytest.approx(0.25)
    assert network["cost_saving"] == pytest.approx(100.0)
    assert network["cost_saving_percent"] == pytest.approx(10.0)
    assert network["open_bases"] == 2
    assert network["open_lockers"] == 7
    assert network["active_lockers"] == 5
    assert network["idle_open_lockers"] == 2
    assert network["open_swap_stations"] == 3
    assert network["active_connections"] == 18
    assert network["fleet_size"] == pytest.approx(12.0)
    assert network["maximum_locker_utilization"] == pytest.approx(0.72)
    assert network["network_is_nontrivial"] is True
    assert network["network_opened"] is True

    no_build = report._solution_metrics(_summary("MX", network=False))
    assert no_build["service_fraction"] == 0.0
    assert no_build["cost_saving"] == 0.0
    assert no_build["open_lockers"] == 0
    assert no_build["active_lockers"] == 0
    assert no_build["network_is_nontrivial"] is False
    assert no_build["network_opened"] is False


def test_open_facilities_and_positive_drone_service_are_distinct() -> None:
    summary = _summary("MX", network=True)
    summary["network_is_nontrivial"] = False
    summary["courier_only_incumbent"] = True
    summary["drone_served_rate"] = 0.0
    summary["demand_funnel_orders_per_hour"].update(
        {
            "total_drone_realized": 0.0,
            "drone_service_fraction_of_demand": 0.0,
            "direct_courier_fallback": 1_000.0,
        }
    )

    metrics = report._solution_metrics(summary)
    assert metrics["network_opened"] is True
    assert metrics["network_is_nontrivial"] is False


def test_certificate_limits_include_baseline_objectives_and_bounds() -> None:
    lower, upper = report._certificate_x_limits(
        [1_000.0, 1_000.0, 1_000.0],
        [910.0, 930.0, 950.0],
        [870.0, 900.0, 920.0],
    )
    assert lower < 870.0
    assert upper > 1_000.0


def test_certificate_subtitle_switches_between_network_and_no_build() -> None:
    no_build = report._certificate_subtitle(_summaries(set()))
    assert "No reported incumbent opens drone facilities" in no_build
    assert "positive drone service occurs" not in no_build

    mixed = report._certificate_subtitle(_summaries({"MX", "LO"}))
    assert "Facilities open in 2 of 3" in mixed
    assert "positive drone service occurs in 2 of 3" in mixed


def test_candidate_locker_count_is_dynamic_and_cross_checked() -> None:
    summaries = _summaries({"MX"})
    assert report._candidate_locker_count(summaries) == 120
    assert report._artifact_prefix(120) == "h120"

    summaries["LO"]["instance"]["candidate_lockers"] = 100
    with pytest.raises(ValueError, match="inconsistent candidate-locker counts"):
        report._candidate_locker_count(summaries)


def test_mode_summary_csv_contains_operational_metrics(tmp_path: Path) -> None:
    path = report._write_mode_summary(tmp_path, _summaries({"MX", "HD", "LO"}))
    assert path.name == "h120_mode_summary.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 3
    mx = rows[0]
    assert mx["run_signature_sha256"] == RUN_SIGNATURE_SHA256
    assert int(mx["candidate_bases"]) == 8
    assert int(mx["candidate_lockers"]) == 120
    assert int(mx["candidate_swap_stations"]) == 13
    assert float(mx["drone_service_fraction_percent"]) == pytest.approx(25.0)
    assert float(mx["cost_saving_cny_per_hour"]) == pytest.approx(100.0)
    assert float(mx["cost_saving_percent"]) == pytest.approx(10.0)
    assert int(mx["open_bases"]) == 2
    assert int(mx["open_lockers"]) == 7
    assert int(mx["active_lockers"]) == 5
    assert int(mx["idle_open_lockers"]) == 2
    assert int(mx["open_swap_stations"]) == 3
    assert int(mx["active_connections"]) == 18
    assert float(mx["fleet_size_drones"]) == pytest.approx(12.0)
    assert float(mx["maximum_locker_utilization_percent"]) == pytest.approx(72.0)
    assert float(mx["base_full_battery_departures_per_hour"]) == pytest.approx(40.0)
    assert float(mx["swap_events_per_hour"]) == pytest.approx(10.0)
    assert float(mx["battery_supply_events_per_hour"]) == pytest.approx(50.0)
    assert float(mx["maximum_energy_violation_equivalent_drones"]) == pytest.approx(0.0)
    assert mx["network_start_found"] == "True"
    assert mx["network_start_key"] == "H001-H002"
    assert mx["network_is_nontrivial"] == "True"
    assert mx["network_opened"] == "True"


def test_solver_certificate_uses_dynamic_network_text_and_baseline_extent(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}

    def capture(figure: plt.Figure, stem: Path) -> tuple[Path, Path]:
        captured["figure"] = figure
        captured["stem"] = stem
        return stem.with_suffix(".png"), stem.with_suffix(".svg")

    monkeypatch.setattr(report, "_save_figure", capture)
    summaries = _summaries({"MX", "HD", "LO"})
    report._plot_solver_certificate(
        summaries,
        tmp_path,
        run_config=_run_config(),
    )

    figure = captured["figure"]
    assert isinstance(figure, plt.Figure)
    assert "open facilities and provide drone service" in " ".join(
        text.get_text() for text in figure.texts
    )
    assert "Nanshan H120 candidate set" in figure._suptitle.get_text()
    all_text = " ".join(text.get_text() for axis in figure.axes for text in axis.texts)
    assert "TIME_LIMIT incumbent" in all_text
    assert "gaps apply only to this screened formulation" in " ".join(
        text.get_text() for text in figure.texts
    )
    assert Path(captured["stem"]).name == "h120_solver_certificate"
    assert figure.axes[0].get_xscale() == "log"
    lower, upper = figure.axes[0].get_xlim()
    assert lower <= 0.01
    assert upper > 1.0
    assert "obj / bound" in all_text
    plt.close(figure)


def test_network_figure_title_and_name_use_dynamic_candidates(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}

    def capture(figure: plt.Figure, stem: Path) -> tuple[Path, Path]:
        captured["figure"] = figure
        captured["stem"] = stem
        return stem.with_suffix(".png"), stem.with_suffix(".svg")

    monkeypatch.setattr(report, "_save_figure", capture)
    layout = _layout()
    run_config = _run_config()
    summaries = _summaries({"MX", "HD", "LO"})
    _write_mode_artifacts(tmp_path, summaries)
    report._plot_mode_networks(
        layout,
        run_config,
        summaries,
        tmp_path,
        tmp_path,
        candidate_lockers=120,
    )

    figure = captured["figure"]
    assert isinstance(figure, plt.Figure)
    assert "Nanshan H120 candidate set" in figure._suptitle.get_text()
    description = " ".join(text.get_text() for text in figure.texts)
    assert "20×20 km" in description
    assert "11,427 OD markets" in description
    assert "Candidate sites L/H/E 8/120/13" in description
    assert "filled/enlarged symbols are opened facilities" in description
    assert "H with flow is reported per panel" in description
    assert "gaps apply only to this screened formulation" in description
    axis_text = " ".join(
        [
            *(axis.get_title(loc="left") for axis in figure.axes),
            *(text.get_text() for axis in figure.axes for text in axis.texts),
        ]
    )
    assert "TIME_LIMIT incumbent" in axis_text
    assert "H with flow 5" in axis_text
    legend_text = " ".join(text.get_text() for text in figure.legends[0].get_texts())
    assert "Opened facility" in legend_text
    assert "Loaded H–H physical flow" in legend_text
    assert "Base dispatch/recovery flow" in legend_text
    assert Path(captured["stem"]).name == "h120_three_mode_networks"
    plt.close(figure)


def test_h34_artifact_names_remain_compatible(tmp_path: Path) -> None:
    summaries = {
        mode: _summary(mode, network=False, candidate_lockers=34)
        for mode in report.MODES
    }
    path = report._write_mode_summary(tmp_path, summaries)
    assert path.name == "h34_mode_summary.csv"


def test_missing_metric_fails_instead_of_becoming_zero() -> None:
    summary = _summary("MX", network=True)
    del summary["fleet_size"]
    with pytest.raises(ValueError, match="fleet_size"):
        report._solution_metrics(summary)


def test_cross_mode_validation_checks_screening_speed_gap_and_demand() -> None:
    summaries = _summaries({"MX", "HD", "LO"})
    report._validate_benchmark_consistency(_layout(), _run_config(), summaries)

    summaries["LO"]["energy_balance"]["drone_speed_mps"] = 19.0
    with pytest.raises(ValueError, match="usable battery time|drone speed"):
        report._validate_benchmark_consistency(_layout(), _run_config(), summaries)


def test_benchmark_requires_hd_before_mx_and_installed_cross_mode_start() -> None:
    summaries = _summaries({"MX", "HD", "LO"})
    run_config = _run_config()
    run_config["modes"] = ["MX", "HD", "LO"]
    with pytest.raises(ValueError, match="ordered HD, MX, LO"):
        report._validate_benchmark_consistency(_layout(), run_config, summaries)

    run_config = _run_config()
    summaries["MX"]["solver"]["external_start_installed"] = False
    with pytest.raises(
        ValueError, match="installation of the HD-sourced external start"
    ):
        report._validate_benchmark_consistency(_layout(), run_config, summaries)


def test_benchmark_requires_matching_run_signatures() -> None:
    summaries = _summaries({"MX", "HD", "LO"})
    del summaries["LO"]["run_signature_sha256"]
    with pytest.raises(ValueError, match="run_signature_sha256"):
        report._validate_benchmark_consistency(_layout(), _run_config(), summaries)

    summaries = _summaries({"MX", "HD", "LO"})
    summaries["LO"]["run_signature_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="run signature disagrees"):
        report._validate_benchmark_consistency(_layout(), _run_config(), summaries)


def test_active_locker_count_must_not_exceed_open_lockers() -> None:
    summary = _summary("MX", network=True)
    summary["active_lockers"] = 8
    summary["facilities"]["active_lockers"] = 8
    with pytest.raises(ValueError, match="0 <= active <= open"):
        report._solution_metrics(summary)


def test_open_facility_artifact_must_match_summary(tmp_path: Path) -> None:
    summaries = _summaries({"MX", "HD", "LO"})
    _write_mode_artifacts(tmp_path, summaries)
    path = tmp_path / "MX" / "open_facilities.csv"
    rows = report._read_csv(path)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("facility", "type"))
        writer.writeheader()
        writer.writerows(rows[:-1])
    with pytest.raises(ValueError, match="counts disagree"):
        report._validate_mode_artifacts(_layout(), summaries, tmp_path)


def test_corridor_artifact_cannot_be_mixed_across_mode_summaries(
    tmp_path: Path,
) -> None:
    summaries = _summaries({"MX"})
    _write_mode_artifacts(tmp_path, summaries)
    (tmp_path / "MX" / "corridor_flows.csv").write_bytes(
        (tmp_path / "HD" / "corridor_flows.csv").read_bytes()
    )

    with pytest.raises(ValueError, match="CSV loaded vs realized"):
        report._validate_mode_artifacts(_layout(), summaries, tmp_path)


def test_corridor_artifact_requires_current_fields_and_row_count(
    tmp_path: Path,
) -> None:
    summaries = _summaries({"MX", "HD", "LO"})
    _write_mode_artifacts(tmp_path, summaries)
    path = tmp_path / "MX" / "corridor_flows.csv"
    fields, rows = report._read_csv_table(path)
    reduced_fields = tuple(field for field in fields if field != "direct_flow_per_hour")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=reduced_fields)
        writer.writeheader()
        writer.writerows(
            {field: row[field] for field in reduced_fields} for row in rows
        )
    with pytest.raises(ValueError, match="incompatible columns"):
        report._validate_mode_artifacts(_layout(), summaries, tmp_path)

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
    with pytest.raises(ValueError, match="row count disagrees"):
        report._validate_mode_artifacts(_layout(), summaries, tmp_path)


def test_physical_route_solution_must_reconcile_with_summary(tmp_path: Path) -> None:
    summaries = _summaries({"MX", "HD", "LO"})
    _write_mode_artifacts(tmp_path, summaries)
    path = tmp_path / "MX" / "incumbent.sol"
    text = path.read_text(encoding="utf-8").replace(
        "varphi[H001,H002,__direct__] 250.0",
        "varphi[H001,H002,__direct__] 249.0",
    )
    path.write_text(text, encoding="utf-8")

    with pytest.raises(ValueError, match="incumbent.sol loaded_direct_flow"):
        report._validate_mode_artifacts(_layout(), summaries, tmp_path)


def test_required_result_files_report_all_missing_modes(tmp_path: Path) -> None:
    (tmp_path / "run_config.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="MX/summary.json") as error:
        report._require_complete_result_files(tmp_path)
    assert "HD/open_facilities.csv" in str(error.value)
    assert "LO/corridor_flows.csv" in str(error.value)
    assert "LO/incumbent.sol" in str(error.value)
