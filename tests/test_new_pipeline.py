from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest
from gurobipy import GRB

from main import _data_config, _model_config, build_parser
from src.config import ChoiceConfig, DataConfig, ModelConfig, SolverConfig, _window_hours
from src.data_processor.generator import DataGenerator
from src.data_processor.loader import DataLoader
from src.model.gurobi_model import MLNDModel
from src.utils.output_processor import ResultProcessor


def _write_canonical_instance(root: Path) -> Path:
    root.mkdir()
    sites = [
        ("H001", "H", 0.0, 0.0),
        ("H002", "H", 2.0, 0.0),
        ("H003", "H", 1.0, 1.0),
        ("L001", "L", 0.0, 1.0),
        ("E001", "E", 1.0, 0.0),
    ]
    with open(root / "sites.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["site_id", "type", "x_km", "y_km", "activity_rate"])
        writer.writerows((*row, 10.0) for row in sites)
    nodes = [
        ("O0001", "origin", -0.5, 0.0),
        ("O0002", "origin", 0.2, 0.2),
        ("D0001", "destination", 2.4, 0.0),
        ("D0002", "destination", 1.1, 1.2),
    ]
    with open(root / "od_nodes.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["node_id", "role", "x_km", "y_km", "raw_endpoint_count"])
        writer.writerows((*row, 20) for row in nodes)
    with open(root / "od_demand.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "od_id",
                "origin_node",
                "destination_node",
                "order_count",
                "exposure_hours",
                "lambda_per_hour",
                "direct_distance_km",
            ]
        )
        writer.writerow(["OD000001", "O0001", "D0001", 200, 10, 20.0, 2.9])
        writer.writerow(["OD000002", "O0002", "D0002", 150, 10, 15.0, 1.35])
    with open(root / "metadata.json", "w") as handle:
        json.dump({"schema_version": 2, "source": "test", "demand_exposure_hours": 10}, handle)
    return root


@pytest.mark.parametrize("mode", ["MX", "HD", "LO"])
def test_od_and_corridor_coexist_in_each_model(tmp_path: Path, mode: str) -> None:
    instance_dir = _write_canonical_instance(tmp_path / mode)
    config = ModelConfig(
        choice=ChoiceConfig(max_walk_km=None),
        max_direct_range_km=5.0,
        max_swap_leg_km=5.0,
        origin_locker_choices=None,
        home_corridor_choices=None,
    )
    instance = DataLoader().load_instance_dir(str(instance_dir), model_config=config)
    assert set(instance.h_od) == {("O0001", "D0001"), ("O0002", "D0002")}
    assert set(instance.h_od).isdisjoint(instance.A_HH)
    for arc in instance.A_Y - instance.A_F:
        assert instance.arc_activation_cost[arc] == pytest.approx(
            config.connection_cost_per_km_hour * instance.distance(*arc)
        )

    model = MLNDModel(instance, mode).build()
    assert "delta" in model.vars
    assert "rho_l" not in model.vars and "rho_s" not in model.vars
    assert len(model.vars["h"]) > 0
    assert model.model.ModelSense == GRB.MINIMIZE

    solved = model.solve(
        SolverConfig(
            time_limit=10,
            mip_gap=0.01,
            threads=1,
            output_flag=0,
            use_valid_inequalities=True,
        )
    )
    assert solved
    assert model.model.Status in {GRB.OPTIMAL, GRB.TIME_LIMIT}
    courier_baseline = sum(instance.rider_cost(o, d) * rate for (o, d), rate in instance.h_od.items())
    assert model.model.ObjVal <= courier_baseline + 1e-5
    if mode == "MX":
        output = tmp_path / "output"
        summary = ResultProcessor(model, instance).write_all(str(output))
        assert summary["screening_is_exact"] is True
        assert "active_virtual_state_count" in summary["discrete_availability"]
        assert isinstance(summary["network_is_nontrivial"], bool)
        assert summary["requested_mip_gap"] == pytest.approx(0.01)
        assert summary["open_bases"] == summary["facilities"]["open"]["bases"]
        assert summary["open_lockers"] == summary["facilities"]["open"]["lockers"]
        assert summary["selected_rho_by_idle_state"] == summary["discrete_availability"][
            "selected_rho_by_idle_state"
        ]
        assert summary["solver"]["algorithm"]["name"] == (
            "demand_aware_perspective_branch_and_cut"
        )
        assert summary["solver"]["algorithm"]["static_valid_inequality_counts"]
        json.loads((output / "summary.json").read_text())
        for stem in (
            "network_map",
            "od_service_map",
            "cost_breakdown",
            "demand_funnel",
            "locker_utilization",
        ):
            assert (output / f"{stem}.png").stat().st_size > 0
            assert (output / f"{stem}.svg").stat().st_size > 0


def test_branch_and_cut_preserves_the_certified_small_instance_optimum(
    tmp_path: Path,
) -> None:
    instance_dir = _write_canonical_instance(tmp_path / "branch_cut_equivalence")
    config = ModelConfig(
        choice=ChoiceConfig(max_walk_km=None),
        max_direct_range_km=5.0,
        max_swap_leg_km=5.0,
        origin_locker_choices=None,
        home_corridor_choices=None,
    )
    objectives = []
    strengthened_model = None
    for strengthened in (False, True):
        instance = DataLoader().load_instance_dir(
            str(instance_dir), model_config=config
        )
        model = MLNDModel(instance, "MX")
        solved = model.solve(
            SolverConfig(
                time_limit=20.0,
                mip_gap=0.0,
                threads=1,
                output_flag=0,
                network_start_seconds=0.0,
                use_valid_inequalities=strengthened,
                use_user_cuts=strengthened,
                use_local_branching=False,
                max_root_cut_rounds=3,
                max_cuts_per_round=100,
            )
        )
        assert solved and model.model.Status == GRB.OPTIMAL
        objectives.append(model.model.ObjVal)
        if strengthened:
            strengthened_model = model

    assert objectives[1] == pytest.approx(objectives[0], abs=1e-7)
    assert strengthened_model is not None
    assert strengthened_model.static_cut_counts
    assert strengthened_model.root_cut_rounds > 0
    assert sum(strengthened_model.user_cut_counts.values()) > 0


def _ts(day: str, hour: int) -> int:
    local = datetime.strptime(f"{day} {hour}:00", "%Y%m%d %H:%M").replace(
        tzinfo=ZoneInfo("Asia/Shanghai")
    )
    return int(local.timestamp())


def _write_tsl_sample(path: Path) -> None:
    fields = [
        "order_id",
        "dt",
        "is_courier_grabbed",
        "platform_order_time",
        "sender_lng",
        "sender_lat",
        "recipient_lng",
        "recipient_lat",
        "poi_id",
    ]
    rows = [
        ("1", "20221017", "1", _ts("20221016", 12), 114000000, 22000000, 114004000, 22002000, "p1"),
        ("2", "20221017", "1", _ts("20221017", 12), 114002000, 22001000, 114006000, 22003000, "p1"),
        ("3", "20221018", "1", _ts("20221018", 12), 113998000, 22002000, 114003000, 22004000, "p2"),
        ("4", "20221018", "1", _ts("20221018", 12), 114001000, 21999000, 114005000, 22001000, "p2"),
        ("1", "20221017", "0", _ts("20221017", 12), 114000000, 22000000, 114004000, 22002000, "p1"),
    ]
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for values in rows:
            writer.writerow(dict(zip(fields, values)))


def test_tsl_filtering_average_and_candidate_invariance(tmp_path: Path) -> None:
    raw = tmp_path / "tsl.csv"
    _write_tsl_sample(raw)
    totals = []
    market_counts = []
    od_payloads = []
    for candidate_count in (2, 3):
        output = tmp_path / f"h{candidate_count}"
        DataGenerator().generate(
            DataConfig(
                source="tsl",
                raw_path=str(raw),
                output_dir=str(output),
                candidate_count=candidate_count,
                num_bases=1,
                num_swaps=1,
                od_grid_km=0.2,
                site_grid_km=0.1,
            )
        )
        with open(output / "metadata.json") as handle:
            metadata = json.load(handle)
        totals.append(metadata["total_demand_per_hour"])
        market_counts.append(metadata["od_market_count"])
        od_payloads.append(
            ((output / "od_nodes.csv").read_bytes(), (output / "od_demand.csv").read_bytes())
        )
        assert metadata["source_orders_all_days"] == 4
        assert metadata["accepted_waybills_all_days"] == 4
        assert metadata["demand_exposure_hours"] == 2
        assert metadata["date_basis"] == "source_partition_dt"
        assert metadata["timestamp_date_mismatch_orders_all_days"] == 1
        assert len(metadata["raw_file_sha256"]) == 64
    assert totals == pytest.approx([2.0, 2.0])
    assert market_counts[0] == market_counts[1]
    assert od_payloads[0] == od_payloads[1]


def test_dates_are_unique_and_must_exist(tmp_path: Path) -> None:
    raw = tmp_path / "tsl.csv"
    _write_tsl_sample(raw)
    output = tmp_path / "unique_dates"
    DataGenerator().generate(
        DataConfig(
            source="tsl",
            raw_path=str(raw),
            output_dir=str(output),
            candidate_count=2,
            num_bases=1,
            num_swaps=1,
            dates=("20221017", "20221017"),
            od_grid_km=0.2,
            site_grid_km=0.1,
        )
    )
    metadata = json.loads((output / "metadata.json").read_text())
    assert metadata["selected_days"] == ["20221017"]
    assert metadata["demand_exposure_hours"] == 1
    assert metadata["total_demand_per_hour"] == pytest.approx(2.0)

    with pytest.raises(ValueError, match="requested dates are not present"):
        DataGenerator().generate(
            DataConfig(
                source="tsl",
                raw_path=str(raw),
                output_dir=str(tmp_path / "invalid_date"),
                candidate_count=2,
                num_bases=1,
                num_swaps=1,
                dates=("20990101",),
            )
        )


def _write_nanshan_sample(path: Path) -> None:
    fields = [
        "dt_seq_id",
        "order_seq_id",
        "pickup_latitude",
        "pickup_longitude",
        "dropoff_latitude",
        "dropoff_longitude",
        "customer_pay_time",
        "fetch_time",
        "arrived_time",
        "package_price",
        "biz_line",
    ]
    rows = [
        ("1", "1", 22.540, 113.940, 22.545, 113.950, "12:05", "", "", 20, "waimai"),
        ("1", "2", 22.550, 113.950, 22.555, 113.960, "12:15", "", "", 25, "waimai"),
        ("2", "3", 22.560, 113.960, 22.565, 113.970, "12:25", "", "", 30, "waimai"),
        ("2", "4", 22.570, 113.970, 22.575, 113.980, "12:35", "", "", 35, "waimai"),
    ]
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for values in rows:
            writer.writerow(dict(zip(fields, values)))


def test_nanshan_uses_the_same_canonical_interface(tmp_path: Path) -> None:
    raw = tmp_path / "nanshan.csv"
    _write_nanshan_sample(raw)
    output = tmp_path / "nanshan"
    DataGenerator().generate(
        DataConfig(
            source="nanshan",
            raw_path=str(raw),
            output_dir=str(output),
            candidate_count=2,
            num_bases=1,
            num_swaps=1,
            od_grid_km=0.2,
            site_grid_km=0.1,
        )
    )
    metadata = json.loads((output / "metadata.json").read_text())
    assert metadata["source"] == "nanshan"
    assert metadata["date_basis"] == "dt_seq_id"
    assert metadata["demand_exposure_hours"] == 2
    assert metadata["total_demand_per_hour"] == pytest.approx(2.0)
    loaded = DataLoader().load_instance_dir(str(output))
    assert loaded.total_demand == pytest.approx(2.0)


def test_invalid_windows_and_solver_parameters_are_rejected() -> None:
    with pytest.raises(ValueError, match="start and end must differ"):
        _window_hours(12, 12)
    with pytest.raises(ValueError, match="mip_gap"):
        SolverConfig(mip_gap=float("nan")).validate()
    with pytest.raises(ValueError, match="ranges"):
        ModelConfig(max_direct_range_km=-1).validate()
    with pytest.raises(ValueError, match="data seed"):
        DataConfig(source="tsl", raw_path="x", output_dir="y", seed=-1).validate()
    with pytest.raises(ValueError, match="network_start_seconds"):
        SolverConfig(network_start_seconds=-1).validate()


def test_cli_imports_defaults_and_solve_seed() -> None:
    parser = build_parser()
    prepare = parser.parse_args(["prepare", "--source", "tsl"])
    data_config = _data_config(prepare)
    assert Path(data_config.raw_path).is_absolute()
    assert Path(data_config.output_dir).is_absolute()
    assert "tsl_od_h35_" in Path(data_config.output_dir).name

    solve = parser.parse_args(
        ["solve", "--instance-dir", "/tmp/example", "--modes", "MX,LO", "--seed", "17"]
    )
    assert solve.modes == ("MX", "LO")
    assert solve.solver_seed == 17

    tuned = parser.parse_args(
        [
            "solve",
            "--instance-dir",
            "/tmp/example",
            "--locker-service-rates",
            "100,200",
            "--locker-capacity-costs",
            "2,4",
            "--base-cost-per-hour",
            "20",
            "--flight-cost-per-km",
            "0.4",
            "--rider-cost-per-km",
            "1.6",
            "--max-virtual-wait-minutes",
            "3",
        ]
    )
    tuned_config = _model_config(tuned)
    tuned_config.validate()
    assert tuned_config.locker_service_rates == (100.0, 200.0)
    assert tuned_config.locker_capacity_costs == (2.0, 4.0)
    assert tuned_config.base_cost_per_hour == pytest.approx(20.0)
    assert tuned_config.flight_cost_per_km == pytest.approx(0.4)
    assert tuned_config.rider_cost_per_km == pytest.approx(1.6)
    assert tuned_config.max_virtual_wait_hours == pytest.approx(3.0 / 60.0)

    disabled_start = parser.parse_args(
        [
            "solve",
            "--instance-dir",
            "/tmp/example",
            "--network-start-seconds",
            "0",
        ]
    )
    assert disabled_start.network_start_seconds == pytest.approx(0.0)
