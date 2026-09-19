"""Check financial/mode reconciliation and full-edge export semantics."""

import json

from src.reporting.spatial import aggregate_edges
from src.reporting.loader import load_case


def _saved_case(path, *, cost=25.0, passed=True):
    path.mkdir()
    (path / "result.json").write_text(
        json.dumps(dict(n_od=1, gap=0.1, wall=5.0, bound=22.5, all_ground_cost=30.0))
    )
    (path / "input_metadata.json").write_text(
        json.dumps(dict(values=dict(n_l=1, n_h=2, n_k=0, n_od=1, hub_cost=7.0)))
    )
    (path / "summary.json").write_text(
        json.dumps(dict(passed=passed, status="feasible_limit"))
    )
    summary = dict(
        name="test",
        demand=10.0,
        served=5.0,
        fleet=2.0,
        objective=25.0,
        empty_flow=0.0,
        cross_base_service=0.0,
        cabinet_total=2.0,
        opened=dict(bases=1, lockers=2, swaps=0),
        checks=dict(passed=True),
    )
    tables = dict(
        od_times=[dict(pickup_per_hour=3.0, home_per_hour=2.0, direct_per_hour=5.0)],
        fleet=[dict(component="idle", drones=2.0)],
        costs=[dict(component="fleet", cny_per_hour=cost)],
        bases=[dict(base=0, rho=0.6)],
    )
    (path / "operations.json").write_text(
        json.dumps(dict(summary=summary, tables=tables))
    )
    return path


def test_report_uses_saved_parameters_and_order_denominators(tmp_path):
    case = load_case(_saved_case(tmp_path / "H002_5s"))
    assert case["row"]["validation"] == "通过"
    assert case["row"]["served_pct"] == 50.0
    assert case["row"]["pickup_pct_of_drone"] == 60.0
    assert case["row"]["cabinet_cost"] == 7.0
    assert case["row"]["hh_link_cost"] is None
    assert case["reconciliation"]["demand_split_error"] == 0.0


def test_mismatching_cost_component_blocks_validation(tmp_path):
    case = load_case(_saved_case(tmp_path / "H002_5s", cost=24.0))
    assert case["row"]["validation"] == "未通过"
    assert case["reconciliation"]["cost_split_error"] == 1.0


def test_failed_source_run_is_not_reported_as_verified(tmp_path):
    case = load_case(_saved_case(tmp_path / "H002_5s", passed=False))
    assert case["row"]["validation"] == "未通过"


def test_missing_operations_remains_pending(tmp_path):
    (tmp_path / "result.json").write_text("{}")
    (tmp_path / "manifest.json").write_text('{"status":"running"}')
    case = load_case(tmp_path)
    assert case["pending"] and case["status"] == "running"


def test_edge_aggregation_preserves_direction_phase_and_tiny_flows():
    def segment(u, v, phase, flow, kind="service"):
        return dict(u_index=u, v_index=v, phase=phase, flow_per_hour=flow, kind=kind)

    segments = [
        segment(0, 1, "loaded", 2.0),
        segment(0, 1, "loaded", 3.0),
        segment(1, 0, "loaded", 0.0001),
        segment(0, 1, "pre_pickup", 4.0),
        segment(0, 1, "empty", 0.2, "empty"),
    ]
    edges = {
        (e["phase"], e["u"], e["v"]): e["flow_per_hour"]
        for e in aggregate_edges(segments)
    }
    assert edges == {
        ("loaded", 0, 1): 5.0,
        ("loaded", 1, 0): 0.0001,
        ("pre_pickup", 0, 1): 4.0,
        ("empty", 0, 1): 0.2,
    }


def test_setup_cost_report_reads_resolved_configuration():
    from src.reporting.metrics import _site_cost_metrics

    data = {
        "base_site_cost": 20000 / 21900,
        "hub_site_cost": 20000 / 21900,
        "swap_site_cost": 10000 / 21900,
        "provenance": {
            "resolved_config": {
                "depreciation": {"years": 5, "days_per_year": 365, "hours_per_day": 12},
                "costs": {
                    "base_site_investment": 20000,
                    "locker_site_investment": 20000,
                    "swap_site_investment": 10000,
                },
            }
        },
    }
    rebuilt = {
        "tables": {
            "site_costs": [
                {
                    "component": "base",
                    "opened_locations": 1,
                    "cny_per_hour": 20000 / 21900,
                },
                {
                    "component": "locker",
                    "opened_locations": 2,
                    "cny_per_hour": 40000 / 21900,
                },
                {"component": "swap", "opened_locations": 0, "cny_per_hour": 0.0},
            ]
        }
    }
    actual = _site_cost_metrics(data, rebuilt)
    assert actual["active"]
    assert [r["included_investment_cny"] for r in actual["rows"]] == [
        20000,
        20000,
        10000,
    ]
    assert all(r["amortization_operating_hours"] == 21900 for r in actual["rows"])
    assert (
        max(r["investment_hourly_reconciliation_error"] for r in actual["rows"]) == 0.0
    )


def test_map_projection_inverts_recorded_local_coordinates():
    import pytest
    from src.reporting.projection import (
        local_km_to_plot_lonlat,
        lonlat_to_tile,
        tile_to_lonlat,
    )

    metadata = dict(projection_center_lon=113.94, projection_center_lat=22.53)
    lonlat = local_km_to_plot_lonlat(0.0, 0.0, metadata, assume_gcj02=False)
    assert lonlat == pytest.approx((113.94, 22.53))
    assert tile_to_lonlat(*lonlat_to_tile(*lonlat, 14), 14) == pytest.approx(lonlat)


def test_current_solver_stages_take_precedence_and_keep_elapsed_clock():
    import pytest
    from src.reporting.metrics import _solver_metrics
    from src.reporting.solver_text import solver_section

    result = dict(
        method="hybrid",
        objective=100.0,
        bound=90.0,
        gap=0.1,
        wall=94.0,
        budget=100.0,
        solve_wall=999.0,
        selection_heuristic_wall=999.0,
        full_cg_refinement={"wall": 999.0},
        params={"Threads": 4, "Seed": 43, "MIPGap": 0.0001, "MIPGapAbs": 0.000001},
        stages={
            "benders": {
                "wall": 60.0,
                "budget": 65.0,
                "status": 9,
                "termination_reason": "time_limit",
            },
            "design_search": {
                "wall": 23.0,
                "budget": 25.0,
                "statistics": {"service_columns": 123},
            },
            "refinement": {
                "wall": 9.0,
                "budget": 10.0,
                "cg_complete": True,
                "full_graph_pricing": True,
                "accepted": True,
            },
        },
    )
    metrics = _solver_metrics(
        result,
        {"solver": {"threads": 8, "seed": 99, "time_limit": 200.0}},
        {"total_wall": 110.0},
    )
    assert metrics["benders_wall"] == 60.0
    assert metrics["benders_budget"] == 65.0
    assert metrics["selection_heuristic_wall"] == 23.0
    assert metrics["selection_heuristic_budget"] == 25.0
    assert metrics["selection_heuristic"]["stats"]["service_columns"] == 123
    assert metrics["full_cg_wall"] == 9.0
    assert metrics["refinement_budget"] == 10.0
    assert metrics["full_cg_complete"] and metrics["full_graph_pricing"]
    assert metrics["polish_wall"] is None
    assert metrics["wall"] == 94.0
    assert metrics["total_wall"] == 110.0
    assert metrics["cumulative_solver_seconds"] == 92.0
    assert metrics["threads"] == 4 and metrics["seed"] == 43
    assert metrics["mip_gap"] == pytest.approx(0.0001)
    assert metrics["budget"] == 100.0
    text = "\n".join(solver_section(metrics))
    assert "|Benders + column generation|65.0|60.00|time_limit|" in text
    assert "|Integer design search|25.0|23.00|" in text
    assert "|Full-graph column-generation refinement|10.0|9.00|" in text
    assert "|Random seed|43|" in text
    assert "sequentially" in text and "999" not in text


def test_direct_and_archived_solver_metadata_remain_readable():
    from src.reporting.metrics import _solver_metrics

    direct = _solver_metrics(
        dict(
            method="direct",
            objective=10.0,
            bound=10.0,
            gap=0.0,
            wall=3.0,
            params={"Threads": 2, "Seed": 7, "TimeLimit": None},
        ),
        {"method": "direct", "solver": {"time_limit": None}},
        {"total_wall": 5.0},
    )
    assert direct["benders_wall"] is None and direct["benders_budget"] is None
    assert direct["direct_wall"] == 3.0 and direct["direct_budget"] is None
    assert direct["stage_rows"][0]["name"] == "direct"
    assert direct["threads"] == 2 and direct["seed"] == 7

    archived = _solver_metrics(
        dict(
            method="benders",
            objective=10.0,
            wall=50.0,
            solve_wall=40.0,
            polish_wall=8.0,
            full_cg_refinement={"wall": 3.0},
            selection_heuristic={"wall": 10.0, "budget": 12.0},
            timing_note="Search phases partially overlap.",
        ),
        {"seconds": 45.0, "threads": 3},
        {},
    )
    assert archived["benders_wall"] == 40.0 and archived["benders_budget"] == 45.0
    assert archived["full_cg_wall"] == 3.0 and archived["polish_wall"] == 5.0
    assert archived["selection_heuristic_wall"] == 10.0
    assert archived["wall"] == 50.0 and archived["threads"] == 3
    assert archived["timing_note"] == "Search phases partially overlap."


def test_stage_figure_uses_recorded_stage_times_and_budgets(tmp_path, monkeypatch):
    import numpy as np
    from src.reporting.metrics import _solver_metrics
    from src.reporting import charts

    solver = _solver_metrics(
        dict(
            method="hybrid",
            objective=10.0,
            wall=9.0,
            budget=10.0,
            params={"Threads": 2, "Seed": 5},
            stages={
                "benders": {"wall": 5.0, "budget": 6.0},
                "design_search": {"wall": 3.0, "budget": 4.0},
            },
        ),
        {},
        {},
    )
    captured = []

    def save(fig, out, name):
        ax = fig.axes[0]
        captured.append(
            (
                [patch.get_width() for patch in ax.patches],
                np.asarray(ax.collections[0].get_offsets())[:, 0].tolist(),
                ax.get_title(loc="left"),
                name,
            )
        )
        charts.plt.close(fig)

    monkeypatch.setattr(charts, "_save", save)
    charts.solver_stage_figure({"case": "H2", "solver": solver}, tmp_path, "en")
    widths, budgets, title, name = captured[0]
    assert widths == [5.0, 3.0] and budgets == [6.0, 4.0]
    assert "Threads: 2 | Seed: 5" in title and "9.00 s" in title
    assert name == "H2_solver_stages"
