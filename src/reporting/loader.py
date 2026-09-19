"""Load and reconcile saved case reports without current parameter defaults."""

from __future__ import annotations

from pathlib import Path
from .io import _json


def load_case(path):
    """Read and reconcile a saved run; no current defaults enter its numbers."""
    path = Path(path)
    result = _json(path / "result.json", {})
    metadata = _json(path / "input_metadata.json", {})
    values = metadata.get("values", {})
    operations = _json(path / "operations.json")
    run = _json(path / "summary.json", {})
    manifest = _json(path / "manifest.json", {})
    if not operations:
        return dict(
            path=path,
            result=result,
            metadata=values,
            pending=True,
            status=run.get("status", manifest.get("status", "运营核验尚未完成")),
        )
    summary, tables = operations["summary"], operations["tables"]
    ods = tables.get("od_times", [])
    mode = {
        key: float(sum(row.get(key + "_per_hour", 0.0) or 0.0 for row in ods))
        for key in ("pickup", "home", "direct")
    }
    fleet = {row["component"]: float(row["drones"]) for row in tables["fleet"]}
    costs = {row["component"]: float(row["cny_per_hour"]) for row in tables["costs"]}
    demand = float(summary["demand"])
    reconciliation = dict(
        demand_split_error=abs(sum(mode.values()) - demand),
        service_split_error=abs(mode["pickup"] + mode["home"] - summary["served"]),
        fleet_split_error=abs(sum(fleet.values()) - summary["fleet"]),
        cost_split_error=abs(sum(costs.values()) - summary["objective"]),
    )
    checks = summary.get("checks", {})
    verified = (
        bool(checks.get("passed"))
        and max(reconciliation.values()) <= 1e-5
        and run.get("passed", True) is not False
        and manifest.get("passed", True) is not False
    )
    row = dict(
        case=path.name,
        H=int(values.get("n_h", 0)),
        L=int(values.get("n_l", 0)),
        E=int(values.get("n_k", 0)),
        OD=int(values.get("n_od", result.get("n_od", 0))),
        demand=demand,
        served=float(summary["served"]),
        pickup=mode["pickup"],
        home=mode["home"],
        direct=mode["direct"],
        served_pct=100.0 * summary["served"] / demand if demand else 0.0,
        pickup_pct_of_drone=(
            100.0 * mode["pickup"] / summary["served"] if summary["served"] else 0.0
        ),
        open_bases=int(summary["opened"]["bases"]),
        open_lockers=int(summary["opened"]["lockers"]),
        open_swaps=int(summary["opened"]["swaps"]),
        cabinets=float(summary["cabinet_total"]),
        fleet=float(summary["fleet"]),
        empty_flow=float(summary["empty_flow"]),
        cross_base_pct=(
            max(0.0, 100.0 * summary["cross_base_service"] / summary["served"])
            if summary["served"]
            else 0.0
        ),
        cost=float(summary["objective"]),
        cost_per_order=float(summary["objective"]) / demand if demand else 0.0,
        baseline_cost=result.get("all_ground_cost", values.get("c0")),
        gap_pct=100.0 * float(result["gap"]) if result.get("gap") is not None else None,
        bound=result.get("bound"),
        solver_seconds=result.get("wall"),
        validation="通过" if verified else "未通过",
        status=run.get("status", manifest.get("status", "unknown")),
        rho=";".join(f'{b["base"]}:{b["rho"]:.6f}' for b in tables["bases"]),
        cabinet_cost=values.get("hub_cost"),
        hh_link_cost=values.get("airline_hh_cost_per_km_hour"),
        other_link_cost=values.get("airline_connection_cost_per_km_hour"),
    )
    row.update(
        {
            "minutes_" + key: value
            for key, value in summary.get("time_averages_minutes", {}).items()
        }
    )
    row["minutes_direct_same_served_OD"] = summary.get(
        "direct_postfetch_average_minutes"
    )
    return dict(
        path=path,
        result=result,
        metadata=values,
        summary=summary,
        tables=tables,
        segments=operations.get("segments", []),
        mode=mode,
        fleet=fleet,
        costs=costs,
        row=row,
        checks=checks,
        reconciliation=reconciliation,
        pending=False,
    )
