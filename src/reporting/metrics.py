"""Independent case audit and report metric definitions."""

from __future__ import annotations

from collections import Counter
import json
import numpy as np
from src.experiments.storage import load_seed
from src.operations.validation import build_operations
from .io import _json


def _weighted(values, weights):
    v, w = np.asarray(values, float), np.asarray(weights, float)
    mask = np.isfinite(v) & np.isfinite(w) & (w > 0)
    v, w = v[mask], w[mask]
    if not len(v):
        return dict(mean=None, median=None, p90=None)
    order = np.argsort(v)
    v, w = v[order], w[order]
    cumulative = np.cumsum(w) / w.sum()
    return dict(
        mean=float(v @ w / w.sum()),
        median=float(v[np.searchsorted(cumulative, 0.5)]),
        p90=float(v[np.searchsorted(cumulative, 0.9)]),
    )


def _site_cost_metrics(data, rebuilt):
    evidence = (
        data.get("provenance", {})
        .get("requested_sensitivity", {})
        .get("site_costs", {})
    )
    config = data.get("provenance", {}).get("resolved_config", {})
    if not evidence and config.get("depreciation"):
        costs = config.get("costs", {})
        investments = {
            "L": costs.get("base_site_investment"),
            "H": costs.get("locker_site_investment"),
            "E": costs.get("swap_site_investment"),
        }
        evidence = dict(
            source="Resolved configuration",
            amortization_operating_hours=float(
                np.prod(list(config["depreciation"].values()))
            ),
            gross_investment_cny=investments,
            incremental_fraction={
                key: 1.0 for key, value in investments.items() if value is not None
            },
        )
    paid = {row["component"]: row for row in rebuilt["tables"].get("site_costs", [])}
    hours = evidence.get("amortization_operating_hours")
    rows = []
    for kind, node, key in (
        ("base", "L", "base_site_cost"),
        ("locker", "H", "hub_site_cost"),
        ("swap", "E", "swap_site_cost"),
    ):
        hourly = np.asarray(data.get(key, 0.0), float)
        gross = evidence.get("gross_investment_cny", {}).get(node)
        fraction = evidence.get("incremental_fraction", {}).get(node)
        included = (
            float(gross) * float(fraction)
            if gross is not None and fraction is not None
            else None
        )
        row = dict(
            component=kind,
            node=node,
            parameter=key,
            hourly_per_site=hourly.tolist(),
            gross_investment_cny=gross,
            incremental_fraction=fraction,
            included_investment_cny=included,
            amortization_operating_hours=hours,
            opened_locations=paid.get(kind, {}).get("opened_locations"),
            paid_cny_per_hour=paid.get(kind, {}).get("cny_per_hour", 0.0),
        )
        if included is not None and hours:
            row["investment_hourly_reconciliation_error"] = float(
                np.max(abs(hourly - included / float(hours)), initial=0.0)
            )
        rows.append(row)
    return dict(
        active=any(np.any(np.asarray(row["hourly_per_site"]) > 0) for row in rows),
        rows=rows,
        total_paid_cny_per_hour=sum(row["paid_cny_per_hour"] for row in rows),
        provenance=evidence,
        included_in_facility_cost_components=True,
    )


def _legacy_solver_metrics(result, manifest, run_summary):
    heuristic = result.get("selection_heuristic") or {}
    if not isinstance(heuristic, dict):
        heuristic = dict(description=str(heuristic))
    heuristic_wall = (
        result.get("selection_heuristic_wall", heuristic.get("wall", 0.0)) or 0.0
    )
    refinement = result.get("full_cg_refinement") or {}
    full_cg_wall = float(refinement.get("wall", 0.0) or 0.0)
    solve_wall = result.get("solve_wall")
    recorded_polish_wall = result.get("polish_wall")
    polish_wall = (
        max(0.0, float(recorded_polish_wall) - full_cg_wall)
        if recorded_polish_wall is not None
        else None
    )
    polish_passes = result.get("polish", {}).get("passes")
    cleanup_wall = (
        sum(float(p.get("wall", 0.0) or 0.0) for p in polish_passes)
        if polish_passes is not None and refinement
        else None
    )
    initial_polish_wall = (
        max(0.0, polish_wall - cleanup_wall)
        if polish_wall is not None and cleanup_wall is not None
        else None
    )
    phase_wall = sum(
        float(v or 0.0) for v in (solve_wall, polish_wall, heuristic_wall, full_cg_wall)
    )
    recorded_wall = result.get("wall")
    combined = float(recorded_wall) if recorded_wall is not None else phase_wall
    return dict(
        method=result.get("method"),
        status=result.get("status"),
        termination_reason=result.get("termination_reason"),
        bound=result.get("bound"),
        gap=result.get("gap"),
        objective=result["objective"],
        wall=combined,
        recorded_wall=recorded_wall,
        total_wall=result.get("total_wall", run_summary.get("total_wall")),
        budget=result.get("budget", manifest.get("seconds")),
        benders_budget=manifest.get("seconds"),
        polish_budget=manifest.get("polish_seconds"),
        benders_wall=solve_wall,
        polish_wall=polish_wall,
        recorded_polish_wall=recorded_polish_wall,
        initial_polish_wall=initial_polish_wall,
        post_selection_cleanup_wall=cleanup_wall,
        post_selection_polish_budget=manifest.get("post_selection_polish_seconds"),
        polish_accepted=result.get("polish", {}).get("accepted"),
        full_cg_refinement=refinement,
        full_cg_wall=full_cg_wall,
        refinement_budget=refinement.get("budget", manifest.get("refinement_seconds")),
        full_cg_complete=refinement.get("cg_complete"),
        full_graph_pricing=refinement.get("full_graph_pricing"),
        selection_heuristic=heuristic,
        selection_heuristic_wall=float(heuristic_wall),
        selection_heuristic_budget=heuristic.get("budget", heuristic.get("time_limit")),
        wall_components_include_selection_heuristic=bool(heuristic or heuristic_wall),
        timing_note=result.get("timing_note"),
        cumulative_solver_seconds=result.get("cumulative_solver_seconds"),
        threads=manifest.get("threads"),
        stats=result.get("stats", {}),
    )


def _solver_metrics(result, manifest, run_summary):
    """Normalize current stage records and archived run metadata for reporting.

    Actual solver parameters take precedence over requested manifest settings.
    The recorded solve wall clock and whole-run clock remain separate from the
    sum of stage durations, which can overlap in archived experiments.
    """
    metrics = _legacy_solver_metrics(result, manifest, run_summary)
    settings = manifest.get("solver") or {}
    stages = result.get("stages") or {}
    benders = stages.get("benders") or {}
    search = stages.get("design_search") or {}
    refinement = stages.get("refinement") or {}
    params = {**(benders.get("params") or {}), **(result.get("params") or {})}
    metrics.update(
        schema="stages" if stages else "archived",
        stages=stages,
        params=params,
        threads=params.get("Threads", settings.get("threads", manifest.get("threads"))),
        seed=params.get("Seed", settings.get("seed", manifest.get("seed"))),
        mip_gap=params.get("MIPGap", settings.get("mip_gap")),
        mip_gap_abs=params.get("MIPGapAbs", settings.get("mip_gap_abs")),
        budget=result.get(
            "budget",
            params.get(
                "TimeLimit", settings.get("time_limit", manifest.get("seconds"))
            ),
        ),
        total_wall=result.get(
            "total_wall", run_summary.get("total_wall", manifest.get("total_wall"))
        ),
    )
    if stages:
        heuristic = dict(search)
        heuristic["stats"] = search.get("statistics") or search.get("stats") or {}
        recorded_wall = result.get("wall")
        stage_wall = sum(
            float(stage.get("wall", 0.0) or 0.0) for stage in stages.values()
        )
        metrics.update(
            wall=float(recorded_wall) if recorded_wall is not None else stage_wall,
            recorded_wall=recorded_wall,
            benders_budget=benders.get("budget"),
            benders_wall=benders.get("wall"),
            selection_heuristic=heuristic if search else {},
            selection_heuristic_budget=search.get("budget"),
            selection_heuristic_wall=float(search.get("wall", 0.0) or 0.0),
            wall_components_include_selection_heuristic=bool(search),
            full_cg_refinement=refinement,
            full_cg_wall=float(refinement.get("wall", 0.0) or 0.0),
            refinement_budget=refinement.get("budget"),
            full_cg_complete=refinement.get("cg_complete"),
            full_graph_pricing=refinement.get("full_graph_pricing"),
            cumulative_solver_seconds=stage_wall,
            # The production runner has three sequential stages, without an
            # additional legacy polishing phase.
            polish_budget=None,
            polish_wall=None,
            recorded_polish_wall=None,
            initial_polish_wall=None,
            post_selection_cleanup_wall=None,
            post_selection_polish_budget=None,
            polish_accepted=None,
        )
    elif (
        result.get("method", "").startswith("direct")
        or manifest.get("method") == "direct"
        or result.get("method") in {"arc", "full_path"}
        or manifest.get("method") in {"arc", "full_path"}
    ):
        metrics.update(
            schema="direct",
            benders_budget=None,
            benders_wall=None,
            direct_wall=metrics["wall"],
            direct_budget=metrics["budget"],
        )
    metrics["stage_rows"] = []
    if stages:
        order = [
            name
            for name in ("benders", "design_search", "refinement")
            if name in stages
        ]
        order += [name for name in stages if name not in order]
        for name in order:
            stage = stages[name]
            metrics["stage_rows"].append(
                dict(
                    name=name,
                    wall=stage.get("wall"),
                    budget=stage.get("budget"),
                    status=stage.get("status"),
                    termination_reason=stage.get("termination_reason"),
                    objective=stage.get("objective"),
                    bound=stage.get("bound"),
                )
            )
    elif metrics["schema"] == "direct":
        metrics["stage_rows"] = [
            dict(
                name=(
                    result.get("method")
                    if result.get("method") in {"arc", "full_path"}
                    else "direct"
                ),
                wall=metrics["wall"],
                budget=metrics["budget"],
                status=metrics["status"],
                termination_reason=metrics["termination_reason"],
                objective=metrics["objective"],
                bound=metrics["bound"],
            )
        ]
    else:
        for name, wall_key, budget_key in (
            ("benders", "benders_wall", "benders_budget"),
            ("design_search", "selection_heuristic_wall", "selection_heuristic_budget"),
            ("refinement", "full_cg_wall", "refinement_budget"),
            ("polish", "polish_wall", "polish_budget"),
        ):
            wall = metrics.get(wall_key)
            if wall is not None and (wall or metrics.get(budget_key) is not None):
                metrics["stage_rows"].append(
                    dict(name=name, wall=wall, budget=metrics.get(budget_key))
                )
    return metrics


def audit(case):
    """Reconstruct saved paths, then reconcile all headline metrics."""
    data, result = load_seed(case["path"])
    rebuilt = build_operations(data, result)
    if not rebuilt["summary"]["checks"]["passed"]:
        raise ValueError("Independent operational reconstruction failed")
    for key in (
        "objective",
        "demand",
        "served",
        "fleet",
        "empty_flow",
        "cabinet_total",
    ):
        if abs(rebuilt["summary"][key] - case["summary"][key]) > 1e-5:
            raise ValueError(f"Saved/reconstructed summary mismatch: {key}")
    s, tables, f = case["summary"], case["tables"], case["fleet"]
    opened = [r for r in tables["lockers"] if r["open"]]
    capacity = sum(r["capacity"] for r in opened)
    handling = sum(r["pickup"] + r["delivery"] for r in opened)
    if abs(handling - 2 * s["served"]) > 1e-5:
        raise ValueError("Handling load is not twice served order flow")
    levels = Counter(int(round(r["count"])) for r in opened)
    ods = tables["od_times"]
    served = np.array([r["served_per_hour"] for r in ods])
    delivery = np.array([r.get("mean_delivery_minutes") or 0 for r in ods])
    courier = np.array([r["direct_postfetch_minutes"] for r in ods])
    first_distances, first_weights, swaps = [], [], 0.0
    swapped_flow = 0.0
    for p in tables["paths"]:
        if p["kind"] != "service":
            continue
        od, i, flow = p["od"], p["i"], p["flow_per_hour"]
        o = int(data["od_origin"][od])
        first_distances.append(
            1000 * np.linalg.norm(data["origins"][o] - data["hubs"][i])
        )
        first_weights.append(flow)
        count = sum(max(0, len(leg) - 2) for leg in p["legs"])
        swaps += count * flow
        swapped_flow += flow if count else 0.0
    first_distances, first_weights = np.array(first_distances), np.array(first_weights)
    assignment = np.asarray(result["_solution"]["assignment"])
    demand = np.asarray(data["demand"])
    assigned = float(demand @ assignment.sum(axis=(1, 2)))
    empty = np.asarray(s["empty_flow_by_base_pair"])
    pairs = [
        dict(
            origin=f"L{i + 1}",
            destination=f"L{j + 1}",
            flow_per_hour=float(empty[i, j]),
        )
        for i, j in np.argwhere(empty > 1e-4)
    ]
    flow = float(s["empty_flow"])
    empty_occupancy = f.get("empty_flight", 0) + f.get("empty_swap", 0)
    manifest = _json(case["path"] / "manifest.json", {})
    run_summary = _json(case["path"] / "summary.json", {})
    settings = data.get("provenance", {}).get("scenario_settings", {})
    site_costs = _site_cost_metrics(data, rebuilt)
    check_path = case["path"].parent / "selection_check.json"
    selection_check = json.loads(check_path.read_text()) if check_path.is_file() else {}
    share = float(s["served"])
    swap_rows = [
        dict(
            station=f"E{k + 1}",
            open=int(
                result["_solution"]["facility"][data["n_l"] + data["n_h"] + k] > 0.5
            ),
            service_visits_per_hour=0.0,
            reposition_visits_per_hour=0.0,
            drones_swapping=0.0,
        )
        for k in range(data["n_k"])
    ]
    for segment in case["segments"]:
        if segment.get("swap_at_head"):
            k = segment["v_index"] - data["n_l"] - data["n_h"]
            key = (
                "service_visits_per_hour"
                if segment["kind"] == "service"
                else "reposition_visits_per_hour"
            )
            swap_rows[k][key] += segment["flow_per_hour"]
            swap_rows[k]["drones_swapping"] += (
                segment["flow_per_hour"] * segment["swap_hours"]
            )
    return dict(
        case=f"H{case['row']['H']}",
        candidates=dict(bases=data["n_l"], lockers=data["n_h"], swaps=data["n_k"]),
        opened=s["opened"],
        cabinet_total=s["cabinet_total"],
        parameters={
            k: data.get(k, 0.0 if k.endswith("_site_cost") else None)
            for k in (
                "unit_capacity",
                "max_cabinets",
                "base_cost",
                "hub_cost",
                "swap_cost",
                "base_site_cost",
                "hub_site_cost",
                "swap_site_cost",
                "fleet_cost",
                "airline_hh_cost_per_km_hour",
                "airline_connection_cost_per_km_hour",
                "speed",
                "drone_time_cost",
                "hover_cost",
                "epsilon",
                "battery_time",
                "swap_time",
                "terminal_dispatch_time",
                "pickup_handling_time",
                "delivery_handling_time",
                "rider_time_family",
                "rider_slope_minutes_per_straight_km",
            )
        },
        scenario_settings=settings,
        site_costs=site_costs,
        selection_check=selection_check,
        cost_stage=data.get("provenance", {})
        .get("requested_sensitivity", {})
        .get("stage"),
        evaluation_scope=data.get("provenance", {}).get(
            "evaluation_scope",
            data.get("provenance", {})
            .get("requested_sensitivity", {})
            .get("evaluation_scope"),
        ),
        demand=dict(
            total_OD_count=data["n_od"],
            total_per_hour=s["demand"],
            served_per_hour=share,
            served_share=share / s["demand"],
            **case["mode"],
            assigned_potential_per_hour=assigned,
            unassigned_potential_per_hour=float(demand.sum()) - assigned,
            unavailable_assigned_per_hour=assigned - share,
        ),
        fleet=dict(
            components=f,
            total=s["fleet"],
            mission_occupancy=sum(
                v
                for k, v in f.items()
                if k not in ("idle", "empty_flight", "empty_swap", "fleet_slack")
            ),
            empty_occupancy=empty_occupancy,
        ),
        repositioning=dict(
            flow_per_hour=flow,
            pairs=pairs,
            display_threshold_per_hour=1e-4,
            base_pair_matrix=empty.tolist(),
            mean_duration_minutes=60 * empty_occupancy / flow if flow else 0.0,
            cross_base_service_per_hour=max(0.0, s["cross_base_service"]),
            service_base_pair_matrix=s["service_flow_by_base_pair"],
        ),
        locker_capacity=dict(
            capacity_per_hour=capacity,
            handling_per_hour=handling,
            utilization=handling / capacity if capacity else 0.0,
            count_distribution=dict(levels),
            full_sites=sum(
                r["capacity"] - r["pickup"] - r["delivery"] <= 1e-5 for r in opened
            ),
            at_least_95pct_sites=sum(r["utilization"] >= 0.95 - 1e-8 for r in opened),
        ),
        times=dict(
            **s["time_averages_minutes"],
            direct_same_served_OD=s["direct_postfetch_average_minutes"],
            OD_mean_delivery=_weighted(delivery, served),
            OD_mean_courier=_weighted(courier, served),
            fraction_at_least_5min_faster=(
                float(served[courier - delivery >= 5].sum() / share) if share else 0.0
            ),
        ),
        first_leg=dict(
            distance_m=_weighted(first_distances, first_weights),
            colocated_within_1m_flow_fraction=(
                float(first_weights[first_distances <= 1].sum() / share)
                if share
                else 0.0
            ),
        ),
        swapping=dict(
            service_visits_per_hour=swaps,
            service_flow_with_swap_fraction=swapped_flow / share if share else 0.0,
            stations=swap_rows,
            reposition_visits_per_hour=sum(
                x["reposition_visits_per_hour"] for x in swap_rows
            ),
        ),
        costs=dict(
            components=case["costs"],
            total=s["objective"],
            per_potential_order=s["objective"] / s["demand"],
        ),
        solver=_solver_metrics(result, manifest, run_summary),
        validation=dict(
            independent_checks=rebuilt["summary"]["checks"],
            reconciliation=case["reconciliation"],
        ),
    )
