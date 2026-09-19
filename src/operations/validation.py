"""Independent operating, capacity and financial reconciliation."""

from __future__ import annotations

from collections import defaultdict
import numpy as np
from src.model.network import network
from .arrays import _plain, _array, _broadcast, _maximum
from .routes import _trace


def build_operations(data, result):
    """Return all operating tables and independently verified constraint errors."""
    L, H, K, D = (int(data[k]) for k in ("n_l", "n_h", "n_k", "n_od"))
    net = network(data)
    sol = result.get("_solution", result)
    missions = result.get("_missions", result.get("missions", []))
    relocations = result.get("_relocations", result.get("relocations", []))
    levels = np.asarray(data["capacity_levels"], float)
    S = len(levels)
    facility = _array(sol["facility"], (L + H + K,), "facility")
    z = _array(sol["capacity_level"], (H, S), "capacity_level")
    alpha = _array(sol["assignment"], (D, L, L), "assignment")
    rho = _array(sol["rho"], (L,), "rho")
    idle = _array(sol["idle"], (L,), "idle")
    airline = _array(sol["airline"], (net["count"],), "airline")
    demand = _array(data["demand"], (D,), "demand")
    capacity = _array(data["level_capacity"], (H, S), "level_capacity")
    level_cost = _array(data["level_cost"], (H, S), "level_cost")
    fleet = float(sol.get("fleet", result.get("fleet", float("nan"))))
    if not np.isfinite(fleet):
        raise ValueError("Finite explicit continuous fleet size is required")
    served_pair = np.zeros((D, L, L))
    service_ll, empty_ll = np.zeros((L, L)), np.zeros((L, L))
    locker_load = np.zeros((H, 2))
    link_service, link_empty = np.zeros(net["count"]), np.zeros(net["count"])
    weighted = defaultdict(lambda: np.zeros(D))
    pickup_flow = np.zeros(D)
    components = defaultdict(float)
    site_specs = (
        ("base", "base_site_cost", L, facility[:L]),
        ("locker", "hub_site_cost", H, facility[L : L + H]),
        ("swap", "swap_site_cost", K, facility[L + H :]),
    )
    site_costs = [
        dict(
            component=name,
            parameter=key,
            opened_locations=int(np.sum(opened > 0.5)),
            cny_per_hour=float(_broadcast(data.get(key, 0.0), count, key) @ opened),
        )
        for name, key, count, opened in site_specs
    ]
    site_totals = {row["component"]: row["cny_per_hour"] for row in site_costs}
    costs = dict(
        base_fixed=float(_broadcast(data["base_cost"], L, "base_cost") @ facility[:L])
        + site_totals["base"],
        locker_fixed=float(np.sum(level_cost * z)) + site_totals["locker"],
        swap_fixed=float(
            _broadcast(data["swap_cost"], K, "swap_cost") @ facility[L + H :]
        )
        + site_totals["swap"],
        link_fixed=float(net["fees"] @ airline),
        fleet=float(data["fleet_cost"]) * fleet,
        first_rider=0.0,
        terminal_rider=0.0,
        service_flight=0.0,
        aerial_wait=0.0,
        empty_flight=0.0,
    )
    checks = {}
    binary = np.r_[facility, z.ravel(), alpha.ravel(), airline]
    checks["binary_integrality_error"] = _maximum(abs(binary - np.rint(binary)))
    checks["binary_domain_violation"] = _maximum(np.r_[-binary, binary - 1])
    checks["capacity_installation_error"] = _maximum(
        abs(z.sum(axis=1) - facility[L : L + H])
    )
    checks["assignment_count_violation"] = _maximum(alpha.sum(axis=(1, 2)) - 1)
    checks["assignment_source_violation"] = _maximum(alpha.sum(axis=2) - facility[:L])
    checks["assignment_return_violation"] = _maximum(alpha.sum(axis=1) - facility[:L])
    checks["rho_opening_violation"] = _maximum(np.r_[-rho, rho - facility[:L]])
    checks["nonnegative_resource_violation"] = _maximum(np.r_[-idle, -fleet])
    checks["idle_cone_violation"] = _maximum(
        np.sqrt(4 + (idle + rho) ** 2) - (idle + 2 - rho)
    )
    u, v = net["uv"].T
    checks["link_endpoint_violation"] = _maximum(
        np.r_[airline - facility[u], airline - facility[v]]
    )
    if "last_rider_cost" in data:
        expected_terminal = (1 - np.asarray(data["pickup_share"])) * np.asarray(
            data["last_rider_cost"]
        )
        checks["terminal_cost_reconstruction_error"] = _maximum(
            abs(expected_terminal - np.asarray(data["terminal_cost"]))
        )
    if "last_ground_time" in data and "terminal_dispatch_time" in data:
        expected_terminal_time = np.asarray(data["last_ground_time"]) + float(
            data["terminal_dispatch_time"]
        )
        checks["terminal_time_reconstruction_error"] = _maximum(
            abs(expected_terminal_time - np.asarray(data["terminal_time"]))
        )
    coefficient_error = battery_violation = delivery_violation = 0.0
    min_reliability = 1.0
    path_rows, segment_rows = [], []
    for is_service, records, loads in (
        (True, missions, link_service),
        (False, relocations, link_empty),
    ):
        for index, record in enumerate(records):
            flow = float(record["flow"])
            if not np.isfinite(flow) or flow < 0:
                raise ValueError("Path flows must be finite and nonnegative")
            if flow == 0:
                continue
            tr = _trace(data, record, service=is_service)
            l, m = tr["l"], tr["m"]
            for flat, count in tr["arc_counts"].items():
                e = int(net["lookup"][flat])
                if e < 0:
                    raise ValueError("Physical path uses a noncandidate link")
                loads[e] += flow * count
            for component, time in tr["components"].items():
                components[component] += flow * time
            coefficient_error = max(
                coefficient_error, max(tr["coefficient_errors"].values(), default=0.0)
            )
            battery_violation = max(battery_violation, tr["battery_violation"])
            delivery_violation = max(delivery_violation, tr["delivery_violation"])
            min_reliability = min(min_reliability, tr["joint_reliability"])
            fly = sum(s["flight_hours"] for s in tr["segments"])
            if is_service:
                od, i, j = tr["od"], tr["i"], tr["j"]
                service_ll[l, m] += flow
                served_pair[od, l, m] += flow
                locker_load[i, 0] += flow
                locker_load[j, 1] += flow
                pickup_flow[od] += flow * tr["pickup_share"]
                for field in (
                    "time",
                    "first_rider",
                    "prefix",
                    "wait",
                    "terminal",
                    "mean_delivery",
                    "quantile_delivery",
                ):
                    weighted[field][od] += flow * tr[field]
                weighted["fly"][od] += flow * fly
                for field, duration in tr["components"].items():
                    weighted[field][od] += flow * duration
                costs["first_rider"] += flow * tr["first_cost"]
                costs["terminal_rider"] += flow * tr["terminal_cost"]
                costs["service_flight"] += flow * fly * float(data["drone_time_cost"])
                costs["aerial_wait"] += (
                    flow * tr["wait"] * float(data.get("hover_cost", 0.0))
                )
            else:
                empty_ll[l, m] += flow
                costs["empty_flight"] += flow * fly * float(data["drone_time_cost"])
            path_rows.append(
                dict(
                    kind="service" if is_service else "reposition",
                    index=index,
                    od=tr["od"],
                    l=l,
                    m=m,
                    i=tr["i"],
                    j=tr["j"],
                    flow_per_hour=flow,
                    duration_minutes=60 * tr["time"],
                    operating_cost_per_mission=tr["operating_cost"],
                    mean_delivery_minutes=(
                        60 * tr["mean_delivery"] if is_service else None
                    ),
                    delivery_quantile_minutes=(
                        60 * tr["quantile_delivery"] if is_service else None
                    ),
                    joint_reliability=tr["joint_reliability"],
                    battery_interval_minutes=[
                        60 * t for t in tr["battery_interval_consumptions"]
                    ],
                    legs=tr["legs"],
                )
            )
            segment_rows.extend(
                dict(
                    kind="service" if is_service else "reposition",
                    path_index=index,
                    flow_per_hour=flow,
                    **s,
                )
                for s in tr["segments"]
            )
    served = served_pair.sum(axis=(1, 2))
    total = float(served.sum())
    expected = demand[:, None, None] * alpha * rho[None, :, None]
    checks["assignment_service_error"] = _maximum(abs(served_pair - expected))
    checks["served_demand_violation"] = _maximum(served - demand)
    checks["base_flow_balance_error"] = _maximum(
        abs((service_ll + empty_ll).sum(axis=1) - (service_ll + empty_ll).sum(axis=0))
    )
    checks["locker_capacity_violation"] = _maximum(
        locker_load.sum(axis=1) - (capacity * z).sum(axis=1)
    )
    checks["link_activation_violation"] = _maximum(
        link_service + link_empty - net["upper"] * airline
    )
    checks["coefficient_reconstruction_error"] = coefficient_error
    checks["battery_quantile_violation_hours"] = battery_violation
    checks["delivery_quantile_violation_hours"] = delivery_violation
    checks["joint_reliability_shortfall"] = max(
        0.0, 1 - float(data["epsilon"]) - min_reliability
    )
    actual_occupancy = float(idle.sum()) + sum(components.values())
    checks["fleet_occupancy_violation"] = max(0.0, actual_occupancy - fleet)
    costs["direct_rider"] = float(np.dot(demand - served, data["ground"]))
    objective = sum(costs.values())
    checks["objective_reconstruction_error"] = abs(
        objective - float(result["objective"])
    )
    for field, actual in (
        ("served", total),
        ("fleet", fleet),
        ("airline_fixed_cost", costs["link_fixed"]),
    ):
        if field in result:
            checks["saved_" + field + "_error"] = abs(actual - float(result[field]))
    if "hub_count" in sol:
        checks["saved_cabinet_count_error"] = _maximum(
            abs(np.asarray(sol["hub_count"]) - z @ levels)
        )
    if "empty" in sol:
        checks["saved_empty_flow_error"] = _maximum(
            abs(np.asarray(sol["empty"]) - empty_ll)
        )
    if "airline_load" in result or "airline_load" in sol:
        saved = result.get("airline_load", sol.get("airline_load"))
        checks["saved_link_load_error"] = _maximum(
            abs(np.asarray(saved) - link_service - link_empty)
        )
    if result.get("bound") is not None:
        checks["bound_above_incumbent"] = max(0.0, float(result["bound"]) - objective)
    checks["passed"] = all(
        v <= (1e-3 if "objective" in k or "cost" in k else 1e-4)
        for k, v in checks.items()
    )
    direct = np.asarray(
        data.get(
            "direct_postfetch_hours",
            data.get("direct_rider_hours", data["delivery_deadline"]),
        ),
        float,
    )
    observed = np.asarray(
        data.get("observed_postfetch_hours", np.full(D, np.nan)), float
    )
    observed_count = np.asarray(data.get("observed_postfetch_count", np.zeros(D)), int)
    od_rows = []
    for od in range(D):
        row = dict(
            od=od,
            origin=int(data["od_origin"][od]),
            destination=int(data["od_destination"][od]),
            demand_per_hour=float(demand[od]),
            served_per_hour=float(served[od]),
            direct_per_hour=float(demand[od] - served[od]),
            pickup_per_hour=float(pickup_flow[od]),
            home_per_hour=float(served[od] - pickup_flow[od]),
            direct_postfetch_minutes=60 * float(direct[od]),
            observed_postfetch_minutes=(
                60 * float(observed[od]) if np.isfinite(observed[od]) else None
            ),
            observed_count=int(observed_count[od]),
            deadline_minutes=60 * float(data["delivery_deadline"][od]),
        )
        for key, values in weighted.items():
            row[key + "_minutes"] = (
                60 * float(values[od]) / served[od] if served[od] > 0 else None
            )
        od_rows.append(row)
    bases = [
        dict(
            base=l,
            open=int(facility[l] > 0.5),
            rho=float(rho[l]),
            idle=float(idle[l]),
            service_out=float(service_ll[l].sum()),
            service_in=float(service_ll[:, l].sum()),
            empty_out=float(empty_ll[l].sum()),
            empty_in=float(empty_ll[:, l].sum()),
        )
        for l in range(L)
    ]
    lockers = [
        dict(
            locker=i,
            open=int(facility[L + i] > 0.5),
            count=float(z[i] @ levels),
            capacity=float(capacity[i] @ z[i]),
            pickup=float(locker_load[i, 0]),
            delivery=float(locker_load[i, 1]),
            utilization=(
                float(locker_load[i].sum() / (capacity[i] @ z[i]))
                if capacity[i] @ z[i] > 0
                else 0.0
            ),
        )
        for i in range(H)
    ]
    airlines = [
        dict(
            link=e,
            u=int(u),
            v=int(v),
            open=int(airline[e] > 0.5),
            service_flow_per_hour=float(link_service[e]),
            empty_flow_per_hour=float(link_empty[e]),
            total_flow_per_hour=float(link_service[e] + link_empty[e]),
            fee_per_hour=float(net["fees"][e]),
            flight_minutes=60 * float(data["flight_times"][u, v]),
        )
        for e, (u, v) in enumerate(net["uv"])
    ]
    summary = dict(
        name=data.get("name", "path model"),
        objective=objective,
        solver={
            k: result.get(k)
            for k in (
                "method",
                "status",
                "gap",
                "bound",
                "wall",
                "total_wall",
                "termination_reason",
            )
        },
        demand=float(demand.sum()),
        served=total,
        served_fraction=total / float(demand.sum()) if demand.sum() else 0.0,
        opened=dict(
            bases=int(np.sum(facility[:L] > 0.5)),
            lockers=int(np.sum(facility[L : L + H] > 0.5)),
            swaps=int(np.sum(facility[L + H :] > 0.5)),
        ),
        cabinet_total=float(np.sum(z @ levels)),
        open_links=int(np.sum(airline > 0.5)),
        used_links=int(np.sum(link_service + link_empty > 1e-6)),
        fleet=fleet,
        fleet_occupancy=actual_occupancy,
        fleet_slack=fleet - actual_occupancy,
        service_flow_by_base_pair=service_ll,
        empty_flow_by_base_pair=empty_ll,
        cross_base_service=float(total - np.trace(service_ll)),
        empty_flow=float(empty_ll.sum()),
        min_joint_reliability=min_reliability,
        time_averages_minutes={
            k: 60 * float(v.sum()) / total if total else None
            for k, v in weighted.items()
        },
        direct_postfetch_average_minutes=(
            60 * float(direct @ served) / total if total else None
        ),
        observed_postfetch_average_minutes=(
            60
            * float(observed[np.isfinite(observed)] @ served[np.isfinite(observed)])
            / float(served[np.isfinite(observed)].sum())
            if served[np.isfinite(observed)].sum() > 0
            else None
        ),
        terminal_dispatch_minutes=60 * float(data.get("terminal_dispatch_time", 0.0)),
        model_parameters={
            k: data.get(k)
            for k in (
                "epsilon",
                "battery_time",
                "rider_time_family",
                "hover_cost",
                "drone_time_cost",
            )
        },
        site_cost_parameters={key: data.get(key, 0.0) for _, key, _, _ in site_specs},
        terminal_calibration=data.get("provenance", {}).get(
            "calibration", data.get("terminal_calibration", {})
        ),
        checks=checks,
    )
    fleet_rows = [dict(component="idle", drones=float(idle.sum()))]
    fleet_rows += [dict(component=k, drones=v) for k, v in components.items()]
    fleet_rows.append(dict(component="fleet_slack", drones=fleet - actual_occupancy))
    return _plain(
        dict(
            summary=summary,
            tables=dict(
                bases=bases,
                lockers=lockers,
                airlines=airlines,
                od_times=od_rows,
                paths=path_rows,
                costs=[dict(component=k, cny_per_hour=v) for k, v in costs.items()],
                site_costs=site_costs,
                fleet=fleet_rows,
            ),
            segments=segment_rows,
        )
    )
