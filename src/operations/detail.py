"""Detailed, reconciled operating accounts for an independently checked solution.

Drone-hours/hour is a mean number of drones, not an integer fleet.  The common
continuation window overlaps the first loaded flight: it must never be added to
busy time again.  Locker load includes reservations and is not measured physical
meal occupancy.  Base busy resources below are attributed to task departure.
"""

from collections import defaultdict
import math

import numpy as np


STAGES = (
    "first_dispatch",
    "first_loaded",
    "continuation_transfer",
    "second_loaded",
    "return",
    "empty",
)
BUSY = ("service_flight", "first_pickup_wait", "second_pickup_wait", "empty_flight")


def _ratio(numerator, denominator):
    return float(numerator / denominator) if denominator > 0 else None


def _coordinates(inputs, key, index):
    values = inputs.get(key) if isinstance(inputs, dict) else None
    if values is None:
        return dict(x_km=None, y_km=None)
    return dict(x_km=float(values[index][0]), y_km=float(values[index][1]))


def enrich_operations(inputs, data, result, payload):
    """Add details after the original physical and financial validation passed.

    Existing table fields remain compatible.  We read actual node routes and
    checked order/path rows; no solver coefficient or model resource builder is
    used to derive the decomposition.  Reconciliation errors join the ordinary
    validation residuals so a broken accounting export cannot silently pass.
    """
    tables, summary = payload["tables"], payload["summary"]
    sol = result["_solution"]
    pickup_share = inputs.get("pickup_share") if isinstance(inputs, dict) else None
    if pickup_share is not None:
        pickup_share = np.asarray(pickup_share, float)
        if (
            pickup_share.shape != (data.n_od, data.n_h)
            or not np.isfinite(pickup_share).all()
            or np.any((pickup_share < 0) | (pickup_share > 1))
        ):
            raise ValueError(
                "Saved pickup_share must be an OD-by-locker probability matrix"
            )
    assignment = np.asarray(sol["assignment"], float)
    facility = np.asarray(sol["facility"], float)
    airlines = np.asarray(sol["airline"], float)
    rho, idle = np.asarray(sol["rho"], float), np.asarray(sol["idle"], float)
    base = [defaultdict(float) for _ in range(data.n_l)]
    locker = [defaultdict(float) for _ in range(data.n_h)]
    od = [defaultdict(float) for _ in range(data.n_od)]
    stage = {name: defaultdict(float) for name in STAGES}
    busy = dict.fromkeys(BUSY, 0.0)
    arc = defaultdict(lambda: defaultdict(float))
    pair_actual = defaultdict(float)
    single_actual = defaultdict(float)
    pool_actual = defaultdict(lambda: defaultdict(float))
    path_rows = {(row["kind"], row["path"]): row for row in tables["paths"]}
    order_rows = defaultdict(list)
    for row in tables["order_paths"]:
        order_rows[row["path"]].append(row)
    detailed_cost = defaultdict(float)
    for kind, records in (
        ("service", result.get("_missions", [])),
        ("empty", result.get("_relocations", [])),
    ):
        for record in records:
            flow, path = float(record["flow"]), record["path"]
            if flow == 0:
                continue
            pid = int(path["id"])
            p = path_rows[kind, pid]
            origin, target = p["departure"], p["return_base"]
            double = kind == "service" and path.get("second_od") is not None
            names = (
                ("empty",)
                if kind == "empty"
                else (
                    (
                        "first_dispatch",
                        "first_loaded",
                        "continuation_transfer",
                        "second_loaded",
                        "return",
                    )
                    if double
                    else ("first_dispatch", "first_loaded", "return")
                )
            )
            durations = []
            for name, leg in zip(names, path["legs"]):
                duration = sum(
                    float(data.flight_times[u, v]) for u, v in zip(leg, leg[1:])
                )
                durations.append(duration)
                stage[name]["task_legs_per_hour"] += flow
                stage[name]["busy_drones"] += flow * duration
                # A zero-length transfer is a task leg but no physical flight.
                for u, v in zip(leg, leg[1:]):
                    arc[u, v][kind + "_per_hour"] += flow
                    if name != "empty":
                        arc[u, v][name + "_per_hour"] += flow
                    stage[name]["traversals_per_hour"] += flow
            flight_drones = flow * sum(durations)
            base[origin][kind + "_tasks_depart_per_hour"] += flow
            base[target][kind + "_tasks_arrive_per_hour"] += flow
            pool_actual[origin, target][kind + "_tasks_per_hour"] += flow
            p.update(
                first_od=path.get("first_od"),
                second_od=path.get("second_od"),
                first_pickup=path.get("pickups", [None])[0],
                first_delivery=path.get("deliveries", [None])[0],
                second_pickup=path["pickups"][1] if double else None,
                second_delivery=path["deliveries"][1] if double else None,
                busy_drones=flow * p["mean_operating_minutes"] / 60,
                busy_minutes_per_hour=flow * p["mean_operating_minutes"],
                operating_cost_per_hour=flow * p["unit_cost"],
                endurance_margin_minutes=60 * data.operating_limit
                - p["mean_operating_minutes"],
            )
            for name in STAGES:
                p[name + "_minutes"] = (
                    60 * durations[names.index(name)] if name in names else 0.0
                )
            if kind == "empty":
                busy["empty_flight"] += flight_drones
                base[origin]["empty_flight_drones"] += flight_drones
                detailed_cost["empty_flight"] += data.flight_cost * flight_drones
                p.update(first_wait_minutes=0.0, second_wait_minutes=0.0)
                continue
            busy["service_flight"] += flight_drones
            base[origin]["service_flight_drones"] += flight_drones
            detailed_cost["service_flight"] += data.flight_cost * flight_drones
            count = 2 if double else 1
            base[origin][
                ("double" if double else "single") + "_tasks_depart_per_hour"
            ] += flow
            base[origin]["served_orders_depart_per_hour"] += count * flow
            base[target]["served_orders_arrive_per_hour"] += count * flow
            pool_actual[origin, target]["served_orders_per_hour"] += count * flow
            if double:
                pair_actual[origin, target, path["first_od"], path["second_od"]] += flow
            else:
                single_actual[origin, target, path["first_od"]] += flow
            p.update(first_wait_minutes=0.0, second_wait_minutes=0.0)
            for row in order_rows[pid]:
                r, i, j = row["od"], row["pickup"], row["delivery"]
                position = row["position"]
                role = "first" if position == 1 else "second"
                before = durations[0 if position == 1 else 2]
                loaded = durations[1 if position == 1 else 3]
                pickup = row["pickup_hours"]
                # First pickup clock: t0+w1. Second: transfer+delta/2+w2.
                wait = (
                    pickup
                    - before
                    - (data.effective_delta / 2 if position == 2 else 0.0)
                )
                # Guard against floating cancellation without hiding discrepancies.
                if abs(wait) < 1e-13:
                    wait = 0.0
                wait_drones = flow * wait
                busy[role + "_pickup_wait"] += wait_drones
                base[origin][role + "_wait_drones"] += wait_drones
                locker[i]["pickup_wait_drones"] += wait_drones
                p[role + "_wait_minutes"] = 60 * wait
                detailed_cost["drone_wait"] += data.waiting_cost * wait_drones
                detailed_cost["first_rider"] += flow * data.first_cost[r, i]
                detailed_cost["terminal_rider"] += flow * data.terminal_cost[r, j]
                pickup_reservation = flow * pickup
                delivery_reservation = flow * (
                    pickup + loaded + data.removal_mean[r, j]
                )
                locker[i]["pickup_orders_per_hour"] += flow
                locker[j]["delivery_orders_per_hour"] += flow
                locker[i][role + "_pickup_orders_per_hour"] += flow
                locker[j][role + "_delivery_orders_per_hour"] += flow
                locker[i]["pickup_reservation_mean"] += pickup_reservation
                locker[j]["delivery_reservation_mean"] += delivery_reservation
                od[r][role + "_receipt_weight"] += flow * row["receipt_hours"]
                od[r][role + "_flow"] += flow
                od[r]["pickup_weight"] += flow * pickup
                od[r]["loaded_weight"] += flow * loaded
                od[r]["terminal_weight"] += flow * data.receipt_mean[r, j]
                od[r]["rider_weight"] += flow * data.rider_mean[r, i]
                od[r]["drone_wait_weight"] += wait_drones
                od[r]["first_rider_cost_per_hour"] += flow * data.first_cost[r, i]
                od[r]["terminal_rider_cost_per_hour"] += flow * data.terminal_cost[r, j]
                row.update(
                    departure=origin,
                    return_base=target,
                    pickup_minutes=60 * pickup,
                    loaded_flight_minutes=60 * loaded,
                    terminal_minutes=60 * data.receipt_mean[r, j],
                    origin_rider_minutes=60 * data.rider_mean[r, i],
                    drone_pickup_wait_minutes=60 * wait,
                    receipt_minutes=60 * row["receipt_hours"],
                    receipt_margin_minutes=60
                    * (data.deadline[r] - row["receipt_hours"]),
                    pickup_reservation_minutes=60 * pickup,
                    delivery_reservation_minutes=60
                    * (pickup + loaded + data.removal_mean[r, j]),
                    first_rider_cost_per_hour=flow * data.first_cost[r, i],
                    terminal_rider_cost_per_hour=flow * data.terminal_cost[r, j],
                )
                if pickup_share is not None:
                    self_pickup = flow * pickup_share[r, j]
                    terminal_courier = flow - self_pickup
                    for account in (locker[j], od[r], base[origin]):
                        account["self_pickup_orders_per_hour"] += self_pickup
                        account["terminal_courier_orders_per_hour"] += terminal_courier
                    row.update(
                        self_pickup_probability=float(pickup_share[r, j]),
                        self_pickup_orders_per_hour=float(self_pickup),
                        terminal_courier_orders_per_hour=float(terminal_courier),
                    )

    arc_rows, nodes = [], [defaultdict(float) for _ in range(data.n_nodes)]
    for u in range(data.n_nodes):
        for v in range(data.n_nodes):
            if u == v:
                continue
            load = arc[u, v]
            service, empty = load["service_per_hour"], load["empty_per_hour"]
            total = service + empty
            duration = float(data.flight_times[u, v])
            row = dict(
                u=u,
                v=v,
                u_label=_node_label(data, u),
                v_label=_node_label(data, v),
                allowed=bool(data.allowed_links[u, v]),
                opened=bool(airlines[u, v] > 0.5),
                service_per_hour=service,
                empty_per_hour=empty,
                total_per_hour=total,
                flight_minutes=60 * duration if math.isfinite(duration) else None,
                busy_drones=total * duration if total else 0.0,
                fixed_cost_per_hour=float(data.link_cost[u, v] * airlines[u, v]),
                **{
                    name + "_per_hour": load[name + "_per_hour"]
                    for name in STAGES
                    if name != "empty"
                },
            )
            arc_rows.append(row)
            for kind in ("service", "empty"):
                nodes[u][kind + "_outgoing_flights_per_hour"] += load[
                    kind + "_per_hour"
                ]
                nodes[v][kind + "_incoming_flights_per_hour"] += load[
                    kind + "_per_hour"
                ]
    for i, row in enumerate(tables["bases"]):
        assigned = assignment[:, i, :].sum(axis=1)
        offered = float(data.demand @ assigned)
        b = base[i]
        components = (
            "service_flight_drones",
            "first_wait_drones",
            "second_wait_drones",
            "empty_flight_drones",
        )
        base_busy = sum(b[name] for name in components)
        attributed = idle[i] + base_busy
        minimum = _ratio(rho[i], 1 - rho[i])
        names = (
            "service_tasks_depart_per_hour",
            "service_tasks_arrive_per_hour",
            "empty_tasks_depart_per_hour",
            "empty_tasks_arrive_per_hour",
            "single_tasks_depart_per_hour",
            "double_tasks_depart_per_hour",
            "served_orders_depart_per_hour",
            "served_orders_arrive_per_hour",
            *components,
        )
        row.update(
            **_coordinates(inputs, "bases", i),
            **{name: float(b[name]) for name in names},
            assigned_od_count=int(np.count_nonzero(assigned > 0.5)),
            offered_orders_per_hour=offered,
            expected_served_orders_per_hour=float(offered * rho[i]),
            direct_assigned_orders_per_hour=float(offered * (1 - rho[i])),
            busy_drones=float(base_busy),
            attributed_total_drones=float(attributed),
            attributed_busy_fraction=_ratio(base_busy, attributed),
            idle_minimum=minimum,
            idle_slack=float(idle[i] - minimum) if minimum is not None else None,
            flow_balance_residual=float(row["outflow"] - row["inflow"]),
            mean_service_task_minutes=_ratio(
                60 * (base_busy - b["empty_flight_drones"]),
                b["service_tasks_depart_per_hour"],
            ),
            mean_empty_task_minutes=_ratio(
                60 * b["empty_flight_drones"], b["empty_tasks_depart_per_hour"]
            ),
            self_pickup_orders_per_hour=float(b["self_pickup_orders_per_hour"])
            if pickup_share is not None
            else None,
            terminal_courier_orders_per_hour=float(
                b["terminal_courier_orders_per_hour"]
            )
            if pickup_share is not None
            else None,
        )
    for i, row in enumerate(tables["lockers"]):
        locker_values = locker[i]
        names = (
            "pickup_orders_per_hour",
            "delivery_orders_per_hour",
            "first_pickup_orders_per_hour",
            "first_delivery_orders_per_hour",
            "second_pickup_orders_per_hour",
            "second_delivery_orders_per_hour",
            "pickup_reservation_mean",
            "delivery_reservation_mean",
            "pickup_wait_drones",
        )
        node = nodes[data.n_l + i]
        incoming = sum(
            node[k + "_incoming_flights_per_hour"] for k in ("service", "empty")
        )
        outgoing = sum(
            node[k + "_outgoing_flights_per_hour"] for k in ("service", "empty")
        )
        row.update(
            **_coordinates(inputs, "hubs", i),
            **{name: float(locker_values[name]) for name in names},
            incoming_flights_per_hour=incoming,
            outgoing_flights_per_hour=outgoing,
            mean_pickup_reservation_minutes=_ratio(
                60 * locker_values["pickup_reservation_mean"],
                locker_values["pickup_orders_per_hour"],
            ),
            mean_delivery_reservation_minutes=_ratio(
                60 * locker_values["delivery_reservation_mean"],
                locker_values["delivery_orders_per_hour"],
            ),
            reservation_load_fraction=_ratio(
                row["mean_occupancy"], row["mean_threshold"]
            ),
            compartments_load_fraction=_ratio(
                row["mean_occupancy"], row["compartments"]
            ),
            capacity_slack=float(row["mean_threshold"] - row["mean_occupancy"]),
            probability_slack=float(
                row["allowed_probability"] - row["potential_overflow_probability"]
            ),
            handling_events_per_hour=locker_values["pickup_orders_per_hour"]
            + locker_values["delivery_orders_per_hour"],
            self_pickup_orders_per_hour=float(
                locker_values["self_pickup_orders_per_hour"]
            )
            if pickup_share is not None
            else None,
            terminal_courier_orders_per_hour=float(
                locker_values["terminal_courier_orders_per_hour"]
            )
            if pickup_share is not None
            else None,
        )
    for r, row in enumerate(tables["orders"]):
        selected = np.argwhere(assignment[r] > 0.5)
        selected_pair = tuple(map(int, selected[0])) if len(selected) else None
        received = row["mean_receipt_minutes"]
        row.update(
            assigned=bool(len(selected)),
            departure=selected_pair[0] if selected_pair else None,
            return_base=selected_pair[1] if selected_pair else None,
            assigned_availability=float(rho[selected_pair[0]])
            if selected_pair
            else 0.0,
            drone_fraction=_ratio(row["drone_per_hour"], row["demand_per_hour"]),
            direct_courier_cost_per_hour=float(
                row["direct_per_hour"] * data.ground_cost[r]
            ),
            all_courier_cost_per_hour=float(
                row["demand_per_hour"] * data.ground_cost[r]
            ),
            receipt_margin_minutes=row["deadline_minutes"] - received
            if received is not None
            else None,
            unassigned_direct_orders_per_hour=float(row["demand_per_hour"])
            if selected_pair is None
            else 0.0,
            unavailability_direct_orders_per_hour=float(row["direct_per_hour"])
            if selected_pair
            else 0.0,
            self_pickup_orders_per_hour=float(od[r]["self_pickup_orders_per_hour"])
            if pickup_share is not None
            else None,
            terminal_courier_orders_per_hour=float(
                od[r]["terminal_courier_orders_per_hour"]
            )
            if pickup_share is not None
            else None,
        )
        for role in ("first", "second"):
            row[role + "_mean_receipt_minutes"] = _ratio(
                60 * od[r][role + "_receipt_weight"], od[r][role + "_flow"]
            )
        for name, weight in (
            ("pickup", "pickup"),
            ("loaded_flight", "loaded"),
            ("terminal", "terminal"),
            ("origin_rider", "rider"),
            ("drone_pickup_wait", "drone_wait"),
        ):
            row["mean_" + name + "_minutes"] = _ratio(
                60 * od[r][weight + "_weight"], row["drone_per_hour"]
            )
        row.update(
            {
                name: float(od[r][name])
                for name in (
                    "first_rider_cost_per_hour",
                    "terminal_rider_cost_per_hour",
                )
            }
        )

    pair_rows, single_rows, assignment_rows = [], [], []
    for pool in tables["pools"]:
        departure, arrival = pool["departure"], pool["return_base"]
        h = data.demand * assignment[:, departure, arrival]
        lam = float(h.sum())
        q = float(pool["continuation_probability"])
        pi = h / lam if lam else np.zeros(data.n_od)
        theta = (
            np.linalg.solve(np.eye(data.n_od) + q * pi[:, None] * data.kappa.T, h)
            if lam
            else h
        )
        offers = theta[:, None] * q * data.kappa * pi
        singles = theta - offers.sum(axis=1)
        actual = pool_actual[departure, arrival]
        pool.update(
            availability=float(rho[departure]),
            assigned_od_count=int(
                np.count_nonzero(assignment[:, departure, arrival] > 0.5)
            ),
            offered_single_tasks_per_hour=float(singles.sum()),
            offered_double_tasks_per_hour=float(offers.sum()),
            expected_served_orders_per_hour=float(lam * rho[departure]),
            served_orders_per_hour=float(actual["served_orders_per_hour"]),
            single_tasks_per_hour=float(
                sum(single_actual[departure, arrival, r] for r in range(data.n_od))
            ),
            double_tasks_per_hour=float(
                sum(
                    pair_actual[departure, arrival, r, s]
                    for r in range(data.n_od)
                    for s in range(data.n_od)
                )
            ),
            empty_tasks_per_hour=float(actual["empty_tasks_per_hour"]),
        )
        for r in range(data.n_od):
            assignment_rows.append(
                dict(
                    od=r,
                    departure=departure,
                    return_base=arrival,
                    assigned=bool(assignment[r, departure, arrival] > 0.5),
                    assignment=float(assignment[r, departure, arrival]),
                    demand_per_hour=float(data.demand[r]),
                    availability=float(rho[departure]),
                    offered_orders_per_hour=float(h[r]),
                    expected_served_orders_per_hour=float(h[r] * rho[departure]),
                )
            )
            single_rows.append(
                dict(
                    departure=departure,
                    return_base=arrival,
                    od=r,
                    reconstructed_head_per_hour=float(theta[r]),
                    offered_tasks_per_hour=float(singles[r]),
                    target_tasks_per_hour=float(rho[departure] * singles[r]),
                    actual_tasks_per_hour=float(single_actual[departure, arrival, r]),
                )
            )
            for s in range(data.n_od):
                pair_rows.append(
                    dict(
                        departure=departure,
                        return_base=arrival,
                        first_od=r,
                        second_od=s,
                        kappa=float(data.kappa[r, s]),
                        offered_tasks_per_hour=float(offers[r, s]),
                        target_tasks_per_hour=float(rho[departure] * offers[r, s]),
                        actual_tasks_per_hour=float(
                            pair_actual[departure, arrival, r, s]
                        ),
                    )
                )

    node_rows = []
    for i, values in enumerate(nodes):
        is_base = i < data.n_l
        local = i if is_base else i - data.n_l
        n = dict(
            node=i,
            label=_node_label(data, i),
            kind="base" if is_base else "locker",
            local_index=local,
            opened=bool(facility[i] > 0.5),
            **_coordinates(inputs, "bases" if is_base else "hubs", local),
            **{
                k + "_" + direction + "_flights_per_hour": values[
                    k + "_" + direction + "_flights_per_hour"
                ]
                for k in ("service", "empty")
                for direction in ("incoming", "outgoing")
            },
        )
        n["incoming_flights_per_hour"] = (
            n["service_incoming_flights_per_hour"]
            + n["empty_incoming_flights_per_hour"]
        )
        n["outgoing_flights_per_hour"] = (
            n["service_outgoing_flights_per_hour"]
            + n["empty_outgoing_flights_per_hour"]
        )
        n["flight_balance_residual"] = (
            n["outgoing_flights_per_hour"] - n["incoming_flights_per_hour"]
        )
        n["pickup_orders_per_hour"] = (
            0.0 if is_base else locker[local]["pickup_orders_per_hour"]
        )
        n["delivery_orders_per_hour"] = (
            0.0 if is_base else locker[local]["delivery_orders_per_hour"]
        )
        n["pickup_wait_drones"] = (
            0.0 if is_base else locker[local]["pickup_wait_drones"]
        )
        node_rows.append(n)

    old_costs = {row["component"]: row["cny_per_hour"] for row in tables["costs"]}
    for name in ("bases", "lockers", "links", "fleet", "direct_courier"):
        detailed_cost[name] = old_costs[name]
    for name in (
        "first_rider",
        "terminal_rider",
        "service_flight",
        "drone_wait",
        "empty_flight",
    ):
        detailed_cost.setdefault(name, 0.0)
    demand, served = summary["demand"], summary["served"]
    busy_total = float(sum(busy.values()))
    service_busy = busy_total - busy["empty_flight"]
    baseline = float(data.demand @ data.ground_cost)
    receipt = sum(
        row["drone_per_hour"] * (row["mean_receipt_minutes"] or 0.0)
        for row in tables["orders"]
    )
    receipt_margin = sum(
        row["drone_per_hour"] * (row["receipt_margin_minutes"] or 0.0)
        for row in tables["orders"]
    )
    summary.update(
        max_orders=data.max_orders,
        service_fraction=_ratio(served, demand),
        all_courier_cost=baseline,
        savings_per_hour=baseline - summary["objective"],
        savings_fraction=_ratio(baseline - summary["objective"], baseline),
        cost_per_order=_ratio(summary["objective"], demand),
        drone_system_cost_per_drone_order=_ratio(
            summary["objective"] - old_costs["direct_courier"], served
        ),
        busy_drones=busy_total,
        service_busy_drones=float(service_busy),
        idle_drones=float(idle.sum()),
        fleet_slack=float(summary["fleet"] - idle.sum() - busy_total),
        busy_fraction=_ratio(busy_total, summary["fleet"]),
        mean_task_minutes=_ratio(60 * service_busy, summary["task_flights"]),
        mean_empty_task_minutes=_ratio(
            60 * busy["empty_flight"], summary["empty_tasks"]
        ),
        mean_receipt_minutes=_ratio(receipt, served),
        mean_receipt_margin_minutes=_ratio(receipt_margin, served),
        orders_per_service_task=_ratio(served, summary["task_flights"]),
        double_task_fraction=_ratio(summary["double_tasks"], summary["task_flights"]),
        continued_order_fraction=_ratio(summary["double_tasks"], served),
        single_order_fraction=_ratio(summary["single_tasks"], served),
        opened_links=int(np.count_nonzero(airlines > 0.5)),
        assigned_od_count=sum(row["assigned"] for row in tables["orders"]),
        served_od_count=sum(row["drone_per_hour"] > 1e-8 for row in tables["orders"]),
        busy_components={key: float(value) for key, value in busy.items()},
        self_pickup_orders_per_hour=float(
            sum(values["self_pickup_orders_per_hour"] for values in od)
        )
        if pickup_share is not None
        else None,
        terminal_courier_orders_per_hour=float(
            sum(values["terminal_courier_orders_per_hour"] for values in od)
        )
        if pickup_share is not None
        else None,
        terminal_choice_available=pickup_share is not None,
        unassigned_direct_orders_per_hour=float(
            sum(row["unassigned_direct_orders_per_hour"] for row in tables["orders"])
        ),
        unavailability_direct_orders_per_hour=float(
            sum(
                row["unavailability_direct_orders_per_hour"] for row in tables["orders"]
            )
        ),
        cross_base_service_tasks_per_hour=float(
            sum(
                values["service_tasks_per_hour"]
                for (departure, arrival), values in pool_actual.items()
                if departure != arrival
            )
        ),
        cross_base_service_fraction=_ratio(
            sum(
                values["service_tasks_per_hour"]
                for (departure, arrival), values in pool_actual.items()
                if departure != arrival
            ),
            summary["task_flights"],
        ),
    )
    tables.update(
        nodes=node_rows,
        arcs=arc_rows,
        task_pairs=pair_rows,
        task_singles=single_rows,
        od_assignments=assignment_rows,
        flight_stages=[
            dict(
                stage=name,
                task_legs_per_hour=values["task_legs_per_hour"],
                traversals_per_hour=values["traversals_per_hour"],
                mean_minutes=_ratio(
                    60 * values["busy_drones"], values["task_legs_per_hour"]
                ),
                busy_drones=values["busy_drones"],
                busy_minutes_per_hour=60 * values["busy_drones"],
            )
            for name, values in stage.items()
        ],
        busy_components=[
            dict(
                component=name,
                drones=float(value),
                busy_minutes_per_hour=60 * value,
                fleet_fraction=_ratio(value, summary["fleet"]),
            )
            for name, value in busy.items()
        ],
        cost_detail=[
            dict(
                component=name,
                cny_per_hour=float(value),
                cny_per_total_order=_ratio(value, demand),
                cny_per_drone_order=_ratio(value, served)
                if name != "direct_courier"
                else None,
                objective_fraction=_ratio(value, summary["objective"]),
            )
            for name, value in detailed_cost.items()
        ],
    )
    tables["market_nodes"] = _market_nodes(inputs, tables["orders"])
    _reconcile(payload, tables, busy)
    tables["metric_dictionary"] = metric_dictionary(tables)
    payload["metric_notes"] = [
        "All rates refer to the same one-hour planning period; figures are model means, not simulation observations.",
        "rho/availability is the probability of finding an available drone at a base; it is not utilization.",
        "Busy drones = task rate (tasks/hour) times mean task duration (hours); busy minutes/hour is 60 times this quantity.",
        "Fleet and idle inventories are continuous planning variables. Rounding a fleet does not establish feasibility of an integer fleet plan.",
        "Base busy drones are attributed to the departure base of the complete task; drones need not physically remain at that base.",
        "The continuation window is inside the first loaded flight. delta/2 appears in the second order's own-request clock, not as additional drone busy time.",
        "Locker mean_occupancy is reservation-inclusive offered load; pickup_reservation_mean and delivery_reservation_mean sum to it. Physical meal occupancy is not separately identified here.",
        "Locker capacity_slack is Poisson mean-load threshold minus reservation load, not empty compartments. Tail probabilities use P(Poisson(load)>compartments).",
        "Order receipt times are conditional flow-weighted means of served drone orders and include each order's own terminal leg. No observed or simulated latency quantiles are claimed.",
        "Zero-flow bases, lockers, ODs and all directed site pairs are retained. A task leg with identical endpoints has zero traversals but still counts as a stage occurrence.",
        "Task-pair offered rates are reconstructed before availability thinning; target and actual task rates include departure-base rho. Two-order tasks count twice toward served orders.",
        "Cost-per-total-order divides by all demand; drone-system cost excludes direct-courier cost and divides by drone-served orders. This is a system-average accounting ratio, not a marginal fare.",
        "Terminal self-pickup/terminal-courier flows use the explicitly saved exogenous pickup_share array; blank values mean this auxiliary array was not saved. These are expected choice flows, not observed choices.",
    ]
    return payload


def _node_label(data, i):
    return f"B{i}" if i < data.n_l else f"H{i - data.n_l}"


def _market_nodes(inputs, orders):
    """Aggregate origin/destination site flows if the saved input has geometry."""
    if not isinstance(inputs, dict):
        return []
    rows = []
    for kind, key, index_key in (
        ("origin", "origins", "od_origin"),
        ("destination", "destinations", "od_destination"),
    ):
        if key not in inputs or index_key not in inputs:
            continue
        indices = np.asarray(inputs[index_key], int)
        for i in range(len(inputs[key])):
            included = [row for row in orders if indices[row["od"]] == i]
            row = dict(
                kind=kind,
                node=i,
                label=("O" if kind == "origin" else "D") + str(i),
                **_coordinates(inputs, key, i),
                od_count=len(included),
            )
            for name in (
                "demand_per_hour",
                "drone_per_hour",
                "direct_per_hour",
                "first_per_hour",
                "second_per_hour",
            ):
                row[name] = float(sum(o[name] for o in included))
            row["drone_fraction"] = _ratio(
                row["drone_per_hour"], row["demand_per_hour"]
            )
            rows.append(row)
    return rows


def _reconcile(payload, tables, busy):
    summary = payload["summary"]
    residuals = summary["checks"]["residuals"]
    fleet = {row["component"]: row["drones"] for row in tables["fleet"]}
    checks = {
        "detail_service_busy": abs(
            sum(busy[k] for k in BUSY if k != "empty_flight") - fleet["service"]
        ),
        "detail_empty_busy": abs(busy["empty_flight"] - fleet["repositioning"]),
        "detail_stage_flight": abs(
            sum(row["busy_drones"] for row in tables["flight_stages"])
            - busy["service_flight"]
            - busy["empty_flight"]
        ),
        "detail_arc_flight": abs(
            sum(row["busy_drones"] for row in tables["arcs"])
            - busy["service_flight"]
            - busy["empty_flight"]
        ),
        "detail_base_busy": abs(
            sum(row["busy_drones"] for row in tables["bases"]) - sum(busy.values())
        ),
        "detail_pickup_orders": abs(
            sum(row["pickup_orders_per_hour"] for row in tables["lockers"])
            - summary["served"]
        ),
        "detail_delivery_orders": abs(
            sum(row["delivery_orders_per_hour"] for row in tables["lockers"])
            - summary["served"]
        ),
        "detail_locker_reservations": max(
            (
                abs(
                    row["pickup_reservation_mean"]
                    + row["delivery_reservation_mean"]
                    - row["mean_occupancy"]
                )
                for row in tables["lockers"]
            ),
            default=0,
        ),
        "detail_single_tasks": abs(
            sum(row["actual_tasks_per_hour"] for row in tables["task_singles"])
            - summary["single_tasks"]
        ),
        "detail_double_tasks": abs(
            sum(row["actual_tasks_per_hour"] for row in tables["task_pairs"])
            - summary["double_tasks"]
        ),
        "detail_cost_relative": abs(
            sum(row["cny_per_hour"] for row in tables["cost_detail"])
            - summary["objective"]
        )
        / (1 + abs(summary["objective"])),
        "detail_node_flow_balance": max(
            (abs(row["flight_balance_residual"]) for row in tables["nodes"]), default=0
        ),
        "detail_direct_split": abs(
            summary["unassigned_direct_orders_per_hour"]
            + summary["unavailability_direct_orders_per_hour"]
            - summary["direct"]
        ),
    }
    if summary["terminal_choice_available"]:
        checks["detail_terminal_choice"] = abs(
            summary["self_pickup_orders_per_hour"]
            + summary["terminal_courier_orders_per_hour"]
            - summary["served"]
        )
    residuals.update({name: float(value) for name, value in checks.items()})
    summary["checks"]["passed"] = (
        summary["checks"]["passed"]
        and max(checks.values()) <= summary["checks"]["tolerance"]
    )


def metric_dictionary(tables):
    """A machine-readable dictionary covering every exported table column."""
    meanings = {
        "availability": "Base drone-availability probability rho, not utilization",
        "assigned_availability": "Availability rho of the assigned departure base; zero if unassigned",
        "idle": "Mean idle drone inventory I at this base",
        "idle_minimum": "Minimum idle inventory rho/(1-rho) required by the availability cone",
        "idle_slack": "Idle inventory minus rho/(1-rho)",
        "busy_drones": "Mean busy drones: flow rate times mean duration",
        "attributed_total_drones": "Idle inventory plus busy drones attributed to tasks departing this base; excludes global fleet slack",
        "attributed_busy_fraction": "Departure-attributed busy drones divided by departure-attributed busy plus idle drones",
        "mean_occupancy": "Reservation-inclusive mean offered locker load; not physical meal count",
        "self_pickup_probability": "Saved exogenous probability of customer self-pickup at this order's delivery locker",
        "self_pickup_orders_per_hour": "Expected drone-served orders ending in customer self-pickup; not measured choices",
        "terminal_courier_orders_per_hour": "Expected drone-served orders requiring the delivery-locker-to-customer courier leg",
        "unassigned_direct_orders_per_hour": "Demand sent directly by courier because the OD has no assigned base pair",
        "unavailability_direct_orders_per_hour": "Assigned demand sent directly by courier through departure-base unavailability 1-rho",
        "mean_threshold": "Maximum Poisson mean offered load allowed by installed compartments and overflow tolerance",
        "pickup_reservation_mean": "Mean number of pickup reservations: order rate times request-to-pickup mean",
        "delivery_reservation_mean": "Mean delivery reservations: order rate times request-to-deposit plus meal-removal mean",
        "potential_overflow_probability": "Poisson offered-load tail P(N > installed compartments)",
        "allowed_probability": "Allowed locker offered-load overflow probability epsilon",
        "reservation_load_fraction": "Reservation-inclusive mean load divided by allowed Poisson mean-load threshold",
        "compartments_load_fraction": "Reservation-inclusive mean load divided by installed physical compartments",
        "capacity_slack": "Allowed Poisson mean-load threshold minus reservation-inclusive mean load",
        "probability_slack": "Allowed overflow probability minus computed Poisson tail probability",
        "continuation_probability": "Probability of at least one pool arrival during delta: 1-exp(-lambda*delta), before compatibility filtering",
        "reconstructed_tasks_per_hour": "Total reconstructed first-task rate theta before availability thinning",
        "reconstructed_head_per_hour": "Reconstructed first-task rate theta for this OD before availability thinning",
        "executed_tasks_per_hour": "Actual complete service-task start rate (one or two orders each)",
        "offered_tasks_per_hour": "Reconstructed task-class rate before availability thinning",
        "target_tasks_per_hour": "Reconstructed task-class rate times departure-base availability rho",
        "actual_tasks_per_hour": "Sum of actual path flows for this singleton or ordered OD pair",
        "kappa": "Compatibility coefficient for ordered OD pair; global physical support does not guarantee support at every base pair",
        "offered_orders_per_hour": "Assigned original order demand before availability thinning",
        "expected_served_orders_per_hour": "Assigned original order demand multiplied by departure-base rho",
        "task_legs_per_hour": "Task-stage occurrences, including zero-length transfer legs",
        "traversals_per_hour": "Physical directed-arc traversals, counting multiplicity within a task",
        "busy_minutes_per_hour": "Flow-weighted drone busy duration; divide by 60 for mean busy drones",
        "flight_balance_residual": "Outgoing minus incoming physical drone traversals at this node",
        "flow_balance_residual": "Complete task departures minus complete task arrivals at this base",
        "receipt_margin_minutes": "Own-order mean delivery deadline minus mean request-to-final-receipt time",
        "mean_receipt_minutes": "Flow-weighted mean own-request-to-final-receipt time among drone-served orders only",
        "drone_pickup_wait_minutes": "Mean drone waiting duration at this order's pickup locker",
        "origin_rider_minutes": "Model mean exponential courier trip duration from order origin to pickup locker",
        "terminal_minutes": "Model mean duration from delivery-locker deposit to final customer receipt",
        "pickup_hours": "Mean duration from this order's own request to drone pickup; second order uses uniform backward window approximation",
        "pickup_minutes": "Mean duration from this order's own request to drone pickup",
        "receipt_hours": "Mean duration from this order's own request to final customer receipt",
        "deadline_hours": "This OD's mean request-to-final-receipt deadline",
        "mean_operating_minutes": "Complete drone task mean duration: all flight stages plus both applicable pickup waits",
        "endurance_margin_minutes": "Mean operating-time limit minus complete task mean duration",
        "unit_cost": "Per-task operating cost including flight, drone waiting and served-order first/terminal courier legs",
        "cny_per_total_order": "Cost component divided by all offered order demand",
        "cny_per_drone_order": "Non-direct-courier cost component divided by drone-served order rate; system-average accounting, not marginal cost",
        "objective_fraction": "Cost component divided by total hourly objective",
        "fleet_fraction": "Component mean drones divided by continuous total fleet",
        "x_km": "Local projected east-west coordinate; use saved data projection provenance",
        "y_km": "Local projected north-south coordinate; use saved data projection provenance",
        "outflow": "Complete service plus empty task departures per hour",
        "inflow": "Complete service plus empty task arrivals per hour",
        "opened": "Whether the site or directed flight link is activated in the solution",
        "allowed": "Whether the directed flight link exists in the candidate network",
        "assigned": "Whether this OD is assigned to a departure-return base pair; assignment does not force positive flow if rho is zero",
        "assignment": "Saved binary assignment value",
        "first_per_hour": "Orders served in first position, including singleton tasks",
        "second_per_hour": "Orders served in second position of a two-order task",
        "direct_per_hour": "Original demand served by direct ground courier",
        "drone_per_hour": "Drone-served orders, counting both positions separately",
        "demand_per_hour": "Original order arrival rate from the prepared data",
        "service_per_hour": "Physical arc traversals by complete service tasks",
        "empty_per_hour": "Physical arc traversals by empty base repositioning tasks",
        "total_per_hour": "Physical arc traversals by service plus empty tasks",
        "compartments": "Installed physical locker compartments",
        "position": "Order position within task: 1 for singleton/first, 2 for continuation",
        "path": "Path ID within service or empty path catalogue; use kind to distinguish catalogues",
        "node": "Zero-based site ID; for market_nodes ID is local to origin/destination kind",
        "base": "Zero-based candidate base index",
        "locker": "Zero-based candidate locker index",
        "od": "Zero-based prepared origin-destination demand index",
        "departure": "Zero-based departure base index",
        "return_base": "Zero-based return base index",
        "first_od": "Zero-based OD index served first",
        "second_od": "Zero-based OD index served second; blank for single-order/empty paths",
        "first_pickup": "Zero-based first pickup locker index",
        "second_pickup": "Zero-based second pickup locker index; blank when absent",
        "first_delivery": "Zero-based first delivery locker index",
        "second_delivery": "Zero-based second delivery locker index; blank when absent",
        "pickup": "Zero-based pickup locker index",
        "delivery": "Zero-based delivery locker index",
        "u": "Global source site index: bases first, then lockers",
        "v": "Global target site index: bases first, then lockers",
        "kind": "Row category: service/empty path, base/locker site, or origin/destination market node",
        "mode": "Complete path mode: single, double or empty",
        "stage": "Flight stage; continuation_transfer connects first delivery to second pickup; empty is base repositioning",
        "component": "Named mutually exclusive component within this accounting table",
        "drones": "Continuous mean number of drones in this fleet/resource component",
        "label": "Human-readable zero-based node label",
        "u_label": "Source node label",
        "v_label": "Target node label",
        "local_index": "Zero-based index within site type",
    }
    rows = []
    for table, records in tables.items():
        for column in dict.fromkeys(key for record in records for key in record):
            meaning = meanings.get(column)
            if meaning is None:
                meaning = column.replace("_", " ")
                if column.startswith("mean_"):
                    meaning += "; flow-weighted mean among relevant served orders/tasks"
                if column.endswith("_drones"):
                    meaning += "; rate times duration, attributed to task departure for base rows"
            if column in (
                "availability",
                "assigned_availability",
                "continuation_probability",
                "kappa",
                "assignment",
                "potential_overflow_probability",
                "allowed_probability",
                "probability_slack",
                "self_pickup_probability",
            ) or column.endswith("_fraction"):
                unit = "dimensionless"
            elif column.endswith("_minutes_per_hour"):
                unit = "drone-minutes/hour"
            elif column.endswith("_cost_per_hour") or column == "cny_per_hour":
                unit = "CNY/hour"
            elif column.startswith("cny_per_") or column == "unit_cost":
                unit = "CNY/order" if column != "unit_cost" else "CNY/task"
            elif column.endswith("_per_hour") or column in (
                "outflow",
                "inflow",
                "flow_balance_residual",
                "flight_balance_residual",
            ):
                unit = "events/hour (see definition)"
            elif column.endswith("_minutes"):
                unit = "minutes"
            elif column.endswith("_hours"):
                unit = "hours"
            elif column.endswith("_drones") or column in (
                "idle",
                "idle_minimum",
                "idle_slack",
                "drones",
            ):
                unit = "mean drones (continuous)"
            elif column in (
                "mean_occupancy",
                "mean_threshold",
                "pickup_reservation_mean",
                "delivery_reservation_mean",
                "capacity_slack",
            ):
                unit = "mean reservation slots"
            elif column in ("compartments",):
                unit = "compartments"
            elif column.endswith("_km"):
                unit = "km"
            elif column.endswith("_count"):
                unit = "count"
            elif column in ("opened", "allowed", "assigned"):
                unit = "boolean"
            else:
                unit = "identifier/category"
            rows.append(dict(table=table, column=column, unit=unit, definition=meaning))
    return rows
