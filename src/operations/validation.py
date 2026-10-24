"""Reconstruct saved physical paths, balances and costs independently of solvers."""

from __future__ import annotations

from collections import Counter, defaultdict
import math
import numpy as np
from scipy.stats import poisson

from src.data.instances import to_planning_data
from src.model.locker_capacity import occupancy_limits


def _scalar(value, name):
    array = np.asarray(value)
    if array.ndim != 0:
        raise ValueError(f"{name} must be a scalar")
    result = float(array)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _index(value, name, upper=None):
    number = _scalar(value, name)
    if isinstance(value, (bool, np.bool_)) or number != int(number) or number < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    integer = int(number)
    if upper is not None and integer >= upper:
        raise ValueError(f"{name} is outside the candidate set")
    return integer


def _cached(path, expected):
    """Compare saved numeric fields to independently reconstructed coefficients."""
    for key, value in expected.items():
        actual = np.asarray(path[key], dtype=float)
        wanted = np.asarray(value, dtype=float)
        if (
            actual.shape != wanted.shape
            or not np.isfinite(actual).all()
            or not np.allclose(actual, wanted, rtol=1e-10, atol=1e-12)
        ):
            raise ValueError(f"Cached {key} disagrees with the physical path")


def _routes(data, path):
    legs = tuple(
        tuple(_index(v, "route node", data.n_nodes) for v in leg)
        for leg in path["legs"]
    )
    b = tuple(_index(v, "base", data.n_l) for v in path["b"])
    if len(b) != 2 or any(v < 0 or v >= data.n_l for v in b):
        raise ValueError("Invalid endpoint base pair")
    if not legs or any(not leg for leg in legs):
        raise ValueError("Every path must have nonempty flight legs")
    arcs = Counter()
    times = []
    for k, leg in enumerate(legs):
        if not leg or len(set(leg)) != len(leg):
            raise ValueError("Each flight leg must be an elementary route")
        if k and legs[k - 1][-1] != leg[0]:
            raise ValueError("Disconnected flight legs")
        if any(v < 0 or v >= data.n_nodes for v in leg):
            raise ValueError("Route node is outside the candidate network")
        if (
            data.candidate_routes is not None
            and len(leg) > 1
            and leg not in data.candidate_routes
        ):
            raise ValueError("Route is absent from the supplied finite route set")
        if (
            data.candidate_routes is None
            and data.route_mode == "direct"
            and len(leg) > 2
        ):
            raise ValueError("Multi-arc route in a direct-flight instance")
        duration = 0.0
        for u, v in zip(leg, leg[1:]):
            if not data.allowed_links[u, v]:
                raise ValueError("Path traverses a forbidden link")
            duration += data.flight_times[u, v]
            arcs[u, v] += 1
        times.append(duration)
    return b, legs, arcs, times


def _segment(data, segment):
    """Validate every retained first leg, including zero-flow ones."""
    sid = _index(segment["id"], "segment id")
    b, legs, _, times = _routes(data, segment)
    r = _index(segment["first_od"], "first OD", data.n_od)
    i = _index(segment["pickup"], "pickup locker", data.n_h)
    j = _index(segment["delivery"], "delivery locker", data.n_h)
    if (
        len(legs) != 2
        or i == j
        or legs[0][0] != b[0]
        or legs[0][-1] != data.n_l + i
        or legs[1][-1] != data.n_l + j
    ):
        raise ValueError("Invalid shared first-leg endpoints or stages")
    if times[1] + 1e-12 < data.effective_delta:
        raise ValueError("First loaded flight is shorter than the common window")
    mean = float(data.rider_mean[r, i])
    wait = mean * math.exp(-times[0] / mean) if mean else 0.0
    if times[0] + wait + times[1] + data.receipt_mean[r, j] > data.deadline[r] + 1e-12:
        raise ValueError("Shared first leg violates its first order's mean deadline")
    _cached(segment, dict(t0=times[0], t1=times[1]))
    return dict(segment, id=sid, b=b, first_od=r, pickup=i, delivery=j, legs=legs)


def _physical(data, path, *, empty=False):
    """Recompute directly from node routes, without model resource functions."""
    b, legs, arcs, times = _routes(data, path)
    if legs[0][0] != b[0] or legs[-1][-1] != b[1]:
        raise ValueError("Path does not connect its endpoint bases")
    saved_arcs = tuple(
        tuple(_index(v, "arc count or node") for v in entry)
        for entry in path["arc_counts"]
    )
    if saved_arcs != tuple((u, v, count) for (u, v), count in sorted(arcs.items())):
        raise ValueError("Cached arc_counts disagree with route traversals")
    flight = float(sum(times))
    if empty:
        if len(legs) != 1 or b[0] == b[1]:
            raise ValueError("Empty repositioning must connect distinct bases")
        if flight > data.operating_limit + 1e-8:
            raise ValueError("Empty path exceeds the operating-time limit")
        _cached(
            path, dict(flight_time=flight, time=flight, cost=data.flight_cost * flight)
        )
        return dict(
            b=b,
            arcs=arcs,
            flight=flight,
            time=flight,
            wait=0.0,
            cost=data.flight_cost * flight,
            slots={},
            orders=[],
        )
    pickups = tuple(_index(v, "pickup locker", data.n_h) for v in path["pickups"])
    deliveries = tuple(
        _index(v, "delivery locker", data.n_h) for v in path["deliveries"]
    )
    ods = (
        (_index(path["first_od"], "first OD", data.n_od),)
        if path.get("second_od") is None
        else (
            _index(path["first_od"], "first OD", data.n_od),
            _index(path["second_od"], "second OD", data.n_od),
        )
    )
    k = len(ods)
    if len(legs) != 2 * k + 1 or len(pickups) != k or len(deliveries) != k:
        raise ValueError("Incorrect service-stage count")
    if path.get("mode") != ("single" if k == 1 else "double") or k > data.max_orders:
        raise ValueError("Invalid service mode")
    if any(r < 0 or r >= data.n_od for r in ods):
        raise ValueError("Invalid OD index")
    for n, (i, j) in enumerate(zip(pickups, deliveries)):
        if not (0 <= i < data.n_h and 0 <= j < data.n_h) or i == j:
            raise ValueError("Invalid pickup/delivery locker pair")
        if (
            legs[2 * n][-1] != data.n_l + i
            or legs[2 * n + 1][0] != data.n_l + i
            or legs[2 * n + 1][-1] != data.n_l + j
        ):
            raise ValueError("Flight routes violate pickup/delivery precedence")
    delta = data.effective_delta
    if times[1] + 1e-9 < delta:
        raise ValueError("First loaded flight is shorter than the common window")
    if k == 2 and (delta <= 0 or data.kappa[ods[0], ods[1]] <= 0):
        raise ValueError("Ineligible consecutive OD pair")
    r1 = float(data.rider_mean[ods[0], pickups[0]])
    w1 = r1 * math.exp(-times[0] / r1) if r1 > 0 else 0.0
    waits = [w1]
    pickups_mean = [times[0] + w1]
    if k == 2:
        r2 = float(data.rider_mean[ods[1], pickups[1]])
        w2 = (
            r2 * math.exp(-times[2] / r2) * (-math.expm1(-delta / r2)) / (delta / r2)
            if r2 > 0
            else 0.0
        )
        waits.append(w2)
        pickups_mean.append(times[2] + delta / 2 + w2)
    slots = defaultdict(float)
    orders = []
    cost = data.flight_cost * flight + data.waiting_cost * sum(waits)
    for n, (r, i, j, q) in enumerate(zip(ods, pickups, deliveries, pickups_mean)):
        deposited = q + times[2 * n + 1]
        received = deposited + data.receipt_mean[r, j]
        if received > data.deadline[r] + 1e-8:
            raise ValueError(f"Order {n + 1} exceeds its own mean receipt deadline")
        slots[i] += q
        slots[j] += deposited + data.removal_mean[r, j]
        cost += data.first_cost[r, i] + data.terminal_cost[r, j]
        orders.append(
            dict(
                od=r,
                position=n + 1,
                pickup=int(i),
                delivery=int(j),
                pickup_hours=float(q),
                receipt_hours=float(received),
                deadline_hours=float(data.deadline[r]),
            )
        )
    duration = flight + sum(waits)
    if duration > data.operating_limit + 1e-8:
        raise ValueError("Service path exceeds the mean operating-time limit")
    counts = tuple(
        tuple(_index(v, "served count or OD") for v in row)
        for row in path["served_counts"]
    )
    if counts != tuple(sorted(Counter(ods).items())):
        raise ValueError("Cached served_counts disagree with the order occurrences")
    _cached(
        path,
        dict(
            flight_time=flight,
            time=duration,
            wait=sum(waits),
            cost=cost,
            pickup_times=pickups_mean,
            delivery_times=[order["receipt_hours"] for order in orders],
            slot_times=tuple(sorted(slots.items())),
        ),
    )
    return dict(
        b=b,
        arcs=arcs,
        flight=flight,
        time=float(duration),
        wait=float(sum(waits)),
        cost=float(cost),
        slots=dict(slots),
        orders=orders,
    )


def build_operations(inputs, result, *, tolerance=5e-5):
    tolerance = _scalar(tolerance, "tolerance")
    if tolerance < 0:
        raise ValueError("tolerance must be nonnegative")
    data = to_planning_data(inputs)
    if result.get("model_version") != "consecutive":
        raise ValueError("Results must belong to consecutive")
    if result.get("flow_policy", "prefix") != data.flow_policy:
        raise ValueError("Result flow_policy disagrees with its saved inputs")
    sol = result.get("_solution")
    if sol is None:
        raise ValueError("No saved incumbent to validate")
    bases, h, n, r = data.n_l, data.n_h, data.n_nodes, data.n_od
    shapes = dict(
        facility=(n,),
        capacity_level=data.slot_counts.shape,
        assignment=(r, bases, bases),
        airline=(n, n),
        rho=(bases,),
        idle=(bases,),
    )
    values = {}
    errors = {}
    physical_errors = []

    def check(name, violation):
        a = np.asarray(violation, float)
        if not np.isfinite(a).all():
            raise ValueError(f"Nonfinite validation residual: {name}")
        errors[name] = float(max(0.0, np.max(a, initial=0.0)))

    for key, shape in shapes.items():
        value = np.asarray(sol[key], float)
        if value.shape != shape or not np.isfinite(value).all():
            raise ValueError(f"Invalid saved {key} shape or nonfinite values")
        values[key] = value
        check(key + "_nonnegative", -value)
    x, z, a, y, rho, idle = (values[key] for key in shapes)
    fleet = _scalar(sol["fleet"], "fleet")
    check("fleet_nonnegative", -fleet)
    for key in ("facility", "capacity_level", "assignment", "airline"):
        check(key + "_binary", np.abs(values[key] - np.rint(values[key])))
        check(key + "_upper", values[key] - 1)
    check("capacity_install", np.abs(z.sum(axis=1) - x[bases:]))
    check("single_assignment", a.sum(axis=(1, 2)) - 1)
    check("assignment_departure", a - x[:bases][None, :, None])
    check("assignment_return", a - x[:bases][None, None, :])
    check("link_departure", y - x[:, None])
    check("link_return", y - x[None, :])
    check("forbidden_links", y * (~data.allowed_links))
    check("availability_open", rho - x[:bases])
    check("availability_upper", rho - 1)
    check("idle_inventory_cone", rho - idle * (1 - rho))
    head = np.asarray(sol.get("head", []), float)
    eta = np.asarray(sol.get("eta", []), float)
    segments = [_segment(data, segment) for segment in result.get("_segments", [])]
    if (
        head.shape != (len(segments),)
        or eta.shape != head.shape
        or not np.isfinite(head).all()
        or not np.isfinite(eta).all()
    ):
        raise ValueError("Saved first-leg arrays disagree with segments")
    check("head_nonnegative", -head)
    check("eta_nonnegative", -eta)
    heads = {int(s["id"]): float(head[k]) for k, s in enumerate(segments)}
    etas = {int(s["id"]): float(eta[k]) for k, s in enumerate(segments)}
    segment_map = {int(s["id"]): s for s in segments}
    if len(segment_map) != len(segments):
        raise ValueError("Duplicate first-leg identifiers")
    branch = defaultdict(float)
    outflow, inflow, loads, link_load = (
        np.zeros(bases),
        np.zeros(bases),
        np.zeros(h),
        np.zeros((n, n)),
    )
    served, first, second, receipt_weight = (np.zeros(r) for _ in range(4))
    service_time = empty_time = service_cost = empty_cost = single_tasks = (
        double_tasks
    ) = empty_tasks = 0.0
    path_rows, order_rows, arc_rows = [], [], []
    traversal_bounds = {kind: np.zeros((n, n)) for kind in ("service", "empty")}
    for kind, records in [
        ("service", result.get("_missions", [])),
        ("empty", result.get("_relocations", [])),
    ]:
        flow_array = np.asarray(
            sol.get("flow" if kind == "service" else "empty_flow", []), float
        )
        if flow_array.ndim != 1 or not np.isfinite(flow_array).all():
            raise ValueError("Path flows must be a finite one-dimensional array")
        check(kind + "_flow_nonnegative", -flow_array)
        check(kind + "_negative_flow_total", np.maximum(-flow_array, 0).sum())
        reconstructed_flow = np.zeros_like(flow_array)
        seen = set()
        for record in records:
            path = record["path"]
            flow = _scalar(record["flow"], "path flow")
            pid = _index(path["id"], "path id")
            if (
                pid in seen
                or pid < 0
                or pid >= len(flow_array)
                or not math.isfinite(flow)
            ):
                raise ValueError("Duplicate or invalid saved path identifier/flow")
            seen.add(pid)
            reconstructed_flow[pid] = flow
            try:
                q = _physical(data, path, empty=kind == "empty")
                if kind == "service":
                    sid = _index(path["segment_id"], "segment id")
                    segment = segment_map[sid]
                    if (
                        tuple(segment["b"]) != q["b"]
                        or segment["first_od"] != path["first_od"]
                        or segment["pickup"] != path["pickups"][0]
                        or segment["delivery"] != path["deliveries"][0]
                        or tuple(map(tuple, segment["legs"]))
                        != tuple(map(tuple, path["legs"][:2]))
                    ):
                        raise ValueError(
                            "Service path disagrees with its shared first leg"
                        )
            except (ValueError, KeyError, IndexError, TypeError) as exc:
                physical_errors.append(f"{kind} {pid}: {exc}")
                continue
            for (u, v), count in q["arcs"].items():
                traversal_bounds[kind][u, v] = max(traversal_bounds[kind][u, v], count)
            # Zero records still undergo structural and cached-coefficient checks.
            # Every positive flow, however small, contributes to all resources.
            if flow == 0:
                continue
            b = q["b"]
            outflow[b[0]] += flow
            inflow[b[1]] += flow
            for (u, v), count in q["arcs"].items():
                link_load[u, v] += flow * count
                arc_rows.append(
                    dict(
                        kind=kind, path=pid, u=u, v=v, traversals_per_hour=flow * count
                    )
                )
            if kind == "empty":
                empty_time += flow * q["time"]
                empty_cost += flow * q["cost"]
                empty_tasks += flow
            else:
                sid = int(path["segment_id"])
                s = path.get("second_od")
                branch[sid, s] += flow
                service_time += flow * q["time"]
                service_cost += flow * q["cost"]
                if s is None:
                    single_tasks += flow
                else:
                    double_tasks += flow
                for i, t in q["slots"].items():
                    loads[i] += flow * t
                for order in q["orders"]:
                    od = order["od"]
                    served[od] += flow
                    receipt_weight[od] += flow * order["receipt_hours"]
                    (first if order["position"] == 1 else second)[od] += flow
                    order_rows.append(dict(path=pid, flow_per_hour=flow, **order))
            path_rows.append(
                dict(
                    kind=kind,
                    path=pid,
                    departure=b[0],
                    return_base=b[1],
                    mode=path.get("mode", "empty"),
                    flow_per_hour=flow,
                    flight_minutes=60 * q["flight"],
                    mean_operating_minutes=60 * q["time"],
                    mean_wait_minutes=60 * q["wait"],
                    unit_cost=q["cost"],
                )
            )
        check(kind + "_flow_storage", np.abs(flow_array - reconstructed_flow))
        check(
            kind + "_flow_storage_total", np.abs(flow_array - reconstructed_flow).sum()
        )
        check(kind + "_record_flow_nonnegative", -reconstructed_flow)
    check("base_balance", np.abs(outflow - inflow))
    check("fleet_budget", idle.sum() + service_time + empty_time - fleet)
    thresholds = occupancy_limits(data.slot_counts, data.slot_epsilon[:, None])
    installed_limit = (thresholds * z).sum(axis=1)
    check("locker_load", loads - installed_limit)
    check("closed_link_use", link_load * (1 - y))
    activation = float(data.demand.sum()) * (
        traversal_bounds["service"] + (bases - 1) * traversal_bounds["empty"]
    )
    check("link_activation_bound", link_load - activation * y)
    checks_head = []
    checks_eta = []
    checks_single = []
    checks_double = []
    checks_starts = []
    checks_order = []
    checks_hbound = []
    for s in segments:
        sid = int(s["id"])
        b = tuple(s["b"])
        od = int(s["first_od"])
        hv = heads[sid]
        ev = etas[sid]
        lam = float(data.demand @ a[:, b[0], b[1]])
        qb = -math.expm1(-lam * data.effective_delta)
        checks_eta.extend(
            [
                abs(lam * ev - qb * hv),
                ev - data.effective_delta * hv,
                ev - min(1, data.effective_delta * data.demand[od]),
            ]
        )
        checks_hbound.append(hv - data.demand[od])
        targets = data.kappa[od] * data.demand * a[:, b[0], b[1]] * ev
        checks_starts.append(
            abs(branch[sid, None] + sum(branch[sid, j] for j in range(r)) - hv)
        )
        if data.flow_policy == "prefix":
            checks_single.append(abs(branch[sid, None] - (hv - targets.sum())))
            checks_double.extend(abs(branch[sid, j] - targets[j]) for j in range(r))
    pool_rows = []
    for bl in range(bases):
        for bm in range(bases):
            aa = a[:, bl, bm]
            hh = data.demand * aa
            lam = float(hh.sum())
            qb = -math.expm1(-lam * data.effective_delta)
            pi = hh / lam if lam else np.zeros(r)
            theta = (
                np.linalg.solve(np.eye(r) + qb * pi[:, None] * data.kappa.T, hh)
                if lam
                else hh
            )
            actual_head = np.zeros(r)
            continuation = np.zeros(r)
            actual_single = np.zeros(r)
            actual_double = np.zeros((r, r))
            for s in segments:
                if tuple(s["b"]) == (bl, bm):
                    sid = int(s["id"])
                    od = int(s["first_od"])
                    actual_head[od] += heads[sid]
                    continuation += hh * data.kappa[od] * etas[sid]
                    actual_single[od] += branch[sid, None]
                    actual_double[od] += [branch[sid, j] for j in range(r)]
            checks_head.extend(np.abs(actual_head + continuation - hh * rho[bl]))
            checks_order.extend(np.abs(actual_head - rho[bl] * theta))
            if data.flow_policy == "aggregate":
                # Reconstruct task rates directly from the fixed pool demand,
                # independently of saved head/eta lifts or solver row assembly.
                targets_double = rho[bl] * theta[:, None] * qb * data.kappa * pi
                targets_single = rho[bl] * theta - targets_double.sum(axis=1)
                checks_single.extend(np.abs(actual_single - targets_single))
                checks_double.extend(np.abs(actual_double - targets_double).ravel())
            pool_rows.append(
                dict(
                    departure=bl,
                    return_base=bm,
                    offered_orders_per_hour=lam,
                    continuation_probability=qb,
                    reconstructed_tasks_per_hour=float(theta.sum()),
                    executed_tasks_per_hour=float(actual_head.sum()),
                )
            )
    check("head_upper", checks_hbound)
    check("eta_identity_and_bounds", checks_eta)
    check("first_leg_task_flow", checks_starts)
    scope = "aggregate_" if data.flow_policy == "aggregate" else ""
    check(scope + "single_branches", checks_single)
    check(scope + "double_branches", checks_double)
    check("first_leg_order_balance", checks_head)
    check("task_reconstruction", checks_order)
    expected = data.demand * np.sum(a * rho[None, :, None], axis=(1, 2))
    check("served_order_identity", np.abs(served - expected))
    check("served_demand", served - data.demand)
    direct = data.demand - served
    if "courier" in sol:
        saved_courier = np.asarray(sol["courier"], dtype=float)
        if saved_courier.shape != (r,) or not np.isfinite(saved_courier).all():
            raise ValueError("Invalid courier flow array")
        check("courier_nonnegative", -saved_courier)
        check("courier_order_identity", np.abs(saved_courier - direct))
    costs = dict(
        bases=float(data.base_cost @ x[:bases]),
        lockers=float(np.sum(data.slot_cost * z)),
        links=float(np.sum(data.link_cost * y)),
        fleet=data.fleet_cost * fleet,
        service=service_cost,
        repositioning=empty_cost,
        direct_courier=float(data.ground_cost @ direct),
    )
    objective = sum(costs.values())
    saved_objective = _scalar(result["objective"], "objective")
    if result.get("gap") is not None:
        check("gap_nonnegative", -_scalar(result["gap"], "gap"))
    check(
        "objective_relative",
        abs(objective - saved_objective) / (1 + abs(objective)),
    )
    for key, expected_value in dict(
        served=served.sum(),
        demand=data.demand.sum(),
        fleet=fleet,
        opening_cost=sum(costs[k] for k in ("bases", "lockers", "links")),
        continuous_cost=sum(
            costs[k] for k in ("fleet", "service", "repositioning", "direct_courier")
        ),
    ).items():
        if result.get(key) is not None:
            check("reported_" + key, abs(_scalar(result[key], key) - expected_value))
    if result.get("bound") is not None:
        bound = _scalar(result["bound"], "bound")
        check("bound_above_incumbent", bound - objective)
        if result.get("gap") is not None:
            expected_gap = max(0.0, saved_objective - bound) / max(
                1.0, abs(saved_objective)
            )
            check("reported_gap", abs(_scalar(result["gap"], "gap") - expected_gap))
    passed = max(errors.values(), default=0) <= tolerance and not physical_errors
    capacities = (data.slot_counts * z).sum(axis=1)
    tables = dict(
        costs=[dict(component=k, cny_per_hour=float(v)) for k, v in costs.items()],
        fleet=[
            dict(component=k, drones=float(v))
            for k, v in dict(
                idle=idle.sum(),
                service=service_time,
                repositioning=empty_time,
                slack=fleet - idle.sum() - service_time - empty_time,
            ).items()
        ],
        bases=[
            dict(
                base=i,
                opened=bool(x[i] > 0.5),
                availability=float(rho[i]),
                idle=float(idle[i]),
                outflow=float(outflow[i]),
                inflow=float(inflow[i]),
            )
            for i in range(bases)
        ],
        lockers=[
            dict(
                locker=i,
                opened=bool(x[bases + i] > 0.5),
                compartments=float(capacities[i]),
                mean_occupancy=float(loads[i]),
                mean_threshold=float(installed_limit[i]),
                potential_overflow_probability=float(
                    poisson.sf(int(round(capacities[i])), loads[i])
                ),
                allowed_probability=float(data.slot_epsilon[i]),
            )
            for i in range(h)
        ],
        orders=[
            dict(
                od=i,
                demand_per_hour=float(data.demand[i]),
                drone_per_hour=float(served[i]),
                direct_per_hour=float(direct[i]),
                first_per_hour=float(first[i]),
                second_per_hour=float(second[i]),
                mean_receipt_minutes=float(60 * receipt_weight[i] / served[i])
                if served[i] > 0
                else None,
                deadline_minutes=float(60 * data.deadline[i]),
            )
            for i in range(r)
        ],
        pools=pool_rows,
        paths=path_rows,
        order_paths=order_rows,
        links=arc_rows,
    )
    summary = dict(
        model_version="consecutive",
        flow_policy=data.flow_policy,
        objective=objective,
        demand=float(data.demand.sum()),
        served=float(served.sum()),
        direct=float(direct.sum()),
        single_tasks=single_tasks,
        double_tasks=double_tasks,
        task_flights=single_tasks + double_tasks,
        continued_orders=double_tasks,
        empty_tasks=empty_tasks,
        fleet=fleet,
        slot_total=float(capacities.sum()),
        opened=dict(
            bases=int(np.sum(x[:bases] > 0.5)), lockers=int(np.sum(x[bases:] > 0.5))
        ),
        checks=dict(
            passed=passed,
            tolerance=tolerance,
            residuals=errors,
            physical_errors=physical_errors,
        ),
    )
    payload = dict(summary=summary, tables=tables)
    if passed:
        from .detail import enrich_operations

        enrich_operations(inputs, data, result, payload)
    return payload
