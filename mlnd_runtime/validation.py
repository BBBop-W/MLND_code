"""Independent current-model physical, flow, queue and cost residual checks.

This validates primal feasibility and input-derived task rates. It does not
reconstruct a global optimization lower-bound certificate.
"""
from collections import Counter, defaultdict
import math
import numpy as np
from .demand import generalized_demand, physical_task_statistics
from .input import to_planning_data
from .capacity import occupancy_limits

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
    if times[0] + wait + times[1] + data.home_receipt_mean[r, j] > data.deadline[r] + 1e-12:
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
        received = deposited + data.home_receipt_mean[r, j]
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


def _incomplete_preprocessing_courier_fallback(data, result, sol):
    """Recognize only the exact zero-drone checkpoint saved before task rates.

    Labels alone never waive provenance. Every decision and path-record flow
    must be exactly zero, and the explicitly saved courier vector must equal
    raw demand. Ordinary primal residual checks and full-domain reconstruction
    still run after this gate.
    """
    if not (
        result.get("preprocessing_complete") is False
        and result.get("task_rates_available") is False
        and result.get("incumbent_kind") == "all_courier_baseline"
        and result.get("pricing_complete") is False
        and result.get("domain_complete") is False
    ):
        return False
    if _scalar(result.get("bound"), "fallback bound") != 0:
        raise ValueError("An incomplete-preprocessing fallback cannot claim a nonzero bound")
    zero_fields = (
        "facility", "capacity_level", "assignment", "airline", "rho", "idle",
        "fleet", "head", "eta", "flow", "empty_flow",
    )
    for name in zero_fields + (("v",) if "v" in sol else ()):
        value = np.asarray(sol[name], dtype=float)
        if not np.isfinite(value).all() or np.any(value != 0):
            raise ValueError(f"Incomplete-preprocessing courier fallback has nonzero {name}")
    courier = np.asarray(sol.get("courier"), dtype=float)
    if courier.shape != data.demand.shape or not np.array_equal(courier, data.demand):
        raise ValueError("Incomplete-preprocessing courier fallback must serve all raw demand by courier")
    for records in (result.get("_missions", []), result.get("_relocations", [])):
        for record in records:
            if _scalar(record["flow"], "fallback path flow") != 0:
                raise ValueError("Incomplete-preprocessing courier fallback has nonzero recorded path flow")
    return True


def validate_solution(inputs, result, *, tolerance=5e-5):
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
    if (result.get("demand_formulation") != "preprocessed"
            or result.get("assignment_scope") != "task" or data.flow_policy != "aggregate"):
        raise ValueError("Validation requires the current preprocessed aggregate task-assignment model")
    saved_scope = "task"
    first_leg_consistency = result.get("first_leg_consistency", False)
    if not isinstance(first_leg_consistency, bool):
        raise ValueError("first_leg_consistency must be a boolean")
    if first_leg_consistency:
        raise ValueError("First-leg consistency is incompatible with independent tasks")
    bases, h, n, r = data.n_l, data.n_h, data.n_nodes, data.n_od
    shapes = dict(
        facility=(n,),
        capacity_level=data.slot_counts.shape,
        assignment=((r if data.max_orders == 1 else r+r*r), bases),
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
    check("single_assignment", a.sum(axis=tuple(range(1, a.ndim))) - 1)
    check("assignment_departure", a - x[:bases][None, :])
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
    served = np.zeros(r)
    service_time = empty_time = service_cost = empty_cost = single_tasks = (
        double_tasks
    ) = empty_tasks = 0.0
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
    early_courier_fallback = _incomplete_preprocessing_courier_fallback(data, result, sol)
    metadata = result.get("preprocessed_demand")
    if metadata is None and early_courier_fallback:
        metadata = {}
    if not isinstance(metadata, dict):
        raise ValueError("Preprocessed result is missing fixed-demand provenance")
    single_cost, double_cost, single_support, double_support, path_count = physical_task_statistics(data)
    preprocessing_domain = dict(
        complete=True, source="input_physical_domain",
        service_paths_enumerated=path_count, output_witnesses_trusted_for_completeness=False,
        incomplete_preprocessing_courier_fallback_verified=early_courier_fallback,
    )
    reference = generalized_demand(
        data.demand,
        data.effective_delta,
        data.kappa,
        single_cost,
        double_cost,
        single_support,
        double_support,
        candidate_order=metadata.get("candidate_order"),
    )
    for key, target in (
        ("single_rates", reference.single_rates),
        ("pair_rates", reference.pair_rates),
        ("head_rates", reference.head_rates),
        ("common_origin_support", reference.common_origin_support),
    ):
        if key not in metadata:
            if early_courier_fallback:
                continue
            raise ValueError(f"Preprocessed result is missing fixed-demand provenance: {key}")
        saved = np.asarray(metadata[key], float)
        if saved.shape != target.shape or not np.isfinite(saved).all():
            raise ValueError(f"Invalid preprocessed {key}")
        check("preprocessed_" + key, np.abs(saved - target))
    check(
        "preprocessed_order_conservation",
        np.abs(reference.reconstructed_orders - data.demand),
    )
    check("preprocessed_single_nonnegative", -reference.single_rates)
    check("preprocessed_pair_nonnegative", -reference.pair_rates)
    check("preprocessed_eta_zero", np.abs(eta))
    # The native design model fixes exactly zero-rate task assignments to
    # zero. Flow equalities alone cannot detect such unused assignments.
    # Use independently reconstructed rates, with no small-rate threshold.
    task_rates = (np.concatenate((reference.single_rates, reference.pair_rates.ravel()))
                  if data.max_orders == 2 else reference.single_rates)
    check("zero_rate_task_assignment", np.abs(a[task_rates == 0]))
    v = a * rho[None, :]
    if "v" in sol:
        saved_v = np.asarray(sol["v"], float)
        if saved_v.shape != v.shape or not np.isfinite(saved_v).all():
            raise ValueError("Invalid preprocessed availability product shape")
        check("preprocessed_availability_product", np.abs(saved_v - v))
    single_v = v[:r]
    pair_v = v[r:].reshape(r, r, bases) if data.max_orders == 2 else np.zeros((r, r, bases))
    expected_single = reference.single_rates[:, None] * single_v
    expected_pair = reference.pair_rates[:, :, None] * pair_v
    if result.get("task_indexing") != "singles_then_row_major_pairs":
        raise ValueError("Missing or inconsistent task indexing")
    expected_table = [dict(task=q, first_od=q if q < r else (q-r)//r,
                           second_od=-1 if q < r else (q-r)%r,
                           rate=float(reference.single_rates[q] if q < r else reference.pair_rates[(q-r)//r, (q-r)%r]))
                      for q in range(len(a))]
    table = result.get("task_table")
    if not isinstance(table, list) or len(table) != len(expected_table):
        raise ValueError("Missing or invalid task table")
    for saved, target in zip(table, expected_table):
        if any(saved.get(key) != target[key] for key in ("task", "first_od", "second_od")):
            raise ValueError("Task identity mismatch")
        if early_courier_fallback and "rate" in saved and saved["rate"] is None:
            continue
        check(f"task_rate_{target['task']}", abs(_scalar(saved.get("rate"), "task rate") - target["rate"]))
    singles = np.zeros((r, bases))
    doubles = np.zeros((r, r, bases))
    actual_heads = np.zeros((r, bases))
    task_flow_errors = []
    for segment in segments:
        sid, od, bl = int(segment["id"]), segment["first_od"], segment["b"][0]
        singles[od, bl] += branch[sid, None]
        doubles[od, :, bl] += [branch[sid, j] for j in range(r)]
        actual_heads[od, bl] += heads[sid]
        task_flow_errors.append(
            abs(
                heads[sid]
                - branch[sid, None]
                - sum(branch[sid, j] for j in range(r))
            )
        )
    check("first_leg_task_flow", task_flow_errors)
    check(
        "aggregate_single_branches",
        np.abs(singles - expected_single),
    )
    check(
        "aggregate_double_branches",
        np.abs(doubles - expected_pair),
    )
    check(
        "task_reconstruction",
        np.abs(actual_heads - expected_single - expected_pair.sum(axis=1)),
    )
    expected = (expected_single + expected_pair.sum(axis=1) + expected_pair.sum(axis=0)).sum(axis=1)
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
    summary = dict(
        model_version="consecutive",
        flow_policy=data.flow_policy,
        assignment_scope=saved_scope,
        demand_formulation="preprocessed",
        first_leg_consistency=first_leg_consistency,
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
    summary["preprocessing_domain_validation"] = preprocessing_domain
    return summary
