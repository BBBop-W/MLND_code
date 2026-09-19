"""Physical column pools, seed repricing, and feasible incumbent packaging."""

from collections import defaultdict
import numpy as np
from src.model.network import facility_costs, incidence, levels, network, route_key
from src.operations.validation import build_operations
from src.pricing.oracle import (
    _materialize,
    _validated,
    price_design,
    trace_empty,
    trace_mission,
)


def rebuild_route(context, record, service):
    """Recompute a stored physical route with the current timing and costs."""
    old = record.get("trace", record)
    get = lambda key: record.get(key, old.get(key))
    legs = old["legs"]
    swaps = []
    for leg in legs:
        if len(leg) not in (2, 3):
            raise ValueError("A path leg must be direct or use one swapping station")
        swaps.append(-1 if len(leg) == 2 else int(leg[1]) - context["L"] - context["H"])
        if swaps[-1] < -1 or swaps[-1] >= context["K"]:
            raise ValueError("A path uses an invalid swapping station")
    result = _materialize(
        context,
        int(get("od")) if service else None,
        int(get("l")),
        int(get("m")),
        int(get("i")) if service else None,
        int(get("j")) if service else None,
        swaps,
    )
    if result["legs"] != legs:
        raise ValueError("Rebuilt physical path differs from seed")
    if max(result["battery_interval_consumptions"]) > context["battery"] + 1e-8:
        raise ValueError("Seed path violates current battery parameters")
    if (
        service
        and result["deadline_quantile_time"]
        > context["deadline"][int(get("od"))] + 1e-8
    ):
        raise ValueError("Seed path violates current delivery parameters")
    return result


def record_flow(trace, flow):
    """Attach a nonnegative flow and grouping indices to a route trace."""
    return dict(
        trace=trace,
        flow=float(flow),
        time=trace["time"],
        cost=trace["cost"],
        **{key: trace[key] for key in ("od", "l", "m", "i", "j") if key in trace},
    )


def build_result(data, solution, missions, relocations, global_lower_bound):
    """Build and independently validate a feasible incumbent with its global bound."""
    if not np.isfinite(global_lower_bound):
        raise ValueError("A finite full-model lower bound is required")
    net = network(data)
    sizes, _, levelcost = levels(data)
    sol = {
        key: np.asarray(solution[key], float).copy()
        for key in (
            "facility",
            "capacity_level",
            "assignment",
            "airline",
            "rho",
            "idle",
        )
    }
    sol["fleet"] = float(solution["fleet"])
    L, H = data["n_l"], data["n_h"]
    sol["hub_count"] = sol["capacity_level"] @ sizes
    sol["empty"] = np.zeros((L, L))
    hubload = np.zeros(H)
    linkload = np.zeros(net["count"])
    for service, records in ((True, missions), (False, relocations)):
        for row in records:
            tr, flow = row["trace"], row["flow"]
            for edge, count in incidence(net, tr):
                linkload[edge] += count * flow
            if service:
                hubload[tr["i"]] += flow
                hubload[tr["j"]] += flow
            else:
                sol["empty"][tr["l"], tr["m"]] += flow
    opening = float(
        facility_costs(data) @ sol["facility"]
        + np.sum(levelcost * sol["capacity_level"])
        + net["fees"] @ sol["airline"]
    )
    c0 = float(np.asarray(data["demand"]) @ np.asarray(data["ground"]))
    operating = c0 + float(data["fleet_cost"]) * sol["fleet"]
    operating += sum(
        (row["trace"]["operating_cost"] - data["ground"][row["od"]]) * row["flow"]
        for row in missions
    )
    operating += sum(
        row["trace"]["operating_cost"] * row["flow"] for row in relocations
    )
    objective = opening + operating
    if global_lower_bound > objective + 1e-5:
        raise ValueError(
            "Supplied full-model lower bound exceeds the feasible objective"
        )
    result = dict(
        name=data.get("name", "path_instance"),
        model_version="path_cqn_v1",
        method="path_incumbent",
        objective=objective,
        bound=float(global_lower_bound),
        gap=max(0.0, objective - global_lower_bound) / max(1.0, abs(objective)),
        fleet=sol["fleet"],
        served=float(sum(row["flow"] for row in missions)),
        demand=float(np.sum(data["demand"])),
        n_od=data["n_od"],
        rho=sol["rho"].tolist(),
        cabinet_total=float(sol["hub_count"].sum()),
        cabinet_counts=sol["hub_count"].tolist(),
        hub_load=hubload.tolist(),
        airline_load=linkload.tolist(),
        opened=dict(
            L=int(round(sol["facility"][:L].sum())),
            H=int(round(sol["facility"][L : L + H].sum())),
            E=int(round(sol["facility"][L + H :].sum())),
        ),
        open_airlines=int(round(sol["airline"][net["physical"]].sum())),
        open_connectors=int(round(sol["airline"][~net["physical"]].sum())),
        airline_fixed_cost=float(net["fees"] @ sol["airline"]),
        all_ground_cost=c0,
        provenance=data.get("provenance", {}),
        _solution=sol,
        _missions=missions,
        _relocations=relocations,
        errors=[],
    )
    checks = build_operations(data, result)["summary"]["checks"]
    result["validation"] = checks
    result["primal_residual"] = max(
        (
            float(v)
            for k, v in checks.items()
            if k != "passed" and "cost" not in k and "objective" not in k
        ),
        default=0.0,
    )
    if not checks["passed"]:
        raise ValueError(
            f"Independent full-model feasibility validation failed: {checks}"
        )
    return result


def prepare_pool(
    data, seed_result, *, add_oracle_pairs=2, global_lower_bound=0.0, columns=None
):
    """Reprice all seed paths and add exact minimum-cost candidate routes."""
    L, H, D = (int(data[key]) for key in ("n_l", "n_h", "n_od"))
    context = _validated(data)
    service, empty, skeys, ekeys = [], [], {}, {}
    swarms, ewarms = defaultdict(float), defaultdict(float)

    def insert(record, is_service, flow=0.0):
        tr = rebuild_route(context, record, is_service)
        key = (tr.get("od"), tr["l"], tr["m"], route_key(tr))
        collection, keys, warms = (
            (service, skeys, swarms) if is_service else (empty, ekeys, ewarms)
        )
        if key not in keys:
            keys[key] = len(collection)
            collection.append(tr)
        warms[keys[key]] += float(flow)
        return tr

    seed_missions, seed_empty = [], []
    for flag, rows, target in (
        (
            True,
            seed_result.get("_missions", seed_result.get("missions", [])),
            seed_missions,
        ),
        (
            False,
            seed_result.get("_relocations", seed_result.get("relocations", [])),
            seed_empty,
        ),
    ):
        for row in rows:
            if float(row["flow"]) > 0:
                tr = insert(row, flag, row["flow"])
                target.append(record_flow(tr, row["flow"]))
    seed_sol = seed_result.get("_solution", seed_result)
    seed = build_result(data, seed_sol, seed_missions, seed_empty, global_lower_bound)
    seed["heuristic_seed_objective"] = seed["objective"]
    for flag, rows in (
        (True, (columns or {}).get("service", [])),
        (False, (columns or {}).get("empty", [])),
    ):
        for row in rows:
            insert(row, flag)
    oracle_wall = 0.0
    if add_oracle_pairs:
        priced = price_design(data)
        oracle_wall = priced["wall"]
        paircost = priced["cost"].reshape(D, L * L)
        for od in range(D):
            for pair in np.argsort(paircost[od], kind="stable")[
                : int(add_oracle_pairs)
            ]:
                if np.isfinite(paircost[od, pair]):
                    insert(
                        trace_mission(priced, od, int(pair) // L, int(pair) % L), True
                    )
        for l in range(L):
            for m in range(L):
                if l != m and np.isfinite(priced["empty_cost"][l, m]):
                    insert(trace_empty(priced, l, m), False)
    return dict(
        seed=seed,
        seed_solution=seed_sol,
        service=service,
        empty=empty,
        service_starts=swarms,
        empty_starts=ewarms,
        oracle_wall=oracle_wall,
    )


# Compatibility names for historical test fixtures; production uses public names.
_pack = build_result
_record = record_flow
_rebuild = rebuild_route
