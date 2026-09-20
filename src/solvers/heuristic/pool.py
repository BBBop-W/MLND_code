"""Physical column pools, seed repricing, and feasible incumbent packaging."""

from collections import defaultdict
import numpy as np
from src.model.network import route_key
from src.solvers.result import build_result
from src.solvers.pricing.label import (
    _materialize,
    _validated,
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


def prepare_pool(
    data,
    seed_result,
    *,
    add_oracle_pairs=2,
    global_lower_bound=0.0,
    columns=None,
    pricing="label",
    time_limit=None,
    threads=1,
    seed=43,
    stop_requested=None,
):
    """Reprice all seed paths and add exact minimum-cost candidate routes."""
    solver_seed = seed
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
        from src.solvers.pricing.backend import price_design as price

        priced = price(
            data,
            backend=pricing,
            time_limit=time_limit,
            threads=threads,
            seed=solver_seed,
            stop_requested=stop_requested,
        )
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
