"""Continuous SOCP recourse solved by dual column generation.

Every candidate facility and link is retained in universal pricing.
Primal recovery and affine cut construction are separate components.
"""

from __future__ import annotations
import time
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from src.model.network import price_matrix, incidence, route_key
from src.model.design import fixed_design
from src.solvers.settings import settings
from src.solvers.benders.primal import recover_primal
from src.solvers.benders.cuts import build_cut, _idle_conjugate as _idle_conjugate
from src.solvers.pricing.backend import price_design, PricingStopped
from src.solvers.pricing.routes import trace_mission, trace_empty


class RecourseStopped(RuntimeError):
    """A stop request or exhausted pricing budget prevented certification."""


def solve(
    data,
    facility,
    capacity_level,
    assignment,
    airlines,
    time_limit=60.0,
    threads=1,
    certificate_tol=2e-6,
    max_iterations=300,
    initial_pool=None,
    initial_oracle=None,
    stop_requested=None,
    progress=None,
    seed=43,
    pricing="label",
    warm_start=True,
):
    if pricing not in {"label", "solver"}:
        raise ValueError("pricing must be 'label' or 'solver'")
    started = time.perf_counter()
    deadline = float("inf") if time_limit is None else started + time_limit
    last_progress = started

    def price(**kwargs):
        try:
            return price_design(
                data,
                backend=pricing,
                threads=threads,
                seed=seed,
                time_limit=None
                if time_limit is None
                else deadline - time.perf_counter(),
                stop_requested=stop_requested,
                **kwargs,
            )
        except PricingStopped as error:
            raise RecourseStopped(str(error)) from error

    def check_stop():
        if stop_requested is not None and stop_requested():
            raise RecourseStopped("user_stop")

    def callback(model, where):
        nonlocal last_progress
        if stop_requested is not None and stop_requested():
            model.terminate()
        if progress is not None and time.perf_counter() - last_progress >= 30.0:
            last_progress = time.perf_counter()
            progress(
                dict(
                    stage="recourse_optimize",
                    callback_where=int(where),
                    cg_iteration=iterations,
                    elapsed=last_progress - started,
                )
            )

    def optimize(model):
        check_stop()
        if not warm_start:
            model.reset(1)
        if stop_requested is None and progress is None:
            model.optimize()
        else:
            model.optimize(callback)
        check_stop()

    check_stop()
    design = fixed_design(data, facility, capacity_level, assignment, airlines)
    L, H, K, D = (data[k] for k in ("n_l", "n_h", "n_k", "n_od"))
    net = design.net
    A = net["count"]
    selected = design.selected
    T = len(selected)
    os, ls, ms = design.origins, design.sources, design.returns
    rate, caps = design.rates, design.base_caps
    capacity_rhs, y = design.capacity_rhs, design.airlines
    c0 = design.baseline_cost
    # Seed dual column constraints on selected groups. This is CG initialization,
    # not the optional all-group lower bound used to strengthen the master.
    initial = price(groups=selected) if initial_oracle is None else initial_oracle
    if initial_oracle is not None and (
        not initial.get("complete", False) or not initial.get("full_graph", False)
    ):
        raise ValueError("Complete universal pricing required")
    pool = []
    epool = []
    keys = set()
    ekeys = set()
    pricing_calls = int(initial_oracle is None)
    pricing_seconds = initial.get("wall", 0.0) if initial_oracle is None else 0.0
    dual = gp.Model("dual_column_generation")
    settings(dual, time_limit, threads, seed)
    a = dual.addMVar(T, lb=-GRB.INFINITY, name="group_price")
    u = dual.addMVar(H, lb=0, name="locker_price")
    eta = dual.addMVar(A, lb=0, name="link_price")
    pi = dual.addMVar(L, lb=-GRB.INFINITY, name="balance_price")
    sigma = dual.addMVar(L, lb=0, name="rho_cap_price")
    cones = []
    for l in range(L):
        t = dual.addVar(lb=0)
        s = dual.addVar(lb=-GRB.INFINITY)
        z = dual.addVar(lb=-GRB.INFINITY)
        dual.addQConstr(s * s + z * z <= t * t)
        dual.addConstr(t + z <= data["fleet_cost"])
        groups = np.flatnonzero(ls == l)
        expr = gp.quicksum(
            float(rate[g]) * (a[g].item() + pi[l].item() - pi[int(ms[g])].item())
            for g in groups
        )
        dual.addConstr(expr + sigma[l] + t - z >= 0)
        cones.append((t, s, z))
    dual.setObjective(
        c0
        - capacity_rhs @ u
        - (net["upper"] * y) @ eta
        - caps @ sigma
        - gp.quicksum(2 * (t + s) for t, s, z in cones),
        GRB.MAXIMIZE,
    )

    def insert_service(g, tr):
        o, l, m = map(int, selected[g])
        key = (g, route_key(tr))
        if key in keys:
            return 0
        inc = incidence(net, tr)
        pool.append(
            dict(
                group=int(g),
                od=o,
                l=l,
                m=m,
                i=tr["i"],
                j=tr["j"],
                cost=float(tr["cost"]),
                time=float(tr["time"]),
                trace=tr,
                inc=inc,
            )
        )
        charge = gp.quicksum(k * eta[e].item() for e, k in inc)
        dual.addConstr(a[g] - u[tr["i"]] - u[tr["j"]] - charge <= tr["cost"])
        keys.add(key)
        return 1

    def insert_empty(l, m, tr):
        key = (l, m, route_key(tr))
        if key in ekeys:
            return 0
        inc = incidence(net, tr)
        epool.append(
            dict(
                l=int(l),
                m=int(m),
                cost=float(tr["cost"]),
                time=float(tr["time"]),
                trace=tr,
                inc=inc,
            )
        )
        charge = gp.quicksum(k * eta[e].item() for e, k in inc)
        dual.addConstr(tr["cost"] + charge + pi[l] - pi[m] >= 0)
        ekeys.add(key)
        return 1

    def add_oracle(oracle, groups, empty_pairs):
        added = 0
        for g in groups:
            o, l, m = map(int, selected[g])
            if np.isfinite(oracle["cost"][o, l, m]):
                added += insert_service(g, trace_mission(oracle, o, l, m))
        for l, m in empty_pairs:
            if l != m and np.isfinite(oracle["empty_cost"][l, m]):
                added += insert_empty(l, m, trace_empty(oracle, l, m))
        return added

    add_oracle(
        initial, range(T), [(l, m) for l in range(L) for m in range(L) if l != m]
    )
    if initial_pool:
        group_lookup = {tuple(map(int, row)): g for g, row in enumerate(selected)}
        for p in initial_pool.get("missions", []):
            g = group_lookup.get((p["od"], p["l"], p["m"]))
            if g is not None:
                insert_service(g, p["trace"])
        for p in initial_pool.get("relocations", []):
            insert_empty(p["l"], p["m"], p["trace"])
    last = None
    iterations = 0
    fully_priced = False
    dual_status = None
    reserve = 0.0 if time_limit is None else max(3.0, min(30.0, time_limit * 0.18))
    try:
        while time.perf_counter() < deadline - reserve and (
            max_iterations is None or iterations < max_iterations
        ):
            check_stop()
            if time_limit is not None:
                dual.Params.TimeLimit = max(
                    0.01, deadline - reserve - time.perf_counter()
                )
            optimize(dual)
            iterations += 1
            dual_status = int(dual.Status)
            if not dual.SolCount:
                break
            last = dict(
                a=a.X.copy(),
                u=np.maximum(0, u.X),
                eta=np.maximum(0, eta.X),
                pi=pi.X.copy(),
                sigma=np.maximum(0, sigma.X),
            )
            priced = price(
                hub_prices=last["u"],
                arc_prices=price_matrix(net, last["eta"]),
                groups=selected,
            )
            check_stop()
            pricing_calls += 1
            pricing_seconds += priced.get("wall", 0.0)
            last["priced"] = priced
            vals = priced["pricing_cost"][os, ls, ms] if T else np.empty(0)
            neg = np.flatnonzero(last["a"] - vals > 1e-7)
            empty_neg = [
                (l, m)
                for l in range(L)
                for m in range(L)
                if l != m
                and priced["empty_pricing_cost"][l, m] + last["pi"][l] - last["pi"][m]
                < -1e-7
            ]
            fully_priced = not len(neg) and not empty_neg
            if fully_priced:
                break
            if not add_oracle(priced, neg, empty_neg):
                break
    finally:
        dual.dispose()
    if last is None:
        last = dict(
            a=np.zeros(T),
            u=np.zeros(H),
            eta=np.zeros(A),
            pi=np.zeros(L),
            sigma=np.zeros(L),
            priced=initial,
        )
    check_stop()
    # Complete the dual over EVERY OD/base pair with these same dual prices.
    global_priced = price(
        hub_prices=last["u"], arc_prices=price_matrix(net, last["eta"])
    )
    check_stop()
    pricing_calls += 1
    pricing_seconds += global_priced.get("wall", 0.0)
    if not global_priced.get("complete", False) or not global_priced.get(
        "full_graph", False
    ):
        raise RuntimeError("Incomplete universal pricing cannot certify a Benders cut")
    last["priced"] = global_priced
    recovered = recover_primal(
        data,
        design,
        pool,
        epool,
        time_limit=(
            None if time_limit is None else max(0.05, deadline - time.perf_counter())
        ),
        threads=threads,
        seed=seed,
        optimize=optimize,
    )
    cut = build_cut(data, design, last, recovered["objective"])
    gap = cut["duality_gap"]
    return dict(
        **recovered,
        **cut,
        cg_complete=bool(
            recovered["primal_residual"] < 1e-5
            and abs(gap) <= certificate_tol * (1 + abs(recovered["objective"]))
        ),
        fully_priced=fully_priced,
        dual_status=dual_status,
        pool_size=len(pool) + len(epool),
        service_pool_size=len(pool),
        empty_pool_size=len(epool),
        cg_iterations=iterations,
        pricing_calls=pricing_calls,
        pricing_wall=pricing_seconds,
        wall=time.perf_counter() - started,
        pool=dict(missions=pool, relocations=epool),
        model_version="path_cqn_v1",
        pricing_backend=pricing,
        certificate="complete candidate-graph pricing; natural availability domain; capacity-level affine dual",
    )
