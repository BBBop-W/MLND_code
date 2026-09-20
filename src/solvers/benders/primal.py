"""Recover feasible operational flows from the generated physical columns."""

import numpy as np
from scipy import sparse
import gurobipy as gp
from gurobipy import GRB
from src.solvers.settings import settings


def recover_primal(
    data, design, pool, epool, *, time_limit, threads=1, seed=43, optimize=None
):
    """Solve the restricted SOCP and recompute fleet and feasibility residuals."""
    L, H = data["n_l"], data["n_h"]
    net = design.net
    A = net["count"]
    T = len(design.selected)
    y = design.airlines
    capacity_rhs = design.capacity_rhs
    caps = design.base_caps
    ls = design.sources
    rate = design.rates
    c0 = design.baseline_cost
    if optimize is None:
        optimize = lambda model: model.optimize()
    P = len(pool)
    R = len(epool)
    primal = gp.Model("restricted_primal")
    settings(primal, time_limit, threads, seed)
    # Remove structurally disabled columns by exact zero bounds. This avoids
    # tiny barrier flows on closed links being mistaken for used facilities.
    service_enabled = np.array(
        [
            all(y[e] > 0.5 for e, _ in p["inc"])
            and capacity_rhs[p["i"]] > 0
            and capacity_rhs[p["j"]] > 0
            for p in pool
        ],
        bool,
    )
    empty_enabled = np.array(
        [all(y[e] > 0.5 for e, _ in p["inc"]) for p in epool], bool
    )
    restricted_caps = caps.copy()
    group_enabled = np.bincount(
        [p["group"] for k, p in enumerate(pool) if service_enabled[k]], minlength=T
    )
    for l in range(L):
        groups = np.flatnonzero(ls == l)
        if not len(groups) or np.any(group_enabled[groups] == 0):
            restricted_caps[l] = 0.0
    rho = primal.addMVar(L, lb=0, ub=restricted_caps, name="rho")
    idle = primal.addMVar(L, lb=0, name="idle")
    theta = primal.addMVar(
        P, lb=0, ub=np.where(service_enabled, GRB.INFINITY, 0.0), name="psi"
    )
    phi = primal.addMVar(
        R, lb=0, ub=np.where(empty_enabled, GRB.INFINITY, 0.0), name="varphi"
    )
    Gamma = primal.addVar(lb=0, name="Gamma")
    match = sparse.coo_matrix(
        (np.ones(P), ([p["group"] for p in pool], np.arange(P))), shape=(T, P)
    ).tocsr()
    rhs = sparse.coo_matrix((rate, (np.arange(T), ls)), shape=(T, L)).tocsr()
    if T:
        primal.addConstr(match @ theta == rhs @ rho, name="OD_group_flow")
    cap = sparse.coo_matrix(
        (
            np.ones(2 * P),
            ([p["i"] for p in pool] + [p["j"] for p in pool], list(range(P)) * 2),
        ),
        shape=(H, P),
    ).tocsr()
    primal.addConstr(cap @ theta <= capacity_rhs, name="locker_capacity")

    def link_matrix(rows):
        rr = []
        cc = []
        vv = []
        for k, p in enumerate(rows):
            for e, c in p["inc"]:
                rr.append(e)
                cc.append(k)
                vv.append(c)
        return sparse.coo_matrix((vv, (rr, cc)), shape=(A, len(rows))).tocsr()

    smat = link_matrix(pool)
    emat = link_matrix(epool)
    primal.addConstr(
        smat @ theta + emat @ phi <= net["upper"] * y, name="link_activation"
    )
    out = sparse.coo_matrix(
        (np.ones(P), ([p["l"] for p in pool], np.arange(P))), shape=(L, P)
    ).tocsr()
    ins = sparse.coo_matrix(
        (np.ones(P), ([p["m"] for p in pool], np.arange(P))), shape=(L, P)
    ).tocsr()
    eo = sparse.coo_matrix(
        (np.ones(R), ([p["l"] for p in epool], np.arange(R))), shape=(L, R)
    ).tocsr()
    ei = sparse.coo_matrix(
        (np.ones(R), ([p["m"] for p in epool], np.arange(R))), shape=(L, R)
    ).tocsr()
    primal.addConstr((out - ins) @ theta + (eo - ei) @ phi == 0, name="base_balance")
    for l in range(L):
        up = primal.addVar(lb=0)
        lo = primal.addVar(lb=0)
        one = primal.addVar(lb=1, ub=1)
        primal.addConstr(up - idle[l] == 1)
        primal.addConstr(lo + rho[l] == 1)
        primal.addQConstr(one * one <= up * lo)
    costs = np.array([p["cost"] for p in pool])
    ec = np.array([p["cost"] for p in epool])
    stimes = np.array([p["time"] for p in pool])
    etimes = np.array([p["time"] for p in epool])
    primal.addConstr(
        idle.sum() + stimes @ theta + etimes @ phi <= Gamma, name="fleet_requirement"
    )
    primal.setObjective(
        c0
        + (costs - data["fleet_cost"] * stimes) @ theta
        + data["fleet_cost"] * Gamma
        + (ec - data["fleet_cost"] * etimes) @ phi
    )
    try:
        optimize(primal)
        status = int(primal.Status)
        if primal.SolCount and np.all(rho.X < 1.0):
            rv = np.maximum(0, rho.X)
            rv[restricted_caps == 0] = 0.0
            iv = rv / (1 - rv)
            tv = np.maximum(0, theta.X)
            ev = np.maximum(0, phi.X)
            tv[~service_enabled] = 0.0
            ev[~empty_enabled] = 0.0
        else:
            rv = np.zeros(L)
            iv = np.zeros(L)
            tv = np.zeros(P)
            ev = np.zeros(R)
    finally:
        primal.dispose()
    loads = np.asarray(cap @ tv).ravel()
    links = np.asarray(smat @ tv + emat @ ev).ravel()
    balance = (out - ins) @ tv + (eo - ei) @ ev
    residual = max(
        float(np.max(np.abs(balance), initial=0)),
        float(np.max(loads - capacity_rhs, initial=0)),
        float(np.max(links - net["upper"] * y, initial=0)),
        float(np.max(np.abs(match @ tv - rhs @ rv), initial=0)) if T else 0.0,
    )
    objective = float(c0 + costs @ tv + data["fleet_cost"] * iv.sum() + ec @ ev)
    missions = [dict(p, flow=float(tv[k])) for k, p in enumerate(pool) if tv[k] > 0.0]
    relocations = [
        dict(p, flow=float(ev[k])) for k, p in enumerate(epool) if ev[k] > 0.0
    ]
    earray = np.zeros((L, L))
    for p in relocations:
        earray[p["l"], p["m"]] += p["flow"]
    fleet = float(iv.sum() + sum(p["flow"] * p["time"] for p in missions + relocations))
    return dict(
        objective=objective,
        primal_residual=residual,
        status=status,
        rho=rv,
        idle=iv,
        empty=earray,
        fleet=fleet,
        served=float(tv.sum()),
        hub_load=loads,
        airline_load=links,
        missions=missions,
        relocations=relocations,
    )
