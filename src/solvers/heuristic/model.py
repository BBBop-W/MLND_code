"""Sparse integer design improvement model on a finite feasible path pool."""

from collections import defaultdict
from dataclasses import dataclass
import time
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from src.model.network import network, levels, facility_costs, incidence


@dataclass
class RestrictedModel:
    model: object
    x: object
    z: object
    y: object
    alpha: object
    rho: object
    idle: object
    psi: object
    varphi: object
    fleet: object
    statistics: dict


def build_restricted_master(
    data,
    pool,
    *,
    started=None,
    threads=4,
    seed=43,
    mip_gap=1e-4,
    mip_gap_abs=1e-6,
    log=None,
):
    """Keep every strategic variable binary while using a supplied column pool."""
    start = time.perf_counter() if started is None else started
    L, H, D = (int(data[key]) for key in ("n_l", "n_h", "n_od"))
    net = network(data)
    sizes, capacities, levelcost = levels(data)
    service, empty = pool["service"], pool["empty"]
    swarms, ewarms = pool["service_starts"], pool["empty_starts"]
    seed_sol = pool["seed_solution"]
    oracle_wall = pool["oracle_wall"]
    groups, byod, bysource, byreturn = (
        defaultdict(list),
        defaultdict(list),
        defaultdict(list),
        defaultdict(list),
    )
    locker = [[] for _ in range(H)]
    sout, sin = [[] for _ in range(L)], [[] for _ in range(L)]
    eout, ein = [[] for _ in range(L)], [[] for _ in range(L)]
    for index, tr in enumerate(service):
        groups[(tr["od"], tr["l"], tr["m"])].append(index)
        locker[tr["i"]].append(index)
        locker[tr["j"]].append(index)
        sout[tr["l"]].append(index)
        sin[tr["m"]].append(index)
    for group in groups:
        od, l, m = group
        byod[od].append(group)
        bysource[od, l].append(group)
        byreturn[od, m].append(group)
    for index, tr in enumerate(empty):
        eout[tr["l"]].append(index)
        ein[tr["m"]].append(index)

    model = gp.Model("restricted_column_design_improvement")
    model.Params.OutputFlag = 0
    model.Params.Threads = threads
    model.Params.Seed = seed
    model.Params.FeasibilityTol = 1e-8
    model.Params.OptimalityTol = 1e-8
    model.Params.BarQCPConvTol = 1e-9
    model.Params.NonConvex = 0
    model.Params.MIPFocus = 1
    if mip_gap is not None:
        model.Params.MIPGap = mip_gap
    if mip_gap_abs is not None:
        model.Params.MIPGapAbs = mip_gap_abs
    if log:
        model.Params.OutputFlag = 1
        model.Params.LogToConsole = 0
        model.Params.LogFile = str(log)
    x = model.addMVar(net["N"], vtype=GRB.BINARY, name="x")
    z = model.addMVar(capacities.shape, vtype=GRB.BINARY, name="z")
    y = model.addMVar(net["count"], vtype=GRB.BINARY, name="y")
    alpha = model.addVars(list(groups), vtype=GRB.BINARY, name="alpha")
    rho = model.addVars(L, lb=0.0, ub=1.0, name="rho")
    idle = model.addVars(L, lb=0.0, name="idle")
    psi = model.addVars(len(service), lb=0.0, name="psi")
    varphi = model.addVars(len(empty), lb=0.0, name="varphi")
    Gamma = model.addVar(lb=0.0, name="Gamma")
    model.addConstr(z.sum(axis=1) == x[L : L + H], name="capacity_choice")
    for od, row in byod.items():
        model.addConstr(
            gp.quicksum(alpha[group] for group in row) <= 1, name=f"one_assignment_{od}"
        )
    for (od, l), row in bysource.items():
        assign = gp.quicksum(alpha[group] for group in row)
        model.addConstr(assign <= x[l].item(), name=f"source_open_{od}_{l}")
        flow = gp.quicksum(psi[k] for group in row for k in groups[group])
        h = float(data["demand"][od])
        model.addConstr(flow <= h * rho[l], name=f"source_upper_{od}_{l}")
        model.addConstr(
            h * rho[l] - flow <= h * (1 - assign), name=f"source_lower_{od}_{l}"
        )
    for (od, m), row in byreturn.items():
        model.addConstr(
            gp.quicksum(alpha[group] for group in row) <= x[m].item(),
            name=f"return_open_{od}_{m}",
        )
    for group, row in groups.items():
        model.addConstr(
            gp.quicksum(psi[k] for k in row)
            <= float(data["demand"][group[0]]) * alpha[group]
        )
    model.addConstr(y <= x[net["uv"][:, 0]], name="link_origin")
    model.addConstr(y <= x[net["uv"][:, 1]], name="link_destination")
    for l in range(L):
        model.addConstr(rho[l] <= x[l].item(), name=f"available_{l}")
        model.addConstr(
            gp.quicksum(psi[k] for k in sout[l])
            + gp.quicksum(varphi[k] for k in eout[l])
            == gp.quicksum(psi[k] for k in sin[l])
            + gp.quicksum(varphi[k] for k in ein[l]),
            name=f"balance_{l}",
        )
        up = model.addVar(lb=0.0, name=f"cone_u_{l}")
        down = model.addVar(lb=0.0, name=f"cone_v_{l}")
        one = model.addVar(lb=1.0, ub=1.0, name=f"cone_one_{l}")
        model.addConstr(up == idle[l] + 1)
        model.addConstr(down == 1 - rho[l])
        model.addQConstr(one * one <= up * down, name=f"idle_cone_{l}")
        up.Start = float(seed_sol["idle"][l]) + 1
        down.Start = 1 - float(seed_sol["rho"][l])
        one.Start = 1.0
    for i in range(H):
        model.addConstr(
            gp.quicksum(psi[k] for k in locker[i])
            <= gp.quicksum(
                float(capacities[i, s]) * z[i, s].item() for s in range(len(sizes))
            ),
            name=f"locker_{i}",
        )
    links = [[] for _ in range(net["count"])]
    for rows, variables in ((service, psi), (empty, varphi)):
        for index, tr in enumerate(rows):
            for edge, count in incidence(net, tr):
                links[edge].append((variables[index], count))
    for edge, row in enumerate(links):
        model.addConstr(
            gp.quicksum(variable * count for variable, count in row)
            <= float(net["upper"][edge]) * y[edge].item(),
            name=f"link_{edge}",
        )
    model.addConstr(
        gp.quicksum(idle.values())
        + gp.quicksum(tr["time"] * psi[k] for k, tr in enumerate(service))
        + gp.quicksum(tr["time"] * varphi[k] for k, tr in enumerate(empty))
        <= Gamma,
        name="fleet",
    )
    objective = (
        float(np.asarray(data["demand"]) @ np.asarray(data["ground"]))
        + float(data["fleet_cost"]) * Gamma
    )
    objective += gp.quicksum(
        (tr["operating_cost"] - float(data["ground"][tr["od"]])) * psi[k]
        for k, tr in enumerate(service)
    )
    objective += gp.quicksum(
        tr["operating_cost"] * varphi[k] for k, tr in enumerate(empty)
    )
    objective += facility_costs(data) @ x + (levelcost * z).sum() + net["fees"] @ y
    model.setObjective(objective)
    x.Start = np.asarray(seed_sol["facility"])
    z.Start = np.asarray(seed_sol["capacity_level"])
    y.Start = np.asarray(seed_sol["airline"])
    for group, variable in alpha.items():
        variable.Start = float(seed_sol["assignment"][group])
    for l in range(L):
        rho[l].Start = float(seed_sol["rho"][l])
        idle[l].Start = float(seed_sol["idle"][l])
    for index, variable in psi.items():
        variable.Start = swarms[index]
    for index, variable in varphi.items():
        variable.Start = ewarms[index]
    Gamma.Start = float(seed_sol["fleet"])
    model.update()
    statistics = dict(
        service_columns=len(service),
        empty_columns=len(empty),
        assignment_binaries=len(groups),
        dense_assignment_size=D * L * L,
        strategic_binaries=int(model.NumBinVars),
        constraints=int(model.NumConstrs),
        construction_wall=time.perf_counter() - start,
        oracle_wall=oracle_wall,
        full_model_lower_bound_source="explicit_argument",
        restricted_bound_exported=False,
    )
    return RestrictedModel(
        model, x, z, y, alpha, rho, idle, psi, varphi, Gamma, statistics
    )
