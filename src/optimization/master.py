"""Integer strategic master with the initial operating-cost lower bound."""

import gurobipy as gp
from gurobipy import GRB
import numpy as np
from src.model.network import network, levels, facility_costs


def build_master(
    data, allopen, threads=1, mip_gap=1e-4, log=None, mip_gap_abs=1e-6, seed=43
):
    L, H, K, D = (data[k] for k in ("n_l", "n_h", "n_k", "n_od"))
    net = network(data)
    sizes, capacities, level_cost = levels(data)
    model = gp.Model("integer_master")
    model.Params.OutputFlag = 0
    model.Params.Threads = threads
    model.Params.Seed = seed
    if mip_gap is not None:
        model.Params.MIPGap = mip_gap
    if mip_gap_abs is not None:
        model.Params.MIPGapAbs = mip_gap_abs
    if log:
        model.Params.LogToConsole = 0
        model.Params.LogFile = str(log)
        model.Params.OutputFlag = 1
    x = model.addMVar(net["N"], vtype=GRB.BINARY, name="x")
    z = model.addMVar(capacities.shape, vtype=GRB.BINARY, name="z")
    alpha = model.addMVar((D, L, L), vtype=GRB.BINARY, name="alpha")
    y = model.addMVar(net["count"], vtype=GRB.BINARY, name="y")
    zeta = model.addVar(lb=0, name="zeta")
    model.addConstr(z.sum(axis=1) == x[L : L + H], name="one_locker_level")
    model.addConstr(alpha.sum(axis=(1, 2)) <= 1, name="one_base_pair")
    model.addConstr(alpha.sum(axis=2) <= x[:L], name="source_open")
    model.addConstr(alpha.sum(axis=1) <= x[:L], name="return_open")
    model.addConstr(y <= x[net["uv"][:, 0]], name="link_origin")
    model.addConstr(y <= x[net["uv"][:, 1]], name="link_destination")
    # The initial bound includes the mission fleet-time cost.
    saving = data["demand"][:, None, None] * np.maximum(0.0, -allopen["cost"])
    c0 = float(data["demand"] @ data["ground"])
    model.addConstr(zeta >= c0 - (saving * alpha).sum(), name="initial_operating_bound")
    model.setObjective(
        facility_costs(data) @ x + (level_cost * z).sum() + net["fees"] @ y + zeta
    )
    model.update()
    return model, x, z, alpha, y, zeta
