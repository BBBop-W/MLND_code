"""Strict pure-MILP master for the isolated 2026-06-23 framework."""

from __future__ import annotations

import math
from typing import Dict

import gurobipy as gp
from gurobipy import GRB

from src import config


def build_pure_master(owner) -> None:
    """Build a strategic-only MILP master on a Benders oracle instance."""

    if owner.master is not None:
        return

    model = gp.Model("MLND_RegimeII_BDD_LR_Benders_20260623_Master")
    model.Params.OutputFlag = int(getattr(config, "BENDERS_MASTER_LOG", 0))
    model.Params.Threads = int(getattr(config, "BENDERS_MASTER_THREADS", 1))
    model.Params.Seed = int(getattr(config, "BENDERS_RANDOM_SEED", 0))

    capacities = range(len(config.MU_HUB))
    x = model.addVars(owner.N_x, vtype=GRB.BINARY, name="x")
    y = model.addVars(owner.AY, vtype=GRB.BINARY, name="y")
    z = model.addVars(owner.H, capacities, vtype=GRB.BINARY, name="z")
    gamma = model.addVar(lb=0.0, vtype=GRB.INTEGER, name="Gamma")
    eta = model.addVar(lb=-GRB.INFINITY, name="eta")
    use_proxy = bool(getattr(owner, "use_master_proxy", True))
    proxy_pair_limit = min(
        len(owner.A_HH),
        max(20, min(200, 2 * len(owner.H))),
    )
    proxy_top_pairs = tuple(
        sorted(
            owner.A_HH,
            key=lambda pair: float(owner.inst.h.get(pair, 0.0)),
            reverse=True,
        )[:proxy_pair_limit]
        if use_proxy
        else ()
    )
    proxy_top_set = set(proxy_top_pairs)
    top_pair_possible = model.addVars(
        proxy_top_pairs,
        vtype=GRB.BINARY,
        name="proxy_top_pair",
    )
    service_proxy = model.addVars(
        owner.H if use_proxy else (),
        lb=0.0,
        name="proxy_service_by_origin",
    )

    fixed_cost = gp.quicksum(
        float(owner.inst.f.get(l, config.COST_BASE)) * x[l]
        for l in owner.L
    )
    fixed_cost += gp.quicksum(
        float(getattr(config, "COST_HUB_FIXED", 0.0)) * x[i]
        for i in owner.H
    )
    fixed_cost += gp.quicksum(
        float(owner.inst.f.get(k, config.COST_SWAPPING)) * x[k]
        for k in owner.E
    )
    fixed_cost += gp.quicksum(
        float(config.COST_HUB[c]) * z[i, c]
        for i in owner.H
        for c in capacities
    )
    fixed_cost += gp.quicksum(
        owner.arc_cost[arc] * y[arc] for arc in owner.AY
    )
    fixed_cost += float(config.COST_DRONE) * gamma
    model.setObjective(eta - fixed_cost, GRB.MAXIMIZE)

    for hub in owner.H:
        assignments = [
            y[l, h] for l, h in owner.A_LH if h == hub and (l, h) in y
        ]
        model.addConstr(
            gp.quicksum(assignments) == x[hub],
            name=f"assign_{hub}",
        )
        model.addConstr(
            gp.quicksum(z[hub, c] for c in capacities) == x[hub],
            name=f"capacity_level_{hub}",
        )
        model.addConstr(
            gp.quicksum(
                float(config.MU_HUB[c]) * z[hub, c] for c in capacities
            )
            >= x[hub] / float(config.TAU_H),
            name=f"minimum_hub_slack_{hub}",
        )

    for base, hub in owner.A_LH:
        if (base, hub) in y:
            model.addConstr(
                y[base, hub] <= x[base],
                name=f"base_assignment_active_{base}_{hub}",
            )

    for u, v in owner.AY:
        if (u, v) in owner.A_LH:
            continue
        model.addConstr(
            y[u, v] <= owner._node_activation_expr({"x": x}, u),
            name=f"arc_tail_active_{u}_{v}",
        )
        model.addConstr(
            y[u, v] <= owner._node_activation_expr({"x": x}, v),
            name=f"arc_head_active_{u}_{v}",
        )

    revenue = float(getattr(config, "ORDER_REVENUE_COEF", 0.0))
    if use_proxy:
        for i, j in proxy_top_pairs:
            possible = top_pair_possible[i, j]
            model.addConstr(
                possible <= x[i],
                name=f"proxy_top_origin_{i}_{j}",
            )
            model.addConstr(
                possible <= x[j],
                name=f"proxy_top_destination_{i}_{j}",
            )
            if (i, j) in y:
                model.addConstr(
                    possible <= y[i, j],
                    name=f"proxy_top_arc_{i}_{j}",
                )
            else:
                model.addConstr(
                    possible == 0.0,
                    name=f"proxy_top_no_arc_{i}_{j}",
                )

        for origin in owner.H:
            origin_pairs = [
                pair for pair in owner.A_HH if pair[0] == origin
            ]
            top_pairs = [
                pair for pair in origin_pairs if pair in proxy_top_set
            ]
            residual_pairs = [
                pair for pair in origin_pairs if pair not in proxy_top_set
            ]
            top_volume = gp.quicksum(
                float(owner.inst.h.get(pair, 0.0))
                * top_pair_possible[pair]
                for pair in top_pairs
            )
            residual_total = sum(
                float(owner.inst.h.get(pair, 0.0))
                for pair in residual_pairs
            )
            model.addConstr(
                service_proxy[origin]
                <= top_volume + residual_total * x[origin],
                name=f"proxy_service_origin_{origin}",
            )
            model.addConstr(
                service_proxy[origin]
                <= top_volume
                + gp.quicksum(
                    float(owner.inst.h.get((i, j), 0.0)) * x[j]
                    for i, j in residual_pairs
                ),
                name=f"proxy_service_destination_{origin}",
            )
            model.addConstr(
                service_proxy[origin]
                <= top_volume
                + gp.quicksum(
                    float(owner.inst.h.get((i, j), 0.0)) * y[i, j]
                    for i, j in residual_pairs
                    if (i, j) in y
                ),
                name=f"proxy_service_arc_{origin}",
            )
        model.addConstr(
            eta
            <= revenue * gp.quicksum(service_proxy[hub] for hub in owner.H),
            name="eta_aggregate_service_proxy_upper_bound",
        )
    else:
        total_demand = sum(
            max(0.0, float(owner.inst.h.get(pair, 0.0)))
            for pair in owner.A_HH
        )
        model.addConstr(
            eta <= revenue * total_demand,
            name="eta_total_demand_upper_bound",
        )
    model.addConstr(
        eta
        <= revenue
        * gp.quicksum(
            gp.quicksum(
                float(config.MU_HUB[c]) * z[i, c] for c in capacities
            )
            - x[i] / float(config.TAU_H)
            for i in owner.H
        ),
        name="eta_hub_capacity_upper_bound",
    )

    drone_cost = float(config.COST_DRONE)
    if drone_cost > 1e-9:
        total_revenue = revenue * sum(
            max(0.0, float(owner.inst.h.get(pair, 0.0)))
            for pair in owner.A_HH
        )
        model.addConstr(
            gamma <= int(math.ceil(total_revenue / drone_cost)),
            name="gamma_dominance_upper_bound",
        )

    owner.master = model
    owner.master_vars = {
        "x": x,
        "y": y,
        "z": z,
        "Gamma": gamma,
        "eta": eta,
        "service_proxy": service_proxy,
        "top_pair_possible": top_pair_possible,
    }
    owner.proxy_top_pairs = proxy_top_pairs
    owner.master_proxy_metadata = {
        "master_proxy_enabled": use_proxy,
        "master_proxy_service_variables": len(owner.H) if use_proxy else 0,
        "master_proxy_top_pair_variables": len(proxy_top_pairs),
        "master_proxy_total_variables": (
            len(owner.H) + len(proxy_top_pairs) if use_proxy else 0
        ),
    }
    model.update()
    assert_pure_master(model)


def assert_pure_master(model: gp.Model) -> Dict[str, object]:
    """Fail fast if operational variables or conic constraints leak in."""

    forbidden_prefixes = (
        "psi",
        "rho_",
        "alpha",
        "beta",
        "varphi",
        "lambda",
        "mu",
        "xi",
        "q_v",
        "q_t",
        "h_v",
        "h_t",
    )
    forbidden = [
        var.VarName
        for var in model.getVars()
        if var.VarName.startswith(forbidden_prefixes)
    ]
    if forbidden or int(model.NumQConstrs) != 0:
        raise RuntimeError(
            "20260623 master invariant failed: "
            f"forbidden={forbidden[:10]}, qconstrs={model.NumQConstrs}"
        )
    return {
        "master_is_milp": True,
        "master_qconstrs": int(model.NumQConstrs),
        "master_forbidden_operational_variables": forbidden,
        "master_num_vars": int(model.NumVars),
        "master_num_constrs": int(model.NumConstrs),
    }
