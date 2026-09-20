"""Full-column mathematical formulation for reference and fixed-design solves."""

from dataclasses import dataclass
import time
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from src.model.network import network, levels, facility_costs, vector
from src.solvers.settings import settings


@dataclass
class FullColumnModel:
    model: object
    recourse: object
    continuous: bool
    service: list
    empty: list
    x: object
    z: object
    alpha: object
    y: object
    rho: object
    idle: object
    psi: object
    varphi: object
    fleet: object


class ModelBuildStopped(RuntimeError):
    """The complete-column model could not be built within the requested limits."""


def build_full_column_model(
    data,
    columns,
    *,
    threads=1,
    seed=43,
    mip_gap=1e-4,
    mip_gap_abs=1e-6,
    fixed_facility=None,
    fixed_capacity_level=None,
    fixed_assignment=None,
    fixed_airlines=None,
    time_limit=None,
    stop_requested=None,
):
    """Construct the exact MISOCP, or SOCP when all strategic variables are fixed."""
    deadline = (
        None if time_limit is None else time.perf_counter() + max(0.0, time_limit)
    )

    def check():
        if stop_requested is not None and stop_requested():
            raise ModelBuildStopped("user_stop")
        if deadline is not None and time.perf_counter() >= deadline:
            raise ModelBuildStopped("time_limit")

    check()
    L, H, K, D = (data[k] for k in ("n_l", "n_h", "n_k", "n_od"))
    net = network(data)
    sizes, capacity, levelcost = levels(data)
    cols = columns
    service, empty = cols["service"], cols["empty"]
    P, R = len(service), len(empty)
    model = gp.Model("full_column_reference")
    try:
        check()
        settings(model, None, threads, seed)
        if mip_gap is not None:
            model.Params.MIPGap = mip_gap
        if mip_gap_abs is not None:
            model.Params.MIPGapAbs = mip_gap_abs
        fixed = [fixed_facility, fixed_capacity_level, fixed_assignment, fixed_airlines]
        continuous = all(v is not None for v in fixed)
        dtype = GRB.CONTINUOUS if continuous else GRB.BINARY
        x = model.addMVar(net["N"], lb=0, ub=1, vtype=dtype, name="x")
        z = model.addMVar(capacity.shape, lb=0, ub=1, vtype=dtype, name="z")
        alpha = model.addMVar((D, L, L), lb=0, ub=1, vtype=dtype, name="alpha")
        y = model.addMVar(net["count"], lb=0, ub=1, vtype=dtype, name="y")
        specs = [
            (x, fixed_facility, (net["N"],)),
            (z, fixed_capacity_level, capacity.shape),
            (alpha, fixed_assignment, (D, L, L)),
            (y, fixed_airlines, (net["count"],)),
        ]
        for variable, values, shape in specs:
            if values is None:
                continue
            a = (
                vector(data, values, net)
                if variable is y
                else np.asarray(values, float)
            )
            if (
                a.shape != shape
                or not np.isfinite(a).all()
                or np.any((a != 0) & (a != 1))
            ):
                raise ValueError(
                    "Fixed strategic variables must be binary with the required dimensions"
                )
            variable.LB = a
            variable.UB = a
        rho = model.addVars(L, lb=0, ub=1, name="rho")
        idle = model.addVars(L, lb=0, name="idle")
        psi = model.addVars(P, lb=0, name="psi")
        phi = model.addVars(R, lb=0, name="varphi")
        Gamma = model.addVar(lb=0, name="Gamma")
        model.addConstr(z.sum(axis=1) == x[L : L + H], name="capacity_choice")
        model.addConstr(alpha.sum(axis=(1, 2)) <= 1, name="one_assignment")
        model.addConstr(alpha.sum(axis=2) <= x[:L], name="source_open")
        model.addConstr(alpha.sum(axis=1) <= x[:L], name="destination_open")
        model.addConstr(y <= x[net["uv"][:, 0]], name="link_origin")
        model.addConstr(y <= x[net["uv"][:, 1]], name="link_destination")
        group = {}
        locker = [[] for _ in range(H)]
        sout = [[] for _ in range(L)]
        sin = [[] for _ in range(L)]
        links = [[] for _ in range(net["count"])]
        for k, p in enumerate(service):
            if k % 512 == 0:
                check()
            group.setdefault((p["od"], p["l"], p["m"]), []).append(k)
            locker[p["i"]].append(k)
            locker[p["j"]].append(k)
            sout[p["l"]].append(k)
            sin[p["m"]].append(k)
            for flat, count in p["arc_counts"]:
                e = int(net["lookup"][flat])
                links[e].append((psi[k], count))
        eout = [[] for _ in range(L)]
        ein = [[] for _ in range(L)]
        for k, p in enumerate(empty):
            if k % 512 == 0:
                check()
            eout[p["l"]].append(k)
            ein[p["m"]].append(k)
            for flat, count in p["arc_counts"]:
                e = int(net["lookup"][flat])
                links[e].append((phi[k], count))
        for od in range(D):
            check()
            h = float(data["demand"][od])
            for l in range(L):
                source = gp.LinExpr()
                for m in range(L):
                    flow = gp.quicksum(psi[k] for k in group.get((od, l, m), []))
                    source += flow
                    model.addConstr(
                        flow <= h * alpha[od, l, m], name=f"pair_{od}_{l}_{m}"
                    )
                model.addConstr(source <= h * rho[l], name=f"source_upper_{od}_{l}")
                model.addConstr(
                    h * rho[l] - source <= h * (1 - alpha[od, l, :].sum()),
                    name=f"source_lower_{od}_{l}",
                )
        for l in range(L):
            check()
            model.addConstr(rho[l] <= x[l], name=f"available_{l}")
            model.addConstr(
                gp.quicksum(psi[k] for k in sout[l])
                + gp.quicksum(phi[k] for k in eout[l])
                == gp.quicksum(psi[k] for k in sin[l])
                + gp.quicksum(phi[k] for k in ein[l]),
                name=f"balance_{l}",
            )
            up = model.addVar(lb=0, name=f"cone_u_{l}")
            down = model.addVar(lb=0, name=f"cone_v_{l}")
            one = model.addVar(lb=1, ub=1, name=f"cone_one_{l}")
            model.addConstr(up == idle[l] + 1)
            model.addConstr(down == 1 - rho[l])
            model.addQConstr(one * one <= up * down, name=f"idle_cone_{l}")
        for i in range(H):
            check()
            model.addConstr(
                gp.quicksum(psi[k] for k in locker[i])
                <= gp.quicksum(
                    float(capacity[i, s]) * z[i, s].item() for s in range(len(sizes))
                ),
                name=f"locker_{i}",
            )
        for e, row in enumerate(links):
            if e % 512 == 0:
                check()
            model.addConstr(
                gp.quicksum(count * variable for variable, count in row)
                <= net["upper"][e] * y[e],
                name=f"link_{e}",
            )
        model.addConstr(
            gp.quicksum(idle.values())
            + gp.quicksum(p["time"] * psi[k] for k, p in enumerate(service))
            + gp.quicksum(p["time"] * phi[k] for k, p in enumerate(empty))
            <= Gamma,
            name="fleet",
        )
        c0 = float(np.asarray(data["demand"]) @ np.asarray(data["ground"]))
        recourse = c0 + data["fleet_cost"] * Gamma
        recourse += gp.quicksum(
            (p["operating_cost"] - data["ground"][p["od"]]) * psi[k]
            for k, p in enumerate(service)
        )
        recourse += gp.quicksum(
            p["operating_cost"] * phi[k] for k, p in enumerate(empty)
        )
        strategy = facility_costs(data) @ x + (levelcost * z).sum() + net["fees"] @ y
        model.setObjective(strategy + recourse)
        check()
        return FullColumnModel(
            model,
            recourse,
            continuous,
            service,
            empty,
            x,
            z,
            alpha,
            y,
            rho,
            idle,
            psi,
            phi,
            Gamma,
        )
    except BaseException:
        model.dispose()
        raise
