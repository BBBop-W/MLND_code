"""Conic path formulation and strict, all-candidate column generation.

The data contain the complete finite candidate path universe.  Pricing scans that
universe, including paths visiting closed facilities.  A separate explicit SOC
dual is solved so that Benders cuts do not depend on quadratic-dual recovery.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Any

import gurobipy as gp
from gurobipy import GRB
import numpy as np
from scipy import sparse

from .data import Instance


STATUS = {GRB.OPTIMAL: "optimal", GRB.TIME_LIMIT: "time_limit",
          GRB.SUBOPTIMAL: "suboptimal", GRB.INFEASIBLE: "infeasible",
          GRB.INF_OR_UNBD: "inf_or_unbounded", GRB.UNBOUNDED: "unbounded",
          GRB.NUMERIC: "numeric", GRB.INTERRUPTED: "interrupted"}


@dataclass
class _Matrix:
    names: list[str]
    senses: list[str]
    constants: np.ndarray
    maps: list[dict[str, float]]
    base: np.ndarray
    service: sparse.csr_matrix
    empty: sparse.csr_matrix
    cost_service: np.ndarray
    cost_empty: np.ndarray


def _matrix(inst: Instance) -> _Matrix:
    """Compile immutable coefficients once per instance, shared by CG rounds."""
    if hasattr(inst, "_solver_matrix"):
        return inst._solver_matrix
    names, senses, constants, maps, bases = [], [], [], [], []
    rid = {}
    L = inst.n_l

    def row(name, sense="<", constant=0.0, mapping=None, base=None):
        rid[name] = len(names)
        names.append(name)
        senses.append(sense)
        constants.append(constant)
        maps.append(mapping or {})
        coeff = np.zeros(2 * L)
        for k, value in (base or {}).items():
            coeff[k] = value
        bases.append(coeff)

    for l in range(L):
        row(f"open_{l}", mapping={f"X{l}": inst.rho_max}, base={l: 1})
        row(f"balance_{l}", sense="=")
        row(f"clock_{l}", base={l: -inst.kappa})
    for o, h in enumerate(inst.demand):
        M = h * inst.rho_max
        for l in range(L):
            mapping = {f"B{o}:{l}:{m}": -M for m in range(L)}
            row(f"upper_{o}_{l}", constant=M, mapping=mapping, base={l: -h})
            row(f"lower_{o}_{l}", constant=M, mapping=mapping, base={l: h})
        for l in range(L):
            for m in range(L):
                row(f"pair_{o}_{l}_{m}", mapping={f"B{o}:{l}:{m}": M})
    for h in range(inst.n_h):
        row(f"hub_{h}", mapping={f"Y{h}": inst.M})
    for k in range(L + inst.n_k):
        row(f"swap_{k}", mapping={inst.station_key(k): inst.MK})

    sr, sc, sv, er, ec, ev = [], [], [], [], [], []

    def add(rows, cols, values, name, col, value):
        if value:
            rows.append(rid[name]); cols.append(col); values.append(value)

    for p, c in enumerate(inst.service):
        o, l, m = c["od"], c["l"], c["m"]
        for a in range(L):
            add(sr, sc, sv, f"upper_{o}_{a}", p, 1)
            add(sr, sc, sv, f"lower_{o}_{a}", p, -1)
        add(sr, sc, sv, f"pair_{o}_{l}_{m}", p, 1)
        add(sr, sc, sv, f"balance_{l}", p, 1)
        add(sr, sc, sv, f"balance_{m}", p, -1)
        for h in {c["i"], c["j"]}:
            add(sr, sc, sv, f"hub_{h}", p, 1)
        for k, count in enumerate(c["swaps"]):
            add(sr, sc, sv, f"swap_{k}", p, count)
    for r, c in enumerate(inst.empty):
        l, m = c["l"], c["m"]
        add(er, ec, ev, f"balance_{l}", r, 1)
        add(er, ec, ev, f"balance_{m}", r, -1)
        add(er, ec, ev, f"clock_{l}", r, 1)
        for k, count in enumerate(c["swaps"]):
            add(er, ec, ev, f"swap_{k}", r, count)
    nr = len(names)
    answer = _Matrix(names, senses, np.asarray(constants), maps, np.asarray(bases),
        sparse.coo_matrix((sv, (sr, sc)), shape=(nr, len(inst.service))).tocsr(),
        sparse.coo_matrix((ev, (er, ec)), shape=(nr, len(inst.empty))).tocsr(),
        np.asarray([inst.service_cost(p) for p in range(len(inst.service))]),
        np.asarray([inst.empty_cost(r) for r in range(len(inst.empty))]))
    inst._solver_matrix = answer
    return answer


def _parameters(model, time_limit, threads, output, mip=False):
    model.Params.OutputFlag = int(output)
    model.Params.Threads = threads
    model.Params.NonConvex = 0
    model.Params.TimeLimit = max(0.001, float(time_limit))
    model.Params.FeasibilityTol = 1e-8
    model.Params.OptimalityTol = 1e-8
    model.Params.BarQCPConvTol = 1e-8
    if mip:
        model.Params.MIPGap = 1e-6


def build_model(inst: Instance, design=None, service_pool=None, empty_pool=None,
                relax_design=False, time_limit=3600.0, threads=1, output=False,
                mip_gap=1e-4):
    """Return a Gurobi model; caller owns disposal. Metadata are in _path_data."""
    matrix = _matrix(inst)
    S = list(range(len(inst.service))) if service_pool is None else sorted(set(service_pool))
    E = list(range(len(inst.empty))) if empty_pool is None else sorted(set(empty_pool))
    model = gp.Model("mlnd_path_recourse" if design is not None else "mlnd_path_full")
    _parameters(model, time_limit, threads, output, design is None)
    if design is None:
        model.Params.MIPGap = mip_gap
    rho = model.addVars(inst.n_l, lb=0, name="rho")
    idle = model.addVars(inst.n_l, lb=0, obj=inst.fleet_cost, name="idle")
    theta = model.addVars(S, lb=0, obj={p: matrix.cost_service[p] for p in S}, name="theta")
    phi = model.addVars(E, lb=0, obj={r: matrix.cost_empty[r] for r in E}, name="phi")
    dv = None
    if design is None:
        dv = model.addVars(inst.design_keys(), lb=0, ub=1,
                           vtype=GRB.CONTINUOUS if relax_design else GRB.BINARY, name="design")
        for l, c in enumerate(inst.base_cost): dv[f"X{l}"].Obj = c
        for h, c in enumerate(inst.hub_cost): dv[f"Y{h}"].Obj = c
        for k, c in enumerate(inst.swap_cost): dv[f"S{k+inst.n_l}"].Obj = c
        for o in range(inst.n_od):
            model.addConstr(gp.quicksum(dv[f"B{o}:{l}:{m}"]
                for l in range(inst.n_l) for m in range(inst.n_l)) <= 1, name=f"assign_{o}")
            for l in range(inst.n_l):
                for m in range(inst.n_l):
                    b = dv[f"B{o}:{l}:{m}"]
                    model.addConstr(b <= dv[f"X{l}"], name=f"origin_{o}_{l}_{m}")
                    model.addConstr(b <= dv[f"X{m}"], name=f"destination_{o}_{l}_{m}")
    basevars = list(rho.values()) + list(idle.values())
    sv, ev = list(theta.values()), list(phi.values())
    Smat, Emat = matrix.service[:, S], matrix.empty[:, E]
    constraints = []
    rhs_values = []
    for j, name in enumerate(matrix.names):
        left = gp.LinExpr(matrix.base[j].tolist(), basevars)
        for mat, variables in ((Smat, sv), (Emat, ev)):
            a, b = mat.indptr[j:j+2]
            left += gp.LinExpr(mat.data[a:b].tolist(), [variables[k] for k in mat.indices[a:b]])
        if design is None:
            rhs = matrix.constants[j] + gp.quicksum(c * dv[k] for k, c in matrix.maps[j].items())
        else:
            rhs = matrix.constants[j] + sum(c * design[k] for k, c in matrix.maps[j].items())
            rhs_values.append(float(rhs))
        constraints.append(model.addConstr(left == rhs if matrix.senses[j] == "=" else left <= rhs,
                                           name=name))
    for l in range(inst.n_l):
        up = model.addVar(lb=0, name=f"cone_up_{l}")
        low = model.addVar(lb=0, name=f"cone_low_{l}")
        one = model.addVar(lb=1, ub=1, name=f"cone_one_{l}")
        model.addConstr(up - idle[l] == 1)
        model.addConstr(low + rho[l] == 1)
        model.addQConstr(one * one <= up * low, name=f"idle_cone_{l}")
    model.ObjCon = inst.c0
    model.ModelSense = GRB.MINIMIZE
    model._path_data = dict(instance=inst, matrix=matrix, S=S, E=E, rho=rho, idle=idle,
                            theta=theta, phi=phi, design=dv, fixed_design=design,
                            constraints=constraints, rhs=np.asarray(rhs_values))
    model.update()
    return model


def _solution(model, elapsed):
    d = model._path_data
    inst, matrix = d["instance"], d["matrix"]
    answer = dict(status=STATUS.get(model.Status, str(model.Status)), status_code=model.Status,
                  wall=elapsed, runtime=float(model.Runtime), modelruntime=float(model.Runtime),
                  nodes=float(model.NodeCount) if model.IsMIP else 0.0,
                  objective=None, bound=None, gap=None, cut_valid=False,
                  service_pool=list(d["S"]), empty_pool=list(d["E"]))
    if model.IsMIP:
        answer["bound"] = float(model.ObjBound)
    if model.SolCount == 0:
        return answer
    rho = np.asarray([d["rho"][l].X for l in range(inst.n_l)])
    idle = np.asarray([d["idle"][l].X for l in range(inst.n_l)])
    theta = np.zeros(len(inst.service)); phi = np.zeros(len(inst.empty))
    for p, v in d["theta"].items(): theta[p] = v.X
    for r, v in d["phi"].items(): phi[r] = v.X
    design = ({k: float(v.X) for k, v in d["design"].items()}
              if d["design"] is not None else dict(d["fixed_design"]))
    rhs = matrix.constants + np.asarray([sum(value * design[k] for k, value in mapping.items())
                                         for mapping in matrix.maps])
    residual = matrix.base @ np.r_[rho, idle] + matrix.service @ theta + matrix.empty @ phi - rhs
    row_res = max([abs(v) if sense == "=" else max(0., v)
                   for v, sense in zip(residual, matrix.senses)] + [0.])
    cone_res = max([max(0., r / (1-r) - i) if r < 1 else math.inf
                    for r, i in zip(rho, idle)] + [0.])
    nonneg = max(0., -min(np.r_[rho, idle, theta, phi], default=0.))
    fleet = float(idle.sum() + sum(theta[p]*c["time"] for p, c in enumerate(inst.service))
                  + sum(phi[r]*c["time"] for r, c in enumerate(inst.empty)))
    objective = float(model.ObjVal)
    answer.update(objective=objective, rho=rho.tolist(), idle=idle.tolist(),
                  theta=theta.tolist(), phi=phi.tolist(), design=design, fleet=fleet,
                  served=float(theta.sum()), primal_residual=max(row_res, cone_res, nonneg),
                  row_residual=row_res, cone_residual=cone_res)
    if model.IsMIP:
        answer["gap"] = float(model.MIPGap)
    elif model.Status == GRB.OPTIMAL:
        answer.update(bound=objective, gap=0.0)
    return answer


def solve_full(inst: Instance, time_limit=3600.0, threads=1, output=False,
               relax_design=False, mip_gap=1e-4, **kwargs):
    start = time.monotonic()
    model = build_model(inst, relax_design=relax_design, time_limit=time_limit,
                        threads=threads, output=output, mip_gap=mip_gap, **kwargs)
    try:
        remaining = time_limit-(time.monotonic()-start)
        if remaining > 0:
            model.Params.TimeLimit = max(.001, remaining)
            model.optimize()
            answer = _solution(model, time.monotonic()-start)
        else:
            answer = dict(status="time_limit", status_code=GRB.TIME_LIMIT,
                          wall=time.monotonic()-start, runtime=0., modelruntime=0.,
                          nodes=0., objective=None, bound=0., gap=None, cut_valid=False,
                          service_pool=list(model._path_data["S"]),
                          empty_pool=list(model._path_data["E"]))
        answer["build_time"] = answer["wall"] - answer["runtime"]
        if answer["objective"] is None:
            answer.update(objective=inst.c0, design=inst.zero_design(),
                          rho=[0.]*inst.n_l, idle=[0.]*inst.n_l,
                          theta=[0.]*len(inst.service), phi=[0.]*len(inst.empty),
                          fleet=0., served=0., primal_residual=0., row_residual=0.,
                          cone_residual=0., feasible_fallback="all_ground")
        if answer["bound"] is None or not math.isfinite(answer["bound"]):
            answer["bound"] = 0.
        if answer["gap"] is None or not math.isfinite(answer["gap"]):
            answer["gap"] = max(0., answer["objective"]-answer["bound"]) / max(1., abs(answer["objective"]))
        return answer
    finally:
        model.dispose()


def _explicit_dual(inst, design, S, E, time_limit, threads, output):
    """Solve the dual and return a numerically projected, globally bounded cut."""
    start = time.monotonic()
    matrix = _matrix(inst)
    dual = gp.Model("mlnd_path_explicit_soc_dual")
    _parameters(dual, time_limit, threads, output)
    dual.Params.BarQCPConvTol = 1e-9
    nr = len(matrix.names)
    prices = dual.addVars(nr, lb=-GRB.INFINITY,
                          ub={j: GRB.INFINITY if s == "=" else 0 for j, s in enumerate(matrix.senses)},
                          name="price")
    pv = list(prices.values())
    cones = []
    for l in range(inst.n_l):
        t = dual.addVar(lb=0, name=f"soc_t_{l}")
        s = dual.addVar(lb=-GRB.INFINITY, name=f"soc_s_{l}")
        z = dual.addVar(lb=-GRB.INFINITY, name=f"soc_z_{l}")
        dual.addQConstr(s*s + z*z <= t*t)
        cones.append((t, s, z))
    for mat, ids, costs, name in ((matrix.service.tocsc(), S, matrix.cost_service, "service"),
                                  (matrix.empty.tocsc(), E, matrix.cost_empty, "empty")):
        for p in ids:
            a, b = mat.indptr[p:p+2]
            dual.addConstr(gp.LinExpr(mat.data[a:b].tolist(), [pv[j] for j in mat.indices[a:b]])
                           <= costs[p], name=f"{name}_{p}")
    for l, (t, s, z) in enumerate(cones):
        dual.addConstr(gp.LinExpr(matrix.base[:, l].tolist(), pv) - t + z <= 0,
                       name=f"rho_{l}")
        dual.addConstr(t + z <= inst.fleet_cost, name=f"idle_{l}")
    rhs = matrix.constants + np.asarray([sum(c*design[k] for k, c in m.items()) for m in matrix.maps])
    dual.setObjective(inst.c0 + gp.LinExpr(rhs.tolist(), pv)
                      - gp.quicksum(2*(t+s) for t, s, z in cones), GRB.MAXIMIZE)
    build = time.monotonic()-start
    try:
        dual.Params.TimeLimit = max(.001, time_limit-build)
        dual.optimize()
        result = dict(dual_status=STATUS.get(dual.Status, str(dual.Status)),
                      dual_runtime=float(dual.Runtime), dual_build_time=build,
                      dual_wall=time.monotonic()-start, dual_value=None, cut_valid=False)
        if dual.SolCount == 0:
            return result
        y = np.asarray([p.X for p in pv])
        for j, sense in enumerate(matrix.senses):
            if sense != "=": y[j] = min(0., y[j])
        projected = []
        for tvar, svar, zvar in cones:
            t, s, z = tvar.X, svar.X, zvar.X
            t = max(0., t, math.hypot(s, z))
            if t + z > inst.fleet_cost:
                factor = inst.fleet_cost / (t + z)
                t *= factor; s *= factor; z *= factor
            projected.append((t, s, z))
        pricing_start = time.monotonic()
        cs = matrix.cost_service - np.asarray(matrix.service.T @ y).reshape(-1)
        ce = matrix.cost_empty - np.asarray(matrix.empty.T @ y).reshape(-1)
        brc = -np.asarray(matrix.base[:, :inst.n_l].T @ y)
        for l, (t, s, z) in enumerate(projected): brc[l] += t-z
        # Keep the ordinary idle cone in the Lagrangian minimization domain.
        # Its one-dimensional conjugate is analytic, so the certificate never
        # requires a floating-point claim about an unbounded idle variable:
        # min_{0<=rho<1,I>=rho/(1-rho)} cN*I+a*rho
        # = 0 if a>=-cN, else -(sqrt(-a)-sqrt(cN))**2.
        a_rho = -np.asarray(matrix.base[:, :inst.n_l].T @ y)
        idle_conjugate = sum(0. if a >= -inst.fleet_cost else
                             -(math.sqrt(-a)-math.sqrt(inst.fleet_cost))**2 for a in a_rho)
        # Finite global bounds repair remaining path-column dual violations.
        # sum_{p:od=o} theta_p <= demand[o]*rho_max, and each origin's total
        # relocation flow <= kappa*rho_max.  These bounds hold for every design.
        worst_service = np.zeros(inst.n_od)
        worst_empty = np.zeros(inst.n_l)
        for p, c in enumerate(inst.service):
            worst_service[c["od"]] = max(worst_service[c["od"]], -cs[p])
        for r, c in enumerate(inst.empty):
            worst_empty[c["l"]] = max(worst_empty[c["l"]], -ce[r])
        correction = float(inst.rho_max * (np.dot(worst_service, inst.demand)
                            + inst.kappa*worst_empty.sum()))
        intercept = inst.c0 + float(np.dot(matrix.constants, y)) + idle_conjugate
        slope = dict.fromkeys(inst.design_keys(), 0.)
        for price, mapping in zip(y, matrix.maps):
            for key, value in mapping.items(): slope[key] += float(price * value)
        # Explicit floating point reserve is distinct from the analytic residual
        # correction; its size is recorded rather than silently loosening cuts.
        reserve = 2e-9 * (1 + abs(inst.c0) + abs(intercept)
                           + sum(abs(v) for v in slope.values()))
        corrected_intercept = intercept-correction-reserve
        value = corrected_intercept + sum(slope[k]*design[k] for k in slope)
        result.update(dual_value=float(value), dual_value_raw=float(dual.ObjVal),
                      cut_intercept=float(corrected_intercept), cut_slope=slope,
                      rc_service=cs.tolist(), rc_empty=ce.tolist(),
                      minimum_rc=float(min(np.r_[cs, ce], default=0.)),
                      dual_row_residual=float(max(0., -min(np.r_[cs, ce, brc], default=0.))),
                      cut_correction=correction, roundoff_reserve=reserve,
                      dual_cone_projected=True, cut_valid=True,
                      full_pricing_seconds=time.monotonic()-pricing_start,
                      certificate_method="analytic_idle_conjugate_and_global_flow_bounds")
        return result
    finally:
        dual.dispose()


def solve_recourse(inst: Instance, design, service_pool=None, empty_pool=None,
                    time_limit=3600.0, threads=1, output=False,
                    pricing_tol=1e-6, certificate_tol=2e-6):
    start = time.monotonic()
    model = build_model(inst, design=design, service_pool=service_pool,
                        empty_pool=empty_pool, time_limit=time_limit,
                        threads=threads, output=output)
    build_time = time.monotonic()-start
    try:
        model.Params.TimeLimit = max(.001, time_limit-build_time)
        model.optimize()
        answer = _solution(model, time.monotonic()-start)
        answer["build_time"] = build_time
        S, E = list(model._path_data["S"]), list(model._path_data["E"])
    finally:
        model.dispose()
    if answer["objective"] is None:
        # Relative complete recourse supplies a genuine incumbent even when the
        # conic solver is interrupted before its first numerical solution.
        answer.update(objective=inst.c0, bound=0., gap=1., design=dict(design),
                      rho=[0.]*inst.n_l, idle=[0.]*inst.n_l,
                      theta=[0.]*len(inst.service), phi=[0.]*len(inst.empty),
                      fleet=0., served=0., primal_residual=0., row_residual=0.,
                      cone_residual=0., feasible_fallback="all_ground")
    remaining = time_limit-(time.monotonic()-start)
    if answer["objective"] is not None and remaining > .005:
        dual = _explicit_dual(inst, design, S, E, remaining, threads, output)
        answer.update(dual)
        if dual.get("dual_value") is not None:
            gap = answer["objective"]-dual["dual_value"]
            answer["duality_gap"] = float(gap)
            answer["corrected_duality_gap"] = float(gap)
            answer["raw_duality_gap"] = float(answer["objective"]-dual["dual_value_raw"])
            answer["bound"] = dual["dual_value"]
            answer["gap"] = max(0., gap)/(1+abs(answer["objective"]))
            answer["cg_complete"] = bool(dual["minimum_rc"] >= -pricing_tol
                and answer.get("primal_residual", math.inf) <= 2e-6
                and abs(gap) <= certificate_tol*(1+abs(answer["objective"])))
    answer["wall"] = time.monotonic()-start
    return answer


def solve_cg(inst: Instance, design, pools=None, time_limit=3600.0, threads=1,
             output=False, pricing_tol=1e-6, max_iterations=1000,
             certificate_tol=2e-6, **kwargs):
    """Solve a fixed design from empty pools, scanning every candidate each round.

    The restricted primal is always feasible. A time-limited answer carries that
    feasible objective but never advertises a Benders certificate prematurely.
    """
    start = time.monotonic()
    if pools is None:
        S, E = set(), set()
    elif isinstance(pools, dict):
        S = set(pools.get("service", pools.get("service_pool", [])))
        E = set(pools.get("empty", pools.get("empty_pool", [])))
    else:
        S, E = set(pools[0]), set(pools[1])
    trace, cumulative_primal, cumulative_dual, cumulative_build = [], 0., 0., 0.
    answer = dict(objective=inst.c0, bound=0., gap=1., status="time_limit",
                  cut_valid=False, design=dict(design), rho=[0.]*inst.n_l,
                  idle=[0.]*inst.n_l, theta=[0.]*len(inst.service),
                  phi=[0.]*len(inst.empty), fleet=0., primal_residual=0.)
    for iteration in range(max_iterations):
        remaining = time_limit-(time.monotonic()-start)
        if remaining <= .005: break
        answer = solve_recourse(inst, design, S, E, time_limit=remaining,
                                threads=threads, output=output, pricing_tol=pricing_tol,
                                certificate_tol=certificate_tol)
        cumulative_primal += answer.get("runtime", 0.)
        cumulative_dual += answer.get("dual_runtime", 0.)
        cumulative_build += answer.get("build_time", 0.) + answer.get("dual_build_time", 0.)
        entry = dict(iteration=iteration+1, service_columns=len(S), empty_columns=len(E),
                     objective=answer.get("objective"), min_rc=answer.get("minimum_rc"),
                     duality_gap=answer.get("duality_gap"), wall=time.monotonic()-start,
                     pricing_seconds=answer.get("full_pricing_seconds", 0.))
        trace.append(entry)
        if answer.get("cg_complete"):
            answer["status"] = "optimal"
            break
        if "rc_service" not in answer:
            break
        pricing_start = time.monotonic()
        bestS, bestE = {}, {}
        for p, rc in enumerate(answer["rc_service"]):
            if p not in S and rc < -pricing_tol:
                c = inst.service[p]; key = (c["od"], c["l"], c["m"])
                if key not in bestS or rc < bestS[key][0]: bestS[key] = (rc, p)
        for r, rc in enumerate(answer["rc_empty"]):
            if r not in E and rc < -pricing_tol:
                c = inst.empty[r]; key = (c["l"], c["m"])
                if key not in bestE or rc < bestE[key][0]: bestE[key] = (rc, r)
        entry["pricing_seconds"] += time.monotonic()-pricing_start
        entry["added_service"] = len(bestS); entry["added_empty"] = len(bestE)
        if not bestS and not bestE:
            answer["status"] = "uncertified_dual"
            break
        S.update(p for rc, p in bestS.values()); E.update(r for rc, r in bestE.values())
    else:
        answer["status"] = "iteration_limit"
    if not answer.get("cg_complete") and time.monotonic()-start >= time_limit:
        answer["status"] = "time_limit"
    answer.update(wall=time.monotonic()-start, cg_trace=trace, cg_iterations=len(trace),
                  service_pool=sorted(S), empty_pool=sorted(E),
                  cumulative_primal_runtime=cumulative_primal,
                  cumulative_dual_runtime=cumulative_dual,
                  cumulative_build_time=cumulative_build,
                  pricing_time=sum(t.get("pricing_seconds", 0.) for t in trace),
                  pricing_wall=sum(t.get("pricing_seconds", 0.) for t in trace))
    return answer
