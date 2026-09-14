"""Optional core-point selection of a strong, globally priced Benders cut.

This is a Magnanti--Wong-style auxiliary dual: preserve the generating design's
near-optimal face, and maximize the cut at an interior core point. Missing path
constraints are separated over the entire finite candidate universe. The final
certificate uses the analytic idle-cone conjugate and global flow bounds, exactly
as the baseline solver, rather than trusting numerical SOC multipliers.

If the deadline expires before complete pricing and a tight generating-point
certificate, the original certified cut is returned unchanged.
"""
from __future__ import annotations

import math
import time

import gurobipy as gp
from gurobipy import GRB
import numpy as np

from .data import Instance
from .solver import _matrix, _parameters, STATUS


def _evaluate_cut(answer, point):
    return answer["cut_intercept"] + sum(c*point[k] for k, c in answer["cut_slope"].items())


def strengthen_cut(inst: Instance, design, recourse_answer, core, pools=None,
                   time_limit=10.0, threads=1, pricing_tol=1e-6,
                   face_tol=1e-7, tightness_tol=2e-6, output=False):
    """Return the strengthened cut, or a marked copy of the baseline answer.

    ``cg_complete`` describes the returned cut. ``strengthening_complete`` and
    ``strengthened`` separately describe this auxiliary solve. Therefore a
    fallback does not erase the baseline's existing pricing certificate.
    """
    started = time.monotonic()
    fallback = dict(recourse_answer)
    stats = dict(iterations=0, added_service=0, added_empty=0,
                 dual_runtime=0., pricing_seconds=0., build_seconds=0.)

    def finish_fallback(reason):
        elapsed=time.monotonic()-started
        fallback.update(strengthened=False, strengthening_complete=False,
                        strengthening_status=reason,
                        strengthening_wall=elapsed, wall=elapsed,
                        strengthening_stats=stats, use=False,
                        core_improvement=0.,
                        auxiliary_optimal=False,
                        tight_gap=recourse_answer.get("corrected_duality_gap"))
        return fallback

    if not recourse_answer.get("cg_complete") or not recourse_answer.get("cut_valid"):
        return finish_fallback("uncertified_baseline")
    matrix = _matrix(inst)
    if pools is None:
        S, E = set(recourse_answer.get("service_pool", [])), set(recourse_answer.get("empty_pool", []))
    elif isinstance(pools, dict):
        S = set(pools.get("service", pools.get("service_pool", [])))
        E = set(pools.get("empty", pools.get("empty_pool", [])))
    else:
        S, E = set(pools[0]), set(pools[1])
    dual = gp.Model("mlnd_path_core_point_dual")
    _parameters(dual, time_limit, threads, output)
    dual.Params.BarQCPConvTol = 1e-9
    nr = len(matrix.names)
    prices = dual.addVars(nr, lb=-GRB.INFINITY,
        ub={j: GRB.INFINITY if sense == "=" else 0 for j, sense in enumerate(matrix.senses)},
        name="price")
    pv = list(prices.values())
    cones = []
    for l in range(inst.n_l):
        t = dual.addVar(lb=0, name=f"soc_t_{l}")
        s = dual.addVar(lb=-GRB.INFINITY, name=f"soc_s_{l}")
        z = dual.addVar(lb=-GRB.INFINITY, name=f"soc_z_{l}")
        dual.addQConstr(s*s+z*z <= t*t)
        dual.addConstr(gp.LinExpr(matrix.base[:, l].tolist(), pv)-t+z <= 0)
        dual.addConstr(t+z <= inst.fleet_cost)
        cones.append((t,s,z))
    smat, emat = matrix.service.tocsc(), matrix.empty.tocsc()

    def add_rows(mat, ids, costs, prefix):
        for p in ids:
            lo, hi = mat.indptr[p:p+2]
            dual.addConstr(gp.LinExpr(mat.data[lo:hi].tolist(),
                               [pv[j] for j in mat.indices[lo:hi]]) <= costs[p],
                           name=f"{prefix}_{p}")

    add_rows(smat, sorted(S), matrix.cost_service, "service")
    add_rows(emat, sorted(E), matrix.cost_empty, "empty")

    def rhs(point):
        return matrix.constants + np.asarray([sum(c*point[k] for k,c in mapping.items())
                                               for mapping in matrix.maps])

    common = inst.c0-gp.quicksum(2*(t+s) for t,s,z in cones)
    at_design = common+gp.LinExpr(rhs(design).tolist(), pv)
    at_core = common+gp.LinExpr(rhs(core).tolist(), pv)
    qhat = float(recourse_answer["objective"])
    baseline_hat = _evaluate_cut(recourse_answer, design)
    target = baseline_hat-face_tol*(1+abs(qhat))
    dual.addConstr(at_design >= target, name="generating_design_face")
    dual.setObjective(at_core, GRB.MAXIMIZE)
    stats["build_seconds"] = time.monotonic()-started
    baseline_core = _evaluate_cut(recourse_answer, core)
    try:
        while time.monotonic()-started < time_limit:
            dual.Params.TimeLimit = max(.001, time_limit-(time.monotonic()-started))
            dual.optimize()
            stats["iterations"] += 1
            stats["dual_runtime"] += float(dual.Runtime)
            stats["last_dual_status"] = STATUS.get(dual.Status, str(dual.Status))
            if dual.SolCount == 0:
                return finish_fallback("no_auxiliary_solution")
            pricing_start = time.monotonic()
            y = np.asarray([variable.X for variable in pv])
            for j, sense in enumerate(matrix.senses):
                if sense != "=": y[j] = min(0., y[j])
            cs = matrix.cost_service-np.asarray(matrix.service.T@y).reshape(-1)
            ce = matrix.cost_empty-np.asarray(matrix.empty.T@y).reshape(-1)
            minrc = float(min(np.r_[cs,ce], default=0.))
            if minrc >= -pricing_tol:
                # Exact minimization over every ordinary idle cone. All other
                # linear rows have sign-correct multipliers. For feasible w,
                # L(w,y)<=objective(w), and the following finite flow bounds
                # make any remaining negative path coefficients explicit.
                a = -np.asarray(matrix.base[:, :inst.n_l].T@y)
                conjugate = sum(0. if value >= -inst.fleet_cost else
                    -(math.sqrt(-value)-math.sqrt(inst.fleet_cost))**2 for value in a)
                worst_service = np.zeros(inst.n_od)
                worst_empty = np.zeros(inst.n_l)
                for p,c in enumerate(inst.service):
                    worst_service[c["od"]] = max(worst_service[c["od"]], -cs[p])
                for r,c in enumerate(inst.empty):
                    worst_empty[c["l"]] = max(worst_empty[c["l"]], -ce[r])
                correction = float(inst.rho_max*(np.dot(worst_service, inst.demand)
                                                 +inst.kappa*worst_empty.sum()))
                intercept = inst.c0+float(matrix.constants@y)+conjugate
                slope = dict.fromkeys(inst.design_keys(),0.)
                for value,mapping in zip(y,matrix.maps):
                    for key,c in mapping.items(): slope[key] += float(value*c)
                reserve = 2e-9*(1+abs(inst.c0)+abs(intercept)+sum(abs(c) for c in slope.values()))
                intercept -= correction+reserve
                hat_value = intercept+sum(c*design[k] for k,c in slope.items())
                core_value = intercept+sum(c*core[k] for k,c in slope.items())
                stats["pricing_seconds"] += time.monotonic()-pricing_start
                stats.update(minimum_rc=minrc, face_target=target,
                             baseline_core_value=baseline_core, core_value=core_value,
                             core_improvement=core_value-baseline_core,
                             generating_gap=qhat-hat_value)
                if abs(qhat-hat_value) > tightness_tol*(1+abs(qhat)):
                    return finish_fallback("generating_point_certificate_not_tight")
                if core_value <= baseline_core+1e-8:
                    return finish_fallback("no_core_improvement")
                answer = dict(recourse_answer)
                answer.update(cut_intercept=float(intercept),cut_slope=slope,
                    dual_value=float(hat_value),duality_gap=float(qhat-hat_value),
                    corrected_duality_gap=float(qhat-hat_value),cut_correction=correction,
                    roundoff_reserve=reserve,rc_service=cs.tolist(),rc_empty=ce.tolist(),
                    minimum_rc=minrc,cut_valid=True,cg_complete=True,
                    service_pool=sorted(S),empty_pool=sorted(E),
                    strengthened=True,strengthening_complete=True,
                    strengthening_status="complete",strengthening_wall=time.monotonic()-started,
                    strengthening_stats=stats,
                    use=True,core_improvement=float(core_value-baseline_core),
                    auxiliary_optimal=bool(dual.Status == GRB.OPTIMAL),
                    tight_gap=float(qhat-hat_value),wall=time.monotonic()-started,
                    certificate_method="core_point_dual_with_analytic_idle_conjugate")
                return answer
            bestS,bestE={},{}
            for p,rc in enumerate(cs):
                if p not in S and rc < -pricing_tol:
                    c=inst.service[p];key=(c["od"],c["l"],c["m"])
                    if key not in bestS or rc < bestS[key][0]: bestS[key]=(rc,p)
            for r,rc in enumerate(ce):
                if r not in E and rc < -pricing_tol:
                    c=inst.empty[r];key=(c["l"],c["m"])
                    if key not in bestE or rc < bestE[key][0]: bestE[key]=(rc,r)
            stats["pricing_seconds"] += time.monotonic()-pricing_start
            if not bestS and not bestE:
                return finish_fallback("existing_dual_row_numerics")
            addedS=[p for rc,p in bestS.values()];addedE=[r for rc,r in bestE.values()]
            build_start=time.monotonic()
            add_rows(smat,addedS,matrix.cost_service,"service")
            add_rows(emat,addedE,matrix.cost_empty,"empty")
            S.update(addedS);E.update(addedE)
            stats["added_service"] += len(addedS);stats["added_empty"] += len(addedE)
            stats["build_seconds"] += time.monotonic()-build_start
        return finish_fallback("time_limit")
    finally:
        dual.dispose()
