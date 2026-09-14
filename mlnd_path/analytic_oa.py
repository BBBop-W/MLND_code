"""MILP master with adaptive outer approximation of continuous idle queues.

This experimental variant leaves all previously benchmarked implementations
unchanged. It reuses the full-candidate analytic path oracle and replaces only
the master idle cones with globally valid convex tangents. Availability remains
continuous: tangent abscissae are cutting-plane points, not allowed rho levels.
"""
from __future__ import annotations

import copy
import math
import time

import gurobipy as gp
from gurobipy import GRB
import numpy as np

from .data import Instance
from .partial_benders import _master
from .solver import _matrix
from .strengthening import add_disaggregated_gates
from .analytic_benders import AnalyticPathOracle, _cheap_global_bound


def solve_analytic_oa(inst: Instance, time_limit=30., mip_gap=1e-4, threads=1,
                      output=False, queue_tolerance=1e-7,
                      initial_tangents=(0., .2, .4, .6, .8),
                      inject_incumbent=False):
    """MILP outer approximation with continuous availability and valid true UBs.

    Initial and lazy tangents underestimate the convex idle function globally.
    Every reported incumbent raises idle counts to the exact queue requirement
    before the original matrix and objective are evaluated. The returned gap
    therefore always uses the true nonlinear upper bound and a MILP lower bound.
    """
    if queue_tolerance < 0:
        raise ValueError('queue_tolerance must be nonnegative')
    start = time.monotonic()
    deadline = start + max(0., float(time_limit))
    oracle = AnalyticPathOracle(inst)
    checked_instance = copy.copy(inst)
    checked_instance.metadata = dict(inst.metadata)
    checked_matrix = _matrix(add_disaggregated_gates(checked_instance))
    model, x, rho, idle, T, E, zeta = _master(inst, threads, mip_gap)
    model.Params.OutputFlag = int(output)
    idle_quadratics = model.getQConstrs()
    if len(idle_quadratics) != inst.n_l:
        model.dispose()
        raise ValueError('Expected exactly one idle quadratic constraint per base')
    model.remove(idle_quadratics)
    knots = sorted({min(inst.rho_max, max(0., float(a)))
                    for a in (*initial_tangents, inst.rho_max)})
    for l in range(inst.n_l):
        for j, a in enumerate(knots):
            value = a/(1-a)
            derivative = 1/(1-a)**2
            model.addConstr(idle[l] >= value+derivative*(rho[l]-a),
                            name=f'idle_initial_tangent_{l}_{j}')
    model.update()
    if model.NumQConstrs:
        model.dispose()
        raise ValueError('The OA master must be a MILP')
    variables = dict(T)
    variables.update(E)
    eta = model.addVars([g.key for g in oracle.groups], lb=-GRB.INFINITY, name='group_cost')
    for group in oracle.groups:
        model.addConstr(eta[group.key] >= group.cheapest * variables[group.key],
                        name=f'group_global_bound_{group.key}')
        eta[group.key].Start = 0.
    model.addConstr(zeta >= gp.quicksum(eta.values()), name='sum_group_path_costs')
    model.update()
    all_variables = model.getVars()
    idle_aux_up = [model.getVarByName(f'up_{l}') for l in range(inst.n_l)]
    idle_aux_low = [model.getVarByName(f'low_{l}') for l in range(inst.n_l)]
    idle_aux_one = [model.getVarByName(f'one_{l}') for l in range(inst.n_l)]
    stats = dict(subproblem_calls=0, oracle_calls=0, oracle_patterns=0, oracle_cache_hits=0,
                 cuts=0, feasibility_cuts=0, optimality_cuts=0, cg_rounds=0, lp_runtime=0.,
                 subproblem_wall=0., pricing_wall=0., callback_wall=0., master_wall=0.,
                 maximum_cut_dual_residual=0., rejected_primal_candidates=0,
                 queue_cuts=0, initial_queue_cuts=len(knots)*inst.n_l,
                 maximum_queue_violation=0., maximum_idle_repair=0.,
                 incumbent_injection_requests=0, incumbent_injection_immediate=0,
                 incumbent_injection_errors=[])
    best = dict(objective=inst.c0, design=inst.zero_design(), rho=[0.]*inst.n_l,
                idle=[0.]*inst.n_l, theta=[0.]*len(inst.service), phi=[0.]*len(inst.empty),
                fleet=0., served=0., primal_residual=0.)
    lower = _cheap_global_bound(inst)
    trace, errors, incomplete = [], [], []
    touched_service, touched_empty = set(), set()

    def validate_point(design, rv, iv, point, answer):
        if np.any(rv >= 1):
            return None, math.inf
        exact_idle = rv/(1-rv)
        repaired_idle = np.maximum(iv, exact_idle)
        idle_repair = float(np.maximum(repaired_idle-iv, 0.).sum())
        iv = repaired_idle
        stats['maximum_idle_repair'] = max(stats['maximum_idle_repair'], idle_repair)
        theta = np.zeros(len(inst.service))
        phi = np.zeros(len(inst.empty))
        aggregate_error = 0.
        for group in answer['groups']:
            amount = point[group['key']]
            if not group['feasible']:
                aggregate_error = max(aggregate_error, abs(amount))
            elif group['service']:
                theta[group['column']] = amount
                touched_service.add(group['column'])
            else:
                phi[group['column']] = amount
                touched_empty.add(group['column'])
        matrix = checked_matrix
        rhs = matrix.constants + np.asarray([sum(c*design[k] for k,c in mapping.items())
                                             for mapping in matrix.maps])
        residual = matrix.base @ np.r_[rv,iv] + matrix.service @ theta + matrix.empty @ phi - rhs
        row_error = max([abs(a) if sense=='=' else max(0.,a)
                         for a,sense in zip(residual,matrix.senses)] + [0.])
        cone_error = max([max(0.,r/(1-r)-i) if r<1 else math.inf for r,i in zip(rv,iv)] + [0.])
        nonnegative = max(0.,-min(np.r_[rv,iv,theta,phi],default=0.))
        assignment_error = 0.
        for o in range(inst.n_od):
            assigned = 0.
            for l in range(inst.n_l):
                for m in range(inst.n_l):
                    b = design[f'B{o}:{l}:{m}']
                    assigned += b
                    assignment_error = max(assignment_error,b-design[f'X{l}'],b-design[f'X{m}'])
            assignment_error = max(assignment_error,assigned-1.)
        error = float(max(row_error,cone_error,nonnegative,aggregate_error,assignment_error))
        if error > 1e-5 or not math.isfinite(error):
            return None, error
        path_cost = float(matrix.cost_service @ theta + matrix.cost_empty @ phi)
        objective = float(inst.fixed_cost(design)+inst.c0+inst.fleet_cost*iv.sum()+path_cost)
        fleet = float(iv.sum() + sum(theta[p]*c['time'] for p,c in enumerate(inst.service))
                      + sum(phi[r]*c['time'] for r,c in enumerate(inst.empty)))
        return dict(objective=objective,design=design,rho=rv.tolist(),idle=iv.tolist(),
                    theta=theta.tolist(),phi=phi.tolist(),fleet=fleet,served=float(theta.sum()),
                    primal_residual=error, idle_repair=idle_repair),error

    def callback(master, where):
        nonlocal lower, best
        if time.monotonic() >= deadline:
            master.terminate()
            return
        if where == GRB.Callback.MIP:
            value = master.cbGet(GRB.Callback.MIP_OBJBND)
            if math.isfinite(value) and abs(value)<1e90:
                lower = max(lower,float(value))
        if where != GRB.Callback.MIPSOL:
            return
        callback_start = time.monotonic()
        try:
            design = {key:float(round(master.cbGetSolution(v))) for key,v in x.items()}
            point = {key:float(master.cbGetSolution(v)) for key,v in variables.items()}
            rv = np.asarray([master.cbGetSolution(rho[l]) for l in range(inst.n_l)])
            iv = np.asarray([master.cbGetSolution(idle[l]) for l in range(inst.n_l)])
            estimates = {key:float(master.cbGetSolution(v)) for key,v in eta.items()}
            queue_added = 0
            for l in range(inst.n_l):
                a = min(inst.rho_max, max(0., float(rv[l])))
                value = a/(1-a)
                derivative = 1/(1-a)**2
                tangent_value = value+derivative*(float(rv[l])-a)
                violation = tangent_value-float(iv[l])
                stats['maximum_queue_violation'] = max(stats['maximum_queue_violation'], violation)
                if violation > queue_tolerance:
                    master.cbLazy(idle[l] >= value+derivative*(rho[l]-a))
                    stats['queue_cuts'] += 1
                    stats['cuts'] += 1
                    queue_added += 1
            oracle_start = time.monotonic()
            answer, cached = oracle.evaluate(design,deadline)
            duration = time.monotonic()-oracle_start
            stats['subproblem_calls'] += 1
            stats['oracle_calls'] += 1
            stats['oracle_cache_hits'] += int(cached)
            stats['oracle_patterns'] = len(oracle.cache)
            stats['subproblem_wall'] += duration
            if not cached:
                stats['pricing_wall'] += duration
            stats['maximum_cut_dual_residual'] = max(stats['maximum_cut_dual_residual'],
                                                     answer['maximum_dual_residual'])
            added = queue_added
            for group in answer['groups']:
                key = group['key']
                if not group['feasible']:
                    rhs_value = group['maximum'] * sum(design[h] for h in group['cover'])
                    if point[key] > rhs_value+1e-7:
                        master.cbLazy(variables[key] <= group['maximum'] *
                                      gp.quicksum(x[h] for h in group['cover']))
                        stats['feasibility_cuts'] += 1
                        stats['cuts'] += 1
                        added += 1
                else:
                    value = (group['q']*point[key]-group['maximum']*
                             sum(c*design[h] for h,c in group['penalties'].items())-group['reserve'])
                    if estimates[key] < value-1e-7:
                        expression = (group['q']*variables[key]-group['maximum']*
                                      gp.quicksum(c*x[h] for h,c in group['penalties'].items())
                                      -group['reserve'])
                        master.cbLazy(eta[key] >= expression)
                        stats['optimality_cuts'] += 1
                        stats['cuts'] += 1
                        added += 1
            candidate, error = validate_point(design,rv,iv,point,answer)
            if candidate is not None and candidate['objective'] < best['objective']:
                best = candidate
                if inject_incumbent:
                    # Supply all master variables, including auxiliary idle
                    # links and group costs, to avoid partial-solution search.
                    try:
                        proposed = list(master.cbGetSolution(all_variables))
                        for key, variable in x.items():
                            proposed[variable.index] = design[key]
                        for l in range(inst.n_l):
                            exact_i = candidate['idle'][l]
                            proposed[idle[l].index] = exact_i
                            proposed[idle_aux_up[l].index] = 1+exact_i
                            proposed[idle_aux_low[l].index] = 1-rv[l]
                            proposed[idle_aux_one[l].index] = 1.
                        cost_sum = 0.
                        for group in answer['groups']:
                            cost = group['q']*point[group['key']] if group['feasible'] else 0.
                            proposed[eta[group['key']].index] = cost
                            cost_sum += cost
                        proposed[zeta.index] = cost_sum
                        master.cbSetSolution(all_variables, proposed)
                        stats['incumbent_injection_requests'] += 1
                        injected_value = master.cbUseSolution()
                        if math.isfinite(injected_value) and abs(injected_value)<1e90:
                            stats['incumbent_injection_immediate'] += 1
                    except gp.GurobiError as exc:
                        stats['incumbent_injection_errors'].append(str(exc))
            elif candidate is None:
                stats['rejected_primal_candidates'] += 1
                # Missing paths imply cuts; a point with no remaining violated
                # cuts must also pass the independent original-model check.
                if not added:
                    incomplete.append(f'unvalidated_master_point:{error:.6g}')
                    master.terminate()
            trace.append(dict(iteration=len(trace),elapsed=time.monotonic()-start,
                lower_bound=float(lower),upper_bound=best['objective'],
                gap=max(0.,best['objective']-lower)/max(1.,abs(best['objective'])),
                oracle_cached=cached,oracle_patterns=len(oracle.cache),added_cuts=added,
                primal_residual=error, added_queue_cuts=queue_added,
                idle_repair=candidate.get('idle_repair', 0.) if candidate is not None else None))
        except TimeoutError as exc:
            incomplete.append(str(exc))
            master.terminate()
        except Exception as exc:
            errors.append(f'{type(exc).__name__}: {exc}')
            master.terminate()
        finally:
            stats['callback_wall'] += time.monotonic()-callback_start

    try:
        if time.monotonic() < deadline:
            model.Params.TimeLimit = max(.001,deadline-time.monotonic())
            master_start = time.monotonic()
            model.optimize(callback)
            stats['master_wall'] = max(0.,time.monotonic()-master_start-stats['callback_wall'])
            try:
                value = float(model.ObjBound)
                if math.isfinite(value) and abs(value)<1e90:
                    lower = max(lower,value)
            except (gp.GurobiError,AttributeError):
                pass
        gap = max(0.,best['objective']-lower)/max(1.,abs(best['objective']))
        status = 'OPTIMAL_TOL' if gap<=mip_gap and not errors and not incomplete else 'TIME_LIMIT'
        if status == 'TIME_LIMIT' and model.Status == GRB.OPTIMAL and not errors and not incomplete:
            status = 'MASTER_TOL_ONLY'
        if errors:
            status = 'CALLBACK_ERROR'
        elif incomplete and time.monotonic()<deadline:
            status = 'INCOMPLETE_RECOURSE'
        if lower-best['objective'] > 1e-5*(1+abs(best['objective'])):
            errors.append('Validated upper bound is below the reported master lower bound')
            status = 'NUMERICAL_BOUND_ERROR'
        return dict(best,algorithm='analytic_oa',status=status,bound=float(lower),
            lower_bound=float(lower),upper_bound=best['objective'],gap=float(gap),
            wall=time.monotonic()-start,stats=stats,trace=trace,errors=errors,
            incomplete_recourse=incomplete,service_pool=len(touched_service),empty_pool=len(touched_empty),
            validated_incumbent=True,activation_certificate=oracle.activation_certificate,
            candidate_service_columns=len(inst.service),candidate_empty_columns=len(inst.empty),
            queue_tolerance=queue_tolerance, initial_tangent_points=knots,
            master_is_milp=model.NumQConstrs==0, master_status_code=int(model.Status),
            inject_incumbent=inject_incumbent)
    finally:
        model.dispose()
