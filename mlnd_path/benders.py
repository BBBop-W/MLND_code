"""Integer design master with globally priced continuous conic recourse."""
from __future__ import annotations

import math
import time

import gurobipy as gp
from gurobipy import GRB

from .data import Instance
from .solver import solve_cg


def initial_bound_coefficients(inst: Instance):
    """A valid per-OD lower bound allocating a share of the idle cost.

    For each base, assigned demand weights sum to at most one. Dropping empty
    relocation and facility restrictions and minimizing each OD's availability
    independently therefore gives a lower bound for every integer design.
    """
    cheapest = {}
    for p, col in enumerate(inst.service):
        key = (col['od'], col['l'], col['m'])
        cheapest[key] = min(cheapest.get(key, math.inf), inst.service_cost(p))
    coefficients = {}
    for od, h in enumerate(inst.demand):
        beta = inst.fleet_cost * h / sum(inst.demand)
        for l in range(inst.n_l):
            for m in range(inst.n_l):
                c = cheapest.get((od, l, m))
                if c is None:
                    lower = 0.0
                else:
                    a = h * c
                    rho = 0.0 if a >= -beta else min(inst.rho_max, 1 - math.sqrt(beta / -a))
                    lower = a * rho + beta * rho / (1 - rho)
                coefficients[f'B{od}:{l}:{m}'] = lower
    return coefficients


def build_master(inst: Instance, time_limit=60.0, threads=1, mip_gap=1e-4):
    model = gp.Model('MLND_Path_integer_master')
    model.Params.OutputFlag = 0
    model.Params.Threads = threads
    model.Params.Seed = 0
    model.Params.TimeLimit = max(0.01, time_limit)
    model.Params.MIPGap = mip_gap
    model.Params.MIPGapAbs = 1e-6
    design = {k: model.addVar(vtype=GRB.BINARY, name=k) for k in inst.design_keys()}
    zeta = model.addVar(lb=0, name='recourse_bound')
    for od in range(inst.n_od):
        model.addConstr(gp.quicksum(design[f'B{od}:{l}:{m}']
                                    for l in range(inst.n_l) for m in range(inst.n_l)) <= 1)
        for l in range(inst.n_l):
            for m in range(inst.n_l):
                b = design[f'B{od}:{l}:{m}']
                model.addConstr(b <= design[f'X{l}'])
                model.addConstr(b <= design[f'X{m}'])
    coefficient = initial_bound_coefficients(inst)
    model.addConstr(zeta >= inst.c0 + gp.quicksum(c * design[k] for k, c in coefficient.items()),
                    name='idle_allocated_initial_bound')
    fixed = inst.fixed_cost(design)
    model.setObjective(fixed + zeta, GRB.MINIMIZE)
    for v in design.values():
        v.Start = 0
    zeta.Start = inst.c0
    model.update()
    return model, design, zeta


def greedy_design(inst: Instance):
    design = inst.zero_design()
    for od in range(inst.n_od):
        choices = [(inst.service_cost(p), p) for p, c in enumerate(inst.service) if c['od'] == od]
        if not choices or min(choices)[0] >= 0:
            continue
        col = inst.service[min(choices)[1]]
        design[f'B{od}:{col["l"]}:{col["m"]}'] = 1.0
        design[f'X{col["l"]}'] = design[f'X{col["m"]}'] = 1.0
        design[f'Y{col["i"]}'] = design[f'Y{col["j"]}'] = 1.0
        for k, count in enumerate(col['swaps']):
            if count:
                design[inst.station_key(k)] = 1.0
    return design


def _relative_gap(upper, lower):
    return max(0.0, upper - lower) / max(1.0, abs(upper))


def solve_benders(inst: Instance, time_limit=60.0, mip_gap=1e-4, threads=1,
                  mode='loop', warm_start=True, max_iterations=10000):
    """Solve the complete integer design problem, returning certified bounds.

    Both loop and lazy modes use the same globally valid recourse cuts. A
    deadline may interrupt pricing: its feasible primal still improves UB,
    while only certified cuts are passed to the master.
    """
    start = time.perf_counter()
    deadline = start + time_limit
    master, x, zeta = build_master(inst, time_limit, threads, mip_gap)
    cache = {}
    pools = None
    best = {'objective': inst.c0, 'design': inst.zero_design(), 'fleet': 0.0,
            'rho': [0.0] * inst.n_l, 'theta': [], 'phi': [], 'primal_residual': 0.0}
    lower = 0.0
    trace = []
    stats = {'subproblem_calls': 0, 'cg_rounds': 0, 'cuts': 0, 'cache_hits': 0,
             'subproblem_wall': 0.0, 'pricing_wall': 0.0, 'master_wall': 0.0,
             'pareto_calls': 0, 'pareto_used': 0, 'pareto_wall': 0.0,
             'pareto_core_improvement': 0.0}
    core_initial = {k: (0.5 / inst.n_l**2 if k.startswith('B') else 0.5)
                    for k in inst.design_keys()}
    core = dict(core_initial)
    status = 'TIME_LIMIT'
    errors = []

    def remaining():
        return max(0.0, deadline - time.perf_counter())

    def evaluate(point):
        nonlocal pools, best, core
        key = tuple(int(round(point[k])) for k in inst.design_keys())
        if key in cache:
            stats['cache_hits'] += 1
            return cache[key]
        if remaining() <= 0.005:
            return None
        t0 = time.perf_counter()
        answer = solve_cg(inst, point, pools=pools, time_limit=remaining(), threads=threads)
        stats['subproblem_wall'] += time.perf_counter() - t0
        stats['subproblem_calls'] += 1
        stats['cg_rounds'] += answer.get('cg_iterations', len(answer.get('cg_trace', [])))
        stats['pricing_wall'] += answer.get('pricing_wall', 0.0)
        if 'service_pool' in answer and 'empty_pool' in answer:
            pools = (answer['service_pool'], answer['empty_pool'])
        objective = answer.get('objective', math.inf) + inst.fixed_cost(point)
        if math.isfinite(objective) and answer.get('primal_residual', math.inf) <= 1e-5:
            if objective < best['objective']:
                best = {**answer, 'objective': objective, 'design': dict(point)}
        complete = answer.get('cg_complete', False)
        if complete and answer.get('cut_valid', False):
            if mode == 'pareto' and remaining() > 0.02:
                from .pareto import strengthen_cut
                strengthened = strengthen_cut(inst, point, answer, core, pools=pools,
                                              time_limit=min(5.0, remaining()/3), threads=threads)
                stats['pareto_calls'] += 1
                stats['pareto_wall'] += strengthened.get('wall', 0.0)
                stats['subproblem_wall'] += strengthened.get('wall', 0.0)
                if strengthened.get('use', False):
                    for k in ['cut_intercept', 'cut_slope', 'cut_valid']:
                        answer[k] = strengthened[k]
                    stats['pareto_used'] += 1
                    stats['pareto_core_improvement'] += strengthened.get('core_improvement', 0.0)
                core = {k: 0.5*core[k]+0.45*point[k]+0.05*core_initial[k] for k in core}
            cache[key] = answer
        return answer

    def cut_expression(answer):
        return answer['cut_intercept'] + gp.quicksum(v * x[k] for k, v in answer['cut_slope'].items())

    if warm_start and inst.service and remaining() > 0.01:
        initial = greedy_design(inst)
        answer = evaluate(initial)
        if answer and answer.get('cut_valid', False) and answer.get('cg_complete', False):
            master.addConstr(zeta >= cut_expression(answer), name='initial_recourse_cut')
            stats['cuts'] += 1
        for k, variable in x.items():
            variable.Start = best['design'][k]
        zeta.Start = best['objective'] - inst.fixed_cost(best['design'])

    root_result = None
    if mode == 'root' and remaining() > 0.03:
        from .root_cuts import generate_root_cuts
        root_result = generate_root_cuts(inst, master, x, zeta, pools=pools,
                                         time_limit=min(8.0, remaining()*0.25),
                                         max_rounds=80, threads=threads)
        pools = root_result['pools']
        lower = max(lower, root_result['lower_bound'])
        stats['cuts'] += root_result['cuts']
        stats['subproblem_calls'] += root_result['rounds']
        stats['cg_rounds'] += sum(t.get('cg_iterations',0) for t in root_result['trace'])
        stats['subproblem_wall'] += root_result['subproblem_wall']
        stats['pricing_wall'] += root_result['pricing_wall']
        stats['master_wall'] += root_result['master_wall']

    try:
        if mode == 'loop':
            for iteration in range(max_iterations):
                if remaining() <= 0.01:
                    break
                master.Params.TimeLimit = remaining()
                t0 = time.perf_counter()
                master.optimize()
                stats['master_wall'] += time.perf_counter() - t0
                if master.SolCount == 0:
                    status = 'MASTER_' + str(master.Status)
                    break
                lower = max(lower, float(master.ObjBound))
                point = {k: float(round(v.X)) for k, v in x.items()}
                answer = evaluate(point)
                trace.append({'iteration': iteration, 'elapsed': time.perf_counter() - start,
                              'lower_bound': lower, 'upper_bound': best['objective'],
                              'gap': _relative_gap(best['objective'], lower),
                              'master_nodes': master.NodeCount,
                              'cg_complete': bool(answer and answer.get('cg_complete', False))})
                if _relative_gap(best['objective'], lower) <= mip_gap:
                    status = 'OPTIMAL_TOL'
                    break
                if answer is None:
                    break
                if not answer.get('cut_valid', False):
                    status = 'INCOMPLETE_RECOURSE'
                    break
                if not answer.get('cg_complete', False):
                    status = 'INCOMPLETE_PRICING'
                    break
                master.addConstr(zeta >= cut_expression(answer), name=f'recourse_cut_{iteration}')
                stats['cuts'] += 1
            else:
                status = 'ITERATION_LIMIT'
        elif mode in ('lazy', 'pareto', 'root'):
            master.Params.LazyConstraints = 1

            def callback(model, where):
                nonlocal lower
                if remaining() <= 0.01:
                    model.terminate()
                    return
                if where == GRB.Callback.MIP:
                    bnd = model.cbGet(GRB.Callback.MIP_OBJBND)
                    if math.isfinite(bnd) and abs(bnd) < 1e90:
                        lower = max(lower, bnd)
                if where != GRB.Callback.MIPSOL:
                    return
                try:
                    point = {k: float(round(model.cbGetSolution(v))) for k, v in x.items()}
                    estimate = model.cbGetSolution(zeta)
                    answer = evaluate(point)
                    if answer is None:
                        model.terminate()
                        return
                    if answer.get('cut_valid', False) and answer.get('cg_complete', False):
                        cut_value = answer['cut_intercept'] + sum(c * point[k] for k, c in answer['cut_slope'].items())
                        if estimate < cut_value - 1e-7:
                            model.cbLazy(zeta >= cut_expression(answer))
                            stats['cuts'] += 1
                    else:
                        model.terminate()
                    trace.append({'iteration': len(trace), 'elapsed': time.perf_counter() - start,
                                  'lower_bound': lower, 'upper_bound': best['objective'],
                                  'gap': _relative_gap(best['objective'], lower),
                                  'cg_complete': answer.get('cg_complete', False)})
                    if not answer.get('cg_complete', False):
                        model.terminate()
                except Exception as exc:
                    errors.append(f'{type(exc).__name__}: {exc}')
                    model.terminate()

            if remaining() > 0.01:
                master.Params.TimeLimit = remaining()
                t0 = time.perf_counter()
                subproblem_before = stats['subproblem_wall']
                master.optimize(callback)
                stats['master_wall'] += max(0.0, time.perf_counter() - t0
                                           - (stats['subproblem_wall'] - subproblem_before))
                if master.SolCount > 0:
                    lower = max(lower, float(master.ObjBound))
                if _relative_gap(best['objective'], lower) <= mip_gap and not errors:
                    status = 'OPTIMAL_TOL'
        else:
            raise ValueError(f'Unknown Benders mode: {mode}')
        if errors:
            status = 'CALLBACK_ERROR'
        return {'algorithm': 'benders_' + mode, 'status': status,
                'objective': best['objective'], 'bound': lower,
                'gap': _relative_gap(best['objective'], lower),
                'wall': time.perf_counter() - start, 'design': best['design'],
                'fleet': best.get('fleet'), 'rho': best.get('rho'),
                'theta': best.get('theta'), 'phi': best.get('phi'),
                'primal_residual': best.get('primal_residual'),
                'iterations': len(trace), 'trace': trace, 'stats': stats,
                'service_pool_size': len(pools[0]) if pools else 0,
                'empty_pool_size': len(pools[1]) if pools else 0,
                'errors': errors, 'root_cuts': root_result}
    finally:
        master.dispose()
