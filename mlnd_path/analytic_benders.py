"""Analytic multicut projection of activation-only path allocation.

Facility binaries, aggregate flows, availability and idle cones stay in the
master. At an integer facility pattern every aggregate group independently uses
its cheapest open path. Closed-facility penalties certify each cost cut against
the complete candidate universe. No path LP or incomplete open-only dual is used.
"""
from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass

import gurobipy as gp
from gurobipy import GRB
import numpy as np

from .data import Instance
from .partial_benders import _master, _tkey, _ekey
from .solver import _matrix
from .strengthening import add_disaggregated_gates


@dataclass
class PathGroup:
    key: str
    maximum: float
    service: bool
    ids: np.ndarray
    costs: np.ndarray
    masks: tuple[int, ...]
    cheapest: float
    facility_union: int


class AnalyticPathOracle:
    """Finite-pattern oracle with all-candidate affine cut certificates.

    Each returned optimality cut is
      eta_g >= q_g * T_g - M_g * sum_h penalty_gh * y_h - reserve_g.
    The penalties are nonnegative, vanish on the generating pattern's open
    facilities, and satisfy q_g <= cost_p + sum_{h in p} penalty_gh for every
    candidate path. Missing open paths produce a hitting-set flow cut instead.
    """
    def __init__(self, inst: Instance):
        if not 0 <= inst.rho_max < 1 or inst.kappa < 0 or inst.fleet_cost <= 0:
            raise ValueError('Analytic projection requires bounded nonnegative flows and positive fleet cost')
        self.inst = inst
        self.facilities = ([f'X{l}' for l in range(inst.n_l)]
                           + [f'Y{h}' for h in range(inst.n_h)]
                           + [f'S{k}' for k in range(inst.n_l, inst.n_l + inst.n_k)])
        self.facility_ids = {key: j for j, key in enumerate(self.facilities)}
        self.all_mask = (1 << len(self.facilities)) - 1
        self.cache: dict[int, dict] = {}
        self.groups: list[PathGroup] = []
        self._check_activation_bounds()
        grouped: dict[str, list] = {}
        for p, col in enumerate(inst.service):
            key = _tkey(col['od'], col['l'], col['m'])
            grouped.setdefault(key, []).append((p, inst.service_cost(p), self._mask(col, True)))
        for r, col in enumerate(inst.empty):
            if col['l'] == col['m']:
                raise ValueError('Aggregate empty groups require distinct endpoints')
            key = _ekey(col['l'], col['m'])
            grouped.setdefault(key, []).append((r, inst.empty_cost(r), self._mask(col, False)))
        for key, columns in grouped.items():
            service = key.startswith('T')
            maximum = (inst.demand[int(key[1:].split(':')[0])] * inst.rho_max
                       if service else inst.kappa * inst.rho_max)
            masks = tuple(int(c[2]) for c in columns)
            union = 0
            for mask in masks:
                union |= mask
            costs = np.asarray([c[1] for c in columns], dtype=float)
            if not np.all(np.isfinite(costs)):
                raise ValueError('All column costs must be finite')
            self.groups.append(PathGroup(key, maximum, service,
                np.asarray([c[0] for c in columns], dtype=int), costs, masks,
                float(costs.min()), union))

    def _check_activation_bounds(self):
        """Sufficient data-level certificate that open aggregate gates are redundant."""
        inst = self.inst
        if any(h < 0 for h in inst.demand):
            raise ValueError('Demand rates must be nonnegative')
        n = inst.n_l + inst.n_k
        max_service = np.zeros((inst.n_od, n))
        max_empty = np.zeros((inst.n_l, n))
        for col in inst.service:
            counts = np.asarray(col['swaps'], dtype=float)
            if counts.shape != (n,) or np.any(counts < 0):
                raise ValueError('Invalid service swap incidence')
            max_service[col['od']] = np.maximum(max_service[col['od']], counts)
        for col in inst.empty:
            counts = np.asarray(col['swaps'], dtype=float)
            if counts.shape != (n,) or np.any(counts < 0):
                raise ValueError('Invalid empty swap incidence')
            max_empty[col['l']] = np.maximum(max_empty[col['l']], counts)
        bound = (inst.rho_max * np.asarray(inst.demand) @ max_service
                 + inst.kappa * inst.rho_max * max_empty.sum(axis=0))
        if np.max(bound, initial=0.) > inst.MK + 1e-9:
            raise ValueError('Swap gate can bind at an open station; independent path groups do not apply')
        # A service path uses a locker at most once in the original incidence
        # matrix (it uses a set of i,j), so its rate is bounded by inst.M.
        self.activation_certificate = {
            'service_flow_bound': float(inst.M),
            'empty_flow_bound': float(inst.ME),
            'swap_gate': float(inst.MK),
            'maximum_possible_station_swap_load': float(np.max(bound, initial=0.)),
        }

    def _mask(self, col, service):
        keys = {f"X{col['l']}", f"X{col['m']}"}
        if service:
            keys.update((f"Y{col['i']}", f"Y{col['j']}"))
        keys.update(self.inst.station_key(k) for k, count in enumerate(col['swaps']) if count)
        mask = 0
        for key in keys:
            mask |= 1 << self.facility_ids[key]
        return mask

    def pattern(self, design):
        return sum(1 << j for j, key in enumerate(self.facilities) if design[key] > .5)

    @staticmethod
    def _bits(mask):
        while mask:
            bit = mask & -mask
            yield bit.bit_length() - 1
            mask ^= bit

    def evaluate(self, design, deadline=math.inf):
        pattern = self.pattern(design)
        if pattern in self.cache:
            return self.cache[pattern], True
        closed = self.all_mask ^ pattern
        answers = []
        max_residual = 0.
        for position, group in enumerate(self.groups):
            if position % 16 == 0 and time.monotonic() >= deadline:
                raise TimeoutError('analytic_oracle_deadline')
            blocked = [mask & closed for mask in group.masks]
            available = [j for j, mask in enumerate(blocked) if not mask]
            if not available:
                # This set intersects every path, including zero-flow groups.
                cover = group.facility_union & closed
                if not cover:
                    raise ValueError('Unavailable group has no all-candidate hitting set')
                answers.append(dict(key=group.key, feasible=False,
                    cover=tuple(self.facilities[j] for j in self._bits(cover)),
                    maximum=group.maximum))
                continue
            local_id = min(available, key=lambda j: (group.costs[j], int(group.ids[j])))
            q = float(group.costs[local_id])
            penalty = {}
            # Share each cheaper blocked path's deficiency among its closed
            # facilities, then take each facility's maximum over all paths.
            # Every path therefore receives at least its own full deficiency.
            for cost, mask in zip(group.costs, blocked):
                if mask and cost < q:
                    share = (q - float(cost)) / mask.bit_count()
                    for j in self._bits(mask):
                        penalty[j] = max(penalty.get(j, 0.), share)
            minimum_rc = math.inf
            for cost, mask in zip(group.costs, group.masks):
                rc = float(cost) - q + sum(penalty.get(j, 0.) for j in self._bits(mask))
                minimum_rc = min(minimum_rc, rc)
            correction = group.maximum * max(0., -minimum_rc)
            numerical_reserve = 2e-11 * (1 + group.maximum * (abs(q) + sum(penalty.values())))
            reserve = correction + numerical_reserve
            max_residual = max(max_residual, -minimum_rc)
            answers.append(dict(key=group.key, feasible=True, q=q,
                column=int(group.ids[local_id]), service=group.service,
                penalties={self.facilities[j]: float(value) for j, value in penalty.items()},
                maximum=group.maximum, reserve=float(reserve), minimum_reduced_cost=float(minimum_rc)))
        result = dict(pattern=pattern, groups=answers,
                      maximum_dual_residual=float(max_residual), full_candidate_certificate=True)
        self.cache[pattern] = result
        return result, False


def _cheap_global_bound(inst):
    service = [0.] * inst.n_od
    empty = [0.] * inst.n_l
    for p, col in enumerate(inst.service):
        service[col['od']] = min(service[col['od']], inst.service_cost(p))
    for r, col in enumerate(inst.empty):
        empty[col['l']] = min(empty[col['l']], inst.empty_cost(r))
    fixed = sum(min(0., c) for c in inst.base_cost + inst.hub_cost + inst.swap_cost)
    return float(inst.c0 + fixed + inst.rho_max *
                 (sum(h*c for h,c in zip(inst.demand,service)) + inst.kappa*sum(empty)))


def solve_analytic(inst: Instance, time_limit=30., mip_gap=1e-4, threads=1, output=False):
    """Solve the same integer model by analytic group-wise optimality cuts.

    Returned upper bounds are checked against an independently constructed full
    strengthened matrix, including its original counted swap gates and cones.
    Caching never bypasses callback enforcement: all relevant cached cuts are
    tested again at every MIPSOL point.
    """
    start = time.monotonic()
    deadline = start + max(0., float(time_limit))
    oracle = AnalyticPathOracle(inst)
    checked_instance = copy.copy(inst)
    checked_instance.metadata = dict(inst.metadata)
    checked_matrix = _matrix(add_disaggregated_gates(checked_instance))
    model, x, rho, idle, T, E, zeta = _master(inst, threads, mip_gap)
    model.Params.OutputFlag = int(output)
    variables = dict(T)
    variables.update(E)
    eta = model.addVars([g.key for g in oracle.groups], lb=-GRB.INFINITY, name='group_cost')
    for group in oracle.groups:
        model.addConstr(eta[group.key] >= group.cheapest * variables[group.key],
                        name=f'group_global_bound_{group.key}')
        eta[group.key].Start = 0.
    model.addConstr(zeta >= gp.quicksum(eta.values()), name='sum_group_path_costs')
    model.update()
    stats = dict(subproblem_calls=0, oracle_calls=0, oracle_patterns=0, oracle_cache_hits=0,
                 cuts=0, feasibility_cuts=0, optimality_cuts=0, cg_rounds=0, lp_runtime=0.,
                 subproblem_wall=0., pricing_wall=0., callback_wall=0., master_wall=0.,
                 maximum_cut_dual_residual=0., rejected_primal_candidates=0)
    best = dict(objective=inst.c0, design=inst.zero_design(), rho=[0.]*inst.n_l,
                idle=[0.]*inst.n_l, theta=[0.]*len(inst.service), phi=[0.]*len(inst.empty),
                fleet=0., served=0., primal_residual=0.)
    lower = _cheap_global_bound(inst)
    trace, errors, incomplete = [], [], []
    touched_service, touched_empty = set(), set()

    def validate_point(design, rv, iv, point, answer):
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
                    primal_residual=error),error

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
            added = 0
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
                primal_residual=error))
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
        if errors:
            status = 'CALLBACK_ERROR'
        elif incomplete and time.monotonic()<deadline:
            status = 'INCOMPLETE_RECOURSE'
        if lower-best['objective'] > 1e-5*(1+abs(best['objective'])):
            errors.append('Validated upper bound is below the reported master lower bound')
            status = 'NUMERICAL_BOUND_ERROR'
        return dict(best,algorithm='analytic_benders',status=status,bound=float(lower),
            lower_bound=float(lower),upper_bound=best['objective'],gap=float(gap),
            wall=time.monotonic()-start,stats=stats,trace=trace,errors=errors,
            incomplete_recourse=incomplete,service_pool=len(touched_service),empty_pool=len(touched_empty),
            validated_incumbent=True,activation_certificate=oracle.activation_certificate,
            candidate_service_columns=len(inst.service),candidate_empty_columns=len(inst.empty))
    finally:
        model.dispose()
