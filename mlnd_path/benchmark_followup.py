"""Sequential comparison of full, conic-Benders, and partial LP-Benders solves.

Every method uses the same integer model and strengthened activation rows.
The new methods move a continuous network skeleton to the master; they do not
add demand scenarios or change the random-service expectation coefficients.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import time

import gurobipy as gp
import numpy as np

from .benchmark import SCALES
from .benders import solve_benders
from .instances import make_instance
from .solution_metrics import summarize_solution
from .solver import _matrix, solve_full
from .strengthening import add_disaggregated_gates


def save(path, payload):
    # Full flow arrays remain recoverable without hundreds of thousands of lines.
    path.write_text(json.dumps(payload, ensure_ascii=False, allow_nan=False,
                               separators=(',', ':')) + '\n')


def audit(inst, result):
    if result.get('objective') is None:
        return {}
    matrix = _matrix(inst)
    def vector(key, n):
        a = result.get(key, [])
        return np.asarray(a if len(a) else np.zeros(n), dtype=float)
    theta, phi = vector('theta', len(inst.service)), vector('phi', len(inst.empty))
    rho = vector('rho', inst.n_l)
    idle = (np.asarray(result['idle'], dtype=float) if 'idle' in result
            else rho / (1-rho))
    design = result['design']
    rhs = matrix.constants + np.asarray([
        sum(c * design[k] for k, c in mapping.items()) for mapping in matrix.maps])
    residual = matrix.base @ np.r_[rho, idle] + matrix.service @ theta + matrix.empty @ phi - rhs
    row_residual = max((abs(float(a)) if sense == '=' else max(0., float(a))
                        for a, sense in zip(residual, matrix.senses)), default=0.)
    cone_residual = float(max(0., np.max(rho/(1-rho)-idle)))
    expected = (inst.c0 + inst.fixed_cost(design) + inst.fleet_cost * idle.sum()
                + matrix.cost_service @ theta + matrix.cost_empty @ phi)
    return dict(row_residual=row_residual, cone_residual=cone_residual,
                objective_recomputation_error=abs(float(expected)-result['objective']))


def run(scales, seeds, algorithms, time_limit, mip_gap, output):
    from .partial_benders import solve_partial
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    source = Path(__file__).parent
    config = dict(created_utc=datetime.now(timezone.utc).isoformat(), scales=scales,
                  seeds=seeds, algorithms=algorithms, time_limit_seconds=time_limit,
                  mip_gap=mip_gap, threads=1, python=platform.python_version(),
                  platform=platform.platform(), gurobi=gp.gurobi.version(),
                  source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in source.glob('*.py')},
                  branch=subprocess.check_output(['git', 'branch', '--show-current'], text=True).strip(),
                  scope='Same synthetic one-regime model; analytic service-time expectation; no added scenarios; all methods strengthened; preprocessing separately timed.')
    save(out/'config.json', config)
    warm = gp.Model(); warm.Params.OutputFlag = 0
    v = warm.addVar(); warm.setObjective(v); warm.optimize(); warm.dispose()
    summaries = []
    for si, scale in enumerate(scales):
        for sj, seed in enumerate(seeds):
            name = f'{scale}_seed{seed}'
            tick = time.perf_counter()
            inst = make_instance(*SCALES[scale], seed=seed, name=name)
            generation = time.perf_counter()-tick
            shape = dict(n_l=inst.n_l, n_h=inst.n_h, n_od=inst.n_od, n_k=inst.n_k,
                         service_columns=len(inst.service), empty_columns=len(inst.empty))
            save(out/f'{name}_instance_metadata.json', {**shape, **inst.metadata,
                                                       'generation_wall':generation, 'c0':inst.c0})
            offset = (si+sj) % len(algorithms)
            order = algorithms[offset:]+algorithms[:offset]
            for algorithm in order:
                if hasattr(inst, '_solver_matrix'):
                    delattr(inst, '_solver_matrix')
                tick = time.perf_counter()
                add_disaggregated_gates(inst)
                remaining = max(.001, time_limit-(time.perf_counter()-tick))
                if algorithm == 'full_strong':
                    result = solve_full(inst, time_limit=remaining, mip_gap=mip_gap, threads=1)
                elif algorithm == 'benders_strong':
                    result = solve_benders(inst, time_limit=remaining, mip_gap=mip_gap,
                                           threads=1, mode='lazy')
                elif algorithm == 'analytic':
                    from .analytic_benders import solve_analytic
                    result = solve_analytic(inst, time_limit=remaining, mip_gap=mip_gap,
                                            threads=1)
                else:
                    result = solve_partial(inst, time_limit=remaining, mip_gap=mip_gap,
                                           threads=1, use_cg=algorithm == 'partial_cg')
                result['wall'] = time.perf_counter()-tick
                result.update(instance=name, scale=scale, seed=seed, algorithm=algorithm,
                              generation_wall=generation,
                              end_to_end_wall=generation+result['wall'], **shape)
                result['independent_audit'] = audit(inst, result)
                result['operations'] = summarize_solution(inst, result)
                save(out/f'{name}_{algorithm}.json', result)
                summary = {k:v for k,v in result.items() if k not in {
                    'theta','phi','trace','cg_trace','design','cut_slope','rc_service','rc_empty',
                    'service_pool','empty_pool','cut_records'}}
                summaries.append(summary)
                save(out/'summary.json', summaries)
                print(json.dumps({k:result.get(k) for k in ('instance','algorithm','status',
                     'objective','bound','gap','wall','independent_audit','stats')},
                     ensure_ascii=False), flush=True)
    return summaries


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--scales', nargs='+', choices=SCALES, default=['medium','large'])
    parser.add_argument('--seeds', nargs='+', type=int, default=[11,29])
    parser.add_argument('--algorithms', nargs='+', choices=['full_strong','benders_strong',
                        'partial_full','partial_cg','analytic'], default=['full_strong','benders_strong',
                        'partial_full','partial_cg'])
    parser.add_argument('--time-limit', type=float, default=60.)
    parser.add_argument('--mip-gap', type=float, default=1e-4)
    parser.add_argument('--output', default='mlnd_path/ts_followup_20260914/benchmarks')
    args = parser.parse_args()
    run(args.scales, args.seeds, args.algorithms, args.time_limit, args.mip_gap, args.output)
