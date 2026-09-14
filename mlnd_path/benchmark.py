"""Reproducible, sequential, single-thread multi-scale algorithm comparison."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import time

import gurobipy as gp

from .instances import make_instance
from .solver import solve_full
from .benders import solve_benders


SCALES = {
    'tiny': (2, 3, 2, 1),
    'small': (2, 4, 4, 1),
    'medium': (3, 5, 8, 1),
    'large': (3, 6, 12, 2),
    'stress': (4, 8, 24, 2),
}


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))


def run(scales, seeds, algorithms, time_limit, mip_gap, output, save_instances=False):
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
              for p in Path(__file__).parent.glob('*.py')}
    config = dict(created_utc=datetime.now(timezone.utc).isoformat(), scales=scales, seeds=seeds,
                  algorithms=algorithms, time_limit_seconds=time_limit, mip_gap=mip_gap,
                  threads=1, platform=platform.platform(), python=platform.python_version(),
                  gurobi=gp.gurobi.version(), source_sha256=hashes,
                  branch=subprocess.check_output(['git','branch','--show-current'],text=True).strip(),
                  scope='Synthetic geometric instances; exact integer-design-equivalent route preprocessing; no fleet cap or swap congestion; analytic random service-time expectations; no SAA.')
    write_json(out / 'config.json', config)
    # Exclude one-time license and runtime initialization from every algorithm.
    warm = gp.Model();warm.Params.OutputFlag=0
    x=warm.addVar();warm.setObjective(x);warm.optimize();warm.dispose()
    results = []
    for scale_index, scale in enumerate(scales):
        for seed_index, seed in enumerate(seeds):
            name = f'{scale}_seed{seed}'
            t0 = time.perf_counter()
            inst = make_instance(*SCALES[scale], seed=seed, name=name)
            generation_wall = time.perf_counter() - t0
            shape = dict(n_l=inst.n_l,n_h=inst.n_h,n_od=inst.n_od,n_k=inst.n_k,
                         service_columns=len(inst.service),empty_columns=len(inst.empty))
            write_json(out / f'{name}_instance_metadata.json', {**shape, **inst.metadata,
                       'c0':inst.c0,'generation_wall':generation_wall})
            if save_instances:
                write_json(out / f'{name}_instance.json', asdict(inst))
            # Rotate order to reduce systematic warm-cache advantages.
            offset = (scale_index + seed_index) % len(algorithms)
            order = algorithms[offset:] + algorithms[:offset]
            print(json.dumps({'instance':name,**shape,'generation_wall':round(generation_wall,3)}),flush=True)
            for algorithm in order:
                if hasattr(inst, '_solver_matrix'):
                    delattr(inst, '_solver_matrix')
                started = time.perf_counter()
                try:
                    if algorithm in ('full_strong', 'benders_strong', 'benders_strong_pareto', 'benders_root'):
                        from .strengthening import add_disaggregated_gates
                        add_disaggregated_gates(inst)
                    if algorithm in ('full', 'full_strong'):
                        result = solve_full(inst,time_limit=max(.001, time_limit-(time.perf_counter()-started)),mip_gap=mip_gap,threads=1)
                    else:
                        mode = {'benders_strong':'lazy', 'benders_strong_pareto':'pareto', 'benders_root':'root'}.get(
                            algorithm, algorithm.removeprefix('benders_'))
                        result = solve_benders(inst,time_limit=max(.001, time_limit-(time.perf_counter()-started)),mip_gap=mip_gap,threads=1,
                                               mode=mode)
                except Exception as exc:
                    result = dict(status='ERROR',error=f'{type(exc).__name__}: {exc}',
                                  wall=time.perf_counter()-started)
                result['wall'] = time.perf_counter() - started
                if result.get('objective') is not None and 'design' in result:
                    from .solution_metrics import summarize_solution
                    result['operations'] = summarize_solution(inst, result)
                result.update(instance=name,scale=scale,seed=seed,algorithm=algorithm,
                              generation_wall=generation_wall,**shape)
                result['end_to_end_wall'] = generation_wall + result['wall']
                write_json(out / f'{name}_{algorithm}.json',result)
                brief = {k:v for k,v in result.items()
                         if k not in ['trace','cg_trace','theta','phi','design','rc_service','rc_empty',
                                      'cut_slope','service_pool','empty_pool']}
                results.append(brief)
                write_json(out / 'summary.json',results)
                print(json.dumps({k:brief.get(k) for k in ['instance','algorithm','status','objective',
                                  'bound','gap','wall','iterations','error']}),flush=True)
    return results


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--scales',nargs='+',choices=SCALES,default=['tiny','small','medium','large'])
    parser.add_argument('--seeds',nargs='+',type=int,default=[11,29])
    parser.add_argument('--algorithms',nargs='+',choices=['full','full_strong','benders_loop','benders_lazy',
                        'benders_pareto','benders_strong','benders_strong_pareto','benders_root'],
                        default=['full','benders_loop','benders_lazy'])
    parser.add_argument('--time-limit',type=float,default=30)
    parser.add_argument('--mip-gap',type=float,default=1e-4)
    parser.add_argument('--output',default='mlnd_path/benchmarks/20260914')
    parser.add_argument('--save-instances',action='store_true')
    args=parser.parse_args()
    run(args.scales,args.seeds,args.algorithms,args.time_limit,args.mip_gap,args.output,args.save_instances)


if __name__ == '__main__':
    main()
