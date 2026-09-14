"""Serial, matched full-demand benchmarks of the pure-integer master."""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
import platform
import time
from pathlib import Path

import gurobipy as gp
import numpy as np
from .nanshan_integer import solve
from .nanshan_oracle import load_nanshan,price_design


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--levels',default='11,24,43,68,87,120')
    p.add_argument('--seconds',type=float,default=60.)
    p.add_argument('--gap',type=float,default=.001)
    p.add_argument('--output',default='mlnd_path/nanshan_integer_20260914/formal_60s')
    args=p.parse_args();directory=Path(args.output);directory.mkdir(parents=True,exist_ok=True)
    # Compile the four numerical kernels before matched wall-clock budgets.
    compile_start=time.perf_counter();price_design(load_nanshan(11))
    manifest=dict(gurobi=gp.gurobi.version(),python=platform.python_version(),platform=platform.platform(),
        threads=1,seed=43,seconds=args.seconds,mip_gap=args.gap,
        levels=list(map(int,args.levels.split(','))),jit_warmup_seconds=time.perf_counter()-compile_start,
        scope='all canonical ODs and every candidate facility; no top-k corridor screening',
        master='facility and OD/base-pair binaries plus exactly one cost epigraph',
        timing='solve wall includes all-open pricing, master build, warm starts, callback pricing, recourse and finalization; data loading and compiler warmup separate',
        runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),runs=[])
    for index,h in enumerate(manifest['levels']):
        load_start=time.perf_counter();data=load_nanshan(h);load_wall=time.perf_counter()-load_start
        for pooled in ([False,True] if index%2==0 else [True,False]):
            name=f'H{h:03d}_'+('cuts' if pooled else 'base')
            path=directory/(name+'.json')
            result=solve(data,time_limit=args.seconds,mip_gap=args.gap,pooled=pooled,log=directory/(name+'.log'))
            solution=result.pop('_solution')
            np.savez_compressed(directory/(name+'.npz'),**solution)
            result['data_loading_wall']=load_wall
            path.write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
            row={k:result[k] for k in ['name','method','wall','objective','bound','gap','errors','stats']}
            row['file']=path.name;manifest['runs'].append(row)
            (directory/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n')
            print(json.dumps(row,ensure_ascii=False),flush=True)
            del result,solution;gc.collect()


if __name__=='__main__':main()
