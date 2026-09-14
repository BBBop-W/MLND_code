"""Sequential OA follow-up under the same instances and solve budget."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import time

import gurobipy as gp

from mlnd_path.analytic_oa import solve_analytic_oa
from mlnd_path.benchmark import SCALES
from mlnd_path.benchmark_followup import audit, save
from mlnd_path.instances import make_instance
from mlnd_path.solution_metrics import summarize_solution
from mlnd_path.strengthening import add_disaggregated_gates


def run(scales,seeds,limit,output,inject=False):
    directory=Path(output);directory.mkdir(parents=True,exist_ok=True)
    source=Path('mlnd_path')
    algorithm='analytic_oa_inject' if inject else 'analytic_oa'
    config=dict(created_utc=datetime.now(timezone.utc).isoformat(),scales=scales,seeds=seeds,
        algorithms=[algorithm],time_limit_seconds=limit,mip_gap=1e-4,threads=1,
        python=platform.python_version(),platform=platform.platform(),gurobi=gp.gurobi.version(),
        branch=subprocess.check_output(['git','branch','--show-current'],text=True).strip(),
        source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in source.glob('*.py')},
        scope='Same one-regime model and all candidate paths; dynamic tangent OA of continuous availability; exact queue formula for incumbents.')
    config['source_sha256']['ts_followup_20260914/run_oa.py'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    save(directory/'config.json',config)
    warm=gp.Model();warm.Params.OutputFlag=0;v=warm.addVar();warm.setObjective(v);warm.optimize();warm.dispose()
    summaries=[]
    for scale in scales:
        for seed in seeds:
            name=f'{scale}_seed{seed}';start=time.perf_counter()
            inst=make_instance(*SCALES[scale],seed=seed,name=name)
            generation=time.perf_counter()-start
            shape=dict(n_l=inst.n_l,n_h=inst.n_h,n_od=inst.n_od,n_k=inst.n_k,
                       service_columns=len(inst.service),empty_columns=len(inst.empty))
            save(directory/f'{name}_instance_metadata.json',{**shape,**inst.metadata,'generation_wall':generation})
            start=time.perf_counter();add_disaggregated_gates(inst)
            result=solve_analytic_oa(inst,time_limit=max(.001,limit-(time.perf_counter()-start)),
                                     mip_gap=1e-4,threads=1,inject_incumbent=inject)
            result['wall']=time.perf_counter()-start
            result.update(algorithm=algorithm,instance=name,scale=scale,seed=seed,
                          generation_wall=generation,end_to_end_wall=generation+result['wall'],**shape)
            result['independent_audit']=audit(inst,result)
            result['operations']=summarize_solution(inst,result)
            save(directory/f'{name}_{algorithm}.json',result)
            summaries.append({k:v for k,v in result.items() if k not in
                {'theta','phi','design','trace','service_pool','empty_pool'}})
            save(directory/'summary.json',summaries)
            print(json.dumps({k:result.get(k) for k in ['instance','algorithm','status','objective',
                'bound','gap','wall','independent_audit','stats']},ensure_ascii=False),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--scales',nargs='+',default=['medium','large'],choices=SCALES)
    p.add_argument('--seeds',nargs='+',type=int,default=[11,29]);p.add_argument('--time-limit',type=float,default=30)
    p.add_argument('--output',default='mlnd_path/ts_followup_20260914/benchmarks_oa')
    p.add_argument('--inject',action='store_true');a=p.parse_args()
    run(a.scales,a.seeds,a.time_limit,a.output,a.inject)
