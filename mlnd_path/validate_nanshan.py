"""Recompute saved strategic/operating solutions and cross-method bounds."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from .nanshan_oracle import load_nanshan,price_design


def validate_file(path):
    result=json.loads(path.read_text());h=int(result['name'].split('H')[-1])
    data=load_nanshan(h);L=data['n_l'];H=data['n_h']
    with np.load(path.with_suffix('.npz'),allow_pickle=False) as saved:
        f=saved['facility'];B=saved['assignment'];rho=saved['rho'];idle=saved['idle'];empty=saved['empty']
    q=price_design(data,f)
    flow=data['demand'][:,None,None]*B*rho[None,:,None]
    matrix=flow.sum(axis=0)
    active=flow>1e-8;used_empty=empty>1e-8
    route_ok=bool(np.all(np.isfinite(q['cost'][active])) and np.all(np.isfinite(q['empty_cost'][used_empty])))
    fixed=np.r_[[data['base_cost']]*L,[data['hub_cost']]*H,[data['swap_cost']]*data['n_k']]
    operating=float(np.sum(np.where(np.isfinite(q['cost']),q['cost'],0.)*flow)
                    +np.sum(np.where(np.isfinite(q['empty_cost']),q['empty_cost'],0.)*empty))
    objective=float(data['c0']+fixed@f+data['fleet_cost']*idle.sum()+operating)
    fleet=float(idle.sum()+np.sum(q['time']*flow)+np.sum(q['empty_time']*empty))
    balance=matrix.sum(axis=1)-matrix.sum(axis=0)+empty.sum(axis=1)-empty.sum(axis=0)
    residual=max(float(np.max(np.abs(balance),initial=0.)),
                 float(np.max(empty.sum(axis=1)-data['kappa']*rho,initial=0.)),
                 float(np.max(rho-data['rho_max']*f[:L],initial=0.)),
                 float(np.max(rho/(1-rho)-idle,initial=0.)),
                 float(np.max(B.sum(axis=(1,2))-1,initial=0.)),
                 float(np.max(B.sum(axis=2)-f[:L],initial=0.)),
                 float(np.max(B.sum(axis=1)-f[:L],initial=0.)),
                 float(np.max(np.abs(B-np.rint(B)),initial=0.)),
                 float(np.max(np.abs(f-np.rint(f)),initial=0.)),
                 float(max(0.,-np.min(np.r_[rho,idle,empty.ravel(),B.ravel(),f]))))
    mismatched=[]
    for name,sha in result['source_sha256'].items():
        if hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()!=sha:mismatched.append(name)
    passed=(route_ok and residual<=1e-5 and abs(objective-result['objective'])<=1e-5
            and abs(fleet-result['fleet'])<=1e-5 and result['bound']<=objective+1e-5
            and result['stats']['master_continuous']==1 and not result['errors'] and not mismatched)
    return dict(file=str(path),passed=bool(passed),route_exists=route_ok,residual=residual,
        objective=objective,objective_error=abs(objective-result['objective']),fleet=fleet,
        fleet_error=abs(fleet-result['fleet']),source_mismatch=mismatched,
        served=float(flow.sum()),bound=result['bound'])


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('directories',nargs='+')
    parser.add_argument('--output',required=True);args=parser.parse_args()
    files=[p for d in args.directories for p in sorted(Path(d).glob('H*.json'))]
    rows=[validate_file(p) for p in files]
    groups={}
    for p in files:
        r=json.loads(p.read_text());groups.setdefault(r['name'],[]).append(r)
    cross={name:dict(best_upper=min(r['objective'] for r in runs),best_lower=max(r['bound'] for r in runs)) for name,runs in groups.items()}
    for v in cross.values():v['passed']=v['best_lower']<=v['best_upper']+1e-5
    report=dict(passed=all(r['passed'] for r in rows) and all(r['passed'] for r in cross.values()),
                runs=rows,cross_method_bounds=cross)
    Path(args.output).write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(dict(passed=report['passed'],runs=len(rows),
        max_residual=max(r['residual'] for r in rows),max_objective_error=max(r['objective_error'] for r in rows)),indent=2))
