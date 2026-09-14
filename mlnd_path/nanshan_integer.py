"""Integer strategic master and small continuous mission recourse.

The master contains only facility/OD-assignment binaries and one cost epigraph.
No availability, idle, or operating flow variable is moved into the master.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import gurobipy as gp
import numpy as np
from gurobipy import GRB

from .data import Instance
from .integer_master_cuts import pooled_idle_slopes
from .nanshan_oracle import load_nanshan, price_design


def as_instance(data):
    L,H,K=data['n_l'],data['n_h'],data['n_k']
    return Instance(data['name'],L,H,K,data['demand'].tolist(),data['ground'].tolist(),
                    [data['base_cost']]*L,[data['hub_cost']]*H,[data['swap_cost']]*K,
                    rho_max=data['rho_max'],kappa=data['kappa'],fleet_cost=data['fleet_cost'])


def sparse_design(facility,assignment,L,H):
    point=defaultdict(float)
    for l in range(L):point[f'X{l}']=float(facility[l])
    for i in range(H):point[f'Y{i}']=float(facility[L+i])
    for k in range(len(facility)-L-H):point[f'S{k+L}']=float(facility[L+H+k])
    for od,l,m in zip(*np.where(assignment>.5)):point[f'B{od}:{l}:{m}']=1.
    return point


def initial_coefficients(data,all_open):
    q=all_open['cost'];h=data['demand'][:,None,None]
    beta=data['fleet_cost']*h/data['demand'].sum()
    a=h*np.where(np.isfinite(q),q,0.)
    r=np.where(a < -beta,np.minimum(data['rho_max'],1-np.sqrt(beta/np.maximum(-a,1e-300))),0.)
    return a*r+beta*r/(1-r)


def build_master(data,all_open,time_limit,threads=1,mip_gap=.01,pooled=True,log=None):
    L,H,K,D=data['n_l'],data['n_h'],data['n_k'],data['n_od']
    model=gp.Model('MLND_Path_Nanshan_integer_master')
    model.Params.OutputFlag=0
    model.Params.Threads=threads;model.Params.Seed=43
    model.Params.TimeLimit=max(.01,time_limit);model.Params.MIPGap=mip_gap
    model.Params.MIPGapAbs=1e-5
    if log:model.Params.LogFile=str(log)
    facility=model.addMVar(L+H+K,vtype=GRB.BINARY,name='facility')
    B=model.addMVar((D,L,L),vtype=GRB.BINARY,name='B')
    zeta=model.addVar(lb=0,name='recourse_cost')
    model.addConstr(B.sum(axis=(1,2))<=1,name='one_base_pair')
    # Aggregated endpoint links strengthen individual B<=X while preserving
    # every integer assignment and reduce the number of master rows.
    model.addConstr(B.sum(axis=2)<=facility[:L],name='source_open')
    model.addConstr(B.sum(axis=1)<=facility[:L],name='return_open')
    old=initial_coefficients(data,all_open)
    model.addConstr(zeta>=data['c0']+(old*B).sum(),name='allocated_idle_lower_bound')
    weight=data['demand'][:,None,None]*np.maximum(0,-all_open['cost'])
    if pooled:
        slope=pooled_idle_slopes(weight,fleet_cost=data['fleet_cost'],rho_max=data['rho_max'])
        model.addConstr(zeta>=data['c0']+(slope*B).sum(),name='pooled_idle_initial')
        # A complete service mission needs two distinct lockers. With at most
        # one open locker every OD uses ground fallback; otherwise a global
        # maximum-saving bound makes these cost-cover rows valid.
        maximum_saving=data['rho_max']*float(weight.max(axis=(1,2)).sum())
        hub_sum=facility[L:L+H].sum()
        for h in range(H):
            model.addConstr(zeta>=data['c0']-maximum_saving*(hub_sum-facility[L+h]),
                            name=f'two_locker_cost_cover_{h}')
    fixed=np.r_[[data['base_cost']]*L,[data['hub_cost']]*H,[data['swap_cost']]*K]
    model.setObjective(fixed@facility+zeta)
    model.Params.LazyConstraints=1
    model.update()
    return model,facility,B,zeta,fixed,weight


def solve(data,time_limit=60.,threads=1,mip_gap=.001,pooled=True,log=None):
    from .integer_recourse import solve_integer_recourse
    start=time.perf_counter();deadline=start+time_limit
    inst=as_instance(data);L,H,K,D=data['n_l'],data['n_h'],data['n_k'],data['n_od']
    all_open=price_design(data)
    oracle_cache={tuple([1]*(L+H+K)):all_open}
    oracle_seconds=all_open['wall'];recourse_seconds=0.
    master,fac,B,zeta,fixed,weights=build_master(data,all_open,time_limit,threads,mip_gap,pooled,log)
    built=time.perf_counter()-start
    stats=dict(oracle_calls=1,oracle_cache_hits=0,recourse_calls=0,benders_cuts=0,pooled_cuts=int(pooled),
               master_binaries=int(master.NumBinVars),master_continuous=int(master.NumVars-master.NumBinVars),
               master_rows=int(master.NumConstrs),cost_cover_cuts=H if pooled else 0,incumbent_injections=0,
               maximum_recourse_duality_gap=0.,incomplete_recourse_certificates=0)
    best=dict(objective=data['c0'],facility=np.zeros(L+H+K),assignment=np.zeros((D,L,L)),
              rho=[0.]*L,idle=[0.]*L,empty=np.zeros((L,L)),fleet=0.,served=0.,recourse=data['c0'],primal_residual=0.)
    lower=0.;trace=[];errors=[]

    def evaluate(facility,assignment):
        nonlocal oracle_seconds,recourse_seconds,best
        key=tuple(np.rint(facility).astype(int))
        if key in oracle_cache:
            oracle=oracle_cache[key];stats['oracle_cache_hits']+=1
        else:
            oracle=price_design(data,facility);stats['oracle_calls']+=1;oracle_seconds+=oracle['wall']
            if len(oracle_cache)>12:
                # Preserve all-open costs used to certify every ungenerated route.
                for oldkey in list(oracle_cache):
                    if oldkey!=tuple([1]*(L+H+K)):
                        del oracle_cache[oldkey];break
            oracle_cache[key]=oracle
        t=time.perf_counter()
        answer=solve_integer_recourse(inst,design_array=assignment,facility_array=facility,
                  oracle_result=oracle,universal_cost_lower=all_open['cost'],
                  universal_empty_lower=all_open['empty_cost'],
                  time_limit=max(.02,deadline-time.perf_counter()),threads=threads)
        recourse_seconds+=time.perf_counter()-t;stats['recourse_calls']+=1
        stats['maximum_recourse_duality_gap']=max(stats['maximum_recourse_duality_gap'],abs(answer.get('duality_gap',0.)))
        stats['incomplete_recourse_certificates']+=int(not answer.get('cg_complete',False))
        if answer.get('objective') is not None and answer.get('primal_residual',math.inf)<=1e-5:
            obj=float(answer['objective']+fixed@facility)
            if obj<best['objective']:
                empty=np.zeros((L,L))
                for item in answer.get('empty_flows',[]):empty[item['l'],item['m']]=item['rate']
                best=dict(objective=obj,facility=facility.copy(),assignment=assignment.copy(),
                    rho=answer.get('rho'),idle=answer.get('idle'),empty=empty,
                    fleet=answer.get('fleet'),served=answer.get('served'),
                    recourse=answer['objective'],primal_residual=answer['primal_residual'])
        return answer

    def cut_arrays(answer):
        if 'cut_B' in answer:return answer['cut_intercept'],np.asarray(answer['cut_B']),np.asarray(answer['cut_facility'])
        bs=np.zeros((D,L,L));fs=np.zeros(L+H+K)
        for key,c in answer['cut_slope'].items():
            if key[0]=='B':o,l,m=map(int,key[1:].split(':'));bs[o,l,m]=c
            elif key[0]=='X':fs[int(key[1:])]=c
            elif key[0]=='Y':fs[L+int(key[1:])]=c
            else:fs[H+int(key[1:])]=c
        return answer['cut_intercept'],bs,fs

    # Same-base missions form a balanced warm start at every availability.
    starts=[np.ones(L+H+K)]
    if K:starts.append(np.r_[np.ones(L+H),np.zeros(K)])
    for index,initial_fac in enumerate(starts):
        if time.perf_counter()>=deadline:break
        key=tuple(initial_fac.astype(int))
        if key not in oracle_cache:
            oracle_cache[key]=price_design(data,initial_fac)
            oracle_seconds+=oracle_cache[key]['wall'];stats['oracle_calls']+=1
        initial=np.zeros((D,L,L));diagonal=oracle_cache[key]['cost'][:,np.arange(L),np.arange(L)]
        chosen=np.argmin(diagonal,axis=1)
        for od,l in enumerate(chosen):
            if diagonal[od,l]<0:initial[od,l,l]=1
        ans=evaluate(initial_fac,initial)
        if ans.get('cut_valid'):
            c,bs,fs=cut_arrays(ans);master.addConstr(zeta>=c+(bs*B).sum()+fs@fac,name=f'warm_benders_{index}');stats['benders_cuts']+=1
    fac.Start=best['facility'];B.Start=best['assignment'];zeta.Start=best['recourse']

    def callback(model,where):
        nonlocal lower
        if time.perf_counter()>=deadline:model.terminate();return
        if where==GRB.Callback.MIP:
            bound=model.cbGet(GRB.Callback.MIP_OBJBND)
            if abs(bound)<1e90:lower=max(lower,float(bound))
        if where!=GRB.Callback.MIPSOL:return
        try:
            f=np.rint(model.cbGetSolution(fac));b=np.rint(model.cbGetSolution(B));estimate=model.cbGetSolution(zeta)
            if pooled:
                slope=pooled_idle_slopes(weights,b,fleet_cost=data['fleet_cost'],rho_max=data['rho_max'])
                val=data['c0']+float(np.sum(slope*b))
                if estimate<val-1e-6:
                    model.cbLazy(zeta>=data['c0']+(slope*B).sum());stats['pooled_cuts']+=1
            answer=evaluate(f,b)
            if answer.get('cut_valid'):
                c,bs,fs=cut_arrays(answer)
                val=c+float(np.sum(bs*b)+fs@f)
                if estimate<val-1e-6:
                    model.cbLazy(zeta>=c+(bs*B).sum()+fs@fac);stats['benders_cuts']+=1
                if answer.get('primal_residual',math.inf)<=1e-5:
                    model.cbSetSolution(fac,f);model.cbSetSolution(B,b)
                    model.cbSetSolution(zeta,answer['objective']);model.cbUseSolution();stats['incumbent_injections']+=1
            else:
                errors.append({'kind':'uncertified_recourse','status':answer.get('status')});model.terminate()
            trace.append(dict(elapsed=time.perf_counter()-start,upper=best['objective'],lower=lower,
                              recourse_seconds=recourse_seconds,oracle_seconds=oracle_seconds,
                              benders_cuts=stats['benders_cuts'],pooled_cuts=stats['pooled_cuts']))
        except Exception as e:
            errors.append(repr(e));model.terminate()
    try:
        left=deadline-time.perf_counter()
        if left>0:
            master.Params.TimeLimit=left;master.optimize(callback)
            if abs(master.ObjBound)<1e90:lower=max(lower,float(master.ObjBound))
        else:errors.append('budget exhausted before master optimization')
        status=int(master.Status)
        runtime=float(master.Runtime);nodes=float(master.NodeCount)
    finally:master.dispose()
    elapsed=time.perf_counter()-start
    result=dict(name=data['name'],method='integer_benders_pooled' if pooled else 'integer_benders',
        status=status,wall=elapsed,budget=time_limit,build_wall=built,master_runtime=runtime,nodes=nodes,
        objective=best['objective'],bound=lower,gap=max(0,best['objective']-lower)/max(1,abs(best['objective'])),
        oracle_wall=oracle_seconds,recourse_wall=recourse_seconds,stats=stats,trace=trace,errors=errors,
        rho=best['rho'],fleet=best['fleet'],served=best['served'],primal_residual=best['primal_residual'],
        opened={'L':int(best['facility'][:L].sum()),'H':int(best['facility'][L:L+H].sum()),'E':int(best['facility'][L+H:].sum())},
        n_od=D,demand=float(data['demand'].sum()),all_ground_cost=data['c0'],
        basic_path_combinations=all_open['implicit_basic_paths'],explicit_path_enumeration=False,
        provenance=data['provenance'],threads=threads,mip_gap_target=mip_gap,
        source_sha256={name:hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()
                       for name in ['data.py','integer_master_cuts.py','integer_recourse.py',
                                    'nanshan_integer.py','nanshan_oracle.py','nanshan_timing.py']})
    result['_solution']=dict(facility=best['facility'],assignment=best['assignment'],rho=np.asarray(best['rho']),
                             idle=np.asarray(best['idle']),empty=best['empty'])
    return result


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--hubs',type=int,default=11);parser.add_argument('--seconds',type=float,default=60)
    parser.add_argument('--gap',type=float,default=.001)
    parser.add_argument('--no-pooled',action='store_true');parser.add_argument('--output',required=True)
    args=parser.parse_args();out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    data=load_nanshan(args.hubs);result=solve(data,time_limit=args.seconds,mip_gap=args.gap,pooled=not args.no_pooled,log=out.with_suffix('.log'))
    sol=result.pop('_solution');np.savez_compressed(out.with_suffix('.npz'),**sol)
    out.write_text(json.dumps(result,indent=2,ensure_ascii=False))
    print(json.dumps({k:v for k,v in result.items() if k not in ['trace','provenance','source_sha256']},ensure_ascii=False),flush=True)
