"""Small fixed-integer-design recourse and implicit universal-column cuts.

The master contains only strategic binaries and its value epigraph. This module
keeps rho, idle inventory, and all operating flows inside the subproblem. An
exact open-route oracle supplies group minimum adjusted costs; universal route
cost lower bounds certify every closed route without enumerating those routes.
"""
from __future__ import annotations

import math
import time

import gurobipy as gp
from gurobipy import GRB
import numpy as np


class FinitePathOracle:
    """Adapter for an existing finite Instance; implicit oracles use the same dict."""
    def __init__(self, inst): self.inst=inst

    def evaluate(self, design):
        a=self.inst;shape=(a.n_od,a.n_l,a.n_l)
        cost=np.full(shape,np.inf);duration=np.zeros(shape);index=np.full(shape,-1,dtype=int)
        ecost=np.full((a.n_l,a.n_l),np.inf);etime=np.zeros_like(ecost);eindex=np.full(ecost.shape,-1,dtype=int)
        def available(c,service):
            if design[f"X{c['l']}"]<.5 or design[f"X{c['m']}"]<.5:return False
            if service and (design[f"Y{c['i']}"]<.5 or design[f"Y{c['j']}"]<.5):return False
            return all(not n or design[a.station_key(k)]>.5 for k,n in enumerate(c['swaps']))
        for p,c in enumerate(a.service):
            if available(c,True):
                key=(c['od'],c['l'],c['m']);value=a.service_cost(p)
                if value<cost[key]:cost[key]=value;duration[key]=c['time'];index[key]=p
        for r,c in enumerate(a.empty):
            if available(c,False):
                key=(c['l'],c['m']);value=a.empty_cost(r)
                if value<ecost[key]:ecost[key]=value;etime[key]=c['time'];eindex[key]=r
        return dict(best_cost=cost,best_time=duration,best_index=index,
                    empty_cost=ecost,empty_time=etime,empty_index=eindex,
                    universal_cost_lower=-np.asarray(a.ground_cost),
                    universal_empty_lower=0.,costs_are_adjusted=True,
                    oracle_exact=True)


def _settings(model,time_limit,threads):
    model.Params.OutputFlag=0;model.Params.Threads=threads;model.Params.NonConvex=0
    model.Params.TimeLimit=max(.001,float(time_limit))
    model.Params.FeasibilityTol=1e-8;model.Params.OptimalityTol=1e-8
    model.Params.BarQCPConvTol=1e-9


def _idle_conjugate(coefficient,cost,rhomax):
    if coefficient>=-cost:return 0.
    r=min(rhomax,1-math.sqrt(cost/(-coefficient)))
    return cost*r/(1-r)+coefficient*r


def solve_integer_recourse(inst,design=None,oracle_result=None,oracle=None,time_limit=30.,
                           threads=1,certificate_tol=2e-6,design_array=None,
                           facility_array=None,universal_cost_lower=None,
                           universal_empty_lower=None,return_sparse_cut=None):
    """Return Q(design), a binary-only cut, and sparse operating flows.

    Oracle schema (all costs already include fleet-time cost):
      best_cost[od,l,m] = min(cost-ground_cost[od]+fleet_cost*time),
      best_time[od,l,m], empty_cost[l,m], empty_time[l,m].
    Infeasible groups have +inf cost. universal_cost_lower[od] bounds EVERY
    possible service route, including closed facilities; default is -ground.
    The default empty-route universal lower bound is zero. The supplied oracle
    must solve open-route minimization exactly for its declared path universe.
    """
    start=time.monotonic();deadline=start+max(0.,time_limit)
    if inst.fleet_cost<=0 or not 0<=inst.rho_max<1:
        raise ValueError('Positive fleet cost and rho_max in [0,1) are required')
    L=inst.n_l;O=inst.n_od
    facility_keys=([f'X{l}' for l in range(L)]+[f'Y{h}' for h in range(inst.n_h)]
                   +[f'S{k}' for k in range(L,L+inst.n_k)])
    if facility_array is not None:
        facility_array=np.asarray(facility_array,dtype=float)
        design=dict(design or {})
        design.update({key:float(value) for key,value in zip(facility_keys,facility_array)})
    elif design is not None:
        facility_array=np.asarray([design.get(key,0.) for key in facility_keys])
    else:raise ValueError('Supply strategic design or facility_array')
    oracle_start=time.monotonic()
    if oracle_result is None:
        oracle_result=(oracle or FinitePathOracle(inst)).evaluate(design)
    if oracle_result.get('costs_are_adjusted',True) is not True:
        raise ValueError('Oracle best_cost/empty_cost must be adjusted column costs')
    if oracle_result.get('oracle_exact',True) is not True or oracle_result.get('complete',True) is not True:
        raise ValueError('An exact open-route oracle is required for complete recourse cuts')
    q=np.asarray(oracle_result.get('best_cost',oracle_result.get('cost')),dtype=float)
    qt=np.asarray(oracle_result.get('best_time',oracle_result.get('time')),dtype=float)
    qe=np.asarray(oracle_result['empty_cost'],dtype=float)
    qet=np.asarray(oracle_result['empty_time'],dtype=float)
    raw_lower=universal_cost_lower if universal_cost_lower is not None else oracle_result.get('universal_cost_lower',-np.asarray(inst.ground_cost))
    lower_cost=np.asarray(raw_lower,dtype=float)
    if lower_cost.shape==(O,):lower_cost=np.broadcast_to(lower_cost[:,None,None],(O,L,L))
    elif lower_cost.shape!=(O,L,L):raise ValueError('Universal service lower bound must have shape OD or OD,L,L')
    lower_cost=np.where(np.isfinite(lower_cost),lower_cost,-np.asarray(inst.ground_cost)[:,None,None])
    raw_empty=universal_empty_lower if universal_empty_lower is not None else oracle_result.get('universal_empty_lower',0.)
    empty_lower=np.broadcast_to(np.asarray(raw_empty,dtype=float),(L,L))
    empty_lower=np.where(np.isfinite(empty_lower),empty_lower,0.)
    if q.shape!=(O,L,L) or qt.shape!=q.shape or qe.shape!=(L,L) or qet.shape!=qe.shape:
        raise ValueError('Unexpected oracle array dimensions')
    if np.isnan(q).any() or np.isnan(qe).any() or not np.isfinite(lower_cost).all():
        raise ValueError('Oracle costs must be finite or positive infinity')
    if np.any(q<lower_cost-1e-7):raise ValueError('Open service cost violates its universal route lower bound')
    if np.any(qe<empty_lower-1e-7):
        raise ValueError('Open empty cost violates its universal route lower bound')
    assignment={}
    if design_array is not None:
        barray=np.asarray(design_array)
        if barray.shape!=(O,L,L):raise ValueError('design_array must have shape OD,L,L')
        selected=np.argwhere(barray>.5)
    else:
        selected=[tuple(map(int,key[1:].split(':'))) for key,value in design.items() if key.startswith('B') and value>.5]
    for o,l,m in selected:
            o,l,m=int(o),int(l),int(m)
            if o in assignment:raise ValueError('Each OD can select at most one base pair')
            if design[f'X{l}']<.5 or design[f'X{m}']<.5:raise ValueError('Assigned endpoints must be open')
            assignment[o]=(l,m)
    D=np.zeros((L,L));C=np.zeros(L);caps=np.asarray([inst.rho_max*round(design[f'X{l}']) for l in range(L)])
    chosen_cost={};unreachable={}
    for o,(l,m) in assignment.items():
        h=inst.demand[o];D[l,m]+=h
        chosen_cost[o]=float(q[o,l,m]) if np.isfinite(q[o,l,m]) else 0.
        C[l]+=h*chosen_cost[o]
        if h>0 and not np.isfinite(q[o,l,m]):caps[l]=0.;unreachable.setdefault(l,[]).append(o)
    arcs=[(l,m) for l in range(L) for m in range(L) if l!=m and np.isfinite(qe[l,m])]
    oracle_wall=time.monotonic()-oracle_start
    answer=dict(objective=inst.c0,bound=0.,dual_value=None,cut_valid=False,cg_complete=False,
                status='time_limit',rho=[0.]*L,idle=[0.]*L,fleet=0.,served=0.,
                service_flows=[],empty_flows=[],primal_residual=0.,
                oracle_wall=oracle_wall,unreachable_assigned_ods=sum(map(len,unreachable.values())))
    if time.monotonic()>=deadline:answer['wall']=time.monotonic()-start;return answer
    primal=gp.Model('integer_design_small_recourse');_settings(primal,deadline-time.monotonic(),threads)
    rho=primal.addVars(L,lb=0,name='rho');idle=primal.addVars(L,lb=0,name='idle')
    flow=primal.addVars(arcs,lb=0,name='empty')
    for l in range(L):
        primal.addConstr(rho[l]<=caps[l])
        primal.addConstr(float(D[l].sum())*rho[l]-gp.quicksum(float(D[m,l])*rho[m] for m in range(L))
                         +gp.quicksum(flow[l,m] for m in range(L) if (l,m) in flow)
                         -gp.quicksum(flow[m,l] for m in range(L) if (m,l) in flow)==0)
        primal.addConstr(gp.quicksum(flow[l,m] for m in range(L) if (l,m) in flow)<=inst.kappa*rho[l])
        up=primal.addVar(lb=0);lo=primal.addVar(lb=0);one=primal.addVar(lb=1,ub=1)
        primal.addConstr(up-idle[l]==1);primal.addConstr(lo+rho[l]==1)
        primal.addQConstr(one*one<=up*lo)
    primal.setObjective(inst.c0+inst.fleet_cost*gp.quicksum(idle.values())
        +gp.quicksum(float(C[l])*rho[l] for l in range(L))
        +gp.quicksum(float(qe[l,m])*flow[l,m] for l,m in arcs))
    try:
        primal.Params.TimeLimit=max(.001,deadline-time.monotonic());primal.optimize()
        answer['primal_runtime']=float(primal.Runtime)
        if primal.SolCount:
            rv=np.asarray([rho[l].X for l in range(L)]);iv=np.asarray([idle[l].X for l in range(L)])
            iv=np.maximum(iv,rv/(1-rv))
            ev=np.zeros((L,L))
            for a,var in flow.items():ev[a]=var.X
            balance=D.sum(axis=1)*rv-D.T@rv+ev.sum(axis=1)-ev.sum(axis=0)
            residual=max(float(np.max(np.abs(balance),initial=0.)),
                         float(np.max(ev.sum(axis=1)-inst.kappa*rv,initial=0.)),
                         float(np.max(rv-caps,initial=0.)),float(max(0.,-min(np.r_[rv,iv,ev.ravel()],default=0.))))
            objective=float(inst.c0+inst.fleet_cost*iv.sum()+C@rv+sum(qe[a]*ev[a] for a in arcs))
            services=[];empties=[];fleet=float(iv.sum())
            for o,(l,m) in assignment.items():
                amount=float(inst.demand[o]*rv[l])
                if amount>1e-10 and np.isfinite(q[o,l,m]):
                    services.append(dict(od=o,l=l,m=m,rate=amount,time=float(qt[o,l,m])))
                    fleet+=amount*qt[o,l,m]
            for l,m in arcs:
                if ev[l,m]>1e-10:
                    empties.append(dict(l=l,m=m,rate=float(ev[l,m]),time=float(qet[l,m])))
                    fleet+=ev[l,m]*qet[l,m]
            if residual<=1e-5:
                answer.update(objective=objective,rho=rv.tolist(),idle=iv.tolist(),fleet=float(fleet),
                    served=sum(r['rate'] for r in services),service_flows=services,empty_flows=empties,
                    primal_residual=residual)
                if 'best_index' in oracle_result and getattr(inst,'service',None) is not None:
                    theta=np.zeros(len(inst.service));phi=np.zeros(len(inst.empty))
                    for r in services:theta[int(oracle_result['best_index'][r['od'],r['l'],r['m']])]+=r['rate']
                    for r in empties:phi[int(oracle_result['empty_index'][r['l'],r['m']])]+=r['rate']
                    answer.update(theta=theta.tolist(),phi=phi.tolist())
    finally:primal.dispose()
    if time.monotonic()>=deadline:answer['wall']=time.monotonic()-start;return answer
    dual=gp.Model('integer_design_small_dual');_settings(dual,deadline-time.monotonic(),threads)
    pi=dual.addVars(L,lb=-GRB.INFINITY,name='balance_price')
    nu=dual.addVars(L,lb=0,name='clock_price');sigma=dual.addVars(L,lb=0,name='cap_price')
    cones=[]
    for l in range(L):
        t=dual.addVar(lb=0);s=dual.addVar(lb=-GRB.INFINITY);z=dual.addVar(lb=-GRB.INFINITY)
        dual.addQConstr(s*s+z*z<=t*t);dual.addConstr(t+z<=inst.fleet_cost)
        dual.addConstr(float(C[l])+gp.quicksum(float(D[l,m])*(pi[l]-pi[m]) for m in range(L))
                       -inst.kappa*nu[l]+sigma[l]+t-z>=0)
        cones.append((t,s,z))
    for l,m in arcs:dual.addConstr(float(qe[l,m])+pi[l]-pi[m]+nu[l]>=0)
    dual.setObjective(inst.c0-gp.quicksum(float(caps[l])*sigma[l] for l in range(L))
                      -gp.quicksum(2*(t+s) for t,s,z in cones),GRB.MAXIMIZE)
    try:
        dual.Params.TimeLimit=max(.001,deadline-time.monotonic());dual.optimize()
        answer['dual_runtime']=float(dual.Runtime)
        if not dual.SolCount:answer['wall']=time.monotonic()-start;return answer
        piv=np.asarray([pi[l].X for l in range(L)])
        nuv=np.maximum(0.,[nu[l].X for l in range(L)])
        sv=np.maximum(0.,[sigma[l].X for l in range(L)])
        original_sigma=sv.copy();effective=dict(chosen_cost)
        # A no-route forced-zero cap has no cost at the generating point.
        # Transfer its multiplier to one unavailable assigned OD's source price.
        for l,ods in unreachable.items():
            if design[f'X{l}']>.5:
                o=ods[0];effective[o]+=sv[l]/inst.demand[o];original_sigma[l]=0.
        alpha=np.zeros((O,L))
        for o,(l,m) in assignment.items():alpha[o,l]=effective[o]+piv[l]-piv[m]
        A=np.asarray(inst.demand)@alpha-inst.kappa*nuv+original_sigma
        intercept=float(inst.c0+sum(_idle_conjugate(a,inst.fleet_cost,inst.rho_max) for a in A))
        M=np.asarray(inst.demand)*inst.rho_max
        b=np.maximum(alpha,0.)
        intercept-=float(M@b.sum(axis=1))
        gamma=np.maximum(0.,alpha[:,:,None]-piv[None,:,None]+piv[None,None,:]-lower_cost)
        for o,(l,m) in assignment.items():gamma[o,l,m]=0.
        cut_B=M[:,None,None]*(b[:,:,None]-gamma)
        pair_norm=float(np.max(np.abs(cut_B),axis=(1,2)).sum())
        cut_facility=np.zeros(len(facility_keys))
        cut_facility[:L]=-inst.rho_max*original_sigma
        service_penalty=sum(M[o]*max(0.,effective[o]-lower_cost[(o,)+assignment[o]]) for o in assignment)
        empty_penalty=0.;empty_residual=0.
        for l in range(L):
            deficit=max([max(0.,piv[m]-piv[l]-nuv[l]-empty_lower[l,m]) for m in range(L) if m!=l]+[0.])
            empty_penalty+=inst.kappa*inst.rho_max*deficit
            residual=max([max(0.,-(qe[l,m]+piv[l]-piv[m]+nuv[l]))
                          for m in range(L) if (l,m) in arcs]+[0.])
            empty_residual+=inst.kappa*inst.rho_max*residual
        for j,key in enumerate(facility_keys):
            if facility_array[j]<.5:
                cut_facility[j]-=service_penalty+(0. if key.startswith('Y') else empty_penalty)
        facility_norm=float(np.abs(cut_facility).sum())
        reserve=2e-10*(1+abs(intercept)+pair_norm+facility_norm)
        intercept-=empty_residual+reserve
        value=float(intercept+cut_facility@facility_array+sum(cut_B[o,l,m] for o,(l,m) in assignment.items()))
        gap=answer['objective']-value
        if return_sparse_cut is None:return_sparse_cut=(O*L*L<=10000)
        slope={}
        if return_sparse_cut:
            slope={key:float(c) for key,c in zip(facility_keys,cut_facility) if c}
            for o,l,m in np.argwhere(cut_B!=0):slope[f'B{o}:{l}:{m}']=float(cut_B[o,l,m])
        answer.update(dual_value=value,dual_value_raw=float(dual.ObjVal),bound=value,
            cut_intercept=intercept,intercept=intercept,cut_slope=slope,cut_B=cut_B,
            cut_facility=cut_facility,cut_valid=True,
            cg_complete=bool(abs(gap)<=certificate_tol*(1+abs(answer['objective'])) and answer['primal_residual']<=1e-5),
            duality_gap=float(gap),raw_duality_gap=float(answer['objective']-dual.ObjVal),
            certificate='implicit_universal_columns_source_hull',cut_correction=float(empty_residual),
            roundoff_reserve=reserve,service_closed_penalty=float(service_penalty),
            empty_closed_penalty=float(empty_penalty),
            universal_service_lower_shape=list(lower_cost.shape),universal_empty_lower_shape=list(empty_lower.shape))
        answer['status']='optimal' if answer['cg_complete'] else 'certified_lower_bound'
    finally:dual.dispose()
    answer['wall']=time.monotonic()-start
    return answer
