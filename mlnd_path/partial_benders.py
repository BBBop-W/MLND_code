"""Partial Benders: conic aggregate master and persistent path-allocation LP.

Only this new module defines the alternative decomposition. The baseline model,
solver and earlier benchmark results remain unchanged.
"""
from __future__ import annotations

import math
import time

import gurobipy as gp
from gurobipy import GRB
import numpy as np
from scipy import sparse

from .data import Instance
from .solver import _matrix, STATUS


def _tkey(o,l,m): return f"T{o}:{l}:{m}"
def _ekey(l,m): return f"E{l}:{m}"


class PathAllocationLP:
    """One reusable dual-simplex LP, including incremental global pricing."""
    def __init__(self, inst, threads=1, use_cg=True):
        self.inst=inst; self.use_cg=use_cg
        self.tkeys=[_tkey(o,l,m) for o in range(inst.n_od)
                    for l in range(inst.n_l) for m in range(inst.n_l)]
        self.ekeys=[_ekey(l,m) for l in range(inst.n_l) for m in range(inst.n_l) if l!=m]
        self.groups=self.tkeys+self.ekeys
        self.ng=len(self.groups)
        self.names=list(self.groups); self.maps=[{k:1.} for k in self.groups]
        self.senses=["="]*self.ng
        self.rowids={k:j for j,k in enumerate(self.names)}
        def gate(name,mapping):
            self.rowids[name]=len(self.names);self.names.append(name)
            self.maps.append(mapping);self.senses.append("<")
        for h in range(inst.n_h): gate(f"H{h}",{f"Y{h}":inst.M})
        for k in range(inst.n_l+inst.n_k): gate(f"K{k}",{inst.station_key(k):inst.MK})
        # Integer-equivalent disaggregated gates strengthen the LP cuts.
        for o,h in enumerate(inst.demand):
            for i in range(inst.n_h): gate(f"OH{o}:{i}",{f"Y{i}":h*inst.rho_max})
            for k in range(inst.n_l+inst.n_k):
                gate(f"OK{o}:{k}",{inst.station_key(k):h*inst.rho_max})
        for l in range(inst.n_l):
            for k in range(inst.n_l+inst.n_k):
                gate(f"EK{l}:{k}",{inst.station_key(k):inst.kappa*inst.rho_max})
        sr=[];sc=[];sv=[];er=[];ec=[];ev=[]
        def coefficient(rows,cols,values,name,col,value=1):
            if value: rows.append(self.rowids[name]);cols.append(col);values.append(value)
        for p,c in enumerate(inst.service):
            o,l,m=c['od'],c['l'],c['m']
            coefficient(sr,sc,sv,_tkey(o,l,m),p)
            for h in {c['i'],c['j']}:
                coefficient(sr,sc,sv,f"H{h}",p);coefficient(sr,sc,sv,f"OH{o}:{h}",p)
            for k,count in enumerate(c['swaps']):
                coefficient(sr,sc,sv,f"K{k}",p,count)
                if count: coefficient(sr,sc,sv,f"OK{o}:{k}",p)
        for r,c in enumerate(inst.empty):
            coefficient(er,ec,ev,_ekey(c['l'],c['m']),r)
            for k,count in enumerate(c['swaps']):
                coefficient(er,ec,ev,f"K{k}",r,count)
                if count: coefficient(er,ec,ev,f"EK{c['l']}:{k}",r)
        self.S=sparse.coo_matrix((sv,(sr,sc)),shape=(len(self.names),len(inst.service))).tocsc()
        self.E=sparse.coo_matrix((ev,(er,ec)),shape=(len(self.names),len(inst.empty))).tocsc()
        self.cs=np.asarray([inst.service_cost(p) for p in range(len(inst.service))])
        self.ce=np.asarray([inst.empty_cost(r) for r in range(len(inst.empty))])
        self.model=gp.Model('partial_path_allocation')
        self.model.Params.OutputFlag=0;self.model.Params.Threads=threads
        self.model.Params.Method=1
        self.model.Params.FeasibilityTol=1e-8;self.model.Params.OptimalityTol=1e-8
        self.rows=[self.model.addConstr(gp.LinExpr()==0 if sense=='=' else gp.LinExpr()<=0,name=name)
                   for name,sense in zip(self.names,self.senses)]
        self.art=[self.model.addVar(lb=0,obj=1,column=gp.Column([1.],[self.rows[j]]),name=f"unmet_{j}")
                  for j in range(self.ng)]
        self.theta={};self.phi={}
        self.phase=1
        if not use_cg:
            self.add_columns(range(len(inst.service)),range(len(inst.empty)))
        self.model.update()

    def dispose(self): self.model.dispose()

    def add_columns(self,S,E):
        for mat,ids,variables,costs,prefix in ((self.S,S,self.theta,self.cs,'theta'),
                                            (self.E,E,self.phi,self.ce,'phi')):
            for p in ids:
                if p in variables: continue
                lo,hi=mat.indptr[p:p+2]
                column=gp.Column(mat.data[lo:hi].tolist(),[self.rows[j] for j in mat.indices[lo:hi]])
                variables[p]=self.model.addVar(lb=0,obj=0 if self.phase==1 else costs[p],
                                               column=column,name=f'{prefix}_{p}')

    def _certificate(self,prices,phase):
        inst=self.inst
        y=np.asarray(prices,dtype=float)
        if phase==1: y[:self.ng]=np.minimum(y[:self.ng],1.)
        y[self.ng:]=np.minimum(y[self.ng:],0.)
        cs=(np.zeros(len(self.cs)) if phase==1 else self.cs)-np.asarray(self.S.T@y).reshape(-1)
        ce=(np.zeros(len(self.ce)) if phase==1 else self.ce)-np.asarray(self.E.T@y).reshape(-1)
        worstS=np.zeros(inst.n_od);worstE=np.zeros(inst.n_l)
        for p,c in enumerate(inst.service): worstS[c['od']]=max(worstS[c['od']],-cs[p])
        for r,c in enumerate(inst.empty): worstE[c['l']]=max(worstE[c['l']],-ce[r])
        correction=float(inst.rho_max*(np.dot(worstS,inst.demand)+inst.kappa*worstE.sum()))
        slope={}
        for price,mapping in zip(y,self.maps):
            for key,value in mapping.items(): slope[key]=slope.get(key,0.)+float(price*value)
        reserve=2e-10*(1+sum(abs(c) for c in slope.values()))
        return dict(intercept=-correction-reserve,slope=slope,rcS=cs,rcE=ce,
                    minimum_rc=float(min(np.r_[cs,ce],default=0.)),
                    correction=correction,reserve=reserve)

    def solve(self,point,time_limit=5.,pricing_tol=1e-7,phase_tol=1e-7):
        start=time.monotonic();deadline=start+time_limit
        info=dict(status='time_limit',feasible=False,complete=False,cut_valid=False,
                  lp_runtime=0.,pricing_wall=0.,cg_rounds=0,phase1_rounds=0,phase2_rounds=0)
        self.model.setAttr('RHS',self.rows,[sum(c*point[k] for k,c in mapping.items()) for mapping in self.maps])
        for phase in (1,2):
            self.phase=phase
            for variable in self.art:
                variable.UB=GRB.INFINITY if phase==1 else 0.
                variable.Obj=1. if phase==1 else 0.
            for p,v in self.theta.items():v.Obj=0. if phase==1 else self.cs[p]
            for r,v in self.phi.items():v.Obj=0. if phase==1 else self.ce[r]
            while time.monotonic()<deadline:
                self.model.Params.TimeLimit=max(.001,deadline-time.monotonic())
                self.model.optimize()
                info['lp_runtime']+=self.model.Runtime;info['cg_rounds']+=1
                info[f'phase{phase}_rounds']+=1
                if self.model.Status!=GRB.OPTIMAL:
                    info['status']='lp_'+STATUS.get(self.model.Status,str(self.model.Status))
                    info['wall']=time.monotonic()-start
                    return info
                if phase==2:
                    # A restricted LP allocation is already a physical feasible
                    # plan. Preserve it even if full pricing hits the deadline.
                    theta=np.zeros(len(self.cs));phi=np.zeros(len(self.ce))
                    for p,v in self.theta.items():theta[p]=v.X
                    for r,v in self.phi.items():phi[r]=v.X
                    residual=max([abs(row.Slack) if sense=='=' else max(0.,-row.Slack)
                                  for row,sense in zip(self.rows,self.senses)]+[0.])
                    info.update(feasible=residual<=1e-6,objective=float(self.model.ObjVal),
                                theta=theta.tolist(),phi=phi.tolist(),primal_residual=float(residual))
                pricing_start=time.monotonic()
                certificate=self._certificate(self.model.getAttr('Pi',self.rows),phase)
                certificate['value']=certificate['intercept']+sum(c*point[k] for k,c in certificate['slope'].items())
                info['pricing_wall']+=time.monotonic()-pricing_start
                if certificate['minimum_rc']>=-pricing_tol:
                    if phase==1:
                        phase_value=float(self.model.ObjVal)
                        info['phase1_objective']=phase_value
                        if phase_value>phase_tol:
                            info.update(status='infeasible' if certificate['value']>phase_tol else 'uncertified_phase1',
                                complete=certificate['value']>phase_tol,cut_valid=certificate['value']>phase_tol,
                                cut_type='feasibility',certificate=certificate,wall=time.monotonic()-start)
                            return info
                        break
                    theta=np.zeros(len(self.cs));phi=np.zeros(len(self.ce))
                    for p,v in self.theta.items():theta[p]=v.X
                    for r,v in self.phi.items():phi[r]=v.X
                    objective=float(self.model.ObjVal)
                    residual=max([abs(row.Slack) if sense=='=' else max(0.,-row.Slack)
                                  for row,sense in zip(self.rows,self.senses)]+[0.])
                    gap=objective-certificate['value']
                    valid=residual<=1e-6 and abs(gap)<=2e-6*(1+abs(objective))
                    info.update(status='optimal' if valid else 'uncertified_optimality',
                        feasible=residual<=1e-6,complete=valid,cut_valid=valid,cut_type='optimality',
                        objective=objective,theta=theta.tolist(),phi=phi.tolist(),
                        primal_residual=float(residual),duality_gap=float(gap),
                        certificate=certificate,wall=time.monotonic()-start,
                        service_pool=len(self.theta),empty_pool=len(self.phi))
                    return info
                # Add one most negative column per aggregate equality group.
                pricing_start=time.monotonic();bestS={};bestE={}
                for p,rc in enumerate(certificate['rcS']):
                    if p not in self.theta and rc < -pricing_tol:
                        c=self.inst.service[p];key=(c['od'],c['l'],c['m'])
                        if key not in bestS or rc<bestS[key][0]:bestS[key]=(rc,p)
                for r,rc in enumerate(certificate['rcE']):
                    if r not in self.phi and rc < -pricing_tol:
                        c=self.inst.empty[r];key=(c['l'],c['m'])
                        if key not in bestE or rc<bestE[key][0]:bestE[key]=(rc,r)
                info['pricing_wall']+=time.monotonic()-pricing_start
                if not bestS and not bestE:
                    info.update(status='existing_column_numerics',wall=time.monotonic()-start)
                    return info
                self.add_columns([p for rc,p in bestS.values()],[r for rc,r in bestE.values()])
            else:
                info['wall']=time.monotonic()-start
                return info
        info['wall']=time.monotonic()-start
        return info


def _master(inst,threads,mip_gap):
    model=gp.Model('partial_conic_aggregate_master')
    model.Params.OutputFlag=0;model.Params.Threads=threads;model.Params.Seed=0
    model.Params.NonConvex=0;model.Params.LazyConstraints=1;model.Params.MIPGap=mip_gap
    model.Params.FeasibilityTol=1e-8;model.Params.OptimalityTol=1e-8
    model.Params.BarQCPConvTol=1e-8
    x=model.addVars(inst.design_keys(),vtype=GRB.BINARY,name='design')
    rho=model.addVars(inst.n_l,lb=0,name='rho');idle=model.addVars(inst.n_l,lb=0,name='idle')
    T=model.addVars([_tkey(o,l,m) for o in range(inst.n_od) for l in range(inst.n_l)
                    for m in range(inst.n_l)],lb=0,name='aggregate_service')
    E=model.addVars([_ekey(l,m) for l in range(inst.n_l) for m in range(inst.n_l) if l!=m],lb=0,name='aggregate_empty')
    zeta=model.addVar(lb=-GRB.INFINITY,name='path_cost')
    cheapestT={};cheapestE={}
    commonT={};commonE={};distinct={o:True for o in range(inst.n_od)}
    for p,c in enumerate(inst.service):
        key=_tkey(c['od'],c['l'],c['m']);cheapestT[key]=min(cheapestT.get(key,math.inf),inst.service_cost(p))
        stations={k for k,count in enumerate(c['swaps']) if count}
        commonT[key]=commonT[key]&stations if key in commonT else stations
        if c['i']==c['j']:distinct[c['od']]=False
    for r,c in enumerate(inst.empty):
        key=_ekey(c['l'],c['m']);cheapestE[key]=min(cheapestE.get(key,math.inf),inst.empty_cost(r))
        stations={k for k,count in enumerate(c['swaps']) if count}
        commonE[key]=commonE[key]&stations if key in commonE else stations
    for key,variable in T.items():
        if key not in cheapestT:variable.UB=0.
    for key,variable in E.items():
        if key not in cheapestE:variable.UB=0.
    model.addConstr(zeta>=gp.quicksum(c*T[k] for k,c in cheapestT.items())
                    +gp.quicksum(c*E[k] for k,c in cheapestE.items()),name='cheapest_group_bound')
    for o,h in enumerate(inst.demand):
        M=h*inst.rho_max
        model.addConstr(gp.quicksum(x[f'B{o}:{l}:{m}'] for l in range(inst.n_l) for m in range(inst.n_l))<=1)
        if distinct[o]:
            total=gp.quicksum(T[_tkey(o,l,m)] for l in range(inst.n_l) for m in range(inst.n_l))
            for excluded in range(inst.n_h):
                model.addConstr(total<=M*gp.quicksum(x[f'Y{h0}'] for h0 in range(inst.n_h) if h0!=excluded))
        for l in range(inst.n_l):
            source=gp.quicksum(T[_tkey(o,l,m)] for m in range(inst.n_l))
            assigned=gp.quicksum(x[f'B{o}:{l}:{m}'] for m in range(inst.n_l))
            model.addConstr(source<=h*rho[l])
            model.addConstr(source>=h*rho[l]-M*(1-assigned))
            for m in range(inst.n_l):
                b=x[f'B{o}:{l}:{m}']
                model.addConstr(b<=x[f'X{l}']);model.addConstr(b<=x[f'X{m}'])
                model.addConstr(T[_tkey(o,l,m)]<=M*b)
                for k in commonT.get(_tkey(o,l,m),set()):
                    if k not in (l,m):model.addConstr(T[_tkey(o,l,m)]<=M*x[inst.station_key(k)])
    for key,stations in commonE.items():
        l,m=map(int,key[1:].split(':'))
        for k in stations:
            if k not in (l,m):model.addConstr(E[key]<=inst.kappa*inst.rho_max*x[inst.station_key(k)])
    for l in range(inst.n_l):
        model.addConstr(rho[l]<=inst.rho_max*x[f'X{l}'])
        outT=gp.quicksum(T[_tkey(o,l,m)] for o in range(inst.n_od) for m in range(inst.n_l))
        inT=gp.quicksum(T[_tkey(o,m,l)] for o in range(inst.n_od) for m in range(inst.n_l))
        outE=gp.quicksum(E[_ekey(l,m)] for m in range(inst.n_l) if m!=l)
        inE=gp.quicksum(E[_ekey(m,l)] for m in range(inst.n_l) if m!=l)
        model.addConstr(outT+outE==inT+inE)
        model.addConstr(outE<=inst.kappa*rho[l])
        for m in range(inst.n_l):
            if m!=l:model.addConstr(E[_ekey(l,m)]<=inst.kappa*inst.rho_max*x[f'X{m}'])
        up=model.addVar(lb=0,name=f'up_{l}');low=model.addVar(lb=0,name=f'low_{l}')
        one=model.addVar(lb=1,ub=1,name=f'one_{l}')
        model.addConstr(up-idle[l]==1);model.addConstr(low+rho[l]==1)
        model.addQConstr(one*one<=up*low)
        up.Start=1;low.Start=1;one.Start=1
    model.setObjective(inst.fixed_cost(x)+inst.c0+inst.fleet_cost*gp.quicksum(idle.values())+zeta)
    for variables in (x,rho,idle,T,E):
        for variable in variables.values():variable.Start=0.
    zeta.Start=0.
    model.update()
    return model,x,rho,idle,T,E,zeta


def solve_partial(inst:Instance,time_limit=30.,mip_gap=1e-4,threads=1,
                  use_cg=True,full_columns=None,output=False):
    """Solve the original integer physical model using an aggregate conic master.

    Only independently validated path allocations update the returned incumbent.
    A timeout/callback interruption never promotes an unvalidated master point.
    """
    if full_columns is not None:use_cg=not full_columns
    start=time.monotonic();deadline=start+time_limit
    model,x,rho,idle,T,E,zeta=_master(inst,threads,mip_gap)
    model.Params.OutputFlag=int(output)
    allocator=PathAllocationLP(inst,threads,use_cg)
    stats=dict(subproblem_calls=0,cg_rounds=0,phase1_rounds=0,phase2_rounds=0,
               cuts=0,feasibility_cuts=0,optimality_cuts=0,subproblem_wall=0.,
               pricing_wall=0.,lp_runtime=0.,master_wall=0.)
    best=dict(objective=inst.c0,design=inst.zero_design(),rho=[0.]*inst.n_l,idle=[0.]*inst.n_l,
              theta=[0.]*len(inst.service),phi=[0.]*len(inst.empty),fleet=0.,served=0.,primal_residual=0.)
    lower=0.;trace=[];errors=[];incomplete=[]
    variables=dict(x);variables.update(T);variables.update(E)

    def callback(master,where):
        nonlocal lower,best
        if time.monotonic()>=deadline:
            master.terminate();return
        if where==GRB.Callback.MIP:
            value=master.cbGet(GRB.Callback.MIP_OBJBND)
            if math.isfinite(value) and abs(value)<1e90:lower=max(lower,float(value))
        if where!=GRB.Callback.MIPSOL:return
        try:
            design={k:float(round(master.cbGetSolution(v))) for k,v in x.items()}
            point=dict(design)
            point.update({k:float(master.cbGetSolution(v)) for k,v in T.items()})
            point.update({k:float(master.cbGetSolution(v)) for k,v in E.items()})
            rv=np.asarray([master.cbGetSolution(rho[l]) for l in range(inst.n_l)])
            iv=np.asarray([master.cbGetSolution(idle[l]) for l in range(inst.n_l)])
            estimate=float(master.cbGetSolution(zeta))
            answer=allocator.solve(point,time_limit=max(.001,deadline-time.monotonic()))
            stats['subproblem_calls']+=1
            for key in ('cg_rounds','phase1_rounds','phase2_rounds','pricing_wall','lp_runtime'):
                stats[key]+=answer.get(key,0.)
            stats['subproblem_wall']+=answer['wall']
            if answer.get('feasible'):
                theta=np.asarray(answer['theta']);phi=np.asarray(answer['phi'])
                matrix=_matrix(inst)
                rhs=matrix.constants+np.asarray([sum(c*design[k] for k,c in mapping.items()) for mapping in matrix.maps])
                residual=matrix.base@np.r_[rv,iv]+matrix.service@theta+matrix.empty@phi-rhs
                row_error=max([abs(a) if sense=='=' else max(0.,a) for a,sense in zip(residual,matrix.senses)]+[0.])
                cone_error=max([max(0.,r/(1-r)-i) if r<1 else math.inf for r,i in zip(rv,iv)]+[0.])
                nonnegative=max(0.,-min(np.r_[rv,iv,theta,phi],default=0.))
                error=float(max(row_error,cone_error,nonnegative,answer['primal_residual']))
                objective=inst.fixed_cost(design)+inst.c0+inst.fleet_cost*float(iv.sum())+answer['objective']
                if error<=1e-5 and objective<best['objective']:
                    fleet=float(iv.sum()+sum(theta[p]*c['time'] for p,c in enumerate(inst.service))
                                +sum(phi[r]*c['time'] for r,c in enumerate(inst.empty)))
                    best=dict(objective=float(objective),design=design,rho=rv.tolist(),idle=iv.tolist(),
                        theta=theta.tolist(),phi=phi.tolist(),fleet=fleet,served=float(theta.sum()),primal_residual=error)
            if not answer.get('cut_valid') or not answer.get('complete'):
                incomplete.append(answer['status']);master.terminate();return
            certificate=answer['certificate']
            expression=certificate['intercept']+gp.quicksum(c*variables[k] for k,c in certificate['slope'].items())
            cut_value=certificate['value']
            if answer['cut_type']=='feasibility':
                master.cbLazy(expression<=0.)
                stats['feasibility_cuts']+=1;stats['cuts']+=1
            elif estimate<cut_value-1e-7:
                master.cbLazy(zeta>=expression)
                stats['optimality_cuts']+=1;stats['cuts']+=1
            trace.append(dict(iteration=len(trace),elapsed=time.monotonic()-start,
                              lower_bound=lower,upper_bound=best['objective'],
                              gap=max(0.,best['objective']-lower)/max(1.,abs(best['objective'])),
                              phase1_objective=answer.get('phase1_objective'),cut_type=answer['cut_type'],
                              cg_rounds=answer['cg_rounds']))
        except Exception as exc:
            errors.append(f'{type(exc).__name__}: {exc}');master.terminate()
    try:
        if time.monotonic()<deadline:
            model.Params.TimeLimit=max(.001,deadline-time.monotonic())
            master_start=time.monotonic();model.optimize(callback)
            master_elapsed=time.monotonic()-master_start
            stats['master_wall']=max(0.,master_elapsed-stats['subproblem_wall'])
            try:
                value=float(model.ObjBound)
                if math.isfinite(value) and abs(value)<1e90:lower=max(lower,value)
            except (gp.GurobiError,AttributeError):pass
        gap=max(0.,best['objective']-lower)/max(1.,abs(best['objective']))
        status='OPTIMAL_TOL' if gap<=mip_gap and not errors and not incomplete else 'TIME_LIMIT'
        if errors:status='CALLBACK_ERROR'
        elif incomplete and time.monotonic()<deadline:status='INCOMPLETE_RECOURSE'
        return dict(best,algorithm='partial_cg' if use_cg else 'partial_full',status=status,
                    bound=float(lower),lower_bound=float(lower),upper_bound=best['objective'],gap=float(gap),
                    wall=time.monotonic()-start,stats=stats,trace=trace,errors=errors,
                    incomplete_recourse=incomplete,service_pool=len(allocator.theta),empty_pool=len(allocator.phi),
                    validated_incumbent=True)
    finally:
        allocator.dispose();model.dispose()
