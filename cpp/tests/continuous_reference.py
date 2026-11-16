"""Independent same-matrix continuous reference and bounded conic certificates.

Consumes only the C++ sparse LinearModel export, not the Python formulation.
Cone-coordinate repair is an LP restriction, never a lower-bound calculation.
The complementary-ray dual is also a restriction; its independently corrected
feasible dual objective is a lower bound, even without optimal solver status.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import gurobipy as gp
import numpy as np
from scipy import sparse


def json_default(value):
    if isinstance(value,np.generic):return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def read_matrix(path):
    raw = path.read_bytes()
    data = json.loads(raw)
    rows, cols, values = [], [], []
    for i, row in enumerate(data["rows"]):
        for j, value in row[3]:
            rows.append(i); cols.append(j); values.append(value)
    n, m = len(data["variables"]), len(data["rows"])
    data["A"] = sparse.coo_matrix((values, (rows, cols)), shape=(m, n)).tocsr()
    data["A"].sum_duplicates()
    data["lb"] = np.array([v[1] for v in data["variables"]])
    data["ub"] = np.array([v[2] for v in data["variables"]])
    data["cost"] = np.array([v[3] for v in data["variables"]])
    data["rhs"] = np.array([r[2] for r in data["rows"]])
    data["sense"] = np.array([{"E": "=", "L": "<", "G": ">"}[r[1]] for r in data["rows"]])
    data["sha256"] = hashlib.sha256(raw).hexdigest()
    assert data["minimize"] and all(not v[4] for v in data["variables"])
    assert np.isfinite(data["ub"]).all() and (data["lb"] >= 0).all()
    return data


def residuals(d, x):
    x = np.asarray(x, dtype=np.longdouble)
    activity = d["A"].astype(np.longdouble) @ x
    magnitude = abs(d["A"]).astype(np.longdouble) @ abs(x) + abs(d["rhs"])
    diff = activity - d["rhs"]
    violation = np.where(d["sense"] == "=", abs(diff),
                         np.where(d["sense"] == "<", np.maximum(diff, 0), np.maximum(-diff, 0)))
    scaled = violation / (1 + magnitude)
    def worst(index):
        row = d["rows"][index]
        return dict(index=int(index), name=row[0], sense=row[1], rhs=row[2],
                    activity=float(activity[index]), magnitude=float(magnitude[index]),
                    absolute_violation=float(violation[index]), scaled_violation=float(scaled[index]),
                    coefficient_range=[min(abs(a) for _, a in row[3] if a), max(abs(a) for _, a in row[3])])
    bounds = float(max(0, np.max(d["lb"] - x), np.max(x - d["ub"])))
    cone = 0.0
    for u, v, t, factor in d["cones"]:
        lhs, rhs = x[t] ** 2, factor * x[u] * x[v]
        cone = max(cone, float((lhs-rhs)/(1+abs(lhs)+abs(rhs))))
    return dict(max_bound_violation=bounds, max_linear_violation=float(max(violation)),
                max_scaled_linear_violation=float(max(scaled)), max_scaled_cone_violation=cone,
                worst_linear_constraint=worst(int(np.argmax(violation))),
                worst_scaled_linear_constraint=worst(int(np.argmax(scaled))),
                passed=bounds <= 2e-6 and max(scaled) <= 2e-7 and cone <= 2e-7)


def model(d, seconds, cones=True):
    m = gp.Model()
    m.Params.OutputFlag = 0; m.Params.Threads = 1
    m.Params.TimeLimit = max(.001, seconds)
    m.Params.FeasibilityTol = 1e-8; m.Params.OptimalityTol = 1e-8
    if cones:
        m.Params.Method = 2; m.Params.BarQCPConvTol = 1e-9
    else:
        m.Params.Method = 1
    x = m.addMVar(len(d["lb"]), lb=d["lb"], ub=d["ub"], obj=d["cost"])
    m.ObjCon = d["constant"]
    m.addMConstr(d["A"], x, d["sense"], d["rhs"])
    if cones:
        for u, v, t, factor in d["cones"]:
            m.addQConstr(x[t].item()**2 <= factor*x[u].item()*x[v].item())
    m.update()
    delta = m.getA()-d["A"]
    checks = dict(variables=m.NumVars, rows=m.NumConstrs, cones=m.NumQConstrs,
                  matrix_exact=delta.nnz == 0 or np.max(abs(delta.data)) == 0,
                  lower_bounds_exact=np.array_equal(x.LB, d["lb"]),
                  upper_bounds_exact=np.array_equal(x.UB, d["ub"]),
                  objective_exact=np.array_equal(x.Obj, d["cost"]) and m.ObjCon == d["constant"],
                  rhs_exact=np.array_equal(m.getAttr("RHS"), d["rhs"]),
                  senses_exact=np.array_equal(m.getAttr("Sense"), d["sense"]))
    # Check actual quadratic terms, not only the requested cone count.
    cone_checks = []
    for q, (u, v, t, factor) in zip(m.getQConstrs(), d["cones"]):
        expr = m.getQCRow(q)
        actual = {}
        for i in range(expr.size()):
            pair=tuple(sorted((expr.getVar1(i).index,expr.getVar2(i).index)))
            actual[pair]=actual.get(pair,0)+expr.getCoeff(i)
        cone_checks.append(actual == {(t,t):1.0,tuple(sorted((u,v))):-factor}
                           and expr.getLinExpr().size()==0 and q.QCRHS==0 and q.QCSense=='<')
    checks["cones_exact"] = all(cone_checks) and (not cones or len(cone_checks)==len(d["cones"]))
    assert all(v for k,v in checks.items() if k.endswith("exact"))
    return m,x,checks


def primal_result(d, m, x):
    out=dict(status=int(m.Status), solver_optimal=m.Status==gp.GRB.OPTIMAL,
             solver_seconds=m.Runtime, has_solution=m.SolCount>0,
             certified_lower_bound=None)
    if m.SolCount:
        point=x.X
        check=residuals(d,point)
        objective=float(np.dot(d["cost"].astype(np.longdouble),point)+d["constant"])
        out.update(objective=objective,residuals=check,
                   validated_primal_upper_bound=objective if check["passed"] else None)
        return out,point
    return out,None


def repair_primal(d, point, seconds):
    # Fix all cone coordinates to a feasible nearby point; optimizing the
    # remaining original rows is an LP restriction of the same conic model.
    m,x,checks=model(d,seconds,False)
    repaired=np.asarray(point).copy()
    for u,v,t,factor in d["cones"]:
        if repaired[v] <= 0:
            m.dispose(); return dict(reason="nonpositive_cone_coordinate"),None
        repaired[u]=max(repaired[u],np.nextafter(repaired[t]**2/(factor*repaired[v]),math.inf))
        for j in (u,v,t):
            if repaired[j]<d["lb"][j]-1e-10 or repaired[j]>d["ub"][j]+1e-10:
                m.dispose();return dict(reason="cone_coordinate_outside_original_bound"),None
            x[j].LB=repaired[j];x[j].UB=repaired[j]
    m.optimize()
    out,point=primal_result(d,m,x)
    out["method"]="fixed_cone_coordinates_lp_restriction"
    out["constraint_matrix_check"]=checks
    m.dispose();return out,point


def verify_dual(d, certificate):
    pi=np.asarray(certificate["pi"],dtype=np.longdouble)
    upper=np.maximum(0,np.asarray(certificate["upper"],dtype=np.longdouble))
    lower=np.maximum(0,np.asarray(certificate["lower"],dtype=np.longdouble))
    pi=np.where(d["sense"]=='<',np.minimum(pi,0),np.where(d["sense"]=='>',np.maximum(pi,0),pi))
    cone=np.zeros(len(d["lb"]),dtype=np.longdouble)
    repaired=[]
    for (u,v,t,factor),(a,b,c) in zip(d["cones"],certificate["cones"]):
        a,b,c=np.longdouble(max(0,a)),np.longdouble(max(0,b)),np.longdouble(c)
        if (4/np.longdouble(factor))*a*b < c*c:
            if a>0:b=np.nextafter(c*c/(4*a/factor),np.longdouble(math.inf))
            elif b>0:a=np.nextafter(c*c/(4*b/factor),np.longdouble(math.inf))
            else:a=b=np.nextafter(abs(c)*np.sqrt(np.longdouble(factor))/2,np.longdouble(math.inf))
        assert a>=0 and b>=0 and (4/np.longdouble(factor))*a*b>=c*c
        cone[u]+=a;cone[v]+=b;cone[t]+=c;repaired.append([float(a),float(b),float(c)])
    residual=d["A"].astype(np.longdouble).T@pi-upper+lower+cone-d["cost"]
    correction=np.dot(d["ub"].astype(np.longdouble),np.maximum(residual,0))
    raw=np.longdouble(d["constant"])+np.dot(d["rhs"].astype(np.longdouble),pi)-np.dot(d["ub"].astype(np.longdouble),upper)+np.dot(d["lb"].astype(np.longdouble),lower)
    magnitude=abs(d["constant"])+np.dot(abs(d["rhs"]).astype(np.longdouble),abs(pi))+np.dot(d["ub"].astype(np.longdouble),upper)+np.dot(abs(d["lb"]).astype(np.longdouble),lower)
    margin=1e-8*(1+abs(raw))+128*np.finfo(float).eps*(1+magnitude)
    lower_bound=float(np.nextafter(raw-correction-margin,np.longdouble(-math.inf)))
    return dict(certified=bool(np.isfinite(lower_bound)),certified_lower_bound=lower_bound,
                scope=d["scope"],bound_scope="exported_matrix_only_no_omitted_column_certificate",
                raw_dual_objective=float(raw),stationarity_correction=float(correction),
                maximum_stationarity_violation=float(max(0,max(residual))),numerical_margin=float(margin),
                proof="weak_conic_duality_with_full_matrix_finite_bound_residual_correction")


def dual_ray(d, point, seconds):
    started=time.perf_counter();n,m=len(d["lb"]),len(d["rhs"])
    ccount=len(d["cones"])
    clb=np.r_[np.where(d["sense"]=='<',-gp.GRB.INFINITY,np.where(d["sense"]=='>',0,-gp.GRB.INFINITY)),np.zeros(2*n+ccount)]
    cub=np.r_[np.where(d["sense"]=='<',0,gp.GRB.INFINITY),np.full(2*n+ccount,gp.GRB.INFINITY)]
    objective=np.r_[d["rhs"],-d["ub"],d["lb"],np.zeros(ccount)]
    rr,cc,vv,rays=[],[],[],[]
    for k,(u,v,t,factor) in enumerate(d["cones"]):
        if point[u]<=0 or point[v]<=0:return dict(reason="nonpositive_ray_anchor"),None
        a=math.sqrt(factor*point[v]/point[u]);b=np.nextafter(max(math.sqrt(factor*point[u]/point[v]),factor/a),math.inf)
        c=-2*(1 if point[t]>=0 else -1)
        rays.append((a,b,c))
        rr.extend((u,v,t));cc.extend((k,k,k));vv.extend((a,b,c))
    C=sparse.coo_matrix((vv,(rr,cc)),shape=(n,ccount)).tocsr()
    A=sparse.hstack((d["A"].T,-sparse.eye(n),sparse.eye(n),C),format="csr")
    dm=gp.Model();dm.Params.OutputFlag=0;dm.Params.Threads=1;dm.Params.Method=1
    dm.Params.FeasibilityTol=1e-9;dm.Params.OptimalityTol=1e-9
    y=dm.addMVar(m+2*n+ccount,lb=clb,ub=cub,obj=objective)
    dm.ModelSense=-1;dm.ObjCon=d["constant"]
    dm.addMConstr(A,y,'<',d["cost"])
    dm.Params.TimeLimit=max(.001,seconds-(time.perf_counter()-started));dm.optimize()
    out=dict(status=int(dm.Status),solver_optimal=dm.Status==gp.GRB.OPTIMAL,
             solver_seconds=dm.Runtime,method="complementary_cone_ray_dual_lp_restriction")
    certificate=None
    if dm.SolCount:
        values=y.X
        certificate=dict(matrix_sha256=d["sha256"],pi=values[:m].tolist(),
                         upper=values[m:m+n].tolist(),lower=values[m+n:m+2*n].tolist(),
                         cones=[[a*values[m+2*n+k],b*values[m+2*n+k],c*values[m+2*n+k]] for k,(a,b,c) in enumerate(rays)])
        out.update(verify_dual(d,certificate))
    dm.dispose();out["total_seconds"]=time.perf_counter()-started
    return out,certificate


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("matrix",type=Path);parser.add_argument("output_prefix",type=Path)
    parser.add_argument("--seconds",type=float,default=120)
    parser.add_argument("--cplex-point",type=Path)
    parser.add_argument("--verify-certificate",type=Path)
    args=parser.parse_args();started=time.perf_counter();d=read_matrix(args.matrix)
    if args.verify_certificate:
        c=json.loads(args.verify_certificate.read_text());assert c["matrix_sha256"]==d["sha256"]
        print(json.dumps(verify_dual(d,c),indent=2,default=json_default));return
    report=dict(matrix_sha256=d["sha256"],K=d["K"],scope=d["scope"],
                solver="Gurobi",threads=1,budget_seconds=args.seconds,
                bound_scope="exported_matrix_only_no_omitted_column_certificate")
    remaining=lambda:max(0,args.seconds-(time.perf_counter()-started))
    m,x,checks=model(d,max(.01,.5*remaining()))
    report["same_matrix_check"]=checks;m.optimize()
    report["primal"],point=primal_result(d,m,x);m.dispose()
    if point is not None:
        Path(str(args.output_prefix)+".gurobi.values.json").write_text(json.dumps(point.tolist()))
    if args.cplex_point:
        cp=np.array(json.loads(args.cplex_point.read_text()))
        report["cplex_point_independent_residuals"]=residuals(d,cp)
        if remaining()>1:
            report["cplex_primal_repair"],rp=repair_primal(d,cp,min(30,.4*remaining()))
            if rp is not None:Path(str(args.output_prefix)+".cplex_repaired.values.json").write_text(json.dumps(rp.tolist()))
    if point is not None and remaining()>1:
        Path(str(args.output_prefix)+".gurobi.json").write_text(json.dumps(report,indent=2,default=json_default))
        if not report["primal"]["residuals"]["passed"]:
            report["gurobi_primal_repair"],rp=repair_primal(d,point,min(30,.4*remaining()))
            if rp is not None and report["gurobi_primal_repair"]["residuals"]["passed"]:point=rp
        report["dual"],certificate=dual_ray(d,point,remaining())
        if certificate:Path(str(args.output_prefix)+".dual_certificate.json").write_text(json.dumps(certificate))
    report["total_seconds"]=time.perf_counter()-started
    Path(str(args.output_prefix)+".gurobi.json").write_text(json.dumps(report,indent=2,default=json_default))
    print(json.dumps(report,indent=2,default=json_default))


if __name__=="__main__":main()
