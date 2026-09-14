"""Fully priced Benders separation at fractional root designs.

The master is temporarily continuous. Its fractional designs are passed to the
conic recourse unchanged, and their recourse values are never integer incumbent
bounds. All added cuts carry the same global certificate as the main solver.
Binary types are restored even on an exception or an expired deadline.
"""
from __future__ import annotations

import math
import time

from gurobipy import GRB
import gurobipy as gp

from .solver import solve_cg, STATUS


def generate_root_cuts(inst, master, x, zeta, pools=None, time_limit=5.0,
                       max_rounds=30, tolerance=1e-5, threads=1):
    """Add cuts to ``master`` in place and restore every design variable binary.

    ``lower_bound`` is the best *solved master LP* bound observed. It remains
    valid if the final iteration adds a cut but the deadline precludes re-solving
    the master. Fractional primal recourse objectives only appear in ``trace``.
    """
    started = time.monotonic()
    trace=[]
    result=dict(cuts=0,rounds=0,wall=0.,subproblem_wall=0.,pricing_wall=0.,
                master_wall=0.,pools=pools,lower_bound=0.,trace=trace,
                status="round_limit",binary_restored=False)
    original_time_limit=master.Params.TimeLimit
    original_threads=master.Params.Threads
    original_feasibility=master.Params.FeasibilityTol

    def remaining():
        return time_limit-(time.monotonic()-started)

    def cut_expression(answer):
        return answer["cut_intercept"]+gp.quicksum(c*x[k] for k,c in answer["cut_slope"].items())

    try:
        for variable in x.values(): variable.VType=GRB.CONTINUOUS
        master.Params.Threads=threads
        master.Params.FeasibilityTol=1e-8
        master.update()
        for round_id in range(max_rounds):
            if remaining() <= .005:
                result["status"]="time_limit"
                break
            master.Params.TimeLimit=max(.001,remaining())
            master_start=time.monotonic()
            master.optimize()
            result["master_wall"] += time.monotonic()-master_start
            if master.Status != GRB.OPTIMAL:
                result["status"]="master_"+STATUS.get(master.Status,str(master.Status))
                break
            lower=float(master.ObjVal)
            if math.isfinite(lower): result["lower_bound"]=max(result["lower_bound"],lower)
            point={k:float(variable.X) for k,variable in x.items()}
            estimate=float(zeta.X)
            if remaining() <= .005:
                result["status"]="time_limit"
                break
            sp_start=time.monotonic()
            answer=solve_cg(inst,point,pools=result["pools"],time_limit=remaining(),threads=threads)
            result["subproblem_wall"] += time.monotonic()-sp_start
            result["pricing_wall"] += answer.get("pricing_wall",answer.get("pricing_time",0.))
            if "service_pool" in answer and "empty_pool" in answer:
                result["pools"]=(answer["service_pool"],answer["empty_pool"])
            result["rounds"] += 1
            entry=dict(round=round_id+1,elapsed=time.monotonic()-started,
                       master_lower_bound=result["lower_bound"],zeta=estimate,
                       fractional_recourse_primal=answer.get("objective"),
                       fractional_recourse_dual=answer.get("dual_value"),
                       cg_complete=answer.get("cg_complete",False),
                       fractional_variables=sum(1 for value in point.values() if 1e-7<value<1-1e-7),
                       cg_iterations=answer.get("cg_iterations",0))
            trace.append(entry)
            if not answer.get("cut_valid") or not answer.get("cg_complete"):
                result["status"]="incomplete_recourse"
                break
            cut_value=answer["cut_intercept"]+sum(c*point[k] for k,c in answer["cut_slope"].items())
            violation=cut_value-estimate
            entry["cut_violation"]=violation
            if violation <= tolerance:
                result["status"]="separated"
                break
            master.addConstr(zeta>=cut_expression(answer),name=f"fractional_root_cut_{round_id}")
            result["cuts"] += 1
        # A last inexpensive LP solve records the bound after the final cut.
        if result["cuts"] and remaining() > .005:
            master.Params.TimeLimit=max(.001,remaining())
            master_start=time.monotonic()
            master.optimize()
            result["master_wall"] += time.monotonic()-master_start
            if master.Status == GRB.OPTIMAL:
                result["lower_bound"]=max(result["lower_bound"],float(master.ObjVal))
    finally:
        for variable in x.values(): variable.VType=GRB.BINARY
        master.Params.TimeLimit=original_time_limit
        master.Params.Threads=original_threads
        master.Params.FeasibilityTol=original_feasibility
        master.update()
        result["binary_restored"]=all(variable.VType==GRB.BINARY for variable in x.values())
        result["wall"]=time.monotonic()-started
    return result
