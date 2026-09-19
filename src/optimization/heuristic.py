"""Primal incumbent improvement using a sparse restricted-column MISOCP.

Only the supplied complete-model lower bound is retained. The restricted
search bound never replaces the global Benders bound.
"""

from __future__ import annotations
import time
import numpy as np
from gurobipy import GRB
from .column_pool import build_result, record_flow, rebuild_route, prepare_pool
from .restricted_master import build_restricted_master


def solve(
    data,
    seed_result,
    *,
    time_limit=120.0,
    threads=4,
    add_oracle_pairs=2,
    global_lower_bound=0.0,
    columns=None,
    progress=None,
    log=None,
    seed=43,
    mip_gap=1e-4,
    mip_gap_abs=1e-6,
    stop_requested=None,
):
    """Improve a feasible seed by jointly selecting sites, capacities and routes.

    ``columns`` is optional additional physical routes in full-column reference format.
    Every positive seed flow is retained and repriced, even after cost changes.
    Up to ``add_oracle_pairs`` cheapest feasible OD/base-pairs are added per OD.
    ``global_lower_bound`` must come from a complete-model solve using these
    exact input parameters. Zero is always valid with nonnegative costs.
    The result is independently validated and never worse than the seed after
    repricing. No restricted-column MIP lower bound leaves this function.
    """
    solver_seed = seed
    start = time.perf_counter()
    if (
        time_limit <= 0
        or int(add_oracle_pairs) != add_oracle_pairs
        or add_oracle_pairs < 0
    ):
        raise ValueError(
            "Positive time_limit and nonnegative integer add_oracle_pairs required"
        )
    if not np.isfinite(global_lower_bound) or global_lower_bound < 0:
        raise ValueError("A finite nonnegative full-model lower bound is required")
    L, H, D = (int(data[key]) for key in ("n_l", "n_h", "n_od"))
    pool = prepare_pool(
        data,
        seed_result,
        add_oracle_pairs=add_oracle_pairs,
        global_lower_bound=global_lower_bound,
        columns=columns,
    )
    seed = pool["seed"]
    service, empty = pool["service"], pool["empty"]
    bundle = build_restricted_master(
        data,
        pool,
        started=start,
        threads=threads,
        seed=solver_seed,
        mip_gap=mip_gap,
        mip_gap_abs=mip_gap_abs,
        log=log,
    )
    model = bundle.model
    statistics = bundle.statistics
    x, z, y, alpha = bundle.x, bundle.z, bundle.y, bundle.alpha
    rho, idle, psi, varphi, Gamma = (
        bundle.rho,
        bundle.idle,
        bundle.psi,
        bundle.varphi,
        bundle.fleet,
    )
    last_report = [0.0]

    def callback(mm, where):
        if stop_requested is not None and stop_requested():
            mm.terminate()
            return
        now = time.perf_counter()
        if progress and now - last_report[0] >= 30 and where == GRB.Callback.MIP:
            value = float(mm.cbGet(GRB.Callback.MIP_OBJBST))
            progress(
                dict(
                    stage="restricted_design_search",
                    elapsed=now - start,
                    objective=min(seed["objective"], value),
                    columns=len(service) + len(empty),
                )
            )
            last_report[0] = now

    try:
        model.Params.TimeLimit = max(0.01, time_limit - (time.perf_counter() - start))
        model.optimize(
            callback if progress is not None or stop_requested is not None else None
        )
        answer = seed
        rejected = None
        if model.SolCount and model.ObjVal < seed["objective"] - 1e-7:
            aa = np.zeros((D, L, L))
            for group, variable in alpha.items():
                aa[group] = round(variable.X)
            solution = dict(
                facility=np.rint(x.X),
                capacity_level=np.rint(z.X),
                assignment=aa,
                airline=np.rint(y.X),
                rho=np.asarray([rho[l].X for l in range(L)]),
                idle=np.asarray([idle[l].X for l in range(L)]),
                fleet=float(Gamma.X),
            )
            sm = [
                record_flow(tr, psi[k].X)
                for k, tr in enumerate(service)
                if psi[k].X > 0.0
            ]
            em = [
                record_flow(tr, varphi[k].X)
                for k, tr in enumerate(empty)
                if varphi[k].X > 0.0
            ]
            try:
                answer = build_result(data, solution, sm, em, global_lower_bound)
            except ValueError as error:
                rejected = str(error)
        statistics.update(
            solver_wall=float(model.Runtime),
            nodes=float(model.NodeCount),
            restricted_status=int(model.Status),
            candidate_rejected=rejected,
            improved=answer["objective"] < seed["objective"] - 1e-7,
        )
        answer.update(
            method="restricted_column_MISOCP_primal_heuristic",
            params=dict(
                Seed=solver_seed,
                Threads=threads,
                MIPGap=float(model.Params.MIPGap),
                MIPGapAbs=float(model.Params.MIPGapAbs),
            ),
            wall=time.perf_counter() - start,
            budget=float(time_limit),
            status=int(model.Status),
            termination_reason=(
                "user_stop"
                if stop_requested is not None and stop_requested()
                else "restricted_column_primal_improvement"
            ),
            heuristic_seed_objective=seed["objective"],
            heuristic_stats=statistics,
        )
        return answer
    finally:
        model.dispose()


# Compatibility names for earlier test fixtures.
_pack = build_result
_record = record_flow
_rebuild = rebuild_route
