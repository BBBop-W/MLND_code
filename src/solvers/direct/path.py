"""Complete-column reference solver for small-instance algorithm validation."""

from __future__ import annotations
import time
import numpy as np
from src.solvers.pricing.label import enumerate_all_columns
from src.solvers.pricing.enumeration import EnumerationStopped
from src.solvers.direct.path_model import ModelBuildStopped, build_full_column_model
from src.solvers.settings import mip_bound


def solve(
    data,
    *,
    time_limit=60.0,
    threads=1,
    mip_gap=1e-4,
    fixed_facility=None,
    fixed_capacity_level=None,
    fixed_assignment=None,
    fixed_airlines=None,
    columns=None,
    seed=43,
    mip_gap_abs=1e-6,
    column_limit=None,
    stop_requested=None,
    progress=None,
):
    """Solve all columns, optionally fixing any/all strategic variables.

    ``objective`` includes infrastructure. ``recourse_objective`` includes the
    all-courier baseline, mission costs, empty movements and the explicit fleet
    cost; it is the quantity to compare with ``recourse.solve``.
    Supplying every fixed_* argument creates a continuous fixed-design SOCP.
    """
    started = time.perf_counter()

    def remaining():
        return (
            None if time_limit is None else time_limit - (time.perf_counter() - started)
        )

    def stopped(reason, construction_complete):
        return dict(
            model_version="path_cqn_v1",
            method="full_column_MISOCP",
            objective=None,
            bound=None,
            gap=None,
            status=11 if reason == "user_stop" else 9,
            termination_reason=reason,
            construction_complete=construction_complete,
            wall=time.perf_counter() - started,
        )

    try:
        columns = (
            enumerate_all_columns(
                data,
                time_limit=time_limit,
                column_limit=column_limit,
                stop_requested=stop_requested,
                progress=progress,
            )
            if columns is None
            else columns
        )
    except EnumerationStopped as error:
        return stopped(str(error), False)
    try:
        bundle = build_full_column_model(
            data,
            columns,
            threads=threads,
            seed=seed,
            mip_gap=mip_gap,
            mip_gap_abs=mip_gap_abs,
            fixed_facility=fixed_facility,
            fixed_capacity_level=fixed_capacity_level,
            fixed_assignment=fixed_assignment,
            fixed_airlines=fixed_airlines,
            time_limit=remaining(),
            stop_requested=stop_requested,
        )
    except ModelBuildStopped as error:
        return stopped(str(error), False)
    model = bundle.model
    recourse = bundle.recourse
    continuous = bundle.continuous
    service, empty = bundle.service, bundle.empty
    L = data["n_l"]
    P, R = len(service), len(empty)
    x, z, alpha, y = bundle.x, bundle.z, bundle.alpha, bundle.y
    rho, idle, psi, phi, Gamma = (
        bundle.rho,
        bundle.idle,
        bundle.psi,
        bundle.varphi,
        bundle.fleet,
    )
    try:

        def callback(model, where):
            if stop_requested is not None and stop_requested():
                model.terminate()

        if stop_requested is not None and stop_requested():
            return stopped("user_stop", True)
        seconds = remaining()
        if seconds is not None:
            if seconds <= 0:
                return stopped("time_limit", True)
            model.Params.TimeLimit = seconds
        model.optimize(callback if stop_requested is not None else None)
        result = dict(
            model_version="path_cqn_v1",
            params=dict(
                Seed=seed,
                Threads=threads,
                MIPGap=float(model.Params.MIPGap),
                MIPGapAbs=float(model.Params.MIPGapAbs),
            ),
            status=int(model.Status),
            construction_complete=True,
            wall=time.perf_counter() - started,
            service_columns=P,
            empty_columns=R,
            method="full_column_SOCP" if continuous else "full_column_MISOCP",
            objective=None,
            recourse_objective=None,
            bound=None,
            gap=None,
        )
        if not continuous:
            result["bound"] = mip_bound(model)
        if model.SolCount:
            rv = np.asarray([rho[l].X for l in range(L)])
            iv = np.asarray([idle[l].X for l in range(L)])
            sv = np.asarray([psi[k].X for k in range(P)])
            ev = np.asarray([phi[k].X for k in range(R)])
            result.update(
                objective=float(model.ObjVal),
                recourse_objective=float(recourse.getValue()),
                bound=(
                    float(model.ObjBound)
                    if not continuous
                    else float(model.ObjVal)
                    if model.Status == 2
                    else None
                ),
                gap=(
                    float(model.MIPGap)
                    if not continuous
                    else 0.0
                    if model.Status == 2
                    else None
                ),
                facility=x.X.copy(),
                capacity_level=z.X.copy(),
                assignment=alpha.X.copy(),
                airline=y.X.copy(),
                rho=rv,
                idle=iv,
                fleet=float(Gamma.X),
                served=float(sv.sum()),
                missions=[
                    dict(
                        trace=p,
                        od=p["od"],
                        l=p["l"],
                        m=p["m"],
                        i=p["i"],
                        j=p["j"],
                        flow=float(sv[k]),
                        time=p["time"],
                        cost=p["cost"],
                    )
                    for k, p in enumerate(service)
                    if sv[k] > 1e-9
                ],
                relocations=[
                    dict(
                        trace=p,
                        l=p["l"],
                        m=p["m"],
                        flow=float(ev[k]),
                        time=p["time"],
                        cost=p["cost"],
                    )
                    for k, p in enumerate(empty)
                    if ev[k] > 1e-9
                ],
            )
        return result
    finally:
        model.dispose()
