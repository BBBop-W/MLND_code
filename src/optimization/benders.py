"""Integer-master Benders decomposition with full-graph column generation."""

from __future__ import annotations
import hashlib
import itertools
import time
from gurobipy import GRB
import numpy as np
from src.model.network import network, levels, facility_costs, incidence
from src.model.provenance import source_hashes
from .master import build_master
from .recourse import solve as recourse, RecourseStopped


def solve(
    data,
    time_limit=120.0,
    threads=1,
    mip_gap=1e-4,
    log=None,
    fixed_facility=None,
    fixed_capacity_level=None,
    max_iterations=200,
    progress=None,
    mip_gap_abs=1e-6,
    checkpoint=None,
    stop_requested=None,
    initial_result=None,
    seed=43,
):
    from src.pricing.oracle import price_design

    started = time.perf_counter()
    deadline = float("inf") if time_limit is None else started + time_limit
    L, H, K, D = (data[k] for k in ("n_l", "n_h", "n_k", "n_od"))
    net = network(data)
    sizes, capacities, level_cost = levels(data)
    fixed = facility_costs(data)
    c0 = float(data["demand"] @ data["ground"])
    allopen = price_design(data)
    model, x, z, alpha, y, zeta = build_master(
        data, allopen, threads, mip_gap, log, mip_gap_abs, seed
    )
    if fixed_facility is not None:
        fixed_facility = np.asarray(fixed_facility, float)
        if fixed_facility.shape != (net["N"],) or np.any(
            (fixed_facility != 0) & (fixed_facility != 1)
        ):
            raise ValueError("Invalid fixed facilities")
        x.LB = fixed_facility
        x.UB = fixed_facility
    if fixed_capacity_level is not None:
        fixed_capacity_level = np.asarray(fixed_capacity_level, float)
        if (
            fixed_capacity_level.shape != capacities.shape
            or np.any((fixed_capacity_level != 0) & (fixed_capacity_level != 1))
            or np.any(fixed_capacity_level.sum(axis=1) > 1)
        ):
            raise ValueError("Invalid fixed capacity levels")
        if fixed_facility is not None and np.any(
            fixed_capacity_level.sum(axis=1) != fixed_facility[L : L + H]
        ):
            raise ValueError("Fixed levels do not match facilities")
        z.LB = fixed_capacity_level
        z.UB = fixed_capacity_level
    model.update()
    stats = dict(
        master_binaries=int(model.NumBinVars),
        master_integers=int(model.NumIntVars),
        master_continuous=int(model.NumVars - model.NumIntVars),
        recourse_calls=0,
        benders_cuts=0,
        cg_iterations=0,
        pricing_calls=1,
        pricing_wall=allopen["wall"],
        recourse_wall=0.0,
        generated_columns=0,
        complete_subproblems=0,
        interrupted_subproblems=0,
    )
    f = np.zeros(net["N"]) if fixed_facility is None else fixed_facility.copy()
    zz = np.zeros_like(capacities)
    if fixed_capacity_level is not None:
        zz = fixed_capacity_level.copy()
        f[L : L + H] = zz.sum(axis=1)
    else:
        for i in range(H):
            if f[L + i]:
                zz[i, int(np.argmin(level_cost[i]))] = 1
    baseline = dict(
        facility=f,
        capacity_level=zz,
        hub_count=zz @ sizes,
        assignment=np.zeros((D, L, L)),
        airline=np.zeros(net["count"]),
        rho=np.zeros(L),
        idle=np.zeros(L),
        empty=np.zeros((L, L)),
        fleet=0.0,
    )
    best = dict(
        objective=c0 + fixed @ f + float((level_cost * zz).sum()),
        recourse=c0,
        solution=baseline,
        missions=[],
        relocations=[],
        hub_load=np.zeros(H),
        airline_load=np.zeros(net["count"]),
        served=0.0,
        primal_residual=0.0,
    )
    pools = dict(missions=[], relocations=[])
    trace = []
    errors = []
    lower = 0.0
    status = GRB.LOADED
    termination = "iteration_limit"
    evaluated = {}
    master_runtime = 0.0
    nodes = 0.0
    last_report = 0.0
    params = dict(
        MIPGap=float(model.Params.MIPGap),
        MIPGapAbs=float(model.Params.MIPGapAbs),
        Threads=int(model.Params.Threads),
        Seed=int(model.Params.Seed),
        TimeLimit=None if time_limit is None else float(time_limit),
        max_iterations=max_iterations,
        cg_max_iterations=None if max_iterations is None else 300,
        NodefileStart=float(model.Params.NodefileStart),
        NodefileDir=str(model.Params.NodefileDir),
    )
    source_sha256 = source_hashes()
    if initial_result is not None:
        old = initial_result
        initial_sol = {
            k: np.asarray(v, float).copy() if k != "fleet" else float(v)
            for k, v in old["_solution"].items()
        }
        shapes = dict(
            facility=(net["N"],),
            capacity_level=capacities.shape,
            hub_count=(H,),
            assignment=(D, L, L),
            airline=(net["count"],),
            rho=(L,),
            idle=(L,),
            empty=(L, L),
        )
        if any(initial_sol[k].shape != shape for k, shape in shapes.items()):
            model.dispose()
            raise ValueError("Initial result dimensions do not match the instance")
        if fixed_facility is not None and not np.array_equal(
            initial_sol["facility"], fixed_facility
        ):
            model.dispose()
            raise ValueError("Initial result violates fixed facilities")
        if fixed_capacity_level is not None and not np.array_equal(
            initial_sol["capacity_level"], fixed_capacity_level
        ):
            model.dispose()
            raise ValueError("Initial result violates fixed cabinet levels")
        if (
            not np.isfinite(old["objective"])
            or float(old.get("primal_residual", 0.0)) > 1e-5
        ):
            model.dispose()
            raise ValueError("Initial result must have a finite feasible incumbent")
        if float(old["objective"]) < best["objective"]:
            opening = float(
                fixed @ initial_sol["facility"]
                + (level_cost * initial_sol["capacity_level"]).sum()
                + net["fees"] @ initial_sol["airline"]
            )
            best = dict(
                objective=float(old["objective"]),
                recourse=float(old["objective"]) - opening,
                solution=initial_sol,
                missions=old["_missions"],
                relocations=old["_relocations"],
                hub_load=np.asarray(old["hub_load"], float),
                airline_load=np.asarray(old["airline_load"], float),
                served=float(old["served"]),
                primal_residual=float(old.get("primal_residual", 0.0)),
            )
            pools = dict(missions=best["missions"], relocations=best["relocations"])
        initial_bound = float(old.get("bound", 0.0))
        if not np.isfinite(initial_bound) or initial_bound > best["objective"] + 1e-5:
            model.dispose()
            raise ValueError("Invalid initial global lower bound")
        lower = max(lower, initial_bound)

    def requested():
        return stop_requested is not None and bool(stop_requested())

    def result_snapshot(reason="running", stage=None):
        sol = best["solution"]
        result = dict(
            name=data.get("name", "path_instance"),
            model_version="path_cqn_v1",
            method="binary_master_benders_CG",
            objective=float(best["objective"]),
            bound=float(lower),
            gap=max(0.0, best["objective"] - lower) / max(1.0, abs(best["objective"])),
            wall=time.perf_counter() - started,
            budget=time_limit,
            status=status,
            termination_reason=reason,
            master_runtime=master_runtime,
            nodes=nodes,
            stats=stats.copy(),
            trace=list(trace),
            errors=list(errors),
            params=params.copy(),
            fleet=float(sol["fleet"]),
            served=float(best["served"]),
            demand=float(np.sum(data["demand"])),
            n_od=D,
            rho=sol["rho"].tolist(),
            cabinet_total=float(sol["hub_count"].sum()),
            cabinet_counts=sol["hub_count"].tolist(),
            hub_load=best["hub_load"].tolist(),
            airline_load=best["airline_load"].tolist(),
            primal_residual=float(best["primal_residual"]),
            opened=dict(
                L=int(sol["facility"][:L].sum()),
                H=int(sol["facility"][L : L + H].sum()),
                E=int(sol["facility"][L + H :].sum()),
            ),
            open_airlines=int(sol["airline"][net["physical"]].sum()),
            open_connectors=int(sol["airline"][~net["physical"]].sum()),
            airline_fixed_cost=float(net["fees"] @ sol["airline"]),
            all_ground_cost=c0,
            provenance=data.get("provenance", {}),
            source_sha256=source_sha256,
            _solution=sol,
            _missions=best["missions"],
            _relocations=best["relocations"],
        )
        if stage is not None:
            result["checkpoint_stage"] = stage
        return result

    def save(stage, reason="running"):
        if checkpoint is not None:
            checkpoint(result_snapshot(reason, stage))

    def emit(stage, **extra):
        nonlocal last_report
        last_report = time.perf_counter()
        if progress is None:
            return
        ss = best["solution"]
        progress(
            dict(
                stage=stage,
                elapsed=last_report - started,
                objective=float(best["objective"]),
                bound=float(lower),
                gap=max(0.0, best["objective"] - lower)
                / max(1.0, abs(best["objective"])),
                fleet=float(ss["fleet"]),
                served=float(best["served"]),
                rho=ss["rho"].tolist(),
                opened=dict(
                    L=int(ss["facility"][:L].sum()),
                    H=int(ss["facility"][L : L + H].sum()),
                    E=int(ss["facility"][L + H :].sum()),
                ),
                cabinet_total=float(ss["hub_count"].sum()),
                stats=stats.copy(),
                **extra,
            )
        )

    def master_progress(mm, where):
        nonlocal lower
        if requested():
            mm.terminate()
        if where == GRB.Callback.MIP:
            bound = float(mm.cbGet(GRB.Callback.MIP_OBJBND))
            if np.isfinite(bound) and abs(bound) < 1e90:
                lower = max(lower, bound)
        if time.perf_counter() - last_report >= 30.0:
            emit("master_search", callback_where=int(where))
            save("master_search")

    def recourse_progress(event):
        emit(
            event["stage"],
            cg_iteration=event["cg_iteration"],
            callback_where=event["callback_where"],
        )
        save(event["stage"])

    emit("master_ready")
    save("master_ready")

    def keep(ans, ff, zz, bb):
        nonlocal best
        if ans["primal_residual"] > 1e-5:
            return
        ff = ff.copy()
        zz = zz.copy()
        chosen = np.zeros(net["count"])
        used = set()
        for p in ans["missions"] + ans["relocations"]:
            for e, _ in incidence(net, p["trace"]):
                chosen[e] = 1
                used.update(net["uv"][e])
        if fixed_capacity_level is None:
            zz[:] = 0
            for i, load in enumerate(ans["hub_load"]):
                needs_open = (L + i in used) or (
                    fixed_facility is not None and fixed_facility[L + i]
                )
                if needs_open:
                    eligible = np.flatnonzero(capacities[i] >= load - 1e-7)
                    if not len(eligible):
                        return
                    zz[i, eligible[np.argmin(level_cost[i, eligible])]] = 1
            if fixed_facility is None:
                ff[L : L + H] = zz.sum(axis=1)
        if fixed_facility is None:
            for k in range(L + H, net["N"]):
                ff[k] = int(k in used)
            assigned = bb.sum(axis=(0, 2)) + bb.sum(axis=(0, 1))
            for l in range(L):
                ff[l] = int(l in used or assigned[l] > 0.5 or ans["rho"][l] > 0.0)
        if np.any(chosen > ff[net["uv"][:, 0]]) or np.any(chosen > ff[net["uv"][:, 1]]):
            return
        value = float(
            ans["objective"]
            + fixed @ ff
            + (level_cost * zz).sum()
            + net["fees"] @ chosen
        )
        if value < best["objective"] - 1e-8:
            sol = dict(
                facility=ff,
                capacity_level=zz,
                hub_count=zz @ sizes,
                assignment=bb.copy(),
                airline=chosen,
                rho=ans["rho"],
                idle=ans["idle"],
                empty=ans["empty"],
                fleet=ans["fleet"],
            )
            best = dict(
                objective=value,
                recourse=ans["objective"],
                solution=sol,
                **{
                    k: ans[k]
                    for k in (
                        "missions",
                        "relocations",
                        "hub_load",
                        "airline_load",
                        "served",
                        "primal_residual",
                    )
                },
            )
            save("incumbent_improved")

    def evaluate(ff, zz, bb, yy, budget):
        nonlocal pools
        stats["recourse_calls"] += 1
        sub_started = time.perf_counter()
        try:
            ans = recourse(
                data,
                ff,
                zz,
                bb,
                yy,
                time_limit=None if budget is None else max(0.1, budget),
                threads=threads,
                seed=seed,
                max_iterations=None if max_iterations is None else 300,
                initial_pool=pools,
                initial_oracle=allopen,
                stop_requested=stop_requested,
                progress=(
                    recourse_progress
                    if progress is not None or checkpoint is not None
                    else None
                ),
            )
        except RecourseStopped:
            stats["recourse_wall"] += time.perf_counter() - sub_started
            stats["interrupted_subproblems"] += 1
            raise
        stats["recourse_wall"] += ans["wall"]
        stats["pricing_calls"] += ans["pricing_calls"]
        stats["pricing_wall"] += ans["pricing_wall"]
        stats["cg_iterations"] += ans["cg_iterations"]
        stats["generated_columns"] += ans["pool_size"]
        stats["complete_subproblems"] += int(ans["cg_complete"])
        pools = ans["pool"]
        keep(ans, ff, zz, bb)
        if ans["cut_valid"]:
            model.addConstr(
                zeta
                >= ans["cut_intercept"]
                + ans["cut_facility"] @ x
                + (ans["cut_level"] * z).sum()
                + (ans["cut_B"] * alpha).sum()
                + ans["cut_airline"] @ y
            )
            stats["benders_cuts"] += 1
        return ans

    try:
        if requested():
            raise RecourseStopped("user_stop")
        # Obtain a nontrivial feasible plan before the main Benders loop.
        ff = np.ones(net["N"]) if fixed_facility is None else fixed_facility.copy()
        zz = np.zeros_like(capacities)
        if fixed_capacity_level is None:
            for i in range(H):
                if ff[L + i]:
                    zz[i, int(np.argmax(capacities[i]))] = 1
        else:
            zz = fixed_capacity_level.copy()
            ff[L : L + H] = zz.sum(axis=1)
        yy = ff[net["uv"][:, 0]] * ff[net["uv"][:, 1]]
        q = allopen if np.all(ff == 1) else price_design(data, facility=ff)
        alternatives = []
        for same_base in (True, False):
            costs = q["cost"].copy()
            if same_base:
                costs[:, ~np.eye(L, dtype=bool)] = np.inf
            flat = costs.reshape(D, L * L)
            which = flat.argmin(axis=1)
            bb = np.zeros((D, L, L))
            use = flat[np.arange(D), which] < 0
            bb[np.arange(D)[use], which[use] // L, which[use] % L] = 1
            if not any(np.array_equal(bb, old) for old in alternatives):
                alternatives.append(bb)
        for bb in alternatives:
            if requested():
                raise RecourseStopped("user_stop")
            remaining = deadline - time.perf_counter()
            if remaining < 0.2:
                break
            ans = evaluate(
                ff,
                zz,
                bb,
                yy,
                (
                    None
                    if time_limit is None
                    else min(remaining, max(3.0, time_limit * 0.2))
                ),
            )
            trace.append(
                dict(
                    stage="initial_design",
                    elapsed=time.perf_counter() - started,
                    upper=best["objective"],
                    lower=lower,
                    complete=ans["cg_complete"],
                    subproblem_gap=ans["duality_gap"],
                    columns=ans["pool_size"],
                )
            )
            emit(
                "initial_design", complete=ans["cg_complete"], columns=ans["pool_size"]
            )
            save("initial_design")
        iterations = (
            itertools.count() if max_iterations is None else range(max_iterations)
        )
        for iteration in iterations:
            if requested():
                raise RecourseStopped("user_stop")
            remaining = deadline - time.perf_counter()
            if remaining <= 0.1:
                termination = "time_limit"
                break
            sol = best["solution"]
            x.Start = sol["facility"]
            z.Start = sol["capacity_level"]
            alpha.Start = sol["assignment"]
            y.Start = sol["airline"]
            zeta.Start = best["recourse"]
            if time_limit is not None:
                model.Params.TimeLimit = remaining
            emit("master_start", iteration=iteration)
            if progress is None and checkpoint is None and stop_requested is None:
                model.optimize()
            else:
                model.optimize(master_progress)
            status = int(model.Status)
            master_runtime += float(model.Runtime)
            nodes += float(model.NodeCount)
            if abs(model.ObjBound) < 1e90:
                lower = max(lower, float(model.ObjBound))
            save("master_finished")
            if requested():
                raise RecourseStopped("user_stop")
            if best["objective"] - lower <= max(
                params["MIPGapAbs"], params["MIPGap"] * abs(best["objective"])
            ):
                termination = "gap_limit"
                break
            if not model.SolCount:
                termination = "master_no_incumbent"
                break
            if time.perf_counter() >= deadline:
                termination = "time_limit"
                break
            ff = np.rint(x.X)
            zz = np.rint(z.X)
            bb = np.rint(alpha.X)
            yy = np.rint(y.X)
            key = hashlib.sha256(
                np.r_[ff, zz.ravel(), bb.ravel(), yy].astype(np.uint8).tobytes()
            ).hexdigest()
            emit("subproblem_start", iteration=iteration)
            ans = evaluate(
                ff,
                zz,
                bb,
                yy,
                None if time_limit is None else deadline - time.perf_counter(),
            )
            trace.append(
                dict(
                    stage="integer_design",
                    iteration=iteration,
                    design_hash=key,
                    elapsed=time.perf_counter() - started,
                    upper=best["objective"],
                    lower=lower,
                    complete=ans["cg_complete"],
                    subproblem_gap=ans["duality_gap"],
                    columns=ans["pool_size"],
                    cut_correction=ans["cut_correction"],
                )
            )
            emit(
                "integer_design",
                iteration=iteration,
                complete=ans["cg_complete"],
                columns=ans["pool_size"],
            )
            save("integer_design")
            if not ans["cut_valid"]:
                errors.append("invalid_recourse_certificate")
                termination = "recourse_failure"
                break
            if not ans["cg_complete"] and key in evaluated:
                termination = "incomplete_recourse"
                break
            evaluated[key] = ans["bound"]
    except RecourseStopped:
        termination = "user_stop"
        status = GRB.INTERRUPTED
    finally:
        model.dispose()
    emit("finished", termination_reason=termination)
    save("finished", termination)
    return result_snapshot(termination)
