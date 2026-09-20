"""Integer-master Benders with interchangeable CG pricing and explicit options."""

import hashlib
import itertools
import time
import numpy as np
from gurobipy import GRB
from src.solvers.pricing.backend import price_design, PricingStopped
from src.solvers.benders.master import build_master, add_initial_bound
from src.solvers.benders.subproblem import solve as recourse, RecourseStopped
from src.solvers.benders.state import BendersState
from src.solvers.benders import pareto
from src.solvers.benders.initialization import (
    fix_master,
    baseline,
    initial_designs,
    incumbent,
    load_initial,
)


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
    *,
    pricing="label",
    pareto_cuts=True,
    initial_bound=True,
    initial_design=True,
    reuse_columns=True,
    warm_start=True,
    incumbent_cleanup=True,
    valid_inequalities=False,
):
    """Solve the complete strategic model; options control accelerations only.

    Benders optimality cuts, SOC constraints, initial CG columns and complete
    candidate-graph cut certification are mandatory under every configuration.
    Extra master valid inequalities are not implemented.
    """
    if pricing not in {"label", "solver"}:
        raise ValueError("pricing must be 'label' or 'solver'")
    if valid_inequalities:
        raise NotImplementedError("Extra master valid inequalities are not implemented")
    started = time.perf_counter()
    deadline = float("inf") if time_limit is None else started + time_limit
    model, x, z, alpha, y, zeta = build_master(
        data,
        threads=threads,
        mip_gap=mip_gap,
        log=log,
        mip_gap_abs=mip_gap_abs,
        seed=seed,
        initial_bound=False,
    )
    try:
        fixed_facility, fixed_capacity_level = fix_master(
            data,
            x,
            z,
            fixed_facility,
            fixed_capacity_level,
        )
        best = baseline(data, fixed_facility, fixed_capacity_level)
        lower = 0.0
        if initial_result is not None:
            best, lower = load_initial(
                data, initial_result, best, fixed_facility, fixed_capacity_level
            )
    except Exception:
        model.dispose()
        raise
    model.update()
    params = dict(
        MIPGap=float(model.Params.MIPGap),
        MIPGapAbs=float(model.Params.MIPGapAbs),
        Threads=int(model.Params.Threads),
        Seed=int(model.Params.Seed),
        TimeLimit=time_limit,
        max_iterations=max_iterations,
        cg_max_iterations=None if max_iterations is None else 300,
        NodefileStart=float(model.Params.NodefileStart),
        NodefileDir=str(model.Params.NodefileDir),
    )
    features = dict(
        pricing=pricing,
        pareto_cuts=bool(pareto_cuts),
        initial_bound=bool(initial_bound),
        initial_design=bool(initial_design),
        reuse_columns=bool(reuse_columns),
        warm_start=bool(warm_start),
        incumbent_cleanup=bool(incumbent_cleanup),
        extra_valid_inequalities=[],
    )
    state = BendersState(
        data, best, started, time_limit, params, features, progress, checkpoint
    )
    state.lower = lower
    state.stats.update(
        master_binaries=int(model.NumBinVars),
        master_integers=int(model.NumIntVars),
        master_continuous=int(model.NumVars - model.NumIntVars),
    )
    pool = dict(missions=best["missions"], relocations=best["relocations"])
    core = (
        pareto.CorePoint(
            data,
            fixed_facility=fixed_facility,
            fixed_capacity_level=fixed_capacity_level,
        )
        if pareto_cuts
        else None
    )
    initial_prices = None
    evaluated = set()
    termination = "iteration_limit"

    def requested():
        return stop_requested is not None and bool(stop_requested())

    def remaining():
        return None if time_limit is None else deadline - time.perf_counter()

    def price(**kwargs):
        answer = price_design(
            data,
            backend=pricing,
            time_limit=remaining(),
            threads=threads,
            seed=seed,
            stop_requested=stop_requested,
            **kwargs,
        )
        state.stats["pricing_calls"] += 1
        state.stats["pricing_wall"] += answer.get("wall", 0.0)
        return answer

    def master_progress(mm, where):
        if requested():
            mm.terminate()
        if where == GRB.Callback.MIP:
            bound = float(mm.cbGet(GRB.Callback.MIP_OBJBND))
            if np.isfinite(bound) and abs(bound) < 1e90:
                state.lower = max(state.lower, bound)
        if time.perf_counter() - state.last_report >= 30.0:
            state.emit("master_search", callback_where=int(where))
            state.save("master_search")

    def recourse_progress(event):
        state.emit(
            event["stage"],
            cg_iteration=event["cg_iteration"],
            callback_where=event["callback_where"],
        )
        state.save(event["stage"])

    def add_cut(answer, prefix="benders"):
        model.addConstr(
            zeta
            >= answer["cut_intercept"]
            + answer["cut_facility"] @ x
            + (answer["cut_level"] * z).sum()
            + (answer["cut_B"] * alpha).sum()
            + answer["cut_airline"] @ y,
            name=f"{prefix}_{state.stats['benders_cuts']}",
        )
        state.stats["benders_cuts"] += 1

    def strengthen(design, answer, budget, sub_started):
        if core is None:
            return
        try:
            if not (answer["cut_valid"] and answer["cg_complete"]):
                return
            auxiliary_budget = 30.0
            if budget is not None:
                auxiliary_budget = min(
                    auxiliary_budget, budget - (time.perf_counter() - sub_started)
                )
            if remaining() is not None:
                auxiliary_budget = min(auxiliary_budget, remaining())
            if auxiliary_budget <= 0.1:
                return
            if requested():
                raise RecourseStopped("user_stop")
            state.stats["pareto_attempts"] += 1
            state.emit("pareto_start")
            aux_started = time.perf_counter()
            try:
                improved = pareto.strengthen(
                    data,
                    design,
                    core.point,
                    answer,
                    time_limit=auxiliary_budget,
                    threads=threads,
                    seed=seed,
                    pricing=pricing,
                    stop_requested=stop_requested,
                )
            except (RecourseStopped, PricingStopped) as error:
                state.stats["pareto_wall"] += time.perf_counter() - aux_started
                state.stats["pareto_interrupted"] += 1
                if requested() or str(error) == "user_stop":
                    raise
                if str(error) != "time_limit":
                    raise
                # The ordinary cut is already certified and remains available.
                state.emit("pareto_finished", accepted=False, reason="time_limit")
                return
            state.record_pareto(improved, time.perf_counter() - aux_started)
            accepted = improved.get("accepted", False) and improved.get("cut_valid", False)
            if accepted:
                add_cut(improved, prefix="pareto")
                state.stats["pareto_cuts"] += 1
            state.emit(
                "pareto_finished", accepted=bool(accepted), reason=improved.get("reason")
            )
        finally:
            core.update(*design)

    def evaluate(facility, cabinets, assignment, airlines, budget):
        nonlocal pool
        if budget is not None and budget <= 0:
            raise RecourseStopped("time_limit")
        state.stats["recourse_calls"] += 1
        sub_started = time.perf_counter()
        try:
            answer = recourse(
                data,
                facility,
                cabinets,
                assignment,
                airlines,
                time_limit=budget,
                threads=threads,
                seed=seed,
                max_iterations=params["cg_max_iterations"],
                initial_pool=pool if reuse_columns else None,
                initial_oracle=initial_prices if reuse_columns else None,
                pricing=pricing,
                warm_start=warm_start,
                stop_requested=stop_requested,
                progress=recourse_progress
                if progress is not None or checkpoint is not None
                else None,
            )
        except RecourseStopped:
            state.stats["recourse_wall"] += time.perf_counter() - sub_started
            state.stats["interrupted_subproblems"] += 1
            raise
        state.record_subproblem(answer)
        if reuse_columns:
            pool = answer["pool"]
        candidate = incumbent(
            data,
            answer,
            facility,
            cabinets,
            assignment,
            airlines,
            cleanup=incumbent_cleanup,
            fixed_facility=fixed_facility,
            fixed_capacity_level=fixed_capacity_level,
        )
        if (
            candidate is not None
            and candidate["objective"] < state.best["objective"] - 1e-8
        ):
            state.best = candidate
            state.save("incumbent_improved")
        if answer["cut_valid"]:
            add_cut(answer)
        strengthen((facility, cabinets, assignment, airlines), answer, budget, sub_started)
        return answer

    state.emit("master_ready")
    state.save("master_ready")
    try:
        if requested():
            raise RecourseStopped("user_stop")
        if initial_bound or initial_design:
            initial_prices = price()
        if initial_bound:
            add_initial_bound(model, data, initial_prices, alpha, zeta)
        if initial_design:
            prices = initial_prices
            mask = (
                np.ones(state.net["N"])
                if fixed_facility is None
                else fixed_facility.copy()
            )
            if fixed_capacity_level is not None:
                start, count = data["n_l"], data["n_h"]
                mask[start : start + count] = fixed_capacity_level.sum(axis=1)
            if not np.all(mask == 1):
                prices = price(facility=mask)
            for design in initial_designs(
                data, prices, fixed_facility, fixed_capacity_level
            ):
                budget = remaining()
                if budget is not None:
                    if budget <= 0.1:
                        break
                    budget = min(budget, max(3.0, time_limit * 0.2))
                try:
                    answer = evaluate(*design, budget)
                except RecourseStopped as error:
                    if str(error) != "time_limit" or time.perf_counter() >= deadline:
                        raise
                    state.emit("initial_design_limit")
                    break
                state.trace.append(
                    dict(
                        stage="initial_design",
                        elapsed=time.perf_counter() - started,
                        upper=state.best["objective"],
                        lower=state.lower,
                        complete=answer["cg_complete"],
                        subproblem_gap=answer["duality_gap"],
                        columns=answer["pool_size"],
                    )
                )
                state.emit(
                    "initial_design",
                    complete=answer["cg_complete"],
                    columns=answer["pool_size"],
                )
                state.save("initial_design")

        iterations = (
            itertools.count() if max_iterations is None else range(max_iterations)
        )
        for iteration in iterations:
            if requested():
                raise RecourseStopped("user_stop")
            budget = remaining()
            if budget is not None and budget <= 0.1:
                termination = "time_limit"
                break
            if warm_start:
                sol = state.best["solution"]
                x.Start, z.Start, alpha.Start = (
                    sol["facility"],
                    sol["capacity_level"],
                    sol["assignment"],
                )
                y.Start, zeta.Start = sol["airline"], state.best["recourse"]
            else:
                model.reset(1)
            if budget is not None:
                model.Params.TimeLimit = budget
            state.emit("master_start", iteration=iteration)
            if progress is None and checkpoint is None and stop_requested is None:
                model.optimize()
            else:
                model.optimize(master_progress)
            state.status = int(model.Status)
            state.master_runtime += float(model.Runtime)
            state.nodes += float(model.NodeCount)
            if abs(model.ObjBound) < 1e90:
                state.lower = max(state.lower, float(model.ObjBound))
            state.save("master_finished")
            if requested():
                raise RecourseStopped("user_stop")
            if state.best["objective"] - state.lower <= max(
                params["MIPGapAbs"], params["MIPGap"] * abs(state.best["objective"])
            ):
                termination = "gap_limit"
                break
            if not model.SolCount:
                termination = "master_no_incumbent"
                break
            if time.perf_counter() >= deadline:
                termination = "time_limit"
                break
            design = tuple(np.rint(var.X) for var in (x, z, alpha, y))
            key = hashlib.sha256(
                np.concatenate([v.ravel() for v in design]).astype(np.uint8).tobytes()
            ).hexdigest()
            state.emit("subproblem_start", iteration=iteration)
            answer = evaluate(*design, remaining())
            state.trace.append(
                dict(
                    stage="integer_design",
                    iteration=iteration,
                    design_hash=key,
                    elapsed=time.perf_counter() - started,
                    upper=state.best["objective"],
                    lower=state.lower,
                    complete=answer["cg_complete"],
                    subproblem_gap=answer["duality_gap"],
                    columns=answer["pool_size"],
                    cut_correction=answer["cut_correction"],
                )
            )
            state.emit(
                "integer_design",
                iteration=iteration,
                complete=answer["cg_complete"],
                columns=answer["pool_size"],
            )
            state.save("integer_design")
            if not answer["cut_valid"]:
                state.errors.append("invalid_recourse_certificate")
                termination = "recourse_failure"
                break
            if not answer["cg_complete"] and key in evaluated:
                termination = "incomplete_recourse"
                break
            evaluated.add(key)
    except (RecourseStopped, PricingStopped) as error:
        if requested() or str(error) == "user_stop":
            termination, state.status = "user_stop", GRB.INTERRUPTED
        elif str(error) == "time_limit":
            termination, state.status = "time_limit", GRB.TIME_LIMIT
        else:
            termination, state.status = "pricing_failure", GRB.NUMERIC
            state.errors.append(str(error))
    finally:
        model.dispose()
    state.emit("finished", termination_reason=termination)
    state.save("finished", termination)
    return state.snapshot(termination)
