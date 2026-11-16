"""Finite-catalogue scalar CG and certified global SOCP Benders cuts."""

# ruff: noqa: E741 -- l and m denote departure and return bases.
from __future__ import annotations
from dataclasses import asdict
from contextlib import contextmanager
import time
import math
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from src.model.types import MODEL_VERSION
from src.model.assignment import assignment_scope
from .formulation import (
    DESIGN_NAMES,
    BuildStopped,
    add_design,
    add_path_columns,
    build_model,
    read_solution,
    shapes,
    setup,
)
from .dual import solve_dual, price_catalogue, certified_cut, cut_value


def baseline(data, paths=None):
    solution = {name: np.zeros(shape) for name, shape in shapes(data).items()}
    segments = () if paths is None else paths.initial_segments
    service = () if paths is None else paths.service_paths
    empty = () if paths is None else paths.empty_paths
    solution.update(
        rho=np.zeros(data.n_l),
        idle=np.zeros(data.n_l),
        fleet=np.array(0.0),
        head=np.zeros(len(segments)),
        eta=np.zeros(len(segments)),
        flow=np.zeros(len(service)),
        empty_flow=np.zeros(len(empty)),
    )
    solution["courier"] = data.demand.copy()
    cost = float(data.demand @ data.ground_cost)
    return dict(
        model_version=MODEL_VERSION,
        flow_policy=data.flow_policy,
        assignment_scope=assignment_scope(data),
        objective=cost,
        continuous_cost=cost,
        opening_cost=0.0,
        bound=0.0,
        gap=cost / max(1.0, abs(cost)),
        termination_reason="all_courier_baseline",
        pricing_complete=False,
        incumbent_kind="all_courier_baseline",
        _solution=solution,
        _segments=[asdict(s) for s in segments],
        _missions=[
            dict(path=dict(asdict(p), id=i), flow=0.0) for i, p in enumerate(service)
        ],
        _relocations=[
            dict(path=dict(asdict(p), id=i), flow=0.0) for i, p in enumerate(empty)
        ],
    )


def _remaining(started, seconds):
    return (
        None if seconds is None else max(0.0, seconds - (time.perf_counter() - started))
    )


def _valid_primal(data, result):
    """Only independently checked physical solutions may update an upper bound."""
    from src.operations.validation import build_operations

    try:
        passed = build_operations(data, result)["summary"]["checks"]["passed"]
    except (ValueError, TypeError, KeyError, IndexError):
        return False
    if passed:
        result["independently_validated"] = True
    return passed


def optimize(model, *, seconds=None, stop_requested=None, progress=None, timings=None):
    if stop_requested and stop_requested():
        raise BuildStopped("user_stop")
    if seconds is not None:
        if seconds <= 0:
            raise BuildStopped("time_limit")
        model.Params.TimeLimit = max(1e-6, seconds)

    last_progress = float("-inf")

    def callback(m, where):
        nonlocal last_progress
        if stop_requested and stop_requested():
            m.terminate()
        if progress and where == GRB.Callback.MIP:
            now = time.perf_counter()
            if now - last_progress >= 1.0:
                last_progress = now
                best = float(m.cbGet(GRB.Callback.MIP_OBJBST))
                bound = float(m.cbGet(GRB.Callback.MIP_OBJBND))
                progress(
                    dict(
                        stage="mip_search",
                        solver_incumbent=best if abs(best) < GRB.INFINITY else None,
                        solver_bound=bound if abs(bound) < GRB.INFINITY else None,
                        node_count=float(m.cbGet(GRB.Callback.MIP_NODCNT)),
                    )
                )

    started = time.perf_counter()
    try:
        model.optimize(callback if stop_requested or progress else None)
    finally:
        if timings is not None:
            timings["optimize"] = (
                timings.get("optimize", 0.0) + time.perf_counter() - started
            )
            timings["calls"] = timings.get("calls", 0) + 1
            try:
                runtime = float(model.Runtime)
            except (AttributeError, gp.GurobiError):
                runtime = 0.0
            timings["solver_runtime"] = timings.get("solver_runtime", 0.0) + runtime


def _reason(status):
    return {
        GRB.OPTIMAL: "gap_limit",
        GRB.TIME_LIMIT: "time_limit",
        GRB.INTERRUPTED: "user_stop",
        GRB.NUMERIC: "numerical_failure",
        GRB.INFEASIBLE: "infeasible",
        GRB.INF_OR_UNBD: "infeasible_or_unbounded",
    }.get(status, "solver_stopped")


def _checked_bound(candidate, incumbent, previous=0.0):
    """Reject a material bound/incumbent contradiction instead of hiding it."""
    if not math.isfinite(candidate):
        return previous, False, 0.0
    tolerance = 1e-7 * (1.0 + abs(incumbent))
    if candidate > incumbent + tolerance:
        return previous, True, 0.0
    adjustment = max(0.0, candidate - incumbent)
    return max(0.0, previous, min(candidate, incumbent)), False, adjustment


@contextmanager
def _phase(timings, name):
    started = time.perf_counter()
    try:
        yield
    finally:
        timings[name] = timings.get(name, 0.0) + time.perf_counter() - started


def _merge_timings(target, source):
    """Accumulate nested timing diagnostics without flattening parent totals."""
    for key, value in source.items():
        if isinstance(value, dict):
            _merge_timings(target.setdefault(key, {}), value)
        else:
            target[key] = target.get(key, 0) + value


def solve_subproblem(
    data,
    paths,
    design,
    *,
    time_limit=None,
    threads=1,
    seed=43,
    column_generation=True,
    initial_service=(),
    initial_empty=(),
    stop_requested=None,
    progress=None,
    pareto_core=None,
    pareto_time_limit=15.0,
):
    """Solve one design with incremental columns and fully certified cuts.

    The ordinary cut is always retained before an optional MW auxiliary solve.
    Auxiliary timeout or certificate failure cannot discard that valid cut.
    """
    started = time.perf_counter()
    selected_p = (
        set(initial_service)
        if column_generation
        else set(range(len(paths.service_paths)))
    )
    selected_k = (
        set(initial_empty) if column_generation else set(range(len(paths.empty_paths)))
    )
    best, cut, ordinary, pricing, iterations = None, None, None, None, 0
    reason, status, built = "time_limit", None, None
    pareto = dict(attempted=False, accepted=False, reason="disabled")
    timings = dict(
        build=0.0,
        primal=0.0,
        dual=0.0,
        pricing=0.0,
        add_columns=0.0,
        pareto=0.0,
        read_solution=0.0,
        validation=0.0,
        certificate=0.0,
        dispose=0.0,
    )
    timing_details = dict(primal_solver={}, dual={}, pareto={})

    def check():
        if stop_requested and stop_requested():
            raise BuildStopped("user_stop")
        remaining = _remaining(started, time_limit)
        if remaining is not None and remaining <= 0:
            raise BuildStopped("time_limit")

    try:
        check()
        with _phase(timings, "build"):
            built = build_model(
                data,
                paths,
                fixed_design=design,
                service_indices=selected_p,
                empty_indices=selected_k,
                time_limit=_remaining(started, time_limit),
                threads=threads,
                seed=seed,
                stop_requested=stop_requested,
            )
        while True:
            check()
            iterations += 1
            if progress:
                progress(
                    dict(
                        stage="restricted_model_built",
                        cg_iteration=iterations,
                        variables=built.model.NumVars,
                        constraints=built.model.NumConstrs,
                        cones=built.model.NumQConstrs,
                        service_columns=len(selected_p),
                        incremental=iterations > 1,
                    )
                )
            with _phase(timings, "primal"):
                optimize(
                    built.model,
                    seconds=_remaining(started, time_limit),
                    stop_requested=stop_requested,
                    timings=timing_details["primal_solver"],
                )
            status, reason = int(built.model.Status), _reason(built.model.Status)
            if not built.model.SolCount:
                break
            with _phase(timings, "read_solution"):
                trial = read_solution(built)
            with _phase(timings, "validation"):
                valid = _valid_primal(data, trial)
            if not valid:
                reason = "primal_validation_failed"
                break
            if best is None or trial["objective"] < best["objective"]:
                best = trial
            with _phase(timings, "dual"):
                dual = solve_dual(
                    built,
                    time_limit=_remaining(started, time_limit),
                    threads=threads,
                    seed=seed,
                    stop_requested=stop_requested,
                    timings=timing_details["dual"],
                )
            if dual is None:
                reason = "dual_unavailable"
                break
            with _phase(timings, "pricing"):
                pricing = price_catalogue(
                    built,
                    dual,
                    time_limit=_remaining(started, time_limit),
                    stop_requested=stop_requested,
                )
            if progress:
                progress(
                    dict(
                        stage="column_generation",
                        iteration=iterations,
                        service_columns=len(selected_p),
                        empty_columns=len(selected_k),
                        pricing_complete=pricing["complete"],
                        minimum_reduced_cost=pricing.get("minimum_reduced_cost"),
                    )
                )
            if not pricing["complete"]:
                reason = pricing["reason"]
                break
            with _phase(timings, "certificate"):
                cut = certified_cut(dual, pricing)
            if cut is not None:
                anchor = cut_value(cut, design)
                if anchor > best["continuous_cost"] + 1e-5 * (
                    1 + best["continuous_cost"]
                ):
                    raise RuntimeError(
                        "The independently constructed dual exceeds the primal cost"
                    )
                cut["anchor"] = anchor
                cut["duality_gap"] = best["continuous_cost"] - anchor
                ordinary = cut
                reason = "pricing_complete"
                if pareto_core is not None:
                    from .pareto import solve_pareto_cut

                    remaining = _remaining(started, time_limit)
                    budget = (
                        pareto_time_limit
                        if remaining is None
                        else min(pareto_time_limit, remaining)
                    )
                    begin = time.perf_counter()
                    try:
                        pareto = solve_pareto_cut(
                            built,
                            ordinary,
                            design,
                            pareto_core,
                            best["continuous_cost"],
                            time_limit=max(0.0, budget),
                            threads=threads,
                            seed=seed,
                            stop_requested=stop_requested,
                            timings=timing_details["pareto"],
                        )
                        cut = pareto["cut"]
                        pareto = {k: v for k, v in pareto.items() if k != "cut"}
                        pareto["attempted"] = True
                    except BuildStopped as exc:
                        pareto = dict(attempted=True, accepted=False, reason=exc.reason)
                    timings["pareto"] += time.perf_counter() - begin
                break
            with _phase(timings, "add_columns"):
                count = add_path_columns(
                    built,
                    (i for _, i in pricing["service"]),
                    (i for _, i in pricing["empty"]),
                    check=check,
                )
            selected_p = set(built.service_indices)
            selected_k = set(built.empty_indices)
            if count == 0:
                reason = "pricing_stalled"
                break
    except BuildStopped as exc:
        reason = exc.reason
    finally:
        if built is not None:
            selected_p = set(built.service_indices)
            selected_k = set(built.empty_indices)
            with _phase(timings, "dispose"):
                built.model.dispose()
    return dict(
        solution=best,
        cut=cut,
        ordinary_cut=ordinary,
        pareto=pareto,
        pricing=pricing,
        pricing_complete=cut is not None,
        termination_reason=reason,
        status=status,
        iterations=iterations,
        timings=timings,
        timing_details=timing_details,
        wall=time.perf_counter() - started,
        service_columns=tuple(sorted(selected_p)),
        empty_columns=tuple(sorted(selected_k)),
    )


def solve_full_path(
    data,
    paths,
    *,
    time_limit=None,
    threads=1,
    seed=43,
    mip_gap=1e-4,
    stop_requested=None,
    progress=None,
    checkpoint=None,
):
    started = time.perf_counter()
    result = baseline(data, paths)
    try:
        built = build_model(
            data,
            paths,
            time_limit=time_limit,
            threads=threads,
            seed=seed,
            mip_gap=mip_gap,
            stop_requested=stop_requested,
        )
        try:
            result["model_statistics"] = dict(
                variables=built.model.NumVars,
                constraints=built.model.NumConstrs,
                cones=built.model.NumQConstrs,
                build_seconds=time.perf_counter() - started,
            )
            if progress:
                progress(
                    dict(
                        stage="full_path_built",
                        **result["model_statistics"],
                    )
                )
            optimize(
                built.model,
                seconds=_remaining(started, time_limit),
                stop_requested=stop_requested,
                progress=progress,
            )
            reason = _reason(built.model.Status)
            if built.model.SolCount:
                trial = read_solution(built)
                if not _valid_primal(data, trial):
                    reason = "primal_validation_failed"
                elif trial["objective"] < result["objective"] + 1e-8:
                    result.update(trial, incumbent_kind="optimized")
            raw_bound = float(built.model.ObjBound)
            if built.model.Status == GRB.NUMERIC:
                raw_bound = 0.0
            bound, inconsistent, adjustment = _checked_bound(
                raw_bound, result["objective"]
            )
            if inconsistent:
                reason = "numerical_bound_inconsistency"
            result.update(
                bound=bound,
                bound_numerical_adjustment=adjustment,
                gap=_gap(result["objective"], bound),
                status=int(built.model.Status),
                termination_reason=reason,
                pricing_complete=True,
            )
            if checkpoint:
                checkpoint(result)
        finally:
            built.model.dispose()
    except BuildStopped as exc:
        result["termination_reason"] = exc.reason
        result["construction_complete"] = False
    return result


def _gap(upper, lower):
    return max(0.0, upper - lower) / max(1.0, abs(upper))


def solve_benders(
    data,
    paths,
    *,
    time_limit=None,
    threads=1,
    seed=43,
    mip_gap=1e-4,
    stop_requested=None,
    progress=None,
    checkpoint=None,
    accelerated=True,
    accelerations=None,
    max_iterations=1000,
):
    from .methods import resolve_accelerations
    from .heuristics import initial_design, cleanup_incumbent, design_neighbors
    from .pareto import CorePoint
    from .strengthening import master_bounds
    from src.model.locker_capacity import occupancy_limits

    options = resolve_accelerations(
        "benders_cg" if accelerated else "benders_cg_plain", accelerations
    )
    started = time.perf_counter()
    algorithm_timings = {}
    master_solver_timings = {}
    with _phase(algorithm_timings, "baseline"):
        result = baseline(data, paths)
    result["accelerations"] = options
    result["master_bound_metadata"] = dict(
        enabled=False,
        reason="not_built" if options["courier_savings_bound"] else "disabled",
    )
    if (time_limit is not None and time_limit <= 0) or (
        stop_requested and stop_requested()
    ):
        result.update(
            termination_reason="user_stop"
            if stop_requested and stop_requested()
            else "time_limit",
            construction_complete=False,
        )
        return result
    with _phase(algorithm_timings, "master_build"):
        master = gp.Model(f"{MODEL_VERSION}_master")
        setup(master, threads, seed, mip_gap)
        variables, strategy = add_design(master, data)
        theta = master.addVar(ub=result["objective"], name="operating_cost")
        master.setObjective(strategy + theta)
        if options["courier_savings_bound"]:
            bounds = master_bounds(data, paths)
            savings = bounds["perassignment_savings"]
            beta = bounds["locker_value_per_load"]
            result["master_bound_metadata"] = dict(enabled=True, **bounds["metadata"])
            cost_floor = result["objective"] - bounds["metadata"]["numerical_margin"]
            master.addConstr(
                theta
                >= cost_floor
                - gp.quicksum(
                    data.demand[key[0]] * savings[key] * variables["assignment"][key]
                    for key in np.ndindex(savings.shape)
                ),
                name="courier_savings_bound",
            )
            if beta is not None:
                thresholds = occupancy_limits(
                    data.slot_counts, data.slot_epsilon[:, None]
                )
                master.addConstr(
                    theta
                    >= cost_floor
                    - beta
                    * gp.quicksum(
                        thresholds[i, s] * variables["capacity_level"][i, s]
                        for i, s in np.ndindex(data.slot_counts.shape)
                    ),
                    name="locker_savings_bound",
                )
        core = CorePoint(data) if options["pareto_cuts"] else None
    lower, cuts, records, pool_p, pool_k = 0.0, 0, [], set(), set()
    seen_p, seen_k = set(), set()
    reason, complete = "iteration_limit", False
    stats = dict(
        ordinary_cuts=0,
        pareto_cuts=0,
        pareto_attempts=0,
        pareto_rejections={},
        ordinary_cg_iterations=0,
        heuristic_cg_iterations=0,
        pareto_auxiliary_solves=0,
        pareto_auxiliary_calls=0,
        initial_design_attempts=0,
        primal_search_attempts=0,
        primal_improvements=0,
        cleanup_improvements=0,
        cleanup_savings=0.0,
        master_starts=0,
        reused_service_columns=0,
        reused_empty_columns=0,
        unique_service_columns=0,
        unique_empty_columns=0,
        subproblem_timings={},
        subproblem_timing_details={},
        subproblem_wall=0.0,
        algorithm_timings=algorithm_timings,
        master_solver_timings=master_solver_timings,
        pareto_seconds=0.0,
        core_gain_total=0.0,
    )
    heuristic_records = []
    last_primal_design = None

    def keep_cut(cut, kind):
        nonlocal cuts
        if cut is None:
            return
        with _phase(algorithm_timings, "master_update"):
            expression = cut["intercept"] + gp.quicksum(
                cut["coefficients"][name][key] * variable
                for name in DESIGN_NAMES
                for key, variable in variables[name].items()
            )
            master.addConstr(theta >= expression, name=f"{kind}_cut_{cuts}")
        cuts += 1
        stats[kind + "_cuts"] += 1

    def update_primal(trial):
        nonlocal result
        if trial is None:
            return False
        with _phase(algorithm_timings, "incumbent_validation"):
            valid = _valid_primal(data, trial)
        if not valid:
            return False
        if options["incumbent_cleanup"]:
            with _phase(algorithm_timings, "incumbent_cleanup"):
                cleaned = cleanup_incumbent(data, trial)
            if cleaned["objective"] < trial["objective"] - 1e-8:
                stats["cleanup_improvements"] += 1
                stats["cleanup_savings"] += trial["objective"] - cleaned["objective"]
                trial = cleaned
        if trial["objective"] >= result["objective"] - 1e-8:
            return False
        _, inconsistent, _ = _checked_bound(lower, trial["objective"])
        if inconsistent:
            raise RuntimeError(
                "A validated primal contradicts the certified master bound"
            )
        result.update(trial, incumbent_kind="optimized")
        stats["primal_improvements"] += 1
        return True

    def account(sub, heuristic=False):
        nonlocal complete
        stats["heuristic_cg_iterations" if heuristic else "ordinary_cg_iterations"] += (
            sub["iterations"]
        )
        for key, value in sub.get("timings", {}).items():
            stats["subproblem_timings"][key] = (
                stats["subproblem_timings"].get(key, 0.0) + value
            )
        _merge_timings(
            stats["subproblem_timing_details"], sub.get("timing_details", {})
        )
        stats["subproblem_wall"] += sub.get("wall", 0.0)
        seen_p.update(sub["service_columns"])
        seen_k.update(sub["empty_columns"])
        stats["unique_service_columns"] = len(seen_p)
        stats["unique_empty_columns"] = len(seen_k)
        if options["reuse_columns"]:
            pool_p.update(sub["service_columns"])
            pool_k.update(sub["empty_columns"])
        update_primal(sub["solution"])
        ordinary = sub.get("ordinary_cut") or sub["cut"]
        keep_cut(ordinary, "ordinary")
        complete = sub["pricing_complete"]
        auxiliary = sub.get("pareto", {})
        if auxiliary.get("attempted"):
            stats["pareto_attempts"] += 1
            stats["pareto_seconds"] += sub.get("timings", {}).get("pareto", 0.0)
            if auxiliary.get("auxiliary_called"):
                stats["pareto_auxiliary_calls"] += 1
            if auxiliary.get("auxiliary_status") is not None:
                stats["pareto_auxiliary_solves"] += 1
            if auxiliary.get("accepted"):
                keep_cut(sub["cut"], "pareto")
                stats["core_gain_total"] += auxiliary.get("core_gain", 0.0)
            else:
                why = auxiliary.get("reason", "unknown")
                stats["pareto_rejections"][why] = (
                    stats["pareto_rejections"].get(why, 0) + 1
                )
        if sub["termination_reason"] == "model_size_limit":
            result["construction_complete"] = False
            raise BuildStopped("model_size_limit")

    def subproblem(design, budget, *, heuristic=False):
        if options["reuse_columns"]:
            stats["reused_service_columns"] += len(pool_p)
            stats["reused_empty_columns"] += len(pool_k)
        return solve_subproblem(
            data,
            paths,
            design,
            time_limit=budget,
            threads=threads,
            seed=seed,
            stop_requested=stop_requested,
            progress=progress,
            initial_service=pool_p if options["reuse_columns"] else (),
            initial_empty=pool_k if options["reuse_columns"] else (),
            pareto_core=core.point if core is not None and not heuristic else None,
            pareto_time_limit=15.0
            if budget is None
            else min(15.0, max(0.0, 0.2 * budget)),
        )

    def snapshot():
        result.update(
            bound=min(lower, result["objective"]),
            gap=_gap(result["objective"], lower),
            certified_cuts=cuts,
            pricing_complete=complete,
            acceleration_statistics=stats,
            acceleration_records=heuristic_records,
            iterations=records,
        )
        if checkpoint:
            with _phase(algorithm_timings, "checkpoint"):
                checkpoint(result)

    def heuristic(design, stage, maximum, fraction):
        remaining = _remaining(started, time_limit)
        budget = (
            maximum
            if remaining is None
            else min(maximum, max(0.0, fraction * remaining))
        )
        if budget <= 0:
            return
        stats[stage + "_attempts"] += 1
        if progress:
            progress(dict(stage=stage, budget_seconds=budget))
        before = result["objective"]
        sub = subproblem(design, budget, heuristic=True)
        account(sub, heuristic=True)
        heuristic_records.append(
            dict(
                stage=stage,
                before=before,
                after=result["objective"],
                termination_reason=sub["termination_reason"],
                pricing_complete=sub["pricing_complete"],
                timings=sub["timings"],
            )
        )
        snapshot()

    try:
        if options["initial_design"]:
            with _phase(algorithm_timings, "initial_design"):
                proposal = initial_design(data, paths)
            heuristic(proposal, "initial_design", 30.0, 0.15)
        for iteration in range(max_iterations):
            if options["warm_start"]:
                with _phase(algorithm_timings, "master_update"):
                    incumbent = result["_solution"]
                    for name in DESIGN_NAMES:
                        for key, variable in variables[name].items():
                            variable.Start = float(incumbent[name][key])
                    theta.Start = float(result["continuous_cost"])
                    stats["master_starts"] += 1
            optimize(
                master,
                seconds=_remaining(started, time_limit),
                stop_requested=stop_requested,
                timings=master_solver_timings,
            )
            if master.Status == GRB.NUMERIC:
                reason = "numerical_failure"
                break
            candidate, inconsistent, adjustment = _checked_bound(
                float(master.ObjBound), result["objective"], lower
            )
            if inconsistent:
                reason = "numerical_bound_inconsistency"
                break
            lower = candidate
            if not master.SolCount:
                reason = _reason(master.Status)
                break
            with _phase(algorithm_timings, "master_update"):
                design = {name: np.zeros(shape) for name, shape in shapes(data).items()}
                for name in DESIGN_NAMES:
                    for key, value in variables[name].items():
                        design[name][key] = round(value.X)
            if _gap(result["objective"], lower) <= mip_gap:
                reason = "gap_limit"
                break
            sub = subproblem(design, _remaining(started, time_limit))
            account(sub)
            record = dict(
                iteration=iteration + 1,
                objective=result["objective"],
                bound=lower,
                pricing_complete=sub["pricing_complete"],
                cg_iterations=sub["iterations"],
                service_columns=len(sub["service_columns"]),
                empty_columns=len(sub["empty_columns"]),
                pareto=sub["pareto"],
                timings=sub["timings"],
                certified_cuts=cuts,
            )
            records.append(record)
            if progress:
                progress(dict(stage="benders", **record))
            snapshot()
            if sub["cut"] is None:
                reason = sub["termination_reason"]
                break
            if core is not None:
                with _phase(algorithm_timings, "master_update"):
                    core.update(design)
            if _gap(result["objective"], lower) <= mip_gap:
                reason = "gap_limit"
                break
            if options["primal_search"] and (iteration + 1) % 5 == 0:
                # Search a new incumbent only once; all neighbors require a new
                # fixed-design solve and independent primal/cut certification.
                signature = tuple(
                    result["_solution"][n].tobytes() for n in DESIGN_NAMES
                )
                if signature != last_primal_design:
                    last_primal_design = signature
                    with _phase(algorithm_timings, "neighbor_generation"):
                        neighbors = design_neighbors(
                            data, paths, result, max_neighbors=1
                        )
                    for neighbor in neighbors:
                        heuristic(neighbor, "primal_search", 10.0, 0.1)
    except BuildStopped as exc:
        reason = exc.reason
    finally:
        with _phase(algorithm_timings, "dispose"):
            master.dispose()
    result.update(
        bound=min(lower, result["objective"]),
        gap=_gap(result["objective"], lower),
        certified_cuts=cuts,
        iterations=records,
        termination_reason=reason,
        pricing_complete=complete,
        acceleration_statistics=stats,
        acceleration_records=heuristic_records,
        accelerations=options,
        algorithm_wall=time.perf_counter() - started,
    )
    return result
