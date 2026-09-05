"""Execute one immutable branch-and-cut profile with complete provenance."""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from src.config import SolverConfig as CoreSolverConfig

from .artifacts import AttemptClaim, write_canonical_json_exclusive
from .hashing import (
    build_attempt_id,
    build_logical_run_id,
    build_pair_id,
    build_profile_id,
    canonical_json,
    canonical_sha256,
)
from .formal_solution_evidence import (
    FormalQualificationTrace,
    FormalSolutionStructure,
    qualification_trace_from_context,
    recompute_solution_structure,
)
from .schema import PairSpec, ProfileConfig, TelemetryConfig
from .telemetry import (
    ArtifactPaths,
    GurobiParameterNameProvider,
    assert_fresh_artifact_paths,
    collect_applied_gurobi_parameters,
    collect_gurobi_parameter_audit,
    collect_solver_metrics,
    ensure_solver_log,
    exception_payload,
    write_telemetry_artifacts,
)


ModelFactory = Callable[[Any, str], object]
VersionProvider = Callable[[], str | tuple[int, ...]]


# Gurobi environments are not thread-safe, and their creation and destruction
# must not race.  Each benchmark attempt owns a private environment for its
# complete model lifecycle; only these short lifecycle operations are
# serialized, while models in distinct environments may optimize concurrently.
_GUROBI_ENVIRONMENT_LIFECYCLE_LOCK = threading.Lock()


@dataclass(frozen=True, slots=True)
class ProfileExecutionResult:
    """Structured return value for a completed or failed attempt."""

    completed_without_exception: bool
    has_incumbent: bool
    outcome: str
    attempt_id: str
    directory: Path
    run_record: Mapping[str, Any]
    failure: Mapping[str, Any] | None


def build_core_solver_config(
    pair: PairSpec,
    profile: ProfileConfig,
    telemetry: TelemetryConfig,
) -> CoreSolverConfig:
    """Translate the immutable experiment contract into the core solver API."""

    common = pair.solver
    static = profile.static
    cuts = profile.user_cuts
    start = profile.start
    config = CoreSolverConfig(
        time_limit=pair.time_budget_seconds,
        mip_gap=common.target_mip_gap,
        threads=common.threads,
        output_flag=common.output_flag,
        seed=pair.solver_seed,
        network_start_seconds=start.network_start_seconds,
        # The exact formal setting below takes precedence, but freezing the
        # f884 compatibility allowance to the same value keeps the serialized
        # core contract unambiguous.
        no_rel_heur_seconds=common.no_rel_heur_time,
        network_start_method=start.network_start_method,
        fixed_integer_repair_seconds=start.fixed_integer_repair_seconds,
        use_valid_inequalities=static.enabled,
        use_user_cuts=cuts.enabled,
        max_root_cut_rounds=cuts.max_root_rounds,
        # f884's bounded native-root policy is frozen in the matched solver
        # contract and is common to every A--E profile.
        root_cut_passes=common.root_cut_passes,
        max_cuts_per_round=cuts.max_cuts_per_round,
        cut_violation_tolerance=cuts.violation_tolerance,
        cut_efficacy_tolerance=cuts.efficacy_tolerance,
        use_local_branching=start.local_branching,
        local_branching_seconds=start.local_branching_seconds,
        local_branching_radius=start.local_branching_radius,
        local_branching_include_y=start.local_branching_include_topology,
        root_relaxation_method=common.method,
        miqcp_method=common.miqcp_method,
        start_mode=start.start_mode.value,
        # Development partial starts and rejected route/suffix families are
        # deliberately excluded from the current formal profile contract.
        use_objective_partial_start=False,
        objective_partial_start_seconds=None,
        objective_partial_start_max_candidates=3,
        mip_focus=common.mip_focus,
        heuristics=common.heuristics,
        no_rel_heur_time=common.no_rel_heur_time,
        presolve=common.presolve,
        cuts=common.cuts,
        numeric_focus=common.numeric_focus,
        feasibility_tol=common.feasibility_tol,
        optimality_tol=common.optimality_tol,
        integer_feasibility_tol=common.integer_feasibility_tol,
        static_source_level=static.source_level_upper,
        static_virtual_level_lower=static.virtual_level_lower,
        static_lo_choice_corridor=static.lo_choice_corridor_cover,
        static_lo_choice_capacity=static.lo_choice_capacity_projection,
        static_queue_perspective=static.queue_perspective_support,
        static_nearest_display_suffix=False,
        static_chi_vub_top_k_per_corridor=0,
        cut_chi_vub=cuts.aggregated_od_corridor_vub,
        cut_chi_facility_vub=False,
        cut_route_swap_station_vub=False,
        cut_route_swap_leg_vub=False,
        cut_queue_perspective=cuts.queue_perspective,
        cut_lo_choice_corridor=cuts.lo_choice_corridor_support,
        cut_lo_choice_capacity=cuts.lo_choice_capacity_cover,
        cut_directed_return=cuts.directed_return_connectivity,
        # Development-only: never enters the frozen A--E profile contract
        # without a separately registered promotion decision.
        cut_od_disaggregated_return=False,
        # This must remain true for every A--E run so callback overhead and
        # progress measurements are matched, including the native baseline.
        collect_progress=True,
        progress_interval_seconds=telemetry.progress_interval_seconds,
    )
    config.validate()
    return config


def _create_private_gurobi_environment() -> object:
    import gurobipy as gp

    with _GUROBI_ENVIRONMENT_LIFECYCLE_LOCK:
        # ``empty=True`` is essential: it prevents gurobipy from silently
        # allocating/starting its process-wide default environment.  Starting
        # the owned environment is a separate, auditable lifecycle transition.
        environment = gp.Env(empty=True)
        try:
            environment.start()
        except BaseException:
            # A failed start must not leak a partially initialized native
            # environment.  Preserve the start exception even if cleanup fails.
            try:
                environment.dispose()
            except BaseException:
                pass
            raise
        return environment


def _dispose_private_gurobi_environment(environment: object) -> None:
    dispose = getattr(environment, "dispose", None)
    if callable(dispose):
        with _GUROBI_ENVIRONMENT_LIFECYCLE_LOCK:
            dispose()


def _default_model_factory(instance: Any, mode: str) -> object:
    from src.model.gurobi_model import MLNDModel

    # Allocate the Python wrapper separately and attach its owned environment
    # before model initialization.  This keeps both resources reachable if
    # construction fails after either native allocation.
    wrapper = MLNDModel.__new__(MLNDModel)
    initialized = False
    try:
        environment = _create_private_gurobi_environment()
        setattr(wrapper, "_owned_environment", environment)
        MLNDModel.__init__(wrapper, instance, mode, environment=environment)
        initialized = True
        return wrapper
    finally:
        if not initialized:
            try:
                _dispose_model_resources(wrapper)
            except MemoryError:
                raise
            except Exception:
                # Preserve the construction exception; there is no benchmark
                # wrapper yet on which to record a secondary cleanup failure.
                pass


def _default_gurobi_version() -> str:
    try:
        import gurobipy as gp

        return ".".join(str(part) for part in gp.gurobi.version())
    except (ImportError, AttributeError, RuntimeError):
        return "unavailable"


def _version_string(provider: VersionProvider) -> str:
    value = provider()
    if isinstance(value, tuple):
        return ".".join(str(part) for part in value)
    return str(value)


def _json_payload(value: Any) -> Any:
    """Return the exact JSON-ready value used by canonical artifact writes."""

    return json.loads(canonical_json(value))


def _formal_context_payload(
    formal_context: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    """Canonicalize optional formal-campaign provenance before execution.

    The formal scheduler constructs and validates the field contract.  This
    lower layer deliberately treats it as opaque provenance, but insists on a
    JSON object so the exact same immutable value can be copied into every
    attempt artifact.
    """

    if formal_context is None:
        return None
    if not isinstance(formal_context, Mapping):
        raise TypeError("formal_context must be a mapping or None")
    payload = _json_payload(formal_context)
    if not isinstance(payload, dict):  # pragma: no cover - guarded by Mapping
        raise TypeError("formal_context must canonicalize to a JSON object")
    return payload


def _validate_claim(pair: PairSpec, profile: ProfileConfig, claim: AttemptClaim) -> None:
    logical_run_id = build_logical_run_id(pair, profile)
    expected_attempt_id = build_attempt_id(logical_run_id, claim.attempt_number)
    if claim.logical_run_id != logical_run_id:
        raise ValueError("attempt claim belongs to a different logical run")
    if claim.attempt_id != expected_attempt_id:
        raise ValueError("attempt claim has an inconsistent attempt ID")
    if claim.directory.name != claim.attempt_id:
        raise ValueError("attempt directory name must equal attempt_id")
    if claim.directory.parent.name != claim.logical_run_id:
        raise ValueError("attempt directory parent must equal logical_run_id")
    if not claim.directory.is_dir() or not claim.claim_file.is_file():
        raise ValueError("attempt claim directory and claim.json must already exist")


def _set_solver_log(model_wrapper: object, path: Path, enabled: bool) -> None:
    if not enabled:
        return
    backend = getattr(model_wrapper, "model")
    parameters = getattr(backend, "Params")
    setattr(parameters, "LogFile", str(path))


def _unavailable_solver_metrics() -> dict[str, Any]:
    """Return a stable fail-closed metric payload after extraction failure."""

    return {
        "status": None,
        "status_name": "TELEMETRY_UNAVAILABLE",
        "has_incumbent": False,
        "solution_count": 0,
        "objective": None,
        "best_bound": None,
        "relative_gap": None,
        "node_count": None,
        "main_solve_runtime_seconds": None,
        "total_algorithm_runtime_seconds": None,
        "budget_exhausted_before_main_solve": None,
        "main_solve_started": None,
        "optimize_called": None,
    }


def _fail_closed_unstarted_main_solve(metrics: dict[str, Any]) -> None:
    """Remove any stale auxiliary-solve outcome from formal solver metrics."""

    metrics.update(
        {
            "status": 1,
            "status_name": "LOADED",
            "has_incumbent": False,
            "solution_count": 0,
            "objective": None,
            "best_bound": None,
            "relative_gap": None,
            "node_count": None,
            "main_solve_runtime_seconds": 0.0,
        }
    )


def _optional_bool_attribute(value: object | None, name: str) -> bool | None:
    """Read an explicit execution marker without inventing one for old wrappers."""

    if value is None:
        return None
    missing = object()
    marker = getattr(value, name, missing)
    if marker is missing or marker is None:
        return None
    return bool(marker)


def _dispose_model_resources(model_wrapper: object | None) -> None:
    """Release the owned backend and any explicitly exposed private environment."""

    if model_wrapper is None:
        return
    backend = getattr(model_wrapper, "model", None)
    backend_dispose = getattr(backend, "dispose", None)
    cleanup_error: Exception | None = None
    cleanup_memory_error: MemoryError | None = None
    if callable(backend_dispose):
        try:
            backend_dispose()
        except MemoryError as error:
            cleanup_memory_error = error
        except Exception as error:
            cleanup_error = error

    # Release only explicitly exposed environments rather than retrieving and
    # disposing Gurobi's process-wide default environment indirectly.  The
    # benchmark factory uses _owned_environment; the legacy names preserve the
    # cleanup contract for injected factories.
    seen: set[int] = {id(backend)}
    for name in (
        "_owned_environment",
        "environment",
        "env",
        "_environment",
        "_env",
    ):
        environment = getattr(model_wrapper, name, None)
        if environment is None or id(environment) in seen:
            continue
        seen.add(id(environment))
        try:
            _dispose_private_gurobi_environment(environment)
        except MemoryError as error:
            if cleanup_memory_error is None:
                cleanup_memory_error = error
        except Exception as error:
            if cleanup_error is None:
                cleanup_error = error
    if cleanup_memory_error is not None:
        raise cleanup_memory_error
    if cleanup_error is not None:
        raise cleanup_error


def _with_failure_stage(stage: str, error: Exception) -> dict[str, Any]:
    payload = exception_payload(error)
    payload["stage"] = stage
    return payload


def execute_profile(
    instance: Any,
    pair: PairSpec,
    profile: ProfileConfig,
    telemetry: TelemetryConfig,
    claimed_attempt: AttemptClaim,
    *,
    model_factory: ModelFactory | None = None,
    gurobi_version_provider: VersionProvider | None = None,
    gurobi_parameter_name_provider: GurobiParameterNameProvider | None = None,
    formal_context: Mapping[str, Any] | None = None,
) -> ProfileExecutionResult:
    """Run one profile and write a complete, non-overwriting attempt bundle.

    Ordinary solver exceptions are converted into ``failure.json`` plus a failed
    ``run_record.json`` so a batch can continue.  Artifact reuse errors are
    raised before execution because an existing attempt must never be mutated.
    Fatal process-control exceptions are never converted into benchmark data.
    """

    _validate_claim(pair, profile, claimed_attempt)
    frozen_formal_context = _formal_context_payload(formal_context)
    qualification_trace: FormalQualificationTrace | None = None
    if frozen_formal_context is not None and (
        "qualification_trace" in frozen_formal_context
        or "qualification_trace_sha256" in frozen_formal_context
    ):
        qualification_trace = qualification_trace_from_context(
            frozen_formal_context
        )
    paths = ArtifactPaths.from_claim(claimed_attempt)
    assert_fresh_artifact_paths(paths)
    core_config = build_core_solver_config(pair, profile, telemetry)
    version_provider = gurobi_version_provider or _default_gurobi_version
    gurobi_version = _version_string(version_provider)
    ids = {
        "pair_id": build_pair_id(pair),
        "profile_id": build_profile_id(profile),
        "logical_run_id": build_logical_run_id(pair, profile),
        "attempt_id": claimed_attempt.attempt_id,
        "attempt_number": claimed_attempt.attempt_number,
    }
    run_spec = {
        "schema_version": 1,
        "ids": ids,
        "pair": pair,
        "profile": profile,
        "profile_label": profile.label,
        "telemetry": telemetry,
        "core_solver_config": asdict(core_config),
        "gurobi_version": gurobi_version,
    }
    if frozen_formal_context is not None:
        run_spec["formal_context"] = frozen_formal_context
    write_canonical_json_exclusive(paths.run_spec, run_spec)

    factory = model_factory or _default_model_factory
    model_wrapper: object | None = None
    solve_returned_incumbent = False
    solve_error: Exception | None = None
    post_solve_errors: list[dict[str, Any]] = []
    progress: tuple[dict[str, Any], ...] = ()
    root_rounds: tuple[dict[str, Any], ...] = ()
    cut_stats: dict[str, Any] = {"captured": False}
    metrics: dict[str, Any] = _unavailable_solver_metrics()
    solution_structure: FormalSolutionStructure | None = None
    applied_parameters: dict[str, Any] = {}
    parameter_audit = collect_gurobi_parameter_audit(
        None,
        strict=False,
        parameter_name_provider=gurobi_parameter_name_provider,
    )
    execution_started = time.perf_counter()
    try:
        try:
            model_wrapper = factory(instance, pair.mode.value)
            _set_solver_log(model_wrapper, paths.solver_log, telemetry.solver_log)
            solve_method = getattr(model_wrapper, "solve")
            solve_returned_incumbent = bool(solve_method(core_config))
        except MemoryError:
            raise
        except Exception as error:
            solve_error = error
    finally:
        wall_runtime = time.perf_counter() - execution_started
        try:
            try:
                ensure_solver_log(paths.solver_log)
            except MemoryError:
                raise
            except Exception as error:
                post_solve_errors.append(_with_failure_stage("solver_log", error))
            try:
                progress, root_rounds, cut_stats = write_telemetry_artifacts(
                    paths,
                    model_wrapper,
                    telemetry,
                )
            except MemoryError:
                raise
            except Exception as error:
                post_solve_errors.append(
                    _with_failure_stage("telemetry_artifacts", error)
                )
            try:
                metrics = collect_solver_metrics(model_wrapper)
                budget_exhausted_before_main_solve = _optional_bool_attribute(
                    model_wrapper,
                    "budget_exhausted_before_main_solve",
                )
                main_solve_started = _optional_bool_attribute(
                    model_wrapper,
                    "main_solve_started",
                )
                optimize_called = _optional_bool_attribute(
                    model_wrapper,
                    "optimize_called",
                )
                if optimize_called is False:
                    _fail_closed_unstarted_main_solve(metrics)
                metrics["budget_exhausted_before_main_solve"] = (
                    budget_exhausted_before_main_solve
                )
                metrics["main_solve_started"] = main_solve_started
                metrics["optimize_called"] = optimize_called
            except MemoryError:
                raise
            except Exception as error:
                post_solve_errors.append(_with_failure_stage("solver_metrics", error))
            if qualification_trace is not None:
                try:
                    solution_structure = recompute_solution_structure(
                        model_wrapper,
                        qualification_trace,
                        expected_has_incumbent=bool(metrics["has_incumbent"]),
                    )
                except MemoryError:
                    raise
                except Exception as error:
                    post_solve_errors.append(
                        _with_failure_stage("solution_structure", error)
                    )
            try:
                applied_parameters = collect_applied_gurobi_parameters(model_wrapper)
            except MemoryError:
                raise
            except Exception as error:
                post_solve_errors.append(
                    _with_failure_stage("gurobi_parameters", error)
                )
            try:
                parameter_audit = collect_gurobi_parameter_audit(
                    model_wrapper,
                    strict=frozen_formal_context is not None,
                    parameter_name_provider=gurobi_parameter_name_provider,
                )
            except MemoryError:
                raise
            except Exception as error:
                post_solve_errors.append(
                    _with_failure_stage("gurobi_parameter_audit", error)
                )
        finally:
            try:
                _dispose_model_resources(model_wrapper)
            except MemoryError:
                raise
            except Exception as error:
                post_solve_errors.append(_with_failure_stage("model_disposal", error))

    failure: dict[str, Any] | None = None
    if solve_error is not None:
        failure = _with_failure_stage("solve", solve_error)
        if post_solve_errors:
            failure["secondary_errors"] = post_solve_errors
    elif post_solve_errors:
        failure = dict(post_solve_errors[0])
        if len(post_solve_errors) > 1:
            failure["secondary_errors"] = post_solve_errors[1:]

    completed = failure is None
    has_incumbent = bool(metrics["has_incumbent"])
    if completed and has_incumbent:
        outcome = "completed_with_incumbent"
    elif completed:
        outcome = "completed_no_incumbent"
    else:
        outcome = "failed"

    audit_metadata = {
        key: value
        for key, value in parameter_audit.items()
        if key
        not in {
            "effective_nondefault_parameters",
            "execution_context",
            "unallowed_nondefault_parameters",
        }
    }
    run_record: dict[str, Any] = {
        "schema_version": 1,
        "ids": ids,
        "profile_code": profile.code.value,
        "profile_label": profile.label,
        "outcome": outcome,
        "completed_without_exception": completed,
        "solve_returned_incumbent": solve_returned_incumbent,
        "solver_metrics": metrics,
        "execution_wall_runtime_seconds": wall_runtime,
        "budget_exhausted_before_main_solve": metrics.get(
            "budget_exhausted_before_main_solve"
        ),
        "main_solve_started": metrics.get("main_solve_started"),
        "optimize_called": metrics.get("optimize_called"),
        "gurobi_version": gurobi_version,
        "core_solver_config": asdict(core_config),
        "applied_gurobi_parameters": applied_parameters,
        "gurobi_parameter_audit": audit_metadata,
        "effective_nondefault_parameters": parameter_audit.get(
            "effective_nondefault_parameters",
            {},
        ),
        "unallowed_nondefault_parameters": parameter_audit.get(
            "unallowed_nondefault_parameters",
            [],
        ),
        "gurobi_execution_context": parameter_audit.get(
            "execution_context",
            {
                "execution_mode": "unobserved",
                "observation_basis": "gurobi_effective_parameters",
                "remote_identity_sha256": None,
                "remote_mechanisms": [],
            },
        ),
        "telemetry_counts": {
            "progress_events": len(progress),
            "root_round_events": len(root_rounds),
            "static_rows_added": cut_stats.get("static_rows_added_total"),
            "user_cuts_added": cut_stats.get("user_cuts_added_total"),
        },
        "failure": failure,
    }
    if frozen_formal_context is not None:
        run_record["formal_context"] = frozen_formal_context
    if qualification_trace is not None:
        solution_structure_payload = (
            solution_structure.to_payload()
            if solution_structure is not None
            else None
        )
        run_record["solution_structure"] = solution_structure_payload
        run_record["solution_structure_sha256"] = (
            canonical_sha256(solution_structure_payload)
            if solution_structure_payload is not None
            else None
        )
    if failure is not None:
        failure_artifact = {
            "schema_version": 1,
            "ids": ids,
            "gurobi_version": gurobi_version,
            "solver_metrics_at_failure": metrics,
            **failure,
        }
        if frozen_formal_context is not None:
            failure_artifact["formal_context"] = frozen_formal_context
        write_canonical_json_exclusive(
            paths.failure,
            failure_artifact,
        )
    write_canonical_json_exclusive(paths.run_record, run_record)
    ready_record = _json_payload(run_record)
    ready_failure = _json_payload(failure) if failure is not None else None
    return ProfileExecutionResult(
        completed_without_exception=completed,
        has_incumbent=has_incumbent,
        outcome=outcome,
        attempt_id=claimed_attempt.attempt_id,
        directory=claimed_attempt.directory,
        run_record=ready_record,
        failure=ready_failure,
    )


__all__ = [
    "ModelFactory",
    "ProfileExecutionResult",
    "VersionProvider",
    "build_core_solver_config",
    "execute_profile",
]
