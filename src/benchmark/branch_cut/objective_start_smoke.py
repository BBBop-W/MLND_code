"""Fail-closed, no-optimize smoke test for MX proxy partial starts.

The smoke test builds the frozen H9 instance by default (with explicit H4/H6/H10
selection available and H10 requiring a second acknowledgement), constructs the
development sparse structural MIP assignments twice, and compares their exact
support and proxy telemetry.
It never completes a start, calls ``MLNDModel.solve``, or invokes Gurobi's
``optimize`` method.  Passing therefore establishes neither full-model
feasibility, realized service, Gurobi start acceptance, nor an incumbent, and
cannot be used as timed benchmark evidence.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

from main import REPO_ROOT as MAIN_REPO_ROOT, _validate_instance_demand_scale
from src.config import ModelConfig, SolverConfig, to_dict
from src.model_contract import FORMULATION_ID, RESULT_SCHEMA_VERSION
from src.model.primal_heuristic import (
    OBJECTIVE_START_PROXY_ENERGY_REL_TOLERANCE,
    OBJECTIVE_START_PROXY_QOS_ABS_TOLERANCE,
    OBJECTIVE_START_PROXY_SEARCH_REL_TOLERANCE,
)

from .artifacts import write_canonical_json_exclusive
from . import dev_micro_ce as dev
from .resource_lock import BenchmarkResourceLock, BenchmarkResourceLockError


ARTIFACT_SCHEMA_VERSION = 1
REPETITIONS = 2
START_BUDGET_SECONDS = 0.25
TIME_LIMIT_CONTEXT_SECONDS = 5.0
FEEDER_RIDER_COST_FACTOR = 0.30
POLICY_VERSION = "objective_partial_v3_system_energy"
HARNESS_RELATIVE_PATH = Path("src/benchmark/branch_cut/objective_start_smoke.py")


class ObjectiveStartSmokeError(RuntimeError):
    """The construction smoke cannot proceed without weakening its contract."""


@dataclass(frozen=True, slots=True)
class SmokeConfig:
    repo_root: Path
    output_root: Path
    formulation_source: Path | None = None
    expected_formulation_sha256: str = dev.EXPECTED_FORMULATION_SHA256
    requested_stages: tuple[str, ...] | None = None
    include_h10: bool = False
    confirm_h10: bool = False
    resource_lock: Path = dev.DEFAULT_BENCHMARK_RESOURCE_LOCK

    @property
    def stages(self) -> tuple[str, ...]:
        if self.requested_stages is not None:
            return tuple(self.requested_stages)
        return ("H9", "H10") if self.include_h10 else ("H9",)


@dataclass(frozen=True, slots=True)
class ConstructionHooks:
    """Narrow dependency seam used by pure tests; production uses real builders."""

    load_instance: Callable[[Path, ModelConfig], Any]
    create_built_model: Callable[[Any], Any]
    clock: Callable[[], float] = time.perf_counter


ConstructionRunner = Callable[[str, Path, int, int], Mapping[str, Any]]


class _NoOptimizeModelProxy:
    """Delegate model access while making backend optimization impossible."""

    __slots__ = ("_backend", "blocked_optimize_calls")

    def __init__(self, backend: Any) -> None:
        object.__setattr__(self, "_backend", backend)
        object.__setattr__(self, "blocked_optimize_calls", 0)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._backend, name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name in self.__slots__:
            object.__setattr__(self, name, value)
        else:
            setattr(self._backend, name, value)

    def optimize(self, *_args: Any, **_kwargs: Any) -> None:
        object.__setattr__(
            self,
            "blocked_optimize_calls",
            self.blocked_optimize_calls + 1,
        )
        raise ObjectiveStartSmokeError(
            "objective-start construction attempted forbidden backend optimization"
        )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _model_config() -> ModelConfig:
    config = ModelConfig(
        feeder_rider_cost_factor=FEEDER_RIDER_COST_FACTOR,
        display_origin_choices=None,
        general_corridor_choices=None,
    )
    config.validate()
    return config


def _solver_config() -> SolverConfig:
    config = SolverConfig(
        time_limit=TIME_LIMIT_CONTEXT_SECONDS,
        mip_gap=0.0,
        threads=1,
        output_flag=0,
        seed=11,
        start_mode="none",
        use_objective_partial_start=True,
        objective_partial_start_seconds=START_BUDGET_SECONDS,
        objective_partial_start_max_candidates=3,
        network_start_seconds=0.0,
        network_start_method=2,
        fixed_integer_repair_seconds=0.0,
        use_local_branching=False,
        local_branching_seconds=0.0,
        use_valid_inequalities=True,
        use_user_cuts=True,
        max_root_cut_rounds=3,
        root_cut_passes=dev.ROOT_CUT_PASSES,
        max_cuts_per_round=250,
        cut_violation_tolerance=1e-6,
        cut_efficacy_tolerance=1e-9,
        static_nearest_display_suffix=False,
        static_chi_vub_top_k_per_corridor=0,
        cut_chi_facility_vub=False,
        cut_route_swap_station_vub=False,
        cut_route_swap_leg_vub=False,
        root_relaxation_method=-1,
        miqcp_method=-1,
        mip_focus=1,
        heuristics=0.2,
        no_rel_heur_seconds=0.0,
        no_rel_heur_time=0.0,
        presolve=-1,
        cuts=-1,
        numeric_focus=0,
        collect_progress=True,
        progress_interval_seconds=0.25,
    )
    config.validate()
    return config


def _default_load_instance(instance_dir: Path, model_config: ModelConfig) -> Any:
    from src.data_processor.loader import DataLoader

    metadata_path = dev._require_regular_file(
        instance_dir / "metadata.json", kind="instance metadata"
    )
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ObjectiveStartSmokeError(
            f"cannot read instance metadata {metadata_path}: {error}"
        ) from error
    _validate_instance_demand_scale(metadata, 1.0)
    return DataLoader().load_instance_dir(
        str(instance_dir), model_config=model_config, demand_scale=1.0
    )


def _default_create_built_model(instance: Any) -> Any:
    from src.model.gurobi_model import MLNDModel

    return MLNDModel(instance, mode="MX").build()


DEFAULT_CONSTRUCTION_HOOKS = ConstructionHooks(
    load_instance=_default_load_instance,
    create_built_model=_default_create_built_model,
)


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ObjectiveStartSmokeError(f"{label} must be a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ObjectiveStartSmokeError(f"{label} must be a finite number") from error
    if not math.isfinite(number):
        raise ObjectiveStartSmokeError(f"{label} must be a finite number")
    return number


def _capture_sparse_start_pool(model: Any) -> list[dict[str, Any]]:
    """Capture assigned support only; unassigned Gurobi values stay undefined."""

    from gurobipy import GRB

    model.update()
    variables = list(model.getVars())
    num_start = int(model.NumStart)
    original_start_number = int(model.Params.StartNumber)
    pool: list[dict[str, Any]] = []
    try:
        for index in range(num_start):
            model.Params.StartNumber = index
            values = model.getAttr(GRB.Attr.Start, variables)
            assignments: list[list[Any]] = []
            for variable, raw_value in zip(variables, values):
                value = float(raw_value)
                if value == float(GRB.UNDEFINED):
                    continue
                if not math.isfinite(value):
                    raise ObjectiveStartSmokeError(
                        f"start {index} contains a nonfinite assigned value"
                    )
                if value == 0.0:
                    value = 0.0
                assignments.append([str(variable.VarName), value])
            assignments.sort(key=lambda item: item[0])
            if len({item[0] for item in assignments}) != len(assignments):
                raise ObjectiveStartSmokeError(
                    f"start {index} contains duplicate variable names"
                )
            pool.append(
                {
                    "start_index": index,
                    "assigned_variable_count": len(assignments),
                    "assignment_sha256": _canonical_sha256(assignments),
                    "first_assigned_variable_names": [
                        item[0] for item in assignments[:8]
                    ],
                }
            )
    finally:
        model.Params.StartNumber = original_start_number if num_start else 0
        model.update()
    return pool


def _construct_once(
    stage: str,
    instance_dir: Path,
    candidate_h: int,
    repetition: int,
    *,
    hooks: ConstructionHooks = DEFAULT_CONSTRUCTION_HOOKS,
) -> dict[str, Any]:
    """Build MX proxy assignments through the no-optimize entry point only."""

    model_config = _model_config()
    solver_config = _solver_config()
    instance = hooks.load_instance(instance_dir, model_config)
    observed_candidate_h = len(instance.H)
    if observed_candidate_h != candidate_h:
        raise ObjectiveStartSmokeError(
            f"{stage} candidate-H mismatch: expected {candidate_h}, "
            f"loaded {observed_candidate_h}"
        )
    wrapper = hooks.create_built_model(instance)
    backend = wrapper.model
    guarded_backend = _NoOptimizeModelProxy(backend)
    wrapper.model = guarded_backend
    try:
        optimize_before = bool(getattr(wrapper, "optimize_called", False))
        main_solve_before = bool(getattr(wrapper, "main_solve_started", False))
        started = hooks.clock()
        # This is intentionally the construction method, never solve() and
        # never model.optimize().  Its implementation installs sparse starts.
        wrapper._construct_objective_partial_starts(
            START_BUDGET_SECONDS,
            solver_config,
        )
        elapsed = hooks.clock() - started
        optimize_after = bool(getattr(wrapper, "optimize_called", False))
        main_solve_after = bool(getattr(wrapper, "main_solve_started", False))
        start_pool = _capture_sparse_start_pool(wrapper.model)
        wrapper.model.update()
        variable_count = len(wrapper.model.getVars())
        telemetry = json.loads(
            json.dumps(
                wrapper.objective_partial_start_telemetry,
                ensure_ascii=False,
                sort_keys=True,
                allow_nan=False,
            )
        )
        return {
            "stage": stage,
            "repetition": repetition,
            "construction_entrypoint": "_construct_objective_partial_starts",
            "backend_optimize_guard_installed": True,
            "timed_solver_run_started": False,
            "optimization_api_call_count": guarded_backend.blocked_optimize_calls,
            "optimize_called_before": optimize_before,
            "optimize_called_after": optimize_after,
            "main_solve_started_before": main_solve_before,
            "main_solve_started_after": main_solve_after,
            "candidate_h": observed_candidate_h,
            "total_demand_per_hour": float(instance.total_demand),
            "model_variable_count": variable_count,
            "constructor_elapsed_seconds": elapsed,
            "backend_num_start": int(wrapper.model.NumStart),
            "start_pool": start_pool,
            "telemetry": telemetry,
        }
    finally:
        wrapper.model = backend
        dispose = getattr(backend, "dispose", None)
        if callable(dispose):
            dispose()


def _tolerance(*values: float) -> float:
    return 1e-9 * max(1.0, *(abs(value) for value in values))


def _audit_construction(
    record: Mapping[str, Any], *, candidate_h: int
) -> dict[str, Any]:
    errors: list[str] = []
    if record.get("construction_entrypoint") != "_construct_objective_partial_starts":
        errors.append("wrong_construction_entrypoint")
    if record.get("backend_optimize_guard_installed") is not True:
        errors.append("backend_optimize_guard_missing")
    if record.get("timed_solver_run_started") is not False:
        errors.append("timed_solver_run_started")
    if record.get("optimization_api_call_count") != 0:
        errors.append("optimization_api_called")
    for field in (
        "optimize_called_before",
        "optimize_called_after",
        "main_solve_started_before",
        "main_solve_started_after",
    ):
        if record.get(field) is not False:
            errors.append(field)
    if record.get("candidate_h") != candidate_h:
        errors.append("candidate_h_mismatch")

    try:
        elapsed = _finite_number(
            record.get("constructor_elapsed_seconds"), "constructor elapsed"
        )
        if elapsed < 0.0 or elapsed > START_BUDGET_SECONDS + 1e-6:
            errors.append("constructor_elapsed_exceeds_hard_budget")
    except ObjectiveStartSmokeError:
        errors.append("invalid_constructor_elapsed")

    variable_count = record.get("model_variable_count")
    if isinstance(variable_count, bool) or not isinstance(variable_count, int):
        errors.append("invalid_model_variable_count")
        variable_count = 0
    elif variable_count <= 1:
        errors.append("invalid_model_variable_count")

    telemetry = record.get("telemetry")
    if not isinstance(telemetry, Mapping):
        return {
            "valid": False,
            "errors": [*errors, "missing_telemetry"],
            "determinism_sha256": None,
        }
    if telemetry.get("policy_version") != POLICY_VERSION:
        errors.append("policy_version")
    if telemetry.get("proxy_only") is not True:
        errors.append("proxy_only_flag")
    expected_tolerances = {
        "qos_arrival_absolute": OBJECTIVE_START_PROXY_QOS_ABS_TOLERANCE,
        "system_energy_relative": OBJECTIVE_START_PROXY_ENERGY_REL_TOLERANCE,
        "construction_search_relative": (
            OBJECTIVE_START_PROXY_SEARCH_REL_TOLERANCE
        ),
        "solver_feasibility_tolerance_linked": False,
    }
    if telemetry.get("proxy_feasibility_tolerances") != expected_tolerances:
        errors.append("proxy_feasibility_tolerances")
    if telemetry.get("enabled") is not True:
        errors.append("not_enabled")
    if telemetry.get("used") is not True:
        errors.append("not_used")
    if telemetry.get("timed_out") is not False:
        errors.append("timed_out")
    if telemetry.get("fallback_reason") is not None:
        errors.append("fallback_reason")

    try:
        budget = _finite_number(telemetry.get("budget_seconds"), "declared budget")
        preparation = _finite_number(
            telemetry.get("preparation_seconds"), "preparation seconds"
        )
        if abs(budget - START_BUDGET_SECONDS) > 1e-12:
            errors.append("declared_budget_not_exact")
        if preparation < 0.0 or preparation > START_BUDGET_SECONDS + 1e-6:
            errors.append("preparation_exceeds_hard_budget")
    except ObjectiveStartSmokeError:
        errors.append("invalid_budget_telemetry")

    network_count = telemetry.get("network_candidate_count")
    safety_count = telemetry.get("safety_start_count")
    installed_count = telemetry.get("installed_start_count")
    if any(
        isinstance(value, bool) or not isinstance(value, int)
        for value in (network_count, safety_count, installed_count)
    ):
        errors.append("invalid_start_counts")
        network_count = 0
        safety_count = 0
        installed_count = 0
    else:
        if network_count < 1:
            errors.append("no_network_candidate")
        if safety_count != 1:
            errors.append("safety_start_count")
        if installed_count != network_count + safety_count:
            errors.append("installed_start_count")
    if record.get("backend_num_start") != installed_count:
        errors.append("backend_start_count_mismatch")

    pool = record.get("start_pool")
    if not isinstance(pool, list) or len(pool) != installed_count:
        errors.append("start_pool_count")
        pool = []
    for index, start in enumerate(pool):
        if not isinstance(start, Mapping):
            errors.append(f"start_{index}_not_mapping")
            continue
        count = start.get("assigned_variable_count")
        if (
            isinstance(count, bool)
            or not isinstance(count, int)
            or not 0 < count < variable_count
        ):
            errors.append(f"start_{index}_not_sparse")
        if start.get("start_index") != index:
            errors.append(f"start_{index}_index")
        signature = start.get("assignment_sha256")
        if not isinstance(signature, str) or len(signature) != 64:
            errors.append(f"start_{index}_signature")

    candidates = telemetry.get("candidate_supports")
    if not isinstance(candidates, list) or len(candidates) != network_count:
        errors.append("candidate_support_count")
        candidates = []
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, Mapping):
            errors.append(f"candidate_{index}_not_mapping")
            continue
        open_h = candidate.get("open_h")
        if (
            not isinstance(open_h, list)
            or len(open_h) != len(set(open_h))
            or not 0 < len(open_h) < candidate_h
        ):
            errors.append(f"candidate_{index}_open_h_not_interior")
        try:
            service = _finite_number(
                candidate.get("estimated_service"), "estimated service"
            )
            estimated_service_fraction = _finite_number(
                candidate.get("estimated_service_fraction"),
                "estimated service fraction",
            )
            fraction_basis_service = _finite_number(
                candidate.get("fraction_basis_service"),
                "fraction basis service",
            )
            score = _finite_number(candidate.get("score"), "candidate score")
            if service <= 0.0:
                errors.append(f"candidate_{index}_nonpositive_estimated_service")
            if not 0.0 < estimated_service_fraction <= 1.0:
                errors.append(
                    f"candidate_{index}_invalid_estimated_service_fraction"
                )
            if fraction_basis_service <= 0.0:
                errors.append(f"candidate_{index}_invalid_fraction_basis_service")
            elif abs(
                service / fraction_basis_service - estimated_service_fraction
            ) > _tolerance(
                service / fraction_basis_service,
                estimated_service_fraction,
            ):
                errors.append(f"candidate_{index}_fraction_identity_mismatch")
            if candidate.get("proxy_only") is not True:
                errors.append(f"candidate_{index}_proxy_only_flag")
            if score <= 0.0:
                errors.append(f"candidate_{index}_nonpositive_proxy_score")
        except ObjectiveStartSmokeError:
            errors.append(f"candidate_{index}_invalid_proxy_service_or_score")
        assigned = candidate.get("assigned_variable_count")
        if (
            isinstance(assigned, bool)
            or not isinstance(assigned, int)
            or not 0 < assigned < variable_count
        ):
            errors.append(f"candidate_{index}_assigned_count")
        elif index < len(pool) and assigned != pool[index].get(
            "assigned_variable_count"
        ):
            errors.append(f"candidate_{index}_assigned_count_mismatch")

        energy = candidate.get("energy_proxy")
        if not isinstance(energy, Mapping):
            errors.append(f"candidate_{index}_missing_system_energy_proxy")
            continue
        if any(key in energy for key in ("energy_base", "energy_locker")):
            errors.append(f"candidate_{index}_legacy_energy_proxy")
        try:
            use = _finite_number(energy.get("system_energy_use"), "energy use")
            supply = _finite_number(
                energy.get("system_energy_supply"), "energy supply"
            )
            base = _finite_number(
                energy.get("base_departure_population"), "base departures"
            )
            swaps = _finite_number(
                energy.get("swap_event_population"), "swap events"
            )
            if use <= 0.0 or supply <= 0.0 or base <= 0.0 or swaps < 0.0:
                errors.append(f"candidate_{index}_invalid_system_energy_proxy")
            expected_supply = _model_config().usable_battery_time_hours * (
                base + swaps
            )
            if abs(supply - expected_supply) > _tolerance(
                supply, expected_supply
            ):
                errors.append(f"candidate_{index}_system_energy_supply_mismatch")
            if use > supply + _tolerance(use, supply):
                errors.append(
                    f"candidate_{index}_system_energy_proxy_exceeds_supply"
                )
        except ObjectiveStartSmokeError:
            errors.append(f"candidate_{index}_invalid_system_energy_proxy")

    determinism_payload = {
        "candidate_h": record.get("candidate_h"),
        "model_variable_count": variable_count,
        "backend_num_start": record.get("backend_num_start"),
        "start_pool": pool,
        "telemetry": {
            key: value
            for key, value in telemetry.items()
            if key != "preparation_seconds"
        },
    }
    try:
        determinism_sha256 = _canonical_sha256(determinism_payload)
    except (TypeError, ValueError):
        errors.append("nondeterministic_payload_not_canonical")
        determinism_sha256 = None
    return {
        "valid": not errors,
        "errors": errors,
        "determinism_sha256": determinism_sha256,
    }


def _normalize_config(config: SmokeConfig) -> SmokeConfig:
    repo_root = config.repo_root.expanduser().absolute()
    formulation_source = (
        config.formulation_source.expanduser().absolute()
        if config.formulation_source is not None
        else repo_root / dev.FORMULATION_RELATIVE_PATH
    )
    return SmokeConfig(
        repo_root=repo_root,
        output_root=config.output_root.expanduser().absolute(),
        formulation_source=formulation_source,
        expected_formulation_sha256=config.expected_formulation_sha256,
        requested_stages=config.requested_stages,
        include_h10=config.include_h10,
        confirm_h10=config.confirm_h10,
        resource_lock=dev._canonical_lock_path(Path(config.resource_lock)),
    )


def _batch_config(config: SmokeConfig) -> dev.BatchConfig:
    return dev.BatchConfig(
        repo_root=config.repo_root,
        output_root=config.output_root,
        formulation_source=config.formulation_source,
        expected_formulation_sha256=config.expected_formulation_sha256,
        requested_stages=config.requested_stages,
        include_h10=config.include_h10,
        confirm_h10=config.confirm_h10,
        jobs=1,
        resource_lock=config.resource_lock,
    )


def _process_audit(stage: str, repetition: int) -> dict[str, Any]:
    table = dev._process_table()
    conflicts = dev._conflicting_processes(table)
    return {
        "stage": stage,
        "repetition": repetition,
        "process_table_sha256": hashlib.sha256(table.encode("utf-8")).hexdigest(),
        "conflicting_processes": list(conflicts),
        "valid": not conflicts,
    }


def _manifest_payload(
    config: SmokeConfig,
    frozen: dev.FrozenInputs,
    harness_sha256: str,
) -> dict[str, Any]:
    return {
        "artifact_type": "mlnd_objective_start_construction_smoke_manifest",
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "development_only": True,
        "formal_evidence_eligible": False,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "repository": {
            "root": str(config.repo_root),
            "commit": frozen.commit,
            "branch": frozen.branch,
            "required_baseline_ancestor": dev.REQUIRED_BASELINE_COMMIT,
            "implementation_sha256": frozen.implementation_sha256,
            "implementation_files": dict(frozen.implementation_files),
            "harness_relative_path": HARNESS_RELATIVE_PATH.as_posix(),
            "harness_sha256": harness_sha256,
        },
        "formulation_source": {
            "path": frozen.formulation_source,
            "sha256": frozen.formulation_sha256,
        },
        "instances": dict(frozen.instances),
        "protocol": {
            "formulation_id": FORMULATION_ID,
            "result_schema_version": RESULT_SCHEMA_VERSION,
            "stages": list(config.stages),
            "repetitions_per_stage": REPETITIONS,
            "mode": "MX",
            "objective_partial_policy_version": POLICY_VERSION,
            "objective_partial_start_budget_seconds": START_BUDGET_SECONDS,
            "time_limit_context_seconds": TIME_LIMIT_CONTEXT_SECONDS,
            "optimization_allowed": False,
            "backend_optimize_guard_required": True,
            "timed_solver_runs_planned": 0,
            "full_model_feasibility_checked": False,
            "realized_service_checked": False,
            "gurobi_start_acceptance_checked": False,
            "incumbent_checked": False,
            "construction_entrypoint": "_construct_objective_partial_starts",
            "h10_requires_include_and_confirm": True,
            "determinism_check": "exact sparse-support and telemetry digest",
            "required_gates": [
                "sparse_partial_mip_assignments",
                "zero_lt_open_h_lt_candidate_h",
                "positive_estimated_service",
                "system_energy_proxy_use_le_supply",
                "preparation_within_0.25_seconds",
                "complete_objective_partial_telemetry",
                "two_identical_constructions",
            ],
        },
        "model_config": to_dict(_model_config()),
        "solver_context": to_dict(_solver_config()),
        "resource_lock": str(config.resource_lock),
        "process_table_sha256": frozen.process_table_sha256,
        "planned_constructions": [
            {"stage": stage, "repetition": repetition}
            for stage in config.stages
            for repetition in range(1, REPETITIONS + 1)
        ],
    }


def run_smoke(
    config: SmokeConfig,
    *,
    construction_runner: ConstructionRunner | None = None,
) -> dict[str, Any]:
    """Execute only start construction and persist a fail-closed audit."""

    config = _normalize_config(config)
    batch = _batch_config(config)
    dev._require_shared_resource_lock(config.resource_lock)
    runner = construction_runner or _construct_once
    try:
        with BenchmarkResourceLock(
            config.resource_lock, command="objective-start-construction-smoke"
        ):
            frozen = dev._freeze_inputs(batch)
            harness_path = dev._require_regular_file(
                config.repo_root / HARNESS_RELATIVE_PATH,
                kind="objective-start smoke harness",
            )
            config.output_root.mkdir(parents=False, exist_ok=False)
            manifest = _manifest_payload(
                config, frozen, _sha256_file(harness_path)
            )
            write_canonical_json_exclusive(
                config.output_root / "manifest.json", manifest
            )

            records: list[dict[str, Any]] = []
            process_audits: list[dict[str, Any]] = []
            stage_audits: list[dict[str, Any]] = []
            stop_reason: str | None = None
            for stage in config.stages:
                stage_records: list[dict[str, Any]] = []
                stage_digests: list[str | None] = []
                for repetition in range(1, REPETITIONS + 1):
                    try:
                        process_audit = _process_audit(stage, repetition)
                    except (MemoryError, KeyboardInterrupt, SystemExit):
                        raise
                    except Exception as error:
                        process_audit = {
                            "stage": stage,
                            "repetition": repetition,
                            "process_table_sha256": None,
                            "conflicting_processes": [],
                            "valid": False,
                            "error": f"{type(error).__name__}: {error}",
                        }
                    process_audits.append(process_audit)
                    if not process_audit["valid"]:
                        stop_reason = f"{stage}_repeat{repetition}_process_conflict"
                        break
                    instance = frozen.instances[stage]
                    try:
                        raw_record = dict(
                            runner(
                                stage,
                                Path(str(instance["path"])),
                                int(instance["candidate_h"]),
                                repetition,
                            )
                        )
                        audit = _audit_construction(
                            raw_record, candidate_h=int(instance["candidate_h"])
                        )
                        record = {**raw_record, "audit": audit}
                    except (MemoryError, KeyboardInterrupt, SystemExit):
                        raise
                    except Exception as error:
                        record = {
                            "stage": stage,
                            "repetition": repetition,
                            "timed_solver_run_started": False,
                            "optimization_api_call_count": 0,
                            "error": f"{type(error).__name__}: {error}",
                            "audit": {
                                "valid": False,
                                "errors": ["construction_exception"],
                                "determinism_sha256": None,
                            },
                        }
                        stop_reason = f"{stage}_repeat{repetition}_construction_exception"
                    records.append(record)
                    stage_records.append(record)
                    stage_digests.append(record["audit"]["determinism_sha256"])
                    if stop_reason is not None:
                        break

                complete = len(stage_records) == REPETITIONS
                individually_valid = complete and all(
                    record["audit"]["valid"] for record in stage_records
                )
                deterministic = (
                    complete
                    and stage_digests[0] is not None
                    and len(set(stage_digests)) == 1
                )
                passed = individually_valid and deterministic
                if not passed and stop_reason is None:
                    if complete and individually_valid:
                        stop_reason = f"{stage}_nondeterministic_construction"
                    else:
                        stop_reason = f"{stage}_construction_gate_failed"
                stage_audits.append(
                    {
                        "stage": stage,
                        "completed_repetitions": len(stage_records),
                        "complete": complete,
                        "individually_valid": individually_valid,
                        "deterministic": deterministic,
                        "determinism_sha256": (
                            stage_digests[0] if deterministic else None
                        ),
                        "passed": passed,
                    }
                )
                if not passed:
                    break

            postflight_errors = list(dev._postflight(batch, frozen))
            if postflight_errors and stop_reason is None:
                stop_reason = "postflight_drift"
            summary = {
                "artifact_type": "mlnd_objective_start_construction_smoke_summary",
                "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
                "development_only": True,
                "formal_evidence_eligible": False,
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                "status": "passed" if stop_reason is None else "failed_closed",
                "stop_reason": stop_reason,
                "timed_solver_run_started": False,
                "optimization_api_call_count": 0,
                "full_model_feasibility_checked": False,
                "realized_service_checked": False,
                "gurobi_start_acceptance_checked": False,
                "incumbent_checked": False,
                "construction_records": records,
                "process_audits": process_audits,
                "stage_audits": stage_audits,
                "postflight_errors": postflight_errors,
                "manifest": "manifest.json",
            }
            write_canonical_json_exclusive(
                config.output_root / "summary.json", summary
            )
            return summary
    except BenchmarkResourceLockError as error:
        raise ObjectiveStartSmokeError(str(error)) from error
    except dev.DevMicroCEError as error:
        raise ObjectiveStartSmokeError(str(error)) from error


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Construct and audit MX sparse structural MIP assignments and "
            "proxy telemetry twice without completing or optimizing them."
        )
    )
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--repo-root", default=str(MAIN_REPO_ROOT))
    parser.add_argument(
        "--formulation-source",
        default=None,
        help="must be this worktree's MLND/main.tex",
    )
    parser.add_argument(
        "--expected-formulation-sha256",
        default=dev.EXPECTED_FORMULATION_SHA256,
    )
    stage_group = parser.add_mutually_exclusive_group()
    stage_group.add_argument(
        "--stages",
        action="append",
        nargs="+",
        choices=tuple(dev.INSTANCE_SPECS),
        default=None,
        help=(
            "ordered micro stages; may be repeated as an option, but each stage "
            "may appear only once (default: H9)"
        ),
    )
    stage_group.add_argument(
        "--include-h10",
        action="store_true",
        help="legacy compatibility shorthand for --stages H9 H10",
    )
    parser.add_argument(
        "--confirm-h10",
        action="store_true",
        help="second explicit acknowledgement required before constructing H10",
    )
    parser.add_argument(
        "--resource-lock",
        default=str(dev.DEFAULT_BENCHMARK_RESOURCE_LOCK),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    requested_stages = (
        tuple(stage for group in args.stages for stage in group)
        if args.stages is not None
        else None
    )
    config = SmokeConfig(
        repo_root=Path(args.repo_root),
        output_root=Path(args.output_root),
        formulation_source=(
            Path(args.formulation_source) if args.formulation_source else None
        ),
        expected_formulation_sha256=args.expected_formulation_sha256,
        requested_stages=requested_stages,
        include_h10=args.include_h10,
        confirm_h10=args.confirm_h10,
        resource_lock=Path(args.resource_lock),
    )
    try:
        summary = run_smoke(config)
    except ObjectiveStartSmokeError as error:
        print(f"objective-start construction smoke refused to run: {error}", file=sys.stderr)
        return 2
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0 if summary["status"] == "passed" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "ConstructionHooks",
    "ObjectiveStartSmokeError",
    "SmokeConfig",
    "main",
    "run_smoke",
]
