from __future__ import annotations

import hashlib
import json
import threading
import time
import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any

import pytest

import src.benchmark.branch_cut.formal_execution as formal_execution_module
from src.benchmark.branch_cut.artifacts import (
    claim_attempt_directory,
    write_canonical_json_exclusive,
)
from src.benchmark.branch_cut.formal_environment import (
    FormalEnvironmentContract,
    GurobiVersions,
    ModuleDistributionVersions,
    OperatingSystemIdentity,
    PythonIdentity,
    SessionSnapshot,
)
from src.benchmark.branch_cut.formal_execution import (
    FORMAL_SLOT_BUNDLE_HASH_DOMAIN,
    AuthenticatedFormalSlotSnapshot,
    FormalExecutionBinding,
    FormalExecutionError,
    FormalInputAttestation,
    FormalSlotTask,
    PreparedFormalInput,
    _claim_pending_slot,
    execute_formal_block,
    finish_formal_session,
    prepare_formal_campaign as _prepare_formal_campaign,
    reconcile_formal_campaign,
    run_formal_block_schedule,
    seal_formal_campaign_evidence,
)
from src.benchmark.branch_cut.formal_plan import build_formal_execution_plan
from src.benchmark.branch_cut.formal_solution_evidence import (
    FormalQualificationTrace,
    bind_qualification_trace,
    build_solution_structure_from_facts,
    no_incumbent_solution_structure,
)
from src.benchmark.branch_cut.hashing import canonical_json, canonical_sha256
from src.benchmark.branch_cut.execution import build_core_solver_config
from src.benchmark.branch_cut.manifest import FreezeMetadata, freeze_manifest
from src.benchmark.branch_cut.pool import CandidateCase, CandidatePool
from src.benchmark.branch_cut.profiles import build_paper_profile_suite
from src.benchmark.branch_cut.qualification import (
    QualificationEvidence,
    SCALES,
    assess_qualification,
)
from src.benchmark.branch_cut.schema import BenchmarkMode, SolverConfig
from src.benchmark.branch_cut.telemetry import (
    APPLIED_GUROBI_PARAMETER_ATTRIBUTES,
    write_jsonl_exclusive,
)


_MANIFEST_ARTIFACT_SHA256 = "a" * 64
_MODEL_ID = "model-" + "b" * 64
_FORMAL_EXECUTION_IMPLEMENTATION_SHA256 = "c" * 64
_GUROBI_VERSION = "13.0.1-test"
_HOST_IDENTITY_SHA256 = "d" * 64
_SESSION_STARTED_AT = "2026-08-27T00:00:00Z"
_SESSION_ENDED_AT = "2026-08-27T00:01:00Z"
_RESOURCE_LOCK_PATH = "/tmp/mlnd-formal-test.lock"


def _environment_contract() -> FormalEnvironmentContract:
    return FormalEnvironmentContract(
        operating_system=OperatingSystemIdentity(
            name="Darwin",
            formal_version="15.6.1",
            kernel_release="24.6.0",
            kernel_version="Darwin Kernel Version 24.6.0",
        ),
        machine="arm64",
        pointer_bits=64,
        byteorder="little",
        cpu_model="Apple M4 Max",
        physical_cores=16,
        logical_cores=16,
        installed_ram_bytes=68_719_476_736,
        python=PythonIdentity("CPython", "3.13.5"),
        numpy=ModuleDistributionVersions("2.3.2", "2.3.2"),
        scipy=ModuleDistributionVersions("1.16.1", "1.16.1"),
        gurobipy=GurobiVersions(
            module_version=_GUROBI_VERSION,
            distribution_version=_GUROBI_VERSION,
            native_version=_GUROBI_VERSION,
        ),
    )


def _session_snapshot(*, pid: int = 1234) -> SessionSnapshot:
    return SessionSnapshot(
        pid=pid,
        executable="/opt/test/python",
        cwd="/test/worktree",
        affinity_cpu_ids=(0,),
        blas_environment=(("OMP_NUM_THREADS", "1"),),
        available_memory_bytes=8_000_000_000,
        load_average=(0.1, 0.2, 0.3),
    )


def prepare_formal_campaign(
    plan,
    results_root,
    current_binding_provider,
    pre_claim_preparer,
    *,
    session_snapshot: SessionSnapshot | None = None,
    resource_lock_path: str = _RESOURCE_LOCK_PATH,
    now_utc=lambda: _SESSION_STARTED_AT,
    session_nonce_provider=lambda: uuid.uuid4().hex,
):
    return _prepare_formal_campaign(
        plan,
        results_root,
        current_binding_provider,
        pre_claim_preparer,
        session_snapshot=session_snapshot or _session_snapshot(),
        resource_lock_path=resource_lock_path,
        now_utc=now_utc,
        session_nonce_provider=session_nonce_provider,
    )


def _candidate(case_id: str, scale: str, index: int) -> CandidateCase:
    return CandidateCase(
        case_id=case_id,
        scale=scale,
        demand_block_id=f"block-{index:02d}",
        layout_seed=11,
        instance_ref=case_id,
        instance_sha256=hashlib.sha256(case_id.encode()).hexdigest(),
        generator_parameters={"map_crop_scale": scale},
    )


def _accepted_decision(candidate: CandidateCase):
    return assess_qualification(
        QualificationEvidence(
            case_id=candidate.case_id,
            scale=candidate.scale,
            courier_only_objective=1_000.0,
            witness_objective=900.0,
            total_demand_per_hour=100.0,
            witness_service_per_hour=20.0,
            open_facilities=10,
            positive_cost_open_facilities=10,
            candidate_h=20,
            open_h=8,
            witness_feasible=True,
            qualification_run_id=f"qualification-{candidate.case_id}",
            instance_sha256=candidate.instance_sha256,
            active_h=8,
            active_h_threshold_per_hour=0.001,
        )
    )


def _manifest(case_prefix: str = "formal"):
    candidates = tuple(
        _candidate(f"{case_prefix}-{scale}-{index:02d}", scale, index)
        for scale in SCALES
        for index in range(12)
    )
    pool = CandidatePool(
        candidates=candidates,
        candidate_seed_order=(11,),
        pool_label=f"{case_prefix}-pool",
    )
    return freeze_manifest(
        pool,
        tuple(_accepted_decision(candidate) for candidate in candidates),
        freeze_metadata=FreezeMetadata(
            qualification_completed_at_utc="2026-08-26T23:58:00Z",
            manifest_frozen_at_utc="2026-08-26T23:59:00Z",
        ),
    )


def _plan(
    *,
    solver: SolverConfig | None = None,
    case_prefix: str = "formal",
    formal_execution_implementation_sha256: str = (
        _FORMAL_EXECUTION_IMPLEMENTATION_SHA256
    ),
):
    manifest = _manifest(case_prefix)
    plan = build_formal_execution_plan(
        manifest,
        manifest_artifact_sha256=_MANIFEST_ARTIFACT_SHA256,
        model_id=_MODEL_ID,
        formal_execution_implementation_sha256=(formal_execution_implementation_sha256),
        qualification_mode=BenchmarkMode.MX,
        gurobi_version=_GUROBI_VERSION,
        environment_contract=_environment_contract(),
        suite=build_paper_profile_suite(solver=solver),
    )
    return manifest, plan


def _binding(plan) -> FormalExecutionBinding:
    return FormalExecutionBinding(
        benchmark_manifest_sha256=plan.benchmark_manifest_sha256,
        manifest_artifact_sha256=plan.manifest_artifact_sha256,
        manifest_id=plan.manifest_id,
        model_id=plan.model_id,
        formal_execution_implementation_sha256=(
            plan.formal_execution_implementation_sha256
        ),
        qualification_mode=plan.qualification_mode,
        gurobi_version=plan.gurobi_version,
        environment_contract=plan.environment_contract,
        host_identity_sha256=_HOST_IDENTITY_SHA256,
    )


def _provider(binding: FormalExecutionBinding):
    return lambda: binding


def _prepared_input(
    task: FormalSlotTask,
    *,
    payload: object | None = None,
    expected_instance_sha256: str | None = None,
    payload_instance_sha256: str | None = None,
    source_sha256_before: str | None = None,
    source_sha256_after: str | None = None,
    case_id: str | None = None,
    slot_id: str | None = None,
) -> PreparedFormalInput:
    expected = expected_instance_sha256 or hashlib.sha256(
        task.block.case_id.encode()
    ).hexdigest()
    decision = assess_qualification(
        QualificationEvidence(
            case_id=case_id or task.block.case_id,
            scale=task.block.scale,
            courier_only_objective=1_000.0,
            witness_objective=900.0,
            total_demand_per_hour=100.0,
            witness_service_per_hour=20.0,
            open_facilities=10,
            positive_cost_open_facilities=10,
            candidate_h=20,
            open_h=8,
            witness_feasible=True,
            qualification_run_id=f"qualification-{task.block.case_id}",
            instance_sha256=expected,
            active_h=8,
            active_h_threshold_per_hour=0.001,
        )
    )
    trace = FormalQualificationTrace.from_decision(
        decision,
        manifest_id=task.pair.manifest_id,
        benchmark_manifest_sha256=task.benchmark_manifest_sha256,
        manifest_artifact_sha256=task.manifest_artifact_sha256,
    )
    return PreparedFormalInput(
        payload=object() if payload is None else payload,
        attestation=FormalInputAttestation(
            case_id=case_id or task.block.case_id,
            slot_id=slot_id or task.slot.slot_id,
            expected_instance_sha256=expected,
            payload_instance_sha256=payload_instance_sha256 or expected,
            source_sha256_before=source_sha256_before or expected,
            source_sha256_after=source_sha256_after or expected,
            qualification_trace=trace,
        ),
    )


def _prepare_input(task: FormalSlotTask) -> PreparedFormalInput:
    return _prepared_input(task)


def _expected_ids(task: FormalSlotTask) -> dict[str, Any]:
    return {
        "attempt_id": task.slot.attempt_id,
        "attempt_number": 1,
        "logical_run_id": task.slot.logical_run_id,
        "pair_id": task.slot.pair_id,
        "profile_id": task.slot.profile_id,
    }


def _complete_gurobi_parameter_audit(
    applied: dict[str, Any],
) -> dict[str, Any]:
    effective: dict[str, dict[str, Any]] = {}
    for applied_name, gurobi_name in APPLIED_GUROBI_PARAMETER_ATTRIBUTES:
        current = applied[applied_name]
        if isinstance(current, bool):
            raise AssertionError("synthetic applied parameters must not be boolean")
        if isinstance(current, int):
            value_type = "int"
            default: Any = current + 1
        elif isinstance(current, float):
            value_type = "float"
            default = current + 1.0
        elif isinstance(current, str):
            value_type = "str"
            default = current + "-default"
        else:
            raise AssertionError("unsupported synthetic applied parameter type")
        effective[gurobi_name] = {
            "classification": "core_solver_allowed",
            "current": current,
            "default": default,
            "value_type": value_type,
        }
    return {
        "gurobi_parameter_audit": {
            "schema_version": 1,
            "enumeration_complete": True,
            "reason_code": None,
            "enumerated_parameter_count": len(effective),
            "parameter_catalog_sha256": "9" * 64,
        },
        "effective_nondefault_parameters": effective,
        "unallowed_nondefault_parameters": [],
        "gurobi_execution_context": {
            "execution_mode": "local",
            "observation_basis": "gurobi_effective_parameters",
            "remote_identity_sha256": None,
            "remote_mechanisms": [],
        },
    }


def _replace_applied_and_effective_gurobi_parameters(
    path: Path,
    applied: dict[str, Any],
) -> None:
    record = json.loads(path.read_text(encoding="utf-8"))
    record["applied_gurobi_parameters"] = applied
    effective = record["effective_nondefault_parameters"]
    record["effective_nondefault_parameters"] = {
        gurobi_name: {
            **effective[gurobi_name],
            "current": applied[applied_name],
        }
        for applied_name, gurobi_name in APPLIED_GUROBI_PARAMETER_ATTRIBUTES
        if applied_name in applied
    }
    path.write_text(canonical_json(record) + "\n", encoding="utf-8")


def _write_bundle(
    task: FormalSlotTask,
    claim,
    _prepared: PreparedFormalInput | None = None,
    *,
    outcome: str = "completed_no_incumbent",
) -> None:
    prepared = _prepared or _prepared_input(task)
    qualification_trace = prepared.attestation.qualification_trace
    formal_context = bind_qualification_trace(
        task.formal_context,
        qualification_trace,
    )
    ids = _expected_ids(task)
    core_solver_config = asdict(
        build_core_solver_config(task.pair, task.profile, task.telemetry)
    )
    applied_gurobi_parameters = {
        "time_limit": 0.5 * core_solver_config["time_limit"],
        "mip_gap": core_solver_config["mip_gap"],
        "threads": core_solver_config["threads"],
        "output_flag": core_solver_config["output_flag"],
        "seed": core_solver_config["seed"],
        "method": core_solver_config["root_relaxation_method"],
        "miqcp_method": core_solver_config["miqcp_method"],
        "mip_focus": core_solver_config["mip_focus"],
        "heuristics": core_solver_config["heuristics"],
        "no_rel_heur_time": core_solver_config["no_rel_heur_time"],
        "presolve": core_solver_config["presolve"],
        "cuts": core_solver_config["cuts"],
        "root_cut_passes": core_solver_config["root_cut_passes"],
        "numeric_focus": core_solver_config["numeric_focus"],
        "feasibility_tol": core_solver_config["feasibility_tol"],
        "optimality_tol": core_solver_config["optimality_tol"],
        "integer_feasibility_tol": core_solver_config[
            "integer_feasibility_tol"
        ],
        "pre_crush": int(
            core_solver_config["use_user_cuts"]
            and core_solver_config["max_root_cut_rounds"] > 0
            and core_solver_config["max_cuts_per_round"] > 0
        ),
    }
    parameter_audit = _complete_gurobi_parameter_audit(
        applied_gurobi_parameters
    )
    write_canonical_json_exclusive(
        claim.directory / "run_spec.json",
        {
            "schema_version": 1,
            "ids": ids,
            "formal_context": formal_context,
            "pair": task.pair,
            "profile": task.profile,
            "profile_label": task.profile.label,
            "telemetry": task.telemetry,
            "core_solver_config": core_solver_config,
            "gurobi_version": task.gurobi_version,
        },
    )
    failed = outcome == "failed"
    has_incumbent = outcome == "completed_with_incumbent"
    solution_structure = (
        build_solution_structure_from_facts(
            qualification_trace,
            C0=1_000.0,
            incumbent_objective=900.0,
            total_demand_per_hour=100.0,
            realized_service_per_hour=20.0,
            open_facility_count=10,
            positive_cost_open_facility_count=10,
            candidate_H_count=20,
            open_H_count=8,
            active_H_count=8,
            active_H_threshold_per_hour=0.001,
        )
        if has_incumbent
        else no_incumbent_solution_structure(qualification_trace)
    )
    failure = {"stage": "solve", "message": "synthetic failure"} if failed else None
    if failed:
        solver_metrics = {"has_incumbent": False}
        progress: tuple[dict[str, Any], ...] = ()
        cut_stats: dict[str, Any] = {"captured": False}
    else:
        objective = 100.0 if has_incumbent else None
        bound = 95.0
        gap = 0.05 if has_incumbent else None
        solver_metrics = {
            "status": 9,
            "status_name": "TIME_LIMIT",
            "has_incumbent": has_incumbent,
            "solution_count": int(has_incumbent),
            "objective": objective,
            "best_bound": bound,
            "relative_gap": gap,
            "node_count": 12.0,
            "main_solve_runtime_seconds": 40.0,
            "total_algorithm_runtime_seconds": 45.0,
            "algorithm_prepare_runtime_seconds": 3.0,
            "start_preparation_runtime_seconds": 2.0,
            "first_incumbent_time_seconds": 5.0 if has_incumbent else None,
            "first_incumbent_objective": 110.0 if has_incumbent else None,
            "root_bound_before_cuts": None,
            "root_bound_after_cuts": None,
            "callback_runtime_seconds": 0.25,
            "budget_exhausted_before_main_solve": False,
            "main_solve_started": True,
            "optimize_called": True,
        }
        incumbent_events = (
            (
                {
                    "event": "incumbent",
                    "runtime_seconds": 5.0,
                    "incumbent": 110.0,
                    "bound": 90.0,
                    "relative_gap": 20.0 / 110.0,
                },
            )
            if has_incumbent
            else ()
        )
        progress = (
            *incumbent_events,
            {
                "event": "final",
                "runtime_seconds": 40.0,
                "total_algorithm_runtime_seconds": 45.0,
                "budget_exhausted_before_main_solve": False,
                "main_solve_started": True,
                "optimize_called": True,
                "incumbent": objective,
                "bound": bound,
                "relative_gap": gap,
                "node_count": 12.0,
                "solution_count": int(has_incumbent),
                "status": 9,
            },
        )
        cut_stats = {
            "captured": True,
            "static_rows_added_by_family": {},
            "user_cuts_added_by_family": {},
            "user_cut_candidates_by_family": {},
            "user_cut_duplicates_by_family": {},
            "user_cut_rejections_by_family": {},
            "user_cut_separation_seconds_by_family": {},
            "route_bundle_index_counts": {
                "swap_route_options": 0,
                "swap_leg_bundles": 0,
                "swap_station_bundles": 0,
                "strengthening_swap_leg_bundles": 0,
                "strengthening_swap_station_bundles": 0,
            },
            "static_rows_added_total": 0,
            "user_cuts_added_total": 0,
            "root_cut_rounds": 0,
            "callback_runtime_seconds": 0.25,
        }
    write_canonical_json_exclusive(
        claim.directory / "run_record.json",
        {
            "schema_version": 1,
            "ids": ids,
            "formal_context": formal_context,
            "profile_code": task.slot.profile_code.value,
            "profile_label": task.profile.label,
            "outcome": outcome,
            "completed_without_exception": not failed,
            "solve_returned_incumbent": has_incumbent,
            "solver_metrics": solver_metrics,
            "execution_wall_runtime_seconds": 46.0,
            "budget_exhausted_before_main_solve": (
                None if failed else False
            ),
            "main_solve_started": None if failed else True,
            "optimize_called": None if failed else True,
            "solution_structure": solution_structure.to_payload(),
            "solution_structure_sha256": canonical_sha256(
                solution_structure.to_payload()
            ),
            "gurobi_version": task.gurobi_version,
            "core_solver_config": core_solver_config,
            "applied_gurobi_parameters": applied_gurobi_parameters,
            **parameter_audit,
            "telemetry_counts": {
                "progress_events": len(progress),
                "root_round_events": 0,
                "static_rows_added": (
                    cut_stats.get("static_rows_added_total")
                ),
                "user_cuts_added": cut_stats.get("user_cuts_added_total"),
            },
            "failure": failure,
        },
    )
    if failed:
        write_canonical_json_exclusive(
            claim.directory / "failure.json",
            {
                "schema_version": 1,
                "ids": ids,
                "formal_context": formal_context,
                **failure,
            },
        )
    (claim.directory / "solver.log").write_text(
        "synthetic solver log\n",
        encoding="utf-8",
    )
    write_jsonl_exclusive(claim.directory / "progress.jsonl", progress)
    write_jsonl_exclusive(claim.directory / "root_rounds.jsonl", ())
    write_canonical_json_exclusive(
        claim.directory / "cut_stats.json",
        cut_stats,
    )


def _task_for(campaign, block, slot) -> FormalSlotTask:
    return FormalSlotTask(
        campaign_id=campaign.identity.campaign_id,
        campaign_identity_sha256=campaign.identity.sha256,
        benchmark_manifest_sha256=campaign.plan.benchmark_manifest_sha256,
        manifest_artifact_sha256=campaign.plan.manifest_artifact_sha256,
        formal_plan_id=campaign.plan.plan_id,
        formal_plan_sha256=campaign.plan.sha256,
        formal_execution_implementation_sha256=(
            campaign.plan.formal_execution_implementation_sha256
        ),
        model_id=campaign.plan.model_id,
        gurobi_version=campaign.plan.gurobi_version,
        session_id=campaign.session.session_id,
        block=block,
        slot=slot,
        pair=campaign.plan.blocks[block.block_index] and task_pair(campaign, block),
        profile=campaign.plan.suite.profile(slot.profile_code),
        telemetry=campaign.plan.suite.telemetry,
    )


def task_pair(campaign, block):
    from src.benchmark.branch_cut.schema import PairSpec

    return PairSpec(
        manifest_id=campaign.plan.manifest_id,
        case_id=block.case_id,
        mode=campaign.plan.qualification_mode,
        solver_seed=block.solver_seed,
        time_budget_seconds=block.time_budget_seconds,
        solver=campaign.plan.suite.solver,
    )


def _claim_and_write(campaign, block, slot, *, outcome="completed_no_incumbent"):
    claim, task, prepared = _claim_pending_slot(campaign, block, slot)
    _write_bundle(task, claim, prepared, outcome=outcome)
    return claim


def _replace_json_field(path: Path, field_name: str, value: object) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload[field_name] = value
    path.write_text(json.dumps(payload), encoding="utf-8")


def _synthetic_authenticated_slots(
    campaign,
    *,
    marker: str = "original",
) -> tuple[AuthenticatedFormalSlotSnapshot, ...]:
    snapshots: list[AuthenticatedFormalSlotSnapshot] = []
    for block in campaign.plan.blocks:
        for slot in block.slots:
            file_sha256 = tuple(
                (name, hashlib.sha256(f"{marker}:{slot.slot_id}:{name}".encode()).hexdigest())
                for name in sorted(
                    {
                        "claim.json",
                        "cut_stats.json",
                        "formal_claim.json",
                        "progress.jsonl",
                        "root_rounds.jsonl",
                        "run_record.json",
                        "run_spec.json",
                        "solver.log",
                    }
                )
            )
            bundle_sha256 = canonical_sha256(
                {
                    "domain": FORMAL_SLOT_BUNDLE_HASH_DOMAIN,
                    "slot_id": slot.slot_id,
                    "slot_index": slot.slot_index,
                    "attempt_id": slot.attempt_id,
                    "file_sha256": dict(file_sha256),
                }
            )
            snapshots.append(
                AuthenticatedFormalSlotSnapshot(
                    slot_id=slot.slot_id,
                    slot_index=slot.slot_index,
                    block_id=block.block_id,
                    logical_run_id=slot.logical_run_id,
                    session_id=campaign.session.session_id,
                    outcome="completed_no_incumbent",
                    completed_without_exception=True,
                    has_incumbent=False,
                    run_spec=MappingProxyType({"slot_id": slot.slot_id}),
                    run_record=MappingProxyType(
                        {
                            "slot_id": slot.slot_id,
                            "solver_metrics": MappingProxyType(
                                {"has_incumbent": False, "relative_gap": 0.5}
                            ),
                        }
                    ),
                    progress=(),
                    root_rounds=(),
                    cut_stats=MappingProxyType({"captured": True}),
                    file_sha256=file_sha256,
                    attempt_bundle_sha256=bundle_sha256,
                    checkpoint_bytes_sha256=hashlib.sha256(
                        f"{marker}:checkpoint:{slot.slot_id}".encode()
                    ).hexdigest(),
                )
            )
    return tuple(snapshots)


def test_campaign_identity_cross_checks_environment_and_namespaces_plan(
    tmp_path: Path,
) -> None:
    _manifest_value, plan = _plan()
    binding = _binding(plan)
    results_root = tmp_path / "formal-results"
    campaign = prepare_formal_campaign(
        plan,
        results_root,
        _provider(binding),
        _prepare_input,
    )

    assert campaign.results_root == results_root.resolve()
    assert campaign.plan_root == results_root.resolve() / plan.plan_id
    assert campaign.root == campaign.plan_root / campaign.identity.campaign_id
    assert campaign.attempts_root == campaign.root / "attempts"
    assert campaign.sessions_root == campaign.root / "sessions"
    assert campaign.identity.formal_plan_id == plan.plan_id
    assert campaign.identity.manifest_artifact_sha256 == plan.manifest_artifact_sha256
    assert campaign.identity.model_id == plan.model_id
    assert (
        campaign.identity.formal_execution_implementation_sha256
        == plan.formal_execution_implementation_sha256
    )
    assert campaign.identity.qualification_mode is plan.qualification_mode
    assert campaign.identity.gurobi_version == plan.gurobi_version
    assert campaign.identity.environment_contract == plan.environment_contract
    assert (
        campaign.identity.environment_contract_sha256
        == plan.environment_contract.sha256
    )
    assert campaign.identity.host_identity_sha256 == _HOST_IDENTITY_SHA256
    assert campaign.identity.attempt_number == 1
    assert "results_root" not in campaign.identity.to_payload()
    assert "campaign_root" not in campaign.identity.to_payload()

    original_identity = campaign.identity_path.read_bytes()
    reopened = prepare_formal_campaign(
        plan,
        results_root,
        _provider(binding),
        _prepare_input,
    )
    assert reopened.identity == campaign.identity
    assert reopened.identity_path.read_bytes() == original_identity
    assert reopened.session.session_id != campaign.session.session_id
    assert reopened.session_root.parent == campaign.session_root.parent

    changed_bindings = (
        replace(binding, benchmark_manifest_sha256="0" * 64),
        replace(
            binding,
            manifest_artifact_sha256="d" * 64,
            manifest_id="manifest-" + "d" * 64,
        ),
        replace(binding, model_id="model-" + "f" * 64),
        replace(binding, formal_execution_implementation_sha256="e" * 64),
        replace(binding, qualification_mode=BenchmarkMode.LO),
        replace(binding, gurobi_version="99.0-drift"),
        replace(
            binding,
            environment_contract=replace(
                binding.environment_contract,
                cpu_model="different cpu",
            ),
        ),
    )
    for changed in changed_bindings:
        with pytest.raises(FormalExecutionError, match="identity differs"):
            prepare_formal_campaign(
                plan,
                results_root,
                _provider(changed),
                _prepare_input,
            )

    _other_manifest, other_plan = _plan(
        formal_execution_implementation_sha256="d" * 64,
    )
    other_campaign = prepare_formal_campaign(
        other_plan,
        results_root,
        _provider(_binding(other_plan)),
        _prepare_input,
    )
    assert other_campaign.root != campaign.root
    assert other_campaign.plan_root != campaign.plan_root
    assert other_campaign.plan_root.parent == campaign.plan_root.parent
    assert other_plan.blocks == plan.blocks
    assert (
        other_plan.blocks[0].slots[0].logical_run_id
        == plan.blocks[0].slots[0].logical_run_id
    )


def test_campaign_identity_is_path_independent_and_host_namespaced(
    tmp_path: Path,
) -> None:
    _manifest_value, plan = _plan()
    binding = _binding(plan)
    first = prepare_formal_campaign(
        plan,
        tmp_path / "first-location",
        _provider(binding),
        _prepare_input,
    )
    relocated = prepare_formal_campaign(
        plan,
        tmp_path / "second-location",
        _provider(binding),
        _prepare_input,
    )
    assert relocated.identity == first.identity
    assert relocated.identity.campaign_id == first.identity.campaign_id
    assert relocated.root != first.root

    other_host_binding = replace(binding, host_identity_sha256="e" * 64)
    other_host = prepare_formal_campaign(
        plan,
        tmp_path / "first-location",
        _provider(other_host_binding),
        _prepare_input,
    )
    assert other_host.plan_root == first.plan_root
    assert other_host.identity.campaign_id != first.identity.campaign_id
    assert other_host.root != first.root
    assert tuple(first.attempts_root.iterdir()) == ()
    assert tuple(other_host.attempts_root.iterdir()) == ()


def test_campaign_session_artifacts_are_immutable_and_record_volatile_state(
    tmp_path: Path,
) -> None:
    _manifest_value, plan = _plan()
    snapshot = _session_snapshot(pid=9876)
    campaign = prepare_formal_campaign(
        plan,
        tmp_path,
        _provider(_binding(plan)),
        _prepare_input,
        session_snapshot=snapshot,
        resource_lock_path="/locks/formal.lock",
        now_utc=lambda: _SESSION_STARTED_AT,
        session_nonce_provider=lambda: "session-nonce-001",
    )
    start = json.loads(campaign.session_start_path.read_text(encoding="utf-8"))
    assert start == campaign.session.start_artifact_payload()
    assert start["session_id"] == campaign.session.session_id
    assert start["session"]["started_at_utc"] == _SESSION_STARTED_AT
    assert start["session"]["resource_lock_path"] == "/locks/formal.lock"
    assert start["session"]["session_snapshot"]["pid"] == 9876
    assert start["session"]["session_snapshot"]["available_memory_bytes"] == (
        8_000_000_000
    )
    assert start["session"]["session_snapshot"]["volatile"] is True
    assert start["session"]["session_snapshot"]["blas_environment"] == {
        "OMP_NUM_THREADS": "1"
    }

    finish_formal_session(
        campaign,
        status="completed",
        now_utc=lambda: _SESSION_ENDED_AT,
    )
    end = json.loads(campaign.session_end_path.read_text(encoding="utf-8"))
    assert end["session_id"] == campaign.session.session_id
    assert end["started_at_utc"] == _SESSION_STARTED_AT
    assert end["ended_at_utc"] == _SESSION_ENDED_AT
    assert end["status"] == "completed"
    with pytest.raises(FormalExecutionError, match="already exists"):
        finish_formal_session(
            campaign,
            status="completed",
            now_utc=lambda: _SESSION_ENDED_AT,
        )
    with pytest.raises(FormalExecutionError, match="session has ended"):
        execute_formal_block(
            campaign,
            plan.blocks[0],
            _write_bundle,
            campaign.current_binding_provider,
        )
    assert tuple(campaign.attempts_root.iterdir()) == ()


def test_host_drift_is_rejected_before_pending_claim(tmp_path: Path) -> None:
    _manifest_value, plan = _plan()
    expected = _binding(plan)
    drifted = replace(expected, host_identity_sha256="e" * 64)
    provider_calls = 0

    def provider() -> FormalExecutionBinding:
        nonlocal provider_calls
        provider_calls += 1
        return expected if provider_calls == 1 else drifted

    campaign = prepare_formal_campaign(
        plan,
        tmp_path,
        provider,
        _prepare_input,
    )
    with pytest.raises(FormalExecutionError, match="host_identity_sha256"):
        execute_formal_block(campaign, plan.blocks[0], _write_bundle, provider)
    assert tuple(campaign.attempts_root.iterdir()) == ()


def test_nonempty_root_without_identity_and_tampered_identity_fail_closed(
    tmp_path: Path,
) -> None:
    _manifest_value, plan = _plan()
    binding = _binding(plan)
    results_root = tmp_path / "results"
    plan_root = results_root / plan.plan_id
    plan_root.mkdir(parents=True)
    (plan_root / "stray.txt").write_text("unowned", encoding="utf-8")

    with pytest.raises(FormalExecutionError, match="unknown entry"):
        prepare_formal_campaign(
            plan,
            results_root,
            _provider(binding),
            _prepare_input,
        )

    clean_root = tmp_path / "clean-results"
    campaign = prepare_formal_campaign(
        plan,
        clean_root,
        _provider(binding),
        _prepare_input,
    )
    identity = json.loads(campaign.identity_path.read_text(encoding="utf-8"))
    identity["campaign_identity"]["formal_execution_implementation_sha256"] = "d" * 64
    campaign.identity_path.write_text(json.dumps(identity), encoding="utf-8")
    with pytest.raises(FormalExecutionError, match="campaign identity differs"):
        prepare_formal_campaign(
            plan,
            clean_root,
            _provider(binding),
            _prepare_input,
        )


def test_prepare_rejects_symlinked_results_plan_and_campaign_roots(
    tmp_path: Path,
) -> None:
    _manifest_value, plan = _plan()
    binding = _binding(plan)

    actual_results = tmp_path / "actual-results"
    actual_results.mkdir()
    results_link = tmp_path / "results-link"
    results_link.symlink_to(actual_results, target_is_directory=True)
    with pytest.raises(FormalExecutionError, match="results root.*non-symlink"):
        prepare_formal_campaign(
            plan,
            results_link,
            _provider(binding),
            _prepare_input,
        )

    plan_results = tmp_path / "plan-results"
    plan_results.mkdir()
    external_plan = tmp_path / "external-plan"
    external_plan.mkdir()
    (plan_results / plan.plan_id).symlink_to(
        external_plan,
        target_is_directory=True,
    )
    with pytest.raises(FormalExecutionError, match="plan root.*non-symlink"):
        prepare_formal_campaign(
            plan,
            plan_results,
            _provider(binding),
            _prepare_input,
        )

    seed = prepare_formal_campaign(
        plan,
        tmp_path / "seed-results",
        _provider(binding),
        _prepare_input,
    )
    campaign_results = tmp_path / "campaign-results"
    campaign_plan = campaign_results / plan.plan_id
    campaign_plan.mkdir(parents=True)
    external_campaign = tmp_path / "external-campaign"
    external_campaign.mkdir()
    (campaign_plan / seed.identity.campaign_id).symlink_to(
        external_campaign,
        target_is_directory=True,
    )
    with pytest.raises(FormalExecutionError, match="campaign root.*non-symlink"):
        prepare_formal_campaign(
            plan,
            campaign_results,
            _provider(binding),
            _prepare_input,
        )


@pytest.mark.parametrize(
    ("attribute", "label"),
    (
        ("attempts_root", "attempts root"),
        ("journal_root", "journal root"),
        ("sessions_root", "sessions root"),
    ),
)
def test_reconcile_rejects_symlinked_campaign_container_roots(
    tmp_path: Path,
    attribute: str,
    label: str,
) -> None:
    _manifest_value, plan = _plan()
    campaign = prepare_formal_campaign(
        plan,
        tmp_path / attribute,
        _provider(_binding(plan)),
        _prepare_input,
    )
    container = getattr(campaign, attribute)
    moved = tmp_path / f"real-{attribute}"
    container.rename(moved)
    container.symlink_to(moved, target_is_directory=True)

    with pytest.raises(FormalExecutionError, match=rf"{label}.*non-symlink"):
        reconcile_formal_campaign(campaign)


@pytest.mark.parametrize("level", ("logical-run", "attempt"))
def test_reconcile_rejects_symlinked_attempt_container_directories(
    tmp_path: Path,
    level: str,
) -> None:
    _manifest_value, plan = _plan()
    campaign = prepare_formal_campaign(
        plan,
        tmp_path / level,
        _provider(_binding(plan)),
        _prepare_input,
    )
    slot = plan.blocks[0].slots[0]
    logical_root = campaign.attempts_root / slot.logical_run_id
    external = tmp_path / f"external-{level}"
    external.mkdir()
    if level == "logical-run":
        logical_root.symlink_to(external, target_is_directory=True)
        expected_label = "logical-run directory"
    else:
        logical_root.mkdir()
        (logical_root / slot.attempt_id).symlink_to(
            external,
            target_is_directory=True,
        )
        expected_label = "attempt directory"

    with pytest.raises(FormalExecutionError, match=rf"{expected_label}.*non-symlink"):
        reconcile_formal_campaign(campaign)


def test_reconcile_rejects_symlinked_session_directory(tmp_path: Path) -> None:
    _manifest_value, plan = _plan()
    campaign = prepare_formal_campaign(
        plan,
        tmp_path / "session",
        _provider(_binding(plan)),
        _prepare_input,
    )
    moved = tmp_path / "real-session"
    campaign.session_root.rename(moved)
    campaign.session_root.symlink_to(moved, target_is_directory=True)

    with pytest.raises(FormalExecutionError, match="session root.*non-symlink"):
        reconcile_formal_campaign(campaign)


def test_complete_attempt_without_checkpoint_is_recovered_and_then_skipped(
    tmp_path: Path,
) -> None:
    _manifest_value, plan = _plan()
    binding = _binding(plan)
    binding_checks_allowed = True

    def binding_provider() -> FormalExecutionBinding:
        if not binding_checks_allowed:
            raise AssertionError("completed slots must not request a new binding")
        return binding

    campaign = prepare_formal_campaign(
        plan,
        tmp_path,
        binding_provider,
        _prepare_input,
    )
    block = plan.blocks[0]
    first = block.slots[0]
    original_session_id = campaign.session.session_id
    _claim_and_write(campaign, block, first, outcome="failed")

    state = reconcile_formal_campaign(campaign)
    assert state.completed_slot_ids == (first.slot_id,)
    assert state.recovered_slot_ids == (first.slot_id,)
    assert len(state.pending_slot_ids) == 899
    checkpoint = campaign.journal_root / "slot-0000.json"
    checkpoint_payload = json.loads(checkpoint.read_text(encoding="utf-8"))
    assert checkpoint_payload["recovered_from_attempt_bundle"] is True
    assert checkpoint_payload["outcome"] == "failed"
    assert checkpoint_payload["session_id"] == original_session_id
    assert (
        checkpoint_payload["formal_execution_implementation_sha256"]
        == plan.formal_execution_implementation_sha256
    )
    assert set(checkpoint_payload["bundle_file_sha256"]) == {
        "claim.json",
        "cut_stats.json",
        "failure.json",
        "formal_claim.json",
        "progress.jsonl",
        "root_rounds.jsonl",
        "run_record.json",
        "run_spec.json",
        "solver.log",
    }
    checkpoint_bytes = checkpoint.read_bytes()

    campaign = prepare_formal_campaign(
        plan,
        tmp_path,
        binding_provider,
        _prepare_input,
    )
    assert campaign.session.session_id != original_session_id
    resumed_state = reconcile_formal_campaign(campaign)
    assert resumed_state.completed_slot_ids == (first.slot_id,)

    executed_codes: list[str] = []

    def execute(
        task: FormalSlotTask,
        claim,
        prepared: PreparedFormalInput,
    ) -> None:
        executed_codes.append(task.slot.profile_code.value)
        _write_bundle(task, claim, prepared)

    result = execute_formal_block(campaign, block, execute, binding_provider)
    assert result.skipped_slot_ids == (first.slot_id,)
    assert result.executed_slot_ids == tuple(slot.slot_id for slot in block.slots[1:])
    assert executed_codes == [slot.profile_code.value for slot in block.slots[1:]]
    assert checkpoint.read_bytes() == checkpoint_bytes

    executed_codes.clear()
    binding_checks_allowed = False
    second = execute_formal_block(
        campaign,
        block,
        execute,
        binding_provider,
    )
    assert second.executed_slot_ids == ()
    assert second.skipped_slot_ids == tuple(slot.slot_id for slot in block.slots)
    assert executed_codes == []


def test_recovery_rejects_partial_two_field_core_solver_config(
    tmp_path: Path,
) -> None:
    _manifest_value, plan = _plan()
    campaign = prepare_formal_campaign(
        plan,
        tmp_path,
        _provider(_binding(plan)),
        _prepare_input,
    )
    block = plan.blocks[0]
    slot = block.slots[0]
    claim = _claim_and_write(campaign, block, slot)
    partial = {"threads": 1, "seed": block.solver_seed}
    _replace_json_field(claim.directory / "run_spec.json", "core_solver_config", partial)
    _replace_json_field(
        claim.directory / "run_record.json",
        "core_solver_config",
        partial,
    )

    with pytest.raises(FormalExecutionError, match="run_spec core solver config"):
        reconcile_formal_campaign(campaign)


def test_recovery_rejects_run_record_core_solver_config_drift(
    tmp_path: Path,
) -> None:
    _manifest_value, plan = _plan()
    campaign = prepare_formal_campaign(
        plan,
        tmp_path,
        _provider(_binding(plan)),
        _prepare_input,
    )
    block = plan.blocks[0]
    claim = _claim_and_write(campaign, block, block.slots[0])
    record_path = claim.directory / "run_record.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    record["core_solver_config"]["threads"] = 2
    record_path.write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(FormalExecutionError, match="run_record core solver config"):
        reconcile_formal_campaign(campaign)


def test_completed_bundle_requires_full_exact_applied_gurobi_parameters(
    tmp_path: Path,
) -> None:
    _manifest_value, plan = _plan()

    partial_campaign = prepare_formal_campaign(
        plan,
        tmp_path / "partial",
        _provider(_binding(plan)),
        _prepare_input,
    )
    block = plan.blocks[0]
    partial_claim = _claim_and_write(partial_campaign, block, block.slots[0])
    _replace_json_field(
        partial_claim.directory / "run_record.json",
        "applied_gurobi_parameters",
        {"threads": 1, "seed": block.solver_seed},
    )
    with pytest.raises(FormalExecutionError, match="parameters are incomplete"):
        reconcile_formal_campaign(partial_campaign)

    drift_campaign = prepare_formal_campaign(
        plan,
        tmp_path / "drift",
        _provider(_binding(plan)),
        _prepare_input,
    )
    drift_claim = _claim_and_write(drift_campaign, block, block.slots[0])
    record_path = drift_claim.directory / "run_record.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    record["applied_gurobi_parameters"]["miqcp_method"] = 1
    record_path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(FormalExecutionError, match="parameters differ.*miqcp_method"):
        reconcile_formal_campaign(drift_campaign)


def test_applied_time_limit_must_be_remaining_frozen_budget(
    tmp_path: Path,
) -> None:
    _manifest_value, plan = _plan()
    block = plan.blocks[0]

    for label, allowed_time_limit in (
        ("zero", 0.0),
        ("full", block.time_budget_seconds),
    ):
        campaign = prepare_formal_campaign(
            plan,
            tmp_path / label,
            _provider(_binding(plan)),
            _prepare_input,
        )
        claim = _claim_and_write(campaign, block, block.slots[0])
        record_path = claim.directory / "run_record.json"
        record = json.loads(record_path.read_text(encoding="utf-8"))
        record["applied_gurobi_parameters"]["time_limit"] = allowed_time_limit
        record["effective_nondefault_parameters"]["TimeLimit"][
            "current"
        ] = allowed_time_limit
        record_path.write_text(json.dumps(record), encoding="utf-8")
        state = reconcile_formal_campaign(campaign)
        assert state.recovered_slot_ids == (block.slots[0].slot_id,)

    invalid_campaign = prepare_formal_campaign(
        plan,
        tmp_path / "over-budget",
        _provider(_binding(plan)),
        _prepare_input,
    )
    invalid_claim = _claim_and_write(invalid_campaign, block, block.slots[0])
    record_path = invalid_claim.directory / "run_record.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    record["applied_gurobi_parameters"]["time_limit"] = (
        block.time_budget_seconds + 1e-6
    )
    record_path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(FormalExecutionError, match="remaining-budget interval"):
        reconcile_formal_campaign(invalid_campaign)


def test_failed_bundle_allows_partial_parameters_but_audits_present_values(
    tmp_path: Path,
) -> None:
    _manifest_value, plan = _plan()
    block = plan.blocks[0]

    partial_campaign = prepare_formal_campaign(
        plan,
        tmp_path / "matching-partial",
        _provider(_binding(plan)),
        _prepare_input,
    )
    partial_claim = _claim_and_write(
        partial_campaign,
        block,
        block.slots[0],
        outcome="failed",
    )
    _replace_applied_and_effective_gurobi_parameters(
        partial_claim.directory / "run_record.json",
        {"threads": 1, "seed": block.solver_seed},
    )
    state = reconcile_formal_campaign(partial_campaign)
    assert state.recovered_slot_ids == (block.slots[0].slot_id,)

    drift_campaign = prepare_formal_campaign(
        plan,
        tmp_path / "drifted-partial",
        _provider(_binding(plan)),
        _prepare_input,
    )
    drift_claim = _claim_and_write(
        drift_campaign,
        block,
        block.slots[0],
        outcome="failed",
    )
    _replace_applied_and_effective_gurobi_parameters(
        drift_claim.directory / "run_record.json",
        {"threads": 2},
    )
    with pytest.raises(FormalExecutionError, match="parameters differ.*threads"):
        reconcile_formal_campaign(drift_campaign)


def test_completed_bundle_requires_gurobi_parameter_audit_fields(
    tmp_path: Path,
) -> None:
    _manifest_value, plan = _plan()
    block = plan.blocks[0]
    campaign = prepare_formal_campaign(
        plan,
        tmp_path,
        _provider(_binding(plan)),
        _prepare_input,
    )
    claim = _claim_and_write(campaign, block, block.slots[0])
    record_path = claim.directory / "run_record.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    del record["gurobi_parameter_audit"]
    record_path.write_text(canonical_json(record) + "\n", encoding="utf-8")

    with pytest.raises(FormalExecutionError, match="audit metadata is invalid"):
        reconcile_formal_campaign(campaign)


@pytest.mark.parametrize(
    ("field_name", "value", "message"),
    (
        ("schema_version", 2, "audit schema differs"),
        ("reason_code", "unexpected_reason", "complete.*has a reason"),
        ("enumerated_parameter_count", 0, "audit count is invalid"),
        ("parameter_catalog_sha256", "not-a-sha256", "catalog hash is invalid"),
    ),
)
def test_completed_bundle_rejects_invalid_gurobi_audit_metadata(
    tmp_path: Path,
    field_name: str,
    value: Any,
    message: str,
) -> None:
    _manifest_value, plan = _plan()
    block = plan.blocks[0]
    campaign = prepare_formal_campaign(
        plan,
        tmp_path / field_name,
        _provider(_binding(plan)),
        _prepare_input,
    )
    claim = _claim_and_write(campaign, block, block.slots[0])
    record_path = claim.directory / "run_record.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    record["gurobi_parameter_audit"][field_name] = value
    record_path.write_text(canonical_json(record) + "\n", encoding="utf-8")

    with pytest.raises(FormalExecutionError, match=message):
        reconcile_formal_campaign(campaign)


def test_completed_bundle_rejects_incomplete_or_remote_gurobi_audit(
    tmp_path: Path,
) -> None:
    _manifest_value, plan = _plan()
    block = plan.blocks[0]

    incomplete_campaign = prepare_formal_campaign(
        plan,
        tmp_path / "incomplete",
        _provider(_binding(plan)),
        _prepare_input,
    )
    incomplete_claim = _claim_and_write(
        incomplete_campaign,
        block,
        block.slots[0],
    )
    incomplete_path = incomplete_claim.directory / "run_record.json"
    incomplete = json.loads(incomplete_path.read_text(encoding="utf-8"))
    incomplete["gurobi_parameter_audit"] = {
        "schema_version": 1,
        "enumeration_complete": False,
        "reason_code": "parameter_enumeration_failed",
        "enumerated_parameter_count": 0,
        "parameter_catalog_sha256": None,
    }
    incomplete["effective_nondefault_parameters"] = {}
    incomplete["unallowed_nondefault_parameters"] = []
    incomplete["gurobi_execution_context"] = {
        "execution_mode": "unobserved",
        "observation_basis": "gurobi_effective_parameters",
        "remote_identity_sha256": None,
        "remote_mechanisms": [],
    }
    incomplete_path.write_text(
        canonical_json(incomplete) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(FormalExecutionError, match="requires a complete"):
        reconcile_formal_campaign(incomplete_campaign)

    remote_campaign = prepare_formal_campaign(
        plan,
        tmp_path / "remote",
        _provider(_binding(plan)),
        _prepare_input,
    )
    remote_claim = _claim_and_write(remote_campaign, block, block.slots[0])
    remote_path = remote_claim.directory / "run_record.json"
    remote = json.loads(remote_path.read_text(encoding="utf-8"))
    remote["gurobi_execution_context"] = {
        "execution_mode": "remote",
        "observation_basis": "gurobi_effective_parameters",
        "remote_identity_sha256": "8" * 64,
        "remote_mechanisms": ["compute_server"],
    }
    remote_path.write_text(canonical_json(remote) + "\n", encoding="utf-8")
    with pytest.raises(FormalExecutionError, match="must be local"):
        reconcile_formal_campaign(remote_campaign)


def test_completed_bundle_rejects_effective_classification_and_unallowed_drift(
    tmp_path: Path,
) -> None:
    _manifest_value, plan = _plan()
    block = plan.blocks[0]

    classification_campaign = prepare_formal_campaign(
        plan,
        tmp_path / "classification",
        _provider(_binding(plan)),
        _prepare_input,
    )
    classification_claim = _claim_and_write(
        classification_campaign,
        block,
        block.slots[0],
    )
    classification_path = classification_claim.directory / "run_record.json"
    classification = json.loads(classification_path.read_text(encoding="utf-8"))
    classification["effective_nondefault_parameters"]["Threads"][
        "classification"
    ] = "unallowed"
    classification_path.write_text(
        canonical_json(classification) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(FormalExecutionError, match="classification differs.*Threads"):
        reconcile_formal_campaign(classification_campaign)

    unallowed_campaign = prepare_formal_campaign(
        plan,
        tmp_path / "unallowed",
        _provider(_binding(plan)),
        _prepare_input,
    )
    unallowed_claim = _claim_and_write(
        unallowed_campaign,
        block,
        block.slots[0],
    )
    unallowed_path = unallowed_claim.directory / "run_record.json"
    unallowed = json.loads(unallowed_path.read_text(encoding="utf-8"))
    unallowed["effective_nondefault_parameters"]["BranchDir"] = {
        "classification": "unallowed",
        "current": 1,
        "default": 0,
        "value_type": "int",
    }
    unallowed["gurobi_parameter_audit"]["enumerated_parameter_count"] += 1
    unallowed["unallowed_nondefault_parameters"] = ["BranchDir"]
    unallowed_path.write_text(canonical_json(unallowed) + "\n", encoding="utf-8")
    with pytest.raises(FormalExecutionError, match="contains unallowed controls"):
        reconcile_formal_campaign(unallowed_campaign)


def test_completed_bundle_rejects_effective_applied_cross_field_drift(
    tmp_path: Path,
) -> None:
    _manifest_value, plan = _plan()
    block = plan.blocks[0]
    campaign = prepare_formal_campaign(
        plan,
        tmp_path,
        _provider(_binding(plan)),
        _prepare_input,
    )
    claim = _claim_and_write(campaign, block, block.slots[0])
    record_path = claim.directory / "run_record.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    record["effective_nondefault_parameters"]["Threads"]["current"] = 2
    record_path.write_text(canonical_json(record) + "\n", encoding="utf-8")

    with pytest.raises(FormalExecutionError, match="effective and applied.*Threads"):
        reconcile_formal_campaign(campaign)


def test_failed_bundle_accepts_only_canonical_incomplete_gurobi_audit(
    tmp_path: Path,
) -> None:
    _manifest_value, plan = _plan()
    block = plan.blocks[0]

    accepted_campaign = prepare_formal_campaign(
        plan,
        tmp_path / "accepted",
        _provider(_binding(plan)),
        _prepare_input,
    )
    accepted_claim = _claim_and_write(
        accepted_campaign,
        block,
        block.slots[0],
        outcome="failed",
    )
    accepted_path = accepted_claim.directory / "run_record.json"
    accepted = json.loads(accepted_path.read_text(encoding="utf-8"))
    accepted["gurobi_parameter_audit"] = {
        "schema_version": 1,
        "enumeration_complete": False,
        "reason_code": "parameter_enumeration_failed",
        "enumerated_parameter_count": 0,
        "parameter_catalog_sha256": None,
    }
    accepted["effective_nondefault_parameters"] = {}
    accepted["unallowed_nondefault_parameters"] = []
    accepted["gurobi_execution_context"] = {
        "execution_mode": "unobserved",
        "observation_basis": "gurobi_effective_parameters",
        "remote_identity_sha256": None,
        "remote_mechanisms": [],
    }
    accepted_path.write_text(canonical_json(accepted) + "\n", encoding="utf-8")
    state = reconcile_formal_campaign(accepted_campaign)
    assert state.recovered_slot_ids == (block.slots[0].slot_id,)

    rejected_campaign = prepare_formal_campaign(
        plan,
        tmp_path / "rejected",
        _provider(_binding(plan)),
        _prepare_input,
    )
    rejected_claim = _claim_and_write(
        rejected_campaign,
        block,
        block.slots[0],
        outcome="failed",
    )
    rejected_path = rejected_claim.directory / "run_record.json"
    rejected = json.loads(rejected_path.read_text(encoding="utf-8"))
    rejected["gurobi_parameter_audit"] = {
        "schema_version": 1,
        "enumeration_complete": False,
        "reason_code": "parameter_enumeration_failed",
        "enumerated_parameter_count": 1,
        "parameter_catalog_sha256": None,
    }
    rejected["effective_nondefault_parameters"] = {}
    rejected["unallowed_nondefault_parameters"] = []
    rejected["gurobi_execution_context"] = {
        "execution_mode": "unobserved",
        "observation_basis": "gurobi_effective_parameters",
        "remote_identity_sha256": None,
        "remote_mechanisms": [],
    }
    rejected_path.write_text(canonical_json(rejected) + "\n", encoding="utf-8")
    with pytest.raises(FormalExecutionError, match="noncanonical incomplete"):
        reconcile_formal_campaign(rejected_campaign)


def test_implementation_digest_drift_fails_before_campaign_root_creation(
    tmp_path: Path,
) -> None:
    _manifest_value, plan = _plan()
    drifted = replace(
        _binding(plan),
        formal_execution_implementation_sha256="d" * 64,
    )
    results_root = tmp_path / "must-not-exist"

    with pytest.raises(
        FormalExecutionError,
        match="formal_execution_implementation_sha256",
    ):
        prepare_formal_campaign(
            plan,
            results_root,
            _provider(drifted),
            _prepare_input,
        )

    assert not results_root.exists()


def test_implementation_digest_is_rechecked_before_each_pending_slot_claim(
    tmp_path: Path,
) -> None:
    _manifest_value, plan = _plan()
    expected = _binding(plan)
    drifted = replace(
        expected,
        formal_execution_implementation_sha256="d" * 64,
    )
    provider_calls = 0

    def current_binding_provider() -> FormalExecutionBinding:
        nonlocal provider_calls
        provider_calls += 1
        return expected if provider_calls <= 2 else drifted

    campaign = prepare_formal_campaign(
        plan,
        tmp_path,
        current_binding_provider,
        _prepare_input,
    )
    block = plan.blocks[0]

    with pytest.raises(
        FormalExecutionError,
        match="formal_execution_implementation_sha256",
    ):
        execute_formal_block(
            campaign,
            block,
            _write_bundle,
            current_binding_provider,
        )

    first, second = block.slots[:2]
    assert provider_calls == 3
    assert (campaign.attempts_root / first.logical_run_id / first.attempt_id).is_dir()
    assert (campaign.journal_root / "slot-0000.json").is_file()
    assert not (campaign.attempts_root / second.logical_run_id).exists()


def test_checkpoint_hash_detects_later_attempt_bundle_mutation(tmp_path: Path) -> None:
    _manifest_value, plan = _plan()
    campaign = prepare_formal_campaign(
        plan,
        tmp_path,
        _provider(_binding(plan)),
        _prepare_input,
    )
    block = plan.blocks[0]
    slot = block.slots[0]
    claim = _claim_and_write(campaign, block, slot)
    reconcile_formal_campaign(campaign)

    with (claim.directory / "solver.log").open("a", encoding="utf-8") as handle:
        handle.write("post-checkpoint mutation\n")
    with pytest.raises(FormalExecutionError, match="checkpoint differs"):
        reconcile_formal_campaign(campaign)


@pytest.mark.parametrize(
    "artifact_name",
    ("run_spec.json", "run_record.json", "failure.json"),
)
def test_recovery_rejects_attempt_bundle_without_exact_formal_context(
    tmp_path: Path,
    artifact_name: str,
) -> None:
    _manifest_value, plan = _plan()
    campaign = prepare_formal_campaign(
        plan,
        tmp_path / artifact_name,
        _provider(_binding(plan)),
        _prepare_input,
    )
    block = plan.blocks[0]
    slot = block.slots[0]
    claim = _claim_and_write(campaign, block, slot, outcome="failed")
    artifact_path = claim.directory / artifact_name
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    del artifact["formal_context"]
    artifact_path.write_text(json.dumps(artifact), encoding="utf-8")

    with pytest.raises(FormalExecutionError, match="formal context differs"):
        reconcile_formal_campaign(campaign)

    assert not (campaign.journal_root / "slot-0000.json").exists()


def test_generic_attempt_claim_cannot_be_recovered_as_formal(
    tmp_path: Path,
) -> None:
    _manifest_value, plan = _plan()
    campaign = prepare_formal_campaign(
        plan,
        tmp_path,
        _provider(_binding(plan)),
        _prepare_input,
    )
    block = plan.blocks[0]
    slot = block.slots[0]
    claim = claim_attempt_directory(
        campaign.attempts_root,
        slot.logical_run_id,
        attempt_number=1,
    )
    _write_bundle(_task_for(campaign, block, slot), claim)

    with pytest.raises(FormalExecutionError, match="formal_claim.json"):
        reconcile_formal_campaign(campaign)

    assert not (campaign.journal_root / "slot-0000.json").exists()


def test_incomplete_attempt1_fails_closed(tmp_path: Path) -> None:
    _manifest_value, plan = _plan()
    campaign = prepare_formal_campaign(
        plan,
        tmp_path,
        _provider(_binding(plan)),
        _prepare_input,
    )
    slot = plan.blocks[0].slots[0]
    claim_attempt_directory(
        campaign.attempts_root,
        slot.logical_run_id,
        attempt_number=1,
    )

    with pytest.raises(FormalExecutionError, match="incomplete attempt1"):
        reconcile_formal_campaign(campaign)


def test_attempt2_unknown_logical_run_and_unknown_artifacts_fail_closed(
    tmp_path: Path,
) -> None:
    _manifest_value, plan = _plan()

    attempt2_campaign = prepare_formal_campaign(
        plan,
        tmp_path / "attempt2",
        _provider(_binding(plan)),
        _prepare_input,
    )
    slot = plan.blocks[0].slots[0]
    claim_attempt_directory(
        attempt2_campaign.attempts_root,
        slot.logical_run_id,
        attempt_number=2,
    )
    with pytest.raises(FormalExecutionError, match="non-attempt1"):
        reconcile_formal_campaign(attempt2_campaign)

    unknown_campaign = prepare_formal_campaign(
        plan,
        tmp_path / "unknown-run",
        _provider(_binding(plan)),
        _prepare_input,
    )
    (unknown_campaign.attempts_root / ("run-" + "f" * 64)).mkdir()
    with pytest.raises(FormalExecutionError, match="unknown logical run"):
        reconcile_formal_campaign(unknown_campaign)

    artifact_campaign = prepare_formal_campaign(
        plan,
        tmp_path / "unknown-artifact",
        _provider(_binding(plan)),
        _prepare_input,
    )
    block = plan.blocks[0]
    claim = _claim_and_write(artifact_campaign, block, block.slots[0])
    (claim.directory / "unregistered.txt").write_text("extra", encoding="utf-8")
    with pytest.raises(FormalExecutionError, match="unknown artifacts"):
        reconcile_formal_campaign(artifact_campaign)


def test_unknown_checkpoint_and_plan_root_entries_fail_closed(tmp_path: Path) -> None:
    _manifest_value, plan = _plan()
    campaign = prepare_formal_campaign(
        plan,
        tmp_path,
        _provider(_binding(plan)),
        _prepare_input,
    )
    (campaign.journal_root / "slot-copy.json").write_text("{}", encoding="utf-8")
    with pytest.raises(FormalExecutionError, match="unknown checkpoint"):
        reconcile_formal_campaign(campaign)

    second = prepare_formal_campaign(
        plan,
        tmp_path / "root-entry",
        _provider(_binding(plan)),
        _prepare_input,
    )
    (second.root / "unregistered-result.json").write_text("{}", encoding="utf-8")
    with pytest.raises(FormalExecutionError, match="unknown entries"):
        reconcile_formal_campaign(second)


def test_tampered_or_unknown_session_artifacts_fail_closed(tmp_path: Path) -> None:
    _manifest_value, plan = _plan()
    binding = _binding(plan)
    campaign = prepare_formal_campaign(
        plan,
        tmp_path / "tampered",
        _provider(binding),
        _prepare_input,
    )
    start = json.loads(campaign.session_start_path.read_text(encoding="utf-8"))
    start["session"]["session_snapshot"]["pid"] = 0
    campaign.session_start_path.write_text(json.dumps(start), encoding="utf-8")
    with pytest.raises(FormalExecutionError, match="formal session"):
        reconcile_formal_campaign(campaign)

    unknown = prepare_formal_campaign(
        plan,
        tmp_path / "unknown",
        _provider(binding),
        _prepare_input,
    )
    (unknown.sessions_root / "unknown-session.txt").write_text(
        "unexpected",
        encoding="utf-8",
    )
    with pytest.raises(FormalExecutionError, match="sessions root"):
        reconcile_formal_campaign(unknown)


def test_scheduler_enforces_scale_barriers_and_block_limits() -> None:
    _manifest_value, plan = _plan()
    lock = threading.Lock()
    active = Counter({scale: 0 for scale in SCALES})
    maxima = Counter({scale: 0 for scale in SCALES})
    violations: list[str] = []
    start_events: list[tuple[str, int]] = []
    first_small_pair = threading.Barrier(2)

    def run_block(block):
        with lock:
            if any(active[scale] for scale in SCALES if scale != block.scale):
                violations.append("cross-scale overlap")
            active[block.scale] += 1
            maxima[block.scale] = max(maxima[block.scale], active[block.scale])
            start_events.append((block.scale, block.block_index))
        if block.block_index in {0, 1}:
            first_small_pair.wait(timeout=2.0)
        time.sleep(0.001)
        with lock:
            active[block.scale] -= 1
        return block.block_index

    results = run_formal_block_schedule(plan, run_block)

    assert results == tuple(range(180))
    assert violations == []
    assert maxima == {"small": 2, "medium": 1, "large": 1}
    assert Counter(scale for scale, _index in start_events) == {
        "small": 60,
        "medium": 60,
        "large": 60,
    }
    scales_in_start_order = [scale for scale, _index in start_events]
    assert scales_in_start_order[:60] == ["small"] * 60
    assert scales_in_start_order[60:120] == ["medium"] * 60
    assert scales_in_start_order[120:] == ["large"] * 60


def test_injected_execution_exception_leaves_auditable_incomplete_attempt(
    tmp_path: Path,
) -> None:
    _manifest_value, plan = _plan()
    binding_provider = _provider(_binding(plan))
    campaign = prepare_formal_campaign(
        plan,
        tmp_path,
        binding_provider,
        _prepare_input,
    )

    def fail(_task, _claim, _prepared):
        raise RuntimeError("synthetic callback crash")

    with pytest.raises(FormalExecutionError, match="after attempt1 claim"):
        execute_formal_block(campaign, plan.blocks[0], fail, binding_provider)
    with pytest.raises(FormalExecutionError, match="incomplete attempt1"):
        reconcile_formal_campaign(campaign)


def test_execute_block_rejects_replacement_binding_provider_before_claim(
    tmp_path: Path,
) -> None:
    _manifest_value, plan = _plan()
    binding = _binding(plan)
    campaign_provider = _provider(binding)
    campaign = prepare_formal_campaign(
        plan,
        tmp_path,
        campaign_provider,
        _prepare_input,
    )

    with pytest.raises(FormalExecutionError, match="reuse the campaign"):
        execute_formal_block(
            campaign,
            plan.blocks[0],
            _write_bundle,
            _provider(binding),
        )

    assert tuple(campaign.attempts_root.iterdir()) == ()


def test_campaign_provider_is_factory_bound_and_cannot_be_replaced(
    tmp_path: Path,
) -> None:
    _manifest_value, plan = _plan()
    binding_provider = _provider(_binding(plan))
    campaign = prepare_formal_campaign(
        plan,
        tmp_path,
        binding_provider,
        _prepare_input,
    )

    with pytest.raises(TypeError, match="init=False"):
        replace(
            campaign,
            current_binding_provider=_provider(_binding(plan)),
        )
    with pytest.raises(TypeError, match="init=False"):
        replace(
            campaign,
            pre_claim_preparer=_prepare_input,
        )

    unsealed_copy = replace(campaign)
    with pytest.raises(FormalExecutionError, match="not issued"):
        execute_formal_block(
            unsealed_copy,
            plan.blocks[0],
            _write_bundle,
            binding_provider,
        )

    assert tuple(campaign.attempts_root.iterdir()) == ()


def test_pre_claim_preparer_rejects_instance_drift_before_claim(
    tmp_path: Path,
) -> None:
    _manifest_value, plan = _plan()
    binding_provider = _provider(_binding(plan))
    validated_tasks: list[FormalSlotTask] = []

    def reject_drifted_instance(task: FormalSlotTask) -> PreparedFormalInput:
        validated_tasks.append(task)
        raise ValueError("instance digest drift")

    campaign = prepare_formal_campaign(
        plan,
        tmp_path,
        binding_provider,
        reject_drifted_instance,
    )

    with pytest.raises(FormalExecutionError, match="pre-claim preparation failed"):
        execute_formal_block(
            campaign,
            plan.blocks[0],
            _write_bundle,
            binding_provider,
        )

    assert len(validated_tasks) == 1
    assert validated_tasks[0].slot == plan.blocks[0].slots[0]
    assert (
        validated_tasks[0].formal_execution_implementation_sha256
        == plan.formal_execution_implementation_sha256
    )
    assert tuple(campaign.attempts_root.iterdir()) == ()


@pytest.mark.parametrize(
    ("override", "message"),
    (
        ({"case_id": "wrong-case"}, "case_id differs"),
        ({"slot_id": "slot-wrong"}, "slot_id differs"),
        ({"expected_instance_sha256": "d" * 64}, "differs from the formal plan"),
        ({"payload_instance_sha256": "d" * 64}, "payload_instance_sha256"),
        ({"source_sha256_before": "d" * 64}, "source_sha256_before"),
        ({"source_sha256_after": "d" * 64}, "source_sha256_after"),
    ),
)
def test_pre_claim_preparer_strictly_rejects_attestation_drift_without_claim(
    tmp_path: Path,
    override: dict[str, str],
    message: str,
) -> None:
    _manifest_value, plan = _plan()
    binding_provider = _provider(_binding(plan))

    def prepare(task: FormalSlotTask) -> PreparedFormalInput:
        return _prepared_input(task, **override)

    campaign = prepare_formal_campaign(
        plan,
        tmp_path,
        binding_provider,
        prepare,
    )
    with pytest.raises(FormalExecutionError, match=message):
        execute_formal_block(
            campaign,
            plan.blocks[0],
            _write_bundle,
            binding_provider,
        )
    assert tuple(campaign.attempts_root.iterdir()) == ()


def test_execute_callback_receives_same_prepared_object_and_claim_binds_it(
    tmp_path: Path,
) -> None:
    _manifest_value, plan = _plan()
    binding_provider = _provider(_binding(plan))
    prepared_by_slot: dict[str, PreparedFormalInput] = {}

    def prepare(task: FormalSlotTask) -> PreparedFormalInput:
        prepared = _prepared_input(task, payload={"slot": task.slot.slot_id})
        prepared_by_slot[task.slot.slot_id] = prepared
        return prepared

    def execute(
        task: FormalSlotTask,
        claim,
        prepared: PreparedFormalInput,
    ) -> None:
        assert prepared is prepared_by_slot[task.slot.slot_id]
        assert prepared.payload == {"slot": task.slot.slot_id}
        _write_bundle(task, claim, prepared)

    campaign = prepare_formal_campaign(
        plan,
        tmp_path,
        binding_provider,
        prepare,
    )
    block = plan.blocks[0]
    result = execute_formal_block(campaign, block, execute, binding_provider)
    assert result.executed_slot_ids == tuple(slot.slot_id for slot in block.slots)

    first = block.slots[0]
    attempt = campaign.attempts_root / first.logical_run_id / first.attempt_id
    formal_claim = json.loads(
        (attempt / "formal_claim.json").read_text(encoding="utf-8")
    )
    expected_attestation = prepared_by_slot[first.slot_id].attestation
    assert formal_claim["prepared_input_attestation"] == (
        expected_attestation.to_payload()
    )
    assert formal_claim["prepared_input_attestation_sha256"] == (
        expected_attestation.sha256
    )
    checkpoint = json.loads(
        (campaign.journal_root / "slot-0000.json").read_text(encoding="utf-8")
    )
    assert checkpoint["prepared_input_attestation_sha256"] == (
        expected_attestation.sha256
    )
    assert checkpoint["prepared_instance_sha256"] == (
        expected_attestation.expected_instance_sha256
    )
    assert checkpoint["qualification_trace"] == (
        expected_attestation.qualification_trace.to_payload()
    )
    assert checkpoint["qualification_trace_sha256"] == (
        expected_attestation.qualification_trace.sha256
    )
    assert checkpoint["qualification_run_id"] == (
        expected_attestation.qualification_trace.qualification_run_id
    )
    assert checkpoint["solution_structure"] == (
        no_incumbent_solution_structure(
            expected_attestation.qualification_trace
        ).to_payload()
    )


def test_formal_claim_write_failure_rolls_back_new_attempt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _manifest_value, plan = _plan()
    binding_provider = _provider(_binding(plan))
    campaign = prepare_formal_campaign(
        plan,
        tmp_path,
        binding_provider,
        _prepare_input,
    )
    original_write = formal_execution_module.write_canonical_json_exclusive

    def fail_formal_claim(path, payload):
        if Path(path).name == "formal_claim.json":
            raise OSError("synthetic attestation write failure")
        return original_write(path, payload)

    monkeypatch.setattr(
        formal_execution_module,
        "write_canonical_json_exclusive",
        fail_formal_claim,
    )
    with pytest.raises(FormalExecutionError, match="cannot attest formal attempt1"):
        execute_formal_block(
            campaign,
            plan.blocks[0],
            _write_bundle,
            binding_provider,
        )
    assert tuple(campaign.attempts_root.iterdir()) == ()


def test_concurrent_claims_serialize_preparation_under_claim_lock(
    tmp_path: Path,
) -> None:
    _manifest_value, plan = _plan()
    binding_provider = _provider(_binding(plan))
    counter_lock = threading.Lock()
    active_preparers = 0
    maximum_active_preparers = 0

    def prepare(task: FormalSlotTask) -> PreparedFormalInput:
        nonlocal active_preparers, maximum_active_preparers
        with counter_lock:
            active_preparers += 1
            maximum_active_preparers = max(
                maximum_active_preparers,
                active_preparers,
            )
        time.sleep(0.01)
        with counter_lock:
            active_preparers -= 1
        return _prepared_input(task)

    campaign = prepare_formal_campaign(
        plan,
        tmp_path,
        binding_provider,
        prepare,
    )
    slots = (
        (plan.blocks[0], plan.blocks[0].slots[0]),
        (plan.blocks[1], plan.blocks[1].slots[0]),
    )
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = tuple(
            executor.submit(_claim_pending_slot, campaign, block, slot)
            for block, slot in slots
        )
        claimed = tuple(future.result() for future in futures)

    assert maximum_active_preparers == 1
    assert len({id(prepared) for _claim, _task, prepared in claimed}) == 2


def test_completed_resume_does_not_prepare_again(tmp_path: Path) -> None:
    _manifest_value, plan = _plan()
    binding_provider = _provider(_binding(plan))
    prepared_slot_ids: list[str] = []

    def prepare(task: FormalSlotTask) -> PreparedFormalInput:
        prepared_slot_ids.append(task.slot.slot_id)
        return _prepared_input(task)

    campaign = prepare_formal_campaign(
        plan,
        tmp_path,
        binding_provider,
        prepare,
    )
    block = plan.blocks[0]
    execute_formal_block(campaign, block, _write_bundle, binding_provider)
    assert prepared_slot_ids == [slot.slot_id for slot in block.slots]

    prepared_slot_ids.clear()
    resumed = execute_formal_block(campaign, block, _write_bundle, binding_provider)
    assert resumed.executed_slot_ids == ()
    assert prepared_slot_ids == []


def test_recovery_rejects_self_rehashed_but_inconsistent_input_attestation(
    tmp_path: Path,
) -> None:
    _manifest_value, plan = _plan()
    campaign = prepare_formal_campaign(
        plan,
        tmp_path,
        _provider(_binding(plan)),
        _prepare_input,
    )
    block = plan.blocks[0]
    slot = block.slots[0]
    claim = _claim_and_write(campaign, block, slot)
    path = claim.directory / "formal_claim.json"
    formal_claim = json.loads(path.read_text(encoding="utf-8"))
    formal_claim["prepared_input_attestation"]["source_sha256_after"] = "d" * 64
    formal_claim["prepared_input_attestation_sha256"] = canonical_sha256(
        formal_claim["prepared_input_attestation"]
    )
    path.write_text(json.dumps(formal_claim), encoding="utf-8")

    with pytest.raises(FormalExecutionError, match="source_sha256_after"):
        reconcile_formal_campaign(campaign)
    assert not (campaign.journal_root / "slot-0000.json").exists()


def test_recovery_rejects_missing_formal_solution_structure(
    tmp_path: Path,
) -> None:
    _manifest_value, plan = _plan()
    campaign = prepare_formal_campaign(
        plan,
        tmp_path,
        _provider(_binding(plan)),
        _prepare_input,
    )
    block = plan.blocks[0]
    slot = block.slots[0]
    claim = _claim_and_write(campaign, block, slot)
    path = claim.directory / "run_record.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    record.pop("solution_structure")
    path.write_text(canonical_json(record) + "\n", encoding="utf-8")

    with pytest.raises(FormalExecutionError, match="solution structure"):
        reconcile_formal_campaign(campaign)


def test_recovery_rejects_self_rehashed_solution_structure_tampering(
    tmp_path: Path,
) -> None:
    _manifest_value, plan = _plan()
    campaign = prepare_formal_campaign(
        plan,
        tmp_path,
        _provider(_binding(plan)),
        _prepare_input,
    )
    block = plan.blocks[0]
    slot = block.slots[0]
    claim = _claim_and_write(
        campaign,
        block,
        slot,
        outcome="completed_with_incumbent",
    )
    path = claim.directory / "run_record.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    record["solution_structure"]["open_H_count"] = 9
    record["solution_structure_sha256"] = canonical_sha256(
        record["solution_structure"]
    )
    path.write_text(canonical_json(record) + "\n", encoding="utf-8")

    with pytest.raises(FormalExecutionError, match="solution structure"):
        reconcile_formal_campaign(campaign)


def test_recovery_rejects_self_rehashed_qualification_run_tampering(
    tmp_path: Path,
) -> None:
    _manifest_value, plan = _plan()
    campaign = prepare_formal_campaign(
        plan,
        tmp_path,
        _provider(_binding(plan)),
        _prepare_input,
    )
    block = plan.blocks[0]
    slot = block.slots[0]
    claim = _claim_and_write(campaign, block, slot)
    path = claim.directory / "formal_claim.json"
    formal_claim = json.loads(path.read_text(encoding="utf-8"))
    attestation = formal_claim["prepared_input_attestation"]
    trace = attestation["qualification_trace"]
    trace["qualification_run_id"] = "qualification-tampered"
    trace["qualification_decision"]["qualification_run_id"] = (
        "qualification-tampered"
    )
    trace["qualification_decision_sha256"] = canonical_sha256(
        trace["qualification_decision"]
    )
    attestation["qualification_trace"] = trace
    formal_claim["prepared_input_attestation_sha256"] = canonical_sha256(
        attestation
    )
    path.write_text(canonical_json(formal_claim) + "\n", encoding="utf-8")

    with pytest.raises(FormalExecutionError, match="qualification trace differs"):
        reconcile_formal_campaign(campaign)


def test_tampered_task_implementation_digest_cannot_publish_checkpoint(
    tmp_path: Path,
) -> None:
    _manifest_value, plan = _plan()
    binding_provider = _provider(_binding(plan))
    campaign = prepare_formal_campaign(
        plan,
        tmp_path,
        binding_provider,
        _prepare_input,
    )

    def write_with_tampered_task(
        task: FormalSlotTask,
        claim,
        prepared: PreparedFormalInput,
    ) -> None:
        assert (
            task.formal_execution_implementation_sha256
            == plan.formal_execution_implementation_sha256
        )
        _write_bundle(
            replace(
                task,
                formal_execution_implementation_sha256="d" * 64,
            ),
            claim,
            prepared,
        )

    with pytest.raises(FormalExecutionError, match="formal context differs"):
        execute_formal_block(
            campaign,
            plan.blocks[0],
            write_with_tampered_task,
            binding_provider,
        )

    assert not (campaign.journal_root / "slot-0000.json").exists()


def test_checkpoint_payload_is_canonical_and_strictly_bound(tmp_path: Path) -> None:
    _manifest_value, plan = _plan()
    campaign = prepare_formal_campaign(
        plan,
        tmp_path,
        _provider(_binding(plan)),
        _prepare_input,
    )
    block = plan.blocks[0]
    slot = block.slots[0]
    _claim_and_write(campaign, block, slot)
    reconcile_formal_campaign(campaign)
    checkpoint = campaign.journal_root / "slot-0000.json"
    payload = json.loads(checkpoint.read_text(encoding="utf-8"))
    assert checkpoint.read_text(encoding="utf-8") == canonical_json(payload) + "\n"

    assert (
        payload["formal_execution_implementation_sha256"]
        == plan.formal_execution_implementation_sha256
    )
    payload["formal_execution_implementation_sha256"] = "d" * 64
    checkpoint.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(FormalExecutionError, match="checkpoint differs"):
        reconcile_formal_campaign(campaign)


def test_attempt_files_are_read_once_and_authenticated_snapshot_is_immutable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _manifest_value, plan = _plan()
    campaign = prepare_formal_campaign(
        plan,
        tmp_path,
        _provider(_binding(plan)),
        _prepare_input,
    )
    block = plan.blocks[0]
    slot = block.slots[0]
    claim = _claim_and_write(campaign, block, slot)
    read_counts: Counter[str] = Counter()
    original_reader = formal_execution_module._read_regular_file_once

    def counted_reader(path: Path, *, label: str):
        if label == "attempt artifact":
            read_counts[path.name] += 1
        return original_reader(path, label=label)

    monkeypatch.setattr(
        formal_execution_module,
        "_read_regular_file_once",
        counted_reader,
    )
    state = reconcile_formal_campaign(campaign)

    assert read_counts == Counter(
        {
            "claim.json": 1,
            "cut_stats.json": 1,
            "formal_claim.json": 1,
            "progress.jsonl": 1,
            "root_rounds.jsonl": 1,
            "run_record.json": 1,
            "run_spec.json": 1,
            "solver.log": 1,
        }
    )
    snapshot = state.authenticated_slots[0]
    assert snapshot.slot_id == slot.slot_id
    assert snapshot.run_record["solver_metrics"]["status_name"] == "TIME_LIMIT"
    assert snapshot.run_record["solver_metrics"]["relative_gap"] is None
    assert snapshot.run_record["solver_metrics"]["has_incumbent"] is False
    with pytest.raises(TypeError):
        snapshot.run_record["solver_metrics"]["has_incumbent"] = True  # type: ignore[index]

    run_record_path = claim.directory / "run_record.json"
    changed_record = json.loads(run_record_path.read_text(encoding="utf-8"))
    changed_record["solver_metrics"]["relative_gap"] = 0.0
    run_record_path.write_text(canonical_json(changed_record) + "\n", encoding="utf-8")

    assert snapshot.run_record["solver_metrics"]["relative_gap"] is None
    assert snapshot.run_record["solver_metrics"]["has_incumbent"] is False
    with pytest.raises(FormalExecutionError, match="formal paper metrics are invalid"):
        reconcile_formal_campaign(campaign)


def test_evidence_seal_rejects_rehashed_metrics_and_synchronized_checkpoint(
    tmp_path: Path,
) -> None:
    _manifest_value, plan = _plan()
    campaign = prepare_formal_campaign(
        plan,
        tmp_path,
        _provider(_binding(plan)),
        _prepare_input,
    )
    snapshots = _synthetic_authenticated_slots(campaign)
    original_seal = seal_formal_campaign_evidence(campaign, snapshots)
    original_bytes = original_seal.path.read_bytes()

    first_block = plan.blocks[0]
    first_slot = first_block.slots[0]
    first = snapshots[0]
    changed_run_record = MappingProxyType(
        {
            "slot_id": first.slot_id,
            "solver_metrics": MappingProxyType(
                {"has_incumbent": False, "relative_gap": 0.0}
            ),
        }
    )
    changed_file_hashes = dict(first.file_sha256)
    changed_file_hashes["run_record.json"] = hashlib.sha256(
        (canonical_json(changed_run_record) + "\n").encode("utf-8")
    ).hexdigest()
    changed_file_sha256 = tuple(sorted(changed_file_hashes.items()))
    changed_bundle_sha256 = canonical_sha256(
        {
            "domain": FORMAL_SLOT_BUNDLE_HASH_DOMAIN,
            "slot_id": first_slot.slot_id,
            "slot_index": first_slot.slot_index,
            "attempt_id": first_slot.attempt_id,
            "file_sha256": dict(changed_file_sha256),
        }
    )
    synchronized_checkpoint_bytes = (
        canonical_json(
            {
                "bundle_file_sha256": dict(changed_file_sha256),
                "slot_id": first_slot.slot_id,
            }
        )
        + "\n"
    ).encode("utf-8")
    changed_first = replace(
        first,
        run_record=changed_run_record,
        file_sha256=changed_file_sha256,
        attempt_bundle_sha256=changed_bundle_sha256,
        checkpoint_bytes_sha256=hashlib.sha256(
            synchronized_checkpoint_bytes
        ).hexdigest(),
    )
    changed_snapshots = (changed_first, *snapshots[1:])

    with pytest.raises(FormalExecutionError, match="externally expected root"):
        seal_formal_campaign_evidence(
            campaign,
            changed_snapshots,
            expected_evidence_root_sha256=original_seal.evidence_root_sha256,
        )
    assert original_seal.path.read_bytes() == original_bytes
    with pytest.raises(FormalExecutionError, match="seal differs"):
        seal_formal_campaign_evidence(campaign, changed_snapshots)

    resumed = prepare_formal_campaign(
        plan,
        tmp_path,
        _provider(_binding(plan)),
        _prepare_input,
    )
    resumed_seal = seal_formal_campaign_evidence(
        resumed,
        snapshots,
        expected_evidence_root_sha256=original_seal.evidence_root_sha256,
    )
    assert resumed_seal.evidence_id == original_seal.evidence_id


def test_complete_reconciliation_auto_seals_exact_plan_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _manifest_value, plan = _plan()
    campaign = prepare_formal_campaign(
        plan,
        tmp_path,
        _provider(_binding(plan)),
        _prepare_input,
    )
    snapshots = _synthetic_authenticated_slots(campaign)
    by_slot_id = {snapshot.slot_id: snapshot for snapshot in snapshots}
    expected_logical_runs = {
        slot.logical_run_id for block in plan.blocks for slot in block.slots
    }
    checkpoint_paths = {
        slot.slot_index: campaign.journal_root / f"slot-{slot.slot_index:04d}.json"
        for block in plan.blocks
        for slot in block.slots
    }

    class SyntheticBundle:
        def __init__(self, snapshot: AuthenticatedFormalSlotSnapshot) -> None:
            self.snapshot = snapshot

        def authenticated_snapshot(
            self,
            _block,
            _slot,
            *,
            checkpoint_bytes_sha256: str,
        ) -> AuthenticatedFormalSlotSnapshot:
            assert checkpoint_bytes_sha256 == self.snapshot.checkpoint_bytes_sha256
            return self.snapshot

    monkeypatch.setattr(
        formal_execution_module,
        "_validate_attempt_tree",
        lambda _campaign: expected_logical_runs,
    )
    monkeypatch.setattr(
        formal_execution_module,
        "_validate_journal_tree",
        lambda _campaign: checkpoint_paths,
    )
    monkeypatch.setattr(
        formal_execution_module,
        "_validate_attempt_bundle",
        lambda _campaign, _block, slot: SyntheticBundle(by_slot_id[slot.slot_id]),
    )
    monkeypatch.setattr(
        formal_execution_module,
        "_validate_checkpoint",
        lambda _campaign, _block, slot, _bundle, path: (
            formal_execution_module._ValidatedCheckpoint(
                path=path,
                bytes_sha256=by_slot_id[slot.slot_id].checkpoint_bytes_sha256,
            )
        ),
    )

    state = reconcile_formal_campaign(campaign)

    assert state.complete
    assert len(state.authenticated_slots) == 900
    assert tuple(snapshot.slot_id for snapshot in state.authenticated_slots) == tuple(
        slot.slot_id for block in plan.blocks for slot in block.slots
    )
    assert state.evidence_id == f"formal-evidence-{state.evidence_root_sha256}"
    assert campaign.evidence_seal_path.is_file()
    evidence_payload = json.loads(
        campaign.evidence_seal_path.read_text(encoding="utf-8")
    )
    assert evidence_payload["authenticated_slot_count"] == 900
    assert "sessions" not in evidence_payload["evidence_payload"]

    repeated = reconcile_formal_campaign(
        campaign,
        expected_evidence_root_sha256=state.evidence_root_sha256,
    )
    assert repeated.evidence_id == state.evidence_id
