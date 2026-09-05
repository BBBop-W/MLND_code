from __future__ import annotations

import hashlib
import json
import math
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import Any, Mapping

import pytest

import src.benchmark.branch_cut.cli as cli_module
import src.benchmark.branch_cut.formal_execution as formal_execution_module
from src.benchmark.branch_cut.analysis import PairWinner
from src.benchmark.branch_cut.artifacts import write_canonical_json_exclusive
from src.benchmark.branch_cut.cli import (
    CLIError,
    CLIHooks,
    _default_instance_loader,
    _prepare_formal_input_snapshot,
    _qualification_outcome_from_payload,
    _qualification_task_result_from_payload,
    _solver_suite,
    build_parser,
    dispatch,
    load_benchmark_manifest,
)
from src.benchmark.branch_cut.case_builder import (
    CANONICAL_INSTANCE_FILES,
    QUALIFICATION_MODEL_IDENTITY_SCHEMA,
    LOCKED_QUALIFICATION_PROFILES,
    QualificationSolveProfile,
    canonical_instance_hash,
    locked_replacement_order,
    qualification_case_sha256,
)
from src.benchmark.branch_cut.execution import ProfileExecutionResult
from src.benchmark.branch_cut.formal_environment import (
    FormalEnvironmentCapture,
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
    FormalCampaignState,
    FormalCampaignRunResult,
    FormalExecutionBinding,
    FormalExecutionError,
    FormalInputAttestation,
    FormalSlotTask,
    PreparedFormalInput,
    prepare_formal_campaign,
    seal_formal_campaign_evidence,
)
from src.benchmark.branch_cut.formal_plan import formal_plan_from_artifact
from src.benchmark.branch_cut.formal_solution_evidence import (
    FormalQualificationTrace,
    bind_qualification_trace,
)
from src.benchmark.branch_cut.hashing import (
    build_model_id,
    build_pair_id,
    canonical_json,
    canonical_sha256,
)
from src.benchmark.branch_cut.manifest import (
    OVERCOMPLETE_SELECTION_RULE,
    FreezeMetadata,
    freeze_manifest,
)
from src.benchmark.branch_cut.pool import CandidateCase, CandidatePool
from src.benchmark.branch_cut.qualification import (
    QualificationEvidence,
    QualificationPolicy,
    assess_qualification,
)
from src.benchmark.branch_cut.qualification_geometry_start import (
    GEOMETRY_START_PLAN_VERSION,
    GeometryStartAttemptAudit,
    GeometryStartInstallAudit,
    GeometryStartPlanAudit,
    GeometryStartPolicy,
)
from src.benchmark.branch_cut.qualification_runner import (
    QualificationModelAudit,
    QualificationModelContract,
    QualificationRunnerOutcome,
    QualificationRunnerReason,
)
from src.benchmark.branch_cut.resource_lock import BenchmarkResourceLock
from src.benchmark.branch_cut.schema import (
    BenchmarkMode,
    ModelIdentity,
    PairSpec,
    SolverConfig,
    StartMode,
)


_NOW = "2026-08-27T00:00:00Z"
_MODEL_IDENTITY = ModelIdentity(
    formulation_name="mlnd-test-formulation",
    formulation_schema_version=1,
    implementation_sha256="c" * 64,
    model_config_sha256="d" * 64,
)
_QUALIFICATION_IMPLEMENTATION_SHA256 = "e" * 64
_FORMAL_EXECUTION_IMPLEMENTATION_SHA256 = "f" * 64
_GUROBI_VERSION = "12.0.3-test"


_FORMAL_ENVIRONMENT = FormalEnvironmentContract(
    operating_system=OperatingSystemIdentity(
        name="TestOS",
        formal_version="1.0",
        kernel_release="test-kernel",
        kernel_version="test-build",
    ),
    machine="test-machine",
    pointer_bits=64,
    byteorder="little",
    cpu_model="test cpu",
    physical_cores=4,
    logical_cores=8,
    installed_ram_bytes=16 * 1024**3,
    python=PythonIdentity("CPython", "3.13.0"),
    numpy=ModuleDistributionVersions("2.0.0", "2.0.0"),
    scipy=ModuleDistributionVersions("1.14.0", "1.14.0"),
    gurobipy=GurobiVersions("12.0.3", "12.0.3", _GUROBI_VERSION),
)
_FORMAL_ENVIRONMENT_CAPTURE = FormalEnvironmentCapture(
    contract=_FORMAL_ENVIRONMENT,
    host_identity_sha256="a" * 64,
    session=SessionSnapshot(
        pid=123,
        executable="/test/python",
        cwd="/test/worktree",
        affinity_cpu_ids=(0, 1, 2, 3),
        blas_environment=(("OMP_NUM_THREADS", "1"),),
        available_memory_bytes=8 * 1024**3,
        load_average=(0.1, 0.2, 0.3),
    ),
)


def _test_formal_environment_capture() -> FormalEnvironmentCapture:
    return _FORMAL_ENVIRONMENT_CAPTURE


def _candidate(
    case_id: str = "small-b00-s11",
    *,
    scale: str = "small",
    demand_block_id: str = "b00",
    demand_scale: float = 3.0,
    feeder_rider_cost_factor: float = 0.35,
    solve_profile: QualificationSolveProfile | None = None,
) -> CandidateCase:
    profile = solve_profile or next(
        item
        for item in LOCKED_QUALIFICATION_PROFILES
        if item.demand_scale == demand_scale
        and item.feeder_rider_cost_factor == feeder_rider_cost_factor
    )
    instance_sha256 = hashlib.sha256(case_id.encode()).hexdigest()
    return CandidateCase(
        case_id=case_id,
        scale=scale,
        demand_block_id=demand_block_id,
        layout_seed=11,
        instance_ref=case_id,
        instance_sha256=instance_sha256,
        generator_parameters={
            "map_half_size_km": 4.0,
            "qualification_solve_profile": profile.to_payload(),
            "qualification_model_identity": {
                "schema": QUALIFICATION_MODEL_IDENTITY_SCHEMA,
                "effective_model_config_sha256": (
                    profile.effective_model_config_sha256()
                ),
                "qualification_case_sha256": qualification_case_sha256(
                    instance_sha256,
                    profile,
                ),
                "expected_screening_is_exact": (profile.expected_screening_is_exact),
            },
        },
    )


def _replacement_candidate(
    *,
    block_id: str,
    scale: str,
    replacement_rank: int,
    expected_strata: int,
    replacement_count: int,
) -> CandidateCase:
    seeds = (11, 29, 47, 83)
    if not 0 <= replacement_rank < replacement_count <= len(seeds):
        raise ValueError("test replacement rank/count is invalid")
    profile_id, layout_seed = locked_replacement_order()[replacement_rank]
    profile = next(
        item for item in LOCKED_QUALIFICATION_PROFILES if item.profile_id == profile_id
    )
    assert layout_seed == seeds[replacement_rank]
    case_id = (
        f"{block_id}__{scale}__rank{replacement_rank}__profile-{profile.profile_id}"
    )
    base = _candidate(
        case_id,
        scale=scale,
        demand_block_id=block_id,
        solve_profile=profile,
    )
    candidate_h = {"small": 23, "medium": 51, "large": 90}[scale]
    return CandidateCase(
        case_id=case_id,
        scale=scale,
        demand_block_id=block_id,
        layout_seed=layout_seed,
        instance_ref=case_id,
        instance_sha256=hashlib.sha256(case_id.encode()).hexdigest(),
        generator_parameters={
            **base.generator_parameters,
            "candidate_counts": {
                "H": candidate_h,
                "L": {"small": 2, "medium": 3, "large": 6}[scale],
                "E": {"small": 3, "medium": 5, "large": 9}[scale],
            },
            "pre_registered_selection": {
                "selection_rule": OVERCOMPLETE_SELECTION_RULE,
                "stratum_id": f"{block_id}__{scale}",
                "replacement_rank_within_stratum": replacement_rank,
                "target_cases_per_stratum": 2,
                "expected_stratum_count": expected_strata,
                "selection_uses_algorithm_performance": False,
                "large_scale_open_h_qualification_target": (
                    [50, 60] if scale == "large" else None
                ),
                "forced_opening_constraints_added": False,
            },
        },
    )


def _runner_outcome(
    candidate: CandidateCase,
    *,
    accepted: bool,
    qualification_run_id: str,
    policy: QualificationPolicy,
    geometry_start_policy: GeometryStartPolicy | None = None,
) -> QualificationRunnerOutcome:
    contract = cli_module._candidate_model_contract(candidate)
    model_audit = QualificationModelAudit(
        effective_model_config_sha256=contract.effective_model_config_sha256,
        demand_scale=contract.demand_scale,
        locker_fixed_cost_per_hour=contract.locker_fixed_cost_per_hour,
        display_origin_choices=contract.display_origin_choices,
        general_corridor_choices=contract.general_corridor_choices,
        screening_is_exact=contract.screening_is_exact,
        model_fingerprint=17,
    )
    candidate_counts = candidate.generator_parameters.get("candidate_counts")
    candidate_h = (
        candidate_counts["H"]
        if isinstance(candidate_counts, dict)
        else {"small": 23, "medium": 51, "large": 90}[candidate.scale]
    )
    accepted_open_h = {"small": 8, "medium": 20, "large": 55}[candidate.scale]
    evidence = QualificationEvidence(
        case_id=candidate.case_id,
        scale=candidate.scale,
        courier_only_objective=1_000.0,
        witness_objective=900.0 if accepted else 1_000.0,
        total_demand_per_hour=100.0,
        witness_service_per_hour=20.0 if accepted else 0.0,
        open_facilities=accepted_open_h if accepted else 0,
        positive_cost_open_facilities=accepted_open_h if accepted else 0,
        candidate_h=candidate_h,
        open_h=accepted_open_h if accepted else 0,
        witness_feasible=True,
        qualification_run_id=qualification_run_id,
        instance_sha256=candidate.instance_sha256,
        active_h=accepted_open_h if accepted else 0,
        active_h_threshold_per_hour=0.001,
        qualification_case_sha256=contract.qualification_case_sha256,
        effective_model_config_sha256=contract.effective_model_config_sha256,
        screening_is_exact=contract.screening_is_exact,
    )
    decision = assess_qualification(evidence, policy)
    assert decision.accepted is accepted
    return QualificationRunnerOutcome(
        accepted=decision.accepted,
        rejection_reasons=decision.rejection_reasons,
        evidence=evidence,
        decision=decision,
        required_service_per_hour=0.1,
        required_objective_upper_bound=999.0,
        temporary_constraint_used=False,
        temporary_constraint_removed=None,
        fresh_model_validation_performed=True,
        witness_status="TIME_LIMIT",
        validation_status="TIME_LIMIT",
        captured_variable_count=100,
        exception_messages=(),
        geometry_start_policy=geometry_start_policy or GeometryStartPolicy(),
        expected_model_contract=contract,
        witness_model_audit=model_audit,
        fresh_model_audit=model_audit,
    )


def _geometry_plan_audit(target: int, marker: str) -> GeometryStartPlanAudit:
    return GeometryStartPlanAudit(
        plan_version=GEOMETRY_START_PLAN_VERSION,
        target_support_size=target,
        plan_sha256=marker * 64,
        support_order=tuple(f"H{index}" for index in range(target)),
        cycle_descriptors=tuple(
            ("B0", f"H{index}", f"H{index + 1}", "E0", "E0", "E0")
            for index in range(target - 1)
        ),
        availability_index=1,
        availability_probability=0.8,
        offered_flow_per_cycle=1.0,
        flow_per_cycle=0.8,
        planned_service_per_hour=8.8,
        family_start_counts=(
            ("psi", 3),
            ("varphi", 3),
            ("x", 4),
            ("y", 9),
            ("z", 2),
        ),
        prechecks_passed=True,
        precheck_metrics=(("queue_slack", 1.0),),
    )


def _safe_geometry_install() -> GeometryStartInstallAudit:
    return GeometryStartInstallAudit(
        reset_performed=True,
        undefined_variable_count=100,
        family_start_counts=(
            ("psi", 3),
            ("varphi", 3),
            ("x", 4),
            ("y", 9),
            ("z", 2),
        ),
        fingerprint_before=17,
        fingerprint_with_start=19,
        fingerprint_after=17,
        fingerprint_unchanged=True,
        constraint_counts_before=(100, 10, 0),
        constraint_counts_after=(100, 10, 0),
        hard_constraints_added=False,
    )


def _successful_geometry_attempt(
    target: int,
    marker: str,
) -> GeometryStartAttemptAudit:
    return GeometryStartAttemptAudit(
        plan=_geometry_plan_audit(target, marker),
        per_attempt_seconds=10.0,
        start_node_limit=500,
        install=_safe_geometry_install(),
        solve_status="TIME_LIMIT",
        solve_seconds=9.0,
        full_witness_found=True,
    )


def _accepted_decision(candidate: CandidateCase):
    contract = cli_module._candidate_model_contract(candidate)
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
            qualification_case_sha256=contract.qualification_case_sha256,
            effective_model_config_sha256=contract.effective_model_config_sha256,
            screening_is_exact=contract.screening_is_exact,
        )
    )


def _pool(candidates: tuple[CandidateCase, ...]) -> CandidatePool:
    return CandidatePool(
        candidates=candidates,
        candidate_seed_order=(11,),
        pool_label="test-locked-pool",
    )


def _write_pool(path: Path, pool: CandidatePool) -> Path:
    return write_canonical_json_exclusive(
        path,
        {
            "artifact_type": "mlnd_branch_cut_candidate_pool",
            "pool_sha256": pool.sha256,
            "pool": pool.to_payload(),
        },
    )


def _qualification_hooks(
    qualify_instance: Any,
    *,
    load_instance: Any | None = None,
) -> CLIHooks:
    return CLIHooks(
        canonical_hash=lambda path: (
            hashlib.sha256(path.name.encode()).hexdigest(),
            {},
        ),
        load_instance=(
            load_instance
            if load_instance is not None
            else (lambda _path, candidate: candidate)
        ),
        qualify_instance=qualify_instance,
        model_identity=lambda: _MODEL_IDENTITY,
        qualification_implementation_sha256=(
            lambda: _QUALIFICATION_IMPLEMENTATION_SHA256
        ),
        gurobi_version=lambda: _GUROBI_VERSION,
        now_utc=lambda: _NOW,
    )


def _qualification_cli_args(
    *,
    pool_path: Path,
    instances_root: Path,
    output: Path,
    extra: tuple[str, ...] = (),
) -> Any:
    return build_parser().parse_args(
        [
            "qualify",
            "--pool",
            str(pool_path),
            "--instances-root",
            str(instances_root),
            "--output",
            str(output),
            *extra,
        ]
    )


def _formal_manifest(tmp_path: Path) -> tuple[Path, Any]:
    candidates: list[CandidateCase] = []
    decisions = []
    for scale in ("small", "medium", "large"):
        for index in range(12):
            case = _candidate(
                f"{scale}-b{index:02d}-s11",
                scale=scale,
                demand_block_id=f"b{index:02d}",
            )
            candidates.append(case)
            decisions.append(_accepted_decision(case))
    pool = _pool(tuple(candidates))
    manifest = freeze_manifest(
        pool,
        decisions,
        freeze_metadata=FreezeMetadata(
            qualification_completed_at_utc="2026-08-26T23:58:00Z",
            manifest_frozen_at_utc="2026-08-26T23:59:00Z",
        ),
    )
    identity_payload = {
        "manifest": manifest.artifact_payload(),
        "qualification_mode": "MX",
        "model_identity": _MODEL_IDENTITY,
        "model_id": build_model_id(_MODEL_IDENTITY),
    }
    path = write_canonical_json_exclusive(
        tmp_path / "manifest.json",
        {
            "artifact_type": "mlnd_branch_cut_benchmark_manifest",
            "artifact_schema_version": 1,
            **identity_payload,
            "benchmark_manifest_artifact_sha256": canonical_sha256(identity_payload),
        },
    )
    return path, manifest


def test_parser_exposes_complete_workflow_including_formal_commands() -> None:
    parser = build_parser()
    commands = parser._subparsers._group_actions[0].choices  # type: ignore[attr-defined]
    assert set(commands) == {
        "analyze",
        "analyze-formal",
        "build-pool",
        "freeze",
        "freeze-formal-plan",
        "qualify",
        "run",
        "run-formal",
    }


def _freeze_formal_plan_for_cli_test(
    tmp_path: Path,
    *,
    analysis_args: tuple[str, ...] = (),
) -> tuple[Path, Any, Path, Mapping[str, Any]]:
    manifest_path, manifest = _formal_manifest(tmp_path)
    plan_path = tmp_path / "formal-plan.json"
    args = build_parser().parse_args(
        [
            "freeze-formal-plan",
            "--manifest",
            str(manifest_path),
            "--output",
            str(plan_path),
            "--analysis-bootstrap-resamples",
            "10",
            *analysis_args,
        ]
    )
    result = dispatch(
        args,
        CLIHooks(
            model_identity=lambda: _MODEL_IDENTITY,
            formal_execution_implementation_sha256=(
                lambda: _FORMAL_EXECUTION_IMPLEMENTATION_SHA256
            ),
            gurobi_version=lambda: _GUROBI_VERSION,
            formal_environment_capture=_test_formal_environment_capture,
        ),
    )
    return manifest_path, manifest, plan_path, result


def _formal_run_cli_args(
    *,
    manifest_path: Path,
    plan_path: Path,
    expected_plan_id: str,
    tmp_path: Path,
) -> Any:
    return build_parser().parse_args(
        [
            "run-formal",
            "--manifest",
            str(manifest_path),
            "--plan",
            str(plan_path),
            "--expected-plan-id",
            expected_plan_id,
            "--instances-root",
            str(tmp_path / "instances"),
            "--results-root",
            str(tmp_path / "formal-results"),
            "--resource-lock",
            str(tmp_path / "formal.lock"),
        ]
    )


def _formal_snapshot_fixture(
    tmp_path: Path,
) -> tuple[Path, CandidateCase, Any]:
    instance_dir = tmp_path / "instances" / "snapshot-case"
    instance_dir.mkdir(parents=True)
    for index, filename in enumerate(CANONICAL_INSTANCE_FILES):
        (instance_dir / filename).write_bytes(f"{filename}:payload-{index}\n".encode())
    instance_sha256, _ = canonical_instance_hash(instance_dir)
    candidate = replace(
        _candidate("snapshot-case"),
        instance_ref="snapshot-case",
        instance_sha256=instance_sha256,
    )
    task = SimpleNamespace(
        block=SimpleNamespace(
            case_id=candidate.case_id,
            instance_ref=candidate.instance_ref,
            instance_sha256=candidate.instance_sha256,
        ),
        slot=SimpleNamespace(slot_id="slot-" + "1" * 64),
    )
    return instance_dir, candidate, task


def _formal_snapshot_trace(candidate: CandidateCase) -> FormalQualificationTrace:
    decision = assess_qualification(
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
            qualification_run_id="qualification-snapshot",
            instance_sha256=candidate.instance_sha256,
            active_h=8,
            active_h_threshold_per_hour=0.001,
        )
    )
    return FormalQualificationTrace.from_decision(
        decision,
        manifest_id="manifest-test",
        benchmark_manifest_sha256="1" * 64,
        manifest_artifact_sha256="2" * 64,
    )


def test_formal_input_preparer_loads_only_attested_snapshot_bytes(
    tmp_path: Path,
) -> None:
    instance_dir, candidate, task = _formal_snapshot_fixture(tmp_path)
    loaded_roots: list[Path] = []

    def loader(snapshot_root: Path, _candidate: CandidateCase) -> object:
        loaded_roots.append(snapshot_root)
        return SimpleNamespace(
            metadata={},
            observed={
                filename: (snapshot_root / filename).read_bytes()
                for filename in CANONICAL_INSTANCE_FILES
            },
        )

    prepared = _prepare_formal_input_snapshot(
        task,
        instance_dir=instance_dir,
        candidate=candidate,
        qualification_trace=_formal_snapshot_trace(candidate),
        canonical_hash=canonical_instance_hash,
        loader=loader,
    )

    assert prepared.attestation.expected_instance_sha256 == candidate.instance_sha256
    assert prepared.attestation.payload_instance_sha256 == candidate.instance_sha256
    assert prepared.attestation.source_sha256_before == candidate.instance_sha256
    assert prepared.attestation.source_sha256_after == candidate.instance_sha256
    assert prepared.payload.metadata["instance_dir"] == (
        "formal-snapshot:snapshot-case"
    )
    assert prepared.payload.metadata["instance_sha256"] == candidate.instance_sha256
    assert len(loaded_roots) == 1
    assert not loaded_roots[0].exists()


def test_formal_input_preparer_rejects_payload_drift_even_if_source_hash_hook_lies(
    tmp_path: Path,
) -> None:
    instance_dir, candidate, task = _formal_snapshot_fixture(tmp_path)
    (instance_dir / CANONICAL_INSTANCE_FILES[-1]).write_bytes(b"drifted\n")

    with pytest.raises(CLIError, match="payload"):
        _prepare_formal_input_snapshot(
            task,
            instance_dir=instance_dir,
            candidate=candidate,
            qualification_trace=_formal_snapshot_trace(candidate),
            canonical_hash=lambda _path: (candidate.instance_sha256, {}),
            loader=lambda _path, _candidate: SimpleNamespace(metadata={}),
        )


def test_formal_input_preparer_rejects_symlinked_canonical_file(
    tmp_path: Path,
) -> None:
    instance_dir, candidate, task = _formal_snapshot_fixture(tmp_path)
    target = instance_dir / CANONICAL_INSTANCE_FILES[-1]
    target.unlink()
    target.symlink_to(instance_dir / CANONICAL_INSTANCE_FILES[0])

    with pytest.raises(CLIError, match="regular non-symlink"):
        _prepare_formal_input_snapshot(
            task,
            instance_dir=instance_dir,
            candidate=candidate,
            qualification_trace=_formal_snapshot_trace(candidate),
            canonical_hash=lambda _path: (candidate.instance_sha256, {}),
            loader=lambda _path, _candidate: SimpleNamespace(metadata={}),
        )


def test_freeze_formal_plan_writes_exact_immutable_matrix(tmp_path: Path) -> None:
    manifest_path, _manifest, plan_path, result = _freeze_formal_plan_for_cli_test(
        tmp_path
    )
    artifact = json.loads(plan_path.read_text(encoding="utf-8"))
    plan = artifact["formal_plan"]

    assert result["case_count"] == 36
    assert result["block_count"] == 180
    assert result["slot_count"] == 900
    assert result["formal_plan_id"] == artifact["formal_plan_id"]
    assert result["formal_plan_sha256"] == artifact["formal_plan_sha256"]
    assert result["formal_execution_implementation_sha256"] == (
        _FORMAL_EXECUTION_IMPLEMENTATION_SHA256
    )
    assert plan["expected_case_count"] == 36
    assert plan["expected_block_count"] == 180
    assert plan["expected_slot_count"] == 900
    assert len(plan["blocks"]) == 180
    assert sum(len(block["slots"]) for block in plan["blocks"]) == 900
    assert plan["suite"]["solver"]["threads"] == 1
    analysis = plan["analysis_contract"]
    assert analysis["baseline_profile"] == "A"
    assert analysis["candidate_profile"] == "E"
    assert analysis["primary_metric"] == "gap"
    assert analysis["minimum_medium_large_win_fraction"] == pytest.approx(0.60)
    assert analysis["minimum_improved_scale_tiers"] == 2
    assert analysis["minimum_runtime_speedup"] == pytest.approx(1.50)
    assert analysis["minimum_gap_reduction"] == pytest.approx(0.20)
    assert analysis["confidence_level"] == pytest.approx(0.95)
    assert analysis["bootstrap_resamples"] == 10
    assert analysis["bootstrap_method"] == (
        "scale_stratified_physical_case_cluster_percentile_bootstrap_v2"
    )
    assert analysis["bootstrap_sampling_unit"] == (
        "physical_case_with_all_solver_seeds_v1"
    )
    assert analysis["bootstrap_expected_solver_seeds"] == [11, 29, 47, 83, 101]
    assert analysis["minimum_effect_cases"] == 5
    assert analysis["median_summary_policy"] == (
        "median_of_within_case_seed_medians_v1"
    )
    assert analysis["sign_test_sampling_unit"] == (
        "physical_case_median_over_solver_seeds_v1"
    )
    assert result["formal_analysis_contract_sha256"] == canonical_sha256(analysis)

    frozen_bytes = plan_path.read_bytes()
    duplicate_args = build_parser().parse_args(
        [
            "freeze-formal-plan",
            "--manifest",
            str(manifest_path),
            "--output",
            str(plan_path),
        ]
    )
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        dispatch(
            duplicate_args,
            CLIHooks(
                model_identity=lambda: _MODEL_IDENTITY,
                formal_execution_implementation_sha256=(
                    lambda: _FORMAL_EXECUTION_IMPLEMENTATION_SHA256
                ),
                gurobi_version=lambda: _GUROBI_VERSION,
            ),
        )
    assert plan_path.read_bytes() == frozen_bytes


def test_run_formal_requires_anchor_and_exposes_no_axis_overrides(
    tmp_path: Path,
) -> None:
    parser = build_parser()
    commands = parser._subparsers._group_actions[0].choices  # type: ignore[attr-defined]
    run_formal = commands["run-formal"]
    option_strings = {
        option
        for action in run_formal._actions  # type: ignore[attr-defined]
        for option in action.option_strings
    }
    assert option_strings == {
        "--expected-plan-id",
        "--help",
        "--instances-root",
        "--manifest",
        "--plan",
        "--resource-lock",
        "--results-root",
        "-h",
    }
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "run-formal",
                "--manifest",
                str(tmp_path / "manifest.json"),
                "--plan",
                str(tmp_path / "plan.json"),
                "--instances-root",
                str(tmp_path / "instances"),
                "--results-root",
                str(tmp_path / "results"),
            ]
        )


def test_analyze_formal_requires_anchor_and_has_no_attempt_override(
    tmp_path: Path,
) -> None:
    parser = build_parser()
    commands = parser._subparsers._group_actions[0].choices  # type: ignore[attr-defined]
    analyze_formal = commands["analyze-formal"]
    option_strings = {
        option
        for action in analyze_formal._actions  # type: ignore[attr-defined]
        for option in action.option_strings
    }
    assert option_strings == {
        "--expected-evidence-root",
        "--expected-plan-id",
        "--help",
        "--manifest",
        "--output",
        "--plan",
        "--resource-lock",
        "--results-root",
        "-h",
    }
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "analyze-formal",
                "--manifest",
                str(tmp_path / "manifest.json"),
                "--plan",
                str(tmp_path / "plan.json"),
                "--results-root",
                str(tmp_path / "results"),
                "--output",
                str(tmp_path / "analysis.json"),
            ]
        )

    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "analyze-formal",
                "--manifest",
                str(tmp_path / "manifest.json"),
                "--plan",
                str(tmp_path / "plan.json"),
                "--expected-plan-id",
                "formal-plan-" + "0" * 64,
                "--expected-evidence-root",
                "not-a-sha256",
                "--results-root",
                str(tmp_path / "results"),
                "--output",
                str(tmp_path / "analysis.json"),
            ]
        )

    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "analyze-formal",
                "--manifest",
                str(tmp_path / "manifest.json"),
                "--plan",
                str(tmp_path / "plan.json"),
                "--expected-plan-id",
                "formal-plan-" + "0" * 64,
                "--results-root",
                str(tmp_path / "results"),
                "--output",
                str(tmp_path / "analysis.json"),
            ]
        )

    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "analyze-formal",
                "--manifest",
                str(tmp_path / "manifest.json"),
                "--plan",
                str(tmp_path / "plan.json"),
                "--expected-plan-id",
                "formal-plan-" + "0" * 64,
                "--expected-evidence-root",
                "0" * 64,
                "--results-root",
                str(tmp_path / "results"),
                "--output",
                str(tmp_path / "analysis.json"),
                "--bootstrap-resamples",
                "1",
            ]
        )


def test_run_formal_rejects_wrong_external_anchor_before_execution(
    tmp_path: Path,
) -> None:
    manifest_path, _manifest, plan_path, _result = _freeze_formal_plan_for_cli_test(
        tmp_path
    )
    executor_called = False
    hash_called = False

    def should_not_execute(*_args: Any, **_kwargs: Any) -> None:
        nonlocal executor_called
        executor_called = True

    def should_not_hash(_path: Path) -> tuple[str, Mapping[str, str]]:
        nonlocal hash_called
        hash_called = True
        return "0" * 64, {}

    args = _formal_run_cli_args(
        manifest_path=manifest_path,
        plan_path=plan_path,
        expected_plan_id=f"formal-plan-{'0' * 64}",
        tmp_path=tmp_path,
    )
    with pytest.raises(CLIError, match="expected-plan-id"):
        dispatch(
            args,
            CLIHooks(
                canonical_hash=should_not_hash,
                execute_formal=should_not_execute,
                model_identity=lambda: _MODEL_IDENTITY,
                formal_execution_implementation_sha256=(
                    lambda: _FORMAL_EXECUTION_IMPLEMENTATION_SHA256
                ),
                gurobi_version=lambda: _GUROBI_VERSION,
            ),
        )
    assert not executor_called
    assert not hash_called


def test_run_formal_preflights_all_cases_and_forwards_exact_context(
    tmp_path: Path,
) -> None:
    manifest_path, manifest, plan_path, frozen = _freeze_formal_plan_for_cli_test(
        tmp_path
    )
    hashed_instance_names: list[str] = []
    loaded_candidates: list[CandidateCase] = []
    binding_calls = {"model": 0, "implementation": 0, "gurobi": 0}
    observed_contexts: list[dict[str, Any]] = []
    observed_task_contexts: list[dict[str, Any]] = []

    def canonical_hash(path: Path) -> tuple[str, Mapping[str, str]]:
        hashed_instance_names.append(path.name)
        return hashlib.sha256(path.name.encode()).hexdigest(), {}

    def current_model_identity() -> ModelIdentity:
        binding_calls["model"] += 1
        return _MODEL_IDENTITY

    def current_implementation_sha256() -> str:
        binding_calls["implementation"] += 1
        return _FORMAL_EXECUTION_IMPLEMENTATION_SHA256

    def current_gurobi_version() -> str:
        binding_calls["gurobi"] += 1
        return _GUROBI_VERSION

    def fake_execute(
        _instance: object,
        _pair: PairSpec,
        _profile: Any,
        _telemetry: Any,
        _claim: Any,
        *,
        formal_context: Mapping[str, Any],
    ) -> object:
        observed_contexts.append(dict(formal_context))
        return object()

    def fake_formal_input_preparer(
        task: FormalSlotTask,
        instance_dir: Path,
        candidate: CandidateCase,
        qualification_trace: FormalQualificationTrace,
    ) -> PreparedFormalInput:
        loaded_candidates.append(candidate)
        observed_sha256, _files = canonical_hash(instance_dir)
        return PreparedFormalInput(
            payload=candidate,
            attestation=FormalInputAttestation(
                case_id=candidate.case_id,
                slot_id=task.slot.slot_id,
                expected_instance_sha256=candidate.instance_sha256,
                payload_instance_sha256=observed_sha256,
                source_sha256_before=observed_sha256,
                source_sha256_after=observed_sha256,
                qualification_trace=qualification_trace,
            ),
        )

    def fake_execute_formal(
        plan: Any,
        _results_root: str | Path,
        current_binding_provider: Any,
        pre_claim_preparer: Any,
        execute_slot: Any,
        *,
        session_snapshot: SessionSnapshot,
        resource_lock_path: str,
        now_utc: Any,
    ) -> FormalCampaignRunResult:
        # The real scheduler calls the provider immediately before every claim.
        # Exercise the injected provider more than once to prove it is fresh.
        assert session_snapshot == _FORMAL_ENVIRONMENT_CAPTURE.session
        assert resource_lock_path == str((tmp_path / "formal.lock").resolve())
        assert now_utc is not None
        current_binding_provider().validate_plan(plan)
        current_binding_provider().validate_plan(plan)
        block = plan.blocks[0]
        slot = block.slots[0]
        task = FormalSlotTask(
            campaign_id="formal-campaign-test",
            campaign_identity_sha256="1" * 64,
            benchmark_manifest_sha256=plan.benchmark_manifest_sha256,
            manifest_artifact_sha256=plan.manifest_artifact_sha256,
            formal_plan_id=plan.plan_id,
            formal_plan_sha256=plan.sha256,
            formal_execution_implementation_sha256=(
                plan.formal_execution_implementation_sha256
            ),
            model_id=plan.model_id,
            gurobi_version=plan.gurobi_version,
            session_id="formal-session-test",
            block=block,
            slot=slot,
            pair=PairSpec(
                manifest_id=plan.manifest_id,
                case_id=block.case_id,
                mode=plan.qualification_mode,
                solver_seed=block.solver_seed,
                time_budget_seconds=block.time_budget_seconds,
                solver=plan.suite.solver,
            ),
            profile=plan.suite.profile(slot.profile_code),
            telemetry=plan.suite.telemetry,
        )
        prepared = pre_claim_preparer(task)
        observed_task_contexts.append(
            bind_qualification_trace(
                task.formal_context,
                prepared.attestation.qualification_trace,
            )
        )
        execute_slot(
            task,
            SimpleNamespace(
                attempt_id=slot.attempt_id,
                directory=tmp_path / "claimed-attempt",
            ),
            prepared,
        )
        return FormalCampaignRunResult(
            campaign_id="formal-campaign-test",
            session_id="formal-session-test",
            initial_completed_count=0,
            recovered_count=0,
            executed_count=1,
            skipped_count=0,
            final_completed_count=1,
            evidence_id="formal-evidence-" + "2" * 64,
            evidence_root_sha256="2" * 64,
        )

    args = _formal_run_cli_args(
        manifest_path=manifest_path,
        plan_path=plan_path,
        expected_plan_id=str(frozen["formal_plan_id"]),
        tmp_path=tmp_path,
    )
    result = dispatch(
        args,
        CLIHooks(
            canonical_hash=canonical_hash,
            execute=fake_execute,
            execute_formal=fake_execute_formal,
            formal_input_preparer=fake_formal_input_preparer,
            model_identity=current_model_identity,
            formal_execution_implementation_sha256=(current_implementation_sha256),
            gurobi_version=current_gurobi_version,
            formal_environment_capture=_test_formal_environment_capture,
            preload_formal_runtime=lambda: None,
        ),
    )

    expected_names = {
        frozen_case.candidate.instance_ref for frozen_case in manifest.cases
    }
    assert set(hashed_instance_names[:36]) == expected_names
    assert len(hashed_instance_names[:36]) == 36
    assert len(hashed_instance_names) == 37
    assert hashed_instance_names[-1] == manifest.cases[0].candidate.instance_ref
    assert all(count == 3 for count in binding_calls.values())
    assert loaded_candidates == [manifest.cases[0].candidate]
    assert observed_contexts == observed_task_contexts
    assert set(observed_contexts[0]) == {
        "block_id",
        "block_index",
        "campaign_id",
        "campaign_identity_sha256",
        "formal_execution_implementation_sha256",
        "formal_plan_id",
        "formal_plan_sha256",
        "slot_id",
        "slot_index",
        "session_id",
        "qualification_trace",
        "qualification_trace_sha256",
    }
    assert result["session_id"] == "formal-session-test"
    assert result["executed_count"] == 1
    assert result["final_completed_count"] == 1
    assert result["evidence_id"] == "formal-evidence-" + "2" * 64
    assert result["evidence_root_sha256"] == "2" * 64


@pytest.mark.parametrize(
    ("drift", "expected_field"),
    (
        ("model", "model_id"),
        ("implementation", "formal_execution_implementation_sha256"),
        ("gurobi", "gurobi_version"),
    ),
)
def test_run_formal_rejects_environment_drift_before_executor(
    tmp_path: Path,
    drift: str,
    expected_field: str,
) -> None:
    manifest_path, _manifest, plan_path, frozen = _freeze_formal_plan_for_cli_test(
        tmp_path
    )
    executor_called = False

    def should_not_execute(*_args: Any, **_kwargs: Any) -> None:
        nonlocal executor_called
        executor_called = True

    drifted_model = replace(_MODEL_IDENTITY, implementation_sha256="9" * 64)
    args = _formal_run_cli_args(
        manifest_path=manifest_path,
        plan_path=plan_path,
        expected_plan_id=str(frozen["formal_plan_id"]),
        tmp_path=tmp_path,
    )
    with pytest.raises(CLIError, match=expected_field):
        dispatch(
            args,
            CLIHooks(
                canonical_hash=lambda path: (
                    hashlib.sha256(path.name.encode()).hexdigest(),
                    {},
                ),
                execute_formal=should_not_execute,
                model_identity=lambda: (
                    drifted_model if drift == "model" else _MODEL_IDENTITY
                ),
                formal_execution_implementation_sha256=lambda: (
                    "8" * 64
                    if drift == "implementation"
                    else _FORMAL_EXECUTION_IMPLEMENTATION_SHA256
                ),
                gurobi_version=lambda: (
                    "99.0-drifted" if drift == "gurobi" else _GUROBI_VERSION
                ),
                formal_environment_capture=_test_formal_environment_capture,
                preload_formal_runtime=lambda: None,
            ),
        )
    assert not executor_called


def test_run_formal_rejects_manifest_artifact_drift_before_executor(
    tmp_path: Path,
) -> None:
    manifest_path, _manifest, plan_path, frozen = _freeze_formal_plan_for_cli_test(
        tmp_path
    )
    original = json.loads(manifest_path.read_text(encoding="utf-8"))
    drifted_identity = {
        "manifest": original["manifest"],
        "qualification_mode": "LO",
        "model_identity": original["model_identity"],
        "model_id": original["model_id"],
    }
    drifted_manifest_path = write_canonical_json_exclusive(
        tmp_path / "manifest-drifted.json",
        {
            "artifact_type": "mlnd_branch_cut_benchmark_manifest",
            "artifact_schema_version": 1,
            **drifted_identity,
            "benchmark_manifest_artifact_sha256": canonical_sha256(drifted_identity),
        },
    )
    executor_called = False
    hash_called = False

    def should_not_execute(*_args: Any, **_kwargs: Any) -> None:
        nonlocal executor_called
        executor_called = True

    def should_not_hash(_path: Path) -> tuple[str, Mapping[str, str]]:
        nonlocal hash_called
        hash_called = True
        return "0" * 64, {}

    args = _formal_run_cli_args(
        manifest_path=drifted_manifest_path,
        plan_path=plan_path,
        expected_plan_id=str(frozen["formal_plan_id"]),
        tmp_path=tmp_path,
    )
    with pytest.raises(CLIError, match="manifest_artifact_sha256"):
        dispatch(
            args,
            CLIHooks(
                canonical_hash=should_not_hash,
                execute_formal=should_not_execute,
                model_identity=lambda: _MODEL_IDENTITY,
                formal_execution_implementation_sha256=(
                    lambda: _FORMAL_EXECUTION_IMPLEMENTATION_SHA256
                ),
                gurobi_version=lambda: _GUROBI_VERSION,
                formal_environment_capture=_test_formal_environment_capture,
                preload_formal_runtime=lambda: None,
            ),
        )
    assert not executor_called
    assert not hash_called


def test_run_formal_rejects_instance_drift_during_full_preflight(
    tmp_path: Path,
) -> None:
    manifest_path, manifest, plan_path, frozen = _freeze_formal_plan_for_cli_test(
        tmp_path
    )
    drifted = manifest.cases[17].candidate
    hashed_names: list[str] = []
    executor_called = False

    def canonical_hash(path: Path) -> tuple[str, Mapping[str, str]]:
        hashed_names.append(path.name)
        digest = hashlib.sha256(path.name.encode()).hexdigest()
        if path.name == drifted.instance_ref:
            digest = "7" * 64
        return digest, {}

    def should_not_execute(*_args: Any, **_kwargs: Any) -> None:
        nonlocal executor_called
        executor_called = True

    args = _formal_run_cli_args(
        manifest_path=manifest_path,
        plan_path=plan_path,
        expected_plan_id=str(frozen["formal_plan_id"]),
        tmp_path=tmp_path,
    )
    with pytest.raises(CLIError, match=f"instance hash mismatch for {drifted.case_id}"):
        dispatch(
            args,
            CLIHooks(
                canonical_hash=canonical_hash,
                execute_formal=should_not_execute,
                model_identity=lambda: _MODEL_IDENTITY,
                formal_execution_implementation_sha256=(
                    lambda: _FORMAL_EXECUTION_IMPLEMENTATION_SHA256
                ),
                gurobi_version=lambda: _GUROBI_VERSION,
                formal_environment_capture=_test_formal_environment_capture,
                preload_formal_runtime=lambda: None,
            ),
        )
    assert hashed_names[-1] == drifted.instance_ref
    assert len(hashed_names) == 18
    assert not executor_called


def test_qualification_parser_hard_limits_jobs_to_one(tmp_path: Path) -> None:
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "qualify",
                "--pool",
                str(tmp_path / "pool.json"),
                "--instances-root",
                str(tmp_path / "instances"),
                "--output",
                str(tmp_path / "qualification.json"),
                "--jobs",
                "2",
            ]
        )


def test_qualification_and_formal_run_share_nonblocking_resource_lock(
    tmp_path: Path,
) -> None:
    lock_path = tmp_path / "shared-benchmark.lock"
    candidate = _candidate("small-lock", demand_block_id="lock")
    pool_path = _write_pool(tmp_path / "pool.json", _pool((candidate,)))
    instances_root = tmp_path / "instances"
    instances_root.joinpath(candidate.instance_ref).mkdir(parents=True)
    qualification_called = False

    def should_not_qualify(*_args: Any, **_kwargs: Any) -> None:
        nonlocal qualification_called
        qualification_called = True

    qualify_args = _qualification_cli_args(
        pool_path=pool_path,
        instances_root=instances_root,
        output=tmp_path / "qualification.json",
        extra=("--resource-lock", str(lock_path)),
    )
    run_args = build_parser().parse_args(
        [
            "run",
            "--manifest",
            str(tmp_path / "missing-manifest.json"),
            "--instances-root",
            str(instances_root),
            "--results-root",
            str(tmp_path / "results"),
            "--batch-output",
            str(tmp_path / "batch.json"),
            "--resource-lock",
            str(lock_path),
        ]
    )

    with BenchmarkResourceLock(lock_path, command="test-owner"):
        with pytest.raises(CLIError, match="already holds"):
            dispatch(qualify_args, _qualification_hooks(should_not_qualify))
        with pytest.raises(CLIError, match="already holds"):
            dispatch(run_args, CLIHooks())

    assert not qualification_called
    assert not (tmp_path / "qualification.json").exists()
    assert not (tmp_path / "batch.json").exists()


def test_default_loader_applies_logical_candidate_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.data_processor import loader as loader_module

    captured: dict[str, Any] = {}

    class FakeDataLoader:
        def load_instance_dir(
            self,
            instance_dir: str,
            model_config: Any,
            demand_scale: float,
        ) -> str:
            captured.update(
                instance_dir=instance_dir,
                model_config=model_config,
                demand_scale=demand_scale,
            )
            return "profiled-instance"

    monkeypatch.setattr(loader_module, "DataLoader", FakeDataLoader)
    profile = replace(
        LOCKED_QUALIFICATION_PROFILES[0],
        profile_id="test-loader-all-model-axes",
        demand_scale=3.5,
        feeder_rider_cost_factor=0.42,
        locker_fixed_cost_per_hour=7.25,
        display_origin_choices=2,
        general_corridor_choices=3,
        expected_screening_is_exact=False,
    )
    monkeypatch.setattr(cli_module, "LOCKED_QUALIFICATION_PROFILES", (profile,))
    candidate = _candidate(solve_profile=profile)

    loaded = _default_instance_loader(tmp_path, candidate)

    assert loaded == "profiled-instance"
    assert captured["instance_dir"] == str(tmp_path)
    assert captured["demand_scale"] == 3.5
    assert captured["model_config"].feeder_rider_cost_factor == 0.42
    assert captured["model_config"].locker_fixed_cost_per_hour == 7.25
    assert captured["model_config"].display_origin_choices == 2
    assert captured["model_config"].general_corridor_choices == 3


@pytest.mark.parametrize(
    ("drift", "message"),
    (
        ("config", "does not retain the qualification model config"),
        ("demand", "does not retain the qualification demand scale"),
        ("screening", "has screening exactness"),
    ),
)
def test_default_loader_rejects_real_instance_profile_readback_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    drift: str,
    message: str,
) -> None:
    from src.data_processor import loader as loader_module
    from src.data_processor.instance import Instance

    candidate = _candidate(f"small-loader-{drift}")
    profile = LOCKED_QUALIFICATION_PROFILES[0]
    instance = Instance(
        config=profile.effective_model_config(),
        metadata={"demand_scale": profile.demand_scale},
    )
    if drift == "config":
        instance.config = replace(instance.config, locker_fixed_cost_per_hour=99.0)
    elif drift == "demand":
        instance.metadata["demand_scale"] = profile.demand_scale + 1.0
    else:
        instance.screening_is_exact = lambda _mode: False  # type: ignore[method-assign]

    class FakeDataLoader:
        def load_instance_dir(self, *_args: Any, **_kwargs: Any) -> Instance:
            return instance

    monkeypatch.setattr(loader_module, "DataLoader", FakeDataLoader)

    with pytest.raises(CLIError, match=message):
        _default_instance_loader(tmp_path, candidate)


def test_candidate_model_contract_rejects_legacy_profile_payload() -> None:
    candidate = _candidate("small-legacy-profile")
    legacy_profile = dict(candidate.generator_parameters["qualification_solve_profile"])
    legacy_profile.pop("locker_fixed_cost_per_hour")
    legacy_candidate = replace(
        candidate,
        generator_parameters={
            **candidate.generator_parameters,
            "qualification_solve_profile": legacy_profile,
        },
    )

    with pytest.raises(CLIError, match="invalid qualification_solve_profile"):
        cli_module._candidate_model_contract(legacy_candidate)


def test_candidate_model_contract_rejects_every_identity_tamper() -> None:
    candidate = _candidate("small-model-identity-tamper")
    identity = dict(candidate.generator_parameters["qualification_model_identity"])
    variants = (
        {key: value for key, value in identity.items() if key != "schema"},
        {**identity, "unexpected": True},
        {**identity, "schema": "legacy-model-identity"},
        {**identity, "effective_model_config_sha256": "0" * 64},
        {**identity, "qualification_case_sha256": "1" * 64},
        {**identity, "expected_screening_is_exact": False},
    )

    for tampered_identity in variants:
        tampered = replace(
            candidate,
            generator_parameters={
                **candidate.generator_parameters,
                "qualification_model_identity": tampered_identity,
            },
        )
        with pytest.raises(CLIError, match="inconsistent qualification_model_identity"):
            cli_module._candidate_model_contract(tampered)


def test_build_pool_command_maps_paths_and_locked_options(tmp_path: Path) -> None:
    pool = _pool((_candidate(),))
    received: list[Any] = []

    def fake_build(config: Any) -> Any:
        received.append(config)
        return SimpleNamespace(pool=pool, pool_path=tmp_path / "pool.json")

    args = build_parser().parse_args(
        [
            "build-pool",
            "--output-root",
            str(tmp_path / "instances"),
            "--nanshan-raw",
            str(tmp_path / "nanshan.csv"),
            "--tsl-raw",
            str(tmp_path / "tsl.csv"),
            "--candidate-hours",
            "9",
            "22",
        ]
    )
    result = dispatch(args, CLIHooks(build_pool=fake_build))

    assert result["candidate_count"] == 1
    assert received[0].candidate_hour_start == 9
    assert received[0].candidate_hour_end == 22
    assert received[0].output_root == tmp_path / "instances"
    assert received[0].pool_label == "mlnd-real-data-overcomplete-qualification-v5"


def test_qualify_uses_single_thread_and_records_normal_solver_rejection(
    tmp_path: Path,
) -> None:
    candidate = _candidate()
    pool_path = _write_pool(tmp_path / "pool.json", _pool((candidate,)))
    instances_root = tmp_path / "instances"
    instances_root.joinpath(candidate.case_id).mkdir(parents=True)
    outcome = _runner_outcome(
        candidate,
        accepted=False,
        qualification_run_id="qualification-test",
        policy=QualificationPolicy(large_scale_open_h_target=(50, 60)),
    )
    seen: list[dict[str, Any]] = []

    def fake_qualify(instance: object, mode: str, **kwargs: Any):
        seen.append({"instance": instance, "mode": mode, **kwargs})
        return outcome

    hooks = CLIHooks(
        canonical_hash=lambda _path: (candidate.instance_sha256, {}),
        load_instance=lambda _path, _candidate: "loaded-instance",
        qualify_instance=fake_qualify,
        model_identity=lambda: _MODEL_IDENTITY,
        qualification_implementation_sha256=(
            lambda: _QUALIFICATION_IMPLEMENTATION_SHA256
        ),
        gurobi_version=lambda: _GUROBI_VERSION,
        now_utc=lambda: _NOW,
    )
    output = tmp_path / "qualification.json"
    args = build_parser().parse_args(
        [
            "qualify",
            "--pool",
            str(pool_path),
            "--instances-root",
            str(instances_root),
            "--output",
            str(output),
            "--mode",
            "MX",
            "--qualification-run-id",
            "qualification-test",
            "--method",
            "2",
            "--miqcp-method",
            "1",
            "--mip-focus",
            "1",
            "--heuristics",
            "0.5",
            "--no-rel-heur-time",
            "5",
            "--presolve",
            "2",
            "--native-cuts",
            "3",
            "--numeric-focus",
            "2",
            "--feasibility-tolerance",
            "1e-7",
            "--optimality-tolerance",
            "2e-7",
            "--integer-feasibility-tolerance",
            "3e-6",
        ]
    )
    result = dispatch(args, hooks)
    artifact = json.loads(output.read_text(encoding="utf-8"))

    assert result["decision_count"] == 1
    assert result["accepted_count"] == 0
    assert seen[0]["settings"].threads == 1
    assert seen[0]["settings"].method == 2
    assert seen[0]["settings"].miqcp_method == 1
    assert seen[0]["settings"].mip_focus == 1
    assert seen[0]["settings"].heuristics == pytest.approx(0.5)
    assert seen[0]["settings"].no_rel_heur_time == pytest.approx(5.0)
    assert seen[0]["settings"].presolve == 2
    assert seen[0]["settings"].native_cuts == 3
    assert seen[0]["settings"].numeric_focus == 2
    assert seen[0]["settings"].feasibility_tolerance == pytest.approx(1e-7)
    assert seen[0]["settings"].optimality_tolerance == pytest.approx(2e-7)
    assert seen[0]["settings"].integer_feasibility_tolerance == pytest.approx(3e-6)
    assert not seen[0]["geometry_start_policy"].enabled
    assert seen[0]["mode"] == "MX"
    assert isinstance(seen[0]["expected_model_contract"], QualificationModelContract)
    assert seen[0]["expected_model_contract"] == (
        cli_module._candidate_model_contract(candidate)
    )
    frozen_decision = artifact["qualification_ledger"][0]
    assert frozen_decision["accepted"] is False
    assert frozen_decision["rejection_reasons"] == list(
        outcome.decision.rejection_reasons
    )
    assert artifact["runner_outcomes"][0]["fresh_model_validation_performed"]
    assert artifact["runner_outcomes"][0]["feasibility_objective_used"] is False
    assert artifact["runner_outcomes"][0]["phase_timings"]["total_seconds"] >= 0.0
    assert artifact["qualification_implementation_sha256"] == (
        _QUALIFICATION_IMPLEMENTATION_SHA256
    )
    assert artifact["gurobi_version"] == _GUROBI_VERSION
    assert artifact["native_settings"]["mip_focus"] == 1
    assert artifact["native_settings"]["miqcp_method"] == 1
    assert artifact["geometry_start_policy"]["enabled"] is False
    assert artifact["qualification_journal"]["journal_file_sha256"]


def test_qualify_freezes_geometry_fallback_in_run_id_journal_and_ledger(
    tmp_path: Path,
) -> None:
    candidate = _candidate("small-geometry-policy")
    pool_path = _write_pool(tmp_path / "pool.json", _pool((candidate,)))
    instances_root = tmp_path / "instances"
    instances_root.joinpath(candidate.case_id).mkdir(parents=True)
    seen: list[GeometryStartPolicy] = []

    def fake_qualify(candidate: CandidateCase, _mode: str, **kwargs: Any):
        geometry_policy = kwargs["geometry_start_policy"]
        seen.append(geometry_policy)
        return _runner_outcome(
            candidate,
            accepted=True,
            qualification_run_id=kwargs["qualification_run_id"],
            policy=kwargs["policy"],
            geometry_start_policy=geometry_policy,
        )

    output = tmp_path / "qualification.json"
    dispatch(
        _qualification_cli_args(
            pool_path=pool_path,
            instances_root=instances_root,
            output=output,
            extra=(
                "--geometry-start",
                "--geometry-start-seconds",
                "7.5",
                "--geometry-start-node-limit",
                "91",
            ),
        ),
        _qualification_hooks(fake_qualify),
    )
    artifact = json.loads(output.read_text(encoding="utf-8"))
    journal_identity_path = (
        output.with_name(f"{output.name}.journal") / "journal_identity.json"
    )
    journal_identity = json.loads(journal_identity_path.read_text(encoding="utf-8"))

    default_output = tmp_path / "qualification-default.json"
    dispatch(
        _qualification_cli_args(
            pool_path=pool_path,
            instances_root=instances_root,
            output=default_output,
        ),
        _qualification_hooks(fake_qualify),
    )
    default_artifact = json.loads(default_output.read_text(encoding="utf-8"))

    assert seen == [
        GeometryStartPolicy(
            enabled=True,
            per_attempt_seconds=7.5,
            start_node_limit=91,
        ),
        GeometryStartPolicy(),
    ]
    assert artifact["geometry_start_policy"] == seen[0].to_payload()
    assert artifact["runner_outcomes"][0]["geometry_start_policy"] == (
        seen[0].to_payload()
    )
    assert journal_identity["identity"]["geometry_start_policy"] == (
        seen[0].to_payload()
    )
    model_contract = cli_module._candidate_model_contract(candidate)
    assert (
        journal_identity["identity"]["ordered_cases"][0]["qualification_case_sha256"]
        == model_contract.qualification_case_sha256
    )
    checkpoint = json.loads(
        next(output.with_name(f"{output.name}.journal").glob("case-*.json")).read_text(
            encoding="utf-8"
        )
    )
    assert checkpoint["identity"]["qualification_case_sha256"] == (
        model_contract.qualification_case_sha256
    )
    assert "geometry_start_policy" in next(
        output.with_name(f"{output.name}.journal").glob("case-*.json")
    ).read_text(encoding="utf-8")
    assert artifact["qualification_run_id"] != default_artifact["qualification_run_id"]


def test_qualify_advances_contiguous_rank_layers_and_stops_each_stratum(
    tmp_path: Path,
) -> None:
    candidates = tuple(
        _replacement_candidate(
            block_id=block_id,
            scale="small",
            replacement_rank=rank,
            expected_strata=2,
            replacement_count=4,
        )
        for block_id in ("block-a", "block-b")
        for rank in range(4)
    )
    pool = CandidatePool(
        candidates=candidates,
        candidate_seed_order=(11, 29, 47, 83),
        pool_label="test-overcomplete-v5",
    )
    pool_path = _write_pool(tmp_path / "pool.json", pool)
    instances_root = tmp_path / "instances"
    for candidate in candidates:
        instances_root.joinpath(candidate.instance_ref).mkdir(parents=True)

    attempts: list[tuple[str, int]] = []

    def fake_qualify(candidate: CandidateCase, _mode: str, **kwargs: Any):
        selection = candidate.generator_parameters["pre_registered_selection"]
        rank = selection["replacement_rank_within_stratum"]
        attempts.append((candidate.demand_block_id, rank))
        accepted = candidate.demand_block_id == "block-a" or rank in {1, 2}
        return _runner_outcome(
            candidate,
            accepted=accepted,
            qualification_run_id=kwargs["qualification_run_id"],
            policy=kwargs["policy"],
        )

    output = tmp_path / "qualification.json"
    result = dispatch(
        build_parser().parse_args(
            [
                "qualify",
                "--pool",
                str(pool_path),
                "--instances-root",
                str(instances_root),
                "--output",
                str(output),
            ]
        ),
        CLIHooks(
            canonical_hash=lambda path: (
                hashlib.sha256(path.name.encode()).hexdigest(),
                {},
            ),
            load_instance=lambda _path, candidate: candidate,
            qualify_instance=fake_qualify,
            model_identity=lambda: _MODEL_IDENTITY,
            qualification_implementation_sha256=(
                lambda: _QUALIFICATION_IMPLEMENTATION_SHA256
            ),
            gurobi_version=lambda: _GUROBI_VERSION,
            now_utc=lambda: _NOW,
        ),
    )
    artifact = json.loads(output.read_text(encoding="utf-8"))

    assert attempts == [
        ("block-a", 0),
        ("block-b", 0),
        ("block-a", 1),
        ("block-b", 1),
        ("block-b", 2),
    ]
    assert result["candidate_count"] == 8
    assert result["attempted_count"] == 5
    assert result["unattempted_count"] == 3
    case_ids = {
        (
            candidate.demand_block_id,
            candidate.generator_parameters["pre_registered_selection"][
                "replacement_rank_within_stratum"
            ],
        ): candidate.case_id
        for candidate in candidates
    }
    assert [item["case_id"] for item in artifact["qualification_ledger"]] == [
        case_ids[("block-a", 0)],
        case_ids[("block-a", 1)],
        case_ids[("block-b", 0)],
        case_ids[("block-b", 1)],
        case_ids[("block-b", 2)],
    ]
    outcomes = {item["case_id"]: item for item in artifact["runner_outcomes"]}
    assert not outcomes[case_ids[("block-a", 2)]]["attempted"]
    assert outcomes[case_ids[("block-a", 2)]]["not_attempted_reason"] == (
        "not_attempted_after_stratum_target_reached"
    )
    assert not outcomes[case_ids[("block-b", 3)]]["attempted"]
    assert artifact["replacement_schedule"] == {
        "attempted_candidate_count": 5,
        "enabled": True,
        "selection_rule": OVERCOMPLETE_SELECTION_RULE,
        "stratum_count": 2,
        "unattempted_candidate_count": 3,
    }


def test_qualify_crash_resume_matches_uninterrupted_final_artifact(
    tmp_path: Path,
) -> None:
    candidates = tuple(
        _candidate(
            f"small-resume-{index}",
            demand_block_id=f"resume-{index}",
        )
        for index in range(3)
    )
    pool_path = _write_pool(tmp_path / "pool.json", _pool(candidates))
    instances_root = tmp_path / "instances"
    for candidate in candidates:
        instances_root.joinpath(candidate.instance_ref).mkdir(parents=True)
    output = tmp_path / "qualification.json"
    common_extra = ("--qualification-run-id", "qualification-resume-test")

    first_run_attempts: list[str] = []

    def crashing_qualify(candidate: CandidateCase, _mode: str, **kwargs: Any):
        first_run_attempts.append(candidate.case_id)
        if candidate == candidates[1]:
            raise KeyboardInterrupt("synthetic qualification crash")
        return _runner_outcome(
            candidate,
            accepted=True,
            qualification_run_id=kwargs["qualification_run_id"],
            policy=kwargs["policy"],
        )

    with pytest.raises(KeyboardInterrupt, match="synthetic qualification crash"):
        dispatch(
            _qualification_cli_args(
                pool_path=pool_path,
                instances_root=instances_root,
                output=output,
                extra=common_extra,
            ),
            _qualification_hooks(crashing_qualify),
        )

    journal_dir = output.with_name(f"{output.name}.journal")
    assert not output.exists()
    assert first_run_attempts == [candidates[0].case_id, candidates[1].case_id]
    assert len(tuple(journal_dir.glob("case-*.json"))) == 1

    resumed_attempts: list[str] = []

    def resumed_qualify(candidate: CandidateCase, _mode: str, **kwargs: Any):
        resumed_attempts.append(candidate.case_id)
        return _runner_outcome(
            candidate,
            accepted=True,
            qualification_run_id=kwargs["qualification_run_id"],
            policy=kwargs["policy"],
        )

    resumed_result = dispatch(
        _qualification_cli_args(
            pool_path=pool_path,
            instances_root=instances_root,
            output=output,
            extra=common_extra,
        ),
        _qualification_hooks(resumed_qualify),
    )
    assert resumed_attempts == [candidates[1].case_id, candidates[2].case_id]
    assert resumed_result["attempted_count"] == 3

    clean_output = tmp_path / "qualification-clean.json"
    dispatch(
        _qualification_cli_args(
            pool_path=pool_path,
            instances_root=instances_root,
            output=clean_output,
            extra=common_extra,
        ),
        _qualification_hooks(resumed_qualify),
    )
    resumed_artifact = json.loads(output.read_text(encoding="utf-8"))
    clean_artifact = json.loads(clean_output.read_text(encoding="utf-8"))
    assert resumed_artifact == clean_artifact
    assert len(resumed_artifact["qualification_journal"]["journal_file_sha256"]) == 4


def test_qualify_revalidates_current_instance_before_checkpoint_reuse(
    tmp_path: Path,
) -> None:
    candidates = tuple(
        _candidate(f"small-drift-{index}", demand_block_id=f"drift-{index}")
        for index in range(2)
    )
    pool_path = _write_pool(tmp_path / "pool.json", _pool(candidates))
    instances_root = tmp_path / "instances"
    for candidate in candidates:
        instances_root.joinpath(candidate.instance_ref).mkdir(parents=True)
    output = tmp_path / "qualification.json"
    extra = ("--qualification-run-id", "qualification-instance-drift")

    def crashing_qualify(candidate: CandidateCase, _mode: str, **kwargs: Any):
        if candidate == candidates[1]:
            raise KeyboardInterrupt("synthetic crash after first checkpoint")
        return _runner_outcome(
            candidate,
            accepted=True,
            qualification_run_id=kwargs["qualification_run_id"],
            policy=kwargs["policy"],
        )

    with pytest.raises(KeyboardInterrupt):
        dispatch(
            _qualification_cli_args(
                pool_path=pool_path,
                instances_root=instances_root,
                output=output,
                extra=extra,
            ),
            _qualification_hooks(crashing_qualify),
        )

    solver_called = False

    def should_not_run(*_args: Any, **_kwargs: Any) -> None:
        nonlocal solver_called
        solver_called = True
        raise AssertionError("instance drift must abort before solver work")

    resume_hooks = replace(
        _qualification_hooks(should_not_run),
        canonical_hash=lambda path: (
            "f" * 64
            if path.name == candidates[0].instance_ref
            else hashlib.sha256(path.name.encode()).hexdigest(),
            {},
        ),
    )
    with pytest.raises(CLIError, match="instance hash mismatch"):
        dispatch(
            _qualification_cli_args(
                pool_path=pool_path,
                instances_root=instances_root,
                output=output,
                extra=extra,
            ),
            resume_hooks,
        )
    assert not solver_called
    assert not output.exists()


def test_qualify_rejects_semantically_inconsistent_checkpoint(
    tmp_path: Path,
) -> None:
    candidates = tuple(
        _candidate(f"small-tamper-{index}", demand_block_id=f"tamper-{index}")
        for index in range(2)
    )
    pool_path = _write_pool(tmp_path / "pool.json", _pool(candidates))
    instances_root = tmp_path / "instances"
    for candidate in candidates:
        instances_root.joinpath(candidate.instance_ref).mkdir(parents=True)
    output = tmp_path / "qualification.json"
    extra = ("--qualification-run-id", "qualification-semantic-tamper")

    def crashing_qualify(candidate: CandidateCase, _mode: str, **kwargs: Any):
        if candidate == candidates[1]:
            raise KeyboardInterrupt("synthetic crash after first checkpoint")
        return _runner_outcome(
            candidate,
            accepted=True,
            qualification_run_id=kwargs["qualification_run_id"],
            policy=kwargs["policy"],
        )

    with pytest.raises(KeyboardInterrupt):
        dispatch(
            _qualification_cli_args(
                pool_path=pool_path,
                instances_root=instances_root,
                output=output,
                extra=extra,
            ),
            _qualification_hooks(crashing_qualify),
        )

    checkpoint = next(output.with_name(f"{output.name}.journal").glob("case-*.json"))
    payload = json.loads(checkpoint.read_text(encoding="utf-8"))
    decision = payload["result"]["outcome"]["decision"]
    decision["open_h"] = 0
    decision["h_open_fraction"] = 0.0
    entry = {"identity": payload["identity"], "result": payload["result"]}
    payload["checkpoint_sha256"] = canonical_sha256(entry)
    checkpoint.write_text(canonical_json(payload) + "\n", encoding="utf-8")

    with pytest.raises(CLIError, match="decision does not match its evidence"):
        dispatch(
            _qualification_cli_args(
                pool_path=pool_path,
                instances_root=instances_root,
                output=output,
                extra=extra,
            ),
            _qualification_hooks(crashing_qualify),
        )


def test_checkpoint_outcome_enforces_process_invariants_and_numeric_types() -> None:
    candidate = _candidate("small-process-invariants")
    outcome = _runner_outcome(
        candidate,
        accepted=True,
        qualification_run_id="qualification-process-invariants",
        policy=QualificationPolicy(),
    )

    missing_validation = outcome.to_payload()
    missing_validation["fresh_model_validation_performed"] = False
    missing_validation["captured_variable_count"] = 0
    with pytest.raises(CLIError, match="fresh original-model validation"):
        _qualification_outcome_from_payload(missing_validation)

    malformed_evidence = outcome.to_payload()
    malformed_evidence["evidence"]["total_demand_per_hour"] = "100.0"
    with pytest.raises(CLIError, match="must be a number or null"):
        _qualification_outcome_from_payload(malformed_evidence)

    malformed_process_value = outcome.to_payload()
    malformed_process_value["required_service_per_hour"] = "0.1"
    with pytest.raises(CLIError, match="must be positive and finite"):
        _qualification_outcome_from_payload(malformed_process_value)

    malformed_audit = outcome.to_payload()
    malformed_audit["witness_model_audit"]["unexpected"] = True
    with pytest.raises(CLIError, match="witness_model_audit fields are inconsistent"):
        _qualification_outcome_from_payload(malformed_audit)

    mismatched_audit = outcome.to_payload()
    mismatched_audit["witness_model_audit"]["demand_scale"] = 4.0
    with pytest.raises(CLIError, match="witness model audit"):
        _qualification_task_result_from_payload(
            {
                "case_id": candidate.case_id,
                "attempted": True,
                "outcome": mismatched_audit,
                "orchestration_error": None,
            },
            candidate=candidate,
            qualification_run_id="qualification-process-invariants",
            policy=QualificationPolicy(),
            geometry_start_policy=GeometryStartPolicy(),
        )


def test_historical_null_outcome_checkpoint_is_never_reused() -> None:
    candidate = _candidate("small-historical-null")
    with pytest.raises(CLIError, match="historical orchestration-error checkpoint"):
        _qualification_task_result_from_payload(
            {
                "case_id": candidate.case_id,
                "attempted": True,
                "outcome": None,
                "orchestration_error": "legacy model construction failure",
            },
            candidate=candidate,
            qualification_run_id="qualification-historical-null",
            policy=QualificationPolicy(),
            geometry_start_policy=GeometryStartPolicy(),
        )


def test_checkpoint_rejects_geometry_schedule_and_terminal_attempt_tampering() -> None:
    candidate = _candidate("small-geometry-checkpoint")
    geometry_policy = GeometryStartPolicy(enabled=True)
    first_success = _successful_geometry_attempt(12, "a")
    second_success = _successful_geometry_attempt(8, "b")
    planning_failure = GeometryStartAttemptAudit(
        plan=GeometryStartPlanAudit.planning_failure(
            plan_version=GEOMETRY_START_PLAN_VERSION,
            target_support_size=8,
        ),
        per_attempt_seconds=10.0,
        start_node_limit=500,
        error="synthetic planning failure",
    )
    geometry_outcome = replace(
        _runner_outcome(
            candidate,
            accepted=True,
            qualification_run_id="qualification-geometry-checkpoint",
            policy=QualificationPolicy(),
            geometry_start_policy=geometry_policy,
        ),
        temporary_constraint_used=True,
        temporary_constraint_removed=True,
        feasibility_objective_used=True,
        geometry_start_attempts=(first_success,),
        geometry_start_selected_plan_sha256="a" * 64,
    )

    assert _qualification_outcome_from_payload(geometry_outcome.to_payload()) == (
        geometry_outcome
    )

    wrong_prefix = replace(
        geometry_outcome,
        geometry_start_attempts=(second_success,),
        geometry_start_selected_plan_sha256="b" * 64,
    ).to_payload()
    with pytest.raises(CLIError, match="not a registered prefix"):
        _qualification_outcome_from_payload(wrong_prefix)

    early_success = replace(
        geometry_outcome,
        geometry_start_attempts=(first_success, planning_failure),
    ).to_payload()
    with pytest.raises(CLIError, match="terminal geometry attempt"):
        _qualification_outcome_from_payload(early_success)

    multiple_successes = replace(
        geometry_outcome,
        geometry_start_attempts=(first_success, second_success),
        geometry_start_selected_plan_sha256="b" * 64,
    ).to_payload()
    with pytest.raises(CLIError, match="multiple successful"):
        _qualification_outcome_from_payload(multiple_successes)

    wrong_selected = replace(
        geometry_outcome,
        geometry_start_selected_plan_sha256="c" * 64,
    ).to_payload()
    with pytest.raises(CLIError, match="does not match one successful"):
        _qualification_outcome_from_payload(wrong_selected)

    cleanup_failure = GeometryStartAttemptAudit(
        plan=_geometry_plan_audit(12, "c"),
        per_attempt_seconds=10.0,
        start_node_limit=500,
        install=_safe_geometry_install(),
        solve_status="TIME_LIMIT",
        solve_seconds=9.0,
        cleanup_performed=True,
        cleanup_succeeded=False,
        error="synthetic cleanup failure",
    )
    early_cleanup_failure = replace(
        geometry_outcome,
        accepted=False,
        rejection_reasons=(QualificationRunnerReason.GEOMETRY_START_CLEANUP_FAILED,),
        geometry_start_attempts=(cleanup_failure, planning_failure),
        geometry_start_selected_plan_sha256=None,
    ).to_payload()
    with pytest.raises(CLIError, match="terminal geometry attempt"):
        _qualification_outcome_from_payload(early_cleanup_failure)

    unsafe_attempt = GeometryStartAttemptAudit(
        plan=_geometry_plan_audit(12, "d"),
        per_attempt_seconds=10.0,
        start_node_limit=500,
        install=replace(
            _safe_geometry_install(),
            fingerprint_after=18,
            fingerprint_unchanged=False,
        ),
        cleanup_performed=True,
        cleanup_succeeded=True,
        error="synthetic start-neutral fingerprint mismatch",
    )
    early_safety_failure = replace(
        geometry_outcome,
        accepted=False,
        rejection_reasons=(
            QualificationRunnerReason.GEOMETRY_START_SAFETY_CHECK_FAILED,
        ),
        geometry_start_attempts=(unsafe_attempt, planning_failure),
        geometry_start_selected_plan_sha256=None,
    ).to_payload()
    with pytest.raises(CLIError, match="terminal geometry attempt"):
        _qualification_outcome_from_payload(early_safety_failure)

    install_failure = GeometryStartAttemptAudit(
        plan=_geometry_plan_audit(12, "e"),
        per_attempt_seconds=10.0,
        start_node_limit=500,
        error="synthetic installation failure",
    )
    early_install_failure = replace(
        geometry_outcome,
        accepted=False,
        rejection_reasons=(
            QualificationRunnerReason.GEOMETRY_START_INSTALLATION_FAILED,
        ),
        geometry_start_attempts=(install_failure, planning_failure),
        geometry_start_selected_plan_sha256=None,
    ).to_payload()
    with pytest.raises(CLIError, match="terminal geometry attempt"):
        _qualification_outcome_from_payload(early_install_failure)

    solve_failure = GeometryStartAttemptAudit(
        plan=_geometry_plan_audit(12, "f"),
        per_attempt_seconds=10.0,
        start_node_limit=500,
        install=_safe_geometry_install(),
        solve_status="ERROR",
        solve_seconds=1.0,
        cleanup_performed=True,
        cleanup_succeeded=True,
        error="synthetic solve failure",
    )
    early_solve_failure = replace(
        geometry_outcome,
        accepted=False,
        rejection_reasons=(QualificationRunnerReason.GEOMETRY_START_SOLVE_FAILED,),
        geometry_start_attempts=(solve_failure, planning_failure),
        geometry_start_selected_plan_sha256=None,
    ).to_payload()
    with pytest.raises(CLIError, match="terminal geometry attempt"):
        _qualification_outcome_from_payload(early_solve_failure)

    capture_failure = GeometryStartAttemptAudit(
        plan=_geometry_plan_audit(12, "1"),
        per_attempt_seconds=10.0,
        start_node_limit=500,
        install=_safe_geometry_install(),
        solve_status="TIME_LIMIT",
        solve_seconds=9.0,
        cleanup_performed=True,
        cleanup_succeeded=True,
        error="synthetic capture failure",
    )
    early_capture_failure = replace(
        geometry_outcome,
        accepted=False,
        rejection_reasons=(
            QualificationRunnerReason.COMPLETE_INCUMBENT_CAPTURE_FAILED,
        ),
        geometry_start_attempts=(capture_failure, planning_failure),
        geometry_start_selected_plan_sha256=None,
    ).to_payload()
    with pytest.raises(CLIError, match="terminal geometry attempt"):
        _qualification_outcome_from_payload(early_capture_failure)


def test_qualification_integrity_drift_is_fatal_not_a_case_rejection(
    tmp_path: Path,
) -> None:
    candidate = _candidate("small-code-drift", demand_block_id="code-drift")
    pool_path = _write_pool(tmp_path / "pool.json", _pool((candidate,)))
    instances_root = tmp_path / "instances"
    instances_root.joinpath(candidate.instance_ref).mkdir(parents=True)
    output = tmp_path / "qualification.json"
    model_identity_calls = 0

    def drifting_model_identity() -> ModelIdentity:
        nonlocal model_identity_calls
        model_identity_calls += 1
        if model_identity_calls == 1:
            return _MODEL_IDENTITY
        return replace(_MODEL_IDENTITY, implementation_sha256="f" * 64)

    solver_called = False

    def should_not_run(*_args: Any, **_kwargs: Any) -> None:
        nonlocal solver_called
        solver_called = True
        raise AssertionError("integrity drift must abort before solver work")

    hooks = replace(
        _qualification_hooks(should_not_run),
        model_identity=drifting_model_identity,
    )
    with pytest.raises(CLIError, match="model implementation/config changed"):
        dispatch(
            _qualification_cli_args(
                pool_path=pool_path,
                instances_root=instances_root,
                output=output,
            ),
            hooks,
        )

    assert not solver_called
    assert not output.exists()
    journal = output.with_name(f"{output.name}.journal")
    assert not tuple(journal.glob("case-*.json"))


def test_qualify_rejects_journal_identity_mismatch_before_solver_work(
    tmp_path: Path,
) -> None:
    candidates = tuple(
        _candidate(f"small-identity-{index}", demand_block_id=f"identity-{index}")
        for index in range(2)
    )
    pool_path = _write_pool(tmp_path / "pool.json", _pool(candidates))
    instances_root = tmp_path / "instances"
    for candidate in candidates:
        instances_root.joinpath(candidate.instance_ref).mkdir(parents=True)
    output = tmp_path / "qualification.json"
    journal_dir = tmp_path / "custom-checkpoints"
    first_extra = (
        "--qualification-run-id",
        "qualification-fixed-identity",
        "--journal-dir",
        str(journal_dir),
    )

    def crashing_qualify(candidate: CandidateCase, _mode: str, **kwargs: Any):
        if candidate == candidates[1]:
            raise SystemExit("synthetic stop after first checkpoint")
        return _runner_outcome(
            candidate,
            accepted=True,
            qualification_run_id=kwargs["qualification_run_id"],
            policy=kwargs["policy"],
        )

    with pytest.raises(SystemExit):
        dispatch(
            _qualification_cli_args(
                pool_path=pool_path,
                instances_root=instances_root,
                output=output,
                extra=first_extra,
            ),
            _qualification_hooks(crashing_qualify),
        )
    assert journal_dir.is_dir()

    solver_called = False

    def should_not_run(*_args: Any, **_kwargs: Any) -> None:
        nonlocal solver_called
        solver_called = True
        raise AssertionError("identity mismatch must fail before solver work")

    with pytest.raises(CLIError, match="journal identity mismatch"):
        dispatch(
            _qualification_cli_args(
                pool_path=pool_path,
                instances_root=instances_root,
                output=output,
                extra=(*first_extra, "--time-limit", "31"),
            ),
            _qualification_hooks(should_not_run),
        )
    assert not solver_called


def test_qualify_resumes_partial_replacement_layer_before_next_rank(
    tmp_path: Path,
) -> None:
    candidates = tuple(
        _replacement_candidate(
            block_id=block_id,
            scale="small",
            replacement_rank=rank,
            expected_strata=2,
            replacement_count=4,
        )
        for block_id in ("partial-a", "partial-b")
        for rank in range(4)
    )
    pool = CandidatePool(
        candidates=candidates,
        candidate_seed_order=(11, 29, 47, 83),
        pool_label="partial-layer-pool",
    )
    pool_path = _write_pool(tmp_path / "pool.json", pool)
    instances_root = tmp_path / "instances"
    for candidate in candidates:
        instances_root.joinpath(candidate.instance_ref).mkdir(parents=True)
    output = tmp_path / "qualification.json"
    extra = ("--qualification-run-id", "qualification-partial-layer")

    def crashing_qualify(candidate: CandidateCase, _mode: str, **kwargs: Any):
        selection = candidate.generator_parameters["pre_registered_selection"]
        if candidate.demand_block_id == "partial-b" and (
            selection["replacement_rank_within_stratum"] == 0
        ):
            raise KeyboardInterrupt("synthetic half-layer crash")
        return _runner_outcome(
            candidate,
            accepted=True,
            qualification_run_id=kwargs["qualification_run_id"],
            policy=kwargs["policy"],
        )

    with pytest.raises(KeyboardInterrupt):
        dispatch(
            _qualification_cli_args(
                pool_path=pool_path,
                instances_root=instances_root,
                output=output,
                extra=extra,
            ),
            _qualification_hooks(crashing_qualify),
        )

    resumed_attempts: list[tuple[str, int]] = []

    def resumed_qualify(candidate: CandidateCase, _mode: str, **kwargs: Any):
        rank = candidate.generator_parameters["pre_registered_selection"][
            "replacement_rank_within_stratum"
        ]
        resumed_attempts.append((candidate.demand_block_id, rank))
        return _runner_outcome(
            candidate,
            accepted=True,
            qualification_run_id=kwargs["qualification_run_id"],
            policy=kwargs["policy"],
        )

    result = dispatch(
        _qualification_cli_args(
            pool_path=pool_path,
            instances_root=instances_root,
            output=output,
            extra=extra,
        ),
        _qualification_hooks(resumed_qualify),
    )

    assert resumed_attempts == [
        ("partial-b", 0),
        ("partial-a", 1),
        ("partial-b", 1),
    ]
    assert result["attempted_count"] == 4
    assert result["unattempted_count"] == 4


def test_qualify_orchestration_error_aborts_without_checkpoint_or_replacement(
    tmp_path: Path,
) -> None:
    candidates = tuple(
        _replacement_candidate(
            block_id="orchestration-block",
            scale="small",
            replacement_rank=rank,
            expected_strata=1,
            replacement_count=4,
        )
        for rank in range(4)
    )
    pool = CandidatePool(
        candidates=candidates,
        candidate_seed_order=(11, 29, 47, 83),
        pool_label="orchestration-replacement-pool",
    )
    pool_path = _write_pool(tmp_path / "pool.json", pool)
    instances_root = tmp_path / "instances"
    for candidate in candidates:
        instances_root.joinpath(candidate.instance_ref).mkdir(parents=True)
    output = tmp_path / "qualification.json"
    first_attempts: list[int] = []

    def crashing_qualify(candidate: CandidateCase, _mode: str, **kwargs: Any):
        rank = candidate.generator_parameters["pre_registered_selection"][
            "replacement_rank_within_stratum"
        ]
        first_attempts.append(rank)
        raise RuntimeError("synthetic orchestration error")

    with pytest.raises(CLIError, match="synthetic orchestration error"):
        dispatch(
            _qualification_cli_args(
                pool_path=pool_path,
                instances_root=instances_root,
                output=output,
            ),
            _qualification_hooks(crashing_qualify),
        )
    assert first_attempts == [0]
    journal_dir = output.with_name(f"{output.name}.journal")
    assert tuple(journal_dir.glob("case-*.json")) == ()
    assert not output.exists()

    resumed_attempts: list[int] = []

    def resumed_qualify(candidate: CandidateCase, _mode: str, **kwargs: Any):
        rank = candidate.generator_parameters["pre_registered_selection"][
            "replacement_rank_within_stratum"
        ]
        resumed_attempts.append(rank)
        return _runner_outcome(
            candidate,
            accepted=True,
            qualification_run_id=kwargs["qualification_run_id"],
            policy=kwargs["policy"],
        )

    result = dispatch(
        _qualification_cli_args(
            pool_path=pool_path,
            instances_root=instances_root,
            output=output,
        ),
        _qualification_hooks(resumed_qualify),
    )
    artifact = json.loads(output.read_text(encoding="utf-8"))

    assert resumed_attempts == [0, 1]
    assert result["attempted_count"] == 2
    assert result["accepted_count"] == 2
    assert result["orchestration_error_count"] == 0
    assert all(item["accepted"] for item in artifact["qualification_ledger"])
    assert not artifact["runner_outcomes"][3]["attempted"]


def test_qualify_rejects_non_solver_infrastructure_outcome_without_checkpoint(
    tmp_path: Path,
) -> None:
    candidates = tuple(
        _candidate(f"small-null-{index}", demand_block_id=f"null-{index}")
        for index in range(2)
    )
    pool_path = _write_pool(tmp_path / "pool.json", _pool(candidates))
    instances_root = tmp_path / "instances"
    for candidate in candidates:
        instances_root.joinpath(candidate.instance_ref).mkdir(parents=True)
    output = tmp_path / "qualification.json"

    def rejected_outcome(
        candidate: CandidateCase,
        *,
        qualification_run_id: str,
        policy: QualificationPolicy,
    ) -> QualificationRunnerOutcome:
        evidence = QualificationEvidence(
            case_id=candidate.case_id,
            scale=candidate.scale,
            courier_only_objective=math.nan,
            witness_objective=math.nan,
            total_demand_per_hour=math.nan,
            witness_service_per_hour=math.nan,
            open_facilities=0,
            positive_cost_open_facilities=0,
            candidate_h=23,
            open_h=0,
            witness_feasible=False,
            qualification_run_id=qualification_run_id,
            instance_sha256=candidate.instance_sha256,
            active_h=0,
            active_h_threshold_per_hour=None,
        )
        decision = assess_qualification(evidence, policy)
        return QualificationRunnerOutcome(
            accepted=False,
            rejection_reasons=decision.rejection_reasons,
            evidence=evidence,
            decision=decision,
            required_service_per_hour=0.001,
            required_objective_upper_bound=None,
            temporary_constraint_used=False,
            temporary_constraint_removed=None,
            fresh_model_validation_performed=False,
            witness_status="NOT_RUN",
            validation_status="NOT_RUN",
            captured_variable_count=0,
            exception_messages=("synthetic initialization rejection",),
        )

    def crashing_qualify(candidate: CandidateCase, _mode: str, **kwargs: Any):
        return rejected_outcome(
            candidate,
            qualification_run_id=kwargs["qualification_run_id"],
            policy=kwargs["policy"],
        )

    with pytest.raises(CLIError, match="unexpected witness status"):
        dispatch(
            _qualification_cli_args(
                pool_path=pool_path,
                instances_root=instances_root,
                output=output,
            ),
            _qualification_hooks(crashing_qualify),
        )

    journal_dir = output.with_name(f"{output.name}.journal")
    assert tuple(journal_dir.glob("case-*.json")) == ()
    assert not output.exists()


@pytest.mark.parametrize(
    "fatal_error",
    [
        MemoryError("synthetic CLI memory exhaustion"),
        KeyboardInterrupt("synthetic CLI interrupt"),
        SystemExit("synthetic CLI exit"),
    ],
)
def test_qualify_process_exception_stops_without_case_checkpoint(
    tmp_path: Path,
    fatal_error: BaseException,
) -> None:
    candidate = _candidate("small-fatal", demand_block_id="fatal")
    pool_path = _write_pool(tmp_path / "pool.json", _pool((candidate,)))
    instances_root = tmp_path / "instances"
    instances_root.joinpath(candidate.instance_ref).mkdir(parents=True)
    output = tmp_path / "qualification.json"

    def fatal_qualify(*_args: Any, **_kwargs: Any) -> None:
        raise fatal_error

    with pytest.raises(type(fatal_error)):
        dispatch(
            _qualification_cli_args(
                pool_path=pool_path,
                instances_root=instances_root,
                output=output,
            ),
            _qualification_hooks(fatal_qualify),
        )

    journal_dir = output.with_name(f"{output.name}.journal")
    assert not output.exists()
    assert tuple(journal_dir.glob("case-*.json")) == ()


@pytest.mark.parametrize(
    "infrastructure_reason",
    [
        QualificationRunnerReason.ADAPTER_INITIALIZATION_FAILED,
        QualificationRunnerReason.COURIER_BASELINE_EXTRACTION_FAILED,
        QualificationRunnerReason.NATIVE_WITNESS_SOLVE_FAILED,
        QualificationRunnerReason.TEMPORARY_CONSTRAINT_ADD_FAILED,
        QualificationRunnerReason.TEMPORARY_CONSTRAINT_REMOVAL_FAILED,
        QualificationRunnerReason.CONSTRAINED_WITNESS_SOLVE_FAILED,
        QualificationRunnerReason.GEOMETRY_START_POLICY_FAILED,
        QualificationRunnerReason.GEOMETRY_START_INSTALLATION_FAILED,
        QualificationRunnerReason.GEOMETRY_START_SOLVE_FAILED,
        QualificationRunnerReason.GEOMETRY_START_SAFETY_CHECK_FAILED,
        QualificationRunnerReason.GEOMETRY_START_CLEANUP_FAILED,
        QualificationRunnerReason.COMPLETE_INCUMBENT_CAPTURE_FAILED,
        QualificationRunnerReason.WITNESS_MODEL_DISPOSAL_FAILED,
        QualificationRunnerReason.FRESH_MODEL_INITIALIZATION_FAILED,
        QualificationRunnerReason.FRESH_MODEL_REUSED,
        QualificationRunnerReason.FRESH_MODEL_BASELINE_EXTRACTION_FAILED,
        QualificationRunnerReason.FRESH_MODEL_BASELINE_MISMATCH,
        QualificationRunnerReason.COMPLETE_START_INSTALLATION_FAILED,
        QualificationRunnerReason.FRESH_VALIDATION_SOLVE_FAILED,
        QualificationRunnerReason.FRESH_VALIDATION_NO_INCUMBENT,
        QualificationRunnerReason.FRESH_MODEL_DISPOSAL_FAILED,
    ],
)
def test_qualification_mechanism_failure_aborts_without_checkpoint(
    tmp_path: Path,
    infrastructure_reason: str,
) -> None:
    candidate = _candidate("small-mechanism", demand_block_id="mechanism")
    pool_path = _write_pool(tmp_path / "pool.json", _pool((candidate,)))
    instances_root = tmp_path / "instances"
    instances_root.joinpath(candidate.instance_ref).mkdir(parents=True)
    output = tmp_path / "qualification.json"

    def failed_qualify(candidate: CandidateCase, _mode: str, **kwargs: Any):
        normal = _runner_outcome(
            candidate,
            accepted=False,
            qualification_run_id=kwargs["qualification_run_id"],
            policy=kwargs["policy"],
        )
        return replace(
            normal,
            rejection_reasons=(
                infrastructure_reason,
                *normal.decision.rejection_reasons,
            ),
            exception_messages=("synthetic infrastructure failure",),
        )

    with pytest.raises(CLIError, match="infrastructure failure"):
        dispatch(
            _qualification_cli_args(
                pool_path=pool_path,
                instances_root=instances_root,
                output=output,
            ),
            _qualification_hooks(failed_qualify),
        )

    journal_dir = output.with_name(f"{output.name}.journal")
    assert tuple(journal_dir.glob("case-*.json")) == ()
    assert not output.exists()


@pytest.mark.parametrize(
    "status",
    ["MEM_LIMIT", "INTERRUPTED", "NUMERIC", "NODE_LIMIT", "STATUS_999"],
)
def test_abnormal_qualification_status_aborts_without_checkpoint(
    tmp_path: Path,
    status: str,
) -> None:
    candidate = _candidate("small-status", demand_block_id="status")
    pool_path = _write_pool(tmp_path / "pool.json", _pool((candidate,)))
    instances_root = tmp_path / "instances"
    instances_root.joinpath(candidate.instance_ref).mkdir(parents=True)
    output = tmp_path / "qualification.json"

    def status_qualify(candidate: CandidateCase, _mode: str, **kwargs: Any):
        return replace(
            _runner_outcome(
                candidate,
                accepted=False,
                qualification_run_id=kwargs["qualification_run_id"],
                policy=kwargs["policy"],
            ),
            witness_status=status,
        )

    with pytest.raises(CLIError, match="unexpected witness status"):
        dispatch(
            _qualification_cli_args(
                pool_path=pool_path,
                instances_root=instances_root,
                output=output,
            ),
            _qualification_hooks(status_qualify),
        )

    assert tuple(output.with_name(f"{output.name}.journal").glob("case-*.json")) == ()
    assert not output.exists()


def test_instance_load_failure_aborts_before_solver_and_checkpoint(
    tmp_path: Path,
) -> None:
    candidate = _candidate("small-load", demand_block_id="load")
    pool_path = _write_pool(tmp_path / "pool.json", _pool((candidate,)))
    instances_root = tmp_path / "instances"
    instances_root.joinpath(candidate.instance_ref).mkdir(parents=True)
    output = tmp_path / "qualification.json"
    solver_called = False

    def broken_load(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("synthetic instance-load failure")

    def should_not_solve(*_args: Any, **_kwargs: Any) -> None:
        nonlocal solver_called
        solver_called = True

    with pytest.raises(CLIError, match="instance-load failure"):
        dispatch(
            _qualification_cli_args(
                pool_path=pool_path,
                instances_root=instances_root,
                output=output,
            ),
            _qualification_hooks(should_not_solve, load_instance=broken_load),
        )

    assert not solver_called
    assert tuple(output.with_name(f"{output.name}.journal").glob("case-*.json")) == ()
    assert not output.exists()


def test_solver_environment_failure_aborts_before_journal_or_model_build(
    tmp_path: Path,
) -> None:
    candidate = _candidate("small-licence", demand_block_id="licence")
    pool_path = _write_pool(tmp_path / "pool.json", _pool((candidate,)))
    instances_root = tmp_path / "instances"
    instances_root.joinpath(candidate.instance_ref).mkdir(parents=True)
    output = tmp_path / "qualification.json"
    model_called = False

    def broken_environment() -> str:
        raise RuntimeError("synthetic Gurobi licence failure")

    def should_not_build(*_args: Any, **_kwargs: Any) -> None:
        nonlocal model_called
        model_called = True

    hooks = replace(
        _qualification_hooks(should_not_build),
        gurobi_version=broken_environment,
    )
    with pytest.raises(CLIError, match="licence failure"):
        dispatch(
            _qualification_cli_args(
                pool_path=pool_path,
                instances_root=instances_root,
                output=output,
            ),
            hooks,
        )

    assert not model_called
    assert not output.with_name(f"{output.name}.journal").exists()
    assert not output.exists()


def test_checkpoint_publication_failure_removes_case_file_and_stops_ladder(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidates = tuple(
        _replacement_candidate(
            block_id="checkpoint-write",
            scale="small",
            replacement_rank=rank,
            expected_strata=1,
            replacement_count=4,
        )
        for rank in range(4)
    )
    pool = CandidatePool(
        candidates=candidates,
        candidate_seed_order=(11, 29, 47, 83),
        pool_label="checkpoint-write-pool",
    )
    pool_path = _write_pool(tmp_path / "pool.json", pool)
    instances_root = tmp_path / "instances"
    for candidate in candidates:
        instances_root.joinpath(candidate.instance_ref).mkdir(parents=True)
    output = tmp_path / "qualification.json"
    attempts: list[int] = []

    def normal_rejection(candidate: CandidateCase, _mode: str, **kwargs: Any):
        attempts.append(
            candidate.generator_parameters["pre_registered_selection"][
                "replacement_rank_within_stratum"
            ]
        )
        return _runner_outcome(
            candidate,
            accepted=False,
            qualification_run_id=kwargs["qualification_run_id"],
            policy=kwargs["policy"],
        )

    def publish_then_fail(path: str | Path, payload: Any) -> Path:
        published = write_canonical_json_exclusive(path, payload)
        if Path(path).name.startswith("case-"):
            raise OSError("synthetic post-publication sync failure")
        return published

    monkeypatch.setattr(
        "src.benchmark.branch_cut.cli.write_canonical_json_exclusive",
        publish_then_fail,
    )
    with pytest.raises(CLIError, match="checkpoint write failed"):
        dispatch(
            _qualification_cli_args(
                pool_path=pool_path,
                instances_root=instances_root,
                output=output,
            ),
            _qualification_hooks(normal_rejection),
        )

    assert attempts == [0]
    assert tuple(output.with_name(f"{output.name}.journal").glob("case-*.json")) == ()
    assert not output.exists()


def test_freeze_binds_qualification_mode_into_wrapper_identity(tmp_path: Path) -> None:
    candidate = _candidate()
    pool = _pool((candidate,))
    pool_path = _write_pool(tmp_path / "pool.json", pool)
    decision = _accepted_decision(candidate)
    qualification_path = write_canonical_json_exclusive(
        tmp_path / "qualification.json",
        {
            "source_pool_sha256": pool.sha256,
            "qualification_mode": "MX",
            "completed_at_utc": "2026-08-26T23:59:00Z",
            "model_identity": _MODEL_IDENTITY,
            "model_id": build_model_id(_MODEL_IDENTITY),
            "qualification_ledger": [decision.to_payload()],
        },
    )
    received: list[Any] = []

    class FakeManifest:
        sha256 = "b" * 64
        cases = (object(),)

        def artifact_payload(self) -> dict[str, Any]:
            return {"manifest_sha256": self.sha256, "cases": ["fake"]}

    def fake_freeze(pool_value: Any, decisions_value: Any, **kwargs: Any) -> Any:
        received.append((pool_value, tuple(decisions_value), kwargs))
        return FakeManifest()

    output = tmp_path / "manifest.json"
    args = build_parser().parse_args(
        [
            "freeze",
            "--pool",
            str(pool_path),
            "--qualification",
            str(qualification_path),
            "--output",
            str(output),
        ]
    )
    result = dispatch(
        args,
        CLIHooks(
            freeze=fake_freeze,
            model_identity=lambda: _MODEL_IDENTITY,
            now_utc=lambda: _NOW,
        ),
    )
    artifact = json.loads(output.read_text(encoding="utf-8"))

    assert result["qualification_mode"] == "MX"
    assert artifact["qualification_mode"] == "MX"
    expected_identity = canonical_sha256(
        {
            "manifest": artifact["manifest"],
            "qualification_mode": "MX",
            "model_identity": _MODEL_IDENTITY,
            "model_id": build_model_id(_MODEL_IDENTITY),
        }
    )
    assert artifact["benchmark_manifest_artifact_sha256"] == expected_identity
    assert received[0][2]["freeze_metadata"].manifest_frozen_at_utc == _NOW


def test_overcomplete_qualify_freeze_load_and_matched_run_round_trip(
    tmp_path: Path,
) -> None:
    candidates = tuple(
        _replacement_candidate(
            block_id=f"block-{block_index:02d}",
            scale=scale,
            replacement_rank=rank,
            expected_strata=18,
            replacement_count=3,
        )
        for scale in ("small", "medium", "large")
        for block_index in range(6)
        for rank in range(3)
    )
    pool = CandidatePool(
        candidates=candidates,
        candidate_seed_order=(11, 29, 47, 83),
        pool_label="mlnd-real-data-overcomplete-qualification-v5",
    )
    pool_path = _write_pool(tmp_path / "pool.json", pool)
    instances_root = tmp_path / "instances"
    for candidate in candidates:
        instances_root.joinpath(candidate.instance_ref).mkdir(parents=True)

    qualification_loads: list[CandidateCase] = []

    def fake_qualify(candidate: CandidateCase, _mode: str, **kwargs: Any):
        selection = candidate.generator_parameters["pre_registered_selection"]
        rejected_first_replacement = (
            candidate.scale == "small"
            and candidate.demand_block_id == "block-00"
            and selection["replacement_rank_within_stratum"] == 0
        )
        return _runner_outcome(
            candidate,
            accepted=not rejected_first_replacement,
            qualification_run_id=kwargs["qualification_run_id"],
            policy=kwargs["policy"],
        )

    qualification_output = tmp_path / "qualification.json"
    common_hooks = CLIHooks(
        canonical_hash=lambda path: (
            hashlib.sha256(path.name.encode()).hexdigest(),
            {},
        ),
        load_instance=lambda _path, candidate: (
            qualification_loads.append(candidate) or candidate
        ),
        qualify_instance=fake_qualify,
        model_identity=lambda: _MODEL_IDENTITY,
        qualification_implementation_sha256=(
            lambda: _QUALIFICATION_IMPLEMENTATION_SHA256
        ),
        gurobi_version=lambda: _GUROBI_VERSION,
        now_utc=lambda: _NOW,
    )
    qualification_result = dispatch(
        build_parser().parse_args(
            [
                "qualify",
                "--pool",
                str(pool_path),
                "--instances-root",
                str(instances_root),
                "--output",
                str(qualification_output),
            ]
        ),
        common_hooks,
    )
    assert qualification_result["attempted_count"] == 37
    assert qualification_result["unattempted_count"] == 17
    assert len(qualification_loads) == 37

    manifest_path = tmp_path / "manifest.json"
    freeze_result = dispatch(
        build_parser().parse_args(
            [
                "freeze",
                "--pool",
                str(pool_path),
                "--qualification",
                str(qualification_output),
                "--output",
                str(manifest_path),
            ]
        ),
        common_hooks,
    )
    loaded = load_benchmark_manifest(manifest_path)
    assert freeze_result["case_count"] == 36
    assert loaded.manifest.replacement_selection_protocol is not None
    assert loaded.manifest.replacement_selection_protocol.target_cases_per_stratum == 2
    assert loaded.manifest.replacement_selection_protocol.expected_stratum_count == 18
    assert len(loaded.manifest.selection_ledger) == 54
    assert (
        sum(
            item.qualification_status == "not_attempted"
            for item in loaded.manifest.selection_ledger
        )
        == 17
    )
    rejected_entries = tuple(
        item
        for item in loaded.manifest.selection_ledger
        if item.qualification_status == "rejected"
    )
    assert len(rejected_entries) == 1
    assert rejected_entries[0].rejection_reasons
    raw_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))["manifest"]
    assert loaded.manifest.sha256 == raw_manifest["manifest_sha256"]

    run_loads: list[CandidateCase] = []

    def fake_execute(
        _instance: object,
        _pair: Any,
        _profile: Any,
        _telemetry: Any,
        claim: Any,
    ) -> ProfileExecutionResult:
        return ProfileExecutionResult(
            completed_without_exception=True,
            has_incumbent=False,
            outcome="completed_no_incumbent",
            attempt_id=claim.attempt_id,
            directory=claim.directory,
            run_record={},
            failure=None,
        )

    frozen_candidate = loaded.manifest.cases[0].candidate
    batch_path = tmp_path / "batch.json"
    run_result = dispatch(
        build_parser().parse_args(
            [
                "run",
                "--manifest",
                str(manifest_path),
                "--instances-root",
                str(instances_root),
                "--results-root",
                str(tmp_path / "runs"),
                "--batch-output",
                str(batch_path),
                "--case-ids",
                frozen_candidate.case_id,
                "--profiles",
                "A",
                "E",
                "--time-budgets",
                "5",
            ]
        ),
        CLIHooks(
            canonical_hash=common_hooks.canonical_hash,
            load_instance=lambda _path, candidate: (
                run_loads.append(candidate) or candidate
            ),
            execute=fake_execute,
            model_identity=lambda: _MODEL_IDENTITY,
            now_utc=lambda: _NOW,
        ),
    )
    assert run_result["task_count"] == 2
    assert run_loads == [frozen_candidate, frozen_candidate]
    expected_profile = frozen_candidate.generator_parameters[
        "qualification_solve_profile"
    ]
    batch = json.loads(batch_path.read_text(encoding="utf-8"))
    model_contract = cli_module._candidate_model_contract(frozen_candidate)
    assert {item["qualification_case_sha256"] for item in batch["results"]} == {
        model_contract.qualification_case_sha256
    }
    assert {item["effective_model_config_sha256"] for item in batch["results"]} == {
        model_contract.effective_model_config_sha256
    }
    assert {
        (
            item["qualification_solve_profile"]["demand_scale"],
            item["qualification_solve_profile"]["feeder_rider_cost_factor"],
        )
        for item in batch["results"]
    } == {
        (
            expected_profile["demand_scale"],
            expected_profile["feeder_rider_cost_factor"],
        )
    }


def test_run_schedules_matched_profiles_without_gurobi(tmp_path: Path) -> None:
    manifest_path, manifest = _formal_manifest(tmp_path)
    case = manifest.cases[0].candidate
    seen: list[tuple[Any, Any, Any]] = []
    loaded_candidates: list[CandidateCase] = []

    def fake_execute(
        instance: object,
        pair: Any,
        profile: Any,
        telemetry: Any,
        claim: Any,
    ) -> ProfileExecutionResult:
        seen.append((pair, profile, telemetry))
        return ProfileExecutionResult(
            completed_without_exception=True,
            has_incumbent=False,
            outcome="completed_no_incumbent",
            attempt_id=claim.attempt_id,
            directory=claim.directory,
            run_record={},
            failure=None,
        )

    def fake_load(path: Path, candidate: CandidateCase) -> dict[str, Any]:
        loaded_candidates.append(candidate)
        return {"path": path, "candidate": candidate}

    hooks = CLIHooks(
        canonical_hash=lambda path: (
            hashlib.sha256(path.name.encode()).hexdigest(),
            {},
        ),
        load_instance=fake_load,
        execute=fake_execute,
        model_identity=lambda: _MODEL_IDENTITY,
        now_utc=lambda: _NOW,
    )
    batch_output = tmp_path / "batch.json"
    args = build_parser().parse_args(
        [
            "run",
            "--manifest",
            str(manifest_path),
            "--instances-root",
            str(tmp_path / "instances"),
            "--results-root",
            str(tmp_path / "runs"),
            "--batch-output",
            str(batch_output),
            "--case-ids",
            case.case_id,
            "--profiles",
            "A",
            "B",
            "C",
            "D",
            "E",
            "--time-budgets",
            "15",
        ]
    )
    result = dispatch(args, hooks)

    assert result["task_count"] == 5
    assert len(seen) == 5
    assert loaded_candidates == [case] * 5
    assert {
        (
            item.generator_parameters["qualification_solve_profile"]["demand_scale"],
            item.generator_parameters["qualification_solve_profile"][
                "feeder_rider_cost_factor"
            ],
        )
        for item in loaded_candidates
    } == {(3.0, 0.35)}
    assert all(pair.solver.threads == 1 for pair, _profile, _telemetry in seen)
    assert all(pair.solver.method == -1 for pair, _profile, _telemetry in seen)
    assert all(pair.solver.miqcp_method == -1 for pair, _profile, _telemetry in seen)
    by_code = {profile.code.value: profile for _pair, profile, _telemetry in seen}
    assert not by_code["A"].static.enabled
    assert not by_code["A"].user_cuts.enabled
    assert by_code["A"].start.mode.value == "none"
    for code in ("A", "B", "C"):
        assert by_code[code].start.fixed_integer_repair_seconds == 0.0
        assert not by_code[code].start.local_branching
    assert by_code["E"].static.enabled
    assert by_code["E"].user_cuts.enabled
    assert by_code["E"].start.mode.value == "network_and_courier"
    for code in ("D", "E"):
        assert by_code[code].start.fixed_integer_repair_seconds == 10.0
        assert by_code[code].start.network_start_method == 2
        assert not by_code[code].start.local_branching
        assert by_code[code].start.local_branching_seconds == 0.0
    serialized_profile = json.loads(canonical_json(by_code["E"]))
    assert serialized_profile["start"]["fixed_integer_repair_seconds"] == 10.0
    assert serialized_profile["start"]["network_start_method"] == 2
    assert serialized_profile["start"]["local_branching"] is False
    assert serialized_profile["start"]["local_branching_seconds"] == 0.0
    batch = json.loads(batch_output.read_text(encoding="utf-8"))
    assert batch["qualification_mode"] == "MX"
    assert not batch["exploratory_unqualified_modes"]


def _solver_suite_args(*extra: str) -> Any:
    return build_parser().parse_args(
        [
            "run",
            "--manifest",
            "manifest.json",
            "--instances-root",
            "instances",
            "--results-root",
            "runs",
            "--batch-output",
            "batch.json",
            *extra,
        ]
    )


def test_run_rejects_candidate_model_identity_before_claim(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = _candidate("small-run-identity-tamper")
    tampered_identity = {
        **candidate.generator_parameters["qualification_model_identity"],
        "qualification_case_sha256": "0" * 64,
    }
    candidate = replace(
        candidate,
        generator_parameters={
            **candidate.generator_parameters,
            "qualification_model_identity": tampered_identity,
        },
    )
    suite = _solver_suite(_solver_suite_args())
    pair = PairSpec(
        manifest_id="manifest-test",
        case_id=candidate.case_id,
        mode=BenchmarkMode.MX,
        solver_seed=42,
        time_budget_seconds=15.0,
        solver=suite.solver,
    )
    claimed = False

    def should_not_claim(*_args: Any, **_kwargs: Any) -> None:
        nonlocal claimed
        claimed = True
        raise AssertionError("invalid model identity must fail before claim")

    monkeypatch.setattr(cli_module, "claim_attempt_directory", should_not_claim)
    result = cli_module._run_one(
        cli_module.RunTask(
            candidate=candidate,
            instance_dir=tmp_path,
            model_identity=_MODEL_IDENTITY,
            results_root=tmp_path / "results",
            pair=pair,
            profile=suite.profile(cli_module.ProfileCode.A),
            telemetry=suite.telemetry,
            attempt_number=None,
        ),
        CLIHooks(model_identity=lambda: _MODEL_IDENTITY),
    )

    assert result["outcome"] == "orchestration_failed"
    assert "inconsistent qualification_model_identity" in result["orchestration_error"]
    assert result["qualification_case_sha256"] is None
    assert result["effective_model_config_sha256"] is None
    assert not claimed


def test_run_cli_allows_local_branching_when_repair_is_explicitly_disabled() -> None:
    suite = _solver_suite(
        _solver_suite_args(
            "--fixed-integer-repair-seconds",
            "0",
            "--local-branching-seconds",
            "7",
        )
    )

    for code in ("A", "B", "C"):
        assert suite.profile(code).start.fixed_integer_repair_seconds == 0.0
        assert not suite.profile(code).start.local_branching
    for code in ("D", "E"):
        start = suite.profile(code).start
        assert start.fixed_integer_repair_seconds == 0.0
        assert start.local_branching
        assert start.local_branching_seconds == 7.0


def test_run_cli_separates_matched_main_and_network_start_methods() -> None:
    suite = _solver_suite(
        _solver_suite_args(
            "--method",
            "-1",
            "--miqcp-method",
            "1",
            "--network-start-method",
            "2",
        )
    )

    assert suite.solver.method == -1
    assert suite.solver.miqcp_method == 1
    for code in ("A", "B", "C"):
        assert suite.profile(code).start.mode is StartMode.NONE
    for code in ("D", "E"):
        assert suite.profile(code).start.network_start_method == 2


def test_run_cli_rejects_competing_repair_and_local_branching_budgets() -> None:
    args = _solver_suite_args(
        "--fixed-integer-repair-seconds",
        "5",
        "--local-branching-seconds",
        "7",
    )

    with pytest.raises(CLIError, match="cannot both have positive budgets"):
        _solver_suite(args)


def _write_observed_attempt(
    root: Path,
    *,
    profile_code: str,
    manifest_id: str,
    case_id: str,
    gap: float,
    nodes: float,
    first_incumbent: float,
) -> None:
    pair = PairSpec(
        manifest_id=manifest_id,
        case_id=case_id,
        mode=BenchmarkMode.MX,
        solver_seed=42,
        time_budget_seconds=60.0,
        solver=SolverConfig(),
    )
    pair_id = build_pair_id(pair)
    directory = root / pair_id / profile_code
    directory.mkdir(parents=True)
    write_canonical_json_exclusive(
        directory / "run_spec.json",
        {
            "pair": json.loads(canonical_json(pair)),
        },
    )
    write_canonical_json_exclusive(
        directory / "run_record.json",
        {
            "ids": {"pair_id": pair_id, "attempt_number": 1},
            "profile_code": profile_code,
            "solver_metrics": {
                "status_name": "TIME_LIMIT",
                "has_incumbent": True,
                "relative_gap": gap,
                "total_algorithm_runtime_seconds": 60.0,
                "node_count": nodes,
                "first_incumbent_time_seconds": first_incumbent,
            },
        },
    )


def test_runtime_effect_censoring_is_explicit_and_other_statuses_fail_closed() -> None:
    def observed(status: str, *, has_incumbent: bool, runtime: float) -> Any:
        return cli_module._ObservedRun(
            pair_id="pair-test",
            profile_code="A",
            pair={},
            record={
                "solver_metrics": {
                    "status_name": status,
                    "has_incumbent": has_incumbent,
                    "total_algorithm_runtime_seconds": runtime,
                }
            },
            attempt_directory=None,
        )

    timeout = cli_module._runtime_effect_component(
        observed("TIME_LIMIT", has_incumbent=True, runtime=60.5),
        60.0,
        runtime_relative_tolerance=1e-8,
    )
    assert timeout == (60.0, True, "administratively_censored_time_limit")

    unsupported = cli_module._runtime_effect_component(
        observed("INTERRUPTED", has_incumbent=True, runtime=30.0),
        60.0,
        runtime_relative_tolerance=1e-8,
    )
    assert unsupported == (None, False, "unsupported_nonoptimal_status:INTERRUPTED")

    invalid_optimal = cli_module._runtime_effect_component(
        observed("OPTIMAL", has_incumbent=False, runtime=30.0),
        60.0,
        runtime_relative_tolerance=1e-8,
    )
    assert invalid_optimal == (None, False, "optimal_status_without_incumbent")


def test_sign_test_uses_one_median_contrast_per_complete_physical_case() -> None:
    seeds = (11, 29, 47, 83, 101)
    observations = tuple(
        cli_module.PairedCaseObservation(
            case_id=case_id,
            solver_seed=solver_seed,
            scale=scale,
            baseline=10.0 + solver_seed / 1000.0,
            candidate=9.0 + solver_seed / 1000.0,
        )
        for case_id, scale in (("small-1", "small"), ("large-1", "large"))
        for solver_seed in seeds
    )

    result, unavailable = cli_module._case_cluster_sign_test(
        observations,
        minimum_cases=2,
        expected_solver_seeds=seeds,
        alternative="greater",
        zero_tolerance=0.0,
    )

    assert unavailable is None
    assert result is not None
    assert result["case_count"] == 2
    assert result["pair_count"] == 10
    assert result["positive"] == 2
    assert result["effective_cases"] == 2
    assert result["p_value"] == pytest.approx(0.25)

    incomplete, reason = cli_module._case_cluster_sign_test(
        observations[:-1],
        minimum_cases=2,
        expected_solver_seeds=seeds,
        alternative="greater",
        zero_tolerance=0.0,
    )
    assert incomplete is None
    assert "expected solver-seed set" in reason


def test_analyze_reports_honest_time_limits_bootstrap_and_success_gate(
    tmp_path: Path,
) -> None:
    manifest_path, manifest = _formal_manifest(tmp_path)
    loaded = load_benchmark_manifest(manifest_path)
    cases = tuple(frozen.candidate for frozen in manifest.cases)
    results_root = tmp_path / "observed"
    for candidate in cases:
        _write_observed_attempt(
            results_root,
            profile_code="A",
            manifest_id=loaded.manifest_id,
            case_id=candidate.case_id,
            gap=0.20,
            nodes=100.0,
            first_incumbent=10.0,
        )
        _write_observed_attempt(
            results_root,
            profile_code="E",
            manifest_id=loaded.manifest_id,
            case_id=candidate.case_id,
            gap=0.10,
            nodes=50.0,
            first_incumbent=5.0,
        )
    output = tmp_path / "analysis.json"
    args = build_parser().parse_args(
        [
            "analyze",
            "--manifest",
            str(manifest_path),
            "--results-root",
            str(results_root),
            "--output",
            str(output),
            "--bootstrap-resamples",
            "100",
        ]
    )
    result = dispatch(args, CLIHooks(now_utc=lambda: _NOW))
    artifact = json.loads(output.read_text(encoding="utf-8"))

    assert result["pair_count"] == 36
    assert result["winner_counts"][PairWinner.CANDIDATE.value] == 36
    assert result["success_gate_passed"]
    assert result["evidence_scope"] == "exploratory"
    assert not result["paper_claim_eligible"]
    assert not result["paper_success_gate_passed"]
    assert artifact["complete_proven_runtime_pair_count"] == 0
    assert artifact["complete_gap_pair_count"] == 36
    assert artifact["gap_reduction"]["estimate"] == pytest.approx(0.5, rel=1e-4)
    assert artifact["gap_reduction"]["ci_lower"] > 0.0
    assert artifact["speedup"]["estimate"] == pytest.approx(1.0)
    assert artifact["speedup"]["case_count"] == 36
    assert artifact["speedup"]["pair_count"] == 36
    assert artifact["speedup"]["stratified_by_scale"] is True
    assert artifact["speedup"]["baseline_censored_count"] == 36
    assert artifact["speedup"]["candidate_censored_count"] == 36
    assert artifact["runtime_effect_input_pair_count"] == 36
    assert artifact["effect_denominator_audits"]["runtime"]["excluded_pair_count"] == 0
    assert artifact["primary_sign_test"] == {
        "alternative": "greater",
        "case_contrast": "median_baseline_minus_candidate_over_solver_seeds",
        "case_count": 36,
        "effective_cases": 36,
        "effective_pairs": 36,
        "negative": 0,
        "p_value": pytest.approx(2**-36),
        "pair_count": 36,
        "positive": 36,
        "sampling_unit": "physical_case_median_over_solver_seeds_v1",
        "ties": 0,
    }
    assert artifact["primary_sign_test_unavailable_reason"] is None
    assert artifact["median_summaries"]["gap"]["overall"] == {
        "baseline_median_of_case_seed_medians": pytest.approx(0.20),
        "candidate_median_of_case_seed_medians": pytest.approx(0.10),
        "candidate_minus_baseline_median_of_case_seed_medians": pytest.approx(-0.10),
        "case_count": 36,
    }
    assert artifact["pair_assessments"][0]["baseline_status"] == (
        "time_limit_with_incumbent"
    )
    assert artifact["performance_profile"] is None
    assert (
        artifact["performance_profile_denominator_audit"][
            "excluded_all_failed_problem_count"
        ]
        == 36
    )
    assert artifact["formal_coverage"] == {
        "complete": True,
        "expected_case_count": 36,
        "missing_case_ids": [],
        "observed_case_count": 36,
        "observed_cases_by_scale": {"large": 12, "medium": 12, "small": 12},
    }
    assert artifact["evidence_scope"] == "exploratory"
    assert not artifact["paper_claim_eligible"]
    assert not artifact["paper_success_gate"]["passed"]
    assert (
        "all benchmark problems failed"
        in (artifact["performance_profile_unavailable_reason"])
    )


def _authenticated_formal_analysis_snapshots(
    campaign: Any,
    *,
    failed_slot_ids: frozenset[str] = frozenset(),
    proven_runtime_speedup: bool = False,
    marker: str = "original",
) -> tuple[AuthenticatedFormalSlotSnapshot, ...]:
    snapshots: list[AuthenticatedFormalSlotSnapshot] = []
    required_files = {
        "claim.json",
        "cut_stats.json",
        "formal_claim.json",
        "progress.jsonl",
        "root_rounds.jsonl",
        "run_record.json",
        "run_spec.json",
        "solver.log",
    }
    for block in campaign.plan.blocks:
        for slot in block.slots:
            pair = PairSpec(
                manifest_id=campaign.plan.manifest_id,
                case_id=block.case_id,
                mode=campaign.plan.qualification_mode,
                solver_seed=block.solver_seed,
                time_budget_seconds=block.time_budget_seconds,
                solver=campaign.plan.suite.solver,
            )
            failed = slot.slot_id in failed_slot_ids
            gap = {"A": 0.20, "E": 0.10}.get(slot.profile_code.value, 0.15)
            runtime = block.time_budget_seconds
            if proven_runtime_speedup and slot.profile_code.value == "E":
                runtime /= 2.0
            run_spec = formal_execution_module._freeze_json_object(
                {"pair": json.loads(canonical_json(pair))}
            )
            run_record = formal_execution_module._freeze_json_object(
                {
                    "ids": {
                        "attempt_id": slot.attempt_id,
                        "attempt_number": 1,
                        "logical_run_id": slot.logical_run_id,
                        "pair_id": slot.pair_id,
                        "profile_id": slot.profile_id,
                    },
                    "formal_context": {"slot_id": slot.slot_id},
                    "profile_code": slot.profile_code.value,
                    "outcome": "failed" if failed else "completed_with_incumbent",
                    "completed_without_exception": not failed,
                    # Failed snapshots retain plausible metrics so the formal
                    # analyzer must exclude them based on authenticated outcome.
                    "solver_metrics": {
                        "status_name": (
                            "OPTIMAL" if proven_runtime_speedup else "TIME_LIMIT"
                        ),
                        "has_incumbent": True,
                        "relative_gap": gap,
                        "total_algorithm_runtime_seconds": runtime,
                        "node_count": 10.0,
                    },
                }
            )
            file_names = required_files | ({"failure.json"} if failed else set())
            file_sha256 = tuple(
                (
                    name,
                    hashlib.sha256(
                        f"{marker}:{slot.slot_id}:{name}".encode()
                    ).hexdigest(),
                )
                for name in sorted(file_names)
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
                    outcome="failed" if failed else "completed_with_incumbent",
                    completed_without_exception=not failed,
                    has_incumbent=True,
                    run_spec=run_spec,
                    run_record=run_record,
                    progress=(),
                    root_rounds=(),
                    cut_stats=MappingProxyType({}),
                    file_sha256=file_sha256,
                    attempt_bundle_sha256=bundle_sha256,
                    checkpoint_bytes_sha256=hashlib.sha256(
                        f"{marker}:checkpoint:{slot.slot_id}".encode()
                    ).hexdigest(),
                )
            )
    return tuple(snapshots)


def test_analyze_formal_requires_exact_slots_and_excludes_failed_metrics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path, _manifest, plan_path, frozen = _freeze_formal_plan_for_cli_test(
        tmp_path,
        analysis_args=("--analysis-minimum-gap-reduction", "0.75"),
    )
    plan_payload = json.loads(plan_path.read_text(encoding="utf-8"))["formal_plan"]
    expected_plan_id = str(frozen["formal_plan_id"])
    expected_evidence_root = "3" * 64
    expected_evidence_id = f"formal-evidence-{expected_evidence_root}"
    results_root = tmp_path / "formal-analysis-results"
    # analyze-formal refuses to manufacture a missing formal campaign.
    (results_root / expected_plan_id).mkdir(parents=True)
    prepared: dict[str, Any] = {"complete": False, "campaigns": []}
    monkeypatch.setattr(
        cli_module,
        "_load_observed_runs",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("formal analysis must not scan attempt paths")
        ),
    )

    def prepare_and_seed(
        plan: Any,
        root: Path,
        provider: Any,
        pre_claim_preparer: Any,
        *,
        session_snapshot: SessionSnapshot,
        resource_lock_path: str,
        now_utc: Any,
    ) -> Any:
        campaign = prepare_formal_campaign(
            plan,
            root,
            provider,
            pre_claim_preparer,
            session_snapshot=session_snapshot,
            resource_lock_path=resource_lock_path,
            now_utc=now_utc,
        )
        failed_slot_id = next(
            slot.slot_id
            for block in plan.blocks
            for slot in block.slots
            if slot.profile_code.value == "E"
        )
        snapshots = _authenticated_formal_analysis_snapshots(
            campaign,
            failed_slot_ids=frozenset({failed_slot_id}),
            proven_runtime_speedup=True,
        )
        prepared["campaigns"].append(campaign)
        prepared["snapshots"] = snapshots
        prepared["failed_slot_id"] = failed_slot_id
        return campaign

    def reconcile_as_complete(
        campaign: Any,
        *,
        expected_evidence_root_sha256: str,
    ) -> FormalCampaignState:
        assert expected_evidence_root_sha256 == expected_evidence_root
        if not prepared["complete"]:
            raise FormalExecutionError(
                "cannot match an expected formal evidence root with pending slots"
            )
        slot_ids = tuple(
            slot.slot_id for block in campaign.plan.blocks for slot in block.slots
        )
        snapshots = prepared["snapshots"]
        # Mutate on-disk paths after reconciliation captured immutable values.
        # Formal analysis must still use only the returned snapshots.
        first = snapshots[0]
        first_slot = campaign.plan.blocks[0].slots[0]
        attempt = (
            campaign.attempts_root / first_slot.logical_run_id / first_slot.attempt_id
        )
        attempt.mkdir(parents=True, exist_ok=True)
        (attempt / "run_record.json").write_text("{tampered", encoding="utf-8")
        (attempt / "progress.jsonl").write_text("{tampered", encoding="utf-8")
        assert first.run_record["solver_metrics"]["relative_gap"] == pytest.approx(0.20)
        return FormalCampaignState(
            slot_ids,
            (),
            (),
            authenticated_slots=snapshots,
            evidence_id=expected_evidence_id,
            evidence_root_sha256=expected_evidence_root,
        )

    def formal_args(output: Path) -> Any:
        return build_parser().parse_args(
            [
                "analyze-formal",
                "--manifest",
                str(manifest_path),
                "--plan",
                str(plan_path),
                "--expected-plan-id",
                expected_plan_id,
                "--expected-evidence-root",
                expected_evidence_root,
                "--results-root",
                str(results_root),
                "--output",
                str(output),
                "--resource-lock",
                str(tmp_path / "formal-analysis.lock"),
            ]
        )

    hooks = CLIHooks(
        prepare_formal=prepare_and_seed,
        reconcile_formal=reconcile_as_complete,
        model_identity=lambda: _MODEL_IDENTITY,
        formal_execution_implementation_sha256=(
            lambda: _FORMAL_EXECUTION_IMPLEMENTATION_SHA256
        ),
        gurobi_version=lambda: _GUROBI_VERSION,
        formal_environment_capture=_test_formal_environment_capture,
        preload_formal_runtime=lambda: None,
        now_utc=lambda: _NOW,
    )
    with pytest.raises(CLIError, match="pending slots"):
        dispatch(formal_args(tmp_path / "missing-analysis.json"), hooks)
    first_campaign = prepared["campaigns"][0]
    first_end = json.loads(first_campaign.session_end_path.read_text(encoding="utf-8"))
    assert first_end["status"] == "failed"

    prepared["complete"] = True
    output = tmp_path / "formal-analysis.json"
    result = dispatch(formal_args(output), hooks)
    artifact = json.loads(output.read_text(encoding="utf-8"))
    completed_campaign = prepared["campaigns"][1]
    completed_end = json.loads(
        completed_campaign.session_end_path.read_text(encoding="utf-8")
    )

    assert len(plan_payload["blocks"]) == 180
    assert result["command"] == "analyze-formal"
    assert result["pair_count"] == 180
    assert result["evidence_scope"] == "authenticated_formal_campaign"
    assert not result["paper_claim_eligible"]
    assert not result["paper_success_gate_passed"]
    assert not result["success_gate_passed"]
    assert artifact["formal_campaign_audit"]["authenticated_slot_count"] == 900
    assert completed_end["status"] == "completed"
    for field, expected in {
        "expected_evidence_id": expected_evidence_id,
        "expected_evidence_root_sha256": expected_evidence_root,
        "actual_evidence_id": expected_evidence_id,
        "actual_evidence_root_sha256": expected_evidence_root,
    }.items():
        assert result[field] == expected
        assert artifact[field] == expected
        assert artifact["formal_campaign_audit"][field] == expected
    assert artifact["analysis_parameters_source"] == "formal_plan.analysis_contract"
    assert artifact["formal_analysis_contract"]["bootstrap_resamples"] == 10
    assert artifact["formal_analysis_contract"]["minimum_gap_reduction"] == (
        pytest.approx(0.75)
    )
    assert (
        artifact["aggregate_gate_policy"]
        == (artifact["formal_analysis_contract"]["aggregate_gate_policy"])
    )
    assert artifact["gap_reduction"] is None
    assert "expected solver-seed set" in artifact["gap_reduction_unavailable_reason"]
    assert not artifact["success_gate"]["gap_gate_passed"]
    assert artifact["speedup"] is None
    assert not artifact["success_gate"]["speed_gate_passed"]
    assert not artifact["success_gate"]["conditions"][
        "substantive_aggregate_improvement"
    ]
    assert artifact["formal_campaign_audit"]["failed_slot_ids"] == [
        prepared["failed_slot_id"]
    ]
    assert artifact["complete_gap_pair_count"] == 179
    assert artifact["effect_denominator_audits"]["gap"] == {
        "complete_case_count": 35,
        "excluded_pair_count": 1,
        "exclusion_reasons": {
            "baseline=proven_optimal;candidate=formal_attempt_failed": 1
        },
        "expected_case_count": 36,
        "expected_pair_count": 180,
        "expected_solver_seeds": [11, 29, 47, 83, 101],
        "included_pair_count": 179,
        "represented_case_count": 36,
        "represented_cases_by_scale": {"large": 12, "medium": 12, "small": 12},
    }
    assert artifact["median_summaries"]["gap"] is None
    assert artifact["primary_sign_test"] is None
    assert (
        "expected solver-seed set" in (artifact["primary_sign_test_unavailable_reason"])
    )
    assert artifact["performance_profile_denominator_audit"]["failure_counts"]["E"] == 1
    assert not artifact["success_gate"]["conditions"]["formal_all_slots_succeeded"]


def test_analyze_formal_rejects_rehashed_metrics_and_checkpoint_against_anchor(
    tmp_path: Path,
) -> None:
    manifest_path, _manifest, plan_path, frozen = _freeze_formal_plan_for_cli_test(
        tmp_path
    )
    loaded = load_benchmark_manifest(manifest_path)
    plan = formal_plan_from_artifact(
        json.loads(plan_path.read_text(encoding="utf-8")),
        loaded.manifest,
    )
    results_root = tmp_path / "tampered-formal-results"
    environment = _test_formal_environment_capture()
    binding = FormalExecutionBinding(
        benchmark_manifest_sha256=plan.benchmark_manifest_sha256,
        manifest_artifact_sha256=plan.manifest_artifact_sha256,
        manifest_id=plan.manifest_id,
        model_id=plan.model_id,
        formal_execution_implementation_sha256=(
            plan.formal_execution_implementation_sha256
        ),
        qualification_mode=plan.qualification_mode,
        gurobi_version=plan.gurobi_version,
        environment_contract=environment.contract,
        host_identity_sha256=environment.host_identity_sha256,
    )

    def no_input(_task: FormalSlotTask) -> PreparedFormalInput:
        raise AssertionError("no pending formal slot may be prepared during analysis")

    campaign = prepare_formal_campaign(
        plan,
        results_root,
        lambda: binding,
        no_input,
        session_snapshot=environment.session,
        resource_lock_path=str((tmp_path / "tampered-formal.lock").resolve()),
        now_utc=lambda: _NOW,
    )
    original = _authenticated_formal_analysis_snapshots(campaign)
    _artifact, _evidence_id, original_root = (
        formal_execution_module._campaign_evidence_artifact(campaign, original)
    )

    first = original[0]
    changed_record = json.loads(canonical_json(first.run_record))
    changed_record["solver_metrics"]["relative_gap"] = 0.0
    frozen_changed_record = formal_execution_module._freeze_json_object(changed_record)
    changed_file_hashes = dict(first.file_sha256)
    changed_file_hashes["run_record.json"] = hashlib.sha256(
        (canonical_json(changed_record) + "\n").encode()
    ).hexdigest()
    changed_file_sha256 = tuple(sorted(changed_file_hashes.items()))
    first_slot = plan.blocks[0].slots[0]
    changed_bundle_sha256 = canonical_sha256(
        {
            "domain": FORMAL_SLOT_BUNDLE_HASH_DOMAIN,
            "slot_id": first_slot.slot_id,
            "slot_index": first_slot.slot_index,
            "attempt_id": first_slot.attempt_id,
            "file_sha256": dict(changed_file_sha256),
        }
    )
    synchronized_checkpoint = (
        canonical_json(
            {
                "bundle_file_sha256": dict(changed_file_sha256),
                "slot_id": first_slot.slot_id,
            }
        )
        + "\n"
    ).encode()
    tampered = (
        replace(
            first,
            run_record=frozen_changed_record,
            file_sha256=changed_file_sha256,
            attempt_bundle_sha256=changed_bundle_sha256,
            checkpoint_bytes_sha256=hashlib.sha256(synchronized_checkpoint).hexdigest(),
        ),
        *original[1:],
    )

    def reconcile_tampered(
        observed_campaign: Any,
        *,
        expected_evidence_root_sha256: str,
    ) -> FormalCampaignState:
        assert observed_campaign is campaign
        seal_formal_campaign_evidence(
            campaign,
            tampered,
            expected_evidence_root_sha256=expected_evidence_root_sha256,
        )
        raise AssertionError("mismatched external evidence root must fail closed")

    output = tmp_path / "tampered-analysis.json"
    args = build_parser().parse_args(
        [
            "analyze-formal",
            "--manifest",
            str(manifest_path),
            "--plan",
            str(plan_path),
            "--expected-plan-id",
            str(frozen["formal_plan_id"]),
            "--expected-evidence-root",
            original_root,
            "--results-root",
            str(results_root),
            "--output",
            str(output),
            "--resource-lock",
            str(tmp_path / "tampered-formal.lock"),
        ]
    )
    with pytest.raises(CLIError, match="externally expected root"):
        dispatch(
            args,
            CLIHooks(
                prepare_formal=lambda *_args, **_kwargs: campaign,
                reconcile_formal=reconcile_tampered,
                model_identity=lambda: _MODEL_IDENTITY,
                formal_execution_implementation_sha256=(
                    lambda: _FORMAL_EXECUTION_IMPLEMENTATION_SHA256
                ),
                gurobi_version=lambda: _GUROBI_VERSION,
                formal_environment_capture=_test_formal_environment_capture,
                preload_formal_runtime=lambda: None,
                now_utc=lambda: _NOW,
            ),
        )

    assert not output.exists()
    assert not campaign.evidence_seal_path.exists()
    session_end = json.loads(campaign.session_end_path.read_text(encoding="utf-8"))
    assert session_end["status"] == "failed"


def test_analyze_formal_default_reconciliation_rejects_empty_campaign(
    tmp_path: Path,
) -> None:
    manifest_path, _manifest, plan_path, frozen = _freeze_formal_plan_for_cli_test(
        tmp_path
    )
    expected_plan_id = str(frozen["formal_plan_id"])
    results_root = tmp_path / "empty-formal-results"
    (results_root / expected_plan_id).mkdir(parents=True)
    args = build_parser().parse_args(
        [
            "analyze-formal",
            "--manifest",
            str(manifest_path),
            "--plan",
            str(plan_path),
            "--expected-plan-id",
            expected_plan_id,
            "--expected-evidence-root",
            "0" * 64,
            "--results-root",
            str(results_root),
            "--output",
            str(tmp_path / "must-not-exist.json"),
            "--resource-lock",
            str(tmp_path / "empty-formal.lock"),
        ]
    )

    with pytest.raises(CLIError, match="pending slots"):
        dispatch(
            args,
            CLIHooks(
                model_identity=lambda: _MODEL_IDENTITY,
                formal_execution_implementation_sha256=(
                    lambda: _FORMAL_EXECUTION_IMPLEMENTATION_SHA256
                ),
                gurobi_version=lambda: _GUROBI_VERSION,
                formal_environment_capture=_test_formal_environment_capture,
                preload_formal_runtime=lambda: None,
            ),
        )
    assert not (tmp_path / "must-not-exist.json").exists()
    session_ends = tuple(results_root.rglob("sessions/*/end.json"))
    assert len(session_ends) == 1
    assert json.loads(session_ends[0].read_text(encoding="utf-8"))["status"] == (
        "failed"
    )


def test_run_rejects_mode_not_qualified_by_manifest(tmp_path: Path) -> None:
    manifest_path, _manifest = _formal_manifest(tmp_path)
    args = build_parser().parse_args(
        [
            "run",
            "--manifest",
            str(manifest_path),
            "--instances-root",
            str(tmp_path / "instances"),
            "--results-root",
            str(tmp_path / "runs"),
            "--batch-output",
            str(tmp_path / "batch.json"),
            "--modes",
            "LO",
        ]
    )
    with pytest.raises(CLIError, match="qualification mode"):
        dispatch(args, CLIHooks(model_identity=lambda: _MODEL_IDENTITY))


def test_invalid_pool_hash_is_rejected_before_solver_work(tmp_path: Path) -> None:
    pool = _pool((_candidate(),))
    path = write_canonical_json_exclusive(
        tmp_path / "pool.json",
        {"pool_sha256": "0" * 64, "pool": pool.to_payload()},
    )
    args = build_parser().parse_args(
        [
            "qualify",
            "--pool",
            str(path),
            "--instances-root",
            str(tmp_path),
            "--output",
            str(tmp_path / "qualification.json"),
        ]
    )
    with pytest.raises(CLIError, match="SHA256"):
        dispatch(args, CLIHooks())
