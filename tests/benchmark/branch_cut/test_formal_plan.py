from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import FrozenInstanceError, replace

import pytest

from src.benchmark.branch_cut.formal_environment import (
    FormalEnvironmentContract,
    GurobiVersions,
    ModuleDistributionVersions,
    OperatingSystemIdentity,
    PythonIdentity,
)
from src.benchmark.branch_cut.formal_plan import (
    FORMAL_AGGREGATE_GATE_POLICY,
    FORMAL_ATTEMPT_NUMBER,
    FORMAL_ANALYSIS_CONTRACT_SCHEMA_VERSION,
    FORMAL_BLOCK_COUNT,
    FORMAL_BOOTSTRAP_METHOD,
    FORMAL_BOOTSTRAP_SAMPLING_UNIT,
    FORMAL_BOOTSTRAP_STRATIFICATION_POLICY,
    FORMAL_CASE_COUNT,
    FORMAL_CASES_PER_SCALE,
    FORMAL_PLAN_SCHEMA_VERSION,
    FORMAL_RUNTIME_CENSORING_POLICY,
    FORMAL_PROFILE_ORDER,
    FORMAL_ROTATION_RULE,
    FORMAL_SCALE_CONTRACTS,
    FORMAL_SIGN_TEST_SAMPLING_UNIT,
    FORMAL_SLOT_COUNT,
    FORMAL_SOLVER_SEEDS,
    FormalAnalysisContract,
    FormalScaleContract,
    build_formal_execution_plan,
    formal_plan_from_artifact,
    latin_profile_order,
    validate_formal_execution_plan,
    validate_formal_plan_environment,
)
from src.benchmark.branch_cut.hashing import canonical_sha256
from src.benchmark.branch_cut.manifest import FreezeMetadata, freeze_manifest
from src.benchmark.branch_cut.pool import CandidateCase, CandidatePool
from src.benchmark.branch_cut.profiles import build_paper_profile_suite
from src.benchmark.branch_cut.qualification import (
    QualificationEvidence,
    SCALES,
    assess_qualification,
)
from src.benchmark.branch_cut.schema import BenchmarkMode, ProfileCode, SolverConfig


_MANIFEST_ARTIFACT_SHA256 = "a" * 64
_MODEL_ID = "model-" + "b" * 64
_FORMAL_EXECUTION_IMPLEMENTATION_SHA256 = "c" * 64
_GUROBI_VERSION = "13.0.1"


def _environment_contract(
    *,
    native_gurobi_version: str = _GUROBI_VERSION,
) -> FormalEnvironmentContract:
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
            module_version=native_gurobi_version,
            distribution_version=native_gurobi_version,
            native_version=native_gurobi_version,
        ),
    )


def _candidate(
    case_id: str,
    scale: str,
    index: int,
    *,
    instance_marker: str = "",
) -> CandidateCase:
    instance_ref = f"{case_id}{instance_marker}"
    return CandidateCase(
        case_id=case_id,
        scale=scale,
        demand_block_id=f"block-{index:02d}",
        layout_seed=11,
        instance_ref=instance_ref,
        instance_sha256=hashlib.sha256(instance_ref.encode()).hexdigest(),
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


def _manifest(
    *,
    cases_per_scale: tuple[int, int, int] = (12, 12, 12),
    case_prefix: str = "formal",
    instance_marker: str = "",
):
    candidates = tuple(
        _candidate(
            f"{case_prefix}-{scale}-{index:02d}",
            scale,
            index,
            instance_marker=instance_marker,
        )
        for scale, count in zip(SCALES, cases_per_scale, strict=True)
        for index in range(count)
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
    manifest=None,
    *,
    solver: SolverConfig | None = None,
    environment_contract: FormalEnvironmentContract | None = None,
    formal_execution_implementation_sha256: str = (
        _FORMAL_EXECUTION_IMPLEMENTATION_SHA256
    ),
    analysis_contract: FormalAnalysisContract | None = None,
):
    active_manifest = manifest or _manifest()
    suite = build_paper_profile_suite(solver=solver)
    return build_formal_execution_plan(
        active_manifest,
        manifest_artifact_sha256=_MANIFEST_ARTIFACT_SHA256,
        model_id=_MODEL_ID,
        formal_execution_implementation_sha256=(formal_execution_implementation_sha256),
        qualification_mode=BenchmarkMode.MX,
        gurobi_version=_GUROBI_VERSION,
        environment_contract=environment_contract or _environment_contract(),
        suite=suite,
        analysis_contract=analysis_contract,
    )


def test_formal_plan_has_exact_axes_counts_and_scale_budgets() -> None:
    manifest = _manifest()
    plan = _plan(manifest)
    slots = tuple(slot for block in plan.blocks for slot in block.slots)

    assert plan.expected_case_count == FORMAL_CASE_COUNT == 36
    assert plan.schema_version == FORMAL_PLAN_SCHEMA_VERSION == 6
    assert plan.expected_cases_per_scale == FORMAL_CASES_PER_SCALE == 12
    assert plan.expected_block_count == FORMAL_BLOCK_COUNT == 180
    assert plan.expected_slot_count == FORMAL_SLOT_COUNT == 900
    assert plan.solver_seeds == FORMAL_SOLVER_SEEDS == (11, 29, 47, 83, 101)
    assert plan.scale_contracts == FORMAL_SCALE_CONTRACTS
    assert [
        (item.scale, item.time_budget_seconds, item.max_concurrent_blocks)
        for item in plan.scale_contracts
    ] == [
        ("small", 300.0, 2),
        ("medium", 600.0, 1),
        ("large", 1800.0, 1),
    ]
    assert plan.scale_serial
    assert plan.within_block_serial
    assert plan.rotation_rule == FORMAL_ROTATION_RULE
    assert plan.attempt_number == FORMAL_ATTEMPT_NUMBER == 1
    assert plan.profile_base_order == FORMAL_PROFILE_ORDER
    assert (
        plan.formal_execution_implementation_sha256
        == _FORMAL_EXECUTION_IMPLEMENTATION_SHA256
    )
    assert plan.environment_contract == _environment_contract()
    assert plan.environment_contract.gurobipy.native_version == plan.gurobi_version
    first_case = manifest.cases[0]
    first_block = plan.blocks[0]
    assert first_block.qualification_run_id == (
        first_case.qualification.qualification_run_id
    )
    assert first_block.qualification_decision_sha256 == canonical_sha256(
        first_case.qualification.to_payload()
    )
    assert first_block.qualification_case_sha256 == (
        first_case.qualification.qualification_case_sha256
    )
    analysis = plan.analysis_contract
    assert analysis.schema_version == FORMAL_ANALYSIS_CONTRACT_SCHEMA_VERSION == 2
    assert analysis.baseline_profile is ProfileCode.A
    assert analysis.candidate_profile is ProfileCode.E
    assert analysis.primary_metric == "gap"
    assert analysis.aggregate_gate_policy == FORMAL_AGGREGATE_GATE_POLICY
    assert analysis.medium_large_tiers == ("medium", "large")
    assert analysis.minimum_medium_large_win_fraction == pytest.approx(0.60)
    assert analysis.minimum_improved_scale_tiers == 2
    assert analysis.minimum_runtime_speedup == pytest.approx(1.50)
    assert analysis.minimum_gap_reduction == pytest.approx(0.20)
    assert analysis.confidence_level == pytest.approx(0.95)
    assert analysis.bootstrap_seed == 20260826
    assert analysis.bootstrap_resamples == 10_000
    assert analysis.bootstrap_method == FORMAL_BOOTSTRAP_METHOD
    assert analysis.bootstrap_sampling_unit == FORMAL_BOOTSTRAP_SAMPLING_UNIT
    assert (
        analysis.bootstrap_stratification_policy
        == FORMAL_BOOTSTRAP_STRATIFICATION_POLICY
    )
    assert analysis.bootstrap_expected_solver_seeds == FORMAL_SOLVER_SEEDS
    assert analysis.bootstrap_minimum_cases_per_scale == 2
    assert analysis.minimum_effect_cases == 5
    assert analysis.sign_test_sampling_unit == FORMAL_SIGN_TEST_SAMPLING_UNIT
    assert analysis.runtime_censoring_policy == FORMAL_RUNTIME_CENSORING_POLICY
    assert analysis.runtime_shift == pytest.approx(1.0)
    assert analysis.gap_shift == pytest.approx(1e-6)
    assert analysis.require_complete_manifest_coverage
    assert analysis.require_all_formal_slots_succeeded

    assert Counter(block.scale for block in plan.blocks) == {
        "small": 60,
        "medium": 60,
        "large": 60,
    }
    assert Counter(slot.profile_code.value for slot in slots) == {
        code.value: 180 for code in FORMAL_PROFILE_ORDER
    }
    assert Counter(block.solver_seed for block in plan.blocks) == {
        seed: 36 for seed in FORMAL_SOLVER_SEEDS
    }
    assert Counter(block.time_budget_seconds for block in plan.blocks) == {
        300.0: 60,
        600.0: 60,
        1800.0: 60,
    }
    assert Counter(block.case_id for block in plan.blocks) == {
        case.candidate.case_id: 5 for case in manifest.cases
    }
    manifest_instances = {
        case.candidate.case_id: (
            case.candidate.instance_ref,
            case.candidate.instance_sha256,
        )
        for case in manifest.cases
    }
    assert all(
        (block.instance_ref, block.instance_sha256) == manifest_instances[block.case_id]
        for block in plan.blocks
    )
    assert all(slot.attempt_number == 1 for slot in slots)
    assert all(slot.attempt_id.endswith("-a0001") for slot in slots)
    assert len({slot.logical_run_id for slot in slots}) == 900


def test_latin_rotation_is_exact_and_position_balanced_per_scale() -> None:
    plan = _plan()

    assert latin_profile_order(0, 0) == tuple(FORMAL_PROFILE_ORDER)
    assert tuple(code.value for code in latin_profile_order(0, 1)) == tuple("BCDEA")
    assert tuple(code.value for code in latin_profile_order(1, 0)) == tuple("BCDEA")
    assert tuple(code.value for code in latin_profile_order(11, 4)) == tuple("ABCDE")

    for block in plan.blocks:
        assert (
            block.rotation_offset
            == (block.selection_rank_within_scale + block.seed_rank) % 5
        )
        assert tuple(slot.profile_code for slot in block.slots) == latin_profile_order(
            block.selection_rank_within_scale,
            block.seed_rank,
        )
        assert [slot.profile_position for slot in block.slots] == list(range(5))

    for scale in SCALES:
        position_counts = Counter(
            (slot.profile_code, slot.profile_position)
            for block in plan.blocks
            if block.scale == scale
            for slot in block.slots
        )
        assert position_counts == Counter(
            {
                (profile, position): 12
                for profile in FORMAL_PROFILE_ORDER
                for position in range(5)
            }
        )


def test_plan_is_deterministic_content_addressed_and_round_trips() -> None:
    manifest = _manifest()
    first = _plan(manifest)
    second = _plan(manifest)

    assert first == second
    assert first.sha256 == second.sha256
    assert first.plan_id == f"formal-plan-{first.sha256}"
    assert first.manifest_id == f"manifest-{_MANIFEST_ARTIFACT_SHA256}"
    artifact = first.artifact_payload()
    loaded = formal_plan_from_artifact(artifact, manifest)
    assert loaded == first
    assert loaded.artifact_payload() == artifact
    validate_formal_execution_plan(loaded, manifest)


def test_formal_plan_environment_binding_accepts_exact_external_identity() -> None:
    manifest = _manifest()
    plan = _plan(manifest)

    validate_formal_plan_environment(
        plan,
        manifest,
        manifest_artifact_sha256=_MANIFEST_ARTIFACT_SHA256,
        model_id=_MODEL_ID,
        formal_execution_implementation_sha256=(
            _FORMAL_EXECUTION_IMPLEMENTATION_SHA256
        ),
        qualification_mode=BenchmarkMode.MX,
        gurobi_version=_GUROBI_VERSION,
        environment_contract=_environment_contract(),
    )


@pytest.mark.parametrize(
    ("overrides", "mismatch"),
    (
        ({"manifest_artifact_sha256": "c" * 64}, "manifest_artifact_sha256"),
        ({"model_id": "model-" + "d" * 64}, "model_id"),
        (
            {"formal_execution_implementation_sha256": "d" * 64},
            "formal_execution_implementation_sha256",
        ),
        ({"qualification_mode": BenchmarkMode.HD}, "qualification_mode"),
        (
            {
                "gurobi_version": "13.0.2",
                "environment_contract": _environment_contract(
                    native_gurobi_version="13.0.2"
                ),
            },
            "gurobi_version, environment_contract",
        ),
        (
            {
                "environment_contract": replace(
                    _environment_contract(),
                    installed_ram_bytes=34_359_738_368,
                )
            },
            "environment_contract",
        ),
    ),
)
def test_formal_plan_environment_binding_rejects_external_drift(
    overrides: dict[str, object],
    mismatch: str,
) -> None:
    manifest = _manifest()
    plan = _plan(manifest)
    environment: dict[str, object] = {
        "manifest_artifact_sha256": _MANIFEST_ARTIFACT_SHA256,
        "model_id": _MODEL_ID,
        "formal_execution_implementation_sha256": (
            _FORMAL_EXECUTION_IMPLEMENTATION_SHA256
        ),
        "qualification_mode": BenchmarkMode.MX,
        "gurobi_version": _GUROBI_VERSION,
        "environment_contract": _environment_contract(),
    }
    environment.update(overrides)

    with pytest.raises(ValueError, match=mismatch):
        validate_formal_plan_environment(plan, manifest, **environment)


def test_rehashed_self_consistent_plan_cannot_override_external_binding() -> None:
    manifest = _manifest()
    forged_environment = _environment_contract(native_gurobi_version="99.0.0")
    forged = replace(
        _plan(manifest),
        model_id="model-" + "e" * 64,
        gurobi_version="99.0.0-forged",
        environment_contract=replace(
            forged_environment,
            gurobipy=replace(
                forged_environment.gurobipy,
                native_version="99.0.0-forged",
            ),
        ),
    )
    loaded = formal_plan_from_artifact(forged.artifact_payload(), manifest)

    with pytest.raises(
        ValueError,
        match="model_id, gurobi_version, environment_contract",
    ):
        validate_formal_plan_environment(
            loaded,
            manifest,
            manifest_artifact_sha256=_MANIFEST_ARTIFACT_SHA256,
            model_id=_MODEL_ID,
            formal_execution_implementation_sha256=(
                _FORMAL_EXECUTION_IMPLEMENTATION_SHA256
            ),
            qualification_mode=BenchmarkMode.MX,
            gurobi_version=_GUROBI_VERSION,
            environment_contract=_environment_contract(),
        )


def test_plan_identity_changes_with_locked_solver_configuration() -> None:
    manifest = _manifest()
    baseline = _plan(manifest, solver=SolverConfig(threads=1, mip_focus=0))
    changed = _plan(manifest, solver=SolverConfig(threads=1, mip_focus=1))
    changed_cut_passes = _plan(
        manifest,
        solver=SolverConfig(threads=1, mip_focus=0, root_cut_passes=3),
    )

    assert baseline.sha256 != changed.sha256
    assert baseline.plan_id != changed.plan_id
    assert baseline.blocks[0].pair_id != changed.blocks[0].pair_id
    assert baseline.blocks[0].slots[0].logical_run_id != (
        changed.blocks[0].slots[0].logical_run_id
    )
    assert baseline.sha256 != changed_cut_passes.sha256
    assert baseline.blocks[0].pair_id != changed_cut_passes.blocks[0].pair_id


def test_plan_identity_changes_with_formal_execution_implementation() -> None:
    manifest = _manifest()
    baseline = _plan(manifest)
    changed = _plan(
        manifest,
        formal_execution_implementation_sha256="d" * 64,
    )

    assert baseline.sha256 != changed.sha256
    assert baseline.plan_id != changed.plan_id
    assert baseline.blocks == changed.blocks


def test_plan_identity_changes_with_shareable_environment_contract() -> None:
    manifest = _manifest()
    baseline_environment = _environment_contract()
    changed_environment = replace(
        baseline_environment,
        operating_system=replace(
            baseline_environment.operating_system,
            kernel_release="24.6.1",
        ),
    )

    baseline = _plan(manifest, environment_contract=baseline_environment)
    changed = _plan(manifest, environment_contract=changed_environment)

    assert baseline.sha256 != changed.sha256
    assert baseline.plan_id != changed.plan_id
    assert baseline.blocks == changed.blocks


def test_plan_identity_changes_with_preregistered_analysis_axes() -> None:
    manifest = _manifest()
    baseline = _plan(manifest)
    changed = _plan(
        manifest,
        analysis_contract=replace(
            FormalAnalysisContract(),
            primary_metric="normalized_pdi",
            bootstrap_seed=20260827,
            minimum_gap_reduction=0.25,
        ),
    )

    assert baseline.analysis_contract != changed.analysis_contract
    assert baseline.sha256 != changed.sha256
    assert baseline.plan_id != changed.plan_id
    assert baseline.blocks == changed.blocks


def test_formal_analysis_contract_is_immutable_and_fail_closed() -> None:
    contract = FormalAnalysisContract()
    with pytest.raises(FrozenInstanceError):
        contract.bootstrap_seed = 1  # type: ignore[misc]
    with pytest.raises(ValueError, match="baseline A with candidate E"):
        replace(contract, candidate_profile=ProfileCode.D)
    with pytest.raises(ValueError, match="aggregate_gate_policy"):
        replace(contract, aggregate_gate_policy="primary_metric_only_v1")
    with pytest.raises(ValueError, match="one minus confidence"):
        replace(contract, confidence_level=0.90)
    with pytest.raises(ValueError, match="five formal solver seeds"):
        replace(contract, bootstrap_expected_solver_seeds=(11, 29, 47))
    with pytest.raises(ValueError, match="bootstrap_sampling_unit"):
        replace(contract, bootstrap_sampling_unit="solver_seed")


def test_plan_rejects_environment_native_gurobi_version_mismatch() -> None:
    with pytest.raises(ValueError, match="native Gurobi version"):
        _plan(
            environment_contract=_environment_contract(native_gurobi_version="13.0.2")
        )


def test_formal_execution_implementation_digest_must_be_canonical() -> None:
    with pytest.raises(
        ValueError,
        match="formal_execution_implementation_sha256",
    ):
        _plan(formal_execution_implementation_sha256="not-a-sha256")


def test_formal_builder_rejects_more_than_exactly_36_cases() -> None:
    overfilled = _manifest(cases_per_scale=(13, 12, 12))

    with pytest.raises(ValueError, match="exactly 36"):
        _plan(overfilled)


def test_formal_manifest_pipeline_rejects_fewer_than_36_cases() -> None:
    with pytest.raises(ValueError, match="requires at least 36"):
        _manifest(cases_per_scale=(11, 12, 12))


def test_formal_manifest_pipeline_rejects_scale_imbalance() -> None:
    with pytest.raises(ValueError, match="minimum number of cases"):
        _manifest(cases_per_scale=(13, 11, 12))


def test_formal_builder_rejects_multithreaded_solver_suite() -> None:
    with pytest.raises(ValueError, match="threads=1"):
        _plan(solver=SolverConfig(threads=2))


def test_fixed_seeds_budgets_concurrency_and_attempt_cannot_drift() -> None:
    plan = _plan()

    with pytest.raises(ValueError, match="solver seeds"):
        replace(plan, solver_seeds=(11, 29, 47, 83, 102))
    changed_scale_contracts = (
        FormalScaleContract("small", 300.0, 3),
        *FORMAL_SCALE_CONTRACTS[1:],
    )
    with pytest.raises(ValueError, match="budgets/concurrency"):
        replace(plan, scale_contracts=changed_scale_contracts)
    with pytest.raises(ValueError, match="attempt_number=1"):
        replace(plan, attempt_number=2)
    with pytest.raises(ValueError, match="serially"):
        replace(plan, scale_serial=False)
    with pytest.raises(ValueError, match="base order"):
        replace(plan, profile_base_order=tuple(reversed(FORMAL_PROFILE_ORDER)))


def test_artifact_hash_and_exact_fields_are_fail_closed() -> None:
    manifest = _manifest()
    artifact = _plan(manifest).artifact_payload()

    bad_hash = {**artifact, "formal_plan_sha256": "0" * 64}
    with pytest.raises(ValueError, match="formal_plan_sha256"):
        formal_plan_from_artifact(bad_hash, manifest)

    extra_field = {**artifact, "unexpected": True}
    with pytest.raises(ValueError, match="fields"):
        formal_plan_from_artifact(extra_field, manifest)

    boolean_schema = {**artifact, "artifact_schema_version": True}
    with pytest.raises(ValueError, match="artifact_schema_version"):
        formal_plan_from_artifact(boolean_schema, manifest)

    bad_plan_payload = dict(artifact["formal_plan"])
    bad_plan_payload["unexpected"] = True
    bad_plan_artifact = {**artifact, "formal_plan": bad_plan_payload}
    with pytest.raises(ValueError, match="fields"):
        formal_plan_from_artifact(bad_plan_artifact, manifest)

    analysis_payload = dict(artifact["formal_plan"]["analysis_contract"])
    analysis_payload["unexpected"] = True
    plan_with_bad_analysis = {
        **artifact["formal_plan"],
        "analysis_contract": analysis_payload,
    }
    with pytest.raises(ValueError, match="analysis_contract.*fields"):
        formal_plan_from_artifact(
            {**artifact, "formal_plan": plan_with_bad_analysis},
            manifest,
        )

    tampered_analysis = dict(artifact["formal_plan"]["analysis_contract"])
    tampered_analysis["bootstrap_seed"] += 1
    tampered_plan = {
        **artifact["formal_plan"],
        "analysis_contract": tampered_analysis,
    }
    with pytest.raises(ValueError, match="formal_plan_sha256"):
        formal_plan_from_artifact(
            {**artifact, "formal_plan": tampered_plan},
            manifest,
        )


def test_strict_reconstruction_rejects_valid_plan_for_different_cases() -> None:
    expected_manifest = _manifest(case_prefix="expected")
    other_manifest = _manifest(case_prefix="other")
    other_plan = _plan(other_manifest)
    forged_for_expected_manifest = replace(
        other_plan,
        benchmark_manifest_sha256=expected_manifest.sha256,
    )

    with pytest.raises(ValueError, match="instance identity differs"):
        formal_plan_from_artifact(
            forged_for_expected_manifest.artifact_payload(),
            expected_manifest,
        )


def test_strict_reconstruction_rejects_per_case_instance_identity_drift() -> None:
    expected_manifest = _manifest()
    drifted_manifest = _manifest(instance_marker="-different-instance")
    drifted_plan = _plan(drifted_manifest)
    forged_for_expected_manifest = replace(
        drifted_plan,
        benchmark_manifest_sha256=expected_manifest.sha256,
    )

    with pytest.raises(ValueError, match="instance identity differs"):
        formal_plan_from_artifact(
            forged_for_expected_manifest.artifact_payload(),
            expected_manifest,
        )


def test_nested_contracts_are_immutable_and_ids_detect_tampering() -> None:
    plan = _plan()
    first_block = plan.blocks[0]
    first_slot = first_block.slots[0]

    with pytest.raises(FrozenInstanceError):
        first_slot.attempt_number = 2  # type: ignore[misc]
    with pytest.raises(ValueError, match="slot_id"):
        replace(first_slot, slot_id="slot-" + "0" * 64)
    with pytest.raises(ValueError, match="block_id"):
        replace(first_block, block_id="block-" + "0" * 64)
    with pytest.raises(ValueError, match="block_id"):
        replace(first_block, instance_ref="different-instance.json")
    with pytest.raises(ValueError, match="instance_sha256"):
        replace(first_block, instance_sha256="not-a-sha256")
    with pytest.raises(ValueError, match="attempt_id"):
        replace(first_slot, attempt_id=first_slot.logical_run_id + "-a0002")
    with pytest.raises(ValueError, match="attempt_number"):
        replace(first_slot, attempt_number=True)


@pytest.mark.parametrize("seed_rank", (-1, 5))
def test_latin_rotation_rejects_invalid_seed_rank(seed_rank: int) -> None:
    with pytest.raises(ValueError, match="seed_rank"):
        latin_profile_order(0, seed_rank)
