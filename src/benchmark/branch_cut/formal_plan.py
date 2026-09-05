"""Immutable execution plan for the paper-grade branch-and-cut benchmark.

This module is deliberately solver-free.  It freezes the complete A--E
experiment matrix and its execution order before any formal comparison is
started.  The execution layer consumes these contracts later; it must not
reconstruct solver settings, budgets, seeds, or profile order from CLI defaults.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import Any

from .formal_environment import (
    FormalEnvironmentContract,
    FormalEnvironmentError,
    GurobiVersions,
    ModuleDistributionVersions,
    OperatingSystemIdentity,
    PythonIdentity,
)
from .hashing import (
    build_attempt_id,
    build_logical_run_id,
    build_pair_id,
    build_profile_id,
    canonical_json,
    canonical_sha256,
)
from .manifest import BenchmarkManifest, FrozenCase
from .qualification import SCALES
from .schema import (
    BenchmarkMode,
    MatchedProfileSuite,
    PairSpec,
    ProfileCode,
    ProfileConfig,
    SolverConfig,
    StartConfig,
    StaticStrengtheningConfig,
    TelemetryConfig,
    UserCutConfig,
)


FORMAL_PLAN_SCHEMA_VERSION = 6
FORMAL_PLAN_ARTIFACT_TYPE = "mlnd_branch_cut_formal_execution_plan"
FORMAL_PLAN_IDENTITY_SCHEMA = "mlnd-branch-cut-formal-plan-identity-v1"
FORMAL_SOLVER_SEEDS = (11, 29, 47, 83, 101)
FORMAL_PROFILE_ORDER = tuple(ProfileCode)
FORMAL_ATTEMPT_NUMBER = 1
FORMAL_CASE_COUNT = 36
FORMAL_CASES_PER_SCALE = 12
FORMAL_BLOCK_COUNT = 180
FORMAL_SLOT_COUNT = 900
FORMAL_ROTATION_RULE = (
    "latin_profile_rotation_v1:offset=(selection_rank_within_scale+seed_rank)%5"
)
FORMAL_ANALYSIS_CONTRACT_SCHEMA_VERSION = 2
FORMAL_AGGREGATE_GATE_POLICY = "runtime_or_gap_preregistered_or_v1"
FORMAL_AGGREGATE_SIGNIFICANCE_POLICY = (
    "point_threshold_and_lower_ci_strictly_excludes_neutral_v1"
)
FORMAL_BOOTSTRAP_METHOD = (
    "scale_stratified_physical_case_cluster_percentile_bootstrap_v2"
)
FORMAL_BOOTSTRAP_SAMPLING_UNIT = "physical_case_with_all_solver_seeds_v1"
FORMAL_BOOTSTRAP_STRATIFICATION_POLICY = "preserve_scale_case_counts_v1"
FORMAL_MEDIAN_SUMMARY_POLICY = "median_of_within_case_seed_medians_v1"
FORMAL_SIGN_TEST_SAMPLING_UNIT = "physical_case_median_over_solver_seeds_v1"
FORMAL_RUNTIME_EFFECT_POLICY = "shifted_geomean_administrative_horizon_time_to_proof_v1"
FORMAL_RUNTIME_CENSORING_POLICY = (
    "optimal_uses_observed_total_runtime_time_limit_uses_scale_budget_"
    "other_status_or_invalid_attempt_fails_closed_v1"
)
FORMAL_PERFORMANCE_PROFILE_POLICY = "dolan_more_proven_optimal_total_runtime_v1"
FORMAL_PERFORMANCE_PROFILE_DENOMINATOR_POLICY = (
    "all_case_seed_problems_nonoptimal_is_infinite_all_failed_excluded_v1"
)
FORMAL_TRAJECTORY_POLICY_VERSION = "matched_total_runtime_trajectory_v1"
FORMAL_PREPARATION_TIME_POLICY = (
    "total_minus_main_then_progress_then_prepare_components_v1"
)
FORMAL_COMMON_TARGET_POLICY = "worse_paired_final_incumbent_v1"
FORMAL_MISSING_PROGRESS_POLICY = "missing_or_invalid_is_failure_v1"
FORMAL_STATUS_RECONCILIATION_POLICY = (
    "progress_and_telemetry_must_match_when_both_known_v1"
)
FORMAL_HORIZON_POLICY = "matched_total_algorithm_budget_v1"
FORMAL_SYSTEMATIC_REGRESSION_POLICY = "inclusive_baseline_win_fraction_v1"
FORMAL_SUPPORTED_PRIMARY_METRICS = (
    "gap",
    "runtime",
    "time_to_target",
    "normalized_pdi",
)
FORMAL_SUPPORTED_KEY_REGRESSION_METRICS = (
    "node_count",
    "time_to_first_incumbent",
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_MANIFEST_ID_RE = re.compile(r"^manifest-[0-9a-f]{64}$")
_MODEL_ID_RE = re.compile(r"^model-[0-9a-f]{64}$")
_PAIR_ID_RE = re.compile(r"^pair-[0-9a-f]{64}$")
_PROFILE_ID_RE = re.compile(r"^profile-[0-9a-f]{64}$")
_RUN_ID_RE = re.compile(r"^run-[0-9a-f]{64}$")
_BLOCK_ID_RE = re.compile(r"^block-[0-9a-f]{64}$")
_SLOT_ID_RE = re.compile(r"^slot-[0-9a-f]{64}$")


def _require_nonempty(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a nonempty string")


def _require_sha256(value: str, field_name: str) -> None:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{field_name} must be a lowercase SHA256 digest")


def _require_int(value: int, field_name: str, *, minimum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{field_name} must be an integer >= {minimum}")


def _require_id(value: str, field_name: str, pattern: re.Pattern[str]) -> None:
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        raise ValueError(f"{field_name} is not a canonical identity")


def _identity(kind: str, payload: object) -> str:
    digest = canonical_sha256(
        {
            "identity_schema": FORMAL_PLAN_IDENTITY_SCHEMA,
            "kind": kind,
            "payload": payload,
        }
    )
    return f"{kind}-{digest}"


@dataclass(frozen=True, slots=True)
class FormalScaleContract:
    """One fixed scale budget and its maximum concurrent block count."""

    scale: str
    time_budget_seconds: float
    max_concurrent_blocks: int

    def __post_init__(self) -> None:
        if self.scale not in SCALES:
            raise ValueError(f"scale must be one of {SCALES}")
        if isinstance(self.time_budget_seconds, bool) or not isinstance(
            self.time_budget_seconds, (int, float)
        ):
            raise ValueError("time_budget_seconds must be numeric")
        budget = float(self.time_budget_seconds)
        if not math.isfinite(budget) or budget <= 0.0:
            raise ValueError("time_budget_seconds must be positive and finite")
        object.__setattr__(self, "time_budget_seconds", budget)
        _require_int(
            self.max_concurrent_blocks,
            "max_concurrent_blocks",
            minimum=1,
        )


FORMAL_SCALE_CONTRACTS = (
    FormalScaleContract("small", 300.0, 2),
    FormalScaleContract("medium", 600.0, 1),
    FormalScaleContract("large", 1800.0, 1),
)


def _finite_number(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field_name} must be finite")
    return number


@dataclass(frozen=True, slots=True)
class FormalAnalysisContract:
    """Immutable preregistration for every paper-claim analysis decision."""

    baseline_profile: ProfileCode = ProfileCode.A
    candidate_profile: ProfileCode = ProfileCode.E
    primary_metric: str = "gap"
    primary_direction: str = "lower"
    primary_absolute_tolerance: float = 0.0
    primary_relative_tolerance: float = 0.0
    primary_missing_is_failure: bool = True
    aggregate_gate_policy: str = FORMAL_AGGREGATE_GATE_POLICY
    medium_large_tiers: tuple[str, ...] = ("medium", "large")
    minimum_medium_large_win_fraction: float = 0.60
    minimum_improved_scale_tiers: int = 2
    minimum_improved_instances: int = 2
    minimum_runtime_speedup: float = 1.50
    minimum_gap_reduction: float = 0.20
    aggregate_significance_policy: str = FORMAL_AGGREGATE_SIGNIFICANCE_POLICY
    confidence_level: float = 0.95
    bootstrap_method: str = FORMAL_BOOTSTRAP_METHOD
    bootstrap_seed: int = 20260826
    bootstrap_resamples: int = 10_000
    bootstrap_sampling_unit: str = FORMAL_BOOTSTRAP_SAMPLING_UNIT
    bootstrap_stratification_policy: str = FORMAL_BOOTSTRAP_STRATIFICATION_POLICY
    bootstrap_expected_solver_seeds: tuple[int, ...] = FORMAL_SOLVER_SEEDS
    bootstrap_minimum_cases_per_scale: int = 2
    minimum_effect_cases: int = 5
    runtime_shift: float = 1.0
    gap_shift: float = 1e-6
    median_summary_policy: str = FORMAL_MEDIAN_SUMMARY_POLICY
    runtime_effect_policy: str = FORMAL_RUNTIME_EFFECT_POLICY
    runtime_censoring_policy: str = FORMAL_RUNTIME_CENSORING_POLICY
    performance_profile_policy: str = FORMAL_PERFORMANCE_PROFILE_POLICY
    performance_profile_denominator_policy: str = (
        FORMAL_PERFORMANCE_PROFILE_DENOMINATOR_POLICY
    )
    key_regression_metrics: tuple[str, ...] = (
        "node_count",
        "time_to_first_incumbent",
    )
    systematic_regression_policy: str = FORMAL_SYSTEMATIC_REGRESSION_POLICY
    regression_fraction: float = 0.60
    regression_minimum_pairs: int = 5
    sign_test_alternative: str = "greater"
    sign_test_sampling_unit: str = FORMAL_SIGN_TEST_SAMPLING_UNIT
    sign_test_zero_tolerance: float = 0.0
    sign_test_alpha: float = 0.05
    performance_taus: tuple[float, ...] = (1.0, 1.25, 1.5, 2.0, 4.0, 8.0)
    trajectory_policy_version: str = FORMAL_TRAJECTORY_POLICY_VERSION
    trajectory_sense: str = "min"
    preparation_time_policy: str = FORMAL_PREPARATION_TIME_POLICY
    runtime_consistency_relative_tolerance: float = 1e-8
    missing_gap_penalty: float = 1.0
    common_target_policy: str = FORMAL_COMMON_TARGET_POLICY
    target_absolute_tolerance: float = 1e-9
    target_relative_tolerance: float = 1e-9
    missing_progress_policy: str = FORMAL_MISSING_PROGRESS_POLICY
    status_reconciliation_policy: str = FORMAL_STATUS_RECONCILIATION_POLICY
    horizon_policy: str = FORMAL_HORIZON_POLICY
    require_complete_manifest_coverage: bool = True
    require_all_formal_slots_succeeded: bool = True
    schema_version: int = FORMAL_ANALYSIS_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        try:
            baseline = ProfileCode(self.baseline_profile)
            candidate = ProfileCode(self.candidate_profile)
        except (TypeError, ValueError) as error:
            raise ValueError("formal analysis profiles are invalid") from error
        object.__setattr__(self, "baseline_profile", baseline)
        object.__setattr__(self, "candidate_profile", candidate)
        if baseline is not ProfileCode.A or candidate is not ProfileCode.E:
            raise ValueError("formal analysis must compare baseline A with candidate E")
        if self.primary_metric not in FORMAL_SUPPORTED_PRIMARY_METRICS:
            raise ValueError("formal analysis primary_metric is unsupported")
        if self.primary_direction != "lower":
            raise ValueError("formal analysis primary_direction must be 'lower'")
        for name in (
            "primary_missing_is_failure",
            "require_complete_manifest_coverage",
            "require_all_formal_slots_succeeded",
        ):
            if getattr(self, name) is not True:
                raise ValueError(f"{name} must be true for formal claims")

        fixed_policies = {
            "aggregate_gate_policy": FORMAL_AGGREGATE_GATE_POLICY,
            "aggregate_significance_policy": FORMAL_AGGREGATE_SIGNIFICANCE_POLICY,
            "bootstrap_method": FORMAL_BOOTSTRAP_METHOD,
            "bootstrap_sampling_unit": FORMAL_BOOTSTRAP_SAMPLING_UNIT,
            "bootstrap_stratification_policy": FORMAL_BOOTSTRAP_STRATIFICATION_POLICY,
            "median_summary_policy": FORMAL_MEDIAN_SUMMARY_POLICY,
            "runtime_effect_policy": FORMAL_RUNTIME_EFFECT_POLICY,
            "runtime_censoring_policy": FORMAL_RUNTIME_CENSORING_POLICY,
            "performance_profile_policy": FORMAL_PERFORMANCE_PROFILE_POLICY,
            "performance_profile_denominator_policy": (
                FORMAL_PERFORMANCE_PROFILE_DENOMINATOR_POLICY
            ),
            "sign_test_sampling_unit": FORMAL_SIGN_TEST_SAMPLING_UNIT,
            "systematic_regression_policy": FORMAL_SYSTEMATIC_REGRESSION_POLICY,
            "trajectory_policy_version": FORMAL_TRAJECTORY_POLICY_VERSION,
            "preparation_time_policy": FORMAL_PREPARATION_TIME_POLICY,
            "common_target_policy": FORMAL_COMMON_TARGET_POLICY,
            "missing_progress_policy": FORMAL_MISSING_PROGRESS_POLICY,
            "status_reconciliation_policy": FORMAL_STATUS_RECONCILIATION_POLICY,
            "horizon_policy": FORMAL_HORIZON_POLICY,
        }
        for name, expected in fixed_policies.items():
            if getattr(self, name) != expected:
                raise ValueError(f"unsupported formal analysis {name}")
        if self.trajectory_sense != "min":
            raise ValueError("formal trajectory_sense must be 'min'")

        medium_large = tuple(str(tier).lower() for tier in self.medium_large_tiers)
        if medium_large != ("medium", "large"):
            raise ValueError("medium_large_tiers must be exactly medium,large")
        object.__setattr__(self, "medium_large_tiers", medium_large)
        key_metrics = tuple(str(name) for name in self.key_regression_metrics)
        if (
            not key_metrics
            or len(set(key_metrics)) != len(key_metrics)
            or any(
                name not in FORMAL_SUPPORTED_KEY_REGRESSION_METRICS
                for name in key_metrics
            )
        ):
            raise ValueError("key_regression_metrics are invalid")
        object.__setattr__(self, "key_regression_metrics", key_metrics)

        for name, minimum in (
            ("minimum_improved_scale_tiers", 1),
            ("minimum_improved_instances", 1),
            ("bootstrap_resamples", 1),
            ("bootstrap_minimum_cases_per_scale", 2),
            ("minimum_effect_cases", 2),
            ("regression_minimum_pairs", 1),
        ):
            _require_int(getattr(self, name), name, minimum=minimum)
        if self.minimum_improved_scale_tiers > len(SCALES):
            raise ValueError("minimum_improved_scale_tiers exceeds known scales")
        _require_int(self.bootstrap_seed, "bootstrap_seed", minimum=0)
        bootstrap_seeds = tuple(self.bootstrap_expected_solver_seeds)
        if bootstrap_seeds != FORMAL_SOLVER_SEEDS:
            raise ValueError(
                "bootstrap_expected_solver_seeds must equal the five formal solver seeds"
            )
        object.__setattr__(self, "bootstrap_expected_solver_seeds", bootstrap_seeds)
        _require_int(self.schema_version, "schema_version", minimum=1)
        if self.schema_version != FORMAL_ANALYSIS_CONTRACT_SCHEMA_VERSION:
            raise ValueError("unsupported formal-analysis contract schema_version")

        normalized_numbers: dict[str, float] = {}
        for name in (
            "primary_absolute_tolerance",
            "primary_relative_tolerance",
            "runtime_consistency_relative_tolerance",
            "missing_gap_penalty",
            "target_absolute_tolerance",
            "target_relative_tolerance",
            "sign_test_zero_tolerance",
        ):
            number = _finite_number(getattr(self, name), name)
            if number < 0.0:
                raise ValueError(f"{name} must be nonnegative")
            normalized_numbers[name] = number
        for name in ("runtime_shift", "gap_shift"):
            number = _finite_number(getattr(self, name), name)
            if number <= 0.0:
                raise ValueError(f"{name} must be positive")
            normalized_numbers[name] = number
        for name in (
            "minimum_medium_large_win_fraction",
            "minimum_gap_reduction",
            "confidence_level",
            "regression_fraction",
            "sign_test_alpha",
        ):
            number = _finite_number(getattr(self, name), name)
            if not 0.0 < number < 1.0:
                raise ValueError(f"{name} must lie strictly between zero and one")
            normalized_numbers[name] = number
        speedup = _finite_number(
            self.minimum_runtime_speedup, "minimum_runtime_speedup"
        )
        if speedup <= 1.0:
            raise ValueError("minimum_runtime_speedup must exceed one")
        normalized_numbers["minimum_runtime_speedup"] = speedup
        if not math.isclose(
            normalized_numbers["sign_test_alpha"],
            1.0 - normalized_numbers["confidence_level"],
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("sign_test_alpha must equal one minus confidence_level")
        for name, number in normalized_numbers.items():
            object.__setattr__(self, name, number)

        if self.sign_test_alternative not in {"greater", "less", "two-sided"}:
            raise ValueError("sign_test_alternative is unsupported")
        taus = tuple(
            _finite_number(value, f"performance_taus[{index}]")
            for index, value in enumerate(self.performance_taus)
        )
        if (
            not taus
            or any(value < 1.0 for value in taus)
            or tuple(sorted(set(taus))) != taus
            or taus[0] != 1.0
        ):
            raise ValueError(
                "performance_taus must be unique, increasing, and start at one"
            )
        object.__setattr__(self, "performance_taus", taus)

    def to_payload(self) -> dict[str, Any]:
        return json.loads(canonical_json(self))

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.to_payload())


def latin_profile_order(
    selection_rank_within_scale: int,
    seed_rank: int,
) -> tuple[ProfileCode, ...]:
    """Return the pre-registered cyclic A--E order for one matched block."""

    _require_int(
        selection_rank_within_scale,
        "selection_rank_within_scale",
        minimum=0,
    )
    _require_int(seed_rank, "seed_rank", minimum=0)
    if seed_rank >= len(FORMAL_SOLVER_SEEDS):
        raise ValueError("seed_rank is outside the formal solver-seed order")
    offset = (selection_rank_within_scale + seed_rank) % len(FORMAL_PROFILE_ORDER)
    return FORMAL_PROFILE_ORDER[offset:] + FORMAL_PROFILE_ORDER[:offset]


def _block_id_payload(
    *,
    block_index: int,
    case_id: str,
    instance_ref: str,
    instance_sha256: str,
    scale: str,
    selection_rank_within_scale: int,
    solver_seed: int,
    seed_rank: int,
    time_budget_seconds: float,
    rotation_offset: int,
    pair_id: str,
    qualification_run_id: str,
    qualification_decision_sha256: str,
    qualification_case_sha256: str | None,
    effective_model_config_sha256: str | None,
    qualification_screening_is_exact: bool | None,
) -> dict[str, object]:
    return {
        "block_index": block_index,
        "case_id": case_id,
        "instance_ref": instance_ref,
        "instance_sha256": instance_sha256,
        "pair_id": pair_id,
        "qualification_case_sha256": qualification_case_sha256,
        "qualification_decision_sha256": qualification_decision_sha256,
        "qualification_run_id": qualification_run_id,
        "effective_model_config_sha256": effective_model_config_sha256,
        "qualification_screening_is_exact": qualification_screening_is_exact,
        "rotation_offset": rotation_offset,
        "scale": scale,
        "seed_rank": seed_rank,
        "selection_rank_within_scale": selection_rank_within_scale,
        "solver_seed": solver_seed,
        "time_budget_seconds": time_budget_seconds,
    }


def _build_block_id(**values: object) -> str:
    return _identity("block", _block_id_payload(**values))


def _slot_id_payload(
    *,
    slot_index: int,
    block_index: int,
    block_id: str,
    pair_id: str,
    profile_position: int,
    profile_code: ProfileCode,
    profile_id: str,
    logical_run_id: str,
    attempt_number: int,
    attempt_id: str,
) -> dict[str, object]:
    return {
        "attempt_id": attempt_id,
        "attempt_number": attempt_number,
        "block_id": block_id,
        "block_index": block_index,
        "logical_run_id": logical_run_id,
        "pair_id": pair_id,
        "profile_code": profile_code.value,
        "profile_id": profile_id,
        "profile_position": profile_position,
        "slot_index": slot_index,
    }


def _build_slot_id(**values: object) -> str:
    return _identity("slot", _slot_id_payload(**values))


@dataclass(frozen=True, slots=True)
class FormalSlot:
    """One immutable profile attempt in a case-by-seed matched block."""

    slot_index: int
    slot_id: str
    block_index: int
    block_id: str
    pair_id: str
    profile_position: int
    profile_code: ProfileCode
    profile_id: str
    logical_run_id: str
    attempt_number: int
    attempt_id: str

    def __post_init__(self) -> None:
        for name in ("slot_index", "block_index", "profile_position"):
            _require_int(getattr(self, name), name, minimum=0)
        try:
            normalized_code = ProfileCode(self.profile_code)
        except (TypeError, ValueError) as error:
            raise ValueError(f"invalid profile_code: {self.profile_code!r}") from error
        object.__setattr__(self, "profile_code", normalized_code)
        _require_id(self.slot_id, "slot_id", _SLOT_ID_RE)
        _require_id(self.block_id, "block_id", _BLOCK_ID_RE)
        _require_id(self.pair_id, "pair_id", _PAIR_ID_RE)
        _require_id(self.profile_id, "profile_id", _PROFILE_ID_RE)
        _require_id(self.logical_run_id, "logical_run_id", _RUN_ID_RE)
        _require_int(self.attempt_number, "attempt_number", minimum=1)
        if self.attempt_number != FORMAL_ATTEMPT_NUMBER:
            raise ValueError("formal slots require attempt_number=1")
        expected_attempt_id = build_attempt_id(
            self.logical_run_id,
            FORMAL_ATTEMPT_NUMBER,
        )
        if self.attempt_id != expected_attempt_id:
            raise ValueError("attempt_id does not match logical_run_id and attempt 1")
        expected_slot_id = _build_slot_id(
            slot_index=self.slot_index,
            block_index=self.block_index,
            block_id=self.block_id,
            pair_id=self.pair_id,
            profile_position=self.profile_position,
            profile_code=self.profile_code,
            profile_id=self.profile_id,
            logical_run_id=self.logical_run_id,
            attempt_number=self.attempt_number,
            attempt_id=self.attempt_id,
        )
        if self.slot_id != expected_slot_id:
            raise ValueError("slot_id does not match the canonical slot payload")


@dataclass(frozen=True, slots=True)
class FormalBlock:
    """Five serial A--E attempts for one case and one solver seed."""

    block_index: int
    block_id: str
    case_id: str
    instance_ref: str
    instance_sha256: str
    scale: str
    selection_rank_within_scale: int
    solver_seed: int
    seed_rank: int
    time_budget_seconds: float
    rotation_offset: int
    pair_id: str
    qualification_run_id: str
    qualification_decision_sha256: str
    qualification_case_sha256: str | None
    effective_model_config_sha256: str | None
    qualification_screening_is_exact: bool | None
    slots: tuple[FormalSlot, ...]

    def __post_init__(self) -> None:
        for name in (
            "block_index",
            "selection_rank_within_scale",
            "solver_seed",
            "seed_rank",
            "rotation_offset",
        ):
            _require_int(getattr(self, name), name, minimum=0)
        _require_nonempty(self.case_id, "case_id")
        _require_nonempty(self.instance_ref, "instance_ref")
        _require_sha256(self.instance_sha256, "instance_sha256")
        _require_nonempty(self.qualification_run_id, "qualification_run_id")
        _require_sha256(
            self.qualification_decision_sha256,
            "qualification_decision_sha256",
        )
        qualification_identity = (
            self.qualification_case_sha256,
            self.effective_model_config_sha256,
            self.qualification_screening_is_exact,
        )
        if any(value is None for value in qualification_identity) and not all(
            value is None for value in qualification_identity
        ):
            raise ValueError(
                "formal qualification model identity must be complete or absent"
            )
        if self.qualification_case_sha256 is not None:
            _require_sha256(
                self.qualification_case_sha256,
                "qualification_case_sha256",
            )
            _require_sha256(
                self.effective_model_config_sha256,
                "effective_model_config_sha256",
            )
            if type(self.qualification_screening_is_exact) is not bool:
                raise ValueError(
                    "qualification_screening_is_exact must be boolean"
                )
        if self.scale not in SCALES:
            raise ValueError(f"scale must be one of {SCALES}")
        if isinstance(self.time_budget_seconds, bool) or not isinstance(
            self.time_budget_seconds, (int, float)
        ):
            raise ValueError("time_budget_seconds must be numeric")
        budget = float(self.time_budget_seconds)
        if not math.isfinite(budget) or budget <= 0.0:
            raise ValueError("time_budget_seconds must be positive and finite")
        object.__setattr__(self, "time_budget_seconds", budget)
        object.__setattr__(self, "slots", tuple(self.slots))
        _require_id(self.block_id, "block_id", _BLOCK_ID_RE)
        _require_id(self.pair_id, "pair_id", _PAIR_ID_RE)
        if self.seed_rank >= len(FORMAL_SOLVER_SEEDS):
            raise ValueError("seed_rank is outside the formal solver-seed order")
        if self.rotation_offset >= len(FORMAL_PROFILE_ORDER):
            raise ValueError("rotation_offset must be in [0, 4]")
        expected_block_id = _build_block_id(
            block_index=self.block_index,
            case_id=self.case_id,
            instance_ref=self.instance_ref,
            instance_sha256=self.instance_sha256,
            scale=self.scale,
            selection_rank_within_scale=self.selection_rank_within_scale,
            solver_seed=self.solver_seed,
            seed_rank=self.seed_rank,
            time_budget_seconds=self.time_budget_seconds,
            rotation_offset=self.rotation_offset,
            pair_id=self.pair_id,
            qualification_run_id=self.qualification_run_id,
            qualification_decision_sha256=self.qualification_decision_sha256,
            qualification_case_sha256=self.qualification_case_sha256,
            effective_model_config_sha256=self.effective_model_config_sha256,
            qualification_screening_is_exact=(
                self.qualification_screening_is_exact
            ),
        )
        if self.block_id != expected_block_id:
            raise ValueError("block_id does not match the canonical block payload")
        if len(self.slots) != len(FORMAL_PROFILE_ORDER):
            raise ValueError("every formal block must contain exactly five slots")
        expected_codes = latin_profile_order(
            self.selection_rank_within_scale,
            self.seed_rank,
        )
        if self.rotation_offset != (
            self.selection_rank_within_scale + self.seed_rank
        ) % len(FORMAL_PROFILE_ORDER):
            raise ValueError("rotation_offset does not match the Latin rotation rule")
        for position, (slot, expected_code) in enumerate(
            zip(self.slots, expected_codes, strict=True)
        ):
            if not isinstance(slot, FormalSlot):
                raise ValueError("slots must contain only FormalSlot values")
            if (
                slot.block_index != self.block_index
                or slot.block_id != self.block_id
                or slot.pair_id != self.pair_id
                or slot.profile_position != position
                or slot.profile_code is not expected_code
            ):
                raise ValueError("slot order or block identity is inconsistent")


@dataclass(frozen=True, slots=True)
class FormalExecutionPlan:
    """Complete immutable contract for exactly 900 formal solver attempts."""

    benchmark_manifest_sha256: str
    manifest_artifact_sha256: str
    manifest_id: str
    model_id: str
    formal_execution_implementation_sha256: str
    qualification_mode: BenchmarkMode
    gurobi_version: str
    environment_contract: FormalEnvironmentContract
    analysis_contract: FormalAnalysisContract
    suite: MatchedProfileSuite
    solver_seeds: tuple[int, ...]
    scale_contracts: tuple[FormalScaleContract, ...]
    profile_base_order: tuple[ProfileCode, ...]
    scale_serial: bool
    within_block_serial: bool
    rotation_rule: str
    attempt_number: int
    expected_case_count: int
    expected_cases_per_scale: int
    expected_block_count: int
    expected_slot_count: int
    blocks: tuple[FormalBlock, ...]
    schema_version: int = FORMAL_PLAN_SCHEMA_VERSION

    def __post_init__(self) -> None:
        try:
            normalized_mode = BenchmarkMode(self.qualification_mode)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"invalid qualification_mode: {self.qualification_mode!r}"
            ) from error
        object.__setattr__(self, "qualification_mode", normalized_mode)
        object.__setattr__(self, "solver_seeds", tuple(self.solver_seeds))
        object.__setattr__(self, "scale_contracts", tuple(self.scale_contracts))
        try:
            normalized_order = tuple(
                ProfileCode(code) for code in self.profile_base_order
            )
        except (TypeError, ValueError) as error:
            raise ValueError("profile_base_order contains an invalid code") from error
        object.__setattr__(self, "profile_base_order", normalized_order)
        object.__setattr__(self, "blocks", tuple(self.blocks))

        _require_sha256(
            self.benchmark_manifest_sha256,
            "benchmark_manifest_sha256",
        )
        _require_sha256(self.manifest_artifact_sha256, "manifest_artifact_sha256")
        _require_id(self.manifest_id, "manifest_id", _MANIFEST_ID_RE)
        if self.manifest_id != f"manifest-{self.manifest_artifact_sha256}":
            raise ValueError("manifest_id must contain manifest_artifact_sha256")
        _require_id(self.model_id, "model_id", _MODEL_ID_RE)
        _require_sha256(
            self.formal_execution_implementation_sha256,
            "formal_execution_implementation_sha256",
        )
        _require_nonempty(self.gurobi_version, "gurobi_version")
        if not isinstance(self.environment_contract, FormalEnvironmentContract):
            raise ValueError("environment_contract must be a FormalEnvironmentContract")
        if self.environment_contract.gurobipy.native_version != self.gurobi_version:
            raise ValueError(
                "environment contract native Gurobi version must equal gurobi_version"
            )
        if not isinstance(self.analysis_contract, FormalAnalysisContract):
            raise ValueError("analysis_contract must be a FormalAnalysisContract")
        if not isinstance(self.suite, MatchedProfileSuite):
            raise ValueError("suite must be a MatchedProfileSuite")
        self.suite.validate()
        if self.suite.solver.threads != 1:
            raise ValueError("formal plan requires solver threads=1")
        if self.solver_seeds != FORMAL_SOLVER_SEEDS:
            raise ValueError("formal solver seeds must be exactly 11,29,47,83,101")
        if self.scale_contracts != FORMAL_SCALE_CONTRACTS:
            raise ValueError("formal scale budgets/concurrency contract has changed")
        if self.profile_base_order != FORMAL_PROFILE_ORDER:
            raise ValueError("formal profile base order must be exactly A,B,C,D,E")
        if type(self.scale_serial) is not bool or not self.scale_serial:
            raise ValueError("formal scales must execute serially")
        if type(self.within_block_serial) is not bool or not self.within_block_serial:
            raise ValueError("profiles within a formal block must execute serially")
        if self.rotation_rule != FORMAL_ROTATION_RULE:
            raise ValueError("formal rotation rule has changed")
        _require_int(self.attempt_number, "attempt_number", minimum=1)
        if self.attempt_number != FORMAL_ATTEMPT_NUMBER:
            raise ValueError("formal plan requires attempt_number=1")
        expected_counts = {
            "expected_case_count": FORMAL_CASE_COUNT,
            "expected_cases_per_scale": FORMAL_CASES_PER_SCALE,
            "expected_block_count": FORMAL_BLOCK_COUNT,
            "expected_slot_count": FORMAL_SLOT_COUNT,
        }
        for name, expected in expected_counts.items():
            _require_int(getattr(self, name), name, minimum=1)
            if getattr(self, name) != expected:
                raise ValueError(f"{name} must equal {expected}")
        _require_int(self.schema_version, "schema_version", minimum=1)
        if self.schema_version != FORMAL_PLAN_SCHEMA_VERSION:
            raise ValueError("unsupported formal-plan schema_version")
        self._validate_internal_coverage()

    def _validate_internal_coverage(self) -> None:
        if len(self.blocks) != FORMAL_BLOCK_COUNT:
            raise ValueError("formal plan must contain exactly 180 blocks")
        if any(not isinstance(block, FormalBlock) for block in self.blocks):
            raise ValueError("blocks must contain only FormalBlock values")
        if [block.block_index for block in self.blocks] != list(
            range(FORMAL_BLOCK_COUNT)
        ):
            raise ValueError("formal block indices must be contiguous from zero")
        if len({block.case_id for block in self.blocks}) != FORMAL_CASE_COUNT:
            raise ValueError(
                "formal plan must contain exactly 36 globally unique case_id values"
            )
        instance_identity_by_case: dict[str, set[tuple[str, str]]] = {}
        for block in self.blocks:
            instance_identity_by_case.setdefault(block.case_id, set()).add(
                (block.instance_ref, block.instance_sha256)
            )
        if any(
            len(identities) != 1 for identities in instance_identity_by_case.values()
        ):
            raise ValueError(
                "every formal case must use one exact instance_ref/instance_sha256"
            )

        contract_by_scale = {item.scale: item for item in self.scale_contracts}
        scale_rank = {scale: index for index, scale in enumerate(SCALES)}
        for block in self.blocks:
            expected_block_index = (
                scale_rank[block.scale] * 60
                + block.selection_rank_within_scale * len(FORMAL_SOLVER_SEEDS)
                + block.seed_rank
            )
            if block.block_index != expected_block_index:
                raise ValueError(
                    "formal blocks are not in canonical scale/case/seed order"
                )
            if block.selection_rank_within_scale >= FORMAL_CASES_PER_SCALE:
                raise ValueError("formal selection ranks must be in [0, 11]")
            if block.solver_seed != FORMAL_SOLVER_SEEDS[block.seed_rank]:
                raise ValueError("formal solver seed does not match its seed rank")
            if (
                block.time_budget_seconds
                != contract_by_scale[block.scale].time_budget_seconds
            ):
                raise ValueError("formal block budget does not match its scale")
            pair = PairSpec(
                manifest_id=self.manifest_id,
                case_id=block.case_id,
                mode=self.qualification_mode,
                solver_seed=block.solver_seed,
                time_budget_seconds=block.time_budget_seconds,
                solver=self.suite.solver,
            )
            if block.pair_id != build_pair_id(pair):
                raise ValueError("formal block pair_id does not match its pair axes")
            for slot in block.slots:
                profile = self.suite.profile(slot.profile_code)
                if slot.profile_id != build_profile_id(profile):
                    raise ValueError(
                        "formal slot profile_id does not match its profile"
                    )
                if slot.logical_run_id != build_logical_run_id(pair, profile):
                    raise ValueError(
                        "formal slot logical_run_id does not match its pair/profile"
                    )

        case_axes = {
            (block.scale, block.case_id, block.selection_rank_within_scale)
            for block in self.blocks
        }
        if len(case_axes) != FORMAL_CASE_COUNT:
            raise ValueError("formal plan must contain exactly 36 distinct cases")
        cases_by_scale = Counter(scale for scale, _case, _rank in case_axes)
        if cases_by_scale != Counter(
            {scale: FORMAL_CASES_PER_SCALE for scale in SCALES}
        ):
            raise ValueError("formal plan must contain exactly 12 cases per scale")
        blocks_by_case = Counter(
            (block.scale, block.case_id, block.selection_rank_within_scale)
            for block in self.blocks
        )
        if set(blocks_by_case.values()) != {len(FORMAL_SOLVER_SEEDS)}:
            raise ValueError("every formal case must contain exactly five seed blocks")
        for case_axis in case_axes:
            seeds = {
                block.solver_seed
                for block in self.blocks
                if (
                    block.scale,
                    block.case_id,
                    block.selection_rank_within_scale,
                )
                == case_axis
            }
            if seeds != set(FORMAL_SOLVER_SEEDS):
                raise ValueError("every formal case must use all five solver seeds")

        block_ids = [block.block_id for block in self.blocks]
        pair_ids = [block.pair_id for block in self.blocks]
        if len(set(block_ids)) != FORMAL_BLOCK_COUNT:
            raise ValueError("formal block IDs must be unique")
        if len(set(pair_ids)) != FORMAL_BLOCK_COUNT:
            raise ValueError("formal pair IDs must be unique")
        blocks_by_scale = Counter(block.scale for block in self.blocks)
        if blocks_by_scale != Counter({scale: 60 for scale in SCALES}):
            raise ValueError("formal plan must contain 60 blocks per scale")

        slots = tuple(slot for block in self.blocks for slot in block.slots)
        if len(slots) != FORMAL_SLOT_COUNT:
            raise ValueError("formal plan must contain exactly 900 slots")
        if [slot.slot_index for slot in slots] != list(range(FORMAL_SLOT_COUNT)):
            raise ValueError("formal slot indices must be contiguous from zero")
        for name, values in (
            ("slot IDs", (slot.slot_id for slot in slots)),
            ("logical run IDs", (slot.logical_run_id for slot in slots)),
            ("attempt IDs", (slot.attempt_id for slot in slots)),
        ):
            materialized = tuple(values)
            if len(set(materialized)) != FORMAL_SLOT_COUNT:
                raise ValueError(f"formal {name} must be unique")
        if Counter(slot.profile_code for slot in slots) != Counter(
            {code: 180 for code in FORMAL_PROFILE_ORDER}
        ):
            raise ValueError("every formal profile must occur exactly 180 times")

    def to_payload(self) -> dict[str, Any]:
        """Return the exact canonical JSON-ready logical plan payload."""

        return json.loads(canonical_json(self))

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.to_payload())

    @property
    def plan_id(self) -> str:
        return f"formal-plan-{self.sha256}"

    def artifact_payload(self) -> dict[str, Any]:
        return {
            "artifact_schema_version": FORMAL_PLAN_SCHEMA_VERSION,
            "artifact_type": FORMAL_PLAN_ARTIFACT_TYPE,
            "formal_plan": self.to_payload(),
            "formal_plan_id": self.plan_id,
            "formal_plan_sha256": self.sha256,
        }


def _validate_exact_manifest(manifest: BenchmarkManifest) -> None:
    if not isinstance(manifest, BenchmarkManifest):
        raise ValueError("manifest must be a BenchmarkManifest")
    if len(manifest.cases) != FORMAL_CASE_COUNT:
        raise ValueError("formal plan requires exactly 36 frozen cases")
    counts = Counter(case.candidate.scale for case in manifest.cases)
    if counts != Counter({scale: FORMAL_CASES_PER_SCALE for scale in SCALES}):
        raise ValueError("formal plan requires exactly 12 frozen cases per scale")
    for scale in SCALES:
        ranks = sorted(
            case.selection_rank_within_scale
            for case in manifest.cases
            if case.candidate.scale == scale
        )
        if ranks != list(range(FORMAL_CASES_PER_SCALE)):
            raise ValueError(
                f"formal selection ranks for scale {scale!r} must be 0..11"
            )


def _cases_in_formal_order(manifest: BenchmarkManifest) -> tuple[FrozenCase, ...]:
    return tuple(
        case
        for scale in SCALES
        for case in sorted(
            (item for item in manifest.cases if item.candidate.scale == scale),
            key=lambda item: item.selection_rank_within_scale,
        )
    )


def _expected_blocks(
    manifest: BenchmarkManifest,
    *,
    manifest_id: str,
    qualification_mode: BenchmarkMode,
    suite: MatchedProfileSuite,
) -> tuple[FormalBlock, ...]:
    scale_contract = {item.scale: item for item in FORMAL_SCALE_CONTRACTS}
    blocks: list[FormalBlock] = []
    slot_index = 0
    for frozen_case in _cases_in_formal_order(manifest):
        candidate = frozen_case.candidate
        qualification = frozen_case.qualification
        qualification_decision_sha256 = canonical_sha256(
            qualification.to_payload()
        )
        selection_rank = frozen_case.selection_rank_within_scale
        budget = scale_contract[candidate.scale].time_budget_seconds
        for seed_rank, solver_seed in enumerate(FORMAL_SOLVER_SEEDS):
            block_index = len(blocks)
            rotation_offset = (selection_rank + seed_rank) % len(FORMAL_PROFILE_ORDER)
            pair = PairSpec(
                manifest_id=manifest_id,
                case_id=candidate.case_id,
                mode=qualification_mode,
                solver_seed=solver_seed,
                time_budget_seconds=budget,
                solver=suite.solver,
            )
            pair_id = build_pair_id(pair)
            block_id = _build_block_id(
                block_index=block_index,
                case_id=candidate.case_id,
                instance_ref=candidate.instance_ref,
                instance_sha256=candidate.instance_sha256,
                scale=candidate.scale,
                selection_rank_within_scale=selection_rank,
                solver_seed=solver_seed,
                seed_rank=seed_rank,
                time_budget_seconds=budget,
                rotation_offset=rotation_offset,
                pair_id=pair_id,
                qualification_run_id=qualification.qualification_run_id,
                qualification_decision_sha256=qualification_decision_sha256,
                qualification_case_sha256=(
                    qualification.qualification_case_sha256
                ),
                effective_model_config_sha256=(
                    qualification.effective_model_config_sha256
                ),
                qualification_screening_is_exact=qualification.screening_is_exact,
            )
            slots: list[FormalSlot] = []
            for profile_position, profile_code in enumerate(
                latin_profile_order(selection_rank, seed_rank)
            ):
                profile = suite.profile(profile_code)
                profile_id = build_profile_id(profile)
                logical_run_id = build_logical_run_id(pair, profile)
                attempt_id = build_attempt_id(
                    logical_run_id,
                    FORMAL_ATTEMPT_NUMBER,
                )
                slot_id = _build_slot_id(
                    slot_index=slot_index,
                    block_index=block_index,
                    block_id=block_id,
                    pair_id=pair_id,
                    profile_position=profile_position,
                    profile_code=profile_code,
                    profile_id=profile_id,
                    logical_run_id=logical_run_id,
                    attempt_number=FORMAL_ATTEMPT_NUMBER,
                    attempt_id=attempt_id,
                )
                slots.append(
                    FormalSlot(
                        slot_index=slot_index,
                        slot_id=slot_id,
                        block_index=block_index,
                        block_id=block_id,
                        pair_id=pair_id,
                        profile_position=profile_position,
                        profile_code=profile_code,
                        profile_id=profile_id,
                        logical_run_id=logical_run_id,
                        attempt_number=FORMAL_ATTEMPT_NUMBER,
                        attempt_id=attempt_id,
                    )
                )
                slot_index += 1
            blocks.append(
                FormalBlock(
                    block_index=block_index,
                    block_id=block_id,
                    case_id=candidate.case_id,
                    instance_ref=candidate.instance_ref,
                    instance_sha256=candidate.instance_sha256,
                    scale=candidate.scale,
                    selection_rank_within_scale=selection_rank,
                    solver_seed=solver_seed,
                    seed_rank=seed_rank,
                    time_budget_seconds=budget,
                    rotation_offset=rotation_offset,
                    pair_id=pair_id,
                    qualification_run_id=qualification.qualification_run_id,
                    qualification_decision_sha256=(
                        qualification_decision_sha256
                    ),
                    qualification_case_sha256=(
                        qualification.qualification_case_sha256
                    ),
                    effective_model_config_sha256=(
                        qualification.effective_model_config_sha256
                    ),
                    qualification_screening_is_exact=(
                        qualification.screening_is_exact
                    ),
                    slots=tuple(slots),
                )
            )
    return tuple(blocks)


def build_formal_execution_plan(
    manifest: BenchmarkManifest,
    *,
    manifest_artifact_sha256: str,
    model_id: str,
    formal_execution_implementation_sha256: str,
    qualification_mode: BenchmarkMode | str,
    gurobi_version: str,
    environment_contract: FormalEnvironmentContract,
    suite: MatchedProfileSuite,
    analysis_contract: FormalAnalysisContract | None = None,
) -> FormalExecutionPlan:
    """Build the only valid 36-case, 180-block, 900-slot formal plan."""

    _validate_exact_manifest(manifest)
    _require_sha256(manifest_artifact_sha256, "manifest_artifact_sha256")
    _require_sha256(
        formal_execution_implementation_sha256,
        "formal_execution_implementation_sha256",
    )
    manifest_id = f"manifest-{manifest_artifact_sha256}"
    try:
        mode = BenchmarkMode(qualification_mode)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"invalid qualification_mode: {qualification_mode!r}"
        ) from error
    if not isinstance(suite, MatchedProfileSuite):
        raise ValueError("suite must be a MatchedProfileSuite")
    blocks = _expected_blocks(
        manifest,
        manifest_id=manifest_id,
        qualification_mode=mode,
        suite=suite,
    )
    plan = FormalExecutionPlan(
        benchmark_manifest_sha256=manifest.sha256,
        manifest_artifact_sha256=manifest_artifact_sha256,
        manifest_id=manifest_id,
        model_id=model_id,
        formal_execution_implementation_sha256=(formal_execution_implementation_sha256),
        qualification_mode=mode,
        gurobi_version=gurobi_version,
        environment_contract=environment_contract,
        analysis_contract=(
            FormalAnalysisContract() if analysis_contract is None else analysis_contract
        ),
        suite=suite,
        solver_seeds=FORMAL_SOLVER_SEEDS,
        scale_contracts=FORMAL_SCALE_CONTRACTS,
        profile_base_order=FORMAL_PROFILE_ORDER,
        scale_serial=True,
        within_block_serial=True,
        rotation_rule=FORMAL_ROTATION_RULE,
        attempt_number=FORMAL_ATTEMPT_NUMBER,
        expected_case_count=FORMAL_CASE_COUNT,
        expected_cases_per_scale=FORMAL_CASES_PER_SCALE,
        expected_block_count=FORMAL_BLOCK_COUNT,
        expected_slot_count=FORMAL_SLOT_COUNT,
        blocks=blocks,
    )
    validate_formal_execution_plan(plan, manifest)
    return plan


def validate_formal_execution_plan(
    plan: FormalExecutionPlan,
    manifest: BenchmarkManifest,
) -> None:
    """Strictly rebuild all blocks/slots and reject any stored-plan drift."""

    if not isinstance(plan, FormalExecutionPlan):
        raise ValueError("plan must be a FormalExecutionPlan")
    _validate_exact_manifest(manifest)
    if plan.benchmark_manifest_sha256 != manifest.sha256:
        raise ValueError("formal plan belongs to a different benchmark manifest")
    manifest_instance_by_case = {
        case.candidate.case_id: (
            case.candidate.instance_ref,
            case.candidate.instance_sha256,
        )
        for case in manifest.cases
    }
    for block in plan.blocks:
        if manifest_instance_by_case.get(block.case_id) != (
            block.instance_ref,
            block.instance_sha256,
        ):
            raise ValueError(
                "formal block instance identity differs from the benchmark "
                f"manifest for case {block.case_id!r}"
            )
    expected = _expected_blocks(
        manifest,
        manifest_id=plan.manifest_id,
        qualification_mode=plan.qualification_mode,
        suite=plan.suite,
    )
    if plan.blocks != expected:
        raise ValueError("formal blocks/slots differ from strict reconstruction")


def validate_formal_plan_environment(
    plan: FormalExecutionPlan,
    manifest: BenchmarkManifest,
    *,
    manifest_artifact_sha256: str,
    model_id: str,
    formal_execution_implementation_sha256: str,
    qualification_mode: BenchmarkMode | str,
    gurobi_version: str,
    environment_contract: FormalEnvironmentContract,
) -> None:
    """Bind a self-consistent plan to independently observed execution state.

    ``formal_plan_from_artifact`` proves that an artifact is internally
    canonical.  That alone cannot prove that identifiers stored inside the
    artifact describe the manifest wrapper, model implementation, or solver
    available to the process that is about to execute it.  Formal execution
    must call this function with values obtained independently from the loaded
    manifest wrapper and the current runtime before claiming any slot.
    """

    validate_formal_execution_plan(plan, manifest)
    _require_sha256(manifest_artifact_sha256, "manifest_artifact_sha256")
    _require_id(model_id, "model_id", _MODEL_ID_RE)
    _require_sha256(
        formal_execution_implementation_sha256,
        "formal_execution_implementation_sha256",
    )
    try:
        normalized_mode = BenchmarkMode(qualification_mode)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"invalid qualification_mode: {qualification_mode!r}"
        ) from error
    _require_nonempty(gurobi_version, "gurobi_version")
    if not isinstance(environment_contract, FormalEnvironmentContract):
        raise ValueError("environment_contract must be a FormalEnvironmentContract")
    if environment_contract.gurobipy.native_version != gurobi_version:
        raise ValueError(
            "observed environment native Gurobi version must equal gurobi_version"
        )

    expected_manifest_id = f"manifest-{manifest_artifact_sha256}"
    mismatches: list[str] = []
    if plan.manifest_artifact_sha256 != manifest_artifact_sha256:
        mismatches.append("manifest_artifact_sha256")
    if plan.manifest_id != expected_manifest_id:
        mismatches.append("manifest_id")
    if plan.model_id != model_id:
        mismatches.append("model_id")
    if (
        plan.formal_execution_implementation_sha256
        != formal_execution_implementation_sha256
    ):
        mismatches.append("formal_execution_implementation_sha256")
    if plan.qualification_mode is not normalized_mode:
        mismatches.append("qualification_mode")
    if plan.gurobi_version != gurobi_version:
        mismatches.append("gurobi_version")
    if plan.environment_contract != environment_contract:
        mismatches.append("environment_contract")
    if mismatches:
        raise ValueError(
            "formal plan does not match the independently observed execution "
            "environment: " + ", ".join(mismatches)
        )


def _as_mapping(value: object, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be an object")
    if any(not isinstance(key, str) for key in value):
        raise ValueError(f"{field_name} keys must be strings")
    return value


def _require_exact_fields(
    payload: Mapping[str, Any],
    contract: type[object],
    field_name: str,
) -> None:
    expected = {item.name for item in fields(contract)}
    actual = set(payload)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(
            f"{field_name} fields are inconsistent; missing={missing}, extra={extra}"
        )


def _dataclass_from_mapping(
    contract: type[Any],
    payload: object,
    field_name: str,
) -> Any:
    values = _as_mapping(payload, field_name)
    _require_exact_fields(values, contract, field_name)
    try:
        return contract(**dict(values))
    except (TypeError, ValueError, FormalEnvironmentError) as error:
        raise ValueError(f"invalid {field_name}: {error}") from error


def _environment_contract_from_payload(
    payload: object,
) -> FormalEnvironmentContract:
    field_name = "formal_plan.environment_contract"
    values = _as_mapping(payload, field_name)
    _require_exact_fields(values, FormalEnvironmentContract, field_name)
    contract_values = dict(values)
    contract_values["operating_system"] = _dataclass_from_mapping(
        OperatingSystemIdentity,
        values["operating_system"],
        f"{field_name}.operating_system",
    )
    contract_values["python"] = _dataclass_from_mapping(
        PythonIdentity,
        values["python"],
        f"{field_name}.python",
    )
    for package_name in ("numpy", "scipy"):
        contract_values[package_name] = _dataclass_from_mapping(
            ModuleDistributionVersions,
            values[package_name],
            f"{field_name}.{package_name}",
        )
    contract_values["gurobipy"] = _dataclass_from_mapping(
        GurobiVersions,
        values["gurobipy"],
        f"{field_name}.gurobipy",
    )
    try:
        return FormalEnvironmentContract(**contract_values)
    except (TypeError, ValueError, FormalEnvironmentError) as error:
        raise ValueError(f"invalid {field_name}: {error}") from error


def _analysis_contract_from_payload(payload: object) -> FormalAnalysisContract:
    field_name = "formal_plan.analysis_contract"
    values = _as_mapping(payload, field_name)
    _require_exact_fields(values, FormalAnalysisContract, field_name)
    contract_values = dict(values)
    for name in (
        "bootstrap_expected_solver_seeds",
        "medium_large_tiers",
        "key_regression_metrics",
        "performance_taus",
    ):
        raw = values[name]
        if not isinstance(raw, list):
            raise ValueError(f"{field_name}.{name} must be a list")
        contract_values[name] = tuple(raw)
    try:
        return FormalAnalysisContract(**contract_values)
    except (TypeError, ValueError) as error:
        raise ValueError(f"invalid {field_name}: {error}") from error


def _profile_from_payload(payload: object, field_name: str) -> ProfileConfig:
    values = _as_mapping(payload, field_name)
    _require_exact_fields(values, ProfileConfig, field_name)
    try:
        return ProfileConfig(
            code=ProfileCode(values["code"]),
            static=_dataclass_from_mapping(
                StaticStrengtheningConfig,
                values["static"],
                f"{field_name}.static",
            ),
            user_cuts=_dataclass_from_mapping(
                UserCutConfig,
                values["user_cuts"],
                f"{field_name}.user_cuts",
            ),
            start=_dataclass_from_mapping(
                StartConfig,
                values["start"],
                f"{field_name}.start",
            ),
        )
    except (TypeError, ValueError) as error:
        raise ValueError(f"invalid {field_name}: {error}") from error


def _suite_from_payload(payload: object) -> MatchedProfileSuite:
    values = _as_mapping(payload, "formal_plan.suite")
    _require_exact_fields(values, MatchedProfileSuite, "formal_plan.suite")
    raw_profiles = values["profiles"]
    if not isinstance(raw_profiles, list):
        raise ValueError("formal_plan.suite.profiles must be a list")
    try:
        return MatchedProfileSuite(
            solver=_dataclass_from_mapping(
                SolverConfig,
                values["solver"],
                "formal_plan.suite.solver",
            ),
            telemetry=_dataclass_from_mapping(
                TelemetryConfig,
                values["telemetry"],
                "formal_plan.suite.telemetry",
            ),
            profiles=tuple(
                _profile_from_payload(item, f"formal_plan.suite.profiles[{index}]")
                for index, item in enumerate(raw_profiles)
            ),
        )
    except (TypeError, ValueError) as error:
        raise ValueError(f"invalid formal_plan.suite: {error}") from error


def _slot_from_payload(payload: object, field_name: str) -> FormalSlot:
    values = _as_mapping(payload, field_name)
    _require_exact_fields(values, FormalSlot, field_name)
    try:
        return FormalSlot(**dict(values))
    except (TypeError, ValueError) as error:
        raise ValueError(f"invalid {field_name}: {error}") from error


def _block_from_payload(payload: object, field_name: str) -> FormalBlock:
    values = _as_mapping(payload, field_name)
    _require_exact_fields(values, FormalBlock, field_name)
    raw_slots = values["slots"]
    if not isinstance(raw_slots, list):
        raise ValueError(f"{field_name}.slots must be a list")
    block_values = dict(values)
    block_values["slots"] = tuple(
        _slot_from_payload(item, f"{field_name}.slots[{index}]")
        for index, item in enumerate(raw_slots)
    )
    try:
        return FormalBlock(**block_values)
    except (TypeError, ValueError) as error:
        raise ValueError(f"invalid {field_name}: {error}") from error


def _plan_from_payload(
    payload: object,
    manifest: BenchmarkManifest,
) -> FormalExecutionPlan:
    values = _as_mapping(payload, "formal_plan")
    _require_exact_fields(values, FormalExecutionPlan, "formal_plan")
    raw_scale_contracts = values["scale_contracts"]
    raw_profile_order = values["profile_base_order"]
    raw_solver_seeds = values["solver_seeds"]
    raw_blocks = values["blocks"]
    if not isinstance(raw_scale_contracts, list):
        raise ValueError("formal_plan.scale_contracts must be a list")
    if not isinstance(raw_profile_order, list):
        raise ValueError("formal_plan.profile_base_order must be a list")
    if not isinstance(raw_solver_seeds, list):
        raise ValueError("formal_plan.solver_seeds must be a list")
    if not isinstance(raw_blocks, list):
        raise ValueError("formal_plan.blocks must be a list")
    plan_values = dict(values)
    plan_values["environment_contract"] = _environment_contract_from_payload(
        values["environment_contract"]
    )
    plan_values["analysis_contract"] = _analysis_contract_from_payload(
        values["analysis_contract"]
    )
    plan_values["suite"] = _suite_from_payload(values["suite"])
    plan_values["solver_seeds"] = tuple(raw_solver_seeds)
    plan_values["scale_contracts"] = tuple(
        _dataclass_from_mapping(
            FormalScaleContract,
            item,
            f"formal_plan.scale_contracts[{index}]",
        )
        for index, item in enumerate(raw_scale_contracts)
    )
    plan_values["profile_base_order"] = tuple(raw_profile_order)
    plan_values["blocks"] = tuple(
        _block_from_payload(item, f"formal_plan.blocks[{index}]")
        for index, item in enumerate(raw_blocks)
    )
    try:
        plan = FormalExecutionPlan(**plan_values)
    except (TypeError, ValueError) as error:
        raise ValueError(f"invalid formal_plan: {error}") from error
    validate_formal_execution_plan(plan, manifest)
    if canonical_json(plan.to_payload()) != canonical_json(payload):
        raise ValueError("formal plan payload is not a canonical round trip")
    return plan


def formal_plan_from_artifact(
    artifact: object,
    manifest: BenchmarkManifest,
) -> FormalExecutionPlan:
    """Load, hash-check, reconstruct, and validate a formal-plan artifact."""

    values = _as_mapping(artifact, "formal-plan artifact")
    expected_fields = {
        "artifact_schema_version",
        "artifact_type",
        "formal_plan",
        "formal_plan_id",
        "formal_plan_sha256",
    }
    if set(values) != expected_fields:
        raise ValueError("formal-plan artifact fields are inconsistent")
    if values["artifact_type"] != FORMAL_PLAN_ARTIFACT_TYPE:
        raise ValueError("unexpected formal-plan artifact_type")
    artifact_schema_version = values["artifact_schema_version"]
    _require_int(artifact_schema_version, "artifact_schema_version", minimum=1)
    if artifact_schema_version != FORMAL_PLAN_SCHEMA_VERSION:
        raise ValueError("unsupported formal-plan artifact schema version")
    plan = _plan_from_payload(values["formal_plan"], manifest)
    if values["formal_plan_sha256"] != plan.sha256:
        raise ValueError("formal_plan_sha256 does not match the canonical plan")
    if values["formal_plan_id"] != plan.plan_id:
        raise ValueError("formal_plan_id does not match the canonical plan")
    return plan


__all__ = [
    "FORMAL_ATTEMPT_NUMBER",
    "FORMAL_AGGREGATE_GATE_POLICY",
    "FORMAL_AGGREGATE_SIGNIFICANCE_POLICY",
    "FORMAL_ANALYSIS_CONTRACT_SCHEMA_VERSION",
    "FORMAL_BLOCK_COUNT",
    "FORMAL_BOOTSTRAP_METHOD",
    "FORMAL_BOOTSTRAP_SAMPLING_UNIT",
    "FORMAL_BOOTSTRAP_STRATIFICATION_POLICY",
    "FORMAL_CASE_COUNT",
    "FORMAL_CASES_PER_SCALE",
    "FORMAL_PLAN_ARTIFACT_TYPE",
    "FORMAL_PLAN_SCHEMA_VERSION",
    "FORMAL_COMMON_TARGET_POLICY",
    "FORMAL_HORIZON_POLICY",
    "FORMAL_MISSING_PROGRESS_POLICY",
    "FORMAL_MEDIAN_SUMMARY_POLICY",
    "FORMAL_PERFORMANCE_PROFILE_DENOMINATOR_POLICY",
    "FORMAL_PERFORMANCE_PROFILE_POLICY",
    "FORMAL_PREPARATION_TIME_POLICY",
    "FORMAL_PROFILE_ORDER",
    "FORMAL_ROTATION_RULE",
    "FORMAL_RUNTIME_CENSORING_POLICY",
    "FORMAL_RUNTIME_EFFECT_POLICY",
    "FORMAL_SCALE_CONTRACTS",
    "FORMAL_SIGN_TEST_SAMPLING_UNIT",
    "FORMAL_SLOT_COUNT",
    "FORMAL_SOLVER_SEEDS",
    "FORMAL_STATUS_RECONCILIATION_POLICY",
    "FORMAL_SYSTEMATIC_REGRESSION_POLICY",
    "FORMAL_TRAJECTORY_POLICY_VERSION",
    "FormalAnalysisContract",
    "FormalBlock",
    "FormalExecutionPlan",
    "FormalScaleContract",
    "FormalSlot",
    "build_formal_execution_plan",
    "formal_plan_from_artifact",
    "latin_profile_order",
    "validate_formal_execution_plan",
    "validate_formal_plan_environment",
]
