"""Preregister and run the development-only H6 OD-disaggregated return ablation.

The experiment is intentionally outside qualification and the formal A--E
campaign.  It compares the current paper-C cut profile against the same
profile with only the OD-disaggregated directed-return separator enabled.  Planning
and verification are solver-free; execution is serial, resource-locked, and
requires an explicit plan identity and slot cap.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import shlex
import subprocess
import sys
from typing import Any

from main import (
    FORMULATION_IMPLEMENTATION_FILES,
    REPO_ROOT as MAIN_REPO_ROOT,
    _screening_is_exact,
    build_parser as build_main_parser,
)
from src.config import ModelConfig, SolverConfig, to_dict
from src.data_processor.loader import DataLoader
from src.model_contract import FORMULATION_ID, RESULT_SCHEMA_VERSION

from .artifacts import write_canonical_json_exclusive
from .case_builder import (
    QualificationSolveProfile,
    canonical_instance_hash,
    qualification_case_sha256,
)
from .hashing import canonical_json, canonical_sha256
from .qualification import (
    QualificationEvidence,
    QualificationPolicy,
    assess_qualification,
    service_tolerance,
)
from .qualification_geometry_start import GeometryStartPolicy
from .qualification_runner import (
    NativeQualificationSettings,
    QualificationInfrastructureError,
    QualificationModelContract,
    QualificationRunnerOutcome,
    qualify_canonical_instance,
)
from .resource_lock import (
    DEFAULT_BENCHMARK_RESOURCE_LOCK,
    BenchmarkResourceLock,
    BenchmarkResourceLockError,
)


PLAN_ARTIFACT_TYPE = "mlnd_h6_od_disaggregated_return_development_ablation_plan"
PLAN_SCHEMA_VERSION = 1
PLAN_IDENTITY_SCHEMA = "mlnd-h6-od-disaggregated-return-ablation-plan-v1"
PLAN_ID_PREFIX = "h6-od-disaggregated-return-plan-"
SUMMARY_ARTIFACT_TYPE = "mlnd_h6_od_disaggregated_return_development_ablation_summary"
SUMMARY_SCHEMA_VERSION = 1
QUALIFICATION_ARTIFACT_TYPE = "mlnd_h6_current_model_qualification_summary"
QUALIFICATION_CHECKPOINT_ARTIFACT_TYPE = (
    "mlnd_h6_current_model_qualification_checkpoint"
)
QUALIFICATION_SCHEMA_VERSION = 1
QUALIFICATION_IDENTITY_SCHEMA = "mlnd-h6-current-model-qualification-v1"
QUALIFICATION_ID_PREFIX = "h6-current-model-qualification-"
REQUIRED_BASELINE_COMMIT = "f8849384058f86f4da9e39337b91d0697462f42f"
FORMULATION_RELATIVE_PATH = Path("MLND/main.tex")
EXPECTED_FORMULATION_SHA256 = (
    "6575f7ad5af2e62ba2c73f141ccd3f0b39d7c90a5e08868ac439d5b7d3531720"
)
INSTANCE_RELATIVE_PATH = Path(
    "results/branch_cut_paper/dev/micro_crop_matrix_v2/"
    "nanshan_d04_05_noon_12_13/H006_L02_E02_R02p20"
)
EXPECTED_INSTANCE_SHA256 = (
    "d6d70495f240c12b35ad3b1fc6d817c7f7b3948a8c62e5662f21fb77790e7617"
)
DEVELOPMENT_ROOT_RELATIVE_PATH = Path("results/branch_cut_paper/dev")

FIXED_COSTS = (24.0, 26.0)
SEEDS = (11, 29, 47)
VARIANTS = ("C-current", "C+OD-return")
TIME_LIMIT_SECONDS = 5.0
EXPECTED_PAIR_COUNT = len(FIXED_COSTS) * len(SEEDS)
EXPECTED_SLOT_COUNT = EXPECTED_PAIR_COUNT * len(VARIANTS)
ALLOWED_MAX_SLOTS = tuple(range(2, EXPECTED_SLOT_COUNT + 1, 2))
DISPLAY_ORIGIN_CHOICES = 8
GENERAL_CORRIDOR_CHOICES = 10
FEEDER_RIDER_COST_FACTOR = 0.30
ROOT_CUT_ROUNDS = 3
ROOT_CUT_PASSES = 2
MAX_CUTS_PER_ROUND = 250
PROGRESS_INTERVAL_SECONDS = 0.25
QUALIFICATION_TIME_LIMIT_SECONDS = 15.0
QUALIFICATION_SOLVER_SEED = 42
ADVANCE_MIN_ELIGIBLE_PAIRS = 5
ADVANCE_MIN_ACTIVE_TREATMENT_RUNS = 4
ADVANCE_MIN_GAP_WINS = 4
ADVANCE_MAX_GEOMETRIC_MEAN_GAP_RATIO = 0.90

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA1_RE = re.compile(r"^[0-9a-f]{40}$")
_PLAN_ID_RE = re.compile(r"^h6-od-disaggregated-return-plan-[0-9a-f]{64}$")
_QUALIFICATION_ID_RE = re.compile(r"^h6-current-model-qualification-[0-9a-f]{64}$")
_NO_INCUMBENT_RE = re.compile(
    r"MX finished without a feasible incumbent \(status=(?P<status>-?[0-9]+)\)"
)

_RUNNER_RELATIVE_PATHS = (
    Path("src/benchmark/branch_cut/h6_od_disaggregated_return_ablation.py"),
    Path("src/benchmark/branch_cut/artifacts.py"),
    Path("src/benchmark/branch_cut/resource_lock.py"),
)
_MODEL_IMPLEMENTATION_RELATIVE_PATHS = tuple(
    path.relative_to(MAIN_REPO_ROOT) for path in FORMULATION_IMPLEMENTATION_FILES
)
_QUALIFICATION_IMPLEMENTATION_RELATIVE_PATHS = (
    Path("src/benchmark/branch_cut/h6_od_disaggregated_return_ablation.py"),
    Path("src/benchmark/branch_cut/case_builder.py"),
    Path("src/benchmark/branch_cut/hashing.py"),
    Path("src/benchmark/branch_cut/qualification.py"),
    Path("src/benchmark/branch_cut/qualification_geometry_start.py"),
    Path("src/benchmark/branch_cut/qualification_runner.py"),
    Path("src/benchmark/branch_cut/qualification_incumbent_reuse.py"),
    Path("src/benchmark/branch_cut/artifacts.py"),
    Path("src/benchmark/branch_cut/resource_lock.py"),
)
IMPLEMENTATION_RELATIVE_PATHS = tuple(
    dict.fromkeys(
        (
            *_MODEL_IMPLEMENTATION_RELATIVE_PATHS,
            *_QUALIFICATION_IMPLEMENTATION_RELATIVE_PATHS,
            *_RUNNER_RELATIVE_PATHS,
        )
    )
)


class H6ODDisaggregatedReturnAblationError(RuntimeError):
    """The development ablation cannot proceed without weakening its contract."""


@dataclass(frozen=True, slots=True)
class FrozenInputs:
    repo_root: Path
    commit: str
    branch: str
    implementation_sha256: str
    implementation_files: Mapping[str, str]
    model_implementation_sha256: str
    model_implementation_files: Mapping[str, str]
    qualification_implementation_sha256: str
    qualification_implementation_files: Mapping[str, str]
    formulation_path: Path
    formulation_sha256: str
    instance_dir: Path
    instance_sha256: str
    instance_files: Mapping[str, str]
    screening_by_fixed_cost: Mapping[str, bool]
    python_executable: Path


@dataclass(frozen=True, slots=True)
class VerifiedPlan:
    path: Path
    payload: Mapping[str, Any]

    @property
    def plan_id(self) -> str:
        return str(self.payload["plan_id"])

    @property
    def run_root(self) -> Path:
        return Path(str(self.payload["run_root"]))


@dataclass(frozen=True, slots=True)
class VerifiedQualification:
    path: Path
    payload: Mapping[str, Any]

    @property
    def qualification_id(self) -> str:
        return str(self.payload["qualification_id"])


ProcessRunner = Callable[..., Any]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(repo_root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ("git", *arguments),
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        raise H6ODDisaggregatedReturnAblationError(
            f"git {' '.join(arguments)} failed: {detail}"
        )
    return completed.stdout.strip()


def _real_directory(path: Path, *, label: str) -> Path:
    absolute = path.expanduser().absolute()
    try:
        resolved = absolute.resolve(strict=True)
    except OSError as error:
        raise H6ODDisaggregatedReturnAblationError(
            f"{label} does not resolve: {absolute}: {error}"
        ) from error
    if absolute != resolved or absolute.is_symlink() or not resolved.is_dir():
        raise H6ODDisaggregatedReturnAblationError(
            f"{label} must be a real non-symlink directory: {absolute}"
        )
    return resolved


def _real_file(path: Path, *, label: str) -> Path:
    absolute = path.expanduser().absolute()
    try:
        resolved = absolute.resolve(strict=True)
    except OSError as error:
        raise H6ODDisaggregatedReturnAblationError(
            f"{label} does not resolve: {absolute}: {error}"
        ) from error
    if absolute != resolved or absolute.is_symlink() or not resolved.is_file():
        raise H6ODDisaggregatedReturnAblationError(
            f"{label} must be a real non-symlink file: {absolute}"
        )
    return resolved


def _resolved_python_executable() -> Path:
    try:
        resolved = Path(sys.executable).resolve(strict=True)
    except OSError as error:
        raise H6ODDisaggregatedReturnAblationError(
            f"Python executable does not resolve: {sys.executable}: {error}"
        ) from error
    return _real_file(resolved, label="Python executable")


def _prospective_path(path: Path, *, label: str) -> Path:
    absolute = path.expanduser().absolute()
    if absolute.exists() or absolute.is_symlink():
        raise H6ODDisaggregatedReturnAblationError(
            f"{label} must not exist: {absolute}"
        )
    parent = _real_directory(absolute.parent, label=f"{label} parent")
    if parent != absolute.parent:
        raise H6ODDisaggregatedReturnAblationError(
            f"{label} parent must not traverse a symlink: {absolute}"
        )
    return absolute


def _require_under(path: Path, parent: Path, *, label: str) -> None:
    try:
        path.relative_to(parent)
    except ValueError as error:
        raise H6ODDisaggregatedReturnAblationError(
            f"{label} must be inside {parent}: {path}"
        ) from error


def _clean_repository(repo_root: Path) -> tuple[str, str]:
    if _git(repo_root, "rev-parse", "--is-inside-work-tree") != "true":
        raise H6ODDisaggregatedReturnAblationError(
            "repository root is not a Git worktree"
        )
    status = _git(repo_root, "status", "--porcelain=v1", "--untracked-files=all")
    if status:
        raise H6ODDisaggregatedReturnAblationError(
            "H6 ablation requires a clean committed worktree"
        )
    commit = _git(repo_root, "rev-parse", "--verify", "HEAD")
    object_format = _git(repo_root, "rev-parse", "--show-object-format")
    commit_pattern = {
        "sha1": _GIT_SHA1_RE,
        "sha256": _SHA256_RE,
    }.get(object_format)
    if commit_pattern is None or commit_pattern.fullmatch(commit) is None:
        raise H6ODDisaggregatedReturnAblationError(
            "HEAD does not match the repository object format"
        )
    branch = _git(repo_root, "branch", "--show-current")
    if not branch.startswith("codex/"):
        raise H6ODDisaggregatedReturnAblationError(
            "H6 ablation must run from an isolated codex/ branch"
        )
    try:
        _git(
            repo_root,
            "merge-base",
            "--is-ancestor",
            REQUIRED_BASELINE_COMMIT,
            commit,
        )
    except H6ODDisaggregatedReturnAblationError as error:
        raise H6ODDisaggregatedReturnAblationError(
            "H6 ablation HEAD must descend from the frozen f884 baseline"
        ) from error
    return commit, branch


def _implementation_provenance(repo_root: Path) -> tuple[str, dict[str, str]]:
    files: dict[str, str] = {}
    for relative in IMPLEMENTATION_RELATIVE_PATHS:
        path = _real_file(repo_root / relative, label="ablation implementation file")
        files[relative.as_posix()] = _sha256_file(path)
    digest = canonical_sha256(files)
    return digest, files


def _subset_provenance(
    implementation_files: Mapping[str, str], relative_paths: Sequence[Path]
) -> tuple[str, dict[str, str]]:
    files = {
        relative.as_posix(): implementation_files[relative.as_posix()]
        for relative in relative_paths
    }
    return canonical_sha256(files), files


def _model_config(fixed_cost: float) -> ModelConfig:
    config = ModelConfig(
        locker_fixed_cost_per_hour=fixed_cost,
        feeder_rider_cost_factor=FEEDER_RIDER_COST_FACTOR,
        display_origin_choices=DISPLAY_ORIGIN_CHOICES,
        general_corridor_choices=GENERAL_CORRIDOR_CHOICES,
    )
    config.validate()
    return config


def _solver_config(seed: int, *, od_disaggregated_return: bool) -> SolverConfig:
    config = SolverConfig(
        time_limit=TIME_LIMIT_SECONDS,
        mip_gap=0.0,
        threads=1,
        output_flag=1,
        seed=seed,
        network_start_seconds=0.0,
        no_rel_heur_seconds=0.0,
        network_start_method=2,
        use_valid_inequalities=True,
        use_user_cuts=True,
        max_root_cut_rounds=ROOT_CUT_ROUNDS,
        root_cut_passes=ROOT_CUT_PASSES,
        max_cuts_per_round=MAX_CUTS_PER_ROUND,
        cut_violation_tolerance=1e-6,
        cut_efficacy_tolerance=1e-9,
        fixed_integer_repair_seconds=0.0,
        use_local_branching=False,
        local_branching_seconds=0.0,
        local_branching_radius=12,
        local_branching_include_y=False,
        root_relaxation_method=-1,
        miqcp_method=-1,
        start_mode="none",
        use_objective_partial_start=False,
        objective_partial_start_seconds=0.25,
        objective_partial_start_max_candidates=3,
        mip_focus=1,
        heuristics=0.2,
        no_rel_heur_time=0.0,
        presolve=-1,
        cuts=-1,
        numeric_focus=0,
        static_nearest_display_suffix=False,
        static_chi_vub_top_k_per_corridor=0,
        cut_route_swap_station_vub=False,
        cut_route_swap_leg_vub=False,
        cut_chi_facility_vub=False,
        cut_od_disaggregated_return=od_disaggregated_return,
        cut_directed_return=False,
        collect_progress=True,
        progress_interval_seconds=PROGRESS_INTERVAL_SECONDS,
    )
    config.validate()
    return config


def _screening_audit(
    instance_dir: Path,
) -> dict[str, bool]:
    result: dict[str, bool] = {}
    for fixed_cost in FIXED_COSTS:
        instance = DataLoader().load_instance_dir(
            str(instance_dir),
            model_config=_model_config(fixed_cost),
            demand_scale=1.0,
        )
        exact = _screening_is_exact(instance, ("MX",))["MX"]
        result[f"fc{int(fixed_cost):02d}"] = exact
    return result


def _freeze_inputs(repo_root: Path) -> FrozenInputs:
    root = _real_directory(repo_root, label="repository root")
    active_root = _real_directory(MAIN_REPO_ROOT, label="active module repository root")
    if root != active_root:
        raise H6ODDisaggregatedReturnAblationError(
            "repo_root must be the repository that supplied the running H6 wrapper"
        )
    commit, branch = _clean_repository(root)
    implementation_sha256, implementation_files = _implementation_provenance(root)
    model_implementation_sha256, model_implementation_files = _subset_provenance(
        implementation_files, _MODEL_IMPLEMENTATION_RELATIVE_PATHS
    )
    qualification_implementation_sha256, qualification_implementation_files = (
        _subset_provenance(
            implementation_files, _QUALIFICATION_IMPLEMENTATION_RELATIVE_PATHS
        )
    )
    formulation_path = _real_file(
        root / FORMULATION_RELATIVE_PATH, label="formulation source"
    )
    formulation_sha256 = _sha256_file(formulation_path)
    if formulation_sha256 != EXPECTED_FORMULATION_SHA256:
        raise H6ODDisaggregatedReturnAblationError(
            "formulation hash mismatch: expected "
            f"{EXPECTED_FORMULATION_SHA256}, got {formulation_sha256}"
        )
    instance_dir = _real_directory(root / INSTANCE_RELATIVE_PATH, label="H6 instance")
    instance_sha256, instance_files = canonical_instance_hash(instance_dir)
    if instance_sha256 != EXPECTED_INSTANCE_SHA256:
        raise H6ODDisaggregatedReturnAblationError(
            "H6 instance hash mismatch: expected "
            f"{EXPECTED_INSTANCE_SHA256}, got {instance_sha256}"
        )
    for filename in instance_files:
        _real_file(instance_dir / filename, label=f"H6 instance file {filename}")
    screening = _screening_audit(instance_dir)
    if set(screening) != {"fc24", "fc26"} or any(screening.values()):
        raise H6ODDisaggregatedReturnAblationError(
            "H6 ablation requires screening_is_exact=false at fc24 and fc26"
        )
    python_executable = _resolved_python_executable()
    return FrozenInputs(
        repo_root=root,
        commit=commit,
        branch=branch,
        implementation_sha256=implementation_sha256,
        implementation_files=implementation_files,
        model_implementation_sha256=model_implementation_sha256,
        model_implementation_files=model_implementation_files,
        qualification_implementation_sha256=qualification_implementation_sha256,
        qualification_implementation_files=qualification_implementation_files,
        formulation_path=formulation_path,
        formulation_sha256=formulation_sha256,
        instance_dir=instance_dir,
        instance_sha256=instance_sha256,
        instance_files=instance_files,
        screening_by_fixed_cost=screening,
        python_executable=python_executable,
    )


def _od_disaggregated_return_for_variant(variant: str) -> bool:
    if variant == "C-current":
        return False
    if variant == "C+OD-return":
        return True
    raise H6ODDisaggregatedReturnAblationError(f"unknown ablation variant: {variant}")


def _variant_slug(variant: str) -> str:
    return variant.replace("+", "_plus_").replace("-", "_")


def _qualification_profile(fixed_cost: float) -> QualificationSolveProfile:
    return QualificationSolveProfile(
        profile_id=f"h6-current-model-fc{int(fixed_cost):02d}",
        demand_scale=1.0,
        feeder_rider_cost_factor=FEEDER_RIDER_COST_FACTOR,
        locker_fixed_cost_per_hour=fixed_cost,
        display_origin_choices=DISPLAY_ORIGIN_CHOICES,
        general_corridor_choices=GENERAL_CORRIDOR_CHOICES,
        expected_screening_is_exact=False,
        interpretation=(
            "Algorithm-neutral current-model H6 prerequisite; temporary witness "
            "rows, if needed, are removed before fresh-original-model validation."
        ),
        adds_forced_opening_constraint=False,
    )


def _qualification_settings() -> NativeQualificationSettings:
    return NativeQualificationSettings(
        time_limit_seconds=QUALIFICATION_TIME_LIMIT_SECONDS,
        mip_gap=0.0,
        solver_seed=QUALIFICATION_SOLVER_SEED,
        threads=1,
        output_flag=0,
        method=-1,
        miqcp_method=-1,
        mip_focus=0,
        heuristics=0.05,
        no_rel_heur_time=0.0,
        presolve=-1,
        native_cuts=-1,
        root_cut_passes=ROOT_CUT_PASSES,
        numeric_focus=0,
        feasibility_tolerance=1e-6,
        optimality_tolerance=1e-6,
        integer_feasibility_tolerance=1e-5,
    )


def _qualification_policy() -> QualificationPolicy:
    return QualificationPolicy()


def _qualification_geometry_policy() -> GeometryStartPolicy:
    # H6 is intentionally outside the frozen H23/H51/H90 geometry schedules.
    return GeometryStartPolicy(enabled=False)


def _qualification_contract(
    frozen: FrozenInputs, fixed_cost: float
) -> tuple[QualificationSolveProfile, QualificationModelContract]:
    profile = _qualification_profile(fixed_cost)
    contract = QualificationModelContract(
        qualification_case_sha256=qualification_case_sha256(
            frozen.instance_sha256, profile
        ),
        effective_model_config_sha256=profile.effective_model_config_sha256(),
        demand_scale=profile.demand_scale,
        locker_fixed_cost_per_hour=profile.locker_fixed_cost_per_hour,
        display_origin_choices=profile.display_origin_choices,
        general_corridor_choices=profile.general_corridor_choices,
        screening_is_exact=False,
    )
    return profile, contract


def _qualification_identity_payload(frozen: FrozenInputs) -> dict[str, Any]:
    cases: list[dict[str, Any]] = []
    for fixed_cost in FIXED_COSTS:
        profile, contract = _qualification_contract(frozen, fixed_cost)
        cases.append(
            {
                "case_id": f"h6-fc{int(fixed_cost):02d}-current-model",
                "scale_label": "small",
                "fixed_cost": fixed_cost,
                "profile": profile.to_payload(),
                "contract": contract.to_payload(),
            }
        )
    return {
        "identity_schema": QUALIFICATION_IDENTITY_SCHEMA,
        "development_only": True,
        "formal_evidence_eligible": False,
        "eligible_only_as_ablation_prerequisite": True,
        "repository": {
            "root": str(frozen.repo_root),
            "commit": frozen.commit,
            "branch": frozen.branch,
            "required_baseline_ancestor": REQUIRED_BASELINE_COMMIT,
            "model_implementation_sha256": frozen.model_implementation_sha256,
            "model_implementation_files": dict(frozen.model_implementation_files),
            "qualification_implementation_sha256": (
                frozen.qualification_implementation_sha256
            ),
            "qualification_implementation_files": dict(
                frozen.qualification_implementation_files
            ),
            "full_wrapper_implementation_sha256": frozen.implementation_sha256,
            "full_wrapper_implementation_files": dict(frozen.implementation_files),
        },
        "formulation": {
            "id": FORMULATION_ID,
            "result_schema_version": RESULT_SCHEMA_VERSION,
            "path": str(frozen.formulation_path),
            "sha256": frozen.formulation_sha256,
        },
        "instance": {
            "stage": "H6",
            "path": str(frozen.instance_dir),
            "sha256": frozen.instance_sha256,
            "canonical_file_sha256": dict(frozen.instance_files),
            "candidate_h": 6,
        },
        "screening_is_exact_by_fixed_cost": dict(frozen.screening_by_fixed_cost),
        "policy": _qualification_policy().to_payload(),
        "native_settings": asdict(_qualification_settings()),
        "geometry_start_policy": _qualification_geometry_policy().to_payload(),
        "cases": cases,
        "execution_contract": {
            "case_order": ["fc24", "fc26"],
            "worker_count": 1,
            "shared_resource_lock": str(
                Path(DEFAULT_BENCHMARK_RESOURCE_LOCK).resolve()
            ),
            "temporary_witness_rows_are_not_instance_constraints": True,
            "fresh_original_model_validation_required": True,
            "both_cases_must_be_fresh_accepted": True,
            "fail_fast_on_infrastructure_error": True,
        },
    }


def _qualification_id(identity: Mapping[str, Any]) -> str:
    return QUALIFICATION_ID_PREFIX + canonical_sha256(identity)


def _source_snapshot(frozen: FrozenInputs) -> dict[str, Any]:
    if _git(frozen.repo_root, "rev-parse", "--verify", "HEAD") != frozen.commit:
        raise H6ODDisaggregatedReturnAblationError(
            "repository commit changed during H6 qualification"
        )
    if _git(frozen.repo_root, "branch", "--show-current") != frozen.branch:
        raise H6ODDisaggregatedReturnAblationError(
            "repository branch changed during H6 qualification"
        )
    implementation_sha256, implementation_files = _implementation_provenance(
        frozen.repo_root
    )
    if (
        implementation_sha256 != frozen.implementation_sha256
        or implementation_files != dict(frozen.implementation_files)
    ):
        raise H6ODDisaggregatedReturnAblationError(
            "implementation changed during H6 qualification"
        )
    formulation_sha256 = _sha256_file(frozen.formulation_path)
    if formulation_sha256 != frozen.formulation_sha256:
        raise H6ODDisaggregatedReturnAblationError(
            "formulation changed during H6 qualification"
        )
    instance_sha256, instance_files = canonical_instance_hash(frozen.instance_dir)
    if instance_sha256 != frozen.instance_sha256 or instance_files != dict(
        frozen.instance_files
    ):
        raise H6ODDisaggregatedReturnAblationError(
            "canonical H6 instance changed during qualification"
        )
    screening = _screening_audit(frozen.instance_dir)
    if screening != dict(frozen.screening_by_fixed_cost):
        raise H6ODDisaggregatedReturnAblationError(
            "H6 screening audit changed during qualification"
        )
    return {
        "repository_commit": frozen.commit,
        "implementation_sha256": implementation_sha256,
        "formulation_sha256": formulation_sha256,
        "instance_sha256": instance_sha256,
        "canonical_file_sha256": instance_files,
        "screening_is_exact_by_fixed_cost": screening,
    }


def _load_qualification_instance(
    instance_dir: Path, model_config: ModelConfig, demand_scale: float
) -> object:
    return DataLoader().load_instance_dir(
        str(instance_dir),
        model_config=model_config,
        demand_scale=demand_scale,
    )


def _validate_loaded_qualification_instance(
    instance: object,
    *,
    profile: QualificationSolveProfile,
) -> None:
    expected_config = profile.effective_model_config()
    if getattr(instance, "config", None) != expected_config:
        raise QualificationInfrastructureError(
            "loaded H6 qualification instance changed its full ModelConfig"
        )
    metadata = getattr(instance, "metadata", None)
    if not isinstance(metadata, Mapping) or metadata.get("demand_scale") != 1.0:
        raise QualificationInfrastructureError(
            "loaded H6 qualification instance did not retain demand_scale=1"
        )
    screening_reader = getattr(instance, "screening_is_exact", None)
    if not callable(screening_reader) or screening_reader("MX") is not False:
        raise QualificationInfrastructureError(
            "H6 qualification requires screening_is_exact=false"
        )


def _audit_matches_qualification_contract(
    audit: object, contract: QualificationModelContract
) -> bool:
    contract_values = getattr(audit, "contract_values", None)
    return callable(contract_values) and contract_values() == (
        contract.effective_model_config_sha256,
        contract.demand_scale,
        contract.locker_fixed_cost_per_hour,
        contract.display_origin_choices,
        contract.general_corridor_choices,
        False,
    )


def _validate_qualification_outcome(
    outcome: QualificationRunnerOutcome,
    *,
    case_id: str,
    qualification_id: str,
    contract: QualificationModelContract,
) -> None:
    if not isinstance(outcome, QualificationRunnerOutcome):
        raise QualificationInfrastructureError(
            "H6 qualifier did not return QualificationRunnerOutcome"
        )
    if outcome.expected_model_contract != contract:
        raise QualificationInfrastructureError(
            "H6 qualification outcome changed its expected model contract"
        )
    if outcome.geometry_start_policy != _qualification_geometry_policy():
        raise QualificationInfrastructureError(
            "H6 qualification outcome did not keep geometry start disabled"
        )
    if outcome.exception_messages:
        raise QualificationInfrastructureError(
            "H6 qualification outcome contains mechanism exceptions"
        )
    if outcome.evidence.case_id != case_id:
        raise QualificationInfrastructureError("qualification case_id mismatch")
    if outcome.evidence.qualification_run_id != qualification_id:
        raise QualificationInfrastructureError("qualification run_id mismatch")
    if outcome.evidence.instance_sha256 != EXPECTED_INSTANCE_SHA256:
        raise QualificationInfrastructureError("qualification instance hash mismatch")
    if outcome.evidence.qualification_case_sha256 != (
        contract.qualification_case_sha256
    ):
        raise QualificationInfrastructureError("qualification case hash mismatch")
    if outcome.evidence.effective_model_config_sha256 != (
        contract.effective_model_config_sha256
    ):
        raise QualificationInfrastructureError("qualification config hash mismatch")
    if outcome.evidence.screening_is_exact is not False:
        raise QualificationInfrastructureError(
            "qualification evidence changed the screening boundary"
        )
    if not _audit_matches_qualification_contract(outcome.witness_model_audit, contract):
        raise QualificationInfrastructureError(
            "qualification witness model audit does not match the contract"
        )
    if outcome.fresh_model_validation_performed:
        if not _audit_matches_qualification_contract(
            outcome.fresh_model_audit, contract
        ):
            raise QualificationInfrastructureError(
                "fresh qualification model audit does not match the contract"
            )
        witness_fingerprint = getattr(
            outcome.witness_model_audit, "model_fingerprint", None
        )
        fresh_fingerprint = getattr(
            outcome.fresh_model_audit, "model_fingerprint", None
        )
        if (
            witness_fingerprint is None
            or fresh_fingerprint is None
            or witness_fingerprint != fresh_fingerprint
        ):
            raise QualificationInfrastructureError(
                "fresh qualification model fingerprint differs from witness model"
            )
    elif outcome.fresh_model_audit is not None:
        raise QualificationInfrastructureError(
            "fresh model audit exists without fresh validation"
        )
    if (
        outcome.temporary_constraint_used
        and outcome.temporary_constraint_removed is not True
    ):
        raise QualificationInfrastructureError(
            "temporary qualification rows were not proven removed"
        )
    if outcome.accepted and (
        outcome.decision.accepted is not True
        or outcome.evidence.witness_feasible is not True
        or outcome.fresh_model_validation_performed is not True
        or outcome.captured_variable_count <= 0
        or outcome.validation_status not in {"OPTIMAL", "TIME_LIMIT"}
    ):
        raise QualificationInfrastructureError(
            "accepted H6 outcome lacks fresh-original-model evidence"
        )
    # Force strict canonical-JSON conversion before publishing any checkpoint.
    canonical_json(outcome.to_payload())


def _qualification_record_is_fresh_accepted(
    record: Mapping[str, Any],
    contract: QualificationModelContract,
    *,
    expected_case_id: str,
    expected_qualification_id: str,
    expected_instance_sha256: str,
) -> bool:
    if (
        set(record)
        != {
            "artifact_type",
            "schema_version",
            "development_only",
            "formal_evidence_eligible",
            "qualification_id",
            "case",
            "source_before",
            "source_after",
            "status",
            "infrastructure_error",
            "outcome",
        }
        or record.get("infrastructure_error") is not None
    ):
        return False
    if record.get("status") != "completed":
        return False
    outcome = record.get("outcome")
    expected_outcome_fields = {
        "accepted",
        "rejection_reasons",
        "evidence",
        "decision",
        "required_service_per_hour",
        "required_objective_upper_bound",
        "temporary_constraint_used",
        "temporary_constraint_removed",
        "fresh_model_validation_performed",
        "witness_status",
        "validation_status",
        "captured_variable_count",
        "exception_messages",
        "feasibility_objective_used",
        "expected_model_contract",
        "witness_model_audit",
        "fresh_model_audit",
        "geometry_start_policy",
        "geometry_start_attempts",
        "geometry_start_selected_plan_sha256",
        "phase_timings",
    }
    if not isinstance(outcome, Mapping) or set(outcome) != expected_outcome_fields:
        return False
    evidence = outcome.get("evidence")
    decision = outcome.get("decision")
    expected_contract = outcome.get("expected_model_contract")
    fresh_audit = outcome.get("fresh_model_audit")
    witness_audit = outcome.get("witness_model_audit")
    geometry = outcome.get("geometry_start_policy")
    try:
        recomputed_evidence = QualificationEvidence(**dict(evidence or {}))
        recomputed_decision = assess_qualification(
            recomputed_evidence, _qualification_policy()
        )
    except (TypeError, ValueError):
        return False
    expected_audit = {
        "effective_model_config_sha256": contract.effective_model_config_sha256,
        "demand_scale": contract.demand_scale,
        "locker_fixed_cost_per_hour": contract.locker_fixed_cost_per_hour,
        "display_origin_choices": contract.display_origin_choices,
        "general_corridor_choices": contract.general_corridor_choices,
        "screening_is_exact": False,
    }
    expected_audit_fields = {*expected_audit, "model_fingerprint"}
    expected_required_service = max(
        float(recomputed_decision.service_tolerance or 0.0),
        _qualification_policy().minimum_service_fraction
        * float(recomputed_decision.total_demand_per_hour or 0.0),
    )
    expected_required_objective = float(
        recomputed_decision.courier_only_objective or 0.0
    ) - max(
        float(recomputed_decision.objective_margin or 0.0),
        _qualification_policy().minimum_savings_fraction
        * float(recomputed_decision.courier_only_objective or 0.0),
    )
    raw_required_service = outcome.get("required_service_per_hour")
    raw_required_objective = outcome.get("required_objective_upper_bound")
    thresholds_match = bool(
        isinstance(raw_required_service, (int, float))
        and not isinstance(raw_required_service, bool)
        and math.isclose(
            float(raw_required_service), expected_required_service, abs_tol=1e-9
        )
        and isinstance(raw_required_objective, (int, float))
        and not isinstance(raw_required_objective, bool)
        and math.isclose(
            float(raw_required_objective),
            expected_required_objective,
            abs_tol=1e-9,
        )
    )
    return bool(
        record.get("artifact_type") == QUALIFICATION_CHECKPOINT_ARTIFACT_TYPE
        and record.get("schema_version") == QUALIFICATION_SCHEMA_VERSION
        and record.get("development_only") is True
        and record.get("formal_evidence_eligible") is False
        and record.get("qualification_id") == expected_qualification_id
        and outcome.get("accepted") is True
        and outcome.get("rejection_reasons") == []
        and outcome.get("fresh_model_validation_performed") is True
        and outcome.get("captured_variable_count", 0) > 0
        and outcome.get("validation_status") in {"OPTIMAL", "TIME_LIMIT"}
        and outcome.get("exception_messages") == []
        and thresholds_match
        and isinstance(evidence, Mapping)
        and evidence.get("case_id") == expected_case_id
        and evidence.get("qualification_run_id") == expected_qualification_id
        and evidence.get("instance_sha256") == expected_instance_sha256
        and evidence.get("witness_feasible") is True
        and evidence.get("qualification_case_sha256")
        == contract.qualification_case_sha256
        and evidence.get("effective_model_config_sha256")
        == contract.effective_model_config_sha256
        and evidence.get("screening_is_exact") is False
        and isinstance(decision, Mapping)
        and decision.get("accepted") is True
        and decision == recomputed_decision.to_payload()
        and expected_contract == contract.to_payload()
        and isinstance(fresh_audit, Mapping)
        and isinstance(witness_audit, Mapping)
        and set(fresh_audit) == expected_audit_fields
        and set(witness_audit) == expected_audit_fields
        and all(fresh_audit.get(key) == value for key, value in expected_audit.items())
        and all(
            witness_audit.get(key) == value for key, value in expected_audit.items()
        )
        and fresh_audit.get("model_fingerprint") is not None
        and fresh_audit.get("model_fingerprint")
        == witness_audit.get("model_fingerprint")
        and geometry == _qualification_geometry_policy().to_payload()
        and outcome.get("geometry_start_attempts") == []
        and outcome.get("geometry_start_selected_plan_sha256") is None
        and (
            outcome.get("temporary_constraint_used") is not True
            or outcome.get("temporary_constraint_removed") is True
        )
    )


def _command(
    frozen: FrozenInputs,
    *,
    results_dir: Path,
    fixed_cost: float,
    seed: int,
    od_disaggregated_return: bool,
) -> tuple[str, ...]:
    return (
        str(frozen.python_executable),
        str(frozen.repo_root / "main.py"),
        "solve",
        "--instance-dir",
        str(frozen.instance_dir),
        "--results-dir",
        str(results_dir),
        "--modes",
        "MX",
        "--demand-scale",
        "1",
        "--display-origin-choices",
        str(DISPLAY_ORIGIN_CHOICES),
        "--general-corridor-choices",
        str(GENERAL_CORRIDOR_CHOICES),
        "--feeder-rider-cost-factor",
        str(FEEDER_RIDER_COST_FACTOR),
        "--locker-fixed-cost-per-hour",
        str(int(fixed_cost)),
        "--time-limit",
        str(int(TIME_LIMIT_SECONDS)),
        "--mip-gap",
        "0",
        "--threads",
        "1",
        "--start-mode",
        "none",
        "--network-start-seconds",
        "0",
        "--network-start-method",
        "2",
        "--fixed-integer-repair-seconds",
        "0",
        "--no-local-branching",
        "--local-branching-seconds",
        "0",
        "--local-branching-radius",
        "12",
        "--valid-inequalities",
        "--max-root-cut-rounds",
        str(ROOT_CUT_ROUNDS),
        "--root-cut-passes",
        str(ROOT_CUT_PASSES),
        "--max-cuts-per-round",
        str(MAX_CUTS_PER_ROUND),
        "--cut-violation-tolerance",
        "1e-6",
        "--cut-efficacy-tolerance",
        "1e-9",
        "--static-od-vub-per-corridor",
        "0",
        "--no-nearest-display-suffix-rows",
        "--no-route-swap-station-vub-cuts",
        "--no-route-swap-leg-vub-cuts",
        "--no-od-facility-vub-cuts",
        (
            "--od-disaggregated-return-cuts"
            if od_disaggregated_return
            else "--no-od-disaggregated-return-cuts"
        ),
        "--no-directed-return-cuts",
        "--root-method",
        "-1",
        "--miqcp-method",
        "-1",
        "--mip-focus",
        "1",
        "--heuristics",
        "0.2",
        "--no-rel-heur-seconds",
        "0",
        "--no-rel-heur-time",
        "0",
        "--presolve",
        "-1",
        "--native-cuts",
        "-1",
        "--numeric-focus",
        "0",
        "--collect-progress",
        "--progress-interval-seconds",
        str(PROGRESS_INTERVAL_SECONDS),
        "--no-dev-objective-partial-start",
        "--objective-partial-start-seconds",
        "0.25",
        "--objective-partial-start-max-candidates",
        "3",
        "--solver-seed",
        str(seed),
    )


def _pair_variant_order(pair_index_zero_based: int) -> tuple[str, str]:
    seed_rank, fixed_cost_rank = divmod(pair_index_zero_based, len(FIXED_COSTS))
    if (seed_rank + fixed_cost_rank) % 2 == 0:
        return VARIANTS
    return tuple(reversed(VARIANTS))  # type: ignore[return-value]


def _planned_jobs(
    frozen: FrozenInputs,
    run_root: Path,
) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    pair_index = 0
    slot_index = 0
    for seed in SEEDS:
        for fixed_cost in FIXED_COSTS:
            model_payload = to_dict(_model_config(fixed_cost))
            model_sha256 = canonical_sha256(model_payload)
            shared_solver_payload = to_dict(
                _solver_config(seed, od_disaggregated_return=True)
            )
            shared_solver_payload.pop("cut_od_disaggregated_return")
            shared_solver_sha256 = canonical_sha256(shared_solver_payload)
            pair_index += 1
            pair_payload = {
                "instance_sha256": frozen.instance_sha256,
                "model_config_sha256": model_sha256,
                "shared_solver_config_sha256": shared_solver_sha256,
                "fixed_cost": fixed_cost,
                "seed": seed,
                "mode": "MX",
                "time_limit_seconds": TIME_LIMIT_SECONDS,
            }
            pair_id = "h6-od-disaggregated-return-pair-" + canonical_sha256(
                pair_payload
            )
            variant_order = _pair_variant_order(pair_index - 1)
            for position, variant in enumerate(variant_order, start=1):
                slot_index += 1
                od_disaggregated_return = _od_disaggregated_return_for_variant(variant)
                solver_payload = to_dict(
                    _solver_config(
                        seed, od_disaggregated_return=od_disaggregated_return
                    )
                )
                results_dir = (
                    run_root
                    / "runs"
                    / f"fc{int(fixed_cost):02d}"
                    / f"seed{seed:03d}"
                    / _variant_slug(variant)
                )
                stem = (
                    f"fc{int(fixed_cost):02d}-seed{seed:03d}-{_variant_slug(variant)}"
                )
                job_identity = {
                    "pair_id": pair_id,
                    "variant": variant,
                    "model_config_sha256": model_sha256,
                    "solver_config_sha256": canonical_sha256(solver_payload),
                }
                jobs.append(
                    {
                        "slot_index": slot_index,
                        "pair_index": pair_index,
                        "pair_id": pair_id,
                        "shared_solver_config_sha256": shared_solver_sha256,
                        "job_id": "h6-od-disaggregated-return-job-"
                        + canonical_sha256(job_identity),
                        "fixed_cost": fixed_cost,
                        "seed": seed,
                        "variant": variant,
                        "variant_position_within_pair": position,
                        "cut_od_disaggregated_return": od_disaggregated_return,
                        "model_config": model_payload,
                        "model_config_sha256": model_sha256,
                        "solver_config": solver_payload,
                        "solver_config_sha256": canonical_sha256(solver_payload),
                        "instance_dir": str(frozen.instance_dir),
                        "results_dir": str(results_dir),
                        "driver_log": str(run_root / "driver_logs" / f"{stem}.log"),
                        "cache_dir": str(run_root / "cache" / stem),
                        "command": list(
                            _command(
                                frozen,
                                results_dir=results_dir,
                                fixed_cost=fixed_cost,
                                seed=seed,
                                od_disaggregated_return=od_disaggregated_return,
                            )
                        ),
                    }
                )
    return jobs


def _plan_identity_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in payload.items()
        if key not in {"created_at_utc", "plan_id"}
    }


def _plan_id(payload: Mapping[str, Any]) -> str:
    digest = canonical_sha256(
        {
            "identity_schema": PLAN_IDENTITY_SCHEMA,
            "payload": _plan_identity_payload(payload),
        }
    )
    return PLAN_ID_PREFIX + digest


def _qualification_gate_payload(
    verified: VerifiedQualification,
    frozen: FrozenInputs,
) -> dict[str, Any]:
    identity = verified.payload["identity"]
    repository = identity["repository"]
    return {
        "summary_path": str(verified.path),
        "summary_sha256": _sha256_file(verified.path),
        "qualification_id": verified.qualification_id,
        "identity_sha256": canonical_sha256(identity),
        "repository_commit": repository["commit"],
        "current_repository_commit": frozen.commit,
        "model_implementation_sha256": repository["model_implementation_sha256"],
        "model_implementation_files": dict(repository["model_implementation_files"]),
        "current_model_implementation_sha256": (frozen.model_implementation_sha256),
        "current_model_implementation_files": dict(frozen.model_implementation_files),
        "model_implementation_identical": (
            repository["model_implementation_sha256"]
            == frozen.model_implementation_sha256
        ),
        "reuse_policy": "exact_current_commit_and_wrapper_v1",
        "wrapper_identity_reused": False,
        "qualification_implementation_sha256": repository[
            "qualification_implementation_sha256"
        ],
        "both_fresh_accepted": True,
        "checkpoint_sha256": [
            record["checkpoint_sha256"] for record in verified.payload["case_records"]
        ],
    }


def _build_plan_payload(
    frozen: FrozenInputs,
    run_root: Path,
    qualification: VerifiedQualification,
    *,
    created_at_utc: str,
) -> dict[str, Any]:
    jobs = _planned_jobs(frozen, run_root)
    payload: dict[str, Any] = {
        "artifact_type": PLAN_ARTIFACT_TYPE,
        "schema_version": PLAN_SCHEMA_VERSION,
        "development_only": True,
        "formal_evidence_eligible": False,
        "qualification_evidence_eligible": False,
        "created_at_utc": created_at_utc,
        "run_root": str(run_root),
        "repository": {
            "root": str(frozen.repo_root),
            "commit": frozen.commit,
            "branch": frozen.branch,
            "required_baseline_ancestor": REQUIRED_BASELINE_COMMIT,
            "implementation_sha256": frozen.implementation_sha256,
            "implementation_files": dict(frozen.implementation_files),
        },
        "formulation": {
            "id": FORMULATION_ID,
            "result_schema_version": RESULT_SCHEMA_VERSION,
            "path": str(frozen.formulation_path),
            "sha256": frozen.formulation_sha256,
        },
        "instance": {
            "stage": "H6",
            "path": str(frozen.instance_dir),
            "sha256": frozen.instance_sha256,
            "canonical_file_sha256": dict(frozen.instance_files),
            "candidate_h": 6,
        },
        "qualification_gate": _qualification_gate_payload(qualification, frozen),
        "screening_is_exact_by_fixed_cost": dict(frozen.screening_by_fixed_cost),
        "experiment_contract": {
            "mode": "MX",
            "fixed_costs": list(FIXED_COSTS),
            "solver_seeds": list(SEEDS),
            "variants": list(VARIANTS),
            "pair_count": EXPECTED_PAIR_COUNT,
            "slot_count": EXPECTED_SLOT_COUNT,
            "authorized_even_slot_caps": list(ALLOWED_MAX_SLOTS),
            "full_run_slot_cap": EXPECTED_SLOT_COUNT,
            "time_limit_seconds": TIME_LIMIT_SECONDS,
            "threads": 1,
            "worker_count": 1,
            "native_cuts": -1,
            "native_root_cut_passes": ROOT_CUT_PASSES,
            "mip_focus": 1,
            "heuristics": 0.2,
            "maximum_root_cut_rounds": ROOT_CUT_ROUNDS,
            "maximum_cuts_per_round": MAX_CUTS_PER_ROUND,
            "screening_expected_exact": False,
            "pair_order": "solver_seed_major_then_fixed_cost_order",
            "within_pair_rotation": (
                "latin_two_variant_v1: offset=(fixed_cost_rank+seed_rank)%2"
            ),
            "sole_treatment_field": "solver_config.cut_od_disaggregated_return",
            "retain_every_terminal_outcome": True,
            "retain_all_open_incumbents": True,
            "failures_are_not_dropped": True,
            "both_fresh_qualification_cases_required_before_plan": True,
            "qualification_use": "forbidden",
            "formal_use": "forbidden",
        },
        "resource_lock": str(Path(DEFAULT_BENCHMARK_RESOURCE_LOCK).resolve()),
        "controlled_environment": {
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONHASHSEED": "0",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        },
        "planned_jobs": jobs,
    }
    payload["plan_id"] = _plan_id(payload)
    canonical = json.loads(canonical_json(payload))
    if not isinstance(canonical, dict):  # pragma: no cover - payload is an object
        raise H6ODDisaggregatedReturnAblationError(
            "plan did not canonicalize to an object"
        )
    return canonical


def _parse_created_at(value: object) -> str:
    if not isinstance(value, str):
        raise H6ODDisaggregatedReturnAblationError("created_at_utc must be a string")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise H6ODDisaggregatedReturnAblationError(
            "created_at_utc is not ISO-8601"
        ) from error
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise H6ODDisaggregatedReturnAblationError("created_at_utc must use UTC")
    return value


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    source = _real_file(path, label=label)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise H6ODDisaggregatedReturnAblationError(
            f"cannot read {label} {source}: {error}"
        ) from error
    if not isinstance(payload, dict):
        raise H6ODDisaggregatedReturnAblationError(f"{label} must be a JSON object")
    return payload


def _validate_command(job: Mapping[str, Any]) -> None:
    raw = job.get("command")
    if not isinstance(raw, list) or any(not isinstance(item, str) for item in raw):
        raise H6ODDisaggregatedReturnAblationError(
            "planned command must be a string list"
        )
    command = tuple(raw)
    if len(command) < 3:
        raise H6ODDisaggregatedReturnAblationError("planned command is incomplete")
    try:
        parsed = build_main_parser().parse_args(command[2:])
    except SystemExit as error:
        raise H6ODDisaggregatedReturnAblationError(
            "planned command is not accepted by the current main.py parser"
        ) from error
    if parsed.command != "solve":
        raise H6ODDisaggregatedReturnAblationError(
            "planned command is not main.py solve"
        )
    expected_enabled = bool(job["cut_od_disaggregated_return"])
    checks = {
        "threads": parsed.threads == 1,
        "time_limit": parsed.time_limit == TIME_LIMIT_SECONDS,
        "mip_gap": parsed.mip_gap == 0.0,
        "seed": parsed.solver_seed == job["seed"],
        "fixed_cost": parsed.locker_fixed_cost_per_hour == job["fixed_cost"],
        "display_top_k": parsed.display_origin_choices == DISPLAY_ORIGIN_CHOICES,
        "general_top_k": (parsed.general_corridor_choices == GENERAL_CORRIDOR_CHOICES),
        "cuts": parsed.native_cuts == -1,
        "cut_passes": parsed.root_cut_passes == ROOT_CUT_PASSES,
        "mip_focus": parsed.mip_focus == 1,
        "heuristics": parsed.heuristics == 0.2,
        "root_rounds": parsed.max_root_cut_rounds == ROOT_CUT_ROUNDS,
        "round_cap": parsed.max_cuts_per_round == MAX_CUTS_PER_ROUND,
        "od_disaggregated_return": (
            parsed.od_disaggregated_return_cuts is expected_enabled
        ),
        "facility_incident_star_disabled": parsed.od_facility_vub_cuts is False,
        "directed_return_disabled": parsed.directed_return_cuts is False,
        "valid_inequalities": parsed.use_valid_inequalities is True,
        "user_cuts": parsed.no_user_cuts is False,
        "no_start": (
            parsed.start_mode == "none"
            and parsed.network_start_seconds == 0.0
            and parsed.fixed_integer_repair_seconds == 0.0
            and parsed.no_local_branching is True
            and parsed.dev_objective_partial_start is False
        ),
    }
    failures = [name for name, valid in checks.items() if not valid]
    if failures:
        raise H6ODDisaggregatedReturnAblationError(
            "planned command violates the frozen contract: " + ", ".join(failures)
        )


def _validate_plan_structure(payload: Mapping[str, Any]) -> None:
    required = {
        "artifact_type",
        "schema_version",
        "development_only",
        "formal_evidence_eligible",
        "qualification_evidence_eligible",
        "created_at_utc",
        "plan_id",
        "run_root",
        "repository",
        "formulation",
        "instance",
        "qualification_gate",
        "screening_is_exact_by_fixed_cost",
        "experiment_contract",
        "resource_lock",
        "controlled_environment",
        "planned_jobs",
    }
    if set(payload) != required:
        raise H6ODDisaggregatedReturnAblationError("plan top-level fields drifted")
    if (
        payload["artifact_type"] != PLAN_ARTIFACT_TYPE
        or payload["schema_version"] != PLAN_SCHEMA_VERSION
        or payload["development_only"] is not True
        or payload["formal_evidence_eligible"] is not False
        or payload["qualification_evidence_eligible"] is not False
    ):
        raise H6ODDisaggregatedReturnAblationError(
            "plan violates its development-only evidence boundary"
        )
    _parse_created_at(payload["created_at_utc"])
    plan_id = payload["plan_id"]
    if not isinstance(plan_id, str) or _PLAN_ID_RE.fullmatch(plan_id) is None:
        raise H6ODDisaggregatedReturnAblationError("plan_id is not canonical")
    if plan_id != _plan_id(payload):
        raise H6ODDisaggregatedReturnAblationError("plan_id recomputation mismatch")
    screening = payload["screening_is_exact_by_fixed_cost"]
    if screening != {"fc24": False, "fc26": False}:
        raise H6ODDisaggregatedReturnAblationError(
            "plan must retain screening_is_exact=false for both fixed costs"
        )
    contract = payload["experiment_contract"]
    if not isinstance(contract, dict):
        raise H6ODDisaggregatedReturnAblationError(
            "experiment_contract must be an object"
        )
    expected_contract_values = {
        "pair_count": EXPECTED_PAIR_COUNT,
        "slot_count": EXPECTED_SLOT_COUNT,
        "authorized_even_slot_caps": list(ALLOWED_MAX_SLOTS),
        "full_run_slot_cap": EXPECTED_SLOT_COUNT,
        "threads": 1,
        "worker_count": 1,
        "sole_treatment_field": "solver_config.cut_od_disaggregated_return",
        "retain_every_terminal_outcome": True,
        "retain_all_open_incumbents": True,
        "failures_are_not_dropped": True,
        "both_fresh_qualification_cases_required_before_plan": True,
        "qualification_use": "forbidden",
        "formal_use": "forbidden",
    }
    for field, expected in expected_contract_values.items():
        if contract.get(field) != expected:
            raise H6ODDisaggregatedReturnAblationError(
                f"experiment contract field drifted: {field}"
            )
    jobs = payload["planned_jobs"]
    if not isinstance(jobs, list) or len(jobs) != EXPECTED_SLOT_COUNT:
        raise H6ODDisaggregatedReturnAblationError("plan must contain exactly 12 jobs")
    if [job.get("slot_index") for job in jobs] != list(
        range(1, EXPECTED_SLOT_COUNT + 1)
    ):
        raise H6ODDisaggregatedReturnAblationError("slot indexes are not canonical")
    pair_orders: list[tuple[str, str]] = []
    for pair_offset in range(EXPECTED_PAIR_COUNT):
        pair_jobs = jobs[2 * pair_offset : 2 * pair_offset + 2]
        if len({job.get("pair_id") for job in pair_jobs}) != 1:
            raise H6ODDisaggregatedReturnAblationError(
                "each adjacent slot pair must share pair_id"
            )
        variants = tuple(str(job.get("variant")) for job in pair_jobs)
        if variants != _pair_variant_order(pair_offset):
            raise H6ODDisaggregatedReturnAblationError("within-pair rotation drifted")
        pair_orders.append(variants)
        normalized: list[dict[str, Any]] = []
        for job in pair_jobs:
            _validate_command(job)
            solver = job.get("solver_config")
            model = job.get("model_config")
            if not isinstance(solver, dict) or not isinstance(model, dict):
                raise H6ODDisaggregatedReturnAblationError(
                    "job model and solver configs must be objects"
                )
            if canonical_sha256(solver) != job.get("solver_config_sha256"):
                raise H6ODDisaggregatedReturnAblationError(
                    "solver config hash mismatch"
                )
            if canonical_sha256(model) != job.get("model_config_sha256"):
                raise H6ODDisaggregatedReturnAblationError("model config hash mismatch")
            expected_enabled = _od_disaggregated_return_for_variant(str(job["variant"]))
            if (
                job.get("cut_od_disaggregated_return") is not expected_enabled
                or solver.get("cut_od_disaggregated_return") is not expected_enabled
            ):
                raise H6ODDisaggregatedReturnAblationError(
                    "variant od-disaggregated-return treatment is inconsistent"
                )
            normalized_solver = dict(solver)
            normalized_solver.pop("cut_od_disaggregated_return", None)
            shared_solver_sha256 = canonical_sha256(normalized_solver)
            if job.get("shared_solver_config_sha256") != shared_solver_sha256:
                raise H6ODDisaggregatedReturnAblationError(
                    "shared solver config hash mismatch"
                )
            normalized.append(
                {
                    "fixed_cost": job.get("fixed_cost"),
                    "seed": job.get("seed"),
                    "model": model,
                    "solver": normalized_solver,
                    "instance_dir": job.get("instance_dir"),
                }
            )
        if normalized[0] != normalized[1]:
            raise H6ODDisaggregatedReturnAblationError(
                "pair differs beyond solver_config.cut_od_disaggregated_return"
            )
        first_job = pair_jobs[0]
        expected_pair_id = "h6-od-disaggregated-return-pair-" + canonical_sha256(
            {
                "instance_sha256": payload["instance"]["sha256"],
                "model_config_sha256": first_job["model_config_sha256"],
                "shared_solver_config_sha256": first_job["shared_solver_config_sha256"],
                "fixed_cost": first_job["fixed_cost"],
                "seed": first_job["seed"],
                "mode": "MX",
                "time_limit_seconds": TIME_LIMIT_SECONDS,
            }
        )
        if first_job["pair_id"] != expected_pair_id:
            raise H6ODDisaggregatedReturnAblationError("pair_id identity mismatch")
    if (
        pair_orders.count(VARIANTS) != 3
        or pair_orders.count(tuple(reversed(VARIANTS))) != 3
    ):
        raise H6ODDisaggregatedReturnAblationError(
            "variant-first allocation is unbalanced"
        )


def build_plan(
    *,
    repo_root: Path,
    qualification_summary_path: Path,
    plan_path: Path,
    run_root: Path,
    now: Callable[[], datetime] | None = None,
) -> dict[str, Any]:
    """Build one immutable, solver-free plan from a clean committed worktree."""

    frozen = _freeze_inputs(repo_root)
    if (
        not qualification_summary_path.is_file()
        or qualification_summary_path.is_symlink()
    ):
        qualification_root = qualification_summary_path.expanduser().absolute().parent
        qualify_command = (
            str(frozen.python_executable),
            "-m",
            "src.benchmark.branch_cut.h6_od_disaggregated_return_ablation",
            "qualify",
            "--repo-root",
            str(frozen.repo_root),
            "--qualification-root",
            str(qualification_root),
        )
        raise H6ODDisaggregatedReturnAblationError(
            "fresh exact-current-commit qualification is missing; run this first: "
            + shlex.join(qualify_command)
        )
    qualification = verify_qualification_artifact(
        qualification_summary_path, frozen=frozen
    )
    development_root = _real_directory(
        frozen.repo_root / DEVELOPMENT_ROOT_RELATIVE_PATH,
        label="development result root",
    )
    target_plan = _prospective_path(plan_path, label="ablation plan")
    target_run = _prospective_path(run_root, label="ablation run root")
    _require_under(target_plan, development_root, label="ablation plan")
    _require_under(target_run, development_root, label="ablation run root")
    if target_plan == target_run or target_plan.is_relative_to(target_run):
        raise H6ODDisaggregatedReturnAblationError(
            "plan path must be outside the prospective run root"
        )
    clock = now or (lambda: datetime.now(timezone.utc))
    created_at = clock()
    if created_at.tzinfo is None:
        raise H6ODDisaggregatedReturnAblationError("plan clock must be timezone-aware")
    created_at_utc = created_at.astimezone(timezone.utc).isoformat()
    payload = _build_plan_payload(
        frozen,
        target_run,
        qualification,
        created_at_utc=created_at_utc,
    )
    _validate_plan_structure(payload)
    write_canonical_json_exclusive(target_plan, payload)
    return payload


def verify_plan(
    plan_path: Path,
    *,
    expected_plan_id: str,
    require_fresh_run_root: bool = True,
) -> VerifiedPlan:
    """Independently rebuild a plan from authoritative current-state inputs."""

    if _PLAN_ID_RE.fullmatch(expected_plan_id) is None:
        raise H6ODDisaggregatedReturnAblationError("expected plan id is not canonical")
    source = _real_file(plan_path, label="ablation plan")
    payload = _read_json(source, label="ablation plan")
    _validate_plan_structure(payload)
    if payload["plan_id"] != expected_plan_id:
        raise H6ODDisaggregatedReturnAblationError("explicit expected plan id mismatch")
    repository = payload.get("repository")
    if not isinstance(repository, dict) or not isinstance(repository.get("root"), str):
        raise H6ODDisaggregatedReturnAblationError(
            "plan repository contract is invalid"
        )
    frozen = _freeze_inputs(Path(repository["root"]))
    qualification_gate = payload.get("qualification_gate")
    if not isinstance(qualification_gate, Mapping) or not isinstance(
        qualification_gate.get("summary_path"), str
    ):
        raise H6ODDisaggregatedReturnAblationError(
            "plan qualification prerequisite is invalid"
        )
    qualification = verify_qualification_artifact(
        Path(str(qualification_gate["summary_path"])), frozen=frozen
    )
    run_root = Path(str(payload["run_root"])).expanduser().absolute()
    development_root = _real_directory(
        frozen.repo_root / DEVELOPMENT_ROOT_RELATIVE_PATH,
        label="development result root",
    )
    _require_under(run_root, development_root, label="ablation run root")
    if require_fresh_run_root:
        _prospective_path(run_root, label="ablation run root")
    expected = _build_plan_payload(
        frozen,
        run_root,
        qualification,
        created_at_utc=_parse_created_at(payload["created_at_utc"]),
    )
    if payload != expected:
        raise H6ODDisaggregatedReturnAblationError(
            "plan differs from independently reconstructed current-state plan"
        )
    return VerifiedPlan(path=source, payload=payload)


def _process_table() -> str:
    completed = subprocess.run(
        ("ps", "-axo", "pid=,ppid=,command="),
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise H6ODDisaggregatedReturnAblationError(
            "cannot audit concurrent processes: "
            + (completed.stderr or completed.stdout).strip()
        )
    return completed.stdout


_CONFLICTING_PROCESS_PATTERNS = (
    re.compile(r"(?:^|\s)(?:\S*/)?main\.py\s+(?:solve|run)(?:\s|$)"),
    re.compile(r"(?:^|\s)-m\s+main\s+(?:solve|run)(?:\s|$)"),
    re.compile(
        r"(?:^|\s)(?:\S*/)?mlnd-bnc-benchmark\s+"
        r"(?:qualify|run|run-formal)(?:\s|$)"
    ),
    re.compile(
        r"(?:^|\s)-m\s+src\.benchmark\.branch_cut(?:\.cli)?\s+"
        r"(?:qualify|run|run-formal)(?:\s|$)"
    ),
    re.compile(
        r"(?:^|[/\.\s])branch_cut\s+"
        r"(?:qualify|run|run-formal)(?:\s|$)"
    ),
    re.compile(
        r"(?:^|[/\.\s])"
        r"(?:calibration_runner|dev_micro_ce|h6_od_disaggregated_return_ablation|"
        r"mlnd_qualification|objective_start_smoke)(?:\.py)?(?:\s|$)"
    ),
    re.compile(r"(?:^|[/\s])gurobi_cl(?:\s|$)"),
    re.compile(r"(?:^|[/\s])pytest(?:\s|$)"),
)


def _known_conflicting_command(command: str) -> bool:
    return any(
        pattern.search(command) is not None for pattern in _CONFLICTING_PROCESS_PATTERNS
    )


def _conflicting_processes(process_table: str) -> tuple[str, ...]:
    """Return known external benchmark/test processes, excluding this ancestry.

    The shared advisory lock is the authoritative exclusion mechanism for
    participating benchmark runners.  This command-name audit is a best-effort
    guard for known entry points; it cannot identify every arbitrary Python
    process that happens to import ``gurobipy``.
    """

    processes: list[tuple[int, int, str, str]] = []
    parent_by_pid: dict[int, int] = {}
    for raw_line in process_table.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        fields = line.split(maxsplit=2)
        if len(fields) != 3:
            continue
        try:
            pid = int(fields[0])
            ppid = int(fields[1])
        except ValueError:
            continue
        command = fields[2]
        processes.append((pid, ppid, command, line))
        parent_by_pid[pid] = ppid

    excluded_pids: set[int] = set()
    current_pid: int | None = os.getpid()
    while (
        current_pid is not None and current_pid > 0 and current_pid not in excluded_pids
    ):
        excluded_pids.add(current_pid)
        current_pid = parent_by_pid.get(current_pid)

    conflicts: list[str] = []
    for pid, _ppid, command, line in processes:
        if pid not in excluded_pids and _known_conflicting_command(command):
            conflicts.append(line)
    return tuple(conflicts)


def _expected_source_snapshot(frozen: FrozenInputs) -> dict[str, Any]:
    return {
        "repository_commit": frozen.commit,
        "implementation_sha256": frozen.implementation_sha256,
        "formulation_sha256": frozen.formulation_sha256,
        "instance_sha256": frozen.instance_sha256,
        "canonical_file_sha256": dict(frozen.instance_files),
        "screening_is_exact_by_fixed_cost": dict(frozen.screening_by_fixed_cost),
    }


def _qualification_checkpoint_path(root: Path, fixed_cost: float) -> Path:
    return root / "checkpoints" / f"fc{int(fixed_cost):02d}.json"


def run_h6_qualification(
    *,
    repo_root: Path,
    qualification_root: Path,
    qualifier: Callable[..., QualificationRunnerOutcome] = qualify_canonical_instance,
    instance_loader: Callable[[Path, ModelConfig, float], object] = (
        _load_qualification_instance
    ),
    process_table_provider: Callable[[], str] | None = None,
    now: Callable[[], datetime] | None = None,
) -> dict[str, Any]:
    """Run the two algorithm-neutral H6 prerequisite qualifications serially."""

    frozen = _freeze_inputs(repo_root)
    development_root = _real_directory(
        frozen.repo_root / DEVELOPMENT_ROOT_RELATIVE_PATH,
        label="development result root",
    )
    root = _prospective_path(qualification_root, label="H6 qualification root")
    _require_under(root, development_root, label="H6 qualification root")
    identity = _qualification_identity_payload(frozen)
    qualification_id = _qualification_id(identity)
    clock = now or (lambda: datetime.now(timezone.utc))
    process_reader = process_table_provider or _process_table
    settings = _qualification_settings()
    policy = _qualification_policy()
    geometry_policy = _qualification_geometry_policy()
    case_records: list[dict[str, Any]] = []
    process_audits: list[dict[str, Any]] = []
    stop_reason: str | None = None
    started_at_utc = clock().astimezone(timezone.utc).isoformat()
    try:
        with BenchmarkResourceLock(
            Path(DEFAULT_BENCHMARK_RESOURCE_LOCK).resolve(),
            command="h6-current-model-qualification",
        ):
            root.mkdir(parents=False, exist_ok=False)
            (root / "checkpoints").mkdir(parents=False, exist_ok=False)
            write_canonical_json_exclusive(
                root / "run_identity.json",
                {
                    "artifact_type": "mlnd_h6_current_model_qualification_identity",
                    "schema_version": QUALIFICATION_SCHEMA_VERSION,
                    "qualification_id": qualification_id,
                    "identity": identity,
                    "started_at_utc": started_at_utc,
                },
            )
            for fixed_cost in FIXED_COSTS:
                profile, contract = _qualification_contract(frozen, fixed_cost)
                case_id = f"h6-fc{int(fixed_cost):02d}-current-model"
                table = process_reader()
                conflicts = _conflicting_processes(table)
                process_audits.append(
                    {
                        "fixed_cost": fixed_cost,
                        "process_table_sha256": hashlib.sha256(
                            table.encode("utf-8")
                        ).hexdigest(),
                        "conflicting_process_count": len(conflicts),
                        "valid": not conflicts,
                    }
                )
                if conflicts:
                    stop_reason = f"fc{int(fixed_cost):02d}_process_conflict"
                    break
                before = _source_snapshot(frozen)
                checkpoint_path = _qualification_checkpoint_path(root, fixed_cost)
                case_identity = {
                    "case_id": case_id,
                    "scale_label": "small",
                    "fixed_cost": fixed_cost,
                    "profile": profile.to_payload(),
                    "contract": contract.to_payload(),
                }
                try:
                    instance = instance_loader(
                        frozen.instance_dir,
                        profile.effective_model_config(),
                        profile.demand_scale,
                    )
                    _validate_loaded_qualification_instance(instance, profile=profile)
                    outcome = qualifier(
                        instance,
                        "MX",
                        case_id=case_id,
                        scale="small",
                        qualification_run_id=qualification_id,
                        instance_sha256=frozen.instance_sha256,
                        policy=policy,
                        settings=settings,
                        geometry_start_policy=geometry_policy,
                        expected_model_contract=contract,
                        fail_fast_infrastructure=True,
                    )
                    _validate_qualification_outcome(
                        outcome,
                        case_id=case_id,
                        qualification_id=qualification_id,
                        contract=contract,
                    )
                    after = _source_snapshot(frozen)
                    if after != before:
                        raise QualificationInfrastructureError(
                            "qualification changed canonical source state"
                        )
                    checkpoint = {
                        "artifact_type": QUALIFICATION_CHECKPOINT_ARTIFACT_TYPE,
                        "schema_version": QUALIFICATION_SCHEMA_VERSION,
                        "development_only": True,
                        "formal_evidence_eligible": False,
                        "qualification_id": qualification_id,
                        "case": case_identity,
                        "source_before": before,
                        "source_after": after,
                        "status": "completed",
                        "infrastructure_error": None,
                        "outcome": outcome.to_payload(),
                    }
                    write_canonical_json_exclusive(checkpoint_path, checkpoint)
                    checkpoint_sha256 = _sha256_file(checkpoint_path)
                    fresh_accepted = _qualification_record_is_fresh_accepted(
                        checkpoint,
                        contract,
                        expected_case_id=case_id,
                        expected_qualification_id=qualification_id,
                        expected_instance_sha256=frozen.instance_sha256,
                    )
                    case_records.append(
                        {
                            "case_id": case_id,
                            "fixed_cost": fixed_cost,
                            "status": "completed",
                            "accepted": outcome.accepted,
                            "fresh_accepted": fresh_accepted,
                            "checkpoint_path": str(checkpoint_path),
                            "checkpoint_sha256": checkpoint_sha256,
                        }
                    )
                except (MemoryError, KeyboardInterrupt, SystemExit):
                    raise
                except Exception as error:
                    try:
                        after = _source_snapshot(frozen)
                    except Exception as snapshot_error:
                        after = {
                            "snapshot_error": (
                                f"{type(snapshot_error).__name__}: {snapshot_error}"
                            )
                        }
                    checkpoint = {
                        "artifact_type": QUALIFICATION_CHECKPOINT_ARTIFACT_TYPE,
                        "schema_version": QUALIFICATION_SCHEMA_VERSION,
                        "development_only": True,
                        "formal_evidence_eligible": False,
                        "qualification_id": qualification_id,
                        "case": case_identity,
                        "source_before": before,
                        "source_after": after,
                        "status": "infrastructure_failure",
                        "infrastructure_error": {
                            "type": type(error).__name__,
                            "message": str(error),
                        },
                        "outcome": None,
                    }
                    write_canonical_json_exclusive(checkpoint_path, checkpoint)
                    case_records.append(
                        {
                            "case_id": case_id,
                            "fixed_cost": fixed_cost,
                            "status": "infrastructure_failure",
                            "accepted": False,
                            "fresh_accepted": False,
                            "checkpoint_path": str(checkpoint_path),
                            "checkpoint_sha256": _sha256_file(checkpoint_path),
                        }
                    )
                    stop_reason = (
                        f"fc{int(fixed_cost):02d}_qualification_infrastructure_failure"
                    )
                    break

            both_fresh_accepted = len(case_records) == len(FIXED_COSTS) and all(
                record["fresh_accepted"] is True for record in case_records
            )
            if stop_reason is not None:
                status = "failed_closed"
            elif both_fresh_accepted:
                status = "qualified"
            else:
                status = "not_qualified"
            summary = {
                "artifact_type": QUALIFICATION_ARTIFACT_TYPE,
                "schema_version": QUALIFICATION_SCHEMA_VERSION,
                "development_only": True,
                "formal_evidence_eligible": False,
                "eligible_only_as_ablation_prerequisite": True,
                "qualification_id": qualification_id,
                "identity": identity,
                "started_at_utc": started_at_utc,
                "completed_at_utc": clock().astimezone(timezone.utc).isoformat(),
                "status": status,
                "both_fresh_accepted": both_fresh_accepted,
                "stop_reason": stop_reason,
                "case_records": case_records,
                "process_audits": process_audits,
            }
            write_canonical_json_exclusive(root / "summary.json", summary)
            return summary
    except BenchmarkResourceLockError as error:
        raise H6ODDisaggregatedReturnAblationError(str(error)) from error


def verify_qualification_artifact(
    qualification_summary_path: Path,
    *,
    frozen: FrozenInputs,
) -> VerifiedQualification:
    """Verify a fresh H6 gate produced by this exact committed wrapper.

    This candidate changes model implementation files, so qualification from
    any earlier commit or another experiment wrapper is deliberately rejected.
    """

    source = _real_file(qualification_summary_path, label="H6 qualification summary")
    payload = _read_json(source, label="H6 qualification summary")
    required_fields = {
        "artifact_type",
        "schema_version",
        "development_only",
        "formal_evidence_eligible",
        "eligible_only_as_ablation_prerequisite",
        "qualification_id",
        "identity",
        "started_at_utc",
        "completed_at_utc",
        "status",
        "both_fresh_accepted",
        "stop_reason",
        "case_records",
        "process_audits",
    }
    if set(payload) != required_fields:
        raise H6ODDisaggregatedReturnAblationError(
            "H6 qualification summary fields drifted"
        )
    if (
        payload["artifact_type"] != QUALIFICATION_ARTIFACT_TYPE
        or payload["schema_version"] != QUALIFICATION_SCHEMA_VERSION
        or payload["development_only"] is not True
        or payload["formal_evidence_eligible"] is not False
        or payload["eligible_only_as_ablation_prerequisite"] is not True
        or payload["status"] != "qualified"
        or payload["both_fresh_accepted"] is not True
        or payload["stop_reason"] is not None
    ):
        raise H6ODDisaggregatedReturnAblationError(
            "H6 qualification is not an accepted development prerequisite"
        )
    _parse_created_at(payload["started_at_utc"])
    _parse_created_at(payload["completed_at_utc"])
    identity = payload.get("identity")
    if not isinstance(identity, Mapping):
        raise H6ODDisaggregatedReturnAblationError(
            "H6 qualification identity must be an object"
        )
    repository = identity.get("repository")
    formulation = identity.get("formulation")
    instance = identity.get("instance")
    if not all(
        isinstance(value, Mapping) for value in (repository, formulation, instance)
    ):
        raise H6ODDisaggregatedReturnAblationError(
            "H6 qualification source identity is incomplete"
        )
    assert isinstance(repository, Mapping)
    assert isinstance(formulation, Mapping)
    assert isinstance(instance, Mapping)
    source_root_value = repository.get("root")
    source_commit = repository.get("commit")
    source_branch = repository.get("branch")
    if (
        not isinstance(source_root_value, str)
        or not isinstance(source_commit, str)
        or _GIT_SHA1_RE.fullmatch(source_commit) is None
        or not isinstance(source_branch, str)
        or not source_branch.startswith("codex/")
    ):
        raise H6ODDisaggregatedReturnAblationError(
            "H6 qualification repository identity is invalid"
        )
    source_root = _real_directory(
        Path(source_root_value), label="qualification repository root"
    )
    source_development_root = _real_directory(
        source_root / DEVELOPMENT_ROOT_RELATIVE_PATH,
        label="qualification development result root",
    )
    _require_under(source, source_development_root, label="H6 qualification summary")
    if source_commit != frozen.commit:
        raise H6ODDisaggregatedReturnAblationError(
            "H6 qualification must come from the exact current commit"
        )

    expected_identity = _qualification_identity_payload(frozen)
    if (
        repository.get("required_baseline_ancestor") != REQUIRED_BASELINE_COMMIT
        or repository.get("model_implementation_sha256")
        != frozen.model_implementation_sha256
        or repository.get("model_implementation_files")
        != dict(frozen.model_implementation_files)
        or canonical_sha256(repository.get("model_implementation_files"))
        != repository.get("model_implementation_sha256")
        or canonical_sha256(repository.get("qualification_implementation_files"))
        != repository.get("qualification_implementation_sha256")
        or canonical_sha256(repository.get("full_wrapper_implementation_files"))
        != repository.get("full_wrapper_implementation_sha256")
        or repository.get("qualification_implementation_files")
        != dict(frozen.qualification_implementation_files)
        or repository.get("qualification_implementation_sha256")
        != frozen.qualification_implementation_sha256
        or repository.get("full_wrapper_implementation_files")
        != dict(frozen.implementation_files)
        or repository.get("full_wrapper_implementation_sha256")
        != frozen.implementation_sha256
    ):
        raise H6ODDisaggregatedReturnAblationError(
            "H6 qualification model/provenance does not match the current model"
        )
    if identity != expected_identity:
        raise H6ODDisaggregatedReturnAblationError(
            "H6 qualification identity differs from the exact current wrapper"
        )
    if (
        formulation.get("sha256") != frozen.formulation_sha256
        or instance.get("sha256") != frozen.instance_sha256
        or instance.get("canonical_file_sha256") != dict(frozen.instance_files)
    ):
        raise H6ODDisaggregatedReturnAblationError(
            "H6 qualification formulation or instance differs from the current plan"
        )
    qualification_id = payload["qualification_id"]
    if (
        not isinstance(qualification_id, str)
        or _QUALIFICATION_ID_RE.fullmatch(qualification_id) is None
        or qualification_id != _qualification_id(identity)
    ):
        raise H6ODDisaggregatedReturnAblationError(
            "H6 qualification ID does not match its exact identity"
        )
    records = payload["case_records"]
    if not isinstance(records, list) or [
        record.get("fixed_cost") if isinstance(record, Mapping) else None
        for record in records
    ] != list(FIXED_COSTS):
        raise H6ODDisaggregatedReturnAblationError(
            "H6 qualification must contain fc24 then fc26"
        )
    process_audits = payload["process_audits"]
    if (
        not isinstance(process_audits, list)
        or len(process_audits) != len(FIXED_COSTS)
        or any(
            not isinstance(audit, Mapping)
            or set(audit)
            != {
                "fixed_cost",
                "process_table_sha256",
                "conflicting_process_count",
                "valid",
            }
            or audit.get("fixed_cost") != fixed_cost
            or not isinstance(audit.get("process_table_sha256"), str)
            or _SHA256_RE.fullmatch(str(audit["process_table_sha256"])) is None
            or audit.get("conflicting_process_count") != 0
            or audit.get("valid") is not True
            for fixed_cost, audit in zip(FIXED_COSTS, process_audits, strict=True)
        )
    ):
        raise H6ODDisaggregatedReturnAblationError(
            "H6 qualification process audits are incomplete or conflicting"
        )
    expected_snapshot = {
        "repository_commit": source_commit,
        "implementation_sha256": repository["full_wrapper_implementation_sha256"],
        "formulation_sha256": formulation["sha256"],
        "instance_sha256": instance["sha256"],
        "canonical_file_sha256": dict(instance["canonical_file_sha256"]),
        "screening_is_exact_by_fixed_cost": dict(
            identity["screening_is_exact_by_fixed_cost"]
        ),
    }
    checkpoint_root = source.parent / "checkpoints"
    for fixed_cost, record in zip(FIXED_COSTS, records, strict=True):
        if not isinstance(record, Mapping) or set(record) != {
            "case_id",
            "fixed_cost",
            "status",
            "accepted",
            "fresh_accepted",
            "checkpoint_path",
            "checkpoint_sha256",
        }:
            raise H6ODDisaggregatedReturnAblationError(
                "H6 qualification case record must be an object"
            )
        expected_path = _qualification_checkpoint_path(source.parent, fixed_cost)
        checkpoint_path = _real_file(
            Path(str(record.get("checkpoint_path"))),
            label="H6 qualification checkpoint",
        )
        if (
            checkpoint_path != expected_path
            or checkpoint_path.parent != checkpoint_root
        ):
            raise H6ODDisaggregatedReturnAblationError(
                "H6 qualification checkpoint path is not canonical"
            )
        if record.get("checkpoint_sha256") != _sha256_file(checkpoint_path):
            raise H6ODDisaggregatedReturnAblationError(
                "H6 qualification checkpoint hash mismatch"
            )
        checkpoint = _read_json(checkpoint_path, label="H6 qualification checkpoint")
        profile, contract = _qualification_contract(frozen, fixed_cost)
        expected_case = {
            "case_id": f"h6-fc{int(fixed_cost):02d}-current-model",
            "scale_label": "small",
            "fixed_cost": fixed_cost,
            "profile": profile.to_payload(),
            "contract": contract.to_payload(),
        }
        if (
            checkpoint.get("artifact_type") != QUALIFICATION_CHECKPOINT_ARTIFACT_TYPE
            or checkpoint.get("schema_version") != QUALIFICATION_SCHEMA_VERSION
            or checkpoint.get("development_only") is not True
            or checkpoint.get("formal_evidence_eligible") is not False
            or checkpoint.get("qualification_id") != qualification_id
            or checkpoint.get("case") != expected_case
            or checkpoint.get("source_before") != expected_snapshot
            or checkpoint.get("source_after") != expected_snapshot
            or record.get("status") != "completed"
            or record.get("accepted") is not True
            or record.get("fresh_accepted") is not True
            or not _qualification_record_is_fresh_accepted(
                checkpoint,
                contract,
                expected_case_id=expected_case["case_id"],
                expected_qualification_id=qualification_id,
                expected_instance_sha256=frozen.instance_sha256,
            )
        ):
            raise H6ODDisaggregatedReturnAblationError(
                "H6 qualification checkpoint lacks fresh accepted evidence"
            )
    return VerifiedQualification(path=source, payload=payload)


def _controlled_environment(
    plan: Mapping[str, Any], job: Mapping[str, Any]
) -> dict[str, str]:
    environment = dict(os.environ)
    environment.update(
        {
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONHASHSEED": "0",
            "MLND_FORMULATION_SOURCE": str(plan["formulation"]["path"]),
            "MPLCONFIGDIR": str(Path(str(job["cache_dir"])) / "mpl"),
            "XDG_CACHE_HOME": str(Path(str(job["cache_dir"])) / "xdg"),
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        }
    )
    return environment


def _open_binary_exclusive(path: Path) -> Any:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o644)
    return os.fdopen(descriptor, "wb")


def _run_signature_payload(run_config: Mapping[str, Any]) -> dict[str, Any]:
    instance = run_config.get("instance")
    if not isinstance(instance, dict):
        raise H6ODDisaggregatedReturnAblationError(
            "run_config.instance must be an object"
        )
    return {
        "schema_version": run_config.get("schema_version"),
        "formulation_id": run_config.get("formulation_id"),
        "formulation_source": run_config.get("formulation_source"),
        "implementation": run_config.get("implementation"),
        "instance_dir": instance.get("directory"),
        "canonical_file_sha256": instance.get("canonical_file_sha256"),
        "demand_scale": run_config.get("demand_scale"),
        "modes": run_config.get("modes"),
        "cross_mode_start_policy": run_config.get("cross_mode_start_policy"),
        "warm_start_from": run_config.get("warm_start_from"),
        "file_warm_start_policy": run_config.get("file_warm_start_policy"),
        "screening_is_exact": run_config.get("screening_is_exact"),
        "model": run_config.get("model"),
        "model_derived": run_config.get("model_derived"),
        "solver": run_config.get("solver"),
    }


def _normalized_run_config_sha256(run_config: Mapping[str, Any]) -> str:
    normalized = json.loads(json.dumps(run_config, sort_keys=True, allow_nan=False))
    normalized.pop("run_signature_sha256", None)
    solver = normalized.get("solver")
    if not isinstance(solver, dict):
        raise H6ODDisaggregatedReturnAblationError(
            "run_config.solver must be an object"
        )
    solver.pop("cut_od_disaggregated_return", None)
    return canonical_sha256(normalized)


def _close_enough(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=1e-9, abs_tol=1e-7)


def _independent_solution_recomputation(
    mode_dir: Path,
    summary: Mapping[str, Any],
    *,
    candidate_h: int,
) -> tuple[dict[str, Any], dict[str, str]]:
    """Recompute H activity, demand, service, and C0 from independent CSV rows."""

    facility_path = _real_file(
        mode_dir / "open_facilities.csv", label="open-facilities artifact"
    )
    service_path = _real_file(mode_dir / "od_service.csv", label="OD-service artifact")
    artifacts = {
        "MX/open_facilities.csv": _sha256_file(facility_path),
        "MX/od_service.csv": _sha256_file(service_path),
    }
    errors: list[str] = []
    od_rows: list[dict[str, str]] = []
    with service_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {
            "demand_per_hour",
            "drone_realized_per_hour",
            "all_rider_baseline_cost_per_hour",
            "service_conservation_error",
        }
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise H6ODDisaggregatedReturnAblationError(
                "OD-service artifact has an invalid schema"
            )
        od_rows = list(reader)
    try:
        total_demand = math.fsum(float(row["demand_per_hour"]) for row in od_rows)
        realized_service = math.fsum(
            float(row["drone_realized_per_hour"]) for row in od_rows
        )
        courier_only_baseline = math.fsum(
            float(row["all_rider_baseline_cost_per_hour"]) for row in od_rows
        )
        maximum_service_error = max(
            (abs(float(row["service_conservation_error"])) for row in od_rows),
            default=0.0,
        )
    except (KeyError, TypeError, ValueError) as error:
        raise H6ODDisaggregatedReturnAblationError(
            f"OD-service artifact contains an invalid number: {error}"
        ) from error
    if not all(
        math.isfinite(value)
        for value in (
            total_demand,
            realized_service,
            courier_only_baseline,
            maximum_service_error,
        )
    ):
        errors.append("nonfinite_od_solution_recomputation")
    active_threshold = service_tolerance(total_demand)

    facility_rows: list[dict[str, str]] = []
    with facility_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {
            "facility",
            "type",
            "fixed_cost_per_hour",
            "incident_physical_flow_per_hour",
        }
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise H6ODDisaggregatedReturnAblationError(
                "open-facilities artifact has an invalid schema"
            )
        facility_rows = list(reader)
    try:
        open_h = sum(row["type"] == "locker" for row in facility_rows)
        active_h = sum(
            row["type"] == "locker"
            and float(row["incident_physical_flow_per_hour"]) > active_threshold
            for row in facility_rows
        )
        positive_cost_open_facilities = sum(
            float(row["fixed_cost_per_hour"]) > 0.0 for row in facility_rows
        )
    except (KeyError, TypeError, ValueError) as error:
        raise H6ODDisaggregatedReturnAblationError(
            f"open-facilities artifact contains an invalid number: {error}"
        ) from error

    funnel = summary.get("demand_funnel_orders_per_hour")
    if not isinstance(funnel, Mapping):
        errors.append("missing_summary_demand_funnel")
    else:
        comparisons = (
            (total_demand, funnel.get("total_demand"), "total_demand"),
            (
                realized_service,
                funnel.get("total_drone_realized"),
                "realized_service",
            ),
            (
                courier_only_baseline,
                summary.get("courier_only_baseline"),
                "courier_only_baseline",
            ),
        )
        for recomputed, reported, label in comparisons:
            if (
                not isinstance(reported, (int, float))
                or isinstance(reported, bool)
                or not _close_enough(recomputed, float(reported))
            ):
                errors.append(f"recomputed_{label}_mismatch")
    if summary.get("open_lockers") != open_h:
        errors.append("recomputed_open_h_mismatch")
    if summary.get("active_lockers") != active_h:
        errors.append("recomputed_active_h_mismatch")
    if not 0 <= open_h <= candidate_h or not 0 <= active_h <= open_h:
        errors.append("recomputed_h_counts_out_of_range")
    if positive_cost_open_facilities <= 0:
        errors.append("no_recomputed_positive_cost_open_facility")
    if maximum_service_error > max(1e-7, active_threshold):
        errors.append("od_service_conservation_error")

    objective = summary.get("objective")
    costs = summary.get("costs_cny_per_hour")
    if not isinstance(objective, (int, float)) or isinstance(objective, bool):
        errors.append("invalid_reported_objective")
    elif not isinstance(costs, Mapping):
        errors.append("missing_cost_reconciliation")
    else:
        optimized_total = costs.get("optimized_total")
        reconciliation_error = costs.get("objective_reconciliation_error")
        if (
            not isinstance(optimized_total, (int, float))
            or isinstance(optimized_total, bool)
            or not _close_enough(float(objective), float(optimized_total))
        ):
            errors.append("recomputed_objective_total_mismatch")
        if (
            not isinstance(reconciliation_error, (int, float))
            or isinstance(reconciliation_error, bool)
            or abs(float(reconciliation_error)) > 1e-7
        ):
            errors.append("objective_reconciliation_error")

    return (
        {
            "evaluated": True,
            "valid": not errors,
            "errors": errors,
            "candidate_h": candidate_h,
            "open_h": open_h,
            "active_h": active_h,
            "positive_cost_open_facilities": positive_cost_open_facilities,
            "total_demand_per_hour": total_demand,
            "realized_service_per_hour": realized_service,
            "courier_only_baseline": courier_only_baseline,
            "maximum_service_conservation_error": maximum_service_error,
            "active_h_threshold_per_hour": active_threshold,
        },
        artifacts,
    )


def _inspect_job(
    plan: Mapping[str, Any],
    job: Mapping[str, Any],
    process_record: Mapping[str, Any],
) -> dict[str, Any]:
    errors: list[str] = []
    artifact_file_sha256: dict[str, str] = {}
    run_config: dict[str, Any] | None = None
    summary: dict[str, Any] | None = None
    results_dir = Path(str(job["results_dir"]))
    run_config_path = results_dir / "run_config.json"
    summary_path = results_dir / "MX" / "summary.json"
    if run_config_path.is_file() and not run_config_path.is_symlink():
        artifact_file_sha256["run_config.json"] = _sha256_file(run_config_path)
        try:
            run_config = _read_json(run_config_path, label="run config")
        except H6ODDisaggregatedReturnAblationError as error:
            errors.append(f"run_config:{error}")
    else:
        errors.append("run_config_missing_or_unsafe")
    if summary_path.is_file() and not summary_path.is_symlink():
        artifact_file_sha256["MX/summary.json"] = _sha256_file(summary_path)
        try:
            summary = _read_json(summary_path, label="MX summary")
        except H6ODDisaggregatedReturnAblationError as error:
            errors.append(f"summary:{error}")
    elif process_record.get("returncode") == 0:
        errors.append("successful_process_missing_summary")

    if run_config is not None:
        repository = plan["repository"]
        formulation = plan["formulation"]
        instance = plan["instance"]
        expected = {
            "schema_version": RESULT_SCHEMA_VERSION,
            "formulation_id": FORMULATION_ID,
            "demand_scale": 1.0,
            "modes": ["MX"],
            "cross_mode_start_policy": None,
            "warm_start_from": None,
            "file_warm_start_policy": "first_mode_only",
            "screening_is_exact": {"MX": False},
            "model": job["model_config"],
            "solver": job["solver_config"],
        }
        for key, value in expected.items():
            if run_config.get(key) != value:
                errors.append(f"run_config.{key}")
        if run_config.get("formulation_source") != {
            "path": formulation["path"],
            "sha256": formulation["sha256"],
        }:
            errors.append("run_config.formulation_source")
        main_files = {
            key: value
            for key, value in repository["implementation_files"].items()
            if key in {path.as_posix() for path in _MODEL_IMPLEMENTATION_RELATIVE_PATHS}
        }
        if run_config.get("implementation") != {
            "sha256": canonical_sha256(main_files),
            "files": main_files,
        }:
            errors.append("run_config.implementation")
        run_instance = run_config.get("instance")
        if not isinstance(run_instance, dict):
            errors.append("run_config.instance")
        else:
            if run_instance.get("directory") != instance["path"]:
                errors.append("run_config.instance.directory")
            if (
                run_instance.get("canonical_file_sha256")
                != instance["canonical_file_sha256"]
            ):
                errors.append("run_config.instance.files")
        signature = run_config.get("run_signature_sha256")
        if signature != canonical_sha256(_run_signature_payload(run_config)):
            errors.append("run_config.run_signature_sha256")

    if summary is not None:
        if summary.get("schema_version") != RESULT_SCHEMA_VERSION:
            errors.append("summary.schema_version")
        if summary.get("formulation_id") != FORMULATION_ID:
            errors.append("summary.formulation_id")
        if run_config is not None and summary.get(
            "run_signature_sha256"
        ) != run_config.get("run_signature_sha256"):
            errors.append("summary.run_signature_sha256")
        if summary.get("screening_is_exact") is not False:
            errors.append("summary.screening_is_exact")
        summary_instance = summary.get("instance")
        if (
            not isinstance(summary_instance, dict)
            or summary_instance.get("screening_is_exact") is not False
        ):
            errors.append("summary.instance.screening_is_exact")
        solver_metrics = summary.get("solver")
        if not isinstance(solver_metrics, dict):
            errors.append("summary.solver")
        else:
            for top_field, solver_field in (
                ("status", "status"),
                ("status_name", "status_name"),
                ("objective", "objective_cost_per_hour"),
                ("objective_bound", "objective_bound"),
                ("mip_gap", "mip_gap"),
                ("runtime_seconds", "runtime_seconds"),
            ):
                top_value = summary.get(top_field)
                solver_value = solver_metrics.get(solver_field)
                if isinstance(top_value, (int, float)) and isinstance(
                    solver_value, (int, float)
                ):
                    if not math.isclose(
                        float(top_value),
                        float(solver_value),
                        rel_tol=1e-10,
                        abs_tol=1e-8,
                    ):
                        errors.append(f"summary_solver_mismatch.{top_field}")
                elif top_value != solver_value:
                    errors.append(f"summary_solver_mismatch.{top_field}")
            objective = summary.get("objective")
            bound = summary.get("objective_bound")
            gap = summary.get("mip_gap")
            if (
                not isinstance(objective, (int, float))
                or isinstance(objective, bool)
                or not math.isfinite(float(objective))
            ):
                errors.append("invalid_objective")
            elif bound is None and gap is None:
                # A time-limited solve may have an incumbent before Gurobi has
                # established any finite dual bound.  Preserve that algorithmic
                # outcome; it simply cannot enter the terminal-gap gate.
                pass
            elif all(
                isinstance(value, (int, float)) and not isinstance(value, bool)
                for value in (bound, gap)
            ):
                objective_value = float(objective)
                bound_value = float(bound)
                gap_value = float(gap)
                if not all(
                    math.isfinite(value)
                    for value in (objective_value, bound_value, gap_value)
                ):
                    errors.append("nonfinite_objective_bound_or_gap")
                else:
                    tolerance = 1e-8 * max(1.0, abs(objective_value), abs(bound_value))
                    if bound_value > objective_value + tolerance:
                        errors.append("minimization_bound_exceeds_incumbent")
                    recomputed_gap = abs(objective_value - bound_value) / max(
                        abs(objective_value), 1e-10
                    )
                    if not math.isclose(
                        gap_value, recomputed_gap, rel_tol=1e-8, abs_tol=1e-10
                    ):
                        errors.append("reported_gap_recomputation_mismatch")
            else:
                errors.append("bound_and_gap_presence_mismatch")

    log_path = Path(str(job["driver_log"]))
    no_incumbent_status: int | None = None
    if log_path.is_file() and not log_path.is_symlink():
        artifact_file_sha256["driver_log"] = _sha256_file(log_path)
        tail = log_path.read_text(encoding="utf-8", errors="replace")[-131_072:]
        matches = tuple(_NO_INCUMBENT_RE.finditer(tail))
        if matches:
            no_incumbent_status = int(matches[-1].group("status"))
    else:
        errors.append("driver_log_missing_or_unsafe")

    mode_dir = results_dir / "MX"
    solver_log_path = mode_dir / "solver.log"
    incumbent_path = mode_dir / "incumbent.sol"
    for label, artifact_path, required in (
        ("MX/solver.log", solver_log_path, summary is not None),
        ("MX/incumbent.sol", incumbent_path, summary is not None),
    ):
        if artifact_path.is_file() and not artifact_path.is_symlink():
            if artifact_path.stat().st_size <= 0:
                errors.append(f"{label}_empty")
            else:
                artifact_file_sha256[label] = _sha256_file(artifact_path)
        elif required:
            errors.append(f"{label}_missing_or_unsafe")

    independent_solution_recomputation: dict[str, Any] = {
        "evaluated": False,
        "valid": summary is None,
        "errors": [],
    }
    if summary is not None:
        try:
            independent_solution_recomputation, recomputed_artifacts = (
                _independent_solution_recomputation(
                    mode_dir,
                    summary,
                    candidate_h=int(plan["instance"]["candidate_h"]),
                )
            )
            artifact_file_sha256.update(recomputed_artifacts)
            errors.extend(
                "independent_solution_recomputation." + str(error)
                for error in independent_solution_recomputation["errors"]
            )
        except (OSError, UnicodeError, H6ODDisaggregatedReturnAblationError) as error:
            independent_solution_recomputation = {
                "evaluated": True,
                "valid": False,
                "errors": [str(error)],
            }
            errors.append(f"independent_solution_recomputation:{error}")

    returncode = process_record.get("returncode")
    if returncode != 0 and no_incumbent_status is None:
        errors.append("subprocess_failed_without_recognized_no_incumbent")

    if summary is not None:
        outcome = "incumbent"
    elif no_incumbent_status is not None:
        outcome = "no_incumbent"
    elif process_record.get("returncode") not in (0, None):
        outcome = "process_failure"
    else:
        outcome = "infrastructure_failure"

    candidate_h = int(plan["instance"]["candidate_h"])
    open_h = summary.get("open_lockers") if summary is not None else None
    all_h_open = (
        isinstance(open_h, int)
        and not isinstance(open_h, bool)
        and open_h == candidate_h
    )
    algorithm = None
    solver_metrics_payload: Mapping[str, Any] | None = None
    if summary is not None:
        solver_summary = summary.get("solver")
        if isinstance(solver_summary, dict):
            solver_metrics_payload = solver_summary
            possible = solver_summary.get("algorithm")
            if isinstance(possible, dict):
                algorithm = possible
    if isinstance(algorithm, dict):
        family_maps = (
            "root_user_cut_separation_seconds",
            "root_user_cut_candidate_counts",
            "root_user_cut_counts",
            "root_user_cut_duplicate_counts",
            "root_user_cut_rejection_counts",
        )
        if job["cut_od_disaggregated_return"] is True:
            separation = algorithm.get("root_user_cut_separation_seconds")
            if (
                not isinstance(separation, dict)
                or "od_disaggregated_directed_return" not in separation
            ):
                errors.append(
                    "treatment.od_disaggregated_return_separator_not_observed"
                )
        else:
            for field in family_maps:
                values = algorithm.get(field, {})
                if (
                    isinstance(values, dict)
                    and "od_disaggregated_directed_return" in values
                ):
                    errors.append(
                        f"treatment.baseline_contains_od_disaggregated_return_{field}"
                    )
    elif summary is not None:
        errors.append("summary.solver.algorithm")

    od_disaggregated_return_telemetry: dict[str, Any] | None = None
    if isinstance(algorithm, dict):
        od_disaggregated_return_telemetry = {
            field: (
                algorithm.get(field, {}).get("od_disaggregated_directed_return")
                if isinstance(algorithm.get(field, {}), dict)
                else None
            )
            for field in (
                "root_user_cut_separation_seconds",
                "root_user_cut_candidate_counts",
                "root_user_cut_counts",
                "root_user_cut_duplicate_counts",
                "root_user_cut_rejection_counts",
            )
        }

    comparison_reasons: list[str] = []
    courier_only_baseline: float | None = None
    realized_service: float | None = None
    active_h: int | None = None
    if summary is None:
        comparison_reasons.append("no_terminal_incumbent_summary")
    else:
        if summary.get("network_is_nontrivial") is not True:
            comparison_reasons.append("zero_physical_network_service")
        if all_h_open:
            comparison_reasons.append("all_h_open")
        if not isinstance(open_h, int) or isinstance(open_h, bool):
            comparison_reasons.append("invalid_open_h_count")
        else:
            open_fraction = open_h / candidate_h
            if not 0.15 <= open_fraction <= 0.85:
                comparison_reasons.append("h_open_fraction_outside_development_gate")
        raw_active_h = summary.get("active_lockers")
        if (
            isinstance(raw_active_h, int)
            and not isinstance(raw_active_h, bool)
            and 0 <= raw_active_h <= candidate_h
        ):
            active_h = raw_active_h
        else:
            comparison_reasons.append("invalid_active_h_count")
        if active_h is not None and active_h < 1:
            comparison_reasons.append("no_active_h")
        raw_baseline = summary.get("courier_only_baseline")
        raw_objective = summary.get("objective")
        if (
            isinstance(raw_baseline, (int, float))
            and not isinstance(raw_baseline, bool)
            and math.isfinite(float(raw_baseline))
            and float(raw_baseline) > 0.0
        ):
            courier_only_baseline = float(raw_baseline)
        else:
            comparison_reasons.append("invalid_courier_only_baseline")
        if (
            courier_only_baseline is not None
            and isinstance(raw_objective, (int, float))
            and not isinstance(raw_objective, bool)
            and math.isfinite(float(raw_objective))
        ):
            savings = courier_only_baseline - float(raw_objective)
            required_savings = max(
                1.0,
                1e-4 * abs(courier_only_baseline),
                100.0 * 1e-6 * max(1.0, abs(courier_only_baseline)),
                1e-3 * courier_only_baseline,
            )
            if savings + 1e-9 < required_savings:
                comparison_reasons.append(
                    "objective_improvement_below_development_gate"
                )
        elif courier_only_baseline is not None:
            comparison_reasons.append("invalid_incumbent_objective")
        funnel = summary.get("demand_funnel_orders_per_hour")
        if isinstance(funnel, dict):
            raw_demand = funnel.get("total_demand")
            raw_service = funnel.get("total_drone_realized")
            if (
                isinstance(raw_demand, (int, float))
                and not isinstance(raw_demand, bool)
                and math.isfinite(float(raw_demand))
                and float(raw_demand) > 0.0
                and isinstance(raw_service, (int, float))
                and not isinstance(raw_service, bool)
                and math.isfinite(float(raw_service))
            ):
                total_demand = float(raw_demand)
                realized_service = float(raw_service)
                required_service = max(1e-3, 1e-5 * total_demand, 1e-3 * total_demand)
                if realized_service + 1e-9 < required_service:
                    comparison_reasons.append("service_below_development_gate")
            else:
                comparison_reasons.append("invalid_demand_or_service")
        else:
            comparison_reasons.append("missing_demand_funnel")
    normalized_run_config_sha256: str | None = None
    if run_config is not None:
        try:
            normalized_run_config_sha256 = _normalized_run_config_sha256(run_config)
        except (TypeError, ValueError, H6ODDisaggregatedReturnAblationError) as error:
            errors.append(f"run_config.normalization:{error}")

    return {
        **dict(process_record),
        "infrastructure_valid": not errors,
        "infrastructure_errors": errors,
        "artifact_file_sha256": artifact_file_sha256,
        "outcome": outcome,
        "no_incumbent_status": no_incumbent_status,
        "normalized_run_config_sha256": normalized_run_config_sha256,
        "status": summary.get("status") if summary is not None else no_incumbent_status,
        "status_name": summary.get("status_name") if summary is not None else None,
        "objective": summary.get("objective") if summary is not None else None,
        "objective_bound": (
            summary.get("objective_bound") if summary is not None else None
        ),
        "mip_gap": summary.get("mip_gap") if summary is not None else None,
        "runtime_seconds": (
            summary.get("runtime_seconds") if summary is not None else None
        ),
        "first_incumbent_time_seconds": (
            solver_metrics_payload.get("first_incumbent_time_seconds")
            if solver_metrics_payload is not None
            else None
        ),
        "first_incumbent_objective": (
            solver_metrics_payload.get("first_incumbent_objective")
            if solver_metrics_payload is not None
            else None
        ),
        "node_count": (
            solver_metrics_payload.get("node_count")
            if solver_metrics_payload is not None
            else None
        ),
        "open_h": open_h,
        "candidate_h": candidate_h,
        "active_h": active_h,
        "all_h_open": all_h_open,
        "courier_only_baseline": courier_only_baseline,
        "realized_service_per_hour": realized_service,
        "development_comparison_eligible": not comparison_reasons,
        "development_comparison_ineligibility_reasons": comparison_reasons,
        "network_is_nontrivial": (
            summary.get("network_is_nontrivial") if summary is not None else None
        ),
        "root_user_cut_counts": (
            algorithm.get("root_user_cut_counts")
            if isinstance(algorithm, dict)
            else None
        ),
        "root_user_cut_candidate_counts": (
            algorithm.get("root_user_cut_candidate_counts")
            if isinstance(algorithm, dict)
            else None
        ),
        "root_user_cut_duplicate_counts": (
            algorithm.get("root_user_cut_duplicate_counts")
            if isinstance(algorithm, dict)
            else None
        ),
        "root_user_cut_rejection_counts": (
            algorithm.get("root_user_cut_rejection_counts")
            if isinstance(algorithm, dict)
            else None
        ),
        "root_user_cut_separation_seconds": (
            algorithm.get("root_user_cut_separation_seconds")
            if isinstance(algorithm, dict)
            else None
        ),
        "od_disaggregated_return_cut_telemetry": (od_disaggregated_return_telemetry),
        "independent_solution_recomputation": independent_solution_recomputation,
    }


def _run_one_job(
    plan: Mapping[str, Any],
    job: Mapping[str, Any],
    process_runner: ProcessRunner,
) -> dict[str, Any]:
    results_dir = Path(str(job["results_dir"]))
    driver_log = Path(str(job["driver_log"]))
    cache_dir = Path(str(job["cache_dir"]))
    if results_dir.exists() or results_dir.is_symlink():
        return {
            "job_id": job["job_id"],
            "pair_id": job["pair_id"],
            "variant": job["variant"],
            "returncode": None,
            "error": "results_directory_already_exists",
        }
    driver_log.parent.mkdir(parents=True, exist_ok=True)
    results_dir.parent.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=False)
    started = datetime.now(timezone.utc).isoformat()
    command = tuple(str(item) for item in job["command"])
    try:
        with _open_binary_exclusive(driver_log) as handle:
            completed = process_runner(
                command,
                cwd=Path(str(plan["repository"]["root"])),
                env=_controlled_environment(plan, job),
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=False,
            )
        returncode = int(completed.returncode)
        error = None if returncode == 0 else f"subprocess_returncode_{returncode}"
    except (MemoryError, KeyboardInterrupt, SystemExit):
        raise
    except Exception as exc:
        returncode = None
        error = f"{type(exc).__name__}: {exc}"
    return {
        "job_id": job["job_id"],
        "slot_index": job["slot_index"],
        "pair_index": job["pair_index"],
        "pair_id": job["pair_id"],
        "fixed_cost": job["fixed_cost"],
        "seed": job["seed"],
        "variant": job["variant"],
        "cut_od_disaggregated_return": job["cut_od_disaggregated_return"],
        "returncode": returncode,
        "error": error,
        "started_at_utc": started,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": list(command),
        "results_dir": str(results_dir),
        "driver_log": str(driver_log),
    }


def _pair_audit(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if len(records) != 2:
        raise H6ODDisaggregatedReturnAblationError(
            "pair audit requires exactly two records"
        )
    variants = {str(record["variant"]): record for record in records}
    if set(variants) != set(VARIANTS):
        raise H6ODDisaggregatedReturnAblationError("pair audit variants are incomplete")
    baseline = variants["C-current"]
    enabled = variants["C+OD-return"]
    only_treatment = baseline.get(
        "normalized_run_config_sha256"
    ) is not None and baseline.get("normalized_run_config_sha256") == enabled.get(
        "normalized_run_config_sha256"
    )
    individual_infrastructure_valid = all(
        record.get("infrastructure_valid") is True for record in records
    )
    infrastructure_valid = individual_infrastructure_valid and only_treatment
    individual_comparison_eligible = all(
        record.get("development_comparison_eligible") is True for record in records
    )
    development_comparison_eligible = (
        individual_comparison_eligible and infrastructure_valid and only_treatment
    )
    return {
        "pair_id": records[0]["pair_id"],
        "fixed_cost": records[0]["fixed_cost"],
        "seed": records[0]["seed"],
        "execution_order": [record["variant"] for record in records],
        "infrastructure_valid": infrastructure_valid,
        "only_difference_is_cut_od_disaggregated_return": only_treatment,
        "development_comparison_eligible": development_comparison_eligible,
        "all_open_variants": [
            record["variant"] for record in records if record.get("all_h_open") is True
        ],
        "outcomes": {record["variant"]: record.get("outcome") for record in records},
        "terminal_mip_gaps": {
            record["variant"]: record.get("mip_gap") for record in records
        },
    }


def _finite_nonnegative(value: object) -> float | None:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or float(value) < 0.0
    ):
        return None
    return float(value)


def _unshifted_geometric_mean_gap_ratio(
    paired_gaps: Sequence[tuple[float, float]],
) -> float | None:
    """Return treatment/current geometric-mean ratio without adding a shift."""

    if not paired_gaps:
        return None
    log_ratios: list[float] = []
    for current_gap, treatment_gap in paired_gaps:
        if current_gap == 0.0:
            if treatment_gap == 0.0:
                log_ratios.append(0.0)
                continue
            return math.inf
        if treatment_gap == 0.0:
            return 0.0
        log_ratios.append(math.log(treatment_gap) - math.log(current_gap))
    return math.exp(math.fsum(log_ratios) / len(log_ratios))


def _development_advance_gate(
    records: Sequence[Mapping[str, Any]],
    pairs: Sequence[Mapping[str, Any]],
    *,
    complete_twelve_slot_plan: bool,
    stop_reason: str | None,
) -> dict[str, Any]:
    """Recompute the preregistered H6 decision from retained terminal records."""

    records_by_pair: dict[str, dict[str, Mapping[str, Any]]] = {}
    for record in records:
        records_by_pair.setdefault(str(record.get("pair_id")), {})[
            str(record.get("variant"))
        ] = record

    eligible_pairs = [
        pair for pair in pairs if pair.get("development_comparison_eligible") is True
    ]
    paired_gaps: list[tuple[float, float]] = []
    gap_win_pair_ids: list[str] = []
    for pair in eligible_pairs:
        pair_id = str(pair["pair_id"])
        variants = records_by_pair.get(pair_id, {})
        current_gap = _finite_nonnegative(variants.get("C-current", {}).get("mip_gap"))
        treatment_gap = _finite_nonnegative(
            variants.get("C+OD-return", {}).get("mip_gap")
        )
        if current_gap is None or treatment_gap is None:
            continue
        paired_gaps.append((current_gap, treatment_gap))
        if treatment_gap + 1e-12 < current_gap:
            gap_win_pair_ids.append(pair_id)

    active_treatment_job_ids: list[str] = []
    for record in records:
        if record.get("variant") != "C+OD-return":
            continue
        cut_counts = record.get("root_user_cut_counts")
        added = (
            cut_counts.get("od_disaggregated_directed_return")
            if isinstance(cut_counts, Mapping)
            else None
        )
        if _finite_nonnegative(added) not in (None, 0.0):
            active_treatment_job_ids.append(str(record["job_id"]))

    geometric_ratio = _unshifted_geometric_mean_gap_ratio(paired_gaps)
    no_consistency_failures = all(
        record.get("infrastructure_valid") is True for record in records
    )
    checks = {
        "complete_twelve_slot_plan": complete_twelve_slot_plan,
        "no_stop_reason": stop_reason is None,
        "no_feasibility_or_metric_consistency_failure": no_consistency_failures,
        "minimum_eligible_pairs": len(eligible_pairs) >= ADVANCE_MIN_ELIGIBLE_PAIRS,
        "minimum_active_treatment_runs": (
            len(active_treatment_job_ids) >= ADVANCE_MIN_ACTIVE_TREATMENT_RUNS
        ),
        "all_eligible_pairs_have_finite_terminal_gaps": (
            len(paired_gaps) == len(eligible_pairs)
        ),
        "minimum_terminal_gap_wins": len(gap_win_pair_ids) >= ADVANCE_MIN_GAP_WINS,
        "geometric_mean_gap_ratio": (
            geometric_ratio is not None
            and math.isfinite(geometric_ratio)
            and geometric_ratio <= ADVANCE_MAX_GEOMETRIC_MEAN_GAP_RATIO
        ),
    }
    failed_checks = [name for name, passed in checks.items() if not passed]
    return {
        "preregistered": True,
        "development_only": True,
        "formal_promotion_authorized": False,
        "passed": not failed_checks,
        "failed_checks": failed_checks,
        "thresholds": {
            "minimum_eligible_pairs": ADVANCE_MIN_ELIGIBLE_PAIRS,
            "minimum_active_treatment_runs": ADVANCE_MIN_ACTIVE_TREATMENT_RUNS,
            "minimum_terminal_gap_wins": ADVANCE_MIN_GAP_WINS,
            "maximum_unshifted_geometric_mean_gap_ratio": (
                ADVANCE_MAX_GEOMETRIC_MEAN_GAP_RATIO
            ),
        },
        "checks": checks,
        "eligible_pair_count": len(eligible_pairs),
        "eligible_pair_ids": [str(pair["pair_id"]) for pair in eligible_pairs],
        "finite_gap_pair_count": len(paired_gaps),
        "gap_win_count": len(gap_win_pair_ids),
        "gap_win_pair_ids": gap_win_pair_ids,
        "active_treatment_run_count": len(active_treatment_job_ids),
        "active_treatment_job_ids": active_treatment_job_ids,
        "unshifted_geometric_mean_treatment_to_current_gap_ratio": geometric_ratio,
    }


def run_plan(
    plan_path: Path,
    *,
    expected_plan_id: str,
    max_slots: int,
    process_runner: ProcessRunner = subprocess.run,
    process_table_provider: Callable[[], str] = _process_table,
) -> dict[str, Any]:
    """Execute an explicitly capped prefix as complete serial matched pairs."""

    if max_slots not in ALLOWED_MAX_SLOTS:
        raise H6ODDisaggregatedReturnAblationError(
            f"max_slots must be one of {ALLOWED_MAX_SLOTS}; odd partial pairs are forbidden"
        )
    verified = verify_plan(
        plan_path,
        expected_plan_id=expected_plan_id,
        require_fresh_run_root=True,
    )
    plan = verified.payload
    run_root = verified.run_root
    lock_path = Path(str(plan["resource_lock"])).resolve()
    if lock_path != Path(DEFAULT_BENCHMARK_RESOURCE_LOCK).resolve():
        raise H6ODDisaggregatedReturnAblationError(
            "plan does not use the shared benchmark resource lock"
        )
    try:
        with BenchmarkResourceLock(
            lock_path, command="h6-od-disaggregated-return-ablation"
        ):
            initial_table = process_table_provider()
            initial_conflicts = _conflicting_processes(initial_table)
            if initial_conflicts:
                raise H6ODDisaggregatedReturnAblationError(
                    "conflicting solver/test processes are active: "
                    + " | ".join(initial_conflicts)
                )
            run_root.mkdir(parents=False, exist_ok=False)
            (run_root / "checkpoints").mkdir(parents=False, exist_ok=False)
            write_canonical_json_exclusive(run_root / "plan.json", dict(plan))
            write_canonical_json_exclusive(
                run_root / "run_identity.json",
                {
                    "artifact_type": "mlnd_h6_od_disaggregated_return_ablation_run_identity",
                    "schema_version": 1,
                    "development_only": True,
                    "formal_evidence_eligible": False,
                    "qualification_evidence_eligible": False,
                    "plan_id": verified.plan_id,
                    "max_slots": max_slots,
                    "worker_count": 1,
                    "runtime_environment": {
                        "python_executable": sys.executable,
                        "python_version": platform.python_version(),
                        "platform": platform.platform(),
                    },
                    "started_at_utc": datetime.now(timezone.utc).isoformat(),
                    "initial_process_table_sha256": hashlib.sha256(
                        initial_table.encode("utf-8")
                    ).hexdigest(),
                },
            )
            selected = list(plan["planned_jobs"][:max_slots])
            records: list[dict[str, Any]] = []
            pairs: list[dict[str, Any]] = []
            process_audits: list[dict[str, Any]] = []
            stop_reason: str | None = None
            for offset in range(0, len(selected), 2):
                pair_jobs = selected[offset : offset + 2]
                pair_records: list[dict[str, Any]] = []
                for job in pair_jobs:
                    table = process_table_provider()
                    conflicts = _conflicting_processes(table)
                    process_audit = {
                        "slot_index": job["slot_index"],
                        "pair_id": job["pair_id"],
                        "variant": job["variant"],
                        "process_table_sha256": hashlib.sha256(
                            table.encode("utf-8")
                        ).hexdigest(),
                        "conflicting_processes": list(conflicts),
                        "valid": not conflicts,
                    }
                    process_audits.append(process_audit)
                    if conflicts:
                        stop_reason = "concurrent_process_conflict"
                        break
                    process_record = _run_one_job(plan, job, process_runner)
                    inspected = _inspect_job(plan, job, process_record)
                    records.append(inspected)
                    pair_records.append(inspected)
                    write_canonical_json_exclusive(
                        run_root / "checkpoints" / f"slot-{job['slot_index']:02d}.json",
                        {
                            "artifact_type": (
                                "mlnd_h6_od_disaggregated_return_ablation_slot_checkpoint"
                            ),
                            "schema_version": 1,
                            "development_only": True,
                            "plan_id": verified.plan_id,
                            "record": inspected,
                        },
                    )
                    if not inspected["infrastructure_valid"]:
                        stop_reason = f"slot_{job['slot_index']}_infrastructure_failure"
                        break
                if len(pair_records) == 2:
                    pair = _pair_audit(pair_records)
                    pairs.append(pair)
                    if not pair["infrastructure_valid"]:
                        stop_reason = "pair_contract_mismatch"
                if stop_reason is not None:
                    break

            executed_ids = {record["job_id"] for record in records}
            skipped = [
                job["job_id"]
                for job in plan["planned_jobs"]
                if job["job_id"] not in executed_ids
            ]
            completed_requested_slots = len(records) == max_slots
            if stop_reason is not None or not completed_requested_slots:
                status = "failed_closed"
            elif max_slots == EXPECTED_SLOT_COUNT:
                status = "completed_full"
            else:
                status = "completed_partial"
            advance_gate = _development_advance_gate(
                records,
                pairs,
                complete_twelve_slot_plan=(
                    max_slots == EXPECTED_SLOT_COUNT and completed_requested_slots
                ),
                stop_reason=stop_reason,
            )
            summary = {
                "artifact_type": SUMMARY_ARTIFACT_TYPE,
                "schema_version": SUMMARY_SCHEMA_VERSION,
                "development_only": True,
                "formal_evidence_eligible": False,
                "qualification_evidence_eligible": False,
                "plan_id": verified.plan_id,
                "max_slots": max_slots,
                "complete_twelve_slot_plan_requested": max_slots == EXPECTED_SLOT_COUNT,
                "completed_requested_slots": completed_requested_slots,
                "status": status,
                "stop_reason": stop_reason,
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                "job_records": records,
                "pair_audits": pairs,
                "development_comparison_eligible_pair_count": sum(
                    pair.get("development_comparison_eligible") is True
                    for pair in pairs
                ),
                "all_completed_pairs_development_comparison_eligible": bool(pairs)
                and all(
                    pair.get("development_comparison_eligible") is True
                    for pair in pairs
                ),
                "development_advance_gate": advance_gate,
                "process_audits": process_audits,
                "skipped_planned_job_ids": skipped,
                "all_open_job_ids_retained": [
                    record["job_id"]
                    for record in records
                    if record.get("all_h_open") is True
                ],
                "failed_job_ids_retained": [
                    record["job_id"]
                    for record in records
                    if record.get("infrastructure_valid") is not True
                    or record.get("outcome")
                    in {"no_incumbent", "process_failure", "infrastructure_failure"}
                ],
            }
            write_canonical_json_exclusive(run_root / "summary.json", summary)
            return summary
    except BenchmarkResourceLockError as error:
        raise H6ODDisaggregatedReturnAblationError(str(error)) from error


def build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build, verify, or serially run the development-only H6 "
            "od-disaggregated-return single-factor ablation."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    qualify = subparsers.add_parser(
        "qualify",
        help="run the two-case algorithm-neutral H6 prerequisite",
    )
    qualify.add_argument("--repo-root", default=str(MAIN_REPO_ROOT))
    qualify.add_argument("--qualification-root", required=True)
    freeze = subparsers.add_parser(
        "freeze", help="freeze an immutable solver-free plan after qualification"
    )
    freeze.add_argument("--repo-root", default=str(MAIN_REPO_ROOT))
    freeze.add_argument("--qualification-summary", required=True)
    freeze.add_argument("--plan-path", required=True)
    freeze.add_argument("--run-root", required=True)
    verify = subparsers.add_parser("verify", help="verify a frozen plan")
    verify.add_argument("--plan-path", required=True)
    verify.add_argument("--expected-plan-id", required=True)
    run = subparsers.add_parser("run", help="run an explicitly capped plan prefix")
    run.add_argument("--plan-path", required=True)
    run.add_argument("--expected-plan-id", required=True)
    run.add_argument(
        "--max-slots",
        type=int,
        required=True,
        choices=ALLOWED_MAX_SLOTS,
        help="explicit even slot cap; 12 requests the complete preregistered ablation",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_cli_parser().parse_args(argv)
    try:
        if args.command == "qualify":
            result = run_h6_qualification(
                repo_root=Path(args.repo_root),
                qualification_root=Path(args.qualification_root),
            )
        elif args.command == "freeze":
            payload = build_plan(
                repo_root=Path(args.repo_root),
                qualification_summary_path=Path(args.qualification_summary),
                plan_path=Path(args.plan_path),
                run_root=Path(args.run_root),
            )
            result: Mapping[str, Any] = {
                "plan_id": payload["plan_id"],
                "plan_sha256": _sha256_file(Path(args.plan_path)),
                "planned_slots": EXPECTED_SLOT_COUNT,
                "development_only": True,
            }
        elif args.command == "verify":
            verified = verify_plan(
                Path(args.plan_path), expected_plan_id=args.expected_plan_id
            )
            result = {
                "plan_id": verified.plan_id,
                "verified": True,
                "development_only": True,
            }
        else:
            result = run_plan(
                Path(args.plan_path),
                expected_plan_id=args.expected_plan_id,
                max_slots=args.max_slots,
            )
    except H6ODDisaggregatedReturnAblationError as error:
        print(f"H6 OD-disaggregated return ablation refused: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    if args.command == "qualify" and result.get("status") != "qualified":
        return 1
    if args.command == "run" and result.get("status") not in {
        "completed_full",
        "completed_partial",
    }:
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "ALLOWED_MAX_SLOTS",
    "EXPECTED_PAIR_COUNT",
    "EXPECTED_SLOT_COUNT",
    "FIXED_COSTS",
    "H6ODDisaggregatedReturnAblationError",
    "PLAN_ARTIFACT_TYPE",
    "PLAN_SCHEMA_VERSION",
    "QUALIFICATION_ARTIFACT_TYPE",
    "QUALIFICATION_SCHEMA_VERSION",
    "SEEDS",
    "VARIANTS",
    "VerifiedPlan",
    "VerifiedQualification",
    "build_cli_parser",
    "build_plan",
    "main",
    "run_plan",
    "run_h6_qualification",
    "verify_qualification_artifact",
    "verify_plan",
]
