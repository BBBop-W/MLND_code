"""Run development-only qualification-profile calibration jobs.

The runner consumes only the immutable input artifact emitted by
``calibration_builder``.  It is intentionally separate from the formal pool,
formal qualification ledger, and A--E benchmark machinery.  Before selecting
any jobs it independently reconstructs the registered six physical instances,
24 profiles, and all 144 logical jobs.  Solver results are written only below
one newly claimed result root and can never become formal benchmark evidence.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any

from src.config import DataConfig, ModelConfig, to_dict

from .artifacts import write_canonical_json_exclusive
from .calibration_builder import (
    DEVELOPMENT_BASE_LOCKER_FIXED_COST_PER_DEMAND_UNIT_ORDER,
    DEVELOPMENT_CALIBRATION_JOB_SCHEMA,
    DEVELOPMENT_CALIBRATION_MANIFEST_SCHEMA,
    DEVELOPMENT_DEMAND_SCALE_ORDER,
    DEVELOPMENT_DISPLAY_ORIGIN_CHOICES,
    DEVELOPMENT_EXPECTED_SCREENING_IS_EXACT,
    DEVELOPMENT_FEEDER_FACTOR_ORDER,
    DEVELOPMENT_GENERAL_CORRIDOR_CHOICES,
    DEVELOPMENT_LAYOUT_SEED_ORDER,
    DevelopmentCalibrationProfile,
    development_calibration_profiles,
    development_instance_plans,
)
from .case_builder import (
    CANONICAL_INSTANCE_FILES,
    DEVELOPMENT_DEMAND_BLOCK,
    LOCKED_DEMAND_BLOCKS,
    LOCKED_LAYOUT_SEED_ORDER,
    LOCKED_QUALIFICATION_PROFILES,
    LOCKED_SCALE_SPECS,
    QualificationSolveProfile,
    canonical_instance_hash,
    qualification_case_sha256,
    verify_generated_case,
)
from .hashing import canonical_json, canonical_sha256
from .qualification import QualificationPolicy, assess_qualification
from .qualification_geometry_start import GeometryStartPolicy
from .qualification_runner import (
    NORMAL_QUALIFICATION_RUNNER_REASONS,
    NORMAL_QUALIFICATION_SOLVER_STATUSES,
    NativeQualificationSettings,
    QualificationInfrastructureError,
    QualificationModelAudit,
    QualificationModelContract,
    QualificationRunnerOutcome,
    qualify_canonical_instance,
)
from .resource_lock import (
    DEFAULT_BENCHMARK_RESOURCE_LOCK,
    BenchmarkResourceLock,
    BenchmarkResourceLockError,
)


DEVELOPMENT_CALIBRATION_RUN_SCHEMA = "mlnd-development-calibration-run-v1"
DEVELOPMENT_CALIBRATION_CHECKPOINT_SCHEMA = "mlnd-development-calibration-checkpoint-v1"
DEVELOPMENT_CALIBRATION_SUMMARY_SCHEMA = "mlnd-development-calibration-summary-v1"
REGISTERED_DEVELOPMENT_CALIBRATION_MANIFEST_SHA256 = (
    "6f89250a2f2d3b2c3ec23593e985e1ca3ce018db96a4ed3febf651f19290a8e8"
)
EXPECTED_PHYSICAL_INSTANCE_COUNT = 6
EXPECTED_LOGICAL_JOB_COUNT = 144
_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
_SCALES = tuple(spec.scale for spec in LOCKED_SCALE_SPECS)
_STAGE_ORDER = (
    ("small", 0),
    ("medium", 0),
    ("large", 0),
    ("small", 1),
    ("medium", 1),
    ("large", 1),
)
_NOT_REACHED_REASON = "not_reached_by_preregistered_stage_gate"

_IMPLEMENTATION_FILES = (
    Path("src/config.py"),
    Path("src/data_processor/loader.py"),
    Path("src/data_processor/instance.py"),
    Path("src/model/gurobi_model.py"),
    Path("src/benchmark/branch_cut/artifacts.py"),
    Path("src/benchmark/branch_cut/hashing.py"),
    Path("src/benchmark/branch_cut/case_builder.py"),
    Path("src/benchmark/branch_cut/calibration_builder.py"),
    Path("src/benchmark/branch_cut/calibration_runner.py"),
    Path("src/benchmark/branch_cut/qualification.py"),
    Path("src/benchmark/branch_cut/qualification_geometry_start.py"),
    Path("src/benchmark/branch_cut/qualification_runner.py"),
    Path("src/benchmark/branch_cut/qualification_incumbent_reuse.py"),
)


class CalibrationRunnerError(RuntimeError):
    """The development calibration request or source artifact is invalid."""


class CalibrationBatchInfrastructureError(RuntimeError):
    """One job failed for infrastructure reasons and the batch was stopped."""


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_sha256(value: object, *, field_name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise CalibrationRunnerError(
            f"{field_name} must be a lowercase 64-character SHA256 digest"
        )
    return value


def _require_mapping(value: object, *, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise CalibrationRunnerError(f"{field_name} must be an object")
    return value


def _require_exact_keys(
    payload: Mapping[str, Any], expected: set[str], *, field_name: str
) -> None:
    if set(payload) != expected:
        missing = sorted(expected.difference(payload))
        extra = sorted(set(payload).difference(expected))
        raise CalibrationRunnerError(
            f"{field_name} fields are inconsistent; missing={missing}, extra={extra}"
        )


def _require_int(value: object, *, field_name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise CalibrationRunnerError(f"{field_name} must be an integer >= {minimum}")
    return value


def _require_number(value: object, *, field_name: str, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CalibrationRunnerError(f"{field_name} must be numeric")
    number = float(value)
    if not math.isfinite(number) or (positive and number <= 0.0):
        qualifier = "positive and finite" if positive else "finite"
        raise CalibrationRunnerError(f"{field_name} must be {qualifier}")
    return number


def _read_canonical_object(path: Path, *, field_name: str) -> dict[str, Any]:
    try:
        raw = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise CalibrationRunnerError(
            f"cannot read {field_name} {path}: {error}"
        ) from error
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as error:
        raise CalibrationRunnerError(
            f"{field_name} is not valid JSON: {error}"
        ) from error
    if not isinstance(payload, dict):
        raise CalibrationRunnerError(f"{field_name} must contain one JSON object")
    try:
        expected_raw = canonical_json(payload) + "\n"
    except (TypeError, ValueError) as error:
        raise CalibrationRunnerError(
            f"{field_name} is not canonical-JSON-safe: {error}"
        ) from error
    if raw != expected_raw:
        raise CalibrationRunnerError(f"{field_name} is not canonical JSON")
    return payload


def _data_config_payload(config: DataConfig) -> dict[str, Any]:
    payload = asdict(config)
    payload["dates"] = list(config.dates)
    return payload


@dataclass(frozen=True, slots=True)
class CalibrationJob:
    """One authenticated logical job in canonical manifest order."""

    job_order_index: int
    job_id: str
    profile_id: str
    scale: str
    scale_order_index: int
    layout_seed: int
    development_seed_stage: int
    profile_order_index: int
    physical_instance_id: str
    instance_ref: str
    instance_sha256: str
    effective_model_config_sha256: str
    qualification_case_sha256: str

    def to_payload(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class LoadedDevelopmentCalibrationManifest:
    """Fully revalidated development input artifact."""

    path: Path
    root: Path
    manifest_sha256: str
    manifest_file_sha256: str
    raw_path: Path
    raw_sha256: str
    raw_size_bytes: int
    physical_instances: tuple[Mapping[str, Any], ...]
    physical_directories: Mapping[str, Path]
    profiles: tuple[DevelopmentCalibrationProfile, ...]
    jobs: tuple[CalibrationJob, ...]


def _expected_profile_payloads() -> list[dict[str, Any]]:
    profiles = development_calibration_profiles()
    locked_profile_ids = {
        profile.profile_id for profile in LOCKED_QUALIFICATION_PROFILES
    }
    for profile in profiles:
        if profile.profile_id in locked_profile_ids:
            raise CalibrationRunnerError(
                "development profile overlaps a formal qualification profile"
            )
        raw_solve_profile = profile.to_payload()["qualification_solve_profile"]
        reconstructed = QualificationSolveProfile(**raw_solve_profile)
        if reconstructed != profile.solve_profile():
            raise CalibrationRunnerError("registered development profile is unstable")
    return [profile.to_payload() for profile in profiles]


def _expected_jobs(
    physical_instances: Sequence[Mapping[str, Any]],
) -> tuple[CalibrationJob, ...]:
    physical_by_id = {
        str(item["physical_instance_id"]): item for item in physical_instances
    }
    jobs: list[CalibrationJob] = []
    profiles = development_calibration_profiles()
    for scale_index, scale in enumerate(LOCKED_SCALE_SPECS):
        for seed_stage, seed in enumerate(DEVELOPMENT_LAYOUT_SEED_ORDER):
            physical_id = (
                f"{DEVELOPMENT_DEMAND_BLOCK.block_id}__{scale.scale}"
                f"__devseed{seed:010d}"
            )
            physical = physical_by_id[physical_id]
            for profile_index, development_profile in enumerate(profiles):
                profile = development_profile.solve_profile()
                instance_sha256 = _require_sha256(
                    physical["instance_sha256"],
                    field_name=f"physical instance {physical_id} hash",
                )
                case_sha256 = qualification_case_sha256(instance_sha256, profile)
                job_identity = {
                    "hash_schema": DEVELOPMENT_CALIBRATION_JOB_SCHEMA,
                    "qualification_case_sha256": case_sha256,
                    "profile_id": profile.profile_id,
                    "scale_order_index": scale_index,
                    "development_seed_stage": seed_stage,
                    "profile_order_index": profile_index,
                }
                jobs.append(
                    CalibrationJob(
                        job_order_index=len(jobs),
                        job_id=(
                            "development-calibration-job-"
                            + canonical_sha256(job_identity)
                        ),
                        profile_id=profile.profile_id,
                        scale=scale.scale,
                        scale_order_index=scale_index,
                        layout_seed=seed,
                        development_seed_stage=seed_stage,
                        profile_order_index=profile_index,
                        physical_instance_id=physical_id,
                        instance_ref=str(physical["instance_ref"]),
                        instance_sha256=instance_sha256,
                        effective_model_config_sha256=(
                            profile.effective_model_config_sha256()
                        ),
                        qualification_case_sha256=case_sha256,
                    )
                )
    return tuple(jobs)


def _validate_manifest_flags(payload: Mapping[str, Any]) -> None:
    expected_flags = {
        "artifact_type": "mlnd_branch_cut_development_calibration_manifest",
        "artifact_schema_version": 1,
        "manifest_schema": DEVELOPMENT_CALIBRATION_MANIFEST_SCHEMA,
        "development_only": True,
        "eligible_for_formal_pool": False,
        "input_artifact_only": True,
        "contains_solver_results": False,
    }
    for key, expected in expected_flags.items():
        if payload.get(key) != expected or type(payload.get(key)) is not type(expected):
            raise CalibrationRunnerError(
                f"development calibration manifest has invalid {key}"
            )
    expected_formal_contract = {
        "locked_demand_blocks_used": False,
        "locked_layout_seeds_used": False,
        "locked_qualification_profiles_modified": False,
        "candidate_pool_created": False,
        "excluded_locked_demand_block_ids": [
            block.block_id for block in LOCKED_DEMAND_BLOCKS
        ],
        "excluded_locked_layout_seeds": list(LOCKED_LAYOUT_SEED_ORDER),
    }
    if payload.get("formal_contract") != expected_formal_contract:
        raise CalibrationRunnerError(
            "development calibration manifest violates its non-formal contract"
        )


def load_development_calibration_manifest(
    manifest_path: str | Path,
) -> LoadedDevelopmentCalibrationManifest:
    """Load and independently reconstruct the fixed 6x24 calibration design."""

    path = Path(manifest_path).expanduser().resolve()
    if not path.is_file():
        raise CalibrationRunnerError(f"calibration manifest does not exist: {path}")
    if Path(manifest_path).expanduser().is_symlink():
        raise CalibrationRunnerError("calibration manifest may not be a symbolic link")
    payload = _read_canonical_object(path, field_name="calibration manifest")
    expected_top_level_keys = {
        "artifact_type",
        "artifact_schema_version",
        "manifest_schema",
        "development_only",
        "eligible_for_formal_pool",
        "input_artifact_only",
        "contains_solver_results",
        "formal_contract",
        "development_demand_block",
        "development_layout_seed_order",
        "canonical_instance_files",
        "scale_order",
        "raw_input",
        "build_config",
        "screening_contract",
        "calibration_grid",
        "physical_instance_count",
        "physical_instances",
        "logical_job_count",
        "logical_job_order_contract",
        "logical_job_order",
        "development_calibration_manifest_sha256",
    }
    _require_exact_keys(
        payload, expected_top_level_keys, field_name="calibration manifest"
    )
    _validate_manifest_flags(payload)

    claimed_manifest_sha256 = _require_sha256(
        payload["development_calibration_manifest_sha256"],
        field_name="development_calibration_manifest_sha256",
    )
    unhashed_payload = {
        key: value
        for key, value in payload.items()
        if key != "development_calibration_manifest_sha256"
    }
    observed_manifest_sha256 = canonical_sha256(unhashed_payload)
    if observed_manifest_sha256 != claimed_manifest_sha256:
        raise CalibrationRunnerError("calibration manifest content hash mismatch")
    if claimed_manifest_sha256 != REGISTERED_DEVELOPMENT_CALIBRATION_MANIFEST_SHA256:
        raise CalibrationRunnerError(
            "calibration manifest is not the preregistered development artifact"
        )

    root = path.parent.resolve()
    raw_input = _require_mapping(payload["raw_input"], field_name="raw_input")
    _require_exact_keys(
        raw_input,
        {"source", "path", "sha256", "size_bytes"},
        field_name="raw_input",
    )
    if raw_input["source"] != DEVELOPMENT_DEMAND_BLOCK.source:
        raise CalibrationRunnerError("calibration raw input is not the TSL source")
    if not isinstance(raw_input["path"], str) or not raw_input["path"]:
        raise CalibrationRunnerError("raw_input.path must be nonempty")
    raw_path = Path(raw_input["path"]).expanduser().resolve()
    if str(raw_path) != raw_input["path"] or not raw_path.is_file():
        raise CalibrationRunnerError(
            "raw_input.path must be an existing resolved regular file"
        )
    if Path(raw_input["path"]).is_symlink():
        raise CalibrationRunnerError("calibration raw input may not be a symbolic link")
    raw_sha256 = _require_sha256(raw_input["sha256"], field_name="raw_input.sha256")
    raw_size = _require_int(raw_input["size_bytes"], field_name="raw_input.size_bytes")
    if _sha256_file(raw_path) != raw_sha256 or raw_path.stat().st_size != raw_size:
        raise CalibrationRunnerError("calibration raw input hash or size mismatch")

    build_config = _require_mapping(payload["build_config"], field_name="build_config")
    _require_exact_keys(
        build_config,
        {
            "output_root",
            "tsl_raw_path",
            "candidate_hour_start",
            "candidate_hour_end",
            "site_grid_km",
        },
        field_name="build_config",
    )
    if build_config["output_root"] != str(root):
        raise CalibrationRunnerError(
            "build_config.output_root does not match manifest root"
        )
    if build_config["tsl_raw_path"] != str(raw_path):
        raise CalibrationRunnerError(
            "build_config.tsl_raw_path does not match raw input"
        )
    hour_start = _require_int(
        build_config["candidate_hour_start"],
        field_name="build_config.candidate_hour_start",
    )
    hour_end = _require_int(
        build_config["candidate_hour_end"],
        field_name="build_config.candidate_hour_end",
    )
    if not 0 <= hour_start < hour_end <= 24:
        raise CalibrationRunnerError("build_config candidate hours are invalid")
    site_grid_km = _require_number(
        build_config["site_grid_km"],
        field_name="build_config.site_grid_km",
        positive=True,
    )

    if payload["development_demand_block"] != DEVELOPMENT_DEMAND_BLOCK.to_payload():
        raise CalibrationRunnerError("development demand block contract drifted")
    if payload["development_layout_seed_order"] != list(DEVELOPMENT_LAYOUT_SEED_ORDER):
        raise CalibrationRunnerError("development layout seed order drifted")
    if payload["canonical_instance_files"] != list(CANONICAL_INSTANCE_FILES):
        raise CalibrationRunnerError("canonical instance file order drifted")
    if payload["scale_order"] != [asdict(spec) for spec in LOCKED_SCALE_SPECS]:
        raise CalibrationRunnerError("scale order drifted")
    if payload["screening_contract"] != {
        "display_origin_choices": DEVELOPMENT_DISPLAY_ORIGIN_CHOICES,
        "general_corridor_choices": DEVELOPMENT_GENERAL_CORRIDOR_CHOICES,
        "expected_screening_is_exact": DEVELOPMENT_EXPECTED_SCREENING_IS_EXACT,
    }:
        raise CalibrationRunnerError("development screening contract drifted")

    raw_physical = payload["physical_instances"]
    if not isinstance(raw_physical, list):
        raise CalibrationRunnerError("physical_instances must be a list")
    if (
        _require_int(
            payload["physical_instance_count"],
            field_name="physical_instance_count",
        )
        != EXPECTED_PHYSICAL_INSTANCE_COUNT
        or len(raw_physical) != EXPECTED_PHYSICAL_INSTANCE_COUNT
    ):
        raise CalibrationRunnerError("calibration must contain exactly 6 instances")

    physical_instances: list[Mapping[str, Any]] = []
    physical_directories: dict[str, Path] = {}
    shared_center: tuple[float, float] | None = None
    for plan, raw_item in zip(development_instance_plans(), raw_physical, strict=True):
        item = _require_mapping(
            raw_item, field_name=f"physical instance {plan.case_id}"
        )
        instance_ref = f"instances/{plan.case_id}"
        claimed_instance_dir = root / instance_ref
        if claimed_instance_dir.is_symlink():
            raise CalibrationRunnerError(
                f"physical instance directory may not be a symbolic link: {plan.case_id}"
            )
        instance_dir = claimed_instance_dir.resolve()
        if instance_dir.parent != (root / "instances").resolve():
            raise CalibrationRunnerError("physical instance path escapes input root")
        symlinked_files = [
            filename
            for filename in CANONICAL_INSTANCE_FILES
            if (instance_dir / filename).is_symlink()
        ]
        if symlinked_files:
            raise CalibrationRunnerError(
                f"physical instance {plan.case_id} contains symbolic links: "
                f"{symlinked_files}"
            )
        expected_data_config = plan.data_config(
            raw_path=raw_path,
            output_dir=instance_dir,
            candidate_hour_start=hour_start,
            candidate_hour_end=hour_end,
            site_grid_km=site_grid_km,
        )
        expected_data_config.validate()
        try:
            verified = verify_generated_case(instance_dir, expected_data_config)
        except (OSError, UnicodeError, ValueError) as error:
            raise CalibrationRunnerError(
                f"physical instance {plan.case_id} verification failed: {error}"
            ) from error
        if shared_center is None:
            shared_center = verified.projection_center
        elif verified.projection_center != shared_center:
            raise CalibrationRunnerError(
                "development physical instances do not share one projection center"
            )
        expected_item: dict[str, Any] = {
            "physical_instance_id": plan.case_id,
            "development_only": True,
            "scale": plan.scale.scale,
            "development_seed_stage": DEVELOPMENT_LAYOUT_SEED_ORDER.index(
                plan.layout_seed
            ),
            "layout_seed": plan.layout_seed,
            "instance_ref": instance_ref,
            "instance_sha256": verified.instance_sha256,
            "raw_file_sha256": raw_sha256,
            "raw_file_size_bytes": raw_size,
            "canonical_file_sha256": dict(verified.canonical_file_sha256),
            "data_config": _data_config_payload(expected_data_config),
            "projection_center_lon": verified.projection_center[0],
            "projection_center_lat": verified.projection_center[1],
        }
        if canonical_json(item) != canonical_json(expected_item):
            raise CalibrationRunnerError(
                f"physical instance metadata mismatch for {plan.case_id}"
            )
        physical_instances.append(expected_item)
        physical_directories[plan.case_id] = instance_dir

    expected_profile_payloads = _expected_profile_payloads()
    expected_grid = {
        "profile_order": "demand_scale_then_feeder_factor_then_base_fixed_cost_ratio",
        "demand_scale": list(DEVELOPMENT_DEMAND_SCALE_ORDER),
        "feeder_rider_cost_factor": list(DEVELOPMENT_FEEDER_FACTOR_ORDER),
        "base_locker_fixed_cost_per_demand_unit": list(
            DEVELOPMENT_BASE_LOCKER_FIXED_COST_PER_DEMAND_UNIT_ORDER
        ),
        "derived_locker_fixed_cost_rule": (
            "locker_fixed_cost_per_hour = demand_scale * "
            "base_locker_fixed_cost_per_demand_unit"
        ),
        "economic_interpretation": (
            "hold H fixed cost per unit of demand scale constant while "
            "separately probing size and congestion effects"
        ),
        "profile_count": len(expected_profile_payloads),
        "profiles": expected_profile_payloads,
    }
    if canonical_json(payload["calibration_grid"]) != canonical_json(expected_grid):
        raise CalibrationRunnerError(
            "calibration profiles, ModelConfig payloads, or profile order drifted"
        )

    expected_jobs = _expected_jobs(physical_instances)
    if len(expected_jobs) != EXPECTED_LOGICAL_JOB_COUNT:
        raise CalibrationRunnerError(
            "registered logical calibration design is not 144 jobs"
        )
    raw_jobs = payload["logical_job_order"]
    if not isinstance(raw_jobs, list):
        raise CalibrationRunnerError("logical_job_order must be a list")
    if (
        _require_int(payload["logical_job_count"], field_name="logical_job_count")
        != EXPECTED_LOGICAL_JOB_COUNT
        or len(raw_jobs) != EXPECTED_LOGICAL_JOB_COUNT
    ):
        raise CalibrationRunnerError("calibration must contain exactly 144 jobs")
    if payload["logical_job_order_contract"] != (
        "scale_order_then_development_seed_stage_then_profile_order"
    ):
        raise CalibrationRunnerError("logical job order contract drifted")
    if canonical_json(raw_jobs) != canonical_json(
        [job.to_payload() for job in expected_jobs]
    ):
        raise CalibrationRunnerError(
            "logical job IDs, config/case hashes, or 144-job order mismatch"
        )

    return LoadedDevelopmentCalibrationManifest(
        path=path,
        root=root,
        manifest_sha256=claimed_manifest_sha256,
        manifest_file_sha256=_sha256_file(path),
        raw_path=raw_path,
        raw_sha256=raw_sha256,
        raw_size_bytes=raw_size,
        physical_instances=tuple(physical_instances),
        physical_directories=physical_directories,
        profiles=development_calibration_profiles(),
        jobs=expected_jobs,
    )


_REGISTERED_POLICY = QualificationPolicy(large_scale_open_h_target=(50, 60))
_REGISTERED_NATIVE_SETTINGS = NativeQualificationSettings(
    time_limit_seconds=5.0,
    mip_gap=0.01,
    solver_seed=42,
    threads=1,
    output_flag=0,
    method=-1,
    miqcp_method=-1,
    mip_focus=0,
    heuristics=0.05,
    no_rel_heur_time=0.0,
    presolve=-1,
    native_cuts=-1,
    root_cut_passes=2,
    numeric_focus=0,
    feasibility_tolerance=1e-6,
    optimality_tolerance=1e-6,
    integer_feasibility_tolerance=1e-5,
)
_REGISTERED_GEOMETRY_START_POLICY = GeometryStartPolicy(enabled=False)


@dataclass(frozen=True, slots=True)
class DevelopmentCalibrationRunConfig:
    """Explicit bounded settings for one serial development calibration run."""

    manifest_path: Path
    results_root: Path
    max_jobs: int
    time_limit_seconds: float
    solver_seed: int
    miqcp_method: int
    native_cuts: int
    root_cut_passes: int
    mode: str = "MX"
    scales: tuple[str, ...] | None = None
    development_seed_stages: tuple[int, ...] | None = None
    profile_order_indices: tuple[int, ...] | None = None
    concurrency: int = 1
    mip_gap: float = 0.01
    output_flag: int = 0
    method: int = -1
    mip_focus: int = 0
    heuristics: float = 0.05
    no_rel_heur_time: float = 0.0
    presolve: int = -1
    numeric_focus: int = 0
    feasibility_tolerance: float = 1e-6
    optimality_tolerance: float = 1e-6
    integer_feasibility_tolerance: float = 1e-5
    policy: QualificationPolicy = field(default_factory=lambda: _REGISTERED_POLICY)
    geometry_start_policy: GeometryStartPolicy = field(
        default_factory=lambda: _REGISTERED_GEOMETRY_START_POLICY
    )
    resource_lock: Path = DEFAULT_BENCHMARK_RESOURCE_LOCK

    def __post_init__(self) -> None:
        if (
            isinstance(self.max_jobs, bool)
            or not isinstance(self.max_jobs, int)
            or self.max_jobs < 1
        ):
            raise ValueError("max_jobs must be an explicitly supplied positive integer")
        if self.concurrency != 1 or isinstance(self.concurrency, bool):
            raise ValueError("development calibration currently requires concurrency=1")
        normalized_mode = str(self.mode).upper()
        if normalized_mode not in {"MX", "HD", "LO"}:
            raise ValueError("mode must be one of MX, HD, or LO")
        if normalized_mode != "MX":
            raise ValueError("the preregistered calibration ladder is fixed to MX")
        object.__setattr__(self, "mode", normalized_mode)
        for name, values, registered_order in (
            ("scales", self.scales, _SCALES),
            (
                "development_seed_stages",
                self.development_seed_stages,
                tuple(range(len(DEVELOPMENT_LAYOUT_SEED_ORDER))),
            ),
            (
                "profile_order_indices",
                self.profile_order_indices,
                tuple(range(len(development_calibration_profiles()))),
            ),
        ):
            if values is None:
                continue
            normalized = tuple(values)
            if not normalized or len(normalized) != len(set(normalized)):
                raise ValueError(f"{name} must be a nonempty unique selection")
            if any(value not in registered_order for value in normalized):
                raise ValueError(f"{name} contains an out-of-grid value")
            selected = set(normalized)
            object.__setattr__(
                self,
                name,
                tuple(value for value in registered_order if value in selected),
            )
        if not isinstance(self.policy, QualificationPolicy):
            raise TypeError("policy must be a QualificationPolicy")
        if not isinstance(self.geometry_start_policy, GeometryStartPolicy):
            raise TypeError("geometry_start_policy must be a GeometryStartPolicy")
        if self.policy != _REGISTERED_POLICY:
            raise ValueError(
                "qualification policy is fixed by the preregistered ladder"
            )
        if self.geometry_start_policy != _REGISTERED_GEOMETRY_START_POLICY:
            raise ValueError(
                "geometry start is disabled by the preregistered calibration ladder"
            )
        if self.native_settings() != _REGISTERED_NATIVE_SETTINGS:
            raise ValueError(
                "native qualification settings drift from the preregistered "
                "5-second single-thread ladder"
            )

    def native_settings(self) -> NativeQualificationSettings:
        return NativeQualificationSettings(
            time_limit_seconds=self.time_limit_seconds,
            mip_gap=self.mip_gap,
            solver_seed=self.solver_seed,
            threads=1,
            output_flag=self.output_flag,
            method=self.method,
            miqcp_method=self.miqcp_method,
            mip_focus=self.mip_focus,
            heuristics=self.heuristics,
            no_rel_heur_time=self.no_rel_heur_time,
            presolve=self.presolve,
            native_cuts=self.native_cuts,
            root_cut_passes=self.root_cut_passes,
            numeric_focus=self.numeric_focus,
            feasibility_tolerance=self.feasibility_tolerance,
            optimality_tolerance=self.optimality_tolerance,
            integer_feasibility_tolerance=self.integer_feasibility_tolerance,
        )


def select_calibration_jobs(
    manifest: LoadedDevelopmentCalibrationManifest,
    config: DevelopmentCalibrationRunConfig,
) -> tuple[CalibrationJob, ...]:
    """Select the deterministic filter universe in canonical manifest order."""

    scale_filter = set(config.scales or _SCALES)
    seed_filter = set(
        config.development_seed_stages or range(len(DEVELOPMENT_LAYOUT_SEED_ORDER))
    )
    profile_filter = set(
        config.profile_order_indices or range(len(development_calibration_profiles()))
    )
    filtered = tuple(
        job
        for job in manifest.jobs
        if job.scale in scale_filter
        and job.development_seed_stage in seed_filter
        and job.profile_order_index in profile_filter
    )
    if not filtered:
        raise CalibrationRunnerError("calibration selectors matched no jobs")
    return filtered


InstanceLoader = Callable[[Path, ModelConfig, float], object]
Qualifier = Callable[..., QualificationRunnerOutcome]
EnvironmentCapture = Callable[[Path], Mapping[str, Any]]


def _default_instance_loader(
    instance_dir: Path, model_config: ModelConfig, demand_scale: float
) -> object:
    from src.data_processor.loader import DataLoader

    return DataLoader().load_instance_dir(
        str(instance_dir),
        model_config=model_config,
        demand_scale=demand_scale,
    )


def _git(repository_root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ("git", *arguments),
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _default_environment_capture(repository_root: Path) -> Mapping[str, Any]:
    import gurobipy as gp

    implementation_files = {
        path.as_posix(): _sha256_file(repository_root / path)
        for path in _IMPLEMENTATION_FILES
    }
    status_lines = tuple(
        line
        for line in _git(
            repository_root, "status", "--porcelain", "--untracked-files=all"
        ).splitlines()
        if line
    )
    commit = _git(repository_root, "rev-parse", "HEAD")
    if len(commit) != 40 or any(
        character not in "0123456789abcdef" for character in commit
    ):
        raise CalibrationRunnerError("git HEAD is not a lowercase commit digest")
    gurobi_version = ".".join(str(item) for item in gp.gurobi.version())
    return {
        "git": {
            "repository_root": str(repository_root),
            "commit": commit,
            "worktree_dirty": bool(status_lines),
            "status_includes_untracked": True,
            "status_lines": list(status_lines),
            "status_sha256": canonical_sha256(status_lines),
        },
        "implementation": {
            "ordered_file_sha256": implementation_files,
            "implementation_sha256": canonical_sha256(implementation_files),
        },
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
            "executable": sys.executable,
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "gurobi_version": gurobi_version,
    }


def _stable_environment_identity(environment: Mapping[str, Any]) -> dict[str, Any]:
    """Exclude only the one-time clean-status observation from drift checks."""

    git_payload = dict(
        _require_mapping(environment.get("git"), field_name="environment.git")
    )
    for key in (
        "worktree_dirty",
        "status_includes_untracked",
        "status_lines",
        "status_sha256",
    ):
        git_payload.pop(key, None)
    return {
        **dict(environment),
        "git": git_payload,
    }


@dataclass(frozen=True, slots=True)
class CalibrationRunnerHooks:
    load_instance: InstanceLoader = _default_instance_loader
    qualify_instance: Qualifier = qualify_canonical_instance
    capture_environment: EnvironmentCapture = _default_environment_capture
    now_utc: Callable[[], str] = _utc_now


DEFAULT_CALIBRATION_HOOKS = CalibrationRunnerHooks()


@dataclass(frozen=True, slots=True)
class DevelopmentCalibrationRunResult:
    results_root: Path
    run_identity_path: Path
    summary_path: Path
    calibration_run_id: str
    selected_job_count: int
    attempted_job_count: int
    accepted_job_count: int
    unattempted_job_count: int
    locked_profile_ids: tuple[str, ...]
    checkpoint_paths: tuple[Path, ...]


def _runtime_model_contract(
    job: CalibrationJob, profile: QualificationSolveProfile
) -> QualificationModelContract:
    if profile.expected_screening_is_exact is not False:
        raise CalibrationRunnerError(
            "development calibration requires expected screening_is_exact=false"
        )
    return QualificationModelContract(
        qualification_case_sha256=job.qualification_case_sha256,
        effective_model_config_sha256=job.effective_model_config_sha256,
        demand_scale=profile.demand_scale,
        locker_fixed_cost_per_hour=profile.locker_fixed_cost_per_hour,
        display_origin_choices=profile.display_origin_choices,
        general_corridor_choices=profile.general_corridor_choices,
        screening_is_exact=False,
    )


def _validate_loaded_instance(
    instance: object,
    *,
    model_config: ModelConfig,
    profile: QualificationSolveProfile,
    mode: str,
) -> None:
    if getattr(instance, "config", None) != model_config:
        raise QualificationInfrastructureError(
            "loaded calibration instance did not retain the full ModelConfig"
        )
    metadata = getattr(instance, "metadata", None)
    if not isinstance(metadata, Mapping):
        raise QualificationInfrastructureError(
            "loaded calibration instance has no metadata mapping"
        )
    observed_scale = metadata.get("demand_scale")
    if (
        isinstance(observed_scale, bool)
        or not isinstance(observed_scale, (int, float))
        or float(observed_scale) != profile.demand_scale
    ):
        raise QualificationInfrastructureError(
            "loaded calibration instance did not retain demand_scale"
        )
    screening_reader = getattr(instance, "screening_is_exact", None)
    if not callable(screening_reader) or screening_reader(mode) is not False:
        raise QualificationInfrastructureError(
            "development calibration runtime requires screening_is_exact=false"
        )


def _audit_matches_contract(
    audit: QualificationModelAudit, contract: QualificationModelContract
) -> bool:
    return audit.contract_values() == (
        contract.effective_model_config_sha256,
        contract.demand_scale,
        contract.locker_fixed_cost_per_hour,
        contract.display_origin_choices,
        contract.general_corridor_choices,
        False,
    )


def _validate_outcome(
    outcome: QualificationRunnerOutcome,
    *,
    job: CalibrationJob,
    qualification_run_id: str,
    contract: QualificationModelContract,
    policy: QualificationPolicy,
    geometry_start_policy: GeometryStartPolicy,
) -> None:
    if not isinstance(outcome, QualificationRunnerOutcome):
        raise QualificationInfrastructureError(
            "qualification runner did not return QualificationRunnerOutcome"
        )
    if type(outcome.accepted) is not bool:
        raise QualificationInfrastructureError(
            "qualification outcome accepted flag must be boolean"
        )
    if type(outcome.evidence.witness_feasible) is not bool:
        raise QualificationInfrastructureError(
            "qualification evidence witness_feasible must be boolean"
        )
    if type(outcome.fresh_model_validation_performed) is not bool:
        raise QualificationInfrastructureError(
            "qualification fresh-validation flag must be boolean"
        )
    independently_assessed = assess_qualification(outcome.evidence, policy)
    if outcome.decision != independently_assessed:
        raise QualificationInfrastructureError(
            "qualification decision does not match independent evidence assessment"
        )
    if outcome.accepted != (not outcome.rejection_reasons):
        raise QualificationInfrastructureError(
            "qualification accepted/rejection state is internally inconsistent"
        )
    decision_reason_iterator = iter(outcome.rejection_reasons)
    if not all(
        any(candidate == reason for candidate in decision_reason_iterator)
        for reason in outcome.decision.rejection_reasons
    ):
        raise QualificationInfrastructureError(
            "qualification decision reasons are not an ordered subset of outcome reasons"
        )
    if not set(outcome.decision.rejection_reasons).issubset(outcome.rejection_reasons):
        raise QualificationInfrastructureError(
            "qualification outcome dropped independently assessed rejection reasons"
        )
    if (
        outcome.evidence.witness_feasible
        and not outcome.fresh_model_validation_performed
    ):
        raise QualificationInfrastructureError(
            "feasible calibration witness lacks fresh-model validation"
        )
    if outcome.expected_model_contract != contract:
        raise QualificationInfrastructureError(
            "qualification outcome model contract does not match calibration job"
        )
    if outcome.geometry_start_policy != geometry_start_policy:
        raise QualificationInfrastructureError(
            "qualification outcome geometry-start policy drifted"
        )
    for label, record in (
        ("evidence", outcome.evidence),
        ("decision", outcome.decision),
    ):
        if (
            record.case_id != job.job_id
            or record.scale != job.scale
            or record.qualification_run_id != qualification_run_id
            or record.instance_sha256 != job.instance_sha256
            or record.qualification_case_sha256 != job.qualification_case_sha256
            or record.effective_model_config_sha256 != job.effective_model_config_sha256
            or record.screening_is_exact is not False
        ):
            raise QualificationInfrastructureError(
                f"qualification {label} identity does not match calibration job"
            )
    if outcome.decision.policy != policy:
        raise QualificationInfrastructureError(
            "qualification outcome policy does not match calibration run"
        )
    decision_reasons = set(outcome.decision.rejection_reasons)
    unexpected_runner_reasons = sorted(
        set(outcome.rejection_reasons)
        - decision_reasons
        - NORMAL_QUALIFICATION_RUNNER_REASONS
    )
    if unexpected_runner_reasons:
        raise QualificationInfrastructureError(
            "qualification outcome contains infrastructure runner reasons: "
            + ", ".join(unexpected_runner_reasons)
        )
    if outcome.witness_status not in NORMAL_QUALIFICATION_SOLVER_STATUSES:
        raise QualificationInfrastructureError(
            f"unexpected calibration witness status {outcome.witness_status!r}"
        )
    if outcome.fresh_model_validation_performed:
        if outcome.validation_status not in NORMAL_QUALIFICATION_SOLVER_STATUSES:
            raise QualificationInfrastructureError(
                "unexpected calibration fresh-validation status "
                f"{outcome.validation_status!r}"
            )
    elif outcome.validation_status != "NOT_RUN":
        raise QualificationInfrastructureError(
            "calibration outcome records validation status without fresh validation"
        )
    if outcome.exception_messages:
        raise QualificationInfrastructureError(
            "calibration outcome contains qualification mechanism exceptions"
        )
    if outcome.witness_model_audit is None or not _audit_matches_contract(
        outcome.witness_model_audit, contract
    ):
        raise QualificationInfrastructureError(
            "qualification outcome lacks the required screened witness audit"
        )
    if outcome.fresh_model_audit is not None and not _audit_matches_contract(
        outcome.fresh_model_audit, contract
    ):
        raise QualificationInfrastructureError(
            "qualification outcome fresh-model audit does not match calibration job"
        )
    if outcome.fresh_model_validation_performed and outcome.fresh_model_audit is None:
        raise QualificationInfrastructureError(
            "fresh calibration validation lacks its model audit"
        )
    if (
        not outcome.fresh_model_validation_performed
        and outcome.fresh_model_audit is not None
    ):
        raise QualificationInfrastructureError(
            "fresh calibration audit exists although validation was not performed"
        )
    if outcome.fresh_model_audit is not None and (
        outcome.witness_model_audit.model_fingerprint is None
        or outcome.fresh_model_audit.model_fingerprint is None
        or outcome.witness_model_audit.model_fingerprint
        != outcome.fresh_model_audit.model_fingerprint
    ):
        raise QualificationInfrastructureError(
            "fresh calibration model fingerprint differs from witness model"
        )
    if outcome.accepted and not outcome.fresh_model_validation_performed:
        raise QualificationInfrastructureError(
            "accepted calibration outcome lacks fresh-model validation"
        )
    if outcome.accepted and (
        not outcome.decision.accepted or not outcome.evidence.witness_feasible
    ):
        raise QualificationInfrastructureError(
            "accepted calibration outcome is not supported by its decision and witness"
        )
    if outcome.accepted and outcome.captured_variable_count <= 0:
        raise QualificationInfrastructureError(
            "accepted calibration outcome lacks a captured complete incumbent"
        )
    payload = outcome.to_payload()
    for field_name in (
        "phase_timings",
        "expected_model_contract",
        "witness_model_audit",
        "fresh_model_audit",
        "evidence",
        "decision",
    ):
        if field_name not in payload:
            raise QualificationInfrastructureError(
                f"qualification outcome payload is missing {field_name}"
            )


def _revalidate_job_inputs(
    manifest: LoadedDevelopmentCalibrationManifest, job: CalibrationJob
) -> None:
    if _sha256_file(manifest.path) != manifest.manifest_file_sha256:
        raise QualificationInfrastructureError(
            "calibration manifest changed during the run"
        )
    if (
        _sha256_file(manifest.raw_path) != manifest.raw_sha256
        or manifest.raw_path.stat().st_size != manifest.raw_size_bytes
    ):
        raise QualificationInfrastructureError(
            "calibration raw source changed during the run"
        )
    instance_dir = manifest.physical_directories[job.physical_instance_id]
    observed_sha256, _ = canonical_instance_hash(instance_dir)
    if observed_sha256 != job.instance_sha256:
        raise QualificationInfrastructureError(
            f"canonical instance changed during job {job.job_id}"
        )


def _claim_new_results_root(results_root: Path, input_root: Path) -> Path:
    target = results_root.expanduser().resolve()
    try:
        target.relative_to(input_root)
    except ValueError:
        pass
    else:
        raise CalibrationRunnerError(
            "results_root may not be the calibration input root or its descendant"
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        target.mkdir(exist_ok=False)
    except FileExistsError as error:
        raise CalibrationRunnerError(
            f"refusing to reuse or overwrite results_root: {target}"
        ) from error
    return target


def _checkpoint_path(checkpoints_root: Path, job: CalibrationJob) -> Path:
    digest = hashlib.sha256(job.job_id.encode("utf-8")).hexdigest()
    return checkpoints_root / f"job-{job.job_order_index:03d}-{digest}.json"


def _write_checkpoint(
    path: Path,
    *,
    identity: Mapping[str, Any],
    result: Mapping[str, Any],
) -> tuple[Path, str]:
    entry = {"identity": dict(identity), "result": dict(result)}
    payload = {
        "artifact_type": "mlnd_branch_cut_development_calibration_checkpoint",
        "artifact_schema_version": 1,
        "checkpoint_schema": DEVELOPMENT_CALIBRATION_CHECKPOINT_SCHEMA,
        "development_only": True,
        "eligible_for_formal_pool": False,
        "contains_algorithm_comparison_results": False,
        **entry,
        "checkpoint_sha256": canonical_sha256(entry),
    }
    return write_canonical_json_exclusive(path, payload), payload["checkpoint_sha256"]


def _infrastructure_error_payload(stage: str, error: BaseException) -> dict[str, Any]:
    return {
        "stage": stage,
        "type": type(error).__name__,
        "message": str(error),
        "qualification_infrastructure_error": isinstance(
            error, QualificationInfrastructureError
        ),
    }


def _stage_index(job: CalibrationJob) -> int:
    try:
        return _STAGE_ORDER.index((job.scale, job.development_seed_stage))
    except ValueError as error:
        raise CalibrationRunnerError(
            f"job {job.job_id} is outside the preregistered stage ladder"
        ) from error


def _canonical_job_ledger(
    manifest: LoadedDevelopmentCalibrationManifest,
    records_by_job_id: Mapping[str, Mapping[str, Any]],
    selected_job_ids: set[str],
    *,
    termination_reason: str,
) -> list[dict[str, Any]]:
    if termination_reason not in {
        "completed",
        "max_jobs_limit",
        "infrastructure_abort",
    }:
        raise CalibrationRunnerError("invalid calibration ledger termination reason")
    jobs_by_stage_profile = {
        (_stage_index(job), job.profile_order_index): job for job in manifest.jobs
    }
    ledger: list[dict[str, Any]] = []
    for job in manifest.jobs:
        record = records_by_job_id.get(job.job_id)
        if record is None:
            if job.job_id not in selected_job_ids:
                reason_code = "selector_excluded"
                reason = "selector_excluded"
            else:
                blocked_by_stage = any(
                    (
                        predecessor := records_by_job_id.get(
                            jobs_by_stage_profile[
                                (prior_stage, job.profile_order_index)
                            ].job_id
                        )
                    )
                    is None
                    or predecessor.get("accepted") is not True
                    for prior_stage in range(_stage_index(job))
                )
                if blocked_by_stage:
                    reason_code = "stage_gate"
                    reason = _NOT_REACHED_REASON
                elif termination_reason == "max_jobs_limit":
                    reason_code = "max_jobs_limit"
                    reason = "max_jobs_limit"
                elif termination_reason == "infrastructure_abort":
                    reason_code = "infrastructure_abort"
                    reason = "infrastructure_abort"
                else:
                    reason_code = "stage_gate"
                    reason = _NOT_REACHED_REASON
            ledger.append(
                {
                    "job_order_index": job.job_order_index,
                    "job_id": job.job_id,
                    "profile_id": job.profile_id,
                    "scale": job.scale,
                    "development_seed_stage": job.development_seed_stage,
                    "profile_order_index": job.profile_order_index,
                    "preregistered_stage_index": _stage_index(job),
                    "selected_by_filters": job.job_id in selected_job_ids,
                    "attempted": False,
                    "accepted": False,
                    "not_attempted_reason_code": reason_code,
                    "not_attempted_reason": reason,
                    "checkpoint_path": None,
                    "checkpoint_sha256": None,
                    "infrastructure_failure": None,
                    "outcome": None,
                }
            )
            continue
        ledger.append(dict(record))
    if len(ledger) != EXPECTED_LOGICAL_JOB_COUNT or [
        item["job_order_index"] for item in ledger
    ] != list(range(EXPECTED_LOGICAL_JOB_COUNT)):
        raise CalibrationRunnerError("calibration ledger lost canonical 144-job order")
    return ledger


def run_development_calibration(
    config: DevelopmentCalibrationRunConfig,
    *,
    hooks: CalibrationRunnerHooks = DEFAULT_CALIBRATION_HOOKS,
) -> DevelopmentCalibrationRunResult:
    """Run a bounded serial subset and persist one immutable checkpoint per job."""

    if not isinstance(config, DevelopmentCalibrationRunConfig):
        raise TypeError("config must be DevelopmentCalibrationRunConfig")
    manifest = load_development_calibration_manifest(config.manifest_path)
    selected_jobs = select_calibration_jobs(manifest, config)
    selected_job_ids = {job.job_id for job in selected_jobs}
    settings = config.native_settings()

    with BenchmarkResourceLock(
        config.resource_lock, command="calibrate-qualification-profiles"
    ):
        environment = dict(hooks.capture_environment(_REPOSITORY_ROOT))
        try:
            environment_sha256 = canonical_sha256(environment)
        except (TypeError, ValueError) as error:
            raise CalibrationRunnerError(
                f"runtime environment is not canonical-JSON-safe: {error}"
            ) from error
        git_payload = _require_mapping(
            environment.get("git"), field_name="environment.git"
        )
        if (
            git_payload.get("status_includes_untracked") is not True
            or git_payload.get("worktree_dirty") is not False
            or git_payload.get("status_lines") != []
        ):
            raise CalibrationRunnerError(
                "development calibration requires a clean worktree status that "
                "includes untracked files"
            )
        commit = git_payload.get("commit")
        if (
            not isinstance(commit, str)
            or len(commit) != 40
            or any(character not in "0123456789abcdef" for character in commit)
        ):
            raise CalibrationRunnerError(
                "runtime environment must contain a lowercase 40-character git commit"
            )
        stable_environment_sha256 = canonical_sha256(
            _stable_environment_identity(environment)
        )

        run_identity = {
            "run_schema": DEVELOPMENT_CALIBRATION_RUN_SCHEMA,
            "development_only": True,
            "eligible_for_formal_pool": False,
            "uses_formal_pool": False,
            "uses_locked_qualification_profiles": False,
            "uses_a_e_algorithm_performance": False,
            "algorithm_comparison_performed": False,
            "source_manifest_path": str(manifest.path),
            "source_manifest_sha256": manifest.manifest_sha256,
            "source_manifest_file_sha256": manifest.manifest_file_sha256,
            "source_raw_sha256": manifest.raw_sha256,
            "mode": config.mode,
            "concurrency": 1,
            "policy": config.policy.to_payload(),
            "native_settings": asdict(settings),
            "geometry_start_policy": config.geometry_start_policy.to_payload(),
            "selectors": {
                "scales": list(config.scales) if config.scales is not None else None,
                "development_seed_stages": (
                    list(config.development_seed_stages)
                    if config.development_seed_stages is not None
                    else None
                ),
                "profile_order_indices": (
                    list(config.profile_order_indices)
                    if config.profile_order_indices is not None
                    else None
                ),
                "max_jobs": config.max_jobs,
            },
            "selected_jobs": [job.to_payload() for job in selected_jobs],
            "preregistered_ladder": {
                "stage_order": [
                    {
                        "stage_index": stage_index,
                        "scale": scale,
                        "development_seed_stage": seed_stage,
                    }
                    for stage_index, (scale, seed_stage) in enumerate(_STAGE_ORDER)
                ],
                "advance_on_fresh_accepted_outcome_only": True,
                "unattempted_reason": _NOT_REACHED_REASON,
                "lock_first_profile_count": 4,
                "lock_order": "original_profile_order_index",
            },
            "environment": environment,
            "environment_sha256": environment_sha256,
            "stable_environment_sha256": stable_environment_sha256,
            "git_commit": commit,
        }
        run_identity_sha256 = canonical_sha256(run_identity)
        calibration_run_id = "development-calibration-run-" + run_identity_sha256

        results_root = _claim_new_results_root(config.results_root, manifest.root)
        checkpoints_root = results_root / "checkpoints"
        checkpoints_root.mkdir(exist_ok=False)
        run_identity_path = write_canonical_json_exclusive(
            results_root / "run_identity.json",
            {
                "artifact_type": "mlnd_branch_cut_development_calibration_run_identity",
                "artifact_schema_version": 1,
                "calibration_run_id": calibration_run_id,
                "run_identity": run_identity,
                "run_identity_sha256": run_identity_sha256,
            },
        )

        checkpoint_paths: list[Path] = []
        checkpoint_records: list[dict[str, Any]] = []
        records_by_job_id: dict[str, dict[str, Any]] = {}
        accepted_count = 0

        def revalidate_runtime_environment() -> None:
            try:
                observed = canonical_sha256(
                    _stable_environment_identity(
                        dict(hooks.capture_environment(_REPOSITORY_ROOT))
                    )
                )
            except (TypeError, ValueError) as error:
                raise QualificationInfrastructureError(
                    f"runtime environment revalidation failed: {error}"
                ) from error
            if observed != stable_environment_sha256:
                raise QualificationInfrastructureError(
                    "runtime environment, git commit, or implementation changed "
                    "during calibration"
                )

        def execute_job(job: CalibrationJob) -> QualificationRunnerOutcome:
            nonlocal accepted_count
            profile = manifest.profiles[job.profile_order_index].solve_profile()
            model_config = profile.effective_model_config()
            contract = _runtime_model_contract(job, profile)
            instance_dir = manifest.physical_directories[job.physical_instance_id]
            identity = {
                "calibration_run_id": calibration_run_id,
                "run_identity_sha256": run_identity_sha256,
                "source_manifest_sha256": manifest.manifest_sha256,
                "source_manifest_file_sha256": manifest.manifest_file_sha256,
                "environment_sha256": environment_sha256,
                "git_commit": commit,
                "job": job.to_payload(),
                "preregistered_stage_index": _stage_index(job),
                "execution_attempt_index": len(checkpoint_paths),
                "physical_instance_directory": str(instance_dir),
                "qualification_solve_profile": profile.to_payload(),
                "effective_model_config": to_dict(model_config),
                "qualification_model_contract": contract.to_payload(),
            }
            checkpoint_path = _checkpoint_path(checkpoints_root, job)
            stage = "pre_solve_input_revalidation"
            try:
                revalidate_runtime_environment()
                _revalidate_job_inputs(manifest, job)
                stage = "instance_load"
                instance = hooks.load_instance(
                    instance_dir, model_config, profile.demand_scale
                )
                _validate_loaded_instance(
                    instance,
                    model_config=model_config,
                    profile=profile,
                    mode=config.mode,
                )
                stage = "qualification"
                outcome = hooks.qualify_instance(
                    instance,
                    config.mode,
                    case_id=job.job_id,
                    scale=job.scale,
                    qualification_run_id=calibration_run_id,
                    instance_sha256=job.instance_sha256,
                    policy=config.policy,
                    settings=settings,
                    geometry_start_policy=config.geometry_start_policy,
                    expected_model_contract=contract,
                    fail_fast_infrastructure=True,
                )
                _validate_outcome(
                    outcome,
                    job=job,
                    qualification_run_id=calibration_run_id,
                    contract=contract,
                    policy=config.policy,
                    geometry_start_policy=config.geometry_start_policy,
                )
                stage = "post_solve_input_revalidation"
                _revalidate_job_inputs(manifest, job)
                revalidate_runtime_environment()
                result_payload = {
                    "status": "completed",
                    "completed_at_utc": hooks.now_utc(),
                    "infrastructure_error": None,
                    "outcome": outcome.to_payload(),
                }
                published_path, checkpoint_sha256 = _write_checkpoint(
                    checkpoint_path,
                    identity=identity,
                    result=result_payload,
                )
                checkpoint_paths.append(published_path)
                checkpoint_record = {
                    "job_id": job.job_id,
                    "job_order_index": job.job_order_index,
                    "preregistered_stage_index": _stage_index(job),
                    "execution_attempt_index": len(checkpoint_records),
                    "path": str(published_path),
                    "checkpoint_sha256": checkpoint_sha256,
                    "status": "completed",
                    "accepted": outcome.accepted,
                }
                checkpoint_records.append(checkpoint_record)
                records_by_job_id[job.job_id] = {
                    "job_order_index": job.job_order_index,
                    "job_id": job.job_id,
                    "profile_id": job.profile_id,
                    "scale": job.scale,
                    "development_seed_stage": job.development_seed_stage,
                    "profile_order_index": job.profile_order_index,
                    "preregistered_stage_index": _stage_index(job),
                    "selected_by_filters": True,
                    "attempted": True,
                    "accepted": outcome.accepted,
                    "not_attempted_reason_code": None,
                    "not_attempted_reason": None,
                    "checkpoint_path": str(published_path),
                    "checkpoint_sha256": checkpoint_sha256,
                    "infrastructure_failure": None,
                    "outcome": outcome.to_payload(),
                }
                accepted_count += int(outcome.accepted)
                return outcome
            except BaseException as error:
                error_payload = _infrastructure_error_payload(stage, error)
                failure_result = {
                    "status": "infrastructure_failure",
                    "completed_at_utc": hooks.now_utc(),
                    "infrastructure_error": error_payload,
                    "outcome": None,
                }
                try:
                    published_path, checkpoint_sha256 = _write_checkpoint(
                        checkpoint_path,
                        identity=identity,
                        result=failure_result,
                    )
                    checkpoint_paths.append(published_path)
                    checkpoint_record = {
                        "job_id": job.job_id,
                        "job_order_index": job.job_order_index,
                        "preregistered_stage_index": _stage_index(job),
                        "execution_attempt_index": len(checkpoint_records),
                        "path": str(published_path),
                        "checkpoint_sha256": checkpoint_sha256,
                        "status": "infrastructure_failure",
                        "accepted": False,
                    }
                    checkpoint_records.append(checkpoint_record)
                    records_by_job_id[job.job_id] = {
                        "job_order_index": job.job_order_index,
                        "job_id": job.job_id,
                        "profile_id": job.profile_id,
                        "scale": job.scale,
                        "development_seed_stage": job.development_seed_stage,
                        "profile_order_index": job.profile_order_index,
                        "preregistered_stage_index": _stage_index(job),
                        "selected_by_filters": True,
                        "attempted": True,
                        "accepted": False,
                        "not_attempted_reason_code": None,
                        "not_attempted_reason": None,
                        "checkpoint_path": str(published_path),
                        "checkpoint_sha256": checkpoint_sha256,
                        "infrastructure_failure": error_payload,
                        "outcome": None,
                    }
                    canonical_ledger = _canonical_job_ledger(
                        manifest,
                        records_by_job_id,
                        selected_job_ids,
                        termination_reason="infrastructure_abort",
                    )
                    aborted_entry = {
                        "summary_schema": DEVELOPMENT_CALIBRATION_SUMMARY_SCHEMA,
                        "development_only": True,
                        "eligible_for_formal_pool": False,
                        "calibration_run_id": calibration_run_id,
                        "run_identity_sha256": run_identity_sha256,
                        "source_manifest_sha256": manifest.manifest_sha256,
                        "status": "aborted_on_infrastructure_failure",
                        "selected_job_count": len(selected_jobs),
                        "attempted_job_count": len(checkpoint_records),
                        "completed_job_count": sum(
                            item["status"] == "completed" for item in checkpoint_records
                        ),
                        "failed_job_id": job.job_id,
                        "checkpoints": checkpoint_records,
                        "logical_job_ledger": canonical_ledger,
                        "completed_at_utc": hooks.now_utc(),
                    }
                    write_canonical_json_exclusive(
                        results_root / "batch_aborted.json",
                        {
                            "artifact_type": (
                                "mlnd_branch_cut_development_calibration_batch_summary"
                            ),
                            "artifact_schema_version": 1,
                            **aborted_entry,
                            "summary_sha256": canonical_sha256(aborted_entry),
                        },
                    )
                except BaseException as publication_error:
                    error.add_note(
                        "secondary calibration failure-artifact publication error: "
                        f"{type(publication_error).__name__}: {publication_error}"
                    )
                if isinstance(error, (MemoryError, KeyboardInterrupt, SystemExit)):
                    raise
                raise CalibrationBatchInfrastructureError(
                    f"development calibration stopped at {job.job_id} during "
                    f"{stage}: {type(error).__name__}: {error}; artifacts: "
                    f"{results_root}"
                ) from error

        jobs_by_stage_profile = {
            (_stage_index(job), job.profile_order_index): job for job in manifest.jobs
        }
        surviving_profile_indices = set(range(len(development_calibration_profiles())))
        limit_reached = False
        completed_stage_count = 0
        for stage_index, _stage in enumerate(_STAGE_ORDER):
            stage_survivors: set[int] = set()
            for profile_index in sorted(surviving_profile_indices):
                job = jobs_by_stage_profile[(stage_index, profile_index)]
                if job.job_id not in selected_job_ids:
                    continue
                if len(checkpoint_paths) >= config.max_jobs:
                    limit_reached = True
                    break
                outcome = execute_job(job)
                if outcome.accepted:
                    stage_survivors.add(profile_index)
            if limit_reached:
                break
            surviving_profile_indices = stage_survivors
            completed_stage_count += 1

        canonical_ledger = _canonical_job_ledger(
            manifest,
            records_by_job_id,
            selected_job_ids,
            termination_reason=("max_jobs_limit" if limit_reached else "completed"),
        )
        full_grid_scope = selected_job_ids == {job.job_id for job in manifest.jobs}
        ladder_complete = (
            full_grid_scope
            and not limit_reached
            and completed_stage_count == len(_STAGE_ORDER)
        )
        locked_profile_ids: tuple[str, ...] = ()
        if not ladder_complete:
            conclusion = "bounded_partial_run"
        elif len(surviving_profile_indices) < 4:
            conclusion = "development_no_go_fewer_than_four_profiles"
        else:
            conclusion = "development_profile_lock_ready"
            locked_profile_ids = tuple(
                manifest.profiles[index].profile_id
                for index in sorted(surviving_profile_indices)[:4]
            )

        summary_entry = {
            "summary_schema": DEVELOPMENT_CALIBRATION_SUMMARY_SCHEMA,
            "development_only": True,
            "eligible_for_formal_pool": False,
            "uses_a_e_algorithm_performance": False,
            "algorithm_comparison_performed": False,
            "calibration_run_id": calibration_run_id,
            "run_identity_sha256": run_identity_sha256,
            "source_manifest_sha256": manifest.manifest_sha256,
            "source_manifest_file_sha256": manifest.manifest_file_sha256,
            "environment_sha256": environment_sha256,
            "git_commit": commit,
            "status": "completed",
            "calibration_conclusion": conclusion,
            "ladder_complete": ladder_complete,
            "completed_stage_count": completed_stage_count,
            "max_jobs_limit_reached": limit_reached,
            "selected_job_count": len(selected_jobs),
            "attempted_job_count": len(checkpoint_paths),
            "unattempted_job_count": (
                EXPECTED_LOGICAL_JOB_COUNT - len(checkpoint_paths)
            ),
            "accepted_job_count": accepted_count,
            "rejected_attempted_job_count": len(checkpoint_paths) - accepted_count,
            "surviving_profile_order_indices": sorted(surviving_profile_indices),
            "locked_development_profile_ids": list(locked_profile_ids),
            "formal_pool_created": False,
            "checkpoints": checkpoint_records,
            "logical_job_ledger": canonical_ledger,
            "completed_at_utc": hooks.now_utc(),
        }
        summary_path = write_canonical_json_exclusive(
            results_root / "batch_summary.json",
            {
                "artifact_type": "mlnd_branch_cut_development_calibration_batch_summary",
                "artifact_schema_version": 1,
                **summary_entry,
                "summary_sha256": canonical_sha256(summary_entry),
            },
        )
        return DevelopmentCalibrationRunResult(
            results_root=results_root,
            run_identity_path=run_identity_path,
            summary_path=summary_path,
            calibration_run_id=calibration_run_id,
            selected_job_count=len(selected_jobs),
            attempted_job_count=len(checkpoint_paths),
            accepted_job_count=accepted_count,
            unattempted_job_count=(EXPECTED_LOGICAL_JOB_COUNT - len(checkpoint_paths)),
            locked_profile_ids=locked_profile_ids,
            checkpoint_paths=tuple(checkpoint_paths),
        )


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("value must be positive and finite")
    return parsed


def _fraction(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("value must lie in [0, 1]")
    return parsed


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _root_cut_passes(value: str) -> int:
    parsed = int(value)
    if parsed < -1:
        raise argparse.ArgumentTypeError("CutPasses must be -1 or nonnegative")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.benchmark.branch_cut.calibration_runner",
        description=(
            "Run bounded, serial, development-only qualification-profile "
            "calibration; this never creates or modifies a formal pool."
        ),
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--max-jobs", type=_positive_int, required=True)
    parser.add_argument(
        "--time-limit", type=_positive_float, choices=(5.0,), required=True
    )
    parser.add_argument("--solver-seed", type=int, choices=(42,), required=True)
    parser.add_argument("--miqcp-method", type=int, choices=(-1,), required=True)
    parser.add_argument("--native-cuts", type=int, choices=(-1,), required=True)
    parser.add_argument(
        "--root-cut-passes", type=_root_cut_passes, choices=(2,), required=True
    )
    parser.add_argument("--mode", choices=("MX",), default="MX")
    parser.add_argument("--scale", action="append", choices=_SCALES)
    parser.add_argument(
        "--development-seed-stage",
        action="append",
        type=int,
        choices=tuple(range(len(DEVELOPMENT_LAYOUT_SEED_ORDER))),
    )
    parser.add_argument(
        "--profile-order-index",
        action="append",
        type=int,
        choices=tuple(range(len(development_calibration_profiles()))),
    )
    parser.add_argument("--concurrency", type=int, choices=(1,), default=1)
    parser.add_argument("--mip-gap", type=_fraction, default=0.01)
    parser.add_argument("--solver-output", action="store_true")
    parser.add_argument("--method", type=int, choices=tuple(range(-1, 6)), default=-1)
    parser.add_argument("--mip-focus", type=int, choices=(0, 1, 2, 3), default=0)
    parser.add_argument("--heuristics", type=_fraction, default=0.05)
    parser.add_argument("--no-rel-heur-time", type=float, default=0.0)
    parser.add_argument("--presolve", type=int, choices=(-1, 0, 1, 2), default=-1)
    parser.add_argument("--numeric-focus", type=int, choices=(0, 1, 2, 3), default=0)
    parser.add_argument("--feasibility-tolerance", type=_positive_float, default=1e-6)
    parser.add_argument("--optimality-tolerance", type=_positive_float, default=1e-6)
    parser.add_argument(
        "--integer-feasibility-tolerance", type=_positive_float, default=1e-5
    )
    parser.add_argument(
        "--qualification-optimality-tolerance", type=float, default=1e-6
    )
    parser.add_argument("--minimum-savings-fraction", type=_fraction, default=1e-3)
    parser.add_argument("--minimum-service-fraction", type=_fraction, default=1e-3)
    parser.add_argument("--minimum-h-open-fraction", type=_fraction, default=0.15)
    parser.add_argument("--maximum-h-open-fraction", type=_fraction, default=0.85)
    parser.add_argument(
        "--large-scale-open-h-target",
        nargs=2,
        type=_positive_int,
        default=(50, 60),
        metavar=("MIN", "MAX"),
    )
    parser.add_argument(
        "--geometry-start", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--geometry-start-seconds", type=_positive_float, default=10.0)
    parser.add_argument("--geometry-start-node-limit", type=int, default=500)
    parser.add_argument("--resource-lock", default=str(DEFAULT_BENCHMARK_RESOURCE_LOCK))
    return parser


def main(
    argv: list[str] | None = None,
    *,
    runner: Callable[..., DevelopmentCalibrationRunResult] = (
        run_development_calibration
    ),
) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = DevelopmentCalibrationRunConfig(
            manifest_path=Path(args.manifest),
            results_root=Path(args.results_root),
            max_jobs=args.max_jobs,
            time_limit_seconds=args.time_limit,
            solver_seed=args.solver_seed,
            miqcp_method=args.miqcp_method,
            native_cuts=args.native_cuts,
            root_cut_passes=args.root_cut_passes,
            mode=args.mode,
            scales=tuple(args.scale) if args.scale is not None else None,
            development_seed_stages=(
                tuple(args.development_seed_stage)
                if args.development_seed_stage is not None
                else None
            ),
            profile_order_indices=(
                tuple(args.profile_order_index)
                if args.profile_order_index is not None
                else None
            ),
            concurrency=args.concurrency,
            mip_gap=args.mip_gap,
            output_flag=int(args.solver_output),
            method=args.method,
            mip_focus=args.mip_focus,
            heuristics=args.heuristics,
            no_rel_heur_time=args.no_rel_heur_time,
            presolve=args.presolve,
            numeric_focus=args.numeric_focus,
            feasibility_tolerance=args.feasibility_tolerance,
            optimality_tolerance=args.optimality_tolerance,
            integer_feasibility_tolerance=args.integer_feasibility_tolerance,
            policy=QualificationPolicy(
                optimality_tolerance=args.qualification_optimality_tolerance,
                minimum_savings_fraction=args.minimum_savings_fraction,
                minimum_service_fraction=args.minimum_service_fraction,
                minimum_h_open_fraction=args.minimum_h_open_fraction,
                maximum_h_open_fraction=args.maximum_h_open_fraction,
                large_scale_open_h_target=tuple(args.large_scale_open_h_target),
            ),
            geometry_start_policy=GeometryStartPolicy(
                enabled=args.geometry_start,
                per_attempt_seconds=args.geometry_start_seconds,
                start_node_limit=args.geometry_start_node_limit,
            ),
            resource_lock=Path(args.resource_lock),
        )
        result = runner(config)
    except (
        BenchmarkResourceLockError,
        CalibrationRunnerError,
        CalibrationBatchInfrastructureError,
        ValueError,
    ) as error:
        parser.exit(2, f"development calibration run failed: {error}\n")
    print(
        canonical_json(
            {
                "calibration_run_id": result.calibration_run_id,
                "results_root": str(result.results_root),
                "run_identity_path": str(result.run_identity_path),
                "summary_path": str(result.summary_path),
                "selected_job_count": result.selected_job_count,
                "attempted_job_count": result.attempted_job_count,
                "accepted_job_count": result.accepted_job_count,
                "unattempted_job_count": result.unattempted_job_count,
                "locked_profile_ids": list(result.locked_profile_ids),
            }
        )
    )
    return 0


__all__ = [
    "DEVELOPMENT_CALIBRATION_CHECKPOINT_SCHEMA",
    "DEVELOPMENT_CALIBRATION_RUN_SCHEMA",
    "DEVELOPMENT_CALIBRATION_SUMMARY_SCHEMA",
    "CalibrationBatchInfrastructureError",
    "CalibrationJob",
    "CalibrationRunnerError",
    "CalibrationRunnerHooks",
    "DevelopmentCalibrationRunConfig",
    "DevelopmentCalibrationRunResult",
    "LoadedDevelopmentCalibrationManifest",
    "build_parser",
    "load_development_calibration_manifest",
    "main",
    "run_development_calibration",
    "select_calibration_jobs",
]


if __name__ == "__main__":
    raise SystemExit(main())
