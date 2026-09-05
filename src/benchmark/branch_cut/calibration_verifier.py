"""Independently verify development calibration result artifacts.

This module is deliberately read-only and solver-free.  It does not import or
call :mod:`calibration_runner`: the manifest, run identity, scheduler state,
checkpoints, and canonical 144-row ledger are reconstructed independently from
their frozen domain contracts.  The only shared executable rule is the pure
qualification decision function; no Gurobi adapter is imported.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
from typing import Any, Mapping, Sequence

from src.config import DataConfig, to_dict

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
    QUALIFICATION_CASE_HASH_SCHEMA,
)
from .hashing import canonical_json, canonical_sha256
from .qualification import (
    QualificationEvidence,
    QualificationPolicy,
    assess_qualification,
    objective_margin,
    service_tolerance,
)
from .qualification_geometry_start import GeometryStartPolicy


RUN_SCHEMA = "mlnd-development-calibration-run-v1"
CHECKPOINT_SCHEMA = "mlnd-development-calibration-checkpoint-v1"
SUMMARY_SCHEMA = "mlnd-development-calibration-summary-v1"
EXPECTED_PHYSICAL_INSTANCE_COUNT = 6
EXPECTED_PROFILE_COUNT = 24
EXPECTED_LOGICAL_JOB_COUNT = 144
STAGE_ORDER = (
    ("small", 0),
    ("medium", 0),
    ("large", 0),
    ("small", 1),
    ("medium", 1),
    ("large", 1),
)
NOT_REACHED_REASON = "not_reached_by_preregistered_stage_gate"
NORMAL_SOLVER_STATUSES = frozenset({"OPTIMAL", "TIME_LIMIT", "INFEASIBLE"})
NORMAL_RUNNER_REASONS = frozenset(
    {
        "geometry_start_fallback_failed",
        "no_objective_improving_incumbent",
        "no_positive_service_incumbent",
    }
)
LEGACY_IMPLEMENTATION_FILES = (
    "src/config.py",
    "src/data_processor/loader.py",
    "src/data_processor/instance.py",
    "src/model/gurobi_model.py",
    "src/benchmark/branch_cut/artifacts.py",
    "src/benchmark/branch_cut/hashing.py",
    "src/benchmark/branch_cut/case_builder.py",
    "src/benchmark/branch_cut/calibration_builder.py",
    "src/benchmark/branch_cut/calibration_runner.py",
    "src/benchmark/branch_cut/qualification.py",
    "src/benchmark/branch_cut/qualification_geometry_start.py",
    "src/benchmark/branch_cut/qualification_runner.py",
)
IMPLEMENTATION_FILES = (
    *LEGACY_IMPLEMENTATION_FILES,
    "src/benchmark/branch_cut/qualification_incumbent_reuse.py",
)


def _registered_implementation_files(recorded: Mapping[str, Any]) -> tuple[str, ...]:
    """Select only a complete registered source set for a pinned result."""

    for registered in (IMPLEMENTATION_FILES, LEGACY_IMPLEMENTATION_FILES):
        if set(recorded) == set(registered):
            return registered
    _fail("implementation file set differs from the registered runner")

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_RUN_ID_RE = re.compile(r"^development-calibration-run-([0-9a-f]{64})$")
_UTC_RE = re.compile(
    r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$"
)


class CalibrationVerificationError(RuntimeError):
    """A calibration artifact failed independent verification."""


@dataclass(frozen=True, slots=True)
class CalibrationVerificationRequest:
    """Explicit pins required to verify one immutable result directory."""

    manifest_path: Path
    results_root: Path
    expected_run_id: str
    expected_manifest_sha256: str
    expected_git_commit: str
    expected_implementation_sha256: str

    def __post_init__(self) -> None:
        if _RUN_ID_RE.fullmatch(self.expected_run_id) is None:
            raise ValueError("expected_run_id is not a calibration run identity")
        _require_sha256(
            self.expected_manifest_sha256,
            field_name="expected_manifest_sha256",
        )
        if _COMMIT_RE.fullmatch(self.expected_git_commit) is None:
            raise ValueError("expected_git_commit must be a lowercase Git digest")
        _require_sha256(
            self.expected_implementation_sha256,
            field_name="expected_implementation_sha256",
        )


@dataclass(frozen=True, slots=True)
class CalibrationVerificationReport:
    """Compact result returned only after every artifact has verified."""

    status: str
    summary_path: Path
    calibration_run_id: str
    source_manifest_sha256: str
    git_commit: str
    implementation_sha256: str
    selected_job_count: int
    attempted_job_count: int
    accepted_job_count: int
    rejected_attempted_job_count: int
    infrastructure_failure_count: int
    unattempted_job_count: int
    calibration_conclusion: str | None
    locked_development_profile_ids: tuple[str, ...]

    def to_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["summary_path"] = str(self.summary_path)
        payload["locked_development_profile_ids"] = list(
            self.locked_development_profile_ids
        )
        return payload


@dataclass(frozen=True, slots=True)
class _VerifiedManifest:
    path: Path
    root: Path
    payload: Mapping[str, Any]
    manifest_sha256: str
    manifest_file_sha256: str
    raw_path: Path
    raw_sha256: str
    physical_directories: Mapping[str, Path]
    profiles: tuple[Any, ...]
    jobs: tuple[Mapping[str, Any], ...]


@dataclass(frozen=True, slots=True)
class _VerifiedCheckpoint:
    path: Path
    payload: Mapping[str, Any]
    job: Mapping[str, Any]
    checkpoint_sha256: str
    execution_attempt_index: int
    stage_index: int
    status: str
    accepted: bool
    outcome: Mapping[str, Any] | None
    infrastructure_error: Mapping[str, Any] | None


@dataclass(frozen=True, slots=True)
class _Simulation:
    attempted_job_ids: tuple[str, ...]
    surviving_profile_indices: tuple[int, ...]
    completed_stage_count: int
    limit_reached: bool
    infrastructure_aborted: bool


def _fail(message: str) -> None:
    raise CalibrationVerificationError(message)


def _require_sha256(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        _fail(f"{field_name} must be a lowercase 64-character SHA256 digest")
    return value


def _require_mapping(value: object, *, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        _fail(f"{field_name} must be an object with string keys")
    return value


def _require_list(value: object, *, field_name: str) -> list[Any]:
    if not isinstance(value, list):
        _fail(f"{field_name} must be a list")
    return value


def _require_exact_keys(
    payload: Mapping[str, Any], expected: set[str], *, field_name: str
) -> None:
    if set(payload) != expected:
        missing = sorted(expected.difference(payload))
        extra = sorted(set(payload).difference(expected))
        _fail(f"{field_name} fields differ; missing={missing}, extra={extra}")


def _require_bool(value: object, *, field_name: str) -> bool:
    if type(value) is not bool:
        _fail(f"{field_name} must be boolean")
    return value


def _require_int(value: object, *, field_name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        _fail(f"{field_name} must be an integer >= {minimum}")
    return value


def _require_finite_number(
    value: object, *, field_name: str, minimum: float | None = None
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(f"{field_name} must be numeric")
    number = float(value)
    if not math.isfinite(number) or (minimum is not None and number < minimum):
        _fail(f"{field_name} is outside its finite numeric domain")
    return number


def _require_utc(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or _UTC_RE.fullmatch(value) is None:
        _fail(f"{field_name} must be a second-resolution UTC timestamp")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolved_path_without_symlinks(
    raw_path: str | Path,
    *,
    field_name: str,
    kind: str,
) -> Path:
    lexical = Path(raw_path).expanduser()
    if not lexical.is_absolute():
        lexical = Path.cwd() / lexical
    lexical = Path(os.path.abspath(lexical))
    current = Path(lexical.anchor)
    for component in lexical.parts[1:]:
        current /= component
        if current.is_symlink():
            _fail(f"{field_name} contains symbolic-link component {current}")
    try:
        resolved = lexical.resolve(strict=True)
    except OSError as error:
        _fail(f"{field_name} does not resolve: {error}")
    if resolved != lexical:
        _fail(f"{field_name} is not an already-resolved path")
    if kind == "file" and not resolved.is_file():
        _fail(f"{field_name} is not a regular file")
    if kind == "directory" and not resolved.is_dir():
        _fail(f"{field_name} is not a directory")
    return resolved


def _require_inside(path: Path, root: Path, *, field_name: str) -> None:
    try:
        path.relative_to(root)
    except ValueError:
        _fail(f"{field_name} escapes {root}")


def _read_canonical_object(path: Path, *, field_name: str) -> dict[str, Any]:
    try:
        raw = path.read_text(encoding="utf-8")
        payload = json.loads(raw)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        _fail(f"cannot read {field_name}: {error}")
    if not isinstance(payload, dict):
        _fail(f"{field_name} must contain one JSON object")
    try:
        expected = canonical_json(payload) + "\n"
    except (TypeError, ValueError) as error:
        _fail(f"{field_name} is not canonical-JSON-safe: {error}")
    if raw != expected:
        _fail(f"{field_name} is not canonical JSON")
    return payload


def _data_config_payload(config: DataConfig) -> dict[str, Any]:
    payload = asdict(config)
    payload["dates"] = list(config.dates)
    return payload


def _canonical_instance_hash_independent(
    instance_dir: Path,
) -> tuple[str, dict[str, str]]:
    entries = sorted(path.name for path in instance_dir.iterdir())
    if entries != sorted(CANONICAL_INSTANCE_FILES):
        _fail(
            f"canonical instance {instance_dir} files differ: observed={entries}"
        )
    file_hashes: dict[str, str] = {}
    for filename in CANONICAL_INSTANCE_FILES:
        path = _resolved_path_without_symlinks(
            instance_dir / filename,
            field_name=f"canonical input {filename}",
            kind="file",
        )
        _require_inside(path, instance_dir, field_name=f"canonical input {filename}")
        file_hashes[filename] = _sha256_file(path)
    aggregate = canonical_sha256(
        {
            "hash_schema": "mlnd-canonical-instance-files-v1",
            "ordered_files": [
                {"name": name, "sha256": file_hashes[name]}
                for name in CANONICAL_INSTANCE_FILES
            ],
        }
    )
    return aggregate, file_hashes


def _qualification_case_hash(
    instance_sha256: str, profile: Any
) -> str:
    return canonical_sha256(
        {
            "hash_schema": QUALIFICATION_CASE_HASH_SCHEMA,
            "physical_instance_sha256": instance_sha256,
            "qualification_solve_profile": profile.solve_profile().to_payload(),
            "effective_model_config_sha256": (
                profile.effective_model_config_sha256()
            ),
        }
    )


def _expected_jobs(
    physical_instances: Sequence[Mapping[str, Any]],
) -> tuple[Mapping[str, Any], ...]:
    physical_by_id = {
        str(item["physical_instance_id"]): item for item in physical_instances
    }
    profiles = development_calibration_profiles()
    jobs: list[Mapping[str, Any]] = []
    for scale_index, scale in enumerate(LOCKED_SCALE_SPECS):
        for seed_stage, seed in enumerate(DEVELOPMENT_LAYOUT_SEED_ORDER):
            physical_id = (
                f"{DEVELOPMENT_DEMAND_BLOCK.block_id}__{scale.scale}"
                f"__devseed{seed:010d}"
            )
            physical = physical_by_id[physical_id]
            for profile_index, profile in enumerate(profiles):
                instance_sha = str(physical["instance_sha256"])
                case_sha = _qualification_case_hash(instance_sha, profile)
                identity = {
                    "hash_schema": DEVELOPMENT_CALIBRATION_JOB_SCHEMA,
                    "qualification_case_sha256": case_sha,
                    "profile_id": profile.profile_id,
                    "scale_order_index": scale_index,
                    "development_seed_stage": seed_stage,
                    "profile_order_index": profile_index,
                }
                jobs.append(
                    {
                        "job_order_index": len(jobs),
                        "job_id": "development-calibration-job-"
                        + canonical_sha256(identity),
                        "profile_id": profile.profile_id,
                        "scale": scale.scale,
                        "scale_order_index": scale_index,
                        "layout_seed": seed,
                        "development_seed_stage": seed_stage,
                        "profile_order_index": profile_index,
                        "physical_instance_id": physical_id,
                        "instance_ref": physical["instance_ref"],
                        "instance_sha256": instance_sha,
                        "effective_model_config_sha256": (
                            profile.effective_model_config_sha256()
                        ),
                        "qualification_case_sha256": case_sha,
                    }
                )
    return tuple(jobs)


def _verify_manifest(
    path_input: Path, expected_manifest_sha256: str
) -> _VerifiedManifest:
    path = _resolved_path_without_symlinks(
        path_input, field_name="manifest_path", kind="file"
    )
    payload = _read_canonical_object(path, field_name="calibration manifest")
    _require_exact_keys(
        payload,
        {
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
        },
        field_name="calibration manifest",
    )
    flags = {
        "artifact_type": "mlnd_branch_cut_development_calibration_manifest",
        "artifact_schema_version": 1,
        "manifest_schema": DEVELOPMENT_CALIBRATION_MANIFEST_SCHEMA,
        "development_only": True,
        "eligible_for_formal_pool": False,
        "input_artifact_only": True,
        "contains_solver_results": False,
    }
    for key, expected in flags.items():
        if payload[key] != expected or type(payload[key]) is not type(expected):
            _fail(f"calibration manifest has invalid {key}")
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
    if payload["formal_contract"] != expected_formal_contract:
        _fail("manifest violates the development-only formal contract")
    claimed = _require_sha256(
        payload["development_calibration_manifest_sha256"],
        field_name="development_calibration_manifest_sha256",
    )
    observed = canonical_sha256(
        {
            key: value
            for key, value in payload.items()
            if key != "development_calibration_manifest_sha256"
        }
    )
    if observed != claimed or claimed != expected_manifest_sha256:
        _fail("manifest content hash does not match the explicit expected pin")

    root = _resolved_path_without_symlinks(
        path.parent, field_name="manifest root", kind="directory"
    )
    raw_input = _require_mapping(payload["raw_input"], field_name="raw_input")
    _require_exact_keys(
        raw_input,
        {"source", "path", "sha256", "size_bytes"},
        field_name="raw_input",
    )
    raw_path = _resolved_path_without_symlinks(
        str(raw_input["path"]), field_name="raw_input.path", kind="file"
    )
    if str(raw_path) != raw_input["path"]:
        _fail("raw_input.path is not the resolved absolute path")
    raw_sha = _require_sha256(raw_input["sha256"], field_name="raw_input.sha256")
    raw_size = _require_int(raw_input["size_bytes"], field_name="raw_input.size_bytes")
    if (
        raw_input["source"] != DEVELOPMENT_DEMAND_BLOCK.source
        or _sha256_file(raw_path) != raw_sha
        or raw_path.stat().st_size != raw_size
    ):
        _fail("raw input source, hash, or size mismatch")

    build = _require_mapping(payload["build_config"], field_name="build_config")
    _require_exact_keys(
        build,
        {
            "output_root",
            "tsl_raw_path",
            "candidate_hour_start",
            "candidate_hour_end",
            "site_grid_km",
        },
        field_name="build_config",
    )
    if build["output_root"] != str(root) or build["tsl_raw_path"] != str(raw_path):
        _fail("manifest build paths are not bound to verified inputs")
    hour_start = _require_int(
        build["candidate_hour_start"], field_name="candidate_hour_start"
    )
    hour_end = _require_int(
        build["candidate_hour_end"], field_name="candidate_hour_end"
    )
    site_grid = _require_finite_number(
        build["site_grid_km"], field_name="site_grid_km", minimum=0.0
    )
    if not 0 <= hour_start < hour_end <= 24 or site_grid <= 0.0:
        _fail("manifest build configuration is outside its domain")
    if payload["development_demand_block"] != DEVELOPMENT_DEMAND_BLOCK.to_payload():
        _fail("development demand block drifted")
    if payload["development_layout_seed_order"] != list(DEVELOPMENT_LAYOUT_SEED_ORDER):
        _fail("development layout seed order drifted")
    if payload["canonical_instance_files"] != list(CANONICAL_INSTANCE_FILES):
        _fail("canonical instance file order drifted")
    if payload["scale_order"] != [asdict(scale) for scale in LOCKED_SCALE_SPECS]:
        _fail("scale order drifted")
    if payload["screening_contract"] != {
        "display_origin_choices": DEVELOPMENT_DISPLAY_ORIGIN_CHOICES,
        "general_corridor_choices": DEVELOPMENT_GENERAL_CORRIDOR_CHOICES,
        "expected_screening_is_exact": DEVELOPMENT_EXPECTED_SCREENING_IS_EXACT,
    }:
        _fail("screening contract drifted")

    raw_physical = _require_list(
        payload["physical_instances"], field_name="physical_instances"
    )
    if (
        _require_int(payload["physical_instance_count"], field_name="physical count")
        != EXPECTED_PHYSICAL_INSTANCE_COUNT
        or len(raw_physical) != EXPECTED_PHYSICAL_INSTANCE_COUNT
    ):
        _fail("manifest does not contain exactly six physical instances")
    instances_root = _resolved_path_without_symlinks(
        root / "instances", field_name="instances root", kind="directory"
    )
    if sorted(path.name for path in instances_root.iterdir()) != sorted(
        plan.case_id for plan in development_instance_plans()
    ):
        _fail("instances root contains missing or extra physical directories")
    physical_items: list[Mapping[str, Any]] = []
    physical_directories: dict[str, Path] = {}
    shared_center: tuple[float, float] | None = None
    for plan, raw_item in zip(
        development_instance_plans(), raw_physical, strict=True
    ):
        item = _require_mapping(raw_item, field_name=f"physical {plan.case_id}")
        expected_ref = f"instances/{plan.case_id}"
        instance_dir = _resolved_path_without_symlinks(
            root / expected_ref,
            field_name=f"physical directory {plan.case_id}",
            kind="directory",
        )
        _require_inside(instance_dir, instances_root, field_name=expected_ref)
        aggregate, file_hashes = _canonical_instance_hash_independent(instance_dir)
        expected_config = plan.data_config(
            raw_path=raw_path,
            output_dir=instance_dir,
            candidate_hour_start=hour_start,
            candidate_hour_end=hour_end,
            site_grid_km=site_grid,
        )
        expected_config.validate()
        metadata = _read_canonical_or_plain_json(
            instance_dir / "metadata.json", field_name=f"metadata {plan.case_id}"
        )
        if (
            metadata.get("source") != expected_config.source
            or metadata.get("raw_file_sha256") != raw_sha
            or metadata.get("selected_days") != list(expected_config.dates)
            or metadata.get("data_config") != _data_config_payload(expected_config)
        ):
            _fail(f"canonical metadata/config mismatch for {plan.case_id}")
        center = (
            _require_finite_number(
                metadata.get("projection_center_lon"),
                field_name=f"{plan.case_id} projection lon",
            ),
            _require_finite_number(
                metadata.get("projection_center_lat"),
                field_name=f"{plan.case_id} projection lat",
            ),
        )
        if shared_center is None:
            shared_center = center
        elif center != shared_center:
            _fail("physical instances do not share one projection center")
        expected_item = {
            "physical_instance_id": plan.case_id,
            "development_only": True,
            "scale": plan.scale.scale,
            "development_seed_stage": DEVELOPMENT_LAYOUT_SEED_ORDER.index(
                plan.layout_seed
            ),
            "layout_seed": plan.layout_seed,
            "instance_ref": expected_ref,
            "instance_sha256": aggregate,
            "raw_file_sha256": raw_sha,
            "raw_file_size_bytes": raw_size,
            "canonical_file_sha256": file_hashes,
            "data_config": _data_config_payload(expected_config),
            "projection_center_lon": center[0],
            "projection_center_lat": center[1],
        }
        if canonical_json(item) != canonical_json(expected_item):
            _fail(f"physical instance identity mismatch for {plan.case_id}")
        physical_items.append(expected_item)
        physical_directories[plan.case_id] = instance_dir

    profiles = development_calibration_profiles()
    if len(profiles) != EXPECTED_PROFILE_COUNT:
        _fail("registered development grid does not contain 24 profiles")
    if {profile.profile_id for profile in profiles} & {
        profile.profile_id for profile in LOCKED_QUALIFICATION_PROFILES
    }:
        _fail("development profiles overlap locked formal profiles")
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
        "profile_count": EXPECTED_PROFILE_COUNT,
        "profiles": [profile.to_payload() for profile in profiles],
    }
    if canonical_json(payload["calibration_grid"]) != canonical_json(expected_grid):
        _fail("calibration profile/config grid drifted")
    expected_jobs = _expected_jobs(physical_items)
    raw_jobs = _require_list(payload["logical_job_order"], field_name="logical jobs")
    if (
        payload["logical_job_order_contract"]
        != "scale_order_then_development_seed_stage_then_profile_order"
        or _require_int(payload["logical_job_count"], field_name="logical job count")
        != EXPECTED_LOGICAL_JOB_COUNT
        or len(raw_jobs) != EXPECTED_LOGICAL_JOB_COUNT
        or canonical_json(raw_jobs) != canonical_json(expected_jobs)
    ):
        _fail("canonical 144-job identities or order drifted")
    return _VerifiedManifest(
        path=path,
        root=root,
        payload=payload,
        manifest_sha256=claimed,
        manifest_file_sha256=_sha256_file(path),
        raw_path=raw_path,
        raw_sha256=raw_sha,
        physical_directories=physical_directories,
        profiles=profiles,
        jobs=expected_jobs,
    )


def _read_canonical_or_plain_json(path: Path, *, field_name: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        _fail(f"cannot read {field_name}: {error}")
    if not isinstance(payload, dict):
        _fail(f"{field_name} must be a JSON object")
    return payload


def _stable_environment(environment: Mapping[str, Any]) -> dict[str, Any]:
    git = dict(_require_mapping(environment.get("git"), field_name="environment.git"))
    for key in (
        "worktree_dirty",
        "status_includes_untracked",
        "status_lines",
        "status_sha256",
    ):
        git.pop(key, None)
    return {**dict(environment), "git": git}


def _verify_environment(
    environment: Mapping[str, Any], request: CalibrationVerificationRequest
) -> tuple[str, str]:
    _require_exact_keys(
        environment,
        {"git", "implementation", "python", "platform", "gurobi_version"},
        field_name="run environment",
    )
    git = _require_mapping(environment["git"], field_name="environment.git")
    _require_exact_keys(
        git,
        {
            "repository_root",
            "commit",
            "worktree_dirty",
            "status_includes_untracked",
            "status_lines",
            "status_sha256",
        },
        field_name="environment.git",
    )
    if (
        git["commit"] != request.expected_git_commit
        or git["worktree_dirty"] is not False
        or git["status_includes_untracked"] is not True
        or git["status_lines"] != []
        or git["status_sha256"] != canonical_sha256([])
    ):
        _fail("recorded Git commit or clean-worktree observation is invalid")
    repository_root = _resolved_path_without_symlinks(
        str(git["repository_root"]),
        field_name="environment.git.repository_root",
        kind="directory",
    )
    if str(repository_root) != git["repository_root"]:
        _fail("recorded repository root is not resolved")

    implementation = _require_mapping(
        environment["implementation"], field_name="environment.implementation"
    )
    _require_exact_keys(
        implementation,
        {"ordered_file_sha256", "implementation_sha256"},
        field_name="environment.implementation",
    )
    ordered = _require_mapping(
        implementation["ordered_file_sha256"],
        field_name="implementation.ordered_file_sha256",
    )
    # Canonical JSON sorts object keys, so the serialized mapping cannot retain
    # the tuple's insertion order.  The exact registered set plus the canonical
    # aggregate hash is the stable contract.
    registered_files = _registered_implementation_files(ordered)
    observed_mapping: dict[str, str] = {}
    for relative in registered_files:
        claimed = _require_sha256(
            ordered[relative], field_name=f"implementation hash {relative}"
        )
        source = _resolved_path_without_symlinks(
            repository_root / relative,
            field_name=f"implementation file {relative}",
            kind="file",
        )
        _require_inside(source, repository_root, field_name=relative)
        observed = _sha256_file(source)
        if observed != claimed:
            _fail(f"implementation file changed: {relative}")
        observed_mapping[relative] = observed
    implementation_sha = _require_sha256(
        implementation["implementation_sha256"],
        field_name="implementation_sha256",
    )
    if (
        implementation_sha != canonical_sha256(observed_mapping)
        or implementation_sha != request.expected_implementation_sha256
    ):
        _fail("implementation aggregate does not match the explicit expected pin")
    if not isinstance(environment["gurobi_version"], str) or not environment[
        "gurobi_version"
    ]:
        _fail("recorded Gurobi version must be nonempty")
    _require_mapping(environment["python"], field_name="environment.python")
    _require_mapping(environment["platform"], field_name="environment.platform")
    return request.expected_git_commit, implementation_sha


def _expected_native_settings() -> dict[str, Any]:
    return {
        "time_limit_seconds": 5.0,
        "mip_gap": 0.01,
        "solver_seed": 42,
        "threads": 1,
        "output_flag": 0,
        "method": -1,
        "miqcp_method": -1,
        "mip_focus": 0,
        "heuristics": 0.05,
        "no_rel_heur_time": 0.0,
        "presolve": -1,
        "native_cuts": -1,
        "root_cut_passes": 2,
        "numeric_focus": 0,
        "feasibility_tolerance": 1e-6,
        "optimality_tolerance": 1e-6,
        "integer_feasibility_tolerance": 1e-5,
    }


def _selector_values(
    value: object,
    *,
    field_name: str,
    registered_order: Sequence[Any],
) -> tuple[Any, ...] | None:
    if value is None:
        return None
    values = _require_list(value, field_name=field_name)
    if not values or len(values) != len(set(values)):
        _fail(f"{field_name} must be a nonempty unique selection")
    selected = set(values)
    expected = tuple(item for item in registered_order if item in selected)
    if tuple(values) != expected or len(expected) != len(values):
        _fail(f"{field_name} is out of grid or not in registered order")
    return expected


def _verify_run_identity(
    path: Path,
    manifest: _VerifiedManifest,
    request: CalibrationVerificationRequest,
) -> tuple[Mapping[str, Any], str, tuple[Mapping[str, Any], ...], int, str, str]:
    payload = _read_canonical_object(path, field_name="run identity")
    _require_exact_keys(
        payload,
        {
            "artifact_type",
            "artifact_schema_version",
            "calibration_run_id",
            "run_identity",
            "run_identity_sha256",
        },
        field_name="run identity artifact",
    )
    if (
        payload["artifact_type"]
        != "mlnd_branch_cut_development_calibration_run_identity"
        or payload["artifact_schema_version"] != 1
    ):
        _fail("run identity artifact type/schema is invalid")
    identity = _require_mapping(payload["run_identity"], field_name="run_identity")
    _require_exact_keys(
        identity,
        {
            "run_schema",
            "development_only",
            "eligible_for_formal_pool",
            "uses_formal_pool",
            "uses_locked_qualification_profiles",
            "uses_a_e_algorithm_performance",
            "algorithm_comparison_performed",
            "source_manifest_path",
            "source_manifest_sha256",
            "source_manifest_file_sha256",
            "source_raw_sha256",
            "mode",
            "concurrency",
            "policy",
            "native_settings",
            "geometry_start_policy",
            "selectors",
            "selected_jobs",
            "preregistered_ladder",
            "environment",
            "environment_sha256",
            "stable_environment_sha256",
            "git_commit",
        },
        field_name="run_identity",
    )
    fixed = {
        "run_schema": RUN_SCHEMA,
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
        "mode": "MX",
        "concurrency": 1,
        "policy": QualificationPolicy(
            large_scale_open_h_target=(50, 60)
        ).to_payload(),
        "native_settings": _expected_native_settings(),
        "geometry_start_policy": GeometryStartPolicy(enabled=False).to_payload(),
        "preregistered_ladder": {
            "stage_order": [
                {
                    "stage_index": index,
                    "scale": scale,
                    "development_seed_stage": seed_stage,
                }
                for index, (scale, seed_stage) in enumerate(STAGE_ORDER)
            ],
            "advance_on_fresh_accepted_outcome_only": True,
            "unattempted_reason": NOT_REACHED_REASON,
            "lock_first_profile_count": 4,
            "lock_order": "original_profile_order_index",
        },
        "git_commit": request.expected_git_commit,
    }
    for key, expected in fixed.items():
        if canonical_json(identity[key]) != canonical_json(expected):
            _fail(f"run identity field {key} drifted")

    selectors = _require_mapping(identity["selectors"], field_name="selectors")
    _require_exact_keys(
        selectors,
        {"scales", "development_seed_stages", "profile_order_indices", "max_jobs"},
        field_name="selectors",
    )
    scales = _selector_values(
        selectors["scales"],
        field_name="selectors.scales",
        registered_order=tuple(scale.scale for scale in LOCKED_SCALE_SPECS),
    )
    seed_stages = _selector_values(
        selectors["development_seed_stages"],
        field_name="selectors.development_seed_stages",
        registered_order=tuple(range(len(DEVELOPMENT_LAYOUT_SEED_ORDER))),
    )
    profile_indices = _selector_values(
        selectors["profile_order_indices"],
        field_name="selectors.profile_order_indices",
        registered_order=tuple(range(EXPECTED_PROFILE_COUNT)),
    )
    max_jobs = _require_int(selectors["max_jobs"], field_name="max_jobs", minimum=1)
    scale_set = set(scales or (scale.scale for scale in LOCKED_SCALE_SPECS))
    seed_set = set(seed_stages or range(len(DEVELOPMENT_LAYOUT_SEED_ORDER)))
    profile_set = set(profile_indices or range(EXPECTED_PROFILE_COUNT))
    selected = tuple(
        job
        for job in manifest.jobs
        if job["scale"] in scale_set
        and job["development_seed_stage"] in seed_set
        and job["profile_order_index"] in profile_set
    )
    if not selected or canonical_json(identity["selected_jobs"]) != canonical_json(
        selected
    ):
        _fail("selected_jobs does not match the independently applied selectors")

    environment = _require_mapping(identity["environment"], field_name="environment")
    commit, implementation_sha = _verify_environment(environment, request)
    environment_sha = _require_sha256(
        identity["environment_sha256"], field_name="environment_sha256"
    )
    stable_sha = _require_sha256(
        identity["stable_environment_sha256"],
        field_name="stable_environment_sha256",
    )
    if environment_sha != canonical_sha256(environment):
        _fail("environment_sha256 does not hash the embedded environment")
    if stable_sha != canonical_sha256(_stable_environment(environment)):
        _fail("stable_environment_sha256 does not hash the stable environment")
    run_sha = _require_sha256(
        payload["run_identity_sha256"], field_name="run_identity_sha256"
    )
    if run_sha != canonical_sha256(identity):
        _fail("run identity self-hash mismatch")
    run_id = "development-calibration-run-" + run_sha
    if payload["calibration_run_id"] != run_id or run_id != request.expected_run_id:
        _fail("calibration run ID does not match its hash and explicit pin")
    return identity, run_sha, selected, max_jobs, commit, implementation_sha


def _stage_index(job: Mapping[str, Any]) -> int:
    pair = (job["scale"], job["development_seed_stage"])
    try:
        return STAGE_ORDER.index(pair)
    except ValueError:
        _fail(f"job {job['job_id']} is outside the stage ladder")


def _expected_contract(job: Mapping[str, Any], profile: Any) -> dict[str, Any]:
    solve = profile.solve_profile()
    return {
        "qualification_case_sha256": job["qualification_case_sha256"],
        "effective_model_config_sha256": job["effective_model_config_sha256"],
        "demand_scale": solve.demand_scale,
        "locker_fixed_cost_per_hour": solve.locker_fixed_cost_per_hour,
        "display_origin_choices": solve.display_origin_choices,
        "general_corridor_choices": solve.general_corridor_choices,
        "screening_is_exact": False,
    }


def _normalize_nonfinite(value: object) -> object:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: _normalize_nonfinite(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize_nonfinite(item) for item in value]
    return value


def _evidence_from_payload(payload: Mapping[str, Any]) -> QualificationEvidence:
    values = dict(payload)
    for key in (
        "courier_only_objective",
        "witness_objective",
        "total_demand_per_hour",
        "witness_service_per_hour",
    ):
        if values[key] is None:
            values[key] = math.nan
    try:
        return QualificationEvidence(**values)
    except (TypeError, ValueError) as error:
        _fail(f"qualification evidence is invalid: {error}")


def _verify_audit(
    value: object,
    *,
    field_name: str,
    contract: Mapping[str, Any],
) -> Mapping[str, Any]:
    audit = _require_mapping(value, field_name=field_name)
    _require_exact_keys(
        audit,
        {
            "effective_model_config_sha256",
            "demand_scale",
            "locker_fixed_cost_per_hour",
            "display_origin_choices",
            "general_corridor_choices",
            "screening_is_exact",
            "model_fingerprint",
        },
        field_name=field_name,
    )
    expected = {
        key: contract[key]
        for key in (
            "effective_model_config_sha256",
            "demand_scale",
            "locker_fixed_cost_per_hour",
            "display_origin_choices",
            "general_corridor_choices",
            "screening_is_exact",
        )
    }
    if any(audit[key] != expected[key] for key in expected):
        _fail(f"{field_name} does not match the qualification contract")
    fingerprint = audit["model_fingerprint"]
    if fingerprint is not None and (
        isinstance(fingerprint, bool) or not isinstance(fingerprint, int)
    ):
        _fail(f"{field_name}.model_fingerprint must be an integer or null")
    return audit


def _verify_outcome(
    outcome_value: object,
    *,
    job: Mapping[str, Any],
    run_id: str,
    policy_payload: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> tuple[Mapping[str, Any], bool]:
    outcome = _require_mapping(outcome_value, field_name="checkpoint outcome")
    _require_exact_keys(
        outcome,
        {
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
        },
        field_name="checkpoint outcome",
    )
    accepted = _require_bool(outcome["accepted"], field_name="outcome.accepted")
    reasons = _require_list(outcome["rejection_reasons"], field_name="reasons")
    if any(not isinstance(reason, str) or not reason for reason in reasons):
        _fail("outcome rejection reasons must be nonempty strings")
    if len(reasons) != len(set(reasons)) or accepted != (not reasons):
        _fail("outcome accepted/rejection state is inconsistent")
    if outcome["expected_model_contract"] != contract:
        _fail("outcome expected model contract drifted")
    witness_audit = _verify_audit(
        outcome["witness_model_audit"],
        field_name="witness_model_audit",
        contract=contract,
    )
    fresh_performed = _require_bool(
        outcome["fresh_model_validation_performed"],
        field_name="fresh_model_validation_performed",
    )
    fresh_value = outcome["fresh_model_audit"]
    fresh_audit: Mapping[str, Any] | None = None
    if fresh_performed:
        fresh_audit = _verify_audit(
            fresh_value, field_name="fresh_model_audit", contract=contract
        )
        if (
            witness_audit["model_fingerprint"] is None
            or fresh_audit["model_fingerprint"] is None
            or witness_audit["model_fingerprint"]
            != fresh_audit["model_fingerprint"]
        ):
            _fail("fresh/witness model fingerprints differ")
    elif fresh_value is not None:
        _fail("fresh model audit exists without fresh validation")

    evidence_payload = _require_mapping(
        outcome["evidence"], field_name="outcome.evidence"
    )
    _require_exact_keys(
        evidence_payload,
        {
            "case_id",
            "scale",
            "courier_only_objective",
            "witness_objective",
            "total_demand_per_hour",
            "witness_service_per_hour",
            "open_facilities",
            "positive_cost_open_facilities",
            "candidate_h",
            "open_h",
            "witness_feasible",
            "qualification_run_id",
            "instance_sha256",
            "active_h",
            "active_h_threshold_per_hour",
            "qualification_case_sha256",
            "effective_model_config_sha256",
            "screening_is_exact",
        },
        field_name="outcome.evidence",
    )
    identity_expected = {
        "case_id": job["job_id"],
        "scale": job["scale"],
        "qualification_run_id": run_id,
        "instance_sha256": job["instance_sha256"],
        "qualification_case_sha256": job["qualification_case_sha256"],
        "effective_model_config_sha256": job["effective_model_config_sha256"],
        "screening_is_exact": False,
    }
    if any(evidence_payload[key] != value for key, value in identity_expected.items()):
        _fail("qualification evidence identity does not match its job")
    evidence = _evidence_from_payload(evidence_payload)
    try:
        target = policy_payload["large_scale_open_h_target"]
        policy = QualificationPolicy(
            optimality_tolerance=policy_payload["optimality_tolerance"],
            minimum_savings_fraction=policy_payload["minimum_savings_fraction"],
            minimum_service_fraction=policy_payload["minimum_service_fraction"],
            minimum_h_open_fraction=policy_payload["minimum_h_open_fraction"],
            maximum_h_open_fraction=policy_payload["maximum_h_open_fraction"],
            large_scale_open_h_target=tuple(target) if target is not None else None,
        )
        expected_decision = _normalize_nonfinite(
            assess_qualification(evidence, policy).to_payload()
        )
    except (KeyError, TypeError, ValueError) as error:
        _fail(f"cannot independently assess qualification evidence: {error}")
    decision = _require_mapping(outcome["decision"], field_name="outcome.decision")
    if canonical_json(decision) != canonical_json(expected_decision):
        _fail("stored decision differs from independently assessed evidence")
    decision_reasons = _require_list(
        decision["rejection_reasons"], field_name="decision.rejection_reasons"
    )
    cursor = iter(reasons)
    if not all(any(candidate == reason for candidate in cursor) for reason in decision_reasons):
        _fail("decision rejection reasons are not an ordered outcome subset")
    extras = set(reasons).difference(decision_reasons)
    if not extras.issubset(NORMAL_RUNNER_REASONS):
        _fail("normal checkpoint contains infrastructure-level runner reasons")

    witness_status = outcome["witness_status"]
    validation_status = outcome["validation_status"]
    if witness_status not in NORMAL_SOLVER_STATUSES:
        _fail("completed outcome contains an abnormal witness status")
    if fresh_performed:
        if validation_status not in NORMAL_SOLVER_STATUSES:
            _fail("fresh validation status is not a normal solver status")
    elif validation_status != "NOT_RUN":
        _fail("validation status exists without fresh validation")
    if outcome["exception_messages"] != []:
        _fail("completed outcome contains exception messages")
    captured = _require_int(
        outcome["captured_variable_count"], field_name="captured_variable_count"
    )
    if accepted and (
        not fresh_performed
        or decision["accepted"] is not True
        or evidence_payload["witness_feasible"] is not True
        or captured <= 0
    ):
        _fail("accepted outcome lacks a fresh validated complete witness")

    required_service = _require_finite_number(
        outcome["required_service_per_hour"],
        field_name="required_service_per_hour",
        minimum=0.0,
    )
    if required_service <= 0.0:
        _fail("required service must be positive")
    demand = evidence.total_demand_per_hour
    expected_service = (
        max(service_tolerance(demand), policy.minimum_service_fraction * demand)
        if math.isfinite(demand) and demand >= 0.0
        else 1e-3
    )
    if required_service != expected_service:
        _fail("required service does not match demand and policy")
    courier = evidence.courier_only_objective
    expected_upper: float | None = None
    if math.isfinite(courier):
        improvement = max(
            objective_margin(courier, policy.optimality_tolerance),
            policy.minimum_savings_fraction * courier if courier > 0.0 else 0.0,
        )
        expected_upper = courier - improvement
    if outcome["required_objective_upper_bound"] != expected_upper:
        _fail("required objective bound does not match baseline and policy")

    temporary_used = _require_bool(
        outcome["temporary_constraint_used"], field_name="temporary_constraint_used"
    )
    temporary_removed = outcome["temporary_constraint_removed"]
    if (temporary_used and temporary_removed is not True) or (
        not temporary_used and temporary_removed is not None
    ):
        _fail("temporary witness-constraint cleanup state is inconsistent")
    _require_bool(
        outcome["feasibility_objective_used"],
        field_name="feasibility_objective_used",
    )
    expected_geometry = GeometryStartPolicy(enabled=False).to_payload()
    if (
        outcome["geometry_start_policy"] != expected_geometry
        or outcome["geometry_start_attempts"] != []
        or outcome["geometry_start_selected_plan_sha256"] is not None
    ):
        _fail("disabled geometry-start contract contains activity")
    timings = _require_mapping(outcome["phase_timings"], field_name="phase_timings")
    expected_timing_keys = {
        "witness_adapter_initialization_seconds",
        "baseline_extraction_seconds",
        "native_witness_solve_seconds",
        "temporary_constraint_add_seconds",
        "constrained_witness_solve_seconds",
        "geometry_start_planning_seconds",
        "geometry_start_installation_seconds",
        "geometry_start_solve_seconds",
        "geometry_start_cleanup_seconds",
        "complete_incumbent_capture_seconds",
        "temporary_constraint_removal_seconds",
        "witness_model_disposal_seconds",
        "fresh_adapter_initialization_seconds",
        "fresh_baseline_extraction_seconds",
        "complete_start_installation_seconds",
        "fresh_validation_solve_seconds",
        "fresh_model_disposal_seconds",
        "total_seconds",
    }
    _require_exact_keys(timings, expected_timing_keys, field_name="phase_timings")
    for key, value in timings.items():
        _require_finite_number(value, field_name=f"phase_timings.{key}", minimum=0.0)
    return outcome, accepted


def _verify_checkpoint(
    path: Path,
    *,
    expected_job: Mapping[str, Any],
    manifest: _VerifiedManifest,
    run_identity: Mapping[str, Any],
    run_sha: str,
    execution_attempt_index: int,
) -> _VerifiedCheckpoint:
    payload = _read_canonical_object(path, field_name=f"checkpoint {path.name}")
    _require_exact_keys(
        payload,
        {
            "artifact_type",
            "artifact_schema_version",
            "checkpoint_schema",
            "development_only",
            "eligible_for_formal_pool",
            "contains_algorithm_comparison_results",
            "identity",
            "result",
            "checkpoint_sha256",
        },
        field_name=f"checkpoint {path.name}",
    )
    if (
        payload["artifact_type"]
        != "mlnd_branch_cut_development_calibration_checkpoint"
        or payload["artifact_schema_version"] != 1
        or payload["checkpoint_schema"] != CHECKPOINT_SCHEMA
        or payload["development_only"] is not True
        or payload["eligible_for_formal_pool"] is not False
        or payload["contains_algorithm_comparison_results"] is not False
    ):
        _fail(f"checkpoint {path.name} violates its development schema")
    identity = _require_mapping(payload["identity"], field_name="checkpoint identity")
    _require_exact_keys(
        identity,
        {
            "calibration_run_id",
            "run_identity_sha256",
            "source_manifest_sha256",
            "source_manifest_file_sha256",
            "environment_sha256",
            "git_commit",
            "job",
            "preregistered_stage_index",
            "execution_attempt_index",
            "physical_instance_directory",
            "qualification_solve_profile",
            "effective_model_config",
            "qualification_model_contract",
        },
        field_name="checkpoint identity",
    )
    profile = manifest.profiles[int(expected_job["profile_order_index"])]
    expected_identity = {
        "calibration_run_id": "development-calibration-run-" + run_sha,
        "run_identity_sha256": run_sha,
        "source_manifest_sha256": manifest.manifest_sha256,
        "source_manifest_file_sha256": manifest.manifest_file_sha256,
        "environment_sha256": run_identity["environment_sha256"],
        "git_commit": run_identity["git_commit"],
        "job": expected_job,
        "preregistered_stage_index": _stage_index(expected_job),
        "execution_attempt_index": execution_attempt_index,
        "physical_instance_directory": str(
            manifest.physical_directories[expected_job["physical_instance_id"]]
        ),
        "qualification_solve_profile": profile.solve_profile().to_payload(),
        "effective_model_config": to_dict(profile.effective_model_config()),
        "qualification_model_contract": _expected_contract(expected_job, profile),
    }
    if canonical_json(identity) != canonical_json(expected_identity):
        _fail(f"checkpoint identity mismatch for {expected_job['job_id']}")
    checkpoint_sha = _require_sha256(
        payload["checkpoint_sha256"], field_name="checkpoint_sha256"
    )
    result = _require_mapping(payload["result"], field_name="checkpoint result")
    _require_exact_keys(
        result,
        {"status", "completed_at_utc", "infrastructure_error", "outcome"},
        field_name="checkpoint result",
    )
    _require_utc(result["completed_at_utc"], field_name="checkpoint completed_at_utc")
    expected_hash = canonical_sha256({"identity": identity, "result": result})
    if checkpoint_sha != expected_hash:
        _fail(f"checkpoint self-hash mismatch for {expected_job['job_id']}")
    status = result["status"]
    outcome: Mapping[str, Any] | None = None
    infrastructure: Mapping[str, Any] | None = None
    accepted = False
    if status == "completed":
        if result["infrastructure_error"] is not None:
            _fail("completed checkpoint contains an infrastructure error")
        outcome, accepted = _verify_outcome(
            result["outcome"],
            job=expected_job,
            run_id=expected_identity["calibration_run_id"],
            policy_payload=run_identity["policy"],
            contract=expected_identity["qualification_model_contract"],
        )
    elif status == "infrastructure_failure":
        if result["outcome"] is not None:
            _fail("infrastructure checkpoint contains a qualification outcome")
        infrastructure = _require_mapping(
            result["infrastructure_error"], field_name="infrastructure_error"
        )
        _require_exact_keys(
            infrastructure,
            {"stage", "type", "message", "qualification_infrastructure_error"},
            field_name="infrastructure_error",
        )
        if infrastructure["stage"] not in {
            "pre_solve_input_revalidation",
            "instance_load",
            "qualification",
            "post_solve_input_revalidation",
        }:
            _fail("infrastructure failure stage is invalid")
        if any(
            not isinstance(infrastructure[key], str) or not infrastructure[key]
            for key in ("type", "message")
        ):
            _fail("infrastructure error type/message must be nonempty")
        _require_bool(
            infrastructure["qualification_infrastructure_error"],
            field_name="qualification_infrastructure_error",
        )
    else:
        _fail(f"checkpoint has invalid status {status!r}")
    return _VerifiedCheckpoint(
        path=path,
        payload=payload,
        job=expected_job,
        checkpoint_sha256=checkpoint_sha,
        execution_attempt_index=execution_attempt_index,
        stage_index=_stage_index(expected_job),
        status=status,
        accepted=accepted,
        outcome=outcome,
        infrastructure_error=infrastructure,
    )


def _simulate_ladder(
    *,
    manifest: _VerifiedManifest,
    selected_job_ids: set[str],
    checkpoints: Sequence[_VerifiedCheckpoint],
    max_jobs: int,
) -> _Simulation:
    by_stage_profile = {
        (_stage_index(job), job["profile_order_index"]): job for job in manifest.jobs
    }
    cursor = 0
    survivors = set(range(EXPECTED_PROFILE_COUNT))
    completed_stage_count = 0
    limit_reached = False
    aborted = False
    for stage_index in range(len(STAGE_ORDER)):
        stage_survivors: set[int] = set()
        for profile_index in sorted(survivors):
            job = by_stage_profile[(stage_index, profile_index)]
            if job["job_id"] not in selected_job_ids:
                continue
            if cursor >= max_jobs:
                limit_reached = True
                break
            if cursor >= len(checkpoints):
                _fail(
                    "checkpoint sequence ended before the scheduler reached a "
                    "registered termination condition"
                )
            checkpoint = checkpoints[cursor]
            if checkpoint.job["job_id"] != job["job_id"]:
                _fail("checkpoint order differs from the six-stage ladder")
            cursor += 1
            if checkpoint.status == "infrastructure_failure":
                aborted = True
                break
            if checkpoint.accepted:
                stage_survivors.add(profile_index)
        if aborted or limit_reached:
            break
        survivors = stage_survivors
        completed_stage_count += 1
    if cursor != len(checkpoints):
        _fail("checkpoint list contains attempts after scheduler termination")
    if aborted and checkpoints[-1].status != "infrastructure_failure":
        _fail("infrastructure failure is not the final attempted job")
    if not aborted and any(
        checkpoint.status == "infrastructure_failure" for checkpoint in checkpoints
    ):
        _fail("infrastructure checkpoint did not abort the ladder")
    return _Simulation(
        attempted_job_ids=tuple(
            str(checkpoint.job["job_id"]) for checkpoint in checkpoints
        ),
        surviving_profile_indices=tuple(sorted(survivors)),
        completed_stage_count=completed_stage_count,
        limit_reached=limit_reached,
        infrastructure_aborted=aborted,
    )


def _checkpoint_record(checkpoint: _VerifiedCheckpoint) -> dict[str, Any]:
    return {
        "job_id": checkpoint.job["job_id"],
        "job_order_index": checkpoint.job["job_order_index"],
        "preregistered_stage_index": checkpoint.stage_index,
        "execution_attempt_index": checkpoint.execution_attempt_index,
        "path": str(checkpoint.path),
        "checkpoint_sha256": checkpoint.checkpoint_sha256,
        "status": checkpoint.status,
        "accepted": checkpoint.accepted,
    }


def _ledger_record(checkpoint: _VerifiedCheckpoint) -> dict[str, Any]:
    job = checkpoint.job
    return {
        "job_order_index": job["job_order_index"],
        "job_id": job["job_id"],
        "profile_id": job["profile_id"],
        "scale": job["scale"],
        "development_seed_stage": job["development_seed_stage"],
        "profile_order_index": job["profile_order_index"],
        "preregistered_stage_index": checkpoint.stage_index,
        "selected_by_filters": True,
        "attempted": True,
        "accepted": checkpoint.accepted,
        "not_attempted_reason_code": None,
        "not_attempted_reason": None,
        "checkpoint_path": str(checkpoint.path),
        "checkpoint_sha256": checkpoint.checkpoint_sha256,
        "infrastructure_failure": checkpoint.infrastructure_error,
        "outcome": checkpoint.outcome,
    }


def _canonical_ledger(
    *,
    manifest: _VerifiedManifest,
    selected_job_ids: set[str],
    checkpoints: Sequence[_VerifiedCheckpoint],
    termination_reason: str,
) -> list[dict[str, Any]]:
    records = {str(item.job["job_id"]): _ledger_record(item) for item in checkpoints}
    by_stage_profile = {
        (_stage_index(job), job["profile_order_index"]): job for job in manifest.jobs
    }
    ledger: list[dict[str, Any]] = []
    for job in manifest.jobs:
        job_id = str(job["job_id"])
        if job_id in records:
            ledger.append(records[job_id])
            continue
        if job_id not in selected_job_ids:
            reason_code = reason = "selector_excluded"
        else:
            blocked = any(
                (
                    predecessor := records.get(
                        str(
                            by_stage_profile[
                                (prior_stage, job["profile_order_index"])
                            ]["job_id"]
                        )
                    )
                )
                is None
                or predecessor["accepted"] is not True
                for prior_stage in range(_stage_index(job))
            )
            if blocked:
                reason_code = "stage_gate"
                reason = NOT_REACHED_REASON
            elif termination_reason == "max_jobs_limit":
                reason_code = reason = "max_jobs_limit"
            elif termination_reason == "infrastructure_abort":
                reason_code = reason = "infrastructure_abort"
            else:
                reason_code = "stage_gate"
                reason = NOT_REACHED_REASON
        ledger.append(
            {
                "job_order_index": job["job_order_index"],
                "job_id": job_id,
                "profile_id": job["profile_id"],
                "scale": job["scale"],
                "development_seed_stage": job["development_seed_stage"],
                "profile_order_index": job["profile_order_index"],
                "preregistered_stage_index": _stage_index(job),
                "selected_by_filters": job_id in selected_job_ids,
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
    return ledger


def _verify_summary(
    path: Path,
    *,
    manifest: _VerifiedManifest,
    run_identity: Mapping[str, Any],
    run_sha: str,
    selected_jobs: Sequence[Mapping[str, Any]],
    max_jobs: int,
    checkpoints: Sequence[_VerifiedCheckpoint],
) -> tuple[Mapping[str, Any], _Simulation, str | None, tuple[str, ...]]:
    payload = _read_canonical_object(path, field_name="calibration batch summary")
    if (
        payload.get("artifact_type")
        != "mlnd_branch_cut_development_calibration_batch_summary"
        or payload.get("artifact_schema_version") != 1
        or payload.get("summary_schema") != SUMMARY_SCHEMA
        or payload.get("development_only") is not True
        or payload.get("eligible_for_formal_pool") is not False
    ):
        _fail("summary violates its development-only schema")
    summary_sha = _require_sha256(payload.get("summary_sha256"), field_name="summary_sha256")
    entry = {
        key: value
        for key, value in payload.items()
        if key not in {"artifact_type", "artifact_schema_version", "summary_sha256"}
    }
    if summary_sha != canonical_sha256(entry):
        _fail("summary self-hash mismatch")
    selected_ids = {str(job["job_id"]) for job in selected_jobs}
    simulation = _simulate_ladder(
        manifest=manifest,
        selected_job_ids=selected_ids,
        checkpoints=checkpoints,
        max_jobs=max_jobs,
    )
    checkpoint_records = [_checkpoint_record(item) for item in checkpoints]
    if payload.get("checkpoints") != checkpoint_records:
        _fail("summary checkpoint records differ from checkpoint artifacts")
    status = payload.get("status")
    conclusion: str | None = None
    locked_ids: tuple[str, ...] = ()
    if status == "completed":
        _require_exact_keys(
            payload,
            {
                "artifact_type",
                "artifact_schema_version",
                "summary_schema",
                "development_only",
                "eligible_for_formal_pool",
                "uses_a_e_algorithm_performance",
                "algorithm_comparison_performed",
                "calibration_run_id",
                "run_identity_sha256",
                "source_manifest_sha256",
                "source_manifest_file_sha256",
                "environment_sha256",
                "git_commit",
                "status",
                "calibration_conclusion",
                "ladder_complete",
                "completed_stage_count",
                "max_jobs_limit_reached",
                "selected_job_count",
                "attempted_job_count",
                "unattempted_job_count",
                "accepted_job_count",
                "rejected_attempted_job_count",
                "surviving_profile_order_indices",
                "locked_development_profile_ids",
                "formal_pool_created",
                "checkpoints",
                "logical_job_ledger",
                "completed_at_utc",
                "summary_sha256",
            },
            field_name="completed summary",
        )
        if simulation.infrastructure_aborted:
            _fail("completed summary contains an infrastructure checkpoint")
        full_scope = selected_ids == {str(job["job_id"]) for job in manifest.jobs}
        ladder_complete = (
            full_scope
            and not simulation.limit_reached
            and simulation.completed_stage_count == len(STAGE_ORDER)
        )
        if not ladder_complete:
            conclusion = "bounded_partial_run"
        elif len(simulation.surviving_profile_indices) < 4:
            conclusion = "development_no_go_fewer_than_four_profiles"
        else:
            conclusion = "development_profile_lock_ready"
            locked_ids = tuple(
                manifest.profiles[index].profile_id
                for index in simulation.surviving_profile_indices[:4]
            )
        accepted_count = sum(item.accepted for item in checkpoints)
        ledger = _canonical_ledger(
            manifest=manifest,
            selected_job_ids=selected_ids,
            checkpoints=checkpoints,
            termination_reason=(
                "max_jobs_limit" if simulation.limit_reached else "completed"
            ),
        )
        expected_entry = {
            "summary_schema": SUMMARY_SCHEMA,
            "development_only": True,
            "eligible_for_formal_pool": False,
            "uses_a_e_algorithm_performance": False,
            "algorithm_comparison_performed": False,
            "calibration_run_id": "development-calibration-run-" + run_sha,
            "run_identity_sha256": run_sha,
            "source_manifest_sha256": manifest.manifest_sha256,
            "source_manifest_file_sha256": manifest.manifest_file_sha256,
            "environment_sha256": run_identity["environment_sha256"],
            "git_commit": run_identity["git_commit"],
            "status": "completed",
            "calibration_conclusion": conclusion,
            "ladder_complete": ladder_complete,
            "completed_stage_count": simulation.completed_stage_count,
            "max_jobs_limit_reached": simulation.limit_reached,
            "selected_job_count": len(selected_jobs),
            "attempted_job_count": len(checkpoints),
            "unattempted_job_count": EXPECTED_LOGICAL_JOB_COUNT - len(checkpoints),
            "accepted_job_count": accepted_count,
            "rejected_attempted_job_count": len(checkpoints) - accepted_count,
            "surviving_profile_order_indices": list(
                simulation.surviving_profile_indices
            ),
            "locked_development_profile_ids": list(locked_ids),
            "formal_pool_created": False,
            "checkpoints": checkpoint_records,
            "logical_job_ledger": ledger,
            "completed_at_utc": _require_utc(
                payload["completed_at_utc"], field_name="summary completed_at_utc"
            ),
        }
    elif status == "aborted_on_infrastructure_failure":
        _require_exact_keys(
            payload,
            {
                "artifact_type",
                "artifact_schema_version",
                "summary_schema",
                "development_only",
                "eligible_for_formal_pool",
                "calibration_run_id",
                "run_identity_sha256",
                "source_manifest_sha256",
                "status",
                "selected_job_count",
                "attempted_job_count",
                "completed_job_count",
                "failed_job_id",
                "checkpoints",
                "logical_job_ledger",
                "completed_at_utc",
                "summary_sha256",
            },
            field_name="aborted summary",
        )
        if not simulation.infrastructure_aborted or not checkpoints:
            _fail("aborted summary lacks a terminal infrastructure checkpoint")
        ledger = _canonical_ledger(
            manifest=manifest,
            selected_job_ids=selected_ids,
            checkpoints=checkpoints,
            termination_reason="infrastructure_abort",
        )
        expected_entry = {
            "summary_schema": SUMMARY_SCHEMA,
            "development_only": True,
            "eligible_for_formal_pool": False,
            "calibration_run_id": "development-calibration-run-" + run_sha,
            "run_identity_sha256": run_sha,
            "source_manifest_sha256": manifest.manifest_sha256,
            "status": "aborted_on_infrastructure_failure",
            "selected_job_count": len(selected_jobs),
            "attempted_job_count": len(checkpoints),
            "completed_job_count": sum(
                item.status == "completed" for item in checkpoints
            ),
            "failed_job_id": checkpoints[-1].job["job_id"],
            "checkpoints": checkpoint_records,
            "logical_job_ledger": ledger,
            "completed_at_utc": _require_utc(
                payload["completed_at_utc"], field_name="summary completed_at_utc"
            ),
        }
    else:
        _fail(f"summary status {status!r} is invalid")
    if canonical_json(entry) != canonical_json(expected_entry):
        _fail("summary fields or canonical 144-row ledger do not recompute")
    return payload, simulation, conclusion, locked_ids


def verify_development_calibration_results(
    request: CalibrationVerificationRequest,
) -> CalibrationVerificationReport:
    """Read and independently verify one immutable calibration result root."""

    if not isinstance(request, CalibrationVerificationRequest):
        raise TypeError("request must be CalibrationVerificationRequest")
    manifest = _verify_manifest(
        request.manifest_path, request.expected_manifest_sha256
    )
    results_root = _resolved_path_without_symlinks(
        request.results_root, field_name="results_root", kind="directory"
    )
    try:
        results_root.relative_to(manifest.root)
    except ValueError:
        pass
    else:
        _fail("results_root is inside the development input artifact")
    if "formal" in {part.lower() for part in results_root.parts}:
        _fail("development calibration results may not be stored under formal")
    entries = {path.name for path in results_root.iterdir()}
    summaries = entries & {"batch_summary.json", "batch_aborted.json"}
    if len(summaries) != 1 or entries != {
        "run_identity.json",
        "checkpoints",
        *summaries,
    }:
        _fail("results root contains missing, extra, or ambiguous artifacts")
    run_identity_path = _resolved_path_without_symlinks(
        results_root / "run_identity.json",
        field_name="run_identity.json",
        kind="file",
    )
    checkpoints_root = _resolved_path_without_symlinks(
        results_root / "checkpoints",
        field_name="checkpoints root",
        kind="directory",
    )
    summary_name = next(iter(summaries))
    summary_path = _resolved_path_without_symlinks(
        results_root / summary_name, field_name=summary_name, kind="file"
    )
    for path in (run_identity_path, checkpoints_root, summary_path):
        _require_inside(path, results_root, field_name=path.name)
    (
        run_identity,
        run_sha,
        selected_jobs,
        max_jobs,
        commit,
        implementation_sha,
    ) = _verify_run_identity(run_identity_path, manifest, request)

    summary_preview = _read_canonical_object(
        summary_path, field_name="calibration batch summary preview"
    )
    raw_records = _require_list(
        summary_preview.get("checkpoints"), field_name="summary checkpoints"
    )
    jobs_by_id = {str(job["job_id"]): job for job in manifest.jobs}
    checkpoints: list[_VerifiedCheckpoint] = []
    observed_paths: set[Path] = set()
    observed_jobs: set[str] = set()
    for attempt_index, raw_record in enumerate(raw_records):
        record = _require_mapping(
            raw_record, field_name=f"summary checkpoint record {attempt_index}"
        )
        job_id = record.get("job_id")
        if not isinstance(job_id, str) or job_id not in jobs_by_id:
            _fail("summary checkpoint references an unknown job")
        if job_id in observed_jobs:
            _fail("one logical job was attempted more than once")
        observed_jobs.add(job_id)
        job = jobs_by_id[job_id]
        expected_name = (
            f"job-{int(job['job_order_index']):03d}-"
            f"{hashlib.sha256(job_id.encode('utf-8')).hexdigest()}.json"
        )
        claimed_path = record.get("path")
        if not isinstance(claimed_path, str):
            _fail("checkpoint record path must be a string")
        checkpoint_path = _resolved_path_without_symlinks(
            claimed_path,
            field_name=f"checkpoint path {attempt_index}",
            kind="file",
        )
        if (
            str(checkpoint_path) != claimed_path
            or checkpoint_path.parent != checkpoints_root
            or checkpoint_path.name != expected_name
        ):
            _fail("checkpoint path escapes its root or has a noncanonical name")
        if checkpoint_path in observed_paths:
            _fail("two checkpoint records reference the same file")
        observed_paths.add(checkpoint_path)
        checkpoints.append(
            _verify_checkpoint(
                checkpoint_path,
                expected_job=job,
                manifest=manifest,
                run_identity=run_identity,
                run_sha=run_sha,
                execution_attempt_index=attempt_index,
            )
        )
    directory_files = {
        _resolved_path_without_symlinks(
            path, field_name=f"checkpoint directory entry {path.name}", kind="file"
        )
        for path in checkpoints_root.iterdir()
    }
    if directory_files != observed_paths:
        _fail("checkpoint directory contains unreferenced or missing files")
    summary, simulation, conclusion, locked_ids = _verify_summary(
        summary_path,
        manifest=manifest,
        run_identity=run_identity,
        run_sha=run_sha,
        selected_jobs=selected_jobs,
        max_jobs=max_jobs,
        checkpoints=checkpoints,
    )
    accepted_count = sum(item.accepted for item in checkpoints)
    infrastructure_count = sum(
        item.status == "infrastructure_failure" for item in checkpoints
    )
    return CalibrationVerificationReport(
        status=str(summary["status"]),
        summary_path=summary_path,
        calibration_run_id=request.expected_run_id,
        source_manifest_sha256=manifest.manifest_sha256,
        git_commit=commit,
        implementation_sha256=implementation_sha,
        selected_job_count=len(selected_jobs),
        attempted_job_count=len(checkpoints),
        accepted_job_count=accepted_count,
        rejected_attempted_job_count=(
            len(checkpoints) - accepted_count - infrastructure_count
        ),
        infrastructure_failure_count=infrastructure_count,
        unattempted_job_count=EXPECTED_LOGICAL_JOB_COUNT - len(checkpoints),
        calibration_conclusion=conclusion,
        locked_development_profile_ids=locked_ids,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.benchmark.branch_cut.calibration_verifier",
        description=(
            "Read-only independent verification of development calibration "
            "artifacts; this command never imports or runs Gurobi."
        ),
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--expected-run-id", required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--expected-git-commit", required=True)
    parser.add_argument("--expected-implementation-sha256", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        request = CalibrationVerificationRequest(
            manifest_path=Path(args.manifest),
            results_root=Path(args.results_root),
            expected_run_id=args.expected_run_id,
            expected_manifest_sha256=args.expected_manifest_sha256,
            expected_git_commit=args.expected_git_commit,
            expected_implementation_sha256=args.expected_implementation_sha256,
        )
        report = verify_development_calibration_results(request)
    except (CalibrationVerificationError, OSError, ValueError) as error:
        print(f"calibration verification failed: {error}", file=sys.stderr)
        return 2
    print(canonical_json(report.to_payload()))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
