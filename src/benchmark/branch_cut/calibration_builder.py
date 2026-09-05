"""Build immutable development-only inputs for qualification-profile calibration.

This module creates real canonical instances and a deterministic logical job
plan.  It deliberately has no solver entry point: the output is an input-only
development artifact and cannot be used as a formal candidate pool or frozen
benchmark manifest.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import math
from pathlib import Path
from typing import Any, Mapping

from src.config import DataConfig, ModelConfig, to_dict
from src.data_processor.generator import DataGenerator

from .artifacts import write_canonical_json_exclusive
from .case_builder import (
    CANONICAL_INSTANCE_FILES,
    DEVELOPMENT_DEMAND_BLOCK,
    LOCKED_DEMAND_BLOCKS,
    LOCKED_LAYOUT_SEED_ORDER,
    LOCKED_QUALIFICATION_PROFILES,
    LOCKED_SCALE_SPECS,
    GeneratorFactory,
    LockedCasePlan,
    QualificationSolveProfile,
    VerifiedCanonicalInstance,
    qualification_case_sha256,
    verify_generated_case,
)
from .hashing import canonical_json, canonical_sha256


DEVELOPMENT_CALIBRATION_MANIFEST_SCHEMA = (
    "mlnd-development-profile-calibration-manifest-v1"
)
DEVELOPMENT_CALIBRATION_JOB_SCHEMA = "mlnd-development-profile-calibration-job-v1"
DEVELOPMENT_LAYOUT_SEED_ORDER = (1729, 2718)
DEVELOPMENT_DEMAND_SCALE_ORDER = (1.0, 2.0, 3.0, 4.0)
DEVELOPMENT_FEEDER_FACTOR_ORDER = (0.30, 0.35)
DEVELOPMENT_BASE_LOCKER_FIXED_COST_PER_DEMAND_UNIT_ORDER = (20.0, 24.0, 28.0)
DEVELOPMENT_DISPLAY_ORIGIN_CHOICES = 8
DEVELOPMENT_GENERAL_CORRIDOR_CHOICES = 10
DEVELOPMENT_EXPECTED_SCREENING_IS_EXACT = False


@dataclass(frozen=True, slots=True)
class DevelopmentCalibrationProfile:
    """One preregistered screened economic scenario, never a formal profile."""

    demand_scale: float
    feeder_rider_cost_factor: float
    base_locker_fixed_cost_per_demand_unit: float

    def __post_init__(self) -> None:
        for field_name in (
            "demand_scale",
            "feeder_rider_cost_factor",
            "base_locker_fixed_cost_per_demand_unit",
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{field_name} must be numeric")
            normalized = float(value)
            if not math.isfinite(normalized) or normalized <= 0.0:
                raise ValueError(f"{field_name} must be positive and finite")
            object.__setattr__(self, field_name, normalized)
        if not self.demand_scale.is_integer():
            raise ValueError("demand_scale must be integral in the registered grid")
        if not self.base_locker_fixed_cost_per_demand_unit.is_integer():
            raise ValueError(
                "base_locker_fixed_cost_per_demand_unit must be integral"
            )
        if not (100 * self.feeder_rider_cost_factor).is_integer():
            raise ValueError(
                "feeder_rider_cost_factor must use two-decimal grid precision"
            )
        # Reuse the qualification profile's complete model-axis validation.
        self.solve_profile()

    @property
    def locker_fixed_cost_per_hour(self) -> float:
        return self.demand_scale * self.base_locker_fixed_cost_per_demand_unit

    @property
    def profile_id(self) -> str:
        demand_code = int(self.demand_scale)
        feeder_code = int(round(100 * self.feeder_rider_cost_factor))
        base_cost_code = int(self.base_locker_fixed_cost_per_demand_unit)
        fixed_cost_code = int(self.locker_fixed_cost_per_hour)
        return (
            f"devcal-demand-{demand_code}x-feeder-{feeder_code:03d}"
            f"-basefc-{base_cost_code:03d}-fc-{fixed_cost_code:03d}"
            "-display-008-general-010-exact-false"
        )

    @property
    def interpretation(self) -> str:
        return (
            "development-only screened calibration; H fixed cost equals demand "
            "scale times the preregistered base cost-per-demand-unit ratio, "
            "holding unit-demand economics fixed while probing scale and congestion"
        )

    def solve_profile(self) -> QualificationSolveProfile:
        return QualificationSolveProfile(
            profile_id=self.profile_id,
            demand_scale=self.demand_scale,
            feeder_rider_cost_factor=self.feeder_rider_cost_factor,
            locker_fixed_cost_per_hour=self.locker_fixed_cost_per_hour,
            display_origin_choices=DEVELOPMENT_DISPLAY_ORIGIN_CHOICES,
            general_corridor_choices=DEVELOPMENT_GENERAL_CORRIDOR_CHOICES,
            expected_screening_is_exact=DEVELOPMENT_EXPECTED_SCREENING_IS_EXACT,
            interpretation=self.interpretation,
            adds_forced_opening_constraint=False,
        )

    def effective_model_config(self) -> ModelConfig:
        return self.solve_profile().effective_model_config()

    def effective_model_config_sha256(self) -> str:
        return self.solve_profile().effective_model_config_sha256()

    def to_payload(self) -> dict[str, Any]:
        solve_profile = self.solve_profile()
        return {
            "profile_id": self.profile_id,
            "development_only": True,
            "demand_scale": self.demand_scale,
            "feeder_rider_cost_factor": self.feeder_rider_cost_factor,
            "base_locker_fixed_cost_per_demand_unit": (
                self.base_locker_fixed_cost_per_demand_unit
            ),
            "locker_fixed_cost_per_hour": self.locker_fixed_cost_per_hour,
            "display_origin_choices": solve_profile.display_origin_choices,
            "general_corridor_choices": solve_profile.general_corridor_choices,
            "expected_screening_is_exact": (
                solve_profile.expected_screening_is_exact
            ),
            "adds_forced_opening_constraint": False,
            "interpretation": self.interpretation,
            "qualification_solve_profile": solve_profile.to_payload(),
            "effective_model_config": to_dict(self.effective_model_config()),
            "effective_model_config_sha256": (
                self.effective_model_config_sha256()
            ),
        }


DEVELOPMENT_CALIBRATION_PROFILES = tuple(
    DevelopmentCalibrationProfile(
        demand_scale=demand_scale,
        feeder_rider_cost_factor=feeder_factor,
        base_locker_fixed_cost_per_demand_unit=base_fixed_cost,
    )
    for demand_scale in DEVELOPMENT_DEMAND_SCALE_ORDER
    for feeder_factor in DEVELOPMENT_FEEDER_FACTOR_ORDER
    for base_fixed_cost in (
        DEVELOPMENT_BASE_LOCKER_FIXED_COST_PER_DEMAND_UNIT_ORDER
    )
)


@dataclass(frozen=True, slots=True)
class DevelopmentCalibrationBuildConfig:
    """Filesystem inputs for the fixed development-only calibration design."""

    output_root: Path
    tsl_raw_path: Path
    candidate_hour_start: int = 8
    candidate_hour_end: int = 24
    site_grid_km: float = 0.25

    def validate(self) -> None:
        _validate_registered_development_design()
        if not 0 <= self.candidate_hour_start < self.candidate_hour_end <= 24:
            raise ValueError("candidate hours must be a non-empty interval in [0, 24]")
        if not math.isfinite(self.site_grid_km) or self.site_grid_km <= 0.0:
            raise ValueError("site_grid_km must be positive and finite")
        if not self.tsl_raw_path.expanduser().is_file():
            raise FileNotFoundError(
                f"TSL development raw source does not exist: {self.tsl_raw_path}"
            )

    @property
    def resolved_raw_path(self) -> Path:
        return self.tsl_raw_path.expanduser().resolve()


@dataclass(frozen=True, slots=True)
class DevelopmentCalibrationBuildResult:
    manifest_path: Path
    manifest_sha256: str
    instance_directories: Mapping[str, Path]
    physical_instance_count: int
    logical_job_count: int


def _validate_registered_development_design() -> None:
    locked_block_ids = {block.block_id for block in LOCKED_DEMAND_BLOCKS}
    if DEVELOPMENT_DEMAND_BLOCK.locked:
        raise AssertionError("development demand block may not be locked")
    if DEVELOPMENT_DEMAND_BLOCK.block_id in locked_block_ids:
        raise AssertionError("development demand block overlaps a formal block")
    if DEVELOPMENT_DEMAND_BLOCK.source != "tsl":
        raise AssertionError("development calibration must use the held-out TSL block")
    if any(
        block.source == DEVELOPMENT_DEMAND_BLOCK.source
        and set(block.dates) & set(DEVELOPMENT_DEMAND_BLOCK.dates)
        for block in LOCKED_DEMAND_BLOCKS
    ):
        raise AssertionError("development demand dates overlap a formal demand block")
    if set(DEVELOPMENT_LAYOUT_SEED_ORDER) & set(LOCKED_LAYOUT_SEED_ORDER):
        raise AssertionError("development layout seeds overlap formal layout seeds")
    if (
        len(DEVELOPMENT_LAYOUT_SEED_ORDER)
        != len(set(DEVELOPMENT_LAYOUT_SEED_ORDER))
        or any(
            isinstance(seed, bool) or not isinstance(seed, int) or seed < 0
            for seed in DEVELOPMENT_LAYOUT_SEED_ORDER
        )
    ):
        raise AssertionError("development layout seeds must be distinct integers")
    if tuple(spec.candidate_h for spec in LOCKED_SCALE_SPECS) != (23, 51, 90):
        raise AssertionError("development calibration scale H counts must be 23/51/90")
    expected_profile_count = (
        len(DEVELOPMENT_DEMAND_SCALE_ORDER)
        * len(DEVELOPMENT_FEEDER_FACTOR_ORDER)
        * len(DEVELOPMENT_BASE_LOCKER_FIXED_COST_PER_DEMAND_UNIT_ORDER)
    )
    expected_profile_axes = {
        (demand_scale, feeder_factor, base_fixed_cost)
        for demand_scale in DEVELOPMENT_DEMAND_SCALE_ORDER
        for feeder_factor in DEVELOPMENT_FEEDER_FACTOR_ORDER
        for base_fixed_cost in (
            DEVELOPMENT_BASE_LOCKER_FIXED_COST_PER_DEMAND_UNIT_ORDER
        )
    }
    observed_profile_axes = {
        (
            profile.demand_scale,
            profile.feeder_rider_cost_factor,
            profile.base_locker_fixed_cost_per_demand_unit,
        )
        for profile in DEVELOPMENT_CALIBRATION_PROFILES
    }
    if (
        len(DEVELOPMENT_CALIBRATION_PROFILES) != expected_profile_count
        or len(
            {profile.profile_id for profile in DEVELOPMENT_CALIBRATION_PROFILES}
        )
        != expected_profile_count
        or observed_profile_axes != expected_profile_axes
    ):
        raise AssertionError("development calibration profile grid is not unique")
    locked_profile_ids = {
        profile.profile_id for profile in LOCKED_QUALIFICATION_PROFILES
    }
    if any(
        profile.profile_id in locked_profile_ids
        for profile in DEVELOPMENT_CALIBRATION_PROFILES
    ):
        raise AssertionError("development calibration profiles overlap formal profiles")
    for profile in DEVELOPMENT_CALIBRATION_PROFILES:
        solve_profile = profile.solve_profile()
        if (
            solve_profile.display_origin_choices
            != DEVELOPMENT_DISPLAY_ORIGIN_CHOICES
            or solve_profile.general_corridor_choices
            != DEVELOPMENT_GENERAL_CORRIDOR_CHOICES
            or solve_profile.expected_screening_is_exact
            is not DEVELOPMENT_EXPECTED_SCREENING_IS_EXACT
        ):
            raise AssertionError("development screening contract drifted")


def development_instance_plans() -> tuple[LockedCasePlan, ...]:
    """Return scale-major, development-seed-minor physical generation plans."""

    _validate_registered_development_design()
    plans = tuple(
        LockedCasePlan(
            case_id=(
                f"{DEVELOPMENT_DEMAND_BLOCK.block_id}__{scale.scale}"
                f"__devseed{seed:010d}"
            ),
            block=DEVELOPMENT_DEMAND_BLOCK,
            scale=scale,
            layout_seed=seed,
        )
        for scale in LOCKED_SCALE_SPECS
        for seed in DEVELOPMENT_LAYOUT_SEED_ORDER
    )
    expected_count = len(LOCKED_SCALE_SPECS) * len(DEVELOPMENT_LAYOUT_SEED_ORDER)
    if len(plans) != expected_count or len({plan.case_id for plan in plans}) != len(
        plans
    ):
        raise AssertionError("development physical-instance plan is not unique")
    return plans


def development_calibration_profiles() -> tuple[DevelopmentCalibrationProfile, ...]:
    _validate_registered_development_design()
    return DEVELOPMENT_CALIBRATION_PROFILES


def _require_pristine_output_root(output_root: Path) -> None:
    if output_root.exists():
        if not output_root.is_dir():
            raise FileExistsError(f"output root is not a directory: {output_root}")
        existing = sorted(path.name for path in output_root.iterdir())
        if existing:
            raise FileExistsError(
                "refusing to reuse a non-empty development calibration root; "
                f"existing or partial artifacts: {existing}"
            )
        return
    output_root.mkdir(parents=True, exist_ok=False)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _data_config_payload(config: DataConfig) -> dict[str, Any]:
    payload = asdict(config)
    payload["dates"] = list(config.dates)
    return payload


def _verified_instance_payload(
    *,
    plan: LockedCasePlan,
    data_config: DataConfig,
    verified: VerifiedCanonicalInstance,
    case_dir: Path,
    raw_sha256: str,
    raw_size_bytes: int,
) -> dict[str, Any]:
    observed_raw_sha256 = verified.metadata.get("raw_file_sha256")
    if observed_raw_sha256 != raw_sha256:
        raise ValueError(
            f"{case_dir}: generated raw_file_sha256 does not match the claimed input"
        )
    return {
        "physical_instance_id": plan.case_id,
        "development_only": True,
        "scale": plan.scale.scale,
        "development_seed_stage": DEVELOPMENT_LAYOUT_SEED_ORDER.index(
            plan.layout_seed
        ),
        "layout_seed": plan.layout_seed,
        "instance_ref": f"instances/{plan.case_id}",
        "instance_sha256": verified.instance_sha256,
        "raw_file_sha256": raw_sha256,
        "raw_file_size_bytes": raw_size_bytes,
        "canonical_file_sha256": dict(verified.canonical_file_sha256),
        "data_config": _data_config_payload(data_config),
        "projection_center_lon": verified.projection_center[0],
        "projection_center_lat": verified.projection_center[1],
    }


def _logical_job_payloads(
    physical_instances: tuple[dict[str, Any], ...],
) -> list[dict[str, Any]]:
    physical_by_id = {
        item["physical_instance_id"]: item for item in physical_instances
    }
    jobs: list[dict[str, Any]] = []
    for scale_index, scale in enumerate(LOCKED_SCALE_SPECS):
        for seed_stage, seed in enumerate(DEVELOPMENT_LAYOUT_SEED_ORDER):
            physical_id = (
                f"{DEVELOPMENT_DEMAND_BLOCK.block_id}__{scale.scale}"
                f"__devseed{seed:010d}"
            )
            physical = physical_by_id[physical_id]
            for profile_index, profile in enumerate(
                DEVELOPMENT_CALIBRATION_PROFILES
            ):
                solve_profile = profile.solve_profile()
                case_sha256 = qualification_case_sha256(
                    physical["instance_sha256"],
                    solve_profile,
                )
                job_identity = {
                    "hash_schema": DEVELOPMENT_CALIBRATION_JOB_SCHEMA,
                    "qualification_case_sha256": case_sha256,
                    "profile_id": profile.profile_id,
                    "scale_order_index": scale_index,
                    "development_seed_stage": seed_stage,
                    "profile_order_index": profile_index,
                }
                jobs.append(
                    {
                        "job_order_index": len(jobs),
                        "job_id": (
                            "development-calibration-job-"
                            + canonical_sha256(job_identity)
                        ),
                        "profile_id": profile.profile_id,
                        "scale": scale.scale,
                        "scale_order_index": scale_index,
                        "layout_seed": seed,
                        "development_seed_stage": seed_stage,
                        "profile_order_index": profile_index,
                        "physical_instance_id": physical_id,
                        "instance_ref": physical["instance_ref"],
                        "instance_sha256": physical["instance_sha256"],
                        "effective_model_config_sha256": (
                            profile.effective_model_config_sha256()
                        ),
                        "qualification_case_sha256": case_sha256,
                    }
                )
    return jobs


def build_development_calibration_manifest(
    config: DevelopmentCalibrationBuildConfig,
    *,
    generator_factory: GeneratorFactory = DataGenerator,
) -> DevelopmentCalibrationBuildResult:
    """Generate held-out development instances and an input-only job manifest."""

    config.validate()
    output_root = config.output_root.expanduser().resolve()
    _require_pristine_output_root(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    raw_path = config.resolved_raw_path
    raw_sha256_before = _sha256_file(raw_path)
    raw_size_bytes_before = raw_path.stat().st_size
    instances_root = output_root / "instances"
    instances_root.mkdir(exist_ok=False)
    physical_payloads: list[dict[str, Any]] = []
    instance_directories: dict[str, Path] = {}
    shared_center: tuple[float, float] | None = None

    for plan in development_instance_plans():
        if plan.block is not DEVELOPMENT_DEMAND_BLOCK or plan.block.locked:
            raise AssertionError("development builder received a formal demand block")
        case_dir = instances_root / plan.case_id
        case_dir.mkdir(exist_ok=False)
        data_config = plan.data_config(
            raw_path=raw_path,
            output_dir=case_dir,
            candidate_hour_start=config.candidate_hour_start,
            candidate_hour_end=config.candidate_hour_end,
            site_grid_km=config.site_grid_km,
        )
        data_config.validate()
        generated_path = Path(generator_factory().generate(data_config)).resolve()
        if generated_path != case_dir:
            raise ValueError(
                f"generator returned {generated_path}, expected claimed directory "
                f"{case_dir}"
            )
        verified = verify_generated_case(case_dir, data_config)
        if shared_center is None:
            shared_center = verified.projection_center
        elif verified.projection_center != shared_center:
            raise ValueError(
                "development demand block does not share one projection center"
            )
        physical_payloads.append(
            _verified_instance_payload(
                plan=plan,
                data_config=data_config,
                verified=verified,
                case_dir=case_dir,
                raw_sha256=raw_sha256_before,
                raw_size_bytes=raw_size_bytes_before,
            )
        )
        instance_directories[plan.case_id] = case_dir

    raw_sha256_after = _sha256_file(raw_path)
    raw_size_bytes_after = raw_path.stat().st_size
    if (
        raw_sha256_after != raw_sha256_before
        or raw_size_bytes_after != raw_size_bytes_before
    ):
        raise ValueError("development raw source changed during instance generation")

    physical_instances = tuple(physical_payloads)
    logical_jobs = _logical_job_payloads(physical_instances)
    profiles = [profile.to_payload() for profile in DEVELOPMENT_CALIBRATION_PROFILES]
    manifest_payload: dict[str, Any] = {
        "artifact_type": "mlnd_branch_cut_development_calibration_manifest",
        "artifact_schema_version": 1,
        "manifest_schema": DEVELOPMENT_CALIBRATION_MANIFEST_SCHEMA,
        "development_only": True,
        "eligible_for_formal_pool": False,
        "input_artifact_only": True,
        "contains_solver_results": False,
        "formal_contract": {
            "locked_demand_blocks_used": False,
            "locked_layout_seeds_used": False,
            "locked_qualification_profiles_modified": False,
            "candidate_pool_created": False,
            "excluded_locked_demand_block_ids": [
                block.block_id for block in LOCKED_DEMAND_BLOCKS
            ],
            "excluded_locked_layout_seeds": list(LOCKED_LAYOUT_SEED_ORDER),
        },
        "development_demand_block": DEVELOPMENT_DEMAND_BLOCK.to_payload(),
        "development_layout_seed_order": list(DEVELOPMENT_LAYOUT_SEED_ORDER),
        "canonical_instance_files": list(CANONICAL_INSTANCE_FILES),
        "scale_order": [asdict(scale) for scale in LOCKED_SCALE_SPECS],
        "raw_input": {
            "source": DEVELOPMENT_DEMAND_BLOCK.source,
            "path": str(raw_path),
            "sha256": raw_sha256_before,
            "size_bytes": raw_size_bytes_before,
        },
        "build_config": {
            "output_root": str(output_root),
            "tsl_raw_path": str(raw_path),
            "candidate_hour_start": config.candidate_hour_start,
            "candidate_hour_end": config.candidate_hour_end,
            "site_grid_km": config.site_grid_km,
        },
        "screening_contract": {
            "display_origin_choices": DEVELOPMENT_DISPLAY_ORIGIN_CHOICES,
            "general_corridor_choices": DEVELOPMENT_GENERAL_CORRIDOR_CHOICES,
            "expected_screening_is_exact": (
                DEVELOPMENT_EXPECTED_SCREENING_IS_EXACT
            ),
        },
        "calibration_grid": {
            "profile_order": (
                "demand_scale_then_feeder_factor_then_base_fixed_cost_ratio"
            ),
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
            "profile_count": len(profiles),
            "profiles": profiles,
        },
        "physical_instance_count": len(physical_instances),
        "physical_instances": list(physical_instances),
        "logical_job_count": len(logical_jobs),
        "logical_job_order_contract": (
            "scale_order_then_development_seed_stage_then_profile_order"
        ),
        "logical_job_order": logical_jobs,
    }
    manifest_sha256 = canonical_sha256(manifest_payload)
    manifest_path = write_canonical_json_exclusive(
        output_root / "development_calibration_manifest.json",
        {
            **manifest_payload,
            "development_calibration_manifest_sha256": manifest_sha256,
        },
    )
    return DevelopmentCalibrationBuildResult(
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha256,
        instance_directories=instance_directories,
        physical_instance_count=len(physical_instances),
        logical_job_count=len(logical_jobs),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.benchmark.branch_cut.calibration_builder",
        description=(
            "Build the immutable development-only profile-calibration inputs; "
            "this command does not run a solver."
        ),
    )
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--tsl-raw", required=True)
    parser.add_argument(
        "--candidate-hours",
        nargs=2,
        type=int,
        default=(8, 24),
        metavar=("START", "END"),
    )
    parser.add_argument("--site-grid-km", type=float, default=0.25)
    return parser


def main(
    argv: list[str] | None = None,
    *,
    generator_factory: GeneratorFactory = DataGenerator,
) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = DevelopmentCalibrationBuildConfig(
        output_root=Path(args.output_root),
        tsl_raw_path=Path(args.tsl_raw),
        candidate_hour_start=args.candidate_hours[0],
        candidate_hour_end=args.candidate_hours[1],
        site_grid_km=args.site_grid_km,
    )
    try:
        result = build_development_calibration_manifest(
            config,
            generator_factory=generator_factory,
        )
    except Exception as error:
        parser.exit(2, f"development calibration build failed: {error}\n")
    print(
        canonical_json(
            {
                "manifest_path": str(result.manifest_path),
                "manifest_sha256": result.manifest_sha256,
                "physical_instance_count": result.physical_instance_count,
                "logical_job_count": result.logical_job_count,
            }
        )
    )
    return 0


__all__ = [
    "DEVELOPMENT_BASE_LOCKER_FIXED_COST_PER_DEMAND_UNIT_ORDER",
    "DEVELOPMENT_CALIBRATION_PROFILES",
    "DEVELOPMENT_DEMAND_SCALE_ORDER",
    "DEVELOPMENT_DISPLAY_ORIGIN_CHOICES",
    "DEVELOPMENT_EXPECTED_SCREENING_IS_EXACT",
    "DEVELOPMENT_FEEDER_FACTOR_ORDER",
    "DEVELOPMENT_GENERAL_CORRIDOR_CHOICES",
    "DEVELOPMENT_LAYOUT_SEED_ORDER",
    "DevelopmentCalibrationBuildConfig",
    "DevelopmentCalibrationBuildResult",
    "DevelopmentCalibrationProfile",
    "build_parser",
    "build_development_calibration_manifest",
    "development_calibration_profiles",
    "development_instance_plans",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
