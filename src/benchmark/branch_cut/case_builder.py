"""Build the overcomplete real-data pool used to freeze 36 formal cases.

This module is a narrow orchestration layer around :class:`DataGenerator`.
It does not reimplement preprocessing: each case is a genuine spatial crop of
the source rows under the generator's existing rule that both order endpoints
must lie inside the centered square.  The formal pool is fixed before any
native-Gurobi versus branch-and-cut comparison is run.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, replace
import json
import math
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol

from src.config import DataConfig, ModelConfig, to_dict
from src.data_processor.generator import DataGenerator

from .artifacts import write_canonical_json_exclusive
from .hashing import canonical_sha256
from .manifest import OVERCOMPLETE_SELECTION_RULE
from .pool import CandidateCase, CandidatePool, build_candidate_pool


CANONICAL_INSTANCE_FILES = (
    "metadata.json",
    "sites.csv",
    "od_nodes.csv",
    "od_demand.csv",
    "periods.csv",
)

QUALIFICATION_CASE_HASH_SCHEMA = "mlnd-qualification-case-v1"
QUALIFICATION_MODEL_IDENTITY_SCHEMA = "mlnd-qualification-model-identity-v1"


@dataclass(frozen=True, slots=True)
class ScaleCaseSpec:
    """One map crop and its separately reclustered candidate-site targets."""

    scale: str
    map_half_size_km: float
    candidate_h: int
    od_grid_km: float
    candidate_l: int
    candidate_e: int


LOCKED_SCALE_SPECS = (
    ScaleCaseSpec("small", 4.0, 23, 1.0, 2, 3),
    ScaleCaseSpec("medium", 6.0, 51, 0.75, 3, 5),
    ScaleCaseSpec("large", 8.0, 90, 0.5, 6, 9),
)

# The tuple order is part of the selection protocol.  It is intentionally not
# inferred from a set or chosen after observing qualification/solver results.
LOCKED_LAYOUT_SEED_ORDER = (11, 29, 47, 83)


@dataclass(frozen=True, slots=True)
class QualificationSolveProfile:
    """Ex-ante demand/economic scenario used only during qualification.

    These are ordinary model parameters.  In particular, no profile adds a
    constraint that forces any H site (or any other facility) to open.
    """

    profile_id: str
    demand_scale: float
    feeder_rider_cost_factor: float
    locker_fixed_cost_per_hour: float
    display_origin_choices: int | None
    general_corridor_choices: int | None
    expected_screening_is_exact: bool
    interpretation: str
    adds_forced_opening_constraint: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.profile_id, str) or not self.profile_id:
            raise ValueError("profile_id must be non-empty")
        for name in (
            "demand_scale",
            "feeder_rider_cost_factor",
            "locker_fixed_cost_per_hour",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be numeric")
            normalized = float(value)
            if not math.isfinite(normalized):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, normalized)
        if self.demand_scale <= 0.0:
            raise ValueError("demand_scale must be positive and finite")
        if self.feeder_rider_cost_factor <= 0.0:
            raise ValueError("feeder_rider_cost_factor must be positive and finite")
        if self.locker_fixed_cost_per_hour < 0.0:
            raise ValueError(
                "locker_fixed_cost_per_hour must be nonnegative and finite"
            )
        for name in ("display_origin_choices", "general_corridor_choices"):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 1
            ):
                raise ValueError(f"{name} must be a positive integer or None")
        if type(self.expected_screening_is_exact) is not bool:
            raise ValueError("expected_screening_is_exact must be boolean")
        if not isinstance(self.interpretation, str) or not self.interpretation:
            raise ValueError("interpretation must be non-empty")
        if type(self.adds_forced_opening_constraint) is not bool:
            raise ValueError("adds_forced_opening_constraint must be boolean")
        if self.adds_forced_opening_constraint:
            raise ValueError("qualification profiles may not force facilities open")

    def to_payload(self) -> dict[str, Any]:
        return asdict(self)

    def effective_model_config(self) -> ModelConfig:
        """Return the complete model configuration used by this profile."""

        config = replace(
            ModelConfig(),
            feeder_rider_cost_factor=self.feeder_rider_cost_factor,
            locker_fixed_cost_per_hour=self.locker_fixed_cost_per_hour,
            display_origin_choices=self.display_origin_choices,
            general_corridor_choices=self.general_corridor_choices,
        )
        config.validate()
        return config

    def effective_model_config_sha256(self) -> str:
        """Hash the full effective ``ModelConfig`` without demand scaling."""

        return canonical_sha256(to_dict(self.effective_model_config()))


def qualification_case_sha256(
    instance_sha256: str,
    profile: QualificationSolveProfile,
) -> str:
    """Content-address one physical instance plus its qualification profile."""

    if (
        not isinstance(instance_sha256, str)
        or len(instance_sha256) != 64
        or any(character not in "0123456789abcdef" for character in instance_sha256)
    ):
        raise ValueError(
            "instance_sha256 must be a lowercase 64-character SHA256 digest"
        )
    if not isinstance(profile, QualificationSolveProfile):
        raise TypeError("profile must be a QualificationSolveProfile")
    return canonical_sha256(
        {
            "hash_schema": QUALIFICATION_CASE_HASH_SCHEMA,
            "physical_instance_sha256": instance_sha256,
            "qualification_solve_profile": profile.to_payload(),
            "effective_model_config_sha256": profile.effective_model_config_sha256(),
        }
    )


# A small, transparent 2x2 scenario grid.  Its profile-major order is locked
# before any algorithm comparison.  Development evidence showed that demand
# multipliers near one do not activate enough H sites at the large scale, so all
# three spatial scales use the predeclared 3x/4x demand levels.  The second axis
# is a moderately versus a more efficient contracted-feeder scenario.  These
# values apply only to the not-yet-frozen qualification pool; no A--E result was
# used to choose them.
LOCKED_QUALIFICATION_PROFILES = (
    QualificationSolveProfile(
        profile_id="demand-3x-feeder-035-fc-001-display-all-general-all-exact-true",
        demand_scale=3.00,
        feeder_rider_cost_factor=0.35,
        locker_fixed_cost_per_hour=1.0,
        display_origin_choices=None,
        general_corridor_choices=None,
        expected_screening_is_exact=True,
        interpretation=(
            "threefold order-rate scenario with a moderately efficient contracted feeder"
        ),
    ),
    QualificationSolveProfile(
        profile_id="demand-4x-feeder-035-fc-001-display-all-general-all-exact-true",
        demand_scale=4.00,
        feeder_rider_cost_factor=0.35,
        locker_fixed_cost_per_hour=1.0,
        display_origin_choices=None,
        general_corridor_choices=None,
        expected_screening_is_exact=True,
        interpretation=(
            "fourfold order-rate scenario with a moderately efficient contracted feeder"
        ),
    ),
    QualificationSolveProfile(
        profile_id="demand-3x-feeder-030-fc-001-display-all-general-all-exact-true",
        demand_scale=3.00,
        feeder_rider_cost_factor=0.30,
        locker_fixed_cost_per_hour=1.0,
        display_origin_choices=None,
        general_corridor_choices=None,
        expected_screening_is_exact=True,
        interpretation=(
            "threefold order-rate scenario with a more efficient contracted feeder"
        ),
    ),
    QualificationSolveProfile(
        profile_id="demand-4x-feeder-030-fc-001-display-all-general-all-exact-true",
        demand_scale=4.00,
        feeder_rider_cost_factor=0.30,
        locker_fixed_cost_per_hour=1.0,
        display_origin_choices=None,
        general_corridor_choices=None,
        expected_screening_is_exact=True,
        interpretation=(
            "fourfold order-rate scenario with a more efficient contracted feeder"
        ),
    ),
)

TARGET_CASES_PER_BLOCK_SCALE = 2
EXPECTED_BLOCK_SCALE_STRATA = 18


@dataclass(frozen=True, slots=True)
class DemandBlock:
    block_id: str
    source: str
    dates: tuple[str, ...]
    demand_hour_start: int
    demand_hour_end: int
    locked: bool

    def __post_init__(self) -> None:
        if not self.block_id:
            raise ValueError("block_id must be non-empty")
        if self.source not in {"nanshan", "tsl"}:
            raise ValueError("source must be 'nanshan' or 'tsl'")
        if not self.dates or len(set(self.dates)) != len(self.dates):
            raise ValueError("dates must be a non-empty sequence without duplicates")
        if tuple(sorted(self.dates)) != self.dates:
            raise ValueError("dates must be in fixed sorted order")
        if not 0 <= self.demand_hour_start < self.demand_hour_end <= 24:
            raise ValueError("demand hours must be a non-empty interval in [0, 24]")

    def to_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["dates"] = list(self.dates)
        return payload


LOCKED_DEMAND_BLOCKS = (
    DemandBlock("01-nanshan-d04-05-noon", "nanshan", ("4", "5"), 12, 13, True),
    DemandBlock("02-nanshan-d06-07-noon", "nanshan", ("6", "7"), 12, 13, True),
    DemandBlock(
        "03-nanshan-d04-07-evening",
        "nanshan",
        ("4", "5", "6", "7"),
        18,
        19,
        True,
    ),
    DemandBlock(
        "04-tsl-20221021-22-noon",
        "tsl",
        ("20221021", "20221022"),
        12,
        13,
        True,
    ),
    DemandBlock(
        "05-tsl-20221023-24-noon",
        "tsl",
        ("20221023", "20221024"),
        12,
        13,
        True,
    ),
    DemandBlock(
        "06-tsl-20221021-24-evening",
        "tsl",
        ("20221021", "20221022", "20221023", "20221024"),
        18,
        19,
        True,
    ),
)

# Development/calibration data are explicit but disjoint from the locked pool.
DEVELOPMENT_DEMAND_BLOCK = DemandBlock(
    "dev-tsl-20221017-20-noon",
    "tsl",
    ("20221017", "20221018", "20221019", "20221020"),
    12,
    13,
    False,
)


class GeneratorProtocol(Protocol):
    def generate(self, config: DataConfig) -> str: ...


GeneratorFactory = Callable[[], GeneratorProtocol]


@dataclass(frozen=True, slots=True)
class LockedCasePlan:
    """One physical generation plan, shared by all solve profiles."""

    case_id: str
    block: DemandBlock
    scale: ScaleCaseSpec
    layout_seed: int

    def data_config(
        self,
        *,
        raw_path: Path,
        output_dir: Path,
        candidate_hour_start: int,
        candidate_hour_end: int,
        site_grid_km: float,
    ) -> DataConfig:
        return DataConfig(
            source=self.block.source,
            raw_path=str(raw_path),
            output_dir=str(output_dir),
            candidate_count=self.scale.candidate_h,
            num_bases=self.scale.candidate_l,
            num_swaps=self.scale.candidate_e,
            candidate_hour_start=candidate_hour_start,
            candidate_hour_end=candidate_hour_end,
            demand_hour_start=self.block.demand_hour_start,
            demand_hour_end=self.block.demand_hour_end,
            map_half_size_km=self.scale.map_half_size_km,
            od_grid_km=self.scale.od_grid_km,
            site_grid_km=site_grid_km,
            seed=self.layout_seed,
            dates=self.block.dates,
        )


@dataclass(frozen=True, slots=True)
class QualificationCandidatePlan:
    """One logical qualification candidate over a generated spatial instance."""

    instance_plan: LockedCasePlan
    solve_profile: QualificationSolveProfile
    replacement_rank_within_stratum: int

    @property
    def case_id(self) -> str:
        return f"{self.instance_plan.case_id}__profile-{self.solve_profile.profile_id}"

    @property
    def stratum_id(self) -> str:
        return f"{self.instance_plan.block.block_id}__{self.instance_plan.scale.scale}"


@dataclass(frozen=True, slots=True)
class CaseBuildConfig:
    """Caller-owned source paths and output location for the fixed design."""

    output_root: Path
    nanshan_raw_path: Path
    tsl_raw_path: Path
    candidate_hour_start: int = 8
    candidate_hour_end: int = 24
    site_grid_km: float = 0.25
    pool_label: str = "mlnd-real-data-overcomplete-qualification-v5"

    def validate(self) -> None:
        if not self.pool_label:
            raise ValueError("pool_label must be non-empty")
        if not 0 <= self.candidate_hour_start < self.candidate_hour_end <= 24:
            raise ValueError("candidate hours must be a non-empty interval in [0, 24]")
        if not math.isfinite(self.site_grid_km) or self.site_grid_km <= 0.0:
            raise ValueError("site_grid_km must be positive and finite")
        for source, path in self.source_paths().items():
            if not path.expanduser().is_file():
                raise FileNotFoundError(f"{source} raw source does not exist: {path}")

    def source_paths(self) -> dict[str, Path]:
        return {
            "nanshan": self.nanshan_raw_path.expanduser().resolve(),
            "tsl": self.tsl_raw_path.expanduser().resolve(),
        }


@dataclass(frozen=True, slots=True)
class VerifiedCanonicalInstance:
    instance_sha256: str
    canonical_file_sha256: Mapping[str, str]
    projection_center: tuple[float, float]
    metadata: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class CaseBuildResult:
    pool: CandidatePool
    pool_path: Path
    case_directories: Mapping[str, Path]


def locked_instance_plans() -> tuple[LockedCasePlan, ...]:
    """Return 6 blocks x 3 map scales x 4 registered physical layouts."""

    plans = tuple(
        LockedCasePlan(
            case_id=f"{block.block_id}__{scale.scale}__seed{seed:010d}",
            block=block,
            scale=scale,
            layout_seed=seed,
        )
        for block in LOCKED_DEMAND_BLOCKS
        for scale in LOCKED_SCALE_SPECS
        for seed in LOCKED_LAYOUT_SEED_ORDER
    )
    expected_count = (
        len(LOCKED_DEMAND_BLOCKS)
        * len(LOCKED_SCALE_SPECS)
        * len(LOCKED_LAYOUT_SEED_ORDER)
    )
    if (
        len(plans) != expected_count
        or len({plan.case_id for plan in plans}) != expected_count
    ):
        raise AssertionError(
            "locked physical-instance design has duplicate/missing cases"
        )
    return plans


def locked_replacement_order() -> tuple[tuple[str, int], ...]:
    """Return the explicit profile-major, seed-minor replacement ladder."""

    return tuple(
        (profile.profile_id, seed)
        for profile in LOCKED_QUALIFICATION_PROFILES
        for seed in LOCKED_LAYOUT_SEED_ORDER
    )


def locked_case_plans() -> tuple[QualificationCandidatePlan, ...]:
    """Return the overcomplete logical pool in its pre-registered order."""

    instance_by_key = {
        (
            plan.block.block_id,
            plan.scale.scale,
            plan.layout_seed,
        ): plan
        for plan in locked_instance_plans()
    }
    plans: list[QualificationCandidatePlan] = []
    for block in LOCKED_DEMAND_BLOCKS:
        for scale in LOCKED_SCALE_SPECS:
            for replacement_rank, (profile_id, seed) in enumerate(
                locked_replacement_order()
            ):
                profile = next(
                    item
                    for item in LOCKED_QUALIFICATION_PROFILES
                    if item.profile_id == profile_id
                )
                plans.append(
                    QualificationCandidatePlan(
                        instance_plan=instance_by_key[
                            (block.block_id, scale.scale, seed)
                        ],
                        solve_profile=profile,
                        replacement_rank_within_stratum=replacement_rank,
                    )
                )
    expected_count = (
        EXPECTED_BLOCK_SCALE_STRATA
        * len(LOCKED_LAYOUT_SEED_ORDER)
        * len(LOCKED_QUALIFICATION_PROFILES)
    )
    if (
        len(plans) != expected_count
        or len({plan.case_id for plan in plans}) != expected_count
    ):
        raise AssertionError("overcomplete qualification design is not unique")
    return tuple(plans)


def _require_empty_output_root(output_root: Path) -> None:
    if output_root.exists():
        if not output_root.is_dir():
            raise FileExistsError(f"output root is not a directory: {output_root}")
        existing = sorted(path.name for path in output_root.iterdir())
        if existing:
            raise FileExistsError(
                "refusing to reuse a non-empty candidate-pool root; existing or "
                f"partial artifacts: {existing}"
            )
        return
    output_root.mkdir(parents=True, exist_ok=False)


def _sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_instance_hash(instance_dir: Path) -> tuple[str, dict[str, str]]:
    """Content-address the five canonical instance files in fixed order."""

    missing = [
        filename
        for filename in CANONICAL_INSTANCE_FILES
        if not (instance_dir / filename).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"canonical instance {instance_dir} is missing files: {missing}"
        )
    file_hashes = {
        filename: _sha256_file(instance_dir / filename)
        for filename in CANONICAL_INSTANCE_FILES
    }
    aggregate = canonical_sha256(
        {
            "hash_schema": "mlnd-canonical-instance-files-v1",
            "ordered_files": [
                {"name": filename, "sha256": file_hashes[filename]}
                for filename in CANONICAL_INSTANCE_FILES
            ],
        }
    )
    return aggregate, file_hashes


def _read_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"canonical CSV has no header: {path}")
        return [dict(row) for row in reader]


def _finite_float(value: Any, *, label: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be numeric") from error
    if not math.isfinite(parsed):
        raise ValueError(f"{label} must be finite")
    return parsed


def _validate_expected_data_config(
    metadata: Mapping[str, Any],
    expected: DataConfig,
    instance_dir: Path,
) -> None:
    if metadata.get("source") != expected.source:
        raise ValueError(f"{instance_dir}: generated source does not match plan")
    if metadata.get("coordinate_system") != "local_km":
        raise ValueError(f"{instance_dir}: coordinate_system must be local_km")
    data_config = metadata.get("data_config")
    if not isinstance(data_config, Mapping):
        raise ValueError(f"{instance_dir}: metadata is missing data_config")
    expected_fields: dict[str, Any] = {
        "source": expected.source,
        "raw_path": expected.raw_path,
        "output_dir": expected.output_dir,
        "candidate_count": expected.candidate_count,
        "num_bases": expected.num_bases,
        "num_swaps": expected.num_swaps,
        "candidate_hour_start": expected.candidate_hour_start,
        "candidate_hour_end": expected.candidate_hour_end,
        "demand_hour_start": expected.demand_hour_start,
        "demand_hour_end": expected.demand_hour_end,
        "map_half_size_km": expected.map_half_size_km,
        "od_grid_km": expected.od_grid_km,
        "site_grid_km": expected.site_grid_km,
        "seed": expected.seed,
        "dates": list(expected.dates),
    }
    mismatches = [
        f"{field}: expected {expected_value!r}, got {data_config.get(field)!r}"
        for field, expected_value in expected_fields.items()
        if data_config.get(field) != expected_value
    ]
    if mismatches:
        raise ValueError(
            f"{instance_dir}: preprocessing contract mismatch: " + "; ".join(mismatches)
        )
    if metadata.get("selected_days") != list(expected.dates):
        raise ValueError(f"{instance_dir}: selected_days do not match locked dates")


def _validate_site_counts_and_crop(
    instance_dir: Path,
    expected: DataConfig,
) -> None:
    sites = _read_csv(instance_dir / "sites.csv")
    counts = {site_type: 0 for site_type in ("H", "L", "E")}
    for row in sites:
        site_type = row.get("type")
        if site_type not in counts:
            raise ValueError(f"{instance_dir}: unknown site type {site_type!r}")
        counts[site_type] += 1
        x = _finite_float(row.get("x_km"), label="sites.csv x_km")
        y = _finite_float(row.get("y_km"), label="sites.csv y_km")
        if (
            abs(x) > expected.map_half_size_km + 1e-8
            or abs(y) > expected.map_half_size_km + 1e-8
        ):
            raise ValueError(f"{instance_dir}: sites.csv contains a point outside crop")
    expected_counts = {
        "H": expected.candidate_count,
        "L": expected.num_bases,
        "E": expected.num_swaps,
    }
    if counts != expected_counts:
        raise ValueError(
            f"{instance_dir}: candidate counts mismatch; "
            f"expected={expected_counts}, actual={counts}"
        )

    for row in _read_csv(instance_dir / "od_nodes.csv"):
        x = _finite_float(row.get("x_km"), label="od_nodes.csv x_km")
        y = _finite_float(row.get("y_km"), label="od_nodes.csv y_km")
        if (
            abs(x) > expected.map_half_size_km + 1e-8
            or abs(y) > expected.map_half_size_km + 1e-8
        ):
            raise ValueError(
                f"{instance_dir}: od_nodes.csv contains a point outside crop"
            )


def verify_generated_case(
    instance_dir: Path,
    expected: DataConfig,
) -> VerifiedCanonicalInstance:
    """Check the generated metadata, candidate counts, crop, and file identity."""

    instance_sha256, file_hashes = canonical_instance_hash(instance_dir)
    metadata = _read_json_object(instance_dir / "metadata.json")
    _validate_expected_data_config(metadata, expected, instance_dir)
    _validate_site_counts_and_crop(instance_dir, expected)
    center = (
        _finite_float(
            metadata.get("projection_center_lon"),
            label="projection_center_lon",
        ),
        _finite_float(
            metadata.get("projection_center_lat"),
            label="projection_center_lat",
        ),
    )
    return VerifiedCanonicalInstance(
        instance_sha256=instance_sha256,
        canonical_file_sha256=file_hashes,
        projection_center=center,
        metadata=metadata,
    )


def _generator_metadata(
    plan: QualificationCandidatePlan,
    data_config: DataConfig,
    verified: VerifiedCanonicalInstance,
) -> dict[str, Any]:
    instance_plan = plan.instance_plan
    solve_profile = plan.solve_profile
    return {
        "design": "same-center spatial crop with per-case site reclustering",
        "crop_rule": "retain order only when both endpoints are inside centered square",
        "orders_sampled": False,
        "demand_rescaled": False,
        "source": instance_plan.block.source,
        "demand_block": instance_plan.block.to_payload(),
        "map_half_size_km": instance_plan.scale.map_half_size_km,
        "od_grid_km": instance_plan.scale.od_grid_km,
        "site_grid_km": data_config.site_grid_km,
        "candidate_counts": {
            "H": instance_plan.scale.candidate_h,
            "L": instance_plan.scale.candidate_l,
            "E": instance_plan.scale.candidate_e,
        },
        "candidate_hours": [
            data_config.candidate_hour_start,
            data_config.candidate_hour_end,
        ],
        "layout_seed": instance_plan.layout_seed,
        "qualification_solve_profile": solve_profile.to_payload(),
        "qualification_model_identity": {
            "schema": QUALIFICATION_MODEL_IDENTITY_SCHEMA,
            "effective_model_config_sha256": (
                solve_profile.effective_model_config_sha256()
            ),
            "qualification_case_sha256": qualification_case_sha256(
                verified.instance_sha256,
                solve_profile,
            ),
            "expected_screening_is_exact": solve_profile.expected_screening_is_exact,
        },
        "pre_registered_selection": {
            "selection_rule": OVERCOMPLETE_SELECTION_RULE,
            "stratum_id": plan.stratum_id,
            "replacement_rank_within_stratum": (plan.replacement_rank_within_stratum),
            "target_cases_per_stratum": TARGET_CASES_PER_BLOCK_SCALE,
            "expected_stratum_count": EXPECTED_BLOCK_SCALE_STRATA,
            "selection_uses_algorithm_performance": False,
            "large_scale_open_h_qualification_target": (
                [50, 60] if instance_plan.scale.scale == "large" else None
            ),
            "forced_opening_constraints_added": False,
        },
        "raw_file_sha256": verified.metadata.get("raw_file_sha256"),
        "canonical_file_sha256": dict(verified.canonical_file_sha256),
        "projection_center_lon": verified.projection_center[0],
        "projection_center_lat": verified.projection_center[1],
    }


def build_locked_candidate_pool(
    config: CaseBuildConfig,
    *,
    generator_factory: GeneratorFactory = DataGenerator,
) -> CaseBuildResult:
    """Generate the overcomplete pool from which 36 formal cases are frozen.

    The output root must be absent or empty.  Claimed case directories and the
    final ``pool.json`` are never overwritten; an interrupted build therefore
    remains visible and must be inspected rather than silently resumed.
    """

    config.validate()
    output_root = config.output_root.expanduser().resolve()
    _require_empty_output_root(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    source_paths = config.source_paths()

    verified_by_instance: dict[
        str, tuple[DataConfig, VerifiedCanonicalInstance, Path]
    ] = {}
    centers_by_block: dict[str, tuple[float, float]] = {}
    instances_root = output_root / "instances"
    instances_root.mkdir(exist_ok=False)
    for instance_plan in locked_instance_plans():
        case_dir = instances_root / instance_plan.case_id
        case_dir.mkdir(exist_ok=False)
        data_config = instance_plan.data_config(
            raw_path=source_paths[instance_plan.block.source],
            output_dir=case_dir,
            candidate_hour_start=config.candidate_hour_start,
            candidate_hour_end=config.candidate_hour_end,
            site_grid_km=config.site_grid_km,
        )
        data_config.validate()
        generated_path = Path(generator_factory().generate(data_config)).resolve()
        if generated_path != case_dir:
            raise ValueError(
                f"generator returned {generated_path}, expected claimed directory {case_dir}"
            )
        verified = verify_generated_case(case_dir, data_config)
        previous_center = centers_by_block.setdefault(
            instance_plan.block.block_id,
            verified.projection_center,
        )
        if verified.projection_center != previous_center:
            raise ValueError(
                f"demand block {instance_plan.block.block_id!r} does not share one map center"
            )

        verified_by_instance[instance_plan.case_id] = (
            data_config,
            verified,
            case_dir,
        )

    candidates: list[CandidateCase] = []
    case_directories: dict[str, Path] = {}
    for candidate_plan in locked_case_plans():
        instance_plan = candidate_plan.instance_plan
        data_config, verified, case_dir = verified_by_instance[instance_plan.case_id]
        candidates.append(
            CandidateCase(
                case_id=candidate_plan.case_id,
                scale=instance_plan.scale.scale,
                demand_block_id=instance_plan.block.block_id,
                layout_seed=instance_plan.layout_seed,
                instance_ref=f"instances/{instance_plan.case_id}",
                instance_sha256=verified.instance_sha256,
                generator_parameters=_generator_metadata(
                    candidate_plan,
                    data_config,
                    verified,
                ),
            )
        )
        case_directories[candidate_plan.case_id] = case_dir

    pool = build_candidate_pool(
        candidates,
        candidate_seed_order=LOCKED_LAYOUT_SEED_ORDER,
        pool_label=config.pool_label,
    )
    expected_pool_size = (
        EXPECTED_BLOCK_SCALE_STRATA
        * len(LOCKED_LAYOUT_SEED_ORDER)
        * len(LOCKED_QUALIFICATION_PROFILES)
    )
    if len(pool.candidates) != expected_pool_size:
        raise AssertionError("overcomplete candidate pool size changed unexpectedly")
    pool_path = write_canonical_json_exclusive(
        output_root / "pool.json",
        {
            "artifact_type": "mlnd_branch_cut_candidate_pool",
            "artifact_schema_version": 2,
            "pool_sha256": pool.sha256,
            "pool": pool.to_payload(),
            "canonical_instance_files": list(CANONICAL_INSTANCE_FILES),
            "locked_scale_specs": [asdict(spec) for spec in LOCKED_SCALE_SPECS],
            "locked_demand_blocks": [
                block.to_payload() for block in LOCKED_DEMAND_BLOCKS
            ],
            "locked_layout_seed_order": list(LOCKED_LAYOUT_SEED_ORDER),
            "locked_qualification_profiles": [
                profile.to_payload() for profile in LOCKED_QUALIFICATION_PROFILES
            ],
            "selection_protocol": {
                "selection_rule": OVERCOMPLETE_SELECTION_RULE,
                "target_cases_per_block_scale_stratum": (TARGET_CASES_PER_BLOCK_SCALE),
                "expected_block_scale_strata": EXPECTED_BLOCK_SCALE_STRATA,
                "replacement_order": [
                    {
                        "replacement_rank_within_stratum": rank,
                        "qualification_profile_id": profile_id,
                        "layout_seed": seed,
                    }
                    for rank, (profile_id, seed) in enumerate(
                        locked_replacement_order()
                    )
                ],
                "selection_uses_algorithm_performance": False,
                "rejection_ledger_required": True,
                "forced_opening_constraints_added": False,
            },
            "development_block_excluded_from_pool": (
                DEVELOPMENT_DEMAND_BLOCK.to_payload()
            ),
        },
    )
    return CaseBuildResult(
        pool=pool,
        pool_path=pool_path,
        case_directories=case_directories,
    )
