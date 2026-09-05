"""Immutable experiment-contract types for the branch-and-cut benchmark."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from enum import Enum


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _require_nonempty(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a nonempty string")


def _require_sha256(value: str, field_name: str) -> None:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{field_name} must be a lowercase 64-character SHA256 digest")


def _require_bool(value: bool, field_name: str) -> None:
    if not isinstance(value, bool):
        raise ValueError(f"{field_name} must be a boolean")


def _require_integer(value: int, field_name: str, *, minimum: int, maximum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer")
    if not minimum <= value <= maximum:
        raise ValueError(f"{field_name} must be in [{minimum}, {maximum}]")


def _finite_float(value: float, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be numeric")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{field_name} must be finite")
    return normalized


class BenchmarkMode(str, Enum):
    """Supported MLND service-design modes."""

    MX = "MX"
    HD = "HD"
    LO = "LO"


class StartMode(str, Enum):
    """Exactly which custom primal starts may be installed."""

    NONE = "none"
    COURIER = "courier"
    NETWORK = "network"
    NETWORK_AND_COURIER = "network_and_courier"


class ProfileCode(str, Enum):
    """Pre-registered algorithm profiles used in the paper benchmark."""

    A = "A"
    B = "B"
    C = "C"
    D = "D"
    E = "E"


@dataclass(frozen=True, slots=True)
class ModelIdentity:
    """Content identity of the formulation and implementation being tested."""

    formulation_name: str
    formulation_schema_version: int
    implementation_sha256: str
    model_config_sha256: str

    def __post_init__(self) -> None:
        _require_nonempty(self.formulation_name, "formulation_name")
        _require_integer(
            self.formulation_schema_version,
            "formulation_schema_version",
            minimum=1,
            maximum=2_000_000_000,
        )
        _require_sha256(self.implementation_sha256, "implementation_sha256")
        _require_sha256(self.model_config_sha256, "model_config_sha256")


@dataclass(frozen=True, slots=True)
class SolverConfig:
    """Native solver settings held fixed across every matched profile.

    The time budget and random seed are intentionally run axes in
    :class:`PairSpec`, not fields here.  This prevents an A--E comparison from
    silently changing either axis through a profile-specific solver object.
    """

    threads: int = 1
    method: int = -1
    miqcp_method: int = -1
    mip_focus: int = 0
    heuristics: float = 0.05
    no_rel_heur_time: float = 0.0
    presolve: int = -1
    cuts: int = -1
    root_cut_passes: int = 2
    numeric_focus: int = 0
    feasibility_tol: float = 1e-6
    optimality_tol: float = 1e-6
    integer_feasibility_tol: float = 1e-5
    target_mip_gap: float = 0.0
    output_flag: int = 1

    def __post_init__(self) -> None:
        _require_integer(self.threads, "threads", minimum=1, maximum=1024)
        _require_integer(self.method, "method", minimum=-1, maximum=5)
        _require_integer(
            self.miqcp_method,
            "miqcp_method",
            minimum=-1,
            maximum=1,
        )
        _require_integer(self.mip_focus, "mip_focus", minimum=0, maximum=3)
        _require_integer(self.presolve, "presolve", minimum=-1, maximum=2)
        _require_integer(self.cuts, "cuts", minimum=-1, maximum=3)
        _require_integer(
            self.root_cut_passes,
            "root_cut_passes",
            minimum=-1,
            maximum=2_000_000_000,
        )
        _require_integer(self.numeric_focus, "numeric_focus", minimum=0, maximum=3)
        _require_integer(self.output_flag, "output_flag", minimum=0, maximum=1)
        for name in (
            "heuristics",
            "no_rel_heur_time",
            "feasibility_tol",
            "optimality_tol",
            "integer_feasibility_tol",
            "target_mip_gap",
        ):
            object.__setattr__(self, name, _finite_float(getattr(self, name), name))
        if not 0.0 <= self.heuristics <= 1.0:
            raise ValueError("heuristics must be in [0, 1]")
        if self.no_rel_heur_time < 0.0:
            raise ValueError("no_rel_heur_time must be nonnegative")
        if not 1e-9 <= self.feasibility_tol <= 1e-2:
            raise ValueError("feasibility_tol must be in [1e-9, 1e-2]")
        if not 1e-9 <= self.optimality_tol <= 1e-2:
            raise ValueError("optimality_tol must be in [1e-9, 1e-2]")
        if not 1e-9 <= self.integer_feasibility_tol <= 1e-1:
            raise ValueError("integer_feasibility_tol must be in [1e-9, 1e-1]")
        if not 0.0 <= self.target_mip_gap <= 1.0:
            raise ValueError("target_mip_gap must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class StaticStrengtheningConfig:
    """Per-family controls for valid inequalities added before optimization."""

    enabled: bool = False
    source_level_upper: bool = False
    virtual_level_lower: bool = False
    lo_choice_corridor_cover: bool = False
    lo_choice_capacity_projection: bool = False
    queue_perspective_support: bool = False

    def __post_init__(self) -> None:
        for name in (
            "enabled",
            "source_level_upper",
            "virtual_level_lower",
            "lo_choice_corridor_cover",
            "lo_choice_capacity_projection",
            "queue_perspective_support",
        ):
            _require_bool(getattr(self, name), name)
        family_values = self.family_values()
        if self.enabled != any(family_values):
            raise ValueError(
                "static enabled must be true exactly when at least one family is enabled"
            )

    def family_values(self) -> tuple[bool, ...]:
        return (
            self.source_level_upper,
            self.virtual_level_lower,
            self.lo_choice_corridor_cover,
            self.lo_choice_capacity_projection,
            self.queue_perspective_support,
        )

    @classmethod
    def disabled(cls) -> StaticStrengtheningConfig:
        return cls()

    @classmethod
    def full(cls) -> StaticStrengtheningConfig:
        return cls(
            enabled=True,
            source_level_upper=True,
            virtual_level_lower=True,
            lo_choice_corridor_cover=True,
            lo_choice_capacity_projection=True,
            queue_perspective_support=True,
        )


@dataclass(frozen=True, slots=True)
class UserCutConfig:
    """Per-family controls and a global root-separation budget."""

    enabled: bool = False
    aggregated_od_corridor_vub: bool = False
    lo_choice_corridor_support: bool = False
    lo_choice_capacity_cover: bool = False
    queue_perspective: bool = False
    directed_return_connectivity: bool = False
    max_root_rounds: int = 3
    max_cuts_per_round: int = 250
    violation_tolerance: float = 1e-6
    efficacy_tolerance: float = 1e-9

    def __post_init__(self) -> None:
        for name in (
            "enabled",
            "aggregated_od_corridor_vub",
            "lo_choice_corridor_support",
            "lo_choice_capacity_cover",
            "queue_perspective",
            "directed_return_connectivity",
        ):
            _require_bool(getattr(self, name), name)
        family_values = self.family_values()
        if self.enabled != any(family_values):
            raise ValueError(
                "user-cut enabled must be true exactly when at least one family is enabled"
            )
        _require_integer(
            self.max_root_rounds,
            "max_root_rounds",
            minimum=0,
            maximum=1_000_000,
        )
        _require_integer(
            self.max_cuts_per_round,
            "max_cuts_per_round",
            minimum=0,
            maximum=1_000_000,
        )
        object.__setattr__(
            self,
            "violation_tolerance",
            _finite_float(self.violation_tolerance, "violation_tolerance"),
        )
        object.__setattr__(
            self,
            "efficacy_tolerance",
            _finite_float(self.efficacy_tolerance, "efficacy_tolerance"),
        )
        if self.violation_tolerance <= 0.0:
            raise ValueError("violation_tolerance must be positive")
        if self.efficacy_tolerance < 0.0:
            raise ValueError("efficacy_tolerance must be nonnegative")
        if self.enabled and (self.max_root_rounds == 0 or self.max_cuts_per_round == 0):
            raise ValueError("enabled user cuts require positive round and cut budgets")

    def family_values(self) -> tuple[bool, ...]:
        return (
            self.aggregated_od_corridor_vub,
            self.lo_choice_corridor_support,
            self.lo_choice_capacity_cover,
            self.queue_perspective,
            self.directed_return_connectivity,
        )

    @classmethod
    def disabled(cls) -> UserCutConfig:
        return cls()

    @classmethod
    def full(cls) -> UserCutConfig:
        return cls(
            enabled=True,
            aggregated_od_corridor_vub=True,
            lo_choice_corridor_support=True,
            lo_choice_capacity_cover=True,
            queue_perspective=True,
            directed_return_connectivity=True,
        )


@dataclass(frozen=True, slots=True)
class StartConfig:
    """Custom primal-start and local-branching contract.

    ``start_mode=none`` is a strict state: it carries zero construction time and
    disables local branching.  The execution layer can therefore map it to
    ``NumStart=0`` and undefined variable starts without guessing intent.
    """

    start_mode: StartMode = StartMode.NONE
    network_start_seconds: float = 0.0
    network_start_method: int = 2
    fixed_integer_repair_seconds: float = 0.0
    local_branching: bool = False
    local_branching_seconds: float = 0.0
    local_branching_radius: int = 12
    local_branching_include_topology: bool = False

    def __post_init__(self) -> None:
        try:
            normalized_mode = StartMode(self.start_mode)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid start mode: {self.start_mode!r}") from exc
        object.__setattr__(self, "start_mode", normalized_mode)
        _require_integer(
            self.network_start_method,
            "network_start_method",
            minimum=-1,
            maximum=5,
        )
        _require_bool(self.local_branching, "local_branching")
        _require_bool(
            self.local_branching_include_topology,
            "local_branching_include_topology",
        )
        for name in (
            "network_start_seconds",
            "fixed_integer_repair_seconds",
            "local_branching_seconds",
        ):
            object.__setattr__(self, name, _finite_float(getattr(self, name), name))
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be nonnegative")
        _require_integer(
            self.local_branching_radius,
            "local_branching_radius",
            minimum=1,
            maximum=1_000_000,
        )
        if self.start_mode is StartMode.NONE:
            if self.network_start_seconds != 0.0 or self.local_branching:
                raise ValueError(
                    "start_mode=none requires zero network time and local branching off"
                )
            if self.local_branching_seconds != 0.0:
                raise ValueError(
                    "start_mode=none requires zero local-branching time"
                )
            if self.fixed_integer_repair_seconds != 0.0:
                raise ValueError(
                    "start_mode=none requires zero fixed-integer repair time"
                )
        if self.start_mode is StartMode.COURIER and self.network_start_seconds != 0.0:
            raise ValueError("courier-only start requires zero network_start_seconds")
        if (
            self.start_mode is StartMode.COURIER
            and self.fixed_integer_repair_seconds != 0.0
        ):
            raise ValueError("courier-only start requires zero repair time")
        if self.start_mode in {StartMode.NETWORK, StartMode.NETWORK_AND_COURIER}:
            if self.network_start_seconds <= 0.0:
                raise ValueError("network start modes require positive network_start_seconds")
        if self.local_branching and self.local_branching_seconds <= 0.0:
            raise ValueError("local branching requires positive local_branching_seconds")
        if not self.local_branching and self.local_branching_seconds != 0.0:
            raise ValueError("disabled local branching requires zero local_branching_seconds")

    @classmethod
    def none(cls) -> StartConfig:
        return cls(start_mode=StartMode.NONE)

    @classmethod
    def full_heuristic(cls) -> StartConfig:
        return cls(
            start_mode=StartMode.NETWORK_AND_COURIER,
            network_start_seconds=20.0,
            network_start_method=2,
            fixed_integer_repair_seconds=10.0,
            local_branching=True,
            local_branching_seconds=10.0,
            local_branching_radius=12,
            local_branching_include_topology=True,
        )

    @property
    def mode(self) -> StartMode:
        """Compatibility shorthand; serialized contracts use ``start_mode``."""

        return self.start_mode


@dataclass(frozen=True, slots=True)
class TelemetryConfig:
    """Read-only instrumentation shared by every matched run."""

    progress_interval_seconds: float = 1.0
    capture_root_rounds: bool = True
    capture_cut_stats: bool = True
    capture_incumbent_events: bool = True
    solver_log: bool = True

    def __post_init__(self) -> None:
        for name in (
            "capture_root_rounds",
            "capture_cut_stats",
            "capture_incumbent_events",
            "solver_log",
        ):
            _require_bool(getattr(self, name), name)
        object.__setattr__(
            self,
            "progress_interval_seconds",
            _finite_float(self.progress_interval_seconds, "progress_interval_seconds"),
        )
        if self.progress_interval_seconds <= 0.0:
            raise ValueError("progress_interval_seconds must be positive")


@dataclass(frozen=True, slots=True)
class ProfileConfig:
    """One algorithm profile; native solver settings live in its suite."""

    code: ProfileCode
    static: StaticStrengtheningConfig = field(
        default_factory=StaticStrengtheningConfig.disabled
    )
    user_cuts: UserCutConfig = field(default_factory=UserCutConfig.disabled)
    start: StartConfig = field(default_factory=StartConfig.none)

    def __post_init__(self) -> None:
        try:
            object.__setattr__(self, "code", ProfileCode(self.code))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid profile code: {self.code!r}") from exc
        if not isinstance(self.static, StaticStrengtheningConfig):
            raise ValueError("static must be a StaticStrengtheningConfig")
        if not isinstance(self.user_cuts, UserCutConfig):
            raise ValueError("user_cuts must be a UserCutConfig")
        if not isinstance(self.start, StartConfig):
            raise ValueError("start must be a StartConfig")

    @property
    def label(self) -> str:
        return {
            ProfileCode.A: "native",
            ProfileCode.B: "static_only",
            ProfileCode.C: "static_and_cuts",
            ProfileCode.D: "heuristic_only",
            ProfileCode.E: "full_branch_and_cut",
        }[self.code]


@dataclass(frozen=True, slots=True)
class MatchedProfileSuite:
    """A--E profiles sharing one immutable solver and telemetry contract."""

    solver: SolverConfig
    telemetry: TelemetryConfig
    profiles: tuple[ProfileConfig, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.solver, SolverConfig):
            raise ValueError("solver must be a SolverConfig")
        if not isinstance(self.telemetry, TelemetryConfig):
            raise ValueError("telemetry must be a TelemetryConfig")
        object.__setattr__(self, "profiles", tuple(self.profiles))
        if any(not isinstance(profile, ProfileConfig) for profile in self.profiles):
            raise ValueError("profiles must contain only ProfileConfig values")
        if self.solver.threads != 1:
            raise ValueError("paper-grade matched profiles require solver threads=1")
        self.validate()

    def validate(self) -> None:
        """Enforce the pre-registered order and component meaning of A--E."""

        expected_codes = tuple(ProfileCode)
        actual_codes = tuple(profile.code for profile in self.profiles)
        if actual_codes != expected_codes:
            raise ValueError("paper profile order must be exactly A, B, C, D, E")
        expected_components = {
            ProfileCode.A: (False, False, False),
            ProfileCode.B: (True, False, False),
            ProfileCode.C: (True, True, False),
            ProfileCode.D: (False, False, True),
            ProfileCode.E: (True, True, True),
        }
        for profile in self.profiles:
            static_on, cuts_on, heuristic_on = expected_components[profile.code]
            actual = (
                profile.static.enabled,
                profile.user_cuts.enabled,
                profile.start.mode is not StartMode.NONE,
            )
            if actual != (static_on, cuts_on, heuristic_on):
                raise ValueError(
                    f"profile {profile.code.value} must have components "
                    f"static={static_on}, cuts={cuts_on}, heuristic={heuristic_on}"
                )
            if static_on and not all(profile.static.family_values()):
                raise ValueError(
                    f"profile {profile.code.value} must enable every static family"
                )
            if cuts_on and (
                not all(
                    (
                        profile.user_cuts.aggregated_od_corridor_vub,
                        profile.user_cuts.lo_choice_corridor_support,
                        profile.user_cuts.lo_choice_capacity_cover,
                        profile.user_cuts.queue_perspective,
                    )
                )
                or profile.user_cuts.directed_return_connectivity
            ):
                raise ValueError(
                    f"profile {profile.code.value} must use the canonical paper "
                    "user-cut families with directed return disabled"
                )

        by_code = {profile.code: profile for profile in self.profiles}
        no_static = StaticStrengtheningConfig.disabled()
        no_cuts = UserCutConfig.disabled()
        no_start = StartConfig.none()
        for code in (ProfileCode.A, ProfileCode.D):
            if by_code[code].static != no_static:
                raise ValueError(f"profile {code.value} must use the canonical static-off config")
        for code in (ProfileCode.A, ProfileCode.B, ProfileCode.D):
            if by_code[code].user_cuts != no_cuts:
                raise ValueError(f"profile {code.value} must use the canonical user-cut-off config")
        for code in (ProfileCode.A, ProfileCode.B, ProfileCode.C):
            if by_code[code].start != no_start:
                raise ValueError(f"profile {code.value} must use exact start_mode=none")
        if not (
            by_code[ProfileCode.B].static
            == by_code[ProfileCode.C].static
            == by_code[ProfileCode.E].static
        ):
            raise ValueError("profiles B, C, and E must share one static configuration")
        if by_code[ProfileCode.C].user_cuts != by_code[ProfileCode.E].user_cuts:
            raise ValueError("profiles C and E must share one user-cut configuration")
        if by_code[ProfileCode.D].start != by_code[ProfileCode.E].start:
            raise ValueError("profiles D and E must share one heuristic configuration")

    def profile(self, code: ProfileCode | str) -> ProfileConfig:
        wanted = ProfileCode(code)
        matches = tuple(profile for profile in self.profiles if profile.code is wanted)
        if len(matches) != 1:
            raise KeyError(f"suite does not contain exactly one profile {wanted.value}")
        return matches[0]


@dataclass(frozen=True, slots=True)
class PairSpec:
    """Axes shared by a matched A--E solve group.

    ``manifest_id`` is expected to identify a frozen manifest that, in turn,
    pins the model identity and the canonical instance artifacts.
    """

    manifest_id: str
    case_id: str
    mode: BenchmarkMode
    solver_seed: int
    time_budget_seconds: float
    solver: SolverConfig

    def __post_init__(self) -> None:
        _require_nonempty(self.manifest_id, "manifest_id")
        _require_nonempty(self.case_id, "case_id")
        try:
            object.__setattr__(self, "mode", BenchmarkMode(self.mode))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid benchmark mode: {self.mode!r}") from exc
        _require_integer(
            self.solver_seed,
            "solver_seed",
            minimum=0,
            maximum=2_000_000_000,
        )
        object.__setattr__(
            self,
            "time_budget_seconds",
            _finite_float(self.time_budget_seconds, "time_budget_seconds"),
        )
        if self.time_budget_seconds <= 0.0:
            raise ValueError("time_budget_seconds must be positive")
        if not isinstance(self.solver, SolverConfig):
            raise ValueError("solver must be a SolverConfig")
        if self.solver.threads != 1:
            raise ValueError("matched benchmark pairs require solver threads=1")
