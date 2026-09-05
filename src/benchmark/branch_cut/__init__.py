"""Contracts for matched native-Gurobi and branch-and-cut experiments.

This package deliberately has no :mod:`gurobipy` dependency.  It describes
what a run means before the execution layer translates the contract into
solver parameters and algorithm switches.
"""

from .artifacts import (
    AttemptClaim,
    attempt_directory,
    claim_attempt_directory,
    write_canonical_json_exclusive,
)
from .hashing import (
    build_attempt_id,
    build_logical_run_id,
    build_model_id,
    build_pair_id,
    build_profile_id,
    build_solver_id,
    canonical_json,
    canonical_sha256,
)
from .profiles import build_paper_profile_suite, validate_paper_profile_suite
from .schema import (
    BenchmarkMode,
    MatchedProfileSuite,
    ModelIdentity,
    PairSpec,
    ProfileCode,
    ProfileConfig,
    SolverConfig,
    StartConfig,
    StartMode,
    StaticStrengtheningConfig,
    TelemetryConfig,
    UserCutConfig,
)

__all__ = [
    "AttemptClaim",
    "BenchmarkMode",
    "MatchedProfileSuite",
    "ModelIdentity",
    "PairSpec",
    "ProfileCode",
    "ProfileConfig",
    "SolverConfig",
    "StartConfig",
    "StartMode",
    "StaticStrengtheningConfig",
    "TelemetryConfig",
    "UserCutConfig",
    "attempt_directory",
    "build_attempt_id",
    "build_logical_run_id",
    "build_model_id",
    "build_pair_id",
    "build_paper_profile_suite",
    "build_profile_id",
    "build_solver_id",
    "canonical_json",
    "canonical_sha256",
    "claim_attempt_directory",
    "validate_paper_profile_suite",
    "write_canonical_json_exclusive",
]
