"""Pre-registered, solver-matched A--E branch-and-cut profiles."""

from __future__ import annotations

from dataclasses import replace

from .schema import (
    MatchedProfileSuite,
    ProfileCode,
    ProfileConfig,
    SolverConfig,
    StartConfig,
    StartMode,
    StaticStrengtheningConfig,
    TelemetryConfig,
    UserCutConfig,
)


def build_paper_profile_suite(
    *,
    solver: SolverConfig | None = None,
    telemetry: TelemetryConfig | None = None,
    heuristic_start: StartConfig | None = None,
) -> MatchedProfileSuite:
    """Build the strict five-profile paper comparison.

    A is native Gurobi, B adds only static valid inequalities, C adds static
    inequalities and root user cuts, D adds only the custom primal heuristic,
    and E combines all three components.  All five consume the one solver
    object stored by the returned suite.
    """

    common_solver = solver or SolverConfig(threads=1)
    common_telemetry = telemetry or TelemetryConfig()
    start = heuristic_start or StartConfig.full_heuristic()
    if start.mode is StartMode.NONE:
        raise ValueError("heuristic profiles D and E require a non-none start mode")

    static_off = StaticStrengtheningConfig.disabled()
    static_on = StaticStrengtheningConfig.full()
    cuts_off = UserCutConfig.disabled()
    cuts_on = replace(
        UserCutConfig.full(),
        directed_return_connectivity=False,
    )
    no_start = StartConfig.none()

    suite = MatchedProfileSuite(
        solver=common_solver,
        telemetry=common_telemetry,
        profiles=(
            ProfileConfig(ProfileCode.A, static_off, cuts_off, no_start),
            ProfileConfig(ProfileCode.B, static_on, cuts_off, no_start),
            ProfileConfig(ProfileCode.C, static_on, cuts_on, no_start),
            ProfileConfig(ProfileCode.D, static_off, cuts_off, start),
            ProfileConfig(ProfileCode.E, static_on, cuts_on, start),
        ),
    )
    validate_paper_profile_suite(suite)
    return suite


def validate_paper_profile_suite(suite: MatchedProfileSuite) -> None:
    """Reject missing, duplicate, mislabeled, or algorithmically mixed profiles."""

    suite.validate()
