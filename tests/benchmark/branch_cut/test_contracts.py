from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest
import src.benchmark.branch_cut.artifacts as artifact_module

from src.benchmark.branch_cut import (
    BenchmarkMode,
    MatchedProfileSuite,
    ModelIdentity,
    PairSpec,
    ProfileCode,
    SolverConfig,
    StartConfig,
    StartMode,
    StaticStrengtheningConfig,
    UserCutConfig,
    build_attempt_id,
    build_logical_run_id,
    build_model_id,
    build_pair_id,
    build_paper_profile_suite,
    build_profile_id,
    build_solver_id,
    canonical_json,
    canonical_sha256,
    claim_attempt_directory,
    validate_paper_profile_suite,
    write_canonical_json_exclusive,
)


def _pair(solver: SolverConfig | None = None) -> PairSpec:
    return PairSpec(
        manifest_id="manifest-locked-v1",
        case_id="nanshan-small-block01-layout00",
        mode=BenchmarkMode.MX,
        solver_seed=42,
        time_budget_seconds=60,
        solver=solver or SolverConfig(),
    )


def test_canonical_json_and_sha256_are_order_independent() -> None:
    left = {"中文": [1, 2.0], "nested": {"b": False, "a": None}}
    right = {"nested": {"a": None, "b": False}, "中文": [1, 2.0]}

    assert canonical_json(left) == canonical_json(right)
    assert canonical_sha256(left) == canonical_sha256(right)
    assert len(canonical_sha256(left)) == 64
    with pytest.raises(TypeError, match="sets"):
        canonical_json({"unordered": {1, 2}})
    with pytest.raises(ValueError, match="NaN"):
        canonical_json({"bad": float("nan")})


def test_configs_are_frozen_and_none_start_is_exact() -> None:
    start = StartConfig.none()

    assert start.mode is StartMode.NONE
    assert start.network_start_seconds == 0.0
    assert start.network_start_method == 2
    assert start.local_branching is False
    assert start.local_branching_seconds == 0.0
    assert json.loads(canonical_json(start))["start_mode"] == "none"
    with pytest.raises(FrozenInstanceError):
        start.start_mode = StartMode.COURIER  # type: ignore[misc]
    with pytest.raises(ValueError, match="start_mode=none"):
        StartConfig(start_mode=StartMode.NONE, network_start_seconds=1.0)
    with pytest.raises(ValueError, match="network_start_method"):
        StartConfig(network_start_method=6)
    with pytest.raises(ValueError, match="miqcp_method"):
        SolverConfig(miqcp_method=2)
    with pytest.raises(ValueError, match="root_cut_passes"):
        SolverConfig(root_cut_passes=-2)
    with pytest.raises(ValueError, match="exactly"):
        StaticStrengtheningConfig(enabled=True)


def test_paper_profiles_have_strict_components_and_one_solver() -> None:
    solver = SolverConfig(
        threads=1,
        method=2,
        mip_focus=1,
        heuristics=0.25,
        no_rel_heur_time=0.0,
    )
    suite = build_paper_profile_suite(solver=solver)

    assert suite.solver is solver
    assert tuple(profile.code.value for profile in suite.profiles) == tuple("ABCDE")
    expected = {
        "A": (False, False, "none"),
        "B": (True, False, "none"),
        "C": (True, True, "none"),
        "D": (False, False, "network_and_courier"),
        "E": (True, True, "network_and_courier"),
    }
    assert {
        profile.code.value: (
            profile.static.enabled,
            profile.user_cuts.enabled,
            profile.start.mode.value,
        )
        for profile in suite.profiles
    } == expected
    assert all(suite.profile(code).start.mode is StartMode.NONE for code in "ABC")
    assert suite.profile(ProfileCode.E).start.local_branching_include_topology
    generic_full_cuts = UserCutConfig.full()
    assert generic_full_cuts.directed_return_connectivity
    expected_paper_cuts = replace(
        generic_full_cuts,
        directed_return_connectivity=False,
    )
    assert suite.profile(ProfileCode.C).user_cuts == expected_paper_cuts
    assert suite.profile(ProfileCode.E).user_cuts == expected_paper_cuts
    for code in (ProfileCode.C, ProfileCode.E):
        profile = suite.profile(code)
        assert build_profile_id(profile) != build_profile_id(
            replace(profile, user_cuts=generic_full_cuts)
        )

    with pytest.raises(ValueError, match="threads=1"):
        build_paper_profile_suite(solver=replace(solver, threads=2))


def test_profile_validator_rejects_semantic_drift() -> None:
    suite = build_paper_profile_suite()
    bad_profiles = list(suite.profiles)
    bad_profiles[0] = replace(
        bad_profiles[0],
        static=StaticStrengtheningConfig.full(),
    )
    with pytest.raises(ValueError, match="profile A"):
        MatchedProfileSuite(
            solver=suite.solver,
            telemetry=suite.telemetry,
            profiles=tuple(bad_profiles),
        )

    mismatched_heuristic = list(suite.profiles)
    mismatched_heuristic[-1] = replace(
        mismatched_heuristic[-1],
        start=replace(mismatched_heuristic[-1].start, network_start_seconds=21.0),
    )
    with pytest.raises(ValueError, match="D and E"):
        MatchedProfileSuite(
            solver=suite.solver,
            telemetry=suite.telemetry,
            profiles=tuple(mismatched_heuristic),
        )

    directed_return_on = list(suite.profiles)
    for index in (2, 4):
        directed_return_on[index] = replace(
            directed_return_on[index],
            user_cuts=UserCutConfig.full(),
        )
    with pytest.raises(ValueError, match="directed return disabled"):
        MatchedProfileSuite(
            solver=suite.solver,
            telemetry=suite.telemetry,
            profiles=tuple(directed_return_on),
        )
    validate_paper_profile_suite(suite)


def test_identity_builders_are_deterministic_and_domain_separated() -> None:
    model = ModelIdentity(
        formulation_name="mlnd-unified-chi-epsilon-energy-v2",
        formulation_schema_version=3,
        implementation_sha256="a" * 64,
        model_config_sha256="b" * 64,
    )
    suite = build_paper_profile_suite()
    pair = _pair(suite.solver)

    assert build_model_id(model) == build_model_id(model)
    assert build_model_id(model).startswith("model-")
    assert build_solver_id(suite.solver).startswith("solver-")
    assert build_solver_id(suite.solver) != build_solver_id(
        replace(suite.solver, miqcp_method=1)
    )
    assert build_profile_id(suite.profile("A")).startswith("profile-")
    assert build_profile_id(suite.profile("E")) != build_profile_id(
        replace(
            suite.profile("E"),
            start=replace(suite.profile("E").start, network_start_method=1),
        )
    )
    assert build_pair_id(pair) == build_pair_id(replace(pair, time_budget_seconds=60.0))
    assert build_pair_id(pair) != build_pair_id(replace(pair, solver_seed=43))

    run_a = build_logical_run_id(pair, suite.profile("A"))
    run_e = build_logical_run_id(pair, suite.profile("E"))
    assert run_a.startswith("run-")
    assert run_a != run_e
    assert build_attempt_id(run_a, 1) == f"{run_a}-a0001"
    with pytest.raises(ValueError, match=r"\[1, 9999\]"):
        build_attempt_id(run_a, 0)


def test_attempt_claims_are_exclusive_and_never_overwrite(tmp_path: Path) -> None:
    suite = build_paper_profile_suite()
    logical_run_id = build_logical_run_id(_pair(suite.solver), suite.profile("A"))

    first = claim_attempt_directory(tmp_path, logical_run_id)
    second = claim_attempt_directory(tmp_path, logical_run_id)
    assert first.attempt_number == 1
    assert second.attempt_number == 2
    assert first.directory != second.directory
    assert json.loads(first.claim_file.read_text(encoding="utf-8")) == {
        "attempt_id": first.attempt_id,
        "attempt_number": 1,
        "logical_run_id": logical_run_id,
    }
    with pytest.raises(FileExistsError):
        claim_attempt_directory(tmp_path, logical_run_id, attempt_number=1)
    with pytest.raises(FileExistsError):
        write_canonical_json_exclusive(first.claim_file, {"overwrite": True})


def test_atomic_exclusive_write_never_exposes_a_partial_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "artifact.json"
    real_link = artifact_module.os.link

    def interrupted_publish(_source: object, _target: object) -> None:
        raise KeyboardInterrupt("synthetic interruption before atomic publish")

    monkeypatch.setattr(artifact_module.os, "link", interrupted_publish)
    with pytest.raises(KeyboardInterrupt, match="synthetic interruption"):
        write_canonical_json_exclusive(target, {"complete": True})

    assert not target.exists()
    assert not tuple(tmp_path.glob(".__mlnd_atomic_write__.*"))

    monkeypatch.setattr(artifact_module.os, "link", real_link)
    write_canonical_json_exclusive(target, {"complete": True})
    assert json.loads(target.read_text(encoding="utf-8")) == {"complete": True}


def test_concurrent_attempt_claims_receive_distinct_directories(tmp_path: Path) -> None:
    suite = build_paper_profile_suite()
    logical_run_id = build_logical_run_id(_pair(suite.solver), suite.profile("E"))

    with ThreadPoolExecutor(max_workers=8) as executor:
        claims = tuple(
            executor.map(
                lambda _: claim_attempt_directory(tmp_path, logical_run_id),
                range(8),
            )
        )

    assert {claim.attempt_number for claim in claims} == set(range(1, 9))
    assert len({claim.directory for claim in claims}) == 8
    assert all(claim.claim_file.is_file() for claim in claims)
