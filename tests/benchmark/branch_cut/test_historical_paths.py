from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from src.benchmark.branch_cut.hashing import canonical_json, canonical_sha256
from src.benchmark.branch_cut.historical_paths import preserved_development_root
import src.benchmark.branch_cut.h23_demand1_exact_reuse_batch as batch
import src.benchmark.branch_cut.h23_demand1_exact_reuse_batch_verifier as verifier


@pytest.mark.parametrize("layout", ("original", "integrated"))
def test_preserved_plan_relocation_keeps_both_independent_hash_gates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, layout: str
) -> None:
    root = tmp_path / "repository"
    source = root
    if layout == "integrated":
        source = (
            root
            / "results/branch_cut_paper/worktrees/bnc_incumbent_reuse_dev_20260831"
        )
    development_root = source / "results/branch_cut_paper/dev"
    relative_plan = Path("superseded/plan.json")
    plan = development_root / relative_plan
    plan.parent.mkdir(parents=True)
    payload = {
        "plan_id": "temporary-superseded-plan",
        "required_execution_environment": {
            "git_commit": "1" * 40,
            "implementation": {"implementation_sha256": "2" * 64},
        },
    }
    payload["plan_sha256"] = canonical_sha256(payload)
    raw = (canonical_json(payload) + "\n").encode()
    plan.write_bytes(raw)

    located = preserved_development_root(root, required_relative_path=relative_plan)
    assert located == development_root
    binding = {
        "plan_id": payload["plan_id"],
        "git_commit": "1" * 40,
        "plan_path": str(located / relative_plan),
        "plan_sha256": payload["plan_sha256"],
        "plan_file_sha256": hashlib.sha256(raw).hexdigest(),
        "implementation_sha256": "2" * 64,
        "unused_run_root": str(located / "never-created"),
    }
    monkeypatch.setattr(batch, "_SUPERSEDES", binding)
    monkeypatch.setattr(verifier, "_SUPERSEDES", binding)
    batch._validate_superseded_v1_evidence()
    verifier._verify_superseded_v1_evidence()

    # Moving the source must not turn path discovery into trust in its contents.
    changed = dict(payload)
    changed["unexpected_change"] = True
    plan.write_text(canonical_json(changed) + "\n")
    with pytest.raises(batch.H23Demand1ExactReuseBatchError, match="binding drifted"):
        batch._validate_superseded_v1_evidence()
    with pytest.raises(
        verifier.H23Demand1ExactReuseVerificationError, match="evidence drifted"
    ):
        verifier._verify_superseded_v1_evidence()


def test_explicit_history_source_is_authoritative_even_when_missing(tmp_path: Path) -> None:
    root = tmp_path / "repository"
    relative_plan = Path("superseded/plan.json")
    local = root / "results/branch_cut_paper/dev"
    (local / relative_plan).parent.mkdir(parents=True)
    (local / relative_plan).write_text("local evidence")
    explicit = tmp_path / "explicit-source"
    assert preserved_development_root(
        root,
        required_relative_path=relative_plan,
        source_repository_root=explicit,
    ) == explicit / "results/branch_cut_paper/dev"


def test_local_history_is_never_silently_replaced_by_a_saved_copy(tmp_path: Path) -> None:
    root = tmp_path / "repository"
    relative_plan = Path("superseded/plan.json")
    local = root / "results/branch_cut_paper/dev"
    saved = (
        root
        / "results/branch_cut_paper/worktrees/bnc_incumbent_reuse_dev_20260831"
        / "results/branch_cut_paper/dev"
    )
    for directory in (local, saved):
        (directory / relative_plan).parent.mkdir(parents=True)
        (directory / relative_plan).write_text("contents still require validation")
    assert preserved_development_root(root, required_relative_path=relative_plan) == local


def test_v1_diagnostic_is_readable_only_by_explicit_identity_and_cannot_run_as_v2(
    tmp_path: Path,
) -> None:
    import test_h23_qualification_diagnostic as fixtures
    from src.benchmark.branch_cut import h23_qualification_diagnostic as diagnostic
    from src.benchmark.branch_cut import h23_incumbent_reuse_contract as legacy

    stack, manifest, _verifier_manifest, _source, bindings = (
        fixtures._dependency_patches(tmp_path)
    )
    with stack:
        implementation = fixtures._fake_implementation()
        result = diagnostic.build_h23_diagnostic_plan(
            diagnostic.H23PlanBuildConfig(
                manifest_path=manifest.path,
                source_run_root=tmp_path / "source-run",
                output_path=tmp_path / "plan.json",
                plan_id=diagnostic.H23_DIAGNOSTIC_PLAN_ID,
                max_jobs=24,
                expected_git_commit="a" * 40,
            ),
            hooks=diagnostic.H23PlanBuildHooks(
                capture_repository_state=lambda _root: {
                    "repository_root": str(tmp_path),
                    "git_commit": "a" * 40,
                    "status_includes_untracked": True,
                    "status_lines": [],
                    "worktree_dirty": False,
                    "implementation": implementation,
                }
            ),
        )
        payload = json.loads(result.path.read_text())
        assert payload["native_settings"]["time_limit_seconds"] == 60.0
        assert payload["geometry_start_policy"]["per_attempt_seconds"] == 30.0
        payload.update(
            plan_id=legacy.H23_DIAGNOSTIC_PLAN_ID,
            plan_schema="mlnd-h23-qualification-diagnostic-plan-v1",
            native_settings=legacy.REGISTERED_NATIVE_SETTINGS_PAYLOAD,
            geometry_start_policy=legacy.REGISTERED_GEOMETRY_POLICY_PAYLOAD,
            jobs=diagnostic._diagnostic_jobs(
                manifest, bindings, plan_id=legacy.H23_DIAGNOSTIC_PLAN_ID
            ),
        )
        fixtures._rehash_plan(payload)
        result.path.write_text(canonical_json(payload) + "\n")
        loaded = diagnostic.load_h23_diagnostic_plan(
            result.path,
            expected_plan_id=legacy.H23_DIAGNOSTIC_PLAN_ID,
            expected_max_jobs=24,
        )
        assert loaded.payload["native_settings"]["time_limit_seconds"] == 30.0
        assert loaded.payload["geometry_start_policy"]["per_attempt_seconds"] == 10.0
        with pytest.raises(diagnostic.H23DiagnosticError, match="fixed flags"):
            diagnostic.load_h23_diagnostic_plan(
                result.path,
                expected_plan_id=diagnostic.H23_DIAGNOSTIC_PLAN_ID,
                expected_max_jobs=24,
            )
        with pytest.raises(ValueError, match="plan_id must be"):
            diagnostic.H23DiagnosticRunConfig(
                plan_path=result.path,
                results_root=tmp_path / "cannot-run-legacy",
                plan_id=legacy.H23_DIAGNOSTIC_PLAN_ID,
                max_jobs=24,
            )
