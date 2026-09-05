from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from src.benchmark.branch_cut.hashing import canonical_json, canonical_sha256
import src.benchmark.branch_cut.h23_demand1_exact_reuse_batch as batch
import src.benchmark.branch_cut.h23_demand1_exact_reuse_batch_verifier as verifier
from src.benchmark.branch_cut.qualification import (
    QualificationEvidence,
    assess_qualification,
)
from src.benchmark.branch_cut.qualification_incumbent_reuse import (
    QualificationReuseProvenance,
    build_qualification_constraint_signature,
)
from src.benchmark.branch_cut.qualification_runner import (
    CompleteIncumbent,
    QualificationInfrastructureError,
    QualificationModelAudit,
    QualificationModelContract,
    QualificationRunnerOutcome,
)


_HISTORICAL_DEVELOPMENT_ROOT = Path(str(batch._SUPERSEDES["plan_path"])).parent.parent
_VERIFIED_PLAN = _HISTORICAL_DEVELOPMENT_ROOT / (
    "h23_incumbent_reuse_smoke_plan_dba917c_20260831/plan.json"
)
_VERIFIED_RESULTS = (
    _HISTORICAL_DEVELOPMENT_ROOT / "h23_incumbent_reuse_smoke_dba917c_30s_v1"
)


def _classification_payload(
    *,
    candidate_h: int,
    open_h: int,
    captured: int,
    fresh: bool,
    accepted: bool,
) -> dict[str, Any]:
    return {
        "accepted": accepted,
        "captured_variable_count": captured,
        "fresh_model_validation_performed": fresh,
        "rejection_reasons": [] if accepted else ["development_rejection"],
        "witness_status": "TIME_LIMIT",
        "validation_status": "TIME_LIMIT" if fresh else "NOT_RUN",
        "evidence": {
            "candidate_h": candidate_h,
            "open_h": open_h,
            "active_h": min(open_h, 3),
        },
    }


def test_classification_preserves_all_open_and_no_incumbent() -> None:
    all_open = batch.classify_outcome_payload(
        _classification_payload(
            candidate_h=23,
            open_h=23,
            captured=10,
            fresh=True,
            accepted=False,
        )
    )
    no_incumbent = batch.classify_outcome_payload(
        _classification_payload(
            candidate_h=23,
            open_h=0,
            captured=0,
            fresh=False,
            accepted=False,
        )
    )

    assert all_open["all_open"] is True
    assert all_open["interior_open"] is False
    assert all_open["has_incumbent"] is True
    assert no_incumbent["all_closed"] is True
    assert no_incumbent["has_incumbent"] is False
    assert no_incumbent["fresh_validation_required"] is False
    assert no_incumbent["fresh_validation_requirement_satisfied"] is True


def _signature(*, rhs: float):
    return build_qualification_constraint_signature(
        variable_records=(
            {
                "index": 0,
                "name": "x[H001]",
                "vtype": "B",
                "lower_bound": 0.0,
                "upper_bound": 1.0,
            },
        ),
        linear_constraint_records=(
            {
                "index": 0,
                "name": "balance",
                "sense": "=",
                "rhs": rhs,
                "lazy": 0,
                "terms": ((0, 1.0),),
            },
        ),
        quadratic_constraint_records=(),
        sos_constraint_records=(),
    )


def _contract() -> QualificationModelContract:
    return QualificationModelContract(
        qualification_case_sha256="1" * 64,
        effective_model_config_sha256="2" * 64,
        demand_scale=1.0,
        locker_fixed_cost_per_hour=20.0,
        display_origin_choices=8,
        general_corridor_choices=10,
        screening_is_exact=False,
    )


@dataclass
class _ProbeAdapter:
    signature: object
    events: list[str]

    def qualification_model_audit(self) -> QualificationModelAudit:
        self.events.append("model_audit")
        contract = _contract()
        return QualificationModelAudit(
            effective_model_config_sha256=contract.effective_model_config_sha256,
            demand_scale=contract.demand_scale,
            locker_fixed_cost_per_hour=contract.locker_fixed_cost_per_hour,
            display_origin_choices=contract.display_origin_choices,
            general_corridor_choices=contract.general_corridor_choices,
            screening_is_exact=False,
            model_fingerprint=7,
        )

    def qualification_constraint_signature(self):
        self.events.append("constraint_signature")
        return self.signature

    def solve_native(self, *_args: object) -> str:
        raise AssertionError("signature preflight must never optimize")

    def install_constraint_equivalent_complete_incumbent(self, *_args: object) -> None:
        raise AssertionError("signature preflight must never install a start")

    def dispose(self) -> None:
        self.events.append("dispose")


def test_exact_target_preflight_disposes_and_accepts_only_exact_signature() -> None:
    events: list[str] = []
    anchor = _signature(rhs=5.0)
    adapter = _ProbeAdapter(anchor, events)

    probe, comparison = batch.exact_target_signature_preflight(
        object(),
        contract=_contract(),
        anchor_signature=anchor,
        target_demand_scale=1.0,
        adapter_factory=lambda _instance, _mode: adapter,
    )

    assert events == ["model_audit", "constraint_signature", "dispose"]
    assert probe["solve_started"] is False
    assert probe["start_installation_attempted"] is False
    assert probe["adapter_disposed"] is True
    assert comparison["constraint_signature_equal"] is True


def test_exact_target_preflight_fails_before_any_solve_on_signature_mismatch() -> None:
    events: list[str] = []
    adapter = _ProbeAdapter(_signature(rhs=6.0), events)

    with pytest.raises(
        QualificationInfrastructureError,
        match="constraint signature differs from anchor",
    ):
        batch.exact_target_signature_preflight(
            object(),
            contract=_contract(),
            anchor_signature=_signature(rhs=5.0),
            target_demand_scale=1.0,
            adapter_factory=lambda _instance, _mode: adapter,
        )

    assert events == ["model_audit", "constraint_signature", "dispose"]


def _fake_clean_state(commit: str) -> dict[str, Any]:
    return {
        "repository_root": str(batch._REPOSITORY_ROOT),
        "git_commit": commit,
        "status_includes_untracked": True,
        "status_lines": [],
        "worktree_dirty": False,
        "implementation": batch._implementation_payload(batch._REPOSITORY_ROOT),
    }


def _build_plan(tmp_path: Path) -> tuple[Path, str, str]:
    commit = "a" * 40
    plan_root = tmp_path / "plans"
    plan_root.mkdir()
    result = batch.build_h23_demand1_exact_reuse_plan(
        batch.H23Demand1ExactReusePlanBuildConfig(
            verified_smoke_plan_path=_VERIFIED_PLAN,
            verified_smoke_results_root=_VERIFIED_RESULTS,
            output_path=plan_root / "plan.json",
            expected_git_commit=commit,
        ),
        hooks=batch.H23Demand1ExactReusePlanBuildHooks(
            capture_repository_state=lambda _root: _fake_clean_state(commit)
        ),
    )
    return result.path, commit, result.implementation_sha256


def test_plan_freezes_post_hoc_order_and_historical_failure_states(
    tmp_path: Path,
) -> None:
    path, _commit, _implementation = _build_plan(tmp_path)
    loaded = batch.load_h23_demand1_exact_reuse_plan(path)

    assert [action["original_profile_order_index"] for action in loaded.actions] == [
        4,
        0,
        1,
        2,
        3,
        5,
    ]
    assert loaded.payload["native_settings"]["threads"] == 1
    assert loaded.payload["native_settings"]["time_limit_seconds"] == 30.0
    assert loaded.payload["geometry_start_policy"]["per_attempt_seconds"] == 10.0
    assert loaded.payload["order_bias"]["post_hoc_anchor_first"] is True
    assert loaded.payload["supersedes"] == batch._SUPERSEDES
    assert Path(str(batch._SUPERSEDES["plan_path"])).is_file()
    assert not Path(str(batch._SUPERSEDES["unused_run_root"])).exists()
    assert loaded.payload["eligible_for_formal_pool"] is False
    assert loaded.payload["may_lock_profiles"] is False
    history = [
        action["historical_checkpoint"]["classification"] for action in loaded.actions
    ]
    assert history[0]["accepted"] is True
    assert history[0]["interior_open"] is True
    assert all(item["has_incumbent"] is False for item in history[1:])
    assert all(item["all_closed"] is True for item in history[1:])
    assert all(item["all_open"] is False for item in history[1:])


def test_independent_plan_verifier_recomputes_current_files_and_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, commit, implementation = _build_plan(tmp_path)

    def fake_git(_root: Path, *arguments: str) -> str:
        if arguments == ("rev-parse", "HEAD"):
            return commit
        if arguments == ("status", "--porcelain", "--untracked-files=all"):
            return ""
        raise AssertionError(arguments)

    monkeypatch.setattr(verifier, "_git", fake_git)
    report = verifier.verify_h23_demand1_exact_reuse_plan(
        verifier.H23Demand1ExactReusePlanVerificationRequest(
            plan_path=path,
            expected_git_commit=commit,
            expected_implementation_sha256=implementation,
        )
    )

    assert report.status == "verified_plan"
    assert report.execution_order_original_indices == (4, 0, 1, 2, 3, 5)


def test_independent_plan_verifier_rejects_reordered_self_hashed_actions(
    tmp_path: Path,
) -> None:
    path, _commit, _implementation = _build_plan(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["actions"][0], payload["actions"][1] = (
        payload["actions"][1],
        payload["actions"][0],
    )
    entry = {key: value for key, value in payload.items() if key != "plan_sha256"}
    payload["plan_sha256"] = canonical_sha256(entry)
    corrupted = tmp_path / "reordered.json"
    corrupted.write_text(canonical_json(payload) + "\n", encoding="utf-8")

    with pytest.raises(
        verifier.H23Demand1ExactReuseVerificationError,
        match="identity/order drifted",
    ):
        verifier._verify_plan(corrupted)


def _fake_qualifier(instance: object, mode: str, **kwargs: Any):
    del instance
    assert mode == "MX"
    pool = kwargs["development_incumbent_reuse_pool"]
    contract: QualificationModelContract = kwargs["expected_model_contract"]
    profile_id = kwargs["development_reuse_profile_id"]
    reject_all_open = "feeder-030-basefc-028" in profile_id
    reject_no_incumbent = "feeder-030-basefc-024" in profile_id
    signature = _signature(rhs=5.0)
    expected_actual_signature = kwargs[
        "development_expected_target_constraint_signature"
    ]
    if profile_id == (
        "devcal-demand-1x-feeder-035-basefc-024-fc-024-"
        "display-008-general-010-exact-false"
    ):
        assert expected_actual_signature is None
    else:
        assert expected_actual_signature == signature
    provenance = QualificationReuseProvenance(
        qualification_run_id=kwargs["qualification_run_id"],
        case_id=kwargs["case_id"],
        profile_id=profile_id,
        scale=kwargs["scale"],
        mode="MX",
        physical_instance_sha256=kwargs["instance_sha256"],
        qualification_case_sha256=contract.qualification_case_sha256,
        effective_model_config_sha256=contract.effective_model_config_sha256,
        demand_scale=contract.demand_scale,
        screening_is_exact=False,
    )
    selected = pool.select(
        target_provenance=provenance,
        target_constraint_signature=signature,
    )
    if selected is not None:
        pool.record_installation(
            target_provenance=provenance,
            target_constraint_signature=signature,
            selected=selected,
        )
    incumbent = CompleteIncumbent.from_ordered_values(
        objective=900.0,
        variable_names=("x[H001]",),
        values=(1.0,),
        model_fingerprint=int(contract.locker_fixed_cost_per_hour * 100),
    )
    audit = QualificationModelAudit(
        effective_model_config_sha256=contract.effective_model_config_sha256,
        demand_scale=contract.demand_scale,
        locker_fixed_cost_per_hour=contract.locker_fixed_cost_per_hour,
        display_origin_choices=contract.display_origin_choices,
        general_corridor_choices=contract.general_corridor_choices,
        screening_is_exact=False,
        model_fingerprint=int(contract.locker_fixed_cost_per_hour * 100),
    )
    evidence = QualificationEvidence(
        case_id=kwargs["case_id"],
        scale="small",
        courier_only_objective=1_000.0,
        witness_objective=float("nan") if reject_no_incumbent else 900.0,
        total_demand_per_hour=100.0,
        witness_service_per_hour=(float("nan") if reject_no_incumbent else 20.0),
        open_facilities=(0 if reject_no_incumbent else 23 if reject_all_open else 10),
        positive_cost_open_facilities=(
            0 if reject_no_incumbent else 23 if reject_all_open else 10
        ),
        candidate_h=23,
        open_h=0 if reject_no_incumbent else 23 if reject_all_open else 8,
        witness_feasible=not reject_no_incumbent,
        qualification_run_id=kwargs["qualification_run_id"],
        instance_sha256=kwargs["instance_sha256"],
        active_h=0 if reject_no_incumbent else 23 if reject_all_open else 8,
        active_h_threshold_per_hour=0.001,
        qualification_case_sha256=contract.qualification_case_sha256,
        effective_model_config_sha256=contract.effective_model_config_sha256,
        screening_is_exact=False,
    )
    decision = assess_qualification(evidence, kwargs["policy"])
    if decision.accepted:
        pool.offer(
            provenance=provenance,
            constraint_signature=signature,
            incumbent=incumbent,
            accepted=True,
            fresh_validation_performed=True,
        )
    captured = not reject_no_incumbent
    return QualificationRunnerOutcome(
        accepted=decision.accepted,
        rejection_reasons=decision.rejection_reasons,
        evidence=evidence,
        decision=decision,
        required_service_per_hour=0.1,
        required_objective_upper_bound=999.0,
        temporary_constraint_used=False,
        temporary_constraint_removed=None,
        fresh_model_validation_performed=captured,
        witness_status="TIME_LIMIT",
        validation_status="TIME_LIMIT" if captured else "NOT_RUN",
        captured_variable_count=1 if captured else 0,
        exception_messages=(),
        expected_model_contract=contract,
        witness_model_audit=audit,
        fresh_model_audit=audit if captured else None,
        geometry_start_policy=kwargs["geometry_start_policy"],
    )


def _run_fake_batch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    qualifier: Any = _fake_qualifier,
    expect_infrastructure_failure: bool = False,
    process_table: Any | None = None,
) -> tuple[Any, Path, str, str]:
    plan_path, commit, implementation = _build_plan(tmp_path)
    loaded = batch.load_h23_demand1_exact_reuse_plan(plan_path)
    contracts = {
        (
            float(
                action["source_job"]["effective_model_config"][
                    "feeder_rider_cost_factor"
                ]
            ),
            float(
                action["source_job"]["effective_model_config"][
                    "locker_fixed_cost_per_hour"
                ]
            ),
        ): QualificationModelContract(
            **action["source_job"]["qualification_model_contract"]
        )
        for action in loaded.actions
    }
    repository_state = _fake_clean_state(commit)
    environment = {
        "repository": repository_state,
        "python": {
            "implementation": "CPython",
            "version": "test",
            "executable": "test",
        },
        "platform": {
            "system": "test",
            "release": "test",
            "machine": "test",
            "processor": "test",
        },
        "gurobi_version": "not-imported-test-double",
    }
    monkeypatch.setattr(
        batch,
        "canonical_instance_hash",
        lambda _path: (batch.H23_REUSE_SMOKE_PHYSICAL_INSTANCE_SHA256, {}),
    )

    def load_instance(_path: Path, config: object, demand_scale: float):
        key = (
            float(config.feeder_rider_cost_factor),
            float(config.locker_fixed_cost_per_hour),
        )
        return SimpleNamespace(
            config=config,
            metadata={"demand_scale": demand_scale},
            contract=contracts[key],
            screening_is_exact=lambda _mode: False,
        )

    def adapter_factory(instance: object, _mode: str):
        events: list[str] = []
        adapter = _ProbeAdapter(_signature(rhs=5.0), events)
        adapter.qualification_model_audit = lambda: QualificationModelAudit(
            effective_model_config_sha256=(
                instance.contract.effective_model_config_sha256
            ),
            demand_scale=instance.contract.demand_scale,
            locker_fixed_cost_per_hour=(instance.contract.locker_fixed_cost_per_hour),
            display_origin_choices=instance.contract.display_origin_choices,
            general_corridor_choices=instance.contract.general_corridor_choices,
            screening_is_exact=False,
            model_fingerprint=int(
                instance.contract.locker_fixed_cost_per_hour * 100
            ),
        )
        return adapter

    results_root = tmp_path / "results"
    run_config = batch.H23Demand1ExactReuseBatchRunConfig(
        plan_path=plan_path,
        results_root=results_root,
        resource_lock=tmp_path / "resource.lock",
    )
    hooks = batch.H23Demand1ExactReuseBatchHooks(
        load_instance=load_instance,
        qualify_instance=qualifier,
        adapter_factory=adapter_factory,
        capture_environment=lambda _root: environment,
        process_table=process_table or (lambda: ""),
        now_utc=lambda: "2026-08-31T12:00:00Z",
    )
    try:
        result = batch.run_h23_demand1_exact_reuse_batch(run_config, hooks=hooks)
    except batch.H23Demand1ExactReuseBatchInfrastructureError:
        if not expect_infrastructure_failure:
            raise
        identity = json.loads(
            (results_root / "run_identity.json").read_text(encoding="utf-8")
        )
        result = SimpleNamespace(
            results_root=results_root,
            batch_run_id=identity["batch_run_id"],
            conclusion="infrastructure_failed",
        )
    else:
        if expect_infrastructure_failure:
            raise AssertionError("fake batch unexpectedly completed")

    def fake_git(_root: Path, *arguments: str) -> str:
        return commit if arguments == ("rev-parse", "HEAD") else ""

    monkeypatch.setattr(verifier, "_git", fake_git)
    return result, plan_path, commit, implementation


def _verify_fake_batch(
    result: Any,
    *,
    plan_path: Path,
    commit: str,
    implementation: str,
) -> verifier.H23Demand1ExactReuseVerificationReport:
    return verifier.verify_h23_demand1_exact_reuse_results(
        verifier.H23Demand1ExactReuseVerificationRequest(
            plan_path=plan_path,
            results_root=result.results_root,
            expected_run_id=result.batch_run_id,
            expected_git_commit=commit,
            expected_implementation_sha256=implementation,
        )
    )


def test_full_fake_batch_and_independent_result_verifier(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result, plan_path, commit, implementation = _run_fake_batch(
        tmp_path, monkeypatch
    )
    report = _verify_fake_batch(
        result,
        plan_path=plan_path,
        commit=commit,
        implementation=implementation,
    )

    assert result.conclusion == "completed"
    assert report.status == "verified"
    assert report.attempted_action_count == 6
    assert report.accepted_count == 4
    assert report.no_incumbent_count == 1
    assert report.all_open_count == 1


def _write_self_hashed(path: Path, payload: dict[str, Any], hash_field: str) -> str:
    payload.pop(hash_field, None)
    payload[hash_field] = canonical_sha256(payload)
    serialized = canonical_json(payload) + "\n"
    path.write_text(serialized, encoding="utf-8")
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _mutate_completed_checkpoint(
    results_root: Path,
    *,
    position: int,
    mutation: Any,
) -> None:
    summary_path = results_root / "batch_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    record = summary["checkpoints"][position]
    checkpoint_path = Path(record["path"])
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    mutation(checkpoint["result"])
    file_sha = _write_self_hashed(
        checkpoint_path, checkpoint, "checkpoint_sha256"
    )
    record["checkpoint_sha256"] = checkpoint["checkpoint_sha256"]
    record["file_sha256"] = file_sha
    _write_self_hashed(summary_path, summary, "summary_sha256")


@pytest.mark.parametrize("mutation_kind", ["fresh_fingerprint", "exception_message"])
def test_result_verifier_rejects_semantic_mutation_after_all_hashes_recompute(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation_kind: str,
) -> None:
    result, plan_path, commit, implementation = _run_fake_batch(
        tmp_path, monkeypatch
    )

    def mutate(payload: dict[str, Any]) -> None:
        outcome = payload["qualification_outcome"]
        if mutation_kind == "fresh_fingerprint":
            outcome["fresh_model_audit"]["model_fingerprint"] += 1
        else:
            outcome["exception_messages"] = [
                "fresh_validation_solve: RuntimeError: synthetic"
            ]

    _mutate_completed_checkpoint(
        result.results_root,
        position=0,
        mutation=mutate,
    )
    with pytest.raises(verifier.H23Demand1ExactReuseVerificationError):
        _verify_fake_batch(
            result,
            plan_path=plan_path,
            commit=commit,
            implementation=implementation,
        )


@pytest.mark.parametrize("mutation_kind", ["summary_error", "checkpoint_action"])
def test_aborted_package_is_verified_then_rejects_rehashed_failure_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation_kind: str,
) -> None:
    calls: list[str] = []

    def failing_qualifier(instance: object, mode: str, **kwargs: Any):
        calls.append(kwargs["development_reuse_profile_id"])
        if len(calls) == 2:
            raise QualificationInfrastructureError("synthetic solver exception")
        return _fake_qualifier(instance, mode, **kwargs)

    result, plan_path, commit, implementation = _run_fake_batch(
        tmp_path,
        monkeypatch,
        qualifier=failing_qualifier,
        expect_infrastructure_failure=True,
    )
    report = _verify_fake_batch(
        result,
        plan_path=plan_path,
        commit=commit,
        implementation=implementation,
    )
    assert report.conclusion == "infrastructure_failed"
    assert len(calls) == 2

    summary_path = result.results_root / "batch_aborted.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if mutation_kind == "summary_error":
        summary["infrastructure_failure"]["message"] = "forged failure"
    else:
        record = summary["checkpoints"][-1]
        checkpoint_path = Path(record["path"])
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        checkpoint["result"]["infrastructure_error"]["action_index"] = 99
        record["file_sha256"] = _write_self_hashed(
            checkpoint_path, checkpoint, "checkpoint_sha256"
        )
        record["checkpoint_sha256"] = checkpoint["checkpoint_sha256"]
    _write_self_hashed(summary_path, summary, "summary_sha256")

    with pytest.raises(verifier.H23Demand1ExactReuseVerificationError):
        _verify_fake_batch(
            result,
            plan_path=plan_path,
            commit=commit,
            implementation=implementation,
        )


def test_process_scan_covers_known_h6_h23_modules_as_secondary_gate() -> None:
    markers = (
        "h6_directed_return_ablation",
        "h6_facility_incident_star_ablation",
        "h6_od_disaggregated_return_ablation",
        "h23_qualification_diagnostic",
        "h23_incumbent_reuse_smoke",
        "h23_demand1_exact_reuse_batch",
    )
    table = "\n".join(
        f"{10 + index} 1 /opt/anaconda3/bin/python -m "
        f"src.benchmark.branch_cut.{marker} run"
        for index, marker in enumerate(markers)
    )
    audit = batch._process_audit(table)

    assert audit["valid"] is False
    assert len(audit["conflicting_processes"]) == len(markers)
    assert batch._EXECUTION_CONTRACT[
        "shared_resource_lock_is_primary_concurrency_gate"
    ] is True
    assert batch._EXECUTION_CONTRACT[
        "process_scan_is_secondary_concurrency_gate"
    ] is True
    assert tuple(batch._EXECUTION_CONTRACT["process_conflict_markers"]) == (
        batch._PROCESS_CONFLICT_MARKERS
    )


def test_per_action_process_conflict_writes_verifiable_abort_and_stops(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tables = iter(
        (
            "",
            "",
            "77 1 /opt/anaconda3/bin/python -m "
            "src.benchmark.branch_cut.h6_od_disaggregated_return_ablation run",
        )
    )
    calls: list[str] = []

    def tracking_qualifier(instance: object, mode: str, **kwargs: Any):
        calls.append(kwargs["development_reuse_profile_id"])
        return _fake_qualifier(instance, mode, **kwargs)

    result, plan_path, commit, implementation = _run_fake_batch(
        tmp_path,
        monkeypatch,
        qualifier=tracking_qualifier,
        expect_infrastructure_failure=True,
        process_table=lambda: next(tables),
    )
    report = _verify_fake_batch(
        result,
        plan_path=plan_path,
        commit=commit,
        implementation=implementation,
    )
    summary = json.loads(
        (result.results_root / "batch_aborted.json").read_text(encoding="utf-8")
    )

    assert report.conclusion == "infrastructure_failed"
    assert len(calls) == 1
    assert summary["infrastructure_failure"]["stage"] == "process_conflict_check"
    assert summary["infrastructure_failure"]["action_index"] == 1


def test_plan_builder_rejects_dirty_repository_before_publishing(
    tmp_path: Path,
) -> None:
    commit = "a" * 40
    dirty = _fake_clean_state(commit)
    dirty["worktree_dirty"] = True
    dirty["status_lines"] = ["?? unexpected"]

    with pytest.raises(
        batch.H23Demand1ExactReuseBatchError,
        match="exact clean expected commit",
    ):
        batch.build_h23_demand1_exact_reuse_plan(
            batch.H23Demand1ExactReusePlanBuildConfig(
                verified_smoke_plan_path=_VERIFIED_PLAN,
                verified_smoke_results_root=_VERIFIED_RESULTS,
                output_path=tmp_path / "plan.json",
                expected_git_commit=commit,
            ),
            hooks=batch.H23Demand1ExactReusePlanBuildHooks(
                capture_repository_state=lambda _root: dirty
            ),
        )

    assert not (tmp_path / "plan.json").exists()


def test_v2_plan_and_results_cannot_create_superseded_v1_unused_root(
    tmp_path: Path,
) -> None:
    old_root = Path(str(batch._SUPERSEDES["unused_run_root"]))
    assert not old_root.exists()
    commit = "a" * 40

    with pytest.raises(
        batch.H23Demand1ExactReuseBatchError,
        match="plan output overlaps the preserved superseded v1 run root",
    ):
        batch.build_h23_demand1_exact_reuse_plan(
            batch.H23Demand1ExactReusePlanBuildConfig(
                verified_smoke_plan_path=_VERIFIED_PLAN,
                verified_smoke_results_root=_VERIFIED_RESULTS,
                output_path=old_root / "nested" / "plan.json",
                expected_git_commit=commit,
            ),
            hooks=batch.H23Demand1ExactReusePlanBuildHooks(
                capture_repository_state=lambda _root: _fake_clean_state(commit)
            ),
        )

    plan_path, _commit, _implementation = _build_plan(tmp_path)
    loaded = batch.load_h23_demand1_exact_reuse_plan(plan_path)
    with pytest.raises(
        batch.H23Demand1ExactReuseBatchError,
        match="results root overlaps the preserved superseded v1 run root",
    ):
        batch._claim_results_root(old_root / "nested-results", loaded)

    assert not old_root.exists()
