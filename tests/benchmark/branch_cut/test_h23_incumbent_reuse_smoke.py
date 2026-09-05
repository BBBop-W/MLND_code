from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from src.config import ModelConfig
from src.benchmark.branch_cut.hashing import canonical_json, canonical_sha256
from src.benchmark.branch_cut.h23_qualification_diagnostic import (
    LoadedH23DiagnosticPlan,
)
import src.benchmark.branch_cut.h23_incumbent_reuse_smoke as smoke
import src.benchmark.branch_cut.h23_incumbent_reuse_smoke_verifier as smoke_verifier
from src.benchmark.branch_cut.h23_incumbent_reuse_smoke_verifier import (
    H23IncumbentReuseSmokePlanVerificationRequest,
    H23IncumbentReuseSmokeVerificationError,
    H23IncumbentReuseSmokeVerificationRequest,
    verify_h23_incumbent_reuse_smoke_plan,
    verify_h23_incumbent_reuse_smoke_results,
)
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
    QualificationModelAudit,
    QualificationModelContract,
    QualificationRunnerOutcome,
)


_EXPECTED = smoke._EXPECTED_PROFILE_IDENTITIES
_INSTANCE_SHA = smoke.H23_REUSE_SMOKE_PHYSICAL_INSTANCE_SHA256


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_canonical(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(canonical_json(payload) + "\n", encoding="utf-8")


def _signature(*, rhs: float = 5.0):
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
                "name": "demand_balance",
                "sense": "=",
                "rhs": rhs,
                "lazy": 0,
                "terms": ((0, 1.0),),
            },
        ),
        quadratic_constraint_records=(),
        sos_constraint_records=(),
    )


@dataclass
class _ProbeAdapter:
    audit: QualificationModelAudit
    signature: object
    events: list[str]
    signature_error: BaseException | None = None

    def qualification_model_audit(self) -> QualificationModelAudit:
        self.events.append("model_audit")
        return self.audit

    def qualification_constraint_signature(self):
        self.events.append("constraint_signature")
        if self.signature_error is not None:
            raise self.signature_error
        return self.signature

    def solve_native(self, *_args: object) -> str:
        raise AssertionError("signature-only probe must never optimize")

    def install_constraint_equivalent_complete_incumbent(self, *_args: object) -> None:
        raise AssertionError("signature-only probe must never install a start")

    def dispose(self) -> None:
        self.events.append("dispose")


def _contract(index: int) -> QualificationModelContract:
    identity = _EXPECTED[index]
    return QualificationModelContract(
        qualification_case_sha256=identity["qualification_case_sha256"],
        effective_model_config_sha256=identity["effective_model_config_sha256"],
        demand_scale=identity["demand_scale"],
        locker_fixed_cost_per_hour=identity["locker_fixed_cost_per_hour"],
        display_origin_choices=8,
        general_corridor_choices=10,
        screening_is_exact=False,
    )


def _audit(index: int, *, fingerprint: int) -> QualificationModelAudit:
    contract = _contract(index)
    return QualificationModelAudit(
        effective_model_config_sha256=contract.effective_model_config_sha256,
        demand_scale=contract.demand_scale,
        locker_fixed_cost_per_hour=contract.locker_fixed_cost_per_hour,
        display_origin_choices=contract.display_origin_choices,
        general_corridor_choices=contract.general_corridor_choices,
        screening_is_exact=False,
        model_fingerprint=fingerprint,
    )


def test_signature_only_probe_disposes_without_optimize_or_start() -> None:
    events: list[str] = []
    adapter = _ProbeAdapter(_audit(10, fingerprint=10), _signature(rhs=10.0), events)

    outcome = smoke.probe_constraint_signature_only(
        object(),
        "MX",
        expected_model_contract=_contract(10),
        adapter_factory=lambda _instance, _mode: adapter,
    )

    assert events == ["model_audit", "constraint_signature", "dispose"]
    assert outcome.solve_started is False
    assert outcome.start_installation_attempted is False
    assert outcome.adapter_disposed is True


def test_signature_only_probe_disposes_on_signature_failure() -> None:
    events: list[str] = []
    adapter = _ProbeAdapter(
        _audit(10, fingerprint=10),
        _signature(rhs=10.0),
        events,
        signature_error=RuntimeError("signature failed"),
    )

    with pytest.raises(RuntimeError, match="signature failed"):
        smoke.probe_constraint_signature_only(
            object(),
            "MX",
            expected_model_contract=_contract(10),
            adapter_factory=lambda _instance, _mode: adapter,
        )

    assert events == ["model_audit", "constraint_signature", "dispose"]


def test_signature_comparison_requires_full_exact_relation() -> None:
    source = _signature(rhs=5.0)
    positive = smoke.build_signature_comparison(
        role="objective_only_positive_target",
        expected_relation="equal",
        source=source,
        target=_signature(rhs=5.0),
        source_demand_scale=1.0,
        target_demand_scale=1.0,
    )
    negative = smoke.build_signature_comparison(
        role="constraint_changing_negative_probe",
        expected_relation="different",
        source=source,
        target=_signature(rhs=10.0),
        source_demand_scale=1.0,
        target_demand_scale=2.0,
    )

    assert positive["constraint_signature_equal"] is True
    assert positive["comparison_passed"] is True
    assert negative["constraint_signature_equal"] is False
    assert negative["demand_scale_equal"] is False
    assert negative["comparison_passed"] is True


@dataclass(frozen=True)
class _Profile:
    profile_id: str
    demand_scale: float
    locker_fixed_cost_per_hour: float
    config: ModelConfig

    def solve_profile(self) -> _Profile:
        return self

    def effective_model_config(self) -> ModelConfig:
        return self.config


def _fake_h23_inputs(
    tmp_path: Path,
) -> tuple[
    LoadedH23DiagnosticPlan,
    dict[str, Any],
    Path,
    Path,
]:
    source_root = tmp_path / "frozen_source"
    source_root.mkdir()
    source_plan_path = source_root / "source_h23_plan.json"
    source_summary_path = source_root / "source_h23_summary.json"
    _write_canonical(source_plan_path, {"fixture": "source-plan"})
    _write_canonical(source_summary_path, {"fixture": "source-summary"})
    instance_dir = tmp_path / "instance"
    instance_dir.mkdir()
    profiles: list[object] = [None] * 24
    jobs: list[dict[str, Any]] = [dict() for _ in range(24)]
    ledger: list[dict[str, Any]] = []
    for index in smoke.H23_REUSE_SMOKE_ACTION_INDICES:
        identity = _EXPECTED[index]
        config = replace(
            ModelConfig(),
            locker_fixed_cost_per_hour=identity["locker_fixed_cost_per_hour"],
            feeder_rider_cost_factor=0.35,
            display_origin_choices=8,
            general_corridor_choices=10,
        )
        profile = _Profile(
            profile_id=identity["profile_id"],
            demand_scale=identity["demand_scale"],
            locker_fixed_cost_per_hour=identity["locker_fixed_cost_per_hour"],
            config=config,
        )
        profiles[index] = profile
        job_id = f"source-calibration-job-{index:02d}"
        jobs[index] = {
            "diagnostic_job_order_index": index,
            "diagnostic_job_id": identity["diagnostic_job_id"],
            "source_job": {
                "job_id": job_id,
                "job_order_index": index,
                "profile_order_index": index,
                "profile_id": identity["profile_id"],
                "instance_sha256": _INSTANCE_SHA,
                "physical_instance_id": (smoke.H23_REUSE_SMOKE_PHYSICAL_INSTANCE_ID),
                "scale": "small",
                "qualification_case_sha256": identity["qualification_case_sha256"],
                "effective_model_config_sha256": identity[
                    "effective_model_config_sha256"
                ],
            },
            "physical_instance_directory": str(instance_dir),
            "qualification_solve_profile": {
                "profile_id": identity["profile_id"],
                "demand_scale": identity["demand_scale"],
                "locker_fixed_cost_per_hour": identity["locker_fixed_cost_per_hour"],
                "display_origin_choices": 8,
                "general_corridor_choices": 10,
                "expected_screening_is_exact": False,
            },
            "effective_model_config": {},
            "qualification_model_contract": _contract(index).to_payload(),
        }
        historical_path = tmp_path / f"historical-{index}.json"
        _write_canonical(historical_path, {"index": index})
        ledger.append(
            {
                "diagnostic_job_order_index": index,
                "checkpoint_path": str(historical_path),
                "checkpoint_sha256": hashlib.sha256(
                    f"checkpoint-{index}".encode()
                ).hexdigest(),
                "accepted": index == 4,
            }
        )
    plan = LoadedH23DiagnosticPlan(
        path=source_plan_path,
        file_sha256=_file_sha(source_plan_path),
        plan_sha256=smoke.H23_REUSE_SMOKE_SOURCE_PLAN_SHA256,
        payload={},
        manifest=SimpleNamespace(profiles=profiles),
        jobs=tuple(jobs),
    )
    summary = {"diagnostic_job_ledger": ledger}
    return plan, summary, source_plan_path, source_summary_path


def _implementation_state(commit: str) -> tuple[dict[str, Any], str]:
    files = {
        relative: _file_sha(smoke._REPOSITORY_ROOT / relative)
        for relative in smoke._IMPLEMENTATION_FILES
    }
    implementation_sha = canonical_sha256(files)
    return (
        {
            "repository_root": str(smoke._REPOSITORY_ROOT.resolve()),
            "git_commit": commit,
            "status_includes_untracked": True,
            "status_lines": [],
            "worktree_dirty": False,
            "implementation": {
                "ordered_file_sha256": files,
                "implementation_sha256": implementation_sha,
            },
        },
        implementation_sha,
    )


def _fake_qualifier(instance: object, mode: str, **kwargs: Any):
    del instance
    assert mode == "MX"
    pool = kwargs["development_incumbent_reuse_pool"]
    contract: QualificationModelContract = kwargs["expected_model_contract"]
    profile_id = kwargs["development_reuse_profile_id"]
    signature = _signature(rhs=5.0)
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
    pool.offer(
        provenance=provenance,
        constraint_signature=signature,
        incumbent=incumbent,
        accepted=True,
        fresh_validation_performed=True,
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
        witness_objective=900.0,
        total_demand_per_hour=100.0,
        witness_service_per_hour=20.0,
        open_facilities=10,
        positive_cost_open_facilities=10,
        candidate_h=20,
        open_h=8,
        witness_feasible=True,
        qualification_run_id=kwargs["qualification_run_id"],
        instance_sha256=kwargs["instance_sha256"],
        active_h=8,
        active_h_threshold_per_hour=0.001,
        qualification_case_sha256=contract.qualification_case_sha256,
        effective_model_config_sha256=contract.effective_model_config_sha256,
        screening_is_exact=False,
    )
    decision = assess_qualification(evidence, kwargs["policy"])
    return QualificationRunnerOutcome(
        accepted=True,
        rejection_reasons=(),
        evidence=evidence,
        decision=decision,
        required_service_per_hour=0.1,
        required_objective_upper_bound=999.0,
        temporary_constraint_used=False,
        temporary_constraint_removed=None,
        fresh_model_validation_performed=True,
        witness_status="TIME_LIMIT",
        validation_status="TIME_LIMIT",
        captured_variable_count=1,
        exception_messages=(),
        expected_model_contract=contract,
        witness_model_audit=audit,
        fresh_model_audit=audit,
        geometry_start_policy=kwargs["geometry_start_policy"],
    )


def _build_and_run_fake_smoke(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[smoke.H23IncumbentReuseSmokeRunResult, Path, str, str, list[str]]:
    h23_plan, source_summary, source_plan_path, source_summary_path = _fake_h23_inputs(
        tmp_path
    )
    monkeypatch.setattr(smoke, "load_h23_diagnostic_plan", lambda *_a, **_k: h23_plan)
    monkeypatch.setattr(
        smoke,
        "_validate_source_summary",
        lambda path: (source_summary, _file_sha(path)),
    )
    monkeypatch.setattr(
        smoke,
        "canonical_instance_hash",
        lambda _path: (_INSTANCE_SHA, {}),
    )
    commit = "1" * 40
    monkeypatch.setattr(
        smoke_verifier,
        "_git",
        lambda _root, *arguments: commit if arguments == ("rev-parse", "HEAD") else "",
    )
    repository_state, implementation_sha = _implementation_state(commit)
    plan_path = tmp_path / "smoke_plan.json"
    smoke.build_h23_incumbent_reuse_smoke_plan(
        smoke.H23IncumbentReuseSmokePlanBuildConfig(
            source_h23_plan_path=source_plan_path,
            source_h23_summary_path=source_summary_path,
            output_path=plan_path,
            expected_git_commit=commit,
        ),
        hooks=smoke.H23IncumbentReuseSmokePlanBuildHooks(
            capture_repository_state=lambda _root: repository_state
        ),
    )
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
    negative_events: list[str] = []

    def load_instance(_path: Path, config: ModelConfig, demand_scale: float):
        return SimpleNamespace(
            config=config,
            metadata={"demand_scale": demand_scale},
            screening_is_exact=lambda _mode: False,
        )

    def adapter_factory(_instance: object, _mode: str):
        return _ProbeAdapter(
            _audit(10, fingerprint=1_000),
            _signature(rhs=10.0),
            negative_events,
        )

    result = smoke.run_h23_incumbent_reuse_smoke(
        smoke.H23IncumbentReuseSmokeRunConfig(
            plan_path=plan_path,
            results_root=tmp_path / "results",
            resource_lock=tmp_path / "resource.lock",
        ),
        hooks=smoke.H23IncumbentReuseSmokeHooks(
            load_instance=load_instance,
            qualify_instance=_fake_qualifier,
            adapter_factory=adapter_factory,
            capture_environment=lambda _root: environment,
            process_table=lambda: "",
            now_utc=lambda: "2026-08-31T12:00:00Z",
        ),
    )
    return result, plan_path, commit, implementation_sha, negative_events


def test_full_fake_smoke_and_independent_verifier(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result, plan_path, commit, implementation_sha, negative_events = (
        _build_and_run_fake_smoke(tmp_path, monkeypatch)
    )

    plan_report = verify_h23_incumbent_reuse_smoke_plan(
        H23IncumbentReuseSmokePlanVerificationRequest(
            plan_path=plan_path,
            expected_git_commit=commit,
            expected_implementation_sha256=implementation_sha,
        )
    )
    report = verify_h23_incumbent_reuse_smoke_results(
        H23IncumbentReuseSmokeVerificationRequest(
            plan_path=plan_path,
            results_root=result.results_root,
            expected_run_id=result.smoke_run_id,
            expected_git_commit=commit,
            expected_implementation_sha256=implementation_sha,
        )
    )

    assert result.conclusion == "passed"
    assert plan_report.status == "verified_plan"
    assert negative_events == ["model_audit", "constraint_signature", "dispose"]
    assert report.status == "verified"
    assert report.positive_signature_equal is True
    assert report.negative_signature_different is True
    plan_payload = json.loads(plan_path.read_text(encoding="utf-8"))
    assert plan_payload["order_bias"]["execution_order_original_indices"] == [4, 3, 10]
    assert plan_payload["order_bias"]["timing_comparison_allowed"] is False


def test_verifier_rejects_semantic_negative_solve_even_with_rehashed_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result, plan_path, commit, implementation_sha, _events = _build_and_run_fake_smoke(
        tmp_path, monkeypatch
    )
    summary_path = result.summary_path
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    negative_record = summary["checkpoints"][2]
    negative_path = Path(negative_record["path"])
    checkpoint = json.loads(negative_path.read_text(encoding="utf-8"))
    checkpoint["result"]["solve_started"] = True
    checkpoint.pop("checkpoint_sha256")
    checkpoint["checkpoint_sha256"] = canonical_sha256(checkpoint)
    _write_canonical(negative_path, checkpoint)
    negative_record["checkpoint_sha256"] = checkpoint["checkpoint_sha256"]
    negative_record["file_sha256"] = _file_sha(negative_path)
    summary.pop("summary_sha256")
    summary["summary_sha256"] = canonical_sha256(summary)
    _write_canonical(summary_path, summary)

    with pytest.raises(
        H23IncumbentReuseSmokeVerificationError,
        match="negative probe solved",
    ):
        verify_h23_incumbent_reuse_smoke_results(
            H23IncumbentReuseSmokeVerificationRequest(
                plan_path=plan_path,
                results_root=result.results_root,
                expected_run_id=result.smoke_run_id,
                expected_git_commit=commit,
                expected_implementation_sha256=implementation_sha,
            )
        )
