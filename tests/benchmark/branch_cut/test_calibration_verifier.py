from __future__ import annotations

import builtins
import csv
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Callable

import pytest

from src.benchmark.branch_cut.calibration_builder import (
    DevelopmentCalibrationBuildConfig,
    build_development_calibration_manifest,
)
from src.benchmark.branch_cut.calibration_runner import (
    CalibrationBatchInfrastructureError,
    CalibrationRunnerHooks,
    DevelopmentCalibrationRunConfig,
    run_development_calibration,
)
from src.benchmark.branch_cut.calibration_verifier import (
    IMPLEMENTATION_FILES,
    LEGACY_IMPLEMENTATION_FILES,
    CalibrationVerificationError,
    CalibrationVerificationRequest,
    main,
    verify_development_calibration_results,
)
from src.benchmark.branch_cut.hashing import canonical_json, canonical_sha256
from src.benchmark.branch_cut.qualification import (
    QualificationEvidence,
    QualificationPolicy,
    assess_qualification,
    service_tolerance,
)
from src.benchmark.branch_cut.qualification_geometry_start import (
    GeometryStartPolicy,
)
from src.benchmark.branch_cut.qualification_runner import (
    QualificationInfrastructureError,
    QualificationModelAudit,
    QualificationModelContract,
    QualificationPhaseTimings,
    QualificationRunnerOutcome,
)
from src.config import DataConfig, ModelConfig


def _write_csv(
    path: Path,
    fieldnames: list[str],
    rows: list[dict[str, object]],
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


class _FakeGenerator:
    def generate(self, config: DataConfig) -> str:
        output_dir = Path(config.output_dir)
        raw_sha = hashlib.sha256(Path(config.raw_path).read_bytes()).hexdigest()
        metadata = {
            "schema_version": 2,
            "source": config.source,
            "coordinate_system": "local_km",
            "projection_center_lon": 113.9,
            "projection_center_lat": 22.5,
            "selected_days": list(config.dates),
            "raw_file_sha256": raw_sha,
            "data_config": {**asdict(config), "dates": list(config.dates)},
        }
        (output_dir / "metadata.json").write_text(
            json.dumps(metadata), encoding="utf-8"
        )
        sites = [
            {
                "site_id": f"{site_type}{index:03d}",
                "type": site_type,
                "x_km": 0.1,
                "y_km": -0.1,
                "activity_rate": 1.0,
            }
            for site_type, count in (
                ("H", config.candidate_count),
                ("L", config.num_bases),
                ("E", config.num_swaps),
            )
            for index in range(1, count + 1)
        ]
        _write_csv(
            output_dir / "sites.csv",
            ["site_id", "type", "x_km", "y_km", "activity_rate"],
            sites,
        )
        _write_csv(
            output_dir / "od_nodes.csv",
            ["node_id", "role", "x_km", "y_km", "raw_endpoint_count"],
            [
                {
                    "node_id": "O0001",
                    "role": "origin",
                    "x_km": 0.2,
                    "y_km": 0.2,
                    "raw_endpoint_count": 1,
                },
                {
                    "node_id": "D0001",
                    "role": "destination",
                    "x_km": -0.2,
                    "y_km": -0.2,
                    "raw_endpoint_count": 1,
                },
            ],
        )
        _write_csv(
            output_dir / "od_demand.csv",
            [
                "od_id",
                "origin_node",
                "destination_node",
                "order_count",
                "exposure_hours",
                "lambda_per_hour",
                "direct_distance_km",
            ],
            [
                {
                    "od_id": "OD000001",
                    "origin_node": "O0001",
                    "destination_node": "D0001",
                    "order_count": 2,
                    "exposure_hours": len(config.dates),
                    "lambda_per_hour": 2 / len(config.dates),
                    "direct_distance_km": 0.5,
                }
            ],
        )
        _write_csv(
            output_dir / "periods.csv",
            [
                "period",
                "start_hour",
                "end_hour",
                "selected_days",
                "exposure_hours",
                "order_count",
                "total_rate_per_hour",
            ],
            [
                {
                    "period": "P1",
                    "start_hour": config.demand_hour_start,
                    "end_hour": config.demand_hour_end,
                    "selected_days": "|".join(config.dates),
                    "exposure_hours": len(config.dates),
                    "order_count": 2,
                    "total_rate_per_hour": 2 / len(config.dates),
                }
            ],
        )
        return str(output_dir)


@dataclass
class _FakeInstance:
    config: ModelConfig
    demand_scale: float

    @property
    def metadata(self) -> dict[str, Any]:
        return {"demand_scale": self.demand_scale}

    def screening_is_exact(self, mode: str) -> bool:
        assert mode == "MX"
        return False


def _outcome(call: dict[str, Any], *, accepted: bool) -> QualificationRunnerOutcome:
    contract: QualificationModelContract = call["expected_model_contract"]
    scale = call["scale"]
    candidate_h = {"small": 23, "medium": 51, "large": 90}[scale]
    open_h = 55 if accepted and scale == "large" else candidate_h // 2
    if not accepted:
        open_h = 0
    service = 20.0 if accepted else 0.0
    evidence = QualificationEvidence(
        case_id=call["case_id"],
        scale=scale,
        courier_only_objective=1000.0,
        witness_objective=900.0 if accepted else 1000.0,
        total_demand_per_hour=100.0,
        witness_service_per_hour=service,
        open_facilities=open_h,
        positive_cost_open_facilities=open_h,
        candidate_h=candidate_h,
        open_h=open_h,
        witness_feasible=True,
        qualification_run_id=call["qualification_run_id"],
        instance_sha256=call["instance_sha256"],
        active_h=open_h,
        active_h_threshold_per_hour=service_tolerance(100.0),
        qualification_case_sha256=contract.qualification_case_sha256,
        effective_model_config_sha256=contract.effective_model_config_sha256,
        screening_is_exact=False,
    )
    decision = assess_qualification(evidence, call["policy"])
    assert decision.accepted is accepted
    audit = QualificationModelAudit(
        effective_model_config_sha256=contract.effective_model_config_sha256,
        demand_scale=contract.demand_scale,
        locker_fixed_cost_per_hour=contract.locker_fixed_cost_per_hour,
        display_origin_choices=contract.display_origin_choices,
        general_corridor_choices=contract.general_corridor_choices,
        screening_is_exact=False,
        model_fingerprint=12345,
    )
    return QualificationRunnerOutcome(
        accepted=accepted,
        rejection_reasons=decision.rejection_reasons,
        evidence=evidence,
        decision=decision,
        required_service_per_hour=0.1,
        required_objective_upper_bound=999.0,
        temporary_constraint_used=False,
        temporary_constraint_removed=None,
        fresh_model_validation_performed=True,
        witness_status="TIME_LIMIT",
        validation_status="TIME_LIMIT",
        captured_variable_count=3,
        exception_messages=(),
        expected_model_contract=contract,
        witness_model_audit=audit,
        fresh_model_audit=audit,
        geometry_start_policy=call["geometry_start_policy"],
        phase_timings=QualificationPhaseTimings(total_seconds=0.25),
    )


def _environment(
    repository_root: Path, source_files: tuple[str, ...] = IMPLEMENTATION_FILES
) -> dict[str, Any]:
    ordered = {
        relative: hashlib.sha256((repository_root / relative).read_bytes()).hexdigest()
        for relative in source_files
    }
    return {
        "git": {
            "repository_root": str(repository_root),
            "commit": "d" * 40,
            "worktree_dirty": False,
            "status_includes_untracked": True,
            "status_lines": [],
            "status_sha256": canonical_sha256([]),
        },
        "implementation": {
            "ordered_file_sha256": ordered,
            "implementation_sha256": canonical_sha256(ordered),
        },
        "python": {
            "implementation": "test",
            "version": "test",
            "executable": "test",
        },
        "platform": {
            "system": "test",
            "release": "test",
            "machine": "test",
            "processor": "test",
        },
        "gurobi_version": "mocked",
    }


@dataclass(frozen=True)
class _ArtifactFixture:
    manifest_path: Path
    results_root: Path
    request: CalibrationVerificationRequest


def _build_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    accept: Callable[[dict[str, Any]], bool],
    infrastructure_error: bool = False,
    source_files: tuple[str, ...] = IMPLEMENTATION_FILES,
) -> _ArtifactFixture:
    import src.benchmark.branch_cut.calibration_runner as runner_module

    raw_path = tmp_path / "held-out.csv"
    raw_path.write_text("held-out development input", encoding="utf-8")
    built = build_development_calibration_manifest(
        DevelopmentCalibrationBuildConfig(
            output_root=tmp_path / "inputs",
            tsl_raw_path=raw_path,
        ),
        generator_factory=_FakeGenerator,
    )
    monkeypatch.setattr(
        runner_module,
        "REGISTERED_DEVELOPMENT_CALIBRATION_MANIFEST_SHA256",
        built.manifest_sha256,
    )
    repository_root = Path(__file__).resolve().parents[3]
    environment = _environment(repository_root, source_files)

    def load_instance(
        _path: Path, model_config: ModelConfig, demand_scale: float
    ) -> _FakeInstance:
        return _FakeInstance(model_config, demand_scale)

    def qualify(
        _instance: object, mode: str, **kwargs: Any
    ) -> QualificationRunnerOutcome:
        call = {"mode": mode, **kwargs}
        if infrastructure_error:
            raise QualificationInfrastructureError("synthetic infrastructure drift")
        return _outcome(call, accepted=accept(call))

    results_root = tmp_path / "development-results"
    config = DevelopmentCalibrationRunConfig(
        manifest_path=built.manifest_path,
        results_root=results_root,
        max_jobs=2,
        time_limit_seconds=5.0,
        solver_seed=42,
        miqcp_method=-1,
        native_cuts=-1,
        root_cut_passes=2,
        scales=("small",),
        development_seed_stages=(0,),
        profile_order_indices=(0, 1),
        policy=QualificationPolicy(large_scale_open_h_target=(50, 60)),
        geometry_start_policy=GeometryStartPolicy(enabled=False),
        resource_lock=tmp_path / "resource.lock",
    )
    hooks = CalibrationRunnerHooks(
        load_instance=load_instance,
        qualify_instance=qualify,
        capture_environment=lambda _root: environment,
        now_utc=lambda: "2026-08-31T00:00:00Z",
    )
    if infrastructure_error:
        with pytest.raises(CalibrationBatchInfrastructureError):
            run_development_calibration(config, hooks=hooks)
    else:
        run_development_calibration(config, hooks=hooks)
    identity = json.loads(
        (results_root / "run_identity.json").read_text(encoding="utf-8")
    )
    request = CalibrationVerificationRequest(
        manifest_path=built.manifest_path,
        results_root=results_root,
        expected_run_id=identity["calibration_run_id"],
        expected_manifest_sha256=built.manifest_sha256,
        expected_git_commit="d" * 40,
        expected_implementation_sha256=environment["implementation"][
            "implementation_sha256"
        ],
    )
    return _ArtifactFixture(built.manifest_path, results_root, request)


def _rewrite_summary(path: Path, mutate: Callable[[dict[str, Any]], None]) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    mutate(payload)
    entry = {
        key: value
        for key, value in payload.items()
        if key not in {"artifact_type", "artifact_schema_version", "summary_sha256"}
    }
    payload["summary_sha256"] = canonical_sha256(entry)
    path.write_text(canonical_json(payload) + "\n", encoding="utf-8")


def test_registered_legacy_source_set_still_verifies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _build_artifacts(
        tmp_path,
        monkeypatch,
        accept=lambda _call: False,
        source_files=LEGACY_IMPLEMENTATION_FILES,
    )
    report = verify_development_calibration_results(fixture.request)
    assert report.status == "completed"


@pytest.mark.parametrize(
    "source_files",
    (LEGACY_IMPLEMENTATION_FILES[:-1], (*LEGACY_IMPLEMENTATION_FILES, "main.py")),
)
def test_resealed_unregistered_source_set_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    source_files: tuple[str, ...],
) -> None:
    fixture = _build_artifacts(
        tmp_path,
        monkeypatch,
        accept=lambda _call: False,
        source_files=source_files,
    )
    with pytest.raises(CalibrationVerificationError, match="implementation file set"):
        verify_development_calibration_results(fixture.request)


def _rewrite_checkpoint_and_references(
    fixture: _ArtifactFixture,
    mutate: Callable[[dict[str, Any]], None],
) -> None:
    summary_path = fixture.results_root / "batch_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    record = summary["checkpoints"][0]
    checkpoint_path = Path(record["path"])
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    mutate(checkpoint)
    checkpoint["checkpoint_sha256"] = canonical_sha256(
        {"identity": checkpoint["identity"], "result": checkpoint["result"]}
    )
    checkpoint_path.write_text(canonical_json(checkpoint) + "\n", encoding="utf-8")
    record["checkpoint_sha256"] = checkpoint["checkpoint_sha256"]
    job_index = record["job_order_index"]
    summary["logical_job_ledger"][job_index]["checkpoint_sha256"] = checkpoint[
        "checkpoint_sha256"
    ]
    entry = {
        key: value
        for key, value in summary.items()
        if key not in {"artifact_type", "artifact_schema_version", "summary_sha256"}
    }
    summary["summary_sha256"] = canonical_sha256(entry)
    summary_path.write_text(canonical_json(summary) + "\n", encoding="utf-8")


def test_verifier_accepts_normal_rejection_and_recomputes_canonical_ledger(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = 0

    def accept(_call: dict[str, Any]) -> bool:
        nonlocal calls
        calls += 1
        return calls == 2

    fixture = _build_artifacts(tmp_path, monkeypatch, accept=accept)
    report = verify_development_calibration_results(fixture.request)

    assert report.status == "completed"
    assert report.attempted_job_count == 2
    assert report.accepted_job_count == 1
    assert report.rejected_attempted_job_count == 1
    assert report.infrastructure_failure_count == 0
    assert report.unattempted_job_count == 142
    assert report.calibration_conclusion == "bounded_partial_run"


def test_verifier_accepts_infrastructure_abort_as_distinct_terminal_semantics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _build_artifacts(
        tmp_path,
        monkeypatch,
        accept=lambda _call: True,
        infrastructure_error=True,
    )

    report = verify_development_calibration_results(fixture.request)

    assert report.status == "aborted_on_infrastructure_failure"
    assert report.attempted_job_count == 1
    assert report.infrastructure_failure_count == 1
    assert report.rejected_attempted_job_count == 0
    assert report.calibration_conclusion is None


def test_verifier_rejects_rehashed_144_ledger_reason_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _build_artifacts(
        tmp_path, monkeypatch, accept=lambda _call: True
    )
    summary = fixture.results_root / "batch_summary.json"

    def mutate(payload: dict[str, Any]) -> None:
        payload["logical_job_ledger"][2]["not_attempted_reason_code"] = (
            "max_jobs_limit"
        )
        payload["logical_job_ledger"][2]["not_attempted_reason"] = "max_jobs_limit"

    _rewrite_summary(summary, mutate)

    with pytest.raises(CalibrationVerificationError, match="144-row ledger"):
        verify_development_calibration_results(fixture.request)


def test_verifier_rejects_reordered_attempts_even_with_rehashed_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _build_artifacts(
        tmp_path, monkeypatch, accept=lambda _call: True
    )
    summary = fixture.results_root / "batch_summary.json"

    def mutate(payload: dict[str, Any]) -> None:
        payload["checkpoints"].reverse()

    _rewrite_summary(summary, mutate)

    with pytest.raises(CalibrationVerificationError, match="checkpoint identity"):
        verify_development_calibration_results(fixture.request)


@pytest.mark.parametrize("tamper", ["profile", "model_config", "case_hash"])
def test_verifier_rejects_self_consistently_rehashed_checkpoint_identity_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tamper: str,
) -> None:
    fixture = _build_artifacts(
        tmp_path, monkeypatch, accept=lambda _call: True
    )

    def mutate(checkpoint: dict[str, Any]) -> None:
        identity = checkpoint["identity"]
        if tamper == "profile":
            identity["qualification_solve_profile"]["demand_scale"] = 9.0
        elif tamper == "model_config":
            identity["effective_model_config"]["locker_fixed_cost_per_hour"] += 1.0
        else:
            identity["job"]["qualification_case_sha256"] = "0" * 64

    _rewrite_checkpoint_and_references(fixture, mutate)

    with pytest.raises(CalibrationVerificationError, match="checkpoint identity"):
        verify_development_calibration_results(fixture.request)


def test_verifier_rejects_checkpoint_symlink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _build_artifacts(
        tmp_path, monkeypatch, accept=lambda _call: True
    )
    summary = json.loads(
        (fixture.results_root / "batch_summary.json").read_text(encoding="utf-8")
    )
    checkpoint = Path(summary["checkpoints"][0]["path"])
    copy = tmp_path / "checkpoint-copy.json"
    copy.write_bytes(checkpoint.read_bytes())
    checkpoint.unlink()
    checkpoint.symlink_to(copy)

    with pytest.raises(CalibrationVerificationError, match="symbolic-link"):
        verify_development_calibration_results(fixture.request)


def test_verifier_rejects_wrong_explicit_implementation_pin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _build_artifacts(
        tmp_path, monkeypatch, accept=lambda _call: True
    )
    bad_request = CalibrationVerificationRequest(
        manifest_path=fixture.request.manifest_path,
        results_root=fixture.request.results_root,
        expected_run_id=fixture.request.expected_run_id,
        expected_manifest_sha256=fixture.request.expected_manifest_sha256,
        expected_git_commit=fixture.request.expected_git_commit,
        expected_implementation_sha256="0" * 64,
    )

    with pytest.raises(CalibrationVerificationError, match="explicit expected pin"):
        verify_development_calibration_results(bad_request)


def test_verifier_rehashes_each_canonical_input_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _build_artifacts(
        tmp_path, monkeypatch, accept=lambda _call: True
    )
    manifest = json.loads(fixture.manifest_path.read_text(encoding="utf-8"))
    sites = (
        fixture.manifest_path.parent
        / manifest["physical_instances"][0]["instance_ref"]
        / "sites.csv"
    )
    sites.write_text(sites.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    with pytest.raises(CalibrationVerificationError, match="physical instance identity"):
        verify_development_calibration_results(fixture.request)


def test_cli_verifies_without_importing_gurobi(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fixture = _build_artifacts(
        tmp_path, monkeypatch, accept=lambda _call: True
    )
    real_import = builtins.__import__

    def guarded_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "gurobipy" or name.startswith("gurobipy."):
            raise AssertionError("verifier must not import Gurobi")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    request = fixture.request
    code = main(
        [
            "--manifest",
            str(request.manifest_path),
            "--results-root",
            str(request.results_root),
            "--expected-run-id",
            request.expected_run_id,
            "--expected-manifest-sha256",
            request.expected_manifest_sha256,
            "--expected-git-commit",
            request.expected_git_commit,
            "--expected-implementation-sha256",
            request.expected_implementation_sha256,
        ]
    )

    assert code == 0
    assert json.loads(capsys.readouterr().out)["status"] == "completed"
