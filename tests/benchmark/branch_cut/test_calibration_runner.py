from __future__ import annotations

import builtins
import csv
from dataclasses import asdict, dataclass, replace
import hashlib
import json
from pathlib import Path
from typing import Any, Callable

import pytest

import src.benchmark.branch_cut.calibration_runner as runner_module
from src.benchmark.branch_cut.calibration_builder import (
    DevelopmentCalibrationBuildConfig,
    build_development_calibration_manifest,
)
from src.benchmark.branch_cut.calibration_runner import (
    CalibrationBatchInfrastructureError,
    CalibrationRunnerError,
    CalibrationRunnerHooks,
    DevelopmentCalibrationRunConfig,
    DevelopmentCalibrationRunResult,
    load_development_calibration_manifest,
    main,
    run_development_calibration,
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
    QualificationRunnerReason,
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


class FakeGenerator:
    def generate(self, config: DataConfig) -> str:
        output_dir = Path(config.output_dir)
        raw_sha256 = hashlib.sha256(Path(config.raw_path).read_bytes()).hexdigest()
        metadata = {
            "schema_version": 2,
            "source": config.source,
            "coordinate_system": "local_km",
            "projection_center_lon": 113.9,
            "projection_center_lat": 22.5,
            "selected_days": list(config.dates),
            "raw_file_sha256": raw_sha256,
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


@dataclass(frozen=True)
class ManifestFixture:
    path: Path
    sha256: str


@pytest.fixture
def calibration_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> ManifestFixture:
    raw_path = tmp_path / "tsl-development.csv"
    raw_path.write_text("held-out development input", encoding="utf-8")
    result = build_development_calibration_manifest(
        DevelopmentCalibrationBuildConfig(
            output_root=tmp_path / "calibration-inputs",
            tsl_raw_path=raw_path,
        ),
        generator_factory=FakeGenerator,
    )
    monkeypatch.setattr(
        runner_module,
        "REGISTERED_DEVELOPMENT_CALIBRATION_MANIFEST_SHA256",
        result.manifest_sha256,
    )
    return ManifestFixture(result.manifest_path, result.manifest_sha256)


def _repin_manifest(
    fixture: ManifestFixture,
    monkeypatch: pytest.MonkeyPatch,
    mutate: Callable[[dict[str, Any]], None],
) -> None:
    payload = json.loads(fixture.path.read_text(encoding="utf-8"))
    mutate(payload)
    unhashed = {
        key: value
        for key, value in payload.items()
        if key != "development_calibration_manifest_sha256"
    }
    digest = canonical_sha256(unhashed)
    payload["development_calibration_manifest_sha256"] = digest
    fixture.path.write_text(canonical_json(payload) + "\n", encoding="utf-8")
    monkeypatch.setattr(
        runner_module,
        "REGISTERED_DEVELOPMENT_CALIBRATION_MANIFEST_SHA256",
        digest,
    )


def _environment() -> dict[str, Any]:
    return {
        "git": {
            "commit": "d" * 40,
            "worktree_dirty": False,
            "status_includes_untracked": True,
            "status_lines": [],
            "status_sha256": canonical_sha256(()),
        },
        "python": {"version": "test"},
        "platform": {"machine": "test"},
        "gurobi_version": "mocked",
        "implementation": {"implementation_sha256": "e" * 64},
    }


@dataclass
class FakeInstance:
    config: ModelConfig
    demand_scale: float
    exact: bool = False

    @property
    def metadata(self) -> dict[str, Any]:
        return {"demand_scale": self.demand_scale}

    def screening_is_exact(self, mode: str) -> bool:
        assert mode == "MX"
        return self.exact


def _outcome(call: dict[str, Any], *, accepted: bool) -> QualificationRunnerOutcome:
    contract: QualificationModelContract = call["expected_model_contract"]
    scale = call["scale"]
    candidate_h = {"small": 23, "medium": 51, "large": 90}[scale]
    if accepted:
        open_h = 55 if scale == "large" else max(4, candidate_h // 2)
        service = 20.0
        witness_objective = 900.0
        open_facilities = open_h
        active_h = open_h
    else:
        open_h = 0
        service = 0.0
        witness_objective = 1000.0
        open_facilities = 0
        active_h = 0
    evidence = QualificationEvidence(
        case_id=call["case_id"],
        scale=scale,
        courier_only_objective=1000.0,
        witness_objective=witness_objective,
        total_demand_per_hour=100.0,
        witness_service_per_hour=service,
        open_facilities=open_facilities,
        positive_cost_open_facilities=open_facilities,
        candidate_h=candidate_h,
        open_h=open_h,
        witness_feasible=True,
        qualification_run_id=call["qualification_run_id"],
        instance_sha256=call["instance_sha256"],
        active_h=active_h,
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
        model_fingerprint=123456,
    )
    return QualificationRunnerOutcome(
        accepted=accepted,
        rejection_reasons=decision.rejection_reasons,
        evidence=evidence,
        decision=decision,
        required_service_per_hour=0.001,
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


@dataclass
class HookRecorder:
    accept: Callable[[dict[str, Any]], bool]
    exact_screening: bool = False
    qualifier_error: BaseException | None = None
    loader_calls: list[tuple[Path, ModelConfig, float]] = None  # type: ignore[assignment]
    qualifier_calls: list[dict[str, Any]] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self.loader_calls = []
        self.qualifier_calls = []

    def load_instance(
        self, path: Path, model_config: ModelConfig, demand_scale: float
    ) -> FakeInstance:
        self.loader_calls.append((path, model_config, demand_scale))
        return FakeInstance(model_config, demand_scale, self.exact_screening)

    def qualify(
        self, _instance: object, mode: str, **kwargs: Any
    ) -> QualificationRunnerOutcome:
        call = {"mode": mode, **kwargs}
        self.qualifier_calls.append(call)
        if self.qualifier_error is not None:
            raise self.qualifier_error
        return _outcome(call, accepted=self.accept(call))

    def hooks(self) -> CalibrationRunnerHooks:
        return CalibrationRunnerHooks(
            load_instance=self.load_instance,
            qualify_instance=self.qualify,
            capture_environment=lambda _root: _environment(),
            now_utc=lambda: "2026-08-31T00:00:00Z",
        )


def _config(
    fixture: ManifestFixture,
    tmp_path: Path,
    *,
    max_jobs: int,
    scales: tuple[str, ...] | None = None,
    seed_stages: tuple[int, ...] | None = None,
    profiles: tuple[int, ...] | None = None,
    results_name: str = "calibration-results",
) -> DevelopmentCalibrationRunConfig:
    return DevelopmentCalibrationRunConfig(
        manifest_path=fixture.path,
        results_root=tmp_path / results_name,
        max_jobs=max_jobs,
        time_limit_seconds=5.0,
        solver_seed=42,
        miqcp_method=-1,
        native_cuts=-1,
        root_cut_passes=2,
        scales=scales,
        development_seed_stages=seed_stages,
        profile_order_indices=profiles,
        resource_lock=tmp_path / f"{results_name}.lock",
    )


def test_manifest_preflight_recomputes_registered_144_job_design(
    calibration_manifest: ManifestFixture,
) -> None:
    loaded = load_development_calibration_manifest(calibration_manifest.path)

    assert loaded.manifest_sha256 == calibration_manifest.sha256
    assert len(loaded.physical_instances) == 6
    assert len(loaded.profiles) == 24
    assert len(loaded.jobs) == 144
    assert [job.job_order_index for job in loaded.jobs] == list(range(144))
    assert [
        (loaded.jobs[index].scale, loaded.jobs[index].development_seed_stage)
        for index in (0, 24, 48, 72, 96, 120)
    ] == [
        ("small", 0),
        ("small", 1),
        ("medium", 0),
        ("medium", 1),
        ("large", 0),
        ("large", 1),
    ]


def test_manifest_rejects_unregistered_but_self_consistent_hash(
    calibration_manifest: ManifestFixture,
) -> None:
    payload = json.loads(calibration_manifest.path.read_text(encoding="utf-8"))
    payload["calibration_grid"]["economic_interpretation"] += " changed"
    unhashed = {
        key: value
        for key, value in payload.items()
        if key != "development_calibration_manifest_sha256"
    }
    payload["development_calibration_manifest_sha256"] = canonical_sha256(unhashed)
    calibration_manifest.path.write_text(
        canonical_json(payload) + "\n", encoding="utf-8"
    )

    with pytest.raises(CalibrationRunnerError, match="not the preregistered"):
        load_development_calibration_manifest(calibration_manifest.path)


def test_manifest_rejects_embedded_hash_mismatch(
    calibration_manifest: ManifestFixture,
) -> None:
    payload = json.loads(calibration_manifest.path.read_text(encoding="utf-8"))
    payload["logical_job_count"] = 143
    calibration_manifest.path.write_text(
        canonical_json(payload) + "\n", encoding="utf-8"
    )

    with pytest.raises(CalibrationRunnerError, match="content hash mismatch"):
        load_development_calibration_manifest(calibration_manifest.path)


def test_manifest_rejects_raw_and_canonical_file_drift(
    calibration_manifest: ManifestFixture,
) -> None:
    payload = json.loads(calibration_manifest.path.read_text(encoding="utf-8"))
    raw_path = Path(payload["raw_input"]["path"])
    raw_path.write_text("changed", encoding="utf-8")
    with pytest.raises(CalibrationRunnerError, match="raw input hash or size mismatch"):
        load_development_calibration_manifest(calibration_manifest.path)


def test_manifest_rejects_rehashed_formal_eligibility_flag(
    calibration_manifest: ManifestFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _repin_manifest(
        calibration_manifest,
        monkeypatch,
        lambda payload: payload.__setitem__("eligible_for_formal_pool", True),
    )

    with pytest.raises(
        CalibrationRunnerError, match="invalid eligible_for_formal_pool"
    ):
        load_development_calibration_manifest(calibration_manifest.path)


def test_manifest_rejects_canonical_instance_drift(
    calibration_manifest: ManifestFixture,
) -> None:
    payload = json.loads(calibration_manifest.path.read_text(encoding="utf-8"))
    sites = (
        calibration_manifest.path.parent
        / payload["physical_instances"][0]["instance_ref"]
        / "sites.csv"
    )
    sites.write_text(sites.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    with pytest.raises(
        CalibrationRunnerError, match="physical instance metadata mismatch"
    ):
        load_development_calibration_manifest(calibration_manifest.path)


@pytest.mark.parametrize("tamper", ["profile", "case_hash", "job_order"])
def test_manifest_rejects_rehashed_internal_identity_tampering(
    calibration_manifest: ManifestFixture,
    monkeypatch: pytest.MonkeyPatch,
    tamper: str,
) -> None:
    def mutate(payload: dict[str, Any]) -> None:
        if tamper == "profile":
            profile = payload["calibration_grid"]["profiles"][0]
            profile["effective_model_config"]["locker_fixed_cost_per_hour"] += 1.0
            profile["effective_model_config_sha256"] = canonical_sha256(
                profile["effective_model_config"]
            )
        elif tamper == "case_hash":
            payload["logical_job_order"][0]["qualification_case_sha256"] = "0" * 64
        else:
            payload["logical_job_order"][0], payload["logical_job_order"][1] = (
                payload["logical_job_order"][1],
                payload["logical_job_order"][0],
            )

    _repin_manifest(calibration_manifest, monkeypatch, mutate)

    expected = "calibration profiles" if tamper == "profile" else "logical job IDs"
    with pytest.raises(CalibrationRunnerError, match=expected):
        load_development_calibration_manifest(calibration_manifest.path)


def test_h23_bounded_run_uses_full_profile_config_and_screened_contract_without_gurobi(
    calibration_manifest: ManifestFixture,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    imports: list[str] = []
    real_import = builtins.__import__

    def guarded_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "gurobipy" or name.startswith("gurobipy."):
            imports.append(name)
            raise AssertionError("mocked calibration may not import Gurobi")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    recorder = HookRecorder(lambda _call: True)
    config = _config(
        calibration_manifest,
        tmp_path,
        max_jobs=1,
        scales=("small",),
        seed_stages=(0,),
        profiles=(0,),
    )

    result = run_development_calibration(config, hooks=recorder.hooks())

    assert not imports
    assert result.selected_job_count == 1
    assert result.attempted_job_count == 1
    assert result.accepted_job_count == 1
    assert result.unattempted_job_count == 143
    assert result.locked_profile_ids == ()
    assert len(recorder.loader_calls) == len(recorder.qualifier_calls) == 1
    _path, model_config, demand_scale = recorder.loader_calls[0]
    assert demand_scale == 1.0
    assert model_config.feeder_rider_cost_factor == 0.30
    assert model_config.locker_fixed_cost_per_hour == 20.0
    assert model_config.display_origin_choices == 8
    assert model_config.general_corridor_choices == 10
    call = recorder.qualifier_calls[0]
    assert call["mode"] == "MX"
    assert call["fail_fast_infrastructure"] is True
    assert call["settings"].threads == 1
    assert call["settings"].time_limit_seconds == 5.0
    assert call["settings"].solver_seed == 42
    assert call["settings"].miqcp_method == -1
    assert call["settings"].native_cuts == -1
    assert call["settings"].root_cut_passes == 2
    assert call["geometry_start_policy"] == GeometryStartPolicy(enabled=False)
    assert call["expected_model_contract"].screening_is_exact is False

    checkpoint = json.loads(result.checkpoint_paths[0].read_text(encoding="utf-8"))
    assert checkpoint["result"]["outcome"] == _outcome(call, accepted=True).to_payload()
    assert checkpoint["result"]["outcome"]["phase_timings"]["total_seconds"] == 0.25
    assert (
        checkpoint["result"]["outcome"]["witness_model_audit"]["screening_is_exact"]
        is False
    )
    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert len(summary["logical_job_ledger"]) == 144
    assert summary["logical_job_ledger"][0]["attempted"] is True
    assert all(
        row["not_attempted_reason_code"] == "selector_excluded"
        for row in summary["logical_job_ledger"][1:]
    )
    assert summary["calibration_conclusion"] == "bounded_partial_run"
    assert summary["formal_pool_created"] is False


def test_preregistered_scheduler_advances_only_fresh_accepted_profiles(
    calibration_manifest: ManifestFixture,
    tmp_path: Path,
) -> None:
    loaded = load_development_calibration_manifest(calibration_manifest.path)
    index_by_id = {job.job_id: job.job_order_index for job in loaded.jobs}
    accepted_by_index = {0, 2, 4, 48, 52, 100, 28, 76, 124}
    recorder = HookRecorder(
        lambda call: index_by_id[call["case_id"]] in accepted_by_index
    )

    result = run_development_calibration(
        _config(calibration_manifest, tmp_path, max_jobs=144),
        hooks=recorder.hooks(),
    )

    observed = [index_by_id[call["case_id"]] for call in recorder.qualifier_calls]
    assert observed == [*range(24), 48, 50, 52, 96, 100, 28, 76, 124]
    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert [row["job_order_index"] for row in summary["logical_job_ledger"]] == list(
        range(144)
    )
    assert summary["logical_job_ledger"][49]["attempted"] is False
    assert summary["logical_job_ledger"][49]["not_attempted_reason_code"] == (
        "stage_gate"
    )
    assert summary["logical_job_ledger"][49]["not_attempted_reason"] == (
        "not_reached_by_preregistered_stage_gate"
    )
    assert summary["calibration_conclusion"] == (
        "development_no_go_fewer_than_four_profiles"
    )


def test_scheduler_runs_all_six_stages_before_locking_first_four_profiles(
    calibration_manifest: ManifestFixture,
    tmp_path: Path,
) -> None:
    loaded = load_development_calibration_manifest(calibration_manifest.path)
    job_by_id = {job.job_id: job for job in loaded.jobs}
    recorder = HookRecorder(
        lambda call: job_by_id[call["case_id"]].profile_order_index < 5
    )

    result = run_development_calibration(
        _config(calibration_manifest, tmp_path, max_jobs=144),
        hooks=recorder.hooks(),
    )

    observed = [
        job_by_id[call["case_id"]].job_order_index for call in recorder.qualifier_calls
    ]
    assert observed == [
        *range(24),
        *range(48, 53),
        *range(96, 101),
        *range(24, 29),
        *range(72, 77),
        *range(120, 125),
    ]
    assert 124 in observed
    assert result.locked_profile_ids == tuple(
        profile.profile_id for profile in loaded.profiles[:4]
    )
    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert summary["calibration_conclusion"] == "development_profile_lock_ready"
    assert summary["ladder_complete"] is True
    assert summary["formal_pool_created"] is False


def test_normal_rejection_is_checkpointed_and_next_profile_continues(
    calibration_manifest: ManifestFixture,
    tmp_path: Path,
) -> None:
    loaded = load_development_calibration_manifest(calibration_manifest.path)
    index_by_id = {job.job_id: job.job_order_index for job in loaded.jobs}
    recorder = HookRecorder(lambda call: index_by_id[call["case_id"]] == 1)
    result = run_development_calibration(
        _config(
            calibration_manifest,
            tmp_path,
            max_jobs=2,
            scales=("small",),
            seed_stages=(0,),
            profiles=(0, 1),
        ),
        hooks=recorder.hooks(),
    )

    assert [index_by_id[call["case_id"]] for call in recorder.qualifier_calls] == [
        0,
        1,
    ]
    assert result.accepted_job_count == 1
    first = json.loads(result.checkpoint_paths[0].read_text(encoding="utf-8"))
    assert first["result"]["status"] == "completed"
    assert first["result"]["outcome"]["accepted"] is False


def test_max_jobs_reason_is_distinct_from_selector_and_stage_gate(
    calibration_manifest: ManifestFixture,
    tmp_path: Path,
) -> None:
    recorder = HookRecorder(lambda _call: True)
    result = run_development_calibration(
        _config(
            calibration_manifest,
            tmp_path,
            max_jobs=1,
            scales=("small",),
            seed_stages=(0,),
            profiles=(0, 1),
        ),
        hooks=recorder.hooks(),
    )

    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert summary["logical_job_ledger"][1]["not_attempted_reason_code"] == (
        "max_jobs_limit"
    )
    assert summary["logical_job_ledger"][2]["not_attempted_reason_code"] == (
        "selector_excluded"
    )


def test_normal_extra_runner_reason_is_rejection_not_infrastructure(
    calibration_manifest: ManifestFixture,
    tmp_path: Path,
) -> None:
    recorder = HookRecorder(lambda _call: False)

    def qualifier(
        _instance: object, mode: str, **kwargs: Any
    ) -> QualificationRunnerOutcome:
        call = {"mode": mode, **kwargs}
        recorder.qualifier_calls.append(call)
        outcome = _outcome(call, accepted=False)
        return replace(
            outcome,
            rejection_reasons=(
                QualificationRunnerReason.NO_POSITIVE_SERVICE_INCUMBENT,
                *outcome.decision.rejection_reasons,
            ),
        )

    result = run_development_calibration(
        _config(
            calibration_manifest,
            tmp_path,
            max_jobs=1,
            scales=("small",),
            seed_stages=(0,),
            profiles=(0,),
        ),
        hooks=replace(recorder.hooks(), qualify_instance=qualifier),
    )

    assert result.attempted_job_count == 1
    assert result.accepted_job_count == 0
    checkpoint = json.loads(result.checkpoint_paths[0].read_text(encoding="utf-8"))
    assert checkpoint["result"]["status"] == "completed"
    assert checkpoint["result"]["outcome"]["rejection_reasons"][0] == (
        QualificationRunnerReason.NO_POSITIVE_SERVICE_INCUMBENT
    )


def test_malicious_accepted_outcome_with_rejected_decision_aborts(
    calibration_manifest: ManifestFixture,
    tmp_path: Path,
) -> None:
    recorder = HookRecorder(lambda _call: False)

    def qualifier(
        _instance: object, mode: str, **kwargs: Any
    ) -> QualificationRunnerOutcome:
        call = {"mode": mode, **kwargs}
        recorder.qualifier_calls.append(call)
        return replace(
            _outcome(call, accepted=False), accepted=True, rejection_reasons=()
        )

    with pytest.raises(
        CalibrationBatchInfrastructureError,
        match="decision reasons|decision and witness",
    ):
        run_development_calibration(
            _config(
                calibration_manifest,
                tmp_path,
                max_jobs=1,
                scales=("small",),
                seed_stages=(0,),
                profiles=(0,),
            ),
            hooks=replace(recorder.hooks(), qualify_instance=qualifier),
        )


def test_infrastructure_failure_is_archived_and_stops_batch(
    calibration_manifest: ManifestFixture,
    tmp_path: Path,
) -> None:
    recorder = HookRecorder(
        lambda _call: True,
        qualifier_error=QualificationInfrastructureError("synthetic audit drift"),
    )
    config = _config(
        calibration_manifest,
        tmp_path,
        max_jobs=2,
        scales=("small",),
        seed_stages=(0,),
        profiles=(0, 1),
    )

    with pytest.raises(
        CalibrationBatchInfrastructureError, match="synthetic audit drift"
    ):
        run_development_calibration(config, hooks=recorder.hooks())

    assert len(recorder.qualifier_calls) == 1
    checkpoints = list((config.results_root / "checkpoints").glob("*.json"))
    assert len(checkpoints) == 1
    failure = json.loads(checkpoints[0].read_text(encoding="utf-8"))
    assert failure["result"]["status"] == "infrastructure_failure"
    assert failure["result"]["outcome"] is None
    aborted = json.loads(
        (config.results_root / "batch_aborted.json").read_text(encoding="utf-8")
    )
    assert aborted["status"] == "aborted_on_infrastructure_failure"
    assert len(aborted["logical_job_ledger"]) == 144


def test_runtime_exact_screening_is_infrastructure_failure_before_solver(
    calibration_manifest: ManifestFixture,
    tmp_path: Path,
) -> None:
    recorder = HookRecorder(lambda _call: True, exact_screening=True)
    config = _config(
        calibration_manifest,
        tmp_path,
        max_jobs=1,
        scales=("small",),
        seed_stages=(0,),
        profiles=(0,),
    )

    with pytest.raises(
        CalibrationBatchInfrastructureError, match="screening_is_exact=false"
    ):
        run_development_calibration(config, hooks=recorder.hooks())
    assert not recorder.qualifier_calls


def test_dirty_environment_including_untracked_files_is_rejected_before_claim(
    calibration_manifest: ManifestFixture,
    tmp_path: Path,
) -> None:
    recorder = HookRecorder(lambda _call: True)
    dirty_environment = _environment()
    dirty_environment["git"] = {
        **dirty_environment["git"],
        "worktree_dirty": True,
        "status_lines": ["?? untracked.txt"],
        "status_sha256": canonical_sha256(("?? untracked.txt",)),
    }
    hooks = replace(
        recorder.hooks(),
        capture_environment=lambda _root: dirty_environment,
    )
    config = _config(
        calibration_manifest,
        tmp_path,
        max_jobs=1,
        scales=("small",),
        seed_stages=(0,),
        profiles=(0,),
    )

    with pytest.raises(CalibrationRunnerError, match="clean worktree"):
        run_development_calibration(config, hooks=hooks)
    assert not config.results_root.exists()
    assert not recorder.loader_calls
    assert not recorder.qualifier_calls


def test_run_rejects_existing_results_root_without_overwrite(
    calibration_manifest: ManifestFixture,
    tmp_path: Path,
) -> None:
    recorder = HookRecorder(lambda _call: True)
    config = _config(
        calibration_manifest,
        tmp_path,
        max_jobs=1,
        scales=("small",),
        seed_stages=(0,),
        profiles=(0,),
    )
    config.results_root.mkdir()
    marker = config.results_root / "keep.txt"
    marker.write_text("keep", encoding="utf-8")

    with pytest.raises(CalibrationRunnerError, match="refusing to reuse"):
        run_development_calibration(config, hooks=recorder.hooks())
    assert marker.read_text(encoding="utf-8") == "keep"
    assert not recorder.qualifier_calls


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"time_limit_seconds": 6.0}, "native qualification settings drift"),
        ({"solver_seed": 43}, "native qualification settings drift"),
        ({"miqcp_method": 0}, "native qualification settings drift"),
        ({"native_cuts": 0}, "native qualification settings drift"),
        ({"root_cut_passes": 1}, "native qualification settings drift"),
        ({"concurrency": 2}, "concurrency=1"),
        ({"mode": "HD"}, "fixed to MX"),
        (
            {"geometry_start_policy": GeometryStartPolicy(enabled=True)},
            "geometry start is disabled",
        ),
        (
            {"policy": QualificationPolicy()},
            "qualification policy is fixed",
        ),
    ],
)
def test_run_config_fails_closed_on_preregistered_setting_drift(
    calibration_manifest: ManifestFixture,
    tmp_path: Path,
    changes: dict[str, Any],
    message: str,
) -> None:
    baseline = _config(calibration_manifest, tmp_path, max_jobs=1)
    with pytest.raises(ValueError, match=message):
        replace(baseline, **changes)


def test_cli_requires_explicit_bound_and_builds_locked_settings(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    captured: list[DevelopmentCalibrationRunConfig] = []

    def fake_runner(
        config: DevelopmentCalibrationRunConfig,
    ) -> DevelopmentCalibrationRunResult:
        captured.append(config)
        return DevelopmentCalibrationRunResult(
            results_root=config.results_root,
            run_identity_path=config.results_root / "run_identity.json",
            summary_path=config.results_root / "batch_summary.json",
            calibration_run_id="development-calibration-run-" + "f" * 64,
            selected_job_count=1,
            attempted_job_count=1,
            accepted_job_count=0,
            unattempted_job_count=143,
            locked_profile_ids=(),
            checkpoint_paths=(),
        )

    arguments = [
        "--manifest",
        str(tmp_path / "manifest.json"),
        "--results-root",
        str(tmp_path / "results"),
        "--max-jobs",
        "1",
        "--time-limit",
        "5",
        "--solver-seed",
        "42",
        "--miqcp-method",
        "-1",
        "--native-cuts",
        "-1",
        "--root-cut-passes",
        "2",
        "--scale",
        "small",
        "--development-seed-stage",
        "0",
        "--profile-order-index",
        "0",
    ]
    assert main(arguments, runner=fake_runner) == 0
    assert captured[0].max_jobs == 1
    assert captured[0].native_settings().time_limit_seconds == 5.0
    assert captured[0].native_settings().threads == 1
    assert captured[0].geometry_start_policy.enabled is False
    assert "development-calibration-run" in capsys.readouterr().out

    with pytest.raises(SystemExit) as exc_info:
        main(
            [item for index, item in enumerate(arguments) if index not in {4, 5}],
            runner=fake_runner,
        )
    assert exc_info.value.code == 2
