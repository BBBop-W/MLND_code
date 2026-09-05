from __future__ import annotations

import builtins
import csv
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

import src.benchmark.branch_cut.calibration_builder as calibration_module
from src.benchmark.branch_cut.calibration_builder import (
    DEVELOPMENT_BASE_LOCKER_FIXED_COST_PER_DEMAND_UNIT_ORDER,
    DEVELOPMENT_CALIBRATION_PROFILES,
    DEVELOPMENT_DEMAND_SCALE_ORDER,
    DEVELOPMENT_DISPLAY_ORIGIN_CHOICES,
    DEVELOPMENT_EXPECTED_SCREENING_IS_EXACT,
    DEVELOPMENT_FEEDER_FACTOR_ORDER,
    DEVELOPMENT_GENERAL_CORRIDOR_CHOICES,
    DEVELOPMENT_LAYOUT_SEED_ORDER,
    DevelopmentCalibrationBuildConfig,
    build_parser,
    build_development_calibration_manifest,
    development_instance_plans,
    main,
)
from src.benchmark.branch_cut.case_builder import (
    DEVELOPMENT_DEMAND_BLOCK,
    LOCKED_DEMAND_BLOCKS,
    LOCKED_LAYOUT_SEED_ORDER,
    LOCKED_QUALIFICATION_PROFILES,
    QualificationSolveProfile,
    canonical_instance_hash,
    qualification_case_sha256,
)
from src.benchmark.branch_cut.hashing import canonical_json, canonical_sha256
from src.config import DataConfig, to_dict


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
    calls: list[DataConfig] = []
    wrong_raw_hash = False
    wrong_h_count = False
    drift_center = False

    @classmethod
    def reset(cls) -> None:
        cls.calls = []
        cls.wrong_raw_hash = False
        cls.wrong_h_count = False
        cls.drift_center = False

    def generate(self, config: DataConfig) -> str:
        type(self).calls.append(config)
        output_dir = Path(config.output_dir)
        raw_sha256 = hashlib.sha256(Path(config.raw_path).read_bytes()).hexdigest()
        if type(self).wrong_raw_hash:
            raw_sha256 = "0" * 64
        center_lon = 113.9
        center_lat = 22.5
        if type(self).drift_center and config.map_half_size_km == 6.0:
            center_lon += 0.1
        metadata = {
            "schema_version": 2,
            "source": config.source,
            "coordinate_system": "local_km",
            "projection_center_lon": center_lon,
            "projection_center_lat": center_lat,
            "selected_days": list(config.dates),
            "raw_file_sha256": raw_sha256,
            "data_config": {
                **asdict(config),
                "dates": list(config.dates),
            },
        }
        (output_dir / "metadata.json").write_text(
            json.dumps(metadata),
            encoding="utf-8",
        )

        h_count = config.candidate_count - int(type(self).wrong_h_count)
        sites = [
            {
                "site_id": f"{site_type}{index:03d}",
                "type": site_type,
                "x_km": 0.1,
                "y_km": -0.1,
                "activity_rate": 1.0,
            }
            for site_type, count in (
                ("H", h_count),
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


@pytest.fixture(autouse=True)
def _reset_fake_generator() -> None:
    FakeGenerator.reset()


def _config(tmp_path: Path) -> DevelopmentCalibrationBuildConfig:
    tmp_path.mkdir(parents=True, exist_ok=True)
    raw_path = tmp_path / "tsl-development.csv"
    raw_path.write_text("held-out development input", encoding="utf-8")
    return DevelopmentCalibrationBuildConfig(
        output_root=tmp_path / "development-calibration",
        tsl_raw_path=raw_path,
    )


def test_registered_development_design_is_disjoint_and_cost_ratio_scaled() -> None:
    assert DEVELOPMENT_LAYOUT_SEED_ORDER == (1729, 2718)
    assert not set(DEVELOPMENT_LAYOUT_SEED_ORDER) & set(LOCKED_LAYOUT_SEED_ORDER)
    plans = development_instance_plans()
    assert len(plans) == 6
    assert all(plan.block is DEVELOPMENT_DEMAND_BLOCK for plan in plans)
    assert all(not plan.block.locked for plan in plans)
    assert [
        (plan.scale.scale, plan.scale.candidate_h, plan.layout_seed) for plan in plans
    ] == [
        (scale, candidate_h, seed)
        for scale, candidate_h in (("small", 23), ("medium", 51), ("large", 90))
        for seed in DEVELOPMENT_LAYOUT_SEED_ORDER
    ]
    assert DEVELOPMENT_DEMAND_BLOCK.block_id not in {
        block.block_id for block in LOCKED_DEMAND_BLOCKS
    }

    assert len(DEVELOPMENT_CALIBRATION_PROFILES) == 24
    assert len(
        {profile.profile_id for profile in DEVELOPMENT_CALIBRATION_PROFILES}
    ) == 24
    assert {
        (
            profile.demand_scale,
            profile.feeder_rider_cost_factor,
            profile.base_locker_fixed_cost_per_demand_unit,
        )
        for profile in DEVELOPMENT_CALIBRATION_PROFILES
    } == {
        (demand, feeder, base_cost)
        for demand in DEVELOPMENT_DEMAND_SCALE_ORDER
        for feeder in DEVELOPMENT_FEEDER_FACTOR_ORDER
        for base_cost in (
            DEVELOPMENT_BASE_LOCKER_FIXED_COST_PER_DEMAND_UNIT_ORDER
        )
    }
    locked_profile_ids = {
        profile.profile_id for profile in LOCKED_QUALIFICATION_PROFILES
    }
    for profile in DEVELOPMENT_CALIBRATION_PROFILES:
        solve_profile = profile.solve_profile()
        assert profile.profile_id not in locked_profile_ids
        assert profile.locker_fixed_cost_per_hour == (
            profile.demand_scale
            * profile.base_locker_fixed_cost_per_demand_unit
        )
        assert solve_profile.display_origin_choices == (
            DEVELOPMENT_DISPLAY_ORIGIN_CHOICES
        )
        assert solve_profile.general_corridor_choices == (
            DEVELOPMENT_GENERAL_CORRIDOR_CHOICES
        )
        assert solve_profile.expected_screening_is_exact is (
            DEVELOPMENT_EXPECTED_SCREENING_IS_EXACT
        )
        assert not solve_profile.adds_forced_opening_constraint
        effective = profile.effective_model_config()
        assert effective.display_origin_choices == 8
        assert effective.general_corridor_choices == 10
        assert effective.locker_fixed_cost_per_hour == (
            profile.locker_fixed_cost_per_hour
        )


def test_builder_generates_each_physical_instance_once_and_writes_job_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    gurobi_import_attempts: list[str] = []
    real_import = builtins.__import__

    def guarded_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "gurobipy" or name.startswith("gurobipy."):
            gurobi_import_attempts.append(name)
            raise AssertionError("calibration builder may not import Gurobi")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    result = build_development_calibration_manifest(
        config,
        generator_factory=FakeGenerator,
    )

    assert gurobi_import_attempts == []
    assert len(FakeGenerator.calls) == 6
    assert result.physical_instance_count == 6
    assert result.logical_job_count == 144
    assert len(result.instance_directories) == 6
    artifact = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    reported_sha256 = artifact.pop("development_calibration_manifest_sha256")
    assert reported_sha256 == result.manifest_sha256
    assert canonical_sha256(artifact) == reported_sha256
    assert artifact["development_only"] is True
    assert artifact["eligible_for_formal_pool"] is False
    assert artifact["input_artifact_only"] is True
    assert artifact["contains_solver_results"] is False
    assert artifact["formal_contract"] == {
        "locked_demand_blocks_used": False,
        "locked_layout_seeds_used": False,
        "locked_qualification_profiles_modified": False,
        "candidate_pool_created": False,
        "excluded_locked_demand_block_ids": [
            block.block_id for block in LOCKED_DEMAND_BLOCKS
        ],
        "excluded_locked_layout_seeds": list(LOCKED_LAYOUT_SEED_ORDER),
    }
    raw_sha256 = hashlib.sha256(config.resolved_raw_path.read_bytes()).hexdigest()
    assert artifact["raw_input"] == {
        "source": "tsl",
        "path": str(config.resolved_raw_path),
        "sha256": raw_sha256,
        "size_bytes": config.resolved_raw_path.stat().st_size,
    }
    assert artifact["screening_contract"] == {
        "display_origin_choices": 8,
        "general_corridor_choices": 10,
        "expected_screening_is_exact": False,
    }
    grid = artifact["calibration_grid"]
    assert grid["profile_count"] == 24
    assert grid["demand_scale"] == [1.0, 2.0, 3.0, 4.0]
    assert grid["feeder_rider_cost_factor"] == [0.3, 0.35]
    assert grid["base_locker_fixed_cost_per_demand_unit"] == [20.0, 24.0, 28.0]
    for profile_payload, profile in zip(
        grid["profiles"],
        DEVELOPMENT_CALIBRATION_PROFILES,
        strict=True,
    ):
        assert profile_payload["base_locker_fixed_cost_per_demand_unit"] == (
            profile.base_locker_fixed_cost_per_demand_unit
        )
        assert profile_payload["locker_fixed_cost_per_hour"] == (
            profile.demand_scale
            * profile.base_locker_fixed_cost_per_demand_unit
        )
        assert profile_payload["qualification_solve_profile"] == (
            profile.solve_profile().to_payload()
        )
        assert profile_payload["effective_model_config"] == json.loads(
            canonical_json(to_dict(profile.effective_model_config()))
        )
        assert profile_payload["effective_model_config_sha256"] == (
            profile.effective_model_config_sha256()
        )

    assert artifact["physical_instance_count"] == 6
    assert len(artifact["physical_instances"]) == 6
    for physical, call in zip(
        artifact["physical_instances"],
        FakeGenerator.calls,
        strict=True,
    ):
        instance_dir = config.output_root / physical["instance_ref"]
        instance_sha256, canonical_hashes = canonical_instance_hash(instance_dir)
        assert physical["instance_sha256"] == instance_sha256
        assert physical["canonical_file_sha256"] == canonical_hashes
        assert physical["raw_file_sha256"] == raw_sha256
        assert physical["raw_file_size_bytes"] == (
            config.resolved_raw_path.stat().st_size
        )
        expected_data_config = asdict(call)
        expected_data_config["dates"] = list(call.dates)
        assert physical["data_config"] == expected_data_config
        assert physical["development_only"] is True

    jobs = artifact["logical_job_order"]
    assert artifact["logical_job_count"] == 144
    assert [job["job_order_index"] for job in jobs] == list(range(144))
    assert len({job["job_id"] for job in jobs}) == 144
    assert [
        (job["scale"], job["layout_seed"], job["profile_order_index"])
        for job in jobs
    ] == [
        (scale, seed, profile_index)
        for scale in ("small", "medium", "large")
        for seed in DEVELOPMENT_LAYOUT_SEED_ORDER
        for profile_index in range(24)
    ]
    profile_by_id = {
        profile.profile_id: profile for profile in DEVELOPMENT_CALIBRATION_PROFILES
    }
    physical_by_id = {
        item["physical_instance_id"]: item
        for item in artifact["physical_instances"]
    }
    for job in jobs:
        profile = profile_by_id[job["profile_id"]]
        physical = physical_by_id[job["physical_instance_id"]]
        assert job["instance_ref"] == physical["instance_ref"]
        assert job["instance_sha256"] == physical["instance_sha256"]
        assert job["effective_model_config_sha256"] == (
            profile.effective_model_config_sha256()
        )
        assert job["qualification_case_sha256"] == qualification_case_sha256(
            physical["instance_sha256"],
            profile.solve_profile(),
        )
    assert "outcome" not in canonical_json(artifact)
    assert "gurobi" not in canonical_json(artifact).lower()

    with pytest.raises(FileExistsError, match="refusing to reuse"):
        build_development_calibration_manifest(
            config,
            generator_factory=FakeGenerator,
        )
    assert len(FakeGenerator.calls) == 6


def test_module_parser_and_main_are_reproducible_and_fail_closed(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = _config(tmp_path)
    parsed = build_parser().parse_args(
        [
            "--output-root",
            str(config.output_root),
            "--tsl-raw",
            str(config.tsl_raw_path),
            "--candidate-hours",
            "9",
            "22",
            "--site-grid-km",
            "0.5",
        ]
    )
    assert parsed.candidate_hours == [9, 22]
    assert parsed.site_grid_km == 0.5

    argv = [
        "--output-root",
        str(config.output_root),
        "--tsl-raw",
        str(config.tsl_raw_path),
    ]
    assert main(argv, generator_factory=FakeGenerator) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["physical_instance_count"] == 6
    assert summary["logical_job_count"] == 144
    assert Path(summary["manifest_path"]) == (
        config.output_root / "development_calibration_manifest.json"
    )
    assert len(summary["manifest_sha256"]) == 64
    assert len(FakeGenerator.calls) == 6

    with pytest.raises(SystemExit) as exit_info:
        main(argv, generator_factory=FakeGenerator)
    assert exit_info.value.code == 2
    assert "refusing to reuse" in capsys.readouterr().err
    assert len(FakeGenerator.calls) == 6


def test_builder_refuses_partial_root_before_generation(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config.output_root.mkdir()
    (config.output_root / "partial-instance").mkdir()

    with pytest.raises(FileExistsError, match="partial artifacts"):
        build_development_calibration_manifest(
            config,
            generator_factory=FakeGenerator,
        )
    assert FakeGenerator.calls == []


@pytest.mark.parametrize("failure", ("raw_hash", "h_count", "center"))
def test_builder_fails_closed_on_generated_provenance_or_geometry_drift(
    tmp_path: Path,
    failure: str,
) -> None:
    config = _config(tmp_path / failure)
    if failure == "raw_hash":
        FakeGenerator.wrong_raw_hash = True
        expected = "raw_file_sha256"
    elif failure == "h_count":
        FakeGenerator.wrong_h_count = True
        expected = "candidate counts mismatch"
    else:
        FakeGenerator.drift_center = True
        expected = "does not share one projection center"

    with pytest.raises(ValueError, match=expected):
        build_development_calibration_manifest(
            config,
            generator_factory=FakeGenerator,
        )
    assert not (config.output_root / "development_calibration_manifest.json").exists()


def test_builder_rejects_formal_block_or_seed_collision_before_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    block_config = _config(tmp_path / "block")
    monkeypatch.setattr(
        calibration_module,
        "DEVELOPMENT_DEMAND_BLOCK",
        LOCKED_DEMAND_BLOCKS[0],
    )
    with pytest.raises(AssertionError, match="demand block"):
        build_development_calibration_manifest(
            block_config,
            generator_factory=FakeGenerator,
        )
    assert not block_config.output_root.exists()
    assert FakeGenerator.calls == []

    monkeypatch.setattr(
        calibration_module,
        "DEVELOPMENT_DEMAND_BLOCK",
        DEVELOPMENT_DEMAND_BLOCK,
    )
    monkeypatch.setattr(
        calibration_module,
        "DEVELOPMENT_LAYOUT_SEED_ORDER",
        (LOCKED_LAYOUT_SEED_ORDER[0], 2718),
    )
    seed_config = _config(tmp_path / "seed")
    with pytest.raises(AssertionError, match="layout seeds overlap"):
        build_development_calibration_manifest(
            seed_config,
            generator_factory=FakeGenerator,
        )
    assert not seed_config.output_root.exists()
    assert FakeGenerator.calls == []


def test_profile_payload_reconstructs_nonformal_screened_solve_profile() -> None:
    profile = DEVELOPMENT_CALIBRATION_PROFILES[-1]
    payload = profile.to_payload()
    reconstructed = QualificationSolveProfile(**payload["qualification_solve_profile"])
    assert reconstructed == profile.solve_profile()
    assert reconstructed not in LOCKED_QUALIFICATION_PROFILES
