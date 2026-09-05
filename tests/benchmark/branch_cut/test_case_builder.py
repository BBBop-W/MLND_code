from __future__ import annotations

import csv
from dataclasses import asdict, replace
import json
import math
from pathlib import Path

import pytest

from src.benchmark.branch_cut.case_builder import (
    CANONICAL_INSTANCE_FILES,
    DEVELOPMENT_DEMAND_BLOCK,
    LOCKED_DEMAND_BLOCKS,
    LOCKED_LAYOUT_SEED_ORDER,
    LOCKED_QUALIFICATION_PROFILES,
    LOCKED_SCALE_SPECS,
    OVERCOMPLETE_SELECTION_RULE,
    QUALIFICATION_MODEL_IDENTITY_SCHEMA,
    CaseBuildConfig,
    QualificationSolveProfile,
    build_locked_candidate_pool,
    canonical_instance_hash,
    locked_case_plans,
    locked_instance_plans,
    locked_replacement_order,
    qualification_case_sha256,
)
from src.benchmark.branch_cut.hashing import canonical_sha256
from src.benchmark.branch_cut.manifest import FreezeMetadata, freeze_manifest
from src.benchmark.branch_cut.qualification import (
    QualificationEvidence,
    QualificationPolicy,
    QualificationReason,
    assess_qualification,
)
from src.config import DataConfig, ModelConfig, to_dict


def _write_csv(
    path: Path, fieldnames: list[str], rows: list[dict[str, object]]
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


class FakeGenerator:
    calls: list[DataConfig] = []
    wrong_h_count = False
    drift_center = False

    @classmethod
    def reset(cls) -> None:
        cls.calls = []
        cls.wrong_h_count = False
        cls.drift_center = False

    def generate(self, config: DataConfig) -> str:
        type(self).calls.append(config)
        output_dir = Path(config.output_dir)
        center_lon = 113.9 + 0.001 * sum(len(day) for day in config.dates)
        center_lat = 22.5 + 0.001 * len(config.dates)
        if type(self).drift_center and config.map_half_size_km == 6.0:
            center_lon += 0.1
        metadata = {
            "schema_version": 2,
            "source": config.source,
            "coordinate_system": "local_km",
            "projection_center_lon": center_lon,
            "projection_center_lat": center_lat,
            "selected_days": list(config.dates),
            "raw_file_sha256": "f" * 64,
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
        sites = []
        for site_type, count in (
            ("H", h_count),
            ("L", config.num_bases),
            ("E", config.num_swaps),
        ):
            sites.extend(
                {
                    "site_id": f"{site_type}{index:03d}",
                    "type": site_type,
                    "x_km": 0.1,
                    "y_km": -0.1,
                    "activity_rate": 1.0,
                }
                for index in range(1, count + 1)
            )
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


def _config(tmp_path: Path) -> CaseBuildConfig:
    tmp_path.mkdir(parents=True, exist_ok=True)
    nanshan = tmp_path / "nanshan.csv"
    tsl = tmp_path / "tsl.csv"
    nanshan.write_text("fake nanshan", encoding="utf-8")
    tsl.write_text("fake tsl", encoding="utf-8")
    return CaseBuildConfig(
        output_root=tmp_path / "locked-pool",
        nanshan_raw_path=nanshan,
        tsl_raw_path=tsl,
    )


def _qualification_profile(
    **overrides: object,
) -> QualificationSolveProfile:
    values: dict[str, object] = {
        "profile_id": "test-profile",
        "demand_scale": 3.0,
        "feeder_rider_cost_factor": 0.35,
        "locker_fixed_cost_per_hour": 1.0,
        "display_origin_choices": None,
        "general_corridor_choices": None,
        "expected_screening_is_exact": True,
        "interpretation": "test qualification profile",
        "adds_forced_opening_constraint": False,
    }
    values.update(overrides)
    return QualificationSolveProfile(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "missing_field",
    [
        "locker_fixed_cost_per_hour",
        "display_origin_choices",
        "general_corridor_choices",
        "expected_screening_is_exact",
    ],
)
def test_qualification_profile_rejects_each_missing_model_axis(
    missing_field: str,
) -> None:
    payload = _qualification_profile().to_payload()
    del payload[missing_field]

    with pytest.raises(TypeError):
        QualificationSolveProfile(**payload)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("field_name", "bad_value"),
    [
        ("demand_scale", True),
        ("feeder_rider_cost_factor", False),
        ("locker_fixed_cost_per_hour", True),
        ("locker_fixed_cost_per_hour", -1.0),
        ("locker_fixed_cost_per_hour", math.inf),
        ("locker_fixed_cost_per_hour", math.nan),
        ("display_origin_choices", True),
        ("display_origin_choices", 0),
        ("display_origin_choices", 1.0),
        ("general_corridor_choices", False),
        ("general_corridor_choices", 0),
        ("general_corridor_choices", 1.0),
        ("expected_screening_is_exact", 1),
    ],
)
def test_qualification_profile_strictly_validates_model_axes(
    field_name: str,
    bad_value: object,
) -> None:
    with pytest.raises(ValueError):
        _qualification_profile(**{field_name: bad_value})


def test_qualification_profile_normalizes_and_hashes_complete_model_config() -> None:
    profile = _qualification_profile(
        demand_scale=3,
        feeder_rider_cost_factor=1,
        locker_fixed_cost_per_hour=24,
        display_origin_choices=8,
        general_corridor_choices=10,
        expected_screening_is_exact=False,
    )

    assert profile.demand_scale == 3.0
    assert type(profile.demand_scale) is float
    assert profile.feeder_rider_cost_factor == 1.0
    assert type(profile.feeder_rider_cost_factor) is float
    assert profile.locker_fixed_cost_per_hour == 24.0
    assert type(profile.locker_fixed_cost_per_hour) is float
    effective_config = profile.effective_model_config()
    assert isinstance(effective_config, ModelConfig)
    assert effective_config.feeder_rider_cost_factor == 1.0
    assert effective_config.locker_fixed_cost_per_hour == 24.0
    assert effective_config.display_origin_choices == 8
    assert effective_config.general_corridor_choices == 10
    assert profile.effective_model_config_sha256() == canonical_sha256(
        to_dict(effective_config)
    )
    assert (
        replace(
            profile,
            demand_scale=4.0,
        ).effective_model_config_sha256()
        == profile.effective_model_config_sha256()
    )


@pytest.mark.parametrize(
    "overrides",
    [
        {"profile_id": "different-profile-id"},
        {"demand_scale": 4.0},
        {"feeder_rider_cost_factor": 0.30},
        {"locker_fixed_cost_per_hour": 24.0},
        {"display_origin_choices": 8},
        {"general_corridor_choices": 10},
        {"expected_screening_is_exact": False},
        {"interpretation": "different interpretation"},
    ],
)
def test_qualification_case_hash_changes_with_every_profile_axis(
    overrides: dict[str, object],
) -> None:
    instance_sha256 = "a" * 64
    baseline = _qualification_profile()

    assert qualification_case_sha256(
        instance_sha256,
        _qualification_profile(**overrides),
    ) != qualification_case_sha256(instance_sha256, baseline)


def test_same_physical_instance_has_distinct_locked_profile_case_hashes() -> None:
    instance_sha256 = "a" * 64

    assert len(
        {
            qualification_case_sha256(instance_sha256, profile)
            for profile in LOCKED_QUALIFICATION_PROFILES
        }
    ) == len(LOCKED_QUALIFICATION_PROFILES)


def test_fixed_design_has_locked_scales_blocks_seeds_and_profile_ladder() -> None:
    assert [
        (
            spec.scale,
            spec.map_half_size_km,
            spec.candidate_h,
            spec.od_grid_km,
            spec.candidate_l,
            spec.candidate_e,
        )
        for spec in LOCKED_SCALE_SPECS
    ] == [
        ("small", 4.0, 23, 1.0, 2, 3),
        ("medium", 6.0, 51, 0.75, 3, 5),
        ("large", 8.0, 90, 0.5, 6, 9),
    ]
    assert LOCKED_LAYOUT_SEED_ORDER == (11, 29, 47, 83)
    assert len(LOCKED_DEMAND_BLOCKS) == 6
    assert [
        (block.source, block.dates, block.demand_hour_start, block.demand_hour_end)
        for block in LOCKED_DEMAND_BLOCKS
    ] == [
        ("nanshan", ("4", "5"), 12, 13),
        ("nanshan", ("6", "7"), 12, 13),
        ("nanshan", ("4", "5", "6", "7"), 18, 19),
        ("tsl", ("20221021", "20221022"), 12, 13),
        ("tsl", ("20221023", "20221024"), 12, 13),
        (
            "tsl",
            ("20221021", "20221022", "20221023", "20221024"),
            18,
            19,
        ),
    ]
    assert len(locked_instance_plans()) == 72
    assert len(locked_case_plans()) == 288
    assert [
        (
            profile.profile_id,
            profile.demand_scale,
            profile.feeder_rider_cost_factor,
            profile.locker_fixed_cost_per_hour,
            profile.display_origin_choices,
            profile.general_corridor_choices,
            profile.expected_screening_is_exact,
            profile.adds_forced_opening_constraint,
        )
        for profile in LOCKED_QUALIFICATION_PROFILES
    ] == [
        (
            "demand-3x-feeder-035-fc-001-display-all-general-all-exact-true",
            3.0,
            0.35,
            1.0,
            None,
            None,
            True,
            False,
        ),
        (
            "demand-4x-feeder-035-fc-001-display-all-general-all-exact-true",
            4.0,
            0.35,
            1.0,
            None,
            None,
            True,
            False,
        ),
        (
            "demand-3x-feeder-030-fc-001-display-all-general-all-exact-true",
            3.0,
            0.30,
            1.0,
            None,
            None,
            True,
            False,
        ),
        (
            "demand-4x-feeder-030-fc-001-display-all-general-all-exact-true",
            4.0,
            0.30,
            1.0,
            None,
            None,
            True,
            False,
        ),
    ]
    assert locked_replacement_order()[:5] == (
        (
            "demand-3x-feeder-035-fc-001-display-all-general-all-exact-true",
            11,
        ),
        (
            "demand-3x-feeder-035-fc-001-display-all-general-all-exact-true",
            29,
        ),
        (
            "demand-3x-feeder-035-fc-001-display-all-general-all-exact-true",
            47,
        ),
        (
            "demand-3x-feeder-035-fc-001-display-all-general-all-exact-true",
            83,
        ),
        (
            "demand-4x-feeder-035-fc-001-display-all-general-all-exact-true",
            11,
        ),
    )
    assert [profile.interpretation for profile in LOCKED_QUALIFICATION_PROFILES] == [
        "threefold order-rate scenario with a moderately efficient contracted feeder",
        "fourfold order-rate scenario with a moderately efficient contracted feeder",
        "threefold order-rate scenario with a more efficient contracted feeder",
        "fourfold order-rate scenario with a more efficient contracted feeder",
    ]
    assert all(block.locked for block in LOCKED_DEMAND_BLOCKS)
    assert not DEVELOPMENT_DEMAND_BLOCK.locked
    assert DEVELOPMENT_DEMAND_BLOCK.dates == (
        "20221017",
        "20221018",
        "20221019",
        "20221020",
    )
    assert DEVELOPMENT_DEMAND_BLOCK.block_id not in {
        plan.instance_plan.block.block_id for plan in locked_case_plans()
    }


def test_builder_creates_real_config_contract_hashes_and_exclusive_pool(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    assert config.pool_label == "mlnd-real-data-overcomplete-qualification-v5"
    result = build_locked_candidate_pool(
        config,
        generator_factory=FakeGenerator,
    )

    assert len(FakeGenerator.calls) == 72
    assert len(result.pool.candidates) == 288
    assert result.pool.candidate_seed_order == (11, 29, 47, 83)
    assert result.pool_path == config.output_root / "pool.json"
    artifact = json.loads(result.pool_path.read_text(encoding="utf-8"))
    assert artifact["artifact_schema_version"] == 2
    assert artifact["pool_sha256"] == result.pool.sha256
    assert artifact["pool"]["pool_label"] == (
        "mlnd-real-data-overcomplete-qualification-v5"
    )
    assert artifact["locked_qualification_profiles"] == [
        profile.to_payload() for profile in LOCKED_QUALIFICATION_PROFILES
    ]
    first_candidate = result.pool.candidates[0]
    changed_pool_hashes = set()
    for field_name, value in (
        ("locker_fixed_cost_per_hour", 24.0),
        ("display_origin_choices", 8),
        ("general_corridor_choices", 10),
    ):
        changed_parameters = dict(first_candidate.generator_parameters)
        changed_profile = dict(changed_parameters["qualification_solve_profile"])
        changed_profile[field_name] = value
        changed_parameters["qualification_solve_profile"] = changed_profile
        changed_candidate = replace(
            first_candidate,
            generator_parameters=changed_parameters,
        )
        changed_pool_hashes.add(
            replace(
                result.pool,
                candidates=(changed_candidate, *result.pool.candidates[1:]),
            ).sha256
        )
    missing_parameters = dict(first_candidate.generator_parameters)
    missing_profile = dict(missing_parameters["qualification_solve_profile"])
    del missing_profile["display_origin_choices"]
    missing_parameters["qualification_solve_profile"] = missing_profile
    missing_candidate = replace(
        first_candidate,
        generator_parameters=missing_parameters,
    )
    missing_pool_hash = replace(
        result.pool,
        candidates=(missing_candidate, *result.pool.candidates[1:]),
    ).sha256
    assert result.pool.sha256 not in changed_pool_hashes
    assert len(changed_pool_hashes) == 3
    assert missing_pool_hash != result.pool.sha256
    assert artifact["canonical_instance_files"] == list(CANONICAL_INSTANCE_FILES)
    assert artifact["development_block_excluded_from_pool"]["locked"] is False
    assert (
        artifact["selection_protocol"]["selection_uses_algorithm_performance"] is False
    )
    assert artifact["selection_protocol"]["forced_opening_constraints_added"] is False
    assert len(artifact["selection_protocol"]["replacement_order"]) == 16
    first_stratum = result.pool.ordered_candidates()[:16]
    assert [
        candidate.generator_parameters["pre_registered_selection"][
            "replacement_rank_within_stratum"
        ]
        for candidate in first_stratum
    ] == list(range(16))
    assert {
        candidate.generator_parameters["qualification_solve_profile"][
            "feeder_rider_cost_factor"
        ]
        for candidate in first_stratum
    } == {0.35, 0.30}

    scale_counts = {scale: 0 for scale in ("small", "medium", "large")}
    source_counts = {source: 0 for source in ("nanshan", "tsl")}
    seed_counts = {seed: 0 for seed in LOCKED_LAYOUT_SEED_ORDER}
    for candidate in result.pool.candidates:
        scale_counts[candidate.scale] += 1
        source_counts[candidate.generator_parameters["source"]] += 1
        seed_counts[candidate.layout_seed] += 1
        instance_dir = result.case_directories[candidate.case_id]
        digest, file_hashes = canonical_instance_hash(instance_dir)
        assert digest == candidate.instance_sha256
        assert file_hashes == candidate.generator_parameters["canonical_file_sha256"]
        assert candidate.generator_parameters["orders_sampled"] is False
        assert candidate.generator_parameters["demand_rescaled"] is False
        selection = candidate.generator_parameters["pre_registered_selection"]
        assert selection["selection_uses_algorithm_performance"] is False
        assert selection["forced_opening_constraints_added"] is False
        profile_payload = candidate.generator_parameters["qualification_solve_profile"]
        profile = QualificationSolveProfile(**profile_payload)
        assert profile.locker_fixed_cost_per_hour == 1.0
        assert profile.display_origin_choices is None
        assert profile.general_corridor_choices is None
        assert profile.expected_screening_is_exact is True
        assert profile.adds_forced_opening_constraint is False
        assert candidate.generator_parameters["qualification_model_identity"] == {
            "schema": QUALIFICATION_MODEL_IDENTITY_SCHEMA,
            "effective_model_config_sha256": profile.effective_model_config_sha256(),
            "qualification_case_sha256": qualification_case_sha256(
                candidate.instance_sha256,
                profile,
            ),
            "expected_screening_is_exact": True,
        }
        if candidate.scale == "large":
            assert selection["large_scale_open_h_qualification_target"] == [50, 60]
        else:
            assert selection["large_scale_open_h_qualification_target"] is None
    assert scale_counts == {"small": 96, "medium": 96, "large": 96}
    assert source_counts == {"nanshan": 144, "tsl": 144}
    assert seed_counts == {11: 72, 29: 72, 47: 72, 83: 72}

    first_twelve = FakeGenerator.calls[:12]
    assert [call.candidate_count for call in first_twelve] == [
        23,
        23,
        23,
        23,
        51,
        51,
        51,
        51,
        90,
        90,
        90,
        90,
    ]
    assert [call.seed for call in first_twelve] == [11, 29, 47, 83] * 3
    assert [call.map_half_size_km for call in first_twelve] == [4] * 4 + [6] * 4 + [
        8
    ] * 4
    assert [call.od_grid_km for call in first_twelve] == [1.0] * 4 + [0.75] * 4 + [
        0.5
    ] * 4
    assert all(call.dates == ("4", "5") for call in first_twelve)
    assert all(call.demand_hour_start == 12 for call in first_twelve)

    with pytest.raises(FileExistsError, match="refusing to reuse"):
        build_locked_candidate_pool(config, generator_factory=FakeGenerator)
    assert len(FakeGenerator.calls) == 72


def test_builder_refuses_existing_partial_root_before_generation(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    config.output_root.mkdir()
    (config.output_root / "partial-case").mkdir()

    with pytest.raises(FileExistsError, match="partial artifacts"):
        build_locked_candidate_pool(config, generator_factory=FakeGenerator)
    assert FakeGenerator.calls == []


def test_generated_overcomplete_pool_freezes_first_qualified_replacements(
    tmp_path: Path,
) -> None:
    result = build_locked_candidate_pool(
        _config(tmp_path),
        generator_factory=FakeGenerator,
    )
    decisions = []
    for candidate in result.pool.ordered_candidates():
        selection = candidate.generator_parameters["pre_registered_selection"]
        rank = selection["replacement_rank_within_stratum"]
        if rank > 2:
            continue
        candidate_h = candidate.generator_parameters["candidate_counts"]["H"]
        open_h = {"small": 8, "medium": 20, "large": 55}[candidate.scale]
        model_identity = candidate.generator_parameters["qualification_model_identity"]
        decision = assess_qualification(
            QualificationEvidence(
                case_id=candidate.case_id,
                scale=candidate.scale,
                courier_only_objective=100_000.0,
                witness_objective=99_800.0,
                total_demand_per_hour=10_000.0,
                witness_service_per_hour=500.0,
                open_facilities=open_h + 3,
                positive_cost_open_facilities=open_h + 3,
                candidate_h=candidate_h,
                open_h=open_h,
                witness_feasible=True,
                qualification_run_id=f"qual-{candidate.case_id}",
                instance_sha256=candidate.instance_sha256,
                active_h=open_h,
                active_h_threshold_per_hour=0.1,
                qualification_case_sha256=model_identity["qualification_case_sha256"],
                effective_model_config_sha256=model_identity[
                    "effective_model_config_sha256"
                ],
                screening_is_exact=model_identity["expected_screening_is_exact"],
            ),
            QualificationPolicy(large_scale_open_h_target=(50, 60)),
        )
        if rank == 0:
            decision = replace(
                decision,
                accepted=False,
                rejection_reasons=(QualificationReason.NO_OPEN_FACILITY,),
            )
        decisions.append(decision)

    manifest = freeze_manifest(
        result.pool,
        reversed(decisions),
        freeze_metadata=FreezeMetadata(
            qualification_completed_at_utc="2026-08-27T01:00:00Z",
            manifest_frozen_at_utc="2026-08-27T01:05:00Z",
            selection_rule=OVERCOMPLETE_SELECTION_RULE,
        ),
    )

    assert len(manifest.cases) == 36
    assert {
        case.candidate.generator_parameters["pre_registered_selection"][
            "replacement_rank_within_stratum"
        ]
        for case in manifest.cases
    } == {1, 2}
    assert len(manifest.artifact_payload()["rejection_ledger"]) == 18


def test_builder_detects_candidate_count_and_center_contract_drift(
    tmp_path: Path,
) -> None:
    bad_count_config = _config(tmp_path / "bad-count")
    FakeGenerator.wrong_h_count = True
    with pytest.raises(ValueError, match="candidate counts mismatch"):
        build_locked_candidate_pool(
            bad_count_config,
            generator_factory=FakeGenerator,
        )

    FakeGenerator.reset()
    drift_config = _config(tmp_path / "drift")
    FakeGenerator.drift_center = True
    with pytest.raises(ValueError, match="does not share one map center"):
        build_locked_candidate_pool(
            drift_config,
            generator_factory=FakeGenerator,
        )


def test_canonical_instance_hash_changes_when_any_canonical_file_changes(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    result = build_locked_candidate_pool(config, generator_factory=FakeGenerator)
    instance_dir = next(iter(result.case_directories.values()))
    before, before_files = canonical_instance_hash(instance_dir)

    with (instance_dir / "periods.csv").open("a", encoding="utf-8") as handle:
        handle.write("\n")
    after, after_files = canonical_instance_hash(instance_dir)

    assert before != after
    assert before_files["metadata.json"] == after_files["metadata.json"]
    assert before_files["periods.csv"] != after_files["periods.csv"]
