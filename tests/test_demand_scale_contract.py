from __future__ import annotations

import json

import pytest

import main as cli_main
from main import _validate_instance_demand_scale, build_parser


def test_solve_rejects_nonunit_scale_before_model_construction(tmp_path) -> None:
    metadata = {"nested_scale": {"required_solve_demand_scale": 1.0}}
    (tmp_path / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    args = build_parser().parse_args(
        [
            "solve",
            "--instance-dir",
            str(tmp_path),
            "--demand-scale",
            "1.2",
        ]
    )

    with pytest.raises(
        ValueError,
        match=r"requires demand_scale=1 .* got 1\.2",
    ):
        cli_main._solve(args, str(tmp_path))


def test_nested_scale_instance_accepts_unit_demand_scale() -> None:
    metadata = {"nested_scale": {"required_solve_demand_scale": 1.0}}

    _validate_instance_demand_scale(metadata, 1.0)


def test_legacy_instance_still_accepts_positive_demand_scaling() -> None:
    _validate_instance_demand_scale({"source": "legacy"}, 1.75)


def test_top_level_demand_scale_contract_is_supported() -> None:
    metadata = {"required_solve_demand_scale": 1.0}

    with pytest.raises(ValueError, match=r"requires demand_scale=1"):
        _validate_instance_demand_scale(metadata, 0.9)
