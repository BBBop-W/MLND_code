"""Data-only scale-ladder checks; these tests never run optimization."""

from copy import deepcopy
import json

import numpy as np
import pandas as pd
import pytest

from src.data.performance_orders import read_orders
from src.data.prepare import prepare_orders
from src.data.scale_ladder import DEFAULT_ANCHOR, ladder_specs, prepare_scale_ladder
from src.data.standard import INPUT_FILES, load_standard
from src.experiments.storage import load_inputs, sha256


def test_scale_geometry_holds_h_density_constant():
    specs, metadata = ladder_specs()
    assert [s["instance_id"] for s in specs] == [f"L{n:02d}" for n in range(1, 11)]
    assert [(s["n_l"], s["n_h"], s["n_od"]) for s in specs] == [
        (n, 10 * n, 25 * n) for n in range(1, 11)
    ]
    np.testing.assert_allclose([s["crop_area_km2"] for s in specs], np.arange(1, 11) * 50 / 3)
    np.testing.assert_allclose([s["candidate_h_density_per_km2"] for s in specs], 0.6)
    assert all(s["crop_center_local_km"] == metadata["crop_center_local_km"] for s in specs)
    assert [s["level"] for s in specs if s["exact_frozen_anchor"]] == [2]
    assert specs[1]["crop_half_size_km"] == metadata["crop_half_size_km"]


@pytest.mark.parametrize("levels", [[], [1, 1], [0], [11], [True], [1.5]])
def test_bad_scale_levels_rejected(levels):
    with pytest.raises(ValueError, match="Levels"):
        ladder_specs(levels=levels)


def test_frozen_anchor_is_copied_exactly_without_preparing_or_solving(tmp_path, monkeypatch):
    import src.data.scale_ladder as module

    def forbidden(*_args, **_kwargs):
        raise AssertionError("Frozen level must not be regenerated")

    monkeypatch.setattr(module, "prepare_orders", forbidden)
    monkeypatch.setattr(module, "read_orders", forbidden)
    before = {name: sha256(DEFAULT_ANCHOR / name) for name in (*INPUT_FILES, "manifest.json")}
    result = prepare_scale_ladder(tmp_path / "ladder", levels=[2])
    target = tmp_path / "ladder/instances/L02"
    assert result["solver_invoked"] is False
    assert result["status"] == "prepared_validated_not_solved"
    assert result["levels"][0]["demand_orders"] == 17382
    assert {name: sha256(target / name) for name in before} == before
    assert {name: sha256(DEFAULT_ANCHOR / name) for name in before} == before
    for case, k in (("K1", 1), ("K2", 2)):
        assert load_standard(target, case=case)["max_orders"] == k
    audit = json.loads((tmp_path / "ladder/provenance/L02/preparation.json").read_text())
    assert audit["membership_audit"]["cluster_counts_match"]
    assert audit["membership_audit"]["raw_order_rows"] == 17382
    assert not list((tmp_path / "ladder").glob(".scale-prepare-*"))
    with pytest.raises(FileExistsError):
        prepare_scale_ladder(tmp_path / "ladder", levels=[2])


def test_modified_raw_source_rejected_before_generation(tmp_path):
    raw = tmp_path / "different.csv"
    raw.write_text("not the frozen TSL data\n")
    with pytest.raises(ValueError, match="differs from the frozen"):
        prepare_scale_ladder(tmp_path / "ladder", levels=[1], raw_path=raw)
    assert not (tmp_path / "ladder/instances/L01").exists()


def test_cached_observed_orders_are_identical_and_not_mutated(tmp_path, monkeypatch):
    rows = []
    for day in ("1", "2"):
        for i in range(8):
            rows.append(dict(
                order_seq_id=f"{day}-{i}", dt_seq_id=day,
                customer_pay_time="12:00", fetch_time="12:03", arrived_time=f"12:{10+i:02d}",
                pickup_longitude=113.9 + i * .001, pickup_latitude=22.5 + i * .0003,
                dropoff_longitude=113.902 + i * .0015, dropoff_latitude=22.502 + i * .0004,
            ))
    raw = tmp_path / "raw.csv"
    pd.DataFrame(rows).to_csv(raw, index=False)
    cached = read_orders("nanshan", path=raw)
    frame_before, meta_before = cached[0].copy(deep=True), deepcopy(cached[1])
    config = {"compatibility": {"mode": "none", "require_feasible_path": False}}
    prepare_orders("nanshan", tmp_path / "uncached", n_l=1, n_h=3, n_od=2, raw_path=raw, config=config)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("Cached preparation reread the raw source")

    monkeypatch.setattr("src.data.prepare.read_orders", forbidden)
    prepare_orders("nanshan", tmp_path / "cached", n_l=1, n_h=3, n_od=2,
                   raw_path=raw, config=config, prepared_orders=cached)
    left, right = load_inputs(tmp_path / "uncached"), load_inputs(tmp_path / "cached")
    assert left.keys() == right.keys()
    for key, value in left.items():
        if isinstance(value, np.ndarray):
            np.testing.assert_array_equal(value, right[key])
        else:
            assert value == right[key]
    pd.testing.assert_frame_equal(cached[0], frame_before)
    assert cached[1] == meta_before
    with pytest.raises(ValueError, match="different source"):
        prepare_orders("tsl", tmp_path / "wrong", raw_path=raw, prepared_orders=cached)
