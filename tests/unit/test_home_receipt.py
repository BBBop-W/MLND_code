"""Courier-home deadlines and lossless input migration; no optimizer runs."""

from dataclasses import replace
import json

import numpy as np
import pytest

from src.data.instances import home_receipt_mean, to_planning_data
from src.data.standard import SCHEMA, export_standard, load_standard
from src.experiments.storage import load_inputs, save_inputs
from src.model.resources import build_initial_segment, PathInfeasible
from test_operations import instance, single_solution
from src.operations.validation import build_operations
from test_standard import SNAPSHOT, standard_market  # noqa: F401


def test_home_deadline_is_not_the_mixed_receipt_deadline():
    data = instance(home_receipt_mean=.3, deadline=.35)
    legs = ((0, 1), (1, 2))
    with pytest.raises(PathInfeasible, match="receipt-time"):
        build_initial_segment(data, (0, 0), 0, 0, 1, legs)
    mixed = replace(data, home_receipt_mean=.08)
    assert build_initial_segment(mixed, (0, 0), 0, 0, 1, legs) is not None
    result = single_solution(mixed)
    with pytest.raises(ValueError, match="mean deadline"):
        build_operations(data, result)
    # Home time may be shorter than the mixture's locker-removal moment.
    assert instance(receipt_mean=.5, removal_mean=.4, home_receipt_mean=.2)


def test_legacy_mixture_needs_identifying_courier_parameters():
    raw = dict(demand=[1, 2], n_h=2, receipt_mean=.4, pickup_share=.5)
    with pytest.raises(ValueError, match="frozen courier"):
        home_receipt_mean(raw)
    raw["metadata"] = {"resolved_config": {"terminal": {"courier_receipt_minutes": [6, 12]}}}
    np.testing.assert_allclose(home_receipt_mean(raw), [[.1, .2], [.1, .2]])
    raw["metadata"]["resolved_config"]["terminal"] = {"courier_receipt_minutes": None, "courier_removal_minutes": 3}
    raw["last_rider_mean"] = np.array([[.1, .2], [.3, .4]])
    np.testing.assert_allclose(home_receipt_mean(raw), raw["last_rider_mean"]+.05)
    raw["home_receipt_mean"] = .7
    np.testing.assert_allclose(home_receipt_mean(raw), .7)
    raw["home_receipt_mean"] = -1
    with pytest.raises(ValueError, match="nonnegative"):
        home_receipt_mean(raw)


def test_new_schema_explicit_home_round_trip(tmp_path, standard_market):
    raw = load_inputs(SNAPSHOT)
    home = home_receipt_mean(raw)
    assert not np.allclose(home, raw["receipt_mean"])
    raw["home_receipt_mean"] = home
    snapshot = tmp_path / "snapshot"
    save_inputs(raw, snapshot)
    target = tmp_path / "standard"
    report = export_standard(snapshot, standard_market, target)
    assert report["passed"] and report["reference_exact_match"]
    assert report["schema"] == SCHEMA
    restored = load_standard(target)
    np.testing.assert_array_equal(restored["home_receipt_mean"], home)
    np.testing.assert_array_equal(restored["receipt_mean"], raw["receipt_mean"])
    np.testing.assert_array_equal(to_planning_data(restored).home_receipt_mean, home)
    assert json.loads((target / "manifest.json").read_text())["schema"] == SCHEMA
