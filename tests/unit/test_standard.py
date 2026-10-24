"""Lossless exchange and semantic rejection without running any solver."""

import csv
import json
from pathlib import Path
import shutil

import numpy as np
import pytest
from scipy.stats import poisson

from src.data.instances import to_planning_data
from src.data.standard import (
    BASE_HEADERS,
    INPUT_FILES,
    SCHEMA,
    TABLES,
    export_standard,
    load_standard,
    validate_standard,
)
from src.experiments.storage import load_inputs

ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT = ROOT / "data/benchmarks/tsl_2l20h50od"
EXAMPLE = ROOT / "examples/tsl_2l20h50od"


def _assert_exact(expected, actual):
    assert set(actual) == set(expected)
    for key, value in expected.items():
        if isinstance(value, np.ndarray):
            assert isinstance(actual[key], np.ndarray), key
            assert actual[key].dtype == value.dtype, key
            assert actual[key].shape == value.shape, key
            np.testing.assert_array_equal(actual[key], value, err_msg=key)
        else:
            assert actual[key] == value, key


def _read_table(path):
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        return reader.fieldnames, list(reader)


def _write_table(path, headers, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, headers, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _edit_json(path, mutate):
    payload = json.loads(path.read_text())
    mutate(payload)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


@pytest.fixture(scope="module")
def standard_market(tmp_path_factory):
    """Rebuild market provenance from the versioned example, not old outputs."""
    target = tmp_path_factory.mktemp("standard_market")
    for name in ("od_nodes.csv", "od_demand.csv"):
        shutil.copyfile(EXAMPLE / name, target / name)
    daily = json.loads((EXAMPLE / "metadata.json").read_text())["daily_periods"]
    _write_table(target / "periods.csv", list(daily[0]), daily)
    return target


@pytest.fixture(scope="module")
def standard_instance(tmp_path_factory, standard_market):
    target = tmp_path_factory.mktemp("standard") / "tsl_2l20h50od"
    report = export_standard(SNAPSHOT, standard_market, target)
    assert report["passed"]
    assert report["solver_run"] is False
    return target


@pytest.fixture
def mutable_instance(standard_instance, tmp_path):
    target = tmp_path / "instance"
    shutil.copytree(standard_instance, target)
    return target


def test_export_and_load_preserve_every_snapshot_field(standard_instance):
    expected = load_inputs(SNAPSHOT)
    actual = load_standard(standard_instance)
    _assert_exact(expected, actual)
    report = validate_standard(standard_instance, reference=SNAPSHOT)
    assert report["schema"] == SCHEMA
    assert report["reference_exact_match"]
    assert report["snapshot_fields"] == len(expected)
    assert report["arrays_compared"] == sum(
        isinstance(value, np.ndarray) for value in expected.values()
    )
    assert report["dimensions"] == {"n_l": 2, "n_h": 20, "n_od": 50}
    assert report["demand_per_hour"] == expected["demand"].sum()
    assert report["cases_change_max_orders_only"]
    assert report["solver_run"] is False
    manifest = json.loads((standard_instance / "manifest.json").read_text())
    assert set(manifest["files"]) == set(INPUT_FILES)


@pytest.mark.parametrize("case,k", [("K1", 1), ("K2", 2)])
def test_cases_change_only_max_orders(standard_instance, case, k):
    expected = load_inputs(SNAPSHOT)
    expected["max_orders"] = k
    actual = load_standard(standard_instance, case=case)
    _assert_exact(expected, actual)
    planning = to_planning_data(actual)
    assert planning.effective_delta == (expected["delta"] if k == 2 else 0)


def test_csv_row_order_is_independent_of_all_model_indices(mutable_instance):
    for name in INPUT_FILES:
        if not name.endswith(".csv"):
            continue
        path = mutable_instance / name
        headers, rows = _read_table(path)
        _write_table(path, headers, reversed(rows))
    _assert_exact(
        load_inputs(SNAPSHOT), load_standard(mutable_instance, verify_hashes=False)
    )


def test_export_keeps_unknown_counts_blank_and_documents_distance(standard_instance):
    _, sites = _read_table(standard_instance / "sites.csv")
    _, nodes = _read_table(standard_instance / "od_nodes.csv")
    assert all(row["activity_rate"] == "" for row in sites)
    assert all(row["raw_endpoint_count"] == "" for row in nodes)
    metadata = json.loads((standard_instance / "metadata.json").read_text())
    assert metadata["legacy_layout"] == "five-file-v2"
    assert "actual orders" in metadata["direct_distance_semantics"]
    assert (
        "not distance between representative nodes"
        in metadata["direct_distance_semantics"]
    )
    _, observed = _read_table(EXAMPLE / "od_demand.csv")
    _, exported = _read_table(standard_instance / "od_demand.csv")
    assert exported == observed


def test_exported_poisson_limits_have_strict_overflow_semantics(standard_instance):
    _, rows = _read_table(standard_instance / "coefficients/locker_levels.csv")
    for row in rows:
        count, mean, epsilon = (
            int(row["slot_counts"]),
            float(row["occupancy_limit"]),
            float(row["slot_epsilon"]),
        )
        assert mean >= 0
        assert poisson.sf(count, mean) == pytest.approx(epsilon, abs=2e-12)
        assert poisson.sf(count, mean + 1e-6) > epsilon


def test_manifest_detects_byte_level_corruption(mutable_instance):
    path = mutable_instance / "coefficients/kappa.csv"
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_standard(mutable_instance)


@pytest.mark.parametrize("name", TABLES)
@pytest.mark.parametrize("damage", ["missing", "duplicate"])
def test_coefficient_cartesian_tables_require_exactly_one_row(
    mutable_instance, name, damage
):
    path = mutable_instance / f"coefficients/{name}.csv"
    headers, rows = _read_table(path)
    if damage == "missing":
        rows.pop()
        message = "Missing coefficient row"
    else:
        rows.append(dict(rows[0]))
        message = "Duplicate coefficient row"
    _write_table(path, headers, rows)
    with pytest.raises(ValueError, match=message):
        load_standard(mutable_instance, verify_hashes=False)


@pytest.mark.parametrize("name", ["sites.csv", "od_nodes.csv", "od_demand.csv"])
@pytest.mark.parametrize("damage", ["missing", "duplicate"])
def test_geographic_tables_require_exactly_one_row_per_id(
    mutable_instance, name, damage
):
    path = mutable_instance / name
    headers, rows = _read_table(path)
    if damage == "missing":
        rows.pop()
        message = "Missing or unknown"
    else:
        rows.append(dict(rows[0]))
        message = "Duplicate"
    _write_table(path, headers, rows)
    with pytest.raises(ValueError, match=message):
        load_standard(mutable_instance, verify_hashes=False)


def test_unit_changes_are_not_silently_converted(mutable_instance):
    _edit_json(
        mutable_instance / "model.json",
        lambda model: model["units"].update(time_coefficients="minute"),
    )
    with pytest.raises(ValueError, match="schema or units"):
        load_standard(mutable_instance, verify_hashes=False)


def test_precomputed_poisson_limit_is_independently_verified(mutable_instance):
    path = mutable_instance / "coefficients/locker_levels.csv"
    headers, rows = _read_table(path)
    rows[0]["occupancy_limit"] = str(float(rows[0]["occupancy_limit"]) + 0.01)
    _write_table(path, headers, rows)
    with pytest.raises(ValueError, match="Poisson occupancy threshold mismatch"):
        load_standard(mutable_instance, verify_hashes=False)


@pytest.mark.parametrize(
    "field,value", [("formula", "P(X >= C) <= epsilon"), ("verification_atol", 1)]
)
def test_poisson_definition_and_verification_tolerance_are_fixed(
    mutable_instance, field, value
):
    _edit_json(
        mutable_instance / "model.json",
        lambda model: model["capacity_definition"].update({field: value}),
    )
    with pytest.raises(ValueError, match="capacity_definition"):
        load_standard(mutable_instance, verify_hashes=False)


def test_epsilon_must_be_the_same_for_each_hubs_levels(mutable_instance):
    path = mutable_instance / "coefficients/locker_levels.csv"
    headers, rows = _read_table(path)
    rows[0]["slot_epsilon"] = str(float(rows[0]["slot_epsilon"]) / 2)
    _write_table(path, headers, rows)
    with pytest.raises(ValueError, match="Inconsistent epsilon"):
        load_standard(mutable_instance, verify_hashes=False)


def test_boolean_link_coefficients_do_not_accept_truthy_strings(mutable_instance):
    path = mutable_instance / "coefficients/links.csv"
    headers, rows = _read_table(path)
    rows[0]["allowed_links"] = "true"
    _write_table(path, headers, rows)
    with pytest.raises(ValueError, match="Boolean CSV values must be 0 or 1"):
        load_standard(mutable_instance, verify_hashes=False)


def test_self_link_is_rejected_instead_of_silently_normalized(mutable_instance):
    path = mutable_instance / "coefficients/links.csv"
    headers, rows = _read_table(path)
    row = next(row for row in rows if row["from_site"] == row["to_site"])
    row["allowed_links"] = "1"
    _write_table(path, headers, rows)
    with pytest.raises(ValueError, match="Self links must be explicitly disabled"):
        load_standard(mutable_instance, verify_hashes=False)


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("pickup_share", "1.01", "pickup_share"),
        ("pickup_share", "-0.01", "pickup_share"),
        ("last_rider_cost", "-1", "Last rider coefficients"),
        ("last_rider_mean", "-1", "Last rider coefficients"),
    ],
)
def test_reporting_coefficients_obey_probability_and_nonnegative_bounds(
    mutable_instance, field, value, message
):
    path = mutable_instance / "coefficients/od_hub.csv"
    headers, rows = _read_table(path)
    rows[0][field] = value
    _write_table(path, headers, rows)
    with pytest.raises(ValueError, match=message):
        load_standard(mutable_instance, verify_hashes=False)


def test_order_counts_must_explain_observed_rate(mutable_instance):
    path = mutable_instance / "od_demand.csv"
    headers, rows = _read_table(path)
    rows[0]["lambda_per_hour"] = str(float(rows[0]["lambda_per_hour"]) + 1)
    _write_table(path, headers, rows)
    with pytest.raises(ValueError, match="Inconsistent OD counts/exposure/distance"):
        load_standard(mutable_instance, verify_hashes=False)


def test_case_override_cannot_change_a_physical_parameter(mutable_instance):
    _edit_json(
        mutable_instance / "model.json",
        lambda model: model["cases"]["K1"].update(delta=0),
    )
    with pytest.raises(ValueError, match="Cases may change max_orders only"):
        load_standard(mutable_instance, verify_hashes=False)


def test_export_does_not_replace_an_existing_instance(
    standard_instance, standard_market
):
    with pytest.raises(FileExistsError):
        export_standard(SNAPSHOT, standard_market, standard_instance)


def test_all_legacy_csv_headers_are_explicit(standard_instance):
    for name, columns in BASE_HEADERS.items():
        headers, _ = _read_table(standard_instance / name)
        assert headers == list(columns)
