"""Independent diagnostic checks; these tests never launch a solver."""

from copy import deepcopy

import pytest

from src.experiments import fixed_network_diagnostic_20261024 as diagnostic


def network():
    return {
        "facility": [1, 1, 1],
        "capacity_level": [[1, 0], [0, 1]],
        "airline": [[0, 1, 0], [0, 0, 1], [1, 0, 0]],
    }


def solution():
    return dict(network(), assignment=[[[1]], [[0]]], rho=[0.4], fleet=3)


def test_network_identity_excludes_assignments_and_operations():
    original = solution()
    changed = deepcopy(original)
    changed.update(assignment=[[[0]], [[1]]], rho=[0.8], fleet=7)
    first = diagnostic.extract_network(original)
    second = diagnostic.extract_network(changed)
    assert first == second == network()
    assert set(first) == {"facility", "capacity_level", "airline"}
    assert diagnostic.network_fingerprint(first) == diagnostic.network_fingerprint(second)
    assert original == solution(), "Extraction must not mutate the source incumbent"


@pytest.mark.parametrize("key", ["facility", "capacity_level", "airline"])
def test_network_identity_changes_when_a_fixed_decision_changes(key):
    first, second = network(), network()
    if key == "facility":
        second[key][0] = 0
    else:
        second[key][0][0] = 1 - second[key][0][0]
    assert diagnostic.network_fingerprint(first) != diagnostic.network_fingerprint(second)


def test_network_identity_does_not_depend_on_dictionary_key_order():
    first = network()
    second = {key: first[key] for key in reversed(first)}
    assert diagnostic.network_fingerprint(first) == diagnostic.network_fingerprint(second)


def test_near_binary_network_is_canonicalized():
    source = solution()
    source["facility"][0] = 1 - 1e-8
    source["airline"][0][0] = 1e-8
    assert diagnostic.extract_network(source) == network()


@pytest.mark.parametrize("bad", [0.01, 0.5, -0.01, 1.01, float("nan"), float("inf")])
def test_nonbinary_or_nonfinite_network_is_rejected(bad):
    source = solution()
    source["facility"][0] = bad
    with pytest.raises((ValueError, TypeError)):
        diagnostic.extract_network(source)


def test_flat_layout_extraction_skips_exact_assignment_slice():
    expected = network()
    source = (expected["facility"]
              + [x for row in expected["capacity_level"] for x in row]
              + [1, 0]
              + [x for row in expected["airline"] for x in row])
    kwargs = dict(n_nodes=3, n_h=2, n_levels=2, n_od=2, n_l=1)
    assert diagnostic.network_from_design(source, **kwargs) == expected
    source[7:9] = [0, 1]
    assert diagnostic.network_from_design(source, **kwargs) == expected
    with pytest.raises((ValueError, TypeError)):
        diagnostic.network_from_design(source[:-1], **kwargs)


def test_probe_design_contains_zero_assignment_placeholders_only():
    fixed = network()
    encoded = diagnostic.design_for_probe(fixed, n_od=2, n_l=1)
    assert encoded[7:9] == [0, 0]
    assert diagnostic.network_from_design(
        encoded, n_nodes=3, n_h=2, n_levels=2, n_od=2, n_l=1) == fixed


def test_paired_jobs_and_commands_change_only_assignment_mode_and_output(tmp_path):
    records = [dict(id="L01_N1", case="L01", fingerprint="network-1")]
    jobs = diagnostic.jobs_for(records)
    assert len(jobs) == 2
    assert {job["mode"] for job in jobs} == {"integer", "relaxed"}
    assert len({(job["network_id"], job["network_fingerprint"], job["instance"], job["design"], job["K"])
                for job in jobs}) == 1
    commands = [diagnostic.command_for(tmp_path, {"protocol": {"seconds_per_job": 600}},
                                      job, tmp_path / job["id"]) for job in jobs]
    first, second = commands
    assert first[:4] == second[:4]
    assert first[6:] == second[6:] == ["600"]
    assert {first[4], second[4]} == {"integer", "relaxed"}


def bound_row(lower=None, upper=None, *, certified=True, validated=True):
    return dict(lower_bound=lower, upper_bound=upper,
                lower_bound_certified=certified, upper_bound_validated=validated)


def test_assignment_gap_uses_cross_model_bounds_not_incumbent_difference():
    result = diagnostic.assignment_gap_interval(bound_row(110, 120), bound_row(90, 100))
    assert result["valid"]
    assert result["lower"] == pytest.approx(10)
    assert result["upper"] == pytest.approx(30)


def test_common_fixed_cost_cancels_in_gap_interval():
    original = diagnostic.assignment_gap_interval(bound_row(110, 120), bound_row(90, 100))
    shifted = diagnostic.assignment_gap_interval(bound_row(160, 170), bound_row(140, 150))
    assert shifted["valid"]
    assert (original["lower"], original["upper"]) == (shifted["lower"], shifted["upper"])


def test_gap_lower_endpoint_uses_relaxation_inclusion():
    result = diagnostic.assignment_gap_interval(bound_row(90, 120), bound_row(80, 100))
    assert result["valid"]
    assert result["lower"] == 0
    assert result["upper"] == 40


def test_missing_or_uncertified_bound_does_not_become_zero():
    result = diagnostic.assignment_gap_interval(
        bound_row(110, 120, certified=False), bound_row(90, 100))
    assert result["valid"]
    assert result["lower"] is None
    assert result["upper"] == 30
    result = diagnostic.assignment_gap_interval(bound_row(110, 120), bound_row(None, 100))
    assert result["valid"]
    assert result["lower"] == 10
    assert result["upper"] is None


def test_unvalidated_incumbent_is_not_a_gap_endpoint():
    result = diagnostic.assignment_gap_interval(
        bound_row(110, 120, validated=False), bound_row(90, 100))
    assert result["valid"]
    assert result["lower"] == 10
    assert result["upper"] is None


@pytest.mark.parametrize("integer,relaxed", [
    (bound_row(100, 110), bound_row(120, 130)),
    (bound_row(121, 120), bound_row(90, 100)),
    (bound_row(110, 120), bound_row(101, 100)),
])
def test_inconsistent_certificates_are_reported_not_clipped(integer, relaxed):
    result = diagnostic.assignment_gap_interval(integer, relaxed)
    assert not result["valid"]
    assert result["lower"] is None and result["upper"] is None
    assert result["errors"]


def probe_result():
    return dict(schema="mlnd_fixed_network_probe_v1", scope="conditional_on_fixed_network",
        mode="integer", K=2, status="completed", termination="Optimal",
        total_upper_bound=120.0, total_lower_bound=110.0, fixed_cost=30.0,
        operating_upper_bound=90.0, operating_lower_bound=80.0,
        upper_bound_validated=True, lower_bound_certified=True,
        original_integer_feasible=True,
        bound_certificate=dict(valid=True, scope="conditional_on_fixed_network"),
        catalogue=dict(complete=True, support_complete=True, service_paths=8, empty_paths=0),
        settings=dict(tight_airline_bounds=True, conditional_demand=False,
            extra_strengthening=False, assignment_incident_clique=False,
            shared_rho_joint=False, queue_perspective=False),
        validation=dict(selected_candidate=dict(passed=True), original_integrality=dict(passed=True)),
        solver=dict(status="Optimal", optimal=True, has_solution=True, nodes=0),
        timings=dict(total_seconds=5.0, solver_seconds=4.0),
        model=dict(variables=100, binary_variables=2))


def probe_job():
    return dict(id="L01_N1_integer", case="L01", network_id="L01_N1",
                mode="integer", K=2, network_fingerprint="fixed-network")


def test_summary_reads_probe_total_cost_fields_without_adding_fixed_cost():
    row = diagnostic.summarize_result(probe_result(), {"returncode": 0}, probe_job())
    assert row["upper_bound_validated"] and row["lower_bound_certified"]
    assert row["upper_bound"] == 120
    assert row["lower_bound"] == 110
    assert row["gap"] == pytest.approx(10 / 120)
    assert row["fixed_cost"] == 30
    assert row["operating_upper_bound"] == 90


@pytest.mark.parametrize("change", [
    dict(mode="relaxed"), dict(K=1), dict(scope="global"),
    dict(catalogue=dict(complete=False, support_complete=True)),
    dict(settings=dict(tight_airline_bounds=False, conditional_demand=False, extra_strengthening=False)),
])
def test_wrong_problem_or_incomplete_catalogue_has_no_certified_comparison(change):
    source = probe_result()
    source.update(change)
    row = diagnostic.summarize_result(source, {"returncode": 0}, probe_job())
    assert not row["lower_bound_certified"]
    assert row["lower_bound"] is None


def test_summary_requires_the_explicit_bound_certificate_to_agree():
    source = probe_result()
    source["bound_certificate"]["valid"] = False
    row = diagnostic.summarize_result(source, {"returncode": 0}, probe_job())
    assert not row["lower_bound_certified"]
    assert row["lower_bound"] is None


def test_continuous_numerical_termination_cannot_supply_lower_bound():
    source, job = probe_result(), probe_job()
    source["mode"] = job["mode"] = "relaxed"
    source["solver"].update(status="NumBest", optimal=False)
    source["termination"] = "NumBest"
    source["original_integer_feasible"] = False
    row = diagnostic.summarize_result(source, {"returncode": 0}, job)
    assert not row["lower_bound_certified"]
    assert row["lower_bound"] is None
