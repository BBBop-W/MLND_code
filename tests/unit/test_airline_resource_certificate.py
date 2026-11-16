import json
from pathlib import Path

import pytest

from src.experiments import airline_resource_certificate_20261027 as bench


def test_failed_candidate_design_roundtrip_and_layout_validation():
    model = dict(values=dict(n_l=2, n_h=1, n_od=2), ids=dict(levels=[10, 20]))
    key = "101" + "01" + "1001" + "010101010"
    design = bench.design_from_key(key, model, "origin")
    reconstructed = (design["facility"] + sum(design["capacity_level"], [])
                     + sum(design["assignment"], []) + sum(design["airline"], []))
    assert "".join(str(v) for v in reconstructed) == key
    assert design["assignment"] == [[1, 0], [0, 1]]
    assert design["assignment_scope"] == "origin"
    with pytest.raises(ValueError, match="length"):
        bench.design_from_key(key[:-1], model, "origin")
    with pytest.raises(ValueError, match="binary"):
        bench.design_from_key(key[:-1] + "x", model, "origin")
    with pytest.raises(ValueError, match="origin-assignment"):
        bench.design_from_key(key, model, "base_pair")


@pytest.mark.parametrize("flags", [[], ["--seed", "44"], ["--gap=0.5"],
    ["--plain"], ["--task-airline-bounds"], ["--airline-resource-cuts"] * 2])
def test_proposed_flags_cannot_change_model_or_runtime(flags):
    with pytest.raises(ValueError):
        bench.validate_flags(flags)


def test_prepare_freezes_six_new_runs_with_shared_repaired_binary(tmp_path, monkeypatch):
    old_binary, new_binary = tmp_path / "old", tmp_path / "new"
    old_binary.write_text("old")
    new_binary.write_text("repaired")
    monkeypatch.setattr(bench, "verify_snapshot", lambda root: {
        "binary_sha256": bench.base.sha256(old_binary)})
    calls = []

    def fake_prepare(batch, **kwargs):
        calls.append(kwargs)
        batch.mkdir(parents=True)
        jobs = [dict(id=f"{case[0]}_{variant}") for case in kwargs["cases"]
                for variant in kwargs["variants"]]
        (batch / "experiment_manifest.json").write_text(json.dumps(dict(
            schema="fixture", protocol={}, jobs=jobs, frozen_sha256={},
            binary_sha256=bench.base.sha256(new_binary))))

    monkeypatch.setattr(bench.task, "prepare", fake_prepare)
    flags = ["--airline-connectivity-cuts", "--airline-resource-cuts"]
    result = bench.prepare(tmp_path / "root", binary=new_binary, proposed_flags=flags)
    assert result["jobs"] == 6 and result["solver_started"] is False
    assert calls[0]["cases"] == bench.CASES
    assert calls[0]["variants"]["fixed_baseline"] == bench.BASELINE_FLAGS
    assert calls[0]["variants"]["proposed"] == (*bench.BASELINE_FLAGS, *flags)
    script = (tmp_path / "root/comparison/run.py").read_text()
    assert bench.MODULE in script and "--root" in script and "--suite" not in script
    with pytest.raises(ValueError, match="new binary"):
        bench.prepare(tmp_path / "bad", binary=old_binary, proposed_flags=flags)
    assert len(calls) == 1


def test_shared_task_command_uses_frozen_variant_map():
    command = ["old-binary", "--input", "old", "--out", "old", "--seconds", "600",
               "--threads", "1", "--seed", "43", "--gap", "0.01"]
    flags = [*bench.BASELINE_FLAGS, "--airline-resource-cuts"]
    manifest = {"protocol": {"seconds_per_job": 600,
                              "algorithm_variants": {"proposed": flags}}}
    job = dict(reference_command=command, instance="instances/tsl_L02_M01", variant="proposed")
    actual = bench.task.command_for(Path("/frozen"), manifest, job, "/attempt")
    assert actual[len(command):] == ["--path-support-cuts", *flags]
    assert actual[0] == "/frozen/frozen/bin/mlnd_cpp"
    assert actual[actual.index("--threads") + 1] == "1"
    assert actual[actual.index("--seed") + 1] == "43"


def test_snapshot_verification_detects_mutation(tmp_path):
    target = tmp_path / "pre_fix"
    target.mkdir()
    binary = target / "binary"
    binary.write_text("before")
    (target / "snapshot_manifest.json").write_text(json.dumps({
        "frozen_sha256": {"binary": bench.base.sha256(binary)}}))
    assert bench.verify_snapshot(tmp_path)
    binary.write_text("after")
    with pytest.raises(ValueError, match="changed"):
        bench.verify_snapshot(tmp_path)


def test_adapter_restores_task_and_base_functions():
    guard, record, reporter = bench.base.guarded_execute, bench.base._record, bench.base.report
    with bench.adapters():
        assert bench.base.guarded_execute is bench.guarded_execute
        assert bench.base.report is bench.report
        assert bench.base._record is bench.record
    assert (bench.base.guarded_execute, bench.base._record, bench.base.report) == (guard, record, reporter)


def test_new_rows_are_checked_against_every_final_point():
    # theta + 2*x >= 5: the first point satisfies it, the other disproves it.
    rows = [("cut", dict(rhs=5.0, coefficients=[[1, 1.0], [0, 2.0]]))]
    result = bench.cross_check_rows(rows, [("own", [1], 3.0), ("other", [0], 4.0)])
    assert not result["passed"] and result["checks"] == 2
    assert result["maximum_violation"] == 1.0
    assert result["worst_row_point"] == ["cut", "other"]
    result = bench.cross_check_rows([("connect", dict(sense="L", rhs=0,
        coefficients=[[0, 1], [1, -1]]))], [("valid", [0, 1], 10)])
    assert result["passed"] and result["checks"] == 1


def test_self_gap_and_numerical_termination_remain_distinct():
    result = dict(objective=100, bound=96, gap=.04, requested_gap=.01,
                  target_gap_met=False, termination_reason="rmp_duality_gap_failed")
    summary = bench.self_gap_summary(result)
    assert summary["self_gap_consistent"] and summary["computed_self_gap"] == .04
    assert summary["certificate_failure"] and not summary["normal_budget_or_gap_termination"]
    assert not summary["computed_target_met"]
    result["gap"] = .01  # e.g. incorrectly substituted common-UB gap.
    assert not bench.self_gap_summary(result)["self_gap_consistent"]


def test_new_rows_reject_layout_and_nonfinite_coefficients():
    for coefficients in ([[3, 1]], [[0, float("nan")]]):
        result = bench.cross_check_rows([("bad", dict(rhs=0, coefficients=coefficients))],
                                       [("point", [1], 10)])
        assert not result["passed"] and result["checks"] == 0


def test_certificate_recovery_deduplicates_trace_copies_and_uses_original_gap():
    trace = dict(iteration=2, duality_tolerance=.01,
        dual_certificate_method="minimum_norm_cone_ray_lp", termination_reason="columns_added",
        dual_attempts=[dict(numerical_profile=5, method="minimum_norm_cone_ray_lp",
            valid=True, optimal=True, corrected_gap=.001, seconds=.1, correction=.0005)])
    call = dict(candidate_id="integer:4", cg_call=3, cg_trace=[trace])
    result = dict(iterations=[call], acceleration_records=[json.loads(json.dumps(call))],
                  termination_reason="time_limit")
    summary = bench.certificate_recovery_summary(result)
    assert summary["unique_cg_iterations"] == 1
    assert summary["duplicate_trace_copies_merged"] == 1
    assert summary["minimum_norm_attempts"] == summary["minimum_norm_original_gap_passes"] == 1
    failed = json.loads(json.dumps(call))
    failed["cg_call"] = 4
    failed["cg_trace"][0]["dual_attempts"][0]["corrected_gap"] = .02
    failed["cg_trace"][0]["termination_reason"] = "rmp_duality_gap_failed"
    result["iterations"].append(failed)
    summary = bench.certificate_recovery_summary(result)
    assert summary["minimum_norm_attempts"] == 2
    assert summary["minimum_norm_original_gap_passes"] == 1
    assert summary["rmp_duality_gap_failed_iterations"] == 1


def test_certificate_recovery_does_not_claim_untriggered_fallback():
    summary = bench.certificate_recovery_summary(dict(termination_reason="gap_limit",
        continuous_root=dict(cg_trace=[dict(iteration=1, duality_tolerance=.01,
            dual_attempts=[dict(numerical_profile=4, method="cone_ray_lp_recovery",
                               valid=True, optimal=True, corrected_gap=.001)])])))
    assert summary["minimum_norm_attempts"] == summary["minimum_norm_original_gap_passes"] == 0


def test_target_time_requires_first_certified_global_event(tmp_path):
    events = [dict(event="global_bounds", bound_scope="fixed_design", validated_incumbent=True,
        upper_bound=100, lower_bound=99.5, elapsed_seconds=1),
        dict(event="global_bounds", bound_scope="global", validated_incumbent=False,
             upper_bound=100, lower_bound=99.5, elapsed_seconds=2),
        dict(event="global_bounds", bound_scope="global", validated_incumbent=True,
             upper_bound=100, lower_bound=98, elapsed_seconds=3),
        dict(event="global_bounds", bound_scope="global", validated_incumbent=True,
             upper_bound=100, lower_bound=99.5, elapsed_seconds=7)]
    (tmp_path / "progress.jsonl").write_text("\n".join(json.dumps(e) for e in events))
    summary = dict(computed_target_met=True, requested_gap=.01, termination_reason="gap_limit")
    result = bench.target_time_evidence(tmp_path, summary, 10)
    assert result["time_to_target_seconds"] == 7
    assert result["observed_target_time_upper_bound_seconds"] == 10
    assert not result["budget_censored"]
    summary.update(computed_target_met=False, termination_reason="time_limit")
    result = bench.target_time_evidence(tmp_path, summary, 600)
    assert result["time_to_target_seconds"] is None
    assert result["observed_target_time_upper_bound_seconds"] is None and result["budget_censored"]


def test_record_does_not_publish_unvalidated_raw_bounds(tmp_path, monkeypatch):
    monkeypatch.setattr(bench.task, "record", lambda *args: dict(validation_passed=True,
        upper_bound=None, lower_bound=None, gap=None, solve_seconds=20))
    (tmp_path / "result.json").write_text(json.dumps(dict(objective=100, bound=99.5, gap=.005,
        requested_gap=.01, target_gap_met=True, termination_reason="gap_limit")))
    (tmp_path / "native_validation.json").write_text(json.dumps(dict(passed=True)))
    (tmp_path / "artifact_audit.json").write_text(json.dumps(dict(passed=False)))
    row = bench.record(tmp_path, {}, dict(result_directory="."))
    assert not row["validation_passed"]
    assert row["upper_bound"] is None and row["time_to_target_seconds"] is None
    assert row["raw_result_self_gap_diagnostic"]["computed_target_met"]
    (tmp_path / "artifact_audit.json").write_text(json.dumps(dict(passed=True)))
    row = bench.record(tmp_path, {}, dict(result_directory="."))
    assert row["validation_passed"] and row["upper_bound"] == 100
    assert row["time_to_target_seconds"] is None
    assert row["observed_target_time_upper_bound_seconds"] == 20


def test_resource_execution_distinguishes_repaired_fallback_from_optimize():
    result = bench.resource_execution_summary(dict(airline_resource_attempts=[
        dict(stage="root", status="budget_exhausted", solver_seconds=0, valid=True, accepted=True),
        dict(stage="root", status="Optimal", solver_seconds=.1, valid=True, accepted=True),
        dict(stage="root", status="solver_exception: fixture", solver_seconds=.2, valid=False, accepted=False)]))[0]
    assert result["attempts"] == 3 and result["actual_optimize_calls"] == 1
    assert result["budget_exhausted"] == 1 and result["solver_exception_attempts"] == 1
    assert result["valid_certificates"] == result["accepted_cuts"] == 2


@pytest.mark.parametrize("suite,cases,expected_jobs", [
    ("l1-expanded", bench.L1_CASES, 12), ("l2-comparison", bench.L2_CASES, 6)])
def test_expansion_prepare_has_fresh_jobs_without_prefix_snapshot(tmp_path, monkeypatch, suite, cases, expected_jobs):
    workspace = tmp_path / "workspace"
    reference = tmp_path / "validated"
    original = tmp_path / "original"
    binary = tmp_path / "binary"
    binary.write_text("same validated binary")
    for path in (workspace / "cpp/src/fixture.cpp", reference / "frozen/cpp/src/fixture.cpp"):
        path.parent.mkdir(parents=True)
        path.write_text("same C++ source")
    (workspace / "cpp/src/fixture.cpp").write_text("another task changed the live source")
    (reference / "experiment_manifest.json").write_text("{}")
    original.mkdir()
    jobs = [dict(id=case[0], instance="instances/" + case[0]) for case in cases]
    (original / "experiment_manifest.json").write_text(json.dumps(dict(jobs=jobs)))
    (original / "state.json").write_text(json.dumps(dict(jobs={j["id"]:
        dict(status="pending", attempts=[]) for j in jobs})))
    monkeypatch.setattr(bench, "ROOT", workspace)
    monkeypatch.setattr(bench, "VALIDATED_REFERENCE", reference)
    monkeypatch.setattr(bench.task, "REFERENCE", original)
    monkeypatch.setattr(bench.base, "verify", lambda _: dict(binary_sha256=bench.base.sha256(binary)))
    historical_calls = []
    def fake_history(selected):
        historical_calls.append(selected)
        return [dict(instance=c[0], historical_reference_only=True) for c in selected]
    monkeypatch.setattr(bench, "historical_full_cplex", fake_history)
    calls = []
    def fake_prepare(batch, **kwargs):
        calls.append(kwargs)
        (batch / "provenance").mkdir(parents=True)
        (batch / "experiment_manifest.json").write_text(json.dumps(dict(
            protocol={"suite": kwargs["suite"]}, frozen_sha256={},
            binary_sha256=bench.base.sha256(binary), jobs=[dict(id=str(i)) for i in range(expected_jobs)])))
    monkeypatch.setattr(bench.task, "prepare", fake_prepare)
    batch = tmp_path / suite
    builder = bench.prepare_l1 if suite == "l1-expanded" else bench.prepare_l2
    result = builder(batch, binary=binary)
    assert result["jobs"] == expected_jobs and result["solver_started"] is False
    assert calls[0]["cases"] == cases
    assert calls[0]["variants"]["fixed_baseline"] == bench.BASELINE_FLAGS
    assert calls[0]["variants"]["proposed"] == (*bench.BASELINE_FLAGS, *bench.PROPOSED_FLAGS)
    assert suite in (batch / "run.py").read_text()
    assert not (batch / "pre_fix").exists()
    assert not (batch / "experiment_manifest.json").read_text().count('"reused_baselines": "')
    assert (batch / "frozen/cpp/src/fixture.cpp").read_text() == "same C++ source"
    preflight = "l1_preflight.json" if suite == "l1-expanded" else "l2_preflight.json"
    provenance = json.loads((batch / "provenance" / preflight).read_text())
    assert provenance["frozen_cpp_matches_validated_snapshot"] is True
    assert provenance["production_cpp_matches_validated_snapshot"] is False
    assert provenance["live_checkout_differences"] == ["src/fixture.cpp"]
    if suite == "l2-comparison":
        assert historical_calls == [cases]
        historical = json.loads((batch / "provenance/historical_full_cplex.json").read_text())
        assert len(historical["rows"]) == 3
    else:
        assert historical_calls == []


@pytest.mark.parametrize("suite", ["l1-expanded", "l2-comparison", "primal-comparison", "optimization-comparison", "refinement-comparison"])
def test_expansion_verify_does_not_require_old_l2_snapshot(monkeypatch, suite):
    monkeypatch.setattr(bench.base, "verify", lambda _: {"protocol": {"suite": suite}})
    monkeypatch.setattr(bench, "verify_snapshot", lambda _: (_ for _ in ()).throw(AssertionError("unneeded snapshot")))
    assert bench.verify_context(Path("batch"), Path("unrelated-old-root"))


def test_prepare_preserves_diagnosis_and_refuses_existing_experiment(tmp_path, monkeypatch):
    batch = tmp_path / "new_batch"
    (batch / "analysis").mkdir(parents=True)
    diagnosis = batch / "analysis/diagnosis.json"
    diagnosis.write_text('{"preserve": true}')
    calls = []
    def fake_prepare(staging, **kwargs):
        calls.append(staging)
        staging.mkdir()
        (staging / "experiment_manifest.json").write_text('{}')
        (staging / "run.py").write_text('from pathlib import Path\nbatch=Path(__file__).parent\n')
        return dict(batch=str(staging), solver_started=False)
    monkeypatch.setattr(bench.task, "prepare", fake_prepare)
    result = bench.prepare_preserving_analysis(batch, binary="fixture")
    assert result["batch"] == str(batch) and not result["solver_started"]
    assert diagnosis.read_text() == '{"preserve": true}'
    assert not calls[0].exists()
    with pytest.raises(FileExistsError, match="existing experiment"):
        bench.prepare_preserving_analysis(batch, binary="fixture")
    assert len(calls) == 1


def test_prepare_primal_uses_same_new_binary_and_only_polish_differs(tmp_path, monkeypatch):
    workspace, reference = tmp_path / "workspace", tmp_path / "validated"
    source = workspace / "cpp/src/fixture.cpp"
    source.parent.mkdir(parents=True)
    source.write_text('new heuristic source')
    binary = tmp_path / "new_binary"
    binary.write_text('new compiled binary')
    monkeypatch.setattr(bench, "ROOT", workspace)
    monkeypatch.setattr(bench, "VALIDATED_REFERENCE", reference)
    monkeypatch.setattr(bench.base, "verify", lambda _: dict(binary_sha256="old_binary_hash"))
    monkeypatch.setattr(bench, "historical_full_cplex", lambda cases: [dict(instance=c[0]) for c in cases])
    calls=[]
    def fake_prepare(batch, **kwargs):
        calls.append(kwargs)
        (batch / "frozen/cpp/src").mkdir(parents=True)
        (batch / "frozen/cpp/src/fixture.cpp").write_text(source.read_text())
        (batch / "provenance").mkdir()
        jobs=[dict(id=f'{case[0]}_{variant}') for case in kwargs['cases'] for variant in kwargs['variants']]
        (batch / "experiment_manifest.json").write_text(json.dumps(dict(protocol={},
            frozen_sha256={}, jobs=jobs, binary_sha256=bench.base.sha256(binary))))
    monkeypatch.setattr(bench, "prepare_preserving_analysis", fake_prepare)
    result=bench.prepare_primal(tmp_path/'batch',binary=binary)
    assert result['jobs']==6 and result['solver_started'] is False
    assert calls[0]['cases']==bench.L2_CASES
    common=(*bench.BASELINE_FLAGS,*bench.PROPOSED_FLAGS)
    assert calls[0]['variants']==dict(fixed_baseline=common,proposed=(*common,'--primal-polish'))
    manifest=json.loads((tmp_path/'batch/experiment_manifest.json').read_text())
    assert manifest['protocol']['external_warm_start'] is None
    assert manifest['protocol']['external_root_certificate'] is None
    assert manifest['protocol']['comparison_feature']=='--primal-polish'
    assert 'primal-comparison' in (tmp_path/'batch/run.py').read_text()
    monkeypatch.setattr(bench.base,'verify',lambda _:dict(binary_sha256=bench.base.sha256(binary)))
    with pytest.raises(ValueError,match='newly compiled'):
        bench.prepare_primal(tmp_path/'other',binary=binary)
    assert len(calls)==1


def test_first_ub_improvement_requires_validated_global_event(tmp_path):
    events=[dict(event='global_bounds',bound_scope='global',validated_incumbent=True,
        upper_bound=100,elapsed_seconds=1,stage='initial'),
        dict(event='global_bounds',bound_scope='restricted',validated_incumbent=True,
        upper_bound=90,elapsed_seconds=2),
        dict(event='global_bounds',bound_scope='global',validated_incumbent=False,
        upper_bound=80,elapsed_seconds=3),
        dict(event='global_bounds',bound_scope='global',validated_incumbent=True,
        upper_bound=95,elapsed_seconds=4,stage='initial_design')]
    (tmp_path/'progress.jsonl').write_text('\n'.join(json.dumps(x) for x in events))
    row=bench.incumbent_improvement_evidence(tmp_path)
    assert row['initial_progress_upper_bound']==100
    assert row['first_incumbent_improvement_seconds']==4
    assert row['first_incumbent_improvement_evidence']['stage']=='initial_design'


def test_optimization_variants_isolate_two_predeclared_stages():
    stages = bench.optimization_variants()
    common = (*bench.BASELINE_FLAGS, *bench.PROPOSED_FLAGS, '--primal-polish')
    assert list(stages) == ['baseline', 'schedule', 'full']
    assert stages['baseline'] == common
    assert stages['schedule'] == (*common, '--search-scheduling', '--dual-lp-first')
    assert stages['full'] == (*stages['schedule'], '--fractional-path-support', '--airline-resource-balance')
    for flags in stages.values():
        assert len(flags) == len(set(flags))
        assert not {'--start', '--fixed-design', '--gap', '--seed'}.intersection(flags)


def test_prepare_optimization_freezes_nine_case_major_jobs(tmp_path, monkeypatch):
    workspace, reference = tmp_path/'workspace', tmp_path/'old_primal'
    source = workspace/'cpp/src/fixture.cpp'
    source.parent.mkdir(parents=True)
    source.write_text('new scheduling source')
    binary = tmp_path/'binary'
    binary.write_text('new scheduling binary')
    monkeypatch.setattr(bench, 'ROOT', workspace)
    monkeypatch.setattr(bench, 'PRIMAL_BATCH', reference)
    monkeypatch.setattr(bench.base, 'verify', lambda _:dict(binary_sha256='prior_hash'))
    monkeypatch.setattr(bench, 'historical_full_cplex', lambda cases:[dict(instance=c[0]) for c in cases])
    calls = []
    def fake_prepare(batch, **kwargs):
        calls.append(kwargs)
        (batch/'frozen/cpp/src').mkdir(parents=True)
        (batch/'frozen/cpp/src/fixture.cpp').write_text(source.read_text())
        (batch/'provenance').mkdir()
        jobs=[dict(id=f'{case[0]}_{variant}', variant=variant) for case in kwargs['cases']
              for variant in kwargs['variants']]
        (batch/'experiment_manifest.json').write_text(json.dumps(dict(protocol={},
            frozen_sha256={}, jobs=jobs, binary_sha256=bench.base.sha256(binary))))
    monkeypatch.setattr(bench, 'prepare_preserving_analysis', fake_prepare)
    out = bench.prepare_optimization(tmp_path/'batch', binary=binary)
    assert out['jobs'] == 9 and not out['solver_started']
    assert calls[0]['variants'] == bench.optimization_variants()
    manifest=json.loads((tmp_path/'batch/experiment_manifest.json').read_text())
    assert [j['variant'] for j in manifest['jobs']] == ['baseline','schedule','full']*3
    assert manifest['protocol']['external_warm_start'] is None
    assert manifest['protocol']['external_root_certificate'] is None
    assert manifest['protocol']['reused_baselines'] is None
    assert manifest['protocol']['stage_contrasts']['full_minus_schedule'] == list(bench.STRUCTURAL_FLAGS)
    assert 'optimization-comparison' in (tmp_path/'batch/run.py').read_text()
    monkeypatch.setattr(bench.base,'verify',lambda _:dict(binary_sha256=bench.base.sha256(binary)))
    with pytest.raises(ValueError, match='newly compiled'):
        bench.prepare_optimization(tmp_path/'other', binary=binary)
    assert len(calls) == 1


def test_search_diagnostics_does_not_treat_accepted_resource_cut_as_skip(tmp_path):
    events = [dict(event='global_bounds', bound_scope='global', validated_incumbent=True,
                   lower_bound=10, upper_bound=20, elapsed_seconds=1),
              dict(event='global_bounds', bound_scope='global', validated_incumbent=True,
                   lower_bound=12, upper_bound=19, elapsed_seconds=30),
              dict(event='global_bounds', bound_scope='restricted', validated_incumbent=True,
                   lower_bound=18, upper_bound=19, elapsed_seconds=40),
              dict(event='global_bounds', bound_scope='global', validated_incumbent=False,
                   lower_bound=18, upper_bound=19, elapsed_seconds=50),
              dict(event='task_resource_separation', candidate_id='integer:1', accepted=True),
              dict(event='airline_resource_separation', candidate_id='integer:2', accepted=True,
                   skipped_integer_subproblem=True),
              dict(event='path_support_separation', candidate_id='integer:3',
                   skipped_integer_subproblem=True)]
    (tmp_path/'progress.jsonl').write_text('\n'.join(json.dumps(e) for e in events))
    result=dict(wall_before_output=100, acceleration_statistics=dict(integer_cg_calls=2),
                master_solves=[dict(relative_gap_target=0,seconds=5,nodes=12,status='AbortTimeLim')])
    diag=bench.search_diagnostics(result,tmp_path)
    assert diag['resource_accepted_candidates'] == 2
    assert diag['explicitly_reported_resource_skipped_candidates'] == 1
    assert diag['support_skipped_candidates'] == 1
    assert diag['integer_cg_calls'] == 2
    assert diag['zero_gap_master_count'] == 1 and diag['zero_gap_master_seconds'] == 5
    assert diag['last_observed_lb_improvement_seconds'] == 30
    assert diag['seconds_since_last_observed_lb_improvement'] == 70
    assert diag['global_bound_checkpoints'][0]['lower_bound'] == 12
    unvalidated=bench.search_diagnostics(result,tmp_path,verified=False)
    assert unvalidated['global_bound_checkpoints'] == []
    assert unvalidated['last_observed_lb_improvement_seconds'] is None


def test_dual_profile_timings_deduplicate_trace_copies():
    trace=dict(iteration=1, duality_tolerance=.01, dual_attempts=[
        dict(numerical_profile=0, method='socp', seconds=2, valid=False, status='AbortTimeLim'),
        dict(numerical_profile=4, method='lp', seconds=.2, valid=True, status='Optimal')])
    call=dict(candidate_id='integer:2', cg_call=1, cg_trace=[trace])
    summary=bench.certificate_recovery_summary(dict(iterations=[call], acceleration_records=[call]))
    profiles={p['numerical_profile']:p for p in summary['all_profile_summary']}
    assert profiles[0]['attempts'] == 1 and profiles[0]['seconds'] == 2
    assert profiles[4]['attempts'] == 1 and profiles[4]['seconds'] == .2
    assert profiles[4]['valid'] == 1


def test_optimization_report_compares_three_stages_and_keeps_per_run_gap(tmp_path, monkeypatch):
    (tmp_path/'analysis').mkdir()
    (tmp_path/'experiment_manifest.json').write_text(json.dumps(dict(
        protocol=dict(suite=bench.OPTIMIZATION_SUITE))))
    rows=[]
    for variant, ub, lb in [('baseline',100,98),('schedule',99,98.5),('full',98.8,98.5)]:
        gap=(ub-lb)/ub
        rows.append(dict(instance='instances/example',variant=variant,
            upper_bound=ub,lower_bound=lb,gap=gap,reported_gap=gap,
            scheduler_status='completed',validation_passed=True,
            computed_target_met=gap<=.01,normal_budget_or_gap_termination=True,
            termination_reason='gap_limit' if gap<=.01 else 'time_limit',solve_seconds=600))
    (tmp_path/'analysis/comparison.json').write_text(json.dumps(dict(rows=rows)))
    monkeypatch.setattr(bench.task,'report',lambda *args,**kwargs:dict(jobs=3))
    bench.report(tmp_path)
    contrasts=json.loads((tmp_path/'analysis/stage_contrasts.json').read_text())['rows']
    assert len(contrasts)==3
    assert contrasts[0]['upper_bound_reduction']==1
    assert contrasts[0]['lower_bound_increase']==.5
    summary=json.loads((tmp_path/'analysis/target_attainment_summary.json').read_text())
    assert summary['compared_variants']==['baseline','schedule','full']
    assert summary['all_variants_reached_cases']==0
    assert summary['both_variants_reached_cases'] is None
    report=(tmp_path/'analysis/REPORT.md').read_text()
    assert '三阶段' in report and 'baseline增加' not in report
    assert '2.0000%' in report and '0.3036%' in report


def test_support_canonicalization_only_removes_exactly_idle_assignments():
    model=dict(values=dict(n_od=1,n_l=1,n_h=1),ids=dict(levels=[0]))
    demand=dict(single_rates=[1],pair_rates=[[0]])
    catalogue=dict(missions=[dict(path=dict(id=0,first_od=0,second_od=None,
        b=[0,0],arc_counts=[[0,1,1],[1,0,1]]))])
    solution=dict(assignment=[[1]],rho=[0.0],flow=[0.0],airline=[[0,0],[0,0]])
    point=('fixture',[1,1,1,1,0,0,0,0],5.0)
    normalized, evidence=bench.canonicalize_support_point(model,demand,catalogue,solution,point)
    assert evidence['passed'] and normalized[1][3] == 0
    assert len(evidence['changed_assignments'])==1 and normalized[2]==point[2]
    solution['rho']=[1e-12]
    _, evidence=bench.canonicalize_support_point(model,demand,catalogue,solution,point)
    assert not evidence['passed']  # No audit-only tolerance relaxation.
    solution['rho']=[0.0];solution['flow']=[1e-12]
    _, evidence=bench.canonicalize_support_point(model,demand,catalogue,solution,point)
    assert not evidence['passed']
    solution.update(rho=[.5],flow=[.5],airline=[[0,1],[1,0]])
    normalized,evidence=bench.canonicalize_support_point(model,demand,catalogue,solution,point)
    assert evidence['passed'] and not evidence['changed_assignments']
    assert normalized[1]==point[1]


def test_refinement_refuses_overlap_with_original_dispatcher(tmp_path,monkeypatch):
    jobs=[dict(id=str(i)) for i in range(9)]
    monkeypatch.setattr(bench.base,'verify',lambda _:dict(jobs=jobs))
    state=dict(jobs={str(i):dict(status='running' if i==8 else 'completed') for i in range(9)})
    (tmp_path/'state.json').write_text(json.dumps(state))
    with pytest.raises(ValueError,match='all nine original jobs completed'):
        bench.prior_algorithm_results(tmp_path)


def test_refinement_pins_separate_binary_and_allows_only_root_production_change(tmp_path,monkeypatch):
    workspace, prior, batch=tmp_path/'workspace',tmp_path/'prior',tmp_path/'refined'
    source=workspace/'cpp/src/benders.cpp';source.parent.mkdir(parents=True);source.write_text('new root control')
    old=prior/'frozen/cpp/src/benders.cpp';old.parent.mkdir(parents=True);old.write_text('old root control')
    (prior/'experiment_manifest.json').write_text('{}')
    for identity,_,_ in bench.L2_CASES:
        p=prior/'instances'/identity/'manifest.json';p.parent.mkdir(parents=True);p.write_text('{}')
    binary=tmp_path/'binary';binary.write_text('new root binary')
    monkeypatch.setattr(bench,'ROOT',workspace);monkeypatch.setattr(bench,'OPTIMIZATION_BATCH',prior)
    monkeypatch.setattr(bench,'prior_algorithm_results',lambda _: (dict(binary_sha256='oldhash'),[]))
    monkeypatch.setattr(bench,'historical_full_cplex',lambda _:[])
    calls=[]
    def fake_prepare(destination,**kwargs):
        calls.append(kwargs)
        (destination/'frozen/cpp/src').mkdir(parents=True)
        (destination/'frozen/cpp/src/benders.cpp').write_text(source.read_text())
        (destination/'provenance').mkdir()
        for identity,_,_ in bench.L2_CASES:
            p=destination/'instances'/identity/'manifest.json';p.parent.mkdir(parents=True);p.write_text('{}')
        jobs=[dict(id=c[0],variant='full_refined') for c in kwargs['cases']]
        (destination/'experiment_manifest.json').write_text(json.dumps(dict(protocol={},frozen_sha256={},
            jobs=jobs,binary_sha256=bench.base.sha256(binary))))
    monkeypatch.setattr(bench,'prepare_preserving_analysis',fake_prepare)
    result=bench.prepare_refinement(batch,binary=binary)
    assert result['jobs']==3 and not result['solver_started']
    assert calls[0]['variants']==dict(full_refined=bench.optimization_variants()['full'])
    m=json.loads((batch/'experiment_manifest.json').read_text())
    assert m['protocol']['prior_algorithm_reference']=='provenance/prior_algorithm_results.json'
    assert m['protocol']['reused_baselines'] is None
    assert 'refinement-comparison' in (batch/'run.py').read_text()
    evidence=json.loads((batch/'provenance/refinement_preflight.json').read_text())
    assert len(evidence['changes']) == 2
    assert 'initial_design' in evidence['changes'][1]
    (workspace/'cpp/src/model.cpp').write_text('undeclared model change')
    with pytest.raises(ValueError,match='undeclared production-source'):
        bench.prepare_refinement(tmp_path/'bad',binary=binary)
    assert len(calls)==1
