import ast
from decimal import Decimal
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.experiments import baseline_benchmark_20261029 as bench
from src.experiments import baseline_cut_audit_20261029 as audit


def test_predeclared_jobs_share_model_budget_binary_and_keep_full_domain():
    jobs = bench.jobs_for()
    assert len(jobs) == len({j["id"] for j in jobs}) == 8
    assert {j["instance"] for j in jobs} == {"instances/" + k for k in bench.CASES}
    for job in jobs:
        command = bench.command_for(Path("/batch"), {"protocol": bench.protocol()}, job, "/attempt")
        get = lambda name: command[command.index(name) + 1]
        assert command[0] == "/batch/frozen/bin/mlnd_cpp"
        for name, value in (("--seconds", "600"), ("--threads", "1"), ("--seed", "43"),
                            ("--gap", "0.01"), ("--k", "2"), ("--path-column-limit", "0"),
                            ("--demand-formulation", "preprocessed")):
            assert command.count(name) == 1 and get(name) == value
        full = job["method"] == "full_cplex"
        assert get("--method") == ("full_path" if full else "benders_cg")
        assert get("--preset") == (bench.FULL_PRESET if full else bench.PRESET)
        assert ("--export-catalogue" in command) == full
        assert not {"--start", "--fixed-design", "--first-leg-consistency", "--plain"} & set(command)


def test_inspect_rejects_changed_input_before_any_solve(tmp_path, monkeypatch):
    name = next(iter(bench.CASES))
    folder = tmp_path / "instances" / name
    folder.mkdir(parents=True)
    (folder / "model.json").write_text(json.dumps({"values": {"n_l": 1, "n_h": 10, "n_od": 25}}))
    (folder / "manifest.json").write_text("changed")
    monkeypatch.setattr(bench.base, "run", lambda *a, **k: pytest.fail("inspect must not run"))
    with pytest.raises(ValueError, match="identity differs"):
        bench.inspect(tmp_path)


def test_guarded_children_use_new_frozen_entrypoint(monkeypatch):
    seen = []
    monkeypatch.setattr(bench, "_BASE_GUARD", lambda *args, **kw: seen.append(args[3]))
    for action in ("audit-worker", "report"):
        command = ["python", "-m", bench.base.__name__, action, "--batch", "/batch"]
        bench.guarded_execute("/batch", {}, "id", command, "/attempt", None)
        assert seen[-1][2] == bench.MODULE
        assert command[2] == bench.base.__name__


def test_single_adapter_restores_every_patched_function_after_error():
    names = ("command_for", "_record", "guarded_execute", "verify", "report")
    before = {name: getattr(bench.base, name) for name in names}
    with pytest.raises(RuntimeError):
        with bench.adapters():
            assert bench.base.verify is bench.verify
            assert bench.base.command_for is bench.command_for
            raise RuntimeError("forced")
    assert all(getattr(bench.base, name) is old for name, old in before.items())


@pytest.mark.parametrize("method,expected", [("benders_baseline", "benders_cplex"), ("full_cplex", "full_cplex")])
def test_native_audit_receives_correct_method_not_unrecognized_baseline(tmp_path, monkeypatch, method, expected):
    job = dict(id="job", instance="instances/map", K=2, method=method)
    (tmp_path / "experiment_manifest.json").write_text(json.dumps({"jobs": [job]}))
    attempt = tmp_path / "attempt"
    attempt.mkdir()
    seen = []
    def fake_audit(batch, manifest, current, folder):
        seen.append(current["method"])
        return dict(passed=True, result_directory="attempt")
    monkeypatch.setattr(bench.base.existing, "audit_attempt", fake_audit)
    monkeypatch.setattr(bench.base, "_record", lambda *args: {})
    bench.base.audit_worker(tmp_path, "job", attempt)
    assert seen == [expected]


def test_scalar_queue_math_and_budget_functions_are_unchanged_from_reviewed_source():
    fixture = Path(__file__).resolve().parents[1] / "fixtures/baseline_release_20261029/inherited_audit_ast.json"
    old = json.loads(fixture.read_text())["functions"]
    new = {n.name: hashlib.sha256(ast.dump(n, include_attributes=False).encode()).hexdigest()
           for n in ast.parse(Path(audit.__file__).read_text()).body if isinstance(n, ast.FunctionDef)}
    assert old.keys() <= new.keys()
    # Index-dependent routines were deliberately migrated from raw head OD
    # groups to independent tasks; behavioral task tests cover those changes.
    migrated = {"input_constants", "validate_profile", "replay_queue_row"}
    assert all(old[name] == new[name] for name in old if name not in migrated)


def test_exact_demand_guard_does_not_accept_nearby_forged_total():
    raw = [audit.D(40.0), audit.D(60.0)]
    assert audit.verified_demand_totals(raw, 100.0) == (Decimal(100), Decimal(100))
    with pytest.raises(ValueError, match="ordered binary64"):
        audit.verified_demand_totals(raw, 99.9999999)
    with pytest.raises(ValueError, match="unsafe resource affine"):
        audit.check_resource_affine(Decimal(-100000000000) + 100, Decimal(-100000000000),
                                   Decimal(-100000000000) + 100, 100)


def _full_export(tmp_path, raw):
    path = tmp_path / "full_catalogue.json"
    path.write_text(json.dumps(raw))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    (tmp_path / "artifact_audit.json").write_text(json.dumps({"artifact_sha256": {path.name: digest}}))
    result = dict(method="full_path", catalogue_complete=True, support_complete=True, construction_complete=True,
                  model_version="consecutive", demand_formulation="preprocessed", max_orders=2,
                  assignment_scope="origin", flow_policy="aggregate", path_scope="full", catalogue=raw["counts"],
                  complete_catalogue_export=dict(file=path.name, complete=True, schema=raw["schema"], counts=raw["counts"], sha256=digest))
    return dict(result=result, solution={"flow": [0], "empty_flow": []})


def test_fresh_catalogue_rejects_omitted_zero_flow_path_and_wrong_hash(tmp_path):
    raw = dict(schema="mlnd-complete-catalogue/1", complete=True, scope="full", model_version="consecutive",
               demand_formulation="preprocessed", max_orders=2, assignment_scope="origin", flow_policy="aggregate",
               first_leg_consistency=False, counts=dict(segments=0, service=1, empty=0), segments=[], service=[], empty=[])
    payload = _full_export(tmp_path, raw)
    with pytest.raises(ValueError, match="Missing complete catalogue"):
        audit.fresh_catalogue(tmp_path, payload, tmp_path)
    payload["result"]["complete_catalogue_export"]["sha256"] = "wrong"
    with pytest.raises(ValueError, match="hash mismatch"):
        audit.fresh_catalogue(tmp_path, payload, tmp_path)


def test_audit_runtime_has_no_previous_output_directory_dependency():
    source = Path(audit.__file__).read_text()
    assert "outputs/" not in source
    assert "historical_full_cplex_reference" not in source
    assert "prior_queue_cut_audit" not in source


@pytest.mark.parametrize("method", bench.METHODS)
def test_record_configuration_rejects_model_budget_and_policy_drift(method):
    full = method == "full_cplex"
    preset = bench.FULL_PRESET if full else bench.PRESET
    enabled = bench.SHARED_NAMES if full else bench.BASELINE_ENABLED
    config = dict(schema="mlnd-solver-configuration/1", preset=preset, method="full_path" if full else "benders_cg",
                  demand_formulation="preprocessed", first_leg_consistency=False, seconds=600, gap=.01, threads=1,
                  seed=43, max_iterations=1000000, cg_max_columns=512, path_column_limit_argument=0,
                  export_catalogue=full, flags={name: name in enabled for name in bench.FLAG_NAMES})
    result = dict(preset=preset, resolved_configuration=config,
                  structural_strengthening={name: True for name in ("shared_rho_joint", "queue_perspective", "task_airline_bounds")})
    assert not bench.configuration_errors(result, method)
    config["seconds"] = 601
    assert bench.configuration_errors(result, method)
    config["seconds"] = 600
    config["flags"]["primal_asset_neighborhood"] = True
    assert bench.configuration_errors(result, method)
    config["flags"]["primal_asset_neighborhood"] = False
    result["structural_strengthening"]["task_airline_bounds"] = False
    assert bench.configuration_errors(result, method)


def test_preflight_only_invokes_inspect_and_records_every_command(tmp_path, monkeypatch):
    binary = tmp_path / "binary"
    binary.write_text("test executable identity")
    seen = []
    def inspect_only(command, **kwargs):
        assert "--inspect" in command and "--out" not in command
        assert kwargs["check"] is True and kwargs["timeout"] == 30
        seen.append(command)
        full = command[command.index("--method") + 1] == "full_path"
        identity = Path(command[command.index("--input") + 1]).name
        l, h, r, _ = bench.CASES[identity]
        preset = bench.FULL_PRESET if full else bench.PRESET
        enabled = bench.SHARED_NAMES if full else bench.BASELINE_ENABLED
        config = dict(schema="mlnd-solver-configuration/1", preset=preset, method="full_path" if full else "benders_cg",
            demand_formulation="preprocessed", first_leg_consistency=False, seconds=600, gap=.01, threads=1, seed=43,
            max_iterations=1000000, cg_max_columns=512, path_column_limit_argument=0, export_catalogue=full,
            flags={name: name in enabled for name in bench.FLAG_NAMES})
        return SimpleNamespace(stdout=json.dumps(dict(preset=preset, resolved_configuration=config,
            assignment_scope="origin", max_orders=2, n_l=l, n_h=h, n_od=r)), stderr="")
    monkeypatch.setattr(bench.subprocess, "run", inspect_only)
    output = bench.preflight(binary, tmp_path)
    assert len(seen) == len(output["rows"]) == 8
    assert output["passed"] and output["solver_invoked"] is False
    assert output["binary_sha256"] == bench.base.sha256(binary)
    assert all(row["stdout_sha256"] == hashlib.sha256(row["stdout"].encode()).hexdigest() for row in output["rows"])


def test_fresh_catalogue_reads_actual_cpp_export_metadata_and_rebuilds_zero_flow_paths(tmp_path, monkeypatch):
    from src.data import instances, standard
    from src.operations import validation
    root = Path(__file__).resolve().parents[1] / "fixtures/baseline_release_20261029"
    exported = root / "full_catalogue.json"
    probe = json.loads((root / "catalogue_export_probe.json").read_text())
    assert probe["passed"] and probe["solver_invoked"] is False
    raw = json.loads(exported.read_text())
    target = tmp_path / "full_catalogue.json"
    target.write_bytes(exported.read_bytes())
    metadata = probe["metadata"]
    assert metadata["file"] == target.name and "filename" not in metadata
    (tmp_path / "artifact_audit.json").write_text(json.dumps({"artifact_sha256": {target.name: metadata["sha256"]}}))
    result = dict(method="full_path", catalogue_complete=True, support_complete=True, construction_complete=True,
        path_scope=raw["scope"], catalogue=raw["counts"], complete_catalogue_export=metadata,
        path_record_counts=dict(service=1, empty=1), termination_reason="time_limit",
        **{key: raw[key] for key in ("model_version", "demand_formulation", "max_orders", "assignment_scope", "flow_policy")})
    payload = dict(result=result, solution=dict(flow=[1.0, 0.0], empty_flow=[0.0]),
        path_flows=dict(segments=raw["segments"], missions=[dict(path=raw["service"][0], flow=1.0)],
                        relocations=[dict(path=raw["empty"][0], flow=0.0)]))
    # The exporter probe deliberately uses synthetic, nonphysical paths.
    # This contract test isolates serialization/coverage and confirms every
    # zero-flow path reaches the real audit's physical-rebuild interface.
    rebuilt = []
    monkeypatch.setattr(standard, "load_standard", lambda *args, **kwargs: object())
    monkeypatch.setattr(instances, "to_planning_data", lambda value: value)
    monkeypatch.setattr(validation, "_segment", lambda data, path: rebuilt.append(("segment", path["id"])))
    monkeypatch.setattr(validation, "_physical", lambda data, path, empty=False: rebuilt.append(("empty" if empty else "service", path["id"])))
    catalogue, evidence = audit.fresh_catalogue(tmp_path, payload, tmp_path)
    assert rebuilt == [("segment", 0), ("service", 0), ("service", 1), ("empty", 0)]
    assert len(catalogue["missions"]) == 2
    assert evidence["current_record_coverage"]["service"] == dict(full_count=2, saved_count=1, nonzero_count=1)
    assert evidence["termination_reason"] == "time_limit"
