"""Full-domain audit constants must not be inferred from compact witnesses.

These are small saved-artifact fixtures only; no optimizer is imported or run.
"""
from copy import deepcopy
import json

import pytest

from src.experiments import task_cost_audit as audit


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def seal(folder):
    native = dict(passed=True, manifest_sha256=audit.read(folder / "result.json")["input_manifest_sha256"])
    for filename, field in (("result.json", "result_sha256"), ("solution.json", "solution_sha256"),
                            ("path_flows.json", "path_flows_sha256")):
        native[field] = audit.sha(folder / filename)
    write(folder / "native_validation.json", native)
    names = ["result.json", "solution.json", "path_flows.json", "input_reference.json",
             "preprocessed_demand.json", "native_validation.json"]
    write(folder / "artifact_audit.json", dict(passed=True, artifact_sha256={n: audit.sha(folder / n) for n in names}))


def pin(batch, source):
    result = audit.read(source / "result.json")
    row = dict(instance="case", K=2, validation_passed=True, source_directory=str(source),
               input_manifest_sha256=result["input_manifest_sha256"], all_service_paths=3, all_empty_paths=0)
    for name, field in (("result.json", "result_sha256"), ("native_validation.json", "native_validation_sha256"),
                        ("artifact_audit.json", "artifact_audit_sha256")):
        row[field] = audit.sha(source / name)
    ref = "provenance/historical_full_cplex.json"
    write(batch / ref, dict(rows=[row]))
    manifest = audit.read(batch / "experiment_manifest.json") if (batch / "experiment_manifest.json").exists() else {}
    manifest.update(protocol=dict(historical_full_cplex_reference=ref), frozen_sha256={ref: audit.sha(batch / ref)})
    write(batch / "experiment_manifest.json", manifest)


@pytest.fixture
def saved_batch(tmp_path):
    batch, source = tmp_path / "batch", tmp_path / "historical"
    instance = batch / "instances/case"
    csvs = {
        "coefficients/od_costs.csv": "od_id,ground_cost\nr,20\n",
        "od_demand.csv": "od_id,lambda_per_hour\nr,1\n",
        "coefficients/locker_levels.csv": "hub_id,level_id,occupancy_limit\nh0,0,100\nh1,0,100\n",
        "coefficients/links.csv": "from_site,to_site,allowed_links\nl,h0,1\nh0,h1,1\nh1,l,1\n",
    }
    for name, content in csvs.items():
        (instance / name).parent.mkdir(parents=True, exist_ok=True)
        (instance / name).write_text(content)
    manifest = dict(files={name: audit.sha(instance / name) for name in csvs})
    write(instance / "manifest.json", manifest)
    model = dict(ids=dict(ods=["r"], hubs=["h0", "h1"], levels=[0], flight_nodes=["l", "h0", "h1"]),
                 values=dict(n_od=1, n_l=1, n_h=2, fleet_cost=1))
    reference = dict(manifest=manifest, model=model, max_orders=2, assignment_scope="origin",
                     demand_formulation="preprocessed", first_leg_consistency=False)
    demand = dict(single_rates=[1], pair_rates=[[0]], head_rates=[1])
    catalogue = dict(missions=[], relocations=[], segments=[dict(id=i, test_segment=i) for i in range(3)])
    for i, (cost, time, slots) in enumerate([(1, 10, [[0, 10], [1, 1]]),
                                            (3, 1, [[0, 1], [1, 10]]),
                                            (6, 1, [[0, .2], [1, .2]])]):
        catalogue["missions"].append(dict(flow=1 if i == 0 else 0,
            path=dict(id=i, first_od=0, second_od=None, b=[0, 0], cost=cost, time=time,
                      slot_times=slots, arc_counts=[[0, 1, 1], [1, 2, 1], [2, 0, 1]])))
    result = dict(catalogue=dict(service=3, empty=0, segments=3), catalogue_complete=True, support_complete=True,
                  max_orders=2, input_manifest_sha256=audit.sha(instance / "manifest.json"),
                  objective=11, bound=0, gap=1, continuous_cost=11, termination_reason="time_limit")
    solution = dict(flow=[1, 0, 0], empty_flow=[], facility=[1, 1, 1], capacity_level=[[1], [1]],
                    assignment=[[1]], airline=[[0, 1, 0], [0, 0, 1], [1, 0, 0]], rho=[1])
    for folder in [source, batch / "off", batch / "on"]:
        for name, data in [("result", result), ("solution", solution), ("input_reference", reference),
                           ("preprocessed_demand", demand), ("path_flows", catalogue)]:
            write(folder / (name + ".json"), deepcopy(data))
    for folder, identifier in [(batch / "off", 0), (batch / "on", 2)]:
        r, s, p = (audit.read(folder / (name + ".json")) for name in ["result", "solution", "path_flows"])
        r.update(path_record_storage="nonzero_and_statistics_witnesses", path_record_counts=dict(service=1, empty=0))
        s["flow"] = [int(i == identifier) for i in range(3)]
        p["missions"] = [p["missions"][identifier]]
        p["missions"][0]["flow"] = 1
        for name, value in [("result", r), ("solution", s), ("path_flows", p)]:
            write(folder / (name + ".json"), value)
        seal(folder)
    seal(source)
    pin(batch, source)
    return batch, source, ("instances/case", 2)


def test_distinct_compact_supports_use_same_verified_full_extrema(saved_batch):
    batch, source, key = saved_batch
    full0, proof0 = audit.resolve_full_catalogue(batch, key, batch / "off")
    full1, proof1 = audit.resolve_full_catalogue(batch, key, batch / "on")
    assert audit.catalogue_digest(full0) == audit.catalogue_digest(full1)
    assert proof0["resolution"] == proof1["resolution"] == "verified_historical_full_catalogue"
    ref, demand = audit.read(source / "input_reference.json"), audit.read(source / "preprocessed_demand.json")
    constants = audit.physical_constants(ref["model"], demand, full0, batch / key[0])
    assert constants["saving"] == [16]  # Min cost + fleet*time is 4, not minimum flight cost 1.
    assert float(constants["total_load"][0]) == pytest.approx(.4)
    assert constants["catalogue_statistics"]["paths"] == 3
    for folder in [batch / "off", batch / "on"]:
        partial = audit.physical_constants(ref["model"], demand, audit.read(folder / "path_flows.json"), batch / key[0])
        assert partial["saving"] != constants["saving"]
    jobs = [dict(id=name, variant=name) for name in ["off", "on"]]
    state = dict(jobs={name: dict(status="completed", result_directory=name) for name in ["off", "on"]})
    checked = audit.audit_case(batch, key, jobs, state)
    assert checked["passed"], checked["errors"]
    assert checked["catalogue_statistics"]["paths"] == 3
    assert checked["audited_runs"] == 2


def test_dense_legacy_needs_no_external_reference(saved_batch):
    batch, source, key = saved_batch
    (batch / "experiment_manifest.json").unlink()
    full, proof = audit.resolve_full_catalogue(batch, key, source)
    assert len(full["missions"]) == 3
    assert proof["resolution"] == "local_full_catalogue"


def test_explicit_global_ids_not_record_positions(saved_batch):
    batch, source, key = saved_batch
    p = audit.read(source / "path_flows.json")
    p["missions"] = [p["missions"][2], p["missions"][0], p["missions"][1]]
    write(source / "path_flows.json", p)
    seal(source)
    pin(batch, source)
    audit.resolve_full_catalogue(batch, key, batch / "on")


@pytest.mark.parametrize("mutation,match", [
    ("missing_reference", "requires a verified full catalogue"),
    ("unfrozen_reference", "provenance is not frozen"),
    ("source_tamper", "hash mismatch"),
    ("incomplete_source", "incomplete full catalogue"),
    ("demand", "task demand mismatch"),
    ("input", "input mismatch"),
    ("current_path", "path differs"),
    ("omitted_nonzero", "omitted nonzero"),
    ("omitted_tiny_nonzero", "omitted nonzero"),
    ("flow_mismatch", "record/vector flow mismatch"),
    ("duplicate_id", "duplicate global path id"),
])
def test_rejects_unverified_or_incompatible_catalogues(saved_batch, mutation, match):
    batch, source, key = saved_batch
    current = batch / "off"
    if mutation in ["missing_reference", "unfrozen_reference"]:
        m = audit.read(batch / "experiment_manifest.json")
        m["protocol" if mutation == "missing_reference" else "frozen_sha256"] = {}
        write(batch / "experiment_manifest.json", m)
    elif mutation == "source_tamper":
        (source / "result.json").write_text((source / "result.json").read_text() + " ")
    elif mutation in ["incomplete_source", "demand", "input"]:
        name = {"incomplete_source": "path_flows", "demand": "preprocessed_demand", "input": "input_reference"}[mutation]
        p = audit.read(source / (name + ".json"))
        if mutation == "incomplete_source": p["missions"].pop()
        elif mutation == "demand": p["single_rates"][0] += .1
        else: p["model"]["values"]["fleet_cost"] += 1
        write(source / (name + ".json"), p)
        seal(source)
        pin(batch, source)
    else:
        p = audit.read(current / "path_flows.json")
        if mutation == "current_path": p["missions"][0]["path"]["time"] += .1
        elif mutation == "omitted_nonzero": p["missions"] = []
        elif mutation == "omitted_tiny_nonzero":
            s = audit.read(current / "solution.json")
            s["flow"][1] = 1e-300
            write(current / "solution.json", s)
        elif mutation == "flow_mismatch": p["missions"][0]["flow"] += 1e-10
        else: p["missions"].append(deepcopy(p["missions"][0]))
        write(current / "path_flows.json", p)
        r = audit.read(current / "result.json")
        r["path_record_counts"]["service"] = len(p["missions"])
        write(current / "result.json", r)
        seal(current)
    with pytest.raises(ValueError, match=match):
        audit.resolve_full_catalogue(batch, key, current)
