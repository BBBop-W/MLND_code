import json
from pathlib import Path

import pytest

from src.experiments import task_cost_benchmark_20261027 as bench


def reference_fixture(tmp_path, monkeypatch):
    reference = tmp_path / "reference"
    reference.mkdir()
    cfg = dict(seconds_per_job=3600.0, threads_per_job=1, seed=43,
               relative_gap_target=.01, common_solver_args=list(bench.unified.COMMON_FLAGS),
               algorithm_solver_args=list(bench.unified.ALGORITHM_FLAGS),
               cg_max_columns_per_iteration=512, max_benders_iterations=1000000)
    jobs = []
    for identity, mode, k in (*bench.SUITES["default"], *bench.SUITES["expanded"]):
        dataset, level, map_id = identity.split("_")
        job = dict(id=f"{identity}_{mode}_benders_unified", instance=f"instances/{identity}",
                   dataset=dataset, map_id=map_id, n_l=int(level[1:]), K=k, task_mode=mode,
                   method="benders_unified", output=f"runs/{identity}/{mode}/benders_unified")
        jobs.append(job)
        (reference / job["instance"]).mkdir(parents=True)
    manifest = dict(protocol=cfg, jobs=jobs)
    (reference / "experiment_manifest.json").write_text(json.dumps(manifest))
    for job in jobs:
        if job["dataset"] == "nanshan":
            continue  # Prepared input, no claim of a historical execution.
        target = reference / job["output"] / "attempts/001/job.json"
        target.parent.mkdir(parents=True)
        command = bench.unified.command_for(reference, manifest, job, target.parent)
        target.write_text(json.dumps(dict(job=job, command=command)))
    monkeypatch.setattr(bench, "REFERENCE", reference)
    return reference, manifest


def test_expanded_reference_preserves_unrun_nanshan_provenance(tmp_path, monkeypatch):
    reference, _ = reference_fixture(tmp_path, monkeypatch)
    refs = bench.reference_jobs("expanded")
    assert len(refs) == 4
    assert {r["identity"] for r in refs} == {x[0] for x in bench.SUITES["expanded"]}
    nanshan = next(r for r in refs if r["job"]["dataset"] == "nanshan")
    assert nanshan["reference_source"] == "manifest_recipe"
    assert nanshan["reference_job"] is None and nanshan["reference_job_sha256"] is None
    assert not (reference / "runs/nanshan_L01_M01/double/benders_unified/attempts/001/job.json").exists()
    assert all(r["reference_source"] == "saved_job" for r in refs if r is not nanshan)
    assert bench.reference_jobs()[0]["identity"] == "tsl_L01_M01"


def test_reference_recipe_mismatch_is_rejected(tmp_path, monkeypatch):
    reference, _ = reference_fixture(tmp_path, monkeypatch)
    target = reference / "runs/tsl_L01_M02/double/benders_unified/attempts/001/job.json"
    item = json.loads(target.read_text())
    item["command"][item["command"].index("--seed") + 1] = "44"
    target.write_text(json.dumps(item))
    with pytest.raises(ValueError, match="differs from its manifest"):
        bench.reference_jobs("expanded")


def test_four_variants_only_add_declared_strengthening_flags(tmp_path, monkeypatch):
    reference_fixture(tmp_path, monkeypatch)
    for item in bench.reference_jobs("expanded"):
        for variant, flags in bench.VARIANTS.items():
            job = dict(reference_command=item["command"], instance=item["job"]["instance"], variant=variant)
            original = list(item["command"])
            command = bench.command_for(Path("/frozen-batch"),
                {"protocol": {"seconds_per_job": 600}}, job, "/new-attempt")
            option = lambda key: command[command.index(key) + 1]
            assert option("--seconds") == "600" and option("--threads") == "1"
            assert option("--seed") == "43" and option("--gap") == "0.01"
            assert option("--input") == "/frozen-batch/" + job["instance"]
            assert command[len(original):] == ["--path-support-cuts", *flags]
            assert item["command"] == original


def test_expanded_prepare_freezes_metadata_and_requires_same_binary(tmp_path, monkeypatch):
    reference_fixture(tmp_path, monkeypatch)
    binary = tmp_path / "mlnd_cpp"
    binary.write_text("fixture executable, never run")
    pilot = tmp_path / "pilot"
    pilot.mkdir()
    (pilot / "experiment_manifest.json").write_text(json.dumps({"binary_sha256": bench.base.sha256(binary)}))
    monkeypatch.setattr(bench, "DEFAULT_BATCH", pilot)
    calls = []
    def fake_prepare(batch, instances, **kwargs):
        calls.append((list(instances), kwargs))
        batch = Path(batch)
        batch.mkdir()
        frozen = batch / "frozen/bin/mlnd_cpp"
        frozen.parent.mkdir(parents=True)
        frozen.write_bytes(binary.read_bytes())
        (batch / "experiment_manifest.json").write_text(json.dumps(dict(protocol={}, frozen_sha256={})))
    monkeypatch.setattr(bench.base, "prepare", fake_prepare)
    batch = tmp_path / "expanded"
    result = bench.prepare(batch, binary=binary, suite="expanded")
    assert result["jobs"] == 16 and result["solver_started"] is False
    manifest = json.loads((batch / "experiment_manifest.json").read_text())
    assert manifest["protocol"]["physical_instances"] == 4
    assert manifest["protocol"]["total_jobs"] == 16
    assert len(manifest["jobs"]) == len({j["id"] for j in manifest["jobs"]}) == 16
    assert {j["dataset"] for j in manifest["jobs"]} == {"tsl", "nanshan"}
    assert {j["map_id"] for j in manifest["jobs"]} == {"M01", "M02", "M03"}
    assert {j["n_l"] for j in manifest["jobs"]} == {1, 2}
    assert all(j["reference_job_sha256"] is None for j in manifest["jobs"] if j["dataset"] == "nanshan")
    assert "'--suite', 'expanded'" in (batch / "run.py").read_text()
    assert (batch / "provenance/reference_experiment_manifest.json").is_file()
    assert not (batch / "provenance/nanshan_L01_M01/reference_job.json").exists()
    assert len(calls[0][0]) == 4
    assert {key: calls[0][1][key] for key in ("seconds", "threads", "parallel", "gap", "seed")} == {
        "seconds": 600, "threads": 1, "parallel": 4, "gap": .01, "seed": 43}
    binary.write_text("different binary")
    with pytest.raises(ValueError, match="same binary"):
        bench.prepare(tmp_path / "bad", binary=binary, suite="expanded")
    assert len(calls) == 1  # Rejected before any freeze or dispatch.


def test_adapter_restores_functions():
    original = bench.base.command_for
    with bench.adapters():
        assert bench.base.command_for is bench.command_for
    assert bench.base.command_for is original
