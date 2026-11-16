"""Frozen 40-job L2 size series, run serially by the detached generic dispatcher.

Only run/launch starts optimization. Native solution validation uses compact
saved path records; this suite never exports full catalogues or replays cuts.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

from src.experiments import conditional_benchmark_20261022 as base
from src.experiments import baseline_benchmark_20261029 as policy
from src.experiments.storage import write_json

ROOT = base.ROOT
MODULE = "src.experiments.l2_series_20261029"
RELEASE = ROOT / "outputs/l2_series_20261029"
DEFAULT_MANIFEST = RELEASE / "preparation_manifest.json"
DEFAULT_BATCH = RELEASE / "benchmark"
METHODS = ("benders_baseline", "full_cplex")
SIZES = ((2, 10, 20), (2, 20, 20), (2, 20, 40), (2, 40, 40))
REQUIRED_PROVENANCE = ("SPEC.md", "PROTOCOL.md", "BINARY_IDENTITY.json", "generate_instances.py",
    "instance_config.json", "INPUT_VALIDATION.json", "GENERATOR_TESTS.json", "RUNNER_TESTS.json")
_BASE_VERIFY, _BASE_RECORD, _BASE_GUARD = base.verify, base._record, base.guarded_execute


def jobs_for(instances):
    return [dict(id=f"{item['id']}_K2_{method}", instance=f"instances/{item['id']}",
                 dataset="tsl", map_id=item["id"], n_l=item["n_l"], n_h=item["n_h"], n_od=item["n_od"],
                 replicate=item["replicate"], generation_seed=item["seed"],
                 K=2, task_mode="double", method=method, variant=method,
                 output=f"runs/{item['id']}/{method}/K2")
            for item in instances for method in METHODS]


def protocol():
    return dict(suite="l2-series-20261029", seconds_per_job=600, threads_per_job=1,
        parallel_jobs=1, relative_gap_target=.01, seed=43, hard_wall_seconds=720,
        resource_limits=dict(job_rss_bytes=8 * 1024**3, batch_rss_bytes=12 * 1024**3,
            system_available_floor_bytes=2 * 1024**3, sample_seconds=.5,
            terminate_grace_seconds=5, pressure_cooldown_seconds=10),
        max_orders=2, demand_formulation="preprocessed", assignment_scope="origin", flow_policy="aggregate",
        baseline_preset=policy.PRESET, full_cplex_preset=policy.FULL_PRESET,
        baseline_enabled_flags=sorted(policy.BASELINE_ENABLED), full_cplex_enabled_flags=sorted(policy.SHARED_NAMES),
        same_binary_for_all_jobs=True, external_warm_start=None, external_root_certificate=None, fixed_design=None,
        full_path_column_limit=0, full_model_variable_limit=0, export_catalogue=False,
        cg_max_columns_per_iteration=512, max_benders_iterations=1000000,
        physical_instances=20, total_jobs=40, size_tiers=[list(s) for s in SIZES], replicates_per_tier=5,
        automatic_retries=False, independent_cut_audit=False,
        budget_scope="600 seconds input/path preparation/construction/optimization; output and native validation within 720-second external wall",
        validation_scope="native feasibility/cost and artifact/input/binary/configuration identity; no complete-catalogue or cut replay")


def command_for(batch, manifest, job, attempt):
    if job["method"] not in METHODS:
        raise ValueError("Undeclared method: " + job["method"])
    # Reuse the release's exact preset arguments, omitting its optional export.
    return [arg for arg in policy.command_for(batch, manifest, job, attempt) if arg != "--export-catalogue"]


def prepared_instances(path, *, inputs_root=None):
    path = Path(path).resolve()
    manifest = base.read(path)
    expected = dict(schema="tsl-matched-l2-preparation/1", status="prepared_validated_not_solved",
        physical_instances=20, expected_solve_jobs=40, K=2, demand_formulation="preprocessed",
        assignment_scope="origin", flow_policy="aggregate", solver_invoked=False)
    if any(manifest.get(k) != value for k, value in expected.items()):
        raise ValueError("Preparation manifest is incomplete or has a different model")
    rows = manifest["instances"]
    if len(rows) != 20 or len({r["id"] for r in rows}) != 20:
        raise ValueError("Exactly twenty unique prepared instances are required")
    expected_groups = {(s, r) for s in SIZES for r in range(1, 6)}
    actual_groups = {(tuple(row[k] for k in ("n_l", "n_h", "n_od")), row["replicate"]) for row in rows}
    if actual_groups != expected_groups:
        raise ValueError("Instance dimensions/replicates differ from the four declared tiers")
    for row in rows:
        if row["instance"] != f"instances/{row['id']}" or Path(row["id"]).name != row["id"] or row["id"] in (".", ".."):
            raise ValueError("Unsafe or mismatched instance identity")
        if row.get("status") != "prepared_validated_not_solved" or row["seed"] != 2026102900 + row["replicate"]:
            raise ValueError("Instance preparation/replicate seed differs")
        folder = Path(inputs_root or path.parent) / row["instance"]
        if base.sha256(folder / "manifest.json") != row["manifest_sha256"]:
            raise ValueError("Prepared input hash differs: " + row["id"])
        model = base.read(folder / "model.json")
        if any(model["values"].get(k) != row[k] for k in ("n_l", "n_h", "n_od")):
            raise ValueError("Prepared input dimensions differ: " + row["id"])
    return rows


def inspect(instances_manifest=DEFAULT_MANIFEST):
    rows = prepared_instances(instances_manifest)
    cfg = protocol()
    return dict(protocol=cfg, inputs=rows, solver_started=False,
        preparation_manifest_sha256=base.sha256(instances_manifest),
        jobs=[dict(job=job, command=command_for(Path("BATCH"), {"protocol": cfg}, job, "ATTEMPT"))
              for job in jobs_for(rows)])


def configuration_errors(result, method, *, check_structure=True):
    # The release's export option is incidental to the algorithm preset. Keep
    # every release check while explicitly requiring export off for this suite.
    config = result.get("resolved_configuration", {})
    adapted = dict(result, resolved_configuration=dict(config, export_catalogue=method == "full_cplex"))
    errors = policy.configuration_errors(adapted, method, check_structure=check_structure)
    if config.get("export_catalogue") is not False:
        errors.append("resolved configuration differs: export_catalogue must be false")
    return errors


def preflight(binary, instances_manifest=DEFAULT_MANIFEST):
    path = Path(instances_manifest).resolve()
    rows = []
    for job in jobs_for(prepared_instances(path)):
        command = command_for(Path("_unused_batch"), {"protocol": protocol()}, job, "_unused_output")
        command[0] = str(Path(binary).resolve())
        command[command.index("--input") + 1] = str(path.parent / job["instance"])
        index = command.index("--out")
        del command[index:index + 2]
        command.remove("--log")
        command.append("--inspect")
        completed = subprocess.run(command, capture_output=True, text=True, check=True, timeout=30)
        payload = json.loads(completed.stdout)
        errors = configuration_errors(payload, job["method"], check_structure=False)
        expected = dict(assignment_scope="origin", max_orders=2, n_l=job["n_l"], n_h=job["n_h"], n_od=job["n_od"])
        errors.extend("inspected input differs: " + k for k, v in expected.items() if payload.get(k) != v)
        if errors:
            raise ValueError(job["id"] + ": " + "; ".join(errors))
        rows.append(dict(job=job["id"], command=command, payload=payload, stdout=completed.stdout,
            stdout_sha256=hashlib.sha256(completed.stdout.encode()).hexdigest(), stderr=completed.stderr))
    return dict(passed=True, solver_invoked=False, binary_sha256=base.sha256(binary), rows=rows)


def prepare(batch=DEFAULT_BATCH, *, binary, instances_manifest=DEFAULT_MANIFEST, evidence=RELEASE):
    batch, binary, instances_manifest, evidence = (Path(p).resolve() for p in (batch, binary, instances_manifest, evidence))
    if batch.exists():
        raise FileExistsError("Refusing to replace existing batch: " + str(batch))
    checked = inspect(instances_manifest)
    identity = base.read(evidence / "BINARY_IDENTITY.json")
    binary_sha = base.sha256(binary)
    if identity.get("passed") is not True or identity.get("source_matches_validated_release") is not True or identity.get("binary_sha256") != binary_sha:
        raise ValueError("Binary identity must match the validated release")
    if not identity.get("source_sha256") or any(base.sha256(ROOT / p) != digest for p, digest in identity["source_sha256"].items()):
        raise ValueError("Production C++ differs from the validated binary sources")
    for name in REQUIRED_PROVENANCE:
        if not (evidence / name).is_file():
            raise FileNotFoundError(evidence / name)
    for name in ("INPUT_VALIDATION.json", "GENERATOR_TESTS.json", "RUNNER_TESTS.json"):
        if base.read(evidence / name).get("passed") is not True:
            raise ValueError("Preparation/public tests have not passed: " + name)
    inputs_checked = base.read(evidence / "INPUT_VALIDATION.json")
    expected_inputs = dict(preparation_manifest_sha256=base.sha256(instances_manifest),
        config_sha256=base.sha256(evidence / "instance_config.json"),
        generator_sha256=base.sha256(evidence / "generate_instances.py"))
    if any(inputs_checked.get(k) != v for k, v in expected_inputs.items()):
        raise ValueError("Input validation does not identify the generation sources/manifest")
    before = {name: policy.source_hashes(ROOT / name) for name in ("src", "cpp", "configs")}
    generation_sha = base.sha256(instances_manifest)
    preflight_evidence = preflight(binary, instances_manifest)
    base.prepare(batch, [instances_manifest.parent / row["instance"] for row in checked["inputs"]],
        binary=binary, seconds=600, threads=1, parallel=1, gap=.01, seed=43,
        job_rss_gib=8, batch_rss_gib=12, system_available_gib=2, hard_wall_seconds=720)
    manifest = base.read(batch / "experiment_manifest.json")
    for name, hashes in before.items():
        if policy.source_hashes(ROOT / name) != hashes or policy.source_hashes(batch / "frozen" / name) != hashes:
            raise ValueError("Source changed during freeze: " + name)
    if base.sha256(binary) != binary_sha or base.sha256(batch / "frozen/bin/mlnd_cpp") != binary_sha:
        raise ValueError("Binary changed during freeze")
    if base.sha256(instances_manifest) != generation_sha:
        raise ValueError("Preparation manifest changed during freeze")
    manifest.update(schema="l2-series-benchmark/1", jobs=jobs_for(checked["inputs"]), binary_sha256=binary_sha)
    manifest["protocol"].update(protocol())
    provenance = batch / "provenance"; provenance.mkdir()
    shutil.copy2(instances_manifest, provenance / "preparation_manifest.json")
    write_json(provenance / "PREFLIGHT.json", preflight_evidence)
    write_json(provenance / "predeclared_inputs.json", checked)
    optional = ("BEFORE.json", "before_git_status.txt", "REVIEW_SPEC.md", "REVIEW_STANDARDS.md", "test_generate_instances.py")
    for name in (*REQUIRED_PROVENANCE, *optional):
        if (evidence / name).is_file():
            shutil.copy2(evidence / name, provenance / name)
    shutil.copy2(ROOT / "tests/unit/test_l2_series_20261029.py", provenance / "test_l2_series_20261029.py")
    if (evidence / "provenance").is_dir():
        shutil.copytree(evidence / "provenance", provenance / "generation")
    for path in provenance.rglob("*"):
        if path.is_file():
            manifest["frozen_sha256"][str(path.relative_to(batch))] = base.sha256(path)
    (batch / "run.py").write_text("from pathlib import Path\nimport sys\n"
        "sys.dont_write_bytecode = True\nbatch = Path(__file__).resolve().parent\n"
        "sys.path.insert(0, str(batch / 'frozen'))\n"
        f"from {MODULE} import main\nmain([*sys.argv[1:], '--batch', str(batch)])\n")
    manifest["frozen_sha256"]["run.py"] = base.sha256(batch / "run.py")
    write_json(batch / "experiment_manifest.json", manifest)
    write_json(batch / "resource_limits.json", protocol()["resource_limits"])
    write_json(batch / "state.json", dict(status="prepared_not_started", updated_utc=base.existing.utc(),
        jobs={j["id"]: dict(status="pending", attempts=[]) for j in manifest["jobs"]}))
    verify(batch)
    return dict(batch=str(batch), jobs=40, binary_sha256=binary_sha, solver_started=False)


def verify(batch):
    batch = Path(batch).resolve()
    manifest = _BASE_VERIFY(batch)
    if manifest.get("schema") != "l2-series-benchmark/1":
        raise ValueError("Not a frozen L2 series batch")
    instances = prepared_instances(batch / "provenance/preparation_manifest.json", inputs_root=batch)
    if manifest["jobs"] != jobs_for(instances):
        raise ValueError("Frozen job set/order differs from the forty predeclared jobs")
    for key, value in protocol().items():
        if manifest["protocol"].get(key) != value:
            raise ValueError("Frozen protocol differs: " + key)
    binary_sha = base.sha256(batch / "frozen/bin/mlnd_cpp")
    if manifest["binary_sha256"] != binary_sha:
        raise ValueError("Binary identity differs")
    required = (*REQUIRED_PROVENANCE, "preparation_manifest.json", "PREFLIGHT.json")
    if any("provenance/" + name not in manifest["frozen_sha256"] for name in required):
        raise ValueError("Required generation/validation provenance was not frozen")
    identity = base.read(batch / "provenance/BINARY_IDENTITY.json")
    if identity.get("passed") is not True or identity.get("binary_sha256") != binary_sha:
        raise ValueError("Frozen validated-binary identity differs")
    evidence = base.read(batch / "provenance/PREFLIGHT.json")
    if evidence.get("passed") is not True or evidence.get("solver_invoked") is not False or evidence.get("binary_sha256") != binary_sha:
        raise ValueError("Preflight does not identify the frozen binary")
    if [row["job"] for row in evidence["rows"]] != [j["id"] for j in manifest["jobs"]]:
        raise ValueError("Preflight must cover all forty commands in order")
    for row, job in zip(evidence["rows"], manifest["jobs"]):
        if row["stdout_sha256"] != hashlib.sha256(row["stdout"].encode()).hexdigest() or json.loads(row["stdout"]) != row["payload"]:
            raise ValueError("Preflight output identity differs: " + row["job"])
        if configuration_errors(row["payload"], job["method"], check_structure=False):
            raise ValueError("Frozen resolved configuration differs: " + row["job"])
        expected = dict(assignment_scope="origin", max_orders=2, **{k: job[k] for k in ("n_l", "n_h", "n_od")})
        if any(row["payload"].get(k) != value for k, value in expected.items()):
            raise ValueError("Frozen preflight dimensions differ: " + row["job"])
    return manifest


def record(batch, job, state):
    batch = Path(batch)
    row = _BASE_RECORD(batch, job, state)
    row.update({key: job[key] for key in ("n_l", "n_h", "n_od", "replicate", "generation_seed")})
    folder = batch / state.get("result_directory", state.get("directory", "_missing"))
    result = base.optional_read(folder / "result.json")
    if not result:
        return row
    manifest = base.read(batch / "experiment_manifest.json")
    artifact = base.optional_read(folder / "artifact_audit.json")
    reference = base.optional_read(folder / "input_reference.json")
    valid = row["validation_passed"] and artifact.get("passed") is True
    for name in ("result.json", "solution.json", "path_flows.json", "input_reference.json", "preprocessed_demand.json", "native_validation.json"):
        valid = valid and (folder / name).is_file() and artifact.get("artifact_sha256", {}).get(name) == base.sha256(folder / name)
    expected = dict(binary_sha256=manifest["binary_sha256"], max_orders=2, solver="CPLEX",
        method="full_path" if job["method"] == "full_cplex" else "benders_cg", demand_formulation="preprocessed",
        assignment_scope="origin", flow_policy="aggregate", threads=1, seed=43, budget=600, requested_gap=.01,
        bound_scope="global", path_column_limit_argument=0, warm_start_source=None, fixed_design_source=None,
        task_airline_bounds=True)
    identity_errors = ["result identity differs: " + k for k, value in expected.items() if result.get(k) != value]
    if reference.get("first_leg_consistency") is not False:
        identity_errors.append("input reference first-leg policy differs")
    identity_errors += configuration_errors(result, job["method"])
    inspected = [r for r in base.read(batch / "provenance/PREFLIGHT.json")["rows"] if r["job"] == job["id"]]
    if len(inspected) != 1 or result.get("resolved_configuration") != inspected[0]["payload"]["resolved_configuration"]:
        identity_errors.append("final resolved configuration differs from frozen binary preflight")
    valid = bool(valid and not identity_errors)
    row.update(validation_passed=valid, upper_bound=result.get("objective") if valid else None,
        lower_bound=base.certified_bound(result) if valid else None, gap=None,
        binary_sha256=result.get("binary_sha256"), algorithm_preset=result.get("preset"),
        configuration_errors=identity_errors, algorithm_seconds=result.get("wall_before_output"),
        output_seconds=result.get("output_seconds"), reported_wall_seconds=result.get("wall"),
        target_gap_met=result.get("target_gap_met"), bound_scope=result.get("bound_scope"),
        recovered_from_checkpoint=result.get("recovered_from_checkpoint", False))
    if row["upper_bound"] is not None and row["lower_bound"] is not None:
        row["gap"] = (row["upper_bound"] - row["lower_bound"]) / max(1, abs(row["upper_bound"]))
    return row


def audit_worker(batch, job_id, attempt):
    """Existing physical incumbent validation plus this suite's identity policy."""
    with adapters():
        audit = base.audit_worker(batch, job_id, attempt)
    row = base.read(Path(attempt) / "report_row.json")
    if audit.get("passed") and not row["validation_passed"]:
        audit.update(passed=False, errors=[*audit.get("errors", []), "series configuration/artifact identity failed",
                                          *row.get("configuration_errors", [])])
        row["scheduler_status"] = "failed"
        write_json(Path(attempt) / "artifact_audit.json", audit)
        write_json(Path(attempt) / "report_row.json", row)
    return audit


def report(batch):
    batch = Path(batch).resolve()
    manifest, state = verify(batch), base.read(batch / "state.json")
    rows = [record(batch, job, state["jobs"][job["id"]]) for job in manifest["jobs"]]
    pairs = []
    for instance in dict.fromkeys(row["instance"] for row in rows):
        pair = {row["method"]: row for row in rows if row["instance"] == instance}
        a, b = pair["benders_baseline"], pair["full_cplex"]
        pairs.append(dict(instance=instance, **{key + "_baseline_minus_full": None if a.get(key) is None or b.get(key) is None
            else a[key] - b[key] for key in ("upper_bound", "lower_bound", "gap", "algorithm_seconds")}))
    output = batch / "analysis"; output.mkdir(exist_ok=True)
    payload = dict(schema="l2-series-comparison/1", protocol=manifest["protocol"], rows=rows, paired_differences=pairs,
        declared_jobs_terminal=all(row["scheduler_status"] in ("completed", "failed", "interrupted") for row in rows),
        all_jobs_completed=all(row["scheduler_status"] == "completed" for row in rows),
        validated_runs=sum(row["validation_passed"] for row in rows), independent_cut_audit_performed=False,
        validation_scope=manifest["protocol"]["validation_scope"],
        production_manifest_sha256=base.sha256(batch / "experiment_manifest.json"), binary_sha256=manifest["binary_sha256"])
    write_json(output / "comparison.json", payload)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (output / "comparison.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    def fmt(value):
        return "—" if value is None else f"{value:.6f}" if isinstance(value, float) else str(value)
    lines = ["# L2 四档规模串行比较", "", "20 个新实例、40 个任务；600 秒，1% gap，单线程，seed 43，严格串行。",
        "两法均使用 8 GiB 单任务 / 12 GiB 批次限制、2 GiB 系统可用内存底线；720 秒外部墙钟包含保存及验证。",
        "", f"终态任务齐全：{payload['declared_jobs_terminal']}；全部完成：{payload['all_jobs_completed']}；通过原生可行性及身份验证：{payload['validated_runs']}/40。",
        "未导出完整路径目录，未执行全量 cut 审计。上下界仅展示通过当前文件、输入、二进制及配置验证的运行。", "",
        "| 实例 / 方法 | 状态 | 验证 | UB | global LB | gap | 算法秒 | 进程秒 | 终止 |",
        "|---|---|---|---:|---:|---:|---:|---:|---|"]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(k)) for k in ("job", "scheduler_status", "validation_passed", "upper_bound",
            "lower_bound", "gap", "algorithm_seconds", "external_wall_seconds", "termination_reason")) + " |")
    lines += ["", "completed 指执行、保存和验证完成；time_limit、达到 1% gap 与精确最优不同。内存终止不表示数学不可行。",
        "失败保留原始日志、资源原因和状态；不自动重试、不追加内存、不用截断路径结果替代完整 CPLEX。",
        "配对差值仅在两法均有有效值时计算；单 seed 和这组地图的结果不代表普遍性能。"]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n")
    return dict(report=str(output / "REPORT.md"), jobs=len(rows), validated=payload["validated_runs"])


def guarded_execute(batch, manifest, job_id, command, attempt, cancel, **kwargs):
    command = list(command)
    if "-m" in command and command[command.index("-m") + 1] == base.__name__:
        command[command.index("-m") + 1] = MODULE
    return _BASE_GUARD(batch, manifest, job_id, command, attempt, cancel, **kwargs)


@contextmanager
def adapters():
    replacement = dict(command_for=command_for, _record=record, guarded_execute=guarded_execute, verify=verify, report=report)
    old = {name: getattr(base, name) for name in replacement}
    try:
        for name, value in replacement.items(): setattr(base, name, value)
        yield
    finally:
        for name, value in old.items(): setattr(base, name, value)


def run(batch):
    with adapters():
        return base.run(batch)  # No retry or concurrency override in this suite.


def launch(batch):
    with adapters():
        result = base.launch(batch)
    batch = Path(batch).resolve()
    result.update(manifest_sha256=base.sha256(batch / "experiment_manifest.json"),
        binary_sha256=base.sha256(batch / "frozen/bin/mlnd_cpp"), entrypoint_sha256=base.sha256(batch / "run.py"))
    write_json(batch / "dispatcher.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("inspect", "prepare", "verify", "launch", "run", "status", "report", "audit-worker"))
    parser.add_argument("--batch", "--out", type=Path, default=DEFAULT_BATCH)
    parser.add_argument("--binary", type=Path)
    parser.add_argument("--instances-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--evidence", type=Path, default=RELEASE)
    parser.add_argument("--job-id")
    parser.add_argument("--attempt", type=Path)
    args = parser.parse_args(argv)
    if args.action == "inspect": result = inspect(args.instances_manifest)
    elif args.action == "prepare":
        if args.binary is None: parser.error("prepare requires --binary")
        result = prepare(args.batch, binary=args.binary, instances_manifest=args.instances_manifest, evidence=args.evidence)
    elif args.action == "verify": result = dict(passed=bool(verify(args.batch)), solver_started=False)
    elif args.action == "run": result = run(args.batch)
    elif args.action == "launch": result = launch(args.batch)
    elif args.action == "report": result = report(args.batch)
    elif args.action == "audit-worker":
        if not args.job_id or args.attempt is None: parser.error("audit-worker requires --job-id and --attempt")
        result = audit_worker(args.batch, args.job_id, args.attempt)
    else: result = base.existing.status_summary(args.batch)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
