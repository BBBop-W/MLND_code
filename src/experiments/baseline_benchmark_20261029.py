"""Frozen eight-job baseline/full-CPLEX comparison on predeclared existing maps.

Only run/launch starts solvers. Scheduling, resource protection and independent
native validation use the established generic runner through one adapter layer.
No previous experiment is imported or used as a performance comparator.
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
from src.experiments.storage import write_json

ROOT = base.ROOT
MODULE = "src.experiments.baseline_benchmark_20261029"
RELEASE = ROOT / "outputs/baseline_release_20261029"
DEFAULT_BATCH = RELEASE / "benchmark"
MAPS = ROOT / "outputs/unified_maps_20261027"
PRESET = "baseline-20261029"
FULL_PRESET = "full-cplex-20261029"
CASES = {
    "tsl_L01_M07": (1, 10, 25, "3b582f3a24fa4111741491f7e140dcd025685937b64d6cd57c4de4d0ba5327c2"),
    "tsl_L01_M08": (1, 10, 25, "3ce136572deb0a29f5c768ec2bce346dcb3d9a73b5bbbf3cbe9e39ebe04a0e1c"),
    "tsl_L02_M02": (2, 20, 50, "37e2e3283a2c74052cd5b84a7f9f1e0d83d35814c89f71869bb6435111c66fce"),
    "tsl_L02_M03": (2, 20, 50, "956e4083e70e173e29002576b5127571cb5d8b2a05cebff37831436ffe610bb1"),
}
METHODS = ("benders_baseline", "full_cplex")
SHARED_FLAGS = ("--shared-rho-joint", "--queue-perspective", "--tight-airline-bounds", "--task-airline-bounds")
SHARED_NAMES = frozenset(flag[2:].replace("-", "_") for flag in SHARED_FLAGS)
BASELINE_ENABLED = SHARED_NAMES | frozenset((
    "pareto_cuts", "reuse_dual_lp", "reuse_columns", "courier_savings_bound", "initial_design", "warm_start",
    "incumbent_cleanup", "primal_search", "branch_covering_seeds", "resource_bounds", "capacity_cuts",
    "resource_lp_cuts", "adaptive_master", "neighborhood_cuts", "continuous_root", "root_projection", "root_inout",
    "auxiliary_valid_cuts", "resource_airline_bounds", "root_cuts", "restricted_mip", "local_branching",
    "incumbent_cutoff", "od_airline_bounds", "topology_cuts", "path_support_cuts", "task_savings_bound",
    "task_resource_cuts", "airline_connectivity_cuts", "airline_resource_cuts", "primal_polish", "search_scheduling",
    "fractional_path_support", "airline_resource_balance", "weak_dual_cuts", "interior_cuts", "queue_polymatroid_cuts",
    "queue_price_search", "target_master_refinement", "queue_fractional_separation"))
FLAG_NAMES = BASELINE_ENABLED | frozenset(("dual_lp_first", "primal_asset_neighborhood", "assignment_incident_clique",
    "assignment_support_cuts", "double_support_cuts", "joint_base_cost_cuts"))
_BASE_VERIFY, _BASE_RECORD, _BASE_GUARD = base.verify, base._record, base.guarded_execute


def jobs_for():
    return [dict(id=f"{identity}_K2_{method}", instance=f"instances/{identity}",
                 dataset="tsl", map_id=identity[-3:], n_l=size[0], n_h=size[1], n_od=size[2],
                 K=2, task_mode="double", method=method, variant=method,
                 output=f"runs/{identity}/{method}/K2")
            for identity, size in CASES.items() for method in METHODS]


def protocol():
    return dict(suite="baseline-release-20261029", seconds_per_job=600, threads_per_job=1,
                parallel_jobs=3, relative_gap_target=.01, seed=43, hard_wall_seconds=720,
                max_orders=2, demand_formulation="preprocessed", assignment_scope="origin",
                flow_policy="aggregate", solver="CPLEX", baseline_preset=PRESET,
                full_cplex_preset=FULL_PRESET,
                baseline_enabled_flags=sorted(BASELINE_ENABLED), full_cplex_enabled_flags=sorted(SHARED_NAMES),
                common_solver_args=list(SHARED_FLAGS), same_binary_for_all_jobs=True,
                external_warm_start=None, external_root_certificate=None, fixed_design=None,
                full_path_column_limit=0, full_model_variable_limit=0,
                cg_max_columns_per_iteration=512, max_benders_iterations=1000000,
                physical_instances=4, total_jobs=8, cases=list(CASES),
                comparator="fresh complete-domain full_path MISOCP; shared equivalent strengthening",
                selection_rule="first two TSL L1 maps absent from strengthening-development batches, then predeclared L2 M02/M03; no performance selection",
                budget_scope="input read, path preparation, model construction and optimization; validation reported separately")


def command_for(batch, manifest, job, attempt):
    cfg = manifest["protocol"]
    command = [str(Path(batch) / "frozen/bin/mlnd_cpp"), "--preset",
        FULL_PRESET if job["method"] == "full_cplex" else PRESET,
        "--input", str(Path(batch) / job["instance"]),
        "--method", "full_path" if job["method"] == "full_cplex" else "benders_cg",
        "--k", "2", "--demand-formulation", "preprocessed", "--seconds", str(cfg["seconds_per_job"]),
        "--threads", str(cfg["threads_per_job"]), "--seed", str(cfg["seed"]),
        "--gap", str(cfg["relative_gap_target"]), "--path-column-limit", "0", "--log", "--out", str(attempt)]
    if job["method"] == "benders_baseline":
        command += ["--cg-max-columns", str(cfg["cg_max_columns_per_iteration"]),
                    "--max-iterations", str(cfg["max_benders_iterations"])]
    elif job["method"] == "full_cplex":
        command += ["--export-catalogue"]
    else:
        raise ValueError("Undeclared method: " + job["method"])
    return command


def inspect(maps=MAPS):
    maps = Path(maps)
    rows = []
    for identity, (l, h, r, digest) in CASES.items():
        folder = maps / "instances" / identity
        model = base.read(folder / "model.json")
        actual = tuple(int(model["values"][k]) for k in ("n_l", "n_h", "n_od"))
        if actual != (l, h, r) or base.sha256(folder / "manifest.json") != digest:
            raise ValueError("Predeclared input identity differs: " + identity)
        rows.append(dict(instance=identity, bases=l, lockers=h, ods=r, input_manifest_sha256=digest))
    cfg = protocol()
    return dict(protocol=cfg, inputs=rows, jobs=[dict(job=j, command=command_for(Path("BATCH"),
                {"protocol": cfg}, j, "ATTEMPT")) for j in jobs_for()], solver_started=False,
                previous_use="Existing maps appeared in the early unified benchmark; no strengthening-development use found for these four.")


def source_hashes(root):
    root = Path(root)
    return {str(p.relative_to(root)): base.sha256(p) for p in root.rglob("*") if p.is_file()
            and not any(part.startswith("build") or part == "__pycache__" for part in p.relative_to(root).parts)
            and p.name != ".DS_Store" and p.suffix != ".pyc"}


def preflight(binary, maps):
    """Read inputs/resolved CLI policy; --inspect exits before path preparation."""
    rows = []
    for job in jobs_for():
        command = command_for(Path("_unused_batch"), {"protocol": protocol()}, job, "_unused_output")
        command[0] = str(binary)
        command[command.index("--input") + 1] = str(Path(maps) / job["instance"])
        at = command.index("--out")
        del command[at:at + 2]
        command.remove("--log")
        command.append("--inspect")
        completed = subprocess.run(command, capture_output=True, text=True, check=True, timeout=30)
        payload = json.loads(completed.stdout)
        errors = configuration_errors(payload, job["method"], check_structure=False)
        expected = dict(assignment_scope="origin", max_orders=2, n_l=job["n_l"], n_h=job["n_h"], n_od=job["n_od"])
        errors.extend("inspected input differs: " + key for key, value in expected.items() if payload.get(key) != value)
        if errors:
            raise ValueError(job["id"] + ": " + "; ".join(errors))
        rows.append(dict(job=job["id"], command=command, payload=payload, stdout=completed.stdout,
                         stdout_sha256=hashlib.sha256(completed.stdout.encode()).hexdigest(), stderr=completed.stderr))
    return dict(passed=True, solver_invoked=False, binary_sha256=base.sha256(binary), rows=rows)


def prepare(batch=DEFAULT_BATCH, *, binary, maps=MAPS, evidence=RELEASE):
    batch, binary, maps, evidence = map(lambda p: Path(p).resolve(), (batch, binary, maps, evidence))
    checked = inspect(maps)
    validation = base.read(evidence / "VALIDATION.json")
    binary_sha = base.sha256(binary)
    if validation.get("binary_sha256") != binary_sha or validation.get("build_exit_code") != 0:
        raise ValueError("Validation must identify this successful final binary build")
    probes = validation.get("probes", [])
    if not probes:
        raise ValueError("Final validation must identify passing probe artifacts")
    for probe in probes:
        if probe.get("passed") is not True or base.sha256(evidence / probe["file"]) != probe["sha256"]:
            raise ValueError("Probe provenance differs: " + probe["file"])
    before = {name: source_hashes(ROOT / name) for name in ("src", "cpp", "configs")}
    preflight_evidence = preflight(binary, maps)
    base.prepare(batch, [maps / "instances" / name for name in CASES], binary=binary, seconds=600,
                 threads=1, parallel=3, gap=.01, seed=43, job_rss_gib=4, batch_rss_gib=12,
                 system_available_gib=2, hard_wall_seconds=720)
    manifest = base.read(batch / "experiment_manifest.json")
    for name, hashes in before.items():
        if source_hashes(ROOT / name) != hashes or source_hashes(batch / "frozen" / name) != hashes:
            raise ValueError("Source changed during freeze: " + name)
    if base.sha256(binary) != binary_sha or base.sha256(batch / "frozen/bin/mlnd_cpp") != binary_sha:
        raise ValueError("Binary changed during freeze")
    manifest.update(schema="baseline-benchmark/1", jobs=jobs_for(), binary_sha256=binary_sha)
    manifest["protocol"].update(protocol())
    provenance = batch / "provenance"
    provenance.mkdir()
    write_json(provenance / "predeclared_inputs.json", checked)
    write_json(provenance / "PREFLIGHT.json", preflight_evidence)
    required = ["PROTOCOL.md", "CASE_SELECTION.json", "AUDIT_EXTRACTION.json", "VALIDATION.json"]
    optional = ["before_manifest.json", "before_git_status.txt", "REVIEW_SPEC.md", "REVIEW_STANDARDS.md",
                "final_build.log", "PYTHON_VALIDATION.json"]
    for name in dict.fromkeys(required + [p["file"] for p in probes] + optional):
        source = evidence / name
        if name in required and not source.is_file():
            raise FileNotFoundError(source)
        if source.is_file():
            target = provenance / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    for path in provenance.rglob("*"):
        if path.is_file():
            manifest["frozen_sha256"][str(path.relative_to(batch))] = base.sha256(path)
    (batch / "run.py").write_text("from pathlib import Path\nimport sys\n"
        "sys.dont_write_bytecode = True\nbatch = Path(__file__).resolve().parent\n"
        "sys.path.insert(0, str(batch / 'frozen'))\n"
        f"from {MODULE} import main\nmain([*sys.argv[1:], '--batch', str(batch)])\n")
    manifest["frozen_sha256"]["run.py"] = base.sha256(batch / "run.py")
    write_json(batch / "experiment_manifest.json", manifest)
    write_json(batch / "state.json", dict(status="prepared_not_started", updated_utc=base.existing.utc(),
        jobs={j["id"]: dict(status="pending", attempts=[]) for j in manifest["jobs"]}))
    verify(batch)
    return dict(batch=str(batch), jobs=8, binary_sha256=binary_sha, solver_started=False)


def verify(batch):
    batch = Path(batch).resolve()
    manifest = _BASE_VERIFY(batch)
    if manifest["jobs"] != jobs_for():
        raise ValueError("Frozen job set/order differs from the eight predeclared jobs")
    for key, value in protocol().items():
        if manifest["protocol"].get(key) != value:
            raise ValueError("Frozen protocol differs: " + key)
    limits = manifest["protocol"]["resource_limits"]
    if tuple(limits[k] for k in ("job_rss_bytes", "batch_rss_bytes", "system_available_floor_bytes")) != (
            4 * 1024**3, 12 * 1024**3, 2 * 1024**3):
        raise ValueError("Resource protections differ")
    if manifest["binary_sha256"] != base.sha256(batch / "frozen/bin/mlnd_cpp"):
        raise ValueError("Binary identity differs")
    for identity, (_, _, _, digest) in CASES.items():
        if base.sha256(batch / "instances" / identity / "manifest.json") != digest:
            raise ValueError("Input identity differs: " + identity)
    # Required metadata paths are checked now, before any expensive audit work.
    for name in ("PROTOCOL.md", "CASE_SELECTION.json", "AUDIT_EXTRACTION.json", "VALIDATION.json", "PREFLIGHT.json"):
        if "provenance/" + name not in manifest["frozen_sha256"]:
            raise ValueError("Required provenance was not frozen: " + name)
    preflight_evidence = base.read(batch / "provenance/PREFLIGHT.json")
    if preflight_evidence.get("passed") is not True or preflight_evidence.get("binary_sha256") != manifest["binary_sha256"]:
        raise ValueError("Preflight does not identify the frozen binary")
    if [r["job"] for r in preflight_evidence["rows"]] != [j["id"] for j in manifest["jobs"]]:
        raise ValueError("Preflight does not cover all eight commands in order")
    for row in preflight_evidence["rows"]:
        if row["stdout_sha256"] != hashlib.sha256(row["stdout"].encode()).hexdigest() or json.loads(row["stdout"]) != row["payload"]:
            raise ValueError("Preflight output identity differs: " + row["job"])
    return manifest


def guarded_execute(batch, manifest, job_id, command, attempt, cancel, **kwargs):
    command = list(command)
    if "-m" in command:
        index = command.index("-m") + 1
        if command[index] == base.__name__:
            command[index] = MODULE
    return _BASE_GUARD(batch, manifest, job_id, command, attempt, cancel, **kwargs)


def configuration_errors(result, method, *, check_structure=True):
    full = method == "full_cplex"
    preset = FULL_PRESET if full else PRESET
    config = result.get("resolved_configuration", {})
    expected = dict(schema="mlnd-solver-configuration/1", preset=preset,
        method="full_path" if full else "benders_cg", demand_formulation="preprocessed", first_leg_consistency=False,
        seconds=600, gap=.01, threads=1, seed=43, max_iterations=1000000, cg_max_columns=512,
        path_column_limit_argument=0, export_catalogue=full)
    errors = ["resolved configuration differs: " + k for k, value in expected.items() if config.get(k) != value]
    if result.get("preset") != preset:
        errors.append("top-level preset differs")
    enabled = SHARED_NAMES if full else BASELINE_ENABLED
    if config.get("flags") != {name: name in enabled for name in FLAG_NAMES}:
        errors.append("resolved acceleration flags differ")
    if check_structure:
        structure = result.get("structural_strengthening", {})
        for name in ("shared_rho_joint", "queue_perspective", "task_airline_bounds"):
            if structure.get(name) is not True:
                errors.append("common structure differs: " + name)
    return errors


def record(batch, job, state):
    batch = Path(batch)
    row = _BASE_RECORD(batch, job, state)
    row.update({k: job[k] for k in ("K", "task_mode", "dataset", "map_id", "n_l", "n_h", "n_od")})
    folder = batch / state.get("result_directory", state.get("directory", "_missing"))
    result = base.optional_read(folder / "result.json")
    if not result:
        return row
    manifest = base.read(batch / "experiment_manifest.json")
    reference = base.optional_read(folder / "input_reference.json")
    artifact = base.optional_read(folder / "artifact_audit.json")
    valid = row["validation_passed"] and artifact.get("passed") is True
    for name in ("result.json", "solution.json", "path_flows.json", "input_reference.json", "preprocessed_demand.json", "native_validation.json"):
        valid = valid and (folder / name).is_file() and artifact.get("artifact_sha256", {}).get(name) == base.sha256(folder / name)
    expected = dict(binary_sha256=manifest["binary_sha256"], max_orders=2, solver="CPLEX",
                    method="full_path" if job["method"] == "full_cplex" else "benders_cg",
                    demand_formulation="preprocessed", assignment_scope="origin", flow_policy="aggregate",
                    threads=1, seed=43, budget=600, requested_gap=.01)
    valid = valid and all(result.get(k) == v for k, v in expected.items())
    valid = valid and reference.get("first_leg_consistency") is False
    valid = valid and result.get("bound_scope") == "global" and result.get("path_column_limit_argument") == 0
    valid = valid and result.get("warm_start_source") is None and result.get("fixed_design_source") is None
    valid = valid and result.get("task_airline_bounds") is True
    identity_errors = configuration_errors(result, job["method"])
    preflight_evidence = base.read(batch / "provenance/PREFLIGHT.json")
    inspected = [r for r in preflight_evidence["rows"] if r["job"] == job["id"]]
    if len(inspected) != 1 or result.get("resolved_configuration") != inspected[0]["payload"]["resolved_configuration"]:
        identity_errors.append("final resolved configuration differs from frozen binary preflight")
    valid = valid and not identity_errors
    row.update(validation_passed=bool(valid), upper_bound=result.get("objective") if valid else None,
               lower_bound=base.certified_bound(result) if valid else None, gap=None,
               binary_sha256=result.get("binary_sha256"), assignment_scope=result.get("assignment_scope"),
               algorithm_preset=result.get("preset"), resolved_configuration=result.get("resolved_configuration"),
               configuration_errors=identity_errors,
               algorithm_seconds=result.get("wall_before_output"), output_seconds=result.get("output_seconds"),
               reported_wall_seconds=result.get("wall"), root_lower_bound=result.get("root_diagnostics", {}).get("lower_bound"),
               target_gap_met=result.get("target_gap_met"), bound_scope=result.get("bound_scope"),
               recovered_from_checkpoint=result.get("recovered_from_checkpoint", False),
               path_record_storage=result.get("path_record_storage"), pricing_domain=result.get("pricing_domain"))
    if row["upper_bound"] is not None and row["lower_bound"] is not None:
        row["gap"] = (row["upper_bound"] - row["lower_bound"]) / max(1, abs(row["upper_bound"]))
    for name in ("queue_polymatroid_summary", "queue_price_search_summary", "queue_fractional_summary", "target_master_refinement_summary"):
        row[name] = result.get(name)
    return row


def report(batch):
    batch = Path(batch).resolve()
    manifest, state = verify(batch), base.read(batch / "state.json")
    rows = [record(batch, j, state["jobs"][j["id"]]) for j in manifest["jobs"]]
    audit = base.optional_read(batch / "analysis/CUT_AUDIT.json")
    audit_current = audit.get("production_manifest_sha256") == base.sha256(batch / "experiment_manifest.json")
    audit_current = audit_current and all((batch / name).is_file() and base.sha256(batch / name) == digest
                                        for name, digest in audit.get("artifact_sha256", {}).items())
    audit_current = audit_current and audit.get("reviewer_sha256") == base.sha256(Path(__file__).with_name("baseline_cut_audit_20261029.py"))
    pairs = []
    for identity in CASES:
        pair = {r["method"]: r for r in rows if r["instance"] == "instances/" + identity}
        a, b = pair["benders_baseline"], pair["full_cplex"]
        pairs.append(dict(instance=identity, **{key + "_baseline_minus_full":
            None if a.get(key) is None or b.get(key) is None else a[key] - b[key]
            for key in ("upper_bound", "lower_bound", "gap", "algorithm_seconds")}))
    output = batch / "analysis"
    output.mkdir(exist_ok=True)
    payload = dict(schema="baseline-comparison/1", rows=rows, paired_differences=pairs,
                   production_manifest_sha256=base.sha256(batch / "experiment_manifest.json"),
                   independent_cut_audit_passed=bool(audit_current and audit.get("passed") and audit.get("complete")))
    write_json(output / "comparison.json", payload)
    fields = ["job", "scheduler_status", "validation_passed", "upper_bound", "lower_bound", "gap",
              "algorithm_seconds", "external_wall_seconds", "termination_reason", "root_lower_bound",
              "catalogue_complete", "construction_complete", "all_paths", "peak_rss_bytes"]
    with (output / "comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)
    def fmt(value):
        return "—" if value is None else f"{value:.6f}" if isinstance(value, float) else str(value)
    lines = ["# 新 baseline 与完整 CPLEX 对比", "", "四例预先锁定；每种方法均为本批新运行。600 秒覆盖输入、路径预处理、模型构造及优化；写出和独立验证另列。", "",
             f"独立 cut 审计通过且证据仍匹配：{payload['independent_cut_audit_passed']}。", "",
             "| 实例 / 方法 | 调度状态 | 可行性及身份验证 | UB | global LB | gap | 算法秒 | 进程秒 | 终止 |",
             "|---|---|---|---:|---:|---:|---:|---:|---|"]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(k)) for k in ("job", "scheduler_status", "validation_passed",
            "upper_bound", "lower_bound", "gap", "algorithm_seconds", "external_wall_seconds", "termination_reason")) + " |")
    lines += ["", "completed 表示执行及保存/验证流程完成；time_limit 与达到 1% gap、精确最优分别报告。共同最佳 UB 不替换各运行自身 gap。",
              "单 seed、固定四图的结果不构成普遍速度结论；完整目录或构造失败保留失败状态，不以限制路径结果替代完整 CPLEX。"]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n")
    return dict(report=str(output / "REPORT.md"), jobs=len(rows), independent_cut_audit_passed=payload["independent_cut_audit_passed"])


@contextmanager
def adapters():
    replacement = dict(command_for=command_for, _record=record, guarded_execute=guarded_execute, verify=verify, report=report)
    old = {name: getattr(base, name) for name in replacement}
    try:
        for name, value in replacement.items():
            setattr(base, name, value)
        yield
    finally:
        for name, value in old.items():
            setattr(base, name, value)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("inspect", "prepare", "verify", "launch", "run", "status", "report", "audit", "audit-worker"))
    parser.add_argument("--batch", type=Path, default=DEFAULT_BATCH)
    parser.add_argument("--binary", type=Path)
    parser.add_argument("--maps", type=Path, default=MAPS)
    parser.add_argument("--evidence", type=Path, default=RELEASE)
    parser.add_argument("--jobs", nargs="+")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--job-id")
    parser.add_argument("--attempt", type=Path)
    args = parser.parse_args(argv)
    with adapters():
        if args.action == "inspect":
            result = inspect(args.maps)
        elif args.action == "prepare":
            if args.binary is None:
                parser.error("prepare requires --binary")
            result = prepare(args.batch, binary=args.binary, maps=args.maps, evidence=args.evidence)
        elif args.action == "verify":
            result = dict(passed=bool(verify(args.batch)), solver_started=False)
        elif args.action == "launch":
            result = base.launch(args.batch)
        elif args.action == "run":
            result = base.run(args.batch, only=args.jobs, retry_failed=args.retry_failed)
        elif args.action == "report":
            result = report(args.batch)
        elif args.action == "audit":
            from src.experiments.baseline_cut_audit_20261029 import audit
            result = audit(args.batch, require_complete=args.require_complete)
        elif args.action == "audit-worker":
            if not args.job_id or args.attempt is None:
                parser.error("audit-worker requires --job-id and --attempt")
            result = base.audit_worker(args.batch, args.job_id, args.attempt)
        else:
            result = base.existing.status_summary(args.batch)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
