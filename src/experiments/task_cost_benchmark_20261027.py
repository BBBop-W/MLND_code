"""Controlled task-cost strengthening ablation on existing real-map instances.

The frozen-runner, resource guard, native/Python validation, operational exports,
and primary result metrics are reused from the unified benchmark. This adapter
selects a fixed suite and adds three strengthening stages. ``inspect`` never
freezes files or invokes a solver; ``prepare`` freezes, and ``launch``/``run``
must be invoked explicitly after implementation validity has been checked.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
import json
from pathlib import Path
import shutil
import statistics

from src.experiments import unified_benchmark_20261027 as unified
from src.experiments.storage import write_json

base = unified.base
ROOT = base.ROOT
MODULE = "src.experiments.task_cost_benchmark_20261027"
REFERENCE = ROOT / "outputs/unified_benchmark_20261027"
DEFAULT_BATCH = ROOT / "outputs/task_cost_benchmark_20261027"
EXPANDED_BATCH = ROOT / "outputs/task_cost_expanded_20261027"
CASES = (("tsl_L01_M01", "double", 2),)
SUITES = {
    "default": CASES,
    "expanded": (("tsl_L01_M02", "double", 2), ("tsl_L01_M03", "double", 2),
                 ("nanshan_L01_M01", "double", 2), ("tsl_L02_M01", "double", 2)),
}
VARIANTS = {
    "support": (),
    "task_savings": ("--task-savings-bound",),
    "task_resource": ("--task-savings-bound", "--task-resource-cuts"),
    "task_airline": ("--task-savings-bound", "--task-resource-cuts", "--task-airline-bounds"),
}


def reference_jobs(suite="default", *, cases=None):
    manifest_path = REFERENCE / "experiment_manifest.json"
    manifest = base.read(manifest_path)
    rows = []
    for identity, mode, cardinality in (SUITES[suite] if cases is None else cases):
        matches = [job for job in manifest["jobs"] if job["instance"] == f"instances/{identity}"
                   and job["task_mode"] == mode and job["K"] == cardinality
                   and job["method"] == "benders_unified"]
        if len(matches) != 1:
            raise ValueError(f"Reference manifest does not uniquely identify {identity}/{mode}")
        job = matches[0]
        source = REFERENCE / "runs" / identity / mode / "benders_unified/attempts/001/job.json"
        # This is a command recipe, not a claim that a historical job ran.
        # Some existing Nanshan instances were prepared but never dispatched.
        command = unified.command_for(REFERENCE, manifest, job, source.parent)
        existing_job = source.is_file()
        if existing_job:
            item = base.read(source)
            if item["command"] != command or any(item["job"].get(k) != job[k]
                    for k in ("id", "instance", "dataset", "map_id", "n_l", "K", "task_mode", "method")):
                raise ValueError(f"Saved reference job differs from its manifest recipe: {identity}")
        expected = {"--threads": "1", "--seed": "43", "--gap": "0.01",
                    "--method": "benders_cg", "--demand-formulation": "preprocessed",
                    "--k": str(cardinality)}
        for key, value in expected.items():
            if command[command.index(key) + 1] != value:
                raise ValueError(f"Reference protocol changed: {identity} {key}")
        if "--path-support-cuts" in command or "--first-leg-consistency" in command:
            raise ValueError("Unexpected support or first-leg flag in reference")
        rows.append(dict(identity=identity, task_mode=mode, K=cardinality, job=job,
                         command=command, reference_source="saved_job" if existing_job else "manifest_recipe",
                         reference_manifest=str(manifest_path), reference_manifest_sha256=base.sha256(manifest_path),
                         reference_job=str(source) if existing_job else None,
                         reference_job_sha256=base.sha256(source) if existing_job else None))
    return rows


def command_for(batch, manifest, job, attempt):
    command = list(job["reference_command"])
    command[0] = str(Path(batch) / "frozen/bin/mlnd_cpp")
    overrides = {"--input": str(Path(batch) / job["instance"]), "--out": str(attempt),
                 "--seconds": str(manifest["protocol"]["seconds_per_job"])}
    for key, value in overrides.items():
        command[command.index(key) + 1] = value
    variants = manifest["protocol"].get("algorithm_variants", VARIANTS)
    command.extend(("--path-support-cuts", *variants[job["variant"]]))
    return command


def prepare(batch=None, *, binary=None, suite="default", cases=None, variants=None):
    refs = reference_jobs(suite, cases=cases)
    variants = VARIANTS if variants is None else variants
    if not variants:
        raise ValueError("At least one declared algorithm variant is required")
    batch = Path(batch or (EXPANDED_BATCH if suite == "expanded" else DEFAULT_BATCH)).resolve()
    binary = Path(binary or ROOT / "cpp/build-unified/mlnd_cpp").resolve()
    pilot_binary = None
    if suite == "expanded":
        pilot_binary = base.read(DEFAULT_BATCH / "experiment_manifest.json")["binary_sha256"]
        if base.sha256(binary) != pilot_binary:
            raise ValueError("Expanded suite must use the same binary as the completed L1 pilot")
    base.prepare(batch, [REFERENCE / "instances" / item["identity"] for item in refs],
                 binary=binary, seconds=600,
                 threads=1, parallel=4, gap=.01, seed=43, job_rss_gib=4,
                 batch_rss_gib=12, system_available_gib=2, hard_wall_seconds=720)
    manifest = base.read(batch / "experiment_manifest.json")
    provenance_manifest = batch / "provenance/reference_experiment_manifest.json"
    provenance_manifest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(REFERENCE / "experiment_manifest.json", provenance_manifest)
    manifest["frozen_sha256"][str(provenance_manifest.relative_to(batch))] = base.sha256(provenance_manifest)
    jobs = []
    for item in refs:
        identity = item["identity"]
        if item["reference_job"]:
            source = Path(item["reference_job"])
            target = batch / "provenance" / identity / "reference_job.json"
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            manifest["frozen_sha256"][str(target.relative_to(batch))] = base.sha256(target)
        for variant in variants:
            method = "benders_" + variant
            jobs.append(dict(id=f"{identity}_{item['task_mode']}_{method}",
                instance=f"instances/{identity}", dataset=item["job"]["dataset"], map_id=item["job"]["map_id"],
                n_l=item["job"]["n_l"], K=item["K"], task_mode=item["task_mode"],
                method=method, variant=variant,
                output=f"runs/{identity}/{item['task_mode']}/{method}",
                reference_command=item["command"],
                reference_source=item["reference_source"],
                reference_manifest_sha256=item["reference_manifest_sha256"],
                reference_job_sha256=item["reference_job_sha256"]))
    manifest.update(schema="task-cost-ablation/1", jobs=jobs,
                    binary_sha256=base.sha256(batch / "frozen/bin/mlnd_cpp"))
    manifest["protocol"].update(suite=suite, physical_instances=len(refs), total_jobs=len(jobs),
        demand_formulation="preprocessed", flow_policy="aggregate", solver="CPLEX",
        task_modes=["double"], max_orders=[2],
        same_binary_for_all_jobs=True, input_generation=False,
        pilot_binary_sha256=pilot_binary,
        algorithm_variants={name: list(flags) for name, flags in variants.items()},
        common_algorithm_flag="--path-support-cuts",
        legacy_3600_seconds_results="reference only; not the controlled baseline",
        root_initialization="inside each job budget; no external root certificate or warm start")
    write_json(batch / "state.json", dict(status="prepared_not_started", updated_utc=base.existing.utc(),
        jobs={job["id"]: dict(status="pending", attempts=[]) for job in jobs}))
    (batch / "run.py").write_text("from pathlib import Path\nimport sys\n"
        "batch = Path(__file__).resolve().parent\nsys.path.insert(0, str(batch / 'frozen'))\n"
        f"from {MODULE} import main\nmain([*sys.argv[1:], '--batch', str(batch), '--suite', {suite!r}])\n", encoding="utf-8")
    manifest["frozen_sha256"]["run.py"] = base.sha256(batch / "run.py")
    write_json(batch / "experiment_manifest.json", manifest)
    return dict(batch=str(batch), jobs=len(jobs), suite=suite,
                binary_sha256=manifest["binary_sha256"], solver_started=False)


def guarded_execute(batch, manifest, job_id, command, attempt, cancel, **kwargs):
    command = list(command)
    if "-m" in command:
        at = command.index("-m") + 1
        if command[at] == base.__name__:
            command[at] = MODULE
    return unified._BASE_GUARD(batch, manifest, job_id, command, attempt, cancel, **kwargs)


def record(batch, job, state):
    row = unified.record(batch, job, state)
    row["variant"] = job["variant"]
    directory = Path(batch) / state.get("result_directory", state.get("directory", "_missing"))
    result = base.optional_read(directory / "result.json")
    if not result:
        return row
    iterations = result.get("iterations", [])
    row["evaluated_integer_candidates"] = len(iterations)
    row["pricing_complete_candidates"] = sum(bool(i.get("pricing_complete")) for i in iterations)
    row["master_optimal_solves"] = sum(bool(i.get("optimal")) for i in result.get("master_solves", []))
    for source, label in (("master_theta", "candidate_theta"),
                          ("operating_primal_upper", "candidate_operating_cost"),
                          ("ordinary_cut_violation", "ordinary_cut_violation")):
        values = [i[source] for i in iterations if isinstance(i.get(source), (int, float))]
        for name, operation in (("min", min), ("median", statistics.median), ("max", max)):
            row[label + "_" + name] = operation(values) if values else None
    gaps = [i["operating_primal_upper"] - i["operating_certified_lower"] for i in iterations
            if isinstance(i.get("operating_primal_upper"), (int, float))
            and isinstance(i.get("operating_certified_lower"), (int, float))]
    row["candidate_certificate_gap_max"] = max(gaps) if gaps else None
    ref = base.optional_read(directory / "input_reference.json")
    model = ref.get("model", {}) if ref else {}
    values, ids = model.get("values", {}), model.get("ids", {})
    if all(k in values for k in ("n_l", "n_h", "n_od")) and ids.get("levels"):
        l, h, orders, levels = values["n_l"], values["n_h"], values["n_od"], len(ids["levels"])
        assignment_start = l + h + h * levels
        airline_start = assignment_start + orders * l
        keys = [i["design_key"] for i in iterations if isinstance(i.get("design_key"), str)]
        row["candidate_unique_assignment_patterns"] = len({k[assignment_start:airline_start] for k in keys})
        row["candidate_unique_airline_patterns"] = len({k[airline_start:] for k in keys})
        row["candidate_unique_capacity_patterns"] = len({k[l + h:assignment_start] for k in keys})
    for name, value in result.get("acceleration_statistics", {}).items():
        if isinstance(value, (int, float, bool)):
            row["acceleration_" + name] = value
    row["all_courier_candidates"] = sum(bool(i.get("all_courier_candidate")) for i in iterations)
    row["master_solves_count"] = len(result.get("master_solves", []))
    row["benders_cuts_count"] = len(result.get("benders_cuts", []))
    row["root_lower_bound"] = result.get("root_diagnostics", {}).get("lower_bound")
    row["continuous_lower_bound"] = result.get("continuous_root", {}).get("applied_total_cost_lower_bound")
    row["root_to_final_lb_gain"] = None if row.get("lower_bound") is None or row["root_lower_bound"] is None else row["lower_bound"] - row["root_lower_bound"]
    row["native_model_validated"] = result.get("native_model_validated")
    for key, value in result.items():
        if key.startswith(("path_support", "task_savings", "task_resource", "task_airline",
                           "airline_resource", "airline_connectivity")):
            if isinstance(value, list):
                # Detailed cut/attempt evidence stays in result.json; avoid
                # duplicating a potentially large log inside every CSV row.
                row[key + "_count"] = len(value)
                continue
            row[key + "_json"] = json.dumps(value, sort_keys=True)
            if isinstance(value, dict):
                for name, scalar in value.items():
                    if not isinstance(scalar, (dict, list)):
                        row[key + "_" + name] = scalar
    for key, value in result.get("timings", {}).items():
        if not isinstance(value, (dict, list)):
            row["timing_" + key] = value
    return row


def report(batch, *, record_fn=None):
    batch = Path(batch).resolve()
    manifest = base.verify(batch)
    state = base.read(batch / "state.json")
    make_record = record if record_fn is None else record_fn
    rows = [make_record(batch, job, state["jobs"][job["id"]]) for job in manifest["jobs"]]
    catalogues = {}
    for row in rows:
        if row.get("all_paths") is None:
            continue
        identity = row["instance"], row["K"]
        catalogue = row["all_service_paths"], row["all_empty_paths"]
        if identity in catalogues and catalogues[identity] != catalogue:
            raise ValueError(f"Complete path catalogues differ between variants: {identity}")
        catalogues[identity] = catalogue
    output = batch / "analysis"
    output.mkdir(exist_ok=True)
    write_json(output / "comparison.json", dict(protocol=manifest["protocol"], rows=rows))
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (output / "comparison.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    fmt = lambda v: "—" if v is None else f"{v:.6f}" if isinstance(v, float) else str(v)
    identities = list(dict.fromkeys(Path(job["instance"]).name for job in manifest["jobs"]))
    lines = ["# 完整任务成本与容量强化的受控比较", "",
        "既有实例：" + "、".join(identities) + "。同一二进制，四组均保留路径支持割。依次加入完整任务收益界、任务容量资源割和任务航线流量界。",
        "每任务 600 秒、单线程、seed 43、gap 1%，最多四任务并行。support 为本轮同时重跑的受控基线。",
        "复用原生及 Python 校验、运营导出、资源监护和基础指标。UB/LB 只展示通过结果及输入身份校验的数据。", "",
        "| 实例 | 模式 | 方法 | UB | LB | gap | Root LB | Root 后提升 | 秒 | 状态 | 验证 |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---|---|"]
    for row in rows:
        values = [Path(row["instance"]).name, row["task_mode"], row["variant"],
            row.get("upper_bound"), row.get("lower_bound"),
            None if row.get("gap") is None else f"{100*row['gap']:.3f}%",
            row.get("root_lower_bound"), row.get("root_to_final_lb_gain"), row.get("solve_seconds"),
            row["scheduler_status"], row["validation_passed"]]
        lines.append("| " + " | ".join(fmt(v) for v in values) + " |")
    lines += ["", "| 实例 | 方法 | 主问题变量/行 | 子问题最大变量/行/锥 | 完整路径 | 生成服务列 | 主问题次数 | 完成评估/全骑手 | Benders 割 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        values = [Path(row["instance"]).name, row["variant"],
            f"{fmt(row.get('master_variables'))}/{fmt(row.get('master_linear_rows'))}",
            "/".join(fmt(row.get("subproblem_peak_" + key)) for key in ("variables", "linear_rows", "cones")),
            row.get("all_paths"), row.get("generated_service_columns"), row.get("master_solves_count"),
            f"{fmt(row.get('evaluated_integer_candidates'))}/{fmt(row.get('all_courier_candidates'))}",
            row.get("benders_cuts_count")]
        lines.append("| " + " | ".join(fmt(v) for v in values) + " |")
    lines += ["", "| 实例 | 方法 | theta 最小/中位/最大 | 实际运营成本中位数 | 分配模式/航线模式 | 完成定价/评估 | 子问题证书最大差 |",
        "|---|---|---:|---:|---:|---:|---:|"]
    for row in rows:
        values = [Path(row["instance"]).name, row["variant"],
            "/".join(fmt(row.get("candidate_theta_" + k)) for k in ("min", "median", "max")),
            row.get("candidate_operating_cost_median"),
            f"{fmt(row.get('candidate_unique_assignment_patterns'))}/{fmt(row.get('candidate_unique_airline_patterns'))}",
            f"{fmt(row.get('pricing_complete_candidates'))}/{fmt(row.get('evaluated_integer_candidates'))}",
            row.get("candidate_certificate_gap_max")]
        lines.append("| " + " | ".join(fmt(v) for v in values) + " |")
    lines += ["", "任务收益预处理、任务资源割、航线流量界及路径支持诊断逐项保存在 comparison.csv/JSON；外部资源 LP 的规模与主问题规模分别保存。",
        "完整路径目录和实际进入列生成模型的列数分别报告；支持约束仅保证必要路径，仍由原 SOCP 验证容量和闭网络运营。",
        "本轮支配约束保持整数最优值，但强化后的根松弛不能与原连续松弛混称。",
        "原始结果、候选设计、割证据、资源采样、验证及 operations 导出均保存在 runs 的 attempt 目录。"]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return dict(report=str(output / "REPORT.md"), jobs=len(rows), validated=sum(r["validation_passed"] for r in rows))


@contextmanager
def adapters():
    replacements = dict(command_for=command_for, _record=record,
                        guarded_execute=guarded_execute, report=report)
    previous = {name: getattr(base, name) for name in replacements}
    try:
        for name, value in replacements.items():
            setattr(base, name, value)
        yield
    finally:
        for name, value in previous.items():
            setattr(base, name, value)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("inspect", "prepare", "verify", "launch", "run", "stop", "status", "report", "audit-worker"))
    parser.add_argument("--batch", type=Path)
    parser.add_argument("--suite", choices=tuple(SUITES), default="default")
    parser.add_argument("--binary", type=Path)
    parser.add_argument("--jobs", nargs="+")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--job-id")
    parser.add_argument("--attempt", type=Path)
    args = parser.parse_args(argv)
    args.batch = args.batch or (EXPANDED_BATCH if args.suite == "expanded" else DEFAULT_BATCH)
    with adapters():
        if args.action == "inspect":
            result = dict(reference_jobs=reference_jobs(args.suite), suite=args.suite,
                          batch=str(args.batch), seconds=600, threads=1,
                          parallel=4, gap=.01, jobs=len(SUITES[args.suite])*len(VARIANTS), solver_started=False)
        elif args.action == "prepare":
            result = prepare(args.batch, binary=args.binary, suite=args.suite)
        elif args.action == "verify":
            result = dict(passed=bool(base.verify(args.batch)), solver_started=False)
        elif args.action == "launch":
            result = base.launch(args.batch)
        elif args.action == "run":
            result = base.run(args.batch, only=args.jobs, retry_failed=args.retry_failed)
        elif args.action == "stop":
            result = unified.stop(args.batch)
        elif args.action == "status":
            result = base.existing.status_summary(args.batch)
        elif args.action == "report":
            result = report(args.batch)
        else:
            if not args.job_id or not args.attempt:
                parser.error("audit-worker requires --job-id and --attempt")
            result = base.audit_worker(args.batch, args.job_id, args.attempt)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
