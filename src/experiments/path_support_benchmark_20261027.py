"""Controlled path-support ablation on two existing real-map instances.

The frozen-runner, resource guard, native/Python validation, operational exports,
and primary result metrics are reused from the unified benchmark. This adapter
only selects the two cases and changes the path-support flag. ``inspect`` never
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

from src.experiments import unified_benchmark_20261027 as unified
from src.experiments.storage import write_json

base = unified.base
ROOT = base.ROOT
MODULE = "src.experiments.path_support_benchmark_20261027"
REFERENCE = ROOT / "outputs/unified_benchmark_20261027"
DEFAULT_BATCH = ROOT / "outputs/path_support_benchmark_20261027"
CASES = (("tsl_L01_M01", "double", 2), ("tsl_L02_M01", "single", 1))


def reference_jobs():
    rows = []
    for identity, mode, cardinality in CASES:
        source = REFERENCE / "runs" / identity / mode / "benders_unified/attempts/001/job.json"
        item = base.read(source)
        job, command = item["job"], item["command"]
        expected = {"--threads": "1", "--seed": "43", "--gap": "0.01",
                    "--method": "benders_cg", "--demand-formulation": "preprocessed",
                    "--k": str(cardinality)}
        for key, value in expected.items():
            if command[command.index(key) + 1] != value:
                raise ValueError(f"Reference protocol changed: {identity} {key}")
        if "--path-support-cuts" in command or "--first-leg-consistency" in command:
            raise ValueError("Unexpected support or first-leg flag in reference")
        rows.append(dict(identity=identity, task_mode=mode, K=cardinality, job=job,
                         command=command, reference_job=str(source),
                         reference_job_sha256=base.sha256(source)))
    return rows


def command_for(batch, manifest, job, attempt):
    command = list(job["reference_command"])
    command[0] = str(Path(batch) / "frozen/bin/mlnd_cpp")
    overrides = {"--input": str(Path(batch) / job["instance"]), "--out": str(attempt),
                 "--seconds": str(manifest["protocol"]["seconds_per_job"])}
    for key, value in overrides.items():
        command[command.index(key) + 1] = value
    if job["variant"] == "support":
        command.append("--path-support-cuts")
    return command


def prepare(batch=DEFAULT_BATCH, *, binary=None):
    refs = reference_jobs()
    batch = Path(batch).resolve()
    base.prepare(batch, [REFERENCE / "instances" / item["identity"] for item in refs],
                 binary=binary or ROOT / "cpp/build-unified/mlnd_cpp", seconds=600,
                 threads=1, parallel=4, gap=.01, seed=43, job_rss_gib=4,
                 batch_rss_gib=12, system_available_gib=2, hard_wall_seconds=720)
    manifest = base.read(batch / "experiment_manifest.json")
    jobs = []
    for item in refs:
        identity = item["identity"]
        source = Path(item["reference_job"])
        target = batch / "provenance" / identity / "reference_job.json"
        target.parent.mkdir(parents=True)
        shutil.copy2(source, target)
        manifest["frozen_sha256"][str(target.relative_to(batch))] = base.sha256(target)
        for variant in ("baseline", "support"):
            method = "benders_" + variant
            jobs.append(dict(id=f"{identity}_{item['task_mode']}_{method}",
                instance=f"instances/{identity}", dataset="tsl", map_id="M01",
                n_l=item["job"]["n_l"], K=item["K"], task_mode=item["task_mode"],
                method=method, variant=variant,
                output=f"runs/{identity}/{item['task_mode']}/{method}",
                reference_command=item["command"],
                reference_job_sha256=item["reference_job_sha256"]))
    manifest.update(schema="path-support-ablation/1", jobs=jobs,
                    binary_sha256=base.sha256(batch / "frozen/bin/mlnd_cpp"))
    manifest["protocol"].update(physical_instances=2, total_jobs=4,
        demand_formulation="preprocessed", flow_policy="aggregate", solver="CPLEX",
        task_modes=["double", "single"], max_orders=[2, 1],
        same_binary_for_all_jobs=True, input_generation=False,
        sole_algorithm_difference="support adds --path-support-cuts",
        legacy_3600_seconds_results="reference only; not the controlled baseline",
        root_initialization="inside each job budget; no external root certificate or warm start")
    write_json(batch / "state.json", dict(status="prepared_not_started", updated_utc=base.existing.utc(),
        jobs={job["id"]: dict(status="pending", attempts=[]) for job in jobs}))
    (batch / "run.py").write_text("from pathlib import Path\nimport sys\n"
        "batch = Path(__file__).resolve().parent\nsys.path.insert(0, str(batch / 'frozen'))\n"
        f"from {MODULE} import main\nmain([*sys.argv[1:], '--batch', str(batch)])\n", encoding="utf-8")
    manifest["frozen_sha256"]["run.py"] = base.sha256(batch / "run.py")
    write_json(batch / "experiment_manifest.json", manifest)
    return dict(batch=str(batch), jobs=4, binary_sha256=manifest["binary_sha256"], solver_started=False)


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
    row["all_courier_candidates"] = sum(bool(i.get("all_courier_candidate")) for i in iterations)
    row["master_solves_count"] = len(result.get("master_solves", []))
    row["benders_cuts_count"] = len(result.get("benders_cuts", []))
    row["root_lower_bound"] = result.get("root_diagnostics", {}).get("lower_bound")
    row["continuous_lower_bound"] = result.get("continuous_root", {}).get("applied_total_cost_lower_bound")
    row["root_to_final_lb_gain"] = None if row.get("lower_bound") is None or row["root_lower_bound"] is None else row["lower_bound"] - row["root_lower_bound"]
    row["native_model_validated"] = result.get("native_model_validated")
    for key, value in result.items():
        if key.startswith("path_support"):
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


def report(batch):
    batch = Path(batch).resolve()
    manifest = base.verify(batch)
    state = base.read(batch / "state.json")
    rows = [record(batch, job, state["jobs"][job["id"]]) for job in manifest["jobs"]]
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
    lines = ["# 路径支持割的受控比较", "",
        "两个既有 TSL 实例；同一新二进制，baseline 不启用新割，support 增加 --path-support-cuts。",
        "每任务 600 秒、单线程、seed 43、gap 1%，最多四任务并行。旧 3600 秒结果仅作背景，不混入本轮基线。",
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
    lines += ["", "路径支持预处理、静态固定、动态割、跳过子问题及耗时等原生诊断逐项保存在 comparison.csv/JSON。",
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
    parser.add_argument("--batch", type=Path, default=DEFAULT_BATCH)
    parser.add_argument("--binary", type=Path)
    parser.add_argument("--jobs", nargs="+")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--job-id")
    parser.add_argument("--attempt", type=Path)
    args = parser.parse_args(argv)
    with adapters():
        if args.action == "inspect":
            result = dict(reference_jobs=reference_jobs(), seconds=600, threads=1,
                          parallel=4, gap=.01, solver_started=False)
        elif args.action == "prepare":
            result = prepare(args.batch, binary=args.binary)
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
