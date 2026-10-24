"""Frozen K=2 topology/bound ablation on existing L01/L02 standard instances.

One binary serves all variants. The conditional benchmark's guarded scheduler,
validation, recovery and lightweight live status are reused without source edits.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
from pathlib import Path

from src.data.standard import load_standard
from src.experiments import conditional_benchmark_20261022 as base
from src.experiments.storage import write_json


ROOT = base.ROOT
MODULE = "src.experiments.topology_benchmark_20261022"
DEFAULT_BATCH = ROOT / "outputs/topology_benchmark_20261022"
VARIANTS = (
    ("lifted_baseline", "benders_lifted", False),
    ("conditional_baseline", "benders_conditional", False),
    ("lifted_strengthened", "benders_lifted", True),
    ("conditional_strengthened", "benders_conditional", True),
    ("full_cplex_reference", "full_cplex", False),
)
_BASE_COMMAND = base.command_for
_BASE_RECORD = base._record
_BASE_GUARD = base.guarded_execute
_BASE_REPORT = base.report


def jobs_for(origins, *, full_reference=True):
    variants = VARIANTS if full_reference else VARIANTS[:-1]
    return [dict(id=f"{origin['id']}_K2_{variant}", variant=variant,
                 instance=origin["instance"], K=2, method=method,
                 tight_airline_bounds=strong, topology_cuts=strong,
                 output=f"runs/{origin['id']}/{variant}/K2")
            for origin in origins for variant, method, strong in variants]


def command_for(batch, manifest, job, attempt):
    command = _BASE_COMMAND(batch, manifest, job, attempt)
    if job.get("tight_airline_bounds"):
        command.append("--tight-airline-bounds")
    if job.get("topology_cuts"):
        cfg = manifest["protocol"]
        command += ["--topology-cuts", "--topology-seconds", str(cfg.get("topology_seconds", 2)),
                    "--topology-cut-limit", str(cfg.get("topology_cut_limit", 1000))]
    return command


def prepare(batch, instances, *, full_reference=True, **kwargs):
    settings = dict(seconds=600, threads=1, parallel=4, gap=.005, seed=43,
                    job_rss_gib=4, batch_rss_gib=8, system_available_gib=2,
                    hard_wall_seconds=720, resource_sample_seconds=.5,
                    terminate_grace_seconds=5)
    settings.update(kwargs)
    base.prepare(batch, instances, **settings)
    batch = Path(batch).resolve()
    manifest = base.read(batch / "experiment_manifest.json")
    manifest["schema"] = "topology-benchmark/1"
    manifest["jobs"] = jobs_for(manifest["input_origins"], full_reference=full_reference)
    manifest["protocol"].update(
        comparison="lifted/conditional x baseline/tight_airline_bounds+topology_cuts",
        same_binary_for_all_variants=True, baseline_new_flags=False,
        full_reference=full_reference, input_generation=False,
        topology_seconds=2, topology_cut_limit=1000)
    for origin in manifest["input_origins"]:
        data = load_standard(batch / origin["instance"], case="K2")
        origin["all_courier_cost"] = sum(float(a)*float(b) for a, b in zip(data["demand"], data["ground_cost"]))
        origin["size"] = dict(bases=data["n_l"], lockers=data["n_h"], orders=data["n_od"])
    write_json(batch / "experiment_manifest.json", manifest)
    write_json(batch / "state.json", dict(status="prepared_not_started", updated_utc=base.existing.utc(),
        jobs={job["id"]: dict(status="pending", attempts=[]) for job in manifest["jobs"]}))
    (batch / "run.py").write_text(
        "from pathlib import Path\nimport sys\n"
        "batch = Path(__file__).resolve().parent\n"
        "sys.path.insert(0, str(batch / 'frozen'))\n"
        f"from {MODULE} import main\n"
        "main([*sys.argv[1:], '--batch', str(batch)])\n", encoding="utf-8")
    return dict(batch=str(batch), jobs=len(manifest["jobs"]), solver_started=False,
                input_generation=False, same_binary=True)


def _guarded_execute(batch, manifest, job_id, command, attempt, cancel, **kwargs):
    # The inherited dispatcher starts audit/report in fresh monitored processes.
    # Route those two internal commands through this module's record adapter.
    routed = list(command)
    if "-m" in routed:
        index = routed.index("-m") + 1
        if routed[index] == "src.experiments.conditional_benchmark_20261022":
            routed[index] = MODULE
    return _BASE_GUARD(batch, manifest, job_id, routed, attempt, cancel, **kwargs)


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _record(batch, job, state):
    row = _BASE_RECORD(batch, job, state)
    row.update(variant=job["variant"], tight_airline_bounds_requested=job["tight_airline_bounds"],
               topology_cuts_requested=job["topology_cuts"])
    source = Path(batch) / state.get("result_directory", state.get("directory", "_missing"))
    result = base.optional_read(source / "result.json")
    statistics = result.get("acceleration_statistics") or {}
    accelerations = result.get("accelerations") or {}
    topology = result.get("topology_summary") or {}
    root = result.get("continuous_root") or {}
    root_diag = result.get("root_diagnostics") or {}
    master = result.get("master_solves") or []
    bounds = [entry["certified_bound"] for entry in master
              if isinstance(entry, dict) and _number(entry.get("certified_bound"))]
    row.update(continuous_root_bound_valid=root.get("bound_valid"),
        continuous_root_lower_bound=root.get("certified_lower_bound") if root.get("bound_valid") else None,
        continuous_root_applied_lower_bound=root.get("applied_total_cost_lower_bound"),
        continuous_root_termination=root.get("termination_reason"),
        continuous_root_seconds=root.get("algorithm_wall"),
        root_lower_bound=root_diag.get("lower_bound"),
        root_termination=root_diag.get("termination_reason"),
        initial_operating_lower_bound=(result.get("master_bound_metadata") or {}).get("unconstrained_operating_floor"),
        integer_master_first_lower_bound=bounds[0] if bounds else None,
        integer_master_best_lower_bound=max(bounds) if bounds else None,
        integer_master_last_reported_lower_bound=bounds[-1] if bounds else None,
        integer_master_calls=statistics.get("integer_master_solves"),
        integer_socp_calls=statistics.get("integer_cg_calls"),
        all_courier_candidates=statistics.get("all_courier_candidates"),
        tight_airline_bounds_enabled=result.get("tight_airline_bounds", accelerations.get("tight_airline_bounds")),
        topology_cuts_enabled=topology.get("enabled", accelerations.get("topology_cuts")),
        topology_attempts=topology.get("attempts"),
        topology_cuts=topology.get("cuts"),
        topology_skipped_integer_subproblems=topology.get("skipped_integer_subproblems"),
        topology_zero_service_certificates=topology.get("zero_service_certificates"),
        topology_seconds=topology.get("seconds"))
    calls = row["integer_socp_calls"]
    count = row["all_courier_candidates"]
    row["all_courier_candidate_ratio"] = count/calls if _number(count) and _number(calls) and calls > 0 else None
    # Retain diagnostic names verbatim; no inference of zero from absent fields.
    for key, value in statistics.items():
        if key.startswith("topology") or key.startswith("tight_airline"):
            row[key] = value if not isinstance(value, (list, dict)) else json.dumps(value, sort_keys=True)
    for key in ("topology_metadata", "tight_airline_bound_metadata", "topology_statistics", "topology_summary"):
        if key in result:
            row[key] = json.dumps(result[key], sort_keys=True)
    for key in ("topology_cut_records", "topology_attempts"):
        if isinstance(result.get(key), list):
            row[key+"_record_count"] = len(result[key])
    manifest = base.read(Path(batch) / "experiment_manifest.json")
    origin = next((entry for entry in manifest.get("input_origins", [])
                   if entry["instance"] == job["instance"]), {})
    baseline = origin.get("all_courier_cost")
    row["all_courier_cost"] = baseline
    row["upper_bound_ratio_to_all_courier"] = row["upper_bound"]/baseline if _number(row["upper_bound"]) and baseline else None
    row["lower_bound_ratio_to_all_courier"] = row["lower_bound"]/baseline if _number(row["lower_bound"]) and baseline else None
    return row


@contextmanager
def adapters():
    """Scoped hooks for the unchanged scheduler; child processes install their own."""
    replacement = dict(command_for=command_for, _record=_record,
                       guarded_execute=_guarded_execute, report=_report)
    previous = {key: getattr(base, key) for key in replacement}
    try:
        for key, value in replacement.items():
            setattr(base, key, value)
        yield
    finally:
        for key, value in previous.items():
            setattr(base, key, value)


def _fmt(value, *, percent=False):
    if value is None:
        return "—"
    if percent:
        return f"{100*value:.3f}%"
    return f"{value:,.3f}" if isinstance(value, float) else str(value)


def _table(headers, records):
    return ["| " + " | ".join(headers) + " |", "|" + "---|"*len(headers),
            *["| " + " | ".join(_fmt(value) for value in row) + " |" for row in records]]


def _report(batch):
    outcome = _BASE_REPORT(batch)
    batch = Path(batch)
    data = base.read(batch / "analysis/comparison.json")
    manifest = base.read(batch / "experiment_manifest.json")
    rows, cfg = data["rows"], data["protocol"]
    complete = all(row["scheduler_status"] not in ("pending", "running") for row in rows)
    title = "Topology + airline bound 比较" + ("" if complete else "（尚未完成）")
    lines = [f"# {title}", "",
        "比较同一冻结二进制的四种 Benders 设置：原 lifted / 新 conditional × 原默认 / 同时开启两项强化。"
        "full_cplex_reference 使用原默认设置，提供完整路径参考。", "",
        f"每组求解预算 {cfg['seconds_per_job']:g} 秒，{cfg['parallel_jobs']} 组并行、每组 {cfg['threads_per_job']} 线程，"
        f"目标 gap {100*cfg['relative_gap_target']:.2f}%。预算包括输入、路径准备、模型构造和优化；"
        "另列进程墙钟及验证时间。共享计算资源会影响限时结果，不据单次结果宣称普遍提速。", "",
        "UB 和全局 LB 仅使用独立验证及哈希通过的结果。optimal/OptimalTol 表示达到配置容差，仍以实际 gap 为准。"
        "内存或墙钟停止不是数学不可行。原始记录与资源保护状态完整保留。", ""]
    lines += _table(["实例", "变体", "UB", "全局 LB", "gap", "预算耗时 s", "进程墙钟 s", "终止", "验证"],
        [(Path(row["instance"]).name, row["variant"], row["upper_bound"], row["lower_bound"],
          _fmt(row["gap"], percent=True), row.get("solve_seconds"), row.get("external_wall_seconds"),
          row.get("termination_reason"), row["validation_passed"]) for row in rows])
    lines += ["", "## 下界与子问题调用", "",
        "初始运营界不包含固定设施费用；连续根界只有 bound_valid=true 才展示。整数主问题界来自 master_solves 的 certified_bound，"
        "该列不替代最终全局 LB。全骑手候选比例的分母是实际 integer_cg_calls；被拓扑割跳过的候选不计入这一分母。", ""]
    lines += _table(["任务", "初始运营界", "连续根界", "根阶段 LB", "整数主问题最好 LB", "整数 SOCP 次数", "全骑手候选比例", "UB/全骑手"],
        [(row["job"], row.get("initial_operating_lower_bound"), row.get("continuous_root_lower_bound"),
          row.get("root_lower_bound"), row.get("integer_master_best_lower_bound"), row.get("integer_socp_calls"),
          _fmt(row.get("all_courier_candidate_ratio"), percent=True),
          _fmt(row.get("upper_bound_ratio_to_all_courier"), percent=True)) for row in rows])
    diagnostics = sorted({key for row in rows for key in row
                          if key.startswith("topology_") and not key.endswith(("metadata", "statistics", "summary", "requested", "enabled"))})
    lines += ["", "## 拓扑强化诊断", "", "缺失诊断显示 —，不视作零。详细原始字段保留在 comparison.csv/JSON。", ""]
    lines += _table(["任务", "边界强化", "拓扑割", *diagnostics],
        [(row["job"], row.get("tight_airline_bounds_enabled"), row.get("topology_cuts_enabled"),
          *(row.get(key) for key in diagnostics)) for row in rows])
    def size(row, name):
        return "/".join(_fmt(row.get(name+"_"+key)) for key in ("variables", "linear_rows", "cones"))
    lines += ["", "## 模型与路径规模", "",
        "规模依次为变量/线性约束/锥。整数子问题峰值包含启发式整数设计、排除连续根；峰值也受列池变化影响。"
        "完整路径总数只取枚举完整的参考结果；生成列数单列。", ""]
    lines += _table(["任务", "完整模型", "主问题", "整数子问题峰值", "完整服务/空飞路径", "已生成服务/空飞列"],
        [(row["job"], size(row,"full_model"), size(row,"master"), size(row,"subproblem_peak"),
          f"{_fmt(row.get('all_service_paths'))}/{_fmt(row.get('all_empty_paths'))}",
          f"{_fmt(row.get('generated_service_columns'))}/{_fmt(row.get('generated_empty_columns'))}") for row in rows])
    lines += ["", "## 阶段耗时", ""]
    timing_fields = ("enumeration_seconds", "support_preprocessing_seconds", "build_seconds", "optimize_seconds",
                     "continuous_root_seconds", "master_optimize_seconds", "subproblem_wall_seconds", "validation_seconds")
    lines += _table(["任务", *timing_fields], [(row["job"], *(row.get(key) for key in timing_fields)) for row in rows])
    lines += ["", "## 资源保护", "",
        "单组 RSS 上限 4 GiB、同时活动组总 RSS 上限 8 GiB、系统可用内存保底 2 GiB；"
        f"每组含验证的外部墙钟上限 {cfg.get('hard_wall_seconds',720):g} 秒。"
        "RSS 为专用进程组采样值，采样周期 0.5 秒。", ""]
    lines += _table(["任务", "资源原因", "阶段", "求解峰值 GiB", "验证停止", "验证峰值 GiB"],
        [(row["job"], row.get("resource_limit_kind"), row.get("stopped_stage"),
          None if row.get("peak_sampled_tree_rss_bytes") is None else row["peak_sampled_tree_rss_bytes"]/1024**3,
          row.get("validation_stop_reason"), None if row.get("validation_peak_sampled_tree_rss_bytes") is None
          else row["validation_peak_sampled_tree_rss_bytes"]/1024**3) for row in rows])
    lines += ["", "## 输入与证据", "",
        f"完成终止 {sum(row['scheduler_status'] not in ('pending','running') for row in rows)}/{len(rows)}，"
        f"验证通过 {sum(bool(row['validation_passed']) for row in rows)}/{len(rows)}。", ""]
    lines += _table(["实例", "原输入 manifest SHA256", "全骑手成本"],
        [(origin["id"], origin["manifest_sha256"], origin["all_courier_cost"]) for origin in manifest["input_origins"]])
    lines += ["", "实验来源与源码指纹：`../experiment_manifest.json`；原始日志与验证：`../runs/`；"
        "实时状态和资源原因：`../live_status.md`。所有标准输入来自既有实例的逐文件复制，没有生成新输入。"]
    (batch / "analysis/REPORT.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    return dict(outcome, complete=complete)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "verify", "run", "launch", "report", "status", "audit-worker"))
    parser.add_argument("--batch", type=Path, default=DEFAULT_BATCH)
    parser.add_argument("--levels", type=int, choices=(1, 2), nargs="+", default=[1, 2])
    parser.add_argument("--binary", type=Path)
    parser.add_argument("--seconds", type=float, default=600)
    parser.add_argument("--parallel", type=int, default=4)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--gap", type=float, default=.005)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--no-full-reference", action="store_true")
    parser.add_argument("--hard-wall-seconds", type=float, default=720)
    parser.add_argument("--jobs", nargs="+")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--job-id")
    parser.add_argument("--attempt", type=Path)
    args = parser.parse_args(argv)
    with adapters():
        if args.action == "prepare":
            if len(set(args.levels)) != len(args.levels):
                parser.error("Levels must not repeat")
            result = prepare(args.batch, [ROOT / f"outputs/tsl_scale_ladder/instances/L{n:02d}" for n in args.levels],
                full_reference=not args.no_full_reference, binary=args.binary, seconds=args.seconds,
                parallel=args.parallel, threads=args.threads, gap=args.gap, seed=args.seed,
                hard_wall_seconds=args.hard_wall_seconds)
        elif args.action == "verify":
            result = dict(passed=bool(base.verify(args.batch)), solver_started=False)
        elif args.action == "run":
            result = base.run(args.batch, only=args.jobs, retry_failed=args.retry_failed)
        elif args.action == "launch":
            result = base.launch(args.batch)
        elif args.action == "report":
            result = _report(args.batch)
        elif args.action == "audit-worker":
            if not args.job_id or not args.attempt:
                parser.error("audit-worker requires --job-id and --attempt")
            result = base.audit_worker(args.batch, args.job_id, args.attempt)
        else:
            result = base.existing.status_summary(args.batch)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
