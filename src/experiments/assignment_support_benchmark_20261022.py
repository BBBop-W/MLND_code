"""Four guarded K=2 runs adding assignment-support cuts to the previous setting.

The previous topology benchmark remains read-only. Its independently checked
results and progress traces are copied into the new batch before solving.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
import json
from pathlib import Path
import shutil

from src.experiments import conditional_benchmark_20261022 as base
from src.experiments import topology_benchmark_20261022 as topology
from src.experiments.storage import write_json


ROOT = base.ROOT
MODULE = "src.experiments.assignment_support_benchmark_20261022"
DEFAULT_BATCH = ROOT / "outputs/assignment_support_benchmark_20261022"
DEFAULT_REFERENCE = ROOT / "outputs/topology_benchmark_20261022"
VARIANTS = (("lifted_assignment_support", "benders_lifted"),
            ("conditional_assignment_support", "benders_conditional"))
REFERENCE_VARIANTS = {"lifted_baseline", "conditional_baseline", "lifted_strengthened",
                      "conditional_strengthened", "full_cplex_reference"}
_BASE_COMMAND = base.command_for
_BASE_GUARD = base.guarded_execute
_BASE_REPORT = base.report
_TOPOLOGY_RECORD = topology._record


def jobs_for(origins):
    return [dict(id=f"{origin['id']}_K2_{variant}", variant=variant,
                 instance=origin["instance"], K=2, method=method,
                 tight_airline_bounds=True, topology_cuts=True, assignment_support_cuts=True,
                 output=f"runs/{origin['id']}/{variant}/K2")
            for origin in origins for variant, method in VARIANTS]


def command_for(batch, manifest, job, attempt):
    command = _BASE_COMMAND(batch, manifest, job, attempt)
    cfg = manifest["protocol"]
    command += ["--tight-airline-bounds", "--topology-cuts", "--topology-seconds",
                str(cfg.get("topology_seconds", 2)), "--topology-cut-limit",
                str(cfg.get("topology_cut_limit", 1000)), "--assignment-support-cuts"]
    for key, flag in (("assignment_support_seconds", "--assignment-support-seconds"),
                      ("assignment_support_cut_limit", "--assignment-support-cut-limit"),
                      ("assignment_support_max_cuts", "--assignment-support-max-cuts"),
                      ("assignment_support_floor", "--assignment-support-floor")):
        if key in cfg:
            command += [flag, str(cfg[key])]
    return command


def _reference_snapshot(reference):
    """Refresh validation hashes without changing any old report or result."""
    reference = Path(reference).resolve()
    manifest = base.verify(reference)
    state = base.read(reference / "state.json")
    rows, traces = [], []
    for job in manifest["jobs"]:
        if job.get("variant") not in REFERENCE_VARIANTS:
            continue
        saved = state["jobs"][job["id"]]
        if saved["status"] in ("pending", "running"):
            raise ValueError(f"Reference job is unfinished: {job['id']}")
        row = _TOPOLOGY_RECORD(reference, job, saved)
        if not row["validation_passed"]:
            raise ValueError(f"Reference result failed current hash/validation checks: {job['id']}")
        row["reference_batch"] = str(reference)
        rows.append(row)
        progress = reference / saved["directory"] / "progress.jsonl"
        if not job["method"].startswith("full_") and progress.is_file():
            traces.append((job["id"], progress))
    expected = {f"{origin['id']}_K2_{variant}" for origin in manifest["input_origins"]
                for variant in REFERENCE_VARIANTS}
    if {row["job"] for row in rows} != expected:
        raise ValueError("Reference batch does not contain all expected completed settings")
    counts = {row["instance"]: row for row in rows if row.get("all_paths") is not None}
    for row in rows:
        if row.get("all_paths") is None and row["instance"] in counts:
            for key in ("all_paths", "all_service_paths", "all_empty_paths", "initial_segments", "full_path_count_source"):
                row[key] = counts[row["instance"]].get(key)
    return manifest, rows, traces


def prepare(batch, instances, *, reference=DEFAULT_REFERENCE, assignment_support_seconds=None,
            assignment_support_cut_limit=None, assignment_support_max_cuts=None,
            assignment_support_floor=None, **kwargs):
    reference = Path(reference).resolve()
    previous, rows, traces = _reference_snapshot(reference)
    settings = dict(seconds=600, threads=1, parallel=4, gap=.005, seed=43,
                    job_rss_gib=4, batch_rss_gib=8, system_available_gib=2,
                    hard_wall_seconds=720, resource_sample_seconds=.5, terminate_grace_seconds=5)
    settings.update(kwargs)
    for current, saved in (("seconds", "seconds_per_job"), ("threads", "threads_per_job"),
                           ("gap", "relative_gap_target"), ("seed", "seed")):
        if settings[current] != previous["protocol"][saved]:
            raise ValueError(f"Reference comparison requires matching {current}")
    old_inputs = {origin["id"]: origin["manifest_sha256"] for origin in previous["input_origins"]}
    for value in instances:
        path = Path(value).resolve()
        if old_inputs.get(path.name) != base.sha256(path / "manifest.json"):
            raise ValueError(f"Input is not byte-identical to reference: {path}")
    topology.prepare(batch, instances, full_reference=False, **settings)
    batch = Path(batch).resolve()
    manifest = base.read(batch / "experiment_manifest.json")
    manifest["schema"] = "assignment-support-benchmark/1"
    manifest["jobs"] = jobs_for(manifest["input_origins"])
    manifest["protocol"].update(
        comparison="previous tight_airline_bounds+topology_cuts versus adding assignment_support_cuts",
        same_binary_for_all_new_variants=True, same_binary_as_reference=False,
        cross_batch_single_seed=True, full_reference=False,
        assignment_support_before_topology=True, assignment_support_cuts=True)
    for key, value in (("assignment_support_seconds", assignment_support_seconds),
                       ("assignment_support_cut_limit", assignment_support_cut_limit),
                       ("assignment_support_max_cuts", assignment_support_max_cuts),
                       ("assignment_support_floor", assignment_support_floor)):
        if value is not None:
            manifest["protocol"][key] = value
    ref_dir = batch / "reference"
    ref_dir.mkdir()
    shutil.copy2(reference / "experiment_manifest.json", ref_dir / "experiment_manifest.json")
    shutil.copy2(reference / "analysis/comparison.json", ref_dir / "previous_report.json")
    write_json(ref_dir / "validated_comparison.json", dict(protocol=previous["protocol"], rows=rows))
    trace_meta = []
    for job_id, path in traces:
        target = ref_dir / "progress" / job_id / "progress.jsonl"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        trace_meta.append(dict(job=job_id, source=str(path), copied=str(target.relative_to(batch)),
                               sha256=base.sha256(target)))
    manifest["reference"] = dict(source=str(reference), revalidated_utc=base.existing.utc(),
        manifest_sha256=base.sha256(reference / "experiment_manifest.json"),
        previous_report_sha256=base.sha256(reference / "analysis/comparison.json"),
        compared_variants=sorted(REFERENCE_VARIANTS), progress=trace_meta)
    manifest["frozen_sha256"].update({str(path.relative_to(batch)): base.sha256(path)
                                     for path in sorted(ref_dir.rglob("*")) if path.is_file()})
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
                input_generation=False, reference_unchanged=True)


def _guarded_execute(batch, manifest, job_id, command, attempt, cancel, **kwargs):
    routed = list(command)
    if "-m" in routed:
        index = routed.index("-m") + 1
        if routed[index] in (base.__name__, topology.MODULE):
            routed[index] = MODULE
    return _BASE_GUARD(batch, manifest, job_id, routed, attempt, cancel, **kwargs)


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _difference(value, baseline):
    return value-baseline if _number(value) and _number(baseline) else None


def _record(batch, job, state):
    row = _TOPOLOGY_RECORD(batch, job, state)
    row["assignment_support_cuts_requested"] = job["assignment_support_cuts"]
    source = Path(batch) / state.get("result_directory", state.get("directory", "_missing"))
    result = base.optional_read(source / "result.json")
    summary = result.get("assignment_support_summary") or {}
    accelerations = result.get("accelerations") or {}
    row["assignment_support_cuts_enabled"] = summary.get("enabled", accelerations.get("assignment_support_cuts"))
    for key, value in summary.items():
        row["assignment_support_"+key] = json.dumps(value, sort_keys=True) if isinstance(value, (list, dict)) else value
    for key, value in (result.get("acceleration_statistics") or {}).items():
        if key.startswith("assignment_support"):
            row[key] = json.dumps(value, sort_keys=True) if isinstance(value, (list, dict)) else value
    for key in ("assignment_support_cut_records", "assignment_support_attempts"):
        if isinstance(result.get(key), list):
            row[key+"_record_count"] = len(result[key])
    if "assignment_support_preprocessing" in result:
        row["assignment_support_preprocessing_metadata"] = json.dumps(result["assignment_support_preprocessing"], sort_keys=True)
    root = row.get("continuous_root_lower_bound")
    row["global_lb_minus_continuous_root"] = _difference(row["lower_bound"], root)
    row["integer_lb_minus_continuous_root"] = _difference(row.get("integer_master_best_lower_bound"), root)
    row["root_comparison_tolerance"] = 1e-7*max(1, abs(root)) if _number(root) else None
    difference = row["global_lb_minus_continuous_root"]
    row["global_lb_exceeds_continuous_root"] = (difference > row["root_comparison_tolerance"]
                                               if _number(difference) else None)
    reference = base.optional_read(Path(batch) / "reference/validated_comparison.json")
    old = next((item for item in reference.get("rows", []) if item["instance"] == job["instance"]
                and item["variant"] == job["variant"].replace("assignment_support", "strengthened")), {})
    for field in ("lower_bound", "upper_bound", "gap", "continuous_root_lower_bound",
                  "integer_master_best_lower_bound", "solve_seconds", "integer_socp_calls",
                  "topology_seconds", "topology_skipped_integer_subproblems"):
        row["previous_strengthened_"+field] = old.get(field)
        row["change_from_previous_"+field] = _difference(row.get(field), old.get(field))
    if row.get("all_paths") is None and old:
        for key in ("all_paths", "all_service_paths", "all_empty_paths", "initial_segments"):
            row[key] = old.get(key)
        row["full_path_count_source"] = "reference/"+str(old.get("full_path_count_source"))
    return row


@contextmanager
def adapters():
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


def _report(batch):
    outcome = _BASE_REPORT(batch)
    batch = Path(batch)
    data = base.read(batch / "analysis/comparison.json")
    reference = base.read(batch / "reference/validated_comparison.json")
    rows, cfg = data["rows"], data["protocol"]
    complete = all(row["scheduler_status"] not in ("pending", "running") for row in rows)
    fmt, table = topology._fmt, topology._table
    lines = ["# 分配与路径支持成本割比较" + ("" if complete else "（尚未完成）"), "",
        "本轮仅增加 assignment-support cuts，保留上一轮的 tight airline bounds 与 topology cuts。"
        "L01/L02、K=2、输入、需求重构、运营限制和其他加速均沿用当前代码。", "",
        f"每组预算 {cfg['seconds_per_job']:g} 秒，{cfg['parallel_jobs']} 组并行，每组 {cfg['threads_per_job']} 线程，"
        f"目标 gap {100*cfg['relative_gap_target']:.2f}%，seed={cfg['seed']}。"
        "新增分离和缓存构造计入求解预算。旧结果只读复制并重新核对结果、解、路径流和输入哈希。"
        "这是跨批次单种子比较；并发资源竞争可能影响搜索，不据单次结果宣称普遍提速。", "",
        "## 最终上下界", "",
        "UB/LB 只展示独立验证通过的结果。各组 gap 使用该组自己的 UB/LB，未拼接不同方法的界。", ""]
    lines += table(["实例", "方法", "原 LB", "新 LB", "LB 改变", "原 UB", "新 UB", "原 gap", "新 gap", "预算耗时 s", "验证"],
        [(Path(row["instance"]).name, row["method"], row.get("previous_strengthened_lower_bound"), row["lower_bound"],
          row.get("change_from_previous_lower_bound"), row.get("previous_strengthened_upper_bound"), row["upper_bound"],
          fmt(row.get("previous_strengthened_gap"), percent=True), fmt(row["gap"], percent=True),
          row.get("solve_seconds"), row["validation_passed"]) for row in rows])
    lines += ["", "## 是否越过连续根界", "",
        "连续根界仅在 bound_valid=true 时列出。原始差值和整数主问题最好界均保留；"
        "判断越过连续根界使用 1e-7 × max(1, |连续根界|) 的报告容差，避免把数值舍入算作提高。"
        "整数主问题最好界不代替最终全局 LB。", ""]
    lines += table(["任务", "连续根界", "最终全局 LB", "LB−连续根界", "超过容差", "整数主问题最好 LB", "整数 LB−连续根界"],
        [(row["job"], row.get("continuous_root_lower_bound"), row["lower_bound"],
          row.get("global_lb_minus_continuous_root"), row.get("global_lb_exceeds_continuous_root"),
          row.get("integer_master_best_lower_bound"), row.get("integer_lb_minus_continuous_root")) for row in rows])
    lines += ["", "## 割与求解开销", "", "缺失字段显示 —，不视作零；完整诊断保留在 CSV/JSON。", ""]
    diagnostic = sorted({key for row in rows for key in row if key.startswith("assignment_support_")
                         and not key.endswith("metadata")
                         and key not in ("assignment_support_cuts_requested", "assignment_support_cuts_enabled")})
    lines += table(["任务", *diagnostic], [(row["job"], *(row.get(key) for key in diagnostic)) for row in rows])
    lines += [""]
    lines += table(["任务", "整数主问题次数", "整数 SOCP 次数", "全骑手候选比例", "拓扑检查 s", "拓扑跳过 SOCP", "主问题优化 s", "子问题总 s"],
        [(row["job"], row.get("integer_master_calls"), row.get("integer_socp_calls"),
          fmt(row.get("all_courier_candidate_ratio"), percent=True), row.get("topology_seconds"),
          row.get("topology_skipped_integer_subproblems"), row.get("master_optimize_seconds"),
          row.get("subproblem_wall_seconds")) for row in rows])
    def size(row, name):
        return "/".join(fmt(row.get(name+"_"+key)) for key in ("variables", "linear_rows", "cones"))
    lines += ["", "## 规模与终止", "", "模型规模依次为变量/线性约束/锥。完整路径计数来自相同输入的上一轮完整枚举；来源逐行保留。", ""]
    lines += table(["任务", "主问题", "整数子问题峰值", "完整服务/空飞路径", "已生成服务/空飞列", "终止", "资源原因", "进程墙钟 s"],
        [(row["job"], size(row,"master"), size(row,"subproblem_peak"),
          f"{fmt(row.get('all_service_paths'))}/{fmt(row.get('all_empty_paths'))}",
          f"{fmt(row.get('generated_service_columns'))}/{fmt(row.get('generated_empty_columns'))}",
          row.get("termination_reason"), row.get("resource_limit_kind"), row.get("external_wall_seconds")) for row in rows])
    lines += ["", "## 上一轮全部参考结果", "", "原 baseline、strengthened 和完整路径参考均保留，原批次未修改。", ""]
    lines += table(["实例", "变体", "UB", "LB", "gap", "预算耗时 s", "验证"],
        [(Path(row["instance"]).name, row["variant"], row["upper_bound"], row["lower_bound"],
          fmt(row["gap"], percent=True), row.get("solve_seconds"), row["validation_passed"]) for row in reference["rows"]])
    lines += ["", "## 证据", "",
        f"终止 {sum(row['scheduler_status'] not in ('pending','running') for row in rows)}/{len(rows)}；"
        f"独立验证通过 {sum(bool(row['validation_passed']) for row in rows)}/{len(rows)}。", "",
        "源码、二进制、输入及参考结果的哈希见 `../experiment_manifest.json`；"
        "旧数据快照见 `../reference/`；新原始日志和验证见 `../runs/`。"
        "单组 RSS 4 GiB、批次活动组 RSS 合计 8 GiB、系统可用内存保底 2 GiB；"
        f"含验证外部墙钟上限 {cfg.get('hard_wall_seconds',720):g} 秒。资源终止不表示数学不可行。"]
    (batch / "analysis/REPORT.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    return dict(outcome, complete=complete)


def plot(batch):
    """Inspect lower-bound trajectories only after all new runs are validated."""
    from src.reporting.scale_ladder import _progress
    from src.reporting.style import _style, _grid, plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import StrMethodFormatter

    batch = Path(batch).resolve()
    manifest = base.verify(batch)
    state = base.read(batch / "state.json")["jobs"]
    rows = base.read(batch / "analysis/comparison.json")["rows"]
    if any(row["scheduler_status"] in ("pending", "running") or not row["validation_passed"] for row in rows):
        raise ValueError("All four new runs must terminate and pass validation before plotting")
    previous = base.read(batch / "reference/validated_comparison.json")["rows"]
    selected = [(row, batch / state[row["job"]]["directory"], "new") for row in rows]
    selected += [(row, batch / "reference/progress" / row["job"], "previous") for row in previous
                 if row["variant"] in ("lifted_strengthened", "conditional_strengthened")]
    points, identities, rejected = [], [], []
    for row, attempt, cohort in selected:
        tracking = dict(id=row["job"], level=Path(row["instance"]).name, K=2, method=row["method"],
                        state="validated", upper_bound=row["upper_bound"])
        accepted, refused = _progress(dict(row=tracking, attempt=attempt))
        rejected.extend(refused)
        if not accepted:
            raise ValueError(f"No accepted global-bound trajectory for {row['job']}")
        for point in accepted:
            point.update(cohort=cohort, point_source="program_log")
        final_time = max(float(row.get("solve_seconds") or 0), accepted[-1]["elapsed_seconds"])
        final = dict(**{key: tracking[key] for key in ("id", "level", "K", "method")},
                     state="validated", line=None, elapsed_seconds=final_time,
                     upper_bound=row["upper_bound"], lower_bound=row["lower_bound"], gap=row["gap"],
                     event="independently_validated_final_result", evidence="validated_comparison_row",
                     cohort=cohort, point_source="validated_final")
        points += [*accepted, final]
        identities.append(dict(job=row["job"], cohort=cohort, progress_sha256=base.sha256(attempt / "progress.jsonl"),
                               result_sha256=row["result_sha256"], final_lower=row["lower_bound"]))
    output = batch / "analysis/figures"
    output.mkdir(exist_ok=True)
    with (batch / "analysis/global_bound_progress.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(points[0]))
        writer.writeheader()
        writer.writerows(points)
    _style()
    plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":10})
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.8))
    colors = {"benders_lifted":"#386b91", "benders_conditional":"#a95779"}
    for ax, level in zip(axes, ("L01", "L02")):
        values, finals = [], []
        roots = [row["continuous_root_lower_bound"] for row in rows if Path(row["instance"]).name == level
                 and _number(row.get("continuous_root_lower_bound"))]
        for method, color in colors.items():
            for cohort, linestyle in (("previous", "--"), ("new", "-")):
                subset = sorted((point for point in points if point["level"] == level
                                 and point["method"] == method and point["cohort"] == cohort),
                                key=lambda point: point["elapsed_seconds"])
                ax.step([point["elapsed_seconds"] for point in subset], [point["lower_bound"] for point in subset],
                        where="post", color=color, linestyle=linestyle, linewidth=1.5)
                final = next(point for point in subset if point["point_source"] == "validated_final")
                ax.plot(final["elapsed_seconds"], final["lower_bound"], marker="o", color=color,
                        markerfacecolor=color if cohort == "new" else "white", markersize=4)
                finals.append(final["lower_bound"])
                values.extend(point["lower_bound"] for point in subset if point["elapsed_seconds"] >= 30)
        values.extend(roots)
        if roots:
            ax.axhline(max(roots), color="#777777", linewidth=.9, linestyle=":")
            # Focus the requested integer-phase comparison without changing values.
            threshold = min(roots)-1e-7*max(1, abs(min(roots)))
            values = [value for value in values if value >= threshold]
        values.extend(finals)
        lo, hi = min(values), max(values)
        pad = max((hi-lo)*.12, abs(hi)*.0001, .01)
        ax.set_ylim(lo-pad, hi+pad)
        ax.set_xlim(0, manifest["protocol"]["seconds_per_job"]*1.025)
        ax.set_xlabel("Elapsed time (s)")
        ax.set_ylabel("Certified global lower bound (CNY/h)")
        ax.set_title(f"{level}: expanded lower-bound scale", loc="left")
        ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.2f}"))
        _grid(ax, "both")
    handles = [Line2D([0],[0],color=color,label=label) for color,label in
               ((colors["benders_lifted"],"Lifted"),(colors["benders_conditional"],"Conditional"))]
    handles += [Line2D([0],[0],color="#333333",linestyle=style,label=label) for style,label in
                (("--","Previous strengthened"),("-","With assignment-support cuts"),(":","Continuous root (new)"))]
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(.5,.045), frameon=False)
    fig.suptitle("Assignment-support cuts: certified lower bounds", y=.98)
    fig.text(.5,.015,"Expanded cost scales; early points may fall below the axes. Final markers are independently validated.",
             ha="center",fontsize=8)
    fig.subplots_adjust(left=.09,right=.985,top=.86,bottom=.27,wspace=.28)
    for extension in ("png","pdf"):
        fig.savefig(output / ("lower_bound_comparison."+extension), dpi=240, bbox_inches="tight", pad_inches=.12)
    plt.close(fig)
    write_json(batch / "analysis/figure_audit.json", dict(identities=identities, rejected_progress=rejected,
        accepted_points=len(points), final_values="independently validated rows", expanded_axes=True))
    return dict(figure=str(output / "lower_bound_comparison.pdf"), accepted_points=len(points))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "verify", "run", "launch", "report", "status", "plot", "audit-worker"))
    parser.add_argument("--batch", type=Path, default=DEFAULT_BATCH)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--binary", type=Path)
    parser.add_argument("--assignment-support-seconds", type=float)
    parser.add_argument("--assignment-support-cut-limit", type=int)
    parser.add_argument("--assignment-support-max-cuts", type=int)
    parser.add_argument("--assignment-support-floor", type=float)
    parser.add_argument("--jobs", nargs="+")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--job-id")
    parser.add_argument("--attempt", type=Path)
    args = parser.parse_args(argv)
    with adapters():
        if args.action == "prepare":
            result = prepare(args.batch, [ROOT / f"outputs/tsl_scale_ladder/instances/L{level:02d}" for level in (1,2)],
                reference=args.reference, binary=args.binary,
                assignment_support_seconds=args.assignment_support_seconds,
                assignment_support_cut_limit=args.assignment_support_cut_limit,
                assignment_support_max_cuts=args.assignment_support_max_cuts,
                assignment_support_floor=args.assignment_support_floor)
        elif args.action == "verify":
            result = dict(passed=bool(base.verify(args.batch)), solver_started=False)
        elif args.action == "run":
            result = base.run(args.batch, only=args.jobs, retry_failed=args.retry_failed)
        elif args.action == "launch":
            result = base.launch(args.batch)
        elif args.action == "report":
            result = _report(args.batch)
        elif args.action == "plot":
            result = plot(args.batch)
        elif args.action == "audit-worker":
            if not args.job_id or not args.attempt:
                parser.error("audit-worker requires --job-id and --attempt")
            result = base.audit_worker(args.batch, args.job_id, args.attempt)
        else:
            result = base.existing.status_summary(args.batch)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
