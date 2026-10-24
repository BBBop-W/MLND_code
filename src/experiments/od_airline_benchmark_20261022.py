"""Same-binary lifted/conditional x OD airline gate comparison, K=1 and K=2.

Reuse the frozen conditional runner's guarded scheduler and independent audit.
No standard inputs are generated and no older batch is modified.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
import json
import math
from pathlib import Path

from src.data.standard import load_standard
from src.experiments import conditional_benchmark_20261022 as base
from src.experiments import topology_benchmark_20261022 as topology
from src.experiments.storage import write_json
from src.reporting.native import native_phase_times


ROOT = base.ROOT
MODULE = "src.experiments.od_airline_benchmark_20261022"
DEFAULT_BATCH = ROOT / "outputs/od_airline_benchmark_20261022"
COMMON_SOLVER_ARGS = (
    "--tight-airline-bounds", "--topology-cuts", "--topology-seconds", "2",
    "--topology-cut-limit", "1000", "--assignment-support-cuts",
    "--assignment-support-seconds", "0.25", "--assignment-support-cut-limit", "2000",
    "--assignment-support-max-cuts", "16", "--assignment-support-floor", "0.0001",
    "--double-support-cuts", "--joint-base-cost-cuts", "--branch-joint-seconds", "0.25",
    "--branch-joint-preprocessing-seconds", "2", "--branch-joint-floor", "0.0001",
    "--branch-joint-cut-limit", "2000", "--branch-joint-max-cuts", "16",
)
_BASE_COMMAND = base.command_for
_BASE_RECORD = base._record
_BASE_GUARD = base.guarded_execute


def jobs_for(origins):
    return [dict(id=f"{origin['id']}_K{k}_{form}_od_{'on' if enabled else 'off'}",
                 variant=f"{form}_od_{'on' if enabled else 'off'}",
                 instance=origin["instance"], K=k, method="benders_"+form,
                 od_airline_bounds=enabled,
                 output=f"runs/{origin['id']}/{form}_od_{'on' if enabled else 'off'}/K{k}")
            for origin in origins for k in (1, 2)
            for enabled in (False, True) for form in ("lifted", "conditional")]


def command_for(batch, manifest, job, attempt):
    command = _BASE_COMMAND(batch, manifest, job, attempt)
    command[command.index("--k")+1] = str(job["K"])
    command += manifest["protocol"]["common_solver_args"]
    if job["od_airline_bounds"]:
        command.append("--od-airline-bounds")
    return command


def prepare(batch, instances, *, binary=None, seconds=600, threads=1, parallel=3,
            gap=.005, seed=43, hard_wall_seconds=720):
    # Both saved cases must be readable before freezing anything.
    for instance in instances:
        for k in (1, 2):
            load_standard(instance, case=f"K{k}")
    base.prepare(batch, instances, binary=binary, seconds=seconds, threads=threads,
                 parallel=parallel, gap=gap, seed=seed, hard_wall_seconds=hard_wall_seconds)
    batch = Path(batch).resolve()
    manifest = base.read(batch / "experiment_manifest.json")
    manifest["schema"] = "od-airline-benchmark/1"
    manifest["jobs"] = jobs_for(manifest["input_origins"])
    manifest["protocol"].update(
        comparison="lifted/conditional x OD airline bounds off/on x K=1/2",
        max_orders=[1, 2], same_binary_for_all_variants=True,
        common_solver_args=list(COMMON_SOLVER_ARGS),
        changed_solver_args=["--od-airline-bounds"], input_generation=False,
        strengthened_cut_domain="integer strategic decisions; not the original continuous relaxation")
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
    routed = list(command)
    if "-m" in routed:
        index = routed.index("-m")+1
        if routed[index] == "src.experiments.conditional_benchmark_20261022":
            routed[index] = MODULE
    return _BASE_GUARD(batch, manifest, job_id, routed, attempt, cancel, **kwargs)


def _record(batch, job, state):
    batch = Path(batch)
    row = _BASE_RECORD(batch, job, state)
    row.update(K=job["K"], variant=job["variant"], od_airline_bounds_requested=job["od_airline_bounds"])
    source = batch / state.get("result_directory", state.get("directory", "_missing"))
    result = base.optional_read(source / "result.json")
    if not result:
        return row
    validation = base.optional_read(source / "native_validation.json")
    valid = bool(validation.get("passed"))
    for name, key in (("result.json", "result_sha256"), ("solution.json", "solution_sha256"),
                      ("path_flows.json", "path_flows_sha256")):
        valid = valid and (source / name).is_file() and validation.get(key) == base.sha256(source / name)
    valid = valid and validation.get("manifest_sha256") == base.sha256(batch / job["instance"] / "manifest.json")
    valid = valid and result.get("max_orders") == job["K"] and result.get("solver") == "CPLEX"
    valid = valid and result.get("method") == "benders_cg"
    valid = valid and result.get("demand_formulation") == job["method"].removeprefix("benders_")
    actual_gate = result.get("od_airline_bounds", (result.get("accelerations") or {}).get("od_airline_bounds"))
    valid = valid and actual_gate is job["od_airline_bounds"]
    row.update(validation_passed=bool(valid), od_airline_bounds_enabled=actual_gate,
               upper_bound=result.get("objective") if valid else None,
               lower_bound=base.certified_bound(result) if valid else None, gap=None)
    if row["upper_bound"] is not None and row["lower_bound"] is not None:
        row["gap"] = (row["upper_bound"]-row["lower_bound"]) / max(1, abs(row["upper_bound"]))
    # Preserve new diagnostics without guessing a missing field means zero.
    for prefix in ("continuous_root", "root_diagnostics", "acceleration_statistics",
                   "conditional_statistics", "topology_summary", "assignment_support_summary",
                   "branch_joint_summary", "resource_lp_summary", "resource_lp_metadata", "od_airline_summary"):
        value = result.get(prefix)
        if isinstance(value, dict):
            for key, item in value.items():
                if not isinstance(item, (list, dict)):
                    row[f"{prefix}_{key}"] = item
            if prefix == "resource_lp_metadata":
                row[prefix+"_json"] = json.dumps(value, sort_keys=True)
    root = result.get("continuous_root") or {}
    row["continuous_root_certified_lower_bound"] = root.get("certified_lower_bound") if root.get("bound_valid") else None
    master = [entry["certified_bound"] for entry in result.get("master_solves", [])
              if topology._number(entry.get("certified_bound"))]
    row["integer_master_best_lower_bound"] = max(master) if master else None
    for key, value in result.items():
        if isinstance(value, list) and (key.endswith("_cuts") or key.endswith("_cut_records")):
            row[key+"_record_count"] = len(value)
    for key, value in (result.get("timings") or {}).items():
        if not isinstance(value, (list, dict)):
            row["timing_"+key] = value
    return row


@contextmanager
def adapters():
    replacements = dict(command_for=command_for, _record=_record,
                        guarded_execute=_guarded_execute, report=report)
    previous = {key: getattr(base, key) for key in replacements}
    try:
        for key, value in replacements.items():
            setattr(base, key, value)
        yield
    finally:
        for key, value in previous.items():
            setattr(base, key, value)


def _csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def exclusive_phase_times(result):
    """Include new separator/continuous-root timers; retain nested CG separately."""
    timing = result.get("timings") or {}
    if "master_build" not in timing:
        return native_phase_times(result)
    total = result.get("wall_before_output")
    if not topology._number(total) or total <= 0:
        return dict(solve_seconds=None, phases={}, available=False)
    phases = {key: float(result.get(source, 0)) for key, source in (
        ("input_read", "input_read_seconds"), ("enumeration", "enumeration_seconds"),
        ("support_preprocessing", "support_preprocessing_seconds"))}
    for key in ("master_build", "greedy_initial_columns", "master_optimize", "master_update",
                "root_master", "capacity_cut_build", "capacity_cut_separation", "resource_lp_build",
                "resource_lp_separation", "continuous_root", "restricted_mip", "incumbent_cleanup",
                "assignment_support_preprocessing", "branch_joint_preprocessing", "topology_separation",
                "assignment_support_separation", "branch_joint_separation", "bookkeeping_and_interrupted_work"):
        phases[key] = float(timing.get(key, 0))
    nested = timing.get("subproblem") or {}
    phases.update({"subproblem_"+key: float(value) for key, value in nested.items()})
    phases["subproblem_other"] = float(timing.get("subproblem_wall", 0))-sum(float(v) for v in nested.values())
    phases["wrapper_other"] = total-sum(phases.values())
    for key, value in phases.items():
        if not math.isfinite(value) or value < -1e-5:
            raise ValueError(f"Invalid disjoint phase {key}: {value}")
        phases[key] = max(0., value)
    return dict(solve_seconds=total, phases=phases, available=True)


def report(batch):
    batch = Path(batch).resolve()
    manifest = base.verify(batch)
    state = base.read(batch / "state.json")["jobs"]
    rows = [_record(batch, job, state[job["id"]]) for job in manifest["jobs"]]
    phases, cut_rows = [], []
    for row in rows:
        source = Path(row.get("result_directory", batch / "_missing"))
        result = base.optional_read(source / "result.json")
        try:
            times = exclusive_phase_times(result)
            for phase, seconds in times["phases"].items():
                phases.append(dict(job=row["job"], K=row["K"], variant=row["variant"], phase=phase,
                                   seconds=seconds, fraction=seconds/times["solve_seconds"] if times["solve_seconds"] else None))
        except (ValueError, TypeError) as exc:
            row["timing_audit_error"] = str(exc)
        for family, entries in result.items():
            if isinstance(entries, list) and (family.endswith("_cuts") or family.endswith("_cut_records")):
                for i, cut in enumerate(entries):
                    cut_rows.append(dict(job=row["job"], family=family, index=i, record=json.dumps(cut, sort_keys=True)))
    pairs = []
    for row in rows:
        if not row["od_airline_bounds_requested"]:
            continue
        previous = next(other for other in rows if other["instance"] == row["instance"]
                        and other["K"] == row["K"] and other["method"] == row["method"]
                        and not other["od_airline_bounds_requested"])
        pair = dict(instance=row["instance"], K=row["K"], method=row["method"],
                    off_job=previous["job"], on_job=row["job"], both_valid=bool(row["validation_passed"] and previous["validation_passed"]))
        for key in ("upper_bound", "lower_bound", "gap", "solve_seconds", "external_wall_seconds"):
            pair["on_minus_off_"+key] = row.get(key)-previous.get(key) if pair["both_valid"] and topology._number(row.get(key)) and topology._number(previous.get(key)) else None
        pairs.append(pair)
    output = batch / "analysis"
    output.mkdir(exist_ok=True)
    write_json(output / "comparison.json", dict(protocol=manifest["protocol"], rows=rows, pairs=pairs))
    _csv(output / "comparison.csv", rows)
    _csv(output / "paired_changes.csv", pairs)
    _csv(output / "stage_times.csv", phases)
    _csv(output / "cut_records.csv", cut_rows)
    cfg = manifest["protocol"]
    fmt, table = topology._fmt, topology._table
    lines = ["# OD 需求与可行取货航线割：新旧 Benders 比较", "",
        f"已终止 {sum(row['scheduler_status'] not in ('pending','running') for row in rows)}/{len(rows)}；"
        f"独立验证及方法/参数身份检查通过 {sum(row['validation_passed'] for row in rows)}/{len(rows)}。", "",
        f"同一份冻结二进制、固定 TSL L02（2L/20H/50OD）、K=1/2；每组 {cfg['seconds_per_job']:g} 秒、"
        f"{cfg['threads_per_job']} 线程、{cfg['parallel_jobs']} 组并行、seed={cfg['seed']}。"
        "每种需求表示单独比较 OD 航线割关闭/开启，所有组保留上一轮 tight-airline、topology、assignment-support、double-support、joint-base-cost 强化。"
        "共享资源与单种子会影响搜索轨迹，本轮不据此宣称普遍提速。", "",
        "预算包括输入、预处理、构造和优化，独立验证另计。UB/LB/gap 只展示通过文件哈希、物理验证和方法身份检查的结果。"
        "限时、数值中止、构造/定价完整性与数学不可行分开报告。", ""]
    lines += table(["K", "需求表示", "OD割", "UB", "全局LB", "gap", "预算时间s", "墙钟s", "终止", "验证"],
        [(r["K"], r["method"], r["od_airline_bounds_requested"], r["upper_bound"], r["lower_bound"], fmt(r["gap"], percent=True),
          r.get("solve_seconds"), r.get("external_wall_seconds"), r.get("termination_reason"), r["validation_passed"]) for r in rows])
    lines += ["", "## 同方法割开关的变化", "", "下列差值为开启−关闭；gap 差值用百分点表示。", ""]
    lines += table(["K", "需求表示", "ΔUB", "ΔLB", "Δgap百分点", "Δ预算时间s"],
        [(p["K"], p["method"], p.get("on_minus_off_upper_bound"), p.get("on_minus_off_lower_bound"),
          None if p.get("on_minus_off_gap") is None else 100*p["on_minus_off_gap"], p.get("on_minus_off_solve_seconds")) for p in pairs])
    lines += ["", "## 根节点与整数有效性", "",
        "原连续模型的可行目标只上界原连续最优值。新增割依赖战略变量整数性，强化后的根界可以超过原连续最优值；"
        "不能把二者的差值解释成错误，也不能把原连续可行解当作强化后根节点的可行解。", ""]
    lines += table(["任务", "原连续认证界", "原连续采用界", "整数主问题最好LB", "根阶段终止", "定价完成"],
        [(r["job"], r.get("continuous_root_certified_lower_bound"), r.get("continuous_root_applied_total_cost_lower_bound"),
          r.get("integer_master_best_lower_bound"), r.get("root_diagnostics_termination_reason"), r.get("pricing_complete")) for r in rows])
    lines += ["", "## 各阶段耗时", "", "以下为完整算法阶段计时；子阶段和所属阶段不重复相加。完整无重叠分解及占比见 stage_times.csv。", ""]
    fields = ("timing_master_optimize", "timing_root_master", "timing_subproblem_wall", "timing_restricted_mip",
              "timing_capacity_cut_separation", "continuous_root_algorithm_wall")
    lines += table(["任务", *fields], [(r["job"], *(r.get(f) for f in fields)) for r in rows])
    lines += ["", "## 新割与现有割", "", "资源 LP 割是在主问题中投影后的成本割；有活跃 OD 对偶价格表示该次割确实使用了新增约束。", ""]
    lines += table(["任务", "资源割", "含OD价格的割", "OD新增主变量", "拓扑割", "分配支持割", "双单与联合割", "资源分离s"],
        [(r["job"], r.get("resource_lp_cuts_record_count"), r.get("od_airline_summary_cuts_with_active_od_prices"),
          r.get("od_airline_summary_master_added_variables"), r.get("topology_summary_cuts"),
          r.get("assignment_support_summary_cuts"), r.get("branch_joint_summary_cuts"),
          r.get("timing_resource_lp_separation")) for r in rows])
    lines += [""]
    lines += table(["任务", "旧聚合航线行", "新增OD航线行", "外部LP对偶变量", "外部LP对偶约束", "割的有效域"],
        [(r["job"], r.get("resource_lp_metadata_aggregate_airline_resource_rows"),
          r.get("resource_lp_metadata_od_airline_resource_rows"), r.get("resource_lp_metadata_dual_variables"),
          r.get("resource_lp_metadata_dual_constraints"), r.get("resource_lp_metadata_validity_scope")) for r in rows])
    lines += ["", "## 证据与资源", "",
        "comparison.csv/JSON 保留全部根节点、容量割、资源割、支持割和条件子问题诊断；cut_records.csv 保存逐割原始记录。"
        "原始解、逐路径流量、独立验证、日志和资源采样位于 runs/；源码、可执行文件及输入 SHA256 位于 experiment_manifest.json。", "",
        "单组 RSS 4 GiB、批次活动组 RSS 8 GiB、系统可用内存保底 2 GiB；每组含验证外部墙钟 720 秒。"
        "资源终止不表示数学不可行。保留失败记录，不把缺失计时或计数视作零。", ""]
    (output / "REPORT.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    return dict(report=str(output / "REPORT.md"), jobs=len(rows), validated=sum(r["validation_passed"] for r in rows))


def plot(batch):
    from src.reporting.scale_ladder import _progress
    from src.reporting.style import _style, _grid, plt
    from matplotlib.lines import Line2D
    batch = Path(batch).resolve()
    manifest = base.verify(batch)
    states = base.read(batch / "state.json")["jobs"]
    rows = base.read(batch / "analysis/comparison.json")["rows"]
    if any(r["scheduler_status"] in ("pending", "running") or not r["validation_passed"] for r in rows):
        raise ValueError("Every run must terminate and pass independent validation before plotting")
    points, identities, rejected = [], [], []
    for row in rows:
        attempt = batch / states[row["job"]]["directory"]
        tracking = dict(id=row["job"], level=Path(row["instance"]).name, K=row["K"],
                        method=row["method"], state="validated", upper_bound=row["upper_bound"])
        accepted, refused = _progress(dict(row=tracking, attempt=attempt))
        rejected += refused
        for point in accepted:
            point.update(variant=row["variant"], od_airline_bounds=row["od_airline_bounds_requested"])
        final = dict(**tracking, line=None, elapsed_seconds=row["solve_seconds"], lower_bound=row["lower_bound"],
                     gap=row["gap"], event="independently_validated_final_result", evidence="validated_result", variant=row["variant"],
                     od_airline_bounds=row["od_airline_bounds_requested"])
        points += [*accepted, final]
        identities.append(dict(job=row["job"], progress_sha256=base.sha256(attempt / "progress.jsonl"), result_sha256=row["result_sha256"]))
    _csv(batch / "analysis/global_bound_progress.csv", points)
    output = batch / "analysis/figures"
    output.mkdir(exist_ok=True)
    _style()
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    colors = {"benders_lifted": "#386b91", "benders_conditional": "#a95779"}
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.6))
    for ax, k in zip(axes, (1, 2)):
        for row in (r for r in rows if r["K"] == k):
            subset = sorted((p for p in points if p["id"] == row["job"]), key=lambda p: p["elapsed_seconds"])
            color = colors[row["method"]]
            ax.step([p["elapsed_seconds"] for p in subset], [p["lower_bound"] for p in subset], where="post",
                    color=color, linestyle="-" if row["od_airline_bounds_requested"] else "--", linewidth=1.5)
            ax.plot(row["solve_seconds"], row["lower_bound"], marker="o", color=color, markersize=4)
        ax.set(title=f"K={k}: certified global lower bounds", xlabel="Elapsed time (s)", ylabel="Cost (CNY/h)")
        ax.set_xlim(0, manifest["protocol"]["seconds_per_job"]*1.025)
        _grid(ax, "both")
    handles = [Line2D([0], [0], color=c, label=m.removeprefix("benders_").title()) for m, c in colors.items()]
    handles += [Line2D([0], [0], color="#333333", linestyle=s, label=label) for s, label in (("--", "OD cuts off"), ("-", "OD cuts on"))]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False)
    fig.tight_layout(rect=(0, .08, 1, 1))
    for extension in ("png", "pdf"):
        fig.savefig(output / f"lower_bound_comparison.{extension}", dpi=240, bbox_inches="tight")
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.3))
    for ax, k in zip(axes, (1, 2)):
        selected = [r for r in rows if r["K"] == k]
        bars = ax.bar(range(len(selected)), [100*r["gap"] for r in selected], color=[colors[r["method"]] for r in selected])
        for bar, row in zip(bars, selected):
            if not row["od_airline_bounds_requested"]:
                bar.set_hatch("//")
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height(), f"{100*row['gap']:.3f}%", ha="center", va="bottom", fontsize=9)
        ax.set_xticks(range(len(selected)), [r["variant"].replace("_od_", "\nOD ") for r in selected])
        ax.set(title=f"K={k}: final validated gap", ylabel="Relative gap (%)")
        ax.set_ylim(0, max(100*r["gap"] for r in selected)*1.2+.05)
        _grid(ax, "y")
    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(output / f"final_gap.{extension}", dpi=240, bbox_inches="tight")
    plt.close(fig)
    write_json(batch / "analysis/figure_audit.json", dict(identities=identities, rejected_progress=rejected,
        accepted_points=len(points), curve_sha256=base.sha256(batch / "analysis/global_bound_progress.csv")))
    return dict(figures=str(output), accepted_points=len(points))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "verify", "run", "launch", "report", "plot", "status", "audit-worker"))
    parser.add_argument("--batch", type=Path, default=DEFAULT_BATCH)
    parser.add_argument("--binary", type=Path)
    parser.add_argument("--seconds", type=float, default=600)
    parser.add_argument("--parallel", type=int, default=3)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--gap", type=float, default=.005)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--hard-wall-seconds", type=float, default=720)
    parser.add_argument("--jobs", nargs="+")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--job-id")
    parser.add_argument("--attempt", type=Path)
    args = parser.parse_args(argv)
    with adapters():
        if args.action == "prepare":
            result = prepare(args.batch, [ROOT / "outputs/tsl_scale_ladder/instances/L02"], binary=args.binary,
                             seconds=args.seconds, parallel=args.parallel, threads=args.threads,
                             gap=args.gap, seed=args.seed, hard_wall_seconds=args.hard_wall_seconds)
        elif args.action == "verify":
            result = dict(passed=bool(base.verify(args.batch)), solver_started=False)
        elif args.action == "run":
            result = base.run(args.batch, only=args.jobs, retry_failed=args.retry_failed)
        elif args.action == "launch":
            result = base.launch(args.batch)
        elif args.action == "report":
            result = report(args.batch)
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
