"""Compare old/new structural formulations within each Benders algorithm."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

from src.experiments import conditional_benchmark_20261022 as base
from src.experiments.od_airline_benchmark_20261022 import exclusive_phase_times
from src.experiments.storage import write_json
from .style import _grid, _save, _style, plt


def number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def csv_file(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def structural_point_violations(result, data, *, tolerance=1e-6):
    """Independent arithmetic on saved fractional points, without optimization."""
    l, h, r = data["n_l"], data["n_h"], data["n_od"]  # noqa: E741
    levels = data["slot_counts"].shape[1]
    assignment_offset = l + h + h * levels
    design = result.get("design")
    point = result.get("operating_point") or {}
    if not result.get("primal_valid") or not isinstance(design, list):
        return dict(available=False, reason="no_validated_fractional_primal"), []
    origin = result.get("assignment_scope") == "origin" or point.get("v_order") == "r_then_departure"
    assignment_count = r*l if origin else r*l*l
    expected_order = "r_then_departure" if origin else "r_then_departure_then_return"
    if (result.get("assignment_scope", "origin" if origin else "base_pair") != ("origin" if origin else "base_pair")
            or len(design) < assignment_offset + assignment_count
            or point.get("v_order") != expected_order
            or any(not isinstance(point.get(key), list) or len(point[key]) != size
                for key, size in (("rho",l),("idle",l),("v",assignment_count)))):
        return dict(available=False, reason="saved_point_dimensions_or_order_mismatch"), []
    if not all(number(value) for values in (design, point["rho"], point["idle"], point["v"]) for value in values):
        return dict(available=False, reason="nonfinite_saved_point"), []
    rows = []
    def add(family, od, site, lhs, rhs):
        rows.append(dict(family=family, od=od, base=site, lhs=lhs, rhs=rhs,
                         violation=max(0., lhs-rhs), exceeds_tolerance=lhs-rhs>tolerance))
    def a(od, departure, returned):
        return design[assignment_offset + (od*l+departure)*l+returned]
    for od in range(r):
        for site in range(l):
            if origin:
                incident = assigned = design[assignment_offset + od*l + site]
                served = point["v"][od*l+site]
            else:
                incident = sum(a(od, departure, returned) for departure in range(l) for returned in range(l)
                               if site in (departure, returned))
                assigned = sum(a(od,site,returned) for returned in range(l))
                served = sum(point["v"][(od*l+site)*l+returned] for returned in range(l))
            add("assignment_incident_clique", od, site, incident, design[site])
            add("shared_rho_executed", od, site, served, point["rho"][site])
            add("shared_rho_unexecuted", od, site, assigned-served, design[site]-point["rho"][site])
    for site in range(l):
        opened, rho, idle = design[site], point["rho"][site], point["idle"][site]
        add("queue_perspective_cone", None, site, opened**2, (idle+opened)*(opened-rho))
    summary = dict(available=True, tolerance=tolerance, variable_indices="zero_based", assessed_point="saved_validated_fractional_primal")
    for family in dict.fromkeys(row["family"] for row in rows):
        selected = [row for row in rows if row["family"] == family]
        summary[family+"_count"] = sum(row["exceeds_tolerance"] for row in selected)
        summary[family+"_max"] = max(row["violation"] for row in selected)
    return summary, rows


def root_plots(directory, rows):
    _style()
    variants = ("baseline", "joint", "perspective", "all")
    names = ("原始", "联合分配", "排队 perspective", "三项全部")
    colors = ("#818a94", "#247d8d", "#bf8b45", "#4b6a9b")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for axis, k in zip(axes, (1, 2)):
        selected = [next(row for row in rows if row["K"] == k and row["variant"] == variant) for variant in variants]
        values = [row.get("root_lower_bound_gain") for row in selected]
        if not all(number(value) for value in values):
            axis.text(.5,.5,"证书不完整",ha="center",va="center",transform=axis.transAxes)
            continue
        bars = axis.bar(names, values, color=colors, width=.6)
        axis.bar_label(bars, labels=[f"{value:+.3f}" for value in values], padding=4)
        axis.set(title=f"K={k} · 原始 LB={selected[0]['root_lower_bound']:.3f}",
                 ylabel="相对原始松弛的认证下界增量", ylim=(min(0,min(values)*1.2),max(1,max(values)*1.2)))
        _grid(axis,"y")
        axis.tick_params(axis="x",labelsize=10)
    fig.tight_layout()
    _save(fig,directory,"root_lower_bound_gain")


def histories(source, row):
    """Only globally scoped certificates enter curves; raw master bounds do not."""
    points = []
    if not row.get("validation_passed"):
        return points
    path = Path(source) / "progress.jsonl"
    if path.is_file():
        lines = path.read_text().splitlines(keepends=True)
        for i, line in enumerate(lines):
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                if i == len(lines) - 1 and not line.endswith("\n"):
                    break
                raise
            elapsed = event.get("elapsed_seconds")
            if not number(elapsed) or event.get("bound_scope") != "global":
                continue
            lower = event.get("lower_bound", event.get("bound"))
            upper = event.get("upper_bound", event.get("objective")) if event.get("validated_incumbent") else None
            if number(lower):
                points.append(dict(job=row["job"], K=row["K"], method=row["method"], version=row["version"],
                                   elapsed_seconds=elapsed, lower_bound=lower,
                                   upper_bound=upper if number(upper) else None, source="global_certificate"))
    if row.get("validation_passed") and number(row.get("solve_seconds")) and number(row.get("lower_bound")):
        points.append(dict(job=row["job"], K=row["K"], method=row["method"], version=row["version"],
                           elapsed_seconds=row["solve_seconds"], lower_bound=row["lower_bound"],
                           upper_bound=row["upper_bound"], source="independently_validated_final_result"))
    return points


def paired(rows):
    pairs = []
    for new in rows:
        if new["version"] != "new":
            continue
        old = next(r for r in rows if r["instance"] == new["instance"] and r["K"] == new["K"]
                   and r["method"] == new["method"] and r["version"] == "old")
        valid = bool(new["validation_passed"] and old["validation_passed"])
        completed = new["scheduler_status"] == old["scheduler_status"] == "completed"
        item = dict(instance=new["instance"], K=new["K"], method=new["method"], old_job=old["job"], new_job=new["job"],
                    both_valid=valid, both_completed=completed, controlled_pair_available=valid and completed)
        for key in ("upper_bound", "lower_bound", "gap", "solve_seconds", "external_wall_seconds",
                    "continuous_root_certified_lower_bound", "timing_master_optimize", "timing_subproblem_wall",
                    "timing_continuous_root", "timing_root_master", "timing_restricted_mip"):
            item["new_minus_old_" + key] = new[key] - old[key] if valid and completed and number(new.get(key)) and number(old.get(key)) else None
        pairs.append(item)
    return pairs


def plots(directory, rows, curves, seconds):
    _style()
    labels = {"benders_lifted": "Lifted Benders–CG", "benders_conditional": "Conditional Benders–CG"}
    colors = {"old": "#818a94", "new": "#247d8d"}
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.4), sharex=True)
    for i, k in enumerate((1, 2)):
        for j, method in enumerate(labels):
            ax = axes[i, j]
            for version in ("old", "new"):
                values = sorted([p for p in curves if p["K"] == k and p["method"] == method and p["version"] == version], key=lambda p: p["elapsed_seconds"])
                if values:
                    ax.step([p["elapsed_seconds"] for p in values], [p["lower_bound"] for p in values], where="post",
                            label="旧版" if version == "old" else "新版（三项加强）", color=colors[version], linestyle="--" if version == "old" else "-")
            ax.set(title=f"K={k} · {labels[method]}", ylabel="全局认证下界", xlim=(0, seconds * 1.02))
            ax.set_xlabel("算法累计时间 / 秒")
            _grid(ax, "both")
            if ax.lines:
                ax.legend()
    fig.tight_layout()
    _save(fig, directory, "lower_bound_curves")
    for i, k in enumerate((1,2)):
        for j, method in enumerate(labels):
            bounds = [row["lower_bound"] for row in rows if row["K"]==k and row["method"]==method and number(row.get("lower_bound"))]
            if bounds:
                margin = max(1., .2*(max(bounds)-min(bounds)))
                axes[i,j].set_ylim(min(bounds)-margin,max(bounds)+margin)
                axes[i,j].set_ylabel("全局认证下界（根下界附近放大）")
    fig.tight_layout()
    _save(fig,directory,"lower_bound_curves_zoom")
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))
    for ax, k in zip(axes, (1, 2)):
        labels_x, heights, bar_colors = [], [], []
        for method in labels:
            for version in ("old", "new"):
                row = next(r for r in rows if r["K"] == k and r["method"] == method and r["version"] == version)
                labels_x.append(("Lifted" if method.endswith("lifted") else "Conditional") + "\n" + ("旧版" if version == "old" else "新版"))
                heights.append(100 * row["gap"] if number(row.get("gap")) else 0)
                bar_colors.append(colors[version])
        bars = ax.bar(labels_x, heights, color=bar_colors, width=.6)
        selected = [next(r for r in rows if r["K"] == k and r["method"] == method and r["version"] == version) for method in labels for version in ("old", "new")]
        ax.bar_label(bars, labels=[f"{height:.3f}%" if number(row.get("gap")) else "待验证" for row, height in zip(selected, heights)], padding=4)
        ax.set(title=f"K={k}", ylabel="各方法自身 UB 与 LB 的 gap / %", ylim=(0, max(1, max(heights) * 1.18)))
        _grid(ax, "y")
    fig.tight_layout()
    _save(fig, directory, "final_gap")


def fmt(value, percent=False):
    if value is None:
        return "—"
    if isinstance(value, int) and not isinstance(value, bool) and not percent:
        return str(value)
    if number(value):
        if percent and 0 < abs(value) < 1e-5:
            return "<0.001%" if value > 0 else ">−0.001%"
        return f"{100 * value:.3f}%" if percent else f"{value:,.3f}"
    return str(value)


def table(headers, rows):
    return ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"] + ["| " + " | ".join(fmt(v) for v in row) + " |" for row in rows]


def report(batch, *, draw=True):
    from src.experiments.structural_benchmark import record
    batch = Path(batch).resolve()
    manifest = base.verify(batch)
    states = base.read(batch / "state.json")["jobs"]
    rows = [record(batch, manifest, job, states[job["id"]]) for job in manifest["jobs"]]
    phases, cuts, curves, evidence = [], [], [], []
    for row in rows:
        source = Path(row.get("result_directory", batch / "_missing"))
        result = base.optional_read(source / "result.json")
        try:
            clock = exclusive_phase_times(result)
            if clock["available"]:
                row.pop("timing_audit_error", None)
                row["solve_seconds"] = clock["solve_seconds"]
                for phase, elapsed in clock["phases"].items():
                    phases.append(dict(job=row["job"], K=row["K"], method=row["method"], version=row["version"], phase=phase,
                                       seconds=elapsed, fraction=elapsed / clock["solve_seconds"]))
                row["exclusive_phase_sum_seconds"] = sum(clock["phases"].values())
        except (ValueError, TypeError) as error:
            row["timing_audit_error"] = str(error)
        for family, entries in result.items():
            if isinstance(entries, list) and (family.endswith("_cuts") or family.endswith("_cut_records")):
                for index, cut in enumerate(entries):
                    cuts.append(dict(job=row["job"], K=row["K"], family=family, index=index, record=json.dumps(cut, sort_keys=True)))
        curves.extend(histories(source, row))
        evidence.append(dict(job=row["job"], result_directory=str(source), result_sha256=row.get("result_sha256"),
                             input_manifest_sha256=row.get("input_manifest_sha256"), binary_sha256=row.get("binary_sha256"),
                             validated=row["validation_passed"], timing_audit_error=row.get("timing_audit_error")))
    pairs = paired(rows)
    output = batch / "analysis"
    output.mkdir(exist_ok=True)
    payload = dict(protocol=manifest["protocol"], binaries=manifest["binaries"], rows=rows, pairs=pairs)
    write_json(output / "comparison.json", payload)
    csv_file(output / "comparison.csv", rows)
    csv_file(output / "paired_changes.csv", pairs)
    csv_file(output / "stage_times.csv", phases)
    csv_file(output / "cut_records.csv", cuts)
    csv_file(output / "bound_curves.csv", curves)
    write_json(output / "analysis_audit.json", dict(manifest_sha256=base.sha256(batch / "experiment_manifest.json"),
        report_source_sha256=base.sha256(__file__),
        record_source_sha256=base.sha256(Path(__file__).parents[1] / "experiments/structural_benchmark.py"),
        frozen_files_verified=True, records=evidence,
        all_validated=all(r["validation_passed"] for r in rows),
        completed=sum(r["scheduler_status"] == "completed" for r in rows),
        phase_audits_passed=all(not r.get("timing_audit_error") for r in rows)))
    cfg = manifest["protocol"]
    lines = ["# 两种 Benders 方法的新旧结构比较", "",
        f"共 {len(rows)} 组，已完成 {sum(r['scheduler_status'] == 'completed' for r in rows)} 组；独立物理验证、方法参数与二进制身份通过 {sum(r['validation_passed'] for r in rows)} 组。", "",
        f"固定 TSL 2L / 20H / 50OD，K=1、2。每组预算 {cfg['seconds_per_job']:g} 秒，{cfg['threads_per_job']} 线程、seed={cfg['seed']}，最多 {cfg['parallel_jobs']} 个并行任务。"
        "分别在 lifted Benders–CG 和 conditional reduced-recourse Benders–CG 内比较旧版与新版，不把两种方法合并成一组。", "",
        "lifted 在子问题中保留完整需求重构辅助变量；conditional 在整数分配固定后计算需求系数，并生成带分配条件的割。"
        "两者的分数根阶段共享 lifted 连续模型。三项加强在固定整数策略下是冗余约束，改进针对分数松弛和后续搜索，不改变固定策略的真实运营最优值。", "",
        "新版共同开启 assignment incident clique、shared rho joint 和 queue perspective 三项结构加强；旧版使用修改前冻结的可执行文件。"
        "其他现有加速设置相同：tight-airline、topology、assignment-support、double-support、joint-base-cost 保留，OD airline 新割关闭。"
        "连续根初始化、定价、构造和优化均计入各自预算，没有外部白送的根证书或 warm start。", "",
        "UB 是该次运行自己保留并独立验证的可行成本，LB 是全局认证下界；gap=(UB−LB)/max(1,|UB|)。"
        "失败、被中断、限时完成、定价未完成和数学不可行分别保留；缺失结果不会补成零。", ""]
    lines += table(["K", "方法", "版本", "UB", "LB", "gap", "算法秒", "进程墙钟秒", "状态", "终止", "验证"],
        [(r["K"], r["method"], r["version"], r["upper_bound"], r["lower_bound"], fmt(r["gap"], True), r.get("solve_seconds"),
          r.get("external_wall_seconds"), r["scheduler_status"], r.get("termination_reason"), r["validation_passed"]) for r in rows])
    lines += ["", "同一 K、同一方法的新旧差值（新版−旧版）；只有两边均正常完成且验证通过才计算正式差值。", ""]
    lines += table(["K", "方法", "ΔUB", "ΔLB", "Δgap百分点", "Δ算法秒", "有效对照"],
        [(p["K"], p["method"], p["new_minus_old_upper_bound"], p["new_minus_old_lower_bound"],
          100 * p["new_minus_old_gap"] if p["new_minus_old_gap"] is not None else None,
          p["new_minus_old_solve_seconds"], p["controlled_pair_available"]) for p in pairs])
    lines += ["", "根阶段诊断：连续根证书和生产 Benders 全局下界分别记录；原连续可行目标不能作为加强后根问题的上界。连续根认证值进入主问题时作保守数值下调；搜索阶段 ΔLB 为最终全局 LB 减去实际加入主问题的根下界。", ""]
    lines += table(["任务", "连续根认证 LB", "搜索阶段ΔLB", "根轮数", "根终止", "完成的整数Benders迭代", "末次定价完成"],
        [(r["job"], r.get("continuous_root_certified_lower_bound"), r.get("global_lower_bound_gain_after_continuous_root"), r.get("root_diagnostics_rounds_count"),
          r.get("root_diagnostics_termination_reason"), r.get("iterations_count"), r.get("pricing_complete")) for r in rows])
    lines += ["", "主问题重解与整数子问题调用分开计数。完成的整数 Benders 迭代为零不表示算法未运行：根初始化、启发式和先行割分离仍可能执行。下表记录现有联合成本/双单割使本轮跳过整数子问题的次数。", ""]
    lines += table(["任务", "主问题重解", "整数CG调用", "根CG调用", "联合/双单割跳过整数CG", "Benders割记录"],
        [(r["job"],r.get("integer_master_solves_count"),r.get("acceleration_statistics_integer_cg_calls"),
          r.get("acceleration_statistics_root_cg_calls"),r.get("branch_joint_summary_skipped_integer_subproblems"),r.get("benders_cuts_record_count")) for r in rows])
    lines += ["", "这轮比较保留了既有割调度：联合成本/双单割或分配支持割一旦加入，就跳过当前候选的整数 CG 并重解主问题，同时重置主问题停滞计数。"
        "因此，持续出现不同的新割也可能占满预算，使认证下界长期不变却始终不进入精确主问题阶段。不能据此将大量主问题耗时归因于单个主问题特别困难，也不能将很少的子问题耗时解释为该方法已高效完成整数迭代。"
        "后续应让同一候选完成这些割与 CG/Benders 割的生成后再重解主问题，并按认证下界是否改善更新停滞计数。此处是对冻结实验的诊断与后续建议，本轮没有修改这项调度。", ""]
    lines += ["", "主要阶段耗时如下。完整互不重叠分解见 stage_times.csv；子问题壁钟包含其构造、求解和定价，不能再与这些子项相加。独立物理验证另计。", ""]
    lines += table(["任务", "主问题优化", "Benders子问题合计", "连续根", "根主问题", "受限MIP", "物理验证"],
        [(r["job"], r.get("timing_master_optimize"), r.get("timing_subproblem_wall"), r.get("timing_continuous_root"),
          r.get("timing_root_master"), r.get("timing_restricted_mip"), r.get("validation_seconds")) for r in rows])
    lines += ["", "静态结构行与动态割分别保存。全部结构摘要和逐割原始记录见 comparison.csv 与 cut_records.csv；原始解、各点流量、运营重构、日志、检查点和资源采样位于每个 runs/*/attempts 目录。", ""]
    lines += table(["任务", "主问题变量/行", "末次子问题变量/行/锥", "理论新增主问题行", "理论新增子问题行"],
        [(r["job"], f"{r.get('master_variables', '—')} / {r.get('master_linear_rows', '—')}",
          f"{r.get('subproblem_last_variables', '—')} / {r.get('subproblem_last_linear_rows', '—')} / {r.get('subproblem_last_cones', '—')}",
          r.get("theoretical_structural_master_added_rows"), r.get("theoretical_structural_subproblem_added_rows")) for r in rows])
    lines += ["", "理论行数按冻结输入的 R、L 和实际开关计算：主问题增加 R×L 行，三项全开时子问题增加 3R×L+2L 行（incident R×L、shared rho 2R×L、perspective 链接与有限库存界 2L）；结构加强本身不增加变量或锥。末次子问题实际大小还受本次列池影响，不能直接当作相同列池下的结构增量。", ""]
    lines += table(["任务", "incident clique", "shared rho", "queue perspective", "容量割", "资源割", "拓扑割", "支持割", "双单/联合割"],
        [(r["job"], r.get("assignment_incident_clique_enabled"), r.get("shared_rho_joint_enabled"), r.get("queue_perspective_enabled"),
          r.get("capacity_cuts_record_count"), r.get("resource_lp_cuts_record_count"), r.get("topology_summary_cuts"),
          r.get("assignment_support_summary_cuts"), r.get("branch_joint_summary_cuts")) for r in rows])
    if draw:
        figures = output / "figures"
        figures.mkdir(exist_ok=True)
        plots(figures, rows, curves, cfg["seconds_per_job"])
        lines += ["", "![各方法自身最终 gap](figures/final_gap.png)", "", "![认证下界曲线：根下界附近放大](figures/lower_bound_curves_zoom.png)", "",
                  "下界图放大根下界附近，以展示这次较小的改进；初始化阶段的较低证书未在放大图纵轴范围内。[完整纵轴历程](figures/lower_bound_curves.png)和原始曲线数据均保留。"]
    lines += ["", "输入、旧/新二进制及求解源码散列全部冻结在 experiment_manifest.json 中。当前报告与指标提取源码散列另见 analysis_audit.json，报告增强没有改动冻结求解实现或原始证书。时间比较仅对应这一输入、种子与共享机器并发条件；单次结果不能推出普遍提速。", ""]
    (output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    return dict(report=str(output / "REPORT.md"), jobs=len(rows), validated=sum(r["validation_passed"] for r in rows), pairs=len(pairs))
