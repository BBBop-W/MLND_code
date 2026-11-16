"""Read-only, provenance-aware reports for the independent BDD root experiment.

Usage: python -m src.reporting.bdd outputs/.../comparison
Running jobs are supported: certificate JSONL records fill the intentionally
compact live snapshots. Solver-reported progress is never a validated primal.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

from .style import _grid, _save, _style, plt


LABELS = {"fixed": "固定输入乘子池", "search": "BDD 乘子搜索", "full_cplex": "CPLEX 完整模型"}
COLORS = {"fixed": "#7f8790", "search": "#217b8c", "full_cplex": "#bd7434"}


def finite(value):
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (ValueError, TypeError):
        return None
    return number if math.isfinite(number) and abs(number) < 1e90 else None


def read(path):
    path = Path(path)
    return json.loads(path.read_text()) if path.is_file() else {}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def records(path):
    """Only a truncated final append is ignorable; corrupt complete lines fail."""
    if not Path(path).is_file():
        return []
    lines = Path(path).read_text().splitlines(keepends=True)
    result = []
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            result.append(json.loads(line))
        except json.JSONDecodeError:
            if i == len(lines) - 1 and not line.endswith("\n"):
                break
            raise
    return result


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        writer.writerows(rows)


def histories(result, path):
    search = result.get("search", {})
    names = ("root_curve", "evaluations", "cuts", "primal_planes", "multiplier_steps")
    if all(isinstance(search.get(name), list) for name in names):
        return {name: search[name] for name in names}
    found = {name: {} for name in names}
    for entry in records(str(path) + ".search.jsonl"):
        name = entry.get("history")
        if name in found:
            index = int(entry["index"])
            if index in found[name] and found[name][index] != entry["record"]:
                raise ValueError(f"Conflicting certificate index {name}:{index}")
            found[name][index] = entry["record"]
    return {name: [found[name][i] for i in sorted(found[name])] for name in names}


def validated_primal(result):
    return bool(result.get("validated_incumbent") or (result.get("matrix_residuals") or {}).get("passed")
                or result.get("feasibility_basis") == "all_courier_construction"
                or result.get("incumbent_kind") == "all_courier_baseline")


def stage_rows(job, result, execution, prep, history, events):
    """Every returned phase is exclusive; parent aggregates are not repeated."""
    stages = []

    def add(stage, value, description):
        value = finite(value)
        if value is not None:
            if value < -1e-5:
                raise ValueError(f"Negative exclusive stage {stage}: {value}")
            stages.append(dict(K=job["K"], method=job["method"], stage=stage,
                               seconds=max(0.0, value), description=description, additive=True))

    if job["method"] != "full_cplex":
        add("common_continuous_cg", prep.get("source_continuous_seconds"), "共同连续根 CG，复用证书但计入原始实际时间")
        attempts = prep.get("anchor_attempts", [])
        good = sum(a.get("process_wall_seconds", 0) for a in attempts if a.get("pricing_complete") and a.get("bound_valid"))
        failed = sum(a.get("process_wall_seconds", 0) for a in attempts if not (a.get("pricing_complete") and a.get("bound_valid")))
        add("common_anchor_success", good, "本次成功锚点准备")
        add("common_anchor_failed", failed, "失败锚点尝试，仍计入预算")
        add("common_anchor_reused", prep.get("reused_anchor_source_seconds", 0), "复用锚点原始实际时间")
        add("input_read", result.get("input_read_seconds"), "读取及来源核验")
        add("full_enumeration", result.get("enumeration_seconds"), "完整物理路径枚举")
        setup = result.get("oracle_setup", {})
        add("oracle_matrix_build", setup.get("common_matrix_build_seconds"), "完整内问题公共矩阵构造")
        copies = sum(x.get("matrix_copy_seconds", 0) for x in setup.get("solver_models", []))
        builds = sum(x.get("solver_build_seconds", 0) for x in setup.get("solver_models", []))
        add("oracle_matrix_copies", copies, "三种整数性内问题矩阵副本")
        add("oracle_solver_build", builds, "持久化 CPLEX 内问题模型构造")
        if "oracle_setup_seconds" in result:
            add("oracle_setup_other", max(0, result["oracle_setup_seconds"] - setup.get("common_matrix_build_seconds", 0) - copies - builds), "内问题构造其他开销")
        search = result.get("search", {})
        times = search.get("times", {})
        oracle_detail = {}
        for field, label in (("objective_update_seconds", "oracle_objective_update"), ("optimize_seconds", "oracle_optimize"), ("validation_seconds", "oracle_validation")):
            oracle_detail[label] = sum(x.get(field, 0) for x in history.get("evaluations", []))
            add(label, oracle_detail[label], {"oracle_objective_update": "内问题目标更新", "oracle_optimize": "整数拉格朗日内问题优化", "oracle_validation": "内问题原始矩阵、整数性及证书验证"}[label])
        add("oracle_other", max(0, times.get("oracle_wall", 0) - sum(oracle_detail.values())), "内问题封装及其余开销；不重复 oracle_wall")
        for stage, value in times.items():
            if stage != "oracle_wall":
                add(stage, value, {"master_record_build": "精简根主问题记录构造", "master_solver_build": "根主问题 CPLEX 构造", "master_optimize": "根主问题优化", "master_validation": "根 LP 认证", "multiplier_lp_build": "乘子 box LP 构造", "multiplier_lp_optimize": "乘子 box LP 优化", "multiplier_lp_validation": "乘子 LP 解及 box 上界认证", "cut_insert": "插入认证 BDD 割", "checkpoint_io": "增量证书与快照写入", "other_seconds": "搜索其余开销"}.get(stage, stage))
    else:
        add("input_read", result.get("input_read_seconds"), "读取输入")
        add("full_enumeration", result.get("enumeration_seconds"), "完整路径枚举")
        times = result.get("timings", {})
        add("full_model_build", times.get("build"), "完整 MISOCP 记录与 CPLEX 构造")
        add("full_model_optimize", times.get("optimize"), "完整 MISOCP 分支定界")
        if all(key in times for key in ("total", "build", "optimize")):
            add("full_model_postprocess", max(0, times["total"] - times["build"] - times["optimize"]), "完整模型解修复、验证及内部输出")
        if not times:
            for e in events:
                if e.get("event") == "phase_end" and e.get("stage") in ("input_read", "path_enumeration"):
                    add(e["stage"], e.get("seconds"), "运行中已完成阶段")
                if e.get("event") in ("model_record_complete", "solver_model_build_end"):
                    add(e["event"], e.get("seconds"), "运行中已完成构造阶段")
    # Close the ledger against the process timer only after process completion.
    process = finite(execution.get("process_wall_seconds"))
    common = float(job.get("common_seconds", 0))
    if process is not None:
        known = sum(s["seconds"] for s in stages)
        add("process_startup_final_output_unallocated", max(0, process + common - known), "进程启动、最终输出与未单列开销")
    return stages


def collect(root):
    root = Path(root).resolve()
    manifest = read(root / "manifest.json")
    prep_doc = read(manifest["preparation_file"])
    prep = {case["K"]: case for case in prep_doc["cases"]}
    frozen_checks = []
    for name, expected in manifest.get("frozen_files", {}).items():
        file = root / name
        frozen_checks.append(dict(file=str(file), expected=expected, actual=digest(file) if file.is_file() else None))
    preparation_match = digest(manifest["preparation_file"]) == manifest["preparation_sha256"]
    provenance_ok = preparation_match and all(c["actual"] == c["expected"] for c in frozen_checks)
    comparisons, stages, curves, job_audits = [], [], [], []
    expected_model = manifest["frozen_files"]["instance/model.json"]
    expected_manifest = manifest["frozen_files"]["instance/manifest.json"]
    expected_manifest_document = read(root / "instance/manifest.json")
    for job in manifest["jobs"]:
        k, method = job["K"], job["method"]
        directory, result_path = Path(job["directory"]), Path(job["result_path"])
        execution = read(directory / "execution.json")
        result = read(result_path)
        complete = execution.get("status") == "completed" and execution.get("returncode") == 0 and bool(result)
        row = dict(K=k, method=method, label=LABELS[method], process_status=execution.get("status", "not_started"),
                   returncode=execution.get("returncode"), completed=complete, termination=result.get("status", result.get("termination_reason")),
                   budget_seconds=manifest["budget_seconds"], common_seconds=job.get("common_seconds", 0),
                   process_wall_seconds=finite(execution.get("process_wall_seconds")), charged_wall_seconds=finite(execution.get("charged_wall_seconds")),
                   configured_process_budget_seconds=job["run_budget_seconds"], lower_bound=None, upper_bound=None, gap_percent=None,
                   bound_type="certified_root_lower_bound" if method != "full_cplex" else "full_model_global_tree_lower_bound",
                   bound_evidence=None, upper_bound_validated=False, baseline_root_objective=None, certified_root_gain=None,
                   common_continuous_reference=prep[k]["source_floor"], delta_vs_continuous_reference=None,
                   cuts=None, evaluations=None, root_batches=None, elapsed_snapshot_seconds=None, result_path=str(result_path))
        history, events = {}, []
        match = provenance_ok
        if method != "full_cplex":
            search = result.get("search", {})
            history = histories(result, result_path)
            if result:
                match = match and result.get("input_model_sha256") == expected_model and result.get("K") == k
                match = match and result.get("input_manifest") == expected_manifest_document
                match = match and result.get("anchors_sha256") == manifest["frozen_files"][f"frozen/anchors_K{k}.json"]
                match = match and result.get("continuous_bound_sha256") == manifest["frozen_files"][f"frozen/continuous_K{k}.json"]
            baseline = search.get("baseline", {}) or {}
            if baseline.get("certified"):
                row["baseline_root_objective"] = finite(baseline.get("objective"))
            row["lower_bound"] = finite(search.get("best_lower_bound"))
            row["upper_bound"] = finite(search.get("validated_original_integer_upper_bound"))
            row["upper_bound_validated"] = row["upper_bound"] is not None
            row["bound_evidence"] = "continuous_floor_and_certified_root_LPs" if row["lower_bound"] is not None else None
            row["cuts"], row["evaluations"] = len(history["cuts"]), len(history["evaluations"])
            row["root_batches"] = search.get("root_batches", max(0, len(history["root_curve"]) - 1))
            row["termination"] = search.get("status", row["termination"])
            row["elapsed_snapshot_seconds"] = result.get("wall_before_output", result.get("elapsed_seconds"))
            row["oracle_valid_lower_bound_calls"] = sum(v.get("valid_global_lower_bound", False) for v in history["evaluations"])
            row["oracle_valid_primal_calls"] = sum(v.get("primal_validated", False) for v in history["evaluations"])
            row["oracle_optimal_calls"] = sum(v.get("solver_optimal", False) for v in history["evaluations"])
            row["cuts_violating_generation_target_above_0_001"] = sum(c.get("target_violation", 0) > .001 for c in history["cuts"])
            row["maximum_generation_target_violation"] = max((c.get("target_violation", 0) for c in history["cuts"]), default=0)
            row["primal_plane_count"] = len(history["primal_planes"])
            row["search_wall_seconds"] = finite(search.get("search_wall_seconds"))
            row["master_optimize_seconds"] = finite(search.get("times", {}).get("master_optimize"))
            row["oracle_wall_seconds"] = finite(search.get("times", {}).get("oracle_wall"))
            for point in history["root_curve"]:
                if point.get("certified") and finite(point.get("lower_bound")) is not None:
                    curves.append(dict(K=k, method=method, charged_seconds=job["common_seconds"] + point["elapsed_seconds"],
                                       lower_bound=point["lower_bound"], bound_type=row["bound_type"], evidence="validated_optimal_root_LP", batch=point["batch"]))
            if row["baseline_root_objective"] is not None and row["lower_bound"] is not None:
                row["certified_root_gain"] = max(0, row["lower_bound"] - row["baseline_root_objective"])
            row["method_internal_wall_seconds"] = finite(result.get("wall_before_output"))
        else:
            native = directory / "native"
            events = records(native / "progress.jsonl")
            checkpoint = read(native / "checkpoint.json")
            if result:
                match = match and result.get("input_manifest_sha256") == expected_manifest and result.get("max_orders") == k
                match = match and result.get("binary_sha256") == manifest["binary_hashes"]["full"]
            catalogued = result.get("catalogue_complete", False) or any(e.get("catalogue_complete") for e in events)
            for e in events:
                if e.get("stage") == "full_model" and e.get("event") == "solver_progress" and catalogued:
                    bound = finite(e.get("solver_bound"))
                    if bound is not None:
                        curves.append(dict(K=k, method=method, charged_seconds=e["elapsed_seconds"], lower_bound=bound,
                                           bound_type=row["bound_type"], evidence="solver_reported_complete_full_model", batch=None))
            if result.get("bound_scope") == "global" and result.get("catalogue_complete") and result.get("support_complete"):
                row["lower_bound"] = finite(result.get("bound"))
                row["bound_evidence"] = "complete_full_model_final_global_bound"
            elif catalogued:
                candidates = [p for p in curves if p["K"] == k and p["method"] == method]
                if candidates:
                    row["lower_bound"] = max(p["lower_bound"] for p in candidates)
                    row["bound_evidence"] = "solver_reported_progress_pending_final_validation"
            feasible = [x for x in (checkpoint, result) if validated_primal(x) and finite(x.get("objective")) is not None]
            if feasible:
                row["upper_bound"] = min(x["objective"] for x in feasible)
                row["upper_bound_validated"] = True
            row["termination"] = result.get("termination_reason", "running" if execution.get("status") == "running" else "not_started")
            row["elapsed_snapshot_seconds"] = result.get("wall_before_output", max((e.get("elapsed_seconds", 0) for e in events), default=0))
            row["method_internal_wall_seconds"] = finite(result.get("wall_before_output"))
            row["nodes"] = result.get("node_count")
            row["has_solver_solution"] = result.get("has_solver_solution")
            row["reported_incumbent_kind"] = result.get("incumbent_kind")
            row["solver_candidate_objective"] = finite(result.get("solver_candidate_objective"))
            row["solver_raw_objective"] = finite(result.get("solver_raw_objective"))
            validation_path = native / "native_validation.json"
            validation = read(validation_path)
            validation_matches = bool(validation) and validation.get("result_sha256") == digest(result_path) and validation.get("manifest_sha256") == expected_manifest and validation.get("max_orders") == k
            row["independent_physical_validation"] = bool(validation_matches and validation.get("passed") and validation.get("checks", {}).get("passed")) if validation else None
            row["independent_validation_sha256"] = digest(validation_path) if validation else None
            row["postrun_validation_seconds_excluded"] = (validation.get("read_seconds", 0) + validation.get("validation_and_export_seconds", 0)) if validation else None
            if validation:
                match = match and validation_matches
        row["provenance_verified"] = match
        row["internal_plus_common_seconds"] = (row["method_internal_wall_seconds"] + job.get("common_seconds", 0)) if row.get("method_internal_wall_seconds") is not None else None
        if row["lower_bound"] is not None and row["upper_bound"] is not None:
            if row["lower_bound"] > row["upper_bound"] + 1e-5 * (1 + abs(row["upper_bound"])):
                raise ValueError(f"LB exceeds independently validated UB in K{k} {method}")
            row["gap_percent"] = 100 * max(0, row["upper_bound"] - row["lower_bound"]) / max(1e-10, abs(row["upper_bound"]))
        if row["lower_bound"] is not None:
            row["delta_vs_continuous_reference"] = row["lower_bound"] - row["common_continuous_reference"]
            endpoint = row["charged_wall_seconds"] or ((row["elapsed_snapshot_seconds"] or 0) + job.get("common_seconds", 0))
            curves.append(dict(K=k, method=method, charged_seconds=endpoint, lower_bound=row["lower_bound"],
                               bound_type=row["bound_type"], evidence="final_endpoint" if complete else "live_snapshot", batch=None))
        stages.extend(stage_rows(job, result, execution, prep[k], history, events))
        comparisons.append(row)
        job_audits.append(dict(K=k, method=method, provenance_verified=match, final_result_present=result_path.is_file(),
                               result_sha256=digest(result_path) if result_path.is_file() else None, completion_verified=complete,
                               independent_physical_validation=row.get("independent_physical_validation"),
                               independent_validation_sha256=row.get("independent_validation_sha256")))
    curves.sort(key=lambda p: (p["K"], p["method"], p["charged_seconds"]))
    audit = dict(created_utc=datetime.now(timezone.utc).isoformat(), manifest_sha256=digest(root / "manifest.json"),
                 preparation_matches=preparation_match, frozen_files=frozen_checks, jobs=job_audits,
                 provenance_passed=all(j["provenance_verified"] for j in job_audits),
                 all_runs_complete=all(r["completed"] for r in comparisons), script_sha256=digest(__file__))
    return manifest, prep, comparisons, stages, curves, audit


def plots(directory, rows, curves, budget):
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.8))
    for ax, k in zip(axes, (1, 2)):
        for method in LABELS:
            selected = [p for p in curves if p["K"] == k and p["method"] == method]
            if not selected:
                continue
            ax.step([p["charged_seconds"] for p in selected], [p["lower_bound"] for p in selected], where="post",
                    color=COLORS[method], linestyle={"fixed": ":", "search": "-", "full_cplex": "--"}[method],
                    linewidth=3 if method == "fixed" else 1.8, zorder=4 if method == "fixed" else 3, label=LABELS[method])
            ax.scatter(selected[-1]["charged_seconds"], selected[-1]["lower_bound"], color=COLORS[method],
                       marker="x" if method == "fixed" else "o", s=30, zorder=5)
        ax.set(title=f"K={k}", xlabel="计费时间 / 秒", ylabel="已取得的下界")
        ax.set_xlim(0, max(budget, max((p["charged_seconds"] for p in curves), default=budget)) * 1.015)
        _grid(ax, "both")
        ax.legend(loc="center right")
    fig.suptitle("BDD 根下界与完整模型树下界", fontsize=17)
    fig.text(.5, -.015, "灰色点线/蓝色实线：BDD 根下界（K=2 重合）；橙色虚线：完整模型树下界。计入预处理；末端标记实际停止/当前时点。", ha="center", fontsize=10)
    fig.tight_layout()
    _save(fig, directory, "lower_bound_curve")
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.7))
    for ax, k in zip(axes, (1, 2)):
        selected = [r for r in rows if r["K"] == k and r["method"] != "full_cplex"]
        heights = [r["certified_root_gain"] or 0 for r in selected]
        bars = ax.bar([LABELS[r["method"]] for r in selected], heights, color=[COLORS[r["method"]] for r in selected], width=.55)
        ax.bar_label(bars, labels=["尚无认证值" if r["certified_root_gain"] is None else f"{h:.4f}" for r, h in zip(selected, heights)], padding=5)
        ax.set(title=f"K={k}", ylabel="认证根下界提升")
        ax.set_ylim(0, max(1, max(heights, default=0) * 1.2))
        _grid(ax, "y")
    fig.suptitle("相对同一初始根模型的认证提升", fontsize=17)
    fig.text(.5, -.01, "新根保守下界 − 原根已验证可行目标；CPLEX 树下界不计入这张根界提升图。", ha="center", fontsize=10)
    fig.tight_layout()
    _save(fig, directory, "certified_root_gain")


def fmt(value, digits=2):
    return "—" if finite(value) is None else f"{value:,.{digits}f}"


def report_text(manifest, prep, rows, stages, audit):
    completed = sum(r["completed"] for r in rows)
    text = ["# BDD 下界试验", "", f"当前完成 {completed}/{len(rows)} 组。固定 TSL 2L / 20H / 50OD，K=1、2；每组总预算上限 {manifest['budget_seconds']:.0f} 秒，单线程、seed={manifest['seed']}、最多 {manifest['max_parallel']} 个进程并发。以下为" + ("最终结果。" if completed == len(rows) else "运行中快照；未完成项不是最终比较。"), "",
            "固定输入乘子池只对两条输入普通割的系数求整数内问题；它不是动态 BDD1。乘子搜索使用带 box 限制的 Kelley LP，未复现论文中的二次近端 QP，也不宣称已求得无限制拉格朗日对偶最优值。主问题始终保留原战略变量及一个运营成本变量。", "",
            "表中的 UB 是该方法自己得到并验证的原整数问题可行成本，LB 是可用下界，gap=(UB−LB)/|UB|。BDD 的 LB 来自加强后的连续根主问题；CPLEX 的 LB 来自完整模型的分支定界树。不能把不同方法的 UB 混合后再解释为算法本身的 gap。", "",
            "| K | 方法 | 状态 / 终止 | UB | LB | gap | 实际进程秒 | 共同准备秒 | 合计计费秒 |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|"]
    if completed == len(rows) and all((r["certified_root_gain"] or 0) < 1e-6 for r in rows if r["method"] != "full_cplex"):
        text[4:4] = ["**本次 boxed Kelley 变体在 K=1、K=2 都没有提高认证根下界。** 这只说明本次实现、搜索范围和预算下没有取得提界，不意味着 BDD 理论无效。K=1 搜索先达到 2000 次调用上限，未耗尽 600 秒；本实验是下界诊断，没有接入生产算法的上界启发式。", ""]
    for r in rows:
        status = f"{r['process_status']} / {r['termination']}"
        text.append(f"| {r['K']} | {r['label']} | {status} | {fmt(r['upper_bound'])} | {fmt(r['lower_bound'])} | {fmt(r['gap_percent'],3)}% | {fmt(r['process_wall_seconds'])} | {fmt(r['common_seconds'])} | {fmt(r['charged_wall_seconds'])} |")
    text.extend(["", "`all_fixed_multipliers_solved` 表示固定乘子池已完成精确内问题求值，因此正常提前停止；不能称为耗满 600 秒。`evaluation_limit` 表示达到预设调用上限；`time_limit` 表示时间限制。进程未结束时，合计实际时间留空，曲线只显示已记录时点。", "",
                 "| K | BDD 方法 | 初始根目标 | 认证根提升 | 割数 | 内问题调用 | 最优完成调用 | 有有效 LB 调用 | 有验证 primal 调用 |",
                 "|---|---|---:|---:|---:|---:|---:|---:|---:|"])
    full_k2 = next((r for r in rows if r["K"] == 2 and r["method"] == "full_cplex"), None)
    if full_k2 and full_k2.get("reported_incumbent_kind") == "all_courier_baseline":
        explanation = (f"K=2 的完整 CPLEX 并非完全没有求得解：has_solver_solution={full_k2.get('has_solver_solution')}，"
                       f"其候选成本为 {fmt(full_k2.get('solver_candidate_objective'))}，高于已知全骑手方案 {fmt(full_k2['upper_bound'])}；"
                       "程序因此保留全骑手方案作为该方法的有效 UB。这个 gap 不能替换为其他历史运行的较好解所计算的 gap。")
        # Keep the explanatory paragraph ahead of the second table.
        table_start = len(text) - 2
        text[table_start:table_start] = [explanation, ""]
    for r in rows:
        if r["method"] != "full_cplex":
            text.append(f"| {r['K']} | {r['label']} | {fmt(r['baseline_root_objective'],5)} | {fmt(r['certified_root_gain'],5)} | {r['cuts']} | {r['evaluations']} | {r.get('oracle_optimal_calls',0)} | {r.get('oracle_valid_lower_bound_calls',0)} | {r.get('oracle_valid_primal_calls',0)} |")
    text.extend(["", "认证根提升采用新根保守下界减去同一初始根模型的已验证可行目标，正值才能排除纯数值误差。单条割截距提高或切掉一个分数点，不等于整个根问题的下界提高。", "",
                 "![下界随计费时间变化](figures/lower_bound_curve.png)", "", "![认证根提升](figures/certified_root_gain.png)", "", "共同准备包括连续 CG 证书和普通割锚点准备，失败尝试同样计费：", ""])
    observations = []
    for r in rows:
        if r["method"] != "search" or not r["completed"]:
            continue
        if r["K"] == 1:
            observations.append(f"K=1 的 {r['cuts']} 条加入割中，{r['cuts_violating_generation_target_above_0_001']} 条在生成点的违反量大于 0.001，最大违反量 {r['maximum_generation_target_violation']:.2f}。这些割确实排除了多个生成点，但重新优化后的根目标仍未上升；不能把结果解释成完全没有产生有效分离。加入本轮有限割后，剩余分数方案仍允许根模型达到原下界。")
            charged = r["charged_wall_seconds"]
            internal = r["search_wall_seconds"]
            if charged and internal:
                observations.append(f"K=1 根主问题优化耗时 {r['master_optimize_seconds']:.2f} 秒，整数内问题总耗时 {r['oracle_wall_seconds']:.2f} 秒，分别占总计费时间的 {100*r['master_optimize_seconds']/charged:.1f}% 和 {100*r['oracle_wall_seconds']/charged:.1f}%；若以搜索内部时间 {internal:.2f} 秒为分母，则分别为 {100*r['master_optimize_seconds']/internal:.1f}% 和 {100*r['oracle_wall_seconds']/internal:.1f}%。这里的内问题总耗时包括更新与验证，阶段表已将它拆开且不重复相加。")
        if r["K"] == 2:
            observations.append(f"K=2 搜索的 {r['evaluations']} 次整数内问题调用中，{r['oracle_valid_primal_calls']} 次返回通过原矩阵验证的整数方案；上平面池只有 {r['primal_plane_count']} 条，即初始解析构造的全关闭、全骑手方案。因此，这次系数搜索没有得到新的整数见证来补充上平面。这不等于原问题没有可行解，也不否定已有全骑手方案的可行性。")
    if observations:
        index = text.index("![下界随计费时间变化](figures/lower_bound_curve.png)")
        text[index:index] = [paragraph for observation in observations for paragraph in (observation, "")]
    for k, p in sorted(prep.items()):
        text.append(f"- K={k}：连续 CG {p['source_continuous_seconds']:.3f} 秒；锚点准备 {p['anchor_preparation_seconds']:.3f} 秒；合计 {p['common_initialization_seconds']:.3f} 秒。" + ("K=2 使用向既有内部点混合 0.001 的分数锚点及此前认证的全骑手锚点；它不是原连续最优点本身。" if p.get("anchor_changed_from_original_root") else ""))
    text.extend(["", "各阶段不重叠计时如下；详细逐项记录见 stage_times.csv。`oracle optimize/update/validation` 已替代父项 oracle_wall，不再重复相加。", "",
                 "| K | 方法 | 共同准备 | 枚举/构造/读取 | 内问题优化 | 内问题更新/验证/其他 | 根主问题优化 | 乘子 LP 构造/优化/验证 | 写出/其他 |",
                 "|---|---|---:|---:|---:|---:|---:|---:|---:|"])
    for r in rows:
        parts = [s for s in stages if s["K"] == r["K"] and s["method"] == r["method"]]
        groups = dict(common=0.0, setup=0.0, oracle=0.0, oracle_other=0.0, master=0.0, multipliers=0.0, other=0.0)
        for s in parts:
            name = s["stage"]
            if name.startswith("common_"): group = "common"
            elif name == "oracle_optimize" or name == "full_model_optimize": group = "oracle"
            elif name.startswith("oracle_") and name in ("oracle_objective_update", "oracle_validation", "oracle_other"): group = "oracle_other"
            elif name == "master_optimize": group = "master"
            elif name.startswith("multiplier_lp_"): group = "multipliers"
            elif any(word in name for word in ("build", "setup", "enumeration", "input_read", "matrix_cop")): group = "setup"
            else: group = "other"
            groups[group] += s["seconds"]
        text.append(f"| {r['K']} | {r['label']} | " + " | ".join(fmt(groups[g]) for g in groups) + " |")
    text.extend(["", "完整模型行的“内问题优化”列表示完整 MISOCP 优化时间，未拆分为 Benders 阶段。实际进程时间包括启动与最终输出；搜索内部计时和进程计时分列，差额保留为其他开销。运行中阶段表只是已经完成调用的快照。", "",
                 f"来源核验：{'通过' if audit['provenance_passed'] else '未通过，必须检查 audit.json'}。输入 model.json SHA256：`{manifest['frozen_files']['instance/model.json']}`。冻结二进制、输入、锚点、连续证书及完整结果的散列见 manifest.json / audit.json。", "",
                 "这是同一台机器上单个种子、最多三个并发进程的一次诊断；只支持本次预算和参数下的结果，不构成普遍提速证据。若根界没有提升，应区分乘子搜索未找到有效分离、内问题下界认证不足、box 范围或根目标变化等原因，不能直接归因于 BDD 理论无效。", ""])
    k1_search = next((r for r in rows if r["K"] == 1 and r["method"] == "search"), None)
    if k1_search and k1_search["completed"]:
        text[-2:-2] = [f"时间口径：K=1 搜索的 common + wall_before_output 为 {k1_search['internal_plus_common_seconds']:.2f} 秒；本报告表格采用 common + 实际进程墙钟，为 {k1_search['charged_wall_seconds']:.2f} 秒。后者还包含进程启动与最终输出。独立审计若采用前一种口径会得到不同的合计数，两者不能混称为同一时间。", ""]
    text[-2:-2] = ["验证记录：Python 单元测试 404 项通过；完整试验独立审计见 [run_audit.json](../../../bdd_multiplier_audit_20261022/run_audit.json)，测试原始输出见 [python_unit_tests.log](../../../bdd_multiplier_audit_20261022/python_unit_tests.log)。本报告的增量历史重建、阶段账目与来源检查见 [report_validation.json](report_validation.json)。", ""]
    independently_checked = [r for r in rows if r["method"] == "full_cplex" and r.get("independent_physical_validation")]
    if independently_checked:
        text[-2:-2] = ["完整模型最终解另经独立物理运营重构检查：" + "；".join(f"K={r['K']} 通过，检查与导出 {r['postrun_validation_seconds_excluded']:.3f} 秒" for r in independently_checked) + "。检查文件中的结果散列与最终结果一致；这些求解后检查时间单独记录，不改变求解预算或最终解。", ""]
    return "\n".join(text)


def analyze(root):
    root = Path(root).resolve()
    manifest, prep, rows, stages, curves, audit = collect(root)
    directory = root / "analysis"
    figures = directory / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    write_csv(directory / "comparison.csv", rows)
    write_csv(directory / "stage_times.csv", stages)
    write_csv(directory / "root_curve.csv", curves)
    (directory / "comparison.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2))
    (directory / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2))
    plots(figures, rows, curves, manifest["budget_seconds"])
    (directory / "REPORT.md").write_text(report_text(manifest, prep, rows, stages, audit))
    return directory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("comparison_directory", type=Path)
    arguments = parser.parse_args()
    print(analyze(arguments.comparison_directory))


if __name__ == "__main__":
    main()
