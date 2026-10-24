"""Report a frozen scale experiment; inspect saved evidence and never solve.

Every planned job remains visible, including pending, failed and stale outputs.
Numerical comparisons require current physical-validation hashes. Certified
feasible recovered checkpoints retain their upper bounds with explicit partial
status; a lower bound additionally needs an identified global source.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path

from src.data.standard import load_standard
from src.experiments.storage import sha256, write_json
from .native import RUNS, _csv, _fmt, _number, _read_run, _table, native_phase_times
from .scale_ladder_plots import (
    plot_comparisons, plot_operations, plot_phases, plot_preparation, plot_progress,
)

METHODS = {key: (label, color) for key, label, color in RUNS}
IDENTITY = ("id", "level", "K", "method")
COMPARISON = (
    *IDENTITY, "state", "scheduler_status", "completion_kind", "validation_error",
    "instance", "attempt_directory", "result_directory", "termination_reason",
    "solver_status", "returncode", "stop_reason", "fallback", "upper_bound", "lower_bound",
    "lower_bound_source", "gap", "solve_seconds", "external_wall_seconds",
    "end_to_end_with_validation_seconds", "input_read_seconds", "enumeration_seconds",
    "support_preprocessing_seconds", "construction_seconds", "optimizer_seconds",
    "master_seconds", "master_optimize_seconds", "subproblem_seconds", "subproblem_primal_seconds",
    "subproblem_dual_phase_seconds", "pricing_seconds", "pareto_phase_seconds",
    "greedy_initial_columns_seconds", "heuristic_budget_used_seconds", "restricted_mip_seconds",
    "cleanup_seconds", "capacity_cut_build_seconds", "capacity_cut_separation_seconds",
    "ordinary_cuts", "pareto_cuts", "pareto_attempts", "capacity_cuts", "root_cuts",
    "outer_iterations", "cg_iterations", "initial_service_columns", "initial_empty_columns",
    "generated_service_columns", "generated_empty_columns", "catalogue_complete",
    "construction_complete", "pricing_complete", "peak_rss_bytes", "peak_rss_gib",
    "cpu_user_seconds", "cpu_system_seconds", "sampled_peak_rss_bytes", "sampled_peak_rss_gib",
    "demand_per_hour", "served_per_hour", "service_fraction", "direct_per_hour", "fleet",
    "busy_drones", "idle_drones", "compartments", "installed_cabinets", "open_bases", "open_lockers",
    "input_manifest_sha256", "result_sha256", "validation_sha256", "operation_tables_state",
)
POINT_TABLES = (
    "bases", "lockers", "nodes", "market_nodes", "orders", "od_assignments", "pools", "arcs",
    "fleet", "costs", "cost_detail", "busy_components", "flight_stages",
)
LARGE_TABLES = ("paths", "order_paths", "links", "task_pairs", "task_singles")


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _inside(root, value):
    candidate = (Path(root) / value).resolve()
    root = Path(root).resolve()
    if candidate != root and root not in candidate.parents:
        raise ValueError("Manifest path escapes the experiment directory")
    return candidate


def _jobs(batch):
    path = batch / "experiment_manifest.json"
    if path.is_file():
        manifest = _json(path)
        jobs = manifest["jobs"]
    else:
        preparation = _json(batch / "preparation_manifest.json")
        manifest = {"status": "data_prepared_experiment_not_frozen", "protocol": {}}
        jobs = [dict(id=f"L{n:02d}_K{k}_{method}", level=n, K=k, method=method,
                     instance=f"instances/L{n:02d}", output=f"runs/L{n:02d}/{method}/K{k}")
                for n in [row["level"] for row in preparation["levels"]]
                for k in (1, 2) for method in METHODS]
    if len({job["id"] for job in jobs}) != len(jobs):
        raise ValueError("Duplicate experiment job IDs")
    for job in jobs:
        if job["method"] not in METHODS or job["K"] not in (1, 2):
            raise ValueError("Unknown scale-experiment method or K")
    return manifest, jobs


def _operation_hashes(path, validation):
    directory = path / "operations"
    if validation.get("operations_sha256") != sha256(directory / "operations.json"):
        raise ValueError("Missing or stale operations.json validation hash")
    certified = validation.get("operations_files_sha256")
    if not isinstance(certified, dict):
        raise ValueError("Missing operation-table hash certificate")
    for name in ("operations.json", *(key + ".csv" for key in POINT_TABLES)):
        source = directory / name
        if not source.is_file() or certified.get(name) != sha256(source):
            raise ValueError(f"Missing or stale operation-table validation hash: {name}")
    for name, expected in certified.items():
        source = _inside(directory, name)
        if not source.is_file() or sha256(source) != expected:
            raise ValueError(f"Operation artifact changed after validation: {name}")


def _global_lower_bound(result, job):
    value = _number(result.get("bound"))
    if value is None:
        return None, "unavailable"
    scope = result.get("bound_scope")
    if scope == "global":
        return value, "saved_global_bound"
    if scope is not None:
        return None, f"excluded_{scope}"
    if (job["method"] in ("full_cplex", "full_gurobi")
            and result.get("method") == "full_path"
            and result.get("catalogue_complete") is True
            and result.get("construction_complete") is True):
        return value, "complete_full_model_solver_bound"
    return None, "global_validity_not_established"


def _resources(attempt, execution):
    usage_path = attempt / "resource_usage.json"
    usage = _json(usage_path) if usage_path.is_file() else execution.get("resource_usage", {})
    actual = {}
    for key in ("peak_rss_bytes", "cpu_user_seconds", "cpu_system_seconds"):
        value = _number(usage.get(key))
        if value is None:
            value = _number(execution.get(key))
        actual[key] = value if value is not None and value >= 0 else None
    actual["peak_rss_gib"] = None if actual["peak_rss_bytes"] is None else actual["peak_rss_bytes"] / 1024**3
    peak = _number(execution.get("sampled_peak_rss_bytes"))
    if peak is not None and peak < 0:
        peak = None
    file = attempt / "resources.csv"
    if file.is_file():
        with file.open(encoding="utf-8-sig", newline="") as stream:
            values = [_number(row.get("rss_bytes")) for row in csv.DictReader(stream)]
        values = [v for v in values if v is not None and v >= 0]
        if values:
            peak = max([*values, *([] if peak is None else [peak])])
    actual.update(sampled_peak_rss_bytes=peak, sampled_peak_rss_gib=None if peak is None else peak / 1024**3)
    return actual


def _capacity_cuts(result):
    statistics = result.get("acceleration_statistics", {})
    for key in ("capacity_cuts", "capacity_cuts_added"):
        value = _number(statistics.get(key))
        if value is not None:
            return value
    records = result.get("capacity_cuts")
    return len(records) if isinstance(records, list) else None


def _read_job(batch, manifest, job, state, inputs):
    row = {key: None for key in COMPARISON}
    row.update({key: job[key] for key in IDENTITY})
    row.update(state="not_run", scheduler_status=state.get("status", "pending"),
               instance=job["instance"], operation_tables_state="not_available")
    record = {"row": row, "path": None, "operations": {}, "result": {}}
    try:
        instance = _inside(batch, job["instance"])
        if instance not in inputs:
            load_standard(instance)  # verifies every current input-file hash
            inputs[instance] = _json(instance / "manifest.json")
        row["input_manifest_sha256"] = sha256(instance / "manifest.json")
        directory = state.get("directory")
        if not directory:
            attempts = state.get("attempts", [])
            directory = attempts[-1].get("directory") if attempts else None
        if not directory:
            record["instance"] = instance
            return record
        attempt = _inside(batch, directory)
        source = _inside(batch, state.get("result_directory", directory))
        expected_output = _inside(batch, job["output"])
        if expected_output not in attempt.parents or (source != attempt and attempt not in source.parents):
            raise ValueError("Attempt/result directory does not belong to the planned job")
        row.update(attempt_directory=str(attempt.relative_to(batch)), result_directory=str(source.relative_to(batch)))
        record.update(path=source, attempt=attempt, instance=instance)
        execution = _json(attempt / "execution.json") if (attempt / "execution.json").is_file() else {}
        row.update(returncode=execution.get("returncode"), stop_reason=execution.get("stop_reason"))
        label, color = METHODS[job["method"]]
        native = _read_run(source, job["id"], label, color, job["K"])
        if native["row"]["state"] != "validated":
            row["state"] = native["row"]["state"]
            return record
        validation = _json(source / "native_validation.json")
        result = native["result"]
        if validation.get("manifest_sha256") != row["input_manifest_sha256"]:
            raise ValueError("Validation certificate is for another current input manifest")
        if native["input_reference"]["manifest"] != inputs[instance]:
            raise ValueError("Saved input reference differs from the current frozen instance")
        if result.get("method") == "fixed_design" or result.get("bound_scope") == "fixed_design":
            raise ValueError("Fixed-design solves do not belong to the scale experiment")
        expected_method, expected_solver = {
            "full_cplex": ("full_path", "CPLEX"),
            "full_gurobi": ("full_path", "Gurobi"),
            "benders_cplex": ("benders_cg", "CPLEX"),
        }[job["method"]]
        if (result.get("method"), result.get("solver")) != (expected_method, expected_solver):
            raise ValueError("Saved method/solver differs from the planned job")
        _operation_hashes(source, validation)
        protocol = manifest.get("protocol", {})
        for key, actual in (("threads_per_job", result.get("threads")),
                            ("seconds_per_job", result.get("budget"))):
            if key in protocol and actual is not None and actual != protocol[key]:
                raise ValueError(f"Run differs from frozen protocol: {key}")
        row.update({key: value for key, value in native["row"].items() if key in COMPARISON and key != "method"})
        row.update(state="validated", operation_tables_state="current_hashes_verified")
        result_timing = result.get("timings", {})
        nested = result_timing.get("subproblem", {})
        statistics = result.get("acceleration_statistics", {})
        generated = result.get("generated_columns", result.get("catalogue") or {})
        lower, source_label = _global_lower_bound(result, job)
        upper = float(result["objective"])
        resources = _resources(attempt, execution)
        partial = bool(result.get("fallback") or result.get("recovered_from_checkpoint") or
                       result.get("execution_failed") or execution.get("stop_reason") or
                       execution.get("returncode") not in (None, 0))
        row.update(
            upper_bound=upper, lower_bound=lower, lower_bound_source=source_label,
            gap=None if lower is None else max(0.0, upper-lower)/max(1.0, abs(upper)),
            completion_kind="partial_validated_incumbent" if partial else "validated_run",
            fallback=bool(result.get("fallback") or result.get("recovered_from_checkpoint") or
                          result.get("incumbent_kind") in ("all_courier_fallback", "all_courier_baseline")),
            external_wall_seconds=_number(execution.get("external_wall_seconds")),
            support_preprocessing_seconds=_number(result.get("support_preprocessing_seconds")),
            master_seconds=sum(float(result_timing.get(key, 0)) for key in
                               ("master_build", "master_optimize", "master_update", "root_master"))
                           if "master_build" in result_timing else None,
            subproblem_seconds=_number(result_timing.get("subproblem_wall")),
            greedy_initial_columns_seconds=_number(result_timing.get("greedy_initial_columns")),
            heuristic_budget_used_seconds=_number(result.get("heuristic_budget_used")),
            restricted_mip_seconds=_number(result_timing.get("restricted_mip")),
            cleanup_seconds=_number(result_timing.get("incumbent_cleanup")),
            capacity_cut_build_seconds=_number(result_timing.get("capacity_cut_build")),
            capacity_cut_separation_seconds=_number(result_timing.get("capacity_cut_separation")),
            capacity_cuts=_capacity_cuts(result), root_cuts=statistics.get("root_cuts"),
            initial_service_columns=result.get("initial_service_columns"), initial_empty_columns=result.get("initial_empty_columns"),
            generated_service_columns=generated.get("service"), generated_empty_columns=generated.get("empty"),
            construction_complete=result.get("construction_complete"),
            **resources,
            result_sha256=sha256(source / "result.json"), validation_sha256=sha256(source / "native_validation.json"),
        )
        # Keep overlapping diagnostic heuristic budgets separate from disjoint
        # phase accounting; they must not be summed into the timing stack.
        record.update(result=result, operations=native["operations"], native=native,
                      phases=native_phase_times(result), nested_timing=nested)
    except (ValueError, KeyError, TypeError, OSError, json.JSONDecodeError) as error:
        # Never leave numeric comparison values populated after a failed gate.
        identity = {key: row.get(key) for key in (*IDENTITY, "instance", "attempt_directory",
                    "result_directory", "scheduler_status", "returncode", "stop_reason")}
        row.clear()
        row.update({key: None for key in COMPARISON})
        row.update(identity, state="invalid_evidence", validation_error=f"{type(error).__name__}: {error}",
                   operation_tables_state="unverified")
    return record


def _progress(record):
    row = record["row"]
    if row["state"] != "validated":
        return [], []
    attempt = record["attempt"]
    source = attempt / "progress.jsonl"
    if not source.is_file():
        return [], [dict(id=row["id"], reason="missing_progress")]
    audit_path = attempt / "artifact_audit.json"
    if audit_path.is_file():
        expected = _json(audit_path).get("artifact_sha256", {}).get("progress.jsonl")
        if expected is not None and expected != sha256(source):
            return [], [dict(id=row["id"], reason="stale_progress_hash")]
    accepted, rejected = [], []
    with source.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                rejected.append(dict(id=row["id"], line=line_number, reason="malformed_event"))
                continue
            if item.get("bound_scope") != "global":
                continue
            if item.get("validated_incumbent", item.get("incumbent_validated")) is not True:
                continue
            elapsed = _number(item.get("elapsed_seconds", item.get("elapsed")))
            upper = _number(item.get("upper_bound", item.get("objective")))
            lower = _number(item.get("lower_bound", item.get("bound")))
            if elapsed is None or elapsed < 0 or upper is None or lower is None or lower > upper + 1e-6*max(1, abs(upper)):
                rejected.append(dict(id=row["id"], line=line_number, reason="invalid_global_bounds"))
                continue
            if lower > row["upper_bound"] + 1e-6*max(1, abs(row["upper_bound"])):
                rejected.append(dict(id=row["id"], line=line_number, reason="bound_exceeds_final_feasible_incumbent"))
                continue
            accepted.append(dict(**{key: row[key] for key in IDENTITY}, state="validated", line=line_number,
                                 elapsed_seconds=elapsed, upper_bound=upper, lower_bound=lower,
                                 gap=max(0., upper-lower)/max(1.,abs(upper)), event=item.get("event"),
                                 evidence="global_scope_and_program_validated_incumbent_from_certified_run"))
    accepted.sort(key=lambda item: (item["elapsed_seconds"], item["line"]))
    return accepted, rejected


def _merge_tables(records, output):
    merged, links, metrics = defaultdict(list), [], []
    for record in records:
        row = record["row"]
        if row["state"] != "validated":
            continue
        prefix = {key: row[key] for key in IDENTITY}
        summary = record["operations"]["summary"]
        for key, value in summary.items():
            if isinstance(value, (str, int, float, bool)) or value is None:
                metrics.append(dict(**prefix, metric=key, value=value))
        for name in POINT_TABLES:
            source = record["path"] / "operations" / (name + ".csv")
            with source.open(encoding="utf-8-sig", newline="") as stream:
                for values in csv.DictReader(stream):
                    merged[name].append({**prefix, **values})
        for name in (*LARGE_TABLES, "metric_dictionary"):
            source = record["path"] / "operations" / (name + ".csv")
            links.append(dict(**prefix, table=name, available=source.is_file(), path=str(source),
                              sha256=sha256(source) if source.is_file() else None))
    tables = output / "operations"
    tables.mkdir(exist_ok=True)
    for name in POINT_TABLES:
        _csv(tables / (name + ".csv"), merged[name], required=IDENTITY)
    _csv(output / "operational_metrics.csv", metrics, required=(*IDENTITY, "metric", "value"))
    _csv(output / "large_table_links.csv", links, required=(*IDENTITY, "table", "available", "path", "sha256"))


def build_scale_report(batch):
    """Write a current report for every planned job; do not launch experiments."""
    batch = Path(batch).resolve()
    manifest, jobs = _jobs(batch)
    states = _json(batch / "state.json").get("jobs", {}) if (batch / "state.json").is_file() else {}
    output = batch / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    # Rebuild owned figures, so an invalidated or missing run cannot leave an
    # old result chart displayed as current. Solver artifacts are read-only.
    figures = output / "figures"
    figures.mkdir(exist_ok=True)
    old_manifest = output / "report_manifest.json"
    if old_manifest.is_file():
        for name in _json(old_manifest).get("figures", []):
            old = _inside(output, name)
            if old.is_file():
                old.unlink()
    inputs = {}
    records = [_read_job(batch, manifest, job, states.get(job["id"], {}), inputs) for job in jobs]
    rows = [record["row"] for record in records]
    phase_rows, progress_rows, progress_rejected = [], [], []
    for record in records:
        row = record["row"]
        if row["state"] != "validated":
            continue
        total = record["phases"]["solve_seconds"]
        for phase, seconds in record["phases"]["phases"].items():
            phase_rows.append(dict(**{key: row[key] for key in IDENTITY}, state="validated", phase=phase,
                                   seconds=seconds, solve_seconds=total,
                                   fraction=None if not total else seconds/total))
        accepted, rejected = _progress(record)
        progress_rows.extend(accepted)
        progress_rejected.extend(rejected)
    _csv(output / "comparison.csv", rows, required=COMPARISON)
    _csv(output / "timing_phases.csv", phase_rows, required=(*IDENTITY, "phase", "seconds", "solve_seconds", "fraction"))
    _csv(output / "global_progress.csv", progress_rows, required=(*IDENTITY, "elapsed_seconds", "upper_bound", "lower_bound", "gap"))
    _csv(output / "progress_exclusions.csv", progress_rejected, required=("id", "line", "reason"))
    _merge_tables(records, output)
    images = plot_preparation(batch, figures)
    images += plot_comparisons(rows, figures)
    images += plot_phases(phase_rows, figures)
    images += plot_progress(progress_rows, figures)
    for record in records:
        if record["row"]["state"] == "validated":
            images += plot_operations(record, figures)
    counts = dict(Counter(row["state"] for row in rows))
    valid = counts.get("validated", 0)
    text = ["# TSL 规模实验：准备与结果核验", "",
            f"计划 {len(rows)} 个任务；当前 {valid} 个结果通过当前文件哈希及独立物理校验。", "",
            "未运行、缺失、失败或校验证书失效的任务保留在比较表中；其未获证实的指标为空。"
            "恢复的可行解标为 partial，可保留上界；下界只采用明确的全局界或完整模型求解器界。", "",
            "[完整比较表](comparison.csv) · [互不重叠的阶段计时](timing_phases.csv) · "
            "[运营指标](operational_metrics.csv) · [大路径表来源](large_table_links.csv)", "",
            _table(("状态", "任务数"), sorted(counts.items())), "",
            "## 实例与地图", "",
            "10 级实例使用相同八个日期、12–13 时需求及 8–24 时点位候选。"
            "正方形切片中心固定；面积为 50n/3 km²，H 候选密度为 0.6/km²。"
            "图使用真实输入的本地投影坐标；L02 保留冻结实例，其他级别独立聚类，点集不保证嵌套。", ""]
    for image in images:
        if image.suffix == ".png":
            text += [f"![{image.stem}]({image.relative_to(output).as_posix()})", ""]
    if valid:
        text += ["## 经过校验的结果", "",
                 _table(("任务", "结束状态", "成本上界 元/h", "全局下界 元/h", "gap", "求解计时 s", "外部计时 s"),
                        [(r["id"], str(r["termination_reason"]), _fmt(r["upper_bound"]), _fmt(r["lower_bound"]),
                          _fmt(r["gap"], percent=True), _fmt(r["solve_seconds"]), _fmt(r["external_wall_seconds"]))
                         for r in rows if r["state"] == "validated"]), ""]
    text += ["## 指标口径", "",
             "求解计时包括读取、路径枚举或支持预处理、模型构造、优化与读解，不含磁盘导出及独立校验。"
             "外部计时来自进程监控，不能与内部阶段重复相加。阶段长表使用互不重叠计时；"
             "heuristic_budget_used 是可能包含子问题工作的诊断预算，不进入堆叠总和。", "",
             "ρ 表示基地立即有无人机可用的概率；busy_drones 是按任务流量×忙碌时间计算的平均忙碌无人机量，"
             "不是单机忙碌率。无人机配置为模型中的连续等效数量。peak_rss_bytes 与 CPU 用户/系统时间优先读取 "
             "resource_usage.json 中操作系统 wait4 的子进程资源统计；缺失时留空。sampled_peak_rss_bytes 单列保留进程树采样峰值，"
             "可能漏掉采样间尖峰，不充当操作系统记录的真实峰值。", "",
             "各点表位于 operations/；列定义从 large_table_links.csv 中的 metric_dictionary 查阅。"
             "完整路径和订单路径表只保留原文件链接，避免复制大量明细。", "",
             "收敛曲线仅使用已通过终局校验的运行中、bound_scope=global 且程序标记可行上界已验证的事件。"
             "事件不逐点重新运行独立运营重构；非全局子问题界和未经验证的求解器候选值不绘制。", "",
             "本报告只读取保存的数据，不启动求解器。", ""]
    (output / "REPORT.md").write_text("\n".join(text), encoding="utf-8")
    audit = dict(schema="tsl-scale-report/1", solver_invoked=False, planned_jobs=len(rows),
                 validated_jobs=valid, states=counts,
                 experiment_manifest_sha256=sha256(batch / "experiment_manifest.json") if (batch / "experiment_manifest.json").is_file() else None,
                 preparation_manifest_sha256=sha256(batch / "preparation_manifest.json"),
                 figures=[str(path.relative_to(output)) for path in images],
                 comparison_sha256=sha256(output / "comparison.csv"),
                 source_code_sha256=sha256(Path(__file__)))
    write_json(output / "report_manifest.json", audit)
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch", type=Path)
    args = parser.parse_args()
    print(json.dumps(build_scale_report(args.batch), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
