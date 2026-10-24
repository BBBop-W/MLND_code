"""Source-backed report for saved C++/CPLEX and Python/Gurobi comparisons.

Only independently validated runs enter numeric comparisons or figures. Missing
or unfinished runs remain visible as missing; this module never starts a solve.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from src.experiments.storage import sha256, write_json
from .style import _grid, _save, _style, plt


RUNS = (
    ("full_cplex", "C++ / CPLEX full", "#31688e"),
    ("full_gurobi", "Python / Gurobi full", "#bd7b27"),
    ("benders_cplex", "C++ / CPLEX Benders", "#7d729a"),
)
COSTS = (
    ("direct_courier", "Direct courier", "#a8b3bc"),
    ("service", "Service tasks", "#31688e"),
    ("fleet", "Fleet", "#bd7b27"),
    ("bases", "Bases", "#4f7570"),
    ("lockers", "Lockers", "#7d729a"),
    ("links", "Links", "#b6ac8c"),
    ("repositioning", "Empty flights", "#d5b27c"),
)
EVIDENCE = (
    ("experiment_protocol.json", "实验协议、版本来源与测试记录"),
    ("input_verification.json", "标准实例读取"),
    ("path_verification.json", "路径与物理资源"),
    ("path_edge_cases.json", "路径边界情况"),
    ("formulation_verification.json", "完整模型矩阵"),
    ("benders_verification.json", "Benders 与割有效性的小算例检查"),
    ("fixed_design_verification.json", "固定设计边界行为"),
    ("native_output_verification.json", "原生 CSV 与详细运营指标核对"),
    ("pareto_diagnostics/diagnosis.json", "Pareto 数值问题的独立诊断"),
)
STATES = {
    "validated": "已保存并通过独立校验",
    "missing_result": "尚无保存结果",
    "missing_operations": "已有结果，尚缺独立运营校验",
    "missing_validation_certificate": "已有运营表，尚缺当前文件的哈希校验证书",
    "validation_failed": "运营校验未通过，排除数值比较",
}
BENDERS_PHASES = (
    ("master", "主问题：构造、优化、更新", "Master", "#31688e"),
    (
        "subproblem_core",
        "子问题：构造、primal、读解、核验、证书",
        "Subproblem core",
        "#4f7570",
    ),
    ("root_master", "根节点主问题", "Root master", "#527c9b"),
    ("restricted_mip", "受限列 MISOCP 与局部分支", "Restricted MISOCP", "#d0a251"),
    ("capacity", "容量网络与割分离", "Capacity cuts", "#af7f9b"),
    ("initial_columns", "初始列与物理支持", "Initial columns / support", "#8d9c85"),
    ("dual", "普通对偶阶段", "Ordinary dual", "#729ba8"),
    ("pricing", "定价", "Pricing", "#b6ac8c"),
    ("add_columns", "增量添加列", "Add columns", "#bd7b27"),
    ("pareto", "Pareto 辅助阶段", "Pareto auxiliary", "#7d729a"),
    ("cleanup", "候选解清理", "Incumbent cleanup", "#aa8066"),
    ("io_enumeration", "输入、路径枚举、输出", "Input, enumeration, output", "#cad1d6"),
    ("unallocated", "未分配余项", "Unallocated remainder", "#737e87"),
)


def _json(path):
    return json.loads(Path(path).read_text())


def _number(value):
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _fmt(value, decimals=3, percent=False):
    number = _number(value)
    if number is None:
        return "—"
    display = 100 * number if percent else number
    if round(display, decimals) == 0:
        display = 0.0
    return f"{display:.{decimals}f}%" if percent else f"{display:,.{decimals}f}"


def _link(label, path):
    return f"[{label}](<{Path(path).resolve()}>)"


def _table(headers, rows):
    def cell(value):
        return str(value).replace("|", "\\|").replace("\n", " ")

    return "\n".join(
        [
            "| " + " | ".join(map(cell, headers)) + " |",
            "| " + " | ".join("---" for _ in headers) + " |",
        ]
        + ["| " + " | ".join(map(cell, row)) + " |" for row in rows]
    )


def _csv(path, rows, required=()):
    fields = list(dict.fromkeys([*required, *(key for row in rows for key in row)]))
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _close(actual, expected, name, path):
    if actual is None or expected is None:
        return
    if not math.isclose(float(actual), float(expected), rel_tol=1e-7, abs_tol=5e-5):
        raise ValueError(f"Saved result and operations disagree for {name}: {path}")


def native_phase_times(result):
    """Disjoint inclusive-budget phase seconds, independent of report layout.

    Callback progress/checkpoint times remain inside their enclosing optimize
    phase. Their diagnostic counters must never be added a second time.
    """
    timing = result.get("timings", {})
    python = (
        result.get("implementation") == "python" or result.get("solver") == "Gurobi"
    )
    if python:
        total = _number(result.get("solve_wall_before_validation"))
        if total is None:
            total = _number(timing.get("total"))
            if total is not None:
                total -= float(timing.get("validation_and_export", 0)) + float(
                    timing.get("save", 0)
                )
    else:
        total = _number(result.get("wall_before_output"))
        if total is None:
            wall = _number(result.get("wall"))
            total = (
                None if wall is None else wall - float(result.get("output_seconds", 0))
            )
    if total is None or total < 0:
        return {"solve_seconds": None, "phases": {}, "available": False}
    phases = {}

    def take(name, value):
        value = float(value)
        if not math.isfinite(value) or value < -1e-6:
            raise ValueError(f"Invalid native timing component {name}: {value}")
        phases[name] = max(0.0, value)

    benders = "master_build" in timing
    if python:
        for key in (
            "read",
            "initial_checkpoint",
            "enumeration",
            "build",
            "optimize",
            "read_solution",
            "dispose",
        ):
            take(key, timing.get(key, 0))
    else:
        take("read", result.get("input_read_seconds", 0))
        take("enumeration", result.get("enumeration_seconds", 0))
        take("support_preprocessing", result.get("support_preprocessing_seconds", 0))
        if benders:
            for key in (
                "greedy_initial_columns",
                "master_build",
                "root_master",
                "master_optimize",
                "master_update",
                "capacity_cut_build",
                "capacity_cut_separation",
                "restricted_mip",
                "incumbent_cleanup",
            ):
                take(key, timing.get(key, 0))
            nested = timing.get("subproblem", {})
            for key, value in nested.items():
                take("subproblem_" + key, value)
            take(
                "subproblem_other",
                float(timing.get("subproblem_wall", 0))
                - sum(float(v) for v in nested.values()),
            )
        else:
            for key in ("build", "optimize"):
                take(key, timing.get(key, 0))
    take("other", total - sum(phases.values()))
    return {
        "solve_seconds": total,
        "phases": phases,
        "available": True,
        "scope": "input_preprocessing_construction_optimization_solution_reading",
        "excluded": "disk_export_and_independent_validation",
        "callback_checkpoint_overlap": "included_in_optimize_do_not_add",
    }


def _timing(result, validation):
    timing = result.get("timings", {})
    python = (
        result.get("implementation") == "python" or result.get("solver") == "Gurobi"
    )
    if python:
        reported = _number(timing.get("total", result.get("wall")))
        validation_export = _number(timing.get("validation_and_export"))
        validation_read = 0.0
        # Python's recorded total already contains its independent validation.
        end_to_end = reported
        run_without_validation = (
            max(0.0, reported - validation_export)
            if reported is not None and validation_export is not None
            else reported
        )
        read = _number(timing.get("read"))
        enumeration = _number(timing.get("enumeration"))
        output = _number(timing.get("save"))
        convention = "python_total_includes_validation_export"
    else:
        reported = _number(result.get("wall"))
        validation_export = _number(validation.get("validation_and_export_seconds"))
        validation_read = _number(validation.get("read_seconds"))
        end_to_end = (
            reported + validation_export + validation_read
            if None not in (reported, validation_export, validation_read)
            else None
        )
        run_without_validation = reported
        read = _number(result.get("input_read_seconds"))
        enumeration = _number(result.get("enumeration_seconds"))
        output = _number(result.get("output_seconds"))
        convention = "native_wall_includes_output_external_validation_added_once"
    nested = timing.get("subproblem", {})
    benders = "master_build" in timing
    construction = (
        float(timing.get("master_build", 0)) + float(nested.get("build", 0))
        if benders
        else _number(timing.get("build"))
    )
    exclusive = (
        [
            timing.get(key)
            for key in (
                "master_build",
                "master_optimize",
                "master_update",
                "root_master",
                "restricted_mip",
                "capacity_cut_build",
                "capacity_cut_separation",
                "greedy_initial_columns",
                "subproblem_wall",
                "incumbent_cleanup",
            )
        ]
        if benders
        else [timing.get("build"), timing.get("optimize")]
    )
    if python:
        exclusive += [timing.get("read_solution"), timing.get("dispose")]
    measured = [read, enumeration, output, *exclusive]
    unallocated = (
        max(0.0, run_without_validation - sum(float(v) for v in measured))
        if run_without_validation is not None and all(v is not None for v in measured)
        else None
    )
    primary = native_phase_times(result)
    return {
        "solve_seconds": primary["solve_seconds"],
        "solve_time_scope": "read_preprocessing_build_solve_read_solution_excluding_export_validation",
        "reported_wall_seconds": reported,
        "run_excluding_external_validation_seconds": run_without_validation,
        "validation_export_seconds": validation_export,
        "validation_read_seconds": validation_read,
        "end_to_end_with_validation_seconds": end_to_end,
        "input_read_seconds": read,
        "enumeration_seconds": enumeration,
        "construction_seconds": construction,
        "optimizer_seconds": None if benders else _number(timing.get("optimize")),
        "master_optimize_seconds": _number(timing.get("master_optimize"))
        if benders
        else None,
        "subproblem_primal_seconds": _number(nested.get("primal")) if benders else None,
        "subproblem_dual_phase_seconds": _number(nested.get("dual"))
        if benders
        else None,
        "pricing_seconds": _number(nested.get("pricing")) if benders else None,
        "pareto_phase_seconds": _number(nested.get("pareto")) if benders else None,
        "output_seconds": output,
        "unallocated_runtime_seconds": unallocated,
        "timing_convention": convention,
        "additional_saved_file_validation_seconds": (
            float(validation["read_seconds"])
            + float(validation["validation_and_export_seconds"])
            if python
            and validation.get("read_seconds") is not None
            and validation.get("validation_and_export_seconds") is not None
            else None
        ),
    }


def _read_run(path, run_id, label, color, k):
    path = Path(path)
    row = {
        "run": run_id,
        "case": f"K{k}",
        "max_orders": k,
        "label": label,
        "run_directory": str(path.resolve()),
        "state": "missing_result",
    }
    record = {"row": row, "path": path, "color": color, "result": {}, "operations": {}}
    result_file = path / "result.json"
    if not result_file.is_file():
        return record
    result = record["result"] = _json(result_file)
    if result.get("max_orders") != k:
        raise ValueError(f"Saved K differs from directory case: {path}")
    row.update(
        solver=result.get("solver"),
        method=result.get("method"),
        termination_reason=result.get("termination_reason"),
        budget_seconds=result.get("budget"),
        target_gap=result.get("requested_gap", result.get("mip_gap_target")),
        budget_scope=result.get(
            "budget_scope", "read_enumeration_model_construction_and_optimization"
        ),
        solver_status="suboptimal"
        if result.get("solver") == "Gurobi" and result.get("solver_status_code") == 13
        else result.get("solver_status"),
        pricing_complete=result.get("pricing_complete"),
        pricing_complete_scope=result.get("pricing_complete_scope"),
        catalogue_complete=result.get("catalogue_complete"),
        state="missing_operations",
        source_result_sha256=sha256(result_file),
    )
    operation_file = path / "operations" / "operations.json"
    if not operation_file.is_file():
        return record
    operations = record["operations"] = _json(operation_file)
    summary = operations["summary"]
    if not summary.get("checks", {}).get("passed"):
        row["state"] = "validation_failed"
        return record
    if not (path / "native_validation.json").is_file():
        row["state"] = "missing_validation_certificate"
        return record
    if summary.get("max_orders") != k:
        raise ValueError(f"Operations K differs from run: {path}")
    if (
        result.get("model_version") != "consecutive"
        or summary.get("model_version") != "consecutive"
    ):
        raise ValueError(f"Unexpected model version: {path}")
    reference = _json(path / "input_reference.json")
    if reference.get("max_orders") != k or "manifest" not in reference:
        raise ValueError(f"Incomplete or conflicting input reference: {path}")
    fingerprint = hashlib.sha256(
        json.dumps(
            reference["manifest"], sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()
    values = reference.get("model", {}).get("values", {})
    if (
        values.get("route_mode") != "direct"
        or values.get("candidate_routes") is not None
    ):
        raise ValueError(
            f"This report expects the verified direct/null route universe: {path}"
        )
    for name in ("objective", "served", "demand", "fleet"):
        _close(result.get(name), summary.get(name), name, path)
    bound = _number(result.get("bound"))
    upper = float(result["objective"])
    if bound is not None and bound > upper + 1e-6 * max(1, abs(upper)):
        raise ValueError(f"Bound exceeds incumbent: {path}")
    relative_gap = (
        max(0.0, upper - bound) / max(1, abs(upper)) if bound is not None else None
    )
    _close(result.get("gap"), relative_gap, "gap", path)
    validation = _json(path / "native_validation.json")
    if not validation.get("passed") or validation.get("max_orders") != k:
        raise ValueError(
            f"Native validation metadata failed or refers to another K: {path}"
        )
    if validation.get("native_manifest_matches") is not True:
        raise ValueError(
            f"Native validation did not certify the input manifest: {path}"
        )
    recorded_manifest = result.get("input_manifest_sha256")
    if recorded_manifest and validation.get("manifest_sha256") != recorded_manifest:
        raise ValueError(f"Native validation refers to another input manifest: {path}")
    if validation.get("summary") != summary:
        raise ValueError(
            f"Operations summary differs from its validation certificate: {path}"
        )
    if validation.get("result_sha256") != sha256(result_file):
        raise ValueError(f"Native validation is stale for result.json: {path}")
    for name in ("solution", "path_flows"):
        if validation.get(name + "_sha256") != sha256(path / (name + ".json")):
            raise ValueError(f"Native validation is stale for {name}.json: {path}")
    # Historical certificates only fingerprinted the numerical solution. New
    # runs bind the detailed point/OD/path tables as well as their summary.
    if "operations_sha256" in validation or "operations_files_sha256" in validation:
        if validation.get("operations_sha256") != sha256(operation_file):
            raise ValueError(f"Native validation is stale for operations.json: {path}")
        operation_hashes = {
            str(filename.relative_to(path / "operations")): sha256(filename)
            for filename in sorted((path / "operations").rglob("*"))
            if filename.is_file()
            and filename.suffix.lower() in {".csv", ".json", ".md"}
        }
        if validation.get("operations_files_sha256") != operation_hashes:
            raise ValueError(
                f"Native validation is stale for operations tables: {path}"
            )
    statistics = result.get("acceleration_statistics", {})
    row.update(
        state="validated",
        input_manifest_fingerprint=fingerprint,
        objective=upper,
        independently_rebuilt_objective=summary["objective"],
        bound=bound,
        gap=relative_gap,
        demand_per_hour=summary["demand"],
        served_per_hour=summary["served"],
        service_fraction=summary["service_fraction"],
        direct_per_hour=summary["direct"],
        fleet=summary["fleet"],
        compartments=summary["slot_total"],
        installed_cabinets=summary["slot_total"] / 8,
        open_bases=summary["opened"]["bases"],
        open_lockers=summary["opened"]["lockers"],
        opened_links=summary.get("opened_links"),
        single_tasks_per_hour=summary["single_tasks"],
        double_tasks_per_hour=summary["double_tasks"],
        empty_tasks_per_hour=summary["empty_tasks"],
        idle_drones=summary.get("idle_drones"),
        busy_drones=summary.get("busy_drones"),
        mean_task_minutes=summary.get("mean_task_minutes"),
        mean_receipt_minutes=summary.get("mean_receipt_minutes"),
        outer_iterations=len(result.get("iterations", [])),
        ordinary_cuts=statistics.get("ordinary_cuts"),
        pareto_cuts=statistics.get("pareto_cuts"),
        pareto_attempts=statistics.get("pareto_attempts"),
        cg_iterations=statistics.get("cg_iterations"),
        source_operations_sha256=sha256(operation_file),
        all_courier_cost=summary.get("all_courier_cost"),
        dominated_by_all_courier=(
            summary.get("all_courier_cost") is not None
            and upper > summary["all_courier_cost"] + 5e-5
        ),
        **_timing(result, validation),
    )
    record["input_reference"] = reference
    return record


def _evidence(root):
    records = []
    for filename, label in EVIDENCE:
        path = root / filename
        value = _json(path) if path.is_file() else None
        records.append(
            {
                "file": filename,
                "label": label,
                "path": path,
                "value": value,
                "passed": (
                    value.get("passed") is True
                    if value is not None
                    and filename
                    not in (
                        "experiment_protocol.json",
                        "pareto_diagnostics/diagnosis.json",
                    )
                    else None
                ),
            }
        )
    return records


def _fixed_pairs(root, fixed):
    pairs = []
    by_run = {record["row"]["run"]: record for record in fixed}
    for k in (1, 2):
        source = root / "full_gurobi" / f"K{k}" / "solution.json"
        pair = {
            "case": f"K{k}",
            "max_orders": k,
            "design_source": str(source),
            "state": "awaiting_both_validated",
            "design_matches": None,
            "both_certified_to_tolerance": False,
        }
        cplex = by_run[f"fixed_parity/K{k}/cplex"]
        gurobi = by_run[f"fixed_parity/K{k}/gurobi"]
        if not all(r["row"]["state"] == "validated" for r in (cplex, gurobi)):
            pairs.append(pair)
            continue
        if not source.is_file():
            pair["state"] = "missing_source_design"
            pairs.append(pair)
            continue
        original = _json(source)
        source_reference = _json(source.parent / "input_reference.json")
        for record in (cplex, gurobi):
            if record["input_reference"]["manifest"] != source_reference.get(
                "manifest"
            ):
                raise ValueError(
                    f"Fixed-design comparison does not use its source instance: {record['path']}"
                )
        names = ("facility", "capacity_level", "assignment", "airline")
        for record in (cplex, gurobi):
            solution = _json(record["path"] / "solution.json")
            for name in names:
                if not np.array_equal(
                    np.asarray(original[name]), np.asarray(solution[name])
                ):
                    raise ValueError(
                        f"Fixed solution differs from the shared Gurobi design in {name}: {record['path']}"
                    )
        first, second = cplex["row"], gurobi["row"]
        continuous_a = float(cplex["result"]["continuous_cost"])
        continuous_b = float(gurobi["result"]["continuous_cost"])
        tolerance = 1e-5 + 1e-7 * max(abs(continuous_a), abs(continuous_b))
        pair.update(
            state="feasible_same_design",
            design_matches=True,
            source_design_sha256=sha256(source),
            cplex_objective=first["objective"],
            gurobi_objective=second["objective"],
            cplex_continuous_cost=continuous_a,
            gurobi_continuous_cost=continuous_b,
            cplex_status=first.get("solver_status"),
            gurobi_status=second.get("solver_status"),
            cplex_bound=first["bound"],
            gurobi_bound=second["bound"],
            cplex_gap=first["gap"],
            gurobi_gap=second["gap"],
            objective_difference_cplex_minus_gurobi=first["objective"]
            - second["objective"],
            continuous_cost_difference=continuous_a - continuous_b,
            numeric_comparison_tolerance=tolerance,
            costs_within_numeric_tolerance=abs(continuous_a - continuous_b)
            <= tolerance,
            both_certified_to_tolerance=all(
                r["gap"] is not None and r["gap"] <= 1e-6 for r in (first, second)
            ),
        )
        pairs.append(pair)
    return pairs


def _cost_bounds(records, directory):
    valid = [r for r in records if r["row"]["state"] == "validated"]
    if not valid:
        return False
    height = max(4.7, 0.65 * len(valid) + 1.8)
    fig, axes = plt.subplots(
        1, 2, figsize=(14.2, height), gridspec_kw={"width_ratios": [1.05, 1.4]}
    )
    y = np.arange(len(valid))
    labels = [
        f"{r['row']['case']} | {r['row']['label']}\nBudget {_fmt(r['row']['budget_seconds'], 0)} s"
        for r in valid
    ]
    for position, record in enumerate(valid):
        row, color = record["row"], record["color"]
        upper, lower = row["objective"], row["bound"]
        if lower is not None:
            axes[0].plot(
                [lower, upper], [position, position], color=color, linewidth=2.5
            )
            axes[0].plot(
                lower,
                position,
                marker="|",
                color=color,
                markersize=12,
                markeredgewidth=2,
            )
        axes[0].plot(upper, position, marker="o", color=color, markersize=6)
        axes[0].annotate(f"  {upper:,.2f}", (upper, position), va="center", fontsize=9)
    axes[0].set_yticks(y, labels)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Total cost (CNY/hour); focused numeric axis")
    axes[0].set_title("Certified lower bound to feasible incumbent")
    axes[0].margins(x=0.2)
    _grid(axes[0])
    left = np.zeros(len(valid))
    for key, label, color in COSTS:
        values = []
        for record in valid:
            components = {
                r["component"]: r["cny_per_hour"]
                for r in record["operations"]["tables"]["costs"]
            }
            if key not in components:
                raise ValueError(f"Missing required cost component: {key}")
            values.append(components[key])
        axes[1].barh(y, values, left=left, color=color, height=0.6, label=label)
        left += np.asarray(values)
    axes[1].set_yticks(y, [r["row"]["case"] for r in valid])
    axes[1].invert_yaxis()
    axes[1].set_xlim(left=0)
    axes[1].set_xlabel("Independently reconstructed cost (CNY/hour)")
    axes[1].set_title("Cost composition of the saved feasible solution")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        ncol=4,
        loc="center",
        bbox_to_anchor=(0.72, 0.09),
        fontsize=9,
        frameon=False,
    )
    _grid(axes[1])
    fig.suptitle("TSL | 2 bases, 20 candidate lockers, 50 OD pairs", fontsize=16)
    fig.text(
        0.5,
        0.005,
        "Intervals are optimization bounds, not confidence intervals. Budgets and final gaps differ by run.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.2, 1, 0.95))
    _save(fig, directory, "cost_bounds")
    return True


def _timing_plot(records, directory):
    valid = [r for r in records if r["row"]["state"] == "validated"]
    if not valid:
        return False
    fig, axes = plt.subplots(1, 2, figsize=(14.2, max(4.7, 0.65 * len(valid) + 1.8)))
    y = np.arange(len(valid))
    for index, record in enumerate(valid):
        row = record["row"]
        runtime = row["run_excluding_external_validation_seconds"]
        check = row["validation_export_seconds"]
        check_read = row["validation_read_seconds"]
        unallocated = row["unallocated_runtime_seconds"]
        if runtime is not None:
            remainder = unallocated if unallocated is not None else 0
            axes[0].barh(
                index,
                max(0.0, runtime - remainder),
                color="#31688e",
                height=0.6,
                label="Measured run phases and output" if index == 0 else None,
            )
            axes[0].barh(
                index,
                remainder,
                left=max(0.0, runtime - remainder),
                color="#a8b3bc",
                height=0.6,
                label="Other runtime, including teardown" if index == 0 else None,
            )
        if runtime is not None and check is not None and check_read is not None:
            axes[0].barh(
                index,
                check + check_read,
                left=runtime,
                color="#bd7b27",
                hatch="//",
                height=0.6,
                label="Independent validation and export" if index == 0 else None,
            )
            axes[0].annotate(
                f" {runtime + check + check_read:.2f}",
                (runtime + check + check_read, index),
                va="center",
                fontsize=9,
            )
        else:
            axes[0].annotate(
                " Validation timing unavailable",
                (runtime or 0, index),
                va="center",
                fontsize=9,
            )
    phases = (
        ("input_read_seconds", "Read", "#a8b3bc"),
        ("enumeration_seconds", "Enumerate", "#4f7570"),
        ("construction_seconds", "Build fields only", "#7d729a"),
    )
    for index, record in enumerate(valid):
        left = 0.0
        for key, label, color in phases:
            value = record["row"][key]
            if value is None:
                continue
            axes[1].barh(
                index,
                value,
                left=left,
                color=color,
                height=0.6,
                label=label if index == 0 else None,
            )
            left += value
        axes[1].annotate(f" {left:.4f}", (left, index), va="center", fontsize=9)
    for ax in axes:
        ax.set_yticks(
            y,
            [f"{r['row']['case']} | {r['row']['label']}" for r in valid]
            if ax is axes[0]
            else [r["row"]["case"] for r in valid],
        )
        ax.invert_yaxis()
        ax.set_xlim(left=0)
        ax.margins(x=0.18)
        ax.set_xlabel("Seconds (separate scales for the two panels)")
        _grid(ax)
    for ax, x in zip(axes, (0.31, 0.79)):
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="center",
            bbox_to_anchor=(x, 0.105),
            fontsize=9,
            frameon=False,
        )
    axes[0].set_title("End-to-end runtime with independent validation")
    axes[1].set_title("Read, enumerate and recorded build fields")
    fig.suptitle("TSL runtime accounting", fontsize=16)
    fig.text(
        0.5,
        0.005,
        "Python validation is counted once. Build fields exclude add_columns and construction inside dual / Pareto phases; see the Benders phase chart.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.24, 1, 0.95))
    _save(fig, directory, "timing")
    return True


def _benders_breakdown(records):
    phases, reasons = [], []
    for record in records:
        row, result = record["row"], record["result"]
        if row["state"] != "validated" or not row["run"].startswith("benders_cplex/"):
            continue
        timing, nested = result["timings"], result["timings"]["subproblem"]
        components = {
            "master": {
                name: timing[name]
                for name in ("master_build", "master_optimize", "master_update")
            },
            "subproblem_core": {
                name: nested[name]
                for name in (
                    "build",
                    "primal",
                    "read_solution",
                    "validation",
                    "certificate",
                )
            },
            **{
                name: {name: nested[name]}
                for name in ("dual", "pricing", "add_columns", "pareto")
            },
            "root_master": {"root_master": timing.get("root_master", 0)},
            "restricted_mip": {"restricted_mip": timing.get("restricted_mip", 0)},
            "capacity": {
                name: timing.get(name, 0)
                for name in ("capacity_cut_build", "capacity_cut_separation")
            },
            "initial_columns": {
                "greedy_initial_columns": timing.get("greedy_initial_columns", 0),
                "support_preprocessing_seconds": result.get(
                    "support_preprocessing_seconds", 0
                ),
            },
            "cleanup": {"incumbent_cleanup": timing.get("incumbent_cleanup", 0)},
            "io_enumeration": {
                name: result[name]
                for name in (
                    "input_read_seconds",
                    "enumeration_seconds",
                    "output_seconds",
                )
            },
        }
        wall = float(result["wall"])
        if wall <= 0 or any(
            _number(value) is None or value < 0
            for fields in components.values()
            for value in fields.values()
        ):
            raise ValueError(f"Invalid Benders timing component: {record['path']}")
        measured = sum(sum(fields.values()) for fields in components.values())
        if measured > wall + 1e-5:
            raise ValueError(
                f"Benders exclusive phase times exceed wall: {record['path']}"
            )
        components["unallocated"] = {
            "wall_minus_exclusive_phases": max(0.0, wall - measured)
        }
        for phase, chinese, _, _ in BENDERS_PHASES:
            fields = components[phase]
            seconds = sum(fields.values())
            phases.append(
                {
                    "run": row["run"],
                    "case": row["case"],
                    "phase": phase,
                    "label": chinese,
                    "seconds": seconds,
                    "share_of_wall": seconds / wall,
                    "wall_seconds": wall,
                    "budget_seconds": row["budget_seconds"],
                    "component_seconds": json.dumps(
                        fields, ensure_ascii=False, sort_keys=True
                    ),
                    "subproblem_wall_reference_only": timing["subproblem_wall"],
                    "independent_operations_validation_included": False,
                }
            )
        counts = Counter()
        for iteration in result.get("iterations", []):
            pareto = iteration.get("pareto", {})
            counts[
                (
                    pareto.get("reason", "unrecorded"),
                    pareto.get("auxiliary_status") or "not_recorded",
                    bool(pareto.get("attempted")),
                    bool(pareto.get("accepted")),
                )
            ] += 1
        for (reason, status, attempted, accepted), count in sorted(counts.items()):
            reasons.append(
                {
                    "run": row["run"],
                    "case": row["case"],
                    "reason": reason,
                    "auxiliary_status": status,
                    "attempted": attempted,
                    "accepted": accepted,
                    "count": count,
                }
            )
        statistics = result.get("acceleration_statistics", {})
        for field, position in (("pareto_attempts", 2), ("pareto_cuts", 3)):
            count = sum(n for key, n in counts.items() if key[position])
            if statistics.get(field) is not None and statistics[field] != count:
                raise ValueError(
                    f"Pareto event counts disagree with {field}: {record['path']}"
                )
    return phases, reasons


def _benders_phase_plot(phases, directory):
    if not phases:
        return False
    runs = list(dict.fromkeys(row["run"] for row in phases))
    lookup = {(row["run"], row["phase"]): row for row in phases}
    fig, ax = plt.subplots(figsize=(12.5, 5.6))
    positions, left = np.arange(len(runs)), np.zeros(len(runs))
    for phase, _, label, color in BENDERS_PHASES:
        values = np.asarray([lookup[run, phase]["seconds"] for run in runs])
        ax.barh(positions, values, left=left, height=0.52, color=color, label=label)
        for index, value in enumerate(values):
            share = lookup[runs[index], phase]["share_of_wall"]
            if share >= 0.08:
                ax.text(
                    left[index] + value / 2,
                    index,
                    f"{value:.2f} s\n{share:.1%}",
                    ha="center",
                    va="center",
                    fontsize=10,
                    color="white",
                )
        left += values
    for index, run in enumerate(runs):
        ax.annotate(
            f" {left[index]:.2f} s", (left[index], index), va="center", fontsize=10
        )
    ax.set_yticks(
        positions,
        [
            f"{lookup[run, 'master']['case']} | budget {lookup[run, 'master']['budget_seconds']:.0f} s"
            for run in runs
        ],
    )
    ax.invert_yaxis()
    ax.set_xlim(0, max(left) * 1.09)
    ax.set_xlabel(
        "Seconds; exclusive recorded phases plus remainder sum to native wall"
    )
    _grid(ax)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        ncol=3,
        loc="center",
        bbox_to_anchor=(0.5, 0.16),
        frameon=False,
        fontsize=9,
    )
    fig.suptitle("Benders | 120-second migration acceptance runs", fontsize=16)
    ax.set_title(
        "Pareto and incremental column construction are shown separately", fontsize=12
    )
    fig.text(
        0.5,
        0.015,
        "External operations validation is excluded. Subproblem wall is a parent timer, never added again.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.29, 1, 0.94))
    _save(fig, directory, "benders_phases")
    return True


def _benders_section(phases, reasons, out):
    if not phases:
        return []
    runs = list(dict.fromkeys(row["run"] for row in phases))
    lookup = {(row["run"], row["phase"]): row for row in phases}
    table = []
    for phase, label, _, _ in BENDERS_PHASES:
        table.append(
            [
                label,
                *(
                    f"{_fmt(lookup[run, phase]['seconds'])} / {_fmt(lookup[run, phase]['share_of_wall'], 2, True)}"
                    for run in runs
                ),
            ]
        )
    lines = [
        "## Benders 阶段耗时与 Pareto 接受情况",
        "",
        "以下为各 120 秒冷启动功能验收。所有比例以原生主入口 wall 为分母；外部运营核验在此不计入。"
        "master 包含构造、优化与更新；子问题核心包含 build、primal、read_solution、validation、certificate。"
        "普通 dual、pricing、增量 add_columns 与 Pareto 阶段单独列出，subproblem_wall 是其父级计时，"
        "不能再与这些子项相加。余项包括尚未拆分的控制流程与局部求解器析构。",
        "",
        _table(["互斥阶段", *[f"{run} 秒 / wall占比" for run in runs]], table),
        "",
        "CSV 的 component_seconds 保留主问题三项与子问题五项各自的原始时间。"
        "前图仅统计 build 字段，不能据此认定全部模型维护只需约一秒：增量添加列也在修改模型。",
        "",
    ]
    k2 = next(
        (
            row
            for row in phases
            if row["case"] == "K2" and row["phase"] == "add_columns"
        ),
        None,
    )
    if k2 is not None:
        lines += [
            f"本次 K=2 的 add_columns 为 {_fmt(k2['seconds'])} 秒，占 wall 的 {_fmt(k2['share_of_wall'], 2, True)}，"
            "已在本图独立显示；该时间没有包含在 build 小柱中。",
            "",
        ]
    lines += [
        f"![Benders阶段耗时]({(out / 'figures/benders_phases.png').resolve()})",
        "",
        _table(
            ["运行", "原因", "辅助问题状态", "实际尝试", "接受", "次数"],
            [
                [
                    r["run"],
                    r["reason"],
                    r["auxiliary_status"],
                    "是" if r["attempted"] else "否",
                    "是" if r["accepted"] else "否",
                    r["count"],
                ]
                for r in reasons
            ],
        ),
        "",
        "auxiliary_not_optimal 表示辅助问题没有给出最优证书；NumBest 为数值终止状态，"
        "AbortTimeLim/time_limit 为限时终止。certificate_check_failed 表示候选未通过独立割证书检查，"
        "即使辅助求解器显示 Optimal 也不接受该割。disabled 表示该轮未尝试，不能计为一次被拒绝的强化割。",
        "",
    ]
    if sum(r["count"] for r in reasons if r["accepted"]) == 0:
        lines += [
            "当前已完成的 TSL 验收运行中，Pareto 强化割接受数为 0。生成与核验流程已经迁移，"
            "但这组试验尚未取得强化割加速效果；表中的耗时不能被描述为已实现的加速收益。",
            "",
        ]
    return lines


def _verification_text(evidence):
    rows = []
    for item in evidence:
        value = item["value"]
        detail = "尚无校验文件"
        if value is not None:
            detail = "通过" if item["passed"] else "未通过或未声明通过"
            if item["file"] == "experiment_protocol.json":
                detail = "已记录预算、版本差异、运行环境及测试摘要；不是额外的数学证明"
            elif item["file"] == "pareto_diagnostics/diagnosis.json":
                detail = f"独立探索性诊断；{len(value.get('variants', []))} 种设置；未据此修改正式求解实现"
            elif item["file"] == "path_verification.json":
                values = value.get("cases", [])
                detail += f"；{len(values)} 个 K 用例；资源最大绝对差 {_fmt(max((r.get('max_absolute_difference', 0) for r in values), default=0), 12)}"
            elif item["file"] == "formulation_verification.json":
                reports = value.get("reports", [])
                difference = max(
                    (
                        max(
                            r.get("max_variable_numeric_difference", 0),
                            r.get("max_row_coefficient_difference", 0),
                        )
                        for r in reports
                    ),
                    default=0,
                )
                detail += (
                    f"；{len(reports)} 个模型对照；最大数值差 {_fmt(difference, 12)}"
                )
            elif item["file"] == "benders_verification.json":
                comparisons = value.get("comparisons", [])
                detail += f"；{len(comparisons)} 个小算例/策略检查；与 TSL 大算例的收敛结论分别报告"
            elif item["file"] == "fixed_design_verification.json":
                detail += f"；{len(value.get('cases', []))} 个固定设计边界检查"
        rows.append(
            [
                item["label"],
                detail,
                _link(item["file"], item["path"]) if value is not None else "—",
            ]
        )
    return _table(["核对对象", "保存证据", "原始记录"], rows)


def _report(
    root,
    out,
    records,
    evidence,
    fixed,
    fixed_pairs,
    warm,
    figures,
    benders_phases,
    pareto_reasons,
):
    valid = [r for r in records if r["row"]["state"] == "validated"]
    pending = [r for r in records if r["row"]["state"] != "validated"]
    matrix_passed = next(
        (r["passed"] for r in evidence if r["file"] == "formulation_verification.json"),
        None,
    )
    protocol = next(
        (r["value"] for r in evidence if r["file"] == "experiment_protocol.json"),
        None,
    )
    lines = [
        "# TSL 20H：C++ / CPLEX 与 Python / Gurobi 的模型与求解对照",
        "",
        f"固定算例为 2 个基地、20 个柜候选点和 50 个 OD；目前读取到 {len(valid)}/6 个通过独立运营校验的正式结果。"
        "完整模型与 Benders 各自的实际预算、上下界和停止原因见下表。",
    ]
    lines += [
        "",
        "本轮完整模型按每例 600 秒运行；Benders 按每例 120 秒冷启动，用于迁移功能验收并启用既定加速策略。"
        "这不是同预算的算法性能排名；若实际源文件预算不同，以各行保存记录为准。",
    ]
    if protocol is not None:
        lines += [
            "",
            "这些时间是本机实现验收中的观测：主 MIP 求解按顺序进行，但部分核验与编译并行，"
            "因此不是隔离环境下的性能基准。当前二进制、源码哈希与历史运行版本差异见 "
            + _link("实验协议", root / "experiment_protocol.json")
            + "。",
        ]
    if matrix_passed:
        lines += [
            "",
            "已保存的逐元素检查确认 C++ 与 Python 的完整模型矩阵、变量界、目标系数和锥约束一致。"
            "限时或容差停止得到的可行解可以不同；不能把这些成本差异解释成模型不同，也不能将 gap 尚未归零的结果称为精确最优解。",
        ]
    full = {r["row"]["run"]: r["row"] for r in valid}
    if "full_cplex/K1" in full and "full_gurobi/K1" in full:
        native, python = full["full_cplex/K1"], full["full_gurobi/K1"]
        lines += [
            "",
            f"K=1 的原生构造时间为 {_fmt(native['construction_seconds'], 6)} 秒，"
            f"Python 构造时间为 {_fmt(python['construction_seconds'], 6)} 秒；"
            f"包含一次独立核验的完整流程分别为 {_fmt(native['end_to_end_with_validation_seconds'])} 秒和 "
            f"{_fmt(python['end_to_end_with_validation_seconds'])} 秒。构造加速不等于完整求解与验证流程更快。",
        ]
    unresolved_k2 = [
        r["row"]
        for r in valid
        if r["row"]["max_orders"] == 2
        and r["row"]["run"].startswith("full_")
        and (r["row"]["gap"] is None or r["row"]["gap"] > 0.01)
    ]
    if unresolved_k2:
        lines += [
            "",
            "K=2 冷启动完整模型仍有明显的最优性缺口，当前服务量和设施配置取决于搜索找到的可行解。"
            "不能用这些尚未充分收敛的解与 K=1 的成本差，判断 K=2 运营政策本身更优或更差。",
        ]
    if pending:
        lines += [
            "",
            "尚缺的结果保留为未完成状态，不填零、不进入图表：",
            "",
            _table(
                ["计划运行", "状态"],
                [[r["row"]["run"], STATES[r["row"]["state"]]] for r in pending],
            ),
        ]
    lines += [
        "",
        "## 成本、服务与设施配置",
        "",
        "UB 为已经通过独立校验的可行解成本；LB 为求解器或有效 Benders 割给出的下界。"
        "相对 gap = (UB−LB)/max(1,|UB|)。服务量指无人机系统实际完成的订单流量，单位为单/小时。",
        "停止状态含义：OptimalTol/gap_limit，以及带非零 MIP gap 的 optimal，表示达到所设容差；"
        "time_limit/AbortTimeLim 表示预算耗尽；suboptimal 表示求解器未给出最优证明。"
        "独立运营校验通过只证明保存解满足模型与运营核对，不自动证明最优性。缺失 LB/gap 保留为“—”。",
        "",
    ]
    for k in (1, 2):
        selected = [r for r in valid if r["row"]["max_orders"] == k]
        lines += [f"### K={k}", ""]
        if not selected:
            lines += ["尚无通过独立运营校验的结果。", ""]
            continue
        lines += [
            _table(
                [
                    "实现",
                    "预算秒",
                    "UB 元/小时",
                    "LB 元/小时",
                    "gap",
                    "服务单/小时",
                    "平均无人机数",
                    "柜点/格口",
                    "停止原因",
                ],
                [
                    [
                        r["row"]["label"],
                        _fmt(r["row"]["budget_seconds"], 0),
                        _fmt(r["row"]["objective"]),
                        _fmt(r["row"]["bound"]),
                        _fmt(r["row"]["gap"], 4, True),
                        _fmt(r["row"]["served_per_hour"]),
                        _fmt(r["row"]["fleet"]),
                        f"{r['row']['open_lockers']}/{_fmt(r['row']['compartments'], 0)}",
                        r["row"]["termination_reason"],
                    ]
                    for r in selected
                ],
            ),
            "",
        ]
        full = [r for r in selected if r["row"]["run"].startswith("full_")]
        if len(full) == 2:
            costs = [r["row"]["objective"] for r in full]
            lines += [
                f"两个完整模型的当前可行解成本相差 {_fmt(abs(costs[0] - costs[1]), 6)} 元/小时。"
            ]
            if all(r["row"]["bound"] is not None for r in full):
                low, high = max(r["row"]["bound"] for r in full), min(costs)
                lines += [
                    f"两者证书合并后，最优值位于 [{_fmt(low, 6)}, {_fmt(high, 6)}] 元/小时。"
                    if low <= high + 1e-5
                    else "两者的已保存上下界不相容，需要复核，不能形成一致最优值区间。"
                ]
            lines += [""]
        dominated = [r for r in selected if r["row"].get("dominated_by_all_courier")]
        for record in dominated:
            row = record["row"]
            lines += [
                f"{row['run']} 保存的是求解器实际返回的可行候选：成本 {_fmt(row['objective'], 6)} 元/小时，"
                f"无人机服务量 {_fmt(row['served_per_hour'], 6)} 单/小时。"
                f"已知零设施全骑手基准 {_fmt(row['all_courier_cost'], 6)} 元/小时更低，差额 "
                f"{_fmt(row['objective'] - row['all_courier_cost'], 6)} 元/小时。"
                "该历史候选含空开设施费用，不是回退基准；原始结果保留，不能将它称为优于基准的设计。",
                "",
            ]
    if figures:
        lines += [
            f"![成本上下界及构成]({(out / 'figures/cost_bounds.png').resolve()})",
            "",
        ]
    lines += [
        "## 两个基地的流量、可用率与忙碌资源",
        "",
        "ρ 是基地可立即派出无人机的可用率；idle 是平均空闲无人机数。"
        "busy 为该基地发出的任务流量乘相应平均任务时间，按出发基地记账。"
        "idle+busy 是资源归属统计，模型没有逐基地整数机队决策。",
        "",
        _table(
            [
                "运行",
                "基地索引",
                "ρ",
                "idle",
                "busy",
                "idle+busy",
                "出发任务/小时",
                "返回任务/小时",
                "平均服务任务分钟",
            ],
            [
                [
                    r["row"]["run"],
                    b["base"],
                    _fmt(b["availability"], 6),
                    _fmt(b["idle"]),
                    _fmt(b.get("busy_drones")),
                    _fmt(b.get("attributed_total_drones")),
                    _fmt(b["outflow"]),
                    _fmt(b["inflow"]),
                    _fmt(b.get("mean_service_task_minutes")),
                ]
                for r in valid
                for b in r["operations"]["tables"]["bases"]
            ],
        ),
        "",
        "## 计时口径",
        "",
        "原生 wall 包含读入、路径枚举、构造、求解与原生输出；独立 Python 运营核验在其后运行，单独列出并仅加一次。"
        "Python 参考运行的 timings.total 已包含自身的独立校验和导出，完整流程时间直接使用该值，不再次叠加核验。"
        "不同预算的 Benders 与完整模型不能只比较一个耗时或一个 UB 就判断算法优劣。",
        "",
        _table(
            [
                "运行",
                "预算秒",
                "记录 wall 秒",
                "独立核验及导出秒",
                "核验读入秒",
                "完整流程秒",
                "构造秒",
                "完整模型优化秒",
                "附加文件验收秒",
            ],
            [
                [
                    r["row"]["run"],
                    _fmt(r["row"]["budget_seconds"], 0),
                    _fmt(r["row"]["reported_wall_seconds"]),
                    _fmt(r["row"]["validation_export_seconds"]),
                    _fmt(r["row"]["validation_read_seconds"]),
                    _fmt(r["row"]["end_to_end_with_validation_seconds"]),
                    _fmt(r["row"]["construction_seconds"], 6),
                    _fmt(r["row"]["optimizer_seconds"]),
                    _fmt(r["row"]["additional_saved_file_validation_seconds"]),
                ]
                for r in valid
            ],
        ),
        "",
        "Python 各行的附加文件验收是运行结束后再次读取保存 JSON、绑定 SHA-256 并重建运营指标的检查；"
        "它在原 timings.total 之外单列，不再加入上表完整流程或图中。原生各行的外部核验已经计入完整流程，"
        "故附加栏留空。",
        "",
        "Benders 的主问题优化、子问题 primal、dual、pricing 与 Pareto 阶段时间分别保存在 comparison.csv；"
        "dual/Pareto 是完整阶段计时，不能当成单纯优化器调用时间。两个求解器的预处理、启发式和树搜索不同，"
        "C++ 构造更快不保证 CPLEX 的搜索时间一定比 Gurobi 更短。",
        "Benders 图中的构造时间仅为已单独记录的 master_build + subproblem.build；"
        "dual/Pareto 阶段内部尚未拆出的构造时间不在其中，不能称为 Benders 总构造时间。",
        "未分配到已测阶段的 wall 余项保留在图中，包括局部求解器析构、解提取与记账等；不会强行归入优化或构造。"
        "Benders 的 pricing_complete_scope=last_processed_subproblem 只说明最近处理的子问题定价状态，不能当成全局最优收敛。",
        "",
    ]
    old_scope = [
        r["row"]["run"]
        for r in valid
        if r["row"].get("budget_scope") == "model_construction_and_optimization"
    ]
    if old_scope:
        lines += [
            "历史预算口径例外："
            + "、".join(old_scope)
            + " 使用旧驱动，600 秒预算只扣构造与优化，读入和枚举另计；这一区别原样保存在 comparison.csv。"
            "其已记录的完整流程时间仍包含实际读入、枚举与输出。",
            "",
        ]
    if figures:
        lines += [f"![运行与构造时间]({(out / 'figures/timing.png').resolve()})", ""]
    lines += _benders_section(benders_phases, pareto_reasons, out)
    diagnosis = next(
        (
            r
            for r in evidence
            if r["file"] == "pareto_diagnostics/diagnosis.json"
            and r["value"] is not None
        ),
        None,
    )
    if diagnosis is not None:
        variants = diagnosis["value"].get("variants", [])
        lines += [
            "另做了独立的 K=1 数值诊断：每种设置只观察前三轮，不属于上述正式性能比较。"
            "正式源码、二进制与证书容差未改变，原始记录见 "
            + _link("Pareto 诊断", diagnosis["path"])
            + "。",
            "",
            _table(
                ["辅助问题数值设置", "尝试数", "接受数", "辅助状态", "拒绝原因"],
                [
                    [
                        r["name"],
                        r["attempts"],
                        r["accepted"],
                        "; ".join(
                            f"{key}: {count}"
                            for key, count in r.get("statuses", {}).items()
                        ),
                        "; ".join(
                            f"{key}: {count}"
                            for key, count in r.get("rejection_reasons", {}).items()
                        ),
                    ]
                    for r in variants
                ],
            ),
            "",
            "四种设置均未产生可接受的强化割。等价缩放将辅助求解状态从 NumBest 改为 Optimal，"
            "但原始单位下的证书检查仍未通过。诊断中的前三轮主问题设计不同，却都没有开启航线，"
            "因此子问题成本均退化为全骑手成本；这些轮次中的重复下界来自较弱的航线—服务联动松弛，"
            "不能解释成同一设计被缓存反复使用。分配允许零流量，不能为了消除这种现象强制分配带来正服务。",
            "",
        ]
    lines += [
        "## 实例、模型与算法一致性证据",
        "",
        _verification_text(evidence),
        "",
        "本报告的路径范围为 direct，candidate_routes=null：每一飞行段采用允许的直达航线，并完整枚举该范围内的服务路径。"
        "本次 TSL 比较使用 aggregate 连续接单流量约束。矩阵一致性是模型等价的直接证据；"
        "小算例的割核验支撑 Benders 实现检查，不能替代大算例的最终 gap。",
        "",
    ]
    if protocol is not None:
        tests = protocol.get("tests", {})
        suite = tests.get("python_full_suite", {})
        lines += [
            "实验过程中包装器和验收接口有更新，不能把所有记录描述为完全同一可执行版本："
            "历史 full_cplex 的构建记录保存在 full_run_build.json；当前包装器增加固定设计边界检查，"
            "并在非固定设计运行中保留全骑手基准与求解器候选中更好的可行解。历史 K=2 原始结果没有被覆盖。",
            "Gurobi K=2 求解期间，磁盘上的驱动仅更新状态输出、覆盖保护与退出码检查；"
            "该运行结束时的驱动哈希不一定等于启动时加载的代码。数学模型、路径源码及冻结输入未变，"
            "后续运行改为求解前记录源码哈希。具体差异与当前哈希均以实验协议为准。",
            f"实验协议记录 Python 全套回归 {suite.get('passed', '未记录')} 项通过，"
            f"原生接口专项 {tests.get('native_interface_tests', '未记录')} 项通过，"
            f"冻结文件 {tests.get('frozen_files_verified', '未记录')} 个复核未变；"
            "这些是验收记录，与上述逐元素数学系数核对分别提供依据。",
            "",
        ]
    lines += ["## 固定设计交叉复算", ""]
    if fixed:
        lines += [
            _table(
                ["保存目录", "状态", "求解器", "UB 元/小时", "gap"],
                [
                    [
                        _link(r["path"].relative_to(root).as_posix(), r["path"]),
                        STATES[r["row"]["state"]],
                        r["row"].get("solver", "—"),
                        _fmt(r["row"].get("objective"), 8),
                        _fmt(r["row"].get("gap"), 6, True),
                    ]
                    for r in fixed
                ],
            ),
            "",
            "每个 K 的两个求解器均固定该 K 的 full_gurobi/solution.json 中全部选址、柜容量、分配及航线二元决策，"
            "各自连续求解预算为 120 秒。报告直接核对这些决策数组与共同来源文件相同。",
            "",
        ]
        completed = [
            pair for pair in fixed_pairs if pair["state"] == "feasible_same_design"
        ]
        if completed:
            lines += [
                _table(
                    [
                        "K",
                        "共同设计",
                        "C−G 成本差 元/小时",
                        "CPLEX 状态/gap",
                        "Gurobi 状态/gap",
                        "能作出的结论",
                    ],
                    [
                        [
                            pair["case"],
                            _link("Gurobi 完整模型设计", pair["design_source"]),
                            _fmt(pair["objective_difference_cplex_minus_gurobi"], 8),
                            f"{pair['cplex_status']} / {_fmt(pair['cplex_gap'], 6, True)}",
                            f"{pair['gurobi_status']} / {_fmt(pair['gurobi_gap'], 6, True)}",
                            (
                                "同设计，连续解成本数值接近"
                                if pair["costs_within_numeric_tolerance"]
                                else "同设计，可行连续解成本仍有差异"
                            )
                            + (
                                "；两者均有小 gap 证书"
                                if pair["both_certified_to_tolerance"]
                                else "；未同时获得最优性证书"
                            ),
                        ]
                        for pair in completed
                    ],
                ),
                "",
                "数值接近使用绝对容差 1e−5 加连续成本规模的 1e−7，原始差值保留在 fixed_parity.csv。"
                "SUBOPTIMAL（包括原始状态码 13）可以产生已独立核验的可行解，但缺失的 bound/gap 不能填为零，也不证明最优。",
                "",
            ]
    else:
        lines += ["尚无已保存的固定设计交叉复算结果；不据此宣称已经完成该项验证。", ""]
    lines += [
        "## 热启动补充试验（不并入冷启动比较）",
        "",
        "warm_cplex/K2 使用 full_gurobi/K2/solution.json 的二元设计作为部分 MIP 起点，预算为 120 秒。"
        "它允许重新优化设计，与固定设计连续复算不同，也不与 600 秒冷启动视作同预算比较。",
        "",
    ]
    if warm["row"]["state"] == "validated":
        row = warm["row"]
        lines += [
            _table(
                [
                    "运行",
                    "预算秒",
                    "UB 元/小时",
                    "LB 元/小时",
                    "gap",
                    "服务单/小时",
                    "平均机队",
                    "停止原因",
                ],
                [
                    [
                        row["run"],
                        _fmt(row["budget_seconds"], 0),
                        _fmt(row["objective"]),
                        _fmt(row["bound"]),
                        _fmt(row["gap"], 4, True),
                        _fmt(row["served_per_hour"]),
                        _fmt(row["fleet"]),
                        row["termination_reason"],
                    ]
                ],
            ),
            "",
            _link("热启动详细运营报告", warm["path"] / "operations/OPERATIONS.md"),
            "",
        ]
    else:
        lines += [STATES[warm["row"]["state"]] + "；未进入正式六次冷启动的图表。", ""]
    lines += [
        "## 详细指标、单位与原始结果",
        "",
        "成本为元/小时，订单与任务流量为每小时；机队数是稳态平均资源量，可为非整数。"
        "柜子每整柜 8 格，open_lockers 是开设柜点数而不是整柜台数；compartments 是实际格数，整柜台数=compartments/8。"
        "mean_occupancy 包含预留与占用，mean_threshold 是满足溢出概率要求的泊松均值上限。"
        "K=2 的双订单任务贡献两单服务量，但只计一个任务；窗口已与首程载货飞行重叠，不重复加到忙碌时间。",
        "",
        "逐点与逐 OD 表保留全部原始详细指标，没有只截取正流量点：",
        "",
        "- " + _link("综合对照 comparison.csv", out / "comparison.csv"),
        "- " + _link("成本分项 cost_components.csv", out / "cost_components.csv"),
        "- " + _link("各基地 base_metrics.csv", out / "base_metrics.csv"),
        "- "
        + _link("全部 20 个柜候选点 locker_metrics.csv", out / "locker_metrics.csv"),
        "- " + _link("全部 50 个 OD od_metrics.csv", out / "od_metrics.csv"),
        "- " + _link("固定设计交叉复算 fixed_parity.csv", out / "fixed_parity.csv"),
        "- " + _link("Benders 互斥阶段 benders_phases.csv", out / "benders_phases.csv"),
        "- " + _link("Pareto 原因 pareto_reasons.csv", out / "pareto_reasons.csv"),
        "",
    ]
    for record in valid:
        path = record["path"]
        lines += [
            f"- {record['row']['run']}："
            + "；".join(
                [
                    _link("详细运营报告", path / "operations/OPERATIONS.md"),
                    _link("求解概览", path / "result.json"),
                    _link(
                        "完整运营数据与指标字典", path / "operations/operations.json"
                    ),
                    _link("原始解", path / "solution.json"),
                ]
            )
        ]
    lines += [
        "",
        "该报告仅汇总保存结果，没有启动新求解；缺失项和求解停止状态按源文件如实保留。",
        "",
    ]
    return "\n".join(lines)


def build_native_report(experiment_root, output=None):
    root = Path(experiment_root).resolve()
    out = Path(output).resolve() if output is not None else root / "report"
    records = [
        _read_run(root / folder / f"K{k}", f"{folder}/K{k}", label, color, k)
        for k in (1, 2)
        for folder, label, color in RUNS
    ]
    valid = [r for r in records if r["row"]["state"] == "validated"]
    fingerprints = {r["row"]["input_manifest_fingerprint"] for r in valid}
    if len(fingerprints) > 1:
        raise ValueError(
            "Compared runs do not share the same standard instance manifest"
        )
    evidence = _evidence(root)
    fixed = [
        _read_run(
            root / "fixed_parity" / f"K{k}" / engine,
            f"fixed_parity/K{k}/{engine}",
            "Fixed-design cross-check",
            "#4f7570",
            k,
        )
        for k in (1, 2)
        for engine in ("cplex", "gurobi")
    ]
    fixed_pairs = _fixed_pairs(root, fixed)
    warm = _read_run(
        root / "warm_cplex/K2", "warm_cplex/K2", "C++ / CPLEX warm start", "#4f7570", 2
    )
    all_fingerprints = {
        r["row"]["input_manifest_fingerprint"]
        for r in [*records, *fixed, warm]
        if r["row"]["state"] == "validated"
    }
    if len(all_fingerprints) > 1:
        raise ValueError(
            "Formal, fixed-design and warm-start runs do not share the same instance"
        )
    out.mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(exist_ok=True)
    _csv(out / "comparison.csv", [r["row"] for r in records], ("run", "case", "state"))
    _csv(out / "fixed_parity.csv", fixed_pairs, ("case", "state", "design_source"))
    _csv(out / "warm_start.csv", [warm["row"]], ("run", "case", "state"))
    benders_phases, pareto_reasons = _benders_breakdown(records)
    _csv(
        out / "benders_phases.csv",
        benders_phases,
        ("run", "case", "phase", "seconds", "share_of_wall"),
    )
    _csv(
        out / "pareto_reasons.csv",
        pareto_reasons,
        ("run", "case", "reason", "auxiliary_status", "count"),
    )
    for source, target in (
        ("costs", "cost_components"),
        ("bases", "base_metrics"),
        ("lockers", "locker_metrics"),
        ("orders", "od_metrics"),
    ):
        rows = [
            {
                "run": r["row"]["run"],
                "case": r["row"]["case"],
                "solver": r["row"]["solver"],
                "method": r["row"]["method"],
                **item,
            }
            for r in valid
            for item in r["operations"]["tables"][source]
        ]
        _csv(out / (target + ".csv"), rows, ("run", "case", "solver", "method"))
    _style()
    plt.rcParams["font.family"] = "DejaVu Sans"
    figures = _cost_bounds(records, out / "figures")
    _timing_plot(records, out / "figures")
    _benders_phase_plot(benders_phases, out / "figures")
    (out / "REPORT.md").write_text(
        _report(
            root,
            out,
            records,
            evidence,
            fixed,
            fixed_pairs,
            warm,
            figures,
            benders_phases,
            pareto_reasons,
        ),
        encoding="utf-8",
    )
    manifest = {
        "generator_source_sha256": sha256(Path(__file__).resolve()),
        "expected_runs": 6,
        "validated_runs": len(valid),
        "all_expected_runs_present_and_validated": len(valid) == 6,
        "rows": [r["row"] for r in records],
        "evidence": [
            {
                "file": r["file"],
                "passed": r["passed"],
                "sha256": sha256(r["path"]) if r["path"].is_file() else None,
            }
            for r in evidence
        ],
        "solver_started": False,
        "fixed_design_pairs": fixed_pairs,
        "warm_start": warm["row"],
    }
    write_json(out / "report_sources.json", manifest)
    return {
        "report": str(out / "REPORT.md"),
        "validated_runs": len(valid),
        "missing_or_unvalidated": [
            r["row"]["run"] for r in records if r["row"]["state"] != "validated"
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_root", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            build_native_report(args.experiment_root, args.out),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
