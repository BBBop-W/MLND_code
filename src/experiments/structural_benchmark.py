"""Frozen old/new structural strengthening comparison for both Benders forms.

Preparing never starts a solver. The established guarded scheduler performs
independent physical validation and retains failed attempts and checkpoints.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import csv
import json
from pathlib import Path
import shutil
import sys
import threading

from src.data.standard import load_standard
from src.experiments import conditional_benchmark_20261022 as base
from src.experiments.od_airline_benchmark_20261022 import COMMON_SOLVER_ARGS
from src.experiments.storage import write_json

ROOT = base.ROOT
MODULE = "src.experiments.structural_benchmark"
DEFAULT_ROOT = ROOT / "outputs/structural_strengthening_20261022"
DEFAULT_BATCH = DEFAULT_ROOT / "comparison"
FLAGS = ("assignment_incident_clique", "shared_rho_joint", "queue_perspective")
_BASE_COMMAND = base.command_for
_BASE_RECORD = base._record
_BASE_GUARD = base.guarded_execute
_BASE_AUDIT = base.audit_worker


def jobs_for(origins):
    return [dict(id=f"{origin['id']}_K{k}_{form}_{version}",
                 variant=f"{form}_{version}", version=version,
                 instance=origin["instance"], K=k, method="benders_" + form,
                 structural_strengthening=version == "new",
                 output=f"runs/{origin['id']}/{form}_{version}/K{k}")
            for origin in origins for k in (1, 2)
            for form in ("lifted", "conditional") for version in ("old", "new")]


def command_for(batch, manifest, job, attempt):
    command = _BASE_COMMAND(batch, manifest, job, attempt)
    command[0] = str(Path(batch) / manifest["binaries"][job["version"]]["path"])
    command[command.index("--k") + 1] = str(job["K"])
    command += manifest["protocol"]["common_solver_args"]
    if job["structural_strengthening"]:
        command.append("--structural-strengthening")
    return command


def prepare(batch, instances, *, new_binary=None, baseline_freeze=None, seconds=600,
            threads=1, parallel=3, gap=.005, seed=43, hard_wall_seconds=720):
    if threads != 1 or parallel > 3:
        raise ValueError("This comparison requires one solver thread and at most three concurrent jobs")
    baseline_file = Path(baseline_freeze or DEFAULT_ROOT / "preparation/baseline_freeze.json").resolve()
    baseline = base.read(baseline_file)
    old_binary = Path(baseline["frozen_binary"])
    if base.sha256(old_binary) != baseline["binary_sha256"]:
        raise ValueError("Previously frozen baseline executable changed")
    for instance in instances:
        for k in (1, 2):
            load_standard(instance, case=f"K{k}")
    base.prepare(batch, instances, binary=new_binary, seconds=seconds, threads=threads,
                 parallel=parallel, gap=gap, seed=seed, hard_wall_seconds=hard_wall_seconds)
    batch = Path(batch).resolve()
    manifest = base.read(batch / "experiment_manifest.json")
    old_relative = "frozen/bin/mlnd_cpp_baseline"
    baseline_relative = "frozen/baseline_freeze.json"
    shutil.copy2(old_binary, batch / old_relative)
    shutil.copy2(baseline_file, batch / baseline_relative)
    manifest["schema"] = "structural-benders-comparison/1"
    manifest["binaries"] = {
        "old": dict(path=old_relative, sha256=base.sha256(batch / old_relative), source_hashes=baseline["source_hashes"]),
        "new": dict(path="frozen/bin/mlnd_cpp", sha256=base.sha256(batch / "frozen/bin/mlnd_cpp"),
                    source_hashes={name.removeprefix("frozen/"): value for name, value in manifest["frozen_sha256"].items() if name.startswith("frozen/cpp/")}),
    }
    for relative in (old_relative, baseline_relative):
        manifest["frozen_sha256"][relative] = base.sha256(batch / relative)
    manifest["baseline_freeze_source"] = str(baseline_file)
    manifest["baseline_freeze_sha256"] = base.sha256(baseline_file)
    manifest["jobs"] = jobs_for(manifest["input_origins"])
    manifest["protocol"].update(
        comparison="old/new structural strengthening within lifted/conditional Benders, K=1/2",
        max_orders=[1, 2], common_solver_args=list(COMMON_SOLVER_ARGS),
        changed_solver_args=["--structural-strengthening"],
        structural_features=list(FLAGS), od_airline_bounds=False,
        root_initialization="inside each job budget; no external root certificate or warm start",
        input_generation=False, same_binary_for_all_variants=False,
        baseline_source_hash_scope="source snapshot observed when the old executable was frozen",
    )
    for origin in manifest["input_origins"]:
        data = load_standard(batch / origin["instance"], case="K2")
        origin["all_courier_cost"] = sum(float(a) * float(b) for a, b in zip(data["demand"], data["ground_cost"]))
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
                input_generation=False, baseline_sha256=manifest["binaries"]["old"]["sha256"],
                new_sha256=manifest["binaries"]["new"]["sha256"])


def _guarded_execute(batch, manifest, job_id, command, attempt, cancel, **kwargs):
    routed = list(command)
    if "-m" in routed:
        index = routed.index("-m") + 1
        if routed[index] == "src.experiments.conditional_benchmark_20261022":
            routed[index] = MODULE
    return _BASE_GUARD(batch, manifest, job_id, routed, attempt, cancel, **kwargs)


def enabled(result, key):
    for container in (result, result.get("accelerations") or {}, result.get("structural_strengthening") or {}):
        if isinstance(container, dict) and key in container:
            return container[key]
    return None


def record(batch, manifest, job, state):
    batch = Path(batch)
    row = _BASE_RECORD(batch, job, state)
    row.update(K=job["K"], version=job["version"], variant=job["variant"],
               structural_strengthening_requested=job["structural_strengthening"])
    source = batch / state.get("result_directory", state.get("directory", "_missing"))
    result = base.optional_read(source / "result.json")
    if not result:
        return row
    check = base.optional_read(source / "native_validation.json")
    valid = bool(check.get("passed"))
    for name, key in (("result.json", "result_sha256"), ("solution.json", "solution_sha256"), ("path_flows.json", "path_flows_sha256")):
        valid = valid and (source / name).is_file() and check.get(key) == base.sha256(source / name)
    valid = valid and check.get("manifest_sha256") == base.sha256(batch / job["instance"] / "manifest.json")
    valid = valid and result.get("max_orders") == job["K"] and result.get("solver") == "CPLEX"
    valid = valid and result.get("method") == "benders_cg" and result.get("demand_formulation") == job["method"].removeprefix("benders_")
    expected_binary = manifest["binaries"][job["version"]]["sha256"]
    binary_valid = result.get("binary_sha256") == expected_binary
    valid = valid and binary_valid
    for flag in FLAGS:
        actual = enabled(result, flag)
        # The frozen old executable predates these options; absence has a
        # justified meaning only for that exact old binary identity.
        if actual is None and job["version"] == "old" and binary_valid:
            actual = False
        row[flag + "_enabled"] = actual
        valid = valid and actual is job["structural_strengthening"]
    row.update(validation_passed=bool(valid), binary_identity_verified=binary_valid,
               binary_sha256=result.get("binary_sha256"),
               upper_bound=result.get("objective") if valid else None,
               lower_bound=base.certified_bound(result) if valid else None, gap=None,
               validation_read_seconds=check.get("read_seconds"),
               validation_seconds=check.get("validation_and_export_seconds"))
    if row["upper_bound"] is not None and row["lower_bound"] is not None:
        row["gap"] = (row["upper_bound"] - row["lower_bound"]) / max(1, abs(row["upper_bound"]))
    for prefix in ("continuous_root", "root_diagnostics", "acceleration_statistics", "conditional_statistics",
                   "topology_summary", "assignment_support_summary", "branch_joint_summary", "resource_lp_summary",
                   "structural_summary", "structural_strengthening_summary", "assignment_incident_clique_summary",
                   "shared_rho_joint_summary", "queue_perspective_summary"):
        values = result.get(prefix)
        if isinstance(values, dict):
            for key, value in values.items():
                if not isinstance(value, (list, dict)):
                    row[f"{prefix}_{key}"] = value
            if prefix not in ("continuous_root", "root_diagnostics"):
                row[prefix + "_json"] = json.dumps(values, sort_keys=True)
    continuous = result.get("continuous_root") or {}
    row["continuous_root_certified_lower_bound"] = continuous.get("certified_lower_bound") if valid and continuous.get("bound_valid") else None
    floor = continuous.get("applied_total_cost_lower_bound")
    row["global_lower_bound_gain_after_continuous_root"] = (row["lower_bound"]-floor
        if valid and isinstance(floor,(int,float)) and isinstance(row["lower_bound"],(int,float)) else None)
    root_rounds = (result.get("root_diagnostics") or {}).get("rounds")
    row["root_diagnostics_rounds_count"] = len(root_rounds) if isinstance(root_rounds,list) else root_rounds
    solves = result.get("master_solves")
    row["integer_master_solves_count"] = len(solves) if isinstance(solves,list) else None
    if isinstance(solves,list):
        row["integer_master_primal_count"] = sum(bool(item.get("has_solution")) for item in solves)
        row["integer_master_status_counts"] = json.dumps({status: sum(item.get("status")==status for item in solves)
            for status in dict.fromkeys(item.get("status") for item in solves)},sort_keys=True)
    for key, value in (continuous.get("model_size") or {}).items():
        if not isinstance(value, (dict, list)):
            row["continuous_root_model_" + key] = value
    sizes = next((origin.get("size", {}) for origin in manifest.get("input_origins", []) if origin["instance"] == job["instance"]), {})
    if "bases" in sizes and "orders" in sizes:
        l, r = sizes["bases"], sizes["orders"]
        row["theoretical_structural_master_added_rows"] = r * l if row.get("assignment_incident_clique_enabled") else 0
        row["theoretical_structural_subproblem_added_rows"] = (
            (r * l if row.get("assignment_incident_clique_enabled") else 0)
            + (2 * r * l if row.get("shared_rho_joint_enabled") else 0)
            + (2 * l if row.get("queue_perspective_enabled") else 0))
        row["theoretical_structural_added_variables"] = 0
        row["theoretical_structural_added_cones"] = 0
    for key, value in result.items():
        if isinstance(value, list) and (key.endswith("_cuts") or key.endswith("_cut_records")):
            row[key + "_record_count"] = len(value)
    for key, value in (result.get("timings") or {}).items():
        if not isinstance(value, (dict, list)):
            row["timing_" + key] = value
    return row


def _record(batch, job, state):
    return record(batch, base.read(Path(batch) / "experiment_manifest.json"), job, state)


def audit_worker(batch, job_id, attempt):
    """A feasible result from the wrong executable/settings is not this job."""
    audit = _BASE_AUDIT(batch, job_id, attempt)
    row = base.optional_read(Path(attempt) / "report_row.json")
    audit["method_and_binary_identity_verified"] = bool(row.get("validation_passed"))
    if not row.get("validation_passed"):
        audit["passed"] = False
        audit["errors"] = [*(audit.get("errors") or []), "structural_job_identity_or_validation_failed"]
    write_json(Path(attempt) / "artifact_audit.json", audit)
    return audit


def report(batch):
    from src.reporting.structural_comparison import report as render
    return render(batch)


def prepare_root_ablations(batch, instance, *, probe=None, seconds=180, parallel=3):
    """Freeze an independent diagnostic plan; root certificates are not free starts."""
    batch, instance = Path(batch).resolve(), Path(instance).resolve()
    if batch.exists():
        raise FileExistsError(f"Refusing existing root-ablation directory: {batch}")
    if seconds <= 0 or parallel < 1 or parallel > 3:
        raise ValueError("Invalid root-ablation budget/concurrency")
    for k in (1, 2):
        load_standard(instance, case=f"K{k}")
    probe = Path(probe or ROOT / "cpp/build/mlnd_continuous_probe").resolve()
    if not probe.is_file():
        raise FileNotFoundError(probe)
    (batch / "frozen").mkdir(parents=True)
    shutil.copytree(ROOT / "src", batch / "frozen/src", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copytree(ROOT / "configs", batch / "frozen/configs", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copy2(probe, batch / "frozen/mlnd_continuous_probe")
    shutil.copytree(instance, batch / "instance")
    variants = {
        "baseline": dict(zip(FLAGS, (False, False, False))),
        "joint": dict(zip(FLAGS, (True, True, False))),
        "perspective": dict(zip(FLAGS, (False, False, True))),
        "all": dict(zip(FLAGS, (True, True, True))),
    }
    jobs = []
    for k in (1, 2):
        for variant, features in variants.items():
            output = batch / "runs" / f"K{k}_{variant}" / "result.json"
            command = [str(batch / "frozen/mlnd_continuous_probe"), str(batch / "instance"), str(k), str(seconds), str(output),
                       "--cg-max-columns", "512", "--reuse-dual-lp", "true", "--tight-airline-bounds", "true"]
            for feature, active in features.items():
                command += ["--" + feature.replace("_", "-"), str(active).lower()]
            jobs.append(dict(id=f"K{k}_{variant}", K=k, variant=variant, features=features,
                             output=str(output), command=command, seconds=seconds))
    (batch / "run.py").write_text(
        "from pathlib import Path\nimport sys\n"
        "batch = Path(__file__).resolve().parent\n"
        "sys.path.insert(0, str(batch / 'frozen'))\n"
        f"from {MODULE} import main\n"
        "main([*sys.argv[1:], '--batch', str(batch)])\n", encoding="utf-8")
    hashes = {str(p.relative_to(batch)): base.sha256(p) for p in sorted(batch.rglob("*")) if p.is_file()}
    source_hashes = {str(p.relative_to(ROOT)): base.sha256(p) for directory, pattern in ((ROOT / "cpp/src", "*.cpp"), (ROOT / "cpp/include/mlnd", "*.hpp"), (ROOT / "cpp/tests", "continuous_probe.cpp")) for p in sorted(directory.glob(pattern))}
    manifest = dict(schema="structural-continuous-root-ablation/1", created_utc=base.existing.utc(), jobs=jobs,
                    seconds_per_job=seconds, threads=1, seed=43, parallel=parallel,
                    python=sys.executable, wall_limit_seconds=seconds + 30,
                    solver_started=False, root_certificates_supplied_to_production=False,
                    input_manifest_sha256=base.sha256(instance / "manifest.json"), frozen_sha256=hashes,
                    source_hashes=source_hashes, budget_scope="support, model construction, CG and internal validation within each root job; input reading is outside the native budget but included in process wall time")
    write_json(batch / "manifest.json", manifest)
    return dict(batch=str(batch), jobs=len(jobs), solver_started=False,
                note="Independent root-ablation command plan only; no solver dispatched")


def root_ablation_report(batch):
    batch = Path(batch).resolve()
    manifest = base.read(batch / "manifest.json")
    for relative, expected in manifest["frozen_sha256"].items():
        if not (batch / relative).is_file() or base.sha256(batch / relative) != expected:
            raise ValueError(f"Root-ablation frozen artifact mismatch: {relative}")
    from src.reporting.structural_comparison import csv_file, fmt, number, root_plots, structural_point_violations, table
    rows, violation_rows, evidence = [], [], []
    data = None
    for job in manifest["jobs"]:
        output = Path(job["output"])
        result = base.optional_read(output)
        execution = base.optional_read(output.parent / "execution.json")
        audit = base.optional_read(output.parent / "audit.json")
        valid = bool(audit.get("provenance_verified")) and output.is_file() and audit.get("result_sha256") == base.sha256(output)
        lower = result.get("certified_lower_bound") if valid and result.get("bound_valid") else None
        primal = result.get("continuous_objective", result.get("objective")) if valid and result.get("primal_valid") else None
        gap = max(0, primal - lower) / max(1, abs(primal)) if isinstance(lower, (int, float)) and isinstance(primal, (int, float)) else None
        row = dict(job=job["id"], K=job["K"], variant=job["variant"], returncode=execution.get("returncode"),
                   stop_reason=execution.get("stop_reason"), termination=result.get("termination_reason"),
                   provenance_verified=valid, root_lower_bound=lower, root_primal_objective=primal,
                   root_gap=gap, is_integer_upper_bound=False, bound_valid=result.get("bound_valid"),
                   primal_valid=result.get("primal_valid"), pricing_complete=result.get("pricing_complete"),
                   process_wall_seconds=execution.get("external_wall_seconds"),
                   method_wall_seconds=result.get("end_to_end_seconds", result.get("algorithm_wall")),
                   result_path=str(output), features=json.dumps(job["features"], sort_keys=True))
        for field in ("timings", "model_size", "structural_strengthening", "primal_residuals"):
            value = result.get(field)
            if isinstance(value, dict):
                for key, item in value.items():
                    if not isinstance(item, (dict, list)):
                        row[f"{field}_{key}"] = item
        for field, value in result.items():
            if "violation" in field or "residual" in field or field == "operating_point":
                row[field + "_json"] = json.dumps(value, sort_keys=True)
        design = result.get("design")
        if isinstance(design, list):
            row["strategic_variables"] = len(design)
            row["fractional_strategic_variables"] = sum(1e-6 < x < 1 - 1e-6 for x in design)
        if valid and result.get("operating_point"):
            if data is None:
                data = load_standard(batch / "instance", case="K1")
            violations, details = structural_point_violations(result, data)
            row.update({"structural_violation_"+key: value for key, value in violations.items()})
            row["base_opening_values"] = json.dumps(result["design"][:data["n_l"]])
            for key in ("rho", "idle", "fleet"):
                row["root_"+key+"_values"] = json.dumps(result["operating_point"].get(key))
            violation_rows.extend(dict(job=job["id"], K=job["K"], variant=job["variant"], **detail) for detail in details)
        evidence.append(dict(job=job["id"], result_sha256=base.sha256(output) if output.is_file() else None,
                             audited_result_sha256=audit.get("result_sha256"), provenance_verified=valid))
        rows.append(row)
    for row in rows:
        original = next(item for item in rows if item["K"] == row["K"] and item["variant"] == "baseline")
        available = number(row.get("root_lower_bound")) and number(original.get("root_lower_bound"))
        row["root_lower_bound_gain"] = row["root_lower_bound"]-original["root_lower_bound"] if available else None
        row["root_lower_bound_relative_gain"] = row["root_lower_bound_gain"] / max(1,abs(original["root_lower_bound"])) if available else None
    output = batch / "analysis"
    output.mkdir(exist_ok=True)
    write_json(output / "comparison.json", dict(protocol=manifest, rows=rows))
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (output / "comparison.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    csv_file(output / "structural_violations.csv", violation_rows)
    from src.reporting import structural_comparison
    write_json(output / "analysis_audit.json", dict(manifest_sha256=base.sha256(batch / "manifest.json"),
               report_source_sha256=base.sha256(__file__), arithmetic_source_sha256=base.sha256(structural_comparison.__file__),
               frozen_files_verified=True, records=evidence, all_provenance_verified=all(row["provenance_verified"] for row in rows),
               independent_optimization_performed=False))
    lines = ["# 独立连续根消融", "", "四种设置为 baseline、joint（incident clique + shared rho）、perspective only、all，分别测试 K=1、2。"
             f"每组独立预算 {manifest['seconds_per_job']:g} 秒、单线程、最多三个并发；支持构造、模型构造与 CG 包含在原生预算内，输入读取另包含在进程墙钟时间中。"
             "这些根结果不作为正式算法比较的免费初始化。root primal objective 只上界对应连续根最优值，不是原整数问题的 UB。", ""]
    lines += table(["K", "设置", "根 LB", "根 primal", "根 gap", "方法秒", "进程秒", "定价完整", "终止", "身份"],
        [(r["K"], r["variant"], r["root_lower_bound"], r["root_primal_objective"], fmt(r["root_gap"], True),
          r["method_wall_seconds"], r["process_wall_seconds"], r["pricing_complete"], r["termination"], r["provenance_verified"]) for r in rows])
    lines += ["", "同一 K 下相对原始连续松弛的下界增量；只有定价完成且上下界足够接近时，才可将增量解释为松弛最优值的差异。", ""]
    lines += table(["K", "设置", "Δ根 LB", "Δ根 LB / 原根 LB"],
                   [(row["K"],row["variant"],row["root_lower_bound_gain"],fmt(row["root_lower_bound_relative_gain"],True)) for row in rows])
    if violation_rows:
        lines += ["", "逐点结构核查使用保存的、已验证的分数可行点，独立重新计算新约束左侧减右侧的正部，阈值为 1e-6。下表为违反条数及最大绝对违反量；锥残差为 x²−(I+x)(x−ρ)，量纲不同于线性行，不能跨行比较大小。", ""]
        families = ("assignment_incident_clique", "shared_rho_executed", "shared_rho_unexecuted", "queue_perspective_cone")
        lines += table(["K", "设置", "incident", "共同ρ已服务", "共同ρ未服务", "排队perspective"],
            [(row["K"],row["variant"], *[f"{row.get('structural_violation_'+family+'_count','—')} / {fmt(row.get('structural_violation_'+family+'_max'))}" for family in families]) for row in rows])
        def vector(value):
            items = json.loads(value) if value else None
            return " / ".join(fmt(item) for item in items) if isinstance(items, list) else "—"
        lines += ["", "以下依次列出各基地的开设值 x、共同可用率 ρ 和待机无人机数量 I。这些是连续根分数解，不能直接作为整数配置或实际部署方案。", ""]
        lines += table(["K", "设置", "基地开设 x", "可用率 ρ", "待机量 I"],
            [(row["K"],row["variant"],vector(row.get("base_opening_values")),vector(row.get("root_rho_values")),vector(row.get("root_idle_values"))) for row in rows])
        figures = output / "figures"; figures.mkdir(exist_ok=True)
        root_plots(figures, rows)
        lines += ["", "![连续松弛认证下界增量](figures/root_lower_bound_gain.png)"]
    lines += ["", "原始残差、结构违反量与 operating_point（如果本轮证书输出）保存在 comparison.csv 与各运行 result.json。"
              "未给出有效根 LB 或 primal 时留空，不拿未认证 RMP 值补齐，也不把定价未结束当作数学不可行。", ""]
    (output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    return dict(report=str(output / "REPORT.md"), jobs=len(rows), verified=sum(r["provenance_verified"] for r in rows))


def run_root_ablations(batch):
    batch = Path(batch).resolve()
    manifest = base.read(batch / "manifest.json")
    for name, expected in manifest["frozen_sha256"].items():
        if not (batch / name).is_file() or base.sha256(batch / name) != expected:
            raise ValueError(f"Root-ablation frozen artifact mismatch: {name}")
    if base.sha256(batch / "instance/manifest.json") != manifest["input_manifest_sha256"]:
        raise ValueError("Root-ablation input manifest mismatch")
    if (batch / "runs").exists():
        raise FileExistsError("Root runs already exist; refusing to overwrite/repeat a partial run")
    expected_input = base.read(batch / "instance/manifest.json")
    expected_model = base.sha256(batch / "instance/model.json")
    cancel, lock = threading.Event(), threading.Lock()
    state = dict(status="running", jobs={job["id"]: dict(status="pending") for job in manifest["jobs"]})
    config = dict(seconds_per_job=manifest["seconds_per_job"], watchdog_grace_seconds=30,
                  resource_sample_seconds=.5)

    def one(job):
        output = Path(job["output"]); output.parent.mkdir(parents=True)
        with lock:
            state["jobs"][job["id"]] = dict(status="running", started_utc=base.existing.utc())
            write_json(batch / "state.json", state)
        write_json(output.parent / "request.json", dict(job=job, manifest_sha256=base.sha256(batch / "manifest.json")))
        execution = base.existing.execute(job["command"], batch / "frozen", output.parent, config, cancel)
        result = base.optional_read(output)
        provenance = bool(result) and result.get("K") == job["K"] and result.get("input_manifest") == expected_input and result.get("input_model_sha256") == expected_model
        expected_scope = ("structurally_strengthened_continuous_relaxation"
                          if any(job["features"].values()) else "full_path_continuous_relaxation")
        provenance = provenance and result.get("method") == "full_continuous_cg" and result.get("scope") == expected_scope and result.get("objective_includes_fixed_costs") is True
        for feature, wanted in job["features"].items():
            provenance = provenance and enabled(result, feature) is wanted
        audit = dict(provenance_verified=bool(provenance), result_sha256=base.sha256(output) if output.is_file() else None,
                     frozen_binary_sha256=manifest["frozen_sha256"]["frozen/mlnd_continuous_probe"],
                     input_manifest_sha256=manifest["input_manifest_sha256"], returncode=execution.get("returncode"),
                     bound_valid=result.get("bound_valid"), primal_valid=result.get("primal_valid"),
                     pricing_complete=result.get("pricing_complete"), termination=result.get("termination_reason"),
                     bound_scope="continuous_root_model", is_original_integer_primal=False)
        write_json(output.parent / "audit.json", audit)
        with lock:
            state["jobs"][job["id"]] = dict(status="completed" if provenance and execution.get("returncode") == 0 else "failed",
                finished_utc=base.existing.utc(), audit=audit)
            write_json(batch / "state.json", state)
    with base.existing.batch_lock(batch):
        if (batch / "runs").exists():
            raise FileExistsError("Root runs already exist; refusing to overwrite/repeat a partial run")
        write_json(batch / "state.json", state)
        with ThreadPoolExecutor(max_workers=min(3, manifest["parallel"])) as pool:
            list(pool.map(one, manifest["jobs"]))
    state["status"] = "finished_dispatch"
    write_json(batch / "state.json", state)
    return root_ablation_report(batch)


@contextmanager
def adapters():
    replacements = dict(command_for=command_for, _record=_record, guarded_execute=_guarded_execute,
                        audit_worker=audit_worker, report=report)
    previous = {key: getattr(base, key) for key in replacements}
    try:
        for key, value in replacements.items():
            setattr(base, key, value)
        yield
    finally:
        for key, value in previous.items():
            setattr(base, key, value)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "prepare-root-ablations", "run-root-ablations", "root-report", "verify", "run", "launch", "report", "status", "audit-worker"))
    parser.add_argument("--batch", type=Path, default=DEFAULT_BATCH)
    parser.add_argument("--instance", type=Path, default=ROOT / "outputs/tsl_scale_ladder/instances/L02")
    parser.add_argument("--new-binary", type=Path)
    parser.add_argument("--root-probe", type=Path)
    parser.add_argument("--baseline-freeze", type=Path)
    parser.add_argument("--seconds", type=float)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--parallel", type=int, default=3)
    parser.add_argument("--gap", type=float, default=.005)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--hard-wall-seconds", type=float, default=720)
    parser.add_argument("--jobs", nargs="+")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--job-id")
    parser.add_argument("--attempt", type=Path)
    args = parser.parse_args(argv)
    seconds = args.seconds if args.seconds is not None else (180 if args.action == "prepare-root-ablations" else 600)
    with adapters():
        if args.action == "prepare-root-ablations":
            result = prepare_root_ablations(args.batch, args.instance, probe=args.root_probe,
                                           seconds=seconds, parallel=args.parallel)
        elif args.action == "run-root-ablations":
            result = run_root_ablations(args.batch)
        elif args.action == "root-report":
            result = root_ablation_report(args.batch)
        elif args.action == "prepare":
            result = prepare(args.batch, [args.instance], new_binary=args.new_binary, baseline_freeze=args.baseline_freeze,
                             seconds=seconds, threads=args.threads, parallel=args.parallel, gap=args.gap,
                             seed=args.seed, hard_wall_seconds=args.hard_wall_seconds)
        elif args.action == "verify":
            result = dict(passed=bool(base.verify(args.batch)), solver_started=False)
        elif args.action == "run":
            result = base.run(args.batch, only=args.jobs, retry_failed=args.retry_failed)
        elif args.action == "launch":
            result = base.launch(args.batch)
        elif args.action == "report":
            result = report(args.batch)
        elif args.action == "audit-worker":
            if not args.job_id or not args.attempt:
                parser.error("audit-worker requires --job-id and --attempt")
            result = base.audit_worker(args.batch, args.job_id, args.attempt)
        else:
            result = base.existing.status_summary(args.batch)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
