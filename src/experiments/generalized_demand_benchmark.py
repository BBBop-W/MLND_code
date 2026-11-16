"""Freeze and compare single, legacy double, and preprocessed double demand.

The experiment reuses an existing standard instance, the native CPLEX entry,
the shared process guardian, and independent native operations validation.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import shutil
import subprocess
import sys
import threading

from src.experiments import conditional_benchmark_20261022 as shared
from src.experiments import scale_ladder
from src.experiments.storage import sha256, source_hashes, write_json
from src.reporting.native import native_phase_times


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BATCH = ROOT / "outputs/generalized_demand_L01_20261026"
DEFAULT_FIRST_LEG_BATCH = ROOT / "outputs/generalized_first_leg_L01_20261026"
DEFAULT_INSTANCE = ROOT / "outputs/tsl_scale_ladder/instances/L01"
JOBS = (
    {"id": "single", "label": "单单 K=1", "k": 1, "demand_formulation": "lifted"},
    {"id": "legacy_double", "label": "原双单 K=2", "k": 2, "demand_formulation": "lifted"},
    {"id": "preprocessed_double", "label": "预处理双单 K=2", "k": 2, "demand_formulation": "preprocessed"},
)
FIRST_LEG_JOBS = (
    {"id": "preprocessed_double", "label": "预处理双单：汇总比例", "k": 2,
     "demand_formulation": "preprocessed", "first_leg_consistency": False},
    {"id": "preprocessed_first_leg", "label": "预处理双单：首程一致", "k": 2,
     "demand_formulation": "preprocessed", "first_leg_consistency": True},
)


def read(path, default=None):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else default


def _source_files():
    hashes = source_hashes()
    for directory in ("cpp/src", "cpp/include"):
        for path in sorted((ROOT / directory).rglob("*")):
            if path.is_file():
                hashes[str(path.relative_to(ROOT))] = sha256(path)
    for name in ("cpp/Makefile", "cpp/CMakeLists.txt"):
        hashes[name] = sha256(ROOT / name)
    return hashes


def prepare(batch, instance=DEFAULT_INSTANCE, binary=None, *, first_leg_comparison=False,
            reference_batch=None):
    """Freeze sources/binary without optimizing or regenerating input data."""
    batch, instance = Path(batch).resolve(), Path(instance).resolve()
    binary = Path(binary or ROOT / "cpp/build/mlnd_cpp").resolve()
    if batch.exists():
        raise FileExistsError(f"Refusing to replace an existing experiment: {batch}")
    if not binary.is_file():
        raise FileNotFoundError(binary)
    original_manifest = read(instance / "manifest.json")
    if not original_manifest or original_manifest.get("schema") != "consecutive-csv/1":
        raise ValueError("Use an existing consecutive-csv/1 standard instance")
    for name, digest in original_manifest["files"].items():
        if sha256(instance / name) != digest:
            raise ValueError(f"Input hash mismatch: {name}")
    model = read(instance / "model.json")["values"]
    if model.get("flow_policy") != "aggregate":
        raise ValueError("The prototype experiment requires existing aggregate flow policy")
    jobs = FIRST_LEG_JOBS if first_leg_comparison else JOBS
    reference = _reference_rows(reference_batch or DEFAULT_BATCH, instance) if first_leg_comparison else None
    batch.mkdir(parents=True)
    sources = _source_files()
    for name, digest in sources.items():
        target = batch / "frozen" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
        if sha256(target) != digest:
            raise RuntimeError(f"Source changed while freezing: {name}")
    frozen_binary = batch / "frozen/bin/mlnd_cpp"
    frozen_binary.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(binary, frozen_binary)
    manifest = {
        "schema": "generalized-demand-comparison/1",
        "created_utc": scale_ladder.utc(),
        "python": sys.executable,
        "source_root": str(ROOT),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "git_status": subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True),
        "source_sha256": sources,
        "binary_source": str(binary),
        "binary_sha256": sha256(frozen_binary),
        "input_directory": str(instance),
        "input_manifest_sha256": sha256(instance / "manifest.json"),
        "input_manifest": original_manifest,
        "instance_values": model,
        "jobs": list(jobs),
        "comparison": "first_leg_consistency" if first_leg_comparison else "single_legacy_preprocessed",
        "reference": reference,
        "protocol": {
            "method": "full_path", "seconds_per_job": 600, "threads_per_job": 1,
            "parallel_jobs": 1, "relative_gap_target": 1e-6, "seed": 43,
            "hard_wall_seconds": 660, "validation_wall_seconds": 120,
            "resource_sample_seconds": 0.5,
            "watchdog_grace_seconds": 60,
            "memory_measurement": "optional process-tree samples plus child wait4 peak RSS",
            "memory_hard_limit": None,
            "input_generation": False,
            "budget_scope": "input_preprocessing_construction_and_optimization",
            "validation_outside_solve_budget": True,
        },
    }
    write_json(batch / "experiment_manifest.json", manifest)
    write_json(batch / "state.json", {"jobs": {job["id"]: {"status": "pending"} for job in jobs}})
    (batch / "run.py").write_text(
        "from pathlib import Path\nimport sys\n"
        "batch = Path(__file__).resolve().parent\n"
        "sys.path.insert(0, str(batch / 'frozen'))\n"
        "from src.experiments.generalized_demand_benchmark import main\n"
        "main([*sys.argv[1:], '--batch', str(batch)])\n", encoding="utf-8"
    )
    return {"batch": str(batch), "jobs": len(jobs), "solver_started": False}


def verify(batch):
    batch = Path(batch).resolve()
    manifest = read(batch / "experiment_manifest.json")
    if not manifest or manifest.get("schema") != "generalized-demand-comparison/1":
        raise ValueError("Unrecognized experiment manifest")
    if sha256(batch / "frozen/bin/mlnd_cpp") != manifest["binary_sha256"]:
        raise ValueError("Frozen executable hash changed")
    for name, digest in manifest["source_sha256"].items():
        if sha256(batch / "frozen" / name) != digest:
            raise ValueError(f"Frozen source hash changed: {name}")
    instance = Path(manifest["input_directory"])
    if sha256(instance / "manifest.json") != manifest["input_manifest_sha256"]:
        raise ValueError("Input manifest changed")
    for name, digest in manifest["input_manifest"]["files"].items():
        if sha256(instance / name) != digest:
            raise ValueError(f"Input file changed: {name}")
    for name, digest in (manifest.get("reference") or {}).get("files_sha256", {}).items():
        if sha256(name) != digest:
            raise ValueError(f"Read-only reference changed: {name}")
    return manifest


def command_for(batch, manifest, job, output):
    cfg = manifest["protocol"]
    command = [
        str(Path(batch) / "frozen/bin/mlnd_cpp"), "--input", manifest["input_directory"],
        "--method", "full_path", "--k", str(job["k"]),
        "--demand-formulation", job["demand_formulation"],
        "--seconds", str(cfg["seconds_per_job"]), "--gap", str(cfg["relative_gap_target"]),
        "--threads", str(cfg["threads_per_job"]), "--seed", str(cfg["seed"]),
        "--path-column-limit", "0", "--log", "--out", str(output),
    ]
    if job.get("first_leg_consistency"):
        command.append("--first-leg-consistency")
    return command


def run(batch, selected=None):
    """Run pending cases sequentially; existing attempt files are never replaced."""
    batch = Path(batch).resolve()
    manifest = verify(batch)
    selected = set(selected or [job["id"] for job in manifest["jobs"]])
    if selected - {job["id"] for job in manifest["jobs"]}:
        raise ValueError("Unknown job selection")
    cancel = threading.Event()
    with scale_ladder.batch_lock(batch):
        state = read(batch / "state.json")
        for job in manifest["jobs"]:
            if job["id"] not in selected:
                continue
            if state["jobs"][job["id"]]["status"] != "pending":
                continue
            verify(batch)
            output = batch / "runs" / job["id"]
            output.mkdir(parents=True, exist_ok=False)
            state["jobs"][job["id"]] = {"status": "running", "directory": str(output.relative_to(batch))}
            write_json(batch / "state.json", state)
            command = command_for(batch, manifest, job, output)
            execution = scale_ladder.execute(command, batch / "frozen", output, manifest["protocol"], cancel)
            validation_execution = None
            if all((output / name).is_file() for name in ("result.json", "solution.json", "path_flows.json")):
                validation_dir = output / "postprocess"
                validation_dir.mkdir()
                validation_command = [manifest["python"], "-m", "src.experiments.native", "validate",
                                      str(output), "--instance", manifest["input_directory"]]
                validation_execution = scale_ladder.execute(
                    validation_command, batch / "frozen", validation_dir,
                    dict(manifest["protocol"], seconds_per_job=manifest["protocol"]["validation_wall_seconds"],
                         watchdog_grace_seconds=0), cancel,
                )
            validation = read(output / "native_validation.json", {})
            state["jobs"][job["id"]].update(
                status="completed" if validation.get("passed") else "failed",
                process_returncode=execution.get("returncode"),
                stop_reason=execution.get("stop_reason"),
                validation_returncode=(validation_execution or {}).get("returncode"),
            )
            write_json(batch / "state.json", state)
            report(batch)
    return report(batch)


def _record(batch, manifest, job, state):
    output = Path(batch) / state.get("directory", f"runs/{job['id']}")
    result = read(output / "result.json", {})
    validation = read(output / "native_validation.json", {})
    execution = read(output / "execution.json", {})
    valid = bool(validation.get("passed"))
    for filename, key in (("result.json", "result_sha256"), ("solution.json", "solution_sha256"),
                          ("path_flows.json", "path_flows_sha256")):
        valid = valid and (output / filename).is_file() and validation.get(key) == sha256(output / filename)
    valid = valid and validation.get("manifest_sha256") == manifest["input_manifest_sha256"]
    valid = valid and result.get("max_orders") == job["k"]
    valid = valid and result.get("demand_formulation") == job["demand_formulation"]
    valid = valid and result.get("binary_sha256") == manifest["binary_sha256"]
    valid = valid and bool(result.get("first_leg_consistency", False)) == job.get("first_leg_consistency", False)
    summary = validation.get("summary", {})
    metrics = result.get("metrics", {})
    size, catalogue = result.get("model_size", {}), result.get("catalogue", {})
    row = {
        **job, "status": state["status"], "validation_passed": valid,
        "solver_status": result.get("solver_status"),
        "termination_reason": execution.get("stop_reason") or result.get("termination_reason"),
        "assignment_scope": result.get("assignment_scope"), "flow_policy": result.get("flow_policy"),
        "first_leg_consistency": result.get("first_leg_consistency", job.get("first_leg_consistency", False)),
        "first_leg_consistency_diagnostics": result.get("first_leg_consistency_diagnostics"),
        "upper_bound": result.get("objective") if valid else None,
        "lower_bound": shared.certified_bound(result) if valid else None,
        "solver_reported_gap": result.get("gap"),
        "solve_seconds": native_phase_times(result)["solve_seconds"] if result else None,
        "external_wall_seconds": execution.get("external_wall_seconds"),
        "validation_seconds": validation.get("validation_and_export_seconds"),
        "peak_rss_bytes": execution.get("peak_rss_bytes"),
        "variables": size.get("variables"), "linear_rows": size.get("linear_rows"),
        "cones": size.get("cones"), "full_service_paths": catalogue.get("service"),
        "full_empty_paths": catalogue.get("empty"), "initial_segments": catalogue.get("segments"),
        "catalogue_complete": result.get("catalogue_complete"),
        "construction_complete": result.get("construction_complete"),
        "result_directory": str(output),
    }
    row["gap"] = ((row["upper_bound"] - row["lower_bound"]) / max(1, abs(row["upper_bound"]))
                  if row["upper_bound"] is not None and row["lower_bound"] is not None else None)
    for key in ("demand", "served", "direct", "single_tasks", "double_tasks", "task_flights",
                "fleet", "slot_total", "service_fraction", "mean_task_minutes", "orders_per_service_task",
                "assigned_od_count", "served_od_count"):
        row[key] = summary.get(key) if valid else None
    for name, value in metrics.get("costs_per_hour", {}).items():
        row["cost_" + name] = value if valid else None
    row["opened_bases"] = summary.get("opened", {}).get("bases") if valid else None
    row["opened_lockers"] = summary.get("opened", {}).get("lockers") if valid else None
    for name in ("demand_preprocessing", "preprocessed_demand", "generalized_demand"):
        if name in result:
            row["preprocessing"] = result[name]
            for key in ("eligible_positive_saving_pairs", "accepted_pairs", "support_rejected_pairs",
                        "max_conservation_error", "raw_order_rate", "double_task_rate", "single_task_rate",
                        "task_rate", "seconds"):
                row["preprocessing_" + key] = result[name].get(key)
    for key, value in (result.get("first_leg_consistency_diagnostics") or {}).items():
        if not isinstance(value, (dict, list)):
            row[key if key.startswith("first_leg_") else "first_leg_" + key] = value
    first_leg_residuals = {key: value for key, value in validation.get("checks", {}).get("residuals", {}).items()
                          if "first_leg" in key or "proportion" in key}
    row["first_leg_validation_residuals"] = first_leg_residuals
    return row


def _reference_rows(reference_batch, instance):
    """Capture only already validated single/legacy results; never solve them."""
    reference_batch = Path(reference_batch).resolve()
    if not (reference_batch / "experiment_manifest.json").is_file():
        return {"source": str(reference_batch), "available": False, "reason": "reference_batch_missing"}
    manifest = verify(reference_batch)
    if manifest["input_manifest_sha256"] != sha256(Path(instance) / "manifest.json"):
        raise ValueError("Reference input differs from the current experiment")
    state = read(reference_batch / "state.json")
    rows, files = [], {}
    for job in manifest["jobs"]:
        if job["id"] not in ("single", "legacy_double"):
            continue
        row = _record(reference_batch, manifest, job, state["jobs"][job["id"]])
        if not row["validation_passed"]:
            raise ValueError(f"Reference case is not independently validated: {job['id']}")
        row["reference_only"] = True
        rows.append(row)
        for name in ("result.json", "solution.json", "path_flows.json", "native_validation.json"):
            path = Path(row["result_directory"]) / name
            files[str(path)] = sha256(path)
    if len(rows) != 2:
        raise ValueError("Reference must contain both existing single and legacy-double cases")
    files[str(reference_batch / "experiment_manifest.json")] = sha256(reference_batch / "experiment_manifest.json")
    return {"source": str(reference_batch), "available": True, "rows": rows, "files_sha256": files,
            "reference_binary_sha256": manifest["binary_sha256"], "solvers_rerun": False}


def _first_leg_comparison(rows):
    by_id = {row["id"]: row for row in rows}
    aggregate, consistent = by_id.get("preprocessed_double"), by_id.get("preprocessed_first_leg")
    if not aggregate or not consistent:
        return None
    first, second = aggregate.get("preprocessing"), consistent.get("preprocessing")
    if not first or not second:
        return {"available": False}
    same = all(first.get(key) == second.get(key) for key in
               ("single_rates", "pair_rates", "head_rates", "candidate_order", "common_origin_support"))
    if not same:
        raise ValueError("First-leg comparison changed the preprocessed rates or candidate order")
    comparison = {"available": True, "demand_arrays_identical": same,
                  "same_complete_catalogue": aggregate["full_service_paths"] == consistent["full_service_paths"],
                  "theoretical_cost_order": "optimal_consistent_cost >= optimal_aggregate_cost"}
    if aggregate["validation_passed"] and consistent["validation_passed"]:
        comparison["incumbent_cost_difference"] = consistent["upper_bound"] - aggregate["upper_bound"]
        if aggregate["lower_bound"] is not None and consistent["lower_bound"] is not None:
            comparison["optimal_cost_difference_lower_bound"] = consistent["lower_bound"] - aggregate["upper_bound"]
            comparison["optimal_cost_difference_upper_bound"] = consistent["upper_bound"] - aggregate["lower_bound"]
    return comparison


def _fmt(value):
    if value is None:
        return "—"
    return f"{value:,.6f}" if isinstance(value, float) else str(value)


def _table(columns, rows):
    return "\n".join([
        "| " + " | ".join(label for _, label in columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
        *("| " + " | ".join(_fmt(row.get(key)) for key, _ in columns) + " |" for row in rows),
    ])


def report(batch):
    batch = Path(batch).resolve()
    manifest = verify(batch)
    state = read(batch / "state.json")
    rows = [_record(batch, manifest, job, state["jobs"][job["id"]]) for job in manifest["jobs"]]
    first_leg = manifest.get("comparison") == "first_leg_consistency"
    pair_check = _first_leg_comparison(rows) if first_leg else None
    output = batch / "analysis"
    output.mkdir(exist_ok=True)
    write_json(output / "comparison.json", {"protocol": manifest["protocol"], "rows": rows,
                                           "first_leg_comparison": pair_check,
                                           "reference": manifest.get("reference")})
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (output / "comparison.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value
                          for key, value in row.items()} for row in rows)
    n = manifest["instance_values"]
    text = [
        "# 固定需求下的首程一致性比较" if first_leg else "# 单单、原双单与预处理双单：一个既有真实算例", "",
        f"输入：`{manifest['input_directory']}`，{n['n_l']} 个基地候选、{n['n_h']} 个柜点、{n['n_od']} 个 OD。"
        f"输入未重新生成；{len(rows)} 组采用同一冻结 CPLEX 程序、完整路径、600 秒总预算、单线程、seed=43、gap=1e-6，顺序运行。", "",
        "UB 仅纳入通过独立运营核验且原始文件哈希相符的可行解；LB 仅采用全局认证界。"
        "预算内时间包含输入、需求预处理、路径枚举、构模与求解，独立验证另计。", "",
        _table((("label", "模型"), ("upper_bound", "UB"), ("lower_bound", "LB"), ("gap", "相对 gap"),
                ("solve_seconds", "预算内秒数"), ("solver_status", "求解状态"),
                ("termination_reason", "终止原因"), ("validation_passed", "验证")), rows), "",
        _table((("label", "模型"), ("variables", "变量"), ("linear_rows", "线性行"), ("cones", "锥"),
                ("full_service_paths", "完整服务路径"), ("full_empty_paths", "空飞路径")), rows), "",
        _table((("label", "模型"), ("served", "无人机订单/h"), ("direct", "直接骑手订单/h"),
                ("single_tasks", "单单任务/h"), ("double_tasks", "双单任务/h"), ("fleet", "机队"),
                ("opened_lockers", "开柜点"), ("slot_total", "格口")), rows), "",
        _table((("label", "模型"), ("preprocessing_accepted_pairs", "正需求双单类型"),
                ("preprocessing_single_task_rate", "重构单单任务/h"),
                ("preprocessing_double_task_rate", "重构双单任务/h"),
                ("preprocessing_max_conservation_error", "逐 OD 最大守恒误差")), rows), "",
        "新模型先对全体原始 OD 需求进行残余流量扣除，得到固定的单单和有序双单需求；"
        "同一首单 OD 的任务共享出发基地，第二个 OD 可作为其他任务的首单另行分配。"
        "未由无人机执行的双单任务，两单均计入直接骑手需求，不在结果阶段隐含拆回无人机单单。", "",
        ("两组需求率、配对顺序和完整路径目录相同。首程一致组增加逐首程的单单/各双单分支比例约束，"
         "返回基地仍可由分支路径选择。它的可行域包含于汇总比例组，因此理论最优成本不能更低。"
         "限时可行解的目标差不直接证明这一最优值差，报告同时保留两组上下界。") if first_leg else
        ("L01 仅有一个基地候选，因此出发基地分配与基地对分配没有返回基地自由度的差异。"
         "但新模型允许一个 OD 即使未作为首单分配，也通过其他 OD 的双单任务被服务；"
         "同时需求在选址前固定。这些均属于本次模型变体，不能把成本变化单独归因于求解器提速或连续接单次数。"), "",
        "路径的飞行、骑手等待、每单时效、柜格占用和机队参数沿用现有物理资源函数。"
        "预处理排序是给定启发式，需求守恒不证明在线匹配可实现；单一算例也不支持普遍成本或速度结论。", "",
        "变量、线性行和锥数量为输入求解器的原始完整模型统计，不是求解器 presolve 后的规模。"
        "求解进程组外部时限为 660 秒；验证另有 120 秒。峰值内存采用已有 child wait4 记录，"
        "进程树采样在系统允许时补充，本小算例未设置内存硬限制。", "",
        "原始解、路径流、独立验证与运营明细位于 `runs/`；`experiment_manifest.json` 保存源码、程序和输入哈希。"
        "完整数值和分项成本见 `comparison.csv` / `comparison.json`。",
    ]
    if first_leg:
        text += ["", "## 首程支持与比例核验", "",
                 "首程一致约束保留原有重构需求，不会为缺少共同可行首程的首单重新筛选或减小双单率。"
                 "若某首单没有同时支持全部正需求分支的首程，则该首单的无人机发出任务量会受限或为零；"
                 "这属于模型加强后的结果，未被服务的订单进入既有骑手余量。", "",
                 _table((("label", "模型"), ("first_leg_consistency", "首程一致"),
                         ("first_leg_groups", "首程数"),
                         ("first_leg_complete_first_leg_groups", "支持全部分支的首程数"),
                         ("first_leg_supported_positive_head_ods", "有完整首程支持的首单 OD"),
                         ("first_leg_unsupported_positive_head_ods", "无完整首程支持的首单 OD")), rows), "",
                 "需求相同及界的比较：`" + json.dumps(pair_check, ensure_ascii=False) + "`。", "",
                 "逐首程和比例核验残差保存于 CSV/JSON 的 `first_leg_validation_residuals`，"
                 "支持矩阵位于各结果文件的 `first_leg_consistency_diagnostics`。"]
        reference = manifest.get("reference") or {}
        if reference.get("available"):
            text += ["", "## 已有单单和原双单结果（只读参考）", "",
                     f"来源：`{reference['source']}`。下表沿用已验证的原结果，本批次没有重跑。"
                     "这些结果来自此前冻结程序，不作为本轮同二进制算法耗时对照。", "",
                     _table((("label", "模型"), ("upper_bound", "UB"), ("lower_bound", "LB"),
                             ("gap", "相对 gap"), ("served", "无人机订单/h"),
                             ("double_tasks", "双单任务/h"), ("fleet", "机队")), reference["rows"])]
        else:
            text += ["", "已有单单/原双单只读参考不可用，本批次没有自动补跑。"]
    (output / "REPORT.md").write_text("\n".join(text) + "\n", encoding="utf-8")
    return {"report": str(output / "REPORT.md"), "validated": sum(row["validation_passed"] for row in rows),
            "jobs": len(rows)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "verify", "run", "report"))
    parser.add_argument("--batch", type=Path)
    parser.add_argument("--instance", type=Path, default=DEFAULT_INSTANCE)
    parser.add_argument("--binary", type=Path)
    parser.add_argument("--jobs", nargs="+", choices=sorted({job["id"] for job in (*JOBS, *FIRST_LEG_JOBS)}))
    parser.add_argument("--first-leg-comparison", action="store_true",
                        help="Prepare only aggregate/preprocessed and first-leg-consistent/preprocessed cases")
    parser.add_argument("--reference-batch", type=Path,
                        help="Existing validated single/legacy results, read only")
    args = parser.parse_args(argv)
    args.batch = args.batch or (DEFAULT_FIRST_LEG_BATCH if args.first_leg_comparison else DEFAULT_BATCH)
    if args.action == "prepare":
        result = prepare(args.batch, args.instance, args.binary,
                         first_leg_comparison=args.first_leg_comparison, reference_batch=args.reference_batch)
    elif args.action == "verify":
        manifest = verify(args.batch)
        result = {"verified": True, "instance": manifest["input_directory"], "jobs": len(manifest["jobs"])}
    elif args.action == "run":
        result = run(args.batch, args.jobs)
    else:
        result = report(args.batch)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
