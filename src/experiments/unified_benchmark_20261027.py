"""Real-map unified-task comparison using the existing guarded benchmark runner.

``generate`` prepares inputs only; ``prepare`` freezes inputs/code/binary only.
Only an explicit ``run`` or ``launch`` starts the solver. No plotting pipeline
is introduced: native validation and the established operations export are reused.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
import csv
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import signal
import tempfile
import time

import numpy as np

from src.config import load_config
from src.data.performance_orders import read_orders
from src.data.prepare import prepare_orders
from src.data.scale_ladder import _membership_audit, _validate_level
from src.data.standard import INPUT_FILES, export_standard, load_standard
from src.experiments import conditional_benchmark_20261022 as base
from src.experiments.storage import write_json

ROOT = base.ROOT
MODULE = "src.experiments.unified_benchmark_20261027"
DEFAULT_MAPS = ROOT / "outputs/unified_maps_20261027"
DEFAULT_BATCH = ROOT / "outputs/unified_benchmark_20261027"
CONFIG = ROOT / "configs/tsl_20h_benchmark.json"
TSL_ANCHORS = {
    1: ROOT / "outputs/tsl_scale_ladder/instances/L01",
    2: ROOT / "examples/tsl_2l20h50od",
}
MARKETS = {
    name: ROOT / f"outputs/consecutive_real_comparison_20261015/instances/{name}"
    for name in ("tsl", "nanshan")
}
HALF_L2 = 2.8867513459481287
METHODS = ("full_cplex", "benders_unified")
COMMON_FLAGS = ("--shared-rho-joint", "--queue-perspective", "--tight-airline-bounds")
ALGORITHM_FLAGS = ("--od-airline-bounds", "--topology-cuts")
_BASE_RECORD = base._record
_BASE_GUARD = base.guarded_execute


def choose_centers(orders, anchor, *, count=10, half=HALF_L2 / np.sqrt(2),
                   grid_km=.5, min_orders=1000):
    """Select distinct windows by geography, without looking at solver outcomes.

    Candidates are occupied half-kilometre pickup grid cells. Require at least
    ``min_orders`` observed noon orders with both endpoints in the smaller L=1
    window. Starting at the historical center, greedy farthest-point selection
    spreads the remaining centers; ties use demand count then coordinates.
    Windows may overlap. Identical centers are never replicated as new maps.
    """
    if count < 1 or half <= 0 or grid_km <= 0 or min_orders < 1:
        raise ValueError("Positive map count, geometry and sample threshold required")
    demand = orders.loc[orders.hour.between(12, 13, inclusive="left")]
    endpoints = demand[["ox", "oy", "dx", "dy"]].to_numpy(float)
    anchor = np.asarray(anchor, dtype=float)
    if anchor.shape != (2,) or not np.isfinite(anchor).all():
        raise ValueError("Invalid historical center")
    grid = np.unique(np.round(endpoints[:, :2] / grid_km) * grid_km, axis=0)
    candidates = []
    for center in grid:
        inside = np.all(np.abs(endpoints - np.tile(center, 2)) <= half, axis=1)
        size = int(inside.sum())
        if size >= min_orders and np.unique(endpoints[inside], axis=0).shape[0] >= 50:
            candidates.append((center, size))
    anchor_count = int(np.all(np.abs(endpoints - np.tile(anchor, 2)) <= half, axis=1).sum())
    if anchor_count < min_orders:
        raise ValueError("Historical center fails the common minimum observation requirement")
    chosen = [(anchor, anchor_count)]
    candidates = [(center, size) for center, size in candidates
                  if not np.array_equal(center, anchor)]
    while len(chosen) < count:
        if not candidates:
            raise ValueError("Insufficient distinct observed map windows")
        best = max(range(len(candidates)), key=lambda i: (
            min(float(np.square(candidates[i][0] - c).sum()) for c, _ in chosen),
            candidates[i][1], -float(candidates[i][0][0]), -float(candidates[i][0][1])))
        chosen.append(candidates.pop(best))
    return [dict(map_id=f"M{i:02d}", center=center.tolist(), l1_noon_orders=size,
                 historical_center=i == 1) for i, (center, size) in enumerate(chosen, 1)]


def generate(output=DEFAULT_MAPS, *, maps_per_group=10, config_path=CONFIG):
    """Build 2 datasets x 2 sizes x map count; never invoke a solver."""
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to replace map set: {output}")
    if ROOT / "outputs" not in output.parents:
        raise ValueError("Store new benchmark maps under the repository outputs directory")
    cfg = load_config(config_path)
    anchor_metadata = base.read(TSL_ANCHORS[2] / "metadata.json")
    for section in ("depreciation", "operations", "terminal", "compatibility", "choice", "costs"):
        if cfg[section] != anchor_metadata["frozen_input_metadata"]["resolved_config"][section]:
            raise ValueError(f"Configuration differs from historical TSL anchor: {section}")
    output.mkdir(parents=True)
    manifest = dict(schema="unified-real-maps/1", status="preparing", solver_invoked=False,
        maps_per_dataset_and_size=maps_per_group, datasets=["tsl", "nanshan"], levels=[1, 2],
        physical_instances=4 * maps_per_group, expected_solve_jobs=16 * maps_per_group,
        config_path=str(Path(config_path).resolve()), config_sha256=base.sha256(config_path),
        preparation_code_sha256=base.sha256(Path(__file__)),
        map_selection=dict(grid_km=.5, minimum_noon_orders_in_l1=1000,
            rule="historical center first, then greedy farthest eligible observed pickup-grid centers",
            crop_rule="both pickup and delivery endpoints inside the square",
            overlap_allowed=True, same_centers_for_l1_and_l2=True,
            demand_hours=[12, 13], candidate_hours=[8, 24], seed=43),
        sources={}, instances=[])
    write_json(output / "preparation_manifest.json", manifest)
    for dataset in ("tsl", "nanshan"):
        source_metadata = base.read(MARKETS[dataset] / "metadata.json")
        prepared = read_orders(dataset, path=source_metadata["raw_path"])
        if prepared[1]["raw_file_sha256"] != source_metadata["raw_file_sha256"]:
            raise ValueError(f"Original real-order source changed for {dataset}")
        centers = choose_centers(prepared[0], source_metadata["crop_center_local_km"], count=maps_per_group)
        manifest["sources"][dataset] = dict(raw_path=source_metadata["raw_path"],
            raw_file_sha256=source_metadata["raw_file_sha256"], selected_days=source_metadata["selected_days"],
            map_centers=centers)
        write_json(output / "preparation_manifest.json", manifest)
        for map_spec in centers:
            for n_l in (1, 2):
                start = time.monotonic()
                identity = f"{dataset}_L{n_l:02d}_{map_spec['map_id']}"
                target = output / "instances" / identity
                provenance = output / "provenance" / identity
                spec = dict(id=identity, dataset=dataset, map_id=map_spec["map_id"],
                    n_l=n_l, n_h=10 * n_l, n_od=25 * n_l,
                    crop_center_local_km=map_spec["center"],
                    crop_half_size_km=float(HALF_L2 * np.sqrt(n_l / 2)),
                    historical_center=map_spec["historical_center"],
                    exact_historical_instance=dataset == "tsl" and map_spec["historical_center"])
                with tempfile.TemporaryDirectory(prefix=".map-prepare-", dir=output) as temporary:
                    temporary = Path(temporary)
                    staged = temporary / "standard"
                    if spec["exact_historical_instance"]:
                        original = TSL_ANCHORS[n_l]
                        shutil.copytree(original, staged)
                        if n_l == 1:
                            market = ROOT / "outputs/tsl_scale_ladder/provenance/L01"
                        else:
                            market = MARKETS["tsl"]
                        for name in (*INPUT_FILES, "manifest.json"):
                            if base.sha256(staged / name) != base.sha256(original / name):
                                raise AssertionError("Historical instance payload changed")
                    else:
                        market = temporary / "market"
                        scenario = deepcopy(cfg)
                        scenario["name"] = identity
                        scenario["data"]["input_path"] = str(target)
                        prepare_orders(dataset, market, n_l=n_l, n_h=spec["n_h"], n_od=spec["n_od"],
                            raw_path=source_metadata["raw_path"], dates=source_metadata["selected_days"],
                            demand_hours=(12, 13), candidate_hours=(8, 24), center=map_spec["center"],
                            half_size_km=spec["crop_half_size_km"], seed=43, config=scenario,
                            prepared_orders=prepared)
                        export_standard(market, market, staged)
                    expected = dict(anchor_metadata, **{key: source_metadata[key] for key in
                        ("selected_days", "demand_hour_window", "candidate_hour_window", "raw_file_sha256")})
                    checked = _validate_level(staged, spec, expected)
                    membership = _membership_audit(market / "order_membership.csv.gz", staged)
                    provenance.mkdir(parents=True)
                    for name in ("order_membership.csv.gz", "calibration.json"):
                        shutil.copy2(market / name, provenance / name)
                    note = dict(**spec, solver_invoked=False, seed=43, membership_audit=membership,
                        historical_input=str(TSL_ANCHORS[n_l]) if spec["exact_historical_instance"] else None,
                        site_policy="exact historical CSV/JSON copy" if spec["exact_historical_instance"] else
                            "weighted observed-site clustering independently within each map and size",
                        historical_nanshan_note="Historical map geometry reused; 20 lockers freshly clustered instead of the former 10" if dataset == "nanshan" and map_spec["historical_center"] and n_l == 2 else None,
                        source_sha256=source_metadata["raw_file_sha256"],
                        files={name: base.sha256(provenance / name) for name in ("order_membership.csv.gz", "calibration.json")})
                    write_json(provenance / "preparation.json", note)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    staged.rename(target)
                metadata = base.read(target / "metadata.json")
                manifest["instances"].append(dict(**spec, instance=str(target.relative_to(output)),
                    provenance=str(provenance.relative_to(output)), manifest_sha256=base.sha256(target / "manifest.json"),
                    demand_orders=metadata["demand_orders_in_area"], candidate_orders=metadata["candidate_orders_in_area"],
                    hourly_demand=checked["demand_per_hour"], nonzero_kappa=checked["nonzero_kappa"],
                    preparation_seconds=time.monotonic() - start, status="prepared_validated_not_solved"))
                write_json(output / "preparation_manifest.json", manifest)
                print(json.dumps(dict(event="map_prepared", instance=identity,
                    completed=len(manifest["instances"]), total=manifest["physical_instances"])), flush=True)
    manifest["status"] = "prepared_validated_not_solved"
    write_json(output / "preparation_manifest.json", manifest)
    return dict(output=str(output), instances=len(manifest["instances"]), solver_started=False)


def jobs_for(origins):
    return [dict(id=f"{origin['id']}_{mode}_{method}", instance=origin["instance"],
                 dataset=origin["dataset"], map_id=origin["map_id"], n_l=origin["n_l"],
                 K=k, task_mode=mode, method=method,
                 output=f"runs/{origin['id']}/{mode}/{method}")
            for origin in origins for k, mode in ((1, "single"), (2, "double")) for method in METHODS]


def command_for(batch, manifest, job, attempt):
    batch = Path(batch)
    cfg = manifest["protocol"]
    method = "full_path" if job["method"] == "full_cplex" else "benders_cg"
    command = [str(batch / "frozen/bin/mlnd_cpp"), "--input", str(batch / job["instance"]),
        "--method", method, "--k", str(job["K"]), "--demand-formulation", "preprocessed",
        "--seconds", str(cfg["seconds_per_job"]), "--threads", str(cfg["threads_per_job"]),
        "--seed", str(cfg["seed"]), "--gap", str(cfg["relative_gap_target"]),
        "--path-column-limit", "0", "--log", "--out", str(attempt),
        *cfg["common_solver_args"]]
    if method == "benders_cg":
        command += ["--cg-max-columns", str(cfg["cg_max_columns_per_iteration"]),
                    "--max-iterations", str(cfg["max_benders_iterations"]), *cfg["algorithm_solver_args"]]
    return command


def prepare(batch=DEFAULT_BATCH, *, maps=DEFAULT_MAPS, binary=None, seconds=3600,
            threads=1, parallel=4, gap=.01, seed=43, job_rss_gib=4,
            batch_rss_gib=12, system_available_gib=2, hard_wall_seconds=None,
            common_solver_args=COMMON_FLAGS, algorithm_solver_args=ALGORITHM_FLAGS, instances=None):
    maps = Path(maps).resolve()
    prepared = base.read(maps / "preparation_manifest.json")
    if prepared["status"] != "prepared_validated_not_solved":
        raise ValueError("Complete and validate map preparation before freezing the experiment")
    selected = prepared["instances"]
    if instances is not None:
        wanted = set(instances)
        selected = [item for item in selected if item["id"] in wanted]
        if {item["id"] for item in selected} != wanted:
            raise ValueError("Unknown requested map identity")
    for item in selected:
        if base.sha256(maps / item["instance"] / "manifest.json") != item["manifest_sha256"]:
            raise ValueError("Prepared instance changed")
    binary = binary or ROOT / "cpp/build-unified/mlnd_cpp"
    base.prepare(batch, [maps / item["instance"] for item in selected], binary=binary,
        seconds=seconds, threads=threads, parallel=parallel, gap=gap, seed=seed,
        job_rss_gib=job_rss_gib, batch_rss_gib=batch_rss_gib,
        system_available_gib=system_available_gib, hard_wall_seconds=hard_wall_seconds)
    batch = Path(batch).resolve()
    manifest = base.read(batch / "experiment_manifest.json")
    lookup = {item["id"]: item for item in selected}
    for origin in manifest["input_origins"]:
        item = lookup[origin["id"]]
        origin.update({key: item[key] for key in ("dataset", "map_id", "n_l", "n_h", "n_od", "hourly_demand")})
        provenance = batch / "provenance" / origin["id"]
        shutil.copytree(maps / item["provenance"], provenance)
        for path in provenance.rglob("*"):
            if path.is_file():
                manifest["frozen_sha256"][str(path.relative_to(batch))] = base.sha256(path)
    shutil.copy2(maps / "preparation_manifest.json", batch / "map_preparation_manifest.json")
    manifest["frozen_sha256"]["map_preparation_manifest.json"] = base.sha256(batch / "map_preparation_manifest.json")
    manifest["schema"] = "unified-map-benchmark/1"
    manifest["jobs"] = jobs_for(manifest["input_origins"])
    manifest["protocol"].update(max_orders=[1, 2], task_modes=["single", "double"],
        demand_formulation="preprocessed", flow_policy="aggregate", solver="CPLEX",
        common_solver_args=list(common_solver_args), algorithm_solver_args=list(algorithm_solver_args),
        physical_instances=len(selected), total_jobs=len(manifest["jobs"]),
        same_binary_for_all_jobs=True, map_selection_uses_solver_results=False,
        catalogue_policy="full preprocessing for both methods; all limits and stages within solve budget",
        root_initialization="inside each job budget; no external root certificate or warm start")
    manifest["binary_sha256"] = base.sha256(batch / "frozen/bin/mlnd_cpp")
    manifest["runtime"] = dict(platform=platform.platform(), python=platform.python_version(),
        packages={name: importlib.metadata.version(name) for name in ("numpy", "scipy", "pandas", "psutil")})
    write_json(batch / "experiment_manifest.json", manifest)
    write_json(batch / "state.json", dict(status="prepared_not_started", updated_utc=base.existing.utc(),
        jobs={job["id"]: dict(status="pending", attempts=[]) for job in manifest["jobs"]}))
    (batch / "run.py").write_text("from pathlib import Path\nimport sys\n"
        "batch = Path(__file__).resolve().parent\nsys.path.insert(0, str(batch / 'frozen'))\n"
        f"from {MODULE} import main\nmain([*sys.argv[1:], '--batch', str(batch)])\n", encoding="utf-8")
    manifest["frozen_sha256"]["run.py"] = base.sha256(batch / "run.py")
    write_json(batch / "experiment_manifest.json", manifest)
    return dict(batch=str(batch), instances=len(selected), jobs=len(manifest["jobs"]), solver_started=False)


def _guarded_execute(batch, manifest, job_id, command, attempt, cancel, **kwargs):
    command = list(command)
    if "-m" in command:
        index = command.index("-m") + 1
        if command[index] == base.__name__:
            command[index] = MODULE
    return _BASE_GUARD(batch, manifest, job_id, command, attempt, cancel, **kwargs)


def record(batch, job, state):
    batch = Path(batch)
    row = _BASE_RECORD(batch, job, state)
    row.update({key: job[key] for key in ("K", "task_mode", "dataset", "map_id", "n_l")})
    source = batch / state.get("result_directory", state.get("directory", "_missing"))
    result = base.optional_read(source / "result.json")
    if not result:
        return row
    checked = base.optional_read(source / "native_validation.json")
    valid = bool(checked.get("passed"))
    for name, key in (("result.json", "result_sha256"), ("solution.json", "solution_sha256"),
                      ("path_flows.json", "path_flows_sha256")):
        valid = valid and (source / name).is_file() and checked.get(key) == base.sha256(source / name)
    valid = valid and checked.get("manifest_sha256") == base.sha256(batch / job["instance"] / "manifest.json")
    manifest = base.read(batch / "experiment_manifest.json")
    valid = valid and result.get("binary_sha256") == manifest["binary_sha256"]
    valid = valid and result.get("max_orders") == job["K"] and result.get("solver") == "CPLEX"
    valid = valid and result.get("demand_formulation") == "preprocessed"
    valid = valid and result.get("method") == ("full_path" if job["method"] == "full_cplex" else "benders_cg")
    row.update(validation_passed=bool(valid), upper_bound=result.get("objective") if valid else None,
        lower_bound=base.certified_bound(result) if valid else None, gap=None,
        binary_sha256=result.get("binary_sha256"),
        generated_columns_scope=result.get("generated_columns_scope"),
        path_storage=result.get("path_storage"), pricing_domain=result.get("pricing_domain"))
    if row["upper_bound"] is not None and row["lower_bound"] is not None:
        row["gap"] = (row["upper_bound"] - row["lower_bound"]) / max(1, abs(row["upper_bound"]))
    for key in ("preprocessed_demand", "preprocessed_statistics", "structural_strengthening", "accelerations",
                "acceleration_statistics", "continuous_root", "root_diagnostics"):
        value = result.get(key)
        if isinstance(value, dict):
            row[key + "_json"] = json.dumps(value, sort_keys=True)
            for name, scalar in value.items():
                if not isinstance(scalar, (dict, list)):
                    row[key + "_" + name] = scalar
    operation = base.optional_read(source / "operations/operations_summary.json")
    for key, value in operation.items():
        row["operations_" + key] = json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value
    return row


def report(batch):
    batch = Path(batch).resolve()
    manifest = base.verify(batch)
    state = base.read(batch / "state.json")
    rows = [record(batch, job, state["jobs"][job["id"]]) for job in manifest["jobs"]]
    # Single and double catalogues are different, even for identical physical input.
    counts = {}
    for row in rows:
        if row["all_paths"] is not None:
            key = row["instance"], row["K"]
            previous = counts.get(key)
            if previous and (previous["all_service_paths"], previous["all_empty_paths"]) != (row["all_service_paths"], row["all_empty_paths"]):
                raise ValueError(f"Complete catalogues disagree for {key}")
            counts[key] = row
    for row in rows:
        reference = counts.get((row["instance"], row["K"]))
        if row["all_paths"] is None and reference:
            for key in ("all_paths", "all_service_paths", "all_empty_paths", "initial_segments"):
                row[key] = reference.get(key)
            row["full_path_count_source"] = reference["job"]
    output = batch / "analysis"
    output.mkdir(exist_ok=True)
    write_json(output / "comparison.json", dict(protocol=manifest["protocol"], rows=rows))
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (output / "comparison.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    fmt = lambda value: "—" if value is None else f"{value:.6g}" if isinstance(value, float) else str(value)
    lines = ["# 真实地图统一任务模型比较", "",
        f"{manifest['protocol']['physical_instances']} 个物理算例，{len(rows)} 个任务；单单/双单 × CPLEX 全路径/算法。",
        f"每任务 {manifest['protocol']['seconds_per_job']} 秒，gap {manifest['protocol']['relative_gap_target']}，"
        f"单任务 {manifest['protocol']['threads_per_job']} 线程，并发 {manifest['protocol']['parallel_jobs']}。", "",
        "UB 仅显示独立验证及输入、结果、二进制身份检查通过的可行解；LB 仅显示全局有效界。限时可行解不表示最优。",
        "资源超限和构造失败单独保留，不视作数学不可行。地图可重叠；原始订单未经缩放。", "",
        "| 地图 | 模式 | 方法 | UB | LB | gap | 求解秒 | 终止 | 验证 |",
        "|---|---|---|---:|---:|---:|---:|---|---|"]
    for row in rows:
        lines.append("| " + " | ".join(fmt(value) for value in (Path(row["instance"]).name,
            row["task_mode"], row["method"], row["upper_bound"], row["lower_bound"], row["gap"],
            row.get("solve_seconds"), row.get("termination_reason"), row["validation_passed"])) + " |")
    lines += ["", "完整的变量、线性约束、锥约束、全路径数、已生成列数、阶段耗时及运营指标见 comparison.csv/JSON。",
        "原始解、路径流、逐订单/基地/柜子/航线运营明细、检查点、独立验证、资源采样和求解日志保存在 runs 下。",
        "全路径数只在同一输入、同一种任务模式间引用；不同任务模式不混用目录规模。"]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return dict(report=str(output / "REPORT.md"), jobs=len(rows), validated=sum(row["validation_passed"] for row in rows))


def stop(batch):
    """Request graceful cancellation only of this batch's recorded dispatcher.

    PID reuse and command identity are checked before SIGTERM. The established
    dispatcher propagates cancellation to each guarded process group, preserves
    attempt artifacts, and leaves unstarted jobs pending. This is not a global
    solver-process kill and does not launch any validation or optimization.
    """
    import psutil

    batch = Path(batch).resolve()
    identity = base.optional_read(batch / "dispatcher.json")
    if not identity.get("pid"):
        raise ValueError("No detached dispatcher identity is recorded for this batch")
    try:
        process = psutil.Process(identity["pid"])
        if process.create_time() != identity.get("create_time"):
            raise RuntimeError("Dispatcher PID has been reused; refusing to signal it")
        if str(batch / "run.py") not in process.cmdline():
            raise RuntimeError("Recorded process does not run this batch's frozen entrypoint")
        process.send_signal(signal.SIGTERM)
    except psutil.NoSuchProcess:
        return dict(batch=str(batch), dispatcher_running=False, stop_requested=False)
    record = dict(batch=str(batch), pid=identity["pid"], create_time=identity["create_time"],
                  requested_utc=base.existing.utc(), signal="SIGTERM", stop_requested=True)
    write_json(batch / "stop_request.json", record)
    return record


@contextmanager
def adapters():
    replacement = dict(command_for=command_for, _record=record, guarded_execute=_guarded_execute, report=report)
    previous = {name: getattr(base, name) for name in replacement}
    try:
        for name, value in replacement.items():
            setattr(base, name, value)
        yield
    finally:
        for name, value in previous.items():
            setattr(base, name, value)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("generate", "prepare", "verify", "launch", "run", "stop", "report", "status", "audit-worker"))
    parser.add_argument("--batch", type=Path, default=DEFAULT_BATCH)
    parser.add_argument("--maps", type=Path, default=DEFAULT_MAPS)
    parser.add_argument("--maps-per-group", type=int, default=10)
    parser.add_argument("--instances", nargs="+")
    parser.add_argument("--binary", type=Path)
    parser.add_argument("--seconds", type=float, default=3600)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--parallel", type=int, default=4)
    parser.add_argument("--gap", type=float, default=.01)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--job-rss-gib", type=float, default=4)
    parser.add_argument("--batch-rss-gib", type=float, default=12)
    parser.add_argument("--system-available-gib", type=float, default=2)
    parser.add_argument("--hard-wall-seconds", type=float)
    parser.add_argument("--common-args-json", default=json.dumps(COMMON_FLAGS))
    parser.add_argument("--algorithm-args-json", default=json.dumps(ALGORITHM_FLAGS))
    parser.add_argument("--jobs", nargs="+")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--job-id")
    parser.add_argument("--attempt", type=Path)
    args = parser.parse_args(argv)
    with adapters():
        if args.action == "generate":
            result = generate(args.maps, maps_per_group=args.maps_per_group)
        elif args.action == "prepare":
            result = prepare(args.batch, maps=args.maps, binary=args.binary, seconds=args.seconds,
                threads=args.threads, parallel=args.parallel, gap=args.gap, seed=args.seed,
                job_rss_gib=args.job_rss_gib, batch_rss_gib=args.batch_rss_gib,
                system_available_gib=args.system_available_gib, hard_wall_seconds=args.hard_wall_seconds,
                common_solver_args=json.loads(args.common_args_json),
                algorithm_solver_args=json.loads(args.algorithm_args_json), instances=args.instances)
        elif args.action == "verify":
            result = dict(passed=bool(base.verify(args.batch)), solver_started=False)
        elif args.action == "launch":
            result = base.launch(args.batch)
        elif args.action == "run":
            result = base.run(args.batch, only=args.jobs, retry_failed=args.retry_failed)
        elif args.action == "stop":
            result = stop(args.batch)
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
