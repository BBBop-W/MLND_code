"""Prepare the 50 km2 TSL baseline and freeze its two guarded comparisons.

Both commands are preparation only. Run the frozen batch's ``run.py`` explicitly
to start the sequential full-path and Benders comparison.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
import shutil
import tempfile
import time

from src.config import load_config
from src.data.prepare import prepare_orders
from src.data.scale_ladder import _membership_audit, _validate_level
from src.data.standard import export_standard
from src.experiments import unified_benchmark_20261027 as unified
from src.experiments.storage import write_json

base = unified.base
ROOT = base.ROOT
IDENTITY = "tsl_A50_L04_H040_OD100"
DEFAULT_MAPS = ROOT / "outputs/baseline_tsl_A50_L4_H40_OD100_20261027_inputs"
DEFAULT_BATCH = ROOT / "outputs/baseline_tsl_A50_L4_H40_OD100_20261027"
COMMON_FLAGS = (*unified.COMMON_FLAGS, "--task-airline-bounds")
ALGORITHM_FLAGS = (*unified.ALGORITHM_FLAGS, "--path-support-cuts",
                   "--task-savings-bound", "--task-resource-cuts")


def generate(output=DEFAULT_MAPS, *, config_path=unified.CONFIG, n_od=100):
    """Reuse the historical source scope and validate a newly clustered map."""
    if isinstance(n_od, bool) or not isinstance(n_od, int) or n_od <= 0:
        raise ValueError("n_od must be a positive integer")
    identity = f"tsl_A50_L04_H040_OD{n_od:03d}"
    output = Path(output).resolve()
    config_path = Path(config_path).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to replace prepared inputs: {output}")
    if ROOT / "outputs" not in output.parents:
        raise ValueError("Store baseline inputs under repository outputs")
    cfg = load_config(config_path)
    anchor = base.read(unified.TSL_ANCHORS[2] / "metadata.json")
    source = base.read(unified.MARKETS["tsl"] / "metadata.json")
    raw_path = Path(source["raw_path"])
    if base.sha256(raw_path) != source["raw_file_sha256"]:
        raise ValueError("Historical TSL raw source changed")
    for section in ("depreciation", "operations", "terminal", "compatibility", "choice", "costs"):
        if cfg[section] != anchor["frozen_input_metadata"]["resolved_config"][section]:
            raise ValueError(f"Configuration differs from historical TSL anchor: {section}")
    half = math.sqrt(50.0) / 2
    center = [-.75, 3.25]
    spec = dict(id=identity, dataset="tsl", map_id="A50", n_l=4, n_h=40, n_od=n_od,
        crop_center_local_km=center, crop_half_size_km=half,
        crop_side_km=2 * half, crop_area_km2=50.0,
        candidate_h_density_per_km2=.8, candidate_l_density_per_km2=.08,
        crop_bounds_local_km=dict(x_min=center[0]-half, x_max=center[0]+half,
                                  y_min=center[1]-half, y_max=center[1]+half))
    output.mkdir(parents=True)
    manifest = dict(schema="tsl-baseline-capacity-inputs/1", status="preparing",
        solver_invoked=False, physical_instances=1, expected_solve_jobs=2,
        config_path=str(config_path), config_sha256=base.sha256(config_path),
        preparation_code_sha256=base.sha256(Path(__file__)),
        map_selection=dict(rule="user-specified square at historical TSL center",
            crop_rule="both pickup and delivery endpoints inside the square",
            demand_hours=[12, 13], candidate_hours=[8, 24], seed=43),
        sources={"tsl": {key: source[key] for key in
            ("raw_path", "raw_file_sha256", "selected_days")}}, instances=[])
    write_json(output / "preparation_manifest.json", manifest)
    started = time.monotonic()
    target = output / "instances" / identity
    provenance = output / "provenance" / identity
    with tempfile.TemporaryDirectory(prefix=".baseline-prepare-", dir=output) as temp:
        temp = Path(temp)
        market, staged = temp / "market", temp / "standard"
        scenario = deepcopy(cfg)
        scenario["name"] = identity
        scenario["data"]["input_path"] = str(target)
        prepare_orders("tsl", market, n_l=4, n_h=40, n_od=n_od,
            raw_path=raw_path, dates=source["selected_days"],
            demand_hours=(12, 13), candidate_hours=(8, 24), center=center,
            half_size_km=half, seed=43, config=scenario)
        export_standard(market, market, staged)
        expected = dict(anchor, **{key: source[key] for key in
            ("selected_days", "demand_hour_window", "candidate_hour_window", "raw_file_sha256")})
        checked = _validate_level(staged, spec, expected)
        membership = _membership_audit(market / "order_membership.csv.gz", staged)
        provenance.mkdir(parents=True)
        for name in ("order_membership.csv.gz", "calibration.json", "metadata.json", "manifest.json"):
            shutil.copy2(market / name, provenance / name)
        shutil.copy2(config_path, provenance / "source_config.json")
        write_json(provenance / "resolved_config.json", scenario)
        write_json(provenance / "membership_audit.json", membership)
        write_json(provenance / "input_validation.json", checked)
        files = {str(p.relative_to(provenance)): base.sha256(p)
                 for p in sorted(provenance.rglob("*")) if p.is_file()}
        write_json(provenance / "preparation.json", dict(**spec, seed=43,
            solver_invoked=False, source_sha256=source["raw_file_sha256"],
            site_policy="weighted observed-site clustering within the specified map",
            membership_audit=membership, files=files))
        target.parent.mkdir(parents=True)
        staged.rename(target)
    metadata = base.read(target / "metadata.json")
    manifest["instances"].append(dict(**spec, instance=str(target.relative_to(output)),
        provenance=str(provenance.relative_to(output)),
        manifest_sha256=base.sha256(target / "manifest.json"),
        demand_orders=metadata["demand_orders_in_area"],
        candidate_orders=metadata["candidate_orders_in_area"],
        hourly_demand=checked["demand_per_hour"], nonzero_kappa=checked["nonzero_kappa"],
        preparation_seconds=time.monotonic()-started, status="prepared_validated_not_solved"))
    manifest["status"] = "prepared_validated_not_solved"
    write_json(output / "preparation_manifest.json", manifest)
    return dict(output=str(output), instances=1, spec=spec,
                hourly_demand=checked["demand_per_hour"], solver_started=False)


def prepare(batch=DEFAULT_BATCH, *, maps=DEFAULT_MAPS, binary=None,
            seconds=600, hard_wall_seconds=None,
            job_rss_gib=16, batch_rss_gib=16, system_available_gib=2,
            source_snapshot=None):
    """Freeze current or explicitly selected prior sources for the two K2 jobs."""
    maps = Path(maps).resolve()
    prepared = base.read(maps / "preparation_manifest.json")
    instances = prepared.get("instances", [])
    if len(instances) != 1:
        raise ValueError("A baseline comparison requires exactly one prepared map")
    instance = instances[0]
    if (instance.get("dataset") != "tsl" or instance.get("n_l") != 4
            or instance.get("n_h") != 40
            or not math.isclose(instance.get("crop_area_km2", -1), 50.0)):
        raise ValueError("Expected the TSL 50 km2 / 4L / 40H baseline geometry")
    snapshot = None if source_snapshot is None else Path(source_snapshot).resolve()
    snapshot_manifest = None
    if snapshot is not None:
        for name in ("src", "configs", "cpp"):
            if not (snapshot / name).is_dir():
                raise FileNotFoundError(f"Incomplete source snapshot: {snapshot / name}")
        for name in ("main.py", "pyproject.toml", "bin/mlnd_cpp"):
            if not (snapshot / name).is_file():
                raise FileNotFoundError(f"Incomplete source snapshot: {snapshot / name}")
        binary = snapshot / "bin/mlnd_cpp" if binary is None else Path(binary).resolve()
        if base.sha256(binary) != base.sha256(snapshot / "bin/mlnd_cpp"):
            raise ValueError("The executable must match the selected source snapshot")
        candidate = snapshot.parent / "experiment_manifest.json"
        if candidate.is_file():
            snapshot_manifest = candidate
    previous_root = base.ROOT
    try:
        if snapshot is not None:
            base.ROOT = snapshot
        unified.prepare(batch, maps=maps, binary=binary, seconds=seconds, threads=1,
            parallel=1, gap=.01, seed=43, job_rss_gib=job_rss_gib,
            batch_rss_gib=batch_rss_gib, system_available_gib=system_available_gib,
            hard_wall_seconds=hard_wall_seconds, common_solver_args=COMMON_FLAGS,
            algorithm_solver_args=ALGORITHM_FLAGS, instances=[instance["id"]])
    finally:
        base.ROOT = previous_root
    batch = Path(batch).resolve()
    manifest = base.read(batch / "experiment_manifest.json")
    manifest["jobs"] = [job for job in manifest["jobs"] if job["K"] == 2]
    if [job["method"] for job in manifest["jobs"]] != ["full_cplex", "benders_unified"]:
        raise AssertionError("Expected exactly the full-path and Benders K2 jobs")
    manifest["schema"] = "tsl-baseline-capacity-benchmark/1"
    if snapshot is not None:
        manifest["source_snapshot"] = dict(path=str(snapshot),
            binary_sha256=base.sha256(snapshot / "bin/mlnd_cpp"),
            comparison_control="Prior frozen source and executable; only declared input aggregation and solve budget change")
        if snapshot_manifest is not None:
            previous = base.read(snapshot_manifest)
            manifest["source_snapshot"].update(experiment_manifest=str(snapshot_manifest),
                experiment_manifest_sha256=base.sha256(snapshot_manifest))
            manifest["preparation_checkout"] = {
                key: manifest.get(key) for key in ("git_head", "git_status")}
            for key in ("git_head", "git_status"):
                if key in previous:
                    manifest[key] = previous[key]
    manifest["protocol"].update(max_orders=[2], task_modes=["double"],
        total_jobs=2, physical_instances=1, execution_order="full_cplex then benders_unified")
    write_json(batch / "experiment_manifest.json", manifest)
    write_json(batch / "state.json", dict(status="prepared_not_started",
        updated_utc=base.existing.utc(),
        jobs={job["id"]: dict(status="pending", attempts=[]) for job in manifest["jobs"]}))
    base.verify(batch)
    return dict(batch=str(batch), jobs=[job["id"] for job in manifest["jobs"]],
        binary_sha256=manifest["binary_sha256"], solver_started=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("generate", "prepare"))
    parser.add_argument("--maps", type=Path, default=DEFAULT_MAPS)
    parser.add_argument("--batch", type=Path, default=DEFAULT_BATCH)
    parser.add_argument("--config", type=Path, default=unified.CONFIG)
    parser.add_argument("--binary", type=Path)
    parser.add_argument("--n-od", type=int, default=100,
                        help="OD aggregation count for generate; area and facility candidates remain fixed")
    parser.add_argument("--source-snapshot", type=Path,
                        help="Prior complete frozen directory to reuse for a controlled comparison")
    parser.add_argument("--seconds", type=float, default=600,
                        help="Per-job preparation and optimization budget (default: 600)")
    parser.add_argument("--hard-wall-seconds", type=float,
                        help="Per-job external guard including finalization; default: seconds + 120")
    parser.add_argument("--job-rss-gib", type=float, default=16)
    parser.add_argument("--batch-rss-gib", type=float, default=16)
    parser.add_argument("--system-available-gib", type=float, default=2)
    args = parser.parse_args(argv)
    if args.action == "generate":
        result = generate(args.maps, config_path=args.config, n_od=args.n_od)
    else:
        result = prepare(args.batch, maps=args.maps, binary=args.binary,
            seconds=args.seconds, hard_wall_seconds=args.hard_wall_seconds,
            job_rss_gib=args.job_rss_gib, batch_rss_gib=args.batch_rss_gib,
            system_available_gib=args.system_available_gib,
            source_snapshot=args.source_snapshot)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
