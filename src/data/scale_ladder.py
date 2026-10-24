"""Prepare the real-TSL size ladder without constructing or solving a model.

The frozen 2L/20H/50OD exchange instance is the exact anchor. Other levels use
the same dates, hour windows, economic parameters, square center and observed
order processing, with constant candidate-H density and area proportional to L.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil
import tempfile
import time

import numpy as np
import pandas as pd

from src.config import ROOT, load_config
from src.data.prepare import prepare_orders
from src.data.performance_orders import read_orders
from src.data.standard import INPUT_FILES, export_standard, load_standard, validate_standard
from src.experiments.storage import sha256, write_json

DEFAULT_ANCHOR = ROOT / "examples/tsl_2l20h50od"
DEFAULT_OUTPUT = ROOT / "outputs/tsl_scale_ladder"
DEFAULT_CONFIG = ROOT / "configs/tsl_20h_benchmark.json"
ANCHOR_MARKET = ROOT / "outputs/consecutive_real_comparison_20261015/instances/tsl"
SCHEMA = "tsl-scale-ladder/1"


def ladder_specs(anchor=DEFAULT_ANCHOR, levels=range(1, 11)):
    """Return verified level geometry and the unmodified anchor parameters."""
    anchor = Path(anchor).resolve()
    data = load_standard(anchor)
    if (data["n_l"], data["n_h"], len(data["demand"])) != (2, 20, 50):
        raise ValueError("The scale anchor must have exactly 2L, 20H and 50OD")
    metadata = json.loads((anchor / "metadata.json").read_text())
    if metadata["source"] != "tsl":
        raise ValueError("The scale ladder requires the frozen real TSL anchor")
    if len(metadata["selected_days"]) != 8:
        raise ValueError("The TSL anchor must contain the same eight selected dates")
    if metadata["demand_hour_window"] != [12, 13] or metadata["candidate_hour_window"] != [8, 24]:
        raise ValueError("The TSL ladder requires demand 12–13 and candidates 8–24")
    half = float(metadata["crop_half_size_km"])
    center = np.asarray(metadata["crop_center_local_km"], dtype=float)
    if half <= 0 or not np.isfinite(half) or center.shape != (2,) or not np.isfinite(center).all():
        raise ValueError("Invalid anchor crop geometry")
    levels = list(levels)
    if not levels or len(set(levels)) != len(levels):
        raise ValueError("Levels must be nonempty and unique")
    if any(isinstance(n, bool) or not isinstance(n, int) or not 1 <= n <= 10 for n in levels):
        raise ValueError("Levels must be integers from 1 through 10")
    specs = []
    for n in sorted(levels):
        radius = half * np.sqrt(n / 2)
        area = (2 * radius) ** 2
        specs.append(
            dict(
                level=n,
                instance_id=f"L{n:02d}",
                n_l=n,
                n_h=10 * n,
                n_od=25 * n,
                exact_frozen_anchor=n == 2,
                crop_center_local_km=center.tolist(),
                crop_half_size_km=float(radius),
                crop_side_km=float(2 * radius),
                crop_area_km2=float(area),
                candidate_h_density_per_km2=float(10 * n / area),
                crop_bounds_local_km=dict(
                    x_min=float(center[0] - radius), x_max=float(center[0] + radius),
                    y_min=float(center[1] - radius), y_max=float(center[1] + radius),
                ),
            )
        )
    return specs, metadata


def _validate_level(path, spec, anchor_metadata):
    """Check lossless interchange, demand accounting and the crop contract."""
    report = validate_standard(path)
    data = load_standard(path)
    metadata = json.loads((Path(path) / "metadata.json").read_text())
    if (data["n_l"], data["n_h"], len(data["demand"])) != (spec["n_l"], spec["n_h"], spec["n_od"]):
        raise ValueError("Prepared dimensions do not match the scale plan")
    for field in ("selected_days", "demand_hour_window", "candidate_hour_window", "raw_file_sha256"):
        if metadata[field] != anchor_metadata[field]:
            raise ValueError(f"Prepared source scope differs from anchor: {field}")
    if not np.allclose(metadata["crop_center_local_km"], spec["crop_center_local_km"], rtol=0, atol=0):
        raise ValueError("Prepared crop center differs from the plan")
    if metadata["crop_half_size_km"] != spec["crop_half_size_km"]:
        raise ValueError("Prepared crop half-side differs from the plan")
    for key in ("bases", "hubs", "origins", "destinations"):
        distance = np.abs(data[key] - np.asarray(spec["crop_center_local_km"]))
        if np.any(distance > spec["crop_half_size_km"] + 1e-12):
            raise ValueError(f"Prepared {key} lie outside the square crop")
    before = anchor_metadata["frozen_input_metadata"]["resolved_config"]
    after = metadata["frozen_input_metadata"]["resolved_config"]
    for section in ("depreciation", "operations", "terminal", "compatibility", "choice", "costs"):
        if before[section] != after[section]:
            raise ValueError(f"Physical/economic configuration changed: {section}")
    if before["flow_policy"] != after["flow_policy"] or after["data"]["demand_scale"] != 1.0:
        raise ValueError("Flow policy or demand scale changed")
    expected = metadata["demand_orders_in_area"] / metadata["demand_exposure_hours"]
    if not np.isclose(data["demand"].sum(), expected, rtol=0, atol=1e-10):
        raise ValueError("Demand aggregation lost or rescaled raw orders")
    return report


def _membership_audit(membership_path, standard_path):
    membership = pd.read_csv(membership_path, dtype={"day": str, "order_id": str})
    demand = pd.read_csv(Path(standard_path) / "od_demand.csv")
    metadata = json.loads((Path(standard_path) / "metadata.json").read_text())
    counts = membership.groupby("od_id").size().reindex(range(len(demand)), fill_value=0).to_numpy()
    if not np.array_equal(counts, demand.order_count.to_numpy()):
        raise ValueError("Raw membership counts disagree with standard OD demand")
    if len(membership) != metadata["demand_orders_in_area"]:
        raise ValueError("Membership does not contain exactly the selected demand rows")
    if not set(membership.day).issubset(metadata["selected_days"]):
        raise ValueError("Membership contains dates outside the fixed source scope")
    start, end = metadata["demand_hour_window"]
    if not membership.hour.between(start, end, inclusive="left").all():
        raise ValueError("Membership contains orders outside the demand hour window")
    center = np.asarray(metadata["crop_center_local_km"])
    half = metadata["crop_half_size_km"]
    for columns in (("ox", "oy"), ("dx", "dy")):
        if np.any(np.abs(membership[list(columns)].to_numpy() - center) > half + 1e-12):
            raise ValueError("Membership endpoint lies outside the square")
    return dict(
        raw_order_rows=len(membership),
        duplicate_day_order_keys=int(membership.duplicated(["day", "order_id"]).sum()),
        cluster_counts_match=True,
        time_and_crop_scope_match=True,
    )


def prepare_scale_ladder(output=DEFAULT_OUTPUT, *, anchor=DEFAULT_ANCHOR,
                         levels=range(1, 11), raw_path=None, config_path=DEFAULT_CONFIG):
    """Create standard instances plus auditable raw-order membership; never solve.

    Existing instance/provenance targets are rejected. Temporary NPZ/market
    intermediates are removed after exact standard round-trip validation.
    """
    output, anchor = Path(output).resolve(), Path(anchor).resolve()
    protected = [ROOT / name for name in ("data", "Meituan-TSL", "Meituan-Shenzhen", "examples", "MLND")]
    if any(output == directory or directory in output.parents for directory in protected):
        raise ValueError("Write scale-ladder outputs outside original data, examples and papers")
    specs, anchor_metadata = ladder_specs(anchor, levels)
    for spec in specs:
        for subtree in ("instances", "provenance"):
            target = output / subtree / spec["instance_id"]
            if target.exists():
                raise FileExistsError(f"Refusing to replace a prepared scale level: {target}")
    manifest_path = output / "preparation_manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"Preparation manifest already exists: {manifest_path}")
    raw = Path(raw_path or anchor_metadata["raw_path"]).expanduser().resolve()
    if not raw.is_file() or sha256(raw) != anchor_metadata["raw_file_sha256"]:
        raise ValueError("Raw TSL file is missing or differs from the frozen anchor source")
    config_path = Path(config_path).resolve()
    config = load_config(config_path)
    anchor_config = anchor_metadata["frozen_input_metadata"]["resolved_config"]
    for section in ("depreciation", "operations", "terminal", "compatibility", "choice", "costs"):
        if config[section] != anchor_config[section]:
            raise ValueError(f"Benchmark config differs from frozen anchor: {section}")
    if config["flow_policy"] != anchor_config["flow_policy"] or config["data"]["demand_scale"] != 1.0:
        raise ValueError("Benchmark flow policy or demand scale differs")
    prepared_orders = read_orders("tsl", path=raw) if any(s["level"] != 2 for s in specs) else None
    output.mkdir(parents=True, exist_ok=True)
    levels_out = []
    started = time.perf_counter()
    for spec in specs:
        stage_started = time.perf_counter()
        target = output / "instances" / spec["instance_id"]
        provenance = output / "provenance" / spec["instance_id"]
        with tempfile.TemporaryDirectory(prefix=".scale-prepare-", dir=output) as temporary:
            temporary = Path(temporary)
            staged = temporary / "standard"
            if spec["exact_frozen_anchor"]:
                shutil.copytree(anchor, staged)
                market = ANCHOR_MARKET
                for name, expected in anchor_metadata["source_files"]["market"].items():
                    if not (market / name).is_file() or sha256(market / name) != expected:
                        raise ValueError(f"Frozen anchor membership market cannot be verified: {name}")
            else:
                market = temporary / "market"
                scenario = deepcopy(config)
                scenario["name"] = f"tsl_scale_L{spec['n_l']}_H{spec['n_h']}_OD{spec['n_od']}"
                scenario["data"]["input_path"] = str(target)
                prepare_orders(
                    "tsl", market, n_l=spec["n_l"], n_h=spec["n_h"], n_od=spec["n_od"],
                    raw_path=raw, dates=anchor_metadata["selected_days"],
                    demand_hours=tuple(anchor_metadata["demand_hour_window"]),
                    candidate_hours=tuple(anchor_metadata["candidate_hour_window"]),
                    center=spec["crop_center_local_km"], half_size_km=spec["crop_half_size_km"],
                    seed=43, config=scenario, prepared_orders=prepared_orders,
                )
                export_standard(market, market, staged)
            verification = _validate_level(staged, spec, anchor_metadata)
            audit = _membership_audit(market / "order_membership.csv.gz", staged)
            if spec["exact_frozen_anchor"]:
                for name in (*INPUT_FILES, "manifest.json"):
                    if sha256(staged / name) != sha256(anchor / name):
                        raise AssertionError("Frozen anchor payload changed")
            provenance.mkdir(parents=True)
            for name in ("order_membership.csv.gz", "calibration.json"):
                shutil.copy2(market / name, provenance / name)
            note = dict(
                **spec, raw_file_sha256=anchor_metadata["raw_file_sha256"],
                dates=anchor_metadata["selected_days"], demand_hours=[12, 13], candidate_hours=[8, 24],
                seed=43, solver_invoked=False,
                config_path=str(config_path), config_sha256=sha256(config_path),
                site_policy="exact frozen anchor copy" if spec["exact_frozen_anchor"] else
                    "independent weighted observed-site clustering within the level crop; counts exact; levels not nested",
                membership_audit=audit,
                files={name: sha256(provenance / name) for name in ("order_membership.csv.gz", "calibration.json")},
            )
            write_json(provenance / "preparation.json", note)
            target.parent.mkdir(parents=True, exist_ok=True)
            staged.rename(target)
        metadata = json.loads((target / "metadata.json").read_text())
        record = dict(
            **spec, instance_path=str(target.relative_to(output)),
            provenance_path=str(provenance.relative_to(output)),
            demand_orders=metadata["demand_orders_in_area"],
            candidate_orders=metadata["candidate_orders_in_area"],
            demand_exposure_hours=metadata["demand_exposure_hours"],
            hourly_demand=verification["demand_per_hour"],
            nonzero_kappa=verification["nonzero_kappa"],
            manifest_sha256=sha256(target / "manifest.json"),
            preparation_seconds=time.perf_counter() - stage_started,
            status="prepared_validated_not_solved",
        )
        levels_out.append(record)
        print(f"Prepared {spec['instance_id']}: {spec['n_l']}L/{spec['n_h']}H/{spec['n_od']}OD; "
              f"{record['demand_orders']} orders; {record['hourly_demand']:.3f}/hour", flush=True)
    manifest = dict(
        schema=SCHEMA, status="prepared_validated_not_solved", solver_invoked=False,
        anchor=str(anchor), anchor_manifest_sha256=sha256(anchor / "manifest.json"),
        config_path=str(config_path), config_sha256=sha256(config_path),
        raw_path=str(raw), raw_file_sha256=anchor_metadata["raw_file_sha256"],
        selected_days=anchor_metadata["selected_days"],
        demand_hours=[12, 13], candidate_hours=[8, 24], cases=["K1", "K2"],
        level_rule="nL / 10nH / 25nOD, n=1..10",
        area_rule="anchor area times n/2; same square center; constant H density",
        candidate_sets_nested=False,
        source_policy="accepted real orders; all crop/window demand retained; no rescaling or synthetic demand",
        levels=levels_out, preparation_seconds=time.perf_counter() - started,
        preparation_code_sha256=sha256(Path(__file__)),
        pipeline_code_sha256={
            name: sha256(ROOT / name) for name in (
                "src/data/scale_ladder.py", "src/data/prepare.py", "src/data/performance_orders.py",
                "src/data/performance_clusters.py", "src/data/instances.py", "src/data/standard.py",
                "src/model/compatibility.py", "src/model/paths.py", "src/model/resources.py", "src/config.py",
            ) if (ROOT / name).is_file()
        },
    )
    write_json(manifest_path, manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--anchor", type=Path, default=DEFAULT_ANCHOR)
    parser.add_argument("--raw-path", type=Path)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--levels", type=int, nargs="+", default=list(range(1, 11)))
    args = parser.parse_args()
    prepare_scale_ladder(args.output, anchor=args.anchor, levels=args.levels,
                         raw_path=args.raw_path, config_path=args.config)


if __name__ == "__main__":
    main()
