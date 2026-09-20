"""Rebuild OD timing tables for a generated geographic family.

The distance slope and rider calling delay are shared with the baseline. Each
crop gets its own observed OD deadlines and payment-to-fetch statistics.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from .calibration import clock_minutes, deadline_table, postfetch_records
from .instance_io import ROOT, read_instance, sha256


def pre_dispatch_statistics(raw, metadata, n_od):
    cfg = metadata["data_config"]
    pay, fetch, arrive = (
        clock_minutes(raw[name])
        for name in ("customer_pay_time", "fetch_time", "arrived_time")
    )
    lon, lat = metadata["projection_center_lon"], metadata["projection_center_lat"]
    scale = 111.32 * np.cos(np.deg2rad(lat))
    distances = [
        (raw.pickup_longitude - lon) * scale,
        (raw.pickup_latitude - lat) * 111.32,
        (raw.dropoff_longitude - lon) * scale,
        (raw.dropoff_latitude - lat) * 111.32,
    ]
    selected = raw.dt_seq_id.astype(str).isin(metadata["selected_days"])
    selected &= pay.ge(60 * cfg["demand_hour_start"]) & pay.lt(
        60 * cfg["demand_hour_end"]
    )
    selected &= np.maximum.reduce([abs(v) for v in distances]) <= cfg["map_half_size_km"]
    valid = selected & pay.lt(fetch) & fetch.lt(arrive)
    values = (fetch - pay).loc[valid]
    if len(values) == 0 or int(selected.sum()) != metadata["demand_orders_in_area"]:
        raise ValueError("Timing window differs from generated demand")
    return dict(
        unit="hour",
        crop_half_size_km=cfg["map_half_size_km"],
        canonical_demand_orders=int(selected.sum()),
        canonical_od_count=n_od,
        valid_timing_samples=len(values),
        excluded_timing_samples=int(selected.sum()) - len(values),
        mean_hours=float(values.mean() / 60),
        mean_minutes=float(values.mean()),
        median_minutes=float(values.median()),
        p90_minutes=float(values.quantile(0.9)),
        p95_minutes=float(values.quantile(0.95)),
    )


def build(instance_root, output, *, baseline=ROOT / "data/model_inputs/calibration.json"):
    root, out, baseline = Path(instance_root), Path(output), Path(baseline)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Preserving existing calibration: {out}")
    manifest = json.loads(baseline.read_text())
    fit_path = baseline.parent / manifest["first_leg_mean_fit"]
    if sha256(fit_path) != manifest["first_leg_mean_fit_sha256"]:
        raise ValueError("Baseline rider fit checksum mismatch")
    fit = json.loads(fit_path.read_text())
    out.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(fit_path, out / "rider_time.json")
    cases, pre = {}, {}
    raw, raw_path, raw_hash = None, None, None
    for directory in sorted(root.glob("H*_L*_E*")):
        metadata = json.loads((directory / "metadata.json").read_text())
        h = int(metadata["data_config"]["candidate_count"])
        data, _, _ = read_instance(h, root, demand_hubs=None)
        path = Path(metadata["raw_path"])
        if path != raw_path:
            raw_path, raw = path, pd.read_csv(path)
            raw_hash = sha256(path)
        if raw_hash != metadata["raw_file_sha256"]:
            raise ValueError("Raw orders changed after candidate generation")
        records, audit = postfetch_records(raw, metadata)
        if audit["canonical_window_rows"] != metadata["demand_orders_in_area"]:
            raise ValueError("Observed timing and canonical demand windows differ")
        table = deadline_table(
            data, records, metadata,
            slope=fit["slope_minutes_per_straight_km"],
            intercept=fit["fitted_postfetch_intercept_minutes"],
            prior_weight=manifest["prior_weight"],
        )
        target = out / f"H{h:03d}_postfetch_deadlines.csv"
        table.to_csv(target, index=False)
        cases[str(h)] = dict(
            file=target.name, sha256=sha256(target), **audit, n_od=data["n_od"],
            source=str(path), source_sha256=raw_hash,
            metadata_sha256=sha256(directory / "metadata.json"),
            od_demand_sha256=sha256(directory / "od_demand.csv"),
            od_nodes_sha256=sha256(directory / "od_nodes.csv"),
            observed_empty_od=int(table.observed_count.eq(0).sum()),
            observed_less_than_five_od=int(table.observed_count.lt(5).sum()),
            weighted_deadline_minutes=float(np.average(table.deadline_minutes, weights=data["demand"])),
            mean_observed_postfetch_minutes=float(records.minutes.mean()),
            prior_weight=manifest["prior_weight"],
        )
        pre[str(h)] = pre_dispatch_statistics(raw, metadata, data["n_od"])
        if pre[str(h)]["valid_timing_samples"] != audit["valid_rows"]:
            raise ValueError("Pre- and post-fetch timing samples differ")
    if not cases:
        raise ValueError("No generated geographic instances found")
    stats = out / "pre_dispatch_statistics.json"
    stats.write_text(json.dumps(dict(
        unit="hour", definition="payment-to-fetch duration excluded from drone workload",
        cases=pre,
    ), indent=2) + "\n")
    manifest.update(
        cases=cases,
        first_leg_mean_fit="rider_time.json",
        pre_dispatch_statistics=stats.name,
        pre_dispatch_statistics_sha256=sha256(stats),
        baseline_calibration=str(baseline.resolve()),
        baseline_calibration_sha256=sha256(baseline),
    )
    target = out / "calibration.json"
    target.write_text(json.dumps(manifest, indent=2) + "\n")
    return target


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instance-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(build(args.instance_root, args.output))
