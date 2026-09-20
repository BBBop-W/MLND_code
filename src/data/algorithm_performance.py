"""Frozen, demand-preserving instances for algorithm performance comparisons.

Two sources x two geographic regions x six sizes. Raw OD tuples are clustered
jointly; every retained order contributes to exactly one aggregate OD market.
The saved model arrays are the authoritative reusable benchmark inputs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from src.config import load_config
from src.data.choice import rider_tariff
from src.data.demand import demand_signature
from src.data.generator import DataGenerator
from src.data.instance_io import ROOT, sha256
from src.data.instances import load_instance
from src.data.performance_clusters import cluster_od, select_sites, choose_regions
from src.data.performance_orders import read_orders, fit_rider
from src.experiments.storage import save_inputs, write_json, load_inputs


SIZES = (10, 20, 30, 40, 50, 60)
FIXED_COSTS = (
    "base_site_investment", "locker_site_investment", "swap_site_investment",
    "base_investment", "cabinet_investment", "swap_investment", "drone_investment",
)
DEFAULT_ROOT = ROOT / "data/instances/algorithm_performance"
HALF_SIZE_10 = 10 * np.sqrt(10 / 120)


def benchmark_config():
    cfg = load_config()
    cfg["name"] = "algorithm_performance"
    for key in FIXED_COSTS:
        cfg["costs"][key] *= 0.8
    cfg["data"].update(demand_hubs=None, demand_scale=1.0)
    cfg["solver"].update(
        time_limit=7200.0, threads=1, mip_gap=1e-4, mip_gap_abs=1e-10,
        column_limit=None, graph_limit=None, max_iterations=None,
    )
    return cfg


def crop_orders(orders, center, half_size):
    xy = orders[["ox", "oy", "dx", "dy"]].to_numpy()
    offsets = np.tile(np.asarray(center), 2)
    return orders.loc[np.max(np.abs(xy - offsets), axis=1) <= half_size].copy()


def write_od(directory, demand, labels, representatives, days):
    k = len(representatives)
    counts = np.bincount(labels, minlength=k)
    assert np.all(counts > 0) and counts.sum() == len(demand)
    nodes = []
    for role, prefix, cols in (("origin", "O", (0, 1)), ("destination", "D", (2, 3))):
        for j, row in enumerate(representatives):
            nodes.append(dict(node_id=f"{prefix}{j:04d}", role=role,
                              x_km=row[cols[0]], y_km=row[cols[1]],
                              raw_endpoint_count=int(counts[j])))
    pd.DataFrame(nodes).to_csv(directory / "od_nodes.csv", index=False)
    distance = np.linalg.norm(demand[["ox", "oy"]].to_numpy()
                              - demand[["dx", "dy"]].to_numpy(), axis=1)
    # Keep mean observed direct distance separately from representative geometry.
    mean_distance = np.bincount(labels, weights=distance, minlength=k) / counts
    table = pd.DataFrame(dict(
        od_id=[f"OD{j:04d}" for j in range(k)],
        origin_node=[f"O{j:04d}" for j in range(k)],
        destination_node=[f"D{j:04d}" for j in range(k)],
        order_count=counts, exposure_hours=days, lambda_per_hour=counts / days,
        direct_distance_km=mean_distance,
    ))
    table.to_csv(directory / "od_demand.csv", index=False)
    pd.DataFrame([dict(period_id="h12_13", start_hour=12, end_hour=13,
                       dates=";".join(sorted(demand.day.unique())),
                       exposure_hours=days, order_count=len(demand),
                       total_rate_per_hour=len(demand) / days)]).to_csv(
        directory / "periods.csv", index=False
    )
    membership = demand[["order_id", "day"]].copy()
    membership.insert(0, "accepted_source_row", demand.index)
    membership["od"] = labels
    membership.to_csv(directory / "order_membership.csv.gz", index=False,
                      compression={"method": "gzip", "mtime": 0})
    return counts, distance, mean_distance


def write_calibration(directory, h, demand, labels, counts, mean_distance, fit, dispatch):
    out = directory / "calibration"
    out.mkdir()
    write_json(out / "rider_time.json", fit)
    valid = np.isfinite(demand.post_minutes.to_numpy())
    observed_counts = np.bincount(labels[valid], minlength=len(counts))
    sums = np.bincount(labels[valid], weights=demand.post_minutes.to_numpy()[valid],
                       minlength=len(counts))
    prior = (fit["fitted_postfetch_intercept_minutes"]
             + fit["slope_minutes_per_straight_km"] * mean_distance)
    observed = np.divide(sums, observed_counts, out=np.full(len(counts), np.nan),
                         where=observed_counts > 0)
    table = pd.DataFrame(dict(
        od=np.arange(len(counts)), observed_count=observed_counts,
        observed_postfetch_minutes=observed, prior_minutes=prior,
        deadline_minutes=(sums + 5 * prior) / (observed_counts + 5),
    ))
    table.to_csv(out / "deadlines.csv", index=False)
    pre = demand.pre_minutes.dropna()
    if pre.empty:
        raise ValueError("No valid pre-dispatch observations in demand crop")
    write_json(out / "pre_dispatch.json", dict(cases={str(h):dict(
        mean_hours=float(pre.mean() / 60), mean_minutes=float(pre.mean()),
        median_minutes=float(pre.median()), p90_minutes=float(pre.quantile(.9)),
        canonical_demand_orders=len(demand), canonical_od_count=len(counts),
        valid_timing_samples=len(pre), excluded_timing_samples=len(demand)-len(pre),
        unit="hour",
    )}))
    calibration = dict(
        schema=1, model_version="path_cqn_v1", dispatch=dispatch, prior_weight=5.0,
        first_leg_mean_fit="rider_time.json",
        first_leg_mean_fit_sha256=sha256(out / "rider_time.json"),
        pre_dispatch_statistics="pre_dispatch.json",
        pre_dispatch_statistics_sha256=sha256(out / "pre_dispatch.json"),
        terminal_time="home-delivery rider travel mean plus constant request-to-acceptance proxy",
        terminal_time_choice="same home-delivery benchmark for both choices, self-pickup travel excluded",
        cases={str(h):dict(
            file="deadlines.csv", sha256=sha256(out / "deadlines.csv"),
            od_demand_sha256=sha256(directory / "od_demand.csv"),
            od_nodes_sha256=sha256(directory / "od_nodes.csv"),
            prior_weight=5.0, n_od=len(counts), valid_rows=int(valid.sum()),
            canonical_window_rows=len(demand),
        )},
        deadline_method="Aggregate all valid observed post-fetch times within each OD cluster; shrink mean with five fitted-prior observations",
    )
    write_json(out / "calibration.json", calibration)
    return out / "calibration.json"


def build_case(root, source, region, orders, metadata, center, h, fit, dispatch, seed):
    bases_count, e, k = h // 5, 2 * h // 5, 20 * h
    case_id = f"{source}_{region}_H{h:02d}L{bases_count:02d}E{e:02d}OD{k:04d}"
    parent = root / source / region
    directory = parent / f"H{h:03d}_L{bases_count:02d}_E{e:02d}_OD{k:04d}"
    directory.mkdir(parents=True, exist_ok=False)
    half = HALF_SIZE_10 * np.sqrt(h / 10)
    crop = crop_orders(orders, center, half)
    candidates = crop.loc[crop.hour.between(8, 23)]
    demand = crop.loc[crop.hour.eq(12)]
    if len(demand) < k:
        raise ValueError(f"{case_id}: insufficient raw orders for {k} OD clusters")
    labels, representatives, clustering = cluster_od(
        demand[["ox", "oy", "dx", "dy"]].to_numpy(), k, seed
    )
    pickup = candidates[["ox", "oy"]].to_numpy()
    dropoff = candidates[["dx", "dy"]].to_numpy()
    hubs = select_sites(np.vstack([pickup, dropoff]), h, seed + 1)
    bases = select_sites(pickup, bases_count, seed + 2)
    swaps = select_sites((pickup + dropoff) / 2, e, seed + 3)
    days = len(metadata["selected_days"])
    DataGenerator._write_sites(directory / "sites.csv", hubs, bases, swaps, 16 * days)
    counts, distances, mean_distance = write_od(
        directory, demand, labels, representatives, days
    )
    meta = dict(
        metadata, family="algorithm_performance", case_id=case_id, region=region,
        coordinate_system="local_km", candidate_orders_in_area=len(candidates),
        demand_orders_in_area=len(demand), total_demand_per_hour=len(demand) / days,
        candidate_exposure_hours=16 * days, demand_exposure_hours=days,
        od_market_count=k, origin_node_count=k, destination_node_count=k,
        region_center_local_km=list(map(float, center)), crop_half_size_km=float(half),
        region_bounds_local_km=dict(x_min=center[0]-half, x_max=center[0]+half,
                                    y_min=center[1]-half, y_max=center[1]+half),
        data_config=dict(source=source, candidate_count=h, num_bases=bases_count, num_swaps=e,
                         demand_hour_start=12, demand_hour_end=13,
                         candidate_hour_start=8, candidate_hour_end=24,
                         map_half_size_km=float(half), seed=seed,
                         dates=metadata["selected_days"], od_method="joint_4d_clustering"),
        aggregation=clustering,
    )
    write_json(directory / "metadata.json", meta)
    calibration = write_calibration(
        directory, h, demand, labels, counts, mean_distance, fit, dispatch
    )
    cfg = benchmark_config()
    cfg["data"].update(instance_root=str(parent), calibration=str(calibration))
    write_json(directory / "config.json", cfg)
    data = load_instance(h, cfg)
    data["name"] = case_id
    # Preserve the original courier tariff exactly, including its distance kink.
    raw_ground = rider_tariff(distances, cfg)
    data["ground"] = np.bincount(labels, weights=raw_ground, minlength=k) / counts
    data["c0"] = float(data["ground"] @ data["demand"])
    data["provenance"]["aggregation"] = dict(
        **clustering, raw_orders=len(demand), demand_preserved=True,
        ground_cost_method="mean raw-order courier tariff per cluster",
        canonical_direct_distance="mean raw-order straight distance per cluster",
        canonical_positions="observed 4D tuple nearest cluster centroid",
        authoritative_replay="input_arrays.npz plus input_metadata.json",
    )
    data["demand_signature"] = demand_signature(data)
    data["provenance"]["demand_selection"].update(
        signature=data["demand_signature"], all_ground_cost=data["c0"],
        aggregation="joint_4d_clustering", original_orders=len(demand),
    )
    hashes = save_inputs(data, directory)
    replay = load_inputs(directory)
    if demand_signature(replay) != data["demand_signature"]:
        raise AssertionError("Saved-input replay changed demand")
    if not np.isclose(data["demand"].sum(), len(demand) / days, atol=1e-8):
        raise AssertionError("Clustering lost demand")
    if not np.isclose(data["c0"], raw_ground.sum() / days, atol=1e-7):
        raise AssertionError("Clustering changed all-courier baseline cost")
    if (data["n_h"], data["n_l"], data["n_k"], data["n_od"]) != (h, bases_count, e, k):
        raise AssertionError("Requested dimensions were not generated exactly")
    files = {p.name:sha256(p) for p in directory.iterdir() if p.is_file()}
    result = dict(
        id=case_id, source=source, region=region, H=h, L=bases_count, E=e, OD=k,
        input_path=str(directory), config=str(directory / "config.json"),
        raw_orders=len(demand), hourly_demand=float(data["demand"].sum()),
        all_ground_cost=data["c0"], side_km=float(2 * half),
        center_local_km=list(map(float, center)), demand_signature=data["demand_signature"],
        input_hashes=hashes, files=files, validation_passed=True,
    )
    print(json.dumps({key:result[key] for key in (
        "id", "raw_orders", "hourly_demand", "all_ground_cost"
    )}), flush=True)
    return result


def build(output=DEFAULT_ROOT):
    root = Path(output).resolve()
    if root.exists():
        raise FileExistsError(f"Preserving fixed benchmark instances: {root}")
    root.mkdir(parents=True)
    manifest = dict(
        schema=1, name="algorithm performance", version=1, status="building",
        scales=[dict(H=h, L=h//5, E=2*h//5, OD=20*h) for h in SIZES],
        candidate_hours=[8, 24], demand_hours=[12, 13],
        demand_policy="Both endpoints inside crop; all retained orders clustered, no sampling or demand rescaling",
        cost_multiplier=0.8, discounted_cost_fields=list(FIXED_COSTS),
        unchanged_costs="directed routes, rider tariffs, hourly flight and hover costs",
        reference_config_sha256=sha256(ROOT / "configs/default.json"),
        sources={}, cases=[],
    )
    write_json(root / "config.json", benchmark_config())
    # Nanshan lacks a rider-acceptance timestamp; retain the established TSL transfer.
    tsl_orders, tsl_meta = read_orders("tsl")
    tsl_fit = fit_rider(tsl_orders.loc[tsl_orders.hour.eq(12)])
    dispatch = fit_rider(tsl_orders)["dispatch"]
    if not np.isfinite(dispatch["mean_minutes"]):
        raise ValueError("TSL dispatch estimate is unavailable")
    for source in ("nanshan", "tsl"):
        orders, metadata = ((tsl_orders, tsl_meta) if source == "tsl"
                            else read_orders(source))
        fit = tsl_fit if source == "tsl" else fit_rider(orders.loc[orders.hour.eq(12)])
        centers, region_audit = choose_regions(orders, HALF_SIZE_10, seed=43)
        manifest["sources"][source] = dict(metadata, region_selection=region_audit)
        source_dispatch = dict(dispatch, transfer_to=source)
        for region_index, center in enumerate(centers, 1):
            region = f"region_{region_index}"
            for h in SIZES:
                row = build_case(root, source, region, orders, metadata, center,
                                 h, fit, source_dispatch, seed=43)
                manifest["cases"].append(row)
                write_json(root / "manifest.json", manifest)
        del orders
    manifest["status"] = "frozen"
    manifest["case_count"] = len(manifest["cases"])
    manifest["all_dimensions_exact"] = True
    manifest["all_raw_demand_preserved"] = True
    manifest["all_direct_courier_costs_preserved"] = True
    write_json(root / "manifest.json", manifest)
    rows = ["# Algorithm performance", "",
            "Fixed benchmark inputs: two sources, two regions per source, six scales.",
            "All in-crop 12:00–13:00 orders are retained in joint origin–destination clustering.",
            "Candidate density: 0.3 lockers/km². Regions may overlap at larger sizes.",
            "Fixed facility/equipment and drone investment costs are 80% of the baseline; route and operating costs are unchanged.",
            "Replay the saved input arrays to preserve cluster-mean courier tariffs exactly.", "",
            "| Case | Raw orders | Demand/hour | Side/km |", "|---|---:|---:|---:|"]
    for row in manifest["cases"]:
        rows.append(f"| {row['id']} | {row['raw_orders']:,} | {row['hourly_demand']:.2f} | {row['side_km']:.3f} |")
    (root / "README.md").write_text("\n".join(rows) + "\n")
    return root / "manifest.json"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        print(build(args.output), flush=True)
