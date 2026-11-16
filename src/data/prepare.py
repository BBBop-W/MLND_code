"""One configurable observed-order to L/H-market preparation pipeline."""

from pathlib import Path
from copy import deepcopy
import numpy as np
import pandas as pd

from src.config import ROOT, load_config
from .performance_orders import read_orders, fit_rider
from .performance_clusters import cluster_od, select_sites
from .choice import rider_tariff
from .instances import _scenario


def _window(hours, window):
    start, end = window
    if not 0 <= start < 24 or not 0 <= end <= 24 or start == end:
        raise ValueError("Hour windows must be nonempty within [0, 24]")
    return (
        (hours >= start) & (hours < end)
        if start < end
        else (hours >= start) | (hours < end)
    )


def prepare_orders(
    source,
    output,
    *,
    n_l=2,
    n_h=10,
    n_od=200,
    raw_path=None,
    dates=None,
    demand_hours=(12, 13),
    candidate_hours=(8, 24),
    center=None,
    half_size_km=None,
    seed=43,
    config=None,
    prepared_orders=None,
):
    """Prepare new inputs without fixed L:H:OD ratios or any E facilities.

    Existing output directories are rejected. Original data and manuscript
    directories are never targets. Missing clocks exclude timing observations,
    not demand rows. Each raw order enters one weighted joint-OD cluster.
    """
    from src.experiments.storage import save_inputs, write_json

    output = Path(output).expanduser().resolve()
    protected = [
        ROOT / name for name in ("data", "Meituan-Shenzhen", "Meituan-TSL", "MLND")
    ]
    if any(output == path or path in output.parents for path in protected):
        raise ValueError(
            "Write prepared instances under outputs/prepared, not original data/manuscript directories"
        )
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    for name, value in (("n_l", n_l), ("n_h", n_h), ("n_od", n_od)):
        if isinstance(value, bool) or int(value) != value or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    cfg = load_config(config)
    if prepared_orders is None:
        orders, source_meta = read_orders(source, path=raw_path)
    else:
        orders, source_meta = prepared_orders
        if not isinstance(orders, pd.DataFrame) or not isinstance(source_meta, dict):
            raise TypeError("prepared_orders must contain the read_orders DataFrame and metadata")
        if orders.attrs.get("source") != source or source_meta.get("source") != source:
            raise ValueError("Cached observed orders belong to a different source")
        if raw_path is not None and Path(raw_path).expanduser().resolve() != Path(source_meta["raw_path"]).resolve():
            raise ValueError("Cached observed orders belong to a different raw file")
        # All subsequent frame operations select/copy before mutation; the
        # metadata is copied here because it is augmented for each crop.
        source_meta = deepcopy(source_meta)
    selected = sorted(set(map(str, dates))) if dates else source_meta["selected_days"]
    missing = set(selected) - set(source_meta["selected_days"])
    if missing:
        raise ValueError(f"Unavailable dates: {sorted(missing)}")
    crop = orders.loc[orders.day.isin(selected)].copy()
    selected_date_orders = len(crop)
    bounds = None
    if half_size_km is not None:
        if not np.isfinite(half_size_km) or half_size_km <= 0:
            raise ValueError("half_size_km must be positive")
        center = np.asarray(
            center if center is not None else crop[["ox", "oy"]].median(), float
        )
        if center.shape != (2,) or not np.isfinite(center).all():
            raise ValueError("center must be a finite local-km pair")
        first = np.abs(crop[["ox", "oy"]].to_numpy() - center).max(axis=1)
        last = np.abs(crop[["dx", "dy"]].to_numpy() - center).max(axis=1)
        crop = crop.loc[(first <= half_size_km) & (last <= half_size_km)]
        bounds = dict(
            x_min=float(center[0] - half_size_km),
            x_max=float(center[0] + half_size_km),
            y_min=float(center[1] - half_size_km),
            y_max=float(center[1] + half_size_km),
        )
    demand = crop.loc[_window(crop.hour, demand_hours)].copy()
    candidates = crop.loc[_window(crop.hour, candidate_hours)]
    if not len(demand) or not len(candidates):
        raise ValueError("Selected windows contain no demand/candidate orders")
    labels, representatives, audit = cluster_od(
        demand[["ox", "oy", "dx", "dy"]].to_numpy(), n_od, seed
    )
    pickup = candidates[["ox", "oy"]].to_numpy()
    dropoff = candidates[["dx", "dy"]].to_numpy()
    bases = select_sites(pickup, n_l, seed + 1)[:, :2]
    hubs = select_sites(np.concatenate((pickup, dropoff)), n_h, seed + 2)[:, :2]
    window_length = demand_hours[1] - demand_hours[0]
    exposure = len(selected) * (
        window_length if window_length > 0 else window_length + 24
    )
    counts = np.bincount(labels, minlength=n_od)
    distances = np.linalg.norm(
        demand[["ox", "oy"]].to_numpy() - demand[["dx", "dy"]].to_numpy(), axis=1
    )
    ground = (
        np.bincount(labels, weights=rider_tariff(distances, cfg), minlength=n_od)
        / counts
    )
    fit = fit_rider(crop)
    prior = (
        fit["fitted_postfetch_intercept_minutes"]
        + fit["slope_minutes_per_straight_km"]
        * np.bincount(labels, weights=distances, minlength=n_od)
        / counts
    )
    valid = demand.post_minutes.notna().to_numpy()
    valid_counts = np.bincount(labels[valid], minlength=n_od)
    sums = np.bincount(
        labels[valid], weights=demand.post_minutes.to_numpy()[valid], minlength=n_od
    )
    deadline = (sums + 5 * prior) / (valid_counts + 5) / 60
    source_meta.update(
        selected_days=selected,
        demand_exposure_hours=exposure,
        candidate_hour_window=list(candidate_hours),
        demand_hour_window=list(demand_hours),
        crop_center_local_km=None if center is None else list(map(float, center)),
        crop_half_size_km=half_size_km,
        crop_bounds_local_km=bounds,
        crop_rule="both pickup and dropoff endpoints within inclusive square bounds",
        selected_date_orders=selected_date_orders,
        all_day_orders_in_area=len(crop),
        candidate_orders_in_area=len(candidates),
        demand_orders_in_area=len(demand),
        excluded_spatial_orders=selected_date_orders - len(crop),
        valid_demand_timing_rows=int(valid.sum()),
        raw_hourly_demand=len(demand) / exposure,
        aggregation=audit,
        deadline_estimation="OD post-fetch sample means shrunk with five observations of fitted distance prior",
    )
    market = dict(
        name=f"{source}_L{n_l}_H{n_h}_OD{n_od}",
        bases=bases,
        hubs=hubs,
        origins=representatives[:, :2],
        destinations=representatives[:, 2:],
        od_origin=np.arange(n_od),
        od_destination=np.arange(n_od),
        demand=counts / exposure,
        ground=ground,
        delivery_deadline=deadline,
        rider_slope_minutes_per_straight_km=fit["slope_minutes_per_straight_km"],
        od_pairs=[[f"O{i:05d}", f"D{i:05d}"] for i in range(n_od)],
        provenance={"instance_metadata": source_meta},
    )
    data = _scenario(market, cfg, dict(kind="raw_orders", **source_meta))
    if not np.isclose(
        data["demand"].sum(), len(demand) / exposure * cfg["data"]["demand_scale"]
    ):
        raise AssertionError("Clustering changed demand")
    output.mkdir(parents=True)
    hashes = save_inputs(data, output)
    sites = [
        dict(site_id=f"{role}{i:04d}", type=role, x_km=p[0], y_km=p[1])
        for role, points in (("L", bases), ("H", hubs))
        for i, p in enumerate(points)
    ]
    pd.DataFrame(sites).to_csv(output / "sites.csv", index=False)
    nodes = [
        dict(node_id=f"{prefix}{i:05d}", role=role, x_km=p[0], y_km=p[1])
        for prefix, role, points in (
            ("O", "origin", representatives[:, :2]),
            ("D", "destination", representatives[:, 2:]),
        )
        for i, p in enumerate(points)
    ]
    pd.DataFrame(nodes).to_csv(output / "od_nodes.csv", index=False)
    pd.DataFrame(
        dict(
            od_id=np.arange(n_od),
            origin_node=[v[0] for v in market["od_pairs"]],
            destination_node=[v[1] for v in market["od_pairs"]],
            order_count=counts,
            exposure_hours=exposure,
            lambda_per_hour=counts / exposure,
            direct_distance_km=np.bincount(labels, weights=distances, minlength=n_od)
            / counts,
        )
    ).to_csv(output / "od_demand.csv", index=False)
    membership = demand[
        ["day", "order_id", "hour", "ox", "oy", "dx", "dy", "post_minutes"]
    ].copy()
    membership["od_id"] = labels
    membership.to_csv(
        output / "order_membership.csv.gz",
        index=False,
        compression={"method": "gzip", "mtime": 0},
    )
    pd.DataFrame(
        [
            dict(
                day=day,
                demand_orders=int(demand.day.eq(day).sum()),
                candidate_orders=int(candidates.day.eq(day).sum()),
                demand_exposure_hours=exposure / len(selected),
            )
            for day in selected
        ]
    ).to_csv(output / "periods.csv", index=False)
    write_json(output / "metadata.json", source_meta)
    write_json(
        output / "calibration.json",
        dict(rider=fit, deadline_hours=deadline, observed_counts=valid_counts),
    )
    result = dict(
        input_path=str(output),
        model_version="consecutive",
        n_l=n_l,
        n_h=n_h,
        n_od=n_od,
        hourly_demand=float(data["demand"].sum()),
        input_hashes=hashes,
    )
    write_json(output / "manifest.json", result)
    return result
