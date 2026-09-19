"""Observed-clock filters, OD deadline estimates, and verified calibration inputs."""

from __future__ import annotations
import json
import numpy as np
import pandas as pd
from .instance_io import sha256, project_path


def clock_minutes(series):
    parts = series.astype(str).str.extract(r"^(\d{1,2}):(\d{2})$")
    hours = pd.to_numeric(parts[0], errors="coerce")
    minutes = pd.to_numeric(parts[1], errors="coerce")
    return (60 * hours + minutes).where(hours.between(0, 23) & minutes.between(0, 59))


def estimate_dispatch_delay(raw):
    """Estimate request-to-acceptance from completed, non-prebook waybills."""
    names = ["platform_order_time", "grab_time", "fetch_time", "arrive_time"]
    clocks = raw[names].apply(pd.to_numeric, errors="coerce")
    selected = raw.is_courier_grabbed.eq(1) & raw.is_prebook.eq(0)
    valid = selected & np.isfinite(clocks).all(axis=1) & clocks.gt(0).all(axis=1)
    valid &= clocks.platform_order_time.le(clocks.grab_time)
    valid &= clocks.grab_time.le(clocks.fetch_time) & clocks.fetch_time.lt(
        clocks.arrive_time
    )
    minutes = (
        clocks.loc[valid, "grab_time"] - clocks.loc[valid, "platform_order_time"]
    ) / 60
    if not len(minutes):
        raise ValueError("No valid completed non-prebook dispatch records")
    answer = dict(
        raw_rows=len(raw),
        selected_nonprebook_accepted_rows=int(selected.sum()),
        valid_rows=len(minutes),
        rejected_selected_rows=int((selected & ~valid).sum()),
        mean_minutes=float(minutes.mean()),
        median_minutes=float(minutes.median()),
        q90_minutes=float(minutes.quantile(0.9)),
        definition="(grab_time-platform_order_time)/60; request creation to rider acceptance",
        observed_event="acceptance, not rider arrival or meal pickup",
        transfer_assumption="constant last-mile rider calling delay transferred from accepted TSL orders",
        unobserved_component="acceptance-to-locker approach travel is not identified separately",
    )
    alternatives = {}
    for column in ("order_push_time", "dispatch_time"):
        if column in raw:
            start = pd.to_numeric(raw[column], errors="coerce")
            mask = valid & start.gt(0) & start.le(clocks.grab_time)
            values = (clocks.loc[mask, "grab_time"] - start[mask]) / 60
            alternatives[column] = dict(
                n=len(values), mean_minutes=float(values.mean())
            )
    answer["alternative_start_clocks"] = alternatives
    return answer


def postfetch_records(raw, metadata):
    """Select the canonical spatial/time window and strict observed chronology."""
    cfg = metadata["data_config"]
    pay, fetch, arrive = (
        clock_minutes(raw[k])
        for k in ("customer_pay_time", "fetch_time", "arrived_time")
    )
    scale = 111.32 * np.cos(np.deg2rad(metadata["projection_center_lat"]))
    ox = (raw.pickup_longitude - metadata["projection_center_lon"]) * scale
    oy = (raw.pickup_latitude - metadata["projection_center_lat"]) * 111.32
    dx = (raw.dropoff_longitude - metadata["projection_center_lon"]) * scale
    dy = (raw.dropoff_latitude - metadata["projection_center_lat"]) * 111.32
    selected = raw.dt_seq_id.astype(str).isin(metadata["selected_days"])
    selected &= pay.ge(60 * cfg["demand_hour_start"]) & pay.lt(
        60 * cfg["demand_hour_end"]
    )
    selected &= (
        np.maximum.reduce([abs(ox), abs(oy), abs(dx), abs(dy)])
        <= cfg["map_half_size_km"]
    )
    valid = selected & pay.lt(fetch) & fetch.lt(arrive)
    values = pd.DataFrame(dict(ox=ox, oy=oy, dx=dx, dy=dy, minutes=arrive - fetch))
    valid &= np.isfinite(values).all(axis=1)
    return values.loc[valid].reset_index(drop=True), dict(
        raw_rows=len(raw),
        canonical_window_rows=int(selected.sum()),
        valid_rows=int(valid.sum()),
        rejected_chronology_or_nonfinite=int((selected & ~valid).sum()),
        time_definition="arrived_time-fetch_time, minutes; strict same-day payment < fetch < arrived",
        preparation_time_included=False,
    )


def deadline_table(data, records, metadata, *, slope, intercept, prior_weight=5.0):
    """OD sample means shrunk toward the fitted post-fetch conditional mean."""
    if prior_weight < 0 or not np.isfinite(prior_weight):
        raise ValueError("prior_weight must be finite and nonnegative")
    grid = float(metadata["data_config"]["od_grid_km"])
    cell = lambda point: tuple(np.floor(np.asarray(point) / grid).astype(int))
    origins = {cell(p): k for k, p in enumerate(data["origins"])}
    destinations = {cell(p): k for k, p in enumerate(data["destinations"])}
    od_lookup = {
        (int(o), int(d)): k
        for k, (o, d) in enumerate(zip(data["od_origin"], data["od_destination"]))
    }
    indices = []
    for row in records.itertuples(index=False):
        key = (
            origins.get(cell((row.ox, row.oy))),
            destinations.get(cell((row.dx, row.dy))),
        )
        if key not in od_lookup:
            raise ValueError(
                f"Observed OD is absent from the canonical instance: {key}"
            )
        indices.append(od_lookup[key])
    counts = np.bincount(indices, minlength=data["n_od"])
    sums = np.bincount(indices, weights=records.minutes, minlength=data["n_od"])
    length = np.linalg.norm(
        data["origins"][data["od_origin"]]
        - data["destinations"][data["od_destination"]],
        axis=1,
    )
    prior = intercept + slope * length
    means = np.divide(sums, counts, out=np.full(len(counts), np.nan), where=counts > 0)
    denominator = counts + prior_weight
    deadline = np.divide(
        sums + prior_weight * prior,
        denominator,
        out=prior.copy(),
        where=denominator > 0,
    )
    if np.any(deadline <= 0) or not np.isfinite(deadline).all():
        raise ValueError("Post-fetch deadline must be positive and finite")
    return pd.DataFrame(
        dict(
            od=np.arange(data["n_od"]),
            observed_count=counts,
            observed_postfetch_minutes=means,
            prior_minutes=prior,
            deadline_minutes=deadline,
        )
    )


def load_calibration(hubs, data, path):
    """Verify the processed calibration against canonical OD CSV checksums."""
    path = project_path(path)
    if path.is_dir():
        path /= "calibration.json"
    fit = json.loads(path.read_text())
    mean_path = path.parent / fit["first_leg_mean_fit"]
    if sha256(mean_path) != fit["first_leg_mean_fit_sha256"]:
        raise ValueError("Rider time calibration checksum mismatch")
    mean_fit = json.loads(mean_path.read_text())
    case = fit["cases"][str(hubs)]
    deadline_path = path.parent / case["file"]
    if sha256(deadline_path) != case["sha256"]:
        raise ValueError("Deadline table checksum mismatch")
    table = pd.read_csv(deadline_path)
    if len(table) != data["n_od"] or not np.array_equal(
        table.od, np.arange(data["n_od"])
    ):
        raise ValueError("Deadline table and instance OD ordering differ")
    for filename, key in [
        ("od_demand.csv", "od_demand_sha256"),
        ("od_nodes.csv", "od_nodes_sha256"),
    ]:
        if data["provenance"]["sha256"][filename] != case[key]:
            raise ValueError(f"Calibration refers to different {filename}")
    if (
        np.any(table.deadline_minutes <= 0)
        or not np.isfinite(table.deadline_minutes).all()
    ):
        raise ValueError("Delivery deadlines must be positive and finite")
    stats_path = path.parent / fit["pre_dispatch_statistics"]
    if sha256(stats_path) != fit["pre_dispatch_statistics_sha256"]:
        raise ValueError("Pre-dispatch statistics checksum mismatch")
    stats = json.loads(stats_path.read_text())["cases"][str(hubs)]
    return dict(
        manifest=fit,
        mean_fit=mean_fit,
        case=case,
        table=table,
        pre_dispatch=stats,
        path=path,
        mean_path=mean_path,
        deadline_path=deadline_path,
        stats_path=stats_path,
    )
