"""Observed orders and rider timing for reproducible performance instances."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import nnls

from .calibration import clock_minutes
from .instance_io import ROOT, project_path, sha256


RAW_PATHS = {
    "nanshan": ROOT / "Meituan-Shenzhen/深圳市南山区运单抽样_已脱敏.csv",
    "tsl": ROOT / "Meituan-TSL/all_waybill_info_meituan_0322.csv",
}


def read_orders(source, *, path=None):
    """Return accepted demand rows; missing timing never removes an order.

    Coordinates use one median-pickup projection for the entire source. TSL
    dates follow its complete ``dt`` partitions and hours follow event clocks.
    """
    if source not in RAW_PATHS:
        raise ValueError(f"Unknown source: {source}")
    path = project_path(path) if path is not None else RAW_PATHS[source]
    raw = pd.read_csv(path, dtype=str)
    raw_rows = len(raw)
    if source == "tsl":
        raw = raw.loc[pd.to_numeric(raw.is_courier_grabbed).eq(1)].copy()
        numeric = raw[
            [
                "platform_order_time",
                "grab_time",
                "fetch_time",
                "arrive_time",
            ]
        ].apply(pd.to_numeric, errors="coerce")
        pay, grab, fetch, arrive = (numeric[name] for name in numeric)
        local = pd.to_datetime(pay, unit="s", utc=True).dt.tz_convert("Asia/Shanghai")
        hour = local.dt.hour
        day = raw.dt
        valid = raw.is_prebook.eq("0") & np.isfinite(numeric).all(axis=1)
        valid &= (
            numeric.gt(0).all(axis=1) & pay.le(grab) & grab.le(fetch) & fetch.lt(arrive)
        )
        pre, post = (fetch - pay) / 60, (arrive - fetch) / 60
        dispatch = ((grab - pay) / 60).where(valid)
        names = ["sender_lng", "sender_lat", "recipient_lng", "recipient_lat"]
        coordinates = raw[names].apply(pd.to_numeric, errors="coerce") / 1_000_000
        order_id = raw.order_id
        date_mismatches = int(day.ne(local.dt.strftime("%Y%m%d")).sum())
        timing_rule = "accepted non-prebook; positive platform <= grab <= fetch < arrive timestamps"
    else:
        pay, fetch, arrive = (
            clock_minutes(raw[name])
            for name in ("customer_pay_time", "fetch_time", "arrived_time")
        )
        hour, day = np.floor(pay / 60), raw.dt_seq_id
        valid = pay.lt(fetch) & fetch.lt(arrive)
        pre, post = fetch - pay, arrive - fetch
        dispatch = pd.Series(np.nan, index=raw.index)
        names = [
            "pickup_longitude",
            "pickup_latitude",
            "dropoff_longitude",
            "dropoff_latitude",
        ]
        coordinates = raw[names].apply(pd.to_numeric, errors="coerce")
        order_id = raw.order_seq_id
        date_mismatches = 0
        timing_rule = "strict same-day payment < fetch < arrived; invalid and placeholder clocks excluded only from timing"
    if (
        not len(raw)
        or not np.isfinite(coordinates).all().all()
        or not hour.between(0, 23).all()
    ):
        raise ValueError("Demand requires finite coordinates and valid order hours")
    lon, lat = coordinates.iloc[:, :2].median().to_numpy()
    ox, oy, dx, dy = coordinates.to_numpy().T
    scale = 111.32 * np.cos(np.deg2rad(lat))
    orders = pd.DataFrame(
        dict(
            order_id=order_id.astype(str),
            day=day.astype(str),
            hour=hour.astype(int),
            ox=(ox - lon) * scale,
            oy=(oy - lat) * 111.32,
            dx=(dx - lon) * scale,
            dy=(dy - lat) * 111.32,
            pre_minutes=pre.where(valid),
            post_minutes=post.where(valid),
            dispatch_minutes=dispatch,
        )
    ).reset_index(drop=True)
    orders.attrs["source"] = source
    metadata = dict(
        source=source,
        raw_path=str(path.resolve()),
        raw_file_sha256=sha256(path),
        raw_rows=raw_rows,
        accepted_orders=len(orders),
        excluded_unaccepted_rows=raw_rows - len(orders),
        duplicate_day_order_keys=int(orders.duplicated(["day", "order_id"]).sum()),
        coordinate_system="local_km",
        source_coordinate_crs="unspecified",
        coordinate_disclosure=(
            "Nanshan source documentation describes grid-aggregated coordinates; datum is not specified"
            if source == "nanshan"
            else "source latitude/longitude datum is not established by this adapter"
        ),
        projection_center_lon=float(lon),
        projection_center_lat=float(lat),
        projection_rule="median accepted pickup coordinates over all source days",
        selected_days=sorted(orders.day.unique().tolist()),
        date_basis="source_partition_dt" if source == "tsl" else "dt_seq_id",
        hour_basis="Asia/Shanghai event timestamp"
        if source == "tsl"
        else "customer_pay_time",
        timestamp_date_mismatch_orders_all_days=date_mismatches,
        timing_rule=timing_rule,
        valid_timing_rows=int(valid.sum()),
        valid_dispatch_rows=int(dispatch.notna().sum()),
        pre_definition="order creation/payment to fetch; includes waiting and preparation",
        post_definition="fetch to arrival; includes travel and unobserved handoff/stops",
        dispatch_definition="order creation to rider acceptance; unavailable in Nanshan",
        demand_rule="all accepted source orders retained regardless of timing validity or prebooking",
    )
    return orders, metadata


def fit_rider(orders):
    """Fit a nonnegative post-fetch mean and observed acceptance delay.

    Only the fitted distance slope transfers to first/last rider travel. The
    fitted intercept is retained as the prior intercept for OD deadlines.
    """
    distance = np.hypot(orders.ox - orders.dx, orders.oy - orders.dy)
    valid = (
        np.isfinite(distance)
        & np.isfinite(orders.post_minutes)
        & orders.post_minutes.gt(0)
    )
    x, y = distance.loc[valid].to_numpy(), orders.post_minutes.loc[valid].to_numpy()
    if len(y) < 2 or np.ptp(x) <= 1e-12:
        raise ValueError(
            "Rider slope requires valid delivery times at distinct distances"
        )
    (intercept, slope), _ = nnls(np.column_stack([np.ones(len(x)), x]), y)
    if slope <= 0:
        raise ValueError("Estimated rider distance slope must be positive")
    dispatch = orders.dispatch_minutes.dropna()
    dispatch = dispatch[np.isfinite(dispatch) & dispatch.ge(0)]
    return dict(
        schema=1,
        family="exponential",
        slope_minutes_per_straight_km=float(slope),
        fitted_postfetch_intercept_minutes=float(intercept),
        mean_function=dict(
            form="slope_times_straight_distance",
            intercept_minutes=0.0,
            slope_minutes_per_straight_km=float(slope),
            input_distance_unit="km",
            output_unit="minutes",
        ),
        mean_source=f"nonnegative least-squares distance slope of observed {orders.attrs.get('source', 'source')} post-fetch elapsed time",
        transfer_assumption="distance-associated post-fetch component transfers to merchant-locker and locker-customer rider travel; the fitted intercept is used only for the OD deadline prior",
        family_status="working exponential travel-time assumption; mean fitted from post-fetch proxy observations",
        selectedcounts=dict(
            rows=len(orders),
            valid_timing_rows=len(y),
            rejected_timing_rows=len(orders) - len(y),
            zero_distance_rows=int((x == 0).sum()),
        ),
        fit=dict(
            method="nonnegative least squares with intercept",
            rmse_minutes=float(np.sqrt(np.mean((y - intercept - slope * x) ** 2))),
        ),
        first_leg_observed=False,
        shape_parameter_estimated=False,
        zero_distance="T=0 almost surely by continuity",
        expected_max="a + mean*exp(-a/mean), with deterministic a when mean=0",
        dispatch=dict(
            valid_rows=len(dispatch),
            mean_minutes=float(dispatch.mean()) if len(dispatch) else None,
            median_minutes=float(dispatch.median()) if len(dispatch) else None,
            q90_minutes=float(dispatch.quantile(0.9)) if len(dispatch) else None,
            definition="order creation to rider acceptance, minutes",
            observed_event="acceptance, not arrival or meal pickup",
        ),
    )
